import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, mod, positionWorld, select, sin, smoothstep, step, texture, uniform, uv, uvec2, vec2 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { coverage, noise, rgb, sdBox, sdCircle } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { CALIBRATION_FREQUENCY } from '@sim/envs/calibration';
import { wordFloat } from '@sim/words';
import { parameter } from '../../data/envs';
import { formatPercent, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Shape = (p: N<'vec2'>) => N<'float'>;

const SWEEP = 256;
const RING = 512;
const TURN = 2 * Math.PI / SWEEP;
const ZERO = 0.8;
const FULL = 3.5;
const PAPER = 3.64;
const LED_IN = 3.76;
const LED_OUT = 4.0;
const LED_MID = (LED_IN + LED_OUT) / 2;
const BEZEL = 4.2;
const CASE = 4.62;
const WIDTH = 8.7;
const TIP = 3.72;
const LAMP_U = -3.98;
const LAMP_V = 3.98;
const KEY_V = 0.35;
const KNOB_U = -8.0;
const CONTACT_U = -7.3;
const SCREW_U = -6.62;
const TRUNNION_U = -6.05;
const POST_U = -5.5;
const INK = 0.017;
const LABEL_PX = 256;
const LABEL_U = 0.78;
const LABEL_V0 = ZERO - 0.34;
const LABEL_V1 = FULL + 0.06;
const LABEL_CAP = 0.15;
const FONT = 'system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const LAMP_GAIN = 16;
const CONTACT_GAIN = 10;
const PRESS_SECONDS = 0.045;
const GLOW_SECONDS = 0.35;
const PACE_SECONDS = 0.2;
const FLASH_PACE = 12;
const REPAIR_SECONDS = 0.25;
const SHADE = new THREE.Vector3(0.07, 0, 0.09);

const PAPER_TINT = 0xefe7d2;
const GRID = 0xa7bfb9;
const GRID_MAJOR = 0x7c9993;
const INK_TINT = 0x3b2789;
const TARGET = 0x1b9fbb;
const TARGET_BAND = 0x9fdfe9;
const TARGET_GLOW = 0x4fd6ee;
const NUMERAL = 0x5d6a6a;
const BAKELITE = 0x17191f;
const STEEL = 0x9aa3ae;
const OUTLINE = 0x1a1d23;
const BRASS = 0xc9a45c;
const WALNUT = 0x4a2e1b;
const LED_OFF = 0x2a2317;
const GOLD = 0xffd166;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const disc = (radius: number) => new THREE.CircleGeometry(radius, 96).rotateX(-Math.PI / 2);
const radiusOf = (value: number) => ZERO + value * (FULL - ZERO);
const chartPoint = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const local = (width: number, depth: number, shift = 0): N<'vec2'> => uv().sub(0.5).mul(vec2(width, depth)).add(vec2(shift, 0));
const texel = (step: N<'float'>) => uvec2(vec2(mod(step.add(RING), RING), 0));
const soft = (d: N<'float'>, blur: N<'float'> | number): N<'float'> => smoothstep(blur, float(0).sub(blur).mul(0.3), d);

function polar(slot: N<'float'>, value: N<'float'>): N<'vec2'> {
  const angle = float(Math.PI / 2).sub(slot.mul(TURN));
  return vec2(cos(angle), sin(angle)).mul(value.mul(FULL - ZERO).add(ZERO));
}

function segment(p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>): N<'float'> {
  const pa = p.sub(a), ba = b.sub(a), h = clamp(dot(pa, ba).div(max(dot(ba, ba), 1e-9)), 0, 1);
  return length(pa.sub(ba.mul(h)));
}

function cable(p: N<'vec2'>, from: THREE.Vector2, to: THREE.Vector2, radius: number): N<'float'> {
  const curve = new THREE.CubicBezierCurve(from, new THREE.Vector2(from.x + 0.45, from.y), new THREE.Vector2(to.x - 0.35, to.y), to);
  const points = curve.getPoints(12);
  let d: N<'float'> = float(1e3);
  for (let k = 1; k < points.length; k++) d = min(d, segment(p, vec2(points[k - 1]!.x, points[k - 1]!.y), vec2(points[k]!.x, points[k]!.y)));
  return d.sub(radius);
}

function sprite(geometry: THREE.BufferGeometry, order: number, color: N<'vec3'>, opacity: N<'float'>): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

function shadow(geometry: THREE.BufferGeometry, shape: N<'float'>, strength = 0.34): THREE.Mesh {
  return sprite(geometry, 3, rgb(0x000000), soft(shape, 0.08).mul(strength));
}

function labelTexture(target: number): THREE.DataTexture {
  const width = Math.round(2 * LABEL_U * LABEL_PX), height = Math.round((LABEL_V1 - LABEL_V0) * LABEL_PX);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Calibration labels need a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textBaseline = 'alphabetic';
  const put = (text: string, u: number, value: number, align: CanvasTextAlign, weight: number) => {
    context.font = `${weight} 100px ${FONT}`;
    const probe = context.measureText('0');
    context.font = `${weight} ${100 * LABEL_CAP * LABEL_PX / probe.actualBoundingBoxAscent}px ${FONT}`;
    context.textAlign = align;
    context.fillText(text, (u + LABEL_U) * LABEL_PX, (LABEL_V1 - radiusOf(value) + 0.05 + LABEL_CAP) * LABEL_PX);
  };
  for (let k = 0; k <= 5; k++) put(k === 5 ? '100%' : String(k * 20), -0.07, k / 5, 'right', 600);
  put(String(Math.round(target * 100)), 0.07, target, 'left', 700);
  const pixels = context.getImageData(0, 0, width, height).data, alpha = new Uint8Array(width * height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = pixels[i * 4 + 3]!;
  const labels = new THREE.DataTexture(alpha, width, height, THREE.RedFormat, THREE.UnsignedByteType);
  labels.minFilter = THREE.LinearMipmapLinearFilter;
  labels.magFilter = THREE.LinearFilter;
  labels.generateMipmaps = true;
  labels.needsUpdate = true;
  return labels;
}

const ARM: Shape = (p) => {
  const bar = sdBox(p.sub(vec2(1.66, 0)), vec2(1.76, 0.036), 0.036);
  const needle = segment(p, vec2(3.3, 0), vec2(TIP - 0.014, 0)).sub(0.014);
  const weight = sdBox(p.sub(vec2(-0.4, 0)), vec2(0.2, 0.085), 0.05);
  return min(min(bar, needle), weight);
};
const CARRIAGE: Shape = (p) => sdBox(p, vec2(0.25, 0.17), 0.06);
const HUB: Shape = (p) => sdCircle(p, 0.3);

class CalibrationScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'calibration';

  private target = 0;
  private readonly texels = new Float32Array(RING * 4);
  private readonly history = new THREE.DataTexture(this.texels, RING, 1, THREE.RGBAFormat, THREE.FloatType);
  private labels: THREE.DataTexture | null = null;
  private readonly head = uniform(0);
  private readonly base = uniform(0);
  private readonly alpha = uniform(0);
  private readonly pen = uniform(0);
  private readonly press = uniform(0);
  private readonly lampLight = uniform(new THREE.Vector3());
  private readonly contactLight = uniform(new THREE.Vector3());
  private readonly accent = new THREE.Color();
  private readonly sweep = new THREE.Group();
  private readonly shade = new THREE.Group();
  private readonly carriage = new THREE.Group();
  private readonly carriageShade = new THREE.Group();
  private through = -Infinity;
  private wall = 0;
  private repairAt = 0;
  private held = 0;
  private pace = 0;
  private last = 0;
  private readonly ledger = { frequency: NaN, key: -1, duty: NaN, reward: 0 };

  override legend: LegendSpec | null = null;

  private slot(step: number): number {
    return (step % RING + RING) % RING * 4;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.target = parameter(ctx.environment, 'target');
    const half = CASE + 0.3;
    return new THREE.Box2(new THREE.Vector2(-WIDTH, -half), new THREE.Vector2(WIDTH, half));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xd2cdc2).multiplyScalar(1.1);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const at = new THREE.Vector3();
    const spot = (slot: number, radius: number) => {
      const angle = Math.PI / 2 - slot * TURN;
      return at.set(radius * Math.cos(angle), 0.05, -radius * Math.sin(angle));
    };
    return {
      debug: (frame, kit) => {
        const i = frame.state.step, base = i % SWEEP, goal = radiusOf(this.target);
        const ring = kit.curve('target', 129, TOKENS.cueMagenta);
        for (let k = 0; k <= 128; k++) {
          spot(k * SWEEP / 128, goal);
          ring.point(at.x, at.y, at.z);
        }
        const ink = kit.curve('ink', SWEEP, TOKENS.uiCyan);
        for (let s = Math.max(0, i - SWEEP + 1); s <= i; s++) {
          const found = frame.frameAt(s);
          if (found.step !== s) continue;
          spot(base - (i - s), radiusOf(wordFloat(found.words[0]!)));
          ink.point(at.x, at.y, at.z);
        }
        spot(base, radiusOf(wordFloat(frame.frameAt(i).words[0]!)));
        kit.marks('pen', 1).set(at.x, at.y, at.z, 0.07, 0xffffff);
        const keys = kit.marks('keys', SWEEP), press = kit.marks('press', 1);
        for (let d = 0; d < SWEEP && i + 1 - d >= 1; d++) {
          const found = frame.frameAt(i + 1 - d);
          if (found.step !== i + 1 - d || (Math.trunc(found.actions[0]!) & 1) === 0) continue;
          spot(base - d + 0.5, LED_MID);
          keys.set(at.x, at.y, at.z, 0.035, TOKENS.cueMagenta);
          if (d === 0) press.set(CONTACT_U, 0.05, -KEY_V, 0.12, TOKENS.cueMagenta);
        }
        const arm = kit.curve('arm', 2, 0xffffff);
        arm.point(0, 0.05, 0);
        spot(base, TIP);
        arm.point(at.x, at.y, at.z);
      },
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    this.accent.set(this.accentOf(ctx));
    const accent = this.accent.getHex();
    this.legend = {
      title: 'CALIBRATION',
      rows: [
        { key: 'frequency', label: 'frequency', tint: accent, value: () => formatPercent(this.ledger.frequency) },
        { key: 'target', label: 'target', tint: TARGET_GLOW, value: () => formatPercent(this.target) },
        { key: 'key', label: 'key', tint: accent, value: () => (this.ledger.key < 0 ? '—' : this.ledger.key ? 'pressed' : 'released') },
        { key: 'duty', label: `presses, last ${SWEEP}`, tint: accent, value: () => formatPercent(this.ledger.duty) },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(this.ledger.reward) },
      ],
      note: () => `one sweep · ${SWEEP} steps`,
    };
    this.scene.background = new THREE.Color(0x05070a);
    this.history.magFilter = this.history.minFilter = THREE.NearestFilter;
    this.history.needsUpdate = true;
    this.labels = labelTexture(this.target);
    const lit = light.lit;

    this.scene.add(this.bench(lit), this.housing(lit), this.paper(lit), this.lampMesh(lit), this.keyMesh(lit));
    const lamp = light.emitter(disc(0.24), this.lampLight), contact = light.emitter(disc(0.1), this.contactLight);
    lamp.position.set(LAMP_U, 0, -LAMP_V);
    contact.position.set(CONTACT_U, 0, -KEY_V);
    this.scene.add(lamp, contact);

    const armGeometry = flat(TIP + 1.0, 0.44).translate((TIP - 0.8) / 2, 0, 0), armP = local(TIP + 1.0, 0.44, (TIP - 0.8) / 2);
    const carriageGeometry = flat(0.8, 0.62), carriageP = local(0.8, 0.62), hubGeometry = flat(0.9, 0.9), hubP = local(0.9, 0.9);
    this.carriageShade.add(shadow(carriageGeometry, CARRIAGE(carriageP)));
    this.shade.add(shadow(armGeometry, ARM(armP)), shadow(hubGeometry, HUB(hubP)), this.carriageShade);
    this.shade.position.copy(SHADE);
    this.carriage.add(this.carriageMesh(lit, carriageGeometry, carriageP), light.occluder(flat(0.5, 0.34)));
    this.sweep.add(this.armMesh(lit, armGeometry, armP), light.occluder(flat(TIP + 0.5, 0.07).translate((TIP - 0.5) / 2, 0, 0)), this.carriage, this.hubMesh(lit, hubGeometry, hubP), light.occluder(disc(0.3)));
    this.scene.add(this.shade, this.sweep);
  }

  private bench(lit: Lit): THREE.Mesh {
    const q = chartPoint(), caseEdge = sdBox(q, vec2(CASE), 0.62);
    const baseEdge = sdBox(q.sub(vec2(-6.85, KEY_V)), vec2(1.6, 0.66), 0.22);
    const felt = rgb(0x11141b).mul(noise(q.mul(2.3)).mul(0.1).add(1));
    const occlusion = min(smoothstep(-0.05, 1.1, caseEdge.sub(0.12)), smoothstep(-0.05, 0.7, baseEdge.sub(0.1))).mul(0.55).add(0.45);
    return sprite(flat(80, 60), 0, lit(felt.mul(occlusion)), float(1));
  }

  private housing(lit: Lit): THREE.Mesh {
    const q = chartPoint(), r = length(q), caseEdge = sdBox(q, vec2(CASE), 0.62);
    const grain = noise(q.mul(3.1)).mul(0.07).add(1);
    const rim = coverage(abs(caseEdge.add(0.06)).sub(0.035)).mul(smoothstep(-1, 1, q.y.sub(q.x)).mul(0.7).add(0.3));
    let color: N<'vec3'> = mix(rgb(BAKELITE).mul(grain), rgb(0x4a4e58), rim);
    const well = coverage(r.sub(BEZEL)), bevel = coverage(abs(r.sub(BEZEL - 0.05)).sub(0.035));
    const facing = smoothstep(-1, 1, dot(q.div(max(r, 1e-4)), vec2(-0.7, 0.7)));
    color = mix(color, rgb(0x20232a), well);
    color = mix(color, mix(rgb(0x3a3e47), rgb(0x9197a2), facing), bevel);
    color = mix(color, rgb(0x0a0b0e), coverage(abs(r.sub(LED_MID)).sub((LED_OUT - LED_IN) / 2 + 0.035)));
    let screws: N<'float'> = float(0), slots: N<'float'> = float(0);
    for (const [u, v] of [[1, 1], [1, -1], [-1, -1]] as const) {
      const at = q.sub(vec2(u * (CASE - 0.42), v * (CASE - 0.42)));
      screws = max(screws, coverage(sdCircle(at, 0.13)));
      slots = max(slots, coverage(sdBox(at, vec2(0.1, 0.018), 0.01)));
    }
    color = mix(color, mix(rgb(0x7a808b), rgb(0x1a1c21), slots), screws);
    const s = fract(float(1.25).sub(atan(q.y, q.x).div(2 * Math.PI))).mul(SWEEP);
    const d = mod(this.base.sub(floor(s)).add(SWEEP), SWEEP);
    const decision = texture(this.history).load(texel(this.head.add(1).sub(d)));
    const on = decision.y.mul(decision.w);
    const cell = coverage(sdBox(vec2(fract(s).sub(0.5).mul(TURN).mul(r), r.sub(LED_MID)), vec2(0.33 * TURN * LED_MID, (LED_OUT - LED_IN) / 2), 0.012));
    const fade = float(1).sub(smoothstep(0.6 * SWEEP, SWEEP, d).mul(0.7));
    const bright = exp(d.negate().div(4)).mul(1.8).add(1.3).mul(fade);
    const led = mix(lit(rgb(LED_OFF)), rgb(this.accent).mul(bright), on);
    return sprite(flat(2 * CASE + 0.2, 2 * CASE + 0.2), 1, mix(lit(color), led, cell), coverage(caseEdge));
  }

  private paper(lit: Lit): THREE.Mesh {
    const q = chartPoint(), r = length(q), angle = atan(q.y, q.x);
    const s = fract(float(1.25).sub(angle.div(2 * Math.PI))).mul(SWEEP);
    const band = step(ZERO - 0.012, r).mul(step(r, FULL + 0.012));
    const tenth = (FULL - ZERO) / 10, t = r.sub(ZERO).div(tenth);
    const minor = coverage(abs(fract(t.add(0.5)).sub(0.5)).mul(tenth).sub(0.006)).mul(band);
    const major = coverage(min(min(abs(r.sub(ZERO)), abs(r.sub(radiusOf(0.5)))), abs(r.sub(FULL))).sub(0.012));
    const spoke = (every: number, half: number) => coverage(abs(fract(s.div(every).add(0.5)).sub(0.5)).mul(every * TURN).mul(r).sub(half)).mul(band);
    const goal = abs(r.sub(radiusOf(this.target)));
    let color: N<'vec3'> = rgb(PAPER_TINT).mul(noise(q.mul(7)).mul(0.035).add(1));
    color = mix(color, rgb(GRID), max(minor, spoke(16, 0.005)).mul(0.8));
    color = mix(color, rgb(GRID_MAJOR), max(major, spoke(64, 0.009)));
    color = mix(color, rgb(TARGET_BAND), coverage(goal.sub(0.05)));
    color = mix(color, rgb(TARGET), coverage(goal.sub(0.014)));
    const labelUv = vec2(q.x.add(LABEL_U).div(2 * LABEL_U), float(LABEL_V1).sub(q.y).div(LABEL_V1 - LABEL_V0));
    const label = texture(this.labels!, labelUv).r.mul(step(abs(q.x), LABEL_U)).mul(step(LABEL_V0, q.y)).mul(step(q.y, LABEL_V1));
    color = mix(color, select(q.x.lessThan(0), rgb(NUMERAL), rgb(TARGET)), label);

    const phase = this.base.add(this.alpha);
    const age = fract(phase.sub(s).div(SWEEP).add(1)).mul(SWEEP);
    const j0 = floor(this.alpha.sub(age));
    const ring = texture(this.history);
    const samples = [-1, 0, 1, 2].map((offset) => ring.load(texel(this.head.add(j0).add(offset))));
    const tip = polar(phase, this.pen);
    let ink: N<'float'> = float(0);
    for (let k = 0; k < 3; k++) {
      const j = j0.add(k - 1), a = samples[k]!, b = samples[k + 1]!, partial = j.greaterThan(-0.5);
      const start = polar(this.base.add(j), a.x), end = select(partial, tip, polar(this.base.add(j).add(1), b.x));
      const live = a.z.mul(select(partial, float(1), b.z)).mul(step(j, float(0.5))).mul(step(this.alpha.sub(SWEEP), j));
      ink = max(ink, coverage(segment(q, start, end).sub(INK)).mul(live));
    }
    color = mix(color, rgb(INK_TINT), ink.mul(float(1).sub(smoothstep(0.7 * SWEEP, SWEEP, age))));
    const rimShade = smoothstep(PAPER - 0.16, PAPER, r).mul(0.3).oneMinus();
    return sprite(disc(PAPER + 0.02), 2, lit(color.mul(rimShade)), coverage(r.sub(PAPER)));
  }

  private lampMesh(lit: Lit): THREE.Mesh {
    const p = local(1, 1), r = length(p);
    const chrome = mix(rgb(0x4d525c), rgb(0xc3c9d2), smoothstep(-0.4, 0.4, p.y.sub(p.x).mul(0.7)));
    const ring = coverage(abs(r.sub(0.36)).sub(0.012));
    const facet = abs(fract(atan(p.y, p.x).div(2 * Math.PI).mul(8)).sub(0.5)).mul(0.25).add(0.9);
    const lens = coverage(r.sub(0.29)), off = rgb(0x3a2710).mul(facet);
    const on = rgb(this.accent).mul(float(1).sub(r.div(0.29)).mul(0.9).add(0.75)).mul(this.press);
    const color = mix(mix(lit(chrome), lit(rgb(0x2a2d33)), ring), lit(off).add(on), lens);
    const mesh = sprite(flat(1, 1), 3, color, coverage(r.sub(0.41)));
    mesh.position.set(LAMP_U, 0, -LAMP_V);
    return mesh;
  }

  private keyMesh(lit: Lit): THREE.Mesh {
    const q = chartPoint(), press = this.press, lift = press.oneMinus();
    const axis = (u: number) => vec2(u, KEY_V);
    const baseShape = sdBox(q.sub(vec2(-6.85, KEY_V)), vec2(1.6, 0.66), 0.22);
    const grain = noise(vec2(q.x.mul(0.8), q.y.mul(11))).mul(0.2).add(0.9);
    const bevel = coverage(abs(baseShape.add(0.06)).sub(0.04)).mul(smoothstep(-0.8, 0.8, q.y.sub(KEY_V).sub(q.x.add(6.85).mul(0.3))));
    let color: N<'vec3'> = mix(lit(rgb(WALNUT).mul(grain)), lit(rgb(0x8a5f3b)), bevel.mul(0.8));
    let alpha: N<'float'> = max(coverage(baseShape), soft(baseShape.sub(0.02), 0.3).mul(0.45));
    color = mix(rgb(0x000000), color, coverage(baseShape));
    const layer = (shape: N<'float'>, paint: N<'vec3'>) => {
      const cover = coverage(shape);
      color = mix(color, paint, cover);
      alpha = max(alpha, cover);
    };
    const shade = (shape: N<'float'>, blur: N<'float'> | number, strength: N<'float'> | number) => {
      const dark = soft(shape, blur).mul(strength);
      color = mix(color, rgb(0x000000), dark);
      alpha = max(alpha, dark);
    };
    const wires = [
      [new THREE.Vector2(POST_U, KEY_V + 0.36), new THREE.Vector2(-CASE + 0.05, KEY_V + 1.05), 0x8e2f25],
      [new THREE.Vector2(POST_U, KEY_V - 0.36), new THREE.Vector2(-CASE + 0.05, KEY_V - 1.05), 0x202228],
    ] as const;
    for (const [from, to, tint] of wires) {
      const wire = cable(q, from, to, 0.038);
      shade(wire.sub(0.01), 0.06, 0.4);
      layer(wire, lit(rgb(tint).mul(smoothstep(-0.04, 0.04, wire.add(0.02)).mul(-0.35).add(1.1))));
      layer(sdCircle(q.sub(vec2(to.x - 0.02, to.y)), 0.1), lit(rgb(0x0d0e11)));
    }
    for (const side of [1, -1]) {
      const post = q.sub(vec2(POST_U, KEY_V + side * 0.36));
      shade(sdCircle(post.sub(vec2(0.04, -0.05)), 0.13), 0.06, 0.45);
      layer(sdCircle(post, 0.13), lit(rgb(BRASS).mul(smoothstep(-0.13, 0.13, post.y.sub(post.x)).mul(0.5).add(0.7))));
      layer(sdCircle(post, 0.05), lit(rgb(0x6b5429)));
    }
    const contact = q.sub(axis(CONTACT_U));
    layer(sdCircle(contact, 0.15), lit(rgb(0x8e959f)));
    const spark = exp(length(contact).negate().div(0.07)).mul(press).mul(3.5);
    color = color.add(rgb(this.accent).mul(spark));
    const lever = sdBox(q.sub(axis((KNOB_U + TRUNNION_U + 0.2) / 2)), vec2((TRUNNION_U + 0.2 - KNOB_U) / 2, 0.085), 0.06);
    shade(lever.sub(0.01), lift.mul(0.05).add(0.05), 0.4);
    const tilt = smoothstep(CONTACT_U + 0.6, KNOB_U, q.x).mul(press).mul(0.28).oneMinus();
    const ridge = smoothstep(-0.085, 0.085, q.y.sub(KEY_V)).mul(0.45).add(0.75);
    layer(lever, lit(rgb(this.accent).mul(0.82).mul(ridge).mul(tilt)));
    const trunnion = q.sub(axis(TRUNNION_U));
    shade(sdBox(trunnion.sub(vec2(0.04, -0.06)), vec2(0.17, 0.4), 0.05), 0.07, 0.4);
    layer(sdBox(trunnion, vec2(0.17, 0.4), 0.05), lit(rgb(0x7d848f).mul(smoothstep(-0.4, 0.4, trunnion.y).mul(0.4).add(0.75))));
    layer(sdBox(trunnion, vec2(0.03, 0.44), 0.02), lit(rgb(0xc8cdd4)));
    const screw = q.sub(axis(SCREW_U));
    const knurl = abs(fract(atan(screw.y, screw.x).div(2 * Math.PI).mul(18)).sub(0.5)).mul(0.5).add(0.7);
    layer(sdCircle(screw, 0.14), lit(rgb(0x9aa1ab).mul(knurl)));
    layer(sdCircle(screw, 0.05), lit(rgb(0x3c4048)));
    const knob = q.sub(axis(KNOB_U));
    shade(sdCircle(knob.sub(vec2(lift.mul(0.08).add(0.03), lift.mul(-0.1).sub(0.035))), 0.36), lift.mul(0.1).add(0.05), 0.55);
    const gloss = exp(length(knob.sub(vec2(-0.1, 0.12))).negate().div(0.09)).mul(lift.mul(0.25).add(0.3));
    layer(sdCircle(knob, 0.37), lit(rgb(0x2b2e34)));
    layer(sdCircle(knob, 0.33), lit(rgb(0x121317).mul(press.mul(-0.2).add(1))).add(rgb(0xb8c0cc).mul(gloss)));
    return sprite(flat(4.6, 3.4).translate(-6.6, 0, -KEY_V), 3, color, alpha);
  }

  private armMesh(lit: Lit, geometry: THREE.BufferGeometry, p: N<'vec2'>): THREE.Mesh {
    const shape = ARM(p), sheen = smoothstep(-0.04, 0.04, p.y).mul(0.5).add(0.7);
    const color = mix(lit(rgb(OUTLINE)), lit(rgb(STEEL).mul(sheen)), coverage(shape.add(0.012)));
    return sprite(geometry, 4, color, coverage(shape));
  }

  private carriageMesh(lit: Lit, geometry: THREE.BufferGeometry, p: N<'vec2'>): THREE.Mesh {
    const body = CARRIAGE(p), bevel = smoothstep(-0.07, 0, body);
    const slot = coverage(sdBox(p.sub(vec2(0.15, 0)), vec2(0.014, 0.1), 0.01));
    const well = coverage(sdCircle(p, 0.065)), nib = exp(length(p).negate().div(0.03));
    let color: N<'vec3'> = mix(rgb(this.accent), rgb(BRASS).mul(0.5), bevel);
    color = mix(color, rgb(0x5b4520), slot);
    color = mix(lit(color), lit(rgb(0x1c1530)), well);
    color = mix(lit(rgb(OUTLINE)), color, coverage(body.add(0.012)));
    return sprite(geometry, 5, color.add(rgb(0x8f6bff).mul(nib.mul(1.4))), coverage(body));
  }

  private hubMesh(lit: Lit, geometry: THREE.BufferGeometry, p: N<'vec2'>): THREE.Mesh {
    const r = length(p), knurl = abs(fract(atan(p.y, p.x).div(2 * Math.PI).mul(40)).sub(0.5)).mul(2);
    const cap = mix(rgb(BRASS), rgb(BRASS).mul(knurl.mul(0.5).add(0.45)), smoothstep(0.22, 0.28, r));
    const slot = coverage(sdBox(p, vec2(0.15, 0.022), 0.012));
    const color = mix(cap.mul(smoothstep(-0.3, 0.3, p.y.sub(p.x)).mul(0.35).add(0.8)), rgb(0x3a2a12), slot);
    return sprite(geometry, 6, mix(lit(rgb(OUTLINE)), lit(color), coverage(HUB(p).add(0.014))), coverage(HUB(p)));
  }

  private write(frame: FrameUpdate, at: number): boolean {
    const slot = this.slot(at), t = this.texels;
    const found = at < 0 ? null : frame.frameAt(at);
    if (!found || found.step !== at) {
      t[slot + 2] = t[slot + 3] = 0;
      return false;
    }
    t[slot] = wordFloat(found.words[0]!);
    t[slot + 1] = Math.trunc(found.actions[0]!) & 1;
    t[slot + 2] = 1;
    t[slot + 3] = at > 0 ? 1 : 0;
    return true;
  }

  private sync(frame: FrameUpdate): void {
    const newest = frame.state.step + 1, oldest = newest - SWEEP - 2;
    let dirty = false;
    if (!frame.prev || newest < this.through || newest - this.through > SWEEP + 2) {
      for (let s = oldest; s <= newest; s++) this.write(frame, s);
      dirty = true;
    } else if (newest > this.through) {
      for (let s = this.through + 1; s <= newest; s++) this.write(frame, s);
      dirty = true;
    }
    this.through = newest;
    if (this.wall - this.repairAt > REPAIR_SECONDS) {
      this.repairAt = this.wall;
      for (let s = Math.max(0, oldest); s <= newest; s++) if (this.texels[this.slot(s) + 2] === 0 && this.write(frame, s)) dirty = true;
    }
    if (dirty) this.history.needsUpdate = true;
  }

  private mean(from: number, to: number): number {
    let presses = 0, seen = 0;
    for (let s = from; s <= to; s++) {
      const slot = this.slot(s);
      if (this.texels[slot + 3] === 0) continue;
      presses += this.texels[slot + 1]!;
      seen++;
    }
    return seen > 0 ? presses / seen : NaN;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.wall += frame.dtReal;
    this.sync(frame);
    const i = frame.state.step, newest = i + 1, alpha = Math.min(1, Math.max(0, frame.position - i));
    const pen = frame.state.pose[CALIBRATION_FREQUENCY]!, angle = Math.PI / 2 - (i % SWEEP + alpha) * TURN;
    this.head.value = i % RING;
    this.base.value = i % SWEEP;
    this.alpha.value = alpha;
    this.pen.value = pen;
    this.sweep.rotation.y = this.shade.rotation.y = angle;
    this.carriage.position.x = this.carriageShade.position.x = radiusOf(pen);

    const pending = this.slot(newest), known = this.texels[pending + 3] !== 0, bit = known ? this.texels[pending + 1]! : 0;
    const duty = this.mean(newest - SWEEP + 1, newest), moved = frame.position - this.last;
    this.last = frame.position;
    if (moved < 0 || moved > 4 * SWEEP) this.pace = 0;
    else if (frame.dtReal > 0) this.pace += (moved / frame.dtReal - this.pace) * (1 - Math.exp(-frame.dtReal / PACE_SECONDS));
    const flashing = this.pace < FLASH_PACE, level = flashing ? bit : Number.isFinite(duty) ? duty : 0;
    this.held = level + (this.held - level) * Math.exp(-frame.dtReal / (flashing ? PRESS_SECONDS : GLOW_SECONDS));
    this.press.value = this.held;
    const lamp = this.held * LAMP_GAIN, spark = this.held * CONTACT_GAIN;
    this.lampLight.value.set(this.accent.r * lamp, this.accent.g * lamp, this.accent.b * lamp);
    this.contactLight.value.set(this.accent.r * spark, this.accent.g * spark, this.accent.b * spark);

    this.ledger.frequency = wordFloat(frame.frameAt(i).words[0]!);
    this.ledger.key = known ? bit : -1;
    this.ledger.duty = duty;
    this.ledger.reward = frame.state.reward;
  }

  protected override drop(): void {
    this.history.dispose();
    this.labels?.dispose();
    this.labels = null;
  }
}

export default (() => new CalibrationScene()) satisfies SceneFactory;
