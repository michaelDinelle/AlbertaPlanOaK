import * as THREE from 'three/webgpu';
import { abs, atan, exp, float, floor, fract, fwidth, instancedDynamicBufferAttribute, length, max, min, mix, mod, select, sin, smoothstep, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { coverage, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { fbm } from '@kit/textures';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import {
  ALCHEMY_CAULDRON, ALCHEMY_CONTAINERS, ALCHEMY_CURSOR, ALCHEMY_EDGES, ALCHEMY_EPISODE, ALCHEMY_POTIONS, ALCHEMY_SPENT_POTION, ALCHEMY_SPENT_STONE,
  ALCHEMY_STONES, ALCHEMY_TICK, ALCHEMY_TRIAL, ALCHEMY_VERTICES, ALCHEMY_WORDS, alchemyAction, alchemyEdge, alchemyFeature, alchemyPotion, alchemyStone,
} from '@sim/envs/alchemy';
import { constantList, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

type N<T extends string> = THREE.Node<T>;

const TAU = Math.PI * 2;
const RING = 3.3;
const TICKS_AT = 4.3;
const TRIALS_AT = 2.45;
const HOME = 1.45;
const QB = 1.3;
const QS = 1.9;
const QC = 2.2;
const QR = 2.4;
const TABLE = 40;
const MOVE = 0.24;
const FLASH_SECONDS = 0.6;
const LABEL_W = 256;
const LABEL_H = 128;
const LABEL_COLUMNS = 4;
const FONT = 'ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const TINTS = [0xe0485a, 0x2fc4b2, 0xf2a93b, 0x5a62e8, 0x9bd83a, 0xd154c8] as const;
const AXES = [0xff6b6b, 0x6bff9a, 0x6bb8ff] as const;
const LINKS = [0xfff1a8, 0xffb3e6, 0xb3fff0] as const;
const VIOLET = 0x8a4df0;
const BLUE = 0x4f7df2;
const AQUA = 0x3ccfd8;
const GOLD = 0xffd166;
const EMBER = 0xffb870;
const LOSS = 0x6f9bff;
const INK_HIGH = 0xffe3a0;
const INK_MID = 0xf3ead8;
const INK_LOW = 0x9db8e0;
const CUBE = 0.95;
const DEPTH = [0.42, 0.34] as const;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const easeBack = (t: number) => 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2;
const slotAngle = (k: number) => Math.PI / 2 - (k + 1) * TAU / ALCHEMY_CONTAINERS;
const homeAngle = (s: number) => Math.PI * (5 / 6 - (2 / 3) * s);
const slotRadius = (k: number) => k === ALCHEMY_CAULDRON ? 0.92 : 0.58;
const cubeX = (v: number) => ((v & 1) - 0.5) * CUBE + ((v >> 2 & 1) - 0.5) * DEPTH[0];
const cubeZ = (v: number) => -(((v >> 1 & 1) - 0.5) * CUBE + ((v >> 2 & 1) - 0.5) * DEPTH[1]);
const sign = (v: number) => v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0';

function labelAtlas(values: readonly number[]): THREE.DataTexture {
  const rows = Math.ceil(values.length / LABEL_COLUMNS), canvas = document.createElement('canvas');
  canvas.width = LABEL_COLUMNS * LABEL_W;
  canvas.height = rows * LABEL_H;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Alchemy labels need a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  values.forEach((value, i) => {
    const text = sign(value);
    context.font = `700 100px ${FONT}`;
    const probe = context.measureText(text);
    const size = 100 * Math.min(0.62 * LABEL_H / (probe.actualBoundingBoxAscent + probe.actualBoundingBoxDescent), 0.8 * LABEL_W / probe.width);
    context.font = `700 ${size}px ${FONT}`;
    const fit = context.measureText(text);
    context.fillText(text, (i % LABEL_COLUMNS + 0.5) * LABEL_W, (Math.floor(i / LABEL_COLUMNS) + 0.5) * LABEL_H + (fit.actualBoundingBoxAscent - fit.actualBoundingBoxDescent) / 2);
  });
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data, alpha = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = pixels[i * 4 + 3]!;
  const atlas = new THREE.DataTexture(alpha, canvas.width, canvas.height, THREE.RedFormat, THREE.UnsignedByteType);
  atlas.minFilter = THREE.LinearMipmapLinearFilter;
  atlas.magFilter = THREE.LinearFilter;
  atlas.generateMipmaps = true;
  atlas.needsUpdate = true;
  return atlas;
}

const dynamic = (count: number, size: number) => new THREE.InstancedBufferAttribute(new Float32Array(count * size), size).setUsage(THREE.DynamicDrawUsage);
const basic = (additive = false) => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending });

type Kind = 'rest' | 'walk' | 'dip' | 'cash' | 'idle' | 'deal';

class SymbolicAlchemyScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.3, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'symbolicAlchemy';

  private values: number[] = [];
  private top = 0;
  private trials = 10;
  private ticks = 20;
  private rotating = false;
  private potions: THREE.InstancedMesh | null = null;
  private stones: THREE.InstancedMesh | null = null;
  private potionTint = dynamic(ALCHEMY_POTIONS, 3);
  private potionState = dynamic(ALCHEMY_POTIONS, 4);
  private stoneLook = dynamic(ALCHEMY_STONES, 4);
  private stoneState = dynamic(ALCHEMY_STONES, 4);
  private stoneTone = dynamic(ALCHEMY_STONES, 4);
  private selector: THREE.Mesh | null = null;
  private cauldron: THREE.Mesh | null = null;
  private textures: THREE.Texture[] = [];
  private readonly clock = uniform(0);
  private readonly tickU = uniform(0);
  private readonly trialU = uniform(0);
  private readonly ringU = uniform(0.58);
  private readonly flash = uniform(0);
  private readonly flashTint = uniform(new THREE.Color(GOLD));

  private prev = new Uint32Array(ALCHEMY_WORDS);
  private cur = new Uint32Array(ALCHEMY_WORDS);
  private kind: Kind = 'rest';
  private mover = -1;
  private progress = 1;
  private since = Infinity;
  private step = -1;
  private pending: (() => void) | null = null;
  private readonly spots: { x: number; z: number }[] = [];
  private readonly game = { from: 0, to: -1, sum: 0, found: false };
  private readonly ledger = { reward: 0 };

  private readonly matrix = new THREE.Matrix4();
  private readonly at = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();
  private readonly color = new THREE.Color();

  override legend: LegendSpec = {
    title: 'Symbolic Alchemy',
    rows: [
      { key: 'trial', label: 'trial', tint: 0xf4c55f, value: () => `${this.cur[ALCHEMY_TRIAL]! + 1} / ${this.trials}` },
      { key: 'tick', label: 'tick', tint: 0xf0c36e, value: () => `${this.cur[ALCHEMY_TICK]!} / ${this.ticks}` },
      { key: 'stones', label: 'stones', tint: 0x8fb4ff, value: () => this.stoneValues() },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(this.ledger.reward) },
      { key: 'episode', label: 'episode', tint: 0xedc46a, value: () => this.game.found ? formatCount(this.game.sum) : '—' },
    ],
    note: () => `${this.rotating ? 'chemistry redrawn each episode' : 'fixed chemistry'} · cursor on ${this.cur[ALCHEMY_CURSOR] === ALCHEMY_CAULDRON ? 'cauldron' : `potion ${this.cur[ALCHEMY_CURSOR]! + 1}`}`,
    bar: () => this.cur[ALCHEMY_TICK]! / this.ticks,
  };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.values = constantList(env, 'ALCHEMY_VALUE');
    if (this.values.length !== ALCHEMY_VERTICES) throw new Error(`${env.envId} values ${this.values.length} vertices where the cube has ${ALCHEMY_VERTICES}`);
    this.top = Math.max(...this.values);
    this.trials = parameter(env, 'trials');
    this.ticks = parameter(env, 'ticks');
    this.rotating = parameter(env, 'randomChemistry') > 0;
    for (let s = 0; s < ALCHEMY_STONES; s++) this.spots[s] = { x: 0, z: 0 };
    return new THREE.Box2(new THREE.Vector2(-4.75, -4.75), new THREE.Vector2(4.75, 4.75));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xe8ecf4).multiplyScalar(0.8);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const now = frame.frameAt(frame.state.step);
    if (now.step !== frame.state.step) return;
    const w = now.words, y = 0.3;
    for (let axis = 0; axis < 3; axis++) {
      for (let lower = 0; lower < ALCHEMY_VERTICES; lower++) {
        if (lower >> axis & 1) continue;
        const upper = lower | 1 << axis, open = alchemyEdge(w, axis, lower);
        const line = kit.curve(open ? `open${axis}${lower}` : `shut${axis}${lower}`, 2, open ? AXES[axis]! : 0x4a5566);
        line.point(cubeX(lower), y, cubeZ(lower));
        line.point(cubeX(upper), y, cubeZ(upper));
      }
    }
    const marks = kit.marks('latent', ALCHEMY_STONES + 1), best = this.values.indexOf(this.top);
    marks.set(cubeX(best), y, cubeZ(best), 0.09, GOLD);
    for (let s = 0; s < ALCHEMY_STONES; s++) {
      const held = alchemyStone(w, s);
      if (held & ALCHEMY_SPENT_STONE) continue;
      const v = held & 7, x = cubeX(v) + (s - 1) * 0.07, z = cubeZ(v) + (s - 1) * 0.05, spot = this.spots[s]!;
      marks.set(x, y, z, 0.055, LINKS[s]!);
      const link = kit.curve(`link${s}`, 2, LINKS[s]!);
      link.point(spot.x, y, spot.z);
      link.point(x, y, z);
    }
    const reach = [[1, 0], [0, 1], [DEPTH[0], DEPTH[1]]] as const;
    for (let p = 0; p < ALCHEMY_POTIONS; p++) {
      const dose = alchemyPotion(w, p);
      if (dose & ALCHEMY_SPENT_POTION) continue;
      const type = dose & 7, axis = type >> 1, dir = type & 1 ? -1 : 1, [ux, uy] = reach[axis]!, n = Math.hypot(ux, uy);
      const dx = dir * ux / n * 0.2, dz = -dir * uy / n * 0.2, a = slotAngle(p), cx = (RING + 0.62) * Math.cos(a), cz = -(RING + 0.62) * Math.sin(a);
      const arrow = kit.curve(`dose${p}`, 5, 0xffffff);
      arrow.point(cx - dx, y, cz - dz);
      arrow.point(cx + dx, y, cz + dz);
      arrow.point(cx + dx * 0.45 - dz * 0.4, y, cz + dz * 0.45 + dx * 0.4);
      arrow.point(cx + dx, y, cz + dz);
      arrow.point(cx + dx * 0.45 + dz * 0.4, y, cz + dz * 0.45 - dx * 0.4);
    }
  }

  protected build(ctx: SceneContext): void {
    this.scene.background = new THREE.Color(0x07090c);
    this.buildTable(ctx);
    this.buildSelector();
    this.buildCauldron();
    this.buildPotions();
    this.buildStones();
  }

  private buildTable(ctx: SceneContext): void {
    const grain = fbm(ctx.seed, { size: 256, frequency: 5, octaves: 5 });
    this.textures.push(grain);
    const p = uv().sub(0.5).mul(TABLE), r = length(p);
    const turn = fract(float(Math.PI / 2).sub(atan(p.y, p.x)).div(TAU));
    const g = texture(grain, p.mul(0.05)).r, fine = texture(grain, p.mul(0.23)).r;
    let base: N<'vec3'> = mix(rgb(0x14181d), rgb(0x232a31), smoothstep(0.25, 0.8, g)).mul(fine.mul(0.18).add(0.9));
    base = base.mul(smoothstep(4.5, 13, r).oneMinus().mul(0.6).add(0.4));
    const lane = abs(r.sub(RING)).sub(0.56);
    base = mix(base, base.mul(0.7), coverage(lane)).add(vec3(exp(abs(lane).mul(-45)).mul(0.035)));
    for (let s = 0; s < ALCHEMY_STONES; s++) {
      const a = homeAngle(s), d = sdCircle(p.sub(vec2(HOME * Math.cos(a), HOME * Math.sin(a))), 0.5);
      base = mix(base, base.mul(0.55), coverage(d)).add(vec3(exp(abs(d).mul(-40)).mul(0.05)));
    }
    const a20 = fract(turn.mul(this.ticks)), arc = min(a20, a20.oneMinus()).mul(TAU * TICKS_AT / this.ticks);
    const tickD = max(abs(r.sub(TICKS_AT)).sub(0.07), float(0.04).sub(arc));
    const tickLit = floor(turn.mul(this.ticks)).lessThan(this.tickU);
    base = mix(base, select(tickLit, rgb(0xf0c36e), rgb(0x3b3833)), coverage(tickD));
    const a10 = fract(turn.mul(this.trials)).sub(0.5).mul(TAU * TRIALS_AT / this.trials);
    const dotD = length(vec2(a10, r.sub(TRIALS_AT))).sub(0.09), trial = floor(turn.mul(this.trials));
    const dotTint = select(trial.lessThan(this.trialU), rgb(0xcdbfa6), select(trial.equal(this.trialU), rgb(0xf4c55f), rgb(0x34322e)));
    base = mix(base, dotTint, coverage(dotD));
    base = mix(base, rgb(0x8a6d3e), coverage(abs(r.sub(TICKS_AT + 0.24)).sub(0.012)).mul(0.8));
    const table = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
    table.colorNode = base;
    const mesh = new THREE.Mesh(flat(TABLE, TABLE), table);
    mesh.renderOrder = 0;
    this.scene.add(mesh);
  }

  private buildSelector(): void {
    const k = uv().sub(0.5).mul(QR), off = abs(length(k).sub(this.ringU));
    const material = basic();
    material.colorNode = rgb(0xf5c768).mul(0.9);
    material.opacityNode = max(coverage(off.sub(0.03)).mul(0.9), exp(off.mul(-10)).mul(0.4));
    const mesh = new THREE.Mesh(flat(QR, QR), material);
    mesh.renderOrder = 1;
    this.selector = mesh;
    this.scene.add(mesh);
  }

  private buildCauldron(): void {
    const c = uv().sub(0.5).mul(QC);
    const pot = (q: N<'vec2'>) => {
      const bowl = max(sdCircle(q.sub(vec2(0, -0.08)), 0.66), q.y.sub(0.24));
      const rim = sdBox(q.sub(vec2(0, 0.26)), vec2(0.76, 0.085), 0.085);
      const lugs = min(abs(sdCircle(q.sub(vec2(-0.8, 0.02)), 0.1)).sub(0.03), abs(sdCircle(q.sub(vec2(0.8, 0.02)), 0.1)).sub(0.03));
      const legs = min(sdBox(q.sub(vec2(-0.38, -0.72)), vec2(0.08, 0.07), 0.03), sdBox(q.sub(vec2(0.38, -0.72)), vec2(0.08, 0.07), 0.03));
      return { body: min(min(bowl, lugs), legs), rim };
    };
    const { body, rim } = pot(c), shade = pot(c.sub(vec2(0.06, -0.09)));
    const shadow = smoothstep(-0.05, 0.16, min(shade.body, shade.rim)).oneMinus().mul(0.45);
    const iron = rgb(0x262b31).mul(smoothstep(-0.75, 0.3, c.y).mul(0.55).add(0.75)).add(vec3(exp(length(c.sub(vec2(-0.26, -0.02))).mul(-7)).mul(0.1)));
    const mouth = sdBox(c.sub(vec2(0, 0.27)), vec2(0.62, 0.045), 0.045);
    const swirl = noise(c.mul(vec2(9, 40)).add(vec2(this.clock.mul(0.6), this.clock.mul(1.5))));
    const brew = mix(rgb(0x1f6b4a), (this.flashTint as unknown as N<'vec3'>).mul(1.1), this.flash).mul(swirl.mul(0.35).add(0.8));
    const sy = c.y.sub(0.62), steam = exp(sy.mul(sy).mul(-14).sub(c.x.mul(c.x).mul(6))).mul(this.flash).mul(0.8);
    let col: N<'vec3'> = vec3(0), alpha: N<'float'> = shadow;
    col = mix(col, (this.flashTint as unknown as N<'vec3'>), steam);
    alpha = max(alpha, steam);
    const bodyC = coverage(body), rimC = coverage(rim), mouthC = coverage(mouth);
    col = mix(col, iron, bodyC);
    alpha = max(alpha, bodyC);
    col = mix(col, rgb(0x48515b).mul(c.y.sub(0.26).mul(2.4).add(1)), rimC);
    alpha = max(alpha, rimC);
    col = mix(col, brew, mouthC);
    const material = basic();
    material.colorNode = col;
    material.opacityNode = alpha;
    const mesh = new THREE.Mesh(flat(QC, QC), material);
    const a = slotAngle(ALCHEMY_CAULDRON);
    mesh.position.set(RING * Math.cos(a), 0, -RING * Math.sin(a));
    mesh.renderOrder = 2;
    this.cauldron = mesh;
    this.scene.add(mesh);
  }

  private buildPotions(): void {
    const tint = instancedDynamicBufferAttribute(this.potionTint, 'vec3') as N<'vec3'>;
    const state = instancedDynamicBufferAttribute(this.potionState, 'vec4') as N<'vec4'>;
    const b = uv().sub(0.5).mul(QB);
    const flask = (q: N<'vec2'>) => min(min(sdCircle(q.sub(vec2(0, -0.14)), 0.3), sdBox(q.sub(vec2(0, 0.19)), vec2(0.085, 0.15), 0.03)), sdBox(q.sub(vec2(0, 0.35)), vec2(0.13, 0.035), 0.03));
    const glass = flask(b), body = sdCircle(b.sub(vec2(0, -0.14)), 0.3);
    const shadow = smoothstep(-0.04, 0.12, flask(b.sub(vec2(0.04, -0.06)))).oneMinus().mul(0.4);
    const fill = state.x, chosen = state.z;
    const level = float(-0.44).add(fill.mul(0.56)).add(sin(b.x.mul(16).add(this.clock.mul(3)).add(state.w)).mul(0.012).mul(fill));
    const liquid = select(fill.greaterThan(0.002), coverage(max(body.add(0.045), b.y.sub(level))), float(0));
    const depth = smoothstep(-0.44, level, b.y).mul(0.45).add(0.65);
    const surface = exp(abs(b.y.sub(level)).mul(-60)).mul(0.25);
    const brew = tint.mul(depth.add(surface).add(chosen.mul(0.3)));
    const rimLine = coverage(abs(glass).sub(0.016)).mul(0.85);
    const shine = coverage(sdSegment(b, vec2(-0.2, -0.22), vec2(-0.15, -0.02), 0.026)).mul(0.35);
    const cork = coverage(sdBox(b.sub(vec2(0, 0.45)), vec2(0.07, 0.06), 0.02)).mul(state.y);
    const glassC = coverage(glass);
    let col: N<'vec3'> = vec3(0), alpha: N<'float'> = shadow;
    col = mix(col, rgb(0x4d6270), glassC);
    alpha = mix(alpha, float(0.34), glassC);
    col = mix(col, brew, liquid);
    alpha = mix(alpha, float(0.96), liquid);
    col = mix(col, mix(rgb(0xc9d8e2), rgb(0xffd37a), chosen), rimLine);
    alpha = max(alpha, rimLine);
    col = mix(col, vec3(1), shine);
    alpha = max(alpha, shine);
    col = mix(col, rgb(0x8a5a33).mul(b.y.sub(0.45).mul(3).add(1)), cork);
    alpha = max(alpha, cork);
    const material = basic();
    material.colorNode = col;
    material.opacityNode = alpha;
    const mesh = instancedMesh(flat(QB, QB), material, ALCHEMY_POTIONS);
    mesh.renderOrder = 3;
    mesh.frustumCulled = false;
    this.potions = mesh;
    this.scene.add(mesh);
  }

  private buildStones(): void {
    const look = instancedDynamicBufferAttribute(this.stoneLook, 'vec4') as N<'vec4'>;
    const state = instancedDynamicBufferAttribute(this.stoneState, 'vec4') as N<'vec4'>;
    const tone = instancedDynamicBufferAttribute(this.stoneTone, 'vec4') as N<'vec4'>;
    const atlas = labelAtlas(this.values);
    this.textures.push(atlas);
    const rows = Math.ceil(this.values.length / LABEL_COLUMNS);
    const s = uv().sub(0.5).mul(QS), q = s.sub(vec2(0, 0.12));
    const rad = look.y.mul(0.1).add(0.34), round = look.z.add(1).mul(0.5);
    const gem = (at: N<'vec2'>) => {
      const turned = vec2(at.x.add(at.y), at.y.sub(at.x)).mul(Math.SQRT1_2);
      return { d: mix(sdBox(turned, vec2(rad.mul(0.8), rad.mul(0.8)), rad.mul(0.1)), sdCircle(at, rad), round), turned };
    };
    const { d, turned } = gem(q), cast = gem(q.sub(vec2(0.05, -0.08))).d;
    const hue = select(look.x.lessThan(0), mix(rgb(VIOLET), rgb(BLUE), look.x.add(1)), mix(rgb(BLUE), rgb(AQUA), look.x));
    const px = fwidth(s.x);
    const facet = smoothstep(px.negate(), px, turned.x).mul(0.16).add(smoothstep(px.negate(), px, turned.y).mul(0.1)).sub(0.14).mul(round.oneMinus());
    const dome = q.y.div(rad).mul(0.2).mul(round);
    const glint = length(q.sub(vec2(rad.mul(-0.35), rad.mul(0.4)))).div(rad.mul(0.24));
    const spec = exp(glint.mul(glint).negate()).mul(0.5);
    const body = hue.mul(facet.add(dome).add(0.95).sub(smoothstep(-0.07, 0, d).mul(0.3))).add(vec3(spec));
    const face = coverage(d), shadow = smoothstep(-0.04, 0.14, cast).oneMinus().mul(0.45);
    const halo = exp(max(d, 0).mul(-8)).mul(tone.w).mul(face.oneMinus());
    const plateAt = s.sub(vec2(0, -0.56)), lp = plateAt.div(vec2(0.72, 0.36)).add(0.5);
    const inside = select(max(abs(lp.x.sub(0.5)), abs(lp.y.sub(0.5))).lessThan(0.5), float(1), float(0));
    const glyph = vec2(mod(look.w, LABEL_COLUMNS).add(lp.x), floor(look.w.div(LABEL_COLUMNS)).add(lp.y.oneMinus())).div(vec2(LABEL_COLUMNS, rows));
    const ink = texture(atlas, glyph).r.mul(inside).mul(state.y);
    const plate = coverage(sdBox(plateAt, vec2(0.3, 0.14), 0.13)).mul(0.62).mul(state.y);
    let col: N<'vec3'> = vec3(0), alpha: N<'float'> = shadow;
    col = mix(col, rgb(GOLD).mul(1.15), halo);
    alpha = max(alpha, halo.mul(0.9));
    col = mix(col, body, face);
    alpha = max(alpha, face);
    col = mix(col, rgb(0x0b0d10), plate);
    alpha = max(alpha, plate);
    col = mix(col, vec3(tone.x, tone.y, tone.z), ink);
    alpha = max(alpha, ink);
    const material = basic();
    material.colorNode = col;
    material.opacityNode = alpha.mul(state.x);
    const mesh = instancedMesh(flat(QS, QS), material, ALCHEMY_STONES);
    mesh.renderOrder = 4;
    mesh.frustumCulled = false;
    this.stones = mesh;
    this.scene.add(mesh);
  }

  protected override drop(): void {
    for (const t of this.textures) t.dispose();
    this.potions?.dispose();
    this.stones?.dispose();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.since += frame.dtReal;
    this.clock.value += frame.dtReal;
    const step = frame.state.step;
    if (step !== this.step || frame.scrubbed) this.advance(frame, step);
    this.tally(frame, step);
    this.progress = Math.min(1, this.progress + frame.dtReal / MOVE);
    if (this.pending && this.progress >= 0.5) {
      this.pending();
      this.pending = null;
    }
    this.flash.value *= Math.exp(-frame.dtReal / FLASH_SECONDS);
    this.tickU.value = this.cur[ALCHEMY_TICK]!;
    this.trialU.value = this.cur[ALCHEMY_TRIAL]!;
    this.draw(pool);
  }

  private advance(frame: FrameUpdate, step: number): void {
    const now = frame.frameAt(step), before = step > 0 ? frame.frameAt(step - 1) : null, last = this.step;
    const known = before !== null && before.step === step - 1 && now.step === step;
    const forward = !frame.scrubbed && known && step > last;
    const animate = forward && step === last + 1 && this.since >= MOVE;
    this.since = 0;
    this.step = step;
    this.cur.set(now.words);
    this.prev.set(known ? before.words : now.words);
    if (frame.scrubbed) {
      this.flash.value = 0;
      this.pending = null;
    }
    const prev = this.prev, action = alchemyAction(now.actions), cursor = prev[ALCHEMY_CURSOR]!;
    const redeal = known && (now.words[ALCHEMY_TRIAL] !== prev[ALCHEMY_TRIAL] || now.words[ALCHEMY_EPISODE] !== prev[ALCHEMY_EPISODE]);
    this.mover = action >= 2 && action < 5 ? action - 2 : -1;
    const live = this.mover >= 0 && !(alchemyStone(prev, this.mover) & ALCHEMY_SPENT_STONE);
    this.kind = !animate ? 'rest' : redeal ? 'deal' : action <= 1 ? 'walk' : !live ? 'idle'
      : cursor === ALCHEMY_CAULDRON ? 'cash' : alchemyPotion(prev, cursor) & ALCHEMY_SPENT_POTION ? 'idle' : 'dip';
    this.progress = animate ? 0 : 1;
    this.ledger.reward = now.reward;
    if (!forward) return;
    const reward = now.reward, won = frame.events.some((e) => e.type === 'goal'), lost = frame.events.some((e) => e.type === 'harm');
    const pending: (() => void)[] = [];
    if (won) pending.push(() => this.celebrate(reward > 0 ? reward : 1));
    else if (lost) pending.push(() => this.celebrate(reward < 0 ? reward : -1));
    if (this.kind === 'dip') {
      const s = this.mover, changed = (alchemyStone(prev, s) & 7) !== (alchemyStone(now.words, s) & 7);
      const tint = TINTS[alchemyPotion(prev, cursor) >> 3 & 7] ?? 0xffffff, a = slotAngle(cursor);
      if (changed) pending.push(() => this.pool?.ripple({ x: RING * Math.cos(a), y: 0, z: -RING * Math.sin(a) }, tint, 0.7));
    }
    this.pending = pending.length ? () => pending.forEach((fire) => fire()) : null;
  }

  private celebrate(reward: number): void {
    const pool = this.pool;
    if (!pool) return;
    const a = slotAngle(ALCHEMY_CAULDRON), at = { x: RING * Math.cos(a), y: 0, z: -RING * Math.sin(a) };
    if (reward >= this.top) {
      pool.burst({ x: at.x, y: 40, z: at.z }, GOLD, 1.4);
      pool.ripple(at, GOLD, 1.3);
      pool.pulse.fire(0.5);
      this.flashTint.value.set(GOLD);
      this.flash.value = 1;
    } else if (reward > 0) {
      pool.ripple(at, EMBER, 0.8);
      this.flashTint.value.set(EMBER);
      this.flash.value = 0.7;
    } else {
      pool.ripple(at, LOSS, 0.9);
      this.flashTint.value.set(LOSS);
      this.flash.value = 0.8;
    }
  }

  private tally(frame: FrameUpdate, step: number): void {
    const game = this.game;
    if (frame.scrubbed || step < game.to) Object.assign(game, { from: step, to: step, sum: 0, found: false });
    for (let s = game.to + 1; s <= step; s++) {
      const f = frame.frameAt(s);
      if (f.step !== s) { Object.assign(game, { from: step, to: step, sum: 0, found: false }); break; }
      if (f.done) Object.assign(game, { from: s, to: s, sum: 0, found: true });
      else { game.sum += f.reward; game.to = s; }
    }
    for (let budget = 1024; !game.found && budget > 0; budget--) {
      const f = frame.frameAt(game.from);
      if (f.step !== game.from) break;
      if (f.done || game.from === 0) game.found = true;
      else { game.sum += f.reward; game.from--; }
    }
  }

  private stoneValues(): string {
    const out: string[] = [];
    for (let s = 0; s < ALCHEMY_STONES; s++) {
      const held = alchemyStone(this.cur, s);
      out.push(held & ALCHEMY_SPENT_STONE ? '·' : sign(this.values[held & 7]!));
    }
    return out.join('  ');
  }

  private draw(_pool: Stage2D['pool']): void {
    const potions = this.potions, stones = this.stones, selector = this.selector;
    if (!potions || !stones || !selector) return;
    const u = this.progress, kind = this.kind, prev = this.prev, cur = this.cur, from = prev[ALCHEMY_CURSOR]!, to = cur[ALCHEMY_CURSOR]!;
    const glide = kind === 'walk' ? smoothstep01(u) : 1;
    const delta = ((to - from + ALCHEMY_CONTAINERS + 6) % ALCHEMY_CONTAINERS) - 6;
    const angle = slotAngle(from) - delta * glide * TAU / ALCHEMY_CONTAINERS;
    selector.position.set(RING * Math.cos(angle), 0, -RING * Math.sin(angle));
    this.ringU.value = slotRadius(from) + (slotRadius(to) - slotRadius(from)) * glide;
    const target = prev[ALCHEMY_CURSOR]!;

    for (let p = 0; p < ALCHEMY_POTIONS; p++) {
      const drained = kind === 'dip' && p === target;
      const dose = drained && u < 0.5 ? alchemyPotion(prev, p) : alchemyPotion(cur, p);
      const spent = (dose & ALCHEMY_SPENT_POTION) !== 0;
      const fill = drained ? 1 - smoothstep01(clamp01((u - 0.3) / 0.3)) : spent ? 0 : 1;
      const pop = kind === 'deal' ? easeBack(clamp01((u - p * 0.025) / 0.7)) : 1;
      const a = slotAngle(p);
      this.place(potions, p, RING * Math.cos(a), -RING * Math.sin(a), Math.max(0, pop));
      this.color.set(TINTS[dose >> 3 & 7] ?? 0xffffff);
      this.potionTint.setXYZ(p, this.color.r, this.color.g, this.color.b);
      const chosen = (kind === 'walk' ? (u < 0.5 ? from : to) : to) === p ? 1 : 0;
      this.potionState.setXYZW(p, fill, drained ? 1 - smoothstep01(clamp01((u - 0.25) / 0.2)) : spent ? 0 : 1, chosen, p * 1.7);
    }

    const ca = slotAngle(ALCHEMY_CAULDRON), cauldron = { x: RING * Math.cos(ca), z: -RING * Math.sin(ca) };
    for (let s = 0; s < ALCHEMY_STONES; s++) {
      const ha = homeAngle(s), hx = HOME * Math.cos(ha), hz = -HOME * Math.sin(ha);
      const moving = s === this.mover && (kind === 'dip' || kind === 'cash');
      const held = moving && u < 0.5 ? alchemyStone(prev, s) : alchemyStone(cur, s);
      const shown = moving && kind === 'cash' ? alchemyStone(prev, s) : held;
      let x = hx, z = hz, size = 1, alpha = shown & ALCHEMY_SPENT_STONE ? 0 : 1, label = alpha;
      if (moving) {
        const go = smoothstep01(clamp01(u / 0.42)), back = kind === 'dip' ? smoothstep01(clamp01((u - 0.58) / 0.42)) : 0;
        const tx = kind === 'cash' ? cauldron.x : RING * Math.cos(slotAngle(target)), tz = kind === 'cash' ? cauldron.z : -RING * Math.sin(slotAngle(target));
        x = hx + (tx - hx) * (go - back);
        z = hz + (tz - hz) * (go - back);
        size = 1 + 0.18 * (Math.sin(Math.PI * go) + Math.sin(Math.PI * back)) - 0.3 * (go - back);
        label = 1 - (go - back);
        if (kind === 'cash') {
          const sink = smoothstep01(clamp01((u - 0.4) / 0.2));
          size *= 1 - sink;
          alpha = 1 - sink;
        }
      }
      if (kind === 'deal') size *= Math.max(0, easeBack(clamp01((u - 0.2 - s * 0.05) / 0.6)));
      this.spots[s]!.x = x;
      this.spots[s]!.z = z;
      this.place(stones, s, x, z, alpha > 0 ? size : 0);
      const vertex = shown & 7, start = moving ? alchemyStone(prev, s) & 7 : vertex, end = alchemyStone(cur, s) & 7;
      const blend = moving && kind === 'dip' ? smoothstep01(clamp01((u - 0.45) / 0.15)) : 0, lookOf = moving ? prev : cur;
      const f = (k: number) => alchemyFeature(lookOf, start, k) + (alchemyFeature(lookOf, end, k) - alchemyFeature(lookOf, start, k)) * blend;
      this.stoneLook.setXYZW(s, f(0), f(1), f(2), vertex);
      this.stoneState.setXYZW(s, alpha, label, 0, 0);
      const value = this.values[vertex]!;
      this.color.set(value >= this.top ? INK_HIGH : value > 0 ? INK_MID : INK_LOW);
      this.stoneTone.setXYZW(s, this.color.r, this.color.g, this.color.b, value >= this.top ? 0.55 : 0);
    }
    for (const attr of [this.potionTint, this.potionState, this.stoneLook, this.stoneState, this.stoneTone]) attr.needsUpdate = true;
    potions.instanceMatrix.needsUpdate = true;
    stones.instanceMatrix.needsUpdate = true;
  }

  private place(mesh: THREE.InstancedMesh, i: number, x: number, z: number, size: number): void {
    this.matrix.compose(this.at.set(x, 0, z), this.quaternion, this.scale.setScalar(size));
    mesh.setMatrixAt(i, this.matrix);
  }
}

export default (() => new SymbolicAlchemyScene()) satisfies SceneFactory;
