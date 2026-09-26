import * as THREE from 'three/webgpu';
import { abs, clamp, exp, float, fract, fwidth, length, max, min, mix, positionWorld, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { coverage, rgb, sdBox, sdCircle } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { permutedAction, permutedAnswer, permutedCode, permutedEpoch, permutedState } from '@sim/envs/permuted';
import type { EnvFrame } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned, formatStep } from '../../format';

type N<T extends string> = THREE.Node<T>;
type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;
type Scalar = THREE.UniformNode<'float', number>;

const WINDOW = 4096;
const RING = WINDOW + 4;
const BINS = 128;
const PER_BIN = WINDOW / BINS;
const PIPS = 10;
const PIP_STEP = 0.17;
const X_PIP = -7.72;
const X_STATE = -6.9;
const X_CODE = 0;
const X_ACT = 6.2;
const L0 = -6.75;
const L1 = -0.15;
const R0 = 0.15;
const R1 = 5.95;
const WIRE = 0.3;
const HALF = 0.032;
const TOKEN = 0.3;
const LAMP = 0.29;
const SOCKET = 0.38;
const TOP = 4.5;
const LINE_Y = -5.95;
const LINE_H = 0.95;
const LINE_X0 = -9.45;
const LINE_X1 = 7.65;
const FONT = 'system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const GOLD = 0xffcf6a;
const MISS = 0xff5468;
const INK = 0x7f8a8f;
const WARM = new THREE.Color(0xffe2ae);
const CYAN = new THREE.Color(0x6fe3ff);
const GLINT_SECONDS = 0.36;
const FLASH_SECONDS = 0.4;
const RECODE_SECONDS = 0.9;
const FAST = 14;

const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const crisp = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(max(fwidth(positionWorld.x), 1e-6))), 0, 1);
const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const local = (size: number): N<'vec2'> => uv().sub(0.5).mul(size);
const unit = (): N<'vec2'> => uv().sub(0.5).mul(2);
const v3 = (c: THREE.Color) => new THREE.Vector3(c.r, c.g, c.b);

function sprite(color: N<'vec3'>, opacity: N<'float'>, order: number, geometry: THREE.BufferGeometry): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, side: THREE.DoubleSide });
  material.colorNode = color;
  material.opacityNode = opacity;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

function bezier(x0: number, y0: number, x1: number, y1: number, t: number, out: THREE.Vector2): THREE.Vector2 {
  const d = (x1 - x0) * 0.5, s = 1 - t;
  out.x = s * s * s * x0 + 3 * s * s * t * (x0 + d) + 3 * s * t * t * (x1 - d) + t * t * t * x1;
  out.y = s * s * s * y0 + 3 * s * s * t * y0 + 3 * s * t * t * y1 + t * t * t * y1;
  return out;
}

function ribbon(x0: number, y0: number, x1: number, y1: number): THREE.BufferGeometry {
  const n = 48, d = (x1 - x0) * 0.5, pos: number[] = [], uvs: number[] = [], index: number[] = [], p = new THREE.Vector2();
  for (let i = 0; i <= n; i++) {
    const t = i / n, s = 1 - t;
    bezier(x0, y0, x1, y1, t, p);
    const dx = 3 * s * s * d + 6 * s * t * (x1 - x0 - 2 * d) + 3 * t * t * d, dy = 6 * s * t * (y1 - y0), l = Math.hypot(dx, dy) || 1;
    const nx = (-dy / l) * (WIRE / 2), ny = (dx / l) * (WIRE / 2);
    pos.push(p.x - nx, 0, -(p.y - ny), p.x + nx, 0, -(p.y + ny));
    uvs.push(t, 0, t, 1);
    if (i < n) index.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(index);
  return g;
}

function wireMaterial(tint: Tint, alpha: Scalar, lift: Scalar): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, side: THREE.DoubleSide });
  const dist = abs(uv().y.sub(0.5)).mul(WIRE), core = crisp(dist.sub(HALF).sub(lift.mul(0.014)));
  const glow = exp(dist.mul(-26)).mul(lift).mul(0.55);
  material.colorNode = tint.mul(float(0.7).add(lift.mul(0.75)));
  material.opacityNode = max(core, glow).mul(alpha);
  return material;
}

function label(text: string, height: number, tint: number, weight = 600): { mesh: THREE.Mesh; width: number; texture: THREE.DataTexture } {
  const px = 96, canvas = document.createElement('canvas'), context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Permuted labels need a 2D canvas');
  const font = `${weight} ${px}px ${FONT}`;
  context.font = font;
  const w = Math.ceil(context.measureText(text).width + px * 0.3), h = Math.ceil(px * 1.4);
  canvas.width = w;
  canvas.height = h;
  context.font = font;
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, w / 2, h / 2);
  const pixels = context.getImageData(0, 0, w, h).data, alpha = new Uint8Array(w * h);
  for (let row = 0; row < h; row++) for (let column = 0; column < w; column++) alpha[(h - 1 - row) * w + column] = pixels[(row * w + column) * 4 + 3]!;
  const map = new THREE.DataTexture(alpha, w, h, THREE.RedFormat, THREE.UnsignedByteType);
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = true;
  map.needsUpdate = true;
  const tall = height * 1.4, wide = (tall * w) / h;
  return { mesh: sprite(rgb(tint), texture(map, uv()).r, 6, flat(wide, tall)), width: wide, texture: map };
}

class Tape {
  readonly state = new Int8Array(RING);
  readonly code = new Int8Array(RING);
  readonly action = new Int8Array(RING);
  readonly hit = new Int8Array(RING);
  readonly epoch = new Int32Array(RING);
  private readonly seen = new Int32Array(RING).fill(-1);

  constructor(private readonly states: number, private readonly actions: number) {}

  has(s: number): boolean {
    return s >= 0 && this.seen[s % RING] === s;
  }

  clear(): void {
    this.seen.fill(-1);
  }

  sync(first: number, last: number, at: (s: number) => EnvFrame | null): boolean {
    let fresh = false;
    for (let s = Math.max(0, first); s <= last; s++) {
      const slot = s % RING;
      if (this.seen[slot] === s) continue;
      const f = at(s);
      if (!f) continue;
      this.state[slot] = permutedState(f);
      this.code[slot] = permutedCode(f, this.states);
      this.action[slot] = permutedAction(f, this.actions);
      this.hit[slot] = f.reward > 0 ? 1 : 0;
      this.epoch[slot] = permutedEpoch(f);
      this.seen[slot] = s;
      fresh = true;
    }
    return fresh;
  }
}

interface Cell {
  fill: Tint;
  ring: Tint;
  ringFade: Scalar;
  glow: Tint;
}

interface Wire {
  mesh: THREE.Mesh;
  tint: Tint;
  alpha: Scalar;
  lift: Scalar;
  key: number;
}

class PermutedScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 3;
  protected readonly stageName = 'permuted';
  override legend: LegendSpec | null = null;

  private states = 8;
  private actions = 4;
  private period = 0;
  private tape: Tape | null = null;
  private answers: Int8Array = new Int8Array(0);
  private readonly wiring = new Int8Array(64).fill(-1);
  private readonly hues: THREE.Color[] = [];
  private readonly textures: THREE.Texture[] = [];
  private readonly geometries = new Map<number, THREE.BufferGeometry>();
  private readonly timeline = new Float32Array(BINS * 4);
  private readonly timelineTexture = new THREE.DataTexture(this.timeline, BINS, 1, THREE.RGBAFormat, THREE.FloatType);
  private tokens: Cell[] = [];
  private lamps: Cell[] = [];
  private sockets: Cell[] = [];
  private left: Wire[] = [];
  private right: Wire[] = [];
  private pips: THREE.InstancedMesh | null = null;
  private pulse: { mesh: THREE.Mesh; fade: Scalar } | null = null;
  private glint: { mesh: THREE.Mesh; fade: Scalar } | null = null;
  private flash: { mesh: THREE.Mesh; fade: Scalar } | null = null;
  private sweep: { mesh: THREE.Mesh; fade: Scalar } | null = null;
  private readonly trial = { valid: false, state: -1, code: -1, press: -1, answer: -1, hit: false };
  private readonly ledger = { step: 0, reward: 0, hits: 0, judged: 0, epoch: 0 };
  private readonly color = new THREE.Color();
  private readonly matrix = new THREE.Matrix4();
  private readonly point = new THREE.Vector2();
  private arrived = -Infinity;
  private glintAt = -Infinity;
  private flashAt = -Infinity;
  private recodeAt = -Infinity;
  private lastStep = -1;
  private lastPosition = 0;
  private pace = 0;
  private started = false;

  private rowY(k: number): number {
    return (this.states - 1) / 2 - k;
  }

  private actY(a: number): number {
    const pitch = Math.min(1.5, (this.states - 0.5) / this.actions);
    return ((this.actions - 1) / 2 - a) * pitch;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.states = constant(env, 'PERMUTED_STATES');
    this.actions = constant(env, 'PERMUTED_ACTIONS');
    this.period = parameter(env, 'recodePeriod');
    if (env.obs !== this.states + 1 || env.view !== this.states + 1 || env.words !== 1) throw new Error(`${env.envId} shows ${env.obs} observations and ${env.view} view words for ${this.states} states`);
    this.tape = new Tape(this.states, this.actions);
    return new THREE.Box2(new THREE.Vector2(-9.85, -(TOP + 0.45)), new THREE.Vector2(8.05, -(LINE_Y - LINE_H / 2 - 0.25)));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xb4bcc2).multiplyScalar(0.55);
  }

  protected override overlayOptions(): GroundTruthOptions {
    return {
      debug: (_frame, kit) => {
        const trial = this.trial;
        for (let k = 0; k < this.states; k++) {
          const c = this.wiring[k]!;
          if (c < 0) continue;
          const wire = kit.curve(`code${k}`, 2, TOKENS.uiCyan);
          wire.point(X_STATE, 0.1, -this.rowY(k));
          wire.point(X_CODE, 0.1, -this.rowY(c));
          const a = this.answers[k];
          if (a === undefined) continue;
          const key = kit.curve(`key${k}`, 2, TOKENS.cueMagenta);
          key.point(X_CODE, 0.1, -this.rowY(c));
          key.point(X_ACT, 0.1, -this.actY(a));
        }
        if (!trial.valid) return;
        kit.marks('state', 1).set(X_STATE, 0.1, -this.rowY(trial.state), 0.22, 0xffffff);
        kit.marks('code', 1).set(X_CODE, 0.1, -this.rowY(trial.code), 0.22, 0xffffff);
        kit.marks('answer', 1).set(X_ACT, 0.1, -this.actY(trial.answer), 0.3, TOKENS.celebrationGold);
        kit.marks('press', 1).set(X_ACT + 0.55, 0.1, -this.actY(trial.press), 0.16, trial.hit ? TOKENS.celebrationGold : TOKENS.failureBlue);
      },
    };
  }

  protected build(ctx: SceneContext, { light }: Stage2D): void {
    const accent = this.accentOf(ctx), n = this.states;
    this.scene.background = new THREE.Color(0x07090a);
    for (let k = 0; k < n; k++) this.hues.push(new THREE.Color().setHSL((k / n + 0.02) % 1, 0.62, 0.6));
    this.timelineTexture.magFilter = THREE.NearestFilter;
    this.timelineTexture.minFilter = THREE.NearestFilter;
    this.timelineTexture.needsUpdate = true;
    this.textures.push(this.timelineTexture);

    const rows = [
      { key: 'state', label: 'hidden state', tint: accent, value: () => (this.trial.valid ? `${this.trial.state} · lit input x${this.trial.code}` : '—') },
      { key: 'action', label: 'action', tint: 0xdce4e8, value: () => (this.trial.valid ? `pressed ${this.trial.press} · key ${this.trial.answer}` : '—') },
      { key: 'reward', label: 'reward', tint: GOLD, value: () => (this.ledger.step < 1 ? '—' : formatSigned(this.ledger.reward)) },
      { key: 'accuracy', label: 'recent accuracy', tint: accent, value: () => (this.ledger.judged > 0 ? `${formatPercent(this.ledger.hits / this.ledger.judged)} over ${formatCount(this.ledger.judged)}` : '—') },
    ];
    if (this.period > 0) rows.push({ key: 'epoch', label: 'code', tint: CYAN.getHex(), value: () => `epoch ${formatCount(this.ledger.epoch)} · recodes in ${formatCount(this.period - (this.ledger.step % this.period))}` });
    this.legend = {
      title: 'PERMUTED',
      rows,
      note: () => (this.period > 0 ? `inputs reshuffle every ${formatStep(this.period)} steps; the key stays` : 'one permutation for the whole run'),
      bar: () => (this.ledger.judged > 0 ? this.ledger.hits / this.ledger.judged : 0),
    };

    this.scene.add(this.backdrop(light.lit));
    this.scene.add(this.strip());

    for (let k = 0; k < n; k++) {
      const tint: Tint = uniform(v3(this.hues[k]!)), alpha: Scalar = uniform(0), lift: Scalar = uniform(0);
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), wireMaterial(tint, alpha, lift));
      mesh.renderOrder = 2;
      mesh.frustumCulled = false;
      mesh.visible = false;
      this.scene.add(mesh);
      this.left.push({ mesh, tint, alpha, lift, key: -1 });
      const tintR: Tint = uniform(v3(this.hues[k]!)), alphaR: Scalar = uniform(0), liftR: Scalar = uniform(0);
      const meshR = new THREE.Mesh(new THREE.BufferGeometry(), wireMaterial(tintR, alphaR, liftR));
      meshR.renderOrder = 2;
      meshR.frustumCulled = false;
      meshR.visible = false;
      this.scene.add(meshR);
      this.right.push({ mesh: meshR, tint: tintR, alpha: alphaR, lift: liftR, key: -1 });
    }

    for (let k = 0; k < n; k++) {
      const y = this.rowY(k);
      this.tokens.push(this.disc(X_STATE, y, this.hues[k]!));
      this.put(`${k}`, 0.26, 0x101214, X_STATE, y, 0, 800);
      this.lamps.push(this.square(X_CODE, y, LAMP, 0.08, light));
      this.put(`x${k}`, 0.2, 0x59636a, X_CODE, y, 0, 700);
    }
    const bias = this.square(X_CODE, this.rowY(n - 1) - 1.05, LAMP, 0.08, light);
    bias.fill.value.copy(v3(WARM.clone().multiplyScalar(0.55)));
    this.put('bias', 0.2, 0x3a3226, X_CODE, this.rowY(n - 1) - 1.05, 0, 700);
    for (let a = 0; a < this.actions; a++) {
      this.sockets.push(this.square(X_ACT, this.actY(a), SOCKET, 0.12, light));
      this.put(`${a}`, 0.3, 0x6c767c, X_ACT, this.actY(a), 0, 700);
    }

    this.put('recent answers', 0.22, INK, X_PIP - ((PIPS - 1) * PIP_STEP) / 2, TOP - 0.05);
    this.put('hidden state', 0.22, INK, X_STATE + 0.2, TOP - 0.05);
    this.put('permutation', 0.26, INK, (L0 + L1) / 2, TOP - 0.05);
    this.put('observation', 0.22, INK, X_CODE, TOP - 0.05);
    this.put('answer key', 0.26, INK, (R0 + R1) / 2, TOP - 0.05);
    this.put('action', 0.22, INK, X_ACT, TOP - 0.05);
    this.put(`reward over the last ${formatCount(WINDOW)} steps`, 0.2, INK, LINE_X0, LINE_Y + LINE_H / 2 + 0.22, -1);
    this.put(`dashes: chance, 1 in ${this.actions}`, 0.2, 0x6a7479, LINE_X1, LINE_Y + LINE_H / 2 + 0.22, 1);

    const pipMaterial = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    pipMaterial.colorNode = vec3(1);
    pipMaterial.opacityNode = coverage(sdCircle(unit(), 0.8));
    const pips = instancedMesh(flat(1, 1), pipMaterial, n * PIPS);
    pips.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * PIPS * 3), 3);
    pips.count = 0;
    pips.frustumCulled = false;
    pips.renderOrder = 3;
    this.scene.add(pips);
    this.pips = pips;

    const pulseFade: Scalar = uniform(0), p = unit();
    const pulse = sprite(vec3(1.5, 1.45, 1.35), max(coverage(sdCircle(p, 0.3)), exp(length(p).mul(-4)).mul(0.7)).mul(pulseFade), 5, flat(0.5, 0.5));
    const glintFade: Scalar = uniform(0), arms = min(sdBox(p, vec2(0.95, 0.045), 0.045), sdBox(p, vec2(0.045, 0.95), 0.045));
    const glint = sprite(rgb(GOLD).mul(1.5), max(coverage(arms).mul(float(1).sub(length(p).mul(0.7))), exp(length(p).mul(-5)).mul(0.8)).mul(glintFade), 8, flat(1, 1));
    const flashFade: Scalar = uniform(0), flash = sprite(rgb(MISS).mul(1.3), coverage(abs(length(p).sub(0.85)).sub(0.08)).mul(flashFade), 8, flat(1, 1));
    const sweepFade: Scalar = uniform(0), q = chart();
    const band = exp(abs(q.x.sub(float(L0).add(sweepFade.oneMinus().mul(L1 - L0)))).mul(-3)).mul(sweepFade);
    const sweep = sprite(rgb(CYAN), band.mul(0.35), 1, flat(L1 - L0, n + 0.2).translate((L0 + L1) / 2, 0, 0));
    for (const m of [pulse, glint, flash, sweep]) {
      m.visible = false;
      this.scene.add(m);
    }
    this.pulse = { mesh: pulse, fade: pulseFade };
    this.glint = { mesh: glint, fade: glintFade };
    this.flash = { mesh: flash, fade: flashFade };
    this.sweep = { mesh: sweep, fade: sweepFade };
  }

  private backdrop(lit: (albedo: N<'vec3'>) => N<'vec3'>): THREE.Mesh {
    const p = chart(), n = this.states, top = this.rowY(0) + 0.62, bottom = this.rowY(n - 1) - 0.62;
    const bay = (x0: number, x1: number, y0: number, y1: number) => sdBox(p.sub(vec2((x0 + x1) / 2, (y0 + y1) / 2)), vec2((x1 - x0) / 2, (y1 - y0) / 2), 0.22);
    const left = bay(-9.55, -6.3, bottom, top), mid = bay(-0.62, 0.62, bottom - 1.05, top), end = bay(X_ACT - 0.72, X_ACT + 0.72, this.actY(this.actions - 1) - 0.7, this.actY(0) + 0.7);
    const bays = min(min(left, mid), end);
    const hole = length(fract(p.mul(2)).sub(0.5)).mul(0.5).sub(0.035);
    const field = crisp(max(abs(p.x.sub((L0 + L1) / 2)).sub((L1 - L0) / 2 + 0.1), abs(p.y).sub(n / 2 + 0.1))).add(crisp(max(abs(p.x.sub((R0 + R1) / 2)).sub((R1 - R0) / 2 - 0.2), abs(p.y).sub(n / 2 + 0.1))));
    let color: N<'vec3'> = rgb(0x0b0d0e);
    color = mix(color, rgb(0x050607), crisp(hole).mul(clamp(field, 0, 1)).mul(0.8));
    color = mix(color, rgb(0x15191b), crisp(bays));
    color = mix(color, rgb(0x262c2f), crisp(abs(bays).sub(0.012)));
    return sprite(lit(color), float(1), 0, flat(20, 13.5).translate(-0.9, 0, 0.35));
  }

  private strip(): THREE.Mesh {
    const w = LINE_X1 - LINE_X0, h = LINE_H, cell = texture(this.timelineTexture, vec2(uv().x, 0.5));
    const y = uv().y.mul(h), frame = sdBox(uv().sub(0.5).mul(vec2(w, h)), vec2(w / 2 - 0.02, h / 2 - 0.02), 0.08);
    const bar = crisp(y.sub(cell.r.mul(h - 0.08).add(0.04))).mul(cell.g);
    const chance = crisp(abs(y.sub(0.25 * (h - 0.08) + 0.04)).sub(0.007)).mul(crisp(abs(fract(positionWorld.x.mul(5)).sub(0.5)).sub(0.25)));
    let color: N<'vec3'> = rgb(0x0e1113);
    color = mix(color, rgb(CYAN).mul(0.32), cell.b.mul(crisp(frame)));
    color = mix(color, rgb(GOLD).mul(0.78), bar.mul(crisp(frame.add(0.03))));
    color = mix(color, rgb(0x8a959b), chance.mul(0.7));
    color = mix(color, rgb(0x2a3134), crisp(abs(frame).sub(0.012)));
    const mesh = sprite(color, crisp(frame.sub(0.012)), 1, flat(w, h));
    mesh.position.set((LINE_X0 + LINE_X1) / 2, 0, -LINE_Y);
    return mesh;
  }

  private disc(x: number, y: number, hue: THREE.Color): Cell {
    const fill: Tint = uniform(v3(hue)), ring: Tint = uniform(v3(hue)), ringFade: Scalar = uniform(0), glow: Tint = uniform(new THREE.Vector3());
    const size = 1.1, p = local(size), d = sdCircle(p, TOKEN), halo = abs(length(p).sub(TOKEN + 0.13)).sub(0.03);
    let color: N<'vec3'> = fill;
    color = mix(color, fill.mul(1.35), crisp(abs(d.add(0.035)).sub(0.012)));
    const mesh = sprite(mix(ring, color, crisp(d)), max(crisp(d), crisp(halo).mul(ringFade)), 3, flat(size, size));
    mesh.position.set(x, 0, -y);
    this.scene.add(mesh);
    return { fill, ring, ringFade, glow };
  }

  private square(x: number, y: number, half: number, radius: number, light: Stage2D['light']): Cell {
    const fill: Tint = uniform(new THREE.Vector3(0.07, 0.085, 0.09)), ring: Tint = uniform(new THREE.Vector3(1, 1, 1)), ringFade: Scalar = uniform(0), glow: Tint = uniform(new THREE.Vector3());
    const size = 2 * half + 0.4, p = local(size), d = sdBox(p, vec2(half, half), radius), outer = abs(sdBox(p, vec2(half + 0.1, half + 0.1), radius + 0.08)).sub(0.028);
    let color: N<'vec3'> = fill.mul(float(0.9).add(uv().y.mul(0.2)));
    color = mix(color, rgb(0x303739), crisp(abs(d).sub(0.014)));
    const mesh = sprite(mix(ring, color, crisp(d)), max(crisp(d), crisp(outer).mul(ringFade)), 3, flat(size, size));
    mesh.position.set(x, 0, -y);
    const emitter = light.emitter(flat(2 * half + 0.1, 2 * half + 0.1), glow);
    emitter.position.set(x, 0, -y);
    this.scene.add(mesh, emitter);
    return { fill, ring, ringFade, glow };
  }

  private put(text: string, height: number, tint: number, x: number, y: number, align: -1 | 0 | 1 = 0, weight = 600): void {
    const made = label(text, height, tint, weight);
    made.mesh.position.set(x - (align * made.width) / 2, 0, -y);
    this.textures.push(made.texture);
    this.scene.add(made.mesh);
  }

  private route(wire: Wire, key: number, x0: number, y0: number, x1: number, y1: number): void {
    wire.mesh.visible = key >= 0;
    if (key < 0 || wire.key === key) return;
    wire.key = key;
    let g = this.geometries.get(key);
    if (!g) {
      g = ribbon(x0, y0, x1, y1);
      this.geometries.set(key, g);
    }
    wire.mesh.geometry = g;
  }

  private rebuild(i: number): void {
    const tape = this.tape!, n = this.states, pips = this.pips!;
    const o = Math.max(0, i - 1), epoch = tape.has(o) ? tape.epoch[o % RING]! : -1;
    this.wiring.fill(-1);
    for (let s = o; s >= Math.max(0, i - WINDOW) && epoch >= 0; s--) {
      if (!tape.has(s)) continue;
      const slot = s % RING;
      if (tape.epoch[slot] !== epoch) break;
      if (this.wiring[tape.state[slot]!]! < 0 && tape.code[slot]! >= 0) this.wiring[tape.state[slot]!] = tape.code[slot]!;
    }
    for (let k = 0; k < n; k++) {
      const c = this.wiring[k]!, a = this.answers[k];
      this.route(this.left[k]!, c < 0 ? -1 : k * 64 + c, L0, this.rowY(k), L1, this.rowY(c));
      this.route(this.right[k]!, c < 0 || a === undefined ? -1 : 4096 + c * 64 + a, R0, this.rowY(c), R1, this.actY(a ?? 0));
    }

    const filled = new Int32Array(n);
    let count = 0, hits = 0, judged = 0;
    this.timeline.fill(0);
    const base = (Math.floor(i / PER_BIN) + 1) * PER_BIN - WINDOW;
    for (let s = i; s >= Math.max(1, base); s--) {
      if (!tape.has(s) || !tape.has(s - 1)) continue;
      const slot = s % RING, k = tape.state[(s - 1) % RING]!, hit = tape.hit[slot] === 1;
      judged++;
      if (hit) hits++;
      const b = Math.floor((s - base) / PER_BIN);
      this.timeline[b * 4] = this.timeline[b * 4]! + (hit ? 1 : 0);
      this.timeline[b * 4 + 1] = this.timeline[b * 4 + 1]! + 1;
      if (tape.epoch[slot] !== tape.epoch[(s - 1) % RING]) this.timeline[b * 4 + 2] = 1;
      if (k >= 0 && k < n && filled[k]! < PIPS) {
        const j = filled[k]!++;
        this.matrix.makeScale(0.12, 1, 0.12).setPosition(X_PIP - j * PIP_STEP, 0, -this.rowY(k));
        pips.setMatrixAt(count, this.matrix);
        this.color.set(hit ? GOLD : MISS).multiplyScalar(hit ? 0.95 - 0.05 * j : 1.1 - 0.05 * j);
        pips.setColorAt(count++, this.color);
      }
    }
    for (let b = 0; b < BINS; b++) {
      const total = this.timeline[b * 4 + 1]!;
      this.timeline[b * 4] = total > 0 ? this.timeline[b * 4]! / total : 0;
      this.timeline[b * 4 + 1] = total > 0 ? 1 : 0;
    }
    this.timelineTexture.needsUpdate = true;
    pips.count = count;
    pips.instanceMatrix.needsUpdate = true;
    pips.instanceColor!.needsUpdate = true;
    this.ledger.hits = hits;
    this.ledger.judged = judged;

    const t = this.trial;
    t.valid = i >= 1 && tape.has(i) && tape.has(i - 1) && tape.code[(i - 1) % RING]! >= 0;
    if (t.valid) {
      const prev = (i - 1) % RING;
      t.state = tape.state[prev]!;
      t.code = tape.code[prev]!;
      t.press = tape.action[i % RING]!;
      t.answer = this.answers[t.state] ?? -1;
      t.hit = tape.hit[i % RING] === 1;
      if (t.answer < 0) t.valid = false;
    }
    this.ledger.epoch = tape.has(i) ? tape.epoch[i % RING]! : 0;
  }

  protected tick(frame: FrameUpdate, { time }: Stage2D): void {
    const tape = this.tape;
    if (!tape || !this.pulse || !this.glint || !this.flash || !this.sweep) return;
    const i = frame.state.step;
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const fast = this.pace > FAST, at = (s: number): EnvFrame | null => {
      const f = frame.frameAt(s);
      return f.step === s ? f : null;
    };
    if (frame.scrubbed || !this.started) {
      tape.clear();
      this.arrived = this.glintAt = this.flashAt = this.recodeAt = -Infinity;
      this.started = true;
    }
    if (this.answers.length === 0) {
      const f = at(i);
      if (f && f.view.length > this.states) this.answers = Int8Array.from({ length: this.states }, (_, k) => permutedAnswer(f, k));
    }
    const fresh = tape.sync(i - WINDOW, i, at), advanced = !frame.scrubbed && i !== this.lastStep;
    if (advanced && !fast) {
      this.arrived = time;
      for (const event of frame.events) {
        if (event.type === 'goal') this.glintAt = time;
        else if (event.type === 'harm') this.flashAt = time;
        else if (event.type === 'cue') this.recodeAt = time;
      }
    }
    if (fresh || advanced || frame.scrubbed) this.rebuild(i);
    this.lastStep = i;
    this.ledger.step = i;
    this.ledger.reward = frame.state.reward;
    this.animate(time);
  }

  private animate(time: number): void {
    const t = this.trial, age = time - this.arrived, live = Number.isFinite(age);
    const p = live ? clamp01(age * this.controlHz * 1.35) : 1;
    const travel = smoothstep01(p / 0.45), lamp = smoothstep01((p - 0.38) / 0.12), press = smoothstep01((p - 0.5) / 0.1), verdict = smoothstep01((p - 0.66) / 0.12);
    const recode = (time - this.recodeAt) / RECODE_SECONDS, settle = recode >= 0 && recode < 1 ? smoothstep01(recode) : 1;
    for (let k = 0; k < this.states; k++) {
      const on = t.valid && t.state === k, hue = this.hues[k]!;
      const left = this.left[k]!, right = this.right[k]!;
      left.lift.value = on ? travel : 0;
      left.alpha.value = (t.valid && !on ? 0.5 : 0.85) * settle;
      right.lift.value = on ? verdict : 0;
      right.alpha.value = t.valid && !on ? 0.3 : 0.6;
      const token = this.tokens[k]!;
      this.color.copy(hue).multiplyScalar(on ? 1.05 : t.valid ? 0.42 : 0.7);
      token.fill.value.set(this.color.r, this.color.g, this.color.b);
      token.ringFade.value = on ? 0.9 : 0;
      const lampCell = this.lamps[k]!, lit = t.valid && t.code === k ? lamp : 0;
      this.color.set(0x121617).lerp(WARM, lit).multiplyScalar(1 + 0.12 * lit);
      lampCell.fill.value.set(this.color.r, this.color.g, this.color.b);
      lampCell.ring.value.copy(v3(this.color.copy(WARM)));
      lampCell.ringFade.value = lit * 0.7;
      this.color.copy(WARM).multiplyScalar(2.6 * lit);
      lampCell.glow.value.set(this.color.r, this.color.g, this.color.b);
    }
    for (let a = 0; a < this.actions; a++) {
      const socket = this.sockets[a]!, pressed = t.valid && t.press === a, answer = t.valid && t.answer === a;
      this.color.set(0x121617);
      if (pressed) this.color.lerp(new THREE.Color(0xdfe6ea), press).lerp(new THREE.Color(t.hit ? GOLD : MISS), verdict);
      socket.fill.value.set(this.color.r, this.color.g, this.color.b);
      socket.ring.value.copy(v3(this.color.set(GOLD)));
      socket.ringFade.value = answer ? verdict : 0;
      this.color.set(pressed ? (t.hit ? GOLD : MISS) : 0x000000).multiplyScalar(pressed ? 2.2 * press : 0);
      socket.glow.value.set(this.color.r, this.color.g, this.color.b);
    }

    const pulse = this.pulse!;
    pulse.mesh.visible = live && t.valid && p < 0.46;
    if (pulse.mesh.visible) {
      bezier(L0, this.rowY(t.state), L1, this.rowY(t.code), travel, this.point);
      pulse.mesh.position.set(this.point.x, 0, -this.point.y);
      pulse.fade.value = 1 - smoothstep01((p - 0.36) / 0.1);
    }
    const glint = this.glint!, shine = (time - this.glintAt) / GLINT_SECONDS - 0.2;
    glint.mesh.visible = t.valid && t.hit && shine >= 0 && shine < 1;
    if (glint.mesh.visible) {
      glint.mesh.position.set(X_ACT, 0, -this.actY(t.press));
      glint.mesh.scale.setScalar(0.9 + 0.8 * (1 - (1 - shine) ** 3));
      glint.fade.value = (1 - shine) ** 2;
    }
    const flash = this.flash!, burst = (time - this.flashAt) / FLASH_SECONDS - 0.2;
    flash.mesh.visible = t.valid && !t.hit && burst >= 0 && burst < 1;
    if (flash.mesh.visible) {
      flash.mesh.position.set(X_ACT, 0, -this.actY(t.press));
      flash.mesh.scale.setScalar(0.9 + 0.9 * (1 - (1 - burst) ** 3));
      flash.fade.value = (1 - burst) ** 2;
    }
    const sweep = this.sweep!;
    sweep.mesh.visible = recode >= 0 && recode < 1;
    sweep.fade.value = sweep.mesh.visible ? 1 - recode : 0;
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    for (const g of this.geometries.values()) g.dispose();
    this.textures.length = 0;
    this.geometries.clear();
  }
}

export default (() => new PermutedScene()) satisfies SceneFactory;
