import * as THREE from 'three/webgpu';
import { uniform, uniformArray } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { regressionBin, regressionGuess, regressionTarget } from '@sim/envs/regression';
import type { EnvFrame } from '@sim/types';
import { wordFloat } from '@sim/words';
import { constant, constantList, parameter } from '../../data/envs';
import { formatCount, formatPercent, formatRate, formatSigned } from '../../format';
import {
  AXIS, BEAD, EXTENT, FLOOR, GAUGE, HALF, HEAD, INK, IVORY, LABEL_X, MOUTH, PITCH, PLATE_H, PLATE_W, PLATE_Y, ROWS, STACK, WG, XG, X_RANGE,
  backdrop, ball, beadMaterial, board, classColor, flat, gauges, label, path, plateX, rowY, strip, token, verdict,
  type Label, type Points, type Scalar, type Tint,
} from './art';

type At = (s: number) => EnvFrame | null;
type Span = readonly [number, number];

const WINDOW = 180;
const RECENT = 1024;
const RING = RECENT + 2;
const FAST = 12;
const SWEEP = 64;
const SPREAD = 3.5;
const SHRINK = 0.8;
const MANTISSAS = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8] as const;
const MAJORS = [1, 2, 5] as const;
const PREP: Span = [0, 0.12];
const SLIDE: Span = [0, 0.14];
const DRAIN: Span = [0, 0.16];
const FALL: Span = [0.12, 0.76];
const DROP: Span = [0.76, 1];
const PAUSE = 0.12;
const FLASH_SECONDS = 0.55;
const GLOW_STEPS = 2;
const REST = 0.32;
const HERO = 1.45;
const GOLD = TOKENS.celebrationGold;
const MISS = TOKENS.failureBlue;
const INNER = GAUGE / 2 - 0.08;

const phase = (t: number, [a, b]: Span) => clamp01((t - a) / (b - a));
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
const minus = (text: string) => text.replaceAll('-', '−');
const edgeText = (e: number) => minus(String(Number(e.toPrecision(3))));

function ladder(value: number): number {
  for (let e = Math.floor(Math.log10(value)) - 1; ; e++) for (const m of MANTISSAS) if (m * 10 ** e >= value * (1 - 1e-9)) return m * 10 ** e;
}

function below(value: number): number {
  let best = 0;
  for (let e = Math.floor(Math.log10(value)) - 1; e <= Math.floor(Math.log10(value)); e++) for (const m of MANTISSAS) if (m * 10 ** e < value * (1 - 1e-9)) best = Math.max(best, m * 10 ** e);
  return best;
}

function majorFor(range: number): number {
  for (let e = 0; ; e++) for (const m of MAJORS) if (range / (m * 10 ** e) <= 7) return m * 10 ** e;
}

interface Sample { x: Float32Array; w: Float32Array; bias: number; sums: Float64Array; y: number; bin: number; norm: number; moved: number; bet: number; hit: boolean }

interface Display { s: number; ok: boolean; complete: boolean; shown: Sample; hits: number; judged: number }

const blankSample = (n: number): Sample => ({ x: new Float32Array(n), w: new Float32Array(n), bias: 0, sums: new Float64Array(n + 1), y: 0, bin: 0, norm: 0, moved: 0, bet: -1, hit: false });
const blankDisplay = (n: number): Display => ({ s: -1, ok: false, complete: false, shown: blankSample(n), hits: 0, judged: 0 });

class RegressionScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 1.5;
  protected readonly stageName = 'regression';
  override legend: LegendSpec | null = null;

  private inputs = 0;
  private bins = 0;
  private edges: number[] = [];
  private drift = 0;
  private reach = 1;
  private origin: Float32Array | null = null;
  private cur = blankDisplay(0);
  private nxt = blankDisplay(0);
  private readonly yOf = new Float64Array(RING);
  private readonly binOf = new Int8Array(RING);
  private readonly hitOf = new Uint8Array(RING);
  private readonly tagOf = new Int32Array(RING).fill(-1);

  private range = 0;
  private span = 1;
  private bounds = new Float64Array(0);
  private slotsIn = new Int32Array(0);
  private firstSlot = new Int32Array(0);
  private slotCount = 0;
  private bead = BEAD;
  private column = new Int32Array(0);
  private readonly winSlot = new Int32Array(WINDOW);
  private readonly winLevel = new Int32Array(WINDOW);
  private readonly winHit = new Uint8Array(WINDOW);
  private winFirst = 0;
  private winSize = 0;
  private stackedFor = -1;
  private stackedRange = 0;
  private drainSlot = -1;
  private drainK = 0;

  private rows = new Float32Array(4);
  private rowTexture: THREE.DataTexture | null = null;
  private readonly textures: THREE.Texture[] = [];
  private readonly spanU = uniform(1);
  private readonly minorU = uniform(0.5);
  private readonly majorU = uniform(1);
  private regionsU: Points | null = null;
  private readonly scaleU = uniform(1);
  private readonly reveals: Scalar[] = [uniform(0), uniform(0)];
  private readonly pathAlphas: Scalar[] = [uniform(0), uniform(0)];
  private paths: Points[] = [];
  private route = new Float64Array(0);
  private lengths = new Float64Array(0);
  private readonly ballAlpha = uniform(0);
  private readonly ballLight: Tint = uniform(new THREE.Vector3());
  private readonly tokenAlpha = uniform(0);
  private readonly tokenLight: Tint = uniform(new THREE.Vector3());
  private readonly verdictTint: Tint = uniform(new THREE.Vector3(1, 1, 1));
  private readonly verdictAlpha = uniform(0);
  private readonly stripAlpha = uniform(0);
  private readonly stripLight: Tint = uniform(new THREE.Vector3());
  private readonly landLight: Tint = uniform(new THREE.Vector3());
  private readonly numberTints: Tint[] = [];
  private ballMesh: THREE.Mesh | null = null;
  private ballLamp: THREE.Mesh | null = null;
  private tokenMesh: THREE.Mesh | null = null;
  private tokenLamp: THREE.Mesh | null = null;
  private verdictMesh: THREE.Mesh | null = null;
  private stripMesh: THREE.Mesh | null = null;
  private stripLamp: THREE.Mesh | null = null;
  private landLamp: THREE.Mesh | null = null;
  private beads: THREE.InstancedMesh | null = null;
  private beadLook: THREE.InstancedBufferAttribute | null = null;
  private readonly ticks = new Map<string, Label>();
  private readonly accent = new THREE.Color();
  private readonly tint = new THREE.Color();
  private palette: THREE.Color[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly spot = new THREE.Vector2();
  private readonly at = new THREE.Vector2();
  private pace = 0;
  private lastPosition = 0;
  private revealAt = -Infinity;
  private landedAt = new THREE.Vector2();
  private flying = false;
  private panel: Sample | null = null;

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.inputs = parameter(env, 'inputs');
    this.bins = constant(env, 'REGRESSION_BINS');
    this.edges = constantList(env, 'REGRESSION_EDGE').map(Math.fround);
    this.drift = parameter(env, 'drift');
    if (env.words !== this.inputs || env.obs !== this.inputs + 1 || env.actions !== this.bins || this.edges.length !== this.bins - 1) {
      throw new Error(`${env.envId} keeps ${env.words} words, shows ${env.obs} observations and takes ${env.actions} actions for ${this.inputs} inputs cut by ${this.edges.length} edges`);
    }
    const n = this.inputs;
    this.reach = 1.4 * Math.max(...this.edges.map(Math.abs));
    this.cur = blankDisplay(n);
    this.nxt = blankDisplay(n);
    this.rows = new Float32Array((n + 1) * 4);
    this.route = new Float64Array((2 * n + 2) * 2);
    this.lengths = new Float64Array(2 * n + 1);
    this.bounds = new Float64Array(this.bins * 2);
    this.slotsIn = new Int32Array(this.bins);
    this.firstSlot = new Int32Array(this.bins);
    this.column = new Int32Array(Math.ceil((2 * HALF) / PITCH) + this.bins);
    return EXTENT;
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xeadcc6).multiplyScalar(0.64);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 36 };
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const accent = this.accentOf(ctx);
    return {
      debug: (_frame, kit) => {
        const cur = this.cur, n = this.inputs;
        const trace = (name: string, sample: Sample | null, color: number) => {
          const line = kit.curve(name, 2 * n + 2, color), joints = kit.marks(name, 2 * n + 2);
          if (!sample) return;
          this.vertices(sample, this.route);
          for (let k = 0; k < 2 * n + 2; k++) {
            line.point(this.route[k * 2]!, 0.1, -this.route[k * 2 + 1]!);
            if (k % 2 === 1 || k === 2 * n + 1) joints.set(this.route[k * 2]!, 0.1, -this.route[k * 2 + 1]!, 0.035, color);
          }
        };
        trace('settled', cur.ok && cur.s >= 1 ? cur.shown : null, TOKENS.uiCyan);
        trace('incoming', this.flying ? this.nxt.shown : null, 0xffffff);
        this.edges.forEach((e, k) => {
          const line = kit.curve(`edge${k}`, 2, 0xffffff), x = e * this.span;
          line.point(x, 0.1, -AXIS);
          line.point(x, 0.1, -FLOOR);
        });
        const panel = this.panel ?? cur.shown, inputs = kit.marks('inputs', n + 1), weights = kit.marks('weights', n);
        if (cur.ok) for (let j = 0; j <= n; j++) {
          const y = j < n ? rowY(j, n) : AXIS, value = j < n ? panel.x[j]! : panel.bias;
          inputs.set(XG + Math.max(-1, Math.min(1, value / X_RANGE)) * INNER, 0.1, -y, 0.03, 0xffffff);
          if (j < n) weights.set(WG + Math.max(-1, Math.min(1, panel.w[j]! / this.scaleU.value)) * INNER, 0.1, -y, 0.03, 0xffffff);
        }
        const tops = kit.marks('columns', this.column.length);
        for (let slot = 0; slot < this.slotCount; slot++) {
          const count = this.column[slot]!;
          const shown = count - (slot === this.drainSlot ? this.drainK : 0);
          if (shown > 0) tops.set(this.slotX(slot), 0.1, -this.levelY(shown - 0.5, count), 0.03, 0xffffff);
        }
        const truth = kit.curve('truth', 5, GOLD), bet = kit.curve('bet', 5, accent), landing = kit.marks('landing', 1);
        if (!cur.ok || cur.s < 1) return;
        this.outline(truth, plateX(cur.shown.bin, this.bins), 0.06);
        this.outline(bet, plateX(cur.shown.bet, this.bins), 0.13);
        landing.set(this.clampX(cur.shown.y * this.span), 0.1, -AXIS, 0.06, GOLD);
      },
    };
  }

  private outline(curve: { point(x: number, y: number, z: number): void }, x: number, pad: number): void {
    for (const [u, v] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] as const) curve.point(x + u * (PLATE_W / 2 + pad), 0.1, -(PLATE_Y + v * (PLATE_H / 2 + pad)));
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const lit = light.lit, n = this.inputs, bins = this.bins, accent = this.accentOf(ctx);
    this.accent.set(accent);
    this.palette = Array.from({ length: bins }, (_, b) => classColor(b, bins));
    this.scene.background = new THREE.Color(0x050403);
    pool.group.renderOrder = 10;
    const rowTexture = new THREE.DataTexture(this.rows, n + 1, 1, THREE.RGBAFormat, THREE.FloatType);
    rowTexture.needsUpdate = true;
    this.rowTexture = rowTexture;
    this.textures.push(rowTexture);
    const regions = uniformArray(Array.from({ length: bins }, () => new THREE.Vector4()), 'vec4');
    this.regionsU = regions;
    this.scene.add(backdrop(lit), board(lit, n, bins, { span: this.spanU, minor: this.minorU, major: this.majorU, regions }), gauges('x', rowTexture, n, this.scaleU, lit), gauges('w', rowTexture, n, this.scaleU, lit));
    for (let k = 0; k < 2; k++) {
      const points = uniformArray(Array.from({ length: 2 * n + 2 }, () => new THREE.Vector4()), 'vec4');
      this.paths.push(points);
      this.scene.add(path(points, 2 * n + 2, this.reveals[k]!, this.pathAlphas[k]!, n));
    }

    const geometry = flat(1, 1), look = new THREE.InstancedBufferAttribute(new Float32Array((WINDOW + 1) * 4), 4).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aBead', look);
    const beads = instancedMesh(geometry, beadMaterial(lit), WINDOW + 1);
    beads.count = 0;
    beads.frustumCulled = false;
    beads.renderOrder = 4;
    this.beads = beads;
    this.beadLook = look;

    const disc = (radius: number) => new THREE.CircleGeometry(radius, 24).rotateX(-Math.PI / 2);
    this.ballMesh = ball(this.ballAlpha);
    this.ballLamp = light.emitter(disc(0.07), this.ballLight);
    this.tokenMesh = token(rgb(accent), this.tokenAlpha);
    this.tokenLamp = light.emitter(disc(0.12), this.tokenLight);
    this.verdictMesh = verdict(this.verdictTint, this.verdictAlpha);
    this.stripMesh = strip(rgb(accent), this.stripAlpha);
    this.stripLamp = light.emitter(flat(1, 0.06), this.stripLight);
    this.landLamp = light.emitter(disc(0.1), this.landLight);
    const gallery = light.emitter(flat(12, 0.12), rgb(0xffe4c0).mul(2.4));
    gallery.position.set(0, 0.02, EXTENT.min.y - 0.3);
    for (const mesh of [this.ballMesh, this.ballLamp, this.tokenMesh, this.tokenLamp, this.verdictMesh, this.stripMesh, this.stripLamp, this.landLamp]) {
      mesh.visible = false;
      mesh.frustumCulled = false;
    }
    for (const lamp of [this.ballLamp, this.tokenLamp, this.stripLamp, this.landLamp, gallery]) lamp.renderOrder = 3;
    this.scene.add(beads, this.ballMesh, this.ballLamp, this.tokenMesh, this.tokenLamp, this.verdictMesh, this.stripMesh, this.stripLamp, this.landLamp, gallery);

    const size = Math.min(0.15, (0.6 * ROWS) / n);
    this.put('x · seen', 0.19, IVORY, XG, HEAD, 0, 0.85);
    this.put('w · hidden', 0.19, IVORY, WG, HEAD, 0, 0.85);
    this.put('y = w · x', 0.19, IVORY, -HALF + 0.05, HEAD, -1, 0.7);
    for (let j = 0; j < n; j++) this.put(`x${j + 1}`, size, INK, LABEL_X, rowY(j, n), 1);
    this.put('1', size, INK, LABEL_X, AXIS, 1);
    this.put('not in y', 0.12, INK, WG, AXIS, 0, 0.8);
    this.put('y', 0.15, IVORY, HALF + 0.2, AXIS + 0.02, 0, 0.7);
    const range = (b: number) => (b === 0 ? `y ≤ ${edgeText(this.edges[0]!)}` : b === bins - 1 ? `y > ${edgeText(this.edges[bins - 2]!)}` : `${edgeText(this.edges[b - 1]!)} < y ≤ ${edgeText(this.edges[b]!)}`);
    for (let b = 0; b < bins; b++) {
      const tint: Tint = uniform(new THREE.Vector3(this.palette[b]!.r, this.palette[b]!.g, this.palette[b]!.b));
      this.numberTints.push(tint);
      this.place(label(`${b}`, 0.3, tint, uniform(1), 700), plateX(b, bins), PLATE_Y + 0.1, 0);
      this.put(range(b), 0.13, INK, plateX(b, bins), PLATE_Y - 0.22, 0, 1);
    }

    const cur = () => this.cur, judged = () => this.cur.ok && this.cur.s >= 1;
    this.legend = {
      title: 'REGRESSION',
      rows: [
        { key: 'bet', label: 'bet', tint: accent, value: () => (judged() ? `bin ${cur().shown.bet} · ${range(cur().shown.bet)}` : '—') },
        { key: 'truth', label: 'truth', tint: GOLD, value: () => (judged() ? `bin ${cur().shown.bin} · y ${minus(formatSigned(cur().shown.y))}` : '—') },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => (judged() ? (cur().shown.hit ? '+1' : '0') : '—') },
        { key: 'rate', label: 'hit rate', tint: accent, value: () => (cur().judged > 0 ? `${formatPercent(cur().hits / cur().judged)} of ${formatCount(cur().judged)}` : '—') },
        { key: 'rule', label: 'rule', tint: IVORY, value: () => (cur().ok ? `|w| ${formatRate(cur().shown.norm)} · moved ${formatRate(cur().shown.moved)}` : '—') },
      ],
      note: () => `${n} inputs · weights drift ${Number(this.drift.toPrecision(3))} a step`,
      bar: () => (cur().judged > 0 ? cur().hits / cur().judged : 0),
    };
  }

  private put(text: string, height: number, tint: number, x: number, y: number, align: -1 | 0 | 1 = 0, alpha = 1): Label {
    return this.place(label(text, height, rgb(tint), uniform(alpha)), x, y, align);
  }

  private place(made: Label, x: number, y: number, align: -1 | 0 | 1): Label {
    made.mesh.position.set(x - (align * made.width) / 2, 0, -y);
    this.textures.push(made.texture);
    this.scene.add(made.mesh);
    return made;
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    this.textures.length = 0;
    this.ticks.clear();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private clampX(x: number): number {
    return Math.max(-HALF + 0.05, Math.min(HALF - 0.05, x));
  }

  private rangeFor(norm: number, fresh: boolean): void {
    const want = Math.max(SPREAD * norm, this.reach), up = ladder(want);
    const range = fresh || this.range === 0 || up > this.range || want < SHRINK * below(this.range) ? up : this.range;
    if (range === this.range) return;
    this.range = range;
    this.span = HALF / range;
    const bins = this.bins, regions = this.regionsU!.array as THREE.Vector4[], width = this.slotWidth(range);
    let first = 0;
    for (let b = 0; b < bins; b++) {
      const a = b === 0 ? -HALF : this.edges[b - 1]! * this.span, c = b === bins - 1 ? HALF : this.edges[b]! * this.span;
      const slots = Math.max(1, Math.round((c - a) / (width * this.span)));
      this.bounds[b * 2] = a;
      this.bounds[b * 2 + 1] = c;
      this.slotsIn[b] = slots;
      this.firstSlot[b] = first;
      first += slots;
      regions[b]!.set(a, c, slots, 0);
    }
    this.slotCount = first;
    this.bead = BEAD;
    for (let b = 0; b < bins; b++) this.bead = Math.min(this.bead, (0.9 * (this.bounds[b * 2 + 1]! - this.bounds[b * 2]!)) / this.slotsIn[b]!);
    if (first > this.column.length) this.column = new Int32Array(first);
    const major = majorFor(range);
    this.spanU.value = this.span;
    this.majorU.value = major;
    const mantissa = Math.round(major / 10 ** Math.floor(Math.log10(major)));
    this.minorU.value = major / (mantissa === 2 ? 4 : mantissa === 5 ? 5 : 2);
    this.rule(major);
    this.stackedFor = -1;
  }

  private slotWidth(range: number): number {
    const least = (PITCH * range) / HALF, inner = this.edges.slice(1).map((e, k) => e - this.edges[k]!);
    let best = least, score = Infinity;
    for (let k = 0; k <= 600; k++) {
      const width = least * (0.7 + k / 1000);
      let error = 0;
      for (const span of inner) error = Math.max(error, Math.abs(span / (width * Math.max(1, Math.round(span / width))) - 1));
      const cost = error + 0.12 * Math.abs(Math.log(width / least));
      if (cost < score) {
        score = cost;
        best = width;
      }
    }
    return best;
  }

  private rule(major: number): void {
    const top = Math.ceil(this.range / major - 1e-9) - 1, shown = new Set<string>();
    for (let v = -top; v <= top; v++) {
      if (v === 0) continue;
      const text = minus(String(Number((v * major).toPrecision(6))));
      shown.add(text);
      let made = this.ticks.get(text);
      if (!made) {
        made = this.put(text, 0.13, 0xc9b28c, 0, 0, 0, 0.9);
        this.ticks.set(text, made);
      }
      made.mesh.position.set(v * major * this.span, 0, -(AXIS + 0.25));
    }
    const zero = this.ticks.get('0') ?? this.put('0', 0.13, IVORY, 0, AXIS + 0.25);
    this.ticks.set('0', zero);
    shown.add('0');
    for (const [text, made] of this.ticks) made.mesh.visible = shown.has(text);
  }

  private slotOf(y: number, bin: number): number {
    const a = this.bounds[bin * 2]!, c = this.bounds[bin * 2 + 1]!, m = this.slotsIn[bin]!;
    return this.firstSlot[bin]! + Math.max(0, Math.min(m - 1, Math.floor(((y * this.span - a) / (c - a)) * m)));
  }

  private regionOf(slot: number): number {
    let b = 0;
    while (b + 1 < this.bins && this.firstSlot[b + 1]! <= slot) b++;
    return b;
  }

  private slotX(slot: number): number {
    const b = this.regionOf(slot), a = this.bounds[b * 2]!, c = this.bounds[b * 2 + 1]!;
    return a + ((slot - this.firstSlot[b]! + 0.5) * (c - a)) / this.slotsIn[b]!;
  }

  private pitchOf(count: number): number {
    return Math.min(this.bead, STACK / Math.max(count, 1));
  }

  private levelY(level: number, count: number): number {
    return FLOOR + 0.03 + (level + 0.5) * this.pitchOf(count);
  }

  private weightScale(display: Display): number {
    let top = 0;
    for (let j = 0; j < this.inputs; j++) top = Math.max(top, Math.abs(display.shown.w[j]!), Math.abs(this.origin?.[j] ?? 0));
    return 1.15 * Math.max(top, 1e-3);
  }

  private reader(frame: FrameUpdate): At {
    return (s) => {
      if (s < 0) return null;
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
  }

  private check(step: number, bin: number, bet: number, reward: number): void {
    if ((reward !== 0 && reward !== 1) || (bet === bin) !== (reward === 1)) throw new Error(`Regression step ${step} paid ${reward} for a bet on bin ${bet} though the target fell in bin ${bin}`);
  }

  private known(at: At, k: number): boolean {
    const slot = k % RING;
    if (this.tagOf[slot] === k) return true;
    const source = at(k), judge = at(k + 1);
    if (!source || !judge) return false;
    const y = regressionTarget(source), bin = regressionBin(y, this.edges);
    this.check(k + 1, bin, regressionGuess(judge, this.bins), judge.reward);
    this.yOf[slot] = y;
    this.binOf[slot] = bin;
    this.hitOf[slot] = judge.reward === 1 ? 1 : 0;
    this.tagOf[slot] = k;
    return true;
  }

  private sample(at: At, q: number, judged: boolean, out: Sample): boolean {
    const source = at(q), judge = judged ? at(q + 1) : null;
    if (!source || (judged && !judge)) return false;
    const n = this.inputs, origin = this.origin;
    let norm = 0, moved = 0;
    for (let j = 0; j < n; j++) {
      const w = wordFloat(source.words[j]!);
      out.x[j] = source.observations[j]!;
      out.w[j] = w;
      norm += w * w;
      if (origin) moved += (w - origin[j]!) ** 2;
    }
    out.bias = source.observations[n]!;
    out.y = regressionTarget(source, out.sums);
    out.bin = regressionBin(out.y, this.edges);
    out.norm = Math.sqrt(norm);
    out.moved = Math.sqrt(moved);
    out.bet = judge ? regressionGuess(judge, this.bins) : -1;
    out.hit = judge?.reward === 1;
    if (judge) this.check(q + 1, out.bin, out.bet, judge.reward);
    return true;
  }

  private fill(at: At, s: number, out: Display): void {
    out.s = s;
    out.ok = this.sample(at, Math.max(s - 1, 0), s >= 1, out.shown);
    let complete = out.ok, hits = 0, judged = 0;
    for (let k = Math.max(0, s - RECENT); k < s; k++) {
      if (!this.known(at, k)) {
        complete = false;
        continue;
      }
      hits += this.hitOf[k % RING]!;
      judged++;
    }
    out.hits = hits;
    out.judged = judged;
    out.complete = complete;
  }

  private stack(s: number): void {
    if (this.stackedFor === s && this.stackedRange === this.range) return;
    this.column.fill(0);
    this.winFirst = Math.max(0, s - WINDOW);
    this.winSize = 0;
    for (let k = this.winFirst; k < s; k++) {
      const ring = k % RING, e = k - this.winFirst;
      if (this.tagOf[ring] !== k) {
        this.winSlot[e] = -1;
        continue;
      }
      const slot = this.slotOf(this.yOf[ring]!, this.binOf[ring]!);
      this.winSlot[e] = slot;
      this.winLevel[e] = this.column[slot]!++;
      this.winHit[e] = this.hitOf[ring]!;
    }
    this.winSize = s - this.winFirst;
    this.stackedFor = s;
    this.stackedRange = this.range;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const rowTexture = this.rowTexture;
    if (!rowTexture || !this.beads) return;
    const dt = frame.dtReal;
    pool.update(dt, time);
    const jump = Math.abs(frame.position - this.lastPosition), seek = frame.scrubbed && jump > SWEEP;
    if (dt > 0 && !seek) this.pace += (jump / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const at = this.reader(frame), i = frame.state.step, fast = this.pace > FAST;
    if (!this.origin) {
      const first = at(0);
      if (first) this.origin = Float32Array.from({ length: this.inputs }, (_, j) => wordFloat(first.words[j]!));
    }
    const cur = this.cur, nxt = this.nxt, stepped = i !== cur.s;
    if (seek) {
      pool.clear();
      this.revealAt = -Infinity;
    }
    if (frame.scrubbed || stepped || !cur.complete) this.fill(at, i, cur);
    if (frame.scrubbed || stepped || !nxt.complete) this.fill(at, i + 1, nxt);
    if (!cur.ok) {
      this.hide();
      return;
    }
    this.rangeFor(cur.shown.norm, seek);
    if (stepped || !cur.complete) this.stackedFor = -1;
    this.stack(i);
    if (stepped && !frame.scrubbed && i >= 1) this.reveal(frame, pool, time, fast);

    const live = nxt.ok && !fast, t = live ? clamp01(frame.position - i) : 0, prep = live ? smoothstep01(phase(t, PREP)) : 0, to = live ? nxt : cur;
    this.scaleU.value = lerp(this.weightScale(cur), this.weightScale(to), prep);
    const n = this.inputs, rows = this.rows, from = cur.shown, next = to.shown;
    this.panel = prep < 0.5 ? from : next;
    for (let j = 0; j < n; j++) {
      rows[j * 4] = lerp(from.x[j]!, next.x[j]!, prep);
      rows[j * 4 + 1] = lerp(from.w[j]!, next.w[j]!, prep);
      rows[j * 4 + 2] = this.origin?.[j] ?? from.w[j]!;
    }
    rows[n * 4] = lerp(from.bias, next.bias, prep);
    rowTexture.needsUpdate = true;

    this.lay(this.paths[0]!, from);
    this.reveals[0]!.value = 2 * n + 1;
    this.pathAlphas[0]!.value = cur.s >= 1 ? (live ? 1 - prep : 1) : 0;
    this.flying = live && t >= FALL[0];
    this.flight(t, live);
    this.pile(frame.position, t, live);
    this.bet(t, live, time, fast);
  }

  private hide(): void {
    for (const mesh of [this.ballMesh, this.ballLamp, this.tokenMesh, this.tokenLamp, this.verdictMesh, this.stripMesh, this.stripLamp, this.landLamp]) if (mesh) mesh.visible = false;
    if (this.beads) this.beads.count = 0;
    for (const alpha of this.pathAlphas) alpha.value = 0;
    this.flying = false;
  }

  private reveal(frame: FrameUpdate, pool: CelebrationPool, time: number, fast: boolean): void {
    this.revealAt = time;
    const shown = this.cur.shown, slot = this.slotOf(shown.y, shown.bin), count = this.column[slot]!;
    this.landedAt.set(this.slotX(slot), this.levelY(count - 1, count));
    if (fast || !frame.events.some((event) => event.type === 'goal')) return;
    pool.burst({ x: this.landedAt.x, y: 38, z: -this.landedAt.y }, GOLD, 0.42, time);
    pool.ripple({ x: this.landedAt.x, y: 0, z: -this.landedAt.y }, GOLD, 0.16, time);
  }

  private vertices(sample: Sample, out: Float64Array | THREE.Vector4[]): void {
    const n = this.inputs, span = this.span;
    const set = (k: number, v: number, y: number) => {
      const x = this.clampX(v * span);
      if (out instanceof Float64Array) {
        out[k * 2] = x;
        out[k * 2 + 1] = y;
      } else out[k]!.set(x, y, 0, Math.abs(v * span) > HALF - 0.05 ? 1 : 0);
    };
    set(0, 0, MOUTH);
    for (let j = 0; j < n; j++) {
      set(2 * j + 1, sample.sums[j]!, rowY(j, n));
      set(2 * j + 2, sample.sums[j + 1]!, rowY(j, n));
    }
    set(2 * n + 1, sample.y, AXIS);
  }

  private lay(points: Points, sample: Sample): void {
    this.vertices(sample, points.array as THREE.Vector4[]);
  }

  private flight(t: number, live: boolean): void {
    const mesh = this.ballMesh!, lamp = this.ballLamp!, nxt = this.nxt, n = this.inputs;
    if (!live || !nxt.ok) {
      mesh.visible = lamp.visible = false;
      this.pathAlphas[1]!.value = 0;
      return;
    }
    const sample = nxt.shown, route = this.route, lengths = this.lengths;
    this.lay(this.paths[1]!, sample);
    this.vertices(sample, route);
    let total = 0;
    for (let k = 0; k < 2 * n + 1; k++) {
      lengths[k] = Math.hypot(route[k * 2 + 2]! - route[k * 2]!, route[k * 2 + 3]! - route[k * 2 + 1]!) + (k % 2) * PAUSE;
      total += lengths[k]!;
    }
    const slot = this.slotOf(sample.y, sample.bin), leaving = this.leavingSlot(), count = this.column[slot]! - (leaving === slot ? 1 : 0) + 1;
    this.spot.set(this.slotX(slot), this.levelY(count - 1, count));
    const pearl = this.at;
    let reveal = 0, alpha = 1;
    if (t < FALL[0]) {
      pearl.set(0, MOUTH);
      alpha = phase(t, [0, FALL[0]]);
    } else if (t < FALL[1]) {
      let left = smoothstep01(phase(t, FALL)) * total, k = 0;
      while (k < 2 * n && left > lengths[k]!) left -= lengths[k++]!;
      const wait = (k % 2) * PAUSE, run = lengths[k]! - wait, travel = run > 1e-6 ? clamp01((left - wait) / run) : 1;
      pearl.set(lerp(route[k * 2]!, route[k * 2 + 2]!, travel), lerp(route[k * 2 + 1]!, route[k * 2 + 3]!, travel));
      reveal = k + travel;
    } else {
      const k = phase(t, DROP) ** 2;
      pearl.set(lerp(route[(2 * n + 1) * 2]!, this.spot.x, k), lerp(AXIS, this.spot.y, k));
      reveal = 2 * n + 1;
    }
    this.reveals[1]!.value = reveal;
    this.pathAlphas[1]!.value = t < FALL[0] ? 0 : 1;
    mesh.visible = lamp.visible = true;
    mesh.position.set(pearl.x, 0, -pearl.y);
    mesh.scale.setScalar(lerp(HERO, this.bead / BEAD, t < DROP[0] ? 0 : smoothstep01(phase(t, DROP))));
    lamp.scale.copy(mesh.scale);
    lamp.position.set(pearl.x, 0.02, -pearl.y);
    this.ballAlpha.value = alpha;
    this.ballLight.value.set(1, 0.9, 0.72).multiplyScalar(9 * alpha);
  }

  private leavingSlot(): number {
    return this.cur.s - WINDOW >= 0 && this.winSize === WINDOW ? this.winSlot[0]! : -1;
  }

  private pile(position: number, t: number, live: boolean): void {
    const beads = this.beads!, look = this.beadLook!, drain = live && this.leavingSlot() >= 0 ? smoothstep01(phase(t, DRAIN)) : 0, leaving = this.leavingSlot();
    this.drainSlot = leaving;
    this.drainK = drain;
    let count = 0;
    for (let e = 0; e < this.winSize; e++) {
      const slot = this.winSlot[e]!;
      if (slot < 0) continue;
      const out = e === 0 && leaving >= 0, total = this.column[slot]!;
      const level = out ? -drain : this.winLevel[e]! - (slot === leaving ? drain : 0);
      const size = this.bead, y = this.levelY(level, total), age = position - (this.winFirst + e + 1);
      beads.setMatrixAt(count, this.matrix.makeScale(size, 1, size).setPosition(this.slotX(slot), 0, -y));
      look.setXYZW(count, this.winHit[e]!, out ? 1 - drain : 1, Math.max(0, 1 - age / GLOW_STEPS), 0);
      count++;
    }
    beads.count = count;
    beads.instanceMatrix.needsUpdate = true;
    look.needsUpdate = true;
    const lamp = this.landLamp!, flash = Math.exp(-Math.max(0, this.ctx!.time() - this.revealAt) / FLASH_SECONDS);
    lamp.visible = this.cur.s >= 1 && this.cur.shown.hit && flash > 0.02;
    lamp.position.set(this.landedAt.x, 0.02, -this.landedAt.y);
    this.landLight.value.set(1, 0.82, 0.4).multiplyScalar(3 * flash);
  }

  private bet(t: number, live: boolean, time: number, fast: boolean): void {
    const cur = this.cur, nxt = this.nxt, bins = this.bins, judged = cur.s >= 1;
    const before = judged ? cur.shown.bet : live ? nxt.shown.bet : -1, after = live ? nxt.shown.bet : before;
    const meshes = [this.tokenMesh!, this.tokenLamp!, this.stripMesh!, this.stripLamp!];
    if (before < 0 || after < 0) {
      for (const mesh of meshes) mesh.visible = false;
      this.verdictMesh!.visible = false;
      return;
    }
    const slide = smoothstep01(phase(t, SLIDE)), x = lerp(plateX(before, bins), plateX(after, bins), slide), shown = judged ? 1 : smoothstep01(phase(t, PREP));
    for (const mesh of meshes) mesh.visible = true;
    this.tokenMesh!.position.set(x, 0, -PLATE_Y);
    this.tokenLamp!.position.set(x + PLATE_W / 2 - 0.36, 0.02, -PLATE_Y);
    this.tokenAlpha.value = shown;
    this.tokenLight.value.set(this.accent.r, this.accent.g, this.accent.b).multiplyScalar(3 * shown);
    const region = slide < 0.5 ? before : after, a = this.bounds[region * 2]!, c = this.bounds[region * 2 + 1]!;
    for (const mesh of [this.stripMesh!, this.stripLamp!]) {
      mesh.position.set((a + c) / 2, 0.01, -(FLOOR + 0.035));
      mesh.scale.set(Math.max(c - a - 0.06, 0.02), 1, 1);
    }
    const settle = Math.abs(slide - 0.5) * 2;
    this.stripAlpha.value = 0.55 * shown * settle;
    this.stripLight.value.set(this.accent.r, this.accent.g, this.accent.b).multiplyScalar(1.4 * shown * settle);

    const flash = judged ? Math.max(REST, fast ? 0 : Math.exp(-Math.max(0, time - this.revealAt) / FLASH_SECONDS)) : 0;
    this.verdictMesh!.visible = flash > 0.01;
    this.verdictMesh!.position.set(plateX(cur.shown.bet, bins), 0, -PLATE_Y);
    this.tint.set(cur.shown.hit ? GOLD : MISS);
    this.verdictTint.value.set(this.tint.r, this.tint.g, this.tint.b);
    this.verdictAlpha.value = flash;
    for (let b = 0; b < bins; b++) {
      const glow = judged && b === cur.shown.bin ? 1 + 1.2 * flash : 1;
      this.numberTints[b]!.value.set(this.palette[b]!.r * glow, this.palette[b]!.g * glow, this.palette[b]!.b * glow);
    }
  }
}

export default (() => new RegressionScene()) satisfies SceneFactory;
