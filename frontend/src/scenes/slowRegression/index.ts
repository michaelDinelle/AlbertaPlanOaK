import * as THREE from 'three/webgpu';
import { float, mix, select, uniform } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { SUPERVISED_SHOWN, SUPERVISED_SOURCE, SUPERVISED_TARGET } from '@sim/envs/supervised';
import type { EnvFrame } from '@sim/types';
import { wordFloat } from '@sim/words';
import { parameter } from '../../data/envs';
import { formatCount, formatRate, formatSigned } from '../../format';
import {
  BANDS, Digits, ERROR, GOLD, INK, MINUS, PLUS,
  angleOf, bench, biasAt, biasLed, board, cover, dips, disc, emitters, label, layout, matrix, meter, rowY, sevenSeg, traces, unitX, unitY,
  type Layout, type Level, type Lit, type Needle,
} from './art';

type At = (s: number) => EnvFrame | null;

interface Example {
  k: number;
  bits: number;
  lastY: number;
  fired: Uint8Array;
  plus: Uint8Array;
  bias: number;
  up: number;
  down: number;
  target: number;
  prediction: number;
  reward: number;
  flip: number;
}

interface Flip { f: number; bit: number; on: boolean }

interface Readout { offset: number; cells: number; text: string }

const FRESH = 64;
const FAST = 5;
const SCAN = 12288;
const BUDGET = 4096;
const RETRY = 0.5;
const IN = [0, 0.22] as const;
const NEEDLE = [0.16, 0.56] as const;
const REVEAL = [0.5, 0.74] as const;
const LED_POWER = 7;

const phase = (t: number, [a, b]: readonly [number, number]) => clamp01((t - a) / (b - a));
const ease = (t: number, span: readonly [number, number]) => smoothstep01(phase(t, span));
const settle = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : 1 - Math.exp(-5 * t) * Math.cos(8.5 * t) * (1 - t));
const signedInt = (v: number) => `${v > 0 ? '+' : v < 0 ? '-' : ''}${Math.abs(Math.round(v))}`;
const signedTenths = (v: number) => (Math.abs(v) < 0.05 ? '0.0' : `${v > 0 ? '+' : '-'}${Math.abs(v).toFixed(1)}`);
const plain = (v: number) => (Number.isFinite(v) ? formatCount(v) : '—');
const level = (value = 0): Level => uniform(value);

class SlowRegressionScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'slowRegression';
  override legend: LegendSpec | null = null;

  private l: Layout | null = null;
  private fields = 0;
  private slowMask = 0;
  private period = 1;
  private accent = 0xffffff;
  private bitsData = new Float32Array(4);
  private unitsData = new Float32Array(4);
  private bandsData = new Float32Array(4);
  private bitsTexture: THREE.DataTexture | null = null;
  private unitsTexture: THREE.DataTexture | null = null;
  private bandsTexture: THREE.DataTexture | null = null;
  private readonly digits = new Digits(40);
  private readonly readouts: Record<'lastY' | 'up' | 'down' | 'bias' | 'sum' | 'prediction' | 'target' | 'error', Readout> = {
    lastY: { offset: 0, cells: 4, text: '' }, up: { offset: 0, cells: 3, text: '' }, down: { offset: 0, cells: 3, text: '' }, bias: { offset: 0, cells: 2, text: '' },
    sum: { offset: 0, cells: 4, text: '' }, prediction: { offset: 0, cells: 4, text: '' }, target: { offset: 0, cells: 4, text: '' }, error: { offset: 0, cells: 5, text: '' },
  };
  private readonly textures: THREE.Texture[] = [];
  private readonly needle: Needle = { value: level(), target: level(), on: level(), over: level() };
  private readonly biasPlus = level(1);
  private readonly flare = level();
  private readonly tipGlow = level();
  private flareLamp: THREE.Mesh | null = null;
  private tipLamp: THREE.Mesh | null = null;
  private goalLamp: THREE.Mesh | null = null;
  private readonly flag = level();
  private readonly flagRow = level(-1);
  private readonly tags = new Map<string, THREE.Mesh>();
  private tag: THREE.Mesh | null = null;

  private cur: Example = this.blank(0);
  private nxt: Example = this.blank(0);
  private scratch: Example = this.blank(0);
  private probe: Example = this.blank(0);
  private curOk = false;
  private nxtOk = false;
  private shownStep = -1;
  private pace = 0;
  private lastPosition = 0;
  private sparked = -1;
  private hit = -1;
  private clock = 0;

  private ringK = new Int32Array(BANDS).fill(-1);
  private ringFired = new Uint8Array(0);
  private readonly ringTarget = new Int16Array(BANDS);
  private readonly ringPredict = new Int16Array(BANDS);
  private readonly ringError = new Float64Array(BANDS);
  private fireCount = new Uint16Array(0);
  private targetCount = new Uint16Array(0);
  private predictCount = new Uint16Array(0);
  private held = 0;
  private errorSum = 0;
  private ringTo = -1;
  private ringComplete = false;
  private ringAt = -Infinity;
  private bandsDirty = true;

  private scanFrom = 1;
  private scanTo = 0;
  private readonly flips: Flip[] = [];
  private scanAt = -Infinity;

  private readonly ledger = { k: -1, prediction: NaN, target: NaN, reward: NaN, up: 0, down: 0, flip: null as Flip | null, age: 0, reach: 0, since: false, mean: NaN, held: 0 };

  private blank(hidden: number): Example {
    return { k: -1, bits: 0, lastY: 0, fired: new Uint8Array(hidden), plus: new Uint8Array(hidden), bias: 1, up: 0, down: 0, target: 0, prediction: NaN, reward: 0, flip: -1 };
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, inputs = parameter(env, 'inputs'), slow = parameter(env, 'flipping'), hidden = parameter(env, 'hidden'), scale = parameter(env, 'scale');
    if (env.obs !== inputs + 1 || env.words !== 3 || env.actions !== 1) throw new Error(`${env.envId} shows ${env.obs} observations, keeps ${env.words} words and takes ${env.actions} actions`);
    if (!(slow > 0 && slow < inputs && inputs <= 31)) throw new Error(`${env.envId} flips ${slow} of ${inputs} inputs`);
    if (env.view < 2 || hidden < 1 || hidden > 16 * (env.view - 1)) throw new Error(`${env.envId} packs ${hidden} threshold units into ${env.view} view words`);
    if (!(scale > 0 && Number.isInteger(scale))) throw new Error(`${env.envId} scales predictions by ${scale}`);
    const l = layout(inputs, slow, hidden, scale);
    this.l = l;
    this.fields = env.view - 1;
    this.slowMask = 2 ** slow - 1;
    this.period = parameter(env, 'period');
    this.cur = this.blank(hidden);
    this.nxt = this.blank(hidden);
    this.scratch = this.blank(hidden);
    this.probe = this.blank(hidden);
    this.bitsData = new Float32Array(inputs * 4);
    this.unitsData = new Float32Array(hidden * 4);
    this.bandsData = new Float32Array((2 * scale + 1) * 4);
    this.ringFired = new Uint8Array(BANDS * hidden);
    this.fireCount = new Uint16Array(hidden);
    this.targetCount = new Uint16Array(2 * scale + 1);
    this.predictCount = new Uint16Array(2 * scale + 1);
    const b = l.board;
    return new THREE.Box2(new THREE.Vector2(b.x0 - 0.1, b.y0 - 0.6), new THREE.Vector2(b.x1 + 0.1, b.y1 + 0.1));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xa4b4c2).multiplyScalar(0.8);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const accent = this.accentOf(ctx);
    return {
      debug: (frame, kit) => {
        const l = this.l, e = this.probe;
        if (!l || !this.read(this.reader(frame), frame.state.step, e)) return;
        const bits = kit.marks('bits', l.inputs);
        for (let j = 0; j < l.inputs; j++) {
          const on = (e.bits >>> j) & 1;
          bits.set(l.slotX - l.travel / 2 + on * l.travel, 0.1, -rowY(l, j), 0.05, on ? 0xffffff : 0x4a4f55);
        }
        const fired = kit.marks('fired', l.hidden);
        for (let j = 0; j < l.hidden; j++) if (e.fired[j]) fired.set(unitX(l, j), 0.1, -unitY(l, j), 0.1, e.plus[j] ? PLUS : MINUS);
        const [bx, by] = biasAt(l);
        kit.marks('bias', 1).set(bx, 0.1, -by, 0.08, e.bias > 0 ? PLUS : MINUS);
        const m = l.meter, goal = angleOf(l, e.target);
        kit.marks('target', 1).set(m.cx + (m.radius + 0.3) * Math.sin(goal), 0.1, -(m.cy + (m.radius + 0.3) * Math.cos(goal)), 0.09, GOLD);
        if (Number.isFinite(e.prediction)) {
          const aim = angleOf(l, e.prediction), line = kit.curve('prediction', 2, accent);
          line.point(m.cx, 0.1, -m.cy);
          line.point(m.cx + (m.radius + 0.15) * Math.sin(aim), 0.1, -(m.cy + (m.radius + 0.15) * Math.cos(aim)));
        }
        if (e.flip >= 0) kit.marks('flip', 1).set(l.dip[0] - 0.2, 0.1, -rowY(l, e.flip), 0.07, GOLD);
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const l = this.l!, lit = light.lit, accent = this.accentOf(ctx);
    this.accent = accent;
    this.scene.background = new THREE.Color(0x030405);
    pool.group.renderOrder = 11;
    const bitsTexture = new THREE.DataTexture(this.bitsData, l.inputs, 1, THREE.RGBAFormat, THREE.FloatType);
    const unitsTexture = new THREE.DataTexture(this.unitsData, l.hidden, 1, THREE.RGBAFormat, THREE.FloatType);
    const bandsTexture = new THREE.DataTexture(this.bandsData, 2 * l.scale + 1, 1, THREE.RGBAFormat, THREE.FloatType);
    for (const texture of [bitsTexture, unitsTexture, bandsTexture]) texture.needsUpdate = true;
    this.bitsTexture = bitsTexture;
    this.unitsTexture = unitsTexture;
    this.bandsTexture = bandsTexture;
    this.textures.push(bitsTexture, unitsTexture, bandsTexture, this.digits.texture);

    this.scene.add(bench(), board(l, lit), traces(l, bitsTexture, lit), cover(l, lit), dips(l, bitsTexture, lit), matrix(l, unitsTexture, lit), biasLed(l, this.biasPlus, lit), meter(l, this.needle, bandsTexture, accent, lit));
    const leds = emitters(l, unitsTexture);
    this.place(light.emitter(leds.geometry, leds.tint.mul(LED_POWER)), 0, 0, 2);
    const [bx, by] = biasAt(l);
    this.place(light.emitter(disc(0.1), mix(rgb(MINUS), rgb(PLUS), this.biasPlus).mul(LED_POWER * 0.7)), bx, by, 2);
    this.flareLamp = this.place(light.emitter(disc(0.12), rgb(GOLD).mul(this.flare.mul(14))), l.slotX, rowY(l, 0), 2);
    this.tipLamp = this.place(light.emitter(disc(0.05), rgb(accent).mul(this.tipGlow.mul(4))), 0, 0, 2);
    this.goalLamp = this.place(light.emitter(disc(0.06), rgb(GOLD).mul(3.5)), 0, 0, 2);

    const s = l.sum, sx = (s.x0 + s.x1) / 2, r = this.readouts, m = l.meter, o = l.readout;
    const show = (key: keyof typeof r, x: number, y: number, height: number, tint: THREE.Node<'vec3'>) => {
      const readout = r[key];
      readout.offset = this.digits.claim(readout.cells);
      this.scene.add(sevenSeg(x, y, readout.cells, height, this.digits, readout.offset, tint, 6));
    };
    show('lastY', l.slotX, rowY(l, l.inputs), 0.2, rgb(GOLD).mul(0.85));
    show('up', sx + 0.25, 1.62, 0.4, rgb(PLUS));
    show('down', sx + 0.25, 0.72, 0.4, rgb(MINUS));
    show('bias', sx + 0.39, -0.05, 0.4, mix(rgb(MINUS), rgb(PLUS), this.biasPlus));
    show('sum', sx + 0.18, -1.55, 0.5, rgb(GOLD));
    const third = (o.x1 - o.x0) / 3;
    show('prediction', o.x0 + third / 2, 2.5, 0.46, rgb(accent));
    show('target', o.x0 + third * 1.5, 2.5, 0.46, rgb(GOLD));
    show('error', o.x0 + third * 2.5, 2.5, 0.46, rgb(ERROR));

    const put = (text: string, height: number, tint: number | THREE.Node<'vec3'>, x: number, y: number, align: -1 | 0 | 1, shade: Lit | null = lit, fade?: Level, turn = 0): THREE.Mesh => {
      const made = label(text, height, tint, shade ?? undefined, fade);
      const shift = align === -1 ? made.width / 2 : align === 1 ? -made.width / 2 : 0;
      made.mesh.position.set(turn ? x : x + shift, 0, turn ? -(y + shift) : -y);
      made.mesh.rotation.y = turn;
      this.textures.push(made.texture);
      this.scene.add(made.mesh);
      return made.mesh;
    };
    const head = l.top + 0.52, mx = (l.matrix.x0 + l.matrix.x1) / 2, period = formatCount(this.period);
    put('SEEN BY THE AGENT', 0.13, INK, l.slotX - 0.1, head, 0);
    put(`TARGET NETWORK · ${l.hidden} THRESHOLD UNITS · HIDDEN FROM THE AGENT`, 0.13, INK, mx, head, 0);
    put('OUTPUT', 0.13, INK, sx, head, 0);
    put('PREDICTION AGAINST TARGET', 0.13, INK, m.cx, head, 0);
    put('SLOW · ONE FLIPS EVERY ' + period + ' STEPS', 0.105, INK, l.labelX - 0.55, (rowY(l, 0) + rowY(l, l.slow - 1)) / 2, 0, lit, undefined, Math.PI / 2);
    put('FAST · EVERY STEP', 0.105, INK, l.labelX - 0.55, (rowY(l, l.slow) + rowY(l, l.inputs - 1)) / 2, 0, lit, undefined, Math.PI / 2);
    for (let j = 0; j < l.inputs; j++) put(`x${j + 1}`, 0.12, mix(rgb(INK), rgb(GOLD).mul(1.5), select(this.flagRow.equal(j), this.flag, float(0))), l.labelX, rowY(l, j), 1);
    put('last y', 0.12, GOLD, l.labelX, rowY(l, l.inputs), 1);
    put('0', 0.1, INK, l.slotX - l.travel / 2, l.top + 0.12, 0);
    put('1', 0.1, INK, l.slotX + l.travel / 2, l.top + 0.12, 0);

    put('Σ', 0.42, INK, sx, 2.62, 0);
    put('+1 UNITS FIRING', 0.1, PLUS, sx, 2.1, 0);
    put('−1 UNITS FIRING', 0.1, MINUS, sx, 1.2, 0);
    put('BIAS', 0.1, INK, sx, 0.34, 0);
    put('TARGET y = SUM', 0.1, GOLD, sx, -0.98, 0);
    put('+', 0.3, PLUS, s.x0 + 0.3, 1.62, 0);
    put('−', 0.3, MINUS, s.x0 + 0.3, 0.72, 0);
    put('=', 0.3, GOLD, s.x0 + 0.24, -1.55, 0);

    const lx = o.x0 + third / 2;
    put('PREDICTION ŷ', 0.11, accent, lx, 3.0, 0);
    put('TARGET y', 0.11, GOLD, lx + third, 3.0, 0);
    put('SQUARED ERROR = −REWARD', 0.11, ERROR, lx + 2 * third, 3.0, 0);
    for (let v = -l.scale; v <= l.scale; v += l.scale / 4) {
      const a = angleOf(l, v), reach = m.radius + 0.55;
      put(v === 0 ? '0' : `${v < 0 ? '−' : ''}${Math.abs(v)}`, 0.14, 0xe6e1d3, m.cx + reach * Math.sin(a), m.cy + reach * Math.cos(a), 0);
    }
    put('needle · agent’s prediction', 0.1, accent, m.x0 + 0.25, m.y0 + 0.3, -1);
    put('pointer · target', 0.1, GOLD, m.x1 - 0.25, m.y0 + 0.3, 1);
    put(`bands · last ${BANDS} predictions (outer) and targets (inner)`, 0.095, INK, m.cx, m.y1 - 0.24, 0);
    const key = [['amber · +1 output weight', PLUS], ['violet · −1 output weight', MINUS], ['bright · firing now', INK], [`ember · how often it fired, last ${BANDS} steps`, INK]] as const;
    const keys = key.map(([text, tint]) => label(text, 0.095, tint, lit)), gap = 0.32, span = keys.reduce((sum, made) => sum + made.width, 0) + gap * (keys.length - 1);
    let cursor = mx - span / 2;
    for (const made of keys) {
      made.mesh.position.set(cursor + made.width / 2, 0, -(l.top + 0.29));
      cursor += made.width + gap;
      this.textures.push(made.texture);
      this.scene.add(made.mesh);
    }

    this.legend = {
      title: 'SLOW REGRESSION',
      rows: [
        { key: 'prediction', label: 'prediction ŷ', tint: accent, value: () => (Number.isFinite(this.ledger.prediction) ? formatRate(this.ledger.prediction) : '—') },
        { key: 'target', label: 'target y', tint: GOLD, value: () => plain(this.ledger.target) },
        { key: 'reward', label: 'reward', tint: ERROR, value: () => (Number.isFinite(this.ledger.reward) ? formatSigned(this.ledger.reward).replace('-', '−') : '—') },
        { key: 'firing', label: 'units firing', tint: PLUS, value: () => `${this.ledger.up + this.ledger.down} of ${l.hidden}` },
        { key: 'flip', label: 'last slow flip', tint: INK, value: () => this.flipText() },
      ],
      note: () => (Number.isFinite(this.ledger.mean) ? `avg error² ${formatRate(this.ledger.mean)}` : ''),
      bar: () => (this.ledger.flip ? this.ledger.age / this.period : 0),
    };
  }

  private flipText(): string {
    const g = this.ledger;
    if (g.flip) return `x${g.flip.bit + 1} → ${g.flip.on ? '1' : '0'} · ${g.age === 0 ? 'now' : `${formatCount(g.age)} steps ago`}`;
    return g.since ? 'none yet' : `none in the last ${formatCount(g.reach)} steps`;
  }

  private place(mesh: THREE.Mesh, x: number, y: number, order: number): THREE.Mesh {
    mesh.position.set(x, 0, -y);
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    return mesh;
  }

  protected override drop(): void {
    for (const texture of this.textures) texture.dispose();
    this.textures.length = 0;
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private reader(frame: FrameUpdate): At {
    return (s) => {
      if (s < 0) return null;
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
  }

  private read(at: At, k: number, out: Example): boolean {
    const l = this.l!, source = at(Math.max(k - 1, 0)), prior = at(Math.max(k - 2, 0)), scored = k >= 1 ? at(k) : null;
    if (!source || !prior || (k >= 1 && !scored)) return false;
    const bits = source.words[SUPERVISED_SOURCE]!, view = source.view, field = (f: number) => (view[f >> 1]! >>> ((f & 1) << 4)) & 0xffff;
    if (bits >>> l.inputs) throw new Error(`SlowRegression step ${k - 1} holds input bits ${bits.toString(2)} beyond its ${l.inputs} inputs`);
    for (let i = 0; i < l.inputs; i++) if (source.observations[i] !== ((bits >>> i) & 1)) throw new Error(`SlowRegression step ${k - 1} shows input ${i + 1} as ${source.observations[i]} though its bits say ${(bits >>> i) & 1}`);
    out.k = k;
    out.bits = bits;
    out.lastY = source.observations[l.inputs]!;
    out.bias = field(2 * this.fields) & 1 ? 1 : -1;
    out.up = 0;
    out.down = 0;
    for (let j = 0; j < l.hidden; j++) {
      const fired = (field(j >> 4) >>> (j & 15)) & 1, plus = (field(this.fields + (j >> 4)) >>> (j & 15)) & 1;
      out.fired[j] = fired;
      out.plus[j] = plus;
      if (fired) plus ? out.up++ : out.down++;
    }
    out.target = wordFloat(source.words[SUPERVISED_TARGET]!);
    if (out.target !== out.bias + out.up - out.down) throw new Error(`SlowRegression step ${k - 1} holds target ${out.target} though bias ${out.bias} with ${out.up} +1 and ${out.down} −1 units firing sums to ${out.bias + out.up - out.down}`);
    out.prediction = NaN;
    out.reward = 0;
    if (scored) {
      if (scored.words[SUPERVISED_SHOWN] !== source.words[SUPERVISED_TARGET]) throw new Error(`SlowRegression step ${k} scored against ${wordFloat(scored.words[SUPERVISED_SHOWN]!)} though step ${k - 1} set the target to ${out.target}`);
      out.prediction = l.scale * Math.max(-1, Math.min(1, scored.actions[0]!));
      out.reward = scored.reward;
      const miss = out.prediction - out.target;
      if (Math.abs(out.reward + miss * miss) > 1e-3 * (1 + miss * miss)) throw new Error(`SlowRegression step ${k} paid ${out.reward} for predicting ${out.prediction} against ${out.target}`);
    }
    const changed = k >= 2 ? (bits ^ prior.words[SUPERVISED_SOURCE]!) & this.slowMask : 0;
    if (changed & (changed - 1)) throw new Error(`SlowRegression step ${k - 1} flipped several slow bits at once`);
    out.flip = changed ? 31 - Math.clz32(changed) : -1;
    return true;
  }

  private bin(value: number): number {
    const S = this.l!.scale;
    return Math.max(-S, Math.min(S, Math.round(value))) + S;
  }

  private forget(slot: number): void {
    const k = this.ringK[slot]!, hidden = this.l!.hidden;
    if (k < 0) return;
    for (let j = 0; j < hidden; j++) this.fireCount[j]! -= this.ringFired[slot * hidden + j]!;
    this.targetCount[this.ringTarget[slot]!]!--;
    this.predictCount[this.ringPredict[slot]!]!--;
    this.errorSum -= this.ringError[slot]!;
    this.ringK[slot] = -1;
    this.held--;
  }

  private remember(e: Example): void {
    const slot = e.k % BANDS, hidden = this.l!.hidden, miss = e.prediction - e.target;
    this.forget(slot);
    for (let j = 0; j < hidden; j++) {
      this.ringFired[slot * hidden + j] = e.fired[j]!;
      this.fireCount[j]! += e.fired[j]!;
    }
    this.ringTarget[slot] = this.bin(e.target);
    this.ringPredict[slot] = this.bin(e.prediction);
    this.ringError[slot] = miss * miss;
    this.targetCount[this.ringTarget[slot]!]!++;
    this.predictCount[this.ringPredict[slot]!]!++;
    this.errorSum += miss * miss;
    this.ringK[slot] = e.k;
    this.held++;
  }

  private gather(at: At, i: number, rebuild: boolean): void {
    let from = this.ringTo + 1;
    if (rebuild || i < this.ringTo || i - this.ringTo > BANDS) {
      this.ringK.fill(-1);
      this.fireCount.fill(0);
      this.targetCount.fill(0);
      this.predictCount.fill(0);
      this.held = 0;
      this.errorSum = 0;
      this.ringComplete = true;
      from = Math.max(1, i - BANDS + 1);
    }
    for (let k = from; k <= i; k++) {
      if (k < 1) continue;
      if (this.read(at, k, this.scratch)) this.remember(this.scratch);
      else {
        this.forget(k % BANDS);
        this.ringComplete = false;
      }
    }
    this.ringTo = i;
    this.bandsDirty = true;
  }

  private check(f: number, a: EnvFrame, b: EnvFrame): void {
    const changed = (a.words[SUPERVISED_SOURCE]! ^ b.words[SUPERVISED_SOURCE]!) & this.slowMask;
    if (!changed) return;
    if (changed & (changed - 1)) throw new Error(`SlowRegression step ${f} flipped several slow bits at once`);
    const bit = 31 - Math.clz32(changed), flip = { f, bit, on: ((a.words[SUPERVISED_SOURCE]! >>> bit) & 1) === 1 };
    const at = this.flips.findIndex((other) => other.f > f);
    if (this.flips.some((other) => other.f === f)) return;
    if (at < 0) this.flips.push(flip);
    else this.flips.splice(at, 0, flip);
  }

  private lastFlip(s: number): Flip | null {
    for (let n = this.flips.length - 1; n >= 0; n--) if (this.flips[n]!.f <= s) return this.flips[n]!;
    return null;
  }

  private scan(at: At, s: number): void {
    if (s < this.scanFrom - 1 || s > this.scanTo + BANDS) {
      this.scanFrom = s + 1;
      this.scanTo = s;
      this.flips.length = 0;
    }
    while (this.scanTo < s) {
      const f = this.scanTo + 1, a = at(f), b = at(f - 1);
      if (!a || !b) break;
      this.check(f, a, b);
      this.scanTo = f;
    }
    for (let work = 0; work < BUDGET && this.scanFrom > 1 && s - this.scanFrom < SCAN && !this.lastFlip(s); work++) {
      const f = this.scanFrom - 1, a = at(f), b = at(f - 1);
      if (!a || !b) break;
      this.check(f, a, b);
      this.scanFrom = f;
    }
  }

  private refresh(at: At, i: number, rebuild: boolean): void {
    this.curOk = this.read(at, i, this.cur);
    this.nxtOk = this.curOk && this.read(at, i + 1, this.nxt);
    this.gather(at, i, rebuild);
    this.ringAt = this.clock;
    this.scan(at, Math.max(0, i - 1));
    this.scanAt = this.clock;
    this.shownStep = i;
    const g = this.ledger, c = this.cur;
    g.k = i;
    g.prediction = this.curOk ? c.prediction : NaN;
    g.target = this.curOk ? c.target : NaN;
    g.reward = this.curOk && i >= 1 ? c.reward : NaN;
    g.up = c.up;
    g.down = c.down;
    this.note(i);
    g.held = this.held;
    g.mean = this.held > 0 ? this.errorSum / this.held : NaN;
  }

  private note(i: number): void {
    const g = this.ledger, s = Math.max(0, i - 1), flip = this.lastFlip(s);
    g.flip = flip;
    g.age = flip ? s - flip.f : 0;
    g.since = !flip && this.scanFrom <= 1;
    g.reach = Math.max(0, s - this.scanFrom + 1);
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const l = this.l;
    if (!l) return;
    const dt = frame.dtReal;
    this.clock += dt;
    pool.update(dt, time);
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const at = this.reader(frame), i = frame.state.step, fast = this.pace > this.controlHz * FAST;
    const stale = !this.curOk || (!this.nxtOk && at(i + 1) !== null);
    if (frame.scrubbed || i !== this.shownStep || stale) this.refresh(at, i, frame.scrubbed);
    else {
      if (!this.ringComplete && this.clock - this.ringAt > RETRY) { this.gather(at, i, true); this.ringAt = this.clock; }
      if (!this.ledger.flip && !this.ledger.since && this.clock - this.scanAt > RETRY) { this.scan(at, Math.max(0, i - 1)); this.note(i); this.scanAt = this.clock; }
    }
    if (!this.curOk) return;
    if (frame.scrubbed) {
      this.sparked = i;
      this.hit = i;
      pool.clear();
    }
    const cur = this.cur, live = this.nxtOk && !fast, nxt = live ? this.nxt : cur, t = live ? clamp01(frame.position - i) : 0;
    this.drawInputs(l, cur, nxt, t, live);
    this.drawUnits(l, cur, nxt, t);
    this.drawOutput(l, cur, nxt, t);
    if (this.bandsDirty) this.drawBands(l);
    this.celebrate(l, cur, nxt, t, live, pool, time);
  }

  private drawInputs(l: Layout, cur: Example, nxt: Example, t: number, live: boolean): void {
    const move = ease(t, IN), data = this.bitsData;
    let flipBit = -1, flipLevel = 0;
    if (live && nxt.flip >= 0) {
      flipBit = nxt.flip;
      flipLevel = move;
    } else {
      const flip = this.ledger.flip, age = this.ledger.age + t;
      if (flip && age < FRESH) {
        flipBit = flip.bit;
        flipLevel = 1 - age / FRESH;
      }
    }
    for (let j = 0; j < l.inputs; j++) {
      const a = (cur.bits >>> j) & 1, b = (nxt.bits >>> j) & 1, o = j * 4;
      data[o] = a + (b - a) * move;
      data[o + 1] = a || !b ? 1 : move;
      data[o + 2] = a && !b ? 1 - move : a || b ? 1 : 0;
      data[o + 3] = j === flipBit ? flipLevel : 0;
    }
    this.bitsTexture!.needsUpdate = true;
    this.flare.value = flipLevel * flipLevel;
    this.flag.value = flipLevel;
    this.flagRow.value = flipBit;
    if (flipBit >= 0) this.flareLamp?.position.set(l.slotX, 0, -rowY(l, flipBit));
    this.showTag(l, flipBit, flipBit < 0 ? false : ((nxt.bits >>> flipBit) & 1) === 1);
    this.write('lastY', signedInt(t < 0.11 ? cur.lastY : nxt.lastY));
  }

  private drawUnits(l: Layout, cur: Example, nxt: Example, t: number): void {
    const k = ease(t, REVEAL), data = this.unitsData, held = Math.max(1, this.held);
    for (let j = 0; j < l.hidden; j++) {
      const o = j * 4;
      data[o] = cur.fired[j]! + (nxt.fired[j]! - cur.fired[j]!) * k;
      data[o + 1] = cur.plus[j]!;
      data[o + 2] = Math.min(0.85, Math.sqrt(this.fireCount[j]! / held) * 0.95);
      data[o + 3] = 1;
    }
    this.unitsTexture!.needsUpdate = true;
    this.biasPlus.value = cur.bias > 0 ? 1 : 0;
    const e = t < 0.62 ? cur : nxt;
    this.write('up', String(e.up));
    this.write('down', String(e.down));
    this.write('bias', signedInt(e.bias));
    this.write('sum', signedInt(e.target));
    this.write('target', signedInt(e.target));
  }

  private drawOutput(l: Layout, cur: Example, nxt: Example, t: number): void {
    const swing = settle(phase(t, NEEDLE)), reveal = ease(t, REVEAL), m = l.meter;
    const had = Number.isFinite(cur.prediction), has = Number.isFinite(nxt.prediction);
    const value = had ? (has ? cur.prediction + (nxt.prediction - cur.prediction) * swing : cur.prediction) : has ? nxt.prediction : 0;
    const on = had ? 1 : has ? smoothstep01(phase(t, NEEDLE)) : 0;
    const target = cur.target + (nxt.target - cur.target) * reveal;
    this.needle.value.value = value;
    this.needle.on.value = on;
    this.needle.target.value = target;
    this.needle.over.value = Math.abs(target) > l.scale ? 1 : 0;
    this.tipGlow.value = on;
    const aim = angleOf(l, value), goal = angleOf(l, target), tip = m.radius + 0.05, flag = m.radius + 0.36;
    this.tipLamp?.position.set(m.cx + tip * Math.sin(aim), 0, -(m.cy + tip * Math.cos(aim)));
    this.goalLamp?.position.set(m.cx + flag * Math.sin(goal), 0, -(m.cy + flag * Math.cos(goal)));
    const shown = t < 0.36 ? cur : nxt, scored = t < REVEAL[1] ? cur : nxt;
    this.write('prediction', Number.isFinite(shown.prediction) ? signedTenths(shown.prediction) : '');
    const miss = scored.prediction - scored.target;
    this.write('error', Number.isFinite(miss) ? String(Math.round(miss * miss)) : '');
  }

  private drawBands(l: Layout): void {
    const data = this.bandsData, n = 2 * l.scale + 1;
    let topTarget = 1, topPredict = 1;
    for (let b = 0; b < n; b++) {
      topTarget = Math.max(topTarget, this.targetCount[b]!);
      topPredict = Math.max(topPredict, this.predictCount[b]!);
    }
    for (let b = 0; b < n; b++) {
      data[b * 4] = this.targetCount[b]! / topTarget;
      data[b * 4 + 1] = this.predictCount[b]! / topPredict;
    }
    this.bandsTexture!.needsUpdate = true;
    this.bandsDirty = false;
  }

  private celebrate(l: Layout, cur: Example, nxt: Example, t: number, live: boolean, pool: Stage2D['pool'], time: number): void {
    const flipping = live && nxt.flip >= 0 && t > 0 ? nxt : cur.flip >= 0 ? cur : null;
    if (flipping && flipping.k > this.sparked) {
      this.sparked = flipping.k;
      const y = rowY(l, flipping.flip);
      pool.burst({ x: l.slotX, y: 38, z: -y }, GOLD, 0.35, time);
      pool.ripple({ x: l.slotX, y: 0, z: -y }, GOLD, 0.16, time);
    }
    const judged = live && t >= REVEAL[1] ? nxt : cur;
    if (judged.k > this.hit && Number.isFinite(judged.prediction)) {
      this.hit = judged.k;
      if (Math.abs(judged.prediction - judged.target) < 0.5) {
        const m = l.meter, a = angleOf(l, judged.prediction), reach = m.radius + 0.05;
        pool.burst({ x: m.cx + reach * Math.sin(a), y: 38, z: -(m.cy + reach * Math.cos(a)) }, GOLD, 0.3, time);
      }
    }
  }

  private showTag(l: Layout, bit: number, on: boolean): void {
    const text = bit < 0 ? '' : `x${bit + 1} FLIPPED TO ${on ? 1 : 0}`;
    let next = this.tags.get(text) ?? null;
    if (!next && text) {
      const made = label(text, 0.1, rgb(GOLD).mul(1.3), undefined, this.flag);
      made.mesh.position.set(l.slotX - 0.1, 0, -(l.top + 0.29));
      this.textures.push(made.texture);
      this.scene.add(made.mesh);
      this.tags.set(text, made.mesh);
      next = made.mesh;
    }
    if (next === this.tag) return;
    if (this.tag) this.tag.visible = false;
    if (next) next.visible = true;
    this.tag = next;
  }

  private write(key: keyof SlowRegressionScene['readouts'], text: string): void {
    const readout = this.readouts[key];
    if (readout.text === text) return;
    readout.text = text;
    this.digits.write(readout.offset, readout.cells, text);
  }
}

export default (() => new SlowRegressionScene()) satisfies SceneFactory;
