import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
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
import { CABLES, ERROR, GOLD, HISTORY, INK, SLOT, cable, clock, desk, disc, flat, glass, glyph, knob, label, lanes, layout, meters, plug, rack, rowY, ruler, square, trace, truthLine, type Layout, type Level } from './art';

type At = (s: number) => EnvFrame | null;

interface Example {
  k: number;
  inputs: Float32Array;
  signs: number;
  truth: number;
  prediction: number;
  reward: number;
  flip: number;
}

const FAST = 5;
const SWING = (3 * Math.PI) / 4;
const IN: readonly [number, number] = [0, 0.3];
const OUT: readonly [number, number] = [0.3, 0.7];
const SCROLL: readonly [number, number] = [0.6, 1];
const TURN: readonly [number, number] = [0, 0.6];
const PINNED = 0.2;

const phase = (t: number, [a, b]: readonly [number, number]) => smoothstep01(clamp01((t - a) / (b - a)));
const swing = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : 1 + 2.2 * (t - 1) ** 3 + 1.2 * (t - 1) ** 2);
const weightOf = (signs: number, j: number) => ((signs >>> j) & 1 ? -1 : 1);
const level = (value = 0): Level => uniform(value);

class IDBDScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'idbd';

  private l: Layout | null = null;
  private inputs = 0;
  private mask = 0;
  private bank = new Float32Array(4);
  private bankTexture: THREE.DataTexture | null = null;
  private readonly strip = new Float32Array((HISTORY + 2) * 4);
  private readonly stripTexture = new THREE.DataTexture(this.strip, HISTORY + 2, 1, THREE.RGBAFormat, THREE.FloatType);
  private readonly textures: THREE.Texture[] = [this.stripTexture];
  private sums: Level[] = [];
  private angles: Level[] = [];
  private strokes: Level[] = [];
  private pulses: Level[] = [];
  private readonly feedback = level(-1);
  private readonly scroll = level(0);
  private readonly count = level(0);
  private readonly squareLevels = { low: level(), high: level(), truth: level(), fade: level() };
  private readonly marks = { truth: level(), pen: level(), penOn: level(), railHigh: level(), railLow: level() };
  private readonly squareGlow = level(0);
  private penLamp: THREE.Mesh | null = null;
  private truthLamp: THREE.Mesh | null = null;
  private squareLamp: THREE.Mesh | null = null;

  private cur: Example = this.blank();
  private nxt: Example = this.blank();
  private readonly scratch: Example = this.blank();
  private curOk = false;
  private nxtOk = false;
  private complete = false;
  private shownStep = -1;
  private curCount = 0;
  private nxtCount = 0;
  private pace = 0;
  private lastPosition = 0;
  private sparked = -1;
  private partial = new Float64Array(1);
  private partialNext = new Float64Array(1);
  private readonly ledger = { k: -1, prediction: NaN, truth: NaN, reward: 0, mean: NaN, flipK: -1, flipIndex: -1, flipSign: 0, clock: 0 };

  override legend: LegendSpec | null = null;

  private blank(): Example {
    return { k: -1, inputs: new Float32Array(this.inputs + 1), signs: 0, truth: 0, prediction: NaN, reward: 0, flip: -1 };
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment, inputs = parameter(env, 'inputs'), signs = parameter(env, 'signs');
    if (env.obs !== inputs + 1 || env.words !== 3 || env.actions !== 1) throw new Error(`${env.envId} shows ${env.obs} observations, keeps ${env.words} words and takes ${env.actions} actions`);
    if (signs < 1 || signs > CABLES.length || signs > inputs) throw new Error(`${env.envId} weights ${signs} of ${inputs} inputs`);
    const l = layout(inputs, signs, parameter(env, 'scale'), parameter(env, 'period'));
    this.l = l;
    this.inputs = inputs;
    this.mask = 2 ** signs - 1;
    this.cur = this.blank();
    this.nxt = this.blank();
    Object.assign(this.scratch, this.blank());
    this.bank = new Float32Array(l.rows * 4);
    this.sums = Array.from({ length: signs + 1 }, () => level());
    this.angles = Array.from({ length: signs }, () => level(SWING));
    this.strokes = Array.from({ length: signs }, () => level(1));
    this.pulses = Array.from({ length: signs }, () => level(-1));
    this.partial = new Float64Array(signs + 1);
    this.partialNext = new Float64Array(signs + 1);
    return new THREE.Box2(new THREE.Vector2(-l.half - 0.25, -l.top - 0.4), new THREE.Vector2(l.half + 0.25, l.top + 0.4));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x9aa6bd).multiplyScalar(0.8);
  }

  protected override overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const accent = this.accentOf(ctx);
    return {
      debug: (frame, kit) => {
        const l = this.l, i = frame.state.step, at = this.reader(frame), e = this.scratch;
        if (!l || !this.read(at, i, e)) return;
        const u = l.unit, tips = kit.marks('inputs', l.rows);
        for (let r = 0; r < l.rows; r++) tips.set(l.meterX + e.inputs[r]! * u, 0.1, -rowY(l, r), 0.045, r === l.rows - 1 ? GOLD : 0xf6ead0);
        const weights = kit.marks('weights', l.signs), stack = kit.marks('sums', l.signs + 1);
        let sum = 0;
        stack.set(l.laneX[0]!, 0.1, 0, 0.04, 0xffffff);
        for (let j = 0; j < l.signs; j++) {
          const w = weightOf(e.signs, j), angle = w * SWING;
          weights.set(l.knobX + Math.sin(angle) * 0.13, 0.1, -(rowY(l, j) + Math.cos(angle) * 0.13), 0.04, CABLES[j]!);
          sum += w * e.inputs[j]!;
          stack.set(l.laneX[j]!, 0.1, -(j === l.signs - 1 ? e.truth : sum) * u, 0.045, CABLES[j]!);
        }
        kit.marks('truth', 1).set(l.ruleX - 0.03, 0.1, -e.truth * u, 0.06, GOLD);
        if (Number.isFinite(e.prediction)) {
          kit.marks('prediction', 1).set(l.ruleX + 0.03, 0.1, -e.prediction * u, 0.06, accent);
          const low = Math.min(e.truth, e.prediction) * u, high = Math.max(e.truth, e.prediction) * u, left = l.ruleX - (high - low);
          const outline = kit.curve('square', 5, ERROR);
          for (const [x, y] of [[l.ruleX, low], [left, low], [left, high], [l.ruleX, high], [l.ruleX, low]] as const) outline.point(x, 0.1, -y);
        }
        const dots = kit.marks('history', HISTORY), path = kit.curve('predictions', HISTORY, accent);
        let flips = 0;
        for (let m = 1; m <= HISTORY; m++) {
          const k = i + 1 - m, x = l.historyX + (m - 1) * SLOT;
          if (k < 1 || !this.read(at, k, e)) continue;
          dots.set(x, 0.1, -e.truth * u, 0.03, GOLD);
          path.point(x, 0.1, -e.prediction * u);
          if (e.flip >= 0 && flips < 4) {
            const line = kit.curve(`flip${flips++}`, 2, CABLES[e.flip]!);
            line.point(x + SLOT / 2, 0.1, -l.rail);
            line.point(x + SLOT / 2, 0.1, l.rail);
          }
        }
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const l = this.l!, lit = light.lit, accent = this.accentOf(ctx), env = ctx.environment;
    this.scene.background = new THREE.Color(0x030405);
    pool.group.renderOrder = 11;
    const bankTexture = new THREE.DataTexture(this.bank, l.rows, 1, THREE.RGBAFormat, THREE.FloatType);
    for (let r = 0; r < l.rows; r++) this.bank[r * 4 + 1] = r === l.rows - 1 ? 1 : 0;
    bankTexture.needsUpdate = true;
    this.bankTexture = bankTexture;
    this.stripTexture.needsUpdate = true;
    this.textures.push(bankTexture);

    this.scene.add(desk(lit), rack(l, lit), glass(l), meters(l, bankTexture), lanes(l, this.sums), square(l, this.squareLevels), truthLine(l, this.marks), ruler(l, this.marks, accent), trace(l, this.stripTexture, this.scroll, accent), clock(l, this.count, lit));
    for (let j = 0; j < l.signs; j++) {
      const y = rowY(l, j);
      this.scene.add(knob(l.knobX, y, CABLES[j]!, this.angles[j]!, lit), glyph(l.glyphX, y, CABLES[j]!, this.strokes[j]!));
      this.scene.add(cable([l.jackX, y], [l.mixJackX, y], 0.1, CABLES[j]!, this.pulses[j]!, lit), plug(l.jackX, y, CABLES[j]!, lit), plug(l.mixJackX, y, CABLES[j]!, lit));
      this.place(light.emitter(disc(0.045), rgb(CABLES[j]!).mul(2.2)), l.glyphX, y, 1);
      this.place(light.occluder(disc(0.15)), l.knobX, y, 1);
    }
    const bottom = rowY(l, l.rows - 1);
    this.scene.add(cable([l.mixJackX, bottom], [l.jackX, bottom], 0.12, GOLD, this.feedback, lit), plug(l.jackX, bottom, GOLD, lit), plug(l.mixJackX, bottom, GOLD, lit));
    this.penLamp = this.place(light.emitter(disc(0.06), rgb(accent).mul(this.marks.penOn.mul(5))), l.ruleX + 0.3, 0, 2);
    this.truthLamp = this.place(light.emitter(disc(0.06), rgb(GOLD).mul(4)), l.ruleX - 0.25, 0, 2);
    this.squareLamp = this.place(light.emitter(flat(1, 1), rgb(ERROR).mul(this.squareGlow)), l.ruleX, 0, 2);

    const top = l.top - 0.13, inputs = parameter(env, 'inputs'), period = l.period;
    this.put('INPUTS · SEEN BY THE AGENT', 0.16, INK, (l.a[0] + l.a[1]) / 2, top, 0, lit, 0.1);
    this.put('WEIGHTS', 0.16, INK, (l.b[0] + l.b[1]) / 2, top, 0, lit, 0.1);
    this.put('Σ', 0.2, INK, (l.laneX[0]! + l.laneX[l.signs - 1]!) / 2, top, 0, lit);
    this.put('ERROR² = −REWARD', 0.16, ERROR, l.arenaX + 0.05, top, -1, lit, 0.08);
    this.put('TRUTH', 0.16, GOLD, l.ruleX - 0.12, top, 1, lit, 0.08);
    this.put('PREDICTION', 0.16, accent, l.ruleX + 0.12, top, -1, lit, 0.08);
    this.put(`${HISTORY} STEPS AGO`, 0.16, INK, l.historyX + SLOT * (HISTORY - 1), top, 1, lit, 0.08);
    for (let r = 0; r < l.rows; r++) this.put(r < inputs ? `x${r + 1}` : 'last y', 0.17, r < inputs ? INK : GOLD, l.labelX + 0.18, rowY(l, r), 1, lit);
    this.put('NEXT FLIP', 0.14, INK, l.knobX, l.clockTop + 0.28, 0, lit, 0.06);
    this.put(`EVERY ${formatCount(period)}`, 0.14, INK, l.knobX, l.clockBottom - 0.28, 0, lit, 0.06);
    this.put('Σ OUT', 0.14, GOLD, l.glyphX - 0.02, bottom, 0, lit);
    for (const value of [l.scale, l.scale / 2, 0, -l.scale / 2, -l.scale]) this.put(value === 0 ? '0' : `${value > 0 ? '+' : '−'}${formatCount(Math.abs(value))}`, 0.15, 0x9fb0c0, l.ruleX + 0.68, value * l.unit, 1);

    const l2 = this.ledger;
    this.legend = {
      title: 'IDBD',
      rows: [
        { key: 'prediction', label: 'prediction', tint: accent, value: () => (Number.isFinite(l2.prediction) ? formatRate(l2.prediction) : '—') },
        { key: 'truth', label: 'truth', tint: GOLD, value: () => (Number.isFinite(l2.truth) ? formatRate(l2.truth) : '—') },
        { key: 'reward', label: 'reward', tint: ERROR, value: () => (l2.k >= 1 ? formatSigned(l2.reward).replace('-', '−') : '—') },
        { key: 'mean', label: `mean error², ${formatCount(period)}`, tint: ERROR, value: () => (Number.isFinite(l2.mean) ? formatRate(l2.mean) : '—') },
        { key: 'flip', label: 'last flip', tint: INK, value: () => (l2.flipK < 0 ? (l2.k > HISTORY ? '…' : 'none yet') : `x${l2.flipIndex + 1} → ${l2.flipSign > 0 ? '+1' : '−1'}`) },
      ],
      note: () => `${l.signs} of ${inputs} inputs count`,
      bar: () => l2.clock / period,
    };
  }

  private place(mesh: THREE.Mesh, x: number, y: number, order: number): THREE.Mesh {
    mesh.position.set(x, 0, -y);
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    return mesh;
  }

  private put(text: string, height: number, tint: number, x: number, y: number, align: -1 | 0 | 1, lit?: Stage2D['light']['lit'], spacing = 0): void {
    const made = label(text, height, tint, spacing, lit);
    made.mesh.position.set(x + (align === -1 ? made.width / 2 : align === 1 ? -made.width / 2 : 0), 0, -y);
    this.textures.push(made.texture);
    this.scene.add(made.mesh);
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
    out.k = k;
    out.inputs.set(source.observations.subarray(0, l.rows));
    out.signs = source.words[SUPERVISED_SOURCE]! & this.mask;
    out.truth = wordFloat(source.words[SUPERVISED_TARGET]!);
    out.prediction = NaN;
    out.reward = 0;
    if (scored) {
      if (scored.words[SUPERVISED_SHOWN] !== source.words[SUPERVISED_TARGET]) throw new Error(`IDBD step ${k} scored against ${wordFloat(scored.words[SUPERVISED_SHOWN]!)} though step ${k - 1} set the truth to ${out.truth}`);
      out.prediction = l.scale * Math.max(-1, Math.min(1, scored.actions[0]!));
      out.reward = scored.reward;
      const miss = out.prediction - out.truth;
      if (Math.abs(out.reward + miss * miss) > 1e-3 * (1 + miss * miss)) throw new Error(`IDBD step ${k} paid ${out.reward} for predicting ${out.prediction} against ${out.truth}`);
    }
    const changed = (out.signs ^ prior.words[SUPERVISED_SOURCE]!) & this.mask;
    if (changed & (changed - 1)) throw new Error(`IDBD step ${k} flipped several signs at once`);
    out.flip = changed ? 31 - Math.clz32(changed) : -1;
    let sum = 0, size = 0;
    for (let j = 0; j < l.signs; j++) {
      sum += weightOf(out.signs, j) * out.inputs[j]!;
      size += Math.abs(out.inputs[j]!);
    }
    if (Math.abs(sum - out.truth) > 1e-4 * (1 + size)) throw new Error(`IDBD step ${k} holds truth ${out.truth} though its signed inputs sum to ${sum}`);
    return true;
  }

  private refresh(at: At, i: number): void {
    const l = this.l!, strip = this.strip, ledger = this.ledger;
    strip.fill(0);
    this.curOk = this.nxtOk = false;
    let flipK = -1, flipIndex = -1, flipSign = 0, sum = 0, count = 0, nextFlip = -1, missing = 0;
    for (let m = 0; m < HISTORY + 2; m++) {
      const k = i + 1 - m, e = m === 0 ? this.nxt : m === 1 ? this.cur : this.scratch;
      if (k < 0) continue;
      if (!this.read(at, k, e)) {
        if (m >= 1) missing++;
        continue;
      }
      if (m === 0) this.nxtOk = true;
      if (m === 1) this.curOk = true;
      if (k < 1) continue;
      const o = m * 4;
      strip[o] = e.truth;
      strip[o + 1] = e.prediction;
      strip[o + 2] = e.flip >= 0 ? (e.flip + 1) * weightOf(e.signs, e.flip) : 0;
      strip[o + 3] = 1;
      if (m === 0 && e.flip >= 0) nextFlip = k;
      if (m >= 1 && e.flip >= 0 && flipK < 0) {
        flipK = k;
        flipIndex = e.flip;
        flipSign = weightOf(e.signs, e.flip);
      }
      if (m >= 1 && m <= l.period) {
        sum -= e.reward;
        count++;
      }
    }
    this.stripTexture.needsUpdate = true;
    this.nxtOk &&= this.curOk;
    this.complete = missing === 0;
    const since = (k: number, last: number) => (k < 1 ? 0 : last >= 1 ? k - last + 1 : Math.min(k, l.period));
    this.curCount = since(i, flipK);
    this.nxtCount = since(i + 1, nextFlip >= 0 ? nextFlip : flipK);
    ledger.k = i;
    ledger.prediction = this.cur.prediction;
    ledger.truth = this.cur.truth;
    ledger.reward = this.cur.reward;
    ledger.mean = count > 0 ? sum / count : NaN;
    ledger.flipK = flipK;
    ledger.flipIndex = flipIndex;
    ledger.flipSign = flipSign;
    ledger.clock = this.curCount;
    this.shownStep = i;
  }

  private stack(e: Example, out: Float64Array): void {
    const l = this.l!;
    out[0] = 0;
    for (let j = 0; j < l.signs; j++) out[j + 1] = out[j]! + weightOf(e.signs, j) * e.inputs[j]!;
    out[l.signs] = e.truth;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const l = this.l;
    if (!l) return;
    const dt = frame.dtReal;
    pool.update(dt, time);
    if (!frame.scrubbed && dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const at = this.reader(frame), i = frame.state.step, fast = this.pace > this.controlHz * FAST;
    if (frame.scrubbed || i !== this.shownStep || !this.complete || (!this.nxtOk && at(i + 1) !== null)) this.refresh(at, i);
    if (!this.curOk) return;
    if (frame.scrubbed) {
      this.sparked = i;
      pool.clear();
    }
    const cur = this.cur, live = this.nxtOk && !fast, nxt = live ? this.nxt : cur, t = live ? clamp01(frame.position - i) : 0;
    const tIn = phase(t, IN), tOut = phase(t, OUT), tTurn = clamp01((t - TURN[0]) / (TURN[1] - TURN[0])), tScroll = live ? phase(t, SCROLL) : 0, u = l.unit;

    for (let r = 0; r < l.rows; r++) this.bank[r * 4] = cur.inputs[r]! + (nxt.inputs[r]! - cur.inputs[r]!) * tIn;
    this.bankTexture!.needsUpdate = true;

    for (let j = 0; j < l.signs; j++) {
      const from = weightOf(cur.signs, j), to = weightOf(nxt.signs, j);
      this.angles[j]!.value = SWING * (from + (to - from) * swing(tTurn));
      this.strokes[j]!.value = (from > 0 ? 1 : 0) + ((to > 0 ? 1 : 0) - (from > 0 ? 1 : 0)) * smoothstep01(tTurn);
      this.pulses[j]!.value = live ? (t - 0.02) / 0.3 : -1;
    }
    this.feedback.value = live ? (t - 0.02) / 0.3 : -1;

    this.stack(cur, this.partial);
    this.stack(nxt, this.partialNext);
    for (let j = 0; j <= l.signs; j++) this.sums[j]!.value = this.partial[j]! + (this.partialNext[j]! - this.partial[j]!) * tOut;

    const truth = cur.truth + (nxt.truth - cur.truth) * tOut;
    const scoredNow = Number.isFinite(cur.prediction), scoredNext = Number.isFinite(nxt.prediction);
    const pen = scoredNow ? (scoredNext ? cur.prediction + (nxt.prediction - cur.prediction) * tOut : cur.prediction) : scoredNext ? nxt.prediction : 0;
    const penOn = scoredNow ? 1 : scoredNext ? tOut : 0;
    const m = this.marks;
    m.truth.value = truth * u;
    m.pen.value = pen * u;
    m.penOn.value = penOn;
    m.railHigh.value = penOn * clamp01(1 - (l.scale - pen) / PINNED);
    m.railLow.value = penOn * clamp01(1 - (pen + l.scale) / PINNED);
    const low = Math.min(truth, pen) * u, high = Math.max(truth, pen) * u, side = high - low;
    const sq = this.squareLevels;
    sq.low.value = low;
    sq.high.value = high;
    sq.truth.value = truth * u;
    sq.fade.value = penOn;
    this.squareGlow.value = 0.9 * penOn * clamp01(side / 0.3);
    if (this.squareLamp) {
      this.squareLamp.position.set(l.ruleX - side / 2, 0, -(low + high) / 2);
      this.squareLamp.scale.set(Math.max(side, 1e-3), 1, Math.max(side, 1e-3));
    }
    this.penLamp?.position.set(l.ruleX + 0.3, 0, -pen * u);
    this.truthLamp?.position.set(l.ruleX - 0.25, 0, -truth * u);
    this.scroll.value = tScroll;
    this.count.value = this.curCount + (this.nxtCount - this.curCount) * tIn;

    const flipping = live && nxt.flip >= 0 && t > 0 ? nxt.k : cur.flip >= 0 ? cur.k : -1;
    if (flipping > this.sparked) {
      const e = flipping === nxt.k ? nxt : cur;
      this.sparked = flipping;
      pool.burst({ x: l.knobX, y: 38, z: -rowY(l, e.flip) }, CABLES[e.flip]!, 0.3, time);
    }
  }
}

export default (() => new IDBDScene()) satisfies SceneFactory;
