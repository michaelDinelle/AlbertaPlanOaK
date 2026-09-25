import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { mulberry32 } from '@kit/rng';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { FROST_CALM, FROST_CLOCK, FROST_PHASE, FROST_POSITION, frostHazard, frostHeat, frostPrediction, frostToken } from '@sim/envs/frost';
import type { EnvFrame, GridCell } from '@sim/types';
import { constant } from '../../data/envs';
import { formatCount, formatRate, formatSigned } from '../../format';
import {
  CUE, FLAME, GOLD, HANG, HEAT, PIP_RING, RIME, SLOT, SPAN,
  disc, ground, hearth, lantern, layout, mist, pines, rail, ring, slots, walker,
  type Layout, type Scalar, type Slots, type Walker,
} from './art';

const FAST = 5;
const NORTH = Math.PI / 2;
const FLARE_LIFE = 1.4;
const CUE_LIFE = 0.7;
const FROST_Y = 0.08;
const TOKEN_Y = -0.075;
const TRUTH_FROST = 0x1f7bff;
const TRUTH_TOKEN = 0x00a7c4;
const TRUTH_HEAT = 0xff7a00;

const span = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
const ease = (t: number, a: number, b: number) => smoothstep01(span(t, a, b));
const after = (time: number, at: number, life: number) => Math.max(0, 1 - (time - at) / life);
const turnBetween = (from: number, to: number, k: number) => from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * k;
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

class FrostHollowScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'accentLight'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'frostHollow';

  private L: Layout = layout(7, 3);
  private capacity = 1;
  private rate = 1;
  private threshold = 0;
  private episode = 1;
  private pipCount = 0;
  private accent = 0;

  private readonly clock: Scalar = uniform(0);
  private readonly fire: Scalar = uniform(1);
  private readonly turn: Scalar = uniform(0);
  private readonly rime: Scalar = uniform(0);
  private readonly fade: Scalar = uniform(1);
  private readonly reach: Scalar = uniform(0);
  private readonly gold: Scalar = uniform(0);
  private readonly ice: Scalar = uniform(0);
  private readonly thaw: Scalar = uniform(0);
  private readonly fill: Scalar = uniform(0);
  private readonly thin: Scalar = uniform(0);
  private readonly glow: Scalar = uniform(0);
  private readonly flash: Scalar = uniform(0);
  private readonly scroll: Scalar = uniform(0);
  private readonly caret: Scalar = uniform(0);
  private readonly caretLit: Scalar = uniform(0);
  private readonly warmth: Scalar = uniform(0);
  private readonly beacon: Scalar = uniform(0);
  private readonly frostSlots: Slots = slots(2 * SPAN + 2);
  private readonly tokenSlots: Slots = slots(2 * SPAN + 2);
  private readonly edgeSlots: Slots = slots(2 * SPAN + 2);

  private body: Walker | null = null;
  private warmLight: THREE.Mesh | null = null;
  private lampLight: THREE.Mesh | null = null;

  private pace = 0;
  private lastPosition = 0;
  private flareAt = -Infinity;
  private cueAt = -Infinity;
  private faces = { step: -1, from: 0, to: 0 };
  private readonly spot = new THREE.Vector3();
  private readonly ledger = { position: 0, heat: 0, hazard: false, wait: 0, token: false, forecast: 0, reward: 0, clock: 0 };

  override legend: LegendSpec | null = null;

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.L = layout(constant(env, 'FROST_CELLS'), constant(env, 'FROST_CENTER'));
    this.capacity = constant(env, 'FROST_CAPACITY');
    this.rate = constant(env, 'FROST_RATE');
    this.threshold = constant(env, 'FROST_THRESHOLD');
    this.episode = constant(env, 'FROST_EPISODE');
    this.pipCount = Math.round(this.capacity / this.rate);
    if (this.pipCount !== this.capacity / this.rate || this.pipCount < 1) throw new Error(`${env.envId} capacity ${this.capacity} is not a whole number of ${this.rate} heat steps`);
    const x = this.L.ledge + 0.6;
    return new THREE.Box2(new THREE.Vector2(-x, -2.65), new THREE.Vector2(x, 1.75));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0x7489c4).multiplyScalar(0.6);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 }, goal: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        agent.x = state.pose[FROST_POSITION]!;
        return agent;
      },
      goalCell: () => {
        goal.x = this.L.centre;
        return goal;
      },
      debug: (frame, kit) => {
        const L = this.L, i = frame.state.step, head = frame.frameAt(i);
        if (head.step !== i) return;
        const t = this.scroll.value, railZ = -L.railY;
        const refuges = kit.marks('refuges', 2);
        for (const cell of [0, L.cells - 1]) refuges.set(L.cellX(cell), 0.05, 0, 0.1, 0x86f0b4);
        const frosted = kit.marks('frost', L.cells);
        if (frostHazard(head)) for (let cell = 1; cell < L.cells - 1; cell++) frosted.set(L.cellX(cell), 0.05, 0.3, 0.09, TRUTH_FROST);
        const x = L.cellX(head.words[FROST_POSITION]!), heat = kit.curve('heat', 2, TRUTH_HEAT);
        heat.point(x - 0.4, 0.05, 0.42);
        heat.point(x - 0.4 + (0.8 * frostHeat(head)) / this.capacity, 0.05, 0.42);
        const timeline = kit.marks('frost timeline', 2 * SPAN + 2), signal = kit.marks('signal timeline', SPAN + 2);
        let first = -1;
        for (let k = -SPAN; k <= SPAN + 1; k++) {
          const f = frame.frameAt(i + k);
          if (i + k < 0 || f.step !== i + k) continue;
          const sx = (k - 0.5 - t) * SLOT;
          if (Math.abs(sx) > SPAN * SLOT) continue;
          if (frostHazard(f)) timeline.set(sx, 0.05, railZ - FROST_Y, 0.065, TRUTH_FROST);
          if (frostHazard(f) && k >= 1 && first < 0) first = k;
          if (k <= 0 && frostToken(f)) signal.set(sx, 0.05, railZ - TOKEN_Y, 0.045, TRUTH_TOKEN);
        }
        const truth = kit.curve('true wait', 2, TRUTH_FROST);
        if (first > 0) {
          truth.point(0, 0.05, railZ - FROST_Y - 0.13);
          truth.point((first - 1 - t) * SLOT, 0.05, railZ - FROST_Y - 0.13);
        }
        kit.marks('forecast', 1).set(this.caret.value, 0.05, railZ - FROST_Y - 0.13, 0.06, TRUTH_HEAT);
        const before = frame.frameAt(i - 1), intent = kit.curve('intent', 5, 0x5ec1ff);
        if (i < 1 || before.step !== i - 1 || head.done) return;
        const dir = Math.trunc(head.actions[0]!) - 1, from = L.cellX(before.words[FROST_POSITION]!), tip = from + dir * 0.45;
        intent.point(from, 0.06, -0.3);
        intent.point(tip, 0.06, -0.3);
        if (dir === 0) return;
        intent.point(tip - dir * 0.12, 0.06, -0.38);
        intent.point(tip, 0.06, -0.3);
        intent.point(tip - dir * 0.12, 0.06, -0.22);
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    const L = this.L, lit = light.lit;
    this.accent = this.accentOf(ctx);
    this.scene.add(ground(L, lit, ctx.seed, this.clock));

    const parts = hearth(lit, this.fire, this.clock);
    for (const part of [parts.base, parts.flames]) part.position.set(L.cellX(L.centre), 0, -L.fireY);
    const flame = light.emitter(disc(0.075), rgb(FLAME).mul(this.fire.mul(7)));
    flame.position.set(L.cellX(L.centre), 0.02, -L.fireY);
    this.scene.add(parts.base, parts.flames, flame);

    const body = walker(this.accent, lit, { turn: this.turn, rime: this.rime, fade: this.fade, reach: this.reach, gold: this.gold, ice: this.ice, thaw: this.thaw }, this.pipCount);
    this.body = body;
    this.warmLight = light.emitter(ring(PIP_RING - 0.015, PIP_RING + 0.015), rgb(HEAT).mul(this.warmth));
    this.scene.add(body.sockets, body.figure, body.embers, this.warmLight, mist(L, this.fill, this.thin, this.clock));

    const lamp = lantern(lit, this.glow, this.flash, this.clock);
    lamp.post.position.set(-HANG.x, 0.01, -L.lanternY);
    lamp.halo.position.set(0, 0.01, -(L.lanternY + HANG.y - 0.02));
    this.lampLight = light.emitter(disc(0.045), rgb(CUE).mul(this.beacon));
    this.lampLight.position.set(0, 0.02, -(L.lanternY + HANG.y - 0.02));
    this.scene.add(lamp.post, lamp.halo, this.lampLight);

    const board = rail(lit, { frost: this.frostSlots, token: this.tokenSlots, edge: this.edgeSlots, scroll: this.scroll, caret: this.caret, caretLit: this.caretLit, threshold: this.threshold });
    board.position.set(0, 0.01, -L.railY);
    this.scene.add(board);

    const spots = this.plant(ctx.seed);
    this.scene.add(pines(ctx.seed, spots));
    for (const spot of spots) {
      const shade = light.occluder(disc(0.36 * spot.y));
      shade.position.set(spot.x, 0.01, spot.z);
      this.scene.add(shade);
    }
    pool.group.traverse((object) => { object.renderOrder = 13; });

    const row = this.ledger, cells = L.cells;
    this.legend = {
      title: 'FROST HOLLOW',
      rows: [
        { key: 'walker', label: 'walker', tint: this.accent, value: () => `cell ${row.position + 1} of ${cells} · ${row.position === L.centre ? 'at the fire' : row.position === 0 || row.position === cells - 1 ? 'sheltered' : 'exposed'}` },
        { key: 'heat', label: 'heat', tint: HEAT, value: () => `${row.heat} of ${this.capacity}` },
        { key: 'frost', label: 'frost', tint: RIME, value: () => (row.hazard ? 'falling' : `in ${plural(row.wait, 'step')}`) },
        { key: 'signal', label: 'signal', tint: CUE, value: () => `${row.token ? 'lit' : 'dark'} · forecast ${formatRate(row.forecast)}` },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => formatSigned(row.reward) },
      ],
      note: () => `episode step ${formatCount(row.clock)} of ${formatCount(this.episode)}`,
      bar: () => row.heat / this.capacity,
    };
  }

  private plant(seed: number): THREE.Vector3[] {
    const L = this.L, random = mulberry32(seed ^ 0x7ee5), spots: THREE.Vector3[] = [], half = SPAN * SLOT + 0.5;
    for (let tries = 0; tries < 600 && spots.length < 24; tries++) {
      const x = (random() * 2 - 1) * 7, y = random() * 6.2 - 2.9, r = 0.3 + 0.24 * random();
      if (Math.abs(x) < L.ledge + 0.3 + r && y - r < L.rimN + 0.12 && y + r > L.rimS - 0.2) continue;
      if (Math.abs(x + HANG.x) < 0.45 + r && y - r < L.lanternY + 0.45 && y + r > L.lanternY - 0.5) continue;
      if (Math.abs(x) < half + r && y + r > L.railY - 0.25 && y - r < L.railY + 0.3) continue;
      if (spots.some((spot) => Math.hypot(spot.x - x, -spot.z - y) < spot.y + r + 0.12)) continue;
      spots.push(new THREE.Vector3(x, r, -y));
    }
    return spots.sort((a, b) => a.z - b.z);
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const body = this.body, L = this.L;
    if (!body) return;
    if (frame.scrubbed) pool.clear();
    pool.update(frame.dtReal, time);
    this.clock.value = time;
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const fast = this.pace > this.controlHz * FAST;

    const i = frame.state.step, head = frame.frameAt(i), ahead = frame.frameAt(i + 1);
    const next = !fast && ahead.step === i + 1 ? ahead : null, t = next ? clamp01(frame.position - i) : 0;
    const from = head.words[FROST_POSITION]!, to = next ? next.words[FROST_POSITION]! : from, ended = next !== null && next.done !== 0;

    let x = L.cellX(from), fade = 1;
    if (ended) {
      x = t < 0.5 ? L.cellX(from) : L.cellX(to);
      fade = t < 0.5 ? 1 - ease(t, 0.05, 0.4) : ease(t, 0.6, 0.95);
    } else if (to !== from) x += (L.cellX(to) - L.cellX(from)) * ease(t, 0.08, 0.7);
    else if (next) {
      const push = Math.trunc(next.actions[0]!) - 1;
      if (push !== 0) x += push * 0.12 * Math.sin(Math.PI * span(t, 0.1, 0.6));
    }
    this.fade.value = fade;

    if (this.faces.step !== i || frame.scrubbed) this.faces = { step: i, from: this.faceOf(frame, i), to: next ? this.faceOf(frame, i + 1) : this.faceOf(frame, i) };
    this.turn.value = turnBetween(this.faces.from, next ? this.faces.to : this.faces.from, ease(t, 0.02, 0.45)) - NORTH;
    this.reach.value = (from === L.centre ? 1 : 0) + ((to === L.centre ? 1 : 0) - (from === L.centre ? 1 : 0)) * ease(t, 0.55, 0.9);

    const pips = body.pips.values, heatA = frostHeat(head), litA = Math.round(heatA / this.rate);
    for (let k = 0; k < pips.length; k++) pips[k] = k < litA ? 1 : 0;
    let gold = 0, ice = 0;
    if (next) {
      const heatB = frostHeat(next);
      if (next.reward > 0) {
        if (litA < pips.length) pips[litA] = ease(t, 0.45, 0.65);
        gold = ease(t, 0.7, 1);
        for (let k = 0; k < pips.length; k++) pips[k] = pips[k]! * (1 - ease(t, 0.92, 1));
      } else if (heatB > heatA && litA < pips.length) pips[litA] = ease(t, 0.55, 0.8);
      else if (heatB < heatA && !ended) {
        ice = ease(t, 0.4, 0.75);
        for (let k = 0; k < pips.length; k++) pips[k] = pips[k]! * (1 - ease(t, 0.78, 1));
      } else if (heatB < heatA) for (let k = 0; k < pips.length; k++) pips[k] = pips[k]! * (1 - ease(t, 0.05, 0.4));
    }
    this.gold.value = gold;
    this.ice.value = ice;
    let warm = 0;
    for (const v of pips) warm += v;
    warm /= pips.length;
    this.thaw.value = warm;
    this.warmth.value = warm * 2.4;
    if (this.warmLight) {
      this.warmLight.visible = warm > 0.01 && fade > 0.05;
      this.warmLight.position.set(x, 0.02, 0);
    }

    const hazA = frostHazard(head), hazB = next ? frostHazard(next) : hazA;
    let fill = hazA ? 1 : 0, thin = 0;
    if (!hazA && hazB) fill = ease(t, 0.05, 0.85);
    if (hazA && !hazB) thin = ease(t, 0, 0.6);
    this.fill.value = fill;
    this.thin.value = thin;
    const exposed = clamp01((L.reach + 0.05 - Math.abs(x)) / 0.1);
    this.rime.value = exposed * fill * (1 - thin);

    const tokA = frostToken(head) ? 1 : 0, tokB = next ? (frostToken(next) ? 1 : 0) : tokA;
    const glow = tokA + (tokB - tokA) * ease(t, 0.8, 1), flash = after(time, this.cueAt, CUE_LIFE);
    this.glow.value = glow;
    this.flash.value = flash;
    this.caretLit.value = glow;
    this.beacon.value = glow * 5 + flash * 4;
    if (this.lampLight) this.lampLight.visible = this.beacon.value > 0.02;

    const flicker = 0.08 * Math.sin(time * 11.3) * Math.sin(time * 4.7) + 0.05 * Math.sin(time * 17.9);
    this.fire.value = 1 + flicker + 0.9 * after(time, this.flareAt, FLARE_LIFE);

    this.timeline(frame, i, t, head, next);

    for (const part of [body.figure, body.sockets, body.embers]) part.position.set(x, 0.01, 0);

    if (!frame.scrubbed && !fast) {
      for (const event of frame.events) {
        if (event.type === 'goal') {
          pool.burst(this.spot.set(x, 30, 0), GOLD, 0.85, time);
          this.flareAt = time;
        } else if (event.type === 'harm') pool.burst(this.spot.set(x, 30, 0), RIME, 0.45, time);
        else if (event.type === 'cue') {
          this.cueAt = time;
        }
      }
    }

    const row = this.ledger;
    row.position = from;
    row.heat = heatA;
    row.hazard = hazA;
    row.wait = hazA ? 0 : head.words[FROST_CALM]! - head.words[FROST_PHASE]!;
    row.token = tokA === 1;
    row.forecast = frostPrediction(head);
    row.reward = head.reward;
    row.clock = head.words[FROST_CLOCK]!;
  }

  private timeline(frame: FrameUpdate, i: number, t: number, head: EnvFrame, next: EnvFrame | null): void {
    const frost = this.frostSlots.values, token = this.tokenSlots.values, edge = this.edgeSlots.values;
    for (let k = -SPAN; k <= SPAN + 1; k++) {
      const s = i + k, f = s >= 0 ? frame.frameAt(s) : null, known = f !== null && f.step === s;
      frost[k + SPAN] = known ? (frostHazard(f) ? 1 : 0) : -1;
      token[k + SPAN] = known && k <= 1 && frostToken(f) ? 1 : 0;
      edge[k + SPAN] = known && f.done !== 0 ? 1 : 0;
    }
    this.scroll.value = t;
    const now = frostPrediction(head) - 1, then = next ? frostPrediction(next) : now + 1;
    this.caret.value = (now + (then - now) * ease(t, 0.6, 1) - t) * SLOT;
  }

  private faceOf(frame: FrameUpdate, j: number): number {
    const L = this.L, at = frame.frameAt(j);
    if (at.step !== j) return NORTH;
    const cell = at.words[FROST_POSITION]!;
    if (cell === L.centre) return NORTH;
    if (cell === 0 || cell === L.cells - 1) return cell === 0 ? 0 : Math.PI;
    for (let s = j; s > Math.max(0, j - 12); s--) {
      const now = frame.frameAt(s), before = frame.frameAt(s - 1);
      if (now.step !== s || before.step !== s - 1 || now.done) break;
      const moved = now.words[FROST_POSITION]! - before.words[FROST_POSITION]!;
      if (moved !== 0) return moved > 0 ? 0 : Math.PI;
    }
    return cell < L.centre ? 0 : Math.PI;
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
    this.body = null;
    this.warmLight = null;
    this.lampLight = null;
  }
}

export default (() => new FrostHollowScene()) satisfies SceneFactory;
