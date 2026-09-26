import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool, CelebrationPoolOptions } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { RPS_CAPACITY, RPS_MOVES, RPS_WORDS, rpsCount, rpsCursor, rpsDecoder, rpsFilled, rpsMine, rpsOutcome, rpsPlayed, rpsSlot, rpsTheirs } from '@sim/envs/rps';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount } from '../../format';
import {
  BADGE, BEADS_Y, BOT, CLASH_X, CLASH_Y, DIAL_X, DIAL_Y, DIGIT_MASKS, LOSS, MOVE_FILL, MOVE_GLOW, MOVE_NAMES, R_CASE, R_TILE_IN, R_TILE_OUT, RECORD, TAU, TIE, WIN, WRIST,
  badge, dial, disc, flash, nodeX, nodeY, rail, slotX, slotY, slotsOf, table, token,
  type BadgeState, type DialState, type RailState, type Scalar, type Tint,
} from './art';
import { FIST, THROWS, easePose, hand, handPose, setPose, type HandPose, type Pose } from './hands';

const HOLD = 0.22;
const CLOSED = 0.32;
const SNAP = 0.6;
const REVEAL = 0.66;
const FLY = 0.7;
const LAND = 0.9;
const SLIDE = 0.16;
const KNUCKLES = 1.3;
const SIZE = 1.2;
const RECOIL = 0.14;
const OUTCOME = [WIN, LOSS, TIE] as const;

const easeOut = (u: number) => 1 - (1 - u) ** 3;
const span = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
const code = (reward: number) => (reward > 0 ? 0 : reward < 0 ? 1 : 2);
const signed = (reward: number) => (reward > 0 ? `+${reward}` : reward < 0 ? `−${-reward}` : '0');

interface Badge extends BadgeState {
  body: THREE.Mesh;
  halo: THREE.Mesh;
  light: Tint;
  flashed: number;
}

interface Hand {
  pose: HandPose;
  mesh: THREE.Mesh;
  mirror: 1 | -1;
}

class RockPaperScissorsScene extends Stage2DScene {
  post: ScenePost = { bloom: 0.5, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 1;
  protected readonly stageName = 'rockPaperScissors';

  private window = 10;
  private digits = 2;
  private accent = new THREE.Color();
  private dialState: DialState | null = null;
  private badges: Badge[] = [];
  private railState: RailState | null = null;
  private agent: Hand | null = null;
  private bot: Hand | null = null;
  private readonly flashTint: Tint = uniform(new THREE.Vector3());
  private readonly flashStrength: Scalar = uniform(0);
  private readonly clashLight: Tint = uniform(new THREE.Vector3());
  private readonly tokenMove: Scalar = uniform(0);
  private readonly tokenLight: Tint = uniform(new THREE.Vector3());
  private tokenMesh: THREE.Mesh | null = null;
  private tokenLamp: THREE.Mesh | null = null;
  private readonly heldStep = new Int32Array(RECORD + 4).fill(-1);
  private readonly heldCode = new Int8Array(RECORD + 4);
  private readonly counts = [0, 0, 0];
  private readonly color = new THREE.Color();
  private readonly cyan = new THREE.Color(BOT);
  private readonly point = new THREE.Vector3();
  private pace = 0;
  private lastPosition = -1;
  private readonly ledger = { step: 0, mine: -1, theirs: -1, reward: 0, read: -1, pick: -1, readCount: 0, won: 0, tied: 0, lost: 0, rounds: 0 };

  override legend: LegendSpec = {
    title: 'ROCK PAPER SCISSORS',
    rows: [
      { key: 'you', label: 'you threw', tint: TOKENS.accents.RockPaperScissors!, value: () => (this.ledger.mine < 0 ? '—' : MOVE_NAMES[this.ledger.mine]!) },
      { key: 'bot', label: 'bot threw', tint: BOT, value: () => (this.ledger.theirs < 0 ? '—' : MOVE_NAMES[this.ledger.theirs]!) },
      { key: 'reward', label: 'reward', tint: WIN, value: () => (this.ledger.mine < 0 ? '—' : signed(this.ledger.reward)) },
      { key: 'read', label: 'bot expects', tint: BOT, value: () => this.readText() },
      { key: 'record', label: `last ${RECORD}`, tint: WIN, value: () => (this.ledger.rounds === 0 ? '—' : `${this.ledger.won} won · ${this.ledger.tied} tied · ${this.ledger.lost} lost`) },
    ],
    note: () => `round ${formatCount(this.ledger.step)} · remembers your last ${this.window}`,
    bar: () => (this.ledger.rounds > 0 ? this.ledger.won / this.ledger.rounds : 0),
  };

  private readText(): string {
    const { read, pick, readCount } = this.ledger;
    if (pick < 0) return '—';
    if (read < 0) return `nothing yet · opens with ${MOVE_NAMES[pick]}`;
    return `${MOVE_NAMES[read]} (${readCount} of ${this.window}) · throws ${MOVE_NAMES[pick]}`;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.window = parameter(ctx.environment, 'window');
    if (ctx.environment.actions !== RPS_MOVES || ctx.environment.words !== RPS_WORDS) throw new Error(`${ctx.environment.envId} carries ${ctx.environment.actions} throws in ${ctx.environment.words} words`);
    if (this.window < 1 || this.window > RPS_CAPACITY) throw new Error(`${ctx.environment.envId} remembers ${this.window} throws; its history words hold ${RPS_CAPACITY}`);
    this.digits = String(this.window).length;
    return new THREE.Box2(new THREE.Vector2(-8, -4.5), new THREE.Vector2(8, 3.5));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xc8b8ac).multiplyScalar(0.72);
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 30 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    const around = (k: number, radius: number, out: THREE.Vector3) => out.set(DIAL_X + radius * Math.sin((k * TAU) / this.window), 0.1, -(DIAL_Y + radius * Math.cos((k * TAU) / this.window)));
    return {
      debug: (frame, kit) => {
        const step = frame.state.step, head = frame.frameAt(step), n = this.window, filled = Math.min(rpsFilled(head), n), at = this.point;
        const slots = kit.marks('window', n), size = Math.min(0.1, (TAU * R_TILE_IN) / n / 3);
        for (let k = 0; k < filled; k++) {
          around(k, R_TILE_IN - 0.11, at);
          slots.set(at.x, at.y, at.z, size, MOVE_FILL[rpsSlot(head, k)]!);
        }
        const cursor = rpsCursor(head);
        around(cursor, R_CASE + 0.14, at);
        kit.marks('cursor', 1).set(at.x, at.y, at.z, 0.08, TOKENS.uiCyan);
        if (filled > 0) {
          around((cursor + n - 1) % n, R_CASE + 0.14, at);
          kit.marks('newest', 1).set(at.x, at.y, at.z, 0.06, this.accent.getHex());
        }
        const bars = kit.bars('counts', RPS_MOVES);
        for (let m = 0; m < RPS_MOVES; m++) {
          const x0 = nodeX(m) - BADGE, y = nodeY(m) - BADGE - 0.2, width = Math.max(1e-3, (2 * BADGE * rpsCount(head, m)) / n), frames = kit.curve(`scale ${MOVE_NAMES[m]}`, 5, 0x8fa6c4);
          bars.set(x0 + width / 2, 0.1, -y, 0.04, MOVE_FILL[m]!, width, 0.05);
          frames.point(x0, 0.1, -y + 0.04);
          frames.point(x0 + 2 * BADGE, 0.1, -y + 0.04);
          frames.point(x0 + 2 * BADGE, 0.1, -y - 0.04);
          frames.point(x0, 0.1, -y - 0.04);
          frames.point(x0, 0.1, -y + 0.04);
        }
        const next = frame.frameAt(step + 1);
        if (next.step === step + 1) {
          const pick = rpsTheirs(next), read = (pick + 2) % RPS_MOVES, plan = kit.curve('plan', 2, TOKENS.uiCyan);
          plan.point(nodeX(pick), 0.1, -nodeY(pick));
          plan.point(nodeX(read), 0.1, -nodeY(read));
          kit.marks('pick', 1).set(nodeX(pick), 0.1, -nodeY(pick), 0.1, TOKENS.uiCyan);
        }
        if (!rpsPlayed(head)) return;
        const throws = kit.marks('throws', 3);
        throws.set(CLASH_X - 1.2, 0.1, -CLASH_Y, 0.16, MOVE_FILL[rpsMine(head)]!);
        throws.set(CLASH_X + 1.2, 0.1, -CLASH_Y, 0.16, MOVE_FILL[rpsTheirs(head)]!);
        throws.set(CLASH_X, 0.1, -CLASH_Y, 0.1, OUTCOME[code(head.reward)]);
      },
    };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    for (const child of pool.group.children) child.renderOrder = 11;
    this.accent.set(this.accentOf(ctx));
    const accent = this.accentOf(ctx), n = this.window, lit = light.lit;
    this.scene.background = new THREE.Color(0x050302);
    this.scene.add(table(lit));

    const dialState: DialState = { slots: slotsOf(n), arrows: slotsOf(3), newest: uniform(-1), flash: uniform(0), fading: uniform(-1), fade: uniform(0), pointer: uniform(0) };
    dialState.arrows.values.fill(0);
    const face = dial(lit, n, accent, dialState);
    face.position.set(DIAL_X, 0, -DIAL_Y);
    this.scene.add(face);
    this.dialState = dialState;

    this.badges = [0, 1, 2].map((move) => {
      const state: BadgeState = { masks: Array.from({ length: this.digits }, () => uniform(0)), fill: uniform(0), read: uniform(0), pick: uniform(0), flash: uniform(0) };
      const { body, halo } = badge(lit, move, this.digits, accent, state);
      body.position.set(nodeX(move), 0, -nodeY(move));
      halo.position.copy(body.position);
      const glow: Tint = uniform(new THREE.Vector3());
      const lamp = light.emitter(new THREE.RingGeometry(BADGE + 0.05, BADGE + 0.1, 48).rotateX(-Math.PI / 2), glow);
      lamp.position.copy(body.position);
      this.scene.add(halo, body, lamp);
      return { ...state, body, halo, light: glow, flashed: -Infinity };
    });

    const railState: RailState = { values: slotsOf(RECORD + 1), slide: uniform(0), latest: uniform(-1), fresh: uniform(0) };
    const beads = rail(lit, railState);
    beads.position.set(0, 0, -BEADS_Y);
    this.scene.add(beads);
    this.railState = railState;

    const make = (mirror: 1 | -1, kind: 'glove' | 'automaton', order: number): Hand => {
      const pose = handPose(), mesh = hand(lit, pose, mirror, { kind, accent });
      mesh.renderOrder = order;
      mesh.position.set(CLASH_X - mirror * WRIST, 0, -CLASH_Y);
      this.scene.add(mesh);
      return { pose, mesh, mirror };
    };
    this.agent = make(1, 'glove', 7);
    this.bot = make(-1, 'automaton', 6);

    const burst = flash(this.flashTint, this.flashStrength);
    burst.position.set(CLASH_X, 0, -CLASH_Y);
    const clash = light.emitter(disc(0.16), this.clashLight);
    clash.position.set(CLASH_X, 0, -CLASH_Y);
    this.scene.add(burst, clash);

    this.tokenMesh = token(this.tokenMove);
    this.tokenMesh.visible = false;
    this.tokenLamp = light.emitter(disc(0.07), this.tokenLight);
    this.scene.add(this.tokenMesh, this.tokenLamp);
  }

  protected override drop(): void {
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private outcomeAt(frame: FrameUpdate, s: number): number {
    if (s < 1) return -1;
    const slot = s % this.heldStep.length;
    if (this.heldStep[slot] !== s) {
      const found = frame.frameAt(s);
      if (found.step !== s) return -1;
      this.heldCode[slot] = code(found.reward);
      this.heldStep[slot] = s;
    }
    return this.heldCode[slot]!;
  }

  private check(f: EnvFrame): void {
    if (!rpsPlayed(f)) return;
    const mine = rpsMine(f), theirs = rpsTheirs(f), outcome = rpsOutcome(mine, theirs);
    if (outcome !== Math.sign(f.reward)) throw new Error(`RockPaperScissors step ${f.step}: ${MOVE_NAMES[mine]} against ${MOVE_NAMES[theirs]} paid ${f.reward}`);
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const agent = this.agent, bot = this.bot, dialState = this.dialState, railState = this.railState, tokenMesh = this.tokenMesh, tokenLamp = this.tokenLamp;
    if (!agent || !bot || !dialState || !railState || !tokenMesh || !tokenLamp) return;
    pool.update(frame.dtReal, time);
    const at = (s: number): EnvFrame | null => {
      if (s < 0) return null;
      const found = frame.frameAt(s);
      return found.step === s ? found : null;
    };
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    const fast = this.pace > this.controlHz * 5;
    const i = frame.state.step, head = frame.frameAt(i), next = fast ? null : at(i + 1), t = next ? clamp01(frame.position - i) : 0;
    this.check(head);
    if (next) this.check(next);

    if (frame.scrubbed || this.lastPosition < 0) {
      pool.clear();
      this.heldStep.fill(-1);
      for (const b of this.badges) b.flashed = -Infinity;
    } else if (!fast) this.crossings(this.lastPosition, frame.position, pool, time, at);
    this.lastPosition = frame.position;

    const shown = next !== null && t >= REVEAL ? next : head, revealed = shown !== head;
    const age = revealed ? t - REVEAL : t + 1 - REVEAL;
    this.animateHands(head, next, t, fast, frame.dtReal);

    const ring = next && t >= LAND ? next : head;
    this.paintRing(ring, head, next, t);
    const planned = next && t >= LAND ? at(i + 2) : at(i + 1);
    this.paintBadges(ring, planned, t, next !== null, time);

    this.paintRail(frame, i, next !== null && t >= REVEAL, t, age);
    const played = rpsPlayed(shown), outcome = played ? code(shown.reward) : -1;
    const glow = played ? 0.38 + 1.4 * Math.exp(-age * 6) : 0;
    this.color.set(OUTCOME[Math.max(outcome, 0)]!);
    this.flashTint.value.set(this.color.r, this.color.g, this.color.b);
    this.flashStrength.value = fast ? (played ? 0.35 : 0) : glow;
    const lamp = (fast ? 0.35 : glow) * (outcome === 0 ? 9 : outcome === 1 ? 6 : 4) * (played ? 1 : 0);
    this.clashLight.value.set(this.color.r * lamp, this.color.g * lamp, this.color.b * lamp);
    const aura = (fast ? 0.35 : glow) * 0.9;
    agent.pose.glow.value.set(0, 0, 0);
    bot.pose.glow.value.set(0, 0, 0);
    if (outcome === 0) agent.pose.glow.value.set(this.color.r * aura, this.color.g * aura, this.color.b * aura);
    else if (outcome === 1) bot.pose.glow.value.set(this.color.r * aura, this.color.g * aura, this.color.b * aura);
    else if (outcome === 2) for (const h of [agent, bot]) h.pose.glow.value.set(this.color.r * aura * 0.35, this.color.g * aura * 0.35, this.color.b * aura * 0.35);

    this.flyToken(head, next, t, tokenMesh, tokenLamp);

    const ledger = this.ledger;
    ledger.step = shown.step;
    ledger.mine = rpsMine(shown);
    ledger.theirs = rpsTheirs(shown);
    ledger.reward = shown.reward;
  }

  private crossings(from: number, to: number, pool: CelebrationPool, time: number, at: (s: number) => EnvFrame | null): void {
    if (!(to > from) || to - from > 4) return;
    for (let k = Math.floor(from); k <= Math.floor(to); k++) {
      if (k + REVEAL > from && k + REVEAL <= to) this.reveal(k, pool, time, at);
      if (k + LAND > from && k + LAND <= to) this.land(k, time, at);
    }
  }

  private reveal(k: number, pool: CelebrationPool, time: number, at: (s: number) => EnvFrame | null): void {
    const before = at(k), after = at(k + 1);
    if (!before || !after) return;
    const clash = this.point.set(CLASH_X, 0, -CLASH_Y);
    for (const event of rpsDecoder.deriveEvents(before, after)) {
      if (event.type === 'goal') pool.burst(clash.setY(40), WIN, 0.7, time);
      else if (event.type === 'harm') pool.ripple(clash.setY(0), LOSS, 0.6, time);
    }
  }

  private land(k: number, time: number, at: (s: number) => EnvFrame | null): void {
    const after = at(k + 1), later = at(k + 2);
    if (!after || !later) return;
    for (const event of rpsDecoder.deriveEvents(after, later)) {
      if (event.type !== 'cue') continue;
      this.badges[event.value]!.flashed = time;
    }
  }

  private loserOf(f: EnvFrame): Hand | null {
    if (!rpsPlayed(f) || f.reward === 0) return null;
    return f.reward > 0 ? this.bot : this.agent;
  }

  private throwOf(f: EnvFrame, mine: boolean): Pose {
    return rpsPlayed(f) ? THROWS[mine ? rpsMine(f) : rpsTheirs(f)]! : FIST;
  }

  private animateHands(head: EnvFrame, next: EnvFrame | null, t: number, fast: boolean, dt: number): void {
    let lift = 0, lunge = 0;
    for (const h of [this.agent!, this.bot!]) {
      const mine = h === this.agent, from = this.throwOf(head, mine);
      if (fast) easePose(h.pose, from, 1 - Math.exp(-dt / 0.06));
      else if (!next || t < HOLD) setPose(h.pose, from, from, 0);
      else if (t < CLOSED) setPose(h.pose, from, FIST, smoothstep01(span(t, HOLD, CLOSED)));
      else if (t < SNAP) setPose(h.pose, FIST, FIST, 0);
      else {
        const to = this.throwOf(next, mine);
        setPose(h.pose, FIST, to, easeOut(span(t, SNAP, REVEAL)));
      }
    }
    let loser = this.loserOf(head), recoil = 1;
    if (next && !fast) {
      lift = t >= CLOSED && t < SNAP ? Math.sin(Math.PI * span(t, CLOSED, SNAP)) : 0;
      lunge = t >= SNAP ? Math.sin(Math.PI * span(t, SNAP, REVEAL + 0.14)) : 0;
      if (t >= REVEAL) {
        loser = this.loserOf(next);
        recoil = easeOut(span(t, REVEAL, REVEAL + 0.16));
      } else recoil = 1 - smoothstep01(span(t, HOLD, CLOSED));
    }
    for (const h of [this.agent!, this.bot!]) {
      h.pose.lift.value = lift;
      const grow = SIZE * (1 + 0.07 * lift), back = h === loser ? RECOIL * recoil : 0;
      h.mesh.scale.set(h.mirror * grow, 1, grow);
      h.mesh.position.x = CLASH_X - h.mirror * (WRIST - 0.16 * lunge + back);
    }
  }

  private paintRing(ring: EnvFrame, head: EnvFrame, next: EnvFrame | null, t: number): void {
    const state = this.dialState!, n = this.window, filled = Math.min(rpsFilled(ring), n);
    this.counts.fill(0);
    for (let k = 0; k < n; k++) {
      const move = k < filled ? rpsSlot(ring, k) : -1;
      if (move >= RPS_MOVES) throw new Error(`RockPaperScissors step ${ring.step} holds throw ${move} in slot ${k}`);
      state.slots.values[k] = move;
      if (move >= 0) this.counts[move]!++;
    }
    for (let m = 0; m < RPS_MOVES; m++) if (this.counts[m] !== rpsCount(ring, m)) throw new Error(`RockPaperScissors step ${ring.step} counts ${rpsCount(ring, m)} ${MOVE_NAMES[m]} but its window holds ${this.counts[m]}`);
    const cursor = rpsCursor(ring);
    state.newest.value = filled > 0 ? (cursor + n - 1) % n : -1;
    const written = next && t >= LAND ? t - LAND : t + 1 - LAND;
    state.flash.value = filled > 0 ? Math.exp(-written * 16) : 0;
    const from = rpsCursor(head);
    if (next && t >= LAND) state.pointer.value = ((from + smoothstep01(span(t, LAND, 1))) * TAU) / n;
    else state.pointer.value = (from * TAU) / n;
    const fading = next && t >= LAND - 0.12 && t < LAND && rpsFilled(head) >= n;
    state.fading.value = fading ? from : -1;
    state.fade.value = fading ? span(t, LAND - 0.12, LAND) : 0;
  }

  private paintBadges(ring: EnvFrame, planned: EnvFrame | null, t: number, moving: boolean, time: number): void {
    const n = this.window, arrows = this.dialState!.arrows.values;
    const pick = planned ? rpsTheirs(planned) : -1, read = pick >= 0 && rpsPlayed(ring) ? (pick + 2) % RPS_MOVES : -1;
    if (read >= 0) for (let m = 0; m < RPS_MOVES; m++) {
      if (rpsCount(ring, m) > rpsCount(ring, read)) throw new Error(`RockPaperScissors step ${ring.step}: the bot read ${MOVE_NAMES[read]} (${rpsCount(ring, read)}) though ${MOVE_NAMES[m]} counts ${rpsCount(ring, m)}`);
    }
    const commit = moving && t >= CLOSED && t < SNAP ? Math.sin(Math.PI * span(t, CLOSED, SNAP)) : 0;
    for (let m = 0; m < RPS_MOVES; m++) {
      const b = this.badges[m]!, count = rpsCount(ring, m);
      const value = Math.min(count, 999), hundreds = Math.floor(value / 100), tens = Math.floor(value / 10) % 10, ones = value % 10;
      const places = [hundreds, tens, ones].slice(3 - this.digits);
      places.forEach((digit, j) => {
        const leading = j < places.length - 1 && places.slice(0, j + 1).every((d) => d === 0);
        b.masks[j]!.value = leading ? 0 : DIGIT_MASKS[digit]!;
      });
      b.fill.value = count / n;
      b.read.value = m === read ? 1 : 0;
      b.pick.value = m === pick ? 1 + 0.8 * commit : 0;
      b.flash.value = Math.exp(-(time - b.flashed) * 4);
      arrows[m] = m === read ? 1 + 0.6 * commit : 0;
      this.color.set(this.accent).multiplyScalar(m === read ? 1.3 : 0);
      const cyan = m === pick ? 1.5 * (1 + commit) : 0, bot = this.cyan;
      b.light.value.set(this.color.r + bot.r * cyan, this.color.g + bot.g * cyan, this.color.b + bot.b * cyan);
    }
    this.ledger.pick = pick;
    this.ledger.read = read;
    this.ledger.readCount = read >= 0 ? rpsCount(ring, read) : 0;
  }

  private paintRail(frame: FrameUpdate, i: number, entering: boolean, t: number, age: number): void {
    const state = this.railState!, values = state.values.values, first = i - RECORD + 1;
    let won = 0, tied = 0, lost = 0, rounds = 0;
    for (let j = 0; j <= RECORD; j++) {
      const s = first + j, value = j < RECORD || entering ? this.outcomeAt(frame, s) : -1;
      values[j] = value;
      const counted = entering ? j >= 1 : j < RECORD;
      if (!counted || value < 0) continue;
      rounds++;
      if (value === 0) won++;
      else if (value === 1) lost++;
      else tied++;
    }
    state.slide.value = entering ? easeOut(span(t, REVEAL, REVEAL + SLIDE)) : 0;
    state.latest.value = entering ? RECORD : RECORD - 1;
    state.fresh.value = Math.exp(-age * 5);
    Object.assign(this.ledger, { won, tied, lost, rounds });
  }

  private flyToken(head: EnvFrame, next: EnvFrame | null, t: number, mesh: THREE.Mesh, lamp: THREE.Mesh): void {
    if (!next || t < REVEAL || t >= LAND || !rpsPlayed(next)) {
      mesh.visible = false;
      this.tokenLight.value.set(0, 0, 0);
      return;
    }
    const n = this.window, move = rpsMine(next), slot = rpsCursor(head);
    const hx = CLASH_X - WRIST + KNUCKLES * SIZE, hy = CLASH_Y + 0.6, sx = slotX(slot, n), sy = slotY(slot, n);
    const u = easeOut(span(t, FLY, LAND)), pop = easeOut(span(t, REVEAL, FLY));
    const cx = (hx + sx) / 2 - 0.7, cy = (hy + sy) / 2 + 0.9;
    const x = (1 - u) ** 2 * hx + 2 * (1 - u) * u * cx + u * u * sx, y = (1 - u) ** 2 * hy + 2 * (1 - u) * u * cy + u * u * sy;
    const tile = Math.min(R_TILE_OUT - R_TILE_IN, (TAU * (R_TILE_IN + R_TILE_OUT)) / 2 / n) / 0.72, size = (0.85 + (tile - 0.85) * u) * pop;
    mesh.visible = true;
    mesh.position.set(x, 0, -y);
    mesh.scale.set(size, 1, size);
    this.tokenMove.value = move;
    lamp.position.set(x, 0, -y);
    this.color.set(MOVE_GLOW[move]!).multiplyScalar(5 * pop);
    this.tokenLight.value.set(this.color.r, this.color.g, this.color.b);
  }
}

export default (() => new RockPaperScissorsScene()) satisfies SceneFactory;
