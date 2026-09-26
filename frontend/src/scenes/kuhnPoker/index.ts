import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { POKER_MINE, POKER_MY_CHIPS, POKER_NODE, POKER_SEAT, POKER_SEEN, POKER_THEIR_CHIPS, POKER_THEIRS, pokerDecoder } from '@sim/envs/poker';
import type { EnvFrame } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount } from '../../format';
import {
  BONE, CARD_H, CARD_QH, CARD_QW, CARD_W, CHIP_Q, COIN_Q, COPPER, GOLD, GRID, IVORY, LABEL_QH, LABEL_QW, LAMP_Q, PUCK_Q, RANK_NAMES, RECENT, SPOTS, TINT,
  atlas, cardMaterial, chipMaterial, coinMaterial, disc, flat, gridMaterial, labelMaterial, lampMaterial, palette, puckMaterial, tableMaterial, wordIndex, type Level, type Tally, type Word,
} from './art';

type N<T extends string> = THREE.Node<T>;
type Act = 'check' | 'bet' | 'call' | 'fold';

const FOLD = 0;
const CALL = 1;
const RAISE = 2;
const LAST_MOVE = 26;
const LAST_ENDED = 28;
const AGENT = 0;
const THEM = 1;
const SPLIT = 2;
const SEATS = [AGENT, THEM] as const;
const side = (who: number): number => (who === AGENT ? 1 : -1);
const ACT_WORD: Record<Act, Word> = { check: 'CHECK', bet: 'BET', call: 'CALL', fold: 'FOLD' };

const CARD_Z = 2.2;
const CHIP_Z = 0.9;
const STACK = { x: 1.12, z: 3.1 };
const TAG_X = 1.7;
const SIDE_X = 1.62;
const FIRST_X = -1.6;
const CHIP_GAP = 0.62;
const DECK = { x: 4.35, z: -1.35, angle: 0 };
const MUCK = { x: 3.55, z: -1.1 };
const muckAngle = (who: number): number => (who === AGENT ? -0.35 : 0.28);
const LAST_X = [3.97, 4.73] as const;
const LAST_Z = 1.45;
const LAST_SCALE = 0.6;
const RESULT = { x: 4.35, z: 2.55 };
const GRID_AT = { x: -4.5, z: 0 };

const REST = 0.1;
const TAIL = 0.03;
const MAX_UNIT = 0.2;
const UNITS = { act: 1, reply: 1.15, reveal: 0.8, award: 0.95, deal: 1.3 } as const;
const WINDOW = 1000;
const FAST = 5;

const ease = smoothstep01;
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const span = (e: number, from: number, length: number) => clamp01((e - from) / length);
const bold = (act: Act) => act === 'bet' || act === 'call';
const signed = (value: number) => `${value > 0 ? '+' : value < 0 ? '−' : ''}${formatCount(Math.abs(value))}`;

interface Table { hand: number; seat: number; mine: number; theirs: number; node: number; seen: number; my: number; their: number }
interface Hand { mine: number; theirs: number; shown: boolean; reward: number | null }
interface View { mine: number; theirs: number; up: boolean; out: [boolean, boolean]; chips: [number, number]; acts: [Act | null, Act | null]; coin: number; first: number; last: Hand | null }

type Beat = { t0: number; t1: number } & (
  | { kind: 'act'; who: typeof AGENT | typeof THEM; act: Act; from: number; to: number }
  | { kind: 'reveal' }
  | { kind: 'award' }
  | { kind: 'deal'; mine: number; theirs: number; first: number }
);

interface Story { step: number; start: View; beats: Beat[]; end: number; reward: number; winner: number; finished: Hand | null }

const slots = new Float32Array(pokerDecoder.slots);
const probe: EnvFrame = { step: 0, done: 0, words: new Uint32Array(2), view: new Uint32Array(0), observations: new Float32Array(0), actions: new Float32Array(0), reward: 0, reset: false };

function read(hand: number, table: number): Table {
  probe.words[0] = hand;
  probe.words[1] = table;
  pokerDecoder.decode(probe, slots);
  return { hand, seat: slots[POKER_SEAT]!, mine: slots[POKER_MINE]!, theirs: slots[POKER_THEIRS]!, node: slots[POKER_NODE]!, seen: slots[POKER_SEEN]!, my: slots[POKER_MY_CHIPS]!, their: slots[POKER_THEIR_CHIPS]! };
}

const tableOf = (frame: EnvFrame): Table => read(frame.words[0]!, frame.words[1]!);
const finishedOf = (frame: EnvFrame): Table => read(frame.words[0]! - 1, frame.words[2]!);
const ended = (frame: EnvFrame): boolean => ((frame.words[2]! >>> LAST_ENDED) & 1) === 1;
const finalMove = (frame: EnvFrame): number => (frame.words[2]! >>> LAST_MOVE) & 3;
const copy = (view: View): View => ({ ...view, out: [...view.out], chips: [...view.chips], acts: [...view.acts] });

class Batch {
  count = 0;
  private readonly matrix = new THREE.Matrix4();
  private readonly scale = new THREE.Vector3();

  constructor(readonly mesh: THREE.InstancedMesh, readonly attrs: THREE.InstancedBufferAttribute[], private readonly capacity: number) {
    mesh.frustumCulled = false;
    mesh.count = 0;
  }

  place(x: number, z: number, sx: number, sz: number, angle: number): number {
    if (this.count >= this.capacity) throw new Error(`${this.mesh.name} holds ${this.capacity} instances`);
    const i = this.count++;
    this.matrix.makeRotationY(angle).scale(this.scale.set(Math.max(sx, 1e-3), 1, sz)).setPosition(x, 0, z);
    this.mesh.setMatrixAt(i, this.matrix);
    return i;
  }

  flush(): void {
    this.mesh.count = this.count;
    this.mesh.instanceMatrix.needsUpdate = true;
    for (const attr of this.attrs) attr.needsUpdate = true;
    this.count = 0;
  }
}

interface Lamp { mesh: THREE.Mesh; tint: THREE.Color }

class KuhnPokerScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 1;
  protected readonly stageName = 'kuhnPoker';

  private suits = 1;
  private ante = 1;
  private actions = 2;
  private stack = 2;
  private accent = 0;
  private cards: Batch | null = null;
  private chips: Batch | null = null;
  private flying: Batch | null = null;
  private labels: Batch | null = null;
  private coin: THREE.Mesh | null = null;
  private puck: THREE.Mesh | null = null;
  private lamp: THREE.Mesh | null = null;
  private grid: THREE.Mesh | null = null;
  private tally: Tally | null = null;
  private readonly coinFace = uniform(-1);
  private readonly coinGlow = uniform(0);
  private readonly lampLevel = uniform(1);
  private lampLight: Lamp | null = null;
  private coinLight: Lamp | null = null;
  private readonly disposables: { dispose(): void }[] = [];

  private readonly held = new Int32Array(WINDOW).fill(-1);
  private readonly spot = new Int8Array(WINDOW);
  private readonly rank = new Int8Array(WINDOW);
  private readonly pressed = new Uint8Array(WINDOW);
  private readonly paid = new Float32Array(WINDOW);
  private readonly over = new Uint8Array(WINDOW);
  private tallied = -1;
  private dirty = true;

  private story: Story | null = null;
  private fired = { step: -1, beat: -1 };
  private pace = 0;
  private lastPosition = 0;
  private readonly ledger = { hand: 0, first: 0, rank: 0, line: '—', my: 0, their: 0, last: '—', net: 0, hands: 0, wins: 0 };

  override legend: LegendSpec | null = null;

  private legendFor(): LegendSpec {
    return {
    title: 'KUHN POKER',
    rows: [
      { key: 'hand', label: 'hand', tint: this.accent, value: () => `#${formatCount(this.ledger.hand + 1)} · you act ${this.ledger.first === AGENT ? 'first' : 'second'}` },
      { key: 'card', label: 'your card', tint: IVORY, value: () => RANK_NAMES[this.ledger.rank] ?? '—' },
      { key: 'line', label: 'so far', tint: BONE, value: () => this.ledger.line },
      { key: 'pot', label: 'in the pot', tint: GOLD, value: () => `you ${this.ledger.my} · them ${this.ledger.their}` },
      { key: 'last', label: 'last hand', tint: GOLD, value: () => this.ledger.last },
      { key: 'net', label: 'net', tint: this.accent, value: () => `${signed(this.ledger.net)} over ${formatCount(this.ledger.hands)} hands` },
    ],
    note: () => `coin-flip opponent · grid: your bet/call rate over ${formatCount(WINDOW)} decisions`,
    bar: () => (this.ledger.hands > 0 ? this.ledger.wins / this.ledger.hands : 0),
    };
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.suits = parameter(env, 'suits');
    this.ante = parameter(env, 'ante');
    this.actions = parameter(env, 'actions');
    this.stack = this.ante + parameter(env, 'firstRaise') * parameter(env, 'raiseCap');
    if (parameter(env, 'rounds') !== 1 || parameter(env, 'opponent') !== 0 || this.actions !== 2) throw new Error(`${env.envId} is not the one-round pass-or-bet game against a coin-flipping opponent that this table seats`);
    this.accent = this.accentOf(ctx);
    this.legend = this.legendFor();
    return new THREE.Box2(new THREE.Vector2(-8, -4.5), new THREE.Vector2(8, 4.5));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff0e0).multiplyScalar(0.95);
  }

  protected override poolOptions() {
    return { burstSize: 44, sighRadius: 1.2, sighHeight: 0.8 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.debugDraw(frame, kit) };
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    this.scene.background = new THREE.Color(0x070506);
    pool.group.traverse((object) => { object.renderOrder = 20; });
    const map = atlas(), tints = palette(this.accent);
    this.disposables.push(map);

    const prints = [
      ...SEATS.map((who) => ({ x: 0, z: side(who) * CARD_Z, w: CARD_W + 0.2, h: CARD_H + 0.2, r: 0.14 })),
      ...SEATS.map((who) => ({ x: 0, z: side(who) * CHIP_Z, w: this.stack * CHIP_GAP + 0.3, h: 0.78, r: 0.39 })),
      { x: DECK.x, z: DECK.z, w: CARD_W + 0.34, h: CARD_H + 0.34, r: 0.16 },
    ];
    const captions = [{ word: 'KUHN POKER' as const, x: 0, z: 0, cap: 0.17 }, { word: 'LAST HAND' as const, x: RESULT.x, z: LAST_Z - CARD_H * LAST_SCALE / 2 - 0.3, cap: 0.1 }];
    const table = new THREE.Mesh(flat(80, 80), tableMaterial(light, map, prints, captions));
    table.renderOrder = 0;
    table.frustumCulled = false;
    this.scene.add(table);

    const grid = gridMaterial(light, map, this.accent);
    this.tally = grid.tally;
    this.grid = new THREE.Mesh(flat(GRID.w, GRID.h), grid.material);
    this.grid.position.set(GRID_AT.x, 0, GRID_AT.z);
    this.grid.renderOrder = 1;
    this.scene.add(this.grid);

    const batch = (name: string, width: number, depth: number, capacity: number, slotsCount: number, material: (attrs: THREE.InstancedBufferAttribute[]) => THREE.Material, order: number): Batch => {
      const attrs = Array.from({ length: slotsCount }, () => new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage));
      const mesh = instancedMesh(flat(width, depth), material(attrs), capacity);
      mesh.name = name;
      mesh.renderOrder = order;
      this.scene.add(mesh);
      return new Batch(mesh, attrs, capacity);
    };
    this.chips = batch('kuhnChips', CHIP_Q, CHIP_Q, 2 * this.stack, 1, ([look]) => chipMaterial(light, this.accent, look!), 3);
    this.flying = batch('kuhnFlyingChips', CHIP_Q, CHIP_Q, 4 * this.stack, 1, ([look]) => chipMaterial(light, this.accent, look!), 5);
    this.cards = batch('kuhnCards', CARD_QW, CARD_QH, 12, 2, ([look, pose]) => cardMaterial(light, map, { look: look!, pose: pose! }, tints), 4);
    this.labels = batch('kuhnLabels', LABEL_QW, LABEL_QH, 8, 1, ([look]) => labelMaterial(map, look!, tints), 6);

    this.coin = new THREE.Mesh(flat(COIN_Q, COIN_Q), coinMaterial(light, map, this.coinFace as Level, this.coinGlow as Level));
    this.coin.position.set(SIDE_X, 0, -CARD_Z);
    this.puck = new THREE.Mesh(flat(PUCK_Q, PUCK_Q), puckMaterial(light, map));
    this.lamp = new THREE.Mesh(flat(LAMP_Q, LAMP_Q), lampMaterial(this.accent, this.lampLevel as Level));
    this.lamp.position.set(SIDE_X, 0, CARD_Z);
    for (const mesh of [this.coin, this.puck, this.lamp]) {
      mesh.renderOrder = 5;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
    }

    const lamp = (radius: number): Lamp => {
      const tint = new THREE.Color(0, 0, 0), mesh = light.emitter(disc(radius), uniform(tint) as unknown as N<'vec3'>);
      this.scene.add(mesh);
      return { mesh, tint };
    };
    this.lampLight = lamp(0.14);
    this.lampLight.mesh.position.set(SIDE_X, 0.02, CARD_Z);
    this.coinLight = lamp(0.2);
    this.coinLight.mesh.position.set(SIDE_X, 0.02, -CARD_Z);
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    });
  }

  private available(frame: FrameUpdate, step: number): EnvFrame | null {
    if (step < 0) return null;
    const found = frame.frameAt(step);
    return found.step === step ? found : null;
  }

  private pressOf(frame: EnvFrame): number {
    const choice = frame.actions[0]!;
    return choice >= 0 && choice < this.actions ? Math.trunc(choice) : 0;
  }

  private viewOf(frame: EnvFrame, at: (s: number) => EnvFrame | null): View {
    const t = tableOf(frame), theirs: Act | null = t.seen === RAISE ? 'bet' : t.seen === CALL ? 'check' : null;
    const mine: Act | null = t.seat === AGENT && theirs !== null ? (t.my > this.ante ? 'bet' : 'check') : null;
    return { mine: t.mine, theirs: t.theirs, up: false, out: [false, false], chips: [t.my, t.their], acts: [mine, theirs], coin: theirs === null ? -1 : theirs === 'bet' ? 1 : 0, first: t.seat, last: this.lastHand(frame, at) };
  }

  private lastHand(frame: EnvFrame, at: (s: number) => EnvFrame | null): Hand | null {
    if (!ended(frame)) return null;
    const l = finishedOf(frame), hand = frame.words[0]!;
    let opener: EnvFrame | null = null;
    for (let s = frame.step; s >= 0 && s > frame.step - 64; s--) {
      const f = at(s);
      if (!f || f.words[0] !== hand) break;
      if (f.done) {
        opener = f;
        break;
      }
    }
    return { mine: l.mine, theirs: l.theirs, shown: finalMove(frame) === CALL, reward: opener ? opener.reward : null };
  }

  private storyOf(before: EnvFrame, after: EnvFrame, at: (s: number) => EnvFrame | null): Story {
    const step = before.step, b = tableOf(before), owed = b.their - b.my, beats: Beat[] = [];
    const act = (who: typeof AGENT | typeof THEM, move: Act, from: number, to: number): Beat => ({ kind: 'act', who, act: move, from, to, t0: 0, t1: 0 });
    let reward = 0, winner = -1, finished: Hand | null = null;
    if (!after.done) {
      const a = tableOf(after);
      if (a.hand !== b.hand) throw new Error(`KuhnPoker step ${step + 1} moves to hand ${a.hand} without ending hand ${b.hand}`);
      beats.push(act(AGENT, a.my > b.my ? (owed > 0 ? 'call' : 'bet') : 'check', b.my, a.my));
      if (a.seen !== 0) beats.push(act(THEM, a.seen === RAISE ? 'bet' : 'check', b.their, a.their));
    } else {
      const a = tableOf(after), l = finishedOf(after), move = finalMove(after);
      if (!ended(after) || a.hand !== b.hand + 1 || l.mine !== b.mine || l.theirs !== b.theirs) throw new Error(`KuhnPoker step ${step + 1} ends a hand the table at step ${step} was not playing`);
      let final: [number, number];
      if (l.node === b.node) {
        const mine: Act = move === FOLD ? 'fold' : owed > 0 ? 'call' : 'check', my = mine === 'call' ? b.their : b.my;
        beats.push(act(AGENT, mine, b.my, my));
        final = [my, b.their];
      } else {
        beats.push(act(AGENT, l.my > b.my ? (owed > 0 ? 'call' : 'bet') : 'check', b.my, l.my));
        const theirs: Act = move === FOLD ? 'fold' : l.my > l.their ? 'call' : 'check', their = theirs === 'call' ? l.my : l.their;
        beats.push(act(THEM, theirs, l.their, their));
        final = [l.my, their];
      }
      reward = after.reward;
      if (reward > 0 ? reward !== final[1] : reward < 0 ? reward !== -final[0] : final[0] !== final[1]) throw new Error(`KuhnPoker step ${step + 1} pays ${reward} from a pot of ${final[0]} and ${final[1]}`);
      winner = reward > 0 ? AGENT : reward < 0 ? THEM : SPLIT;
      finished = { mine: b.mine, theirs: b.theirs, shown: move === CALL, reward };
      if (move === CALL) beats.push({ kind: 'reveal', t0: 0, t1: 0 });
      beats.push({ kind: 'award', t0: 0, t1: 0 });
      beats.push({ kind: 'deal', mine: a.mine, theirs: a.theirs, first: a.seat, t0: 0, t1: 0 });
      if (a.seen !== 0) beats.push(act(THEM, a.seen === RAISE ? 'bet' : 'check', this.ante, a.their));
    }
    const first = beats[0]!;
    if (first.kind !== 'act' || bold(first.act) !== (this.pressOf(after) === 1)) throw new Error(`KuhnPoker step ${step + 1} presses ${this.pressOf(after)} yet the table reads the agent's move otherwise`);
    const units = beats.map((beat) => (beat.kind === 'act' ? (beat.who === THEM ? UNITS.reply : UNITS.act) : UNITS[beat.kind]));
    const unit = Math.min(MAX_UNIT, (1 - REST - TAIL) / units.reduce((sum, value) => sum + value, 0));
    let t = REST;
    beats.forEach((beat, k) => {
      beat.t0 = t;
      t += units[k]! * unit;
      beat.t1 = t;
    });
    return { step, start: this.viewOf(before, at), beats, end: t, reward, winner, finished };
  }

  private apply(view: View, beat: Beat, story: Story): void {
    if (beat.kind === 'act') {
      view.chips[beat.who] = beat.to;
      view.acts[beat.who] = beat.act;
      if (beat.who === THEM) view.coin = bold(beat.act) ? 1 : 0;
      if (beat.act === 'fold') view.out[beat.who] = true;
    } else if (beat.kind === 'reveal') view.up = true;
    else if (beat.kind === 'award') {
      view.chips = [0, 0];
      view.acts = [null, null];
      view.last = story.finished;
    } else {
      view.mine = beat.mine;
      view.theirs = beat.theirs;
      view.up = false;
      view.out = [false, false];
      view.chips = [this.ante, this.ante];
      view.first = beat.first;
      view.coin = -1;
    }
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    if (!frame.scrubbed && frame.dtReal > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / frame.dtReal - this.pace) * (1 - Math.exp(-4 * frame.dtReal));
    this.lastPosition = frame.position;
    const at = (s: number) => this.available(frame, s);
    const fast = this.pace > this.controlHz * FAST, i = frame.state.step, now = frame.frameAt(i), next = fast ? null : at(i + 1);
    const u = next ? clamp01(frame.position - i) : 0;
    if (next && this.story?.step !== i) this.story = this.storyOf(now, next, at);
    const story = next ? this.story : null;
    this.celebrate(story, u, frame.scrubbed || fast, pool, time);
    if (story) this.play(story, u, time);
    else this.paint(this.viewOf(now, at), null, null, 0, 1, time);
    this.cards!.flush();
    this.chips!.flush();
    this.flying!.flush();
    this.labels!.flush();
    this.sync(i, at);
    this.note(now, at);
  }

  private play(story: Story, u: number, time: number): void {
    const view = copy(story.start);
    const lamp = u < REST ? 1 - smoothstep01((u - REST + 0.05) / 0.05) : smoothstep01((u - story.end) / 0.05);
    for (const beat of story.beats) {
      if (u >= beat.t1) {
        this.apply(view, beat, story);
        continue;
      }
      if (u < beat.t0) break;
      this.paint(view, story, beat, (u - beat.t0) / (beat.t1 - beat.t0), lamp, time);
      return;
    }
    this.paint(view, story, null, 0, lamp, time);
  }

  private card(x: number, z: number, card: number, face: number, turn: number, alpha: number, glow: number, tint: number, lift: number, dim: number, scale: number, angle: number): void {
    const cards = this.cards!, i = cards.place(x, z, turn * scale * (1 + 0.06 * lift), scale * (1 + 0.06 * lift), angle);
    cards.attrs[0]!.setXYZW(i, Math.floor(card / this.suits), face, glow, alpha);
    cards.attrs[1]!.setXYZW(i, lift, tint, dim, 0);
  }

  private chip(x: number, z: number, owner: number, alpha: number, lift: number, glow: number): void {
    if (alpha <= 0.001) return;
    const chips = lift > 0 ? this.flying! : this.chips!, i = chips.place(x, z, 1 + 0.12 * lift, 1 + 0.12 * lift, 0);
    chips.attrs[0]!.setXYZW(i, owner, alpha, lift, glow);
  }

  private label(x: number, z: number, kind: number, value: number, tint: number, alpha: number, scale: number): void {
    if (alpha <= 0.001) return;
    const labels = this.labels!, i = labels.place(x, z, scale, scale, 0);
    labels.attrs[0]!.setXYZW(i, kind, value, tint, alpha);
  }

  private slotX(k: number): number {
    return (k - (this.stack - 1) / 2) * CHIP_GAP;
  }

  private paint(view: View, story: Story | null, beat: Beat | null, e: number, lamp: number, time: number): void {
    const dealing = beat?.kind === 'deal' ? beat : null, collect = dealing ? ease(span(e, 0, 0.3)) : 0;
    const acting = beat?.kind === 'act' ? beat : null;
    const glowing = beat?.kind === 'reveal' ? ease(span(e, 0.55, 0.3)) : beat?.kind === 'award' ? 1 : dealing ? 1 - collect : 0;
    const winner = story?.winner ?? -1;
    this.card(DECK.x, DECK.z, 0, 0, 1, 1, 0, TINT.print, 0, 0, 1, DECK.angle);

    for (const who of SEATS) {
      const card = who === AGENT ? view.mine : view.theirs;
      let x = 0, z = side(who) * CARD_Z, face = who === AGENT || view.up ? 1 : 0, turn = 1, dim = 0, lift = 0, alpha = 1, angle = 0;
      let glow = who === AGENT && !beat ? 0.3 * lamp : 0, tint: number = TINT.agent;
      if (view.out[who]) {
        x = MUCK.x;
        z = MUCK.z + 0.18 * who;
        face = 0;
        dim = 0.45;
        angle = muckAngle(who);
      }
      if (acting && acting.who === who && acting.act === 'fold') {
        const start = who === AGENT ? 0 : 0.45, f = ease(span(e, start, 0.5)), flip = span(e, start, 0.22);
        x = lerp(0, MUCK.x, f);
        z = lerp(side(who) * CARD_Z, MUCK.z + 0.18 * who, f);
        dim = 0.45 * f;
        lift = Math.sin(Math.PI * f) * 0.6;
        angle = muckAngle(who) * f;
        if (who === AGENT) {
          turn = Math.abs(Math.cos(Math.PI * flip));
          face = flip < 0.5 ? 1 : 0;
        }
      }
      if (beat?.kind === 'reveal' && who === THEM) {
        turn = Math.abs(Math.cos(Math.PI * e));
        face = e >= 0.5 ? 1 : 0;
        lift = Math.sin(Math.PI * e) * 0.5;
      }
      if (winner >= 0 && glowing > 0 && !view.out[who]) {
        if (winner === who || winner === SPLIT) {
          glow = glowing;
          tint = who === AGENT ? TINT.win : TINT.them;
        } else dim = Math.max(dim, 0.3 * glowing);
      }
      if (dealing) {
        const flip = span(collect, 0, 0.4);
        if (face === 1) {
          turn = Math.abs(Math.cos(Math.PI * flip));
          face = flip < 0.5 ? 1 : 0;
        }
        x = lerp(x, DECK.x, collect);
        z = lerp(z, DECK.z, collect);
        angle = lerp(angle, DECK.angle, collect);
        lift = Math.sin(Math.PI * collect) * 0.4;
        alpha = 1 - ease(span(collect, 0.7, 0.3));
      }
      if (alpha > 0.001) this.card(x, z, card, face, turn, alpha, glow, tint, lift, dim, 1, angle);
    }

    if (dealing) {
      for (const who of SEATS) {
        const f = ease(span(e, who === AGENT ? 0.4 : 0.5, 0.34));
        if (f <= 0) continue;
        const flip = who === AGENT ? span(e, 0.8, 0.2) : 0, card = who === AGENT ? dealing.mine : dealing.theirs;
        this.card(lerp(DECK.x, 0, f), lerp(DECK.z, side(who) * CARD_Z, f), card, flip >= 0.5 ? 1 : 0, Math.abs(Math.cos(Math.PI * flip)), 1, 0, TINT.agent, Math.sin(Math.PI * f) * 0.6, 0, 1, lerp(DECK.angle, 0, f));
      }
    }

    for (const who of SEATS) {
      const count = view.chips[who];
      if (count > this.stack) throw new Error(`KuhnPoker seats ${count} chips in front of one player, more than the ${this.stack} a single bet allows`);
      for (let k = 0; k < count; k++) {
        let x = this.slotX(k), z = side(who) * CHIP_Z, alpha = 1, lift = 0, glow = 0;
        if (beat?.kind === 'award' && story) {
          const target = story.winner === SPLIT ? who : story.winner, f = ease(span(e, 0.08, 0.62));
          x = lerp(x, STACK.x, f);
          z = lerp(z, side(target) * STACK.z, f);
          alpha = 1 - ease(span(f, 0.55, 0.45));
          lift = Math.sin(Math.PI * f) * 0.5;
          glow = target === AGENT && story.reward > 0 ? Math.sin(Math.PI * f) : 0;
        }
        this.chip(x, z, who, alpha, lift, glow);
      }
      if (acting && acting.who === who && acting.to > acting.from) {
        const f = ease(span(e, who === AGENT ? 0.04 : 0.45, 0.5));
        if (f > 0) for (let k = acting.from; k < acting.to; k++) this.chip(lerp(STACK.x, this.slotX(k), f), lerp(side(who) * STACK.z, side(who) * CHIP_Z, f), who, ease(f / 0.2), Math.sin(Math.PI * f) * 0.6, 0);
      }
      if (dealing) {
        const f = ease(span(e, 0.22, 0.35));
        if (f > 0) for (let k = 0; k < this.ante; k++) this.chip(lerp(STACK.x, this.slotX(k), f), lerp(side(who) * STACK.z, side(who) * CHIP_Z, f), who, ease(f / 0.2), Math.sin(Math.PI * f) * 0.6, 0);
      }
    }

    for (const who of SEATS) {
      let act = view.acts[who], alpha = 1, scale = 1;
      if (acting && acting.who === who) {
        const pop = span(e, who === AGENT ? 0.08 : 0.48, 0.22);
        if (pop > 0) {
          act = acting.act;
          alpha = ease(pop);
          scale = 1 + 0.3 * (1 - ease(pop));
        }
      }
      if (beat?.kind === 'award') alpha *= 1 - ease(span(e, 0.35, 0.4));
      if (act) this.label(TAG_X, side(who) * CHIP_Z, wordIndex(ACT_WORD[act]), 0, who === AGENT ? TINT.agent : TINT.them, alpha, scale);
    }

    let coinFace = view.coin, coinTurn = 1, coinScale = 1, coinGlow = 0;
    if (acting && acting.who === THEM) {
      const f = span(e, 0, 0.45), landed = bold(acting.act) ? 1 : 0, segment = Math.floor(3 * f + 0.5);
      coinTurn = Math.abs(Math.cos(3 * Math.PI * f));
      coinFace = f < 1 / 6 ? view.coin : (3 - segment) % 2 === 0 ? landed : 1 - landed;
      coinScale = 1 + 0.22 * Math.sin(Math.PI * f);
      coinGlow = f < 1 ? 0.9 : 0.9 * (1 - span(e, 0.45, 0.45));
    }
    if (dealing && e > 0.3) coinFace = -1;
    this.coinFace.value = coinFace;
    this.coinGlow.value = coinGlow;
    this.coin!.scale.set(Math.max(coinTurn, 1e-3) * coinScale, 1, coinScale);
    this.coinLight!.tint.set(COPPER).multiplyScalar(4 * coinGlow);

    let puckZ = side(view.first) * CARD_Z;
    if (dealing && dealing.first !== view.first) puckZ = lerp(puckZ, side(dealing.first) * CARD_Z, ease(span(e, 0.2, 0.35)));
    this.puck!.position.set(FIRST_X, 0, puckZ);

    const level = 0.22 + 0.78 * lamp * (0.9 + 0.1 * Math.sin(time * 2.6));
    this.lampLevel.value = level;
    this.lampLight!.tint.set(this.accent).multiplyScalar(1 + 5 * level * lamp);

    if (story && story.finished) {
      let alpha = 0, scale = 1;
      if (beat?.kind === 'award') {
        const pop = span(e, 0.1, 0.25);
        alpha = ease(pop);
        scale = 1 + 0.4 * (1 - ease(pop));
      } else if (dealing) alpha = 1 - ease(span(e, 0.2, 0.25));
      this.label(0, 0, -1, story.reward, story.reward > 0 ? TINT.win : TINT.loss, alpha, 1.35 * scale);
    }

    let last = view.last, pop = 0;
    if (beat?.kind === 'award' && story && e >= 0.75) {
      last = story.finished;
      pop = 1 - span(e, 0.75, 0.25);
    }
    if (last) {
      const scale = LAST_SCALE * (1 + 0.12 * pop);
      this.card(LAST_X[0], LAST_Z, last.mine, 1, 1, 1, 0, TINT.agent, 0, 0, scale, 0);
      this.card(LAST_X[1], LAST_Z, last.theirs, last.shown ? 1 : 0, 1, 1, 0, TINT.them, 0, 0, scale, 0);
      if (last.reward !== null) this.label(RESULT.x, RESULT.z, -1, last.reward, last.reward > 0 ? TINT.win : TINT.loss, 1, 1 + 0.25 * pop);
    }
  }

  private celebrate(story: Story | null, u: number, quiet: boolean, pool: Stage2D['pool'], time: number): void {
    if (!story) return;
    if (this.fired.step !== story.step) this.fired = { step: story.step, beat: -1 };
    story.beats.forEach((beat, k) => {
      if (k <= this.fired.beat || u < beat.t0) return;
      this.fired.beat = k;
      if (quiet) return;
      if (beat.kind !== 'award') return;
      if (story.reward > 0) pool.burst({ x: 0, y: 38, z: CHIP_Z }, TOKENS.celebrationGold, 0.8, time);
      else if (story.reward < 0) pool.sigh(0.55, { x: 0, y: 0, z: CARD_Z }, time);
    });
  }

  private sync(i: number, at: (s: number) => EnvFrame | null): void {
    for (let s = Math.max(1, i - WINDOW + 1); s <= i; s++) {
      const slot = s % WINDOW;
      if (this.held[slot] === s) continue;
      const now = at(s), before = at(s - 1);
      if (!now || !before) continue;
      const b = tableOf(before), owed = b.their > b.my;
      this.spot[slot] = b.seat === AGENT ? (owed ? 1 : 0) : owed ? 3 : 2;
      this.rank[slot] = Math.floor(b.mine / this.suits);
      this.pressed[slot] = this.pressOf(now) === 1 ? 1 : 0;
      this.paid[slot] = now.reward;
      this.over[slot] = now.done ? 1 : 0;
      this.held[slot] = s;
      this.dirty = true;
    }
    if (!this.dirty && this.tallied === i) return;
    this.dirty = false;
    this.tallied = i;
    const tally = this.tally!, cells = SPOTS * RANK_NAMES.length, seen = new Array<number>(cells).fill(0), bets = new Array<number>(cells).fill(0);
    let hands = 0, net = 0, wins = 0, recent = RECENT;
    tally.recent.fill(0);
    for (let s = i; s >= Math.max(1, i - WINDOW + 1); s--) {
      const slot = s % WINDOW;
      if (this.held[slot] !== s) continue;
      const cell = this.rank[slot]! * SPOTS + this.spot[slot]!;
      if (cell < 0 || cell >= cells) throw new Error(`KuhnPoker step ${s} holds a card of rank ${this.rank[slot]} the table does not name`);
      seen[cell]!++;
      bets[cell]! += this.pressed[slot]!;
      if (!this.over[slot]) continue;
      hands++;
      net += this.paid[slot]!;
      if (this.paid[slot]! > 0) wins++;
      if (recent > 0) tally.recent[--recent] = this.paid[slot]!;
    }
    for (let c = 0; c < cells; c++) {
      tally.rates[c] = seen[c]! > 0 ? bets[c]! / seen[c]! : 0;
      tally.weights[c] = seen[c]! / (seen[c]! + 4);
    }
    this.ledger.hands = hands;
    this.ledger.net = net;
    this.ledger.wins = wins;
  }

  private note(now: EnvFrame, at: (s: number) => EnvFrame | null): void {
    const view = this.viewOf(now, at), t = tableOf(now), ledger = this.ledger;
    ledger.hand = t.hand;
    ledger.first = t.seat;
    ledger.rank = Math.floor(t.mine / this.suits);
    ledger.my = t.my;
    ledger.their = t.their;
    const [mine, theirs] = view.acts;
    ledger.line = mine && theirs ? `you ${mine} · they ${theirs}` : theirs ? `they ${theirs}` : 'nothing yet';
    const last = view.last;
    ledger.last = !last ? '—' : `${last.reward === null ? '' : `${signed(last.reward)} · `}${last.shown ? `${RANK_NAMES[Math.floor(last.mine / this.suits)]} vs ${RANK_NAMES[Math.floor(last.theirs / this.suits)]}` : last.reward !== null && last.reward > 0 ? 'they folded' : 'you folded'}`;
  }

  private debugDraw(frame: FrameUpdate, kit: DebugKit): void {
    const now = frame.frameAt(frame.state.step), t = tableOf(now), view = this.viewOf(now, (s) => this.available(frame, s));
    const ranks = kit.marks('ranks', 12), chips = kit.marks('chips', 2 * this.stack), seat = kit.marks('seat', 2);
    const pips = (card: number, x: number, z: number, color: number) => {
      const rank = Math.floor(card / this.suits);
      for (let k = 0; k <= rank; k++) ranks.set(x + (k - rank / 2) * 0.2, 0.1, z, 0.07, color);
    };
    pips(t.mine, 0, CARD_Z + CARD_H / 2 + 0.18, TOKENS.uiCyan);
    pips(t.theirs, 0, -CARD_Z - CARD_H / 2 - 0.18, TOKENS.celebrationGold);
    if (view.last) {
      pips(view.last.mine, LAST_X[0], LAST_Z - CARD_H * LAST_SCALE / 2 - 0.14, TOKENS.uiCyan);
      pips(view.last.theirs, LAST_X[1], LAST_Z - CARD_H * LAST_SCALE / 2 - 0.14, TOKENS.celebrationGold);
    }
    for (const who of SEATS) for (let k = 0; k < (who === AGENT ? t.my : t.their); k++) chips.set(this.slotX(k), 0.1, side(who) * CHIP_Z, 0.1, who === AGENT ? TOKENS.uiCyan : TOKENS.celebrationGold);
    seat.set(FIRST_X, 0.1, side(t.seat) * CARD_Z, 0.12, TOKENS.uiCyan);
    if (t.seen !== 0) seat.set(SIDE_X, 0.1, -CARD_Z, t.seen === RAISE ? 0.16 : 0.09, TOKENS.celebrationGold);
    const tally = this.tally!, left = GRID_AT.x - GRID.w / 2 + GRID.pad + GRID.labels;
    for (let row = 0; row < RANK_NAMES.length; row++) for (let spot = 0; spot < SPOTS; spot++) {
      const cell = (RANK_NAMES.length - 1 - row) * SPOTS + spot;
      if (tally.weights[cell]! <= 0) continue;
      const x = left + spot * GRID.pitchX + (GRID.pitchX - GRID.cellW) / 2 + tally.rates[cell]! * GRID.cellW, z = GRID_AT.z - (GRID.top - (row + 0.5) * GRID.pitchY);
      const tick = kit.curve(`rate ${row} ${spot}`, 2, TOKENS.uiCyan);
      tick.point(x, 0.1, z - GRID.cellH / 2);
      tick.point(x, 0.1, z + GRID.cellH / 2);
    }
  }
}

export default (() => new KuhnPokerScene()) satisfies SceneFactory;
