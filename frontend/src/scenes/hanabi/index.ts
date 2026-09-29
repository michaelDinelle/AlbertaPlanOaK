import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import {
  HANABI_COLOURS, HANABI_DISCARDED, HANABI_MISFIRED, HANABI_PLAY, HANABI_PLAYED, HANABI_PLAYERS, HANABI_RANKS, HANABI_TOLD_COLOUR, HANABI_TOLD_RANK,
  hanabiAction, hanabiClock, hanabiClued, hanabiColourOf, hanabiColours, hanabiCursor, hanabiDeck, hanabiDiscarded, hanabiEmpty, hanabiEnding, hanabiFinal, hanabiGame, hanabiGone, hanabiHeight,
  hanabiId, hanabiInfo, hanabiLastCopy, hanabiLatest, hanabiLife, hanabiMarked, hanabiMove, hanabiMoveAt, hanabiMoveCard, hanabiMoveForced, hanabiMoveKind, hanabiMoveSeat, hanabiMoveTouched,
  hanabiMoved, hanabiRankOf, hanabiRanks, hanabiRound, hanabiScore, hanabiTouched, hanabiTurn,
} from '@sim/envs/hanabi';
import type { EnvFrame } from '@sim/types';
import { constant, constantList, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import {
  BADGE_Q, BURST_Q, CARD_H, CARD_QH, CARD_QW, CARD_W, CELL_H, CELL_Q, CELL_W, CHIP_H, CHIP_W, FIREWORK, GOLD, LANTERN_Q, MISFIRE, PAPER, PLATE_H, PLATE_W, RING_R, TOKEN_Q,
  badgeMaterial, burstMaterial, cardMaterial, chipMaterial, cursorMaterial, digitAtlas, lanternMaterial, ledgerMaterial, plateMaterial, tableMaterial, tokenMaterial, type TableZone,
} from './art';

type N<T extends string> = THREE.Node<T>;

const COLOUR_NAMES = ['red', 'yellow', 'green', 'white', 'blue'] as const;
const PRESSES = ['next →', '← previous', 'play', 'discard', 'hint colour', 'hint rank'] as const;
const PARTNERS = ['literal', 'focus'] as const;
const FOCUS = 1;
const PITCH = 1.14;
const SEAT_PITCH = 5.55;
const LANTERN_GAP = 0.45;
const AGENT_Z = 3.3;
const PARTNER_Z = -3.5;
const LIFT = 0.2;
const COLUMN_PITCH = 1.3;
const STACK_Z = 0.12;
const CASCADE = 0.26;
const BURST_Z = -1.3;
const CLUE_X = -7.2;
const CLUE_PITCH = 0.7;
const CLUE_Z = -1.1;
const FUSE_Z = 0.5;
const DECK_X = 4.45;
const DECK_Z = -0.6;
const PLATE_Z = 0.5;
const FINAL_Z = 1.05;
const LEDGER_X = 5.55;
const LEDGER_Z = -1.1;
const LEDGER_PX = 0.48;
const LEDGER_PZ = 0.55;
const CHIP_Z = CARD_H / 2 + 0.12;
const DECK_LAYERS = 8;
const MOVE_SPAN = 4;
const END_SPAN = 9;
const PANEL_HALF_W = 8.45;
const RIM = 0.55;
const HOLD = 2;
const LAND = 0.62;
const SPLIT = 0.45;
const DEAL = 0.2;
const CARDS = 96;
const TOKENS_MAX = 24;

const ease = smoothstep01;
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const bump = (u: number, from: number, peak: number, to: number) => u <= from || u >= to ? 0 : u < peak ? ease((u - from) / (peak - from)) : 1 - ease((u - peak) / (to - peak));
const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const disc = (radius: number) => new THREE.CircleGeometry(radius, 28).rotateX(-Math.PI / 2);
const attribute = (capacity: number) => new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
const plural = (n: number, word: string) => `${formatCount(n)} ${word}${n === 1 ? '' : 's'}`;
const cardName = (id: number) => `${COLOUR_NAMES[hanabiColourOf(id)]} ${hanabiRankOf(id) + 1}`;
const single = (mask: number) => mask !== 0 && (mask & (mask - 1)) === 0 ? 31 - Math.clz32(mask) : -1;

interface Spot { x: number; z: number }
interface Active { m: number; before: EnvFrame; after: EnvFrame; u: number; index: number; b: ArrayLike<number>; w: ArrayLike<number>; record: number }
interface Lamp { mesh: THREE.Mesh; tint: THREE.Color }

class Batch {
  count = 0;
  private readonly matrix = new THREE.Matrix4();
  private readonly scale = new THREE.Vector3();

  constructor(readonly mesh: THREE.InstancedMesh, readonly attrs: THREE.InstancedBufferAttribute[], private readonly capacity: number) {
    mesh.frustumCulled = false;
    mesh.count = 0;
  }

  place(x: number, z: number, sx = 1, sz = sx, angle = 0): number {
    if (this.count >= this.capacity) throw new Error(`${this.mesh.name} holds ${this.capacity} instances`);
    const i = this.count++;
    this.matrix.makeRotationY(angle).scale(this.scale.set(sx, 1, sz)).setPosition(x, 0, z);
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

class HanabiScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.55, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 10;
  protected readonly stageName = 'hanabi';

  private players = 2;
  private hand = 5;
  private partner = 0;
  private patience = 1;
  private copies: number[] = [];
  private clueCapacity = 0;
  private lives = 0;
  private dealt = 0;
  private accent = 0;
  private halfW = 8.45;
  private halfH = 5;

  private cards: Batch | null = null;
  private chips: Batch | null = null;
  private bursts: Batch | null = null;
  private cells: Batch | null = null;
  private tokens: Batch | null = null;
  private lanterns: Batch | null = null;
  private cursorMesh: THREE.Mesh | null = null;
  private badgeMesh: THREE.Mesh | null = null;
  private readonly clock = uniform(0);
  private readonly cursorLook = { alpha: uniform(0), flash: uniform(0), tint: new THREE.Color(1, 1, 1) };
  private readonly badgeLook = { alpha: uniform(0), rank: uniform(-1), colour: uniform(0) };
  private readonly plateLook = { value: uniform(0), alpha: uniform(1) };
  private fireworkLamps: Lamp[] = [];
  private seatLamps: Lamp[] = [];
  private fuseLamps: Lamp[] = [];
  private flightLamp: Lamp | null = null;
  private badgeLamp: Lamp | null = null;
  private readonly disposables: { dispose(): void }[] = [];

  private fired = -1;
  private readonly flared = new Float32Array(HANABI_COLOURS).fill(-Infinity);
  private misfired = -Infinity;
  private eventLog: FrameUpdate['eventLog'] | null = null;
  private readonly spot: Spot = { x: 0, z: 0 };
  private readonly other: Spot = { x: 0, z: 0 };
  private readonly ledger = { step: 0, game: 0, turn: 0, acting: 0, clock: 0, final: 0, cursor: '—', press: '—', move: '—', table: '—', reward: 0, score: 0 };

  override legend: LegendSpec | null = null;

  private legendFor(): LegendSpec {
    return {
      title: 'HANABI',
      rows: [
        { key: 'turn', label: 'turn', tint: this.accent, value: () => this.turnText() },
        { key: 'cursor', label: 'cursor', tint: 0xdcf4ff, value: () => this.ledger.cursor },
        { key: 'press', label: 'press', tint: 0xdcf4ff, value: () => this.ledger.press },
        { key: 'move', label: 'last move', tint: GOLD, value: () => this.ledger.move },
        { key: 'table', label: 'table', tint: TOKENS.uiCyan, value: () => this.ledger.table },
        { key: 'partners', label: 'partners', tint: PAPER, value: () => this.partnersText() },
        { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => formatSigned(this.ledger.reward).replace('-', '−') },
      ],
      note: () => this.record(),
      bar: () => this.ledger.score / (HANABI_COLOURS * HANABI_RANKS),
    };
  }

  private turnText(): string {
    const { acting, clock } = this.ledger;
    return acting === 0 ? `you · move clock ${clock} of ${this.patience}` : `seat ${acting}`;
  }

  private partnersText(): string {
    return plural(this.players - 1, `${PARTNERS[this.partner]} bot`);
  }

  private record(): string {
    const log = this.eventLog, { step, game, turn, final } = this.ledger;
    const played = log ? countEvents(log, 'goal', -1, step) : 0, misfired = log ? countEvents(log, 'harm', -1, step) : 0;
    return `game ${formatCount(game + 1)} · turn ${formatCount(turn + 1)}${final ? ` · last round, ${plural(final, 'turn')} left` : ''} · ${plural(played, 'card')} played · ${plural(misfired, 'misfire')}`;
  }

  private seatCentre(seat: number): Spot {
    return seat === 0 ? { x: 0, z: AGENT_Z } : { x: (seat - 1 - (this.players - 2) / 2) * SEAT_PITCH, z: PARTNER_Z };
  }

  private cardSpot(seat: number, k: number, lifted: boolean, out: Spot): Spot {
    const centre = this.seatCentre(seat);
    out.x = centre.x + (k - (this.hand - 1) / 2) * PITCH;
    out.z = centre.z + (lifted ? (seat === 0 ? -LIFT : LIFT) : 0);
    return out;
  }

  private lanternSpot(seat: number, out: Spot): Spot {
    const centre = this.seatCentre(seat);
    out.x = centre.x - ((this.hand - 1) / 2) * PITCH - CARD_W / 2 - LANTERN_GAP;
    out.z = centre.z;
    return out;
  }

  private stackSpot(colour: number, rank: number, out: Spot): Spot {
    out.x = (colour - (HANABI_COLOURS - 1) / 2) * COLUMN_PITCH;
    out.z = STACK_Z + rank * CASCADE;
    return out;
  }

  private cellSpot(colour: number, rank: number, out: Spot): Spot {
    out.x = LEDGER_X + rank * LEDGER_PX;
    out.z = LEDGER_Z + colour * LEDGER_PZ;
    return out;
  }

  private clueSpot(k: number, out: Spot): Spot {
    out.x = CLUE_X + (k % 4) * CLUE_PITCH;
    out.z = CLUE_Z + Math.floor(k / 4) * CLUE_PITCH;
    return out;
  }

  private fuseSpot(k: number, out: Spot): Spot {
    out.x = CLUE_X + 1.5 * CLUE_PITCH + (k - (this.lives - 1) / 2) * CLUE_PITCH;
    out.z = FUSE_Z;
    return out;
  }

  private deckLayers(count: number): number {
    return count === 0 ? 0 : Math.max(1, Math.ceil(count / (this.dealt / DECK_LAYERS)));
  }

  private deckSpot(layer: number, out: Spot): Spot {
    out.x = DECK_X - 0.035 * layer;
    out.z = DECK_Z - 0.045 * layer;
    return out;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.players = parameter(env, 'players');
    this.hand = parameter(env, 'handSize');
    this.partner = parameter(env, 'partner');
    if (PARTNERS[this.partner] === undefined) throw new Error(`${env.envId} seats partner kind ${this.partner}, which Hanabi does not name`);
    this.patience = constant(env, 'HANABI_PATIENCE');
    this.copies = constantList(env, 'HANABI_COPIES');
    this.dealt = constantList(env, 'HANABI_ORDERED').length;
    this.accent = this.accentOf(ctx);
    const lantern = this.lanternSpot(1, { x: 0, z: 0 }).x - RING_R, row = this.seatCentre(this.players - 1).x + ((this.hand - 1) / 2) * PITCH + CARD_W / 2;
    this.halfW = Math.max(PANEL_HALF_W, Math.max(-lantern, row) + RIM);
    this.legend = this.legendFor();
    return new THREE.Box2(new THREE.Vector2(-this.halfW, -this.halfH), new THREE.Vector2(this.halfW, this.halfH));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff0e8).multiplyScalar(0.8);
  }

  protected override poolOptions() {
    return { burstSize: 56, sighRadius: 1.4, sighHeight: 0.7 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.debugDraw(frame, kit) };
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    this.scene.background = new THREE.Color(0x040509);
    pool.group.traverse((object) => { object.renderOrder = 20; });
    const atlas = digitAtlas();
    this.disposables.push(atlas.texture);

    const zones: TableZone[] = [
      { x: 0, z: -0.2, w: 7.5, h: 4.5 },
      { x: CLUE_X + 1.5 * CLUE_PITCH, z: -0.3, w: 3.0, h: 2.5 },
      { x: (DECK_X - 0.7 + LEDGER_X + 4 * LEDGER_PX + 0.45) / 2, z: 0, w: LEDGER_X + 4 * LEDGER_PX + 0.45 - DECK_X + 0.7, h: 3.2 },
    ];
    const table = new THREE.Mesh(flat(80, 80), tableMaterial(light, this.halfW - 0.08, this.halfH - 0.08, zones));
    table.renderOrder = 0;
    table.frustumCulled = false;
    this.scene.add(table);
    this.disposables.push(table.geometry, table.material as THREE.Material);

    const batch = (name: string, width: number, depth: number, capacity: number, slots: number, material: (attrs: THREE.InstancedBufferAttribute[]) => THREE.Material, order: number): Batch => {
      const attrs = Array.from({ length: slots }, () => attribute(capacity)), mesh = instancedMesh(flat(width, depth), material(attrs), capacity);
      mesh.name = name;
      mesh.renderOrder = order;
      this.scene.add(mesh);
      this.disposables.push(mesh.geometry, mesh.material as THREE.Material, mesh);
      return new Batch(mesh, attrs, capacity);
    };
    this.bursts = batch('hanabiBursts', BURST_Q, BURST_Q, HANABI_COLOURS, 1, ([v]) => burstMaterial(v!, this.clock), 1);
    this.cells = batch('hanabiDiscards', CELL_W + CELL_Q, CELL_H + CELL_Q, HANABI_COLOURS * HANABI_RANKS, 1, ([v]) => ledgerMaterial(light, atlas, v!), 2);
    this.tokens = batch('hanabiTokens', TOKEN_Q, TOKEN_Q, TOKENS_MAX, 1, ([v]) => tokenMaterial(light, v!), 2);
    this.lanterns = batch('hanabiLanterns', LANTERN_Q, LANTERN_Q, this.players, 2, ([v, w]) => lanternMaterial(atlas, v!, w!, this.accent, this.clock), 2);
    this.chips = batch('hanabiTruth', CHIP_W + 0.08, CHIP_H + 0.08, this.hand + 1, 1, ([v]) => chipMaterial(atlas, v!), 2);
    this.cards = batch('hanabiCards', CARD_QW, CARD_QH, CARDS, 2, ([card, look]) => cardMaterial(light, atlas, { card: card!, look: look! }, this.accent), 3);

    const plate = new THREE.Mesh(flat(PLATE_W, PLATE_H), plateMaterial(light, atlas, this.plateLook as unknown as { value: N<'float'>; alpha: N<'float'> }, this.accent));
    plate.position.set(DECK_X, 0, PLATE_Z);
    plate.renderOrder = 2;
    const cursor = new THREE.Mesh(flat(CARD_QW, CARD_QH), cursorMaterial(this.accent, { ...this.cursorLook, clock: this.clock } as unknown as Parameters<typeof cursorMaterial>[1]));
    cursor.renderOrder = 5;
    cursor.frustumCulled = false;
    const badge = new THREE.Mesh(flat(BADGE_Q, BADGE_Q), badgeMaterial(atlas, this.badgeLook as unknown as Parameters<typeof badgeMaterial>[1]));
    badge.renderOrder = 6;
    badge.frustumCulled = false;
    badge.visible = false;
    this.cursorMesh = cursor;
    this.badgeMesh = badge;
    this.scene.add(plate, cursor, badge);
    for (const mesh of [plate, cursor, badge]) this.disposables.push(mesh.geometry, mesh.material as THREE.Material);

    const lamp = (radius: number): Lamp => {
      const tint = new THREE.Color(0, 0, 0), mesh = light.emitter(disc(radius), uniform(tint) as unknown as N<'vec3'>);
      mesh.visible = false;
      this.scene.add(mesh);
      this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
      return { mesh, tint };
    };
    this.fireworkLamps = Array.from({ length: HANABI_COLOURS }, () => lamp(0.3));
    this.seatLamps = Array.from({ length: this.players }, () => lamp(0.15));
    this.fuseLamps = Array.from({ length: 4 }, () => lamp(0.07));
    this.flightLamp = lamp(0.22);
    this.badgeLamp = lamp(0.18);
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
    this.fireworkLamps = [];
    this.seatLamps = [];
    this.fuseLamps = [];
  }

  private available(frame: FrameUpdate, step: number): EnvFrame | null {
    if (step < 0) return null;
    const found = frame.frameAt(step);
    return found.step === step ? found : null;
  }

  private moveAt(frame: FrameUpdate, m: number): boolean {
    const after = this.available(frame, m), before = this.available(frame, m - 1);
    return after !== null && before !== null && hanabiMoved(before, after);
  }

  private startOf(frame: FrameUpdate, m: number): number {
    for (let prev = m - 1; prev >= Math.max(1, m - HOLD); prev--) if (this.moveAt(frame, prev)) return Math.max(m - 1, prev + HOLD - 1);
    return m - 1;
  }

  private active(frame: FrameUpdate, position: number): Active | null {
    const step = Math.floor(position), round = MOVE_SPAN * this.players;
    for (let m = step + 1; m >= Math.max(1, step - Math.max(END_SPAN, round) - HOLD); m--) {
      if (!this.moveAt(frame, m)) continue;
      const start = this.startOf(frame, m);
      if (start > position) continue;
      const before = this.available(frame, m - 1)!, after = this.available(frame, m)!;
      let end = start + (after.done ? END_SPAN : round);
      for (let next = m + 1; next <= Math.ceil(end) + 1; next++) if (this.moveAt(frame, next)) {
        end = Math.min(end, this.startOf(frame, next));
        break;
      }
      if (position >= end) return null;
      const u = (position - start) / (end - start);
      if (after.done) return { m, before, after, u, index: 0, b: before.words, w: after.words, record: 0 };
      const states = hanabiRound(before.words, after.words, m, this.players, this.hand, this.partner === FOCUS, this.copies), index = Math.min(this.players - 1, Math.floor(u * this.players));
      return { m, before, after, u: u * this.players - index, index, b: states[index]!, w: states[index + 1]!, record: hanabiMove(after.words, index) };
    }
    return null;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.clock.value = time;
    this.eventLog = frame.eventLog;
    if (this.clueCapacity === 0) {
      const first = frame.frameAt(0);
      this.clueCapacity = hanabiInfo(first.words);
      this.lives = hanabiLife(first.words);
      if (this.clueCapacity + this.lives > TOKENS_MAX || this.lives > this.fuseLamps.length) throw new Error(`Hanabi opens with ${this.clueCapacity} clues and ${this.lives} fuses, more than the table lays out`);
    }
    const step = frame.state.step, now = frame.frameAt(step), next = this.available(frame, step + 1);
    const a = next ? clamp01(frame.position - step) : 0;
    const act = this.active(frame, frame.position);
    this.celebrate(act, frame, pool, time, step);
    this.flightLamp!.mesh.visible = false;
    this.badgeLamp!.mesh.visible = false;
    this.badgeMesh!.visible = false;
    if (act && act.after.done) this.drawEnd(act, time);
    else if (act) this.drawMove(act, time);
    else this.drawStill(now.words, time);
    const ending = act !== null && act.after.done !== 0, finale = ending && act.u < SPLIT;
    this.drawCursor(finale ? act.before : now, finale ? null : this.available(frame, step - 1), finale ? null : next, a, ending && !finale);
    this.drawLanterns(finale ? act.before : now, act);
    for (const b of [this.cards, this.chips, this.bursts, this.cells, this.tokens, this.lanterns]) b!.flush();
    this.ledgerOf(frame, now, act);
  }

  private drawStill(words: ArrayLike<number>, time: number): void {
    for (let seat = 0; seat < this.players; seat++) this.drawHand(seat, words, 1, 0, -1);
    this.drawFireworks(words, 1, time);
    this.drawLedger(words, -1, 0);
    this.drawTokens(hanabiInfo(words), hanabiLife(words), -1, 0, -1, 0, 1, time);
    this.drawDeck(hanabiDeck(words), hanabiFinal(words), 1);
  }

  private drawHand(seat: number, words: ArrayLike<number>, alpha: number, glowMask: number, glow: number): void {
    for (let k = 0; k < this.hand; k++) this.drawCard(seat, k, words[seat * this.hand + k]!, hanabiTouched(words, seat * this.hand + k), alpha, (glowMask >>> k) & 1 ? glow : 0);
  }

  private drawCard(seat: number, k: number, slot: number, touched: boolean, alpha: number, glow: number): void {
    if (hanabiEmpty(slot)) {
      this.cardSpot(seat, k, false, this.spot);
      this.pushCard(this.spot.x, this.spot.z, 1, 1, 5, 5, 3, 0, 0, 0, alpha * 0.45, 0);
      return;
    }
    const clued = hanabiClued(slot);
    this.cardSpot(seat, k, clued, this.spot);
    this.pushSlot(this.spot.x, this.spot.z, 1, 1, seat, slot, touched, glow, alpha, clued ? 0.6 : 0, seat === 0);
    if (seat === 0) this.pushChip(this.spot.x, this.spot.z + CHIP_Z, hanabiId(slot), alpha);
  }

  private pushSlot(x: number, z: number, sx: number, scale: number, seat: number, slot: number, touched: boolean, glow: number, alpha: number, lift: number, hidden: boolean): void {
    const id = hanabiId(slot), colours = hanabiColours(slot), ranks = hanabiRanks(slot);
    const flags = (hanabiClued(slot) ? 1 : 0) | (hanabiMarked(slot) ? 2 : 0) | (touched ? 4 : 0) | (seat === 0 ? 0 : 8);
    const knownColour = single(colours), knownRank = single(ranks);
    if (hidden) this.pushCard(x, z, sx * scale, scale, knownColour < 0 ? 5 : knownColour, knownRank < 0 ? 5 : knownRank, 1, colours | (ranks << 5), flags, glow, alpha, lift);
    else this.pushCard(x, z, sx * scale, scale, hanabiColourOf(id), hanabiRankOf(id), 0, colours | (ranks << 5), flags, glow, alpha, lift);
  }

  private pushCard(x: number, z: number, sx: number, sz: number, colour: number, rank: number, kind: number, bits: number, flags: number, glow: number, alpha: number, lift: number): void {
    const cards = this.cards!, i = cards.place(x, z, Math.max(sx, 1e-3), sz);
    cards.attrs[0]!.setXYZW(i, colour, rank, kind, bits);
    cards.attrs[1]!.setXYZW(i, flags, glow, alpha, lift);
  }

  private pushFace(x: number, z: number, sx: number, sz: number, id: number, glow: number, alpha: number, lift: number): void {
    this.pushCard(x, z, sx, sz, hanabiColourOf(id), hanabiRankOf(id), 0, 31 | (31 << 5), 0, glow, alpha, lift);
  }

  private pushChip(x: number, z: number, id: number, alpha: number): void {
    const chips = this.chips!, i = chips.place(x, z);
    chips.attrs[0]!.setXYZW(i, hanabiColourOf(id), hanabiRankOf(id), alpha, 0);
  }

  private drawFireworks(words: ArrayLike<number>, alpha: number, time: number, flare = 0, slots = -1): void {
    for (let c = 0; c < HANABI_COLOURS; c++) {
      const height = hanabiHeight(words, c), under = slots >= 0 ? slots : height === 0 ? alpha : 0;
      if (under > 0) {
        this.stackSpot(c, 0, this.spot);
        this.pushCard(this.spot.x, this.spot.z, 1, 1, c, 5, 3, 0, 0, 0, under, 0);
      }
      for (let r = 0; r < height; r++) {
        this.stackSpot(c, r, this.spot);
        this.pushFace(this.spot.x, this.spot.z, 1, 1, c * HANABI_RANKS + r, r === height - 1 ? 0.4 * flare : 0, alpha, 0);
      }
      const glow = Math.max(Math.exp(-(time - this.flared[c]!) / 0.9), flare);
      const bursts = this.bursts!, i = bursts.place(this.stackSpot(c, 0, this.spot).x, BURST_Z);
      bursts.attrs[0]!.setXYZW(i, c, height, glow, alpha);
      const lamp = this.fireworkLamps[c]!;
      lamp.mesh.visible = height > 0 && alpha > 0.01;
      lamp.mesh.position.set(this.spot.x, 0.02, BURST_Z);
      lamp.tint.set(FIREWORK[c]!).multiplyScalar(alpha * (1.2 + 1.3 * height) * (1 + 2.5 * glow));
    }
  }

  private drawLedger(words: ArrayLike<number>, fresh: number, freshGlow: number): void {
    const cells = this.cells!;
    for (let c = 0; c < HANABI_COLOURS; c++) for (let r = 0; r < HANABI_RANKS; r++) {
      const id = c * HANABI_RANKS + r, height = hanabiHeight(words, c), played = r < height;
      const flags = (hanabiGone(words, id) ? 1 : 0) | (played ? 2 : 0) | (!played && hanabiLastCopy(words, id) ? 4 : 0) | (id === fresh && freshGlow > 0 ? 8 : 0) | (this.copies[r]! << 4);
      const i = cells.place(this.cellSpot(c, r, this.spot).x, this.spot.z);
      cells.attrs[0]!.setXYZW(i, c, r, hanabiDiscarded(words, id), flags);
    }
  }

  private drawTokens(info: number, life: number, flipClue: number, clueTurn: number, flipFuse: number, fuseFlash: number, alpha: number, time: number): void {
    const tokens = this.tokens!;
    for (let k = 0; k < this.clueCapacity; k++) {
      this.clueSpot(k, this.spot);
      const turning = k === flipClue ? Math.abs(Math.cos(Math.PI * clueTurn)) : 1;
      const i = tokens.place(this.spot.x, this.spot.z, Math.max(turning, 0.02), 1);
      tokens.attrs[0]!.setXYZW(i, 0, k < info ? 1 : 0, k === flipClue ? bump(clueTurn, 0, 0.5, 1) : 0, alpha);
    }
    const flash = Math.max(fuseFlash, Math.exp(-(time - this.misfired) / 0.6));
    for (let k = 0; k < this.lives; k++) {
      this.fuseSpot(k, this.spot);
      const i = tokens.place(this.spot.x, this.spot.z);
      const alive = k < life;
      tokens.attrs[0]!.setXYZW(i, 1, alive ? 1 : 0, k === flipFuse || (k === life && flash > 0.02) ? flash : 0, alpha);
      const lamp = this.fuseLamps[k]!;
      lamp.mesh.visible = true;
      lamp.mesh.position.set(this.spot.x, 0.02, this.spot.z - 0.05);
      lamp.tint.set(alive ? 0xff7a3d : MISFIRE).multiplyScalar(alive ? 3 * alpha : 12 * (k === life ? flash : 0));
    }
  }

  private drawDeck(count: number, final: number, alpha: number): void {
    const layers = this.deckLayers(count);
    if (count === 0) {
      this.deckSpot(0, this.spot);
      this.pushCard(this.spot.x, this.spot.z, 1, 1, 5, 5, 3, 0, 0, 0, 0.5 * alpha, 0);
    }
    for (let layer = 0; layer < layers; layer++) {
      this.deckSpot(layer, this.spot);
      this.pushCard(this.spot.x, this.spot.z, 1, 1, 5, 5, 2, 0, 0, 0, alpha, layer === layers - 1 ? 0.3 : 0);
    }
    this.plateLook.value.value = count;
    this.plateLook.alpha.value = alpha;
    const tokens = this.tokens!;
    for (let k = 0; k < final; k++) {
      const i = tokens.place(DECK_X + (k - (final - 1) / 2) * 0.32, FINAL_Z);
      tokens.attrs[0]!.setXYZW(i, 2, 1, 0, alpha);
    }
  }

  private drawMove(act: Active, time: number): void {
    const { b, w, u, record } = act;
    const kind = hanabiMoveKind(record), seat = hanabiMoveSeat(record), landed = u >= LAND;
    if (kind === HANABI_PLAYED || kind === HANABI_MISFIRED || kind === HANABI_DISCARDED) {
      const at = hanabiMoveAt(record), id = hanabiMoveCard(record), base = seat * this.hand;
      if (hanabiId(b[base + at]!) !== id) throw new Error(`Hanabi step ${act.m} moves ${cardName(id)} from seat ${seat} slot ${at + 1}, which held ${cardName(hanabiId(b[base + at]!))}`);
      for (let k = at + 1; k < this.hand; k++) if (w[base + k - 1] !== b[base + k]) throw new Error(`Hanabi step ${act.m} does not close seat ${seat}'s hand over slot ${at + 1}`);
      for (let s = 0; s < this.players; s++) if (s !== seat) this.drawHand(s, w, 1, 0, -1);
      const slide = ease(clamp01((u - 0.12) / 0.45)), draw = clamp01((u - 0.35) / 0.6);
      for (let k = 0; k < this.hand; k++) {
        if (k === at) continue;
        const slot = b[base + k]!;
        if (hanabiEmpty(slot)) continue;
        const clued = hanabiClued(slot), touched = hanabiTouched(b, base + k);
        this.cardSpot(seat, k, clued, this.spot);
        if (k > at) {
          this.cardSpot(seat, k - 1, clued, this.other);
          this.spot.x = lerp(this.spot.x, this.other.x, slide);
        }
        this.pushSlot(this.spot.x, this.spot.z, 1, 1, seat, slot, touched, 0, 1, clued ? 0.6 : 0, seat === 0);
        if (seat === 0) this.pushChip(this.spot.x, this.spot.z + CHIP_Z, hanabiId(slot), 1);
      }
      const drawn = w[base + this.hand - 1]!;
      if (!hanabiEmpty(drawn) && draw > 0) {
        const e = ease(draw);
        this.deckSpot(Math.max(0, this.deckLayers(hanabiDeck(b)) - 1), this.other);
        this.cardSpot(seat, this.hand - 1, false, this.spot);
        const x = lerp(this.other.x, this.spot.x, e), z = lerp(this.other.z, this.spot.z, e) - Math.sin(Math.PI * e) * 0.35, turn = seat === 0 ? 1 : Math.abs(Math.cos(Math.PI * e));
        if (seat === 0 || e < 0.5) this.pushCard(x, z, turn * (1 + 0.1 * Math.sin(Math.PI * e)), 1 + 0.1 * Math.sin(Math.PI * e), 5, 5, seat === 0 ? 1 : 2, seat === 0 ? 1023 : 0, 0, 0, 1, Math.sin(Math.PI * e));
        else this.pushFace(x, z, turn * (1 + 0.1 * Math.sin(Math.PI * e)), 1 + 0.1 * Math.sin(Math.PI * e), hanabiId(drawn), 0, 1, Math.sin(Math.PI * e));
        if (seat === 0 && e > 0.6) this.pushChip(this.spot.x, this.spot.z + CHIP_Z, hanabiId(drawn), clamp01((e - 0.6) / 0.4));
      }
      const success = kind === HANABI_PLAYED;
      this.drawFireworks(landed ? w : b, 1, time);
      this.drawLedger(landed ? w : b, success ? -1 : id, landed ? 1 - (u - LAND) / (1 - LAND) : 0);
      const refund = hanabiInfo(w) > hanabiInfo(b);
      this.drawTokens(landed ? hanabiInfo(w) : hanabiInfo(b), landed ? hanabiLife(w) : hanabiLife(b), refund ? hanabiInfo(b) : -1, refund ? clamp01((u - LAND + 0.15) / 0.3) : 0,
        kind === HANABI_MISFIRED ? hanabiLife(w) : -1, kind === HANABI_MISFIRED && landed ? 1 - (u - LAND) / (1 - LAND) : 0, 1, time);
      this.drawDeck(u >= 0.35 ? hanabiDeck(w) : hanabiDeck(b), hanabiFinal(u >= 0.35 ? w : b), 1);
      if (!landed) this.drawFlight(seat, at, b[base + at]!, id, success, u / LAND);
      return;
    }
    const target = hanabiMoveAt(record), value = hanabiMoveCard(record), touched = hanabiMoveTouched(record), told = u >= 0.5;
    if (kind !== HANABI_TOLD_COLOUR && kind !== HANABI_TOLD_RANK) throw new Error(`Hanabi step ${act.m} records move kind ${kind}`);
    const glow = bump(u, 0.42, 0.58, 1);
    for (let s = 0; s < this.players; s++) this.drawHand(s, s === target ? (told ? w : b) : w, 1, s === target ? touched : 0, glow);
    this.drawFireworks(w, 1, time);
    this.drawLedger(w, -1, 0);
    const spent = u >= 0.15;
    this.drawTokens(spent ? hanabiInfo(w) : hanabiInfo(b), hanabiLife(w), hanabiInfo(w), clamp01(u / 0.3), -1, 0, 1, time);
    this.drawDeck(hanabiDeck(w), hanabiFinal(w), 1);
    this.drawBadge(seat, target, kind === HANABI_TOLD_RANK ? value : -1, kind === HANABI_TOLD_COLOUR ? value : 0, u);
  }

  private drawFlight(seat: number, at: number, slot: number, id: number, success: boolean, progress: number): void {
    const e = ease(clamp01(progress)), colour = hanabiColourOf(id), rank = hanabiRankOf(id);
    this.cardSpot(seat, at, hanabiClued(slot), this.spot);
    if (success) this.stackSpot(colour, rank, this.other);
    else this.cellSpot(colour, rank, this.other);
    const x = lerp(this.spot.x, this.other.x, e), z = lerp(this.spot.z, this.other.z, e) - Math.sin(Math.PI * e) * 0.6;
    const scale = (success ? 1 : lerp(1, CELL_W / CARD_W, e)) * (1 + 0.14 * Math.sin(Math.PI * e)), lift = Math.sin(Math.PI * e);
    const alpha = success ? 1 : 1 - smoothstep01((e - 0.8) / 0.2);
    if (seat === 0 && e < 0.25) {
      const turn = Math.abs(Math.cos(Math.PI * e / 0.25));
      if (e < 0.125) this.pushSlot(x, z, turn, scale, 0, slot, false, 0, alpha, lift, true);
      else this.pushFace(x, z, turn * scale, scale, id, 0.4, alpha, lift);
    } else this.pushFace(x, z, scale, scale, id, success ? 0.35 : 0, alpha, lift);
    if (success) {
      const lamp = this.flightLamp!;
      lamp.mesh.visible = true;
      lamp.mesh.position.set(x, 0.02, z);
      lamp.tint.set(FIREWORK[colour]!).multiplyScalar(6);
    }
  }

  private drawBadge(seat: number, target: number, rank: number, colour: number, u: number): void {
    const badge = this.badgeMesh!, travel = ease(clamp01(u / 0.45)), fade = 1 - ease(clamp01((u - 0.72) / 0.28));
    this.lanternSpot(seat, this.spot);
    const centre = this.seatCentre(target), toward = target === 0 ? -1 : 1;
    const x = lerp(this.spot.x, centre.x, travel), z = lerp(this.spot.z, centre.z + toward * (CARD_H / 2 + 0.55), travel) + (seat === 0 ? -0.5 : 0.9) * Math.sin(Math.PI * travel);
    badge.visible = true;
    badge.position.set(x, 0, z);
    badge.scale.setScalar(0.7 + 0.3 * ease(clamp01(u / 0.2)));
    this.badgeLook.alpha.value = fade * ease(clamp01(u / 0.1));
    this.badgeLook.rank.value = rank;
    this.badgeLook.colour.value = colour;
    const lamp = this.badgeLamp!;
    lamp.mesh.visible = true;
    lamp.mesh.position.set(x, 0.02, z);
    lamp.tint.set(rank >= 0 ? 0xfff2dc : FIREWORK[colour]!).multiplyScalar(5 * fade);
  }

  private drawEnd(act: Active, time: number): void {
    const { u, b, w } = act, end = hanabiEnding(b, w), lost = hanabiLife(end) === 0, burned = hanabiLife(end) < hanabiLife(b);
    if (u < SPLIT) {
      this.drawEnding(b, 1);
      this.drawFireworks(end, lost ? 1 - 0.6 * ease(clamp01(u / 0.2)) : 1, time, lost ? 0 : 1 - u / SPLIT);
      this.drawLedger(b, -1, 0);
      this.drawTokens(hanabiInfo(b), hanabiLife(end), -1, 0, burned ? hanabiLife(end) : -1, burned ? 1 - u / SPLIT : 0, 1, time);
      this.drawDeck(hanabiDeck(b), hanabiFinal(b), 1);
      return;
    }
    const deal = (u - SPLIT) / (1 - SPLIT), fade = 1 - ease(clamp01(deal / 0.3)), leave = 1 - ease(clamp01(deal / 0.15));
    if (leave > 0) this.drawEnding(b, leave);
    this.drawFireworks(end, fade * (lost ? 0.4 : 1), time, 0, 1);
    this.drawLedger(deal < 0.3 ? b : w, -1, 0);
    this.drawTokens(hanabiInfo(w), hanabiLife(w), -1, 0, -1, 0, 1, time);
    const seats = this.players * this.hand;
    let left = 0;
    this.deckSpot(0, this.other);
    for (let k = 0; k < this.hand; k++) for (let seat = 0; seat < this.players; seat++) {
      const order = k * this.players + seat, start = 0.15 + 0.6 * order / seats, e = ease(clamp01((deal - start) / DEAL));
      const slot = w[seat * this.hand + k]!;
      if (e <= 0) {
        left++;
        continue;
      }
      this.cardSpot(seat, k, false, this.spot);
      const x = lerp(this.other.x, this.spot.x, e), z = lerp(this.other.z, this.spot.z, e), lift = Math.sin(Math.PI * e);
      if (e >= 1) {
        this.drawCard(seat, k, slot, false, 1, 0);
        continue;
      }
      const turn = seat === 0 ? 1 : Math.abs(Math.cos(Math.PI * e));
      if (seat === 0 || e < 0.5) this.pushCard(x, z, turn, 1, 5, 5, seat === 0 ? 1 : 2, seat === 0 ? 1023 : 0, 0, 0, 1, lift);
      else this.pushFace(x, z, turn, 1, hanabiId(slot), 0, 1, lift);
    }
    this.drawDeck(hanabiDeck(w) + left, hanabiFinal(w), 1);
  }

  private drawEnding(words: ArrayLike<number>, alpha: number): void {
    for (let q = 0; q < this.players * this.hand; q++) this.drawCard(Math.floor(q / this.hand), q % this.hand, words[q]!, hanabiTouched(words, q), alpha, 0);
  }

  private cursorSpot(words: ArrayLike<number>, q: number, out: Spot): Spot {
    return this.cardSpot(Math.floor(q / this.hand), q % this.hand, hanabiClued(words[q]!) && !hanabiEmpty(words[q]!), out);
  }

  private drawCursor(now: EnvFrame, previous: EnvFrame | null, next: EnvFrame | null, a: number, hidden: boolean): void {
    const words = now.words, q = hanabiCursor(words);
    this.cursorSpot(words, q, this.spot);
    if (next && !next.done && hanabiCursor(next.words) !== q) {
      this.cursorSpot(next.words, hanabiCursor(next.words), this.other);
      const e = ease(a);
      this.spot.x = lerp(this.spot.x, this.other.x, e);
      this.spot.z = lerp(this.spot.z, this.other.z, e);
    }
    const pressed = previous !== null && hanabiAction(now.actions) >= HANABI_PLAY;
    const flash = pressed ? 1 - ease(a) : 0;
    this.cursorLook.tint.set(pressed && !hanabiMoved(previous, now) ? TOKENS.failureBlue : 0xffffff);
    this.cursorMesh!.position.set(this.spot.x, 0, this.spot.z);
    this.cursorLook.alpha.value = hidden ? 0 : 1;
    this.cursorLook.flash.value = flash;
  }

  private drawLanterns(now: EnvFrame, act: Active | null): void {
    const lanterns = this.lanterns!, clock = hanabiClock(now.words), acting = act && !act.after.done ? hanabiMoveSeat(act.record) : 0;
    for (let seat = 0; seat < this.players; seat++) {
      this.lanternSpot(seat, this.spot);
      const i = lanterns.place(this.spot.x, this.spot.z), active = seat === acting ? 1 : 0;
      lanterns.attrs[0]!.setXYZW(i, seat === 0 ? 1 : 0, active, seat === 0 ? clock : 0, seat === 0 ? this.patience : 0);
      lanterns.attrs[1]!.setXYZW(i, seat, 1, 0, 0);
      const lamp = this.seatLamps[seat]!;
      lamp.mesh.visible = true;
      lamp.mesh.position.set(this.spot.x, 0.02, this.spot.z);
      lamp.tint.set(seat === 0 ? this.accent : PAPER).multiplyScalar(active ? 7 : 1.2);
    }
  }

  private celebrate(act: Active | null, frame: FrameUpdate, pool: Stage2D['pool'], time: number, step: number): void {
    if (!act) {
      if (frame.scrubbed) this.fired = (step + 1) * HANABI_PLAYERS - 1;
      return;
    }
    const key = act.m * HANABI_PLAYERS + act.index, threshold = act.after.done ? 0.04 : LAND;
    if (frame.scrubbed) {
      this.fired = act.u >= threshold ? key : key - 1;
      this.flared.fill(-Infinity);
      this.misfired = -Infinity;
      return;
    }
    if (act.u < threshold || key <= this.fired) return;
    this.fired = key;
    const at = new THREE.Vector3();
    if (act.after.done) {
      const end = hanabiEnding(act.b, act.w);
      if (hanabiLife(end) < hanabiLife(act.b)) {
        pool.sigh(0.8, at.set(0, 0, BURST_Z), time);
        this.misfired = time;
        if (hanabiLife(end) === 0) return;
      }
      for (let c = 0; c < HANABI_COLOURS; c++) {
        const height = hanabiHeight(end, c);
        if (height === 0) continue;
        this.stackSpot(c, 0, this.spot);
        pool.burst(at.set(this.spot.x, 0, BURST_Z), FIREWORK[c]!, 0.5 + 0.22 * height, time);
        this.flared[c] = time;
      }
      if (hanabiScore(end) >= 15) pool.pulse.fire(0.8);
      return;
    }
    const kind = hanabiMoveKind(act.record), id = hanabiMoveCard(act.record);
    if (kind === HANABI_PLAYED) {
      const c = hanabiColourOf(id), r = hanabiRankOf(id);
      this.stackSpot(c, r, this.spot);
      pool.burst(at.set(this.spot.x, 0, BURST_Z), FIREWORK[c]!, 0.8 + 0.18 * r, time);
      pool.ripple(at.set(this.spot.x, 0, this.spot.z), FIREWORK[c]!, 0.35, time);
      if (r === HANABI_RANKS - 1) pool.burst(at.set(this.spot.x, 0, BURST_Z), TOKENS.celebrationGold, 1.2, time);
      this.flared[c] = time;
    } else if (kind === HANABI_MISFIRED) {
      this.fuseSpot(hanabiLife(act.w), this.spot);
      pool.sigh(0.5, at.set(this.spot.x, 0, this.spot.z), time);
      this.misfired = time;
    } else if (kind === HANABI_DISCARDED) {
      this.cellSpot(hanabiColourOf(id), hanabiRankOf(id), this.spot);
      pool.ripple(at.set(this.spot.x, 0, this.spot.z), 0x9fc4ff, 0.18, time);
    }
  }

  private ledgerOf(frame: FrameUpdate, now: EnvFrame, act: Active | null): void {
    const words = now.words, ledger = this.ledger, step = now.step;
    ledger.step = step;
    ledger.game = hanabiGame(words);
    ledger.turn = hanabiTurn(words);
    ledger.acting = act && !act.after.done ? hanabiMoveSeat(act.record) : 0;
    ledger.clock = hanabiClock(words);
    ledger.final = hanabiFinal(words);
    ledger.reward = now.reward;
    ledger.score = hanabiScore(words);
    ledger.table = `${ledger.score}/${HANABI_COLOURS * HANABI_RANKS} · ${plural(hanabiInfo(words), 'clue')} · ${plural(hanabiLife(words), 'fuse')} · ${formatCount(hanabiDeck(words))} in deck`;
    const q = hanabiCursor(words), seat = Math.floor(q / this.hand), k = q % this.hand, slot = words[q]!;
    ledger.cursor = hanabiEmpty(slot) ? `${seat === 0 ? 'your' : `seat ${seat}`} card ${k + 1} · empty`
      : seat === 0 ? `your card ${k + 1} · hidden · ${this.knowledge(slot)}` : `seat ${seat} card ${k + 1} · ${cardName(hanabiId(slot))}`;
    const before = step > 0 ? this.available(frame, step - 1) : null;
    if (!before) ledger.press = '—';
    else {
      const action = hanabiAction(now.actions), moved = hanabiMoved(before, now);
      ledger.press = moved && !now.done && hanabiMoveForced(hanabiMove(words, 0)) ? `${PRESSES[action]} · move clock ran out`
        : action >= HANABI_PLAY && !moved ? `${PRESSES[action]} · no effect` : PRESSES[action]!;
    }
    if (act && !act.after.done) ledger.move = this.describe(act.record);
    else if (now.done && before) {
      const end = hanabiEnding(before.words, words);
      ledger.move = hanabiLife(end) === 0 ? `game over · last fuse burned · ${hanabiScore(end)} played` : `game over · score ${hanabiScore(end)}`;
    } else {
      const latest = hanabiLatest(words);
      ledger.move = latest ? this.describe(latest) : hanabiTurn(words) === 0 ? 'new deal' : '—';
    }
  }

  private knowledge(slot: number): string {
    const colours = hanabiColours(slot), ranks = hanabiRanks(slot);
    const named = colours === 31 ? '' : COLOUR_NAMES.filter((_, c) => (colours >>> c) & 1).join('/');
    const numbered = ranks === 31 ? '' : Array.from({ length: HANABI_RANKS }, (_, r) => r).filter((r) => (ranks >>> r) & 1).map((r) => r + 1).join('/');
    return `${named} ${numbered}`.trim() || 'no hints';
  }

  private describe(record: number): string {
    const kind = hanabiMoveKind(record), seat = hanabiMoveSeat(record), who = seat === 0 ? 'you' : `seat ${seat}`, value = hanabiMoveCard(record);
    const forced = hanabiMoveForced(record) ? ' · move clock ran out' : '';
    if (kind === HANABI_PLAYED) return `${who} played ${cardName(value)}`;
    if (kind === HANABI_MISFIRED) return `${who} misfired ${cardName(value)}`;
    if (kind === HANABI_DISCARDED) return `${who} discarded ${cardName(value)}${forced}`;
    const target = hanabiMoveAt(record), whom = target === 0 ? 'you' : `seat ${target}`, count = hanabiMoveTouched(record).toString(2).split('1').length - 1;
    const hint = kind === HANABI_TOLD_COLOUR ? COLOUR_NAMES[value] : `${value + 1}s`;
    return `${who} told ${whom} ${hint} · ${plural(count, 'card')}${forced}`;
  }

  private debugDraw(frame: FrameUpdate, kit: Parameters<NonNullable<GroundTruthOptions['debug']>>[1]): void {
    const step = frame.state.step, now = frame.frameAt(step), words = now.words, seats = this.players * this.hand;
    const truth = kit.marks('truth', seats), touched = kit.marks('touched', seats), stacks = kit.marks('fireworks', HANABI_COLOURS * HANABI_RANKS);
    const tokens = kit.marks('tokens', TOKENS_MAX), discards = kit.marks('discards', HANABI_COLOURS * HANABI_RANKS);
    for (let q = 0; q < seats; q++) {
      const slot = words[q]!, seat = Math.floor(q / this.hand), k = q % this.hand;
      if (hanabiEmpty(slot)) continue;
      const id = hanabiId(slot);
      this.cardSpot(seat, k, hanabiClued(slot), this.spot);
      truth.set(this.spot.x, 0.05, this.spot.z + 0.42, 0.1 + 0.03 * hanabiRankOf(id), FIREWORK[hanabiColourOf(id)]!);
      if (hanabiTouched(words, q)) touched.set(this.spot.x, 0.05, this.spot.z + (seat === 0 ? -1 : 1) * (CARD_H / 2 + 0.08), 0.06, 0xffffff);
    }
    for (let c = 0; c < HANABI_COLOURS; c++) for (let r = 0; r < hanabiHeight(words, c); r++) {
      this.stackSpot(c, r, this.spot);
      stacks.set(this.spot.x + 0.36, 0.05, this.spot.z - 0.55, 0.06, TOKENS.celebrationGold);
    }
    for (let c = 0; c < HANABI_COLOURS; c++) for (let r = 0; r < HANABI_RANKS; r++) {
      const count = hanabiDiscarded(words, c * HANABI_RANKS + r);
      if (!count) continue;
      this.cellSpot(c, r, this.spot);
      discards.set(this.spot.x, 0.05, this.spot.z, 0.05 + 0.04 * count, FIREWORK[c]!);
    }
    for (let k = 0; k < hanabiInfo(words); k++) tokens.set(this.clueSpot(k, this.spot).x, 0.05, this.spot.z, 0.1, TOKENS.uiCyan);
    for (let k = 0; k < hanabiLife(words); k++) tokens.set(this.fuseSpot(k, this.spot).x, 0.05, this.spot.z, 0.1, MISFIRE);
    const q = hanabiCursor(words), seat = Math.floor(q / this.hand);
    this.cardSpot(seat, q % this.hand, hanabiClued(words[q]!) && !hanabiEmpty(words[q]!), this.spot);
    const box = kit.curve('cursor', 5, TOKENS.uiCyan), hw = CARD_W / 2 + 0.16, hh = CARD_H / 2 + 0.16;
    for (const [dx, dz] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]] as const) box.point(this.spot.x + dx * hw, 0.05, this.spot.z + dz * hh);
    const seen = kit.curve('observation', now.observations.length, TOKENS.uiCyan), left = -this.halfW + 0.9, width = CLUE_PITCH * 4.6;
    now.observations.forEach((value, k) => seen.point(left + width * k / Math.max(1, now.observations.length - 1), 0.05, this.halfH - 0.75 - 0.9 * value));
  }
}

export default (() => new HanabiScene()) satisfies SceneFactory;
