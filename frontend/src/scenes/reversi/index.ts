import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorldVec, type GridSize } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import type { Light2D } from '@kit/light2d';
import { clamp01, smoothstep01 } from '@kit/mix';
import { rgb } from '@kit/shapes2d';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents, lastEvent } from '@sim/events';
import {
  REVERSI_AGENT, REVERSI_EMPTY, REVERSI_LEGAL, REVERSI_OPP, REVERSI_OWN, REVERSI_PASS, REVERSI_PLACE, REVERSI_RIVAL, REVERSI_X, REVERSI_Y,
  reversiAction, reversiAgentWhite, reversiColumn, reversiCount, reversiCursor, reversiGame, reversiHas, reversiLast, reversiMember, reversiOwner, reversiRow, reversiSquare, reversiThink,
} from '@sim/envs/reversi';
import type { EnvFrame, EventStream, GridCell } from '@sim/types';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import {
  BRACKET, Caption, FRAME, HINT, INLAY, LIP, PLAQUE_H, PLAQUE_W, SPRITE, WELL_Y, WINDOW,
  bannerMaterial, boardMaterial, bracketMaterial, discMaterial, flat, glyph, hintMaterial, plaqueMaterial, roundedRing, tableMaterial, textMaterial, windowMaterial, type Level,
} from './art';

type N<T extends string> = THREE.Node<T>;
type Kind = 'hold' | 'move' | 'bump' | 'deny' | 'think' | 'play' | 'reset';
type Phase = readonly [number, number];

const SQUARES = 64;
const CAPACITY = 2 * SQUARES + 2;
const SEAT_GAP = 0.8;
const SEAT_SCALE = 1.75;
const DISC_Z = -WELL_Y;
const LAND = 0.3;
const SOLO: Phase = [0, 0.92];
const FIRST: Phase = [0, 0.5];
const SECOND: Phase = [0.5, 1];
const RESULT_STEPS = 16;
const IDLE = 0.32;
const RIVAL = 0xf0a24a;
const GOLD = TOKENS.celebrationGold;
const BLOCKED = TOKENS.failureBlue;
const NEUTRAL = 0xdfe7e2;
const TITLE = 0xefe9dc;
const MUTED = 0x9aa39e;
const FILES = 'abcdefgh';
const ALPHA_BETA = 2;
const RIVALS = [
  { title: 'GREEDY', name: 'greedy · most flips' },
  { title: 'WEIGHTED', name: 'weighted squares' },
  { title: 'ALPHA-BETA', name: 'alpha-beta search' },
  { title: 'LEAGUE', name: 'league' },
] as const;
const ARROWS = ['↑ up', '↓ down', '← left', 'right →'] as const;
const DELTA = [[0, -1], [0, 1], [-1, 0], [1, 0]] as const;

const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const squareName = (square: number) => `${FILES[reversiColumn(square)]}${reversiRow(square) + 1}`;
const lastSquare = (words: ArrayLike<number>) => {
  const last = reversiLast(words);
  return last >= 1 && last <= SQUARES ? last - 1 : -1;
};
const chebyshev = (a: number, b: number) => Math.max(Math.abs(reversiColumn(a) - reversiColumn(b)), Math.abs(reversiRow(a) - reversiRow(b)));
const signed = (value: number) => (Number.isInteger(value) ? `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value)}` : formatSigned(value).replace('-', '−'));
const flipAt = (a: number, [p0, p1]: Phase, distance: number, reach: number) => {
  const span = p1 - p0, stagger = Math.min(0.12, 0.34 / Math.max(1, reach - 1));
  return clamp01((a - p0 - span * (0.28 + stagger * (distance - 1))) / (0.3 * span));
};
const landAt = (a: number, [p0, p1]: Phase) => clamp01((a - p0) / (LAND * (p1 - p0)));
const flashAt = (a: number, [p0, p1]: Phase) => {
  const t = (a - p0 - LAND * (p1 - p0)) / (0.35 * (p1 - p0));
  return t < -0.5 ? 0 : t < 0 ? 1 + t / 0.5 : Math.max(0, 1 - t);
};

interface Plan {
  step: number;
  ahead: boolean;
  kind: Kind;
  words: Uint32Array;
  board: Uint8Array;
  after: Uint8Array | null;
  white: boolean;
  afterWhite: boolean;
  member: number;
  afterMember: number;
  cursor: number;
  to: number;
  think: number;
  afterThink: number;
  passing: boolean;
  action: number;
  legal: number[];
  afterLegal: number[];
  last: number;
  afterLast: number;
  agent: number;
  rival: number;
  toAgent: number[];
  toRival: number[];
  taken: boolean;
  split: boolean;
  result: number;
  reward: number;
  afterGame: number;
  doing: string;
}

interface Outcome { reward: number; game: number; level: number }
interface Label { mesh: THREE.Mesh; caption: Caption; tint: THREE.Color }
interface Lamp { mesh: THREE.Mesh; tint: THREE.Color }
interface Seat { x: number; tint: THREE.Color; glow: Level; busy: Level; spin: Level; title: Label; sub: Label; count: Label; unit: Label; status: Label }

class ReversiScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.36, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'reversi';

  private span = 8;
  private half = 4;
  private grid: GridSize = { w: 8, h: 8 };
  private seatX = 0;
  private opponent = 0;
  private searches = false;
  private perNode = 1;
  private accent = 0x5fbf8f;
  private squares: number[] = [];
  private readonly disposables: { dispose(): void }[] = [];

  private discs: THREE.InstancedMesh | null = null;
  private discLook: THREE.InstancedBufferAttribute | null = null;
  private discFx: THREE.InstancedBufferAttribute | null = null;
  private hints: THREE.InstancedMesh | null = null;
  private hintAlpha: THREE.InstancedBufferAttribute | null = null;
  private window: THREE.Mesh | null = null;
  private bracket: THREE.Mesh | null = null;
  private banner: THREE.Mesh | null = null;
  private bannerTitle: Label | null = null;
  private bannerSub: Label | null = null;
  private cursorLamp: Lamp | null = null;
  private flashes: Lamp[] = [];
  private seats: Seat[] = [];
  private readonly windowStrength = uniform(1);
  private readonly bracketTint = new THREE.Color();
  private readonly bracketAlpha = uniform(1);
  private readonly bannerTint = new THREE.Color();
  private readonly bannerAlpha = uniform(0);

  private plan: Plan | null = null;
  private outcomeStep = -1;
  private found: { step: number; reward: number; game: number } | null = null;
  private celebrated = -1;
  private log: EventStream | null = null;
  private used = 0;
  private readonly matrix = new THREE.Matrix4();
  private readonly at = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly color = new THREE.Color();
  private readonly ledger = { turn: '—', rival: '—', cursor: '—', action: '—', discs: '—', reward: 0, share: 0.5, step: 0, game: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.span = parameter(env, 'size');
    this.half = this.span / 2;
    this.grid = { w: this.span, h: this.span };
    this.opponent = parameter(env, 'opponent');
    if (RIVALS[this.opponent] === undefined) throw new Error(`${env.envId} seats opponent ${this.opponent}, which Reversi does not name`);
    this.searches = this.opponent === ALPHA_BETA;
    this.perNode = constant(env, 'REVERSI_STEPS_PER_NODE');
    this.accent = this.accentOf(ctx);
    this.seatX = this.half + FRAME + SEAT_GAP + PLAQUE_W / 2;
    this.squares = [];
    for (let row = 0; row < this.span; row++) for (let column = 0; column < this.span; column++) this.squares.push(reversiSquare(column, row));
    this.legend = this.legendFor();
    const x = this.seatX + PLAQUE_W / 2 + 0.25, z = Math.max(this.half + FRAME, PLAQUE_H / 2) + 0.3;
    return new THREE.Box2(new THREE.Vector2(-x, -z), new THREE.Vector2(x, z));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff1e0).multiplyScalar(0.9);
  }

  protected override poolOptions() {
    return { burstSize: 64, sighRadius: 2.4, sighHeight: 1.2 };
  }

  private legendFor(): LegendSpec {
    return {
      title: 'REVERSI',
      rows: [
        { key: 'turn', label: 'turn', tint: this.accent, value: () => this.ledger.turn },
        { key: 'rival', label: 'rival', tint: RIVAL, value: () => this.ledger.rival },
        { key: 'cursor', label: 'cursor', tint: this.accent, value: () => this.ledger.cursor },
        { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action },
        { key: 'discs', label: 'discs', tint: TITLE, value: () => this.ledger.discs },
        { key: 'reward', label: 'reward', tint: GOLD, value: () => signed(this.ledger.reward) },
        { key: 'last', label: 'last game', tint: GOLD, value: () => this.lastGame() },
      ],
      note: () => this.tally(),
      bar: () => this.ledger.share,
    };
  }

  private lastGame(): string {
    const log = this.log;
    if (!log) return '—';
    const ended = lastEvent(log, 'done', this.ledger.step);
    if (ended === undefined) return '—';
    return lastEvent(log, 'goal', ended) === ended ? 'won' : lastEvent(log, 'harm', ended) === ended ? 'lost' : 'drawn';
  }

  private tally(): string {
    const log = this.log, step = this.ledger.step;
    if (!log) return '';
    const won = countEvents(log, 'goal', -1, step), lost = countEvents(log, 'harm', -1, step), drawn = countEvents(log, 'done', -1, step) - won - lost;
    return `game ${formatCount(this.ledger.game + 1)} · won ${formatCount(won)} · lost ${formatCount(lost)}${drawn ? ` · drawn ${formatCount(drawn)}` : ''}`;
  }

  private cell(square: number, out: THREE.Vector3): THREE.Vector3 {
    return cellToWorldVec(reversiColumn(square), reversiRow(square), this.grid, out);
  }

  protected override overlayOptions(): GroundTruthOptions {
    const cursor: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        cursor.x = state.pose[REVERSI_X]!;
        cursor.y = state.pose[REVERSI_Y]!;
        return cursor;
      },
      debug: (frame, kit) => {
        const now = frame.frameAt(frame.state.step), words = now.words, obs = now.observations, offset = (this.span - 1) / 2;
        const discs = kit.marks('discs', SQUARES), legal = kit.marks('legal', SQUARES), seen = kit.marks('observed', 9), last = kit.marks('last', 1), flag = kit.marks('legalFlag', 1);
        const outline = kit.curve('window', 5, this.accent);
        for (const square of this.squares) {
          const owner = reversiOwner(words, square);
          this.cell(square, this.at);
          if (owner !== REVERSI_EMPTY) discs.set(this.at.x, 0.05, this.at.z, 0.17, owner === REVERSI_AGENT ? TOKENS.uiCyan : TOKENS.cueMagenta);
          if (reversiHas(words, REVERSI_LEGAL, square)) legal.set(this.at.x, 0.05, this.at.z, 0.07, 0x86f0b4);
        }
        const x = reversiColumn(reversiCursor(words)), y = reversiRow(reversiCursor(words));
        for (let k = 0; k < 9; k++) {
          const value = obs[k]!;
          if (value !== 0) seen.set(x + (k % 3) - 1 - offset + 0.3, 0.06, y + Math.floor(k / 3) - 1 - offset + 0.3, 0.08, value > 0 ? TOKENS.uiCyan : TOKENS.cueMagenta);
        }
        for (const [dx, dy] of [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5]] as const) outline.point(x - offset + dx, 0.06, y - offset + dy);
        if (obs[12]! >= 0) last.set(Math.round(obs[12]! * (this.span - 1)) - offset, 0.07, Math.round(obs[13]! * (this.span - 1)) - offset, 0.26, GOLD);
        flag.set(x - offset - 0.3, 0.07, y - offset - 0.3, 0.09, obs[11]! > 0 ? 0x86f0b4 : 0xff6b5a);
      },
    };
  }

  protected build(_ctx: SceneContext, { light, pool }: Stage2D): void {
    this.scene.background = new THREE.Color(0x060708);
    pool.group.traverse((object) => { object.renderOrder = 20; });
    const lit = light.lit, half = this.half, extent = half + FRAME + 0.05;

    this.add(new THREE.Mesh(flat(80, 80), tableMaterial(half, this.seatX, lit)), 0);
    this.add(new THREE.Mesh(flat(2 * extent, 2 * extent), boardMaterial(this.span, lit)), 1);
    const labels = new Caption(2 * extent, 2 * extent);
    labels.draw('coordinates', (context, px) => {
      for (let k = 0; k < this.span; k++) {
        const along = extent - half + k + 0.5, strip = extent - half - FRAME / 2;
        glyph(context, px, FILES[k]!, along, strip, 0.28, 600);
        glyph(context, px, String(k + 1), strip, along, 0.28, 600);
      }
    });
    this.disposables.push(labels);
    this.add(new THREE.Mesh(flat(2 * extent, 2 * extent), textMaterial(labels, rgb(INLAY), uniform(0.8), lit)), 2);
    this.add(light.occluder(roundedRing(half + FRAME, 0.3, half + LIP)), 1);

    this.hintAlpha = new THREE.InstancedBufferAttribute(new Float32Array(2 * SQUARES), 1).setUsage(THREE.DynamicDrawUsage);
    this.hints = this.add(instancedMesh(flat(HINT, HINT), hintMaterial(this.hintAlpha, this.accent), 2 * SQUARES), 3);
    this.window = this.add(new THREE.Mesh(flat(WINDOW, WINDOW), windowMaterial(this.windowStrength, half, this.accent)), 4);
    this.discLook = new THREE.InstancedBufferAttribute(new Float32Array(CAPACITY * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.discFx = new THREE.InstancedBufferAttribute(new Float32Array(CAPACITY * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.discs = this.add(instancedMesh(flat(SPRITE, SPRITE), discMaterial(this.discLook, this.discFx, lit, GOLD), CAPACITY), 5);
    this.bracket = this.add(new THREE.Mesh(flat(BRACKET, BRACKET), bracketMaterial(uniform(this.bracketTint) as unknown as N<'vec3'>, this.bracketAlpha)), 6);

    this.cursorLamp = this.lamp(light, new THREE.CircleGeometry(0.12, 24), 0, 0);
    this.flashes = [this.lamp(light, new THREE.CircleGeometry(0.2, 24), 0, 0), this.lamp(light, new THREE.CircleGeometry(0.2, 24), 0, 0)];
    this.seats = [this.seat(light, -this.seatX, this.accent), this.seat(light, this.seatX, RIVAL)];

    const width = Math.min(6, 0.74 * this.span), height = 1.1, z = half - 0.45;
    this.banner = this.add(new THREE.Mesh(flat(width + 0.6, height + 0.6), bannerMaterial(width, height, uniform(this.bannerTint) as unknown as N<'vec3'>, this.bannerAlpha)), 8);
    this.banner.position.z = z;
    this.bannerTitle = this.label(light, width - 0.4, 0.52, 0, z - 0.15, TITLE, false, this.bannerAlpha, 9);
    this.bannerSub = this.label(light, width - 0.4, 0.28, 0, z + 0.3, MUTED, false, this.bannerAlpha, 9);
  }

  private add<T extends THREE.Mesh>(mesh: T, order: number): T {
    mesh.renderOrder = order;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
    if (mesh instanceof THREE.InstancedMesh) this.disposables.push(mesh);
    return mesh;
  }

  private lamp(light: Light2D, geometry: THREE.BufferGeometry, x: number, z: number): Lamp {
    const tint = new THREE.Color(0), mesh = this.add(light.emitter(geometry, uniform(tint) as unknown as N<'vec3'>), 2);
    mesh.position.set(x, 0, z);
    return { mesh, tint };
  }

  private label(light: Light2D, width: number, height: number, x: number, z: number, color: number, lit: boolean, alpha: Level, order = 3): Label {
    const caption = new Caption(width, height), tint = new THREE.Color(color);
    const mesh = this.add(new THREE.Mesh(flat(width, height), textMaterial(caption, uniform(tint) as unknown as N<'vec3'>, alpha, lit ? light.lit : null)), order);
    mesh.position.set(x, 0, z);
    this.disposables.push(caption);
    return { mesh, caption, tint };
  }

  private seat(light: Light2D, x: number, color: number): Seat {
    const tint = new THREE.Color(color), glow = uniform(0), busy = uniform(0), spin = uniform(0), shown = uniform(1);
    this.add(new THREE.Mesh(flat(PLAQUE_W + 0.3, PLAQUE_H + 0.3), plaqueMaterial(uniform(tint) as unknown as N<'vec3'>, glow, busy, spin, light.lit)), 1).position.set(x, 0, 0);
    return {
      x, tint, glow, busy, spin,
      title: this.label(light, 2.4, 0.5, x, -2.08, TITLE, false, shown),
      sub: this.label(light, 2.4, 0.34, x, -1.68, MUTED, false, shown),
      count: this.label(light, 2.4, 0.9, x, 0.93, TITLE, false, shown),
      unit: this.label(light, 2.4, 0.3, x, 1.5, MUTED, false, shown),
      status: this.label(light, 2.5, 0.42, x, 2.02, color, false, shown),
    };
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
    this.seats = [];
    this.flashes = [];
    this.plan = null;
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    pool.update(frame.dtReal, time);
    this.log = frame.eventLog;
    const step = frame.state.step, plan = this.planFor(frame, step), a = plan.ahead ? clamp01(frame.position - step) : 0;
    if (frame.events.some((event) => event.type === 'done')) this.celebrate(frame, pool, time);
    const outcome = this.outcome(frame, plan, a);
    this.record(plan);
    this.drawDiscs(plan, a, outcome);
    this.drawHints(plan, a);
    this.drawCursor(plan, a);
    this.drawSeats(plan, a, outcome, time);
    this.drawBanner(outcome);
  }

  private owners(words: ArrayLike<number>): Uint8Array {
    const out = new Uint8Array(SQUARES);
    for (const square of this.squares) out[square] = reversiOwner(words, square);
    return out;
  }

  private legalOf(words: ArrayLike<number>): number[] {
    return this.squares.filter((square) => reversiHas(words, REVERSI_LEGAL, square));
  }

  private planFor(frame: FrameUpdate, step: number): Plan {
    const peek = frame.frameAt(step + 1), ahead = peek.step === step + 1;
    if (this.plan && this.plan.step === step && this.plan.ahead === ahead) return this.plan;
    const now = frame.frameAt(step), words = Uint32Array.from(now.words), think = reversiThink(words), cursor = reversiCursor(words), white = reversiAgentWhite(words);
    const plan: Plan = {
      step, ahead, kind: 'hold', words, board: this.owners(words), after: null, white, afterWhite: white, member: reversiMember(words), afterMember: reversiMember(words), cursor, to: cursor, think, afterThink: think,
      passing: this.passing(frame, step, think), action: -1, legal: this.legalOf(words), afterLegal: [], last: lastSquare(words), afterLast: lastSquare(words),
      agent: -1, rival: -1, toAgent: [], toRival: [], taken: false, split: false, result: 0, reward: now.reward, afterGame: reversiGame(words), doing: this.describe(frame, now),
    };
    this.plan = plan;
    if (!ahead) return plan;
    const later = peek.words;
    plan.to = reversiCursor(later);
    plan.afterThink = reversiThink(later);
    plan.afterWhite = reversiAgentWhite(later);
    plan.afterMember = reversiMember(later);
    plan.afterGame = reversiGame(later);
    plan.afterLegal = this.legalOf(later);
    plan.afterLast = lastSquare(later);
    plan.after = this.owners(later);
    plan.action = think === 0 ? reversiAction(peek.actions) : -1;
    if (peek.done) {
      plan.kind = 'reset';
      plan.result = Math.sign(peek.reward);
      return plan;
    }
    const changed = this.squares.some((square) => plan.board[square] !== plan.after![square]);
    if (think === 0 && plan.action !== REVERSI_PLACE) {
      if (changed) throw new Error(`Reversi step ${step + 1} changes the board on a cursor move`);
      plan.kind = plan.to === cursor ? 'bump' : 'move';
    } else if (think === 0 && !reversiHas(words, REVERSI_LEGAL, cursor)) {
      if (changed) throw new Error(`Reversi step ${step + 1} changes the board on an illegal placement at ${squareName(cursor)}`);
      plan.kind = 'deny';
    } else if (!changed) {
      if (think === 0) throw new Error(`Reversi step ${step + 1} leaves the board unchanged after a legal placement at ${squareName(cursor)}`);
      plan.kind = 'think';
    } else this.play(plan, step + 1);
    return plan;
  }

  private play(plan: Plan, step: number): void {
    const after = plan.after!, added: number[] = [];
    plan.kind = 'play';
    if (plan.think === 0) plan.agent = plan.cursor;
    for (const square of this.squares) {
      const was = plan.board[square]!, now = after[square]!;
      if (was !== REVERSI_EMPTY && now === REVERSI_EMPTY) throw new Error(`Reversi step ${step} lifts the disc on ${squareName(square)}`);
      if (was === REVERSI_EMPTY && now !== REVERSI_EMPTY) added.push(square);
      else if (was === REVERSI_RIVAL && now === REVERSI_AGENT) plan.toAgent.push(square);
      else if (was === REVERSI_AGENT && now === REVERSI_RIVAL) plan.toRival.push(square);
    }
    if (plan.agent >= 0 && !added.includes(plan.agent)) throw new Error(`Reversi step ${step} places nothing on ${squareName(plan.agent)} where the agent played`);
    const rival = added.filter((square) => square !== plan.agent);
    if (rival.length > 1 || (rival.length === 1 && rival[0] !== plan.afterLast)) throw new Error(`Reversi step ${step} adds discs its last-move word does not name`);
    plan.rival = rival[0] ?? -1;
    plan.taken = plan.agent >= 0 && after[plan.agent] === REVERSI_RIVAL;
    if (plan.rival < 0 && (plan.toRival.length > 0 || plan.taken)) throw new Error(`Reversi step ${step} turns agent discs without a rival move`);
    if (plan.agent < 0 && (plan.rival < 0 || plan.toAgent.length > 0)) throw new Error(`Reversi step ${step} changes the board while only the rival may move`);
    plan.split = plan.agent >= 0 && plan.rival >= 0;
  }

  private passing(frame: FrameUpdate, step: number, think: number): boolean {
    if (think === 0) return false;
    if (!this.searches) return true;
    const began = step - (think - 1), start = frame.frameAt(began), before = frame.frameAt(began - 1);
    return began >= 1 && start.step === began && before.step === began - 1 && !start.done && reversiThink(before.words) > 0;
  }

  private describe(frame: FrameUpdate, now: EnvFrame): string {
    const before = frame.frameAt(now.step - 1);
    if (now.step < 1 || before.step !== now.step - 1) return '—';
    if (reversiThink(before.words) > 0) return 'waiting';
    const action = reversiAction(now.actions), cursor = reversiCursor(before.words);
    if (action === REVERSI_PLACE) return `place ${squareName(cursor)}${reversiHas(before.words, REVERSI_LEGAL, cursor) ? '' : ' · illegal'}`;
    return `${ARROWS[action]}${!now.done && reversiCursor(now.words) === cursor ? ' · edge' : ''}`;
  }

  private outcome(frame: FrameUpdate, plan: Plan, a: number): Outcome | null {
    if (plan.kind === 'reset') {
      const level = smoothstep01((a - 0.3) / 0.3);
      return level > 0 ? { reward: plan.result, game: plan.afterGame, level } : null;
    }
    if (this.outcomeStep !== plan.step) {
      this.outcomeStep = plan.step;
      this.found = null;
      for (let k = 0; k < RESULT_STEPS; k++) {
        const f = frame.frameAt(plan.step - k);
        if (plan.step - k < 0 || f.step !== plan.step - k) break;
        if (f.done) {
          this.found = { step: f.step, reward: Math.sign(f.reward), game: reversiGame(f.words) };
          break;
        }
      }
    }
    if (!this.found) return null;
    const age = plan.step - this.found.step + a, level = 1 - smoothstep01((age - 0.55 * RESULT_STEPS) / (0.45 * RESULT_STEPS));
    return level > 0 ? { reward: this.found.reward, game: this.found.game, level } : null;
  }

  private record(plan: Plan): void {
    const ledger = this.ledger, words = plan.words, own = reversiCount(words, REVERSI_OWN), opp = reversiCount(words, REVERSI_OPP);
    const nodes = Math.ceil((plan.think - 1) / this.perNode);
    ledger.step = plan.step;
    ledger.game = reversiGame(words);
    ledger.turn = plan.think === 0 ? 'agent' : plan.passing ? 'agent passes · rival' : this.searches ? `rival thinking · ${formatCount(nodes)} node${nodes === 1 ? '' : 's'}` : 'rival';
    ledger.rival = this.opponent === RIVALS.length - 1 ? `league member ${reversiMember(words) + 1}` : RIVALS[this.opponent]!.name;
    ledger.cursor = `${squareName(plan.cursor)}${plan.legal.includes(plan.cursor) ? ' · legal' : ''}`;
    ledger.action = plan.doing;
    ledger.discs = `agent ${own} · rival ${opp}`;
    ledger.reward = plan.reward;
    ledger.share = own + opp > 0 ? own / (own + opp) : 0.5;
  }

  private put(x: number, z: number, white: number, squash: number, angle: number, alpha: number, glow: number, lift: number, marker: number, scale: number): void {
    const i = this.used++, grow = scale * (1 + 0.16 * lift);
    this.matrix.makeScale(grow, 1, grow).setPosition(x, 0, z);
    this.discs!.setMatrixAt(i, this.matrix);
    this.discLook!.setXYZW(i, white, squash, alpha, glow);
    this.discFx!.setXYZW(i, angle, lift, marker, 0);
  }

  private disc(square: number, white: number, alpha: number, lift: number, marker: number): void {
    this.cell(square, this.at);
    this.put(this.at.x, this.at.z, white, 1, 0, alpha, 0, lift, marker, 1);
  }

  private turnOver(square: number, from: number, to: number, u: number, origin: number, marker: number): void {
    this.cell(square, this.at);
    const angle = Math.atan2(reversiRow(origin) - reversiRow(square), reversiColumn(square) - reversiColumn(origin));
    this.put(this.at.x, this.at.z, u < 0.5 ? from : to, Math.abs(Math.cos(Math.PI * u)), angle, 1, 0, 0.35 * Math.sin(Math.PI * u), marker, 1);
  }

  private reach(squares: readonly number[], origin: number): number {
    let reach = 1;
    for (const square of squares) reach = Math.max(reach, chebyshev(square, origin));
    return reach;
  }

  private drawDiscs(plan: Plan, a: number, outcome: Outcome | null): void {
    this.used = 0;
    const face = (owner: number, white: boolean) => ((owner === REVERSI_AGENT) === white ? 1 : 0);
    const board = plan.board, agent = face(REVERSI_AGENT, plan.white), rival = 1 - agent;
    if (plan.kind === 'reset') {
      const fade = smoothstep01((a - 0.25) / 0.35), appear = smoothstep01((a - 0.6) / 0.3), after = plan.after!;
      for (const square of this.squares) if (board[square]) this.disc(square, face(board[square]!, plan.white), 1 - fade, 0, square === plan.last ? 1 - fade : 0);
      if (appear > 0) for (const square of this.squares) if (after[square]) this.disc(square, face(after[square]!, plan.afterWhite), appear, 0.8 * (1 - appear), square === plan.afterLast ? appear : 0);
    } else if (plan.kind === 'play') {
      const first = plan.split ? FIRST : SOLO, second = plan.split ? SECOND : SOLO;
      const firstReach = this.reach(plan.toAgent, plan.agent), secondReach = this.reach(plan.taken ? [...plan.toRival, plan.agent] : plan.toRival, plan.rival);
      const settled = plan.rival >= 0 ? second[0] + LAND * (second[1] - second[0]) : first[1] - 0.1;
      const fading = plan.afterLast === plan.last ? 1 : 1 - smoothstep01((a - settled) / 0.08);
      for (const square of this.squares) {
        const owner = board[square]!, marker = square === plan.last ? fading : 0;
        if (!owner) continue;
        if (plan.toAgent.includes(square)) this.turnOver(square, rival, agent, flipAt(a, first, chebyshev(square, plan.agent), firstReach), plan.agent, marker);
        else if (plan.toRival.includes(square)) this.turnOver(square, agent, rival, flipAt(a, second, chebyshev(square, plan.rival), secondReach), plan.rival, marker);
        else this.disc(square, face(owner, plan.white), 1, 0, marker);
      }
      if (plan.agent >= 0) {
        const land = landAt(a, first), lift = 1 - smoothstep01(land), alpha = clamp01(land * 3);
        if (plan.taken && a > second[0]) this.turnOver(plan.agent, agent, rival, flipAt(a, second, chebyshev(plan.agent, plan.rival), secondReach), plan.rival, 0);
        else if (alpha > 0) {
          this.cell(plan.agent, this.at);
          this.put(this.at.x, this.at.z, agent, 1, 0, alpha, 0, lift, 0, 1);
        }
      }
      if (plan.rival >= 0) {
        const land = landAt(a, second), alpha = clamp01(land * 3);
        if (alpha > 0) {
          this.cell(plan.rival, this.at);
          this.put(this.at.x, this.at.z, rival, 1, 0, alpha, 0, 1 - smoothstep01(land), smoothstep01((land - 0.8) / 0.2), 1);
        }
      }
    } else for (const square of this.squares) if (board[square]) this.disc(square, face(board[square]!, plan.white), 1, 0, square === plan.last ? 1 : 0);

    const turn = plan.kind === 'reset' && plan.afterWhite !== plan.white ? smoothstep01((a - 0.55) / 0.35) : 0;
    for (const [side, owner] of [[0, REVERSI_AGENT], [1, REVERSI_RIVAL]] as const) {
      const before = face(owner, plan.white), after = face(owner, plan.afterWhite);
      const won = outcome ? (owner === REVERSI_AGENT ? outcome.reward > 0 : outcome.reward < 0) : false;
      this.put(this.seats[side]!.x, DISC_Z, turn < 0.5 ? before : after, Math.abs(Math.cos(Math.PI * turn)), 0, 1, won ? 0.55 * outcome!.level : 0, 0.3 * Math.sin(Math.PI * turn), 0, SEAT_SCALE);
    }
    const discs = this.discs!;
    discs.count = this.used;
    discs.instanceMatrix.needsUpdate = true;
    this.discLook!.needsUpdate = true;
    this.discFx!.needsUpdate = true;
  }

  private drawHints(plan: Plan, a: number): void {
    const hints = this.hints!, alpha = this.hintAlpha!, moving = plan.kind === 'play' || plan.kind === 'reset';
    let used = 0;
    const place = (square: number, level: number) => {
      if (level <= 0) return;
      this.cell(square, this.at);
      this.matrix.makeTranslation(this.at.x, 0, this.at.z);
      hints.setMatrixAt(used, this.matrix);
      alpha.setX(used++, level);
    };
    const leaving = moving ? 1 - smoothstep01(a / 0.15) : 1;
    for (const square of plan.legal) place(square, leaving);
    if (moving) for (const square of plan.afterLegal) place(square, smoothstep01((a - 0.85) / 0.15));
    hints.count = used;
    hints.instanceMatrix.needsUpdate = true;
    alpha.needsUpdate = true;
  }

  private drawCursor(plan: Plan, a: number): void {
    const from = this.cell(plan.cursor, this.at), to = this.cell(plan.to, this.other);
    let x = from.x, z = from.z, shown = plan.cursor, press = 0, deny = 0;
    let alpha = lerp(plan.think === 0 ? 1 : IDLE, plan.afterThink === 0 ? 1 : IDLE, smoothstep01((a - 0.5) / 0.5));
    if (plan.kind === 'move') {
      const k = smoothstep01(a / 0.85);
      x = lerp(from.x, to.x, k);
      z = lerp(from.z, to.z, k);
      if (k > 0.5) shown = plan.to;
    } else if (plan.kind === 'bump') {
      const [dx, dz] = DELTA[plan.action]!, k = 0.13 * Math.sin(Math.PI * clamp01(a / 0.7));
      x += dx * k;
      z += dz * k;
    } else if (plan.kind === 'deny') {
      deny = Math.sin(Math.PI * a);
      x += 0.07 * Math.sin(3 * Math.PI * a) * (1 - a);
    } else if (plan.kind === 'play' && plan.agent >= 0) {
      press = Math.sin(Math.PI * clamp01(a / (LAND * ((plan.split ? FIRST : SOLO)[1]))));
      if (plan.split) alpha = 1 - 0.4 * Math.sin(Math.PI * clamp01((a - SECOND[0]) / (SECOND[1] - SECOND[0])));
    } else if (plan.kind === 'reset') {
      const late = a > 0.5;
      if (plan.action === REVERSI_PLACE) press = Math.sin(Math.PI * clamp01(a / 0.25));
      if (late) {
        x = to.x;
        z = to.z;
        shown = plan.to;
      }
      alpha = late ? smoothstep01((a - 0.6) / 0.3) * (plan.afterThink === 0 ? 1 : IDLE) : (1 - smoothstep01((a - 0.15) / 0.25)) * (plan.think === 0 ? 1 : IDLE);
    }
    const legal = (plan.kind === 'reset' && shown === plan.to && a > 0.5 ? plan.afterLegal : plan.legal).includes(shown);
    this.bracketTint.set(legal ? this.accent : NEUTRAL).lerp(this.color.set(BLOCKED), deny);
    if (legal) this.bracketTint.multiplyScalar(1.15 + 0.6 * press);
    this.bracketAlpha.value = alpha;
    this.windowStrength.value = alpha;
    this.bracket!.position.set(x, 0, z);
    this.bracket!.scale.setScalar(1 - 0.1 * press);
    this.window!.position.set(x, 0, z);
    const lamp = this.cursorLamp!;
    lamp.mesh.position.set(x, 0, z);
    lamp.tint.set(this.accent).multiplyScalar(0.8 + 4.2 * alpha + 6 * press);
    const [agentFlash, rivalFlash] = this.flashes as [Lamp, Lamp];
    agentFlash.mesh.visible = rivalFlash.mesh.visible = false;
    if (plan.kind !== 'play') return;
    if (plan.agent >= 0) {
      this.cell(plan.agent, agentFlash.mesh.position);
      agentFlash.tint.set(this.accent).multiplyScalar(7 * flashAt(a, plan.split ? FIRST : SOLO));
      agentFlash.mesh.visible = true;
    }
    if (plan.rival >= 0) {
      this.cell(plan.rival, rivalFlash.mesh.position);
      rivalFlash.tint.set(RIVAL).multiplyScalar(7 * flashAt(a, plan.split ? SECOND : SOLO));
      rivalFlash.mesh.visible = true;
    }
  }

  private drawSeats(plan: Plan, a: number, outcome: Outcome | null, time: number): void {
    const [agent, rival] = this.seats as [Seat, Seat], words = plan.words;
    const white = plan.kind === 'reset' && a > 0.72 ? plan.afterWhite : plan.white;
    const colour = (mine: boolean) => (mine === white ? 'white' : 'black');
    const member = this.opponent === RIVALS.length - 1 ? `member ${(plan.kind === 'reset' && a > 0.72 ? plan.afterMember : plan.member) + 1} · ` : '';
    const nodes = Math.ceil((plan.think - 1) / this.perNode), thinking = this.searches && plan.think > 0;
    agent.title.caption.write('AGENT', 0.3, 700, 0.24);
    agent.sub.caption.write(`plays ${colour(true)}`, 0.2, 500, 0.08);
    rival.title.caption.write(RIVALS[this.opponent]!.title, 0.3, 700, 0.24);
    rival.sub.caption.write(`${member}plays ${colour(false)}`, 0.2, 500, 0.08);
    agent.count.caption.write(String(reversiCount(words, REVERSI_OWN)), 0.8, 600, 0);
    rival.count.caption.write(String(reversiCount(words, REVERSI_OPP)), 0.8, 600, 0);
    agent.unit.caption.write('DISCS', 0.15, 600, 0.3);
    rival.unit.caption.write('DISCS', 0.15, 600, 0.3);

    let agentSays = plan.think === 0 ? 'TO MOVE' : plan.passing ? 'MUST PASS' : 'WAITING';
    let rivalSays = thinking ? `THINKING · ${nodes}` : plan.think > 0 ? 'TO MOVE' : reversiLast(words) === REVERSI_PASS ? 'PASSED' : 'WAITING';
    agent.status.tint.set(this.accent);
    rival.status.tint.set(RIVAL);
    agent.glow.value = plan.think === 0 ? 1 : 0.12;
    rival.glow.value = plan.think > 0 ? 1 : 0.12;
    rival.busy.value = thinking ? 1 : 0;
    rival.spin.value = 3.2 * time;
    if (outcome) {
      const draw = outcome.reward === 0, agentWon = outcome.reward > 0;
      agentSays = draw ? 'DRAW' : agentWon ? 'WINS' : 'LOSES';
      rivalSays = draw ? 'DRAW' : agentWon ? 'LOSES' : 'WINS';
      for (const [seat, won] of [[agent, agentWon], [rival, !agentWon && !draw]] as const) {
        seat.status.tint.set(draw ? NEUTRAL : won ? GOLD : MUTED);
        seat.glow.value = lerp(seat.glow.value, won ? 1.4 : 0.1, outcome.level);
      }
    }
    agent.status.caption.write(agentSays, 0.22, 700, 0.16);
    rival.status.caption.write(rivalSays, 0.22, 700, 0.16);
  }

  private drawBanner(outcome: Outcome | null): void {
    const level = outcome?.level ?? 0;
    this.bannerAlpha.value = level;
    for (const mesh of [this.banner!, this.bannerTitle!.mesh, this.bannerSub!.mesh]) mesh.visible = level > 0.001;
    if (!outcome) return;
    const reward = outcome.reward;
    this.bannerTint.set(reward > 0 ? GOLD : reward < 0 ? RIVAL : NEUTRAL);
    this.bannerTitle!.tint.set(reward > 0 ? GOLD : TITLE);
    this.bannerTitle!.caption.write(reward > 0 ? 'AGENT WINS' : reward < 0 ? 'RIVAL WINS' : 'DRAW', 0.36, 700, 0.2);
    this.bannerSub!.caption.write(`game ${formatCount(outcome.game)} · reward ${signed(reward)}`, 0.17, 500, 0.08);
  }

  private celebrate(frame: FrameUpdate, pool: CelebrationPool, time: number): void {
    const ended = lastEvent(frame.eventLog, 'done', frame.state.step);
    if (ended === undefined || ended === this.celebrated) return;
    this.celebrated = ended;
    const end = frame.frameAt(ended);
    if (end.step !== ended) return;
    if (end.reward > 0) {
      pool.burst(this.at.set(-this.seatX, 30, DISC_Z), this.accent, 1.5, time);
      pool.burst(this.at.set(0, 30, 0), GOLD, 2.6, time);
      pool.ripple(this.at.set(0, 0, 0), GOLD, 1.3, time);
      pool.pulse.fire(0.8);
    } else if (end.reward < 0) {
      pool.burst(this.at.set(this.seatX, 30, DISC_Z), RIVAL, 1.4, time);
      pool.sigh(0.6, this.at.set(-this.seatX, 0, DISC_Z), time);
    } else pool.burst(this.at.set(0, 30, 0), NEUTRAL, 1.8, time);
  }
}

export default (() => new ReversiScene()) satisfies SceneFactory;
