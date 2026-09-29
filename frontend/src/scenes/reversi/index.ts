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
  REVERSI_AGENT, REVERSI_EMPTY, REVERSI_LEGAL, REVERSI_NEXT, REVERSI_OPP, REVERSI_OWN, REVERSI_PASS, REVERSI_PLAY, REVERSI_RIVAL, REVERSI_X, REVERSI_Y,
  reversiAction, reversiAgentWhite, reversiColumn, reversiCount, reversiFlips, reversiGame, reversiHas, reversiLast, reversiMember, reversiOwner, reversiProposal, reversiRow, reversiShown, reversiSquare,
} from '@sim/envs/reversi';
import type { EnvFrame, EventStream, GridCell } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import {
  BRACKET, Caption, FRAME, HINT, INLAY, LIP, PLAQUE_H, PLAQUE_W, SPRITE, WELL_Y,
  bannerMaterial, boardMaterial, bracketMaterial, discMaterial, flat, glyph, hintMaterial, plaqueMaterial, roundedRing, tableMaterial, textMaterial, type Level,
} from './art';

type N<T extends string> = THREE.Node<T>;
type Kind = 'hold' | 'next' | 'play' | 'reset';
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
const GHOST = 0.45;
const PREVIEW = 0.35;
const RIVAL = 0xf0a24a;
const GOLD = TOKENS.celebrationGold;
const NEUTRAL = 0xdfe7e2;
const TITLE = 0xefe9dc;
const MUTED = 0x9aa39e;
const FILES = 'abcdefgh';
const RIVALS = [
  { title: 'GREEDY', name: 'greedy · most flips' },
  { title: 'WEIGHTED', name: 'weighted squares' },
  { title: 'ALPHA-BETA', name: 'alpha-beta search' },
  { title: 'LEAGUE', name: 'league' },
] as const;

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
  proposal: number;
  to: number;
  shown: number;
  options: number;
  preview: number[];
  afterPreview: number[];
  action: number;
  legal: number[];
  afterLegal: number[];
  last: number;
  afterLast: number;
  agent: number;
  rival: number;
  rivals: number[];
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
interface Seat { x: number; tint: THREE.Color; glow: Level; title: Label; sub: Label; count: Label; unit: Label; status: Label }

class ReversiScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.36, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'reversi';

  private span = 8;
  private half = 4;
  private grid: GridSize = { w: 8, h: 8 };
  private seatX = 0;
  private opponent = 0;
  private accent = 0x5fbf8f;
  private squares: number[] = [];
  private readonly disposables: { dispose(): void }[] = [];

  private discs: THREE.InstancedMesh | null = null;
  private discLook: THREE.InstancedBufferAttribute | null = null;
  private discFx: THREE.InstancedBufferAttribute | null = null;
  private hints: THREE.InstancedMesh | null = null;
  private hintAlpha: THREE.InstancedBufferAttribute | null = null;
  private bracket: THREE.Mesh | null = null;
  private banner: THREE.Mesh | null = null;
  private bannerTitle: Label | null = null;
  private bannerSub: Label | null = null;
  private proposalLamp: Lamp | null = null;
  private flashes: Lamp[] = [];
  private seats: Seat[] = [];
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
  private readonly ledger = { proposal: '—', clock: '—', rival: '—', action: '—', discs: '—', reward: 0, share: 0.5, step: 0, game: 0 };

  protected extent(ctx: SceneContext): THREE.Box2 {
    const env = ctx.environment;
    this.span = parameter(env, 'size');
    this.half = this.span / 2;
    this.grid = { w: this.span, h: this.span };
    this.opponent = parameter(env, 'opponent');
    if (RIVALS[this.opponent] === undefined) throw new Error(`${env.envId} seats opponent ${this.opponent}, which Reversi does not name`);
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
        { key: 'proposal', label: 'proposal', tint: this.accent, value: () => this.ledger.proposal },
        { key: 'clock', label: 'shown', tint: this.accent, value: () => this.ledger.clock },
        { key: 'rival', label: 'rival', tint: RIVAL, value: () => this.ledger.rival },
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
    const proposal: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        proposal.x = state.pose[REVERSI_X]!;
        proposal.y = state.pose[REVERSI_Y]!;
        return proposal;
      },
      debug: (frame, kit) => {
        const now = frame.frameAt(frame.state.step), words = now.words, obs = now.observations;
        const discs = kit.marks('discs', SQUARES), legal = kit.marks('legal', SQUARES), seen = kit.marks('observed', SQUARES), last = kit.marks('last', 1);
        for (const square of this.squares) {
          const owner = reversiOwner(words, square), value = obs[this.span * reversiRow(square) + reversiColumn(square)]!;
          this.cell(square, this.at);
          if (owner !== REVERSI_EMPTY) discs.set(this.at.x, 0.05, this.at.z, 0.17, owner === REVERSI_AGENT ? TOKENS.uiCyan : TOKENS.cueMagenta);
          if (reversiHas(words, REVERSI_LEGAL, square)) legal.set(this.at.x, 0.05, this.at.z, 0.07, 0x86f0b4);
          if (value !== 0) seen.set(this.at.x + 0.3, 0.06, this.at.z + 0.3, 0.08, value > 0 ? TOKENS.uiCyan : TOKENS.cueMagenta);
        }
        const moved = lastSquare(words);
        if (moved >= 0) {
          this.cell(moved, this.at);
          last.set(this.at.x, 0.07, this.at.z, 0.26, GOLD);
        }
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
    this.discLook = new THREE.InstancedBufferAttribute(new Float32Array(CAPACITY * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.discFx = new THREE.InstancedBufferAttribute(new Float32Array(CAPACITY * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.discs = this.add(instancedMesh(flat(SPRITE, SPRITE), discMaterial(this.discLook, this.discFx, lit, GOLD), CAPACITY), 5);
    this.bracket = this.add(new THREE.Mesh(flat(BRACKET, BRACKET), bracketMaterial(uniform(this.bracketTint) as unknown as N<'vec3'>, this.bracketAlpha)), 6);

    this.proposalLamp = this.lamp(light, new THREE.CircleGeometry(0.12, 24), 0, 0);
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
    const tint = new THREE.Color(color), glow = uniform(0), shown = uniform(1);
    this.add(new THREE.Mesh(flat(PLAQUE_W + 0.3, PLAQUE_H + 0.3), plaqueMaterial(uniform(tint) as unknown as N<'vec3'>, glow, light.lit)), 1).position.set(x, 0, 0);
    return {
      x, tint, glow,
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
    this.drawProposal(plan, a);
    this.drawSeats(plan, a, outcome);
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
    const now = frame.frameAt(step), words = Uint32Array.from(now.words), proposal = reversiProposal(words), white = reversiAgentWhite(words);
    const plan: Plan = {
      step, ahead, kind: 'hold', words, board: this.owners(words), after: null, white, afterWhite: white, member: reversiMember(words), afterMember: reversiMember(words),
      proposal, to: proposal, shown: reversiShown(words), options: reversiCount(words, REVERSI_LEGAL), preview: reversiFlips(words, proposal), afterPreview: [],
      action: -1, legal: this.legalOf(words), afterLegal: [], last: lastSquare(words), afterLast: lastSquare(words),
      agent: -1, rival: -1, rivals: [], toAgent: [], toRival: [], taken: false, split: false, result: 0, reward: now.reward, afterGame: reversiGame(words), doing: this.describe(frame, now),
    };
    this.plan = plan;
    if (!ahead) return plan;
    const later = peek.words;
    plan.to = reversiProposal(later);
    plan.afterWhite = reversiAgentWhite(later);
    plan.afterMember = reversiMember(later);
    plan.afterGame = reversiGame(later);
    plan.afterLegal = this.legalOf(later);
    plan.afterLast = lastSquare(later);
    plan.after = this.owners(later);
    plan.action = reversiAction(peek.actions);
    if (peek.done) {
      plan.kind = 'reset';
      plan.result = Math.sign(peek.reward);
      return plan;
    }
    const changed = this.squares.some((square) => plan.board[square] !== plan.after![square]);
    if (!changed) {
      if (plan.action !== REVERSI_NEXT || reversiShown(later) !== plan.shown + 1) throw new Error(`Reversi step ${step + 1} leaves the board unchanged without showing the next proposal`);
      plan.kind = 'next';
      plan.afterPreview = reversiFlips(later, plan.to);
    } else {
      if (plan.action === REVERSI_NEXT && plan.shown + 1 !== plan.options) throw new Error(`Reversi step ${step + 1} plays ${squareName(proposal)} before its clock ran out`);
      this.play(plan, step + 1);
    }
    return plan;
  }

  private play(plan: Plan, step: number): void {
    const after = plan.after!, added: number[] = [];
    plan.kind = 'play';
    plan.agent = plan.proposal;
    for (const square of this.squares) {
      const was = plan.board[square]!, now = after[square]!;
      if (was !== REVERSI_EMPTY && now === REVERSI_EMPTY) throw new Error(`Reversi step ${step} lifts the disc on ${squareName(square)}`);
      if (was === REVERSI_EMPTY && now !== REVERSI_EMPTY) added.push(square);
      else if (was === REVERSI_RIVAL && now === REVERSI_AGENT) plan.toAgent.push(square);
      else if (was === REVERSI_AGENT && now === REVERSI_RIVAL) plan.toRival.push(square);
    }
    if (!added.includes(plan.agent)) throw new Error(`Reversi step ${step} places nothing on ${squareName(plan.agent)} where the agent played`);
    plan.rivals = added.filter((square) => square !== plan.agent);
    if (plan.rivals.some((square) => after[square] !== REVERSI_RIVAL)) throw new Error(`Reversi step ${step} adds agent discs away from the proposal it played`);
    if (plan.rivals.length > 0 !== plan.rivals.includes(plan.afterLast)) throw new Error(`Reversi step ${step} adds discs its last-move word does not name`);
    plan.rival = plan.rivals.length > 0 ? plan.afterLast : -1;
    plan.taken = after[plan.agent] === REVERSI_RIVAL;
    if (plan.rival < 0 && (plan.toRival.length > 0 || plan.taken)) throw new Error(`Reversi step ${step} turns agent discs without a rival move`);
    plan.split = plan.rival >= 0;
  }

  private describe(frame: FrameUpdate, now: EnvFrame): string {
    const before = frame.frameAt(now.step - 1);
    if (now.step < 1 || before.step !== now.step - 1) return '—';
    const played = squareName(reversiProposal(before.words));
    if (reversiAction(now.actions) === REVERSI_PLAY) return `play ${played}`;
    if (reversiShown(before.words) + 1 === reversiCount(before.words, REVERSI_LEGAL)) return `next · clock plays ${played}`;
    return `next → ${squareName(reversiProposal(now.words))}`;
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
    const ledger = this.ledger, words = plan.words, own = reversiCount(words, REVERSI_OWN), opp = reversiCount(words, REVERSI_OPP), seen = plan.shown + 1;
    ledger.step = plan.step;
    ledger.game = reversiGame(words);
    ledger.proposal = `${squareName(plan.proposal)} · flips ${plan.preview.length}`;
    ledger.clock = `${seen} of ${plan.options}${seen === plan.options ? ' · next plays' : ''}`;
    ledger.rival = this.opponent === RIVALS.length - 1 ? `league member ${reversiMember(words) + 1}` : RIVALS[this.opponent]!.name;
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

  private disc(square: number, white: number, alpha: number, lift: number, marker: number, glow = 0): void {
    this.cell(square, this.at);
    this.put(this.at.x, this.at.z, white, 1, 0, alpha, glow, lift, marker, 1);
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
      this.disc(plan.proposal, agent, GHOST * (1 - fade), 0, 0);
      if (appear > 0) {
        for (const square of this.squares) if (after[square]) this.disc(square, face(after[square]!, plan.afterWhite), appear, 0.8 * (1 - appear), square === plan.afterLast ? appear : 0);
        this.disc(plan.to, face(REVERSI_AGENT, plan.afterWhite), GHOST * appear, 0, 0);
      }
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
      const land = landAt(a, first);
      if (plan.taken && a > second[0]) this.turnOver(plan.agent, agent, rival, flipAt(a, second, chebyshev(plan.agent, plan.rival), secondReach), plan.rival, 0);
      else this.disc(plan.agent, agent, lerp(GHOST, 1, clamp01(land * 3)), 0.8 * Math.sin(Math.PI * land), 0);
      const landing = landAt(a, second), alpha = clamp01(landing * 3);
      if (alpha > 0) for (const square of plan.rivals) this.disc(square, rival, alpha, 1 - smoothstep01(landing), square === plan.rival ? smoothstep01((landing - 0.8) / 0.2) : 0);
      const arriving = smoothstep01((a - 0.85) / 0.15);
      if (arriving > 0) this.disc(plan.to, agent, GHOST * arriving, 0, 0);
    } else {
      const k = plan.kind === 'next' ? smoothstep01(a / 0.85) : 0;
      for (const square of this.squares) {
        if (!board[square]) continue;
        const glow = (plan.preview.includes(square) ? 1 - k : 0) + (plan.afterPreview.includes(square) ? k : 0);
        this.disc(square, face(board[square]!, plan.white), 1, 0, square === plan.last ? 1 : 0, PREVIEW * glow);
      }
      this.disc(plan.proposal, agent, GHOST * (1 - k), 0, 0);
      if (k > 0) this.disc(plan.to, agent, GHOST * k, 0, 0);
    }

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

  private drawProposal(plan: Plan, a: number): void {
    const from = this.cell(plan.proposal, this.at), to = this.cell(plan.to, this.other);
    let x = from.x, z = from.z, press = 0, alpha = 1;
    if (plan.kind === 'next') {
      const k = smoothstep01(a / 0.85);
      x = lerp(from.x, to.x, k);
      z = lerp(from.z, to.z, k);
    } else if (plan.kind === 'play') {
      const k = smoothstep01((a - 0.85) / 0.15);
      press = Math.sin(Math.PI * clamp01(a / (LAND * ((plan.split ? FIRST : SOLO)[1]))));
      if (plan.split) alpha = 1 - 0.4 * Math.sin(Math.PI * clamp01((a - SECOND[0]) / (SECOND[1] - SECOND[0])));
      x = lerp(from.x, to.x, k);
      z = lerp(from.z, to.z, k);
    } else if (plan.kind === 'reset') {
      const late = a > 0.5;
      press = Math.sin(Math.PI * clamp01(a / 0.25));
      if (late) {
        x = to.x;
        z = to.z;
      }
      alpha = late ? smoothstep01((a - 0.6) / 0.3) : 1 - smoothstep01((a - 0.15) / 0.25);
    }
    this.bracketTint.set(this.accent).multiplyScalar(1.15 + 0.6 * press);
    this.bracketAlpha.value = alpha;
    this.bracket!.position.set(x, 0, z);
    this.bracket!.scale.setScalar(1 - 0.1 * press);
    const lamp = this.proposalLamp!;
    lamp.mesh.position.set(x, 0, z);
    lamp.tint.set(this.accent).multiplyScalar(0.8 + 4.2 * alpha + 6 * press);
    const [agentFlash, rivalFlash] = this.flashes as [Lamp, Lamp];
    agentFlash.mesh.visible = rivalFlash.mesh.visible = false;
    if (plan.kind !== 'play') return;
    this.cell(plan.agent, agentFlash.mesh.position);
    agentFlash.tint.set(this.accent).multiplyScalar(7 * flashAt(a, plan.split ? FIRST : SOLO));
    agentFlash.mesh.visible = true;
    if (plan.rival >= 0) {
      this.cell(plan.rival, rivalFlash.mesh.position);
      rivalFlash.tint.set(RIVAL).multiplyScalar(7 * flashAt(a, plan.split ? SECOND : SOLO));
      rivalFlash.mesh.visible = true;
    }
  }

  private drawSeats(plan: Plan, a: number, outcome: Outcome | null): void {
    const [agent, rival] = this.seats as [Seat, Seat], words = plan.words;
    const white = plan.kind === 'reset' && a > 0.72 ? plan.afterWhite : plan.white;
    const colour = (mine: boolean) => (mine === white ? 'white' : 'black');
    const member = this.opponent === RIVALS.length - 1 ? `member ${(plan.kind === 'reset' && a > 0.72 ? plan.afterMember : plan.member) + 1} · ` : '';
    const replying = plan.kind === 'play' && plan.split ? Math.sin(Math.PI * clamp01((a - SECOND[0]) / (SECOND[1] - SECOND[0]))) : 0;
    agent.title.caption.write('AGENT', 0.3, 700, 0.24);
    agent.sub.caption.write(`plays ${colour(true)}`, 0.2, 500, 0.08);
    rival.title.caption.write(RIVALS[this.opponent]!.title, 0.3, 700, 0.24);
    rival.sub.caption.write(`${member}plays ${colour(false)}`, 0.2, 500, 0.08);
    agent.count.caption.write(String(reversiCount(words, REVERSI_OWN)), 0.8, 600, 0);
    rival.count.caption.write(String(reversiCount(words, REVERSI_OPP)), 0.8, 600, 0);
    agent.unit.caption.write('DISCS', 0.15, 600, 0.3);
    rival.unit.caption.write('DISCS', 0.15, 600, 0.3);

    let agentSays = replying > 0.5 && plan.rivals.length > 1 ? 'PASSES' : 'TO MOVE';
    let rivalSays = replying > 0.5 ? 'MOVES' : reversiLast(words) === REVERSI_PASS ? 'PASSED' : 'WAITING';
    agent.status.tint.set(this.accent);
    rival.status.tint.set(RIVAL);
    agent.glow.value = 1 - 0.88 * replying;
    rival.glow.value = 0.12 + 0.88 * replying;
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
