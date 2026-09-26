import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { CelebrationPool } from '@kit/celebrate';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { instancedMesh } from '@kit/instances';
import { clamp01, smoothstep01 } from '@kit/mix';
import { Stage2DScene, type Stage2D } from '@kit/stage2d';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { countEvents } from '@sim/events';
import { pokerDecoder } from '@sim/envs/poker';
import type { EnvFrame } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { CARD_PAD, CARD_H, CARD_W, CHIP_LIFT, CHIP_QH, CHIP_QW, PILL_PAD, cardMaterial, chipMaterial, flat, gaugeMaterial, labelMaterial, makeAtlas, pillMaterial, tableMaterial, type Atlas, type TableShape } from './art';
import { AGENT, CALL, FICTITIOUS, FOLD, OPPONENTS, RAISE, RANK_NAMES, RIVAL, cardName, deckLeft, handOf, priorOf, rankOf, rulesOf, storyOf, versionOf, wordOf, type Hand, type Rules, type Story } from './hand';

type N<T extends string> = THREE.Node<T>;

const ACCENT = TOKENS.accents.LeducPoker!;
const RIVAL_TINT = 0xc3a6ff;
const GOLD = 0xffd166;
const PRINT = 0x7fcaa4;
const TONES: Record<string, number> = { FOLD: 0x565c69, CHECK: 0x46647f, CALL: 0x2b72b0, BET: 0xc98a25, RAISE: 0xd2562a };
const RANK_TINT = [0x5ec1ff, 0xff5ec8, 0xffd166] as const;
const SPLIT = 2;
const REST = 0.25;
const FAST = 6;
const WEIGHTS = { rest: 0, move: 1, gather: 0.8, reveal: 1, payout: 1.15, collect: 0.55, deal: 1 } as const;
const SHAPE: TableShape = {
  hx: 6.9, hz: 3.2, radius: 2.9, rail: 0.52,
  line: { x: 0, z: 0, w: 6.9, h: 3.3, radius: 1.55 },
  zones: [{ x: -5.0, z: 0.02, w: 1.72, h: 4.12, radius: 0.2 }, { x: 5.0, z: 0.02, w: 1.72, h: 4.12, radius: 0.2 }, { x: 0, z: 0, w: 1.3, h: 1.72, radius: 0.14 }],
};
const SEAT = [2.05, -2.05] as const;
const PLAQUE = [3.46, -3.46] as const;
const EDGE = [3.25, -3.25] as const;
const BET = { x: 1.2, z: [1.12, -1.12] } as const;
const BADGE = { x: 1.78, z: [2.05, -2.05] } as const;
const OPENER = { x: -1.0, z: [2.42, -2.42] } as const;
const COLLECT = { x: 1.35, z: [2.8, -2.72] } as const;
const FLASH = { x: 2.95, z: 2.4 } as const;
const BOARD = { x: 0, z: 0 } as const;
const DECK = { x: -2.25, z: 0 } as const;
const POT = { x: 2.25, z: 0.08, gap: 0.54 } as const;
const PANEL = { x: -5.5, tag: -4.62, label: -1.74, rows: [0.95, -0.95, 0], scale: 0.52, result: 1.68 } as const;
const STRIP = { x: 5.0, shift: 0.3, label: -1.74, top: -1.3, gap: 0.36, round: 0.6 } as const;
const MINI = 0.34;
const PILL_H = 0.34;
const TAG_H = 0.28;

const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const signed = (value: number) => Number.isInteger(value) ? `${value > 0 ? '+' : value < 0 ? '−' : ''}${formatCount(Math.abs(value))}` : formatSigned(value).replace('-', '−');
const tintOf = (p: number) => p === AGENT ? ACCENT : RIVAL_TINT;

interface Moving { x: number; z: number; alpha: number; fx?: number; fz?: number; tx?: number; tz?: number; lag?: number }
interface CardSprite extends Moving { angle: number; scale: number; face: number; peek: number; act: number; win: number; dim: number; card: { rank: number; suit: number } }
interface ChipSprite extends Moving { level: number; glow: number; kind: { denomination: number } }
interface PillSprite extends Moving { w: number; h: number; glow: number; pop: number; look: { fill: number; fillAlpha: number; edge: number; edgeWidth: number; word: string | null; ink: number; text: number } }
interface TextSprite extends Moving { h: number; look: { text: string; ink: number; glyphs: boolean; align: number } }
interface Layout { cards: Map<string, CardSprite>; chips: Map<string, ChipSprite>; pills: Map<string, PillSprite>; texts: Map<string, TextSprite>; lamps: number[]; win: number; winner: number }
interface Prior { hand: Hand; reward: number | null }
interface Table { live: Hand | null; line: number; street: number; pot: number; boardUp: boolean; revealed: boolean; folded: number; paid: number; acting: number; flash: number | null; prior: Prior | null; version: number }
interface Stage { table: Table; kind: keyof typeof WEIGHTS }
interface Plan { step: number; story: Story; before: EnvFrame; after: EnvFrame; stages: Stage[]; layouts: Layout[]; total: number; payoutAt: number; dealAt: number }
interface Rest { step: number; frame: EnvFrame; table: Table; layout: Layout; story: Story | null }
interface Batch { mesh: THREE.InstancedMesh; attrs: THREE.InstancedBufferAttribute[]; count: number }

const emptyLayout = (): Layout => ({ cards: new Map(), chips: new Map(), pills: new Map(), texts: new Map(), lamps: [0, 0], win: 0, winner: AGENT });

function blendOne<T extends Moving>(a: T, b: T, v: number): T {
  const out: Record<string, unknown> = { ...(b as unknown as Record<string, unknown>) };
  for (const key of Object.keys(a)) {
    const x = (a as unknown as Record<string, unknown>)[key], y = (b as unknown as Record<string, unknown>)[key];
    if (typeof x === 'number' && typeof y === 'number') out[key] = lerp(x, y, v);
  }
  return out as unknown as T;
}

function entering<T extends Moving>(b: T, v: number): T {
  const u = clamp01((v - (b.lag ?? 0)) / (1 - (b.lag ?? 0))), e = smoothstep01(u), out: Record<string, unknown> = { ...(b as unknown as Record<string, unknown>) };
  if ('face' in b) Object.assign(out, { face: (b as unknown as CardSprite).face * e, peek: (b as unknown as CardSprite).peek * e });
  if ('pop' in b) out.pop = Math.sin(Math.PI * e);
  if (b.fx === undefined || b.fz === undefined) return { ...out, alpha: b.alpha * e } as unknown as T;
  return { ...out, x: lerp(b.fx, b.x, e), z: lerp(b.fz, b.z, e), alpha: u > 0 ? b.alpha : 0 } as unknown as T;
}

function leaving<T extends Moving>(a: T, v: number): T {
  const e = smoothstep01(v), out: Record<string, unknown> = { ...(a as unknown as Record<string, unknown>) };
  if ('face' in a) Object.assign(out, { face: (a as unknown as CardSprite).face * (1 - e), peek: 0 });
  if (a.tx === undefined || a.tz === undefined) return { ...out, alpha: a.alpha * (1 - e) } as unknown as T;
  return { ...out, x: lerp(a.x, a.tx, e), z: lerp(a.z, a.tz, e), alpha: a.alpha * (1 - smoothstep01((v - 0.7) / 0.3)) } as unknown as T;
}

function blendMap<T extends Moving>(a: Map<string, T>, b: Map<string, T>, v: number, out: Map<string, T>): void {
  for (const [key, from] of a) {
    const to = b.get(key);
    out.set(key, to ? blendOne(from, to, smoothstep01(v)) : leaving(from, v));
  }
  for (const [key, to] of b) if (!a.has(key)) out.set(key, entering(to, v));
}

function blend(a: Layout, b: Layout, v: number): Layout {
  const out = emptyLayout(), e = smoothstep01(v);
  blendMap(a.cards, b.cards, v, out.cards);
  blendMap(a.chips, b.chips, v, out.chips);
  blendMap(a.pills, b.pills, v, out.pills);
  blendMap(a.texts, b.texts, v, out.texts);
  out.lamps = a.lamps.map((lamp, k) => lerp(lamp, b.lamps[k]!, e));
  out.win = lerp(a.win, b.win, e);
  out.winner = b.win > 0 ? b.winner : a.winner;
  return out;
}

class LeducPokerScene extends Stage2DScene {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 1;
  protected readonly stageName = 'leducPoker';

  private rules!: Rules;
  private atlas: Atlas | null = null;
  private cards: Batch | null = null;
  private chips: Batch | null = null;
  private pills: Batch | null = null;
  private labels: Batch | null = null;
  private seatLamps: THREE.Color[] = [];
  private winLamp: { mesh: THREE.Mesh; tint: THREE.Color } | null = null;
  private gauge: { mesh: THREE.Mesh; progress: N<'float'> & { value: number }; flash: N<'float'> & { value: number } } | null = null;
  private rivalName = '';
  private gaugeX = 0;
  private readonly clock = uniform(0);
  private readonly disposables: { dispose(): void }[] = [];
  private rest: Rest | null = null;
  private plan: Plan | null = null;
  private study = { version: -1, at: -1, looked: -Infinity };
  private wall = 0;
  private pace = 0;
  private lastPosition = 0;
  private celebrated = -1;
  private cheered = -Infinity;
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  private readonly spot = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();
  private readonly color = new THREE.Color();
  private readonly ledger = { hand: null as Hand | null, prior: null as Prior | null, story: null as Story | null, reward: 0, version: 0, progress: null as number | null, wins: 0, losses: 0, looked: -Infinity };

  override legend: LegendSpec = {
    title: 'LEDUC POKER',
    rows: [
      { key: 'hand', label: 'hand', tint: ACCENT, value: () => this.handText() },
      { key: 'cards', label: 'cards', tint: 0xf4efe2, value: () => this.cardsText() },
      { key: 'pot', label: 'pot', tint: GOLD, value: () => this.potText() },
      { key: 'line', label: 'line', tint: TOKENS.uiCyan, value: () => this.lineText() },
      { key: 'action', label: 'agent chose', tint: ACCENT, value: () => this.actionText() },
      { key: 'last', label: 'last hand', tint: GOLD, value: () => this.priorText() },
      { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => signed(this.ledger.reward) },
      { key: 'opponent', label: 'opponent', tint: RIVAL_TINT, value: () => this.opponentText() },
    ],
    note: () => `${formatCount(this.ledger.wins)} hands won · ${formatCount(this.ledger.losses)} lost`,
    bar: () => this.ledger.wins + this.ledger.losses > 0 ? this.ledger.wins / (this.ledger.wins + this.ledger.losses) : 0,
  };

  private handText(): string {
    const hand = this.ledger.hand;
    return hand ? `#${formatCount(hand.index + 1)} · ${hand.seat === 0 ? 'agent' : 'opponent'} opens` : '—';
  }

  private cardsText(): string {
    const hand = this.ledger.hand;
    if (!hand) return '—';
    return `agent ${cardName(hand.mine, this.rules)} · opponent ${cardName(hand.theirs, this.rules)} unseen · board ${hand.round ? cardName(hand.board, this.rules) : 'face down'}`;
  }

  private potText(): string {
    const hand = this.ledger.hand;
    return hand ? `${hand.chips[0] + hand.chips[1]} · agent ${hand.chips[0]} · opponent ${hand.chips[1]}` : '—';
  }

  private lineText(): string {
    const hand = this.ledger.hand;
    if (!hand || hand.line.length === 0) return 'no bets yet';
    return [0, 1].map((round) => hand.line.filter((m) => m.round === round).map((m) => `${m.mover === AGENT ? 'agent' : 'opp'} ${wordOf(m).toLowerCase()}`).join(' · ')).filter(Boolean).join(' │ ');
  }

  private actionText(): string {
    const story = this.ledger.story;
    if (!story) return '—';
    const done = story.moves[0]!, did = wordOf(done).toLowerCase();
    return done.move === story.intent ? did : `${['fold', 'call', 'raise'][story.intent]} → ${did}`;
  }

  private priorText(): string {
    const prior = this.ledger.prior;
    if (!prior) return '—';
    const last = prior.hand.line[prior.hand.line.length - 1]!, reward = prior.reward;
    const how = last.move === FOLD ? `${last.mover === AGENT ? 'agent' : 'opponent'} folded` : `showdown vs ${cardName(prior.hand.theirs, this.rules)}`;
    return reward === null ? how : `${reward === 0 ? 'split' : signed(reward)} · ${how}`;
  }

  private opponentText(): string {
    const rules = this.rules;
    if (rules.opponent !== FICTITIOUS) return rules.iterations > 0 ? `${OPPONENTS[rules.opponent]} average · ${formatCount(rules.iterations)} iterations` : OPPONENTS[rules.opponent]!;
    const progress = this.ledger.progress;
    return `${OPPONENTS[rules.opponent]} · plan ${formatCount(this.ledger.version)}${progress === null ? '' : ` · ${Math.floor(progress * 100)}% to the next`}`;
  }

  protected extent(ctx: SceneContext): THREE.Box2 {
    this.rules = rulesOf(ctx.environment);
    const rules = this.rules;
    this.rivalName = rules.opponent === FICTITIOUS ? 'FICTITIOUS PLAY' : rules.iterations > 0 ? `${OPPONENTS[rules.opponent]} · ${formatCount(rules.iterations)} ITERATIONS`.toUpperCase() : 'UNIFORM RANDOM';
    for (const value of [rules.ante, ...rules.raise]) if (!Number.isInteger(value) || value < 0 || value > 9) throw new Error(`${ctx.envId} stakes ${value}, which a single chip cannot print`);
    return new THREE.Box2(new THREE.Vector2(-8.1, -4.25), new THREE.Vector2(8.1, 4.85));
  }

  protected ambient(): THREE.Color {
    return new THREE.Color(0xfff1e0).multiplyScalar(0.9);
  }

  protected override poolOptions() {
    return { burstSize: 60, sighRadius: 1.6, sighHeight: 0.8 };
  }

  protected override overlayOptions(): GroundTruthOptions {
    return { debug: (frame, kit) => this.truth(frame, kit) };
  }

  protected build(ctx: SceneContext, { light, pool }: Stage2D): void {
    this.scene.background = new THREE.Color(0x060508);
    pool.group.traverse((object) => { object.renderOrder = 30; });
    const words = [...Object.keys(TONES), 'AGENT', this.rivalName, 'LAST HAND', 'THIS HAND', 'SHOWDOWN', '1ST', 'PLAN'];
    const atlas = makeAtlas([...RANK_NAMES, ...'0123456789', '+', '−', '±'], words);
    this.atlas = atlas;
    this.disposables.push(atlas.texture);
    const accent = this.accentOf(ctx);

    const table = new THREE.Mesh(flat(60, 40), tableMaterial(light, SHAPE, accent));
    table.renderOrder = 0;
    table.frustumCulled = false;
    this.scene.add(table);
    this.disposables.push(table.geometry, table.material as THREE.Material);

    const batch = (geometry: THREE.BufferGeometry, capacity: number, slots: number, material: (attrs: THREE.InstancedBufferAttribute[]) => THREE.Material, order: number): Batch => {
      const attrs = Array.from({ length: slots }, () => new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage));
      const mesh = instancedMesh(geometry, material(attrs), capacity);
      mesh.renderOrder = order;
      mesh.frustumCulled = false;
      mesh.count = 0;
      this.scene.add(mesh);
      this.disposables.push(geometry, mesh.material as THREE.Material, mesh);
      return { mesh, attrs, count: 0 };
    };
    this.chips = batch(flat(CHIP_QW, CHIP_QH), 40, 2, ([chip, value]) => chipMaterial(light, atlas, { chip: chip!, value: value! }), 2);
    this.cards = batch(flat(CARD_W + CARD_PAD, CARD_H + CARD_PAD), 24, 2, ([card, look]) => cardMaterial(light, atlas, { card: card!, look: look! }, accent, this.clock), 3);
    this.pills = batch(flat(1, 1), 56, 3, ([shape, fill, edge]) => pillMaterial(light, { shape: shape!, fill: fill!, edge: edge! }), 4);
    this.labels = batch(flat(1, 1), 128, 2, ([rect, tint]) => labelMaterial(atlas, { rect: rect!, tint: tint! }), 5);

    this.seatLamps = [AGENT, RIVAL].map((p) => {
      const tint = new THREE.Color(0, 0, 0), mesh = light.emitter(flat(2.2, 0.05), uniform(tint) as unknown as N<'vec3'>);
      mesh.position.set(0, 0, EDGE[p]);
      this.scene.add(mesh);
      this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
      return tint;
    });
    const winTint = new THREE.Color(0, 0, 0), win = light.emitter(new THREE.CircleGeometry(0.07, 20).rotateX(-Math.PI / 2), uniform(winTint) as unknown as N<'vec3'>);
    this.scene.add(win);
    this.disposables.push(win.geometry, win.material as THREE.Material);
    this.winLamp = { mesh: win, tint: winTint };

    const rivalWord = atlas.word(this.rivalName);
    this.gaugeX = (PILL_H * rivalWord.aspect + 0.3) / 2 + 0.5;
    if (this.rules.opponent === FICTITIOUS) {
      const progress = uniform(0), flash = uniform(0), mesh = new THREE.Mesh(flat(0.62, 0.62), gaugeMaterial(light, progress, flash, RIVAL_TINT));
      mesh.position.set(this.gaugeX, 0, PLAQUE[RIVAL]);
      mesh.renderOrder = 4;
      this.scene.add(mesh);
      this.disposables.push(mesh.geometry, mesh.material as THREE.Material);
      this.gauge = { mesh, progress: progress as unknown as N<'float'> & { value: number }, flash: flash as unknown as N<'float'> & { value: number } };
    }
  }

  protected override drop(): void {
    for (const item of this.disposables) item.dispose();
    this.disposables.length = 0;
    this.cards = this.chips = this.pills = this.labels = null;
    this.rest = null;
    this.plan = null;
  }

  private available(frame: FrameUpdate, step: number): EnvFrame | null {
    if (step < 0) return null;
    const found = frame.frameAt(step);
    return found.step === step ? found : null;
  }

  private rewardOf(frame: FrameUpdate, step: number, index: number): number | null {
    for (let s = step; s >= Math.max(0, step - 5); s--) {
      const f = this.available(frame, s);
      if (!f) return null;
      if (f.done && f.words[0] === index + 1) return f.reward;
    }
    return null;
  }

  private tableOf(frame: FrameUpdate, f: EnvFrame): Table {
    const live = handOf(f, this.rules), hand = priorOf(f, this.rules);
    return { live, line: live.line.length, street: live.round, pot: live.round, boardUp: live.round >= 1, revealed: false, folded: -1, paid: -1, acting: AGENT, flash: null, prior: hand ? { hand, reward: this.rewardOf(frame, f.step, hand.index) } : null, version: versionOf(f) };
  }

  private restAt(frame: FrameUpdate, step: number): Rest {
    if (this.rest?.step === step && this.rest.table.prior?.reward !== null && (this.rest.story || step === 0)) return this.rest;
    const f = frame.frameAt(step), table = this.tableOf(frame, f), before = this.available(frame, step - 1);
    const story = before ? storyOf(before, f, this.rules) : null;
    this.rest = { step, frame: f, table, layout: this.layout(table), story };
    this.plan = null;
    return this.rest;
  }

  private planAt(frame: FrameUpdate, rest: Rest, after: EnvFrame): Plan {
    if (this.plan?.step === rest.step) return this.plan;
    const story = storyOf(rest.frame, after, this.rules), stages = this.stagesOf(story, rest.table, this.tableOf(frame, after));
    const layouts = stages.map((stage) => this.layout(stage.table)), total = stages.reduce((sum, stage) => sum + WEIGHTS[stage.kind], 0);
    const at = (kind: keyof typeof WEIGHTS) => {
      let t = 0;
      for (const stage of stages) {
        if (stage.kind === kind) return REST + (1 - REST) * t / total;
        t += WEIGHTS[stage.kind];
      }
      return Infinity;
    };
    this.plan = { step: rest.step, story, before: rest.frame, after, stages, layouts, total, payoutAt: at('payout'), dealAt: at('deal') };
    return this.plan;
  }

  private stagesOf(story: Story, from: Table, to: Table): Stage[] {
    const stages: Stage[] = [{ table: from, kind: 'rest' }], line = story.hand.line, start = story.from.line.length;
    let table: Table = { ...from, live: story.hand };
    const push = (change: Partial<Table>, kind: keyof typeof WEIGHTS) => {
      table = { ...table, ...change };
      stages.push({ table, kind });
    };
    story.moves.forEach((m, j) => {
      const k = start + j, next = line[k + 1];
      push({ line: k + 1, folded: m.move === FOLD ? m.mover : table.folded, acting: story.moves[j + 1]?.mover ?? (story.ended ? -1 : AGENT) }, 'move');
      if (m.round === 0 && this.rules.rounds > 1 && (next ? next.round === 1 : !story.ended && story.to.round === 1)) push({ street: 1, pot: 1, boardUp: true }, 'gather');
    });
    if (story.ended) {
      if (line[line.length - 1]!.move === CALL) push({ revealed: true, pot: this.rules.rounds, acting: -1 }, 'reveal');
      push({ paid: story.reward > 0 ? AGENT : story.reward < 0 ? RIVAL : SPLIT, flash: story.reward, acting: -1 }, 'payout');
      push({ live: null, flash: null, prior: to.prior }, 'collect');
      push({ live: story.to, line: 0, street: 0, pot: 0, boardUp: false, revealed: false, folded: -1, paid: -1, acting: story.to.line[0]?.mover ?? AGENT, version: to.version }, 'deal');
      story.to.line.forEach((_, k) => push({ line: k + 1, acting: story.to.line[k + 1]?.mover ?? AGENT }, 'move'));
    }
    stages[stages.length - 1] = { table: to, kind: stages[stages.length - 1]!.kind };
    return stages;
  }

  private layout(table: Table): Layout {
    const out = emptyLayout(), rules = this.rules;
    for (let k = 0; k < deckLeft(rules); k++) out.cards.set(`deck${k}`, this.card(DECK.x - 0.022 * k, DECK.z - 0.03 * k, 0, 0, { rank: 0, suit: 0 }));
    this.plaque(out, table.version);
    this.text(out, 'lastLabel', SHAPE.zones[0]!.x, PANEL.label, 0.34, 'LAST HAND', PRINT, false, 0, 0.62);
    this.text(out, 'lineLabel', STRIP.x, STRIP.label, 0.34, 'THIS HAND', PRINT, false, 0, 0.62);
    if (table.live) this.handLayout(table, table.live, out);
    if (table.prior) this.priorLayout(table.prior, out);
    out.lamps = [AGENT, RIVAL].map((p) => (table.acting === p ? 1 : 0.12) + (table.paid === p || table.paid === SPLIT ? 0.6 : 0));
    out.win = table.paid === -1 ? 0 : table.paid === SPLIT ? 0.35 : 1;
    out.winner = table.paid === RIVAL ? RIVAL : AGENT;
    return out;
  }

  private card(x: number, z: number, face: number, peek: number, card: { rank: number; suit: number }, extra: Partial<CardSprite> = {}): CardSprite {
    return { x, z, alpha: 1, angle: 0, scale: 1, face, peek, act: 0, win: 0, dim: 0, card, ...extra };
  }

  private identity(card: number): { rank: number; suit: number } {
    return { rank: rankOf(card, this.rules), suit: card % this.rules.suits };
  }

  private pill(out: Layout, key: string, x: number, z: number, word: string | null, fill: number, edge: number, extra: Partial<PillSprite> = {}, look: Partial<PillSprite['look']> = {}): void {
    const text = look.text ?? PILL_H * 1.05, width = word ? text * this.atlas!.word(word).aspect + 0.12 : 0;
    out.pills.set(key, { x, z, alpha: 1, w: Math.max(width, extra.h ?? PILL_H), h: PILL_H, glow: 0, pop: 0, ...extra, look: { fill, fillAlpha: 0.94, edge, edgeWidth: 0.018, word, ink: 0xf6f3ea, text, ...look } });
  }

  private text(out: Layout, key: string, x: number, z: number, h: number, text: string, ink: number, glyphs: boolean, align = 0, alpha = 1, extra: Partial<TextSprite> = {}): void {
    out.texts.set(key, { x, z, alpha, h, ...extra, look: { text, ink, glyphs, align } });
  }

  private plaque(out: Layout, version: number): void {
    this.pill(out, 'plaqueAgent', 0, PLAQUE[AGENT], 'AGENT', 0x0b0d10, ACCENT, {}, { ink: ACCENT, edgeWidth: 0.022 });
    this.pill(out, 'plaqueRival', 0, PLAQUE[RIVAL], this.rivalName, 0x0b0d10, RIVAL_TINT, {}, { ink: RIVAL_TINT, edgeWidth: 0.022 });
    if (this.rules.opponent === FICTITIOUS) {
      this.text(out, 'plan', this.gaugeX + 0.4, PLAQUE[RIVAL], 0.3, 'PLAN', RIVAL_TINT, false, -1, 0.85);
      this.text(out, 'planNumber', this.gaugeX + 0.42 + 0.3 * this.atlas!.word('PLAN').aspect, PLAQUE[RIVAL] + 0.01, 0.4, String(version), RIVAL_TINT, true, -1, 0.95);
    }
  }

  private handLayout(table: Table, hand: Hand, out: Layout): void {
    const key = `h${hand.index}`, line = hand.line.slice(0, table.line), rules = this.rules;
    const pair = (card: number) => table.boardUp && rankOf(card, rules) === rankOf(hand.board, rules);
    const winning = (p: number) => table.paid === p || table.paid === SPLIT ? 1 : 0;
    const seatCard = (p: number, card: number, face: number, peek: number, lag: number) => {
      const folded = table.folded === p, toward = p === AGENT ? -1 : 1;
      out.cards.set(`${key}c${p}`, this.card(folded ? 0.2 : 0, SEAT[p]! + (folded ? 0.3 * toward : 0), face, peek, this.identity(card), {
        angle: folded ? 0.2 * toward : 0, act: table.acting === p ? 1 : 0, win: winning(p) * (table.paid === SPLIT ? 0.5 : 1), dim: folded ? 1 : 0, fx: DECK.x, fz: DECK.z, tx: DECK.x, tz: DECK.z, lag,
      }));
    };
    seatCard(AGENT, hand.mine, 1, 0, 0.18);
    seatCard(RIVAL, hand.theirs, table.revealed ? 1 : 0, table.revealed ? 0 : 1, 0);
    if (rules.rounds > 1) {
      const board = table.paid !== -1 && table.paid !== SPLIT && table.boardUp && pair(table.paid === AGENT ? hand.mine : hand.theirs);
      out.cards.set(`${key}b`, this.card(BOARD.x, BOARD.z, table.boardUp ? 1 : 0, 0, this.identity(hand.board), { win: board ? 1 : 0, fx: DECK.x, fz: DECK.z, tx: DECK.x, tz: DECK.z, lag: 0.34 }));
    }
    const first = hand.seat === 0 ? AGENT : RIVAL;
    this.pill(out, 'opener', OPENER.x, OPENER.z[first], '1ST', 0xe9e2cf, 0xb8a476, { w: 0.66, h: 0.34 }, { ink: 0x2a2418, text: 0.3, edgeWidth: 0.03 });

    const stacks = [0, 0, 0], bets = [0, 0], paid = [[0, 0, 0], [0, 0, 0]];
    const chip = (id: string, owner: number, denomination: number, round: number) => {
      let x: number, z: number, level: number;
      if (table.paid !== -1) {
        const to = table.paid === SPLIT ? owner : table.paid;
        x = COLLECT.x + (denomination - 1) * 0.54;
        z = COLLECT.z[to]!;
        level = paid[to]![denomination]!++;
      } else if (round < table.pot) {
        x = POT.x + (denomination - 1) * POT.gap;
        z = POT.z;
        level = stacks[denomination]!++;
      } else {
        x = BET.x;
        z = BET.z[owner]!;
        level = bets[owner]!++;
      }
      out.chips.set(id, { x, z, level, alpha: 1, glow: table.paid === AGENT || (table.paid === SPLIT && owner === AGENT) ? 1 : 0, kind: { denomination }, fx: BET.x - 0.35, fz: EDGE[owner] });
    };
    chip(`${key}a0`, AGENT, 0, -1);
    chip(`${key}a1`, RIVAL, 0, -1);
    line.forEach((m, k) => { for (let j = 0; j < m.units; j++) chip(`${key}m${k}c${j}`, m.mover, m.round + 1, m.round); });

    for (const p of [AGENT, RIVAL]) {
      let latest = -1;
      line.forEach((m, k) => { if (m.mover === p && m.round >= table.street) latest = k; });
      if (latest < 0) continue;
      const word = wordOf(line[latest]!);
      this.pill(out, `${key}t${latest}`, BADGE.x, BADGE.z[p]!, word, TONES[word]!, tintOf(p), {}, { edgeWidth: 0.026 });
    }
    let z = STRIP.top;
    line.forEach((m, k) => {
      if (k > 0 && m.round !== line[k - 1]!.round) z += STRIP.round;
      const word = wordOf(m);
      this.pill(out, `${key}l${k}`, STRIP.x + (m.mover === AGENT ? STRIP.shift : -STRIP.shift), z, word, TONES[word]!, tintOf(m.mover), { h: TAG_H }, { text: TAG_H * 1.05, fillAlpha: 0.9 });
      z += STRIP.gap;
    });
    if (rules.rounds > 1 && table.boardUp) {
      const split = line.findIndex((m) => m.round === 1), row = split < 0 ? line.length : split;
      out.cards.set(`${key}s`, this.card(STRIP.x, STRIP.top + row * STRIP.gap + (STRIP.round - STRIP.gap) / 2 - 0.02, 1, 0, this.identity(hand.board), { scale: MINI }));
    }
    if (table.flash !== null) {
      const value = table.flash, ink = value > 0 ? GOLD : value < 0 ? TOKENS.failureBlue : 0xc9ccd6;
      this.text(out, `${key}r`, FLASH.x, FLASH.z, 0.62, value === 0 ? '±0' : signed(value), ink, true, 0, 1, { tx: FLASH.x, tz: FLASH.z - 0.6 });
    }
  }

  private priorLayout(prior: Prior, out: Layout): void {
    const hand = prior.hand, key = `p${hand.index}`, last = hand.line[hand.line.length - 1]!, showdown = last.move === CALL;
    const winner = prior.reward === null || prior.reward === 0 ? -1 : prior.reward > 0 ? AGENT : RIVAL;
    const mini = (role: string, row: number, card: number, face: number, peek: number, extra: Partial<CardSprite>) => out.cards.set(`${key}${role}`, this.card(PANEL.x, PANEL.rows[row]!, face, peek, this.identity(card), { scale: PANEL.scale, ...extra }));
    mini('m', 0, hand.mine, 1, 0, { dim: last.move === FOLD && last.mover === AGENT ? 1 : 0, win: winner === AGENT ? 0.6 : 0 });
    mini('t', 1, hand.theirs, showdown ? 1 : 0, showdown ? 0 : 1, { dim: last.move === FOLD && last.mover === RIVAL ? 1 : 0, win: winner === RIVAL ? 0.6 : 0 });
    if (this.rules.rounds > 1) mini('b', 2, hand.board, hand.round >= 1 ? 1 : 0, 0, {});
    if (showdown) this.pill(out, `${key}e`, PANEL.tag, PANEL.rows[2], 'SHOWDOWN', 0x1b3d31, PRINT, { h: TAG_H }, { text: TAG_H * 0.9, ink: 0xdff3e8 });
    else this.pill(out, `${key}e`, PANEL.tag, PANEL.rows[last.mover === AGENT ? 0 : 1]!, 'FOLD', TONES.FOLD!, tintOf(last.mover), { h: TAG_H }, { text: TAG_H * 0.9 });
    if (prior.reward !== null) {
      const value = prior.reward, fill = value > 0 ? 0x6b5212 : value < 0 ? 0x2a3a5c : 0x3a3d45, edge = value > 0 ? GOLD : value < 0 ? TOKENS.failureBlue : 0x9aa0ad;
      out.pills.set(`${key}v`, { x: SHAPE.zones[0]!.x, z: PANEL.result, alpha: 1, w: 1.1, h: 0.4, glow: 0, pop: 0, look: { fill, fillAlpha: 0.92, edge, edgeWidth: 0.022, word: null, ink: edge, text: 0 } });
      this.text(out, `${key}n`, SHAPE.zones[0]!.x, PANEL.result, 0.5, value === 0 ? '±0' : signed(value), 0xfbf6e8, true);
    }
  }

  protected tick(frame: FrameUpdate, { pool, time }: Stage2D): void {
    const dt = frame.dtReal;
    this.wall += dt;
    this.clock.value = this.wall;
    pool.update(dt, time);
    if (frame.scrubbed) {
      this.pace = 0;
      pool.clear();
    } else if (dt > 0) this.pace += (Math.abs(frame.position - this.lastPosition) / dt - this.pace) * (1 - Math.exp(-4 * dt));
    this.lastPosition = frame.position;
    const step = frame.state.step, rest = this.restAt(frame, step), fast = this.pace > FAST;
    const after = fast ? null : this.available(frame, step + 1);
    let layout = rest.layout, flash = 0;
    if (after) {
      const plan = this.planAt(frame, rest, after), u = clamp01(frame.position - step);
      layout = this.viewAt(plan, u);
      if (frame.scrubbed) this.celebrated = u >= plan.payoutAt ? step + 1 : step;
      else if (u >= plan.payoutAt && this.celebrated < step + 1) {
        this.celebrate(plan.before, plan.after, pool, time, 1);
        this.celebrated = step + 1;
      }
      if (plan.story.adopted && u >= plan.dealAt) flash = 1 - clamp01((u - plan.dealAt) / (1 - plan.dealAt));
    } else if (frame.scrubbed) this.celebrated = step;
    else if (frame.events.length > 0 && this.celebrated < step && this.wall - this.cheered > 0.3) {
      const before = this.available(frame, step - 1);
      if (before) this.celebrate(before, rest.frame, pool, time, 0.45);
      this.celebrated = step;
      this.cheered = this.wall;
    }
    this.draw(layout);
    this.record(frame, rest, flash);
  }

  private viewAt(plan: Plan, u: number): Layout {
    if (u <= REST) return plan.layouts[0]!;
    let t = ((u - REST) / (1 - REST)) * plan.total;
    for (let k = 1; k < plan.stages.length; k++) {
      const weight = WEIGHTS[plan.stages[k]!.kind];
      if (t <= weight || k === plan.stages.length - 1) return blend(plan.layouts[k - 1]!, plan.layouts[k]!, clamp01(t / weight));
      t -= weight;
    }
    return plan.layouts[plan.layouts.length - 1]!;
  }

  private celebrate(before: EnvFrame, after: EnvFrame, pool: CelebrationPool, time: number, size: number): void {
    for (const event of pokerDecoder.deriveEvents(before, after)) {
      if (event.type === 'goal') {
        pool.burst(this.spot.set(COLLECT.x, 38, COLLECT.z[AGENT]), TOKENS.celebrationGold, 0.8 * size, time);
        pool.ripple(this.spot.set(COLLECT.x, 0, COLLECT.z[AGENT]), TOKENS.celebrationGold, 0.35 * size, time);
      }
      if (event.type === 'harm') pool.sigh(0.55 * size, this.spot.set(0, 0, SEAT[AGENT]), time);
    }
  }

  private record(frame: FrameUpdate, rest: Rest, flash: number): void {
    const ledger = this.ledger, step = rest.step;
    ledger.hand = rest.table.live;
    ledger.prior = rest.table.prior;
    ledger.story = rest.story;
    ledger.reward = rest.frame.reward;
    if (this.wall - ledger.looked > 1) {
      ledger.wins = countEvents(frame.eventLog, 'goal', -1, step);
      ledger.losses = countEvents(frame.eventLog, 'harm', -1, step);
      ledger.looked = this.wall;
    }
    if (this.rules.opponent !== FICTITIOUS || !this.gauge) return;
    ledger.version = versionOf(rest.frame);
    if (this.study.version !== ledger.version || (this.study.at < 0 && this.wall - this.study.looked > 0.5)) this.scan(frame, step, ledger.version);
    ledger.progress = this.study.at < 0 ? null : Math.min(1, (step - this.study.at + 1) / this.rules.sweep);
    this.gauge.progress.value = flash > 0 ? 0 : ledger.progress ?? 0;
    this.gauge.flash.value = flash;
  }

  private scan(frame: FrameUpdate, step: number, version: number): void {
    this.study = { version, at: -1, looked: this.wall };
    for (let s = step; s >= Math.max(0, step - this.rules.sweep - 64); s--) {
      const f = this.available(frame, s);
      if (!f) return;
      if (versionOf(f) !== version) {
        this.study.at = s + 1;
        return;
      }
      if (s === 0) this.study.at = 0;
    }
  }

  private draw(layout: Layout): void {
    const cards = this.cards, chips = this.chips, pills = this.pills, labels = this.labels, atlas = this.atlas;
    if (!cards || !chips || !pills || !labels || !atlas) return;
    cards.count = chips.count = pills.count = labels.count = 0;
    for (const sprite of layout.cards.values()) {
      if (sprite.alpha <= 0.002) continue;
      const i = this.slot(cards), shown = sprite.face >= 0.5 ? 1 : 0, squeeze = Math.max(0.03, Math.abs(Math.cos(Math.PI * sprite.face)));
      this.quaternion.setFromAxisAngle(this.axis, sprite.angle);
      cards.mesh.setMatrixAt(i, this.matrix.compose(this.spot.set(sprite.x, 0, sprite.z), this.quaternion, this.scale.set(sprite.scale * squeeze, 1, sprite.scale)));
      cards.attrs[0]!.setXYZW(i, sprite.card.rank, sprite.card.suit, shown, shown ? 0 : sprite.peek);
      cards.attrs[1]!.setXYZW(i, sprite.alpha, sprite.act, sprite.win, sprite.dim);
    }
    const stacked = [...layout.chips.values()].filter((c) => c.alpha > 0.002).sort((a, b) => a.z - b.z || a.level - b.level);
    for (const sprite of stacked) {
      const i = this.slot(chips), value = atlas.glyph(String([this.rules.ante, ...this.rules.raise][sprite.kind.denomination]));
      chips.mesh.setMatrixAt(i, this.matrix.makeTranslation(sprite.x, 0, sprite.z - sprite.level * CHIP_LIFT));
      chips.attrs[0]!.setXYZW(i, sprite.kind.denomination, sprite.alpha, sprite.glow, value.aspect);
      chips.attrs[1]!.setXYZW(i, value.rect.x, value.rect.y, value.rect.z, value.rect.w);
    }
    for (const sprite of layout.pills.values()) {
      if (sprite.alpha <= 0.002) continue;
      const i = this.slot(pills), grow = 1 + 0.22 * sprite.pop, look = sprite.look;
      pills.mesh.setMatrixAt(i, this.matrix.makeScale((sprite.w + PILL_PAD) * grow, 1, (sprite.h + PILL_PAD) * grow).setPosition(sprite.x, 0, sprite.z));
      pills.attrs[0]!.setXYZW(i, sprite.w * grow, sprite.h * grow, sprite.h * grow / 2, sprite.glow);
      this.color.set(look.fill);
      pills.attrs[1]!.setXYZW(i, this.color.r, this.color.g, this.color.b, look.fillAlpha * sprite.alpha);
      this.color.set(look.edge);
      pills.attrs[2]!.setXYZW(i, this.color.r, this.color.g, this.color.b, look.edgeWidth * (sprite.alpha > 0.5 ? 1 : sprite.alpha * 2));
      if (look.word) this.label(atlas.word(look.word), sprite.x, sprite.z, look.text * grow, look.ink, sprite.alpha);
    }
    for (const sprite of layout.texts.values()) {
      if (sprite.alpha <= 0.002) continue;
      const look = sprite.look;
      if (!look.glyphs) {
        const word = atlas.word(look.text);
        this.label(word, sprite.x - look.align * sprite.h * word.aspect / 2, sprite.z, sprite.h, look.ink, sprite.alpha);
        continue;
      }
      const glyphs = [...look.text].map((g) => atlas.glyph(g)), advance = (g: { aspect: number }) => sprite.h * (g.aspect - 0.14);
      const width = glyphs.reduce((sum, g) => sum + advance(g), 0);
      let x = sprite.x - width * (look.align + 1) / 2;
      for (const g of glyphs) {
        this.label(g, x + advance(g) / 2, sprite.z, sprite.h, look.ink, sprite.alpha);
        x += advance(g);
      }
    }
    for (const batch of [cards, chips, pills, labels]) {
      batch.mesh.count = batch.count;
      batch.mesh.instanceMatrix.needsUpdate = true;
      for (const attr of batch.attrs) attr.needsUpdate = true;
    }
    this.seatLamps.forEach((tint, p) => tint.set(tintOf(p)).multiplyScalar(2.2 * layout.lamps[p]!));
    const win = this.winLamp;
    if (win) {
      win.mesh.position.set(COLLECT.x - 0.3, 0, COLLECT.z[layout.winner]!);
      win.tint.set(layout.winner === AGENT ? GOLD : RIVAL_TINT).multiplyScalar(5 * layout.win);
    }
  }

  private label(sprite: { rect: THREE.Vector4; aspect: number }, x: number, z: number, h: number, ink: number, alpha: number): void {
    const labels = this.labels!, i = this.slot(labels);
    labels.mesh.setMatrixAt(i, this.matrix.makeScale(h * sprite.aspect, 1, h).setPosition(x, 0, z));
    labels.attrs[0]!.setXYZW(i, sprite.rect.x, sprite.rect.y, sprite.rect.z, sprite.rect.w);
    this.color.set(ink);
    labels.attrs[1]!.setXYZW(i, this.color.r, this.color.g, this.color.b, alpha);
  }

  private slot(batch: Batch): number {
    if (batch.count >= batch.mesh.instanceMatrix.count) throw new Error(`LeducPoker ${batch.mesh.name || 'batch'} holds ${batch.mesh.instanceMatrix.count} instances`);
    return batch.count++;
  }

  private truth(frame: FrameUpdate, kit: DebugKit): void {
    const f = frame.frameAt(frame.state.step), hand = handOf(f, this.rules), rules = this.rules;
    const marks = kit.marks('cards', 3), spots: [number, number, number][] = [[hand.mine, 0, SEAT[AGENT]], [hand.theirs, 0, SEAT[RIVAL]], [hand.board, BOARD.x, BOARD.z]];
    for (const [card, x, z] of spots.slice(0, rules.rounds > 1 ? 3 : 2)) marks.set(x + CARD_W / 2 + 0.12, 0.1, z - CARD_H / 2 + 0.12, card % rules.suits ? 0.14 : 0.09, RANK_TINT[rankOf(card, rules)]!);
    const stakes = kit.bars('stakes', 2);
    for (const p of [AGENT, RIVAL]) stakes.set(BET.x + 0.45 + hand.chips[p]! * 0.07, 0.1, BET.z[p]!, 0.02, tintOf(p), hand.chips[p]! * 0.14, 0.1);
    kit.marks('opener', 1).set(OPENER.x - 0.5, 0.1, OPENER.z[hand.seat === 0 ? AGENT : RIVAL]!, 0.1, 0xffffff);
    const path = kit.curve('node', 24, TOKENS.uiCyan);
    let x = STRIP.x - 1.1, z = 2.55;
    path.point(x, 0.1, z);
    hand.line.forEach((m, k) => {
      if (k > 0 && m.round !== hand.line[k - 1]!.round) { x += 0.2; path.point(x, 0.1, z); }
      x += 0.3;
      z += m.move === RAISE ? -0.2 : 0.2;
      path.point(x, 0.1, z);
    });
    kit.marks('node', 1).set(x, 0.1, z, 0.09, hand.round ? RANK_TINT[rankOf(hand.board, rules)]! : 0xffffff);
    const story = this.rest?.step === f.step ? this.rest.story : null;
    if (story) kit.marks('intent', 1).set(BADGE.x + 0.75, 0.1, BADGE.z[AGENT], 0.08, TONES[['FOLD', 'CALL', 'RAISE'][story.intent]!]!);
  }
}

export default (() => new LeducPokerScene()) satisfies SceneFactory;
