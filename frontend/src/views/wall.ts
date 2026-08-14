import type { ViewHandle, ViewMount } from '../router';
import type { EnvResult, RecordingRef, RunDetail, RunIndexEntry, RunMeta } from '@shared/types';
import { ENVS, ENV_IDS, isEnvId, type EnvCategory, type EnvId } from '@shared/envs';
import type { PlaybackSource } from '@kit/playback';
import { makeGhostLayer, type GhostLayer } from '@kit/ghosts';
import { TOKENS } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import { fieldIndex, tapeValue, type Tape } from '../tape';
import {
  accentCss,
  badge,
  el,
  emptyState,
  errorState,
  eyebrow,
  linkButton,
  spinner,
  viewShell,
} from '../components/table';
import { sparkline, type SparklineHandle } from '../components/sparkline';
import { mountRunScience, type RunScienceHandle } from './run';
import {
  binnedRates,
  deliveredRate,
  envLanes,
  envName,
  envShares,
  formatAgo,
  formatCount,
  formatPercent,
  formatRate,
  formatSigned,
  formatStep,
  laneDeath,
  laneRetention,
  loadRunDetail,
  loadTape,
  refreshTape,
  runIndex,
  sortRuns,
} from '../state';
import {
  loadHero,
  makeInitialSource,
  recordingRef,
  seedsFor,
  seekTargetForTick,
  type HeroHandle,
  type TimelineTick,
} from '../theater/loader';
import { clampPosition, formatClock, stepPosition } from '../theater/player';
import {
  mountSceneHost,
  type HostFrameInfo,
  type SceneHostGhosts,
  type SceneHostHandle,
} from '../theater/sceneHost';

export const WALL_SPEEDS: readonly number[] = [1, 4, 16, 64, 256, 1024];

export const SPARK_ROW_BUDGET = 128;

export const SPARK_BINS = 64;

export const LIVE_POLL_MS = 2_000;

export const DETAIL_POLL_MS = 10_000;

export type WallSort = 'rate' | 'seed' | 'retention';
export const WALL_SORTS: readonly WallSort[] = ['rate', 'seed', 'retention'];

const SORT_LABEL: Record<WallSort, string> = {
  rate: 'reward rate',
  seed: 'seed',
  retention: 'retention',
};

export interface WallParams {
  run?: string;
  env?: EnvId;
  seed?: number;
  at?: number;
  speed?: number;
  sort?: WallSort;
}

export function parseWallParams(hash: string): WallParams {
  const out: WallParams = {};
  const q = hash.indexOf('?');
  if (q < 0) return out;
  const params = new URLSearchParams(hash.slice(q + 1));

  const run = params.get('run');
  if (run !== null && run.trim().length > 0) out.run = run.trim();

  const env = params.get('env');
  if (env !== null && isEnvId(env.trim())) out.env = env.trim() as EnvId;

  const seed = params.get('seed');
  if (seed !== null && /^\d+$/.test(seed.trim())) {
    const value = Number(seed.trim());
    if (Number.isSafeInteger(value) && value >= 0) out.seed = value;
  }

  const at = params.get('at');
  if (at !== null) {
    const value = Number(at);
    if (Number.isFinite(value) && value >= 0) out.at = Math.floor(value);
  }

  const speed = params.get('speed');
  if (speed !== null) {
    const value = Number(speed);
    if (WALL_SPEEDS.includes(value)) out.speed = value;
  }

  const sort = params.get('sort');
  if (sort !== null && (WALL_SORTS as readonly string[]).includes(sort.trim())) {
    out.sort = sort.trim() as WallSort;
  }
  return out;
}

export function wallHref(params: WallParams = {}): string {
  const parts: string[] = [];
  const push = (key: string, value: string): void => {
    parts.push(`${key}=${encodeURIComponent(value)}`);
  };
  if (params.run !== undefined && params.run !== '') push('run', params.run);
  if (params.env !== undefined) push('env', params.env);
  if (params.seed !== undefined && Number.isFinite(params.seed)) push('seed', String(Math.floor(params.seed)));
  if (params.at !== undefined && Number.isFinite(params.at) && params.at > 0) push('at', String(Math.floor(params.at)));
  if (params.speed !== undefined && params.speed !== 1) push('speed', String(params.speed));
  if (params.sort !== undefined && params.sort !== 'rate') push('sort', params.sort);
  return parts.length > 0 ? `#/?${parts.join('&')}` : '#/';
}

export function urlPosition(clockPosition: number | null, pending: number): number | null {
  const at = clockPosition !== null && Number.isFinite(clockPosition) ? clockPosition : pending;
  if (!Number.isFinite(at) || at <= 0) return null;
  return Math.floor(at);
}

export type FeaturedSource = 'param' | 'keeper' | 'newest' | null;

export interface FeaturedPick {
  runId: string | null;
  source: FeaturedSource;
}

export function resolveFeaturedRun(
  entries: readonly RunIndexEntry[],
  requested?: string,
): FeaturedPick {
  const byId = new Map<string, RunIndexEntry>();
  for (const entry of entries) byId.set(entry.id, entry);

  if (requested !== undefined && byId.has(requested)) return { runId: requested, source: 'param' };

  const keeper = entries.find((entry) => entry.role === 'keeper');
  if (keeper?.hasRecordings === true) return { runId: keeper.id, source: 'keeper' };

  const started = (entry: RunIndexEntry): number => (entry.startedAt ? Date.parse(entry.startedAt) : 0);
  const newest = (list: readonly RunIndexEntry[]): RunIndexEntry | null =>
    [...list].sort((a, b) => {
      const ta = started(a);
      const tb = started(b);
      if (Number.isFinite(tb) && Number.isFinite(ta) && tb !== ta) return tb - ta;
      return a.id.localeCompare(b.id);
    })[0] ?? null;

  const playable = entries.filter((entry) => entry.hasRecordings);
  const complete = newest(playable.filter((entry) => entry.complete));
  if (complete) return { runId: complete.id, source: 'newest' };
  const any = newest(playable);
  if (any) return { runId: any.id, source: 'newest' };
  return { runId: null, source: null };
}

export interface HeroSelection {
  leader: number | null;
  promoted: number | null;
}

export function heroSeed(selection: HeroSelection): number | null {
  return selection.promoted ?? selection.leader;
}

export function promoteSeed(
  selection: HeroSelection,
  seed: number,
  playable: (seed: number) => boolean,
): HeroSelection {
  if (!playable(seed)) return selection;
  if (selection.promoted === seed) return selection;
  return { ...selection, promoted: seed };
}

export function escapeSelection(selection: HeroSelection): HeroSelection {
  if (selection.promoted === null) return selection;
  return { ...selection, promoted: null };
}

export function setLeader(
  selection: HeroSelection,
  leader: number | null,
  playable: (seed: number) => boolean,
): HeroSelection {
  const promoted = selection.promoted !== null && playable(selection.promoted) ? selection.promoted : null;
  return { leader, promoted };
}

export function degradedReason(
  recordings: readonly RecordingRef[],
  envId: EnvId | null,
): string | null {
  if (recordings.length === 0) return 'no recordings on this run';
  if (envId === null) return 'no environment selected';
  if (seedsFor(recordings, envId).length === 0) return `no recordings for ${ENVS[envId].name} in this run`;
  return null;
}

export const RECORD_RECIPE = 'This run has no recording for that selection.';

export interface SeedRow {
  seed: number;
  lane: number;
  rate: number | null;
  retention: number | null;
  divergedAt: number | null;
  spark: (number | null)[];
  sparkXs: number[];
  recorded: boolean;
}

export function buildSeedRows(
  tape: Tape,
  meta: Pick<RunMeta, 'envSchedule' | 'seeds' | 'steps'>,
  envId: string,
  recorded: ReadonlySet<number>,
): SeedRow[] {
  const { lanes, seeds } = envLanes(meta, envId);
  const maxStep = tape.rows > 0 ? tape.stepEnd[tape.rows - 1] ?? 0 : 0;
  const spark = binnedRates(tape, lanes, SPARK_BINS);
  const sparkXs = spark.x.map((x) => (maxStep > 0 ? x / maxStep : 0));
  return lanes.map((lane, li) => ({
    seed: seeds[li] ?? 0,
    lane,
    rate: deliveredRate(tape, lane),
    retention: laneRetention(tape, lane, meta.steps)?.retention ?? null,
    divergedAt: laneDeath(tape, lane)?.step ?? null,
    spark: spark.perLane[li] ?? [],
    sparkXs,
    recorded: recorded.has(seeds[li] ?? 0),
  })).sort((a, b) => a.seed - b.seed);
}

export function sortPermutation(rows: readonly SeedRow[], sort: WallSort): number[] {
  const order = rows.map((_, index) => index);
  const direction = sort === 'seed' ? 1 : -1;
  const keyOf = (row: SeedRow): number | null =>
    sort === 'seed' ? row.seed : sort === 'retention' ? row.retention : row.rate;
  order.sort((a, b) => {
    const rowA = rows[a];
    const rowB = rows[b];
    if (rowA === undefined || rowB === undefined) return a - b;
    const keyA = keyOf(rowA);
    const keyB = keyOf(rowB);
    const nullA = keyA === null || !Number.isFinite(keyA);
    const nullB = keyB === null || !Number.isFinite(keyB);
    if (nullA && nullB) return a - b;
    if (nullA) return 1;
    if (nullB) return -1;
    if (keyA !== keyB) return (keyA - keyB) * direction;
    return a - b;
  });
  return order;
}

export function nextSort(sort: WallSort): WallSort {
  const index = WALL_SORTS.indexOf(sort);
  return WALL_SORTS[(index + 1) % WALL_SORTS.length] ?? 'rate';
}

export interface PopulationFigures {
  positive: string;
  nonDiverged: string;
  retention: string;
  solved: string;
  perGflop: string;
  source: 'summary' | 'measured' | 'none';
}

function finite(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function populationFigures(
  slice: EnvResult | null,
  rows: readonly SeedRow[],
): PopulationFigures {
  const rates = rows.map((row) => row.rate).filter((v): v is number => v !== null && Number.isFinite(v));
  const diverged = rows.filter((row) => row.divergedAt !== null).length;
  const retained = rows.filter((row) => row.retention !== null && row.retention >= 0.5).length;

  if (slice !== null && finite(slice.rewardRateIQM) !== null) {
    const seeds = finite(slice.seeds) ?? rows.length;
    const dead = finite(slice.divergence?.diverged) ?? diverged;
    const positive = finite(slice.positiveFraction);
    const retention = finite(slice.retentionFraction);
    const solved = finite(slice.solvedSeeds);
    const perGflop = finite(slice.bestSeedRewardPerGFLOP);
    return {
      positive: positive === null ? '—' : formatPercent(positive, 0),
      nonDiverged: `${formatCount(seeds - dead)}/${formatCount(seeds)}`,
      retention: retention === null ? '—' : formatPercent(retention, 0),
      solved: solved === null ? '—' : formatCount(solved),
      perGflop: perGflop === null ? '—' : formatRate(perGflop),
      source: 'summary',
    };
  }

  if (rows.length === 0) {
    return { positive: '—', nonDiverged: '—', retention: '—', solved: '—', perGflop: '—', source: 'none' };
  }
  return {
    positive: rates.length === 0
      ? '—'
      : formatPercent(rates.filter((v) => v > 0).length / rates.length, 0),
    nonDiverged: `${formatCount(rows.length - diverged)}/${formatCount(rows.length)}`,
    retention: formatPercent(retained / rows.length, 0),
    solved: '—',
    perGflop: '—',
    source: 'measured',
  };
}

export function runScheduleLine(meta: Pick<RunMeta, 'batch' | 'steps' | 'envSchedule'>): string {
  const shares = envShares(meta);
  const horizon = `${formatStep(meta.steps)} steps`;
  if (shares.length === 0) return `${formatCount(meta.batch)} seeds · ${horizon}`;
  if (shares.length === 1) {
    return `${envName(shares[0]!.envId)} · ${formatCount(meta.batch)} seeds · ${horizon}`;
  }
  const counts = new Set(shares.map((share) => share.lanes));
  const perEnv = counts.size === 1 ? shares[0]?.lanes ?? meta.batch : null;
  const seeds = perEnv === null ? `${formatCount(meta.batch)} seeds total` : `${formatCount(perEnv)} seeds`;
  return `${shares.length} tasks × ${seeds} × ${horizon}`;
}

export type WallAction =
  | { kind: 'toggle' }
  | { kind: 'step'; frames: number }
  | { kind: 'env'; direction: 1 | -1 }
  | { kind: 'sort' }
  | { kind: 'debug' }
  | { kind: 'fullscreen' }
  | { kind: 'tier' }
  | { kind: 'escape' };

export function wallKeyAction(key: string, shift = false): WallAction | null {
  switch (key) {
    case ' ':
    case 'Spacebar':
    case 'k':
    case 'K':
      return { kind: 'toggle' };
    case 'ArrowRight':
      return { kind: 'step', frames: shift ? 200 : 20 };
    case 'ArrowLeft':
      return { kind: 'step', frames: shift ? -200 : -20 };
    case '[':
    case '{':
      return { kind: 'env', direction: -1 };
    case ']':
    case '}':
      return { kind: 'env', direction: 1 };
    case 's':
    case 'S':
      return { kind: 'sort' };
    case 'g':
    case 'G':
      return { kind: 'debug' };
    case 'f':
    case 'F':
      return { kind: 'fullscreen' };
    case 't':
    case 'T':
      return { kind: 'tier' };
    case 'Escape':
      return { kind: 'escape' };
    default:
      return null;
  }
}

export function cycleEnv(list: readonly EnvId[], current: EnvId | null, direction: 1 | -1): EnvId | null {
  if (list.length === 0) return null;
  const index = current === null ? -1 : list.indexOf(current);
  const next = index < 0 ? (direction > 0 ? 0 : list.length - 1) : (index + direction + list.length) % list.length;
  return list[next] ?? null;
}

export function wallEnvs(meta: Pick<RunMeta, 'envSchedule'>, recordings: readonly RecordingRef[]): EnvId[] {
  const seen = new Set<EnvId>();
  for (const share of envShares(meta)) seen.add(share.envId);
  for (const rec of recordings) if (isEnvId(rec.envId)) seen.add(rec.envId);
  return ENV_IDS.filter((id) => seen.has(id));
}

export const WALL_GHOST_CAP = 512;

export const POSE_FIELD_NAMES: readonly string[] = [
  'pose0', 'pose1', 'pose2', 'pose3', 'pose4', 'pose5', 'pose6', 'pose7', 'pose8',
];

export function tapeWindowSteps(tape: Pick<Tape, 'stepStart' | 'stepEnd'>): number {
  const window = (tape.stepEnd[0] ?? 0) - (tape.stepStart[0] ?? 0);
  return window > 0 ? window : 1;
}

export function ghostRowIndex(position: number, windowSteps: number, rows: number): number {
  if (rows <= 0) return 0;
  if (!Number.isFinite(position) || position <= 0) return 0;
  const index = Math.ceil(position / Math.max(1, windowSteps)) - 1;
  return index < 0 ? 0 : index >= rows ? rows - 1 : index;
}

export function aliveRowCount(tape: Pick<Tape, 'rows' | 'stepEnd'>, deathStep: number | null): number {
  if (deathStep === null) return tape.rows;
  let n = 0;
  while (n < tape.rows && (tape.stepEnd[n] ?? Infinity) <= deathStep) n++;
  return n;
}

export function ghostAlive(deathStep: number | null, stepEnd: number): boolean {
  return deathStep === null || stepEnd <= deathStep;
}

export function rateQuantiles(rows: readonly SeedRow[]): Map<number, number> {
  const rated = rows.filter((row): row is SeedRow & { rate: number } =>
    row.rate !== null && Number.isFinite(row.rate));
  const out = new Map<number, number>();
  for (const row of rows) out.set(row.seed, 0);
  if (rated.length === 0) return out;
  const sorted = [...rated].sort((a, b) => a.rate - b.rate);
  const span = Math.max(1, sorted.length - 1);
  let rank = 0;
  for (let i = 0; i < sorted.length; i++) {
    const row = sorted[i];
    const previous = i > 0 ? sorted[i - 1] : undefined;
    if (previous === undefined || row === undefined || row.rate !== previous.rate) rank = i;
    if (row !== undefined) out.set(row.seed, rank / span);
  }
  return out;
}

export interface GhostEntry {
  seed: number;
  lane: number;
  deathStep: number | null;
  aliveRows: number;
}

interface GhostView {
  tape: Tape;
  windowSteps: number;
  poseFields: number[];
  entries: GhostEntry[];
}

export type StageFraming = 'track' | 'world';

export function stageFraming(category: EnvCategory): StageFraming {
  return category === 'cheetah' ? 'track' : 'world';
}

export const CHEETAH_CAM_DEPTH = 3.5;
export const CHEETAH_CAM_LOOK_X = 0.68;

export interface TrackWindow {
  back: number;
  ahead: number;
  width: number;
}

export function trackWindow(aspect: number): TrackWindow {
  const a = Number.isFinite(aspect) && aspect > 0 ? aspect : 16 / 9;
  const half = CHEETAH_CAM_DEPTH * Math.tan((TOKENS.fovHero * Math.PI) / 360) * a;
  return {
    back: Math.max(0, half - CHEETAH_CAM_LOOK_X),
    ahead: half + CHEETAH_CAM_LOOK_X,
    width: 2 * half,
  };
}

export function laneTrackX(
  tape: Tape,
  pose0Field: number,
  entry: Pick<GhostEntry, 'lane' | 'aliveRows'>,
  row: number,
): number | null {
  if (entry.aliveRows <= 0) return null;
  const bounded = row < 0 ? 0 : row;
  const at = bounded < entry.aliveRows ? bounded : entry.aliveRows - 1;
  const x = tapeValue(tape, pose0Field, at, entry.lane);
  return Number.isFinite(x) ? x : null;
}

export function inFrameCount(
  values: ArrayLike<number>,
  count: number,
  heroX: number,
  window: TrackWindow,
): number {
  if (!Number.isFinite(heroX)) return 0;
  let n = 0;
  for (let i = 0; i < count; i++) {
    const x = values[i];
    if (x === undefined || !Number.isFinite(x)) continue;
    const dx = x - heroX;
    if (dx >= -window.back && dx <= window.ahead) n++;
  }
  return n;
}

export interface FieldExtent {
  min: number;
  max: number;
  span: number;
}

export function fieldExtent(values: ArrayLike<number>, count: number): FieldExtent | null {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = values[i];
    if (x === undefined || !Number.isFinite(x)) continue;
    if (x < min) min = x;
    if (x > max) max = x;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  return { min, max, span: max - min };
}

export function fieldScale(
  extent: FieldExtent | null,
  heroX: number | null,
  window: TrackWindow,
): { min: number; max: number } {
  let min = extent ? extent.min : (heroX ?? 0);
  let max = extent ? extent.max : (heroX ?? 0);
  if (heroX !== null && Number.isFinite(heroX)) {
    min = Math.min(min, heroX - window.back);
    max = Math.max(max, heroX + window.ahead);
  }
  const span = max - min;
  const pad = span > 0 ? span * 0.04 : Math.max(1, window.width * 0.5);
  return { min: min - pad, max: max + pad };
}

export function fieldPercent(x: number, scale: { min: number; max: number }): number {
  const span = scale.max - scale.min;
  if (!Number.isFinite(x) || !(span > 0)) return 50;
  const t = ((x - scale.min) / span) * 100;
  return t < 0 ? 0 : t > 100 ? 100 : t;
}

export function formatDistance(metres: number): string {
  if (!Number.isFinite(metres)) return '—';
  const m = Math.abs(metres);
  if (m >= 1000) return `${(metres / 1000).toFixed(m >= 10_000 ? 0 : 1)} km`;
  if (m >= 10) return `${Math.round(metres)} m`;
  if (m >= 1) return `${metres.toFixed(1)} m`;
  return `${metres.toFixed(2)} m`;
}

const TIER_BY_LABEL = { low: 'thumb', medium: 'medium', high: 'hero' } as const;
export type TierLabel = keyof typeof TIER_BY_LABEL;
export const TIER_LABELS: readonly TierLabel[] = ['low', 'medium', 'high'];

function figure(label: string, value: string, title?: string): HTMLElement {
  return el('div', { class: 'pop-figure', ...(title ? { title } : {}) }, [
    el('span', { class: 'pop-label', text: label }),
    el('span', { class: 'pop-value', text: value }),
  ]);
}

function tint<T extends HTMLElement>(node: T, color: string): T {
  node.style.setProperty('--wall-accent', color);
  return node;
}

export const mount: ViewMount = (host) => {
  const query = parseWallParams(typeof window === 'undefined' ? '' : window.location.hash);

  const runName = el('button', { class: 'wall-run', type: 'button', text: '—', title: 'Switch run' });
  const runMeta = el('span', { class: 'wall-run-meta', text: '' });
  const runBadges = el('span', { class: 'wall-run-badges' });

  const pickerSearch = el('input', { class: 'input', type: 'search', placeholder: 'find a run…', ariaLabel: 'Find a run' });
  const pickerRows = el('div', { class: 'wall-picker-rows' });
  const picker = el('div', { class: 'wall-picker' }, [pickerSearch, pickerRows]);
  picker.hidden = true;

  function paintPicker(): void {
    const needle = pickerSearch.value.trim().toLowerCase();
    const all: RunIndexEntry[] = runIndex.snapshot() ?? [];
    const entries = sortRuns(all.filter((entry) => !needle || entry.id.toLowerCase().includes(needle)))
      .slice(0, 40);
    pickerRows.replaceChildren(
      ...entries.map((entry): HTMLElement => {
        const row = el('button', { class: 'wall-picker-row', type: 'button' }, [
          entry.inFlight ? el('span', { class: 'live-pip' }) : null,
          el('span', { class: 'wall-picker-id', text: entry.id }),
          entry.role === 'keeper' ? badge('keeper', 'keeper') : null,
          entry.hasRecordings ? badge('recordings', 'quiet') : null,
          entry.complete ? null : badge('incomplete', 'warn'),
          el('span', { class: 'wall-picker-meta', text: `${entry.envIds.length} envs · ${formatStep(entry.steps)}` }),
        ]);
        row.addEventListener('click', () => {
          picker.hidden = true;
          window.location.hash = `#/?run=${encodeURIComponent(entry.id)}`;
        });
        return row;
      }),
    );
    if (entries.length === 0) {
      pickerRows.append(el('div', { class: 'wall-picker-empty', text: 'no runs match' }));
    }
  }
  pickerSearch.addEventListener('input', paintPicker);
  runName.addEventListener('click', () => {
    picker.hidden = !picker.hidden;
    if (!picker.hidden) {
      void runIndex.load(false).then(() => { if (!picker.hidden) paintPicker(); });
      paintPicker();
      pickerSearch.focus();
    }
  });

  const head = el('div', { class: 'wall-head' }, [runName, runMeta, runBadges, picker]);

  const { root, body } = viewShell(head);
  root.classList.add('wall-root');
  host.append(root);

  let destroyed = false;
  const abort = new AbortController();

  let entry: RunIndexEntry | null = null;
  let detail: RunDetail | null = null;
  let tape: Tape | null = null;
  let envList: EnvId[] = [];
  let envId: EnvId | null = query.env ?? null;
  let sort: WallSort = query.sort ?? 'rate';
  let speed: number = query.speed ?? 64;
  let selection: HeroSelection = { leader: null, promoted: query.seed ?? null };
  let rows: SeedRow[] = [];
  let stride = 1;
  let frameAt = query.at ?? 0;
  let pendingSeekFrame: number | null = query.at ?? null;
  let scrubbing = false;
  let statsAccum = 1000;
  let debugOn = false;
  const TIER_KEY = 'oakWallTier';
  let tierLabel: TierLabel = (() => {
    const saved = typeof localStorage === 'undefined' ? null : localStorage.getItem(TIER_KEY);
    return saved !== null && saved in TIER_BY_LABEL ? (saved as TierLabel) : 'medium';
  })();

  let sceneHost: SceneHostHandle | null = null;
  let stageQueue: Promise<void> = Promise.resolve();
  let hero: HeroHandle | null = null;
  const heroCache = new Map<string, HeroHandle>();
  const HERO_CACHE_MAX = 6;
  let heroGeneration = 0;
  let heroAttached = false;
  let ghostView: GhostView | null = null;
  let ghostLayer: GhostLayer | null = null;
  let ghostKey = '';
  let ghostSample = -1;
  let emphasised: number | null = null;
  let heroSource: PlaybackSource | null = null;
  let fieldData: { tape: Tape; pose0: number } | null = null;
  let fieldLanes: GhostEntry[] = [];
  let fieldX = new Float64Array(0);
  let fieldMarks: HTMLElement[] = [];
  let framing: StageFraming = 'world';
  let stageWindow: TrackWindow = trackWindow(16 / 9);
  let ghostsInFrame = 0;
  let fieldSample = -1;
  const ghostPose = new Float32Array(9);
  const sparks: SparklineHandle[] = [];
  let livePoll: ReturnType<typeof setInterval> | null = null;
  let detailPollAt = 0;

  const chipStrip = el('div', { class: 'wall-envs', role: 'tablist', ariaLabel: 'Environment' });
  const envSelect = el('select', { class: 'input input-select wall-select', ariaLabel: 'Environment' });
  envSelect.addEventListener('change', () => {
    const next = envSelect.value;
    if (isEnvId(next)) selectEnv(next);
  });

  const stage = el('div', { class: 'wall-stage' });
  const stageTag = el('div', { class: 'wall-tag' }, [
    el('b', { class: 'wall-tag-env', text: '' }),
    el('span', { class: 'wall-tag-sub', text: '' }),
  ]);
  const stageNote = el('div', { class: 'wall-cover' });
  stageNote.hidden = true;
  stage.append(stageTag, stageNote);

  const fieldNote = el('span', { class: 'wall-field-note', text: '' });
  const fieldMarkHolder = el('div', { class: 'wall-field-marks' });
  const fieldWindow = el('div', {
    class: 'wall-field-window',
    title: 'The slice of track the stage can show, drawn at the field’s own scale',
  });
  const fieldTrack = el('div', { class: 'wall-field-track' }, [
    el('div', { class: 'wall-field-axis' }),
    fieldWindow,
    fieldMarkHolder,
  ]);
  const fieldLo = el('span', { class: 'wall-field-end', text: '' });
  const fieldHi = el('span', { class: 'wall-field-end', text: '' });
  const field = el('div', { class: 'wall-field' }, [
    el('div', { class: 'wall-field-head' }, [
      el('span', { class: 'wall-field-label', text: 'the field' }),
      fieldNote,
    ]),
    fieldTrack,
    el('div', { class: 'wall-field-scale' }, [fieldLo, fieldHi]),
  ]);
  field.hidden = true;

  const playButton = el('button', { class: 'tbtn tbtn-play', type: 'button', text: 'pause' });
  const backButton = el('button', { class: 'tbtn', type: 'button', text: '‹', title: 'Back (←)' });
  const fwdButton = el('button', { class: 'tbtn', type: 'button', text: '›', title: 'Forward (→)' });
  const scrubInput = el('input', { class: 'scrub-input', type: 'range' });
  scrubInput.min = '0';
  scrubInput.max = '1';
  scrubInput.step = 'any';
  scrubInput.value = '0';
  scrubInput.setAttribute('aria-label', 'Scrub the run');
  const scrubFill = el('div', { class: 'wall-scrub-fill' });
  const scrubLoaded = el('div', { class: 'wall-scrub-loaded' });
  const scrubHead = el('div', { class: 'wall-scrub-head' });
  const scrubTicks = el('div', { class: 'wall-scrub-ticks' });
  const scrub = el('div', { class: 'wall-scrub' }, [
    scrubInput,
    el('div', { class: 'wall-scrub-track' }),
    scrubLoaded,
    scrubFill,
    scrubTicks,
    scrubHead,
  ]);
  const readout = el('div', { class: 'wall-readout', text: 'step — / —' });
  const speedRow = el('div', { class: 'wall-speeds' });
  const ghostChip = el('span', { class: 'wall-ghosts', text: 'ghosts —' });
  const debugButton = el('button', { class: 'tbtn', type: 'button', text: 'g', title: 'Ground-truth overlay (g)' });
  const tierRow = el('div', { class: 'wall-speeds' });
  const tierButtons = new Map<TierLabel, HTMLButtonElement>();
  for (const label of TIER_LABELS) {
    const button = el('button', {
      class: 'tbtn',
      type: 'button',
      text: label,
      title: `Graphics quality: ${label} (t cycles)`,
    });
    tierButtons.set(label, button);
    tierRow.append(button);
  }
  function paintTier(): void {
    for (const [label, button] of tierButtons) button.classList.toggle('is-on', label === tierLabel);
  }
  paintTier();
  function setTier(next: TierLabel): void {
    tierLabel = next;
    paintTier();
    try { localStorage.setItem(TIER_KEY, next); } catch {  }
    void sceneHost?.setTier(TIER_BY_LABEL[next]);
  }
  for (const [label, button] of tierButtons) {
    button.addEventListener('click', () => setTier(label));
  }
  const fullButton = el('button', { class: 'tbtn', type: 'button', text: '⛶', title: 'Fullscreen (f)' });
  const transport = el('div', { class: 'wall-transport' }, [
    playButton,
    backButton,
    fwdButton,
    scrub,
    readout,
    el('div', { class: 'tsep' }),
    speedRow,
    el('div', { class: 'tsep' }),
    ghostChip,
    el('div', { class: 'tsep' }),
    tierRow,
    debugButton,
    fullButton,
  ]);

  const popLine = el('div', { class: 'wall-pop' });
  const popNote = el('div', { class: 'wall-pop-note', text: '' });
  const sortButton = el('button', { class: 'chip wall-sort', type: 'button', text: 'sort reward rate' });
  const railRows = el('div', { class: 'wall-rows' });
  const railCount = el('span', { class: 'eyebrow-count', text: '' });
  const rail = el('aside', { class: 'wall-rail' }, [
    el('div', { class: 'wall-rail-head' }, [popLine, popNote]),
    el('div', { class: 'wall-rail-controls' }, [eyebrow('Seeds', railCount), sortButton]),
    railRows,
  ]);

  const status = el('div', { class: 'view-status' }, [spinner('Reading .oak-benchmarks')]);
  body.append(chipStrip, envSelect, status);

  const stageCol = el('div', { class: 'wall-stage-col' }, [stage, field, transport]);
  const splitter = el('div', {
    class: 'wall-split',
    role: 'separator',
    ariaLabel: 'Resize the seed panel — drag; double-click to collapse',
    title: 'Drag to resize · double-click to collapse',
  });
  const main = el('div', { class: 'wall-main' }, [stageCol, splitter, rail]);

  const RAIL_KEY = 'oakWallRailW';
  const RAIL_MIN = 250;
  let railW = Number(typeof localStorage === 'undefined' ? NaN : localStorage.getItem(RAIL_KEY)) || 620;
  let railCollapsed = false;
  function applyRail(): void {
    main.style.setProperty('--rail-w', `${railW}px`);
    main.classList.toggle('rail-hidden', railCollapsed);
  }
  applyRail();
  {
    let dragging = false;
    splitter.addEventListener('pointerdown', (ev) => {
      dragging = true;
      try { splitter.setPointerCapture(ev.pointerId); } catch {  }
    });
    splitter.addEventListener('pointermove', (ev) => {
      if (!dragging) return;
      const right = main.getBoundingClientRect().right;
      const width = right - ev.clientX;
      if (width < RAIL_MIN * 0.6) {
        railCollapsed = true;
      } else {
        railCollapsed = false;
        railW = Math.min(Math.max(width, RAIL_MIN), Math.round(main.getBoundingClientRect().width * 0.6));
      }
      applyRail();
    });
    const settle = (): void => {
      if (!dragging) return;
      dragging = false;
      try { localStorage.setItem(RAIL_KEY, String(railW)); } catch {  }
    };
    splitter.addEventListener('pointerup', settle);
    splitter.addEventListener('pointercancel', settle);
    splitter.addEventListener('dblclick', () => {
      railCollapsed = !railCollapsed;
      applyRail();
    });
  }
  main.hidden = true;
  body.append(main);

  const vsplit = el('div', {
    class: 'wall-vsplit',
    role: 'separator',
    ariaLabel: 'Resize the chart row — drag',
    title: 'Drag to resize the charts',
  });
  const scienceHost = el('div', { class: 'wall-science' });
  body.append(vsplit, scienceHost);

  const SCI_KEY = 'oakWallScienceH';
  let scienceH = Number(typeof localStorage === 'undefined' ? NaN : localStorage.getItem(SCI_KEY)) || 210;
  function applyScienceH(): void {
    scienceHost.style.setProperty('--science-h', `${scienceH}px`);
  }
  applyScienceH();
  {
    let dragging = false;
    vsplit.addEventListener('pointerdown', (ev) => {
      dragging = true;
      try { vsplit.setPointerCapture(ev.pointerId); } catch {  }
    });
    vsplit.addEventListener('pointermove', (ev) => {
      if (!dragging) return;
      const bottom = scienceHost.getBoundingClientRect().bottom;
      scienceH = Math.min(Math.max(Math.round(bottom - ev.clientY - 14), 140), 460);
      applyScienceH();
    });
    const settle = (): void => {
      if (!dragging) return;
      dragging = false;
      try { localStorage.setItem(SCI_KEY, String(scienceH)); } catch {  }
    };
    vsplit.addEventListener('pointerup', settle);
    vsplit.addEventListener('pointercancel', settle);
  }
  let science: RunScienceHandle | null = null;

  function syncUrl(): void {
    if (typeof history === 'undefined' || typeof window === 'undefined') return;
    const params: WallParams = {};
    if (detail) params.run = detail.id;
    if (envId !== null) params.env = envId;
    if (selection.promoted !== null) params.seed = selection.promoted;
    const at = urlPosition(sceneHost?.clock.position() ?? null, frameAt);
    if (at !== null) params.at = at;
    if (speed !== 1) params.speed = speed;
    if (sort !== 'rate') params.sort = sort;
    history.replaceState(null, '', wallHref(params));
  }

  function syncPlay(): void {
    playButton.textContent = sceneHost?.clock.playing ? 'pause' : 'play';
  }

  function applySpeed(): void {
    sceneHost?.setSpeed(speed);
    for (const button of speedRow.querySelectorAll('button')) {
      button.classList.toggle('is-on', button.dataset['speed'] === String(speed));
    }
  }

  for (const preset of WALL_SPEEDS) {
    const button = el('button', {
      class: 'tbtn',
      type: 'button',
      text: `×${preset}`,
      dataset: { speed: String(preset) },
      title: `×${preset} real time (${preset} × the environment's own clock)`,
    });
    button.addEventListener('click', () => {
      speed = preset;
      applySpeed();
      syncUrl();
    });
    speedRow.append(button);
  }

  playButton.addEventListener('click', () => {
    sceneHost?.toggle();
    syncPlay();
  });
  backButton.addEventListener('click', () => stepFrames(-Math.max(1, stride)));
  fwdButton.addEventListener('click', () => stepFrames(Math.max(1, stride)));
  debugButton.addEventListener('click', () => setDebug(!debugOn));
  function toggleFullscreen(): void {
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      stageCol.requestFullscreen().catch(() => {});
    }
  }
  fullButton.addEventListener('click', toggleFullscreen);
  const onFullscreenChange = (): void => {
    const on = document.fullscreenElement !== null;
    fullButton.classList.toggle('is-on', on);
    fullButton.title = on ? 'Exit fullscreen (f or Esc)' : 'Fullscreen (f)';
  };
  if (typeof document !== 'undefined') document.addEventListener('fullscreenchange', onFullscreenChange);

  scrubInput.addEventListener('pointerdown', () => {
    scrubbing = true;
  });
  const endScrub = (): void => {
    if (!scrubbing) return;
    scrubbing = false;
    syncUrl();
  };
  scrubInput.addEventListener('pointerup', endScrub);
  scrubInput.addEventListener('pointercancel', endScrub);
  scrubInput.addEventListener('blur', endScrub);
  scrubInput.addEventListener('input', () => {
    seekFrame(Number(scrubInput.value));
  });

  function frameCount(): number {
    return heroSource?.frameCount ?? 1;
  }
  function heroSteps(): number {
    return hero?.totalSteps ?? Math.max(0, frameCount() - 1);
  }
  function frameFromStep(step: number): number {
    const steps = heroSteps();
    const scale = steps > 0 ? (frameCount() - 1) / steps : 1;
    return clampPosition(step * scale, frameCount());
  }
  function stepFromFrame(frame: number): number {
    const frames = frameCount() - 1;
    return frames > 0 ? (frame * heroSteps()) / frames : 0;
  }
  function seekFrame(frame: number): void {
    if (!sceneHost) return;
    const loaded = frameCount() - 1;
    if (frame > loaded) {
      pendingSeekFrame = Math.floor(frame);
      sceneHost.seek(loaded);
      return;
    }
    pendingSeekFrame = null;
    sceneHost.seek(clampPosition(frame, frameCount()));
  }
  function scrubToStep(step: number, commit: boolean): void {
    seekFrame(frameFromStep(step));
    if (commit) syncUrl();
  }
  function seekToStep(step: number): void {
    scrubToStep(step, true);
  }

  function stepFrames(frames: number): void {
    if (!sceneHost) return;
    sceneHost.pause();
    seekFrame(stepPosition(sceneHost.clock.position(), frames, frameCount()));
    syncPlay();
    syncUrl();
  }

  function setDebug(on: boolean): void {
    debugOn = on;
    sceneHost?.setDebug(on);
    debugButton.classList.toggle('is-on', on);
  }

  sortButton.addEventListener('click', () => {
    sort = nextSort(sort);
    paintRail();
    syncUrl();
  });

  const onKey = (event: KeyboardEvent): void => {
    const node = event.target;
    if (node instanceof HTMLInputElement || node instanceof HTMLSelectElement || node instanceof HTMLTextAreaElement) {
      return;
    }
    const action = wallKeyAction(event.key, event.shiftKey);
    if (!action) return;
    event.preventDefault();
    switch (action.kind) {
      case 'toggle':
        sceneHost?.toggle();
        syncPlay();
        break;
      case 'step':
        stepFrames(action.frames * Math.max(1, stride));
        break;
      case 'env': {
        const next = cycleEnv(envList, envId, action.direction);
        if (next !== null && next !== envId) selectEnv(next);
        break;
      }
      case 'sort':
        sort = nextSort(sort);
        paintRail();
        syncUrl();
        break;
      case 'debug':
        setDebug(!debugOn);
        break;
      case 'fullscreen':
        toggleFullscreen();
        break;
      case 'tier': {
        const index = TIER_LABELS.indexOf(tierLabel);
        setTier(TIER_LABELS[(index + 1) % TIER_LABELS.length] ?? 'medium');
        break;
      }
      case 'escape': {
        const next = escapeSelection(selection);
        if (next !== selection) {
          selection = next;
          applyHero();
          mountHero();
          syncUrl();
        }
        break;
      }
    }
  };
  if (typeof window !== 'undefined') window.addEventListener('keydown', onKey);

  function paintChips(): void {
    const recordings = detail?.recordings ?? [];
    chipStrip.replaceChildren(
      ...envList.map((id) => {
        const info = ENVS[id];
        const count = seedsFor(recordings, id).length;
        const chip = tint(
          el('button', {
            class: `wall-chip${id === envId ? ' is-on' : ''}${count === 0 ? ' is-empty' : ''}`,
            type: 'button',
            role: 'tab',
            ariaCurrent: id === envId ? 'true' : '',
            title: count > 0
              ? `${info.name} — ${formatCount(count)} recorded seed${count === 1 ? '' : 's'}`
              : `${info.name} — no recordings in this run`,
          }, [
            el('span', { class: 'wall-chip-name', text: info.name }),
          ]),
          accentCss(info.slug),
        );
        chip.addEventListener('click', () => selectEnv(id));
        return chip;
      }),
    );
    envSelect.replaceChildren(
      ...envList.map((id) => el('option', { value: id, text: ENVS[id].name })),
    );
    if (envId !== null) envSelect.value = envId;
  }

  function paintPopulation(): void {
    const slice = envId === null ? null : detail?.summary?.envSlices[envId] ?? null;
    const figures = populationFigures(slice, rows);
    popLine.replaceChildren(
      figure('seeds > 0', figures.positive, 'Seeds whose average reward rate is above zero'),
      figure('non-diverged', figures.nonDiverged, 'Seeds whose stream never went non-finite'),
      figure('retention', figures.retention, 'Seeds still holding at least half their peak reward rate'),
      figure('solved', figures.solved, 'Seeds that crossed this environment’s solved threshold (fleet.py report)'),
      figure('reward / GFLOP', figures.perGflop, 'Best seed’s reward rate per agent GFLOP — the SOTA-target metric'),
    );
    popNote.textContent = figures.source === 'summary'
      ? 'from summary.json — fleet.py report'
      : figures.source === 'measured'
        ? 'measured live from this run’s tape'
        : '';
  }

  function paintRail(): void {
    science?.setHighlight(heroSeed(selection));
    for (const spark of sparks) spark.destroy();
    sparks.length = 0;
    sortButton.textContent = `sort ${SORT_LABEL[sort]}`;
    railCount.textContent = rows.length > 0 ? `${formatCount(rows.length)}` : '';

    if (rows.length === 0) {
      railRows.replaceChildren(
        el('div', { class: 'wall-row-empty', text: 'No seeds for this environment.' }),
      );
      return;
    }

    const order = sortPermutation(rows, sort);
    const hero_ = heroSeed(selection);
    const withSparks = rows.length <= SPARK_ROW_BUDGET;
    const nodes: HTMLElement[] = [];

    order.forEach((index, rank) => {
      const row = rows[index];
      if (row === undefined) return;
      const isHero = row.seed === hero_;
      const node = el('button', {
        class: `wall-row${isHero ? ' is-hero' : ''}${row.recorded ? '' : ' is-quiet'}${row.divergedAt !== null ? ' is-dead' : ''}`,
        type: 'button',
        dataset: { seed: String(row.seed) },
        title: row.recorded
          ? `Promote seed ${row.seed} to the hero scene`
          : `Seed ${row.seed} has no recording in this run — it cannot be replayed`,
      }, [
        el('span', { class: 'wall-rank', text: String(rank + 1) }),
        el('span', { class: 'wall-seed', text: String(row.seed) }),
        el('span', {
          class: `wall-rate ${row.rate === null ? 'tone-dim' : row.rate > 0 ? 'tone-pos' : 'tone-neg'}`,
          text: row.rate === null ? '—' : formatSigned(row.rate, 3),
        }),
        row.divergedAt !== null
          ? el('span', { class: 'wall-dead', text: 'div', title: `numerically diverged at step ${formatCount(row.divergedAt)}` })
          : null,
      ]);

      let sparkDragged = false;
      if (withSparks) {
        const holder = el('span', { class: 'wall-spark' });
        const spark = sparkline({
          values: [0, ...row.spark, row.spark[row.spark.length - 1] ?? null],
          xs: [0, ...row.sparkXs, 1],
          height: 16,
          stroke: row.divergedAt !== null ? '#e0614f' : isHero ? '#ffd166' : '#2b98d9',
          endDot: false,
          ariaLabel: `Reward-rate curve for seed ${row.seed}`,
        });
        holder.append(spark.root);
        sparks.push(spark);
        node.append(holder);

        let down = false;
        let startX = 0;
        const fracStep = (ev: PointerEvent): number => {
          const rect = holder.getBoundingClientRect();
          const frac = Math.min(Math.max((ev.clientX - rect.left) / Math.max(1, rect.width), 0), 1);
          return Math.round(frac * heroSteps());
        };
        holder.addEventListener('pointerdown', (ev) => {
          if (ev.button !== 0) return;
          down = true;
          startX = ev.clientX;
          try { holder.setPointerCapture(ev.pointerId); } catch {  }
        });
        holder.addEventListener('pointermove', (ev) => {
          if (!down) return;
          if (Math.abs(ev.clientX - startX) > 3) sparkDragged = true;
          if (sparkDragged) scrubToStep(fracStep(ev), false);
        });
        holder.addEventListener('pointerup', (ev) => {
          if (!down) return;
          down = false;
          if (sparkDragged) scrubToStep(fracStep(ev), true);
        });
        holder.addEventListener('pointercancel', () => { down = false; });
      }

      node.addEventListener('mouseenter', () => emphasise(row.seed));
      node.addEventListener('mouseleave', () => emphasise(null));
      node.addEventListener('focus', () => emphasise(row.seed));
      node.addEventListener('blur', () => emphasise(null));
      node.addEventListener('click', () => {
        if (sparkDragged) { sparkDragged = false; return; }
        watchSeed(row.seed);
      });
      nodes.push(node);
    });

    railRows.replaceChildren(...nodes);
    if (!withSparks) {
      railRows.append(
        el('div', {
          class: 'wall-row-note',
          text: `${formatCount(rows.length)} seeds — per-seed curves are drawn up to ${SPARK_ROW_BUDGET}.`,
        }),
      );
    }
  }

  function ghostEmphasis(seed: number | null): void {
    const layer = sceneHost?.ghosts ?? null;
    if (!layer || !ghostView) return;
    if (emphasised === seed) return;
    emphasised = seed;
    ghostView.entries.forEach((entry, index) => {
      layer.setEmphasis(index, entry.seed === seed ? 1 : 0);
    });
  }

  function emphasise(seed: number | null): void {
    ghostEmphasis(seed);
    fieldEmphasis(seed);
  }

  function fieldEmphasis(seed: number | null): void {
    for (const mark of fieldMarks) {
      mark.classList.toggle('is-lit', seed !== null && Number(mark.dataset['seed']) === seed);
    }
  }

  function fieldSamples(): number {
    return fieldData?.tape.rows ?? 0;
  }

  function buildField(entries: readonly GhostEntry[], hero_: number | null): void {
    fieldLanes = [...entries].sort((a, b) => a.seed - b.seed);
    fieldX = new Float64Array(fieldLanes.length);
    fieldMarks = fieldLanes.map((entry, index) => {
      const isHero = entry.seed === hero_;
      const mark = el('button', {
        class: `wall-field-mark${isHero ? ' is-hero' : ''}${entry.deathStep !== null ? ' is-dead' : ''}`,
        type: 'button',
        dataset: { seed: String(entry.seed) },
        ariaLabel: `Seed ${entry.seed}`,
      });
      mark.addEventListener('mouseenter', () => {
        const x = fieldX[index];
        mark.title = `Seed ${entry.seed}${isHero ? ' — the hero' : ''}`
          + (x !== undefined && Number.isFinite(x) ? ` · ${formatDistance(x)} down the track` : '');
        emphasise(entry.seed);
      });
      mark.addEventListener('mouseleave', () => emphasise(null));
      mark.addEventListener('click', () => {
        const next = promoteSeed(selection, entry.seed, playable);
        if (next === selection) return;
        selection = next;
        applyHero();
        mountHero();
        syncUrl();
      });
      return mark;
    });
    fieldMarkHolder.replaceChildren(...fieldMarks);
    field.hidden = framing !== 'track' || fieldLanes.length < 2;
    fieldSample = -1;
  }

  function paintField(sample: number, heroX: number | null): void {
    if (field.hidden || fieldLanes.length === 0) return;
    const aspect = stage.clientHeight > 0 ? stage.clientWidth / stage.clientHeight : 16 / 9;
    stageWindow = trackWindow(aspect);

    let heroIndex = -1;
    const hero_ = heroSeed(selection);
    for (let i = 0; i < fieldLanes.length; i++) {
      const entry = fieldLanes[i];
      if (entry === undefined) continue;
      if (entry.seed === hero_) heroIndex = i;
      const x = fieldData === null ? null : laneTrackX(fieldData.tape, fieldData.pose0, entry, sample);
      fieldX[i] = x === null ? Number.NaN : x;
    }
    if (heroIndex >= 0 && heroX !== null) fieldX[heroIndex] = heroX;

    const extent = fieldExtent(fieldX, fieldLanes.length);
    const centre = heroX ?? (heroIndex >= 0 ? fieldX[heroIndex] ?? null : null);
    const scale = fieldScale(extent, centre, stageWindow);
    for (let i = 0; i < fieldMarks.length; i++) {
      const mark = fieldMarks[i];
      const x = fieldX[i];
      if (mark === undefined) continue;
      if (x === undefined || !Number.isFinite(x)) {
        mark.hidden = true;
        continue;
      }
      mark.hidden = false;
      mark.style.left = `${fieldPercent(x, scale).toFixed(3)}%`;
    }

    if (centre !== null && Number.isFinite(centre)) {
      const left = fieldPercent(centre - stageWindow.back, scale);
      const right = fieldPercent(centre + stageWindow.ahead, scale);
      fieldWindow.hidden = false;
      fieldWindow.style.left = `${left.toFixed(3)}%`;
      fieldWindow.style.width = `${Math.max(0.6, right - left).toFixed(3)}%`;
    } else {
      fieldWindow.hidden = true;
    }

    const inFrame = centre === null ? 0 : inFrameCount(fieldX, fieldLanes.length, centre, stageWindow);
    const heroCounted = heroIndex >= 0 && Number.isFinite(fieldX[heroIndex] ?? Number.NaN) ? 1 : 0;
    ghostsInFrame = Math.max(0, inFrame - heroCounted);
    fieldLo.textContent = extent === null ? '' : formatDistance(extent.min);
    fieldHi.textContent = extent === null ? '' : formatDistance(extent.max);
    fieldNote.textContent = extent === null
      ? `${formatCount(fieldLanes.length)} seeds`
      : `${formatCount(fieldLanes.length)} seeds · ${formatDistance(extent.span)} apart · `
        + `the stage frames ${formatDistance(stageWindow.width)} of it`;
  }

  function heroTrackX(index: number): number | null {
    if (heroSource === null || heroSource.frameCount === 0) return null;
    const frame = heroSource.frameAt(Math.min(index, heroSource.frameCount - 1));
    const x = frame.pose[0];
    return x === undefined || !Number.isFinite(x) ? null : x;
  }

  function paintGhosts(layer: GhostLayer, view: GhostView, row: number): void {
    const stepEnd = view.tape.stepEnd[row] ?? 0;
    for (let index = 0; index < view.entries.length; index++) {
      const entry = view.entries[index];
      if (entry === undefined || entry.aliveRows <= 0) continue;
      const at = row < entry.aliveRows ? row : entry.aliveRows - 1;
      for (let k = 0; k < 9; k++) {
        ghostPose[k] = tapeValue(view.tape, view.poseFields[k] ?? 0, at, entry.lane);
      }
      if (!Number.isFinite(ghostPose[0] ?? Number.NaN)) continue;
      layer.setGhostFrame(index, ghostPose, ghostAlive(entry.deathStep, stepEnd));
    }
  }

  function playable(seed: number): boolean {
    if (!detail || envId === null) return false;
    return recordingRef(detail.recordings, envId, seed) !== null;
  }

  function paintTicks(ticks: readonly TimelineTick[]): void {
    let busiest = 1;
    for (const tick of ticks) if (tick.count > busiest) busiest = tick.count;
    scrubTicks.replaceChildren(
      ...ticks.map((tick) => {
        const density = Math.sqrt(tick.count / busiest);
        const node = el('button', {
          class: `wtick wtick-${tick.kind}`,
          type: 'button',
          title: tick.label,
          ariaLabel: `Seek to ${tick.label}`,
          style: {
            left: `${(tick.position * 100).toFixed(3)}%`,
            opacity: (0.22 + 0.62 * density).toFixed(3),
            height: `${(6 + 8 * density).toFixed(1)}px`,
            marginTop: `${(-(6 + 8 * density) / 2).toFixed(1)}px`,
          },
        });
        node.addEventListener('click', () => {
          if (!sceneHost) return;
          seekFrame(seekTargetForTick(tick));
          sceneHost.play();
          syncPlay();
        });
        return node;
      }),
    );
  }

  let lastMarkerAt = 0;

  function onFrame(info: HostFrameInfo): void {
    const frames = Math.max(1, frameCount() - 1);
    const progress = info.position / frames;
    frameAt = info.position;
    const nowMs = performance.now();
    if (nowMs - lastMarkerAt > 100) {
      lastMarkerAt = nowMs;
      const step = Math.max(0, stepFromFrame(info.position));
      science?.setPlayhead(step);
      railRows.style.setProperty('--play-x', `${(progress * 100).toFixed(2)}%`);
    }
    const percent = (progress * 100).toFixed(3);
    scrubFill.style.width = `${percent}%`;
    scrubHead.style.left = `${percent}%`;
    if (!scrubbing) scrubInput.value = String(info.position);

    const layer = sceneHost?.ghosts ?? null;
    if (layer && ghostView) {
      const sample = ghostRowIndex(info.position, ghostView.windowSteps, ghostView.tape.rows);
      if (sample !== ghostSample || layer !== ghostLayer) {
        ghostSample = sample;
        if (layer !== ghostLayer) {
          ghostLayer = layer;
          emphasised = null;
          primeGhosts(layer, ghostView);
        }
        paintGhosts(layer, ghostView, sample);
      }
    }
    if (!field.hidden && fieldLanes.length > 0) {
      const sample = ghostRowIndex(info.position, stride, fieldSamples());
      if (sample !== fieldSample) {
        fieldSample = sample;
        paintField(sample, heroTrackX(info.index));
      }
    }

    statsAccum += info.frameMs;
    if (statsAccum < 150) return;
    statsAccum = 0;
    readout.textContent =
      `step ${info.step < 0 ? 'init' : formatCount(info.step)} / ${formatStep(heroSteps())}`
      + ` · ${formatClock(info.simTime)}`;
    if (ghostView && info.ghostDraws > 0) ghostChip.textContent = ghostChipText(ghostView);
    syncPlay();
  }

  function ghostChipText(view: GhostView): string {
    const total = view.entries.length;
    if (framing !== 'track') return `ghosts ${formatCount(total)} · all in frame`;
    const inFrame = ghostsInFrame >= total ? 'all' : formatCount(ghostsInFrame);
    return `ghosts ${formatCount(total)} · ${inFrame} in frame`;
  }

  function primeGhosts(layer: GhostLayer, view: GhostView): void {
    const quantiles = rateQuantiles(rows);
    view.entries.forEach((entry, index) => {
      layer.setRate(index, quantiles.get(entry.seed) ?? 0);
      layer.setEmphasis(index, 0);
    });
  }

  function showStageNote(title: string, children: (HTMLElement | null)[] = []): void {
    stageNote.hidden = false;
    stageNote.replaceChildren(el('div', { class: 'cover-title', text: title }), ...children.filter((c): c is HTMLElement => c !== null));
  }

  function clearStageNote(): void {
    stageNote.hidden = true;
    stageNote.replaceChildren();
  }

  function queueStage(work: () => Promise<void>): void {
    stageQueue = stageQueue.then(work).catch((error: unknown) => {
      if (destroyed) return;
      showStageNote('Could not play this seed', [
        errorState(error instanceof Error ? error.message : String(error)),
      ]);
    });
  }

  function mountHero(): void {
    const generation = ++heroGeneration;
    const run = detail;
    const env = envId;
    if (hero !== null) {
      hero.setCallbacks(null, null);
      hero.setLive(false);
    }
    hero = null;
    heroAttached = false;
    heroSource = null;
    ghostView = null;
    ghostLayer = null;
    ghostKey = '';
    ghostSample = -1;
    scrubTicks.replaceChildren();
    scrubLoaded.style.width = '0%';
    if (!run || env === null) return;

    const seed = heroSeed(selection);
    const reason = degradedReason(run.recordings, env);
    if (reason !== null || seed === null) {
      showStageNote('Nothing to replay here', [
        emptyState(reason ?? 'no seed to play', RECORD_RECIPE, [
          linkButton('Find a run with recordings', '#/runs', { quiet: true }),
        ]),
      ]);
      ghostChip.textContent = 'ghosts —';
      readout.textContent = 'no recording';
      transport.classList.add('is-idle');
      return;
    }
    clearStageNote();
    transport.classList.remove('is-idle');

    const info = ENVS[env];
    const tagEnv = stageTag.querySelector('.wall-tag-env');
    const tagSub = stageTag.querySelector('.wall-tag-sub');
    if (tagEnv) tagEnv.textContent = info.name;
    if (tagSub) {
      tagSub.textContent = `seed ${seed}${selection.promoted === null ? ' · leader' : ' · promoted'}`;
    }
    framing = stageFraming(info.category);

    const ref = recordingRef(run.recordings, env, seed);
    const refSteps = ref?.steps ?? 0;
    if (tape !== null && tape.rows > 0) stride = tapeWindowSteps(tape);
    ghostChip.textContent = 'ghosts …';

    const startAt = pendingSeekFrame ?? frameAt;
    pendingSeekFrame = Number.isFinite(startAt) && startAt > 0 ? Math.floor(startAt) : null;

    const key = `${run.id}|${env}|${seed}`;
    let h = heroCache.get(key);
    if (h === undefined) {
      h = loadHero({
        runId: run.id,
        envId: env,
        seed,
        steps: refSteps,
        horizonSteps: run.meta.steps,
        live: entry?.inFlight === true,
      });
      heroCache.set(key, h);
      for (const [oldKey, old] of heroCache) {
        if (heroCache.size <= HERO_CACHE_MAX) break;
        if (old === h) continue;
        old.destroy();
        heroCache.delete(oldKey);
      }
    } else {
      heroCache.delete(key);
      heroCache.set(key, h);
      h.setLive(entry?.inFlight === true);
    }
    hero = h;
    const handle = h;

    handle.setCallbacks(
      () => {
        if (destroyed || generation !== heroGeneration || hero !== handle) return;
        onHeroProgress();
      },
      (error) => {
        if (destroyed || generation !== heroGeneration) return;
        showStageNote('Could not play this seed', [errorState(error.message)]);
      },
    );

    const attachSource = handle.source.frameCount > 1 ? handle.source : makeInitialSource(env, seed);
    if (attachSource === handle.source) {
      heroAttached = true;
      heroSource = handle.source;
    } else {
      heroSource = attachSource;
    }
    const attachAt = pendingSeekFrame !== null
      ? Math.min(pendingSeekFrame, attachSource.frameCount - 1)
      : 0;
    if (pendingSeekFrame !== null && pendingSeekFrame < attachSource.frameCount) pendingSeekFrame = null;

    queueStage(async () => {
      if (destroyed || generation !== heroGeneration) return;
      if (sceneHost === null) {
        sceneHost = await mountSceneHost(stage, {
          envId: env,
          source: attachSource,
          tier: TIER_BY_LABEL[tierLabel],
          at: attachAt,
          speed,
          debug: debugOn,
          onFrame,
          onReady: (manifest) => {
            if (tagEnv) tagEnv.textContent = manifest.title;
          },
        });
        if (destroyed) {
          sceneHost.destroy();
          sceneHost = null;
          return;
        }
        stage.insertBefore(sceneHost.canvas, stageTag);
      } else if (sceneHost.envId !== env) {
        await sceneHost.setEnv(env, attachSource, null, { at: attachAt });
      } else {
        await sceneHost.setSource(attachSource, { at: attachAt });
        sceneHost.setGhosts(null);
      }
      applySpeed();
      syncPlay();
      if (heroAttached) {
        scrubInput.max = String(Math.max(1, handle.source.frameCount - 1));
        paintTicks(handle.ticks());
        const total = Math.max(1, handle.totalSteps);
        scrubLoaded.style.width = `${Math.min(100, (handle.loadedSteps / total) * 100).toFixed(2)}%`;
      }
    });

    function onHeroProgress(): void {
      queueStage(async () => {
        if (destroyed || generation !== heroGeneration || hero !== handle || sceneHost === null) return;
        if (!heroAttached) {
          heroAttached = true;
          heroSource = handle.source;
          await sceneHost.setSource(handle.source, {
            at: pendingSeekFrame !== null ? Math.min(pendingSeekFrame, handle.source.frameCount - 1) : 0,
          });
          if (pendingSeekFrame !== null && pendingSeekFrame < handle.source.frameCount) {
            pendingSeekFrame = null;
          }
        } else if (pendingSeekFrame !== null && pendingSeekFrame < handle.source.frameCount) {
          sceneHost.seek(pendingSeekFrame);
          pendingSeekFrame = null;
        }
        scrubInput.max = String(Math.max(1, handle.source.frameCount - 1));
        const total = Math.max(1, handle.totalSteps);
        scrubLoaded.style.width = `${Math.min(100, (handle.loadedSteps / total) * 100).toFixed(2)}%`;
        paintTicks(handle.ticks());
        statsAccum = 1000;
      });
    }

    applyGhosts();
  }

  function applyGhosts(): void {
    const run = detail;
    const env = envId;
    const t = tape;
    if (!run || env === null || t === null || t.rows === 0) return;
    const seed = heroSeed(selection);
    const source = heroSource;
    if (seed === null || source === null) return;
    const generation = heroGeneration;
    const info = ENVS[env];
    stride = tapeWindowSteps(t);

    const poseFields = POSE_FIELD_NAMES.map((name) => fieldIndex(t, name));
    const { lanes, seeds } = envLanes(run.meta, env);
    const laneOfSeed = new Map<number, number>();
    seeds.forEach((s, i) => laneOfSeed.set(s, lanes[i] ?? 0));
    const all: GhostEntry[] = [];
    for (const s of seedsFor(run.recordings, env)) {
      const lane = laneOfSeed.get(s);
      if (lane === undefined) continue;
      const deathStep = laneDeath(t, lane)?.step ?? null;
      all.push({ seed: s, lane, deathStep, aliveRows: aliveRowCount(t, deathStep) });
    }
    fieldData = { tape: t, pose0: poseFields[0] ?? 0 };
    buildField(all, seed);

    const entries = all.filter((e) => e.seed !== seed).slice(0, WALL_GHOST_CAP);
    if (entries.length === 0) {
      ghostView = null;
      ghostChip.textContent = 'ghosts —';
      ghostChip.title = 'This environment has no other recorded seed to ghost.';
      return;
    }
    const nextView: GhostView = { tape: t, windowSteps: stride, poseFields, entries };
    const key = `${run.id}|${env}|${seed}|${entries.length}`;
    if (key === ghostKey && ghostLayer !== null && sceneHost?.ghosts === ghostLayer) {
      ghostView = nextView;
      ghostSample = -1;
      fieldSample = -1;
      ghostChip.textContent = ghostChipText(nextView);
      return;
    }
    ghostKey = key;
    ghostView = nextView;
    const build = (tier: QualityTier): GhostLayer => makeGhostLayer({
      category: info.category,
      envId: env,
      geometry: source.geometry,
      maxGhosts: entries.length,
      tokens: TOKENS,
      accent: accentCss(info.slug),
      tier,
    });
    const ghosts: SceneHostGhosts = { layer: build(TIER_BY_LABEL[tierLabel]), rebuild: build };
    queueStage(async () => {
      if (destroyed || generation !== heroGeneration || sceneHost === null || ghostView === null) {
        ghosts.layer.dispose();
        return;
      }
      sceneHost.setGhosts(ghosts);
      ghostLayer = sceneHost.ghosts;
      if (ghostLayer !== null) {
        primeGhosts(ghostLayer, ghostView);
        ghostSample = ghostRowIndex(frameAt, ghostView.windowSteps, ghostView.tape.rows);
        paintGhosts(ghostLayer, ghostView, ghostSample);
        ghostChip.textContent = ghostChipText(ghostView);
        ghostChip.title =
          `${formatCount(ghostView.entries.length)} recorded seeds of ${info.name}, `
          + `one window-end pose every ${formatCount(ghostView.windowSteps)} steps from the run's tape, `
          + "on the hero's own clock. "
          + (framing === 'track'
            ? `The stage frames about ${formatDistance(stageWindow.width)} of track around the hero, `
              + 'so a seed that has walked away from it is off camera — the field strip under the stage '
              + 'is where the whole population is. '
            : 'This scene frames its whole world, so every seed is inside the picture. ')
          + 'A seed that went non-finite freezes at its last finite pose and turns dark.';
      }
      await Promise.resolve();
    });
  }

  function rebuildRows(): void {
    if (!detail || !tape || envId === null) {
      rows = [];
      return;
    }
    const recorded = new Set(seedsFor(detail.recordings, envId));
    rows = buildSeedRows(tape, detail.meta, envId, recorded);
  }

  function applyHero(): void {
    const hero_ = heroSeed(selection);
    science?.setHighlight(hero_);
    for (const node of railRows.querySelectorAll<HTMLElement>('.wall-row')) {
      node.classList.toggle('is-hero', Number(node.dataset['seed']) === hero_);
    }
  }

  function watchSeed(seed: number): void {
    const next = promoteSeed(selection, seed, playable);
    if (next === selection) return;
    selection = next;
    applyHero();
    mountHero();
    syncUrl();
  }

  function selectEnv(next: EnvId): void {
    if (!detail) return;
    envId = next;
    science?.setEnv(next);
    frameAt = 0;
    pendingSeekFrame = null;
    paintChips();
    const recorded = seedsFor(detail.recordings, next);
    rebuildRows();
    selection = setLeader(selection, recorded[0] ?? null, playable);
    paintPopulation();
    paintRail();
    mountHero();
    syncUrl();
  }

  function onTapeUpdate(next: Tape): void {
    tape = next;
    rebuildRows();
    paintPopulation();
    paintRail();
    applyGhosts();
    science?.setTape(next);
  }

  function startLivePoll(): void {
    if (livePoll !== null) return;
    livePoll = setInterval(() => {
      if (destroyed || !detail) return;
      if (entry?.inFlight !== true) return;
      void refreshTape(detail.id, abort.signal).then(
        (next) => {
          if (destroyed) return;
          if (tape === null || next.rows !== tape.rows) onTapeUpdate(next);
        },
        () => undefined,
      );
      const now = Date.now();
      if (now - detailPollAt >= DETAIL_POLL_MS) {
        detailPollAt = now;
        void runIndex.load(true).then((entries) => {
          if (destroyed || !detail) return;
          const fresh = entries.find((e) => e.id === detail?.id);
          if (fresh !== undefined) {
            entry = fresh;
            hero?.setLive(fresh.inFlight);
          }
          return loadRunDetail(detail.id, true).then((freshDetail) => {
            if (destroyed) return;
            detail = freshDetail;
            paintChips();
            rebuildRows();
            paintRail();
            applyGhosts();
          });
        }).catch(() => undefined);
      }
    }, LIVE_POLL_MS);
  }

  function renderRun(next: RunDetail, pick: FeaturedPick, indexEntry: RunIndexEntry | null): void {
    detail = next;
    entry = indexEntry;
    envList = wallEnvs(next.meta, next.recordings);
    status.remove();
    main.hidden = false;

    runName.textContent = next.id;
    const ago = formatAgo(next.meta.startedAt ?? null);
    runMeta.textContent = `${runScheduleLine(next.meta)}${ago === '—' ? '' : ` · ${ago}`}`;
    const ablation = next.meta.ablationLabel;
    const complete = indexEntry?.complete !== false;
    runBadges.replaceChildren(
      ...(pick.source === 'keeper' ? [badge('keeper', 'keeper', 'The current promoted keeper')] : []),
      ...(indexEntry?.inFlight === true ? [badge('live', 'keeper', 'This run is writing its tape right now')] : []),
      ...(ablation !== 'none' ? [badge(`ablate ${ablation}`, 'warn', 'OAK_ABLATE for this run')] : []),
      ...(next.recordings.length === 0 ? [badge('no recordings', 'warn', 'Nothing on this run can be replayed')] : []),
      ...(complete ? [] : [badge('incomplete', 'warn', 'Did not reach its declared horizon')]),
    );

    const first = envId !== null && envList.includes(envId)
      ? envId
      : envList.find((id) => seedsFor(next.recordings, id).length > 0) ?? envList[0] ?? null;
    paintChips();
    if (first === null) {
      main.hidden = true;
      body.append(
        emptyState(
          'This run declares no environments',
          'meta.json has no envSchedule, so there is nothing to show.',
        ),
      );
      return;
    }

    science?.destroy();
    science = null;
    scienceHost.replaceChildren();
    science = mountRunScience(scienceHost, {
      runId: next.id,
      envId: first,
      onWatchSeed: (env, seed, step) => {
        if (isEnvId(env) && env !== envId) selectEnv(env);
        if (step !== undefined) pendingSeekFrame = Math.floor(frameFromStep(step));
        watchSeed(seed);
        if (step !== undefined) seekToStep(step);
        window.scrollTo({ top: 0, behavior: 'smooth' });
      },
      onSeek: (step, commit) => scrubToStep(step, commit),
    });

    selectEnv(first);
    startLivePoll();
    void loadTape(next.id, abort.signal).then(
      (loaded) => {
        if (destroyed) return;
        onTapeUpdate(loaded);
      },
      (error: unknown) => {
        if (destroyed) return;
        body.append(
          errorState(error instanceof Error ? error.message : String(error), undefined, {
            title: `Could not read the tape for ${next.id}`,
          }),
        );
      },
    );
  }

  function start(force: boolean): void {
    runIndex.load(force).then(
      (entries) => {
        if (destroyed) return;
        const pick = resolveFeaturedRun(entries, query.run);
        if (pick.runId === null) {
          status.replaceChildren(
            emptyState(
              'No run to show yet',
              'The wall plays a fleet’s recorded seeds and the server found no run with a tape under '
                + `.oak-benchmarks. ${RECORD_RECIPE}`,
              [linkButton('Browse every run', '#/runs', { quiet: true })],
            ),
          );
          return;
        }
        const indexEntry = entries.find((e) => e.id === pick.runId) ?? null;
        loadRunDetail(pick.runId).then(
          (loaded) => {
            if (destroyed) return;
            renderRun(loaded, pick, indexEntry);
          },
          (error: unknown) => {
            if (destroyed) return;
            status.replaceChildren(
              errorState(error instanceof Error ? error.message : String(error), () => {
                status.replaceChildren(spinner('Retrying'));
                start(true);
              }, { title: `Could not load ${pick.runId}` }),
            );
          },
        );
      },
      (error: unknown) => {
        if (destroyed) return;
        status.replaceChildren(
          errorState(error instanceof Error ? error.message : String(error), () => {
            status.replaceChildren(spinner('Retrying'));
            start(true);
          }, {
            title: 'Could not reach the API',
            actions: [el('span', {
              class: 'empty-detail',
              text: 'The dev server proxies /api to :5174 — start it with `bun run dev`.',
            })],
          }),
        );
      },
    );
  }

  start(false);

  const handle: ViewHandle = {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      abort.abort();
      if (livePoll !== null) clearInterval(livePoll);
      if (typeof window !== 'undefined') window.removeEventListener('keydown', onKey);
      if (typeof document !== 'undefined') document.removeEventListener('fullscreenchange', onFullscreenChange);
      for (const spark of sparks) spark.destroy();
      sparks.length = 0;
      science?.destroy();
      science = null;
      for (const cached of heroCache.values()) cached.destroy();
      heroCache.clear();
      hero = null;
      sceneHost?.destroy();
      sceneHost = null;
      root.remove();
    },
  };
  return handle;
};
