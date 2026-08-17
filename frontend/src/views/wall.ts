import type { EnvResult, Policy, RecordingRef, RunDetail, RunIndexEntry, RunMeta } from '@shared/types';
import { ENVS, ENV_IDS, isEnvId, type EnvId } from '@shared/envs';
import type { PlaybackSource } from '@kit/playback';
import { entryFor, parseCatalog } from '@sim/catalog';
import type { Tape } from '../tape';
import {
  actionButton,
  accentCss,
  badge,
  el,
  emptyState,
  errorState,
  eyebrow,
  spinner,
  viewShell,
} from '../components/table';
import { sparkline, type SparklineHandle } from '../components/sparkline';
import { mountRunScience, type RunScienceHandle } from '../components/science';
import {
  binnedRates,
  deliveredRate,
  envLanes,
  envName,
  envShares,
  formatAgo,
  formatCount,
  formatPercent,
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
import { mountSceneHost, type HostFrameInfo, type SceneHostHandle } from '../theater/sceneHost';

const WALL_SPEEDS: readonly number[] = [1, 4, 16, 64, 256, 1024];

const DEFAULT_SPEED = 1;

type WallSort = 'rate' | 'seed' | 'retention';
const WALL_SORTS: readonly WallSort[] = ['rate', 'seed', 'retention'];

const SORT_LABEL: Record<WallSort, string> = {
  rate: 'reward rate',
  seed: 'seed',
  retention: 'retention',
};

const POLICY_LABEL: Record<Policy, string> = {
  oak: 'oak',
  ppo: 'ppo',
  random: 'rand',
};

interface WallParams {
  run?: string;
  env?: EnvId;
  seed?: number;
  at?: number;
  speed?: number;
  sort?: WallSort;
  policy?: Policy;
}

function parseWallParams(hash: string): WallParams {
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

  const policy = params.get('policy');
  if (policy !== null && (policy.trim() === 'ppo' || policy.trim() === 'random')) {
    out.policy = policy.trim() as Policy;
  }
  return out;
}

function wallHref(params: WallParams = {}): string {
  const parts: string[] = [];
  const push = (key: string, value: string): void => {
    parts.push(`${key}=${encodeURIComponent(value)}`);
  };
  if (params.run !== undefined && params.run !== '') push('run', params.run);
  if (params.env !== undefined) push('env', params.env);
  if (params.seed !== undefined && Number.isFinite(params.seed)) push('seed', String(Math.floor(params.seed)));
  if (params.at !== undefined && Number.isFinite(params.at) && params.at > 0) push('at', String(Math.floor(params.at)));
  if (params.speed !== undefined && params.speed !== DEFAULT_SPEED) push('speed', String(params.speed));
  if (params.sort !== undefined && params.sort !== 'rate') push('sort', params.sort);
  if (params.policy !== undefined && params.policy !== 'oak') push('policy', params.policy);
  return parts.length > 0 ? `#/?${parts.join('&')}` : '#/';
}

function urlPosition(clockPosition: number | null, pending: number): number | null {
  const at = clockPosition !== null && Number.isFinite(clockPosition) ? clockPosition : pending;
  if (!Number.isFinite(at) || at <= 0) return null;
  return Math.floor(at);
}

type FeaturedSource = 'param' | 'keeper' | 'newest' | null;

interface FeaturedPick {
  runId: string | null;
  source: FeaturedSource;
}

function resolveFeaturedRun(
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

  const playable = entries.filter((entry) => entry.policy === 'oak' && entry.hasRecordings);
  const complete = newest(playable.filter((entry) => entry.complete));
  if (complete) return { runId: complete.id, source: 'newest' };
  const any = newest(playable);
  if (any) return { runId: any.id, source: 'newest' };
  return { runId: null, source: null };
}

interface HeroSelection {
  leader: number | null;
  promoted: number | null;
}

function heroSeed(selection: HeroSelection): number | null {
  return selection.promoted ?? selection.leader;
}

function promoteSeed(
  selection: HeroSelection,
  seed: number,
  playable: (seed: number) => boolean,
): HeroSelection {
  if (!playable(seed)) return selection;
  if (selection.promoted === seed) return selection;
  return { ...selection, promoted: seed };
}

function escapeSelection(selection: HeroSelection): HeroSelection {
  if (selection.promoted === null) return selection;
  return { ...selection, promoted: null };
}

function setLeader(
  selection: HeroSelection,
  leader: number | null,
  playable: (seed: number) => boolean,
): HeroSelection {
  const promoted = selection.promoted !== null && playable(selection.promoted) ? selection.promoted : null;
  return { leader, promoted };
}

function degradedReason(
  recordings: readonly RecordingRef[],
  envId: EnvId | null,
): string | null {
  if (recordings.length === 0) return 'no recordings on this run';
  if (envId === null) return 'no environment selected';
  if (seedsFor(recordings, envId).length === 0) return `no recordings for ${ENVS[envId].name} in this run`;
  return null;
}

function baselineReason(
  recordings: readonly RecordingRef[],
  envId: EnvId,
  seed: number | null,
  policy: 'ppo' | 'random',
): string | null {
  if (recordings.length === 0) return `no ${policy} recordings on this baseline`;
  if (seedsFor(recordings, envId).length === 0) return `no ${policy} recordings for ${ENVS[envId].name}`;
  if (seed !== null && recordingRef(recordings, envId, seed) === null) {
    return `no ${policy} recording for ${ENVS[envId].name} seed ${seed}`;
  }
  return null;
}

const RECORD_RECIPE = 'This run has no recording for that selection.';

interface SeedRow {
  seed: number;
  lane: number;
  rate: number | null;
  retention: number | null;
  divergedAt: number | null;
  spark: (number | null)[];
  sparkXs: number[];
  recorded: boolean;
}

function buildSeedRows(
  tape: Tape,
  meta: Pick<RunMeta, 'envSchedule' | 'seeds' | 'steps'>,
  envId: string,
  recorded: ReadonlySet<number>,
): SeedRow[] {
  const { lanes, seeds } = envLanes(meta, envId);
  const maxStep = tape.rows > 0 ? tape.stepEnd[tape.rows - 1] ?? 0 : 0;
  const spark = binnedRates(tape, lanes);
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

function sortPermutation(rows: readonly SeedRow[], sort: WallSort): number[] {
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

function nextSort(sort: WallSort): WallSort {
  const index = WALL_SORTS.indexOf(sort);
  return WALL_SORTS[(index + 1) % WALL_SORTS.length] ?? 'rate';
}

interface PopulationFigures {
  positive: string;
  nonDiverged: string;
  retention: string;
  learned: string;
  source: 'summary' | 'measured' | 'none';
}

function finite(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function populationFigures(
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
    const learned = finite(slice.learnedSeeds);
    return {
      positive: positive === null ? '—' : formatPercent(positive, 0),
      nonDiverged: `${formatCount(seeds - dead)}/${formatCount(seeds)}`,
      retention: retention === null ? '—' : formatPercent(retention, 0),
      learned: learned === null ? '—' : formatCount(learned),
      source: 'summary',
    };
  }

  if (rows.length === 0) {
    return { positive: '—', nonDiverged: '—', retention: '—', learned: '—', source: 'none' };
  }
  return {
    positive: rates.length === 0
      ? '—'
      : formatPercent(rates.filter((v) => v > 0).length / rates.length, 0),
    nonDiverged: `${formatCount(rows.length - diverged)}/${formatCount(rows.length)}`,
    retention: formatPercent(retained / rows.length, 0),
    learned: '—',
    source: 'measured',
  };
}

function runScheduleLine(meta: Pick<RunMeta, 'batch' | 'steps' | 'envSchedule'>): string {
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

type WallAction =
  | { kind: 'toggle' }
  | { kind: 'step'; frames: number }
  | { kind: 'env'; direction: 1 | -1 }
  | { kind: 'sort' }
  | { kind: 'debug' }
  | { kind: 'fullscreen' }
  | { kind: 'escape' };

function wallKeyAction(key: string, shift = false): WallAction | null {
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
    case 'Escape':
      return { kind: 'escape' };
    default:
      return null;
  }
}

function cycleEnv(list: readonly EnvId[], current: EnvId | null, direction: 1 | -1): EnvId | null {
  if (list.length === 0) return null;
  const index = current === null ? -1 : list.indexOf(current);
  const next = index < 0 ? (direction > 0 ? 0 : list.length - 1) : (index + direction + list.length) % list.length;
  return list[next] ?? null;
}

function wallEnvs(meta: Pick<RunMeta, 'envSchedule'>, recordings: readonly RecordingRef[]): EnvId[] {
  const seen = new Set<EnvId>();
  for (const share of envShares(meta)) seen.add(share.envId);
  for (const rec of recordings) if (isEnvId(rec.envId)) seen.add(rec.envId);
  return ENV_IDS.filter((id) => seen.has(id));
}

function nextRowDueMs(tape: Pick<Tape, 'rows' | 'wallSeconds'> | null): number {
  if (tape === null || tape.rows < 2) return 0;
  const last = tape.wallSeconds[tape.rows - 1];
  const prev = tape.wallSeconds[tape.rows - 2];
  if (last === undefined || prev === undefined) return 0;
  const perRow = (last - prev) * 1000;
  return Number.isFinite(perRow) && perRow > 0 ? perRow : 0;
}

function tapeWindowSteps(tape: Pick<Tape, 'stepStart' | 'stepEnd'>): number {
  const window = (tape.stepEnd[0] ?? 0) - (tape.stepStart[0] ?? 0);
  return window > 0 ? window : 1;
}


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

export interface WallHandle { destroy(): void }

export const mount = (host: HTMLElement): WallHandle => {
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
          entry.policy !== 'oak' ? badge(entry.policy, 'quiet') : null,
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
  function openPicker(): void {
    picker.hidden = false;
    void runIndex.load(true).then(() => { if (!picker.hidden) paintPicker(); });
    paintPicker();
    pickerSearch.focus();
  }
  runName.addEventListener('click', () => {
    if (picker.hidden) openPicker();
    else picker.hidden = true;
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
  let speed: number = query.speed ?? DEFAULT_SPEED;
  let policy: Policy = query.policy ?? 'oak';
  const baselineDetails: Partial<Record<'ppo' | 'random', RunDetail | null>> = {};
  let selection: HeroSelection = { leader: null, promoted: query.seed ?? null };
  let rows: SeedRow[] = [];
  let stride = 1;
  let frameAt = query.at ?? 0;
  let pendingSeekFrame: number | null = query.at ?? null;
  let scrubbing = false;
  let statsAccum = 1000;
  let debugOn = false;

  let sceneHost: SceneHostHandle | null = null;
  let stageQueue: Promise<void> = Promise.resolve();
  let hero: HeroHandle | null = null;
  const heroCache = new Map<string, HeroHandle>();
  const HERO_CACHE_MAX = 6;
  let heroGeneration = 0;
  let heroAttached = false;
  let heroSource: PlaybackSource | null = null;
  const sparks: SparklineHandle[] = [];
  let livePoll: ReturnType<typeof setTimeout> | null = null;

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
  const policySep = el('div', { class: 'tsep' });
  const policyRow = el('div', { class: 'wall-speeds' });
  policySep.hidden = true;
  policyRow.hidden = true;
  const debugButton = el('button', { class: 'tbtn', type: 'button', text: 'g', title: 'Ground-truth overlay (g)' });
  const fullButton = el('button', { class: 'tbtn', type: 'button', text: '⛶', title: 'Fullscreen (f)' });
  const transport = el('div', { class: 'wall-transport' }, [
    playButton,
    backButton,
    fwdButton,
    scrub,
    readout,
    el('div', { class: 'tsep' }),
    speedRow,
    policySep,
    policyRow,
    el('div', { class: 'tsep' }),
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

  const stageCol = el('div', { class: 'wall-stage-col' }, [stage, transport]);
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
    if (speed !== DEFAULT_SPEED) params.speed = speed;
    if (sort !== 'rate') params.sort = sort;
    if (policy !== 'oak') params.policy = policy;
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

  function activeDetail(): RunDetail | null {
    return policy === 'oak' ? detail : baselineDetails[policy] ?? null;
  }

  function applyPolicy(): void {
    for (const button of policyRow.querySelectorAll('button')) {
      button.classList.toggle('is-on', button.dataset['policy'] === policy);
    }
  }

  function paintPolicies(): void {
    const baselines = detail?.baselines ?? {};
    const names = (['ppo', 'random'] as const).filter((p) => baselines[p] !== undefined);
    const empty = names.length === 0;
    policySep.hidden = empty;
    policyRow.hidden = empty;
    if (empty) {
      policyRow.replaceChildren();
      return;
    }
    policyRow.replaceChildren(
      ...(['oak' as Policy, ...names]).map((p) => {
        const button = el('button', {
          class: 'tbtn',
          type: 'button',
          text: POLICY_LABEL[p],
          dataset: { policy: p },
          title: p === 'oak' ? 'The OaK fleet' : `The ${p} baseline fleet`,
        });
        button.addEventListener('click', () => selectPolicy(p));
        return button;
      }),
    );
    applyPolicy();
  }

  function ensureBaselineDetail(p: 'ppo' | 'random'): void {
    if (!detail || baselineDetails[p] !== undefined) return;
    const id = detail.baselines[p];
    if (id === undefined) return;
    void loadRunDetail(id).then(
      (loaded) => {
        if (destroyed) return;
        baselineDetails[p] = loaded;
        if (policy === p) refreshPolicyView();
      },
      () => {
        if (destroyed) return;
        baselineDetails[p] = null;
        if (policy === p) refreshPolicyView();
      },
    );
  }

  function refreshPolicyView(): void {
    const run = activeDetail();
    if (run !== null && envId !== null && selection.leader === null) {
      selection = { ...selection, leader: seedsFor(run.recordings, envId)[0] ?? null };
    }
    rebuildRows();
    paintRail();
    mountHero();
  }

  function selectPolicy(p: Policy): void {
    if (policy === p) return;
    const at = sceneHost?.clock.position();
    if (at !== undefined && Number.isFinite(at) && at > 0) pendingSeekFrame = Math.floor(at);
    policy = p;
    if (p !== 'oak') ensureBaselineDetail(p);
    science?.setPolicy(p);
    applyPolicy();
    rebuildRows();
    paintRail();
    mountHero();
    syncUrl();
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
      figure('learned', figures.learned, 'Seeds whose best windowed reward rate cleared their own noise — a null built from that lane’s residuals, not a task-completion test (fleet.py report)'),
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
      {
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

      node.addEventListener('click', () => {
        if (sparkDragged) { sparkDragged = false; return; }
        watchSeed(row.seed);
      });
      nodes.push(node);
    });

    railRows.replaceChildren(...nodes);
  }

  function playable(seed: number): boolean {
    const run = activeDetail();
    if (!run || envId === null) return false;
    return recordingRef(run.recordings, envId, seed) !== null;
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
    if (
      !heroAttached
      && hero !== null
      && sceneHost !== null
      && sceneHost.envId === hero.source.envId
      && hero.source.frameCount > 1
    ) {
      heroAttached = true;
      heroSource = hero.source;
      const at = pendingSeekFrame !== null
        ? Math.min(pendingSeekFrame, hero.source.frameCount - 1)
        : 0;
      pendingSeekFrame = null;
      const host = sceneHost;
      void host
        .setSource(hero.source, { at })
        .then(() => {
          host.setSpeed(speed);
          host.play();
          syncPlay();
        })
        .catch((error: unknown) => {
          heroAttached = false;
          showStageNote('Could not play this seed', [
            errorState(error instanceof Error ? error.message : String(error)),
          ]);
        });
      scrubInput.max = String(Math.max(1, hero.source.frameCount - 1));
    }
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

    statsAccum += info.frameMs;
    if (statsAccum < 150) return;
    statsAccum = 0;
    readout.textContent =
      `step ${info.step < 0 ? 'init' : formatCount(info.step)} / ${formatStep(heroSteps())}`
      + ` · ${formatClock(info.simTime)}`;
    syncPlay();
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
    const run = activeDetail();
    const env = envId;
    if (hero !== null) {
      hero.setCallbacks(null, null);
      hero.setLive(false);
    }
    hero = null;
    heroAttached = false;
    heroSource = null;
    scrubTicks.replaceChildren();
    scrubLoaded.style.width = '0%';
    if (!detail || env === null) return;
    if (run === null) {
      if (policy === 'oak') return;
      const pointer = detail.baselines[policy];
      const failed = pointer === undefined || baselineDetails[policy] === null;
      showStageNote(failed ? `No ${policy} baseline to play` : `Loading the ${policy} baseline`, [
        failed
          ? emptyState(
              pointer === undefined
                ? `no ${policy} baseline pointer for this fleet`
                : `could not load the ${policy} baseline run ${pointer}`,
              RECORD_RECIPE,
              [actionButton('Back to OaK', () => selectPolicy('oak'), { quiet: true })],
            )
          : spinner(`Loading ${pointer}`),
      ]);
      readout.textContent = 'no recording';
      transport.classList.add('is-idle');
      return;
    }

    const seed = heroSeed(selection);
    const reason = policy === 'oak'
      ? degradedReason(run.recordings, env)
      : baselineReason(run.recordings, env, seed, policy);
    if (reason !== null || seed === null) {
      showStageNote('Nothing to replay here', [
        emptyState(reason ?? 'no seed to play', RECORD_RECIPE, [
          policy === 'oak'
            ? actionButton('Find a run with recordings', () => openPicker(), { quiet: true })
            : actionButton('Back to OaK', () => selectPolicy('oak'), { quiet: true }),
        ]),
      ]);
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
      tagSub.textContent = `seed ${seed}${selection.promoted === null ? ' · leader' : ' · promoted'}`
        + (policy === 'oak' ? '' : ` · ${policy}`);
    }

    if (tape !== null && tape.rows > 0) stride = tapeWindowSteps(tape);

    const startAt = pendingSeekFrame ?? frameAt;
    pendingSeekFrame = Number.isFinite(startAt) && startAt > 0 ? Math.floor(startAt) : null;

    const key = `${run.id}|${env}|${seed}`;
    let h = heroCache.get(key);
    if (h === undefined) {
      h = loadHero({
        entry: entryFor(parseCatalog(run.meta.envCatalog), env),
        runId: run.id,
        envId: env,
        seed,
        horizonSteps: run.meta.steps,
        live: policy === 'oak' && entry?.inFlight === true,
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
      h.setLive(policy === 'oak' && entry?.inFlight === true);
    }
    hero = h;
    const handle = h;

    queueStage(async () => {
      if (destroyed || generation !== heroGeneration) return;
      const attachSource = handle.source.frameCount > 1
        ? handle.source
        : makeInitialSource(env, seed);
      if (attachSource === handle.source) {
        heroAttached = true;
        heroSource = handle.source;
      } else {
        heroSource = attachSource;
      }
      const attachAt = pendingSeekFrame !== null
        ? Math.min(pendingSeekFrame, attachSource.frameCount - 1)
        : 0;
      if (pendingSeekFrame !== null && pendingSeekFrame < attachSource.frameCount) {
        pendingSeekFrame = null;
      }
      if (sceneHost === null) {
        sceneHost = await mountSceneHost(stage, {
          envId: env,
          source: attachSource,
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
        await sceneHost.setEnv(env, attachSource, { at: attachAt });
      } else {
        await sceneHost.setSource(attachSource, { at: attachAt });
      }
      applySpeed();
      sceneHost.play();
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
  }

  function rebuildRows(): void {
    if (!detail || !tape || envId === null) {
      rows = [];
      return;
    }
    const recorded = new Set(seedsFor(activeDetail()?.recordings ?? [], envId));
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
    const recorded = seedsFor(activeDetail()?.recordings ?? [], next);
    rebuildRows();
    selection = setLeader(selection, recorded[0] ?? null, playable);
    paintPopulation();
    paintRail();
    mountHero();
    syncUrl();
  }

  function onTapeUpdate(next: Tape): void {
    tape = next;
    stride = tapeWindowSteps(next);
    rebuildRows();
    paintPopulation();
    paintRail();
    science?.setTape(next);
  }

  function startLivePoll(): void {
    if (livePoll !== null) return;
    let pollBackoffMs = 0;
    const tick = (): void => {
      livePoll = null;
      if (destroyed || !detail) return;
      const rowsBefore = tape?.rows ?? 0;
      const startedAt = performance.now();
      const again = (): void => {
        if (destroyed) return;
        if ((tape?.rows ?? 0) > rowsBefore) pollBackoffMs = 0;
        else pollBackoffMs = Math.max(performance.now() - startedAt, pollBackoffMs * 2);
        livePoll = setTimeout(tick, Math.max(nextRowDueMs(tape), pollBackoffMs));
      };
      const refreshDetail = runIndex.load(true).then((entries) => {
        if (destroyed || !detail) return;
        if (!picker.hidden) paintPicker();
        const fresh = entries.find((e) => e.id === detail?.id);
        if (fresh !== undefined) {
          entry = fresh;
          if (policy === 'oak') hero?.setLive(fresh.inFlight);
        }
        return loadRunDetail(detail.id, true).then((freshDetail) => {
          if (destroyed) return;
          detail = freshDetail;
          paintChips();
          paintPolicies();
          rebuildRows();
          paintRail();
        });
      });
      const refreshTapeIfLive = entry?.inFlight === true
        ? refreshTape(detail.id, abort.signal).then((next) => {
          if (destroyed) return;
          if (tape === null || next.rows !== tape.rows) onTapeUpdate(next);
        })
        : Promise.resolve();
      void Promise.allSettled([refreshDetail, refreshTapeIfLive]).then(again);
    };
    tick();
  }

  function renderRun(next: RunDetail, pick: FeaturedPick, indexEntry: RunIndexEntry | null): void {
    detail = next;
    entry = indexEntry;
    envList = wallEnvs(next.meta, next.recordings);
    status.remove();
    main.hidden = false;

    paintPolicies();
    if (policy !== 'oak') ensureBaselineDetail(policy);

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
    science.setPolicy(policy);

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
              [actionButton('Browse every run', () => openPicker(), { quiet: true })],
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

  const handle: WallHandle = {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      abort.abort();
      if (livePoll !== null) clearTimeout(livePoll);
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
