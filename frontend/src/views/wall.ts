// #/ — the Wall. The home page, and the only page that shows a whole population
// living at once.
//
// One environment, one run, every recorded seed. The leader plays full-fidelity
// in the hero scene; every other recorded seed rides the same clock as a ghost.
// The rail on the right is the population as numbers — rank, seed, average
// reward rate, the shape of its learning — and the two halves are the same
// fleet, so hovering a row lights its ghost and clicking it makes that seed the
// hero without touching anything else.
//
// Vocabulary is the lab's (shared/types.ts v2): SEEDS, average reward rate,
// numerically diverged, retention. Never "lanes", never "delivered".
//
// WHAT THIS FILE OWNS: the view. Four things it deliberately does not own:
//
//   * the SCENE — `theater/sceneHost` mounts it, exactly as the Theater does, so
//     a scene that is right there is right here.
//   * the HERO DATA — `theater/loader.loadTheaterRecording`, the same streamed,
//     chunk-aware, damage-reporting path the Theater uses.
//   * the POPULATION DATA — `api.fetchWall`, one request per environment that
//     brings every recorded seed down as strided pose samples.
//   * the GHOST RENDERING — `kit/ghosts`, handed to `sceneHost` as a mount option
//     so it lives inside the hero's own scene and survives its tier rebuilds.
//
// What is left is exactly the wall's own job: WHICH run and seed are on stage,
// WHICH sample of the population the clock is on, and what the numbers mean.
//
// The clock is `sceneHost`'s clock and nothing else. Ghost samples are indexed
// off the very same frame position, so hero and ghosts cannot drift: there is
// one time on this page. Nothing here is ever fabricated — a run with no
// recordings gets a rail and an honest note, never a fake ghost.
//
// THE FIELD. A ghost is drawn at its TRUE position, and on a track environment the
// population's true positions do not fit in one camera: HalfCheetah seeds start
// 0.18 units apart and finish 1.7 KILOMETRES apart, while the savanna frame is
// about five metres wide. So the stage is not asked to lie about that — the strip
// under it (`paintField`, and the geometry section further down) draws every seed's
// position along the track at a scale that fits, with the camera's own window
// marked inside it, and the transport chip prints how many ghosts are genuinely in
// frame rather than how many were submitted to the GPU.

import type { ViewHandle, ViewMount } from '../router';
import type {
  FleetColumnar,
  RecordingRef,
  RolesResponse,
  RunDetail,
  RunIndexEntry,
  RunMeta,
  SummaryPolicyResult,
} from '@shared/types';
import { ENVS, ENV_IDS, isEnvId, type EnvCategory, type EnvId } from '@shared/envs';
import { MAX_CATCHUP_FRAMES, type HarnessSource } from '@kit/harness';
import { makeGhostLayer, type GhostLayer } from '@kit/ghosts';
import { TOKENS } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import { api, fetchWall, type WallData, type WallLane } from '../api';
import {
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
import {
  envName,
  envShares,
  envSliceResult,
  formatCount,
  formatPercent,
  formatSigned,
  formatStep,
  laneSeries,
  loadRunDetail,
  policiesIn,
  primaryResult,
  roles as rolesResource,
  runIndex,
  windowRates,
} from '../state';
import { accentCss } from './gallery';
import {
  loadTheaterRecording,
  recordingRef,
  seedsFor,
  seekTargetForTick,
  theaterHref,
  type TheaterRecording,
  type TimelineTick,
} from '../theater/loader';
import { clampPosition, formatClock, stepPosition } from '../theater/player';
import {
  mountSceneHost,
  type HostFrameInfo,
  type SceneHostGhosts,
  type SceneHostHandle,
} from '../theater/sceneHost';

// ---------------------------------------------------------------------------
// constants — every one of them a stated choice, not a knob
// ---------------------------------------------------------------------------

/**
 * ONE speed model for the whole app: a preset is a REAL-TIME multiple of the
 * environment's own clock (×1 = controlHz record steps per second), the same
 * meaning the playback clock has always had. ×1..×16 are walked playback
 * (events fire); the top presets are honestly a flip-book of the run — the
 * host snaps past MAX_CATCHUP_FRAMES and celebrations thin out on their own.
 */
export const WALL_SPEEDS: readonly number[] = [1, 4, 16, 64, 256, 1024];

/** Ghost samples per lane. 2400 at 12 fps = a 200 s pass at ×1, any run length. */
export const WALL_TARGET_SAMPLES = 2400;

/**
 * Rows that get a sparkline. 1024 rows of text are cheap; 1024 canvases with a
 * ResizeObserver each are not. Past this the rail keeps every seed and drops the
 * per-row curve (the numbers are the ranking; the curve is the flourish).
 */
export const SPARK_ROW_BUDGET = 128;

export type WallSort = 'rate' | 'seed' | 'retention';
export const WALL_SORTS: readonly WallSort[] = ['rate', 'seed', 'retention'];

const SORT_LABEL: Record<WallSort, string> = {
  rate: 'reward rate',
  seed: 'seed',
  retention: 'retention',
};

// ---------------------------------------------------------------------------
// the URL — #/?run=&env=&seed=&at=&speed=&sort=
// ---------------------------------------------------------------------------

export interface WallParams {
  run?: string;
  env?: EnvId;
  /** the PROMOTED seed; absent means "whoever leads" */
  seed?: number;
  /** frame position */
  at?: number;
  speed?: number;
  sort?: WallSort;
}

/**
 * Parse the wall's query. Every field is optional and every bad field is
 * DROPPED rather than corrected: a stale `env=` from a renamed environment
 * should land on the wall's own default, not on a neighbouring env's scene.
 */
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

/** The canonical wall link. Fixed key order, so a round-trip is byte-stable. */
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

/**
 * The frame position a URL write must carry.
 *
 * THE BUG THIS EXISTS TO PREVENT. The wall's `at` used to be serialized from the
 * cached progress FRACTION, which is written by `onFrame` and by nothing else — so
 * every `syncUrl` that runs synchronously after a seek wrote the position the reader
 * had just LEFT. Measured before the fix: an arrow key moved the player to frame 889
 * and the address bar to `at=50`; three arrows reached 2,569 and wrote `at=1730`; and
 * a deep link with `?at=1234` seeked correctly and then immediately rewrote itself
 * WITHOUT `at`, so refreshing your own shared link restarted the run at zero.
 *
 * So the position comes from the clock — which `seek` updates synchronously, before
 * any frame ticks — and from the pending deep-linked position when no host is mounted
 * yet, which is the case the first `syncUrl` of a page load runs in.
 *
 * Returns null when there is nothing worth writing (position 0 is the default and is
 * omitted from the href).
 */
export function urlPosition(clockPosition: number | null, pending: number): number | null {
  const at = clockPosition !== null && Number.isFinite(clockPosition) ? clockPosition : pending;
  if (!Number.isFinite(at) || at <= 0) return null;
  return Math.floor(at);
}

// ---------------------------------------------------------------------------
// which run the wall opens on
// ---------------------------------------------------------------------------

export type FeaturedSource = 'param' | 'bestKernel' | 'newest' | null;

export interface FeaturedPick {
  runId: string | null;
  source: FeaturedSource;
}

/**
 * Featured-run resolution, in order:
 *
 *   1. `?run=` — an explicit ask is always honoured, recordings or not. That is
 *      how a run with nothing recorded (the degraded wall) is reachable at all.
 *   3. ROLES `bestKernel`, IF it has recordings. The current best kernel is the
 *      run worth staring at; a best kernel with no recordings cannot be a wall,
 *      so it is skipped rather than shown empty.
 *   4. the newest COMPLETE run with recordings.
 *   5. any run with recordings (newest first) — better than an empty page.
 *   6. null: the honest empty state.
 */
export function resolveFeaturedRun(
  entries: readonly RunIndexEntry[],
  roles: RolesResponse | null,
  requested?: string,
): FeaturedPick {
  const byId = new Map<string, RunIndexEntry>();
  for (const entry of entries) byId.set(entry.id, entry);

  if (requested !== undefined && byId.has(requested)) return { runId: requested, source: 'param' };


  const best = roles?.bestKernel ?? null;
  if (best !== null && byId.get(best)?.hasRecordings === true) return { runId: best, source: 'bestKernel' };

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

// ---------------------------------------------------------------------------
// the wall clock: stride
// ---------------------------------------------------------------------------

/** Is `n` prime? Trial division — the callers ask for values in the hundreds. */
export function isPrime(n: number): boolean {
  if (!Number.isInteger(n) || n < 2) return false;
  if (n % 2 === 0) return n === 2;
  for (let d = 3; d * d <= n; d += 2) if (n % d === 0) return false;
  return true;
}

/** Smallest prime at or above `n`. */
export function nextPrime(n: number): number {
  let candidate = Math.max(2, Math.ceil(Number.isFinite(n) ? n : 2));
  while (!isPrime(candidate)) candidate++;
  return candidate;
}

/**
 * Record steps per ghost sample, so a lane of any length is ~`target` samples.
 *
 * The stride is a PRIME, not the bare `ceil(steps / target)`, because a sampling
 * grid that shares a factor with an environment's own cycle length is PHASE-LOCKED:
 * every sample lands on the same phase and the ghost stands still while the hero
 * moves. That is measured, not hypothetical. DeepSea descends on a 7-step cycle;
 * the bare ceiling picks 42 on a 100k lane and 417 on a 1M one, and 42 is an
 * exact multiple of 7. Against the live
 * endpoint (`frontend-demo-16env-20260803`, DeepSea, 8 seeds): stride 42 visited
 * 3 distinct poses in 2382 samples and stride 105 visited 3 in 954, while the next
 * primes up — 43 and 107 — visited 28 and 27. Seven frozen ghosts on one cell are
 * indistinguishable from "these seeds never learned", so the wall must not be able
 * to produce that grid at all.
 *
 * A prime `p` divides no integer below itself, so a prime stride is coprime to every
 * cycle length under it — and a run long enough to need a stride past 64 (anything
 * over `64 × target` steps) is coprime to the entire 2..64 gridworld range. Shorter
 * runs get a prime that could itself be a cycle length (a 15k-step run wants 7); it
 * is still the best available choice, since only 1 and the primes above 64 are
 * coprime to ALL of 2..64 and forcing `p > 64` on a short run would throw away most
 * of its samples for a case no root on disk has. `steps <= target` returns 1 — every
 * row, no grid, no aliasing possible.
 */
export function wallStride(steps: number, target: number = WALL_TARGET_SAMPLES): number {
  if (!Number.isFinite(steps) || steps <= 0) return 1;
  const want = Math.max(1, Math.floor(target));
  const need = Math.ceil(steps / want);
  return need <= 1 ? 1 : nextPrime(need);
}

// ---------------------------------------------------------------------------
// the hero selection state machine
// ---------------------------------------------------------------------------

export interface HeroSelection {
  /** the seed that leads by final average reward rate, among PLAYABLE seeds */
  leader: number | null;
  /** the seed the reader promoted, if any */
  promoted: number | null;
}

export const EMPTY_SELECTION: HeroSelection = { leader: null, promoted: null };

/** The seed on stage: the promotion when there is one, the leader otherwise. */
export function heroSeed(selection: HeroSelection): number | null {
  return selection.promoted ?? selection.leader;
}

/**
 * Promote a seed. A seed with no recording cannot be a hero, so the promotion is
 * REFUSED (state returned unchanged) rather than swapping the stage for an error
 * — the rail row for such a seed is marked and does not pretend to be clickable.
 */
export function promoteSeed(
  selection: HeroSelection,
  seed: number,
  playable: (seed: number) => boolean,
): HeroSelection {
  if (!playable(seed)) return selection;
  if (selection.promoted === seed) return selection;
  return { ...selection, promoted: seed };
}

/** Esc: back to the leader. Idempotent when nothing is promoted. */
export function escapeSelection(selection: HeroSelection): HeroSelection {
  if (selection.promoted === null) return selection;
  return { ...selection, promoted: null };
}

/**
 * A new leader (env switch, data refresh). A promotion SURVIVES it only when the
 * promoted seed is still playable here — otherwise the reader would be staring
 * at another environment's seed number.
 */
export function setLeader(
  selection: HeroSelection,
  leader: number | null,
  playable: (seed: number) => boolean,
): HeroSelection {
  const promoted = selection.promoted !== null && playable(selection.promoted) ? selection.promoted : null;
  return { leader, promoted };
}

// ---------------------------------------------------------------------------
// the degraded wall
// ---------------------------------------------------------------------------

/**
 * Why this (run, env) cannot show a stage — or null when it can.
 *
 * A run with no recordings is a perfectly good POPULATION (the rail still reads
 * from fleet.jsonl); it just has no experience to replay. The wall says exactly
 * that and prints the recipe. It never fabricates a ghost.
 */
export function degradedReason(
  recordings: readonly RecordingRef[],
  envId: EnvId | null,
): string | null {
  if (recordings.length === 0) return 'no recordings on this run';
  if (envId === null) return 'no environment selected';
  if (seedsFor(recordings, envId).length === 0) return `no recordings for ${ENVS[envId].name} in this run`;
  return null;
}

export const RECORD_RECIPE =
  'Re-run the fleet with OAK_RECORD_ACTIONS=1 (every seed) or OAK_RECORD_SEEDS=3,8 (a couple) '
  + 'to write the actions.bin streams the wall replays.';

// ---------------------------------------------------------------------------
// the rail: one row per seed
// ---------------------------------------------------------------------------

export interface SeedRow {
  seed: number;
  /** final AVERAGE REWARD RATE — cumulative rewardSum ÷ steps, the lab's headline */
  rate: number | null;
  /** final ÷ peak of the MEASURED windowed rate; null when there is no positive peak */
  retention: number | null;
  /** step of the first non-finite value; null = never numerically diverged */
  divergedAt: number | null;
  /** the windowed reward-rate series — the sparkline's shape */
  spark: (number | null)[];
  /** is there an actions.bin for this seed (can it be the hero) */
  recorded: boolean;
}

export interface SeedRowOptions {
  recorded?: ReadonlySet<number>;
  /** cumulative reward field (default `rewardSum`) */
  rateField?: string;
  /** divergence field (default `firstNonFiniteStep`) */
  divergenceField?: string;
}

function lastFinite(values: readonly (number | null)[]): { value: number; index: number } | null {
  for (let i = values.length - 1; i >= 0; i--) {
    const value = values[i];
    if (value !== null && value !== undefined && Number.isFinite(value)) return { value, index: i };
  }
  return null;
}

/**
 * Retention, per seed: final ÷ peak of the windowed reward rate — the same ratio
 * `tools/fleet.py report` publishes, computed here from fleet.jsonl so it exists
 * on every root instead of only on summarySchemaVersion 2 ones.
 *
 * A seed whose peak is not positive has no meaningful ratio (dividing by ~0 or by
 * a negative peak produces a number that ranks nonsense above sense), so it is
 * null and sinks in the sort.
 */
export function retentionOf(window: readonly (number | null)[]): number | null {
  let peak = -Infinity;
  for (const value of window) {
    if (value !== null && value !== undefined && Number.isFinite(value) && value > peak) peak = value;
  }
  const final = lastFinite(window);
  if (final === null || !Number.isFinite(peak) || peak <= 0) return null;
  return final.value / peak;
}

/** Columnar fleet data -> one row per seed. Pure. */
export function buildSeedRows(columnar: FleetColumnar, options: SeedRowOptions = {}): SeedRow[] {
  const rateField = options.rateField ?? 'rewardSum';
  const divergenceField = options.divergenceField ?? 'firstNonFiniteStep';
  const recorded = options.recorded ?? new Set<number>();
  const steps = columnar.steps;
  const cumulative = laneSeries(columnar, rateField);
  const divergence = new Map<number, (number | null)[]>();
  for (const lane of laneSeries(columnar, divergenceField)) divergence.set(lane.seed, lane.values);

  const rows: SeedRow[] = [];
  for (const lane of cumulative) {
    const final = lastFinite(lane.values);
    const at = final === null ? undefined : steps[final.index];
    const rate = final !== null && at !== undefined && at > 0 ? final.value / at : null;
    const spark = windowRates(steps, lane.values);
    const death = lastFinite(divergence.get(lane.seed) ?? []);
    rows.push({
      seed: lane.seed,
      rate,
      retention: retentionOf(spark),
      divergedAt: death === null ? null : death.value,
      spark,
      recorded: recorded.has(lane.seed),
    });
  }
  rows.sort((a, b) => a.seed - b.seed);
  return rows;
}

/**
 * The sort as an INDEX PERMUTATION over the row array.
 *
 * Rows are built once and never reordered; sorting produces the order to paint
 * them in. That is what lets a re-sort animate (the same node moves) and what
 * guarantees a ghost's tint — which is keyed on the row's own identity — cannot
 * change just because the reader asked for a different ranking.
 *
 * Stable: equal keys keep seed order, and nulls always sink.
 */
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

/** The leader: best final average reward rate among seeds that can actually play. */
export function leaderSeed(rows: readonly SeedRow[]): number | null {
  let best: SeedRow | null = null;
  for (const row of rows) {
    if (!row.recorded || row.rate === null || !Number.isFinite(row.rate)) continue;
    if (best === null || row.rate > (best.rate ?? -Infinity)) best = row;
  }
  return best?.seed ?? null;
}

// ---------------------------------------------------------------------------
// the population header
// ---------------------------------------------------------------------------

export interface PopulationFigures {
  iqm: string;
  /** bracketed CI, or null when the summary carries none */
  ci: string | null;
  positive: string;
  nonDiverged: string;
  retention: string;
  /** where the numbers came from — the UI says so rather than implying a summary */
  source: 'summary' | 'measured' | 'none';
}

function finite(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** Interquartile mean — the lab's headline statistic, over whatever rates exist. */
export function iqmOf(values: readonly number[]): number | null {
  const sorted = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const lo = Math.floor(sorted.length * 0.25);
  const hi = Math.ceil(sorted.length * 0.75);
  const slice = sorted.slice(lo, Math.max(hi, lo + 1));
  if (slice.length === 0) return null;
  let sum = 0;
  for (const value of slice) sum += value;
  return sum / slice.length;
}

/**
 * The one line above the rail, in RL-paper words:
 *
 *   avg reward rate (IQM) +0.323 [+0.18, +0.44] · 100% seeds > 0 · 8/8 non-diverged · retention 62.5%
 *
 * summary.json's per-env slice is preferred because it is what `fleet.py report`
 * published and what a gate would read. When a root has none (the ~600 legacy
 * ones), the same four figures are MEASURED from fleet.jsonl and the caller is
 * told which it got — the two are not interchangeable claims.
 */
export function populationFigures(
  slice: SummaryPolicyResult | null,
  rows: readonly SeedRow[],
): PopulationFigures {
  const rates = rows.map((row) => row.rate).filter((v): v is number => v !== null && Number.isFinite(v));
  const diverged = rows.filter((row) => row.divergedAt !== null).length;
  const retained = rows.filter((row) => row.retention !== null && row.retention >= 0.5).length;

  const summaryIqm = finite(slice?.rewardRateIQM);
  if (slice !== null && summaryIqm !== null) {
    const ci = slice.rewardRateIQMCI95;
    const lo = Array.isArray(ci) ? finite(ci[0]) : null;
    const hi = Array.isArray(ci) ? finite(ci[1]) : null;
    const seeds = finite(slice.seeds) ?? rows.length;
    const dead = finite(slice.divergence?.diverged) ?? diverged;
    const positive = finite(slice.positiveFraction);
    const retention = finite(slice.retentionFraction);
    return {
      iqm: formatSigned(summaryIqm, 4),
      ci: lo !== null && hi !== null ? `[${formatSigned(lo, 3)}, ${formatSigned(hi, 3)}]` : null,
      positive: positive === null ? '—' : formatPercent(positive, 0),
      nonDiverged: `${formatCount(seeds - dead)}/${formatCount(seeds)}`,
      retention: retention === null ? '—' : formatPercent(retention, 0),
      source: 'summary',
    };
  }

  if (rows.length === 0) {
    return { iqm: '—', ci: null, positive: '—', nonDiverged: '—', retention: '—', source: 'none' };
  }
  const measured = iqmOf(rates);
  return {
    iqm: measured === null ? '—' : formatSigned(measured, 4),
    ci: null,
    positive: rates.length === 0
      ? '—'
      : formatPercent(rates.filter((v) => v > 0).length / rates.length, 0),
    nonDiverged: `${formatCount(rows.length - diverged)}/${formatCount(rows.length)}`,
    retention: formatPercent(retained / rows.length, 0),
    source: 'measured',
  };
}

/** The run's own identity line: `16 tasks × 64 seeds × 1M steps`. */
export function runScheduleLine(meta: Pick<RunMeta, 'envId' | 'batch' | 'steps' | 'envSchedule'>): string {
  const shares = envShares(meta);
  const horizon = `${formatStep(meta.steps)} steps`;
  if (shares.length <= 1) {
    return `${envName(meta.envId)} · ${formatCount(meta.batch)} seeds · ${horizon}`;
  }
  const counts = new Set(shares.map((share) => share.lanes));
  const perEnv = counts.size === 1 ? shares[0]?.lanes ?? meta.batch : null;
  const seeds = perEnv === null ? `${formatCount(meta.batch)} seeds total` : `${formatCount(perEnv)} seeds`;
  return `${shares.length} tasks × ${seeds} × ${horizon}`;
}

// ---------------------------------------------------------------------------
// keyboard
// ---------------------------------------------------------------------------

export type WallAction =
  | { kind: 'toggle' }
  | { kind: 'step'; frames: number }
  | { kind: 'env'; direction: 1 | -1 }
  | { kind: 'sort' }
  | { kind: 'debug' }
  | { kind: 'escape' };

/** space play/pause · arrows scrub · `[` `]` environment · `s` sort · `g` debug · Esc leader. */
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
    case 'Escape':
      return { kind: 'escape' };
    default:
      return null;
  }
}

/** Cycle the env chip strip, wrapping. */
export function cycleEnv(list: readonly EnvId[], current: EnvId | null, direction: 1 | -1): EnvId | null {
  if (list.length === 0) return null;
  const index = current === null ? -1 : list.indexOf(current);
  const next = index < 0 ? (direction > 0 ? 0 : list.length - 1) : (index + direction + list.length) % list.length;
  return list[next] ?? null;
}

/** Environments of a run, canonical order, recorded ones first-class. */
export function wallEnvs(meta: Pick<RunMeta, 'envId' | 'batch' | 'envSchedule'>, recordings: readonly RecordingRef[]): EnvId[] {
  const seen = new Set<EnvId>();
  for (const share of envShares(meta)) seen.add(share.envId);
  for (const rec of recordings) if (isEnvId(rec.envId)) seen.add(rec.envId);
  return ENV_IDS.filter((id) => seen.has(id));
}

// ---------------------------------------------------------------------------
// ghosts — the population layer
// ---------------------------------------------------------------------------
//
// `api.fetchWall` brings one env's whole population down as strided pose samples
// (`WallLane`), and `kit/ghosts` draws them as instanced ghosts inside the hero's
// own scene. This file owns exactly two things in between: WHICH sample the wall
// clock is on, and WHAT each ghost's colour and emphasis should be.

/** Ghosts the layer is built for. More than this and the swarm is noise, not data. */
export const WALL_GHOST_CAP = 512;

/** Environments whose ghost samples stay in memory (see `loadWall`). */
export const WALL_CACHE_ENVS = 3;

/**
 * Which strided sample a record-frame position lands on.
 *
 * The server samples rows `0, stride, 2·stride, …` plus the lane's final row, so
 * the last sample is reachable from any position past it and a fractional frame
 * position floors rather than rounds — a ghost is never shown a state the seed
 * has not reached yet.
 */
export function ghostSampleIndex(position: number, stride: number, samples: number): number {
  if (samples <= 0) return 0;
  if (!Number.isFinite(position) || position <= 0) return 0;
  const index = Math.floor(position / Math.max(1, stride));
  return index >= samples ? samples - 1 : index;
}

/**
 * Is this seed still finite at this record row?
 *
 * `firstNonFiniteRow` is the wall's death certificate. Past it the layer freezes
 * the ghost at its last finite pose and turns it dark red — so a numerically
 * diverged seed stops moving ON THE STEP IT DIED, in the same frame the rail row
 * says `div`. Before that row it is alive even if it dies later.
 */
export function ghostAlive(firstNonFiniteRow: number | null, row: number): boolean {
  return firstNonFiniteRow === null || row < firstNonFiniteRow;
}

/**
 * Each seed's POPULATION QUANTILE of average reward rate, in [0, 1].
 *
 * The ghost ramp is a quantile ramp (kit/ghosts), so it is fed a rank, never a
 * raw rate: one rocket seed at 40× the median would otherwise flatten every other
 * ghost onto the bottom colour, and the wall would show one bright dot over a
 * uniform swarm instead of the distribution. Seeds with no rate get 0 and stay
 * at the dark end. Ties share the lower rank, so an all-equal population is all
 * the same colour rather than an arbitrary gradient.
 */
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

/** The ghost bank the wall drives: one entry per lane, in instance order. */
interface GhostView {
  stride: number;
  entries: { lane: WallLane; seed: number }[];
}

// ---------------------------------------------------------------------------
// the field — the population's own extent, because the stage cannot hold it
// ---------------------------------------------------------------------------
//
// THE GEOMETRY PROBLEM, stated plainly, because it is the wall's hardest one.
//
// `kit/ghosts` places a cheetah ghost at its TRUE rootx on the hero's own track,
// and the savanna camera tracks the HERO's rootx (scenes/half-cheetah:
// `rig.setTarget(rootX, …)`), so hero and ghosts share one coordinate frame and a
// ghost is on stage exactly when it is within the camera's own window of the hero.
// That window is about five metres wide. The population's is not: measured on
// `frontend-demo-16env-20260803` (HalfCheetah, 8 seeds, `pose[0] = rootx`), the
// spread across seeds is 0.18 units at step 0, 0.84 at 5k, 10.09 at 25k, 221.56 at
// 50k and 1747.65 at 100k. Seeds that learn different gaits end up KILOMETRES
// apart, and no framing puts a 1.7 km line of one-metre animals on one screen:
// fit the camera to the extent and every seed is a third of a pixel.
//
// So the wall does two honest things instead of one dishonest one:
//
//   * it COUNTS. `inFrameCount` says how many ghosts are actually inside the
//     stage's window this sample, and the transport chip prints that number. It
//     never claims seven bodies are on stage when one is.
//   * it SHOWS THE FIELD. The strip under the stage is the whole population's
//     position along the track, drawn to a scale that fits — every seed a mark,
//     the hero marked apart, and the camera's own window drawn at the SAME scale
//     so the sliver the stage can hold is visible next to the kilometre it cannot.
//     Marks are live, hover-linked to the rail, and click to promote.
//
// Grid and classic scenes frame a BOUNDED world, so their whole population is on
// stage by construction and the strip is not drawn for them.

/** How a scene frames its world: one unbounded track, or a bounded world seen whole. */
export type StageFraming = 'track' | 'world';

export function stageFraming(category: EnvCategory): StageFraming {
  return category === 'cheetah' ? 'track' : 'world';
}

/**
 * The savanna camera, in the two numbers this file needs — mirrored from
 * `scenes/half-cheetah` (`CAM_OFFSET[2]`, `CAM_LOOK[0]`) rather than imported,
 * because reaching into a scene module would pull its whole Three.js world into
 * the wall's bundle to read two constants. `test/wall.test.ts` pins them.
 */
export const CHEETAH_CAM_DEPTH = 3.5;
export const CHEETAH_CAM_LOOK_X = 0.68;

/** The stage's visible window along the track, as distance behind and ahead of the hero. */
export interface TrackWindow {
  back: number;
  ahead: number;
  /** total width, in world units — what the chip and the field strip quote */
  width: number;
}

/**
 * How much track the hero stage actually shows, in world units.
 *
 * Derived, not guessed: the camera sits `CHEETAH_CAM_DEPTH` off the animal's plane
 * with a vertical field of view of `TOKENS.fovHero`, so half the frame height at
 * that plane is `depth · tan(fov/2)` and half its width is that times the stage's
 * aspect ratio; looking `CHEETAH_CAM_LOOK_X` ahead of the runner shifts the whole
 * window forward, which is why `back` and `ahead` differ.
 */
export function trackWindow(aspect: number): TrackWindow {
  const a = Number.isFinite(aspect) && aspect > 0 ? aspect : 16 / 9;
  const half = CHEETAH_CAM_DEPTH * Math.tan((TOKENS.fovHero * Math.PI) / 360) * a;
  // `back` is floored at 0 because the rig is CENTRED on the hero: whatever the
  // aspect, the animal the camera is tracking is in its own frame, so a window that
  // excluded dx = 0 would be arithmetic overruling a fact.
  return {
    back: Math.max(0, half - CHEETAH_CAM_LOOK_X),
    ahead: half + CHEETAH_CAM_LOOK_X,
    width: 2 * half,
  };
}

/** A lane's position along the track at one sample (`pose[0]`), or null when it has none. */
export function laneTrackX(lane: Pick<WallLane, 'pose' | 'samples'>, sample: number): number | null {
  if (lane.samples <= 0) return null;
  const at = sample < 0 ? 0 : sample >= lane.samples ? lane.samples - 1 : sample;
  const x = lane.pose[at * 9];
  return x === undefined || !Number.isFinite(x) ? null : x;
}

/** How many of `values` are inside the stage's window around the hero. Non-finite entries never count. */
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

/** Min/max of the finite track positions, or null when nothing is placeable. */
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

/**
 * The strip's domain: the population's extent UNION the camera's own window.
 *
 * The union is what makes the window drawable to scale in both regimes. Early in a
 * run the whole fleet fits inside one frame, so the window is the wider of the two
 * and fills the strip — which is the true statement "everything is on stage". Late
 * in a run the extent is kilometres and the window is a hairline at the hero's
 * mark — also true, and the reason only one animal is visible. A small pad keeps
 * the end marks off the edges.
 */
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

/** Where a track position sits on the strip, in percent. Clamped, so nothing escapes it. */
export function fieldPercent(x: number, scale: { min: number; max: number }): number {
  const span = scale.max - scale.min;
  if (!Number.isFinite(x) || !(span > 0)) return 50;
  const t = ((x - scale.min) / span) * 100;
  return t < 0 ? 0 : t > 100 ? 100 : t;
}

/** Distances the wall quotes are metres of track — `1.7 km`, `221 m`, `0.84 m`. */
export function formatDistance(metres: number): string {
  if (!Number.isFinite(metres)) return '—';
  const m = Math.abs(metres);
  if (m >= 1000) return `${(metres / 1000).toFixed(m >= 10_000 ? 0 : 1)} km`;
  if (m >= 10) return `${Math.round(metres)} m`;
  if (m >= 1) return `${metres.toFixed(1)} m`;
  return `${metres.toFixed(2)} m`;
}

// ---------------------------------------------------------------------------
// the view
// ---------------------------------------------------------------------------

const HERO_TIER = 'medium' as const;

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

  const { root, body } = viewShell('wall');
  root.classList.add('wall-root');
  host.append(root);

  let destroyed = false;
  const abort = new AbortController();

  // ---- view state ----
  let detail: RunDetail | null = null;
  let envList: EnvId[] = [];
  let envId: EnvId | null = query.env ?? null;
  let sort: WallSort = query.sort ?? 'rate';
  let speed: number = query.speed ?? 64;
  let selection: HeroSelection = { leader: null, promoted: query.seed ?? null };
  let rows: SeedRow[] = [];
  let policy = 'oak';
  let stride = 1;
  let heroSteps = 0;
  let frameCount = 1;
  let progress = 0; // 0..1 across the lane — what survives an env switch
  /**
   * The wall's position in RECORD FRAMES, which is what `?at=` carries.
   *
   * `progress` is a FRACTION and is only ever written by `onFrame`, so a `syncUrl`
   * that runs right after a seek — a deep link landing, an arrow key, a step
   * button — would serialize the position the reader just left. The clock is the
   * truth whenever a host is mounted; this is the truth before one is (a
   * deep-linked `at=` that has not been consumed yet), so between them `syncUrl`
   * always has the real answer.
   */
  let frameAt = query.at ?? 0;
  let scrubbing = false;
  /**
   * Primed past the throttle so the FIRST presented frame writes the readout.
   * A 150 ms cadence is right for a live page and wrong for a still: a headless
   * capture ticks rAF a couple of times, and every review shipped with an empty
   * readout when the gate had to warm up first (the artisan page's own lesson).
   */
  let statsAccum = 1000;
  let debugOn = false;

  let sceneHost: SceneHostHandle | null = null;
  let ghostView: GhostView | null = null;
  let ghostLayer: GhostLayer | null = null;
  /** the layer THIS view built (the host owns only the ones it rebuilds) */
  let createdGhostLayer: GhostLayer | null = null;
  let ghostSample = -1;
  let emphasised: number | null = null;
  let heroGeneration = 0;
  /** the hero's own decoded frames — the exact per-frame track position the stage draws */
  let heroSource: HarnessSource | null = null;
  /** every lane of this env (hero included), in field-mark order */
  let fieldLanes: WallLane[] = [];
  /** their track positions at the current sample; index-parallel to `fieldLanes` */
  let fieldX = new Float64Array(0);
  let fieldMarks: HTMLElement[] = [];
  let framing: StageFraming = 'world';
  let stageWindow: TrackWindow = trackWindow(16 / 9);
  /** ghosts inside the stage's own window at the current sample — the chip's number */
  let ghostsInFrame = 0;
  /** the sample the field is currently drawn at; -1 = never drawn */
  let fieldSample = -1;
  const ghostPose = new Float32Array(9);
  const sparks: SparklineHandle[] = [];
  const columnarCache = new Map<string, Promise<FleetColumnar>>();
  const wallCache = new Map<string, Promise<WallData>>();

  // ---- chrome ----
  const runName = el('a', { class: 'wall-run', href: '#/', text: '—' });
  const runMeta = el('span', { class: 'wall-run-meta', text: '' });
  const runBadges = el('span', { class: 'wall-run-badges' });
  const head = el('div', { class: 'wall-head' }, [
    el('div', { class: 'wall-head-line' }, [runName, runBadges]),
    runMeta,
  ]);

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
  const coverTitle = el('div', { class: 'cover-title', text: 'Loading' });
  const coverLabel = el('div', { class: 'cover-label', text: '' });
  const coverFill = el('div', { class: 'cover-fill' });
  const coverBar = el('div', { class: 'cover-bar' }, [coverFill]);
  const coverExtra = el('div', { class: 'cover-extra' });
  const cover = el('div', { class: 'wall-cover' }, [coverTitle, coverBar, coverLabel, coverExtra]);
  stage.append(stageTag, cover);

  // ---- the field strip ----
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

  // ---- transport ----
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
  const scrubHead = el('div', { class: 'wall-scrub-head' });
  const scrubTicks = el('div', { class: 'wall-scrub-ticks' });
  const scrub = el('div', { class: 'wall-scrub' }, [
    scrubInput,
    el('div', { class: 'wall-scrub-track' }),
    scrubFill,
    scrubTicks,
    scrubHead,
  ]);
  const readout = el('div', { class: 'wall-readout', text: 'step — / —' });
  const speedRow = el('div', { class: 'wall-speeds' });
  const ghostChip = el('span', { class: 'wall-ghosts', text: 'ghosts —' });
  const debugButton = el('button', { class: 'tbtn', type: 'button', text: 'g', title: 'Ground-truth overlay (g)' });
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
    debugButton,
  ]);

  // ---- rail ----
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
  body.append(head, chipStrip, envSelect, status);

  const main = el('div', { class: 'wall-main' }, [
    el('div', { class: 'wall-stage-col' }, [stage, field, transport]),
    rail,
  ]);
  main.hidden = true;
  body.append(main);

  // ---- URL ----
  /**
   * Rewrite the address bar from live state.
   *
   * `at` is read from the CLOCK, never from the cached `progress` fraction: every
   * seek (a keypress, a step button, a tick click, the scrub's release) calls this
   * synchronously, before any frame has ticked, so a cached fraction serializes the
   * position the reader just left. Before a host exists the pending `frameAt` — the
   * deep-linked `at=` — stands in, which is what stops a shared link from erasing
   * its own position the moment it is opened.
   */
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

  // ---- transport wiring ----
  function syncPlay(): void {
    playButton.textContent = sceneHost?.clock.playing ? 'pause' : 'play';
  }

  function applySpeed(): void {
    // ONE speed model: the preset is a real-time multiple of the env's own clock —
    // exactly what the playback clock means by `speed`, everywhere in the app.
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
    sceneHost?.seek(clampPosition(Number(scrubInput.value), frameCount));
  });

  function stepFrames(frames: number): void {
    if (!sceneHost) return;
    sceneHost.pause();
    sceneHost.seek(stepPosition(sceneHost.clock.position(), frames, frameCount));
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
      case 'escape': {
        const next = escapeSelection(selection);
        if (next !== selection) {
          selection = next;
          paintRail();
          void mountHero();
          syncUrl();
        }
        break;
      }
    }
  };
  if (typeof window !== 'undefined') window.addEventListener('keydown', onKey);

  // ---- painting ----
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
            el('span', { class: 'wall-chip-dot' }),
            el('span', { class: 'wall-chip-name', text: info.name }),
            el('span', { class: 'wall-chip-count', text: count > 0 ? String(count) : '·' }),
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
    const slice = envId === null ? null : envSliceResult(detail?.summary ?? null, envId, policy);
    const figures = populationFigures(slice, rows);
    popLine.replaceChildren(
      figure('avg reward rate (IQM)', figures.ci === null ? figures.iqm : `${figures.iqm} ${figures.ci}`,
        'Interquartile mean of the average reward rate (reward ÷ step) over every seed of this environment'),
      figure('seeds > 0', figures.positive, 'Seeds whose average reward rate is above zero'),
      figure('non-diverged', figures.nonDiverged, 'Seeds whose stream never went non-finite'),
      figure('retention', figures.retention, 'Seeds still holding at least half their peak reward rate'),
    );
    popNote.textContent = figures.source === 'summary'
      ? 'from summary.json — tools/fleet.py report'
      : figures.source === 'measured'
        ? 'measured from fleet.jsonl — this root has no summary slice for this environment'
        : '';
  }

  function paintRail(): void {
    for (const spark of sparks) spark.destroy();
    sparks.length = 0;
    sortButton.textContent = `sort ${SORT_LABEL[sort]}`;
    railCount.textContent = rows.length > 0 ? `${formatCount(rows.length)}` : '';

    if (rows.length === 0) {
      railRows.replaceChildren(
        el('div', { class: 'wall-row-empty', text: 'No seeds for this environment in fleet.jsonl.' }),
      );
      return;
    }

    const order = sortPermutation(rows, sort);
    const hero = heroSeed(selection);
    const withSparks = rows.length <= SPARK_ROW_BUDGET;
    const nodes: HTMLElement[] = [];

    order.forEach((index, rank) => {
      const row = rows[index];
      if (row === undefined) return;
      const isHero = row.seed === hero;
      const node = el('button', {
        class: `wall-row${isHero ? ' is-hero' : ''}${row.recorded ? '' : ' is-quiet'}${row.divergedAt !== null ? ' is-dead' : ''}`,
        type: 'button',
        title: row.recorded
          ? `Promote seed ${row.seed} to the hero scene (double-click opens the theater)`
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

      if (withSparks) {
        const holder = el('span', { class: 'wall-spark' });
        const spark = sparkline({
          values: row.spark,
          height: 16,
          stroke: row.divergedAt !== null ? '#e0614f' : isHero ? '#ffd166' : '#2b98d9',
          endDot: false,
          ariaLabel: `Reward-rate curve for seed ${row.seed}`,
        });
        holder.append(spark.root);
        sparks.push(spark);
        node.append(holder);
      }

      node.addEventListener('mouseenter', () => emphasise(row.seed));
      node.addEventListener('mouseleave', () => emphasise(null));
      node.addEventListener('focus', () => emphasise(row.seed));
      node.addEventListener('blur', () => emphasise(null));
      node.addEventListener('click', () => {
        const next = promoteSeed(selection, row.seed, playable);
        if (next === selection) return;
        selection = next;
        paintRail();
        void mountHero();
        syncUrl();
      });
      node.addEventListener('dblclick', () => {
        if (!detail || envId === null || !row.recorded) return;
        window.location.hash = theaterHref({ runId: detail.id, envId, seed: row.seed });
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

  /**
   * Light one seed's ghost. Emphasis 1 ("readable"), never 2 — level 2 is the
   * one gain allowed to rim past the bloom threshold, and a hover is not a
   * reason to bloom. The hero is not a ghost, so the brightest thing on the
   * stage stays the seed actually being replayed.
   */
  function ghostEmphasis(seed: number | null): void {
    const layer = sceneHost?.ghosts ?? null;
    if (!layer || !ghostView) return;
    if (emphasised === seed) return;
    emphasised = seed;
    ghostView.entries.forEach((entry, index) => {
      layer.setEmphasis(index, entry.seed === seed ? 1 : 0);
    });
  }

  /** One seed, lit in both halves of the population view: its ghost and its field mark. */
  function emphasise(seed: number | null): void {
    ghostEmphasis(seed);
    fieldEmphasis(seed);
  }

  /** Light one seed's mark on the field strip, to match its ghost. */
  function fieldEmphasis(seed: number | null): void {
    for (const mark of fieldMarks) {
      mark.classList.toggle('is-lit', seed !== null && Number(mark.dataset['seed']) === seed);
    }
  }

  /** Samples the field can address — the longest lane's, so nothing is cut short. */
  function fieldSamples(): number {
    let most = 0;
    for (const lane of fieldLanes) if (lane.samples > most) most = lane.samples;
    return most;
  }

  /** One mark per seed, built once per hero mount; only their positions move after that. */
  function buildField(lanes: readonly WallLane[], hero: number | null): void {
    fieldLanes = [...lanes].sort((a, b) => a.seed - b.seed);
    fieldX = new Float64Array(fieldLanes.length);
    fieldMarks = fieldLanes.map((lane) => {
      const isHero = lane.seed === hero;
      const mark = el('button', {
        class: `wall-field-mark${isHero ? ' is-hero' : ''}${lane.firstNonFiniteRow !== null ? ' is-dead' : ''}`,
        type: 'button',
        dataset: { seed: String(lane.seed) },
        ariaLabel: `Seed ${lane.seed}`,
      });
      mark.addEventListener('mouseenter', () => {
        const x = fieldX[fieldLanes.indexOf(lane)];
        mark.title = `Seed ${lane.seed}${isHero ? ' — the hero' : ''}`
          + (x !== undefined && Number.isFinite(x) ? ` · ${formatDistance(x)} down the track` : '');
        emphasise(lane.seed);
      });
      mark.addEventListener('mouseleave', () => emphasise(null));
      mark.addEventListener('click', () => {
        const next = promoteSeed(selection, lane.seed, playable);
        if (next === selection) return;
        selection = next;
        paintRail();
        void mountHero();
        syncUrl();
      });
      return mark;
    });
    fieldMarkHolder.replaceChildren(...fieldMarks);
    // One seed is not a field. A track env with a single recorded lane has nothing to
    // spread, and a strip with one mark on it would imply a population that is not there.
    field.hidden = framing !== 'track' || fieldLanes.length < 2;
    fieldSample = -1;
  }

  /**
   * Redraw the field for one strided sample: every seed's true position along the
   * track, the hero apart, and the stage's own window at the same scale.
   *
   * `heroX` is the hero's EXACT per-frame position (its decoded frame), not its
   * strided sample — the stage draws the hero every frame and the ghosts only every
   * `stride`, and at stride 43 a cheetah covers most of a frame width between
   * samples, so using the sample here would make the window jitter around the animal
   * it is supposed to be centred on.
   */
  function paintField(sample: number, heroX: number | null): void {
    if (field.hidden || fieldLanes.length === 0) return;
    const aspect = stage.clientHeight > 0 ? stage.clientWidth / stage.clientHeight : 16 / 9;
    stageWindow = trackWindow(aspect);

    let heroIndex = -1;
    const hero = heroSeed(selection);
    for (let i = 0; i < fieldLanes.length; i++) {
      const lane = fieldLanes[i];
      if (lane === undefined) continue;
      if (lane.seed === hero) heroIndex = i;
      const x = laneTrackX(lane, sample);
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

    // The chip counts GHOSTS, and `fieldX` also carries the hero — who is at dx = 0 and
    // therefore always inside the window the rig centres on him. Discount exactly that.
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

  /** The hero's exact track position at a decoded frame index. */
  function heroTrackX(index: number): number | null {
    if (heroSource === null || heroSource.frameCount === 0) return null;
    const frame = heroSource.frameAt(Math.min(index, heroSource.frameCount - 1));
    const x = frame.pose[0];
    return x === undefined || !Number.isFinite(x) ? null : x;
  }

  /** Write every ghost's pose for one strided sample. Allocation-free. */
  function paintGhosts(layer: GhostLayer, view: GhostView, sample: number): void {
    for (let index = 0; index < view.entries.length; index++) {
      const entry = view.entries[index];
      if (entry === undefined) continue;
      const lane = entry.lane;
      if (lane.samples === 0) continue;
      const at = sample >= lane.samples ? lane.samples - 1 : sample;
      const offset = at * 9;
      for (let k = 0; k < 9; k++) ghostPose[k] = lane.pose[offset + k] ?? 0;
      const row = lane.rows[at] ?? at * view.stride;
      layer.setGhostFrame(index, ghostPose, ghostAlive(lane.firstNonFiniteRow, row));
    }
  }

  function playable(seed: number): boolean {
    if (!detail || envId === null) return false;
    return recordingRef(detail.recordings, envId, seed) !== null;
  }

  /**
   * The landmark strip under the scrub bar, drawn as a DENSITY.
   *
   * A seed that reaches its goal every few hundred steps fills all 180 buckets,
   * and 180 identical full-height ticks is a gold barcode that says only "there
   * were events". Opacity carries the count instead, so the strip shows WHERE
   * the seed was productive — the same data, in the shape a reader can use.
   */
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
          sceneHost.seek(seekTargetForTick(tick));
          sceneHost.play();
          syncPlay();
        });
        return node;
      }),
    );
  }

  function onFrame(info: HostFrameInfo): void {
    progress = frameCount > 1 ? info.position / (frameCount - 1) : 0;
    frameAt = info.position;
    const percent = (progress * 100).toFixed(3);
    scrubFill.style.width = `${percent}%`;
    scrubHead.style.left = `${percent}%`;
    if (!scrubbing) scrubInput.value = String(info.position);

    // The ghosts move on SAMPLE changes, not display frames: at ×1 that is 12
    // writes a second instead of 60, and a tier rebuild (which hands back a new
    // layer) re-primes them by identity rather than by a flag.
    const layer = sceneHost?.ghosts ?? null;
    if (layer && ghostView) {
      const sample = ghostSampleIndex(info.position, ghostView.stride, ghostSamples(ghostView));
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
    // The field is driven off the wall's own stride rather than off the ghost layer,
    // so an environment with too few recorded seeds to ghost still animates its strip.
    if (!field.hidden && fieldLanes.length > 0) {
      const sample = ghostSampleIndex(info.position, stride, fieldSamples());
      if (sample !== fieldSample) {
        fieldSample = sample;
        paintField(sample, heroTrackX(info.index));
      }
    }

    statsAccum += info.frameMs;
    if (statsAccum < 150) return;
    statsAccum = 0;
    // frames[0] is sim-core's SYNTHESIZED init state and carries step -1; naming
    // it beats printing a negative step on the front page of the lab.
    readout.textContent =
      `step ${info.step < 0 ? 'init' : formatCount(info.step)} / ${formatStep(heroSteps)}`
      + ` · ${formatClock(info.simTime)}`;
    if (ghostView && info.ghostDraws > 0) ghostChip.textContent = ghostChipText(ghostView);
    syncPlay();
  }

  /**
   * What the ghost chip is allowed to say.
   *
   * The draw count is NOT the claim a reader hears — instanced draws are submitted
   * whether or not anything they describe is inside the frustum, and on a cheetah
   * wall past ~25k steps the seeds are metres to kilometres apart, so "7 ghosts ·
   * 8 draws" over a stage holding one animal was a true sentence that read as a
   * false one. The number printed is therefore how many ghosts are inside the
   * stage's own window RIGHT NOW, measured from the same positions they are drawn
   * at (`paintField`), with the field strip underneath carrying the rest.
   */
  function ghostChipText(view: GhostView): string {
    const total = view.entries.length;
    if (framing !== 'track') return `ghosts ${formatCount(total)} · all in frame`;
    const inFrame = ghostsInFrame >= total ? 'all' : formatCount(ghostsInFrame);
    return `ghosts ${formatCount(total)} · ${inFrame} in frame`;
  }

  // ---- data ----
  function loadColumnar(id: string, env: EnvId): Promise<FleetColumnar> {
    const key = `${id}|${env}|${policy}`;
    const cached = columnarCache.get(key);
    if (cached) return cached;
    const promise = api.fleet(
      id,
      { fields: ['rewardSum', 'firstNonFiniteStep'], envId: env, policy },
      { signal: abort.signal },
    ).catch((error: unknown) => {
      columnarCache.delete(key);
      throw error;
    });
    columnarCache.set(key, promise);
    return promise;
  }

  /**
   * One env's whole population, strided, cached per (run, env, stride).
   *
   * A promotion re-mounts the hero but must NOT re-download the population: the
   * ghosts do not depend on which seed is on stage.
   */
  function loadWall(id: string, env: EnvId, at: number): Promise<WallData> {
    const key = `${id}|${env}|${at}`;
    const cached = wallCache.get(key);
    if (cached) {
      // touch: Map keeps insertion order, so re-inserting makes this the newest
      wallCache.delete(key);
      wallCache.set(key, cached);
      return cached;
    }
    const promise = fetchWall(id, env, { stride: at }, { signal: abort.signal })
      .catch((error: unknown) => {
        wallCache.delete(key);
        throw error;
      });
    wallCache.set(key, promise);
    // A 64-seed env is ~5 MB of pose samples. Walking all 16 environments of a
    // mixed fleet would otherwise pin ~90 MB in a page nobody reloads, so the
    // cache keeps the few most recent and re-fetches the rest (the server caches
    // its own derivation, so a return visit is cheap).
    while (wallCache.size > WALL_CACHE_ENVS) {
      const oldest = wallCache.keys().next();
      if (oldest.done === true) break;
      wallCache.delete(oldest.value);
    }
    return promise;
  }

  /** Samples the wall can address — the longest lane's, so nothing is cut short. */
  function ghostSamples(view: GhostView): number {
    let most = 0;
    for (const entry of view.entries) if (entry.lane.samples > most) most = entry.lane.samples;
    return most;
  }

  /** Colour and emphasis, once per layer: the ramp is a rank, never a raw rate. */
  function primeGhosts(layer: GhostLayer, view: GhostView): void {
    const quantiles = rateQuantiles(rows);
    view.entries.forEach((entry, index) => {
      layer.setRate(index, quantiles.get(entry.seed) ?? 0);
      layer.setEmphasis(index, 0);
    });
  }

  function teardownStage(): void {
    sceneHost?.destroy();
    sceneHost = null;
    // The host disposes only layers IT built on a tier rebuild (`ghostOwned` in
    // sceneHost); the one handed in at mount is ours to the end.
    createdGhostLayer?.dispose();
    createdGhostLayer = null;
    ghostLayer = null;
    ghostView = null;
    ghostSample = -1;
    emphasised = null;
    heroSource = null;
    fieldLanes = [];
    fieldMarks = [];
    fieldMarkHolder.replaceChildren();
    ghostsInFrame = 0;
    fieldSample = -1;
    field.hidden = true;
  }

  /** Mount (or re-mount) the hero lane, preserving where the reader was. */
  async function mountHero(): Promise<void> {
    const generation = ++heroGeneration;
    const run = detail;
    const env = envId;
    teardownStage();
    scrubTicks.replaceChildren();
    if (!run || env === null) return;

    const seed = heroSeed(selection);
    const reason = degradedReason(run.recordings, env);
    if (reason !== null || seed === null) {
      showDegraded(reason ?? 'no seed to play');
      return;
    }

    cover.hidden = false;
    coverBar.hidden = false;
    coverLabel.hidden = false;
    coverExtra.replaceChildren();
    coverTitle.textContent = `Loading ${ENVS[env].name} seed ${seed}`;
    coverFill.style.width = '0%';

    const info = ENVS[env];
    const tagEnv = stageTag.querySelector('.wall-tag-env');
    const tagSub = stageTag.querySelector('.wall-tag-sub');
    if (tagEnv) tagEnv.textContent = info.name;
    if (tagSub) {
      tagSub.textContent = `seed ${seed}${selection.promoted === null ? ' · leader' : ' · promoted'}`;
    }

    // The stride is fixed BEFORE either request: it comes from the lane's declared
    // step count, so the hero fetch and the population fetch agree on one sampling
    // grid without waiting for each other, and both are in flight at once — the
    // population is not a second page load.
    const ref = recordingRef(run.recordings, env, seed);
    stride = wallStride(ref?.steps ?? detail?.meta.steps ?? 0);
    applySpeed();
    const wallPromise = loadWall(run.id, env, stride);
    ghostChip.textContent = 'ghosts …';

    let recording: TheaterRecording;
    try {
      recording = await loadTheaterRecording({
        runId: run.id,
        envId: env,
        seed,
        ...(ref ? { expectedSteps: ref.steps } : {}),
        signal: abort.signal,
        onProgress: (p) => {
          coverFill.style.width = `${(p.fraction * 100).toFixed(1)}%`;
          coverLabel.textContent = p.label;
        },
      });
    } catch (cause) {
      if (destroyed || generation !== heroGeneration) return;
      showStageError(cause instanceof Error ? cause.message : String(cause));
      return;
    }
    if (destroyed || generation !== heroGeneration) return;

    heroSteps = recording.steps;
    frameCount = recording.source.frameCount;
    heroSource = recording.source;
    framing = stageFraming(info.category);
    scrubInput.max = String(Math.max(1, frameCount - 1));
    paintTicks(recording.ticks);

    const at = query.at !== undefined && progress === 0
      ? clampPosition(query.at, frameCount)
      : progress * (frameCount - 1);
    frameAt = at;

    // ---- the population layer ----
    //
    // Built BEFORE the host, because the layer is a MOUNT option: `sceneHost`
    // attaches its group after the scene's `init()` and re-attaches it across
    // every tier rebuild, which is the only way a layer survives one. It is fed
    // the hero's own `geometry` — the same ground truth the hero is drawn from,
    // never a second derivation.
    coverLabel.textContent = 'reading the population';
    const wall = await wallPromise.catch(() => null);
    if (destroyed || generation !== heroGeneration) return;

    let ghosts: SceneHostGhosts | undefined;
    if (wall !== null) {
      const entries = wall.lanes
        .filter((lane) => lane.seed !== seed && lane.samples > 0 && lane.error === undefined)
        .slice(0, WALL_GHOST_CAP)
        .map((lane) => ({ lane, seed: lane.seed }));
      if (entries.length > 0) {
        // the server's own stride is the truth (it clamps); the tempo follows it
        // so one wall frame is exactly one ghost sample at ×1
        stride = wall.header.stride > 0 ? wall.header.stride : stride;
        ghostView = { stride, entries };
        const build = (tier: QualityTier): GhostLayer => makeGhostLayer({
          category: info.category,
          envId: env,
          geometry: recording.source.geometry,
          maxGhosts: entries.length,
          tokens: TOKENS,
          accent: accentCss(info.slug),
          tier,
        });
        createdGhostLayer = build(HERO_TIER);
        ghosts = { layer: createdGhostLayer, rebuild: build };
      }
      // The FIELD is the whole population — the hero included, and every seed the
      // ghost cap left out — because the strip's job is exactly the seeds the stage
      // cannot hold.
      buildField(wall.lanes.filter((lane) => lane.samples > 0 && lane.error === undefined), seed);
    }
    applySpeed();

    coverLabel.textContent = 'building the scene';
    try {
      sceneHost = await mountSceneHost(stage, {
        envId: env,
        source: recording.source,
        tier: HERO_TIER,
        at,
        speed,
        debug: debugOn,
        ...(ghosts ? { ghosts } : {}),
        onFrame,
        onReady: (manifest) => {
          if (tagEnv) tagEnv.textContent = manifest.title;
        },
      });
    } catch (cause) {
      if (destroyed || generation !== heroGeneration) return;
      showStageError(cause instanceof Error ? cause.message : String(cause));
      return;
    }
    if (destroyed || generation !== heroGeneration) {
      sceneHost?.destroy();
      sceneHost = null;
      return;
    }
    stage.insertBefore(sceneHost.canvas, stageTag);
    cover.hidden = true;
    syncPlay();
    statsAccum = 1000; // a hero swap reports its own numbers on its first frame

    if (recording.health.message) {
      coverExtra.replaceChildren();
      stageTag.title = recording.health.message;
    }

    // First paint before a frame is asked for, so a paused deep link still shows
    // the population rather than an empty stage waiting on rAF. The field goes first:
    // the chip's in-frame count is measured there, and it is drawn whether or not this
    // environment had enough seeds to build a ghost layer.
    fieldSample = ghostSampleIndex(at, stride, fieldSamples());
    paintField(fieldSample, heroTrackX(Math.round(at)));
    const layer = sceneHost.ghosts;
    if (layer && ghostView) {
      ghostLayer = layer;
      primeGhosts(layer, ghostView);
      ghostSample = ghostSampleIndex(at, ghostView.stride, ghostSamples(ghostView));
      paintGhosts(layer, ghostView, ghostSample);
      ghostChip.textContent = ghostChipText(ghostView);
      ghostChip.title =
        `${formatCount(ghostView.entries.length)} recorded seeds of ${info.name}, `
        + `one pose sample every ${formatCount(ghostView.stride)} steps, on the hero's own clock. `
        + (framing === 'track'
          ? `The stage frames about ${formatDistance(stageWindow.width)} of track around the hero, `
            + 'so a seed that has walked away from it is off camera — the field strip under the stage '
            + 'is where the whole population is. '
          : 'This scene frames its whole world, so every seed is inside the picture. ')
        + 'A seed that went non-finite freezes at its last finite pose and turns dark.';
    } else {
      ghostChip.textContent = 'ghosts —';
      ghostChip.title = wall === null
        ? 'The population endpoint did not answer for this environment; the hero is playing alone.'
        : 'This environment has no other recorded seed to ghost.';
    }
  }

  function showDegraded(reason: string): void {
    cover.hidden = false;
    coverBar.hidden = true;
    coverLabel.hidden = true;
    coverTitle.textContent = 'Nothing to replay here';
    coverExtra.replaceChildren(
      emptyState(
        reason,
        `The population on the right is real — it comes from fleet.jsonl. There is simply no recorded `
          + `experience on this run, so there is no scene and there are no ghosts. ${RECORD_RECIPE}`,
        [
          detail ? linkButton('Open the run dashboard', `#/run/${encodeURIComponent(detail.id)}`, { quiet: true }) : null,
          linkButton('Find a run with recordings', '#/runs', { quiet: true }),
        ],
      ),
    );
    ghostChip.textContent = 'ghosts —';
    readout.textContent = 'no recording';
    transport.classList.add('is-idle');
  }

  function showStageError(message: string): void {
    cover.hidden = false;
    coverBar.hidden = true;
    coverLabel.hidden = true;
    coverTitle.textContent = 'Could not play this seed';
    coverExtra.replaceChildren(errorState(message));
    transport.classList.add('is-idle');
  }

  /** Switch environment: new scene, new rail, same place in the run. */
  function selectEnv(next: EnvId): void {
    if (!detail) return;
    envId = next;
    transport.classList.remove('is-idle');
    paintChips();
    rows = [];
    railRows.replaceChildren(spinner('Reading fleet.jsonl'));
    paintPopulation();

    const run = detail;
    loadColumnar(run.id, next).then(
      (columnar) => {
        if (destroyed || envId !== next) return;
        rows = buildSeedRows(columnar, { recorded: new Set(seedsFor(run.recordings, next)) });
        selection = setLeader(selection, leaderSeed(rows), playable);
        paintPopulation();
        paintRail();
        void mountHero();
        syncUrl();
      },
      (error: unknown) => {
        if (destroyed || envId !== next) return;
        railRows.replaceChildren(errorState(error instanceof Error ? error.message : String(error)));
        // The stage does not depend on fleet.jsonl. Without it there is no ranking,
        // so the LEADER is unknowable — but the recordings still exist, and the
        // lowest recorded seed is an honest stand-in that keeps the wall playing
        // while the rail says what failed. The wall degrades one half at a time.
        const fallback = seedsFor(run.recordings, next)[0] ?? null;
        selection = setLeader(selection, fallback, playable);
        void mountHero();
      },
    );
  }

  function renderRun(next: RunDetail, pick: FeaturedPick): void {
    detail = next;
    policy = policiesIn(next.aggregate)[0] ?? next.meta.policies[0] ?? 'oak';
    envList = wallEnvs(next.meta, next.recordings);
    status.remove();
    main.hidden = false;

    runName.textContent = next.id;
    runName.href = `#/run/${encodeURIComponent(next.id)}`;
    runName.title = 'Open this run’s dashboard';
    runMeta.textContent = runScheduleLine(next.meta);
    const result = primaryResult(next.summary, policy);
    runBadges.replaceChildren(
      ...(pick.source === 'bestKernel' ? [badge('best kernel', 'keeper', 'The current promoted best kernel')] : []),
      ...(next.recordings.length === 0 ? [badge('no recordings', 'warn', 'Nothing on this run can be replayed')] : []),
      ...(result?.complete === false ? [badge('incomplete', 'warn', 'Did not reach its declared horizon')] : []),
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
          'meta.json carries neither an envSchedule nor a readable envId, so there is nothing to show.',
        ),
      );
      return;
    }
    selectEnv(first);
  }

  function start(force: boolean): void {
    Promise.all([runIndex.load(force), rolesResource.load(force).catch(() => null)]).then(
      ([entries, roles]) => {
        if (destroyed) return;
        const pick = resolveFeaturedRun(entries, roles, query.run);
        if (pick.runId === null) {
          status.replaceChildren(
            emptyState(
              'No run to show yet',
              'The wall plays a fleet’s recorded seeds and the server found no run with recordings under '
                + `.oak-benchmarks. ${RECORD_RECIPE}`,
              [linkButton('Browse every run', '#/runs', { quiet: true })],
            ),
          );
          return;
        }
        loadRunDetail(pick.runId).then(
          (loaded) => {
            if (destroyed) return;
            renderRun(loaded, pick);
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
      if (typeof window !== 'undefined') window.removeEventListener('keydown', onKey);
      for (const spark of sparks) spark.destroy();
      sparks.length = 0;
      teardownStage();
      root.remove();
    },
  };
  return handle;
};
