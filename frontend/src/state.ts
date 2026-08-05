// App data layer: a minimal store, TTL-cached resources, and the pure helpers
// (formatters, columnar reshaping, gallery filtering) the views build on.
// Zero DOM — everything here is unit-testable.

import { api } from './api';
import type {
  AggregateRow,
  FleetColumnar,
  RolesResponse,
  RunDetail,
  RunIndexEntry,
  RunMeta,
  SummaryPolicyResult,
} from '@shared/types';
import { ENVS, ENV_IDS, isEnvId, type EnvId } from '@shared/envs';

// ---------- store ----------

export interface Store<T> {
  get(): T;
  set(next: T | ((prev: T) => T)): void;
  subscribe(listener: (value: T) => void): () => void;
}

export function createStore<T>(initial: T): Store<T> {
  let value = initial;
  const listeners = new Set<(value: T) => void>();
  return {
    get: () => value,
    set(next) {
      const resolved = typeof next === 'function' ? (next as (prev: T) => T)(value) : next;
      if (Object.is(resolved, value)) return;
      value = resolved;
      for (const listener of [...listeners]) listener(value);
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

// ---------- TTL resource ----------

export type AsyncStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface AsyncState<T> {
  status: AsyncStatus;
  data: T | null;
  error: string | null;
  fetchedAt: number;
}

export interface Resource<T> {
  store: Store<AsyncState<T>>;
  /** returns cached data inside the TTL; refetches otherwise (or when forced) */
  load(force?: boolean): Promise<T>;
  invalidate(): void;
}

export function createResource<T>(
  loader: () => Promise<T>,
  ttlMs: number,
  now: () => number = Date.now,
): Resource<T> {
  const store = createStore<AsyncState<T>>({ status: 'idle', data: null, error: null, fetchedAt: 0 });
  let inflight: Promise<T> | null = null;

  return {
    store,
    load(force = false) {
      const state = store.get();
      if (!force && state.status === 'ready' && state.data !== null && now() - state.fetchedAt < ttlMs) {
        return Promise.resolve(state.data);
      }
      if (inflight) return inflight;
      store.set({ ...store.get(), status: 'loading', error: null });
      inflight = loader().then(
        (data) => {
          inflight = null;
          store.set({ status: 'ready', data, error: null, fetchedAt: now() });
          return data;
        },
        (error: unknown) => {
          inflight = null;
          const message = error instanceof Error ? error.message : String(error);
          store.set({ status: 'error', data: store.get().data, error: message, fetchedAt: now() });
          throw error;
        },
      );
      return inflight;
    },
    invalidate() {
      // back to idle, but keep the data so the UI has something to show
      store.set({ ...store.get(), status: 'idle', fetchedAt: 0 });
    },
  };
}

export const RUN_INDEX_TTL_MS = 60_000;

export const runIndex: Resource<RunIndexEntry[]> = createResource(() => api.runs(), RUN_INDEX_TTL_MS);
export const roles: Resource<RolesResponse> = createResource(() => api.roles(), RUN_INDEX_TTL_MS);

/** per-run detail cache — the gallery hydrates cards lazily, the run view reuses them */
const detailCache = new Map<string, Promise<RunDetail>>();

export function loadRunDetail(id: string, force = false): Promise<RunDetail> {
  if (force) detailCache.delete(id);
  const cached = detailCache.get(id);
  if (cached) return cached;
  const promise = api.run(id).catch((error: unknown) => {
    detailCache.delete(id);
    throw error;
  });
  detailCache.set(id, promise);
  return promise;
}

// ---------- formatters ----------

/**
 * Average reward rates span 1e-5 (MountainCar) to 1e0 (CartPole) in one fleet,
 * so a fixed decimal count is useless. Pick a representation by magnitude.
 */
export function formatRate(value: number | null | undefined, sig = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e5 || magnitude < 1e-4) return value.toExponential(1);
  const decimals = Math.max(0, sig - 1 - Math.floor(Math.log10(magnitude)));
  return value.toFixed(Math.min(decimals, 6));
}

export function formatSigned(value: number | null | undefined, sig = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const body = formatRate(value, sig);
  return value > 0 ? `+${body}` : body;
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return Math.round(value).toLocaleString('en-US');
}

/** 100000 -> "100k", 1000000 -> "1M", 1500000 -> "1.5M" */
export function formatStep(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${trimZero(value / 1e9)}B`;
  if (abs >= 1e6) return `${trimZero(value / 1e6)}M`;
  if (abs >= 1e3) return `${trimZero(value / 1e3)}k`;
  return String(Math.round(value));
}

function trimZero(value: number): string {
  const text = value.toFixed(value < 10 ? 1 : 0);
  return text.endsWith('.0') ? text.slice(0, -2) : text;
}

export function formatPercent(fraction: number | null | undefined, decimals = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return '—';
  return `${(fraction * 100).toFixed(decimals)}%`;
}

export function formatWall(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return '—';
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, '0')}s`;
  return `${s}s`;
}

export function formatAgo(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return 'undated';
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) return 'undated';
  const seconds = Math.max(0, (now - then) / 1000);
  if (seconds < 90) return 'just now';
  const minutes = seconds / 60;
  if (minutes < 90) return `${Math.round(minutes)}m ago`;
  const hours = minutes / 60;
  if (hours < 36) return `${Math.round(hours)}h ago`;
  const days = hours / 24;
  if (days < 30) return `${Math.round(days)}d ago`;
  const months = days / 30.44;
  if (months < 18) return `${Math.round(months)}mo ago`;
  return `${Math.round(days / 365.25)}y ago`;
}

// ---------- run/summary readers ----------

export function primaryResult(summary: RunDetail['summary'], policy?: string): SummaryPolicyResult | null {
  const results = summary?.results;
  if (!results || results.length === 0) return null;
  if (policy) {
    const match = results.find((r) => r.policy === policy);
    if (match) return match;
  }
  return results[0] ?? null;
}

export function envSliceResult(
  summary: RunDetail['summary'],
  envId: string,
  policy?: string,
): SummaryPolicyResult | null {
  const slice = summary?.envSlices?.[envId];
  if (!slice || slice.length === 0) return null;
  if (policy) {
    const match = slice.find((entry) => entry.policy === policy);
    if (match) return match;
  }
  return slice[0] ?? null;
}

/** Everything about a run that changes when the shown policy changes. */
export interface PolicyView {
  policy: string;
  result: SummaryPolicyResult | null;
  /** furthest heartbeat for THIS policy (summary first, aggregate as the fallback) */
  maxStep: number;
  /** did THIS policy reach the declared horizon */
  complete: boolean;
}

/**
 * Derive the policy-dependent block of a run.
 *
 * `summary.results` has one entry per policy and `aggregate.jsonl` one series per
 * policy, so every headline number — IQM, positivity, survival, best lane,
 * completeness — belongs to ONE policy. Capturing them once and then only
 * redrawing the charts on a policy switch leaves confidently mislabelled numbers
 * on screen, which is worse than showing none.
 */
export function policyView(
  detail: Pick<RunDetail, 'summary' | 'aggregate' | 'meta'>,
  policy: string,
): PolicyView {
  const result = primaryResult(detail.summary, policy);
  let policyStep = 0;
  let anyStep = 0;
  for (const row of detail.aggregate) {
    if (row.step > anyStep) anyStep = row.step;
    if (row.policy === policy && row.step > policyStep) policyStep = row.step;
  }
  const declared = typeof result?.['maxStep'] === 'number' && Number.isFinite(result['maxStep'])
    ? (result['maxStep'] as number)
    : null;
  const maxStep = declared ?? (policyStep > 0 ? policyStep : anyStep);
  const horizon = detail.meta.steps;
  return { policy, result, maxStep, complete: result?.complete !== false && maxStep >= horizon };
}

export interface EnvShare {
  envId: EnvId;
  lanes: number;
  fraction: number;
}

/**
 * Seed apportionment per env, in the canonical ENV_IDS order (so two runs with
 * the same envs always draw the same strip). Single-env runs report one share.
 */
export function envShares(meta: Pick<RunMeta, 'envId' | 'batch' | 'envSchedule'>): EnvShare[] {
  const counts = new Map<EnvId, number>();
  const schedule = meta.envSchedule;
  if (schedule && schedule.length > 0) {
    for (const entry of schedule) {
      if (!isEnvId(entry.envId)) continue;
      const lanes = Math.max(0, entry.seedEnd - entry.seedStart + 1);
      counts.set(entry.envId, (counts.get(entry.envId) ?? 0) + lanes);
    }
  } else if (isEnvId(meta.envId)) {
    counts.set(meta.envId, meta.batch);
  }
  const total = [...counts.values()].reduce((sum, n) => sum + n, 0) || 1;
  return ENV_IDS.filter((id) => counts.has(id)).map((envId) => {
    const lanes = counts.get(envId) ?? 0;
    return { envId, lanes, fraction: lanes / total };
  });
}

/**
 * The envs of an index entry, in canonical order. Lane apportionment is NOT
 * derivable here: the run index carries `envIds` and `batch`, never the per-env
 * split (that lives in meta.envSchedule, i.e. behind a detail fetch). Anything
 * that needs true counts calls `envShares(meta)`; anything that only needs the
 * identity strip uses this and shows no numbers.
 */
export function envIdsOf(entry: Pick<RunIndexEntry, 'envIds'>): EnvId[] {
  return ENV_IDS.filter((id) => entry.envIds.includes(id));
}

export function envName(envId: string): string {
  return isEnvId(envId) ? ENVS[envId].name : envId;
}

export function envSlug(envId: string): string {
  return isEnvId(envId) ? ENVS[envId].slug : envId.toLowerCase();
}

// ---------- gallery filtering ----------

export interface GalleryFilter {
  search: string;
  envId: string | null;
  completeOnly: boolean;
  recordingsOnly: boolean;
  inFlightOnly: boolean;
  rolesOnly: boolean;
}

export const EMPTY_FILTER: GalleryFilter = {
  search: '',
  envId: null,
  completeOnly: false,
  recordingsOnly: false,
  inFlightOnly: false,
  rolesOnly: false,
};

export function matchesFilter(entry: RunIndexEntry, filter: GalleryFilter): boolean {
  if (filter.completeOnly && !entry.complete) return false;
  if (filter.recordingsOnly && !entry.hasRecordings) return false;
  if (filter.inFlightOnly && !entry.inFlight) return false;
  if (filter.rolesOnly && entry.role === null) return false;
  if (filter.envId && !entry.envIds.includes(filter.envId)) return false;
  const search = filter.search.trim().toLowerCase();
  if (search) {
    const haystack = `${entry.id} ${entry.ablationLabel} ${entry.envIds.join(' ')} ${entry.role ?? ''}`.toLowerCase();
    for (const term of search.split(/\s+/)) {
      if (!haystack.includes(term)) return false;
    }
  }
  return true;
}

const ROLE_RANK: Record<string, number> = { bestKernel: 0, candidate: 1, parent: 2 };

/** roles first (best kernel, then candidate, then parent), then newest, then id. */
export function sortRuns(entries: readonly RunIndexEntry[]): RunIndexEntry[] {
  return [...entries].sort((a, b) => {
    if (a.inFlight !== b.inFlight) return a.inFlight ? -1 : 1;
    const ra = a.role ? ROLE_RANK[a.role] ?? 9 : 9;
    const rb = b.role ? ROLE_RANK[b.role] ?? 9 : 9;
    if (ra !== rb) return ra - rb;
    const ta = a.startedAt ? Date.parse(a.startedAt) : 0;
    const tb = b.startedAt ? Date.parse(b.startedAt) : 0;
    if (tb !== ta) return tb - ta;
    return a.id.localeCompare(b.id);
  });
}

export function envIdsInIndex(entries: readonly RunIndexEntry[]): EnvId[] {
  const seen = new Set<string>();
  for (const entry of entries) for (const id of entry.envIds) seen.add(id);
  return ENV_IDS.filter((id) => seen.has(id));
}

// ---------- aggregate reshaping ----------

export interface AggregateSeries {
  steps: number[];
  p10: number[];
  p50: number[];
  p90: number[];
  alive: number[];
  dead: number[];
  sps: number[];
  positivity: number[];
  bestLane: number[];
  wallSeconds: number[];
}

const EMPTY_AGGREGATE: AggregateSeries = {
  steps: [], p10: [], p50: [], p90: [], alive: [], dead: [], sps: [], positivity: [], bestLane: [], wallSeconds: [],
};

/** Deduplicates by step (a `final` row repeats the last heartbeat) and sorts ascending. */
export function aggregateSeries(rows: readonly AggregateRow[], policy?: string): AggregateSeries {
  const filtered = policy ? rows.filter((r) => r.policy === policy) : rows;
  if (filtered.length === 0) return { ...EMPTY_AGGREGATE };
  const byStep = new Map<number, AggregateRow>();
  for (const row of filtered) {
    const existing = byStep.get(row.step);
    // a `final` row wins over the `hb` row at the same step
    if (!existing || row.kind === 'final') byStep.set(row.step, row);
  }
  const ordered = [...byStep.values()].sort((a, b) => a.step - b.step);
  const out: AggregateSeries = {
    steps: [], p10: [], p50: [], p90: [], alive: [], dead: [], sps: [], positivity: [], bestLane: [], wallSeconds: [],
  };
  for (const row of ordered) {
    out.steps.push(row.step);
    out.p10.push(row.deliveredRateP10);
    out.p50.push(row.deliveredRateP50);
    out.p90.push(row.deliveredRateP90);
    out.alive.push(row.alive);
    out.dead.push(row.dead);
    out.sps.push(row.aggStepsPerSecond);
    out.positivity.push(row.positiveRateFraction);
    out.bestLane.push(row.bestLaneDeliveredRate);
    out.wallSeconds.push(row.wallSeconds);
  }
  return out;
}

export function policiesIn(rows: readonly AggregateRow[]): string[] {
  const seen = new Set<string>();
  for (const row of rows) seen.add(row.policy);
  return [...seen].sort();
}

/**
 * Dedup key for a LIVE aggregate buffer. A heartbeat is unique per (policy, step),
 * never per step: a run with oak + zero + random emits three rows at every step,
 * so a step-keyed buffer keeps whichever arrived last and the resulting "series"
 * is a splice of three populations that no run ever produced.
 */
export function aggregateKey(row: Pick<AggregateRow, 'policy' | 'step'>): string {
  return `${row.policy}:${row.step}`;
}

/** Absorb one row into a live buffer; the `final` row wins its own `hb` duplicate. */
export function absorbAggregateRow(buffer: Map<string, AggregateRow>, row: AggregateRow): void {
  const key = aggregateKey(row);
  const existing = buffer.get(key);
  if (!existing || row.kind === 'final') buffer.set(key, row);
}

// ---------- Kaplan-Meier ----------

/**
 * summary.survival is a sparse `{step: survivingFraction}` map — only the steps
 * where a lane died. Expand it into a stepped curve anchored at (0, 1) and the
 * horizon so the line spans the whole run.
 */
export function kmCurve(
  survival: Record<string, number> | undefined,
  horizon: number,
): { steps: number[]; fraction: number[] } {
  const steps = [0];
  const fraction = [1];
  const entries = Object.entries(survival ?? {})
    .map(([step, value]) => [Number(step), value] as const)
    .filter(([step, value]) => Number.isFinite(step) && Number.isFinite(value))
    .sort((a, b) => a[0] - b[0]);
  for (const [step, value] of entries) {
    steps.push(step);
    fraction.push(value);
  }
  const last = fraction[fraction.length - 1] ?? 1;
  if (horizon > (steps[steps.length - 1] ?? 0)) {
    steps.push(horizon);
    fraction.push(last);
  }
  return { steps, fraction };
}

// ---------- fleet columnar reshaping ----------

export interface LaneSeries {
  lane: number;
  seed: number;
  envId: string;
  policy: string;
  values: (number | null)[];
  /** last finite value, for ranking/labels */
  last: number | null;
}

/** First present key among candidates — the server picks the field naming. */
export function pickField(columnar: FleetColumnar, candidates: readonly string[]): string | null {
  const keys = Object.keys(columnar.series ?? {});
  for (const candidate of candidates) if (keys.includes(candidate)) return candidate;
  const suffix = candidates.map((c) => c.split('.').pop() ?? c);
  for (const key of keys) if (suffix.includes(key.split('.').pop() ?? key)) return key;
  return keys[0] ?? null;
}

export function laneSeries(columnar: FleetColumnar, field: string, envId?: string): LaneSeries[] {
  const rows = columnar.series?.[field];
  if (!rows) return [];
  const out: LaneSeries[] = [];
  for (let i = 0; i < columnar.lanes.length; i++) {
    const lane = columnar.lanes[i];
    if (!lane) continue;
    if (envId && lane.envId !== envId) continue;
    const values = rows[i] ?? [];
    let last: number | null = null;
    for (let j = values.length - 1; j >= 0; j--) {
      const value = values[j];
      if (value !== null && value !== undefined && Number.isFinite(value)) {
        last = value;
        break;
      }
    }
    out.push({ lane: lane.lane, seed: lane.seed, envId: lane.envId, policy: lane.policy, values, last });
  }
  return out;
}

/**
 * Turn a cumulative `rewardSum` series into the MEASURED reward rate over each
 * heartbeat window: (sum_i − sum_{i−1}) / (step_i − step_{i−1}).
 *
 * This is what "reward rate" means everywhere else in the lab — reward the
 * environment actually paid, divided by steps. It is NOT `health.rewardRate`,
 * which is the agent's own internal average-reward estimate ρ̄ (a learned
 * parameter, and a divergence diagnostic): plotting ρ̄ under a "reward rate"
 * label was the chart telling you what the agent BELIEVES it earns.
 *
 * The first window is measured from the origin (0 steps, 0 reward), which is
 * exactly its cumulative rate. Nulls propagate — a missing endpoint means the
 * window is unknown, never zero — and so does a zero-width step gap.
 */
export function windowRates(
  steps: readonly number[],
  values: readonly (number | null)[],
): (number | null)[] {
  const out: (number | null)[] = [];
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    const value = values[i];
    if (step === undefined || value === null || value === undefined || !Number.isFinite(value)) {
      out.push(null);
      continue;
    }
    const prevStep = i === 0 ? 0 : steps[i - 1] ?? 0;
    const prevValue = i === 0 ? 0 : values[i - 1];
    if (prevValue === null || prevValue === undefined || !Number.isFinite(prevValue)) {
      out.push(null);
      continue;
    }
    const span = step - prevStep;
    out.push(span > 0 ? (value - prevValue) / span : null);
  }
  return out;
}

/** Apply `windowRates` to every seed's series in place of the cumulative one. */
export function toWindowRates(steps: readonly number[], series: readonly LaneSeries[]): LaneSeries[] {
  return series.map((entry) => {
    const values = windowRates(steps, entry.values);
    let last: number | null = null;
    for (let i = values.length - 1; i >= 0; i--) {
      const v = values[i];
      if (v !== null && v !== undefined && Number.isFinite(v)) {
        last = v;
        break;
      }
    }
    return { ...entry, values, last };
  });
}

/**
 * Cap a spaghetti plot without lying about it: always keep the extremes and the
 * median seed, then fill with an even stride over the rank order. Deterministic.
 */
export function capLanes(series: readonly LaneSeries[], cap: number): LaneSeries[] {
  if (series.length <= cap) return [...series];
  const ranked = [...series].sort((a, b) => (a.last ?? -Infinity) - (b.last ?? -Infinity));
  const keep = new Set<number>([0, ranked.length - 1, Math.floor(ranked.length / 2)]);
  const stride = (ranked.length - 1) / Math.max(1, cap - keep.size);
  for (let i = 0; keep.size < cap && i < ranked.length; i++) {
    keep.add(Math.min(ranked.length - 1, Math.round(i * stride)));
  }
  const picked = [...keep].sort((a, b) => a - b).map((index) => ranked[index]);
  return picked.filter((entry): entry is LaneSeries => entry !== undefined);
}

/** Rank seeds by final value, descending — the seed ledger's default order. */
export function rankLanes(series: readonly LaneSeries[]): LaneSeries[] {
  return [...series].sort((a, b) => (b.last ?? -Infinity) - (a.last ?? -Infinity));
}

export type SortDirection = 'asc' | 'desc';

/** Stable comparator used by the sortable table; nulls always sink. */
export function compareValues(a: unknown, b: unknown, direction: SortDirection): number {
  const nullA = a === null || a === undefined || (typeof a === 'number' && !Number.isFinite(a));
  const nullB = b === null || b === undefined || (typeof b === 'number' && !Number.isFinite(b));
  if (nullA && nullB) return 0;
  if (nullA) return 1;
  if (nullB) return -1;
  let result: number;
  if (typeof a === 'number' && typeof b === 'number') result = a - b;
  else result = String(a).localeCompare(String(b), 'en', { numeric: true });
  return direction === 'asc' ? result : -result;
}
