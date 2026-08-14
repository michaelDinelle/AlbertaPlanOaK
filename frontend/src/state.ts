
import { api } from './api';
import type {
  AggregateRow,
  FleetColumnar,
  RunDetail,
  RunIndexEntry,
  RunMeta,
  SummaryPolicyResult,
} from '@shared/types';
import { ENVS, ENV_IDS, isEnvId, type EnvId } from '@shared/envs';


const CACHE_TTL_MS = 60_000;

export interface CachedFeed<T> {
  snapshot(): T | null;
  load(force?: boolean): Promise<T>;
}

function cachedFeed<T>(loader: () => Promise<T>): CachedFeed<T> {
  let data: T | null = null;
  let fetchedAt = 0;
  let inflight: Promise<T> | null = null;
  return {
    snapshot: () => data,
    load(force = false) {
      if (!force && data !== null && Date.now() - fetchedAt < CACHE_TTL_MS) {
        return Promise.resolve(data);
      }
      if (inflight) return inflight;
      inflight = loader().then(
        (value) => {
          inflight = null;
          data = value;
          fetchedAt = Date.now();
          return value;
        },
        (error: unknown) => {
          inflight = null;
          throw error;
        },
      );
      return inflight;
    },
  };
}

export const runIndex: CachedFeed<RunIndexEntry[]> = cachedFeed(() => api.runs());

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

export interface PolicyView {
  policy: string;
  result: SummaryPolicyResult | null;
  maxStep: number;
  complete: boolean;
}

export function policyView(
  detail: Pick<RunDetail, 'summary' | 'aggregate' | 'meta' | 'recordings'>,
  policy: string,
): PolicyView {
  const result = primaryResult(detail.summary, policy);
  let policyStep = 0;
  let anyStep = 0;
  for (const row of detail.aggregate) {
    if (row.step > anyStep) anyStep = row.step;
    if (row.policy === policy && row.step > policyStep) policyStep = row.step;
  }
  let recStep = 0;
  for (const rec of detail.recordings) {
    if (rec.steps > recStep) recStep = rec.steps;
  }
  const declared = typeof result?.['maxStep'] === 'number' && Number.isFinite(result['maxStep'])
    ? (result['maxStep'] as number)
    : null;
  const reached = Math.max(policyStep, anyStep, recStep);
  const maxStep = declared ?? reached;
  const horizon = detail.meta.steps;
  return { policy, result, maxStep, complete: result?.complete !== false && maxStep >= horizon };
}

export interface EnvShare {
  envId: EnvId;
  lanes: number;
  fraction: number;
}

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

export function envIdsOf(entry: Pick<RunIndexEntry, 'envIds'>): EnvId[] {
  return ENV_IDS.filter((id) => entry.envIds.includes(id));
}

export function envName(envId: string): string {
  return isEnvId(envId) ? ENVS[envId].name : envId;
}

export function envSlug(envId: string): string {
  return isEnvId(envId) ? ENVS[envId].slug : envId.toLowerCase();
}


export function sortRuns(entries: readonly RunIndexEntry[]): RunIndexEntry[] {
  return [...entries].sort((a, b) => {
    if (a.inFlight !== b.inFlight) return a.inFlight ? -1 : 1;
    if (a.role !== b.role) return a.role === 'keeper' ? -1 : 1;
    const ta = a.startedAt ? Date.parse(a.startedAt) : 0;
    const tb = b.startedAt ? Date.parse(b.startedAt) : 0;
    if (tb !== ta) return tb - ta;
    return a.id.localeCompare(b.id);
  });
}

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

export function aggregateSeries(rows: readonly AggregateRow[], policy?: string): AggregateSeries {
  const filtered = policy ? rows.filter((r) => r.policy === policy) : rows;
  if (filtered.length === 0) return { ...EMPTY_AGGREGATE };
  const byStep = new Map<number, AggregateRow>();
  for (const row of filtered) {
    const existing = byStep.get(row.step);
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


export interface LaneSeries {
  lane: number;
  seed: number;
  envId: string;
  policy: string;
  values: (number | null)[];
  last: number | null;
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

export function windowMidpoints(steps: readonly number[]): number[] {
  return steps.map((step, i) => (i === 0 ? step / 2 : ((steps[i - 1] ?? 0) + step) / 2));
}

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

export function percentileSorted(sorted: readonly number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return Number.NaN;
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const a = sorted[lo];
  const b = sorted[hi];
  if (a === undefined || b === undefined) return Number.NaN;
  return lo === hi ? a : a + (b - a) * (pos - lo);
}

export function iqmSorted(sorted: readonly number[]): number {
  const n = sorted.length;
  if (n === 0) return Number.NaN;
  const drop = Math.floor(n / 4);
  const lo = drop;
  const hi = n - drop;
  if (hi <= lo) {
    let all = 0;
    for (const v of sorted) all += v;
    return all / n;
  }
  let sum = 0;
  for (let i = lo; i < hi; i++) sum += sorted[i] ?? 0;
  return sum / (hi - lo);
}

export function iqm(values: readonly number[]): number {
  return iqmSorted([...values].sort((a, b) => a - b));
}

export interface PopulationBandValues {
  p10: (number | null)[];
  p90: (number | null)[];
  iqm: (number | null)[];
}

export function populationBand(
  steps: readonly number[],
  lanes: readonly LaneSeries[],
): PopulationBandValues {
  const p10: (number | null)[] = [];
  const p90: (number | null)[] = [];
  const mid: (number | null)[] = [];
  for (let i = 0; i < steps.length; i++) {
    const vals: number[] = [];
    for (const lane of lanes) {
      const v = lane.values[i];
      if (v !== null && v !== undefined && Number.isFinite(v)) vals.push(v);
    }
    vals.sort((a, b) => a - b);
    if (vals.length === 0) {
      p10.push(null);
      p90.push(null);
      mid.push(null);
    } else {
      p10.push(percentileSorted(vals, 10));
      p90.push(percentileSorted(vals, 90));
      mid.push(iqmSorted(vals));
    }
  }
  return { p10, p90, iqm: mid };
}

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

export function rankLanes(series: readonly LaneSeries[]): LaneSeries[] {
  return [...series].sort((a, b) => (b.last ?? -Infinity) - (a.last ?? -Infinity));
}

export type SortDirection = 'asc' | 'desc';

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
