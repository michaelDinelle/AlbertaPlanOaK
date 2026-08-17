import { api } from './api';
import { fetchTape, fieldIndex, tapeStep, tapeValue, type Tape } from './tape';
import type { RunDetail, RunIndexEntry, RunMeta } from '@shared/types';
import { DEATH_DETECTORS } from '@shared/types';
import { ENVS, ENV_IDS, isEnvId, type EnvId } from '@shared/envs';

export interface CachedFeed<T> {
  snapshot(): T | null;
  load(force?: boolean): Promise<T>;
}

function cachedFeed<T>(loader: () => Promise<T>): CachedFeed<T> {
  let data: T | null = null;
  let inflight: Promise<T> | null = null;
  return {
    snapshot: () => data,
    load(force = false) {
      if (!force && data !== null) return Promise.resolve(data);
      if (inflight) return inflight;
      inflight = loader().then(
        (value) => {
          inflight = null;
          data = value;
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

const tapeCache = new Map<string, { tape: Tape | null; inflight: Promise<Tape> | null }>();

export function loadTape(id: string, signal?: AbortSignal): Promise<Tape> {
  let slot = tapeCache.get(id);
  if (slot === undefined) {
    slot = { tape: null, inflight: null };
    tapeCache.set(id, slot);
  }
  if (slot.inflight !== null) return slot.inflight;
  if (slot.tape !== null) return Promise.resolve(slot.tape);
  return refreshTape(id, signal);
}

export function refreshTape(id: string, signal?: AbortSignal): Promise<Tape> {
  let slot = tapeCache.get(id);
  if (slot === undefined) {
    slot = { tape: null, inflight: null };
    tapeCache.set(id, slot);
  }
  if (slot.inflight !== null) return slot.inflight;
  const s = slot;
  s.inflight = fetchTape(id, s.tape, signal).then(
    (tape) => {
      s.tape = tape;
      s.inflight = null;
      return tape;
    },
    (error: unknown) => {
      s.inflight = null;
      throw error;
    },
  );
  return s.inflight;
}

export function tapeSnapshot(id: string): Tape | null {
  return tapeCache.get(id)?.tape ?? null;
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

export interface EnvShare {
  envId: EnvId;
  lanes: number;
  fraction: number;
}

export function envShares(meta: Pick<RunMeta, 'envSchedule'>): EnvShare[] {
  const counts = new Map<EnvId, number>();
  for (const entry of meta.envSchedule) {
    if (!isEnvId(entry.envId)) continue;
    const lanes = Math.max(0, entry.seedEnd - entry.seedStart + 1);
    counts.set(entry.envId, (counts.get(entry.envId) ?? 0) + lanes);
  }
  const total = [...counts.values()].reduce((sum, n) => sum + n, 0) || 1;
  return ENV_IDS.filter((id) => counts.has(id)).map((envId) => {
    const lanes = counts.get(envId) ?? 0;
    return { envId, lanes, fraction: lanes / total };
  });
}

export interface EnvLanes {
  lanes: number[];
  seeds: number[];
}

export function envLanes(meta: Pick<RunMeta, 'envSchedule' | 'seeds'>, envId: string): EnvLanes {
  const laneOfSeed = new Map<number, number>();
  meta.seeds.forEach((seed, lane) => laneOfSeed.set(seed, lane));
  const lanes: number[] = [];
  const seeds: number[] = [];
  for (const entry of meta.envSchedule) {
    if (entry.envId !== envId) continue;
    for (let seed = entry.seedStart; seed <= entry.seedEnd; seed++) {
      const lane = laneOfSeed.get(seed);
      if (lane !== undefined) {
        lanes.push(lane);
        seeds.push(seed);
      }
    }
  }
  return { lanes, seeds };
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

export interface LaneDeath {
  step: number;
  detector: string;
}

export function laneDeath(tape: Tape, lane: number): LaneDeath | null {
  if (tape.rows === 0) return null;
  const step = tapeStep(tape, fieldIndex(tape, 'deathStep'), tape.rows - 1, lane);
  if (!Number.isFinite(step)) return null;
  const code = tapeValue(tape, fieldIndex(tape, 'deathDetector'), tape.rows - 1, lane);
  return { step, detector: DEATH_DETECTORS[code] ?? 'envError' };
}

export function lastFiniteReward(tape: Tape, lane: number): { value: number; step: number } | null {
  const f = fieldIndex(tape, 'rewardSum');
  for (let r = tape.rows - 1; r >= 0; r--) {
    const v = tapeValue(tape, f, r, lane);
    if (Number.isFinite(v)) return { value: v, step: tape.stepEnd[r] ?? 0 };
  }
  return null;
}

export function deliveredRate(tape: Tape, lane: number): number | null {
  const last = lastFiniteReward(tape, lane);
  if (last === null || last.step <= 0) return null;
  return last.value / last.step;
}

export interface BinnedRates {
  x: number[];
  edges: number[];
  perLane: (number | null)[][];
  iqm: (number | null)[];
  p10: (number | null)[];
  p90: (number | null)[];
}

export function binnedRates(tape: Tape, lanes: readonly number[]): BinnedRates {
  const f = fieldIndex(tape, 'rewardSum');
  const x: number[] = [];
  const edges: number[] = [0];
  const perLane: (number | null)[][] = lanes.map(() => []);
  const iqmOut: (number | null)[] = [];
  const p10: (number | null)[] = [];
  const p90: (number | null)[] = [];
  let prevStep = 0;
  let prevRow = -1;
  const scratch: number[] = [];
  for (let r = 0; r < tape.rows; r++) {
    const step = tape.stepEnd[r] ?? 0;
    const span = step - prevStep;
    scratch.length = 0;
    for (let li = 0; li < lanes.length; li++) {
      const lane = lanes[li] ?? 0;
      const s1 = tapeValue(tape, f, r, lane);
      const s0 = prevRow < 0 ? 0 : tapeValue(tape, f, prevRow, lane);
      const dst = perLane[li];
      if (dst === undefined) continue;
      if (span > 0 && Number.isFinite(s1) && Number.isFinite(s0)) {
        const rate = (s1 - s0) / span;
        dst.push(rate);
        scratch.push(rate);
      } else {
        dst.push(null);
      }
    }
    scratch.sort((a, b) => a - b);
    iqmOut.push(scratch.length > 0 ? iqmSorted(scratch) : null);
    p10.push(scratch.length > 0 ? percentileSorted(scratch, 10) : null);
    p90.push(scratch.length > 0 ? percentileSorted(scratch, 90) : null);
    x.push((prevStep + step) / 2);
    edges.push(step);
    prevStep = step;
    prevRow = r;
  }
  return { x, edges, perLane, iqm: iqmOut, p10, p90 };
}

export interface RetentionResult {
  peak: number;
  end: number;
  retention: number;
}

export function laneRetention(tape: Tape, lane: number, horizon: number): RetentionResult | null {
  const f = fieldIndex(tape, 'rewardSum');
  const win = Math.max(Math.floor(horizon / 5), 1);
  let peak = -Infinity;
  let end: number | null = null;
  let j = 0;
  for (let i = 0; i < tape.rows; i++) {
    const endI = tape.stepEnd[i] ?? 0;
    while (j < tape.rows && (tape.stepEnd[j] ?? 0) <= endI - win) j++;
    const jj = j - 1;
    if (jj < 0) continue;
    const si = tapeValue(tape, f, i, lane);
    const sj = tapeValue(tape, f, jj, lane);
    const span = endI - (tape.stepEnd[jj] ?? 0);
    if (!Number.isFinite(si) || !Number.isFinite(sj) || span <= 0) continue;
    const rate = (si - sj) / span;
    if (rate > peak) peak = rate;
    end = rate;
  }
  if (end === null || !Number.isFinite(peak) || peak <= 0) return null;
  return { peak, end, retention: end / peak };
}

export interface ThroughputSeries {
  x: number[];
  stepsPerSecond: (number | null)[];
}

export function throughputSeries(tape: Tape): ThroughputSeries {
  const x: number[] = [];
  const out: (number | null)[] = [];
  let prevStep = 0;
  let prevWall = 0;
  for (let r = 0; r < tape.rows; r++) {
    if (!Number.isFinite(tape.wallSeconds[r])) continue;
    const step = tape.stepEnd[r] ?? 0;
    const wall = tape.wallSeconds[r] ?? NaN;
    const dWall = wall - prevWall;
    const dStep = step - prevStep;
    x.push((prevStep + step) / 2);
    out.push(dWall > 0 && dStep > 0 ? (dStep * tape.laneCount) / dWall : null);
    prevStep = step;
    prevWall = wall;
  }
  return { x, stepsPerSecond: out };
}

export interface SurvivalSeries {
  x: number[];
  surviving: number[];
  positive: number[];
}

export function survivalSeries(tape: Tape, lanes: readonly number[]): SurvivalSeries {
  const f = fieldIndex(tape, 'rewardSum');
  const deaths = lanes.map((lane) => laneDeath(tape, lane)?.step ?? Infinity);
  const x: number[] = [];
  const surviving: number[] = [];
  const positive: number[] = [];
  const n = Math.max(1, lanes.length);
  for (let r = 0; r < tape.rows; r++) {
    const step = tape.stepEnd[r] ?? 0;
    let alive = 0;
    let pos = 0;
    for (let li = 0; li < lanes.length; li++) {
      if ((deaths[li] ?? Infinity) > step) alive++;
      const v = tapeValue(tape, f, r, lanes[li] ?? 0);
      if (Number.isFinite(v) && v > 0) pos++;
    }
    x.push(step);
    surviving.push(alive / n);
    positive.push(pos / n);
  }
  return { x, surviving, positive };
}
