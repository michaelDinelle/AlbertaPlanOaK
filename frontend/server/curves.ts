// fleet.jsonl -> learning curves (the RL-paper chart).
//
// At every heartbeat the population's MEASURED reward rate over that window —
// (rewardSum_i − rewardSum_{i−1}) / (step_i − step_{i−1}), per seed — is reduced to
// an interquartile mean with a percentile-bootstrap 95% CI, plus the P10/P50/P90
// spread. Three properties this module is built around:
//
//   * MEASURED, NOT BELIEVED. The source column is `rewardSum`, the reward the
//     environment actually paid. `health.rewardRate` is the agent's own average-reward
//     estimate ρ̄ — a learned parameter and a divergence diagnostic, not a result.
//   * THE SAME IQM tools/fleet.py REPORTS. `iqm()` here is a transcription of
//     fleet.py's: sort, drop ⌊n/4⌋ from each end, mean the rest. A dashboard that
//     computed a different "IQM" than the promotion gate would be worse than none.
//   * DETERMINISTIC. The bootstrap draws from a seeded PRNG keyed by heartbeat index,
//     so one fleet.jsonl always yields one set of CI bounds — reloading the page must
//     never move the error bars.
//
// Nothing here needs summary.json: legacy roots (fleet.jsonl only) answer in full.
//
// Caching mirrors server/fleetDerive.ts (LRU + disk keyed on fleet.jsonl mtime+size +
// single-flight + a compute semaphore; in-flight runs bypass the disk cache because the
// file is still growing). It is a deliberate copy rather than a shared helper: the two
// modules are owned by different tracks right now, and duplicating ~90 lines of cache
// plumbing is the cheaper of the two mistakes. One difference, and it matters at 16
// envs: the single-flight key is the FILE, not the query, because ONE pass computes
// every env slice — a 16-panel dashboard costs a single stream of a 175 MB file.

import { readFileSync } from 'node:fs';
import { readdir, unlink, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { isEnvId } from '@shared/envs';
import type { CurvesResponse, FleetRow } from '@shared/types';
import {
  FLEET_CACHE_MAX_BYTES,
  FLEET_CACHE_MAX_FILES,
  FLEET_COMPUTE_CONCURRENCY,
  hash32,
} from './config';
import { statOrNull, streamJsonl } from './jsonl';
import { readTape, sampleWindows } from './metricsTape';
import { ensureCacheDir } from './runIndex';

/** bootstrap resamples per heartbeat — fleet.py reports at 10k; 2k is the interactive budget */
export const CURVES_BOOTSTRAP = 2000;
/** tape windows per curve — the bootstrap is O(windows x resamples x lanes) */
const CURVES_TAPE_WINDOWS = 256;
/** in-memory depth: one run's env set (16 envs + the pooled key) has to fit */
const CURVES_LRU_MAX = 24;
/** cache key of the un-filtered (all lanes) curve */
const ALL_ENVS = '*';

export interface CurvesQuery {
  policy: string;
  /** restrict to lanes of one env (mixed-env runs); null = every lane */
  envId: string | null;
}

/** Parse + validate the query string. Same rules as the fleet endpoint's. */
export function parseCurvesQuery(url: URL): { query: CurvesQuery } | { error: string } {
  const policy = url.searchParams.get('policy') ?? 'oak';
  if (!/^[A-Za-z0-9_-]{1,32}$/.test(policy)) return { error: 'bad policy' };

  const rawEnv = url.searchParams.get('envId');
  const envText = rawEnv === null ? '' : rawEnv.trim();
  // Absent/empty means "all lanes"; present-but-unknown is a client bug worth
  // surfacing, not a filter that silently matches nothing.
  if (envText.length > 0 && !isEnvId(envText)) return { error: `bad envId: ${envText}` };
  return { query: { policy, envId: envText.length > 0 ? envText : null } };
}

export function emptyCurves(): CurvesResponse {
  return { steps: [], iqm: [], ciLo: [], ciHi: [], p10: [], p50: [], p90: [], seeds: 0 };
}

// ---------- statistics ----------

/**
 * mulberry32. Chosen for being 4 lines, dependency-free and stable forever: the CI
 * bounds this module ships are only reproducible if the generator never drifts.
 */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * numpy's default `np.percentile` interpolation (linear on the (q/100)·(n−1) index).
 * The factor order is numpy's, not the algebraically equivalent ((n−1)·q)/100 — they
 * differ in the last ulp, and this module's whole claim is that it reports the same
 * numbers as the Python that gates promotions.
 */
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

/** Interquartile mean of an ASCENDING array — tools/fleet.py `iqm()`, transcribed. */
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

/**
 * Percentile-bootstrap 95% CI of the IQM, over an ASCENDING array.
 *
 * Resampling is done as COUNTS over the already-sorted values rather than as a fresh
 * array to sort: a resample's sorted order is just the sorted originals repeated by
 * their draw counts, so the trimmed mean is one linear walk. That turns an O(B·n·log n)
 * loop into O(B·n) — the difference between ~4 s and ~0.2 s at n=1408, B=2000, which is
 * the difference between a chart that loads and one nobody waits for.
 */
export function bootstrapIqmCi(
  sorted: readonly number[],
  boot: number,
  seed: number,
): [number, number] {
  const n = sorted.length;
  if (n === 0) return [Number.NaN, Number.NaN];
  if (n === 1) {
    const only = sorted[0] ?? Number.NaN;
    return [only, only];
  }
  const drop = Math.floor(n / 4);
  const lo = drop;
  const hi = n - drop;
  const width = hi - lo;
  const rng = makeRng(seed);
  const counts = new Int32Array(n);
  const draws = new Float64Array(boot);

  for (let b = 0; b < boot; b++) {
    counts.fill(0);
    for (let k = 0; k < n; k++) {
      const idx = (rng() * n) | 0;
      counts[idx] = (counts[idx] ?? 0) + 1;
    }
    let seen = 0;
    let sum = 0;
    for (let i = 0; i < n && seen < hi; i++) {
      const c = counts[i] ?? 0;
      if (c === 0) continue;
      const from = Math.max(seen, lo);
      const to = Math.min(seen + c, hi);
      if (to > from) sum += (sorted[i] ?? 0) * (to - from);
      seen += c;
    }
    draws[b] = sum / width;
  }

  const ordered = Array.from(draws).sort((a, b) => a - b);
  return [percentileSorted(ordered, 2.5), percentileSorted(ordered, 97.5)];
}

// ---------- LRU ----------

interface LruEntry {
  key: string;
  value: CurvesResponse;
}

const lru: LruEntry[] = [];

function lruGet(key: string): CurvesResponse | undefined {
  const i = lru.findIndex((e) => e.key === key);
  if (i < 0) return undefined;
  const [hit] = lru.splice(i, 1);
  if (hit === undefined) return undefined;
  lru.push(hit);
  return hit.value;
}

function lruPut(key: string, value: CurvesResponse): void {
  const i = lru.findIndex((e) => e.key === key);
  if (i >= 0) lru.splice(i, 1);
  lru.push({ key, value });
  while (lru.length > CURVES_LRU_MAX) lru.shift();
}

/** test hook */
export function clearCurvesCache(): void {
  lru.length = 0;
  inFlight.clear();
}

// ---------- single-flight + compute semaphore ----------

/** keyed on `${dir}|${policy}|${fileKey}` — one pass answers every env slice */
const inFlight = new Map<string, Promise<Map<string, CurvesResponse>>>();

let computeActive = 0;
const computeQueue: (() => void)[] = [];

async function withComputeSlot<T>(fn: () => Promise<T>): Promise<T> {
  // Queue behind existing waiters even when a slot looks free: a released slot is
  // claimed by the woken waiter one microtask later, so a "free" reading between the
  // release and that wake-up would let LIMIT+1 parses run at once.
  if (computeActive >= FLEET_COMPUTE_CONCURRENCY || computeQueue.length > 0) {
    await new Promise<void>((res) => computeQueue.push(res));
  }
  computeActive++;
  try {
    return await fn();
  } finally {
    computeActive--;
    const next = computeQueue.shift();
    if (next !== undefined) next();
  }
}

// ---------- disk cache ----------

function safeName(id: string): string {
  return id.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 96);
}

interface DiskCacheFile {
  v: 2;
  kind: 'curves';
  /** fleet.jsonl `${mtimeMs}:${size}` */
  key: string;
  policy: string;
  envId: string | null;
  data: CurvesResponse;
}

function queryKey(q: CurvesQuery): string {
  return `${q.policy}|${q.envId ?? ALL_ENVS}`;
}

function diskCachePath(cacheDir: string, id: string, q: CurvesQuery): string {
  // The hash names the file; the record inside proves what it is.
  return join(cacheDir, `${safeName(id)}--curves-${hash32(queryKey(q))}.json`);
}

async function readDiskCache(path: string, key: string, q: CurvesQuery): Promise<CurvesResponse | null> {
  try {
    const parsed: unknown = JSON.parse(await Bun.file(path).text());
    if (parsed === null || typeof parsed !== 'object') return null;
    const f = parsed as Partial<DiskCacheFile>;
    if (f.v !== 2 || f.kind !== 'curves' || f.key !== key || f.data === undefined) return null;
    if (f.policy !== q.policy || (f.envId ?? null) !== q.envId) return null; // hash collision
    return f.data;
  } catch {
    return null;
  }
}

async function writeDiskCache(path: string, payload: DiskCacheFile): Promise<void> {
  try {
    await ensureCacheDir(path);
    await writeFile(path, JSON.stringify(payload));
  } catch {
    /* cache is an optimization */
  }
  await evictDiskCache(path);
}

/** Keep the cache directory under both ceilings by deleting oldest-by-mtime first. */
async function evictDiskCache(writtenPath: string): Promise<void> {
  const dir = dirname(writtenPath);
  try {
    const names = (await readdir(dir)).filter((n) => n.endsWith('.json'));
    const entries: { path: string; mtimeMs: number; size: number }[] = [];
    let total = 0;
    for (const name of names) {
      const p = join(dir, name);
      const st = await statOrNull(p);
      if (st === null) continue;
      entries.push({ path: p, mtimeMs: st.mtimeMs, size: st.size });
      total += st.size;
    }
    if (entries.length <= FLEET_CACHE_MAX_FILES && total <= FLEET_CACHE_MAX_BYTES) return;
    entries.sort((a, b) => a.mtimeMs - b.mtimeMs); // oldest first
    let count = entries.length;
    for (const e of entries) {
      if (count <= FLEET_CACHE_MAX_FILES && total <= FLEET_CACHE_MAX_BYTES) break;
      if (e.path === writtenPath) continue; // never evict what we just wrote
      await unlink(e.path).catch(() => undefined);
      count--;
      total -= e.size;
    }
  } catch {
    /* eviction is best-effort */
  }
}

// ---------- derivation ----------

export interface CurvesOptions {
  id: string;
  dir: string;
  query: CurvesQuery;
  /** in-flight runs bypass the disk cache (the file is still being appended to) */
  inFlight: boolean;
  /** disk-cache directory; null/absent disables persistence */
  cacheDir?: string | null;
}

export async function deriveCurves(opts: CurvesOptions): Promise<CurvesResponse> {
  const { dir, query } = opts;
  const live = opts.inFlight;
  const cacheDir = opts.cacheDir ?? null;
  const fleetPath = join(dir, 'fleet.jsonl');
  const st = await statOrNull(fleetPath);
  if (st === null) return emptyCurves();

  const fileKey = `${st.mtimeMs}:${st.size}`;
  const envKey = query.envId ?? ALL_ENVS;
  const memKey = `${dir}|${query.policy}|${envKey}|${fileKey}`;
  const memHit = lruGet(memKey);
  if (memHit !== undefined) return memHit;

  const diskPath = cacheDir === null ? null : diskCachePath(cacheDir, opts.id, query);
  if (!live && diskPath !== null) {
    const cached = await readDiskCache(diskPath, fileKey, query);
    if (cached !== null) {
      lruPut(memKey, cached);
      return cached;
    }
  }

  // Single-flight on the FILE: the 16 env panels of one dashboard open together and
  // must share one parse, not queue 16 of them behind a 2-slot semaphore.
  const passKey = `${dir}|${query.policy}|${fileKey}`;
  let pass = inFlight.get(passKey);
  if (pass === undefined) {
    pass = withComputeSlot(() => computeCurves(fleetPath, query.policy)).finally(() => {
      inFlight.delete(passKey);
    });
    inFlight.set(passKey, pass);
  }
  const slices = await pass;
  for (const [key, value] of slices) lruPut(`${dir}|${query.policy}|${key}|${fileKey}`, value);

  const data = slices.get(envKey) ?? emptyCurves();
  if (!live && diskPath !== null) {
    void writeDiskCache(diskPath, {
      v: 2,
      kind: 'curves',
      key: fileKey,
      policy: query.policy,
      envId: query.envId,
      data,
    });
  }
  return data;
}

interface LaneKeyed {
  lane: number;
  envId: string;
}

/**
 * One stream of fleet.jsonl -> a curve per env plus the pooled one.
 *
 * Staged row-major during the pass (the lane/step extents are unknown until it ends),
 * then materialized into a lane × heartbeat grid of cumulative rewardSum, differenced
 * into windowed rates, and reduced across seeds.
 */
async function computeCurves(fleetPath: string, policy: string): Promise<Map<string, CurvesResponse>> {
  const stepValues: number[] = [];
  const stepSlot = new Map<number, number>();
  const laneValues: LaneKeyed[] = [];
  const laneSlot = new Map<number, number>();
  const rowLane: number[] = [];
  const rowStep: number[] = [];
  const rowSum: (number | null)[] = [];

  await streamJsonl<FleetRow>(fleetPath, (row) => {
    if (row === null || typeof row !== 'object') return;
    // METRICS LAW (2026-08-05): fleet.jsonl carries final certificates only;
    // the series come from the tape below. Pre-tape archives' hb rows are
    // ignored, not adapted.
    if ((row as { kind?: unknown }).kind === 'hb') return;
    if (typeof row.policy === 'string' && row.policy !== policy) return;
    if (typeof row.policy !== 'string' && policy !== 'oak') return;
    const lane = typeof row.lane === 'number' ? row.lane : -1;
    const step = typeof row.step === 'number' ? row.step : -1;
    if (lane < 0 || step < 0) return;

    let si = stepSlot.get(step);
    if (si === undefined) {
      si = stepValues.length;
      stepValues.push(step);
      stepSlot.set(step, si);
    }
    let li = laneSlot.get(lane);
    if (li === undefined) {
      li = laneValues.length;
      laneValues.push({ lane, envId: typeof row.envId === 'string' ? row.envId : '' });
      laneSlot.set(lane, li);
    }
    rowLane.push(li);
    rowStep.push(si);
    const sum = row.rewardSum;
    rowSum.push(typeof sum === 'number' && Number.isFinite(sum) ? sum : null);
  });

  // The continuous curve: tape windows at ~1k-step resolution, capped to keep
  // the bootstrap interactive. Final rows (above) stay authoritative.
  {
    const dir = dirname(fleetPath);
    const tape = readTape(join(dir, `metrics.${policy}.oakt`));
    if (tape !== null) {
      let meta: { envSchedule?: { envId: string; seedStart: number; seedEnd: number }[]; envId?: string; seeds?: number[] } = {};
      try {
        meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8')) as typeof meta;
      } catch {
        /* lane-indexed defaults below */
      }
      const envOf = (seed: number): string => {
        for (const e of meta.envSchedule ?? []) {
          if (seed >= e.seedStart && seed <= e.seedEnd) return e.envId;
        }
        return meta.envId ?? '';
      };
      const seeds = meta.seeds ?? Array.from({ length: tape.laneCount }, (_, i) => i);
      const fi = tape.fields.indexOf('rewardSum');
      if (fi >= 0) {
        for (const w of sampleWindows(tape.data.length, CURVES_TAPE_WINDOWS)) {
          const stepEnd = (tape.stepStart[w] ?? 0) + (tape.stepCount[w] ?? 0);
          if (stepSlot.has(stepEnd)) continue; // finals are authoritative
          const si = stepValues.length;
          stepValues.push(stepEnd);
          stepSlot.set(stepEnd, si);
          const vals = tape.data[w];
          if (vals === undefined) continue;
          for (let lane = 0; lane < tape.laneCount; lane++) {
            let li = laneSlot.get(lane);
            if (li === undefined) {
              li = laneValues.length;
              laneValues.push({ lane, envId: envOf(seeds[lane] ?? lane) });
              laneSlot.set(lane, li);
            }
            rowLane.push(li);
            rowStep.push(si);
            const v = vals[fi * tape.laneCount + lane];
            rowSum.push(v !== undefined && Number.isFinite(v) ? v : null);
          }
        }
      }
    }
  }

  const nS = stepValues.length;
  const nL = laneValues.length;
  if (nS === 0 || nL === 0) return new Map([[ALL_ENVS, emptyCurves()]]);

  // steps ascending; lanes by lane index (so a slice is a stable seed order)
  const stepOrder = stepValues.map((_, i) => i).sort((a, b) => (stepValues[a] ?? 0) - (stepValues[b] ?? 0));
  const stepRemap = new Array<number>(nS).fill(0);
  const steps: number[] = [];
  for (let newIdx = 0; newIdx < stepOrder.length; newIdx++) {
    const oldIdx = stepOrder[newIdx];
    if (oldIdx === undefined) continue;
    stepRemap[oldIdx] = newIdx;
    steps.push(stepValues[oldIdx] ?? 0);
  }
  const laneOrder = laneValues.map((_, i) => i).sort((a, b) => (laneValues[a]?.lane ?? 0) - (laneValues[b]?.lane ?? 0));
  const laneRemap = new Array<number>(nL).fill(0);
  const lanes: LaneKeyed[] = [];
  for (let newIdx = 0; newIdx < laneOrder.length; newIdx++) {
    const oldIdx = laneOrder[newIdx];
    if (oldIdx === undefined) continue;
    laneRemap[oldIdx] = newIdx;
    const lv = laneValues[oldIdx];
    if (lv !== undefined) lanes.push(lv);
  }

  // lane × heartbeat cumulative reward. NaN marks "no row" — distinct from a real 0.
  const grid = new Float64Array(nL * nS).fill(Number.NaN);
  for (let r = 0; r < rowLane.length; r++) {
    const li = laneRemap[rowLane[r] ?? 0] ?? 0;
    const si = stepRemap[rowStep[r] ?? 0] ?? 0;
    const v = rowSum[r];
    // Later rows win: a kind:"final" row lands on the last heartbeat's (lane, step)
    // and is the authoritative value.
    grid[li * nS + si] = v === null || v === undefined ? Number.NaN : v;
  }

  // Windowed measured rate, in place. Walked BACKWARDS so each window still reads the
  // untouched cumulative value of its predecessor. The first window is measured from
  // the origin (0 steps, 0 reward), which is exactly its cumulative rate.
  for (let l = 0; l < nL; l++) {
    const base = l * nS;
    for (let i = nS - 1; i >= 0; i--) {
      const value = grid[base + i] ?? Number.NaN;
      const prevValue = i === 0 ? 0 : (grid[base + i - 1] ?? Number.NaN);
      const span = (steps[i] ?? 0) - (i === 0 ? 0 : (steps[i - 1] ?? 0));
      grid[base + i] = span > 0 ? (value - prevValue) / span : Number.NaN;
    }
  }

  const byEnv = new Map<string, number[]>();
  for (let l = 0; l < lanes.length; l++) {
    const envId = lanes[l]?.envId ?? '';
    if (envId.length === 0) continue;
    const list = byEnv.get(envId);
    if (list === undefined) byEnv.set(envId, [l]);
    else list.push(l);
  }

  const all = allLaneIndices(nL);
  const out = new Map<string, CurvesResponse>();
  const pooled = reduceCurve(grid, nS, all, steps);
  out.set(ALL_ENVS, pooled);
  for (const [envId, indices] of byEnv) {
    // A single-env run's slice IS the pooled curve — same seeds, same numbers. Share
    // the object instead of paying for a second bootstrap of the same population.
    out.set(envId, indices.length === nL ? pooled : reduceCurve(grid, nS, indices, steps));
  }
  return out;
}

function allLaneIndices(n: number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}

/** Across-seed reduction at every heartbeat: IQM ± bootstrap CI, and the P10/50/90 spread. */
function reduceCurve(
  grid: Float64Array,
  nS: number,
  laneIndices: readonly number[],
  steps: readonly number[],
): CurvesResponse {
  const out: CurvesResponse = {
    steps: [...steps],
    iqm: [],
    ciLo: [],
    ciHi: [],
    p10: [],
    p50: [],
    p90: [],
    seeds: laneIndices.length,
  };
  const column: number[] = [];
  for (let i = 0; i < nS; i++) {
    column.length = 0;
    for (const l of laneIndices) {
      const v = grid[l * nS + i];
      if (v !== undefined && Number.isFinite(v)) column.push(v);
    }
    if (column.length === 0) {
      out.iqm.push(null);
      out.ciLo.push(null);
      out.ciHi.push(null);
      out.p10.push(null);
      out.p50.push(null);
      out.p90.push(null);
      continue;
    }
    const sorted = [...column].sort((a, b) => a - b);
    // Seeded per heartbeat index, so a curve is reproducible column by column no
    // matter how many heartbeats the file has grown to.
    const [lo, hi] = bootstrapIqmCi(sorted, CURVES_BOOTSTRAP, (0x9e3779b9 ^ (i * 0x85ebca6b)) >>> 0);
    out.iqm.push(finiteOrNull(iqmSorted(sorted)));
    out.ciLo.push(finiteOrNull(lo));
    out.ciHi.push(finiteOrNull(hi));
    out.p10.push(finiteOrNull(percentileSorted(sorted, 10)));
    out.p50.push(finiteOrNull(percentileSorted(sorted, 50)));
    out.p90.push(finiteOrNull(percentileSorted(sorted, 90)));
  }
  return out;
}

function finiteOrNull(v: number): number | null {
  return Number.isFinite(v) ? v : null;
}
