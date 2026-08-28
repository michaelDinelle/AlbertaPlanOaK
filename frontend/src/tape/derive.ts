import type { RunMeta } from '@shared/types';
import { DEATH_DETECTORS } from '@shared/types';
import { envLanes } from '../lib/run';
import { fieldIndex, tapeStep, tapeValue, type Tape } from './parse';

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
  perLane: (number | null)[][];
  iqm: (number | null)[];
  p10: (number | null)[];
  p90: (number | null)[];
}

export const retentionWindow = (steps: number): number => Math.max(Math.floor(steps / 5), 1);

function binnedRates(
  tape: Tape,
  lanes: readonly number[],
  rowValue: (row: number, lane: number) => number,
): BinnedRates {
  const x: number[] = [];
  const perLane: (number | null)[][] = lanes.map(() => []);
  const iqmOut: (number | null)[] = [];
  const p10: (number | null)[] = [];
  const p90: (number | null)[] = [];
  const scratch: number[] = [];
  for (let r = 0; r < tape.rows; r++) {
    scratch.length = 0;
    for (let li = 0; li < lanes.length; li++) {
      const dst = perLane[li];
      if (dst === undefined) continue;
      const v = rowValue(r, lanes[li] ?? 0);
      if (Number.isFinite(v)) {
        dst.push(v);
        scratch.push(v);
      } else {
        dst.push(null);
      }
    }
    scratch.sort((a, b) => a - b);
    iqmOut.push(scratch.length > 0 ? iqmSorted(scratch) : null);
    p10.push(scratch.length > 0 ? percentileSorted(scratch, 10) : null);
    p90.push(scratch.length > 0 ? percentileSorted(scratch, 90) : null);
    x.push(tape.stepEnd[r] ?? 0);
  }
  return { x, perLane, iqm: iqmOut, p10, p90 };
}

export function windowedRates(tape: Tape, lanes: readonly number[], win: number): BinnedRates {
  const f = fieldIndex(tape, 'rewardSum');
  const priorRow = new Int32Array(tape.rows);
  const spans = new Float64Array(tape.rows);
  let j = 0;
  for (let i = 0; i < tape.rows; i++) {
    const endI = tape.stepEnd[i] ?? 0;
    while (j < tape.rows && (tape.stepEnd[j] ?? 0) <= endI - win) j++;
    const jj = j - 1;
    priorRow[i] = jj;
    spans[i] = endI - (jj < 0 ? 0 : tape.stepEnd[jj] ?? 0);
  }
  return binnedRates(tape, lanes, (row, lane) => {
    const jj = priorRow[row] ?? -1;
    const span = spans[row] ?? 0;
    const si = tapeValue(tape, f, row, lane);
    const sj = jj < 0 ? 0 : tapeValue(tape, f, jj, lane);
    return span > 0 && Number.isFinite(si) && Number.isFinite(sj) ? (si - sj) / span : NaN;
  });
}

export function rhoBinned(tape: Tape, lanes: readonly number[]): BinnedRates {
  const f = fieldIndex(tape, 'rho');
  return binnedRates(tape, lanes, (row, lane) => tapeValue(tape, f, row, lane));
}

export interface RetentionResult {
  peak: number;
  end: number;
  retention: number;
}

export function laneRetention(tape: Tape, lane: number, horizon: number): RetentionResult | null {
  const f = fieldIndex(tape, 'rewardSum');
  const win = retentionWindow(horizon);
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

export interface SeedRow {
  seed: number;
  lane: number;
  rate: number | null;
  retention: number | null;
  divergedAt: number | null;
  spark: Float64Array;
  sparkXs: Float64Array;
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
  const spark = windowedRates(tape, lanes, 1);
  const sparkXs = Float64Array.from(spark.x, (x) => (maxStep > 0 ? x / maxStep : 0));
  return lanes.map((lane, li) => ({
    seed: seeds[li] ?? 0,
    lane,
    rate: deliveredRate(tape, lane),
    retention: laneRetention(tape, lane, meta.steps)?.retention ?? null,
    divergedAt: laneDeath(tape, lane)?.step ?? null,
    spark: Float64Array.from(spark.perLane[li] ?? [], (v) => v ?? NaN),
    sparkXs,
    recorded: recorded.has(seeds[li] ?? 0),
  })).sort((a, b) => a.seed - b.seed);
}

export function tapeWindowSteps(tape: Pick<Tape, 'stepStart' | 'stepEnd'>): number {
  const window = (tape.stepEnd[0] ?? 0) - (tape.stepStart[0] ?? 0);
  return window > 0 ? window : 1;
}

export function nextRowDueMs(tape: Pick<Tape, 'rows' | 'wallSeconds'> | null): number {
  if (tape === null || tape.rows < 2) return 0;
  const last = tape.wallSeconds[tape.rows - 1];
  const prev = tape.wallSeconds[tape.rows - 2];
  if (last === undefined || prev === undefined) return 0;
  const perRow = (last - prev) * 1000;
  return Number.isFinite(perRow) && perRow > 0 ? perRow : 0;
}
