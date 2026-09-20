import { isEnvId, type EnvId } from './data/envs';
import { HEALTH_ALIVE, type Meta, type Recording } from './data/layout';

export function runEnvs(meta: Meta): EnvId[] {
  const out: EnvId[] = [];
  for (const entry of meta.envSchedule) {
    if (!isEnvId(entry.envId)) throw new Error(`run schedule names unknown env ${entry.envId}`);
    if (!out.includes(entry.envId)) out.push(entry.envId);
  }
  return out;
}

export function envLanes(meta: Meta, envId: string): { lanes: number[]; seeds: number[] } {
  const laneOfSeed = new Map(meta.seeds.map((seed, lane) => [seed, lane]));
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

const at = (rec: Recording, row: number, lane: number): number => rec.rewardSum[row * rec.lanes + lane] ?? NaN;

const derived = new WeakMap<Recording, Map<string, unknown>>();

export function cached<T>(rec: Recording, key: string, make: () => T): T {
  let byKey = derived.get(rec);
  if (byKey === undefined) derived.set(rec, (byKey = new Map()));
  if (!byKey.has(key)) byKey.set(key, make());
  return byKey.get(key) as T;
}

export function leaderSeed(rec: Recording, lanes: { lanes: number[]; seeds: number[] }): number | null {
  let top = lanes.seeds[0] ?? null;
  let topRate = -Infinity;
  lanes.lanes.forEach((lane, i) => {
    const rate = deliveredRate(rec, lane);
    if (rate !== null && rate > topRate) {
      topRate = rate;
      top = lanes.seeds[i] ?? null;
    }
  });
  return top;
}

export function percentileSorted(sorted: readonly number[], q: number): number {
  const pos = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const a = sorted[lo] ?? NaN;
  const b = sorted[Math.ceil(pos)] ?? NaN;
  return a + (b - a) * (pos - lo);
}

export function iqmSorted(sorted: readonly number[]): number {
  const drop = Math.floor(sorted.length / 4);
  const slice = sorted.length - 2 * drop > 0 ? sorted.slice(drop, sorted.length - drop) : sorted;
  return slice.reduce((sum, v) => sum + v, 0) / slice.length;
}

export interface Rates {
  x: number[];
  perLane: (number | null)[][];
  iqm: (number | null)[];
  p10: (number | null)[];
  p90: (number | null)[];
}

function binned(rec: Recording, lanes: readonly number[], value: (row: number, lane: number) => number): Rates {
  const out: Rates = { x: [], perLane: lanes.map(() => []), iqm: [], p10: [], p90: [] };
  for (let r = 0; r < rec.windows; r++) {
    const finite: number[] = [];
    lanes.forEach((lane, i) => {
      const v = value(r, lane);
      out.perLane[i]!.push(Number.isFinite(v) ? v : null);
      if (Number.isFinite(v)) finite.push(v);
    });
    finite.sort((a, b) => a - b);
    out.x.push(rec.ends[r] ?? 0);
    out.iqm.push(finite.length > 0 ? iqmSorted(finite) : null);
    out.p10.push(finite.length > 0 ? percentileSorted(finite, 10) : null);
    out.p90.push(finite.length > 0 ? percentileSorted(finite, 90) : null);
  }
  return out;
}

export const rateWindow = (steps: number): number => Math.max(Math.floor(steps / 5), 1);

export function windowedRates(rec: Recording, lanes: readonly number[], win: number): Rates {
  const priorRow = new Int32Array(rec.windows);
  const spans = new Float64Array(rec.windows);
  let j = 0;
  for (let i = 0; i < rec.windows; i++) {
    const endI = rec.ends[i] ?? 0;
    while (j < rec.windows && (rec.ends[j] ?? 0) <= endI - win) j++;
    priorRow[i] = j - 1;
    spans[i] = endI - (j < 1 ? 0 : rec.ends[j - 1] ?? 0);
  }
  return binned(rec, lanes, (row, lane) => {
    const jj = priorRow[row] ?? -1;
    const span = spans[row] ?? 0;
    const si = at(rec, row, lane);
    const sj = jj < 0 ? 0 : at(rec, jj, lane);
    return span > 0 ? (si - sj) / span : NaN;
  });
}

export function deliveredRate(rec: Recording, lane: number): number | null {
  for (let r = rec.windows - 1; r >= 0; r--) {
    const v = at(rec, r, lane);
    const step = rec.ends[r] ?? 0;
    if (Number.isFinite(v)) return step > 0 ? v / step : null;
  }
  return null;
}

export function deathStep(rec: Recording, lane: number): number | null {
  const step = rec.health[lane];
  return step === undefined || step === HEALTH_ALIVE ? null : step;
}

export function survival(rec: Recording, lanes: readonly number[]): { x: number[]; surviving: number[]; positive: number[] } {
  const deaths = lanes.map((lane) => deathStep(rec, lane) ?? Infinity);
  const out = { x: [] as number[], surviving: [] as number[], positive: [] as number[] };
  const n = Math.max(1, lanes.length);
  for (let r = 0; r < rec.windows; r++) {
    const step = rec.ends[r] ?? 0;
    out.x.push(step);
    out.surviving.push(deaths.filter((d) => d > step).length / n);
    out.positive.push(lanes.filter((lane) => at(rec, r, lane) > 0).length / n);
  }
  return out;
}

export function peakStep(rec: Recording, lane: number, win: number): number | null {
  const rates = windowedRates(rec, [lane], win);
  let peak = -Infinity;
  let step: number | null = null;
  rates.perLane[0]!.forEach((rate, row) => {
    if (rate === null || rate <= peak) return;
    peak = rate;
    step = rates.x[row]!;
  });
  return step === null ? null : step - Math.floor(Math.min(step, win) / 2);
}
