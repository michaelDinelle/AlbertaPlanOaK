import type { CumulativeReward } from './derive';
import { iqmSorted, percentileSorted } from './derive';
import type { RewardDetail } from '../data/rewards';

export interface PenCurves {
  x: number[];
  perLane: (number | null)[][];
  iqm: (number | null)[];
  p10: (number | null)[];
  p90: (number | null)[];
}

function sampleGrid(head: number, samples: number): number[] {
  const grid: number[] = [];
  for (let k = 0; k <= samples; k++) grid.push((head * k) / samples);
  return grid;
}

function rowAtOrAfter(rowEnds: readonly number[], t: number): number {
  let lo = 0;
  let hi = rowEnds.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((rowEnds[mid] ?? 0) < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function penCurves(cumulative: CumulativeReward, win: number, head: number, samples: number, detail: RewardDetail | null = null): PenCurves {
  const rowEnds = cumulative.x;
  const rows = rowEnds.length;
  const rowsKnown = rows > 0 ? rowEnds[rows - 1] ?? 0 : 0;
  const detailKnown = detail === null ? 0 : detail.from + detail.steps;
  const known = Math.max(rowsKnown, detailKnown);
  const anchorRow = detail === null ? -1 : rowEnds.indexOf(detail.from);
  const x = sampleGrid(Math.max(0, head), samples);
  const perLane: (number | null)[][] = cumulative.perLane.map(() => []);
  const iqm: (number | null)[] = [];
  const p10: (number | null)[] = [];
  const p90: (number | null)[] = [];
  const scratch: number[] = [];
  const rewardAt = (lane: readonly number[], t: number, laneIndex: number): number => {
    if (t <= 0) return 0;
    if (detail !== null && t >= detail.from && t <= detail.from + detail.steps) {
      const anchor = detail.from === 0 ? 0 : anchorRow >= 0 ? lane[anchorRow] ?? NaN : NaN;
      const base = laneIndex * (detail.steps + 1);
      const offset = t - detail.from;
      const k = Math.floor(offset);
      const lo = detail.prefix[base + k] ?? NaN;
      const hi = detail.prefix[base + Math.min(k + 1, detail.steps)] ?? lo;
      return anchor + lo + (offset - k) * (hi - lo);
    }
    const row = rowAtOrAfter(rowEnds, t);
    if (row >= rows) return lane[rows - 1] ?? NaN;
    const x1 = rowEnds[row] ?? 0;
    const y1 = lane[row] ?? NaN;
    const x0 = row === 0 ? 0 : rowEnds[row - 1] ?? 0;
    const y0 = row === 0 ? 0 : lane[row - 1] ?? NaN;
    return x1 === x0 ? y1 : y0 + ((t - x0) / (x1 - x0)) * (y1 - y0);
  };
  const rateAt = (lane: readonly number[], t: number, laneIndex: number): number => {
    if (rows === 0 && detail === null) return NaN;
    const at = Math.min(t, known);
    if (at <= 0) {
      if (detail !== null && detail.from === 0 && detail.steps > 0) return detail.prefix[laneIndex * (detail.steps + 1) + 1] ?? NaN;
      const x0 = rowEnds[0] ?? 0;
      return x0 > 0 ? (lane[0] ?? NaN) / x0 : NaN;
    }
    const from = Math.max(0, at - win);
    return (rewardAt(lane, at, laneIndex) - rewardAt(lane, from, laneIndex)) / (at - from);
  };
  for (const t of x) {
    scratch.length = 0;
    cumulative.perLane.forEach((lane, li) => {
      const value = rateAt(lane, t, li);
      const dst = perLane[li];
      if (dst === undefined) return;
      if (Number.isFinite(value)) {
        dst.push(value);
        scratch.push(value);
      } else {
        dst.push(null);
      }
    });
    scratch.sort((a, b) => a - b);
    iqm.push(scratch.length > 0 ? iqmSorted(scratch) : null);
    p10.push(scratch.length > 0 ? percentileSorted(scratch, 10) : null);
    p90.push(scratch.length > 0 ? percentileSorted(scratch, 90) : null);
  }
  return { x, perLane, iqm, p10, p90 };
}
