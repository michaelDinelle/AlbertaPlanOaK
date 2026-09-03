import { describe, expect, test } from 'bun:test';
import { penCurves } from './pen';

describe('penCurves', () => {
  const cumulative = { x: [1000, 2000], perLane: [[10, 20], [10, 30]] };

  test('a constant reward rate is flat from step 0 to the head', () => {
    const pen = penCurves({ x: cumulative.x, perLane: [cumulative.perLane[0]!] }, 500, 1500, 4);
    expect(pen.x[0]).toBe(0);
    expect(pen.x[pen.x.length - 1]).toBe(1500);
    for (const value of pen.perLane[0]!) expect(value).toBeCloseTo(0.01, 12);
  });

  test('the head interpolates the cumulative reward between rows', () => {
    const pen = penCurves({ x: cumulative.x, perLane: [cumulative.perLane[1]!] }, 500, 1500, 1);
    const head = pen.perLane[0]![pen.x.length - 1];
    expect(head).toBeCloseTo(0.02, 12);
  });

  test('beyond the last row the pen holds the last known rate', () => {
    const pen = penCurves({ x: cumulative.x, perLane: [cumulative.perLane[1]!] }, 500, 2600, 1);
    const head = pen.perLane[0]![pen.x.length - 1];
    expect(head).toBeCloseTo(0.02, 12);
  });

  test('the fleet statistics are taken across lanes at every sample', () => {
    const pen = penCurves(cumulative, 500, 1500, 2);
    expect(pen.iqm.length).toBe(pen.x.length);
    expect(pen.iqm[pen.x.length - 1]).toBeCloseTo(0.015, 12);
  });
});

describe('penCurves with per-step detail', () => {
  const lanes = 1;
  const rewards = [1, 2, 3, 4, 5, 6, 7, 8];
  const prefix = new Float64Array(rewards.length + 1);
  rewards.forEach((r, i) => { prefix[i + 1] = prefix[i]! + r; });
  const detail = { from: 0, steps: rewards.length, lanes, prefix };
  const cumulative = { x: [8], perLane: [[36]] };

  test('the head is exact at every step inside the detail window', () => {
    const headAt = (t: number) => {
      const pen = penCurves(cumulative, 1000, t, 1, detail);
      return pen.perLane[0]![pen.x.length - 1]!;
    };
    expect(headAt(1)).toBeCloseTo(1, 12);
    expect(headAt(3)).toBeCloseTo((1 + 2 + 3) / 3, 12);
    expect(headAt(6)).toBeCloseTo((1 + 2 + 3 + 4 + 5 + 6) / 6, 12);
  });

  test('a trailing window inside the detail uses only the last steps', () => {
    const pen = penCurves(cumulative, 2, 6, 1, detail);
    expect(pen.perLane[0]![pen.x.length - 1]).toBeCloseTo((5 + 6) / 2, 12);
  });

  test('beyond the loaded detail the pen holds the last exact value', () => {
    const shortDetail = { ...detail, steps: 4, prefix: prefix.slice(0, 5) };
    const pen = penCurves({ x: [], perLane: [[]] }, 1000, 6, 1, shortDetail);
    expect(pen.perLane[0]![pen.x.length - 1]).toBeCloseTo((1 + 2 + 3 + 4) / 4, 12);
  });
});
