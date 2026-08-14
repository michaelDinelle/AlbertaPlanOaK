/// <reference types="bun" />

import { describe, expect, it } from 'bun:test';
import { iqm, iqmSorted, percentileSorted } from '../../src/state';

describe('curve statistics', () => {
  it('iqm is tools/fleet.py\'s: sort, drop ⌊n/4⌋ from each end, mean the rest', () => {
    expect(iqm([4, 1, 3, 2])).toBeCloseTo(2.5, 12);
    expect(iqm([3, 1, 2])).toBeCloseTo(2, 12);
    expect(iqm([-0.5, 0.5])).toBeCloseTo(0, 12);
    expect(iqm([5])).toBeCloseTo(5, 12);
    expect(iqm([-0.5, 0.5, 1, 2, 3, 4])).toBeCloseTo(1.625, 12);
    expect(iqmSorted([1, 2, 3, 4, 5, 6, 7, 100])).toBeCloseTo(4.5, 12);
    expect(Number.isNaN(iqm([]))).toBe(true);
  });

  it('percentileSorted matches numpy linear interpolation', () => {
    const four = [1, 2, 3, 4];
    expect(percentileSorted(four, 10)).toBeCloseTo(1.3, 12);
    expect(percentileSorted(four, 50)).toBeCloseTo(2.5, 12);
    expect(percentileSorted(four, 90)).toBeCloseTo(3.7, 12);
    const three = [1, 2, 3];
    expect(percentileSorted(three, 10)).toBeCloseTo(1.2, 12);
    expect(percentileSorted(three, 90)).toBeCloseTo(2.8, 12);
    const six = [-0.5, 0.5, 1, 2, 3, 4];
    expect(percentileSorted(six, 10)).toBeCloseTo(0, 12);
    expect(percentileSorted(six, 50)).toBeCloseTo(1.5, 12);
    expect(percentileSorted(six, 90)).toBeCloseTo(3.5, 12);
    expect(percentileSorted([5], 42)).toBeCloseTo(5, 12);
    expect(Number.isNaN(percentileSorted([], 50))).toBe(true);
  });
});
