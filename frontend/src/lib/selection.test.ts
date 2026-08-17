import { expect, test } from 'bun:test';
import {
  escapeSelection,
  frameFromStep,
  heroSeed,
  nextSort,
  promoteSeed,
  setLeader,
  sortPermutation,
  stepFromFrame,
  type HeroSelection,
  type SeedOrderKey,
} from './selection';

const playableSet = (...seeds: number[]) => {
  const set = new Set(seeds);
  return (seed: number) => set.has(seed);
};

test('promoteSeed adopts only playable seeds', () => {
  const selection: HeroSelection = { leader: 1, promoted: null };
  expect(promoteSeed(selection, 9, playableSet(1, 2))).toBe(selection);
  expect(promoteSeed(selection, 2, playableSet(1, 2))).toEqual({ leader: 1, promoted: 2 });
});

test('promoteSeed is a no-op on the already promoted seed', () => {
  const selection: HeroSelection = { leader: 1, promoted: 2 };
  expect(promoteSeed(selection, 2, playableSet(2))).toBe(selection);
});

test('escapeSelection clears only an explicit promotion', () => {
  const idle: HeroSelection = { leader: 1, promoted: null };
  expect(escapeSelection(idle)).toBe(idle);
  expect(escapeSelection({ leader: 1, promoted: 5 })).toEqual({ leader: 1, promoted: null });
});

test('setLeader keeps a still-playable promotion and drops a dead one', () => {
  expect(setLeader({ leader: 3, promoted: 7 }, 4, playableSet(4, 7))).toEqual({ leader: 4, promoted: 7 });
  expect(setLeader({ leader: 3, promoted: 7 }, 4, playableSet(4))).toEqual({ leader: 4, promoted: null });
});

test('heroSeed prefers the promotion over the leader', () => {
  expect(heroSeed({ leader: 1, promoted: 5 })).toBe(5);
  expect(heroSeed({ leader: 1, promoted: null })).toBe(1);
  expect(heroSeed({ leader: null, promoted: null })).toBeNull();
});

const rows: SeedOrderKey[] = [
  { seed: 0, rate: 0.5, retention: 0.2 },
  { seed: 1, rate: null, retention: 0.9 },
  { seed: 2, rate: 1.5, retention: null },
  { seed: 3, rate: -0.5, retention: 0.4 },
];

test('sortPermutation orders by rate descending with nulls last', () => {
  expect(sortPermutation(rows, 'rate')).toEqual([2, 0, 3, 1]);
});

test('sortPermutation orders by seed ascending', () => {
  expect(sortPermutation(rows, 'seed')).toEqual([0, 1, 2, 3]);
});

test('sortPermutation orders by retention descending with nulls last', () => {
  expect(sortPermutation(rows, 'retention')).toEqual([1, 3, 0, 2]);
});

test('nextSort cycles rate → seed → retention → rate', () => {
  expect(nextSort('rate')).toBe('seed');
  expect(nextSort('seed')).toBe('retention');
  expect(nextSort('retention')).toBe('rate');
});

test('frame and step conversions invert each other', () => {
  expect(frameFromStep(500, 1000, 101)).toBe(50);
  expect(stepFromFrame(50, 1000, 101)).toBe(500);
  expect(frameFromStep(5000, 1000, 101)).toBe(100);
  expect(stepFromFrame(0, 1000, 101)).toBe(0);
});
