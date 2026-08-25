import { expect, test } from 'bun:test';
import { parseTape } from './parse';
import { deliveredRate, laneDeath, survivalSeries, windowedRates } from './derive';
import { syntheticTapeBytes } from './synthetic';

const FIELDS = ['rewardSum', 'rho', 'deathStep', 'deathDetector'];

const tape = await parseTape(await syntheticTapeBytes(2, FIELDS, [
  {
    start: 0,
    count: 100,
    wall: 1,
    values: [[10, -20], [0.1, -0.2], [NaN, NaN], [0, 0]],
  },
  {
    start: 100,
    count: 100,
    wall: 2,
    values: [[30, -60], [0.3, -0.6], [NaN, { bits: 150 }], [0, 2]],
  },
]));

test('windowedRates with a one-row window recovers per-row rates and their spread', () => {
  const binned = windowedRates(tape, [0, 1], 1);
  expect(binned.x).toEqual([100, 200]);
  expect(binned.perLane[0]).toEqual([0.1, 0.2]);
  expect(binned.perLane[1]).toEqual([-0.2, -0.4]);
  expect(binned.iqm[0]).toBeCloseTo(-0.05, 10);
  expect(binned.iqm[1]).toBeCloseTo(-0.1, 10);
  expect(binned.p10[0]).toBeCloseTo(-0.17, 10);
  expect(binned.p90[0]).toBeCloseTo(0.07, 10);
  expect(binned.p10[1]).toBeCloseTo(-0.34, 10);
  expect(binned.p90[1]).toBeCloseTo(0.14, 10);
});

test('windowedRates expands from the origin until the window fills, then trails at row ends', () => {
  const windowed = windowedRates(tape, [0, 1], 100);
  expect(windowed.x).toEqual([100, 200]);
  expect(windowed.perLane[0]).toEqual([0.1, 0.2]);
  expect(windowed.perLane[1]).toEqual([-0.2, -0.4]);
  expect(windowed.iqm[0]).toBeCloseTo(-0.05, 10);
  expect(windowed.iqm[1]).toBeCloseTo(-0.1, 10);
});

test('laneDeath reads the step out of the bits and names the detector', () => {
  expect(laneDeath(tape, 0)).toBeNull();
  expect(laneDeath(tape, 1)).toEqual({ step: 150, detector: 'policyMean' });
});

test('survivalSeries counts alive and positive lanes per row', () => {
  const survival = survivalSeries(tape, [0, 1]);
  expect(survival.x).toEqual([100, 200]);
  expect(survival.surviving).toEqual([1, 0.5]);
  expect(survival.positive).toEqual([0.5, 0.5]);
});

test('deliveredRate divides the last finite reward by its step', () => {
  expect(deliveredRate(tape, 0)).toBeCloseTo(0.15, 10);
  expect(deliveredRate(tape, 1)).toBeCloseTo(-0.3, 10);
});
