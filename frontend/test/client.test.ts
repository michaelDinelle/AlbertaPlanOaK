/// <reference types="bun" />
// Client-shell pure logic: store, TTL resource, API URL building, formatters,
// and the reshaping helpers the views depend on. No DOM here — happy-dom is not
// installed, so views are verified by hand against the running API server.

import { describe, expect, test } from 'bun:test';

import { apiUrl, liveUrl } from '../src/api';
import {
  absorbAggregateRow,
  aggregateKey,
  aggregateSeries,
  capLanes,
  compareValues,
  createResource,
  createStore,
  EMPTY_FILTER,
  envSliceResult,
  envIdsInIndex,
  envIdsOf,
  envShares,
  formatAgo,
  formatCount,
  formatPercent,
  formatRate,
  formatSigned,
  formatStep,
  formatWall,
  kmCurve,
  laneSeries,
  matchesFilter,
  pickField,
  policiesIn,
  policyView,
  primaryResult,
  rankLanes,
  sortRuns,
} from '../src/state';
import { parseQuartiles } from '../src/components/bars';
import type { AggregateRow, FleetColumnar, RunDetail, RunIndexEntry } from '@shared/types';

// ---------------------------------------------------------------- store

describe('createStore', () => {
  test('get / set / subscribe / unsubscribe', () => {
    const store = createStore({ n: 1 });
    const seen: number[] = [];
    const off = store.subscribe((value) => seen.push(value.n));

    store.set({ n: 2 });
    store.set((prev) => ({ n: prev.n + 1 }));
    off();
    store.set({ n: 99 });

    expect(seen).toEqual([2, 3]);
    expect(store.get().n).toBe(99);
  });

  test('skips notification when the value is identical', () => {
    const value = { n: 1 };
    const store = createStore(value);
    let calls = 0;
    store.subscribe(() => calls++);
    store.set(value);
    expect(calls).toBe(0);
    store.set({ ...value });
    expect(calls).toBe(1);
  });

  test('a listener unsubscribing during notify does not skip its peers', () => {
    const store = createStore(0);
    const seen: string[] = [];
    const offA = store.subscribe(() => {
      seen.push('a');
      offA();
    });
    store.subscribe(() => seen.push('b'));
    store.set(1);
    expect(seen).toEqual(['a', 'b']);
  });
});

// ------------------------------------------------------------- resource

describe('createResource', () => {
  test('serves from cache inside the TTL and refetches after it', async () => {
    let now = 1_000;
    let calls = 0;
    const resource = createResource(async () => {
      calls++;
      return calls;
    }, 60_000, () => now);

    expect(await resource.load()).toBe(1);
    expect(await resource.load()).toBe(1);
    expect(calls).toBe(1);

    now += 59_999;
    expect(await resource.load()).toBe(1);
    expect(calls).toBe(1);

    now += 2;
    expect(await resource.load()).toBe(2);
    expect(calls).toBe(2);
  });

  test('force bypasses the cache; invalidate expires it', async () => {
    let calls = 0;
    const resource = createResource(async () => ++calls, 60_000, () => 0);
    await resource.load();
    await resource.load(true);
    expect(calls).toBe(2);
    resource.invalidate();
    await resource.load();
    expect(calls).toBe(3);
  });

  test('deduplicates concurrent loads', async () => {
    let calls = 0;
    const resource = createResource(async () => {
      calls++;
      await Promise.resolve();
      return calls;
    }, 60_000, () => 0);
    const [a, b] = await Promise.all([resource.load(), resource.load()]);
    expect(calls).toBe(1);
    expect(a).toBe(b);
  });

  test('records the error and keeps the last good data', async () => {
    let fail = false;
    const resource = createResource(async () => {
      if (fail) throw new Error('boom');
      return 'ok';
    }, 0, () => 0);
    await resource.load();
    fail = true;
    await expect(resource.load(true)).rejects.toThrow('boom');
    expect(resource.store.get().status).toBe('error');
    expect(resource.store.get().error).toBe('boom');
    expect(resource.store.get().data).toBe('ok');
  });
});

// ------------------------------------------------------------------ api

describe('apiUrl', () => {
  test('prefixes /api once', () => {
    expect(apiUrl('/runs')).toBe('/api/runs');
    expect(apiUrl('runs')).toBe('/api/runs');
    expect(apiUrl('/api/runs')).toBe('/api/runs');
  });

  test('drops empty values and joins arrays with commas', () => {
    // 0 is a value, not an absence — only '', null and undefined are dropped
    expect(apiUrl('/runs/x/fleet', { fields: ['a.b', 'c'], envId: undefined, limit: 0 })).toBe(
      '/api/runs/x/fleet?fields=a.b%2Cc&limit=0',
    );
    expect(apiUrl('/runs', { envId: '', policy: null })).toBe('/api/runs');
    expect(apiUrl('/runs', {})).toBe('/api/runs');
  });

  test('encodes keys and values', () => {
    expect(apiUrl('/runs', { envId: 'HalfCheetah-v4', q: 'a b&c' })).toBe(
      '/api/runs?envId=HalfCheetah-v4&q=a%20b%26c',
    );
  });

  test('the fleet query is fields + envId + policy, and nothing else', () => {
    // policy is load-bearing: the server filters fleet.jsonl by it (default `oak`),
    // so a zero/random view that omits it silently reads oak's lanes. maxLanes is
    // gone — lane capping is a client-side decision (capLanes), never a request.
    expect(
      apiUrl('/runs/p93/fleet', { fields: ['health.rewardRate'], envId: 'DeepSea-cont-v0', policy: 'zero' }),
    ).toBe('/api/runs/p93/fleet?fields=health.rewardRate&envId=DeepSea-cont-v0&policy=zero');
  });

  test('live url escapes the run id', () => {
    expect(liveUrl('p93-tierL-20260721')).toBe('/api/runs/p93-tierL-20260721/live');
    expect(liveUrl('a/b')).toBe('/api/runs/a%2Fb/live');
  });
});

// ----------------------------------------------------------- formatters

describe('formatters', () => {
  test('formatRate picks a representation by magnitude', () => {
    expect(formatRate(0.38725)).toBe('0.387');
    expect(formatRate(-5.264837)).toBe('-5.26');
    expect(formatRate(0)).toBe('0');
    expect(formatRate(0.0000123)).toBe('1.2e-5');
    expect(formatRate(125609.58)).toBe('1.3e+5');
    expect(formatRate(null)).toBe('—');
    expect(formatRate(Number.NaN)).toBe('—');
  });

  test('formatSigned marks positives', () => {
    expect(formatSigned(0.38725)).toBe('+0.387');
    expect(formatSigned(-0.0003)).toBe('-0.000300');
    expect(formatSigned(0)).toBe('0');
  });

  test('formatStep abbreviates', () => {
    expect(formatStep(100_000)).toBe('100k');
    expect(formatStep(1_000_000)).toBe('1M');
    expect(formatStep(1_500_000)).toBe('1.5M');
    expect(formatStep(999)).toBe('999');
    expect(formatStep(null)).toBe('—');
  });

  test('formatPercent, formatCount, formatWall', () => {
    expect(formatPercent(0.995, 1)).toBe('99.5%');
    expect(formatPercent(null)).toBe('—');
    expect(formatCount(1234)).toBe('1,234');
    expect(formatWall(101.9)).toBe('1m 42s');
    expect(formatWall(3720)).toBe('1h 02m');
    expect(formatWall(9)).toBe('9s');
    expect(formatWall(-1)).toBe('—');
  });

  test('formatAgo is relative to an injected clock', () => {
    const now = Date.parse('2026-08-03T12:00:00Z');
    expect(formatAgo('2026-08-03T11:59:30Z', now)).toBe('just now');
    expect(formatAgo('2026-08-03T11:00:00Z', now)).toBe('60m ago');
    expect(formatAgo('2026-08-02T12:00:00Z', now)).toBe('24h ago');
    expect(formatAgo('2026-07-21T12:00:00Z', now)).toBe('13d ago');
    expect(formatAgo(null, now)).toBe('undated');
    expect(formatAgo('not a date', now)).toBe('undated');
  });
});

// -------------------------------------------------------------- env map

describe('envShares', () => {
  test('expands a mixed schedule in canonical env order', () => {
    const shares = envShares({
      envId: 'HalfCheetah-v4',
      batch: 24,
      envSchedule: [
        { envId: 'DeepSea-cont-v0', seedStart: 8, seedEnd: 23, flopsPerLaneStep: 1 },
        { envId: 'HalfCheetah-v4', seedStart: 0, seedEnd: 7, flopsPerLaneStep: 1 },
      ],
    });
    expect(shares.map((s) => s.envId)).toEqual(['HalfCheetah-v4', 'DeepSea-cont-v0']);
    expect(shares.map((s) => s.lanes)).toEqual([8, 16]);
    expect(shares[1]?.fraction).toBeCloseTo(2 / 3, 10);
  });

  test('falls back to the single declared env', () => {
    const shares = envShares({ envId: 'HalfCheetah-v4', batch: 1024 });
    expect(shares).toEqual([{ envId: 'HalfCheetah-v4', lanes: 1024, fraction: 1 }]);
  });

  test('ignores unknown env ids', () => {
    expect(envShares({ envId: 'Nope-v0', batch: 8 })).toEqual([]);
  });

  test('an index entry yields env IDENTITY only, in canonical order', () => {
    // the index carries envIds + batch but never the split, so there is no lane
    // count to be had here — inventing batch/n printed fiction on every mixed run
    expect(envIdsOf({ envIds: ['DeepSea-cont-v0', 'HalfCheetah-v4'] })).toEqual([
      'HalfCheetah-v4',
      'DeepSea-cont-v0',
    ]);
    expect(envIdsOf({ envIds: [] })).toEqual([]);
    expect(envIdsOf({ envIds: ['Nope-v0'] })).toEqual([]);
  });
});

// -------------------------------------------------------------- gallery

function entry(overrides: Partial<RunIndexEntry> = {}): RunIndexEntry {
  return {
    id: 'run-a',
    envIds: ['HalfCheetah-v4'],
    batch: 1024,
    steps: 1_000_000,
    policies: ['oak'],
    ablationLabel: 'none',
    startedAt: '2026-07-21T00:00:00Z',
    complete: true,
    maxStep: 1_000_000,
    inFlight: false,
    hasSummary: true,
    hasRecordings: false,
    role: null,
    ...overrides,
  };
}

describe('gallery filtering and ordering', () => {
  test('every toggle narrows the set', () => {
    const plain = entry();
    expect(matchesFilter(plain, EMPTY_FILTER)).toBe(true);
    expect(matchesFilter(plain, { ...EMPTY_FILTER, recordingsOnly: true })).toBe(false);
    expect(matchesFilter(plain, { ...EMPTY_FILTER, inFlightOnly: true })).toBe(false);
    expect(matchesFilter(plain, { ...EMPTY_FILTER, rolesOnly: true })).toBe(false);
    expect(matchesFilter(entry({ complete: false }), { ...EMPTY_FILTER, completeOnly: true })).toBe(false);
    expect(matchesFilter(plain, { ...EMPTY_FILTER, envId: 'DeepSea-cont-v0' })).toBe(false);
    expect(matchesFilter(plain, { ...EMPTY_FILTER, envId: 'HalfCheetah-v4' })).toBe(true);
  });

  test('search is an AND over whitespace-separated terms', () => {
    const keeper = entry({ id: 'p93-tierL-20260721', role: 'keeper', ablationLabel: 'model-to-keyboard' });
    expect(matchesFilter(keeper, { ...EMPTY_FILTER, search: 'p93 keeper' })).toBe(true);
    expect(matchesFilter(keeper, { ...EMPTY_FILTER, search: 'p93 planning' })).toBe(false);
    expect(matchesFilter(keeper, { ...EMPTY_FILTER, search: '  TIERL ' })).toBe(true);
  });

  test('running first, then keeper/candidate/parent, then newest', () => {
    const rows = [
      entry({ id: 'old', startedAt: '2026-01-01T00:00:00Z' }),
      entry({ id: 'keeper', role: 'keeper', startedAt: '2026-02-01T00:00:00Z' }),
      entry({ id: 'new', startedAt: '2026-08-01T00:00:00Z' }),
      entry({ id: 'live', inFlight: true, startedAt: '2025-01-01T00:00:00Z' }),
      entry({ id: 'parent', role: 'parent', startedAt: '2026-02-01T00:00:00Z' }),
    ];
    expect(sortRuns(rows).map((r) => r.id)).toEqual(['live', 'keeper', 'parent', 'new', 'old']);
  });

  test('env filter options follow the canonical env order', () => {
    expect(
      envIdsInIndex([
        entry({ envIds: ['DeepSea-cont-v0'] }),
        entry({ envIds: ['HalfCheetah-v4', 'DeepSea-cont-v0'] }),
      ]),
    ).toEqual(['HalfCheetah-v4', 'DeepSea-cont-v0']);
  });
});

// ------------------------------------------------------------ aggregate

function agg(step: number, kind: 'hb' | 'final', p50: number, policy = 'oak'): AggregateRow {
  return {
    step, kind, policy, alive: 128, dead: 0,
    deliveredRateP10: p50 - 1, deliveredRateP50: p50, deliveredRateP90: p50 + 1,
    recentRateP10: 0, recentRateP50: 0, recentRateP90: 0, recentRateMax: 0,
    bestLaneSeed: 3, bestLaneDeliveredRate: 1, bestLaneRewardSum: 1,
    positiveRateLaneCount: 96, positiveRateFraction: 0.75,
    aggStepsPerSecond: 1000, wallSeconds: step / 1000,
  };
}

describe('aggregateSeries', () => {
  test('sorts by step and lets the final row win its duplicate heartbeat', () => {
    const series = aggregateSeries([agg(20_000, 'hb', 2), agg(10_000, 'hb', 1), agg(20_000, 'final', 9)]);
    expect(series.steps).toEqual([10_000, 20_000]);
    expect(series.p50).toEqual([1, 9]);
    expect(series.p10).toEqual([0, 8]);
  });

  test('filters by policy and lists the policies present', () => {
    const rows = [agg(10_000, 'hb', 1, 'oak'), agg(10_000, 'hb', 5, 'random')];
    expect(aggregateSeries(rows, 'random').p50).toEqual([5]);
    expect(policiesIn(rows)).toEqual(['oak', 'random']);
  });

  test('an empty input yields empty arrays, not undefined', () => {
    expect(aggregateSeries([]).steps).toEqual([]);
  });
});

describe('live aggregate buffering (the watch view)', () => {
  test('rows are keyed by POLICY and step — never step alone', () => {
    expect(aggregateKey({ policy: 'oak', step: 10_000 })).toBe('oak:10000');
    expect(aggregateKey({ policy: 'zero', step: 10_000 })).not.toBe(aggregateKey({ policy: 'oak', step: 10_000 }));
  });

  test('a three-policy stream stays three series instead of one splice', () => {
    const buffer = new Map<string, AggregateRow>();
    // exactly what a multi-policy live stream delivers: three rows per step
    for (const step of [10_000, 20_000, 30_000]) {
      absorbAggregateRow(buffer, agg(step, 'hb', 0.4, 'oak'));
      absorbAggregateRow(buffer, agg(step, 'hb', -0.2, 'zero'));
      absorbAggregateRow(buffer, agg(step, 'hb', -0.9, 'random'));
    }
    const rows = [...buffer.values()];
    expect(rows).toHaveLength(9);
    expect(policiesIn(rows)).toEqual(['oak', 'random', 'zero']);
    expect(aggregateSeries(rows, 'oak').p50).toEqual([0.4, 0.4, 0.4]);
    expect(aggregateSeries(rows, 'zero').p50).toEqual([-0.2, -0.2, -0.2]);
    expect(aggregateSeries(rows, 'oak').steps).toEqual([10_000, 20_000, 30_000]);
  });

  test('the final row still wins its own heartbeat, per policy', () => {
    const buffer = new Map<string, AggregateRow>();
    absorbAggregateRow(buffer, agg(10_000, 'hb', 1, 'oak'));
    absorbAggregateRow(buffer, agg(10_000, 'final', 9, 'oak'));
    absorbAggregateRow(buffer, agg(10_000, 'hb', 2, 'zero'));
    // a late duplicate hb must not clobber the final
    absorbAggregateRow(buffer, agg(10_000, 'hb', 3, 'oak'));
    expect(buffer.get('oak:10000')?.deliveredRateP50).toBe(9);
    expect(buffer.get('zero:10000')?.deliveredRateP50).toBe(2);
  });
});

describe('policyView (the run dashboard)', () => {
  const detail = (): Pick<RunDetail, 'summary' | 'aggregate' | 'meta'> => ({
    meta: {
      fleetSchemaVersion: 1, envId: 'HalfCheetah-v4', batch: 1024, seeds: [], steps: 1_000_000,
      policies: ['oak', 'zero'],
    },
    summary: {
      results: [
        { policy: 'oak', iqmDeliveredAll: 0.387, positiveFraction: 0.995, complete: true },
        { policy: 'zero', iqmDeliveredAll: -0.264, positiveFraction: 0.01, complete: false, maxStep: 300_000 },
      ],
    },
    aggregate: [
      agg(1_000_000, 'final', 0.4, 'oak'),
      agg(300_000, 'hb', -0.3, 'zero'),
    ],
  });

  test('every policy-dependent number comes from ITS OWN policy', () => {
    const oak = policyView(detail(), 'oak');
    const zero = policyView(detail(), 'zero');
    expect(oak.result?.iqmDeliveredAll).toBe(0.387);
    expect(zero.result?.iqmDeliveredAll).toBe(-0.264);
    expect(oak.complete).toBe(true);
    expect(zero.complete).toBe(false);
    expect(zero.maxStep).toBe(300_000);
  });

  test('completeness falls back to the policy\'s own furthest heartbeat', () => {
    const d = detail();
    d.summary = { results: [{ policy: 'oak' }, { policy: 'zero' }] };
    expect(policyView(d, 'oak').maxStep).toBe(1_000_000);
    expect(policyView(d, 'oak').complete).toBe(true);
    // zero stalled at 300k — reading oak's last step here is exactly the bug
    expect(policyView(d, 'zero').maxStep).toBe(300_000);
    expect(policyView(d, 'zero').complete).toBe(false);
  });

  test('an unknown policy falls back to the first result, never to nothing', () => {
    const view = policyView(detail(), 'nope');
    expect(view.result?.policy).toBe('oak');
    expect(view.maxStep).toBe(1_000_000);
  });

  test('env slices are policy-keyed too', () => {
    const summary = {
      envSlices: {
        'HalfCheetah-v4': [
          { policy: 'oak', positiveFraction: 0.99 },
          { policy: 'zero', positiveFraction: 0.02 },
        ],
      },
    };
    expect(envSliceResult(summary, 'HalfCheetah-v4', 'zero')?.positiveFraction).toBe(0.02);
    expect(envSliceResult(summary, 'HalfCheetah-v4', 'missing')?.policy).toBe('oak');
    expect(envSliceResult(summary, 'Nope-v0', 'oak')).toBeNull();
  });
});

describe('kmCurve', () => {
  test('anchors at (0, 1) and extends flat to the horizon', () => {
    const curve = kmCurve({ '645809': 0.999, '957478': 0.997 }, 1_000_000);
    expect(curve.steps).toEqual([0, 645_809, 957_478, 1_000_000]);
    expect(curve.fraction).toEqual([1, 0.999, 0.997, 0.997]);
  });

  test('no deaths is a flat survivor line', () => {
    expect(kmCurve(undefined, 100_000)).toEqual({ steps: [0, 100_000], fraction: [1, 1] });
  });
});

// --------------------------------------------------------------- fleet

function columnar(): FleetColumnar {
  return {
    steps: [10_000, 20_000],
    lanes: [
      { lane: 0, seed: 0, policy: 'oak', envId: 'HalfCheetah-v4' },
      { lane: 1, seed: 1, policy: 'oak', envId: 'HalfCheetah-v4' },
      { lane: 8, seed: 8, policy: 'oak', envId: 'DeepSea-cont-v0' },
    ],
    series: {
      'health.rewardRate': [
        [0.1, 0.2],
        [-0.1, null],
        [0.5, 0.6],
      ],
    },
  };
}

describe('fleet columnar reshaping', () => {
  test('pickField prefers an exact key, then a suffix match, then anything', () => {
    const data = columnar();
    expect(pickField(data, ['health.rewardRate'])).toBe('health.rewardRate');
    expect(pickField(data, ['rewardRate'])).toBe('health.rewardRate');
    expect(pickField(data, ['nope'])).toBe('health.rewardRate');
    expect(pickField({ steps: [], lanes: [], series: {} }, ['x'])).toBeNull();
  });

  test('laneSeries carries identity and the last finite value', () => {
    const rows = laneSeries(columnar(), 'health.rewardRate');
    expect(rows.map((r) => r.seed)).toEqual([0, 1, 8]);
    expect(rows.map((r) => r.last)).toEqual([0.2, -0.1, 0.6]);
  });

  test('laneSeries filters by env — metrics never pool across envs', () => {
    const rows = laneSeries(columnar(), 'health.rewardRate', 'DeepSea-cont-v0');
    expect(rows.map((r) => r.lane)).toEqual([8]);
  });

  test('an unknown field yields nothing rather than throwing', () => {
    expect(laneSeries(columnar(), 'missing')).toEqual([]);
  });

  test('rankLanes orders by final value, descending, nulls last', () => {
    const rows = laneSeries(columnar(), 'health.rewardRate');
    rows.push({ lane: 9, seed: 9, envId: 'x', policy: 'oak', values: [], last: null });
    expect(rankLanes(rows).map((r) => r.lane)).toEqual([8, 0, 1, 9]);
  });
});

describe('capLanes', () => {
  const make = (n: number) =>
    Array.from({ length: n }, (_, i) => ({
      lane: i, seed: i, envId: 'e', policy: 'oak', values: [i], last: i,
    }));

  test('passes through when under the cap', () => {
    expect(capLanes(make(5), 10)).toHaveLength(5);
  });

  test('keeps the extremes and the median, and never exceeds the cap', () => {
    const capped = capLanes(make(100), 12);
    expect(capped.length).toBeLessThanOrEqual(12);
    const lanes = capped.map((l) => l.lane);
    expect(lanes).toContain(0);
    expect(lanes).toContain(99);
    expect(lanes).toContain(50);
  });

  test('is deterministic', () => {
    expect(capLanes(make(64), 9).map((l) => l.lane)).toEqual(capLanes(make(64), 9).map((l) => l.lane));
  });
});

// --------------------------------------------------------------- misc

describe('compareValues', () => {
  test('numbers numerically, strings naturally, nulls always last', () => {
    expect(compareValues(2, 10, 'asc')).toBeLessThan(0);
    expect(compareValues(2, 10, 'desc')).toBeGreaterThan(0);
    expect(compareValues('seed2', 'seed10', 'asc')).toBeLessThan(0);
    expect(compareValues(null, 5, 'asc')).toBeGreaterThan(0);
    expect(compareValues(null, 5, 'desc')).toBeGreaterThan(0);
    expect(compareValues(Number.NaN, 5, 'asc')).toBeGreaterThan(0);
    expect(compareValues(null, undefined, 'asc')).toBe(0);
  });
});

describe('parseQuartiles', () => {
  test('reads the fleet.py "q1 / q2 / q3" string', () => {
    expect(parseQuartiles('0.8474 / 1 / 1')).toEqual([0.8474, 1, 1]);
    expect(parseQuartiles('5e+04 / 9e+04 / 1e+05')).toEqual([50_000, 90_000, 100_000]);
  });

  test('rejects anything else', () => {
    expect(parseQuartiles('1 / 2')).toBeNull();
    expect(parseQuartiles('a / b / c')).toBeNull();
    expect(parseQuartiles(undefined)).toBeNull();
    expect(parseQuartiles(42)).toBeNull();
  });
});

describe('primaryResult', () => {
  test('prefers the requested policy, falls back to the first', () => {
    const summary = { results: [{ policy: 'oak', iqmDeliveredAll: 1 }, { policy: 'random', iqmDeliveredAll: 0 }] };
    expect(primaryResult(summary, 'random')?.iqmDeliveredAll).toBe(0);
    expect(primaryResult(summary, 'missing')?.policy).toBe('oak');
    expect(primaryResult(null)).toBeNull();
    expect(primaryResult({ results: [] })).toBeNull();
  });
});
