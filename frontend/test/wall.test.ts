/// <reference types="bun" />
// The Wall's pure logic: which run it opens on, the URL round-trip, the sort as
// an index permutation, the timelapse arithmetic, the promote/Esc state machine,
// the degraded predicate, the rail's per-seed derivation and the population line.
//
// No DOM here (happy-dom is not installed) — `views/wall.ts`'s mount half is the
// DOM/WebGL part and is verified against the running dev server plus headless
// screenshots. Everything it DECIDES is exported and pinned down here.

import { describe, expect, test } from 'bun:test';

import type { FleetColumnar, RecordingRef, RolesResponse, RunIndexEntry, SummaryPolicyResult } from '@shared/types';
import { ENVS } from '@shared/envs';
import { MAX_CATCHUP_FRAMES, makePlaybackClock } from '@kit/harness';
import { TOKENS } from '@kit/tokens';
import { stepPosition } from '../src/theater/player';
import {
  CHEETAH_CAM_DEPTH,
  CHEETAH_CAM_LOOK_X,
  EMPTY_SELECTION,
  RECORD_RECIPE,
  SPARK_ROW_BUDGET,
  WALL_SORTS,
  WALL_SPEEDS,
  WALL_TARGET_SAMPLES,
  buildSeedRows,
  cycleEnv,
  degradedReason,
  escapeSelection,
  fieldExtent,
  fieldPercent,
  fieldScale,
  formatDistance,
  ghostAlive,
  ghostSampleIndex,
  heroSeed,
  inFrameCount,
  iqmOf,
  isPrime,
  laneTrackX,
  leaderSeed,
  nextPrime,
  nextSort,
  parseWallParams,
  populationFigures,
  promoteSeed,
  rateQuantiles,
  resolveFeaturedRun,
  retentionOf,
  runScheduleLine,
  setLeader,
  sortPermutation,
  stageFraming,
  trackWindow,
  urlPosition,
  wallEnvs,
  wallHref,
  wallKeyAction,
  wallStride,
  type SeedRow,
  type WallParams,
  type WallSort,
} from '../src/views/wall';

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

function entry(id: string, overrides: Partial<RunIndexEntry> = {}): RunIndexEntry {
  return {
    id,
    envIds: ['HalfCheetah-v4'],
    batch: 1024,
    steps: 100_000,
    policies: ['oak'],
    ablationLabel: 'none',
    startedAt: '2026-08-01T00:00:00Z',
    complete: true,
    maxStep: 100_000,
    inFlight: false,
    hasSummary: true,
    hasRecordings: true,
    role: null,
    ...overrides,
  };
}

const NO_ROLES: RolesResponse = { bestKernel: null, candidate: null, parent: null, featured: null };

function recording(envId: RecordingRef['envId'], seed: number): RecordingRef {
  return { envId, seed, steps: 100_000 };
}

function row(seed: number, overrides: Partial<SeedRow> = {}): SeedRow {
  return { seed, rate: 0, retention: null, divergedAt: null, spark: [], recorded: true, ...overrides };
}

// ---------------------------------------------------------------------------

describe('featured-run resolution', () => {
  const entries = [
    entry('older-with-recordings', { startedAt: '2026-07-01T00:00:00Z' }),
    entry('newest-complete', { startedAt: '2026-08-03T00:00:00Z' }),
    entry('newest-incomplete', { startedAt: '2026-08-04T00:00:00Z', complete: false }),
    entry('no-recordings', { startedAt: '2026-08-05T00:00:00Z', hasRecordings: false }),
  ];

  test('an explicit ?run= wins over every role, recordings or not', () => {
    const roles: RolesResponse = { ...NO_ROLES, featured: 'newest-complete', bestKernel: 'newest-complete' };
    expect(resolveFeaturedRun(entries, roles, 'no-recordings')).toEqual({
      runId: 'no-recordings',
      source: 'param',
    });
  });

  test('an unknown ?run= falls through instead of showing a run that is not there', () => {
    expect(resolveFeaturedRun(entries, NO_ROLES, 'typo-run').runId).toBe('newest-complete');
  });

  test('bestKernel is used only when it has recordings', () => {
    const withRecs: RolesResponse = { ...NO_ROLES, bestKernel: 'older-with-recordings' };
    expect(resolveFeaturedRun(entries, withRecs)).toEqual({ runId: 'older-with-recordings', source: 'bestKernel' });
    const without: RolesResponse = { ...NO_ROLES, bestKernel: 'no-recordings' };
    expect(resolveFeaturedRun(entries, without)).toEqual({ runId: 'newest-complete', source: 'newest' });
  });

  test('otherwise the newest COMPLETE run with recordings, not the newest overall', () => {
    expect(resolveFeaturedRun(entries, null)).toEqual({ runId: 'newest-complete', source: 'newest' });
  });

  test('an incomplete run with recordings still beats an empty page', () => {
    const only = [entry('running', { complete: false, startedAt: '2026-08-04T00:00:00Z' })];
    expect(resolveFeaturedRun(only, null)).toEqual({ runId: 'running', source: 'newest' });
  });

  test('nothing playable and nothing asked for is the honest empty state', () => {
    expect(resolveFeaturedRun([entry('a', { hasRecordings: false })], NO_ROLES)).toEqual({
      runId: null,
      source: null,
    });
    expect(resolveFeaturedRun([], null)).toEqual({ runId: null, source: null });
    expect(RECORD_RECIPE).toContain('OAK_RECORD_ACTIONS=1');
  });
});

describe('URL round-trip', () => {
  test('every field survives href -> parse', () => {
    const params: WallParams = {
      run: 'wall-reference-tierL-20260804',
      env: 'DeepSea-cont-v0',
      seed: 37,
      at: 4200,
      speed: 16,
      sort: 'retention',
    };
    expect(parseWallParams(wallHref(params))).toEqual(params);
  });

  test('defaults are omitted from the href and re-appear as absent', () => {
    expect(wallHref({ run: 'r', env: 'HalfCheetah-v4', at: 0, speed: 1, sort: 'rate' }))
      .toBe('#/?run=r&env=HalfCheetah-v4');
    expect(parseWallParams('#/?run=r&env=HalfCheetah-v4')).toEqual({ run: 'r', env: 'HalfCheetah-v4' });
  });

  test('an empty wall is just #/', () => {
    expect(wallHref()).toBe('#/');
    expect(parseWallParams('#/')).toEqual({});
    expect(parseWallParams('')).toEqual({});
  });

  test('run ids are encoded, so a name with a slash cannot forge a route', () => {
    const href = wallHref({ run: 'weird/../name' });
    expect(href).toBe('#/?run=weird%2F..%2Fname');
    expect(parseWallParams(href).run).toBe('weird/../name');
  });

  test('bad fields are dropped, never corrected', () => {
    const parsed = parseWallParams('#/?env=NotAnEnv&seed=-3&speed=7&sort=colour&at=abc');
    expect(parsed).toEqual({});
  });

  test('at=0 is absent rather than a scrub to the top', () => {
    expect(parseWallParams('#/?at=0').at).toBe(0);
    expect(wallHref({ at: 0 })).toBe('#/');
  });

  test('only the listed speed presets are accepted', () => {
    for (const speed of WALL_SPEEDS) {
      expect(parseWallParams(`#/?speed=${speed}`).speed).toBe(speed);
    }
    expect(parseWallParams('#/?speed=2').speed).toBeUndefined();
  });
});

describe('?at= survives its own write', () => {
  // The three measured defects, driven through the REAL clock and step helpers with
  // no frame in between — which is exactly the window every one of them lived in.
  const FRAMES = 100_000;
  const CONTROL_HZ = 20;
  const STRIDE = wallStride(FRAMES);

  const href = (clock: number | null, pending: number): string =>
    wallHref({
      run: 'frontend-demo-16env-20260803',
      env: 'DeepSea-cont-v0',
      seed: 19,
      sort: 'retention',
      ...(urlPosition(clock, pending) !== null ? { at: urlPosition(clock, pending) as number } : {}),
    });

  test('a deep-linked at is written back before any host exists, not erased', () => {
    // page load: parse ?at=1234, no scene host yet — the first syncUrl runs here
    const pending = parseWallParams('#/?run=r&env=DeepSea-cont-v0&seed=19&at=1234&sort=retention').at ?? 0;
    expect(pending).toBe(1234);
    expect(urlPosition(null, pending)).toBe(1234);
    expect(href(null, pending)).toContain('&at=1234');
  });

  test('an arrow key writes where the player IS, not where it was', () => {
    const clock = makePlaybackClock(FRAMES, CONTROL_HZ);
    let pending = 0;
    // one ArrowRight: 20 frames × stride, seeked and serialized in the same turn
    clock.seek(stepPosition(clock.position(), 20 * STRIDE, FRAMES));
    expect(Math.round(clock.position())).toBe(20 * STRIDE);
    expect(urlPosition(clock.position(), pending)).toBe(20 * STRIDE);
    // three more, still with no frame between the seek and the URL write
    for (let i = 0; i < 3; i++) clock.seek(stepPosition(clock.position(), 20 * STRIDE, FRAMES));
    expect(urlPosition(clock.position(), pending)).toBe(80 * STRIDE);
    // and the stale cached fraction can no longer win: pending is ignored outright
    pending = 50;
    expect(urlPosition(clock.position(), pending)).toBe(80 * STRIDE);
  });

  test('the round trip is closed: what a seek writes is what a reload parses', () => {
    const clock = makePlaybackClock(FRAMES, CONTROL_HZ);
    clock.seek(80_594);
    const written = href(clock.position(), 0);
    expect(parseWallParams(written).at).toBe(80_594);
    expect(parseWallParams(written).seed).toBe(19);
    expect(parseWallParams(written).sort).toBe('retention');
  });

  test('position 0 stays out of the URL, and a clampless clock cannot smuggle one in', () => {
    const clock = makePlaybackClock(FRAMES, CONTROL_HZ);
    expect(urlPosition(clock.position(), 0)).toBe(null);
    clock.seek(-40);
    expect(urlPosition(clock.position(), 1234)).toBe(null); // a mounted clock wins, even at 0
    expect(urlPosition(Number.NaN, 900)).toBe(900);
    expect(urlPosition(null, 0)).toBe(null);
  });
});

describe('the sampling grid cannot phase-lock', () => {
  // The defect this pins: `ceil(steps/target)` gave 42 on a 100k lane and 105 on a
  // capped 1M one, both exact multiples of DeepSea's 7-step descent cycle, so every
  // sample landed on the same phase — measured against the live endpoint, 3 distinct
  // poses in 2382 samples (and 3 in 954 at stride 105) while 43 and 107 gave 28/27.
  const CYCLES = Array.from({ length: 63 }, (_, i) => i + 2); // every cycle length 2..64
  /** the exact periods the app's periodic environments have: DeepSea 7, the chains 16 and 32 */
  const REAL_CYCLES = [7, 16, 32];

  const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));

  test('every horizon gets a prime stride, so it divides no cycle below itself', () => {
    for (const steps of [10_000, 100_000, 250_000, 1_000_000, 10_000_000]) {
      expect({ steps, prime: isPrime(wallStride(steps)) }).toEqual({ steps, prime: true });
    }
  });

  test('past 64 × the sample budget that is coprimality with the WHOLE 2..64 range', () => {
    // need > 64 ⟺ steps > 64 × WALL_TARGET_SAMPLES; every root the wall ships against
    // (100k, the 250k hero cap, 1M) is here, and a prime above 64 divides nothing in it
    for (const steps of [200_000, 250_000, 1_000_000, 10_000_000]) {
      const stride = wallStride(steps);
      expect(stride).toBeGreaterThan(64);
      for (const cycle of CYCLES) {
        expect({ steps, stride, cycle, gcd: gcd(stride, cycle) })
          .toEqual({ steps, stride, cycle, gcd: 1 });
      }
    }
  });

  test('and below it the prime is still coprime to every period an environment HAS', () => {
    // Under 153,600 steps the prime can be small enough to be a cycle length itself —
    // only 1 and the primes above 64 are coprime to all of 2..64, and forcing p > 64 on
    // a 10k root would throw away 92% of its samples. What must hold is the real thing:
    // no aliasing against a period that actually exists in these environments.
    for (const steps of [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 153_600]) {
      const stride = wallStride(steps);
      for (const cycle of REAL_CYCLES) {
        expect({ steps, stride, cycle, gcd: gcd(stride, cycle) })
          .toEqual({ steps, stride, cycle, gcd: 1 });
      }
    }
  });

  test('the two strides that were measured aliasing are gone', () => {
    expect(wallStride(100_000)).toBe(43); // was 42 = 6 × 7
    expect(wallStride(250_000)).toBe(107); // was 105 = 15 × 7
  });

  test('a lane short enough to sample whole is stride 1 — no grid, nothing to alias', () => {
    expect(wallStride(500)).toBe(1);
    expect(wallStride(WALL_TARGET_SAMPLES)).toBe(1);
    expect(wallStride(0)).toBe(1);
    expect(wallStride(Number.NaN)).toBe(1);
  });

  test('primality is the mechanism, and it is exact', () => {
    expect([2, 3, 5, 7, 43, 107, 419].every(isPrime)).toBe(true);
    expect([1, 4, 42, 105, 121, 1].some(isPrime)).toBe(false);
    expect(nextPrime(42)).toBe(43);
    expect(nextPrime(43)).toBe(43);
    expect(nextPrime(105)).toBe(107);
    expect(nextPrime(1)).toBe(2);
  });
});

describe('sort as an index permutation', () => {
  const rows: SeedRow[] = [
    row(0, { rate: 0.1, retention: 0.9 }),
    row(1, { rate: 0.4, retention: 0.2 }),
    row(2, { rate: null, retention: null }),
    row(3, { rate: 0.4, retention: 0.5 }),
    row(4, { rate: -0.2, retention: 1 }),
  ];

  test('it returns indices into the ORIGINAL array, never a reordered copy', () => {
    const order = sortPermutation(rows, 'rate');
    expect(order).toHaveLength(rows.length);
    expect([...order].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4]);
    expect(order.map((i) => rows[i]?.seed)).toEqual([1, 3, 0, 4, 2]);
  });

  test('ties keep input order and nulls sink — the same input gives the same permutation', () => {
    const first = sortPermutation(rows, 'rate');
    const again = sortPermutation(rows, 'rate');
    expect(again).toEqual(first);
    // seeds 1 and 3 tie at 0.4 and stay in input order
    expect(first.indexOf(1)).toBeLessThan(first.indexOf(3));
    // the null-rate seed is last
    expect(first[first.length - 1]).toBe(2);
  });

  test('seed sorts ascending; rate and retention descending', () => {
    expect(sortPermutation(rows, 'seed').map((i) => rows[i]?.seed)).toEqual([0, 1, 2, 3, 4]);
    expect(sortPermutation(rows, 'retention').map((i) => rows[i]?.seed)).toEqual([4, 0, 3, 1, 2]);
  });

  test('sorting never mutates the rows', () => {
    const snapshot = rows.map((r) => r.seed);
    sortPermutation(rows, 'retention');
    expect(rows.map((r) => r.seed)).toEqual(snapshot);
  });

  test('the sort key cycles through the three published orders', () => {
    let sort: WallSort = 'rate';
    const seen: WallSort[] = [sort];
    for (let i = 0; i < WALL_SORTS.length - 1; i++) {
      sort = nextSort(sort);
      seen.push(sort);
    }
    expect(seen).toEqual([...WALL_SORTS]);
    expect(nextSort(seen[seen.length - 1] as WallSort)).toBe('rate');
  });
});

describe('promote / Esc state machine', () => {
  const playable = (seed: number): boolean => seed % 2 === 0;
  const base = { leader: 4, promoted: null };

  test('the hero is the promotion, or the leader when there is none', () => {
    expect(heroSeed(base)).toBe(4);
    expect(heroSeed({ leader: 4, promoted: 8 })).toBe(8);
    expect(heroSeed(EMPTY_SELECTION)).toBe(null);
  });

  test('promoting a playable seed swaps the hero and nothing else', () => {
    const next = promoteSeed(base, 8, playable);
    expect(next).toEqual({ leader: 4, promoted: 8 });
    expect(base.promoted).toBe(null);
  });

  test('a seed with no recording cannot be promoted — the state is returned unchanged', () => {
    expect(promoteSeed(base, 7, playable)).toBe(base);
  });

  test('promoting the same seed twice is a no-op (identity is the repaint guard)', () => {
    const once = promoteSeed(base, 8, playable);
    expect(promoteSeed(once, 8, playable)).toBe(once);
  });

  test('Esc returns to the leader and is idempotent', () => {
    const promoted = promoteSeed(base, 8, playable);
    const escaped = escapeSelection(promoted);
    expect(escaped).toEqual({ leader: 4, promoted: null });
    expect(heroSeed(escaped)).toBe(4);
    expect(escapeSelection(escaped)).toBe(escaped);
  });

  test('a new leader keeps a promotion only while that seed is still playable', () => {
    const promoted = promoteSeed(base, 8, playable);
    expect(setLeader(promoted, 2, playable)).toEqual({ leader: 2, promoted: 8 });
    // switching to an env where seed 8 has no recording drops the promotion
    expect(setLeader(promoted, 2, (seed) => seed === 2)).toEqual({ leader: 2, promoted: null });
  });

  test('the leader is the best-rate seed that can actually play', () => {
    expect(leaderSeed([
      row(0, { rate: 0.9, recorded: false }),
      row(1, { rate: 0.4 }),
      row(2, { rate: 0.7 }),
      row(3, { rate: null }),
    ])).toBe(2);
    expect(leaderSeed([row(0, { rate: 0.9, recorded: false })])).toBe(null);
    expect(leaderSeed([])).toBe(null);
  });
});

describe('the degraded wall', () => {
  test('a run with no recordings says so, per run and per environment', () => {
    expect(degradedReason([], 'HalfCheetah-v4')).toBe('no recordings on this run');
    const recs = [recording('HalfCheetah-v4', 3)];
    expect(degradedReason(recs, 'HalfCheetah-v4')).toBe(null);
    expect(degradedReason(recs, 'DeepSea-cont-v0')).toBe(`no recordings for ${ENVS['DeepSea-cont-v0'].name} in this run`);
    expect(degradedReason(recs, null)).toBe('no environment selected');
  });
});

describe('the seed rail', () => {
  const columnar: FleetColumnar = {
    steps: [1000, 2000, 3000],
    lanes: [
      { lane: 0, seed: 0, policy: 'oak', envId: 'HalfCheetah-v4' },
      { lane: 1, seed: 1, policy: 'oak', envId: 'HalfCheetah-v4' },
      { lane: 2, seed: 2, policy: 'oak', envId: 'HalfCheetah-v4' },
    ],
    series: {
      // cumulative reward: seed 0 climbs, seed 1 peaks then fades, seed 2 dies
      rewardSum: [
        [100, 400, 900],
        [300, 600, 610],
        [50, null, null],
      ],
      firstNonFiniteStep: [
        [null, null, null],
        [null, null, null],
        [null, 1500, 1500],
      ],
    },
  };

  test('rate is the AVERAGE REWARD RATE — final cumulative reward ÷ steps', () => {
    const rows = buildSeedRows(columnar);
    expect(rows.map((r) => r.seed)).toEqual([0, 1, 2]);
    expect(rows[0]?.rate).toBeCloseTo(900 / 3000, 12);
    expect(rows[1]?.rate).toBeCloseTo(610 / 3000, 12);
    // seed 2 only ever reported at step 1000
    expect(rows[2]?.rate).toBeCloseTo(50 / 1000, 12);
  });

  test('the sparkline is the MEASURED windowed rate, not the cumulative total', () => {
    const rows = buildSeedRows(columnar);
    expect(rows[0]?.spark).toEqual([0.1, 0.3, 0.5]);
    expect(rows[1]?.spark).toEqual([0.3, 0.3, 0.01]);
  });

  test('retention is final ÷ peak of that windowed rate', () => {
    const rows = buildSeedRows(columnar);
    expect(rows[0]?.retention).toBeCloseTo(0.5 / 0.5, 12);
    expect(rows[1]?.retention).toBeCloseTo(0.01 / 0.3, 12);
    // a seed that never had a positive peak has no meaningful ratio
    expect(retentionOf([-0.4, -0.2])).toBe(null);
    expect(retentionOf([])).toBe(null);
  });

  test('a non-null firstNonFiniteStep marks the seed numerically diverged', () => {
    const rows = buildSeedRows(columnar);
    expect(rows[0]?.divergedAt).toBe(null);
    expect(rows[2]?.divergedAt).toBe(1500);
  });

  test('only seeds with a recording are promotable', () => {
    const rows = buildSeedRows(columnar, { recorded: new Set([1]) });
    expect(rows.map((r) => r.recorded)).toEqual([false, true, false]);
  });

  test('the sparkline budget is a stated cap, not an accident', () => {
    expect(SPARK_ROW_BUDGET).toBeGreaterThanOrEqual(64);
    expect(SPARK_ROW_BUDGET).toBeLessThan(1024);
  });
});

describe('the population header', () => {
  const rows = [
    row(0, { rate: 0.4, retention: 0.9 }),
    row(1, { rate: 0.2, retention: 0.4 }),
    row(2, { rate: -0.1, retention: null, divergedAt: 40_000 }),
    row(3, { rate: 0.3, retention: 0.8 }),
  ];

  test('summary.json wins and carries its CI', () => {
    const slice: SummaryPolicyResult = {
      policy: 'oak',
      seeds: 64,
      rewardRateIQM: 0.3872,
      rewardRateIQMCI95: [0.31, 0.46],
      positiveFraction: 0.995,
      retentionFraction: 0.664,
      divergence: { diverged: 2 },
    };
    const figures = populationFigures(slice, rows);
    expect(figures.source).toBe('summary');
    expect(figures.iqm).toBe('+0.3872');
    expect(figures.ci).toBe('[+0.310, +0.460]');
    expect(figures.positive).toBe('100%');
    expect(figures.nonDiverged).toBe('62/64');
    expect(figures.retention).toBe('66%');
  });

  test('without a summary slice the same four figures are MEASURED and say so', () => {
    const figures = populationFigures(null, rows);
    expect(figures.source).toBe('measured');
    expect(figures.ci).toBe(null);
    expect(figures.positive).toBe('75%');
    expect(figures.nonDiverged).toBe('3/4');
    expect(figures.retention).toBe('50%');
    expect(figures.iqm).toBe(`+${iqmOf([0.4, 0.2, -0.1, 0.3])?.toFixed(4)}`);
  });

  test('no seeds at all is dashes, never zeros', () => {
    const figures = populationFigures(null, []);
    expect(figures).toEqual({ iqm: '—', ci: null, positive: '—', nonDiverged: '—', retention: '—', source: 'none' });
  });

  test('the IQM is the interquartile mean', () => {
    expect(iqmOf([1, 2, 3, 4, 5, 6, 7, 8])).toBeCloseTo((3 + 4 + 5 + 6) / 4, 12);
    expect(iqmOf([])).toBe(null);
  });
});

describe('run identity line', () => {
  test('a mixed fleet counts tasks and seeds per task', () => {
    expect(runScheduleLine({
      envId: 'HalfCheetah-v4',
      batch: 128,
      steps: 100_000,
      envSchedule: [
        { envId: 'HalfCheetah-v4', seedStart: 0, seedEnd: 7, flopsPerLaneStep: 0 },
        { envId: 'DeepSea-cont-v0', seedStart: 8, seedEnd: 15, flopsPerLaneStep: 0 },
      ],
    })).toBe('2 tasks × 8 seeds × 100k steps');
  });

  test('an uneven schedule reports the fleet total rather than inventing a split', () => {
    expect(runScheduleLine({
      envId: 'HalfCheetah-v4',
      batch: 1408,
      steps: 1_000_000,
      envSchedule: [
        { envId: 'HalfCheetah-v4', seedStart: 0, seedEnd: 1023, flopsPerLaneStep: 0 },
        { envId: 'DeepSea-cont-v0', seedStart: 1024, seedEnd: 1087, flopsPerLaneStep: 0 },
      ],
    })).toBe('2 tasks × 1,408 seeds total × 1M steps');
  });

  test('a single-env run names the environment', () => {
    expect(runScheduleLine({ envId: 'HalfCheetah-v4', batch: 1024, steps: 1_000_000 }))
      .toBe('HalfCheetah · 1,024 seeds · 1M steps');
  });

  test('a 1M lane strides prime and coprime to every gridworld cycle', () => {
    expect(wallStride(1_000_000)).toBe(419); // prime > 64 — coprime to all of 2..64
  });
});

describe('environment strip', () => {
  const meta = {
    envId: 'HalfCheetah-v4',
    batch: 16,
    envSchedule: [
      { envId: 'DeepSea-cont-v0', seedStart: 8, seedEnd: 15, flopsPerLaneStep: 0 },
      { envId: 'HalfCheetah-v4', seedStart: 0, seedEnd: 7, flopsPerLaneStep: 0 },
    ],
  };

  test('chips follow the canonical env order, never the schedule order', () => {
    expect(wallEnvs(meta, [])).toEqual(['HalfCheetah-v4', 'DeepSea-cont-v0']);
  });

  test('an env that only appears in the recordings still gets a chip', () => {
    expect(wallEnvs(meta, [recording('CartPole-cont-v0', 0)]))
      .toEqual(['HalfCheetah-v4', 'DeepSea-cont-v0', 'CartPole-cont-v0']);
  });

  test('[ and ] wrap in both directions', () => {
    const list = wallEnvs(meta, []);
    expect(cycleEnv(list, 'HalfCheetah-v4', 1)).toBe('DeepSea-cont-v0');
    expect(cycleEnv(list, 'DeepSea-cont-v0', 1)).toBe('HalfCheetah-v4');
    expect(cycleEnv(list, 'HalfCheetah-v4', -1)).toBe('DeepSea-cont-v0');
    expect(cycleEnv(list, null, 1)).toBe('HalfCheetah-v4');
    expect(cycleEnv([], null, 1)).toBe(null);
  });
});

describe('keyboard', () => {
  test('the published keymap', () => {
    expect(wallKeyAction(' ')).toEqual({ kind: 'toggle' });
    expect(wallKeyAction('ArrowRight')).toEqual({ kind: 'step', frames: 20 });
    expect(wallKeyAction('ArrowLeft', true)).toEqual({ kind: 'step', frames: -200 });
    expect(wallKeyAction('[')).toEqual({ kind: 'env', direction: -1 });
    expect(wallKeyAction(']')).toEqual({ kind: 'env', direction: 1 });
    expect(wallKeyAction('s')).toEqual({ kind: 'sort' });
    expect(wallKeyAction('g')).toEqual({ kind: 'debug' });
    expect(wallKeyAction('Escape')).toEqual({ kind: 'escape' });
    expect(wallKeyAction('q')).toBe(null);
  });
});

describe('driving the ghosts', () => {
  test('a record-frame position floors onto a strided sample', () => {
    expect(ghostSampleIndex(0, 42, 100)).toBe(0);
    expect(ghostSampleIndex(41.9, 42, 100)).toBe(0);
    expect(ghostSampleIndex(42, 42, 100)).toBe(1);
    expect(ghostSampleIndex(-5, 42, 100)).toBe(0);
  });

  test('past the last sample it holds the final one — never an index off the end', () => {
    expect(ghostSampleIndex(1e9, 42, 100)).toBe(99);
    expect(ghostSampleIndex(10, 1, 0)).toBe(0);
  });

  test('a seed is alive until its first non-finite ROW, then frozen', () => {
    expect(ghostAlive(null, 999_999)).toBe(true);
    expect(ghostAlive(500, 499)).toBe(true);
    expect(ghostAlive(500, 500)).toBe(false);
    expect(ghostAlive(500, 900)).toBe(false);
  });

  test('the colour ramp is fed a population QUANTILE, not a raw rate', () => {
    // one rocket seed would otherwise flatten every other ghost onto the floor
    const quantiles = rateQuantiles([
      row(0, { rate: 0.01 }),
      row(1, { rate: 0.02 }),
      row(2, { rate: 0.03 }),
      row(3, { rate: 40 }),
    ]);
    expect(quantiles.get(0)).toBe(0);
    expect(quantiles.get(1)).toBeCloseTo(1 / 3, 12);
    expect(quantiles.get(2)).toBeCloseTo(2 / 3, 12);
    expect(quantiles.get(3)).toBe(1);
  });

  test('the population that does not fit is counted, never claimed', () => {
    // Real numbers, from the wall endpoint on frontend-demo-16env-20260803
    // (HalfCheetah, 8 seeds, pose[0] = rootx): the spread across seeds is 0.18 units
    // at step 0, 0.84 at 5k, 10.09 at 25k, 221.56 at 50k, 1747.65 at 100k. The stage
    // frames about five metres. So most of the run has ONE animal on it, and the
    // chip's number has to be the measured one.
    const window = trackWindow(16 / 9);
    expect(window.width).toBeCloseTo(2 * CHEETAH_CAM_DEPTH * Math.tan((TOKENS.fovHero * Math.PI) / 360) * (16 / 9), 9);
    expect(window.ahead - window.back).toBeCloseTo(2 * CHEETAH_CAM_LOOK_X, 12);
    expect(window.width).toBeGreaterThan(3);
    expect(window.width).toBeLessThan(8);

    const heroX = 236.1;
    // step 50k, the frame the review screenshotted: 8 seeds over 221.56 units. Only the
    // hero and the one seed 0.7 units behind it are inside a ~4.5 m frame; the seed at
    // 233.9 is already 2.2 m back, past the window's short edge.
    const spread = [14.6, 60.2, 110.9, 158.4, 200.0, 233.9, 235.4, 236.1];
    expect(inFrameCount(spread, spread.length, heroX, window)).toBe(2);
    expect(window.back).toBeLessThan(2.2);
    // step 0, the same fleet 0.18 units apart: everything is genuinely on stage
    const together = [-0.1, -0.05, 0, 0.02, 0.04, 0.05, 0.06, 0.08];
    expect(inFrameCount(together, together.length, 0, window)).toBe(8);
    // a seed that went non-finite contributes nothing rather than counting as here
    expect(inFrameCount([Number.NaN, 0], 2, 0, window)).toBe(1);
    expect(inFrameCount([0], 1, Number.NaN, window)).toBe(0);
  });

  test('the window is asymmetric because the camera looks ahead of the runner', () => {
    const window = trackWindow(2);
    // a seed the hero has just overtaken drops out of frame sooner than one ahead
    expect(inFrameCount([-window.back + 0.01], 1, 0, window)).toBe(1);
    expect(inFrameCount([-window.back - 0.01], 1, 0, window)).toBe(0);
    expect(inFrameCount([window.ahead - 0.01], 1, 0, window)).toBe(1);
    expect(inFrameCount([window.ahead + 0.01], 1, 0, window)).toBe(0);
    // a wider stage sees more track, in exact proportion
    expect(trackWindow(4).width).toBeCloseTo(2 * trackWindow(2).width, 9);
    expect(trackWindow(0).width).toBeCloseTo(trackWindow(16 / 9).width, 9);
    // and however narrow the stage gets, the hero the rig is centred on is in its own
    // frame — the count can never report the animal on screen as off it
    for (const aspect of [0.2, 0.5, 1, 16 / 9, 3]) {
      const w = trackWindow(aspect);
      expect(w.back).toBeGreaterThanOrEqual(0);
      expect(inFrameCount([0], 1, 0, w)).toBe(1);
    }
  });

  test('only an unbounded track needs the field strip', () => {
    expect(stageFraming('cheetah')).toBe('track');
    expect(stageFraming('grid')).toBe('world');
    expect(stageFraming('classic')).toBe('world');
  });

  test('the field strip carries the whole extent AND the stage window, to one scale', () => {
    const window = trackWindow(16 / 9);
    const xs = [123.5, 400, 900, 1500, 1871.2];
    const extent = fieldExtent(xs, xs.length);
    expect(extent).toEqual({ min: 123.5, max: 1871.2, span: 1747.7 });

    const scale = fieldScale(extent, 1871.2, window);
    // every seed lands inside the strip, the leader at the far end
    for (const x of xs) {
      const p = fieldPercent(x, scale);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(100);
    }
    expect(fieldPercent(xs[0] as number, scale)).toBeLessThan(fieldPercent(xs[4] as number, scale));
    // and the camera's own slice is a hairline at this scale — which is the point
    const left = fieldPercent(1871.2 - window.back, scale);
    const right = fieldPercent(1871.2 + window.ahead, scale);
    expect(right - left).toBeLessThan(1);

    // early in the run the window is WIDER than the fleet, so it fills the strip
    const tight = fieldExtent([-0.1, 0.08], 2);
    const early = fieldScale(tight, 0, window);
    const earlyLeft = fieldPercent(-window.back, early);
    const earlyRight = fieldPercent(window.ahead, early);
    expect(earlyRight - earlyLeft).toBeGreaterThan(80);
  });

  test('the field degrades without inventing a position', () => {
    expect(fieldExtent([Number.NaN, Number.NaN], 2)).toBe(null);
    expect(fieldExtent([], 0)).toBe(null);
    // no extent and no hero: a strip with a domain, not a divide-by-zero
    const scale = fieldScale(null, null, trackWindow(16 / 9));
    expect(scale.max).toBeGreaterThan(scale.min);
    expect(fieldPercent(Number.NaN, scale)).toBe(50);
    expect(fieldPercent(0, { min: 5, max: 5 })).toBe(50);
    expect(fieldPercent(-1e9, { min: 0, max: 10 })).toBe(0);
    expect(fieldPercent(1e9, { min: 0, max: 10 })).toBe(100);
  });

  test('a lane reads its track position from the same pose slot the ghost is drawn at', () => {
    const lane = { pose: new Float32Array([1, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0]), samples: 2 };
    expect(laneTrackX(lane, 0)).toBe(1);
    expect(laneTrackX(lane, 1)).toBe(7);
    // past the end it HOLDS the final sample, exactly as the ghost does
    expect(laneTrackX(lane, 99)).toBe(7);
    expect(laneTrackX(lane, -3)).toBe(1);
    expect(laneTrackX({ pose: new Float32Array(0), samples: 0 }, 0)).toBe(null);
    expect(laneTrackX({ pose: new Float32Array([Number.NaN]), samples: 1 }, 0)).toBe(null);
  });

  test('distances are quoted in the units the reader is looking at', () => {
    expect(formatDistance(0.18)).toBe('0.18 m');
    expect(formatDistance(0.84)).toBe('0.84 m');
    expect(formatDistance(10.09)).toBe('10 m');
    expect(formatDistance(221.56)).toBe('222 m');
    expect(formatDistance(1747.65)).toBe('1.7 km');
    expect(formatDistance(12_400)).toBe('12 km');
    expect(formatDistance(Number.NaN)).toBe('—');
  });

  test('ties share a rank and rate-less seeds sit at the dark end', () => {
    const quantiles = rateQuantiles([
      row(0, { rate: 0.5 }),
      row(1, { rate: 0.5 }),
      row(2, { rate: null }),
      row(3, { rate: 0.9 }),
    ]);
    expect(quantiles.get(0)).toBe(0);
    expect(quantiles.get(1)).toBe(0);
    expect(quantiles.get(2)).toBe(0);
    expect(quantiles.get(3)).toBe(1);
    expect(rateQuantiles([]).size).toBe(0);
    expect(rateQuantiles([row(7, { rate: null })]).get(7)).toBe(0);
  });
});
