/// <reference types="bun" />
// Learning-curve endpoint tests.
//
// The fixture root is built in a temp directory rather than under test/fixtures, so it
// cannot collide with the fixture tree api.test.ts rebuilds in its own beforeAll (bun
// runs both files in one process). Its numbers are chosen to be checkable by hand:
// every seed holds a CONSTANT reward rate, so each heartbeat's population is an exact
// small set whose IQM and percentiles were cross-checked against numpy (the same
// `np.percentile` linear interpolation and the same trimming rule as tools/fleet.py).
//
// The real roots are used for the shape-and-scale claims (16 envs, 6 envs, 1408 seeds)
// and skipped when .oak-benchmarks is absent.

import { afterAll, beforeAll, describe, expect, it } from 'bun:test';
import { existsSync } from 'node:fs';
import { mkdtemp, mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tapeBytes } from '../helpers/tape';

import type { CurvesResponse } from '../../shared/types';
import { createApp } from '../../server/app';
import {
  bootstrapIqmCi,
  clearCurvesCache,
  deriveCurves,
  iqm,
  iqmSorted,
  makeRng,
  parseCurvesQuery,
  percentileSorted,
  type CurvesQuery,
} from '../../server/curves';

const HERE = dirname(fileURLToPath(import.meta.url));
const REAL_BENCH = resolve(HERE, '..', '..', '..', '.oak-benchmarks');
const DEMO_ID = 'frontend-demo-16env-20260805';
const KEEPER_ID = 'tape-reference-tierL-20260805';
const demoPresent = existsSync(join(REAL_BENCH, DEMO_ID, 'fleet.jsonl'));
const keeperPresent = existsSync(join(REAL_BENCH, KEEPER_ID, 'metrics.oak.oakt'));

const ENV_A = 'HalfCheetah-v4';
const ENV_B = 'FourRooms-cont-v0';
const STEPS = [100, 200, 300] as const;

/** every seed holds one constant rate, so rewardSum at step s is simply rate·s */
const RATES: { lane: number; envId: string; rate: number }[] = [
  { lane: 0, envId: ENV_A, rate: 1 },
  { lane: 1, envId: ENV_A, rate: 2 },
  { lane: 2, envId: ENV_A, rate: 3 },
  { lane: 3, envId: ENV_A, rate: 4 },
  { lane: 4, envId: ENV_B, rate: -0.5 },
  { lane: 5, envId: ENV_B, rate: 0.5 },
];

let BENCH = '';
let RUN_DIR = '';
const RUN_ID = 'curves-fixture';
let CACHE = '';

function row(lane: number, envId: string, step: number, rewardSum: number, policy = 'oak'): string {
  return JSON.stringify({
    v: 1,
    kind: 'final',
    envId,
    policy,
    lane,
    seed: lane,
    step,
    wallSeconds: step / 100,
    rewardSum,
    doneCount: 0,
    health: { rewardRate: rewardSum / step },
  });
}

async function buildFixture(): Promise<void> {
  BENCH = await mkdtemp(join(tmpdir(), 'oak-curves-'));
  RUN_DIR = join(BENCH, RUN_ID);
  CACHE = join(BENCH, 'cache');
  await mkdir(RUN_DIR, { recursive: true });
  await writeFile(
    join(RUN_DIR, 'meta.json'),
    JSON.stringify({
      fleetSchemaVersion: 1,
      envId: ENV_A,
      batch: 6,
      seeds: [0, 1, 2, 3, 4, 5],
      steps: 300,
      policies: ['oak', 'zero'],
      ablationLabel: 'none',
      startedAt: '2026-08-04T10:00:00Z',
      envSchedule: [
        { envId: ENV_A, seedStart: 0, seedEnd: 3, flopsPerLaneStep: 230244 },
        { envId: ENV_B, seedStart: 4, seedEnd: 5, flopsPerLaneStep: 210284 },
      ],
    }),
  );

  // METRICS LAW (2026-08-05): fleet.jsonl carries final certificates only;
  // the heartbeat series live on per-policy tapes.
  const finalStep = STEPS[STEPS.length - 1] ?? 300;
  const lines: string[] = [];
  for (const seed of RATES) {
    lines.push(row(seed.lane, seed.envId, finalStep, seed.rate * finalStep));
  }
  // a second policy in the same file — the filter has to keep them apart
  // (a tape covers EVERY lane, so the zero policy carries all 6 at rate 9)
  for (const seed of RATES) {
    lines.push(row(seed.lane, seed.envId, finalStep, 9 * finalStep, 'zero'));
  }
  // ...and a half-written trailing row, exactly as a live writer leaves one
  await writeFile(join(RUN_DIR, 'fleet.jsonl'), `${lines.join('\n')}\n{"v":1,"kind":"final","lane":2,"step":4`);
  await writeFile(join(RUN_DIR, 'aggregate.jsonl'), '');

  const tapeSteps = STEPS.slice(0, -1); // finals own the last step
  const oakWindows = tapeSteps.map((end, i) => ({
    start: i === 0 ? 0 : (tapeSteps[i - 1] ?? 0),
    count: end - (i === 0 ? 0 : (tapeSteps[i - 1] ?? 0)),
    values: {
      // lane 3's window at 200 is unknown (NaN), and so is the 300 window that
      // would have to difference against it — the old missing-row case.
      rewardSum: RATES.map((s) => (s.lane === 3 && end === 200 ? Number.NaN : s.rate * end)),
    },
  }));
  await writeFile(join(RUN_DIR, 'metrics.oak.oakt'), tapeBytes(RATES.length, oakWindows));
  await writeFile(
    join(RUN_DIR, 'metrics.zero.oakt'),
    tapeBytes(RATES.length, tapeSteps.map((end, i) => ({
      start: i === 0 ? 0 : (tapeSteps[i - 1] ?? 0),
      count: end - (i === 0 ? 0 : (tapeSteps[i - 1] ?? 0)),
      values: { rewardSum: RATES.map(() => 9 * end) },
    }))),
  );
}

function query(search: string): CurvesQuery {
  const parsed = parseCurvesQuery(new URL(`http://127.0.0.1/api/runs/x/curves${search}`));
  if ('error' in parsed) throw new Error(`unexpected query error: ${parsed.error}`);
  return parsed.query;
}

function curves(search: string, opts: { cacheDir?: string | null } = {}): Promise<CurvesResponse> {
  return deriveCurves({
    id: RUN_ID,
    dir: RUN_DIR,
    query: query(search),
    inFlight: false,
    cacheDir: opts.cacheDir ?? null,
  });
}

/** every series of a curve has one entry per heartbeat — a chart cannot align otherwise */
function expectAligned(curve: CurvesResponse): void {
  const n = curve.steps.length;
  for (const key of ['iqm', 'ciLo', 'ciHi', 'p10', 'p50', 'p90'] as const) {
    expect(curve[key].length).toBe(n);
  }
}

beforeAll(async () => {
  await buildFixture();
  clearCurvesCache();
});

afterAll(async () => {
  await rm(BENCH, { recursive: true, force: true });
});

// ---------- statistics ----------

describe('curve statistics', () => {
  it('iqm is tools/fleet.py\'s: sort, drop ⌊n/4⌋ from each end, mean the rest', () => {
    expect(iqm([4, 1, 3, 2])).toBeCloseTo(2.5, 12); // n=4 -> middle two
    expect(iqm([3, 1, 2])).toBeCloseTo(2, 12); // n=3 -> nothing dropped
    expect(iqm([-0.5, 0.5])).toBeCloseTo(0, 12);
    expect(iqm([5])).toBeCloseTo(5, 12);
    expect(iqm([-0.5, 0.5, 1, 2, 3, 4])).toBeCloseTo(1.625, 12); // n=6 -> drop 1 each end
    // n=8: drop two each end, mean the middle four
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

  it('the PRNG is a pure function of its seed', () => {
    const a = makeRng(7);
    const b = makeRng(7);
    const c = makeRng(8);
    const first = Array.from({ length: 8 }, () => a());
    expect(Array.from({ length: 8 }, () => b())).toEqual(first);
    expect(Array.from({ length: 8 }, () => c())).not.toEqual(first);
    for (const v of first) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('the bootstrap CI is deterministic and brackets the point estimate', () => {
    const sample = Array.from({ length: 64 }, (_, i) => i / 8 - 4).sort((x, y) => x - y);
    const [lo, hi] = bootstrapIqmCi(sample, 2000, 12345);
    expect(bootstrapIqmCi(sample, 2000, 12345)).toEqual([lo, hi]);
    expect(lo).toBeLessThan(iqmSorted(sample));
    expect(hi).toBeGreaterThan(iqmSorted(sample));
    // a different seed moves the bounds a little, never to the other side of the estimate
    const [lo2, hi2] = bootstrapIqmCi(sample, 2000, 999);
    expect(lo2).toBeLessThan(iqmSorted(sample));
    expect(hi2).toBeGreaterThan(iqmSorted(sample));
    expect(Math.abs(lo2 - lo)).toBeLessThan(Math.abs(hi - lo));
  });

  it('a degenerate population gets a degenerate interval, not jitter', () => {
    expect(bootstrapIqmCi([2, 2, 2, 2, 2, 2], 200, 1)).toEqual([2, 2]);
    expect(bootstrapIqmCi([7], 200, 1)).toEqual([7, 7]);
    const [lo, hi] = bootstrapIqmCi([], 200, 1);
    expect(Number.isNaN(lo)).toBe(true);
    expect(Number.isNaN(hi)).toBe(true);
  });
});

// ---------- query parsing ----------

describe('parseCurvesQuery', () => {
  it('defaults to the oak policy and all lanes', () => {
    expect(query('')).toEqual({ policy: 'oak', envId: null });
    expect(query('?envId=')).toEqual({ policy: 'oak', envId: null });
    expect(query('?policy=zero&envId=DeepSea-cont-v0')).toEqual({ policy: 'zero', envId: 'DeepSea-cont-v0' });
  });

  it('rejects an unknown env and an unsafe policy', () => {
    const bad = parseCurvesQuery(new URL('http://x/api/runs/x/curves?envId=Nope-v9'));
    expect('error' in bad && bad.error).toContain('bad envId');
    const worse = parseCurvesQuery(new URL('http://x/api/runs/x/curves?policy=a/b'));
    expect('error' in worse && worse.error).toBe('bad policy');
  });
});

// ---------- derivation over the fixture root ----------

describe('curves from fleet.jsonl', () => {
  it('measures the reward rate over each heartbeat window, per env', async () => {
    const a = await curves(`?envId=${ENV_A}`);
    expect(a.steps).toEqual([100, 200, 300]);
    expect(a.seeds).toBe(4);
    expectAligned(a);
    // step 100: [1,2,3,4]. Steps 200 and 300: lane 3 is unmeasurable, so [1,2,3].
    expect(a.iqm.map((v) => v ?? Number.NaN)).toEqual([2.5, 2, 2]);
    expect(a.p10.map((v) => Number((v ?? 0).toFixed(12)))).toEqual([1.3, 1.2, 1.2]);
    expect(a.p50.map((v) => v ?? Number.NaN)).toEqual([2.5, 2, 2]);
    expect(a.p90.map((v) => Number((v ?? 0).toFixed(12)))).toEqual([3.7, 2.8, 2.8]);

    const b = await curves(`?envId=${ENV_B}`);
    expect(b.seeds).toBe(2);
    expect(b.iqm).toEqual([0, 0, 0]);
    expect(b.p10).toEqual([-0.4, -0.4, -0.4]);
    expect(b.p90).toEqual([0.4, 0.4, 0.4]);
  });

  it('pools every lane when no env is named', async () => {
    const all = await curves('');
    expect(all.seeds).toBe(6);
    expect(all.steps).toEqual([100, 200, 300]);
    // step 100 pools [-0.5, 0.5, 1, 2, 3, 4]
    expect(all.iqm[0]).toBeCloseTo(1.625, 12);
    expect(all.p10[0]).toBeCloseTo(0, 12);
    expect(all.p50[0]).toBeCloseTo(1.5, 12);
    expect(all.p90[0]).toBeCloseTo(3.5, 12);
  });

  it('brackets every IQM with its own CI', async () => {
    const all = await curves('');
    for (let i = 0; i < all.steps.length; i++) {
      const point = all.iqm[i];
      const lo = all.ciLo[i];
      const hi = all.ciHi[i];
      expect(point).not.toBeNull();
      expect(lo).not.toBeNull();
      expect(hi).not.toBeNull();
      expect(lo ?? 0).toBeLessThanOrEqual(point ?? 0);
      expect(hi ?? 0).toBeGreaterThanOrEqual(point ?? 0);
    }
  });

  it('keeps the policies apart', async () => {
    const zero = await curves('?policy=zero');
    expect(zero.seeds).toBe(6);
    expect(zero.iqm).toEqual([9, 9, 9]);
    const missing = await curves('?policy=random');
    expect(missing).toEqual({ steps: [], iqm: [], ciLo: [], ciHi: [], p10: [], p50: [], p90: [], seeds: 0 });
  });

  it('answers empty for a valid env with no lanes in this run', async () => {
    const none = await curves('?envId=Pendulum-cont-v0');
    expect(none.seeds).toBe(0);
    expect(none.steps).toEqual([]);
  });

  it('is reproducible: the same file always yields the same CI', async () => {
    clearCurvesCache();
    const first = await curves(`?envId=${ENV_A}`);
    clearCurvesCache();
    const second = await curves(`?envId=${ENV_A}`);
    expect(second).toEqual(first);
  });

  it('answers a missing fleet.jsonl with an empty curve, not an error', async () => {
    const empty = await deriveCurves({
      id: 'nope',
      dir: join(BENCH, 'does-not-exist'),
      query: query(''),
      inFlight: false,
      cacheDir: null,
    });
    expect(empty.seeds).toBe(0);
    expect(empty.steps).toEqual([]);
  });
});

// ---------- caching ----------

describe('curves cache', () => {
  it('shares ONE parse between the env panels of a dashboard', async () => {
    clearCurvesCache();
    const [a, b, all] = await Promise.all([
      curves(`?envId=${ENV_A}`),
      curves(`?envId=${ENV_B}`),
      curves(''),
    ]);
    expect(a.seeds).toBe(4);
    expect(b.seeds).toBe(2);
    expect(all.seeds).toBe(6);
    // the slices are distinct objects with distinct populations, from one pass
    expect(a.iqm[0]).toBeCloseTo(2.5, 12);
    expect(b.iqm[0]).toBeCloseTo(0, 12);
  });

  it('writes a disk record and refuses one whose identity does not match', async () => {
    await rm(CACHE, { recursive: true, force: true });
    clearCurvesCache();
    const first = await curves(`?envId=${ENV_A}`, { cacheDir: CACHE });
    expect(first.seeds).toBe(4);

    const files = await waitForCacheFiles(1);
    const path = join(CACHE, files[0] ?? '');
    const rec = JSON.parse(await Bun.file(path).text()) as Record<string, unknown>;
    expect(rec['v']).toBe(2);
    expect(rec['kind']).toBe('curves');
    expect(rec['envId']).toBe(ENV_A);

    // Simulate a 32-bit hash collision: right file name, wrong query inside.
    rec['envId'] = ENV_B;
    rec['data'] = { steps: [-1], iqm: [], ciLo: [], ciHi: [], p10: [], p50: [], p90: [], seeds: -1 };
    await writeFile(path, JSON.stringify(rec));
    clearCurvesCache(); // force the disk path
    const again = await curves(`?envId=${ENV_A}`, { cacheDir: CACHE });
    expect(again.seeds).toBe(4);
    expect(again.steps).toEqual([100, 200, 300]);
  });

  it('serves a cached curve without re-reading the file', async () => {
    await rm(CACHE, { recursive: true, force: true });
    clearCurvesCache();
    const first = await curves('', { cacheDir: CACHE });
    await waitForCacheFiles(1);
    clearCurvesCache();
    const second = await curves('', { cacheDir: CACHE });
    expect(second).toEqual(first);
  });
});

/** the disk record is written fire-and-forget, by design — poll for it */
async function waitForCacheFiles(count: number, ms = 5000): Promise<string[]> {
  const t0 = Date.now();
  for (;;) {
    const names = await readdir(CACHE).then(
      (list) => list.filter((n) => n.endsWith('.json')).sort(),
      () => [] as string[], // the directory itself is created by the first write
    );
    if (names.length >= count) return names;
    if (Date.now() - t0 > ms) throw new Error(`timed out waiting for ${count} cache files`);
    await Bun.sleep(10);
  }
}

// ---------- over HTTP ----------

describe('GET /api/runs/:id/curves', () => {
  let server: ReturnType<typeof Bun.serve> | null = null;
  let base = '';

  beforeAll(() => {
    const app = createApp({ benchRoot: BENCH, cacheFile: null, fleetCacheDir: null, ttlMs: 0 });
    server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: app.fetch });
    base = `http://127.0.0.1:${server.port}`;
  });

  afterAll(() => {
    void server?.stop(true);
  });

  it('serves the curve as JSON', async () => {
    const res = await fetch(`${base}/api/runs/${RUN_ID}/curves?envId=${ENV_A}`);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('application/json');
    const curve = (await res.json()) as CurvesResponse;
    expect(curve.steps).toEqual([100, 200, 300]);
    expect(curve.seeds).toBe(4);
    expect(curve.iqm[0]).toBeCloseTo(2.5, 12);
    expectAligned(curve);
  });

  it('rejects a bad env or policy with 400, and an unknown run with 404', async () => {
    expect((await fetch(`${base}/api/runs/${RUN_ID}/curves?envId=Nope-v9`)).status).toBe(400);
    expect((await fetch(`${base}/api/runs/${RUN_ID}/curves?policy=a%2Fb`)).status).toBe(400);
    expect((await fetch(`${base}/api/runs/no-such-run/curves`)).status).toBe(404);
    expect((await fetch(`${base}/api/runs/${RUN_ID}/curves/extra`)).status).toBe(404);
  });

  it('returns byte-identical JSON on a repeat request', async () => {
    const a = await (await fetch(`${base}/api/runs/${RUN_ID}/curves`)).text();
    const b = await (await fetch(`${base}/api/runs/${RUN_ID}/curves`)).text();
    expect(b).toBe(a);
  });
});

// ---------- real roots ----------

describe('curves against the real bench roots', () => {
  let server: ReturnType<typeof Bun.serve> | null = null;
  let base = '';

  beforeAll(() => {
    if (!demoPresent && !keeperPresent) return;
    const app = createApp({ benchRoot: REAL_BENCH, cacheFile: null, fleetCacheDir: null, ttlMs: 0 });
    server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: app.fetch });
    base = `http://127.0.0.1:${server.port}`;
  });

  afterAll(() => {
    void server?.stop(true);
  });

  it.skipIf(!demoPresent)('slices the 16-env demo run per environment', async () => {
    const all = (await (await fetch(`${base}/api/runs/${DEMO_ID}/curves`)).json()) as CurvesResponse;
    expect(all.seeds).toBe(128);
    // ~100 tape windows (CB cadence) — the final step comes from fleet.jsonl finals
    expect(all.steps.length).toBeGreaterThan(50);
    expect(all.steps.at(-1)).toBe(100000);
    expectAligned(all);

    const cheetah = (await (
      await fetch(`${base}/api/runs/${DEMO_ID}/curves?envId=HalfCheetah-v4`)
    ).json()) as CurvesResponse;
    const cartpole = (await (
      await fetch(`${base}/api/runs/${DEMO_ID}/curves?envId=CartPole-cont-v0`)
    ).json()) as CurvesResponse;
    expect(cheetah.seeds).toBe(8);
    expect(cartpole.seeds).toBe(8);
    // The house rule made measurable: CartPole tops out at 1.0/step and HalfCheetah
    // does not, so the two slices must not be the same numbers.
    expect(cartpole.iqm.at(-1)).not.toBe(cheetah.iqm.at(-1));
    for (let i = 0; i < all.steps.length; i++) {
      const lo = all.p10[i];
      const mid = all.p50[i];
      const hi = all.p90[i];
      expect(lo ?? 0).toBeLessThanOrEqual(mid ?? 0);
      expect(mid ?? 0).toBeLessThanOrEqual(hi ?? 0);
    }
  });

  it.skipIf(!keeperPresent)('slices the 1024-seed keeper run per environment', async () => {
    const all = (await (await fetch(`${base}/api/runs/${KEEPER_ID}/curves`)).json()) as CurvesResponse;
    expect(all.seeds).toBe(1024);
    // 255 sampled tape windows + the final certificate step
    expect(all.steps.length).toBe(256);
    expectAligned(all);
    const cheetah = (await (
      await fetch(`${base}/api/runs/${KEEPER_ID}/curves?envId=HalfCheetah-v4`)
    ).json()) as CurvesResponse;
    expect(cheetah.seeds).toBe(64);
    expect(cheetah.steps).toEqual(all.steps);
    // 64 HalfCheetah seeds at 1M steps: a real, finite, positive-ended curve
    const last = cheetah.iqm.at(-1);
    expect(typeof last).toBe('number');
    expect(Number.isFinite(last ?? Number.NaN)).toBe(true);
  }, 60000);

  /**
   * The numbers themselves, pinned.
   *
   * The structural tests above would all still pass if `iqmSorted` drifted to a plain
   * mean or `percentileSorted` to a nearest-rank rule — the curve would stay monotone,
   * finite and aligned while quietly disagreeing with the Python that gates promotions.
   * These constants were produced independently: fleet.jsonl -> numpy, IQM by
   * tools/fleet.py's own definition (sort, drop ⌊n/4⌋ each end, mean the rest) and the
   * spread by np.percentile's default linear interpolation. The percentiles matched to
   * 0.0 and the IQM to 7e-16 (float summation order) across all 6 p93 slices and all 16
   * of the demo's, so the tolerance here is about summation order, not about agreement.
   */
  it.skipIf(!keeperPresent)('reports the same statistics numpy does for the keeper', async () => {
    const cheetah = (await (
      await fetch(`${base}/api/runs/${KEEPER_ID}/curves?envId=HalfCheetah-v4`)
    ).json()) as CurvesResponse;
    // 255 sampled tape windows + the final step; constants recomputed from the
    // TAPE with independent numpy (window sampling replicated, IQM by
    // tools/fleet.py's definition, percentiles by np.percentile linear).
    expect(cheetah.steps.length).toBe(256);
    expect(cheetah.steps[0]).toBe(3150);
    expect(cheetah.steps.at(-1)).toBe(1000000);
    const close = (got: number | null | undefined, want: number): void => {
      expect(got).not.toBeNull();
      expect(Math.abs((got ?? Number.NaN) - want)).toBeLessThan(1e-9);
    };
    close(cheetah.iqm[0], -0.006311481888332065);
    close(cheetah.iqm.at(-1), 0.5451980642383675);
    close(cheetah.p50[0], -0.006326222798180958);
    close(cheetah.p10.at(-1), 0.24822101637639718);
    close(cheetah.p50.at(-1), 0.530007148427346);
    close(cheetah.p90.at(-1), 0.7833815310631663);
    // and the CI is an interval around the point estimate, not a decoration
    const lo = cheetah.ciLo.at(-1) ?? Number.NaN;
    const hi = cheetah.ciHi.at(-1) ?? Number.NaN;
    expect(lo).toBeLessThanOrEqual(0.5451980642383675);
    expect(hi).toBeGreaterThanOrEqual(0.5451980642383675);
    expect(hi - lo).toBeLessThan(0.3); // 64 seeds: a real interval, not a shrug
  }, 60000);
});
