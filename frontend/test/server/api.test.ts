/// <reference types="bun" />

import { afterAll, beforeAll, describe, expect, it } from 'bun:test';
import { mkdir, mkdtemp, rm, symlink, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { RunDetail, RunIndexEntry, RunSummary } from '@shared/types';
import { createApp, type App } from '../../server/app';
import { scanRecordings } from '../../server/recordings';
import { tapeBytes } from '../helpers/tape';
import { ENV, completeTape, laneBytes, runMeta, writeRun } from './fixture';

let ROOT = '';
let BENCH = '';
let app: App;

const tapeA = completeTape(2, 100);
const laneA0 = laneBytes(ENV, 10);
const laneA1 = laneBytes(ENV, 7);
const summaryA: RunSummary = {
  overall: { rewardRateIQM: 1.5, seeds: 2 },
  envSlices: { [ENV]: { rewardRateIQM: 1.5 } },
};

const warned: string[] = [];
const originalWarn = console.warn;

function bytes(view: Uint8Array): Uint8Array<ArrayBuffer> {
  return new Uint8Array(view);
}

function get(path: string, init?: RequestInit): Promise<Response> {
  return app.fetch(new Request(`http://oak.test${path}`, init));
}

async function listRuns(): Promise<RunIndexEntry[]> {
  const res = await get('/api/runs');
  expect(res.status).toBe(200);
  return (await res.json()) as RunIndexEntry[];
}

async function detailOf(id: string): Promise<RunDetail> {
  const res = await get(`/api/runs/${id}`);
  expect(res.status).toBe(200);
  return (await res.json()) as RunDetail;
}

beforeAll(async () => {
  console.warn = (...args: unknown[]) => {
    warned.push(args.map(String).join(' '));
  };
  ROOT = await mkdtemp(join(tmpdir(), 'oak-api-'));
  BENCH = join(ROOT, 'bench');
  await mkdir(BENCH, { recursive: true });
  await writeFile(join(ROOT, 'keeper'), 'run-a\n');

  const runA = await writeRun(BENCH, 'run-a', {
    meta: runMeta(100, { startedAt: '2026-08-02T00:00:00Z' }),
    tape: tapeA,
    summary: JSON.stringify(summaryA),
    lanes: [
      { env: ENV, seed: 0, bytes: laneA0 },
      { env: ENV, seed: 1, bytes: laneA1 },
      { env: 'DeepSea-cont-v0', seed: 0, bytes: laneBytes(ENV, 6) },
    ],
  });

  const outside = join(ROOT, 'outside');
  await mkdir(join(outside, 'lane9'), { recursive: true });
  await writeFile(join(outside, 'lane9', 'actions.bin'), laneBytes(ENV, 8));
  await writeFile(join(outside, 'stray-actions.bin'), laneBytes(ENV, 8));
  await symlink(join(outside, 'lane9'), join(runA, ENV, 'seed9'));
  await mkdir(join(runA, ENV, 'seed3'), { recursive: true });
  await symlink(join(outside, 'stray-actions.bin'), join(runA, ENV, 'seed3', 'actions.bin'));
  await mkdir(join(runA, ENV, 'notseed'), { recursive: true });

  await writeRun(BENCH, 'run-b', {
    meta: runMeta(100, { startedAt: '2026-08-03T00:00:00Z' }),
    tape: tapeBytes(2, [{ stepStart: 0, stepCount: 40, values: {} }]),
    summary: JSON.stringify({ schemaVersion: 2, envs: {} }),
    lanes: [{ env: ENV, seed: 0, bytes: laneBytes(ENV, 5) }],
  });

  const runC = await writeRun(BENCH, 'run-c', {
    meta: runMeta(100),
    tape: tapeBytes(2, [{ stepStart: 0, stepCount: 40, values: {} }]),
  });
  const past = new Date(Date.now() - 600_000);
  await utimes(join(runC, 'tape.bin'), past, past);

  await writeRun(BENCH, 'no-tape', { meta: runMeta(100) });
  await writeRun(BENCH, 'bad-meta', {
    meta: JSON.stringify({ fleetSchemaVersion: 3, policies: ['oak'] }),
    tape: completeTape(2, 100),
  });
  await mkdir(join(BENCH, 'not-a-run'), { recursive: true });
  await writeFile(join(BENCH, 'stray.txt'), 'x');

  await writeRun(outside, 'run-x', {
    lanes: [{ env: ENV, seed: 0, bytes: laneBytes(ENV, 6) }],
  });

  app = createApp({ benchRoot: BENCH, ttlMs: 0 });
});

afterAll(async () => {
  console.warn = originalWarn;
  await rm(ROOT, { recursive: true, force: true });
});

describe('run index', () => {
  it('lists valid runs newest-first and skips broken dirs', async () => {
    const runs = await listRuns();
    expect(runs.map((r) => r.id)).toEqual(['run-b', 'run-a', 'run-c']);
  });

  it('warns once per skipped dir on a fresh index', async () => {
    warned.length = 0;
    const fresh = createApp({ benchRoot: BENCH, ttlMs: 0 });
    await fresh.fetch(new Request('http://oak.test/api/runs'));
    expect(warned.some((w) => w.includes('skipping no-tape'))).toBe(true);
    expect(warned.some((w) => w.includes('skipping bad-meta'))).toBe(true);
    expect(warned.some((w) => w.includes('run-a'))).toBe(false);
  });

  it('marks the complete keeper run', async () => {
    const runs = await listRuns();
    const a = runs.find((r) => r.id === 'run-a');
    expect(a).toMatchObject({
      envIds: [ENV],
      batch: 2,
      steps: 100,
      ablationLabel: 'none',
      startedAt: '2026-08-02T00:00:00Z',
      complete: true,
      maxStep: 100,
      inFlight: false,
      hasSummary: true,
      hasRecordings: true,
      role: 'keeper',
    });
  });

  it('marks an incomplete run with a fresh tape as in flight', async () => {
    const runs = await listRuns();
    const b = runs.find((r) => r.id === 'run-b');
    expect(b).toMatchObject({
      complete: false,
      maxStep: 40,
      inFlight: true,
      hasSummary: true,
      role: null,
    });
  });

  it('marks an incomplete run with a stale tape as not in flight', async () => {
    const runs = await listRuns();
    const c = runs.find((r) => r.id === 'run-c');
    expect(c).toMatchObject({
      complete: false,
      maxStep: 40,
      inFlight: false,
      startedAt: null,
      hasSummary: false,
      hasRecordings: false,
      role: null,
    });
  });

  it('assigns no keeper role without a keeper file', async () => {
    const nk = join(ROOT, 'nk', 'bench');
    await mkdir(nk, { recursive: true });
    await writeRun(nk, 'solo', { meta: runMeta(100), tape: completeTape(2, 100) });
    const other = createApp({ benchRoot: nk, ttlMs: 0 });
    const res = await other.fetch(new Request('http://oak.test/api/runs'));
    const runs = (await res.json()) as RunIndexEntry[];
    expect(runs.map((r) => [r.id, r.role])).toEqual([['solo', null]]);
  });
});

describe('run detail', () => {
  it('returns meta, summary, and header-counted recordings', async () => {
    const detail = await detailOf('run-a');
    expect(detail.id).toBe('run-a');
    expect(detail.meta.steps).toBe(100);
    expect(detail.meta.envSchedule).toEqual([{ envId: ENV, seedStart: 0, seedEnd: 1 }]);
    expect(detail.summary).toEqual(summaryA);
    expect(detail.recordings).toEqual([
      { envId: ENV, seed: 0, steps: 10, bytes: laneA0.byteLength },
      { envId: ENV, seed: 1, steps: 7, bytes: laneA1.byteLength },
    ]);
  });

  it('nulls out a summary.json that is not a run summary', async () => {
    warned.length = 0;
    const detail = await detailOf('run-b');
    expect(detail.summary).toBeNull();
    expect(warned.some((w) => w.includes('summary.json is not a run summary'))).toBe(true);
  });

  it('nulls out a missing summary.json', async () => {
    const detail = await detailOf('run-c');
    expect(detail.summary).toBeNull();
  });

  it('404s an unknown run', async () => {
    const res = await get('/api/runs/nope');
    expect(res.status).toBe(404);
  });

  it('skips a recording whose header envKind contradicts its directory env', async () => {
    warned.length = 0;
    const detail = await detailOf('run-a');
    expect(detail.recordings.some((r) => r.envId === 'DeepSea-cont-v0')).toBe(false);
    expect(warned.some((w) => w.includes('contradicts directory env'))).toBe(true);
  });
});

describe('tape serving', () => {
  it('serves the whole tape with an etag', async () => {
    const res = await get('/api/runs/run-a/tape');
    expect(res.status).toBe(200);
    expect(res.headers.get('accept-ranges')).toBe('bytes');
    expect(res.headers.get('etag')).toMatch(/^".+"$/);
    expect(res.headers.get('content-length')).toBe(String(tapeA.byteLength));
    expect(new Uint8Array(await res.arrayBuffer())).toEqual(bytes(tapeA));
  });

  it('serves a tail range', async () => {
    const size = tapeA.byteLength;
    const res = await get('/api/runs/run-a/tape', { headers: { range: 'bytes=-10' } });
    expect(res.status).toBe(206);
    expect(res.headers.get('content-range')).toBe(`bytes ${size - 10}-${size - 1}/${size}`);
    expect(new Uint8Array(await res.arrayBuffer())).toEqual(bytes(tapeA.subarray(size - 10)));
  });

  it('serves a bounded range', async () => {
    const res = await get('/api/runs/run-a/tape', { headers: { range: 'bytes=4-11' } });
    expect(res.status).toBe(206);
    expect(new Uint8Array(await res.arrayBuffer())).toEqual(bytes(tapeA.subarray(4, 12)));
  });

  it('416s an unsatisfiable range', async () => {
    const res = await get('/api/runs/run-a/tape', { headers: { range: 'bytes=999999-' } });
    expect(res.status).toBe(416);
    expect(res.headers.get('content-range')).toBe(`bytes */${tapeA.byteLength}`);
  });

  it('returns the full tape when if-range does not match', async () => {
    const res = await get('/api/runs/run-a/tape', {
      headers: { range: 'bytes=0-3', 'if-range': '"stale"' },
    });
    expect(res.status).toBe(200);
    expect(new Uint8Array(await res.arrayBuffer())).toEqual(bytes(tapeA));
  });

  it('honours the range when if-range matches the etag', async () => {
    const etag = (await get('/api/runs/run-a/tape', { method: 'HEAD' })).headers.get('etag');
    expect(etag).not.toBeNull();
    const res = await get('/api/runs/run-a/tape', {
      headers: { range: 'bytes=0-3', 'if-range': etag ?? '' },
    });
    expect(res.status).toBe(206);
    expect(new Uint8Array(await res.arrayBuffer())).toEqual(bytes(tapeA.subarray(0, 4)));
  });

  it('304s a matching if-none-match', async () => {
    const etag = (await get('/api/runs/run-a/tape', { method: 'HEAD' })).headers.get('etag');
    const res = await get('/api/runs/run-a/tape', {
      headers: { 'if-none-match': etag ?? '' },
    });
    expect(res.status).toBe(304);
    expect((await res.arrayBuffer()).byteLength).toBe(0);
  });

  it('answers HEAD without a body', async () => {
    const res = await get('/api/runs/run-a/tape', { method: 'HEAD' });
    expect(res.status).toBe(200);
    expect(res.headers.get('content-length')).toBe(String(tapeA.byteLength));
    expect((await res.arrayBuffer()).byteLength).toBe(0);
  });

  it('404s the tape of a run without one', async () => {
    const res = await get('/api/runs/no-tape/tape');
    expect(res.status).toBe(404);
  });
});

describe('recordings serving', () => {
  it('serves actions.bin', async () => {
    const res = await get(`/api/runs/run-a/recordings/${ENV}/0`);
    expect(res.status).toBe(200);
    expect(new Uint8Array(await res.arrayBuffer())).toEqual(bytes(laneA0));
  });

  it('serves a range of actions.bin', async () => {
    const res = await get(`/api/runs/run-a/recordings/${ENV}/1`, {
      headers: { range: 'bytes=0-7' },
    });
    expect(res.status).toBe(206);
    expect(new Uint8Array(await res.arrayBuffer())).toEqual(bytes(laneA1.subarray(0, 8)));
  });

  it('404s unknown seeds, non-numeric seeds, and unknown envs', async () => {
    for (const path of [
      `/api/runs/run-a/recordings/${ENV}/5`,
      `/api/runs/run-a/recordings/${ENV}/abc`,
      '/api/runs/run-a/recordings/NotAnEnv-v0/0',
      '/api/runs/run-a/recordings/DeepSea-cont-v0/0',
    ]) {
      expect((await get(path)).status).toBe(404);
    }
  });

  it('404s symlinked lane dirs and symlinked actions.bin', async () => {
    expect((await get(`/api/runs/run-a/recordings/${ENV}/9`)).status).toBe(404);
    expect((await get(`/api/runs/run-a/recordings/${ENV}/3`)).status).toBe(404);
  });

  it('re-reads the header of a lane that grew on disk', async () => {
    const before = await detailOf('run-b');
    expect(before.recordings).toEqual([
      { envId: ENV, seed: 0, steps: 5, bytes: laneBytes(ENV, 5).byteLength },
    ]);
    const grown = laneBytes(ENV, 12);
    await writeFile(join(BENCH, 'run-b', ENV, 'seed0', 'actions.bin'), grown);
    const after = await detailOf('run-b');
    expect(after.recordings).toEqual([
      { envId: ENV, seed: 0, steps: 12, bytes: grown.byteLength },
    ]);
  });

  it('drops lanes outside the bench root', async () => {
    const outsideRun = join(ROOT, 'outside', 'run-x');
    expect(await scanRecordings(outsideRun, { root: BENCH })).toEqual([]);
    const unconstrained = await scanRecordings(outsideRun);
    expect(unconstrained.map((r) => [r.envId, r.seed, r.steps])).toEqual([[ENV, 0, 6]]);
  });
});

describe('routing and security', () => {
  it('reports health', async () => {
    const res = await get('/api/health');
    expect(res.status).toBe(200);
    expect((await res.json()) as { ok: boolean }).toMatchObject({ ok: true });
  });

  it('rejects traversal and control characters in segments', async () => {
    for (const path of [
      '/api/runs/%2e%2e%2fsecret/tape',
      '/api/runs/run-a%2fother',
      '/api/runs/run-a%5c..%5cother',
      '/api/runs/%00',
    ]) {
      expect((await get(path)).status).toBe(400);
    }
  });

  it('405s non-GET methods', async () => {
    for (const method of ['POST', 'PUT', 'DELETE', 'PATCH']) {
      expect((await get('/api/runs', { method })).status).toBe(405);
    }
  });

  it('204s preflight', async () => {
    expect((await get('/api/runs', { method: 'OPTIONS' })).status).toBe(204);
  });

  it('404s unknown api endpoints', async () => {
    expect((await get('/api/fleet')).status).toBe(404);
    expect((await get('/api/runs/run-a/meta/extra')).status).toBe(404);
  });
});
