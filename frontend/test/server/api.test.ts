/// <reference types="bun" />

import { afterAll, beforeAll, describe, expect, it } from 'bun:test';
import { existsSync } from 'node:fs';
import { chmod, mkdir, readdir, rm, symlink, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { crc32, deflateRawSync } from 'node:zlib';

import type { FleetColumnar, KeeperResponse, RunDetail, RunIndexEntry } from '../../shared/types';
import { createApp, parseRangeHeader } from '../../server/app';
import { fleetCacheDir, runIndexCacheFile, CACHE_DIR } from '../../server/config';
import {
  clearFleetCache,
  deriveFleetColumnar,
  parseFleetQuery,
  type FleetQuery,
} from '../../server/fleetDerive';
import { clearStepsCache, scanRecordings } from '../../server/recordings';
import { RunIndex, runCompleteness } from '../../server/runIndex';
import { statOrNull } from '../../server/jsonl';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE_ROOT = resolve(HERE, '..', 'fixtures', 'server');
const BENCH = join(FIXTURE_ROOT, 'bench');
const HOSTILE = join(FIXTURE_ROOT, 'bench-hostile');
const OUTSIDE = join(FIXTURE_ROOT, 'outside');
const FLEET_CACHE = join(FIXTURE_ROOT, 'fleet-cache');
const SECRET = 'TOP-SECRET-NOT-A-RECORDING-'.repeat(16);


interface FleetRowSeed {
  kind: 'hb' | 'final';
  lane: number;
  step: number;
  envId: string;
  rewardSum: number;
  rewardRate: number;
}

function fleetRow(s: FleetRowSeed): string {
  return JSON.stringify({
    v: 1,
    kind: s.kind,
    envId: s.envId,
    policy: 'oak',
    ablationLabel: 'none',
    lane: s.lane,
    seed: s.lane,
    step: s.step,
    wallSeconds: s.step / 100,
    stepsPerSecond: 1000 + s.lane,
    rewardSum: s.rewardSum,
    doneCount: 0,
    kernelRho: 0.01,
    paramAbsMax: 1.5,
    firstNonFiniteStep: null,
    firstNonFiniteDetector: null,
    health: {
      rewardRate: s.rewardRate,
      actionDeltaL2Mean: 0.001,
      nonFiniteByPhase: { encode: 0, learn: 0, plan: 0, tdPsi: 0 },
    },
  });
}

function tapeBytes(
  laneCount: number,
  windows: { start: number; count: number; values: Record<string, number[]> }[],
  fields: string[] = ['rewardSum', 'doneCount', 'rewardRate', 'kernelRho', 'deltaSum', 'deltaPeak', 'paramAbsMax'],
): Buffer {
  const rowBytes = 8 + fields.length * laneCount * 4;
  const payload = Buffer.alloc(windows.length * rowBytes);
  windows.forEach((w, wi) => {
    const p = wi * rowBytes;
    payload.writeUInt32LE(w.start, p);
    payload.writeUInt32LE(w.count, p + 4);
    fields.forEach((f, fi) => {
      for (let l = 0; l < laneCount; l++) {
        payload.writeFloatLE(w.values[f]?.[l] ?? 0, p + 8 + (fi * laneCount + l) * 4);
      }
    });
  });
  const json = Buffer.from(JSON.stringify({ fields }), 'utf8');
  const head = Buffer.alloc(16);
  head.writeUInt32LE(0x544b414f, 0);
  head.writeUInt32LE(1, 4);
  head.writeUInt32LE(laneCount, 8);
  head.writeUInt32LE(json.length, 12);
  const comp = deflateRawSync(payload);
  const seg = Buffer.alloc(20);
  seg.writeUInt32LE(0x54474553, 0);
  seg.writeUInt32LE(windows.length, 4);
  seg.writeUInt32LE(payload.length, 8);
  seg.writeUInt32LE(comp.length, 12);
  seg.writeUInt32LE(crc32(comp) >>> 0, 16);
  return Buffer.concat([head, json, seg, comp]);
}

function aggRow(step: number, kind: 'hb' | 'final'): string {
  return JSON.stringify({
    step,
    kind,
    policy: 'oak',
    alive: 2,
    dead: 0,
    deliveredRateP10: -0.01,
    deliveredRateP50: 0.02,
    deliveredRateP90: 0.4,
    recentRateP10: -0.01,
    recentRateP50: 0.03,
    recentRateP90: 0.5,
    recentRateMax: 0.9,
    bestLaneSeed: 1,
    bestLaneDeliveredRate: 0.4,
    bestLaneRewardSum: 40,
    positiveRateLaneCount: 1,
    positiveRateFraction: 0.5,
    aggStepsPerSecond: 20000,
    wallSeconds: step / 100,
  });
}

async function waitUntilAsync(pred: () => Promise<boolean>, ms = 5000): Promise<void> {
  const t0 = Date.now();
  while (!(await pred())) {
    if (Date.now() - t0 > ms) throw new Error('timed out waiting for condition');
    await Bun.sleep(10);
  }
}

function recordBytes(steps: number, seedTag: number): Uint8Array {
  const f = new Float32Array(steps * 16);
  for (let s = 0; s < steps; s++) {
    for (let c = 0; c < 16; c++) f[s * 16 + c] = seedTag * 100 + s * 16 + c;
    f[s * 16 + 6] = 0;
  }
  return new Uint8Array(f.buffer);
}

async function writeLane(
  runDir: string,
  envId: string,
  seed: number,
  recordVersion: number,
  steps: number,
  actDim: number,
): Promise<void> {
  const dir = join(runDir, envId, `seed${seed}`);
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'meta.json'), JSON.stringify({ envId, seed, actDim, recordVersion }, null, 1));
  await Bun.write(join(dir, 'actions.bin'), recordBytes(steps, seed));
}


const V4_FILE_HEADER = 16;
const V4_CHUNK_HEADER = 32;
const V4_FILE_MAGIC = 0x344b414f;
const V4_CHUNK_MAGIC = 0x344b4843;

const v4PayloadBytes = (steps: number): number => 8 + steps * 3;

interface V4Options {
  tornTail?: boolean;
}

function recordBytesV4(
  envKind: number,
  chunkSteps: readonly number[],
  opts: V4Options = {},
): { bytes: Uint8Array; chunks: [number, number, number, number][]; steps: number } {
  let size = V4_FILE_HEADER;
  for (const steps of chunkSteps) size += V4_CHUNK_HEADER + v4PayloadBytes(steps);
  if (opts.tornTail === true) size += V4_CHUNK_HEADER - 5;

  const bytes = new Uint8Array(size);
  const dv = new DataView(bytes.buffer);
  dv.setUint32(0, V4_FILE_MAGIC, true);
  dv.setUint32(4, 4, true);
  dv.setUint32(8, envKind, true);
  dv.setUint32(12, 1, true);

  const chunks: [number, number, number, number][] = [];
  let off = V4_FILE_HEADER;
  let step = 0;
  for (const steps of chunkSteps) {
    const comp = v4PayloadBytes(steps);
    dv.setUint32(off, V4_CHUNK_MAGIC, true);
    dv.setUint32(off + 4, step, true);
    dv.setUint32(off + 8, steps, true);
    dv.setUint32(off + 12, steps * 16, true);
    dv.setUint32(off + 16, comp, true);
    dv.setUint32(off + 20, 0, true);
    dv.setUint32(off + 24, 0, true);
    dv.setFloat32(off + 28, Number.NaN, true);
    for (let i = 0; i < comp; i++) bytes[off + V4_CHUNK_HEADER + i] = (step + i) & 0xff;
    chunks.push([step, steps, off, V4_CHUNK_HEADER + comp]);
    off += V4_CHUNK_HEADER + comp;
    step += steps;
  }
  if (opts.tornTail === true) {
    dv.setUint32(off, V4_CHUNK_MAGIC, true);
    dv.setUint32(off + 4, step, true);
    dv.setUint32(off + 8, 99, true);
    dv.setUint32(off + 16, 4096, true);
  }
  return { bytes, chunks, steps: step };
}

const V4_CODECS: Record<string, unknown> = {
  'Chain-cont-v0': { category: 'grid', ctrl: 'index', poseFields: [{ idx: 0, enc: 'u8' }], poseConst: { 1: 15 } },
  'Pendulum-cont-v0': {
    category: 'classic',
    ctrl: { kind: 'u8box', n: 1 },
    poseFields: [
      { idx: 0, enc: 'i16', scale: 1e-4, wrap: true },
      { idx: 1, enc: 'i16', scale: 1e-3 },
      { idx: 2, enc: 'i16', scale: 1e-3 },
    ],
  },
};

async function writeLaneV4(
  runDir: string,
  envId: string,
  seed: number,
  envKind: number,
  actDim: number,
  chunkSteps: readonly number[],
  opts: V4Options & { finalized?: boolean } = {},
): Promise<number> {
  const dir = join(runDir, envId, `seed${seed}`);
  await mkdir(dir, { recursive: true });
  const built = recordBytesV4(envKind, chunkSteps, opts);
  const meta: Record<string, unknown> = {
    envId, seed, actDim, recordVersion: 4, codecVersion: 1, codec: V4_CODECS[envId],
  };
  if (opts.finalized !== false) {
    meta['steps'] = built.steps;
    meta['chunks'] = built.chunks;
  }
  await writeFile(join(dir, 'meta.json'), JSON.stringify(meta, null, 1));
  await Bun.write(join(dir, 'actions.bin'), built.bytes);
  return built.steps;
}

async function buildFixtures(): Promise<void> {
  await rm(BENCH, { recursive: true, force: true });
  await mkdir(BENCH, { recursive: true });

  await writeFile(join(FIXTURE_ROOT, 'keeper'), 'alpha-complete\n');

  const alpha = join(BENCH, 'alpha-complete');
  await mkdir(alpha, { recursive: true });
  await writeFile(
    join(alpha, 'meta.json'),
    JSON.stringify({
      fleetSchemaVersion: 1,
      command: 'benchmark',
      envId: 'HalfCheetah-v4',
      batch: 2,
      seeds: [0, 1],
      steps: 200,
      policies: ['oak'],
      ablation: [],
      ablationLabel: 'model-to-keyboard',
      startedAt: '2026-08-02T10:00:00Z',
      envSchedule: [
        { envId: 'HalfCheetah-v4', seedStart: 0, seedEnd: 0, flopsPerLaneStep: 230244 },
        { envId: 'FourRooms-cont-v0', seedStart: 1, seedEnd: 1, flopsPerLaneStep: 210284 },
      ],
    }),
  );
  await writeFile(
    join(alpha, 'fleet.jsonl'),
    [
      fleetRow({ kind: 'final', lane: 0, step: 200, envId: 'HalfCheetah-v4', rewardSum: -8, rewardRate: -0.04 }),
      fleetRow({ kind: 'final', lane: 1, step: 200, envId: 'FourRooms-cont-v0', rewardSum: 9, rewardRate: 0.045 }),
      '',
    ].join('\n'),
  );
  await writeFile(
    join(alpha, 'metrics.oak.oakt'),
    tapeBytes(2, [
      {
        start: 0,
        count: 100,
        values: { rewardSum: [-5, 3], rewardRate: [-0.05, 0.03], kernelRho: [0.01, 0.01], paramAbsMax: [1.5, 1.5] },
      },
    ]),
  );
  await writeFile(join(alpha, 'aggregate.jsonl'), `${aggRow(100, 'hb')}\n${aggRow(200, 'final')}\n`);
  await writeFile(
    join(alpha, 'deaths.jsonl'),
    `${JSON.stringify({
      seed: 1,
      lane: 1,
      step: 150,
      detector: 'paramExplosion',
      policy: 'oak',
      rewardSumAtDeath: 2,
      lastGoodRewardRate: 0.01,
    })}\n`,
  );
  await writeFile(join(alpha, 'summary.json'), JSON.stringify({
    envSlices: { 'HalfCheetah-v4': [{ policy: 'oak', recentRewardRateIQM: 0.01, seeds: 2 }] },
  }));
  await writeLane(alpha, 'HalfCheetah-v4', 0, 3, 4, 6);
  await writeLane(alpha, 'FourRooms-cont-v0', 1, 3, 3, 1);
  await writeLane(alpha, 'DeepSea-cont-v0', 2, 2, 5, 1);
  await writeLaneV4(alpha, 'Chain-cont-v0', 5, 8, 1, [4, 3]);
  await writeLaneV4(alpha, 'Pendulum-cont-v0', 6, 7, 1, [5, 6], { finalized: false, tornTail: true });
  await mkdir(join(alpha, 'NotAnEnv-v9', 'seed4'), { recursive: true });
  await writeFile(join(alpha, 'NotAnEnv-v9', 'seed4', 'meta.json'), JSON.stringify({ recordVersion: 3 }));

  const beta = join(BENCH, 'beta-incomplete');
  await mkdir(beta, { recursive: true });
  await writeFile(
    join(beta, 'meta.json'),
    JSON.stringify({
      fleetSchemaVersion: 1,
      envId: 'HalfCheetah-v4',
      batch: 2,
      seeds: [0, 1],
      steps: 1000,
      policies: ['oak'],
      ablationLabel: 'none',
      startedAt: '2026-08-01T10:00:00Z',
    }),
  );
  await writeFile(
    join(beta, 'fleet.jsonl'),
    `${[
      fleetRow({ kind: 'hb', lane: 0, step: 100, envId: 'HalfCheetah-v4', rewardSum: -5, rewardRate: -0.05 }),
      fleetRow({ kind: 'hb', lane: 1, step: 100, envId: 'HalfCheetah-v4', rewardSum: 3, rewardRate: 0.03 }),
      fleetRow({ kind: 'hb', lane: 0, step: 200, envId: 'HalfCheetah-v4', rewardSum: -6, rewardRate: -0.03 }),
    ].join('\n')}\n{"v":1,"kind":"hb","lane":1,"step":200,"health":{"rewardRa`,
  );
  await writeFile(join(beta, 'aggregate.jsonl'), `${aggRow(100, 'hb')}\n${aggRow(200, 'hb')}\n`);
  await writeFile(join(beta, '.done'), '');

  const gamma = join(BENCH, 'gamma-short-final');
  await mkdir(gamma, { recursive: true });
  await writeFile(
    join(gamma, 'meta.json'),
    JSON.stringify({
      fleetSchemaVersion: 1,
      envId: 'DeepSea-cont-v0',
      batch: 1,
      seeds: [0],
      steps: 1000,
      policies: ['oak'],
      startedAt: '2026-07-30T10:00:00Z',
    }),
  );
  await writeFile(
    join(gamma, 'fleet.jsonl'),
    `${fleetRow({ kind: 'final', lane: 0, step: 400, envId: 'DeepSea-cont-v0', rewardSum: 1, rewardRate: 0.0025 })}\n`,
  );

  const delta = join(BENCH, 'delta-lane-meta');
  await mkdir(delta, { recursive: true });
  await writeFile(
    join(delta, 'meta.json'),
    JSON.stringify({ envId: 'HalfCheetah-v4', seed: 3, actDim: 6, recordVersion: 3 }),
  );

  const eps = join(BENCH, 'epsilon-live');
  await mkdir(eps, { recursive: true });
  await writeFile(
    join(eps, 'meta.json'),
    JSON.stringify({
      fleetSchemaVersion: 1,
      envId: 'CartPole-cont-v0',
      batch: 1,
      seeds: [0],
      steps: 300,
      policies: ['oak'],
      startedAt: '2026-08-03T10:00:00Z',
    }),
  );
  await writeFile(
    join(eps, 'fleet.jsonl'),
    `${fleetRow({ kind: 'hb', lane: 0, step: 100, envId: 'CartPole-cont-v0', rewardSum: 10, rewardRate: 0.1 })}\n`,
  );
  await writeFile(join(eps, 'aggregate.jsonl'), `${aggRow(100, 'hb')}\n`);
}

async function writeRunRoot(
  root: string,
  id: string,
  envId: string,
  startedAt: string,
): Promise<string> {
  const dir = join(root, id);
  await mkdir(dir, { recursive: true });
  await writeFile(
    join(dir, 'meta.json'),
    JSON.stringify({
      fleetSchemaVersion: 1,
      envId,
      batch: 1,
      seeds: [0],
      steps: 200,
      policies: ['oak'],
      startedAt,
    }),
  );
  await writeFile(
    join(dir, 'fleet.jsonl'),
    `${fleetRow({ kind: 'final', lane: 0, step: 200, envId, rewardSum: 1, rewardRate: 0.005 })}\n`,
  );
  return dir;
}

async function buildHostileFixtures(): Promise<void> {
  await rm(HOSTILE, { recursive: true, force: true });
  await rm(OUTSIDE, { recursive: true, force: true });
  await mkdir(HOSTILE, { recursive: true });
  await mkdir(join(OUTSIDE, 'lane'), { recursive: true });

  await writeFile(join(OUTSIDE, 'secret.bin'), SECRET);
  await writeFile(
    join(OUTSIDE, 'lane', 'meta.json'),
    JSON.stringify({ envId: 'FourRooms-cont-v0', seed: 8, actDim: 1, recordVersion: 3 }),
  );
  await Bun.write(join(OUTSIDE, 'lane', 'actions.bin'), recordBytes(4, 8));

  const good = await writeRunRoot(HOSTILE, 'good-run', 'HalfCheetah-v4', '2026-08-02T09:00:00Z');
  await writeLaneV4(good, 'Pendulum-cont-v0', 0, 7, 1, [4]);

  const dirFleet = await writeRunRoot(HOSTILE, 'dir-fleet', 'DeepSea-cont-v0', '2026-08-02T08:00:00Z');
  await rm(join(dirFleet, 'fleet.jsonl'), { force: true });
  await mkdir(join(dirFleet, 'fleet.jsonl'), { recursive: true });

  const noperm = await writeRunRoot(HOSTILE, 'noperm-fleet', 'CartPole-cont-v0', '2026-08-02T07:00:00Z');
  await writeFile(join(noperm, 'aggregate.jsonl'), `${aggRow(100, 'hb')}\n`);
  await chmod(join(noperm, 'fleet.jsonl'), 0o000);
  await chmod(join(noperm, 'aggregate.jsonl'), 0o000);

  await writeRunRoot(HOSTILE, 'stale-cache', 'Pendulum-cont-v0', '2026-08-02T06:00:00Z');

  const links = await writeRunRoot(HOSTILE, 'link-lanes', 'HalfCheetah-v4', '2026-08-02T05:00:00Z');
  await writeLaneV4(links, 'Pendulum-cont-v0', 0, 7, 1, [4]);

  const linkedActions = join(links, 'HalfCheetah-v4', 'seed7');
  await mkdir(linkedActions, { recursive: true });
  await writeFile(
    join(linkedActions, 'meta.json'),
    JSON.stringify({ envId: 'HalfCheetah-v4', seed: 7, actDim: 6, recordVersion: 3 }),
  );
  await symlink(join(OUTSIDE, 'secret.bin'), join(linkedActions, 'actions.bin'));

  const linkedMeta = join(links, 'HalfCheetah-v4', 'seed9');
  await mkdir(linkedMeta, { recursive: true });
  await Bun.write(join(linkedMeta, 'actions.bin'), recordBytes(4, 9));
  await symlink(join(OUTSIDE, 'lane', 'meta.json'), join(linkedMeta, 'meta.json'));

  await mkdir(join(links, 'FourRooms-cont-v0'), { recursive: true });
  await symlink(join(OUTSIDE, 'lane'), join(links, 'FourRooms-cont-v0', 'seed8'));
}

function makeIndex(): RunIndex {
  return new RunIndex({ benchRoot: BENCH, cacheFile: null, ttlMs: 0 });
}

function mustQuery(parsed: { query: FleetQuery } | { error: string }): FleetQuery {
  if ('error' in parsed) throw new Error(`unexpected query error: ${parsed.error}`);
  return parsed.query;
}

function fleetQuery(search: string): FleetQuery {
  return mustQuery(parseFleetQuery(new URL(`http://127.0.0.1/api/runs/x/fleet${search}`)));
}

async function jsonCacheFiles(dir: string): Promise<string[]> {
  try {
    return (await readdir(dir)).filter((n) => n.endsWith('.json')).sort();
  } catch {
    return [];
  }
}

beforeAll(async () => {
  await buildFixtures();
  await buildHostileFixtures();
});

afterAll(async () => {
  await chmod(join(HOSTILE, 'noperm-fleet', 'fleet.jsonl'), 0o644).catch(() => undefined);
  await chmod(join(HOSTILE, 'noperm-fleet', 'aggregate.jsonl'), 0o644).catch(() => undefined);
  await rm(FLEET_CACHE, { recursive: true, force: true });
});


describe('run index discovery', () => {
  it('indexes only directories whose meta.json carries fleetSchemaVersion', async () => {
    const entries = await makeIndex().list();
    const ids = entries.map((e) => e.id).sort();
    expect(ids).toEqual(['alpha-complete', 'beta-incomplete', 'epsilon-live', 'gamma-short-final']);
    expect(ids).not.toContain('delta-lane-meta');
  });

  it('emits exactly the RunIndexEntry contract fields', async () => {
    const entries = await makeIndex().list();
    const alpha = entries.find((e) => e.id === 'alpha-complete');
    expect(alpha).toBeDefined();
    expect(Object.keys(alpha as RunIndexEntry).sort()).toEqual([
      'ablationLabel',
      'batch',
      'complete',
      'envIds',
      'hasRecordings',
      'hasSummary',
      'id',
      'inFlight',
      'maxStep',
      'policies',
      'role',
      'startedAt',
      'steps',
    ]);
  });

  it('reads a completed run: final row + maxStep >= meta.steps', async () => {
    const entries = await makeIndex().list();
    const alpha = entries.find((e) => e.id === 'alpha-complete') as RunIndexEntry;
    expect(alpha.complete).toBe(true);
    expect(alpha.maxStep).toBe(200);
    expect(alpha.inFlight).toBe(false);
    expect(alpha.batch).toBe(2);
    expect(alpha.steps).toBe(200);
    expect(alpha.policies).toEqual(['oak']);
    expect(alpha.ablationLabel).toBe('model-to-keyboard');
    expect(alpha.startedAt).toBe('2026-08-02T10:00:00Z');
    expect(alpha.envIds).toEqual(['HalfCheetah-v4', 'FourRooms-cont-v0']);
    expect(alpha.hasSummary).toBe(true);
    expect(alpha.hasRecordings).toBe(true);
    expect(alpha.role).toBe('keeper');
  });

  it('calls an interrupted run incomplete despite a .done marker, and in-flight while fresh', async () => {
    const entries = await makeIndex().list();
    const beta = entries.find((e) => e.id === 'beta-incomplete') as RunIndexEntry;
    expect(await statOrNull(join(BENCH, 'beta-incomplete', '.done'))).not.toBeNull();
    expect(beta.complete).toBe(false);
    expect(beta.maxStep).toBe(200);
    expect(beta.inFlight).toBe(true);
    expect(beta.hasSummary).toBe(false);
    expect(beta.hasRecordings).toBe(false);
    expect(beta.role).toBeNull();
  });

  it('requires the final row to reach the declared horizon', async () => {
    const entries = await makeIndex().list();
    const gamma = entries.find((e) => e.id === 'gamma-short-final') as RunIndexEntry;
    expect(gamma.complete).toBe(false);
    expect(gamma.maxStep).toBe(400);
    expect(gamma.role).toBeNull();
  });

  it('sorts newest-first by startedAt', async () => {
    const entries = await makeIndex().list();
    expect(entries.map((e) => e.id)).toEqual([
      'epsilon-live',
      'alpha-complete',
      'beta-incomplete',
      'gamma-short-final',
    ]);
  });

  it('resolves ids only from the discovered set', async () => {
    const index = makeIndex();
    expect(await index.resolve('alpha-complete')).toBeDefined();
    expect(await index.resolve('delta-lane-meta')).toBeUndefined();
    expect(await index.resolve('../../etc')).toBeUndefined();
    expect(await index.resolve('')).toBeUndefined();
  });

  it('reuses cached derivations until meta/fleet mtime+size change', async () => {
    const index = makeIndex();
    const first = await index.list();
    const second = await index.list();
    expect(second).toEqual(first);

    const fleetPath = join(BENCH, 'gamma-short-final', 'fleet.jsonl');
    const prev = await Bun.file(fleetPath).text();
    await writeFile(
      fleetPath,
      `${prev}${fleetRow({
        kind: 'final',
        lane: 0,
        step: 1000,
        envId: 'DeepSea-cont-v0',
        rewardSum: 2,
        rewardRate: 0.002,
      })}\n`,
    );
    const third = await index.list();
    const gamma = third.find((e) => e.id === 'gamma-short-final') as RunIndexEntry;
    expect(gamma.maxStep).toBe(1000);
    expect(gamma.complete).toBe(true);
    await writeFile(fleetPath, prev);
  });
});

describe('recordings discovery', () => {
  it('serves record lanes in canonical env order, legacy lanes skipped', async () => {
    const recs = await scanRecordings(join(BENCH, 'alpha-complete'));
    expect(recs.map((r) => `${r.envId}/${r.seed}`)).toEqual([
      'Pendulum-cont-v0/6', 'Chain-cont-v0/5',
    ]);
    expect(recs.length).toBe(2);
  });

  it('answers empty for a run with no recordings', async () => {
    expect(await scanRecordings(join(BENCH, 'beta-incomplete'))).toEqual([]);
  });
});


describe('record discovery', () => {
  const find = async (envId: string, seed: number) => {
    const recs = await scanRecordings(join(BENCH, 'alpha-complete'));
    return recs.find((r) => r.envId === envId && r.seed === seed);
  };

  it('takes a finalized lane\'s step count and chunk table straight from meta.json', async () => {
    const rec = await find('Chain-cont-v0', 5);
    expect(rec).toBeDefined();
    expect(rec?.steps).toBe(7);
    expect(rec?.codecVersion).toBe(1);
    expect(rec?.codec).toEqual(V4_CODECS['Chain-cont-v0']);
    expect(rec?.chunks).toEqual([
      [0, 4, V4_FILE_HEADER, V4_CHUNK_HEADER + v4PayloadBytes(4)],
      [4, 3, V4_FILE_HEADER + V4_CHUNK_HEADER + v4PayloadBytes(4), V4_CHUNK_HEADER + v4PayloadBytes(3)],
    ]);
    expect(rec?.bytes).toBe(
      V4_FILE_HEADER + 2 * V4_CHUNK_HEADER + v4PayloadBytes(4) + v4PayloadBytes(3),
    );
  });

  it('walks the chunk headers when meta.json has no steps, and stops at the torn tail', async () => {
    const rec = await find('Pendulum-cont-v0', 6);
    expect(rec).toBeDefined();
    expect(rec?.steps).toBe(11);
    expect(rec?.chunks).toBeUndefined();
    expect(rec?.codecVersion).toBe(1);
    expect(rec?.codec).toEqual(V4_CODECS['Pendulum-cont-v0']);
  });

  it('re-walks a growing lane (the memo is keyed on mtime + size)', async () => {
    const root = join(FIXTURE_ROOT, 'v4-growth');
    await rm(root, { recursive: true, force: true });
    try {
      await mkdir(root, { recursive: true });
      await writeLaneV4(root, 'Chain-cont-v0', 0, 8, 1, [10, 10], { finalized: false });
      expect((await scanRecordings(root, { root }))[0]?.steps).toBe(20);

      await writeLaneV4(root, 'Chain-cont-v0', 0, 8, 1, [10, 10, 10, 5], { finalized: false });
      expect((await scanRecordings(root, { root }))[0]?.steps).toBe(35);

      await writeLaneV4(root, 'Chain-cont-v0', 0, 8, 1, [10, 10, 10, 5, 1], { finalized: false });
      expect((await scanRecordings(root, { root }))[0]?.steps).toBe(36);
    } finally {
      await rm(root, { recursive: true, force: true });
      clearStepsCache();
    }
  });

  it('refuses a file whose header is not ours, and a truncated header', async () => {
    const root = join(FIXTURE_ROOT, 'v4-bad');
    await rm(root, { recursive: true, force: true });
    try {
      await mkdir(root, { recursive: true });
      await writeLaneV4(root, 'Chain-cont-v0', 0, 8, 1, [10], { finalized: false });
      const lane = join(root, 'Chain-cont-v0', 'seed0', 'actions.bin');
      const good = new Uint8Array(await Bun.file(lane).arrayBuffer());

      const foreign = good.slice();
      new DataView(foreign.buffer).setUint32(0, 0xdeadbeef, true);
      await Bun.write(lane, foreign);
      clearStepsCache();
      expect((await scanRecordings(root, { root }))[0]?.steps).toBe(0);

      await Bun.write(lane, good.slice(0, 8));
      clearStepsCache();
      expect(await scanRecordings(root, { root })).toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
      clearStepsCache();
    }
  });
});


describe('parseRangeHeader', () => {
  const size = 6400;
  it('parses the three single-range forms', () => {
    expect(parseRangeHeader('bytes=0-127', size)).toEqual({ start: 0, end: 127 });
    expect(parseRangeHeader('bytes=6336-', size)).toEqual({ start: 6336, end: 6399 });
    expect(parseRangeHeader('bytes=-64', size)).toEqual({ start: 6336, end: 6399 });
  });
  it('clamps an over-long end and rejects an inverted range', () => {
    expect(parseRangeHeader('bytes=0-99999', size)).toEqual({ start: 0, end: 6399 });
    expect(parseRangeHeader('bytes=200-100', size)).toBe('unsatisfiable');
    expect(parseRangeHeader('bytes=99999-', size)).toBe('unsatisfiable');
  });
  it('ignores multi-range and malformed headers (serve the whole entity)', () => {
    expect(parseRangeHeader('bytes=0-1,4-5', size)).toBeNull();
    expect(parseRangeHeader('items=0-1', size)).toBeNull();
    expect(parseRangeHeader('bytes=abc', size)).toBeNull();
    expect(parseRangeHeader(null, size)).toBeNull();
  });
});


describe('http api (fixture bench root)', () => {
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

  it('GET /api/runs lists the fixture runs', async () => {
    const res = await fetch(`${base}/api/runs`);
    expect(res.status).toBe(200);
    const runs = (await res.json()) as RunIndexEntry[];
    expect(runs.length).toBe(4);
    expect(runs.some((r) => r.id === 'alpha-complete' && r.complete)).toBe(true);
  });

  it('GET /api/keeper reads the keeper file, null when absent', async () => {
    const doc = (await (await fetch(`${base}/api/keeper`)).json()) as KeeperResponse;
    expect(doc).toEqual({ keeper: 'alpha-complete' });

    const lone = join(FIXTURE_ROOT, 'keeperless');
    await mkdir(join(lone, 'bench'), { recursive: true });
    const loneApp = createApp({ benchRoot: join(lone, 'bench'), cacheFile: null, fleetCacheDir: null, ttlMs: 0 });
    const loneServer = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: loneApp.fetch });
    try {
      const res = await fetch(`http://127.0.0.1:${loneServer.port}/api/keeper`);
      expect((await res.json()) as KeeperResponse).toEqual({ keeper: null });
    } finally {
      void loneServer.stop(true);
    }
  });

  it('GET /api/runs/:id returns meta + summary + aggregate + deaths + recordings', async () => {
    const res = await fetch(`${base}/api/runs/alpha-complete`);
    expect(res.status).toBe(200);
    const detail = (await res.json()) as RunDetail;
    expect(detail.id).toBe('alpha-complete');
    expect(detail.meta.steps).toBe(200);
    expect(detail.meta.fleetSchemaVersion).toBe(1);
    expect(detail.summary).not.toBeNull();
    expect(detail.aggregate.length).toBe(2);
    expect(detail.aggregate[1]?.kind).toBe('final');
    expect(detail.deathsCount).toBe(1);
    expect(detail.recordings).toEqual([
      { envId: 'Pendulum-cont-v0', seed: 6, steps: 11 },
      { envId: 'Chain-cont-v0', seed: 5, steps: 7 },
    ]);
  });

  it('GET /api/runs/:id/fleet returns a columnar reduction with dotted health fields', async () => {
    const res = await fetch(`${base}/api/runs/alpha-complete/fleet?policy=oak&fields=rewardSum,health.rewardRate`);
    expect(res.status).toBe(200);
    const col = (await res.json()) as FleetColumnar;
    expect(col.steps).toEqual([100, 200]);
    expect(col.lanes).toEqual([
      { lane: 0, seed: 0, policy: 'oak', envId: 'HalfCheetah-v4' },
      { lane: 1, seed: 1, policy: 'oak', envId: 'FourRooms-cont-v0' },
    ]);
    expect(col.series['rewardSum']).toEqual([
      [-5, -8],
      [3, 9],
    ]);
    expect(col.series['health.rewardRate']?.[1]).toEqual([Math.fround(0.03), 0.045]);
  });

  it('GET /api/runs/:id/fleet rejects unsafe field paths and unknown policies', async () => {
    expect((await fetch(`${base}/api/runs/alpha-complete/fleet?fields=health.__proto__`)).status).toBe(400);
    expect((await fetch(`${base}/api/runs/alpha-complete/fleet?fields=../../etc`)).status).toBe(400);
    expect((await fetch(`${base}/api/runs/alpha-complete/fleet?policy=a/b`)).status).toBe(400);
    const empty = (await (
      await fetch(`${base}/api/runs/alpha-complete/fleet?policy=random&fields=rewardSum`)
    ).json()) as FleetColumnar;
    expect(empty.lanes).toEqual([]);
    expect(empty.steps).toEqual([]);
  });

  it('serves actions.bin whole, by range, and by suffix range', async () => {
    const url = `${base}/api/runs/alpha-complete/recordings/Chain-cont-v0/5/actions.bin`;
    const size = V4_FILE_HEADER + 2 * V4_CHUNK_HEADER + v4PayloadBytes(4) + v4PayloadBytes(3);
    const whole = await fetch(url);
    expect(whole.status).toBe(200);
    expect(whole.headers.get('accept-ranges')).toBe('bytes');
    expect(whole.headers.get('content-type')).toBe('application/octet-stream');
    const wholeBuf = await whole.arrayBuffer();
    expect(wholeBuf.byteLength).toBe(size);

    const ranged = await fetch(url, { headers: { range: 'bytes=64-127' } });
    expect(ranged.status).toBe(206);
    expect(ranged.headers.get('content-range')).toBe(`bytes 64-${size - 1}/${size}`);
    expect((await ranged.arrayBuffer()).byteLength).toBe(size - 64);

    const suffix = await fetch(url, { headers: { range: 'bytes=-64' } });
    expect(suffix.status).toBe(206);
    expect(suffix.headers.get('content-range')).toBe(`bytes ${size - 64}-${size - 1}/${size}`);
    expect((await suffix.arrayBuffer()).byteLength).toBe(64);
  });

  it('answers 416 for an unsatisfiable range and 304 for a matching ETag', async () => {
    const url = `${base}/api/runs/alpha-complete/recordings/Chain-cont-v0/5/actions.bin`;
    const size = V4_FILE_HEADER + 2 * V4_CHUNK_HEADER + v4PayloadBytes(4) + v4PayloadBytes(3);
    const bad = await fetch(url, { headers: { range: 'bytes=99999-' } });
    expect(bad.status).toBe(416);
    expect(bad.headers.get('content-range')).toBe(`bytes */${size}`);

    const first = await fetch(url);
    const etag = first.headers.get('etag');
    expect(etag).not.toBeNull();
    await first.arrayBuffer();
    const second = await fetch(url, { headers: { 'if-none-match': etag ?? '' } });
    expect(second.status).toBe(304);
  });

  it('serves the sibling per-lane meta.json', async () => {
    const res = await fetch(`${base}/api/runs/alpha-complete/recordings/Chain-cont-v0/5/meta.json`);
    expect(res.status).toBe(200);
    const meta = (await res.json()) as { recordVersion: number; seed: number };
    expect(meta.recordVersion).toBe(4);
    expect(meta.seed).toBe(5);
  });

  it('the lane descriptor carries the codec, chunk table and step count', async () => {
    interface Descriptor {
      envId: string; seed: number; steps: number; actDim: number;
      bytes: number;
      codec?: unknown; codecVersion?: number; chunks?: [number, number, number, number][];
    }
    const v4 = (await (
      await fetch(`${base}/api/runs/alpha-complete/recordings/Chain-cont-v0/5`)
    ).json()) as Descriptor;
    expect(Object.keys(v4).sort()).toEqual([
      'actDim', 'bytes', 'chunks', 'codec', 'codecVersion', 'envId', 'seed', 'steps',
    ]);
    expect(v4.steps).toBe(7);
    expect(v4.codecVersion).toBe(1);
    expect(v4.codec).toEqual(V4_CODECS['Chain-cont-v0']);
    expect(v4.chunks?.length).toBe(2);
    expect(v4.chunks?.[0]?.[0]).toBe(0);
    expect(v4.chunks?.[1]?.[1]).toBe(3);

    const inFlight = (await (
      await fetch(`${base}/api/runs/alpha-complete/recordings/Pendulum-cont-v0/6`)
    ).json()) as Descriptor;
    expect(inFlight.steps).toBe(11);
    expect(inFlight.chunks).toBeUndefined();
    expect(inFlight.codecVersion).toBe(1);
  });

  it('serves a v4 actions.bin whole and by range, bytes untouched', async () => {
    const url = `${base}/api/runs/alpha-complete/recordings/Chain-cont-v0/5/actions.bin`;
    const whole = new Uint8Array(await (await fetch(url)).arrayBuffer());
    const expected =
      V4_FILE_HEADER + 2 * V4_CHUNK_HEADER + v4PayloadBytes(4) + v4PayloadBytes(3);
    expect(whole.byteLength).toBe(expected);
    expect(new DataView(whole.buffer).getUint32(0, true)).toBe(V4_FILE_MAGIC);

    const firstChunkEnd = V4_FILE_HEADER + V4_CHUNK_HEADER + v4PayloadBytes(4);
    const ranged = await fetch(url, { headers: { range: `bytes=0-${firstChunkEnd - 1}` } });
    expect(ranged.status).toBe(206);
    const prefix = new Uint8Array(await ranged.arrayBuffer());
    expect(prefix.byteLength).toBe(firstChunkEnd);
    expect(prefix).toEqual(whole.subarray(0, firstChunkEnd));
  });

  it('never lets a request param reach the filesystem', async () => {
    const cases: [string, number][] = [
      ['/api/runs/nope-not-a-run', 404],
      ['/api/runs/delta-lane-meta', 404],
      ['/api/runs/%2e%2e%2f%2e%2e%2fetc%2fpasswd', 400],
      ['/api/runs/alpha-complete/recordings/%2e%2e%2f%2e%2e%2fetc/0/actions.bin', 400],
      ['/api/runs/alpha-complete/recordings/NotAnEnv-v9/4/actions.bin', 404],
      ['/api/runs/alpha-complete/recordings/DeepSea-cont-v0/2/actions.bin', 404],
      ['/api/runs/alpha-complete/recordings/HalfCheetah-v4/9/actions.bin', 404],
      ['/api/runs/alpha-complete/recordings/HalfCheetah-v4/0/summary.json', 404],
      ['/api/runs/alpha-complete/nope', 404],
      ['/api/nope', 404],
    ];
    for (const [path, want] of cases) {
      const res = await fetch(`${base}${path}`);
      expect(`${path} -> ${res.status}`).toBe(`${path} -> ${want}`);
      await res.arrayBuffer();
    }
  });

  it('rejects non-GET methods', async () => {
    const res = await fetch(`${base}/api/runs`, { method: 'POST' });
    expect(res.status).toBe(405);
  });
});

describe('production static serving', () => {
  const DIST = join(FIXTURE_ROOT, 'dist');
  let server: ReturnType<typeof Bun.serve> | null = null;
  let base = '';

  beforeAll(async () => {
    await rm(DIST, { recursive: true, force: true });
    await mkdir(join(DIST, 'assets'), { recursive: true });
    await writeFile(join(DIST, 'index.html'), '<!doctype html><title>oak</title><div id="app"></div>');
    await writeFile(join(DIST, 'assets', 'app.js'), 'export const x = 1;\n');
    const app = createApp({
      benchRoot: BENCH,
      cacheFile: null,
      fleetCacheDir: null,
      ttlMs: 0,
      production: true,
      distDir: DIST,
    });
    server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: app.fetch });
    base = `http://127.0.0.1:${server.port}`;
  });

  afterAll(async () => {
    void server?.stop(true);
    await rm(DIST, { recursive: true, force: true });
  });

  it('serves built assets with the right content type', async () => {
    const res = await fetch(`${base}/assets/app.js`);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('text/javascript');
    expect(await res.text()).toContain('export const x');
  });

  it('falls back to index.html for client routes', async () => {
    for (const path of ['/', '/gallery', '/theater/half-cheetah/3']) {
      const res = await fetch(`${base}${path}`);
      expect(res.status).toBe(200);
      expect(res.headers.get('content-type')).toContain('text/html');
      expect(await res.text()).toContain('id="app"');
    }
  });

  it('never falls back for /api paths', async () => {
    const res = await fetch(`${base}/api/nope`);
    expect(res.status).toBe(404);
    expect(res.headers.get('content-type')).toContain('application/json');
  });

  it('refuses to escape the dist directory', async () => {
    const res = await fetch(`${base}/%2e%2e%2f%2e%2e%2fpackage.json`);
    expect(res.status).toBe(400);
    const climbed = await fetch(`${base}/../../package.json`);
    expect(climbed.status).toBeGreaterThanOrEqual(200);
    expect(await climbed.text()).not.toContain('"oak-frontend"');
  });
});

describe('hostile run roots (error isolation)', () => {
  let server: ReturnType<typeof Bun.serve> | null = null;
  let base = '';

  beforeAll(() => {
    const app = createApp({ benchRoot: HOSTILE, cacheFile: null, fleetCacheDir: null, ttlMs: 0 });
    server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: app.fetch });
    base = `http://127.0.0.1:${server.port}`;
  });

  afterAll(() => {
    void server?.stop(true);
  });

  it('indexes every healthy root while a fleet.jsonl-as-directory sits beside them', async () => {
    const index = new RunIndex({ benchRoot: HOSTILE, cacheFile: null, ttlMs: 0 });
    const entries = await index.list();
    const ids = entries.map((e) => e.id).sort();
    expect(ids).toContain('good-run');
    expect(ids).toContain('link-lanes');
    expect(ids).toContain('stale-cache');
    const good = entries.find((e) => e.id === 'good-run') as RunIndexEntry;
    expect(good.complete).toBe(true);
    expect(good.maxStep).toBe(200);
    expect(good.hasRecordings).toBe(true);
  });

  it('GET /api/runs stays 200 with an EISDIR and an EACCES root present', async () => {
    const res = await fetch(`${base}/api/runs`);
    expect(res.status).toBe(200);
    const runs = (await res.json()) as RunIndexEntry[];
    expect(runs.some((r) => r.id === 'good-run' && r.complete)).toBe(true);
    for (const id of ['dir-fleet', 'noperm-fleet']) {
      const bad = runs.find((r) => r.id === id);
      if (bad !== undefined) {
        expect(bad.complete).toBe(false);
        expect(bad.maxStep).toBe(0);
      }
    }
  });

  it('serves a run detail whose aggregate.jsonl is unreadable', async () => {
    const res = await fetch(`${base}/api/runs/noperm-fleet`);
    expect(res.status).toBe(200);
    const detail = (await res.json()) as RunDetail;
    expect(detail.id).toBe('noperm-fleet');
    expect(detail.aggregate).toEqual([]);
    expect(detail.deathsCount).toBe(0);
  });

  it('fleet endpoint fails soft (empty columnar, no path leak) on unreadable fleet.jsonl', async () => {
    for (const id of ['dir-fleet', 'noperm-fleet']) {
      const res = await fetch(`${base}/api/runs/${id}/fleet?fields=rewardSum`);
      expect(res.status).toBe(200);
      const col = (await res.json()) as FleetColumnar;
      expect(col.steps).toEqual([]);
      expect(col.lanes).toEqual([]);
    }
  });

  it('answers the fleet endpoint for a healthy root in the same bench dir', async () => {
    const res = await fetch(`${base}/api/runs/good-run/fleet?fields=rewardSum`);
    expect(res.status).toBe(200);
    const col = (await res.json()) as FleetColumnar;
    expect(col.steps).toEqual([200]);
    expect(col.series['rewardSum']).toEqual([[1]]);
  });
});


describe('recording symlink containment', () => {
  let server: ReturnType<typeof Bun.serve> | null = null;
  let base = '';

  beforeAll(() => {
    const app = createApp({ benchRoot: HOSTILE, cacheFile: null, fleetCacheDir: null, ttlMs: 0 });
    server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: app.fetch });
    base = `http://127.0.0.1:${server.port}`;
  });

  afterAll(() => {
    void server?.stop(true);
  });

  it('discovers only the real lane, never the three symlinked ones', async () => {
    const recs = await scanRecordings(join(HOSTILE, 'link-lanes'), { root: HOSTILE });
    expect(recs.map((r) => `${r.envId}/${r.seed}`)).toEqual(['Pendulum-cont-v0/0']);
    for (const r of recs) {
      expect(r.actionsPath.startsWith(HOSTILE)).toBe(true);
      expect(r.metaPath.startsWith(HOSTILE)).toBe(true);
    }
  });

  it('404s every symlinked lane over HTTP and never leaks the target bytes', async () => {
    const cases = [
      '/api/runs/link-lanes/recordings/HalfCheetah-v4/7/actions.bin',
      '/api/runs/link-lanes/recordings/HalfCheetah-v4/9/meta.json',
      '/api/runs/link-lanes/recordings/FourRooms-cont-v0/8/actions.bin',
    ];
    for (const path of cases) {
      const res = await fetch(`${base}${path}`);
      const body = await res.text();
      expect(`${path} -> ${res.status}`).toBe(`${path} -> 404`);
      expect(body).not.toContain('TOP-SECRET');
    }
  });

  it('still serves the legitimate lane sitting next to them', async () => {
    const res = await fetch(`${base}/api/runs/link-lanes/recordings/Pendulum-cont-v0/0/actions.bin`);
    expect(res.status).toBe(200);
    expect((await res.arrayBuffer()).byteLength).toBe(V4_FILE_HEADER + V4_CHUNK_HEADER + v4PayloadBytes(4));
  });
});


describe('run-index derived cache invalidation', () => {
  it('sees a summary.json and a recording written after the run finished', async () => {
    const dir = join(HOSTILE, 'stale-cache');
    const index = new RunIndex({ benchRoot: HOSTILE, cacheFile: null, ttlMs: 0 });

    const before = (await index.list()).find((e) => e.id === 'stale-cache') as RunIndexEntry;
    expect(before.hasSummary).toBe(false);
    expect(before.hasRecordings).toBe(false);

    await writeFile(join(dir, 'summary.json'), JSON.stringify({ results: [{ policy: 'oak' }] }));
    const withSummary = (await index.list()).find((e) => e.id === 'stale-cache') as RunIndexEntry;
    expect(withSummary.hasSummary).toBe(true);
    expect(withSummary.hasRecordings).toBe(false);

    await writeLaneV4(dir, 'Pendulum-cont-v0', 5, 7, 1, [6]);
    const withRecs = (await index.list()).find((e) => e.id === 'stale-cache') as RunIndexEntry;
    expect(withRecs.hasSummary).toBe(true);
    expect(withRecs.hasRecordings).toBe(true);
    expect((await index.recordingsFor('stale-cache')).map((r) => r.seed)).toEqual([5]);
  });
});


describe('RunIndex.resolve miss throttling', () => {
  it('does not rescan for an unknown id inside the TTL window, and does after it', async () => {
    const root = join(FIXTURE_ROOT, 'bench-throttle');
    await rm(root, { recursive: true, force: true });
    await mkdir(root, { recursive: true });
    await writeRunRoot(root, 'seed-run', 'HalfCheetah-v4', '2026-08-02T04:00:00Z');

    const index = new RunIndex({ benchRoot: root, cacheFile: null, ttlMs: 60_000 });
    expect(await index.resolve('seed-run')).toBeDefined();

    await writeRunRoot(root, 'late-run', 'DeepSea-cont-v0', '2026-08-02T03:00:00Z');
    expect(await index.resolve('late-run')).toBeUndefined();

    const eager = new RunIndex({ benchRoot: root, cacheFile: null, ttlMs: 0 });
    await eager.list();
    expect(await eager.resolve('late-run')).toBeDefined();
    await rm(root, { recursive: true, force: true });
  });
});


describe('parseFleetQuery', () => {
  it('sorts fields so permutations collapse to one cache entry', () => {
    const a = fleetQuery('?fields=rewardSum,health.rewardRate');
    const b = fleetQuery('?fields=health.rewardRate,rewardSum');
    expect(a.fields).toEqual(['health.rewardRate', 'rewardSum']);
    expect(a.fields).toEqual(b.fields);
  });

  it('accepts a known envId, treats empty as absent, and rejects garbage', () => {
    expect(fleetQuery('?envId=HalfCheetah-v4').envId).toBe('HalfCheetah-v4');
    expect(fleetQuery('?envId=').envId).toBeNull();
    expect(fleetQuery('').envId).toBeNull();
    const bad = parseFleetQuery(new URL('http://127.0.0.1/x?envId=NotAnEnv-v9'));
    expect('error' in bad).toBe(true);
  });

  it('still rejects unsafe field paths and bad policies', () => {
    expect('error' in parseFleetQuery(new URL('http://127.0.0.1/x?fields=health.__proto__'))).toBe(true);
    expect('error' in parseFleetQuery(new URL('http://127.0.0.1/x?policy=a/b'))).toBe(true);
  });
});

describe('fleet envId filtering over HTTP', () => {
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

  it('restricts the columnar result to the requested env lanes', async () => {
    const res = await fetch(`${base}/api/runs/alpha-complete/fleet?envId=HalfCheetah-v4&fields=rewardSum`);
    expect(res.status).toBe(200);
    const col = (await res.json()) as FleetColumnar;
    expect(col.lanes).toEqual([{ lane: 0, seed: 0, policy: 'oak', envId: 'HalfCheetah-v4' }]);
    expect(col.series['rewardSum']).toEqual([[-5, -8]]);
  });

  it('keys the cache by envId — a second env answers with its own lanes', async () => {
    const col = (await (
      await fetch(`${base}/api/runs/alpha-complete/fleet?envId=FourRooms-cont-v0&fields=rewardSum`)
    ).json()) as FleetColumnar;
    expect(col.lanes).toEqual([{ lane: 1, seed: 1, policy: 'oak', envId: 'FourRooms-cont-v0' }]);
    expect(col.series['rewardSum']).toEqual([[3, 9]]);

    const all = (await (
      await fetch(`${base}/api/runs/alpha-complete/fleet?fields=rewardSum`)
    ).json()) as FleetColumnar;
    expect(all.lanes.length).toBe(2);
  });

  it('answers empty for an env with no lanes in this run, and 400 for garbage', async () => {
    const empty = (await (
      await fetch(`${base}/api/runs/alpha-complete/fleet?envId=Pendulum-cont-v0&fields=rewardSum`)
    ).json()) as FleetColumnar;
    expect(empty.lanes).toEqual([]);
    expect(empty.steps).toEqual([]);
    expect(empty.series['rewardSum']).toEqual([]);

    expect((await fetch(`${base}/api/runs/alpha-complete/fleet?envId=NotAnEnv-v9`)).status).toBe(400);
    expect((await fetch(`${base}/api/runs/alpha-complete/fleet?envId=../etc`)).status).toBe(400);
  });

  it('ignores an unknown query parameter (maxLanes is client-side)', async () => {
    const res = await fetch(`${base}/api/runs/alpha-complete/fleet?fields=rewardSum&maxLanes=1`);
    expect(res.status).toBe(200);
    expect(((await res.json()) as FleetColumnar).lanes.length).toBe(2);
  });
});


describe('fleet disk cache', () => {
  const dir = join(BENCH, 'alpha-complete');

  it('re-derives instead of trusting a record whose identity does not match', async () => {
    await rm(FLEET_CACHE, { recursive: true, force: true });
    clearFleetCache();
    const query = fleetQuery('?fields=rewardSum');
    const first = await deriveFleetColumnar({ id: 'alpha-complete', dir, query, inFlight: false, cacheDir: FLEET_CACHE });
    expect(first.steps).toEqual([100, 200]);

    await waitUntilAsync(async () => (await jsonCacheFiles(FLEET_CACHE)).length === 1);
    const [name] = await jsonCacheFiles(FLEET_CACHE);
    const path = join(FLEET_CACHE, name ?? '');
    const rec = JSON.parse(await Bun.file(path).text()) as Record<string, unknown>;
    expect(rec['v']).toBe(3);
    expect(rec['fields']).toEqual(['rewardSum']);
    expect(rec['envId']).toBeNull();

    rec['fields'] = ['health.rewardRate'];
    rec['data'] = { steps: [-1], lanes: [], series: { 'health.rewardRate': [] } };
    await writeFile(path, JSON.stringify(rec));

    clearFleetCache();
    const second = await deriveFleetColumnar({ id: 'alpha-complete', dir, query, inFlight: false, cacheDir: FLEET_CACHE });
    expect(second.steps).toEqual([100, 200]);
    expect(second.series['rewardSum']).toEqual([
      [-5, -8],
      [3, 9],
    ]);
  });

  it('rejects a record written for a different policy or envId under the same name', async () => {
    await rm(FLEET_CACHE, { recursive: true, force: true });
    clearFleetCache();
    const query = fleetQuery('?fields=rewardSum&envId=HalfCheetah-v4');
    await deriveFleetColumnar({ id: 'alpha-complete', dir, query, inFlight: false, cacheDir: FLEET_CACHE });
    await waitUntilAsync(async () => (await jsonCacheFiles(FLEET_CACHE)).length === 1);
    const [name] = await jsonCacheFiles(FLEET_CACHE);
    const path = join(FLEET_CACHE, name ?? '');
    const rec = JSON.parse(await Bun.file(path).text()) as Record<string, unknown>;
    expect(rec['envId']).toBe('HalfCheetah-v4');

    rec['envId'] = 'FourRooms-cont-v0';
    rec['data'] = { steps: [-1], lanes: [], series: { rewardSum: [] } };
    await writeFile(path, JSON.stringify(rec));
    clearFleetCache();
    const again = await deriveFleetColumnar({ id: 'alpha-complete', dir, query, inFlight: false, cacheDir: FLEET_CACHE });
    expect(again.steps).toEqual([100, 200]);
    expect(again.lanes.map((l) => l.lane)).toEqual([0]);
  });

  it('shares one parse between simultaneous identical requests (single-flight)', async () => {
    clearFleetCache();
    const query = fleetQuery('?fields=rewardSum,health.rewardRate');
    const calls = Array.from({ length: 8 }, () =>
      deriveFleetColumnar({ id: 'alpha-complete', dir, query, inFlight: true, cacheDir: null }),
    );
    const results = await Promise.all(calls);
    for (const r of results) expect(r).toBe(results[0] as FleetColumnar);
    expect(results[0]?.steps).toEqual([100, 200]);
  });

  it('serves distinct concurrent queries correctly under the compute semaphore', async () => {
    clearFleetCache();
    const queries = ['rewardSum', 'health.rewardRate', 'doneCount', 'kernelRho', 'paramAbsMax', 'wallSeconds'];
    const results = await Promise.all(
      queries.map((f) =>
        deriveFleetColumnar({
          id: 'alpha-complete',
          dir,
          query: fleetQuery(`?fields=${f}`),
          inFlight: true,
          cacheDir: null,
        }),
      ),
    );
    for (let i = 0; i < queries.length; i++) {
      const field = queries[i] as string;
      expect(Object.keys(results[i]?.series ?? {})).toEqual([field]);
      expect(results[i]?.lanes.length).toBe(2);
    }
  });

  it('caps the cache directory, evicting oldest-by-mtime on write', async () => {
    await rm(FLEET_CACHE, { recursive: true, force: true });
    clearFleetCache();
    for (let i = 0; i < 70; i++) {
      await deriveFleetColumnar({
        id: 'alpha-complete',
        dir,
        query: fleetQuery(`?policy=p${i}&fields=rewardSum`),
        inFlight: false,
        cacheDir: FLEET_CACHE,
      });
    }
    await waitUntilAsync(async () => (await jsonCacheFiles(FLEET_CACHE)).length <= 64);
    const files = await jsonCacheFiles(FLEET_CACHE);
    expect(files.length).toBeGreaterThan(0);
    expect(files.length).toBeLessThanOrEqual(64);
    await rm(FLEET_CACHE, { recursive: true, force: true });
  });
});


describe('derived cache paths', () => {
  it('keys the run-index file and fleet cache dir by bench root', () => {
    const a = '/tmp/bench-a';
    const b = '/tmp/bench-b';
    expect(runIndexCacheFile(a)).not.toBe(runIndexCacheFile(b));
    expect(fleetCacheDir(a)).not.toBe(fleetCacheDir(b));
    expect(runIndexCacheFile(a)).toBe(runIndexCacheFile(a));
    expect(runIndexCacheFile(a).startsWith(CACHE_DIR)).toBe(true);
    expect(fleetCacheDir(a).startsWith(join(CACHE_DIR, 'fleet'))).toBe(true);
    expect(runIndexCacheFile(a)).toBe(runIndexCacheFile(`${a}/`));
  });
});


describe('origin policy', () => {
  let dev: ReturnType<typeof Bun.serve> | null = null;
  let prod: ReturnType<typeof Bun.serve> | null = null;
  let devBase = '';
  let prodBase = '';

  beforeAll(async () => {
    const devApp = createApp({ benchRoot: BENCH, cacheFile: null, fleetCacheDir: null, ttlMs: 0 });
    dev = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: devApp.fetch });
    devBase = `http://127.0.0.1:${dev.port}`;

    const dist = join(FIXTURE_ROOT, 'dist-cors');
    await mkdir(dist, { recursive: true });
    await writeFile(join(dist, 'index.html'), '<!doctype html><div id="app"></div>');
    const prodApp = createApp({
      benchRoot: BENCH,
      cacheFile: null,
      fleetCacheDir: null,
      ttlMs: 0,
      production: true,
      distDir: dist,
    });
    prod = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: prodApp.fetch });
    prodBase = `http://127.0.0.1:${prod.port}`;
  });

  afterAll(async () => {
    void dev?.stop(true);
    void prod?.stop(true);
    await rm(join(FIXTURE_ROOT, 'dist-cors'), { recursive: true, force: true });
  });

  it('echoes only the two Vite dev origins', async () => {
    for (const origin of ['http://localhost:5173', 'http://127.0.0.1:5173']) {
      const res = await fetch(`${devBase}/api/runs`, { headers: { origin } });
      expect(res.headers.get('access-control-allow-origin')).toBe(origin);
      expect(res.headers.get('vary')).toBe('origin');
      await res.arrayBuffer();
    }
  });

  it('grants nothing to any other origin (no wildcard)', async () => {
    for (const origin of ['https://evil.example', 'http://localhost:5174', 'null']) {
      const res = await fetch(`${devBase}/api/runs`, { headers: { origin } });
      expect(res.headers.get('access-control-allow-origin')).toBeNull();
      await res.arrayBuffer();
    }
    const bare = await fetch(`${devBase}/api/runs`);
    expect(bare.headers.get('access-control-allow-origin')).toBeNull();
    await bare.arrayBuffer();
  });

  it('answers the preflight with the allow-list, not *', async () => {
    const res = await fetch(`${devBase}/api/runs`, {
      method: 'OPTIONS',
      headers: { origin: 'http://localhost:5173', 'access-control-request-method': 'GET' },
    });
    expect(res.status).toBe(204);
    expect(res.headers.get('access-control-allow-origin')).toBe('http://localhost:5173');
    expect(res.headers.get('access-control-allow-methods')).toContain('GET');
  });

  it('stamps byte-range recording responses too', async () => {
    const res = await fetch(`${devBase}/api/runs/alpha-complete/recordings/Chain-cont-v0/5/actions.bin`, {
      headers: { origin: 'http://127.0.0.1:5173', range: 'bytes=0-63' },
    });
    expect(res.status).toBe(206);
    expect(res.headers.get('access-control-allow-origin')).toBe('http://127.0.0.1:5173');
    expect(res.headers.get('access-control-expose-headers')).toContain('content-range');
    await res.arrayBuffer();
  });

  it('emits no cross-origin grant at all in production', async () => {
    const res = await fetch(`${prodBase}/api/runs`, { headers: { origin: 'http://localhost:5173' } });
    expect(res.status).toBe(200);
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
    await res.arrayBuffer();
  });
});

describe('conditional requests', () => {
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

  const url = (): string => `${base}/api/runs/alpha-complete/recordings/Chain-cont-v0/5/actions.bin`;

  it('304s on If-None-Match: * when the representation exists (RFC 7232 §3.2)', async () => {
    const res = await fetch(url(), { headers: { 'if-none-match': '*' } });
    expect(res.status).toBe(304);
    expect(res.headers.get('etag')).not.toBeNull();
  });

  it('304s on a weak validator and on a list containing the tag', async () => {
    const first = await fetch(url());
    const etag = first.headers.get('etag') ?? '';
    await first.arrayBuffer();
    expect((await fetch(url(), { headers: { 'if-none-match': `W/${etag}` } })).status).toBe(304);
    const list = await fetch(url(), { headers: { 'if-none-match': `"nope", ${etag}` } });
    expect(list.status).toBe(304);
  });

  it('still 200s for a non-matching tag and 404s a missing representation', async () => {
    const res = await fetch(url(), { headers: { 'if-none-match': '"not-the-tag"' } });
    expect(res.status).toBe(200);
    await res.arrayBuffer();
    const missing = await fetch(
      `${base}/api/runs/alpha-complete/recordings/HalfCheetah-v4/9/actions.bin`,
      { headers: { 'if-none-match': '*' } },
    );
    expect(missing.status).toBe(404);
    await missing.arrayBuffer();
  });
});
