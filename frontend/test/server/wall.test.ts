/// <reference types="bun" />

import { afterAll, beforeAll, describe, expect, it } from 'bun:test';
import { cpSync } from 'node:fs';
import { mkdtemp, mkdir, readdir, rm, stat, truncate, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { EnvId } from '../../shared/envs';
import type { RecordingMeta, WallResponse } from '../../shared/types';
import { createApp } from '../../server/app';
import { scanRecordings, clearStepsCache } from '../../server/recordings';
import {
  WALL_CODEC_VERSION,
  WALL_DEFAULT_STRIDE,
  WALL_LANE_OVERSHOOT,
  WALL_MAX_DECODE_ROWS,
  WALL_MAX_RAW_BYTES,
  WALL_MAX_STRIDE,
  WALL_MIN_STRIDE,
  WALL_SAMPLE_BYTES,
  clearWallCache,
  decodedRowsAt,
  deriveWall,
  effectiveStride,
  parseWallRequest,
  wallCacheBytes,
  wallCacheFor,
  type WallPayload,
} from '../../server/wall';
import {
  CHUNK_HEADER_BYTES,
  CODEC_VERSION,
  ESCAPE_ENTRY_BYTES,
  FILE_HEADER_BYTES,
  RECORD_BYTES,
  RECORD_FLOATS,
  RECORD_VERSION,
  codecForEnv,
  crc32,
  envKindOf,
  parseActionsBin,
} from '../../src/sim/record';
import { fetchWall, wallUrl, WALL_POSE_LEN } from '../../src/api';

interface UnpackedLane {
  seed: number;
  steps: number;
  samples: number;
  firstNonFiniteRow: number | null;
  pose: Float32Array;
  done: Uint8Array;
}

async function inflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(copy);
      controller.close();
    },
  });
  return new Uint8Array(
    await new Response(source.pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer(),
  );
}

async function unpack(body: Uint8Array): Promise<{ header: WallResponse; lanes: UnpackedLane[] }> {
  const nl = body.indexOf(0x0a);
  expect(nl).toBeGreaterThan(0);
  const header = JSON.parse(new TextDecoder().decode(body.subarray(0, nl))) as WallResponse;
  expect(header.v).toBe(1);
  expect(header.headerBytes).toBe(nl);
  const compressed = body.subarray(nl + 1);
  expect(header.binaryBytes).toBe(compressed.byteLength);

  let want = 0;
  for (const lane of header.lanes) want += lane.samples * WALL_SAMPLE_BYTES;
  const flat = compressed.byteLength === 0 ? new Uint8Array(0) : await inflateRaw(compressed);
  expect(flat.byteLength).toBe(want);

  const view = new DataView(flat.buffer, flat.byteOffset, flat.byteLength);
  const lanes: UnpackedLane[] = [];
  let off = 0;
  for (const lane of header.lanes) {
    const pose = new Float32Array(lane.samples * 9);
    const done = new Uint8Array(lane.samples);
    for (let s = 0; s < lane.samples; s++) {
      const base = off + s * WALL_SAMPLE_BYTES;
      for (let j = 0; j < 9; j++) pose[s * 9 + j] = view.getFloat32(base + j * 4, true);
      done[s] = flat[base + 36] ?? 0;
    }
    off += lane.samples * WALL_SAMPLE_BYTES;
    lanes.push({
      seed: lane.seed,
      steps: lane.steps,
      samples: lane.samples,
      firstNonFiniteRow: lane.firstNonFiniteRow,
      pose,
      done,
    });
  }
  return { header, lanes };
}


function expectedRows(steps: number, stride: number): number[] {
  if (steps <= 0) return [];
  const rows: number[] = [];
  for (let r = 0; r < steps; r += stride) rows.push(r);
  if (rows[rows.length - 1] !== steps - 1) rows.push(steps - 1);
  return rows;
}

function expectLaneMatches(
  lane: UnpackedLane,
  reference: { data: Float32Array; steps: number },
  stride: number,
  label: string,
): void {
  const rows = expectedRows(reference.steps, stride);
  expect(`${label}: ${lane.steps} steps`).toBe(`${label}: ${reference.steps} steps`);
  expect(lane.samples).toBe(rows.length);
  for (let s = 0; s < rows.length; s++) {
    const row = rows[s] ?? 0;
    for (let j = 0; j < 9; j++) {
      const got = lane.pose[s * 9 + j];
      const want = reference.data[row * RECORD_FLOATS + 7 + j];
      if (!Object.is(got, want)) {
        throw new Error(
          `${label}: sample ${s} (row ${row}) pose[${j}] = ${String(got)}, decode says ${String(want)}`,
        );
      }
    }
    const doneF = reference.data[row * RECORD_FLOATS + 6] ?? 0;
    const wantDone = Number.isFinite(doneF) && doneF !== 0 ? 1 : 0;
    if (lane.done[s] !== wantDone) {
      throw new Error(`${label}: sample ${s} (row ${row}) done = ${String(lane.done[s])}, want ${wantDone}`);
    }
  }
}

function referenceFirstNonFinite(data: Float32Array, steps: number): number | null {
  for (let r = 0; r < steps; r++) {
    const base = r * RECORD_FLOATS;
    if (!Number.isFinite(data[base + 6] ?? 0)) return r;
    for (let j = 0; j < 9; j++) if (!Number.isFinite(data[base + 7 + j] ?? 0)) return r;
  }
  return null;
}

async function readLaneMeta(path: string): Promise<RecordingMeta> {
  return JSON.parse(await Bun.file(path).text()) as RecordingMeta;
}


async function deflateRawBytes(bytes: Uint8Array): Promise<Uint8Array> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(copy);
      controller.close();
    },
  });
  return new Uint8Array(
    await new Response(source.pipeThrough(new CompressionStream('deflate-raw'))).arrayBuffer(),
  );
}

async function v4GridLane(
  envId: EnvId,
  data: Float32Array,
  steps: number,
  chunkSteps: number,
): Promise<{ bytes: Uint8Array; chunks: [number, number, number, number][] }> {
  const desc = codecForEnv(envId);
  if (desc.ctrl !== 'index') throw new Error(`${envId} is not an index-ctrl env`);
  for (const f of desc.poseFields) if (f.enc !== 'u8') throw new Error(`${envId} has non-u8 pose columns`);

  const parts: Uint8Array[] = [];
  const fileHeader = new Uint8Array(FILE_HEADER_BYTES);
  const fv = new DataView(fileHeader.buffer);
  fv.setUint32(0, 0x344b414f, true);
  fv.setUint32(4, RECORD_VERSION, true);
  fv.setUint32(8, envKindOf(envId), true);
  fv.setUint32(12, CODEC_VERSION, true);
  parts.push(fileHeader);

  const chunks: [number, number, number, number][] = [];
  let offset = FILE_HEADER_BYTES;
  const rowBytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);

  for (let start = 0; start < steps; start += chunkSteps) {
    const count = Math.min(chunkSteps, steps - start);
    const hdr = new Uint8Array(count);
    const cols = desc.poseFields.map(() => new Uint8Array(count));
    const escapes: number[] = [];
    let doneCount = 0;

    for (let i = 0; i < count; i++) {
      const base = (start + i) * RECORD_FLOATS;
      let escape = false;
      for (let j = 0; j < RECORD_FLOATS; j++) if (!Number.isFinite(data[base + j] ?? 0)) escape = true;

      const a = data[base] ?? 0;
      let action = 0;
      if (Number.isInteger(a) && a >= 0 && a <= 15) action = a;
      else escape = true;

      for (let k = 0; k < desc.poseFields.length; k++) {
        const field = desc.poseFields[k];
        const col = cols[k];
        if (field === undefined || col === undefined) continue;
        const v = data[base + 7 + field.idx] ?? 0;
        if (!(Number.isInteger(v) && v >= 0 && v <= 255)) escape = true;
        col[i] = Number.isFinite(v) ? Math.min(255, Math.max(0, Math.round(v))) : 0;
      }

      const doneV = data[base + 6] ?? 0;
      let done = false;
      if (doneV === 1) {
        done = true;
        doneCount++;
      } else if (doneV !== 0) {
        escape = true;
      }
      hdr[i] = (action & 0x0f) | (done ? 0x10 : 0) | (escape ? 0x20 : 0);
      if (escape) escapes.push(i);
    }

    const payload = new Uint8Array(
      count + desc.poseFields.length * count + 4 + escapes.length * ESCAPE_ENTRY_BYTES,
    );
    const pv = new DataView(payload.buffer);
    let off = 0;
    payload.set(hdr, off);
    off += count;
    for (const col of cols) {
      payload.set(col, off);
      off += count;
    }
    pv.setUint32(off, escapes.length, true);
    off += 4;
    for (const rowIdx of escapes) {
      pv.setUint32(off, rowIdx, true);
      payload.set(
        rowBytes.subarray((start + rowIdx) * RECORD_BYTES, (start + rowIdx + 1) * RECORD_BYTES),
        off + 4,
      );
      off += ESCAPE_ENTRY_BYTES;
    }

    const comp = await deflateRawBytes(payload);
    const chunkHeader = new Uint8Array(CHUNK_HEADER_BYTES);
    const cv = new DataView(chunkHeader.buffer);
    cv.setUint32(0, 0x344b4843, true);
    cv.setUint32(4, start, true);
    cv.setUint32(8, count, true);
    cv.setUint32(12, payload.byteLength, true);
    cv.setUint32(16, comp.byteLength, true);
    cv.setUint32(20, crc32(comp), true);
    cv.setUint32(24, doneCount, true);
    cv.setFloat32(28, start === 0 ? 1.5 : Number.NaN, true);
    parts.push(chunkHeader, comp);
    chunks.push([start, count, offset, CHUNK_HEADER_BYTES + comp.byteLength]);
    offset += CHUNK_HEADER_BYTES + comp.byteLength;
  }

  const bytes = new Uint8Array(offset);
  let cursor = 0;
  for (const part of parts) {
    bytes.set(part, cursor);
    cursor += part.byteLength;
  }
  return { bytes, chunks };
}


let TMP = '';
let SYN_BENCH = '';
let SYN_RUN = '';
const SYN_ID = 'wall-synthetic';
const V4_NAN_ROW = 137;
const SYN_STEPS = 1200;

async function buildSynthetic(): Promise<void> {
  SYN_BENCH = join(TMP, 'bench');
  SYN_RUN = join(SYN_BENCH, SYN_ID);
  await mkdir(SYN_RUN, { recursive: true });
  await writeFile(
    join(SYN_RUN, 'meta.json'),
    JSON.stringify({
      fleetSchemaVersion: 1,
      envId: 'HalfCheetah-v4',
      batch: 3,
      seeds: [0, 1, 2],
      steps: SYN_STEPS,
      policies: ['oak'],
      startedAt: '2026-08-04T00:00:00Z',
    }),
  );
  await writeFile(join(SYN_RUN, 'fleet.jsonl'), '');

  const gridFloats = new Float32Array(SYN_STEPS * RECORD_FLOATS);
  for (let i = 0; i < SYN_STEPS; i++) {
    const base = i * RECORD_FLOATS;
    gridFloats[base] = i % 4;
    gridFloats[base + 6] = i % 300 === 299 ? 1 : 0;
    gridFloats[base + 7] = i % 11;
    gridFloats[base + 8] = i % 7;
    gridFloats[base + 9] = 9;
    gridFloats[base + 10] = 3;
    if (i === V4_NAN_ROW) gridFloats[base + 8] = Number.POSITIVE_INFINITY;
  }
  const lane = await v4GridLane('FourRooms-cont-v0', gridFloats, SYN_STEPS, 400);
  const gridDir = join(SYN_RUN, 'FourRooms-cont-v0', 'seed7');
  await mkdir(gridDir, { recursive: true });
  await writeFile(
    join(gridDir, 'meta.json'),
    JSON.stringify({
      envId: 'FourRooms-cont-v0',
      seed: 7,
      actDim: 4,
      recordVersion: 4,
      codecVersion: CODEC_VERSION,
      codec: JSON.parse(JSON.stringify(codecForEnv('FourRooms-cont-v0'))) as unknown,
      steps: SYN_STEPS,
      chunks: lane.chunks,
    }),
  );
  await Bun.write(join(gridDir, 'actions.bin'), lane.bytes);
}

beforeAll(async () => {
  TMP = await mkdtemp(join(tmpdir(), 'oak-wall-'));
  await buildSynthetic();
});

afterAll(async () => {
  clearWallCache();
  if (TMP.length > 0) await rm(TMP, { recursive: true, force: true });
});

async function deriveFor(
  benchRoot: string,
  runDir: string,
  id: string,
  envId: string,
  stride: number,
  cacheDir: string | null = null,
): Promise<WallPayload> {
  const recordings = await scanRecordings(runDir, { root: benchRoot });
  return deriveWall({ id, dir: runDir, envId, stride, recordings, inFlight: false, cacheDir });
}


describe('parseWallRequest', () => {
  const url = (q: string): URL => new URL(`http://x/api/runs/r/wall/HalfCheetah-v4${q}`);

  it('defaults to stride 64', () => {
    expect(parseWallRequest('HalfCheetah-v4', url(''))).toEqual({
      envId: 'HalfCheetah-v4',
      stride: WALL_DEFAULT_STRIDE,
    });
    expect(parseWallRequest('HalfCheetah-v4', url('?stride='))).toEqual({
      envId: 'HalfCheetah-v4',
      stride: WALL_DEFAULT_STRIDE,
    });
  });

  it('takes an explicit stride and truncates a fractional one', () => {
    expect(parseWallRequest('DeepSea-cont-v0', url('?stride=7'))).toEqual({
      envId: 'DeepSea-cont-v0',
      stride: 7,
    });
    expect(parseWallRequest('DeepSea-cont-v0', url('?stride=7.9'))).toEqual({
      envId: 'DeepSea-cont-v0',
      stride: 7,
    });
  });

  it('CLAMPS an out-of-range stride rather than rejecting it', () => {
    for (const raw of ['0', '-5', '0.4']) {
      expect(parseWallRequest('CartPole-cont-v0', url(`?stride=${raw}`))).toEqual({
        envId: 'CartPole-cont-v0',
        stride: WALL_MIN_STRIDE,
      });
    }
    for (const raw of ['4097', '999999', '1e12']) {
      expect(parseWallRequest('CartPole-cont-v0', url(`?stride=${raw}`))).toEqual({
        envId: 'CartPole-cont-v0',
        stride: WALL_MAX_STRIDE,
      });
    }
  });

  it('rejects a stride that is not a number', () => {
    for (const raw of ['abc', 'NaN', 'Infinity', '1,2']) {
      expect(parseWallRequest('CartPole-cont-v0', url(`?stride=${raw}`))).toEqual({
        error: 'bad stride',
        status: 400,
      });
    }
  });

  it('404s anything that is not one of the known envs', () => {
    for (const bad of [undefined, '', 'HalfCheetah', 'halfcheetah-v4', '..', '../../etc/passwd', '%2e%2e']) {
      expect(parseWallRequest(bad, url(''))).toEqual({ error: 'unknown env', status: 404 });
    }
  });
});

describe('effectiveStride (the payload budget)', () => {
  const budget = Math.floor(WALL_MAX_RAW_BYTES / WALL_SAMPLE_BYTES);

  it('honours the request whenever the projection fits', () => {
    expect(effectiveStride([{ steps: 10_000 }], 64)).toBe(64);
    expect(effectiveStride(Array.from({ length: 64 }, () => ({ steps: 10_000 })), 1)).toBe(1);
    expect(effectiveStride([], 8)).toBe(8);
    expect(effectiveStride([{ steps: 0 }], 8)).toBe(8);
  });

  it('coarsens rather than allocating a wall it cannot hold', () => {
    const tierL = Array.from({ length: 1024 }, () => ({ steps: 1_000_000 }));
    const s = effectiveStride(tierL, WALL_DEFAULT_STRIDE);
    expect(s).toBeGreaterThan(WALL_DEFAULT_STRIDE);
    let samples = 0;
    for (const lane of tierL) samples += Math.ceil(lane.steps / s) + 1;
    expect(samples).toBeLessThanOrEqual(budget);
    expect(samples * WALL_SAMPLE_BYTES).toBeLessThanOrEqual(WALL_MAX_RAW_BYTES);
  });

  it('never goes finer than asked, and never past the stride ceiling', () => {
    const huge = Array.from({ length: 4096 }, () => ({ steps: 10_000_000 }));
    expect(effectiveStride(huge, 1)).toBe(WALL_MAX_STRIDE);
    expect(effectiveStride([{ steps: 10_000 }], 4096)).toBe(WALL_MAX_STRIDE);
  });

  it('shows up in the header, so a client can map samples back to rows', async () => {
    clearWallCache();
    const { header } = await unpack((await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 37)).body);
    expect(header.stride).toBe(37);
    expect(header.lanes[0]?.samples).toBe(expectedRows(SYN_STEPS, 37).length);
  });
});


describe('effectiveStride (the decode budget)', () => {
  function decoded(steps: number, chunkSteps: number, stride: number): number {
    if (stride <= chunkSteps) return steps;
    let rows = 0;
    for (let start = 0; start < steps; start += chunkSteps) {
      const count = Math.min(chunkSteps, steps - start);
      const first = Math.ceil(start / stride) * stride;
      const holdsSample = first < start + count;
      const holdsFinal = start + count >= steps;
      if (holdsSample || holdsFinal) rows += count;
    }
    return rows;
  }

  it('agrees with a chunk-by-chunk count of what the projection needs', () => {
    const shapes = [
      [10_000, 1000],
      [1_000_000, 1000],
      [2500, 400],
      [2450, 400],
      [1200, 400],
      [999, 1000],
    ] as const;
    for (const stride of [1, 7, 64, 399, 400, 401, 999, 1000, 1001, 4096]) {
      for (const [steps, chunk] of shapes) {
        const got = decodedRowsAt({ steps, chunkSteps: chunk }, stride);
        const want = decoded(steps, chunk, stride);
        const label = `stride ${stride} steps ${steps}/${chunk}`;
        if (stride > chunk) expect(`${label}: ${got}`).toBe(`${label}: ${want}`);
        else expect(`${label}: ${got >= want}`).toBe(`${label}: true`);
        expect(got).toBeLessThanOrEqual(steps);
      }
    }
  });

  it('charges the WHOLE lane when there is no chunk table to skip by', () => {
    expect(decodedRowsAt({ steps: 1_000_000 }, 4096)).toBe(1_000_000);
    expect(decodedRowsAt({ steps: 1_000_000, chunkSteps: 0 }, 4096)).toBe(1_000_000);
    expect(decodedRowsAt({ steps: 10_000, chunkSteps: 1000 }, 64)).toBe(10_000);
  });

  it('coarsens a fully-recorded Tier-L until the DECODE fits, not just the payload', () => {
    const tierL = Array.from({ length: 1024 }, () => ({ steps: 1_000_000, chunkSteps: 1000 }));
    const payloadOnly = 283;
    const s = effectiveStride(tierL, WALL_DEFAULT_STRIDE);
    expect(s).toBeGreaterThan(payloadOnly);
    let rows = 0;
    for (const lane of tierL) rows += decodedRowsAt(lane, s);
    expect(rows).toBeLessThanOrEqual(WALL_MAX_DECODE_ROWS);
    let finer = 0;
    for (const lane of tierL) finer += decodedRowsAt(lane, s - 1);
    expect(finer).toBeGreaterThan(WALL_MAX_DECODE_ROWS);
    let samples = 0;
    for (const lane of tierL) samples += Math.ceil(lane.steps / s) + 1;
    expect(samples * WALL_SAMPLE_BYTES).toBeLessThanOrEqual(WALL_MAX_RAW_BYTES);
  });

  it('stops at the stride ceiling and leaves the rest to the runtime cap', () => {
    const huge = Array.from({ length: 8192 }, () => ({ steps: 10_000_000, chunkSteps: 1000 }));
    expect(effectiveStride(huge, 1)).toBe(WALL_MAX_STRIDE);
  });

  it('leaves every real fixture alone — the budgets are ceilings, not a policy', async () => {
    for (const [id, count, steps] of [['v4ab-tierS-v4', 64, 10_000], ['frontend-demo-16env-20260805', 8, 100_000]] as const) {
      const lanes = Array.from({ length: count }, () => ({ steps, chunkSteps: 1000 }));
      expect(`${id}: ${effectiveStride(lanes, WALL_DEFAULT_STRIDE)}`).toBe(`${id}: ${WALL_DEFAULT_STRIDE}`);
      expect(effectiveStride(lanes, 1)).toBe(1);
    }
  });
});

describe('wallCacheFor', () => {
  it('takes the per-bench-root directory in production, honours null, and nests a test dir', () => {
    expect(wallCacheFor('/bench', undefined)).toContain(join('.cache', 'wall'));
    expect(wallCacheFor('/bench', null)).toBeNull();
    expect(wallCacheFor('/bench', '/tmp/x')).toBe(join('/tmp/x', 'wall'));
  });

  it('keys the production directory on the bench root, so two stores cannot share entries', () => {
    expect(wallCacheFor('/bench-a', undefined)).not.toBe(wallCacheFor('/bench-b', undefined));
  });
});


describe('wall projection (synthetic bench root)', () => {
  it('samples row 0, every stride, and the final row — for every stride', async () => {
    for (const stride of [1, 3, 64, 500, 4096]) {
      clearWallCache();
      const { body } = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', stride);
      const { lanes } = await unpack(body);
      const rows = expectedRows(SYN_STEPS, stride);
      const lane = lanes[0];
      expect(lane?.samples).toBe(rows.length);
      for (let s = 0; s < rows.length; s++) {
        expect(lane?.pose[s * 9]).toBe((rows[s] ?? 0) % 11);
      }
      expect(lane?.pose[(rows.length - 1) * 9]).toBe((SYN_STEPS - 1) % 11);
    }
  });

  it('never duplicates the final row when the stride already lands on it', async () => {
    clearWallCache();
    const { body } = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 1199);
    const { lanes } = await unpack(body);
    expect(lanes[0]?.samples).toBe(2);
    expect(lanes[0]?.pose[9]).toBe(1199 % 11);
  });

  it('carries the done bit through the chunked record', async () => {
    clearWallCache();
    const v4 = await unpack((await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 1)).body);
    for (let r = 0; r < SYN_STEPS; r++) {
      expect(v4.lanes[0]?.done[r]).toBe(r % 300 === 299 ? 1 : 0);
    }
  });

  it('dates a v4 lane’s escaped non-finite row and reports its chunk events', async () => {
    clearWallCache();
    const { body } = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 64);
    const { header, lanes } = await unpack(body);

    expect(lanes[0]?.steps).toBe(SYN_STEPS);
    expect(lanes[0]?.firstNonFiniteRow).toBe(V4_NAN_ROW);

    const events = header.chunkEvents?.[0];
    expect(events?.seed).toBe(7);
    expect(events?.stepStart).toEqual([0, 400, 800]);
    expect(events?.stepCount).toEqual([400, 400, 400]);
    expect(events?.doneCount).toEqual([1, 1, 2]);
    expect(events?.rewardDelta).toEqual([1.5, null, null]);
  });

  it('matches a full parseActionsBin decode at every sampled row', async () => {
    const dir = join(SYN_RUN, 'FourRooms-cont-v0', 'seed7');
    const meta = await readLaneMeta(join(dir, 'meta.json'));
    const raw = await parseActionsBin(await Bun.file(join(dir, 'actions.bin')).arrayBuffer(), meta);
    for (const stride of [1, 7, 64]) {
      clearWallCache();
      const { lanes } = await unpack(
        (await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', stride)).body,
      );
      const lane = lanes[0];
      expect(lane).toBeDefined();
      if (lane !== undefined) expectLaneMatches(lane, raw, stride, `synthetic v4 stride ${stride}`);
    }
  });

  it('answers a valid EMPTY wall for an env with no recorded seeds', async () => {
    clearWallCache();
    const { body } = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'DeepSea-cont-v0', 64);
    const { header, lanes } = await unpack(body);
    expect(header.envId).toBe('DeepSea-cont-v0');
    expect(header.lanes).toEqual([]);
    expect(header.binaryBytes).toBe(0);
    expect(header.chunkEvents).toBeUndefined();
    expect(lanes).toEqual([]);
  });
});


async function wallFiles(dir: string): Promise<string[]> {
  try {
    return (await readdir(dir)).filter((n) => n.endsWith('.wall')).sort();
  } catch {
    return [];
  }
}

async function waitForWallFiles(dir: string, want: number): Promise<string[]> {
  for (let i = 0; i < 200; i++) {
    const names = await wallFiles(dir);
    if (names.length >= want) return names;
    await Bun.sleep(10);
  }
  return wallFiles(dir);
}

describe('wall caching', () => {
  it('serves a byte-identical body from disk, and re-derives when a lane file changes', async () => {
    const root = join(TMP, 'cache-bench');
    cpSync(SYN_BENCH, root, { recursive: true });
    const runDir = join(root, SYN_ID);
    const cacheDir = join(TMP, 'cache-store');

    clearWallCache();
    const first = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    const names = await waitForWallFiles(cacheDir, 1);
    expect(names.length).toBe(1);

    const memHit = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    expect(memHit.body).toBe(first.body);

    clearWallCache();
    const diskHit = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    expect(diskHit.body).not.toBe(first.body);
    expect(Buffer.from(diskHit.body).equals(Buffer.from(first.body))).toBe(true);
    expect(await wallFiles(cacheDir)).toEqual(names);

    const lanePath = join(runDir, 'FourRooms-cont-v0', 'seed7', 'actions.bin');
    const now = new Date(Date.now() + 5_000);
    await utimes(lanePath, now, now);
    clearWallCache();
    clearStepsCache();
    const afterTouch = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    expect(Buffer.from(afterTouch.body).equals(Buffer.from(first.body))).toBe(true);
    const names2 = await waitForWallFiles(cacheDir, 2);
    expect(names2.length).toBe(2);

    const meta = JSON.parse(await Bun.file(join(runDir, 'FourRooms-cont-v0', 'seed7', 'meta.json')).text()) as {
      chunks: [number, number, number, number][];
    };
    const firstChunk = meta.chunks[0];
    if (firstChunk === undefined) throw new Error('no chunk table');
    await truncate(lanePath, firstChunk[2] + firstChunk[3]);
    clearWallCache();
    clearStepsCache();
    const afterEdit = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    const { lanes } = await unpack(afterEdit.body);
    expect(lanes[0]?.steps).toBe(firstChunk[1]);
    expect(Buffer.from(afterEdit.body).equals(Buffer.from(first.body))).toBe(false);
  });

  it('gives each stride its own entry — one stride never answers for another', async () => {
    const cacheDir = join(TMP, 'cache-stride');
    clearWallCache();
    const a = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    const b = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 16, cacheDir);
    expect(a.header.stride).toBe(64);
    expect(b.header.stride).toBe(16);
    expect(b.header.lanes[0]?.samples).toBeGreaterThan(a.header.lanes[0]?.samples ?? 0);
    const names = await waitForWallFiles(cacheDir, 2);
    expect(names.length).toBe(2);
  });

  it('single-flights concurrent callers onto ONE derive', async () => {
    clearWallCache();
    const recordings = await scanRecordings(SYN_RUN, { root: SYN_BENCH });
    const opts = {
      id: SYN_ID,
      dir: SYN_RUN,
      envId: 'FourRooms-cont-v0',
      stride: 64,
      recordings,
      inFlight: false,
      cacheDir: null,
    };
    const [p, q, r] = await Promise.all([deriveWall(opts), deriveWall(opts), deriveWall(opts)]);
    expect(q.body).toBe(p.body);
    expect(r.body).toBe(p.body);
  });

  it('re-derives when a lane’s META changes and its actions.bin does not', async () => {
    const root = join(TMP, 'meta-key-bench');
    cpSync(SYN_BENCH, root, { recursive: true });
    const runDir = join(root, SYN_ID);
    const cacheDir = join(TMP, 'meta-key-store');
    const metaPath = join(runDir, 'FourRooms-cont-v0', 'seed7', 'meta.json');
    const lanePath = join(runDir, 'FourRooms-cont-v0', 'seed7', 'actions.bin');
    const laneBefore = await stat(lanePath);

    clearWallCache();
    const first = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    expect(first.header.lanes[0]?.steps).toBe(SYN_STEPS);
    await waitForWallFiles(cacheDir, 1);

    const meta = JSON.parse(await Bun.file(metaPath).text()) as Record<string, unknown>;
    await writeFile(metaPath, JSON.stringify({ ...meta, steps: 400 }));
    const laneAfter = await stat(lanePath);
    expect(laneAfter.size).toBe(laneBefore.size);
    expect(laneAfter.mtimeMs).toBe(laneBefore.mtimeMs);

    clearWallCache();
    clearStepsCache();
    const second = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    expect(Buffer.from(second.body).equals(Buffer.from(first.body))).toBe(false);
    const names = await waitForWallFiles(cacheDir, 2);
    expect(names.length).toBe(2);

    await writeFile(metaPath, JSON.stringify(meta));
    clearWallCache();
    clearStepsCache();
    const third = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    expect(Buffer.from(third.body).equals(Buffer.from(first.body))).toBe(true);
  });

  it('binds the row DECODER’s revision into the key, not just the projection’s', async () => {
    const cacheDir = join(TMP, 'key-record-store');
    clearWallCache();
    await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    const names = await waitForWallFiles(cacheDir, 1);
    const bytes = new Uint8Array(await Bun.file(join(cacheDir, names[0] ?? '')).arrayBuffer());
    const record = JSON.parse(
      new TextDecoder().decode(bytes.subarray(0, bytes.indexOf(0x0a))),
    ) as { key: string; bodyCrc: number; bodyBytes: number };
    expect(record.key).toContain(`|64|${WALL_CODEC_VERSION}|${CODEC_VERSION}|`);
    expect(record.key).toContain(SYN_ID);
    expect(record.bodyBytes).toBeGreaterThan(0);
    expect(record.bodyCrc).toBeGreaterThanOrEqual(0);
  });

  it('takes a cache MISS on a corrupted body instead of serving it', async () => {
    const root = join(TMP, 'corrupt-bench');
    cpSync(SYN_BENCH, root, { recursive: true });
    const runDir = join(root, SYN_ID);
    const cacheDir = join(TMP, 'corrupt-store');

    clearWallCache();
    const first = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    const names = await waitForWallFiles(cacheDir, 1);
    const cachePath = join(cacheDir, names[0] ?? '');
    const onDisk = new Uint8Array(await Bun.file(cachePath).arrayBuffer());
    expect(onDisk.byteLength).toBeGreaterThan(64);

    const flipped = onDisk.slice();
    const at = flipped.byteLength - 8;
    flipped[at] = (flipped[at] ?? 0) ^ 0x01;
    await Bun.write(cachePath, flipped);
    expect((await stat(cachePath)).size).toBe(onDisk.byteLength);

    clearWallCache();
    const second = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    expect(Buffer.from(second.body).equals(Buffer.from(flipped.subarray(flipped.indexOf(0x0a) + 1)))).toBe(false);
    expect(Buffer.from(second.body).equals(Buffer.from(first.body))).toBe(true);
    const { lanes } = await unpack(second.body);
    expect(lanes[0]?.samples).toBe(expectedRows(SYN_STEPS, 64).length);
  });

  it('never persists an in-flight run (its lane files are still growing)', async () => {
    const cacheDir = join(TMP, 'cache-inflight');
    clearWallCache();
    const recordings = await scanRecordings(SYN_RUN, { root: SYN_BENCH });
    await deriveWall({
      id: SYN_ID,
      dir: SYN_RUN,
      envId: 'FourRooms-cont-v0',
      stride: 64,
      recordings,
      inFlight: true,
      cacheDir,
    });
    await Bun.sleep(60);
    expect(await wallFiles(cacheDir)).toEqual([]);
  });
});


describe('selective chunk decoding', () => {
  it('skips whole chunks the stride never samples, and says so in scannedRows', async () => {
    clearWallCache();
    const { header, lanes } = await unpack(
      (await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 1000)).body,
    );
    const head = header.lanes[0];
    expect(head?.steps).toBe(SYN_STEPS);
    expect(head?.scannedRows).toBe(800);
    expect(lanes[0]?.samples).toBe(expectedRows(SYN_STEPS, 1000).length);

    const dir = join(SYN_RUN, 'FourRooms-cont-v0', 'seed7');
    const meta = await readLaneMeta(join(dir, 'meta.json'));
    const raw = await parseActionsBin(await Bun.file(join(dir, 'actions.bin')).arrayBuffer(), meta);
    const lane = lanes[0];
    if (lane === undefined) throw new Error('no lane');
    expectLaneMatches(lane, raw, 1000, 'skipped-chunk lane');

    expect(header.chunkEvents?.[0]?.stepStart).toEqual([0, 400, 800]);
    expect(header.chunkEvents?.[0]?.doneCount).toEqual([1, 1, 2]);
  });

  it('decodes every chunk when the stride is finer than one, and then omits scannedRows', async () => {
    for (const stride of [1, 64, 400]) {
      clearWallCache();
      const { header } = await unpack(
        (await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', stride)).body,
      );
      expect(`stride ${stride}: ${String(header.lanes[0]?.scannedRows)}`).toBe(`stride ${stride}: undefined`);
      expect(header.lanes[0]?.steps).toBe(SYN_STEPS);
    }
  });

  it('still dates a death inside a decoded chunk exactly', async () => {
    clearWallCache();
    const { header } = await unpack(
      (await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 1000)).body,
    );
    expect(header.lanes[0]?.firstNonFiniteRow).toBe(V4_NAN_ROW);
  });

  it('reports null — never a guess — for a death inside a SKIPPED chunk, and flags the scan', async () => {
    const root = join(TMP, 'skip-bench');
    const runDir = join(root, SYN_ID);
    await mkdir(join(runDir, 'DeepSea-cont-v0', 'seed1'), { recursive: true });
    await writeFile(join(runDir, 'meta.json'), await Bun.file(join(SYN_RUN, 'meta.json')).text());
    await writeFile(join(runDir, 'fleet.jsonl'), '');

    const steps = 1200;
    const floats = new Float32Array(steps * RECORD_FLOATS);
    for (let i = 0; i < steps; i++) {
      const base = i * RECORD_FLOATS;
      floats[base] = i % 2;
      floats[base + 7] = i % 7;
      floats[base + 8] = i % 7;
      floats[base + 9] = 7;
      floats[base + 10] = 7;
    }
    floats[500 * RECORD_FLOATS + 7] = Number.NaN;
    const built = await v4GridLane('DeepSea-cont-v0', floats, steps, 400);
    const dir = join(runDir, 'DeepSea-cont-v0', 'seed1');
    await writeFile(
      join(dir, 'meta.json'),
      JSON.stringify({
        envId: 'DeepSea-cont-v0',
        seed: 1,
        actDim: 2,
        recordVersion: 4,
        codecVersion: CODEC_VERSION,
        steps,
        chunks: built.chunks,
      }),
    );
    await Bun.write(join(dir, 'actions.bin'), built.bytes);

    clearWallCache();
    const coarse = await unpack((await deriveFor(root, runDir, SYN_ID, 'DeepSea-cont-v0', 1000)).body);
    expect(coarse.header.lanes[0]?.scannedRows).toBe(800);
    expect(coarse.header.lanes[0]?.firstNonFiniteRow).toBeNull();

    clearWallCache();
    const fine = await unpack((await deriveFor(root, runDir, SYN_ID, 'DeepSea-cont-v0', 64)).body);
    expect(fine.header.lanes[0]?.scannedRows).toBeUndefined();
    expect(fine.header.lanes[0]?.firstNonFiniteRow).toBe(500);
  });


  it('crc-checks every chunk it decodes, and stops at a corrupted one', async () => {
    const root = join(TMP, 'bad-crc');
    cpSync(SYN_BENCH, root, { recursive: true });
    const runDir = join(root, SYN_ID);
    const dir = join(runDir, 'FourRooms-cont-v0', 'seed7');
    const meta = (await readLaneMeta(join(dir, 'meta.json'))) as RecordingMeta & {
      chunks: [number, number, number, number][];
    };
    const third = meta.chunks[2];
    if (third === undefined) throw new Error('expected 3 chunks');

    const lanePath = join(dir, 'actions.bin');
    const bytes = new Uint8Array(await Bun.file(lanePath).arrayBuffer());
    const at = third[2] + CHUNK_HEADER_BYTES + 3;
    bytes[at] = (bytes[at] ?? 0) ^ 0x55;
    await Bun.write(lanePath, bytes);

    clearWallCache();
    clearStepsCache();
    const { header, lanes } = await unpack(
      (await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64)).body,
    );
    const head = header.lanes[0];
    expect(head?.steps).toBe(800);
    expect(head?.error).toBeUndefined();
    expect(head?.samples).toBe(expectedRows(800, 64).length);
    expect(header.chunkEvents?.[0]?.stepStart).toEqual([0, 400]);
    expect(lanes[0]?.pose[(lanes[0].samples - 1) * 9]).toBe(799 % 11);
  });

  it('keeps @sim/record’s own lane checks — a meta that contradicts the file is an error', async () => {
    const root = join(TMP, 'meta-lies');
    cpSync(SYN_BENCH, root, { recursive: true });
    const runDir = join(root, SYN_ID);
    const metaPath = join(runDir, 'FourRooms-cont-v0', 'seed7', 'meta.json');
    const good = JSON.parse(await Bun.file(metaPath).text()) as Record<string, unknown>;

    const cases: [string, Record<string, unknown>][] = [
      ['wrong codec revision', { ...good, codecVersion: CODEC_VERSION + 1 }],
      ['wrong codec descriptor', { ...good, codec: { ...(good['codec'] as object), category: 'cheetah' } }],
      ['wrong pose columns', {
        ...good,
        codec: {
          ...(good['codec'] as object),
          poseFields: [{ idx: 0, enc: 'u8' }, { idx: 1, enc: 'u8' }],
        },
      }],
    ];
    for (const [label, meta] of cases) {
      await writeFile(metaPath, JSON.stringify(meta));
      clearWallCache();
      clearStepsCache();
      const { header } = await unpack((await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64)).body);
      const head = header.lanes[0];
      expect(`${label}: ${head?.error === undefined ? 'decoded' : 'error'}`).toBe(`${label}: error`);
      expect(head?.samples).toBe(0);
    }
    await writeFile(metaPath, JSON.stringify(good));
    clearWallCache();
    clearStepsCache();
    const { header } = await unpack((await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64)).body);
    expect(header.lanes[0]?.error).toBeUndefined();
    expect(header.lanes[0]?.steps).toBe(SYN_STEPS);
  });
});


describe('budget ceilings', () => {
  async function staleRoot(name: string): Promise<string> {
    const root = join(TMP, name);
    cpSync(SYN_BENCH, root, { recursive: true });
    return root;
  }


  it('cuts a v4 lane at a CHUNK boundary, so the wall never ends mid-chunk', async () => {
    const root = join(TMP, 'stale-v4');
    const runDir = join(root, SYN_ID);
    const dir = join(runDir, 'FourRooms-cont-v0', 'seed7');
    await mkdir(dir, { recursive: true });
    await writeFile(join(runDir, 'meta.json'), await Bun.file(join(SYN_RUN, 'meta.json')).text());
    await writeFile(join(runDir, 'fleet.jsonl'), '');

    const steps = 3600;
    const floats = new Float32Array(steps * RECORD_FLOATS);
    for (let i = 0; i < steps; i++) {
      const base = i * RECORD_FLOATS;
      floats[base] = i % 4;
      floats[base + 7] = i % 11;
      floats[base + 8] = i % 7;
      floats[base + 9] = 9;
      floats[base + 10] = 3;
    }
    const built = await v4GridLane('FourRooms-cont-v0', floats, steps, 400);
    await Bun.write(join(dir, 'actions.bin'), built.bytes);
    await writeFile(
      join(dir, 'meta.json'),
      JSON.stringify({
        envId: 'FourRooms-cont-v0',
        seed: 7,
        actDim: 4,
        recordVersion: 4,
        codecVersion: CODEC_VERSION,
        steps: SYN_STEPS,
        chunks: built.chunks.slice(0, 3),
      }),
    );

    const recordings = await scanRecordings(runDir, { root });
    clearWallCache();
    const { header, lanes } = await unpack(
      (await deriveWall({
        id: SYN_ID, dir: runDir, envId: 'FourRooms-cont-v0', stride: 64,
        recordings, inFlight: false, cacheDir: null,
      })).body,
    );
    const head = header.lanes[0];
    expect(head?.truncated).toBe(true);
    expect(head?.steps).toBe(SYN_STEPS * WALL_LANE_OVERSHOOT);
    expect((head?.steps ?? 0) % 400).toBe(0);
    expect(head?.samples).toBe(expectedRows(SYN_STEPS * WALL_LANE_OVERSHOOT, 64).length);
    expect(header.chunkEvents?.[0]?.stepStart).toEqual([0, 400, 800, 1200, 1600, 2000]);
    const raw = await parseActionsBin(
      await Bun.file(join(dir, 'actions.bin')).arrayBuffer(),
      await readLaneMeta(join(dir, 'meta.json')),
    );
    const lane = lanes[0];
    if (lane === undefined) throw new Error('no lane');
    for (let s = 0; s < lane.samples; s++) {
      const row = Math.min(s * 64, (head?.steps ?? 1) - 1);
      for (let j = 0; j < 9; j++) {
        expect(Object.is(lane.pose[s * 9 + j], raw.data[row * RECORD_FLOATS + 7 + j])).toBe(true);
      }
    }
  });

  it('bounds the in-memory cache by BYTES, not only by entry count', async () => {
    clearWallCache();
    expect(wallCacheBytes()).toBe(0);
    const bodies: number[] = [];
    for (const stride of [16, 32, 64, 128]) {
      const w = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', stride);
      bodies.push(w.body.byteLength);
    }
    expect(wallCacheBytes()).toBe(bodies.reduce((a, b) => a + b, 0));
    await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', 256);
    expect(wallCacheBytes()).toBeLessThan(bodies.reduce((a, b) => a + b, 0) + 4096);
    expect(wallCacheBytes()).toBeGreaterThan(0);
    clearWallCache();
    expect(wallCacheBytes()).toBe(0);
  });
});


describe('HTTP /api/runs/:id/wall/:envId', () => {
  let server: ReturnType<typeof Bun.serve> | null = null;
  let base = '';
  const realFetch = globalThis.fetch;

  beforeAll(() => {
    const app = createApp({
      benchRoot: SYN_BENCH,
      cacheFile: null,
      fleetCacheDir: null,
      ttlMs: 0,
    });
    server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: app.fetch });
    base = `http://127.0.0.1:${server.port}`;
    globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) =>
      realFetch(typeof input === 'string' ? new URL(input, base).toString() : input, init)) as typeof fetch;
  });

  afterAll(() => {
    globalThis.fetch = realFetch;
    server?.stop(true);
  });

  it('builds the client url, stride included only when asked', () => {
    expect(wallUrl('run x', 'HalfCheetah-v4')).toBe('/api/runs/run%20x/wall/HalfCheetah-v4');
    expect(wallUrl('r', 'DeepSea-cont-v0', { stride: 8 })).toBe('/api/runs/r/wall/DeepSea-cont-v0?stride=8');
  });

  it('round-trips the framing through fetchWall, matching a server-side derive', async () => {
    clearWallCache();
    const wire = await fetchWall(SYN_ID, 'HalfCheetah-v4', { stride: 64 });
    clearWallCache();
    const direct = await unpack((await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', 64)).body);

    expect(wire.header.runId).toBe(SYN_ID);
    expect(wire.header.envId).toBe('HalfCheetah-v4');
    expect(wire.header.stride).toBe(64);
    expect(wire.lanes.length).toBe(direct.lanes.length);
    for (let i = 0; i < wire.lanes.length; i++) {
      const a = wire.lanes[i];
      const b = direct.lanes[i];
      if (a === undefined || b === undefined) throw new Error('lane count mismatch');
      expect(a.seed).toBe(b.seed);
      expect(a.steps).toBe(b.steps);
      expect(a.samples).toBe(b.samples);
      expect(a.firstNonFiniteRow).toBe(b.firstNonFiniteRow);
      expect(a.pose.length).toBe(a.samples * WALL_POSE_LEN);
      for (let k = 0; k < a.pose.length; k++) {
        expect(Object.is(a.pose[k], b.pose[k])).toBe(true);
      }
      expect(Buffer.from(a.done).equals(Buffer.from(b.done))).toBe(true);
      expect(a.rows[0]).toBe(0);
      expect(a.rows[a.samples - 1]).toBe(a.steps - 1);
      if (a.samples > 2) expect(a.rows[1]).toBe(64);
    }
  });

  it('decodes the v4 lane and its chunk events over the wire', async () => {
    clearWallCache();
    const wire = await fetchWall(SYN_ID, 'FourRooms-cont-v0', { stride: 32 });
    expect(wire.lanes.length).toBe(1);
    expect(wire.lanes[0]?.seed).toBe(7);
    expect(wire.lanes[0]?.firstNonFiniteRow).toBe(V4_NAN_ROW);
    expect(wire.header.chunkEvents?.[0]?.stepCount).toEqual([400, 400, 400]);
  });

  it('returns an empty wall — not an error — for an env with no recordings', async () => {
    clearWallCache();
    const wire = await fetchWall(SYN_ID, 'Chain-cont-v0');
    expect(wire.lanes).toEqual([]);
    expect(wire.header.binaryBytes).toBe(0);
    expect(wire.header.stride).toBe(WALL_DEFAULT_STRIDE);
  });

  it('advertises the split in headers so a HEAD answers without a body', async () => {
    clearWallCache();
    const get = await realFetch(`${base}/api/runs/${SYN_ID}/wall/HalfCheetah-v4`);
    expect(get.status).toBe(200);
    expect(get.headers.get('content-type')).toBe('application/octet-stream');
    const headerBytes = Number(get.headers.get('x-wall-header-bytes'));
    const binaryBytes = Number(get.headers.get('x-wall-binary-bytes'));
    const body = new Uint8Array(await get.arrayBuffer());
    expect(body.indexOf(0x0a)).toBe(headerBytes);
    expect(body.byteLength).toBe(headerBytes + 1 + binaryBytes);

    const head = await realFetch(`${base}/api/runs/${SYN_ID}/wall/HalfCheetah-v4`, { method: 'HEAD' });
    expect(head.status).toBe(200);
    expect(head.headers.get('x-wall-header-bytes')).toBe(String(headerBytes));
    expect((await head.arrayBuffer()).byteLength).toBe(0);
  });

  it('404s an unknown run, an unknown env, and 400s an encoded traversal', async () => {
    const cases: [string, number][] = [
      [`/api/runs/no-such-run/wall/HalfCheetah-v4`, 404],
      [`/api/runs/${SYN_ID}/wall/NotAnEnv`, 404],
      [`/api/runs/${SYN_ID}/wall/HalfCheetah`, 404],
      [`/api/runs/${SYN_ID}/wall/%2e%2e%2f%2e%2e%2fetc%2fpasswd`, 400],
      [`/api/runs/${SYN_ID}/wall/..%2f..%2fetc%2fpasswd`, 400],
      [`/api/runs/${SYN_ID}/wall/HalfCheetah-v4%00`, 400],
      [`/api/runs/${SYN_ID}/wall`, 404],
      [`/api/runs/${SYN_ID}/wall/HalfCheetah-v4/extra`, 404],
    ];
    for (const [path, status] of cases) {
      const res = await realFetch(`${base}${path}`);
      expect(`${path} -> ${res.status}`).toBe(`${path} -> ${status}`);
      expect(res.headers.get('content-type')).toContain('application/json');
    }
  });

  it('cannot be addressed with a bare dot segment: the URL parser collapses it first', async () => {
    for (const suffix of ['..', '%2e%2e']) {
      const res = await realFetch(`${base}/api/runs/${SYN_ID}/wall/${suffix}`);
      expect(new URL(res.url).pathname).toBe(`/api/runs/${SYN_ID}/`);
      expect(res.headers.get('content-type')).toContain('application/json');
      const detail = (await res.json()) as { id?: string };
      expect(detail.id).toBe(SYN_ID);
    }
  });

  it('400s a malformed stride and clamps an extreme one', async () => {
    const bad = await realFetch(`${base}/api/runs/${SYN_ID}/wall/HalfCheetah-v4?stride=abc`);
    expect(bad.status).toBe(400);
    expect(await bad.json()).toEqual({ error: 'bad stride' });

    clearWallCache();
    const hi = await fetchWall(SYN_ID, 'HalfCheetah-v4', { stride: 10_000_000 });
    expect(hi.header.stride).toBe(WALL_MAX_STRIDE);
    clearWallCache();
    const lo = await fetchWall(SYN_ID, 'HalfCheetah-v4', { stride: 0 });
    expect(lo.header.stride).toBe(WALL_MIN_STRIDE);
    clearWallCache();
    const none = await fetchWall(SYN_ID, 'HalfCheetah-v4');
    expect(none.header.stride).toBe(WALL_DEFAULT_STRIDE);
  });
});

