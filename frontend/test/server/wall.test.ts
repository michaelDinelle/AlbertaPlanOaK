/// <reference types="bun" />
// Wall endpoint tests: /api/runs/:id/wall/:envId.
//
// The claim under test is narrow and total: for EVERY recorded seed of an env, the
// bytes the wall ships at row r are the bytes a full decode of that lane produces at
// row r. So the correctness tests do not check "shapes look right" — they decode each
// lane a second time with `parseActionsBinV4` / `parseActionsBin` (the theater's own
// readers) and compare the sampled rows f32 by f32 with `Object.is`, which makes NaN
// compare equal to NaN and −0 unequal to 0.
//
// Three fixture sources, each for what only it can prove:
//   * .oak-benchmarks/v4ab-tierS-v4        — v4, 16 envs × 64 lanes × 10k. The real
//                                            all-lanes shape and the real codec table.
//   * .oak-benchmarks/frontend-demo-16env  — v3, 8 lanes × 100k. The wall must work on
//                                            the pre-v4 roots too.
//   * a temp bench root built here         — the cases real data cannot contain on
//                                            demand: a non-finite lane, a torn file, a
//                                            lane whose bytes change under the cache.
// Real roots are read ONLY; every mutation happens on a copy under the OS temp dir.

import { afterAll, beforeAll, describe, expect, it } from 'bun:test';
import { cpSync, existsSync } from 'node:fs';
import { mkdtemp, mkdir, readdir, rm, stat, truncate, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { EnvId } from '../../shared/envs';
import type { RecordingMeta, WallResponse } from '../../shared/types';
import { createApp } from '../../server/app';
import { scanRecordings, clearV4StepsCache } from '../../server/recordings';
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
import { RECORD_BYTES, RECORD_FLOATS, parseActionsBin } from '../../src/sim/record';
import {
  CHUNK_HEADER_BYTES,
  CODEC_VERSION,
  ESCAPE_ENTRY_BYTES,
  FILE_HEADER_BYTES,
  RECORD_V4_VERSION,
  codecForEnv,
  crc32,
  envKindOf,
  parseActionsBinV4,
} from '../../src/sim/record4';
import { fetchWall, wallUrl, WALL_POSE_LEN } from '../../src/api';

const HERE = dirname(fileURLToPath(import.meta.url)); // frontend/test/server
const REAL_BENCH = resolve(HERE, '..', '..', '..', '.oak-benchmarks');

const V4_ID = 'v4ab-tierS-v4';
const V4_XS_ID = 'v4probe-xs-v4';
const DEMO_ID = 'frontend-demo-16env-20260803';
/** the degraded-wall case: a real run root with no recordings at all */
const NOREC_ID = 'p93-tierL-20260721';

const v4Present = existsSync(join(REAL_BENCH, V4_ID, 'meta.json'));
const v4XsPresent = existsSync(join(REAL_BENCH, V4_XS_ID, 'meta.json'));
const demoPresent = existsSync(join(REAL_BENCH, DEMO_ID, 'meta.json'));
const norecPresent = existsSync(join(REAL_BENCH, NOREC_ID, 'meta.json'));

// ---------------------------------------------------------------------------
// framing reader — deliberately INDEPENDENT of src/api.ts's fetchWall
// ---------------------------------------------------------------------------

interface UnpackedLane {
  seed: number;
  steps: number;
  samples: number;
  firstNonFiniteRow: number | null;
  /** samples × 9 */
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

/** Split + inflate a framed wall body by the documented rule, checking every claim in it. */
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

// ---------------------------------------------------------------------------
// the sampling rule, restated independently
// ---------------------------------------------------------------------------

/** rows `0, stride, 2·stride, …` plus the final row when the stride missed it */
function expectedRows(steps: number, stride: number): number[] {
  if (steps <= 0) return [];
  const rows: number[] = [];
  for (let r = 0; r < steps; r += stride) rows.push(r);
  if (rows[rows.length - 1] !== steps - 1) rows.push(steps - 1);
  return rows;
}

/** Compare a lane's shipped samples against a full second decode of the same lane. */
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

/** First row whose pose or done is non-finite, from a full decode. */
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

// ---------------------------------------------------------------------------
// synthetic lanes (temp bench root)
// ---------------------------------------------------------------------------

/** record-v3 lane bytes: `steps × 16` f32, filled by the caller. */
function v3Bytes(steps: number, fill: (row: Float32Array, i: number) => void): Uint8Array {
  const f = new Float32Array(steps * RECORD_FLOATS);
  const row = new Float32Array(RECORD_FLOATS);
  for (let i = 0; i < steps; i++) {
    row.fill(0);
    fill(row, i);
    f.set(row, i * RECORD_FLOATS);
  }
  return new Uint8Array(f.buffer);
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

/**
 * A spec-conformant v4 encoder for the `grid` category (hdr + u8 pose columns +
 * escapes) — enough to build FourRooms/DeepSea lanes, which is all the synthetic cases
 * need. The general encoder lives in test/record4.test.ts; duplicating the grid subset
 * here keeps this file from importing another test file's internals.
 */
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
  fv.setUint32(4, RECORD_V4_VERSION, true);
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

// ---------------------------------------------------------------------------
// temp bench root
// ---------------------------------------------------------------------------

let TMP = '';
let SYN_BENCH = '';
let SYN_RUN = '';
const SYN_ID = 'wall-synthetic';
/** row the synthetic v3 lane goes non-finite at */
const V3_NAN_ROW = 500;
/** row the synthetic v4 lane goes non-finite at */
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

  // --- v3 cheetah lanes: seed 0 finite, seed 1 non-finite from V3_NAN_ROW on ---
  for (const seed of [0, 1]) {
    const dir = join(SYN_RUN, 'HalfCheetah-v4', `seed${seed}`);
    await mkdir(dir, { recursive: true });
    await writeFile(
      join(dir, 'meta.json'),
      JSON.stringify({ envId: 'HalfCheetah-v4', seed, actDim: 6, recordVersion: 3 }),
    );
    const bytes = v3Bytes(SYN_STEPS, (row, i) => {
      for (let c = 0; c < 6; c++) row[c] = (i % 17) / 16 - 0.5;
      row[6] = i % 250 === 249 ? 1 : 0;
      for (let p = 0; p < 9; p++) row[7 + p] = i * 0.01 + p;
      if (seed === 1 && i >= V3_NAN_ROW) row[7 + 3] = Number.NaN;
    });
    await Bun.write(join(dir, 'actions.bin'), bytes);
  }

  // --- v4 FourRooms lane: escaped non-finite row at V4_NAN_ROW ---
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

// ===========================================================================
// request parsing
// ===========================================================================

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
    // a fully-recorded Tier-L: 1024 seeds × 1M steps, asked for at the default stride
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
    // The synthetic root is tiny, so its stride is honoured exactly — the point here is
    // that the header REPORTS the stride used rather than the one requested.
    const { header } = await unpack((await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', 37)).body);
    expect(header.stride).toBe(37);
    expect(header.lanes[0]?.samples).toBe(expectedRows(SYN_STEPS, 37).length);
  });
});

// ===========================================================================
// the DECODE budget: what the server spends, not what it ships
// ===========================================================================

describe('effectiveStride (the decode budget)', () => {
  /** rows a lane costs at a stride, restated independently of decodedRowsAt */
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
        // Exact in the regime that matters (a stride past a chunk is where skipping
        // happens); an upper bound by at most one short tail chunk below it.
        if (stride > chunk) expect(`${label}: ${got}`).toBe(`${label}: ${want}`);
        else expect(`${label}: ${got >= want}`).toBe(`${label}: true`);
        expect(got).toBeLessThanOrEqual(steps);
      }
    }
  });

  it('charges the WHOLE lane when there is no chunk table to skip by', () => {
    // v3 lanes, and torn v4 lanes: no chunk geometry, so nothing can be skipped.
    expect(decodedRowsAt({ steps: 1_000_000 }, 4096)).toBe(1_000_000);
    expect(decodedRowsAt({ steps: 1_000_000, chunkSteps: 0 }, 4096)).toBe(1_000_000);
    // and a stride finer than a chunk never skips one either
    expect(decodedRowsAt({ steps: 10_000, chunkSteps: 1000 }, 64)).toBe(10_000);
  });

  it('coarsens a fully-recorded Tier-L until the DECODE fits, not just the payload', () => {
    // 1024 seeds × 1M steps in 1000-step chunks — the case the module doc names. The
    // payload budget alone would answer 283, which decodes 1.02e9 rows: ~92 s of CPU
    // holding one of two compute slots. The decode budget is what forbids that.
    const tierL = Array.from({ length: 1024 }, () => ({ steps: 1_000_000, chunkSteps: 1000 }));
    const payloadOnly = 283;
    const s = effectiveStride(tierL, WALL_DEFAULT_STRIDE);
    expect(s).toBeGreaterThan(payloadOnly);
    let rows = 0;
    for (const lane of tierL) rows += decodedRowsAt(lane, s);
    expect(rows).toBeLessThanOrEqual(WALL_MAX_DECODE_ROWS);
    // …and it is the FINEST stride that fits: one step finer must not.
    let finer = 0;
    for (const lane of tierL) finer += decodedRowsAt(lane, s - 1);
    expect(finer).toBeGreaterThan(WALL_MAX_DECODE_ROWS);
    // the payload budget still holds too
    let samples = 0;
    for (const lane of tierL) samples += Math.ceil(lane.steps / s) + 1;
    expect(samples * WALL_SAMPLE_BYTES).toBeLessThanOrEqual(WALL_MAX_RAW_BYTES);
  });

  it('stops at the stride ceiling and leaves the rest to the runtime cap', () => {
    // 8192 seeds × 10M: no stride inside the ceiling brings this inside the budget, so
    // the answer is the ceiling and each lane is cut by `laneCaps` instead.
    const huge = Array.from({ length: 8192 }, () => ({ steps: 10_000_000, chunkSteps: 1000 }));
    expect(effectiveStride(huge, 1)).toBe(WALL_MAX_STRIDE);
  });

  it('leaves every real fixture alone — the budgets are ceilings, not a policy', async () => {
    for (const [id, count, steps] of [[V4_ID, 64, 10_000], [DEMO_ID, 8, 100_000]] as const) {
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

// ===========================================================================
// synthetic lanes: non-finite tracking, sampling rule, both record versions
// ===========================================================================

describe('wall projection (synthetic bench root)', () => {
  it('dates the first non-finite ROW of a v3 lane and leaves a finite one null', async () => {
    clearWallCache();
    const { body } = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', 64);
    const { header, lanes } = await unpack(body);

    expect(header.envId).toBe('HalfCheetah-v4');
    expect(header.lanes.map((l) => l.seed)).toEqual([0, 1]);
    expect(header.chunkEvents).toBeUndefined(); // v3 lanes carry no chunk framing

    expect(lanes[0]?.firstNonFiniteRow).toBeNull();
    expect(lanes[1]?.firstNonFiniteRow).toBe(V3_NAN_ROW);

    // …and the NaN really is in the shipped samples, not just in the header.
    const nanSample = expectedRows(SYN_STEPS, 64).indexOf(512);
    expect(nanSample).toBeGreaterThanOrEqual(0);
    expect(Number.isNaN(lanes[1]?.pose[nanSample * 9 + 3] ?? 0)).toBe(true);
    expect(Number.isNaN(lanes[0]?.pose[nanSample * 9 + 3] ?? 0)).toBe(false);
  });

  it('samples row 0, every stride, and the final row — for every stride', async () => {
    for (const stride of [1, 3, 64, 500, 4096]) {
      clearWallCache();
      const { body } = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', stride);
      const { lanes } = await unpack(body);
      const rows = expectedRows(SYN_STEPS, stride);
      const lane = lanes[0];
      expect(lane?.samples).toBe(rows.length);
      // pose[0] of the synthetic lane is `row·0.01`, so a sample's own value names
      // the row it came from — the sampling rule is checked against the DATA.
      for (let s = 0; s < rows.length; s++) {
        expect(lane?.pose[s * 9]).toBeCloseTo((rows[s] ?? 0) * 0.01, 4);
      }
      expect(lane?.pose[(rows.length - 1) * 9]).toBeCloseTo((SYN_STEPS - 1) * 0.01, 4);
    }
  });

  it('never duplicates the final row when the stride already lands on it', async () => {
    // 1200 steps, stride 1199 -> rows 0 and 1199; 1199 IS the final row.
    clearWallCache();
    const { body } = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', 1199);
    const { lanes } = await unpack(body);
    expect(lanes[0]?.samples).toBe(2);
    expect(lanes[0]?.pose[9]).toBeCloseTo(1199 * 0.01, 4);
  });

  it('carries the done bit through, chunked v4 and flat v3 alike', async () => {
    clearWallCache();
    const v3 = await unpack((await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', 1)).body);
    // v3 lane sets done at i % 250 === 249.
    for (let r = 0; r < SYN_STEPS; r++) {
      expect(v3.lanes[0]?.done[r]).toBe(r % 250 === 249 ? 1 : 0);
    }
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
    // done at i % 300 === 299 -> rows 299 (chunk 0), 599 (chunk 1), 899 + 1199 (chunk 2)
    expect(events?.doneCount).toEqual([1, 1, 2]);
    // NaN on the wire is null, never a synthesized 0.
    expect(events?.rewardDelta).toEqual([1.5, null, null]);
  });

  it('matches a full parseActionsBinV4 decode at every sampled row', async () => {
    const dir = join(SYN_RUN, 'FourRooms-cont-v0', 'seed7');
    const meta = await readLaneMeta(join(dir, 'meta.json'));
    const raw = await parseActionsBinV4(await Bun.file(join(dir, 'actions.bin')).arrayBuffer(), meta);
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

  it('keeps a lane whose file is cut mid-row: whole rows survive, the fragment does not', async () => {
    const torn = join(TMP, 'torn');
    cpSync(SYN_BENCH, torn, { recursive: true });
    const runDir = join(torn, SYN_ID);
    const lanePath = join(runDir, 'HalfCheetah-v4', 'seed0', 'actions.bin');
    // 900 whole rows + 40 bytes of a 41st
    await truncate(lanePath, 900 * RECORD_BYTES + 40);
    clearWallCache();
    const { lanes } = await unpack((await deriveFor(torn, runDir, SYN_ID, 'HalfCheetah-v4', 64)).body);
    expect(lanes[0]?.steps).toBe(900);
    expect(lanes[0]?.samples).toBe(expectedRows(900, 64).length);
    expect(lanes[0]?.pose[(lanes[0].samples - 1) * 9]).toBeCloseTo(899 * 0.01, 4);
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

// ===========================================================================
// caching
// ===========================================================================

async function wallFiles(dir: string): Promise<string[]> {
  try {
    return (await readdir(dir)).filter((n) => n.endsWith('.wall')).sort();
  } catch {
    return [];
  }
}

/** The disk write is fire-and-forget, so wait for it rather than racing it. */
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
    const first = await deriveFor(root, runDir, SYN_ID, 'HalfCheetah-v4', 64, cacheDir);
    const names = await waitForWallFiles(cacheDir, 1);
    expect(names.length).toBe(1);

    // memory hit: the very same object, not merely an equal one
    const memHit = await deriveFor(root, runDir, SYN_ID, 'HalfCheetah-v4', 64, cacheDir);
    expect(memHit.body).toBe(first.body);

    // disk hit: LRU dropped, bytes must still come back identical
    clearWallCache();
    const diskHit = await deriveFor(root, runDir, SYN_ID, 'HalfCheetah-v4', 64, cacheDir);
    expect(diskHit.body).not.toBe(first.body);
    expect(Buffer.from(diskHit.body).equals(Buffer.from(first.body))).toBe(true);
    expect(await wallFiles(cacheDir)).toEqual(names); // no second entry written

    // an mtime bump alone moves the KEY (the file may have been rewritten identically)
    const lanePath = join(runDir, 'HalfCheetah-v4', 'seed0', 'actions.bin');
    const now = new Date(Date.now() + 5_000);
    await utimes(lanePath, now, now);
    clearWallCache();
    clearV4StepsCache();
    const afterTouch = await deriveFor(root, runDir, SYN_ID, 'HalfCheetah-v4', 64, cacheDir);
    expect(Buffer.from(afterTouch.body).equals(Buffer.from(first.body))).toBe(true);
    const names2 = await waitForWallFiles(cacheDir, 2);
    expect(names2.length).toBe(2);

    // a CONTENT change must change the answer, not just the key
    await truncate(lanePath, 300 * RECORD_BYTES);
    clearWallCache();
    clearV4StepsCache();
    const afterEdit = await deriveFor(root, runDir, SYN_ID, 'HalfCheetah-v4', 64, cacheDir);
    const { lanes } = await unpack(afterEdit.body);
    expect(lanes[0]?.steps).toBe(300);
    expect(Buffer.from(afterEdit.body).equals(Buffer.from(first.body))).toBe(false);
  });

  it('gives each stride its own entry — one stride never answers for another', async () => {
    const cacheDir = join(TMP, 'cache-stride');
    clearWallCache();
    const a = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', 64, cacheDir);
    const b = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', 16, cacheDir);
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
      envId: 'HalfCheetah-v4',
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
    // The lane meta.json is a DECODE INPUT — `recordVersion`, `actDim`, `codec`,
    // `codecVersion` all reach the decoders — so a key over actions.bin alone would keep
    // serving the pre-edit projection. Reproduced exactly: derive, cache to disk, rewrite
    // recordVersion 4 -> 3 leaving the payload byte-identical, drop the LRU, re-derive.
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
    await writeFile(metaPath, JSON.stringify({ ...meta, recordVersion: 3 }));
    const laneAfter = await stat(lanePath);
    expect(laneAfter.size).toBe(laneBefore.size); // the payload did not move
    expect(laneAfter.mtimeMs).toBe(laneBefore.mtimeMs);

    clearWallCache();
    clearV4StepsCache();
    const second = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    // A v4 file read as v3 cannot decode to the same wall — whatever it answers, it must
    // not be the CACHED one.
    expect(Buffer.from(second.body).equals(Buffer.from(first.body))).toBe(false);
    const names = await waitForWallFiles(cacheDir, 2);
    expect(names.length).toBe(2);

    // and putting the meta back brings the original wall back, byte for byte
    await writeFile(metaPath, JSON.stringify(meta));
    clearWallCache();
    clearV4StepsCache();
    const third = await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64, cacheDir);
    expect(Buffer.from(third.body).equals(Buffer.from(first.body))).toBe(true);
  });

  it('binds the row DECODER’s revision into the key, not just the projection’s', async () => {
    // A revision of `@sim/record4` changes what the same bytes mean while no lane file's
    // mtime or size moves, so `CODEC_VERSION` has to be in the key beside the wall's own
    // — otherwise every cached entry keeps serving the pre-fix projection. Read back off
    // the identity record the cache actually wrote.
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
    // `parseWallBody` re-checks headerBytes and binaryBytes, which catches truncation and
    // a wrong split — but a flipped bit inside the deflate section changes no length. The
    // record now carries the body's own crc32, so corruption is a miss (and a correct
    // re-derive) rather than a panel that fails in the browser's inflate, permanently.
    const root = join(TMP, 'corrupt-bench');
    cpSync(SYN_BENCH, root, { recursive: true });
    const runDir = join(root, SYN_ID);
    const cacheDir = join(TMP, 'corrupt-store');

    clearWallCache();
    const first = await deriveFor(root, runDir, SYN_ID, 'HalfCheetah-v4', 64, cacheDir);
    const names = await waitForWallFiles(cacheDir, 1);
    const cachePath = join(cacheDir, names[0] ?? '');
    const onDisk = new Uint8Array(await Bun.file(cachePath).arrayBuffer());
    expect(onDisk.byteLength).toBeGreaterThan(64);

    // one bit, eight bytes from the end — deep inside the deflate section
    const flipped = onDisk.slice();
    const at = flipped.byteLength - 8;
    flipped[at] = (flipped[at] ?? 0) ^ 0x01;
    await Bun.write(cachePath, flipped);
    expect((await stat(cachePath)).size).toBe(onDisk.byteLength); // same length, so only a checksum can tell

    clearWallCache();
    const second = await deriveFor(root, runDir, SYN_ID, 'HalfCheetah-v4', 64, cacheDir);
    // the corrupted bytes were NOT served…
    expect(Buffer.from(second.body).equals(Buffer.from(flipped.subarray(flipped.indexOf(0x0a) + 1)))).toBe(false);
    // …and what came back is the real wall, which still inflates
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
      envId: 'HalfCheetah-v4',
      stride: 64,
      recordings,
      inFlight: true,
      cacheDir,
    });
    await Bun.sleep(60);
    expect(await wallFiles(cacheDir)).toEqual([]);
  });
});

// ===========================================================================
// selective decode: a strided projection must be a strided DECODE
// ===========================================================================

describe('selective chunk decoding', () => {
  it('skips whole chunks the stride never samples, and says so in scannedRows', async () => {
    // The synthetic v4 lane is 1200 steps in 3 chunks of 400. At stride 1000 the
    // sampled rows are 0, 1000 and the final 1199 — chunk 1 ([400, 800)) holds none of
    // them, so it is never crc'd, never inflated and never decoded.
    clearWallCache();
    const { header, lanes } = await unpack(
      (await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 1000)).body,
    );
    const head = header.lanes[0];
    expect(head?.steps).toBe(SYN_STEPS);
    expect(head?.scannedRows).toBe(800); // chunks 0 and 2 only
    expect(lanes[0]?.samples).toBe(expectedRows(SYN_STEPS, 1000).length);

    // …and what it shipped is still exactly what a full decode says.
    const dir = join(SYN_RUN, 'FourRooms-cont-v0', 'seed7');
    const meta = await readLaneMeta(join(dir, 'meta.json'));
    const raw = await parseActionsBinV4(await Bun.file(join(dir, 'actions.bin')).arrayBuffer(), meta);
    const lane = lanes[0];
    if (lane === undefined) throw new Error('no lane');
    expectLaneMatches(lane, raw, 1000, 'skipped-chunk lane');

    // Chunk events come from the chunk HEADERS, so a skipped chunk still reports itself.
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
    // V4_NAN_ROW (137) is in chunk 0, which every stride decodes (row 0 is always
    // sampled), so the certificate is exact even when later chunks are skipped.
    clearWallCache();
    const { header } = await unpack(
      (await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'FourRooms-cont-v0', 1000)).body,
    );
    expect(header.lanes[0]?.firstNonFiniteRow).toBe(V4_NAN_ROW);
  });

  it('reports null — never a guess — for a death inside a SKIPPED chunk, and flags the scan', async () => {
    // The one thing the skip costs, made explicit. A lane whose only non-finite row is
    // in a chunk no sample falls in comes back `firstNonFiniteRow: null` WITH
    // `scannedRows < steps`, which is the client's signal that the answer is
    // chunk-resolution, not row-resolution. The alternative — a full scan of every
    // chunk — is the ~180 s Tier-L derive this whole path exists to avoid.
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
    // row 500 — inside chunk 1 ([400, 800)), which stride 1000 does not sample
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

    // the same lane at a stride that decodes everything finds it exactly
    clearWallCache();
    const fine = await unpack((await deriveFor(root, runDir, SYN_ID, 'DeepSea-cont-v0', 64)).body);
    expect(fine.header.lanes[0]?.scannedRows).toBeUndefined();
    expect(fine.header.lanes[0]?.firstNonFiniteRow).toBe(500);
  });

  it('costs strictly less work at a coarser stride — the point of the whole exercise', async () => {
    if (!v4Present) return;
    const runDir = join(REAL_BENCH, V4_ID);
    // 64 lanes × 10k steps in 1000-step chunks.
    clearWallCache();
    const fine = await deriveFor(REAL_BENCH, runDir, V4_ID, 'HalfCheetah-v4', 64);
    clearWallCache();
    const coarse = await deriveFor(REAL_BENCH, runDir, V4_ID, 'HalfCheetah-v4', 4096);

    const scanned = (p: WallPayload): number =>
      p.header.lanes.reduce((a, l) => a + (l.scannedRows ?? l.steps), 0);
    expect(scanned(fine)).toBe(64 * 10_000); // stride < chunk: every row
    expect(scanned(coarse)).toBe(64 * 4000); // rows 0, 4096, 8192 + the final row's chunk
    expect(scanned(coarse)).toBeLessThan(scanned(fine));

    // and the coarse wall is still exact where it sampled
    const { header, lanes } = await unpack(coarse.body);
    const head = header.lanes[0];
    const lane = lanes[0];
    if (head === undefined || lane === undefined) throw new Error('no lane');
    const dir = join(runDir, 'HalfCheetah-v4', `seed${head.seed}`);
    const meta = await readLaneMeta(join(dir, 'meta.json'));
    const raw = await parseActionsBinV4(await Bun.file(join(dir, 'actions.bin')).arrayBuffer(), meta);
    expectLaneMatches(lane, raw, 4096, `skipped-chunk seed ${head.seed}`);
  });

  it('crc-checks every chunk it decodes, and stops at a corrupted one', async () => {
    // The framing scan runs with `verifyCrc: false` (it would crc chunks this lane is
    // about to skip), so the integrity check moves to the point of use. A chunk whose
    // payload was corrupted must be treated as the torn tail it looks like: the lane
    // ends before it, and nothing after it is served.
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
    const at = third[2] + CHUNK_HEADER_BYTES + 3; // inside chunk 2's compressed payload
    bytes[at] = (bytes[at] ?? 0) ^ 0x55;
    await Bun.write(lanePath, bytes);

    clearWallCache();
    clearV4StepsCache();
    const { header, lanes } = await unpack(
      (await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64)).body,
    );
    const head = header.lanes[0];
    expect(head?.steps).toBe(800); // chunks 0 and 1 only
    expect(head?.error).toBeUndefined(); // a torn tail is DATA, not a lane failure
    expect(head?.samples).toBe(expectedRows(800, 64).length);
    expect(header.chunkEvents?.[0]?.stepStart).toEqual([0, 400]);
    expect(lanes[0]?.pose[(lanes[0].samples - 1) * 9]).toBe(799 % 11);
  });

  it('keeps @sim/record4’s own lane checks — a meta that contradicts the file is an error', async () => {
    // The selective reader restates `prepareV4`'s validations rather than inheriting
    // them, so they are tested where they now live: a lane whose meta names the wrong
    // env, the wrong codec revision or the wrong record version must fail the lane
    // (with `error`), never decode as something else.
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
      clearV4StepsCache();
      const { header } = await unpack((await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64)).body);
      const head = header.lanes[0];
      expect(`${label}: ${head?.error === undefined ? 'decoded' : 'error'}`).toBe(`${label}: error`);
      expect(head?.samples).toBe(0);
    }
    // …and the same lane with its real meta back decodes as it always did
    await writeFile(metaPath, JSON.stringify(good));
    clearWallCache();
    clearV4StepsCache();
    const { header } = await unpack((await deriveFor(root, runDir, SYN_ID, 'FourRooms-cont-v0', 64)).body);
    expect(header.lanes[0]?.error).toBeUndefined();
    expect(header.lanes[0]?.steps).toBe(SYN_STEPS);
  });
});

// ===========================================================================
// the ceilings, made real
// ===========================================================================

describe('budget ceilings', () => {
  /** Copy the synthetic root, then grow one lane BEHIND the discovery it was scanned with. */
  async function staleRoot(name: string): Promise<string> {
    const root = join(TMP, name);
    cpSync(SYN_BENCH, root, { recursive: true });
    return root;
  }

  it('cuts a v3 lane that grew behind a stale index, and says it cut it', async () => {
    const root = await staleRoot('stale-v3');
    const runDir = join(root, SYN_ID);
    const lanePath = join(runDir, 'HalfCheetah-v4', 'seed0', 'actions.bin');
    // discovery FIRST: this is the stale count the budget will be planned from
    const recordings = await scanRecordings(runDir, { root });
    expect(recordings.find((r) => r.seed === 0 && r.envId === 'HalfCheetah-v4')?.steps).toBe(SYN_STEPS);

    // now the lane grows to 5000 rows — 4× what the index says
    const grown = v3Bytes(5000, (row, i) => {
      for (let p = 0; p < 9; p++) row[7 + p] = i * 0.01 + p;
    });
    await Bun.write(lanePath, grown);

    clearWallCache();
    const { header, lanes } = await unpack(
      (await deriveWall({
        id: SYN_ID, dir: runDir, envId: 'HalfCheetah-v4', stride: 64,
        recordings, inFlight: false, cacheDir: null,
      })).body,
    );
    const head = header.lanes[0];
    // WALL_LANE_OVERSHOOT × the plan, and not a row more
    expect(head?.steps).toBe(SYN_STEPS * WALL_LANE_OVERSHOOT);
    expect(head?.truncated).toBe(true);
    expect(head?.samples).toBe(expectedRows(SYN_STEPS * WALL_LANE_OVERSHOOT, 64).length);
    // the samples it DID ship are real rows, and the last one is row `steps − 1`
    expect(lanes[0]?.pose[(lanes[0].samples - 1) * 9]).toBeCloseTo((SYN_STEPS * 2 - 1) * 0.01, 4);
    // the lane that did not grow is untouched — a cap is per lane, not a mode
    expect(header.lanes[1]?.truncated).toBeUndefined();
    expect(header.lanes[1]?.steps).toBe(SYN_STEPS);
  });

  it('cuts a v4 lane at a CHUNK boundary, so the wall never ends mid-chunk', async () => {
    const root = join(TMP, 'stale-v4');
    const runDir = join(root, SYN_ID);
    const dir = join(runDir, 'FourRooms-cont-v0', 'seed7');
    await mkdir(dir, { recursive: true });
    await writeFile(join(runDir, 'meta.json'), await Bun.file(join(SYN_RUN, 'meta.json')).text());
    await writeFile(join(runDir, 'fleet.jsonl'), '');

    const steps = 3600; // 9 chunks of 400
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
    // …but the meta still describes the lane as it was at 1200 steps (3 chunks): the
    // in-flight case, where discovery predates the rows on disk.
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
    expect(head?.steps).toBe(SYN_STEPS * WALL_LANE_OVERSHOOT); // 2400 = 6 whole chunks
    expect((head?.steps ?? 0) % 400).toBe(0);
    expect(head?.samples).toBe(expectedRows(SYN_STEPS * WALL_LANE_OVERSHOOT, 64).length);
    // events end where the decode did — they never describe rows the wall does not carry
    expect(header.chunkEvents?.[0]?.stepStart).toEqual([0, 400, 800, 1200, 1600, 2000]);
    // and every shipped sample is still bit-equal to a full decode of the same file
    const raw = await parseActionsBinV4(
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
    // Four distinct walls of the synthetic root (one per stride) — small, so all four
    // fit and the byte count is the sum of their bodies.
    const bodies: number[] = [];
    for (const stride of [16, 32, 64, 128]) {
      const w = await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', stride);
      bodies.push(w.body.byteLength);
    }
    expect(wallCacheBytes()).toBe(bodies.reduce((a, b) => a + b, 0));
    // a fifth evicts the oldest by COUNT (the cap this cache already had)
    await deriveFor(SYN_BENCH, SYN_RUN, SYN_ID, 'HalfCheetah-v4', 256);
    expect(wallCacheBytes()).toBeLessThan(bodies.reduce((a, b) => a + b, 0) + 4096);
    expect(wallCacheBytes()).toBeGreaterThan(0);
    clearWallCache();
    expect(wallCacheBytes()).toBe(0);
  });
});

// ===========================================================================
// real v4 root — the all-lanes shape
// ===========================================================================

describe.if(v4Present)(`wall over ${V4_ID} (v4, 64 lanes)`, () => {
  const runDir = join(REAL_BENCH, V4_ID);

  it('ships every recorded seed of the env, in ascending seed order', async () => {
    clearWallCache();
    const { header } = await unpack((await deriveFor(REAL_BENCH, runDir, V4_ID, 'HalfCheetah-v4', 64)).body);
    expect(header.lanes.length).toBe(64);
    expect(header.lanes.map((l) => l.seed)).toEqual(Array.from({ length: 64 }, (_, i) => i));
    for (const lane of header.lanes) {
      expect(lane.steps).toBe(10_000);
      expect(lane.samples).toBe(expectedRows(10_000, 64).length);
      expect(lane.error).toBeUndefined();
    }
    expect(header.chunkEvents?.length).toBe(64);
  });

  for (const envId of ['HalfCheetah-v4', 'DeepSea-cont-v0', 'Pendulum-cont-v0'] as const) {
    it(`${envId}: every sampled row is bit-equal to a direct parseActionsBinV4 decode`, async () => {
      clearWallCache();
      const stride = 64;
      const { header, lanes } = await unpack(
        (await deriveFor(REAL_BENCH, runDir, V4_ID, envId, stride)).body,
      );
      expect(lanes.length).toBeGreaterThan(0);

      // Three lanes decoded in full: the first, the last, and one in the middle.
      const probe = [0, lanes.length >> 1, lanes.length - 1];
      for (const idx of probe) {
        const lane = lanes[idx];
        const head = header.lanes[idx];
        if (lane === undefined || head === undefined) throw new Error(`no lane ${idx}`);
        const dir = join(runDir, envId, `seed${head.seed}`);
        const meta = await readLaneMeta(join(dir, 'meta.json'));
        const raw = await parseActionsBinV4(
          await Bun.file(join(dir, 'actions.bin')).arrayBuffer(),
          meta,
        );
        expectLaneMatches(lane, raw, stride, `${envId} seed ${head.seed}`);
        expect(lane.firstNonFiniteRow).toBe(referenceFirstNonFinite(raw.data, raw.steps));
      }
    });
  }

  it('chunk events restate the file’s own framing (contiguous, and the real done counts)', async () => {
    clearWallCache();
    const { header } = await unpack((await deriveFor(REAL_BENCH, runDir, V4_ID, 'DeepSea-cont-v0', 64)).body);
    const events = header.chunkEvents?.[0];
    const seed = header.lanes[0]?.seed ?? -1;
    expect(events?.seed).toBe(seed);
    expect(events).toBeDefined();
    if (events === undefined) return;

    let at = 0;
    for (let i = 0; i < events.stepStart.length; i++) {
      expect(events.stepStart[i]).toBe(at);
      at += events.stepCount[i] ?? 0;
    }
    expect(at).toBe(header.lanes[0]?.steps ?? -1);

    // doneCount is the recorder's own tally: recount it from a full decode.
    const dir = join(runDir, 'DeepSea-cont-v0', `seed${seed}`);
    const meta = await readLaneMeta(join(dir, 'meta.json'));
    const raw = await parseActionsBinV4(await Bun.file(join(dir, 'actions.bin')).arrayBuffer(), meta);
    let row = 0;
    for (let i = 0; i < events.stepCount.length; i++) {
      let count = 0;
      const n = events.stepCount[i] ?? 0;
      for (let k = 0; k < n; k++) if (raw.data[(row + k) * RECORD_FLOATS + 6] === 1) count++;
      expect(events.doneCount[i]).toBe(count);
      row += n;
    }
  });

  it('a coarser stride ships strictly fewer bytes and still ends on the final row', async () => {
    clearWallCache();
    const fine = await deriveFor(REAL_BENCH, runDir, V4_ID, 'HalfCheetah-v4', 16);
    clearWallCache();
    const coarse = await deriveFor(REAL_BENCH, runDir, V4_ID, 'HalfCheetah-v4', 256);
    expect(coarse.body.byteLength).toBeLessThan(fine.body.byteLength);

    const a = await unpack(fine.body);
    const b = await unpack(coarse.body);
    for (let i = 0; i < a.lanes.length; i++) {
      const la = a.lanes[i];
      const lb = b.lanes[i];
      if (la === undefined || lb === undefined) continue;
      // the last sample of both is the SAME record row, so it is the same 9 floats
      for (let j = 0; j < 9; j++) {
        expect(Object.is(la.pose[(la.samples - 1) * 9 + j], lb.pose[(lb.samples - 1) * 9 + j])).toBe(true);
      }
    }
  });
});

// ===========================================================================
// real v3 root — the wall must work on pre-v4 roots
// ===========================================================================

describe.if(demoPresent)(`wall over ${DEMO_ID} (v3)`, () => {
  const runDir = join(REAL_BENCH, DEMO_ID);

  for (const envId of ['FourRooms-cont-v0', 'MountainCar-cont-v0'] as const) {
    it(`${envId}: every sampled row is bit-equal to a direct parseActionsBin decode`, async () => {
      clearWallCache();
      const stride = 64;
      const { header, lanes } = await unpack(
        (await deriveFor(REAL_BENCH, runDir, DEMO_ID, envId, stride)).body,
      );
      expect(lanes.length).toBe(8);
      expect(header.chunkEvents).toBeUndefined(); // v3 has no chunk framing

      for (const idx of [0, lanes.length - 1]) {
        const lane = lanes[idx];
        const head = header.lanes[idx];
        if (lane === undefined || head === undefined) throw new Error(`no lane ${idx}`);
        const dir = join(runDir, envId, `seed${head.seed}`);
        const meta = await readLaneMeta(join(dir, 'meta.json'));
        const raw = parseActionsBin(
          new Uint8Array(await Bun.file(join(dir, 'actions.bin')).arrayBuffer()),
          meta,
        );
        expect(raw.steps).toBe((await stat(join(dir, 'actions.bin'))).size / RECORD_BYTES);
        expectLaneMatches(lane, raw, stride, `${envId} seed ${head.seed}`);
        expect(lane.firstNonFiniteRow).toBe(referenceFirstNonFinite(raw.data, raw.steps));
      }
    });
  }
});

// ===========================================================================
// HTTP: routing, framing round-trip through the real client helper
// ===========================================================================

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
    // `src/api.ts` builds RELATIVE urls (it runs in a page); resolve them against the
    // ephemeral server so the real client helper can be exercised unmodified.
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
      // `rows` names the record row each sample was taken at
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
      [`/api/runs/${SYN_ID}/wall/HalfCheetah`, 404], // near-miss env id
      [`/api/runs/${SYN_ID}/wall/%2e%2e%2f%2e%2e%2fetc%2fpasswd`, 400],
      [`/api/runs/${SYN_ID}/wall/..%2f..%2fetc%2fpasswd`, 400],
      [`/api/runs/${SYN_ID}/wall/HalfCheetah-v4%00`, 400],
      [`/api/runs/${SYN_ID}/wall`, 404], // no env segment
      [`/api/runs/${SYN_ID}/wall/HalfCheetah-v4/extra`, 404],
    ];
    for (const [path, status] of cases) {
      const res = await realFetch(`${base}${path}`);
      expect(`${path} -> ${res.status}`).toBe(`${path} -> ${status}`);
      // nothing that failed may ever answer with lane bytes
      expect(res.headers.get('content-type')).toContain('application/json');
    }
  });

  it('cannot be addressed with a bare dot segment: the URL parser collapses it first', async () => {
    // `wall/..` and its percent-encoded twin are removed by WHATWG dot-segment
    // normalization before the request is issued, so the router never sees them as an
    // env id — they address `/api/runs/:id`, an ordinary endpoint, not a path.
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

// ===========================================================================
// the degraded wall: a real run root with no recordings
// ===========================================================================

describe.if(norecPresent)(`wall over ${NOREC_ID} (no recordings)`, () => {
  it('answers a valid empty wall rather than a 404', async () => {
    clearWallCache();
    const runDir = join(REAL_BENCH, NOREC_ID);
    const { header, lanes } = await unpack(
      (await deriveFor(REAL_BENCH, runDir, NOREC_ID, 'HalfCheetah-v4', 64)).body,
    );
    expect(header.runId).toBe(NOREC_ID);
    expect(header.lanes).toEqual([]);
    expect(header.binaryBytes).toBe(0);
    expect(lanes).toEqual([]);
  });
});

// ===========================================================================
// the tiny real root — every one of the 16 envs walks
// ===========================================================================

describe.if(v4XsPresent)(`wall over ${V4_XS_ID} (all 16 envs)`, () => {
  it('decodes a lane of every env with no lane reporting an error', async () => {
    const runDir = join(REAL_BENCH, V4_XS_ID);
    const recordings = await scanRecordings(runDir, { root: REAL_BENCH });
    const envIds = [...new Set(recordings.map((r) => r.envId))];
    expect(envIds.length).toBe(16);
    for (const envId of envIds) {
      clearWallCache();
      const { header, lanes } = await unpack(
        (await deriveWall({
          id: V4_XS_ID,
          dir: runDir,
          envId,
          stride: 64,
          recordings,
          inFlight: false,
          cacheDir: null,
        })).body,
      );
      expect(`${envId}: ${header.lanes.length} lanes`).toBe(`${envId}: 1 lanes`);
      const lane = lanes[0];
      const head = header.lanes[0];
      if (lane === undefined || head === undefined) throw new Error(`${envId} produced no lane`);
      expect(head.error).toBeUndefined();
      expect(head.steps).toBeGreaterThan(0);
      const dir = join(runDir, envId, `seed${head.seed}`);
      const meta = await readLaneMeta(join(dir, 'meta.json'));
      const raw = await parseActionsBinV4(
        await Bun.file(join(dir, 'actions.bin')).arrayBuffer(),
        meta,
      );
      expectLaneMatches(lane, raw, 64, `${envId} seed ${head.seed}`);
    }
  });
});
