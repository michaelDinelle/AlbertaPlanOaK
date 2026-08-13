/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { ENVS, ENV_IDS, type EnvId } from '@shared/envs';
import type { RecordingMeta } from '@shared/types';
import { decodeRecording, eventTimeline } from '@sim/decode';
import {
  CHUNK_HEADER_BYTES,
  CODEC_VERSION,
  ESCAPE_ENTRY_BYTES,
  FILE_HEADER_BYTES,
  RECORD_FLOATS,
  RECORD_V4_CODECS,
  RECORD_V4_ENV_KINDS,
  RECORD_VERSION,
  RecordParseError,
  chunkTableFromMeta,
  codecForEnv,
  codecMatches,
  crc32,
  dequantizeCtrlU8,
  envKindOf,
  parseActionsBin,
  planPrefixV4,
  quantizeCtrlU8,
  rangeHeaderForSteps,
  readFileHeaderV4,
  roundHalfAwayFromZero,
  scanChunksV4,
  walkChunksV4,
  wrapAngle,
  type ChunkRowsV4,
  type CodecDescriptor,
  type PoseFieldSpec,
} from '@sim/record';
import { decoders } from '@sim/registry';
import type { EnvDecoder, RawRecording } from '@sim/types';
import { parseRawTrace } from './helpers/rawTrace';
import { loadHarnessSource } from '@kit/harness';
import { TheaterLoadError, loadTheaterRecording } from '../src/theater/loader';


const I16_MAX = 32767;
const I32_MAX = 2147483647;
const U8_MAX = 255;

const at = (data: Float32Array, i: number): number => data[i] ?? 0;

function clampInt(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return 0;
  return v < lo ? lo : v > hi ? hi : v;
}

interface EncodeOptions {
  envId: EnvId;
  seed?: number;
  data: Float32Array;
  steps: number;
  chunkSteps?: number;
  omitChunkTable?: boolean;
}

export interface EncodedV4 {
  bytes: Uint8Array;
  meta: RecordingMeta;
  chunks: [number, number, number, number][];
}

function buildChunkPayload(
  desc: CodecDescriptor,
  data: Float32Array,
  stepStart: number,
  stepCount: number,
): { payload: Uint8Array; doneCount: number } {
  const ctrlCount = desc.ctrl === 'index' ? 0 : desc.ctrl.n;
  const indexCtrl = desc.ctrl === 'index';

  const hdr = new Uint8Array(stepCount);
  const ctrlCols = new Uint8Array(ctrlCount * stepCount);
  const poseCols = desc.poseFields.map((f: PoseFieldSpec) =>
    f.enc === 'u8'
      ? new Uint8Array(stepCount)
      : f.enc === 'i16'
        ? new Int16Array(stepCount)
        : new Int32Array(stepCount),
  );
  const escapes: number[] = [];

  const chunkBaseX = desc.category === 'cheetah' ? at(data, stepStart * RECORD_FLOATS + 7) : 0;
  let recon = chunkBaseX;
  let doneCount = 0;

  for (let i = 0; i < stepCount; i++) {
    const row = (stepStart + i) * RECORD_FLOATS;
    let escape = false;
    for (let j = 0; j < RECORD_FLOATS; j++) {
      if (!Number.isFinite(at(data, row + j))) escape = true;
    }

    let action = 0;
    if (indexCtrl) {
      const a = at(data, row);
      if (Number.isInteger(a) && a >= 0 && a <= 15) action = a;
      else escape = true;
    }

    for (let c = 0; c < ctrlCount; c++) {
      const v = at(data, row + c);
      ctrlCols[c * stepCount + i] = Number.isFinite(v) ? quantizeCtrlU8(v) : 0;
    }

    for (let k = 0; k < desc.poseFields.length; k++) {
      const field = desc.poseFields[k] as PoseFieldSpec;
      const col = poseCols[k];
      if (col === undefined) continue;
      const v = at(data, row + 7 + field.idx);
      if (field.enc === 'u8') {
        if (!(Number.isInteger(v) && v >= 0 && v <= U8_MAX)) escape = true;
        col[i] = Number.isFinite(v) ? clampInt(roundHalfAwayFromZero(v), 0, U8_MAX) : 0;
      } else if (field.enc === 'i16') {
        const scale = field.scale ?? 1;
        const w = field.wrap === true && Number.isFinite(v) ? wrapAngle(v) : v;
        const q = Number.isFinite(w) ? roundHalfAwayFromZero(w / scale) : 0;
        if (!(q >= -I16_MAX && q <= I16_MAX)) escape = true;
        col[i] = clampInt(q, -I16_MAX, I16_MAX);
      } else {
        const scale = field.scale ?? 1;
        let q = 0;
        if (Number.isFinite(v) && Number.isFinite(recon)) {
          q = roundHalfAwayFromZero((v - recon) / scale);
          if (!(q >= -I32_MAX && q <= I32_MAX)) escape = true;
          q = clampInt(q, -I32_MAX, I32_MAX);
        } else {
          escape = true;
        }
        col[i] = q;
        recon += q * scale;
      }
    }

    const doneV = at(data, row + 6);
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

  let size = stepCount + ctrlCount * stepCount;
  if (desc.category === 'cheetah') size += 8;
  for (const field of desc.poseFields) {
    size += (field.enc === 'u8' ? 1 : field.enc === 'i16' ? 2 : 4) * stepCount;
  }
  size += 4 + escapes.length * ESCAPE_ENTRY_BYTES;

  const payload = new Uint8Array(size);
  const dv = new DataView(payload.buffer);
  let off = 0;
  payload.set(hdr, off);
  off += stepCount;
  payload.set(ctrlCols, off);
  off += ctrlCols.length;
  if (desc.category === 'cheetah') {
    dv.setFloat64(off, chunkBaseX, true);
    off += 8;
  }
  for (let k = 0; k < desc.poseFields.length; k++) {
    const field = desc.poseFields[k] as PoseFieldSpec;
    const col = poseCols[k];
    if (col === undefined) continue;
    if (field.enc === 'u8') {
      payload.set(col as Uint8Array, off);
      off += stepCount;
    } else if (field.enc === 'i16') {
      for (let i = 0; i < stepCount; i++) dv.setInt16(off + i * 2, col[i] ?? 0, true);
      off += stepCount * 2;
    } else {
      for (let i = 0; i < stepCount; i++) dv.setInt32(off + i * 4, col[i] ?? 0, true);
      off += stepCount * 4;
    }
  }
  dv.setUint32(off, escapes.length, true);
  off += 4;
  const rowBytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  for (const rowIdx of escapes) {
    dv.setUint32(off, rowIdx, true);
    payload.set(
      rowBytes.subarray(
        (stepStart + rowIdx) * RECORD_FLOATS * 4,
        (stepStart + rowIdx + 1) * RECORD_FLOATS * 4,
      ),
      off + 4,
    );
    off += ESCAPE_ENTRY_BYTES;
  }

  return { payload, doneCount };
}

async function deflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
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

interface RawChunkV4 {
  payload: Uint8Array;
  stepStart: number;
  stepCount: number;
  doneCount: number;
}

async function assembleV4(
  envKind: number,
  raw: readonly RawChunkV4[],
): Promise<{ bytes: Uint8Array; chunks: [number, number, number, number][] }> {
  const parts: Uint8Array[] = [];
  const header = new Uint8Array(FILE_HEADER_BYTES);
  const hv = new DataView(header.buffer);
  hv.setUint32(0, 0x344b414f, true);
  hv.setUint32(4, RECORD_VERSION, true);
  hv.setUint32(8, envKind, true);
  hv.setUint32(12, CODEC_VERSION, true);
  parts.push(header);

  const chunks: [number, number, number, number][] = [];
  let offset = FILE_HEADER_BYTES;
  for (const chunk of raw) {
    const comp = await deflateRaw(chunk.payload);
    const chunkHeader = new Uint8Array(CHUNK_HEADER_BYTES);
    const cv = new DataView(chunkHeader.buffer);
    cv.setUint32(0, 0x344b4843, true);
    cv.setUint32(4, chunk.stepStart, true);
    cv.setUint32(8, chunk.stepCount, true);
    cv.setUint32(12, chunk.payload.byteLength, true);
    cv.setUint32(16, comp.byteLength, true);
    cv.setUint32(20, crc32(comp), true);
    cv.setUint32(24, chunk.doneCount, true);
    cv.setFloat32(28, Number.NaN, true);
    parts.push(chunkHeader, comp);
    chunks.push([chunk.stepStart, chunk.stepCount, offset, CHUNK_HEADER_BYTES + comp.byteLength]);
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

async function encodeV4(opts: EncodeOptions): Promise<EncodedV4> {
  const desc = codecForEnv(opts.envId);
  const seed = opts.seed ?? 0;
  const chunkSteps = Math.max(1, opts.chunkSteps ?? Math.max(1, opts.steps));

  const raw: RawChunkV4[] = [];
  for (let start = 0; start < opts.steps; start += chunkSteps) {
    const count = Math.min(chunkSteps, opts.steps - start);
    const { payload, doneCount } = buildChunkPayload(desc, opts.data, start, count);
    raw.push({ payload, stepStart: start, stepCount: count, doneCount });
  }
  const { bytes, chunks } = await assembleV4(envKindOf(opts.envId), raw);

  const meta: RecordingMeta = {
    envId: opts.envId,
    seed,
    actDim: ENVS[opts.envId].action.n,
    codecVersion: CODEC_VERSION,
    codec: JSON.parse(JSON.stringify(desc)) as unknown,
  };
  if (opts.omitChunkTable !== true) {
    meta.steps = opts.steps;
    meta.chunks = chunks;
  }
  return { bytes, meta, chunks };
}


function toleranceFor(desc: CodecDescriptor): { tol: Float64Array; angular: boolean[] } {
  const tol = new Float64Array(RECORD_FLOATS);
  const angular = new Array<boolean>(RECORD_FLOATS).fill(false);
  if (desc.ctrl !== 'index') {
    for (let c = 0; c < desc.ctrl.n; c++) tol[c] = 1 / 255 + 1e-7;
  }
  for (const field of desc.poseFields) {
    if (field.enc === 'u8') continue;
    const scale = field.scale ?? 1;
    tol[7 + field.idx] = scale / 2;
    if (field.wrap === true) angular[7 + field.idx] = true;
  }
  return { tol, angular };
}

const MAX_BUDGET_FRACTION = 1.01;

function worstColumnError(
  envId: EnvId,
  want: Float32Array,
  got: Float32Array,
  steps: number,
): number {
  const desc = codecForEnv(envId);
  const { tol, angular } = toleranceFor(desc);
  let worst = 0;
  for (let i = 0; i < steps; i++) {
    const base = i * RECORD_FLOATS;
    for (let c = 0; c < RECORD_FLOATS; c++) {
      const a = at(want, base + c);
      const b = at(got, base + c);
      const budget = tol[c] ?? 0;
      const limit = budget + Math.max(1e-9, Math.abs(a) * 1.2e-7);
      const delta = angular[c] === true ? Math.abs(wrapAngle(b - a)) : Math.abs(b - a);
      if (!(delta <= limit)) {
        throw new Error(`${envId} row ${i} col ${c}: ${b} vs ${a} (|Δ| ${delta} > ${limit})`);
      }
      if (budget > 0) worst = Math.max(worst, delta / budget);
    }
  }
  return worst;
}

function expectRowsIdentical(want: Float32Array, got: Float32Array, steps: number, label: string): void {
  expect(got.length).toBe(steps * RECORD_FLOATS);
  for (let i = 0; i < steps * RECORD_FLOATS; i++) {
    const a = at(want, i);
    const b = at(got, i);
    if (!Object.is(a, b)) {
      throw new Error(`${label}: float ${i} (row ${Math.floor(i / 16)} col ${i % 16}) ${b} vs ${a}`);
    }
  }
}


interface TraceCase {
  name: string;
  envId: EnvId;
  seed: number;
  raw: RawRecording;
}

function fixtureUrl(name: string): URL {
  return new URL(`./fixtures/traces/${name}`, import.meta.url);
}

function loadTrace(name: string): TraceCase {
  const envId = name.replace(/\.events$/, '') as EnvId;
  const doc = JSON.parse(readFileSync(fixtureUrl(`${name}.json`), 'utf8')) as { seed: number };
  const buf = readFileSync(fixtureUrl(`${name}.bin`));
  const raw = parseRawTrace(buf, { envId, seed: doc.seed });
  return { name, envId, seed: doc.seed, raw };
}

const TRACE_NAMES: string[] = [
  ...ENV_IDS,
  'Acrobot-cont-v0.events',
  'MountainCar-cont-v0.events',
];

const TRACES_WITH_EVENTS = new Set([
  'DeepSea-cont-v0', 'KeyCorridor-cont-v0', 'CartPole-cont-v0', 'Chain-cont-v0',
  'LargeDeepSea-cont-v0', 'MultiKeyCorridor-cont-v0',
  'Acrobot-cont-v0.events', 'MountainCar-cont-v0.events',
]);


describe('the comparators themselves', () => {
  test('bit-identity fails on a one-ulp difference', () => {
    const a = new Float32Array(RECORD_FLOATS);
    const b = new Float32Array(RECORD_FLOATS);
    a[7] = 0.1;
    b[7] = Math.fround(0.1 + 1e-8);
    expect(() => expectRowsIdentical(a, b, 1, 'self-test')).toThrow(/float 7/);
    expect(() => expectRowsIdentical(a, a, 1, 'self-test')).not.toThrow();
    a[3] = Number.NaN;
    b.set(a);
    expect(() => expectRowsIdentical(a, b, 1, 'self-test')).not.toThrow();
    b[3] = 0;
    expect(() => expectRowsIdentical(a, b, 1, 'self-test')).toThrow(/float 3/);
  });

  test('the tolerance comparator fails past half a quantum and is angular where the spec says', () => {
    const a = new Float32Array(RECORD_FLOATS);
    const b = new Float32Array(RECORD_FLOATS);
    a[8] = 1;
    b[8] = 1 + 4e-4;
    expect(worstColumnError('CartPole-cont-v0', a, b, 1)).toBeGreaterThan(0.5);
    b[8] = 1 + 2e-3;
    expect(() => worstColumnError('CartPole-cont-v0', a, b, 1)).toThrow(/col 8/);
    const c = new Float32Array(RECORD_FLOATS);
    const d = new Float32Array(RECORD_FLOATS);
    c[9] = Math.PI - 1e-6;
    d[9] = -Math.PI - 1e-6;
    expect(worstColumnError('CartPole-cont-v0', c, d, 1)).toBeLessThan(1);
    d[9] = Math.PI - 1e-6 + 1e-3;
    expect(() => worstColumnError('CartPole-cont-v0', c, d, 1)).toThrow(/col 9/);
    const e = new Float32Array(RECORD_FLOATS);
    const f = new Float32Array(RECORD_FLOATS);
    f[6] = 1;
    expect(() => worstColumnError('CartPole-cont-v0', e, f, 1)).toThrow(/col 6/);
  });
});

describe('the frozen codec table', () => {
  test('envKind ordering matches the shared env table', () => {
    expect(RECORD_V4_ENV_KINDS.length).toBe(ENV_IDS.length);
    expect([...RECORD_V4_ENV_KINDS]).toEqual([...ENV_IDS]);
    expect(RECORD_V4_CODECS.length).toBe(ENV_IDS.length);
  });

  test('every env has a coherent descriptor', () => {
    for (let kind = 0; kind < RECORD_V4_ENV_KINDS.length; kind++) {
      const envId = RECORD_V4_ENV_KINDS[kind] as EnvId;
      const desc = RECORD_V4_CODECS[kind] as CodecDescriptor;
      expect(envKindOf(envId)).toBe(kind);
      expect(codecForEnv(envId)).toBe(desc);
      expect(desc.category).toBe(ENVS[envId].category);

      const used = new Set<number>();
      for (const field of desc.poseFields) {
        expect(field.idx).toBeGreaterThanOrEqual(0);
        expect(field.idx).toBeLessThan(9);
        expect(used.has(field.idx)).toBe(false);
        used.add(field.idx);
        if (field.enc === 'u8') expect(field.scale).toBeUndefined();
        else expect(typeof field.scale).toBe('number');
      }
      for (const key of Object.keys(desc.poseConst ?? {})) {
        const idx = Number(key);
        expect(Number.isInteger(idx) && idx >= 0 && idx < 9).toBe(true);
        expect(used.has(idx)).toBe(false);
      }
      if (desc.ctrl !== 'index') expect(desc.ctrl.n).toBeLessThanOrEqual(6);
    }
  });

  test('a descriptor compares equal across the JSON round trip and unequal on any change', () => {
    const desc = codecForEnv('CartPole-cont-v0');
    expect(codecMatches(desc, JSON.parse(JSON.stringify(desc)))).toBe(true);
    expect(codecMatches(codecForEnv('FourRooms-cont-v0'), { ...codecForEnv('FourRooms-cont-v0'), poseConst: {} })).toBe(true);
    expect(codecMatches(codecForEnv('DeepSea-cont-v0'), { category: 'grid', ctrl: 'index', poseFields: [{ idx: 0, enc: 'u8' }, { idx: 1, enc: 'u8' }], poseConst: { 3: 7, 2: 7 } })).toBe(true);
    expect(codecMatches(desc, { ...desc, category: 'grid' })).toBe(false);
    expect(codecMatches(desc, { ...desc, ctrl: { kind: 'u8box', n: 1 } })).toBe(false);
    expect(codecMatches(desc, { ...desc, poseFields: desc.poseFields.slice(1) })).toBe(false);
    expect(codecMatches(desc, {
      ...desc,
      poseFields: desc.poseFields.map((f, i) => (i === 0 ? { ...f, scale: 1e-3 } : f)),
    })).toBe(false);
    expect(codecMatches(desc, {
      ...desc,
      poseFields: desc.poseFields.map((f, i) => (i === 0 ? { ...f, wrap: true } : f)),
    })).toBe(false);
    expect(codecMatches(codecForEnv('DeepSea-cont-v0'), { ...codecForEnv('DeepSea-cont-v0'), poseConst: { 2: 7, 3: 8 } })).toBe(false);
    expect(codecMatches(desc, null)).toBe(false);
    expect(codecMatches(desc, 'index')).toBe(false);
  });
});

describe('the spec numerics', () => {
  test('wrapAngle folds into [−π, π) with the floor rule', () => {
    expect(wrapAngle(0)).toBe(0);
    expect(wrapAngle(Math.PI)).toBeCloseTo(-Math.PI, 12);
    expect(wrapAngle(-Math.PI)).toBeCloseTo(-Math.PI, 12);
    expect(wrapAngle(Math.PI - 1e-9)).toBeCloseTo(Math.PI - 1e-9, 12);
    expect(wrapAngle(3 * Math.PI)).toBeCloseTo(-Math.PI, 9);
    expect(wrapAngle(-3 * Math.PI)).toBeCloseTo(-Math.PI, 9);
    for (const v of [-100, -7.3, -0.001, 0.5, 12.75, 400]) {
      const w = wrapAngle(v);
      expect(w).toBeGreaterThanOrEqual(-Math.PI);
      expect(w).toBeLessThan(Math.PI);
      expect(Math.abs(Math.sin(w) - Math.sin(v))).toBeLessThan(1e-9);
      expect(Math.abs(Math.cos(w) - Math.cos(v))).toBeLessThan(1e-9);
    }
  });

  test('round() is half AWAY FROM ZERO, as C and Swift do it — not Math.round', () => {
    expect(roundHalfAwayFromZero(0.5)).toBe(1);
    expect(roundHalfAwayFromZero(-0.5)).toBe(-1);
    expect(Math.round(-0.5)).toBe(-0);
    expect(roundHalfAwayFromZero(-1.5)).toBe(-2);
    expect(roundHalfAwayFromZero(2.5)).toBe(3);
    expect(roundHalfAwayFromZero(-2.4)).toBe(-2);
  });

  test('the ctrl quantizer clamps to [−1, 1] and round-trips inside half a quantum', () => {
    expect(quantizeCtrlU8(-1)).toBe(0);
    expect(quantizeCtrlU8(1)).toBe(255);
    expect(quantizeCtrlU8(-4)).toBe(0);
    expect(quantizeCtrlU8(4)).toBe(255);
    expect(dequantizeCtrlU8(0)).toBe(-1);
    expect(dequantizeCtrlU8(255)).toBe(1);
    for (let i = 0; i <= 400; i++) {
      const v = -1 + (2 * i) / 400;
      expect(Math.abs(dequantizeCtrlU8(quantizeCtrlU8(v)) - v)).toBeLessThanOrEqual(1 / 255 + 1e-12);
    }
  });

  test('crc32 matches the zlib vectors', () => {
    const enc = new TextEncoder();
    expect(crc32(new Uint8Array(0))).toBe(0);
    expect(crc32(enc.encode('a'))).toBe(0xe8b7be43);
    expect(crc32(enc.encode('abc'))).toBe(0x352441c2);
    expect(crc32(enc.encode('123456789'))).toBe(0xcbf43926);
    const padded = enc.encode('XXabcYY');
    expect(crc32(padded, 2, 5)).toBe(0x352441c2);
  });
});


describe('v3 → v4 → v3 over every real trace fixture', () => {
  for (const name of TRACE_NAMES) {
    test(`${name} survives the round trip`, async () => {
      const trace = loadTrace(name);
      const desc = codecForEnv(trace.envId);
      const enc = await encodeV4({
        envId: trace.envId,
        seed: trace.seed,
        data: trace.raw.data,
        steps: trace.raw.steps,
      });
      const raw4 = await parseActionsBin(enc.bytes, enc.meta);

      expect(raw4.envId).toBe(trace.envId);
      expect(raw4.seed).toBe(trace.seed);
      expect(raw4.steps).toBe(trace.raw.steps);

      if (desc.category === 'grid') {
        expectRowsIdentical(trace.raw.data, raw4.data, trace.raw.steps, `${name} v4`);
      } else {
        const worst = worstColumnError(trace.envId, trace.raw.data, raw4.data, trace.raw.steps);
        expect(worst).toBeLessThan(MAX_BUDGET_FRACTION);
        expect(worst).toBeGreaterThan(0.05);
      }

      for (let i = 0; i < trace.raw.steps; i++) {
        expect(at(raw4.data, i * RECORD_FLOATS + 6)).toBe(at(trace.raw.data, i * RECORD_FLOATS + 6));
      }
    });

    test(`${name} decodes to the same events and replay health`, async () => {
      const trace = loadTrace(name);
      const decoder = decoders[trace.envId] as EnvDecoder<unknown, unknown>;
      const enc = await encodeV4({
        envId: trace.envId,
        seed: trace.seed,
        data: trace.raw.data,
        steps: trace.raw.steps,
        chunkSteps: 37,
      });
      const raw4 = await parseActionsBin(enc.bytes, enc.meta);

      const d3 = decodeRecording(trace.raw, decoder);
      const d4 = decodeRecording(raw4, decoder);

      expect(d4.frames.length).toBe(d3.frames.length);
      if (TRACES_WITH_EVENTS.has(name)) expect(eventTimeline(d3).length).toBeGreaterThan(0);
      expect(eventTimeline(d4)).toEqual(eventTimeline(d3));
      expect(d4.nonFiniteRows).toBe(d3.nonFiniteRows);
      expect(d4.replay).toEqual(d3.replay);
      if (ENVS[trace.envId].category === 'grid') {
        expect(d4.frames.map((f) => f.state)).toEqual(d3.frames.map((f) => f.state));
      }
    });
  }

  test('chunking is invisible: 1, 37 and 200-step chunks decode identically', async () => {
    const trace = loadTrace('MultiKeyCorridor-cont-v0');
    const outs: Float32Array[] = [];
    for (const chunkSteps of [1, 37, 200]) {
      const enc = await encodeV4({
        envId: trace.envId,
        seed: trace.seed,
        data: trace.raw.data,
        steps: trace.raw.steps,
        chunkSteps,
      });
      expect(enc.chunks.length).toBe(Math.ceil(200 / chunkSteps));
      const raw4 = await parseActionsBin(enc.bytes, enc.meta);
      outs.push(raw4.data.slice());
    }
    for (const out of outs) {
      expectRowsIdentical(trace.raw.data, out, trace.raw.steps, 'chunked MKC');
    }
  });

  test('v4 is much smaller than v3 on a real cheetah lane', async () => {
    const trace = loadTrace('HalfCheetah-v4');
    const enc = await encodeV4({
      envId: trace.envId,
      seed: trace.seed,
      data: trace.raw.data,
      steps: trace.raw.steps,
      chunkSteps: 200,
    });
    expect(enc.bytes.byteLength).toBeLessThan(trace.raw.steps * 64 * 0.75);
  });
});


function makeRows(steps: number, fill: (row: Float32Array, i: number) => void): Float32Array {
  const data = new Float32Array(steps * RECORD_FLOATS);
  for (let i = 0; i < steps; i++) fill(data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS), i);
  return data;
}

describe('synthetic extremes', () => {
  test('a −400 m rootx teleport survives inside a chunk AND across a chunk seam', async () => {
    const steps = 40;
    const data = makeRows(steps, (row, i) => {
      for (let c = 0; c < 6; c++) row[c] = Math.sin(i * 0.3 + c) * 0.4;
      row[6] = 0;
      const jumped = (i >= 12 ? -400 : 0) + (i >= 20 ? -400 : 0);
      row[7] = jumped + i * 0.05;
      row[8] = 0.35;
      for (let p = 2; p < 9; p++) row[7 + p] = Math.cos(i * 0.11 + p) * 0.3;
    });
    for (const chunkSteps of [20, 40]) {
      const enc = await encodeV4({ envId: 'HalfCheetah-v4', data, steps, chunkSteps });
      const raw4 = await parseActionsBin(enc.bytes, enc.meta);
      expect(raw4.steps).toBe(steps);
      for (let i = 0; i < steps; i++) {
        const want = at(data, i * RECORD_FLOATS + 7);
        const got = at(raw4.data, i * RECORD_FLOATS + 7);
        expect(Math.abs(got - want)).toBeLessThanOrEqual(0.005 + Math.abs(want) * 1.2e-7 + 1e-9);
      }
      expect(scanChunksV4(enc.bytes).steps).toBe(steps);
    }
  });

  test('angles at the ±π boundary wrap losslessly for sin/cos', async () => {
    const boundaries = [
      Math.PI, -Math.PI, Math.PI - 1e-7, -Math.PI + 1e-7,
      3 * Math.PI, -3 * Math.PI, 0, 1e-8, -1e-8, 6.283185307,
    ];
    const steps = boundaries.length;
    const data = makeRows(steps, (row, i) => {
      row[0] = 0.5;
      row[6] = 0;
      row[7] = boundaries[i] ?? 0;
      row[8] = 1.25;
      row[9] = -0.5;
    });
    const enc = await encodeV4({ envId: 'Pendulum-cont-v0', data, steps });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    expect(raw4.steps).toBe(steps);
    for (let i = 0; i < steps; i++) {
      const want = at(data, i * RECORD_FLOATS + 7);
      const got = at(raw4.data, i * RECORD_FLOATS + 7);
      expect(got).toBeGreaterThanOrEqual(-Math.PI - 1e-4);
      expect(got).toBeLessThanOrEqual(Math.PI + 1e-4);
      expect(Math.abs(Math.sin(got) - Math.sin(want))).toBeLessThan(1e-4);
      expect(Math.abs(Math.cos(got) - Math.cos(want))).toBeLessThan(1e-4);
      expect(Math.abs(wrapAngle(got - want))).toBeLessThanOrEqual(5e-5 + 1e-6);
    }
  });

  test('a NaN/Infinity row is escaped verbatim, bit for bit', async () => {
    const steps = 12;
    const data = makeRows(steps, (row, i) => {
      row[0] = 1;
      row[6] = i === 7 ? 1 : 0;
      row[7] = i * 0.25;
      row[8] = -0.5;
      row[9] = 0.125;
    });
    data[3 * RECORD_FLOATS + 8] = Number.NaN;
    data[5 * RECORD_FLOATS + 7] = Number.POSITIVE_INFINITY;
    data[9 * RECORD_FLOATS + 9] = Number.NEGATIVE_INFINITY;

    const enc = await encodeV4({ envId: 'Pendulum-cont-v0', data, steps, chunkSteps: 4 });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    expect(raw4.steps).toBe(steps);
    for (const i of [3, 5, 9]) {
      for (let c = 0; c < RECORD_FLOATS; c++) {
        const a = at(data, i * RECORD_FLOATS + c);
        const b = at(raw4.data, i * RECORD_FLOATS + c);
        if (Number.isNaN(a)) expect(Number.isNaN(b)).toBe(true);
        else expect(b).toBe(a);
      }
    }
    const decoder = decoders['Pendulum-cont-v0'] as EnvDecoder<unknown, unknown>;
    const d3 = decodeRecording(parseRawTrace(new Uint8Array(data.buffer), {
      envId: 'Pendulum-cont-v0', seed: 0,
    }), decoder);
    expect(decodeRecording(raw4, decoder).nonFiniteRows).toBe(d3.nonFiniteRows);
    expect(d3.nonFiniteRows).toBe(3);
  });

  test('an escaped row is verbatim across ALL 16 floats — even channels the codec drops', async () => {
    const steps = 3;
    const data = makeRows(steps, (row, i) => {
      for (let c = 0; c < RECORD_FLOATS; c++) row[c] = (i + 1) * 1000 + c * 7 + 0.5;
      row[6] = i === 1 ? 1 : 0;
    });
    data[1 * RECORD_FLOATS + 4] = Number.NaN;
    const enc = await encodeV4({ envId: 'Pendulum-cont-v0', data, steps });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    expectRowsIdentical(
      data.subarray(RECORD_FLOATS, 2 * RECORD_FLOATS),
      raw4.data.subarray(RECORD_FLOATS, 2 * RECORD_FLOATS),
      1,
      'fully populated escaped row',
    );
  });

  test('a NON-escaped row DROPS the channels the codec does not carry (declared lossy surface)', async () => {
    const steps = 2;
    const data = makeRows(steps, (row) => {
      row[0] = 0.25;
      row[3] = 0.75;
      row[6] = 0;
      row[7] = 1.5;
      row[8] = -2;
      row[9] = 0.5;
      row[13] = 4.25;
      row[15] = 9.75;
    });
    const enc = await encodeV4({ envId: 'Pendulum-cont-v0', data, steps });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    expect(at(raw4.data, 3)).toBe(0);
    expect(at(raw4.data, 13)).toBe(0);
    expect(at(raw4.data, 15)).toBe(0);
    expect(at(raw4.data, 0)).toBeCloseTo(0.25, 2);
    expect(at(raw4.data, 7)).toBeCloseTo(1.5, 4);
  });

  test('a velocity past the i16 ceiling is escaped, not clipped', async () => {
    const steps = 8;
    const data = makeRows(steps, (row, i) => {
      row[0] = i % 2;
      row[6] = 0;
      row[7] = 0.05;
      row[8] = i === 4 ? 1e6 : 0.75;
      row[9] = 0.02;
      row[10] = i === 6 ? -12345.6 : 0.5;
    });
    const enc = await encodeV4({ envId: 'CartPole-cont-v0', data, steps });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    for (const i of [4, 6]) {
      expectRowsIdentical(
        data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS),
        raw4.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS),
        1,
        `escaped CartPole row ${i}`,
      );
    }
    expect(at(raw4.data, 0 * RECORD_FLOATS + 8)).toBeCloseTo(0.75, 3);
  });

  test('an escaped row stores the SYMMETRIC clamp in the quantized columns', async () => {
    const desc = codecForEnv('CartPole-cont-v0');
    const steps = 2;
    const data = makeRows(steps, (row, i) => {
      row[0] = 0;
      row[6] = 0;
      row[7] = 0.05;
      row[8] = i === 0 ? 1e6 : -1e6;
      row[9] = 0.02;
      row[10] = 0.5;
    });
    const { payload } = buildChunkPayload(desc, data, 0, steps);
    const pv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    const pose1Off = steps + 2 * steps;
    expect(pv.getInt16(pose1Off, true)).toBe(32767);
    expect(pv.getInt16(pose1Off + 2, true)).toBe(-32767);
    expect(payload[0]! & 0x20).toBe(0x20);
    expect(payload[1]! & 0x20).toBe(0x20);
  });

  test('an exact −n.5 quantization tie rounds AWAY from zero, end to end', async () => {
    const tie = -3.21875;
    expect(tie / 1e-4).toBe(-32187.5);
    const steps = 2;
    const data = makeRows(steps, (row, i) => {
      row[0] = 0;
      row[6] = 0;
      row[7] = i === 0 ? tie : 0.05;
      row[8] = 0.75;
      row[9] = 0.02;
      row[10] = 0.5;
    });
    const desc = codecForEnv('CartPole-cont-v0');
    const { payload } = buildChunkPayload(desc, data, 0, steps);
    const pv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    expect(pv.getInt16(steps, true)).toBe(-32188);
    expect(payload[0]! & 0x20).toBe(0);
    const enc = await encodeV4({ envId: 'CartPole-cont-v0', data, steps });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    expect(at(raw4.data, 7)).toBeCloseTo(-3.2188, 6);
  });

  test('a dx32 delta past the i32 ceiling clamps symmetrically and keeps the chain in step', async () => {
    const desc = codecForEnv('HalfCheetah-v4');
    const steps = 3;
    const data = makeRows(steps, (row, i) => {
      for (let c = 0; c < 6; c++) row[c] = 0.1;
      row[6] = 0;
      row[7] = i === 1 ? 3e7 : i === 2 ? -3e7 : 0;
      row[8] = 0.35;
      for (let p = 2; p < 9; p++) row[7 + p] = 0.1;
    });
    const { payload } = buildChunkPayload(desc, data, 0, steps);
    const pv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    const dxOff = steps * (1 + 6) + 8;
    expect(pv.getInt32(dxOff + 4, true)).toBe(2147483647);
    expect(pv.getInt32(dxOff + 8, true)).toBe(-2147483647);
    const enc = await encodeV4({ envId: 'HalfCheetah-v4', data, steps });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    for (const i of [1, 2]) {
      expect(at(raw4.data, i * RECORD_FLOATS + 7)).toBe(at(data, i * RECORD_FLOATS + 7));
    }
  });

  test('a u8 pose value outside [0, 255] escapes the whole grid row', async () => {
    const steps = 6;
    const data = makeRows(steps, (row, i) => {
      row[0] = i % 4;
      row[6] = 0;
      row[7] = i === 2 ? 900 : i;
      row[8] = i === 4 ? 1.5 : 3;
      row[9] = 9;
      row[10] = 9;
    });
    const enc = await encodeV4({ envId: 'FourRooms-cont-v0', data, steps });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    expectRowsIdentical(data, raw4.data, steps, 'FourRooms with out-of-range pose');
  });

  test('the hdr nibble carries the FULL 0..15 index alongside done and escape', async () => {
    const steps = 32;
    const data = makeRows(steps, (row, i) => {
      row[0] = i % 16;
      row[6] = i % 3 === 0 ? 1 : 0;
      row[7] = (i % 9) + 1;
      row[8] = (i % 7) + 1;
      row[9] = 9;
      row[10] = 9;
    });
    data[13 * RECORD_FLOATS + 8] = Number.NaN;
    const enc = await encodeV4({ envId: 'FourRooms-cont-v0', data, steps, chunkSteps: 11 });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    expectRowsIdentical(data, raw4.data, steps, 'full action nibble');
    for (let i = 0; i < steps; i++) {
      if (i === 13) continue;
      expect(at(raw4.data, i * RECORD_FLOATS)).toBe(i % 16);
      expect(at(raw4.data, i * RECORD_FLOATS + 6)).toBe(i % 3 === 0 ? 1 : 0);
    }
  });

  test('an action index outside [0, 15] escapes the row', async () => {
    const steps = 4;
    const data = makeRows(steps, (row, i) => {
      row[0] = i === 1 ? 77 : i === 2 ? 1.5 : i;
      row[6] = 0;
      row[7] = 2;
      row[8] = 3;
      row[9] = 9;
      row[10] = 9;
    });
    const enc = await encodeV4({ envId: 'FourRooms-cont-v0', data, steps });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    expectRowsIdentical(data, raw4.data, steps, 'FourRooms with a bad action index');
  });

  test('a done that is not exactly 0 or 1 ESCAPES — it is never rounded into the flag', async () => {
    const steps = 5;
    const doneValues = [0, 1, 0.5, Number.NaN, 1];
    const data = makeRows(steps, (row, i) => {
      row[0] = i % 4;
      row[6] = doneValues[i] as number;
      row[7] = 2;
      row[8] = 3;
      row[9] = 9;
      row[10] = 9;
    });
    const enc = await encodeV4({ envId: 'FourRooms-cont-v0', data, steps });
    const raw4 = await parseActionsBin(enc.bytes, enc.meta);
    expectRowsIdentical(data, raw4.data, steps, 'FourRooms with a non-binary done');
    const counted: number[] = [];
    await walkChunksV4(enc.bytes, enc.meta, (rows) => {
      counted.push(rows.chunk.doneCount);
    });
    expect(counted).toEqual([2]);
  });
});


async function chainFixture(steps = 60, chunkSteps = 20): Promise<EncodedV4 & { data: Float32Array }> {
  const data = makeRows(steps, (row, i) => {
    row[0] = i % 2;
    row[6] = i % 17 === 0 ? 1 : 0;
    row[7] = i % 15;
    row[8] = 15;
  });
  const enc = await encodeV4({ envId: 'Chain-cont-v0', data, steps, chunkSteps });
  return { ...enc, data };
}

describe('torn tails, damage and truncation', () => {
  test('a file cut mid-chunk decodes the whole chunks before the cut', async () => {
    const fx = await chainFixture();
    expect(fx.chunks.length).toBe(3);
    const secondEnd = (fx.chunks[1] as [number, number, number, number])[2] +
      (fx.chunks[1] as [number, number, number, number])[3];
    for (const cut of [secondEnd + 4, secondEnd + CHUNK_HEADER_BYTES - 1, secondEnd + 1]) {
      const torn = fx.bytes.slice(0, cut);
      const scan = scanChunksV4(torn);
      expect(scan.chunks.length).toBe(2);
      expect(scan.steps).toBe(40);
      expect(scan.truncatedReason).not.toBeNull();
      const raw4 = await parseActionsBin(torn, fx.meta);
      expect(raw4.steps).toBe(40);
      expect(raw4.data.length).toBe(40 * RECORD_FLOATS);
      expectRowsIdentical(fx.data, raw4.data, 40, 'torn Chain lane');
    }
  });

  test('a header-only file is a zero-step lane, not an error', async () => {
    const fx = await chainFixture();
    const raw4 = await parseActionsBin(fx.bytes.slice(0, FILE_HEADER_BYTES), fx.meta);
    expect(raw4.steps).toBe(0);
    expect(raw4.data.length).toBe(0);
  });

  test('a crc mismatch drops that chunk and everything after it', async () => {
    const fx = await chainFixture();
    const damaged = fx.bytes.slice();
    const second = fx.chunks[1] as [number, number, number, number];
    const flipAt = second[2] + CHUNK_HEADER_BYTES + 2;
    damaged[flipAt] = (damaged[flipAt] ?? 0) ^ 0xff;
    const scan = scanChunksV4(damaged);
    expect(scan.chunks.length).toBe(1);
    expect(scan.truncatedReason).toBe('chunk crc32 mismatch');
    expect((await parseActionsBin(damaged, fx.meta)).steps).toBe(20);
    expect(scanChunksV4(damaged, { verifyCrc: false }).steps).toBe(60);
  });

  test('a chunk whose payload inflates to the wrong size is dropped', async () => {
    const fx = await chainFixture();
    const damaged = fx.bytes.slice();
    const second = fx.chunks[1] as [number, number, number, number];
    const dv = new DataView(damaged.buffer);
    dv.setUint32(second[2] + 12, 999999, true);
    expect(scanChunksV4(damaged).chunks.length).toBe(3);
    expect((await parseActionsBin(damaged, fx.meta)).steps).toBe(20);
  });

  test('a chunk whose stepStart does not continue the file is dropped', async () => {
    const fx = await chainFixture();
    const damaged = fx.bytes.slice();
    const second = fx.chunks[1] as [number, number, number, number];
    new DataView(damaged.buffer).setUint32(second[2] + 4, 999, true);
    const scan = scanChunksV4(damaged);
    expect(scan.chunks.length).toBe(1);
    expect(scan.truncatedReason).toContain('does not continue');
  });

  test('a crc-valid chunk that is structurally wrong THROWS, it does not truncate', async () => {
    const steps = 10;
    const data = makeRows(steps, (row, i) => {
      row[0] = i % 2;
      row[6] = 0;
      row[7] = i;
      row[8] = 15;
    });
    const desc = codecForEnv('Chain-cont-v0');
    const { payload, doneCount } = buildChunkPayload(desc, data, 0, steps);
    const meta: RecordingMeta = {
      envId: 'Chain-cont-v0', seed: 0, actDim: 2, codecVersion: 1,
    };

    const overrun = payload.slice();
    new DataView(overrun.buffer).setUint32(overrun.byteLength - 4, 3, true);
    const a = await assembleV4(envKindOf('Chain-cont-v0'), [
      { payload: overrun, stepStart: 0, stepCount: steps, doneCount },
    ]);
    expect(scanChunksV4(a.bytes).chunks.length).toBe(1);
    await expect(parseActionsBin(a.bytes, meta)).rejects.toThrow(/escape table/);

    const b = await assembleV4(envKindOf('Chain-cont-v0'), [
      { payload: payload.slice(), stepStart: 0, stepCount: steps + 40, doneCount },
    ]);
    await expect(parseActionsBin(b.bytes, meta)).rejects.toThrow(RecordParseError);

    await expect(walkChunksV4(a.bytes, meta, () => undefined)).rejects.toThrow(RecordParseError);
  });

  test('a foreign chunk magic ends the scan', async () => {
    const fx = await chainFixture();
    const damaged = fx.bytes.slice();
    const second = fx.chunks[1] as [number, number, number, number];
    new DataView(damaged.buffer).setUint32(second[2], 0xdeadbeef, true);
    expect(scanChunksV4(damaged).steps).toBe(20);
  });
});


describe('header and meta validation', () => {
  test('rejects a short, foreign or future file header', async () => {
    const fx = await chainFixture();
    expect(() => readFileHeaderV4(fx.bytes.slice(0, 12))).toThrow(RecordParseError);

    const foreign = fx.bytes.slice();
    new DataView(foreign.buffer).setUint32(0, 0x12345678, true);
    expect(() => readFileHeaderV4(foreign)).toThrow(/file magic/);

    const future = fx.bytes.slice();
    new DataView(future.buffer).setUint32(4, 5, true);
    expect(() => readFileHeaderV4(future)).toThrow(/formatVersion/);

    const newCodec = fx.bytes.slice();
    new DataView(newCodec.buffer).setUint32(12, 2, true);
    expect(() => readFileHeaderV4(newCodec)).toThrow(/codecVersion/);
  });

  test('the parser refuses a non-record buffer', async () => {
    const fx = await chainFixture();
    await expect(parseActionsBin(new Uint8Array(64), fx.meta))
      .rejects.toBeInstanceOf(RecordParseError);
  });

  test('an invalid seed is rejected', async () => {
    const fx = await chainFixture();
    await expect(
      parseActionsBin(fx.bytes, { ...fx.meta, seed: -1 }),
    ).rejects.toBeInstanceOf(RecordParseError);
  });

  test('the header envKind must agree with the meta envId', async () => {
    const fx = await chainFixture();
    const relabelled = fx.bytes.slice();
    new DataView(relabelled.buffer).setUint32(8, 2, true);
    await expect(parseActionsBin(relabelled, fx.meta)).rejects.toThrow(/contradicts meta envId/);
  });

  test('a meta codec descriptor that disagrees with the built-in table is fatal', async () => {
    const fx = await chainFixture();
    const wrong = {
      ...fx.meta,
      codec: { category: 'grid', ctrl: 'index', poseFields: [{ idx: 0, enc: 'u8' }], poseConst: { 1: 16 } },
    };
    await expect(parseActionsBin(fx.bytes, wrong)).rejects.toThrow(/codec descriptor/);
    const bare: RecordingMeta = {
      envId: fx.meta.envId, seed: fx.meta.seed, actDim: fx.meta.actDim,
    };
    expect((await parseActionsBin(fx.bytes, bare)).steps).toBe(60);
  });

  test('a meta codecVersion from another codec revision is fatal', async () => {
    const fx = await chainFixture();
    await expect(
      parseActionsBin(fx.bytes, { ...fx.meta, codecVersion: 2 }),
    ).rejects.toThrow(/codecVersion/);
  });

  test('an unknown envId or a negative seed is rejected before any byte is read', async () => {
    const fx = await chainFixture();
    await expect(parseActionsBin(fx.bytes, { ...fx.meta, envId: 'Nope-v0' })).rejects.toThrow(/envId/);
    await expect(parseActionsBin(fx.bytes, { ...fx.meta, seed: 1.5 })).rejects.toThrow(/seed/);
  });
});


describe('buffer views', () => {
  test('only the view\'s own window is read (the pooled Buffer trap)', async () => {
    const fx = await chainFixture();
    const padded = new Uint8Array(fx.bytes.byteLength + 96);
    padded.fill(0xab);
    padded.set(fx.bytes, 33);
    const view = padded.subarray(33, 33 + fx.bytes.byteLength);

    const raw4 = await parseActionsBin(view, fx.meta);
    expect(raw4.steps).toBe(60);
    expectRowsIdentical(fx.data, raw4.data, 60, 'unaligned view');
    expect((await parseActionsBin(fx.bytes.slice().buffer, fx.meta)).steps).toBe(60);
    expect((await parseActionsBin(new DataView(padded.buffer, 33, fx.bytes.byteLength), fx.meta)).steps)
      .toBe(60);
  });
});


describe('walkChunksV4', () => {
  test('visits every chunk and reproduces parseActionsBin exactly', async () => {
    const trace = loadTrace('HalfCheetah-v4');
    const enc = await encodeV4({
      envId: trace.envId, seed: trace.seed, data: trace.raw.data, steps: trace.raw.steps, chunkSteps: 64,
    });
    const whole = await parseActionsBin(enc.bytes, enc.meta);

    const seen: ChunkRowsV4[] = [];
    const rebuilt = new Float32Array(trace.raw.steps * RECORD_FLOATS);
    const total = await walkChunksV4(enc.bytes, enc.meta, (rows) => {
      seen.push({ ...rows, data: rows.data.slice() });
      rebuilt.set(rows.data, rows.stepStart * RECORD_FLOATS);
    });

    expect(total).toBe(trace.raw.steps);
    expect(seen.map((s) => s.stepCount)).toEqual([64, 64, 64, 8]);
    expect(seen.map((s) => s.stepStart)).toEqual([0, 64, 128, 192]);
    expectRowsIdentical(whole.data, rebuilt, trace.raw.steps, 'walk vs parse');
  });

  test('the row buffer is REUSED between chunks — a retained view aliases', async () => {
    const fx = await chainFixture(60, 20);
    const retained: Float32Array[] = [];
    await walkChunksV4(fx.bytes, fx.meta, (rows) => {
      retained.push(rows.data);
    });
    expect(retained.length).toBe(3);
    expect(retained[1]?.buffer).toBe(retained[0]?.buffer as ArrayBufferLike);
    expect(retained[2]?.buffer).toBe(retained[0]?.buffer as ArrayBufferLike);
  });

  test('returning false stops the walk', async () => {
    const fx = await chainFixture(60, 20);
    let visits = 0;
    const total = await walkChunksV4(fx.bytes, fx.meta, () => {
      visits++;
      return visits < 2;
    });
    expect(visits).toBe(2);
    expect(total).toBe(40);
  });

  test('an async visitor is awaited', async () => {
    const fx = await chainFixture(60, 20);
    const order: string[] = [];
    await walkChunksV4(fx.bytes, fx.meta, async (rows) => {
      order.push(`enter ${rows.stepStart}`);
      await Promise.resolve();
      order.push(`leave ${rows.stepStart}`);
    });
    expect(order).toEqual([
      'enter 0', 'leave 0', 'enter 20', 'leave 20', 'enter 40', 'leave 40',
    ]);
  });

  test('the doneCount and chunk framing in the header match the rows', async () => {
    const fx = await chainFixture(60, 20);
    await walkChunksV4(fx.bytes, fx.meta, (rows) => {
      let done = 0;
      for (let i = 0; i < rows.stepCount; i++) {
        if (at(rows.data, i * RECORD_FLOATS + 6) !== 0) done++;
      }
      expect(done).toBe(rows.chunk.doneCount);
      expect(rows.chunk.byteLength).toBe(CHUNK_HEADER_BYTES + rows.chunk.compBytes);
      expect(Number.isNaN(rows.chunk.rewardDelta)).toBe(true);
    });
  });
});


describe('chunk-table range planning', () => {
  test('plans the fewest WHOLE chunks covering the wanted steps', async () => {
    const fx = await chainFixture(100, 10);
    const table = chunkTableFromMeta(fx.meta);
    expect(table).not.toBeNull();
    expect(table?.length).toBe(10);

    const one = planPrefixV4(table ?? [], 1);
    expect(one.chunks).toBe(1);
    expect(one.steps).toBe(10);
    expect(one.bytes).toBe((fx.chunks[0] as number[])[2]! + (fx.chunks[0] as number[])[3]!);

    const exact = planPrefixV4(table ?? [], 30);
    expect(exact.chunks).toBe(3);
    expect(exact.steps).toBe(30);

    const over = planPrefixV4(table ?? [], 31);
    expect(over.chunks).toBe(4);
    expect(over.steps).toBe(40);

    const all = planPrefixV4(table ?? [], 10_000);
    expect(all.chunks).toBe(10);
    expect(all.bytes).toBe(fx.bytes.byteLength);

    expect(planPrefixV4(table ?? [], 0)).toEqual({ chunks: 0, steps: 0, bytes: 0 });
    expect(planPrefixV4([], 50)).toEqual({ chunks: 0, steps: 0, bytes: 0 });
  });

  test('a planned prefix really does decode to at least the wanted steps', async () => {
    const fx = await chainFixture(100, 10);
    const table = chunkTableFromMeta(fx.meta) ?? [];
    const plan = planPrefixV4(table, 25);
    const prefix = fx.bytes.slice(0, plan.bytes);
    const raw4 = await parseActionsBin(prefix, fx.meta);
    expect(raw4.steps).toBe(plan.steps);
    expect(raw4.steps).toBeGreaterThanOrEqual(25);
    expectRowsIdentical(fx.data, raw4.data, raw4.steps, 'planned prefix');
  });

  test('an absent, empty or incoherent chunk table falls back to the header scan', async () => {
    const fx = await chainFixture(60, 20);
    const bare = await encodeV4({
      envId: 'Chain-cont-v0', data: fx.data, steps: 60, chunkSteps: 20, omitChunkTable: true,
    });
    expect(bare.meta.chunks).toBeUndefined();
    expect(bare.meta.steps).toBeUndefined();
    expect(chunkTableFromMeta(bare.meta)).toBeNull();
    expect(rangeHeaderForSteps(bare.meta, 20)).toBeNull();
    expect((await parseActionsBin(bare.bytes, bare.meta)).steps).toBe(60);

    expect(chunkTableFromMeta({ chunks: [] })).toBeNull();
    expect(chunkTableFromMeta({ chunks: undefined })).toBeNull();
    expect(chunkTableFromMeta({ chunks: [[0, 10, 999, 100]] })).toBeNull();
    expect(chunkTableFromMeta({ chunks: [[0, 10, FILE_HEADER_BYTES, 100], [99, 10, FILE_HEADER_BYTES + 100, 100]] })).toBeNull();
    expect(chunkTableFromMeta({ chunks: [[0, 0, FILE_HEADER_BYTES, 100]] })).toBeNull();
    expect(chunkTableFromMeta({ chunks: [[0, 10, FILE_HEADER_BYTES, 8]] })).toBeNull();
    expect(chunkTableFromMeta({ chunks: [[0, 10, FILE_HEADER_BYTES] as unknown as [number, number, number, number]] })).toBeNull();
  });

  test('rangeHeaderForSteps needs a chunk table', async () => {
    const fx = await chainFixture(60, 20);
    expect(rangeHeaderForSteps({}, 10)).toBeNull();
    const header = rangeHeaderForSteps(fx.meta, 25);
    expect(header).toBe(`bytes=0-${planPrefixV4(chunkTableFromMeta(fx.meta) ?? [], 25).bytes - 1}`);
  });
});


interface LaneServerLog {
  ranges: (string | null)[];
}

function laneFetch(
  enc: EncodedV4,
  log: LaneServerLog,
  opts: { runId?: string; steps?: number } = {},
): typeof fetch {
  const impl = async (input: unknown, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    if (url.endsWith(`/api/runs/${opts.runId ?? 'demo'}`)) {
      return Response.json({
        id: opts.runId ?? 'demo',
        meta: {},
        summary: null,
        aggregate: [],
        deathsCount: 0,
        recordings: [{ envId: enc.meta.envId, seed: enc.meta.seed, steps: opts.steps ?? 0 }],
      });
    }
    if (url.endsWith('/meta.json')) return Response.json(enc.meta);
    const headers = new Headers((init?.headers ?? {}) as HeadersInit);
    const range = headers.get('range');
    log.ranges.push(range);
    if (range === null) {
      return new Response(enc.bytes.slice(), {
        status: 200,
        headers: { 'content-length': String(enc.bytes.byteLength) },
      });
    }
    const m = /^bytes=(\d+)-(\d+)$/.exec(range);
    const start = Number(m?.[1] ?? 0);
    const end = Math.min(Number(m?.[2] ?? 0), enc.bytes.byteLength - 1);
    const slice = enc.bytes.slice(start, end + 1);
    return new Response(slice, {
      status: 206,
      headers: {
        'content-length': String(slice.byteLength),
        'content-range': `bytes ${start}-${end}/${enc.bytes.byteLength}`,
      },
    });
  };
  return impl as unknown as typeof fetch;
}

describe('the theater loader on a v4 lane', () => {
  test('fetches WHOLE chunks covering the wanted steps and decodes them', async () => {
    const fx = await chainFixture(100, 10);
    const log: LaneServerLog = { ranges: [] };
    const rec = await loadTheaterRecording({
      runId: 'demo',
      envId: 'Chain-cont-v0',
      seed: 0,
      expectedSteps: 100,
      fetchImpl: laneFetch(fx, log),
    });

    expect(log.ranges).toEqual([null]);
    expect(rec.steps).toBe(100);
    expect(rec.availableSteps).toBe(100);
    expect(rec.source.kind).toBe('recording');
    expect(rec.source.frameCount).toBe(101);
    expect(rec.health.damaged).toBe(false);
  });

  test('a v4 lane with no chunk table is fetched whole', async () => {
    const fx = await chainFixture(60, 20);
    const bare = await encodeV4({
      envId: 'Chain-cont-v0', data: fx.data, steps: 60, chunkSteps: 20, omitChunkTable: true,
    });
    const log: LaneServerLog = { ranges: [] };
    const rec = await loadTheaterRecording({
      runId: 'demo', envId: 'Chain-cont-v0', seed: 0,
      fetchImpl: laneFetch(bare, log),
    });
    expect(log.ranges).toEqual([null]);
    expect(rec.bytes).toBe(bare.bytes.byteLength);
    expect(rec.steps).toBe(60);
  });

  test('a v4 body cut mid-chunk still opens on the whole chunks before the cut', async () => {
    const fx = await chainFixture(60, 20);
    const cut = (fx.chunks[1] as number[])[2]! + 6;
    const truncated: EncodedV4 = { ...fx, bytes: fx.bytes.slice(0, cut) };
    const log: LaneServerLog = { ranges: [] };
    const rec = await loadTheaterRecording({
      runId: 'demo', envId: 'Chain-cont-v0', seed: 0,
      fetchImpl: laneFetch(truncated, log),
    });
    expect(rec.steps).toBe(20);
  });

  test('a v4 body with no valid chunk is an error, not a zero-frame scene', async () => {
    const fx = await chainFixture(60, 20);
    const headerOnly: EncodedV4 = { ...fx, bytes: fx.bytes.slice(0, FILE_HEADER_BYTES) };
    await expect(
      loadTheaterRecording({
        runId: 'demo', envId: 'Chain-cont-v0', seed: 0,
        fetchImpl: laneFetch(headerOnly, { ranges: [] }),
      }),
    ).rejects.toBeInstanceOf(TheaterLoadError);
  });
});


const HERE = dirname(fileURLToPath(import.meta.url));
const DEMO_ROOT = resolve(HERE, '..', '..', '.oak-benchmarks', 'frontend-demo-16env-20260803');
const demoPresent = existsSync(DEMO_ROOT);

function demoLane(envId: EnvId): { path: string; seed: number } | null {
  const dir = join(DEMO_ROOT, envId);
  if (!existsSync(dir)) return null;
  const seeds = readdirSync(dir).filter((d) => /^seed\d+$/.test(d)).sort();
  const first = seeds[0];
  if (first === undefined) return null;
  return { path: join(dir, first, 'actions.bin'), seed: Number(first.slice(4)) };
}

describe.skipIf(!demoPresent)('real 100k-step demo lanes', () => {
  const CASES: { envId: EnvId; minRatio: number }[] = [
    { envId: 'HalfCheetah-v4', minRatio: 3 },
    { envId: 'CartPole-cont-v0', minRatio: 15 },
    { envId: 'FourRooms-cont-v0', minRatio: 40 },
    { envId: 'MultiKeyCorridor-cont-v0', minRatio: 40 },
  ];

  for (const { envId, minRatio } of CASES) {
    test(`${envId} round-trips at 100k steps and compresses ≥ ${minRatio}×`, async () => {
      const lane = demoLane(envId);
      expect(lane).not.toBeNull();
      const buf = readFileSync((lane as { path: string }).path);
      const seed = (lane as { seed: number }).seed;
      const raw3 = parseRawTrace(buf, { envId, seed });
      expect(raw3.steps).toBeGreaterThanOrEqual(100_000);

      const enc = await encodeV4({
        envId, seed, data: raw3.data, steps: raw3.steps, chunkSteps: 4096,
      });
      const ratio = buf.byteLength / enc.bytes.byteLength;
      const raw4 = await parseActionsBin(enc.bytes, enc.meta);

      expect(raw4.steps).toBe(raw3.steps);
      if (ENVS[envId].category === 'grid') {
        expectRowsIdentical(raw3.data, raw4.data, raw3.steps, `${envId} @100k`);
      } else {
        expect(worstColumnError(envId, raw3.data, raw4.data, raw3.steps)).toBeLessThan(MAX_BUDGET_FRACTION);
      }
      expect(ratio).toBeGreaterThanOrEqual(minRatio);
      console.log(
        `    ${envId.padEnd(26)} ${raw3.steps} steps · ` +
        `${(buf.byteLength / 1048576).toFixed(2)} MiB → ${(enc.bytes.byteLength / 1048576).toFixed(3)} MiB ` +
        `(${ratio.toFixed(1)}×, ${enc.chunks.length} chunks)`,
      );

      const plan = planPrefixV4(chunkTableFromMeta(enc.meta) ?? [], 2000);
      expect(plan.chunks).toBe(1);
      const prefix = await parseActionsBin(enc.bytes.slice(0, plan.bytes), enc.meta);
      expect(prefix.steps).toBe(4096);
    });
  }
});

describe('the artisan harness on a v4 lane', () => {
  test('loads real experience rather than falling back to the fixture loop', async () => {
    const fx = await chainFixture(100, 10);
    const log: LaneServerLog = { ranges: [] };
    const realFetch = globalThis.fetch;
    globalThis.fetch = laneFetch(fx, log, { runId: 'v4-run', steps: 100 });
    try {
      const source = await loadHarnessSource('Chain-cont-v0', { runId: 'v4-run' });
      expect(source.note).toBeUndefined();
      expect(source.kind).toBe('recording');
      expect(source.frameCount).toBe(101);
      expect(log.ranges).toEqual([null]);
    } finally {
      globalThis.fetch = realFetch;
    }
  });
});
