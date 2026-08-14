import { describe, expect, test } from 'bun:test';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ENV_IDS, ENVS, type EnvId } from '@shared/envs';
import {
  CHUNK_HEADER_BYTES,
  FILE_HEADER_BYTES,
  RECORD_FLOATS,
  RecordParseError,
  codecForEnv,
  crc32,
  parseRecord,
  readRecordHeader,
  scanNextChunk,
  scanRecord,
} from '@sim/record';
import { encodeRecord } from './helpers/encode';
import { parseRawTrace } from './helpers/rawTrace';

const tracesDir = fileURLToPath(new URL('./fixtures/traces/', import.meta.url));

interface TraceDoc {
  envId: EnvId;
  seed: number;
  raw: Float32Array;
  steps: number;
}

function loadTraces(): TraceDoc[] {
  const out: TraceDoc[] = [];
  for (const name of readdirSync(tracesDir)) {
    if (!name.endsWith('.json') || name.endsWith('.events.json')) continue;
    const meta = JSON.parse(readFileSync(`${tracesDir}${name}`, 'utf8')) as { envId: EnvId; seed: number };
    const bin = readFileSync(`${tracesDir}${name.replace(/\.json$/, '.bin')}`);
    const raw = parseRawTrace(bin, meta);
    out.push({ envId: meta.envId, seed: meta.seed, raw: new Float32Array(raw.data), steps: raw.steps });
  }
  return out;
}

const traces = loadTraces();

function bitsEqual(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
  for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
  return true;
}

describe('the codec projection', () => {
  test('covers a trace fixture for every environment', () => {
    const covered = new Set(traces.map((t) => t.envId));
    for (const envId of ENV_IDS) expect(covered.has(envId)).toBe(true);
  });

  for (const doc of traces) {
    test(`${doc.envId} seed ${doc.seed}: decode∘encode is idempotent`, async () => {
      const first = encodeRecord(doc.envId, doc.raw, doc.steps, 256, { wrap: true });
      const once = await parseRecord(first.bytes, doc.seed);
      expect(once.envId).toBe(doc.envId);
      expect(once.steps).toBe(first.steps);

      const second = encodeRecord(doc.envId, new Float32Array(once.data), once.steps, 256);
      const twice = await parseRecord(second.bytes, doc.seed);
      expect(twice.steps).toBe(once.steps);
      expect(bitsEqual(new Float32Array(once.data), new Float32Array(twice.data))).toBe(true);
    });
  }

  for (const doc of traces) {
    test(`${doc.envId} seed ${doc.seed}: quantization stays within spec`, async () => {
      const encoded = encodeRecord(doc.envId, doc.raw, doc.steps, 512, { wrap: true });
      const decoded = await parseRecord(encoded.bytes, doc.seed);
      const desc = codecForEnv(doc.envId);
      for (let i = 0; i < decoded.steps; i++) {
        const row = i * RECORD_FLOATS;
        const done = doc.raw[row + 6] ?? 0;
        const gotDone = decoded.data[row + 6] ?? 0;
        if (done === 0 || done === 1) expect(gotDone).toBe(done);
        for (const f of desc.poseFields) {
          const want = doc.raw[row + 7 + f.idx] ?? 0;
          const got = decoded.data[row + 7 + f.idx] ?? 0;
          if (got === want) continue;
          if (f.enc === 'u8') {
            expect(got).toBe(want);
          } else if (f.enc === 'i16') {
            const scale = f.scale ?? 1;
            const wrapped = f.wrap === true
              ? want - 2 * Math.PI * Math.floor((want + Math.PI) / (2 * Math.PI))
              : want;
            let d = Math.abs(got - wrapped);
            if (f.wrap === true) d = Math.min(d, Math.abs(d - 2 * Math.PI));
            expect(d).toBeLessThanOrEqual(scale * 0.5 + Math.abs(wrapped) * 1e-6);
          } else {
            const scale = f.scale ?? 1;
            expect(Math.abs(got - want)).toBeLessThanOrEqual(scale * 0.5 + Math.abs(want) * 1e-6);
          }
        }
      }
    });
  }

  test('escape rows carry exact raw values', async () => {
    const steps = 8;
    const data = new Float32Array(steps * RECORD_FLOATS);
    for (let i = 0; i < steps; i++) {
      data[i * RECORD_FLOATS] = 3;
      data[i * RECORD_FLOATS + 7] = i;
      data[i * RECORD_FLOATS + 8] = 1;
    }
    data[3 * RECORD_FLOATS + 7] = 977.25;
    data[5 * RECORD_FLOATS + 6] = 0.5;
    const encoded = encodeRecord('FourRooms-cont-v0', data, steps, 4);
    const decoded = await parseRecord(encoded.bytes, 0);
    expect(decoded.steps).toBe(steps);
    expect(decoded.data[3 * RECORD_FLOATS + 7]).toBe(977.25);
    expect(decoded.data[5 * RECORD_FLOATS + 6]).toBe(0.5);
  });

  test('a non-finite row ends the recording at the last finite row', async () => {
    const steps = 10;
    const data = new Float32Array(steps * RECORD_FLOATS);
    for (let i = 0; i < steps; i++) data[i * RECORD_FLOATS + 7] = i % 8;
    data[6 * RECORD_FLOATS + 7] = Number.NaN;
    const encoded = encodeRecord('FourRooms-cont-v0', data, steps, 4);
    expect(encoded.closed).toBe(true);
    expect(encoded.steps).toBe(6);
    const decoded = await parseRecord(encoded.bytes, 0);
    expect(decoded.steps).toBe(6);
  });

  test('the delta chain survives full-range i16 swings', async () => {
    const steps = 6;
    const data = new Float32Array(steps * RECORD_FLOATS);
    const values = [3.2767, -3.2767, 3.2767, 0, -3.2767, 3.2767];
    for (let i = 0; i < steps; i++) data[i * RECORD_FLOATS + 7] = values[i] ?? 0;
    const encoded = encodeRecord('Pendulum-cont-v0', data, steps, 6);
    const decoded = await parseRecord(encoded.bytes, 0);
    for (let i = 0; i < steps; i++) {
      const got = decoded.data[i * RECORD_FLOATS + 7] ?? 0;
      expect(Math.abs(got - (values[i] ?? 0))).toBeLessThanOrEqual(1e-4 * 0.5 + 1e-6);
    }
  });
});

describe('the container', () => {
  const sample = (): Uint8Array => {
    const doc = traces.find((t) => t.envId === 'MountainCar-cont-v0') ?? traces[0]!;
    return encodeRecord(doc.envId, doc.raw, doc.steps, 128).bytes;
  };

  test('readRecordHeader rejects a foreign magic', () => {
    const bytes = sample().slice();
    bytes[0] = bytes[0]! ^ 0xff;
    expect(() => readRecordHeader(bytes)).toThrow(RecordParseError);
  });

  test('readRecordHeader rejects an unknown envKind', () => {
    const bytes = sample().slice();
    new DataView(bytes.buffer).setUint32(4, 99, true);
    expect(() => readRecordHeader(bytes)).toThrow(RecordParseError);
  });

  test('a torn tail is dropped, whole chunks survive', () => {
    const bytes = sample();
    const whole = scanRecord(bytes);
    expect(whole.chunks.length).toBeGreaterThan(1);
    const cut = whole.chunks[whole.chunks.length - 1]!;
    const torn = bytes.slice(0, cut.byteOffset + CHUNK_HEADER_BYTES + 3);
    const scan = scanRecord(torn);
    expect(scan.chunks.length).toBe(whole.chunks.length - 1);
    expect(scan.steps).toBe(whole.steps - cut.stepCount);
  });

  test('a corrupted payload stops the scan at the previous chunk', () => {
    const bytes = sample().slice();
    const whole = scanRecord(bytes);
    const second = whole.chunks[1]!;
    bytes[second.payloadOffset] = bytes[second.payloadOffset]! ^ 0xff;
    const scan = scanRecord(bytes);
    expect(scan.chunks.length).toBe(1);
  });

  test('crc verification can be skipped', () => {
    const bytes = sample().slice();
    const whole = scanRecord(bytes);
    const second = whole.chunks[1]!;
    bytes[second.payloadOffset] = bytes[second.payloadOffset]! ^ 0xff;
    const scan = scanRecord(bytes, { verifyCrc: false });
    expect(scan.chunks.length).toBe(whole.chunks.length);
  });

  test('scanNextChunk resumes exactly where scanRecord walks', () => {
    const bytes = sample();
    const whole = scanRecord(bytes);
    let offset = FILE_HEADER_BYTES;
    let steps = 0;
    for (const expected of whole.chunks) {
      const next = scanNextChunk(bytes, offset, steps, expected.index);
      expect(next).not.toBeNull();
      expect(next!.stepStart).toBe(expected.stepStart);
      expect(next!.stepCount).toBe(expected.stepCount);
      expect(crc32(bytes, next!.payloadOffset, next!.payloadOffset + next!.compBytes)).toBe(next!.crc32);
      offset = next!.byteOffset + next!.byteLength;
      steps += next!.stepCount;
    }
    expect(scanNextChunk(bytes, offset, steps, whole.chunks.length)).toBeNull();
  });

  test('scanNextChunk reports an incomplete tail as null, then finds it once whole', () => {
    const bytes = sample();
    const whole = scanRecord(bytes);
    const last = whole.chunks[whole.chunks.length - 1]!;
    const stepsBefore = whole.steps - last.stepCount;
    const torn = bytes.subarray(0, last.byteOffset + 10);
    expect(scanNextChunk(torn, last.byteOffset, stepsBefore, last.index)).toBeNull();
    expect(scanNextChunk(bytes, last.byteOffset, stepsBefore, last.index)).not.toBeNull();
  });

  test('ctrl round-trips for the discrete and box categories', async () => {
    for (const envId of ['DeepSea-cont-v0', 'HalfCheetah-v4'] as const) {
      const doc = traces.find((t) => t.envId === envId)!;
      const encoded = encodeRecord(doc.envId, doc.raw, doc.steps, 256, { wrap: true });
      const decoded = await parseRecord(encoded.bytes, doc.seed);
      const nBox = ENVS[envId].action.kind === 'box' ? ENVS[envId].action.n : 0;
      for (let i = 0; i < Math.min(64, decoded.steps); i++) {
        const row = i * RECORD_FLOATS;
        if (nBox === 0) {
          expect(decoded.data[row]).toBe(doc.raw[row] ?? 0);
        } else {
          for (let c = 0; c < nBox; c++) {
            const want = Math.min(Math.max(doc.raw[row + c] ?? 0, -1), 1);
            expect(Math.abs((decoded.data[row + c] ?? 0) - want)).toBeLessThanOrEqual(1 / 127.5 + 1e-6);
          }
        }
      }
    }
  });
});
