#!/usr/bin/env bun

import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { inflateRawSync } from 'node:zlib';

import { ENV_IDS } from '@shared/envs';
import type { EnvId } from '@shared/envs';
import { parseRecord } from '@sim/record';


type Enc = 'u8' | 'i16' | 'dx32';
interface Field {
  idx: number;
  enc: Enc;
  scale: number;
  wrap: boolean;
}
interface Codec {
  envKind: number;
  category: 'grid' | 'classic' | 'cheetah';
  ctrlBox: number;
  fields: Field[];
  poseConst: Map<number, number>;
}

const u8 = (idx: number): Field => ({ idx, enc: 'u8', scale: 0, wrap: false });
const i16 = (idx: number, scale: number, wrap = false): Field => ({ idx, enc: 'i16', scale, wrap });
const dx32 = (idx: number, scale: number): Field => ({ idx, enc: 'dx32', scale, wrap: false });
const consts = (o: Record<number, number>): Map<number, number> =>
  new Map(Object.entries(o).map(([k, v]) => [Number(k), v]));

const grid = (envKind: number, idxs: number[], c: Record<number, number> = {}): Codec => ({
  envKind, category: 'grid', ctrlBox: 0, fields: idxs.map(u8), poseConst: consts(c),
});

const CODECS: Record<EnvId, Codec> = {
  'HalfCheetah-v4': {
    envKind: 0, category: 'cheetah', ctrlBox: 6,
    fields: [
      dx32(0, 0.01), i16(1, 1e-4),
      i16(2, 1e-4, true), i16(3, 1e-4, true), i16(4, 1e-4, true), i16(5, 1e-4, true),
      i16(6, 1e-4, true), i16(7, 1e-4, true), i16(8, 1e-4, true),
    ],
    poseConst: consts({}),
  },
  'FourRooms-cont-v0': grid(1, [0, 1, 2, 3]),
  'DeepSea-cont-v0': grid(2, [0, 1], { 2: 7, 3: 7 }),
  'KeyCorridor-cont-v0': grid(3, [0, 1, 2, 3, 4, 5, 6, 7]),
  'Acrobot-cont-v0': {
    envKind: 4, category: 'classic', ctrlBox: 0,
    fields: [i16(0, 1e-4, true), i16(1, 1e-4, true), i16(2, 1e-3), i16(3, 1e-3)],
    poseConst: consts({}),
  },
  'MountainCar-cont-v0': {
    envKind: 5, category: 'classic', ctrlBox: 0,
    fields: [i16(0, 1e-4), i16(1, 1e-5)],
    poseConst: consts({ 2: 0.5 }),
  },
  'CartPole-cont-v0': {
    envKind: 6, category: 'classic', ctrlBox: 0,
    fields: [i16(0, 1e-4), i16(1, 1e-3), i16(2, 1e-4, true), i16(3, 1e-3)],
    poseConst: consts({}),
  },
  'Pendulum-cont-v0': {
    envKind: 7, category: 'classic', ctrlBox: 1,
    fields: [i16(0, 1e-4, true), i16(1, 1e-3), i16(2, 1e-3)],
    poseConst: consts({}),
  },
  'Chain-cont-v0': grid(8, [0], { 1: 15 }),
  'LargeDeepSea-cont-v0': grid(9, [0, 1], { 2: 15, 3: 15 }),
  'MultiKeyCorridor-cont-v0': grid(10, [0, 1, 2, 3, 4, 5, 6, 7]),
  'MemoryChain-cont-v0': grid(11, [0, 2], { 1: 31 }),
  'ProceduralMaze-cont-v0': grid(12, [0, 1, 2, 3]),
  'HighDimMaze-cont-v0': grid(13, [0, 1, 2, 3, 4]),
  'POMaze-cont-v0': grid(14, [0, 1, 2, 3]),
  'DelayedChain-cont-v0': grid(15, [0, 2], { 1: 31 }),
};

const FILE_MAGIC = 0x524b414f;
const CHUNK_MAGIC = 0x524b4843;
const FILE_HEADER = 12;
const CHUNK_HEADER = 24;
const ROW_FLOATS = 16;
const ROW_BYTES = 64;


const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]!) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}


interface ChunkInfo {
  stepStart: number;
  stepCount: number;
  rawBytes: number;
  compBytes: number;
  escapes: number;
}

interface Decoded {
  envKind: number;
  steps: number;
  chunks: ChunkInfo[];
  data: Float32Array;
  escaped: Uint8Array;
  hdrBits: Uint8Array;
  escBytes: Map<number, Uint8Array>;
  qCols: Int32Array[];
  dCols: (Int32Array | null)[];
  problems: string[];
}

function readRecord(bytes: Uint8Array, codec: Codec): Decoded {
  const problems: string[] = [];
  if (bytes.length < FILE_HEADER) {
    throw new Error(`file shorter than the ${FILE_HEADER}-byte header (${bytes.length})`);
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = dv.getUint32(0, true);
  if (magic !== FILE_MAGIC) {
    problems.push(`file magic 0x${magic.toString(16)} != 0x${FILE_MAGIC.toString(16)}`);
  }
  const envKind = dv.getUint32(4, true);
  if (envKind !== codec.envKind) problems.push(`header envKind ${envKind} != ${codec.envKind}`);
  const headerSteps = dv.getUint32(8, true);

  const chunks: ChunkInfo[] = [];
  const payloads: Uint8Array[] = [];
  let off = FILE_HEADER;
  let steps = 0;
  while (off < bytes.length) {
    if (off + CHUNK_HEADER > bytes.length) {
      problems.push(`torn tail: ${bytes.length - off} bytes at ${off} < a ${CHUNK_HEADER}-byte chunk header`);
      break;
    }
    if (dv.getUint32(off, true) !== CHUNK_MAGIC) {
      problems.push(`chunk magic at ${off} is 0x${dv.getUint32(off, true).toString(16)}`);
      break;
    }
    const stepStart = dv.getUint32(off + 4, true);
    const stepCount = dv.getUint32(off + 8, true);
    const rawBytes = dv.getUint32(off + 12, true);
    const compBytes = dv.getUint32(off + 16, true);
    const crc = dv.getUint32(off + 20, true);
    if (off + CHUNK_HEADER + compBytes > bytes.length) {
      problems.push(`torn tail: chunk at ${off} wants ${compBytes} bytes, file has ${bytes.length - off - CHUNK_HEADER}`);
      break;
    }
    const comp = bytes.subarray(off + CHUNK_HEADER, off + CHUNK_HEADER + compBytes);
    const got = crc32(comp);
    if (got !== crc) {
      problems.push(`chunk at ${off}: crc32 ${got.toString(16)} != header ${crc.toString(16)}`);
      break;
    }
    let payload: Uint8Array;
    try {
      payload = new Uint8Array(inflateRawSync(comp));
    } catch (e) {
      problems.push(`chunk at ${off}: raw inflate failed (${(e as Error).message})`);
      break;
    }
    if (payload.length !== rawBytes) {
      problems.push(`chunk at ${off}: inflated ${payload.length} != rawBytes ${rawBytes}`);
      break;
    }
    if (stepStart !== steps) {
      problems.push(`chunk at ${off}: stepStart ${stepStart} != running total ${steps}`);
    }
    chunks.push({ stepStart, stepCount, rawBytes, compBytes, escapes: 0 });
    payloads.push(payload);
    steps += stepCount;
    off += CHUNK_HEADER + compBytes;
  }
  if (headerSteps !== steps) {
    problems.push(`header steps ${headerSteps} != decoded steps ${steps}`);
  }

  const data = new Float32Array(steps * ROW_FLOATS);
  const escaped = new Uint8Array(steps);
  const hdrBits = new Uint8Array(steps);
  const escBytes = new Map<number, Uint8Array>();
  const qCols = codec.fields.map(() => new Int32Array(steps));
  const dCols = codec.fields.map((f) => (f.enc === 'i16' ? new Int32Array(steps) : null));
  const nBox = codec.ctrlBox;
  const hasDx = codec.fields.some((f) => f.enc === 'dx32');
  for (let ci = 0; ci < chunks.length; ci++) {
    const chunk = chunks[ci]!;
    const payload = payloads[ci]!;
    const p = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    const n = chunk.stepCount;
    const baseOff = n * (1 + nBox);
    let cursor = baseOff + (hasDx ? 8 : 0);
    const colOff = codec.fields.map((f) => {
      const at = cursor;
      cursor += f.enc === 'u8' ? n : f.enc === 'i16' ? 2 * n : 4 * n;
      return at;
    });
    const fixed = cursor + 4;
    if (payload.length < fixed) {
      problems.push(`chunk ${ci}: payload ${payload.length} < fixed columns ${fixed}`);
      continue;
    }
    const escCount = p.getUint32(fixed - 4, true);
    const expect = fixed + escCount * (4 + ROW_BYTES);
    if (expect !== payload.length) {
      problems.push(
        `chunk ${ci}: escape arithmetic ${fixed} + ${escCount}*68 = ${expect} != rawBytes ${payload.length}`,
      );
      continue;
    }
    chunk.escapes = escCount;

    const escAt = new Map<number, number>();
    let prevIdx = -1;
    for (let k = 0; k < escCount; k++) {
      const at = fixed + k * (4 + ROW_BYTES);
      const rowIdx = p.getUint32(at, true);
      if (rowIdx >= n) problems.push(`chunk ${ci}: escape row ${rowIdx} >= stepCount ${n}`);
      if (rowIdx <= prevIdx) problems.push(`chunk ${ci}: escape rows not strictly increasing at ${k}`);
      prevIdx = rowIdx;
      escAt.set(rowIdx, at + 4);
    }

    let recon = hasDx ? p.getFloat64(baseOff, true) : 0;
    const qAcc = new Int32Array(codec.fields.length);
    for (let i = 0; i < n; i++) {
      const row = (chunk.stepStart + i) * ROW_FLOATS;
      const hdr = payload[i]!;
      hdrBits[chunk.stepStart + i] = hdr;
      const isEsc = (hdr & 0x20) !== 0;
      if ((hdr & 0x10) !== 0) data[row + 6] = 1;
      if (nBox === 0) {
        data[row] = hdr & 0x0f;
      } else {
        for (let c = 0; c < nBox; c++) data[row + c] = payload[n * (1 + c) + i]! / 127.5 - 1;
      }
      for (let j = 0; j < codec.fields.length; j++) {
        const f = codec.fields[j]!;
        const at = colOff[j]!;
        if (f.enc === 'u8') {
          const b = p.getUint8(at + i);
          qCols[j]![chunk.stepStart + i] = b;
          data[row + 7 + f.idx] = b;
        } else if (f.enc === 'i16') {
          const d = p.getInt16(at + 2 * i, true);
          const q = i === 0 ? d : ((qAcc[j]! + d) << 16) >> 16;
          qAcc[j] = q;
          qCols[j]![chunk.stepStart + i] = q;
          dCols[j]![chunk.stepStart + i] = d;
          data[row + 7 + f.idx] = q * f.scale;
        } else {
          const d = p.getInt32(at + 4 * i, true);
          recon += d * f.scale;
          qCols[j]![chunk.stepStart + i] = d;
          data[row + 7 + f.idx] = recon;
        }
      }
      for (const [idx, v] of codec.poseConst) data[row + 7 + idx] = v;
      if (isEsc) {
        escaped[chunk.stepStart + i] = 1;
        const at = escAt.get(i);
        if (at === undefined) {
          problems.push(`chunk ${ci}: row ${i} flags escape but is absent from the table`);
          continue;
        }
        escBytes.set(chunk.stepStart + i, payload.subarray(at, at + ROW_BYTES));
        for (let u = 0; u < ROW_FLOATS; u++) data[row + u] = p.getFloat32(at + 4 * u, true);
      } else if (escAt.has(i)) {
        problems.push(`chunk ${ci}: row ${i} is in the escape table but the hdr bit is clear`);
      }
    }
  }
  return { envKind, steps, chunks, data, escaped, hdrBits, escBytes, qCols, dCols, problems };
}


function wrap(v: number): number {
  return v - 2 * Math.PI * Math.floor((v + Math.PI) / (2 * Math.PI));
}

function sameBits(a: number, b: number): boolean {
  if (a === b) return Object.is(a, 0) === Object.is(b, 0) || (a !== 0 && b !== 0);
  return Number.isNaN(a) && Number.isNaN(b);
}

function compareRows(
  src: Float32Array,
  dec: Decoded,
  codec: Codec,
  steps: number,
  fail: (m: string) => void,
): void {
  const nBox = codec.ctrlBox;
  const carried = new Set<number>(codec.fields.map((f) => f.idx));
  for (const [idx] of codec.poseConst) carried.add(idx);
  const slack = (v: number) => 1e-9 + 1e-6 * Math.abs(v);
  let bad = 0;
  for (let i = 0; i < steps && bad < 8; i++) {
    const a = i * ROW_FLOATS;
    const push = (m: string) => {
      bad++;
      fail(`row ${i}: ${m}`);
    };
    if (dec.escaped[i]) {
      for (let u = 0; u < ROW_FLOATS; u++) {
        if (!sameBits(src[a + u]!, dec.data[a + u]!)) {
          push(`escaped row not byte-identical at float ${u} (${src[a + u]} vs ${dec.data[a + u]})`);
          break;
        }
      }
      continue;
    }
    if (src[a + 6] !== dec.data[a + 6]) push(`done ${src[a + 6]} != ${dec.data[a + 6]}`);
    if (nBox === 0) {
      if (src[a] !== dec.data[a]) push(`action index ${src[a]} != ${dec.data[a]}`);
    } else {
      const hq = 0.5 / 127.5;
      for (let c = 0; c < nBox; c++) {
        const raw = src[a + c]!;
        if (!(raw >= -1 && raw <= 1)) push(`src ctrl[${c}] = ${raw} is outside [-1,1]; the codec clamps it`);
        const ref = Math.min(Math.max(raw, -1), 1);
        const d = Math.abs(ref - dec.data[a + c]!);
        if (d > hq + slack(ref)) push(`ctrl[${c}] off by ${d} (> ${hq})`);
      }
    }
    for (let c = nBox === 0 ? 1 : nBox; c < 6; c++) {
      if (src[a + c] !== 0) push(`src ctrl[${c}] = ${src[a + c]} but the codec drops it`);
      if (dec.data[a + c] !== 0) push(`decoded ctrl[${c}] = ${dec.data[a + c]} (expected 0)`);
    }
    for (const f of codec.fields) {
      const want = src[a + 7 + f.idx]!;
      const got = dec.data[a + 7 + f.idx]!;
      if (f.enc === 'u8') {
        if (want !== got) push(`pose[${f.idx}] u8 ${want} != ${got}`);
        continue;
      }
      const ref = f.wrap ? wrap(want) : want;
      const d = Math.abs(ref - got);
      if (d > 0.5 * f.scale + slack(ref)) {
        push(`pose[${f.idx}] off by ${d} (> half quantum ${0.5 * f.scale})`);
      }
    }
    for (const [idx, v] of codec.poseConst) {
      if (src[a + 7 + idx] !== v) push(`src pose[${idx}] = ${src[a + 7 + idx]} != poseConst ${v}`);
      if (dec.data[a + 7 + idx] !== v) push(`decoded pose[${idx}] = ${dec.data[a + 7 + idx]} != ${v}`);
    }
    for (let idx = 0; idx < 9; idx++) {
      if (carried.has(idx)) continue;
      if (src[a + 7 + idx] !== 0) push(`src pose[${idx}] = ${src[a + 7 + idx]} but the codec drops it`);
      if (dec.data[a + 7 + idx] !== 0) push(`decoded pose[${idx}] = ${dec.data[a + 7 + idx]} (expected 0)`);
    }
  }
}

interface ZooRow {
  label: string;
  values: number[];
  mustEscape: boolean;
}

function healthyValues(codec: Codec): number[] {
  const r = new Array<number>(ROW_FLOATS).fill(0);
  if (codec.ctrlBox === 0) r[0] = 1;
  else for (let c = 0; c < codec.ctrlBox; c++) r[c] = 0.25 * (c % 3) - 0.25;
  for (const f of codec.fields) {
    r[7 + f.idx] = f.enc === 'u8' ? 2 : f.enc === 'dx32' ? 0 : HEALTHY_Q * f.scale;
  }
  for (const [idx, v] of codec.poseConst) r[7 + idx] = v;
  return r;
}

function negativeHalfTie(scale: number): [number, number] | null {
  for (let n = -32000; n > -32760; n--) {
    const v = Math.fround((n + 0.5) * scale);
    if (v / scale === n + 0.5) return [v, n];
  }
  return null;
}

const HEALTHY_Q = 12;
const WRAP_Q = -32760;

function escapeZoo(codec: Codec): ZooRow[] {
  const base = () => healthyValues(codec);
  const rows: ZooRow[] = [];
  const add = (label: string, mustEscape: boolean, mut: (v: number[]) => void) => {
    const values = base();
    mut(values);
    rows.push({ label, values, mustEscape });
  };

  add('done-one', false, (v) => { v[6] = 1; });
  add('done-fractional', true, (v) => { v[6] = 0.5; });

  if (codec.ctrlBox === 0) {
    add('action-16', true, (v) => { v[0] = 16; });
    add('action-fractional', true, (v) => { v[0] = 2.5; });
    add('action-negative', true, (v) => { v[0] = -1; });
  }

  const u8 = codec.fields.find((f) => f.enc === 'u8');
  if (u8) {
    add('u8-256', true, (v) => { v[7 + u8.idx] = 256; });
    add('u8-fractional', true, (v) => { v[7 + u8.idx] = 1.5; });
    add('u8-negative', true, (v) => { v[7 + u8.idx] = -1; });
  }

  const plain = codec.fields.find((f) => f.enc === 'i16' && !f.wrap);
  if (plain) {
    add('i16-overflow', true, (v) => { v[7 + plain.idx] = (32767 + 500) * plain.scale; });
    add('i16-underflow', true, (v) => { v[7 + plain.idx] = -(32767 + 500) * plain.scale; });
  }
  const wrapped = codec.fields.find((f) => f.enc === 'i16' && f.wrap);
  if (wrapped) add('i16-wrapped-safe', false, (v) => { v[7 + wrapped.idx] = 100; });
  if (plain) {
    const tie = negativeHalfTie(plain.scale);
    if (tie) add('i16-negative-tie', false, (v) => { v[7 + plain.idx] = tie[0]; });
    add('i16-delta-wrap', false, (v) => { v[7 + plain.idx] = WRAP_Q * plain.scale; });
  }

  const dx = codec.fields.find((f) => f.enc === 'dx32');
  if (dx) {
    const tie = negativeHalfTie(dx.scale);
    if (tie) add('dx32-negative-tie', false, (v) => { v[7 + dx.idx] = tie[0]; });
    add('dx32-overflow', true, (v) => { v[7 + dx.idx] = 3e7; });
    add('dx32-underflow', true, (v) => { v[7 + dx.idx] = -3e7; });
  }

  return rows;
}

const CLAMP_I16 = 32767;
const CLAMP_I32 = 2147483647;

function zooRows(codec: Codec): { data: Float32Array; mustEscape: boolean[]; labels: string[] } {
  const zoo = escapeZoo(codec);
  const dxTie = zoo.findIndex((z) => z.label === 'dx32-negative-tie');
  if (dxTie % 2 === 1) {
    zoo.splice(dxTie, 0, { label: 'filler', values: healthyValues(codec), mustEscape: false });
  }
  const n = zoo.length * 2;
  const data = new Float32Array(n * ROW_FLOATS);
  const mustEscape: boolean[] = [];
  const labels: string[] = [];
  const healthy = healthyValues(codec);
  for (let k = 0; k < zoo.length; k++) {
    const c = zoo[k]!;
    data.set(healthy, 2 * k * ROW_FLOATS);
    mustEscape.push(false);
    labels.push('healthy');
    data.set(c.values, (2 * k + 1) * ROW_FLOATS);
    mustEscape.push(c.mustEscape);
    labels.push(c.label);
  }
  return { data, mustEscape, labels };
}

interface TruncCase {
  label: string;
  at: number;
  mut?: (v: number[]) => void;
  bits?: [number, number][];
}

const TRUNC_ROWS = 9;

function truncationCases(codec: Codec): TruncCase[] {
  const first = codec.fields[0]!;
  return [
    { label: 'trunc-nan-ctrl0', at: 5, mut: (v) => { v[0] = Number.NaN; } },
    { label: 'trunc-posinf-pose', at: 4, mut: (v) => { v[7 + first.idx] = Number.POSITIVE_INFINITY; } },
    { label: 'trunc-neginf-pose', at: 7, mut: (v) => { v[7 + first.idx] = Number.NEGATIVE_INFINITY; } },
    { label: 'trunc-nan-bits-tail', at: 0, bits: [[15, 0x7f800001]] },
    { label: 'trunc-done-nan', at: 6, mut: (v) => { v[6] = Number.NaN; } },
  ];
}

function truncationRows(codec: Codec, t: TruncCase): Float32Array {
  const healthy = healthyValues(codec);
  const data = new Float32Array(TRUNC_ROWS * ROW_FLOATS);
  for (let i = 0; i < TRUNC_ROWS; i++) data.set(healthy, i * ROW_FLOATS);
  const v = healthy.slice();
  t.mut?.(v);
  data.set(v, t.at * ROW_FLOATS);
  if (t.bits) {
    const dv = new DataView(data.buffer);
    for (const [idx, bits] of t.bits) dv.setUint32((t.at * ROW_FLOATS + idx) * 4, bits, true);
  }
  return data;
}

const ESCAPE_PROBE_CHUNK_STEPS = 4;

function writeCases(path: string, envKind: number, data: Float32Array, chunkSteps: number): void {
  const rows = data.length / ROW_FLOATS;
  const buf = new Uint8Array(12 + data.byteLength);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, envKind, true);
  dv.setUint32(4, rows, true);
  dv.setUint32(8, chunkSteps, true);
  buf.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 12);
  writeFileSync(path, buf);
}

function runProbe(
  binary: string,
  work: string,
  tag: string,
  envKind: number,
  data: Float32Array,
): { bin: Uint8Array } | { error: string } {
  const casesPath = join(work, `${tag}.cases.bin`);
  const outDir = join(work, tag);
  writeCases(casesPath, envKind, data, ESCAPE_PROBE_CHUNK_STEPS);
  const run = spawnSync(binary, ['record-probe', casesPath, outDir], { encoding: 'utf8' });
  if (run.status !== 0) {
    return { error: `record-probe exited ${run.status}: ${(run.stderr || run.stdout || '').trim()}` };
  }
  const listing = readdirSync(outDir).sort();
  if (listing.length !== 1 || listing[0] !== 'actions.bin') {
    return { error: `out-dir holds [${listing.join(', ')}] — expected only actions.bin` };
  }
  const raw = readFileSync(join(outDir, 'actions.bin'));
  return { bin: new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength) };
}

async function crossCheckProduction(
  bin: Uint8Array,
  envId: EnvId,
  dec: Decoded,
  fail: (m: string) => void,
): Promise<void> {
  try {
    const rec = await parseRecord(bin, 0);
    if (rec.envId !== envId) fail(`record envId ${rec.envId} != ${envId}`);
    if (rec.steps !== dec.steps) fail(`record steps ${rec.steps} != ${dec.steps}`);
    const n = Math.min(rec.steps, dec.steps) * ROW_FLOATS;
    for (let u = 0; u < n; u++) {
      if (!sameBits(rec.data[u]!, dec.data[u]!)) {
        fail(`record float ${u} = ${rec.data[u]} != independent decode ${dec.data[u]}`);
        break;
      }
    }
  } catch (e) {
    fail(`record parseRecord threw: ${(e as Error).message}`);
  }
}

interface EnvResult {
  envId: EnvId;
  rows: number;
  escapes: number;
  truncs: number;
  failures: string[];
}

async function checkEnv(binary: string, work: string, envId: EnvId): Promise<EnvResult> {
  const codec = CODECS[envId];
  const failures: string[] = [];
  const fail = (m: string) => failures.push(m);
  const { data, mustEscape, labels } = zooRows(codec);
  const steps = mustEscape.length;
  let escapes = 0;
  let truncs = 0;

  const res = runProbe(binary, work, `${envId}-zoo`, codec.envKind, data);
  if ('error' in res) {
    fail(res.error);
    return { envId, rows: steps, escapes, truncs, failures };
  }
  const dec = readRecord(res.bin, codec);
  dec.problems.forEach(fail);
  if (dec.steps !== steps) fail(`walked ${dec.steps} steps != ${steps} written`);

  const rowBytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const doneBits = new Uint32Array(data.buffer, data.byteOffset, data.length);
  for (let i = 0; i < Math.min(steps, dec.steps); i++) {
    const esc = dec.escaped[i] === 1;
    if (mustEscape[i] !== esc) {
      fail(`row ${i} (${labels[i]}) ${mustEscape[i] ? 'did NOT escape' : 'escaped unexpectedly'}`);
    }
    const hdr = dec.hdrBits[i]!;
    if (((hdr >> 5) & 1) !== (esc ? 1 : 0)) fail(`row ${i}: hdr escape bit != escape table`);
    const wantDone = doneBits[i * ROW_FLOATS + 6] === 0x3f800000 ? 1 : 0;
    if (((hdr >> 4) & 1) !== wantDone) {
      fail(`row ${i} (${labels[i]}): hdr done bit ${(hdr >> 4) & 1} != ${wantDone}`);
    }
    if (!esc) continue;
    const got = dec.escBytes.get(i);
    if (got === undefined) {
      fail(`row ${i}: escaped but absent from the escape table`);
      continue;
    }
    const want = rowBytes.subarray(i * ROW_BYTES, (i + 1) * ROW_BYTES);
    for (let u = 0; u < ROW_BYTES; u++) {
      if (got[u] !== want[u]) {
        fail(`row ${i} (${labels[i]}): escape byte ${u} = ${got[u]} != ${want[u]}`);
        break;
      }
    }
  }
  compareRows(data, dec, codec, Math.min(steps, dec.steps), fail);

  const plainIdx = codec.fields.findIndex((f) => f.enc === 'i16' && !f.wrap);
  const dxIdx = codec.fields.findIndex((f) => f.enc === 'dx32');
  const plainQ = plainIdx >= 0 ? dec.qCols[plainIdx]! : null;
  const plainD = plainIdx >= 0 ? dec.dCols[plainIdx]! : null;
  const dxQ = dxIdx >= 0 ? dec.qCols[dxIdx]! : null;
  for (let i = 0; i < Math.min(steps, dec.steps); i++) {
    const label = labels[i]!;
    if (plainQ !== null && label === 'i16-overflow' && plainQ[i] !== CLAMP_I16) {
      fail(`row ${i} (${label}): i16 quantized ${plainQ[i]} != ${CLAMP_I16}`);
    }
    if (plainQ !== null && label === 'i16-underflow' && plainQ[i] !== -CLAMP_I16) {
      fail(`row ${i} (${label}): i16 quantized ${plainQ[i]} != ${-CLAMP_I16}`);
    }
    if (plainQ !== null && label === 'i16-negative-tie') {
      const tie = negativeHalfTie(codec.fields[plainIdx]!.scale);
      if (tie && plainQ[i] !== tie[1]) {
        fail(`row ${i} (${label}): i16 quantized ${plainQ[i]} != ${tie[1]} (half AWAY from zero)`);
      }
    }
    if (plainQ !== null && plainD !== null && label === 'i16-delta-wrap') {
      if (plainQ[i] !== WRAP_Q) fail(`row ${i} (${label}): i16 quantized ${plainQ[i]} != ${WRAP_Q}`);
      const wrapDelta = ((WRAP_Q - HEALTHY_Q) << 16) >> 16;
      if (plainD[i] !== wrapDelta) {
        fail(`row ${i} (${label}): stored delta ${plainD[i]} != wrapped int16 ${wrapDelta}`);
      }
    }
    if (dxQ !== null && label === 'dx32-overflow' && dxQ[i] !== CLAMP_I32) {
      fail(`row ${i} (${label}): dx32 delta ${dxQ[i]} != ${CLAMP_I32}`);
    }
    if (dxQ !== null && label === 'dx32-underflow' && dxQ[i] !== -CLAMP_I32) {
      fail(`row ${i} (${label}): dx32 delta ${dxQ[i]} != ${-CLAMP_I32}`);
    }
    if (dxQ !== null && label === 'dx32-negative-tie') {
      if (i % ESCAPE_PROBE_CHUNK_STEPS !== 1) {
        fail(`row ${i} (${label}) sits at chunk position ${i % ESCAPE_PROBE_CHUNK_STEPS}, not 1 — the zoo layout drifted`);
      } else {
        const tie = negativeHalfTie(codec.fields[dxIdx]!.scale);
        if (tie && dxQ[i] !== tie[1]) {
          fail(`row ${i} (${label}): dx32 delta ${dxQ[i]} != ${tie[1]} (half AWAY from zero)`);
        }
      }
    }
  }

  await crossCheckProduction(res.bin, envId, dec, fail);

  escapes = dec.chunks.reduce((s, c) => s + c.escapes, 0);
  if (escapes === 0) fail('esc = 0 — the encoder escape branch never executed');

  for (const t of truncationCases(codec)) {
    const tfail = (m: string) => fail(`${t.label}: ${m}`);
    const rows = truncationRows(codec, t);
    const tres = runProbe(binary, work, `${envId}-${t.label}`, codec.envKind, rows);
    if ('error' in tres) {
      tfail(tres.error);
      continue;
    }
    const tdec = readRecord(tres.bin, codec);
    tdec.problems.forEach(tfail);
    if (tdec.steps !== t.at) {
      tfail(`decoded ${tdec.steps} steps — the writer must stop before non-finite row ${t.at}`);
    }
    if (t.at === 0 && tres.bin.length !== FILE_HEADER) {
      tfail(`${tres.bin.length} bytes for zero steps, want the bare ${FILE_HEADER}-byte header`);
    }
    const tesc = tdec.chunks.reduce((s, c) => s + c.escapes, 0);
    if (tesc !== 0) tfail(`${tesc} escapes in an all-healthy prefix`);
    compareRows(rows, tdec, codec, Math.min(tdec.steps, t.at), tfail);
    await crossCheckProduction(tres.bin, envId, tdec, tfail);
    truncs++;
  }

  return { envId, rows: steps, escapes, truncs, failures };
}

async function escapeProbe(binary: string): Promise<number> {
  if (!existsSync(binary)) {
    console.error(`no OaK binary at ${binary} — build it with \`swift build -c release\``);
    return 1;
  }
  console.log(`binary  : ${binary}`);
  console.log('record  : @sim/record parseRecord — cross-checking the decoder');
  const work = mkdtempSync(join(tmpdir(), 'oak-record-escape-'));
  const results: EnvResult[] = [];
  try {
    for (const envId of ENV_IDS) {
      results.push(await checkEnv(binary, work, envId));
    }
  } finally {
    rmSync(work, { recursive: true, force: true });
  }

  const pad = (s: string, n: number) => s.padEnd(n);
  console.log('');
  console.log(pad('env', 26) + pad('rows', 7) + pad('esc', 7) + pad('trunc', 7) + 'status');
  let ok = true;
  let totalEsc = 0;
  for (const r of results) {
    totalEsc += r.escapes;
    if (r.failures.length) ok = false;
    console.log(
      pad(r.envId, 26) + pad(String(r.rows), 7) + pad(String(r.escapes), 7) +
        pad(String(r.truncs), 7) + (r.failures.length ? `FAIL (${r.failures.length})` : 'ok'),
    );
    for (const f of r.failures.slice(0, 6)) console.log(`    ! ${f}`);
    if (r.failures.length > 6) console.log(`    ! ... ${r.failures.length - 6} more`);
  }
  console.log('');
  console.log(`envs ${results.length} | ${totalEsc} escaped rows through the Swift encoder`);
  console.log(ok ? 'VERIFIED — the Swift encoder escape and truncation paths are exact.' : 'FAILED');
  return ok ? 0 : 1;
}


const DEFAULT_OAK_BINARY = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../..',
  '.build/release/OaK',
);

async function main(): Promise<number> {
  const argv = process.argv.slice(2);
  if (argv[0] === '--escape-probe') return escapeProbe(argv[1] ?? DEFAULT_OAK_BINARY);
  console.error('usage: bun scripts/verify-record.ts --escape-probe [<OaK binary>]');
  return 1;
}

process.exit(await main());
