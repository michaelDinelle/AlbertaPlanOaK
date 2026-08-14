import { ENV_IDS } from '@shared/envs';
import type { EnvId } from '@shared/envs';
import type { RawRecording } from './types';

export const RECORD_FLOATS = 16;
export const RECORD_BYTES = RECORD_FLOATS * 4;
export const OFF_CTRL = 0;
export const CTRL_LEN = 6;
export const OFF_DONE = 6;
export const OFF_POSE = 7;
export const POSE_LEN = 9;

export const FILE_MAGIC = 0x524b414f;
export const CHUNK_MAGIC = 0x524b4843;
export const FILE_HEADER_BYTES = 12;
export const CHUNK_HEADER_BYTES = 24;
export const ESCAPE_ENTRY_BYTES = 4 + RECORD_FLOATS * 4;

const LITTLE_ENDIAN = (() => {
  const probe = new Uint8Array(4);
  new Uint32Array(probe.buffer)[0] = 1;
  return probe[0] === 1;
})();

export class RecordParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RecordParseError';
  }
}

export type PoseEncoding = 'u8' | 'i16' | 'dx32';

export interface PoseFieldSpec {
  readonly idx: number;
  readonly enc: PoseEncoding;
  readonly scale?: number;
  readonly wrap?: boolean;
}

export interface CtrlBoxSpec {
  readonly kind: 'u8box';
  readonly n: number;
}

export type CtrlSpec = 'index' | CtrlBoxSpec;

export interface CodecDescriptor {
  readonly category: 'grid' | 'classic' | 'cheetah';
  readonly ctrl: CtrlSpec;
  readonly poseFields: readonly PoseFieldSpec[];
  readonly poseConst?: Readonly<Record<string, number>>;
}

const u8 = (idx: number): PoseFieldSpec => ({ idx, enc: 'u8' });
const u8Range = (from: number, to: number): PoseFieldSpec[] => {
  const out: PoseFieldSpec[] = [];
  for (let i = from; i <= to; i++) out.push(u8(i));
  return out;
};
const i16 = (idx: number, scale: number): PoseFieldSpec => ({ idx, enc: 'i16', scale });
const i16w = (idx: number, scale: number): PoseFieldSpec => ({ idx, enc: 'i16', scale, wrap: true });

export const RECORD_ENV_KINDS: readonly EnvId[] = ENV_IDS;

export const RECORD_CODECS: readonly CodecDescriptor[] = [
  {
    category: 'cheetah',
    ctrl: { kind: 'u8box', n: 6 },
    poseFields: [
      { idx: 0, enc: 'dx32', scale: 0.01 },
      i16(1, 1e-4),
      i16w(2, 1e-4),
      i16w(3, 1e-4),
      i16w(4, 1e-4),
      i16w(5, 1e-4),
      i16w(6, 1e-4),
      i16w(7, 1e-4),
      i16w(8, 1e-4),
    ],
  },
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 3) },
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 1), poseConst: { 2: 7, 3: 7 } },
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 7) },
  {
    category: 'classic',
    ctrl: 'index',
    poseFields: [i16w(0, 1e-4), i16w(1, 1e-4), i16(2, 1e-3), i16(3, 1e-3)],
  },
  {
    category: 'classic',
    ctrl: 'index',
    poseFields: [i16(0, 1e-4), i16(1, 1e-5)],
    poseConst: { 2: 0.5 },
  },
  {
    category: 'classic',
    ctrl: 'index',
    poseFields: [i16(0, 1e-4), i16(1, 1e-3), i16w(2, 1e-4), i16(3, 1e-3)],
  },
  {
    category: 'classic',
    ctrl: { kind: 'u8box', n: 1 },
    poseFields: [i16w(0, 1e-4), i16(1, 1e-3), i16(2, 1e-3)],
  },
  { category: 'grid', ctrl: 'index', poseFields: [u8(0)], poseConst: { 1: 15 } },
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 1), poseConst: { 2: 15, 3: 15 } },
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 7) },
  { category: 'grid', ctrl: 'index', poseFields: [u8(0), u8(2)], poseConst: { 1: 31 } },
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 3) },
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 4) },
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 3) },
  { category: 'grid', ctrl: 'index', poseFields: [u8(0), u8(2)], poseConst: { 1: 31 } },
];

export function envKindOf(envId: string): number {
  return RECORD_ENV_KINDS.indexOf(envId as EnvId);
}

export function codecForEnvKind(envKind: number): CodecDescriptor {
  const desc = RECORD_CODECS[envKind];
  if (desc === undefined) throw new RecordParseError(`unknown envKind ${envKind}`);
  return desc;
}

export function codecForEnv(envId: string): CodecDescriptor {
  const kind = envKindOf(envId);
  if (kind < 0) throw new RecordParseError(`no codec for envId ${JSON.stringify(envId)}`);
  return codecForEnvKind(kind);
}

const TWO_PI = Math.PI * 2;

export function wrapAngle(v: number): number {
  return v - TWO_PI * Math.floor((v + Math.PI) / TWO_PI);
}

export function quantizeCtrlU8(v: number): number {
  const clamped = v < -1 ? -1 : v > 1 ? 1 : v;
  return roundHalfAwayFromZero((clamped + 1) * 127.5);
}

export function dequantizeCtrlU8(q: number): number {
  return q / 127.5 - 1;
}

export function roundHalfAwayFromZero(v: number): number {
  if (!Number.isFinite(v)) return v;
  return v < 0 ? -Math.round(-v) : Math.round(v);
}

export function crc32(bytes: Uint8Array, start = 0, end = bytes.length): number {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) {
    c = (CRC_TABLE[(c ^ (bytes[i] ?? 0)) & 0xff] ?? 0) ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function asWindow(buf: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (ArrayBuffer.isView(buf)) {
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  return new Uint8Array(buf);
}

export interface RecordHeader {
  envKind: number;
  envId: EnvId;
  steps: number;
  desc: CodecDescriptor;
}

export function readRecordHeader(buf: ArrayBuffer | ArrayBufferView): RecordHeader {
  const bytes = asWindow(buf);
  if (bytes.byteLength < FILE_HEADER_BYTES) {
    throw new RecordParseError(
      `actions.bin is ${bytes.byteLength} bytes — shorter than the ${FILE_HEADER_BYTES}-byte header`,
    );
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = dv.getUint32(0, true);
  if (magic !== FILE_MAGIC) {
    throw new RecordParseError(
      `bad recording magic 0x${magic.toString(16)} (expected 0x${FILE_MAGIC.toString(16)})`,
    );
  }
  const envKind = dv.getUint32(4, true);
  const envId = RECORD_ENV_KINDS[envKind];
  if (envId === undefined) {
    throw new RecordParseError(`recording envKind ${envKind} is not a known env`);
  }
  return { envKind, envId, steps: dv.getUint32(8, true), desc: codecForEnvKind(envKind) };
}

export interface RecordChunk {
  index: number;
  stepStart: number;
  stepCount: number;
  rawBytes: number;
  compBytes: number;
  crc32: number;
  byteOffset: number;
  byteLength: number;
  payloadOffset: number;
}

export interface ScanOptions {
  verifyCrc?: boolean;
}

export function scanNextChunk(
  bytes: Uint8Array,
  offset: number,
  expectStart: number,
  index: number,
): RecordChunk | null {
  if (offset + CHUNK_HEADER_BYTES > bytes.byteLength) return null;
  const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, CHUNK_HEADER_BYTES);
  if (dv.getUint32(0, true) !== CHUNK_MAGIC) return null;
  const stepStart = dv.getUint32(4, true);
  const stepCount = dv.getUint32(8, true);
  const rawBytes = dv.getUint32(12, true);
  const compBytes = dv.getUint32(16, true);
  if (stepCount === 0 || compBytes === 0) return null;
  if (stepStart !== expectStart) return null;
  if (offset + CHUNK_HEADER_BYTES + compBytes > bytes.byteLength) return null;
  return {
    index,
    stepStart,
    stepCount,
    rawBytes,
    compBytes,
    crc32: dv.getUint32(20, true),
    byteOffset: offset,
    byteLength: CHUNK_HEADER_BYTES + compBytes,
    payloadOffset: offset + CHUNK_HEADER_BYTES,
  };
}

export interface RecordScan {
  header: RecordHeader;
  desc: CodecDescriptor;
  chunks: RecordChunk[];
  steps: number;
}

export function scanRecord(buf: ArrayBuffer | ArrayBufferView, opts: ScanOptions = {}): RecordScan {
  const bytes = asWindow(buf);
  const header = readRecordHeader(bytes);
  const verify = opts.verifyCrc !== false;
  const chunks: RecordChunk[] = [];
  let offset = FILE_HEADER_BYTES;
  let steps = 0;
  for (;;) {
    const chunk = scanNextChunk(bytes, offset, steps, chunks.length);
    if (chunk === null) break;
    if (verify) {
      const crc = crc32(bytes, chunk.payloadOffset, chunk.payloadOffset + chunk.compBytes);
      if (crc !== chunk.crc32) break;
    }
    chunks.push(chunk);
    steps += chunk.stepCount;
    offset += chunk.byteLength;
  }
  return { header, desc: header.desc, chunks, steps };
}

export async function inflateChunk(bytes: Uint8Array, chunk: RecordChunk): Promise<Uint8Array> {
  const comp = bytes.slice(chunk.payloadOffset, chunk.payloadOffset + chunk.compBytes);
  const source = new ReadableStream<Uint8Array<ArrayBuffer>>({
    start(controller) {
      controller.enqueue(comp as Uint8Array<ArrayBuffer>);
      controller.close();
    },
  });
  const inflate = new DecompressionStream('deflate-raw') as unknown as ReadableWritablePair<
    Uint8Array<ArrayBuffer>,
    Uint8Array<ArrayBuffer>
  >;
  const inflated = new Uint8Array(
    await new Response(source.pipeThrough(inflate)).arrayBuffer(),
  );
  if (inflated.byteLength !== chunk.rawBytes) {
    throw new RecordParseError(
      `chunk ${chunk.index} inflated to ${inflated.byteLength} bytes, header claims ${chunk.rawBytes}`,
    );
  }
  return inflated;
}

export function decodeChunkPayload(
  payload: Uint8Array,
  desc: CodecDescriptor,
  stepCount: number,
  out: Float32Array,
  base: number,
): void {
  const limit = payload.byteLength;
  const dv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  let off = 0;
  const take = (n: number, what: string): number => {
    if (off + n > limit) {
      throw new RecordParseError(
        `chunk payload is ${limit} bytes — too short for ${what} at offset ${off} (+${n})`,
      );
    }
    const at = off;
    off += n;
    return at;
  };

  const hdrOff = take(stepCount, 'the hdr column');

  const ctrlCount = desc.ctrl === 'index' ? 0 : desc.ctrl.n;
  const ctrlOff = take(ctrlCount * stepCount, 'the ctrl columns');

  let chunkBaseX = 0;
  if (desc.category === 'cheetah') {
    chunkBaseX = dv.getFloat64(take(8, 'chunkBaseX'), true);
  }

  const poseOff: number[] = [];
  for (const field of desc.poseFields) {
    const width = field.enc === 'u8' ? 1 : field.enc === 'i16' ? 2 : 4;
    poseOff.push(take(width * stepCount, `pose column ${field.idx}`));
  }

  const escapeCount = dv.getUint32(take(4, 'the escape count'), true);
  const escapeOff = take(escapeCount * ESCAPE_ENTRY_BYTES, 'the escape table');

  const floats = stepCount * RECORD_FLOATS;
  out.fill(0, base, base + floats);

  const indexCtrl = desc.ctrl === 'index';
  let hdrEscapes = 0;
  for (let i = 0; i < stepCount; i++) {
    const h = payload[hdrOff + i] ?? 0;
    const row = base + i * RECORD_FLOATS;
    if ((h & 0x10) !== 0) out[row + 6] = 1;
    if ((h & 0x20) !== 0) hdrEscapes++;
    if (indexCtrl) out[row] = h & 0x0f;
  }
  if (hdrEscapes !== escapeCount) {
    throw new RecordParseError(
      `chunk escape bits (${hdrEscapes}) disagree with the escape table (${escapeCount})`,
    );
  }

  for (let c = 0; c < ctrlCount; c++) {
    const col = ctrlOff + c * stepCount;
    for (let i = 0; i < stepCount; i++) {
      out[base + i * RECORD_FLOATS + c] = dequantizeCtrlU8(payload[col + i] ?? 0);
    }
  }

  if (desc.poseConst !== undefined) {
    for (const [key, value] of Object.entries(desc.poseConst)) {
      const idx = Number(key);
      for (let i = 0; i < stepCount; i++) out[base + i * RECORD_FLOATS + 7 + idx] = value;
    }
  }

  for (let k = 0; k < desc.poseFields.length; k++) {
    const field = desc.poseFields[k];
    if (field === undefined) continue;
    const col = poseOff[k] ?? 0;
    const slot = 7 + field.idx;
    if (field.enc === 'u8') {
      for (let i = 0; i < stepCount; i++) {
        out[base + i * RECORD_FLOATS + slot] = payload[col + i] ?? 0;
      }
    } else if (field.enc === 'i16') {
      const scale = field.scale ?? 1;
      let q = 0;
      for (let i = 0; i < stepCount; i++) {
        const d = dv.getInt16(col + i * 2, true);
        q = i === 0 ? d : ((q + d + 0x8000) & 0xffff) - 0x8000;
        out[base + i * RECORD_FLOATS + slot] = q * scale;
      }
    } else {
      const scale = field.scale ?? 1;
      let recon = chunkBaseX;
      for (let i = 0; i < stepCount; i++) {
        recon += dv.getInt32(col + i * 4, true) * scale;
        out[base + i * RECORD_FLOATS + slot] = recon;
      }
    }
  }

  if (escapeCount > 0) {
    const outBytes = LITTLE_ENDIAN
      ? new Uint8Array(out.buffer, out.byteOffset, out.byteLength)
      : null;
    for (let e = 0; e < escapeCount; e++) {
      const at = escapeOff + e * ESCAPE_ENTRY_BYTES;
      const rowIdx = dv.getUint32(at, true);
      if (rowIdx >= stepCount) {
        throw new RecordParseError(`escape row ${rowIdx} is outside a ${stepCount}-step chunk`);
      }
      if ((payload[hdrOff + rowIdx]! & 0x20) === 0) {
        throw new RecordParseError(`escape row ${rowIdx} has no escape bit in the hdr column`);
      }
      const row = base + rowIdx * RECORD_FLOATS;
      if (outBytes !== null) {
        outBytes.set(payload.subarray(at + 4, at + ESCAPE_ENTRY_BYTES), row * 4);
      } else {
        for (let j = 0; j < RECORD_FLOATS; j++) {
          out[row + j] = dv.getFloat32(at + 4 + j * 4, true);
        }
      }
    }
  }
}

export async function parseRecord(
  buf: ArrayBuffer | ArrayBufferView,
  seed: number,
): Promise<RawRecording> {
  const bytes = asWindow(buf);
  const scan = scanRecord(bytes);
  const data = new Float32Array(scan.steps * RECORD_FLOATS);
  let steps = 0;
  for (const chunk of scan.chunks) {
    const payload = await inflateChunk(bytes, chunk);
    decodeChunkPayload(payload, scan.desc, chunk.stepCount, data, steps * RECORD_FLOATS);
    steps += chunk.stepCount;
  }
  return { envId: scan.header.envId, seed, data, steps };
}
