
import { ENV_IDS, isEnvId } from '@shared/envs';
import type { EnvId } from '@shared/envs';
import type { RecordingMeta } from '@shared/types';
import type { RawRecording } from './types';

export const RECORD_FLOATS = 16;
export const RECORD_BYTES = RECORD_FLOATS * 4;
export const OFF_CTRL = 0;
export const CTRL_LEN = 6;
export const OFF_DONE = 6;
export const OFF_POSE = 7;
export const POSE_LEN = 9;

export const RECORD_VERSION = 4;
export const CODEC_VERSION = 1;
export const FILE_MAGIC = 0x344b414f;
export const CHUNK_MAGIC = 0x344b4843;
export const FILE_HEADER_BYTES = 16;
export const CHUNK_HEADER_BYTES = 32;
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

export const RECORD_V4_ENV_KINDS: readonly EnvId[] = [
  'HalfCheetah-v4',
  'FourRooms-cont-v0',
  'DeepSea-cont-v0',
  'KeyCorridor-cont-v0',
  'Acrobot-cont-v0',
  'MountainCar-cont-v0',
  'CartPole-cont-v0',
  'Pendulum-cont-v0',
  'Chain-cont-v0',
  'LargeDeepSea-cont-v0',
  'MultiKeyCorridor-cont-v0',
  'MemoryChain-cont-v0',
  'ProceduralMaze-cont-v0',
  'HighDimMaze-cont-v0',
  'POMaze-cont-v0',
  'DelayedChain-cont-v0',
];

export const RECORD_V4_CODECS: readonly CodecDescriptor[] = [
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
  return RECORD_V4_ENV_KINDS.indexOf(envId as EnvId);
}

export function codecForEnvKind(envKind: number): CodecDescriptor {
  const desc = RECORD_V4_CODECS[envKind];
  if (desc === undefined) throw new RecordParseError(`unknown envKind ${envKind}`);
  return desc;
}

export function codecForEnv(envId: string): CodecDescriptor {
  const kind = envKindOf(envId);
  if (kind < 0) throw new RecordParseError(`no v4 codec for envId ${JSON.stringify(envId)}`);
  return codecForEnvKind(kind);
}


function normalizeCodec(value: unknown): {
  category: string;
  ctrl: string;
  poseFields: string;
  poseConst: string;
} | null {
  if (value === null || typeof value !== 'object') return null;
  const o = value as Record<string, unknown>;

  const category = o['category'];
  if (typeof category !== 'string') return null;

  const rawCtrl = o['ctrl'];
  let ctrl: string;
  if (rawCtrl === 'index') {
    ctrl = 'index';
  } else if (rawCtrl !== null && typeof rawCtrl === 'object') {
    const c = rawCtrl as Record<string, unknown>;
    if (c['kind'] !== 'u8box' || typeof c['n'] !== 'number') return null;
    ctrl = `u8box:${c['n']}`;
  } else {
    return null;
  }

  const rawFields = o['poseFields'];
  if (!Array.isArray(rawFields)) return null;
  const parts: string[] = [];
  for (const entry of rawFields) {
    if (entry === null || typeof entry !== 'object') return null;
    const f = entry as Record<string, unknown>;
    const idx = f['idx'];
    const enc = f['enc'];
    if (typeof idx !== 'number' || typeof enc !== 'string') return null;
    const scale = f['scale'];
    const scaleText = scale === undefined ? '-' : typeof scale === 'number' ? String(scale) : '?';
    if (scaleText === '?') return null;
    parts.push(`${idx}:${enc}:${scaleText}:${f['wrap'] === true ? 'w' : '-'}`);
  }

  const rawConst = o['poseConst'];
  const constParts: string[] = [];
  if (rawConst !== undefined && rawConst !== null) {
    if (typeof rawConst !== 'object') return null;
    for (const [k, v] of Object.entries(rawConst as Record<string, unknown>)) {
      if (typeof v !== 'number') return null;
      constParts.push(`${k}=${String(v)}`);
    }
    constParts.sort();
  }

  return {
    category,
    ctrl,
    poseFields: parts.join('|'),
    poseConst: constParts.join(','),
  };
}

export function codecMatches(desc: CodecDescriptor, value: unknown): boolean {
  const a = normalizeCodec(desc);
  const b = normalizeCodec(value);
  if (a === null || b === null) return false;
  return (
    a.category === b.category &&
    a.ctrl === b.ctrl &&
    a.poseFields === b.poseFields &&
    a.poseConst === b.poseConst
  );
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


export interface FileHeaderV4 {
  magic: number;
  formatVersion: number;
  envKind: number;
  codecVersion: number;
}

export interface ChunkRecordV4 {
  index: number;
  stepStart: number;
  stepCount: number;
  rawBytes: number;
  compBytes: number;
  crc32: number;
  doneCount: number;
  rewardDelta: number;
  byteOffset: number;
  byteLength: number;
  payloadOffset: number;
}

export interface ChunkScanV4 {
  header: FileHeaderV4;
  chunks: ChunkRecordV4[];
  steps: number;
  truncatedAt: number | null;
  truncatedReason: string | null;
}

export interface ScanOptionsV4 {
  verifyCrc?: boolean;
}

function asWindow(buf: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (ArrayBuffer.isView(buf)) {
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  return new Uint8Array(buf);
}

export function readFileHeaderV4(buf: ArrayBuffer | ArrayBufferView): FileHeaderV4 {
  const bytes = asWindow(buf);
  if (bytes.byteLength < FILE_HEADER_BYTES) {
    throw new RecordParseError(
      `actions.bin is ${bytes.byteLength} bytes — shorter than the ${FILE_HEADER_BYTES}-byte v4 file header`,
    );
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const header: FileHeaderV4 = {
    magic: dv.getUint32(0, true),
    formatVersion: dv.getUint32(4, true),
    envKind: dv.getUint32(8, true),
    codecVersion: dv.getUint32(12, true),
  };
  if (header.magic !== FILE_MAGIC) {
    throw new RecordParseError(
      `bad v4 file magic 0x${header.magic.toString(16)} (expected 0x${FILE_MAGIC.toString(16)})`,
    );
  }
  if (header.formatVersion !== RECORD_VERSION) {
    throw new RecordParseError(`unsupported v4 formatVersion ${header.formatVersion}`);
  }
  if (header.codecVersion !== CODEC_VERSION) {
    throw new RecordParseError(
      `unsupported codecVersion ${header.codecVersion} (this build implements ${CODEC_VERSION})`,
    );
  }
  return header;
}

export function scanChunksV4(
  buf: ArrayBuffer | ArrayBufferView,
  opts: ScanOptionsV4 = {},
): ChunkScanV4 {
  const bytes = asWindow(buf);
  const header = readFileHeaderV4(bytes);
  const verifyCrc = opts.verifyCrc ?? true;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  const chunks: ChunkRecordV4[] = [];
  let steps = 0;
  let offset = FILE_HEADER_BYTES;
  let truncatedAt: number | null = null;
  let truncatedReason: string | null = null;

  const stop = (reason: string): void => {
    truncatedAt = offset;
    truncatedReason = reason;
  };

  for (;;) {
    if (offset === bytes.byteLength) break;
    if (offset + CHUNK_HEADER_BYTES > bytes.byteLength) {
      stop('short chunk header');
      break;
    }
    const magic = dv.getUint32(offset, true);
    if (magic !== CHUNK_MAGIC) {
      stop(`bad chunk magic 0x${magic.toString(16)}`);
      break;
    }
    const stepStart = dv.getUint32(offset + 4, true);
    const stepCount = dv.getUint32(offset + 8, true);
    const rawBytes = dv.getUint32(offset + 12, true);
    const compBytes = dv.getUint32(offset + 16, true);
    const crc = dv.getUint32(offset + 20, true);
    const doneCount = dv.getUint32(offset + 24, true);
    const rewardDelta = dv.getFloat32(offset + 28, true);

    if (stepCount === 0 || rawBytes === 0 || compBytes === 0) {
      stop('empty chunk');
      break;
    }
    if (stepStart !== steps) {
      stop(`chunk stepStart ${stepStart} does not continue at ${steps}`);
      break;
    }
    const payloadOffset = offset + CHUNK_HEADER_BYTES;
    if (payloadOffset + compBytes > bytes.byteLength) {
      stop('chunk payload runs past the buffer');
      break;
    }
    if (verifyCrc && crc32(bytes, payloadOffset, payloadOffset + compBytes) !== crc) {
      stop('chunk crc32 mismatch');
      break;
    }

    chunks.push({
      index: chunks.length,
      stepStart,
      stepCount,
      rawBytes,
      compBytes,
      crc32: crc,
      doneCount,
      rewardDelta,
      byteOffset: offset,
      byteLength: CHUNK_HEADER_BYTES + compBytes,
      payloadOffset,
    });
    steps += stepCount;
    offset = payloadOffset + compBytes;
  }

  return { header, chunks, steps, truncatedAt, truncatedReason };
}


export interface ChunkTableEntryV4 {
  stepStart: number;
  stepCount: number;
  byteOffset: number;
  byteLength: number;
}

export function chunkTableFromMeta(meta: Pick<RecordingMeta, 'chunks'>): ChunkTableEntryV4[] | null {
  const raw = meta.chunks;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const out: ChunkTableEntryV4[] = [];
  let steps = 0;
  let offset = FILE_HEADER_BYTES;
  for (const row of raw) {
    if (!Array.isArray(row) || row.length < 4) return null;
    const [stepStart, stepCount, byteOffset, byteLength] = row;
    if (
      typeof stepStart !== 'number' || !Number.isSafeInteger(stepStart) ||
      typeof stepCount !== 'number' || !Number.isSafeInteger(stepCount) ||
      typeof byteOffset !== 'number' || !Number.isSafeInteger(byteOffset) ||
      typeof byteLength !== 'number' || !Number.isSafeInteger(byteLength)
    ) {
      return null;
    }
    if (stepCount <= 0 || byteLength <= CHUNK_HEADER_BYTES) return null;
    if (stepStart !== steps || byteOffset !== offset) return null;
    out.push({ stepStart, stepCount, byteOffset, byteLength });
    steps += stepCount;
    offset = byteOffset + byteLength;
  }
  return out;
}

export interface PrefixPlanV4 {
  chunks: number;
  steps: number;
  bytes: number;
}

export function planPrefixV4(
  table: readonly ChunkTableEntryV4[],
  wantSteps: number,
): PrefixPlanV4 {
  if (table.length === 0 || !(wantSteps > 0)) return { chunks: 0, steps: 0, bytes: 0 };
  let steps = 0;
  let bytes = 0;
  let chunks = 0;
  for (const entry of table) {
    chunks++;
    steps += entry.stepCount;
    bytes = entry.byteOffset + entry.byteLength;
    if (steps >= wantSteps) break;
  }
  return { chunks, steps, bytes };
}

export function rangeHeaderForSteps(
  meta: Pick<RecordingMeta, 'chunks'>,
  wantSteps: number,
): string | null {
  const table = chunkTableFromMeta(meta);
  if (table === null) return null;
  const plan = planPrefixV4(table, wantSteps);
  if (plan.bytes <= 0) return null;
  return `bytes=0-${plan.bytes - 1}`;
}


async function inflateChunk(bytes: Uint8Array, chunk: ChunkRecordV4): Promise<Uint8Array> {
  const comp = bytes.slice(chunk.payloadOffset, chunk.payloadOffset + chunk.compBytes);
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(comp);
      controller.close();
    },
  });
  const inflated = new Uint8Array(
    await new Response(source.pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer(),
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
      for (let i = 0; i < stepCount; i++) {
        out[base + i * RECORD_FLOATS + slot] = dv.getInt16(col + i * 2, true) * scale;
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


interface PreparedV4 {
  bytes: Uint8Array;
  envId: EnvId;
  seed: number;
  desc: CodecDescriptor;
  scan: ChunkScanV4;
}

export function prepareLane(
  buf: ArrayBuffer | ArrayBufferView,
  meta: RecordingMeta,
  opts: ScanOptionsV4 = {},
): PreparedV4 {
  if (!isEnvId(meta.envId)) {
    throw new RecordParseError(`unknown envId ${JSON.stringify(meta.envId)}`);
  }
  if (!Number.isInteger(meta.seed) || meta.seed < 0) {
    throw new RecordParseError(`invalid seed ${String(meta.seed)}`);
  }
  if (meta.codecVersion !== undefined && meta.codecVersion !== CODEC_VERSION) {
    throw new RecordParseError(
      `meta codecVersion ${meta.codecVersion} (this build implements ${CODEC_VERSION})`,
    );
  }

  const bytes = asWindow(buf);
  const scan = scanChunksV4(bytes, opts);
  const headerEnv = RECORD_V4_ENV_KINDS[scan.header.envKind];
  if (headerEnv === undefined) {
    throw new RecordParseError(`file header envKind ${scan.header.envKind} is not a known env`);
  }
  if (headerEnv !== meta.envId) {
    throw new RecordParseError(
      `file header envKind ${scan.header.envKind} (${headerEnv}) contradicts meta envId ${meta.envId}`,
    );
  }

  const desc = codecForEnvKind(scan.header.envKind);
  if (meta.codec !== undefined && meta.codec !== null && !codecMatches(desc, meta.codec)) {
    throw new RecordParseError(
      `meta codec descriptor for ${meta.envId} does not match this build's codec table`,
    );
  }

  return { bytes, envId: meta.envId, seed: meta.seed, desc, scan };
}

export async function parseActionsBin(
  buf: ArrayBuffer | ArrayBufferView,
  meta: RecordingMeta,
): Promise<RawRecording> {
  const { bytes, envId, seed, desc, scan } = prepareLane(buf, meta);
  const data = new Float32Array(scan.steps * RECORD_FLOATS);

  let steps = 0;
  for (const chunk of scan.chunks) {
    let payload: Uint8Array;
    try {
      payload = await inflateChunk(bytes, chunk);
    } catch {
      break;
    }
    decodeChunkPayload(payload, desc, chunk.stepCount, data, steps * RECORD_FLOATS);
    steps += chunk.stepCount;
  }

  return {
    envId,
    seed,
    data: steps === scan.steps ? data : data.subarray(0, steps * RECORD_FLOATS),
    steps,
  };
}

export interface ChunkRowsV4 {
  chunk: ChunkRecordV4;
  stepStart: number;
  stepCount: number;
  data: Float32Array;
}

export type ChunkVisitorV4 = (rows: ChunkRowsV4) => boolean | void | Promise<boolean | void>;

export async function walkChunksV4(
  buf: ArrayBuffer | ArrayBufferView,
  meta: RecordingMeta,
  visit: ChunkVisitorV4,
): Promise<number> {
  const { bytes, desc, scan } = prepareLane(buf, meta);
  let scratch = new Float32Array(0);
  let steps = 0;

  for (const chunk of scan.chunks) {
    let payload: Uint8Array;
    try {
      payload = await inflateChunk(bytes, chunk);
    } catch {
      break;
    }
    const floats = chunk.stepCount * RECORD_FLOATS;
    if (scratch.length < floats) scratch = new Float32Array(floats);
    decodeChunkPayload(payload, desc, chunk.stepCount, scratch, 0);
    steps += chunk.stepCount;
    const stop = await visit({
      chunk,
      stepStart: chunk.stepStart,
      stepCount: chunk.stepCount,
      data: scratch.subarray(0, floats),
    });
    if (stop === false) break;
  }
  return steps;
}
