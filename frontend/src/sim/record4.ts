// record-v4 `actions.bin` decoder — the chunked, quantized, DEFLATEd lane codec.
//
// THE SPEC IS `src/sim/RECORD_V4_SPEC.md` AND IT IS FROZEN LAW. This module
// implements that document and nothing else; the Swift encoder (`Sources/Main.swift`)
// implements the same document, and the Tier-S v3/v4 A/B proves they agree.
//
// What v4 buys: ~22× smaller lanes. What it must NOT cost: a single semantic
// difference downstream. So the decoder's output is the EXACT v3 `RawRecording`
// row shape — flat LE f32[n][16] = ctrl[6], done, pose[9] — which means every env
// decoder, the MultiKeyCorridor bitwise replay, the DeepSea event simulation and
// the damaged-row semantics in `decode.ts` all run unchanged and unaware.
//
// Two readers, one decode core:
//   * `parseActionsBinV4`  — materializes the whole lane (theater / harness).
//   * `walkChunksV4`       — streams chunk by chunk, never holding the lane
//                            (aggregate passes over many lanes at once).
//
// Torn tails are DATA, not errors: a killed run leaves a half-written chunk, and a
// byte-prefix Range request deliberately cuts the file short. The chunk scan drops
// the first invalid chunk and everything after it, and the readers decode the
// valid prefix — exactly as the spec prescribes.

import { ENV_IDS, isEnvId } from '@shared/envs';
import type { EnvId } from '@shared/envs';
import type { RecordingMeta } from '@shared/types';
import { RECORD_FLOATS, RecordParseError } from './record';
import type { RawRecording } from './types';

// ---------------------------------------------------------------------------
// constants (spec: "File layout")
// ---------------------------------------------------------------------------

/** `meta.recordVersion` this module decodes */
export const RECORD_V4_VERSION = 4;
/** the frozen codec revision; any semantic change bumps it */
export const CODEC_VERSION = 1;
/** u32 file magic — "OAK4" (0x344B414F LE = O,A,K,4) */
export const FILE_MAGIC = 0x344b414f;
/** u32 chunk magic — "CHK4" (0x344B4843 LE = C,H,K,4) */
export const CHUNK_MAGIC = 0x344b4843;
/** magic + formatVersion + envKind + codecVersion */
export const FILE_HEADER_BYTES = 16;
/** magic + stepStart + stepCount + rawBytes + compBytes + crc32 + doneCount + rewardDelta */
export const CHUNK_HEADER_BYTES = 32;
/** escape-table entry: u32 rowIdxInChunk + 64 verbatim v3 bytes */
export const ESCAPE_ENTRY_BYTES = 4 + RECORD_FLOATS * 4;

const LITTLE_ENDIAN = (() => {
  const probe = new Uint8Array(4);
  new Uint32Array(probe.buffer)[0] = 1;
  return probe[0] === 1;
})();

/**
 * Subclasses the v3 error so one `catch (e) { e instanceof RecordParseError }`
 * covers both record versions.
 */
export class Record4ParseError extends RecordParseError {
  constructor(message: string) {
    super(message);
    this.name = 'Record4ParseError';
  }
}

// ---------------------------------------------------------------------------
// the static codec table (spec: "Static codec table")
// ---------------------------------------------------------------------------

export type PoseEncoding = 'u8' | 'i16' | 'dx32';

export interface PoseFieldSpec {
  /** index into pose[9] */
  readonly idx: number;
  readonly enc: PoseEncoding;
  /** quantization step; absent for `u8` */
  readonly scale?: number;
  /** wrap into [−π, π) before quantizing (encoder side) */
  readonly wrap?: boolean;
}

export interface CtrlBoxSpec {
  readonly kind: 'u8box';
  /** number of u8 ctrl columns (cheetah 6, pendulum 1) */
  readonly n: number;
}

/** `'index'` = the action index rides the hdr byte's low nibble. */
export type CtrlSpec = 'index' | CtrlBoxSpec;

export interface CodecDescriptor {
  readonly category: 'grid' | 'classic' | 'cheetah';
  readonly ctrl: CtrlSpec;
  readonly poseFields: readonly PoseFieldSpec[];
  /** pose indices filled with a fixed value on decode, keyed by index-as-string */
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

/**
 * envKind → envId. Mirrors `OakFleetLane.envKind` (Sources/OakKernel.metal, the
 * `oak_stage_fleet_env` switch) and the file header's `envKind` field. It is
 * ASSERTED equal to `ENV_IDS` in the tests rather than aliased to it: the two
 * orderings agree today, and the assertion is what keeps a future insertion into
 * the display list from silently re-labelling every recorded lane.
 */
export const RECORD_V4_ENV_KINDS: readonly EnvId[] = [
  'HalfCheetah-v4',        // 0
  'FourRooms-cont-v0',     // 1
  'DeepSea-cont-v0',       // 2
  'KeyCorridor-cont-v0',   // 3
  'Acrobot-cont-v0',       // 4
  'MountainCar-cont-v0',   // 5
  'CartPole-cont-v0',      // 6
  'Pendulum-cont-v0',      // 7
  'Chain-cont-v0',         // 8
  'LargeDeepSea-cont-v0',  // 9
  'MultiKeyCorridor-cont-v0', // 10
  'MemoryChain-cont-v0',   // 11
  'ProceduralMaze-cont-v0', // 12
  'HighDimMaze-cont-v0',   // 13
  'POMaze-cont-v0',        // 14
  'DelayedChain-cont-v0',  // 15
];

/** The frozen descriptor table, indexed by envKind. */
export const RECORD_V4_CODECS: readonly CodecDescriptor[] = [
  // 0 HalfCheetah — 6 ctrl columns, rootx as a chained delta, everything else i16
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
  // 1 FourRooms — x, y, gx, gy
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 3) },
  // 2 DeepSea — r, c (+ constant 7×7 extent)
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 1), poseConst: { 2: 7, 3: 7 } },
  // 3 KeyCorridor
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 7) },
  // 4 Acrobot — two wrapped joint angles + two velocities
  {
    category: 'classic',
    ctrl: 'index',
    poseFields: [i16w(0, 1e-4), i16w(1, 1e-4), i16(2, 1e-3), i16(3, 1e-3)],
  },
  // 5 MountainCar — position + (fine) velocity, constant goal marker
  {
    category: 'classic',
    ctrl: 'index',
    poseFields: [i16(0, 1e-4), i16(1, 1e-5)],
    poseConst: { 2: 0.5 },
  },
  // 6 CartPole — x, xdot, theta (wrapped), thetadot
  {
    category: 'classic',
    ctrl: 'index',
    poseFields: [i16(0, 1e-4), i16(1, 1e-3), i16w(2, 1e-4), i16(3, 1e-3)],
  },
  // 7 Pendulum — theta (wrapped), thetadot, applied torque; one u8 ctrl column
  {
    category: 'classic',
    ctrl: { kind: 'u8box', n: 1 },
    poseFields: [i16w(0, 1e-4), i16(1, 1e-3), i16(2, 1e-3)],
  },
  // 8 Chain — position, constant length
  { category: 'grid', ctrl: 'index', poseFields: [u8(0)], poseConst: { 1: 15 } },
  // 9 LargeDeepSea
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 1), poseConst: { 2: 15, 3: 15 } },
  // 10 MultiKeyCorridor
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 7) },
  // 11 MemoryChain — position + cue, constant length
  { category: 'grid', ctrl: 'index', poseFields: [u8(0), u8(2)], poseConst: { 1: 31 } },
  // 12 ProceduralMaze
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 3) },
  // 13 HighDimMaze
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 4) },
  // 14 POMaze
  { category: 'grid', ctrl: 'index', poseFields: u8Range(0, 3) },
  // 15 DelayedChain
  { category: 'grid', ctrl: 'index', poseFields: [u8(0), u8(2)], poseConst: { 1: 31 } },
];

/** envKind for an env id, or −1. */
export function envKindOf(envId: string): number {
  return RECORD_V4_ENV_KINDS.indexOf(envId as EnvId);
}

/** The frozen descriptor for an envKind. Throws for an unknown kind. */
export function codecForEnvKind(envKind: number): CodecDescriptor {
  const desc = RECORD_V4_CODECS[envKind];
  if (desc === undefined) throw new Record4ParseError(`unknown envKind ${envKind}`);
  return desc;
}

/** The frozen descriptor for an env id. Throws for an unknown env. */
export function codecForEnv(envId: string): CodecDescriptor {
  const kind = envKindOf(envId);
  if (kind < 0) throw new Record4ParseError(`no v4 codec for envId ${JSON.stringify(envId)}`);
  return codecForEnvKind(kind);
}

// ---------------------------------------------------------------------------
// descriptor comparison — meta.json carries the descriptor, we carry the table
// ---------------------------------------------------------------------------

/**
 * Fold a descriptor (ours, or a `meta.codec` blob of unknown shape) into four
 * comparable strings.
 *
 * Deliberately tolerant of the two places the spec's JSON leaves slack — an
 * absent `poseConst` versus `{}`, and key ORDER inside `poseConst` — because
 * neither changes a decoded byte. Everything that DOES change a decoded byte
 * (category, ctrl shape, the ordered pose columns, each scale and wrap flag,
 * every constant) is compared exactly.
 */
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

/** True when `value` describes the same codec as `desc`. */
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

// ---------------------------------------------------------------------------
// shared numeric contract (spec: "Payload")
// ---------------------------------------------------------------------------

const TWO_PI = Math.PI * 2;

/**
 * The spec's wrap into [−π, π): `w = v − 2π·floor((v + π) / (2π))`.
 * Floor-based on purpose — `floor` agrees bit for bit between Swift and JS,
 * where `%`/`fmod` and `trunc` do not for negative inputs.
 *
 * ENCODER-side only: the decoder just multiplies the stored i16 by its scale.
 * Exported so the reference encoder and any cross-language verifier share ONE
 * implementation instead of two that drift.
 */
export function wrapAngle(v: number): number {
  return v - TWO_PI * Math.floor((v + Math.PI) / TWO_PI);
}

/** Box ctrl quantization: `q = round((clamp(v, −1, 1) + 1) · 127.5)`. */
export function quantizeCtrlU8(v: number): number {
  const clamped = v < -1 ? -1 : v > 1 ? 1 : v;
  return roundHalfAwayFromZero((clamped + 1) * 127.5);
}

/** Box ctrl dequantization: `v = q / 127.5 − 1`. */
export function dequantizeCtrlU8(q: number): number {
  return q / 127.5 - 1;
}

/**
 * The spec's `round(...)`: half away from zero, matching C/Swift `round()`.
 * NOT `Math.round`, which is half toward +∞ and therefore disagrees with the
 * Swift encoder on every exact −x.5 tie.
 */
export function roundHalfAwayFromZero(v: number): number {
  if (!Number.isFinite(v)) return v;
  return v < 0 ? -Math.round(-v) : Math.round(v);
}

/** zlib/IEEE CRC-32 over `bytes[start, end)`. */
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

// ---------------------------------------------------------------------------
// chunk scan (spec: "A torn tail (killed run) is detected by …")
// ---------------------------------------------------------------------------

export interface FileHeaderV4 {
  magic: number;
  formatVersion: number;
  envKind: number;
  codecVersion: number;
}

export interface ChunkRecordV4 {
  /** position in the file's chunk sequence */
  index: number;
  stepStart: number;
  stepCount: number;
  rawBytes: number;
  compBytes: number;
  crc32: number;
  doneCount: number;
  /** lane rewardSum delta over this chunk; NaN when the host had no snapshot */
  rewardDelta: number;
  /** offset of the chunk HEADER from file start */
  byteOffset: number;
  /** CHUNK_HEADER_BYTES + compBytes */
  byteLength: number;
  /** offset of the compressed payload from file start */
  payloadOffset: number;
}

export interface ChunkScanV4 {
  header: FileHeaderV4;
  /** valid chunks, in file order and step-contiguous from 0 */
  chunks: ChunkRecordV4[];
  /** Σ stepCount over `chunks` */
  steps: number;
  /** byte offset the scan stopped at, when it stopped before the end */
  truncatedAt: number | null;
  /** why it stopped; null when the file ended cleanly */
  truncatedReason: string | null;
}

export interface ScanOptionsV4 {
  /**
   * Verify each chunk's crc32 over its compressed payload (default true). The
   * readers always verify; a pure index pass over a file whose bytes are not all
   * present can turn it off.
   */
  verifyCrc?: boolean;
}

/** Resolve any accepted buffer form to the window the caller actually meant. */
function asWindow(buf: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (ArrayBuffer.isView(buf)) {
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  return new Uint8Array(buf);
}

/** Read the 16-byte file header. Throws when it is absent or not ours. */
export function readFileHeaderV4(buf: ArrayBuffer | ArrayBufferView): FileHeaderV4 {
  const bytes = asWindow(buf);
  if (bytes.byteLength < FILE_HEADER_BYTES) {
    throw new Record4ParseError(
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
    throw new Record4ParseError(
      `bad v4 file magic 0x${header.magic.toString(16)} (expected 0x${FILE_MAGIC.toString(16)})`,
    );
  }
  if (header.formatVersion !== RECORD_V4_VERSION) {
    throw new Record4ParseError(`unsupported v4 formatVersion ${header.formatVersion}`);
  }
  if (header.codecVersion !== CODEC_VERSION) {
    throw new Record4ParseError(
      `unsupported codecVersion ${header.codecVersion} (this build implements ${CODEC_VERSION})`,
    );
  }
  return header;
}

/**
 * Walk the chunk headers, keeping the longest valid prefix.
 *
 * A chunk is dropped — along with every chunk after it — on a short header, a
 * foreign magic, a payload that runs past the buffer, a crc mismatch, a
 * non-advancing size, or a stepStart that does not continue the sequence. That
 * covers both real damage and the deliberate cut of a byte-prefix Range request.
 */
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
      // A zero-size field cannot advance the scan; treat it as the torn edge.
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

// ---------------------------------------------------------------------------
// the meta.json chunk table — a HINT, used for range planning only
// ---------------------------------------------------------------------------

export interface ChunkTableEntryV4 {
  stepStart: number;
  stepCount: number;
  byteOffset: number;
  byteLength: number;
}

/**
 * `meta.chunks` as a validated table, or null when it is absent or incoherent.
 *
 * Null is not a failure — it is the spec's own fallback ("readers of in-flight or
 * torn files fall back to a sequential header scan"), which for the loader means
 * "fetch the whole file and let the scan sort it out".
 */
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
  /** whole chunks to fetch */
  chunks: number;
  /** steps those chunks decode to (≥ wantSteps unless the lane is shorter) */
  steps: number;
  /** bytes from file start, header included; 0 when nothing can be planned */
  bytes: number;
}

/**
 * Fewest WHOLE chunks that cover `[0, wantSteps)`.
 *
 * Whole chunks because a chunk is the unit of everything: one DEFLATE stream,
 * one crc, one dx32 reconstruction chain. Half a chunk decodes to nothing.
 */
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

/**
 * The `Range` header value for the first `wantSteps` steps, or null for
 * "ask for the whole file" (v3 lanes, or a v4 lane with no usable chunk table).
 */
export function rangeHeaderForSteps(
  meta: Pick<RecordingMeta, 'recordVersion' | 'chunks'>,
  wantSteps: number,
): string | null {
  if (meta.recordVersion !== RECORD_V4_VERSION) return null;
  const table = chunkTableFromMeta(meta);
  if (table === null) return null;
  const plan = planPrefixV4(table, wantSteps);
  if (plan.bytes <= 0) return null;
  return `bytes=0-${plan.bytes - 1}`;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

/** raw-DEFLATE inflate of one chunk payload, checked against `rawBytes`. */
async function inflateChunk(bytes: Uint8Array, chunk: ChunkRecordV4): Promise<Uint8Array> {
  // `.slice()` detaches the payload from the lane buffer before it enters the
  // stream: a stream implementation is free to hold, transfer or neuter what it
  // is handed, and the lane buffer must survive every chunk.
  const comp = bytes.slice(chunk.payloadOffset, chunk.payloadOffset + chunk.compBytes);
  // Typed as BufferSource, which is what `DecompressionStream.writable` accepts.
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
    throw new Record4ParseError(
      `chunk ${chunk.index} inflated to ${inflated.byteLength} bytes, header claims ${chunk.rawBytes}`,
    );
  }
  return inflated;
}

/**
 * Decode one inflated chunk payload into `out` at float offset `base`,
 * in the v3 row shape.
 *
 * Column order is the spec's, exactly: hdr, Box ctrl columns, cheetah
 * `chunkBaseX`, the descriptor's pose columns, then the escape table. Escaped
 * rows still carry a value in every quantized column (that is what keeps the
 * dx32 chain deterministic on both sides) and are overwritten wholesale at the
 * end from their verbatim 64 bytes.
 */
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
      throw new Record4ParseError(
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

  // ---- rows ----
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
    throw new Record4ParseError(
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
      // dx32: the chained reconstruction, accumulated in f64 exactly as the
      // encoder did, then stored to f32 (the spec's `Math.fround` semantics).
      const scale = field.scale ?? 1;
      let recon = chunkBaseX;
      for (let i = 0; i < stepCount; i++) {
        recon += dv.getInt32(col + i * 4, true) * scale;
        out[base + i * RECORD_FLOATS + slot] = recon;
      }
    }
  }

  // ---- escapes: the whole row, verbatim ----
  if (escapeCount > 0) {
    const outBytes = LITTLE_ENDIAN
      ? new Uint8Array(out.buffer, out.byteOffset, out.byteLength)
      : null;
    for (let e = 0; e < escapeCount; e++) {
      const at = escapeOff + e * ESCAPE_ENTRY_BYTES;
      const rowIdx = dv.getUint32(at, true);
      if (rowIdx >= stepCount) {
        throw new Record4ParseError(`escape row ${rowIdx} is outside a ${stepCount}-step chunk`);
      }
      if ((payload[hdrOff + rowIdx]! & 0x20) === 0) {
        throw new Record4ParseError(`escape row ${rowIdx} has no escape bit in the hdr column`);
      }
      const row = base + rowIdx * RECORD_FLOATS;
      if (outBytes !== null) {
        // Byte copy, not 16 float reads: a NaN payload survives the round trip
        // exactly as it does in the v3 zero-copy view.
        outBytes.set(payload.subarray(at + 4, at + ESCAPE_ENTRY_BYTES), row * 4);
      } else {
        for (let j = 0; j < RECORD_FLOATS; j++) {
          out[row + j] = dv.getFloat32(at + 4 + j * 4, true);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// readers
// ---------------------------------------------------------------------------

interface PreparedV4 {
  bytes: Uint8Array;
  envId: EnvId;
  seed: number;
  desc: CodecDescriptor;
  scan: ChunkScanV4;
}

function prepareV4(buf: ArrayBuffer | ArrayBufferView, meta: RecordingMeta): PreparedV4 {
  if (meta.recordVersion !== RECORD_V4_VERSION) {
    throw new Record4ParseError(
      `unsupported recordVersion ${meta.recordVersion} (this parser reads ${RECORD_V4_VERSION})`,
    );
  }
  if (!isEnvId(meta.envId)) {
    throw new Record4ParseError(`unknown envId ${JSON.stringify(meta.envId)}`);
  }
  if (!Number.isInteger(meta.seed) || meta.seed < 0) {
    throw new Record4ParseError(`invalid seed ${String(meta.seed)}`);
  }
  if (meta.codecVersion !== undefined && meta.codecVersion !== CODEC_VERSION) {
    throw new Record4ParseError(
      `meta codecVersion ${meta.codecVersion} (this build implements ${CODEC_VERSION})`,
    );
  }

  const bytes = asWindow(buf);
  const scan = scanChunksV4(bytes);
  const headerEnv = RECORD_V4_ENV_KINDS[scan.header.envKind];
  if (headerEnv === undefined) {
    throw new Record4ParseError(`file header envKind ${scan.header.envKind} is not a known env`);
  }
  if (headerEnv !== meta.envId) {
    throw new Record4ParseError(
      `file header envKind ${scan.header.envKind} (${headerEnv}) contradicts meta envId ${meta.envId}`,
    );
  }

  const desc = codecForEnvKind(scan.header.envKind);
  if (meta.codec !== undefined && meta.codec !== null && !codecMatches(desc, meta.codec)) {
    throw new Record4ParseError(
      `meta codec descriptor for ${meta.envId} does not match this build's codec table`,
    );
  }

  return { bytes, envId: meta.envId, seed: meta.seed, desc, scan };
}

/**
 * Parse a v4 `actions.bin` into the v3 `RawRecording` row shape.
 *
 * Accepts a bare `ArrayBuffer` or any view of one; only the view's own window is
 * read (a Bun `Buffer` is a view into a shared pool — decoding from offset 0
 * would read a neighbour's bytes). A torn or deliberately truncated tail yields
 * the valid prefix; a file with no valid chunk at all yields `steps === 0`, which
 * the caller reports as an empty lane.
 */
export async function parseActionsBinV4(
  buf: ArrayBuffer | ArrayBufferView,
  meta: RecordingMeta,
): Promise<RawRecording> {
  const { bytes, envId, seed, desc, scan } = prepareV4(buf, meta);
  const data = new Float32Array(scan.steps * RECORD_FLOATS);

  let steps = 0;
  for (const chunk of scan.chunks) {
    let payload: Uint8Array;
    try {
      payload = await inflateChunk(bytes, chunk);
    } catch {
      // The spec's last torn-tail signal: the payload will not inflate, or it
      // inflates to the wrong size. Drop this chunk and everything after it.
      break;
    }
    // NOT caught: a crc-valid chunk that inflated to exactly `rawBytes` and is
    // still structurally wrong is a CODEC bug, not a torn tail. Truncating there
    // would quietly play half a lane and call it the truth.
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
  /** 0-based step index of `data`'s first row */
  stepStart: number;
  stepCount: number;
  /**
   * `stepCount × 16` floats in the v3 row shape. REUSED between chunks — copy
   * anything that must outlive the visit.
   */
  data: Float32Array;
}

export type ChunkVisitorV4 = (rows: ChunkRowsV4) => boolean | void | Promise<boolean | void>;

/**
 * Streaming variant: decode chunk by chunk without ever materializing the lane.
 *
 * One scratch row buffer is reused across chunks, so a 1M-step lane costs one
 * chunk's worth of floats instead of 64 MB. Return `false` from `visit` to stop.
 * Answers the number of steps visited.
 */
export async function walkChunksV4(
  buf: ArrayBuffer | ArrayBufferView,
  meta: RecordingMeta,
  visit: ChunkVisitorV4,
): Promise<number> {
  const { bytes, desc, scan } = prepareV4(buf, meta);
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
    // As in `parseActionsBinV4`: a structural error past the crc is a codec bug
    // and propagates; only the inflate is treated as a torn tail.
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
