// record-v3 `actions.bin` parser.
//
// Layout (Sources/Main.swift record writer, AGENTS.md "Fleet run output"):
//   flat little-endian f32[n][16] = ctrl[6], done, pose[9]  -> 64 bytes/step.
// Row r is the POST-step state after action r. There is no row for the initial
// state; sim-core synthesizes frame -1 from each env's deterministic init.

import { isEnvId } from '@shared/envs';
import type { EnvId } from '@shared/envs';
import type { RecordingMeta } from '@shared/types';
import type { RawRecording } from './types';

/** floats per record-v3 row */
export const RECORD_FLOATS = 16;
/** bytes per record-v3 row */
export const RECORD_BYTES = RECORD_FLOATS * 4;
/** the only servable record version */
export const RECORD_VERSION = 3;

/** field offsets inside a row */
export const OFF_CTRL = 0;
export const CTRL_LEN = 6;
export const OFF_DONE = 6;
export const OFF_POSE = 7;
export const POSE_LEN = 9;

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

/**
 * Parse an `actions.bin` buffer into a RawRecording.
 * Rejects any recordVersion other than 3 and any non-row-aligned length.
 *
 * Accepts a bare `ArrayBuffer` (`Response.arrayBuffer()`) OR any typed-array /
 * DataView view of one. The view case matters: a Node/Bun `Buffer` is a VIEW —
 * `readFileSync(p).buffer` can be a shared allocation pool, and any `subarray`
 * of a buffer starts partway in — so the recording lives at `byteOffset` inside
 * a larger buffer, and decoding from offset 0 would read the neighbouring
 * bytes as if they were record rows. Only the view's own window is ever read.
 */
export function parseActionsBin(
  buf: ArrayBuffer | ArrayBufferView,
  meta: RecordingMeta,
): RawRecording {
  if (meta.recordVersion !== RECORD_VERSION) {
    throw new RecordParseError(
      `unsupported recordVersion ${meta.recordVersion} (only ${RECORD_VERSION} is servable)`,
    );
  }
  if (!isEnvId(meta.envId)) {
    throw new RecordParseError(`unknown envId ${JSON.stringify(meta.envId)}`);
  }
  if (!Number.isInteger(meta.seed) || meta.seed < 0) {
    throw new RecordParseError(`invalid seed ${String(meta.seed)}`);
  }
  const bytes = buf.byteLength;
  const byteOffset = ArrayBuffer.isView(buf) ? buf.byteOffset : 0;
  const buffer: ArrayBufferLike = ArrayBuffer.isView(buf) ? buf.buffer : buf;
  if (bytes % RECORD_BYTES !== 0) {
    throw new RecordParseError(
      `actions.bin length ${bytes} is not a multiple of ${RECORD_BYTES}`,
    );
  }
  const steps = bytes / RECORD_BYTES;
  const floats = steps * RECORD_FLOATS;
  const data = LITTLE_ENDIAN
    ? viewFloat32(buffer, byteOffset, floats)
    : swapToFloat32(buffer, byteOffset, floats);
  return { envId: meta.envId as EnvId, seed: meta.seed, data, steps };
}

/**
 * Zero-copy f32 view of `[byteOffset, byteOffset + 4*floats)`. A Float32Array
 * needs a 4-byte-aligned offset; a pooled Buffer slice is not guaranteed to be
 * one, so an unaligned window is copied out (byte-exact) instead.
 */
function viewFloat32(buffer: ArrayBufferLike, byteOffset: number, floats: number): Float32Array {
  if (byteOffset % 4 === 0) return new Float32Array(buffer, byteOffset, floats);
  const copy = new Uint8Array(floats * 4);
  copy.set(new Uint8Array(buffer, byteOffset, floats * 4));
  return new Float32Array(copy.buffer);
}

/** Big-endian host fallback: read every f32 explicitly little-endian. */
function swapToFloat32(buffer: ArrayBufferLike, byteOffset: number, floats: number): Float32Array {
  const view = new DataView(buffer, byteOffset, floats * 4);
  const out = new Float32Array(floats);
  for (let i = 0; i < floats; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

/** Zero-copy view of row `i` (16 floats). Rows are disjoint and stable. */
export function recordRow(raw: RawRecording, i: number): Float32Array {
  if (i < 0 || i >= raw.steps) throw new RangeError(`row ${i} out of range (${raw.steps})`);
  return raw.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS);
}

/** Zero-copy `ctrl[6]` view of row `i`. */
export function rowCtrl(raw: RawRecording, i: number): Float32Array {
  const base = i * RECORD_FLOATS;
  return raw.data.subarray(base + OFF_CTRL, base + OFF_CTRL + CTRL_LEN);
}

/** Zero-copy `pose[9]` view of row `i`. */
export function rowPose(raw: RawRecording, i: number): Float32Array {
  const base = i * RECORD_FLOATS;
  return raw.data.subarray(base + OFF_POSE, base + OFF_POSE + POSE_LEN);
}

/** `done` flag of row `i`. */
export function rowDone(raw: RawRecording, i: number): number {
  return raw.data[i * RECORD_FLOATS + OFF_DONE] ?? 0;
}

/**
 * Build a RawRecording from already-decoded floats (fixtures, synthetic tests).
 * `envId` is validated at runtime: callers hand it in from JSON fixtures and
 * URL params through a cast, so a typo must surface here as a RecordParseError
 * rather than as a TypeError deep inside the decode driver's `ENVS` lookup.
 */
export function rawRecordingFromFloats(
  envId: EnvId,
  seed: number,
  data: Float32Array,
): RawRecording {
  if (!isEnvId(envId)) {
    throw new RecordParseError(`unknown envId ${JSON.stringify(envId)}`);
  }
  if (data.length % RECORD_FLOATS !== 0) {
    throw new RecordParseError(
      `float count ${data.length} is not a multiple of ${RECORD_FLOATS}`,
    );
  }
  return { envId, seed, data, steps: data.length / RECORD_FLOATS };
}
