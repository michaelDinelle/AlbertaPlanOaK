
import { isEnvId } from '@shared/envs';
import type { EnvId } from '@shared/envs';
import type { RecordingMeta } from '@shared/types';
import type { RawRecording } from './types';

export const RECORD_FLOATS = 16;
export const RECORD_BYTES = RECORD_FLOATS * 4;
export const RECORD_VERSION = 3;

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

function viewFloat32(buffer: ArrayBufferLike, byteOffset: number, floats: number): Float32Array {
  if (byteOffset % 4 === 0) return new Float32Array(buffer, byteOffset, floats);
  const copy = new Uint8Array(floats * 4);
  copy.set(new Uint8Array(buffer, byteOffset, floats * 4));
  return new Float32Array(copy.buffer);
}

function swapToFloat32(buffer: ArrayBufferLike, byteOffset: number, floats: number): Float32Array {
  const view = new DataView(buffer, byteOffset, floats * 4);
  const out = new Float32Array(floats);
  for (let i = 0; i < floats; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

export function recordRow(raw: RawRecording, i: number): Float32Array {
  if (i < 0 || i >= raw.steps) throw new RangeError(`row ${i} out of range (${raw.steps})`);
  return raw.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS);
}

export function rowCtrl(raw: RawRecording, i: number): Float32Array {
  const base = i * RECORD_FLOATS;
  return raw.data.subarray(base + OFF_CTRL, base + OFF_CTRL + CTRL_LEN);
}

export function rowPose(raw: RawRecording, i: number): Float32Array {
  const base = i * RECORD_FLOATS;
  return raw.data.subarray(base + OFF_POSE, base + OFF_POSE + POSE_LEN);
}

export function rowDone(raw: RawRecording, i: number): number {
  return raw.data[i * RECORD_FLOATS + OFF_DONE] ?? 0;
}

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
