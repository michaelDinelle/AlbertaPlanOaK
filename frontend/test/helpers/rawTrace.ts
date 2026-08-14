import { isEnvId } from '@shared/envs';
import type { EnvId } from '@shared/envs';
import { RECORD_BYTES, RECORD_FLOATS, RecordParseError } from '@sim/record';
import type { RawRecording } from '@sim/types';

export interface TraceMeta {
  envId: string;
  seed: number;
}

const LITTLE_ENDIAN = (() => {
  const probe = new Uint8Array(4);
  new Uint32Array(probe.buffer)[0] = 1;
  return probe[0] === 1;
})();

function asFloat32(buf: ArrayBuffer | ArrayBufferView, floats: number): Float32Array {
  const byteOffset = ArrayBuffer.isView(buf) ? buf.byteOffset : 0;
  const buffer: ArrayBufferLike = ArrayBuffer.isView(buf) ? buf.buffer : buf;
  if (LITTLE_ENDIAN) {
    if (byteOffset % 4 === 0) return new Float32Array(buffer, byteOffset, floats);
    const copy = new Uint8Array(floats * 4);
    copy.set(new Uint8Array(buffer, byteOffset, floats * 4));
    return new Float32Array(copy.buffer);
  }
  const view = new DataView(buffer, byteOffset, floats * 4);
  const out = new Float32Array(floats);
  for (let i = 0; i < floats; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

export function parseRawTrace(buf: ArrayBuffer | ArrayBufferView, meta: TraceMeta): RawRecording {
  if (!isEnvId(meta.envId)) {
    throw new RecordParseError(`unknown envId ${JSON.stringify(meta.envId)}`);
  }
  if (!Number.isInteger(meta.seed) || meta.seed < 0) {
    throw new RecordParseError(`invalid seed ${String(meta.seed)}`);
  }
  const bytes = buf.byteLength;
  if (bytes % RECORD_BYTES !== 0) {
    throw new RecordParseError(`trace length ${bytes} is not a multiple of ${RECORD_BYTES}`);
  }
  const steps = bytes / RECORD_BYTES;
  const data = asFloat32(buf, steps * RECORD_FLOATS);
  return { envId: meta.envId as EnvId, seed: meta.seed, data, steps };
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
    throw new RecordParseError(`float count ${data.length} is not a multiple of ${RECORD_FLOATS}`);
  }
  return { envId, seed, data, steps: data.length / RECORD_FLOATS };
}
