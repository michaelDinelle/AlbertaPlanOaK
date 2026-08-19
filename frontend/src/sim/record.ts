import type { EnvEntry } from './catalog';
import { ENV_IDS } from '@shared/envs';
import type { EnvId } from '@shared/envs';

export const POSE_LEN = 9;
export const OFF_DONE = 0;
export const OFF_POSE = 1;
export const RECORD_FLOATS = 1 + POSE_LEN;

export const FILE_MAGIC = 0x524b414f;
export const CHUNK_MAGIC = 0x524b4843;
export const FILE_HEADER_BYTES = 8;
export const CHUNK_HEADER_BYTES = 20;

export class RecordParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RecordParseError';
  }
}

export interface PoseFieldSpec {
  readonly idx: number;
  readonly scale: number;
  readonly wrap?: boolean;
  readonly anchor?: boolean;
}

export interface CodecDescriptor {
  readonly poseFields: readonly PoseFieldSpec[];
}

export const RECORD_ENV_KINDS: readonly EnvId[] = ENV_IDS;

export function codecFromEntry(entry: EnvEntry): CodecDescriptor {
  return {
    poseFields: entry.fields.map((f) => ({
      idx: f.slot,
      scale: f.scale,
      ...(f.wrap ? { wrap: true } : {}),
      ...(f.anchor ? { anchor: true } : {}),
    })),
  };
}

function asWindow(buf: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (ArrayBuffer.isView(buf)) {
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  return new Uint8Array(buf);
}

export interface RecordHeader {
  envKind: number;
  envId: EnvId;
}

export function readRecordHeader(buf: ArrayBuffer | ArrayBufferView): RecordHeader {
  const bytes = asWindow(buf);
  if (bytes.byteLength < FILE_HEADER_BYTES) {
    throw new RecordParseError(
      `pose.bin is ${bytes.byteLength} bytes — shorter than the ${FILE_HEADER_BYTES}-byte header`,
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
  return { envKind, envId };
}

export interface RecordChunk {
  index: number;
  stepStart: number;
  stepCount: number;
  rawBytes: number;
  compBytes: number;
  predictorBits: number;
  byteOffset: number;
  byteLength: number;
  payloadOffset: number;
}

export function scanNextChunk(
  bytes: Uint8Array,
  offset: number,
  expectStart: number,
  index: number,
): RecordChunk | null {
  if (offset + CHUNK_HEADER_BYTES > bytes.byteLength) return null;
  const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, CHUNK_HEADER_BYTES);
  const magic = dv.getUint32(0, true);
  if (magic !== CHUNK_MAGIC) {
    throw new RecordParseError(
      `chunk ${index} at byte ${offset} has magic 0x${magic.toString(16)}, not 0x${CHUNK_MAGIC.toString(16)}`,
    );
  }
  const stepCount = dv.getUint32(4, true);
  const rawBytes = dv.getUint32(8, true);
  const compBytes = dv.getUint32(12, true);
  const predictorBits = dv.getUint32(16, true);
  if (stepCount === 0 || compBytes === 0) {
    throw new RecordParseError(`chunk ${index} at byte ${offset} declares ${stepCount} steps in ${compBytes} bytes`);
  }
  if (offset + CHUNK_HEADER_BYTES + compBytes > bytes.byteLength) return null;
  return {
    index,
    stepStart: expectStart,
    stepCount,
    rawBytes,
    compBytes,
    predictorBits,
    byteOffset: offset,
    byteLength: CHUNK_HEADER_BYTES + compBytes,
    payloadOffset: offset + CHUNK_HEADER_BYTES,
  };
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
  predictorBits: number,
  out: Float32Array,
  base: number,
): void {
  const limit = payload.byteLength;
  let off = 0;
  const take = (n: number, what: string): number => {
    if (off + n > limit) {
      throw new RecordParseError(
        `chunk payload is ${limit} bytes \u2014 too short for ${what} at offset ${off} (+${n})`,
      );
    }
    const at = off;
    off += n;
    return at;
  };

  const doneAt = take((stepCount + 7) >> 3, 'the done bitstream');

  const floats = stepCount * RECORD_FLOATS;
  out.fill(0, base, base + floats);
  for (let i = 0; i < stepCount; i++) {
    const bit = (payload[doneAt + (i >> 3)] ?? 0) >> (i & 7);
    if ((bit & 1) !== 0) out[base + i * RECORD_FLOATS + OFF_DONE] = 1;
  }

  for (let k = 0; k < desc.poseFields.length; k++) {
    const field = desc.poseFields[k];
    if (field === undefined) continue;
    const slot = OFF_POSE + field.idx;
    let cursor = off;
    let anchored = 0;
    if (field.anchor === true) {
      const dv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
      anchored = dv.getFloat64(cursor, true);
      cursor += 8;
    }
    const secondOrder = ((predictorBits >>> k) & 1) !== 0;
    let value = 0;
    let delta = 0;
    for (let i = 0; i < stepCount; i++) {
      let shift = 0;
      let raw = 0;
      for (;;) {
        if (cursor >= limit) {
          throw new RecordParseError(`pose column ${field.idx} ran past the payload`);
        }
        const byte = payload[cursor] ?? 0;
        cursor += 1;
        raw += (byte & 0x7f) * 2 ** shift;
        if ((byte & 0x80) === 0) break;
        shift += 7;
      }
      const residual = raw % 2 === 0 ? raw / 2 : -(raw + 1) / 2;
      if (field.anchor === true) {
        anchored += residual * field.scale;
        out[base + i * RECORD_FLOATS + slot] = anchored;
        continue;
      }
      delta = secondOrder && i >= 2 ? delta + residual : residual;
      value = i === 0 ? delta : value + delta;
      out[base + i * RECORD_FLOATS + slot] = value * field.scale;
    }
    take(cursor - off, `pose column ${field.idx}`);
  }

  if (off !== limit) {
    throw new RecordParseError(`chunk payload has ${limit - off} trailing bytes`);
  }
}
