import type { EnvEntry } from './catalog';
import { ENV_IDS } from '@shared/envs';
import type { EnvId } from '@shared/envs';

export const POSE_LEN = 24;
export const OFF_DONE = 0;
export const OFF_POSE = 1;
export const RECORD_FLOATS = 1 + POSE_LEN;

export const CHUNK_MAGIC = 0x524b4843;
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

export interface RecordChunk {
  index: number;
  stepStart: number;
  stepCount: number;
  rawBytes: number;
  compBytes: number;
  bits: number;
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
  const bits = dv.getUint32(16, true);
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
    bits,
    byteOffset: offset,
    byteLength: CHUNK_HEADER_BYTES + compBytes,
    payloadOffset: offset + CHUNK_HEADER_BYTES,
  };
}

export async function inflateChunk(bytes: Uint8Array, chunk: RecordChunk): Promise<Uint8Array> {
  const comp = bytes.slice(chunk.payloadOffset, chunk.payloadOffset + chunk.compBytes);
  if ((chunk.bits >>> 31) === 0) {
    if (comp.byteLength !== chunk.rawBytes) {
      throw new RecordParseError(
        `chunk ${chunk.index} stores ${comp.byteLength} raw bytes, header claims ${chunk.rawBytes}`,
      );
    }
    return comp;
  }
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

const LPC_ORDER = 8;
const LPC_DIV = 16384;

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

  const metaAt = take(desc.poseFields.length, 'the column meta bytes');
  for (let k = 0; k < desc.poseFields.length; k++) {
    const field = desc.poseFields[k];
    if (field === undefined) continue;
    const slot = OFF_POSE + field.idx;
    const meta = payload[metaAt + k] ?? 0;
    const pred = meta & 3;
    const rice = (meta & 4) !== 0;
    let anchored = 0;
    if (field.anchor === true) {
      anchored = dv.getFloat64(take(8, `anchor seed of column ${field.idx}`), true);
    }
    const coefs: number[] = [];
    if (pred === 2) {
      const at = take(4 * LPC_ORDER, `LPC coefficients of column ${field.idx}`);
      for (let j = 0; j < LPC_ORDER; j++) coefs.push(dv.getInt32(at + 4 * j, true));
    }

    const residuals = new Float64Array(stepCount);
    if (rice) {
      const kAt = take(1, `Rice parameter of column ${field.idx}`);
      const riceK = payload[kAt] ?? 0;
      let cursor = off;
      let bitPos = 0;
      const readBit = (): number => {
        if (cursor >= limit) {
          throw new RecordParseError(`pose column ${field.idx} ran past the payload`);
        }
        const b = ((payload[cursor] ?? 0) >> bitPos) & 1;
        bitPos += 1;
        if (bitPos === 8) { bitPos = 0; cursor += 1; }
        return b;
      };
      for (let i = 0; i < stepCount; i++) {
        let unary = 0;
        while (readBit() === 0) unary += 1;
        let low = 0;
        for (let j = 0; j < riceK; j++) low += readBit() * 2 ** j;
        const z = unary * 2 ** riceK + low;
        residuals[i] = z % 2 === 0 ? z / 2 : -(z + 1) / 2;
      }
      take((bitPos > 0 ? cursor + 1 : cursor) - off, `pose column ${field.idx}`);
    } else {
      let cursor = off;
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
        residuals[i] = raw % 2 === 0 ? raw / 2 : -(raw + 1) / 2;
      }
      take(cursor - off, `pose column ${field.idx}`);
    }

    const q = new Float64Array(stepCount);
    if (pred === 2) {
      let value = 0;
      for (let i = 0; i < stepCount; i++) {
        if (i < LPC_ORDER) {
          value += residuals[i] ?? 0;
        } else {
          let acc = 0;
          for (let j = 0; j < LPC_ORDER; j++) acc += (coefs[j] ?? 0) * (q[i - 1 - j] ?? 0);
          value = Math.floor(acc / LPC_DIV) + (residuals[i] ?? 0);
        }
        q[i] = value;
      }
    } else {
      let value = 0;
      let delta = 0;
      for (let i = 0; i < stepCount; i++) {
        const r = residuals[i] ?? 0;
        delta = pred === 1 && i >= 2 ? delta + r : r;
        value = i === 0 ? delta : value + delta;
        q[i] = value;
      }
    }

    if (field.anchor === true) {
      let prev = 0;
      for (let i = 0; i < stepCount; i++) {
        anchored += ((q[i] ?? 0) - prev) * field.scale;
        prev = q[i] ?? 0;
        out[base + i * RECORD_FLOATS + slot] = anchored;
      }
    } else {
      for (let i = 0; i < stepCount; i++) {
        out[base + i * RECORD_FLOATS + slot] = (q[i] ?? 0) * field.scale;
      }
    }
  }

  if (off !== limit) {
    throw new RecordParseError(`chunk payload has ${limit - off} trailing bytes`);
  }
}
