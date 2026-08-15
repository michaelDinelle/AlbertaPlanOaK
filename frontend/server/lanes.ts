import { open } from 'node:fs/promises';
import type { RecordingEntry } from './recordings';

export const LANES_MAGIC = 0x4c4b414f;
export const GEOMETRY_BYTES = 76;
export const FILE_HEADER_BYTES = 12 + GEOMETRY_BYTES;
export const CHUNK_HEADER_BYTES = 24;
const CHUNK_MAGIC = 0x524b4843;

interface LaneSlice {
  seed: number;
  geometry: Uint8Array;
  chunks: Uint8Array[];
  stepStart: number;
  stepCount: number;
}

async function sliceLane(entry: RecordingEntry, from: number, to: number): Promise<LaneSlice> {
  const handle = await open(entry.actionsPath, 'r');
  try {
    const head = new Uint8Array(FILE_HEADER_BYTES);
    await handle.read(head, 0, head.byteLength, 0);
    const geometry = head.slice(12);

    const chunks: Uint8Array[] = [];
    let offset = FILE_HEADER_BYTES;
    let stepStart = -1;
    let stepCount = 0;
    const header = new Uint8Array(CHUNK_HEADER_BYTES);
    for (;;) {
      const read = await handle.read(header, 0, CHUNK_HEADER_BYTES, offset);
      if (read.bytesRead < CHUNK_HEADER_BYTES) break;
      const dv = new DataView(header.buffer, header.byteOffset, CHUNK_HEADER_BYTES);
      if (dv.getUint32(0, true) !== CHUNK_MAGIC) break;
      const chunkStart = dv.getUint32(4, true);
      const chunkSteps = dv.getUint32(8, true);
      const compBytes = dv.getUint32(16, true);
      const total = CHUNK_HEADER_BYTES + compBytes;
      const chunkEnd = chunkStart + chunkSteps;
      if (chunkEnd > from && chunkStart < to) {
        const body = new Uint8Array(total);
        await handle.read(body, 0, total, offset);
        chunks.push(body);
        if (stepStart < 0) stepStart = chunkStart;
        stepCount = chunkEnd - stepStart;
      }
      if (chunkStart >= to) break;
      offset += total;
    }
    return {
      seed: entry.seed,
      geometry,
      chunks,
      stepStart: stepStart < 0 ? from : stepStart,
      stepCount,
    };
  } finally {
    await handle.close();
  }
}

export async function buildLaneBlock(
  entries: readonly RecordingEntry[],
  from: number,
  count: number,
): Promise<Uint8Array> {
  const to = from + count;
  const slices = await Promise.all(entries.map((e) => sliceLane(e, from, to)));

  const indexBytes = 20 + slices.length * 20;
  let payloadBytes = 0;
  for (const s of slices) {
    payloadBytes += s.geometry.byteLength;
    for (const c of s.chunks) payloadBytes += c.byteLength;
  }

  const out = new Uint8Array(indexBytes + payloadBytes);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, LANES_MAGIC, true);
  dv.setUint32(4, slices.length, true);
  dv.setUint32(8, from, true);
  dv.setUint32(12, count, true);
  dv.setUint32(16, GEOMETRY_BYTES, true);

  let cursor = indexBytes;
  for (let i = 0; i < slices.length; i++) {
    const s = slices[i]!;
    const row = 20 + i * 20;
    const start = cursor;
    out.set(s.geometry, cursor);
    cursor += s.geometry.byteLength;
    for (const c of s.chunks) {
      out.set(c, cursor);
      cursor += c.byteLength;
    }
    dv.setUint32(row, s.seed, true);
    dv.setUint32(row + 4, start, true);
    dv.setUint32(row + 8, cursor - start, true);
    dv.setUint32(row + 12, s.stepStart, true);
    dv.setUint32(row + 16, s.stepCount, true);
  }
  return out;
}
