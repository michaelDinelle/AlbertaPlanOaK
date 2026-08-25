import { TAPE_MAGIC } from '@shared/types';

export type SyntheticCell = number | { bits: number };

export interface SyntheticRow {
  start: number;
  count: number;
  wall: number;
  values: SyntheticCell[][];
}

export async function deflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function syntheticTapeBytes(
  laneCount: number,
  fields: string[],
  rows: SyntheticRow[],
): Promise<Uint8Array> {
  const json = new TextEncoder().encode(JSON.stringify({ fields }));
  const parts: Uint8Array[] = [];
  const header = new Uint8Array(12 + json.byteLength);
  const hv = new DataView(header.buffer);
  hv.setUint32(0, TAPE_MAGIC, true);
  hv.setUint32(4, laneCount, true);
  hv.setUint32(8, json.byteLength, true);
  header.set(json, 12);
  parts.push(header);
  for (const row of rows) {
    const block = new Uint8Array(fields.length * laneCount * 4);
    const bv = new DataView(block.buffer);
    fields.forEach((_, f) => {
      for (let lane = 0; lane < laneCount; lane++) {
        const cell = row.values[f]?.[lane] ?? NaN;
        const offset = (f * laneCount + lane) * 4;
        if (typeof cell === 'number') bv.setFloat32(offset, cell, true);
        else bv.setUint32(offset, cell.bits, true);
      }
    });
    const comp = await deflateRaw(block);
    const head = new DataView(new ArrayBuffer(16));
    head.setUint32(0, row.start, true);
    head.setUint32(4, row.count, true);
    head.setFloat32(8, row.wall, true);
    head.setUint32(12, comp.byteLength, true);
    parts.push(new Uint8Array(head.buffer), comp);
  }
  return new Uint8Array(await new Blob(parts as BlobPart[]).arrayBuffer());
}
