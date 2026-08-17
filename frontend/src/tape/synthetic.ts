import { TAPE_MAGIC } from '@shared/types';

export type SyntheticCell = number | { bits: number };

export interface SyntheticRow {
  start: number;
  count: number;
  wall: number;
  values: SyntheticCell[][];
}

export function syntheticTapeBytes(laneCount: number, fields: string[], rows: SyntheticRow[]): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify({ fields }));
  const headerBytes = 12 + json.byteLength;
  const rowBytes = 12 + fields.length * laneCount * 4;
  const bytes = new Uint8Array(headerBytes + rows.length * rowBytes);
  const dv = new DataView(bytes.buffer);
  dv.setUint32(0, TAPE_MAGIC, true);
  dv.setUint32(4, laneCount, true);
  dv.setUint32(8, json.byteLength, true);
  bytes.set(json, 12);
  rows.forEach((row, r) => {
    const at = headerBytes + r * rowBytes;
    dv.setUint32(at, row.start, true);
    dv.setUint32(at + 4, row.count, true);
    dv.setFloat32(at + 8, row.wall, true);
    fields.forEach((_, f) => {
      for (let lane = 0; lane < laneCount; lane++) {
        const cell = row.values[f]?.[lane] ?? NaN;
        const offset = at + 12 + (f * laneCount + lane) * 4;
        if (typeof cell === 'number') dv.setFloat32(offset, cell, true);
        else dv.setUint32(offset, cell.bits, true);
      }
    });
  });
  return bytes;
}
