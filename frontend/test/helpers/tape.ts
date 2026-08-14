import { TAPE_FIELDS, TAPE_MAGIC } from '@shared/types';

export interface TapeRowSpec {
  stepStart: number;
  stepCount: number;
  wallSeconds?: number;
  values: Record<string, readonly number[]>;
}

export function tapeBytes(laneCount: number, rows: readonly TapeRowSpec[]): Uint8Array {
  const fields = [...TAPE_FIELDS];
  const json = new TextEncoder().encode(JSON.stringify({ fields }));
  const rowBytes = 12 + fields.length * laneCount * 4;
  const out = new Uint8Array(12 + json.byteLength + rows.length * rowBytes);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, TAPE_MAGIC, true);
  dv.setUint32(4, laneCount, true);
  dv.setUint32(8, json.byteLength, true);
  out.set(json, 12);
  let at = 12 + json.byteLength;
  for (const row of rows) {
    dv.setUint32(at, row.stepStart, true);
    dv.setUint32(at + 4, row.stepCount, true);
    dv.setFloat32(at + 8, row.wallSeconds ?? NaN, true);
    for (let f = 0; f < fields.length; f++) {
      const name = fields[f] ?? '';
      const col = row.values[name];
      for (let l = 0; l < laneCount; l++) {
        dv.setFloat32(at + 12 + (f * laneCount + l) * 4, col?.[l] ?? NaN, true);
      }
    }
    at += rowBytes;
  }
  return out;
}
