
import { crc32, deflateRawSync } from 'node:zlib';

export const DEFAULT_TAPE_FIELDS = [
  'rewardSum',
  'doneCount',
  'rewardRate',
  'kernelRho',
  'deltaSum',
  'deltaPeak',
  'paramAbsMax',
] as const;

export interface TapeWindowSpec {
  start: number;
  count: number;
  values: Record<string, number[]>;
}

export function tapeBytes(
  laneCount: number,
  windows: TapeWindowSpec[],
  fields: readonly string[] = DEFAULT_TAPE_FIELDS,
): Buffer {
  const rowBytes = 8 + fields.length * laneCount * 4;
  const payload = Buffer.alloc(windows.length * rowBytes);
  windows.forEach((w, wi) => {
    const p = wi * rowBytes;
    payload.writeUInt32LE(w.start, p);
    payload.writeUInt32LE(w.count, p + 4);
    fields.forEach((f, fi) => {
      for (let l = 0; l < laneCount; l++) {
        payload.writeFloatLE(w.values[f]?.[l] ?? 0, p + 8 + (fi * laneCount + l) * 4);
      }
    });
  });
  const json = Buffer.from(JSON.stringify({ fields }), 'utf8');
  const head = Buffer.alloc(16);
  head.writeUInt32LE(0x544b414f, 0);
  head.writeUInt32LE(1, 4);
  head.writeUInt32LE(laneCount, 8);
  head.writeUInt32LE(json.length, 12);
  const comp = deflateRawSync(payload);
  const seg = Buffer.alloc(20);
  seg.writeUInt32LE(0x54474553, 0);
  seg.writeUInt32LE(windows.length, 4);
  seg.writeUInt32LE(payload.length, 8);
  seg.writeUInt32LE(comp.length, 12);
  seg.writeUInt32LE(crc32(comp) >>> 0, 16);
  return Buffer.concat([head, json, seg, comp]);
}
