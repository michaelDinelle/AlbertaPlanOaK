
import { readFileSync } from 'node:fs';
import { crc32, inflateRawSync } from 'node:zlib';

const FILE_MAGIC = 0x544b414f;
const SEG_MAGIC = 0x54474553;

export interface MetricsTape {
  fields: string[];
  laneCount: number;
  stepStart: number[];
  stepCount: number[];
  data: Float32Array[];
}

export function readTape(path: string): MetricsTape | null {
  let buf: Buffer;
  try {
    buf = readFileSync(path);
  } catch {
    return null;
  }
  if (buf.length < 16) return null;
  if (buf.readUInt32LE(0) !== FILE_MAGIC || buf.readUInt32LE(4) !== 1) return null;
  const laneCount = buf.readUInt32LE(8);
  const jsonLen = buf.readUInt32LE(12);
  if (16 + jsonLen > buf.length) return null;
  let fields: string[];
  try {
    fields = (JSON.parse(buf.subarray(16, 16 + jsonLen).toString('utf8')) as { fields: string[] })
      .fields;
  } catch {
    return null;
  }
  const rowBytes = 8 + fields.length * laneCount * 4;
  const stepStart: number[] = [];
  const stepCount: number[] = [];
  const data: Float32Array[] = [];
  let off = 16 + jsonLen;
  while (off + 20 <= buf.length) {
    if (buf.readUInt32LE(off) !== SEG_MAGIC) break;
    const nWin = buf.readUInt32LE(off + 4);
    const rawBytes = buf.readUInt32LE(off + 8);
    const compBytes = buf.readUInt32LE(off + 12);
    const crc = buf.readUInt32LE(off + 16);
    if (off + 20 + compBytes > buf.length) break;
    const comp = buf.subarray(off + 20, off + 20 + compBytes);
    if ((crc32(comp) >>> 0) !== crc) break;
    let payload: Buffer;
    try {
      payload = inflateRawSync(comp);
    } catch {
      break;
    }
    if (payload.length !== rawBytes || rawBytes !== nWin * rowBytes) break;
    for (let w = 0; w < nWin; w++) {
      const p = w * rowBytes;
      stepStart.push(payload.readUInt32LE(p));
      stepCount.push(payload.readUInt32LE(p + 4));
      const vals = new Float32Array(fields.length * laneCount);
      for (let i = 0; i < vals.length; i++) vals[i] = payload.readFloatLE(p + 8 + i * 4);
      data.push(vals);
    }
    off += 20 + compBytes;
  }
  if (data.length === 0) return null;
  return { fields, laneCount, stepStart, stepCount, data };
}

export const TAPE_FIELD_MAP: Readonly<Record<string, string>> = {
  rewardSum: 'rewardSum',
  doneCount: 'doneCount',
  kernelRho: 'kernelRho',
  paramAbsMax: 'paramAbsMax',
  'health.rewardRate': 'rewardRate',
  'health.actionDeltaPeak': 'deltaPeak',
};

export function tapeValue(
  tape: MetricsTape,
  fieldPath: string,
  window: number,
  lane: number,
): number | null {
  const stepEnd = (tape.stepStart[window] ?? 0) + (tape.stepCount[window] ?? 0);
  const row = tape.data[window];
  if (row === undefined) return null;
  const get = (name: string): number | null => {
    const fi = tape.fields.indexOf(name);
    if (fi < 0) return null;
    const v = row[fi * tape.laneCount + lane];
    return v !== undefined && Number.isFinite(v) ? v : null;
  };
  if (fieldPath === 'health.actionDeltaL2Mean') {
    const ds = get('deltaSum');
    return ds === null ? null : ds / Math.max(stepEnd, 1);
  }
  if (fieldPath === 'paramAbsMax') {
    const v = get('paramAbsMax');
    return v !== null && v >= 0 ? v : null;
  }
  const mapped = TAPE_FIELD_MAP[fieldPath];
  return mapped === undefined ? null : get(mapped);
}
