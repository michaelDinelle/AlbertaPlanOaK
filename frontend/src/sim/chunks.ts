import { inflateSync } from 'three/examples/jsm/libs/fflate.module.js';
import { FRAME_FLOATS, OFF_DONE, OFF_POSE } from '../data/oak';
import type { RecordFieldSpec } from './catalog';

const CHUNK_MAGIC = 0x524b4843;
const LPC_ORDER = 8;
const LPC_SHIFT = 14;

export interface Chunk { at: number; stored: number; payload: number; live: number; start: number; zlib: boolean }

export function indexChunks(bytes: Uint8Array, from: number, into: Chunk[]): number {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let off = from;
  while (off + 20 <= bytes.length) {
    if (dv.getUint32(off, true) !== CHUNK_MAGIC) throw new Error(`chunk ${into.length} at ${off} is not a record chunk`);
    const stored = dv.getUint32(off + 12, true);
    if (off + 20 + stored > bytes.length) break;
    const prev = into[into.length - 1];
    into.push({ at: off + 20, stored, payload: dv.getUint32(off + 8, true), live: dv.getUint32(off + 4, true), start: prev === undefined ? 0 : prev.start + prev.live, zlib: dv.getUint32(off + 16, true) >>> 31 === 1 });
    off += 20 + stored;
  }
  return off;
}

const exact = (x: number): number => {
  if (!Number.isSafeInteger(x)) throw new Error(`pose integer ${x} exceeds 2^53`);
  return x;
};
const unzig = (z: number): number => (exact(z) % 2 ? -(z + 1) / 2 : z / 2);

function decodeColumns(p: Uint8Array, live: number, fields: RecordFieldSpec[]): { doneAt: number; columns: Float64Array[] } {
  const dv = new DataView(p.buffer, p.byteOffset, p.byteLength);
  let off = 0;
  const take = (n: number, what: string): number => {
    if (off + n > p.length) throw new Error(`chunk payload is ${p.length} bytes, too short for ${what} at ${off} (+${n})`);
    const at = off;
    off += n;
    return at;
  };
  const doneAt = take((live + 7) >> 3, 'the done bitstream');
  const metaAt = take(fields.length, 'the column meta bytes');
  const columns = fields.map((f, k) => {
    const m = p[metaAt + k]!;
    const pred = m & 3;
    let anchored = f.anchor ? dv.getFloat64(take(8, `anchor seed of column ${k}`), true) : 0;
    const coefs: number[] = [];
    if (pred === 2) for (let j = 0; j < LPC_ORDER; j++) coefs.push(dv.getInt32(take(4, `LPC coefficients of column ${k}`), true));
    const res = new Float64Array(live);
    if (m & 4) {
      const riceK = p[take(1, `Rice parameter of column ${k}`)]!;
      let bit = off * 8;
      const readBit = (): number => {
        if (bit >> 3 >= p.length) throw new Error(`column ${k} ran past the payload`);
        return (p[bit >> 3]! >> (bit++ & 7)) & 1;
      };
      for (let i = 0; i < live; i++) {
        let unary = 0;
        while (readBit() === 0) unary++;
        let low = 0;
        for (let j = 0; j < riceK; j++) low += readBit() * 2 ** j;
        res[i] = unzig(unary * 2 ** riceK + low);
      }
      off = (bit + 7) >> 3;
    } else {
      for (let i = 0; i < live; i++) {
        let raw = 0;
        let mul = 1;
        for (;;) {
          const b = p[take(1, `column ${k}`)]!;
          raw += (b & 0x7f) * mul;
          if ((b & 0x80) === 0) break;
          mul *= 128;
        }
        res[i] = unzig(raw);
      }
    }
    const q = new Float64Array(live);
    if (pred === 2) {
      for (let i = 0; i < live; i++) {
        let acc = 0;
        if (i >= LPC_ORDER) {
          for (let j = 0; j < LPC_ORDER; j++) acc = exact(acc + exact(coefs[j]! * q[i - 1 - j]!));
          q[i] = exact(Math.floor(acc / 2 ** LPC_SHIFT) + res[i]!);
        } else {
          q[i] = exact((i === 0 ? 0 : q[i - 1]!) + res[i]!);
        }
      }
    } else {
      let value = 0;
      let delta = 0;
      for (let i = 0; i < live; i++) {
        delta = pred === 1 && i >= 2 ? delta + res[i]! : res[i]!;
        value = i === 0 ? delta : value + delta;
        q[i] = exact(value);
      }
    }
    const out = new Float64Array(live);
    if (f.anchor) {
      let prev = 0;
      for (let i = 0; i < live; i++) {
        anchored += exact(q[i]! - prev) * f.scale;
        prev = q[i]!;
        out[i] = anchored;
      }
    } else {
      for (let i = 0; i < live; i++) out[i] = q[i]! * f.scale;
    }
    return out;
  });
  if (off !== p.length) throw new Error(`chunk payload has ${p.length - off} trailing bytes`);
  return { doneAt, columns };
}

export function decodeChunk(bytes: Uint8Array, chunk: Chunk, fields: RecordFieldSpec[], out: Float32Array, row: number): void {
  const stored = bytes.subarray(chunk.at, chunk.at + chunk.stored);
  const p = chunk.zlib ? inflateSync(stored) : stored;
  if (p.length !== chunk.payload) throw new Error(`chunk payload is ${p.length} bytes, header claims ${chunk.payload}`);
  const { doneAt, columns } = decodeColumns(p, chunk.live, fields);
  for (let i = 0; i < chunk.live; i++) {
    const r = (row + i) * FRAME_FLOATS;
    out[r + OFF_DONE] = (p[doneAt + (i >> 3)]! >> (i & 7)) & 1;
    fields.forEach((f, c) => { out[r + OFF_POSE + f.slot] = columns[c]![i]!; });
  }
}
