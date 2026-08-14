import { deflateRawSync } from 'node:zlib';
import {
  CHUNK_MAGIC,
  FILE_MAGIC,
  RECORD_FLOATS,
  crc32,
  envKindOf,
  roundHalfAwayFromZero,
  wrapAngle,
  type CodecDescriptor,
} from '@sim/record';
import { codecForEnvKind } from '@sim/record';

export interface EncodeResult {
  bytes: Uint8Array;
  steps: number;
  closed: boolean;
}

export function encodeChunkPayload(
  rows: Float32Array,
  stepCount: number,
  desc: CodecDescriptor,
  opts: { wrap?: boolean } = {},
): { payload: Uint8Array; live: number; closed: boolean } {
  let live = stepCount;
  let closed = false;
  scan: for (let i = 0; i < stepCount; i++) {
    for (let u = 0; u < RECORD_FLOATS; u++) {
      if (!Number.isFinite(rows[i * RECORD_FLOATS + u])) {
        live = i;
        closed = true;
        break scan;
      }
    }
  }
  if (live === 0) return { payload: new Uint8Array(0), live: 0, closed };

  const nBox = desc.ctrl === 'index' ? 0 : desc.ctrl.n;
  const hasDx = desc.poseFields.some((f) => f.enc === 'dx32');
  const baseOff = live * (1 + nBox);
  let cursor = baseOff + (hasDx ? 8 : 0);
  const off: number[] = [];
  for (const f of desc.poseFields) {
    off.push(cursor);
    cursor += (f.enc === 'u8' ? 1 : f.enc === 'i16' ? 2 : 4) * live;
  }
  const fixed = cursor + 4;
  const escRows: number[] = [];
  const b = new Uint8Array(fixed);
  const dv = new DataView(b.buffer);

  let recon = 0;
  const prevQ = new Array<number>(desc.poseFields.length).fill(0);
  if (hasDx) {
    const dxIdx = desc.poseFields.find((f) => f.enc === 'dx32')?.idx ?? 0;
    recon = rows[7 + dxIdx] ?? 0;
    dv.setFloat64(baseOff, recon, true);
  }
  for (let i = 0; i < live; i++) {
    const row = i * RECORD_FLOATS;
    let esc = false;
    let hdr = 0;
    const doneV = rows[row + 6] ?? 0;
    if (doneV === 1) hdr |= 0x10;
    else if (doneV !== 0) esc = true;
    if (nBox === 0) {
      const a = rows[row] ?? 0;
      if (a === Math.round(a) && a >= 0 && a <= 15) hdr |= a & 0x0f;
      else esc = true;
    } else {
      for (let c = 0; c < nBox; c++) {
        const v = rows[row + c] ?? 0;
        const q = roundHalfAwayFromZero((Math.min(Math.max(v, -1), 1) + 1) * 127.5);
        b[live * (1 + c) + i] = Math.min(Math.max(q, 0), 255);
      }
    }
    for (let j = 0; j < desc.poseFields.length; j++) {
      const f = desc.poseFields[j];
      if (f === undefined) continue;
      const v = rows[row + 7 + f.idx] ?? 0;
      if (f.enc === 'u8') {
        if (!(v === Math.round(v) && v >= 0 && v <= 255)) esc = true;
        b[(off[j] ?? 0) + i] = Math.min(Math.max(Math.round(v), 0), 255);
      } else if (f.enc === 'i16') {
        const scale = f.scale ?? 1;
        let w = v;
        if (opts.wrap === true && f.wrap === true) w = wrapAngle(w);
        let q = roundHalfAwayFromZero(w / scale);
        if (!(q >= -32767 && q <= 32767)) esc = true;
        q = Math.min(Math.max(q, -32767), 32767);
        const delta = i === 0 ? q : q - (prevQ[j] ?? 0);
        prevQ[j] = q;
        dv.setInt16((off[j] ?? 0) + 2 * i, (delta << 16) >> 16, true);
      } else {
        const scale = f.scale ?? 1;
        let q = roundHalfAwayFromZero((v - recon) / scale);
        if (!(q >= -2147483647 && q <= 2147483647)) esc = true;
        q = Math.min(Math.max(q, -2147483647), 2147483647);
        recon += q * scale;
        dv.setInt32((off[j] ?? 0) + 4 * i, q, true);
      }
    }
    if (esc) {
      hdr |= 0x20;
      escRows.push(i);
    }
    b[i] = hdr;
  }
  dv.setUint32(fixed - 4, escRows.length, true);

  const payload = new Uint8Array(fixed + escRows.length * 68);
  payload.set(b, 0);
  const rowBytes = new Uint8Array(rows.buffer, rows.byteOffset, rows.byteLength);
  const pv = new DataView(payload.buffer);
  for (let e = 0; e < escRows.length; e++) {
    const i = escRows[e] ?? 0;
    const at = fixed + e * 68;
    pv.setUint32(at, i, true);
    payload.set(rowBytes.subarray(i * 64, i * 64 + 64), at + 4);
  }
  return { payload, live, closed };
}

export function encodeRecord(
  envId: string,
  data: Float32Array,
  steps: number,
  chunkSteps = 1024,
  opts: { wrap?: boolean } = {},
): EncodeResult {
  const envKind = envKindOf(envId);
  if (envKind < 0) throw new Error(`no codec for ${envId}`);
  const desc = codecForEnvKind(envKind);
  const parts: Uint8Array[] = [];
  const head = new Uint8Array(12);
  const hv = new DataView(head.buffer);
  hv.setUint32(0, FILE_MAGIC, true);
  hv.setUint32(4, envKind, true);
  parts.push(head);

  let written = 0;
  let closed = false;
  let start = 0;
  while (start < steps && !closed) {
    const count = Math.min(chunkSteps, steps - start);
    const rows = data.subarray(start * RECORD_FLOATS, (start + count) * RECORD_FLOATS);
    const enc = encodeChunkPayload(new Float32Array(rows), count, desc, opts);
    closed = enc.closed;
    if (enc.live > 0) {
      const comp = deflateRawSync(enc.payload);
      const out = new Uint8Array(24 + comp.byteLength);
      const ov = new DataView(out.buffer);
      ov.setUint32(0, CHUNK_MAGIC, true);
      ov.setUint32(4, written, true);
      ov.setUint32(8, enc.live, true);
      ov.setUint32(12, enc.payload.byteLength, true);
      ov.setUint32(16, comp.byteLength, true);
      ov.setUint32(20, crc32(comp), true);
      out.set(comp, 24);
      parts.push(out);
      written += enc.live;
    }
    start += count;
  }
  hv.setUint32(8, written, true);

  let total = 0;
  for (const p of parts) total += p.byteLength;
  const bytes = new Uint8Array(total);
  let at = 0;
  for (const p of parts) {
    bytes.set(p, at);
    at += p.byteLength;
  }
  return { bytes, steps: written, closed };
}
