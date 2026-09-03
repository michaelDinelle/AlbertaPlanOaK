import { TAPE_MAGIC } from '@shared/types';
import { ApiError, tapeUrl } from '../data/api';

export interface Tape {
  laneCount: number;
  fields: string[];
  rows: number;
  stepStart: Float64Array;
  stepEnd: Float64Array;
  wallSeconds: Float32Array;
  data: Float32Array;
  bits: Uint32Array;
  headerBytes: number;
  byteLength: number;
}

export function fieldIndex(tape: Tape, field: string): number {
  const i = tape.fields.indexOf(field);
  if (i < 0) throw new Error(`tape has no field ${field}`);
  return i;
}

export function tapeValue(tape: Tape, fieldIdx: number, row: number, lane: number): number {
  return tape.data[(row * tape.fields.length + fieldIdx) * tape.laneCount + lane] ?? NaN;
}

export function tapeStep(tape: Tape, fieldIdx: number, row: number, lane: number): number {
  const at = (row * tape.fields.length + fieldIdx) * tape.laneCount + lane;
  const marker = tape.data[at];
  if (marker === undefined || Number.isNaN(marker)) return NaN;
  return tape.bits[at] ?? NaN;
}

function parseHeader(bytes: Uint8Array): { laneCount: number; fields: string[]; headerBytes: number } {
  if (bytes.byteLength < 12) throw new Error('tape is shorter than its header');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== TAPE_MAGIC) throw new Error('not a tape (bad magic)');
  const laneCount = dv.getUint32(4, true);
  const jsonLen = dv.getUint32(8, true);
  if (12 + jsonLen > bytes.byteLength) throw new Error('tape header is torn');
  const parsed = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + jsonLen))) as {
    fields?: unknown;
  };
  if (!Array.isArray(parsed.fields) || parsed.fields.some((f) => typeof f !== 'string')) {
    throw new Error('tape header has no fields list');
  }
  return { laneCount, fields: parsed.fields as string[], headerBytes: 12 + jsonLen };
}

export async function inflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function parseTape(bytes: Uint8Array): Promise<Tape> {
  const { laneCount, fields, headerBytes } = parseHeader(bytes);
  const empty: Tape = {
    laneCount,
    fields,
    rows: 0,
    stepStart: new Float64Array(0),
    stepEnd: new Float64Array(0),
    wallSeconds: new Float32Array(0),
    data: new Float32Array(0),
    bits: new Uint32Array(0),
    headerBytes,
    byteLength: headerBytes,
  };
  return appendRows(empty, bytes.subarray(headerBytes));
}

export async function appendRows(tape: Tape, more: Uint8Array): Promise<Tape> {
  const dv = new DataView(more.buffer, more.byteOffset, more.byteLength);
  const valuesPerRow = tape.fields.length * tape.laneCount;
  const blockBytes = valuesPerRow * 4;
  const heads: { start: number; count: number; wall: number }[] = [];
  const blocks: Promise<Uint8Array>[] = [];
  let at = 0;
  while (at + 16 <= more.byteLength) {
    const compLen = dv.getUint32(at + 12, true);
    if (at + 16 + compLen > more.byteLength) break;
    heads.push({
      start: dv.getUint32(at, true),
      count: dv.getUint32(at + 4, true),
      wall: dv.getFloat32(at + 8, true),
    });
    blocks.push(inflateRaw(more.subarray(at + 16, at + 16 + compLen)));
    at += 16 + compLen;
  }
  if (heads.length === 0) return tape;
  const inflated = await Promise.all(blocks);
  const rows = tape.rows + heads.length;
  const data = new Float32Array(rows * valuesPerRow);
  data.set(tape.data.subarray(0, tape.rows * valuesPerRow));
  const next: Tape = {
    ...tape,
    rows,
    stepStart: new Float64Array(rows),
    stepEnd: new Float64Array(rows),
    wallSeconds: new Float32Array(rows),
    data,
    bits: new Uint32Array(data.buffer),
    byteLength: tape.byteLength + at,
  };
  next.stepStart.set(tape.stepStart.subarray(0, tape.rows));
  next.stepEnd.set(tape.stepEnd.subarray(0, tape.rows));
  next.wallSeconds.set(tape.wallSeconds.subarray(0, tape.rows));
  heads.forEach((head, a) => {
    const r = tape.rows + a;
    next.stepStart[r] = head.start;
    next.stepEnd[r] = head.start + head.count;
    next.wallSeconds[r] = head.wall;
    const block = inflated[a]!;
    if (block.byteLength !== blockBytes) {
      throw new Error(`tape row block is ${block.byteLength} bytes, expected ${blockBytes}`);
    }
    next.data.set(new Float32Array(block.buffer, 0, valuesPerRow), r * valuesPerRow);
  });
  return next;
}

export async function fetchTape(runId: string, prev: Tape | null, signal?: AbortSignal): Promise<Tape> {
  const url = tapeUrl(runId);
  const headers: Record<string, string> = { accept: 'application/octet-stream' };
  if (prev !== null) headers['range'] = `bytes=${prev.byteLength}-`;
  let response: Response;
  try {
    response = await fetch(url, { headers, ...(signal ? { signal } : {}) });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError(0, url, cause instanceof Error ? cause.message : String(cause));
  }
  if (response.status === 204 || response.status === 416) {
    if (prev !== null) return prev;
    throw new ApiError(response.status, url, 'empty tape');
  }
  if (!response.ok && response.status !== 206) {
    throw new ApiError(response.status, url, await response.text().catch(() => ''));
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (response.status === 206 && prev !== null) {
    const next = await appendRows(prev, bytes);
    if (next.rows > prev.rows
      && next.stepStart[prev.rows] !== (prev.rows > 0 ? prev.stepEnd[prev.rows - 1] : 0)) {
      return fetchTape(runId, null, signal);
    }
    return next;
  }
  return parseTape(bytes);
}
