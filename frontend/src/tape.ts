import { TAPE_MAGIC } from '@shared/types';
import { ApiError, apiUrl } from './api';

export interface Tape {
  laneCount: number;
  fields: string[];
  rows: number;
  stepStart: Float64Array;
  stepEnd: Float64Array;
  wallSeconds: Float32Array;
  data: Float32Array;
  headerBytes: number;
  rowBytes: number;
  byteLength: number;
  etag: string | null;
}

export function fieldIndex(tape: Tape, field: string): number {
  const i = tape.fields.indexOf(field);
  if (i < 0) throw new Error(`tape has no field ${field}`);
  return i;
}

export function tapeValue(tape: Tape, fieldIdx: number, row: number, lane: number): number {
  return tape.data[(row * tape.fields.length + fieldIdx) * tape.laneCount + lane] ?? NaN;
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

export function parseTape(bytes: Uint8Array, etag: string | null = null): Tape {
  const { laneCount, fields, headerBytes } = parseHeader(bytes);
  const rowBytes = 12 + fields.length * laneCount * 4;
  const empty: Tape = {
    laneCount,
    fields,
    rows: 0,
    stepStart: new Float64Array(0),
    stepEnd: new Float64Array(0),
    wallSeconds: new Float32Array(0),
    data: new Float32Array(0),
    headerBytes,
    rowBytes,
    byteLength: headerBytes,
    etag,
  };
  return appendRows(empty, bytes.subarray(headerBytes), etag);
}

function appendRows(tape: Tape, more: Uint8Array, etag: string | null): Tape {
  const added = Math.floor(more.byteLength / tape.rowBytes);
  if (added <= 0) return { ...tape, etag };
  const rows = tape.rows + added;
  const next: Tape = {
    ...tape,
    rows,
    stepStart: new Float64Array(rows),
    stepEnd: new Float64Array(rows),
    wallSeconds: new Float32Array(rows),
    data: new Float32Array(rows * tape.fields.length * tape.laneCount),
    byteLength: tape.byteLength + added * tape.rowBytes,
    etag,
  };
  next.stepStart.set(tape.stepStart);
  next.stepEnd.set(tape.stepEnd);
  next.wallSeconds.set(tape.wallSeconds);
  next.data.set(tape.data);
  const dv = new DataView(more.buffer, more.byteOffset, more.byteLength);
  const valueFloats = next.fields.length * next.laneCount;
  for (let a = 0; a < added; a++) {
    const r = tape.rows + a;
    const at = a * next.rowBytes;
    const start = dv.getUint32(at, true);
    const count = dv.getUint32(at + 4, true);
    next.stepStart[r] = start;
    next.stepEnd[r] = start + count;
    next.wallSeconds[r] = dv.getFloat32(at + 8, true);
    for (let i = 0; i < valueFloats; i++) {
      next.data[r * valueFloats + i] = dv.getFloat32(at + 12 + i * 4, true);
    }
  }
  return next;
}

export async function fetchTape(runId: string, prev: Tape | null, signal?: AbortSignal): Promise<Tape> {
  const url = apiUrl(`/runs/${encodeURIComponent(runId)}/tape`);
  const headers: Record<string, string> = { accept: 'application/octet-stream' };
  if (prev !== null) {
    headers['range'] = `bytes=${prev.byteLength}-`;
    if (prev.etag !== null) {
      headers['if-range'] = prev.etag;
      headers['if-none-match'] = prev.etag;
    }
  }
  let response: Response;
  try {
    response = await fetch(url, { headers, ...(signal ? { signal } : {}) });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError(0, url, cause instanceof Error ? cause.message : String(cause));
  }
  if (response.status === 304 || response.status === 416) {
    if (prev !== null) return prev;
    throw new ApiError(response.status, url, 'empty tape');
  }
  if (!response.ok && response.status !== 206) {
    throw new ApiError(response.status, url, await response.text().catch(() => ''));
  }
  const etag = response.headers.get('etag');
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (response.status === 206 && prev !== null) {
    return appendRows(prev, bytes, etag);
  }
  return parseTape(bytes, etag);
}
