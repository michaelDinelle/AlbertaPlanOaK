import type { RunDetail, RunIndexEntry } from './types';

export const FRAME_FLOATS = 11;
export const OFF_DONE = 0;
export const OFF_POSE = 1;
export const POSE_LEN = 9;

export class ApiError extends Error {
  readonly status: number;
  readonly url: string;
  readonly body: string;

  constructor(status: number, url: string, body: string) {
    super(`${status || 'network'} ${url}${body ? ` — ${body.slice(0, 200)}` : ''}`);
    this.name = 'ApiError';
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

export function apiUrl(path: string): string {
  return `/api${path.startsWith('/') ? path : `/${path}`}`;
}

async function request(url: string, signal?: AbortSignal): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, signal ? { signal } : {});
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError(0, url, cause instanceof Error ? cause.message : String(cause));
  }
  if (!response.ok && response.status !== 204) {
    throw new ApiError(response.status, url, await response.text().catch(() => ''));
  }
  return response;
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await request(url, signal);
  try {
    return (await response.json()) as T;
  } catch (cause) {
    throw new ApiError(response.status, url, `invalid JSON: ${String(cause)}`);
  }
}

async function getFloats(url: string, signal?: AbortSignal): Promise<Float32Array> {
  const response = await request(url, signal);
  if (response.status === 204) return new Float32Array(0);
  return new Float32Array(await response.arrayBuffer());
}

export function fetchRuns(signal?: AbortSignal): Promise<RunIndexEntry[]> {
  return getJson<RunIndexEntry[]>(apiUrl('/runs'), signal);
}

export function fetchRunDetail(id: string, signal?: AbortSignal): Promise<RunDetail> {
  return getJson<RunDetail>(apiUrl(`/runs/${encodeURIComponent(id)}`), signal);
}

export interface Series {
  windows: number;
  lanes: number;
  ends: Float64Array;
  wall: Float32Array;
  values: Float32Array;
}

export async function fetchSeries(id: string, column: string, signal?: AbortSignal): Promise<Series> {
  const response = await request(apiUrl(`/runs/${encodeURIComponent(id)}/series/${encodeURIComponent(column)}`), signal);
  const bytes = await response.arrayBuffer();
  const view = new DataView(bytes);
  const windows = view.getUint32(0, true);
  const lanes = view.getUint32(4, true);
  const ends = new Float64Array(windows);
  for (let w = 0; w < windows; w++) ends[w] = view.getUint32(8 + 4 * w, true);
  const wall = new Float32Array(bytes.slice(8 + 4 * windows, 8 + 8 * windows));
  const values = new Float32Array(bytes.slice(8 + 8 * windows));
  return { windows, lanes, ends, wall, values };
}

export function fetchFrames(id: string, lane: number, from: number, n: number, signal?: AbortSignal): Promise<Float32Array> {
  return getFloats(apiUrl(`/runs/${encodeURIComponent(id)}/frames/${lane}?from=${from}&n=${n}`), signal);
}
