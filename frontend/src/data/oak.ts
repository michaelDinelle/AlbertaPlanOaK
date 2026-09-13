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

export function fetchRewards(id: string, envId: string, from: number, n: number, signal?: AbortSignal): Promise<Float32Array> {
  return getFloats(apiUrl(`/runs/${encodeURIComponent(id)}/rewards/${encodeURIComponent(envId)}?from=${from}&n=${n}`), signal);
}

export interface FollowHandlers {
  columns?(names: string[]): void;
  window?(index: number, end: number, wall: number, values: Float32Array): void;
  frames?(lane: number, from: number, values: Float32Array): void;
}

interface Subscription {
  id: string;
  lane: number;
  handlers: FollowHandlers;
}

const subscriptions = new Set<Subscription>();
let socket: WebSocket | null = null;
let reconnectMs = 500;

function socketUrl(): string {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${scheme}://${location.host}${apiUrl('/ws')}`;
}

function sendFollow(sub: Subscription): void {
  if (socket !== null && socket.readyState === WebSocket.OPEN) socket.send(`follow ${sub.id} ${sub.lane}`);
}

function dispatch(data: ArrayBuffer): void {
  const view = new DataView(data);
  const kind = view.getUint8(0);
  const idLen = view.getUint16(1, true);
  const id = new TextDecoder().decode(new Uint8Array(data, 3, idLen));
  let at = 3 + idLen;
  if (kind === 1) {
    const index = view.getUint32(at, true);
    const end = view.getUint32(at + 4, true);
    const wall = view.getFloat32(at + 8, true);
    const values = new Float32Array(data.slice(at + 12));
    for (const sub of subscriptions) if (sub.id === id) sub.handlers.window?.(index, end, wall, values);
  } else if (kind === 2) {
    const lane = view.getUint32(at, true);
    const from = view.getUint32(at + 4, true);
    const values = new Float32Array(data.slice(at + 8));
    for (const sub of subscriptions) if (sub.id === id && sub.lane === lane) sub.handlers.frames?.(lane, from, values);
  }
}

function ensureSocket(): void {
  if (socket !== null || subscriptions.size === 0) return;
  const ws = new WebSocket(socketUrl());
  ws.binaryType = 'arraybuffer';
  socket = ws;
  ws.onopen = () => {
    reconnectMs = 500;
    for (const sub of subscriptions) sendFollow(sub);
  };
  ws.onmessage = (event: MessageEvent<ArrayBuffer | string>) => {
    if (typeof event.data === 'string') {
      const [verb, id, rest] = event.data.split(' ', 3);
      if (verb === 'columns' && id !== undefined && rest !== undefined) {
        const names = rest.split(',');
        for (const sub of subscriptions) if (sub.id === id) sub.handlers.columns?.(names);
      }
      return;
    }
    dispatch(event.data);
  };
  ws.onclose = () => {
    socket = null;
    if (subscriptions.size === 0) return;
    setTimeout(ensureSocket, reconnectMs);
    reconnectMs = Math.min(reconnectMs * 2, 10000);
  };
  ws.onerror = () => ws.close();
}

export function follow(id: string, lane: number, handlers: FollowHandlers): () => void {
  const sub: Subscription = { id, lane, handlers };
  subscriptions.add(sub);
  ensureSocket();
  sendFollow(sub);
  return () => {
    subscriptions.delete(sub);
    const stillFollowed = [...subscriptions].some((s) => s.id === id && s.lane === lane);
    if (!stillFollowed && socket !== null && socket.readyState === WebSocket.OPEN) socket.send(`unfollow ${id} ${lane}`);
    if (subscriptions.size === 0 && socket !== null) {
      socket.close();
      socket = null;
    }
  };
}
