
import type {
  FleetColumnar,
  KeeperResponse,
  RunDetail,
  RunIndexEntry,
  WallResponse,
} from '@shared/types';

export const API_BASE = '/api';

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

export type QueryValue = string | number | boolean | readonly string[] | null | undefined;

export function apiUrl(path: string, query?: Record<string, QueryValue>): string {
  const rel = path.startsWith('/') ? path : `/${path}`;
  const base = rel.startsWith(`${API_BASE}/`) || rel === API_BASE ? rel : `${API_BASE}${rel}`;
  if (!query) return base;
  const parts: string[] = [];
  for (const key of Object.keys(query)) {
    const raw = query[key];
    if (raw === undefined || raw === null) continue;
    const value = Array.isArray(raw) ? raw.join(',') : String(raw);
    if (value === '') continue;
    parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
  }
  return parts.length > 0 ? `${base}?${parts.join('&')}` : base;
}

export interface RequestOptions {
  signal?: AbortSignal;
}

async function getJson<T>(url: string, options?: RequestOptions): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      headers: { accept: 'application/json' },
      ...(options?.signal ? { signal: options.signal } : {}),
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError(0, url, cause instanceof Error ? cause.message : String(cause));
  }
  if (!response.ok) {
    throw new ApiError(response.status, url, await response.text().catch(() => ''));
  }
  try {
    return (await response.json()) as T;
  } catch (cause) {
    throw new ApiError(response.status, url, `invalid JSON: ${String(cause)}`);
  }
}

export interface FleetQuery {
  fields?: readonly string[];
  envId?: string;
  policy?: string;
}

export const api = {
  runs(options?: RequestOptions): Promise<RunIndexEntry[]> {
    return getJson<RunIndexEntry[]>(apiUrl('/runs'), options);
  },

  run(id: string, options?: RequestOptions): Promise<RunDetail> {
    return getJson<RunDetail>(apiUrl(`/runs/${encodeURIComponent(id)}`), options);
  },

  fleet(id: string, query?: FleetQuery, options?: RequestOptions): Promise<FleetColumnar> {
    return getJson<FleetColumnar>(
      apiUrl(`/runs/${encodeURIComponent(id)}/fleet`, {
        fields: query?.fields,
        envId: query?.envId,
        policy: query?.policy,
      }),
      options,
    );
  },

  keeper(options?: RequestOptions): Promise<KeeperResponse> {
    return getJson<KeeperResponse>(apiUrl('/keeper'), options);
  },
};


export const WALL_SAMPLE_BYTES = 37;
export const WALL_POSE_LEN = 9;

export interface WallLane {
  seed: number;
  steps: number;
  samples: number;
  firstNonFiniteRow: number | null;
  scannedRows?: number;
  truncated?: true;
  error?: string;
  pose: Float32Array;
  done: Uint8Array;
  rows: Int32Array;
}

export interface WallData {
  header: WallResponse;
  lanes: WallLane[];
}

export interface WallQuery {
  stride?: number;
  signal?: AbortSignal;
}

export function wallUrl(id: string, envId: string, query?: WallQuery): string {
  return apiUrl(`/runs/${encodeURIComponent(id)}/wall/${encodeURIComponent(envId)}`, {
    stride: query?.stride,
  });
}

async function inflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(copy);
      controller.close();
    },
  });
  return new Uint8Array(
    await new Response(source.pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer(),
  );
}

export async function fetchWall(
  id: string,
  envId: string,
  query?: WallQuery,
  options?: RequestOptions,
): Promise<WallData> {
  const url = wallUrl(id, envId, query);
  const signal = options?.signal ?? query?.signal;
  let response: Response;
  try {
    response = await fetch(url, {
      headers: { accept: 'application/octet-stream' },
      ...(signal ? { signal } : {}),
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError(0, url, cause instanceof Error ? cause.message : String(cause));
  }
  if (!response.ok) {
    throw new ApiError(response.status, url, await response.text().catch(() => ''));
  }

  const body = new Uint8Array(await response.arrayBuffer());
  const nl = body.indexOf(0x0a);
  if (nl < 0) throw new ApiError(response.status, url, 'wall body has no header terminator');

  let header: WallResponse;
  try {
    header = JSON.parse(new TextDecoder().decode(body.subarray(0, nl))) as WallResponse;
  } catch (cause) {
    throw new ApiError(response.status, url, `invalid wall header: ${String(cause)}`);
  }
  if (header.v !== 1) {
    throw new ApiError(response.status, url, `unsupported wall version ${String(header.v)}`);
  }
  if (header.headerBytes !== nl) {
    throw new ApiError(
      response.status,
      url,
      `wall headerBytes ${header.headerBytes} disagrees with the terminator at ${nl}`,
    );
  }
  const compressed = body.subarray(nl + 1);
  if (header.binaryBytes !== compressed.byteLength) {
    throw new ApiError(
      response.status,
      url,
      `wall binaryBytes ${header.binaryBytes} but ${compressed.byteLength} bytes followed`,
    );
  }

  const lanesIn = Array.isArray(header.lanes) ? header.lanes : [];
  let wanted = 0;
  for (const lane of lanesIn) wanted += lane.samples * WALL_SAMPLE_BYTES;

  const flat = compressed.byteLength === 0 ? new Uint8Array(0) : await inflateRaw(compressed);
  if (flat.byteLength !== wanted) {
    throw new ApiError(
      response.status,
      url,
      `wall section inflated to ${flat.byteLength} bytes, header describes ${wanted}`,
    );
  }
  const view = new DataView(flat.buffer, flat.byteOffset, flat.byteLength);

  const stride = header.stride > 0 ? header.stride : 1;
  const lanes: WallLane[] = [];
  let off = 0;
  for (const lane of lanesIn) {
    const n = lane.samples;
    const pose = new Float32Array(n * WALL_POSE_LEN);
    const done = new Uint8Array(n);
    const rows = new Int32Array(n);
    for (let s = 0; s < n; s++) {
      const base = off + s * WALL_SAMPLE_BYTES;
      for (let j = 0; j < WALL_POSE_LEN; j++) {
        pose[s * WALL_POSE_LEN + j] = view.getFloat32(base + j * 4, true);
      }
      done[s] = flat[base + WALL_POSE_LEN * 4] ?? 0;
      const strided = s * stride;
      rows[s] = s === n - 1 && lane.steps > 0 ? lane.steps - 1 : strided;
    }
    off += n * WALL_SAMPLE_BYTES;
    lanes.push({
      seed: lane.seed,
      steps: lane.steps,
      samples: n,
      firstNonFiniteRow: lane.firstNonFiniteRow,
      ...(lane.scannedRows !== undefined ? { scannedRows: lane.scannedRows } : {}),
      ...(lane.truncated === true ? { truncated: true as const } : {}),
      ...(lane.error !== undefined ? { error: lane.error } : {}),
      pose,
      done,
      rows,
    });
  }

  return { header, lanes };
}

