// Typed fetch client over the FROZEN DTOs in shared/types.ts.
//
// Everything here speaks the contract, not the server: if an endpoint is missing
// or half-built, callers get a typed ApiError they can render instead of a crash.
// URL building is a pure exported function so it can be tested without a network.

import type {
  AggregateRow,
  CurvesResponse,
  FleetColumnar,
  LiveHello,
  LiveStatus,
  RolesResponse,
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

/**
 * Build an /api URL. Empty, null and undefined values are dropped; arrays join
 * with commas (the server-side convention for `fields=a,b,c`).
 */
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
  /** columnar fields to materialize, e.g. ['health.rewardRate'] */
  fields?: readonly string[];
  /** restrict to one env's lane range (mixed-env runs) */
  envId?: string;
  /**
   * Which policy's lanes to materialize. The server filters fleet.jsonl by this
   * and defaults to `oak`, so a multi-policy run MUST pass the policy it is
   * showing or it silently reads the wrong lanes.
   */
  policy?: string;
}

export interface CurvesQuery {
  /** restrict to one env's lanes (mixed-env runs); absent = every lane, pooled */
  envId?: string;
  /** which policy's seeds the curve is built from; the server defaults to `oak` */
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

  /**
   * Learning curve: per-heartbeat IQM ± bootstrap CI and P10/P50/P90 of the MEASURED
   * reward rate. Derived server-side from fleet.jsonl, so it answers on legacy roots
   * that never got a summary.json — and `envId` keeps a mixed fleet's environments on
   * their own scales instead of pooling them.
   */
  curves(id: string, query?: CurvesQuery, options?: RequestOptions): Promise<CurvesResponse> {
    return getJson<CurvesResponse>(
      apiUrl(`/runs/${encodeURIComponent(id)}/curves`, {
        envId: query?.envId,
        policy: query?.policy,
      }),
      options,
    );
  },

  roles(options?: RequestOptions): Promise<RolesResponse> {
    return getJson<RolesResponse>(apiUrl('/roles'), options);
  },
};

// ---------- /api/runs/:id/wall/:envId ----------

/** pose[9] f32 + done u8, tightly packed — the wire record (`WallResponse` framing). */
export const WALL_SAMPLE_BYTES = 37;
/** pose floats per sample */
export const WALL_POSE_LEN = 9;

export interface WallLane {
  seed: number;
  /** rows these samples COVER (a torn tail or a budget stop yields fewer than meta claims) */
  steps: number;
  samples: number;
  /**
   * First record row whose pose/done went non-finite; null when no scanned row did.
   * Exact over the rows the server decoded — when `scannedRows` is set, the death is
   * dated to within one v4 chunk rather than to the row.
   */
  firstNonFiniteRow: number | null;
  /** rows the server decoded; present only when it skipped whole chunks (< `steps`) */
  scannedRows?: number;
  /** the server stopped this seed early to stay inside its budget; `steps` is the span shown */
  truncated?: true;
  /** present only when the server could not decode this lane */
  error?: string;
  /** `samples × 9`, row-major — sample s is `pose.subarray(s*9, s*9+9)` */
  pose: Float32Array;
  /** `samples` done bits */
  done: Uint8Array;
  /** record row index each sample was taken at: `s·stride`, last entry the final row */
  rows: Int32Array;
}

export interface WallData {
  header: WallResponse;
  lanes: WallLane[];
}

export interface WallQuery {
  /**
   * Rows per sample; the server clamps to [1, 4096] and defaults to 64. It is a FLOOR:
   * a run big enough to blow the server's payload budget comes back coarser, so read
   * `header.stride` (and each lane's `rows`) instead of assuming this value was used.
   */
  stride?: number;
  /** accepted here as well as in `options`, since a wall fetch is the thing callers cancel */
  signal?: AbortSignal;
}

export function wallUrl(id: string, envId: string, query?: WallQuery): string {
  return apiUrl(`/runs/${encodeURIComponent(id)}/wall/${encodeURIComponent(envId)}`, {
    stride: query?.stride,
  });
}

async function inflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
  // A fresh ArrayBuffer-backed copy: `BufferSource` excludes SharedArrayBuffer-backed
  // views, and a stream is free to hold or neuter what it is handed.
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

/**
 * Fetch + decode one env's wall.
 *
 * The response is a JSON header, a newline, and a deflate-raw section (see
 * `WallResponse` in shared/types.ts). The split is taken at `headerBytes` and CHECKED
 * against the first newline: the two agreeing is what proves the body arrived whole,
 * and a disagreement is a corrupted response, not something to paper over.
 *
 * The wire packs pose and done together at a 37-byte stride, which leaves every f32 at
 * an unaligned offset. They are de-interleaved here into an aligned `Float32Array` per
 * lane — the shape a renderer can hand straight to a buffer attribute — at the cost of
 * one pass over data that was already going to be walked once.
 */
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
      // The server samples `0, stride, 2·stride, …` and then appends the final row
      // when the stride missed it — so only the LAST sample can be off-grid.
      const strided = s * stride;
      rows[s] = s === n - 1 && lane.steps > 0 ? lane.steps - 1 : strided;
    }
    off += n * WALL_SAMPLE_BYTES;
    lanes.push({
      seed: lane.seed,
      steps: lane.steps,
      samples: n,
      firstNonFiniteRow: lane.firstNonFiniteRow,
      // Both are absent on a fully decoded lane, so they stay absent here too.
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

// ---------- SSE: /api/runs/:id/live ----------

export interface LiveHandlers {
  open?(): void;
  /** initial replay of everything already on disk */
  hello?(payload: LiveHello): void;
  /** one appended aggregate heartbeat */
  row?(row: AggregateRow): void;
  status?(status: LiveStatus): void;
  /** stream ended cleanly (run reached its horizon) */
  done?(): void;
  /** transport error; the browser retries on its own unless we close() */
  error?(error: Event): void;
}

export interface LiveConnection {
  readonly url: string;
  /** true once close() ran or the stream ended */
  readonly closed: boolean;
  close(): void;
}

const HELLO_EVENTS = ['hello', 'init', 'snapshot'] as const;
const ROW_EVENTS = ['agg', 'aggregate', 'row', 'heartbeat', 'hb'] as const;
const STATUS_EVENTS = ['status', 'state'] as const;
const DONE_EVENTS = ['done', 'complete', 'end'] as const;

function parse<T>(raw: string): T | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function liveUrl(id: string): string {
  return apiUrl(`/runs/${encodeURIComponent(id)}/live`);
}

/**
 * EventSource wrapper with typed handlers. Event names are not pinned by the
 * frozen contract, so every plausible name is bound and the default `message`
 * channel additionally accepts a `{type, data}` envelope.
 */
export function openLive(id: string, handlers: LiveHandlers): LiveConnection {
  const url = liveUrl(id);
  let closed = false;

  const connection: LiveConnection = {
    url,
    get closed() {
      return closed;
    },
    close() {
      if (closed) return;
      closed = true;
      source?.close();
    },
  };

  if (typeof EventSource === 'undefined') {
    closed = true;
    return connection;
  }

  const source = new EventSource(url);

  const dispatch = (type: string, raw: string): void => {
    if (closed) return;
    if ((HELLO_EVENTS as readonly string[]).includes(type)) {
      const payload = parse<LiveHello>(raw);
      if (payload) handlers.hello?.(payload);
    } else if ((ROW_EVENTS as readonly string[]).includes(type)) {
      const payload = parse<AggregateRow | { rows?: AggregateRow[] }>(raw);
      if (!payload) return;
      const rows = 'rows' in payload && Array.isArray(payload.rows) ? payload.rows : [payload as AggregateRow];
      for (const row of rows) if (row && typeof row.step === 'number') handlers.row?.(row);
    } else if ((STATUS_EVENTS as readonly string[]).includes(type)) {
      const payload = parse<LiveStatus>(raw);
      if (payload) handlers.status?.(payload);
    } else if ((DONE_EVENTS as readonly string[]).includes(type)) {
      handlers.done?.();
      connection.close();
    }
  };

  source.addEventListener('open', () => handlers.open?.());
  source.addEventListener('error', (event) => {
    if (!closed) handlers.error?.(event);
  });

  for (const name of [...HELLO_EVENTS, ...ROW_EVENTS, ...STATUS_EVENTS, ...DONE_EVENTS]) {
    source.addEventListener(name, (event) => dispatch(name, (event as MessageEvent<string>).data ?? ''));
  }

  source.addEventListener('message', (event) => {
    const raw = (event as MessageEvent<string>).data ?? '';
    const envelope = parse<{ type?: string; event?: string; data?: unknown; payload?: unknown }>(raw);
    if (!envelope) return;
    const type = envelope.type ?? envelope.event;
    const inner = envelope.data ?? envelope.payload;
    if (type && inner !== undefined) {
      dispatch(type, JSON.stringify(inner));
    } else if (type) {
      dispatch(type, raw);
    } else if (envelope && typeof (envelope as unknown as AggregateRow).step === 'number') {
      handlers.row?.(envelope as unknown as AggregateRow);
    }
  });

  return connection;
}
