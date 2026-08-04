// Typed fetch client over the FROZEN DTOs in shared/types.ts.
//
// Everything here speaks the contract, not the server: if an endpoint is missing
// or half-built, callers get a typed ApiError they can render instead of a crash.
// URL building is a pure exported function so it can be tested without a network.

import type {
  AggregateRow,
  FleetColumnar,
  LiveHello,
  LiveStatus,
  RolesResponse,
  RunDetail,
  RunIndexEntry,
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

  roles(options?: RequestOptions): Promise<RolesResponse> {
    return getJson<RolesResponse>(apiUrl('/roles'), options);
  },
};

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
