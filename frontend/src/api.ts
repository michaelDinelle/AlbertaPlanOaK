import type { RunDetail, RunIndexEntry } from '@shared/types';

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

export const api = {
  runs(options?: RequestOptions): Promise<RunIndexEntry[]> {
    return getJson<RunIndexEntry[]>(apiUrl('/runs'), options);
  },

  run(id: string, options?: RequestOptions): Promise<RunDetail> {
    return getJson<RunDetail>(apiUrl(`/runs/${encodeURIComponent(id)}`), options);
  },
};

export function recordingUrl(runId: string, envId: string, seed: number): string {
  return apiUrl(`/runs/${encodeURIComponent(runId)}/recordings/${encodeURIComponent(envId)}/${seed}`);
}
