import type { RunDetail, RunIndexEntry } from '@shared/types';

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

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      headers: { accept: 'application/json' },
      ...(signal ? { signal } : {}),
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

export function fetchRuns(signal?: AbortSignal): Promise<RunIndexEntry[]> {
  return getJson<RunIndexEntry[]>(apiUrl('/runs'), signal);
}

export function fetchRunDetail(id: string, signal?: AbortSignal): Promise<RunDetail> {
  return getJson<RunDetail>(apiUrl(`/runs/${encodeURIComponent(id)}`), signal);
}

export function tapeUrl(runId: string): string {
  return apiUrl(`/runs/${encodeURIComponent(runId)}/tape`);
}

export function rewardUrl(runId: string, envId: string, part: 'pack' | 'idx'): string {
  return apiUrl(`/runs/${encodeURIComponent(runId)}/reward/${encodeURIComponent(envId)}/${part}`);
}

export function recordingUrl(runId: string, envId: string, seed: number): string {
  return apiUrl(`/runs/${encodeURIComponent(runId)}/recordings/${encodeURIComponent(envId)}/${seed}`);
}
