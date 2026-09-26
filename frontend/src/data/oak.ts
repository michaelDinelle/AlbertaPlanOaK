import type { EnvDescriptor } from './envs';
import { unpackRecording, type Recording, type RunIndexEntry } from './layout';

async function api(path: string): Promise<Response> {
  const response = await fetch(`/api${path}`);
  if (!response.ok) throw new Error(`${response.status} ${path}: ${await response.text()}`);
  return response;
}

export async function fetchRuns(): Promise<RunIndexEntry[]> {
  return (await api('/runs')).json() as Promise<RunIndexEntry[]>;
}

export async function fetchRecording(id: string): Promise<Recording> {
  return unpackRecording(id, await (await api(`/runs/${encodeURIComponent(id)}`)).arrayBuffer());
}

export async function fetchCatalog(): Promise<EnvDescriptor[]> { return (await api('/catalog')).json(); }
