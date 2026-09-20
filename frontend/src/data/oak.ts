import { unpackRecording, type Recording, type RunIndexEntry } from './layout';

export const FRAME_FLOATS = 10;
export const OFF_DONE = 0;
export const OFF_POSE = 1;
export const POSE_LEN = 9;

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
