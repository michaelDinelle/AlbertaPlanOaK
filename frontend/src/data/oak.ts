export const FRAME_FLOATS = 10;
export const OFF_DONE = 0;
export const OFF_POSE = 1;
export const POSE_LEN = 9;

export type Policy = 'oak' | 'ppo' | 'random';

export interface RunIndexEntry {
  id: string;
  envIds: string[];
  steps: number;
  startedAt: string;
  inFlight: boolean;
  policy: Policy;
  preset: string;
  role: 'best' | null;
}

export interface RunMeta {
  seeds: number[];
  steps: number;
  batch: number;
  policy: Policy;
  preset: string;
  startedAt: string;
  envSchedule: { envId: string; seedStart: number; seedEnd: number }[];
  envCatalog: unknown;
}

export interface RunDetail {
  id: string;
  meta: RunMeta;
  baselines: Partial<Record<'ppo' | 'random', string>>;
}

export interface Tape {
  rows: number;
  laneCount: number;
  stepEnd: Float64Array;
  rewardSum: Float32Array;
  rho: Float32Array;
  deathStep: Float32Array;
  deathDetector: Float32Array;
}

async function api(path: string): Promise<Response> {
  const response = await fetch(`/api${path}`);
  if (!response.ok) throw new Error(`${response.status} ${path}: ${await response.text()}`);
  return response;
}

export async function fetchRuns(): Promise<RunIndexEntry[]> {
  return (await api('/runs')).json() as Promise<RunIndexEntry[]>;
}

export async function fetchDetail(id: string): Promise<RunDetail> {
  return (await api(`/runs/${encodeURIComponent(id)}`)).json() as Promise<RunDetail>;
}

async function series(id: string, column: string): Promise<{ rows: number; lanes: number; ends: Float64Array; values: Float32Array }> {
  const bytes = await (await api(`/runs/${encodeURIComponent(id)}/series/${column}`)).arrayBuffer();
  const [rows, lanes] = new Uint32Array(bytes, 0, 2) as unknown as [number, number];
  return {
    rows,
    lanes,
    ends: Float64Array.from(new Uint32Array(bytes, 8, rows)),
    values: new Float32Array(bytes, 8 + 8 * rows, rows * lanes),
  };
}

export async function loadTape(id: string): Promise<Tape> {
  const [rewardSum, rho, deathStep, deathDetector] = await Promise.all([
    series(id, 'rewardSum'),
    series(id, 'rho'),
    series(id, 'deathStep'),
    series(id, 'deathDetector'),
  ]);
  return {
    rows: rewardSum.rows,
    laneCount: rewardSum.lanes,
    stepEnd: rewardSum.ends,
    rewardSum: rewardSum.values,
    rho: rho.values,
    deathStep: deathStep.values,
    deathDetector: deathDetector.values,
  };
}
