import { useSyncExternalStore } from 'react';
import type { EnvDescriptor } from './data/envs';
import type { Policy, Recording, RunIndexEntry } from './data/layout';

export type Panel = 'reward' | 'standings' | 'health';

export interface StageMeta {
  status: 'empty' | 'loading' | 'buffering' | 'ready' | 'error';
  note: string | null;
  totalSteps: number;
}

export const EMPTY_STAGE: StageMeta = { status: 'empty', note: null, totalSteps: 0 };

export interface State {
  runs: RunIndexEntry[] | null;
  catalog: EnvDescriptor[];
  recordings: Record<string, Recording>;
  runId: string | null;
  policy: Policy;
  envId: string | null;
  seed: number | null;
  panel: Panel;
  panelOpen: boolean;
  playing: boolean;
  speed: number;
  debug: boolean;
  stage: StageMeta;
  error: string | null;
}

let value: State = {
  runs: null,
  catalog: [],
  recordings: {},
  runId: null,
  policy: 'oak',
  envId: null,
  seed: null,
  panel: 'reward',
  panelOpen: true,
  playing: true,
  speed: 1,
  debug: false,
  stage: EMPTY_STAGE,
  error: null,
};

const listeners = new Set<() => void>();
const get = (): State => value;
const subscribe = (listener: () => void): (() => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

export function patch(part: Partial<State>): void {
  value = { ...value, ...part };
  for (const listener of listeners) listener();
}

export const store = { get, use: <T = State>(select: (state: State) => T = (state) => state as unknown as T): T => useSyncExternalStore(subscribe, () => select(value)) };

export const selectRun = (runId: string): void => patch({ runId, seed: null });
export const selectEnv = (envId: string): void => patch({ envId, seed: null });
export const selectSeed = (seed: number | null): void => patch({ seed });
export const selectPolicy = (policy: Policy): void => patch({ policy });
export const SPEEDS = [1, 4, 16, 64, 256, 1024];
export const cycleSpeed = (): void => patch({ speed: SPEEDS[(SPEEDS.indexOf(value.speed) + 1) % SPEEDS.length] ?? 1 });
export const setStage = (stage: Partial<StageMeta>): void => patch({ stage: { ...value.stage, ...stage } });
export const fail = (error: unknown): void => patch({ error: error instanceof Error ? error.message : String(error) });
export const addRecording = (recording: Recording): void => patch({ recordings: { ...value.recordings, [recording.id]: recording } });
