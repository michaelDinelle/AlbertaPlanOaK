import { useSyncExternalStore } from 'react';
import type { EnvId } from './data/envs';
import type { Policy, RunDetail, RunIndexEntry, Tape } from './data/oak';
import type { Tick } from './theater/recording';

export type Panel = 'reward' | 'standings' | 'health';

export interface StageMeta {
  status: 'empty' | 'loading' | 'ready' | 'error';
  note: string | null;
  frameCount: number;
  totalSteps: number;
  ticks: Tick[];
}

export const EMPTY_STAGE: StageMeta = { status: 'empty', note: null, frameCount: 1, totalSteps: 0, ticks: [] };

export interface State {
  runs: RunIndexEntry[] | null;
  details: Record<string, RunDetail>;
  tapes: Record<string, Tape>;
  runId: string | null;
  policy: Policy;
  envId: EnvId | null;
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
  details: {},
  tapes: {},
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
export const selectEnv = (envId: EnvId): void => patch({ envId, seed: null });
export const selectSeed = (seed: number | null): void => patch({ seed });
export const selectPolicy = (policy: Policy): void => patch({ policy });
export const setStage = (stage: Partial<StageMeta>): void => patch({ stage: { ...value.stage, ...stage } });
export const fail = (error: unknown): void => patch({ error: error instanceof Error ? error.message : String(error) });
export const addDetail = (detail: RunDetail, tape: Tape): void =>
  patch({ details: { ...value.details, [detail.id]: detail }, tapes: { ...value.tapes, [detail.id]: tape } });
export const addTape = (id: string, tape: Tape): void => patch({ tapes: { ...value.tapes, [id]: tape } });
