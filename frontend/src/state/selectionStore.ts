import { useSyncExternalStore } from 'react';
import type { EnvId } from '@shared/envs';
import type { Policy } from '@shared/types';
import {
  escapeSelection,
  nextSort,
  promoteSeed,
  setLeader,
  type HeroSelection,
  type SortMode,
} from '../lib/selection';
import { currentPosition, setTransport } from '../playback/store';
import type { UrlState } from '../url/codec';
import { navigate } from '../url/useUrlState';

export interface SelectionState {
  runId: string | null;
  envId: EnvId | null;
  policy: Policy;
  selection: HeroSelection;
  sort: SortMode;
  pendingSeekStep: number | null;
  debug: boolean;
}

let state: SelectionState = {
  runId: null,
  envId: null,
  policy: 'oak',
  selection: { leader: null, promoted: null },
  sort: 'rate',
  pendingSeekStep: null,
  debug: false,
};

const listeners = new Set<() => void>();

function commit(next: SelectionState): void {
  state = next;
  for (const listener of listeners) listener();
}

export function selectionSnapshot(): SelectionState {
  return state;
}

export function subscribeSelection(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useSelection(): SelectionState {
  return useSyncExternalStore(subscribeSelection, selectionSnapshot);
}

export function adoptFromUrl(url: UrlState): void {
  commit({
    ...state,
    runId: url.run,
    envId: url.env,
    policy: url.policy,
    sort: url.sort,
    selection: { leader: null, promoted: url.seed },
    pendingSeekStep: url.at > 0 ? url.at : null,
  });
  setTransport({ speed: url.speed });
}

export function resolveRun(runId: string): void {
  if (state.runId !== null) return;
  commit({ ...state, runId });
}

function clockStep(): number | null {
  const at = Math.floor(currentPosition());
  return at > 0 ? at : state.pendingSeekStep;
}

export function selectRun(runId: string): void {
  if (state.runId === runId) return;
  const at = clockStep();
  commit({ ...state, runId, selection: { leader: null, promoted: state.selection.promoted }, pendingSeekStep: at });
  navigate({ run: runId, at: at ?? 0 });
}

export function selectEnv(envId: EnvId): void {
  if (state.envId === envId) return;
  commit({ ...state, envId, selection: { leader: null, promoted: null }, pendingSeekStep: null });
  navigate({ env: envId, seed: null, at: 0 });
}

export function adoptDefaultEnv(envId: EnvId): void {
  if (state.envId !== null) return;
  commit({ ...state, envId });
}

export function selectPolicy(policy: Policy): void {
  if (state.policy === policy) return;
  const at = clockStep();
  commit({ ...state, policy, pendingSeekStep: at });
  navigate({ policy, at: at ?? 0 });
}

export function electLeader(recorded: readonly number[]): void {
  const playable = (seed: number): boolean => recorded.includes(seed);
  const next = setLeader(state.selection, recorded[0] ?? null, playable);
  const dropped = state.selection.promoted !== null && next.promoted === null;
  if (next.leader === state.selection.leader && next.promoted === state.selection.promoted) return;
  commit({ ...state, selection: next });
  if (dropped) navigate({ seed: null });
}

export function promote(seed: number, playable: (seed: number) => boolean): void {
  const next = promoteSeed(state.selection, seed, playable);
  if (next === state.selection) return;
  commit({ ...state, selection: next });
  navigate({ seed });
}

export function unpromote(): void {
  const next = escapeSelection(state.selection);
  if (next === state.selection) return;
  commit({ ...state, selection: next });
  navigate({ seed: null });
}

export function cycleSortMode(): void {
  const sort = nextSort(state.sort);
  commit({ ...state, sort });
  navigate({ sort });
}

export function toggleDebug(): void {
  commit({ ...state, debug: !state.debug });
}

export function setPendingSeek(step: number | null): void {
  if (state.pendingSeekStep === step) return;
  commit({ ...state, pendingSeekStep: step });
}

export function consumePendingSeek(step: number): void {
  if (state.pendingSeekStep !== step) return;
  commit({ ...state, pendingSeekStep: null });
}
