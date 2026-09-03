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
import { createStore } from './store';

export interface SelectionState {
  runId: string | null;
  envId: EnvId | null;
  policy: Policy;
  selection: HeroSelection;
  sort: SortMode;
  pendingSeekStep: number | null;
  debug: boolean;
}

const store = createStore<SelectionState>({
  runId: null,
  envId: null,
  policy: 'oak',
  selection: { leader: null, promoted: null },
  sort: 'rate',
  pendingSeekStep: null,
  debug: false,
});

export const selectionSnapshot = store.get;
export const subscribeSelection = store.subscribe;
export const useSelection = store.use;

export function adoptFromUrl(url: UrlState): void {
  store.patch({
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
  if (store.get().runId !== null) return;
  store.patch({ runId });
}

function clockStep(): number | null {
  const at = Math.floor(currentPosition());
  return at > 0 ? at : store.get().pendingSeekStep;
}

export function selectRun(runId: string): void {
  if (store.get().runId === runId) return;
  const at = clockStep();
  store.patch({ runId, selection: { leader: null, promoted: store.get().selection.promoted }, pendingSeekStep: at });
  navigate({ run: runId, at: at ?? 0 });
}

export function selectEnv(envId: EnvId): void {
  if (store.get().envId === envId) return;
  store.patch({ envId, selection: { leader: null, promoted: null }, pendingSeekStep: null });
  navigate({ env: envId, seed: null, at: 0 });
}

export function adoptDefaultEnv(envId: EnvId): void {
  if (store.get().envId !== null) return;
  store.patch({ envId });
}

export function selectPolicy(policy: Policy): void {
  if (store.get().policy === policy) return;
  const at = clockStep();
  store.patch({ policy, pendingSeekStep: at });
  navigate({ policy, at: at ?? 0 });
}

export function electLeader(recorded: readonly number[]): void {
  const playable = (seed: number): boolean => recorded.includes(seed);
  const next = setLeader(store.get().selection, recorded[0] ?? null, playable);
  const dropped = store.get().selection.promoted !== null && next.promoted === null;
  if (next.leader === store.get().selection.leader && next.promoted === store.get().selection.promoted) return;
  store.patch({ selection: next });
  if (dropped) navigate({ seed: null });
}

export function promote(seed: number, playable: (seed: number) => boolean): void {
  const next = promoteSeed(store.get().selection, seed, playable);
  if (next === store.get().selection) return;
  const at = clockStep();
  store.patch({ selection: next, pendingSeekStep: at });
  navigate({ seed, at: at ?? 0 });
}

export function unpromote(): void {
  const next = escapeSelection(store.get().selection);
  if (next === store.get().selection) return;
  const at = clockStep();
  store.patch({ selection: next, pendingSeekStep: at });
  navigate({ seed: null, at: at ?? 0 });
}

export function cycleSortMode(): void {
  const sort = nextSort(store.get().sort);
  store.patch({ sort });
  navigate({ sort });
}

export function toggleDebug(): void {
  store.patch({ debug: !store.get().debug });
}

export function setPendingSeek(step: number | null): void {
  if (store.get().pendingSeekStep === step) return;
  store.patch({ pendingSeekStep: step });
}

export function consumePendingSeek(step: number): void {
  if (store.get().pendingSeekStep !== step) return;
  store.patch({ pendingSeekStep: null });
}
