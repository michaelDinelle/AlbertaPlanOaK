import type { EnvId } from '../data/envs';
import type { Policy } from '../data/types';
import {
  escapeSelection,
  heroSeed,
  nextSort,
  promoteSeed,
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
  const pending = store.get().pendingSeekStep;
  if (pending !== null) return pending;
  const at = Math.floor(currentPosition());
  return at > 0 ? at : null;
}

export function selectRun(runId: string): void {
  if (store.get().runId === runId) return;
  const at = store.get().selection.promoted === null ? null : clockStep();
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
  const at = store.get().selection.promoted === null ? null : clockStep();
  store.patch({ policy, selection: { ...store.get().selection, leader: null }, pendingSeekStep: at });
  navigate({ policy, at: at ?? 0 });
}

export function electLeader(leader: HeroSelection['leader'], seeds: readonly number[]): void {
  const state = store.get(), previous = state.selection;
  const promoted = previous.promoted !== null && seeds.includes(previous.promoted) ? previous.promoted : null;
  const next = { leader, promoted };
  if (leader?.seed === previous.leader?.seed && leader?.step === previous.leader?.step && promoted === previous.promoted) return;
  const changed = promoted === null && heroSeed(next) !== heroSeed(previous);
  const at = state.pendingSeekStep ?? (changed ? leader?.step ?? null : null);
  store.patch({ selection: next, pendingSeekStep: at });
  if (changed || promoted !== previous.promoted) navigate({ seed: promoted, at: at ?? 0 });
}

export function promote(seed: number): void {
  const next = promoteSeed(store.get().selection, seed);
  if (next === store.get().selection) return;
  const at = clockStep();
  store.patch({ selection: next, pendingSeekStep: at });
  navigate({ seed, at: at ?? 0 });
}

export function unpromote(): void {
  const next = escapeSelection(store.get().selection);
  if (next === store.get().selection) return;
  const at = next.leader?.step ?? null;
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
