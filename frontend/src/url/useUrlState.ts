import { useSyncExternalStore } from 'react';
import { mergeUrlState, parseUrlState, serializeUrlState, type UrlState } from './codec';

let cached: { search: string; state: UrlState } | null = null;

const listeners = new Set<() => void>();

function read(): UrlState {
  const search = window.location.search;
  if (cached === null || cached.search !== search) {
    cached = { search, state: parseUrlState(search) };
  }
  return cached.state;
}

export function urlState(): UrlState {
  return read();
}

export function navigate(patch: Partial<UrlState>): void {
  const current = read();
  const next = mergeUrlState(current, patch);
  const search = serializeUrlState(next);
  if (search === serializeUrlState(current)) return;
  const url = `${window.location.pathname}${search}${window.location.hash}`;
  const runChanged = patch.run !== undefined && patch.run !== current.run;
  if (runChanged) history.pushState(null, '', url);
  else history.replaceState(null, '', url);
  cached = { search: window.location.search, state: next };
  emit();
}

function emit(): void {
  for (const listener of listeners) listener();
}

function onPop(): void {
  emit();
}

function subscribe(listener: () => void): () => void {
  if (listeners.size === 0) window.addEventListener('popstate', onPop);
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) window.removeEventListener('popstate', onPop);
  };
}

export function useUrlState(): UrlState {
  return useSyncExternalStore(subscribe, read);
}
