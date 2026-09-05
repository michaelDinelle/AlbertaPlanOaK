import { useSyncExternalStore } from 'react';

export interface Store<T> {
  get(): T;
  set(next: T): void;
  patch(part: Partial<T>): void;
  subscribe(listener: () => void): () => void;
  use(): T;
}

export function createStore<T extends object>(key: string, initial: T): Store<T> {
  const hot = import.meta.hot?.data;
  if (hot?.[key]) return hot[key] as Store<T>;
  let value = initial;
  const listeners = new Set<() => void>();
  const get = (): T => value;
  const subscribe = (listener: () => void): (() => void) => {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  };
  const set = (next: T): void => {
    value = next;
    for (const listener of listeners) listener();
  };
  const store: Store<T> = {
    get,
    set,
    subscribe,
    patch: (part) => set({ ...value, ...part }),
    use: () => useSyncExternalStore(subscribe, get),
  };
  if (hot) hot[key] = store;
  return store;
}
