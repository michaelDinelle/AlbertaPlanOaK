import { createStore } from './store';

export interface OrchardReadout {
  live: boolean;
  sensed: readonly number[];
  scent: readonly [number, number, number];
  inventory: readonly [number, number, number, number];
  moved: boolean;
  polarity: 1 | -1;
  harvest: number;
}

const EMPTY: OrchardReadout = {
  live: false,
  sensed: [0, 0, 0, 0, 0, 0, 0, 0],
  scent: [0, 0, 0],
  inventory: [0, 0, 0, 0],
  moved: false,
  polarity: 1,
  harvest: 0,
};

const store = createStore<OrchardReadout>(EMPTY);

export const useOrchardReadout = store.use;
export const publishOrchardReadout = store.set;

export function clearOrchardReadout(): void {
  store.set(EMPTY);
}
