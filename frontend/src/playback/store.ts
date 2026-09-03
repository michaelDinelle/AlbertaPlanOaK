import { DEFAULT_SPEED } from '../url/codec';
import { createStore } from '../state/store';

export interface TransportState {
  playing: boolean;
  speed: number;
  slice: boolean;
}

const transport = createStore<TransportState>({ playing: true, speed: DEFAULT_SPEED, slice: false });

export const transportSnapshot = transport.get;
export const subscribeTransport = transport.subscribe;
export const useTransport = transport.use;

export function setTransport(patch: Partial<TransportState>): void {
  const now = transport.get();
  const next = { ...now, ...patch };
  if (next.playing === now.playing && next.speed === now.speed && next.slice === now.slice) return;
  transport.set(next);
}

export type PositionListener = (step: number) => void;

const positionListeners = new Set<PositionListener>();

let position = 0;

export function publishPosition(step: number): void {
  position = step;
  for (const listener of positionListeners) listener(step);
}

export function currentPosition(): number {
  return position;
}

export function subscribePosition(listener: PositionListener): () => void {
  positionListeners.add(listener);
  return () => {
    positionListeners.delete(listener);
  };
}

const POSITION_HZ = 10;

export function throttled<A extends unknown[]>(fn: (...args: A) => void): (...args: A) => void {
  const intervalMs = 1000 / POSITION_HZ;
  let last = -Infinity;
  return (...args: A) => {
    const now = performance.now();
    if (now - last < intervalMs) return;
    last = now;
    fn(...args);
  };
}
