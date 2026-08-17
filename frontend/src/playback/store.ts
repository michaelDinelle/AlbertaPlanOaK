import { useSyncExternalStore } from 'react';
import { DEFAULT_SPEED } from '../url/codec';

export interface TransportState {
  playing: boolean;
  speed: number;
}

let transport: TransportState = { playing: true, speed: DEFAULT_SPEED };

const transportListeners = new Set<() => void>();

export function transportSnapshot(): TransportState {
  return transport;
}

export function setTransport(patch: Partial<TransportState>): void {
  const next = { ...transport, ...patch };
  if (next.playing === transport.playing && next.speed === transport.speed) return;
  transport = next;
  for (const listener of transportListeners) listener();
}

export function subscribeTransport(listener: () => void): () => void {
  transportListeners.add(listener);
  return () => {
    transportListeners.delete(listener);
  };
}

export function useTransport(): TransportState {
  return useSyncExternalStore(subscribeTransport, transportSnapshot);
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

export const POSITION_HZ = 10;

export function throttled<A extends unknown[]>(fn: (...args: A) => void, hz = POSITION_HZ): (...args: A) => void {
  const intervalMs = 1000 / hz;
  let last = -Infinity;
  return (...args: A) => {
    const now = performance.now();
    if (now - last < intervalMs) return;
    last = now;
    fn(...args);
  };
}
