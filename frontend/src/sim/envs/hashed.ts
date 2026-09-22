import { philox4x32_10 } from '../philox';
import { NO_EVENTS, type EnvDecoder } from '../types';
import { normalFrom } from './garnet';

export const HASHED_STATE = 0;
export const HASHED_ACTION = 1;

export function hashedNext(seed: number, bits: number, local: boolean, state: number, action: number): number {
  const r = philox4x32_10(seed, 0, state, action, 1, 0)[0];
  return local ? state ^ (1 << (r % bits)) : r & ((1 << bits) - 1);
}

export function hashedReward(seed: number, state: number, action: number): number {
  const r = philox4x32_10(seed, 0, state, action, 2, 0);
  return r[0] % 20 === 0 ? normalFrom(r[1], r[2]) : 0;
}

export const hashedDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[HASHED_STATE] = frame.words[0]!; out[HASHED_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
