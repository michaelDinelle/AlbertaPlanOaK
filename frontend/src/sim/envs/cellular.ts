import { NO_EVENTS, type EnvDecoder } from '../types';

export const CELLULAR_STATE = 0;
export const CELLULAR_ACTION = 1;

export function cellularNext(cells: number, count: number, rule: number): number {
  let next = 0;
  for (let i = 0; i < count; i++) {
    const left = (cells >>> ((i + count - 1) % count)) & 1, mid = (cells >>> i) & 1, right = (cells >>> ((i + 1) % count)) & 1;
    next |= ((rule >>> ((left << 2) | (mid << 1) | right)) & 1) << i;
  }
  return next >>> 0;
}

export const cellularDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[CELLULAR_STATE] = frame.words[0]!; out[CELLULAR_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
