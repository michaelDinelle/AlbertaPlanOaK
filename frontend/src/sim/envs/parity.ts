import { NO_EVENTS, type EnvDecoder } from '../types';

export const PARITY_REGISTER = 96;
export const PARITY_ACTION = PARITY_REGISTER;

export const parityBitAt = (words: Uint32Array, age: number): number => (words[age >> 5]! >>> (age & 31)) & 1;

export const parityDecoder: EnvDecoder = {
  slots: PARITY_REGISTER + 1,
  discrete: true,
  decode(frame, out) {
    for (let age = 0; age < PARITY_REGISTER; age++) out[age] = parityBitAt(frame.words, age);
    out[PARITY_ACTION] = frame.actions[0]!;
  },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => next.reward > 0 ? [{ type: 'goal' }] : NO_EVENTS,
};
