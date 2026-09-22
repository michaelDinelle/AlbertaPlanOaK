import { NO_EVENTS, type EnvDecoder } from '../types';

export const REPEAT_REGISTER = 48;
export const REPEAT_ACTION = REPEAT_REGISTER;

export const repeatSymbolAt = (words: Uint32Array, age: number): number => (words[age >> 4]! >>> (2 * (age & 15))) & 3;

export const repeatDecoder: EnvDecoder = {
  slots: REPEAT_REGISTER + 1,
  discrete: true,
  decode(frame, out) {
    for (let age = 0; age < REPEAT_REGISTER; age++) out[age] = repeatSymbolAt(frame.words, age);
    out[REPEAT_ACTION] = frame.actions[0]!;
  },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => next.reward > 0 ? [{ type: 'goal' }] : NO_EVENTS,
};
