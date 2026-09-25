import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const ACCESS_FREE = 0;
export const ACCESS_PRIORITY = 1;

const PAID: readonly SimEvent[] = Object.freeze([{ type: 'goal' }]);

export const accessDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[ACCESS_FREE] = frame.words[0]!;
    out[ACCESS_PRIORITY] = frame.words[1]!;
  },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    return next.reward > 0 ? (PAID as SimEvent[]) : NO_EVENTS;
  },
};
