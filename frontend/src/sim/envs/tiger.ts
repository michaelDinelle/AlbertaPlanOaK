import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const TIGER_SIDE = 0;
export const TIGER_HEARD = 1;
export const TIGER_HINT = 2;
export const TIGER_ACTION = 3;
export const TIGER_LISTEN = 2;

export const tigerDecoder: EnvDecoder = {
  slots: 4,
  discrete: true,
  decode(frame, out) { out[TIGER_SIDE] = frame.words[0]!; out[TIGER_HEARD] = frame.words[1]!; out[TIGER_HINT] = frame.words[2]!; out[TIGER_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    if (next.words[1]! !== 0) return [{ type: 'cue', value: next.words[2]! }];
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = next.reward > 0 ? [{ type: 'goal' }] : [];
    return [...events, { type: 'done' }];
  },
};
