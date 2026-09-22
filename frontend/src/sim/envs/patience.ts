import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const PATIENCE_POS = 0;
export const PATIENCE_ACTION = 1;

export const patienceDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[PATIENCE_POS] = frame.words[0]!; out[PATIENCE_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = next.reward >= 1 ? [{ type: 'goal' }] : [];
    return [...events, { type: 'done' }];
  },
};
