import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const PASSWORD_PHASE = 0;
export const PASSWORD_ACTION = 1;

export const passwordDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[PASSWORD_PHASE] = frame.words[0]!; out[PASSWORD_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = next.reward > 0 ? [{ type: 'goal' }] : [];
    return [...events, { type: 'done' }];
  },
};
