import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const KEYDOOR_PHASE = 0;
export const KEYDOOR_KEY = 1;
export const KEYDOOR_ACTION = 2;

export const keydoorDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) { out[KEYDOOR_PHASE] = frame.words[0]!; out[KEYDOOR_KEY] = frame.words[1]!; out[KEYDOOR_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = next.reward > 0 ? [{ type: 'goal' }] : [];
    return [...events, { type: 'done' }];
  },
};
