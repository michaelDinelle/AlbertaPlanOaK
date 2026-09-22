import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const RECITE_SEQUENCE = 0;
export const RECITE_PHASE = 1;
export const RECITE_ACTION = 2;

export const reciteDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) { out[RECITE_SEQUENCE] = frame.words[0]!; out[RECITE_PHASE] = frame.words[1]!; out[RECITE_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    const events: SimEvent[] = [];
    if (next.reward > 0) events.push({ type: 'goal' });
    if (next.done) events.push({ type: 'done' });
    return events.length ? events : NO_EVENTS;
  },
};
