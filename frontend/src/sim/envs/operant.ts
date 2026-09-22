import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const OPERANT_CUE = 0;
export const OPERANT_ACTION = 1;

export const operantDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[OPERANT_CUE] = frame.words[0]!; out[OPERANT_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    const events: SimEvent[] = [];
    if (next.words[0]! !== 0) events.push({ type: 'cue', value: 1 });
    if (next.reward > 0) events.push({ type: 'goal' });
    return events.length ? events : NO_EVENTS;
  },
};
