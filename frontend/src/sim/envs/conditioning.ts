import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const CONDITIONING_COUNTDOWN = 0;
export const CONDITIONING_CUES = 1;
export const CONDITIONING_STIMULUS = 2;
export const CONDITIONING_ACTION = 3;

export const conditioningDecoder: EnvDecoder = {
  slots: 4,
  discrete: true,
  decode(frame, out) { out[CONDITIONING_COUNTDOWN] = frame.words[0]!; out[CONDITIONING_CUES] = frame.words[1]!; out[CONDITIONING_STIMULUS] = frame.words[2]!; out[CONDITIONING_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    const events: SimEvent[] = [];
    if (next.words[1]) events.push({ type: 'cue', value: next.words[1]! });
    if (next.words[2]) events.push(next.reward > 0 ? { type: 'goal' } : { type: 'harm' });
    return events.length ? events : NO_EVENTS;
  },
};
