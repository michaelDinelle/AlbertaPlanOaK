import { NO_EVENTS, type EnvDecoder } from '../types';

export const CONDITIONING_COUNTDOWN = 0;
export const CONDITIONING_CUES = 1;
export const CONDITIONING_ACTION = 2;

export const conditioningDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) { out[CONDITIONING_COUNTDOWN] = frame.words[0]!; out[CONDITIONING_CUES] = frame.words[1]!; out[CONDITIONING_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => next.words[1] ? [{ type: 'cue', value: next.words[1]! }] : NO_EVENTS,
};
