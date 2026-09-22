import { NO_EVENTS, type EnvDecoder } from '../types';

export const NEXTING_PHASE = 0;
export const NEXTING_ACTION = 1;

export const nextingDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[NEXTING_PHASE] = frame.words[0]!; out[NEXTING_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => next.words[0] === 0 ? [{ type: 'cue', value: 1 }] : NO_EVENTS,
};
