import { NO_EVENTS, type EnvDecoder } from '../types';

export const WALKBET_POSITION = 0;
export const WALKBET_ACTION = 1;

export const walkbetDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[WALKBET_POSITION] = frame.words[0]!; out[WALKBET_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => next.done ? [{ type: 'teleport' }, { type: 'done' }] : NO_EVENTS,
};
