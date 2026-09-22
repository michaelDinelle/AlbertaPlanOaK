import { NO_EVENTS, type EnvDecoder } from '../types';

export const RING_POSITION = 0;
export const RING_COLOUR = 1;
export const RING_ACTION = 2;

export const ringDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) { out[RING_POSITION] = frame.words[0]!; out[RING_COLOUR] = frame.observations[0]!; out[RING_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
