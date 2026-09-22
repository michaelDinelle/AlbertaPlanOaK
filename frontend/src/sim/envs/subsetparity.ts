import { NO_EVENTS, type EnvDecoder } from '../types';

export const SUBSETPARITY_MASK = 0;
export const SUBSETPARITY_BITS = 1;
export const SUBSETPARITY_ACTION = 2;

export const subsetparityDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) { out[SUBSETPARITY_MASK] = frame.words[0]!; out[SUBSETPARITY_BITS] = frame.words[1]!; out[SUBSETPARITY_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
