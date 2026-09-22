import { NO_EVENTS, type EnvDecoder } from '../types';

export const PERMUTED_STATE = 0;
export const PERMUTED_CODE = 1;
export const PERMUTED_ACTION = 2;

export const permutedDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[PERMUTED_STATE] = frame.words[0]!;
    let code = -1;
    for (let i = 0; i < frame.observations.length - 1; i++) if (frame.observations[i]! > 0.5) code = i;
    out[PERMUTED_CODE] = code;
    out[PERMUTED_ACTION] = frame.actions[0]!;
  },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
