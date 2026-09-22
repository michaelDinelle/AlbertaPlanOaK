import { NO_EVENTS, type EnvDecoder } from '../types';
import { wordFloat } from '../words';

export const COVARIATE_ACTION = 0;
export const COVARIATE_WEIGHTS = 1;

export const covariateDecoder: EnvDecoder = {
  slots: 6,
  discrete: true,
  decode(frame, out) {
    out[COVARIATE_ACTION] = frame.actions[0]!;
    for (let j = 0; j < 5; j++) out[COVARIATE_WEIGHTS + j] = wordFloat(frame.words[j]!);
  },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
