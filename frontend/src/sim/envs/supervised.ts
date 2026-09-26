import { NO_EVENTS, type EnvDecoder } from '../types';
import { wordFloat } from '../words';

export const SUPERVISED_SOURCE = 0;
export const SUPERVISED_TARGET = 1;
export const SUPERVISED_SHOWN = 2;
export const RATE_LATENT = 0;
export const RATE_SIGMA = 1;

const supervised = (first: 'bits' | 'float'): EnvDecoder => ({
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[0] = first === 'bits' ? frame.words[0]! : wordFloat(frame.words[0]!);
    out[1] = wordFloat(frame.words[1]!);
    out[2] = wordFloat(frame.words[2]!);
  },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
});

export const idbdDecoder = supervised('bits');
export const slowRegressionDecoder = supervised('bits');
export const rateTrackingDecoder = supervised('float');
