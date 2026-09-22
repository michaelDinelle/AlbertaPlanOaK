import { NO_EVENTS, type EnvDecoder, type EnvFrame } from '../types';
import { wordFloat } from '../words';

export const REGRESSION_EDGE = [-1, -0.3, 0.3, 1] as const;
export const REGRESSION_ACTION = 0;
export const REGRESSION_WEIGHTS = 1;

export function regressionSum(frame: EnvFrame, inputs: number): number {
  let y = 0;
  for (let j = 0; j < inputs; j++) y += wordFloat(frame.words[j]!) * frame.observations[j]!;
  return y;
}

export function regressionBin(y: number): number {
  let bin = 0;
  while (bin < REGRESSION_EDGE.length && y > REGRESSION_EDGE[bin]!) bin++;
  return bin;
}

export const regressionDecoder: EnvDecoder = {
  slots: 1 + 16,
  discrete: true,
  decode(frame, out) {
    out[REGRESSION_ACTION] = frame.actions[0]!;
    for (let j = 0; j < frame.words.length; j++) out[REGRESSION_WEIGHTS + j] = wordFloat(frame.words[j]!);
  },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
