import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const REGRESSION_Y = 0;

const fma32 = (a: number, b: number, c: number): number => {
  const p = a * b, s = p + c, back = s - p, error = p - (s - back) + (c - back), r = Math.fround(s), other = 2 * s - r;
  return error !== 0 && other !== r && Math.fround(other) === other && error > 0 === other > r ? other : r;
};

export function regressionTarget(frame: EnvFrame, sums?: Float64Array): number {
  let y = 0;
  for (let j = 0; j < frame.words.length; j++) {
    y = fma32(wordFloat(frame.words[j]!), frame.observations[j]!, y);
    if (sums) sums[j + 1] = y;
  }
  return y;
}

export const regressionBin = (y: number, edges: readonly number[]): number => edges.reduce((bin, edge) => (y > edge ? bin + 1 : bin), 0);

export const regressionGuess = (frame: EnvFrame, bins: number): number => {
  const action = frame.actions[0]!;
  return (action > 0 ? Math.min(Math.trunc(action), 0xffffffff) : 0) % bins;
};

const HIT: SimEvent[] = [{ type: 'goal' }];

export const regressionDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) {
    out[REGRESSION_Y] = regressionTarget(frame);
  },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => (next.reward === 1 ? HIT : NO_EVENTS),
};
