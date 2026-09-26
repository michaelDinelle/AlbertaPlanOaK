import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const COVARIATE_ACROSS = 0;
export const COVARIATE_ALONG = 1;

export interface CovariateRule {
  weights: number[];
  bias: number;
  normal: number[];
  along: number[];
  offset: number;
}

export function covariateRule(words: ArrayLike<number>): CovariateRule {
  const inputs = words.length - 1;
  const weights = Array.from({ length: inputs }, (_, j) => wordFloat(words[j]!));
  const bias = wordFloat(words[inputs]!), norm = Math.hypot(...weights);
  const normal = weights.map((w) => w / norm);
  let quiet = 0;
  for (let j = 1; j < inputs; j++) if (Math.abs(normal[j]!) < Math.abs(normal[quiet]!)) quiet = j;
  const raw = normal.map((n, j) => (j === quiet ? 1 : 0) - normal[quiet]! * n), length = Math.hypot(...raw);
  return { weights, bias, normal, along: raw.map((r) => r / length), offset: bias / norm };
}

const project = (axis: readonly number[], x: ArrayLike<number>): number => {
  let sum = 0;
  for (let j = 0; j < axis.length; j++) sum += axis[j]! * x[j]!;
  return sum;
};

export const covariateAcross = (rule: CovariateRule, x: ArrayLike<number>): number => project(rule.normal, x) + rule.offset;
export const covariateAlong = (rule: CovariateRule, x: ArrayLike<number>): number => project(rule.along, x);
export const covariateGuess = (frame: EnvFrame): number => Math.trunc(frame.actions[0]!) & 1;

const MISS: SimEvent[] = [{ type: 'harm' }];

export const covariateDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    const rule = covariateRule(frame.words);
    out[COVARIATE_ACROSS] = covariateAcross(rule, frame.observations);
    out[COVARIATE_ALONG] = covariateAlong(rule, frame.observations);
  },
  geometry: () => ({}),
  deriveEvents: (_prev, next) => (next.reward === 0 ? MISS : NO_EVENTS),
};
