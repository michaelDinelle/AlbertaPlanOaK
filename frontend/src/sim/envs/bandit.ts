import { NO_EVENTS, type EnvDecoder, type EnvFrame } from '../types';
import { wordFloat } from '../words';

export const BANDIT_ACTION = 0;

export const banditMean = (frame: EnvFrame, arm: number): number => wordFloat(frame.words[arm]!);

export const banditDecoder: EnvDecoder = {
  slots: 1,
  discrete: true,
  decode(frame, out) { out[BANDIT_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents: () => NO_EVENTS,
};
