import { NO_EVENTS, type EnvDecoder, type EnvFrame } from '../types';
import { wordFloat } from '../words';

export const BANDIT_ARM = 0;
export const BANDIT_BEST = 1;

export const banditArm = (frame: EnvFrame): number => (frame.actions[0]! >>> 0) % frame.words.length;

export function banditBest(words: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < words.length; i++) if (wordFloat(words[i]!) > wordFloat(words[best]!)) best = i;
  return best;
}

export const banditDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[BANDIT_ARM] = banditArm(frame);
    out[BANDIT_BEST] = banditBest(frame.words);
  },
  geometry: () => ({}),
  deriveEvents(prev, next) {
    const best = banditBest(next.words);
    return best === banditBest(prev.words) ? NO_EVENTS : [{ type: 'cue', value: best }];
  },
};
