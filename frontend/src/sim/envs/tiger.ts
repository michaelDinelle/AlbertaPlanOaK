import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const TIGER_SIDE = 0;
export const TIGER_HEARD = 1;
export const TIGER_HINT = 2;
export const TIGER_TALLY = 3;
export const TIGER_ACTION = 4;
export const TIGER_LISTEN = 2;

export function tigerThreshold(accuracy: number, treasure: number, mauling: number, listenCost: number, tallyMax: number): number {
  const ratio = (1 - accuracy) / accuracy;
  let bestValue = listenCost, best = 0;
  for (let k = 1; k <= tallyMax; k++) {
    const power = ratio ** k, correct = 1 / (1 + power), listens = k / (2 * accuracy - 1) * (1 - power) / (1 + power);
    const value = (correct * treasure + (1 - correct) * mauling + listenCost * listens) / (listens + 1);
    if (value > bestValue) { bestValue = value; best = k; }
  }
  return best;
}

export const tigerDecoder: EnvDecoder = {
  slots: 5,
  discrete: true,
  decode(frame, out) { out[TIGER_SIDE] = frame.words[0]!; out[TIGER_HEARD] = frame.words[1]!; out[TIGER_HINT] = frame.words[2]!; out[TIGER_TALLY] = frame.words[3]! | 0; out[TIGER_ACTION] = frame.actions[0]!; },
  geometry: () => ({}),
  deriveEvents(_prev, next) {
    if (next.words[1]! !== 0) return [{ type: 'cue', value: next.words[2]! }];
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = next.reward > 0 ? [{ type: 'goal' }] : [];
    return [...events, { type: 'done' }];
  },
};
