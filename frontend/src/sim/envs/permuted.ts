import { NO_EVENTS, type EnvDecoder, type EnvFrame, type SimEvent } from '../types';

export const PERMUTED_STATE = 0;
export const PERMUTED_CODE = 1;

export const permutedState = (frame: EnvFrame): number => frame.words[0]!;
export const permutedEpoch = (frame: EnvFrame): number => frame.view[0]!;
export const permutedAnswer = (frame: EnvFrame, state: number): number => frame.view[1 + state]!;
export const permutedAction = (frame: EnvFrame, actions: number): number => ((Math.trunc(frame.actions[0]!) % actions) + actions) % actions;

export function permutedCode(frame: EnvFrame, states: number): number {
  const x = frame.observations;
  for (let c = 0; c < states; c++) if (x[c]! > 0.5) return c;
  return -1;
}

const HIT: SimEvent[] = [{ type: 'goal' }];
const MISS: SimEvent[] = [{ type: 'harm' }];

export const permutedDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[PERMUTED_STATE] = permutedState(frame);
    out[PERMUTED_CODE] = permutedCode(frame, frame.observations.length - 1);
  },
  geometry: () => ({}),
  deriveEvents(prev, next) {
    if (next.step < 1) return NO_EVENTS;
    const base = next.reward > 0 ? HIT : MISS;
    if (prev.view.length > 0 && next.view.length > 0 && permutedEpoch(prev) !== permutedEpoch(next)) return [...base, { type: 'cue', value: permutedEpoch(next) }];
    return base;
  },
};
