import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';
import { wordFloat } from '../words';

export const ARM2_Q1 = 0;
export const ARM2_Q2 = 1;
export const ARM2_W1 = 2;
export const ARM2_W2 = 3;
export const ARM2_TARGET_X = 4;
export const ARM2_TARGET_Y = 5;
export const ARM2_DISTANCE = 6;
export const ARM2_REGIME = 7;

export const arm2Decoder: EnvDecoder = {
  slots: 6,
  wrap: 1 << ARM2_Q1 | 1 << ARM2_Q2,
  decode(frame, out) {
    for (let i = 0; i < 6; i++) out[i] = wordFloat(frame.words[i]!);
  },
  geometry: () => ({}),
  deriveEvents(prev, next) {
    const shifted = next.words[ARM2_REGIME] !== prev.words[ARM2_REGIME];
    if (!next.done && !shifted) return NO_EVENTS;
    const events: SimEvent[] = next.done ? [{ type: 'goal' }, { type: 'done' }] : [];
    if (shifted) events.push({ type: 'cue', value: next.words[ARM2_REGIME]! });
    return events;
  },
};
