import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const TMAZE_POS = 0;
export const TMAZE_CUE = 1;
export const TMAZE_ACTION = 2;

export const tmazeDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) { out[TMAZE_POS] = frame.words[0]!; out[TMAZE_CUE] = frame.words[1]!; out[TMAZE_ACTION] = frame.actions[0]!; },
  geometry({ environment }) {
    const length = parameter(environment, 'length');
    return { grid: { w: 1, h: length, walls: new Uint8Array(length), goal: { x: 0, y: 0 } } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = next.reward > 0 ? [{ type: 'goal' }] : [];
    return [...events, { type: 'cue', value: next.words[1]! }, { type: 'teleport' }, { type: 'done' }];
  },
};
