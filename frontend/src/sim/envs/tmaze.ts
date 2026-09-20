import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type SimEvent } from '../types';

export const TMAZE_POS = 0;
export const TMAZE_CUE = 2;
export const TMAZE_ACTION = 3;

export const tMazeDecoder: EnvDecoder = {
  geometry({ environment }) {
    const length = parameter(environment, 'length');
    return { grid: { w: length, h: 1, walls: new Uint8Array(length), goal: { x: length - 1, y: 0 } } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    const events: SimEvent[] = next.reward > 0 ? [{ type: 'goal' }] : [];
    return [...events, { type: 'cue', value: next.pose[TMAZE_CUE]! }, { type: 'teleport' }, { type: 'done' }];
  },
};
