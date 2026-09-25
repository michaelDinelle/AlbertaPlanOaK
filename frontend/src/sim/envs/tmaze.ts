import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const TMAZE_POS = 0;
export const TMAZE_CUE = 1;
export const TMAZE_TRIAL = 2;

export const tMazeArm = (action: number, cueBits: number): number => Math.trunc(action) & ((1 << cueBits) - 1);

export const tMazeDecoder: EnvDecoder = {
  slots: 3,
  discrete: true,
  decode(frame, out) {
    out[TMAZE_POS] = frame.words[0]!;
    out[TMAZE_CUE] = frame.words[1]!;
    out[TMAZE_TRIAL] = frame.words[3]!;
  },
  geometry({ environment }, initial) {
    const length = parameter(environment, 'length');
    if (initial.words.length !== 4) throw new Error('T-maze state does not carry position, cue, stride and trial words');
    return { grid: { w: length, h: 1, walls: new Uint8Array(length), goal: { x: length - 1, y: 0 } } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return [next.reward > 0 ? { type: 'goal' } : { type: 'harm' }, { type: 'teleport' }, { type: 'done' }];
  },
};
