import { constant } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const CLIFF_X = 0;
export const CLIFF_Y = 1;

export const cliffDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[CLIFF_X] = frame.words[0]!;
    out[CLIFF_Y] = frame.words[1]!;
  },
  geometry({ environment }, initial) {
    const w = constant(environment, 'CLIFF_WIDTH'), h = constant(environment, 'CLIFF_HEIGHT');
    if (initial.words.length !== 2) throw new Error('CliffWalk state does not carry its column and row words');
    return { grid: { w, h, walls: new Uint8Array(w * h) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return [next.reward > 0 ? { type: 'goal' } : { type: 'harm' }, { type: 'teleport' }, { type: 'done' }];
  },
};
