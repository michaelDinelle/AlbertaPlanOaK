import { constant } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const SOKOBAN_X = 0;
export const SOKOBAN_Y = 1;
export const SOKOBAN_BOX_X = 2;
export const SOKOBAN_BOX_Y = 3;
export const SOKOBAN_TARGET_X = 4;
export const SOKOBAN_TARGET_Y = 5;

export const sokobanDecoder: EnvDecoder = {
  slots: 6,
  discrete: true,
  decode(frame, out) { for (let i = 0; i < 6; i++) out[i] = frame.words[i]!; },
  geometry({ environment }, initial) {
    const side = constant(environment, 'SOKOBAN_SIDE');
    return { grid: { w: side, h: side, walls: new Uint8Array(side * side), goal: { x: initial.words[SOKOBAN_TARGET_X]!, y: initial.words[SOKOBAN_TARGET_Y]! } } };
  },
  deriveEvents(prev, next) {
    if (next.done) return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
    if (next.words[SOKOBAN_TARGET_X] !== prev.words[SOKOBAN_TARGET_X] || next.words[SOKOBAN_TARGET_Y] !== prev.words[SOKOBAN_TARGET_Y]) return [{ type: 'teleport' }];
    return NO_EVENTS;
  },
};
