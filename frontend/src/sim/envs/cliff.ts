import { constant } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type GridGeometry } from '../types';

export const CLIFF_X = 0;
export const CLIFF_Y = 1;

export const isCliffCell = (grid: Pick<GridGeometry, 'w'>, x: number, y: number): boolean => y === 0 && x > 0 && x < grid.w - 1;

export const cliffDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) { out[CLIFF_X] = frame.words[0]!; out[CLIFF_Y] = frame.words[1]!; },
  geometry({ environment }) {
    const w = constant(environment, 'CLIFF_WIDTH'), h = constant(environment, 'CLIFF_HEIGHT');
    return { grid: { w, h, walls: new Uint8Array(w * h), goal: { x: w - 1, y: 0 } } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return next.reward > 0 ? [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }] : [{ type: 'harm' }, { type: 'teleport' }, { type: 'done' }];
  },
};
