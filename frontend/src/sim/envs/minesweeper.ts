import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder, type EnvFrame } from '../types';

export const MINESWEEPER_X = 0;
export const MINESWEEPER_Y = 1;
export const MINESWEEPER_GAME = 2;
export const MINESWEEPER_OPENED = 3;
export const MINESWEEPER_FLAGS = 4;
export const MINESWEEPER_FEED = 5;
export const MINESWEEPER_IDLE = 6;
export const MINESWEEPER_FORCED = 7;
export const MINESWEEPER_FEED_X = 11;
export const MINESWEEPER_FEED_Y = 12;

export const MINED = 0;
export const SHOWN = 1;
export const FLAGGED = 2;

export const minesweeperPlane = (frame: EnvFrame, plane: number, cells: number, cell: number): number => {
  const bit = plane * cells + cell;
  return (frame.view[bit >>> 5]! >>> (bit & 31)) & 1;
};

export const minesweeperDecoder: EnvDecoder = {
  slots: 2,
  discrete: true,
  decode(frame, out) {
    out[MINESWEEPER_X] = frame.words[MINESWEEPER_X]!;
    out[MINESWEEPER_Y] = frame.words[MINESWEEPER_Y]!;
  },
  geometry({ environment }, initial) {
    const w = parameter(environment, 'width'), h = parameter(environment, 'height');
    if (initial.words.length !== 8 || initial.view.length !== Math.ceil(3 * w * h / 32)) throw new Error(`${environment.envId} does not carry its cursor and clock words and its mine, shown and flag planes`);
    return { grid: { w, h, walls: new Uint8Array(w * h) } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    return [next.reward > 0 ? { type: 'goal' } : { type: 'harm' }, { type: 'done' }];
  },
};
