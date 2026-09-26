import { parameter } from '../../data/envs';
import { NO_EVENTS, type EnvDecoder } from '../types';

export const MAZE_X = 0;
export const MAZE_Y = 1;
export const MAZE_GX = 2;
export const MAZE_GY = 3;
export const MAZE_HEADING = 4;

export const mazeDecoder: EnvDecoder = {
  slots: 5,
  discrete: true,
  decode(frame, out) {
    for (let i = 0; i < 4; i++) out[i] = frame.words[i]!;
    out[MAZE_HEADING] = frame.words[6]!;
  },
  geometry({ environment }, initial) {
    const side = 2 * parameter(environment, 'cells') - 1, size = side + 2, walls = new Uint8Array(size * size);
    if (environment.grid) {
      if (environment.grid.w !== size || environment.grid.h !== size) throw new Error(`${environment.envId} walls do not match its cells`);
      walls.set(environment.grid.walls);
    } else {
      for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        const index = x - 1 + side * (y - 1);
        walls[y * size + x] = x > 0 && x <= side && y > 0 && y <= side ? (initial.words[7 + (index >>> 5)]! >>> (index & 31)) & 1 : 1;
      }
    }
    return { grid: { w: size, h: size, walls, goal: { x: initial.words[MAZE_GX]!, y: initial.words[MAZE_GY]! } } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    if (next.reward !== 1) throw new Error('A maze reset has no goal reward');
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
