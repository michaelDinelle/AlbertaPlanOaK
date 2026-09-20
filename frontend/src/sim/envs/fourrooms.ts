import { NO_EVENTS, type EnvDecoder } from '../types';

export const FOURROOMS_X = 0;
export const FOURROOMS_Y = 1;
export const FOURROOMS_GX = 2;
export const FOURROOMS_GY = 3;

export function fourRoomsRoom(x: number, y: number, w: number, h: number): number {
  const cx = Math.floor(w / 2), cy = Math.floor(h / 2);
  if (x === cx || y === cy) return -1;
  return (x > cx ? 1 : 0) + (y > cy ? 2 : 0);
}

export const fourRoomsDecoder: EnvDecoder = {
  geometry({ environment }, initial) {
    const grid = environment.grid;
    if (!grid || grid.walls.length !== grid.w * grid.h) throw new Error('FourRooms walls are missing');
    const hallways = [];
    for (let y = 0; y < grid.h; y++) for (let x = 0; x < grid.w; x++) {
      if (!grid.walls[y * grid.w + x] && (x === Math.floor(grid.w / 2) || y === Math.floor(grid.h / 2))) hallways.push({ x, y });
    }
    return { grid: { ...grid, walls: Uint8Array.from(grid.walls), goal: { x: initial.pose[2]!, y: initial.pose[3]! }, hallways } };
  },
  deriveEvents(_prev, next) {
    if (!next.done) return NO_EVENTS;
    if (next.reward !== 1) throw new Error('FourRooms reset has no goal reward');
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
