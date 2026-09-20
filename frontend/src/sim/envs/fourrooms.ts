import { POSE_LEN } from '../../data/oak';
import { NO_EVENTS, type EnvDecoder, type EnvFrame, type EnvGeometry, type GridCell, type SimEvent } from '../types';

export const FOURROOMS_W = 11;
export const FOURROOMS_H = 11;
export const FOURROOMS_HALLWAYS: readonly GridCell[] = Object.freeze([
  { x: 5, y: 2 },
  { x: 5, y: 8 },
  { x: 2, y: 5 },
  { x: 8, y: 5 },
]);

export const FOURROOMS_X = 0;
export const FOURROOMS_Y = 1;
export const FOURROOMS_GX = 2;
export const FOURROOMS_GY = 3;

export const FOURROOMS_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const FOURROOMS_GOAL: GridCell = Object.freeze({ x: 9, y: 9 });

export interface FourRoomsStatic {
  seed: number;
  walls: Uint8Array;
}

export function fourRoomsIsWall(x: number, y: number): boolean {
  if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
  if (x === 5 && y !== 2 && y !== 8) return true;
  if (y === 5 && x !== 2 && x !== 8) return true;
  return false;
}

export function fourRoomsRoom(x: number, y: number): number {
  if (x < 5 && y < 5) return 0;
  if (x > 5 && y < 5) return 1;
  if (x < 5 && y > 5) return 2;
  if (x > 5 && y > 5) return 3;
  return -1;
}

export function fourRoomsWalls(): Uint8Array {
  const walls = new Uint8Array(FOURROOMS_W * FOURROOMS_H);
  for (let y = 0; y < FOURROOMS_H; y++) {
    for (let x = 0; x < FOURROOMS_W; x++) {
      walls[y * FOURROOMS_W + x] = fourRoomsIsWall(x, y) ? 1 : 0;
    }
  }
  return walls;
}

export const fourRoomsDecoder: EnvDecoder<FourRoomsStatic> = {
  envId: 'FourRoomsHall-cont-v0',
  seedGeometry: false,

  initStatic(seed: number): FourRoomsStatic {
    return { seed, walls: fourRoomsWalls() };
  },

  geometry(st: FourRoomsStatic): EnvGeometry {
    return {
      grid: {
        w: FOURROOMS_W,
        h: FOURROOMS_H,
        walls: st.walls,
        goal: { ...FOURROOMS_GOAL },
        hallways: FOURROOMS_HALLWAYS.map((c) => ({ ...c })),
      },
    };
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = FOURROOMS_START.x;
    pose[1] = FOURROOMS_START.y;
    pose[2] = FOURROOMS_GOAL.x;
    pose[3] = FOURROOMS_GOAL.y;
    return {
      step: -1,
      done: 0,
      pose,
    };
  },

  deriveEvents(prev: EnvFrame, next: EnvFrame): SimEvent[] {
    if (
      next.pose[FOURROOMS_X] === prev.pose[FOURROOMS_GX]
      && next.pose[FOURROOMS_Y] === prev.pose[FOURROOMS_GY]
    ) {
      return [{ type: 'goal' }];
    }
    return NO_EVENTS;
  },
};
