
import { POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const POMAZE_W = 11;
export const POMAZE_H = 11;
export const POMAZE_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const POMAZE_GOAL: GridCell = Object.freeze({ x: 9, y: 9 });
export const POMAZE_GAPS: readonly GridCell[] = Object.freeze([
  { x: 8, y: 3 },
  { x: 2, y: 7 },
]);
export const POMAZE_X = 0;
export const POMAZE_Y = 1;
export const POMAZE_GX = 2;
export const POMAZE_GY = 3;

export interface POMazeStatic {
  seed: number;
  walls: Uint8Array;
}

export function poMazeIsWall(x: number, y: number): boolean {
  if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
  if (y === 3 && x !== 8) return true;
  if (y === 7 && x !== 2) return true;
  return false;
}

export function poMazeWalls(): Uint8Array {
  const walls = new Uint8Array(POMAZE_W * POMAZE_H);
  for (let y = 0; y < POMAZE_H; y++) {
    for (let x = 0; x < POMAZE_W; x++) {
      walls[y * POMAZE_W + x] = poMazeIsWall(x, y) ? 1 : 0;
    }
  }
  return walls;
}

export const poMazeDecoder: EnvDecoder<POMazeStatic> = {
  envId: 'POMaze-cont-v0',
  seedGeometry: false,

  initStatic(seed: number): POMazeStatic {
    return { seed, walls: poMazeWalls() };
  },

  geometry(st: POMazeStatic): EnvGeometry {
    return {
      grid: {
        w: POMAZE_W,
        h: POMAZE_H,
        walls: st.walls,
        goal: { ...POMAZE_GOAL },
        hallways: POMAZE_GAPS.map((c) => ({ ...c })),
      },
    };
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = POMAZE_START.x;
    pose[1] = POMAZE_START.y;
    pose[2] = POMAZE_GOAL.x;
    pose[3] = POMAZE_GOAL.y;
    return {
      step: -1,
      done: 0,
      pose,
    };
  },


  deriveEvents(_prev: EnvFrame, next: EnvFrame): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
