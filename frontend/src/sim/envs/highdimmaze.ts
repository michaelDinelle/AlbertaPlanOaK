
import { POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const HIGHDIMMAZE_W = 11;
export const HIGHDIMMAZE_H = 11;
export const HIGHDIMMAZE_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const HIGHDIMMAZE_GOAL: GridCell = Object.freeze({ x: 9, y: 9 });
export const HIGHDIMMAZE_GAPS: readonly GridCell[] = Object.freeze([
  { x: 3, y: 9 },
  { x: 7, y: 1 },
]);
export const HIGHDIMMAZE_ACTIONS = 6;
export const HIGHDIMMAZE_STAY = 4;
export const HIGHDIMMAZE_DASH = 5;

export const HIGHDIMMAZE_X = 0;
export const HIGHDIMMAZE_Y = 1;
export const HIGHDIMMAZE_GX = 2;
export const HIGHDIMMAZE_GY = 3;
export const HIGHDIMMAZE_HEADING = 4;
export const HIGHDIMMAZE_ACTION = 5;

export interface HighDimMazeStatic {
  seed: number;
  walls: Uint8Array;
}

export function highDimMazeIsWall(x: number, y: number): boolean {
  if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
  if (x === 3 && y !== 9) return true;
  if (x === 7 && y !== 1) return true;
  return false;
}

export function highDimMazeWalls(): Uint8Array {
  const walls = new Uint8Array(HIGHDIMMAZE_W * HIGHDIMMAZE_H);
  for (let y = 0; y < HIGHDIMMAZE_H; y++) {
    for (let x = 0; x < HIGHDIMMAZE_W; x++) {
      walls[y * HIGHDIMMAZE_W + x] = highDimMazeIsWall(x, y) ? 1 : 0;
    }
  }
  return walls;
}

export const highDimMazeDecoder: EnvDecoder<HighDimMazeStatic> = {
  envId: 'HighDimMaze-cont-v0',

  initStatic(seed: number): HighDimMazeStatic {
    return { seed, walls: highDimMazeWalls() };
  },

  geometry(st: HighDimMazeStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: {
        w: HIGHDIMMAZE_W,
        h: HIGHDIMMAZE_H,
        walls: st.walls,
        goal: { ...HIGHDIMMAZE_GOAL },
        hallways: HIGHDIMMAZE_GAPS.map((c) => ({ ...c })),
      },
    };
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = HIGHDIMMAZE_START.x;
    pose[1] = HIGHDIMMAZE_START.y;
    pose[2] = HIGHDIMMAZE_GOAL.x;
    pose[3] = HIGHDIMMAZE_GOAL.y;
    pose[4] = 1;
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
