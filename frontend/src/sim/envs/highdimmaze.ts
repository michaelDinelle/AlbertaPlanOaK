// HighDimMaze-cont-v0 decoder.
//
// Oracles (ported, never guessed):
//   Sources/Gridworld.metal      gw_hdm_is_wall L662, gw_hdm_step L688
//   tools/gridworld_ref.py       hdm_is_wall L582, hdm_step L604
//   Sources/OakKernel.metal      pose packing L2787-2809, init L2307-2312
//
// 11x11 serpentine maze: border walls, interior wall x=3 except the gap
// (3,9), interior wall x=7 except the gap (7,1) — three corridors (down
// x 1..2, through (3,9), up x 4..6, through (7,1), down x 8..9). SIX actions:
// 0=up 1=down 2=left 3=right, 4=stay, 5=dash (up to two cells in lastDir,
// stopping at walls). lastDir is the last CARDINAL action and starts at
// 1 (down); it is recorded in pose[4], so a dash is fully readable from the
// record. Goal entry: +1, done=1, agent and goal rejection-redrawn — both
// land in the pose, so no replay is needed.
//
// pose = [x, y, gx, gy, lastDir, 0, 0, 0, 0].

import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const HIGHDIMMAZE_W = 11;
export const HIGHDIMMAZE_H = 11;
export const HIGHDIMMAZE_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const HIGHDIMMAZE_GOAL: GridCell = Object.freeze({ x: 9, y: 9 });
/** the two gaps that stitch the three corridors together */
export const HIGHDIMMAZE_GAPS: readonly GridCell[] = Object.freeze([
  { x: 3, y: 9 },
  { x: 7, y: 1 },
]);
/** action ids: 0=up 1=down 2=left 3=right 4=stay 5=dash */
export const HIGHDIMMAZE_ACTIONS = 6;
export const HIGHDIMMAZE_STAY = 4;
export const HIGHDIMMAZE_DASH = 5;

export interface HighDimMazeState {
  x: number;
  y: number;
  /** LIVE goal cell (rejection-redrawn to a free cell on completion) */
  gx: number;
  gy: number;
  /** last cardinal action, 0..3 — the direction a dash would travel */
  lastDir: number;
}

export interface HighDimMazeStatic {
  seed: number;
  walls: Uint8Array;
}

/** `gw_hdm_is_wall` / `hdm_is_wall`. */
export function highDimMazeIsWall(x: number, y: number): boolean {
  if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
  if (x === 3 && y !== 9) return true;
  if (x === 7 && y !== 1) return true;
  return false;
}

/** w*h row-major occupancy, 1 = wall. */
export function highDimMazeWalls(): Uint8Array {
  const walls = new Uint8Array(HIGHDIMMAZE_W * HIGHDIMMAZE_H);
  for (let y = 0; y < HIGHDIMMAZE_H; y++) {
    for (let x = 0; x < HIGHDIMMAZE_W; x++) {
      walls[y * HIGHDIMMAZE_W + x] = highDimMazeIsWall(x, y) ? 1 : 0;
    }
  }
  return walls;
}

function state(x: number, y: number, gx: number, gy: number, lastDir: number): HighDimMazeState {
  return { x, y, gx, gy, lastDir };
}

export const highDimMazeDecoder: EnvDecoder<HighDimMazeState, HighDimMazeStatic> = {
  envId: 'HighDimMaze-cont-v0',
  needsReplay: false,

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

  initFrame(): EnvFrame<HighDimMazeState> {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = HIGHDIMMAZE_START.x;
    pose[1] = HIGHDIMMAZE_START.y;
    pose[2] = HIGHDIMMAZE_GOAL.x;
    pose[3] = HIGHDIMMAZE_GOAL.y;
    pose[4] = 1; // lastDir = down (oak_fleet_init)
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(
        HIGHDIMMAZE_START.x,
        HIGHDIMMAZE_START.y,
        HIGHDIMMAZE_GOAL.x,
        HIGHDIMMAZE_GOAL.y,
        1,
      ),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<HighDimMazeState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    return {
      step: i,
      ctrl: raw16.subarray(OFF_CTRL, OFF_CTRL + CTRL_LEN),
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: state(pose[0] ?? 0, pose[1] ?? 0, pose[2] ?? 0, pose[3] ?? 0, pose[4] ?? 0),
    };
  },

  deriveEvents(_prev: EnvFrame<HighDimMazeState>, next: EnvFrame<HighDimMazeState>): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
