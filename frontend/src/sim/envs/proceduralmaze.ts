// ProceduralMaze-cont-v0 decoder.
//
// Oracles (ported, never guessed):
//   Sources/Gridworld.metal      gw_pmaze_gen L539, gw_pmaze_is_wall L558,
//                                gw_pmaze_step L589
//   tools/gridworld_ref.py       pmaze_gen L478, pmaze_is_wall L501
//   Sources/OakKernel.metal      pose packing L2773-2786, init L2299-2306
//
// 11x11 with a border wall and a PER-LANE 9x9 interior layout generated once
// at fleet init from the RESET stream: 81 wall bits packed into 3 words (bit
// idx = (x-1) + 9*(y-1), w2 masked to its low 17 bits), then a random
// monotone path from (1,1) to (9,9) carved free (16 moves; move i, when not
// edge-forced, reads RESET ctr = (1 + i/4, 0,0,0) word i%4 and goes right on
// u01 < 0.5, else down), with (9,9) cleared last. The carved path is what
// guarantees the goal is reachable — it is a DEBUG overlay only, never art.
// The maze itself never resets; only the agent and goal are redrawn on a
// completion, and both land in the pose, so no replay is needed.
//
// pose = [x, y, gx, gy, 0, 0, 0, 0, 0].

import { philoxDraw, STREAM_RESET, u01 } from '../philox';
import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const PROCMAZE_W = 11;
export const PROCMAZE_H = 11;
/** interior cells (9x9), the bit domain of the wall words */
export const PROCMAZE_INTERIOR = 9;
export const PROCMAZE_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const PROCMAZE_GOAL: GridCell = Object.freeze({ x: 9, y: 9 });

export interface ProceduralMazeState {
  x: number;
  y: number;
  /** LIVE goal cell (rejection-redrawn to a free cell on completion) */
  gx: number;
  gy: number;
}

export interface ProceduralMazeStatic {
  seed: number;
  /** the 3 packed wall words, exactly as the kernel holds them */
  words: Uint32Array;
  walls: Uint8Array;
  /** the 17 cells the generator carved free, in order — DEBUG OVERLAY ONLY */
  carvedPath: GridCell[];
}

/** `gw_pmaze_is_wall` for a raw word triple. */
export function procMazeIsWall(words: ArrayLike<number>, x: number, y: number): boolean {
  if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
  const idx = x - 1 + PROCMAZE_INTERIOR * (y - 1);
  return (((words[idx >>> 5] ?? 0) >>> (idx & 31)) & 1) !== 0;
}

/**
 * `gw_pmaze_gen`: the per-lane wall words plus the carved monotone path.
 * Word arithmetic runs in a Uint32Array so every clear stays unsigned.
 */
export function procMazeGen(seed: number): { words: Uint32Array; carvedPath: GridCell[] } {
  const raw = philoxDraw(seed, STREAM_RESET, 0, 0);
  const words = Uint32Array.from([raw[0], raw[1], raw[2] & 0x1ffff]);
  const carvedPath: GridCell[] = [];
  let x = 1;
  let y = 1;
  for (let i = 0; i < 16; i++) {
    carvedPath.push({ x, y });
    const idx = x - 1 + PROCMAZE_INTERIOR * (y - 1);
    words[idx >>> 5] = (words[idx >>> 5] ?? 0) & ~(1 << (idx & 31));
    if (x === 9) {
      y++;
      continue;
    }
    if (y === 9) {
      x++;
      continue;
    }
    const c = philoxDraw(seed, STREAM_RESET, 1 + (i >>> 2), 0);
    if (u01(c[i & 3] ?? 0) < 0.5) x++;
    else y++;
  }
  const last = 8 + PROCMAZE_INTERIOR * 8; // (9,9)
  words[last >>> 5] = (words[last >>> 5] ?? 0) & ~(1 << (last & 31));
  carvedPath.push({ x: PROCMAZE_GOAL.x, y: PROCMAZE_GOAL.y });
  return { words, carvedPath };
}

/** w*h row-major occupancy, 1 = wall. */
export function procMazeWalls(words: ArrayLike<number>): Uint8Array {
  const walls = new Uint8Array(PROCMAZE_W * PROCMAZE_H);
  for (let y = 0; y < PROCMAZE_H; y++) {
    for (let x = 0; x < PROCMAZE_W; x++) {
      walls[y * PROCMAZE_W + x] = procMazeIsWall(words, x, y) ? 1 : 0;
    }
  }
  return walls;
}

function state(x: number, y: number, gx: number, gy: number): ProceduralMazeState {
  return { x, y, gx, gy };
}

export const proceduralMazeDecoder: EnvDecoder<ProceduralMazeState, ProceduralMazeStatic> = {
  envId: 'ProceduralMaze-cont-v0',
  needsReplay: false,

  initStatic(seed: number): ProceduralMazeStatic {
    const { words, carvedPath } = procMazeGen(seed);
    return { seed, words, walls: procMazeWalls(words), carvedPath };
  },

  geometry(st: ProceduralMazeStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: {
        w: PROCMAZE_W,
        h: PROCMAZE_H,
        walls: st.walls,
        goal: { ...PROCMAZE_GOAL },
        carvedPath: st.carvedPath.map((c) => ({ ...c })),
      },
    };
  },

  initFrame(): EnvFrame<ProceduralMazeState> {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = PROCMAZE_START.x;
    pose[1] = PROCMAZE_START.y;
    pose[2] = PROCMAZE_GOAL.x;
    pose[3] = PROCMAZE_GOAL.y;
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(PROCMAZE_START.x, PROCMAZE_START.y, PROCMAZE_GOAL.x, PROCMAZE_GOAL.y),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<ProceduralMazeState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    return {
      step: i,
      ctrl: raw16.subarray(OFF_CTRL, OFF_CTRL + CTRL_LEN),
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: state(pose[0] ?? 0, pose[1] ?? 0, pose[2] ?? 0, pose[3] ?? 0),
    };
  },

  deriveEvents(
    _prev: EnvFrame<ProceduralMazeState>,
    next: EnvFrame<ProceduralMazeState>,
  ): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
