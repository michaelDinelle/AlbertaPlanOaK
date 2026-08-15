
import { CTRL_LEN, OFF_DONE, OFF_POSE, POSE_LEN, effortView } from '../record';
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
export const POMAZE_VIEW_RADIUS = 1;

export interface POMazeState {
  x: number;
  y: number;
  gx: number;
  gy: number;
}

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

export function poMazeView(x: number, y: number): Uint8Array {
  const d: readonly (readonly [number, number])[] = [
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
  ];
  const out = new Uint8Array(8);
  for (let i = 0; i < 8; i++) {
    const o = d[i]!;
    out[i] = poMazeIsWall(x + o[0], y + o[1]) ? 1 : 0;
  }
  return out;
}

function state(x: number, y: number, gx: number, gy: number): POMazeState {
  return { x, y, gx, gy };
}

export const poMazeDecoder: EnvDecoder<POMazeState, POMazeStatic> = {
  envId: 'POMaze-cont-v0',

  initStatic(seed: number): POMazeStatic {
    return { seed, walls: poMazeWalls() };
  },

  geometry(st: POMazeStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: {
        w: POMAZE_W,
        h: POMAZE_H,
        walls: st.walls,
        goal: { ...POMAZE_GOAL },
        hallways: POMAZE_GAPS.map((c) => ({ ...c })),
      },
    };
  },

  initFrame(): EnvFrame<POMazeState> {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = POMAZE_START.x;
    pose[1] = POMAZE_START.y;
    pose[2] = POMAZE_GOAL.x;
    pose[3] = POMAZE_GOAL.y;
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(POMAZE_START.x, POMAZE_START.y, POMAZE_GOAL.x, POMAZE_GOAL.y),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<POMazeState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    return {
      step: i,
      ctrl: effortView(raw16, -1),
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: state(pose[0] ?? 0, pose[1] ?? 0, pose[2] ?? 0, pose[3] ?? 0),
    };
  },

  deriveEvents(_prev: EnvFrame<POMazeState>, next: EnvFrame<POMazeState>): SimEvent[] {
    if (next.done === 0) return NO_EVENTS;
    return [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }];
  },
};
