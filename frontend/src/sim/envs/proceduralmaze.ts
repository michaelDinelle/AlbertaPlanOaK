
import { philoxDraw, STREAM_RESET } from '../philox';
import { CTRL_LEN, OFF_DONE, OFF_POSE, POSE_LEN, effortView } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const PROCMAZE_W = 11;
export const PROCMAZE_H = 11;
export const PROCMAZE_INTERIOR = 9;
export const PROCMAZE_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const PROCMAZE_GOAL: GridCell = Object.freeze({ x: 9, y: 9 });

export interface ProceduralMazeState {
  x: number;
  y: number;
  gx: number;
  gy: number;
}

export interface ProceduralMazeStatic {
  seed: number;
  words: Uint32Array;
  walls: Uint8Array;
  carvedPath: GridCell[];
}

export function procMazeIsWall(words: ArrayLike<number>, x: number, y: number): boolean {
  if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
  const idx = x - 1 + PROCMAZE_INTERIOR * (y - 1);
  return (((words[idx >>> 5] ?? 0) >>> (idx & 31)) & 1) !== 0;
}

export function procMazeGen(seed: number): { words: Uint32Array; carvedPath: GridCell[] } {
  const words = Uint32Array.from([0xffffffff, 0xffffffff, 0x1ffff]);
  for (let j = 0; j < 5; j++) {
    for (let i = 0; i < 5; i++) {
      const idx = 2 * i + PROCMAZE_INTERIOR * (2 * j);
      words[idx >>> 5] = (words[idx >>> 5] ?? 0) & ~(1 << (idx & 31));
    }
  }
  const carvedPath: GridCell[] = [];
  let visited = 1;
  const stack = [0];
  let ctr = 0;
  let bi = 0;
  let buf = philoxDraw(seed, STREAM_RESET, 0, 0);
  while (stack.length > 0) {
    const cur = stack[stack.length - 1] ?? 0;
    const ci = cur % 5;
    const cj = (cur / 5) | 0;
    const cand: number[] = [];
    for (const [di, dj] of [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
    ] as const) {
      const ni = ci + di;
      const nj = cj + dj;
      if (ni >= 0 && ni < 5 && nj >= 0 && nj < 5 && ((visited >>> (ni + 5 * nj)) & 1) === 0) {
        cand.push(ni + 5 * nj);
      }
    }
    if (cand.length === 0) {
      stack.pop();
      continue;
    }
    if (bi === 4) {
      ctr++;
      buf = philoxDraw(seed, STREAM_RESET, ctr, 0);
      bi = 0;
    }
    const nxt = cand[(buf[bi] ?? 0) % cand.length] ?? 0;
    bi++;
    const ni = nxt % 5;
    const nj = (nxt / 5) | 0;
    const idx = ci + ni + PROCMAZE_INTERIOR * (cj + nj);
    words[idx >>> 5] = (words[idx >>> 5] ?? 0) & ~(1 << (idx & 31));
    carvedPath.push({ x: ci + ni + 1, y: cj + nj + 1 });
    visited |= 1 << nxt;
    stack.push(nxt);
  }
  return { words, carvedPath };
}

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
      ctrl: effortView(raw16, -1),
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
