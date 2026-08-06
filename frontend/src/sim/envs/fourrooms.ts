
import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const FOURROOMS_W = 11;
export const FOURROOMS_H = 11;
export const FOURROOMS_FREE_CELLS = 68;

export const FOURROOMS_HALLWAYS: readonly GridCell[] = Object.freeze([
  { x: 5, y: 2 },
  { x: 5, y: 8 },
  { x: 2, y: 5 },
  { x: 8, y: 5 },
]);

export const FOURROOMS_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const FOURROOMS_GOAL: GridCell = Object.freeze({ x: 9, y: 9 });

export interface FourRoomsState {
  x: number;
  y: number;
  gx: number;
  gy: number;
  room: number;
}

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

export function fourRoomsFreeCells(): GridCell[] {
  const out: GridCell[] = [];
  for (let y = 0; y < FOURROOMS_H; y++) {
    for (let x = 0; x < FOURROOMS_W; x++) {
      if (!fourRoomsIsWall(x, y)) out.push({ x, y });
    }
  }
  return out;
}

function state(x: number, y: number, gx: number, gy: number): FourRoomsState {
  return { x, y, gx, gy, room: fourRoomsRoom(x, y) };
}

export const fourRoomsDecoder: EnvDecoder<FourRoomsState, FourRoomsStatic> = {
  envId: 'FourRooms-cont-v0',
  needsReplay: false,

  initStatic(seed: number): FourRoomsStatic {
    return { seed, walls: fourRoomsWalls() };
  },

  geometry(st: FourRoomsStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: {
        w: FOURROOMS_W,
        h: FOURROOMS_H,
        walls: st.walls,
        goal: { ...FOURROOMS_GOAL },
        hallways: FOURROOMS_HALLWAYS.map((c) => ({ ...c })),
      },
    };
  },

  initFrame(): EnvFrame<FourRoomsState> {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = FOURROOMS_START.x;
    pose[1] = FOURROOMS_START.y;
    pose[2] = FOURROOMS_GOAL.x;
    pose[3] = FOURROOMS_GOAL.y;
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(FOURROOMS_START.x, FOURROOMS_START.y, FOURROOMS_GOAL.x, FOURROOMS_GOAL.y),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<FourRoomsState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    return {
      step: i,
      ctrl: raw16.subarray(OFF_CTRL, OFF_CTRL + CTRL_LEN),
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: state(pose[0] ?? 0, pose[1] ?? 0, pose[2] ?? 0, pose[3] ?? 0),
    };
  },

  deriveEvents(prev: EnvFrame<FourRoomsState>, next: EnvFrame<FourRoomsState>): SimEvent[] {
    if (next.state.x === prev.state.gx && next.state.y === prev.state.gy) {
      return [{ type: 'goal' }];
    }
    return NO_EVENTS;
  },
};
