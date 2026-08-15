
import { CTRL_LEN, OFF_DONE, OFF_POSE, POSE_LEN, effortView } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const KEYCORRIDOR_W = 9;
export const KEYCORRIDOR_H = 7;
export const KEYCORRIDOR_ROOM_CELLS = 15;

export const KEYCORRIDOR_DOOR: GridCell = Object.freeze({ x: 4, y: 3 });
export const KEYCORRIDOR_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const KEYCORRIDOR_KEY: GridCell = Object.freeze({ x: 3, y: 5 });
export const KEYCORRIDOR_GOAL: GridCell = Object.freeze({ x: 7, y: 1 });

export interface KeyCorridorState {
  x: number;
  y: number;
  gx: number;
  gy: number;
  kx: number;
  ky: number;
  hasKey: boolean;
  doorOpen: boolean;
}

export interface KeyCorridorStatic {
  seed: number;
  walls: Uint8Array;
}

export function keyCorridorIsWall(x: number, y: number): boolean {
  if (x <= 0 || x >= 8 || y <= 0 || y >= 6) return true;
  if (x === 4 && y !== 3) return true;
  return false;
}

export function keyCorridorWalls(): Uint8Array {
  const walls = new Uint8Array(KEYCORRIDOR_W * KEYCORRIDOR_H);
  for (let y = 0; y < KEYCORRIDOR_H; y++) {
    for (let x = 0; x < KEYCORRIDOR_W; x++) {
      walls[y * KEYCORRIDOR_W + x] = keyCorridorIsWall(x, y) ? 1 : 0;
    }
  }
  return walls;
}

export function keyCorridorRoomCells(base: number): GridCell[] {
  const out: GridCell[] = [];
  for (let i = 0; i < KEYCORRIDOR_ROOM_CELLS; i++) {
    out.push({ x: base + (i % 3), y: 1 + Math.floor(i / 3) });
  }
  return out;
}

function state(
  x: number,
  y: number,
  gx: number,
  gy: number,
  kx: number,
  ky: number,
  hasKey: boolean,
  doorOpen: boolean,
): KeyCorridorState {
  return { x, y, gx, gy, kx, ky, hasKey, doorOpen };
}

export const keyCorridorDecoder: EnvDecoder<KeyCorridorState, KeyCorridorStatic> = {
  envId: 'KeyCorridor-cont-v0',

  initStatic(seed: number): KeyCorridorStatic {
    return { seed, walls: keyCorridorWalls() };
  },

  geometry(st: KeyCorridorStatic): EnvGeometry {
    return {
      kind: 'grid',
      grid: {
        w: KEYCORRIDOR_W,
        h: KEYCORRIDOR_H,
        walls: st.walls,
        goal: { ...KEYCORRIDOR_GOAL },
        doors: [{ ...KEYCORRIDOR_DOOR }],
        keys: [{ ...KEYCORRIDOR_KEY }],
        hallways: [{ ...KEYCORRIDOR_DOOR }],
      },
    };
  },

  initFrame(): EnvFrame<KeyCorridorState> {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = KEYCORRIDOR_START.x;
    pose[1] = KEYCORRIDOR_START.y;
    pose[2] = KEYCORRIDOR_GOAL.x;
    pose[3] = KEYCORRIDOR_GOAL.y;
    pose[4] = KEYCORRIDOR_KEY.x;
    pose[5] = KEYCORRIDOR_KEY.y;
    return {
      step: -1,
      ctrl: new Float32Array(CTRL_LEN),
      done: 0,
      pose,
      state: state(
        KEYCORRIDOR_START.x,
        KEYCORRIDOR_START.y,
        KEYCORRIDOR_GOAL.x,
        KEYCORRIDOR_GOAL.y,
        KEYCORRIDOR_KEY.x,
        KEYCORRIDOR_KEY.y,
        false,
        false,
      ),
    };
  },

  decodeFrame(i: number, raw16: Float32Array): EnvFrame<KeyCorridorState> {
    const pose = raw16.subarray(OFF_POSE, OFF_POSE + POSE_LEN);
    return {
      step: i,
      ctrl: effortView(raw16, -1),
      done: raw16[OFF_DONE] ?? 0,
      pose,
      state: state(
        pose[0] ?? 0,
        pose[1] ?? 0,
        pose[2] ?? 0,
        pose[3] ?? 0,
        pose[4] ?? 0,
        pose[5] ?? 0,
        (pose[6] ?? 0) !== 0,
        (pose[7] ?? 0) !== 0,
      ),
    };
  },

  deriveEvents(prev: EnvFrame<KeyCorridorState>, next: EnvFrame<KeyCorridorState>): SimEvent[] {
    const out: SimEvent[] = [];
    const goal = next.done !== 0;
    if (goal) out.push({ type: 'goal' });
    if (!prev.state.hasKey && next.state.hasKey) out.push({ type: 'key', index: 0 });
    if (prev.state.doorOpen !== next.state.doorOpen) {
      out.push({ type: 'door', index: 0, open: next.state.doorOpen });
    }
    if (goal) {
      out.push({ type: 'teleport' });
      out.push({ type: 'done' });
    }
    return out.length === 0 ? NO_EVENTS : out;
  },
};
