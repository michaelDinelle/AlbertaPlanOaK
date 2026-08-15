
import { POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const KEYCORRIDOR_W = 9;
export const KEYCORRIDOR_H = 7;
export const KEYCORRIDOR_ROOM_CELLS = 15;

export const KEYCORRIDOR_X = 0;
export const KEYCORRIDOR_Y = 1;
export const KEYCORRIDOR_GX = 2;
export const KEYCORRIDOR_GY = 3;
export const KEYCORRIDOR_KX = 4;
export const KEYCORRIDOR_KY = 5;
export const KEYCORRIDOR_CARRY = 6;
export const KEYCORRIDOR_GATE = 7;

export const KEYCORRIDOR_DOOR: GridCell = Object.freeze({ x: 4, y: 3 });
export const KEYCORRIDOR_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const KEYCORRIDOR_KEY: GridCell = Object.freeze({ x: 3, y: 5 });
export const KEYCORRIDOR_GOAL: GridCell = Object.freeze({ x: 7, y: 1 });

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

export const keyCorridorDecoder: EnvDecoder<KeyCorridorStatic> = {
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

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[0] = KEYCORRIDOR_START.x;
    pose[1] = KEYCORRIDOR_START.y;
    pose[2] = KEYCORRIDOR_GOAL.x;
    pose[3] = KEYCORRIDOR_GOAL.y;
    pose[4] = KEYCORRIDOR_KEY.x;
    pose[5] = KEYCORRIDOR_KEY.y;
    return {
      step: -1,
      done: 0,
      pose,
    };
  },


  deriveEvents(prev: EnvFrame, next: EnvFrame): SimEvent[] {
    const out: SimEvent[] = [];
    const goal = next.done !== 0;
    if (goal) out.push({ type: 'goal' });
    const wasCarrying = (prev.pose[KEYCORRIDOR_CARRY] ?? 0) !== 0;
    const carrying = (next.pose[KEYCORRIDOR_CARRY] ?? 0) !== 0;
    const wasOpen = (prev.pose[KEYCORRIDOR_GATE] ?? 0) !== 0;
    const open = (next.pose[KEYCORRIDOR_GATE] ?? 0) !== 0;
    if (!wasCarrying && carrying) out.push({ type: 'key', index: 0 });
    if (wasOpen !== open) out.push({ type: 'door', index: 0, open });
    if (goal) {
      out.push({ type: 'teleport' });
      out.push({ type: 'done' });
    }
    return out.length === 0 ? NO_EVENTS : out;
  },
};
