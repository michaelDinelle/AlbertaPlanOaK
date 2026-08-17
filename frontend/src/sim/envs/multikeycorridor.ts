
import { POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const MKC_W = 17;
export const MKC_H = 7;
export const MKC_ROOM_CELLS = 15;
export const MKC_ROOM_BASE: readonly number[] = Object.freeze([1, 5, 9, 13]);
export const MKC_DOORS: readonly GridCell[] = Object.freeze([
  { x: 4, y: 3 },
  { x: 8, y: 3 },
  { x: 12, y: 3 },
]);
export const MKC_START: GridCell = Object.freeze({ x: 1, y: 1 });
export const MKC_KEYS: readonly GridCell[] = Object.freeze([
  { x: 3, y: 5 },
  { x: 7, y: 5 },
  { x: 11, y: 5 },
]);
export const MKC_GOAL: GridCell = Object.freeze({ x: 15, y: 1 });

export const MKC_X = 0;
export const MKC_Y = 1;
export const MKC_GX = 2;
export const MKC_GY = 3;
export const MKC_CARRY = 4;
export const MKC_GATES = 5;
export const MKC_ACTION = 6;

export function mkcBit(bits: number, index: number): boolean {
  return ((Math.trunc(bits) >> index) & 1) !== 0;
}

export interface MultiKeyCorridorStatic {
  seed: number;
  walls: Uint8Array;
}

export function mkcIsWall(x: number, y: number): boolean {
  if (x <= 0 || x >= 16 || y <= 0 || y >= 6) return true;
  if ((x === 4 || x === 8 || x === 12) && y !== 3) return true;
  return false;
}

export function mkcWalls(): Uint8Array {
  const walls = new Uint8Array(MKC_W * MKC_H);
  for (let y = 0; y < MKC_H; y++) {
    for (let x = 0; x < MKC_W; x++) walls[y * MKC_W + x] = mkcIsWall(x, y) ? 1 : 0;
  }
  return walls;
}

export function mkcRoomCells(room: number): GridCell[] {
  const base = MKC_ROOM_BASE[room] ?? 1;
  const out: GridCell[] = [];
  for (let i = 0; i < MKC_ROOM_CELLS; i++) {
    out.push({ x: base + (i % 3), y: 1 + Math.floor(i / 3) });
  }
  return out;
}

export const multiKeyCorridorDecoder: EnvDecoder<MultiKeyCorridorStatic> = {
  envId: 'MultiKeyCorridor-cont-v0',
  seedGeometry: false,

  initStatic(seed: number): MultiKeyCorridorStatic {
    return { seed, walls: mkcWalls() };
  },

  geometry(st: MultiKeyCorridorStatic): EnvGeometry {
    return {
      grid: {
        w: MKC_W,
        h: MKC_H,
        walls: st.walls,
        goal: { ...MKC_GOAL },
        doors: MKC_DOORS.map((c) => ({ ...c })),
        keys: MKC_KEYS.map((c) => ({ ...c })),
        hallways: MKC_DOORS.map((c) => ({ ...c })),
      },
    };
  },

  initFrame(): EnvFrame {
    const pose = new Float32Array(POSE_LEN);
    pose[MKC_X] = MKC_START.x;
    pose[MKC_Y] = MKC_START.y;
    pose[MKC_GX] = MKC_GOAL.x;
    pose[MKC_GY] = MKC_GOAL.y;
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
    const wasCarry = prev.pose[MKC_CARRY] ?? 0;
    const carry = next.pose[MKC_CARRY] ?? 0;
    const wasGates = prev.pose[MKC_GATES] ?? 0;
    const gates = next.pose[MKC_GATES] ?? 0;
    for (let j = 0; j < 3; j++) {
      if (!mkcBit(wasCarry, j) && mkcBit(carry, j)) out.push({ type: 'key', index: j });
    }
    for (let j = 0; j < 3; j++) {
      const open = mkcBit(gates, j);
      if (mkcBit(wasGates, j) !== open) out.push({ type: 'door', index: j, open });
    }
    if (goal) {
      out.push({ type: 'teleport' });
      out.push({ type: 'done' });
    }
    return out.length === 0 ? NO_EVENTS : out;
  },
};
