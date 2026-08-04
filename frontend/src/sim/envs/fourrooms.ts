// FourRooms-cont-v0 decoder.
//
// Oracles (ported, never guessed):
//   Sources/Gridworld.metal      gw_four_is_wall L83, gw_four_step L140
//   tools/gridworld_ref.py       four_is_wall L107, four_step L146
//   Sources/OakKernel.metal      pose packing L2665-2679, init L2251-2256
//
// 11x11, border walls; vertical wall col x=5 except hallways y in {2,8};
// horizontal wall row y=5 except hallways x in {2,8}; centre (5,5) is wall.
// 68 free cells. CONTINUING: on goal entry the agent KEEPS its cell and only
// the goal respawns (Philox stream 3) — so a FourRooms agent never teleports.
//
// pose = [x, y, gx, gy, 0, 0, 0, 0, 0]; done is never emitted.

import { CTRL_LEN, OFF_CTRL, OFF_DONE, OFF_POSE, POSE_LEN } from '../record';
import type { EnvDecoder, EnvFrame, EnvGeometry, GridCell, SimEvent } from '../types';
import { NO_EVENTS } from '../events';

export const FOURROOMS_W = 11;
export const FOURROOMS_H = 11;
/** free-cell count asserted by the reference (`FOUR_NFREE`) */
export const FOURROOMS_FREE_CELLS = 68;

/** the four gaps in the cross walls (the only way between rooms) */
export const FOURROOMS_HALLWAYS: readonly GridCell[] = Object.freeze([
  { x: 5, y: 2 },
  { x: 5, y: 8 },
  { x: 2, y: 5 },
  { x: 8, y: 5 },
]);

/** initial agent cell (oak_fleet_init) */
export const FOURROOMS_START: GridCell = Object.freeze({ x: 1, y: 1 });
/** initial goal cell (oak_fleet_init) */
export const FOURROOMS_GOAL: GridCell = Object.freeze({ x: 9, y: 9 });

export interface FourRoomsState {
  x: number;
  y: number;
  /** LIVE goal cell — respawns on every goal entry */
  gx: number;
  gy: number;
  /** 0=TL 1=TR 2=BL 3=BR, -1 on a hallway cell */
  room: number;
}

export interface FourRoomsStatic {
  seed: number;
  walls: Uint8Array;
}

/** `gw_four_is_wall` / `four_is_wall`. */
export function fourRoomsIsWall(x: number, y: number): boolean {
  if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
  if (x === 5 && y !== 2 && y !== 8) return true;
  if (y === 5 && x !== 2 && x !== 8) return true;
  return false;
}

/** `gw_four_room`: 0=TL 1=TR 2=BL 3=BR, -1 on a hallway. */
export function fourRoomsRoom(x: number, y: number): number {
  if (x < 5 && y < 5) return 0;
  if (x > 5 && y < 5) return 1;
  if (x < 5 && y > 5) return 2;
  if (x > 5 && y > 5) return 3;
  return -1;
}

/** w*h row-major occupancy, 1 = wall. */
export function fourRoomsWalls(): Uint8Array {
  const walls = new Uint8Array(FOURROOMS_W * FOURROOMS_H);
  for (let y = 0; y < FOURROOMS_H; y++) {
    for (let x = 0; x < FOURROOMS_W; x++) {
      walls[y * FOURROOMS_W + x] = fourRoomsIsWall(x, y) ? 1 : 0;
    }
  }
  return walls;
}

/** the 68 free cells in the reference's y-major order (`FOUR_FREE`). */
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
    // Reward condition (four_step): the agent's NEW cell is the OLD goal cell.
    // The agent keeps that cell; only the goal respawns, so there is no
    // teleport and `done` is never emitted by this env.
    if (next.state.x === prev.state.gx && next.state.y === prev.state.gy) {
      return [{ type: 'goal' }];
    }
    return NO_EVENTS;
  },
};
