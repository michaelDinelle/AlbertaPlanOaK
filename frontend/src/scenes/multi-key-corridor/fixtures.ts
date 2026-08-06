
import type { SceneFixture } from '@kit/types';
import {
  MKC_GOAL,
  MKC_KEYS,
  MKC_START,
  type MultiKeyCorridorState,
} from '@sim/envs/multikeycorridor';
import type { GridCell } from '@sim/types';

function pose(state: MultiKeyCorridorState): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = state.x;
  p[1] = state.y;
  p[2] = state.gx;
  p[3] = state.gy;
  p[4] = state.flags & 7;
  p[5] = (state.flags >> 3) & 7;
  return p;
}

function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function state(
  x: number,
  y: number,
  gx: number,
  gy: number,
  keys: readonly GridCell[],
  flags: number,
  respawnIdx: number,
): MultiKeyCorridorState {
  return {
    x,
    y,
    gx,
    gy,
    keys: keys.map((k) => ({ ...k })),
    hasKey: [(flags & 1) !== 0, (flags & 2) !== 0, (flags & 4) !== 0],
    doorOpen: [(flags & 8) !== 0, (flags & 16) !== 0, (flags & 32) !== 0],
    flags,
    respawnIdx,
    poseAgrees: true,
  };
}

function fixture(name: string, s: MultiKeyCorridorState, action: number, step: number): SceneFixture {
  return { name, pose: pose(s), ctrl: ctrl(action), derived: s, step };
}

const START_KEYS = MKC_KEYS.map((k) => ({ ...k }));
const RESPAWN_1_KEYS: GridCell[] = [
  { x: 1, y: 4 },
  { x: 7, y: 1 },
  { x: 11, y: 3 },
];

export const TEMPLE_START = state(MKC_START.x, MKC_START.y, MKC_GOAL.x, MKC_GOAL.y, START_KEYS, 0, 0);
export const TEMPLE_FORD = state(8, 3, MKC_GOAL.x, MKC_GOAL.y, START_KEYS, 27, 0);

export const fixtures: SceneFixture[] = [
  fixture('temple-start', TEMPLE_START, 0, 0),
  fixture('bramble-open', state(5, 3, 15, 1, START_KEYS, 9, 0), 3, 613),
  fixture('on-the-ford', TEMPLE_FORD, 3, 999),
  fixture('sanctum-approach', state(13, 3, 15, 1, START_KEYS, 63, 0), 3, 3020),
  fixture('fresh-respawn', state(2, 4, 13, 5, RESPAWN_1_KEYS, 0, 1), 3, 3093),
];

export default fixtures;
