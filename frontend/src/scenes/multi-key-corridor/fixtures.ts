// Temple of Three Gates — canonical poses.
//
// Every one of these is a REAL decoded state, lifted out of the demo run's own lane
// (`frontend-demo-16env-20260803`, MultiKeyCorridor seed 80) by replaying its ctrl
// stream through the decoder — which is the only way key cells can be known at all,
// since the record's 9-float pose carries the agent, the goal and the two bitmasks but
// not the keys. Between them they exercise every accuracy claim the scene makes:
//
//   * the deterministic fleet init (all three gates shut, keys on their opening cells),
//   * one key held and the bramble gate standing open,
//   * the wisp STANDING ON the river gate's own cell — the threshold case, where the
//     floor tile is a basin and the stepping stones are up,
//   * every key held, every gate open, one cell from the sanctum, and
//   * the first frame after a completion: the agent thrown back to chamber 0, three
//     freshly drawn keys, a freshly drawn relic, and all three gates relocked.
//
// The test asserts every agent/goal/key cell here is a free cell of the real layout.

import type { SceneFixture } from '@kit/types';
import {
  MKC_GOAL,
  MKC_KEYS,
  MKC_START,
  type MultiKeyCorridorState,
} from '@sim/envs/multikeycorridor';
import type { GridCell } from '@sim/types';

/** pose = [x, y, gx, gy, keyMask3, doorMask3, 0, 0, 0] — the record-v3 MKC packing. */
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

/** ctrl[0] is the discrete action index (0..3); the other five floats are unused. */
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

/** the keys as the env's deterministic init places them */
const START_KEYS = MKC_KEYS.map((k) => ({ ...k }));
/** the cells the env drew for respawn #1 on seed 80 */
const RESPAWN_1_KEYS: GridCell[] = [
  { x: 1, y: 4 },
  { x: 7, y: 1 },
  { x: 11, y: 3 },
];

export const TEMPLE_START = state(MKC_START.x, MKC_START.y, MKC_GOAL.x, MKC_GOAL.y, START_KEYS, 0, 0);
/** seed 80, step 999 — the thumbnail pose: mid-temple, on the ford, two keys carried */
export const TEMPLE_FORD = state(8, 3, MKC_GOAL.x, MKC_GOAL.y, START_KEYS, 27, 0);

export const fixtures: SceneFixture[] = [
  // oak_fleet_init: the wisp at the temple mouth, the relic at (15,1), three gates shut
  fixture('temple-start', TEMPLE_START, 0, 0),
  // seed 80, step 613: the seed key is taken and the bramble has unwound
  fixture('bramble-open', state(5, 3, 15, 1, START_KEYS, 9, 0), 3, 613),
  // seed 80, step 999: standing ON the river gate's cell — the threshold case
  fixture('on-the-ford', TEMPLE_FORD, 3, 999),
  // seed 80, step 3020: all three keys, all three gates, one cell short of the sanctum
  fixture('sanctum-approach', state(13, 3, 15, 1, START_KEYS, 63, 0), 3, 3020),
  // seed 80, step 3093: the frame after the first completion — new cells, gates relocked
  fixture('fresh-respawn', state(2, 4, 13, 5, RESPAWN_1_KEYS, 0, 1), 3, 3093),
];

export default fixtures;
