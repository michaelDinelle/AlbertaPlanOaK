// The Ivy Gate — canonical poses.
//
// Five states that between them exercise every accuracy claim the scene makes, and four
// of the five are REAL FRAMES lifted from the demo run's lane (seed 24 of
// frontend-demo-16env-20260803) rather than invented: the deterministic init, the seed
// picked up while the gate is still shut, the wisp standing IN the open gate, the wisp
// through into the formal garden, and the frame immediately after a goal — which in this
// env is a full respawn (agent and seed redrawn in the wild garden, shrine redrawn in the
// formal one, gate re-locked) and therefore the single hardest pose in the stream.
//
// Every cell here is checked free by the env's own `keyCorridorIsWall`, and every key /
// goal cell is checked to be on the side of the wall the env draws it from; the test
// asserts both.

import type { SceneFixture } from '@kit/types';
import {
  KEYCORRIDOR_GOAL,
  KEYCORRIDOR_KEY,
  KEYCORRIDOR_START,
  type KeyCorridorState,
} from '@sim/envs/keycorridor';

/** pose = [x, y, gx, gy, kx, ky, hasKey, doorOpen, 0] — the record-v3 packing. */
function pose(
  x: number,
  y: number,
  gx: number,
  gy: number,
  kx: number,
  ky: number,
  hasKey: boolean,
  doorOpen: boolean,
): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  p[4] = kx;
  p[5] = ky;
  p[6] = hasKey ? 1 : 0;
  p[7] = doorOpen ? 1 : 0;
  return p;
}

/** ctrl[0] is the discrete action index (0..3); the other five floats are unused. */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(
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

export const fixtures: SceneFixture[] = [
  {
    // oak_fleet_init: the wisp on (1,1) in the wild garden, the seed on (3,5), the shrine
    // on (7,1) behind a locked gate
    name: 'garden-start',
    pose: pose(
      KEYCORRIDOR_START.x,
      KEYCORRIDOR_START.y,
      KEYCORRIDOR_GOAL.x,
      KEYCORRIDOR_GOAL.y,
      KEYCORRIDOR_KEY.x,
      KEYCORRIDOR_KEY.y,
      false,
      false,
    ),
    ctrl: ctrl(0),
    derived: derived(
      KEYCORRIDOR_START.x,
      KEYCORRIDOR_START.y,
      KEYCORRIDOR_GOAL.x,
      KEYCORRIDOR_GOAL.y,
      KEYCORRIDOR_KEY.x,
      KEYCORRIDOR_KEY.y,
      false,
      false,
    ),
    step: 0,
  },
  {
    // seed 24, step 27: the seed is in hand and the gate is still shut — the one state in
    // which the carried token and the empty bed are both on screen
    name: 'seed-in-hand',
    pose: pose(3, 4, 7, 1, 3, 5, true, false),
    ctrl: ctrl(0),
    derived: derived(3, 4, 7, 1, 3, 5, true, false),
    step: 27,
  },
  {
    // seed 24, step 117: standing IN the door cell the frame the gate opens
    name: 'in-the-gate',
    pose: pose(4, 3, 7, 1, 3, 5, true, true),
    ctrl: ctrl(3),
    derived: derived(4, 3, 7, 1, 3, 5, true, true),
    step: 117,
  },
  {
    // seed 24, step 122: through into the formal garden, gate standing open behind
    name: 'formal-garden',
    pose: pose(5, 2, 7, 1, 3, 5, true, true),
    ctrl: ctrl(0),
    derived: derived(5, 2, 7, 1, 3, 5, true, true),
    step: 122,
  },
  {
    // seed 24, step 264: the goal frame itself. Everything has moved at once — wisp back
    // in the wild garden, seed redrawn on (2,4), shrine redrawn on (7,4), gate re-locked
    name: 'respawned',
    pose: pose(2, 3, 7, 4, 2, 4, false, false),
    ctrl: ctrl(0),
    derived: derived(2, 3, 7, 4, 2, 4, false, false),
    step: 264,
  },
];

export default fixtures;
