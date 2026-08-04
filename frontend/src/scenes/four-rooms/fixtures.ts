// Four Seasons Courtyard — canonical poses.
//
// Four states that between them exercise every accuracy claim the scene makes: the
// deterministic fleet-init pose, an agent deep in the winter room with the lantern in the
// same quadrant, an agent standing IN a hallway (the moon-gate cell, room = −1), and a
// goal that has respawned onto a hallway — which the env really does, because the
// respawn draws uniformly from all 68 free cells and four of them are thresholds.
//
// Every cell here is checked free by the env's own `fourRoomsIsWall`; the test asserts it.

import type { SceneFixture } from '@kit/types';
import {
  FOURROOMS_GOAL,
  FOURROOMS_START,
  fourRoomsRoom,
  type FourRoomsState,
} from '@sim/envs/fourrooms';

/** pose = [x, y, gx, gy, 0…] — the record-v3 FourRooms packing. */
function pose(x: number, y: number, gx: number, gy: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  return p;
}

/** ctrl[0] is the discrete action index (0..3); the other five floats are unused. */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(x: number, y: number, gx: number, gy: number): FourRoomsState {
  return { x, y, gx, gy, room: fourRoomsRoom(x, y) };
}

export const fixtures: SceneFixture[] = [
  {
    // oak_fleet_init: agent on (1,1) in the spring room, goal on (9,9) in the winter one
    name: 'courtyard-start',
    pose: pose(FOURROOMS_START.x, FOURROOMS_START.y, FOURROOMS_GOAL.x, FOURROOMS_GOAL.y),
    ctrl: ctrl(0),
    derived: derived(FOURROOMS_START.x, FOURROOMS_START.y, FOURROOMS_GOAL.x, FOURROOMS_GOAL.y),
    step: 0,
  },
  {
    // mid-run: deep in the winter room (room 3), lantern two cells away in the frost
    name: 'winter-room',
    pose: pose(7, 8, 9, 6),
    ctrl: ctrl(1),
    derived: derived(7, 8, 9, 6),
    step: 3140,
  },
  {
    // standing IN the south moon-gate (5,8): room === −1, the hallway case
    name: 'moon-gate-south',
    pose: pose(5, 8, 1, 9),
    ctrl: ctrl(3),
    derived: derived(5, 8, 1, 9),
    step: 5602,
  },
  {
    // the goal itself respawned onto the west threshold (2,5) while the agent crosses
    // the summer room — the lantern really can stand in a gateway
    name: 'goal-on-threshold',
    pose: pose(7, 2, 2, 5),
    ctrl: ctrl(2),
    derived: derived(7, 2, 2, 5),
    step: 8873,
  },
];

export default fixtures;
