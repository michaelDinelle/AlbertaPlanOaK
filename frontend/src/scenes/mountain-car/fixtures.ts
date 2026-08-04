// Alpine Valley — canonical poses.
//
// Four states that between them exercise every accuracy claim the scene makes:
// the respawn point, a real climb with momentum, the summit line itself, and the
// left wall where the env zeroes leftward velocity. `derived` matches the
// MountainCarState shape the decoder produces, so a fixture plays through the
// harness exactly like a recorded frame.

import type { SceneFixture } from '@kit/types';
import {
  MC_GOAL_POS,
  MC_MIN_POS,
  MC_START_POS,
  mountainCarHeight,
  type MountainCarState,
} from '@sim/envs/mountaincar';

/** pose = [pos, vel, goalMarker, 0…] — the record-v3 MountainCar packing. */
function pose(pos: number, vel: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = pos;
  p[1] = vel;
  p[2] = MC_GOAL_POS;
  return p;
}

/** ctrl[0] is the discrete action index; the rest of the 6 floats are unused. */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(pos: number, vel: number, action: number): MountainCarState {
  return {
    pos,
    vel,
    goal: MC_GOAL_POS,
    height: mountainCarHeight(pos),
    action,
    force: (action % 3) - 1,
  };
}

export const fixtures: SceneFixture[] = [
  {
    // the respawn / fleet-init state: parked in the basin, no momentum, engine idle
    name: 'valley-rest',
    pose: pose(MC_START_POS, 0),
    ctrl: ctrl(1),
    derived: derived(MC_START_POS, 0, 1),
    step: 0,
  },
  {
    // the hero pose: right flank, real rightward momentum, pushing right
    name: 'mid-slope-climb',
    pose: pose(-0.35, 0.04),
    ctrl: ctrl(2),
    derived: derived(-0.35, 0.04, 2),
    step: 420,
  },
  {
    // the summit line — pos === MC_GOAL_POS, the frame a `goal` fires on
    name: 'summit',
    pose: pose(MC_GOAL_POS, 0.045),
    ctrl: ctrl(2),
    derived: derived(MC_GOAL_POS, 0.045, 2),
    step: 861,
  },
  {
    // hard against the left wall, sliding back: the trail flips downhill
    name: 'left-wall-stall',
    pose: pose(MC_MIN_POS + 0.03, -0.012),
    ctrl: ctrl(0),
    derived: derived(MC_MIN_POS + 0.03, -0.012, 0),
    step: 1204,
  },
];

export default fixtures;
