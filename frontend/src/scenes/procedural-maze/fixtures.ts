// The Overgrown Ruins — canonical poses.
//
// SEED-SPECIFIC BY NECESSITY. Every other grid env has one layout; this one regenerates
// its 9×9 interior from the lane seed, so a fixture is only meaningful together with the
// maze it was picked out of. These four are cells of lane 96 — the first ProceduralMaze
// lane of the demo run and the one the harness loads by default — and the test asserts
// each one against `procMazeIsWall` for that seed rather than trusting this comment.
//
// Between them they exercise every claim the scene makes:
//   * the deterministic fleet init, (1,1) → (9,9);
//   * a real mid-run respawn lifted straight out of lane 96's recording (step 1377);
//   * a goal that has respawned into a SEALED POCKET — (6,6) on this seed is a free cell
//     walled on all four sides, so the sunbeam is falling somewhere the agent physically
//     cannot reach. That is a real property of this env, and the art must not pretend
//     otherwise: no trail, no arrow, no hint that the beam is a promise;
//   * the wisp standing inside a ruined doorway — (4,8) is walled east and west on this
//     seed, which is exactly what `archCells()` looks for.

import type { SceneFixture } from '@kit/types';
import {
  PROCMAZE_GOAL,
  PROCMAZE_START,
  type ProceduralMazeState,
} from '@sim/envs/proceduralmaze';

/** the lane these cells belong to — the harness's default ProceduralMaze recording */
export const FIXTURE_SEED = 96;

/** pose = [x, y, gx, gy, 0…] — the record-v3 ProceduralMaze packing. */
function pose(x: number, y: number, gx: number, gy: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  return p;
}

/** ctrl[0] is the discrete action index (0=up 1=down 2=left 3=right); the rest is unused. */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(x: number, y: number, gx: number, gy: number): ProceduralMazeState {
  return { x, y, gx, gy };
}

export const fixtures: SceneFixture[] = [
  {
    // oak_fleet_init: the agent at the mouth of the maze, the goal at its far corner
    name: 'ruins-start',
    pose: pose(PROCMAZE_START.x, PROCMAZE_START.y, PROCMAZE_GOAL.x, PROCMAZE_GOAL.y),
    ctrl: ctrl(3),
    derived: derived(PROCMAZE_START.x, PROCMAZE_START.y, PROCMAZE_GOAL.x, PROCMAZE_GOAL.y),
    step: 0,
  },
  {
    // lane 96, step 1377: both agent and goal were redrawn two steps ago
    name: 'after-respawn',
    pose: pose(4, 3, 2, 5),
    ctrl: ctrl(2),
    derived: derived(4, 3, 2, 5),
    step: 1377,
  },
  {
    // the goal has respawned into a cell walled on all four sides — unreachable, and the
    // scene must say nothing about that either way
    name: 'sealed-pocket',
    pose: pose(4, 9, 6, 6),
    ctrl: ctrl(0),
    derived: derived(4, 9, 6, 6),
    step: 4820,
  },
  {
    // standing inside a ruined doorway: (4,8) is walled east and west on this seed
    name: 'under-the-arch',
    pose: pose(4, 8, 1, 9),
    ctrl: ctrl(1),
    derived: derived(4, 8, 1, 9),
    step: 7314,
  },
];

export default fixtures;
