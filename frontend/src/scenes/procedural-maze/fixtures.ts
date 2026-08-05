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
//   * a real mid-run respawn lifted straight out of lane 96's recording (step 1812);
//   * a goal that has respawned into a DEAD END — (5,7) on this seed is a free cell
//     walled on three sides. The maze is a perfect maze (every free cell reachable),
//     but the art still states only the goal, never the route to it: no trail, no
//     arrow, no hint that the beam is a promise;
//   * the wisp standing inside a ruined doorway — (3,3) is walled east and west on this
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
    // lane 96, step 1812: the first completion — agent and goal freshly redrawn
    name: 'after-respawn',
    pose: pose(1, 9, 1, 6),
    ctrl: ctrl(0),
    derived: derived(1, 9, 1, 6),
    step: 1812,
  },
  {
    // the goal has respawned into a dead end — (5,7) is walled on three sides; the
    // scene states the goal, never the route to it
    name: 'dead-end',
    pose: pose(7, 9, 5, 7),
    ctrl: ctrl(0),
    derived: derived(7, 9, 5, 7),
    step: 4820,
  },
  {
    // standing inside a ruined doorway: (3,3) is walled east and west on this seed
    name: 'under-the-arch',
    pose: pose(3, 3, 1, 9),
    ctrl: ctrl(1),
    derived: derived(3, 3, 1, 9),
    step: 7314,
  },
];

export default fixtures;
