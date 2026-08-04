// The Fogbound Forest — canonical poses.
//
// Six states that between them exercise every accuracy claim the scene makes: the
// deterministic fleet-init pose, the wisp standing IN each of the two gaps that stitch
// the three bands together (the gate-stone probes), the far corner, the one open cell in
// the middle band where the bubble contains NO wall at all, and a goal one cell away —
// the state the scene must render exactly as it renders a goal across the maze.
//
// Every cell here is checked free by the env's own `poMazeIsWall`; the test asserts it.

import type { SceneFixture } from '@kit/types';
import { POMAZE_GAPS, POMAZE_GOAL, POMAZE_START, type POMazeState } from '@sim/envs/pomaze';

/** pose = [x, y, gx, gy, 0…] — the record-v3 POMaze packing. */
function pose(x: number, y: number, gx: number, gy: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  return p;
}

/** ctrl[0] is the discrete action index (0 = up, 1 = down, 2 = left, 3 = right). */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(x: number, y: number, gx: number, gy: number): POMazeState {
  return { x, y, gx, gy };
}

const NORTH_GAP = POMAZE_GAPS[0] ?? { x: 8, y: 3 };
const SOUTH_GAP = POMAZE_GAPS[1] ?? { x: 2, y: 7 };

export const fixtures: SceneFixture[] = [
  {
    // oak_fleet_init: the wisp in the north-west corner nook, the goal in the far south-east
    name: 'forest-start',
    pose: pose(POMAZE_START.x, POMAZE_START.y, POMAZE_GOAL.x, POMAZE_GOAL.y),
    ctrl: ctrl(3),
    derived: derived(POMAZE_START.x, POMAZE_START.y, POMAZE_GOAL.x, POMAZE_GOAL.y),
    step: 0,
  },
  {
    // standing IN the north gap (8,3) — the only way through the first band, with the
    // south gap as the goal: both gate-stone probes are in play at once
    name: 'north-gap',
    pose: pose(NORTH_GAP.x, NORTH_GAP.y, SOUTH_GAP.x, SOUTH_GAP.y),
    ctrl: ctrl(1),
    derived: derived(NORTH_GAP.x, NORTH_GAP.y, SOUTH_GAP.x, SOUTH_GAP.y),
    step: 1284,
  },
  {
    // and in the south gap (2,7), goal back up in the first band
    name: 'south-gap',
    pose: pose(SOUTH_GAP.x, SOUTH_GAP.y, 5, 1),
    ctrl: ctrl(0),
    derived: derived(SOUTH_GAP.x, SOUTH_GAP.y, 5, 1),
    step: 3894,
  },
  {
    // the deep south-east corner: five of the eight neighbours are border wall
    name: 'deep-corner',
    pose: pose(9, 9, 1, 1),
    ctrl: ctrl(3),
    derived: derived(9, 9, 1, 1),
    step: 4807,
  },
  {
    // dead centre of the middle band: NOT ONE wall in the bubble, and the goal is due
    // north (gx == x, so both east/west compass bits are 0) — the moth's degenerate case
    name: 'open-middle',
    pose: pose(5, 5, 5, 1),
    ctrl: ctrl(0),
    derived: derived(5, 5, 5, 1),
    step: 9327,
  },
  {
    // the goal is on the north-east diagonal, one cell away — the state the frame must
    // give away nothing about. The agent's four sign bits read "north-east" here exactly
    // as they would for a goal four cells off, so the picture must be the same one: no
    // flower, and the moth still out on the rim. The honesty tests pin it.
    name: 'goal-adjacent',
    pose: pose(4, 5, 5, 4),
    ctrl: ctrl(3),
    derived: derived(4, 5, 5, 4),
    step: 13492,
  },
];

export default fixtures;
