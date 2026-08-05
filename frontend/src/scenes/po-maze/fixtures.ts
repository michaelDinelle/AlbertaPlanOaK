
import type { SceneFixture } from '@kit/types';
import { POMAZE_GAPS, POMAZE_GOAL, POMAZE_START, type POMazeState } from '@sim/envs/pomaze';

function pose(x: number, y: number, gx: number, gy: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  return p;
}

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
    name: 'forest-start',
    pose: pose(POMAZE_START.x, POMAZE_START.y, POMAZE_GOAL.x, POMAZE_GOAL.y),
    ctrl: ctrl(3),
    derived: derived(POMAZE_START.x, POMAZE_START.y, POMAZE_GOAL.x, POMAZE_GOAL.y),
    step: 0,
  },
  {
    name: 'north-gap',
    pose: pose(NORTH_GAP.x, NORTH_GAP.y, SOUTH_GAP.x, SOUTH_GAP.y),
    ctrl: ctrl(1),
    derived: derived(NORTH_GAP.x, NORTH_GAP.y, SOUTH_GAP.x, SOUTH_GAP.y),
    step: 1284,
  },
  {
    name: 'south-gap',
    pose: pose(SOUTH_GAP.x, SOUTH_GAP.y, 5, 1),
    ctrl: ctrl(0),
    derived: derived(SOUTH_GAP.x, SOUTH_GAP.y, 5, 1),
    step: 3894,
  },
  {
    name: 'deep-corner',
    pose: pose(9, 9, 1, 1),
    ctrl: ctrl(3),
    derived: derived(9, 9, 1, 1),
    step: 4807,
  },
  {
    name: 'open-middle',
    pose: pose(5, 5, 5, 1),
    ctrl: ctrl(0),
    derived: derived(5, 5, 5, 1),
    step: 9327,
  },
  {
    name: 'goal-adjacent',
    pose: pose(4, 5, 5, 4),
    ctrl: ctrl(3),
    derived: derived(4, 5, 5, 4),
    step: 13492,
  },
];

export default fixtures;
