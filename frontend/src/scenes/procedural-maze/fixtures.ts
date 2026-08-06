
import type { SceneFixture } from '@kit/types';
import {
  PROCMAZE_GOAL,
  PROCMAZE_START,
  type ProceduralMazeState,
} from '@sim/envs/proceduralmaze';

export const FIXTURE_SEED = 96;

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

function derived(x: number, y: number, gx: number, gy: number): ProceduralMazeState {
  return { x, y, gx, gy };
}

export const fixtures: SceneFixture[] = [
  {
    name: 'ruins-start',
    pose: pose(PROCMAZE_START.x, PROCMAZE_START.y, PROCMAZE_GOAL.x, PROCMAZE_GOAL.y),
    ctrl: ctrl(3),
    derived: derived(PROCMAZE_START.x, PROCMAZE_START.y, PROCMAZE_GOAL.x, PROCMAZE_GOAL.y),
    step: 0,
  },
  {
    name: 'after-respawn',
    pose: pose(1, 9, 1, 6),
    ctrl: ctrl(0),
    derived: derived(1, 9, 1, 6),
    step: 1812,
  },
  {
    name: 'dead-end',
    pose: pose(7, 9, 5, 7),
    ctrl: ctrl(0),
    derived: derived(7, 9, 5, 7),
    step: 4820,
  },
  {
    name: 'under-the-arch',
    pose: pose(3, 3, 1, 9),
    ctrl: ctrl(1),
    derived: derived(3, 3, 1, 9),
    step: 7314,
  },
];

export default fixtures;
