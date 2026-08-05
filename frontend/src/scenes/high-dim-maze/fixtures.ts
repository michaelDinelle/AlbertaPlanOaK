
import type { SceneFixture } from '@kit/types';
import {
  HIGHDIMMAZE_DASH,
  HIGHDIMMAZE_GAPS,
  HIGHDIMMAZE_GOAL,
  HIGHDIMMAZE_STAY,
  HIGHDIMMAZE_START,
  type HighDimMazeState,
} from '@sim/envs/highdimmaze';

function pose(x: number, y: number, gx: number, gy: number, lastDir: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  p[4] = lastDir;
  return p;
}

function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(x: number, y: number, gx: number, gy: number, lastDir: number): HighDimMazeState {
  return { x, y, gx, gy, lastDir };
}

const LOW_ARCH = HIGHDIMMAZE_GAPS[0] ?? { x: 3, y: 9 };
const HIGH_ARCH = HIGHDIMMAZE_GAPS[1] ?? { x: 7, y: 1 };

export const fixtures: SceneFixture[] = [
  {
    name: 'canyon-start',
    pose: pose(HIGHDIMMAZE_START.x, HIGHDIMMAZE_START.y, HIGHDIMMAZE_GOAL.x, HIGHDIMMAZE_GOAL.y, 1),
    ctrl: ctrl(1),
    derived: derived(HIGHDIMMAZE_START.x, HIGHDIMMAZE_START.y, HIGHDIMMAZE_GOAL.x, HIGHDIMMAZE_GOAL.y, 1),
    step: 0,
  },
  {
    name: 'low-arch',
    pose: pose(LOW_ARCH.x, LOW_ARCH.y, 6, 2, 3),
    ctrl: ctrl(3),
    derived: derived(LOW_ARCH.x, LOW_ARCH.y, 6, 2, 3),
    step: 1841,
  },
  {
    name: 'high-arch',
    pose: pose(HIGH_ARCH.x, HIGH_ARCH.y, 2, 6, 3),
    ctrl: ctrl(3),
    derived: derived(HIGH_ARCH.x, HIGH_ARCH.y, 2, 6, 3),
    step: 2273,
  },
  {
    name: 'sunlit-spine',
    pose: pose(5, 5, 9, 9, 0),
    ctrl: ctrl(0),
    derived: derived(5, 5, 9, 9, 0),
    step: 4066,
  },
  {
    name: 'settled',
    pose: pose(8, 4, 1, 7, 1),
    ctrl: ctrl(HIGHDIMMAZE_STAY),
    derived: derived(8, 4, 1, 7, 1),
    step: 5602,
  },
  {
    name: 'gust-sprint',
    pose: pose(5, 3, 2, 2, 0),
    ctrl: ctrl(HIGHDIMMAZE_DASH),
    derived: derived(5, 3, 2, 2, 0),
    step: 8873,
  },
];

export default fixtures;
