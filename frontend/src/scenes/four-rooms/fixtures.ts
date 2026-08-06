
import type { SceneFixture } from '@kit/types';
import {
  FOURROOMS_GOAL,
  FOURROOMS_START,
  fourRoomsRoom,
  type FourRoomsState,
} from '@sim/envs/fourrooms';

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

function derived(x: number, y: number, gx: number, gy: number): FourRoomsState {
  return { x, y, gx, gy, room: fourRoomsRoom(x, y) };
}

export const fixtures: SceneFixture[] = [
  {
    name: 'courtyard-start',
    pose: pose(FOURROOMS_START.x, FOURROOMS_START.y, FOURROOMS_GOAL.x, FOURROOMS_GOAL.y),
    ctrl: ctrl(0),
    derived: derived(FOURROOMS_START.x, FOURROOMS_START.y, FOURROOMS_GOAL.x, FOURROOMS_GOAL.y),
    step: 0,
  },
  {
    name: 'winter-room',
    pose: pose(7, 8, 9, 6),
    ctrl: ctrl(1),
    derived: derived(7, 8, 9, 6),
    step: 3140,
  },
  {
    name: 'moon-gate-south',
    pose: pose(5, 8, 1, 9),
    ctrl: ctrl(3),
    derived: derived(5, 8, 1, 9),
    step: 5602,
  },
  {
    name: 'goal-on-threshold',
    pose: pose(7, 2, 2, 5),
    ctrl: ctrl(2),
    derived: derived(7, 2, 2, 5),
    step: 8873,
  },
];

export default fixtures;
