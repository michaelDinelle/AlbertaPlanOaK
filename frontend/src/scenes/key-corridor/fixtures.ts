
import type { SceneFixture } from '@kit/types';
import {
  KEYCORRIDOR_GOAL,
  KEYCORRIDOR_KEY,
  KEYCORRIDOR_START,
  type KeyCorridorState,
} from '@sim/envs/keycorridor';

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
    name: 'seed-in-hand',
    pose: pose(3, 4, 7, 1, 3, 5, true, false),
    ctrl: ctrl(0),
    derived: derived(3, 4, 7, 1, 3, 5, true, false),
    step: 27,
  },
  {
    name: 'in-the-gate',
    pose: pose(4, 3, 7, 1, 3, 5, true, true),
    ctrl: ctrl(3),
    derived: derived(4, 3, 7, 1, 3, 5, true, true),
    step: 117,
  },
  {
    name: 'formal-garden',
    pose: pose(5, 2, 7, 1, 3, 5, true, true),
    ctrl: ctrl(0),
    derived: derived(5, 2, 7, 1, 3, 5, true, true),
    step: 122,
  },
  {
    name: 'respawned',
    pose: pose(2, 3, 7, 4, 2, 4, false, false),
    ctrl: ctrl(0),
    derived: derived(2, 3, 7, 4, 2, 4, false, false),
    step: 264,
  },
];

export default fixtures;
