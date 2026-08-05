
import type { SceneFixture } from '@kit/types';
import {
  CP_FORCE_MAG,
  CP_TH_LIMIT,
  CP_X_LIMIT,
  type CartPoleState,
} from '@sim/envs/cartpole';

function pose(x: number, xDot: number, th: number, thDot: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = xDot;
  p[2] = th;
  p[3] = thDot;
  return p;
}

function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(x: number, xDot: number, th: number, thDot: number, action: number): CartPoleState {
  return {
    x,
    xDot,
    th,
    thDot,
    action,
    force: (action & 1) !== 0 ? CP_FORCE_MAG : -CP_FORCE_MAG,
    xMargin: Math.min(1, Math.abs(x) / CP_X_LIMIT),
    thMargin: Math.min(1, Math.abs(th) / CP_TH_LIMIT),
  };
}

export const fixtures: SceneFixture[] = [
  {
    name: 'upright-reset',
    pose: pose(0, 0, 0, 0),
    ctrl: ctrl(0),
    derived: derived(0, 0, 0, 0, 0),
    step: 0,
  },
  {
    name: 'balancing-right',
    pose: pose(-0.34, 0.46, 0.098, -0.62),
    ctrl: ctrl(1),
    derived: derived(-0.34, 0.46, 0.098, -0.62, 1),
    step: 1174,
  },
  {
    name: 'tilt-limit-left',
    pose: pose(-0.62, -0.71, -CP_TH_LIMIT, -1.35),
    ctrl: ctrl(1),
    derived: derived(-0.62, -0.71, -CP_TH_LIMIT, -1.35, 1),
    step: 2680,
  },
  {
    name: 'far-right-drift',
    pose: pose(1.1875, 0.38, 0.043, 0.22),
    ctrl: ctrl(0),
    derived: derived(1.1875, 0.38, 0.043, 0.22, 0),
    step: 41209,
  },
  {
    name: 'near-track-end',
    pose: pose(CP_X_LIMIT - 0.06, 1.42, -0.031, 0.18),
    ctrl: ctrl(1),
    derived: derived(CP_X_LIMIT - 0.06, 1.42, -0.031, 0.18, 1),
    step: 60332,
  },
];

export default fixtures;
