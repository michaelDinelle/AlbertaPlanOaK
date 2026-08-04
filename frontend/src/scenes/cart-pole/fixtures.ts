// The Boardwalk — canonical poses.
//
// Five states that between them exercise every accuracy claim the scene makes: the
// deterministic reset the env snaps back to on every failure, a real save mid-lean, the
// exact tilt-failure angle (the frame where the sapling's crown sits ON the surveyor's
// cord), the furthest drift actually recorded in the demo run, and the far end of the
// boardwalk where |x| runs out. `derived` matches the CartPoleState shape the decoder
// produces, so a fixture plays through the harness exactly like a recorded frame.

import type { SceneFixture } from '@kit/types';
import {
  CP_FORCE_MAG,
  CP_TH_LIMIT,
  CP_X_LIMIT,
  type CartPoleState,
} from '@sim/envs/cartpole';

/** pose = [x, xdot, th, thdot, 0…] — the record-v3 CartPole packing. */
function pose(x: number, xDot: number, th: number, thDot: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = xDot;
  p[2] = th;
  p[3] = thDot;
  return p;
}

/** ctrl[0] is the discrete action bit; the other five floats are unused. */
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
    // hc_resetData: the all-zero upright state. Every failure lands back here IN THE
    // SAME STEP, so this is by far the most-rendered pose in the whole recording.
    name: 'upright-reset',
    pose: pose(0, 0, 0, 0),
    ctrl: ctrl(0),
    derived: derived(0, 0, 0, 0, 0),
    step: 0,
  },
  {
    // the hero pose: rolling right under a rightward shove, crown leaning right and
    // already swinging back — a save in progress
    name: 'balancing-right',
    pose: pose(-0.34, 0.46, 0.098, -0.62),
    ctrl: ctrl(1),
    derived: derived(-0.34, 0.46, 0.098, -0.62, 1),
    step: 1174,
  },
  {
    // EXACTLY on the tilt bound: the crown sits on the surveyor's cord at
    // y = cos(12°), which is the frame after which the env terminates
    name: 'tilt-limit-left',
    pose: pose(-0.62, -0.71, -CP_TH_LIMIT, -1.35),
    ctrl: ctrl(1),
    derived: derived(-0.62, -0.71, -CP_TH_LIMIT, -1.35, 1),
    step: 2680,
  },
  {
    // the furthest the trolley actually travelled in the demo run (seed 48, |x| max)
    name: 'far-right-drift',
    pose: pose(1.1875, 0.38, 0.043, 0.22),
    ctrl: ctrl(0),
    derived: derived(1.1875, 0.38, 0.043, 0.22, 0),
    step: 41209,
  },
  {
    // the OTHER failure bound: the trolley one plank short of the boardwalk's end,
    // where the deck simply stops over open water
    name: 'near-track-end',
    pose: pose(CP_X_LIMIT - 0.06, 1.42, -0.031, 0.18),
    ctrl: ctrl(1),
    derived: derived(CP_X_LIMIT - 0.06, 1.42, -0.031, 0.18, 1),
    step: 60332,
  },
];

export default fixtures;
