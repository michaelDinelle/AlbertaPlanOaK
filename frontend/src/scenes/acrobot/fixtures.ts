// The Willow Swing — canonical poses.
//
// Five states that between them exercise every accuracy claim the scene makes: the
// deterministic reset (the hang the env snaps back to on every success), a real swing,
// the success surface itself (tipHeight === 1 to eleven digits), a WRAPPED pair of
// angles either side of ±π (the recorded stream wraps th2 about 1.5% of steps — see
// index.ts's arc guard), and a fast folded spin where the forearm outruns the arm.
//
// `derived` matches the AcrobotState shape the decoder produces, so a fixture plays
// through the harness exactly like a recorded frame.

import type { SceneFixture } from '@kit/types';
import { acrobotTipHeight, type AcrobotState } from '@sim/envs/acrobot';

/** pose = [th1, th2, w1, w2, 0…] — the record-v3 Acrobot packing. */
function pose(th1: number, th2: number, w1: number, w2: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = th1;
  p[1] = th2;
  p[2] = w1;
  p[3] = w2;
  return p;
}

/** ctrl[0] is the discrete action index (0/1/2 -> torque -1/0/+1). */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(th1: number, th2: number, w1: number, w2: number, action: number): AcrobotState {
  return {
    th1,
    th2,
    w1,
    w2,
    tipHeight: acrobotTipHeight(th1, th2),
    action,
    torque: (action % 3) - 1,
  };
}

/**
 * `th2` that puts the free end EXACTLY on the success surface for a given `th1`:
 * −cos(th1) − cos(th1 + th2) = 1  ⇒  th1 + th2 = acos(−1 − cos th1).
 */
export function th2ForSuccess(th1: number): number {
  return Math.acos(-1 - Math.cos(th1)) - th1;
}

const CROSSING_TH1 = 2;
const CROSSING_TH2 = th2ForSuccess(CROSSING_TH1);

export const fixtures: SceneFixture[] = [
  {
    // oak_fleet_init / every post-success reset: all four coordinates are zero
    name: 'dawn-hang',
    pose: pose(0, 0, 0, 0),
    ctrl: ctrl(1),
    derived: derived(0, 0, 0, 0, 1),
    step: 0,
  },
  {
    // the hero pose: thrown up and right, elbow driving on (+1 torque)
    name: 'swing-up',
    pose: pose(2.35, -0.5, 2.4, -1.8),
    ctrl: ctrl(2),
    derived: derived(2.35, -0.5, 2.4, -1.8, 2),
    step: 1351,
  },
  {
    // the success surface — tipHeight === 1, the frame a `goal` fires on
    name: 'crossing-the-lights',
    pose: pose(CROSSING_TH1, CROSSING_TH2, 1.6, 0.9),
    ctrl: ctrl(2),
    derived: derived(CROSSING_TH1, CROSSING_TH2, 1.6, 0.9, 2),
    step: 2088,
  },
  {
    // both angles parked either side of the ±π wrap the record stream really produces
    name: 'wrapped-fold',
    pose: pose(-3.02, 2.86, -3.1, 5.2),
    ctrl: ctrl(0),
    derived: derived(-3.02, 2.86, -3.1, 5.2, 0),
    step: 4402,
  },
  {
    // the forearm outrunning the arm: |w2| well past |w1|, elbow braking (−1 torque)
    name: 'folded-spin',
    pose: pose(-1.15, 2.3, -4.2, 8.6),
    ctrl: ctrl(0),
    derived: derived(-1.15, 2.3, -4.2, 8.6, 0),
    step: 7719,
  },
];

export default fixtures;
