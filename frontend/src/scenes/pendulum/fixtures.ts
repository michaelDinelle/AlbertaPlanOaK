// The Lantern Bell — canonical poses.
//
// Six states that between them exercise every accuracy claim the scene makes: the cost
// minimum (hanging, full glow), the fleet init (inverted, the frame at its darkest), a
// real mid-swing with momentum, BOTH recorded wraps on lane 56 — the fast one at |θ̇|=2.6
// and the slow one at |θ̇|=0.01, which the angle-wrap guard has to catch by two different
// bounds — and a full negative-torque push. `derived` matches the decoder's own
// PendulumState, so a fixture plays through the harness exactly like a recorded frame.
//
// Every θ/θ̇/torque below is a verbatim row of
// `.oak-benchmarks/frontend-demo-16env-20260803/Pendulum-cont-v0/seed56/actions.bin`
// (except the two synthetic ones), and `step` is that row's own step index — the decoder
// numbers row i as step i, so a fixture's step is directly comparable to `state.step`.

import type { SceneFixture } from '@kit/types';
import { PD_MAX_TORQUE } from '@sim/envs/pendulum';
import { pendulumDerived, pendulumPose } from './manifest';

function fixture(name: string, th: number, thDot: number, torque: number, step: number): SceneFixture {
  return {
    name,
    pose: pendulumPose(th, thDot, torque),
    // Box(1): ctrl[0] is the raw action in [-1, 1]; the kernel scales it by 2.
    ctrl: [torque / PD_MAX_TORQUE, 0, 0, 0, 0, 0],
    derived: pendulumDerived(th, thDot, torque, torque / PD_MAX_TORQUE),
    step,
  };
}

export const fixtures: SceneFixture[] = [
  // the cost minimum: hanging dead still. cost 0 — the pond at its warmest.
  fixture('at-rest', 0, 0, 0, 0),
  // the fleet init (gw[0] = π): balanced at the top, about to fall. The darkest pose the
  // scene renders at rest — though not the darkest it CAN render (see 'wrap-edge').
  fixture('inverted-init', Math.PI, 0, 0, -1),
  // a real swing through the right quarter with momentum and a positive push
  fixture('swing-right', 1.2, 4.2, 1.5, 420),
  // lane 56 row 39: the last frame before the FAST recorded wrap at +π. The host's linear
  // lerp of row 39 → row 40 is the fabricated revolution the guard exists to refuse.
  fixture('wrap-edge', 3.088365, 2.621751, -0.723766, 39),
  // lane 56 row 3: the SLOW wrap — the bob tips over the top at |θ̇| = 0.013, far below
  // the speed at which the guard's direction test is allowed to fire, so this one has to
  // be caught by the reach bound alone.
  fixture('wrap-edge-slow', 3.137722, 0.013266, 0.301683, 3),
  // full negative torque on the way back down: the curl mirrors to −θ
  fixture('hard-push-left', -0.9, -6, -PD_MAX_TORQUE, 1204),
];

export default fixtures;
