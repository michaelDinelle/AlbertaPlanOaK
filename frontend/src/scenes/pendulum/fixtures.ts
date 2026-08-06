
import type { SceneFixture } from '@kit/types';
import { PD_MAX_TORQUE } from '@sim/envs/pendulum';
import { pendulumDerived, pendulumPose } from './manifest';

function fixture(name: string, th: number, thDot: number, torque: number, step: number): SceneFixture {
  return {
    name,
    pose: pendulumPose(th, thDot, torque),
    ctrl: [torque / PD_MAX_TORQUE, 0, 0, 0, 0, 0],
    derived: pendulumDerived(th, thDot, torque, torque / PD_MAX_TORQUE),
    step,
  };
}

export const fixtures: SceneFixture[] = [
  fixture('at-rest', 0, 0, 0, 0),
  fixture('inverted-init', Math.PI, 0, 0, -1),
  fixture('swing-right', 1.2, 4.2, 1.5, 420),
  fixture('wrap-edge', 3.088365, 2.621751, -0.723766, 39),
  fixture('wrap-edge-slow', 3.137722, 0.013266, 0.301683, 3),
  fixture('hard-push-left', -0.9, -6, -PD_MAX_TORQUE, 1204),
];

export default fixtures;
