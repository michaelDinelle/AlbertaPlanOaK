
import type { SceneFixture } from '@kit/types';
import { CHAIN_END } from '@sim/envs/chain';
import { chainDerived, chainPose } from './manifest';

function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

export const fixtures: SceneFixture[] = [
  {
    name: 'first-stone',
    pose: chainPose(0),
    ctrl: ctrl(0),
    derived: chainDerived(0),
    step: 0,
  },
  {
    name: 'midstream',
    pose: chainPose(7),
    ctrl: ctrl(1),
    derived: chainDerived(7),
    step: 23,
  },
  {
    name: 'last-stone',
    pose: chainPose(CHAIN_END - 1),
    ctrl: ctrl(1),
    derived: chainDerived(CHAIN_END - 1),
    step: 46,
  },
  {
    name: 'wrapped-to-start',
    pose: chainPose(0),
    ctrl: ctrl(1),
    derived: chainDerived(0),
    step: 61,
  },
];

export default fixtures;
