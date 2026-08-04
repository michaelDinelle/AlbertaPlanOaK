// Sixteen Stones — canonical poses.
//
// Four REAL recorded states from lane seed 64 of `frontend-demo-16env-20260803`, chosen so
// that between them they exercise every accuracy claim the scene makes: the start stone,
// mid-crossing, the last stone the agent is ever observed on, and the wrap frame itself —
// the one where the +1 has just been paid at stone 15 and the pose has ALREADY snapped
// back to stone 0.
//
// Note what is NOT here and cannot be: a fixture at pos 15. Reaching the paying stone
// wraps in the same step, so the record never contains it. That is why stone 15 is the one
// stone that can carry a tree.

import type { SceneFixture } from '@kit/types';
import { CHAIN_END } from '@sim/envs/chain';
import { chainDerived, chainPose } from './manifest';

/** ctrl[0] is the discrete action index (0 = step back, 1 = step on); the rest are unused. */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

export const fixtures: SceneFixture[] = [
  {
    // the deterministic fleet init: the wisp on the first stone, the whole ford ahead
    name: 'first-stone',
    pose: chainPose(0),
    ctrl: ctrl(0),
    derived: chainDerived(0),
    step: 0,
  },
  {
    // mid-crossing, walking on (seed 64, step 23)
    name: 'midstream',
    pose: chainPose(7),
    ctrl: ctrl(1),
    derived: chainDerived(7),
    step: 23,
  },
  {
    // stone 14 — the last cell the agent is ever OBSERVED on, one step from the tree
    name: 'last-stone',
    pose: chainPose(CHAIN_END - 1),
    ctrl: ctrl(1),
    derived: chainDerived(CHAIN_END - 1),
    step: 46,
  },
  {
    // the wrap (seed 64, step 61): reward paid at stone 15, pose already back at stone 0.
    // Every `done` frame in this env looks exactly like this.
    name: 'wrapped-to-start',
    pose: chainPose(0),
    ctrl: ctrl(1),
    derived: chainDerived(0),
    step: 61,
  },
];

export default fixtures;
