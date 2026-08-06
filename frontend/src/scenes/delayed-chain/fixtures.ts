
import type { SceneFixture } from '@kit/types';
import { DELAYEDCHAIN_END, DELAYEDCHAIN_L, type DelayedChainState } from '@sim/envs/delayedchain';
import { delayedChainPose } from './manifest';

function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(pos: number, lastA: number): DelayedChainState {
  return { pos, lastA, len: DELAYEDCHAIN_L };
}

export const fixtures: SceneFixture[] = [
  {
    name: 'road-start',
    pose: delayedChainPose(0, 0),
    ctrl: ctrl(1),
    derived: derived(0, 0),
    step: 0,
  },
  {
    name: 'mid-moor',
    pose: delayedChainPose(15, 1),
    ctrl: ctrl(1),
    derived: derived(15, 1),
    step: 3140,
  },
  {
    name: 'aurora-doorstep',
    pose: delayedChainPose(29, 0),
    ctrl: ctrl(0),
    derived: derived(29, 0),
    step: 474,
  },
  {
    name: 'last-observed-cell',
    pose: delayedChainPose(30, 1),
    ctrl: ctrl(1),
    derived: derived(30, 1),
    step: 475,
  },
  {
    name: 'the-wrap',
    pose: delayedChainPose(0, 1),
    ctrl: ctrl(1),
    derived: derived(0, 1),
    step: 476,
  },
];

export const DELAYEDCHAIN_GOAL_CELL = DELAYEDCHAIN_END;

export default fixtures;
