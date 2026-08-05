
import type { SceneFixture } from '@kit/types';
import { MEMORYCHAIN_END } from '@sim/envs/memorychain';
import { memoryChainDerived, memoryChainPose } from './manifest';

function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

export const fixtures: SceneFixture[] = [
  {
    name: 'cue-cyan-lit',
    pose: memoryChainPose(0, 0),
    ctrl: ctrl(0),
    derived: memoryChainDerived(0, 0, 0),
    step: -1,
  },
  {
    name: 'cue-magenta-lit',
    pose: memoryChainPose(0, 1),
    ctrl: ctrl(0),
    derived: memoryChainDerived(0, 1, 0),
    step: 12105,
  },
  {
    name: 'carrying-the-cue',
    pose: memoryChainPose(14, 1),
    ctrl: ctrl(1),
    derived: memoryChainDerived(14, 1, 1),
    step: 12312,
  },
  {
    name: 'at-the-choice',
    pose: memoryChainPose(MEMORYCHAIN_END, 1),
    ctrl: ctrl(1),
    derived: memoryChainDerived(MEMORYCHAIN_END, 1, 1),
    step: 13122,
  },
  {
    name: 'correct-choice-respawn',
    pose: memoryChainPose(0, 0),
    ctrl: ctrl(1),
    derived: memoryChainDerived(0, 0, 1),
    step: 13123,
  },
  {
    name: 'wrong-choice-respawn',
    pose: memoryChainPose(0, 1),
    ctrl: ctrl(1),
    derived: memoryChainDerived(0, 1, 1),
    step: 16155,
  },
];

export default fixtures;
