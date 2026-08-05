
import type { SceneFixture } from '@kit/types';
import type { LargeDeepSeaState } from '@sim/envs/largedeepsea';
import { abyssPose } from './manifest';

function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(r: number, c: number, lastA: number): LargeDeepSeaState {
  return { r, c, lastA };
}

export const fixtures: SceneFixture[] = [
  {
    name: 'abyss-start',
    pose: abyssPose(0, 0),
    ctrl: ctrl(0),
    derived: derived(0, 0, 0),
    step: 0,
  },
  {
    name: 'shallow-shelf',
    pose: abyssPose(3, 1),
    ctrl: ctrl(1),
    derived: derived(3, 1, 1),
    step: 320,
  },
  {
    name: 'shelf-edge',
    pose: abyssPose(6, 6),
    ctrl: ctrl(1),
    derived: derived(6, 6, 1),
    step: 1290,
  },
  {
    name: 'deep-dive',
    pose: abyssPose(14, 12),
    ctrl: ctrl(0),
    derived: derived(14, 12, 0),
    step: 4093,
  },
  {
    name: 'treasure-reach',
    pose: abyssPose(14, 14),
    ctrl: ctrl(0),
    derived: derived(14, 14, 0),
    step: 8878,
  },
];

export default fixtures;
