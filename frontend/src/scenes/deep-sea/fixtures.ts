
import type { SceneFixture } from '@kit/types';
import { DEEPSEA_LAST_ROW, DEEPSEA_N, type DeepSeaState } from '@sim/envs/deepsea';
import { deepSeaPose } from './manifest';

function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(r: number, c: number, lastA: number): DeepSeaState {
  return { r, c, lastA };
}

export const fixtures: SceneFixture[] = [
  {
    name: 'surface-start',
    pose: deepSeaPose(0, 0),
    ctrl: ctrl(0),
    derived: derived(0, 0, 0),
    step: 0,
  },
  {
    name: 'mid-dive',
    pose: deepSeaPose(3, 2),
    ctrl: ctrl(1),
    derived: derived(3, 2, 1),
    step: 1523,
  },
  {
    name: 'left-wall',
    pose: deepSeaPose(5, 0),
    ctrl: ctrl(0),
    derived: derived(5, 0, 0),
    step: 3311,
  },
  {
    name: 'shelf-lip',
    pose: deepSeaPose(4, 4),
    ctrl: ctrl(1),
    derived: derived(4, 4, 1),
    step: 5006,
  },
  {
    name: 'treasure-approach',
    pose: deepSeaPose(DEEPSEA_LAST_ROW, DEEPSEA_N - 2),
    ctrl: ctrl(1),
    derived: derived(DEEPSEA_LAST_ROW, DEEPSEA_N - 2, 1),
    step: 8807,
  },
];

export default fixtures;
