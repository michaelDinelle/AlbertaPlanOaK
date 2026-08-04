// Sunlit Descent — canonical poses.
//
// Five states that between them exercise every accuracy claim the scene makes: the
// deterministic fleet-init pose at the surface, a shallow step, the left wall the column
// clamp holds the diver against, the frontier cell of a mid-row (the sunlit lip of the
// reachable triangle), and the last cell of the last dive row — one right move from the
// clam, which is the only pose from which the treasure can be taken.
//
// Every cell here is on the reef by the env's OWN rule (a dive at row r can only be in
// columns 0..r, and row N−2 is the deepest it ever occupies); the test asserts it against
// `reefReachable`, which runs the decoder's transition rather than trusting this comment.

import type { SceneFixture } from '@kit/types';
import { DEEPSEA_LAST_ROW, DEEPSEA_N, type DeepSeaState } from '@sim/envs/deepsea';
import { deepSeaPose } from './manifest';

/** ctrl[0] is the discrete action index (0 or 1); the other five floats are unused. */
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
    // oak_fleet_init: the diver at the surface, top of the column, on the single cell
    // row 0 can hold
    name: 'surface-start',
    pose: deepSeaPose(0, 0),
    ctrl: ctrl(0),
    derived: derived(0, 0, 0),
    step: 0,
  },
  {
    // three rows down, out on the shelf
    name: 'mid-dive',
    pose: deepSeaPose(3, 2),
    ctrl: ctrl(1),
    derived: derived(3, 2, 1),
    step: 1523,
  },
  {
    // pinned to column 0 five rows down — the clamp really does hold a dive at the wall
    name: 'left-wall',
    pose: deepSeaPose(5, 0),
    ctrl: ctrl(0),
    derived: derived(5, 0, 0),
    step: 3311,
  },
  {
    // the frontier of the reachable triangle at row 4: the sunlit lip of that shelf
    name: 'shelf-lip',
    pose: deepSeaPose(4, 4),
    ctrl: ctrl(1),
    derived: derived(4, 4, 1),
    step: 5006,
  },
  {
    // the deepest cell a dive can occupy: one right move from the clam, which is the
    // ONLY way the treasure is ever taken
    name: 'treasure-approach',
    pose: deepSeaPose(DEEPSEA_LAST_ROW, DEEPSEA_N - 2),
    ctrl: ctrl(1),
    derived: derived(DEEPSEA_LAST_ROW, DEEPSEA_N - 2, 1),
    step: 8807,
  },
];

export default fixtures;
