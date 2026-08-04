// The Abyss — canonical poses.
//
// Five states that between them exercise every accuracy claim the scene makes: the
// deterministic fleet-init pose, an early shelf, a cell ON the triangle's hypotenuse
// (c === r, the only cells where the shelf edge is also a floor the agent stands on), the
// deepest pose lane seed 72 ever actually reached in 100k steps, and the one pose in the
// whole demo run from which the treasure was in reach.
//
// Every cell here satisfies the env's own reachability, c ≤ r; the test asserts it, and
// asserts that the two deep ones are the real recorded frames they claim to be:
//   seed 72, record row 4093 → (r 14, c 12)   — the deepest this lane ever got
//   seed 78, record row 8878 → (r 14, c 14)   — one step from the wreck; the next step
//                                               scored the only +1 in eight lanes

import type { SceneFixture } from '@kit/types';
import type { LargeDeepSeaState } from '@sim/envs/largedeepsea';
import { abyssPose } from './manifest';

/** ctrl[0] is the discrete action index (0 or 1); the other five floats are unused. */
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
    // oak_fleet_init / every resurface: the dive always starts at the top-left cell
    name: 'abyss-start',
    pose: abyssPose(0, 0),
    ctrl: ctrl(0),
    derived: derived(0, 0, 0),
    step: 0,
  },
  {
    // three rows down, still inside the reef's last light
    name: 'shallow-shelf',
    pose: abyssPose(3, 1),
    ctrl: ctrl(1),
    derived: derived(3, 1, 1),
    step: 320,
  },
  {
    // ON the hypotenuse: c === r is the shelf's own edge, and it is a floor cell
    name: 'shelf-edge',
    pose: abyssPose(6, 6),
    ctrl: ctrl(1),
    derived: derived(6, 6, 1),
    step: 1290,
  },
  {
    // seed 72, record row 4093 — the deepest, rightmost cell this lane ever occupied
    name: 'deep-dive',
    pose: abyssPose(14, 12),
    ctrl: ctrl(0),
    derived: derived(14, 12, 0),
    step: 4093,
  },
  {
    // seed 78, record row 8878 — one right-move from the wreck. The next frame wrapped
    // to (0, 0) and the decoder recovered the treasure from the re-simulated step.
    name: 'treasure-reach',
    pose: abyssPose(14, 14),
    ctrl: ctrl(0),
    derived: derived(14, 14, 0),
    step: 8878,
  },
];

export default fixtures;
