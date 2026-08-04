// The Ivy Gate — scene manifest (KeyCorridor-cont-v0).
//
// The thumbFrame is the scene's whole argument in one still: the wisp two cells into the
// wild garden, the amber seed still lying in its bed a step away, the gate SEWN SHUT with
// ivy dead centre, and the shrine burning on the far side of a wall the agent cannot yet
// cross. Locked, unpicked, unreached — every one of the four state bits at its starting
// value, which is also the most beautiful of the four combinations.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import type { KeyCorridorState } from '@sim/envs/keycorridor';

const THUMB_X = 2;
const THUMB_Y = 4;
const THUMB_GX = 6;
const THUMB_GY = 2;
const THUMB_KX = 3;
const THUMB_KY = 2;

/** pose = [x, y, gx, gy, kx, ky, hasKey, doorOpen, 0] — the record-v3 KeyCorridor packing. */
function pose(
  x: number,
  y: number,
  gx: number,
  gy: number,
  kx: number,
  ky: number,
  hasKey: boolean,
  doorOpen: boolean,
): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  p[4] = kx;
  p[5] = ky;
  p[6] = hasKey ? 1 : 0;
  p[7] = doorOpen ? 1 : 0;
  return p;
}

const thumbDerived: KeyCorridorState = {
  x: THUMB_X,
  y: THUMB_Y,
  gx: THUMB_GX,
  gy: THUMB_GY,
  kx: THUMB_KX,
  ky: THUMB_KY,
  hasKey: false,
  doorOpen: false,
};

export const manifest: SceneManifest = {
  envId: 'KeyCorridor-cont-v0',
  slug: 'key-corridor',
  title: 'The Ivy Gate',
  logline:
    'A morning garden split by one ivy-clad wall: a wild moss side where an amber seed ' +
    'waits in its stone dish, a swept gravel court where a lantern shrine burns, and ' +
    'between them a single oak door that the ivy visibly unpicks itself from when the key ' +
    'turns — and sews shut again the moment the shrine is reached.',
  accent: 'key-corridor',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pose(THUMB_X, THUMB_Y, THUMB_GX, THUMB_GY, THUMB_KX, THUMB_KY, false, false),
    derived: thumbDerived,
  },
  postConfig: {
    // The garden has three bloom feeders (seed, ember, celebration) and TWO OF THEM SHARE
    // A CELL whenever the wisp is carrying the seed. At 0.42 that pair fused into one
    // blown white sun over the wild garden and took three cells of grid with it; 0.34 keeps
    // the amber reading as light and lets the two stay two things.
    bloomStrength: 0.3,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.morning.fog,
  },
  // Read straight off the harness HUD at 1600 × 900, shadow pass included (the composer
  // adds ~15 more, which the HUD reports separately as "total"):
  //   hero    46 draws /  91.7k tris · 210–235 fps  ·  51 draws with the overlay on
  //   medium  44 draws /  30.1k tris · 240 fps
  //   thumb   28 draws /   8.0k tris · 240 fps   (STYLE.md thumb cap: 40 / 50k)
  // The vine, the ivy wall, the floor, the kerb and every scatter are ONE instanced draw
  // each, which is where the headroom comes from. Far inside STYLE.md's 300 / 500k hero
  // cap; declared with enough room that a future prop does not need a re-measure.
  budget: { drawCalls: 64, triangles: 120000 },
};

export default manifest;
