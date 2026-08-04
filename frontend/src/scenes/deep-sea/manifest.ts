// Sunlit Descent — scene manifest (DeepSea-cont-v0).
//
// The thumbFrame holds the diver five rows down and four columns across — mid-descent,
// out on the open shelf, with the whole staircase behind it and the giant clam on the
// deepest terrace ahead. `pose = [r, c, 7, 7]` (ROW FIRST): (r = 5, c = 4) is a legal dive
// cell (c ≤ r) and one of the widest, most readable poses the env produces.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { DEEPSEA_N } from '@sim/envs/deepsea';

const THUMB_R = 5;
const THUMB_C = 4;
/** the action that carries the -0.01/N cost — the diver is pushing outward */
const THUMB_A = 1;

/** pose = [r, c, N-1, N-1, 0…] — the record-v3 DeepSea packing (row first). */
export function deepSeaPose(r: number, c: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = r;
  p[1] = c;
  p[2] = DEEPSEA_N - 1;
  p[3] = DEEPSEA_N - 1;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'DeepSea-cont-v0',
  slug: 'deep-sea',
  title: 'Sunlit Descent',
  logline:
    'An eight-step dive down a turquoise reef staircase, where the only rock that exists is ' +
    'the rock the dive can reach, the kelp leans the way the hidden action mask pushes, and a ' +
    'giant clam holds the pearl at the bottom of the last terrace.',
  accent: 'deep-sea',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: deepSeaPose(THUMB_R, THUMB_C),
    derived: { r: THUMB_R, c: THUMB_C, lastA: THUMB_A },
  },
  postConfig: {
    // the underwater column is dark and the pearl is the only HDR emitter in it, so the
    // bloom carries more of the mood here than in a daylight scene — but not enough to
    // let the caustics or the god rays start glowing on their own
    bloomStrength: 0.52,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.underwater.fog,
  },
  // Measured on Apple Silicon from the harness HUD, shadow pass included:
  //   hero    34 draws / 29.5k tris idle · 35 through a treasure dive · ~190–240 fps at
  //           1600×900 DPR 2
  //   medium  33 draws /  8.9k tris · ~170 fps
  //   thumb   18 draws /  3.0k tris · ~240 fps  (STYLE.md thumb cap: 40 / 50k)
  // The reef is cheap by construction: 36 shelves are ONE instanced draw, 36 kelp clumps
  // are another, and the caustics and the whole god-ray curtain are one merged mesh each.
  // Declared with headroom, well inside STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 60, triangles: 60000 },
};

export default manifest;
