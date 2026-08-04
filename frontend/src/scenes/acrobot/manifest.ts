// The Willow Swing — scene manifest (Acrobot-cont-v0).
//
// The thumbFrame is a real, reachable swing state: the upper arm thrown up and right,
// the forearm folded slightly back, and the lantern at tipHeight 0.978 — a hand's width
// under the string of lights it is trying to reach. It reads as the moment before a
// success, which is the whole story of this env.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { acrobotTipHeight } from '@sim/envs/acrobot';

const THUMB_TH1 = 2.35;
const THUMB_TH2 = -0.5;
const THUMB_W1 = 2.4;
const THUMB_W2 = -1.8;
/** ctrl[0] = 2 -> torque +1 (the elbow is driving the swing on) */
const THUMB_ACTION = 2;

/** pose = [th1, th2, w1, w2, 0…] — the record-v3 Acrobot packing. */
function pose(th1: number, th2: number, w1: number, w2: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = th1;
  p[1] = th2;
  p[2] = w1;
  p[3] = w2;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'Acrobot-cont-v0',
  slug: 'acrobot',
  title: 'The Willow Swing',
  logline:
    'A dawn willow leans over a mirror pond; a two-link vine trapeze swings a paper lantern ' +
    'toward the string of lights strung exactly one link above the bough anchor.',
  accent: 'acrobot',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pose(THUMB_TH1, THUMB_TH2, THUMB_W1, THUMB_W2),
    derived: {
      th1: THUMB_TH1,
      th2: THUMB_TH2,
      w1: THUMB_W1,
      w2: THUMB_W2,
      tipHeight: acrobotTipHeight(THUMB_TH1, THUMB_TH2),
      action: THUMB_ACTION,
      torque: (THUMB_ACTION % 3) - 1,
    },
  },
  postConfig: {
    // dawn is a bright, hazy palette and this scene's bloom feeders (lantern, garland,
    // sun disc, water glint) are all small — a middling strength keeps the lantern the
    // brightest thing in frame without hazing the willow.
    bloomStrength: 0.46,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.dawn.fog,
  },
  // MEASURED at hero on Apple Silicon, read off the harness HUD during a celebration
  // (scene draws, shadow pass included, composer passes excluded): 31 draws / 113.0k
  // tris, 46 calls counting the composer — plus one draw for the lily pads, added after
  // that read. Declared with headroom for the ground-truth overlay, and well inside
  // STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 64, triangles: 170000 },
};

export default manifest;
