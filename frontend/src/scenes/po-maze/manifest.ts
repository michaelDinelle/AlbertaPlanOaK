// The Fogbound Forest — scene manifest (POMaze-cont-v0).
//
// The thumbFrame stands the wisp at (5, 4), one cell south of the first wall band, with
// the goal in the far south-east corner. The camera looks EAST down the corridor (see
// world.ts's CAMERA_AZIMUTH), so that band materializes as a rank of spruce along the
// left of the card while the corridor and its fog open away to the right — the whole idea
// of the scene in one still. The goal at (9, 9) is four cells away and utterly unseen;
// only the compass moth on the bubble's rim says which way it lies.
//
// It is deliberately NOT a nook like (1, 4): a cell with a wall to its WEST puts that
// rank between this camera and the wisp and buries the subject in its own forest.
//
// AND THE THUMB CAMERA IS NAILED TO THIS CELL. STYLE.md's thumb contract says "fixed
// camera", and both grid exemplars hold still, so at thumb the scene reads this pose in
// `init`, points the eye at it and never calls `setTarget` again — on the wall this card
// sits as still as the other fifteen, and the framing composed here is the framing that
// ships. A live stream that walks its wisp into a far corner walks it out of the card;
// that is the accepted cost of a held shot in the densest fog in the preserve (hero and
// medium, which are watched rather than glanced at, still ride the agent).

import type { SceneManifest } from '@kit/types';
import { forestSkin } from './world';
import { TOKENS } from '@kit/tokens';

const THUMB_X = 5;
const THUMB_Y = 4;
const THUMB_GX = 9;
const THUMB_GY = 9;

/** pose = [x, y, gx, gy, 0…] — the record-v3 POMaze packing. */
function pose(x: number, y: number, gx: number, gy: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  return p;
}

/** the grade tint is the scene's own haze — the fog IS the look here */
const HAZE = forestSkin(TOKENS.palettes.dawn, TOKENS).haze.getHex();

export const manifest: SceneManifest = {
  envId: 'POMaze-cont-v0',
  slug: 'po-maze',
  title: 'The Fogbound Forest',
  logline:
    'A dawn spruce forest drowned in fog, where the world is only ever the eight cells the ' +
    'agent can actually see — ranks of black spruce condense out of the haze as it walks and ' +
    'dissolve behind it, and a single glowing moth on the rim of the light points the way home.',
  accent: 'po-maze',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pose(THUMB_X, THUMB_Y, THUMB_GX, THUMB_GY),
    derived: { x: THUMB_X, y: THUMB_Y, gx: THUMB_GX, gy: THUMB_GY },
  },
  postConfig: {
    // Restrained bloom on purpose: in this much fog the whole frame sits near the middle
    // of the range, and a strong bloom turns haze into glare. The only HDR emitters are
    // the wisp, the moth and the moonflower — they should be the only things that glow.
    bloomStrength: 0.38,
    bloomThreshold: 0.85,
    fogTint: HAZE,
  },
  // Measured off the harness HUD (`renderer.info`, scene channel, shadow pass included)
  // at 1600×900 on Apple Silicon, driving the page over CDP in real time — the headless
  // `--virtual-time-budget` path freezes the transport clock, so its HUD reads zero:
  //   hero    30 draws / 242.0k tris · 25/241.8k mid-goal-breath (the eye is on the
  //           clearing, so the wisp and its aura cull out) · 34/242.0k with the
  //           ground-truth overlay on · 128–144 fps
  //   medium  29 draws /  84.7k tris · 219 fps
  //   thumb   19 draws /  22.1k tris · 241 fps   (STYLE.md thumb cap: 40 / 50k)
  // Declared with headroom for the celebration pool, inside STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 60, triangles: 300000 },
};

export default manifest;
