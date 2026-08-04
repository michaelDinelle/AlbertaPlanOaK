// The Boardwalk — scene manifest (CartPole-cont-v0).
//
// The thumbFrame is a real, physically-reachable save: the trolley a third of a metre
// left of centre and rolling right (xdot +0.46) while the sapling leans right at 5.6°
// (a bit over a quarter of the 12° failure band) and is already swinging back (thdot
// −0.62). It reads as balancing — the moment the whole env is about, not a parked cart.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { CP_TH_LIMIT, CP_X_LIMIT } from '@sim/envs/cartpole';

const THUMB_X = -0.34;
const THUMB_XDOT = 0.46;
const THUMB_TH = 0.098;
const THUMB_THDOT = -0.62;

/** pose = [x, xdot, th, thdot, 0…] — the record-v3 CartPole packing. */
function pose(x: number, xDot: number, th: number, thDot: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = xDot;
  p[2] = th;
  p[3] = thDot;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'CartPole-cont-v0',
  slug: 'cart-pole',
  title: 'The Boardwalk',
  logline:
    'A windy noon lake crossed by a plank boardwalk that runs exactly from x = −2.4 to x = +2.4: ' +
    'a planter trolley rides it carrying a sapling whose pale terminal bud — the pole tip itself — ' +
    'must stay above the surveyor’s cord strung at the 12° failure height, while the carved cone ' +
    'behind the deck states that same bound as the angle a viewer can actually see.',
  accent: 'cart-pole',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: pose(THUMB_X, THUMB_XDOT, THUMB_TH, THUMB_THDOT),
    derived: {
      x: THUMB_X,
      xDot: THUMB_XDOT,
      th: THUMB_TH,
      thDot: THUMB_THDOT,
      action: 1,
      force: 10,
      xMargin: Math.abs(THUMB_X) / CP_X_LIMIT,
      thMargin: Math.abs(THUMB_TH) / CP_TH_LIMIT,
    },
  },
  postConfig: {
    // Noon is the brightest palette in the preserve and the lake already carries a
    // caustic shimmer, so the bloom sits at the low end of the legal band: this scene
    // never fires a celebration pulse (CartPole's +1 is every step, not an event), so
    // whatever is set here is what the frame looks like forever.
    bloomStrength: 0.34,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.noon.fog,
  },
  // WHAT renderer.info ACTUALLY REPORTS. The shadow pass re-rasterizes every caster, so
  // a caster's triangles are counted TWICE — a scene-only count under-reports by exactly
  // the caster total, which is how this budget was previously stated 25% low (declared
  // 186.1k against a real 233.1k, i.e. 97% of the cap with 2.9% headroom). The numbers
  // below are `scene + shadow` from `cart-pole.test.ts`'s `measure()`, which was fixed to
  // double-count casters and now reproduces renderer.info exactly — cross-checked against
  // a real HUD reading of "45/70 draws · 215.6k tris" on an intermediate build whose
  // probe figure was 45 / 215,648.
  //
  //   hero    41 scene draws + 2 shadow casters = 43 · 188,628 tris  (45 with overlay)
  //   medium  38 + 2 = 40 ·  63,238 tris
  //   thumb   24 + 0 = 24 ·  16,846 tris   (STYLE.md thumb cap: 40 / 50k)
  //
  // 27,024 of those triangles used to be far-shore trees rendering into a shadow map
  // whose frustum they sit 14 units outside of; dropping their castShadow is bitwise
  // identical in the 3D frame (verified by pixel diff — only the HUD's own counters
  // moved) and took the hero total from 233,144 to 188,628. Declared at 230k / 60 draws:
  // 22% triangle and 28% draw headroom, well inside STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 60, triangles: 230000 },
};

export default manifest;
