
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { CP_TH_LIMIT, CP_X_LIMIT } from '@sim/envs/cartpole';

const THUMB_X = -0.34;
const THUMB_XDOT = 0.46;
const THUMB_TH = 0.098;
const THUMB_THDOT = -0.62;

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
    bloomStrength: 0.34,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.noon.fog,
  },
  budget: { drawCalls: 60, triangles: 230000 },
};

export default manifest;
