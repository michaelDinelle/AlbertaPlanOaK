
import type { SceneManifest } from '@kit/types';
import { forestSkin } from './world';
import { TOKENS } from '@kit/tokens';

const THUMB_X = 5;
const THUMB_Y = 4;
const THUMB_GX = 9;
const THUMB_GY = 9;

function pose(x: number, y: number, gx: number, gy: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  return p;
}

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
    bloomStrength: 0.38,
    bloomThreshold: 0.85,
    fogTint: HAZE,
  },
};

export default manifest;
