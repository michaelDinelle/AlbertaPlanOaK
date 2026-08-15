
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { acrobotTipHeight } from '@sim/envs/acrobot';

const THUMB_TH1 = 2.35;
const THUMB_TH2 = -0.5;
const THUMB_W1 = 2.4;
const THUMB_W2 = -1.8;
const THUMB_ACTION = 2;

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
    bloomStrength: 0.46,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.dawn.fog,
  },
};

export default manifest;
