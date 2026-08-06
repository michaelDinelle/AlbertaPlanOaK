
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { LARGE_DEEPSEA_N } from '@sim/envs/largedeepsea';

const THUMB_R = 11;
const THUMB_C = 6;

export function abyssPose(r: number, c: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = r;
  p[1] = c;
  p[2] = LARGE_DEEPSEA_N - 1;
  p[3] = LARGE_DEEPSEA_N - 1;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'LargeDeepSea-cont-v0',
  slug: 'large-deep-sea',
  title: 'The Abyss',
  logline:
    'The dive past the reef: sixteen terraced shelves stepping down into a violet-black trench, ' +
    'lit only by anglerfish lures and drifting glow-jellies, with a sunken wreck burning through ' +
    'its ribs at the one corner the agent has never reached.',
  accent: 'large-deep-sea',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: abyssPose(THUMB_R, THUMB_C),
    derived: { r: THUMB_R, c: THUMB_C, lastA: 1 },
  },
  postConfig: {
    bloomStrength: 0.48,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.abyss.fog,
  },
  budget: { drawCalls: 55, triangles: 110000 },
};

export default manifest;
