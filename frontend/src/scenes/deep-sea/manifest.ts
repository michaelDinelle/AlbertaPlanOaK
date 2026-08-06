
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { DEEPSEA_N } from '@sim/envs/deepsea';

const THUMB_R = 5;
const THUMB_C = 4;
const THUMB_A = 1;

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
    bloomStrength: 0.52,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.underwater.fog,
  },
  budget: { drawCalls: 60, triangles: 60000 },
};

export default manifest;
