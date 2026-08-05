
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { DELAYEDCHAIN_END, DELAYEDCHAIN_L } from '@sim/envs/delayedchain';

const THUMB_POS = 27;
const THUMB_LAST_A = 1;

export function delayedChainPose(pos: number, lastA: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = pos;
  p[1] = DELAYEDCHAIN_END;
  p[2] = lastA;
  return p;
}

export const manifest: SceneManifest = {
  envId: 'DelayedChain-cont-v0',
  slug: 'delayed-chain',
  title: 'The Long Night Road',
  logline:
    'A thirty-two-cell moorland path under a night sky, walked by lantern light that is not ' +
    'lit yet: the aurora hangs over the far end alone, and when the wisp finally reaches it ' +
    'every lantern on the road ignites in a wave running back from the end to the start.',
  accent: 'delayed-chain',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: delayedChainPose(THUMB_POS, THUMB_LAST_A),
    derived: { pos: THUMB_POS, lastA: THUMB_LAST_A, len: DELAYEDCHAIN_L },
  },
  postConfig: {
    bloomStrength: 0.62,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.night.fog,
  },
  budget: { drawCalls: 70, triangles: 190000 },
};

export default manifest;
