
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { CHAIN_END, CHAIN_L, type ChainState } from '@sim/envs/chain';

const THUMB_POS = 12;

export function chainPose(pos: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = pos;
  p[1] = CHAIN_END;
  return p;
}

export function chainDerived(pos: number): ChainState {
  return { pos, len: CHAIN_L };
}

export const manifest: SceneManifest = {
  envId: 'Chain-cont-v0',
  slug: 'chain',
  title: 'Sixteen Stones',
  logline:
    'An afternoon ford where the sixteen-cell corridor is sixteen stepping stones across a ' +
    'creek: the wisp hops them one by one, the bare blossom tree on the far stone breaks into ' +
    'flower on every +1, and its petals drift back down the water to the start.',
  accent: 'chain',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: chainPose(THUMB_POS),
    derived: chainDerived(THUMB_POS),
  },
  postConfig: {
    bloomStrength: 0.34,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.afternoon.fog,
  },
  budget: { drawCalls: 60, triangles: 240000 },
};

export default manifest;
