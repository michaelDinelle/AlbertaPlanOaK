// Sixteen Stones — scene manifest (Chain-cont-v0).
//
// The thumbFrame stands the wisp on stone 12 — four stones short of the paying end, deep
// enough into the crossing that the ford reads as a journey and close enough to the far
// bank that the blossom tree is in frame with it. It is a real recorded pose (seed 64,
// step 42), not a composed one.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { CHAIN_END, CHAIN_L, type ChainState } from '@sim/envs/chain';

const THUMB_POS = 12;

/** pose = [pos, 15, 0…] — the record-v3 Chain packing. */
export function chainPose(pos: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = pos;
  p[1] = CHAIN_END;
  return p;
}

/** the decoder's own state shape — never a scene invention */
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
    // A quiet scene with one bright event every couple of seconds: a heavier bloom turns
    // the creek's specular path into a white band between celebrations.
    bloomStrength: 0.34,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.afternoon.fog,
  },
  // Measured on Apple Silicon at 1600×900 DPR 2, shadow pass included:
  //   hero    110 fps · 36 draws / 184.7k tris  (harness HUD, mid-celebration, 51 total
  //           draws with the composer chain). The bill: creek bed 42.2k, water 32.8k,
  //           flowers 17.2k, bank trees 20.4k, creek rocks 20.8k, reeds + meadow 23.2k.
  //   medium   33 draws /  57.6k tris
  //   thumb    20 draws /  12.8k tris  (STYLE.md thumb cap: 40 / 50k)
  // The medium/thumb figures are a scene-graph accounting (visible meshes × instances ×
  // triangles, doubled for shadow casters) — the same accounting reproduces the hero HUD to
  // within two draws and 0.5k triangles. Declared with headroom for the celebration pool
  // and the ground-truth overlay, well inside STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 60, triangles: 240000 },
};

export default manifest;
