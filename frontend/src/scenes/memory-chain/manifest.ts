// The Firefly Cave — scene manifest (MemoryChain-cont-v0).
//
// The thumbFrame is the one pose that says what this env IS: the wisp standing on cell 0
// with the cue crystal blazing magenta beside it, and the two exit mouths thirty-one
// stones away at the dark end of the passage. Everything the task consists of — see the
// cue here, carry it there, choose — is legible in that single still, and it is the ONLY
// pose in which any cue colour is allowed to exist at all.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { MEMORYCHAIN_END, MEMORYCHAIN_L, type MemoryChainState } from '@sim/envs/memorychain';

const THUMB_POS = 0;
const THUMB_CUE = 1; // magenta: the strongest contrast against the cave's blue-grey rock

/** pose = [pos, 31, cue, 0…] — the record-v3 MemoryChain packing. */
export function memoryChainPose(pos: number, cue: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = pos;
  p[1] = MEMORYCHAIN_END;
  p[2] = cue;
  return p;
}

export function memoryChainDerived(pos: number, cue: number, lastA = 0): MemoryChainState {
  return { pos, cue, lastA, len: MEMORYCHAIN_L };
}

export const manifest: SceneManifest = {
  envId: 'MemoryChain-cont-v0',
  slug: 'memory-chain',
  title: 'The Firefly Cave',
  logline:
    'A thirty-two-stone passage over still black water, lined with fireflies, where a ' +
    'crystal burns cyan or magenta only while the wisp stands beside it — and thirty-one ' +
    'stones later two dim mouths in the dark ask which colour it was.',
  accent: 'memory-chain',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: memoryChainPose(THUMB_POS, THUMB_CUE),
    derived: memoryChainDerived(THUMB_POS, THUMB_CUE),
  },
  postConfig: {
    // A dark palette needs a generous bloom: the crystal, the two mouths and the fireflies
    // are the only emitters in the frame, and they are what the eye is meant to follow.
    bloomStrength: 0.62,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.cave.fog,
  },
  // Measured by `measureBudget` in test/scenes/memory-chain.test.ts — a scene-graph twin
  // of `renderer.info` (one draw per visible drawable, instances multiplied in), because a
  // headless `--screenshot` run ticks rAF too few times for the harness HUD's rolling
  // stats to fill:
  //   hero    23 draws / 33.8k tris idle · 24 through a celebration · 26 with the
  //           ground-truth overlay on
  //   medium  23 draws / 11.4k tris
  //   thumb   16 draws /  3.0k tris     (STYLE.md thumb cap: 40 / 50k)
  // The shadow pass re-submits the ~5 casters (dripstone, crystal, the two arches), so the
  // renderer's own count lands near 29 / 40k. Declared with a little headroom on that, and
  // an order of magnitude inside STYLE.md's 300 / 500k hero cap: a cave is cheap, and the
  // budget it does spend goes on the swept cavern and the water.
  budget: { drawCalls: 45, triangles: 75000 },
};

export default manifest;
