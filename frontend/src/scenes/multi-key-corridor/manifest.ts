// Temple of Three Gates — scene manifest (MultiKeyCorridor-cont-v0).
//
// The thumbFrame stands the wisp ON the river gate (8, 3) — the middle of the temple,
// dead centre of the card — with the seed and pearl keys already orbiting it, the
// bramble unwound behind, the ford's stepping stones up under its feet and the mist
// gate still drawn ahead. One still that says: three gates, three keys, one relic at
// the end of the hall.

import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { TEMPLE_FORD } from './fixtures';

const THUMB_POSE: number[] = [
  TEMPLE_FORD.x,
  TEMPLE_FORD.y,
  TEMPLE_FORD.gx,
  TEMPLE_FORD.gy,
  TEMPLE_FORD.flags & 7,
  (TEMPLE_FORD.flags >> 3) & 7,
  0,
  0,
  0,
];

export const manifest: SceneManifest = {
  envId: 'MultiKeyCorridor-cont-v0',
  slug: 'multi-key-corridor',
  title: 'Temple of Three Gates',
  logline:
    'A morning temple garden of four chambers that warm from dewy blue-green to gold, ' +
    'joined by three gates the wisp must earn — a bramble to unwind, a ford to stone, a ' +
    'mist to part — with the relic burning on whichever sanctum cell the goal has ' +
    'respawned to.',
  accent: 'multi-key-corridor',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: THUMB_POSE,
    derived: TEMPLE_FORD,
  },
  postConfig: {
    bloomStrength: 0.42,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.morning.fog,
  },
  // Measured on Apple Silicon from the harness HUD at 1600×900 DPR 2, shadow pass
  // included (and cross-checked by counting renderables + shadow casters on the built
  // scene graph, which agrees to within the celebration pool's idle ripple meshes):
  //   hero    71 draws /  86.0k tris · 125 fps
  //   medium  67 draws /  29.5k tris
  //   thumb   39 draws /   8.5k tris   (STYLE.md thumb cap: 40 / 50k)
  // The thumb number is the binding one: three separate carried-key meshes and three
  // key-token halos were the difference between 44 and 39, which is why the carry rack
  // is instanced and the halos are dropped at that tier.
  // Declared with headroom, well inside STYLE.md's 300 / 500k hero cap.
  budget: { drawCalls: 90, triangles: 220000 },
};

export default manifest;
