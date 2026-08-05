
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
  budget: { drawCalls: 90, triangles: 220000 },
};

export default manifest;
