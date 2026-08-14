
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { MKC_GOAL, MKC_KEYS, type MultiKeyCorridorState } from '@sim/envs/multikeycorridor';

const THUMB_POSE: number[] = [8, 3, MKC_GOAL.x, MKC_GOAL.y, 3, 3, 0, 0, 0];

const THUMB_DERIVED: MultiKeyCorridorState = {
  x: 8,
  y: 3,
  gx: MKC_GOAL.x,
  gy: MKC_GOAL.y,
  keys: MKC_KEYS.map((k) => ({ ...k })),
  hasKey: [true, true, true],
  doorOpen: [true, true, false],
  flags: 27,
  respawnIdx: 0,
  poseAgrees: true,
};

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
    derived: THUMB_DERIVED,
  },
  postConfig: {
    bloomStrength: 0.42,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.morning.fog,
  },
  budget: { drawCalls: 90, triangles: 220000 },
};

export default manifest;
