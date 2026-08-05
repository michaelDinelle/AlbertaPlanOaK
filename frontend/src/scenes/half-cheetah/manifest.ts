
import * as THREE from 'three';
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';
import { ROOT_Z0 } from '@sim/fk/cheetah';

const THUMB_POSE: readonly number[] = [
  201.27566528320312, -0.08060205727815628, -0.08686039596796036, 0.20897798240184784,
  0.10743971914052963, -0.14246860146522522, 0.08087541162967682, -0.22010691463947296,
  -0.12991316616535187,
];
const THUMB_VEL = 0.966;

export const manifest: SceneManifest = {
  envId: 'HalfCheetah-v4',
  slug: 'half-cheetah',
  title: 'Gold Hour Savanna',
  logline:
    'A dusk grassland that never ends: the one four-limbed hero of the preserve, skinned bone ' +
    'for bone on its own forward kinematics, running past stone cairns that count the metres.',
  accent: 'half-cheetah',
  tiers: ['hero', 'medium', 'thumb'],
  thumbFrame: {
    pose: THUMB_POSE,
    derived: {
      qpos: Float32Array.from(THUMB_POSE),
      rootX: THUMB_POSE[0] ?? 0,
      rootZ: ROOT_Z0 + (THUMB_POSE[1] ?? 0),
      pitch: THUMB_POSE[2] ?? 0,
      velX: THUMB_VEL,
    },
  },
  postConfig: {
    bloomStrength: 0.42,
    bloomThreshold: 0.85,
    fogTint: new THREE.Color(TOKENS.palettes.dusk.fog)
      .lerp(new THREE.Color(TOKENS.palettes.dusk.skyTop), 0.55)
      .getHex(),
  },
  budget: { drawCalls: 72, triangles: 220000 },
};

export default manifest;
