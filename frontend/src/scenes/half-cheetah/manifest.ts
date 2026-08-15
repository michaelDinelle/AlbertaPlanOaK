
import * as THREE from 'three';
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'HalfCheetah-v4',
  slug: 'half-cheetah',
  title: 'Gold Hour Savanna',
  accent: 'half-cheetah',
  postConfig: {
    bloomStrength: 0.42,
    bloomThreshold: 0.85,
    fogTint: new THREE.Color(TOKENS.palettes.dusk.fog)
      .lerp(new THREE.Color(TOKENS.palettes.dusk.skyTop), 0.55)
      .getHex(),
  },
};

export default manifest;
