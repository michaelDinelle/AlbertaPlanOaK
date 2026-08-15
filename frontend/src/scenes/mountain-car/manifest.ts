
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'MountainCar-cont-v0',
  slug: 'mountain-car',
  title: 'Alpine Valley',
  accent: 'mountain-car',
  postConfig: {
    bloomStrength: 0.44,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.dusk.fog,
  },
};

export default manifest;
