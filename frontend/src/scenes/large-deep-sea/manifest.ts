
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'LargeDeepSea-cont-v0',
  slug: 'large-deep-sea',
  title: 'The Abyss',
  accent: 'large-deep-sea',
  postConfig: {
    bloomStrength: 0.48,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.abyss.fog,
  },
};

export default manifest;
