
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'DeepSea-cont-v0',
  slug: 'deep-sea',
  title: 'Sunlit Descent',
  accent: 'deep-sea',
  postConfig: {
    bloomStrength: 0.52,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.underwater.fog,
  },
};

export default manifest;
