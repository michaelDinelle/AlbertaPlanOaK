
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'Acrobot-cont-v0',
  slug: 'acrobot',
  title: 'The Willow Swing',
  accent: 'acrobot',
  postConfig: {
    bloomStrength: 0.46,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.dawn.fog,
  },
};

export default manifest;
