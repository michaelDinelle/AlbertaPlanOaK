
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'Chain-cont-v0',
  slug: 'chain',
  title: 'Sixteen Stones',
  accent: 'chain',
  postConfig: {
    bloomStrength: 0.34,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.afternoon.fog,
  },
};

export default manifest;
