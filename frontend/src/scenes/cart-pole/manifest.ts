
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'CartPole-cont-v0',
  slug: 'cart-pole',
  title: 'The Boardwalk',
  accent: 'cart-pole',
  postConfig: {
    bloomStrength: 0.34,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.noon.fog,
  },
};

export default manifest;
