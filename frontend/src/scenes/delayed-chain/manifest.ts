
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'DelayedChain-cont-v0',
  slug: 'delayed-chain',
  title: 'The Long Night Road',
  accent: 'delayed-chain',
  postConfig: {
    bloomStrength: 0.62,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.night.fog,
  },
};

export default manifest;
