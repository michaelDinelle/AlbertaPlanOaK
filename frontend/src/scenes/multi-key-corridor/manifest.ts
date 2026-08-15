
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'MultiKeyCorridor-cont-v0',
  slug: 'multi-key-corridor',
  title: 'Temple of Three Gates',
  accent: 'multi-key-corridor',
  postConfig: {
    bloomStrength: 0.42,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.morning.fog,
  },
};

export default manifest;
