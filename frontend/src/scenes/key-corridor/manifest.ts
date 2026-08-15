
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'KeyCorridor-cont-v0',
  slug: 'key-corridor',
  title: 'The Ivy Gate',
  accent: 'key-corridor',
  postConfig: {
    bloomStrength: 0.3,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.morning.fog,
  },
};

export default manifest;
