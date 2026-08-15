
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'HighDimMaze-cont-v0',
  slug: 'high-dim-maze',
  title: 'The Slot Canyons',
  accent: 'high-dim-maze',
  postConfig: {
    bloomStrength: 0.42,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.afternoon.fog,
  },
};

export default manifest;
