
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'Pendulum-cont-v0',
  slug: 'pendulum',
  title: 'The Lantern Bell',
  accent: 'pendulum',
  postConfig: {
    bloomStrength: 0.54,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.night.fog,
  },
};

export default manifest;
