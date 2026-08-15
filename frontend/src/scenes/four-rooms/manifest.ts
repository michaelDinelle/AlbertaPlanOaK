
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'FourRooms-cont-v0',
  slug: 'four-rooms',
  title: 'Four Seasons Courtyard',
  accent: 'four-rooms',
  postConfig: {
    bloomStrength: 0.45,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.noon.fog,
  },
};

export default manifest;
