
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'ProceduralMaze-cont-v0',
  slug: 'procedural-maze',
  title: 'The Overgrown Ruins',
  accent: 'procedural-maze',
  postConfig: {
    bloomStrength: 0.44,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.noon.fog,
  },
};

export default manifest;
