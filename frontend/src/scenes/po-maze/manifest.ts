
import type { SceneManifest } from '@kit/types';
import { forestSkin } from './world';
import { TOKENS } from '@kit/tokens';

const HAZE = forestSkin(TOKENS.palettes.dawn, TOKENS).haze.getHex();

export const manifest: SceneManifest = {
  envId: 'POMaze-cont-v0',
  slug: 'po-maze',
  title: 'The Fogbound Forest',
  accent: 'po-maze',
  postConfig: {
    bloomStrength: 0.38,
    bloomThreshold: 0.85,
    fogTint: HAZE,
  },
};

export default manifest;
