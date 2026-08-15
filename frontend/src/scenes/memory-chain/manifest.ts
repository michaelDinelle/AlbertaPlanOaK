
import { TOKENS } from '@kit/tokens';
import type { SceneManifest } from '@kit/types';

export const manifest: SceneManifest = {
  envId: 'MemoryChain-cont-v0',
  slug: 'memory-chain',
  title: 'The Firefly Cave',
  accent: 'memory-chain',
  postConfig: {
    bloomStrength: 0.62,
    bloomThreshold: 0.85,
    fogTint: TOKENS.palettes.cave.fog,
  },
};

export default manifest;
