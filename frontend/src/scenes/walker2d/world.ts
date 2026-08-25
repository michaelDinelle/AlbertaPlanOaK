import * as THREE from 'three';
import { buildGrounds } from '@kit/grounds';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';

export interface WorldHandle extends WorldCore {
  setOrigin(x: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const g = buildGrounds(ctx, scene, {
    paletteName: 'dusk',
    name: 'strideFlats',
    extent: 72,
    relief: 0.07,
    shadowRadius: 3.8,
  });
  return g.w.seal<WorldHandle>({
    setOrigin(x) {
      g.setOrigin(x, 0);
    },
  });
}
