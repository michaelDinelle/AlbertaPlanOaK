import * as THREE from 'three';
import { buildGrounds } from '@kit/grounds';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';

export interface WorldHandle extends WorldCore {
  setOrigin(x: number, z: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const g = buildGrounds(ctx, scene, {
    paletteName: 'dawn',
    name: 'antBasin',
    extent: 80,
    relief: 0.1,
    shadowRadius: 4.4,
  });
  return g.w.seal<WorldHandle>({
    setOrigin(x, z) {
      g.setOrigin(x, z);
    },
  });
}
