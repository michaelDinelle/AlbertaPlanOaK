import * as THREE from 'three';
import { buildGrounds } from '@kit/grounds';
import type { PaletteName } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';

export interface WorldHandle extends WorldCore {
  setOrigin(x: number, z: number): void;
}

export function buildHumanoidWorld(
  ctx: SceneContext,
  scene: THREE.Scene,
  paletteName: PaletteName,
  name: string,
): WorldHandle {
  const g = buildGrounds(ctx, scene, {
    paletteName,
    name,
    extent: 80,
    relief: 0.09,
    shadowRadius: 4.2,
  });
  return g.w.seal<WorldHandle>({
    setOrigin(x, z) {
      g.setOrigin(x, z);
    },
  });
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  return buildHumanoidWorld(ctx, scene, 'noon', 'strideField');
}
