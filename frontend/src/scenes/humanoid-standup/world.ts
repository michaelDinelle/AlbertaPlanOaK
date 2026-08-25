import type * as THREE from 'three';
import type { SceneContext } from '@kit/types';
import { buildHumanoidWorld, type WorldHandle } from '../humanoid/world';

export type { WorldHandle };

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  return buildHumanoidWorld(ctx, scene, 'dawn', 'risingField');
}
