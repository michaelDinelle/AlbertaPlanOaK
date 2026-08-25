import * as THREE from 'three';
import { buildGrounds } from '@kit/grounds';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';
import { IDP_RAIL_HALF } from '@sim/envs/inverteddoublependulum';
import { buildRail, RAIL_Y } from '../inverted-pendulum/world';

export { RAIL_Y };

export type WorldHandle = WorldCore;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const g = buildGrounds(ctx, scene, { paletteName: 'afternoon', name: 'doubleRailYard', shadowRadius: 4.2 });
  const rail = buildRail(g.palette, IDP_RAIL_HALF, RAIL_Y);
  const railMesh = new THREE.Mesh(rail.geometry, rail.material);
  railMesh.name = 'rail';
  railMesh.castShadow = true;
  railMesh.receiveShadow = true;
  g.w.add(railMesh);
  g.w.own({
    dispose: () => {
      rail.geometry.dispose();
      rail.material.dispose();
    },
  });
  return g.w.seal<WorldHandle>({});
}
