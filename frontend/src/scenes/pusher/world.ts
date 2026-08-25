import * as THREE from 'three';
import { buildGrounds } from '@kit/grounds';
import { oakMaterial } from '@kit/materials';
import { blendTokens } from '@kit/mix';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';

export const PUSH_SCALE = 2;
export const TABLE_Y = 0.6;
export const TABLE_W = 4.6;
export const TABLE_D = 3.6;

export type WorldHandle = WorldCore;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const g = buildGrounds(ctx, scene, { paletteName: 'afternoon', name: 'pushBench', extent: 44, shadowRadius: 3.4 });
  const palette = g.palette;

  const topGeometry = new THREE.BoxGeometry(TABLE_W, 0.12, TABLE_D);
  topGeometry.translate(0, TABLE_Y - 0.06, 0);
  const legGeometry = new THREE.CylinderGeometry(0.07, 0.09, TABLE_Y - 0.12, 8);
  const benchMaterial = oakMaterial({
    color: blendTokens(palette.groundB, palette.hemiSky, 0.22),
    flatShading: true,
    roughness: 0.72,
    rimColor: palette.hemiSky,
    rimStrength: 0.32,
    toonSteps: 3,
    toonSoftness: 0.34,
    toonFloor: 0.08,
    name: 'pushBenchMaterial',
  });
  const top = new THREE.Mesh(topGeometry, benchMaterial);
  top.name = 'pushBenchTop';
  top.castShadow = true;
  top.receiveShadow = true;
  g.w.add(top);
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const leg = new THREE.Mesh(legGeometry, benchMaterial);
      leg.name = 'pushBenchLeg';
      leg.castShadow = true;
      leg.position.set(sx * (TABLE_W / 2 - 0.2), (TABLE_Y - 0.12) / 2, sz * (TABLE_D / 2 - 0.2));
      g.w.add(leg);
    }
  }
  g.w.own({
    dispose: () => {
      topGeometry.dispose();
      legGeometry.dispose();
      benchMaterial.dispose();
    },
  });

  return g.w.seal<WorldHandle>({});
}
