import * as THREE from 'three';
import { buildGrounds } from '@kit/grounds';
import { oakMaterial, type OakMaterial } from '@kit/materials';
import { blendTokens } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';
import { IP_RAIL_HALF } from '@sim/envs/invertedpendulum';

export const RAIL_Y = 0.5;
export const RAIL_OVERHANG = 0.35;

export interface WorldHandle extends WorldCore {
  railMaterial: OakMaterial;
}

export function buildRail(palette: {
  hemiGround: number;
  hemiSky: number;
  skyTop: number;
  groundB: number;
}, railHalf: number, railY: number): { geometry: THREE.BufferGeometry; material: OakMaterial } {
  const steel = blendTokens(palette.hemiGround, palette.skyTop, 0.4);
  const steelLight = blendTokens(steel, palette.hemiSky, 0.35);
  const parts: ColoredPart[] = [];
  const span = (railHalf + RAIL_OVERHANG) * 2;
  const beam = new THREE.BoxGeometry(span, 0.06, 0.14);
  beam.translate(0, railY - 0.05, 0);
  parts.push({ geometry: beam, color: steel });
  const rail = new THREE.CylinderGeometry(0.024, 0.024, span, 8);
  rail.rotateZ(Math.PI / 2);
  rail.translate(0, railY, 0);
  parts.push({ geometry: rail, color: steelLight });
  for (const sx of [-1, 1]) {
    const post = new THREE.CylinderGeometry(0.045, 0.06, railY, 8);
    post.translate(sx * (railHalf + RAIL_OVERHANG), railY * 0.5 - 0.05, 0);
    parts.push({ geometry: post, color: steel });
    const foot = new THREE.CylinderGeometry(0.11, 0.14, 0.07, 8);
    foot.translate(sx * (railHalf + RAIL_OVERHANG), 0.03, 0);
    parts.push({ geometry: foot, color: steelLight });
  }
  const geometry = mergeColoredParts(parts, 'rail');
  const material = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.6,
    metalness: 0.15,
    rimColor: palette.hemiSky,
    rimStrength: 0.45,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.1,
    name: 'railMaterial',
  });
  return { geometry, material };
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const g = buildGrounds(ctx, scene, { paletteName: 'noon', name: 'railYard', shadowRadius: 3.6 });
  const rail = buildRail(g.palette, IP_RAIL_HALF, RAIL_Y);
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
  return g.w.seal<WorldHandle>({ railMaterial: rail.material });
}
