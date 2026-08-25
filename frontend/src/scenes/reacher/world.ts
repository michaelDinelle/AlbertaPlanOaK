import * as THREE from 'three';
import { buildGrounds } from '@kit/grounds';
import { emissiveMaterial, oakMaterial, setEmissiveIntensity, type EmissiveMaterial } from '@kit/materials';
import { blendTokens } from '@kit/mix';
import { TOKENS } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';
import { RE_ARENA_HALF } from '@sim/envs/reacher';

export const REACH_SCALE = 6;
export const TABLE_Y = 0.9;
export const ARENA_R = RE_ARENA_HALF * REACH_SCALE;

export interface WorldHandle extends WorldCore {
  setRingGlow(k: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const g = buildGrounds(ctx, scene, { paletteName: 'night', name: 'reachTable', extent: 40, shadowRadius: 3.4 });
  const palette = g.palette;
  const accent = TOKENS.accents['reacher'] ?? TOKENS.celebrationGold;

  const tableGeometry = new THREE.CylinderGeometry(ARENA_R + 0.45, ARENA_R + 0.7, TABLE_Y, 40);
  tableGeometry.translate(0, TABLE_Y * 0.5, 0);
  const tableMaterial = oakMaterial({
    color: blendTokens(palette.groundB, palette.hemiSky, 0.2),
    flatShading: true,
    roughness: 0.75,
    rimColor: palette.hemiSky,
    rimStrength: 0.35,
    toonSteps: 3,
    toonSoftness: 0.34,
    toonFloor: 0.08,
    name: 'reachTableMaterial',
  });
  const table = new THREE.Mesh(tableGeometry, tableMaterial);
  table.name = 'reachTable';
  table.castShadow = true;
  table.receiveShadow = true;
  g.w.add(table);

  const ringGeometry = new THREE.TorusGeometry(ARENA_R, 0.03, 6, 72);
  ringGeometry.rotateX(-Math.PI / 2);
  ringGeometry.translate(0, TABLE_Y + 0.015, 0);
  const ringMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(accent, palette.hemiSky, 0.3), 1.6, {
    transparent: true,
    opacity: 0.85,
    name: 'reachArenaRing',
  });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.name = 'reachArenaRing';
  g.w.add(ring);

  g.w.own({
    dispose: () => {
      tableGeometry.dispose();
      tableMaterial.dispose();
      ringGeometry.dispose();
      ringMaterial.dispose();
    },
  });

  return g.w.seal<WorldHandle>({
    setRingGlow(k) {
      setEmissiveIntensity(ringMaterial, 1.1 + 1.6 * k);
    },
  });
}
