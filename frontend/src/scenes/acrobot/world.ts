import * as THREE from 'three';
import { alpineGroundGeometry, alpineGroundMaterial, alpineCliffMaterial, spruce, scatter } from '@kit/alpine';
import { mountainGeometry, terrainNoise as noise } from '@kit/mountains';
import { rngAt } from '@kit/rng';
import { makeWater } from '@kit/water';
import { simulateWater } from '@kit/fluid';
import { buildAurora } from '@kit/aurora';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';
import { AB_L1, AB_L2 } from '@sim/envs/acrobot';
import type { EnvGeometry } from '@sim/types';
import { fieldstones, makeLandscape, prairieGrass } from '../half-cheetah/world';
import type { ViewCoverage } from '../jelly-bean-world/view';

export const WATER_Y = -(AB_L1 + AB_L2) - 0.42;
const smooth = THREE.MathUtils.smoothstep;

export function pivotAnchor(geometry: EnvGeometry): readonly [number, number, number] {
  const anchor = geometry.classic?.anchors?.pivot;
  if (!anchor) throw new Error('Acrobot pivot anchor required');
  return anchor;
}

export function lakeHeight(x: number, z: number): number {
  const radius = Math.hypot((x + 8 + Math.sin(z * 0.018) * 7) / 92, (z + 105) / 129);
  const shore = (radius - 1) * 92 + (noise(x * 0.055, z * 0.055) - 0.5) * 6;
  const bank = Math.min(7, Math.max(-14, shore * 0.16));
  const hills = smooth(shore, 6, 120) * (3 + 18 * noise(x * 0.007 + 8, z * 0.009) ** 2);
  const headland = Math.exp(-Math.pow((z - 8) / 65, 2)) * (7.2 * Math.exp(-Math.pow((x + 11) / 6, 2)) + 6.2 * Math.exp(-Math.pow((x - 17) / 9, 2)));
  const foothills = smooth(-z, 220, 650) * (12 + 34 * noise(x * 0.008 + 30, z * 0.006) ** 2);
  return WATER_Y + bank + hills + headland + foothills + smooth(shore, -3, 4) * (noise(x * 0.3, z * 0.3) - 0.5) * 0.32;
}

export interface WorldHandle extends WorldCore {
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export async function buildWorld(ctx: SceneContext, scene: THREE.Scene): Promise<WorldHandle> {
  const mountains = await mountainGeometry('moraine-lake');
  const palette = ctx.tokens.palettes.night;
  const land = makeLandscape(scene, 'alpineLake', palette, 'night', {
    sky: { sunDir: [-0.6, 0.48, 0.64], sunSize: 0.016, sunHdr: 2, sunGlow: 0.045, topSpread: 0.25, stars: { intensity: 2.8, density: 180, threshold: 0.974, size: 0.075, color: 0xdbeaff } },
    light: { pointCount: 0, shadowMapSize: 1024 }, fogDensity: 0.00018,
  });
  const { world, rig, sky, fog } = land;
  sky.setColors(0x101c48, 0x718bae, 0x455977);
  world.time(world.own(buildAurora(sky, 0x71ebd2, 0.8))).setGain(1.15);
  fog.color.set(0x718bae);
  rig.sun.color.set(0xc3dcff);
  rig.sun.intensity = 1.7;
  rig.hemi.color.set(0xa8c9e9);
  rig.hemi.groundColor.set(0x485b61);
  rig.hemi.intensity = 1.8;
  rig.setShadowFocus(new THREE.Vector3(0, -1, 0), 7);
  const ground = alpineGroundMaterial(world, WATER_Y);
  const terrain = world.add(new THREE.Mesh(world.own(alpineGroundGeometry(lakeHeight, WATER_Y)), ground));
  terrain.name = 'alpineLakebed';
  terrain.receiveShadow = true;
  const cliff = alpineCliffMaterial(world);
  mountains.rotateY(Math.PI - 0.35).translate(490, 10, -2150);
  const peaks = world.add(new THREE.Mesh(world.own(mountains), cliff));
  peaks.name = 'canadianRockies';
  const water = world.time(world.own(makeWater({ size: 1024, segments: 256, palette, level: WATER_Y, bedHeight: lakeHeight,
    sky: 0x718bae, amplitude: 0.045, waveScale: 1.3, speed: 0.85, depthFalloff: 3.5, caustics: 0.025, sunDir: sky.sunDirection,
  })));
  water.setColors(0x479baf, 0x15466f);
  water.uniforms.uSpecular.value = 0.25;
  world.add(water.mesh);
  world.time(world.own(simulateWater(ctx.renderer, water, lakeHeight, new THREE.Vector4(-128, -256, 256, 288))));
  const rng = rngAt(4, 17, ctx.seed), dummy = new THREE.Object3D();
  const trees: THREE.Matrix4[] = [], stones: THREE.Matrix4[] = [], grass: THREE.Matrix4[] = [];
  for (let i = 0; i < 1700; i++) {
    const angle = rng() * Math.PI * 2, radius = 1.02 + rng() ** 1.7 * 0.9;
    const near = i < 160;
    const x = near ? (i % 2 ? -12 : 19) + (rng() - 0.5) * 12 : i > 1000 ? (rng() - 0.5) * 900 : Math.cos(angle) * 100 * radius - 8;
    const z = near ? -25 + rng() * 52 : i > 1000 ? -280 - rng() * 450 : Math.sin(angle) * 137 * radius - 105;
    const height = lakeHeight(x, z);
    if (height < WATER_Y + 0.85 || height > 85 || Math.hypot(x, z) < 9) continue;
    if (trees.length >= 800 || (!near && noise(x * 0.025, z * 0.025) < 0.25)) continue;
    dummy.position.set(x, height - 0.2, z);
    dummy.rotation.set(0, rng() * Math.PI * 2, (rng() - 0.5) * 0.04);
    const scale = (near ? 0.8 : 1.2) + rng() * 1.2;
    dummy.scale.set(scale * 1.15, scale, scale * 1.15);
    dummy.updateMatrix();
    trees.push(dummy.matrix.clone());
  }
  for (const part of spruce(world)) scatter(world, part.geometry, part.material, trees, 'spruceForest');
  for (let i = 0; i < 1500; i++) {
    const angle = rng() * Math.PI * 2, radius = rng() ** 1.9 * 180;
    const x = Math.cos(angle) * radius, z = Math.sin(angle) * radius, height = lakeHeight(x, z);
    if (height < WATER_Y - 2.8 || height > WATER_Y + 5 || stones.length >= 260) continue;
    const scale = 0.25 + rng() ** 2 * 4;
    dummy.position.set(x, height - 0.09 * scale, z);
    dummy.rotation.set((rng() - 0.5) * 0.3, rng() * Math.PI * 2, (rng() - 0.5) * 0.3);
    dummy.scale.set(scale, scale * (0.7 + rng() * 0.5), scale);
    dummy.updateMatrix();
    stones.push(dummy.matrix.clone());
  }
  const rock = fieldstones(world);
  scatter(world, world.own(rock.geometry), world.own(rock.material), stones, 'shoreBoulders');
  for (let i = 0; i < 5200; i++) {
    const x = (rng() - 0.5) * 150, z = (rng() - 0.5) * 150, height = lakeHeight(x, z);
    if (height < WATER_Y + 0.3 || height > WATER_Y + 6) continue;
    dummy.position.set(x, height, z);
    dummy.rotation.set(0, rng() * Math.PI * 2, 0);
    dummy.scale.setScalar(0.7 + rng() * 1.1);
    dummy.updateMatrix();
    grass.push(dummy.matrix.clone());
  }
  const tuft = prairieGrass();
  tuft.material.color.set(0x87a174);
  scatter(world, world.own(tuft.geometry), world.own(tuft.material), grass, 'alpineSedges');
  return world.seal<WorldHandle>({
    setView(coverage, camera) {
      camera.far = coverage.far = 5000;
      camera.updateProjectionMatrix();
      camera.position.y = Math.max(camera.position.y, lakeHeight(camera.position.x, camera.position.z) + 0.6, WATER_Y + 0.2);
      camera.updateMatrixWorld();
      land.setView(coverage, camera);
    },
  });
}
