import * as THREE from 'three';
import { alpineGroundGeometry, alpineMaterial, spruce, scatter } from '@kit/alpine';
import { makeMountains, terrainNoise as noise } from '@kit/mountains';
import { rngAt } from '@kit/rng';
import { makeWater } from '@kit/water';
import { simulateWater } from '@kit/fluid';
import { makeAmbientParticles } from '@kit/particles';
import type { SceneContext } from '@kit/types';
import type { WorldCore } from '@kit/world';
import { fieldstones, makeLandscape, prairieGrass } from '@kit/landscape';
import type { ViewCoverage } from '@kit/view';

export const WATER_LEVEL = -1.85;
const smooth = THREE.MathUtils.smoothstep;

function riverChannel(z: number) {
  const center = Math.sin(z * 0.016) * 9 + Math.sin(z * 0.047) * 2;
  const width = 5.5 + smooth(Math.abs(z), 25, 190) * 6 + noise(z * 0.03, 5) * smooth(Math.abs(z), 0, 80) * 2;
  return { center, width };
}

export function riverHeight(x: number, z: number): number {
  const { center, width } = riverChannel(z);
  const shore = Math.abs(x - center) - width;
  const bank = smooth(shore, -1, 3.5);
  const hills = smooth(shore, 6, 45) * (5 + noise(x * 0.024 + 7, z * 0.018) ** 2 * 19);
  const foothills = smooth(Math.abs(z), 160, 850) * smooth(shore, 12, 85) * (15 + 54 * noise(x * 0.006, z * 0.008) ** 2);
  return WATER_LEVEL - 1.4 + bank * 3.5 + hills + foothills + bank * (noise(x * 0.22, z * 0.22) - 0.5) * 0.16;
}

export interface WorldHandle extends WorldCore {
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export async function buildWorld(ctx: SceneContext, scene: THREE.Scene): Promise<WorldHandle> {
  const palette = ctx.tokens.palettes.dusk;
  const land = makeLandscape(scene, 'albertaRiver', palette, 'dusk', {
    sky: { sunDir: [-0.45, 0.24, 0.86], sunSize: 0.022, sunHdr: 1.7, sunGlow: 0.18, topSpread: 0.3, stars: { intensity: 1.6, density: 160, threshold: 0.976, size: 0.07, color: 0xffecd7 } },
    light: { pointCount: 0, shadowMapSize: 1024 }, fogDensity: 0.00024,
  });
  const { world, rig, sky, fog } = land;
  sky.setColors(0x303a76, 0xffb780, 0xc49190);
  fog.color.set(0xd5a89c);
  rig.sun.color.set(0xffc991);
  rig.sun.intensity = 2.4;
  rig.hemi.color.set(0xb8c9eb);
  rig.hemi.groundColor.set(0x6d6958);
  rig.hemi.intensity = 1.8;
  rig.setShadowFocus(new THREE.Vector3(0, 0.3, 0), 13);
  const ground = world.add(new THREE.Mesh(world.own(alpineGroundGeometry(riverHeight, WATER_LEVEL)), alpineMaterial(world)));
  ground.name = 'riverGravelBanks';
  ground.receiveShadow = true;
  const peaks = await makeMountains(world, palette);
  peaks.rotation.y = -0.12;
  peaks.position.set(-180, -20, 0);
  peaks.name = 'bowValleyRockies';
  const water = world.time(world.own(makeWater({ size: [320, 4096], segments: 256, palette, level: WATER_LEVEL, bedHeight: riverHeight,
    sky: 0xd5a89c, amplitude: 0.055, waveScale: 1.65, speed: 1.2, depthFalloff: 1.6, caustics: 0.045, sunDir: sky.sunDirection,
  })));
  water.setColors(0x65bfaa, 0x147c79);
  water.uniforms.uSpecular.value = 0.22;
  world.add(water.mesh);
  world.time(world.own(simulateWater(ctx.renderer, water, riverHeight, new THREE.Vector4(-42, -96, 84, 192), new THREE.Vector2(0, 1.1))));
  const rng = rngAt(9, 31, ctx.seed), dummy = new THREE.Object3D();
  const trees: THREE.Matrix4[] = [], stones: THREE.Matrix4[] = [], grass: THREE.Matrix4[] = [];
  const place = (list: THREE.Matrix4[], x: number, y: number, z: number, size: number, width = 1) => {
    dummy.position.set(x, y, z);
    dummy.rotation.set(0, rng() * Math.PI * 2, 0);
    dummy.scale.set(size * width, size, size * width);
    dummy.updateMatrix();
    list.push(dummy.matrix.clone());
  };
  for (let i = 0; i < 1500 && trees.length < 560; i++) {
    const near = i % 3 !== 0, z = near ? rng() * 290 - 100 : -rng() * 850 - 160;
    const { center, width } = riverChannel(z);
    const x = near ? center + (i % 2 ? -1 : 1) * (width + 7 + rng() ** 1.5 * 34) : (rng() - 0.5) * 750;
    const y = riverHeight(x, z);
    if (y < 0.24 || (Math.abs(x) < 24 && Math.abs(z) < 15) || noise(x * 0.03, z * 0.03) < 0.28) continue;
    place(trees, x, y - 0.16, z, 0.72 + rng() * 1.15, 1.06);
  }
  for (const part of spruce(world)) scatter(world, part.geometry, part.material, trees, 'riverSpruce');
  for (let i = 0; i < 1500 && stones.length < 200; i++) {
    const x = (rng() - 0.5) * 135, z = (rng() - 0.5) * 170, y = riverHeight(x, z);
    if (y < WATER_LEVEL - 0.65 || y > 4 || (Math.abs(x) < 13 && Math.abs(z) < 1.7)) continue;
    const size = 0.3 + rng() ** 2 * 2.5, radius = size * 0.46;
    const bed = Math.min(y, riverHeight(x - radius, z), riverHeight(x + radius, z), riverHeight(x, z - radius), riverHeight(x, z + radius));
    place(stones, x, bed - 0.1, z, size, 1.15);
  }
  const rock = fieldstones();
  scatter(world, world.own(rock.geometry), world.own(rock.material), stones, 'riverBoulders');
  for (let i = 0; i < 6500; i++) {
    const x = (rng() - 0.5) * 150, z = (rng() - 0.5) * 180, y = riverHeight(x, z);
    if (y < WATER_LEVEL + 0.7 || y > 7 || (Math.abs(x) < 18 && Math.abs(z) < 1.8)) continue;
    place(grass, x, y, z, 0.65 + rng() * 0.9);
  }
  const tuft = prairieGrass();
  tuft.material.color.set(0x8ca36b);
  scatter(world, world.own(tuft.geometry), world.own(tuft.material), grass, 'riverSedges');
  const fireflies = world.size(world.time(world.own(makeAmbientParticles({ preset: 'fireflies', count: 72, seed: ctx.seed, color: 0xffe6a0, size: 1.3, intensity: 1.8, bounds: { y: 1.1, halfX: 13, halfY: 0.6, halfZ: 10 }, name: 'riverbankFireflies' }))));
  const positions = fireflies.points.geometry.getAttribute('position');
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i), { center, width } = riverChannel(positions.getZ(i));
    positions.setX(i, center + Math.sign(x) * (width + 2 + Math.abs(x) * 0.25));
  }
  world.add(fireflies.points);
  return world.seal<WorldHandle>({
    setView(coverage, camera) {
      camera.far = coverage.far = 5000;
      camera.updateProjectionMatrix();
      camera.position.y = Math.max(camera.position.y, riverHeight(camera.position.x, camera.position.z) + 0.65, WATER_LEVEL + 0.25);
      camera.updateMatrixWorld();
      land.setView(coverage, camera);
    },
  });
}
