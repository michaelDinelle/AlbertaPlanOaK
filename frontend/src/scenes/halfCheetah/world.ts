import * as THREE from 'three/webgpu';
import { attribute, float, length, vec2 } from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { buildAurora } from '@kit/aurora';
import { fieldstones, makeLandscape, prairieGrass } from '@kit/landscape';
import type { LightRig } from '@kit/lighting';
import { makeMountains } from '@kit/mountains';
import { hoodooGeometry } from '@kit/props';
import { rngAt } from '@kit/rng';
import type { SkyHandle } from '@kit/sky';
import type { SceneContext } from '@kit/types';
import type { ViewCoverage } from '@kit/view';
import type { WorldBuilder, WorldCore } from '@kit/world';
import { makePrairieTerrain, prairieGroundHeight, prairieMaterial, prairiePlacement, prairieRidge } from './terrain';

export { prairieHeight } from './terrain';

const ASPEN_FOLIAGE = [0x8ca357, 0xc6ce8a, 0xa6bd70, 0xd1d28b];

function aspenTree() {
  const pieces: THREE.BufferGeometry[] = [];
  const add = (geometry: THREE.BufferGeometry, color: THREE.ColorRepresentation) => {
    const count = geometry.getAttribute('position').count, colors = new Float32Array(count * 3), tint = new THREE.Color(color);
    if (!geometry.hasAttribute('aLeaf')) geometry.setAttribute('aLeaf', new THREE.Float32BufferAttribute(new Float32Array(count * 2).fill(-2), 2));
    for (let i = 0; i < count; i++) tint.toArray(colors, i * 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    pieces.push(geometry);
  };
  add(new THREE.CylinderGeometry(0.018, 0.11, 3.7, 10).translate(0, 1.82, 0), 0xf0e9d5);
  for (let i = 0; i < 18; i++) {
    const angle = i * 2.39996, y = 1.15 + i * 0.148, spread = Math.sin((i + 2) / 21 * Math.PI) * 0.86;
    const end = new THREE.Vector3(Math.cos(angle) * spread, y + 0.38, Math.sin(angle) * spread);
    const start = new THREE.Vector3(0, y, 0), direction = end.clone().sub(start);
    add(new THREE.CylinderGeometry(0.004, 0.019, direction.length(), 5).applyQuaternion(new THREE.Quaternion().setFromUnitVectors(THREE.Object3D.DEFAULT_UP, direction.clone().normalize())).translate(...start.add(end).multiplyScalar(0.5).toArray()), 0xd1cdbb);
    for (let j = 0; j < 38; j++) {
      const t = j / 38, azimuth = j * 2.39996 + angle, radial = Math.sqrt(t) * 0.52;
      const orientation = new THREE.Quaternion().setFromEuler(new THREE.Euler(azimuth * 0.7, azimuth, Math.sin(j * 1.7) * 0.8, 'ZYX'));
      const center = new THREE.Vector3(end.x + Math.cos(azimuth) * radial, end.y + Math.sin(j * 3.7) * radial * 0.9, end.z + Math.sin(azimuth) * radial);
      const leaf = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([0, -0.1425, 0.017, 0.111, 0, -0.009, 0, 0.1425, 0.017, -0.111, 0, -0.009], 3)).setIndex([0, 1, 2, 0, 2, 3]);
      leaf.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(8), 2));
      leaf.setAttribute('aLeaf', new THREE.Float32BufferAttribute([0, -1.5, 1.5, 0, 0, 1.5, -1.5, 0], 2));
      leaf.applyQuaternion(orientation).translate(center.x, center.y, center.z);
      leaf.computeVertexNormals();
      add(leaf, ASPEN_FOLIAGE[(i + j) % ASPEN_FOLIAGE.length]!);
    }
  }
  const geometry = mergeGeometries(pieces);
  pieces.forEach(piece => piece.dispose());
  const material = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.92, side: THREE.DoubleSide });
  const leaf = attribute<'vec2'>('aLeaf', 'vec2');
  const blade = length(vec2(leaf.x.mul(1.06), leaf.y.mul(0.9).add(0.14))).lessThan(1);
  const tip = leaf.y.greaterThan(0.5).and(leaf.x.abs().lessThan(float(1.45).sub(leaf.y).mul(0.55)));
  const stem = leaf.y.lessThan(-0.7).and(leaf.x.abs().lessThan(0.07));
  material.maskNode = leaf.x.lessThan(-1.9).or(blade).or(tip).or(stem);
  material.name = 'aspenGroveBark';
  return { geometry, material };
}

function makeHoodoos(world: WorldBuilder, seed: number) {
  const source = world.own(hoodooGeometry());
  const material = world.own(prairieMaterial(true));
  const meshes = new Map<string, THREE.Mesh>();
  let previous = '';
  let pending: { key: string; x: number; z: number; rng: () => number }[] = [];
  const build = ({ key, x, z, rng }: (typeof pending)[number]): void => {
    const scale = 1.2 + rng() * 0.5, angle = (rng() - 0.5) * 0.4;
    const geometry = source.clone().scale(scale, scale, scale).rotateY(angle);
    const positions = geometry.getAttribute('position'), colors = geometry.getAttribute('color');
    const exposure = new Float32Array(positions.count);
    const anchor = prairieGroundHeight(x, z);
    for (let i = 0; i < positions.count; i++) {
      const px = x + positions.getX(i), pz = z + positions.getZ(i), py = positions.getY(i);
      const ground = prairieGroundHeight(px, pz);
      const y = py + THREE.MathUtils.lerp(ground, anchor, THREE.MathUtils.smoothstep(py / scale, 0.3, 1.8)) - 0.08;
      positions.setXYZ(i, px, y, pz);
      const blend = THREE.MathUtils.smoothstep(py / scale, 0, 0.9);
      exposure[i] = THREE.MathUtils.lerp(prairieRidge(px, pz).exposure, 1, blend);
      colors.setXYZ(i, THREE.MathUtils.lerp(1, colors.getX(i) * 2.2, blend), THREE.MathUtils.lerp(1, colors.getY(i) * 2.5, blend), THREE.MathUtils.lerp(1, colors.getZ(i) * 3.2, blend));
    }
    geometry.setAttribute('exposure', new THREE.BufferAttribute(exposure, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    const mesh = world.add(new THREE.Mesh(geometry, material));
    mesh.name = 'sandstoneHoodoos';
    mesh.castShadow = mesh.receiveShadow = true;
    meshes.set(key, mesh);
  };
  return world.own({
    setView(coverage: ViewCoverage) {
      const cx = Math.floor(coverage.x / 64), cz = Math.floor(coverage.z / 96), center = `${cx},${cz}`;
      if (center !== previous) {
        previous = center;
        const active = new Set<string>();
        pending = [];
        for (let tx = cx - 6; tx <= cx + 6; tx++) for (let tz = cz - 5; tz <= cz + 5; tz++) {
          const rng = rngAt(tx, tz, seed), x = tx * 64 + (rng() - 0.5) * 32;
          const ridge = prairieRidge(x, (tz + 0.5) * 96);
          const z = Math.sign(tz + 0.5) * (ridge.center - ridge.width * 0.57);
          if (tz !== Math.floor(z / 96) || prairieRidge(x, z).exposure < 0.35) continue;
          const key = `${tx},${tz}`;
          active.add(key);
          if (!meshes.has(key)) pending.push({ key, x, z, rng });
        }
        pending.sort((a, b) => (b.x - coverage.x) ** 2 + (b.z - coverage.z) ** 2 - (a.x - coverage.x) ** 2 - (a.z - coverage.z) ** 2);
        for (const [key, mesh] of meshes) if (!active.has(key)) {
          mesh.geometry.dispose();
          mesh.removeFromParent();
          meshes.delete(key);
        }
      }
      const start = performance.now();
      while (pending.length > 0 && performance.now() - start < 2) build(pending.pop()!);
    },
    dispose() {
      for (const mesh of meshes.values()) { mesh.geometry.dispose(); mesh.removeFromParent(); }
      meshes.clear();
    },
  });
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
  setShadowTarget(x: number): void;
  setGroundShift(value: number): void;
}

export async function buildWorld(ctx: SceneContext, scene: THREE.Scene): Promise<WorldHandle> {
  const land = makeLandscape(scene, 'albertaPrairieSunrise', ctx.tokens.palettes.dusk, 'dawn', {
    sky: { topSpread: 0.18, sunSize: 0.035, sunGlow: 0.35, sunDir: [-0.62, 0.28, -0.73], stars: { intensity: 3, density: 180, threshold: 0.976, size: 0.08, color: 0xe4e4ff } }, light: { pointCount: 0 }, fogDensity: 0.00022,
  });
  const { world, sky, rig, fog } = land;
  sky.setColors(0x352d65, 0xff9fad, 0xe9ae9c);
  world.time(world.own(buildAurora(sky, 0x95ddd0, 2.6))).setGain(1.5);
  const terrain = makePrairieTerrain(world);
  const mountains = await makeMountains(world, ctx.tokens.palettes.dusk);
  mountains.name = 'rockyMountainRanges';
  rig.sun.color.set(0xffd2ad);
  rig.sun.intensity = 2.5;
  rig.hemi.color.set(0xd7d8ef);
  rig.hemi.groundColor.set(0xc2a074);
  rig.hemi.intensity = 2.1;
  fog.color.set(0xb8a2b6);
  const hoodoos = makeHoodoos(world, ctx.seed);
  land.scatter({
    name: 'prairieBunchgrass', seed: ctx.seed, tileSize: 28, perTile: 448,
    ...prairieGrass(), colorA: 0xc1c895, colorB: 0xf1dfa9,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 448; i++) {
        const x = (tx + rng()) * 28, z = (tz + rng()) * 28;
        if (Math.abs(z) < 0.9 || prairieRidge(x, z).exposure > 0.16) continue;
        write(x, prairieGroundHeight(x, z) - 0.01, z, rng() * Math.PI * 2, 0.95 + rng() * 0.7, 0.8 + rng() * 0.4, rng());
      }
    },
  }, true);
  land.scatter({
    name: 'aspenGroves', seed: ctx.seed, tileSize: 48, perTile: 5,
    ...aspenTree(),
    colorA: 0xd4d3af, colorB: 0xffffff, castShadow: true,
    fill(tx, tz, rng, write) {
      const x = tx * 48 + 10 + rng() * 26, z = tz * 48 + 9 + rng() * 28;
      if (Math.abs(z) < 7 || prairieRidge(x, z).exposure > 0.06 || Math.sin(x * 0.047 + z * 0.027) < -0.1) return;
      for (let i = 0; i < 5; i++) {
        const px = x + Math.cos(i * 2.4) * Math.sqrt(i) * 1.5, pz = z + Math.sin(i * 2.4) * Math.sqrt(i) * 1.5;
        write(px, prairieGroundHeight(px, pz) - 0.08, pz, rng() * 6.28, 0.85 + rng() * 0.6, 0.9 + rng() * 0.25, rng());
      }
    },
  });
  land.scatter({
    name: 'prairieFieldstones', seed: ctx.seed, tileSize: 20, perTile: 4,
    ...fieldstones(),
    colorA: 0xd7d1c7, colorB: 0xffffff,
    fill(tx, tz, rng, write) {
      const x = tx * 20 + 4 + rng() * 12, z = tz * 20 + 4 + rng() * 12;
      if (Math.abs(z) < 2) return;
      for (let i = 0; i < 4; i++) {
        const px = x + rng() * 2.4, pz = z + rng() * 1.8;
        const scale = 1.3 + rng() * 1.6;
        const place = prairiePlacement(px, pz, scale * 0.4);
        write(px, place.height - 0.04 * scale, pz, rng() * 6.28, scale, 0.7 + rng() * 0.4, rng(), place.normal);
      }
    },
  });
  const shadowRight = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), rig.sunDirection).normalize();
  const shadowUp = new THREE.Vector3().crossVectors(rig.sunDirection, shadowRight).normalize();
  const shadowCenter = new THREE.Vector3();
  return world.seal<WorldHandle>({
    sky, rig,
    setView(coverage, camera) {
      land.setView(coverage, camera);
      terrain.setView(coverage, camera);
      hoodoos.setView(coverage);
      mountains.position.x = coverage.x;
    },
    setGroundShift: terrain.setShift,
    setShadowTarget(x) {
      shadowCenter.set(x, 0.65, 0);
      const texel = 14 / rig.sun.shadow.mapSize.x;
      const right = shadowCenter.dot(shadowRight);
      const up = shadowCenter.dot(shadowUp);
      shadowCenter.addScaledVector(shadowRight, Math.round(right / texel) * texel - right);
      shadowCenter.addScaledVector(shadowUp, Math.round(up / texel) * texel - up);
      rig.setShadowFocus(shadowCenter, 7);
    },
  });
}
