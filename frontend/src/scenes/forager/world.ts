import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { buildCourier } from '@kit/cast/courier';
import { makeHeightField } from '@kit/heightfield';
import { makeLandscape, makeSceneTerrain, prairieGrass } from '@kit/landscape';
import { oakMaterial } from '@kit/materials';
import { makeAmbientParticles } from '@kit/particles';
import type { SceneContext } from '@kit/types';
import { broadleafTreeGeometry, plantInstances } from '@kit/vegetation';
import type { ViewCoverage } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { constant, parameter } from '../../data/envs';

export const MEMORY_REACH = 24;
export const MEMORY_STEPS = 640;
export const TRAIL = 24;
export const BERRY_TINT = 0xffd166;
export const BURR_TINT = 0x7a5cf0;
const FADED_TINT = 0x9aa07c;
const MEMORY_SLOTS = (2 * MEMORY_REACH + 1) ** 2;
const POLLEN_HALF = 36;
const POLLEN_SNAP = 24;
const TREE_TRUNK_VERTICES = 48;
const SHELTERBELT_SPACING = 8;

export type PrairieWorld = ReturnType<typeof buildWorld>;

function mergeParts(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const flat = parts.map(part => part.toNonIndexed());
  const merged = mergeGeometries(flat);
  for (const part of parts) part.dispose();
  for (const part of flat) part.dispose();
  return merged;
}

function berryClusterGeometry(): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [new THREE.CylinderGeometry(0.04, 0.06, 0.22, 6).translate(0, 0.11, 0)];
  for (const [x, y, z] of [[0, 0.42, 0], [0.17, 0.3, 0.06], [-0.15, 0.31, 0.1], [0.02, 0.29, -0.17], [0.11, 0.2, 0.15], [-0.13, 0.19, -0.1]] as const) parts.push(new THREE.SphereGeometry(0.15, 12, 9).translate(x, y, z));
  return mergeParts(parts);
}

function burrGeometry(): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [new THREE.IcosahedronGeometry(0.2, 1).translate(0, 0.3, 0)];
  const spike = new THREE.ConeGeometry(0.05, 0.22, 5).translate(0, 0.11, 0), direction = new THREE.Vector3(), quaternion = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < 14; i++) {
    const phi = Math.acos(1 - 2 * (i + 0.5) / 14), theta = Math.PI * (1 + Math.sqrt(5)) * i;
    direction.setFromSphericalCoords(1, phi, theta);
    parts.push(spike.clone().applyQuaternion(quaternion.setFromUnitVectors(up, direction)).translate(direction.x * 0.18, 0.3 + direction.y * 0.18, direction.z * 0.18));
  }
  spike.dispose();
  return mergeParts(parts);
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, side = parameter(env, 'side'), fov = parameter(env, 'fov'), switching = parameter(env, 'switching') !== 0, period = constant(env, 'FORAGER_SWITCH_PERIOD');
  const palette = ctx.tokens.palettes.morning, accent = ctx.tokens.accents[env.family] ?? ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, 'foragerPrairie', palette, 'morning', {
    sky: { sunDir: [0.58, 0.34, -0.74], sunSize: 0.045, sunHdr: 6, sunGlow: 0.38, sunGlowPower: 16, topSpread: 0.55, bottomSpread: 0.3 },
    light: { pointCount: 1, pointDistance: 7 }, fogDensity: 0.0052,
  });
  const { world, rig, sky, fog } = land;
  sky.setColors(0x79b3e6, 0xf3e6c4, 0xd9dcb2);
  fog.color.set(0xdfe6d2);
  rig.sun.color.set(0xfff1cf); rig.sun.intensity = 2.5;
  rig.hemi.color.set(0xcde3f2); rig.hemi.groundColor.set(0x7c8a4e); rig.hemi.intensity = 0.95;
  const lamp = rig.points[0]!;
  lamp.color.set(0xffe2a8); lamp.intensity = 2.2;
  const field = makeHeightField({ octaves: [{ wavelength: 96, amplitude: 1.5, seed: ctx.seed }, { wavelength: 29, amplitude: 0.42, seed: ctx.seed ^ 0x3d }, { wavelength: 7.5, amplitude: 0.07, seed: ctx.seed ^ 0xb2 }] });
  const heightAt = field.heightAt;
  const terrain = world.own(world.time(makeSceneTerrain({
    name: 'prairieGround', seed: ctx.seed, segments: 192, radius: 2048, colorA: 0x8a9c52, colorB: 0xc9c478, height: field.glsl,
    detail: `
vec2 seam = abs(fract((p.xz + 0.5) / ${side.toFixed(1)}) - 0.5) * ${side.toFixed(1)} / max(fwidth(p.xz), vec2(0.0001));
float border = 1.0 - smoothstep(0.5, 1.5, min(seam.x, seam.y));
float checker = mod(floor(p.x + 0.5) + floor(p.z + 0.5), 2.0);
vec2 edge = abs(fract(p.xz + 0.5) - 0.5) / max(fwidth(p.xz), vec2(0.0001));
float line = 1.0 - smoothstep(0.6, 1.4, min(edge.x, edge.y));
color = mix(color * (0.94 + 0.06 * landNoise(p.xz * 0.9)) * (0.96 + 0.08 * checker), vec3(0.62, 0.66, 0.40), line * 0.35);
color = mix(color, vec3(0.92, 0.84, 0.58), border * 0.6);`,
  }, heightAt, { roughness: 0.96, rimColor: palette.skyHorizon, rimStrength: 0.16, toonSteps: 4 })));
  world.add(terrain.mesh);

  const grass = prairieGrass(6);
  land.scatter({
    name: 'prairieGrass', seed: ctx.seed ^ 0x9a7, tileSize: 12, perTile: 22, geometry: grass.geometry, material: grass.material, colorA: 0xeaf0b8, colorB: 0xa6b478,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 22; i++) {
        const x = (tx + rng()) * 12, z = (tz + rng()) * 12;
        write(x, heightAt(x, z), z, rng() * Math.PI * 2, 1.4 + rng() * 1.2, 0.8 + rng() * 0.7, rng());
      }
    },
  });
  const treeGeometry = broadleafTreeGeometry(0), treeColors = treeGeometry.getAttribute('color');
  for (let i = 0; i < treeColors.count; i++) {
    const shade = treeColors.getX(i);
    if (i < TREE_TRUNK_VERTICES) treeColors.setXYZ(i, 0.3 * shade, 0.19 * shade, 0.1 * shade);
    else treeColors.setXYZ(i, 0.46 * shade, 0.66 * shade, 0.3 * shade);
  }
  const belt = Math.ceil(side / SHELTERBELT_SPACING);
  land.scatter({
    name: 'prairieShelterbelts', seed: ctx.seed ^ 0x51c, tileSize: side, perTile: 2 * belt, geometry: treeGeometry, castShadow: true, receiveShadow: true, colorA: 0xffffff, colorB: 0xd6e0b8,
    material: oakMaterial({ vertexColors: true, roughness: 0.9, flatShading: true, rimColor: palette.skyHorizon, rimStrength: 0.25, wind: { amplitude: 0.025, frequency: 0.7, height: 1.8 }, toonSteps: 4, name: 'prairieShelterbeltMaterial' }),
    fill(tx, tz, rng, write) {
      for (let i = 0; i < belt; i++) for (const edge of [0, 1]) {
        if (rng() > 0.6) continue;
        const along = (i + 0.5) * SHELTERBELT_SPACING + (rng() - 0.5) * 1.5, across = -0.5 + (rng() - 0.5) * 0.6;
        const x = edge ? tx * side + across : tx * side + along, z = edge ? tz * side + along : tz * side + across;
        write(x, heightAt(x, z) - 0.05, z, rng() * Math.PI * 2, 1.0 + rng() * 0.6, 0.9 + rng() * 0.5, rng());
      }
    },
  });
  const pollen = world.own(world.size(world.time(makeAmbientParticles({ preset: 'pollen', seed: ctx.seed, count: 1200, color: 0xfff0b8, size: 0.9, intensity: 1.3, opacity: 0.7, bounds: { y: 1.4, halfX: POLLEN_HALF, halfY: 1.6, halfZ: POLLEN_HALF }, name: 'prairiePollen' }))));
  world.add(pollen.points);

  const berryTint = new THREE.Color(BERRY_TINT), burrTint = new THREE.Color(BURR_TINT), faded = new THREE.Color(FADED_TINT), color = new THREE.Color();
  const itemMaterial = oakMaterial({ roughness: 0.45, toonSteps: 3, rimColor: 0xffffff, rimStrength: 0.5, name: 'prairieItemMaterial' });
  const berries = plantInstances(world, 'berryClusters', berryClusterGeometry(), itemMaterial, MEMORY_SLOTS);
  const burrs = plantInstances(world, 'burrs', burrGeometry(), itemMaterial, MEMORY_SLOTS);
  for (const plant of [berries, burrs]) { plant.mesh.frustumCulled = false; plant.mesh.castShadow = true; plant.mesh.receiveShadow = true; }
  const trail = plantInstances(world, 'footprints', new THREE.CylinderGeometry(0.17, 0.17, 0.03, 12).translate(0, 0.015, 0), new THREE.MeshBasicMaterial({ color: accent, transparent: true, opacity: 0.55, depthWrite: false }), TRAIL);
  trail.mesh.frustumCulled = false;
  const span = 2 * fov + 1, half = span / 2, bar = 0.12;
  const bars: THREE.BufferGeometry[] = [new THREE.BoxGeometry(span + bar, 0.08, bar).translate(0, 0, -half), new THREE.BoxGeometry(span + bar, 0.08, bar).translate(0, 0, half), new THREE.BoxGeometry(bar, 0.08, span - bar).translate(-half, 0, 0), new THREE.BoxGeometry(bar, 0.08, span - bar).translate(half, 0, 0)];
  for (const [x, z] of [[-half, -half], [half, -half], [half, half], [-half, half]] as const) bars.push(new THREE.CylinderGeometry(0.09, 0.11, 0.42, 8).translate(x, 0.17, z));
  const lens = new THREE.Group();
  lens.name = 'observationWindow';
  const frameMesh = new THREE.Mesh(world.own(mergeParts(bars)), world.own(new THREE.MeshStandardMaterial({ color: ctx.tokens.cueCyan, emissive: ctx.tokens.cueCyan, emissiveIntensity: 1.6, roughness: 0.4 })));
  frameMesh.castShadow = true;
  lens.add(frameMesh);
  const floorGlow = new THREE.Mesh(world.own(new THREE.PlaneGeometry(span, span).rotateX(-Math.PI / 2)), world.own(new THREE.MeshBasicMaterial({ color: ctx.tokens.cueCyan, transparent: true, opacity: 0.14, depthWrite: false })));
  floorGlow.position.y = 0.02;
  lens.add(floorGlow);
  world.add(lens);
  const courier = world.own(buildCourier({ name: 'prairieCourier', tint: accent, overgrowth: 0.7, scale: 1.5 }));
  world.add(courier.group);

  const cells = side * side, seen = new Int32Array(cells).fill(-1), kinds = new Uint8Array(cells);
  const wrap = (v: number): number => ((v % side) + side) % side;
  const goodAt = (step: number): number => (switching && (Math.floor(step / period) & 1) ? 2 : 1);
  const handle = {
    heightAt, side, fov, switching, period, courier, lens, trail, goodAt, lights: rig, lamp, accent,
    forget() { seen.fill(-1); kinds.fill(0); },
    remember(step: number, x: number, y: number, obs: Float32Array) {
      const g = goodAt(step);
      for (let dy = 0; dy < span; dy++) for (let dx = 0; dx < span; dx++) {
        const cell = wrap(y + dy - fov) * side + wrap(x + dx - fov), value = obs[dy * span + dx]!;
        seen[cell] = step;
        kinds[cell] = value === 0 ? 0 : value > 0 ? g : 3 - g;
      }
    },
    show(step: number, cx: number, cz: number) {
      const g = goodAt(step);
      let berryCount = 0, burrCount = 0;
      for (let dz = -MEMORY_REACH; dz <= MEMORY_REACH; dz++) for (let dx = -MEMORY_REACH; dx <= MEMORY_REACH; dx++) {
        const ix = cx + dx, iz = cz + dz, cell = wrap(iz) * side + wrap(ix), kind = kinds[cell]!;
        if (kind === 0) continue;
        const strength = 1 - (step - seen[cell]!) / MEMORY_STEPS;
        if (strength <= 0) continue;
        const inView = Math.abs(dx) <= fov && Math.abs(dz) <= fov;
        const plant = kind === 1 ? berries : burrs, transform = plant.transform;
        transform.position.set(ix, heightAt(ix, iz), iz);
        transform.scale.setScalar(inView ? 1.15 : 0.55 + 0.45 * strength);
        transform.rotation.y = (ix * 7 + iz * 13) % 6;
        color.copy(kind === 1 ? berryTint : burrTint);
        if (inView) color.multiplyScalar(kind === g ? 1.9 : 1.3);
        else color.lerp(faded, 0.35 + 0.55 * (1 - strength));
        plant.write(kind === 1 ? berryCount++ : burrCount++, color);
      }
      berries.commit(berryCount); burrs.commit(burrCount);
    },
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) {
      land.setView(coverage, camera, terrain);
      const px = Math.round(coverage.x / POLLEN_SNAP) * POLLEN_SNAP, pz = Math.round(coverage.z / POLLEN_SNAP) * POLLEN_SNAP;
      pollen.points.position.set(px, heightAt(px, pz), pz);
    },
  };
  return world.seal<WorldCore & typeof handle>(handle);
}
