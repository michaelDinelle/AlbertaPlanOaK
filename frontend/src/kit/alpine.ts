import * as THREE from 'three/webgpu';
import { texture } from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { buildAurora } from './aurora';
import { simulateWater } from './fluid';
import { SpatialInstances } from './instances';
import { fieldstones, makeLandscape, prairieGrass } from './landscape';
import { makeMountains, terrainNoise as noise } from './mountains';
import { rngAt } from './rng';
import type { PaletteSkyOptions } from './sky';
import type { Palette } from './tokens';
import type { SceneContext } from './types';
import type { ViewCoverage } from './view';
import { makeWater, type WaterOptions } from './water';
import type { WorldBuilder, WorldCore } from './world';

const smooth = THREE.MathUtils.smoothstep;

export function alpineGroundGeometry(heightAt: (x: number, z: number) => number, waterLevel: number) {
  const geometry = new THREE.PlaneGeometry(2, 2, 256, 256);
  geometry.rotateX(-Math.PI / 2);
  const positions = geometry.getAttribute('position');
  const colors = new Float32Array(positions.count * 3), color = new THREE.Color();
  const gravel = new THREE.Color(0xc1c7b5), moss = new THREE.Color(0x577e3c), silt = new THREE.Color(0x7e9b80);
  const stretch = (p: number) => Math.sign(p) * (5 * Math.abs(p) + 2043 * Math.abs(p) ** 3);
  for (let i = 0; i < positions.count; i++) {
    const x = stretch(positions.getX(i)), z = stretch(positions.getZ(i)), y = heightAt(x, z);
    positions.setXYZ(i, x, y, z);
    color.copy(gravel).lerp(moss, smooth(y - waterLevel, 0.5, 2.4) * (0.9 + noise(x * 0.09, z * 0.09) * 0.1));
    color.lerp(silt, 1 - smooth(y, waterLevel - 8, waterLevel - 2));
    color.multiplyScalar(0.72 + noise(x * 0.18, z * 0.18) * 0.28);
    colors.set([color.r, color.g, color.b], i * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  return geometry;
}

export function spruce(world: WorldBuilder) {
  const wood: THREE.BufferGeometry[] = [], needles: THREE.BufferGeometry[] = [];
  const timber = (top: number, bottom: number, height: number, segments: number) => {
    const geometry = new THREE.CylinderGeometry(top, bottom, height, segments);
    wood.push(geometry);
    return geometry;
  };
  timber(0.022, 0.16, 7.5, 9).translate(0, 3.75, 0);
  for (let i = 0; i < 54; i++) {
    const height = 0.9 + i / 54 * 6.5, angle = i * 2.3999632;
    const length = (1 - (height - 0.9) / 7.2) * (1.2 + noise(i, 0) * 0.65);
    if (i % 2 === 0) {
      const direction = new THREE.Vector3(Math.cos(angle), -0.16, Math.sin(angle)).multiplyScalar(length);
      timber(0.006, 0.032 * (1 - height / 8), direction.length(), 5)
        .applyQuaternion(new THREE.Quaternion().setFromUnitVectors(THREE.Object3D.DEFAULT_UP, direction.clone().normalize()))
        .translate(direction.x / 2, height + direction.y / 2, direction.z / 2);
    }
    for (const tilt of [-0.32, 0.45]) {
      const spray = new THREE.PlaneGeometry(length * 1.15, length * 1.24, 1, 2);
      const p = spray.getAttribute('position');
      for (let j = 0; j < p.count; j++) {
        const t = p.getY(j) / (length * 1.24) + 0.5;
        p.setXYZ(j, p.getX(j), t * length, Math.sin(t * Math.PI) * length * 0.13);
      }
      spray.rotateX(-Math.PI / 2 + tilt).rotateY(-Math.PI / 2 - angle).translate(0, height, 0);
      spray.computeVertexNormals();
      needles.push(spray);
    }
  }
  const trunk = world.own(mergeGeometries(wood)), foliage = world.own(mergeGeometries(needles));
  for (const geometry of [...wood, ...needles]) geometry.dispose();
  const coverage = world.own(new THREE.TextureLoader().load('/textures/alpine-spruce.webp'));
  coverage.anisotropy = 4;
  const bark = world.own(new THREE.MeshStandardNodeMaterial({ color: 0x746350, roughness: 1 }));
  const leaf = world.own(new THREE.MeshStandardNodeMaterial({ color: 0x527961, side: THREE.DoubleSide, alphaTest: 0.4, alphaToCoverage: true, roughness: 0.92 }));
  leaf.opacityNode = texture(coverage).a;
  return [{ geometry: trunk, material: bark }, { geometry: foliage, material: leaf }];
}

export function scatter(world: WorldBuilder, geometry: THREE.BufferGeometry, material: THREE.NodeMaterial, transforms: THREE.Matrix4[], name: string) {
  const instances = world.own(new SpatialInstances([{ geometry, material, padding: 1 }]));
  instances.group.name = name;
  instances.scatter(Float32Array.from(transforms.flatMap(matrix => matrix.elements)));
  world.add(instances.group);
}

export function alpineMaterial(world: WorldBuilder) {
  return world.own(new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 1 }));
}

export type Placer = (x: number, y: number, z: number, scale: readonly [number, number, number], yaw: number, tilt?: readonly [number, number]) => void;

export interface AlpineOptions {
  name: string;
  palette: Palette;
  mood: string;
  sky: PaletteSkyOptions;
  skyColors: readonly [THREE.ColorRepresentation, THREE.ColorRepresentation, THREE.ColorRepresentation];
  fog: THREE.ColorRepresentation;
  fogDensity: number;
  sun: { color: THREE.ColorRepresentation; intensity: number };
  hemi: { sky: THREE.ColorRepresentation; ground: THREE.ColorRepresentation; intensity: number };
  shadow: { focus: readonly [number, number, number]; radius: number };
  aurora?: { color: THREE.ColorRepresentation; intensity: number; gain: number };
  heightAt: (x: number, z: number) => number;
  waterLevel: number;
  mountains: { region?: 'banff-canmore' | 'moraine-lake'; rotation: number; position: readonly [number, number, number]; name: string };
  water: Omit<WaterOptions, 'palette' | 'level' | 'bedHeight' | 'sunDir'> & { colors: readonly [THREE.ColorRepresentation, THREE.ColorRepresentation]; specular: number };
  fluid: { bounds: readonly [number, number, number, number]; flow?: readonly [number, number] };
  seedStream: readonly [number, number];
  trees(rng: () => number, place: Placer): void;
  stones(rng: () => number, place: Placer): void;
  grass(rng: () => number, place: Placer): void;
  grassColor: THREE.ColorRepresentation;
  extras?(world: WorldBuilder, rng: () => number): void;
}

export interface AlpineWorld extends WorldCore {
  heightAt(x: number, z: number): number;
  waterLevel: number;
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export async function alpineWorld(ctx: SceneContext, scene: THREE.Scene, opts: AlpineOptions): Promise<AlpineWorld> {
  const land = makeLandscape(scene, opts.name, opts.palette, opts.mood, { sky: opts.sky, light: { pointCount: 0 }, fogDensity: opts.fogDensity });
  const { world, rig, sky, fog } = land;
  sky.setColors(opts.skyColors[0], opts.skyColors[1], opts.skyColors[2]);
  if (opts.aurora) world.time(world.own(buildAurora(sky, opts.aurora.color, opts.aurora.intensity))).setGain(opts.aurora.gain);
  fog.color.set(opts.fog);
  rig.sun.color.set(opts.sun.color); rig.sun.intensity = opts.sun.intensity;
  rig.hemi.color.set(opts.hemi.sky); rig.hemi.groundColor.set(opts.hemi.ground); rig.hemi.intensity = opts.hemi.intensity;
  rig.setShadowFocus(new THREE.Vector3(...opts.shadow.focus), opts.shadow.radius);
  const terrain = world.add(new THREE.Mesh(world.own(alpineGroundGeometry(opts.heightAt, opts.waterLevel)), alpineMaterial(world)));
  terrain.name = `${opts.name}Ground`;
  terrain.receiveShadow = true;
  const peaks = await makeMountains(world, opts.palette, opts.mountains.region);
  peaks.rotation.y = opts.mountains.rotation;
  peaks.position.set(...opts.mountains.position);
  peaks.name = opts.mountains.name;
  const { colors, specular, ...waterOptions } = opts.water;
  const water = world.time(world.own(makeWater({ ...waterOptions, palette: opts.palette, level: opts.waterLevel, bedHeight: opts.heightAt, sunDir: sky.sunDirection })));
  water.setColors(colors[0], colors[1]);
  water.uniforms.uSpecular.value = specular;
  world.add(water.mesh);
  world.time(world.own(simulateWater(ctx.renderer, water, opts.heightAt, new THREE.Vector4(...opts.fluid.bounds), opts.fluid.flow ? new THREE.Vector2(...opts.fluid.flow) : undefined)));
  const rng = rngAt(opts.seedStream[0], opts.seedStream[1], ctx.seed), dummy = new THREE.Object3D();
  const placer = (list: THREE.Matrix4[]): Placer => (x, y, z, scale, yaw, tilt = [0, 0]) => {
    dummy.position.set(x, y, z);
    dummy.rotation.set(tilt[0], yaw, tilt[1]);
    dummy.scale.set(scale[0], scale[1], scale[2]);
    dummy.updateMatrix();
    list.push(dummy.matrix.clone());
  };
  const trees: THREE.Matrix4[] = [], stones: THREE.Matrix4[] = [], grass: THREE.Matrix4[] = [];
  opts.trees(rng, placer(trees));
  for (const part of spruce(world)) scatter(world, part.geometry, part.material, trees, `${opts.name}Spruce`);
  opts.stones(rng, placer(stones));
  const rock = fieldstones();
  scatter(world, world.own(rock.geometry), world.own(rock.material), stones, `${opts.name}Boulders`);
  opts.grass(rng, placer(grass));
  const tuft = prairieGrass();
  tuft.material.color.set(opts.grassColor);
  scatter(world, world.own(tuft.geometry), world.own(tuft.material), grass, `${opts.name}Sedges`);
  opts.extras?.(world, rng);
  return world.seal<AlpineWorld>({
    heightAt: opts.heightAt,
    waterLevel: opts.waterLevel,
    setView(coverage, camera) { land.setView(coverage, camera); },
  });
}
