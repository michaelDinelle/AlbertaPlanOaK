import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { mountainGeometry, terrainNoise as noise } from '@kit/mountains';
import { rngAt } from '@kit/rng';
import { makeWater } from '@kit/water';
import { simulateWater } from '@kit/fluid';
import type { SceneContext } from '@kit/types';
import type { WorldCore, WorldBuilder } from '@kit/world';
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

function groundGeometry() {
  const geometry = new THREE.PlaneGeometry(2, 2, 256, 256);
  geometry.rotateX(-Math.PI / 2);
  const positions = geometry.getAttribute('position'), uv = geometry.getAttribute('uv');
  const colors = new Float32Array(positions.count * 3), color = new THREE.Color();
  const gravel = new THREE.Color(0xcbd3c8), moss = new THREE.Color(0x687b50), silt = new THREE.Color(0xb2bb9d);
  const stretch = (p: number) => Math.sign(p) * (5 * Math.abs(p) + 2043 * Math.abs(p) ** 3);
  for (let i = 0; i < positions.count; i++) {
    const x = stretch(positions.getX(i)), z = stretch(positions.getZ(i)), y = lakeHeight(x, z);
    positions.setXYZ(i, x, y, z);
    uv.setXY(i, x * 0.6, z * 0.6);
    color.copy(gravel).lerp(moss, smooth(y - WATER_Y, 0.5, 2.4) * (0.45 + noise(x * 0.09, z * 0.09) * 0.55));
    color.lerp(silt, 1 - smooth(y, WATER_Y - 8, WATER_Y - 2));
    color.multiplyScalar(0.72 + noise(x * 0.18, z * 0.18) * 0.28);
    colors.set([color.r, color.g, color.b], i * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function texture(world: WorldBuilder, name: string, wrap: THREE.Wrapping = THREE.MirroredRepeatWrapping) {
  const map = world.own(new THREE.TextureLoader().load(`/textures/${name}.webp`));
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = wrap;
  map.anisotropy = 4;
  return map;
}

function spruce(world: WorldBuilder) {
  const wood: THREE.BufferGeometry[] = [], needles: THREE.BufferGeometry[] = [];
  const timber = (top: number, bottom: number, height: number, segments: number) => {
    const geometry = new THREE.CylinderGeometry(top, bottom, height, segments), uv = geometry.getAttribute('uv');
    for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * Math.max(1, Math.round(bottom * Math.PI * 2 / 0.7)), uv.getY(i) * height / 0.7);
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
  const map = texture(world, 'alpine-spruce', THREE.ClampToEdgeWrapping), barkMap = texture(world, 'alpine-bark', THREE.RepeatWrapping);
  const bark = world.own(new THREE.MeshStandardMaterial({ map: barkMap, bumpMap: barkMap, bumpScale: 0.018, roughness: 1 }));
  const leaf = world.own(new THREE.MeshStandardMaterial({ map, color: 0xffffff, side: THREE.DoubleSide, alphaTest: 0.4, alphaToCoverage: true, roughness: 0.92 }));
  return [{ geometry: trunk, material: bark }, { geometry: foliage, material: leaf }];
}

function scatter(world: WorldBuilder, geometry: THREE.BufferGeometry, material: THREE.Material, transforms: THREE.Matrix4[], name: string) {
  const tiles = new Map<string, THREE.Matrix4[]>();
  for (const matrix of transforms) {
    const key = `${Math.floor(matrix.elements[12]! / 64)},${Math.floor(matrix.elements[14]! / 64)}`;
    if (!tiles.has(key)) tiles.set(key, []);
    tiles.get(key)!.push(matrix);
  }
  for (const matrices of tiles.values()) {
    const mesh = world.own(new THREE.InstancedMesh(geometry, material, matrices.length));
    mesh.name = name;
    matrices.forEach((matrix, i) => mesh.setMatrixAt(i, matrix));
    mesh.computeBoundingSphere();
    world.add(mesh);
  }
}

export interface WorldHandle extends WorldCore {
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export async function buildWorld(ctx: SceneContext, scene: THREE.Scene): Promise<WorldHandle> {
  const mountains = await mountainGeometry('moraine-lake');
  const land = makeLandscape(scene, 'alpineLake', ctx.tokens.palettes.dawn, 'dawn', {
    sky: { sunDir: [-0.6, 0.48, 0.64], sunSize: 0.018, sunHdr: 1.8, sunGlow: 0.12, topSpread: 0.23 },
    light: { pointCount: 0, shadowMapSize: 1024 }, fogDensity: 0.00018,
  });
  const { world, rig, sky, fog } = land;
  sky.setColors(0x426faa, 0xf3c3b7, 0xb0c1c7);
  fog.color.set(0xb3c5d0);
  rig.sun.color.set(0xffe2c2);
  rig.sun.intensity = 2.7;
  rig.hemi.color.set(0xc8def1);
  rig.hemi.groundColor.set(0x65796c);
  rig.hemi.intensity = 2.1;
  rig.setShadowFocus(new THREE.Vector3(0, -1, 0), 7);
  const map = texture(world, 'alpine-pebbles');
  const moss = texture(world, 'prairie-ground');
  const ground = world.own(new THREE.MeshStandardMaterial({ map, bumpMap: map, bumpScale: 0.025, vertexColors: true, roughness: 1 }));
  ground.onBeforeCompile = shader => {
    shader.uniforms.uForestFloor = { value: moss };
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vLakeGround; varying vec3 vLakeNormal;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvLakeGround = position; vLakeNormal = normal;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nuniform sampler2D uForestFloor; varying vec3 vLakeGround; varying vec3 vLakeNormal;')
      .replace('#include <map_fragment>', `
vec3 p = vLakeGround, blend = pow(abs(vLakeNormal), vec3(4.0));
blend /= dot(blend, vec3(1.0));
vec3 gravel = texture2D(map, p.zy * 0.6).rgb * blend.x + texture2D(map, p.xz * 0.6).rgb * blend.y + texture2D(map, p.xy * 0.6).rgb * blend.z;
vec3 forest = texture2D(uForestFloor, p.xz * 0.45).rgb * vec3(1.2, 1.55, 0.95);
float cover = smoothstep(${WATER_Y + 0.4}, ${WATER_Y + 1.8}, p.y) * smoothstep(0.45, 0.85, vLakeNormal.y);
diffuseColor.rgb *= mix(gravel, forest, cover);
`);
  };
  const terrain = world.add(new THREE.Mesh(world.own(groundGeometry()), ground));
  terrain.name = 'alpineLakebed';
  terrain.receiveShadow = true;
  const stone = texture(world, 'alpine-limestone');
  const cliff = world.own(new THREE.MeshStandardMaterial({ map: stone, bumpMap: stone, bumpScale: 0.8, vertexColors: true, roughness: 1 }));
  cliff.onBeforeCompile = shader => {
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vCliff; varying vec3 vCliffNormal;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvCliff = position; vCliffNormal = normal;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <map_pars_fragment>', `#include <map_pars_fragment>
varying vec3 vCliff; varying vec3 vCliffNormal;
vec3 cliffAlbedo(vec2 uv) {
  return mix(texture2D(map, uv).rgb, texture2D(map, uv * 0.373 + vec2(13.17, 7.81)).rgb, 0.55);
}`)
      .replace('#include <map_fragment>', `
vec3 blend = pow(abs(vCliffNormal), vec3(4.0));
blend /= dot(blend, vec3(1.0));
vec3 rock = cliffAlbedo(vCliff.zy * 0.07) * blend.x + cliffAlbedo(vCliff.xz * 0.07) * blend.y + cliffAlbedo(vCliff.xy * 0.07) * blend.z;
float snow = smoothstep(0.6, 0.86, max(vColor.r, max(vColor.g, vColor.b)));
diffuseColor.rgb *= mix(rock * 2.6, vec3(1.0), snow * 0.85);
`);
  };
  mountains.rotateY(Math.PI - 0.35).translate(490, 10, -2150);
  const peaks = world.add(new THREE.Mesh(world.own(mountains), cliff));
  peaks.name = 'canadianRockies';
  const water = world.time(world.own(makeWater({ size: 1024, segments: 256, palette: ctx.tokens.palettes.dawn, level: WATER_Y, bedHeight: lakeHeight,
    shallow: 0x449f94, deep: 0x086b78, sky: 0xc7e5f0, amplitude: 0.045, waveScale: 1.3, speed: 0.85, depthFalloff: 3.5, caustics: 0.04, sunDir: sky.sunDirection,
  })));
  water.setColors(0x4ab7a8, 0x096a7a);
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
