import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { terrainNoise as noise } from './mountains';
import type { WorldBuilder } from './world';

const smooth = THREE.MathUtils.smoothstep;

export function alpineGroundGeometry(heightAt: (x: number, z: number) => number, waterLevel: number) {
  const geometry = new THREE.PlaneGeometry(2, 2, 256, 256);
  geometry.rotateX(-Math.PI / 2);
  const positions = geometry.getAttribute('position'), uv = geometry.getAttribute('uv');
  const colors = new Float32Array(positions.count * 3), color = new THREE.Color();
  const gravel = new THREE.Color(0xcbd3c8), moss = new THREE.Color(0x687b50), silt = new THREE.Color(0xb2bb9d);
  const stretch = (p: number) => Math.sign(p) * (5 * Math.abs(p) + 2043 * Math.abs(p) ** 3);
  for (let i = 0; i < positions.count; i++) {
    const x = stretch(positions.getX(i)), z = stretch(positions.getZ(i)), y = heightAt(x, z);
    positions.setXYZ(i, x, y, z);
    uv.setXY(i, x * 0.6, z * 0.6);
    color.copy(gravel).lerp(moss, smooth(y - waterLevel, 0.5, 2.4) * (0.45 + noise(x * 0.09, z * 0.09) * 0.55));
    color.lerp(silt, 1 - smooth(y, waterLevel - 8, waterLevel - 2));
    color.multiplyScalar(0.72 + noise(x * 0.18, z * 0.18) * 0.28);
    colors.set([color.r, color.g, color.b], i * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  return geometry;
}

export function alpineTexture(world: WorldBuilder, name: string, wrap: THREE.Wrapping = THREE.MirroredRepeatWrapping) {
  const map = world.own(new THREE.TextureLoader().load(`/textures/${name}.webp`));
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = wrap;
  map.anisotropy = 4;
  return map;
}

export function spruce(world: WorldBuilder) {
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
  const map = alpineTexture(world, 'alpine-spruce', THREE.ClampToEdgeWrapping), barkMap = alpineTexture(world, 'alpine-bark', THREE.RepeatWrapping);
  const bark = world.own(new THREE.MeshStandardMaterial({ map: barkMap, bumpMap: barkMap, bumpScale: 0.018, roughness: 1 }));
  const leaf = world.own(new THREE.MeshStandardMaterial({ map, color: 0xffffff, side: THREE.DoubleSide, alphaTest: 0.4, alphaToCoverage: true, roughness: 0.92 }));
  return [{ geometry: trunk, material: bark }, { geometry: foliage, material: leaf }];
}

export function scatter(world: WorldBuilder, geometry: THREE.BufferGeometry, material: THREE.Material, transforms: THREE.Matrix4[], name: string) {
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

export function alpineGroundMaterial(world: WorldBuilder, waterLevel: number) {
  const map = alpineTexture(world, 'alpine-pebbles');
  const moss = alpineTexture(world, 'prairie-ground');
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
float cover = smoothstep(${waterLevel + 0.4}, ${waterLevel + 1.8}, p.y) * smoothstep(0.45, 0.85, vLakeNormal.y);
diffuseColor.rgb *= mix(gravel, forest, cover);
`);
  };
  return ground;
}

export function alpineCliffMaterial(world: WorldBuilder) {
  const stone = alpineTexture(world, 'alpine-limestone');
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
  return cliff;
}
