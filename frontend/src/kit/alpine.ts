import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { terrainNoise as noise } from './mountains';
import type { WorldBuilder } from './world';
import { SpatialInstances } from './instances';

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
  const bark = world.own(new THREE.MeshStandardMaterial({ color: 0x746350, roughness: 1 }));
  const leaf = world.own(new THREE.MeshStandardMaterial({ map: coverage, color: 0x527961, side: THREE.DoubleSide, alphaTest: 0.4, alphaToCoverage: true, roughness: 0.92 }));
  leaf.onBeforeCompile = shader => { shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>', 'diffuseColor.a *= texture2D(map, vMapUv).a;'); };
  return [{ geometry: trunk, material: bark }, { geometry: foliage, material: leaf }];
}

export function scatter(world: WorldBuilder, geometry: THREE.BufferGeometry, material: THREE.Material, transforms: THREE.Matrix4[], name: string) {
  const instances = world.own(new SpatialInstances([{ geometry, material, padding: 1 }]));
  instances.group.name = name;
  instances.scatter(Float32Array.from(transforms.flatMap(matrix => matrix.elements)));
  world.add(instances.group);
}

export function alpineMaterial(world: WorldBuilder) {
  return world.own(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 }));
}
