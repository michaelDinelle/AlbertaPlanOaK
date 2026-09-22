import * as THREE from 'three';
import type { WorldBuilder } from '../world';

export function lampBank(world: WorldBuilder, geometry: THREE.BufferGeometry, count: number, options: { emissive?: number; roughness?: number; metalness?: number; castShadow?: boolean } = {}) {
  const material = world.own(new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: options.emissive ?? 1.6, roughness: options.roughness ?? 0.32, metalness: options.metalness ?? 0 }));
  material.customProgramCacheKey = () => 'oakLampBank';
  material.onBeforeCompile = shader => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float aGlow;\nvarying float vGlow;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvGlow = aGlow;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vGlow;')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance *= vGlow;\n#ifdef USE_COLOR\ntotalEmissiveRadiance *= vColor.rgb;\n#endif');
  };
  const glow = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
  glow.setUsage(THREE.DynamicDrawUsage);
  world.own(geometry).setAttribute('aGlow', glow);
  const mesh = world.own(new THREE.InstancedMesh(geometry, material, count));
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3).fill(1), 3);
  mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  mesh.castShadow = options.castShadow ?? false;
  mesh.receiveShadow = true;
  const transform = new THREE.Object3D(), color = new THREE.Color();
  return {
    mesh,
    place(index: number, x: number, y: number, z: number, yaw = 0, scale = 1, stretch = 1) {
      transform.position.set(x, y, z);
      transform.rotation.set(0, yaw, 0);
      transform.scale.set(scale, scale * stretch, scale);
      transform.updateMatrix();
      mesh.setMatrixAt(index, transform.matrix);
    },
    commit() { mesh.instanceMatrix.needsUpdate = true; mesh.computeBoundingSphere(); },
    tint(index: number, value: THREE.ColorRepresentation, brightness = 1) {
      color.set(value).multiplyScalar(brightness).toArray(mesh.instanceColor!.array, index * 3);
      mesh.instanceColor!.needsUpdate = true;
    },
    glow(index: number, strength: number) { glow.array[index] = strength; glow.needsUpdate = true; },
    glowOf: (index: number) => glow.array[index] ?? 0,
  };
}

export type LampBank = ReturnType<typeof lampBank>;
