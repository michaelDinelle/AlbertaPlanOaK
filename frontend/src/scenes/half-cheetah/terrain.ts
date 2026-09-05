import * as THREE from 'three';
import { terrainNoise as noise } from '@kit/mountains';
import type { WorldBuilder } from '@kit/world';
import type { ViewCoverage } from '../jelly-bean-world/view';

const smooth = THREE.MathUtils.smoothstep;

export function prairieRidge(x: number, z: number) {
  const row = Math.floor(Math.abs(z) / 190);
  const center = 75 + row * 190 + (noise(x * 0.005, row * 2.1) - 0.5) * 90 + (noise(x * 0.017 + 50, row * 3.7) - 0.5) * 28;
  const distance = Math.abs(Math.abs(z) - center);
  const amplitude = 5 + 17 * noise(x * 0.013 + row * 10, row * 4.9 + 0.8) ** 2;
  const width = Math.min(18 + 34 * noise(x * 0.015 + 11, row * 5.7), 0.85 * Math.min(center - row * 190, (row + 1) * 190 - center));
  const profile = 1 - smooth(distance, width * 0.12, width);
  const exposure = smooth(distance, width * 0.18, width * 0.42) * (1 - smooth(distance, width * 0.82, width)) * smooth(amplitude, 2, 6);
  const drainage = noise(x * 0.09 + noise(x * 0.025, row) * 4, z * 0.035 + row * 2);
  const flutes = smooth(drainage, 0.35, 0.8);
  return { center, width, exposure, height: amplitude * profile * (1 - exposure * flutes * 0.5) };
}

export function prairieHeight(x: number, z: number): number {
  const lane = smooth(Math.abs(z), 1, 18);
  const roll = 0.4 + noise(x * 0.018, z * 0.018) * 3 + noise(x * 0.05 + 8, z * 0.05) * 0.6;
  const foothills = smooth(-z, 180, 650) * (8 + 56 * noise(x * 0.005 + 9, z * 0.009) + 18 * noise(x * 0.015, z * 0.016));
  return lane * (roll + prairieRidge(x, z).height + foothills);
}

export function prairieGroundHeight(x: number, z: number): number {
  const gx = Math.floor(x / 2) * 2, gz = Math.floor(z / 2) * 2;
  const u = (x - gx) / 2, v = (z - gz) / 2;
  const h10 = prairieHeight(gx + 2, gz), h01 = prairieHeight(gx, gz + 2);
  if (u + v <= 1) return prairieHeight(gx, gz) * (1 - u - v) + h10 * u + h01 * v;
  return prairieHeight(gx + 2, gz + 2) * (u + v - 1) + h01 * (1 - u) + h10 * (1 - v);
}

export function prairiePlacement(x: number, z: number, radius: number, upright = 1) {
  const normal = new THREE.Vector3((prairieGroundHeight(x - radius, z) - prairieGroundHeight(x + radius, z)) * upright, radius * 2, (prairieGroundHeight(x, z - radius) - prairieGroundHeight(x, z + radius)) * upright).normalize();
  let height = prairieGroundHeight(x, z);
  for (let i = 0; i < 8; i++) {
    const dx = Math.cos(i * Math.PI / 4) * radius, dz = Math.sin(i * Math.PI / 4) * radius;
    height = Math.min(height, prairieGroundHeight(x + dx, z + dz) + (normal.x * dx + normal.z * dz) / normal.y);
  }
  return { height: height - 0.06 * radius, normal };
}

function tileGeometry(tx: number, tz: number, segments: number) {
  const geometry = new THREE.PlaneGeometry(128, 128, segments, segments);
  geometry.rotateX(-Math.PI / 2);
  const positions = geometry.getAttribute('position'), uv = geometry.getAttribute('uv');
  const exposure = new Float32Array(positions.count);
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i) + tx * 128, z = positions.getZ(i) + tz * 128;
    positions.setY(i, prairieHeight(x, z));
    exposure[i] = prairieRidge(x, z).exposure;
    uv.setXY(i, x * 0.5, z * 0.5);
  }
  geometry.setAttribute('exposure', new THREE.BufferAttribute(exposure, 1));
  geometry.computeVertexNormals();
  const normals = geometry.getAttribute('normal');
  const normal = new THREE.Vector3();
  for (let i = 0; i < positions.count; i++) {
    if (Math.abs(positions.getX(i)) < 63.9 && Math.abs(positions.getZ(i)) < 63.9) continue;
    const x = positions.getX(i) + tx * 128, z = positions.getZ(i) + tz * 128;
    normal.set(prairieHeight(x - 1, z) - prairieHeight(x + 1, z), 2, prairieHeight(x, z - 1) - prairieHeight(x, z + 1)).normalize();
    normals.setXYZ(i, normal.x, normal.y, normal.z);
  }
  const edge: number[] = [];
  const row = segments + 1;
  for (let i = 0; i < segments; i++) edge.push(i);
  for (let i = 0; i < segments; i++) edge.push(i * row + segments);
  for (let i = segments; i > 0; i--) edge.push(segments * row + i);
  for (let i = segments; i > 0; i--) edge.push(i * row);
  const skirted = new THREE.BufferGeometry();
  for (const name of ['position', 'normal', 'uv', 'exposure']) {
    const source = geometry.getAttribute(name), data = new Float32Array((positions.count + edge.length) * source.itemSize);
    data.set(source.array);
    for (let i = 0; i < edge.length; i++) for (let k = 0; k < source.itemSize; k++) {
      data[(positions.count + i) * source.itemSize + k] = source.array[edge[i]! * source.itemSize + k]! - (name === 'position' && k === 1 ? 32 : 0);
    }
    skirted.setAttribute(name, new THREE.BufferAttribute(data, source.itemSize));
  }
  const indices = Array.from(geometry.index!.array);
  for (let i = 0; i < edge.length; i++) {
    const j = (i + 1) % edge.length, a = edge[i]!, b = edge[j]!, c = positions.count + i, d = positions.count + j;
    indices.push(a, b, c, b, d, c);
  }
  skirted.setIndex(indices);
  skirted.computeBoundingSphere();
  geometry.dispose();
  return skirted;
}

export function makePrairieTerrain(world: WorldBuilder) {
  const map = world.own(new THREE.TextureLoader().load('/textures/prairie-ground.webp'));
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = THREE.MirroredRepeatWrapping;
  map.anisotropy = 4;
  const rock = world.own(new THREE.TextureLoader().load('/textures/prairie-sandstone.webp'));
  rock.colorSpace = THREE.SRGBColorSpace;
  rock.wrapS = rock.wrapT = THREE.MirroredRepeatWrapping;
  rock.anisotropy = 4;
  const material = world.own(new THREE.MeshStandardMaterial({ map, bumpMap: map, bumpScale: 0.025, roughness: 1 }));
  material.onBeforeCompile = shader => {
    shader.uniforms.uPrairieRock = { value: rock };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float exposure; varying float vExposure; varying vec3 vPrairiePosition;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvExposure = exposure; vPrairiePosition = (modelMatrix * vec4(position, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform sampler2D uPrairieRock; varying float vExposure; varying vec3 vPrairiePosition;')
      .replace('#include <map_fragment>', `
vec3 p = vPrairiePosition;
vec3 grass = texture2D(map, vMapUv).rgb;
grass = mix(grass, texture2D(map, vMapUv.yx * 0.173).rgb, 0.18) * vec3(1.18, 1.55, 1.06);
vec3 stone = mix(texture2D(uPrairieRock, p.xy * 0.35).rgb, texture2D(uPrairieRock, p.zy * 0.35).rgb, 0.35);
float band = sin(p.y * 2.1 + 0.1 * sin(p.x * 0.13));
stone *= 0.94 + 0.06 * band;
float track = 1.0 - smoothstep(0.35, 1.1, abs(p.z));
diffuseColor.rgb *= mix(mix(grass, stone, vExposure), vec3(0.24, 0.17, 0.10), track * 0.55);
`);
  };
  const batches = ([[64, 9], [8, 33]] as const).map(([segments, side]) => {
    const count = side ** 2, vertices = (segments + 1) ** 2 + segments * 4, indices = segments ** 2 * 6 + segments * 24;
    const mesh = world.own(new THREE.BatchedMesh(count, count * vertices, count * indices, material));
    mesh.name = 'prairieTerrain';
    mesh.receiveShadow = true;
    mesh.frustumCulled = mesh.perObjectFrustumCulled = mesh.sortObjects = false;
    world.add(mesh);
    return { mesh, segments, side, slots: new Map<number, { key: string; geometry: number; instance: number }>() };
  });
  const bounds = new THREE.Box3(), matrix = new THREE.Matrix4();
  const previous = new THREE.Matrix4();
  return {
    rock, material,
    setView(coverage: ViewCoverage, camera: THREE.Camera) {
      matrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      if (matrix.equals(previous)) return;
      previous.copy(matrix);
      const cx = Math.round(camera.position.x / 128), cz = Math.round(camera.position.z / 128);
      for (const batch of batches) for (const slot of batch.slots.values()) batch.mesh.setVisibleAt(slot.instance, false);
      for (let z = cz - 16; z <= cz + 16; z++) for (let x = cx - 16; x <= cx + 16; x++) {
        bounds.min.set(x * 128 - 64, -32, z * 128 - 64);
        bounds.max.set(x * 128 + 64, 110, z * 128 + 64);
        if (!coverage.frustum.intersectsBox(bounds)) continue;
        const batch = batches[Math.max(Math.abs(x - cx), Math.abs(z - cz)) <= 4 ? 0 : 1]!;
        const id = THREE.MathUtils.euclideanModulo(x, batch.side) + THREE.MathUtils.euclideanModulo(z, batch.side) * batch.side;
        const key = `${x},${z}`;
        let slot = batch.slots.get(id);
        if (slot?.key !== key) {
          const geometry = tileGeometry(x, z, batch.segments).translate(x * 128, 0, z * 128);
          if (slot) {
            batch.mesh.setGeometryAt(slot.geometry, geometry);
            slot.key = key;
          } else {
            const geometryId = batch.mesh.addGeometry(geometry);
            slot = { key, geometry: geometryId, instance: batch.mesh.addInstance(geometryId) };
            batch.slots.set(id, slot);
          }
          geometry.dispose();
        }
        batch.mesh.setVisibleAt(slot!.instance, true);
      }
    },
  };
}
