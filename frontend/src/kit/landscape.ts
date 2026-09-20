import * as THREE from 'three';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { buildAurora } from '@kit/aurora';
import { makeMountains } from '@kit/mountains';
import { leafSurface } from '@kit/leaves';
import { oakMaterial, oakUniforms } from '@kit/materials';
import { rngAt } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import type { Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import type { VegetationHandle } from '@kit/vegetation';
import { makeWorld, type WorldCore } from '@kit/world';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '@kit/view';
import { makePrairieTerrain, prairieGroundHeight, prairiePlacement, prairieRidge } from './prairie';

export { prairieHeight } from './prairie';

export type Landscape = ReturnType<typeof makeLandscape>;

export function makeLandscape(scene: THREE.Scene, name: string, palette: Palette, mood: Parameters<typeof skyFromPalette>[1], options: {
  sky?: Parameters<typeof skyFromPalette>[2];
  light?: Parameters<typeof makeLightRig>[1];
  fogExtent?: number;
  fogDensity?: number;
} = {}) {
  const world = makeWorld(name);
  const skyRadius = options.sky?.radius ?? 1;
  const sky = world.own(world.time(skyFromPalette(palette, mood, { ...options.sky, radius: skyRadius })));
  world.add(sky.mesh);
  const rig = world.own(makeLightRig(palette, { sunDir: sky.sunDirection, pointCount: 2, ...options.light }));
  world.add(rig.group);
  const fog = applyFog(scene, palette, options.fogExtent ?? 220);
  const surfaces: ReturnType<typeof makeContinuousSurface>[] = [];
  const banks: { bank: ReturnType<typeof makeSceneryBank>; camera: boolean }[] = [];
  return {
    world,
    sky,
    rig,
    fog,
    surface(options: Parameters<typeof makeContinuousSurface>[0]) {
      const surface = world.own(world.time(makeContinuousSurface(options)));
      world.add(surface.mesh);
      surfaces.push(surface);
      return surface;
    },
    scatter(options: Parameters<typeof makeSceneryBank>[0], camera = false) {
      world.own(options.geometry);
      world.own(options.material);
      const motion = oakUniforms(options.material);
      if (motion) world.time(motion);
      const bank = world.own(makeSceneryBank(options));
      world.add(bank.mesh);
      banks.push({ bank, camera });
      return bank;
    },
    setView(coverage: ViewCoverage, camera: THREE.Camera, ...others: { setView(coverage: ViewCoverage): void }[]) {
      for (const surface of surfaces) surface.setView(coverage);
      for (const entry of banks) entry.bank.setView(entry.camera ? { ...coverage, x: camera.position.x, z: camera.position.z } : coverage);
      for (const other of others) other.setView(coverage);
      sky.mesh.position.copy(camera.position);
      sky.mesh.scale.setScalar(coverage.far * 0.85 / skyRadius);
      fog.density = options.fogDensity ?? 2.8 / coverage.far;
    },
  };
}

export function prairieGrass(blades = 6) {
  const positions: number[] = [], colors: number[] = [];
  const root = new THREE.Color(0x667044), tip = new THREE.Color(0xc2b875), color = new THREE.Color();
  for (let blade = 0; blade < blades; blade++) {
    const angle = blade * 2.3999632, radius = 0.07 * Math.sqrt(blade);
    const height = 0.16 + 0.18 * (0.5 + 0.5 * Math.sin(blade * 4.1));
    const vertices: THREE.Vector3[] = [];
    for (let level = 0; level < 4; level++) {
      const t = level / 3, bend = t * t * 0.19;
      const x = Math.cos(angle) * (radius + bend), z = Math.sin(angle) * (radius + bend);
      const width = (1 - t) * 0.011;
      for (const side of level === 3 ? [0] : [-1, 1]) vertices.push(new THREE.Vector3(x + Math.sin(angle) * width * side, height * t, z - Math.cos(angle) * width * side));
    }
    for (const i of [0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4, 4, 5, 6]) {
      const p = vertices[i]!;
      positions.push(p.x, p.y, p.z);
      color.copy(root).lerp(tip, p.y / height);
      colors.push(color.r, color.g, color.b);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  const material = oakMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 1, wind: { amplitude: 0.014, frequency: 0.6, height: 0.46 }, toonSteps: 6, toonSoftness: 1, toonFloor: 0.75 });
  return { geometry, material };
}

export function aspenTree(world: ReturnType<typeof makeWorld>, foliage = ['#8ca357', '#c6ce8a', '#a6bd70', '#d1d28b']) {
  const palette = foliage.map(color => new THREE.Color(color));
  const pieces: THREE.BufferGeometry[] = [];
  const add = (geometry: THREE.BufferGeometry, color: THREE.Color) => {
    const count = geometry.getAttribute('position').count, colors = new Float32Array(count * 3);
    if (!geometry.hasAttribute('aLeaf')) geometry.setAttribute('aLeaf', new THREE.Float32BufferAttribute(new Float32Array(count * 2).fill(-2), 2));
    for (let i = 0; i < count; i++) color.toArray(colors, i * 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3)); pieces.push(geometry);
  };
  add(new THREE.CylinderGeometry(0.018, 0.11, 3.7, 10).translate(0, 1.82, 0), new THREE.Color(0xf0e9d5));
  for (let i = 0; i < 18; i++) {
    const angle = i * 2.39996, y = 1.15 + i * 0.148, spread = Math.sin((i + 2) / 21 * Math.PI) * 0.86;
    const end = new THREE.Vector3(Math.cos(angle) * spread, y + 0.38, Math.sin(angle) * spread);
    const start = new THREE.Vector3(0, y, 0), direction = end.clone().sub(start);
    add(new THREE.CylinderGeometry(0.004, 0.019, direction.length(), 5).applyQuaternion(new THREE.Quaternion().setFromUnitVectors(THREE.Object3D.DEFAULT_UP, direction.clone().normalize())).translate(...start.add(end).multiplyScalar(0.5).toArray()), new THREE.Color(0xd1cdbb));
    if (!palette.length) {
      const snow = new THREE.CapsuleGeometry(0.048, Math.max(0, direction.length() - 0.08), 4, 10);
      snow.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(THREE.Object3D.DEFAULT_UP, direction.clone().normalize())).translate(end.x * 0.5, y + direction.y * 0.5 + 0.045, end.z * 0.5);
      add(snow, new THREE.Color(0xf3f6ff));
      for (let fork = 0; fork < 4; fork++) {
        const t = 0.45 + fork * 0.15, base = new THREE.Vector3(end.x * t, y + direction.y * t, end.z * t), tip = base.clone().add(new THREE.Vector3(Math.cos(angle + fork - 1.5) * 0.24, 0.2 + fork * 0.035, Math.sin(angle + fork - 1.5) * 0.24));
        const branch = tip.clone().sub(base);
        add(new THREE.CylinderGeometry(0.002, 0.007, branch.length(), 6).applyQuaternion(new THREE.Quaternion().setFromUnitVectors(THREE.Object3D.DEFAULT_UP, branch.normalize())).translate(...base.add(tip).multiplyScalar(0.5).toArray()), new THREE.Color(0xa8a5ac));
      }
    }
    const count = palette.length ? 38 : 0;
    for (let j = 0; j < count; j++) {
      const t = j / count, azimuth = j * 2.39996 + angle, radial = Math.sqrt(t) * 0.52;
      const orientation = new THREE.Quaternion().setFromEuler(new THREE.Euler(azimuth * 0.7, azimuth, Math.sin(j * 1.7) * 0.8, 'ZYX'));
      const center = new THREE.Vector3(end.x + Math.cos(azimuth) * radial, end.y + Math.sin(j * 3.7) * radial * 0.9, end.z + Math.sin(azimuth) * radial);
      const leaf = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([0, -0.1425, 0.017, 0.111, 0, -0.009, 0, 0.1425, 0.017, -0.111, 0, -0.009], 3)).setIndex([0, 1, 2, 0, 2, 3]);
      leaf.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(8), 2));
      leaf.setAttribute('aLeaf', new THREE.Float32BufferAttribute([0, -1.5, 1.5, 0, 0, 1.5, -1.5, 0], 2));
      leaf.applyQuaternion(orientation).translate(center.x, center.y, center.z);
      leaf.computeVertexNormals(); add(leaf, palette[(i + j) % palette.length]!);
    }
  }
  const geometry = mergeGeometries(pieces); pieces.forEach(piece => piece.dispose());
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, side: THREE.DoubleSide });
  const depth = world.own(new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking }));
  leafSurface(material, depth);
  return { geometry, material, depth };
}

export function fieldstoneGeometry(segments = 20, rings = 12) {
  const geometry = new THREE.SphereGeometry(0.4, segments, rings);
  const position = geometry.getAttribute('position'), colors = new Float32Array(position.count * 3);
  const stone = new THREE.Color(0xb8b3ae), lichen = new THREE.Color(0xa9ad86), color = new THREE.Color();
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i), y = position.getY(i), z = position.getZ(i);
    const wear = 0.93 + 0.1 * Math.sin(x * 13 + y * 9) * Math.cos(z * 11 - x * 5);
    position.setXYZ(i, x * wear, (y + 0.4) * 0.6 * wear, z * wear * 0.85);
    color.copy(stone).lerp(lichen, THREE.MathUtils.smoothstep(y, 0.1, 0.4) * (0.5 + 0.5 * Math.sin(x * 28 + z * 17)));
    color.multiplyScalar(THREE.MathUtils.lerp(0.48, 1, THREE.MathUtils.smoothstep(y, -0.4, -0.06)));
    colors.set([color.r, color.g, color.b], i * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  return geometry;
}

export function fieldstones() {
  return { geometry: fieldstoneGeometry(), material: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 }) };
}

function hoodooGeometry(): THREE.BufferGeometry {
  const height = (x: number, z: number) => 1.65 * Math.exp(-x * x / 8 - z * z / 2.5)
    + 2.3 * Math.exp(-Math.pow((x + 1.1 + z * 0.22) / 0.9, 6) - Math.pow((z + 0.2) / 0.8, 6))
    + 1.3 * Math.exp(-Math.pow((x - 1.4) / 0.8, 4) - Math.pow((z - 0.15) / 0.7, 4));
  const bluff = new THREE.PlaneGeometry(9, 5, 36, 24);
  bluff.rotateX(-Math.PI / 2);
  const position = bluff.getAttribute('position');
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i), z = position.getZ(i);
    const erosion = 0.06 * Math.sin(x * 15 + z * 3) * Math.sin(z * 11);
    position.setY(i, height(x, z) * (1 + erosion) - 0.16);
  }
  bluff.computeVertexNormals();
  const parts: THREE.BufferGeometry[] = [bluff.toNonIndexed()];
  bluff.dispose();
  for (const [x, z, width] of [[-1.1, -0.2, 1.9]] as const) {
    const cap = new THREE.IcosahedronGeometry(1, 1);
    const vertices = cap.getAttribute('position');
    for (let i = 0; i < vertices.count; i++) {
      const px = vertices.getX(i), py = vertices.getY(i), pz = vertices.getZ(i);
      vertices.setXYZ(i, px * width * 0.62 * (1 + 0.15 * Math.sin(pz * 5 + x)), py * 0.22 + px * 0.06 + pz * 0.045, pz * 0.88 * (1 + 0.13 * Math.cos(px * 4 + x)));
    }
    cap.computeVertexNormals();
    cap.translate(x, height(x, z) - 0.22, z);
    parts.push(cap);
  }
  for (const [part, geometry] of parts.entries()) {
    const vertices = geometry.getAttribute('position');
    const colors = new Float32Array(vertices.count * 3);
    const sand = new THREE.Color(0xdec9a9), shale = new THREE.Color(0x796656), color = new THREE.Color();
    for (let i = 0; i < vertices.count; i++) {
      const y = vertices.getY(i);
      color.copy(sand).lerp(shale, part > 0 ? 0.65 : y < 1.1 ? 0.48 + 0.14 * Math.sin(y * 18) : 0.08);
      colors.set([color.r, color.g, color.b], i * 3);
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }
  const merged = mergeGeometries(parts), geometry = mergeVertices(merged);
  merged.dispose();
  for (const part of parts) part.dispose();
  return geometry;
}

function makeHoodoos(world: ReturnType<typeof makeWorld>, terrainMaterial: THREE.MeshStandardMaterial, seed: number) {
  const source = world.own(hoodooGeometry());
  const material = world.own(terrainMaterial.clone());
  material.vertexColors = true;
  material.onBeforeCompile = terrainMaterial.onBeforeCompile;
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
  const hoodoos = makeHoodoos(world, terrain.material, ctx.seed);
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
    ...aspenTree(world),
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

export const WORLD_NOISE = `
float worldHash(vec2 p, uint seed) {
  uint h = seed ^ 0x9e3779b9u;
  h = (h ^ uint(p.x)) * 0x85ebca6bu;
  h ^= h >> 13u;
  h = (h ^ uint(p.y)) * 0xc2b2ae35u;
  h ^= h >> 16u;
  h *= 0x27d4eb2fu;
  h ^= h >> 15u;
  return float(h) / 4294967296.0;
}
float worldNoise(vec2 p, float period, uint seed) {
  vec2 cell = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  vec2 a = mod(cell, period);
  vec2 b = mod(cell + 1.0, period);
  float value = mix(mix(worldHash(a, seed), worldHash(vec2(b.x, a.y), seed), f.x), mix(worldHash(vec2(a.x, b.y), seed), worldHash(b, seed), f.x), f.y);
  return value;
}
`;

export function makeSceneTerrain(
  options: Parameters<typeof makeContinuousSurface>[0],
  sampleHeight: (x: number, z: number) => number,
  materialOptions: Parameters<typeof oakMaterial>[0],
) {
  const surface = makeContinuousSurface(options);
  const finish = oakMaterial(materialOptions);
  const compile = surface.material.onBeforeCompile;
  const key = surface.material.customProgramCacheKey();
  surface.material.roughness = finish.roughness;
  surface.material.onBeforeCompile = (shader, renderer) => {
    finish.onBeforeCompile.call(finish, shader, renderer);
    compile.call(surface.material, shader, renderer);
    shader.vertexShader = shader.vertexShader.replace('float landHash', `${WORLD_NOISE}\nfloat landHash`);
    shader.fragmentShader = shader.fragmentShader.replace('float landHash', `${WORLD_NOISE}\nfloat landHash`);
  };
  surface.material.customProgramCacheKey = () => `${key}|${finish.customProgramCacheKey()}|world-noise`;
  return {
    ...surface,
    sampleHeight,
    setTime(t: number) {
      surface.setTime(t);
      finish.oak.setTime(t);
    },
    dispose() {
      finish.dispose();
      surface.dispose();
    },
  };
}

export function extendWater(
  mesh: THREE.Mesh,
  coverage: ViewCoverage,
  width: number,
  depth: number,
  bedDepth: string,
): void {
  const material = mesh.material;
  if (!(material instanceof THREE.ShaderMaterial)) throw new Error('water shader required');
  if (!material.userData.worldWater) {
    const waves = material.vertexShader.slice(material.vertexShader.indexOf('const int OAK_WAVES'), material.vertexShader.indexOf('float oakWaveHeight'));
    material.vertexShader = material.vertexShader.replace('world4.y += h;', '');
    material.fragmentShader = material.fragmentShader
      .replace('uniform float uTime;', 'uniform float uTime;\nuniform float uAmplitude;')
      .replace('void main() {', `${waves}
vec3 waterWave(vec2 p, float footprint) {
  vec3 wave = vec3(0.0);
  for (int i = 0; i < OAK_WAVES; i++) {
    vec4 w = OAK_WAVE[i];
    float k = w.z * uWaveScale;
    float phase = dot(p, w.xy) * k + uTime * w.w * uSpeed;
    float amplitude = OAK_WAVE_AMP[i] * uAmplitude / (1.0 + footprint * footprint * k * k);
    wave += vec3(cos(phase) * amplitude * k * w.xy, sin(phase) * amplitude);
  }
  return wave;
}
void main() {`)
      .replace('vec3 n = normalize( vNormalW );', `float footprint = length(fwidth(vWorld.xz));
  vec3 wave = waterWave(vWorld.xz, footprint);
  vec3 n = normalize(vec3(-wave.x, 1.0, -wave.y));`)
      .replace('-vDepth / max( uDepthFalloff, 0.001 )', `-max(0.0, ${bedDepth}) / max(uDepthFalloff, 0.001)`)
      .replace('vHeight * 6.0', 'wave.z * 6.0')
      .replace('oakCaustic( vWorld.xz * uWaveScale, uTime * uSpeed );', 'oakCaustic(vWorld.xz * uWaveScale, uTime * uSpeed) * (1.0 - smoothstep(0.12, 0.7, footprint * uWaveScale));')
      .replace('96.0 );', '36.0 );')
      .replace('spec * uSpecular', 'spec * uSpecular * 0.4 / (1.0 + footprint * footprint * 4.0)');
    material.userData.worldWater = true;
    material.needsUpdate = true;
  }
  mesh.position.x = coverage.x;
  mesh.position.z = coverage.z;
  mesh.scale.set(coverage.radius * 2 / width, 1, coverage.radius * 2 / depth);
  mesh.frustumCulled = false;
}

export function extendVegetation(
  source: VegetationHandle,
  options: {
    name: string; seed: number; tileSize: number; perTile: number;
    scale: readonly [number, number]; stretch: readonly [number, number];
    heightAt: (x: number, z: number) => number; exclude: (x: number, z: number, height: number) => boolean;
    bark?: THREE.ColorRepresentation;
  },
) {
  const count = Math.ceil(options.perTile / source.models.length);
  for (const { geometry, material } of source.models) {
    if (!/^oak(Broadleaf|Conifer|BareTree)$/.test(geometry.name) || geometry.userData.treeTint) continue;
    if (!(material instanceof THREE.MeshStandardMaterial)) throw new Error('tree surface material required');
    const positions = geometry.getAttribute('position');
    const colors = geometry.getAttribute('color');
    const bark = new THREE.Color(options.bark ?? 0x7e6852);
    const leaf = material.color.clone();
    for (let i = 0; i < positions.count; i += 3) {
      let wood = true;
      for (let j = i; j < i + 3; j++) wood &&= Math.hypot(positions.getX(j), positions.getZ(j)) <= 0.101 && positions.getY(j) <= 1.151;
      const tint = wood || geometry.name === 'oakBareTree' ? bark : leaf;
      for (let j = i; j < i + 3; j++) {
        const shade = colors.getX(j);
        colors.setXYZ(j, tint.r * shade, tint.g * shade, tint.b * shade);
      }
    }
    colors.needsUpdate = true;
    material.color.set(0xffffff);
    geometry.userData.treeTint = true;
  }
  const banks = source.models.map((model, variant) => makeSceneryBank({
    name: `${options.name}${variant}`, seed: options.seed + variant,
    tileSize: options.tileSize, perTile: count,
    geometry: model.geometry, material: model.material,
    colorA: 0xd8d8d8, colorB: 0xffffff, castShadow: model.castShadow,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < count; i++) {
        const x = (tx + rng()) * options.tileSize;
        const z = (tz + rng()) * options.tileSize;
        const height = options.heightAt(x, z);
        if (options.exclude(x, z, height)) continue;
        write(x, height, z, rng() * Math.PI * 2,
          options.scale[0] + rng() * (options.scale[1] - options.scale[0]),
          options.stretch[0] + rng() * (options.stretch[1] - options.stretch[0]), rng());
      }
    },
  }));
  return {
    meshes: banks.map((bank) => bank.mesh),
    staticInstances: banks.flatMap(bank => bank.staticInstances),
    setView(coverage: ViewCoverage) {
      for (const bank of banks) bank.setView(coverage);
    },
    dispose() {
      for (const bank of banks) bank.dispose();
    },
  };
}
