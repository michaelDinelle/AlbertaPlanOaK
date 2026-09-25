import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { applyFog, makeLightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms } from '@kit/materials';
import { skyFromPalette } from '@kit/sky';
import type { Palette } from '@kit/tokens';
import type { VegetationHandle } from '@kit/vegetation';
import { makeWorld } from '@kit/world';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '@kit/view';
import { makeAmbientParticles, type AmbientParticlesOptions } from '@kit/particles';

export type Landscape = ReturnType<typeof makeLandscape>;

export function makeLandscape(scene: THREE.Scene, name: string, palette: Palette, mood: Parameters<typeof skyFromPalette>[1], options: {
  sky?: Parameters<typeof skyFromPalette>[2];
  light?: Parameters<typeof makeLightRig>[1];
  fogExtent?: number;
  fogDensity?: number;
  weather?: Omit<AmbientParticlesOptions, 'name'>;
} = {}) {
  const world = makeWorld(name);
  const skyRadius = options.sky?.radius ?? 1;
  const sky = world.own(world.time(skyFromPalette(palette, mood, { ...options.sky, radius: skyRadius })));
  world.add(sky.mesh);
  const rig = world.own(makeLightRig(palette, { sunDir: sky.sunDirection, pointCount: 2, ...options.light }));
  world.add(rig.group);
  const fog = applyFog(scene, palette, options.fogExtent ?? 220);
  if (options.weather) world.add(world.time(world.own(makeAmbientParticles({ ...options.weather, name: `${name}Weather` }))).points);
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
      sky.fit(coverage.far);
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

export function aspenTree() {
  const pieces: THREE.BufferGeometry[] = [];
  const add = (geometry: THREE.BufferGeometry, color: THREE.Color) => {
    const count = geometry.getAttribute('position').count, colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) color.toArray(colors, i * 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3)); pieces.push(geometry);
  };
  add(new THREE.CylinderGeometry(0.018, 0.11, 3.7, 10).translate(0, 1.82, 0), new THREE.Color(0xf0e9d5));
  for (let i = 0; i < 18; i++) {
    const angle = i * 2.39996, y = 1.15 + i * 0.148, spread = Math.sin((i + 2) / 21 * Math.PI) * 0.86;
    const end = new THREE.Vector3(Math.cos(angle) * spread, y + 0.38, Math.sin(angle) * spread);
    const start = new THREE.Vector3(0, y, 0), direction = end.clone().sub(start);
    add(new THREE.CylinderGeometry(0.004, 0.019, direction.length(), 5).applyQuaternion(new THREE.Quaternion().setFromUnitVectors(THREE.Object3D.DEFAULT_UP, direction.clone().normalize())).translate(...start.add(end).multiplyScalar(0.5).toArray()), new THREE.Color(0xd1cdbb));
    const snow = new THREE.CapsuleGeometry(0.048, Math.max(0, direction.length() - 0.08), 4, 10);
    snow.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(THREE.Object3D.DEFAULT_UP, direction.clone().normalize())).translate(end.x * 0.5, y + direction.y * 0.5 + 0.045, end.z * 0.5);
    add(snow, new THREE.Color(0xf3f6ff));
    for (let fork = 0; fork < 4; fork++) {
      const t = 0.45 + fork * 0.15, base = new THREE.Vector3(end.x * t, y + direction.y * t, end.z * t), tip = base.clone().add(new THREE.Vector3(Math.cos(angle + fork - 1.5) * 0.24, 0.2 + fork * 0.035, Math.sin(angle + fork - 1.5) * 0.24));
      const branch = tip.clone().sub(base);
      add(new THREE.CylinderGeometry(0.002, 0.007, branch.length(), 6).applyQuaternion(new THREE.Quaternion().setFromUnitVectors(THREE.Object3D.DEFAULT_UP, branch.normalize())).translate(...base.add(tip).multiplyScalar(0.5).toArray()), new THREE.Color(0xa8a5ac));
    }
  }
  const geometry = mergeGeometries(pieces); pieces.forEach(piece => piece.dispose());
  return { geometry, material: new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.92, side: THREE.DoubleSide }) };
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
  return { geometry: fieldstoneGeometry(), material: new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 1 }) };
}

export function makeSceneTerrain(
  options: Parameters<typeof makeContinuousSurface>[0],
  sampleHeight: (x: number, z: number) => number,
  finish: NonNullable<Parameters<typeof makeContinuousSurface>[0]['finish']>,
) {
  return { ...makeContinuousSurface({ ...options, finish }), sampleHeight };
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
    if (!(material instanceof THREE.MeshStandardNodeMaterial)) throw new Error('tree surface material required');
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
    setView(coverage: ViewCoverage) {
      for (const bank of banks) bank.setView(coverage);
    },
    dispose() {
      for (const bank of banks) bank.dispose();
    },
  };
}
