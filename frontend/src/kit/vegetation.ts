import * as THREE from 'three/webgpu';
import type { WorldBuilder } from './world';
import { oakMaterial, oakUniforms } from './materials';
import { SpatialInstances, type InstanceModel, instancedMesh } from './instances';
import { streamFor } from './rng';

export interface ScatterArea {
  x?: number;
  z?: number;
  halfX: number;
  halfZ: number;
  innerRadius?: number;
}

export type ExcludeFn = (x: number, z: number) => boolean;

export interface ScatterOptions {
  count: number;
  area: ScatterArea;
  seed: number;
  exclude?: ExcludeFn;
  heightAt?: (x: number, z: number) => number;
  maxSlope?: number;
  color: THREE.ColorRepresentation;
  hueJitter?: number;
  scale?: readonly [number, number];
  stretch?: readonly [number, number];
  wind?: number;
  windFrequency?: number;
  rimColor?: THREE.ColorRepresentation;
  rimStrength?: number;
  castShadow?: boolean;
  receiveShadow?: boolean;
  name?: string;
}

export interface VegetationHandle {
  group: THREE.Group;
  models: InstanceModel[];
  setTime(t: number): void;
  setWind(amplitude: number, frequency?: number): void;
  dispose(): void;
}

interface Placement {
  x: number;
  y: number;
  z: number;
  scale: number;
  stretch: number;
  yaw: number;
  jitter: number;
  variant: number;
}

export function scatterPlacements(opts: ScatterOptions, streamName: string, variants = 1): Placement[] {
  const wanted = Math.max(0, Math.round(opts.count));
  const rng = streamFor(opts.seed | 0, streamName);
  const cx = opts.area.x ?? 0;
  const cz = opts.area.z ?? 0;
  const inner = opts.area.innerRadius ?? 0;
  const [sMin, sMax] = opts.scale ?? [0.8, 1.25];
  const [tMin, tMax] = opts.stretch ?? [1, 1];
  const heightAt = opts.heightAt;
  const maxSlope = opts.maxSlope ?? 1;
  const out: Placement[] = [];

  for (let i = 0; i < wanted; i++) {
    let placed = false;
    for (let attempt = 0; attempt < 12 && !placed; attempt++) {
      const x = cx + (rng() * 2 - 1) * opts.area.halfX;
      const z = cz + (rng() * 2 - 1) * opts.area.halfZ;
      const s = sMin + rng() * (sMax - sMin);
      const t = tMin + rng() * (tMax - tMin);
      const yaw = rng() * Math.PI * 2;
      const jitter = rng();
      const variant = variants > 1 ? Math.min(variants - 1, (rng() * variants) | 0) : 0;
      if (inner > 0) {
        const dx = x - cx;
        const dz = z - cz;
        if (dx * dx + dz * dz < inner * inner) continue;
      }
      if (opts.exclude?.(x, z)) continue;
      const y = heightAt ? heightAt(x, z) : 0;
      if (heightAt && maxSlope < 1) {
        const e = 0.15;
        const gx = (heightAt(x + e, z) - heightAt(x - e, z)) / (2 * e);
        const gz = (heightAt(x, z + e) - heightAt(x, z - e)) / (2 * e);
        const slope = 1 - 1 / Math.sqrt(gx * gx + 1 + gz * gz);
        if (slope > maxSlope) continue;
      }
      out.push({ x, y, z, scale: s, stretch: t, yaw, jitter, variant });
      placed = true;
    }
  }
  return out;
}

interface ArchetypeSpec {
  geometry: THREE.BufferGeometry;
  windHeight: number;
  color: THREE.ColorRepresentation;
}

function buildScatter(
  opts: ScatterOptions,
  streamName: string,
  archetypes: ArchetypeSpec[],
  defaults: { wind?: number; windHeight?: number; castShadow: boolean },
): VegetationHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? `oakScatter:${streamName}`;
  const placements = scatterPlacements(opts, streamName, archetypes.length);
  const hueJitter = opts.hueJitter ?? 0.16;
  const windAmp = opts.wind ?? defaults.wind ?? 0;

  const instances = archetypes.map((spec, a) => {
    const material = oakMaterial({
      color: spec.color,
      roughness: 0.9,
      metalness: 0,
      flatShading: true,
      vertexColors: true,
      ...(opts.rimColor !== undefined ? { rimColor: opts.rimColor } : {}),
      rimStrength: opts.rimStrength ?? (opts.rimColor !== undefined ? 0.28 : 0),
      ...(windAmp > 0
        ? { wind: { amplitude: windAmp, frequency: opts.windFrequency ?? 1.3, height: spec.windHeight } }
        : {}),
      name: `${group.name}:${a}`,
    });
    const source = new SpatialInstances([{
      geometry: spec.geometry, material, padding: windAmp ? spec.windHeight : 0,
      castShadow: opts.castShadow ?? defaults.castShadow, receiveShadow: opts.receiveShadow ?? false,
    }]);
    const selected = placements.filter(p => p.variant === a), transform = new THREE.Object3D(), color = new THREE.Color();
    const matrices = new Float32Array(selected.length * 16), colors = new Float32Array(selected.length * 3);
    selected.forEach((p, i) => {
      transform.position.set(p.x, p.y, p.z);
      transform.rotation.set(0, p.yaw, 0);
      transform.scale.set(p.scale, p.scale * p.stretch, p.scale);
      transform.updateMatrix(); transform.matrix.toArray(matrices, i * 16);
      const k = 1 + (p.jitter - 0.5) * 2 * hueJitter;
      color.setRGB(k, k * (1 - hueJitter * 0.25 * (p.jitter - 0.5)), k * (1 + hueJitter * 0.2 * (p.jitter - 0.5))).toArray(colors, i * 3);
    });
    source.scatter(matrices, colors);
    group.add(source.group);
    return source;
  });
  const models = instances.flatMap(source => source.models);
  return {
    group, models,
    setTime(t) {
      for (const model of models) oakUniforms(model.material)?.setTime(t);
    },
    setWind(amplitude, frequency) {
      for (const model of models) oakUniforms(model.material)?.setWind(amplitude, frequency);
    },
    dispose() {
      for (const source of instances) source.dispose();
      for (const model of models) { model.geometry.dispose(); model.material.dispose(); }
      group.removeFromParent();
      group.clear();
    },
  };
}

function gradeByHeight(geometry: THREE.BufferGeometry, low = 0.55, high = 1.15): THREE.BufferGeometry {
  const pos = geometry.getAttribute('position');
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const span = maxY - minY;
  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const t = span > 1e-6 ? (pos.getY(i) - minY) / span : 0;
    const k = low + (high - low) * t;
    colors[i * 3] = k;
    colors[i * 3 + 1] = k;
    colors[i * 3 + 2] = k;
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return geometry;
}

export function grassClumpGeometry(blades = 3, height = 0.45, width = 0.06): THREE.BufferGeometry {
  const tris = blades * 2;
  const positions = new Float32Array(tris * 3 * 3);
  let o = 0;
  for (let b = 0; b < blades; b++) {
    const a = (b / blades) * Math.PI;
    const dx = Math.cos(a) * width;
    const dz = Math.sin(a) * width;
    const lean = ((b % 2 === 0 ? 1 : -1) * width) * 1.8;
    const h = height * (0.72 + 0.35 * ((b * 7919) % 11) / 11);
    const x0 = -dx;
    const z0 = -dz;
    const x1 = dx;
    const z1 = dz;
    const tipX = lean;
    const tipZ = lean * 0.4;
    const mid = 0.55;
    const push = (x: number, y: number, z: number): void => {
      positions[o++] = x;
      positions[o++] = y;
      positions[o++] = z;
    };
    push(x0, 0, z0);
    push(x1, 0, z1);
    push(x1 * 0.5 + tipX * mid, h * mid, z1 * 0.5 + tipZ * mid);
    push(x0, 0, z0);
    push(x1 * 0.5 + tipX * mid, h * mid, z1 * 0.5 + tipZ * mid);
    push(tipX, h, tipZ);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.computeVertexNormals();
  g.name = 'oakGrassClump';
  return gradeByHeight(g, 0.5, 1.2);
}

export function broadleafTreeGeometry(detail = 1): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const trunk = new THREE.CylinderGeometry(0.055, 0.1, 1.15, Math.max(4, 4 + detail * 2), 1);
  trunk.translate(0, 0.575, 0);
  parts.push(trunk);
  const blobs: readonly (readonly [number, number, number, number])[] = [
    [0, 1.45, 0, 0.55],
    [0.34, 1.2, 0.12, 0.36],
    [-0.28, 1.26, -0.18, 0.33],
    [0.06, 1.72, -0.22, 0.3],
  ];
  for (const [x, y, z, r] of blobs) {
    const blob = new THREE.IcosahedronGeometry(r, detail > 0 ? 1 : 0);
    blob.translate(x, y, z);
    parts.push(blob);
  }
  return finishTree(parts, 'oakBroadleaf');
}

export function coniferTreeGeometry(detail = 1): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const radial = Math.max(5, 5 + detail * 3);
  const trunk = new THREE.CylinderGeometry(0.045, 0.075, 0.5, radial, 1);
  trunk.translate(0, 0.25, 0);
  parts.push(trunk);
  const tiers: readonly (readonly [number, number, number])[] = [
    [0.42, 0.62, 0.35],
    [0.33, 0.58, 0.85],
    [0.23, 0.5, 1.3],
  ];
  for (const [r, h, y] of tiers) {
    const cone = new THREE.ConeGeometry(r, h, radial, 1);
    cone.translate(0, y + h / 2, 0);
    parts.push(cone);
  }
  return finishTree(parts, 'oakConifer');
}

export function bareTreeGeometry(detail = 1): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const radial = Math.max(4, 4 + detail * 2);
  const trunk = new THREE.CylinderGeometry(0.05, 0.1, 1.5, radial, 1);
  trunk.translate(0, 0.75, 0);
  parts.push(trunk);
  const boughs: readonly (readonly [number, number, number, number])[] = [
    [0.55, 1.0, 0.9, 0.55],
    [-0.6, 1.15, 1.15, -0.62],
    [0.15, 1.3, 0.75, 2.4],
  ];
  for (const [x, y, len, yaw] of boughs) {
    const bough = new THREE.CylinderGeometry(0.02, 0.042, len, radial, 1);
    bough.translate(0, len / 2, 0);
    bough.rotateZ(x > 0 ? -0.75 : 0.75);
    bough.rotateY(yaw);
    bough.translate(x * 0.35, y, 0);
    parts.push(bough);
  }
  return finishTree(parts, 'oakBareTree');
}

function finishTree(parts: THREE.BufferGeometry[], name: string): THREE.BufferGeometry {
  const merged = mergePositions(parts, name);
  for (const p of parts) p.dispose();
  return gradeByHeight(merged, 0.6, 1.1);
}

export function mergePositions(parts: readonly THREE.BufferGeometry[], name: string): THREE.BufferGeometry {
  let total = 0;
  for (const p of parts) {
    const nonIndexed = p.index ? p.index.count : p.getAttribute('position').count;
    total += nonIndexed;
  }
  const positions = new Float32Array(total * 3);
  let o = 0;
  for (const p of parts) {
    const pos = p.getAttribute('position');
    const index = p.index;
    const n = index ? index.count : pos.count;
    for (let i = 0; i < n; i++) {
      const vi = index ? index.getX(i) : i;
      positions[o++] = pos.getX(vi);
      positions[o++] = pos.getY(vi);
      positions[o++] = pos.getZ(vi);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.computeVertexNormals();
  g.name = name;
  return g;
}

export function rockGeometry(seedFactor = 1, detail = 0): THREE.BufferGeometry {
  const g = new THREE.IcosahedronGeometry(0.3, detail);
  const pos = g.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const k = 0.78 + 0.34 * Math.abs(Math.sin((x * 12.9898 + y * 78.233 + z * 37.719) * seedFactor));
    pos.setXYZ(i, x * k, Math.max(0, y * k * 0.72), z * k);
  }
  pos.needsUpdate = true;
  g.computeVertexNormals();
  g.name = 'oakRock';
  return gradeByHeight(g, 0.72, 1.08);
}

export function flowerGeometry(petals = 5): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const stem = new THREE.CylinderGeometry(0.008, 0.012, 0.22, 4, 1);
  stem.translate(0, 0.11, 0);
  parts.push(stem);
  for (let i = 0; i < petals; i++) {
    const a = (i / petals) * Math.PI * 2;
    const petal = new THREE.CircleGeometry(0.05, 5);
    petal.rotateX(-Math.PI / 2.6);
    petal.translate(Math.cos(a) * 0.045, 0.23, Math.sin(a) * 0.045);
    parts.push(petal);
  }
  const merged = mergePositions(parts, 'oakFlower');
  for (const p of parts) p.dispose();
  return gradeByHeight(merged, 0.5, 1.25);
}

export function scatterGrass(opts: ScatterOptions): VegetationHandle {
  const detail = 4;
  return buildScatter(
    opts,
    'grass',
    [{ geometry: grassClumpGeometry(detail), windHeight: 0.45, color: opts.color }],
    { wind: 0.07, castShadow: false },
  );
}

export function scatterTrees(opts: ScatterOptions & { archetypes?: readonly TreeArchetype[] }): VegetationHandle {
  const detail = 1;
  const wanted = opts.archetypes ?? (['broadleaf', 'conifer', 'bare'] as const);
  const specs: ArchetypeSpec[] = wanted.map((kind) => ({
    geometry:
      kind === 'conifer'
        ? coniferTreeGeometry(detail)
        : kind === 'bare'
          ? bareTreeGeometry(detail)
          : broadleafTreeGeometry(detail),
    windHeight: kind === 'bare' ? 1.5 : 1.7,
    color: opts.color,
  }));
  return buildScatter(opts, 'trees', specs, { wind: 0.03, castShadow: true });
}

export type TreeArchetype = 'broadleaf' | 'conifer' | 'bare';

export function scatterRocks(opts: ScatterOptions): VegetationHandle {
  const detail = 1;
  return buildScatter(
    opts,
    'rocks',
    [
      { geometry: rockGeometry(1, detail), windHeight: 0.3, color: opts.color },
      { geometry: rockGeometry(2.7, detail), windHeight: 0.3, color: opts.color },
    ],
    { wind: 0, castShadow: true },
  );
}

export function scatterFlowers(opts: ScatterOptions): VegetationHandle {
  return buildScatter(
    opts,
    'flowers',
    [{ geometry: flowerGeometry(5), windHeight: 0.25, color: opts.color }],
    { wind: 0.045, castShadow: false },
  );
}

export function excludeRect(x: number, z: number, halfX: number, halfZ: number, margin = 0.5): ExcludeFn {
  const hx = halfX + margin;
  const hz = halfZ + margin;
  return (px, pz) => Math.abs(px - x) <= hx && Math.abs(pz - z) <= hz;
}

export function excludeBand(z: number, halfWidth: number, margin = 0.4): ExcludeFn {
  const h = halfWidth + margin;
  return (_px, pz) => Math.abs(pz - z) <= h;
}

export function excludeAny(...tests: readonly (ExcludeFn | undefined)[]): ExcludeFn {
  const live = tests.filter((t): t is ExcludeFn => typeof t === 'function');
  return (x, z) => {
    for (const t of live) if (t(x, z)) return true;
    return false;
  };
}

export function plantInstances(world: WorldBuilder, name: string, geometry: THREE.BufferGeometry, material: THREE.Material, count: number) {
  world.own(material);
  world.own(geometry);
  const mesh = world.add(world.own(instancedMesh(geometry, material, count)));
  mesh.name = name;
  mesh.count = 0;
  const transform = new THREE.Object3D();
  return {
    mesh,
    transform,
    write(index: number, color?: THREE.Color) {
      transform.updateMatrix();
      mesh.setMatrixAt(index, transform.matrix);
      if (color) mesh.setColorAt(index, color);
    },
    commit(count: number) {
      mesh.count = count;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      if (mesh.frustumCulled) mesh.computeBoundingSphere();
    },
  };
}
