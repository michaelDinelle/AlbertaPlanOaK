// kit/vegetation — the preserve's plant life.
//
// STYLE.md: "Silhouette-first low-poly geometry", "Vertex-color / ramp-driven albedo",
// "kit/rng (seeded with ctx.seed) for ALL scatter/randomness — scenes are deterministic",
// and the tier contract "vegetation ×0.5 (medium) / ×0.15 (thumb)".
//
// Shape of the API: ONE InstancedMesh per archetype (one draw call for ten thousand
// grass blades), per-instance hue jitter carried on `instanceColor`, wind in the vertex
// shader via `oakMaterial({ wind })`. Because three multiplies
// `material.color * vertexColor * instanceColor`, a scene RETINTS a whole scatter with
// `handle.retint(color)` and keeps every per-blade variation — no remodelling, no
// second geometry, no second draw call.
//
// Placement is a deterministic rejection sample: `exclude(x, z)` returns true for cells
// the art must never cover (the playfield, the boardwalk, the track). A scatter that
// lands on the playfield is an accuracy bug, so exclusion is a first-class argument
// rather than something scenes remember to do.

import * as THREE from 'three';
import { oakMaterial, oakUniforms } from './materials';
import type { OakMaterial } from './materials';
import { streamFor } from './rng';
import type { QualityTier } from './types';

/** hero ×1 / medium ×0.5 / thumb ×0.15 (STYLE.md quality tiers). */
export function vegetationTierScale(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 0.15 : tier === 'medium' ? 0.5 : 1;
}

/** Rectangular or annular scatter region, in world units. */
export interface ScatterArea {
  /** center (default origin) */
  x?: number;
  z?: number;
  halfX: number;
  halfZ: number;
  /** skip anything inside this radius of the center (keeps hero space clear) */
  innerRadius?: number;
}

/** true = do NOT place here. This is how "never on the playfield" is enforced. */
export type ExcludeFn = (x: number, z: number) => boolean;

export interface ScatterOptions {
  /** hero-tier instance count (tier scaling is applied inside) */
  count: number;
  area: ScatterArea;
  /** lane seed — ctx.seed, always */
  seed: number;
  tier?: QualityTier;
  /** the playfield test: return true where nothing may be placed */
  exclude?: ExcludeFn;
  /** ground height lookup (terrain.sampleHeight); default flat y = 0 */
  heightAt?: (x: number, z: number) => number;
  /** skip placements on ground steeper than this (0..1, 1 = allow anything) */
  maxSlope?: number;
  /** base tint — `retint()` replaces this later */
  color: THREE.ColorRepresentation;
  /** per-instance multiplicative hue/value jitter, 0..1 (default 0.16) */
  hueJitter?: number;
  /** uniform scale range [min, max] (default [0.8, 1.25]) */
  scale?: readonly [number, number];
  /** extra vertical stretch range applied on top of `scale` */
  stretch?: readonly [number, number];
  /** vertex wind amplitude at full height (0 disables) */
  wind?: number;
  windFrequency?: number;
  /** sky-tinted fresnel */
  rimColor?: THREE.ColorRepresentation;
  rimStrength?: number;
  castShadow?: boolean;
  receiveShadow?: boolean;
  /** attempts per instance before giving up on the exclusion test (default 12) */
  maxTries?: number;
  name?: string;
}

export interface VegetationHandle {
  /** add this ONE object to the scene */
  group: THREE.Group;
  /** one per archetype — trees have three, everything else has one */
  meshes: THREE.InstancedMesh[];
  materials: OakMaterial[];
  /** instances actually placed (after tier scaling and exclusion) */
  count: number;
  /** ctx.time() — drives the vertex wind */
  setTime(t: number): void;
  /** re-tint every archetype, preserving per-instance jitter */
  retint(color: THREE.ColorRepresentation, secondary?: THREE.ColorRepresentation): void;
  setWind(amplitude: number, frequency?: number): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// placement
// ---------------------------------------------------------------------------

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

/**
 * Deterministic rejection sampling inside `area`, honouring `exclude` and `maxSlope`.
 * Exported for scenes that need the same placement for a bespoke hero mesh.
 */
export function scatterPlacements(opts: ScatterOptions, streamName: string, variants = 1): Placement[] {
  const scale = vegetationTierScale(opts.tier);
  const wanted = Math.max(0, Math.round(opts.count * scale));
  const rng = streamFor(opts.seed | 0, streamName);
  const cx = opts.area.x ?? 0;
  const cz = opts.area.z ?? 0;
  const inner = opts.area.innerRadius ?? 0;
  const maxTries = Math.max(1, opts.maxTries ?? 12);
  const [sMin, sMax] = opts.scale ?? [0.8, 1.25];
  const [tMin, tMax] = opts.stretch ?? [1, 1];
  const heightAt = opts.heightAt;
  const maxSlope = opts.maxSlope ?? 1;
  const out: Placement[] = [];

  for (let i = 0; i < wanted; i++) {
    let placed = false;
    for (let attempt = 0; attempt < maxTries && !placed; attempt++) {
      const x = cx + (rng() * 2 - 1) * opts.area.halfX;
      const z = cz + (rng() * 2 - 1) * opts.area.halfZ;
      // draw the remaining stream values unconditionally so a rejected sample does not
      // desynchronize the stream for later instances
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
  /** local height at which wind sway is full */
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

  const meshes: THREE.InstancedMesh[] = [];
  const materials: OakMaterial[] = [];
  const counts = new Array<number>(archetypes.length).fill(0);
  for (const p of placements) counts[p.variant] = (counts[p.variant] ?? 0) + 1;

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scaleVec = new THREE.Vector3();
  const color = new THREE.Color();

  for (let a = 0; a < archetypes.length; a++) {
    const spec = archetypes[a] as ArchetypeSpec;
    const n = counts[a] ?? 0;
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
      tier: opts.tier,
      name: `${group.name}:${a}`,
    });
    materials.push(material);

    const mesh = new THREE.InstancedMesh(spec.geometry, material, Math.max(1, n));
    mesh.name = `${group.name}:mesh${a}`;
    mesh.count = 0;
    mesh.castShadow = (opts.castShadow ?? defaults.castShadow) && opts.tier !== 'thumb';
    mesh.receiveShadow = (opts.receiveShadow ?? false) && opts.tier !== 'thumb';
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, n) * 3), 3);
    meshes.push(mesh);
  }

  const written = new Array<number>(archetypes.length).fill(0);
  for (const p of placements) {
    const mesh = meshes[p.variant];
    if (!mesh) continue;
    const i = written[p.variant] ?? 0;
    position.set(p.x, p.y, p.z);
    euler.set(0, p.yaw, 0);
    quaternion.setFromEuler(euler);
    scaleVec.set(p.scale, p.scale * p.stretch, p.scale);
    matrix.compose(position, quaternion, scaleVec);
    mesh.setMatrixAt(i, matrix);
    const k = 1 + (p.jitter - 0.5) * 2 * hueJitter;
    color.setRGB(k, k * (1 - hueJitter * 0.25 * (p.jitter - 0.5)), k * (1 + hueJitter * 0.2 * (p.jitter - 0.5)));
    mesh.setColorAt(i, color);
    written[p.variant] = i + 1;
  }

  let total = 0;
  for (let a = 0; a < meshes.length; a++) {
    const mesh = meshes[a] as THREE.InstancedMesh;
    const n = written[a] ?? 0;
    mesh.count = n;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    total += n;
    if (n > 0) group.add(mesh);
  }

  const tintA = new THREE.Color();
  const tintB = new THREE.Color();
  return {
    group,
    meshes,
    materials,
    count: total,
    setTime(t) {
      for (const m of materials) oakUniforms(m)?.setTime(t);
    },
    retint(primary, secondary) {
      tintA.set(primary);
      if (secondary !== undefined) tintB.set(secondary);
      for (let i = 0; i < materials.length; i++) {
        const m = materials[i] as OakMaterial;
        if (secondary !== undefined && materials.length > 1) {
          m.color.copy(tintA).lerp(tintB, i / (materials.length - 1));
        } else {
          m.color.copy(tintA);
        }
      }
    },
    setWind(amplitude, frequency) {
      for (const m of materials) oakUniforms(m)?.setWind(amplitude, frequency);
    },
    dispose() {
      for (const mesh of meshes) mesh.dispose();
      for (const spec of archetypes) spec.geometry.dispose();
      for (const m of materials) m.dispose();
      meshes.length = 0;
      materials.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// procedural archetypes
// ---------------------------------------------------------------------------

/** vertical gradient vertex colors: dark at the base, bright at the tip */
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

/** A tufted grass clump: 3 crossed tapered blades, base at y = 0. */
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
    // two triangles per blade (a quad tapering to a point)
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

/** Archetype A — broadleaf: a trunk plus a cluster of icosahedral canopy blobs. */
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

/** Archetype B — conifer: a stack of shrinking cones on a short trunk. */
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

/** Archetype C — bare: a trunk with three angled boughs (winter, moor, ruin). */
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

/**
 * Merge position-only geometries into one non-indexed buffer. Deliberately tiny (no
 * BufferGeometryUtils import): archetypes are built once, at scene init.
 */
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

/** A faceted boulder: an icosahedron pushed around deterministically. */
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

/** A four-petal flower head on a short stem. */
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

// ---------------------------------------------------------------------------
// the four scatters
// ---------------------------------------------------------------------------

/** Wind-leaning grass tufts. The densest scatter in the kit — one draw call. */
export function scatterGrass(opts: ScatterOptions): VegetationHandle {
  const detail = opts.tier === 'thumb' ? 2 : opts.tier === 'medium' ? 3 : 4;
  return buildScatter(
    opts,
    'grass',
    [{ geometry: grassClumpGeometry(detail), windHeight: 0.45, color: opts.color }],
    { wind: 0.07, castShadow: false },
  );
}

/**
 * Trees, in the three house archetypes (broadleaf cluster / conifer stack / bare).
 * `retint(primary, secondary)` ramps across the archetypes, so an autumn scene and a
 * spring scene are the same three meshes with different colors.
 */
export function scatterTrees(opts: ScatterOptions & { archetypes?: readonly TreeArchetype[] }): VegetationHandle {
  const detail = opts.tier === 'thumb' ? 0 : opts.tier === 'medium' ? 0 : 1;
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

/** Faceted boulders and scree. Never wind-animated. */
export function scatterRocks(opts: ScatterOptions): VegetationHandle {
  const detail = opts.tier === 'hero' ? 1 : 0;
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

/** Flower heads for meadows, blossom quadrants and lantern-flowers. */
export function scatterFlowers(opts: ScatterOptions): VegetationHandle {
  return buildScatter(
    opts,
    'flowers',
    [{ geometry: flowerGeometry(opts.tier === 'thumb' ? 4 : 5), windHeight: 0.25, color: opts.color }],
    { wind: 0.045, castShadow: false },
  );
}

/**
 * The standard "never on the playfield" test for grid scenes: excludes an axis-aligned
 * rect (the cell field) plus a margin.
 */
export function excludeRect(x: number, z: number, halfX: number, halfZ: number, margin = 0.5): ExcludeFn {
  const hx = halfX + margin;
  const hz = halfZ + margin;
  return (px, pz) => Math.abs(px - x) <= hx && Math.abs(pz - z) <= hz;
}

/** Exclude a corridor along x (the MountainCar track, the CartPole boardwalk). */
export function excludeBand(z: number, halfWidth: number, margin = 0.4): ExcludeFn {
  const h = halfWidth + margin;
  return (_px, pz) => Math.abs(pz - z) <= h;
}

/** Compose exclusion tests: excluded if ANY of them says so. */
export function excludeAny(...tests: readonly (ExcludeFn | undefined)[]): ExcludeFn {
  const live = tests.filter((t): t is ExcludeFn => typeof t === 'function');
  return (x, z) => {
    for (const t of live) if (t(x, z)) return true;
    return false;
  };
}
