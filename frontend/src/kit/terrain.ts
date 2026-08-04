// kit/terrain — displaced ground with ramp-driven vertex colors.
//
// STYLE.md: "Vertex-color / ramp-driven albedo — never image textures for surface
// detail", "Ground/floor at y = 0", and the accuracy law — every state-bearing mesh is
// positioned from ctx.geometry + FrameState, which for terrain-riding scenes means
// `heightAt`. `sampleHeight()` here IS the rendered surface: it bilinearly interpolates
// the exact height field the vertices were built from, so a cart placed at
// `sampleHeight(x, z)` sits on the visible ground, not near it.
//
// The optional `flattenMask` carves a flat inset (a grid playfield, a courtyard floor)
// out of an otherwise rolling terrain with a feathered edge — one mesh, one draw call.

import * as THREE from 'three';
import { hash2f } from './rng';
import { oakMaterial, type OakMaterialOptions } from './materials';
import type { QualityTier } from './types';

export interface TerrainRampStop {
  /** normalized height in [0, 1] */
  t: number;
  color: THREE.ColorRepresentation;
}

/** Full control: returns the vertex color for a sample. */
export type TerrainColorFn = (height: number, slope: number, x: number, z: number) => THREE.ColorRepresentation;

export interface FlattenRect {
  /** center */
  x: number;
  z: number;
  halfX: number;
  halfZ: number;
  /** world units over which the flatten fades out (0 = hard edge) */
  feather?: number;
}

export type FlattenMask = FlattenRect | ((x: number, z: number) => number);

export interface TerrainOptions {
  /** world extent: a number is square, or [width(x), depth(z)] */
  size: number | readonly [number, number];
  /** hero segment count before tier scaling; a number is square */
  segments?: number | readonly [number, number];
  /** world-space height field; defaults to a flat plane at y = 0 */
  heightFn?: (x: number, z: number) => number;
  /** ramp stops over normalized height, or a full color function */
  colorRamp?: readonly TerrainRampStop[] | TerrainColorFn;
  /** color blended in on steep faces (rock through grass) */
  slopeColor?: THREE.ColorRepresentation;
  slopeStrength?: number;
  /** exponent on the slope term — higher = only the steepest faces show rock */
  slopeSharpness?: number;
  /** per-vertex value break-up, deterministic from `seed` (0 disables) */
  colorNoise?: number;
  flattenMask?: FlattenMask;
  /** the height the flattened region is pulled to (default 0 = the grid floor) */
  flattenHeight?: number;
  /** lane seed — the only randomness source (kit/rng) */
  seed?: number;
  /**
   * Lateral vertex jitter in world units, for a hand-made silhouette. Heights stay on
   * the lattice, so jitter > 0 makes the mesh deviate from `sampleHeight` by
   * O(jitter · slope). Default 0 — accuracy first.
   */
  jitter?: number;
  tier?: QualityTier;
  /** override the material entirely (must handle vertexColors itself) */
  material?: THREE.Material;
  /** extra options for the default oakMaterial */
  materialOptions?: OakMaterialOptions;
  receiveShadow?: boolean;
  castShadow?: boolean;
  name?: string;
}

export interface TerrainHandle {
  mesh: THREE.Mesh;
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
  width: number;
  depth: number;
  segmentsX: number;
  segmentsZ: number;
  /** (segmentsX+1) × (segmentsZ+1) row-major lattice, index = j*(segmentsX+1)+i */
  heights: Float32Array;
  minHeight: number;
  maxHeight: number;
  /** the rendered surface height at any world (x, z); clamped to the terrain bounds */
  sampleHeight(x: number, z: number): number;
  /** analytic surface normal at (x, z), written into `out` (allocation-free) */
  sampleNormal(x: number, z: number, out: THREE.Vector3): THREE.Vector3;
  dispose(): void;
}

/** hero 1.0 / medium 0.6 / thumb 0.3 — kit factories scale themselves (STYLE.md). */
export function terrainSegmentScale(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 0.3 : tier === 'medium' ? 0.6 : 1;
}

function pair(v: number | readonly [number, number] | undefined, fallback: number): [number, number] {
  if (v === undefined) return [fallback, fallback];
  if (typeof v === 'number') return [v, v];
  return [v[0], v[1]];
}

function smoothstep01(t: number): number {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  return t * t * (3 - 2 * t);
}

/** Flatten weight in [0,1] for a rect mask (1 = fully flattened). */
export function rectFlattenWeight(rect: FlattenRect, x: number, z: number): number {
  const feather = Math.max(0, rect.feather ?? 0);
  const dx = Math.abs(x - rect.x);
  const dz = Math.abs(z - rect.z);
  const wx = feather > 0 ? 1 - smoothstep01((dx - rect.halfX) / feather) : dx <= rect.halfX ? 1 : 0;
  const wz = feather > 0 ? 1 - smoothstep01((dz - rect.halfZ) / feather) : dz <= rect.halfZ ? 1 : 0;
  return Math.min(wx, wz);
}

function flattenWeightAt(mask: FlattenMask | undefined, x: number, z: number): number {
  if (!mask) return 0;
  if (typeof mask === 'function') {
    const w = mask(x, z);
    return w < 0 ? 0 : w > 1 ? 1 : w;
  }
  return rectFlattenWeight(mask, x, z);
}

interface LinearStop {
  t: number;
  color: THREE.Color;
}

function buildStops(stops: readonly TerrainRampStop[]): LinearStop[] {
  return stops
    .map((s) => ({ t: s.t, color: new THREE.Color().set(s.color) }))
    .sort((a, b) => a.t - b.t);
}

function sampleStops(stops: readonly LinearStop[], t: number, out: THREE.Color): THREE.Color {
  const n = stops.length;
  if (n === 0) return out.setRGB(1, 1, 1);
  const x = t < 0 ? 0 : t > 1 ? 1 : t;
  const first = stops[0] as LinearStop;
  const last = stops[n - 1] as LinearStop;
  if (x <= first.t) return out.copy(first.color);
  if (x >= last.t) return out.copy(last.color);
  for (let i = 0; i < n - 1; i++) {
    const a = stops[i] as LinearStop;
    const b = stops[i + 1] as LinearStop;
    if (x >= a.t && x <= b.t) {
      const span = b.t - a.t;
      const k = span > 1e-9 ? (x - a.t) / span : 0;
      return out.copy(a.color).lerp(b.color, k);
    }
  }
  return out.copy(last.color);
}

/**
 * Build a terrain.
 *
 * ```ts
 * const t = makeTerrain({
 *   size: [24, 12], segments: 160, tier,
 *   heightFn: (x) => Math.sin(3 * x) * 0.45 + 0.55,     // MountainCar IS this curve
 *   colorRamp: [{ t: 0, color: p.groundA }, { t: 0.7, color: p.groundB }, { t: 1, color: 0xffffff }],
 * });
 * cart.position.set(x, t.sampleHeight(x, 0), 0);         // exactly on the ground
 * ```
 */
export function makeTerrain(opts: TerrainOptions): TerrainHandle {
  const [width, depth] = pair(opts.size, 10);
  const scale = terrainSegmentScale(opts.tier);
  const [baseSegX, baseSegZ] = pair(opts.segments, 64);
  const segmentsX = Math.max(1, Math.round(baseSegX * scale));
  const segmentsZ = Math.max(1, Math.round(baseSegZ * scale));
  const nx = segmentsX + 1;
  const nz = segmentsZ + 1;
  const dx = width / segmentsX;
  const dz = depth / segmentsZ;
  const x0 = -width / 2;
  const z0 = -depth / 2;
  const seed = opts.seed ?? 0;
  const heightFn = opts.heightFn;
  const flattenHeight = opts.flattenHeight ?? 0;

  // ---- height field (the authority for both the mesh and sampleHeight) ----
  const heights = new Float32Array(nx * nz);
  let minHeight = Infinity;
  let maxHeight = -Infinity;
  for (let j = 0; j < nz; j++) {
    const z = z0 + j * dz;
    for (let i = 0; i < nx; i++) {
      const x = x0 + i * dx;
      let h = heightFn ? heightFn(x, z) : 0;
      if (!Number.isFinite(h)) h = 0;
      const w = flattenWeightAt(opts.flattenMask, x, z);
      if (w > 0) h = h + (flattenHeight - h) * w;
      heights[j * nx + i] = h;
      if (h < minHeight) minHeight = h;
      if (h > maxHeight) maxHeight = h;
    }
  }
  if (!Number.isFinite(minHeight)) minHeight = 0;
  if (!Number.isFinite(maxHeight)) maxHeight = 0;
  const span = maxHeight - minHeight;
  const invSpan = span > 1e-9 ? 1 / span : 0;

  const heightAtIndex = (i: number, j: number): number => {
    const ci = i < 0 ? 0 : i > segmentsX ? segmentsX : i;
    const cj = j < 0 ? 0 : j > segmentsZ ? segmentsZ : j;
    return heights[cj * nx + ci] as number;
  };

  // ---- attributes ----
  const count = nx * nz;
  const positions = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  const uvs = new Float32Array(count * 2);
  const colors = new Float32Array(count * 3);

  const ramp = opts.colorRamp;
  const colorFn = typeof ramp === 'function' ? ramp : null;
  const rampStops = ramp !== undefined && typeof ramp !== 'function' ? buildStops(ramp) : null;
  const slopeColor = opts.slopeColor !== undefined ? new THREE.Color().set(opts.slopeColor) : null;
  const slopeStrength = opts.slopeStrength ?? (slopeColor ? 0.6 : 0);
  const slopeSharpness = opts.slopeSharpness ?? 2;
  const colorNoise = opts.colorNoise ?? 0;
  const jitter = opts.jitter ?? 0;
  const tmpColor = new THREE.Color();

  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      const idx = j * nx + i;
      const x = x0 + i * dx;
      const z = z0 + j * dz;
      const h = heights[idx] as number;

      // analytic normal from central differences on the lattice
      const hx = (heightAtIndex(i + 1, j) - heightAtIndex(i - 1, j)) / ((i === 0 || i === segmentsX ? 1 : 2) * dx);
      const hz = (heightAtIndex(i, j + 1) - heightAtIndex(i, j - 1)) / ((j === 0 || j === segmentsZ ? 1 : 2) * dz);
      const nlen = Math.sqrt(hx * hx + 1 + hz * hz);
      const ny = 1 / nlen;

      const edge = i === 0 || i === segmentsX || j === 0 || j === segmentsZ;
      let px = x;
      let pz = z;
      if (jitter > 0 && !edge) {
        px += (hash2f(i, j, seed) - 0.5) * 2 * jitter;
        pz += (hash2f(i, j, seed ^ 0x51ed270b) - 0.5) * 2 * jitter;
      }

      positions[idx * 3] = px;
      positions[idx * 3 + 1] = h;
      positions[idx * 3 + 2] = pz;
      normals[idx * 3] = -hx * ny;
      normals[idx * 3 + 1] = ny;
      normals[idx * 3 + 2] = -hz * ny;
      uvs[idx * 2] = i / segmentsX;
      uvs[idx * 2 + 1] = j / segmentsZ;

      const slope = 1 - ny;
      if (colorFn) {
        tmpColor.set(colorFn(h, slope, x, z));
      } else if (rampStops) {
        sampleStops(rampStops, (h - minHeight) * invSpan, tmpColor);
      } else {
        tmpColor.setRGB(1, 1, 1);
      }
      if (slopeColor && slopeStrength > 0) {
        tmpColor.lerp(slopeColor, Math.min(1, Math.pow(slope, slopeSharpness) * slopeStrength));
      }
      if (colorNoise > 0) {
        const k = 1 + (hash2f(i, j, seed ^ 0x2545f491) - 0.5) * 2 * colorNoise;
        tmpColor.multiplyScalar(k);
      }
      colors[idx * 3] = tmpColor.r;
      colors[idx * 3 + 1] = tmpColor.g;
      colors[idx * 3 + 2] = tmpColor.b;
    }
  }

  // ---- indices (CCW when viewed from +y) ----
  const triCount = segmentsX * segmentsZ * 2;
  const indices = count > 65535 ? new Uint32Array(triCount * 3) : new Uint16Array(triCount * 3);
  let o = 0;
  for (let j = 0; j < segmentsZ; j++) {
    for (let i = 0; i < segmentsX; i++) {
      const a = j * nx + i;
      const b = a + 1;
      const c = a + nx;
      const d = c + 1;
      indices[o++] = a;
      indices[o++] = c;
      indices[o++] = b;
      indices[o++] = b;
      indices[o++] = c;
      indices[o++] = d;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();
  geometry.name = opts.name ?? 'oakTerrain';

  const material =
    opts.material ??
    oakMaterial({
      vertexColors: true,
      flatShading: true,
      roughness: 0.95,
      metalness: 0,
      tier: opts.tier,
      name: 'oakTerrainMaterial',
      ...opts.materialOptions,
    });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = opts.name ?? 'oakTerrain';
  mesh.receiveShadow = opts.receiveShadow ?? opts.tier !== 'thumb';
  mesh.castShadow = opts.castShadow ?? false;

  const ownsMaterial = opts.material === undefined;

  const sampleHeight = (x: number, z: number): number => {
    const fx = Math.min(segmentsX, Math.max(0, (x - x0) / dx));
    const fz = Math.min(segmentsZ, Math.max(0, (z - z0) / dz));
    const i0 = Math.min(segmentsX - 1, Math.floor(fx));
    const j0 = Math.min(segmentsZ - 1, Math.floor(fz));
    const tx = fx - i0;
    const tz = fz - j0;
    const h00 = heights[j0 * nx + i0] as number;
    const h10 = heights[j0 * nx + i0 + 1] as number;
    const h01 = heights[(j0 + 1) * nx + i0] as number;
    const h11 = heights[(j0 + 1) * nx + i0 + 1] as number;
    const a = h00 + (h10 - h00) * tx;
    const b = h01 + (h11 - h01) * tx;
    return a + (b - a) * tz;
  };

  const sampleNormal = (x: number, z: number, out: THREE.Vector3): THREE.Vector3 => {
    const ex = dx * 0.5;
    const ez = dz * 0.5;
    const gx = (sampleHeight(x + ex, z) - sampleHeight(x - ex, z)) / (2 * ex);
    const gz = (sampleHeight(x, z + ez) - sampleHeight(x, z - ez)) / (2 * ez);
    return out.set(-gx, 1, -gz).normalize();
  };

  return {
    mesh,
    geometry,
    material,
    width,
    depth,
    segmentsX,
    segmentsZ,
    heights,
    minHeight,
    maxHeight,
    sampleHeight,
    sampleNormal,
    dispose() {
      geometry.dispose();
      if (ownsMaterial) material.dispose();
      mesh.removeFromParent();
    },
  };
}
