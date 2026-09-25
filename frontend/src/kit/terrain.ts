import * as THREE from 'three/webgpu';
import { hash2f } from './rng';
import { oakMaterial, type OakMaterialOptions } from './materials';

export interface TerrainRampStop {
  t: number;
  color: THREE.ColorRepresentation;
}

export type TerrainColorFn = (height: number, slope: number, x: number, z: number) => THREE.ColorRepresentation;

export interface FlattenRect {
  x: number;
  z: number;
  halfX: number;
  halfZ: number;
  feather?: number;
}

export type FlattenMask = FlattenRect | ((x: number, z: number) => number);

export interface TerrainOptions {
  size: number | readonly [number, number];
  segments?: number | readonly [number, number];
  heightFn?: (x: number, z: number) => number;
  colorRamp?: readonly TerrainRampStop[] | TerrainColorFn;
  slopeColor?: THREE.ColorRepresentation;
  slopeStrength?: number;
  slopeSharpness?: number;
  colorNoise?: number;
  flattenMask?: FlattenMask;
  flattenHeight?: number;
  seed?: number;
  jitter?: number;
  material?: THREE.Material;
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
  heights: Float32Array;
  minHeight: number;
  maxHeight: number;
  sampleHeight(x: number, z: number): number;
  sampleNormal(x: number, z: number, out: THREE.Vector3): THREE.Vector3;
  dispose(): void;
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

export function makeTerrain(opts: TerrainOptions): TerrainHandle {
  const [width, depth] = pair(opts.size, 10);
  const [baseSegX, baseSegZ] = pair(opts.segments, 64);
  const segmentsX = Math.max(1, Math.round(baseSegX));
  const segmentsZ = Math.max(1, Math.round(baseSegZ));
  const nx = segmentsX + 1;
  const nz = segmentsZ + 1;
  const dx = width / segmentsX;
  const dz = depth / segmentsZ;
  const x0 = -width / 2;
  const z0 = -depth / 2;
  const seed = opts.seed ?? 0;
  const heightFn = opts.heightFn;
  const flattenHeight = opts.flattenHeight ?? 0;

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
      name: 'oakTerrainMaterial',
      ...opts.materialOptions,
    });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = opts.name ?? 'oakTerrain';
  mesh.receiveShadow = opts.receiveShadow ?? true;
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
    return tx + tz <= 1
      ? h00 + (h10 - h00) * tx + (h01 - h00) * tz
      : h11 + (h01 - h11) * (1 - tx) + (h10 - h11) * (1 - tz);
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
