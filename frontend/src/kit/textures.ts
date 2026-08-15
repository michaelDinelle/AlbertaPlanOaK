
import * as THREE from 'three';
import { hash2f, hash2i, mulberry32 } from './rng';

export const MAX_TEXTURE_SIZE = 1024;

export function clampSize(size: number): number {
  const s = Math.max(4, Math.min(MAX_TEXTURE_SIZE, Math.floor(size) || 4));
  return 1 << Math.floor(Math.log2(s));
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function smootherstep(t: number): number {
  return t * t * (3 - 2 * t);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function wrapIndex(i: number, period: number): number {
  return ((i % period) + period) % period;
}


export function valueNoise2(x: number, y: number, period: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = smootherstep(x - ix);
  const fy = smootherstep(y - iy);
  const x0 = wrapIndex(ix, period);
  const y0 = wrapIndex(iy, period);
  const x1 = wrapIndex(ix + 1, period);
  const y1 = wrapIndex(iy + 1, period);
  const n00 = hash2f(x0, y0, seed);
  const n10 = hash2f(x1, y0, seed);
  const n01 = hash2f(x0, y1, seed);
  const n11 = hash2f(x1, y1, seed);
  return lerp(lerp(n00, n10, fx), lerp(n01, n11, fx), fy);
}

export interface FbmOptions {
  size?: number;
  frequency?: number;
  octaves?: number;
  gain?: number;
  lacunarity?: number;
  ridged?: boolean;
  contrast?: number;
}

export function fbmField(seed: number, opts: FbmOptions = {}): { size: number; data: Float32Array } {
  const size = clampSize(opts.size ?? 256);
  const frequency = Math.max(1, Math.floor(opts.frequency ?? 4));
  const octaves = Math.max(1, Math.min(8, Math.floor(opts.octaves ?? 4)));
  const gain = opts.gain ?? 0.5;
  const lacunarity = Math.max(1, Math.floor(opts.lacunarity ?? 2));
  const ridged = opts.ridged === true;
  const contrast = opts.contrast ?? 1;

  const data = new Float32Array(size * size);
  let norm = 0;
  {
    let amp = 1;
    for (let o = 0; o < octaves; o++) {
      norm += amp;
      amp *= gain;
    }
  }
  const invNorm = norm > 0 ? 1 / norm : 1;

  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let amp = 1;
      let freq = frequency;
      let sum = 0;
      for (let o = 0; o < octaves; o++) {
        const n = valueNoise2((px / size) * freq, (py / size) * freq, freq, seed + o * 7919);
        sum += amp * (ridged ? 1 - Math.abs(2 * n - 1) : n);
        amp *= gain;
        freq *= lacunarity;
      }
      let v = clamp01(sum * invNorm);
      if (contrast !== 1) v = Math.pow(v, contrast);
      data[py * size + px] = v;
    }
  }
  return { size, data };
}

export function fbm(seed: number, opts: FbmOptions = {}): THREE.DataTexture {
  const { size, data } = fbmField(seed, opts);
  return grayTexture(data, size, 'oak-fbm');
}


export interface VoronoiOptions {
  size?: number;
  cells?: number;
  jitter?: number;
}

export function voronoiField(
  seed: number,
  opts: VoronoiOptions = {},
): { size: number; f1: Float32Array; edge: Float32Array; cell: Float32Array } {
  const size = clampSize(opts.size ?? 256);
  const cells = Math.max(1, Math.floor(opts.cells ?? 8));
  const jit = clamp01(opts.jitter ?? 1);
  const f1 = new Float32Array(size * size);
  const edge = new Float32Array(size * size);
  const cell = new Float32Array(size * size);

  for (let py = 0; py < size; py++) {
    const y = (py / size) * cells;
    const gy = Math.floor(y);
    for (let px = 0; px < size; px++) {
      const x = (px / size) * cells;
      const gx = Math.floor(x);
      let best = Infinity;
      let second = Infinity;
      let bestCell = 0;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          const cx = gx + ox;
          const cy = gy + oy;
          const wx = wrapIndex(cx, cells);
          const wy = wrapIndex(cy, cells);
          const h = hash2i(wx, wy, seed);
          const jx = ((h & 0xffff) / 65536 - 0.5) * jit + 0.5;
          const jy = (((h >>> 16) & 0xffff) / 65536 - 0.5) * jit + 0.5;
          const dx = cx + jx - x;
          const dy = cy + jy - y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < best) {
            second = best;
            best = d;
            bestCell = h;
          } else if (d < second) {
            second = d;
          }
        }
      }
      const i = py * size + px;
      f1[i] = clamp01(best);
      edge[i] = clamp01(second - best);
      cell[i] = (bestCell >>> 8) / 16777216;
    }
  }
  return { size, f1, edge, cell };
}

export function voronoi(seed: number, opts: VoronoiOptions = {}): THREE.DataTexture {
  const { size, f1, edge, cell } = voronoiField(seed, opts);
  const rgba = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const o = i * 4;
    rgba[o] = (clamp01(f1[i] as number) * 255) | 0;
    rgba[o + 1] = (clamp01(edge[i] as number) * 255) | 0;
    rgba[o + 2] = (clamp01(cell[i] as number) * 255) | 0;
    rgba[o + 3] = 255;
  }
  return makeDataTexture(rgba, size, size, 'oak-voronoi');
}


export interface RampStop {
  t: number;
  color: number;
  alpha?: number;
}

export function sampleRamp(stops: readonly RampStop[], t: number, out: Uint8Array, offset = 0): void {
  const n = stops.length;
  if (n === 0) {
    out[offset] = 255;
    out[offset + 1] = 0;
    out[offset + 2] = 255;
    out[offset + 3] = 255;
    return;
  }
  const x = clamp01(t);
  let lo = stops[0] as RampStop;
  let hi = stops[n - 1] as RampStop;
  if (x <= lo.t) {
    hi = lo;
  } else if (x >= hi.t) {
    lo = hi;
  } else {
    for (let i = 0; i < n - 1; i++) {
      const a = stops[i] as RampStop;
      const b = stops[i + 1] as RampStop;
      if (x >= a.t && x <= b.t) {
        lo = a;
        hi = b;
        break;
      }
    }
  }
  const span = hi.t - lo.t;
  const k = span > 1e-9 ? (x - lo.t) / span : 0;
  const la = lo.alpha ?? 1;
  const ha = hi.alpha ?? 1;
  out[offset] = Math.round(lerp((lo.color >> 16) & 0xff, (hi.color >> 16) & 0xff, k));
  out[offset + 1] = Math.round(lerp((lo.color >> 8) & 0xff, (hi.color >> 8) & 0xff, k));
  out[offset + 2] = Math.round(lerp(lo.color & 0xff, hi.color & 0xff, k));
  out[offset + 3] = Math.round(clamp01(lerp(la, ha, k)) * 255);
}

export function rampData(stops: readonly RampStop[], width = 64): Uint8Array {
  const w = clampSize(width);
  const sorted = [...stops].sort((a, b) => a.t - b.t);
  const data = new Uint8Array(w * 4);
  for (let i = 0; i < w; i++) {
    sampleRamp(sorted, (i + 0.5) / w, data, i * 4);
  }
  return data;
}


export interface GrainOptions {
  size?: number;
  strength?: number;
  softness?: number;
}

export function grainField(seed: number, opts: GrainOptions = {}): { size: number; data: Float32Array } {
  const size = clampSize(opts.size ?? 256);
  const strength = clamp01(opts.strength ?? 1);
  const softness = clamp01(opts.softness ?? 0);
  const rng = mulberry32(seed | 0);
  const data = new Float32Array(size * size);
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let n = rng();
      if (softness > 0) {
        n = lerp(n, valueNoise2((px / size) * 32, (py / size) * 32, 32, seed ^ 0x5bf03635), softness);
      }
      data[py * size + px] = clamp01(0.5 + (n - 0.5) * strength);
    }
  }
  return { size, data };
}

export function grain(seed: number, opts: GrainOptions = {}): THREE.DataTexture {
  const { size, data } = grainField(seed, opts);
  return grayTexture(data, size, 'oak-grain');
}


export function fieldToRGBA(field: Float32Array): Uint8Array {
  const rgba = new Uint8Array(field.length * 4);
  for (let i = 0; i < field.length; i++) {
    const v = (clamp01(field[i] as number) * 255) | 0;
    const o = i * 4;
    rgba[o] = v;
    rgba[o + 1] = v;
    rgba[o + 2] = v;
    rgba[o + 3] = 255;
  }
  return rgba;
}

function grayTexture(field: Float32Array, size: number, name: string): THREE.DataTexture {
  return makeDataTexture(fieldToRGBA(field), size, size, name);
}

export function makeDataTexture(data: Uint8Array, width: number, height: number, name = 'oak-data'): THREE.DataTexture {
  const tex = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.name = name;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = height > 1 ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
  tex.generateMipmaps = height > 1;
  tex.anisotropy = 1;
  tex.needsUpdate = true;
  return tex;
}

export function hasCanvas2D(): boolean {
  return typeof document !== 'undefined' && typeof document.createElement === 'function';
}

export function radialSprite(size = 64, power = 2): THREE.DataTexture {
  const s = clampSize(size);
  const data = new Uint8Array(s * s * 4);
  const c = (s - 1) / 2;
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const dx = (x - c) / c;
      const dy = (y - c) / c;
      const d = Math.sqrt(dx * dx + dy * dy);
      const a = Math.pow(clamp01(1 - d), power);
      const o = (y * s + x) * 4;
      data[o] = 255;
      data[o + 1] = 255;
      data[o + 2] = 255;
      data[o + 3] = (clamp01(a) * 255) | 0;
    }
  }
  const tex = makeDataTexture(data, s, s, 'oak-radial');
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  return tex;
}
