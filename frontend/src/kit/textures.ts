import * as THREE from 'three';
import { hash2f } from './rng';

const MAX_TEXTURE_SIZE = 1024;

function clampSize(size: number): number {
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
}

export function fbmField(seed: number, opts: FbmOptions = {}): { size: number; data: Float32Array } {
  const size = clampSize(opts.size ?? 256);
  const frequency = Math.max(1, Math.floor(opts.frequency ?? 4));
  const octaves = Math.max(1, Math.min(8, Math.floor(opts.octaves ?? 4)));
  const gain = opts.gain ?? 0.5;

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
        sum += amp * n;
        amp *= gain;
        freq *= 2;
      }
      data[py * size + px] = clamp01(sum * invNorm);
    }
  }
  return { size, data };
}

export function fbm(seed: number, opts: FbmOptions = {}): THREE.DataTexture {
  const { size, data } = fbmField(seed, opts);
  return grayTexture(data, size, 'oak-fbm');
}


function fieldToRGBA(field: Float32Array): Uint8Array {
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
