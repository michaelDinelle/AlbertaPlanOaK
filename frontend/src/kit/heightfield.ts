import type * as THREE from 'three/webgpu';
import { Fn, abs, exp, float, floor, fract, max, mix, mod, smoothstep, uint, vec2 } from 'three/tsl';

type N<T extends string> = THREE.Node<T>;

const worldHashNode = Fn(([p, seed]: [N<'vec2'>, N<'uint'>]) => {
  const h = seed.bitXor(uint(0x9e3779b9)).toVar();
  h.assign(h.bitXor(uint(p.x)).mul(uint(0x85ebca6b)));
  h.assign(h.bitXor(h.shiftRight(uint(13))));
  h.assign(h.bitXor(uint(p.y)).mul(uint(0xc2b2ae35)));
  h.assign(h.bitXor(h.shiftRight(uint(16))));
  h.assign(h.mul(uint(0x27d4eb2f)));
  h.assign(h.bitXor(h.shiftRight(uint(15))));
  return float(h).div(4294967296);
}).setLayout({ name: 'worldHash', type: 'float', inputs: [{ name: 'p', type: 'vec2' }, { name: 'seed', type: 'uint' }] });

const worldNoiseNode = Fn(([p, period, seed]: [N<'vec2'>, N<'float'>, N<'uint'>]) => {
  const cell = floor(p), t = fract(p);
  const f = t.mul(t).mul(float(3).sub(t.mul(2)));
  const a = mod(cell, period), b = mod(cell.add(1), period);
  return mix(mix(worldHashNode(a, seed), worldHashNode(vec2(b.x, a.y), seed), f.x), mix(worldHashNode(vec2(a.x, b.y), seed), worldHashNode(b, seed), f.x), f.y);
}).setLayout({ name: 'worldNoise', type: 'float', inputs: [{ name: 'p', type: 'vec2' }, { name: 'period', type: 'float' }, { name: 'seed', type: 'uint' }] });

export const worldNoiseAt = (p: N<'vec2'>, period: number, seed: number): N<'float'> => worldNoiseNode(p, float(period), uint(seed >>> 0)) as unknown as N<'float'>;

export function worldHash(x: number, y: number, seed: number): number {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  h = Math.imul((h ^ x) >>> 0, 0x85ebca6b) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul((h ^ y) >>> 0, 0xc2b2ae35) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, 0x27d4eb2f) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  return h / 4294967296;
}

const modulo = (a: number, n: number): number => ((a % n) + n) % n;

export function worldNoise(x: number, y: number, period: number, seed: number): number {
  const cx = Math.floor(x), cy = Math.floor(y);
  let fx = x - cx, fy = y - cy;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
  const ax = modulo(cx, period), ay = modulo(cy, period), bx = modulo(cx + 1, period), by = modulo(cy + 1, period);
  const top = worldHash(ax, ay, seed) + (worldHash(bx, ay, seed) - worldHash(ax, ay, seed)) * fx;
  const bottom = worldHash(ax, by, seed) + (worldHash(bx, by, seed) - worldHash(ax, by, seed)) * fx;
  return top + (bottom - top) * fy;
}

export interface HeightOctave {
  wavelength: number;
  amplitude: number;
  seed?: number;
}

export interface HeightCourt {
  halfX: number;
  halfZ: number;
  rise: number;
}

export interface HeightFieldSpec {
  base?: number;
  octaves: readonly HeightOctave[];
  court?: HeightCourt;
  period?: number;
}

export interface HeightField {
  height(p: N<'vec2'>, spacing: N<'float'>): N<'float'>;
  heightAt(x: number, z: number): number;
}

const smooth = (a: number, b: number, x: number): number => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

export function makeHeightField(spec: HeightFieldSpec): HeightField {
  const base = spec.base ?? 0, period = spec.period ?? 4096, court = spec.court;
  return {
    height(p, spacing) {
      let h: N<'float'> = float(base), roll: N<'float'> = float(1);
      if (court) {
        const outer = max(abs(p.x).sub(court.halfX), abs(p.y).sub(court.halfZ));
        h = h.add(smoothstep(-0.25, 0.45, outer).mul(exp(max(0, outer).negate().div(2.1))).mul(court.rise));
        roll = smoothstep(0, 10, outer);
      }
      spec.octaves.forEach((octave, i) => {
        const wl = octave.wavelength;
        h = h.add(roll.mul(worldNoiseAt(p.div(wl), period, octave.seed ?? i).sub(0.5)).mul(2 * octave.amplitude).mul(smoothstep(wl / 8, wl / 4, spacing).oneMinus()));
      });
      return h;
    },
    heightAt(x, z) {
      let h = base, roll = 1;
      if (court) {
        const outer = Math.max(Math.abs(x) - court.halfX, Math.abs(z) - court.halfZ);
        h += smooth(-0.25, 0.45, outer) * Math.exp(-Math.max(0, outer) / 2.1) * court.rise;
        roll = smooth(0, 10, outer);
      }
      spec.octaves.forEach((octave, i) => { h += roll * (worldNoise(x / octave.wavelength, z / octave.wavelength, period, (octave.seed ?? i) >>> 0) - 0.5) * 2 * octave.amplitude; });
      return h;
    },
  };
}
