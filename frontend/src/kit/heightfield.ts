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

const mod = (a: number, n: number): number => ((a % n) + n) % n;

export function worldNoise(x: number, y: number, period: number, seed: number): number {
  const cx = Math.floor(x), cy = Math.floor(y);
  let fx = x - cx, fy = y - cy;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
  const ax = mod(cx, period), ay = mod(cy, period), bx = mod(cx + 1, period), by = mod(cy + 1, period);
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
  glsl: string;
  heightAt(x: number, z: number): number;
}

const smoothstep = (a: number, b: number, x: number): number => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

export function makeHeightField(spec: HeightFieldSpec): HeightField {
  const base = spec.base ?? 0, period = spec.period ?? 4096, court = spec.court;
  const lines = [`float h = ${base.toFixed(5)};`, `float roll = 1.0;`];
  if (court) lines.push(
    `float outer = max(abs(p.x) - ${court.halfX.toFixed(4)}, abs(p.y) - ${court.halfZ.toFixed(4)});`,
    `h += smoothstep(-0.25, 0.45, outer) * exp(-max(0.0, outer) / 2.1) * ${court.rise.toFixed(5)};`,
    `roll = smoothstep(0.0, 10.0, outer);`);
  spec.octaves.forEach((octave, i) => {
    const wl = octave.wavelength, seed = (octave.seed ?? i) >>> 0;
    lines.push(`h += roll * (worldNoise(p / ${wl.toFixed(4)}, ${period.toFixed(1)}, ${seed}u) - 0.5) * ${(2 * octave.amplitude).toFixed(5)} * (1.0 - smoothstep(${(wl / 8).toFixed(4)}, ${(wl / 4).toFixed(4)}, landSpacing));`);
  });
  lines.push('return h;');
  return {
    glsl: lines.join('\n'),
    heightAt(x, z) {
      let h = base, roll = 1;
      if (court) {
        const outer = Math.max(Math.abs(x) - court.halfX, Math.abs(z) - court.halfZ);
        h += smoothstep(-0.25, 0.45, outer) * Math.exp(-Math.max(0, outer) / 2.1) * court.rise;
        roll = smoothstep(0, 10, outer);
      }
      spec.octaves.forEach((octave, i) => { h += roll * (worldNoise(x / octave.wavelength, z / octave.wavelength, period, (octave.seed ?? i) >>> 0) - 0.5) * 2 * octave.amplitude; });
      return h;
    },
  };
}
