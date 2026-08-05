
export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let a = seed | 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function range(rng: Rng, min: number, max: number): number {
  return min + (max - min) * rng();
}

export function rangeInt(rng: Rng, min: number, max: number): number {
  const lo = Math.ceil(min);
  const hi = Math.floor(max);
  if (hi <= lo) return lo;
  return lo + Math.min(hi - lo - 1, (rng() * (hi - lo)) | 0);
}

export function jitter(rng: Rng, amount = 1): number {
  return (rng() * 2 - 1) * amount;
}

export function chance(rng: Rng, p: number): boolean {
  return rng() < p;
}

export function pick<T>(rng: Rng, items: readonly T[]): T {
  const n = items.length;
  if (n === 0) throw new Error('kit/rng: pick() on an empty list');
  const i = Math.min(n - 1, (rng() * n) | 0);
  return items[i] as T;
}

export function pickWeighted<T>(rng: Rng, items: readonly T[], weights: readonly number[]): T {
  const n = Math.min(items.length, weights.length);
  if (n === 0) throw new Error('kit/rng: pickWeighted() on an empty list');
  let total = 0;
  for (let i = 0; i < n; i++) total += Math.max(0, weights[i] as number);
  if (total <= 0) return items[0] as T;
  let r = rng() * total;
  for (let i = 0; i < n; i++) {
    r -= Math.max(0, weights[i] as number);
    if (r <= 0) return items[i] as T;
  }
  return items[n - 1] as T;
}

export function shuffle<T>(rng: Rng, items: T[]): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.min(i, (rng() * (i + 1)) | 0);
    const a = items[i] as T;
    items[i] = items[j] as T;
    items[j] = a;
  }
  return items;
}

export function gaussian(rng: Rng, mean = 0, sd = 1): number {
  let u = rng();
  if (u <= Number.MIN_VALUE) u = Number.MIN_VALUE;
  const v = rng();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function onCircle(rng: Rng, out: [number, number]): [number, number] {
  const a = rng() * Math.PI * 2;
  out[0] = Math.cos(a);
  out[1] = Math.sin(a);
  return out;
}

export function inDisk(rng: Rng, out: [number, number]): [number, number] {
  const a = rng() * Math.PI * 2;
  const r = Math.sqrt(rng());
  out[0] = Math.cos(a) * r;
  out[1] = Math.sin(a) * r;
  return out;
}


export function hash1i(x: number, seed = 0): number {
  let h = Math.imul((x | 0) ^ (seed | 0), 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

export function hash2i(x: number, y: number, seed = 0): number {
  let h = ((seed | 0) ^ 0x9e3779b9) | 0;
  h = Math.imul(h ^ (x | 0), 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h ^ (y | 0), 0xc2b2ae35);
  h ^= h >>> 16;
  h = Math.imul(h, 0x27d4eb2f);
  h ^= h >>> 15;
  return h >>> 0;
}

export function hash3i(x: number, y: number, z: number, seed = 0): number {
  return hash2i(hash2i(x, y, seed) | 0, z, seed ^ 0x7f4a7c15);
}

export function hash2f(x: number, y: number, seed = 0): number {
  return hash2i(x, y, seed) / 4294967296;
}

export function hash1f(x: number, seed = 0): number {
  return hash1i(x, seed) / 4294967296;
}

export function rngAt(x: number, y: number, seed = 0): Rng {
  return mulberry32(hash2i(x, y, seed) | 0);
}

export function streamFor(seed: number, name: string): Rng {
  let h = seed | 0;
  for (let i = 0; i < name.length; i++) {
    h = Math.imul(h ^ name.charCodeAt(i), 0x01000193) | 0;
  }
  return mulberry32(h);
}
