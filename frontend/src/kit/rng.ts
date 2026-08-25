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
