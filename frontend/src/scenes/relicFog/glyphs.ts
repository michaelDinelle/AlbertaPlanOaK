import * as THREE from 'three/webgpu';
import { abs, float, floor, fwidth, log2, max, min, mod, select, step, texture, vec2 } from 'three/tsl';

type N<T extends string> = THREE.Node<T>;

const CELL = 96;
const SPREAD = 14;
const CHARS = '0123456789+−';
const FONT = '700 {size}px ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const FAR = 1e5;

const PLUS = 10;
const MINUS = 11;

export interface Glyphs {
  texture: THREE.DataTexture;
  advance: number;
}

function squaredDistance(grid: Float64Array, width: number, height: number): void {
  const n = Math.max(width, height), f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  const pass = (offset: number, stride: number, length: number) => {
    for (let i = 0; i < length; i++) f[i] = grid[offset + i * stride]!;
    const meet = (q: number, k: number) => (f[q]! + q * q - (f[v[k]!]! + v[k]! * v[k]!)) / (2 * q - 2 * v[k]!);
    let k = 0;
    v[0] = 0;
    z[0] = -Infinity;
    z[1] = Infinity;
    for (let q = 1; q < length; q++) {
      let s = meet(q, k);
      while (s <= z[k]!) s = meet(q, --k);
      v[++k] = q;
      z[k] = s;
      z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < length; q++) {
      while (z[k + 1]! < q) k++;
      d[q] = (q - v[k]!) ** 2 + f[v[k]!]!;
    }
    for (let i = 0; i < length; i++) grid[offset + i * stride] = d[i]!;
  };
  for (let x = 0; x < width; x++) pass(x, width, height);
  for (let y = 0; y < height; y++) pass(y * width, 1, width);
}

export function glyphs(): Glyphs {
  const width = CHARS.length * CELL, height = CELL, canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('RelicFog numerals need a 2D canvas');
  context.font = FONT.replace('{size}', '100');
  const probe = context.measureText('0123456789'), cap = probe.actualBoundingBoxAscent + probe.actualBoundingBoxDescent;
  context.font = FONT.replace('{size}', String(100 * 0.62 * CELL / cap));
  const fit = context.measureText('0123456789'), advance = context.measureText('0').width / CELL;
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  const baseline = CELL / 2 + (fit.actualBoundingBoxAscent - fit.actualBoundingBoxDescent) / 2;
  [...CHARS].forEach((char, k) => context.fillText(char, (k + 0.5) * CELL, baseline));
  const pixels = context.getImageData(0, 0, width, height).data, outer = new Float64Array(width * height), inner = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const a = pixels[i * 4 + 3]! / 255;
    outer[i] = a >= 1 ? 0 : a <= 0 ? FAR : Math.max(0, 0.5 - a) ** 2;
    inner[i] = a >= 1 ? FAR : a <= 0 ? 0 : Math.max(0, a - 0.5) ** 2;
  }
  squaredDistance(outer, width, height);
  squaredDistance(inner, width, height);
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = y * width + x, d = Math.sqrt(outer[i]!) - Math.sqrt(inner[i]!);
    data[(height - 1 - y) * width + x] = Math.round(255 * Math.min(1, Math.max(0, 0.5 - d / (2 * SPREAD))));
  }
  const atlas = new THREE.DataTexture(data, width, height, THREE.RedFormat, THREE.UnsignedByteType);
  atlas.minFilter = THREE.LinearMipmapLinearFilter;
  atlas.magFilter = THREE.LinearFilter;
  atlas.generateMipmaps = true;
  atlas.needsUpdate = true;
  return { texture: atlas, advance };
}

const sample = (set: Glyphs, local: N<'vec2'>, index: N<'float'>, lod: N<'float'>): N<'float'> => {
  const inside = abs(local.x).lessThan(0.5).and(abs(local.y).lessThan(0.5));
  const coverage = texture(set.texture, vec2(index.add(local.x).add(0.5).div(CHARS.length), local.y.add(0.5))).level(lod).r;
  return select(inside, float(0.5).sub(coverage).mul(2 * SPREAD / CELL), float(SPREAD / CELL));
};

const detail = (q: N<'vec2'>): N<'float'> => log2(max(max(fwidth(q.x), fwidth(q.y)).mul(CELL), 1));

export function numeral(set: Glyphs, p: N<'vec2'>, value: N<'float'>, em: number, anchor: -1 | 0 | 1 = 0, plus = false): N<'float'> {
  const magnitude = floor(abs(value).add(0.5)), negative = value.lessThan(-0.5);
  const signed = plus ? float(1) : select(negative, float(1), float(0)), prefix = select(negative, float(MINUS), float(PLUS));
  const digits = float(1).add(step(10, magnitude)).add(step(100, magnitude)).add(step(1e3, magnitude)).add(step(1e4, magnitude));
  const count = digits.add(signed), advance = set.advance, span = count.mul(advance);
  const q = p.div(em).add(vec2(span.mul(anchor * 0.5), 0)), along = q.x.add(span.mul(0.5)), slot = floor(along.div(advance));
  const neighbour = select(along.div(advance).sub(slot).lessThan(0.5), slot.sub(1), slot.add(1)), lod = detail(q);
  const at = (k: N<'float'>): N<'float'> => {
    const local = vec2(along.sub(k.add(0.5).mul(advance)), q.y);
    const digit = mod(floor(magnitude.div(float(10).pow(digits.sub(1).sub(k.sub(signed))))), 10);
    const index = select(k.lessThan(signed.sub(0.5)), prefix, digit);
    return select(k.greaterThanEqual(0).and(k.lessThan(count)), sample(set, local, index, lod), float(SPREAD / CELL));
  };
  return min(at(slot), at(neighbour)).mul(em);
}
