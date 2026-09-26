import * as THREE from 'three/webgpu';
import { abs, float, floor, fwidth, log2, max, min, mod, select, step, texture, vec2 } from 'three/tsl';

type N<T extends string> = THREE.Node<T>;
type Scalar = N<'float'> | number;

const CELL = 96;
const SPREAD = 14;
const FONT = '800 {size}px ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const FAR = 1e5;

export interface Numerals {
  texture: THREE.DataTexture;
  advance: number;
}

function transform(grid: Float64Array, width: number, height: number): void {
  const n = Math.max(width, height), f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  const line = (offset: number, stride: number, length: number) => {
    for (let i = 0; i < length; i++) f[i] = grid[offset + i * stride]!;
    const parabola = (q: number, k: number) => (f[q]! + q * q - (f[v[k]!]! + v[k]! * v[k]!)) / (2 * q - 2 * v[k]!);
    let k = 0;
    v[0] = 0;
    z[0] = -Infinity;
    z[1] = Infinity;
    for (let q = 1; q < length; q++) {
      let s = parabola(q, k);
      while (s <= z[k]!) s = parabola(q, --k);
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
  for (let x = 0; x < width; x++) line(x, width, height);
  for (let y = 0; y < height; y++) line(y * width, 1, width);
}

export function numerals(): Numerals {
  const width = 10 * CELL, height = CELL, canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Generals numerals need a 2D canvas');
  context.font = FONT.replace('{size}', '100');
  const probe = context.measureText('0123456789'), cap = probe.actualBoundingBoxAscent + probe.actualBoundingBoxDescent;
  context.font = FONT.replace('{size}', String(100 * 0.6 * CELL / cap));
  const fit = context.measureText('0123456789'), advance = context.measureText('0').width / CELL;
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  const baseline = CELL / 2 + (fit.actualBoundingBoxAscent - fit.actualBoundingBoxDescent) / 2;
  for (let digit = 0; digit < 10; digit++) context.fillText(String(digit), (digit + 0.5) * CELL, baseline);
  const pixels = context.getImageData(0, 0, width, height).data;
  const outer = new Float64Array(width * height), inner = new Float64Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const a = pixels[i * 4 + 3]! / 255;
    outer[i] = a >= 1 ? 0 : a <= 0 ? FAR : Math.max(0, 0.5 - a) ** 2;
    inner[i] = a >= 1 ? FAR : a <= 0 ? 0 : Math.max(0, a - 0.5) ** 2;
  }
  transform(outer, width, height);
  transform(inner, width, height);
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

export const digitsOf = (value: N<'float'>): N<'float'> =>
  float(1).add(step(10, value)).add(step(100, value)).add(step(1e3, value)).add(step(1e4, value)).add(step(1e5, value)).add(step(1e6, value)).add(step(1e7, value));

export function numeral(set: Numerals, p: N<'vec2'>, value: N<'float'>, em: Scalar, room: Scalar, anchor = 0): N<'float'> {
  const count = digitsOf(value), advance = set.advance, span = count.mul(advance);
  const unit = min(typeof em === 'number' ? float(em) : em, (typeof room === 'number' ? float(room) : room).div(span)), q = p.div(unit).add(vec2(span.mul(anchor * 0.5), 0));
  const along = q.x.add(span.mul(0.5)), slot = floor(along.div(advance)), side = select(along.div(advance).sub(slot).lessThan(0.5), float(-1), float(1));
  const lod = log2(max(max(fwidth(q.x), fwidth(q.y)).mul(CELL), 1));
  const glyph = (k: N<'float'>): N<'float'> => {
    const local = vec2(along.sub(k.add(0.5).mul(advance)), q.y);
    const digit = mod(floor(value.add(0.5).div(float(10).pow(count.sub(1).sub(k)))), 10);
    const sample = texture(set.texture, vec2(digit.add(local.x).add(0.5).div(10), local.y.add(0.5))).level(lod).r;
    const inside = k.greaterThanEqual(0).and(k.lessThan(count)).and(abs(local.x).lessThan(0.5)).and(abs(local.y).lessThan(0.5));
    return select(inside, float(0.5).sub(sample).mul(2 * SPREAD / CELL), float(SPREAD / CELL));
  };
  return min(glyph(slot), glyph(slot.add(side))).mul(unit);
}
