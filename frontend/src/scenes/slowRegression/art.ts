import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, fwidth, length, max, min, mix, mod, positionWorld, round, select, sin, smoothstep, sqrt, texture, uv, uvec2, vec2, vec3 } from 'three/tsl';
import { noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const PLUS = 0xff9a3d;
export const MINUS = 0xa88cff;
export const GOLD = 0xffd166;
export const ERROR = 0xff5a64;
export const INK = 0xd9ddd2;
export const SIGNAL = 0xfff0d0;
export const LED = 0.15;
export const CELL = 0.66;
export const BANDS = 256;

const PLUS_DARK = 0x3d2310;
const MINUS_DARK = 0x251d44;
const MASK = 0x0e2522;
const COPPER = 0x22493d;
const PAD = 0xb39556;
const FONT = 'system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const LIGHT = vec2(-0.6, 0.8);
const SLANT = 0.1;
const STROKE = 0.046;

export interface Box { x0: number; x1: number; y0: number; y1: number }

export interface Meter extends Box { cx: number; cy: number; radius: number; sweep: number }

export interface Layout {
  inputs: number;
  slow: number;
  hidden: number;
  cols: number;
  rows: number;
  scale: number;
  pitch: number;
  top: number;
  labelX: number;
  dip: readonly [number, number];
  slotX: number;
  travel: number;
  trace: readonly [number, number];
  matrix: Box;
  cell: number;
  sum: Box;
  readout: Box;
  meter: Meter;
  board: Box;
}

export function layout(inputs: number, slow: number, hidden: number, scale: number): Layout {
  const pitch = 0.3, top = 3.15, cell = 2 * pitch, cols = Math.ceil(Math.sqrt(hidden)), rows = Math.ceil(hidden / cols);
  const dip = [-7.35, -6.45] as const, matrix = { x0: -6.05, x1: -6.05 + cols * cell, y0: top - rows * cell, y1: top };
  const sum = { x0: matrix.x1 + 0.35, x1: matrix.x1 + 2.15, y0: matrix.y0, y1: top };
  const meter = { x0: sum.x1 + 0.35, x1: sum.x1 + 5.95, y0: matrix.y0, y1: 1.85, cx: 0, cy: matrix.y0 + 0.55, radius: 3.05, sweep: (46 * Math.PI) / 180 };
  meter.cx = (meter.x0 + meter.x1) / 2;
  return {
    inputs, slow, hidden, cols, rows, scale, pitch, top, cell,
    labelX: dip[0] - 0.1, dip, slotX: (dip[0] + dip[1]) / 2, travel: 0.25,
    trace: [dip[1] + 0.02, matrix.x1 + 0.08], matrix, sum,
    readout: { x0: meter.x0, x1: meter.x1, y0: meter.y1 + 0.3, y1: top }, meter,
    board: { x0: -8.3, x1: meter.x1 + 0.28, y0: top - (inputs + 1) * pitch - 0.5, y1: top + 0.85 },
  };
}

export const rowY = (l: Layout, j: number) => l.top - (j + 0.5) * l.pitch;
export const unitX = (l: Layout, j: number) => l.matrix.x0 + ((j % l.cols) + 0.5) * l.cell;
export const unitY = (l: Layout, j: number) => l.top - (Math.floor(j / l.cols) + 0.5) * l.cell;
export const angleOf = (l: Layout, value: number) => Math.max(-1, Math.min(1, value / l.scale)) * l.meter.sweep;
export const biasAt = (l: Layout): readonly [number, number] => [l.sum.x0 + 0.42, -0.05];
export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 32).rotateX(-Math.PI / 2);

const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const pixel = (): N<'float'> => max(fwidth(positionWorld.x), fwidth(positionWorld.z));
const crisp = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(pixel())), 0, 1);
const halo = (d: N<'float'>, reach: number): N<'float'> => exp(max(d, 0).div(-reach));
const ring = (d: N<'float'>, half: number): N<'float'> => abs(d).sub(half);
const ease = (a: number, b: number, x: N<'float'>): N<'float'> => smoothstep(a, b, x);

interface Paint { color: N<'vec3'>; alpha: N<'float'> }

const clear = (): Paint => ({ color: vec3(0), alpha: float(0) });

const over = (under: Paint, color: N<'vec3'>, alpha: N<'float'>): Paint => {
  const a = alpha.add(under.alpha.mul(alpha.oneMinus()));
  return { color: color.mul(alpha).add(under.color.mul(under.alpha).mul(alpha.oneMinus())).div(max(a, 1e-4)), alpha: a };
};

function sprite(width: number, depth: number, x: number, y: number, order: number, paint: Paint): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = paint.color;
  material.opacityNode = paint.alpha;
  const mesh = new THREE.Mesh(flat(width, depth), material);
  mesh.position.set(x, 0, -y);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

const boxOf = (q: N<'vec2'>, b: Box, radius: number, grow = 0) => sdBox(q.sub(vec2((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2)), vec2((b.x1 - b.x0) / 2 + grow, (b.y1 - b.y0) / 2 + grow), radius);

const bevel = (at: (p: N<'vec2'>) => N<'float'>, q: N<'vec2'>) => {
  const e = 0.01, n = vec2(at(q.add(vec2(e, 0))).sub(at(q.sub(vec2(e, 0)))), at(q.add(vec2(0, e))).sub(at(q.sub(vec2(0, e)))));
  return dot(n.div(max(length(n), 1e-5)), LIGHT);
};

export function bench(): THREE.Mesh {
  const q = chart(), grain = noise(q.mul(0.35)).mul(0.25).add(noise(q.mul(4.1)).mul(0.06)).add(1);
  return sprite(120, 120, 0, 0, 0, { color: rgb(0x050708).mul(grain), alpha: float(1) });
}

export function board(l: Layout, lit: Lit): THREE.Mesh {
  const q = chart(), b = l.board, at = (p: N<'vec2'>) => boxOf(p, b, 0.3), shell = at(q);
  const facing = bevel(at, q), rim = smoothstep(-0.09, 0, shell);
  const weave = noise(q.mul(vec2(34, 31))).mul(0.03).add(noise(q.mul(2.6)).mul(0.05)).add(noise(q.mul(0.4)).mul(0.07)).add(1);
  let color: N<'vec3'> = rgb(MASK).mul(weave).add(vec3(0.07, 0.1, 0.09).mul(rim).mul(facing.max(0))).mul(float(1).sub(rim.mul(facing.min(0).abs()).mul(0.6)));
  const edge = crisp(ring(shell.add(0.018), 0.014));
  color = mix(color, rgb(0x4a3f2a), edge.mul(0.8));

  const pour = boxOf(q, b, 0.22, -0.2), rail = crisp(ring(pour, 0.02));
  color = mix(color, rgb(COPPER), rail);
  const perimeter = select(abs(q.y.sub((b.y0 + b.y1) / 2)).greaterThan((b.y1 - b.y0) / 2 - 0.3), q.x, q.y);
  const via = sdCircle(vec2(mod(perimeter.add(0.25), 0.5).sub(0.25), pour), 0.035);
  color = mix(color, rgb(PAD).mul(0.8), crisp(via).mul(crisp(ring(pour, 0.06))));
  color = mix(color, rgb(0x050606), crisp(via.add(0.018)).mul(crisp(ring(pour, 0.06))));

  const silk = (d: N<'float'>, strength = 0.75) => { color = mix(color, rgb(INK), crisp(ring(d, 0.009)).mul(strength)); };
  const m = l.matrix, mx = (m.x0 + m.x1) / 2, my = (m.y0 + m.y1) / 2, hx = (m.x1 - m.x0) / 2 + 0.12, hy = (m.y1 - m.y0) / 2 + 0.12;
  const corner = vec2(abs(q.x.sub(mx)).sub(hx), abs(q.y.sub(my)).sub(hy));
  const bracket = min(max(abs(corner.x).sub(0.009), max(corner.y.negate().sub(0.4), corner.y.sub(0.009))), max(abs(corner.y).sub(0.009), max(corner.x.negate().sub(0.4), corner.x.sub(0.009))));
  color = mix(color, rgb(INK), crisp(bracket).mul(0.75));

  silk(boxOf(q, l.sum, 0.12), 0.55);
  silk(boxOf(q, l.readout, 0.12, -0.02), 0.55);
  const [bx, by] = biasAt(l);
  color = mix(color, rgb(PAD), crisp(sdCircle(q.sub(vec2(bx, by)), LED + 0.05)).mul(0.6));

  for (const [x, y] of [[b.x0 + 0.34, b.y0 + 0.34], [b.x1 - 0.34, b.y0 + 0.34], [b.x0 + 0.34, b.y1 - 0.34], [b.x1 - 0.34, b.y1 - 0.34]] as const) {
    const p = q.sub(vec2(x, y)), r = length(p), unit = p.div(max(r, 1e-4));
    color = mix(color, rgb(INK), crisp(ring(r.sub(0.25), 0.008)).mul(0.6));
    color = mix(color, rgb(PAD).mul(dot(unit, LIGHT).mul(0.25).add(0.9)), crisp(r.sub(0.18)));
    color = mix(color, rgb(0x020303), crisp(r.sub(0.1)));
  }
  return sprite(b.x1 - b.x0 + 0.4, b.y1 - b.y0 + 0.4, (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, 1, { color: lit(color), alpha: crisp(shell) });
}

export function traces(l: Layout, bits: THREE.DataTexture, lit: Lit): THREE.Mesh {
  const q = chart(), store = texture(bits), [x0, x1] = l.trace, first = l.top - l.pitch / 2;
  const j = clamp(round(float(first).sub(q.y).div(l.pitch)), 0, l.inputs - 1), y = float(first).sub(j.mul(l.pitch));
  const data = store.load(uvec2(vec2(j, 0))), along = q.x.sub(x0).div(x1 - x0);
  const line = max(abs(q.y.sub(y)).sub(0.016), max(float(x0).sub(q.x), q.x.sub(x1)));
  const reach = ease(0, 0.04, data.y.sub(along)), hot = data.z.mul(reach), flare = data.w;
  let paint = clear();
  paint = over(paint, lit(rgb(COPPER).mul(1.15)), crisp(line));
  paint = over(paint, rgb(SIGNAL).mul(0.4), halo(line, 0.04).mul(max(hot, flare)).mul(0.18));
  paint = over(paint, mix(rgb(SIGNAL).mul(0.62), rgb(GOLD).mul(1.5), flare), crisp(line.add(0.006)).mul(max(hot.mul(0.62), flare.mul(0.9))));
  const pad = sdBox(q.sub(vec2(x0 + 0.05, y)), vec2(0.06, 0.05), 0.012), end = length(q.sub(vec2(x1, y)));
  paint = over(paint, lit(rgb(PAD)), crisp(pad));
  paint = over(paint, rgb(SIGNAL).mul(1.05), crisp(pad.add(0.018)).mul(hot).mul(0.85));
  paint = over(paint, lit(rgb(PAD).mul(0.85)), crisp(end.sub(0.05)));
  paint = over(paint, rgb(0x040505), crisp(end.sub(0.024)));
  const width = x1 - x0 + 0.3, height = l.inputs * l.pitch + 0.2;
  return sprite(width, height, (x0 + x1) / 2, l.top - (l.inputs * l.pitch) / 2, 2, paint);
}

export function dips(l: Layout, bits: THREE.DataTexture, lit: Lit): THREE.Mesh {
  const q = chart(), store = texture(bits), [d0, d1] = l.dip, first = l.top - l.pitch / 2;
  const j = clamp(round(float(first).sub(q.y).div(l.pitch)), 0, l.inputs - 1), y = float(first).sub(j.mul(l.pitch));
  const data = store.load(uvec2(vec2(j, 0)));
  const bodyOf = (from: number, to: number) => {
    const top = rowY(l, from) + 0.13, bottom = rowY(l, to) - 0.13;
    const at = (p: N<'vec2'>) => sdBox(p.sub(vec2((d0 + d1) / 2, (top + bottom) / 2)), vec2((d1 - d0) / 2, (top - bottom) / 2), 0.035);
    return { d: at(q), facing: bevel(at, q) };
  };
  const slow = bodyOf(0, l.slow - 1), fast = bodyOf(l.slow, l.inputs - 1);
  let paint = clear();
  for (const body of [slow, fast]) {
    const rim = smoothstep(-0.05, 0, body.d), grain = noise(q.mul(40)).mul(0.03).add(1);
    const shade = rgb(0x22272b).mul(grain).add(vec3(0.12).mul(rim).mul(body.facing.max(0))).mul(float(1).sub(rim.mul(body.facing.min(0).abs()).mul(0.5)));
    paint = over(paint, vec3(0), halo(body.d.add(0.01), 0.06).mul(0.5));
    paint = over(paint, lit(shade), crisp(body.d));
  }
  const p = vec2(q.x.sub(l.slotX), q.y.sub(y));
  const slot = sdBox(p, vec2(l.travel / 2 + 0.115, 0.082), 0.02), depth = ease(-0.082, 0.082, p.y);
  paint = over(paint, mix(rgb(0x0b0c0d), rgb(0x020303), depth), crisp(slot));
  const knob = vec2(p.x.add(l.travel / 2).sub(data.x.mul(l.travel)), p.y);
  paint = over(paint, vec3(0), halo(sdBox(knob.sub(vec2(0.012, -0.018)), vec2(0.1, 0.058), 0.022), 0.025).mul(crisp(slot)).mul(0.8));
  const slider = sdBox(knob, vec2(0.1, 0.058), 0.022), ridge = abs(abs(knob.x).sub(0.035)).sub(0.007);
  const ivory = mix(rgb(0xd9d1bd), rgb(0xf6f0e2), ease(-0.058, 0.04, knob.y)).mul(select(ridge.lessThan(0), float(0.86), float(1)));
  paint = over(paint, lit(ivory), crisp(slider));
  const flare = data.w;
  paint = over(paint, rgb(GOLD).mul(1.3), halo(ring(slot, 0.03), 0.05).mul(flare).mul(0.7));
  paint = over(paint, rgb(GOLD).mul(1.6), crisp(ring(slot.sub(0.035), 0.012)).mul(flare));
  return sprite(d1 - d0 + 0.4, l.inputs * l.pitch + 0.4, (d0 + d1) / 2, l.top - (l.inputs * l.pitch) / 2, 4, paint);
}

function lens(under: Paint, p: N<'vec2'>, plus: N<'float'>, level: N<'float'>, ember: N<'float'>, lit: Lit): Paint {
  const r = length(p), hue = mix(rgb(MINUS), rgb(PLUS), plus), dark = mix(rgb(MINUS_DARK), rgb(PLUS_DARK), plus);
  const inside = crisp(r.sub(LED)), dome = sqrt(max(float(1).sub(r.div(LED).mul(r.div(LED))), 0));
  let paint = under;
  paint = over(paint, vec3(0), halo(length(p.sub(vec2(0.022, -0.03))).sub(LED), 0.04).mul(0.6));
  paint = over(paint, lit(dark.mul(0.55)), crisp(r.sub(LED + 0.022)));
  paint = over(paint, lit(mix(dark.mul(0.6), dark.mul(1.5), dome)), inside);
  paint = over(paint, lit(rgb(0x121212)), crisp(sdBox(p, vec2(0.026, 0.026), 0.005)).mul(0.75));
  paint = over(paint, hue.mul(0.42), inside.mul(ember).mul(dome.mul(0.6).add(0.4)));
  paint = over(paint, hue.mul(level.mul(1.6).add(0.9)), inside.mul(level).mul(dome.mul(0.45).add(0.55)));
  paint = over(paint, hue.mul(1.6).add(vec3(0.9)), crisp(r.sub(LED * 0.38)).mul(level));
  paint = over(paint, hue.mul(1.1), halo(r.sub(LED), 0.04).mul(level).mul(0.55));
  paint = over(paint, vec3(1), exp(length(p.sub(vec2(-0.05, 0.055))).div(-0.022)).mul(inside).mul(0.4));
  return paint;
}

export function cover(l: Layout, lit: Lit): THREE.Mesh {
  const q = chart(), m = l.matrix, at = (p: N<'vec2'>) => boxOf(p, m, 0.1, 0.06), d = at(q), facing = bevel(at, q);
  const sheen = ease(0.35, 0, abs(fract(q.x.add(q.y.mul(0.6)).mul(0.11)).sub(0.5))).mul(0.018);
  const edge = crisp(ring(d, 0.012)).mul(facing.mul(0.5).add(0.5));
  let paint = clear();
  paint = over(paint, lit(rgb(0x04090a)).add(vec3(sheen)), crisp(d).mul(0.55));
  paint = over(paint, lit(rgb(0x5d6b70)), edge.mul(0.8));
  return sprite(m.x1 - m.x0 + 0.5, m.y1 - m.y0 + 0.5, (m.x0 + m.x1) / 2, (m.y0 + m.y1) / 2, 3, paint);
}


export function matrix(l: Layout, units: THREE.DataTexture, lit: Lit): THREE.Mesh {
  const q = chart(), m = l.matrix, store = texture(units);
  const col = clamp(floor(q.x.sub(m.x0).div(l.cell)), 0, l.cols - 1), row = clamp(floor(float(m.y1).sub(q.y).div(l.cell)), 0, l.rows - 1);
  const j = row.mul(l.cols).add(col), centre = vec2(float(m.x0).add(col.add(0.5).mul(l.cell)), float(m.y1).sub(row.add(0.5).mul(l.cell)));
  const data = store.load(uvec2(vec2(j, 0)));
  const paint = lens(clear(), q.sub(centre), data.y, data.x, data.z, lit);
  return sprite(m.x1 - m.x0 + 0.2, m.y1 - m.y0 + 0.2, (m.x0 + m.x1) / 2, (m.y0 + m.y1) / 2, 5, { color: paint.color, alpha: paint.alpha.mul(data.w) });
}

export function biasLed(l: Layout, plus: Level, lit: Lit): THREE.Mesh {
  const [x, y] = biasAt(l), paint = lens(clear(), chart().sub(vec2(x, y)), plus, float(1), float(0), lit);
  return sprite(0.6, 0.6, x, y, 5, paint);
}

export function emitters(l: Layout, units: THREE.DataTexture): { geometry: THREE.BufferGeometry; tint: N<'vec3'> } {
  const segments = 20, per = segments + 1, position = new Float32Array(l.hidden * per * 3), index = new Float32Array(l.hidden * per), order: number[] = [];
  for (let j = 0; j < l.hidden; j++) {
    const x = unitX(l, j), z = -unitY(l, j), base = j * per;
    position.set([x, 0, z], base * 3);
    index[base] = j;
    for (let s = 0; s < segments; s++) {
      const a = (s / segments) * 2 * Math.PI;
      position.set([x + Math.cos(a) * LED * 0.8, 0, z - Math.sin(a) * LED * 0.8], (base + 1 + s) * 3);
      index[base + 1 + s] = j;
      order.push(base, base + 1 + s, base + 1 + ((s + 1) % segments));
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('aUnit', new THREE.BufferAttribute(index, 1));
  geometry.setIndex(order);
  const data = texture(units).load(uvec2(vec2(attribute<'float'>('aUnit', 'float'), 0)));
  return { geometry, tint: mix(rgb(MINUS), rgb(PLUS), data.y).mul(data.x.mul(data.w)) };
}

const GLYPHS: Record<string, number> = { '0': 0x3f, '1': 0x06, '2': 0x5b, '3': 0x4f, '4': 0x66, '5': 0x6d, '6': 0x7d, '7': 0x07, '8': 0x7f, '9': 0x6f, '-': 0x40, '+': 0x140, ' ': 0 };

export class Digits {
  readonly data: Float32Array;
  readonly texture: THREE.DataTexture;
  private used = 0;

  constructor(capacity: number) {
    this.data = new Float32Array(capacity * 4);
    this.texture = new THREE.DataTexture(this.data, capacity, 1, THREE.RGBAFormat, THREE.FloatType);
    this.texture.needsUpdate = true;
  }

  claim(cells: number): number {
    const at = this.used;
    this.used += cells;
    if (this.used * 4 > this.data.length) throw new Error(`SlowRegression readouts need ${this.used} digit cells`);
    return at;
  }

  write(offset: number, cells: number, text: string): void {
    const glyphs: number[] = [];
    for (const ch of text) {
      if (ch === '.') {
        if (glyphs.length === 0) throw new Error(`Readout text ${text} starts with a decimal point`);
        glyphs[glyphs.length - 1]! |= 0x80;
        continue;
      }
      const glyph = GLYPHS[ch];
      if (glyph === undefined) throw new Error(`Readout cannot show ${ch}`);
      glyphs.push(glyph);
    }
    if (glyphs.length > cells) throw new Error(`Readout text ${text} needs ${glyphs.length} of ${cells} cells`);
    let changed = false;
    for (let c = 0; c < cells; c++) {
      const glyph = glyphs[glyphs.length - cells + c] ?? 0, at = (offset + c) * 4;
      if (this.data[at] !== glyph) { this.data[at] = glyph; changed = true; }
    }
    if (changed) this.texture.needsUpdate = true;
  }
}

function segments(p: N<'vec2'>): N<'float'>[] {
  const s = vec2(p.x.sub(p.y.mul(SLANT)), p.y);
  const across = (y: number) => sdBox(s.sub(vec2(0, y)), vec2(0.15, STROKE), STROKE);
  const down = (x: number, y: number) => sdBox(s.sub(vec2(x, y)), vec2(STROKE, 0.15), STROKE);
  return [across(0.4), down(0.2, 0.2), down(0.2, -0.2), across(-0.4), down(-0.2, -0.2), down(-0.2, 0.2), across(0), sdCircle(p.sub(vec2(0.3, -0.41)), 0.05), sdBox(s, vec2(STROKE, 0.14), STROKE)];
}

export function sevenSeg(x: number, y: number, cells: number, height: number, digits: Digits, offset: number, tint: N<'vec3'>, order: number): THREE.Mesh {
  const width = height * CELL, q = chart(), left = x - (cells * width) / 2, u = q.x.sub(left);
  const k = clamp(floor(u.div(width)), 0, cells - 1), p = vec2(u.sub(k.add(0.5).mul(width)), q.y.sub(y)).div(height);
  const mask = texture(digits.texture).load(uvec2(vec2(k.add(offset), 0))).x;
  let on: N<'float'> = float(0), off: N<'float'> = float(0), near: N<'float'> = float(1e3);
  segments(p).forEach((d, b) => {
    const bit = mod(floor(mask.div(2 ** b)), 2), c = crisp(d.mul(height));
    on = max(on, c.mul(bit));
    if (b < 7) off = max(off, c.mul(bit.oneMinus()));
    near = min(near, select(bit.greaterThan(0.5), d.mul(height), float(1e3)));
  });
  const window = sdBox(q.sub(vec2(x, y)), vec2((cells * width) / 2 + height * 0.1, height * 0.62), height * 0.08);
  let paint = clear();
  paint = over(paint, rgb(0x5a6268), crisp(ring(window, 0.008)).mul(0.5));
  paint = over(paint, rgb(0x030405).add(tint.mul(0.012)), crisp(window));
  paint = over(paint, tint.mul(0.05), off);
  paint = over(paint, tint.mul(0.7), halo(near, height * 0.09).mul(0.35).mul(crisp(window)));
  paint = over(paint, tint.mul(1.7), on);
  return sprite(cells * width + height * 0.4, height * 1.5, x, y, order, paint);
}

export interface Needle { value: Level; target: Level; on: Level; over: Level }

export function meter(l: Layout, levels: Needle, bands: THREE.DataTexture, accent: number, lit: Lit): THREE.Mesh {
  const q = chart(), m = l.meter, R = m.radius, sweep = m.sweep, S = l.scale, store = texture(bands);
  const at = (p: N<'vec2'>) => boxOf(p, m, 0.22), face = at(q), facing = bevel(at, q);
  const p = q.sub(vec2(m.cx, m.cy)), r = length(p), bearing = atan(p.x, p.y), inArc = abs(bearing).sub(sweep);
  const radial = (angle: N<'float'>) => vec2(sin(angle), cos(angle));
  const angle = (value: N<'float'>) => clamp(value.div(S), -1, 1).mul(sweep);
  const vignette = ease(0.2, 4.2, length(q.sub(vec2(m.cx, m.cy + 1.4))));
  let paint = clear();
  paint = over(paint, vec3(0), halo(face.add(0.02), 0.12).mul(0.6));
  const rim = smoothstep(-0.07, 0, face);
  const bezel = rgb(0x1c2024).add(vec3(0.14).mul(rim).mul(facing.max(0))).mul(float(1).sub(rim.mul(facing.min(0).abs()).mul(0.6)));
  paint = over(paint, lit(bezel), crisp(face));
  const inner = face.add(0.09);
  paint = over(paint, lit(mix(rgb(0x121619), rgb(0x080a0c), vignette)), crisp(inner));

  const value = bearing.div(sweep).mul(S), k = clamp(round(value), -S, S), tickAngle = k.div(S).mul(sweep);
  const major = mod(abs(k), 6).lessThan(0.5), zero = abs(k).lessThan(0.5);
  const reach = select(zero, float(0.26), select(major, float(0.2), float(0.1)));
  const across = abs(bearing.sub(tickAngle)).mul(r).sub(select(major, float(0.011), float(0.006)));
  const tick = max(across, abs(r.sub(float(R).add(reach.mul(0.5)))).sub(reach.mul(0.5)));
  paint = over(paint, lit(rgb(0xe6e1d3)), crisp(tick).mul(select(major, float(0.95), float(0.62))));
  paint = over(paint, lit(rgb(0xe6e1d3)), crisp(max(ring(r.sub(R), 0.006), inArc.mul(r))).mul(0.8));
  const mirror = max(ring(r.sub(R - 0.03), 0.012), inArc.mul(r));
  paint = over(paint, mix(rgb(0x2d3438), rgb(0x86939a), ease(-0.8, 0.8, bearing)), crisp(mirror).mul(0.55));

  const bin = clamp(round(value), -S, S), share = store.load(uvec2(vec2(bin.add(S), 0))), gap = abs(bearing.sub(bin.div(S).mul(sweep))).mul(r).sub((sweep / S) * R * 0.36);
  const band = (from: number, to: number) => max(gap, max(abs(r.sub((from + to) / 2)).sub((to - from) / 2), inArc.mul(r)));
  const predictions = band(R - 0.27, R - 0.09), targets = band(R - 0.52, R - 0.34);
  paint = over(paint, rgb(0x1d2328), crisp(predictions).mul(0.9));
  paint = over(paint, rgb(0x1d2328), crisp(targets).mul(0.9));
  paint = over(paint, rgb(accent).mul(share.y.mul(1.2).add(0.15)), crisp(predictions).mul(sqrt(share.y)));
  paint = over(paint, rgb(GOLD).mul(share.x.mul(1.2).add(0.15)), crisp(targets).mul(sqrt(share.x)));

  const aim = angle(levels.value), goal = angle(levels.target), low = min(aim, goal), high = max(aim, goal);
  const wedge = max(max(low.sub(bearing), bearing.sub(high)).mul(r), max(float(0.36).sub(r), r.sub(R - 0.6)));
  const spread = ease(0.001, 0.02, high.sub(low));
  paint = over(paint, rgb(ERROR).mul(0.85), crisp(wedge).mul(levels.on).mul(spread).mul(r.div(R - 0.6).mul(0.12).add(0.06)));
  const lip = max(max(low.sub(bearing), bearing.sub(high)).mul(r), abs(r.sub(R - 0.6)).sub(0.012));
  paint = over(paint, rgb(ERROR).mul(1.4), crisp(lip).mul(levels.on).mul(spread));

  const flag = radial(goal), side = vec2(flag.y, flag.x.negate()), f = vec2(dot(p, side), dot(p, flag).sub(R + 0.22));
  const pointer = max(abs(f.x).mul(0.2).sub(f.y.mul(0.085)).div(0.2173), f.y.sub(0.2));
  paint = over(paint, rgb(GOLD).mul(0.9), halo(pointer, 0.06).mul(0.45));
  paint = over(paint, rgb(0x120c02), crisp(pointer.sub(0.014)));
  paint = over(paint, rgb(GOLD).mul(1.35), crisp(pointer));
  paint = over(paint, rgb(GOLD).mul(1.2), crisp(max(abs(f.x).sub(0.007), abs(f.y.add(0.37)).sub(0.37))).mul(0.7));
  paint = over(paint, rgb(GOLD).mul(1.3), crisp(sdCircle(f.sub(vec2(0, 0.3)), 0.035)).mul(levels.over));

  const dir = radial(aim), tip = dir.mul(R + 0.12), normal = vec2(dir.y, dir.x.negate());
  const tail = min(sdSegment(p, vec2(0), dir.mul(-0.3), 0.016), sdBox(vec2(dot(p, normal), dot(p, dir).add(0.33)), vec2(0.06, 0.04), 0.015));
  const shaft = min(sdSegment(p.sub(vec2(0.03, -0.045)), vec2(0), tip, 0.012), tail);
  paint = over(paint, vec3(0), halo(shaft, 0.035).mul(0.55).mul(levels.on));
  paint = over(paint, lit(rgb(0x2c3137)), crisp(tail).mul(levels.on));
  const needle = sdSegment(p, vec2(0), tip, float(0.01).add(float(0.012).mul(float(1).sub(clamp(dot(p, dir).div(R), 0, 1)))));
  paint = over(paint, rgb(accent).mul(0.9), halo(needle, 0.04).mul(0.45).mul(levels.on));
  paint = over(paint, rgb(accent).mul(1.45), crisp(needle).mul(levels.on));

  const unit = p.div(max(r, 1e-4)), shine = dot(unit, LIGHT).mul(0.5).add(0.5);
  paint = over(paint, vec3(0), halo(length(p.sub(vec2(0.03, -0.05))).sub(0.24), 0.05).mul(0.6));
  paint = over(paint, lit(mix(rgb(0x16191c), rgb(0x6c737b), shine)), crisp(r.sub(0.24)));
  paint = over(paint, lit(mix(rgb(0x0c0e10), rgb(0x2f3439), shine.oneMinus())), crisp(r.sub(0.17)));
  paint = over(paint, rgb(accent).mul(0.9), crisp(r.sub(0.05)).mul(levels.on));
  return sprite(m.x1 - m.x0 + 0.6, m.y1 - m.y0 + 0.6, (m.x0 + m.x1) / 2, (m.y0 + m.y1) / 2, 7, paint);
}

export interface Label { mesh: THREE.Mesh; width: number; texture: THREE.DataTexture }

export function label(message: string, height: number, tint: number | N<'vec3'>, lit?: Lit, fade?: Level, weight = 600): Label {
  const px = 112, canvas = document.createElement('canvas'), context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('SlowRegression labels need a 2D canvas');
  const font = `${weight} ${px}px ${FONT}`;
  context.font = font;
  context.letterSpacing = `${px * 0.06}px`;
  const w = Math.ceil(context.measureText(message).width + px * 0.4), h = Math.ceil(px * 1.4);
  canvas.width = w;
  canvas.height = h;
  context.font = font;
  context.letterSpacing = `${px * 0.06}px`;
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(message, w / 2, h / 2);
  const pixels = context.getImageData(0, 0, w, h).data, alpha = new Uint8Array(w * h);
  for (let row = 0; row < h; row++) for (let column = 0; column < w; column++) alpha[(h - 1 - row) * w + column] = pixels[(row * w + column) * 4 + 3]!;
  const map = new THREE.DataTexture(alpha, w, h, THREE.RedFormat, THREE.UnsignedByteType);
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.generateMipmaps = true;
  map.needsUpdate = true;
  const tall = height * 1.4, wide = (tall * w) / h, ink = typeof tint === 'number' ? rgb(tint) : tint;
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit ? lit(ink) : ink;
  material.opacityNode = fade ? texture(map, uv()).r.mul(fade) : texture(map, uv()).r;
  const mesh = new THREE.Mesh(flat(wide, tall), material);
  mesh.renderOrder = 8;
  mesh.frustumCulled = false;
  return { mesh, width: wide, texture: map };
}
