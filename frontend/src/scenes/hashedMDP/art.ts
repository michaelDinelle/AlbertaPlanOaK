import * as THREE from 'three/webgpu';
import { abs, attribute, clamp, exp, exp2, float, floor, fract, fwidth, length, max, min, mix, mod, positionWorld, pow, select, smoothstep, texture, uv, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const SIDE = 16;
export const HALF = SIDE / 2;
export const DEPTH = 4;
export const LAMP_GAP = 0.66;
export const TAPE_X0 = HALF + 1.4;
export const TAPE_X1 = HALF + 4.4;
export const TAPE_Z0 = -HALF - DEPTH;
export const TAPE_Z1 = HALF;
export const SPINE = (TAPE_X0 + TAPE_X1) / 2;
export const ROW = 0.5;
export const ROWS = Math.round((TAPE_Z1 - TAPE_Z0) / ROW);
export const REACH = (TAPE_X1 - TAPE_X0) / 2 - 0.22;
export const PANEL_X0 = -HALF - DEPTH - 4.9;
export const PANEL_X1 = -HALF - DEPTH - 1.1;
export const KEY = 2.1;
export const KEY_X = (PANEL_X0 + PANEL_X1) / 2;
export const KEY_SLOT = (TAPE_Z1 - TAPE_Z0) / 4;
export const SHARE = KEY;
export const BIAS = -HALF - 0.62;
export const DIGIT_W = 0.78;
export const DIGIT_H = 1.36;
export const DISPLAY_X = -HALF - DEPTH / 2 - 0.25;
export const DISPLAY_Z = -HALF - DEPTH / 2 - 0.3;
const DISPLAY_HALF = new THREE.Vector2(1.55, 1.05);
const DIE_X0 = PANEL_X0 - 0.8;
const DIE_X1 = TAPE_X1 + 0.8;
const DIE_Z0 = -HALF - DEPTH - 0.95;
const DIE_Z1 = HALF + LAMP_GAP + 0.95;
export const EXTENT = new THREE.Box2(new THREE.Vector2(DIE_X0 - 0.1, DIE_Z0 - 0.1), new THREE.Vector2(DIE_X1 + 0.1, DIE_Z1 + 0.1));

export const HEX = [0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111, 0b1111111, 0b1101111, 0b1110111, 0b1111100, 0b0111001, 0b1011110, 0b1111001, 0b1110001] as const;

export interface Layout {
  bits: number;
  levels: number;
  span: number;
  cell: number;
  band: number;
  lamp: number;
  digits: number;
}

export function layoutOf(bits: number): Layout {
  const levels = bits / 2, span = 2 ** levels, band = DEPTH / levels;
  return { bits, levels, span, cell: SIDE / span, band, lamp: Math.min(0.3, band * 0.36), digits: Math.ceil(levels / 4) };
}

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export function sheet(additive = false): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  if (additive) material.blending = THREE.AdditiveBlending;
  return material;
}

const box = (q: N<'vec2'>, x0: number, z0: number, x1: number, z1: number, radius: number): N<'float'> =>
  sdBox(q.sub(vec2((x0 + x1) / 2, (z0 + z1) / 2)), vec2((x1 - x0) / 2, (z1 - z0) / 2), radius);

export function keyZ(action: number): number {
  return TAPE_Z0 + (action + 0.42) * KEY_SLOT;
}

export function shareZ(action: number): number {
  return keyZ(action) + KEY / 2 + 0.62;
}

export function groundMaterial(layout: Layout, lit: Lit, actions: number): THREE.MeshBasicNodeMaterial {
  const q = vec2(positionWorld.x, positionWorld.z);
  const px = fwidth(q.x);
  const hair = (d: N<'float'>, width: number) => coverage(abs(d).sub(px.mul(width)));
  const die = box(q, DIE_X0, DIE_Z0, DIE_X1, DIE_Z1, 0.6);
  const mottle = fractal(q.mul(0.12), 3).mul(0.5).add(0.5);
  const sheen = smoothstep(0.15, 1.0, abs(fract(q.x.mul(0.8).add(q.y).mul(0.03)).sub(0.5)).mul(2)).mul(0.12);
  const tooth = noise(q.mul(13)).mul(0.04).add(1);
  const silicon = mix(rgb(0x2b2735), rgb(0x37323f), mottle).mul(tooth).add(rgb(0x4a4558).mul(sheen));
  const board = rgb(0x09080c).mul(noise(q.mul(2.3)).mul(0.1).add(1));
  const seal = max(hair(die.add(0.2), 0.9), hair(die.add(0.36), 0.6).mul(0.6));
  const core = box(q, -HALF - DEPTH - 0.18, -HALF - DEPTH - 0.18, HALF + 0.18, HALF + 0.18, 0.3);
  const tape = box(q, TAPE_X0, TAPE_Z0 - 0.18, TAPE_X1, TAPE_Z1 + 0.18, 0.3);
  const panel = box(q, PANEL_X0, TAPE_Z0 - 0.18, PANEL_X1, TAPE_Z1 + 0.18, 0.3);
  const well = min(min(core, tape), panel);
  const wellFill = rgb(0x15131c).mul(tooth);
  const seam = max(hair(q.x.add(HALF), 0.55), hair(q.y.add(HALF), 0.55)).mul(coverage(core));
  const spine = hair(q.x.sub(SPINE), 0.6).mul(coverage(tape.add(0.12)));
  const nowAt = q.sub(vec2(TAPE_X0 + 0.2, TAPE_Z1 - ROW / 2));
  const caret = coverage(min(sdSegment(nowAt, vec2(-0.09, -0.1), vec2(0.03, 0), 0.022), sdSegment(nowAt, vec2(-0.09, 0.1), vec2(0.03, 0), 0.022)));
  const display = sdBox(q.sub(vec2(DISPLAY_X, DISPLAY_Z)), vec2(DISPLAY_HALF.x, DISPLAY_HALF.y), 0.18);
  const wire = (index: N<'float'>, value: N<'float'>, along: N<'float'>) => {
    const centre = clamp(floor(index.div(layout.band)), 0, layout.levels - 1).add(0.5).mul(layout.band).add(-HALF - DEPTH);
    return hair(value.sub(centre), 0.6).mul(coverage(along));
  };
  const columnWire = wire(q.y.sub(-HALF - DEPTH), q.y, box(q, HALF + 0.18, -HALF - DEPTH, HALF + LAMP_GAP, -HALF, 0));
  const rowWire = wire(q.x.sub(-HALF - DEPTH), q.x, box(q, -HALF - DEPTH, HALF + 0.18, -HALF, HALF + LAMP_GAP, 0));
  let sockets: N<'float'> = float(1e3);
  for (let a = 0; a < actions; a++) sockets = min(sockets, sdBox(q.sub(vec2(KEY_X, keyZ(a))), vec2(KEY / 2 + 0.12), 0.34));
  let tracks: N<'float'> = float(1e3);
  for (let a = 0; a < actions; a++) tracks = min(tracks, sdBox(q.sub(vec2(KEY_X, shareZ(a))), vec2(SHARE / 2 + 0.06, 0.1), 0.1));
  let albedo: N<'vec3'> = mix(board, silicon, coverage(die));
  albedo = mix(albedo, rgb(0x5f586e), seal);
  albedo = mix(albedo, wellFill, coverage(well));
  albedo = mix(albedo, rgb(0x4b4659), max(hair(well, 0.8), seam.mul(0.8)));
  albedo = mix(albedo, rgb(0x474256), spine);
  albedo = mix(albedo, rgb(0x8a82a6), caret);
  albedo = mix(albedo, rgb(0x5d5670), max(columnWire, rowWire));
  albedo = mix(albedo, rgb(0x0a090e), coverage(display));
  albedo = mix(albedo, rgb(0x4b4659), hair(display, 0.8));
  albedo = mix(albedo, rgb(0x0d0c12), coverage(sockets));
  albedo = mix(albedo, rgb(0x0a090e), coverage(tracks));
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(albedo);
  return material;
}

export function arrayMaterial(layout: Layout, lit: Lit, charge: THREE.Texture, accent: number): THREE.MeshBasicNodeMaterial {
  const g = vec2(uv().x, uv().y.oneMinus()).mul(layout.span);
  const c = floor(g), f = fract(g).sub(0.5);
  const cellPx = float(1).div(max(fwidth(g.x), 1e-5));
  const pad = sdBox(f, vec2(0.4), 0.12);
  const dense = smoothstep(3.5, 8, cellPx);
  const heat = texture(charge, c.add(0.5).div(layout.span)).r;
  const tone = noise(c.mul(0.7).add(3.1)).mul(0.06).add(1);
  const padColor = mix(rgb(0x26222f), rgb(0x332e3e), smoothstep(-0.45, 0.45, f.y.negate().add(f.x.mul(0.3)))).mul(tone);
  let ruler: N<'float'> = float(0);
  for (let level = 1; level < layout.levels; level++) {
    const spacing = 2 ** level, weight = 0.3 + 0.7 * level / Math.max(1, layout.levels - 1);
    const off = abs(g.sub(floor(g.div(spacing).add(0.5)).mul(spacing))).mul(cellPx);
    const width = float(0.5 + 0.5 * weight);
    const line = max(clamp(width.sub(off.x).add(0.5), 0, 1), clamp(width.sub(off.y).add(0.5), 0, 1));
    ruler = max(ruler, line.mul(smoothstep(5, 12, cellPx.mul(spacing))).mul(weight));
  }
  const padCover = mix(float(0.5), coverage(pad), dense);
  let albedo: N<'vec3'> = mix(rgb(0x0c0b10), padColor, padCover);
  albedo = mix(albedo, rgb(0x5e5775), ruler.mul(0.7));
  const core = mix(float(1), smoothstep(0.6, 0.0, length(f)).mul(0.7).add(0.3).mul(coverage(pad)), dense);
  const glow = mix(rgb(0x6a4fd8), rgb(accent), heat).mul(pow(heat, float(1.4)).mul(core).mul(0.72));
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(albedo).add(glow);
  return material;
}

export function treeMaterial(layout: Layout, lit: Lit, rows: boolean): THREE.MeshBasicNodeMaterial {
  const { levels, span, cell, band } = layout;
  const u = (rows ? uv().y.oneMinus() : uv().x).mul(span);
  const v = (rows ? uv().x : uv().y.oneMinus()).mul(levels);
  const pxCell = fwidth(u), pxBand = fwidth(v);
  const level = clamp(floor(v), 0, levels - 1);
  const bit = float(levels - 1).sub(level);
  const node = floor(u.div(exp2(bit.add(1)))).add(0.5).mul(exp2(bit.add(1)));
  const child = exp2(bit.sub(1));
  const du = abs(u.sub(node));
  const mid = level.add(0.5);
  const outside = (a: N<'float'>, b: N<'float'>) => max(max(a.sub(v), v.sub(b)), 0);
  const inPx = (across: N<'float'>, along: N<'float'>) => length(vec2(across.div(max(pxCell, 1e-6)), along.div(max(pxBand, 1e-6))));
  const stub = inPx(du, outside(level, mid));
  const bar = inPx(max(du.sub(child), 0), abs(v.sub(mid)));
  const drop = inPx(du.sub(child), outside(mid, level.add(1)));
  const distance = min(min(stub, bar), drop);
  const coarse = bit.div(Math.max(1, levels - 1));
  const width = mix(float(0.6), float(1.2), coarse);
  const spread = exp2(bit).div(max(pxCell, 1e-6));
  const visible = smoothstep(2.2, 6.5, spread);
  const line = clamp(width.sub(distance).add(0.5), 0, 1).mul(visible);
  const haze = float(1).sub(visible).mul(0.22).mul(smoothstep(0.4, 0.1, abs(fract(v).sub(0.5))));
  const rootAt = vec2(span / 2 * cell, 0.22);
  const here = vec2(u.mul(cell), v.mul(band));
  const terminal = coverage(sdCircle(here.sub(rootAt), 0.12));
  const ring = coverage(abs(sdCircle(here.sub(rootAt), 0.21)).sub(0.02));
  const metal = mix(rgb(0x5a5470), rgb(0x837c9c), coarse);
  const alpha = max(max(line, haze), max(terminal, ring));
  const material = sheet();
  material.colorNode = lit(mix(metal, rgb(0xa39cc0), terminal));
  material.opacityNode = alpha;
  return material;
}

export interface LineStyle {
  pad: number;
  halo: number;
  gain: number;
  gradient: boolean;
}

export function lineMaterial(style: LineStyle): THREE.MeshBasicNodeMaterial {
  const shape = attribute<'vec4'>('aLine', 'vec4');
  const span = max(shape.x, 1e-3), width = shape.y;
  const p = uv().sub(0.5).mul(vec2(span.add(style.pad * 2), width.add(style.pad * 2)));
  const d = sdSegment(p, vec2(span.mul(-0.5), 0), vec2(span.mul(0.5), 0), width.mul(0.5));
  const body = coverage(d);
  const glow = exp(max(d, 0).mul(-1 / (style.pad * 0.36))).mul(style.halo).mul(smoothstep(style.pad, style.pad * 0.5, d));
  const along = style.gradient ? clamp(p.x.div(span).add(0.5), 0, 1).mul(0.78).add(0.22) : float(1);
  const material = sheet(true);
  material.colorNode = vec3(style.gain);
  material.opacityNode = max(body, glow).mul(shape.z).mul(along);
  return material;
}

export function reticleMaterial(layout: Layout, accent: number, flash: Level): { material: THREE.MeshBasicNodeMaterial; size: number } {
  const inner = Math.max(layout.cell * 0.5 + 0.08, 0.36), size = inner * 2 + 1.1;
  const p = uv().sub(0.5).mul(size), a = abs(p);
  const reach = Math.min(inner * 0.6, 0.26), stroke = 0.032;
  const corner = min(sdSegment(a, vec2(inner, inner - reach), vec2(inner, inner), stroke), sdSegment(a, vec2(inner - reach, inner), vec2(inner, inner), stroke));
  const dot = sdCircle(p, Math.max(layout.cell * 0.2, 0.05));
  const halo = exp(length(p).mul(-1 / (inner * 0.55))).mul(0.34).mul(smoothstep(size / 2, size / 2 - 0.2, length(p)));
  const material = sheet(true);
  material.colorNode = mix(rgb(accent).mul(1.9), rgb(0xffe6a6).mul(2.6), flash);
  material.opacityNode = max(max(coverage(corner), coverage(dot)), halo.mul(flash.add(1)));
  return { material, size };
}

export function lampMaterial(accent: number): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aLamp', 'vec4');
  const on = state.x, flash = state.y;
  const p = uv().sub(0.5).mul(3.6), r = length(p);
  const disk = sdCircle(p, 1), lens = sdCircle(p, 0.68);
  const bezelShade = smoothstep(1, -1, p.x.sub(p.y).mul(0.7)).mul(0.55).add(0.45);
  const bezel = mix(rgb(0x3a3645), rgb(0x9a94ae), bezelShade);
  const glass = mix(rgb(0x100e16), rgb(0x2a2634), smoothstep(0.7, 0, length(p.sub(vec2(-0.25, 0.25)))));
  const glint = coverage(sdCircle(p.sub(vec2(-0.24, 0.28)), 0.12)).mul(0.3);
  const hot = rgb(accent).mul(smoothstep(0.72, 0, r).mul(0.8).add(0.5)).mul(on.mul(1.05));
  const lensColor = glass.mul(on.oneMinus()).add(vec3(glint.mul(on.oneMinus()))).add(hot).add(vec3(flash.mul(0.9)));
  const color = mix(bezel.mul(0.42), lensColor, coverage(lens));
  const halo = exp(max(r.sub(0.7), 0).mul(-2.8)).mul(on.mul(0.42).add(flash.mul(0.6))).mul(smoothstep(1.8, 1.4, r));
  const ring = coverage(abs(r.sub(float(1.08).add(flash.oneMinus().mul(0.55)))).sub(0.05)).mul(flash);
  const material = sheet();
  material.colorNode = mix(rgb(accent).mul(1.3).add(vec3(ring)), color, coverage(disk));
  material.opacityNode = max(coverage(disk), clamp(max(halo, ring), 0, 1));
  return material;
}

export function keyMaterial(lit: Lit, accent: number): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aKey', 'vec4');
  const pips = state.x, press = state.y, glow = state.z;
  const size = KEY + 0.6, p = uv().sub(0.5).mul(size).div(press.mul(-0.04).add(1));
  const half = KEY / 2;
  const cap = sdBox(p, vec2(half - 0.04), 0.3);
  const top = sdBox(p.sub(vec2(-0.03, 0.05).mul(press.oneMinus())), vec2(half - 0.3), 0.22);
  const rim = smoothstep(0.26, -0.02, top.negate()).mul(coverage(cap));
  const lean = smoothstep(-half, half, p.x.sub(p.y));
  const face = mix(rgb(0x46415a), rgb(0x655e7e), smoothstep(-half, half, p.y.add(p.x.mul(-0.3)))).mul(press.mul(-0.2).add(1));
  const skirt = mix(rgb(0x2a2733), rgb(0x18161f), lean);
  const capColor = mix(face, skirt, rim);
  const d = 0.42;
  const odd = select(pips.greaterThan(2.5).and(pips.lessThan(3.5)), 1, select(pips.lessThan(1.5), 1, 0));
  const pair = select(pips.greaterThan(1.5), 1, 0);
  const quad = select(pips.greaterThan(3.5), 1, 0);
  const dotAt = (x: number, y: number) => sdCircle(p.sub(vec2(x, y)), 0.17);
  const pipShape = min(min(dotAt(0, 0).add(odd.oneMinus().mul(9)), min(dotAt(-d, d), dotAt(d, -d)).add(pair.oneMinus().mul(9))), min(dotAt(d, d), dotAt(-d, -d)).add(quad.oneMinus().mul(9)));
  const pip = coverage(pipShape);
  const socket = coverage(pipShape.sub(0.035)).sub(pip);
  const pipColor = mix(lit(rgb(0x8a83a8)), rgb(accent).mul(2.4), glow);
  const color = mix(mix(lit(capColor), rgb(0x0b0a10), socket.mul(0.8)), pipColor, pip);
  const shadow = sdBox(p.sub(vec2(0.06, -0.09)), vec2(half), 0.32);
  const halo = exp(max(cap, 0).mul(-6)).mul(glow.mul(0.5)).mul(smoothstep(size / 2, size / 2 - 0.12, max(abs(p.x), abs(p.y))));
  const material = sheet();
  material.colorNode = mix(rgb(accent).mul(glow.mul(1.4)), color, coverage(cap));
  material.opacityNode = max(coverage(cap), max(coverage(shadow).mul(0.5), halo));
  return material;
}

export function barMaterial(): THREE.MeshBasicNodeMaterial {
  const shape = attribute<'vec4'>('aBar', 'vec4');
  const size = vec2(shape.x.add(0.2), shape.y.add(0.2));
  const p = uv().sub(0.5).mul(size);
  const d = sdBox(p, vec2(shape.x.mul(0.5), shape.y.mul(0.5)), shape.y.mul(0.5));
  const body = coverage(d);
  const glow = exp(max(d, 0).mul(-24)).mul(0.3).mul(smoothstep(0.1, 0.05, d));
  const material = sheet(true);
  material.colorNode = vec3(1);
  material.opacityNode = max(body, glow).mul(shape.z);
  return material;
}

export function digitMaterial(accent: number): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aDigit', 'vec4');
  const mask = state.x, glow = state.y;
  const p = uv().sub(0.5).mul(vec2(DIGIT_W, DIGIT_H));
  const q = vec2(p.x.sub(p.y.mul(0.1)), p.y);
  const w = DIGIT_W * 0.36, h = DIGIT_H * 0.4, gap = 0.05, stroke = 0.055;
  const segments: readonly (readonly [number, number, number, number])[] = [
    [-w + gap, h, w - gap, h],
    [w, h - gap, w, gap],
    [w, -gap, w, -h + gap],
    [-w + gap, -h, w - gap, -h],
    [-w, -gap, -w, -h + gap],
    [-w, h - gap, -w, gap],
    [-w + gap, 0, w - gap, 0],
  ];
  let ghost: N<'float'> = float(1e3), shown: N<'float'> = float(0);
  segments.forEach(([ax, ay, bx, by], k) => {
    const d = sdSegment(q, vec2(ax, ay), vec2(bx, by), stroke);
    ghost = min(ghost, d);
    shown = max(shown, coverage(d).mul(mod(floor(mask.div(2 ** k)), 2)));
  });
  const material = sheet(true);
  material.colorNode = rgb(accent).mul(shown.mul(glow).mul(1.9).add(coverage(ghost).mul(0.035)));
  material.opacityNode = max(shown, coverage(ghost));
  return material;
}
