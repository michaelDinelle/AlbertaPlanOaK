import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, fract, length, max, min, mix, positionWorld, select, sign, sin, smoothstep, sqrt, uv, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';
import { CARD, RECORD, SCALE, type Layout } from './layout';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const SUITS = ['spades', 'diamonds', 'clubs', 'hearts'] as const;
export const GLYPHS = ['♠', '♦', '♣', '♥'] as const;
export const INKS = [0x1d2233, 0x2c62cf, 0x1b8a4d, 0xcf2d3e] as const;
export const TRUTH = [0xe8edff, 0x5b93ff, 0x3fd683, 0xff5566] as const;
export const HIT = TOKENS.celebrationGold;
export const MISS = TOKENS.failureBlue;
export const ACCENT = TOKENS.accents.Repeat!;
export const PAD = 0.3;
export const QUAD = new THREE.Vector2(CARD.x + 2 * PAD, CARD.y + 2 * PAD);
export const CHIP_QUAD = 1.6;

export const IVORY = 0xf4eddc;
const IVORY_EDGE = 0xd8ccb2;
const FRAME = 0xc9bea6;
const BACK = 0x1c3561;
const BACK_DEEP = 0x10213f;
const LEATHER = 0x1a2640;
const LEATHER_DEEP = 0x0e1526;
const INLAY = 0x121b30;
const VELVET = 0x0b1120;
const TOOL = 0xb08d4d;
const BRASS = 0xcaa45a;
const BRASS_DEEP = 0x6b5328;
const WALNUT = 0x3a2416;
const WALNUT_HI = 0x6a4428;
const FLOOR = 0x06080d;

export const flat = (width: number, depth: number): THREE.BufferGeometry => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const disc = (radius: number): THREE.BufferGeometry => new THREE.CircleGeometry(radius, 40).rotateX(-Math.PI / 2);

const sheet = (): THREE.MeshBasicNodeMaterial => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const fade = (from: number, to: number, x: N<'float'>): N<'float'> => float(1).sub(smoothstep(from, to, x));

const pick = (index: N<'float'>, values: readonly [N<'vec3'>, N<'vec3'>, N<'vec3'>, N<'vec3'>]): N<'vec3'> =>
  select(index.lessThan(0.5), values[0], select(index.lessThan(1.5), values[1], select(index.lessThan(2.5), values[2], values[3])));

const pickFloat = (index: N<'float'>, values: readonly [N<'float'>, N<'float'>, N<'float'>, N<'float'>]): N<'float'> =>
  select(index.lessThan(0.5), values[0], select(index.lessThan(1.5), values[1], select(index.lessThan(2.5), values[2], values[3])));

function heartOf(p: N<'vec2'>): N<'float'> {
  const q = vec2(abs(p.x), p.y), tip = q.sub(max(q.x.add(q.y), 0).mul(0.5)), top = q.sub(vec2(0, 1));
  const upper = length(q.sub(vec2(0.25, 0.75))).sub(Math.SQRT2 / 4);
  const lower = sqrt(min(dot(top, top), dot(tip, tip))).mul(sign(q.x.sub(q.y)));
  return select(q.x.add(q.y).greaterThan(1), upper, lower);
}

function stem(p: N<'vec2'>, top: number, flare: number): N<'float'> {
  const half = clamp(float(top).sub(p.y), 0, 1).mul(flare).add(0.035);
  return max(abs(p.x).sub(half).mul(1 / Math.hypot(1, flare)), max(float(-0.5).sub(p.y), p.y.sub(top)));
}

function rhombus(p: N<'vec2'>, bx: number, by: number): N<'float'> {
  const q = abs(p), b = vec2(bx, by), e = b.sub(q.mul(2));
  const h = clamp(e.x.mul(bx).sub(e.y.mul(by)).div(bx * bx + by * by), -1, 1);
  const d = length(q.sub(b.mul(vec2(float(1).sub(h), float(1).add(h))).mul(0.5)));
  return d.mul(sign(q.x.mul(by).add(q.y.mul(bx)).sub(bx * by)));
}

export function pipOf(p: N<'vec2'>, suit: N<'float'>): N<'float'> {
  const spade = min(heartOf(vec2(p.x, float(0.5).sub(p.y)).div(0.74)).mul(0.74), stem(p, -0.12, 0.42));
  const diamond = rhombus(p, 0.37, 0.5);
  const lobe = (x: number, y: number) => sdCircle(p.sub(vec2(x, y)), 0.215);
  const club = min(min(min(lobe(0, 0.255), lobe(-0.235, -0.085)), min(lobe(0.235, -0.085), sdCircle(p.sub(vec2(0, 0.02)), 0.11))), stem(p, -0.08, 0.4));
  const heart = heartOf(vec2(p.x, p.y.add(0.41)).div(0.74)).mul(0.74);
  return pickFloat(suit, [spade, diamond, club, heart]);
}

const pipAt = (p: N<'vec2'>, centre: readonly [number, number], size: number, suit: N<'float'>, turned = false): N<'float'> => {
  const q = p.sub(vec2(centre[0], centre[1])).div(size);
  return pipOf(turned ? q.negate() : q, suit).mul(size);
};

export const inkOf = (suit: N<'float'>): N<'vec3'> => pick(suit, [rgb(INKS[0]), rgb(INKS[1]), rgb(INKS[2]), rgb(INKS[3])]);

function ripples(p: N<'vec2'>): N<'vec3'> {
  const half = vec2(CARD.x / 2, CARD.y / 2), inner = sdBox(p, half.sub(0.075), 0.045), field = sdBox(p, half.sub(0.12), 0.03);
  const r = length(p.mul(vec2(1, 0.78))), band = abs(fract(r.mul(8.5)).sub(0.5)).sub(0.2).div(8.5);
  const rings = coverage(band.negate()).mul(coverage(field)).mul(fade(0.05, 0.62, r).mul(0.7).add(0.3));
  const eye = coverage(abs(sdCircle(p, 0.1)).sub(0.022));
  let color: N<'vec3'> = mix(rgb(BACK_DEEP), rgb(BACK), fade(0, 0.8, r));
  color = mix(color, rgb(ACCENT).mul(0.62), rings.mul(0.55));
  color = mix(color, rgb(ACCENT).mul(0.9), eye);
  return mix(rgb(IVORY).mul(0.92), color, coverage(inner));
}

export function cardMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const face = attribute<'vec4'>('aCard', 'vec4'), look = attribute<'vec4'>('aGlow', 'vec4');
  const suit = face.x, side = face.y, alpha = face.z, self = face.w, halo = look.x, due = look.y, verdict = look.z, ring = look.w;
  const gold = max(verdict, 0), dim = max(verdict.negate(), 0);
  const p = uv().sub(0.5).mul(vec2(QUAD.x, QUAD.y)), half = vec2(CARD.x / 2, CARD.y / 2);
  const body = sdBox(p, half, 0.075), cover = coverage(body), outside = float(1).sub(cover);

  const paper = mix(rgb(IVORY_EDGE), rgb(IVORY), fade(-0.16, -0.02, body)).mul(float(1).sub(p.y.mul(0.05)));
  const frame = coverage(abs(sdBox(p, half.sub(0.05), 0.05)).sub(0.006));
  const ink = min(min(pipAt(p, [0, 0], 0.42, suit), pipAt(p, [-0.33, 0.5], 0.17, suit)), pipAt(p, [0.33, -0.5], 0.17, suit, true));
  const printed = mix(mix(paper, rgb(FRAME), frame.mul(0.7)), inkOf(suit), coverage(ink));
  const trim = coverage(abs(body.add(0.034)).sub(0.019)).mul(gold);
  const tone = mix(ripples(p), mix(printed, rgb(HIT), trim), side), albedo = mix(tone, tone.mul(vec3(0.26, 0.3, 0.42)).add(rgb(MISS).mul(0.07)), dim);
  const edge = coverage(abs(body).sub(0.006));
  const shaded = mix(lit(albedo), albedo.mul(0.8), self).mul(float(1).sub(edge.mul(0.18)));

  const margin = fade(PAD * 0.55, PAD * 0.95, max(body, 0)).mul(outside);
  const shadow = fade(-0.03, 0.16, sdBox(p.sub(vec2(0.035, -0.065)), half, 0.075)).mul(0.5).mul(outside);
  const bloom = exp(max(body, 0).div(-0.07)).mul(halo).mul(margin);
  const warm = exp(max(body, 0).div(-0.06)).mul(ring).mul(margin).mul(0.7);
  const band = half.add(0.085), along = select(abs(p.y).mul(band.x).greaterThan(abs(p.x).mul(band.y)), p.x, p.y);
  const dash = coverage(abs(fract(along.div(0.2)).sub(0.5)).sub(0.24));
  const outline = coverage(abs(sdBox(p, band, 0.13)).sub(0.017)).mul(dash).mul(due);
  const glowAlpha = clamp(bloom.add(warm).add(outline), 0, 1);
  const glowSum = rgb(ACCENT).mul(bloom.mul(1.25).add(outline.mul(1.1))).add(rgb(HIT).mul(warm.mul(0.95)));
  const outer = clamp(shadow.add(glowAlpha), 0, 1);
  material.colorNode = mix(glowSum.div(max(outer, 1e-4)), shaded, cover);
  material.opacityNode = mix(outer, float(1), cover).mul(alpha);
  return material;
}

export function chipMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aChip', 'vec4');
  const suit = state.x, alpha = state.y, gold = state.z, dim = state.w;
  const p = uv().sub(0.5).mul(CHIP_QUAD), r = length(p);
  const disc = r.sub(0.5), cover = coverage(disc), outside = float(1).sub(cover);
  const angle = atan(p.y, p.x).div(2 * Math.PI).mul(8).add(0.5);
  const spot = coverage(abs(fract(angle).sub(0.5)).sub(0.16).mul(r)).mul(coverage(float(0.39).sub(r)));
  const rim = mix(mix(rgb(ACCENT), rgb(HIT), gold), rgb(MISS).mul(0.8), dim);
  let color: N<'vec3'> = mix(rim.mul(float(0.82).add(p.y.mul(0.25))), rgb(IVORY), spot.mul(0.92));
  color = mix(color, rgb(0x2a3552), coverage(abs(r.sub(0.36)).sub(0.012)));
  color = mix(color, rgb(IVORY), coverage(r.sub(0.345)));
  color = mix(color, inkOf(suit), coverage(pipOf(p.div(0.38), suit).mul(0.38)));
  color = color.mul(mix(vec3(1), vec3(0.5, 0.54, 0.64), dim));
  const slash = coverage(abs(p.x.add(p.y)).mul(Math.SQRT1_2).sub(0.03)).mul(coverage(r.sub(0.44))).mul(dim);
  const albedo = mix(color, rgb(MISS).mul(0.95), slash);
  const shaded = mix(lit(albedo), albedo, 0.85);
  const shadow = fade(-0.04, 0.1, length(p.sub(vec2(0.03, -0.05))).sub(0.5)).mul(0.5).mul(outside);
  const halo = exp(max(disc, 0).div(-0.06)).mul(gold).mul(fade(0.17, 0.28, disc)).mul(outside).mul(0.6);
  const outer = clamp(shadow.add(halo), 0, 1);
  material.colorNode = mix(rgb(HIT).mul(halo).div(max(outer, 1e-4)), shaded, cover);
  material.opacityNode = mix(outer, float(1), cover).mul(alpha);
  return material;
}

function brassRim(d: N<'float'>, width: number): { cover: N<'float'>; color: N<'vec3'> } {
  const across = clamp(d.div(width).add(0.5), 0, 1);
  return { cover: coverage(abs(d).sub(width / 2)), color: mix(rgb(BRASS_DEEP), rgb(BRASS), smoothstep(0, 0.55, across).mul(fade(0.6, 1, across))) };
}

export function tableMaterial(layout: Layout, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ fog: false });
  const q = vec2(positionWorld.x, positionWorld.z), box = layout.table, centre = box.getCenter(new THREE.Vector2()), size = box.getSize(new THREE.Vector2());
  const cardW = CARD.x * SCALE, cardH = CARD.y * SCALE, recW = cardW * RECORD, recH = cardH * RECORD;
  const edge = sdBox(q.sub(vec2(centre.x, centre.y)), vec2(size.x / 2, size.y / 2), 0.7);
  const rail = 0.46, leather = edge.add(rail);
  const tool = (d: N<'float'>, width: number) => coverage(abs(d).sub(width));

  const mottle = fractal(q.mul(0.9), 3).mul(0.07).add(noise(q.mul(7.3)).mul(0.035)).add(1);
  const pool = fade(1.5, 9, length(q.sub(vec2(0, -0.8)).mul(vec2(0.6, 1)))).mul(0.35).add(0.65);
  let top: N<'vec3'> = mix(rgb(LEATHER_DEEP), rgb(LEATHER), pool).mul(mottle);
  top = top.mul(float(1).sub(exp(leather.mul(3.2)).mul(0.5)));
  let tooled: N<'float'> = max(tool(leather.add(0.32), 0.014), tool(leather.add(0.44), 0.006));

  const spreadLeft = layout.due - cardW / 2 - 0.34, spreadRight = layout.shoe.x + layout.shoeSize.x / 2 + 0.3;
  const runway = sdBox(q.sub(vec2((spreadLeft + spreadRight) / 2, layout.y)), vec2((spreadRight - spreadLeft) / 2, cardH / 2 + 0.3), 0.22);
  const recLeft = layout.record.x0 - recW / 2 - 0.3, recRight = layout.record.x0 + (layout.record.count - 1) * layout.record.pitch + recW / 2 + 0.3;
  const shelf = sdBox(q.sub(vec2((recLeft + recRight) / 2, layout.record.y)), vec2((recRight - recLeft) / 2, recH / 2 + 0.26), 0.2);
  const inlay = min(runway, shelf);
  top = mix(top, rgb(INLAY).mul(mottle), coverage(inlay).mul(0.85));
  tooled = max(tooled, tool(inlay, 0.01));

  const tipY = layout.tray.y - layout.traySize.y / 2 - 0.16;
  const drop = sdSegment(q, vec2(layout.due, layout.y + cardH / 2 + 0.36), vec2(layout.due, tipY), 0.016);
  const head = min(sdSegment(q, vec2(layout.due - 0.16, tipY - 0.16), vec2(layout.due, tipY), 0.016), sdSegment(q, vec2(layout.due + 0.16, tipY - 0.16), vec2(layout.due, tipY), 0.016));
  tooled = max(tooled, coverage(min(drop, head)).mul(0.75));

  const tray = sdBox(q.sub(vec2(layout.tray.x, layout.tray.y)), vec2(layout.traySize.x / 2, layout.traySize.y / 2), 0.26);
  const well = mix(rgb(VELVET), rgb(INLAY), smoothstep(-0.9, 0, tray).mul(0.5)).mul(noise(q.mul(23)).mul(0.04).add(1));
  const trayRim = brassRim(tray.add(0.02), 0.07);
  top = mix(top, well.mul(float(1).sub(exp(tray.mul(5)).mul(0.55))), coverage(tray));
  const shoeShadow = fade(-0.1, 0.35, sdBox(q.sub(vec2(layout.shoe.x + 0.08, layout.shoe.y + 0.12)), vec2(layout.shoeSize.x / 2, layout.shoeSize.y / 2), 0.2)).mul(0.55);
  top = top.mul(float(1).sub(shoeShadow));

  let color: N<'vec3'> = mix(top, rgb(TOOL).mul(mottle), tooled.mul(0.8));
  color = mix(color, trayRim.color, trayRim.cover);

  const across = clamp(edge.negate().div(rail), 0, 1), local = q.sub(vec2(centre.x, centre.y)), bend = vec2(size.x / 2 - 0.7, size.y / 2 - 0.7);
  const normal = local.sub(clamp(local, bend.negate(), bend)).normalize(), facing = dot(normal, vec2(-0.6, -0.8));
  const slope = cos(across.mul(Math.PI)), crown = sin(across.mul(Math.PI));
  const grain = noise(vec2(q.x.add(q.y).mul(0.8), across.mul(9))).mul(0.14).add(noise(q.mul(31)).mul(0.05)).add(1);
  const sheen = exp(across.sub(0.3).div(0.07).pow(2).negate()).mul(max(facing, 0)).mul(0.5);
  const wood = mix(rgb(WALNUT).mul(0.5), rgb(WALNUT_HI), clamp(crown.mul(0.45).add(slope.mul(facing).mul(0.4)).add(0.3), 0, 1)).mul(grain).add(rgb(0xffd9a8).mul(sheen.mul(0.18)));
  color = mix(color, wood, coverage(edge).mul(coverage(leather.negate())));
  color = mix(color, rgb(0x0a0604), tool(edge.add(0.012), 0.012).mul(0.8));
  color = mix(color, rgb(BRASS_DEEP), tool(leather, 0.013).mul(0.9));

  const floor = rgb(FLOOR).mul(fade(4, 16, length(q)).mul(0.8).add(0.4)).mul(float(1).sub(fade(0, 0.9, edge).mul(0.65)));
  material.colorNode = mix(floor, lit(color), coverage(edge));
  return material;
}

export function shoeMaterial(layout: Layout, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const w = layout.shoeSize.x, h = layout.shoeSize.y;
  const p = uv().sub(0.5).mul(vec2(w + 0.2, h + 0.2));
  const body = sdBox(p, vec2(w / 2, h / 2), 0.2);
  const across = clamp(body.negate().div(0.22), 0, 1);
  const grain = noise(vec2(p.x.mul(1.4), p.y.mul(12))).mul(0.14).add(1);
  let color: N<'vec3'> = mix(rgb(WALNUT).mul(0.6), rgb(WALNUT_HI), smoothstep(0, 1, across).mul(0.7)).mul(grain);
  const opening = sdBox(p.sub(vec2(0.1, 0.05)), vec2(w / 2 - 0.34, h / 2 - 0.38), 0.1);
  const deck = sdBox(p.sub(vec2(0.1, 0.05)), vec2((CARD.x * SCALE * RECORD) / 2, (CARD.y * SCALE * RECORD) / 2), 0.06);
  color = mix(color, rgb(0x0a0d14), coverage(opening));
  color = mix(color, ripples(p.sub(vec2(0.1, 0.05)).div(RECORD * SCALE)), coverage(deck).mul(coverage(opening)));
  const rim = brassRim(opening, 0.06);
  color = mix(color, rim.color, rim.cover);
  const mouth = sdBox(p.sub(vec2(-w / 2 + 0.05, 0)), vec2(0.07, h / 2 - 0.3), 0.05), lip = brassRim(mouth, 0.05);
  color = mix(color, rgb(0x07090e), coverage(mouth));
  color = mix(color, lip.color, lip.cover);
  for (const [x, y] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
    const bolt = sdCircle(p.sub(vec2(x * (w / 2 - 0.14), y * (h / 2 - 0.14))), 0.05);
    color = mix(color, mix(rgb(BRASS_DEEP), rgb(BRASS), fade(-0.05, 0.05, bolt)), coverage(bolt));
  }
  material.colorNode = lit(color);
  material.opacityNode = coverage(body);
  return material;
}

export function lampSize(layout: Layout): THREE.Vector2 {
  return new THREE.Vector2(1.5, 1.5 + 2 * (layout.lamp.y - layout.table.min.y - 0.23));
}

export function lampMaterial(layout: Layout, lens: Level, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const size = lampSize(layout), reach = (size.y - size.x) / 2;
  const p = uv().sub(0.5).mul(vec2(size.x, size.y));
  const arm = sdSegment(p, vec2(0, 0), vec2(0, reach), 0.075), shade = sdCircle(p, 0.46), glass = sdCircle(p, 0.25);
  const rim = brassRim(shade.add(0.035), 0.07);
  let color: N<'vec3'> = mix(rgb(BRASS_DEEP), rgb(BRASS), fade(-0.02, 0.075, abs(p.x)).mul(0.8));
  color = mix(color, rgb(0x141a26).mul(float(1).add(p.x.mul(0.3))), coverage(shade));
  color = mix(color, rim.color, rim.cover);
  const cover = max(coverage(arm), coverage(shade));
  const shadow = fade(-0.05, 0.12, sdCircle(p.sub(vec2(0.04, -0.07)), 0.46)).mul(0.5).mul(float(1).sub(cover));
  const glow = rgb(ACCENT).mul(mix(float(0.9), float(1.9), fade(0, 0.25, length(p))).mul(lens));
  material.colorNode = mix(mix(vec3(0), lit(color), cover.div(max(cover.add(shadow), 1e-4))), glow, coverage(glass));
  material.opacityNode = clamp(max(cover, shadow), 0, 1);
  return material;
}

