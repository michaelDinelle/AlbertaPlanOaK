import * as THREE from 'three/webgpu';
import { abs, atan, clamp, exp, float, fract, instancedDynamicBufferAttribute, length, max, min, mix, positionWorld, sin, smoothstep, step, texture, uniformArray, uv, vec2, vec3 } from 'three/tsl';
import type { Light2D } from '@kit/light2d';
import { coverage, fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Scalar = N<'float'> | number;

export const CARD_W = 1.05;
export const CARD_H = 1.47;
export const CARD_PAD = 0.56;
export const CHIP_R = 0.23;
export const CHIP_LIFT = 0.06;
export const CHIP_QW = 2 * CHIP_R + 0.2;
export const CHIP_QH = 2 * CHIP_R + CHIP_LIFT + 0.2;
export const PILL_PAD = 0.4;
export const INK = [0x1b2230, 0xc0263a] as const;
export const DENOMINATIONS = [
  { face: 0xebe4d2, spot: 0x1d6f57, rim: 0xcfc5ad },
  { face: 0xb3262d, spot: 0xf3ead6, rim: 0x8e1b22 },
  { face: 0x23262e, spot: 0xd7b45e, rim: 0x14161b },
] as const;

const SERIF = 'ui-serif, "New York", "Iowan Old Style", Georgia, "Times New Roman", serif';
const SANS = 'ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const CELL = 128;
const WORD = 1024;
const SHEET = 2048;
const FELT = 0x12573f;
const GOLD = 0xc9a45c;

export interface Sprite { rect: THREE.Vector4; aspect: number }

export interface Atlas {
  texture: THREE.DataTexture;
  glyph(text: string): Sprite;
  word(text: string): Sprite;
}

export function makeAtlas(glyphs: readonly string[], words: readonly string[]): Atlas {
  const perRow = SHEET / CELL, wordsPerRow = SHEET / WORD, glyphRows = Math.ceil(glyphs.length / perRow), height = (glyphRows + Math.ceil(words.length / wordsPerRow)) * CELL;
  const canvas = document.createElement('canvas');
  canvas.width = SHEET;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('LeducPoker lettering needs a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  const sprites = new Map<string, Sprite>();
  const place = (key: string, x0: number, y0: number, x1: number): void => {
    sprites.set(key, { rect: new THREE.Vector4(x0 / SHEET, 1 - (y0 + CELL) / height, x1 / SHEET, 1 - y0 / height), aspect: (x1 - x0) / CELL });
  };
  glyphs.forEach((glyph, i) => {
    const x = (i % perRow) * CELL, y = Math.floor(i / perRow) * CELL, serif = /[JQK]/.test(glyph);
    context.font = serif ? `900 ${0.84 * CELL}px ${SERIF}` : `800 ${0.74 * CELL}px ${SANS}`;
    const cap = context.measureText('H').actualBoundingBoxAscent;
    context.fillText(glyph, x + CELL / 2, y + CELL / 2 + cap / 2);
    const ink = context.measureText(glyph).width;
    place(`g:${glyph}`, x + CELL / 2 - Math.min(CELL, ink + 12) / 2, y, x + CELL / 2 + Math.min(CELL, ink + 12) / 2);
  });
  words.forEach((word, i) => {
    const x = (i % wordsPerRow) * WORD, y = (glyphRows + Math.floor(i / wordsPerRow)) * CELL;
    context.font = `700 ${0.5 * CELL}px ${SANS}`;
    context.letterSpacing = `${0.045 * CELL}px`;
    const cap = context.measureText('H').actualBoundingBoxAscent, width = Math.min(WORD - 16, context.measureText(word).width);
    context.fillText(word, x + WORD / 2 + 0.0225 * CELL, y + CELL / 2 + cap / 2, WORD - 16);
    place(`w:${word}`, x + WORD / 2 - width / 2 - 20, y, x + WORD / 2 + width / 2 + 20);
  });
  const pixels = context.getImageData(0, 0, SHEET, height).data, alpha = new Uint8Array(SHEET * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < SHEET; x++) alpha[(height - 1 - y) * SHEET + x] = pixels[4 * (y * SHEET + x) + 3]!;
  const out = new THREE.DataTexture(alpha, SHEET, height, THREE.RedFormat, THREE.UnsignedByteType);
  out.minFilter = THREE.LinearMipmapLinearFilter;
  out.magFilter = THREE.LinearFilter;
  out.generateMipmaps = true;
  out.anisotropy = 4;
  out.needsUpdate = true;
  const find = (key: string): Sprite => {
    const sprite = sprites.get(key);
    if (!sprite) throw new Error(`LeducPoker lettering lacks ${key.slice(2)}`);
    return sprite;
  };
  return { texture: out, glyph: (text) => find(`g:${text}`), word: (text) => find(`w:${text}`) };
}

const scalar = (value: Scalar): N<'float'> => typeof value === 'number' ? float(value) : value;

const letter = (atlas: THREE.Texture, rect: N<'vec4'>, aspect: Scalar, p: N<'vec2'>, centre: N<'vec2'>, height: Scalar): N<'float'> => {
  const q = p.sub(centre).div(vec2(scalar(height).mul(aspect), height)).add(0.5);
  const inside = step(0, q.x).mul(step(q.x, 1)).mul(step(0, q.y)).mul(step(q.y, 1));
  return texture(atlas, rect.xy.add(rect.zw.sub(rect.xy).mul(clamp(q, vec2(0), vec2(1))))).r.mul(inside);
};

const turn = (p: N<'vec2'>): N<'vec2'> => vec2(p.x.add(p.y), p.y.sub(p.x)).mul(Math.SQRT1_2);

const heart = (p: N<'vec2'>, size: Scalar): N<'float'> => {
  const q = p.div(size).add(vec2(0, 0.044));
  return min(sdBox(turn(q), vec2(0.3), 0.02), min(sdCircle(q.sub(vec2(-0.212, 0.212)), 0.3), sdCircle(q.sub(vec2(0.212, 0.212)), 0.3))).mul(size);
};

const spade = (p: N<'vec2'>, size: Scalar): N<'float'> => {
  const q = p.div(size), top = q.sub(vec2(0, 0.1)).mul(vec2(1, -1)).div(0.86).add(vec2(0, 0.044));
  const leaf = min(sdBox(turn(top), vec2(0.3), 0.02), min(sdCircle(top.sub(vec2(-0.212, 0.212)), 0.3), sdCircle(top.sub(vec2(0.212, 0.212)), 0.3))).mul(0.86);
  const stem = min(sdSegment(q, vec2(0, 0.0), vec2(0, -0.4), 0.05), sdBox(q.sub(vec2(0, -0.44)), vec2(0.17, 0.04), 0.03));
  return min(leaf, stem).mul(size);
};

const pip = (p: N<'vec2'>, size: Scalar, suit: N<'float'>): N<'float'> => mix(spade(p, size), heart(p, size), suit);

interface Layer { color: N<'vec3'>; alpha: N<'float'> }

const over = (under: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer => {
  const a = alpha.add(under.alpha.mul(alpha.oneMinus()));
  return { color: color.mul(alpha).add(under.color.mul(under.alpha).mul(alpha.oneMinus())).div(max(a, 1e-4)), alpha: a };
};

const surface = (): THREE.MeshBasicNodeMaterial => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const attr = (buffer: THREE.InstancedBufferAttribute): N<'vec4'> => instancedDynamicBufferAttribute(buffer, 'vec4') as unknown as N<'vec4'>;

export interface Zone { x: number; z: number; w: number; h: number; radius: number }

export interface TableShape { hx: number; hz: number; radius: number; rail: number; line: Zone; zones: readonly Zone[] }

export function tableMaterial(light: Light2D, shape: TableShape, accent: number): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const p = positionWorld.xz, felt = sdBox(p, vec2(shape.hx, shape.hz), shape.radius), outer = felt.sub(shape.rail);
  const weave = fractal(p.mul(vec2(9, 13)), 3).mul(0.5).add(0.5), fibre = fractal(p.mul(vec2(46, 3)), 2).mul(0.5).add(0.5);
  const pool = smoothstep(1.25, 0.1, length(p.mul(vec2(0.13, 0.27))));
  let top = mix(rgb(0x0a3226), rgb(FELT), pool).mul(weave.mul(0.12).add(0.94)).mul(fibre.mul(0.06).add(0.97));
  const printed = rgb(0x7fcaa4), outline = (zone: Zone, width: number) => coverage(abs(sdBox(p.sub(vec2(zone.x, zone.z)), vec2(zone.w / 2, zone.h / 2), zone.radius)).sub(width));
  top = mix(top, printed, outline(shape.line, 0.013).mul(0.3));
  for (const zone of shape.zones) top = mix(top, printed, outline(zone, 0.009).mul(0.24));
  const across = felt.div(shape.rail).clamp(0, 1), ridge = sin(across.mul(Math.PI));
  const hide = fractal(p.mul(3.2), 3).mul(0.5).add(0.5);
  let rail = mix(rgb(0x120c0a), rgb(0x3a2419), ridge.mul(0.85)).mul(hide.mul(0.18).add(0.9));
  rail = rail.add(rgb(0xffe2c0).mul(exp(abs(across.sub(0.38)).mul(-22)).mul(0.13)));
  const dash = step(0.5, fract(atan(p.y, p.x).mul(60 / Math.PI))), stitch = coverage(abs(felt.sub(shape.rail * 0.8)).sub(0.008)).mul(dash);
  rail = mix(rail, rgb(0x6b5134), stitch.mul(0.5));
  const brass = coverage(abs(felt.sub(0.035)).sub(0.03)), glint = exp(abs(felt.sub(0.035)).mul(-60)).mul(0.5);
  const trim = rgb(GOLD).mul(glint.add(0.75));
  const shadow = smoothstep(0.9, -0.1, outer.sub(0.18)).mul(0.75);
  const carpet = mix(rgb(0x0b0a0d), rgb(0x151219), fractal(p.mul(0.9), 3).mul(0.5).add(0.5)).mul(shadow.oneMinus());
  const lattice = coverage(min(abs(fract(p.x.add(p.y).mul(1.2)).sub(0.5)), abs(fract(p.x.sub(p.y).mul(1.2)).sub(0.5))).mul(0.59).sub(0.012));
  const floor = mix(carpet, rgb(accent).mul(0.07), lattice.mul(shadow.oneMinus()).mul(0.6));
  let color = mix(top, rail, step(0, felt));
  color = mix(color, trim, brass.mul(step(felt, shape.rail * 0.5)));
  color = mix(floor, color, coverage(outer));
  material.colorNode = light.lit(color);
  return material;
}

export interface CardBuffers { card: THREE.InstancedBufferAttribute; look: THREE.InstancedBufferAttribute }

export function cardMaterial(light: Light2D, atlas: Atlas, buffers: CardBuffers, accent: number, clock: N<'float'>): THREE.MeshBasicNodeMaterial {
  const card = attr(buffers.card), look = attr(buffers.look);
  const rank = card.x, suit = card.y, face = card.z, peek = card.w, alpha = look.x, act = look.y, win = look.z, dim = look.w;
  const ranks = uniformArray(['J', 'Q', 'K'].map((g) => atlas.glyph(g).rect), 'vec4'), aspects = uniformArray(['J', 'Q', 'K'].map((g) => atlas.glyph(g).aspect), 'float');
  const rect = ranks.element(rank.toInt()) as unknown as N<'vec4'>, aspect = aspects.element(rank.toInt()) as unknown as N<'float'>;
  const material = surface();
  const t = uv().sub(0.5).mul(vec2(CARD_W + CARD_PAD, CARD_H + CARD_PAD)), half = vec2(CARD_W / 2, CARD_H / 2);
  const d = sdBox(t, half, 0.085), body = coverage(d);
  const ink = mix(rgb(INK[0]), rgb(INK[1]), suit);

  const paper = mix(rgb(0xfdf9ee), rgb(0xebe1ca), smoothstep(0.15, 0.95, length(t.mul(vec2(1.1, 0.8)))));
  const grain = fractal(t.mul(38), 2).mul(0.025);
  let front = paper.sub(vec3(grain));
  const frame = coverage(abs(sdBox(t, half.sub(0.075), 0.05)).sub(0.007));
  front = mix(front, ink, frame.mul(0.35));
  const big = letter(atlas.texture, rect, aspect, t, vec2(0, 0.1), 0.86);
  const shade = letter(atlas.texture, rect, aspect, t.sub(vec2(0.018, -0.022)), vec2(0, 0.1), 0.86);
  front = mix(front, ink.mul(0.25).add(vec3(0.62, 0.58, 0.5)), shade.mul(0.35));
  front = mix(front, ink, big);
  front = mix(front, ink, coverage(pip(t.sub(vec2(0, -0.43)), 0.25, suit)));
  for (const flip of [1, -1]) {
    const q = t.mul(flip);
    front = mix(front, ink, letter(atlas.texture, rect, aspect, q, vec2(-0.37, 0.535), 0.3));
    front = mix(front, ink, coverage(pip(q.sub(vec2(-0.37, 0.335)), 0.11, suit)));
  }

  const navy = mix(rgb(0x122047), rgb(0x1d356b), smoothstep(-0.8, 0.8, t.y.sub(t.x.mul(0.4))));
  const inner = sdBox(t, half.sub(0.08), 0.05), inside = coverage(inner);
  const lanes = min(abs(fract(t.x.add(t.y).div(0.16)).sub(0.5)), abs(fract(t.x.sub(t.y).div(0.16)).sub(0.5))).mul(0.16 * Math.SQRT1_2);
  let back = mix(navy, rgb(accent).mul(0.8), coverage(lanes.sub(0.0055)).mul(inside).mul(0.55));
  const r = length(t), medal = coverage(r.sub(0.2));
  back = mix(back, rgb(0x0f1a38), medal);
  back = mix(back, rgb(accent), coverage(pip(t.sub(vec2(0, 0.0)), 0.22, float(0))).mul(0.75));
  back = mix(back, rgb(GOLD), coverage(abs(r.sub(0.2)).sub(0.012)));
  back = mix(back, rgb(GOLD), coverage(abs(inner).sub(0.011)));
  back = mix(back, rgb(0xf1f3fb), coverage(abs(d.add(0.03)).sub(0.022)));

  const corner = t.add(half), reach = corner.x.add(corner.y), fold = peek.mul(0.64), flap = coverage(reach.sub(fold).mul(Math.SQRT1_2)).mul(step(0.01, peek));
  const crease = exp(max(reach.sub(fold), 0).mul(-26)).mul(step(0.01, peek)).mul(0.45);
  back = back.mul(crease.oneMinus());
  let underside = mix(rgb(0xfaf5e8), rgb(0xd9ceb5), reach.div(max(fold, 0.01)).clamp(0, 1).oneMinus().mul(0.4));
  underside = mix(underside, ink, letter(atlas.texture, rect, aspect, corner, vec2(0.145, 0.27), 0.3));
  underside = mix(underside, ink, coverage(pip(corner.sub(vec2(0.35, 0.1)), 0.11, suit)));
  underside = mix(underside, rgb(0x5a5242), coverage(abs(reach.sub(fold).mul(Math.SQRT1_2)).sub(0.006)).mul(0.8));
  back = mix(back, underside, flap);

  let color = mix(back, front, face);
  color = mix(color, color.mul(vec3(0.55, 0.58, 0.62)), dim);
  const lit = min(light.lit(color), color.mul(1.12));
  const shadow = smoothstep(0.16, -0.08, sdBox(t.sub(vec2(0.05, -0.07)), half, 0.1)).mul(0.5);
  const pulse = sin(clock.mul(4.2)).mul(0.25).add(0.75);
  const halo = exp(max(d, 0).mul(-10)).mul(step(0, d)).mul(smoothstep(0.26, 0.12, d));
  const glowTone = mix(rgb(accent).mul(0.9), rgb(0xffd166).mul(1.3), win.div(max(win.add(act.mul(pulse)), 1e-3)));
  let layer: Layer = { color: vec3(0), alpha: shadow };
  layer = over(layer, glowTone, halo.mul(max(act.mul(pulse).mul(0.7), win)).clamp(0, 1));
  layer = over(layer, lit, body);
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha.mul(alpha);
  return material;
}

export interface ChipBuffers { chip: THREE.InstancedBufferAttribute; value: THREE.InstancedBufferAttribute }

export function chipMaterial(light: Light2D, atlas: Atlas, buffers: ChipBuffers): THREE.MeshBasicNodeMaterial {
  const chip = attr(buffers.chip), value = attr(buffers.value), denomination = chip.x, alpha = chip.y, glow = chip.z, aspect = chip.w;
  const pick = (key: 'face' | 'spot' | 'rim'): N<'vec3'> => {
    const [a, b, c] = DENOMINATIONS.map((d) => rgb(d[key]));
    return mix(mix(a!, b!, step(0.5, denomination)), c!, step(1.5, denomination));
  };
  const face = pick('face'), spot = pick('spot'), rim = pick('rim');
  const material = surface();
  const t = uv().sub(0.5).mul(vec2(CHIP_QW, CHIP_QH)), top = t.sub(vec2(0, CHIP_LIFT / 2));
  const disc = sdCircle(top, CHIP_R), body = sdSegment(t, vec2(0, -CHIP_LIFT / 2), vec2(0, CHIP_LIFT / 2), CHIP_R);
  const r = length(top), angle = atan(top.y, top.x);
  const spots = step(fract(angle.mul(8 / (2 * Math.PI)).add(0.25)), 0.4).mul(smoothstep(CHIP_R - 0.058, CHIP_R - 0.05, r));
  let color = mix(face, spot, spots);
  color = mix(color, rim, coverage(abs(r.sub(CHIP_R * 0.62)).sub(0.01)).mul(0.9));
  color = mix(color, face.mul(1.08), coverage(r.sub(CHIP_R * 0.55)).mul(0.5));
  color = mix(color, spot, letter(atlas.texture, value, aspect, top, vec2(0, 0), 0.19));
  color = color.mul(smoothstep(-CHIP_R, CHIP_R, top.y.sub(top.x.mul(0.5))).mul(0.16).add(0.9));
  const across = clamp(t.x.div(CHIP_R), -1, 1), side = mix(rim, spot.mul(0.8), step(0.5, fract(across.mul(2.2).add(0.25))).mul(0.55));
  const edge = mix(side.mul(smoothstep(-1, 1, across.negate()).mul(0.3).add(0.7)), color, coverage(disc));
  const lit = min(light.lit(edge), edge.mul(1.15)).add(rgb(0xffd166).mul(glow.mul(0.35)));
  const shadow = smoothstep(0.08, -0.05, sdSegment(t.sub(vec2(0.035, -0.045)), vec2(0, -CHIP_LIFT / 2), vec2(0, CHIP_LIFT / 2), CHIP_R)).mul(0.45);
  let layer: Layer = { color: vec3(0), alpha: shadow };
  layer = over(layer, lit, coverage(body));
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha.mul(alpha);
  return material;
}

export interface PillBuffers { shape: THREE.InstancedBufferAttribute; fill: THREE.InstancedBufferAttribute; edge: THREE.InstancedBufferAttribute }

export function pillMaterial(light: Light2D, buffers: PillBuffers): THREE.MeshBasicNodeMaterial {
  const shape = attr(buffers.shape), fill = attr(buffers.fill), edge = attr(buffers.edge);
  const w = shape.x, h = shape.y, radius = shape.z, glow = shape.w;
  const material = surface();
  const t = uv().sub(0.5).mul(vec2(w.add(PILL_PAD), h.add(PILL_PAD)));
  const d = sdBox(t, vec2(w, h).mul(0.5), radius), inner = coverage(d), line = coverage(abs(d.add(edge.w.mul(0.5))).sub(edge.w.mul(0.5)));
  const sheen = smoothstep(h.mul(-0.5), h.mul(0.5), t.y).mul(0.18).add(0.9);
  const tone = mix(fill.rgb.mul(sheen), edge.rgb, line);
  const lit = min(light.lit(tone), tone.mul(1.1)).add(tone.mul(0.25));
  const shadow = smoothstep(0.1, -0.05, sdBox(t.sub(vec2(0.03, -0.05)), vec2(w, h).mul(0.5), radius)).mul(0.4).mul(fill.a);
  const halo = exp(max(d, 0).mul(-9)).mul(step(0, d)).mul(glow);
  let layer: Layer = { color: vec3(0), alpha: shadow };
  layer = over(layer, edge.rgb.mul(1.6), halo.clamp(0, 1));
  layer = over(layer, lit, max(inner.mul(fill.a), line));
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha;
  return material;
}

export interface LabelBuffers { rect: THREE.InstancedBufferAttribute; tint: THREE.InstancedBufferAttribute }

export function labelMaterial(atlas: Atlas, buffers: LabelBuffers): THREE.MeshBasicNodeMaterial {
  const rect = attr(buffers.rect), tint = attr(buffers.tint);
  const material = surface();
  material.colorNode = tint.rgb;
  material.opacityNode = texture(atlas.texture, rect.xy.add(rect.zw.sub(rect.xy).mul(uv()))).r.mul(tint.a);
  return material;
}

export function gaugeMaterial(light: Light2D, progress: N<'float'>, flash: N<'float'>, tint: number): THREE.MeshBasicNodeMaterial {
  const material = surface();
  const t = uv().sub(0.5), r = length(t), angle = fract(float(0.25).sub(atan(t.y, t.x).div(2 * Math.PI)));
  const ring = coverage(abs(r.sub(0.35)).sub(0.065)), done = step(angle, progress);
  const notch = coverage(abs(fract(angle.mul(16)).sub(0.5)).mul(0.35 * 2 * Math.PI / 16).sub(0.006).negate());
  const tone = mix(rgb(0x4a3f63), rgb(tint).mul(flash.mul(0.8).add(1.15)), done).mul(notch.mul(0.35).oneMinus());
  const disc = coverage(r.sub(0.27));
  const halo = exp(abs(r.sub(0.36)).mul(-18)).mul(flash).mul(0.9);
  let layer: Layer = { color: rgb(tint).mul(1.8), alpha: halo.clamp(0, 1) };
  layer = over(layer, min(light.lit(rgb(0x1d1728)), rgb(0x2e2640)), disc);
  layer = over(layer, rgb(tint).mul(progress.mul(0.6).add(0.4)), coverage(r.sub(0.07)));
  layer = over(layer, tone, ring);
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha.mul(smoothstep(0.5, 0.46, r));
  return material;
}

export const flat = (width: number, depth: number): THREE.BufferGeometry => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

