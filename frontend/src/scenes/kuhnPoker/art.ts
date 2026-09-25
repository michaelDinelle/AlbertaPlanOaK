import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, dot, exp, float, floor, fract, instancedDynamicBufferAttribute, length, max, min, mix, positionWorld, pow, select, sin, smoothstep, step, texture, uniformArray, uv, vec2, vec3 } from 'three/tsl';
import type { Light2D } from '@kit/light2d';
import { coverage, fractal, grain, noise, rgb, sdBox, sdCircle } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';

type N<T extends string> = THREE.Node<T>;
type Scalar = N<'float'> | number;
export type Level = THREE.UniformNode<'float', number>;

export const CARD_W = 1.1;
export const CARD_H = 1.54;
export const CARD_QW = 1.6;
export const CARD_QH = 2.1;
export const CHIP_R = 0.27;
export const CHIP_Q = 0.8;
export const LABEL_W = 1.16;
export const LABEL_H = 0.4;
export const LABEL_QW = 1.5;
export const LABEL_QH = 0.72;
export const COIN_R = 0.36;
export const COIN_Q = 1.2;
export const PUCK_Q = 0.8;
export const LAMP_Q = 1.3;
export const TABLE = { halfL: 7.3, halfW: 4.1, corner: 2.7, rail: 0.52, trim: 0.16 };
export const GRID = { w: 3.74, h: 3.1, pad: 0.18, labels: 0.42, pitchX: 0.74, pitchY: 0.52, cellW: 0.64, cellH: 0.42, top: 0.83, head: 1.26, sub: 1.01, caption: -1.03, dots: -1.26 };
export const RECENT = 16;
export const SPOTS = 4;

export const IVORY = 0xf3ead6;
export const BONE = 0xe6d9bd;
export const COPPER = 0xd08a55;
export const GOLD = TOKENS.celebrationGold;
const LOSS = TOKENS.failureBlue;
export const PRINT = 0xc9a45c;
const INK = 0x211a1f;
const CRIMSON = 0x8e2031;
const NAVY = 0x172241;
const FELT_HI = 0x6c1b2a;
const FELT_LO = 0x36101a;
const LEATHER = 0x120c0b;
const LEATHER_HI = 0x3e2c26;
const WOOD = 0x3f2414;
const WOOD_HI = 0x7c4b29;

const SANS = 'ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const SERIF = 'ui-serif, "New York", Georgia, "Times New Roman", serif';
export const RANK_NAMES = ['J', 'Q', 'K'] as const;
export const WORDS = ['CHECK', 'BET', 'CALL', 'FOLD', 'PASS', '1ST', 'YOU ACT FIRST', 'YOU ACT SECOND', 'OPEN', 'THEY BET', 'THEY CHECK', 'LAST HAND', 'RECENT HANDS', 'KUHN POKER'] as const;
export type Word = (typeof WORDS)[number];
const ENTRIES = [
  ...'0123456789+−'.split('').map((text) => ({ text, family: SANS, weight: 800 })),
  ...RANK_NAMES.map((text) => ({ text, family: SERIF, weight: 700 })),
  ...WORDS.map((text) => ({ text, family: SANS, weight: 800 })),
];
export const PLUS = 10;
export const MINUS = 11;
export const RANK = 12;
export const wordIndex = (word: Word): number => RANK + RANK_NAMES.length + WORDS.indexOf(word);
const CELL_W = 1024;
const CELL_H = 128;
const CAP = 0.56;
const ASPECT = CELL_W / CELL_H;

export const TINT = { agent: 0, them: 1, win: 2, loss: 3, print: 4 } as const;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 32).rotateX(-Math.PI / 2);
const scalar = (value: Scalar): N<'float'> => (typeof value === 'number' ? float(value) : value);
const local = (w: number, h: number): N<'vec2'> => uv().sub(0.5).mul(vec2(w, h));
const surface = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
const rhombus = (p: N<'vec2'>, a: number, b: number): N<'float'> => abs(p.x).mul(b).add(abs(p.y).mul(a)).sub(a * b).div(Math.hypot(a, b));
const ring = (d: N<'float'>, half: number): N<'float'> => coverage(abs(d).sub(half));
const SUN = vec2(-0.55, 0.83).normalize();

interface Layer { color: N<'vec3'>; alpha: N<'float'> }
const over = (under: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer => {
  const a = alpha.add(under.alpha.mul(alpha.oneMinus()));
  return { color: color.mul(alpha).add(under.color.mul(under.alpha).mul(alpha.oneMinus())).div(max(a, 1e-4)), alpha: a };
};

export function atlas(): THREE.DataTexture {
  const canvas = document.createElement('canvas');
  canvas.width = CELL_W;
  canvas.height = CELL_H * ENTRIES.length;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Kuhn poker lettering needs a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  ENTRIES.forEach(({ text, family, weight }, k) => {
    context.font = `${weight} 100px ${family}`;
    let size = (100 * CAP * CELL_H) / context.measureText('H').actualBoundingBoxAscent;
    context.font = `${weight} ${size}px ${family}`;
    const wide = context.measureText(text).width;
    if (wide > CELL_W * 0.94) {
      size *= (CELL_W * 0.94) / wide;
      context.font = `${weight} ${size}px ${family}`;
    }
    context.fillText(text, CELL_W / 2, k * CELL_H + CELL_H / 2 + context.measureText('H').actualBoundingBoxAscent / 2);
  });
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data, alpha = new Uint8Array(canvas.width * canvas.height);
  for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) alpha[(canvas.height - 1 - y) * canvas.width + x] = pixels[4 * (y * canvas.width + x) + 3]!;
  const out = new THREE.DataTexture(alpha, canvas.width, canvas.height, THREE.RedFormat, THREE.UnsignedByteType);
  out.minFilter = THREE.LinearMipmapLinearFilter;
  out.magFilter = THREE.LinearFilter;
  out.generateMipmaps = true;
  out.needsUpdate = true;
  return out;
}

export function glyph(map: THREE.Texture, p: N<'vec2'>, centre: N<'vec2'>, cap: Scalar, index: Scalar): N<'float'> {
  const q = p.sub(centre).div(scalar(cap).div(CAP)).add(vec2(ASPECT / 2, 0.5));
  const inside = step(0, q.x).mul(step(q.x, ASPECT)).mul(step(0, q.y)).mul(step(q.y, 1));
  const c = clamp(q, vec2(0, 0), vec2(ASPECT, 1));
  return texture(map, vec2(c.x.div(ASPECT), float(ENTRIES.length - 1).sub(scalar(index)).add(c.y).div(ENTRIES.length))).r.mul(inside);
}

export function palette(accent: number) {
  return uniformArray([accent, BONE, GOLD, LOSS, PRINT].map((hex) => new THREE.Color(hex)), 'color');
}
export type Palette = ReturnType<typeof palette>;
const tintOf = (tints: Palette, index: N<'float'>): N<'vec3'> => tints.element(clamp(index, 0, Object.keys(TINT).length - 1).toInt()) as unknown as N<'vec3'>;

export interface Print { x: number; z: number; w: number; h: number; r: number }

export function tableMaterial(light: Light2D, map: THREE.Texture, prints: readonly Print[], captions: readonly { word: Word; x: number; z: number; cap: number }[]): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const p = vec2(positionWorld.x, positionWorld.z.negate());
  const outer = sdBox(p, vec2(TABLE.halfL, TABLE.halfW), TABLE.corner), inner = outer.add(TABLE.rail), felt = inner.add(TABLE.trim);
  const boards = fract(p.x.div(0.9).add(floor(p.y.div(2.6)).mul(0.37)));
  const planks = fractal(vec2(p.x.mul(0.5), p.y.mul(7)), 3).mul(0.5).add(0.5);
  const floorTone = mix(rgb(0x0c090a), rgb(0x1a1314), planks.mul(0.6)).mul(float(1).sub(smoothstep(0.47, 0.5, abs(boards.sub(0.5))).mul(0.45)));
  const floorColor = floorTone.mul(float(1).sub(exp(max(outer, 0).mul(-2.4)).mul(0.7)));
  const across = clamp(outer.negate().div(TABLE.rail), 0, 1), cushion = max(sin(across.mul(Math.PI)), 0);
  const crest = across.sub(0.42).mul(4.5), glint = exp(abs(across.sub(0.34)).mul(-38)), rim = exp(across.mul(-30));
  const leather = mix(rgb(LEATHER), rgb(LEATHER_HI), exp(crest.mul(crest).negate()).mul(0.8).add(pow(cushion, float(3)).mul(0.2)))
    .mul(fractal(p.mul(11), 2).mul(0.08).add(1)).add(rgb(0xffd9b0).mul(glint.mul(0.07).add(rim.mul(0.05))));
  const wood = mix(rgb(WOOD), rgb(WOOD_HI), noise(p.mul(3.1)).mul(0.25).add(0.3).add(clamp(inner.negate().div(TABLE.trim), 0, 1).mul(0.3)));
  const inlay = ring(inner.add(TABLE.trim * 0.5), 0.012);
  const reach = length(p.mul(vec2(1 / TABLE.halfL, 1 / TABLE.halfW)));
  const fibre = fractal(p.mul(36), 2).mul(0.04).add(grain(floor(p.mul(260))).x.mul(0.035)).add(0.98);
  let top = mix(rgb(FELT_HI), rgb(FELT_LO), smoothstep(0.08, 1.05, reach)).mul(fibre);
  top = top.mul(float(1).sub(exp(felt.mul(7)).mul(0.45)));
  let ink: N<'float'> = ring(felt.add(0.24), 0.011);
  for (const zone of prints) ink = max(ink, ring(sdBox(p.sub(vec2(zone.x, -zone.z)), vec2(zone.w / 2, zone.h / 2), zone.r), 0.011));
  for (const caption of captions) ink = max(ink, glyph(map, p, vec2(caption.x, -caption.z), caption.cap, wordIndex(caption.word)));
  top = mix(top, rgb(PRINT).mul(0.9), ink.mul(0.45));
  let color: N<'vec3'> = floorColor;
  color = mix(color, leather, coverage(outer));
  color = mix(color, mix(wood, rgb(0xd9b36a), inlay.mul(0.8)), coverage(inner));
  color = mix(color, top, coverage(felt));
  material.colorNode = light.lit(color);
  return material;
}

export interface CardBuffers { look: THREE.InstancedBufferAttribute; pose: THREE.InstancedBufferAttribute }

export function cardMaterial(light: Light2D, map: THREE.Texture, buffers: CardBuffers, tints: Palette): THREE.MeshBasicNodeMaterial {
  const look = instancedDynamicBufferAttribute(buffers.look, 'vec4') as N<'vec4'>, pose = instancedDynamicBufferAttribute(buffers.pose, 'vec4') as N<'vec4'>;
  const rank = look.x, face = look.y, glow = look.z, alpha = look.w, lift = pose.x, tint = pose.y, dim = pose.z;
  const material = surface();
  const t = local(CARD_QW, CARD_QH), half = vec2(CARD_W / 2, CARD_H / 2), d = sdBox(t, half, 0.085), body = coverage(d);
  const shade = t.y.div(CARD_H).add(0.5);
  const paper = mix(rgb(0xe2d5ba), rgb(IVORY), shade).mul(fractal(t.mul(18), 2).mul(0.025).add(1));
  const frame = ring(d.add(0.075), 0.008);
  const big = glyph(map, t, vec2(0, 0.1), 0.5, rank.add(RANK));
  const corner = max(glyph(map, t, vec2(-0.36, 0.55), 0.15, rank.add(RANK)), glyph(map, t.negate(), vec2(-0.36, 0.55), 0.15, rank.add(RANK)));
  let pips: N<'float'> = float(0);
  for (let k = 0; k < RANK_NAMES.length; k++) {
    const x = float(k).sub(rank.mul(0.5)).mul(0.17);
    pips = max(pips, coverage(rhombus(t.sub(vec2(x, -0.4)), 0.055, 0.075)).mul(step(k - 0.5, rank)));
  }
  let faceColor: N<'vec3'> = mix(paper, rgb(CRIMSON), frame.mul(0.55));
  faceColor = mix(faceColor, rgb(INK), max(big, corner));
  faceColor = mix(faceColor, rgb(CRIMSON), pips);
  const a = t.x.add(t.y).mul(6.5), b = t.x.sub(t.y).mul(6.5);
  const lattice = coverage(min(abs(fract(a).sub(0.5)), abs(fract(b).sub(0.5))).sub(0.045)).mul(step(d, -0.13));
  const field = mix(rgb(0x0f1730), rgb(NAVY), shade);
  let backColor: N<'vec3'> = mix(field, rgb(PRINT).mul(0.75), lattice.mul(0.4));
  backColor = mix(backColor, rgb(PRINT), max(ring(d.add(0.075), 0.011), ring(d.add(0.13), 0.006)).mul(0.85));
  backColor = mix(backColor, rgb(PRINT).mul(0.9), ring(sdCircle(t.mul(vec2(1, 0.72)), 0.2), 0.012));
  const cardColor = mix(backColor, faceColor, face).mul(dim.mul(-0.45).add(1));
  const lit = min(light.lit(cardColor), cardColor.mul(1.15));
  const shadowD = sdBox(t.sub(vec2(0.04, lift.mul(-0.12).sub(0.06))), half, 0.085);
  const shadow = smoothstep(lift.mul(0.1).add(0.07), -0.05, shadowD).mul(lift.mul(-0.12).add(0.45));
  const halo = exp(max(d, 0).mul(-11)).mul(step(0, d)).mul(smoothstep(0.24, 0.1, d)).mul(glow);
  let layer: Layer = { color: vec3(0), alpha: shadow };
  layer = over(layer, tintOf(tints, tint).mul(1.5), halo);
  layer = over(layer, lit, body);
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha.mul(alpha);
  return material;
}

export function chipMaterial(light: Light2D, accent: number, look: THREE.InstancedBufferAttribute): THREE.MeshBasicNodeMaterial {
  const v = instancedDynamicBufferAttribute(look, 'vec4') as N<'vec4'>, owner = v.x, alpha = v.y, lift = v.z, glow = v.w;
  const material = surface();
  const t = local(CHIP_Q, CHIP_Q), r = length(t), d = r.sub(CHIP_R);
  const mint = rgb(accent), body = mix(mix(mint.mul(0.52), mint.mul(0.82), smoothstep(CHIP_R, 0, r)), mix(rgb(0xb9aa8c), rgb(BONE), smoothstep(CHIP_R, 0, r)), owner);
  const spotColor = mix(rgb(IVORY), rgb(CRIMSON), owner);
  const turn = (2 * Math.PI) / 6, angle = atan(t.y, t.x), k = floor(angle.div(turn).add(0.5)).mul(turn);
  const q = vec2(t.x.mul(cos(k)).add(t.y.mul(sin(k))), t.y.mul(cos(k)).sub(t.x.mul(sin(k))));
  const spots = coverage(sdBox(q.sub(vec2(CHIP_R * 0.8, 0)), vec2(0.04, 0.055), 0.012));
  const band = ring(r.sub(CHIP_R * 0.58), 0.009);
  const inlay = coverage(r.sub(CHIP_R * 0.46));
  const facing = dot(t.div(max(r, 1e-4)), SUN);
  let color: N<'vec3'> = body.mul(smoothstep(CHIP_R, CHIP_R * 0.82, r).mul(0.22).add(0.78));
  color = mix(color, spotColor, spots);
  color = mix(color, spotColor.mul(0.9), band.mul(0.75));
  color = mix(color, body.mul(1.12).add(vec3(0.03)), inlay);
  color = color.mul(facing.mul(smoothstep(CHIP_R * 0.5, CHIP_R, r)).mul(0.18).add(1));
  const star = coverage(rhombus(t, 0.075, 0.075)).mul(owner.oneMinus()).add(ring(r.sub(0.05), 0.012).mul(owner));
  color = mix(color, spotColor, star.mul(0.85));
  const shadow = smoothstep(lift.mul(0.08).add(0.05), -0.03, length(t.sub(vec2(0.03, lift.mul(-0.1).sub(0.045)))).sub(CHIP_R)).mul(0.5);
  const halo = exp(max(d, 0).mul(-12)).mul(step(0, d)).mul(smoothstep(0.125, 0.04, d)).mul(glow);
  let layer: Layer = { color: vec3(0), alpha: shadow };
  layer = over(layer, rgb(GOLD).mul(1.6), halo);
  layer = over(layer, min(light.lit(color), color.mul(1.2)), coverage(d));
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha.mul(alpha);
  return material;
}

export function labelMaterial(map: THREE.Texture, look: THREE.InstancedBufferAttribute, tints: Palette): THREE.MeshBasicNodeMaterial {
  const v = instancedDynamicBufferAttribute(look, 'vec4') as N<'vec4'>, kind = v.x, value = v.y, tint = v.z, alpha = v.w;
  const material = surface();
  const t = local(LABEL_QW, LABEL_QH), d = sdBox(t, vec2(LABEL_W / 2, LABEL_H / 2), LABEL_H / 2);
  const tone = tintOf(tints, tint);
  const word = glyph(map, t, vec2(0, 0), 0.16, max(kind, 0));
  const magnitude = abs(value), tens = floor(magnitude.div(10)), ones = magnitude.sub(tens.mul(10)), two = step(9.5, magnitude), signed = step(0.5, magnitude);
  const count = signed.add(two).add(1), advance = 0.15, start = count.sub(1).mul(advance * -0.5);
  const sign = glyph(map, t, vec2(start, 0), 0.2, select(value.greaterThan(0), float(PLUS), float(MINUS))).mul(signed);
  const tensGlyph = glyph(map, t, vec2(start.add(signed.mul(advance)), 0), 0.2, tens).mul(two);
  const onesGlyph = glyph(map, t, vec2(start.add(signed.add(two).mul(advance)), 0), 0.2, ones);
  const number = max(sign, max(tensGlyph, onesGlyph));
  const text = select(kind.greaterThanEqual(0), word, number);
  let color: N<'vec3'> = mix(rgb(0x140c0f), tone.mul(0.85), ring(d.add(0.022), 0.012));
  color = mix(color, tone.mul(1.3), text);
  const halo = exp(max(d, 0).mul(-16)).mul(step(0, d)).mul(smoothstep(0.15, 0.05, d)).mul(0.35);
  let layer: Layer = { color: tone, alpha: halo };
  layer = over(layer, color, max(coverage(d), text));
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha.mul(alpha);
  return material;
}

export function coinMaterial(light: Light2D, map: THREE.Texture, face: Level, glow: Level): THREE.MeshBasicNodeMaterial {
  const material = surface();
  const t = local(COIN_Q, COIN_Q), r = length(t), d = r.sub(COIN_R);
  const facing = dot(t.div(COIN_R), SUN);
  const metal = mix(rgb(0x6b3a1f), rgb(COPPER), smoothstep(-1.1, 0.9, facing)).add(rgb(0xffe0c0).mul(pow(clamp(facing, 0, 1), float(6)).mul(0.35)));
  const rim = ring(r.sub(COIN_R * 0.84), 0.014);
  const pass = glyph(map, t, vec2(0, 0), 0.11, wordIndex('PASS')), bet = glyph(map, t, vec2(0, 0), 0.13, wordIndex('BET'));
  const lift = glyph(map, t.add(vec2(0.012, -0.014)), vec2(0, 0), 0.11, wordIndex('PASS')).mul(step(face, 0.5)).add(glyph(map, t.add(vec2(0.012, -0.014)), vec2(0, 0), 0.13, wordIndex('BET')).mul(step(0.5, face)));
  const blank = max(ring(r.sub(0.13), 0.012), coverage(rhombus(t, 0.06, 0.06)));
  const mark = select(face.lessThan(-0.5), blank, select(face.greaterThan(0.5), bet, pass));
  const emboss = select(face.lessThan(-0.5), float(0), lift);
  let color: N<'vec3'> = mix(metal, rgb(0x4a2612), rim.mul(0.7));
  color = mix(color, rgb(0xffd2a8), emboss.mul(0.45).mul(mark.oneMinus()));
  color = mix(color, rgb(0x3d1f0f), mark.mul(0.9));
  const shadow = smoothstep(0.08, -0.04, length(t.sub(vec2(0.035, -0.05))).sub(COIN_R)).mul(0.5);
  const halo = exp(max(d, 0).mul(-9)).mul(step(0, d)).mul(smoothstep(0.23, 0.08, d)).mul(glow);
  let layer: Layer = { color: vec3(0), alpha: shadow };
  layer = over(layer, rgb(COPPER).mul(1.8), halo);
  layer = over(layer, min(light.lit(color), color.mul(1.25)).add(rgb(COPPER).mul(glow.mul(0.25))), coverage(d));
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha;
  return material;
}

export function puckMaterial(light: Light2D, map: THREE.Texture): THREE.MeshBasicNodeMaterial {
  const material = surface();
  const t = local(PUCK_Q, PUCK_Q), r = length(t), radius = 0.27, d = r.sub(radius);
  const facing = dot(t.div(radius), SUN);
  let color: N<'vec3'> = mix(rgb(0xcfc3a8), rgb(0xfbf6ea), smoothstep(-1, 0.8, facing));
  color = mix(color, rgb(INK), ring(r.sub(radius * 0.8), 0.01).mul(0.8));
  color = mix(color, rgb(INK), glyph(map, t, vec2(0, 0), 0.11, wordIndex('1ST')));
  const shadow = smoothstep(0.07, -0.03, length(t.sub(vec2(0.03, -0.045))).sub(radius)).mul(0.5);
  const layer = over({ color: vec3(0), alpha: shadow }, min(light.lit(color), color.mul(1.2)), coverage(d));
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha;
  return material;
}

export function lampMaterial(accent: number, level: Level): THREE.MeshBasicNodeMaterial {
  const material = surface();
  const t = local(LAMP_Q, LAMP_Q), r = length(t), tone = rgb(accent);
  const jewel = coverage(r.sub(0.13)), bezel = ring(r.sub(0.16), 0.03);
  const facing = dot(t.div(0.13), SUN);
  const core = mix(tone.mul(0.35), tone.mul(1.9), level).add(vec3(1).mul(smoothstep(0.4, 1, facing).mul(level.mul(0.5).add(0.15))));
  const brass = mix(rgb(0x6e5325), rgb(0xd9b36a), smoothstep(-1, 1, dot(t.div(max(r, 1e-4)), SUN)));
  const halo = exp(r.mul(-7)).mul(level).mul(smoothstep(LAMP_Q / 2, 0.18, r)).mul(0.9);
  let layer: Layer = { color: tone.mul(1.6), alpha: halo };
  layer = over(layer, brass, bezel);
  layer = over(layer, core, jewel);
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha;
  return material;
}

export interface Tally { rates: number[]; weights: number[]; recent: number[] }

export function gridMaterial(light: Light2D, map: THREE.Texture, accent: number): { material: THREE.MeshBasicNodeMaterial; tally: Tally } {
  const tally: Tally = { rates: new Array<number>(SPOTS * RANK_NAMES.length).fill(0), weights: new Array<number>(SPOTS * RANK_NAMES.length).fill(0), recent: new Array<number>(RECENT).fill(0) };
  const rates = uniformArray(tally.rates, 'float'), weights = uniformArray(tally.weights, 'float'), recent = uniformArray(tally.recent, 'float');
  const material = surface();
  const t = local(GRID.w, GRID.h), frame = sdBox(t, vec2(GRID.w / 2, GRID.h / 2), 0.2);
  const left = -GRID.w / 2 + GRID.pad + GRID.labels, print = rgb(PRINT), mint = rgb(accent);
  let ink: N<'float'> = ring(frame.add(0.05), 0.01);
  ink = max(ink, glyph(map, t, vec2(left + GRID.pitchX, GRID.head), 0.1, wordIndex('YOU ACT FIRST')));
  ink = max(ink, glyph(map, t, vec2(left + 3 * GRID.pitchX, GRID.head), 0.1, wordIndex('YOU ACT SECOND')));
  const subs: Word[] = ['OPEN', 'THEY BET', 'THEY CHECK', 'THEY BET'];
  subs.forEach((word, k) => { ink = max(ink, glyph(map, t, vec2(left + (k + 0.5) * GRID.pitchX, GRID.sub), 0.078, wordIndex(word))); });
  for (const group of [1, 3]) ink = max(ink, coverage(max(abs(t.y.sub((GRID.head + GRID.sub) / 2)).sub(0.006), abs(t.x.sub(left + group * GRID.pitchX)).sub(0.9 * GRID.pitchX))));
  const floorY = GRID.top - RANK_NAMES.length * GRID.pitchY, roofY = GRID.head + 0.12;
  ink = max(ink, coverage(max(abs(t.x.sub(left + 2 * GRID.pitchX)).sub(0.006), abs(t.y.sub((roofY + floorY) / 2)).sub((roofY - floorY) / 2))).mul(0.6));
  for (let row = 0; row < RANK_NAMES.length; row++) ink = max(ink, glyph(map, t, vec2(left - GRID.labels / 2, GRID.top - (row + 0.5) * GRID.pitchY), 0.2, RANK + RANK_NAMES.length - 1 - row));
  ink = max(ink, glyph(map, t, vec2(left + 2 * GRID.pitchX, GRID.caption), 0.078, wordIndex('RECENT HANDS')));
  const column = floor(t.x.sub(left).div(GRID.pitchX)), row = floor(float(GRID.top).sub(t.y).div(GRID.pitchY));
  const within = step(0, column).mul(step(column, SPOTS - 0.5)).mul(step(0, row)).mul(step(row, RANK_NAMES.length - 0.5));
  const index = float(RANK_NAMES.length - 1).sub(clamp(row, 0, RANK_NAMES.length - 1)).mul(SPOTS).add(clamp(column, 0, SPOTS - 1));
  const rate = rates.element(index.toInt()) as unknown as N<'float'>, weight = weights.element(index.toInt()) as unknown as N<'float'>;
  const centre = vec2(column.add(0.5).mul(GRID.pitchX).add(left), float(GRID.top).sub(row.add(0.5).mul(GRID.pitchY)));
  const q = t.sub(centre), cell = sdBox(q, vec2(GRID.cellW / 2, GRID.cellH / 2), 0.07);
  const filled = coverage(q.x.sub(rate.sub(0.5).mul(GRID.cellW)));
  const inside = coverage(cell).mul(within);
  const cellColor = mix(mix(rgb(0x241416), rgb(BONE).mul(0.2), weight), mix(rgb(0x241416), mint.mul(0.62), weight), filled);
  const outline = ring(cell, 0.008).mul(within);
  const slot = floor(t.x.sub(left).div((SPOTS * GRID.pitchX) / RECENT)), dotCentre = vec2(slot.add(0.5).mul((SPOTS * GRID.pitchX) / RECENT).add(left), GRID.dots);
  const inRow = step(0, slot).mul(step(slot, RECENT - 0.5));
  const result = recent.element(clamp(slot, 0, RECENT - 1).toInt()) as unknown as N<'float'>;
  const size = min(float(0.045).add(abs(result).mul(0.018)), 0.085), dotD = length(t.sub(dotCentre)).sub(size);
  const dotFill = coverage(dotD).mul(inRow).mul(step(0.5, abs(result)));
  const dotRing = ring(length(t.sub(dotCentre)).sub(0.04), 0.006).mul(inRow).mul(step(abs(result), 0.5));
  const dotTone = select(result.greaterThan(0), rgb(GOLD), rgb(LOSS));
  let layer: Layer = { color: rgb(0x14050a), alpha: coverage(frame).mul(0.5) };
  layer = over(layer, light.lit(print), max(ink, dotRing.mul(0.5)).mul(0.75));
  const crest = coverage(abs(q.x.sub(rate.sub(0.5).mul(GRID.cellW))).sub(0.012)).mul(step(0.02, rate)).mul(step(rate, 0.98)).mul(weight);
  layer = over(layer, light.lit(cellColor).add(mint.mul(filled.mul(weight).mul(0.1))), inside.mul(0.94));
  layer = over(layer, mint.mul(1.3), crest.mul(inside));
  layer = over(layer, light.lit(print), outline.mul(0.55));
  layer = over(layer, dotTone.mul(1.25), dotFill);
  material.colorNode = layer.color;
  material.opacityNode = layer.alpha;
  return { material, tally };
}
