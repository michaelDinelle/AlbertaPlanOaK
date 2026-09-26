import * as THREE from 'three/webgpu';
import { abs, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, select, sin, smoothstep, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const RIM = 0.46;
export const FIGURE = 0.9;
export const KEY = 0.62;
export const HATCH = 0.94;
export const HOLE = 0.29;
export const CANDLE = new THREE.Vector2(0.1, 0.215);
const HAND = new THREE.Vector2(-0.12, 0.155);

export const BACKDROP = 0x04050a;
export const GOLD = 0xffd166;
export const FLAME = 0xffb866;
export const DEEP = 0xffc36b;
const STONE = 0x6c6964;
const MORTAR = 0x14151d;
const WALL = 0x3a3d51;
const WAX = 0xf1e4c6;
const BRASS = 0xa8844f;
const WOOD = 0x6e4c2f;
const IRON = 0x2a2c33;
const BRASS_KEY = 0xe8b64c;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const turn = (p: N<'vec2'>, angle: number) => vec2(p.x.mul(Math.cos(angle)).add(p.y.mul(Math.sin(angle))), p.y.mul(Math.cos(angle)).sub(p.x.mul(Math.sin(angle))));
const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) => length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));
const hash = (cell: N<'vec2'>, a: number, b: number) => fract(sin(dot(cell, vec2(a, b))).mul(43758.5453));

function smin(a: N<'float'>, b: N<'float'>, k: number): N<'float'> {
  const h = max(float(k).sub(abs(a.sub(b))), 0).div(k);
  return min(a, b).sub(h.mul(h).mul(k / 4));
}

class Layers {
  pre: N<'vec3'> = vec3(0);
  alpha: N<'float'> = float(0);

  over(cover: N<'float'>, color: N<'vec3'>): void {
    this.pre = color.mul(cover).add(this.pre.mul(cover.oneMinus()));
    this.alpha = cover.add(this.alpha.mul(cover.oneMinus()));
  }

  material(): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    material.colorNode = this.pre.div(max(this.alpha, 1e-4));
    material.opacityNode = this.alpha;
    return material;
  }
}

const ghostInk = (accent: number): N<'vec3'> => mix(rgb(accent), vec3(1), 0.22).mul(1.15);

const soft = (lit: Lit, albedo: N<'vec3'>): N<'vec3'> => {
  const level = lit(vec3(1));
  return albedo.mul(level.mul(1.6).div(level.add(0.6)));
};

export function boardMaterial(side: number, span: number, lit: Lit): THREE.MeshBasicNodeMaterial {
  const half = side / 2, centre = vec2(half);
  const p = uv().sub(0.5).mul(span), q = vec2(p.x.add(half), float(half).sub(p.y));
  const cell = floor(q), local = fract(q).sub(0.5);
  const tone = hash(cell, 12.9898, 78.233), tilt = hash(cell, 39.3468, 11.135);
  const slab = sdBox(local.sub(vec2(tone.sub(0.5), tilt.sub(0.5)).mul(0.016)), vec2(0.472), 0.05);
  const mottle = fractal(q.mul(2.7).add(cell.mul(3.1)), 4), fine = fractal(q.mul(13).add(7.1), 2);
  const vein = smoothstep(0.035, 0, abs(fractal(q.mul(1.3).add(cell.mul(0.37)).add(19.4), 3))).mul(smoothstep(0.62, 0.85, hash(cell, 7.13, 3.71)));
  const edge = smoothstep(-0.06, -0.004, slab), sheen = smoothstep(-0.05, -0.012, slab).mul(smoothstep(-0.004, -0.02, slab)).mul(clamp(local.y.negate().sub(local.x).mul(2.2), 0, 1));
  let stone: N<'vec3'> = rgb(STONE).mul(tone.sub(0.5).mul(0.24).add(1)).mul(mottle.mul(0.2).add(1)).mul(fine.mul(0.08).add(1));
  stone = stone.mul(vein.mul(-0.45).add(1)).mul(edge.mul(-0.26).add(1)).add(rgb(0xc9c2b6).mul(sheen.mul(0.12)));
  const dRoom = sdBox(q.sub(centre), vec2(half));
  const shade = smoothstep(0, 0.55, dRoom.negate()).mul(0.5).add(0.5).mul(smoothstep(0, 0.14, dRoom.negate()).mul(0.3).add(0.7));
  const floorColor = lit(mix(rgb(MORTAR), stone, coverage(slab)).mul(shade));

  const off = abs(q.sub(centre)).sub(half), corner = off.x.greaterThan(0).and(off.y.greaterThan(0));
  const vertical = off.x.greaterThan(off.y);
  const along = select(vertical, q.y, q.x), across = max(off.x, off.y);
  const stretch = side / Math.round(side / 0.78), bit = fract(along.div(stretch)).sub(0.5).mul(stretch);
  const block = select(corner, sdBox(off.sub(RIM / 2), vec2(RIM / 2 - 0.035), 0.05), sdBox(vec2(bit, across.sub(RIM / 2)), vec2(stretch / 2 - 0.035, RIM / 2 - 0.035), 0.05));
  const blockTone = hash(floor(q.div(stretch)), 17.1, 5.3);
  const top = rgb(WALL).mul(blockTone.sub(0.5).mul(0.3).add(1)).mul(fractal(q.mul(4.1).add(3.3), 3).mul(0.22).add(1)).mul(smoothstep(-0.07, -0.005, block).mul(-0.45).add(1));
  const dOuter = sdBox(q.sub(centre), vec2(half + RIM), 0.12);
  const wallColor = lit(mix(rgb(MORTAR).mul(0.7), top, coverage(block)));
  const drop = smoothstep(-0.05, 0.55, sdBox(q.sub(centre).sub(vec2(0.05, 0.1)), vec2(half + RIM), 0.2)).oneMinus().mul(0.75);

  const board = new Layers();
  board.over(drop, vec3(0));
  board.over(coverage(dOuter), wallColor);
  board.over(coverage(dRoom), floorColor);
  return board.material();
}

const keyShape = (p: N<'vec2'>): N<'float'> => {
  const bow = abs(sdCircle(p.sub(vec2(-0.125, 0)), 0.056)).sub(0.021);
  const shaft = sdBox(p.sub(vec2(0.035, 0)), vec2(0.108, 0.016), 0.008);
  const collar = sdBox(p.sub(vec2(-0.052, 0)), vec2(0.012, 0.032), 0.006);
  const bit = max(sdBox(p.sub(vec2(0.112, -0.036)), vec2(0.031, 0.031), 0.004), sdBox(p.sub(vec2(0.112, -0.05)), vec2(0.009, 0.03)).negate());
  return min(min(bow, shaft), min(collar, bit));
};

const brass = (p: N<'vec2'>, d: N<'float'>): N<'vec3'> =>
  mix(rgb(0x7a5518), rgb(BRASS_KEY), smoothstep(-0.02, 0.0, d.negate().sub(0.004)).mul(0.35).add(0.65)).mul(p.x.add(p.y).mul(1.1).add(1.05));

export interface KeyArt { material: THREE.MeshBasicNodeMaterial; ghost: Level; flash: Level; time: Level }

export function keyMaterial(accent: number): KeyArt {
  const ghost = uniform(1), flash = uniform(0), time = uniform(0);
  const p = turn(uv().sub(0.5).mul(KEY), 0.62), d = keyShape(p);
  const shimmer = sin(time.mul(1.7).add(p.x.mul(9))).mul(0.12).add(0.88);
  const ink = ghostInk(accent);
  const art = new Layers();
  art.over(coverage(d).mul(0.14).mul(ghost), ink);
  art.over(coverage(abs(d).sub(0.0045)).mul(0.78).mul(ghost).mul(shimmer), ink);
  const ring = abs(length(p).sub(flash.oneMinus().mul(0.2).add(0.08))).sub(flash.mul(0.01).add(0.002));
  art.over(coverage(ring).mul(flash.mul(0.9)), rgb(GOLD).mul(1.4));
  const halo = exp(max(d, 0).mul(-26)).mul(0.55).mul(flash).mul(smoothstep(0.3, 0.2, length(p)));
  art.over(halo, rgb(GOLD).mul(1.6));
  art.over(coverage(d).mul(flash), brass(p, d).mul(flash.mul(1.4).add(1)));
  return { material: art.material(), ghost, flash, time };
}

export interface HatchArt { material: THREE.MeshBasicNodeMaterial; open: Level; ghost: Level; seek: Level; time: Level }

export function hatchMaterial(accent: number, lit: Lit): HatchArt {
  const open = uniform(0), ghost = uniform(1), seek = uniform(0), time = uniform(0);
  const p = uv().sub(0.5).mul(HATCH), ink = ghostInk(accent);
  const frame = sdBox(p, vec2(HOLE + 0.07), 0.035), hole = sdBox(p, vec2(HOLE), 0.012);
  const real = smoothstep(0, 0.25, open);
  const swing = cos(open.mul(Math.PI / 2)), reach = swing.mul(HOLE).add(0.035);
  const leafL = sdBox(p.sub(vec2(reach.mul(0.5).sub(HOLE), 0)), vec2(reach.mul(0.5), HOLE)), leafR = sdBox(p.sub(vec2(float(HOLE).sub(reach.mul(0.5)), 0)), vec2(reach.mul(0.5), HOLE));
  const leaves = min(leafL, leafR);
  const across = select(p.x.lessThan(0), p.x.add(HOLE).div(max(reach, 1e-3)), float(HOLE).sub(p.x).div(max(reach, 1e-3)));
  const plank = abs(fract(across.mul(swing.mul(3).add(1)).add(0.5)).sub(0.5)).div(swing.mul(3).add(1));
  const strap = min(abs(p.y.sub(0.17)), abs(p.y.add(0.17))).sub(0.028);
  let wood: N<'vec3'> = rgb(WOOD).mul(fractal(vec2(across.mul(3), p.y.mul(9)), 3).mul(0.18).add(1)).mul(swing.mul(0.55).add(0.45));
  wood = mix(wood, rgb(0x2b1d12), coverage(plank.sub(0.012)).mul(0.8));
  wood = mix(wood, rgb(IRON).mul(swing.mul(0.5).add(0.6)), coverage(strap));
  const plate = sdBox(p, vec2(0.045, 0.06), 0.012), keyhole = min(sdCircle(p.sub(vec2(0, 0.012)), 0.014), sdBox(p.sub(vec2(0, -0.016)), vec2(0.006, 0.02)));
  const depth = clamp(p.y.add(HOLE).div(2 * HOLE), 0, 0.999).mul(5), tread = floor(depth), within = fract(depth);
  const glowDown = tread.add(within.mul(0.4)).div(5);
  const well = rgb(DEEP).mul(mix(float(0.12), float(1.35), glowDown.mul(glowDown))).mul(smoothstep(0, 0.12, within).mul(0.55).add(0.45)).mul(exp(max(abs(p.x).sub(0.2), 0).mul(-9)));
  const rimLight = exp(hole.mul(-40)).mul(open);
  let body: N<'vec3'> = lit(rgb(0x3d3a3a).mul(fractal(p.mul(9), 2).mul(0.2).add(1)));
  body = body.add(rgb(DEEP).mul(rimLight.mul(0.9)));
  const art = new Layers();
  const dash = select(abs(p.x).greaterThan(abs(p.y)), p.y, p.x);
  const dashes = coverage(abs(fract(dash.mul(8.5).add(0.25)).sub(0.5)).sub(0.2).div(8.5));
  const pulse = sin(time.mul(3.1)).mul(0.5).add(0.5).mul(seek);
  const lineAlpha = ghost.mul(real.oneMinus()).mul(seek.mul(0.35).add(0.5).add(pulse.mul(0.25)));
  art.over(coverage(abs(frame).sub(0.0055)).mul(dashes).mul(lineAlpha).mul(1.5).min(1), ink);
  art.over(coverage(abs(hole).sub(0.0035)).mul(0.4).mul(lineAlpha), ink);
  art.over(coverage(abs(p.x).sub(0.0035)).mul(coverage(hole)).mul(0.3).mul(lineAlpha), ink);
  art.over(coverage(keyhole).mul(0.85).mul(lineAlpha), ink);
  art.over(coverage(frame).mul(real), body);
  art.over(coverage(hole).mul(real), mix(rgb(0x120d09), well, open));
  art.over(coverage(leaves).mul(real), wood.mul(lit(vec3(1)).max(open.mul(0.9))));
  const shut = smoothstep(0.55, 0.75, swing).mul(real);
  art.over(coverage(max(plate, leaves)).mul(shut), rgb(IRON).mul(1.4));
  art.over(coverage(keyhole).mul(shut), rgb(0x050403));
  return { material: art.material(), open, ghost, seek, time };
}

export interface FigureArt { material: THREE.MeshBasicNodeMaterial; carry: Level; flame: Level; fade: Level }

export function figureMaterial(accent: number, lit: Lit): FigureArt {
  const carry = uniform(0), flame = uniform(1), fade = uniform(1);
  const p = uv().sub(0.5).mul(FIGURE);
  const shadow = smoothstep(-0.03, 0.07, ellipse(p, [-0.01, -0.045], [0.245, 0.2])).oneMinus().mul(0.45);
  const cloak = smin(ellipse(p, [0, -0.01], [0.205, 0.118]), ellipse(p, [0, -0.09], [0.165, 0.1]), 0.07);
  const candle = vec2(CANDLE.x, CANDLE.y), grip = candle.sub(vec2(0, 0.05)), hand = mix(vec2(-0.105, 0.115), vec2(HAND.x, HAND.y), carry);
  const right = sdSegment(p, vec2(0.14, 0.025), grip, 0.031), left = sdSegment(p, vec2(-0.14, 0.025), hand, 0.03);
  const fist = min(sdCircle(p.sub(grip), 0.029), sdCircle(p.sub(hand), 0.028));
  const hood = smin(sdCircle(p.sub(vec2(0, 0.02)), 0.09), sdCircle(p.sub(vec2(0, -0.058)), 0.05), 0.06);
  const dish = sdCircle(p.sub(candle), 0.044), wax = sdCircle(p.sub(candle), 0.025), wick = sdCircle(p.sub(candle), 0.013);
  const keyAt = turn(p.sub(hand).sub(vec2(-0.02, 0.05)).div(0.6), -1.9), key = keyShape(keyAt).mul(0.6);
  const silhouette = min(min(cloak, min(right, left)), min(min(hood, fist), dish));
  const cloth = rgb(accent).mul(0.5), sleeve = rgb(accent).mul(0.4), cowl = rgb(accent).mul(0.28);
  const art = new Layers();
  art.over(shadow, vec3(0));
  art.over(coverage(silhouette.sub(0.013)), rgb(0x07070d));
  art.over(coverage(cloak), soft(lit, cloth.mul(p.y.mul(1.1).add(0.95))));
  art.over(coverage(abs(cloak.add(0.012)).sub(0.004)).mul(coverage(cloak)).mul(0.7), soft(lit, mix(rgb(accent), vec3(1), 0.3).mul(0.8)));
  art.over(coverage(min(right, left)), soft(lit, sleeve));
  art.over(coverage(fist), soft(lit, rgb(0xc99a7a)));
  art.over(coverage(hood.add(0.004)), soft(lit, cowl.mul(p.y.sub(0.01).mul(2.6).add(1)).mul(p.x.mul(1.8).add(1))));
  art.over(coverage(sdSegment(p, vec2(0, -0.1), vec2(0, -0.01), 0.0035)).mul(0.6), rgb(0x07070d));
  art.over(coverage(key).mul(carry), brass(keyAt, key).mul(1.3));
  art.over(coverage(dish), soft(lit, rgb(BRASS)).add(rgb(FLAME).mul(0.2)));
  art.over(coverage(wax), rgb(WAX).mul(flame.mul(0.5).add(0.75)));
  const glow = exp(length(p.sub(candle)).mul(-30)).mul(flame).mul(0.7);
  art.over(glow.min(1), rgb(FLAME).mul(2.2));
  art.over(coverage(wick), rgb(0xfff1c9).mul(flame.mul(2.4).add(1.2)));
  const material = art.material();
  material.opacityNode = art.alpha.mul(fade);
  return { material, carry, flame, fade };
}

export function routeMaterial(accent: number): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec2'>('aRoute', 'vec2');
  const p = uv().sub(0.5).mul(vec2(1, 0.2)), t = fract(p.x.mul(4)).sub(0.5).div(4);
  const dot2 = sdCircle(vec2(t, p.y), 0.019);
  const art = new Layers();
  art.over(coverage(dot2).mul(state.x), ghostInk(accent).mul(0.85));
  return art.material();
}

export function startMaterial(accent: number): { material: THREE.MeshBasicNodeMaterial; level: Level } {
  const level = uniform(0);
  const r = length(uv().sub(0.5).mul(0.5));
  const art = new Layers();
  art.over(coverage(abs(r.sub(0.19)).sub(0.006)).mul(level).mul(0.55), ghostInk(accent));
  return { material: art.material(), level };
}
