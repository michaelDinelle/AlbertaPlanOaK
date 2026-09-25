import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, int, length, max, min, mix, positionWorld, select, sin, smoothstep, sqrt, texture, uniform, uniformArray, uv, uvec2, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;
export type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;

export const ANNEX = 0.9;
export const WALL = 0.14;
export const DOOR = 0.13;
export const SPAN = 1 - WALL;
export const HINGE = 1 + WALL / 2 + DOOR / 2;
export const MIDDLE = SPAN / 2 - DOOR / 2;
export const FIGURE = 0.84;
export const KEY = 0.5;
export const PLINTH = 0.62;
export const HAND = new THREE.Vector2(0.19, 0.17);
export const LANTERN = new THREE.Vector2(0.22, -0.15);
export const LOCK = new THREE.Vector2(DOOR / 2 + 0.03, MIDDLE);
export const LEDGER = 16;

export const GOLD = TOKENS.celebrationGold;
export const MISS = TOKENS.failureBlue;
export const AMBER = 0xffb257;
export const FROST = 0x6e92ff;
export const FLAME = 0xffc987;
const SLATE = 0x454b5a;
const GROUT = 0x14161d;
const STONE = 0x5a5856;
const MORTAR = 0x17161a;
const VOID = 0x07080d;
const WOOD = 0x7a4f2c;
const IRON = 0x2a2c33;
const VELVET = 0x7a1f2e;
const ROBE = 0x2d6a70;
const HOOD = 0x1d4a50;
const TRIM = 0xb89150;
const BRICK = 0.33;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 32).rotateX(-Math.PI / 2);

const local = (width: number, depth = width): N<'vec2'> => uv().sub(0.5).mul(vec2(width, depth));
const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) => length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));
const hash = (cell: N<'vec2'>, a: number, b: number) => fract(sin(dot(cell, vec2(a, b))).mul(43758.5453));
const soft = (d: N<'float'>, blur: number) => smoothstep(-blur, blur, d).oneMinus();
const dashes = (v: N<'float'>) => smoothstep(-0.05, 0.05, v);

function sheet(color: N<'vec3'>, opacity: N<'float'>): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  return material;
}

function solid(color: N<'vec3'>): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  return material;
}

class Stack {
  pre: N<'vec3'> = vec3(0);
  alpha: N<'float'> = float(0);

  over(cover: N<'float'>, color: N<'vec3'>): this {
    this.pre = color.mul(cover).add(this.pre.mul(cover.oneMinus()));
    this.alpha = cover.add(this.alpha.mul(cover.oneMinus()));
    return this;
  }

  material(fade: N<'float'> | number = 1): THREE.MeshBasicNodeMaterial {
    return sheet(this.pre.div(max(this.alpha, 1e-4)), this.alpha.mul(fade));
  }
}

export function backdrop(): THREE.MeshBasicNodeMaterial {
  const g = vec2(positionWorld.x, positionWorld.z), cell = floor(g.div(2.4));
  const slab = sdBox(fract(g.div(2.4)).sub(0.5), vec2(0.47), 0.04);
  const tone = hash(cell, 12.9898, 78.233).mul(0.25).add(0.85).mul(fractal(g.mul(0.9), 3).mul(0.25).add(1));
  return solid(mix(rgb(VOID).mul(0.6), rgb(VOID).mul(tone), coverage(slab)));
}

export interface Comet {
  head: Level;
  tail: Level;
  strength: Level;
  tint: Tint;
}

export function flagstones(w: number, h: number, data: THREE.DataTexture, comet: Comet, accent: number, lit: Lit): THREE.MeshBasicNodeMaterial {
  const g = vec2(positionWorld.x.add(w / 2), positionWorld.z.add(h / 2));
  const cell = clamp(floor(g), vec2(0), vec2(w - 1, h - 1)), p = g.sub(cell).sub(0.5);
  const info = texture(data).load(uvec2(cell));
  const glint = info.x, seen = info.y, index = info.z, angle = info.w.mul(Math.PI / 2);
  const ahead = vec2(cos(angle), sin(angle)), q = vec2(dot(p, ahead), dot(p, vec2(ahead.y.negate(), ahead.x)));

  const wobble = vec2(noise(g.mul(4.7)), noise(g.mul(4.7).add(17.3))).mul(0.014);
  const slab = sdBox(p.add(wobble), vec2(0.478), 0.04);
  const tone = hash(cell, 12.9898, 78.233), tilt = hash(cell, 39.3468, 11.135);
  const grain = fractal(g.mul(3.1).add(cell.mul(1.7)), 4), fleck = noise(g.mul(29));
  const crack = smoothstep(0.03, 0, abs(fractal(g.mul(1.9).add(cell.mul(0.61)).add(3.7), 3))).mul(smoothstep(0.72, 0.9, tilt));
  const rim = smoothstep(-0.045, -0.012, slab).mul(smoothstep(-0.003, -0.025, slab)).mul(clamp(p.y.sub(p.x).mul(1.8), 0, 1));
  const stone = rgb(SLATE).mul(tone.sub(0.5).mul(0.26).add(1)).mul(grain.mul(0.2).add(1)).mul(fleck.mul(0.05).add(1))
    .mul(crack.mul(-0.2).add(1)).mul(smoothstep(-0.06, -0.004, slab).mul(-0.22).add(1)).add(rgb(0xa8b0c2).mul(rim.mul(0.08)));
  let albedo: N<'vec3'> = mix(rgb(GROUT), stone, coverage(slab));

  const chevron = min(sdSegment(q, vec2(-0.05, 0.08), vec2(0.045, 0), 0.019), sdSegment(q, vec2(0.045, 0), vec2(-0.05, -0.08), 0.019));
  const ring = abs(length(p).sub(0.2)).sub(0.022);
  const inlay = select(index.lessThan(0.5), ring, chevron), mark = coverage(inlay);
  albedo = mix(albedo, rgb(0x0d0f14), coverage(inlay.sub(0.016)).mul(0.75));
  albedo = mix(albedo, rgb(accent).mul(0.5), mark);

  const trail = rgb(0xd6deef).mul(seen.mul(0.18));
  const shine = rgb(AMBER).mul(sqrt(max(glint, 0)).mul(5)).add(rgb(FROST).mul(sqrt(max(glint.negate(), 0)).mul(5))).add(trail);
  const halo = exp(length(p).mul(-7.5)).mul(coverage(slab));
  const dist = index.sub(comet.head);
  const wave = select(dist.greaterThanEqual(0), exp(dist.negate().div(comet.tail)), exp(dist.mul(dist).mul(-1.6))).mul(comet.strength);
  const glow = shine.mul(mark.mul(1.1).add(halo.mul(0.3))).add(comet.tint.mul(wave).mul(mark.mul(0.9).add(halo.mul(0.4))));
  return solid(lit(albedo).add(glow));
}

export function walls(lit: Lit): THREE.MeshBasicNodeMaterial {
  const run = attribute<'vec2'>('aRun', 'vec2'), vertical = run.x.greaterThan(0.5);
  const along = select(vertical, positionWorld.z, positionWorld.x), across = select(vertical, positionWorld.x, positionWorld.z).sub(run.y);
  const course = select(across.greaterThan(0), float(1), float(0));
  const u = along.div(BRICK).add(course.mul(0.5)), k = floor(u);
  const b = vec2(fract(u).sub(0.5).mul(BRICK), abs(across).sub(WALL / 4));
  const block = sdBox(b, vec2(BRICK / 2 - 0.007, WALL / 4 - 0.007), 0.01);
  const tone = hash(vec2(k, course.add(run.y.mul(3.1))), 12.9898, 78.233);
  const top = rgb(STONE).mul(tone.mul(0.3).add(0.82)).mul(fractal(vec2(along, across).mul(11), 2).mul(0.16).add(1))
    .mul(smoothstep(-0.03, -0.004, block).mul(-0.28).add(1)).add(rgb(0xb4ab9c).mul(smoothstep(-0.03, -0.012, block).mul(smoothstep(-0.002, -0.014, block)).mul(0.1)));
  const stone = mix(rgb(MORTAR), top, coverage(block));
  return solid(lit(stone).add(stone.mul(0.16)));
}

export function annex(w: number, h: number, lit: Lit, open: Level): THREE.MeshBasicNodeMaterial {
  const g = vec2(positionWorld.x.add(w / 2), positionWorld.z.add(h / 2));
  const row = floor(g.y.mul(7)), plankAt = g.x.mul(2.3).add(hash(vec2(row, 1), 7.13, 3.71).mul(3)), board = vec2(fract(plankAt).sub(0.5).div(2.3), fract(g.y.mul(7)).sub(0.5).div(7));
  const plank = sdBox(board, vec2(0.5 / 2.3 - 0.005, 0.5 / 7 - 0.004), 0.004);
  const grain = fractal(vec2(g.x.mul(3), g.y.mul(40)).add(row.mul(3.3)), 3);
  const alcove = mix(rgb(0x120c08), rgb(0x4a3020).mul(hash(vec2(floor(plankAt), row), 12.9898, 78.233).mul(0.3).add(0.8)).mul(grain.mul(0.25).add(1)), coverage(plank));

  const tile = fract(g.mul(4)).sub(0.5);
  let vault: N<'vec3'> = mix(rgb(0x19140e), rgb(0x3f3322).mul(hash(floor(g.mul(4)), 7.13, 3.71).mul(0.3).add(0.85)), coverage(sdBox(tile, vec2(0.46), 0.03)));
  const hinge = g.sub(vec2(0, HINGE)), arc = coverage(abs(length(hinge).sub(SPAN - DOOR / 2)).sub(0.006)).mul(dashes(sin(atan(hinge.y, hinge.x.negate()).mul(40))));
  const quarter = select(hinge.x.lessThan(0).and(hinge.y.greaterThan(0)), float(1), float(0));
  vault = mix(vault, rgb(0x7a6642), arc.mul(quarter).mul(open.oneMinus().mul(0.5).add(0.3)));
  return solid(lit(select(g.y.greaterThan(1), vault, alcove)));
}

export function treasure(lit: Lit, open: Level): THREE.MeshBasicNodeMaterial {
  const p = local(0.8);
  const coins = [[-0.11, 0.07, 0.07], [0.02, 0.1, 0.075], [0.12, 0.04, 0.068], [-0.06, -0.06, 0.074], [0.07, -0.08, 0.07], [-0.15, -0.1, 0.06], [0, 0, 0.08]] as const;
  const stack = new Stack().over(soft(ellipse(p, [0.02, -0.03], [0.28, 0.24]), 0.06).mul(0.5), vec3(0));
  for (const [x, y, r] of coins) {
    const c = p.sub(vec2(x, y)), d = length(c).sub(r);
    const face = rgb(0xd9a93e).mul(dot(c.div(r), vec2(-0.5, 0.6)).mul(0.35).add(0.95));
    stack.over(coverage(d), mix(lit(face), lit(rgb(0x7a5a20)), coverage(abs(d.add(0.012)).sub(0.006))).add(rgb(GOLD).mul(open.mul(0.25))));
  }
  const gp = p.sub(vec2(0.13, 0.15)), gem = abs(gp.x).div(0.06).add(abs(gp.y).div(0.085)).sub(1).mul(0.05);
  stack.over(coverage(gem), mix(rgb(0x8e2c55), rgb(0xff8fbd), smoothstep(-0.02, 0.02, gp.x.add(gp.y.mul(0.4))).mul(0.6)).mul(open.mul(1.6).add(0.45)));
  return stack.material();
}

export function plinth(lit: Lit): THREE.MeshBasicNodeMaterial {
  const p = local(PLINTH), r = length(p);
  const base = r.sub(0.25), lean = dot(p.div(max(r, 1e-4)), vec2(-0.6, 0.8));
  const stone = rgb(0x6e6a64).mul(smoothstep(0.25, 0.19, r).mul(0.18).add(0.82)).mul(lean.mul(smoothstep(0.17, 0.25, r)).mul(0.25).add(1)).mul(fractal(p.mul(18), 2).mul(0.12).add(1));
  const cushion = sdBox(p, vec2(0.15), 0.06);
  const velvet = rgb(VELVET).mul(clamp(cushion.negate().div(0.09), 0, 1).mul(0.45).add(0.6)).mul(dot(p, vec2(-0.7, 0.7)).mul(1.3).add(1));
  const stitch = coverage(abs(cushion.add(0.03)).sub(0.004)).mul(dashes(sin(atan(p.y, p.x).mul(26))));
  const stack = new Stack().over(soft(length(p.sub(vec2(0.025, -0.03))).sub(0.25), 0.05).mul(0.6), vec3(0));
  stack.over(coverage(base), lit(stone));
  stack.over(coverage(abs(base.add(0.035)).sub(0.005)), lit(rgb(0x3b3833)));
  stack.over(soft(sdBox(p.sub(vec2(0.015, -0.02)), vec2(0.15), 0.06), 0.03).mul(0.5), vec3(0));
  stack.over(coverage(cushion), lit(mix(velvet, rgb(0xd9b56b), stitch.mul(0.7))));
  for (const [x, y] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) stack.over(coverage(length(p.sub(vec2(x * 0.165, y * 0.165))).sub(0.022)), lit(rgb(0xd9b56b)));
  return stack.material();
}

export function keySprite(accent: number, fade: Level, shine: Level): THREE.MeshBasicNodeMaterial {
  const p = local(KEY);
  const bow = abs(sdCircle(p.sub(vec2(-0.135, 0)), 0.062)).sub(0.024);
  const shaft = sdBox(p.sub(vec2(0.03, 0)), vec2(0.115, 0.017), 0.008);
  const collar = sdBox(p.sub(vec2(-0.058, 0)), vec2(0.013, 0.032), 0.006);
  const bit = max(sdBox(p.sub(vec2(0.118, -0.042)), vec2(0.03, 0.03), 0.004), sdBox(p.sub(vec2(0.117, -0.058)), vec2(0.007, 0.018), 0).negate());
  const shape = min(min(bow, shaft), min(collar, bit));
  const lean = dot(p.mul(8), vec2(-0.45, 0.6));
  const brass = rgb(accent).mul(lean.mul(0.3).add(1.05)).add(rgb(0xfff2c8).mul(exp(length(p.sub(vec2(-0.16, 0.035))).mul(-40)).mul(0.9)));
  const stack = new Stack().over(soft(shape.sub(0.012), 0.03).mul(0.55), vec3(0));
  stack.over(coverage(shape.sub(0.008)), rgb(0x2b1f0c));
  stack.over(coverage(shape), brass.mul(shine.mul(0.9).add(1)));
  return stack.material(fade);
}

export function traveler(lit: Lit, sway: Level, lamp: Level): THREE.MeshBasicNodeMaterial {
  const p = local(FIGURE), back = p.add(vec2(0, sway.mul(0.016)));
  const shoulders = ellipse(p, [0.005, 0], [0.1, 0.185]), cape = ellipse(back, [-0.095, 0], [0.165, 0.168]);
  const body = min(shoulders, cape);
  const arms = min(sdSegment(p, vec2(0.02, 0.14), vec2(HAND.x - 0.012, HAND.y - 0.004), 0.034), sdSegment(p, vec2(0.02, -0.14), vec2(LANTERN.x - 0.035, LANTERN.y + 0.012), 0.034));
  const hood = min(sdCircle(p.sub(vec2(0.03, 0)), 0.088), ellipse(p, [-0.035, 0], [0.105, 0.062]));
  const face = sdCircle(p.sub(vec2(0.1, 0)), 0.052);
  const grip = min(sdCircle(p.sub(vec2(HAND.x, HAND.y)), 0.028), sdCircle(p.sub(vec2(LANTERN.x - 0.03, LANTERN.y + 0.01)), 0.026));
  const cage = sdCircle(p.sub(vec2(LANTERN.x, LANTERN.y)), 0.046), glass = sdCircle(p.sub(vec2(LANTERN.x, LANTERN.y)), 0.03);
  const lean = p.x.mul(0.8).add(p.y.mul(0.5));
  const folds = sin(atan(back.y, back.x.add(0.02)).mul(11)).mul(smoothstep(-0.02, -0.2, back.x)).mul(0.1);
  const ink = rgb(0x071012);
  const stack = new Stack().over(soft(ellipse(p, [-0.04, -0.035], [0.27, 0.225]), 0.07).mul(0.55), vec3(0));
  stack.over(coverage(min(body, arms).sub(0.014)), ink);
  stack.over(coverage(cape), lit(rgb(ROBE).mul(lean.mul(0.9).add(0.9).add(folds))));
  stack.over(coverage(abs(cape.add(0.018)).sub(0.007)).mul(smoothstep(0.0, -0.06, back.x)), lit(rgb(TRIM)));
  stack.over(coverage(arms), lit(rgb(ROBE).mul(lean.mul(0.8).add(1.05))));
  stack.over(coverage(shoulders), lit(rgb(ROBE).mul(lean.mul(1.1).add(1.1))));
  stack.over(coverage(grip), lit(rgb(0xc9a07c)));
  stack.over(coverage(hood.sub(0.012)), ink);
  stack.over(coverage(hood), lit(rgb(HOOD).mul(lean.mul(1.3).add(1.05))).add(rgb(0x6a9ea2).mul(exp(length(p.sub(vec2(0.0, 0.035))).mul(-26)).mul(0.18))));
  stack.over(coverage(max(face, hood)), rgb(0x0a1416));
  stack.over(coverage(cage), rgb(0x3a2c1a));
  stack.over(coverage(glass), rgb(FLAME).mul(lamp));
  return stack.material();
}

export function doorLeaf(accent: number, lit: Lit): THREE.MeshBasicNodeMaterial {
  const p = local(DOOR + 0.1, SPAN), across = p.x, along = p.y;
  const leaf = sdBox(p, vec2(DOOR / 2, SPAN / 2), 0.012);
  const plank = abs(fract(across.div(DOOR).mul(3).add(0.5)).sub(0.5));
  const wood = rgb(WOOD).mul(fractal(vec2(across.mul(40), along.mul(3)), 3).mul(0.3).add(1)).mul(smoothstep(0, 0.08, plank).mul(0.35).add(0.65));
  const straps = abs(abs(along).sub(SPAN * 0.3)).sub(0.035);
  const rivets = length(vec2(abs(across).sub(DOOR * 0.28), abs(abs(along).sub(SPAN * 0.3)))).sub(0.012);
  const lock = sdBox(p.sub(vec2(DOOR / 2 + 0.012, 0)), vec2(0.028, 0.055), 0.012);
  const keyhole = min(sdCircle(p.sub(vec2(DOOR / 2 + 0.026, 0.012)), 0.009), sdBox(p.sub(vec2(DOOR / 2 + 0.026, -0.006)), vec2(0.004, 0.014), 0.002));
  const body = mix(wood, rgb(IRON), coverage(straps));
  const stack = new Stack().over(coverage(leaf.sub(0.01)), rgb(0x120c07));
  stack.over(coverage(leaf), lit(body).add(body.mul(0.18)));
  stack.over(coverage(rivets).mul(coverage(straps)), lit(rgb(0x7b7e88)).add(rgb(0x7b7e88).mul(0.15)));
  stack.over(coverage(lock.sub(0.006)), rgb(0x2b1f0c));
  stack.over(coverage(lock), lit(rgb(accent)).add(rgb(accent).mul(0.3)).mul(dot(p.sub(vec2(DOOR / 2, 0)).mul(20), vec2(-0.5, 0.6)).mul(0.3).add(1)));
  stack.over(coverage(keyhole), rgb(0x0b0806));
  return stack.material();
}

export function candle(accent: number, lit: Lit, glow: Level): THREE.MeshBasicNodeMaterial {
  const p = local(0.24), r = length(p);
  const brass = rgb(accent).mul(dot(p.mul(12), vec2(-0.5, 0.6)).mul(0.3).add(0.85));
  const stack = new Stack().over(soft(length(p.sub(vec2(0.012, -0.014))).sub(0.07), 0.02).mul(0.55), vec3(0));
  stack.over(coverage(r.sub(0.07)), lit(brass).add(brass.mul(0.15)));
  stack.over(coverage(abs(r.sub(0.052)).sub(0.005)), lit(rgb(0x5a4420)));
  stack.over(coverage(r.sub(0.036)), lit(rgb(0xe8dcc0)).add(rgb(FLAME).mul(glow.mul(0.12))));
  stack.over(coverage(r.sub(0.014)), rgb(FLAME).mul(glow));
  return stack.material();
}

export interface Ledger {
  material: THREE.MeshBasicNodeMaterial;
  values: number[];
}

export function ledger(spacing: number, radius: number, accent: number, lit: Lit): Ledger {
  const values = new Array<number>(LEDGER).fill(0), node = uniformArray(values, 'float' as const);
  const width = spacing * LEDGER, p = local(width, radius * 4);
  const x = p.x.add(width / 2), k = clamp(floor(x.div(spacing)), 0, LEDGER - 1), c = vec2(x.sub(k.add(0.5).mul(spacing)), p.y);
  const value = node.element(int(k)), opened = max(value, 0), locked = max(value.negate(), 0);
  const rail = sdBox(p, vec2(width / 2 - spacing * 0.2, radius * 0.28), radius * 0.28);
  const lamp = rgb(GOLD).mul(opened.mul(1.15)).add(rgb(MISS).mul(locked.mul(0.55))).add(rgb(0x14161d).mul(opened.add(locked).oneMinus()));
  const shine = exp(length(c.sub(vec2(-radius * 0.35, radius * 0.4))).mul(-9 / radius)).mul(opened.mul(0.5).add(0.2));
  const stack = new Stack().over(coverage(rail), lit(rgb(accent).mul(0.45)).add(rgb(accent).mul(0.08)));
  stack.over(coverage(length(c).sub(radius * 1.25)), lit(rgb(0x3a3322)).add(rgb(0x3a3322).mul(0.2)));
  stack.over(coverage(length(c).sub(radius)), lamp.add(vec3(shine)));
  return { material: stack.material(), values };
}

export const vectorOf = (hex: number, gain = 1): THREE.Vector3 => {
  const c = new THREE.Color(hex);
  return new THREE.Vector3(c.r, c.g, c.b).multiplyScalar(gain);
};

export const tint = (hex: number, gain = 1): Tint => uniform(vectorOf(hex, gain));
