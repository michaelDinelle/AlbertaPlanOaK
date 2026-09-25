import * as THREE from 'three/webgpu';
import { abs, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, positionWorld, select, sin, smoothstep, texture, uniform, uniformArray, uv, vec2, vec3 } from 'three/tsl';
import type { Light2D } from '@kit/light2d';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { fbm } from '@kit/textures';
import { BLOCK, MOB_SLOTS } from '@sim/envs/craftax';

type N<T extends string> = THREE.Node<T>;
export type Shade = (albedo: N<'vec3'>) => N<'vec3'>;

export const TERRAIN = { VOID: 0, GRASS: 1, SAND: 2, WATER: 3, EARTH: 4, LAVA: 5, ROCK: 6, PLANKS: 7 } as const;
export const OBJECT = { NONE: 0, TREE: 1, COAL: 2, IRON: 3, DIAMOND: 4, TABLE: 5, FURNACE: 6, SAPLING: 7, RIPE: 8 } as const;

const PAINT: Record<number, readonly [number, number]> = {
  [BLOCK.OUT]: [TERRAIN.VOID, OBJECT.NONE],
  [BLOCK.GRASS]: [TERRAIN.GRASS, OBJECT.NONE],
  [BLOCK.WATER]: [TERRAIN.WATER, OBJECT.NONE],
  [BLOCK.STONE]: [TERRAIN.ROCK, OBJECT.NONE],
  [BLOCK.TREE]: [TERRAIN.GRASS, OBJECT.TREE],
  [BLOCK.WOOD]: [TERRAIN.PLANKS, OBJECT.NONE],
  [BLOCK.PATH]: [TERRAIN.EARTH, OBJECT.NONE],
  [BLOCK.COAL]: [TERRAIN.ROCK, OBJECT.COAL],
  [BLOCK.IRON]: [TERRAIN.ROCK, OBJECT.IRON],
  [BLOCK.DIAMOND]: [TERRAIN.ROCK, OBJECT.DIAMOND],
  [BLOCK.TABLE]: [TERRAIN.EARTH, OBJECT.TABLE],
  [BLOCK.FURNACE]: [TERRAIN.EARTH, OBJECT.FURNACE],
  [BLOCK.SAND]: [TERRAIN.SAND, OBJECT.NONE],
  [BLOCK.LAVA]: [TERRAIN.LAVA, OBJECT.NONE],
  [BLOCK.PLANT]: [TERRAIN.GRASS, OBJECT.SAPLING],
  [BLOCK.RIPE]: [TERRAIN.GRASS, OBJECT.RIPE],
};

export const MINIMAP_COLOR: Record<number, number> = {
  [BLOCK.OUT]: 0x0a0d12, [BLOCK.GRASS]: 0x5a8c38, [BLOCK.WATER]: 0x2f6fa6, [BLOCK.STONE]: 0x85878d, [BLOCK.TREE]: 0x2d5f2c, [BLOCK.WOOD]: 0xa8773f,
  [BLOCK.PATH]: 0x5e4d3e, [BLOCK.COAL]: 0x2a2a30, [BLOCK.IRON]: 0xd49468, [BLOCK.DIAMOND]: 0xdffbff, [BLOCK.TABLE]: 0xc58a48, [BLOCK.FURNACE]: 0xff9a3c,
  [BLOCK.SAND]: 0xd8c08a, [BLOCK.LAVA]: 0xff6a1a, [BLOCK.PLANT]: 0x9ad66a, [BLOCK.RIPE]: 0xe0503a,
};

export class WorldMap {
  readonly texture: THREE.DataTexture;
  private readonly data: Uint8Array;
  private readonly seen: Uint32Array;
  private fresh = false;

  constructor(readonly side: number) {
    this.data = new Uint8Array(side * side * 4);
    this.seen = new Uint32Array(side * side / 8);
    this.texture = new THREE.DataTexture(this.data, side, side, THREE.RGBAFormat);
    this.texture.magFilter = this.texture.minFilter = THREE.NearestFilter;
    this.texture.generateMipmaps = false;
  }

  paint(view: Uint32Array): void {
    if (this.fresh && view.every((word, i) => word === this.seen[i])) return;
    this.fresh = true;
    this.seen.set(view);
    for (let cell = 0; cell < this.side * this.side; cell++) {
      const block = ((view[cell >> 3]! >>> (4 * (cell & 7))) & 15) + 1, [terrain, object] = PAINT[block]!, at = cell * 4;
      this.data[at] = terrain;
      this.data[at + 1] = object;
      this.data[at + 2] = block;
      this.data[at + 3] = 255;
    }
    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.texture.dispose();
  }
}

export interface WorldUniforms {
  clock: THREE.UniformNode<'float', number>;
  view: THREE.UniformNode<'vec2', THREE.Vector2>;
  half: THREE.UniformNode<'vec2', THREE.Vector2>;
  veil: THREE.UniformNode<'float', number>;
}

export const worldUniforms = (half: THREE.Vector2): WorldUniforms => ({ clock: uniform(0), view: uniform(new THREE.Vector2()), half: uniform(half.clone()), veil: uniform(0) });

export interface Noise { mottle: THREE.DataTexture; grain: THREE.DataTexture }

export const makeNoise = (seed: number): Noise => ({ mottle: fbm(seed ^ 0x3c71, { size: 256, frequency: 4, octaves: 5 }), grain: fbm(seed ^ 0x9e21, { size: 256, frequency: 16, octaves: 3 }) });

const NEIGHBOURS = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]] as const;

export const is = (value: N<'float'>, k: number): N<'bool'> => abs(value.sub(k)).lessThan(0.5);
export const hash = (c: N<'vec2'>, salt: number): N<'float'> => fract(sin(dot(c.add(salt), vec2(127.1, 311.7))).mul(43758.5453));
export const rotate = (p: N<'vec2'>, angle: N<'float'> | number): N<'vec2'> => {
  const c = typeof angle === 'number' ? float(Math.cos(angle)) : cos(angle), s = typeof angle === 'number' ? float(Math.sin(angle)) : sin(angle);
  return vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)));
};
export const smin = (a: N<'float'>, b: N<'float'>, k: number): N<'float'> => {
  const h = clamp(b.sub(a).mul(0.5 / k).add(0.5), 0, 1);
  return mix(b, a, h).sub(h.mul(h.oneMinus()).mul(k));
};
export const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]): N<'float'> =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));
const boxGap = (p: N<'vec2'>, centre: N<'vec2'>, half: number): N<'float'> => length(max(abs(p.sub(centre)).sub(half), 0));

function cobbles(p: N<'vec2'>, scale: number): { joint: N<'float'>; tone: N<'float'>; lit: N<'float'>; core: N<'float'> } {
  const q = p.mul(scale), qi = floor(q), qf = q.sub(qi);
  let near: N<'float'> = float(9), second: N<'float'> = float(9), tone: N<'float'> = float(0), lit: N<'float'> = float(0);
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const o = vec2(dx, dy), id = qi.add(o), centre = o.add(vec2(hash(id, 1.7), hash(id, 4.2)).mul(0.8).add(0.1)), r = length(centre.sub(qf));
    const closer = r.lessThan(near);
    second = select(closer, near, min(second, r));
    tone = select(closer, hash(id, 7.9), tone);
    lit = select(closer, dot(qf.sub(centre), vec2(-0.6, -0.8)).mul(1.6).add(0.5).clamp(0, 1), lit);
    near = min(near, r);
  }
  return { joint: second.sub(near).div(scale), tone, lit, core: float(1).sub(near.mul(1.4)) };
}

export function outsideView(g: N<'vec2'>, u: WorldUniforms): N<'float'> {
  return smoothstep(-0.02, 0.05, sdBox(g.sub(u.view), u.half));
}

export function dimmed(color: N<'vec3'>, g: N<'vec2'>, u: WorldUniforms): N<'vec3'> {
  const out = outsideView(g, u), grey = vec3(dot(color, vec3(0.3, 0.5, 0.2)));
  return mix(color, mix(color, grey, 0.3).mul(0.56), out).mul(u.veil.oneMinus());
}

interface Cell { terrain: N<'float'>; object: N<'float'> }

function cells(side: number, map: THREE.Texture, cell: N<'vec2'>): { here: Cell; around: Cell[] } {
  const read = (dx: number, dy: number): Cell => {
    const c = cell.add(vec2(dx, dy));
    const inside = c.x.greaterThanEqual(0).and(c.y.greaterThanEqual(0)).and(c.x.lessThan(side)).and(c.y.lessThan(side));
    const texel = texture(map, c.add(0.5).div(side));
    return { terrain: select(inside, floor(texel.r.mul(255).add(0.5)), float(TERRAIN.VOID)), object: select(inside, floor(texel.g.mul(255).add(0.5)), float(OBJECT.NONE)) };
  };
  return { here: read(0, 0), around: NEIGHBOURS.map(([dx, dy]) => read(dx, dy)) };
}

export function groundMaterial(side: number, map: THREE.Texture, noise: Noise, u: WorldUniforms, shade: Shade): THREE.MeshBasicNodeMaterial {
  const g = vec2(positionWorld.x.add(side / 2), positionWorld.z.add(side / 2));
  const cell = floor(g), f = g.sub(cell), q = vec2(f.x.sub(0.5), f.y.sub(0.5).negate());
  const { here, around } = cells(side, map, cell);
  const gap = NEIGHBOURS.map(([dx, dy]) => boxGap(f, vec2(dx + 0.5, dy + 0.5), 0.5));
  const nearTo = (kind: number): N<'float'> => around.reduce<N<'float'>>((best, n, i) => min(best, select(is(n.terrain, kind), gap[i]!, float(9))), float(9));
  const unlike = around.reduce<N<'float'>>((best, n, i) => min(best, select(abs(n.terrain.sub(here.terrain)).greaterThan(0.5), gap[i]!, float(9))), float(9));
  const t = here.terrain, o = here.object;
  const tex = (source: THREE.Texture, at: N<'vec2'>) => texture(source, at).r;
  const mott = tex(noise.mottle, g.mul(0.13)), broad = tex(noise.mottle, g.mul(0.045).add(0.31));
  const fine = tex(noise.grain, g.mul(0.37)), speck = tex(noise.grain, g.mul(0.71).add(0.7));
  const clock = u.clock;

  const strokes = tex(noise.grain, g.mul(vec2(0.93, 0.34)).add(0.37)), tufts = tex(noise.grain, g.mul(0.47).add(0.11));
  let grass: N<'vec3'> = mix(rgb(0x3f6c2b), rgb(0x5a8a37), smoothstep(0.25, 0.75, mott));
  grass = mix(grass, rgb(0x6f9a41), smoothstep(0.55, 0.85, broad).mul(0.4));
  grass = grass.mul(hash(cell, 9.7).sub(0.5).mul(0.07).add(1));
  grass = grass.mul(smoothstep(0.56, 0.74, strokes).mul(0.2).add(1)).mul(smoothstep(0.44, 0.26, strokes).mul(-0.18).add(1));
  grass = mix(grass, rgb(0x87ad4f), smoothstep(0.7, 0.8, tufts).mul(0.4));
  grass = grass.mul(fine.mul(0.1).add(0.95));
  const drift = tex(noise.mottle, g.mul(0.21).add(0.77));
  const ripple = sin(g.x.mul(0.8).add(g.y.mul(0.55)).mul(13).add(drift.mul(14))).mul(0.5).add(0.5);
  let sand: N<'vec3'> = mix(rgb(0xc9ad76), rgb(0xe2cd97), smoothstep(0.28, 0.72, mott)).mul(fine.mul(0.12).add(0.94));
  sand = sand.mul(smoothstep(0.55, 1, ripple).mul(0.07).add(1)).mul(smoothstep(0.35, 0, ripple).mul(-0.07).add(1));
  sand = mix(sand, rgb(0x9f8457), smoothstep(0.78, 0.84, speck).mul(0.5));
  sand = mix(sand, rgb(0xf0e0b0), smoothstep(0.2, 0.14, speck).mul(0.35));
  sand = mix(sand, sand.mul(0.8), smoothstep(0.28, 0, nearTo(TERRAIN.WATER)).mul(0.6));

  const swell = tex(noise.mottle, g.mul(0.11).add(vec2(clock.mul(0.012), clock.mul(0.007)))).add(tex(noise.mottle, g.mul(0.19).add(vec2(clock.mul(-0.008), clock.mul(0.012))).add(0.5))).mul(0.5);
  let water: N<'vec3'> = mix(rgb(0x1d5486), rgb(0x2f72aa), smoothstep(0.32, 0.7, swell));
  water = mix(water, rgb(0x4f9dc8), smoothstep(0.5, 0, unlike).mul(0.55));
  water = mix(water, rgb(0x8cc8e6), smoothstep(0.6, 0.66, swell).mul(0.22));
  const cycle = fract(clock.mul(0.22).add(tex(noise.mottle, g.mul(0.07)).mul(1.7)));
  const wash = exp(unlike.sub(cycle.mul(0.18).add(0.02)).div(0.018).pow(2).negate()).mul(cycle.oneMinus());
  water = mix(water, rgb(0xe8f5fb), clamp(exp(unlike.div(0.03).pow(2).negate()).mul(0.9).add(wash.mul(0.45)), 0, 1));

  let earth: N<'vec3'> = mix(rgb(0x6a5846), rgb(0x806b55), smoothstep(0.3, 0.7, mott)).mul(fine.mul(0.18).add(0.91));
  const pebbles = cobbles(g, 3.1);
  earth = mix(earth, mix(rgb(0x94826c), rgb(0xa8967e), pebbles.tone), smoothstep(0.62, 0.7, pebbles.core).mul(0.55));
  earth = mix(earth, earth.mul(0.82), smoothstep(0.35, 0.65, speck).mul(0.3));
  earth = earth.mul(mix(float(0.62), float(1), smoothstep(0, 0.45, nearTo(TERRAIN.ROCK))));

  const heat = tex(noise.mottle, g.mul(0.23).add(vec2(clock.mul(0.021), clock.mul(-0.013))));
  const plates = cobbles(g.add(vec2(clock.mul(0.03), clock.mul(-0.02))), 1.7);
  const seams = smoothstep(0.1, 0.02, plates.joint).max(smoothstep(0.14, 0, unlike).mul(0.2));
  const molten = mix(rgb(0xc2300c), rgb(0xffa21f), smoothstep(0.25, 0.8, heat.mul(0.6).add(seams.mul(0.5))));
  const crust = mix(rgb(0x2b120a), rgb(0x5a2210), plates.tone.mul(0.6).add(heat.mul(0.4)));
  const lava = mix(crust, molten, seams);
  const glow = molten.mul(seams.mul(0.75).add(0.12)).mul(heat.mul(0.4).add(0.75));

  const stones = cobbles(g, 2.3);
  let rock: N<'vec3'> = mix(rgb(0x5f6168), rgb(0x7f8188), stones.tone).mul(fine.mul(0.14).add(0.93));
  rock = rock.mul(stones.lit.mul(0.22).add(0.9));
  rock = mix(rock, rgb(0x3c3e44), smoothstep(0.07, 0.025, stones.joint).mul(0.85));
  rock = mix(rock, rgb(0x9fa1a7), smoothstep(0.8, 0.86, speck).mul(0.25));
  const open = (i: number) => abs(around[i]!.terrain.sub(TERRAIN.ROCK)).greaterThan(0.5);
  const lift = select(open(0), exp(f.y.div(0.07).negate()), float(0)).add(select(open(2), exp(f.x.div(0.07).negate()), float(0)));
  const drop = select(open(1), exp(f.y.oneMinus().div(0.1).negate()), float(0)).add(select(open(3), exp(f.x.oneMinus().div(0.1).negate()), float(0)));
  const seam = coverage(min(min(f.x, f.x.oneMinus()), min(f.y, f.y.oneMinus())).sub(0.012));
  rock = rock.mul(lift.mul(0.3).add(1).sub(drop.mul(0.42))).mul(seam.mul(-0.16).add(1));

  const board = floor(f.y.mul(4));
  let planks: N<'vec3'> = mix(rgb(0x94663a), rgb(0xb37e47), hash(cell.add(vec2(0, board)), 3.1)).mul(tex(noise.grain, vec2(g.x.mul(0.8), g.y.mul(6))).mul(0.25).add(0.85));
  planks = planks.mul(coverage(abs(fract(f.y.mul(4)).sub(0.5)).negate().add(0.47)).mul(-0.35).add(1));

  const voidTone = rgb(0x0b0f14).mul(mott.mul(0.4).add(0.8));
  const pick = (values: N<'vec3'>[]): N<'vec3'> => values.slice(1).reduce<N<'vec3'>>((acc, value, i) => select(is(t, i + 1), value, acc), values[0]!);
  let albedo = pick([voidTone, grass, sand, water, earth, lava, rock, planks]);
  const checker = floor(cell.x.add(cell.y)).mod(2);
  albedo = albedo.mul(select(is(t, TERRAIN.VOID).or(is(t, TERRAIN.WATER)).or(is(t, TERRAIN.LAVA)), float(1), checker.sub(0.5).mul(0.05).add(1)));

  const fringe = select(is(t, TERRAIN.SAND).or(is(t, TERRAIN.EARTH)), coverage(nearTo(TERRAIN.GRASS).sub(tex(noise.grain, g.mul(0.9).add(0.3)).mul(0.12).add(0.015))), float(0));
  albedo = mix(albedo, grass.mul(0.9), fringe);
  const edge = nearTo(TERRAIN.VOID);
  albedo = select(is(t, TERRAIN.VOID), albedo.mul(smoothstep(0, 0.5, min(unlike, 9)).mul(0.5).add(0.5)), albedo.mul(smoothstep(0, 0.12, edge).mul(0.3).add(0.7)));

  const casts = (i: number): N<'float'> => {
    const [dx, dy] = NEIGHBOURS[i]!, n = around[i]!;
    const block = boxGap(f.sub(vec2(0.09, 0.09)), vec2(dx + 0.5, dy + 0.5), 0.5);
    const crown = length(f.sub(vec2(dx + 0.59, dy + 0.59))).sub(0.4);
    const bench = boxGap(f.sub(vec2(0.07, 0.07)), vec2(dx + 0.5, dy + 0.5), 0.38);
    return max(max(select(is(n.terrain, TERRAIN.ROCK), smoothstep(0.15, 0, block), float(0)), select(is(n.object, OBJECT.TREE), smoothstep(0.12, -0.06, crown), float(0))),
      select(is(n.object, OBJECT.TABLE).or(is(n.object, OBJECT.FURNACE)), smoothstep(0.1, 0, bench), float(0)));
  };
  const ownCrown = select(is(o, OBJECT.TREE), smoothstep(0.12, -0.06, length(f.sub(vec2(0.59, 0.59))).sub(0.4)), float(0));
  const ownBench = select(is(o, OBJECT.TABLE).or(is(o, OBJECT.FURNACE)), smoothstep(0.1, 0, boxGap(f.sub(vec2(0.07, 0.07)), vec2(0.5, 0.5), 0.38)), float(0));
  const shadow = max(max(casts(0), casts(2)), max(casts(4), max(ownCrown, ownBench)));
  albedo = albedo.mul(select(is(t, TERRAIN.ROCK).or(is(t, TERRAIN.VOID)), float(1), shadow.mul(-0.38).add(1)));

  const piece = objectArt(o, q, cell, clock, noise);
  albedo = mix(albedo, piece.color, piece.cover);
  const emission = select(is(t, TERRAIN.LAVA), glow, vec3(0)).add(piece.glow);
  const shaded = shade(albedo).add(emission);
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = dimmed(shaded, g, u);
  return material;
}

interface Piece { color: N<'vec3'>; cover: N<'float'>; glow: N<'vec3'> }

const INK = 0x14110e;

const inked = (d: N<'float'>, fill: N<'vec3'>, width = 0.02): { color: N<'vec3'>; cover: N<'float'> } => ({ color: mix(rgb(INK), fill, coverage(d.add(width))), cover: coverage(d) });

function nuggets(q: N<'vec2'>, cell: N<'vec2'>, count: number, spread: number, half: readonly [number, number], radius: number, gem: boolean): { d: N<'float'>; lit: N<'float'> } {
  let d: N<'float'> = float(9), lit: N<'float'> = float(0);
  const spin = hash(cell, 8.1).mul(6.283);
  for (let k = 0; k < count; k++) {
    const angle = spin.add(k * 6.283 / count), reach = hash(cell, 11.3 + k).mul(0.35).add(0.65).mul(spread);
    const centre = vec2(cos(angle), sin(angle)).mul(reach);
    const local = rotate(q.sub(centre), gem ? Math.PI / 4 : hash(cell, 3.7 + k).mul(3.14));
    const shape = sdBox(local, vec2(...half), radius);
    const shine = smoothstep(0.02, -0.06, local.x.add(local.y).mul(0.5).add(half[0] * 0.6));
    lit = select(shape.lessThan(d), shine, lit);
    d = min(d, shape);
  }
  return { d, lit };
}

function objectArt(o: N<'float'>, q: N<'vec2'>, cell: N<'vec2'>, clock: N<'float'>, noise: Noise): Piece {
  const leafNoise = texture(noise.grain, q.add(cell).mul(0.83)).r;
  const tilt = hash(cell, 1.3).mul(6.283), size = hash(cell, 5.9).mul(0.08).add(0.95);
  const tq = rotate(q, tilt).div(size);
  let crown: N<'float'> = sdCircle(tq, 0.28);
  for (let k = 0; k < 5; k++) crown = smin(crown, sdCircle(tq.sub(vec2(Math.cos(k * 1.2566), Math.sin(k * 1.2566)).mul(0.2)), 0.18 + 0.025 * (k % 2)), 0.07);
  crown = crown.mul(size);
  const sunward = smoothstep(0.34, -0.3, q.x.sub(q.y));
  let foliage: N<'vec3'> = mix(rgb(0x1f4c25), rgb(0x3b7d38), sunward).mul(leafNoise.mul(0.34).add(0.83));
  foliage = mix(rgb(0x173a1d), foliage, smoothstep(0, 0.09, crown.negate()));
  foliage = mix(foliage, rgb(0x62a551), smoothstep(0.64, 0.7, leafNoise).mul(sunward).mul(0.8));
  const tree = inked(crown, foliage, 0.022);

  const coalBits = nuggets(q, cell, 4, 0.21, [0.11, 0.085], 0.045, false);
  const coal = inked(coalBits.d, mix(rgb(0x1b1a1f), rgb(0x55545c), coalBits.lit), 0.018);
  const ironBits = nuggets(q, cell, 4, 0.21, [0.1, 0.075], 0.04, false);
  const iron = inked(ironBits.d, mix(rgb(0xb8703f), rgb(0xf2c08e), ironBits.lit), 0.018);
  const gemBits = nuggets(q, cell, 3, 0.19, [0.1, 0.1], 0.015, true);
  const gemTone = mix(rgb(0x7fdcf2), rgb(0xf2fdff), gemBits.lit);
  const diamond = inked(gemBits.d, gemTone, 0.016);
  const twinkle = sin(clock.mul(2.3).add(hash(cell, 4.4).mul(6.283))).mul(0.5).add(0.5).pow(10);
  const star = min(sdSegment(q.sub(vec2(0.06, 0.1)), vec2(-0.1, 0), vec2(0.1, 0), 0.006), sdSegment(q.sub(vec2(0.06, 0.1)), vec2(0, -0.1), vec2(0, 0.1), 0.006));
  const sparkle = coverage(star).mul(twinkle);

  const top = sdBox(q, vec2(0.44, 0.39), 0.05);
  const grainAlong = texture(noise.grain, vec2(q.x.add(cell.x).mul(0.35), q.y.add(cell.y).mul(3.2))).r;
  let bench: N<'vec3'> = mix(rgb(0x94612f), rgb(0xbb8447), grainAlong).mul(smoothstep(0.3, -0.35, q.x.sub(q.y)).mul(0.2).add(0.9));
  bench = mix(bench, rgb(0x5e3d1d), coverage(abs(abs(q.y).sub(0.115)).sub(0.008)).mul(0.8));
  bench = mix(bench, rgb(0x6e4722), coverage(abs(top).sub(0.028)));
  const handle = sdSegment(q, vec2(-0.2, -0.16), vec2(0.1, 0.1), 0.024);
  const head = sdBox(rotate(q.sub(vec2(0.115, 0.125)), 0.86), vec2(0.075, 0.034), 0.012);
  bench = mix(bench, rgb(INK), coverage(min(handle, head).sub(0.016)));
  bench = mix(bench, rgb(0x5a371b), coverage(handle));
  bench = mix(bench, mix(rgb(0x8d98a3), rgb(0xd6dde3), smoothstep(-0.03, 0.03, q.y.sub(q.x).add(0.01))), coverage(head));
  const table = inked(top, bench, 0.024);

  const body = sdBox(q, vec2(0.44), 0.07);
  const course = floor(q.y.add(0.5).mul(5)), brick = fract(vec2(q.x.add(0.5).mul(2.5).add(course.mod(2).mul(0.5)), q.y.add(0.5).mul(5)));
  let stone: N<'vec3'> = mix(rgb(0x676970), rgb(0x8e9097), hash(floor(vec2(q.x.add(0.5).mul(2.5).add(course.mod(2).mul(0.5)), course)).add(cell), 6.6)).mul(leafNoise.mul(0.2).add(0.9));
  stone = mix(stone, rgb(0x3b3c42), coverage(min(abs(brick.x.sub(0.5)).negate().add(0.47), abs(brick.y.sub(0.5)).negate().add(0.45)).mul(0.2)).mul(0.7));
  const mouth = sdBox(q.sub(vec2(0, -0.12)), vec2(0.16, 0.11), 0.09);
  const flick = sin(clock.mul(13).add(hash(cell, 2.7).mul(40))).mul(0.12).add(sin(clock.mul(7.3)).mul(0.08)).add(1);
  const flame = smoothstep(0.1, -0.12, q.y.add(0.12).add(sin(q.x.mul(40).add(clock.mul(9))).mul(0.025)));
  const fire = mix(rgb(0xff5a14), rgb(0xffd966), flame).mul(flick);
  stone = mix(stone, rgb(0x120b08), coverage(mouth));
  stone = mix(stone, fire, coverage(mouth.add(0.03)).mul(flame.mul(0.7).add(0.3)));
  const chimney = sdCircle(q.sub(vec2(0.2, 0.22)), 0.065);
  stone = mix(stone, rgb(0x1d1512), coverage(chimney));
  const furnace = inked(body, stone, 0.024);
  const furnaceGlow = fire.mul(coverage(mouth.add(0.03))).mul(flame.mul(1.4).add(0.6)).add(rgb(0xff7a2a).mul(coverage(chimney.add(0.02))).mul(0.35).mul(flick));

  const soil = ellipse(q, [0, -0.1], [0.19, 0.105]);
  const stem = sdSegment(q, vec2(0, -0.1), vec2(0, 0.07), 0.018);
  const leafL = ellipse(rotate(q.sub(vec2(-0.075, 0.1)), -0.6), [0, 0], [0.09, 0.045]);
  const leafR = ellipse(rotate(q.sub(vec2(0.08, 0.125)), 0.55), [0, 0], [0.095, 0.047]);
  let sprout: N<'vec3'> = mix(rgb(0x4b3220), rgb(0x6d4c2f), smoothstep(-0.1, 0.05, q.y.add(0.1)));
  sprout = mix(sprout, rgb(INK), coverage(min(stem, min(leafL, leafR)).sub(0.016)));
  sprout = mix(sprout, rgb(0x3f7d2b), coverage(stem));
  sprout = mix(sprout, mix(rgb(0x5fae45), rgb(0x9ee07a), smoothstep(-0.05, 0.12, q.y)), coverage(min(leafL, leafR)));
  const sapling = inked(min(soil, min(stem, min(leafL, leafR))), sprout, 0.02);

  const bush = smin(smin(sdCircle(q.sub(vec2(-0.11, 0.03)), 0.2), sdCircle(q.sub(vec2(0.11, 0.05)), 0.19), 0.08), sdCircle(q.sub(vec2(0, -0.11)), 0.2), 0.08);
  let ripe: N<'vec3'> = mix(rgb(0x2d6a2d), rgb(0x4f9442), sunward).mul(leafNoise.mul(0.3).add(0.85));
  const fruit = [[-0.13, 0.09], [0.11, 0.13], [0.07, -0.1], [-0.08, -0.13]].reduce<N<'float'>>((d, [x, y]) => min(d, sdCircle(q.sub(vec2(x!, y!)), 0.055)), float(9));
  ripe = mix(ripe, rgb(INK), coverage(fruit.sub(0.014)));
  ripe = mix(ripe, rgb(0xd83a30), coverage(fruit));
  ripe = mix(ripe, rgb(0xffd0c2), coverage(fruit.add(0.032)).mul(0.9));
  const plant = inked(bush, ripe, 0.022);

  const pieces: [number, { color: N<'vec3'>; cover: N<'float'> }][] = [[OBJECT.TREE, tree], [OBJECT.COAL, coal], [OBJECT.IRON, iron], [OBJECT.DIAMOND, diamond], [OBJECT.TABLE, table], [OBJECT.FURNACE, furnace], [OBJECT.SAPLING, sapling], [OBJECT.RIPE, plant]];
  let color: N<'vec3'> = vec3(0), cover: N<'float'> = float(0);
  for (const [kind, piece] of pieces) {
    color = select(is(o, kind), piece.color, color);
    cover = select(is(o, kind), piece.cover, cover);
  }
  const glow = select(is(o, OBJECT.FURNACE), furnaceGlow, select(is(o, OBJECT.DIAMOND), rgb(0xe9fdff).mul(sparkle.mul(2.4)).add(gemTone.mul(coverage(gemBits.d)).mul(0.18)), vec3(0)));
  return { color, cover, glow };
}

export function lightProxy(side: number, map: THREE.Texture, u: WorldUniforms, noise: Noise, light: Light2D): THREE.Mesh {
  const g = vec2(positionWorld.x.add(side / 2), positionWorld.z.add(side / 2));
  const cell = floor(g), q = vec2(g.x.sub(cell.x).sub(0.5), g.y.sub(cell.y).sub(0.5).negate());
  const { here } = cells(side, map, cell);
  const heat = texture(noise.mottle, g.mul(0.23).add(vec2(u.clock.mul(0.021), u.clock.mul(-0.013)))).r;
  const flicker = sin(u.clock.mul(13).add(hash(cell, 2.7).mul(40))).mul(0.12).add(sin(u.clock.mul(7.3)).mul(0.08)).add(1);
  const lava = is(here.terrain, TERRAIN.LAVA), furnace = is(here.object, OBJECT.FURNACE), diamond = is(here.object, OBJECT.DIAMOND);
  const mouth = sdBox(q.sub(vec2(0, -0.12)), vec2(0.15, 0.1), 0.08).lessThan(0);
  const gem = length(q).lessThan(0.2);
  const emit = lava.or(furnace.and(mouth)).or(diamond.and(gem));
  const rockCore = is(here.terrain, TERRAIN.ROCK).and(max(abs(q.x), abs(q.y)).lessThan(0.3));
  const treeCore = is(here.object, OBJECT.TREE).and(length(q).lessThan(0.27));
  const benchCore = is(here.object, OBJECT.TABLE).or(furnace).and(max(abs(q.x), abs(q.y)).lessThan(0.32));
  const tint = select(lava, vec3(1, 0.4, 0.1).mul(heat.mul(0.5).add(1.4)), select(furnace, vec3(1, 0.55, 0.2).mul(flicker.mul(2)), vec3(0.55, 0.85, 1).mul(0.6)));
  const mesh = light.emitter(new THREE.PlaneGeometry(side + 40, side + 40).rotateX(-Math.PI / 2), select(emit, tint, vec3(0)));
  (mesh.material as THREE.NodeMaterial).maskNode = emit.or(rockCore).or(treeCore).or(benchCore);
  mesh.frustumCulled = false;
  return mesh;
}

export interface Minimap { mesh: THREE.Mesh; player: THREE.UniformNode<'vec2', THREE.Vector2>; mobs: THREE.UniformArrayNode<'vec4'>; mobData: THREE.Vector4[]; pulse: THREE.UniformNode<'float', number> }

export function minimap(side: number, map: THREE.Texture, size: number, half: THREE.Vector2, accent: number): Minimap {
  const player = uniform(new THREE.Vector2()), pulse = uniform(0);
  const mobData = Array.from({ length: MOB_SLOTS }, () => new THREE.Vector4(0, 0, 0, 0));
  const mobs = uniformArray(mobData, 'vec4' as const);
  const pad = 0.05, span = size - 2 * pad;
  const p = uv().sub(0.5).mul(size), frame = sdBox(p, vec2(size / 2), 0.08);
  const at = vec2(p.x.add(span / 2), p.y.negate().add(span / 2)).div(span).mul(side);
  const inside = at.x.greaterThanEqual(0).and(at.y.greaterThanEqual(0)).and(at.x.lessThan(side)).and(at.y.lessThan(side));
  const block = floor(texture(map, floor(at).add(0.5).div(side)).b.mul(255).add(0.5));
  let tone: N<'vec3'> = rgb(MINIMAP_COLOR[BLOCK.OUT]!);
  for (const [code, color] of Object.entries(MINIMAP_COLOR)) tone = select(is(block, Number(code)), rgb(color), tone);
  let color: N<'vec3'> = select(inside, tone, rgb(0x0b1016));
  const window = abs(sdBox(at.sub(player), vec2(half.x, half.y))).sub(0.35);
  color = mix(color, rgb(0xf4f7fb), coverage(window.mul(span / side)).mul(0.85));
  const kinds = [0x9bff6a, 0xf3efe6, 0xe9e4d0, 0xffb347];
  for (let i = 0; i < MOB_SLOTS; i++) {
    const m = mobs.element(i), kind = m.z;
    const tint = select(is(kind, 0), rgb(kinds[0]!), select(is(kind, 1), rgb(kinds[1]!), select(is(kind, 2), rgb(kinds[2]!), rgb(kinds[3]!))));
    const dot1 = coverage(length(at.sub(m.xy)).sub(0.9).mul(span / side)).mul(m.w);
    color = mix(color, rgb(0x0b0e12), coverage(length(at.sub(m.xy)).sub(1.3).mul(span / side)).mul(m.w).mul(0.7));
    color = mix(color, tint, dot1);
  }
  const me = length(at.sub(player));
  color = mix(color, rgb(0x0b0e12), coverage(me.sub(1.9).mul(span / side)));
  color = mix(color, rgb(accent).mul(1.25), coverage(me.sub(1.3).mul(span / side)));
  color = mix(color, rgb(accent), coverage(abs(me.sub(pulse.mul(4).add(1.5))).sub(0.3).mul(span / side)).mul(pulse.oneMinus()));
  const rim = coverage(abs(frame).sub(0.012));
  color = mix(color, rgb(0x2b3440), rim);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, toneMapped: false });
  material.colorNode = color;
  material.opacityNode = coverage(frame).mul(0.94);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size).rotateX(-Math.PI / 2), material);
  mesh.frustumCulled = false;
  return { mesh, player, mobs, mobData, pulse };
}

