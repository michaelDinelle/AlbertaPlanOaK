import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, dot, float, floor, fract, length, max, min, mix, mod, select, sign, sin, smoothstep, sqrt, uv, vec2, vec3 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { ACTION } from '@sim/envs/craftax';
import { ellipse, is, rotate, smin } from './world';

type N<T extends string> = THREE.Node<T>;

export const GLYPH = {
  HEART: 0, FOOD: 1, DROP: 2, BOLT: 3, LOG: 4, ROCK: 5, COAL: 6, IRON: 7, DIAMOND: 8, SAPLING: 9, PICKAXE: 10, SWORD: 11, TABLE: 12, FURNACE: 13,
  HAND: 14, WAIT: 15, MOON: 16, COW: 17, ZOMBIE: 18, SKULL: 19, SUN: 20, BERRIES: 21, BRICK: 22, SPROUT: 23, RETICLE: 24, CORNERS: 25, DOT: 26,
} as const;
export const TIER = { NONE: 0, WOOD: 1, STONE: 2, IRON: 3 } as const;
export const MODE = { SLOT: 0, VITAL: 1, BARE: 2, PANEL: 3, FRAME: 4 } as const;

type Glyph = readonly [number, number];

export const VITAL_GLYPHS: readonly Glyph[] = [[GLYPH.HEART, TIER.NONE], [GLYPH.FOOD, TIER.NONE], [GLYPH.DROP, TIER.NONE], [GLYPH.BOLT, TIER.NONE]];
export const ITEM_GLYPHS: readonly Glyph[] = [
  [GLYPH.LOG, TIER.NONE], [GLYPH.ROCK, TIER.NONE], [GLYPH.COAL, TIER.NONE], [GLYPH.IRON, TIER.NONE], [GLYPH.DIAMOND, TIER.NONE], [GLYPH.SAPLING, TIER.NONE],
  [GLYPH.PICKAXE, TIER.WOOD], [GLYPH.PICKAXE, TIER.STONE], [GLYPH.PICKAXE, TIER.IRON], [GLYPH.SWORD, TIER.WOOD], [GLYPH.SWORD, TIER.STONE], [GLYPH.SWORD, TIER.IRON],
];
export const ACHIEVEMENT_GLYPHS: readonly Glyph[] = [
  [GLYPH.LOG, TIER.NONE], [GLYPH.TABLE, TIER.NONE], [GLYPH.COW, TIER.NONE], [GLYPH.SAPLING, TIER.NONE], [GLYPH.DROP, TIER.NONE], [GLYPH.PICKAXE, TIER.WOOD],
  [GLYPH.SWORD, TIER.WOOD], [GLYPH.SPROUT, TIER.NONE], [GLYPH.ZOMBIE, TIER.NONE], [GLYPH.ROCK, TIER.NONE], [GLYPH.BRICK, TIER.NONE], [GLYPH.BERRIES, TIER.NONE],
  [GLYPH.SKULL, TIER.NONE], [GLYPH.PICKAXE, TIER.STONE], [GLYPH.SWORD, TIER.STONE], [GLYPH.SUN, TIER.NONE], [GLYPH.FURNACE, TIER.NONE], [GLYPH.COAL, TIER.NONE],
  [GLYPH.IRON, TIER.NONE], [GLYPH.DIAMOND, TIER.NONE], [GLYPH.PICKAXE, TIER.IRON], [GLYPH.SWORD, TIER.IRON],
];
export const ACTION_GLYPHS: Record<number, Glyph> = {
  [ACTION.NOOP]: [GLYPH.WAIT, TIER.NONE], [ACTION.DO]: [GLYPH.HAND, TIER.NONE], [ACTION.SLEEP]: [GLYPH.MOON, TIER.NONE], [ACTION.PLACE_STONE]: [GLYPH.BRICK, TIER.NONE],
  [ACTION.PLACE_TABLE]: [GLYPH.TABLE, TIER.NONE], [ACTION.PLACE_FURNACE]: [GLYPH.FURNACE, TIER.NONE], [ACTION.PLACE_PLANT]: [GLYPH.SPROUT, TIER.NONE],
  [ACTION.WOOD_PICKAXE]: [GLYPH.PICKAXE, TIER.WOOD], [ACTION.STONE_PICKAXE]: [GLYPH.PICKAXE, TIER.STONE], [ACTION.IRON_PICKAXE]: [GLYPH.PICKAXE, TIER.IRON],
  [ACTION.WOOD_SWORD]: [GLYPH.SWORD, TIER.WOOD], [ACTION.STONE_SWORD]: [GLYPH.SWORD, TIER.STONE], [ACTION.IRON_SWORD]: [GLYPH.SWORD, TIER.IRON],
};
export const ACTION_NAMES: Record<number, string> = {
  [ACTION.NOOP]: 'wait', [ACTION.DO]: 'do', [ACTION.SLEEP]: 'sleep', [ACTION.PLACE_STONE]: 'place stone', [ACTION.PLACE_TABLE]: 'place table', [ACTION.PLACE_FURNACE]: 'place furnace',
  [ACTION.PLACE_PLANT]: 'place sapling', [ACTION.WOOD_PICKAXE]: 'make wood pickaxe', [ACTION.STONE_PICKAXE]: 'make stone pickaxe', [ACTION.IRON_PICKAXE]: 'make iron pickaxe',
  [ACTION.WOOD_SWORD]: 'make wood sword', [ACTION.STONE_SWORD]: 'make stone sword', [ACTION.IRON_SWORD]: 'make iron sword',
};

const INK = 0x101316;

interface Art { d: N<'float'>; fill: N<'vec3'>; mark: N<'float'>; markTone: N<'vec3'> }

const art = (d: N<'float'>, fill: N<'vec3'>, mark: N<'float'> = float(9), markTone: N<'vec3'> = vec3(0)): Art => ({ d, fill, mark, markTone });
const shine = (p: N<'vec2'>, low: number, high: number) => smoothstep(low, high, p.y.sub(p.x.mul(0.6)));
const tone = (hex: number, p: N<'vec2'>) => rgb(hex).mul(shine(p, -0.9, 0.9).mul(0.35).add(0.8));

function heart(p: N<'vec2'>): N<'float'> {
  const h = vec2(abs(p.x), p.y.add(0.8)).div(1.6);
  const lobe = length(h.sub(vec2(0.25, 0.75))).sub(Math.SQRT2 / 4);
  const s = max(h.x.add(h.y), 0).mul(0.5);
  const core = sqrt(min(dot(h.sub(vec2(0, 1)), h.sub(vec2(0, 1))), dot(h.sub(vec2(s, s)), h.sub(vec2(s, s))))).mul(sign(h.x.sub(h.y)));
  return select(h.y.add(h.x).greaterThan(1), lobe, core).mul(1.6);
}

function drop(p: N<'vec2'>, r1: number, r2: number, h: number): N<'float'> {
  const q = vec2(abs(p.x), p.y), b = (r1 - r2) / h, a = Math.sqrt(1 - b * b), k = dot(q, vec2(-b, a));
  return select(k.lessThan(0), length(q).sub(r1), select(k.greaterThan(a * h), length(q.sub(vec2(0, h))).sub(r2), dot(q, vec2(a, b)).sub(r1)));
}

function polygon(p: N<'vec2'>, points: readonly (readonly [number, number])[]): N<'float'> {
  const area = points.reduce((sum, [ax, ay], i) => { const [bx, by] = points[(i + 1) % points.length]!; return sum + ax * by - bx * ay; }, 0), turn = area > 0 ? 1 : -1;
  let d: N<'float'> = float(-9);
  for (let i = 0; i < points.length; i++) {
    const [ax, ay] = points[i]!, [bx, by] = points[(i + 1) % points.length]!, ex = bx - ax, ey = by - ay, len = Math.hypot(ex, ey);
    d = max(d, dot(p.sub(vec2(ax, ay)), vec2((turn * ey) / len, (-turn * ex) / len)));
  }
  return d;
}

const tierTone = (tier: N<'float'>, p: N<'vec2'>) => select(is(tier, TIER.WOOD), tone(0xb07a45, p), select(is(tier, TIER.STONE), tone(0x9a9ca4, p), tone(0xe6ebf0, p)));

function glyphs(p: N<'vec2'>, tier: N<'float'>, clock: N<'float'>): Art[] {
  const heartArt = art(heart(p), tone(0xe8455a, p), sdCircle(p.sub(vec2(-0.32, 0.36)), 0.13), rgb(0xffd4da));
  const meat = ellipse(rotate(p.sub(vec2(-0.12, 0.12)), -0.62), [0, 0], [0.55, 0.4]);
  const bone = min(sdSegment(p, vec2(0.18, -0.2), vec2(0.52, -0.55), 0.09), min(sdCircle(p.sub(vec2(0.6, -0.46)), 0.12), sdCircle(p.sub(vec2(0.46, -0.62)), 0.12)));
  const foodArt = art(min(meat, bone), select(meat.lessThan(bone), tone(0xc4513a, p), rgb(0xf1e6d0)), ellipse(rotate(p.sub(vec2(-0.25, 0.25)), -0.62), [0, 0], [0.22, 0.1]), rgb(0xe98a6a));
  const dropArt = art(drop(p.sub(vec2(0, -0.28)), 0.52, 0.04, 1.1), tone(0x3f9fff, p), ellipse(p.sub(vec2(-0.2, -0.3)), [0, 0], [0.1, 0.18]), rgb(0xcde9ff));
  const bolt = min(polygon(p, [[0.24, 0.95], [-0.46, -0.08], [0.14, -0.08]]), polygon(p, [[-0.24, -0.95], [0.46, 0.08], [-0.14, 0.08]]));
  const boltArt = art(bolt, tone(0xffd23f, p));
  const lp = rotate(p, -0.35);
  const log = sdBox(lp.sub(vec2(-0.12, 0)), vec2(0.5, 0.3), 0.26);
  const face = ellipse(lp, [0.42, 0], [0.2, 0.3]);
  const rings = abs(fract(length(lp.sub(vec2(0.42, 0)).div(vec2(0.2, 0.3))).mul(2.6)).sub(0.5)).sub(0.36);
  const bark = abs(fract(lp.y.mul(3.4).add(sin(lp.x.mul(5)).mul(0.15))).sub(0.5)).sub(0.42).negate().max(face.negate());
  const logArt = art(min(log, face), select(face.lessThan(0), mix(rgb(0xe8b878), rgb(0xa8723e), coverage(rings.negate())), tone(0x7b4f2c, p)), bark, rgb(0x5a3a20));
  const rock = smin(sdBox(rotate(p, 0.3), vec2(0.5, 0.38), 0.22), sdCircle(p.sub(vec2(0.2, 0.28)), 0.34), 0.1);
  const rockArt = art(rock, tone(0x9a9ca3, p), sdSegment(p, vec2(-0.3, 0.2), vec2(0.05, -0.1), 0.035), rgb(0x6a6c73));
  const lump = smin(smin(sdCircle(p.sub(vec2(-0.22, -0.12)), 0.4), sdCircle(p.sub(vec2(0.25, -0.05)), 0.38), 0.15), sdCircle(p.sub(vec2(0.02, 0.3)), 0.34), 0.15);
  const coalArt = art(lump, tone(0x2a292f, p), sdCircle(p.sub(vec2(-0.12, 0.2)), 0.08), rgb(0x8d8c96));
  const ironArt = art(lump, tone(0xcf8a58, p), min(sdCircle(p.sub(vec2(-0.2, 0.15)), 0.1), sdCircle(p.sub(vec2(0.24, -0.12)), 0.08)), rgb(0xf6cfa4));
  const gem = polygon(p, [[-0.62, 0.22], [-0.34, 0.58], [0.34, 0.58], [0.62, 0.22], [0, -0.72]]);
  const facets = min(min(abs(p.y.sub(0.22)).sub(0.02), abs(dot(p.sub(vec2(-0.22, 0.22)), vec2(0.92, 0.39))).sub(0.02)), abs(dot(p.sub(vec2(0.22, 0.22)), vec2(-0.92, 0.39))).sub(0.02));
  const diamondArt = art(gem, mix(rgb(0x8fe6fb), rgb(0xf2fdff), shine(p, -0.5, 0.6)), max(facets, gem.add(0.06)), rgb(0x5cc7e6));
  const stem = sdSegment(p, vec2(0, -0.7), vec2(0, 0.25), 0.07);
  const leafL = ellipse(rotate(p.sub(vec2(-0.3, 0.3)), -0.6), [0, 0], [0.34, 0.17]);
  const leafR = ellipse(rotate(p.sub(vec2(0.3, 0.42)), 0.55), [0, 0], [0.36, 0.18]);
  const saplingArt = art(min(stem, min(leafL, leafR)), select(stem.lessThan(min(leafL, leafR)), rgb(0x3f7d2b), tone(0x74c052, p)), sdSegment(p, vec2(-0.5, 0.18), vec2(-0.12, 0.4), 0.02), rgb(0x3f7d2b));
  const handle = sdSegment(p, vec2(-0.62, -0.62), vec2(0.3, 0.3), 0.085);
  const pickHead = max(abs(sdCircle(p.sub(vec2(-0.28, -0.28)), 0.9)).sub(0.13), length(p.sub(vec2(0.35, 0.35))).sub(0.72));
  const pickArt = art(min(handle, pickHead), select(pickHead.lessThan(handle), tierTone(tier, p), tone(0x7a5230, p)));
  const blade = min(sdSegment(p, vec2(-0.3, -0.3), vec2(0.5, 0.5), 0.12), polygon(p, [[0.38, 0.62], [0.74, 0.74], [0.62, 0.38]]));
  const guard = sdSegment(p, vec2(-0.56, -0.06), vec2(-0.06, -0.56), 0.075);
  const grip = min(sdSegment(p, vec2(-0.34, -0.34), vec2(-0.64, -0.64), 0.065), sdCircle(p.sub(vec2(-0.7, -0.7)), 0.1));
  const swordArt = art(min(blade, min(guard, grip)), select(blade.lessThan(min(guard, grip)), tierTone(tier, p), select(guard.lessThan(grip), rgb(0x8a6a3e), rgb(0x5a3a22))), sdSegment(p, vec2(-0.2, -0.2), vec2(0.42, 0.42), 0.025), rgb(0xffffff).mul(0.85));
  const slab = sdBox(p.sub(vec2(0, 0.28)), vec2(0.78, 0.18), 0.05);
  const legs = min(sdBox(p.sub(vec2(-0.55, -0.3)), vec2(0.1, 0.42), 0.03), sdBox(p.sub(vec2(0.55, -0.3)), vec2(0.1, 0.42), 0.03));
  const tableArt = art(min(slab, legs), tone(0xb17a42, p), abs(p.y.sub(0.28)).sub(0.02).max(slab), rgb(0x6d4722));
  const box = sdBox(p, vec2(0.72, 0.72), 0.1);
  const mouth = sdBox(p.sub(vec2(0, -0.26)), vec2(0.32, 0.26), 0.2);
  const flame = smoothstep(0.1, -0.35, p.y.add(0.18).add(sin(p.x.mul(12).add(clock.mul(8))).mul(0.06)));
  const furnaceArt = art(box, tone(0x7f8188, p), mouth, mix(rgb(0x1a100b), mix(rgb(0xff6a1a), rgb(0xffd966), flame), flame.mul(0.9)));
  const palm = sdBox(p.sub(vec2(0, -0.2)), vec2(0.4, 0.34), 0.22);
  const fingers = [-0.3, -0.1, 0.1, 0.3].reduce<N<'float'>>((d, x) => min(d, sdSegment(p, vec2(x, 0.05), vec2(x, 0.48 - Math.abs(x) * 0.3), 0.105)), float(9));
  const thumb = sdSegment(p, vec2(-0.36, -0.2), vec2(-0.62, 0.12), 0.11);
  const handArt = art(min(palm, min(fingers, thumb)), tone(0xf0c39a, p), min(abs(fingers).sub(0.012).max(p.y.sub(0.1)).max(p.y.negate().sub(0.02)), float(9)), rgb(0xb98a64));
  const glass = min(polygon(p, [[-0.42, 0.62], [0.42, 0.62], [0, 0]]), polygon(p, [[-0.42, -0.62], [0.42, -0.62], [0, 0]]));
  const caps = min(sdBox(p.sub(vec2(0, 0.72)), vec2(0.52, 0.09), 0.04), sdBox(p.sub(vec2(0, -0.72)), vec2(0.52, 0.09), 0.04));
  const grains = min(polygon(p, [[-0.26, -0.62], [0.26, -0.62], [0, -0.3]]), polygon(p, [[-0.14, 0.2], [0.14, 0.2], [0, 0.02]]));
  const waitArt = art(min(glass.sub(0.04), caps), select(caps.lessThan(glass), tone(0x8a6a45, p), rgb(0xc9d6e0)), grains, rgb(0xe8c068));
  const moon = max(sdCircle(p, 0.64), sdCircle(p.sub(vec2(0.32, 0.24)), 0.54).negate());
  const moonArt = art(moon, tone(0xd9e2ff, p), min(sdCircle(p.sub(vec2(0.45, -0.35)), 0.06), sdCircle(p.sub(vec2(0.6, 0.05)), 0.045)), rgb(0xffffff));
  const cowFace = ellipse(p.sub(vec2(0, 0.08)), [0, 0], [0.4, 0.5]);
  const cowSnout = ellipse(p.sub(vec2(0, -0.36)), [0, 0], [0.38, 0.25]);
  const cowHorns = min(sdSegment(p, vec2(-0.26, 0.5), vec2(-0.52, 0.74), 0.075), sdSegment(p, vec2(0.26, 0.5), vec2(0.52, 0.74), 0.075));
  const cowEars = min(ellipse(rotate(p.sub(vec2(-0.5, 0.26)), 0.4), [0, 0], [0.2, 0.1]), ellipse(rotate(p.sub(vec2(0.5, 0.26)), -0.4), [0, 0], [0.2, 0.1]));
  const cowHead = min(min(cowFace, cowSnout), min(cowHorns, cowEars));
  const cowFill = select(cowSnout.lessThan(0), rgb(0xf0aaa2), select(cowHorns.lessThan(min(cowFace, cowEars)), rgb(0xe8dcbc), select(cowEars.lessThan(cowFace), rgb(0x3a302c), tone(0xf3efe6, p))));
  const cowSpots = min(max(sdCircle(p.sub(vec2(0.26, 0.34)), 0.2), cowFace), min(min(sdCircle(p.sub(vec2(-0.16, 0.06)), 0.07), sdCircle(p.sub(vec2(0.16, 0.06)), 0.07)), min(sdCircle(p.sub(vec2(-0.13, -0.38)), 0.055), sdCircle(p.sub(vec2(0.13, -0.38)), 0.055))));
  const cowArt = art(cowHead, cowFill, cowSpots, rgb(0x24201e));
  const zHead = sdBox(p.sub(vec2(0, 0)), vec2(0.46, 0.58), 0.22);
  const zHair = max(zHead, p.y.negate().add(0.3).add(sin(p.x.mul(14)).mul(0.06)));
  const zFeatures = min(min(sdBox(p.sub(vec2(-0.19, 0.06)), vec2(0.11, 0.08), 0.03), sdBox(p.sub(vec2(0.19, 0.06)), vec2(0.11, 0.08), 0.03)), sdBox(p.sub(vec2(0, -0.32)), vec2(0.22, 0.06), 0.02));
  const zArt = art(zHead, select(zHair.lessThan(0), rgb(0x34422a), tone(0x7fae6a, p)), zFeatures, rgb(0x1b2416));
  const skull = min(sdCircle(p.sub(vec2(0, 0.14)), 0.55), sdBox(p.sub(vec2(0, -0.42)), vec2(0.3, 0.2), 0.06));
  const skullArt = art(skull, tone(0xece6d4, p), min(min(sdCircle(p.sub(vec2(-0.2, 0.1)), 0.14), sdCircle(p.sub(vec2(0.2, 0.1)), 0.14)), polygon(p, [[0, -0.08], [-0.07, -0.2], [0.07, -0.2]])), rgb(0x1d1a17));
  const angle = atan(p.y, p.x), rays = abs(fract(angle.div(Math.PI / 4)).sub(0.5)).mul(length(p)).mul(Math.PI / 4).sub(0.07).max(abs(length(p).sub(0.7)).sub(0.16));
  const sunArt = art(min(sdCircle(p, 0.4), rays), tone(0xffc93f, p), sdCircle(p.sub(vec2(-0.12, 0.12)), 0.12), rgb(0xfff1b8));
  const bush = smin(smin(sdCircle(p.sub(vec2(-0.3, 0.05)), 0.42), sdCircle(p.sub(vec2(0.3, 0.1)), 0.4), 0.15), sdCircle(p.sub(vec2(0, -0.3)), 0.42), 0.15);
  const fruit = min(min(sdCircle(p.sub(vec2(-0.3, 0.2)), 0.15), sdCircle(p.sub(vec2(0.28, 0.25)), 0.15)), sdCircle(p.sub(vec2(0.05, -0.3)), 0.15));
  const berriesArt = art(bush, tone(0x3f8a3c, p), fruit, rgb(0xe0453a));
  const brick = sdBox(p, vec2(0.7, 0.62), 0.08);
  const course = floor(p.y.add(0.62).div(0.413)), joint = min(abs(fract(p.y.add(0.62).div(0.413)).sub(0.5)).negate().add(0.47).mul(0.413), abs(fract(p.x.div(0.7).add(mod(course, 2).mul(0.5))).sub(0.5)).negate().add(0.47).mul(0.7));
  const brickArt = art(brick, tone(0x8d8f96, p), joint.max(brick), rgb(0x4b4d54));
  const mound = ellipse(p.sub(vec2(0, -0.45)), [0, 0], [0.7, 0.3]);
  const sprout = min(sdSegment(p, vec2(0, -0.4), vec2(0, 0.25), 0.06), min(ellipse(rotate(p.sub(vec2(-0.25, 0.3)), -0.6), [0, 0], [0.3, 0.15]), ellipse(rotate(p.sub(vec2(0.26, 0.4)), 0.55), [0, 0], [0.32, 0.16])));
  const sproutArt = art(min(mound, sprout), select(mound.lessThan(sprout), tone(0x6d4a2c, p), tone(0x74c052, p)));
  const bracket = (inset: number, width: number) => {
    const a = abs(p), edge = max(a.x, a.y), corner = min(a.x, a.y);
    return max(abs(edge.sub(inset)).sub(width), float(inset - 0.4).sub(corner));
  };
  const reticleArt = art(bracket(0.86, 0.07), rgb(0x7fe8ff));
  const cornersArt = art(bracket(0.9, 0.045), rgb(0xffffff));
  const dotArt = art(sdCircle(p, 0.7), rgb(0xffffff));
  return [heartArt, foodArt, dropArt, boltArt, logArt, rockArt, coalArt, ironArt, diamondArt, saplingArt, pickArt, swordArt, tableArt, furnaceArt, handArt, waitArt, moonArt, cowArt, zArt, skullArt, sunArt, berriesArt, brickArt, sproutArt, reticleArt, cornersArt, dotArt];
}

const SEGMENTS: readonly (readonly [number, number, number, number])[] = [[-0.3, 0.5, 0.3, 0.5], [0.3, 0.5, 0.3, 0], [0.3, 0, 0.3, -0.5], [-0.3, -0.5, 0.3, -0.5], [-0.3, -0.5, -0.3, 0], [-0.3, 0, -0.3, 0.5], [-0.3, 0, 0.3, 0]];
const DIGITS = [0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f] as const;

function digit(p: N<'vec2'>, value: N<'float'>): N<'float'> {
  let mask: N<'float'> = float(DIGITS[0]);
  for (let k = 1; k < 10; k++) mask = select(is(value, k), float(DIGITS[k]!), mask);
  let d: N<'float'> = float(9);
  SEGMENTS.forEach(([ax, ay, bx, by], k) => {
    const on = mod(floor(mask.div(2 ** k)), 2).greaterThan(0.5);
    d = min(d, select(on, sdSegment(p, vec2(ax * 0.78, ay * 0.84), vec2(bx * 0.78, by * 0.84), 0.11), float(9)));
  });
  return d;
}

export class Icons {
  readonly mesh: THREE.InstancedMesh;
  private readonly icon: THREE.InstancedBufferAttribute;
  private readonly state: THREE.InstancedBufferAttribute;
  private readonly shape: THREE.InstancedBufferAttribute;
  private readonly matrix = new THREE.Matrix4();
  private readonly position = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();
  private used = 0;

  constructor(readonly capacity: number, accent: number, clock: N<'float'>, order: number) {
    const geometry = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
    const make = () => new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    this.icon = make();
    this.state = make();
    this.shape = make();
    geometry.setAttribute('aIcon', this.icon);
    geometry.setAttribute('aState', this.state);
    geometry.setAttribute('aShape', this.shape);
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, toneMapped: false });
    const icon = attribute<'vec4'>('aIcon', 'vec4'), state = attribute<'vec4'>('aState', 'vec4'), shape = attribute<'vec4'>('aShape', 'vec4');
    const ink = iconInk(icon, state, shape, accent, clock);
    material.colorNode = ink.color;
    material.opacityNode = ink.opacity;
    this.mesh = instancedMesh(geometry, material, capacity);
    this.mesh.name = 'craftaxIcons';
    this.mesh.renderOrder = order;
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
  }

  begin(): void {
    this.used = 0;
  }

  add(x: number, z: number, width: number, height: number, icon: readonly [number, number, number, number], state: readonly [number, number, number, number], shape: readonly [number, number, number, number] = [width / height, 0, 0, 0]): void {
    if (this.used >= this.capacity) throw new Error(`craftaxIcons holds ${this.capacity} icons`);
    this.mesh.setMatrixAt(this.used, this.matrix.compose(this.position.set(x, 0, z), this.quaternion, this.scale.set(width, 1, height)));
    this.icon.setXYZW(this.used, ...icon);
    this.state.setXYZW(this.used, ...state);
    this.shape.setXYZW(this.used, ...shape);
    this.used++;
  }

  commit(): void {
    this.mesh.count = this.used;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.icon.needsUpdate = true;
    this.state.needsUpdate = true;
    this.shape.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.mesh.removeFromParent();
  }
}

function iconInk(icon: N<'vec4'>, state: N<'vec4'>, shape: N<'vec4'>, accent: number, clock: N<'float'>): { color: N<'vec3'>; opacity: N<'float'> } {
  const glyph = icon.x, tier = icon.y, value = icon.z, mode = icon.w;
  const lit = state.x, ring = state.y, flash = state.z, alpha = state.w, aspect = shape.x, scan = shape.y, fill = shape.z, gold = shape.w;
  const p = uv().sub(0.5).mul(2).mul(vec2(aspect, 1));
  const slot = is(mode, MODE.SLOT), vital = is(mode, MODE.VITAL), panel = is(mode, MODE.PANEL), frame = is(mode, MODE.FRAME);
  const counted = slot.and(value.greaterThanEqual(1));
  const glyphAt = select(vital, p.add(vec2(aspect.sub(1), 0)).mul(1.2), select(slot, p.mul(1.18), p)).mul(select(counted, float(1.12), float(1)));
  const gp = glyphAt.add(select(counted, vec2(0.1, -0.08), vec2(0)));
  const all = glyphs(gp, tier, clock);
  let d: N<'float'> = all[0]!.d, tint: N<'vec3'> = all[0]!.fill, mark: N<'float'> = all[0]!.mark, markTone: N<'vec3'> = all[0]!.markTone;
  all.forEach((entry, k) => {
    if (k === 0) return;
    const hit = is(glyph, k);
    d = select(hit, entry.d, d);
    tint = select(hit, entry.fill, tint);
    mark = select(hit, entry.mark, mark);
    markTone = select(hit, entry.markTone, markTone);
  });
  let face: N<'vec3'> = mix(tint, markTone, coverage(mark));
  const dull = mix(vec3(dot(face, vec3(0.3, 0.5, 0.2))), face, 0.3).mul(0.4).add(0.04);
  face = mix(dull, face, lit);
  face = mix(face, rgb(0xffffff), flash.mul(0.55));
  const outline = select(is(glyph, GLYPH.RETICLE).or(is(glyph, GLYPH.CORNERS)), float(0), float(0.1));
  const glyphCover = coverage(d.sub(outline));
  let color: N<'vec3'> = mix(rgb(INK), face, coverage(d));
  let opacity: N<'float'> = glyphCover;

  const box = sdBox(p, vec2(aspect.sub(0.06), 0.94), 0.26);
  const plate = mix(rgb(0x121920), rgb(0x1a232c), shine(p, -1, 1));
  const edge = coverage(abs(box).sub(0.035));
  const halo = coverage(abs(sdBox(p, vec2(aspect.sub(0.02), 0.98), 0.3)).sub(0.07)).mul(ring);
  const scanned = coverage(abs(sdBox(p, vec2(aspect.sub(0.02), 0.98), 0.3)).sub(0.05)).mul(scan);
  const slotColor = mix(mix(plate, rgb(0x33414e), edge), color, glyphCover);
  const slotOpacity = max(coverage(box).mul(0.94), glyphCover);

  const count = digit(p.sub(vec2(aspect.sub(0.34), -0.52)).div(0.42), value).mul(0.42);
  const countInk = coverage(count.sub(0.045)).mul(select(counted, float(1), float(0)));
  const countFace = coverage(count).mul(select(counted, float(1), float(0)));

  const pipsX = p.x.add(aspect).sub(1.95), span = aspect.mul(2).sub(3.05), cell = span.div(9);
  const pipIndex = clamp(floor(pipsX.div(cell)), 0, 8);
  const pip = sdBox(vec2(pipsX.sub(pipIndex.add(0.5).mul(cell)), p.y), vec2(cell.mul(0.36), 0.3), 0.08);
  const inBar = pipsX.greaterThanEqual(0).and(pipsX.lessThan(span));
  const pipFull = select(pipIndex.lessThan(value.sub(0.5)), float(1), float(0));
  const pipCover = coverage(pip).mul(select(inBar, float(1), float(0)));
  const vitalDigit = digit(p.sub(vec2(aspect.sub(0.42), 0)).div(0.62), value).mul(0.62);
  const warn = select(value.lessThan(2.5), sin(clock.mul(9)).mul(0.5).add(0.5), float(0)), alarm = select(value.lessThan(2.5), warn.mul(0.7).add(0.3), float(0));
  const pipHue = select(is(glyph, GLYPH.HEART), rgb(0xe8455a), select(is(glyph, GLYPH.FOOD), rgb(0xe0874f), select(is(glyph, GLYPH.DROP), rgb(0x3f9fff), rgb(0xffd23f))));
  const pipTone = mix(mix(rgb(0x2a333c), pipHue, pipFull), rgb(0xffffff), warn.mul(pipFull).mul(0.45));
  let vitalColor: N<'vec3'> = mix(plate, rgb(0x33414e), edge);
  vitalColor = mix(vitalColor, pipTone, pipCover);
  vitalColor = mix(vitalColor, color, glyphCover);
  vitalColor = mix(vitalColor, rgb(INK), coverage(vitalDigit.sub(0.05)));
  vitalColor = mix(vitalColor, mix(rgb(0xf2f5f7), rgb(0xff6a5a), alarm), coverage(vitalDigit));
  const vitalOpacity = max(coverage(box).mul(0.94), glyphCover);

  const glass = sdBox(p, vec2(aspect.sub(0.02), 0.98), fill);
  const panelColor = mix(rgb(0x0d1217), rgb(0x28323c), coverage(abs(glass).sub(0.012)));
  const panelOpacity = coverage(glass).mul(0.93);

  const border = abs(sdBox(p, vec2(aspect.sub(0.012), 0.988), fill)).sub(0.006);
  const corner = abs(p.x).greaterThan(aspect.sub(0.32)).and(abs(p.y).greaterThan(0.68));
  const frameCover = max(coverage(border).mul(0.5), coverage(border.sub(0.014)).mul(select(corner, float(1), float(0))));

  color = select(slot, mix(slotColor, rgb(INK), countInk), select(vital, vitalColor, select(panel, panelColor, select(frame, rgb(accent), color))));
  color = select(slot, mix(color, rgb(0xf2f5f7), countFace), color);
  color = mix(color, mix(rgb(accent).mul(1.25), rgb(0xffd166).mul(1.2), gold), halo);
  color = mix(color, rgb(0x7fe8ff), scanned.mul(select(ring.greaterThan(0.5), float(0.5), float(1))));
  opacity = select(slot, max(slotOpacity, countInk), select(vital, vitalOpacity, select(panel, panelOpacity, select(frame, frameCover, opacity))));
  opacity = max(opacity, max(halo, scanned));
  return { color, opacity: opacity.mul(alpha) };
}

