import * as THREE from 'three/webgpu';
import { abs, atan, clamp, dot, exp, float, floor, fract, fwidth, length, max, min, mix, mod, positionWorld, pow, select, sin, smoothstep, sqrt, texture, uniform, uvec2, vec2, vec3 } from 'three/tsl';
import { grain, rgb, sdBox, sdCircle } from '@kit/shapes2d';
import { fbm } from '@kit/textures';
import { SIDE } from '@sim/envs/relicfog';

export type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;
type Pair = THREE.UniformNode<'vec2', THREE.Vector2>;

export const HALF = SIDE / 2;
const FIELD_STEPS = 8;
const SIGHT_STEPS = 5;
const WIPE_STEPS = 16;
const ENERGY_SCALE = 10;
const LUMA = vec3(0.2126, 0.7152, 0.0722);

const SPACE = 0x05060d;
const SLATE = 0x1a1e42;
const SLATE_DEEP = 0x10132c;
const SEAM = 0x06070f;
const CHARGE = 0x2fd8c6;
const DRAIN = 0xc2325c;
const VAPOUR = 0x3a2272;
const VAPOUR_LIGHT = 0xb094f0;
const ROCK = 0x736a61;
const ROCK_LIGHT = 0xbcae9c;
const ROCK_DARK = 0x221d1a;
const RUNE = 0xffcf6e;
const HAZE = 0x0b0f24;
const FOG = 0x181c3c;
const RIM = 0x6a6f9e;
const SIGHT = 0xcfc6ff;

export interface Sector {
  mesh: THREE.Mesh;
  now: THREE.DataTexture;
  was: THREE.DataTexture;
  marks: THREE.DataTexture;
  age: Level;
  fresh: Level;
  time: Level;
  slide: Pair;
  identity: Pair;
  dispose(): void;
}

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const coverOf = (aa: N<'float'>) => (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(aa)), 0, 1);

const cells = () => {
  const map = new THREE.DataTexture(new Float32Array(SIDE * SIDE * 4), SIDE, SIDE, THREE.RGBAFormat, THREE.FloatType);
  map.minFilter = map.magFilter = THREE.NearestFilter;
  map.needsUpdate = true;
  return map;
};

const bit = (value: N<'float'>, k: number): N<'float'> => mod(floor(value.div(2 ** k)), 2);

export function sectorMaterial(seed: number, accent: number, foe: number, lit: Lit): Sector {
  const now = cells(), was = cells(), marks = cells();
  const age = uniform(1e3), fresh = uniform(0), time = uniform(0), slide = uniform(new THREE.Vector2()), identity = uniform(new THREE.Vector2());
  const cloud = fbm((seed ^ 0x51f3) >>> 0, { size: 256, frequency: 4, octaves: 5 }), grit = fbm((seed ^ 0x9e37) >>> 0, { size: 256, frequency: 12, octaves: 3 });
  const nowMap = texture(now), wasMap = texture(was), markMap = texture(marks), cloudMap = texture(cloud), gritMap = texture(grit);

  const g = vec2(positionWorld.x.add(HALF), positionWorld.z.add(HALF));
  const aa = fwidth(g.x), cover = coverOf(aa);
  const inside = g.x.greaterThanEqual(0).and(g.y.greaterThanEqual(0)).and(g.x.lessThan(SIDE)).and(g.y.lessThan(SIDE));
  const cell = floor(g), f = fract(g), p = vec2(f.x.sub(0.5), float(0.5).sub(f.y));
  const at = (c: N<'vec2'>) => uvec2(clamp(c, vec2(0), vec2(SIDE - 1)));
  const wrap = (c: N<'vec2'>) => uvec2(mod(c, SIDE));
  const on = (condition: N<'bool'>) => select(condition, float(1), float(0));

  const centre = g.sub(HALF), radius = length(centre), front = age.div(WIPE_STEPS).mul(HALF * 1.5).sub(0.5);
  const warped = fresh.greaterThan(0.5), newer = select(warped, on(radius.lessThan(front)), float(1));
  const seam = fresh.mul(exp(abs(radius.sub(front)).mul(-3))).mul(on(front.lessThan(HALF * 1.5)));

  const n = nowMap.load(at(cell)), w = wasMap.load(at(cell)), m = markMap.load(at(cell));
  const fieldK = select(warped, newer, smoothstep(0, FIELD_STEPS, age)), sightK = select(warped, newer, smoothstep(0, SIGHT_STEPS, age));
  const energy = mix(w.y, n.y, fieldK);
  const seen = mix(bit(w.w, 0), bit(n.w, 0), sightK);
  const weight = select(newer.greaterThan(0.5), n.z, w.z);

  const gt = g.add(slide), tcell = floor(gt), tf = fract(gt), tp = vec2(tf.x.sub(0.5), float(0.5).sub(tf.y));
  const kindAt = (dx: number, dy: number) => select(newer.greaterThan(0.5), nowMap.load(wrap(tcell.add(vec2(dx, dy)))).x, wasMap.load(at(cell.add(vec2(dx, dy)))).x);
  const kind = kindAt(0, 0);
  const isRock = kind.greaterThan(1.5), isVapour = kind.greaterThan(0.5).and(kind.lessThan(1.5));
  const idTile = mod(tcell.sub(identity), SIDE), id = gt.sub(identity), hash = grain(idTile.add(seed % 101)), hash2 = grain(idTile.add(vec2(17.3, 5.1)));

  const plate = sdBox(p, vec2(0.465), 0.09), edge = plate.negate();
  const tone = gritMap.sample(g.div(SIDE).mul(2)).x;
  let base: N<'vec3'> = mix(rgb(SLATE_DEEP), rgb(SLATE), tone.mul(0.55).add(grain(cell).x.mul(0.2)).add(0.2));
  const charge = smoothstep(0, ENERGY_SCALE, energy), drain = smoothstep(0, ENERGY_SCALE, energy.negate());
  base = mix(base, rgb(CHARGE).mul(0.4), pow(charge, 0.8).mul(0.85));
  base = mix(base, rgb(DRAIN).mul(0.3), pow(drain, 0.8).mul(0.85));
  const hatch = cover(abs(fract(g.x.sub(g.y).mul(3)).sub(0.5)).div(3).sub(0.028)).mul(drain).mul(0.5);
  base = mix(base, rgb(DRAIN).mul(0.62), hatch);
  base = base.mul(smoothstep(0, 0.16, edge).mul(0.16).add(0.84));
  let color: N<'vec3'> = lit(base);
  color = color.add(rgb(CHARGE).mul(pow(charge, 1.5).mul(0.12)).mul(smoothstep(0.02, 0.45, edge)));
  color = mix(rgb(SEAM), color, cover(plate));

  const liveK = select(m.w.greaterThan(0.5), smoothstep(0, WIPE_STEPS, age), float(1)).mul(m.z).mul(on(weight.greaterThan(0.5)));
  const scoreAge = age.sub(6), scored = max(m.x, m.y).mul(exp(scoreAge.negate().div(5))).mul(on(scoreAge.greaterThanEqual(0)));
  const diamond = abs(p.x).add(abs(p.y));
  const rune = min(abs(diamond.sub(0.25)).sub(0.022), diamond.sub(0.075));
  const q = abs(p), ticks = min(sdBox(q.sub(vec2(0.37, 0.3)), vec2(0.014, 0.07)), sdBox(q.sub(vec2(0.3, 0.37)), vec2(0.07, 0.014)));
  const tint = select(m.x.greaterThan(m.y), rgb(accent), rgb(foe));
  const runeColor = mix(rgb(RUNE).mul(0.75), mix(rgb(RUNE), tint, 0.35).mul(2.2), scored);
  color = mix(color, runeColor, cover(rune).mul(liveK).mul(0.85));
  color = mix(color, rgb(RUNE).mul(0.6), cover(ticks).mul(liveK).mul(0.55));
  color = color.add(tint.mul(scored.mul(0.35)).mul(smoothstep(0, 0.5, length(p)).oneMinus()));

  const vapourAt = (dx: number, dy: number) => on(kindAt(dx, dy).greaterThan(0.5).and(kindAt(dx, dy).lessThan(1.5)));
  const right = vapourAt(1, 0), left = vapourAt(-1, 0), down = vapourAt(0, 1), up = vapourAt(0, -1);
  const toRight = float(0.5).sub(tp.x), toLeft = tp.x.add(0.5), toTop = float(0.5).sub(tp.y), toBottom = tp.y.add(0.5);
  const side = (open: N<'float'>, d: N<'float'>) => select(open.greaterThan(0.5), float(1), d);
  const corner = (dx: number, dy: number, a: N<'float'>, b: N<'float'>, da: N<'float'>, db: N<'float'>) =>
    select(a.greaterThan(0.5).and(b.greaterThan(0.5)).and(vapourAt(dx, dy).lessThan(0.5)), length(vec2(da, db)), float(1));
  const inset = min(
    min(min(side(right, toRight), side(left, toLeft)), min(side(up, toTop), side(down, toBottom))),
    min(min(corner(1, -1, right, up, toRight, toTop), corner(-1, -1, left, up, toLeft, toTop)), min(corner(1, 1, right, down, toRight, toBottom), corner(-1, 1, left, down, toLeft, toBottom))),
  );
  const swirl = vec2(time.mul(0.011), time.mul(-0.008));
  const puff = cloudMap.sample(id.div(SIDE).add(swirl)).x, wisp = cloudMap.sample(id.div(SIDE).mul(vec2(3.2, 2.4)).sub(swirl.mul(2))).x;
  const body = smoothstep(0.0, 0.34, inset.add(puff.sub(0.5).mul(0.3)).add(wisp.sub(0.5).mul(0.12)));
  const vapour = on(isVapour).mul(body).mul(wisp.mul(0.4).add(0.6));
  const vapourColor = mix(rgb(VAPOUR), rgb(VAPOUR_LIGHT), smoothstep(0.35, 0.95, puff.mul(0.55).add(wisp.mul(0.45))));
  color = mix(color, mix(lit(vapourColor), vapourColor.mul(0.75), 0.4), clamp(vapour.mul(0.78), 0, 1));

  const angle = atan(tp.y, tp.x);
  const lump = float(0.35).add(sin(angle.mul(3).add(hash.x.mul(6.283))).mul(0.045)).add(sin(angle.mul(5).add(hash.y.mul(6.283))).mul(0.03)).add(hash2.x.mul(0.04));
  const rock = length(tp).sub(lump);
  const bulge = sqrt(max(float(1).sub(pow(length(tp).div(lump.add(0.02)), 2)), 0));
  const normal = vec3(tp.x.div(lump), tp.y.div(lump), bulge);
  const shine = clamp(dot(normal, vec3(-0.5, 0.55, 0.67)), 0, 1);
  const spin = hash2.y.mul(6.283), pit = (turn: number, reach: number, size: number) => sdCircle(tp.sub(vec2(sin(spin.add(turn)), sin(spin.add(turn + 1.571))).mul(reach)), size);
  const pits = min(min(pit(0, 0.13, 0.078), pit(2.3, 0.19, 0.04)), pit(4.1, 0.2, 0.027));
  let stone: N<'vec3'> = mix(rgb(ROCK_DARK), mix(rgb(ROCK), rgb(ROCK_LIGHT), shine), smoothstep(0, 0.35, shine).mul(0.6).add(0.4));
  stone = stone.mul(gritMap.sample(id.div(SIDE).mul(4)).x.mul(0.35).add(0.8));
  stone = mix(stone, rgb(ROCK_DARK).mul(1.3), cover(pits).mul(0.6));
  stone = mix(stone, rgb(ROCK_LIGHT), cover(abs(pits.add(0.01)).sub(0.008)).mul(on(pits.lessThan(0.02))).mul(0.35));
  const shadow = smoothstep(-0.04, 0.12, length(tp.sub(vec2(0.07, -0.08))).sub(lump)).oneMinus().mul(0.55).mul(on(isRock));
  color = color.mul(shadow.oneMinus());
  color = mix(color, lit(stone), cover(rock).mul(on(isRock)));
  color = mix(color, rgb(ROCK_DARK).mul(0.5), cover(abs(rock).sub(0.012)).mul(on(isRock)).mul(0.8));

  const mist = cloudMap.sample(g.div(SIDE).mul(1.6).add(vec2(time.mul(0.008), time.mul(0.005)))).x;
  const curl = cloudMap.sample(g.div(SIDE).mul(3.7).sub(vec2(time.mul(0.013), time.mul(-0.006)))).x;
  const hidden = seen.oneMinus(), billow = smoothstep(0.3, 0.9, mist.mul(0.65).add(curl.mul(0.35)));
  const veiled = mix(vec3(dot(color, LUMA)), color, 0.3).mul(0.26).add(mix(rgb(HAZE), rgb(FOG), billow));
  color = mix(color, veiled, hidden);
  const sightAt = (dx: number, dy: number) => bit(nowMap.load(at(cell.add(vec2(dx, dy)))).w, 0);
  const own = bit(n.w, 0), side1 = (open: N<'float'>, d: N<'float'>) => select(open.lessThan(0.5), d, float(1));
  const border = min(min(side1(sightAt(1, 0), float(0.5).sub(p.x)), side1(sightAt(-1, 0), p.x.add(0.5))), min(side1(sightAt(0, -1), float(0.5).sub(p.y)), side1(sightAt(0, 1), p.y.add(0.5))));
  color = mix(color, rgb(SIGHT), cover(border.sub(0.03)).mul(own).mul(select(warped, newer, smoothstep(0, SIGHT_STEPS, age))).mul(0.6));
  color = color.add(rgb(0xd8ccff).mul(seam.mul(0.45)));

  const star = vec2(positionWorld.x, positionWorld.z).mul(1.7), starCell = floor(star), lucky = grain(starCell), spot = grain(starCell.add(9.1));
  const twinkle = sin(time.mul(lucky.y.mul(2.4).add(0.6)).add(lucky.x.mul(40))).mul(0.3).add(0.7);
  const glint = cover(length(fract(star).sub(spot.mul(0.6).add(0.2))).sub(pow(lucky.y, 3).mul(0.07).add(0.018)).div(1.7)).mul(on(lucky.x.lessThan(0.16)));
  const dust = cloudMap.sample(vec2(positionWorld.x, positionWorld.z).div(90).add(0.37)).x;
  let space: N<'vec3'> = mix(rgb(SPACE), rgb(0x141633), smoothstep(0.45, 0.8, dust).mul(0.8));
  space = space.add(mix(rgb(0xb8c4ff), rgb(0xffe6c8), lucky.y).mul(glint.mul(twinkle).mul(lucky.y.mul(0.6).add(0.5))));

  const reach = max(abs(centre.x), abs(centre.y)).sub(HALF);
  const frame = cover(abs(reach.sub(0.18)).sub(0.03));
  const bracket = frame.mul(on(max(abs(centre.x), abs(centre.y)).greaterThan(HALF - 2.2).and(min(abs(centre.x), abs(centre.y)).greaterThan(HALF - 2.2))));
  space = mix(space, rgb(RIM).mul(0.8), frame.mul(0.55));
  space = mix(space, rgb(0xe8e2ff), bracket.mul(0.7));

  const dock = (x: number, y: number, tint: number) => {
    const d = length(g.sub(vec2(x, y)));
    const arcs = min(abs(d.sub(0.78)).sub(0.03), abs(d.sub(1.06)).sub(0.018));
    return { shape: arcs, tint: rgb(tint), glow: exp(d.mul(-2.4)) };
  };
  const home = dock(0, 0, accent), away = dock(SIDE, SIDE, foe);
  color = mix(color, home.tint.mul(1.2), cover(home.shape).mul(0.6));
  color = mix(color, away.tint.mul(1.2), cover(away.shape).mul(0.6));
  color = color.add(home.tint.mul(home.glow.mul(0.14))).add(away.tint.mul(away.glow.mul(0.14)));

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = select(inside, color, space);
  const mesh = new THREE.Mesh(flat(SIDE * 12, SIDE * 12), material);
  mesh.renderOrder = 0;
  mesh.frustumCulled = false;
  return {
    mesh, now, was, marks, age, fresh, time, slide, identity,
    dispose() {
      for (const map of [now, was, marks, cloud, grit]) map.dispose();
      mesh.geometry.dispose();
      material.dispose();
    },
  };
}
