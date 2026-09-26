import * as THREE from 'three/webgpu';
import { abs, clamp, dot, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, select, sin, smoothstep, texture, uniform, uniformArray, uvec2, vec2, vec3 } from 'three/tsl';
import { fractal, rgb, sdBox, sdCircle } from '@kit/shapes2d';
import { NOBODY } from '@sim/envs/generals';
import { numeral, type Numerals } from './glyphs';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Level = THREE.UniformNode<'float', number>;
type Point = readonly [number, number];

export const RIM = 0.42;
export const WIPE_STEPS = 3;
export const FLOOD_STEPS = 0.45;
export const SWEEP_STEPS = 12;
const FOG_STEPS = 4;
const BORDER = 0.085;
const LINE = 0.018;
const LUMA = vec3(0.2126, 0.7152, 0.0722);

export const INKS = [0x3f78de, 0x3ba35c, 0x9a60d8, 0x27aeb0, 0xd8c23a, 0xcf5aa3, 0x9b6c45] as const;
export const NEUTRAL = 0xe6dcc6;
const GRANITE = 0x4a4642;
const ROCK_LIT = 0x9b9284;
const ROCK_SHADE = 0x5a544e;
const SNOW = 0xf2f4f6;
const GOLD = 0xf4c95e;
const GOLD_DEEP = 0x5e3c10;
const STONE = 0xaaa396;
const RAMPART = 0xd9d1c1;
const MORTAR = 0x35302b;
const WINDOW = 0xffd08a;
const HAZE = 0x222b3d;
const WALNUT = 0x2c1e15;
const BRASS = 0xc39a52;
const TABLE = 0x131922;

export interface BoardUniforms {
  age: Level;
  glow: Level;
  sweep: Level;
  front: Level;
  muster: Level;
  time: Level;
  palette: THREE.Color[];
  former: THREE.Color[];
}

export interface Board {
  mesh: THREE.Mesh;
  now: THREE.DataTexture;
  was: THREE.DataTexture;
  wipe: THREE.DataTexture;
  uniforms: BoardUniforms;
  dispose(): void;
}

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const coverOf = (aa: N<'float'>) => (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(aa)), 0, 1);

const cells = (side: number) => {
  const map = new THREE.DataTexture(new Float32Array(side * side * 4), side, side, THREE.RGBAFormat, THREE.FloatType);
  map.minFilter = map.magFilter = THREE.NearestFilter;
  map.needsUpdate = true;
  return map;
};

const edgeOf = (p: N<'vec2'>, a: Point, b: Point): N<'float'> => {
  const ex = b[0] - a[0], ey = b[1] - a[1], norm = Math.hypot(ex, ey);
  return dot(p.sub(vec2(a[0], a[1])), vec2(ey / norm, -ex / norm));
};

const triangle = (p: N<'vec2'>, a: Point, b: Point, c: Point): N<'float'> => max(max(edgeOf(p, a, b), edgeOf(p, b, c)), edgeOf(p, c, a));

export function crown(p: N<'vec2'>): N<'float'> {
  const x = abs(p.x);
  const band = sdBox(p.sub(vec2(0, -0.165)), vec2(0.25, 0.05), 0.012);
  const zig = select(x.lessThan(0.125), float(0.2).sub(x.mul(1.6)), x.sub(0.125).mul(1.28));
  const body = max(x.sub(0.25), max(float(-0.13).sub(p.y), p.y.sub(zig).mul(0.6)));
  const tips = min(sdCircle(vec2(x.sub(0.25), p.y.sub(0.175)), 0.036), sdCircle(p.sub(vec2(0, 0.215)), 0.04));
  return min(min(band, body), tips);
}

function tower(p: N<'vec2'>): N<'float'> {
  const body = sdBox(p.sub(vec2(0, -0.04)), vec2(0.15, 0.18), 0.008);
  const x = abs(p.x), merlon = sdBox(vec2(select(x.lessThan(0.06), x, x.sub(0.115)), p.y.sub(0.165)), vec2(0.034, 0.04), 0.004);
  const base = sdBox(p.sub(vec2(0, -0.225)), vec2(0.2, 0.035), 0.008);
  return min(min(body, merlon), base);
}

function mountain(p: N<'vec2'>, grain: N<'float'>, aa: N<'float'>): { shape: N<'float'>; color: N<'vec3'> } {
  const a: Point = [-0.07, 0.29], b: Point = [0.21, 0.07];
  const big = triangle(p, a, [-0.42, -0.31], [0.3, -0.31]), small = triangle(p, b, [-0.02, -0.31], [0.43, -0.31]);
  const onBig = smoothstep(aa.negate(), aa, small.sub(big));
  const ridge = mix(p.x.sub(b[0]).add(p.y.sub(b[1]).mul(0.18)), p.x.sub(a[0]).add(p.y.sub(a[1]).mul(0.18)), onBig);
  const lit = smoothstep(aa, aa.negate(), ridge);
  const rock = mix(rgb(ROCK_SHADE), rgb(ROCK_LIT), lit).mul(grain.mul(0.18).add(0.91));
  const snowLine = sin(p.x.mul(41)).mul(0.018).add(sin(p.x.mul(17).add(1.3)).mul(0.012)).add(mix(float(-0.005), float(0.15), onBig));
  const cap = smoothstep(aa.negate(), aa, p.y.sub(snowLine));
  return { shape: min(big, small), color: mix(rock, rgb(SNOW).mul(lit.mul(0.24).add(0.76)), cap) };
}

export function boardMaterial(side: number, set: Numerals, lit: Lit, seed: number): Board {
  const now = cells(side), was = cells(side), wipe = cells(side);
  const age = uniform(1e3), glow = uniform(1e3), sweep = uniform(0), front = uniform(-1e3), muster = uniform(0), time = uniform(0);
  const tones = () => Array.from({ length: NOBODY }, () => new THREE.Color()).concat(new THREE.Color(NEUTRAL));
  const palette = tones(), former = tones();
  const inks = uniformArray(palette, 'color'), formerInks = uniformArray(former, 'color');
  const nowMap = texture(now), wasMap = texture(was), wipeMap = texture(wipe);

  const g = vec2(positionWorld.x.add(side / 2), positionWorld.z.add(side / 2));
  const aa = fwidth(g.x), cover = coverOf(aa);
  const cell = floor(g), f = fract(g), p = vec2(f.x.sub(0.5), float(0.5).sub(f.y));
  const at = (dx: number, dy: number) => uvec2(clamp(cell.add(vec2(dx, dy)), vec2(0), vec2(side - 1)));
  const onBoard = (dx: number, dy: number) => {
    const c = cell.add(vec2(dx, dy));
    return c.x.greaterThanEqual(0).and(c.y.greaterThanEqual(0)).and(c.x.lessThan(side)).and(c.y.lessThan(side));
  };
  const inside = onBoard(0, 0);
  const on = (condition: N<'bool'>) => select(condition, float(1), float(0));

  const change = wipeMap.load(at(0, 0));
  const radius = smoothstep(0, WIPE_STEPS, age.sub(change.x)).mul(0.75);
  const diagonal = g.x.add(g.y);
  const swept = sweep.greaterThan(0.5);
  const fresh = select(swept, diagonal.lessThan(front), change.y.lessThan(0.5).or(length(p).lessThan(radius)));
  const nowCell = nowMap.load(at(0, 0)), wasCell = wasMap.load(at(0, 0));
  const state = select(fresh, nowCell, wasCell);
  const army = state.x, owner = state.y, kind = state.z;
  const ownerAt = (dx: number, dy: number) => select(onBoard(dx, dy), select(fresh, nowMap.load(at(dx, dy)).y, wasMap.load(at(dx, dy)).y), owner);

  const owned = owner.lessThan(NOBODY - 0.5);
  const right = ownerAt(1, 0), left = ownerAt(-1, 0), up = ownerAt(0, -1), down = ownerAt(0, 1);
  const differs = (o: N<'float'>) => abs(o.sub(owner)).greaterThan(0.5);
  const toRight = float(0.5).sub(p.x), toLeft = p.x.add(0.5), toTop = float(0.5).sub(p.y), toBottom = p.y.add(0.5);
  const far = float(1);
  const sideStroke = (o: N<'float'>, d: N<'float'>) => select(differs(o), d, far);
  const corner = (dx: number, dy: number, ex: N<'float'>, ey: N<'float'>, sx: N<'float'>, sy: N<'float'>) =>
    select(differs(ownerAt(dx, dy)).and(differs(sx).not()).and(differs(sy).not()), max(ex, ey), far);
  const stroke = min(
    min(min(sideStroke(right, toRight), sideStroke(left, toLeft)), min(sideStroke(up, toTop), sideStroke(down, toBottom))),
    min(min(corner(1, -1, toRight, toTop, right, up), corner(-1, -1, toLeft, toTop, left, up)), min(corner(1, 1, toRight, toBottom, right, down), corner(-1, 1, toLeft, toBottom, left, down))),
  );

  const isPeak = kind.greaterThan(0.5).and(kind.lessThan(1.5)), isCity = kind.greaterThan(1.5).and(kind.lessThan(2.5)), isCapital = kind.greaterThan(2.5);
  const slot = clamp(owner, 0, NOBODY).toInt();
  const ink = select(fresh, inks.element(slot), formerInks.element(slot)) as unknown as N<'vec3'>;
  const grain = fractal(g.mul(1.7).add(seed % 97), 3).mul(0.5).add(0.5);
  const edge = min(float(0.5).sub(abs(p.x)), float(0.5).sub(abs(p.y)));
  const relief = smoothstep(0.08, 0, edge).mul(mix(float(-0.14), float(0.1), smoothstep(aa.negate(), aa, p.y.sub(p.x)))).add(1);
  let tile: N<'vec3'> = select(isPeak, rgb(GRANITE).mul(grain.mul(0.3).add(0.85)), ink.mul(grain.mul(0.08).add(0.96))).mul(relief);
  tile = tile.add(ink.mul(muster.mul(exp(glow.negate().div(2.5))).mul(0.45)).mul(on(owned)));

  const rock = mountain(p, grain, aa), peak = on(isPeak);
  tile = mix(tile, rgb(0x221f1c), cover(rock.shape.sub(0.024)).mul(peak).mul(0.75));
  tile = mix(tile, rock.color, cover(rock.shape).mul(peak));

  const keep = tower(p.mul(1.08)), city = on(isCity);
  const stone = mix(rgb(STONE), mix(rgb(STONE), ink, 0.45).mul(1.1), on(owned)).mul(smoothstep(-0.16, 0.16, p.x).mul(-0.22).add(1.08));
  const door = sdBox(p.sub(vec2(0, -0.17)), vec2(0.045, 0.06), 0.045), pane = sdBox(p.sub(vec2(0, 0.03)), vec2(0.026, 0.045), 0.02);
  tile = mix(tile, rgb(MORTAR), cover(keep.sub(0.026)).mul(city).mul(0.85));
  tile = mix(tile, stone, cover(keep).mul(city));
  tile = mix(tile, rgb(MORTAR).mul(0.7), cover(door).mul(city));

  const regal = crown(p.mul(1.04)), capital = on(isCapital);
  const metal = mix(rgb(GOLD).mul(1.08), rgb(GOLD).mul(0.7), smoothstep(-0.2, 0.25, p.x.sub(p.y.mul(0.4))));
  tile = mix(tile, rgb(GOLD_DEEP), cover(regal.sub(0.03)).mul(capital));
  tile = mix(tile, metal, cover(regal).mul(capital));
  tile = mix(tile, ink.mul(0.85), cover(sdCircle(p.sub(vec2(0, -0.158)), 0.028)).mul(capital));

  const along = select(abs(p.x).greaterThan(abs(p.y)), p.y, p.x), crenel = abs(fract(along.mul(2.5).add(0.25)).sub(0.5)).lessThan(0.24);
  const wall = edge.sub(select(crenel, float(0.05), float(0.1)));
  const rampart = mix(rgb(RAMPART), ink, select(owned, float(0.22), float(0)));
  tile = mix(tile, rgb(MORTAR), cover(wall.sub(0.018)).mul(city));
  tile = mix(tile, rampart.mul(smoothstep(0, 0.1, edge).mul(0.3).add(0.8)), cover(wall).mul(city));
  const band = edge.sub(0.075), stud = sdCircle(abs(p).sub(vec2(0.39)), 0.055);
  tile = mix(tile, rgb(GOLD_DEEP), cover(min(band, stud).sub(0.016)).mul(capital));
  tile = mix(tile, metal.mul(0.92), cover(min(band, stud)).mul(capital));

  tile = mix(tile, tile.mul(0.32), cover(edge.sub(LINE)));
  tile = mix(tile, ink.mul(0.4), cover(stroke.sub(BORDER)).mul(on(owned)));
  const glint = change.y.mul(exp(glow.sub(change.x).negate().div(2.5))).mul(on(glow.greaterThanEqual(change.x))).mul(on(owned));
  tile = mix(tile, ink.mul(1.7).add(0.12), cover(abs(edge.sub(0.085)).sub(0.03)).mul(glint));

  let color: N<'vec3'> = lit(tile);
  color = mix(color, select(owned, rgb(WINDOW).mul(1.5), rgb(0x3a3f4a)), cover(pane).mul(city));

  const hidden = select(swept, select(fresh, nowCell.w, wasCell.w), mix(wasCell.w, nowCell.w, smoothstep(0, FOG_STEPS, age))).oneMinus();
  const smoke = fractal(g.mul(0.45).add(vec2(time.mul(0.06), time.mul(-0.04))), 4).mul(0.5).add(0.5);
  const veiled = select(owned, mix(color, vec3(dot(color, LUMA)), 0.3).mul(0.36), vec3(0.8, 0.9, 1.1).mul(dot(color, LUMA)).mul(0.085));
  color = mix(color, veiled.add(rgb(HAZE).mul(smoke.mul(0.65).add(0.2))), hidden);

  const shown = on(army.greaterThan(0.5).and(isPeak.not()));
  const number = numeral(set, p, army, 0.56, 0.92);
  color = mix(color, select(owned, ink.mul(0.15), rgb(0x1c1a17)).mul(hidden.mul(-0.5).add(1)), cover(number.sub(0.042)).mul(shown).mul(0.9));
  color = mix(color, vec3(0.97, 0.96, 0.93).mul(hidden.mul(-0.55).add(1)), cover(number).mul(shown));

  const seam = sweep.mul(exp(abs(diagonal.sub(front)).mul(-12))).mul(on(front.lessThan(2 * side + 1)));
  color = color.add(rgb(0xffe6b0).mul(seam.mul(0.28)));

  const centre = g.sub(side / 2), reach = max(abs(centre.x), abs(centre.y)).sub(side / 2);
  const wood = rgb(WALNUT).mul(fractal(g.mul(vec2(0.35, 2.2)), 3).mul(0.25).add(0.9)).mul(smoothstep(RIM, 0, reach).mul(0.35).add(0.8));
  const frame = mix(lit(wood), lit(rgb(BRASS)), cover(abs(reach.sub(RIM * 0.55)).sub(0.022)));
  const solid = select(inside, float(1), cover(reach.sub(RIM)));
  const drop = smoothstep(RIM + 0.9, RIM, reach.sub(0.12)).mul(0.55);
  const alpha = solid.add(solid.oneMinus().mul(drop));

  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = select(inside, color, frame).mul(solid).div(max(alpha, 1e-4));
  material.opacityNode = alpha;
  const span = side + 2 * RIM + 2;
  const mesh = new THREE.Mesh(flat(span, span), material);
  mesh.renderOrder = 2;
  mesh.frustumCulled = false;
  return {
    mesh, now, was, wipe,
    uniforms: { age, glow, sweep, front, muster, time, palette, former },
    dispose() {
      for (const map of [now, was, wipe]) map.dispose();
      mesh.geometry.dispose();
      material.dispose();
    },
  };
}

export function tableMesh(side: number, lit: Lit): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  const g = vec2(positionWorld.x, positionWorld.z);
  const leather = fractal(g.mul(0.21), 5).mul(0.5).add(0.5), pores = fractal(g.mul(2.3), 2).mul(0.5).add(0.5);
  material.colorNode = lit(rgb(TABLE).mul(leather.mul(0.35).add(0.82)).mul(pores.mul(0.12).add(0.94)));
  const mesh = new THREE.Mesh(flat(side * 8, side * 8), material);
  mesh.renderOrder = 0;
  mesh.frustumCulled = false;
  return mesh;
}
