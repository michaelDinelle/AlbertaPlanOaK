import * as THREE from 'three/webgpu';
import { abs, atan, clamp, exp, float, floor, fract, fwidth, length, max, min, mix, select, sin, smoothstep, step, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, grain, noise, rgb, sdBox, sdSegment } from '@kit/shapes2d';
import { finish, KITS, MAST, over, sheet, TOKEN, type Layer, type Level } from './figures';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Tint = THREE.UniformNode<'color', THREE.Color>;

export interface Dims { length: number; width: number; post: number; bar: number; box: number; boxWidth: number; area: number; areaWidth: number; mark: number; circle: number }

export const LINE = 0.12;
export const NET = 2;
export const RUN_X = 5.5;
export const RUN_Y = 4;
export const BOARD = 0.9;
export const APRON = 2.2;
export const SPAN = 420;
export const HEAD = new THREE.Vector2(6.8, 1.5);
const CORNER = 1;
const SPOT = 0.12;
const ROW = 0.85;
const SEAT = 0.55;
const AISLE = 19;
const TIER = 9;
const ROOF = 9;
const PANEL = 7;
const MESH = 0.34;
const TIMER = TOKEN + 1.25;

const GRASS = [0x2a6f32, 0x33823a] as const;
const LINE_WHITE = 0xeef2ea;
const FLAG = 0xf3d34a;
const LED = [0x7ddc8a, 0x1fa99a, 0xdff6ff] as const;

export interface PitchLevels { time: Level; cheer: Level; cheerSide: Level; netAge: Level; netSide: Level; netY: Level; celebrate: Tint }

export function pitchLevels(): PitchLevels {
  return { time: uniform(0), cheer: uniform(0), cheerSide: uniform(1), netAge: uniform(1), netSide: uniform(1), netY: uniform(0), celebrate: uniform(new THREE.Color(KITS[0])) };
}

const inner = (distance: N<'float'>, half: N<'float'>): N<'float'> => abs(distance.add(half)).sub(half);
const tint = (value: Tint): N<'vec3'> => value as unknown as N<'vec3'>;
const floored = (value: N<'float'>, period: number): N<'float'> => value.sub(floor(value.div(period)).mul(period));

function ledColor(p: N<'vec2'>, d: Dims, levels: PitchLevels): N<'vec3'> {
  const ends = abs(p.x).sub(d.length + RUN_X).greaterThan(abs(p.y).sub(d.width + RUN_Y));
  const along = select(ends, p.y, p.x);
  const panel = floor(along.div(PANEL)), local = fract(along.div(PANEL));
  const hue = fract(panel.mul(0.37).add(levels.time.mul(0.045)).add(sin(panel.mul(1.7)).mul(0.2)));
  const base = mix(mix(rgb(LED[1]), rgb(LED[0]), smoothstep(0.1, 0.5, hue)), rgb(LED[2]), smoothstep(0.72, 0.95, hue));
  const sweep = sin(along.mul(0.9).sub(levels.time.mul(2.4))).mul(0.5).add(0.5);
  const seams = smoothstep(0, 0.03, local).mul(smoothstep(1, 0.97, local)).mul(0.8).add(0.2);
  const quiet = base.mul(sweep.mul(0.35).add(0.55)).mul(seams);
  const chevron = step(0.5, fract(along.mul(0.55).sub(levels.time.mul(1.6))));
  return mix(quiet, tint(levels.celebrate).mul(chevron.mul(0.3).add(0.42)), levels.cheer);
}

export function towers(d: Dims): THREE.Vector2[] {
  const x = d.length + RUN_X + BOARD + APRON + 2.5, y = d.width + RUN_Y + BOARD + APRON - 0.2;
  return [new THREE.Vector2(-x, -y), new THREE.Vector2(x, -y), new THREE.Vector2(x, y), new THREE.Vector2(-x, y)];
}

export const aimOf = (tower: THREE.Vector2): number => Math.atan2(-tower.y, -tower.x) + Math.PI / 2;

const turn = (p: N<'vec2'>, angle: number): N<'vec2'> => {
  const c = Math.cos(angle), s = Math.sin(angle);
  return vec2(p.x.mul(c).add(p.y.mul(s)), p.y.mul(c).sub(p.x.mul(s)));
};

function floodlights(p: N<'vec2'>, d: Dims) {
  let frame: N<'float'> = float(1e3), lamps: N<'float'> = float(0), glow: N<'float'> = float(0);
  for (const tower of towers(d)) {
    const u = turn(p.sub(vec2(tower.x, tower.y)), aimOf(tower));
    const head = sdBox(u, vec2(HEAD.x / 2, HEAD.y / 2), 0.12), mast = sdBox(u.sub(vec2(0, HEAD.y / 2 + 0.9)), vec2(0.55, 0.55), 0.08);
    const cellX = HEAD.x / 8, cellY = HEAD.y / 2;
    const local = vec2(fract(u.x.add(HEAD.x / 2).div(cellX)).sub(0.5).mul(cellX), fract(u.y.add(HEAD.y / 2).div(cellY)).sub(0.5).mul(cellY));
    frame = min(frame, min(head, mast));
    lamps = max(lamps, coverage(sdBox(local, vec2(cellX * 0.36, cellY * 0.3), 0.06)).mul(coverage(head.add(0.1))));
    glow = glow.add(exp(max(head, 0).mul(-0.45)).mul(0.35));
  }
  return { frame: coverage(frame), lamps, glow };
}

function goalFrame(p: N<'vec2'>, d: Dims, levels: PitchLevels, px: N<'float'>) {
  const ay = abs(p.y), q = float(d.length).sub(abs(p.x)), behind = q.negate(), side = select(p.x.lessThan(0), float(-1), float(1));
  const mouth = sdBox(vec2(behind.sub(NET / 2), p.y), vec2(NET / 2, d.post));
  const mine = step(0, p.x.mul(levels.netSide));
  const fade = levels.netAge.oneMinus();
  const offset = p.y.sub(levels.netY);
  const push = exp(offset.mul(offset).add(behind.mul(behind)).mul(-0.35)).mul(sin(levels.netAge.mul(Math.PI)).add(fade.mul(0.4))).mul(fade).mul(mine);
  const drawn = behind.sub(push.mul(0.9));
  const knit = vec2(drawn.add(p.y), drawn.sub(p.y)).div(MESH * Math.SQRT2);
  const strands = min(abs(fract(knit.x.add(0.5)).sub(0.5)), abs(fract(knit.y.add(0.5)).sub(0.5))).mul(MESH * Math.SQRT2);
  const net = coverage(mouth);
  const cords = coverage(strands.sub(max(px.mul(0.35), 0.022))).mul(net);
  const at = vec2(behind, p.y);
  const stanchions = min(min(sdSegment(at, vec2(0, d.post), vec2(NET, d.post), 0.06), sdSegment(at, vec2(0, -d.post), vec2(NET, -d.post), 0.06)), sdSegment(at, vec2(NET, -d.post), vec2(NET, d.post), 0.06));
  const foot = vec2(side.mul(d.length), 0), high = vec2(0, d.post), low = vec2(0, -d.post);
  const shade = towers(d).map((tower) => {
    const lean = (x: number) => new THREE.Vector2(x * d.length, 0).sub(tower).multiplyScalar(d.bar / (MAST - d.bar));
    const west = lean(-1), east = lean(1), cast = select(p.x.lessThan(0), vec2(west.x, west.y), vec2(east.x, east.y));
    const bar = sdSegment(p, foot.add(low).add(cast), foot.add(high).add(cast), 0.07);
    const posts = min(sdSegment(p, foot.add(high), foot.add(high).add(cast), 0.07), sdSegment(p, foot.add(low), foot.add(low).add(cast), 0.07));
    return coverage(min(bar, posts)).mul(0.13);
  }).reduce((a, b) => a.add(b));
  const posts = length(vec2(q.sub(LINE / 2), ay.sub(d.post + 0.16))).sub(0.16);
  const bar = sdBox(vec2(q.sub(LINE / 2), p.y), vec2(0.09, d.post + 0.16));
  return { net, cords, stanchions: coverage(stanchions).mul(step(-0.1, behind)), shade, posts: coverage(posts), bar: coverage(bar) };
}

export function pitchMaterial(d: Dims, lit: Lit, levels: PitchLevels): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ fog: false });
  const L = d.length, W = d.width;
  const p = uv().sub(0.5).mul(SPAN), ay = abs(p.y);
  const q = vec2(float(L).sub(abs(p.x)), ay);
  const px = fwidth(p.x), half = max(px.mul(0.55), LINE / 2);

  const stripes = q.x.div(d.area);
  const band = coverage(abs(fract(stripes.sub(0.5).mul(0.5)).mul(2).sub(1)).sub(0.5).mul(-d.area));
  const mottle = fractal(p.mul(0.16), 3).mul(0.035).add(noise(p.mul(1.9)).mul(0.025)).add(1);
  const worn = exp(length(q.sub(vec2(d.area * 0.45, 0))).mul(-0.28)).mul(0.5).add(exp(length(q.sub(vec2(d.mark, 0))).mul(-1.2)).mul(0.35));
  let grass: N<'vec3'> = mix(rgb(GRASS[0]), rgb(GRASS[1]), band).mul(mottle);
  grass = mix(grass, rgb(0x6c7a45), clamp(worn, 0, 1).mul(0.35));

  const lines = [
    inner(sdBox(p, vec2(L, W)), half),
    max(abs(p.x).sub(half), ay.sub(W)),
    inner(length(p).sub(d.circle), half),
    length(p).sub(max(float(SPOT), half)),
    inner(sdBox(q.sub(vec2(d.box / 2, 0)), vec2(d.box / 2, d.boxWidth)), half),
    inner(sdBox(q.sub(vec2(d.area / 2, 0)), vec2(d.area / 2, d.areaWidth)), half),
    length(q.sub(vec2(d.mark, 0))).sub(max(float(SPOT), half)),
    max(inner(length(q.sub(vec2(d.mark, 0))).sub(d.circle), half), float(d.box).sub(q.x)),
    max(inner(length(vec2(q.x, float(W).sub(ay))).sub(CORNER), half), max(q.x.negate(), ay.sub(W))),
  ].reduce((a, b) => min(a, b));
  const goal = goalFrame(p, d, levels, px);
  const flagAt = vec2(q.x.negate(), ay.sub(W));
  const pole = length(flagAt).sub(0.13);
  const pennant = max(max(flagAt.x.negate(), flagAt.y.negate()), flagAt.x.add(flagAt.y).sub(0.62));
  const lights = floodlights(p, d);

  const ring = sdBox(p, vec2(L + RUN_X + BOARD / 2, W + RUN_Y + BOARD / 2));
  const board = coverage(abs(ring).sub(BOARD / 2));
  const lip = coverage(abs(ring.add(BOARD / 2 - 0.06)).sub(0.05));
  const outside = step(0, ring.sub(BOARD / 2));
  const reach = vec2(L + RUN_X + BOARD + APRON, W + RUN_Y + BOARD + APRON);
  const standQ = abs(p).sub(reach), front = max(standQ.x, standQ.y), stand = step(0, front);
  const ends = standQ.x.greaterThan(standQ.y);
  const depth = select(ends, standQ.x, standQ.y), along = select(ends, p.y, p.x);
  const row = floor(depth.div(ROW)), seat = floor(along.div(SEAT));
  const cell = vec2(fract(along.div(SEAT)).sub(0.5).mul(SEAT), fract(depth.div(ROW)).sub(0.5).mul(ROW));
  const side = select(ends, select(p.x.lessThan(0), float(0), float(1)), float(2));
  const draw = grain(vec2(seat.add(side.mul(977)), row.add(side.mul(131))));
  const fan = grain(vec2(row.mul(3.1).add(seat), seat.mul(0.7).sub(row)));
  const home = rgb(KITS[0]), away = rgb(KITS[2]);
  const favoured = select(side.lessThan(0.5), home, select(side.lessThan(1.5), away, mix(home, away, step(0.5, fan.y))));
  const shirt = mix(select(draw.y.lessThan(0.5), favoured, select(draw.y.lessThan(0.68), rgb(0xe8e4dc), select(draw.y.lessThan(0.86), rgb(0x2b303b), rgb(0x8a3a3a)))), rgb(0x3a3f4a), 0.3);
  const plastic = select(side.lessThan(0.5), mix(home, rgb(0x1a1d24), 0.72), select(side.lessThan(1.5), mix(away, rgb(0x1a1d24), 0.72), rgb(0x2b313c)));
  const walkway = step(1, floored(row, TIER));
  const aisle = step(1, floored(seat, AISLE)).mul(step(0.6, abs(standQ.x.sub(standQ.y)))).mul(walkway);
  const occupied = step(draw.x, 0.72).mul(aisle);
  const chair = coverage(sdBox(cell.sub(vec2(0, ROW * 0.08)), vec2(SEAT * 0.4, ROW * 0.26), 0.08));
  const body = coverage(sdBox(cell, vec2(SEAT * 0.34, ROW * 0.26), 0.12));
  const head = coverage(length(cell.sub(vec2(0, ROW * 0.04))).sub(0.12));
  const hair = select(fan.x.lessThan(0.4), rgb(0x1c1512), select(fan.x.lessThan(0.75), rgb(0x4a3322), rgb(0xb89a6a)));
  const riser = smoothstep(0.05, 0.25, fract(depth.div(ROW)));
  let crowd: N<'vec3'> = mix(rgb(0x14181f), rgb(0x232a35), riser).mul(aisle.mul(0.4).add(0.6));
  crowd = mix(crowd, plastic, chair.mul(aisle).mul(0.9));
  crowd = mix(crowd, mix(shirt, hair, head), body.mul(occupied));
  crowd = crowd.mul(smoothstep(0, ROOF, depth).mul(-0.4).add(1));
  const roofed = step(ROOF, depth);
  const panels = coverage(abs(fract(along.div(4)).sub(0.5)).mul(4).sub(0.04)).max(coverage(abs(fract(depth.sub(ROOF).div(3)).sub(0.5)).mul(3).sub(0.03)));
  const roof = mix(rgb(0x1d222a), rgb(0x2a303a), fractal(p.mul(0.08), 2).mul(0.5).add(0.5)).mul(panels.mul(-0.35).add(1));
  const gantry = coverage(abs(depth.sub(ROOF + 0.35)).sub(0.3));
  const lamps = coverage(sdBox(vec2(fract(along.div(3.2)).sub(0.5).mul(3.2), depth.sub(ROOF + 0.35)), vec2(0.45, 0.12), 0.05)).mul(gantry);
  const flashes = step(float(0.994).sub(levels.cheer.mul(0.05)), grain(vec2(seat.add(floor(levels.time.mul(13)).mul(17)), row.add(side.mul(53)))).x);
  const flash = flashes.mul(levels.cheer).mul(select(side.lessThan(1.5), step(0, p.x.mul(levels.cheerSide)), float(0.4))).mul(occupied).mul(stand).mul(roofed.oneMinus());

  const apron = mix(rgb(0x17211c), rgb(0x1d2a23), fractal(p.mul(0.3), 2).mul(0.5).add(0.5));
  const wall = coverage(abs(front).sub(0.18));

  let albedo: N<'vec3'> = grass;
  albedo = mix(albedo, rgb(LINE_WHITE).mul(0.92), coverage(lines));
  albedo = mix(albedo, mix(albedo, rgb(0xdfe6ea), 0.28), goal.net);
  albedo = mix(albedo, rgb(0x0b0f0c), goal.shade);
  albedo = mix(albedo, rgb(0xf4f6f4), goal.cords.mul(0.55));
  albedo = mix(albedo, rgb(0xc9d0d4), goal.stanchions);
  albedo = mix(albedo, apron, outside);
  albedo = mix(albedo, mix(crowd.mul(0.3), roof.mul(0.7), roofed), stand);
  albedo = mix(albedo, rgb(0x0a0c10), gantry.mul(stand));
  albedo = mix(albedo, rgb(0x0d1014), wall.mul(outside));
  let color = lit(albedo);
  color = mix(color, rgb(0xffffff), goal.bar.mul(0.95));
  color = mix(color, rgb(0xffffff), goal.posts);
  color = mix(color, rgb(FLAG), coverage(pennant.sub(0.02)));
  color = mix(color, rgb(0x1b1d22), coverage(pole));
  color = mix(color, ledColor(p, d, levels).mul(1.05), board);
  color = mix(color, rgb(0xffffff).mul(0.9), lip.mul(board).mul(0.35));
  color = mix(color, rgb(0xfff6e2).mul(1.05), lamps.mul(stand));
  color = color.add(rgb(0xffe6b8).mul(lights.glow).mul(0.25));
  color = mix(color, rgb(0x262b33), lights.frame);
  color = mix(color, rgb(0xfff3dc).mul(1.8), lights.lamps);
  material.colorNode = color.add(rgb(0xfff4dc).mul(flash.mul(0.9)));
  return material;
}

export interface RestartLevels { mode: Level; side: Level; age: Level; ready: Level; spot: THREE.UniformNode<'vec2', THREE.Vector2>; tint: Tint }

export function restartLevels(): RestartLevels {
  return { mode: uniform(0), side: uniform(1), age: uniform(99), ready: uniform(0), spot: uniform(new THREE.Vector2()), tint: uniform(new THREE.Color()) };
}

export function restartMaterial(d: Dims, levels: RestartLevels, width: number, depth: number): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const L = d.length, W = d.width;
  const p = uv().sub(0.5).mul(vec2(width, depth)), px = fwidth(p.x), bold = max(px.mul(1.4), 0.22), fine = max(px.mul(0.8), 0.11);
  const is = (mode: number) => step(abs(levels.mode.sub(mode)), 0.5);
  const spot = levels.spot as unknown as N<'vec2'>;
  const toSpot = p.sub(spot), reach = length(toSpot);
  const pitch = sdBox(p, vec2(L, W));
  const pulse = coverage(abs(reach.sub(levels.age.mul(16).add(0.4))).sub(max(px.mul(0.8), 0.14))).mul(clamp(float(1).sub(levels.age.div(0.9)), 0, 1).pow(2)).mul(0.9);
  const dash = step(0.42, fract(atan(toSpot.y, toSpot.x).div(2 * Math.PI).mul(46)));
  const spray = coverage(abs(reach.sub(d.circle)).sub(fine)).mul(dash).mul(step(pitch, 0.05)).mul(is(3).add(is(4))).mul(smoothstep(0, 0.35, levels.age)).mul(0.85);
  const centre = coverage(abs(length(p).sub(d.circle - LINE / 2)).sub(bold)).mul(is(1));
  const q = vec2(float(L).sub(abs(p.x)), p.y);
  const box = sdBox(q.sub(vec2(d.box / 2, 0)), vec2(d.box / 2, d.boxWidth));
  const own = step(0, p.x.mul(levels.side).negate()), theirs = step(0, p.x.mul(levels.side));
  const boxGlow = coverage(abs(box.add(LINE / 2)).sub(bold)).mul(is(2)).mul(own);
  const mark = length(q.sub(vec2(d.mark, 0)));
  const penalty = max(coverage(mark.sub(0.45)), coverage(max(abs(mark.sub(d.circle - LINE / 2)).sub(bold), float(d.box).sub(q.x)))).mul(is(6)).mul(theirs);
  const turn = fract(atan(toSpot.x, toSpot.y).div(2 * Math.PI).add(1));
  const timer = coverage(abs(reach.sub(TIMER)).sub(fine)).mul(step(turn, levels.ready)).mul(step(0.001, levels.ready));
  const glow = tint(levels.tint);
  let layer: Layer = { color: rgb(0xf5f8ff).mul(spray), alpha: spray };
  layer = over(layer, glow.mul(1.3), min(centre.mul(0.55).add(boxGlow.mul(0.7)).add(penalty.mul(0.7)), 1));
  layer = over(layer, glow.mul(1.6), timer.mul(0.95));
  layer = over(layer, vec3(1.4), pulse);
  return finish(material, layer);
}
