import * as THREE from 'three/webgpu';
import { abs, atan, attribute, ceil, clamp, cos, dot, exp, float, floor, fract, int, length, max, min, mix, mod, normalize, positionWorld, pow, select, sin, smoothstep, step, uniformArray, uv, vec2, vec3 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { coverage, grain, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';

export type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Scalar = THREE.UniformNode<'float', number>;

export const GOLD = TOKENS.celebrationGold;
export const CUE = TOKENS.cueCyan;
export const HEAT = 0xffa24a;
export const RIME = 0xdff4ff;
export const FLAME = 0xff8d3f;

export const SLOT = 0.26;
export const SPAN = 12;
export const PIP_RING = 0.3;
const FIGURE = 0.62;
export const HANG = new THREE.Vector2(0.16, 0.2);

const TAU = Math.PI * 2;
const LIGHT = vec2(-0.6, 0.8);
const NEIGHBOURS = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]] as const;

export interface Layout {
  cells: number;
  centre: number;
  reach: number;
  end: number;
  ledge: number;
  floorN: number;
  floorS: number;
  rimN: number;
  rimS: number;
  fireY: number;
  lanternY: number;
  railY: number;
  cellX(cell: number): number;
}

export function layout(cells: number, centre: number): Layout {
  const offset = (cells - 1) / 2;
  return {
    cells, centre, reach: (cells - 2) / 2, end: cells / 2, ledge: cells / 2 + 0.15,
    floorN: 0.72, floorS: -0.5, rimN: 1.1, rimS: -0.88, fireY: 0.5, lanternY: 1.62, railY: 2.36,
    cellX: (cell) => cell - offset,
  };
}

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const board = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const local = (size: number): N<'vec2'> => uv().sub(0.5).mul(size);
const facing = (p: N<'vec2'>): N<'float'> => dot(p.div(max(length(p), 1e-4)), normalize(LIGHT));
const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));

function sheet(additive = false): THREE.MeshBasicNodeMaterial {
  return new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending });
}

function mesh(geometry: THREE.BufferGeometry, material: THREE.Material, order: number): THREE.Mesh {
  const out = new THREE.Mesh(geometry, material);
  out.renderOrder = order;
  out.frustumCulled = false;
  return out;
}

function cracks(p: N<'vec2'>): N<'float'> {
  const cell = floor(p), f = p.sub(cell);
  let near: N<'float'> = float(9), second: N<'float'> = float(9);
  for (const [dx, dy] of NEIGHBOURS) {
    const o = vec2(dx, dy), r = length(o.add(grain(cell.add(o)).mul(0.8).add(0.1)).sub(f));
    second = min(max(near, r), second);
    near = min(near, r);
  }
  return second.sub(near);
}

export function slots(count: number) {
  const values = new Array<number>(count).fill(0);
  return { node: uniformArray(values, 'float' as const), values };
}
export type Slots = ReturnType<typeof slots>;

function basinDistance(L: Layout, p: N<'vec2'>): N<'float'> {
  return sdBox(p.sub(vec2(0, (L.floorN + L.floorS) / 2)), vec2(L.ledge, (L.floorN - L.floorS) / 2), 0.36);
}

function ledgeDistance(L: Layout, p: N<'vec2'>): N<'float'> {
  return max(basinDistance(L, p).add(noise(p.mul(6.1)).mul(0.012)), float(L.reach).sub(abs(p.x)));
}

export function ground(L: Layout, lit: Lit, seed: number, time: Scalar): THREE.Mesh {
  const q = board(), a = vec2(abs(q.x), q.y), side = select(q.x.lessThan(0), float(-1), float(1));
  const o = vec2((seed % 211) * 0.731 + 3.1, (seed % 197) * 0.419 + 1.7);
  const broad = noise(q.mul(0.42).add(o)).mul(0.5).add(0.5);
  const ripple = noise(q.mul(vec2(0.8, 3.4)).add(o.mul(1.3)));
  const fine = noise(q.mul(8.3).add(o.mul(0.7)));
  const crest = pow(abs(noise(q.mul(vec2(0.55, 2.1)).add(o.mul(0.9)))).oneMinus(), 7);
  const lee = pow(abs(noise(q.add(vec2(0, 0.05)).mul(vec2(0.55, 2.1)).add(o.mul(0.9)))).oneMinus(), 7);
  const snow: N<'vec3'> = mix(rgb(0xc6d1e0), rgb(0xf0f4f9), smoothstep(0.15, 0.85, broad)).mul(ripple.mul(0.05).add(fine.mul(0.03)).add(1)).mul(crest.mul(0.07).sub(lee.mul(0.05)).add(1));

  const mid = (L.floorN + L.floorS) / 2, bank = L.rimN - L.floorN;
  const basin = basinDistance(L, q), up = max(basin, 0), rise = smoothstep(0, bank, up);
  const outward = normalize(vec2(select(a.x.greaterThan(L.ledge - 0.36), q.x.sub(side.mul(L.ledge - 0.36)), float(0)), q.y.sub(mid)).add(vec2(0, 1e-4)));
  const lean = rise.mul(rise.oneMinus()).mul(4);
  const shade = lean.mul(dot(outward, normalize(LIGHT)).mul(-0.28).add(-0.06)).add(1);
  const lip = exp(abs(up.sub(bank)).div(-0.035)).mul(0.1);
  const nook = exp(min(basin, 0).div(0.07)).mul(0.14).mul(step(basin, 0));
  const hollow = snow.mul(vec3(0.8, 0.85, 0.93)).mul(noise(q.mul(vec2(5.5, 1.2)).add(o)).mul(0.035).add(1));
  let color: N<'vec3'> = mix(hollow, snow, rise).mul(shade).mul(lip.add(1)).mul(nook.oneMinus());

  const trodden = smoothstep(0.26, 0.1, abs(q.y)).mul(coverage(a.x.sub(L.reach)));
  color = color.mul(trodden.mul(noise(q.mul(vec2(11, 7)).add(o)).mul(0.05).add(-0.07)).add(1));
  const ledgeD = ledgeDistance(L, q), cast = ledgeDistance(L, q.add(vec2(-0.05, 0.06)));
  const shelter = exp(min(a.x.sub(L.reach), 0).div(0.1)).mul(step(a.x, L.reach)).mul(step(basin, 0)).mul(0.32);
  color = color.mul(shelter.oneMinus()).mul(smoothstep(-0.02, 0.12, cast).mul(0.25).add(0.75));

  const thawAt = vec2(L.cellX(L.centre), 0.11), tp = q.sub(thawAt);
  const thawD = sdBox(tp, vec2(0.43, 0.57), 0.24).add(noise(q.mul(9.1).add(o)).mul(0.028));
  const earth = mix(rgb(0x2b2019), rgb(0x4d3a2c), smoothstep(-0.6, 0.6, noise(q.mul(7.5).add(o.mul(1.1)))));
  const tuft = smoothstep(0.3, 0.62, noise(q.mul(21).add(o))).mul(smoothstep(-0.2, 0.3, noise(q.mul(4.3).add(o.mul(0.4)))));
  const pebble = coverage(length(fract(q.mul(8.5).add(o)).sub(0.5)).sub(0.12)).mul(step(0.82, grain(floor(q.mul(8.5).add(o))).x));
  let thaw: N<'vec3'> = mix(earth, rgb(0x6f6b3d), tuft.mul(0.55));
  thaw = mix(thaw, rgb(0x8a8680), pebble.mul(0.7));
  thaw = mix(thaw, rgb(0x55616e).mul(noise(q.mul(15)).mul(0.1).add(1)), exp(min(thawD, 0).div(0.04)).mul(0.75));
  const slush = exp(max(thawD, 0).div(-0.06)).mul(0.22);
  color = mix(color.mul(vec3(0.9, 0.92, 0.96).mul(slush).add(slush.oneMinus())), thaw, coverage(thawD));

  const offset = (L.cells - 1) / 2;
  const cell = clamp(floor(q.x.add(offset + 0.5)), 1, L.cells - 2), sp = q.sub(vec2(cell.sub(offset), 0));
  const around = normalize(sp.add(vec2(1e-4, 0)));
  const stoneD = length(sp.mul(vec2(1, 1.12))).sub(0.168).add(noise(around.mul(1.2).add(vec2(cell.mul(3.3), 1.7))).mul(0.018));
  const onStone = step(0.5, abs(cell.sub(L.centre))).mul(step(abs(q.y), 0.4));
  const edge = smoothstep(0.09, 0.17, length(sp));
  const slate = mix(rgb(0x6f6963), rgb(0x9c948b), noise(q.mul(6.3).add(o.mul(2))).mul(0.5).add(0.5)).mul(facing(sp).mul(edge).mul(0.32).add(1)).mul(edge.mul(-0.12).add(1));
  const frosting = smoothstep(0.2, 0.55, facing(sp).add(noise(q.mul(19).add(o)).mul(0.35))).mul(smoothstep(0.05, 0.14, length(sp))).mul(0.9);
  const stone = mix(slate, rgb(0xe6eef6), frosting);
  color = color.mul(exp(max(stoneD, 0).div(-0.03)).mul(0.22).mul(onStone).oneMinus());
  color = mix(color, stone, coverage(stoneD).mul(onStone));

  const rockTone = mix(rgb(0x847d76), rgb(0xbab2a8), smoothstep(-0.7, 0.7, noise(q.mul(1.9).add(o)).add(noise(q.mul(5.3).add(o.mul(1.4))).mul(0.4))));
  const slabs = cracks(q.mul(2.4).add(o)), broken = smoothstep(0.1, 0.4, noise(q.mul(1.7).add(o.mul(0.8))));
  let rock: N<'vec3'> = rockTone.mul(noise(q.mul(17).add(o.mul(1.9))).mul(0.08).add(1)).mul(smoothstep(0, 0.03, slabs).oneMinus().mul(broken).mul(-0.45).add(1));
  rock = mix(rock, rgb(0x8a9566), smoothstep(0.55, 0.8, noise(q.mul(6.1).add(o.mul(2.3)))).mul(0.3));
  const driftField = noise(q.mul(3.1).add(o.mul(0.6))).mul(0.5).add(ledgeD.mul(6).add(0.35));
  rock = mix(rock, snow, smoothstep(0.05, 0.3, driftField).mul(0.9));
  rock = mix(rock, rgb(0xf4f8fc), coverage(abs(ledgeD.add(0.028)).sub(0.016)).mul(0.75));
  rock = mix(rock, rgb(0x262a31), coverage(abs(a.x.sub(L.reach + 0.028)).sub(0.022)).mul(step(ledgeD, 0)).mul(0.7));

  const refuge = L.end - 0.5;
  let wallD: N<'float'> = float(9), wallP: N<'vec2'> = vec2(9, 9);
  for (let k = 0; k < 6; k++) {
    const angle = ((-100 + 40 * k) * Math.PI) / 180, at = vec2(refuge + 0.41 * Math.cos(angle), 0.41 * Math.sin(angle));
    const pl = a.sub(at), d = length(pl).sub(0.066 + 0.012 * ((k * 5) % 3)).add(noise(pl.mul(19).add(k * 7.1)).mul(0.01));
    wallP = select(d.lessThan(wallD), pl, wallP);
    wallD = min(wallD, d);
  }
  const wp = vec2(wallP.x.mul(side), wallP.y), wl = facing(wp);
  const boulder = mix(rgb(0x625c56), rgb(0xaaa299), wl.mul(0.5).add(0.5)).mul(noise(q.mul(24)).mul(0.08).add(1));
  const topSnow = smoothstep(0.3, 0.6, wl.add(noise(q.mul(31)).mul(0.25))).mul(smoothstep(0.02, 0.05, length(wallP)));
  rock = rock.mul(smoothstep(-0.01, 0.07, wallD).mul(0.35).add(0.65));
  rock = mix(rock, mix(boulder, rgb(0xeef4fa), topSnow), coverage(wallD));

  color = mix(color, rock, coverage(ledgeD));

  const at = q.div(0.075).add(o), id = floor(at), h = grain(id);
  const flake = coverage(length(fract(at).sub(h.mul(0.6).add(0.2))).mul(0.075).sub(0.0045)).mul(step(h.x, 0.035));
  const twinkle = pow(max(sin(time.mul(2.1).add(h.y.mul(300))), 0), 10);
  const bare = coverage(thawD).add(coverage(stoneD).mul(onStone)).add(coverage(ledgeD).mul(smoothstep(0.05, 0.3, driftField).oneMinus()));
  const glint = flake.mul(twinkle).mul(clamp(bare.oneMinus(), 0, 1)).mul(0.85);

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(color).add(rgb(0xe9f4ff).mul(glint));
  return mesh(flat(80, 80), material, 0);
}

export interface Hearth { base: THREE.Mesh; flames: THREE.Mesh }

export function hearth(lit: Lit, level: Scalar, time: Scalar): Hearth {
  const p = local(0.5), r = length(p);
  let stones: N<'float'> = float(9), stoneP: N<'vec2'> = vec2(9, 9);
  for (let k = 0; k < 9; k++) {
    const angle = (TAU * k) / 9 + 0.3, at = vec2(0.132 * Math.cos(angle), 0.132 * Math.sin(angle)), pl = p.sub(at);
    const d = length(pl.mul(vec2(1, 1.12))).sub(0.03 + 0.006 * ((k * 7) % 3));
    stoneP = select(d.lessThan(stones), pl, stoneP);
    stones = min(stones, d);
  }
  const shadow = smoothstep(0.1, 0.2, length(p.sub(vec2(0.02, -0.025)))).oneMinus().mul(0.45);
  const bed = sdCircle(p, 0.108);
  const logA = sdSegment(p, vec2(-0.088, -0.052), vec2(0.084, 0.056), 0.021);
  const logB = sdSegment(p, vec2(-0.078, 0.062), vec2(0.088, -0.046), 0.019);
  const logs = min(logA, logB);
  const rockTone = mix(rgb(0x4d535c), rgb(0x8a909a), facing(stoneP).mul(0.5).add(0.5));
  let albedo: N<'vec3'> = mix(rgb(0x221d1a), rgb(0x3a322d), noise(p.mul(40)).mul(0.5).add(0.5));
  albedo = mix(albedo, rgb(0x3b2b21).mul(noise(p.mul(vec2(60, 12))).mul(0.2).add(0.9)), coverage(logs));
  albedo = mix(albedo, rockTone.mul(noise(p.mul(70)).mul(0.1).add(1)), coverage(stones));
  const glowField = noise(p.mul(34).add(vec2(time.mul(0.9), time.mul(-0.6)))).mul(0.5).add(0.5);
  const embers = smoothstep(0.45, 0.85, glowField).mul(coverage(bed)).mul(coverage(stones.negate())).mul(smoothstep(0.108, 0.02, r));
  const hot = rgb(0xff6a24).mul(embers.mul(level).mul(1.9)).add(rgb(0xff8d3f).mul(coverage(abs(logs).sub(0.004)).mul(smoothstep(0.09, 0.02, r)).mul(level).mul(0.9)));
  const solid = max(max(coverage(bed), coverage(logs)), coverage(stones));
  const base = sheet();
  base.colorNode = mix(rgb(0x000000), lit(albedo).add(hot), solid);
  base.opacityNode = max(solid, shadow);

  const f = local(0.72), fr = length(f), angle = atan(f.y, f.x), dir = vec2(cos(angle), sin(angle));
  const lick = noise(dir.mul(2.3).add(vec2(time.mul(1.7), time.mul(-2.3)))).mul(0.5).add(0.5);
  const lobes = sin(angle.mul(7).add(time.mul(4.1))).mul(0.5).add(0.5).mul(sin(angle.mul(4).sub(time.mul(2.9))).mul(0.5).add(0.5));
  const reach = level.mul(lick.mul(0.05).add(lobes.mul(0.03)).add(0.035));
  const u = fr.div(reach);
  const body = smoothstep(1, 0.5, u);
  const tone = mix(mix(rgb(0xff3a0e), rgb(0xff9a30), smoothstep(1, 0.6, u)), rgb(0xffe6a0), smoothstep(0.5, 0.12, u));
  const core = smoothstep(0.04, 0.0, fr).mul(level.min(1.4));
  let sparks: N<'float'> = float(0);
  for (let k = 0; k < 7; k++) {
    const heading = (TAU * k) / 7 + 0.9 * k, age = fract(time.mul(0.55 + 0.11 * k).add(k * 0.37));
    const at = vec2(Math.cos(heading), Math.sin(heading)).mul(age.mul(0.26).add(0.03));
    sparks = sparks.add(coverage(length(f.sub(at)).sub(0.006)).mul(age.oneMinus().mul(age.oneMinus())));
  }
  const flame = sheet(true);
  flame.colorNode = tone.mul(body.mul(2.3)).add(rgb(0xfff4dc).mul(core.mul(2))).add(rgb(0xffb35e).mul(sparks.mul(level).mul(3)))
    .add(rgb(0xff8d3f).mul(exp(fr.div(-0.1)).mul(0.28).mul(level)));
  flame.opacityNode = smoothstep(0.36, 0.26, fr);
  return { base: mesh(flat(0.5, 0.5), base, 3), flames: mesh(flat(0.72, 0.72), flame, 8) };
}

export interface Walker { figure: THREE.Mesh; sockets: THREE.Mesh; embers: THREE.Mesh; pips: Slots }

export interface WalkerLevels { turn: Scalar; rime: Scalar; fade: Scalar; reach: Scalar; gold: Scalar; ice: Scalar; thaw: Scalar }

export function walker(accent: number, lit: Lit, levels: WalkerLevels, count: number): Walker {
  const { turn, rime, fade, reach, gold, ice, thaw } = levels;
  const p = local(FIGURE), c = cos(turn), s = sin(turn);
  const q = vec2(p.x.mul(c).add(p.y.mul(s)), p.y.mul(c).sub(p.x.mul(s)));
  const shadow = smoothstep(-0.04, 0.05, ellipse(p, [0.03, -0.045], [0.21, 0.16])).oneMinus().mul(0.36);
  const body = ellipse(q, [0, -0.012], [0.176, 0.126]);
  const roll = sdSegment(q, vec2(-0.112, -0.104), vec2(0.112, -0.104), 0.037);
  const hand = reach.mul(0.075);
  const armL = sdSegment(q, vec2(-0.148, 0.0), vec2(-0.104, hand.add(0.078)), 0.041);
  const armR = sdSegment(q, vec2(0.148, 0.0), vec2(0.104, hand.add(0.078)), 0.041);
  const mitL = sdCircle(q.sub(vec2(-0.1, hand.add(0.094))), 0.034), mitR = sdCircle(q.sub(vec2(0.1, hand.add(0.094))), 0.034);
  const hood = sdCircle(q.sub(vec2(0, 0.012)), 0.098);
  const ruffAt = q.sub(vec2(0, 0.05)), ruffAngle = atan(ruffAt.y, ruffAt.x);
  const ruff = length(ruffAt).sub(noise(vec2(cos(ruffAngle), sin(ruffAngle)).mul(3.2)).mul(0.006).add(sin(ruffAngle.mul(27)).mul(0.0022)).add(0.069));
  const face = ellipse(q, [0, 0.056], [0.042, 0.036]);
  const silhouette = min(min(min(body, roll), min(armL, armR)), min(min(mitL, mitR), min(hood, ruff)));

  const parka = mix(rgb(accent), rgb(0x1f5fb8), 0.6);
  const jacket = parka.mul(0.8).mul(q.x.mul(0.7).add(q.y.mul(0.9)).add(1));
  let color: N<'vec3'> = rgb(0x121417);
  color = mix(color, mix(rgb(0x8d3b2e), rgb(0x6e2c23), smoothstep(0.4, 0.6, fract(q.x.mul(16)))), coverage(roll));
  color = mix(color, jacket, coverage(body));
  color = mix(color, parka.mul(0.6), coverage(min(armL, armR)));
  color = mix(color, rgb(0x4a2e28), coverage(min(mitL, mitR)));
  color = mix(color, parka.mul(0.55).mul(q.y.mul(1.6).add(1)), coverage(hood));
  color = mix(color, rgb(0xf0e4cf).mul(facing(ruffAt).mul(0.18).add(0.92)), coverage(ruff));
  color = mix(color, rgb(0x2b1f1b), coverage(face));
  const frosted = mix(color, rgb(RIME), rime.mul(0.55).mul(noise(q.mul(40)).mul(0.3).add(0.7)));
  const warm = rgb(0xffb070).mul(thaw.mul(0.12));
  const ink = coverage(silhouette.sub(0.013)), solid = coverage(silhouette);
  const tone = mix(rgb(0x0d0f12), lit(frosted).add(warm), solid);
  const alpha = max(ink, shadow);
  const figure = sheet();
  figure.colorNode = tone.mul(ink).div(max(alpha, 1e-4));
  figure.opacityNode = alpha.mul(fade);

  const pips = slots(count);
  const r = local(0.8), pa = atan(r.y, r.x), turnOf = mod(float(Math.PI / 2).sub(pa), TAU);
  const k = min(floor(turnOf.div(TAU / count)), count - 1), centre = float(Math.PI / 2).sub(k.add(0.5).mul(TAU / count));
  const pd = length(r.sub(vec2(cos(centre), sin(centre)).mul(PIP_RING)));
  const on = pips.node.element(int(k));
  const socketTone = mix(rgb(0x191412), rgb(0x3a2e27), smoothstep(0.034, 0.0, pd));
  const socket = sheet();
  socket.colorNode = socketTone;
  socket.opacityNode = max(coverage(abs(pd.sub(0.03)).sub(0.007)).mul(0.85), coverage(pd.sub(0.03)).mul(0.5)).mul(fade);

  const lit01 = on.mul(ice.oneMinus().mul(0.35).add(0.65));
  const emberTone = mix(mix(rgb(0xff6a22), rgb(0xffcf86), on), rgb(0xe8f7ff), ice);
  const glow = sheet(true);
  glow.colorNode = emberTone.mul(coverage(pd.sub(0.026)).mul(2.6).add(exp(pd.div(-0.022)).mul(0.9))).mul(lit01)
    .add(rgb(GOLD).mul(gold).mul(coverage(pd.sub(0.03)).mul(2.2).add(exp(pd.div(-0.035)).mul(0.8))));
  glow.opacityNode = fade.mul(smoothstep(0.4, 0.35, length(r)));

  return { figure: mesh(flat(FIGURE, FIGURE), figure, 8), sockets: mesh(flat(0.8, 0.8), socket, 5), embers: mesh(flat(0.8, 0.8), glow, 9), pips };
}

export function mist(L: Layout, fill: Scalar, thin: Scalar, time: Scalar): THREE.Mesh {
  const q = board(), mid = (L.floorN + L.floorS) / 2, bank = L.rimN - L.floorN;
  const across = coverage(abs(q.x).sub(L.reach));
  const up = max(max(q.y.sub(L.floorN), float(L.floorS).sub(q.y)), 0);
  const pool = smoothstep(0, bank * 0.9, up).oneMinus();
  const inward = select(q.y.greaterThan(mid), q.y.sub(mid).div(L.rimN - mid), q.y.sub(mid).div(L.rimS - mid)).oneMinus();
  const drift = vec2(time.mul(0.16), time.mul(-0.03));
  const billow = noise(q.mul(vec2(1.3, 2.6)).add(drift)).mul(0.5).add(noise(q.mul(vec2(3.7, 6.1)).sub(drift.mul(1.7))).mul(0.25)).add(0.5);
  const streak = noise(q.mul(vec2(0.9, 9)).add(drift.mul(vec2(2.5, 0)))).mul(0.5).add(0.5);
  const front = fill.mul(1.7).sub(0.35).sub(inward.mul(0.55)).add(billow.sub(0.5).mul(0.7));
  const reached = smoothstep(0, 0.16, front);
  const wall = exp(min(abs(q.x).sub(L.reach), 0).div(0.1)).mul(0.22);
  const presence = reached.mul(pool).mul(thin.oneMinus()).mul(across);
  const density = presence.mul(billow.mul(0.26).add(streak.mul(0.16)).add(0.1).add(wall));
  const at = q.div(0.05).add(7.3), id = floor(at), h = grain(id);
  const crystal = coverage(length(fract(at).sub(h.mul(0.6).add(0.2))).mul(0.05).sub(0.004)).mul(step(h.x, 0.08));
  const sparkle = crystal.mul(pow(max(sin(time.mul(3.3).add(h.y.mul(400))), 0), 6)).mul(presence);
  const material = sheet();
  material.colorNode = mix(rgb(0x9fbddc), rgb(0xeef8ff), billow.mul(0.55).add(streak.mul(0.45))).add(rgb(0xffffff).mul(sparkle.mul(1.8)));
  material.opacityNode = clamp(density.add(sparkle.mul(0.8)), 0, 0.66);
  const span = L.rimN - L.rimS + 0.3;
  const out = mesh(flat(2 * L.reach + 0.02, span), material, 7);
  out.position.set(0, 0.01, -(L.rimN + L.rimS) / 2);
  return out;
}

export interface Lantern { post: THREE.Mesh; halo: THREE.Mesh }

export function lantern(lit: Lit, glow: Scalar, flash: Scalar, time: Scalar): Lantern {
  const p = local(1).add(vec2(0, 0.02));
  const shadow = smoothstep(-0.03, 0.05, ellipse(p, [0.05, -0.372], [0.12, 0.028])).oneMinus().mul(0.3);
  const mound = ellipse(p, [0, -0.366], [0.075, 0.03]);
  const post = sdBox(p.sub(vec2(0, -0.03)), vec2(0.021, 0.34), 0.008);
  const arc = abs(length(p.sub(vec2(0.08, 0.31))).sub(0.08)).sub(0.017);
  const crook = max(arc, p.y.sub(0.31).negate());
  const hook = sdSegment(p, vec2(HANG.x, 0.31), vec2(HANG.x, HANG.y + 0.075), 0.006);
  const cap = sdBox(p.sub(vec2(HANG.x, HANG.y + 0.062)), vec2(0.045, 0.013), 0.006);
  const glass = sdBox(p.sub(vec2(HANG.x, HANG.y)), vec2(0.036, 0.05), 0.008);
  const frame = sdBox(p.sub(vec2(HANG.x, HANG.y)), vec2(0.042, 0.056), 0.01);
  const foot = sdBox(p.sub(vec2(HANG.x, HANG.y - 0.064)), vec2(0.046, 0.011), 0.005);
  const wood = min(post, crook);
  const iron = min(min(hook, cap), min(frame, foot));
  const grainTone = noise(p.mul(vec2(90, 6))).mul(0.12).add(1);
  let albedo: N<'vec3'> = rgb(0x5b4331).mul(grainTone).mul(p.x.mul(-2.5).add(1));
  albedo = mix(albedo, rgb(0xeef4fa), coverage(abs(length(p.sub(vec2(0.08, 0.31))).sub(0.096)).sub(0.006)).mul(step(0.33, p.y)).mul(0.9));
  albedo = mix(albedo, rgb(0xe8eff7).mul(p.y.add(0.366).mul(3).add(0.92)), coverage(mound));
  albedo = mix(albedo, rgb(0x23252b).mul(facing(p.sub(vec2(HANG.x, HANG.y))).mul(0.3).add(1)), coverage(iron));
  const sheen = exp(length(p.sub(vec2(HANG.x - 0.014, HANG.y + 0.02))).mul(-60)).mul(0.5);
  const pane = mix(rgb(0x1a2733).add(sheen), rgb(CUE).mul(glow.mul(2.6).add(flash.mul(2))).add(rgb(0xe8fbff).mul(glow.mul(exp(length(p.sub(vec2(HANG.x, HANG.y))).mul(-40)).mul(2.2)))), glow);
  const solid = coverage(min(min(wood, iron), mound));
  const body = sheet();
  body.colorNode = mix(mix(rgb(0x0b0c0e), lit(albedo), solid), pane, coverage(glass));
  body.opacityNode = max(max(solid, coverage(glass)), shadow);

  const g = local(1.4), gr = length(g);
  const pulse = sin(time.mul(5.2)).mul(0.08).add(0.92);
  const ring = coverage(abs(gr.sub(flash.oneMinus().mul(0.55).add(0.06))).sub(0.01)).mul(flash);
  const halo = sheet(true);
  halo.colorNode = rgb(CUE).mul(exp(gr.div(-0.09)).mul(0.9).add(exp(gr.div(-0.28)).mul(0.25)).mul(glow).mul(pulse).add(ring.mul(1.4)));
  halo.opacityNode = smoothstep(0.7, 0.55, gr);
  return { post: mesh(flat(1, 1), body, 10), halo: mesh(flat(1.4, 1.4), halo, 11) };
}

export interface RailData { frost: Slots; token: Slots; edge: Slots; scroll: Scalar; caret: Scalar; caretLit: Scalar; threshold: number }

export function rail(lit: Lit, data: RailData): THREE.Mesh {
  const { frost, token, edge, scroll, caret, caretLit, threshold } = data;
  const half = SPAN * SLOT, last = 2 * SPAN + 1;
  const p = local(1).mul(vec2(2 * half + 0.6, 0.62));
  const u = p.x.div(SLOT).add(scroll), k = ceil(u), idx = clamp(k.add(SPAN), 0, last);
  const at = (values: Slots, shift: number) => values.node.element(int(clamp(idx.add(shift), 0, last)));
  const bar = (values: Slots, y: number, h: number, inset: number) => {
    const l = select(at(values, -1).greaterThan(0.5), float(0.5), float(-inset)), r = select(at(values, 1).greaterThan(0.5), float(0.5), float(-inset));
    const centre = k.mul(2).sub(1).sub(l).add(r).mul(0.5), width = r.add(l).add(1).mul(0.5);
    return sdBox(vec2(u.sub(centre).mul(SLOT), p.y.sub(y)), vec2(width.mul(SLOT), h), Math.min(h, 0.05));
  };
  const frostY = 0.08, tokenY = -0.075;
  const iceD = bar(frost, frostY, 0.075, 0.06), ice = coverage(iceD).mul(step(0.5, at(frost, 0)));
  const bell = bar(token, tokenY, 0.026, 0.1), cue = coverage(bell).mul(step(0.5, at(token, 0))).mul(coverage(p.x));
  const past = smoothstep(0.02, -0.08, p.x);
  const facets = smoothstep(0.012, 0.0, abs(fract(u.add(p.y.mul(3.1))).sub(0.5)).mul(SLOT)).mul(0.35).add(smoothstep(0.1, 0.0, abs(p.y.sub(frostY + 0.035))).mul(0.25));
  const iceTone = mix(rgb(0xcdeeff).mul(facets.add(1.1)), rgb(0x8ea9bf).mul(0.75), past.mul(0.65));
  const iceRim = coverage(abs(iceD.add(0.008)).sub(0.006)).mul(ice);

  const plankD = sdBox(p, vec2(half + 0.24, 0.17), 0.07);
  const plank = rgb(0x3a2b21).mul(noise(p.mul(vec2(3, 40))).mul(0.14).add(0.94)).mul(smoothstep(0.17, 0.08, abs(p.y)).mul(0.25).add(0.75));
  const seam = coverage(abs(p.y.sub(0.0)).sub(0.004)).mul(0.5);
  const tick = coverage(float(SLOT / 2 - 0.006).sub(abs(fract(u).sub(0.5).mul(SLOT)))).mul(coverage(abs(p.y.add(0.005)).sub(0.03))).mul(0.6);
  const reachX = (threshold - 1) * SLOT;
  const zone = coverage(p.x.negate()).mul(coverage(p.x.sub(reachX))).mul(coverage(abs(p.y.sub(frostY - 0.092)).sub(0.007))).add(coverage(abs(p.x.sub(reachX)).sub(0.006)).mul(coverage(abs(p.y.sub(frostY - 0.07)).sub(0.03))));
  const nowLine = coverage(abs(p.x).sub(0.008)).mul(coverage(abs(p.y.sub(0.01)).sub(0.16)));
  const nibAt = p.sub(vec2(0, -0.222)), nib = coverage(max(abs(nibAt.x).mul(0.9).sub(nibAt.y.mul(0.5)), nibAt.y.sub(0.06)));
  const cx = clamp(caret, -half + 0.05, half - 0.05), over = step(half - 0.05, caret).add(step(caret, -half + 0.05));
  const tip = p.sub(vec2(cx, frostY + 0.135)), chevron = max(tip.y.sub(0.035), abs(tip.x).sub(tip.y.add(0.045).mul(0.66)));
  const stem = coverage(abs(p.x.sub(cx)).sub(0.006)).mul(coverage(abs(p.y.sub(frostY + 0.02)).sub(0.095)));
  const caretTone = mix(rgb(0xffe7c2), rgb(CUE), caretLit);
  const dash = step(0.5, fract(p.y.mul(18))).mul(coverage(abs(p.y).sub(0.16)));
  const edgeLine = coverage(k.sub(u).mul(SLOT).sub(0.007)).mul(step(0.5, at(edge, 0))).add(coverage(u.sub(k).add(1).mul(SLOT).sub(0.007)).mul(step(0.5, at(edge, -1)))).mul(dash);
  const fade = smoothstep(half + 0.2, half - 0.5, abs(p.x));

  let color: N<'vec3'> = lit(plank.mul(seam.oneMinus()));
  color = mix(color, rgb(0x8fa8bb), tick.mul(0.5));
  color = mix(color, rgb(CUE).mul(1.1), clamp(zone, 0, 1).mul(0.75));
  color = mix(color, rgb(0xd4c2a8), edgeLine.mul(0.8));
  color = mix(color, iceTone, ice.mul(fade.mul(0.6).add(0.4)));
  color = mix(color, rgb(0xffffff).mul(1.3), iceRim.mul(past.oneMinus()).mul(0.7));
  color = mix(color, rgb(CUE).mul(1.6), cue.mul(fade.mul(0.5).add(0.5)));
  color = mix(color, rgb(0xfff6e8).mul(1.5), max(nowLine, nib));
  color = mix(color, caretTone.mul(1.6), max(coverage(chevron), stem.mul(0.8)).mul(over.mul(-0.4).add(1)));
  const material = sheet();
  material.colorNode = color;
  material.opacityNode = max(max(coverage(plankD).mul(0.92), coverage(chevron)), stem.mul(0.8)).mul(fade.mul(0.85).add(0.15));
  return mesh(flat(2 * half + 0.6, 0.62), material, 12);
}

export function pines(seed: number, spots: readonly THREE.Vector3[]): THREE.InstancedMesh {
  const tree = attribute<'vec4'>('aTree', 'vec4');
  const p = local(2.8), r = length(p), angle = atan(p.y, p.x);
  const shadow = smoothstep(-0.15, 0.2, length(p.sub(vec2(0.24, -0.28))).sub(0.86)).oneMinus().mul(0.5);
  const greens = [0x10201a, 0x172b21, 0x1f3629, 0x284332, 0x33523d] as const;
  let color: N<'vec3'> = rgb(greens[0]);
  let crown: N<'float'> = float(0);
  for (let tier = 0; tier < 4; tier++) {
    const radius = 1 - 0.23 * tier, points = 17 - 3 * tier, twist = tree.x.mul(TAU).add(tier * 1.3);
    const jag = pow(abs(cos(angle.mul(points / 2).add(twist))), 1.6).mul(0.13);
    const wobble = noise(vec2(cos(angle), sin(angle)).mul(1.6).add(vec2(tree.x.mul(40), tier * 3.3))).mul(0.09);
    const reach = jag.add(0.84).mul(radius).add(wobble), d = r.sub(reach);
    const bough = mix(rgb(greens[tier]!), rgb(greens[tier + 1]!), smoothstep(reach, reach.mul(0.45), r));
    const patch = noise(p.mul(4.3).add(vec2(tree.x.mul(17), tier * 5.1)));
    const snowy = smoothstep(0.05, 0.3, patch.add(facing(p).mul(0.45)).sub(0.12)).mul(smoothstep(reach.mul(0.5), reach.mul(0.8), r)).mul(smoothstep(reach, reach.mul(0.9), r));
    color = mix(color, color.mul(0.5), smoothstep(0.16, 0, d).mul(coverage(d).oneMinus()).mul(crown).mul(0.7));
    color = mix(color, mix(bough, rgb(0xe3ecf5).mul(facing(p).mul(0.12).add(0.9)), snowy.mul(0.85)), coverage(d));
    crown = max(crown, coverage(d));
  }
  color = mix(color, rgb(0x3d5a44), coverage(r.sub(0.07)).mul(0.8));
  const material = sheet();
  material.colorNode = mix(rgb(0x05080a), color.mul(tree.y), crown);
  material.opacityNode = max(crown, shadow);
  const geometry = flat(1, 1);
  const state = new THREE.InstancedBufferAttribute(new Float32Array(spots.length * 4), 4);
  geometry.setAttribute('aTree', state);
  const out = instancedMesh(geometry, material, spots.length);
  const matrix = new THREE.Matrix4(), scale = new THREE.Matrix4();
  spots.forEach((spot, i) => {
    out.setMatrixAt(i, matrix.makeTranslation(spot.x, 0, spot.z).multiply(scale.makeScale(2.8 * spot.y, 1, 2.8 * spot.y)));
    state.setXYZW(i, ((seed * 0.618 + i * 0.37) % 1 + 1) % 1, 0.85 + 0.3 * (((seed + i * 7) % 5) / 5), 0, 0);
  });
  out.renderOrder = 9;
  out.frustumCulled = false;
  return out;
}

export function disc(radius: number): THREE.BufferGeometry {
  return new THREE.CircleGeometry(radius, 40).rotateX(-Math.PI / 2);
}

export function ring(inner: number, outer: number): THREE.BufferGeometry {
  return new THREE.RingGeometry(inner, outer, 48).rotateX(-Math.PI / 2);
}
