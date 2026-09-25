import * as THREE from 'three/webgpu';
import { abs, atan, clamp, cos, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, round, sin, smoothstep, step, vec2 } from 'three/tsl';
import { noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Level = THREE.UniformNode<'float', number>;
interface Layer { c: N<'vec3'>; a: N<'float'> }

export const DOOR = { x: 3.3, hw: 1.4, spring: 1, mid: -0.6 } as const;
export const FLOOR = -3;
export const GAUGE = { x: 0, y: 1.25, r: 1.1 } as const;
export const TORCH = { x: 5.75, y: 0.9 } as const;
export const COUNT = { y: -3.72, h: 0.62, pitch: 0.5 } as const;
export const doorX = (door: number) => (door === 0 ? -DOOR.x : DOOR.x);

export const LACQUER = 0x2c7a62;
export const GROWL = 0xff8a3a;
export const GOLD = 0xf2c14e;
export const CLAW = 0xd8262a;
export const CHALK = 0xe8e2d0;
const BRASS = 0xc9a24a;
const IRON = 0x2a2c30;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 48).rotateX(-Math.PI / 2);

export const RAISE = 1;
export const worldZ = (y: number) => -(y + RAISE);
const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate().sub(RAISE));
const cover = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(max(fwidth(positionWorld.x), fwidth(positionWorld.z)))), 0, 1);
const hash = (p: N<'vec2'>): N<'float'> => fract(sin(p.x.mul(127.1).add(p.y.mul(311.7))).mul(43758.5453));
const unit = (p: N<'vec2'>, scale: number): N<'float'> => noise(p.mul(scale)).mul(0.5).add(0.5);
const ellipse = (q: N<'vec2'>, a: number, b: number): N<'float'> => length(q.div(vec2(a, b))).sub(1).mul(Math.min(a, b));

function over(under: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer {
  const a = alpha.add(under.a.mul(alpha.oneMinus()));
  return { c: color.mul(alpha).add(under.c.mul(under.a).mul(alpha.oneMinus())).div(max(a, 1e-4)), a };
}

function sprite(width: number, depth: number, x: number, y: number, order: number, color: N<'vec3'>, opacity: N<'float'>): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  const mesh = new THREE.Mesh(flat(width, depth), material);
  mesh.position.set(x, 0, worldZ(y));
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

export function hall(lit: Lit, flick: Level): THREE.Mesh {
  const q = chart();
  const row = floor(q.y.div(0.55)), bx = q.x.div(1.1).add(row.mul(0.5));
  const cell = vec2(fract(bx), fract(q.y.div(0.55)));
  const edge = min(min(cell.x, cell.x.oneMinus()).mul(1.1), min(cell.y, cell.y.oneMinus()).mul(0.55));
  const grit = unit(q, 6);
  const block = mix(rgb(0x5a412d), rgb(0x8c6b48), hash(vec2(floor(bx), row)).mul(0.6).add(grit.mul(0.4)));
  const wall = mix(block, rgb(0x24180f), cover(edge.sub(0.03))).mul(smoothstep(FLOOR, FLOOR + 1.4, q.y).mul(0.4).add(0.6));
  const frow = floor(q.y.div(0.62)), fx = q.x.div(1.5).add(frow.mul(0.37));
  const fcell = vec2(fract(fx), fract(q.y.div(0.62)));
  const fedge = min(min(fcell.x, fcell.x.oneMinus()).mul(1.5), min(fcell.y, fcell.y.oneMinus()).mul(0.62));
  const stone = mix(rgb(0x33261b), rgb(0x514029), hash(vec2(floor(fx), frow)).mul(0.6).add(grit.mul(0.4)));
  const ground = mix(stone, rgb(0x140e09), cover(fedge.sub(0.022)));
  let color = mix(wall, ground, cover(q.y.sub(FLOOR)));
  color = mix(color, rgb(0x9a7a55), cover(abs(q.y.sub(FLOOR)).sub(0.05)));
  const t = vec2(abs(q.x).sub(TORCH.x), q.y.sub(TORCH.y));
  const dBracket = min(sdBox(t.sub(vec2(0, -0.5)), vec2(0.07, 0.42), 0.03), sdBox(t.sub(vec2(0, -0.02)), vec2(0.22, 0.08), 0.04));
  color = mix(color, rgb(IRON), cover(dBracket));
  color = lit(color.mul(float(1).sub(smoothstep(5, 9, length(q.mul(vec2(0.75, 1.1)))).mul(0.4))));
  const fy = t.y.sub(0.22), widen = clamp(fy, 0, 0.6).mul(float(2.2).add(flick.mul(0.3))).add(1);
  const dFlame = length(vec2(t.x.mul(widen), fy.mul(0.6))).sub(0.13);
  const flame = mix(rgb(0xffe0a0), rgb(0xff6a1a), smoothstep(0.02, 0.4, fy)).mul(0.95);
  color = mix(color, flame, cover(dFlame));
  return sprite(40, 30, 0, 0, 0, color, float(1));
}

export interface DoorLevels { open: Level; kind: Level; growl: Level; count: Level; slash: Level; time: Level }

function treasure(room: N<'vec3'>, p: N<'vec2'>, time: Level): N<'vec3'> {
  const dHeap = ellipse(vec2(p.x, p.y.sub(FLOOR)), 1.15, 0.85);
  const coins = unit(p, 7).mul(0.6).add(unit(p.add(4.1), 15).mul(0.4));
  const gold = mix(rgb(0x7a4c0e), rgb(GOLD), coins).mul(smoothstep(-0.1, 0.9, p.y.sub(FLOOR)).mul(0.45).add(0.55));
  const spark = smoothstep(0.86, 1, noise(p.mul(11).add(vec2(time.mul(0.7), 0)))).mul(0.8);
  const glow = rgb(0x8a5212).mul(exp(length(p.sub(vec2(0, -2.4))).mul(-0.95)));
  return mix(room.add(glow), gold.add(spark), cover(dHeap));
}

function tiger(room: N<'vec3'>, p: N<'vec2'>): N<'vec3'> {
  const t = p.sub(vec2(0, -1.05)), s = vec2(abs(t.x), t.y);
  const dBody = sdCircle(p.sub(vec2(0, FLOOR - 0.55)), 1.35);
  const dHead = min(min(sdCircle(t, 0.78), sdCircle(s.sub(vec2(0.4, -0.32)), 0.46)), sdCircle(s.sub(vec2(0.58, 0.6)), 0.25));
  const dStripe = max(min(min(sdSegment(s, vec2(0, 0.76), vec2(0, 0.46), 0.045), sdSegment(s, vec2(0.24, 0.73), vec2(0.17, 0.46), 0.04)), min(sdSegment(s, vec2(0.9, -0.12), vec2(0.58, -0.2), 0.042), sdSegment(s, vec2(0.86, -0.4), vec2(0.6, -0.4), 0.04))), dHead);
  const shade = smoothstep(-1, 0.8, t.y).mul(0.5).add(0.5);
  const warm = rgb(0x3a0c06).mul(exp(length(p.sub(vec2(0, -1.4))).mul(-0.9)));
  let c = mix(room.add(warm), mix(rgb(0x4a1c06), rgb(0x9a4610), smoothstep(-3.6, -2.2, p.y)), cover(dBody));
  c = mix(c, mix(rgb(0x8a3a0a), rgb(0xe07a1f), shade), cover(dHead));
  c = mix(c, rgb(0x2a1208), cover(sdCircle(s.sub(vec2(0.58, 0.6)), 0.12)));
  c = mix(c, rgb(0xefe0c4).mul(shade.mul(0.3).add(0.7)), cover(sdCircle(s.sub(vec2(0.17, -0.44)), 0.25)));
  c = mix(c, rgb(0x160b06), cover(dStripe));
  c = mix(c, rgb(0x4a2018), cover(sdBox(t.sub(vec2(0, -0.25)), vec2(0.1, 0.045), 0.04)));
  c = mix(c, rgb(0xf6d43a).mul(0.9), cover(ellipse(s.sub(vec2(0.3, 0.1)), 0.14, 0.085)));
  return mix(c, rgb(0x0e0604), cover(ellipse(s.sub(vec2(0.3, 0.1)), 0.028, 0.075)));
}

function leaf(p: N<'vec2'>, side: number, open: Level, dOpen: N<'float'>): N<'vec3'> {
  const w = open.mul(-0.8).add(1), uu = p.x.mul(-side).add(DOOR.hw).div(w).sub(DOOR.hw);
  const plank = fract(uu.add(DOOR.hw).div(0.7));
  let c = mix(rgb(0x164538), rgb(LACQUER), unit(vec2(uu.mul(2.5), p.y.mul(0.35)), 1));
  c = mix(c, rgb(0x0a1f19), cover(min(plank, plank.oneMinus()).mul(0.7).mul(w).sub(0.016)));
  for (const by of [-2.3, 0.2]) {
    c = mix(c, rgb(IRON), cover(abs(p.y.sub(by)).sub(0.11)));
    const stud = vec2(plank.sub(0.5).mul(0.7).mul(w), p.y.sub(by));
    c = mix(c, rgb(BRASS), cover(length(stud).sub(0.05)));
  }
  const ring = vec2(uu.sub(0.95).mul(w), p.y.add(0.9));
  c = mix(c, rgb(BRASS), cover(abs(length(ring).sub(0.17)).sub(0.03)));
  c = mix(c, rgb(0x0a1f19), cover(abs(dOpen).sub(0.03)));
  return c.mul(open.mul(-0.55).add(1)).mul(smoothstep(-DOOR.hw, DOOR.hw, uu).mul(0.25).add(0.75));
}

const SEGMENTS = [63, 6, 91, 79, 102, 109, 125, 7, 127, 111];

function tally(p: N<'vec2'>, count: Level): N<'float'> {
  const c = max(count, 0), digits = float(1).add(step(10, c)).add(step(100, c)).add(step(1000, c));
  const x = p.x.add(digits.mul(COUNT.pitch / 2)), j = floor(x.div(COUNT.pitch));
  const place = digits.sub(1).sub(j), digit = floor(c.add(0.5).div(float(10).pow(max(place, 0)))).mod(10);
  let mask: N<'float'> = float(0);
  SEGMENTS.forEach((bits, k) => (mask = mask.add(float(bits).mul(step(abs(digit.sub(k)), 0.5)))));
  const q0 = vec2(fract(x.div(COUNT.pitch)).sub(0.5).mul(COUNT.pitch), p.y.sub(COUNT.y));
  const q = vec2(q0.x.sub(q0.y.mul(0.16)), q0.y), w = COUNT.h * 0.27, h = COUNT.h / 2;
  const ends: [number, number, number, number][] = [[-w, h, w, h], [w, h, w, 0], [w, 0, w, -h], [-w, -h, w, -h], [-w, -h, -w, 0], [-w, 0, -w, h], [-w, 0, w, 0]];
  let d: N<'float'> = float(1e3);
  ends.forEach(([ax, ay, bx, by], s) => {
    const on = floor(mask.div(2 ** s)).mod(2);
    d = min(d, mix(float(1e3), sdSegment(q, vec2(ax * 0.86, ay * 0.9), vec2(bx * 0.86, by * 0.9), 0.034), on));
  });
  const inside = step(0, j).mul(step(j, digits.sub(0.5))).mul(step(0, count));
  return cover(d).mul(inside).mul(unit(p, 38).mul(0.4).add(0.6));
}

export function door(side: -1 | 1, v: DoorLevels, lit: Lit): THREE.Mesh {
  const p = chart().sub(vec2(side * DOOR.x, 0));
  const dOpen = min(sdBox(p.sub(vec2(0, (FLOOR + DOOR.spring) / 2)), vec2(DOOR.hw, (DOOR.spring - FLOOR) / 2)), sdCircle(p.sub(vec2(0, DOOR.spring)), DOOR.hw));
  const inside = cover(dOpen), above = cover(float(FLOOR).sub(p.y));
  const surround = cover(dOpen.sub(0.34)).mul(above);
  const arch = p.sub(vec2(0, DOOR.spring)), seg = Math.PI / 11, a = atan(arch.y, arch.x).div(seg);
  const joint = mix(min(fract(p.y.sub(FLOOR).div(0.55)), fract(p.y.sub(FLOOR).div(0.55)).oneMinus()).mul(0.55), min(fract(a), fract(a).oneMinus()).mul(seg).mul(length(arch)), step(DOOR.spring, p.y));
  let stone = mix(rgb(0x9a7852), rgb(0xc4a077), unit(p, 5));
  stone = mix(stone, rgb(0x3a281a), max(cover(joint.sub(0.022)), cover(abs(dOpen.sub(0.34)).sub(0.025))));
  const room = mix(rgb(0x1b120a), rgb(0x050302), smoothstep(FLOOR, DOOR.spring + DOOR.hw, p.y));
  const isGold = step(0.5, v.kind).mul(step(v.kind, 1.5)), isTiger = step(1.5, v.kind);
  const interior = mix(mix(room, treasure(room, p, v.time), isGold), tiger(room, p), isTiger);
  const w = v.open.mul(-0.8).add(1);
  const covered = cover(p.x.mul(-side).sub(w.mul(2 * DOOR.hw).sub(DOOR.hw)));
  const inner = mix(interior, lit(leaf(p, side, v.open, dOpen)), covered);
  let layer: Layer = { c: mix(lit(stone), inner, inside), a: max(surround, inside) };
  layer = over(layer, rgb(CHALK), tally(p, v.count).mul(0.85));
  const r = length(p.sub(vec2(0, DOOR.mid))), wave = fract(r.mul(1.8).sub(v.time.mul(1.3)));
  const band = smoothstep(0.38, 0.5, wave).mul(float(1).sub(smoothstep(0.5, 0.6, wave)));
  const fade = exp(r.mul(-0.6)).mul(smoothstep(0.3, 0.9, r)).mul(float(1).sub(smoothstep(1.5, 2.15, abs(p.x)))).mul(float(1).sub(smoothstep(2.3, 3.3, p.y))).mul(above);
  layer = over(layer, rgb(GROWL), band.mul(fade).mul(v.growl).mul(0.75));
  let dClaw = sdSegment(p, vec2(-0.95, 1.3), vec2(-0.35, -1.7), 0.07);
  dClaw = min(dClaw, sdSegment(p, vec2(-0.45, 1.5), vec2(0.15, -1.6), 0.08));
  dClaw = min(dClaw, sdSegment(p, vec2(0.05, 1.35), vec2(0.65, -1.5), 0.07));
  layer = over(layer, rgb(CLAW), cover(dClaw).mul(v.slash).mul(0.9));
  return sprite(4.4, 8, side * DOOR.x, -0.6, 2, layer.c, layer.a);
}

export function gauge(needle: Level): THREE.Mesh {
  const p = chart().sub(vec2(GAUGE.x, GAUGE.y)), r = length(p), R = GAUGE.r, a = atan(p.y, p.x);
  const dFace = max(sdCircle(p, R + 0.08), p.y.negate().sub(0.16));
  let c = mix(rgb(0x1e150e), rgb(0x3a2a1a), smoothstep(-0.2, R, p.y));
  const heat = abs(cos(a)).pow(2);
  c = mix(c, mix(rgb(0x4a3a2a), rgb(0xe0502a), heat), cover(abs(r.sub(R - 0.36)).sub(0.035)).mul(step(-0.01, p.y)));
  const k = round(a.div(Math.PI / 10)), major = step(abs(k.sub(5)).mod(5), 0.5);
  const dTick = max(abs(a.sub(k.mul(Math.PI / 10))).mul(r).sub(major.mul(0.01).add(0.012)), abs(r.sub(R - 0.17)).sub(major.mul(0.03).add(0.05)));
  c = mix(c, rgb(0xe9dcbc), cover(dTick).mul(step(-0.01, p.y)));
  c = mix(c, rgb(BRASS), max(cover(abs(r.sub(R)).sub(0.05)), cover(abs(p.y.add(0.12)).sub(0.04)).mul(cover(r.sub(R)))));
  const dir = vec2(cos(needle), sin(needle));
  c = mix(c, rgb(0x0c0806), cover(sdSegment(p.sub(vec2(0.03, -0.04)), dir.mul(-0.12), dir.mul(R - 0.12), 0.035)).mul(0.6));
  c = mix(c, rgb(0xf3e6c4), cover(sdSegment(p, dir.mul(-0.12), dir.mul(R - 0.12), 0.03)));
  c = mix(c, rgb(BRASS), cover(sdCircle(p, 0.1)));
  return sprite(2 * R + 0.5, R + 0.7, GAUGE.x, GAUGE.y + R / 2, 2, c, cover(dFace));
}
