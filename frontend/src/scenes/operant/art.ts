import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, pow, sin, smoothstep, uv, vec2 } from 'three/tsl';
import { fractal, grain, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;
export type Point = THREE.UniformNode<'vec2', THREE.Vector2>;

const BENCH = -3.3;
const ROD_Y = -2.6;
const ROD_R = 0.06;
const ROD_PITCH = 0.34;
const ROD_FIRST = -7.44;
const ROD_COUNT = 18;
export const FLOOR = ROD_Y + ROD_R;
export const BOX = { x0: -7.95, x1: -1.05, y0: BENCH, y1: 3.0 };
export const INSIDE = { x0: -7.62, x1: -1.38, y0: -2.72, y1: 2.72 };
export const PIVOT = new THREE.Vector2(-1.2, -1.25);
export const LEVER_LENGTH = 1.0;
export const LEVER_HALF = 0.05;
export const LEVER_DROP = 0.22;
export const LAMP = new THREE.Vector2(-1.55, -0.05);
const LENS = 0.17;
export const SPEAKER = new THREE.Vector2(-1.47, 1.2);
export const HOUSE = new THREE.Vector2(-1.58, 2.34);
export const CUP = { x0: -1.38, x1: -0.66, y0: -2.5, y1: -1.86 };
export const TROUGH = -2.44;
export const PELLET = 0.062;
export const MAG_LAMP = new THREE.Vector2(-0.9, -1.94);
const FEEDER = new THREE.Vector2(0.15, 0.82);
const WHEEL = 0.3;
export const POCKETS = 6;
export const CHUTE = [new THREE.Vector2(0.15, 0.4), new THREE.Vector2(0.15, -1.2), new THREE.Vector2(-0.72, -1.93)] as const;
const HOPPER = { x0: -0.42, x1: 0.72, y0: 1.26, y1: 3.4, fill: 2.78 };
const CABLE_X = 1.08;
export const RIG = { x0: -8.0, x1: 1.85, y0: BENCH - 0.02, y1: 3.62 };

const STEEL = 0x8e979f;
const STEEL_DARK = 0x2a3036;
const WALL = 0x49525a;
const BAKELITE = 0x17181b;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const picture = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
export const coverage = (d: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(max(max(fwidth(positionWorld.x), fwidth(positionWorld.z)), 1e-5))), 0, 1);
export const place = (mesh: THREE.Object3D, x: number, y: number) => mesh.position.set(x, 0, -y);

export function sheet(color: N<'vec3'>, opacity: N<'float'>, additive = false): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  if (additive) material.blending = THREE.AdditiveBlending;
  return material;
}

const rect = (q: N<'vec2'>, x0: number, y0: number, x1: number, y1: number, r = 0): N<'float'> =>
  sdBox(q.sub(vec2((x0 + x1) / 2, (y0 + y1) / 2)), vec2((x1 - x0) / 2, (y1 - y0) / 2), r);
const disc = (q: N<'vec2'>, c: THREE.Vector2 | readonly [number, number], r: number | N<'float'>): N<'float'> =>
  sdCircle(q.sub(Array.isArray(c) ? vec2(c[0], c[1]) : vec2((c as THREE.Vector2).x, (c as THREE.Vector2).y)), r);
const line = (q: N<'vec2'>, a: readonly [number, number], b: readonly [number, number], r: number): N<'float'> => sdSegment(q, vec2(a[0], a[1]), vec2(b[0], b[1]), r);
const ring = (d: N<'float'>, w: number): N<'float'> => abs(d).sub(w);
const hatch = (q: N<'vec2'>, pitch: number, width: number): N<'float'> =>
  coverage(abs(fract(q.x.add(q.y).div(pitch)).sub(0.5)).mul(pitch * Math.SQRT1_2).sub(width));
const metal = (q: N<'vec2'>, base: number, streak = 0.07): N<'vec3'> => rgb(base).mul(noise(vec2(q.x.mul(0.9), q.y.mul(60))).mul(streak).add(1));
const shine = (d: N<'float'>, depth: number): N<'float'> => smoothstep(0, depth, d.negate());
const step1 = (x: N<'float'>): N<'float'> => clamp(x.mul(1e4), 0, 1);

export function backdropMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const q = picture(), cell = 0.46, k = (2 * Math.PI) / cell;
  const cx = cos(q.x.mul(k)), sx = sin(q.x.mul(k)), cy = cos(q.y.mul(k)), sy = sin(q.y.mul(k));
  const bump = cx.mul(cy);
  const relief = sx.mul(cy).mul(0.7).sub(cx.mul(sy).mul(0.7));
  const speck = noise(q.mul(9)).mul(0.05).add(fractal(q.mul(0.35), 3).mul(0.08));
  let wall: N<'vec3'> = rgb(0x34424d).mul(bump.mul(0.16).add(relief.mul(0.16)).add(1).add(speck));
  const chamberShade = rect(q, BOX.x0 - 0.05, BOX.y0, CABLE_X + 0.25, HOPPER.y1 + 0.12, 0.1);
  const recorderShade = rect(q, 1.66, BENCH, 8.14, 2.55, 0.3);
  const near = min(chamberShade, recorderShade);
  wall = wall.mul(smoothstep(-0.05, 0.75, near).mul(0.55).add(0.45));
  const top = coverage(float(BENCH).sub(q.y));
  const lip = coverage(abs(q.y.sub(BENCH - 0.06)).sub(0.06));
  const grainWood = noise(vec2(q.x.mul(0.7), q.y.mul(14))).mul(0.12).add(noise(vec2(q.x.mul(3.1), q.y.mul(40))).mul(0.05));
  const apron = rgb(0x2b2119).mul(grainWood.add(1)).mul(smoothstep(BENCH - 0.9, BENCH - 0.12, q.y).mul(0.55).add(0.45));
  let bench: N<'vec3'> = mix(apron, rgb(0x5a4633).mul(grainWood.add(1)), lip);
  const underShadow = smoothstep(0.02, 0.5, near).mul(0.5).add(0.5);
  bench = bench.mul(mix(float(1), underShadow, coverage(abs(q.y.sub(BENCH - 0.06)).sub(0.06))));
  bench = mix(bench, rgb(0x08090a), coverage(q.y.sub(BENCH - 0.95)));
  const color: N<'vec3'> = mix(bench, wall, top);
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(color);
  return material;
}

export interface RigGlow {
  lamp: Level;
  house: Level;
  magazine: Level;
  buzz: Level;
  wheel: Level;
}

export function rigMaterial(lit: Lit, accent: number, glow: RigGlow): THREE.MeshBasicNodeMaterial {
  const q = picture();
  let alpha: N<'float'> = float(0);
  const inside = rect(q, INSIDE.x0, INSIDE.y0, INSIDE.x1, INSIDE.y1);
  const seam = min(abs(q.x.add(5.54)), abs(q.x.add(3.46)));
  const rivetRow = min(abs(q.y.sub(INSIDE.y1 - 0.16)), abs(q.y.sub(INSIDE.y0 + 0.3)));
  const rivets = sdCircle(vec2(seam, rivetRow), 0.03);
  let back: N<'vec3'> = metal(q, WALL, 0.06).mul(smoothstep(0, 0.9, inside.negate()).mul(0.4).add(0.6));
  back = mix(back, rgb(0x2a2f34), coverage(seam.sub(0.008)).mul(0.75));
  back = mix(back, rgb(0x9aa3ab), coverage(rivets).mul(0.55));
  let color: N<'vec3'> = lit(back);
  alpha = max(alpha, coverage(inside));

  const rodIndex = clamp(floor(q.x.sub(ROD_FIRST).div(ROD_PITCH).add(0.5)), 0, ROD_COUNT - 1);
  const rodAt = vec2(rodIndex.mul(ROD_PITCH).add(ROD_FIRST), ROD_Y);
  const rodQ = q.sub(rodAt);
  const rod = sdCircle(rodQ, ROD_R);
  const rodShade = mix(rgb(0x4b545c), rgb(0xe9eef2), smoothstep(-0.07, 0.06, rodQ.y.add(rodQ.x.mul(-0.6))));
  color = mix(color, lit(rodShade).add(rgb(0xffffff).mul(coverage(disc(rodQ, [-0.02, 0.025], 0.014)).mul(0.25))), coverage(rod));
  const railD = rect(q, INSIDE.x0, ROD_Y - 0.014, INSIDE.x1, ROD_Y + 0.014);
  color = mix(color, lit(rgb(0x3a4148)), coverage(railD).mul(coverage(rod.negate())).mul(0.6));

  const tray = rect(q, BOX.x0 + 0.03, BENCH, BOX.x1 - 0.03, INSIDE.y0, 0.03);
  const trayInner = rect(q, BOX.x0 + 0.08, BENCH + 0.05, BOX.x1 - 0.08, INSIDE.y0 - 0.1, 0.02);
  const flake = (scale: number, shift: number) => {
    const g = q.mul(scale).add(shift), cell = floor(g), r = grain(cell);
    const f = fract(g).sub(0.5).sub(r.sub(0.5).mul(0.45));
    const a = r.x.mul(Math.PI), c = cos(a), sn = sin(a);
    const turned = vec2(f.x.mul(c).add(f.y.mul(sn)), f.y.mul(c).sub(f.x.mul(sn)));
    return { d: sdBox(turned, vec2(r.y.mul(0.12).add(0.2), r.x.mul(0.06).add(0.08)), 0.05).div(scale), tone: r.y };
  };
  const a1 = flake(7, 0), a2 = flake(9.5, 0.37);
  let bedding: N<'vec3'> = rgb(0x4a3d2f);
  bedding = mix(bedding, mix(rgb(0x9c8a6c), rgb(0xc9b893), a2.tone), coverage(a2.d).mul(0.9));
  bedding = mix(bedding, mix(rgb(0xb3a07e), rgb(0xe0d2b2), a1.tone), coverage(a1.d));
  bedding = bedding.mul(smoothstep(BENCH, INSIDE.y0, q.y).mul(0.3).add(0.7));
  let trayColor: N<'vec3'> = mix(metal(q, 0x78828a), bedding, coverage(trayInner));
  trayColor = mix(trayColor, rgb(0x20252a), coverage(ring(tray, 0.012)));
  color = mix(color, lit(trayColor), coverage(tray));
  alpha = max(alpha, coverage(tray));

  const leftWall = rect(q, BOX.x0, INSIDE.y0, INSIDE.x0, BOX.y1);
  const ceiling = rect(q, BOX.x0, INSIDE.y1, BOX.x1, BOX.y1);
  const slab = rect(q, INSIDE.x1, INSIDE.y0, BOX.x1, BOX.y1);
  const slot = rect(q, INSIDE.x1 - 0.01, PIVOT.y - 0.08, BOX.x1 + 0.01, PIVOT.y + 0.08);
  const mouth = rect(q, INSIDE.x1 - 0.01, CUP.y0, BOX.x1 + 0.01, CUP.y1);
  const lampHole = rect(q, INSIDE.x1 - 0.01, LAMP.y - 0.15, BOX.x1 + 0.01, LAMP.y + 0.15);
  const panel = max(slab, min(min(slot, mouth), lampHole).negate());
  const section = min(min(leftWall, ceiling), panel);
  let cut: N<'vec3'> = mix(metal(q, STEEL, 0.05), rgb(0x5f6870), hatch(q, 0.11, 0.009).mul(0.55));
  cut = mix(cut, rgb(STEEL_DARK), coverage(ring(section, 0.012)));
  color = mix(color, lit(cut), coverage(section));
  alpha = max(alpha, coverage(section));
  color = mix(color, rgb(0x08090b), coverage(slot).mul(coverage(slab)));

  const flange = rect(q, INSIDE.x1 - 0.05, LAMP.y - 0.3, INSIDE.x1, LAMP.y + 0.3, 0.015);
  const barrel = rect(q, LAMP.x, LAMP.y - 0.2, INSIDE.x1 - 0.05, LAMP.y + 0.2, 0.025);
  const lensQ = q.sub(vec2(LAMP.x, LAMP.y));
  const lens = max(sdCircle(lensQ, LENS), lensQ.x);
  const lampOn = glow.lamp;
  const chrome = (d: N<'float'>, y: N<'float'>) => mix(rgb(0x3a4048), rgb(0xdfe5ea), smoothstep(-0.25, 0.25, y)).mul(shine(d, 0.04).mul(-0.25).add(1.1));
  const facet = smoothstep(LENS, 0, length(lensQ.add(vec2(0.06, -0.05))));
  const offLens = mix(rgb(0x4a2c10), rgb(0xa86a2a), facet.mul(0.85)).add(rgb(0xfff0d8).mul(coverage(disc(lensQ, [-0.08, 0.07], 0.028)).mul(0.6)));
  const hot = rgb(accent).mul(lampOn.mul(4.4)).add(rgb(0xfff2d8).mul(pow(facet, float(2)).mul(lampOn).mul(6)));
  color = mix(color, lit(chrome(flange, lensQ.y)), coverage(flange));
  color = mix(color, lit(mix(rgb(0x23272c), rgb(0x59616a), smoothstep(-0.2, 0.2, lensQ.y))), coverage(barrel));
  color = mix(color, lit(rgb(0xc9d0d6)), coverage(ring(barrel, 0.01)).mul(coverage(barrel)).mul(0.6));
  color = mix(color, lit(offLens).mul(lampOn.oneMinus()).add(hot), coverage(lens));
  color = mix(color, lit(rgb(0xd9dfe4)).add(rgb(accent).mul(lampOn.mul(0.8))), coverage(ring(sdCircle(lensQ, LENS), 0.012)).mul(step1(lensQ.x.negate().add(0.012))));
  alpha = max(alpha, max(max(coverage(flange), coverage(barrel)), coverage(lens)));

  const grille = rect(q, SPEAKER.x - 0.06, SPEAKER.y - 0.21, INSIDE.x1, SPEAKER.y + 0.21, 0.03);
  const ridges = coverage(abs(fract(q.y.sub(SPEAKER.y).div(0.06).add(0.5)).sub(0.5)).mul(0.06).sub(0.012)).mul(step1(float(0.16).sub(abs(q.y.sub(SPEAKER.y))))).mul(step1(float(SPEAKER.x + 0.02).sub(q.x)));
  const buzz = glow.buzz;
  let grilleColor: N<'vec3'> = lit(mix(rgb(0x2c3137), rgb(0x7a838b), smoothstep(-0.21, 0.21, q.y.sub(SPEAKER.y))));
  grilleColor = mix(grilleColor, rgb(0x0b0c0e).add(rgb(0x9ab8ff).mul(buzz.mul(2.4))), ridges);
  color = mix(color, grilleColor, coverage(grille));
  color = mix(color, rgb(0x15181b), coverage(ring(grille, 0.008)).mul(coverage(grille)));
  alpha = max(alpha, coverage(grille));

  const bracket = rect(q, HOUSE.x + 0.04, HOUSE.y + 0.1, INSIDE.x1, HOUSE.y + 0.17, 0.02);
  const hoodQ = q.sub(vec2(HOUSE.x, HOUSE.y));
  const hood = max(ring(sdCircle(hoodQ, 0.15), 0.024), hoodQ.y.negate().add(0.02));
  const bulb = sdCircle(hoodQ.sub(vec2(0, -0.01)), 0.09);
  color = mix(color, lit(chrome(min(bracket, hood), hoodQ.y)), coverage(min(bracket, hood)));
  color = mix(color, rgb(0xfff0d6).mul(glow.house.mul(2.6).add(0.3)), coverage(bulb));
  alpha = max(alpha, max(coverage(min(bracket, hood)), coverage(bulb)));

  const cavity = rect(q, CUP.x0 - 0.02, CUP.y0, CUP.x1, CUP.y1, 0.03);
  const cavityQ = q.sub(vec2(MAG_LAMP.x, MAG_LAMP.y));
  const magGlow = glow.magazine;
  const cavityLight = exp(length(cavityQ.mul(vec2(0.7, 1.3))).mul(-3)).mul(magGlow);
  const bowl = max(sdCircle(q.sub(vec2((CUP.x0 + CUP.x1) / 2 - 0.02, TROUGH + 0.5)), 0.56), q.y.sub(TROUGH + 0.1));
  let cavityColor: N<'vec3'> = mix(rgb(0x0b0d10), rgb(0x2a2f35), smoothstep(CUP.y0, CUP.y1, q.y).mul(0.7));
  cavityColor = mix(cavityColor, mix(rgb(0x6d767f), rgb(0xb8c0c7), smoothstep(TROUGH - 0.06, TROUGH + 0.1, q.y)), coverage(bowl.negate()).mul(coverage(q.y.sub(TROUGH + 0.1))));
  cavityColor = lit(cavityColor).add(rgb(0xffd48a).mul(cavityLight.mul(1.3)));
  color = mix(color, cavityColor, coverage(cavity));
  const magBulb = sdCircle(cavityQ, 0.045);
  color = mix(color, rgb(0x4a4030).add(rgb(0xffe2a0).mul(magGlow.mul(5))), coverage(magBulb));
  const lip = max(rect(q, CUP.x0 - 0.06, CUP.y0 - 0.05, CUP.x0 + 0.02, CUP.y1 + 0.05, 0.02), rect(q, CUP.x0 - 0.2, CUP.y0 + 0.01, CUP.x0, CUP.y1 - 0.01).negate());
  color = mix(color, lit(chrome(lip, q.y.sub((CUP.y0 + CUP.y1) / 2))), coverage(lip));
  alpha = max(alpha, max(coverage(cavity), coverage(lip)));

  const cable = min(min(min(line(q, [-0.5, PIVOT.y - 0.06], [0.78, PIVOT.y - 0.06], 0.05), line(q, [0.78, PIVOT.y - 0.06], [CABLE_X, PIVOT.y - 0.38], 0.05)),
    min(line(q, [CABLE_X, PIVOT.y - 0.38], [CABLE_X, BENCH + 0.28], 0.05), line(q, [CABLE_X, BENCH + 0.28], [CABLE_X + 0.2, BENCH + 0.08], 0.05))),
    min(line(q, [CABLE_X + 0.2, BENCH + 0.08], [1.52, BENCH + 0.08], 0.05), line(q, [1.52, BENCH + 0.08], [1.74, BENCH + 0.3], 0.05)));
  const cableTone = mix(rgb(0x111214), rgb(0x44474d), smoothstep(-0.015, -0.045, cable));
  color = mix(color, lit(cableTone), coverage(cable));
  alpha = max(alpha, coverage(cable));

  const boxShade = (d: N<'float'>, base: number, lift = 0.25): N<'vec3'> => lit(rgb(base).mul(shine(d, 0.12).mul(-lift).add(1 + lift)));
  const socket = rect(q, BOX.x1, LAMP.y - 0.15, -0.74, LAMP.y + 0.15, 0.04);
  const speakerBody = rect(q, BOX.x1, SPEAKER.y - 0.16, -0.82, SPEAKER.y + 0.16, 0.05);
  const houseSocket = rect(q, BOX.x1, HOUSE.y - 0.07, -0.9, HOUSE.y + 0.1, 0.03);
  const switchBox = rect(q, BOX.x1, PIVOT.y - 0.27, -0.48, PIVOT.y + 0.27, 0.04);
  const housing = rect(q, BOX.x1, CUP.y0 - 0.1, -0.56, CUP.y1 + 0.1, 0.04);
  for (const d of [socket, speakerBody, houseSocket, switchBox]) {
    color = mix(color, boxShade(d, BAKELITE, 0.9), coverage(d));
    alpha = max(alpha, coverage(d));
  }
  const terminal = min(disc(q, [-0.6, PIVOT.y + 0.12], 0.035), disc(q, [-0.6, PIVOT.y - 0.12], 0.035));
  color = mix(color, lit(rgb(0xc59a4a)), coverage(terminal));
  const microswitch = rect(q, -0.98, PIVOT.y - 0.14, -0.7, PIVOT.y + 0.1, 0.02);
  color = mix(color, lit(rgb(0x3a3f46)), coverage(microswitch));
  color = mix(color, boxShade(housing, 0x6c757d, 0.3), coverage(housing).mul(coverage(cavity.negate())));
  alpha = max(alpha, coverage(housing));

  const tube = (a: THREE.Vector2, b: THREE.Vector2) => sdSegment(q, vec2(a.x, a.y), vec2(b.x, b.y), 0.068);
  const chute = min(tube(CHUTE[0], CHUTE[1]), tube(CHUTE[1], CHUTE[2]));
  const glass = mix(rgb(0x2c3a44), rgb(0x607482), smoothstep(-0.068, 0, chute).mul(-1).add(1));
  color = mix(color, lit(glass).add(rgb(0xcfe6ff).mul(coverage(ring(chute.add(0.03), 0.006)).mul(0.25))), coverage(chute).mul(0.85));
  color = mix(color, lit(rgb(0x1a2026)), coverage(ring(chute, 0.009)));
  alpha = max(alpha, coverage(chute));

  const feeder = rect(q, -0.62, 0.38, 0.92, 1.26, 0.07);
  color = mix(color, boxShade(feeder, 0x3f4a55, 0.4), coverage(feeder));
  const plate = rect(q, 0.52, 0.5, 0.84, 0.66, 0.02);
  color = mix(color, lit(rgb(0xb8995a)), coverage(plate));
  color = mix(color, lit(rgb(0x6e5a33)), coverage(ring(plate, 0.008)));
  const window = disc(q, FEEDER, WHEEL + 0.05);
  const wq = q.sub(vec2(FEEDER.x, FEEDER.y));
  const angle = glow.wheel;
  const spin = vec2(cos(angle), sin(angle));
  const turned = vec2(wq.x.mul(spin.x).add(wq.y.mul(spin.y)), wq.y.mul(spin.x).sub(wq.x.mul(spin.y)));
  const sector = (2 * Math.PI) / POCKETS;
  const theta = atan(turned.y, turned.x);
  const nearest = floor(theta.div(sector).add(0.5)).mul(sector);
  const pocketQ = turned.sub(vec2(cos(nearest), sin(nearest)).mul(WHEEL * 0.62));
  const pocket = sdCircle(pocketQ, 0.07);
  const pelletIn = sdCircle(pocketQ, PELLET);
  let wheelColor: N<'vec3'> = mix(rgb(0x2a3038), rgb(0x9aa4ad), smoothstep(-0.3, 0.3, turned.y.add(turned.x).mul(0.7)));
  wheelColor = mix(wheelColor, rgb(0x0d0f12), coverage(pocket));
  wheelColor = mix(wheelColor, pelletShade(pocketQ), coverage(pelletIn).mul(step1(wq.y.add(WHEEL * 0.3))));
  wheelColor = mix(wheelColor, rgb(0x15181c), coverage(disc(wq, [0, 0], 0.05)));
  color = mix(color, lit(rgb(0x0b0d10)), coverage(window));
  color = mix(color, lit(wheelColor), coverage(disc(q, FEEDER, WHEEL)));
  color = mix(color, lit(rgb(0x6c7680)), coverage(ring(window, 0.018)));
  alpha = max(alpha, coverage(feeder));

  const hopper = rect(q, HOPPER.x0, HOPPER.y0, HOPPER.x1, HOPPER.y1, 0.1);
  const packing = vec2(q.x.div(0.11).add(floor(q.y.div(0.092)).mul(0.5)), q.y.div(0.092));
  const pelletCell = floor(packing);
  const jitter = grain(pelletCell);
  const pelletQ = fract(packing).sub(0.5).sub(jitter.sub(0.5).mul(0.3)).mul(vec2(0.11, 0.092));
  const heap = coverage(q.y.sub(HOPPER.fill).add(sin(q.x.mul(9)).mul(0.03)));
  const packed = pelletShade(pelletQ.div(0.88)).mul(jitter.y.mul(0.3).add(0.75));
  let hopperColor: N<'vec3'> = mix(rgb(0x1b2229), mix(rgb(0x3a2a18), packed, coverage(sdCircle(pelletQ, 0.044))), heap);
  hopperColor = lit(hopperColor).add(rgb(0xd8ecff).mul(smoothstep(0.1, 0, abs(q.x.sub(HOPPER.x0 + 0.2))).mul(0.12)));
  color = mix(color, hopperColor, coverage(hopper));
  color = mix(color, lit(rgb(0x7f8a94)), coverage(ring(hopper, 0.016)));
  const cap = rect(q, HOPPER.x0 - 0.06, HOPPER.y1 - 0.02, HOPPER.x1 + 0.06, HOPPER.y1 + 0.12, 0.04);
  color = mix(color, boxShade(cap, 0x505b66, 0.4), coverage(cap));
  const collar = rect(q, HOPPER.x0 + 0.1, HOPPER.y0 - 0.06, HOPPER.x1 - 0.1, HOPPER.y0 + 0.06, 0.02);
  color = mix(color, lit(rgb(0x59636d)), coverage(collar));
  alpha = max(alpha, max(coverage(hopper), coverage(cap)));

  return sheet(color, alpha);
}

function pelletShade(p: N<'vec2'>): N<'vec3'> {
  const n = p.div(PELLET);
  const light = clamp(dot(n, vec2(-0.55, 0.65)).mul(0.5).add(0.6), 0, 1);
  return mix(rgb(0x6b4a26), rgb(0xd9a766), light).add(rgb(0xfff0d0).mul(coverage(sdCircle(n.sub(vec2(-0.35, 0.4)), 0.22).mul(PELLET)).mul(0.35)));
}

export function leverMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const width = LEVER_LENGTH + 0.1, height = 0.22;
  const p = uv().sub(0.5).mul(vec2(width, height));
  const blade = sdBox(p.sub(vec2(-0.05, 0)), vec2(LEVER_LENGTH / 2, LEVER_HALF), 0.035);
  const across = p.y.div(LEVER_HALF);
  let color: N<'vec3'> = mix(rgb(0x59636c), rgb(0xe3e9ee), smoothstep(-1, 0.6, across)).mul(noise(vec2(p.x.mul(40), p.y.mul(3))).mul(0.05).add(1));
  color = mix(color, rgb(0x2d3339), coverage(ring(blade, 0.008)));
  color = lit(color).add(rgb(0xffffff).mul(coverage(abs(p.y.sub(LEVER_HALF * 0.55)).sub(0.006)).mul(coverage(blade.add(0.02))).mul(0.18)));
  return sheet(color, coverage(blade));
}

export const LEVER_SPRITE = { width: LEVER_LENGTH + 0.1, height: 0.22, pivot: 0.4 };

export function pelletMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const state = attribute<'vec4'>('aPellet', 'vec4');
  const p = uv().sub(0.5).mul(PELLET * 2.6);
  const body = sdBox(p, vec2(PELLET * 0.95, PELLET * 0.78), PELLET * 0.7);
  const color = lit(pelletShade(p)).add(pelletShade(p).mul(state.y.mul(0.9)));
  return sheet(mix(rgb(0x2a1a0c), color, coverage(body.add(0.008))), coverage(body).mul(state.x));
}

export const BUZZ = 2.4;

export function buzzMaterial(level: Level, age: Level): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(BUZZ);
  const r = length(p), toward = p.x.negate().div(max(r, 1e-3));
  const cone = smoothstep(0.35, 0.8, toward);
  const wave = fract(r.mul(2.6).sub(age.mul(3)));
  const arcs = coverage(abs(wave.sub(0.5)).sub(0.12).mul(0.38)).mul(cone).mul(smoothstep(1.2, 0.2, r)).mul(smoothstep(0.1, 0.25, r));
  return sheet(rgb(0x9fb8ff).mul(1.6), arcs.mul(level), true);
}

export function hazeMaterial(tint: number, level: Level, reach: number): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(2);
  const r = length(p);
  const fall = exp(r.mul(-reach)).mul(smoothstep(1, 0.6, r));
  return sheet(rgb(tint), fall.mul(level), true);
}

