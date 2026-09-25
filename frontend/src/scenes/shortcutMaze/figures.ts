import * as THREE from 'three/webgpu';
import { abs, atan, cos, float, floor, fract, length, max, min, mix, sin, smoothstep, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { PALETTE, type Level, type Lit } from './canal';

type N<T extends string> = THREE.Node<T>;
export type Offset = THREE.UniformNode<'vec2', THREE.Vector2>;
export type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));

const turn = (p: N<'vec2'>, angle: N<'float'>) => {
  const c = cos(angle), s = sin(angle);
  return vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)));
};

const round = (d: N<'float'>, depth: number) => smoothstep(-depth, 0, d).mul(-0.38).add(1);
const soft = (d: N<'float'>, blur: number) => smoothstep(-blur, blur, d).oneMinus();
const seams = (at: N<'float'>, pitch: number, width: number) => coverage(float(0.5).sub(abs(fract(at.div(pitch)).sub(0.5))).mul(pitch).sub(width));
const hash1 = (at: N<'float'>) => fract(sin(at.mul(91.7)).mul(43758.5453));
const ring = (p: N<'vec2'>, radius: number, width: number) => abs(length(p).sub(radius)).sub(width);

class Paint {
  private pre: N<'vec3'> = vec3(0);
  private alpha: N<'float'> = float(0);

  over(cover: N<'float'>, color: N<'vec3'>): void {
    this.pre = color.mul(cover).add(this.pre.mul(cover.oneMinus()));
    this.alpha = cover.add(this.alpha.mul(cover.oneMinus()));
  }

  material(fade?: N<'float'>): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    material.colorNode = this.pre.div(max(this.alpha, 1e-4));
    material.opacityNode = fade ? this.alpha.mul(fade) : this.alpha;
    return material;
  }
}

const INK = 0x0d0c0b;
const FRAME = 0x1c1f22;
const TYRE = 0x0a0a0b;
const RACK = 0x34373a;
const BAG = 0x7a2b25;
const WICKER = [0x7a5530, 0xc39a5f] as const;
const TROUSER = 0x252c3e;
const GLOVE = 0x3b2b22;
const HAT = 0x2c6d69;
const POM = 0xece5d6;
export const HEADLAMP = 0xfff0cc;
export const TAILLAMP = 0xff3b2c;

export const RIDER = 0.9;
export const LAMP_AT = 0.3;
export const TAIL_AT = -0.29;

export interface Rider {
  material: THREE.MeshBasicNodeMaterial;
  pedal: Level;
  steer: Level;
  lean: Level;
  fade: Level;
  shadow: Offset;
}

export function riderMaterial(accent: number, lit: Lit): Rider {
  const pedal = uniform(0), steer = uniform(0), lean = uniform(0), fade = uniform(1), shadow = uniform(new THREE.Vector2());
  const p = uv().sub(0.5).mul(RIDER), stem = vec2(0, 0.15);
  const f = turn(p.sub(stem), steer.negate()).add(stem);
  const sway = lean.mul(0.035), r = p.sub(vec2(sway, 0)), kick = sin(pedal).mul(0.03);
  const grip = (side: number) => turn(vec2(side * 0.11, -0.034), steer).add(stem);
  const shoulder = (side: number) => vec2(sway.add(side * 0.09), -0.05);

  const rearTyre = sdSegment(p, vec2(0, -0.305), vec2(0, -0.135), 0.02);
  const frontTyre = sdSegment(f, vec2(0, 0.135), vec2(0, 0.305), 0.02);
  const rearGuard = sdSegment(p, vec2(0, -0.298), vec2(0, -0.142), 0.011);
  const frontGuard = sdSegment(f, vec2(0, 0.142), vec2(0, 0.298), 0.011);
  const tube = sdSegment(p, vec2(0, -0.19), stem, 0.013);
  const rack = sdBox(p.sub(vec2(0, -0.222)), vec2(0.036, 0.06), 0.008);
  const bags = sdBox(vec2(abs(p.x).sub(0.064), p.y.add(0.215)), vec2(0.026, 0.056), 0.012);
  const basket = sdBox(f.sub(vec2(0, 0.238)), vec2(0.058, 0.046), 0.012);
  const bars = min(sdSegment(p, stem, grip(1), 0.01), sdSegment(p, stem, grip(-1), 0.01));
  const hands = min(sdCircle(p.sub(grip(1)), 0.017), sdCircle(p.sub(grip(-1)), 0.017));
  const knees = min(sdCircle(r.sub(vec2(0.068, kick.add(0.03))), 0.029), sdCircle(r.sub(vec2(-0.068, kick.negate().add(0.03))), 0.029));
  const torso = ellipse(r, [0, -0.075], [0.116, 0.076]);
  const arms = min(sdSegment(p, shoulder(1), grip(1), 0.021), sdSegment(p, shoulder(-1), grip(-1), 0.021));
  const head = sdCircle(r.sub(vec2(0, -0.045)), 0.054);
  const pom = sdCircle(r.sub(vec2(0, -0.05)), 0.019);
  const lamp = sdCircle(f.sub(vec2(0, LAMP_AT - 0.004)), 0.016);
  const tail = sdCircle(p.sub(vec2(0, TAIL_AT)), 0.013);
  const body = min(min(min(rearTyre, frontTyre), min(rack, bags)), min(min(basket, bars), min(min(torso, arms), min(head, knees))));

  const paint = new Paint();
  paint.over(soft(ellipse(p.sub(shadow), [0, -0.01], [0.14, 0.33]), 0.05).mul(0.36), vec3(0));
  paint.over(coverage(body.sub(0.013)), rgb(INK));
  paint.over(coverage(rearTyre), lit(rgb(TYRE)));
  paint.over(coverage(rearGuard), lit(rgb(FRAME)));
  paint.over(coverage(rack), lit(rgb(RACK)));
  paint.over(coverage(bags), lit(rgb(BAG).mul(round(bags, 0.03))));
  paint.over(coverage(tube), lit(rgb(FRAME)));
  paint.over(coverage(frontTyre), lit(rgb(TYRE)));
  paint.over(coverage(frontGuard), lit(rgb(FRAME)));
  paint.over(coverage(basket), lit(mix(rgb(WICKER[0]), rgb(WICKER[1]), coverage(abs(basket.add(0.009)).sub(0.008)))));
  paint.over(coverage(bars), lit(rgb(FRAME)));
  paint.over(coverage(knees), lit(rgb(TROUSER).mul(round(knees, 0.02))));
  paint.over(coverage(torso), lit(rgb(accent).mul(round(torso, 0.06))));
  paint.over(coverage(arms), lit(rgb(accent).mul(0.78)));
  paint.over(coverage(hands), lit(rgb(GLOVE)));
  paint.over(coverage(head), lit(rgb(HAT).mul(round(head, 0.04))));
  paint.over(coverage(pom), lit(rgb(POM)));
  paint.over(coverage(tail), rgb(TAILLAMP).mul(2.2));
  paint.over(coverage(lamp), rgb(HEADLAMP).mul(3));
  return { material: paint.material(fade), pedal, steer, lean, fade, shadow };
}

const WHITE = 0xe8e3d7;
const STEEL = 0x5a5f64;
const PLANK = [0x6b5540, 0x866a4d] as const;
const SEAM = 0x1d1611;
const RAIL = 0x2a2e32;

export const SWING = 1.5;
export const SPAN = 1.16;
export const BEAM = 0.44;

export interface Swing {
  material: THREE.MeshBasicNodeMaterial;
  shadow: Offset;
  north: Level;
  south: Level;
}

export function swingMaterial(lit: Lit): Swing {
  const shadow = uniform(new THREE.Vector2()), north = uniform(0), south = uniform(0);
  const p = uv().sub(0.5).mul(SWING), half = vec2(BEAM / 2, SPAN / 2);
  const deck = sdBox(p, half, 0.018);
  const side = abs(p.x).sub(BEAM / 2 - 0.032);
  const girder = max(abs(side).sub(0.032), abs(p.y).sub(SPAN / 2 - 0.004));
  const lip = max(deck, float(SPAN / 2 - 0.05).sub(abs(p.y)));
  const rail = max(abs(side).sub(0.0065), abs(p.y).sub(SPAN / 2 - 0.02));
  const post = length(vec2(side, fract(p.y.div(0.13).add(0.5)).sub(0.5).mul(0.13))).sub(0.013);
  const pivot = sdCircle(p, 0.06);
  const sector = Math.PI / 3, bolt = floor(atan(p.y, p.x).div(sector).add(0.5)).mul(sector);
  const bolts = sdCircle(p.sub(vec2(cos(bolt), sin(bolt)).mul(0.038)), 0.009);

  const board = floor(p.y.div(0.05));
  let wood: N<'vec3'> = mix(rgb(PLANK[0]), rgb(PLANK[1]), hash1(board)).mul(noise(p.mul(vec2(3, 40))).mul(0.12).add(0.95));
  wood = mix(wood, rgb(SEAM), seams(p.y, 0.05, 0.004));
  const worn = mix(south, north, smoothstep(-0.01, 0.01, p.y)).mul(smoothstep(0.03, 0.1, abs(p.x)).oneMinus());
  wood = mix(wood, rgb(PALETTE.polish).mul(0.8), worn.mul(0.42));
  const tread = mix(rgb(STEEL), rgb(STEEL).mul(1.3), seams(p.x.add(p.y), 0.03, 0.006));

  const paint = new Paint();
  paint.over(soft(sdBox(p.sub(shadow), half, 0.02), 0.035).mul(0.5), vec3(0));
  paint.over(coverage(deck.sub(0.009)), rgb(INK));
  paint.over(coverage(deck), lit(wood));
  paint.over(coverage(lip), lit(tread));
  paint.over(coverage(girder), lit(rgb(WHITE).mul(smoothstep(0.032, -0.032, side).mul(0.2).add(0.85))));
  paint.over(coverage(rail), lit(rgb(RAIL)));
  paint.over(coverage(max(post, girder)), lit(rgb(WHITE).mul(1.08)));
  paint.over(coverage(pivot), lit(rgb(STEEL).mul(smoothstep(0.06, 0, length(p)).mul(0.3).add(0.8))));
  paint.over(coverage(bolts), lit(rgb(0x2b2e31)));
  return { material: paint.material(), shadow, north, south };
}

const HULL_C = 0x1c2a26;
const GUNWALE = 0x56625c;
const TEAK = [0x5c4838, 0x735a45] as const;
const ROOF = 0x33463c;
const SKYLIGHT = 0xffc98a;
const LEAF = [0x2f5a35, 0x4f7f3c] as const;
const POT = 0x9a5536;
const ROPE = 0xcdbb95;
const IRON = 0x222527;

export const BOAT = new THREE.Vector2(2.3, 1.0);
export const HULL = new THREE.Vector2(1.94, 0.56);
export const CABIN = { x: 0.05, half: new THREE.Vector2(0.6, 0.19) };
export const PLANK_X = 0.8;
export const MOOR = 0.1;
export const WINDOWS = [-0.42, -0.14, 0.14, 0.42] as const;
export const LANTERN = new THREE.Vector2(0.93, -0.19);

export interface Houseboat {
  material: THREE.MeshBasicNodeMaterial;
  flare: Level;
}

export function houseboatMaterial(lit: Lit, mirror: number, drop: THREE.Vector2): Houseboat {
  const flare = uniform(0);
  const raw = uv().sub(0.5).mul(vec2(BOAT.x, BOAT.y)), p = vec2(raw.x.mul(mirror), raw.y);
  const quay = -(HULL.y / 2 + MOOR);
  const hull = sdBox(p, vec2(HULL.x / 2, HULL.y / 2), 0.17);
  const deck = hull.add(0.042);
  const c = p.sub(vec2(CABIN.x, 0));
  const cabin = sdBox(c, vec2(CABIN.half.x, CABIN.half.y), 0.025);
  let sky: N<'float'> = float(4);
  for (const x of [-0.36, 0, 0.36]) sky = min(sky, sdBox(c.sub(vec2(x, 0)), vec2(0.055, 0.034), 0.008));
  const chimney = sdCircle(c.sub(vec2(0.47, 0.1)), 0.028);
  let plants: N<'float'> = float(4), pots: N<'float'> = float(4);
  for (const [x, y, radius] of [[-0.84, 0.12, 0.07], [-0.86, -0.1, 0.055], [-0.7, 0.17, 0.045]] as const) {
    plants = min(plants, sdCircle(p.sub(vec2(x, y)), radius).add(noise(p.mul(38).add(x * 7)).mul(0.012)));
    pots = min(pots, sdCircle(p.sub(vec2(x + 0.012, y - 0.014)), radius * 0.8));
  }
  const table = sdCircle(p.sub(vec2(-0.66, -0.04)), 0.05);
  const chairs = min(sdBox(p.sub(vec2(-0.66, -0.135)), vec2(0.028, 0.024), 0.006), sdBox(p.sub(vec2(-0.66, 0.055)), vec2(0.028, 0.024), 0.006));
  const plank = sdBox(p.sub(vec2(PLANK_X, (quay - HULL.y / 2) / 2 + 0.02)), vec2(0.075, (HULL.y / 2 + quay) / -2 + 0.045), 0.01);
  const buoyAt = p.sub(vec2(0.8, 0.17));
  const buoy = ring(buoyAt, 0.034, 0.011);
  const stripes = smoothstep(-0.2, 0.2, sin(atan(buoyAt.y, buoyAt.x).mul(4)));
  const fenders = sdBox(vec2(abs(p.x.add(0.05)).sub(0.35), p.y.add(HULL.y / 2)), vec2(0.045, 0.018), 0.018);
  const bollards = sdCircle(vec2(abs(p.x.add(0.125)).sub(0.675), p.y.sub(quay - 0.035)), 0.026);
  const lines = min(sdSegment(p, vec2(-0.8, -0.24), vec2(-0.8, quay - 0.035), 0.005), sdSegment(p, vec2(0.55, -0.24), vec2(0.55, quay - 0.035), 0.005));
  const lantern = sdCircle(p.sub(vec2(LANTERN.x, LANTERN.y)), 0.022);

  const board = floor(p.y.div(0.045));
  let teak: N<'vec3'> = mix(rgb(TEAK[0]), rgb(TEAK[1]), hash1(board)).mul(noise(p.mul(vec2(28, 2))).mul(0.1).add(0.95));
  teak = mix(teak, rgb(SEAM), seams(p.y, 0.045, 0.0035));
  const roof = rgb(ROOF).mul(smoothstep(-CABIN.half.y, CABIN.half.y, c.y).mul(0.22).add(0.88)).mul(seams(c.x, 0.1, 0.003).mul(-0.25).add(1)).mul(coverage(cabin.add(0.022)).mul(0.18).add(0.9)).mul(noise(p.mul(24)).mul(0.08).add(0.96));
  const leaf = mix(rgb(LEAF[0]), rgb(LEAF[1]), noise(p.mul(46)).mul(0.5).add(0.5)).mul(round(plants, 0.04));

  const paint = new Paint();
  paint.over(soft(sdBox(raw.sub(vec2(drop.x, drop.y)), vec2(HULL.x / 2, HULL.y / 2), 0.17), 0.05).mul(0.5), vec3(0));
  paint.over(coverage(lines), lit(rgb(ROPE)));
  paint.over(coverage(bollards), lit(rgb(IRON)));
  paint.over(coverage(hull.sub(0.01)), rgb(INK));
  paint.over(coverage(hull), lit(rgb(HULL_C)));
  paint.over(coverage(fenders), lit(rgb(0x121212)));
  paint.over(coverage(abs(hull.add(0.021)).sub(0.012)), lit(rgb(GUNWALE)));
  paint.over(coverage(deck), lit(teak));
  paint.over(coverage(pots), lit(rgb(POT)));
  paint.over(coverage(plants), lit(leaf));
  paint.over(coverage(chairs), lit(rgb(0x3f3a33)));
  paint.over(coverage(table), lit(rgb(0x8a6a4a)));
  paint.over(coverage(buoy), lit(mix(rgb(0xd8352a), rgb(0xf2ede4), stripes)));
  paint.over(coverage(cabin.sub(0.014)), rgb(0x121314));
  paint.over(coverage(cabin), lit(roof));
  paint.over(coverage(sky.sub(0.008)), lit(rgb(0x18191b)));
  paint.over(coverage(sky), rgb(SKYLIGHT).mul(flare.mul(1.6).add(1.05)));
  paint.over(coverage(chimney), lit(rgb(0x1a1b1d).mul(coverage(abs(chimney.add(0.01)).sub(0.006)).mul(2.2).add(1))));
  paint.over(coverage(plank), lit(mix(rgb(TEAK[1]), rgb(SEAM), seams(p.y, 0.035, 0.004))));
  paint.over(coverage(lantern.sub(0.008)), rgb(IRON));
  paint.over(coverage(lantern), rgb(SKYLIGHT).mul(flare.mul(2).add(2.4)));
  return { material: paint.material(), flare };
}

const DECKPLATE = 0x8b9094;
const SAFETY = 0xd9ae38;
const CANOPY = 0xd8d1bf;
const PILE = 0x2c2f33;

export const LANDING = new THREE.Vector2(1.6, 1.0);
export const PONTOON = new THREE.Vector2(1.24, 0.38);

export function landingMaterial(lit: Lit, drop: THREE.Vector2): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(vec2(LANDING.x, LANDING.y)), half = vec2(PONTOON.x / 2, PONTOON.y / 2), quay = -(PONTOON.y / 2 + MOOR);
  const shift = vec2(drop.x, drop.y);
  const pontoon = sdBox(p, half, 0.03);
  const pileAt = (q: N<'vec2'>) => sdCircle(vec2(abs(q.x).sub(PONTOON.x / 2 + 0.07), q.y), 0.05);
  const ramp = sdBox(p.sub(vec2(0, (quay - PONTOON.y / 2) / 2 + 0.01)), vec2(0.1, (quay + PONTOON.y / 2) / -2 + 0.05), 0.008);
  const canopyAt = (q: N<'vec2'>) => sdBox(q.sub(vec2(0.3, 0.04)), vec2(0.22, 0.115), 0.02);
  const canopy = canopyAt(p);
  const bench = sdBox(p.sub(vec2(-0.32, 0.1)), vec2(0.15, 0.035), 0.01);
  const railing = max(abs(pontoon.add(0.03)).sub(0.006), max(p.y, abs(p.x).sub(0.5)).negate());
  const diamond = abs(fract(p.x.add(p.y).mul(22)).sub(0.5)).add(abs(fract(p.x.sub(p.y).mul(22)).sub(0.5)));
  const plate = rgb(DECKPLATE).mul(smoothstep(0.3, 0.55, diamond).mul(0.12).add(0.92)).mul(noise(p.mul(9)).mul(0.08).add(0.96));
  const stripe = coverage(abs(pontoon.add(0.012)).sub(0.012));
  const cloth = rgb(CANOPY).mul(smoothstep(-0.12, 0.12, p.y.sub(0.04)).mul(0.18).add(0.85)).mul(coverage(abs(canopy.add(0.018)).sub(0.006)).mul(-0.3).add(1));

  const paint = new Paint();
  paint.over(soft(sdBox(p.sub(shift), half, 0.03), 0.04).mul(0.45), vec3(0));
  paint.over(soft(pileAt(p.sub(shift)), 0.03).mul(0.45), vec3(0));
  paint.over(coverage(pontoon.sub(0.01)), rgb(INK));
  paint.over(coverage(pontoon), lit(mix(plate, rgb(SAFETY), stripe)));
  paint.over(coverage(ramp.sub(0.008)), rgb(INK));
  paint.over(coverage(ramp), lit(mix(rgb(DECKPLATE).mul(0.9), rgb(0x3a3d40), seams(p.y, 0.028, 0.004))));
  paint.over(coverage(railing), lit(rgb(0xc9ccce)));
  paint.over(coverage(bench), lit(mix(rgb(TEAK[1]), rgb(SEAM), seams(p.y.sub(0.1), 0.023, 0.003))));
  paint.over(soft(canopyAt(p.sub(shift.mul(0.9))), 0.035).mul(0.4), vec3(0));
  paint.over(coverage(canopy.sub(0.008)), rgb(INK));
  paint.over(coverage(canopy), lit(cloth));
  paint.over(coverage(pileAt(p)), lit(rgb(PILE).mul(coverage(ring(vec2(abs(p.x).sub(PONTOON.x / 2 + 0.07), p.y), 0.036, 0.007)).mul(1.6).add(1))));
  return paint.material();
}

export const SIGNAL = 0.24;

export function signalMaterial(lit: Lit, tint: Tint): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(SIGNAL);
  const base = sdBox(p, vec2(0.034), 0.008);
  const head = sdCircle(p.sub(vec2(0, 0.012)), 0.03);
  const hood = max(ring(p.sub(vec2(0, 0.012)), 0.038, 0.007), float(0.012).sub(p.y));
  const paint = new Paint();
  paint.over(soft(sdBox(p.sub(vec2(0.02, -0.03)), vec2(0.036), 0.01), 0.02).mul(0.45), vec3(0));
  paint.over(coverage(base.sub(0.008)), rgb(INK));
  paint.over(coverage(base), lit(rgb(0x2b2f33)));
  paint.over(coverage(hood), lit(rgb(0x1a1c1e)));
  paint.over(coverage(head), tint.mul(2.4).add(0.08));
  return paint.material();
}
