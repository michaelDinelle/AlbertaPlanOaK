import * as THREE from 'three/webgpu';
import { abs, attribute, cos, exp, float, floor, fract, fwidth, length, max, min, mix, positionWorld, select, sign, sin, smoothstep, step, texture, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { fbm } from '@kit/textures';
import { TOKENS } from '@kit/tokens';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const POD = 1.26;
export const BOT = 1.12;
export const RING = 2.6;
export const SIGHT = 3.3;
export const BAY = 1.2;
export const TINTS = [TOKENS.accents.RWARE!, 0x86b4e2, 0xe3939b, 0xa2cf86] as const;
export const TEAL = 0x40dccb;
export const LAMP = 0xffe2b4;
export const DUST = 0xc3cad2;

const CONCRETE = [0x858c94, 0x676d75] as const;
const SLOT = 0xcfc7bd;
const PAINT = 0xd9dde2;
const HAZARD = 0xd9a321;
const INK = 0x16181c;
const MAT = 0x3b4047;
const CURB = 0x3c424a;
const NIGHT = 0x0d1014;
const STEEL = 0x353b44;
const RAIL = 0x59616b;
const FABRIC = [0x6a70b2, 0x4b508c] as const;
const MOUTH = 0x1b1d2e;
const CARTON = 0xb28b5b;
const PAPER = 0xd8d3c7;
const WRAP = 0x5d86b4;
const TYRE = 0x15171a;
const HOOD = 0x262b31;
const TAIL = 0xff5343;
const DESK = 0xc6cacf;
const VEST = 0xee7a22;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = (additive = false) => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, ...(additive ? { blending: THREE.AdditiveBlending } : {}) });

const turn = (p: N<'vec2'>, angle: N<'float'>) => vec2(p.x.mul(cos(angle)).sub(p.y.mul(sin(angle))), p.x.mul(sin(angle)).add(p.y.mul(cos(angle))));

const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));

class Layers {
  color: N<'vec3'> = vec3(0);
  alpha: N<'float'> = float(0);

  over(color: N<'vec3'>, alpha: N<'float'>): this {
    this.color = color.mul(alpha).add(this.color.mul(alpha.oneMinus()));
    this.alpha = alpha.add(this.alpha.mul(alpha.oneMinus()));
    return this;
  }

  paint(material: THREE.MeshBasicNodeMaterial, fade: N<'float'> | number): THREE.MeshBasicNodeMaterial {
    material.colorNode = this.color.div(max(this.alpha, 1e-4));
    material.opacityNode = this.alpha.mul(fade);
    return material;
  }
}

export interface Floor {
  material: THREE.MeshBasicNodeMaterial;
  cells: THREE.DataTexture;
  textures: THREE.Texture[];
}

export function floorMaterial(w: number, h: number, seed: number, lit: Lit): Floor {
  const cells = new THREE.DataTexture(new Uint8Array((w + 2) * (h + 2) * 4), w + 2, h + 2, THREE.RGBAFormat);
  cells.magFilter = cells.minFilter = THREE.NearestFilter;
  cells.needsUpdate = true;
  const mottle = fbm(seed ^ 0x3a71, { size: 256, frequency: 4, octaves: 5 }), grain = fbm(seed ^ 0x51c3, { size: 256, frequency: 32, octaves: 3 });
  const q = vec2(positionWorld.x.add(w / 2), positionWorld.z.add(h / 2)), cell = floor(q), local = fract(q).sub(0.5), c = fract(q);
  const size = vec2(w + 2, h + 2), at = (dx: number, dy: number) => texture(cells, cell.add(vec2(dx + 1.5, dy + 1.5)).div(size));
  const here = at(0, 0), west = at(-1, 0), east = at(1, 0), north = at(0, -1), south = at(0, 1);
  const pixel = fwidth(q.x);

  const mott = texture(mottle, q.mul(0.09)).r, broad = texture(mottle, q.mul(0.027).add(0.41)).r;
  const fine = texture(grain, q.mul(0.61)).r, speck = texture(grain, q.mul(2.7).add(0.23)).r;
  let ground: N<'vec3'> = mix(rgb(CONCRETE[1]), rgb(CONCRETE[0]), smoothstep(0.22, 0.78, mott.mul(0.6).add(broad.mul(0.4))));
  ground = ground.mul(fine.sub(0.5).mul(0.14).add(1)).mul(smoothstep(0.72, 0.84, speck).mul(0.1).add(1));
  const edge = float(0.5).sub(max(abs(local.x), abs(local.y)));
  ground = ground.mul(coverage(edge.sub(max(pixel.mul(0.5), 0.007))).mul(-0.2).add(1));

  const home = here.r;
  ground = mix(ground, ground.mul(rgb(SLOT)).mul(0.88), home);
  const f = abs(local);
  const bracket = min(sdSegment(f, vec2(0.41, 0.41), vec2(0.27, 0.41), 0.012), sdSegment(f, vec2(0.41, 0.41), vec2(0.41, 0.27), 0.012));
  ground = mix(ground, rgb(PAINT), coverage(bracket).mul(home).mul(0.55));

  const border = (layer: 'r' | 'g', inset: number) => {
    const open = (n: N<'vec4'>) => n[layer].lessThan(0.5);
    const x0 = select(open(west), float(inset), float(-0.2)), x1 = select(open(east), float(1 - inset), float(1.2));
    const y0 = select(open(north), float(inset), float(-0.2)), y1 = select(open(south), float(1 - inset), float(1.2));
    const run = (a: N<'vec2'>, b: N<'vec2'>, shown: N<'bool'>) => select(shown, sdSegment(c, a, b, 0), float(9));
    return min(min(run(vec2(x0, inset), vec2(x1, inset), open(north)), run(vec2(x0, 1 - inset), vec2(x1, 1 - inset), open(south))),
      min(run(vec2(inset, y0), vec2(inset, y1), open(west)), run(vec2(1 - inset, y0), vec2(1 - inset, y1), open(east))));
  };
  ground = mix(ground, rgb(PAINT), coverage(border('r', 0.055).sub(0.017)).mul(home).mul(0.75));

  const tag = sdBox(local, vec2(0.07), 0.012);
  ground = mix(ground, rgb(0xb1b7bd), coverage(tag).mul(home.oneMinus()).mul(0.5));
  ground = mix(ground, rgb(0x2a2e34), coverage(sdBox(local, vec2(0.042), 0.006)).mul(home.oneMinus()).mul(0.75));

  const goal = here.g, zone = border('g', 0);
  const pad = rgb(MAT).mul(fine.sub(0.5).mul(0.2).add(1));
  const stripes = mix(rgb(INK), rgb(HAZARD), step(0.5, fract(q.x.add(q.y).mul(3.2))));
  let dock: N<'vec3'> = mix(pad, stripes, coverage(zone.sub(0.1)));
  const ring = length(local).sub(0.21);
  dock = mix(dock, rgb(PAINT), coverage(abs(ring).sub(0.013)).mul(0.7));
  const chevron = min(sdSegment(local, vec2(-0.075, -0.03), vec2(0, 0.045), 0.017), sdSegment(local, vec2(0.075, -0.03), vec2(0, 0.045), 0.017));
  dock = mix(dock, rgb(HAZARD), coverage(chevron).mul(0.9));
  ground = mix(ground, dock, goal);

  const out = sdBox(q.sub(vec2(w / 2, h / 2)), vec2(w / 2, h / 2), 0), beyond = (reach: number) => coverage(float(reach).sub(out));
  ground = ground.mul(smoothstep(0, 0.5, out.negate()).mul(0.24).add(0.76));
  const hazard = mix(rgb(INK), rgb(HAZARD).mul(0.9), step(0.5, fract(q.x.sub(q.y).mul(3.2))));
  const along = select(abs(q.y.sub(h / 2)).sub(h / 2).greaterThan(abs(q.x.sub(w / 2)).sub(w / 2)), q.x, q.y);
  let curb: N<'vec3'> = rgb(CURB).mul(mott.sub(0.5).mul(0.24).add(1)).mul(fine.sub(0.5).mul(0.12).add(1));
  curb = mix(curb, rgb(0x6d757f), coverage(abs(out.sub(0.11)).sub(0.012)));
  curb = mix(curb, rgb(0x22262b), coverage(abs(fract(along).sub(0.5)).sub(0.01)).mul(0.6));
  const night = rgb(NIGHT).mul(mott.mul(0.3).add(0.85)).mul(smoothstep(0.46, 1.4, out).mul(-0.4).add(1));
  let albedo: N<'vec3'> = mix(ground, hazard, beyond(0));
  albedo = mix(albedo, curb, beyond(0.09));
  albedo = mix(albedo, night, beyond(0.46));

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(albedo);
  return { material, cells, textures: [cells, mottle, grain] };
}

export function podMaterial(lit: Lit, time: N<'float'>): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aPod', 'vec4');
  const fade = state.x, lift = state.y, wanted = state.z, ping = state.w;
  const p = uv().sub(0.5).mul(POD), pp = p.div(lift.mul(0.05).add(1));
  const shift = vec2(lift.mul(0.06).add(0.03), lift.mul(0.08).add(0.04).negate()), blur = lift.mul(0.06).add(0.035);
  const shadow = smoothstep(blur.negate(), blur, sdBox(p.sub(shift), vec2(0.4), 0.06)).oneMinus().mul(lift.mul(-0.16).add(0.5));

  const frame = sdBox(pp, vec2(0.42), 0.045);
  const side = sign(pp), inner = pp.sub(side.mul(0.205));
  const bin = sdBox(inner, vec2(0.183), 0.035), mouth = sdBox(inner, vec2(0.138), 0.026);
  const spin = side.x.add(1).mul(0.5).add(side.y.add(1)).mul(Math.PI / 2), goods = turn(inner, spin);
  const carton = sdBox(goods.sub(vec2(0.03, -0.035)), vec2(0.078, 0.056), 0.012);
  const paper = sdBox(goods.sub(vec2(-0.065, 0.055)), vec2(0.045, 0.04), 0.01);
  const wrap = sdCircle(goods.sub(vec2(0.07, 0.068)), 0.036);

  const light = pp.y.sub(pp.x).mul(0.32).add(1);
  let body: N<'vec3'> = rgb(STEEL).mul(light);
  body = mix(body, rgb(RAIL), coverage(abs(frame.add(0.016)).sub(0.007)).mul(0.7));
  const cloth = mix(rgb(FABRIC[1]), rgb(FABRIC[0]), smoothstep(-0.16, 0.16, inner.y.sub(inner.x))).mul(light);
  body = mix(body, cloth, coverage(bin));
  body = mix(body, cloth.mul(1.28), coverage(abs(bin.add(0.02)).sub(0.006)).mul(coverage(bin)).mul(0.6));
  let hollow: N<'vec3'> = mix(rgb(MOUTH), rgb(MOUTH).add(rgb(TEAL).mul(0.22)), wanted);
  hollow = mix(hollow, rgb(CARTON).mul(smoothstep(-0.06, 0.06, goods.y.sub(goods.x.mul(0.3))).mul(0.25).add(0.75)), coverage(carton));
  hollow = mix(hollow, rgb(CARTON).mul(0.62), coverage(abs(goods.x.sub(0.03)).sub(0.006)).mul(coverage(carton)));
  hollow = mix(hollow, rgb(PAPER), coverage(paper));
  hollow = mix(hollow, rgb(WRAP).mul(smoothstep(0.036, 0, length(goods.sub(vec2(0.058, 0.08)))).mul(0.5).add(0.7)), coverage(wrap));
  hollow = hollow.mul(smoothstep(0, 0.05, mouth.negate()).mul(0.45).add(0.55));
  body = mix(body, hollow, coverage(mouth));

  const pulse = sin(time.mul(3.4)).mul(0.25).add(0.75);
  const rim = coverage(abs(frame.add(0.012)).sub(0.012)).mul(wanted);
  const led = coverage(sdCircle(pp.sub(vec2(-0.355, 0.355)), 0.03));
  const glow = rgb(TEAL).mul(rim.mul(pulse.mul(1.6).add(0.4)).add(led.mul(wanted).mul(pulse.mul(2.6).add(0.8)))).add(rgb(TEAL).mul(coverage(mouth).mul(wanted).mul(0.35)));
  const flash = mix(rgb(TEAL), vec3(1), 0.5).mul(ping.mul(coverage(abs(frame.add(0.02)).sub(0.02)).mul(0.9)));

  return new Layers()
    .over(vec3(0), shadow)
    .over(lit(body).add(glow).add(flash), coverage(frame))
    .paint(material, fade);
}

export function robotMaterial(lit: Lit, time: N<'float'>): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aBot', 'vec4'), pose = attribute<'vec4'>('aPose', 'vec4');
  const tint = state.x, fade = state.y, lift = state.z, cargo = state.w;
  const drive = pose.x, under = pose.y, you = pose.z;
  const p = uv().sub(0.5).mul(BOT);

  const hue = select(tint.lessThan(0.5), rgb(TINTS[0]), select(tint.lessThan(1.5), rgb(TINTS[1]), select(tint.lessThan(2.5), rgb(TINTS[2]), rgb(TINTS[3]))));
  const shadow = smoothstep(-0.05, 0.05, sdBox(p.sub(vec2(0.03, -0.045)), vec2(0.33), 0.11)).oneMinus().mul(0.42);
  const chassis = sdBox(p, vec2(0.315), 0.095);
  const wheel = sdBox(vec2(abs(p.x).sub(0.326), p.y), vec2(0.03, 0.095), 0.02);
  const tread = step(0.5, fract(p.y.mul(18).add(drive.mul(4))));
  const bay = sdBox(p.sub(vec2(0, 0.24)), vec2(0.16, 0.036), 0.026), strip = sdBox(p.sub(vec2(0, 0.24)), vec2(0.126, 0.012), 0.009);
  const tails = sdCircle(vec2(abs(p.x).sub(0.21), p.y.add(0.256)), 0.024);
  const table = length(p), deck = table.sub(0.108), collar = abs(table.sub(0.12)).sub(0.01);
  const status = sdCircle(p, 0.034);

  const light = p.y.sub(p.x).mul(0.5).add(1);
  let body: N<'vec3'> = hue.mul(light).mul(0.92);
  body = mix(body, hue.mul(0.68), coverage(chassis.add(0.024).negate()));
  body = mix(body, hue.mul(0.72), coverage(abs(abs(p.y).sub(0.17)).sub(0.005)).mul(step(0, deck.sub(0.03))).mul(coverage(abs(p.x).sub(0.25))).mul(0.5));
  body = mix(body, rgb(HOOD), coverage(bay));
  body = mix(body, mix(rgb(0x2d3238), rgb(0x3f464e), lift).mul(light), coverage(deck));
  body = mix(body, mix(rgb(0x5b626b), hue.mul(1.1), lift.mul(0.75)), coverage(collar));
  body = mix(body, rgb(0x3a4149), coverage(status));
  body = body.mul(under.mul(-0.2).add(1));
  const tyres = rgb(TYRE).mul(tread.mul(0.35).add(0.75));

  const headlamp = rgb(LAMP).mul(coverage(strip).mul(under.mul(-0.6).add(2.4)));
  const brake = rgb(TAIL).mul(coverage(tails).mul(1.3));
  const beacon = select(cargo.greaterThan(1.5), rgb(TEAL).mul(2.6), rgb(LAMP).mul(1.5)).mul(coverage(status)).mul(step(0.5, cargo));
  const engaged = hue.mul(coverage(collar).mul(lift).mul(0.9));
  const ring = coverage(abs(table.sub(0.455)).sub(0.014)).mul(you);
  const halo = exp(abs(table.sub(0.455)).mul(-38)).mul(you).mul(0.35).mul(sin(time.mul(2.2)).mul(0.2).add(0.8));

  return new Layers()
    .over(rgb(TINTS[0]).mul(1.2), halo)
    .over(rgb(TINTS[0]).mul(1.6), ring.mul(0.9))
    .over(vec3(0), shadow)
    .over(lit(tyres), coverage(wheel))
    .over(mix(lit(body), body.mul(0.9), 0.45).add(headlamp).add(brake).add(beacon).add(engaged), coverage(chassis))
    .paint(material, fade);
}

export function ringMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet(true);
  const state = attribute<'vec4'>('aRing', 'vec4'), tint = attribute<'vec4'>('aTint', 'vec4');
  const progress = state.x, fade = state.y, reach = state.z, width = state.w;
  const p = uv().sub(0.5).mul(RING), radius = mix(tint.w, reach, progress.sqrt());
  const band = abs(length(p).sub(radius)).sub(width.mul(progress.oneMinus().mul(0.7).add(0.3)));
  material.colorNode = tint.xyz.mul(1.7);
  material.opacityNode = coverage(band).mul(progress.oneMinus().pow(2)).mul(fade);
  return material;
}

export function trackMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const fade = attribute<'vec4'>('aTrack', 'vec4').x;
  const p = uv().sub(0.5);
  const lines = sdSegment(vec2(abs(p.x), p.y), vec2(0.295, -0.36), vec2(0.295, 0.36), 0.024);
  const tread = step(0.45, fract(p.y.mul(16))).mul(0.45).add(0.55);
  material.colorNode = lit(rgb(TINTS[0]).mul(0.75)).add(rgb(TINTS[0]).mul(0.18));
  material.opacityNode = coverage(lines).mul(tread).mul(fade);
  return material;
}

export function sightMaterial(w: number, h: number): { material: THREE.MeshBasicNodeMaterial; level: Level } {
  const material = sheet(), level = uniform(0);
  const p = uv().sub(0.5).mul(SIGHT), f = abs(p);
  const corner = min(sdSegment(f, vec2(1.48, 1.48), vec2(1.2, 1.48), 0.018), sdSegment(f, vec2(1.48, 1.48), vec2(1.48, 1.2), 0.018));
  const ticks = min(sdSegment(f, vec2(0.5, 1.44), vec2(0.5, 1.52), 0.012), sdSegment(f, vec2(1.44, 0.5), vec2(1.52, 0.5), 0.012));
  const floorOnly = coverage(sdBox(vec2(positionWorld.x, positionWorld.z), vec2(w / 2, h / 2), 0));
  const inside = coverage(sdBox(p, vec2(1.5), 0.04)).mul(floorOnly);
  material.colorNode = rgb(TINTS[0]).mul(1.35);
  material.opacityNode = max(max(coverage(corner), coverage(ticks).mul(0.7)), inside.mul(0.022)).mul(level);
  return { material, level };
}

export interface Station {
  material: THREE.MeshBasicNodeMaterial;
  flash: Level;
  gold: Level;
}

export function stationMaterial(span: N<'float'>, lit: Lit, time: N<'float'>): Station {
  const material = sheet(), flash = uniform(0), gold = uniform(0);
  const half = span.mul(0.5), p = uv().sub(0.5).mul(vec2(span, BAY)), y = p.y.sub(BAY / 2);
  const bay = sdBox(p, vec2(half.sub(0.04), BAY / 2 - 0.02), 0.1);
  const sill = sdBox(vec2(p.x, y.add(0.06)), vec2(half.sub(0.12), 0.05), 0.01);
  const desk = sdBox(vec2(p.x, y.add(0.3)), vec2(half.sub(0.3), 0.13), 0.03);
  const screenAt = vec2(p.x.add(span.mul(0.2)), y);
  const screen = sdBox(screenAt.add(vec2(0, 0.34)), vec2(0.22, 0.026), 0.008), stand = sdBox(screenAt.add(vec2(0, 0.385)), vec2(0.05, 0.03), 0.01);
  const toteAt = vec2(p.x.sub(span.mul(0.24)), y.add(0.29));
  const tote = sdBox(toteAt, vec2(0.13, 0.09), 0.02), well = sdBox(toteAt, vec2(0.1, 0.062), 0.012);
  const bar = sdBox(vec2(p.x, y.add(0.175)), vec2(half.sub(0.36), 0.012), 0.006);
  const cells = step(0.5, fract(p.x.mul(5).add(0.25)));
  const person = vec2(p.x.sub(0.05).sub(sin(time.mul(0.9)).mul(0.015)), y.add(0.66));
  const shoulders = sdBox(person.sub(vec2(0, -0.01)), vec2(0.165, 0.062), 0.055);
  const straps = coverage(abs(abs(person.x).sub(0.075)).sub(0.014)).mul(coverage(shoulders));
  const arms = min(sdSegment(person, vec2(-0.14, 0.02), vec2(-0.1, 0.25), 0.03), sdSegment(person, vec2(0.14, 0.02), vec2(0.1, 0.25), 0.03));
  const hands = min(sdCircle(person.sub(vec2(-0.1, 0.26)), 0.032), sdCircle(person.sub(vec2(0.1, 0.26)), 0.032));
  const hat = sdCircle(person.sub(vec2(0, 0.015)), 0.083), peak = ellipse(person, [0, 0.085], [0.06, 0.035]);
  const ridge = coverage(abs(person.x).sub(0.012)).mul(coverage(hat.add(0.015)));

  let color: N<'vec3'> = rgb(0x4a5058).mul(smoothstep(-BAY / 2, BAY / 2, p.y).mul(0.25).add(0.75));
  color = mix(color, mix(rgb(INK), rgb(HAZARD), step(0.5, fract(p.x.add(y).mul(3.2)))), coverage(sill));
  color = mix(color, rgb(DESK).mul(y.add(0.3).mul(1.2).add(1)), coverage(desk));
  color = mix(color, rgb(0x8f959c), coverage(abs(desk.add(0.012)).sub(0.008)).mul(coverage(desk)));
  color = mix(color, rgb(0xe07b39), coverage(tote));
  color = mix(color, rgb(0x5a2f16), coverage(well));
  color = mix(color, rgb(0x1a1c20), coverage(min(screen, stand)));
  color = mix(color, rgb(0x2b3037), coverage(bar));
  color = mix(color, rgb(0x2c3646), coverage(arms));
  color = mix(color, rgb(0xd6a57e), coverage(hands));
  color = mix(color, mix(rgb(VEST), rgb(0xe9ecef), straps), coverage(shoulders));
  color = mix(color, rgb(0xefeee8).mul(smoothstep(0.09, -0.02, length(person.sub(vec2(-0.03, 0.05))))).mul(0.35).add(rgb(0xc9c8c2).mul(0.7)), coverage(min(hat, peak)));
  color = mix(color, rgb(0xa9a8a2), ridge);

  const hot = mix(rgb(TEAL), rgb(TOKENS.celebrationGold), gold);
  const glass = coverage(sdBox(screenAt.add(vec2(0, 0.332)), vec2(0.2, 0.014), 0.006));
  const monitor = mix(rgb(0x7fc8ff), hot, flash).mul(glass.mul(flash.mul(2).add(1.2)));
  const lights = mix(rgb(0x3b4a52), hot.mul(flash.mul(2.4).add(0.3)), cells).mul(coverage(bar.add(0.004)));
  new Layers().over(lit(color).add(monitor).add(lights), coverage(bay)).paint(material, 1);
  return { material, flash, gold };
}
