import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, floor, fract, length, max, min, mix, positionWorld, round, select, sin, smoothstep, step, texture, uv, uvec2, vec2, vec3 } from 'three/tsl';
import { coverage, grain, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { TOKENS } from '@kit/tokens';
import type { Kitchen } from '@sim/envs/overcooked';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const CHEF = 1;
export const PIECE = 0.9;
export const ACCENT = TOKENS.accents.OvercookedAI!;
export const COMMIS = 0x5fc8d8;
export const ONION = 0xe2ac52;
export const PLATE = 0xeef2f5;
export const BROTH = 0xf0983a;
export const GOLD = TOKENS.celebrationGold;
export const LAMP = 0xffb14d;
export const FLAME = 0xff8a3c;

const INK = 0x1f1813;
const TAU = 2 * Math.PI;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const oval = (p: N<'vec2'>, centre: N<'vec2'>, rx: number, ry: number) => length(p.sub(centre).div(vec2(rx, ry))).sub(1).mul(Math.min(rx, ry));

class Stack {
  color: N<'vec3'> = vec3(0);
  alpha: N<'float'> = float(0);

  over(color: N<'vec3'>, alpha: N<'float'>): this {
    this.color = color.mul(alpha).add(this.color.mul(alpha.oneMinus()));
    this.alpha = alpha.add(this.alpha.mul(alpha.oneMinus()));
    return this;
  }

  paint(material: THREE.MeshBasicNodeMaterial, fade: N<'float'> | number = 1): void {
    material.colorNode = this.color.div(max(this.alpha, 1e-4));
    material.opacityNode = this.alpha.mul(fade);
  }
}

interface Paint { d: N<'float'>; color: N<'vec3'> }

function onion(p: N<'vec2'>, r: number, tip: number): Paint {
  const c = Math.cos(tip - Math.PI / 2), s = Math.sin(tip - Math.PI / 2), k = r * 0.3;
  const q = vec2(p.x.mul(c).add(p.y.mul(s)), p.y.mul(c).sub(p.x.mul(s)));
  const bulb = length(q.sub(vec2(0, -0.06 * r))).sub(r * 0.94);
  const along = clamp(q.y.sub(r * 0.25).div(r), 0, 1);
  const spire = sdSegment(q, vec2(0, r * 0.25), vec2(0, r * 1.22), 0).sub(mix(float(r * 0.46), float(r * 0.05), along));
  const blend = clamp(spire.sub(bulb).mul(0.5 / k).add(0.5), 0, 1);
  const body = mix(spire, bulb, blend).sub(blend.mul(blend.oneMinus()).mul(k));
  const stem = sdSegment(q, vec2(0, r * 1.12), vec2(r * 0.08, r * 1.42), r * 0.05);
  const meridian = q.x.div(max(float(r * r).sub(q.y.add(0.06 * r).pow(2)), r * r * 0.02).sqrt());
  const vein = coverage(abs(fract(meridian.mul(2.2).add(0.5)).sub(0.5)).sub(0.05)).mul(smoothstep(r, 0.2 * r, length(q))).mul(0.16);
  let color: N<'vec3'> = mix(rgb(0xf6d08a), rgb(0x94511c), smoothstep(0.05, 1.2, length(p.sub(vec2(-0.35 * r, 0.38 * r))).div(r)));
  color = mix(color, rgb(0x7a4418), vein);
  color = mix(color, rgb(0x8b5220), smoothstep(r * 0.6, r * 1.1, q.y).mul(0.6));
  color = mix(color, rgb(0xfff4da), coverage(length(p.sub(vec2(-0.34 * r, 0.36 * r))).sub(0.17 * r)).mul(0.55));
  color = mix(color, rgb(0x5a3514), coverage(stem));
  const d = min(body, stem);
  return { d, color: mix(rgb(INK), color, coverage(d.add(r * 0.1))) };
}

function plate(p: N<'vec2'>, r: number): Paint {
  const d = length(p).sub(r), well = length(p).sub(r * 0.64);
  const lightAt = length(p.sub(vec2(-0.3 * r, 0.32 * r))).div(r);
  let color: N<'vec3'> = mix(rgb(0xfbfcfd), rgb(0xbfc7d0), smoothstep(0.1, 1.35, lightAt));
  const hollow = mix(rgb(0xd5dce3), rgb(0xf6f8fa), smoothstep(-0.9, 0.9, p.x.add(p.y.negate()).div(r).negate()));
  color = mix(color, hollow, coverage(well));
  color = mix(color, rgb(0x6f8db0), coverage(abs(length(p).sub(r * 0.82)).sub(r * 0.022)).mul(0.75));
  return { d, color: mix(rgb(0x4d5660), color, coverage(d.add(r * 0.06))) };
}

function broth(p: N<'vec2'>, r: number, time: N<'float'>, heat: N<'float'>): Paint {
  const d = length(p).sub(r), angle = atan(p.y, p.x), radial = length(p).div(r);
  const swirl = sin(angle.mul(3).add(radial.mul(9)).sub(time.mul(1.3))).mul(0.5).add(0.5);
  let color: N<'vec3'> = mix(rgb(0xc2601c), rgb(0xf6a948), smoothstep(1.1, 0.1, length(p.sub(vec2(-0.25 * r, 0.3 * r))).div(r)));
  color = color.mul(swirl.mul(0.12).add(0.94)).mul(heat.mul(0.25).add(1));
  const bits = [[0.35, 0.2, 0.4], [-0.3, -0.35, 2.2], [-0.4, 0.3, 4.1], [0.25, -0.4, 5.3]] as const;
  for (const [x, y, a] of bits) {
    const q = p.sub(vec2(x * r, y * r)), ring = abs(length(q).sub(r * 0.17)).sub(r * 0.045);
    const cut = step(0, dot(q, vec2(Math.cos(a), Math.sin(a))));
    color = mix(color, rgb(0xfbe3b0), coverage(ring).mul(cut).mul(0.85));
  }
  for (const [x, y] of [[-0.05, 0.12], [0.12, -0.06], [-0.18, -0.12]] as const) color = mix(color, rgb(0x4f8f3a), coverage(sdCircle(p.sub(vec2(x * r * 2, y * r * 2)), r * 0.09)));
  return { d, color };
}

function steam(p: N<'vec2'>, time: N<'float'>, scale: number): N<'float'> {
  let alpha: N<'float'> = float(0);
  for (let k = 0; k < 4; k++) {
    const phase = fract(time.mul(0.38).add(k / 4));
    const centre = vec2(sin(phase.mul(4.5).add(k * 2.1)).mul(0.055 * scale).add((k - 1.5) * 0.035 * scale), phase.mul(0.32 * scale));
    const puff = smoothstep(phase.mul(0.07 * scale).add(0.035 * scale), 0, length(p.sub(centre)));
    alpha = max(alpha, puff.mul(sin(phase.mul(Math.PI))).mul(0.62));
  }
  return alpha;
}

export interface Ground { material: THREE.MeshBasicNodeMaterial; textures: THREE.Texture[] }

export function groundMaterial(kitchen: Kitchen, lit: Lit): Ground {
  const { w, h } = kitchen;
  const data = new Uint8Array(w * h * 4);
  kitchen.tiles.forEach((code, i) => {
    data[4 * i] = code;
    data[4 * i + 3] = 255;
  });
  const map = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  map.magFilter = THREE.NearestFilter;
  map.minFilter = THREE.NearestFilter;
  map.generateMipmaps = false;
  map.needsUpdate = true;
  const tiles = texture(map);

  const q = vec2(positionWorld.x.add(w / 2), positionWorld.z.add(h / 2)), cell = floor(q), local = fract(q).sub(0.5);
  const probe = (dx: number, dy: number) => {
    const c = cell.add(vec2(dx, dy));
    const inside = step(0, c.x).mul(step(0, c.y)).mul(step(c.x, w - 1)).mul(step(c.y, h - 1));
    const code = round(tiles.load(uvec2(clamp(c, vec2(0), vec2(w - 1, h - 1)))).r.mul(255));
    return { solid: inside.mul(step(0.5, code)), open: inside.mul(step(code, 0.5)) };
  };
  const cast = vec2(0.1, 0.14);
  let near: N<'float'> = float(9), far: N<'float'> = float(9), shade: N<'float'> = float(9);
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const { solid, open } = probe(dx, dy), at = local.sub(vec2(dx, dy));
    near = min(near, select(solid.greaterThan(0.5), sdBox(at, vec2(0.5)), float(9)));
    far = min(far, select(open.greaterThan(0.5), sdBox(at, vec2(0.5)), float(9)));
    shade = min(shade, select(solid.greaterThan(0.5), sdBox(at.sub(cast), vec2(0.5)), float(9)));
  }
  const walkable = probe(0, 0).open;
  const centre = vec2(w / 2, h / 2), outline = sdBox(q.sub(centre), centre);

  const hashed = grain(cell), mott = noise(q.mul(1.3).add(0.37)).mul(0.5).add(0.5), fine = noise(q.mul(19).add(0.61)).mul(0.5).add(0.5);
  let tile: N<'vec3'> = mix(rgb(0x9a5b45), rgb(0xb06c52), hashed.x).mul(mott.mul(0.14).add(0.93)).mul(fine.mul(0.07).add(0.965));
  tile = tile.mul(smoothstep(0.66, 0.92, noise(q.mul(37).add(3.7)).mul(0.5).add(0.5)).mul(-0.18).add(1));
  const face = sdBox(local, vec2(0.458), 0.05);
  tile = tile.mul(smoothstep(-0.07, 0, face).mul(clamp(local.x.add(local.y).mul(-4), -1, 1)).mul(0.16).add(1));
  const glaze = smoothstep(0.16, 0, abs(local.x.sub(local.y).add(hashed.y.sub(0.5).mul(0.6)))).mul(smoothstep(0, -0.1, face)).mul(hashed.x.mul(0.03).add(0.012));
  tile = tile.mul(local.x.add(local.y).mul(-0.12).add(1)).add(glaze);
  let floorColor: N<'vec3'> = mix(rgb(0xd8c8ad).mul(fine.mul(0.1).add(0.88)), tile, coverage(face));
  floorColor = floorColor.mul(smoothstep(-0.02, 0.28, shade).mul(0.5).add(0.5)).mul(smoothstep(0, 0.12, near).mul(0.22).add(0.78));

  const edge = min(far, outline.negate());
  const brushed = noise(vec2(q.x.mul(1.6), q.y.mul(110)).add(0.5)).mul(0.5).add(0.5), sheen = sin(q.x.add(q.y.mul(0.6)).mul(2.1)).mul(0.5).add(0.5);
  let steel: N<'vec3'> = mix(rgb(0x6f7982), rgb(0xb4bdc5), sheen.mul(0.55).add(brushed.mul(0.45)));
  steel = steel.mul(local.x.add(local.y).mul(-0.06).add(1));
  steel = mix(steel, rgb(0x4b535b), coverage(float(0.494).sub(max(abs(local.x), abs(local.y)))).mul(step(0.06, edge)).mul(0.6));
  steel = mix(rgb(0xf0f3f5), steel, smoothstep(0.012, 0.06, edge));
  steel = mix(rgb(0x2a2f34), steel, smoothstep(0.004, 0.016, edge));

  const slate = noise(q.mul(0.55)).mul(0.5).add(0.5), flag = sdBox(fract(q.div(1.5)).sub(0.5), vec2(0.49), 0.02);
  let back: N<'vec3'> = mix(rgb(0x1b1917), rgb(0x2a2622), slate).mul(coverage(flag).mul(0.22).add(0.78));
  back = back.mul(smoothstep(-0.1, 0.7, sdBox(q.sub(centre).sub(vec2(0.14, 0.18)), centre, 0.2)).mul(0.65).add(0.35));

  const inner = select(walkable.greaterThan(0.5), floorColor, steel);
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(mix(back, inner, coverage(outline)));
  return { material, textures: [map] };
}

export function crateMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5);
  const outer = sdBox(p, vec2(0.41), 0.05), inner = sdBox(p, vec2(0.345), 0.035);
  const shadow = smoothstep(-0.03, 0.05, sdBox(p.sub(vec2(0.03, -0.035)), vec2(0.4), 0.06)).oneMinus().mul(0.45);
  const across = max(abs(p.x), abs(p.y)), along = select(abs(p.x).greaterThan(abs(p.y)), p.y, p.x);
  const plank = noise(vec2(along.mul(5), across.mul(46))).mul(0.5).add(0.5);
  let wood: N<'vec3'> = mix(rgb(0x6f4524), rgb(0xb27b45), plank);
  wood = wood.mul(smoothstep(0.345, 0.41, across).mul(-0.25).add(1.1));
  wood = mix(wood, rgb(0x3b2412), coverage(abs(abs(p.x).sub(abs(p.y))).sub(0.006)).mul(step(0.345, across)).mul(0.8));
  for (const [x, y] of [[0, 0.378], [0, -0.378], [0.378, 0], [-0.378, 0]] as const) wood = mix(wood, rgb(0x2a2a2e), coverage(sdCircle(p.sub(vec2(x, y)), 0.011)));
  const stack = new Stack()
    .over(vec3(0), shadow)
    .over(lit(mix(rgb(INK), wood, coverage(outer.add(0.012)))), coverage(outer))
    .over(lit(rgb(0x24170d)), coverage(inner));
  const bulbs = [[-0.2, 0.19, 0.5], [0.01, 0.21, 2.2], [0.2, 0.18, 1.1], [-0.11, 0.0, 3.4], [0.1, 0.01, 0.3], [-0.21, -0.19, 4.3], [0.0, -0.2, 2.7], [0.2, -0.18, 5.6]] as const;
  for (const [x, y, tip] of bulbs) {
    const bulb = onion(p.sub(vec2(x, y)), 0.105, tip);
    const deep = smoothstep(0.38, 0.1, max(abs(p.x), abs(p.y))).mul(0.25).add(0.75);
    stack.over(lit(bulb.color.mul(deep)), coverage(bulb.d).mul(coverage(inner)));
  }
  stack.paint(material);
  return material;
}

export function rackMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const p = uv().sub(0.5);
  const tray = sdBox(p, vec2(0.41), 0.06);
  const shadow = smoothstep(-0.03, 0.05, sdBox(p.sub(vec2(0.03, -0.035)), vec2(0.4), 0.07)).oneMinus().mul(0.4);
  let base: N<'vec3'> = mix(rgb(0x7d868e), rgb(0xa9b2ba), smoothstep(0.4, -0.4, p.x.sub(p.y)));
  base = mix(rgb(0x3a4046), base, coverage(tray.add(0.014)));
  base = mix(base, rgb(0x5d656d), coverage(abs(sdBox(p, vec2(0.33), 0.04)).sub(0.006)).mul(0.6));
  const stack = new Stack().over(vec3(0), shadow).over(lit(base), coverage(tray));
  for (let k = 3; k >= 0; k--) {
    const dish = plate(p.sub(vec2(-0.025 + 0.018 * k, 0.025 - 0.018 * k)), 0.29);
    stack.over(vec3(0), smoothstep(0.03, -0.01, dish.d.sub(0.012)).mul(0.25));
    stack.over(lit(k === 0 ? dish.color : mix(dish.color, rgb(0x9aa4ae), 0.35)), coverage(dish.d));
  }
  stack.paint(material);
  return material;
}

export function potMaterial(lit: Lit, time: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aPot', 'vec4'), fx = attribute<'vec4'>('aPotFx', 'vec4');
  const onions = state.x, cook = state.y, ready = state.z, flame = state.w;
  const plop = fx.x, puff = fx.y, pulse = fx.z, fill = fx.w;
  const p = uv().sub(0.5), r = length(p), around = fract(atan(p.x, p.y).div(TAU).add(1)), normal = p.div(max(r, 1e-4));
  const toward = dot(normal, vec2(-0.6, 0.8));

  const shadow = smoothstep(-0.03, 0.06, length(p.sub(vec2(0.025, -0.035))).sub(0.38)).oneMinus().mul(0.4);
  let grate: N<'float'> = abs(r.sub(0.405)).sub(0.019);
  for (let k = 0; k < 4; k++) {
    const a = Math.PI / 4 + k * Math.PI / 2, x = Math.cos(a), y = Math.sin(a);
    grate = min(grate, sdSegment(p, vec2(x * 0.3, y * 0.3), vec2(x * 0.44, y * 0.44), 0.02));
  }
  const iron = mix(rgb(0x17191c), rgb(0x50565e), toward.mul(0.5).add(0.5));
  const cap = r.sub(0.43);

  const tongue = cos(around.mul(TAU * 20)).mul(0.5).add(0.5).pow(3), flicker = sin(time.mul(17).add(floor(around.mul(20)).mul(2.7))).mul(0.5).add(0.5);
  const reach = tongue.mul(flicker.mul(0.45).add(0.55)).mul(0.07).add(0.012).mul(flame);
  const blaze = smoothstep(reach.add(0.34), 0.335, r).mul(step(0.335, r)).mul(flame);
  const fire = mix(rgb(0x3a74ff), rgb(0xffa13a), smoothstep(0.345, 0.375, r)).mul(blaze.mul(0.9).add(0.5));

  const loop = (side: number) => abs(sdBox(p.sub(vec2(side * 0.375, 0)), vec2(0.055, 0.075), 0.045)).sub(0.017);
  const handles = min(loop(-1), loop(1));
  const pot = r.sub(0.34), lip = abs(r.sub(0.32)).sub(0.02), hollow = r.sub(0.3);
  let rim: N<'vec3'> = mix(rgb(0x5d666f), rgb(0xe9eef2), toward.mul(0.5).add(0.5));
  rim = mix(rim, rgb(0xffffff), smoothstep(0.75, 1, toward).mul(coverage(abs(r.sub(0.325)).sub(0.006))).mul(0.8));
  const wall = smoothstep(0.22, 0.3, r), rings = sin(r.mul(210)).mul(0.5).add(0.5).mul(0.05);
  const inside = mix(mix(rgb(0x444b53), rgb(0x2a2f35), r.div(0.3)).mul(rings.add(0.97)), mix(rgb(0x14171a), rgb(0x8a939c), toward.negate().mul(0.5).add(0.5)), wall);

  const stew = broth(p, 0.3, time, ready);
  const simmer = max(step(0.001, cook), ready).mul(fill);
  let contents: N<'vec3'> = mix(inside, stew.color, simmer.mul(coverage(r.sub(0.292))));
  for (let k = 0; k < 5; k++) {
    const phase = fract(time.mul(0.9).add(k * 0.37)), a = time.mul(0.35).add(k * 1.3), rad = 0.08 + 0.04 * k;
    const bubble = abs(length(p.sub(vec2(cos(a).mul(rad), sin(a).mul(rad)))).sub(phase.mul(0.022).add(0.008))).sub(0.004);
    contents = mix(contents, rgb(0xffd79a), coverage(bubble).mul(phase.oneMinus()).mul(simmer).mul(0.8));
  }
  const dissolve = smoothstep(0.4, 0, cook).mul(ready.oneMinus()).mul(fill);
  for (let k = 0; k < 3; k++) {
    const a = Math.PI / 2 + k * TAU / 3, grow = clamp(onions.sub(k), 0, 1);
    const bulb = onion(p.sub(vec2(Math.cos(a) * 0.115, Math.sin(a) * 0.115)).div(max(grow, 0.05)), 0.092, a + 1.2);
    contents = mix(contents, bulb.color, coverage(bulb.d.mul(grow)).mul(step(0.05, grow)).mul(dissolve));
  }
  contents = mix(contents, rgb(0xfff0c8), coverage(abs(r.sub(plop.mul(0.26))).sub(0.008)).mul(plop.oneMinus()).mul(step(0.001, plop)).mul(0.75));

  const track = abs(r.sub(0.468)).sub(0.016);
  const third = fract(around.mul(3)), lit3 = step(floor(around.mul(3)).add(0.5), onions);
  const gap = smoothstep(0.02, 0.05, min(third, third.oneMinus()));
  const counting = cook.lessThan(0.001).and(ready.lessThan(0.5));
  const arc = select(counting, lit3.mul(gap), max(step(around, cook), ready));
  const tint = select(counting, rgb(ONION), mix(rgb(FLAME), rgb(GOLD), max(cook, ready))).mul(pulse.mul(0.8).add(ready.mul(0.25)).add(1.15));
  const trackAlpha = select(counting, gap.mul(0.5), float(0.5));

  const vapour = steam(p.add(vec2(0, 0.04)), time, 1.15).mul(ready).mul(puff.mul(0.6).add(0.7)).mul(fill);

  new Stack()
    .over(vec3(0), shadow)
    .over(lit(rgb(0x121416)), coverage(cap).mul(0.55))
    .over(lit(iron), coverage(grate))
    .over(fire, blaze)
    .over(lit(mix(rgb(INK), rgb(0x59616a), coverage(handles.add(0.008)))), coverage(handles))
    .over(lit(rgb(INK)), coverage(pot))
    .over(lit(rim), coverage(lip))
    .over(mix(lit(contents), contents.mul(1.08), simmer.mul(0.6)), coverage(hollow))
    .over(rgb(0x0d0f12), coverage(track).mul(trackAlpha))
    .over(tint, coverage(track.add(0.004)).mul(arc))
    .over(rgb(0xffffff), vapour)
    .paint(material);
  return material;
}

export const LAMP_AT = 0.27;

export function passMaterial(lit: Lit, time: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aPass', 'vec4');
  const flare = state.x, bell = state.y;
  const p = uv().sub(0.5);
  const frame = sdBox(p, vec2(0.44), 0.05);
  const tray = sdBox(p.sub(vec2(0, 0.04)), vec2(0.36, 0.3), 0.05);
  const housing = (side: number) => sdBox(p.sub(vec2(side * 0.175, LAMP_AT)), vec2(0.14, 0.052), 0.035);
  const housings = min(housing(-1), housing(1));
  const element = (side: number) => sdBox(p.sub(vec2(side * 0.175, LAMP_AT - 0.045)), vec2(0.12, 0.018), 0.016);
  const elements = min(element(-1), element(1));
  const bellAt = vec2(0.3, -0.3), dome = length(p.sub(bellAt)).sub(0.075);

  let steel: N<'vec3'> = mix(rgb(0x8f98a1), rgb(0xd9dfe4), smoothstep(0.45, -0.45, p.x.sub(p.y)));
  steel = mix(rgb(0x2a2f34), steel, coverage(frame.add(0.014)));
  const rail = coverage(abs(p.y.add(0.4)).sub(0.024)).mul(coverage(abs(p.x).sub(0.4)));
  steel = mix(steel, mix(rgb(0x9a6f22), rgb(0xf0c867), smoothstep(0.4, -0.4, p.x)), rail);

  const warmth = smoothstep(-0.3, LAMP_AT, p.y).mul(0.5).add(0.5).mul(flare.mul(0.25).add(1));
  const brushed = noise(vec2(p.x.mul(3), p.y.mul(120))).mul(0.5).add(0.5);
  let shelf: N<'vec3'> = mix(rgb(0xb89468), rgb(0xf5deb0), warmth.mul(0.8).add(brushed.mul(0.2)));
  for (const offset of [-0.12, 0.02]) {
    const chevron = abs(abs(p.x).mul(0.6).add(p.y.sub(offset))).sub(0.012);
    shelf = mix(shelf, rgb(0xc9913e), coverage(chevron).mul(coverage(abs(p.x).sub(0.12))).mul(0.55));
  }
  const glow = rgb(LAMP).mul(exp(max(elements, 0).mul(-14)).mul(flare.mul(0.7).add(0.8)));
  const shimmer = sin(time.mul(3.1).add(p.x.mul(9))).mul(0.06).add(1);

  const brass = mix(rgb(0x6e4f16), rgb(0xf5d98a), smoothstep(0.085, -0.04, length(p.sub(bellAt).sub(vec2(-0.028, 0.03)))));
  const knob = sdCircle(p.sub(bellAt), 0.022);
  const waves = abs(length(p.sub(bellAt)).sub(bell.mul(0.1).add(0.085))).sub(0.007);

  new Stack()
    .over(vec3(0), smoothstep(-0.03, 0.04, sdBox(p.sub(vec2(0.02, -0.025)), vec2(0.43), 0.06)).oneMinus().mul(0.4))
    .over(lit(steel), coverage(frame))
    .over(mix(rgb(INK), lit(shelf).add(glow.mul(0.5)), coverage(tray.add(0.012))), coverage(tray))
    .over(lit(mix(rgb(0x1a1d21), rgb(0x565e66), smoothstep(-0.06, 0.06, p.y.sub(LAMP_AT)))), coverage(housings))
    .over(rgb(LAMP).mul(flare.mul(1.5).add(2.2)).mul(shimmer), coverage(elements))
    .over(vec3(0), smoothstep(0.06, -0.02, dome.sub(0.01)).mul(0.35))
    .over(lit(mix(rgb(INK), brass, coverage(dome.add(0.01)))), coverage(dome))
    .over(lit(rgb(0x3a2a10)), coverage(knob))
    .over(rgb(0xfff0c0).mul(1.4), coverage(waves).mul(bell.oneMinus()).mul(step(0.001, bell)))
    .paint(material);
  return material;
}

export function itemMaterial(lit: Lit, time: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aItem', 'vec4');
  const kind = state.x, fade = state.y, fresh = state.z, lift = state.w;
  const p = uv().sub(0.5).mul(PIECE);
  const isOnion = step(kind, 1.5), isPlate = step(1.5, kind), isSoup = step(2.5, kind);
  const bulb = onion(p, 0.12, 0.9), dish = plate(p, 0.19), stew = broth(p, 0.122, time, float(0));
  const cast = p.sub(vec2(lift.mul(0.05).add(0.018), lift.mul(-0.06).sub(0.024)));
  const silhouette = select(isOnion.greaterThan(0.5), length(cast).sub(0.12), length(cast).sub(0.19));
  const shadow = smoothstep(lift.mul(0.05).add(0.03), float(-0.01), silhouette).mul(lift.mul(-0.12).add(0.34));
  const vapour = steam(p.sub(vec2(0, 0.03)), time.add(kind.mul(0.37)), 0.9).mul(fresh).mul(isSoup);
  new Stack()
    .over(vec3(0), shadow)
    .over(lit(bulb.color), coverage(bulb.d).mul(isOnion))
    .over(lit(dish.color), coverage(dish.d).mul(isPlate))
    .over(mix(lit(stew.color), stew.color, 0.35), coverage(stew.d).mul(isSoup))
    .over(rgb(0xffffff), vapour)
    .paint(material, fade);
  return material;
}

export function chefMaterial(lit: Lit, time: Level): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aChef', 'vec4'), pose = attribute<'vec4'>('aPose', 'vec4');
  const role = state.x, fade = state.y, walk = state.z, carry = state.w;
  const heading = pose.x, reach = pose.y, fumble = pose.z, glowing = pose.w;
  const p = uv().sub(0.5).mul(CHEF);
  const c = cos(heading), s = sin(heading);
  const local = (x: number, y: number) => vec2(c.mul(x).add(s.mul(y)), c.mul(y).sub(s.mul(x)));
  const light = local(-0.6, 0.8), cast = local(0.05, -0.065);
  const hue = select(role.lessThan(0.5), rgb(ACCENT), rgb(COMMIS));

  const stride = sin(walk.mul(TAU));
  const shoes = min(oval(p, vec2(float(-0.085), stride.mul(0.075).add(0.09)), 0.045, 0.066), oval(p, vec2(float(0.085), stride.mul(-0.075).add(0.09)), 0.045, 0.066));
  const body = oval(p, vec2(0, -0.005), 0.215, 0.16);
  const arms = max(carry, reach);
  const hand = (side: number) => mix(vec2(float(side * 0.212), stride.mul(-0.05 * side).sub(0.03)), vec2(float(side * 0.07), reach.mul(0.09).add(0.255)), arms);
  const shoulder = (side: number) => vec2(side * 0.165, 0.025);
  const sleeve = (side: number) => sdSegment(p, shoulder(side), mix(shoulder(side), hand(side), 0.82), 0.05);
  const sleeves = min(sleeve(-1), sleeve(1));
  const hands = min(sdCircle(p.sub(hand(-1)), 0.036), sdCircle(p.sub(hand(1)), 0.036));
  const knot = oval(p, vec2(0, 0.14), 0.048, 0.024);
  const face = oval(p, vec2(0, 0.058), 0.082, 0.07);
  const hatAt = vec2(0, -0.035), hat = length(p.sub(hatAt)).sub(0.125);
  const silhouette = min(min(min(shoes, body), min(sleeves, hands)), min(hat, face));

  const shadow = smoothstep(-0.06, 0.07, oval(p.sub(cast), vec2(0, 0), 0.25, 0.21)).oneMinus().mul(0.36);
  const bend = (d: N<'vec2'>, radius: number) => dot(d.div(radius), light).clamp(-1, 1);
  const jacket = mix(rgb(0xb8b2a8), rgb(0xfcfaf5), bend(p.sub(vec2(0, -0.005)), 0.22).mul(0.45).add(0.55));
  const cloth = mix(rgb(0xc9c3b9), rgb(0xf4f1ea), bend(p, 0.3).mul(0.4).add(0.6));
  const skin = rgb(0xe9b48d);
  const pleat = coverage(abs(fract(atan(p.x.sub(hatAt.x), p.y.sub(hatAt.y)).div(TAU).mul(11)).sub(0.5)).mul(0.07).sub(0.004)).mul(smoothstep(0.045, 0.07, length(p.sub(hatAt)))).mul(0.3);
  const crown = mix(hue.mul(0.55), mix(hue, rgb(0xffffff), 0.35), bend(p.sub(hatAt), 0.13).mul(0.5).add(0.5)).mul(pleat.oneMinus());
  const button = sdCircle(p.sub(hatAt), 0.024);
  const outlined = (d: N<'float'>, color: N<'vec3'>) => mix(rgb(INK), color, coverage(d.add(0.011)));

  let dizzy: N<'float'> = float(9);
  for (let k = 0; k < 3; k++) {
    const a = time.mul(4).add(k * TAU / 3), q = abs(p.sub(hatAt).sub(vec2(cos(a).mul(0.215), sin(a).mul(0.215))));
    dizzy = min(dizzy, q.x.sqrt().add(q.y.sqrt()).pow(2).sub(0.05));
  }
  const agent = step(role, 0.5);
  const ring = abs(length(p).sub(0.365)).sub(0.009);

  new Stack()
    .over(hue.mul(1.2), exp(length(p).mul(-6)).mul(agent).mul(glowing).mul(0.35).mul(smoothstep(0.47, 0.4, length(p))))
    .over(vec3(0), shadow)
    .over(lit(rgb(0x2a231e)), coverage(shoes))
    .over(lit(rgb(INK)), coverage(silhouette.sub(0.012)))
    .over(lit(jacket), coverage(body))
    .over(lit(hue.mul(0.9)), coverage(knot))
    .over(lit(outlined(face, mix(skin.mul(0.8), skin, bend(p.sub(vec2(0, 0.058)), 0.08).mul(0.5).add(0.5)))), coverage(face))
    .over(lit(outlined(sleeves, cloth)), coverage(sleeves))
    .over(lit(outlined(hands, skin)), coverage(hands))
    .over(lit(outlined(hat, mix(crown, mix(hue, rgb(0xffffff), 0.6), coverage(button)))), coverage(hat))
    .over(hue.mul(1.35), coverage(ring).mul(agent).mul(glowing).mul(0.85))
    .over(rgb(INK), coverage(dizzy.sub(0.012)).mul(fumble))
    .over(rgb(0xffe066).mul(1.6), coverage(dizzy).mul(fumble))
    .paint(material, fade);
  return material;
}
