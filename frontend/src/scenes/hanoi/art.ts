import * as THREE from 'three/webgpu';
import { abs, clamp, exp, float, floor, fract, fwidth, instancedDynamicBufferAttribute, length, max, min, mix, positionWorld, select, smoothstep, sqrt, step, uv, vec2, vec3 } from 'three/tsl';
import type { Light2D } from '@kit/light2d';
import { coverage, fractal, rgb, sdBox, sdSegment } from '@kit/shapes2d';
import { HANOI_PEGS } from '@sim/envs/hanoi';

type N<T extends string> = THREE.Node<T>;
type Weights = THREE.UniformNode<'vec3', THREE.Vector3>;
export type Level = THREE.UniformNode<'float', number>;

export const TILT = 0.18;
const PITCH = 3.3;
const PEG = 0.1;
export const RIM = 1.5;
export const TOP = (RIM + 0.2) * TILT;
export const FRONT = 1.3;
export const HALF = PITCH + RIM + 0.45;
const KEY_PITCH = 1.28;
export const KEY_W = 1.02;
export const KEY_H = 0.78;
export const PLATE_Z = TOP + 0.085;
export const PLATE = new THREE.Vector2(0.84, 0.075);
export const KEY_Z = TOP + 0.1 + FRONT / 2;
export const GOLD = 0xffd166;
export const GILT = 0xf2c35b;

const CORE = 0.5;
const HOLE = 0.135;
const SOCKET = 0.25;
const JUNCTION = TOP + 0.35;
const GILD = 0.3;
const KEY_PAD = 0.24;
const LIGHT = [-0.45, 0.8] as const;
const WALL = [0x392c25, 0x211915] as const;
const TABLE = [0x3d2718, 0x24160d] as const;
const WALNUT = [0x7a5233, 0x5a3a22] as const;
const FACE = [0x4d311d, 0x3a2414] as const;
const BED = 0x21150c;
const BRASS = 0xb89250;
const MAPLE = 0xd8b88a;
const HOLLOW = 0x1b120b;
const IVORY = [0xf4ecda, 0xdccdae] as const;
const INK = 0x3b2a1d;
const PIP = 0x2c2119;
const WARM = 0xffb454;

export interface Layout { disks: number; keys: number; height: number; tip: number; fly: number }

export function layoutOf(disks: number, keys: number): Layout {
  const height = Math.min(0.46, 2.72 / disks), tip = -(disks * height + 0.5);
  return { disks, keys, height, tip, fly: tip - 0.32 };
}

export const radiusOf = (layout: Layout, size: number): number => CORE + (RIM - CORE) * size / Math.max(1, layout.disks - 1);
export const pegX = (peg: number): number => (peg - 1) * PITCH;
export const keyX = (layout: Layout, key: number): number => (key - (layout.keys - 1) / 2) * KEY_PITCH;
export const diskQuad = (layout: Layout) => ({ width: 2 * RIM + 0.4, depth: layout.height + 2 * RIM * TILT + 0.4 });
export const keyQuad = { width: KEY_W + 2 * KEY_PAD, depth: KEY_H + 2 * KEY_PAD };
export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const basic = (additive = false) => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, ...(additive ? { blending: THREE.AdditiveBlending } : {}) });
const up = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const squash = (p: N<'vec2'>): N<'vec2'> => vec2(p.x, p.y.div(TILT));
const hairline = (d: N<'float'>, width: number): N<'float'> => clamp(float(width).sub(abs(d).div(max(fwidth(d), 1e-6))), 0, 1);
const pegIndex = (x: N<'float'>): N<'float'> => clamp(floor(x.div(PITCH).add(HANOI_PEGS / 2)), 0, HANOI_PEGS - 1);
const weightOf = (weights: Weights, index: N<'float'>): N<'float'> => select(index.lessThan(0.5), weights.x, select(index.lessThan(1.5), weights.y, weights.z));
const composite = (color: N<'vec3'>, face: N<'float'>, under: N<'float'>) => {
  const alpha = face.add(face.oneMinus().mul(under));
  return { color: color.mul(face).div(max(alpha, 1e-4)), alpha };
};

const cylinder = (s: N<'float'>) => {
  const facing = sqrt(max(s.mul(s).oneMinus(), 0));
  return { facing, lambert: max(s.mul(LIGHT[0]).add(facing.mul(LIGHT[1])), 0) };
};

function pegAlbedo(px: N<'float'>, y: N<'float'>, tipY: number, gild: N<'float'>): N<'vec3'> {
  const s = clamp(px.div(PEG), -1, 1), { lambert } = cylinder(s), shine = exp(s.add(0.42).mul(s.add(0.42)).mul(-28));
  const wood = rgb(MAPLE).mul(lambert.mul(0.62).add(0.46)).add(vec3(shine.mul(0.16)));
  const gold = rgb(GILT).mul(lambert.mul(0.7).add(0.55)).add(vec3(shine.mul(0.45)));
  return mix(wood, gold, smoothstep(tipY - GILD - 0.025, tipY - GILD + 0.025, y).mul(gild));
}

export function roomMaterial(light: Light2D): THREE.MeshBasicNodeMaterial {
  const material = basic(), p = up(), x = p.x, y = p.y;
  const plaster = fractal(p.mul(0.55), 4).mul(0.5).add(0.5);
  const wall = mix(rgb(WALL[0]), rgb(WALL[1]), smoothstep(JUNCTION, JUNCTION + 5, y)).mul(plaster.mul(0.14).add(0.93));
  const grain = fractal(vec2(x.mul(0.2), y.mul(6)), 5).mul(0.5).add(0.5);
  const table = mix(rgb(TABLE[1]), rgb(TABLE[0]), grain).mul(smoothstep(-TOP - FRONT - 3, JUNCTION, y).mul(0.35).add(0.65));
  const seam = exp(abs(y.sub(JUNCTION)).mul(-26)).mul(0.4);
  const footprint = sdBox(p.sub(vec2(0, -FRONT / 2)), vec2(HALF, TOP + FRONT / 2), 0.1);
  const contact = smoothstep(0.45, 0, footprint).mul(0.6);
  const surface = mix(table, wall, coverage(float(JUNCTION).sub(y))).mul(seam.add(contact).oneMinus());
  material.colorNode = light.lit(surface);
  material.opacityNode = float(1);
  return material;
}

export function consoleMaterial(light: Light2D, layout: Layout): THREE.MeshBasicNodeMaterial {
  const material = basic(), p = up(), x = p.x, y = p.y;
  const topD = sdBox(p, vec2(HALF, TOP), 0.06), frontD = sdBox(p.sub(vec2(0, -TOP - FRONT / 2)), vec2(HALF, FRONT / 2), 0.1);
  const body = coverage(min(topD, frontD)), onTop = coverage(topD);
  const grain = fractal(vec2(x.mul(0.35), y.mul(9)), 5).mul(0.5).add(0.5);
  const lid = mix(rgb(WALNUT[1]), rgb(WALNUT[0]), grain).mul(smoothstep(-TOP, TOP, y).mul(-0.12).add(1.08));
  const front = mix(rgb(FACE[1]), rgb(FACE[0]), grain).mul(smoothstep(-TOP - FRONT, -TOP, y).mul(0.22).add(0.78));
  const bevel = exp(abs(y.add(TOP)).mul(-60)).mul(0.5).add(exp(abs(y.add(TOP + FRONT)).mul(-40)).mul(-0.3));
  const index = pegIndex(x), px = x.sub(index.sub(1).mul(PITCH)), r = length(squash(vec2(px, y)));
  const brass = rgb(BRASS).mul(smoothstep(SOCKET, PEG, r).mul(0.3).add(0.85));
  const socket = coverage(r.sub(SOCKET)).mul(onTop), hollow = coverage(r.sub(PEG + 0.025)).mul(onTop);
  const plateAt = vec2(px, y.add(PLATE_Z)), plateD = sdBox(plateAt, vec2(PLATE.x / 2, PLATE.y / 2), 0.03), plate = coverage(plateD);
  const plated = rgb(BRASS).mul(smoothstep(-PLATE.y / 2, PLATE.y / 2, plateAt.y).mul(0.35).add(0.7)).mul(smoothstep(-0.02, 0, plateD).mul(-0.3).add(1));
  const bedSize = vec2(layout.keys * KEY_PITCH / 2 + 0.12, KEY_H / 2 + 0.15), bedAt = p.sub(vec2(0, -KEY_Z));
  const bedD = sdBox(bedAt, bedSize, 0.16), bed = coverage(bedD);
  const recess = rgb(BED).mul(smoothstep(bedSize.y.negate(), bedSize.y, bedAt.y).mul(-0.4).add(1.1)).mul(smoothstep(-0.14, 0, bedD).mul(0.35).add(0.65));
  const wood = mix(front, lid, onTop).add(vec3(bevel.mul(0.16)));
  const albedo = mix(mix(mix(mix(wood, recess, bed), plated, plate), brass, socket), rgb(HOLLOW), hollow);
  material.colorNode = light.lit(albedo);
  material.opacityNode = body;
  return material;
}

export function pegMaterial(light: Light2D, layout: Layout, weights: Weights): THREE.MeshBasicNodeMaterial {
  const material = basic(), p = up(), tipY = -layout.tip;
  const index = pegIndex(p.x), px = p.x.sub(index.sub(1).mul(PITCH));
  const foot = p.y.add(sqrt(max(float(PEG * PEG).sub(px.mul(px)), 0)).mul(TILT));
  const column = coverage(max(sdSegment(vec2(px, p.y), vec2(0, 0), vec2(0, tipY - PEG), PEG), foot.negate()));
  material.colorNode = light.lit(pegAlbedo(px, p.y, tipY, weightOf(weights, index)));
  material.opacityNode = column;
  return material;
}

export function ghostMaterial(layout: Layout, weights: Weights): THREE.MeshBasicNodeMaterial {
  const material = basic(), p = up();
  const index = pegIndex(p.x), px = p.x.sub(index.sub(1).mul(PITCH)), weight = weightOf(weights, index);
  let line: N<'float'> = float(0), fill: N<'float'> = float(0);
  for (let level = 0; level < layout.disks; level++) {
    const radius = radiusOf(layout, layout.disks - 1 - level), q = squash(vec2(px, p.y.sub((level + 1) * layout.height)));
    const d = sdSegment(q, vec2(0, 0), vec2(0, -layout.height / TILT), radius);
    line = max(line, hairline(d, 1.1));
    fill = max(fill, coverage(d));
  }
  material.colorNode = rgb(GOLD).mul(1.15);
  material.opacityNode = line.mul(0.5).add(fill.mul(0.07)).mul(weight);
  return material;
}

export function diskMaterial(light: Light2D, layout: Layout, shape: THREE.InstancedBufferAttribute, body: THREE.InstancedBufferAttribute, weights: Weights): THREE.MeshBasicNodeMaterial {
  const material = basic(), a = instancedDynamicBufferAttribute(shape, 'vec4') as N<'vec4'>, albedo = instancedDynamicBufferAttribute(body, 'vec3') as N<'vec3'>;
  const rx = a.x, threaded = a.y, grounded = a.z, flare = a.w, H = layout.height, { width, depth } = diskQuad(layout);
  const p = uv().sub(0.5).mul(vec2(width, depth)).sub(vec2(0, H / 2)), q = squash(p);
  const silhouette = sdSegment(q, vec2(0, 0), vec2(0, -H / TILT), rx), face = coverage(silhouette);
  const top = coverage(length(q).sub(rx));
  const s = clamp(p.x.div(rx), -1, 1), { facing, lambert } = cylinder(s);
  const drop = clamp(p.y.div(-H), 0, 1);
  const band = lambert.mul(0.66).add(0.4).mul(drop.mul(-0.2).add(1.04));
  const streak = exp(s.add(0.5).mul(s.add(0.5)).mul(-40)).mul(0.2);
  const lip = exp(max(facing.mul(rx).mul(-TILT).sub(p.y), 0).mul(-40)).mul(top.oneMinus()).mul(0.22);
  const sheen = exp(length(q.sub(vec2(rx.mul(-0.35), rx.mul(0.3)))).div(rx).mul(-3.2)).mul(0.14);
  const lacquer = albedo.mul(mix(band, float(1.1).add(sheen), top)).add(vec3(streak.add(lip).mul(top.oneMinus())));
  const hole = coverage(length(q).sub(HOLE)).mul(top);
  const index = pegIndex(positionWorld.x), tipY = -layout.tip;
  const foot = p.y.add(sqrt(max(float(PEG * PEG).sub(p.x.mul(p.x)), 0)).mul(TILT));
  const column = coverage(max(abs(p.x).sub(PEG), foot.negate())).mul(threaded).mul(step(layout.tip, positionWorld.z));
  const peg = pegAlbedo(p.x, positionWorld.z.negate(), tipY, weightOf(weights, index));
  const surface = mix(mix(lacquer, rgb(HOLLOW), hole), peg, column);
  const gilded = mix(surface, rgb(GILT).mul(1.2), flare.mul(0.32));
  const shadow = smoothstep(0.14, -0.05, length(q.sub(vec2(0.05, -H / TILT - 0.03))).sub(rx)).mul(0.5).mul(grounded);
  const out = composite(light.lit(gilded).add(rgb(GOLD).mul(flare.mul(0.22))), face, shadow);
  material.colorNode = out.color;
  material.opacityNode = out.alpha;
  return material;
}

export function keyMaterial(light: Light2D, look: THREE.InstancedBufferAttribute, glow: THREE.InstancedBufferAttribute, accent: number, refused: number, cue: number): THREE.MeshBasicNodeMaterial {
  const material = basic(), k = instancedDynamicBufferAttribute(look, 'vec4') as N<'vec4'>, g = instancedDynamicBufferAttribute(glow, 'vec4') as N<'vec4'>;
  const pips = k.x, from = floor(k.y.div(HANOI_PEGS)), to = k.y.sub(from.mul(HANOI_PEGS)), route = k.z, flip = max(k.w, 0.05), press = g.x, warm = g.y, cold = g.z, shimmer = g.w;
  const t = uv().sub(0.5).mul(vec2(keyQuad.width, keyQuad.depth)), c = t.add(vec2(0, press.mul(0.035)));
  const faceD = sdBox(c, vec2(KEY_W / 2, KEY_H / 2), 0.12), face = coverage(faceD);
  const shadow = smoothstep(0.1, -0.06, sdBox(t.sub(vec2(0.03, press.mul(0.02).sub(0.07))), vec2(KEY_W / 2, KEY_H / 2), 0.12)).mul(press.mul(-0.55).add(0.6));
  const edge = smoothstep(-0.07, 0, faceD), bevel = edge.mul(c.y.div(KEY_H / 2).mul(0.16).sub(0.04));
  const ivory = mix(rgb(IVORY[1]), rgb(IVORY[0]), smoothstep(-KEY_H / 2, KEY_H / 2, c.y)).mul(bevel.add(1)).mul(press.mul(-0.08).add(1));
  const dot = (i: number, j: number, visible: N<'float'>) => coverage(length(c.sub(vec2(-0.29 + 0.1 * i, 0.02 + 0.1 * j))).sub(0.033)).mul(visible);
  const odd = step(0.5, pips.sub(floor(pips.mul(0.5)).mul(2)));
  const pip = max(max(max(dot(0, 0, odd), max(dot(-1, 1, step(1.5, pips)), dot(1, -1, step(1.5, pips)))), max(dot(1, 1, step(3.5, pips)), dot(-1, -1, step(3.5, pips)))), max(dot(-1, 0, step(5.5, pips)), dot(1, 0, step(5.5, pips))));
  const gx = (peg: N<'float'>) => peg.sub(1).mul(0.15).add(0.17);
  let ticks: N<'float'> = float(0);
  for (let peg = 0; peg < HANOI_PEGS; peg++) ticks = max(ticks, coverage(sdSegment(c, vec2((peg - 1) * 0.15 + 0.17, -0.25), vec2((peg - 1) * 0.15 + 0.17, -0.17), 0.018)));
  const r = vec2(c.x, c.y.sub(0.02).div(flip).add(0.02)), fx = gx(from), tx = gx(to);
  const path = min(min(sdSegment(r, vec2(fx, -0.1), vec2(fx, 0.18), 0.017), sdSegment(r, vec2(fx, 0.18), vec2(tx, 0.18), 0.017)), sdSegment(r, vec2(tx, 0.18), vec2(tx, -0.07), 0.017));
  const arrow = min(sdSegment(r, vec2(tx, -0.09), vec2(tx.sub(0.055), -0.025), 0.017), sdSegment(r, vec2(tx, -0.09), vec2(tx.add(0.055), -0.025), 0.017));
  const glyph = coverage(min(path, arrow)).mul(route).mul(step(0.06, flip));
  const lamp = mix(mix(rgb(INK), rgb(accent).mul(1.35), warm), rgb(refused).mul(1.4), cold);
  const ink = mix(mix(ivory, rgb(PIP), pip), mix(rgb(INK), lamp, max(warm, cold)), max(glyph, ticks.mul(0.85)));
  const tinted = mix(ink, rgb(cue), shimmer.mul(0.28));
  const halo = exp(max(faceD, 0).mul(-16)).mul(smoothstep(KEY_PAD - 0.02, KEY_PAD * 0.5, faceD)).mul(max(max(warm, cold), shimmer));
  const tone = mix(mix(rgb(accent), rgb(refused), cold.div(max(warm.add(cold), 1e-3))), rgb(cue), shimmer.div(max(max(warm, cold).add(shimmer), 1e-3)));
  const under = max(shadow, halo), alpha = face.add(face.oneMinus().mul(under));
  const outside = tone.mul(1.6).mul(halo.div(max(under, 1e-4)));
  material.colorNode = light.lit(tinted).add(rgb(WARM).mul(warm.mul(0.12))).mul(face).add(outside.mul(face.oneMinus().mul(under))).div(max(alpha, 1e-4));
  material.opacityNode = alpha;
  return material;
}

export function trailMaterial(layout: Layout, tint: number, ends: THREE.UniformNode<'vec4', THREE.Vector4>, reach: Level, alpha: Level): THREE.MeshBasicNodeMaterial {
  const material = basic(true), p = up(), fly = float(-layout.fly + layout.height / 2);
  const a = vec2(ends.x, ends.y), b = vec2(ends.z, ends.w), ra = vec2(ends.x, fly), rb = vec2(ends.z, fly);
  const rise = abs(fly.sub(ends.y)), span = abs(ends.z.sub(ends.x)), total = rise.add(span).add(abs(fly.sub(ends.w))), limit = reach.mul(total);
  const leg = (from: N<'vec2'>, to: N<'vec2'>, offset: N<'float'> | number) => {
    const along = length(p.sub(from)).add(offset);
    return coverage(sdSegment(p, from, to, 0.028)).mul(step(0.42, fract(along.div(0.15)))).mul(step(along, limit));
  };
  const path = max(max(leg(a, ra, 0), leg(ra, rb, rise)), leg(rb, b, rise.add(span)));
  const tip = vec2(ends.z, -layout.tip + 0.14);
  const chevron = coverage(min(sdSegment(p, tip, tip.add(vec2(-0.15, 0.17)), 0.032), sdSegment(p, tip, tip.add(vec2(0.15, 0.17)), 0.032))).mul(step(0.999, reach));
  material.colorNode = mix(rgb(tint), vec3(1), 0.2).mul(1.4);
  material.opacityNode = max(path, chevron).mul(alpha);
  return material;
}

export function flashMaterial(tint: number, power: Level): THREE.MeshBasicNodeMaterial {
  const material = basic(true), t = uv().sub(0.5).mul(0.9);
  const cross = min(sdSegment(t, vec2(-0.13, -0.13), vec2(0.13, 0.13), 0.035), sdSegment(t, vec2(-0.13, 0.13), vec2(0.13, -0.13), 0.035));
  const ring = exp(length(t).mul(-7)).mul(smoothstep(0.44, 0.24, length(t))).mul(0.6);
  material.colorNode = mix(rgb(tint), vec3(1), 0.12).mul(1.7);
  material.opacityNode = min(coverage(cross).add(ring), 1).mul(power);
  return material;
}

