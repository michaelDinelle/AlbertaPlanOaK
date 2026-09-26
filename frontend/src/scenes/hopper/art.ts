import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dFdx, dFdy, dot, exp, float, floor, fract, length, max, min, mix, mod, positionGeometry, positionWorld, round, sign, sin, smoothstep, sqrt, step, uv, vec2 } from 'three/tsl';
import { fractal, grain, rgb, sdBox, sdSegment } from '@kit/shapes2d';
import type { Capsule } from './model';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Paint = (p: N<'vec2'>, px: N<'float'>) => readonly [N<'vec3'>, N<'float'>];

export interface Ink {
  lit: Lit;
  dpr: N<'float'>;
}

export const PALETTE = {
  night: 0x05070b,
  wall: 0x161a26,
  grid: 0xd8c9a8,
  gold: 0xffd166,
  danger: 0xff5a47,
  floor: 0x3c414c,
  lip: 0x9aa4b8,
  ruleDark: 0x14161b,
  ruleLight: 0xb9b09a,
  panel: 0x2a221d,
  edge: 0x160f0a,
  sole: 0x2a2522,
  steel: 0x323946,
  bolt: 0x9aa3b4,
  effort: 0xff9a3c,
  strobe: 0xf3e6c8,
  ghost: 0xff6b57,
  contact: 0xfff0c8,
  flash: 0xd6e6ff,
  safe: 0x86f0b4,
} as const;

export const GAUGE = { inner: 0.07, outer: 0.6 } as const;
export const PIN = 0.1;
export const SPARK = 0.26;
export const HALO = 0.006;

const TAU = 2 * Math.PI;

const footprint = (v: N<'float'>): N<'float'> => max(length(vec2(dFdx(v), dFdy(v))), 1e-7);
const fill = (d: N<'float'>, px: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(px)), 0, 1);
const wrap = (angle: N<'float'>): N<'float'> => angle.sub(round(angle.div(TAU)).mul(TAU));

function line(ink: Ink, d: N<'float'>, width: number, px: N<'float'>): N<'float'> {
  return clamp(ink.dpr.mul(width / 2).add(0.5).sub(abs(d).div(px)), 0, 1);
}

export const flat = (width: number, height: number): THREE.PlaneGeometry => new THREE.PlaneGeometry(width, height).rotateX(-Math.PI / 2);

export const disc = (radius: number): THREE.BufferGeometry => new THREE.CircleGeometry(radius, 32).rotateX(-Math.PI / 2);

function bar(ax: number, az: number, bx: number, bz: number, thick: number): THREE.BufferGeometry {
  const span = Math.hypot(bx - ax, bz - az);
  return flat(span, thick).rotateY(Math.atan2(bz - az, bx - ax)).translate((ax + bx) / 2, 0, -(az + bz) / 2);
}

export function slab(c: Capsule): THREE.BufferGeometry {
  const dx = c.bx - c.ax, dz = c.bz - c.az, span = Math.hypot(dx, dz), ux = dx / span, uz = dz / span;
  return bar(c.ax - ux * c.radius, c.az - uz * c.radius, c.bx + ux * c.radius, c.bz + uz * c.radius, 2 * c.radius);
}

const sheet = (): THREE.MeshBasicNodeMaterial => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

export function sprite(x0: number, y0: number, x1: number, y1: number, order: number, paint: Paint, world = false): THREE.Mesh {
  const material = sheet();
  const p = world ? vec2(positionWorld.x, positionWorld.z.negate()) : vec2(positionGeometry.x, positionGeometry.z.negate());
  const [color, alpha] = paint(p, footprint(p.x));
  material.colorNode = color;
  material.opacityNode = alpha;
  const mesh = new THREE.Mesh(flat(x1 - x0, y1 - y0).translate((x0 + x1) / 2, 0, -(y0 + y1) / 2), material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

const spacing = (q: N<'vec2'>, gap: number): N<'vec2'> => abs(fract(q.div(gap).add(0.5)).sub(0.5)).mul(gap);

export function wall(ink: Ink, height: number, alarm: N<'float'>): Paint {
  return (q, px) => {
    const minor = spacing(q, 0.1), major = spacing(q, 0.5), metre = spacing(q, 1);
    const grid = max(max(line(ink, min(minor.x, minor.y), 1, px).mul(0.1), line(ink, min(major.x, major.y), 1.1, px).mul(0.22)), line(ink, metre.x, 1.5, px).mul(0.36));
    const dash = fill(abs(fract(q.x.div(0.1)).sub(0.5)).sub(0.275).mul(0.1), px);
    const fall = line(ink, q.y.sub(height), 1.6, px).mul(dash);
    const tone = fractal(q.mul(1.7), 3).mul(0.035).add(1).mul(smoothstep(2.4, 0, q.y).mul(0.25).add(0.85));
    let albedo = rgb(PALETTE.wall).mul(tone);
    albedo = mix(albedo, rgb(PALETTE.grid), grid);
    albedo = mix(albedo, rgb(PALETTE.danger).mul(0.7), fall.mul(0.55));
    return [ink.lit(albedo).add(rgb(PALETTE.danger).mul(fall.mul(alarm).mul(2.4))), float(1)];
  };
}

export const RULE = { lip: 0.012, top: -0.03, bottom: -0.07, pins: -0.125 } as const;

export function ground(ink: Ink): Paint {
  return (q, px) => {
    const tenth = floor(q.x.div(0.1)), light = mod(tenth, 2);
    const band = fill(abs(q.y.sub((RULE.top + RULE.bottom) / 2)).sub((RULE.top - RULE.bottom) / 2), px);
    const tick = (gap: number, reach: number, width: number) => line(ink, spacing(vec2(q.x, 0), gap).x, width, px).mul(fill(max(float(-reach).sub(q.y), q.y.add(RULE.lip)), px));
    const ticks = max(max(tick(0.1, 0.028, 1), tick(0.5, 0.045, 1.4)), tick(1, 0.1, 1.8));
    const lip = fill(abs(q.y.add(RULE.lip / 2)).sub(RULE.lip / 2), px);
    const texture = grain(floor(q.mul(180))).x.mul(0.06).add(fractal(q.mul(vec2(0.6, 4)), 3).mul(0.08)).add(0.94);
    let albedo = rgb(PALETTE.floor).mul(texture).mul(smoothstep(-0.5, 0, q.y).mul(0.45).add(0.6));
    albedo = mix(albedo, mix(rgb(PALETTE.ruleDark), rgb(PALETTE.ruleLight), light), band);
    albedo = mix(albedo, rgb(PALETTE.ruleLight).mul(0.45), line(ink, abs(q.y.sub((RULE.top + RULE.bottom) / 2)).sub((RULE.top - RULE.bottom) / 2), 1, px));
    albedo = mix(albedo, rgb(PALETTE.grid).mul(0.75), ticks.mul(float(1).sub(band)));
    albedo = mix(albedo, rgb(PALETTE.lip), lip);
    albedo = mix(albedo, rgb(PALETTE.gold), line(ink, q.x, 2.4, px).mul(fill(q.y.negate().sub(0.11), px)));
    return [ink.lit(albedo), fill(q.y, px)];
  };
}

interface Frame { d: N<'float'>; s: N<'float'>; t: N<'float'>; span: number }

function frameOf(p: N<'vec2'>, c: Capsule): Frame {
  const dx = c.bx - c.ax, dz = c.bz - c.az, span = Math.hypot(dx, dz), ux = dx / span, uz = dz / span;
  const v = p.sub(vec2(c.ax, c.az));
  return { d: sdSegment(p, vec2(c.ax, c.az), vec2(c.bx, c.bz), c.radius), s: dot(v, vec2(ux, uz)), t: v.y.mul(ux).sub(v.x.mul(uz)), span };
}

function cylinder(tint: number, t: N<'float'>, radius: number): N<'vec3'> {
  const across = clamp(t.div(radius), -1, 1), bulge = sqrt(float(1).sub(across.mul(across))), glint = across.add(0.38).div(0.2);
  return rgb(tint).mul(bulge.mul(0.5).add(0.55)).add(rgb(0xffffff).mul(exp(glint.mul(glint).negate()).mul(0.16)));
}

function slot(f: Frame, from: number, to: number, half: number): N<'float'> {
  return sdBox(vec2(f.s.sub((from + to) / 2), f.t), vec2((to - from) / 2, half), half);
}

export function limb(ink: Ink, c: Capsule, tint: number, margin: number): Paint {
  return (p, px) => {
    const f = frameOf(p, c), groove = slot(f, margin, f.span - margin, c.radius * 0.28);
    let albedo = cylinder(tint, f.t, c.radius);
    albedo = mix(albedo, rgb(PALETTE.panel), fill(groove, px));
    albedo = mix(albedo, rgb(tint).mul(0.5), line(ink, groove, 1, px));
    albedo = mix(albedo, rgb(PALETTE.edge), line(ink, f.d, 1.3, px));
    return [ink.lit(albedo), fill(f.d, px)];
  };
}

export function torso(ink: Ink, c: Capsule, tint: number, danger: N<'float'>, sink: N<'float'>): Paint {
  return (p, px) => {
    const f = frameOf(p, c), r = c.radius, panel = slot(f, 0.075, f.span - 0.075, r * 0.36);
    const vents = line(ink, abs(fract(f.s.div(0.03)).sub(0.5)).mul(0.03), 1, px).mul(fill(max(abs(f.s.sub(f.span * 0.68)).sub(0.05), abs(f.t).sub(r * 0.22)), px));
    const lamp = length(vec2(f.s.sub(0.045), f.t)).sub(r * 0.3);
    const root = length(p).sub(0.019);
    let albedo = cylinder(tint, f.t, r);
    albedo = mix(albedo, rgb(PALETTE.panel), fill(panel, px));
    albedo = mix(albedo, rgb(tint).mul(0.5), line(ink, panel, 1, px));
    albedo = mix(albedo, rgb(tint).mul(0.35), vents);
    albedo = mix(albedo, rgb(PALETTE.edge), line(ink, f.d, 1.3, px));
    const status = mix(rgb(PALETTE.safe), rgb(PALETTE.danger), smoothstep(0.35, 0.9, danger)).mul(2.2);
    let color = mix(ink.lit(albedo), status, fill(lamp, px));
    color = mix(color, rgb(PALETTE.edge), line(ink, lamp, 1, px));
    color = mix(color, mix(rgb(PALETTE.strobe), rgb(PALETTE.danger).mul(2), sink), line(ink, root, 1.6, px));
    color = mix(color, rgb(PALETTE.strobe), fill(length(p).sub(0.005), px));
    return [color, fill(f.d, px)];
  };
}

export function foot(ink: Ink, c: Capsule, tint: number): Paint {
  return (p, px) => {
    const f = frameOf(p, c), r = c.radius, sole = f.t.add(r * 0.3);
    const tread = line(ink, abs(fract(f.s.div(0.028)).sub(0.5)).mul(0.028), 1.2, px).mul(fill(f.t.add(r * 0.62), px));
    let albedo = cylinder(tint, f.t.sub(r * 0.2), r);
    albedo = mix(albedo, rgb(PALETTE.sole).mul(clamp(f.t.div(r).add(1.4), 0.6, 1.2)), fill(sole, px));
    albedo = mix(albedo, rgb(PALETTE.edge), tread);
    albedo = mix(albedo, rgb(PALETTE.edge), line(ink, sole, 1, px).mul(fill(f.d.add(0.004), px)));
    albedo = mix(albedo, rgb(PALETTE.edge), line(ink, f.d, 1.3, px));
    return [ink.lit(albedo), fill(f.d, px)];
  };
}

export function hub(ink: Ink, radius: number, tint: number, effort: N<'float'>): Paint {
  return (p, px) => {
    const r = length(p), angle = atan(p.y, p.x), sector = Math.PI / 2, k = round(angle.div(sector));
    const bolt = length(p.sub(vec2(cos(k.mul(sector)), sin(k.mul(sector))).mul(radius * 0.55))).sub(radius * 0.1);
    const ring = abs(r.sub(radius * 0.82)).sub(radius * 0.085);
    let albedo = rgb(PALETTE.steel).mul(float(1.25).sub(r.div(radius).mul(0.45)));
    albedo = mix(albedo, rgb(PALETTE.bolt), fill(bolt, px));
    albedo = mix(albedo, rgb(tint), fill(r.sub(radius * 0.27), px));
    albedo = mix(albedo, rgb(PALETTE.edge), line(ink, r.sub(radius * 0.27), 1, px));
    let color = ink.lit(albedo);
    color = mix(color, mix(rgb(PALETTE.effort).mul(0.35), rgb(PALETTE.effort).mul(3.4), effort), fill(ring, px));
    color = mix(color, rgb(PALETTE.edge), line(ink, r.sub(radius), 1.4, px));
    return [color, fill(r.sub(radius), px)];
  };
}

export function arc(ink: Ink, radius: number, sweep: N<'float'>): Paint {
  return (p, px) => {
    const r = length(p), turn = sign(sweep), size = abs(sweep), end = turn.mul(size);
    const middle = wrap(atan(p.y, p.x).sub(end.mul(0.5)));
    const body = max(abs(r.sub(radius)).sub(0.0055), abs(middle).sub(size.mul(0.5)).mul(radius));
    const normal = vec2(cos(end), sin(end)), tangent = vec2(sin(end).negate(), cos(end)).mul(turn), at = normal.mul(radius);
    const tip = at.add(tangent.mul(0.028)), back = at.sub(tangent.mul(0.004));
    const head = min(sdSegment(p, tip, back.add(normal.mul(0.022)), 0.0045), sdSegment(p, tip, back.sub(normal.mul(0.022)), 0.0045));
    const shape = min(body, head);
    return [rgb(PALETTE.effort).mul(2.2), fill(shape, px).mul(smoothstep(0.05, 0.2, size))];
  };
}

export function gauge(ink: Ink, limit: number, pitch: N<'float'>, danger: N<'float'>): Paint {
  const { inner, outer } = GAUGE;
  return (p, px) => {
    const r = length(p), theta = atan(p.x, p.y), hot = smoothstep(0.35, 1, danger), forward = step(0, pitch);
    const ray = (side: number) => sdSegment(p, vec2(Math.sin(side * limit), Math.cos(side * limit)).mul(inner), vec2(Math.sin(side * limit), Math.cos(side * limit)).mul(outer + 0.03), 0);
    const inside = fill(max(abs(theta).sub(limit).mul(r), max(r.sub(outer), float(inner).sub(r))), px);
    const rim = line(ink, r.sub(outer), 1.2, px).mul(fill(abs(theta).sub(limit).mul(r), px));
    const notch = Math.max(1, Math.round(limit / 0.05));
    const tickAngle = round(theta.div(limit / notch)).mul(limit / notch);
    const tick = line(ink, sin(theta.sub(tickAngle)).mul(r), 1, px).mul(fill(max(float(outer).sub(r), r.sub(outer + 0.022)), px)).mul(fill(abs(tickAngle).sub(limit + 1e-3), px));
    const needleAxis = vec2(sin(pitch), cos(pitch));
    const dash = fill(abs(fract(r.div(0.035)).sub(0.5)).sub(0.24).mul(0.035), px);
    const needle = line(ink, sdSegment(p, needleAxis.mul(0.28), needleAxis.mul(outer), 0), 1.4, px).mul(dash);
    const cream = rgb(PALETTE.strobe), red = rgb(PALETTE.danger);
    const leanRight = hot.mul(forward), leanLeft = hot.mul(float(1).sub(forward));
    const wash = mix(cream.mul(0.5), red.mul(1.4), hot.mul(mix(float(1).sub(forward), forward, step(0, p.x))));
    let color = wash, alpha = inside.mul(hot.mul(0.14).add(0.05));
    color = mix(color, cream.mul(0.8), max(rim, tick));
    alpha = max(alpha, max(rim, tick).mul(0.6));
    const right = line(ink, ray(1), 1.5, px), left = line(ink, ray(-1), 1.5, px);
    color = mix(color, mix(cream.mul(0.85), red.mul(2.6), leanRight), right);
    color = mix(color, mix(cream.mul(0.85), red.mul(2.6), leanLeft), left);
    alpha = max(alpha, max(right.mul(leanRight.mul(0.4).add(0.6)), left.mul(leanLeft.mul(0.4).add(0.6))));
    color = mix(color, mix(cream, red.mul(2.2), hot), needle);
    alpha = max(alpha, needle.mul(0.85));
    return [color, alpha];
  };
}

export function shade(strength: N<'float'>): Paint {
  return (p) => {
    const q = p.div(vec2(0.3, 0.02));
    return [rgb(0x000000), float(1).sub(smoothstep(0.2, 1, length(q))).mul(strength.mul(0.6))];
  };
}

export function strokeMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aStroke', 'vec4');
  const span = state.x, radius = state.y;
  const p = uv().sub(0.5).mul(vec2(span.add(radius.add(HALO).mul(2)), radius.add(HALO).mul(2)));
  const d = sdSegment(p, vec2(span.mul(-0.5), 0), vec2(span.mul(0.5), 0), radius);
  material.colorNode = mix(rgb(PALETTE.strobe).mul(1.1), rgb(PALETTE.ghost).mul(1.8), state.w);
  material.opacityNode = fill(d, footprint(p.x)).mul(state.z);
  return material;
}

export const LAMP = { radius: 0.085, bulb: 0.032, drop: 3 } as const;

export function lamp(ink: Ink): Paint {
  const { radius, bulb, drop } = LAMP;
  return (p, px) => {
    const rod = sdSegment(p, vec2(0, radius * 0.6), vec2(0, drop), 0.007);
    const dome = max(length(p.sub(vec2(0, -0.012))).sub(radius), p.y.negate().sub(0.004));
    const collar = sdBox(p.sub(vec2(0, radius * 0.95)), vec2(0.018, 0.02), 0.006);
    const shell = min(min(dome, collar), rod);
    const face = sdBox(p.sub(vec2(0, -0.006)), vec2(radius * 0.9, 0.006), 0.004);
    const glass = length(p.sub(vec2(0, -0.004))).sub(bulb);
    let albedo = mix(rgb(PALETTE.steel).mul(smoothstep(-0.05, 0.09, p.y).mul(0.6).add(0.7)), rgb(PALETTE.bolt).mul(0.55), fill(rod, px));
    albedo = mix(albedo, rgb(PALETTE.bolt).mul(0.7), line(ink, dome.add(0.012), 1, px).mul(fill(dome, px)));
    let color = mix(ink.lit(albedo), rgb(PALETTE.edge), line(ink, shell, 1.2, px));
    color = mix(color, rgb(PALETTE.contact).mul(1.8), fill(face, px));
    color = mix(color, rgb(PALETTE.contact).mul(4), fill(glass, px));
    return [color, max(max(fill(shell, px), fill(face, px)), fill(glass, px))];
  };
}

export function pinMaterial(accent: number): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aPin', 'vec4');
  const p = uv().sub(0.5).mul(PIN), px = footprint(p.x), h = PIN * 0.32, w = PIN * 0.26, norm = Math.hypot(2 * h, w);
  const v = p.sub(vec2(0, h));
  const d = max(max(dot(v, vec2(2 * h / norm, w / norm)), dot(v, vec2(-2 * h / norm, w / norm))), p.y.negate().sub(h));
  material.colorNode = mix(rgb(PALETTE.ghost).mul(1.1), rgb(accent).mul(1.5), state.y);
  material.opacityNode = fill(d, px).mul(state.x);
  return material;
}

export function sparkMaterial(): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aSpark', 'vec4');
  const p = uv().sub(0.5).mul(SPARK), q = p.div(vec2(0.055, 0.011));
  material.colorNode = rgb(PALETTE.contact).mul(2.6);
  material.opacityNode = exp(dot(q, q).negate()).mul(state.x);
  return material;
}
