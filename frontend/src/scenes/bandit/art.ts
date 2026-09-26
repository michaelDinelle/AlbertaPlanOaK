import * as THREE from 'three/webgpu';
import { abs, dot, exp, float, floor, fract, length, max, min, mix, mod, normalize, positionWorld, pow, smoothstep, sqrt, step, uv, vec2, vec3 } from 'three/tsl';
import type { Light2D } from '@kit/light2d';
import { coverage, noise, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
export type Tint = THREE.UniformNode<'vec3', THREE.Vector3>;
export type Scalar = THREE.UniformNode<'float', number>;

export const PITCH = 2.7;
export const TUBE = 0.8;
export const REACH = 3.2;
export const CAP = 0.22;
export const CABINET = 0.95;
export const CABINET_TALL = 1.93;
export const TOP = -(REACH + CAP);
export const FLOOR = TOP - CABINET_TALL;
export const PIVOT = FLOOR + 0.95;
export const LEVER = 0.95;
export const LEVER_X = CABINET + 0.13;
export const KNOB = 0.36;
export const CROWN = REACH + CAP + 0.5;
export const PANEL_Y = TOP - 0.42;

const MOUNT = LEVER_X + 0.22;
const ROD = 0.075;
export const LED = 0xffa347;
const BRASS = 0xc39a52;
const CHROME = 0x9aa3b3;
const SEG_W = 0.1;
const SEG_H = 0.19;
const SEG_GAP = 0.026;
const SEG_THICK = 0.026;
const SEGMENTS: readonly (readonly [number, number, number, number])[] = [
  [-SEG_W + SEG_GAP, SEG_H, SEG_W - SEG_GAP, SEG_H],
  [SEG_W, SEG_H - SEG_GAP, SEG_W, SEG_GAP],
  [SEG_W, -SEG_GAP, SEG_W, -SEG_H + SEG_GAP],
  [-SEG_W + SEG_GAP, -SEG_H, SEG_W - SEG_GAP, -SEG_H],
  [-SEG_W, -SEG_GAP, -SEG_W, -SEG_H + SEG_GAP],
  [-SEG_W, SEG_H - SEG_GAP, -SEG_W, SEG_GAP],
  [-SEG_W + SEG_GAP, 0, SEG_W - SEG_GAP, 0],
];
export const DIGITS = [0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111, 0b1111111, 0b1101111] as const;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
const local = (width: number, depth: number): N<'vec2'> => uv().sub(0.5).mul(vec2(width, depth));
const rhombus = (p: N<'vec2'>, a: number, b: number): N<'float'> => abs(p.x).mul(b).add(abs(p.y).mul(a)).sub(a * b).div(Math.hypot(a, b));

export function sprite(color: N<'vec3'>, opacity: N<'float'>, additive = false): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  if (additive) material.blending = THREE.AdditiveBlending;
  return material;
}

const brass = (shine: N<'float'>): N<'vec3'> => mix(rgb(0x6e5325), rgb(BRASS), shine).add(rgb(0xfff0c8).mul(pow(shine, float(6)).mul(0.35)));

export function hallMaterial(light: Light2D, rowHalf: number): THREE.MeshBasicNodeMaterial {
  const q = vec2(positionWorld.x, positionWorld.z.negate());
  const flute = abs(fract(q.x.div(0.45)).sub(0.5)).mul(2);
  const wood = noise(vec2(q.x.mul(0.8), q.y.mul(0.08))).mul(0.06).add(1);
  const wall = mix(rgb(0x5e3e6a), rgb(0x382244), smoothstep(0.55, 1, flute)).mul(wood).mul(smoothstep(REACH + 2.5, TOP, q.y).mul(0.3).add(0.7));
  const rail = coverage(abs(q.y).sub(0.028)).mul(step(abs(q.x), rowHalf));
  const tile = vec2(q.x.div(0.7).add(floor(q.y.div(0.7)).mul(0.5)), q.y.div(0.7));
  const motif = coverage(rhombus(fract(tile).sub(0.5).mul(0.7), 0.12, 0.12)).mul(0.5).add(coverage(sdCircle(fract(tile).sub(0.5).mul(0.7), 0.035)));
  const carpet = mix(rgb(0x1d3a44), rgb(0x8a7444), motif.mul(0.6));
  const nearest = fract(q.x.div(PITCH).add(rowHalf / PITCH)).sub(0.5).mul(PITCH);
  const beneath = smoothstep(0.4, 0, float(FLOOR).sub(q.y)).mul(smoothstep(0.3, -0.1, abs(nearest).sub(CABINET))).mul(step(abs(q.x), rowHalf));
  const skirting = coverage(abs(q.y.sub(FLOOR + 0.07)).sub(0.07));
  const lip = coverage(abs(q.y.sub(FLOOR + 0.15)).sub(0.014));
  const floorBand = step(q.y, FLOOR);
  let color: N<'vec3'> = mix(wall, carpet.mul(float(1).sub(beneath.mul(0.75))), floorBand);
  color = mix(color, rgb(0x0d0710), skirting);
  color = mix(color, brass(float(0.8)), max(lip, rail.mul(0.9)));
  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = light.lit(color);
  return material;
}

export function tubeMaterial(light: Light2D, scale: Scalar): { back: THREE.MeshBasicNodeMaterial; front: THREE.MeshBasicNodeMaterial; width: number; height: number } {
  const width = 2 * (TUBE + 0.12), height = 2 * (REACH + CAP) + 0.1, p = local(width, height);
  const glassShape = sdBox(p, vec2(TUBE, REACH + 0.04), 0.08);
  const caps = min(sdBox(p.sub(vec2(0, REACH + CAP / 2)), vec2(TUBE + 0.09, CAP / 2), 0.07), sdBox(p.sub(vec2(0, -REACH - CAP / 2)), vec2(TUBE + 0.09, CAP / 2), 0.07));
  const across = p.x.div(TUBE);
  const glass = mix(rgb(0x2c3d5a), rgb(0x0e1422), across.mul(across));
  const level = p.y.div(scale);
  const toWhole = abs(fract(level.add(0.5)).sub(0.5)).mul(scale);
  const toHalf = abs(fract(level).sub(0.5)).mul(scale);
  const inward = float(TUBE).sub(abs(p.x));
  const major = coverage(toWhole.sub(0.012)).mul(step(inward, 0.22));
  const minor = coverage(toHalf.sub(0.009)).mul(step(inward, 0.11)).mul(0.7);
  const zero = coverage(abs(p.y).sub(0.016));
  const etched = mix(glass, rgb(0xd8c08a), max(max(major, minor).mul(0.55), zero.mul(0.8)));
  const capY = p.y.sub(p.y.sign().mul(REACH + CAP / 2)).div(CAP / 2);
  const capColor = brass(smoothstep(-1, 0.5, capY).mul(0.55).add(0.3).sub(smoothstep(0.5, 1, capY).mul(0.25)));
  const back = sprite(light.lit(mix(etched, capColor, coverage(caps))), max(coverage(glassShape), coverage(caps)));
  const streak = smoothstep(0.16, 0, abs(across.add(0.55))).mul(0.07).add(smoothstep(0.1, 0, abs(across.sub(0.78))).mul(0.05));
  const rim = coverage(abs(glassShape.add(0.02)).sub(0.014)).mul(0.35);
  const front = sprite(vec3(streak.add(rim)).mul(rgb(0xcfe4ff)), coverage(glassShape.sub(0.02)), true);
  return { back, front, width, height };
}

export function cabinetMaterial(light: Light2D, glow: Scalar, mask: Scalar): { material: THREE.MeshBasicNodeMaterial; width: number } {
  const width = 2 * MOUNT, half = CABINET_TALL / 2, p = local(width, CABINET_TALL);
  const body = sdBox(p, vec2(CABINET, half), 0.14);
  const lacquer = mix(rgb(0x5c1020), rgb(0xc02a42), smoothstep(-half, half, p.y)).mul(smoothstep(0.35, 0, abs(p.x.add(0.42))).mul(0.25).add(0.9));
  const inlay = coverage(abs(body.add(0.08)).sub(0.018));
  const panelAt = p.sub(vec2(0, PANEL_Y - (TOP - half)));
  const panel = sdBox(panelAt, vec2(CABINET - 0.24, 0.26), 0.07);
  const bezel = coverage(abs(panel).sub(0.028));
  const band = coverage(abs(p.y.add(0.12)).sub(0.03)).mul(step(abs(p.x), CABINET - 0.08));
  const tray = sdBox(p.sub(vec2(0, 0.3 - half)), vec2(CABINET - 0.4, 0.1), 0.1);
  const lip = coverage(abs(tray).sub(0.022));
  const mount = min(sdBox(p.sub(vec2((CABINET + LEVER_X) / 2, PIVOT - (TOP - half))), vec2((LEVER_X - CABINET) / 2 + 0.06, 0.1), 0.05), sdCircle(p.sub(vec2(LEVER_X, PIVOT - (TOP - half))), 0.11));
  const hub = coverage(sdCircle(p.sub(vec2(LEVER_X, PIVOT - (TOP - half))), 0.045));
  const digit = vec2(panelAt.x.sub(panelAt.y.mul(0.14)), panelAt.y);
  let ghost: N<'float'> = float(1e3), lit: N<'float'> = float(0);
  SEGMENTS.forEach(([ax, ay, bx, by], k) => {
    const d = sdSegment(digit, vec2(ax, ay), vec2(bx, by), SEG_THICK);
    ghost = min(ghost, d);
    lit = max(lit, coverage(d).mul(mod(floor(mask.div(2 ** k)), 2)));
  });
  let albedo: N<'vec3'> = mix(lacquer, brass(float(0.75)), max(inlay, band.mul(0.85)));
  albedo = mix(albedo, rgb(0x060508), coverage(tray));
  albedo = mix(albedo, brass(float(0.9)), lip);
  albedo = mix(albedo, rgb(0x05060a), coverage(panel));
  albedo = mix(albedo, rgb(CHROME), bezel);
  albedo = mix(albedo, rgb(CHROME).mul(0.8), coverage(mount));
  albedo = mix(albedo, rgb(0x15161b), hub);
  const backlight = rgb(LED).mul(glow.mul(0.05)).mul(coverage(panel.add(0.03)));
  const led = rgb(LED).mul(max(coverage(ghost).mul(0.07), lit.mul(glow)));
  const material = sprite(light.lit(albedo).add(backlight).add(led), max(coverage(body), coverage(mount)));
  return { material, width };
}

export function rodMaterial(light: Light2D): { material: THREE.MeshBasicNodeMaterial; width: number } {
  const across = uv().x.sub(0.5).mul(2);
  const shine = float(1).sub(across.mul(across));
  const color = mix(rgb(0x3c414d), rgb(0xe3e8f0), shine.mul(shine));
  return { material: sprite(light.lit(color), coverage(abs(across).sub(0.8))), width: ROD };
}

export function knobMaterial(light: Light2D): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(2), r2 = dot(p, p);
  const n = vec3(p.x, p.y, sqrt(max(float(1).sub(r2), 0)));
  const key = normalize(vec3(-0.45, 0.6, 0.66));
  const diffuse = max(dot(n, key), 0), spec = pow(max(dot(n, normalize(key.add(vec3(0, 0, 1)))), 0), float(28));
  const color = light.lit(rgb(0xeee1c4).mul(diffuse.mul(0.7).add(0.3))).add(vec3(spec.mul(0.8)));
  return sprite(color, coverage(sdCircle(p, 0.94)));
}

export function orbMaterial(accent: N<'vec3'>, strength: Scalar): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(2), r = length(p);
  const ring = coverage(abs(r.sub(0.5)).sub(0.045));
  const halo = exp(r.mul(-4.5)).mul(0.55).mul(smoothstep(1, 0.7, r));
  return sprite(accent.mul(strength), max(ring, halo), true);
}

export function coinShade(): N<'vec3'> {
  const p = uv().sub(0.5).mul(2), r = length(p);
  const rim = smoothstep(0.62, 0.95, r);
  const relief = coverage(abs(r.sub(0.6)).sub(0.05)).mul(0.18);
  const glint = smoothstep(0.9, 0, length(p.sub(vec2(-0.35, 0.35)))).mul(0.4);
  return vec3(float(0.92).sub(rim.mul(0.38)).sub(relief).add(glint));
}

export const coinCoverage = (): N<'float'> => coverage(sdCircle(uv().sub(0.5).mul(2), 0.95));

export function needleMaterial(tint: Tint): { material: THREE.MeshBasicNodeMaterial; width: number; height: number } {
  const width = 2 * (TUBE + 0.28), height = 0.3, p = local(width, height);
  const bar = sdBox(p, vec2(TUBE - 0.03, 0.016), 0.016);
  const tips = min(rhombus(p.sub(vec2(TUBE + 0.14, 0)), 0.11, 0.07), rhombus(p.sub(vec2(-TUBE - 0.14, 0)), 0.11, 0.07));
  const shape = min(bar, tips);
  return { material: sprite(mix(rgb(0x03050a), tint, coverage(shape)), coverage(shape.sub(0.028))), width, height };
}

export function sparkleMaterial(tint: N<'vec3'>): THREE.MeshBasicNodeMaterial {
  const p = uv().sub(0.5).mul(0.9);
  const star = min(min(rhombus(p, 0.075, 0.4), rhombus(p, 0.4, 0.075)), sdCircle(p, 0.07));
  const halo = exp(length(p).mul(-9)).mul(0.35);
  return sprite(tint, max(coverage(star), halo));
}

export function trailMaterial(tint: Tint): THREE.MeshBasicNodeMaterial {
  const along = uv().x, across = abs(uv().y.sub(0.5)).mul(2);
  return sprite(tint, along.mul(along).mul(float(1).sub(across.mul(across))), true);
}

export function ringMaterial(tint: Tint, fade: Scalar): THREE.MeshBasicNodeMaterial {
  const r = length(uv().sub(0.5).mul(2));
  return sprite(tint, coverage(abs(r.sub(0.8)).sub(0.07)).mul(fade), true);
}
