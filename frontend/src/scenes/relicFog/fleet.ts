import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, exp, float, fract, fwidth, length, max, min, mix, normalize, select, sin, smoothstep, uv, vec2 } from 'three/tsl';
import { rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import { numeral, type Glyphs } from './glyphs';
import { coverOf, type Lit, type N } from './sector';

export const SHIP = 1;
export const GLYPH = 1;
export const LABEL = 0.9;

const INK = 0x080a14;
const DEAD = 0x59607e;
const GLASS = 0xe9f4ff;
const FLAME = 0xfff1d6;
const DOOM = 0xff3b3b;
const TAU = 2 * Math.PI;

export const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const spin = (p: N<'vec2'>, angle: N<'float'>): N<'vec2'> => {
  const c = cos(angle), s = sin(angle);
  return vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)));
};

const edge = (p: N<'vec2'>, a: readonly [number, number], b: readonly [number, number]): N<'float'> => {
  const ex = b[0] - a[0], ey = b[1] - a[1], norm = Math.hypot(ex, ey);
  return dot(p.sub(vec2(a[0], a[1])), vec2(ey / norm, -ex / norm));
};

export const triangle = (p: N<'vec2'>, a: readonly [number, number], b: readonly [number, number], c: readonly [number, number]): N<'float'> => {
  const [u, v] = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0 ? [b, c] : [c, b];
  return max(max(edge(p, a, u), edge(p, u, v)), edge(p, v, a));
};

export function hull(q: N<'vec2'>): N<'float'> {
  const left = triangle(q, [0, 0.3], [-0.25, -0.22], [0, -0.09]), right = triangle(q, [0, 0.3], [0, -0.09], [0.25, -0.22]);
  return min(left, right).sub(0.012);
}

export function shipMaterial(accent: number, foe: number, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const look = attribute<'vec4'>('aShip', 'vec4'), state = attribute<'vec4'>('aState', 'vec4');
  const team = look.x, charge = look.y, heading = look.z;
  const alpha = state.x, ghost = state.y, stalled = state.z, doom = state.w;
  const p = uv().sub(0.5).mul(SHIP), cover = coverOf(fwidth(p.x));
  const q = spin(p, heading);
  const body = hull(q);
  const tone = select(team.lessThan(0.5), rgb(accent), rgb(foe));
  const wing = smoothstep(-0.2, 0.2, q.x.negate().add(q.y.mul(0.3)));
  let paint: N<'vec3'> = mix(tone.mul(0.62), tone.mul(1.18), wing);
  paint = mix(paint, rgb(DEAD).mul(wing.mul(0.4).add(0.7)), stalled.mul(0.7));
  paint = mix(paint, rgb(DOOM), doom.mul(sin(doom.mul(40)).mul(0.5).add(0.5)).mul(0.8));
  const spine = sdSegment(q, vec2(0, -0.06), vec2(0, 0.2), 0.016);
  paint = mix(paint, tone.mul(1.45).add(0.08), cover(spine).mul(stalled.oneMinus()).mul(0.7));
  const canopy = sdCircle(q.sub(vec2(0, 0.075)), 0.045);
  const engines = min(sdCircle(q.sub(vec2(-0.085, -0.15)), 0.032), sdCircle(q.sub(vec2(0.085, -0.15)), 0.032));
  const burn = charge.greaterThan(0.001).and(stalled.lessThan(0.5)), thrust = select(burn, float(1), float(0));
  const plume = min(sdSegment(q, vec2(-0.085, -0.17), vec2(-0.085, -0.23), 0.02), sdSegment(q, vec2(0.085, -0.17), vec2(0.085, -0.23), 0.02));
  let color: N<'vec3'> = mix(lit(paint), paint, 0.55);
  color = mix(color, mix(rgb(GLASS), tone, 0.25).mul(select(burn, float(1.4), float(0.6))), cover(canopy));
  color = mix(color, mix(rgb(FLAME), tone, 0.3).mul(thrust.mul(1.6).add(0.3)), cover(engines));
  const outline = cover(body.sub(0.028));
  const ring = abs(length(p).sub(0.43)).sub(0.03), heading01 = fract(float(1.25).sub(atan(p.y, p.x).div(TAU)));
  const filled = heading01.lessThan(clamp(charge, 0, 1));
  const gauge = cover(ring).mul(select(filled, float(1), float(0.2)));
  const glow = exp(max(body, 0).mul(-26)).mul(thrust.mul(0.35).add(0.15)).mul(ghost.oneMinus());
  const solid = cover(body), flame = cover(plume).mul(thrust).mul(0.8);
  const ink = max(outline, solid);
  let out: N<'vec3'> = mix(rgb(INK), color, solid);
  out = mix(out, mix(rgb(FLAME), tone, 0.4).mul(1.5), flame.mul(solid.oneMinus()));
  out = mix(out, select(filled, tone.mul(1.35).add(0.05), rgb(INK).add(0.08)), gauge.mul(ink.oneMinus()));
  const rim = cover(abs(body).sub(0.022)), veil = mix(rgb(INK).add(tone.mul(0.12)), tone.mul(0.95), rim);
  const seen = max(max(ink, flame), max(gauge, glow.mul(0.6))), hidden = max(solid.mul(0.5), rim);
  material.colorNode = mix(out.add(tone.mul(glow).mul(solid.oneMinus())), veil, ghost);
  material.opacityNode = clamp(mix(seen, hidden.mul(0.8), ghost), 0, 1).mul(alpha);
  return material;
}

export const ARROW = 1;
export const HOLD = 2;
export const AIM = 3;

export function orderMaterial(accent: number): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aOrder', 'vec4');
  const kind = state.x, dir = vec2(state.y, state.z), alpha = state.w;
  const p = uv().sub(0.5).mul(GLYPH), cover = coverOf(fwidth(p.x));
  const toward = normalize(select(length(dir).greaterThan(0.01), dir, vec2(0, 1))), across = vec2(toward.y, toward.x.negate());
  const a = vec2(dot(p, across), dot(p, toward));
  const shaft = sdSegment(a, vec2(0, 0.18), vec2(0, 0.36), 0.03);
  const head = triangle(a, [0, 0.48], [-0.1, 0.34], [0.1, 0.34]);
  const arrow = min(shaft, head.sub(0.01));
  const hold = abs(length(p).sub(0.12)).sub(0.03);
  const cross = min(min(abs(length(p).sub(0.16)).sub(0.025), sdBox(p, vec2(0.26, 0.018))), sdBox(p, vec2(0.018, 0.26)));
  const shape = select(kind.lessThan(ARROW + 0.5), arrow, select(kind.lessThan(HOLD + 0.5), hold, cross));
  const ink = cover(shape.sub(0.03)), solid = cover(shape);
  material.colorNode = mix(rgb(INK), rgb(accent).mul(1.5).add(0.1), solid);
  material.opacityNode = max(ink.mul(0.85), solid).mul(alpha);
  return material;
}

export const TOKEN = 0;
export const POINT = 1;

export function labelMaterial(set: Glyphs, accent: number, foe: number): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aLabel', 'vec4');
  const value = state.x, team = state.y, alpha = state.z, style = state.w;
  const p = uv().sub(0.5).mul(LABEL), cover = coverOf(fwidth(p.x));
  const tone = select(team.lessThan(0.5), rgb(accent), rgb(foe));
  const chip = style.lessThan(TOKEN + 0.5), chipK = select(chip, float(1), float(0));
  const disc = sdCircle(p, 0.41), ring = abs(disc.add(0.045)).sub(0.035);
  const text = select(chip, numeral(set, p, value, 0.5, 0), numeral(set, p, value, 0.42, 0, true)), plate = select(chip, disc, float(1));
  let color: N<'vec3'> = mix(rgb(INK), rgb(INK).add(tone.mul(0.16)), cover(plate));
  color = mix(color, tone.mul(1.05), cover(ring).mul(chipK));
  color = mix(color, select(chip, tone.mul(1.2).add(0.08), tone.mul(1.6).add(0.15)), cover(text));
  const outline = max(cover(plate.sub(0.03)), cover(text.sub(0.035)).mul(chipK.oneMinus()));
  material.colorNode = color;
  material.opacityNode = max(max(cover(plate).mul(0.94), cover(text)), outline.mul(0.9)).mul(alpha);
  return material;
}
