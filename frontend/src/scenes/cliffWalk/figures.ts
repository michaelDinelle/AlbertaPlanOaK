import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, exp, float, floor, fract, length, max, min, mix, mod, pow, sin, smoothstep, sqrt, uniform, uv, vec2, vec3 } from 'three/tsl';
import { coverage, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';
import type { Look } from './coast';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;

export const FIGURE = 0.92;
export const PRINT = 0.24;
export const SPLASH = 0.62;
export const LANTERN = new THREE.Vector2(0.205, 0.2);
export const BEAM = 18;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

const sheet = () => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

const ellipse = (p: N<'vec2'>, centre: readonly [number, number], radii: readonly [number, number]) =>
  length(p.sub(vec2(...centre)).div(vec2(...radii))).sub(1).mul(Math.min(...radii));

export function figureMaterial(accent: number, look: Look, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aFigure', 'vec4');
  const p = uv().sub(0.5).mul(FIGURE);
  const shadow = smoothstep(-0.05, 0.05, ellipse(p, [0.035, -0.045], [0.24, 0.21])).oneMinus().mul(0.34);
  const pack = sdBox(p.sub(vec2(0, -0.125)), vec2(0.118, 0.078), 0.05);
  const roll = sdBox(p.sub(vec2(0, -0.205)), vec2(0.128, 0.03), 0.03);
  const body = ellipse(p, [0, -0.005], [0.212, 0.13]);
  const right = sdSegment(p, vec2(0.15, 0.0), vec2(LANTERN.x - 0.012, LANTERN.y - 0.05), 0.034);
  const left = sdSegment(p, vec2(-0.15, 0.0), vec2(-0.18, 0.1), 0.032);
  const brim = sdCircle(p.sub(vec2(0, 0.025)), 0.104), crown = sdCircle(p.sub(vec2(0, 0.03)), 0.064);
  const bail = abs(sdCircle(p.sub(vec2(LANTERN.x, LANTERN.y + 0.01)), 0.05)).sub(0.008);
  const cage = sdCircle(p.sub(vec2(LANTERN.x, LANTERN.y)), 0.043), glass = sdCircle(p.sub(vec2(LANTERN.x, LANTERN.y)), 0.029);
  const silhouette = min(min(min(pack, roll), min(body, right)), min(min(left, brim), cage));

  const jacket = rgb(accent), sleeve = rgb(accent).mul(0.78);
  let color: N<'vec3'> = rgb(0x120f0d);
  color = mix(color, rgb(0x4f5a3a).mul(p.y.mul(0.9).add(1)), coverage(pack));
  color = mix(color, mix(rgb(0x8a6a45), rgb(0x6f5236), smoothstep(0.4, 0.6, mod(p.x.mul(22), 2).mul(0.5))), coverage(roll));
  color = mix(color, jacket.mul(p.x.mul(0.7).add(p.y.mul(0.9)).add(1)), coverage(body));
  color = mix(color, sleeve, coverage(min(right, left)));
  color = mix(color, rgb(0x3e3128), coverage(brim));
  color = mix(color, rgb(0x5d4838).mul(p.y.sub(0.03).mul(2.2).add(1)), coverage(crown));
  color = mix(color, rgb(0x2a211b), coverage(abs(sdCircle(p.sub(vec2(0, 0.03)), 0.064)).sub(0.008)).mul(coverage(crown.sub(0.004))));
  color = mix(color, rgb(0x241f1a), coverage(min(cage, bail)));
  const sink = state.y;
  const shaded = mix(lit(color), rgb(look.water[0]).mul(0.6), sink.mul(0.85));
  const flame = rgb(0xffd79a).mul(state.z.mul(2.6).add(0.35)).mul(sink.oneMinus());
  const ink = coverage(silhouette.sub(0.016));
  const solid = coverage(silhouette);
  const lamp = coverage(glass);
  const figure = mix(mix(mix(rgb(0x120f0d), shaded, solid), flame, lamp), rgb(0xffe6b8).mul(2.2), state.w);
  const alpha = max(ink, shadow.mul(sink.oneMinus()));
  material.colorNode = figure.mul(ink).div(max(alpha, 1e-4));
  material.opacityNode = alpha.mul(state.x);
  return material;
}

export function printMaterial(look: Look, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aPrint', 'vec4');
  const p = uv().sub(0.5).mul(PRINT).sub(vec2(state.y.mul(0.052), 0));
  const sole = ellipse(p, [0, 0.03], [0.036, 0.058]), heel = ellipse(p, [0, -0.062], [0.03, 0.028]);
  material.colorNode = lit(rgb(look.print));
  material.opacityNode = coverage(min(sole, heel)).mul(state.x);
  return material;
}

export function skidMaterial(lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aSkid', 'vec4');
  const span = state.y, p = uv().sub(0.5).mul(vec2(span, 0.2));
  const a = span.mul(-0.5).add(0.05), b = span.mul(0.5).sub(0.05);
  const streak = sdSegment(vec2(p.x, abs(p.y)), vec2(a, 0.042), vec2(b, 0.042), 0.011);
  const along = clamp(p.x.sub(a).div(max(b.sub(a), 1e-3)), 0, 1);
  material.colorNode = lit(rgb(0xe6f5ff)).mul(1.15);
  material.opacityNode = coverage(streak).mul(along.mul(0.7).add(0.3)).mul(state.x);
  return material;
}

export function splashMaterial(look: Look): THREE.MeshBasicNodeMaterial {
  const material = sheet();
  const state = attribute<'vec4'>('aSplash', 'vec4');
  const age = clamp(state.x, 0, 1), p = uv().sub(0.5).mul(SPLASH), r = length(p);
  const grow = pow(age, 0.55);
  const ring = abs(r.sub(grow.mul(0.19).add(0.03))).sub(age.oneMinus().mul(0.02).add(0.004));
  const angle = atan(p.y, p.x), sector = Math.PI / 5, k = floor(angle.div(sector)).add(0.5);
  const drop = length(p.sub(vec2(cos(k.mul(sector)), sin(k.mul(sector))).mul(grow.mul(0.24).add(0.04)))).sub(age.oneMinus().mul(0.016).add(0.004));
  const foam = max(coverage(ring), coverage(drop).mul(smoothstep(0.1, 0.35, age)));
  material.colorNode = rgb(look.foam).mul(1.1);
  material.opacityNode = foam.mul(pow(age.oneMinus(), 1.4)).mul(state.y);
  return material;
}

export interface Lighthouse { mesh: THREE.Mesh; flare: Level; heading: Level }

export const LAMP = new THREE.Vector2(0, 0.148);

export function lighthouse(accent: number, look: Look, lit: Lit, shade: THREE.Color): Lighthouse {
  const material = sheet(), flare = uniform(0), heading = uniform(0);
  const size = 1.02, q = uv().sub(0.5).mul(size), p = q.div(1.14);
  const plinth = sdBox(q, vec2(0.44), 0.1);
  const joints = min(abs(q.x), abs(q.y));
  let ground: N<'vec3'> = rgb(look.flag[1]).mul(smoothstep(0.33, 0.44, max(abs(q.x), abs(q.y))).mul(-0.2).add(1));
  ground = mix(ground, rgb(look.flag[2]), coverage(joints.sub(0.006)).mul(0.7));
  ground = ground.mul(smoothstep(-0.04, 0.12, ellipse(p, [0.1, -0.27], [0.3, 0.12])).mul(0.5).add(0.5));
  const base = -0.25, top = 0.06, lift = p.y.sub(base).div(top - base);
  const half = mix(float(0.23), float(0.165), clamp(lift, 0, 1));
  const across = clamp(p.x.div(half), -1, 1);
  const bow = sqrt(max(across.mul(across).oneMinus(), 0));
  const body = min(max(abs(p.x).sub(half), max(p.y.sub(top), float(base).sub(p.y))), ellipse(p, [0, base], [0.23, 0.07]));
  const band = coverage(abs(fract(p.y.sub(bow.mul(0.05)).sub(base).div(0.156)).sub(0.5)).sub(0.25).mul(0.156));
  const round = smoothstep(-0.9, 1, across).mul(-0.62).add(1.18);
  let tower: N<'vec3'> = mix(rgb(0xf0ebe0), rgb(accent), band).mul(round);
  const door = sdBox(p.sub(vec2(-0.02, base + 0.03)), vec2(0.034, 0.05), 0.03);
  tower = mix(tower, rgb(0x2a211b), coverage(door));
  const gallery = ellipse(p, [0, top], [0.215, 0.06]);
  const rail = coverage(abs(gallery).sub(0.008)).max(coverage(abs(fract(p.x.div(0.045)).sub(0.5)).mul(0.045).sub(0.005)).mul(coverage(abs(p.y.sub(top).add(0.03)).sub(0.03))).mul(coverage(gallery)));
  const deck = mix(rgb(0x3a3431), rgb(0x151210), rail);
  const room = sdBox(p.sub(vec2(0, top + 0.07)), vec2(0.11, 0.07), 0.012);
  const mullion = coverage(abs(fract(p.x.div(0.073).add(0.5)).sub(0.5)).mul(0.073).sub(0.005));
  const angle = atan(q.y.sub(LAMP.y), q.x.sub(LAMP.x));
  const lens = cos(angle.sub(heading)).mul(0.5).add(0.5).pow(4).add(cos(angle.sub(heading).add(Math.PI)).mul(0.5).add(0.5).pow(4));
  const glass = mix(rgb(0xffe2a8).mul(flare.mul(2).add(1.05).add(lens.mul(0.9))), rgb(0x241e1a), mullion.mul(0.8));
  const dome = ellipse(p, [0, top + 0.145], [0.125, 0.06]);
  const cap = rgb(accent).mul(0.5).mul(smoothstep(-0.12, 0.12, p.x).mul(-0.5).add(1.25));
  const vent = sdCircle(p.sub(vec2(0, top + 0.205)), 0.024);
  const night = vec3(shade.r, shade.g, shade.b);
  let out: N<'vec3'> = lit(ground);
  out = mix(out, tower.mul(night), coverage(body));
  out = mix(out, deck.mul(night), coverage(gallery));
  out = mix(out, glass, coverage(room));
  out = mix(out, cap.mul(night), coverage(dome));
  out = mix(out, rgb(0x1b1714).mul(night), coverage(vent));
  material.colorNode = out;
  material.opacityNode = coverage(plinth);
  const mesh = new THREE.Mesh(flat(size, size), material);
  return { mesh, flare, heading };
}

export function beamMaterial(): { material: THREE.MeshBasicNodeMaterial; strength: Level } {
  const material = sheet(), strength = uniform(0);
  material.blending = THREE.AdditiveBlending;
  const p = uv().sub(0.5).mul(vec2(BEAM, 2.4)), along = abs(p.x);
  const half = along.mul(0.075).add(0.05);
  const cone = exp(p.y.div(half).pow(2).mul(-2.2));
  const fade = exp(along.mul(-0.2)).mul(smoothstep(0.3, 0.9, along)).mul(smoothstep(BEAM * 0.3, BEAM * 0.5, along).oneMinus());
  material.colorNode = vec3(1.0, 0.86, 0.62);
  material.opacityNode = cone.mul(fade).mul(strength);
  return { material, strength };
}

export const POST = 0.64;
export const POST_LAMP = new THREE.Vector2(-0.07, 0.17);

export function postMaterial(lit: Lit): { material: THREE.MeshBasicNodeMaterial; glow: Level } {
  const material = sheet(), glow = uniform(1);
  const p = uv().sub(0.5).mul(POST);
  const shadow = smoothstep(-0.03, 0.06, sdBox(p.sub(vec2(0.025, -0.03)), vec2(0.05), 0.016)).oneMinus().mul(0.42);
  const post = sdBox(p, vec2(0.046), 0.012);
  const east = sdBox(p.sub(vec2(0.13, 0.025)), vec2(0.11, 0.03), 0.01);
  const tip = sdSegment(p, vec2(0.235, 0.025), vec2(0.21, 0.058), 0.006);
  const arm = sdBox(p.sub(vec2(-0.035, 0.1)), vec2(0.056, 0.014), 0.006);
  const lampAt = vec2(POST_LAMP.x, POST_LAMP.y);
  const cage = sdCircle(p.sub(lampAt), 0.042), flame = sdCircle(p.sub(lampAt), 0.026);
  const wood = min(min(post, east), arm);
  let color: N<'vec3'> = lit(rgb(0x6b4e34).mul(p.y.mul(1.2).add(1)));
  color = mix(color, lit(rgb(0x3a2a1c)), coverage(abs(east).sub(0.007)).mul(coverage(east)));
  color = mix(color, lit(rgb(0xe8dcc0)), coverage(sdBox(p.sub(vec2(0.12, 0.025)), vec2(0.07, 0.008), 0.004)).mul(0.55));
  color = mix(color, lit(rgb(0x221c17)), coverage(min(cage, tip)));
  color = mix(color, rgb(0xffd08a).mul(glow.mul(2.2).add(0.6)), coverage(flame));
  const solid = coverage(min(min(wood, cage), tip)), alpha = max(solid, shadow);
  material.colorNode = mix(rgb(0x0e0b09), color, solid).mul(solid).div(max(alpha, 1e-4));
  material.opacityNode = alpha;
  return { material, glow };
}

