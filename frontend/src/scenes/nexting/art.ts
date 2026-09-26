import * as THREE from 'three/webgpu';
import { abs, atan, clamp, dot, exp, float, floor, fract, length, max, min, mix, mod, normalize, positionWorld, round, select, smoothstep, sqrt, texture, uv, uvec2, vec2, vec3 } from 'three/tsl';
import { coverage, fractal, grain, rgb, sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;
export type Point = THREE.UniformNode<'vec2', THREE.Vector2>;

export const TAU = 2 * Math.PI;
export const RT = 4.05;
export const TRACK = 0.34;
export const R0 = 3.48;
export const PITCH = 0.21;
export const LAPS = 10;
const CELL = 0.068;
export const PLATE = 3.62;
export const HUB = 1.24;
export const UNIT = 0.72;
export const LAMP = RT + 0.8;
export const LAMP_SIZE = 1.0;
const GLASS = 0.19;
export const PEN = 5.3;
export const WALL = 0.2;
export const ROBOT = 0.9;
export const BODY = 0.29;
export const BEAM_AT = 0.3;
export const TRAIL = 10;

const GAP = 0.013;
const PEN_R = 0.55;
const TILE = PEN / 4;

export const AMBER = 0xffb44a;
export const CORAL = 0xff6b5a;
export const SENSE = 0xdff3ff;
export const VOID = 0x05070a;
const FLOOR = 0x444d59;
const SEAM = 0x232932;
const WALL_TOP = 0x5d6773;
const WALL_FOOT = 0x2f353d;
const OUTSIDE = 0x1b2027;
const LANE = 0x232931;
const LINE = 0xb4bec9;
const PANEL = 0x141a22;
const OFF = 0x34404e;
const SHELL = 0xe3e7ec;
const RUBBER = 0x15181c;
const STEEL = 0x8a95a3;
const IRON = 0x2b3139;

export const flat = (width: number, depth: number) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);
export const disc = (radius: number) => new THREE.CircleGeometry(radius, 160).rotateX(-Math.PI / 2);

const chart = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate());
const local = (size: number): N<'vec2'> => uv().sub(0.5).mul(size);
const turn = (q: N<'vec2'>, period: number): N<'float'> => fract(atan(q.x, q.y).div(TAU)).mul(period);
const centred = (u: N<'float'>, period: number): N<'float'> => mod(u.add(period / 2), period).sub(period / 2);
const zoneGap = (u: N<'float'>, period: number, horizon: number): N<'float'> => {
  const v = centred(u, period);
  return max(v.negate().sub(horizon), v);
};

function sheet(color: N<'vec3'>, opacity: N<'float'>): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  material.colorNode = color;
  material.opacityNode = opacity;
  return material;
}

function roundedSquare(path: THREE.Path, half: number, radius: number): THREE.Path {
  path.moveTo(-half + radius, -half);
  path.lineTo(half - radius, -half);
  path.absarc(half - radius, -half + radius, radius, -Math.PI / 2, 0, false);
  path.lineTo(half, half - radius);
  path.absarc(half - radius, half - radius, radius, 0, Math.PI / 2, false);
  path.lineTo(-half + radius, half);
  path.absarc(-half + radius, half - radius, radius, Math.PI / 2, Math.PI, false);
  path.lineTo(-half, -half + radius);
  path.absarc(-half + radius, -half + radius, radius, Math.PI, 1.5 * Math.PI, false);
  return path;
}

export function penFrame(): THREE.BufferGeometry {
  const outline = roundedSquare(new THREE.Shape(), PEN + WALL, PEN_R + WALL) as THREE.Shape;
  outline.holes.push(roundedSquare(new THREE.Path(), PEN, PEN_R));
  return new THREE.ShapeGeometry(outline, 24).rotateX(-Math.PI / 2);
}

export function groundMaterial(lit: Lit, period: number, horizon: number): THREE.MeshBasicNodeMaterial {
  const q = chart(), r = length(q), u = turn(q, period), arc = r.mul(TAU / period);
  const box = sdBox(q, vec2(PEN), PEN_R);
  const tone = fractal(q.mul(0.8), 3).mul(0.07).add(fractal(q.mul(9), 2).mul(0.03)).add(1);
  const seam = float(0.5).sub(max(abs(fract(q.x.div(TILE)).sub(0.5)), abs(fract(q.y.div(TILE)).sub(0.5)))).mul(TILE);
  const patch = grain(floor(q.div(TILE))).x.mul(0.08).add(0.96);
  let color: N<'vec3'> = rgb(FLOOR).mul(tone).mul(patch);
  color = mix(color, rgb(SEAM), coverage(seam.sub(0.009)).mul(0.8));
  color = mix(color, rgb(FLOOR).mul(1.25), coverage(abs(seam.sub(0.022)).sub(0.006)).mul(0.25));
  color = color.mul(smoothstep(-1.1, 0, box).mul(-0.4).add(1));

  const across = r.sub(RT), t = abs(across);
  let lane: N<'vec3'> = rgb(LANE).mul(fractal(q.mul(7), 2).mul(0.14).add(1));
  lane = lane.mul(smoothstep(TRACK, TRACK - 0.12, t).mul(0.25).add(0.75));
  lane = mix(lane, rgb(LINE), coverage(abs(t.sub(TRACK - 0.045)).sub(0.01)).mul(0.5));
  const minor = coverage(abs(fract(u.add(0.5)).sub(0.5)).mul(arc).sub(0.006)).mul(coverage(abs(across.sub(TRACK - 0.14)).sub(0.035)));
  const major = coverage(abs(fract(u.div(10).add(0.5)).sub(0.5)).mul(10).mul(arc).sub(0.01)).mul(coverage(abs(across.sub(TRACK - 0.17)).sub(0.065)));
  lane = mix(lane, rgb(LINE), max(minor.mul(0.45), major.mul(0.75)));
  const zone = coverage(zoneGap(u, period, horizon).mul(arc)).mul(coverage(t.sub(TRACK - 0.075)));
  const hatch = coverage(abs(fract(q.x.sub(q.y).div(0.2)).sub(0.5)).mul(0.2).sub(0.05));
  lane = mix(lane, mix(rgb(AMBER).mul(0.3), rgb(AMBER).mul(0.62), hatch), zone);
  const finish = coverage(abs(centred(u, period)).mul(arc).sub(0.016)).mul(coverage(t.sub(TRACK)));
  lane = mix(lane, rgb(AMBER).mul(0.9), finish);
  color = mix(color, lane, coverage(t.sub(TRACK)));

  const lead = new THREE.QuadraticBezierCurve(new THREE.Vector2(0.26, LAMP + 0.2), new THREE.Vector2(0.9, LAMP + 0.25), new THREE.Vector2(1.25, PEN)).getPoints(10);
  let cable: N<'float'> = float(1e3);
  for (let k = 1; k < lead.length; k++) cable = min(cable, sdSegment(q, vec2(lead[k - 1]!.x, lead[k - 1]!.y), vec2(lead[k]!.x, lead[k]!.y), 0.026));
  color = mix(color, rgb(0x000000), coverage(cable.sub(0.02)).mul(0.35));
  color = mix(color, mix(rgb(0x17191d), rgb(0x3a3f47), coverage(cable.add(0.012))), coverage(cable));
  const pad = sdCircle(q.sub(vec2(0, LAMP)), 0.42);
  color = mix(color, rgb(0x20262e).mul(tone), coverage(pad).mul(0.8));
  color = mix(color, rgb(0x3a424c), coverage(abs(pad).sub(0.012)).mul(0.6));

  const lip = smoothstep(0, WALL, box);
  const across2 = float(0.5).sub(abs(fract(q.x.div(TILE)).sub(0.5))).mul(TILE), along2 = float(0.5).sub(abs(fract(q.y.div(TILE)).sub(0.5))).mul(TILE);
  const joint = coverage(select(abs(q.y).greaterThan(abs(q.x)), across2, along2).sub(0.012));
  const wall = mix(mix(rgb(WALL_TOP), rgb(WALL_FOOT), lip).mul(tone), rgb(0x1c2026), joint.mul(0.7));
  const edge = coverage(abs(box.sub(0.02)).sub(0.014));
  color = mix(color, rgb(OUTSIDE).mul(tone), coverage(box.negate()));
  color = mix(color, wall, coverage(abs(box.sub(WALL / 2)).sub(WALL / 2)));
  color = mix(color, rgb(0x8793a1), edge.mul(0.55));

  const material = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false, fog: false });
  material.colorNode = lit(color);
  return material;
}

export interface RecordState { head: Level; phase: Level; alpha: Level; history: THREE.DataTexture; ring: number }

export function recordMaterial(lit: Lit, accent: number, period: number, horizon: number, state: RecordState): THREE.MeshBasicNodeMaterial {
  const q = chart(), r = length(q), u = turn(q, period), arc = r.mul(TAU / period);
  const since = mod(state.phase.add(state.alpha).sub(u), period);
  const lap = clamp(round(float(R0).sub(r).mul(period / PITCH).sub(since).div(period)), 0, LAPS - 1);
  const age = since.add(lap.mul(period));
  const off = r.sub(float(R0).sub(age.mul(PITCH / period)));
  const back = state.alpha.sub(age), f = fract(back);
  const shape = max(max(abs(off).sub(CELL), float(GAP).sub(min(f, f.oneMinus()).mul(arc))), age.negate().mul(arc));
  const slot = mod(state.head.add(floor(back)).add(1 + state.ring), state.ring);
  const cell = texture(state.history).load(uvec2(vec2(slot, 0)));
  const known = cell.z;
  const call = cell.x, right = cell.y;
  const fade = age.div(LAPS * period).mul(-0.5).add(1);

  const inner = R0 - PITCH * LAPS - CELL - 0.03, outer = R0 + CELL + 0.03;
  const wedge = coverage(zoneGap(u, period, horizon).mul(arc)).mul(coverage(abs(r.sub((inner + outer) / 2)).sub((outer - inner) / 2)));
  const panel = rgb(PANEL).mul(fractal(q.mul(5), 2).mul(0.1).add(1));
  let color: N<'vec3'> = lit(mix(panel, rgb(AMBER).mul(0.16), wedge));
  const body = coverage(shape).mul(known);
  const dim = mix(rgb(OFF), rgb(AMBER).mul(0.3), wedge).mul(smoothstep(-CELL, 0, shape).mul(0.35).add(0.85));
  color = mix(color, lit(mix(dim, rgb(CORAL).mul(0.3), call.oneMinus().mul(right.oneMinus()))), body);
  const shine = smoothstep(-CELL, 0, shape).mul(-0.35).add(1.25);
  const glow = mix(rgb(accent).mul(1.15), rgb(CORAL).mul(1.2), right.oneMinus()).mul(fade).mul(shine);
  color = mix(color, glow, body.mul(call));
  const missed = coverage(abs(shape.add(0.016)).sub(0.008)).mul(known).mul(call.oneMinus()).mul(right.oneMinus());
  color = mix(color, rgb(CORAL).mul(1.4).mul(fade), missed);
  const rim = coverage(abs(r.sub(PLATE - 0.035)).sub(0.022));
  color = mix(color, lit(mix(rgb(0x2e353f), rgb(STEEL), smoothstep(-1, 1, q.y.sub(q.x).div(max(r, 1e-3))))), rim);
  return sheet(color, coverage(r.sub(PLATE)));
}

export interface HubState { tip: Point; pulse: Level; trail: THREE.DataTexture }

export function hubMaterial(lit: Lit, accent: number, state: HubState): THREE.MeshBasicNodeMaterial {
  const q = chart(), r = length(q);
  let color: N<'vec3'> = lit(mix(rgb(0x141c26), rgb(0x080b10), smoothstep(0, HUB, r)));
  const facing = smoothstep(-1, 1, q.y.sub(q.x).div(max(r, 1e-3)));
  color = mix(color, rgb(accent).mul(0.34), coverage(abs(r.sub(UNIT)).sub(0.006)));
  const cross = coverage(min(abs(q.x), abs(q.y)).sub(0.004)).mul(coverage(abs(r.sub(UNIT / 2 + 0.12)).sub(UNIT / 2)));
  color = mix(color, rgb(accent).mul(0.16), cross);
  let dots: N<'float'> = float(0);
  for (let k = TRAIL - 1; k >= 1; k--) {
    const mark = texture(state.trail).load(uvec2(k, 0)), life = mark.z.div(TRAIL).oneMinus();
    dots = max(dots, coverage(sdCircle(q.sub(mark.xy.mul(UNIT)), life.mul(0.022).add(0.018))).mul(mark.w).mul(clamp(life, 0, 1)));
  }
  color = mix(color, rgb(SENSE).mul(0.9), dots.mul(0.8));
  const tip = state.tip.mul(UNIT);
  color = mix(color, rgb(SENSE).mul(1.1), coverage(sdSegment(q, vec2(0, 0), tip, 0.012)));
  color = mix(color, rgb(accent).mul(1.8), coverage(sdCircle(q.sub(tip), 0.045)));
  color = mix(color, rgb(SENSE).mul(2.2), coverage(sdCircle(q.sub(tip), 0.018)));
  const cap = sdCircle(q, 0.095);
  color = mix(color, lit(mix(rgb(IRON), rgb(STEEL), facing)), coverage(cap));
  color = mix(color, rgb(AMBER).mul(state.pulse.mul(5).add(0.12)), coverage(sdCircle(q, 0.055)));
  color = color.add(rgb(AMBER).mul(exp(r.negate().div(0.14)).mul(state.pulse).mul(1.6)));
  const bezel = coverage(abs(r.sub(HUB - 0.05)).sub(0.045));
  color = mix(color, lit(mix(rgb(0x2a3038), rgb(0x9aa5b2), facing)), bezel);
  return sheet(color, coverage(r.sub(HUB)));
}

export function lampMaterial(lit: Lit, glow: Level): THREE.MeshBasicNodeMaterial {
  const p = local(LAMP_SIZE), r = length(p);
  const shadow = smoothstep(0.44, 0.24, length(p.sub(vec2(0.035, -0.045)))).mul(0.55);
  const base = sdCircle(p, 0.31);
  const facing = smoothstep(-1, 1, p.y.sub(p.x).div(max(r, 1e-3)));
  let color: N<'vec3'> = lit(mix(rgb(IRON), rgb(0x6a7480), facing.mul(smoothstep(0.2, 0.31, r))));
  let screws: N<'float'> = float(0);
  for (let k = 0; k < 4; k++) {
    const a = Math.PI / 4 + (k * Math.PI) / 2;
    screws = max(screws, coverage(sdCircle(p.sub(vec2(0.255 * Math.cos(a), 0.255 * Math.sin(a))), 0.018)));
  }
  color = mix(color, lit(rgb(0x9aa5b2)), screws);
  const bulb = clamp(float(1).sub(r.div(GLASS)), 0, 1);
  const hot = mix(rgb(0x5a3a14), rgb(AMBER), glow.mul(0.8).add(0.2)).mul(bulb.mul(1.4).add(0.6)).mul(glow.mul(5).add(0.9));
  const core = rgb(0xfff4dc).mul(exp(r.negate().div(0.05)).mul(glow.mul(9).add(0.6)));
  color = mix(color, hot.add(core), coverage(r.sub(GLASS)));
  const spoke = abs(fract(atan(p.y, p.x).div(TAU).mul(8)).sub(0.5)).mul(TAU / 8).mul(r).sub(0.011);
  const cage = min(max(spoke, max(r.sub(GLASS), float(0.06).sub(r))), abs(r.sub(0.125)).sub(0.01));
  color = mix(color, lit(rgb(0x3b424c)), coverage(cage));
  color = mix(color, lit(mix(rgb(0x3a414b), rgb(0xb9c2cc), facing)), coverage(abs(r.sub(GLASS + 0.012)).sub(0.014)));
  return sheet(mix(rgb(0x000000), color, coverage(base)), max(coverage(base), shadow));
}

export function robotMaterial(lit: Lit, accent: number, beam: Level): THREE.MeshBasicNodeMaterial {
  const p = local(ROBOT).div(BODY), r = length(p);
  const shadow = smoothstep(1.3, 0.75, length(p.sub(vec2(0.09, -0.12)))).mul(0.6);
  const wheel = sdBox(vec2(p.x, abs(p.y).sub(1.03)), vec2(0.44, 0.15), 0.1);
  const tread = coverage(abs(fract(p.x.div(0.13)).sub(0.5)).mul(0.13).sub(0.025));
  let color: N<'vec3'> = lit(mix(rgb(RUBBER), rgb(0x3a4047), tread.mul(0.8)));
  const z = sqrt(max(float(1).sub(dot(p, p)), 0));
  const light = clamp(dot(normalize(vec3(p.x, p.y, z.add(0.001))), normalize(vec3(-0.45, 0.55, 0.7))), 0, 1);
  const body = sdCircle(p, 1);
  color = mix(color, lit(rgb(SHELL).mul(light.mul(0.55).add(0.6))), coverage(body));
  const bumper = max(abs(r.sub(0.9)).sub(0.1), float(0.3).sub(p.x));
  color = mix(color, lit(rgb(0x262b32)), coverage(bumper));
  color = mix(color, lit(rgb(accent).mul(light.mul(0.4).add(0.75))), coverage(abs(r.sub(0.66)).sub(0.06)).mul(coverage(p.x.sub(0.42))));
  const lens = sdBox(p.sub(vec2(0.93, 0)), vec2(0.075, 0.4), 0.05);
  color = mix(color, mix(rgb(0x1a2530), rgb(accent).mul(beam.mul(3).add(0.8)), beam), coverage(lens));
  const pod = sdCircle(p.sub(vec2(-0.12, 0)), 0.32);
  color = mix(color, lit(mix(rgb(0x252a31), rgb(0x48505b), smoothstep(0.32, 0, length(p.sub(vec2(-0.2, 0.08)))))), coverage(pod));
  color = mix(color, rgb(accent).mul(beam.mul(2.2).add(0.45)), coverage(sdCircle(p.sub(vec2(-0.12, 0)), 0.13)));
  color = mix(color, rgb(0xffffff).mul(0.85), coverage(sdCircle(p.sub(vec2(-0.18, 0.07)), 0.04)));
  color = mix(color, lit(rgb(0x2a2f36)), coverage(sdCircle(p.sub(vec2(-0.72, 0)), 0.1)));
  const solid = min(body, wheel);
  return sheet(mix(rgb(0x000000), color, coverage(solid)), max(coverage(solid), shadow));
}
