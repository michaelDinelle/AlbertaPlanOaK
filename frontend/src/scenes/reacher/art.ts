import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dFdx, dFdy, dot, exp, float, fract, fwidth, length, max, min, mix, positionWorld, pow, select, sin, smoothstep, sqrt, vec2, vec3 } from 'three/tsl';
import { fractal, rgb, sdBox } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
export type Level = THREE.UniformNode<'float', number>;
export type Point = THREE.UniformNode<'vec2', THREE.Vector2>;

export const SCALE = 10;
export const SWEEP = 1.5 * Math.PI;
const TAU = 2 * Math.PI;
const FEATHER = 1.4;

export const PALETTE = {
  desk: 0x14171b,
  plate: 0xd2d5cd,
  minor: 0xbfc4bd,
  major: 0xa3aaa7,
  workspace: 0x6f7c83,
  rail: 0xb04a72,
  railLight: 0xf0a0bf,
  railDark: 0x5e1d39,
  base: 0xc0517c,
  baseDark: 0x6a2442,
  steel: 0xd9e0e4,
  bolt: 0x7f8990,
  link: 0x2b80b4,
  linkDark: 0x0d344f,
  hub: 0x1b577f,
  hubDark: 0x0a2335,
  tipDark: 0x0b4640,
  target: 0xe5473e,
  targetDark: 0x5e130f,
  torque: 0xef7a19,
  stop: 0xffb23e,
  ink: 0x0f8577,
  gold: 0xffd166,
  amber: 0xeb9c12,
} as const;

export interface Dims {
  link: number;
  forearm: number;
  reach: number;
  range: number;
  rod: number;
  base: number;
  shoulder: number;
  hub: number;
  tip: number;
  target: number;
  ring: number;
  touch: number;
  plate: number;
  corner: number;
  rail: number;
  shoulderArc: number;
  elbowArc: number;
  shadow: THREE.Vector2;
  blur: number;
}

export function dims(link: number, forearm: number, reach: number, range: number): Dims {
  return {
    link, forearm, reach, range,
    rod: 0.1 * link,
    base: 0.3 * link,
    shoulder: 0.17 * link,
    hub: 0.15 * link,
    tip: 0.1 * link,
    target: 0.09 * link,
    ring: 0.22 * link,
    touch: 0.19 * link,
    plate: link + reach + 0.3 * link,
    corner: 0.3 * link,
    rail: 0.14 * link,
    shoulderArc: 0.46 * link,
    elbowArc: 0.31 * link,
    shadow: new THREE.Vector2(0.04 * link, -0.05 * link),
    blur: 0.035 * link,
  };
}

interface Layer { color: N<'vec3'>; alpha: N<'float'> }

const blank = (): Layer => ({ color: vec3(0), alpha: float(0) });
const over = (base: Layer, color: N<'vec3'>, alpha: N<'float'>): Layer => ({
  color: color.mul(alpha).add(base.color.mul(alpha.oneMinus())),
  alpha: alpha.add(base.alpha.mul(alpha.oneMinus())),
});

const meters = (): N<'vec2'> => vec2(positionWorld.x, positionWorld.z.negate()).div(SCALE);
const footprint = (v: N<'float'>): N<'float'> => max(length(vec2(dFdx(v), dFdy(v))), 1e-9);
const fill = (d: N<'float'>, px: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(px)), 0, 1);
const stroke = (d: N<'float'>, width: N<'float'> | number, px: N<'float'>, dpr: Level): N<'float'> => clamp(dpr.mul(width).mul(0.5).add(0.5).sub(abs(d).div(px)), 0, 1);
const soft = (d: N<'float'>, blur: number): N<'float'> => smoothstep(-blur, blur, d).oneMinus();

const sheet = (): THREE.MeshBasicNodeMaterial => new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });

function finish(material: THREE.MeshBasicNodeMaterial, layer: Layer): THREE.MeshBasicNodeMaterial {
  material.colorNode = layer.color.div(max(layer.alpha, 1e-4));
  material.opacityNode = layer.alpha;
  return material;
}

const LIGHT = new THREE.Vector3(-0.42, 0.5, 0.76).normalize();
const HALF = LIGHT.clone().add(new THREE.Vector3(0, 0, 1)).normalize();

function shade(n: N<'vec2'>, base: N<'vec3'>, gloss: number): N<'vec3'> {
  const m = n.div(max(length(n), 1)), normal = vec3(m.x, m.y, sqrt(max(float(1).sub(dot(m, m)), 0)));
  const diffuse = max(dot(normal, vec3(LIGHT.x, LIGHT.y, LIGHT.z)), 0);
  const glint = pow(max(dot(normal, vec3(HALF.x, HALF.y, HALF.z)), 0), 48).mul(gloss);
  return base.mul(diffuse.mul(0.7).add(0.36)).add(vec3(glint));
}

interface Round { d: N<'float'>; n: N<'vec2'> }

function capsule(p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>, radius: number): Round {
  const pa = p.sub(a), ba = b.sub(a), h = clamp(dot(pa, ba).div(max(dot(ba, ba), 1e-12)), 0, 1), v = pa.sub(ba.mul(h));
  return { d: length(v).sub(radius), n: v.div(radius) };
}

function ball(p: N<'vec2'>, centre: N<'vec2'>, radius: N<'float'> | number): Round {
  const v = p.sub(centre);
  return { d: length(v).sub(radius), n: v.div(radius) };
}

function arc(p: N<'vec2'>, radius: number, start: N<'float'>, sweep: N<'float'>, width: number, head: number, flare: number): N<'float'> {
  const flip = select(sweep.lessThan(0), float(-1), float(1)), c = cos(start), s = sin(start);
  const local = vec2(p.x.mul(c).add(p.y.mul(s)), p.y.mul(c).sub(p.x.mul(s)).mul(flip));
  const span = abs(sweep), angle = atan(local.y, local.x.add(1e-9)), phi = select(angle.lessThan(0), angle.add(TAU), angle);
  const toward = vec2(cos(span), sin(span)), end = toward.mul(radius);
  const band = select(phi.lessThanEqual(span), abs(length(local).sub(radius)), min(length(local.sub(vec2(radius, 0))), length(local.sub(end)))).sub(width);
  const e = local.sub(end), u = dot(e, vec2(toward.y.negate(), toward.x)), v = abs(dot(e, toward));
  const arrow = max(u.negate(), v.mul(head).add(u.mul(flare)).sub(flare * head).div(Math.hypot(head, flare)));
  return min(band, arrow);
}

export function tableMaterial(d: Dims, dpr: Level, lit: Lit): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({ fog: false });
  const q = meters(), px = footprint(q.x);
  const plate = sdBox(q, vec2(d.plate), d.corner), rim = sdBox(q, vec2(d.plate + d.rail), d.corner + d.rail);
  const cast = sdBox(q.sub(vec2(d.shadow.x * 2, d.shadow.y * 2)), vec2(d.plate + d.rail), d.corner + d.rail);
  const desk = rgb(PALETTE.desk).mul(fractal(q.mul(14), 3).mul(0.1).add(1)).mul(smoothstep(-0.01, 0.06, cast).mul(0.6).add(0.4));
  const across = clamp(plate.div(d.rail), 0, 1), bump = sin(across.mul(Math.PI));
  const lip = smoothstep(0.1, 0.45, across).mul(smoothstep(0.4, 0.62, across).oneMinus());
  const rail = mix(rgb(PALETTE.railDark), rgb(PALETTE.rail), bump).add(rgb(PALETTE.railLight).mul(lip.mul(0.28)));
  const cell = (spacing: number): N<'float'> => {
    const g = abs(fract(q.div(spacing).add(0.5)).sub(0.5)).mul(spacing);
    return min(g.x, g.y);
  };
  const fine = smoothstep(5, 9, float(0.01).div(px.mul(dpr)));
  const r = length(q), along = atan(q.y, q.x.add(1e-9)).mul(d.link + d.reach), dash = (TAU * (d.link + d.reach)) / 120;
  const dashes = fill(abs(fract(along.div(dash)).sub(0.5)).mul(dash).sub(dash * 0.22), px);
  let top: N<'vec3'> = rgb(PALETTE.plate).mul(smoothstep(0, d.plate * 1.3, r).oneMinus().mul(0.08).add(0.96)).mul(fractal(q.mul(160), 2).mul(0.025).add(1));
  top = mix(top, rgb(PALETTE.minor), stroke(cell(0.01), 1, px, dpr).mul(fine).mul(0.75));
  top = mix(top, rgb(PALETTE.major), stroke(cell(0.05), 1.2, px, dpr).mul(0.9));
  top = mix(top, rgb(PALETTE.workspace), stroke(r.sub(d.link + d.reach), 1.3, px, dpr).mul(dashes));
  top = top.mul(smoothstep(-0.018, 0, plate).mul(-0.14).add(1));
  material.colorNode = lit(mix(mix(desk, rail, fill(rim, px)), top, fill(plate, px)));
  return material;
}

export interface TargetRig {
  at: Point;
  old: Point;
  tip: Point;
  spawn: Level;
  fade: Level;
  won: Level;
  round: Level;
  contact: Level;
  clock: Level;
}

export function markMaterial(d: Dims, rig: TargetRig, dpr: Level): THREE.MeshBasicNodeMaterial {
  const q = meters(), px = footprint(q.x);
  let layer = blank();
  const ro = length(q.sub(rig.old)), gone = rig.fade.oneMinus(), spread = mix(float(0.9), float(2.4), rig.won);
  layer = over(layer, mix(rgb(PALETTE.target), rgb(PALETTE.amber), rig.won), stroke(ro.sub(gone.mul(spread.mul(d.ring)).add(d.ring)), mix(float(1.4), float(3), rig.won), px, dpr).mul(rig.fade.mul(0.85)));
  layer = over(layer, rgb(PALETTE.target).mul(0.9), fill(ro.sub(rig.fade.mul(d.target)), px).mul(rig.fade.mul(0.7)));
  const p = q.sub(rig.at);
  layer = over(layer, vec3(0), soft(length(p.sub(vec2(d.shadow.x, d.shadow.y))).sub(d.target), d.blur).mul(0.3));
  const toward = rig.at.sub(rig.tip), span = length(toward), way = toward.div(max(span, 1e-6));
  const from = rig.tip.add(way.mul(d.tip * 1.7)), reach = span.sub(d.tip * 1.7 + d.ring * 1.2);
  const e = q.sub(from), s = dot(e, way), off = abs(e.x.mul(way.y).sub(e.y.mul(way.x)));
  const gap = d.ring * 0.34, beat = fill(abs(fract(s.div(gap).sub(rig.clock.mul(2.2))).sub(0.5)).mul(gap).sub(gap * 0.2), px);
  const inside = fill(s.negate(), px).mul(fill(s.sub(reach), px)).mul(smoothstep(0, d.ring * 0.4, reach));
  layer = over(layer, rgb(PALETTE.target), stroke(off, 1.4, px, dpr).mul(beat).mul(inside).mul(0.8));
  return finish(sheet(), layer);
}

export function targetMaterial(d: Dims, rig: TargetRig, dpr: Level): THREE.MeshBasicNodeMaterial {
  const q = meters(), px = footprint(q.x);
  let layer = blank();
  const p = q.sub(rig.at), r = length(p), ring = d.ring, landing = rig.spawn.oneMinus();
  const clockwise = atan(p.x, p.y.add(1e-9)), turn = select(clockwise.lessThan(0), clockwise.add(TAU), clockwise).div(TAU);
  const done = fill(turn.sub(rig.round).mul(r).mul(TAU), px);
  const hue = mix(rgb(PALETTE.target), rgb(PALETTE.amber), rig.contact);
  layer = over(layer, rgb(PALETTE.target), stroke(r.sub(landing.mul(landing).mul(ring * 2.2).add(ring)), 2, px, dpr).mul(landing));
  layer = over(layer, hue, stroke(r.sub(ring), 1.2, px, dpr).mul(0.38));
  layer = over(layer, hue, stroke(r.sub(ring), 2.6, px, dpr).mul(done));
  layer = over(layer, rgb(PALETTE.gold).mul(1.6), exp(abs(r.sub(ring)).div(d.ring * 0.1).negate()).mul(rig.contact.mul(0.7)));
  const bead = ball(q, rig.at, d.target), cover = fill(bead.d, px);
  const body = shade(bead.n, rgb(PALETTE.target).mul(rig.contact.mul(0.5).add(1.25)), 0.9);
  layer = over(layer, mix(body, rgb(PALETTE.targetDark), stroke(bead.d, 1.1, px, dpr).mul(cover)), cover);
  return finish(sheet(), layer);
}

export interface ArmRig {
  elbow: Point;
  end: Point;
  tip: Point;
  was: [Point, Point, Point];
  ghost: Level;
  torque: Point;
  heading: Point;
  stop: Level;
}

interface Parts { base: N<'float'>; link: Round; forearm: Round; hub: Round; shoulder: Round; tip: Round }

function parts(q: N<'vec2'>, d: Dims, elbow: N<'vec2'>, end: N<'vec2'>, tip: N<'vec2'>): Parts {
  const origin = vec2(0, 0);
  return {
    base: length(q).sub(d.base),
    link: capsule(q, origin, elbow, d.rod),
    forearm: capsule(q, elbow, end, d.rod),
    hub: ball(q, elbow, d.hub),
    shoulder: ball(q, origin, d.shoulder),
    tip: ball(q, tip, d.tip),
  };
}

const union = (k: Parts): N<'float'> => min(min(min(k.base, k.link.d), min(k.forearm.d, k.hub.d)), min(k.shoulder.d, k.tip.d));

function bolts(q: N<'vec2'>, centre: N<'vec2'>, heading: N<'float'>, radius: number, size: number, count: number): N<'float'> {
  const v = q.sub(centre), sector = TAU / count, a = atan(v.y, v.x.add(1e-9)).sub(heading), k = a.div(sector).add(0.5).floor();
  const b = heading.add(k.mul(sector));
  return length(v.sub(vec2(cos(b), sin(b)).mul(radius))).sub(size);
}

export function armMaterial(d: Dims, rig: ArmRig, dpr: Level): THREE.MeshBasicNodeMaterial {
  const q = meters(), px = footprint(q.x), origin = vec2(0, 0);
  let layer = blank();

  const cast = union(parts(q.sub(vec2(d.shadow.x, d.shadow.y)), d, rig.elbow, rig.end, rig.tip));
  layer = over(layer, vec3(0), soft(cast, d.blur).mul(0.32));

  const ghost = union(parts(q, d, rig.was[0], rig.was[1], rig.was[2]));
  layer = over(layer, mix(rgb(PALETTE.link), rgb(PALETTE.linkDark), stroke(ghost, 1.4, px, dpr)), fill(ghost, px).mul(0.28).max(stroke(ghost, 1.4, px, dpr).mul(0.9)).mul(rig.ghost));

  const k = parts(q, d, rig.elbow, rig.end, rig.tip);
  const edge = (distance: N<'float'>, width = 1.2): N<'float'> => stroke(distance, width, px, dpr).mul(fill(distance, px));

  let base: N<'vec3'> = mix(rgb(PALETTE.baseDark), rgb(PALETTE.base), smoothstep(-d.base * 0.25, 0, k.base).oneMinus());
  base = mix(base, rgb(PALETTE.bolt), fill(bolts(q, origin, float(Math.PI / 4), d.base * 0.72, d.base * 0.1, 4), px));
  layer = over(layer, mix(base, rgb(PALETTE.baseDark), edge(k.base)), fill(k.base, px));

  for (const bar of [k.link, k.forearm]) {
    const tube = shade(bar.n, rgb(PALETTE.link), 0.55);
    layer = over(layer, mix(tube, rgb(PALETTE.linkDark), edge(bar.d)), fill(bar.d, px));
  }

  const joint = (round: Round, centre: N<'vec2'>, heading: N<'float'>, radius: number): N<'vec3'> => {
    let face: N<'vec3'> = mix(rgb(PALETTE.hubDark), rgb(PALETTE.hub), smoothstep(-radius * 0.3, 0, round.d).oneMinus());
    face = mix(face, rgb(PALETTE.steel).mul(0.55), fill(bolts(q, centre, heading, radius * 0.62, radius * 0.12, 3), px));
    face = mix(face, shade(round.n.div(0.36), rgb(PALETTE.steel), 0.8), fill(length(q.sub(centre)).sub(radius * 0.36), px));
    return mix(face, rgb(PALETTE.hubDark), edge(round.d));
  };
  const lamp = stroke(k.hub.d.add(d.hub * 0.2), 2, px, dpr).mul(rig.stop);
  layer = over(layer, mix(joint(k.hub, rig.elbow, rig.heading.y, d.hub), rgb(PALETTE.stop).mul(1.6), lamp), fill(k.hub.d, px));
  layer = over(layer, joint(k.shoulder, origin, rig.heading.x, d.shoulder), fill(k.shoulder.d, px));

  return finish(sheet(), layer);
}

export function tipMaterial(d: Dims, rig: ArmRig, accent: number, dpr: Level): THREE.MeshBasicNodeMaterial {
  const q = meters(), px = footprint(q.x);
  let layer = blank();
  const tip = ball(q, rig.tip, d.tip), pearl = shade(tip.n, rgb(accent), 1);
  layer = over(layer, mix(pearl, rgb(PALETTE.tipDark), stroke(tip.d, 1.3, px, dpr).mul(fill(tip.d, px))), fill(tip.d, px));
  const sweep = rig.torque.mul(SWEEP);
  const head = d.link * 0.085, flare = d.link * 0.06, width = d.link * 0.022;
  const shoulderArc = arc(q, d.shoulderArc, rig.heading.x, sweep.x, width, head, flare);
  const elbowArc = arc(q.sub(rig.elbow), d.elbowArc, rig.heading.y, sweep.y, width, head, flare);
  const shown = (t: N<'float'>) => smoothstep(0.004, 0.02, abs(t));
  const effort = max(fill(shoulderArc, px).mul(shown(rig.torque.x)), fill(elbowArc, px).mul(shown(rig.torque.y)));
  layer = over(layer, rgb(PALETTE.torque), effort.mul(0.92));
  return finish(sheet(), layer);
}

export interface Ribbon {
  mesh: THREE.Mesh;
  level: Level;
  draw(points: Float32Array, count: number, width: number): void;
}

export function ribbon(capacity: number, color: number, level: Level, lit: Lit): Ribbon {
  const position = new THREE.BufferAttribute(new Float32Array(capacity * 6), 3).setUsage(THREE.DynamicDrawUsage);
  const ink = new THREE.BufferAttribute(new Float32Array(capacity * 4), 2).setUsage(THREE.DynamicDrawUsage);
  const index: number[] = [];
  for (let i = 0; i < capacity - 1; i++) index.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', position);
  geometry.setAttribute('aInk', ink);
  geometry.setIndex(index);
  geometry.setDrawRange(0, 0);
  const material = sheet();
  const state = attribute<'vec2'>('aInk', 'vec2'), side = state.x;
  material.colorNode = lit(rgb(color));
  material.opacityNode = clamp(float(1).sub(abs(side)).div(max(fwidth(side), 1e-4)), 0, 1).mul(state.y).mul(level);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  const out = position.array as Float32Array, marks = ink.array as Float32Array;
  return {
    mesh,
    level,
    draw(points, count, width) {
      let nx = 0, ny = 1;
      for (let i = 0; i < count; i++) {
        const a = Math.max(0, i - 1), b = Math.min(count - 1, i + 1);
        const dx = points[2 * b]! - points[2 * a]!, dy = points[2 * b + 1]! - points[2 * a + 1]!, chord = Math.hypot(dx, dy);
        if (chord > 1e-7) {
          nx = -dy / chord;
          ny = dx / chord;
        }
        const x = points[2 * i]!, y = points[2 * i + 1]!, o = 6 * i, strength = count > 1 ? 0.4 + 0.6 * (i / (count - 1)) ** 1.5 : 1;
        out[o] = (x + nx * width * FEATHER) * SCALE;
        out[o + 1] = 0;
        out[o + 2] = -(y + ny * width * FEATHER) * SCALE;
        out[o + 3] = (x - nx * width * FEATHER) * SCALE;
        out[o + 4] = 0;
        out[o + 5] = -(y - ny * width * FEATHER) * SCALE;
        marks[4 * i] = FEATHER;
        marks[4 * i + 1] = strength;
        marks[4 * i + 2] = -FEATHER;
        marks[4 * i + 3] = strength;
      }
      position.needsUpdate = true;
      ink.needsUpdate = true;
      geometry.setDrawRange(0, Math.max(0, count - 1) * 6);
    },
  };
}

export const flat = (width: number, depth: number): THREE.PlaneGeometry => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const disc = (radius: number): THREE.BufferGeometry => new THREE.CircleGeometry(radius, 32).rotateX(-Math.PI / 2);
