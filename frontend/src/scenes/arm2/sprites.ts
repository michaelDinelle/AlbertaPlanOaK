import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dFdx, dFdy, dot, exp, float, fract, length, max, min, mix, positionWorld, round, select, sign, sin, sqrt, uniform, uv, vec2 } from 'three/tsl';
import { rgb, sdBox } from '@kit/shapes2d';

type N<T extends string> = THREE.Node<T>;
type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Paint = (p: N<'vec2'>, px: N<'float'>) => readonly [N<'vec3'>, N<'float'>];

export const DPR = uniform(1);

export const PALETTE = {
  wall: 0x141e33,
  disk: 0x1c2a47,
  grid: 0x3b5584,
  ring: 0x7ea2e0,
  plate: 0x3a4a66,
  steel: 0xdde6f2,
  pocket: 0x1b2638,
  edge: 0x0d1422,
  anodized: 0x243149,
  bolt: 0x8595ad,
  brass: 0xd8a94f,
  gold: 0xffd166,
  effort: 0xff9a4a,
  closing: 0x86f0b4,
  opening: 0xff8a78,
} as const;

const footprint = (v: N<'float'>): N<'float'> => max(length(vec2(dFdx(v), dFdy(v))), 1e-7);
const fill = (d: N<'float'>, px: N<'float'>): N<'float'> => clamp(float(0.5).sub(d.div(px)), 0, 1);
const line = (d: N<'float'>, width: number, px: N<'float'>): N<'float'> => clamp(DPR.mul(width / 2).add(0.5).sub(abs(d).div(px)), 0, 1);

export function flat(width: number, height: number): THREE.PlaneGeometry {
  return new THREE.PlaneGeometry(width, height).rotateX(-Math.PI / 2);
}

export function sprite(width: number, height: number, order: number, paint: Paint): THREE.Mesh {
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
  const p = uv().sub(0.5).mul(vec2(width, height));
  const [color, alpha] = paint(p, footprint(p.x));
  material.colorNode = color;
  material.opacityNode = alpha;
  const mesh = new THREE.Mesh(flat(width, height), material);
  mesh.renderOrder = order;
  mesh.frustumCulled = false;
  return mesh;
}

export function backdrop(reach: number, elbow: number, lit: Lit): Paint {
  return (_p, px) => {
    const q = vec2(positionWorld.x, positionWorld.z.negate()), r = length(q);
    const grid = (spacing: number) => {
      const g = abs(fract(q.div(spacing).add(0.5)).sub(0.5)).mul(spacing);
      return min(g.x, g.y);
    };
    const angle = atan(q.y, q.x), unit = Math.PI / 18, k = round(angle.div(unit)), index = abs(k);
    const tickLength = select(index.mod(9).lessThan(0.5), float(0.17), select(index.mod(3).lessThan(0.5), float(0.11), float(0.06)));
    const tick = line(r.mul(abs(sin(angle.sub(k.mul(unit))))), 1.2, px).mul(fill(max(r.sub(reach).negate(), r.sub(tickLength.add(reach))), px));
    const dashes = 96, dash = abs(fract(angle.div(2 * Math.PI).mul(dashes)).sub(0.5)).sub(0.25).mul(r.mul(2 * Math.PI / dashes));
    const plate = sdBox(q, vec2(0.2), 0.05), bolt = length(abs(q).sub(0.135)).sub(0.022);
    const inside = fill(r.sub(reach), px);
    let albedo = mix(rgb(PALETTE.wall), rgb(PALETTE.disk), inside);
    albedo = mix(albedo, rgb(PALETTE.grid), max(line(grid(0.25), 1, px).mul(0.45), line(grid(1), 1.2, px).mul(0.9)).mul(inside.mul(0.55).add(0.45)));
    albedo = mix(albedo, rgb(PALETTE.ring), line(r.sub(elbow), 1, px).mul(fill(dash, px)).mul(0.6));
    albedo = mix(albedo, rgb(PALETTE.ring), max(line(r.sub(reach), 1.8, px), tick));
    albedo = mix(albedo, rgb(PALETTE.plate), fill(plate, px));
    albedo = mix(albedo, rgb(PALETTE.edge), line(plate, 2, px).mul(fill(plate, px)));
    albedo = mix(albedo, rgb(PALETTE.bolt), fill(bolt, px));
    const plumb = q.add(vec2(0, reach + 0.3));
    const bob = max(abs(plumb.x).mul(0.866).sub(plumb.y.mul(0.5)), plumb.y).sub(0.04);
    albedo = mix(albedo, rgb(PALETTE.ring), max(fill(bob, px), line(plumb.x, 1.2, px).mul(fill(max(plumb.y.sub(0.13), plumb.y.negate()), px))));
    return [lit(albedo), float(1)];
  };
}

export function link(span: number, half: number, lit: Lit): Paint {
  const pitch = 0.16, holes = Math.max(1, Math.floor((span - 0.3) / pitch) + 1), first = -(holes - 1) / 2;
  return (p, px) => {
    const body = sdBox(p, vec2(span / 2 + half, half), half);
    const slot = clamp(round(p.x.div(pitch).sub(first)), 0, holes - 1).add(first).mul(pitch);
    const hole = length(vec2(p.x.sub(slot), p.y)).sub(half * 0.36);
    const across = p.y.div(half);
    let albedo = rgb(PALETTE.steel).mul(float(1).sub(across.mul(across)).mul(0.35).add(0.75));
    albedo = mix(albedo, rgb(PALETTE.pocket), fill(hole, px));
    albedo = mix(albedo, rgb(PALETTE.edge), line(hole, 1.6, px));
    albedo = mix(albedo, rgb(PALETTE.edge), line(body, 2.6, px));
    return [lit(albedo), fill(body, px)];
  };
}

export function hub(radius: number, accent: number, lit: Lit): Paint {
  return (p, px) => {
    const r = length(p), angle = atan(p.y, p.x), sector = Math.PI / 3, k = round(angle.div(sector));
    const bolt = length(p.sub(vec2(cos(k.mul(sector)), sin(k.mul(sector))).mul(radius * 0.62))).sub(radius * 0.09);
    let albedo = rgb(PALETTE.anodized);
    albedo = mix(albedo, rgb(accent), line(r.sub(radius * 0.86), 2.2, px));
    albedo = mix(albedo, rgb(PALETTE.bolt), fill(bolt, px));
    albedo = mix(albedo, rgb(PALETTE.steel), fill(r.sub(radius * 0.28), px));
    albedo = mix(albedo, rgb(PALETTE.pocket), fill(sdBox(p, vec2(radius * 0.2, radius * 0.045), radius * 0.03), px));
    albedo = mix(albedo, rgb(PALETTE.edge), line(r.sub(radius), 2.6, px));
    return [lit(albedo), fill(r.sub(radius), px)];
  };
}

export function claw(wrist: number, accent: number, lit: Lit): Paint {
  const socket = hub(0.072, accent, lit);
  return (p, px) => {
    const finger = sdBox(vec2(p.x.sub(0.03), abs(p.y).sub(0.072)), vec2(0.055, 0.021), 0.014);
    const palm = sdBox(p.sub(vec2(-0.042, 0)), vec2(0.024, 0.093), 0.014);
    const jaw = min(finger, palm);
    const [joint, jointAlpha] = socket(p.add(vec2(wrist, 0)), px);
    let albedo = rgb(PALETTE.steel).mul(float(0.8).add(fill(jaw.add(0.012), px).mul(0.2)));
    albedo = mix(albedo, rgb(PALETTE.edge), line(jaw, 1.8, px));
    const cross = line(min(abs(p.x), abs(p.y)), 1.4, px).mul(fill(length(p).sub(0.028), px));
    const color = mix(mix(joint, lit(albedo), fill(jaw, px)), rgb(PALETTE.gold).mul(2.2), cross);
    return [color, max(max(fill(jaw, px), jointAlpha), cross)];
  };
}

export function payload(lit: Lit, glow: N<'float'>): Paint {
  return (p, px) => {
    const r = length(p);
    let albedo = rgb(PALETTE.brass).mul(mix(float(1.15), float(0.75), r));
    albedo = mix(albedo, rgb(PALETTE.brass).mul(0.55), line(r.sub(0.7), 1.4, px));
    albedo = mix(albedo, rgb(PALETTE.edge), fill(r.sub(0.13), px));
    albedo = mix(albedo, rgb(PALETTE.edge), line(r.sub(1), 2.4, px));
    return [lit(albedo).add(rgb(PALETTE.brass).mul(glow.mul(2.5))), fill(r.sub(1), px)];
  };
}

export function reticle(arrival: number, core: boolean, alpha: N<'float'>): Paint {
  return (p, px) => {
    const r = length(p);
    const ring = line(r.sub(arrival), 1.8, px);
    const ticks = line(min(abs(p.x), abs(p.y)), 1.6, px).mul(fill(max(float(arrival + 0.03).sub(r), r.sub(arrival + 0.08)), px));
    const shape = max(ring, ticks);
    if (!core) return [rgb(PALETTE.gold).mul(1.6), shape.mul(alpha)];
    const centre = fill(r.sub(0.022), px);
    return [rgb(PALETTE.gold).mul(mix(float(1.6), float(4), centre)), max(max(shape, centre), exp(r.mul(-30)).mul(0.5)).mul(alpha)];
  };
}

export function halo(color: number, alpha: N<'float'>): Paint {
  return (p, px) => [rgb(color).mul(1.8), line(length(p).sub(1), 2.2, px).mul(alpha)];
}

export interface ArcControl {
  axis: THREE.Vector2;
  aperture: THREE.Vector2;
  head: [THREE.Vector2, THREE.Vector2, THREE.Vector2];
}

const sdTriangle = (p: N<'vec2'>, p0: N<'vec2'>, p1: N<'vec2'>, p2: N<'vec2'>): N<'float'> => {
  const e0 = p1.sub(p0), e1 = p2.sub(p1), e2 = p0.sub(p2);
  const v0 = p.sub(p0), v1 = p.sub(p1), v2 = p.sub(p2);
  const q0 = v0.sub(e0.mul(clamp(dot(v0, e0).div(dot(e0, e0)), 0, 1)));
  const q1 = v1.sub(e1.mul(clamp(dot(v1, e1).div(dot(e1, e1)), 0, 1)));
  const q2 = v2.sub(e2.mul(clamp(dot(v2, e2).div(dot(e2, e2)), 0, 1)));
  const s = sign(e0.x.mul(e2.y).sub(e0.y.mul(e2.x)));
  const d = min(min(vec2(dot(q0, q0), s.mul(v0.x.mul(e0.y).sub(v0.y.mul(e0.x)))), vec2(dot(q1, q1), s.mul(v1.x.mul(e1.y).sub(v1.y.mul(e1.x))))), vec2(dot(q2, q2), s.mul(v2.x.mul(e2.y).sub(v2.y.mul(e2.x)))));
  return sqrt(d.x).negate().mul(sign(d.y));
};

export function arc(radius: number, thick: number, control: ArcControl): Paint {
  const axis = uniform(control.axis), aperture = uniform(control.aperture);
  const [a, b, c] = control.head.map((v) => uniform(v));
  return (p, px) => {
    const turned = vec2(p.x.mul(axis.x).sub(p.y.mul(axis.y)), p.x.mul(axis.y).add(p.y.mul(axis.x)));
    const folded = vec2(abs(turned.x), turned.y);
    const shaft = select(aperture.y.mul(folded.x).greaterThan(aperture.x.mul(folded.y)), length(folded.sub(aperture.mul(radius))), abs(length(turned).sub(radius))).sub(thick);
    return [rgb(PALETTE.effort).mul(2.4), fill(min(shaft, sdTriangle(p, a!, b!, c!)), px)];
  };
}

export function tether(span: N<'float'>, arrival: number, color: N<'vec3'>): Paint {
  return (p) => {
    const s = uv().x.mul(span), period = 0.09, along = footprint(s);
    const dash = abs(fract(s.div(period)).sub(0.5)).sub(0.28).mul(period);
    const ends = fill(max(float(0.075).sub(s), s.sub(span.sub(arrival + 0.03))), along);
    return [color, line(p.y, 1.8, footprint(p.y)).mul(fill(dash, along)).mul(ends).mul(0.9)];
  };
}

export function ribbon(points: number, color: number): { mesh: THREE.Mesh; positions: Float32Array; attribute: THREE.BufferAttribute } {
  const positions = new Float32Array(points * 2 * 3), side = new Float32Array(points * 2), age = new Float32Array(points * 2), index: number[] = [];
  for (let i = 0; i < points; i++) {
    side[2 * i] = 1;
    side[2 * i + 1] = -1;
    age[2 * i] = age[2 * i + 1] = i / (points - 1);
    if (i > 0) index.push(2 * i - 2, 2 * i - 1, 2 * i, 2 * i - 1, 2 * i + 1, 2 * i);
  }
  const geometry = new THREE.BufferGeometry();
  const position = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', position);
  geometry.setAttribute('side', new THREE.BufferAttribute(side, 1));
  geometry.setAttribute('age', new THREE.BufferAttribute(age, 1));
  geometry.setIndex(index);
  const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false, side: THREE.DoubleSide });
  const fade = float(1).sub(attribute<'float'>('age', 'float'));
  material.colorNode = rgb(color).mul(fade.mul(1.2).add(0.3));
  const across = attribute<'float'>('side', 'float');
  material.opacityNode = fade.mul(fade).mul(0.8).mul(fill(abs(across).sub(1), footprint(across)));
  material.blending = THREE.AdditiveBlending;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return { mesh, positions, attribute: position };
}
