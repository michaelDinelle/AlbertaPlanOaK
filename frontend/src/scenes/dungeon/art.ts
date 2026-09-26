import * as THREE from 'three/webgpu';
import { abs, atan, attribute, clamp, cos, dot, float, floor, fract, length, max, min, mod, select, sign, sin, sqrt, vec2, vec3 } from 'three/tsl';
import { instancedMesh } from '@kit/instances';
import { sdBox, sdCircle, sdSegment } from '@kit/shapes2d';

export type N<T extends string> = THREE.Node<T>;
export type Lit = (albedo: N<'vec3'>) => N<'vec3'>;
type Scalar = N<'float'> | number;

export const flat = (width: number, depth = width) => new THREE.PlaneGeometry(width, depth).rotateX(-Math.PI / 2);

export const ellipse = (p: N<'vec2'>, cx: number, cy: number, rx: Scalar, ry: Scalar): N<'float'> => {
  const r = vec2(rx, ry);
  return length(p.sub(vec2(cx, cy)).div(r)).sub(1).mul(min(r.x, r.y));
};

export function smin(a: N<'float'>, b: N<'float'>, k: number): N<'float'> {
  const h = max(float(k).sub(abs(a.sub(b))), 0).div(k);
  return min(a, b).sub(h.mul(h).mul(k / 4));
}

export const turn = (p: N<'vec2'>, angle: Scalar): N<'vec2'> => {
  const c = cos(angle), s = sin(angle);
  return vec2(p.x.mul(c).sub(p.y.mul(s)), p.x.mul(s).add(p.y.mul(c)));
};

export const ring = (d: N<'float'>, width: Scalar): N<'float'> => abs(d).sub(width);

export const hash = (p: N<'vec2'>): N<'float'> => fract(sin(dot(p, vec2(127.1, 311.7))).mul(43758.5453));

export class Layers {
  pre: N<'vec3'> = vec3(0);
  alpha: N<'float'> = float(0);

  over(cover: N<'float'>, color: N<'vec3'>): this {
    this.pre = color.mul(cover).add(this.pre.mul(cover.oneMinus()));
    this.alpha = cover.add(this.alpha.mul(cover.oneMinus()));
    return this;
  }

  material(fade: N<'float'> | null = null, additive = false): THREE.MeshBasicNodeMaterial {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, fog: false });
    material.colorNode = this.pre.div(max(this.alpha, 1e-4));
    material.opacityNode = fade ? this.alpha.mul(fade) : this.alpha;
    if (additive) material.blending = THREE.AdditiveBlending;
    return material;
  }
}

export class Batch {
  readonly mesh: THREE.InstancedMesh;
  private readonly channels = new Map<string, THREE.InstancedBufferAttribute>();
  private readonly matrix = new THREE.Matrix4();
  private readonly quaternion = new THREE.Quaternion();
  private readonly position = new THREE.Vector3();
  private readonly scale = new THREE.Vector3();
  private readonly axis = new THREE.Vector3(0, 1, 0);
  used = 0;

  constructor(readonly capacity: number, size: number | readonly [number, number], names: readonly string[], paint: (p: N<'vec2'>, channel: (name: string) => N<'vec4'>) => THREE.Material, order: number) {
    const [width, depth] = typeof size === 'number' ? [size, size] : size;
    const geometry = flat(width, depth);
    for (const name of names) {
      const attr = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, attr);
      this.channels.set(name, attr);
    }
    const p = vec2(attribute<'vec2'>('uv', 'vec2').sub(0.5).mul(vec2(width, depth)));
    this.mesh = instancedMesh(geometry, paint(p, (name) => attribute<'vec4'>(name, 'vec4')), capacity);
    this.mesh.renderOrder = order;
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
  }

  begin(): void {
    this.used = 0;
  }

  put(x: number, z: number, scale = 1, angle = 0, lift = 0): number {
    if (this.used >= this.capacity) throw new Error(`${this.mesh.name || 'batch'} holds ${this.capacity} sprites`);
    const i = this.used++;
    this.quaternion.setFromAxisAngle(this.axis, angle);
    this.mesh.setMatrixAt(i, this.matrix.compose(this.position.set(x, lift, z), this.quaternion, this.scale.setScalar(Math.max(1e-4, scale))));
    return i;
  }

  set(name: string, i: number, x: number, y = 0, z = 0, w = 0): void {
    this.channels.get(name)!.setXYZW(i, x, y, z, w);
  }

  end(): void {
    this.mesh.count = this.used;
    this.mesh.instanceMatrix.needsUpdate = true;
    for (const attr of this.channels.values()) attr.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

const SEGMENTS: readonly (readonly [number, number, number, number])[] = [
  [-1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, -1], [-1, -1, 1, -1], [-1, -1, -1, 0], [-1, 0, -1, 1], [-1, 0, 1, 0], [0, 0.55, 0, -0.55],
];
const MASKS = [0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f, 0xc0, 0x40] as const;
const PLUS = 10;
const MINUS = 11;

function glyph(p: N<'vec2'>, code: N<'float'>): N<'float'> {
  let mask: N<'float'> = float(0);
  for (let c = MASKS.length - 1; c >= 0; c--) mask = select(abs(code.sub(c)).lessThan(0.5), float(MASKS[c]!), mask);
  let d: N<'float'> = float(1e3);
  SEGMENTS.forEach(([ax, ay, bx, by], k) => {
    const on = mod(floor(mask.div(2 ** k)), 2).greaterThan(0.5);
    d = min(d, select(on, sdSegment(p, vec2(ax * 0.3, ay * 0.46), vec2(bx * 0.3, by * 0.46), 0.075), float(1e3)));
  });
  return d;
}

export function text(p: N<'vec2'>, codes: N<'vec4'>, count: N<'float'>, advance = 0.86): N<'float'> {
  let d: N<'float'> = float(1e3);
  const start = count.sub(1).mul(advance * -0.5);
  ([codes.x, codes.y, codes.z, codes.w] as N<'float'>[]).forEach((code, i) => {
    const shown = float(i).lessThan(count).and(code.greaterThanEqual(0));
    d = min(d, select(shown, glyph(p.sub(vec2(start.add(i * advance), 0)), code), float(1e3)));
  });
  return d;
}

export function codesOf(value: number, signed: '' | '+' | '-', out: number[]): number {
  const digits = String(Math.min(signed ? 999 : 9999, Math.max(0, Math.round(value))));
  out.fill(-1);
  let n = 0;
  if (signed) out[n++] = signed === '+' ? PLUS : MINUS;
  for (const ch of digits) if (n < 4) out[n++] = ch.charCodeAt(0) - 48;
  return n;
}

export function heart(p: N<'vec2'>, size: number): N<'float'> {
  const q = vec2(abs(p.x), p.y.add(size * 0.55)).div(size);
  const upper = length(q.sub(vec2(0.25, 0.75))).sub(Math.SQRT2 / 4);
  const toTop = q.sub(vec2(0, 1)), toSide = q.sub(max(q.x.add(q.y), 0).mul(0.5));
  const lower = sqrt(min(dot(toTop, toTop), dot(toSide, toSide))).mul(sign(q.x.sub(q.y)));
  return select(q.x.add(q.y).greaterThan(1), upper, lower).mul(size);
}

export function flame(p: N<'vec2'>, size: Scalar): N<'float'> {
  const q = p.div(size);
  const bulb = sdCircle(q.sub(vec2(0, -0.14)), 0.26);
  const tip = sdSegment(q, vec2(0, -0.1), vec2(0.04, 0.46), 0.02);
  return smin(bulb, tip, 0.34).mul(size);
}

export function blaze(p: N<'vec2'>, size: number): N<'float'> {
  const q = p.div(size);
  const body = smin(sdCircle(q.sub(vec2(0, -0.17)), 0.23), sdSegment(q, vec2(0.02, -0.1), vec2(0.05, 0.46), 0.015), 0.3);
  const left = smin(sdCircle(q.sub(vec2(-0.15, -0.16)), 0.12), sdSegment(q, vec2(-0.15, -0.12), vec2(-0.25, 0.16), 0.012), 0.16);
  const right = smin(sdCircle(q.sub(vec2(0.15, -0.2)), 0.1), sdSegment(q, vec2(0.15, -0.16), vec2(0.24, 0.06), 0.012), 0.14);
  return min(min(body, left), right).mul(size);
}

export function star(p: N<'vec2'>, size: number): N<'float'> {
  const k1 = vec2(0.809016994375, -0.587785252292), k2 = vec2(-0.809016994375, -0.587785252292), r = 0.46, rf = 0.45;
  let q: N<'vec2'> = vec2(abs(p.x), p.y).div(size);
  q = q.sub(k1.mul(max(dot(k1, q), 0).mul(2)));
  q = q.sub(k2.mul(max(dot(k2, q), 0).mul(2)));
  q = vec2(abs(q.x), q.y.sub(r));
  const ba = vec2(0.587785252292 * rf, 0.809016994375 * rf - 1);
  const h = clamp(dot(q, ba).div(dot(ba, ba)), 0, r);
  return length(q.sub(ba.mul(h))).mul(sign(q.y.mul(ba.x).sub(q.x.mul(ba.y)))).mul(size);
}

export function skull(p: N<'vec2'>, size: number): N<'float'> {
  const q = p.div(size);
  const head = smin(sdCircle(q.sub(vec2(0, 0.06)), 0.3), sdBox(q.sub(vec2(0, -0.2)), vec2(0.15, 0.12), 0.04), 0.1);
  const sockets = min(sdCircle(q.sub(vec2(-0.12, 0.04)), 0.085), sdCircle(q.sub(vec2(0.12, 0.04)), 0.085));
  const nose = sdBox(q.sub(vec2(0, -0.1)), vec2(0.03, 0.05), 0.01);
  return max(head, min(sockets, nose).negate()).mul(size);
}

export function snowflake(p: N<'vec2'>, size: number): N<'float'> {
  let d: N<'float'> = float(1e3);
  for (let k = 0; k < 3; k++) {
    const a = (k * Math.PI) / 3, dx = Math.cos(a) * size * 0.46, dy = Math.sin(a) * size * 0.46;
    d = min(d, sdSegment(p, vec2(-dx, -dy), vec2(dx, dy), size * 0.055));
    for (const s of [-1, 1]) {
      const bx = dx * 0.62 * s, by = dy * 0.62 * s, ca = Math.cos(a + 0.8) * size * 0.14 * s, sa = Math.sin(a + 0.8) * size * 0.14 * s, cb = Math.cos(a - 0.8) * size * 0.14 * s, sb = Math.sin(a - 0.8) * size * 0.14 * s;
      d = min(d, min(sdSegment(p, vec2(bx, by), vec2(bx + ca, by + sa), size * 0.04), sdSegment(p, vec2(bx, by), vec2(bx + cb, by + sb), size * 0.04)));
    }
  }
  return d;
}

export function spiral(p: N<'vec2'>, size: Scalar): N<'float'> {
  const r = length(p).div(size), a = atan(p.y, p.x), pitch = 0.085;
  const t = r.div(pitch).sub(a.div(2 * Math.PI));
  const arm = abs(fract(t).sub(0.5)).mul(pitch).sub(0.018);
  return max(arm, r.sub(0.44)).mul(size);
}

export function blindEye(p: N<'vec2'>, size: number): N<'float'> {
  const lid = max(sdCircle(p.sub(vec2(0, size * -0.3)), size * 0.46), sdCircle(p.sub(vec2(0, size * 0.3)), size * 0.46));
  const slash = sdSegment(p, vec2(size * -0.4, size * -0.34), vec2(size * 0.4, size * 0.34), size * 0.055);
  return min(min(ring(lid, size * 0.04), sdCircle(p, size * 0.12)), slash);
}

export function cross(p: N<'vec2'>, size: number): N<'float'> {
  return min(sdBox(p, vec2(size * 0.14, size * 0.42), size * 0.03), sdBox(p, vec2(size * 0.42, size * 0.14), size * 0.03));
}

export function crown(p: N<'vec2'>, size: number): N<'float'> {
  const q = p.div(size);
  const band = sdBox(q.sub(vec2(0, -0.2)), vec2(0.4, 0.09), 0.02);
  const peak = abs(fract(q.x.div(0.4)).sub(0.5)).mul(2);
  const points = max(max(abs(q.x).sub(0.4), float(-0.14).sub(q.y)), q.y.sub(peak.mul(0.34).add(-0.06)).mul(0.7));
  return min(band, points).mul(size);
}

export function shield(p: N<'vec2'>, size: number): N<'float'> {
  const q = p.div(size);
  const top = sdBox(q.sub(vec2(0, 0.12)), vec2(0.34, 0.24), 0.05);
  const point = max(sdCircle(q.sub(vec2(-0.3, 0.12)), 0.64), sdCircle(q.sub(vec2(0.3, 0.12)), 0.64));
  return min(top, max(point, q.y.sub(0.12))).mul(size);
}

export function moon(p: N<'vec2'>, size: number): N<'float'> {
  return max(sdCircle(p, size * 0.36), sdCircle(p.sub(vec2(size * 0.17, size * 0.12)), size * 0.3).negate());
}

export function stairs(p: N<'vec2'>, size: number): N<'float'> {
  let d: N<'float'> = float(1e3);
  for (let k = 0; k < 3; k++) d = min(d, sdBox(p.sub(vec2(size * (-0.24 + 0.16 * k), size * (0.22 - 0.2 * k))), vec2(size * (0.2 - 0.02 * k) , size * 0.06), size * 0.02));
  return d;
}

export function loaf(p: N<'vec2'>, size: number): N<'float'> {
  return sdBox(p, vec2(size * 0.38, size * 0.2), size * 0.18);
}

export function flask(p: N<'vec2'>, size: number): N<'float'> {
  const body = sdCircle(p.sub(vec2(0, size * -0.1)), size * 0.3);
  const neck = sdBox(p.sub(vec2(0, size * 0.26)), vec2(size * 0.09, size * 0.12), size * 0.02);
  return smin(body, neck, size * 0.08);
}

export function tomb(p: N<'vec2'>, size: number): N<'float'> {
  const q = p.div(size);
  const slab = min(sdBox(q.sub(vec2(0, -0.12)), vec2(0.3, 0.3)), sdCircle(q.sub(vec2(0, 0.18)), 0.3));
  return max(slab, q.y.sub(0.48)).mul(size);
}

export const POTION_TINTS = [0xe23a4e, 0xf2a93b, 0xb6e04a, 0x33c9a7, 0x4a8ff0, 0x9b62f2, 0xf06fb6, 0xe8e2d0] as const;
export const POTION_NAMES = ['crimson', 'amber', 'lime', 'teal', 'azure', 'violet', 'rose', 'pearl'] as const;
export const ELEMENT_TINTS = [0xc9b49a, 0xff7a3a, 0x9fe2ff, 0x9ee04e, 0xb58cff] as const;
export const EARTH_TINTS = [0x9b8f84, 0x86739a, 0x94a060, 0xb58550, 0xc0a476, 0x8f6d5c, 0x76876c, 0xa3a6b4] as const;

export const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
export const ease = (u: number) => u * u * (3 - 2 * u);
export const headingOf = (dx: number, dy: number) => Math.atan2(-dx, -dy);
