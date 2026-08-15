import * as THREE from 'three';
import { circleOutline, outlineAt, outlineCount, type Outline } from './outline';

export interface SweepPath {
  at(t: number, out: THREE.Vector3): THREE.Vector3;
}

export type SweepRadius = (t: number) => number;
export type SweepTwist = (t: number) => number;
export type SweepColor = (t: number, around: number, out: THREE.Color) => void;

export interface SweepOptions {
  path: SweepPath;
  rings: number;
  shape?: Outline;
  radius?: SweepRadius | number;
  twist?: SweepTwist;
  color?: SweepColor;
  capStart?: boolean;
  capEnd?: boolean;
  name?: string;
}

export function linePath(a: THREE.Vector3, b: THREE.Vector3): SweepPath {
  return { at: (t, out) => out.copy(a).lerp(b, t) };
}

export function curvePath(points: readonly THREE.Vector3[], tension = 0.5): SweepPath {
  const curve = new THREE.CatmullRomCurve3([...points], false, 'catmullrom', tension);
  return { at: (t, out) => curve.getPoint(t, out) };
}

export function arcPath(
  centre: THREE.Vector3,
  radius: number,
  from: number,
  to: number,
  axis: 'x' | 'y' | 'z' = 'y',
): SweepPath {
  return {
    at(t, out) {
      const a = from + (to - from) * t;
      const c = Math.cos(a) * radius;
      const s = Math.sin(a) * radius;
      if (axis === 'y') out.set(centre.x + c, centre.y, centre.z + s);
      else if (axis === 'x') out.set(centre.x, centre.y + c, centre.z + s);
      else out.set(centre.x + c, centre.y + s, centre.z);
      return out;
    },
  };
}

const UP = new THREE.Vector3(0, 1, 0);
const ALT = new THREE.Vector3(1, 0, 0);

export function sweep(opts: SweepOptions): THREE.BufferGeometry {
  const rings = Math.max(2, Math.floor(opts.rings));
  const shape = opts.shape ?? circleOutline(8);
  const around = outlineCount(shape);
  const radiusOf: SweepRadius =
    typeof opts.radius === 'number' ? () => opts.radius as number : (opts.radius ?? (() => 1));
  const twistOf = opts.twist ?? (() => 0);

  const verts = rings * around;
  const position = new Float32Array(verts * 3);
  const normal = new Float32Array(verts * 3);
  const color = opts.color ? new Float32Array(verts * 3) : null;

  const here = new THREE.Vector3();
  const next = new THREE.Vector3();
  const tangent = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  const bin = new THREE.Vector3();
  const point = new THREE.Vector3();
  const radial = new THREE.Vector3();
  const tint = new THREE.Color();
  const xy: [number, number] = [0, 0];
  const ref = new THREE.Vector3();

  for (let r = 0; r < rings; r++) {
    const t = rings === 1 ? 0 : r / (rings - 1);
    opts.path.at(t, here);
    opts.path.at(Math.min(1, t + 1 / (rings - 1 || 1)), next);
    tangent.copy(next).sub(here);
    if (tangent.lengthSq() < 1e-12) {
      opts.path.at(Math.max(0, t - 1 / (rings - 1 || 1)), next);
      tangent.copy(here).sub(next);
    }
    if (tangent.lengthSq() < 1e-12) tangent.set(0, 1, 0);
    tangent.normalize();
    ref.copy(Math.abs(tangent.dot(UP)) > 0.94 ? ALT : UP);
    bin.crossVectors(tangent, ref).normalize();
    nrm.crossVectors(bin, tangent).normalize();

    const rad = radiusOf(t);
    const spin = twistOf(t);
    const cs = Math.cos(spin);
    const sn = Math.sin(spin);

    for (let a = 0; a < around; a++) {
      outlineAt(shape, a, xy);
      const px = xy[0] * cs - xy[1] * sn;
      const py = xy[0] * sn + xy[1] * cs;
      radial.copy(bin).multiplyScalar(px).addScaledVector(nrm, py);
      point.copy(here).addScaledVector(radial, rad);
      const i = (r * around + a) * 3;
      position[i] = point.x;
      position[i + 1] = point.y;
      position[i + 2] = point.z;
      radial.normalize();
      normal[i] = radial.x;
      normal[i + 1] = radial.y;
      normal[i + 2] = radial.z;
      if (color) {
        opts.color?.(t, a / around, tint);
        color[i] = tint.r;
        color[i + 1] = tint.g;
        color[i + 2] = tint.b;
      }
    }
  }

  const quads = (rings - 1) * around;
  const index = new Uint32Array(quads * 6);
  let w = 0;
  for (let r = 0; r < rings - 1; r++) {
    for (let a = 0; a < around; a++) {
      const a1 = (a + 1) % around;
      const p0 = r * around + a;
      const p1 = r * around + a1;
      const p2 = (r + 1) * around + a;
      const p3 = (r + 1) * around + a1;
      index[w++] = p0;
      index[w++] = p2;
      index[w++] = p1;
      index[w++] = p1;
      index[w++] = p2;
      index[w++] = p3;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normal, 3));
  if (color) geometry.setAttribute('color', new THREE.BufferAttribute(color, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.name = opts.name ?? 'oakSweep';
  geometry.computeBoundingSphere();
  return geometry;
}

export function revolve(
  profile: readonly (readonly [number, number])[],
  segments: number,
  opts: { color?: SweepColor; name?: string } = {},
): THREE.BufferGeometry {
  const n = Math.max(2, profile.length);
  const path: SweepPath = {
    at(t, out) {
      const f = t * (n - 1);
      const i = Math.min(n - 2, Math.floor(f));
      const k = f - i;
      const a = profile[i];
      const b = profile[i + 1];
      if (!a || !b) return out.set(0, 0, 0);
      return out.set(0, a[1] + (b[1] - a[1]) * k, 0);
    },
  };
  const radiusOf: SweepRadius = (t) => {
    const f = t * (n - 1);
    const i = Math.min(n - 2, Math.floor(f));
    const k = f - i;
    const a = profile[i];
    const b = profile[i + 1];
    if (!a || !b) return 0;
    return a[0] + (b[0] - a[0]) * k;
  };
  return sweep({
    path,
    rings: n,
    shape: circleOutline(Math.max(3, Math.floor(segments))),
    radius: radiusOf,
    ...(opts.color ? { color: opts.color } : {}),
    name: opts.name ?? 'oakRevolve',
  });
}
