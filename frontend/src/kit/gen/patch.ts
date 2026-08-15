import * as THREE from 'three';

export type PatchPoint = (u: number, v: number, out: THREE.Vector3) => void;
export type PatchColor = (u: number, v: number, out: THREE.Color) => void;

export interface PatchOptions {
  point: PatchPoint;
  segmentsU: number;
  segmentsV: number;
  color?: PatchColor;
  closedU?: boolean;
  closedV?: boolean;
  doubleSided?: boolean;
  name?: string;
}

export function patch(opts: PatchOptions): THREE.BufferGeometry {
  const su = Math.max(1, Math.floor(opts.segmentsU));
  const sv = Math.max(1, Math.floor(opts.segmentsV));
  const nu = opts.closedU ? su : su + 1;
  const nv = opts.closedV ? sv : sv + 1;
  const verts = nu * nv;

  const position = new Float32Array(verts * 3);
  const uv = new Float32Array(verts * 2);
  const color = opts.color ? new Float32Array(verts * 3) : null;

  const p = new THREE.Vector3();
  const tint = new THREE.Color();

  for (let j = 0; j < nv; j++) {
    const v = j / sv;
    for (let i = 0; i < nu; i++) {
      const u = i / su;
      opts.point(u, v, p);
      const k = j * nu + i;
      position[k * 3] = p.x;
      position[k * 3 + 1] = p.y;
      position[k * 3 + 2] = p.z;
      uv[k * 2] = u;
      uv[k * 2 + 1] = v;
      if (color) {
        opts.color?.(u, v, tint);
        color[k * 3] = tint.r;
        color[k * 3 + 1] = tint.g;
        color[k * 3 + 2] = tint.b;
      }
    }
  }

  const faces = su * sv;
  const per = opts.doubleSided ? 12 : 6;
  const index = new Uint32Array(faces * per);
  let w = 0;
  for (let j = 0; j < sv; j++) {
    for (let i = 0; i < su; i++) {
      const i1 = opts.closedU ? (i + 1) % nu : i + 1;
      const j1 = opts.closedV ? (j + 1) % nv : j + 1;
      const a = j * nu + i;
      const b = j * nu + i1;
      const c = j1 * nu + i;
      const d = j1 * nu + i1;
      index[w++] = a;
      index[w++] = c;
      index[w++] = b;
      index[w++] = b;
      index[w++] = c;
      index[w++] = d;
      if (opts.doubleSided) {
        index[w++] = a;
        index[w++] = b;
        index[w++] = c;
        index[w++] = b;
        index[w++] = d;
        index[w++] = c;
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  if (color) geometry.setAttribute('color', new THREE.BufferAttribute(color, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeVertexNormals();
  geometry.name = opts.name ?? 'oakPatch';
  geometry.computeBoundingSphere();
  return geometry;
}

export function planePatch(
  width: number,
  depth: number,
  segmentsU: number,
  segmentsV: number,
  height?: (x: number, z: number) => number,
  name?: string,
): THREE.BufferGeometry {
  return patch({
    segmentsU,
    segmentsV,
    ...(name !== undefined ? { name } : {}),
    point(u, v, out) {
      const x = (u - 0.5) * width;
      const z = (v - 0.5) * depth;
      out.set(x, height ? height(x, z) : 0, z);
    },
  });
}
