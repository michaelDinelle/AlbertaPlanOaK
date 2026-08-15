import * as THREE from 'three';

export interface Part {
  geometry: THREE.BufferGeometry;
  color?: THREE.ColorRepresentation;
  position?: readonly [number, number, number];
  rotation?: readonly [number, number, number];
  scale?: readonly [number, number, number] | number;
}

const M = new THREE.Matrix4();
const E = new THREE.Euler();
const P = new THREE.Vector3();
const Q = new THREE.Quaternion();
const S = new THREE.Vector3();
const C = new THREE.Color();

function placed(part: Part): THREE.BufferGeometry {
  const g = part.geometry.clone();
  const pos = part.position ?? [0, 0, 0];
  const rot = part.rotation ?? [0, 0, 0];
  const sc = part.scale ?? 1;
  P.set(pos[0], pos[1], pos[2]);
  E.set(rot[0], rot[1], rot[2]);
  Q.setFromEuler(E);
  if (typeof sc === 'number') S.setScalar(sc);
  else S.set(sc[0], sc[1], sc[2]);
  M.compose(P, Q, S);
  g.applyMatrix4(M);
  return g;
}

export function assemble(parts: readonly Part[], name = 'oakAssembly'): THREE.BufferGeometry {
  const prepared: THREE.BufferGeometry[] = [];
  let verts = 0;
  let indices = 0;
  for (const part of parts) {
    const g = placed(part);
    if (g.getIndex() === null) {
      const n = g.getAttribute('position').count;
      const idx = new Uint32Array(n);
      for (let i = 0; i < n; i++) idx[i] = i;
      g.setIndex(new THREE.BufferAttribute(idx, 1));
    }
    if (g.getAttribute('normal') === undefined) g.computeVertexNormals();
    prepared.push(g);
    verts += g.getAttribute('position').count;
    indices += g.getIndex()?.count ?? 0;
  }

  const position = new Float32Array(verts * 3);
  const normal = new Float32Array(verts * 3);
  const color = new Float32Array(verts * 3);
  const index = new Uint32Array(indices);

  let vo = 0;
  let io = 0;
  for (let k = 0; k < prepared.length; k++) {
    const g = prepared[k];
    const part = parts[k];
    if (!g || !part) continue;
    const gp = g.getAttribute('position') as THREE.BufferAttribute;
    const gn = g.getAttribute('normal') as THREE.BufferAttribute;
    const gc = g.getAttribute('color') as THREE.BufferAttribute | undefined;
    const gi = g.getIndex();
    const n = gp.count;
    position.set(gp.array as Float32Array, vo * 3);
    normal.set(gn.array as Float32Array, vo * 3);
    if (gc) {
      color.set(gc.array as Float32Array, vo * 3);
    } else {
      C.set(part.color ?? 0xffffff);
      for (let i = 0; i < n; i++) {
        color[(vo + i) * 3] = C.r;
        color[(vo + i) * 3 + 1] = C.g;
        color[(vo + i) * 3 + 2] = C.b;
      }
    }
    if (gi) {
      for (let i = 0; i < gi.count; i++) index[io + i] = gi.getX(i) + vo;
      io += gi.count;
    }
    vo += n;
    g.dispose();
  }

  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(position, 3));
  out.setAttribute('normal', new THREE.BufferAttribute(normal, 3));
  out.setAttribute('color', new THREE.BufferAttribute(color, 3));
  out.setIndex(new THREE.BufferAttribute(index, 1));
  out.name = name;
  out.computeBoundingSphere();
  return out;
}
