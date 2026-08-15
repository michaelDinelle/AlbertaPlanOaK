import * as THREE from 'three';
import { streamFor } from '../rng';

export interface FieldArea {
  kind: 'rect';
  halfX: number;
  halfZ: number;
  centre?: readonly [number, number, number];
}

export interface FieldCurve {
  kind: 'curve';
  at(t: number, out: THREE.Vector3): THREE.Vector3;
}

export interface FieldVolume {
  kind: 'volume';
  halfX: number;
  halfY: number;
  halfZ: number;
  centre?: readonly [number, number, number];
}

export type FieldSource = FieldArea | FieldCurve | FieldVolume;

export type FieldReject = (x: number, y: number, z: number) => boolean;

export interface FieldOptions {
  source: FieldSource;
  count: number;
  seed: number;
  stream: string;
  variants: readonly THREE.BufferGeometry[];
  material: THREE.Material;
  scale?: readonly [number, number];
  lift?: number;
  yaw?: boolean;
  tilt?: number;
  jitter?: number;
  height?(x: number, z: number): number;
  reject?: FieldReject;
  color?(i: number, out: THREE.Color): void;
  name?: string;
}

export interface FieldHandle {
  group: THREE.Group;
  meshes: THREE.InstancedMesh[];
  count: number;
  setPhase(i: number, matrix: THREE.Matrix4): void;
  dispose(): void;
}

const M = new THREE.Matrix4();
const P = new THREE.Vector3();
const Q = new THREE.Quaternion();
const E = new THREE.Euler();
const S = new THREE.Vector3();
const C = new THREE.Color();

export function field(opts: FieldOptions): FieldHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'oakField';
  const rand = streamFor(opts.seed, opts.stream);
  const variants = opts.variants.length > 0 ? opts.variants : [new THREE.BufferGeometry()];
  for (const g of variants) {
    if (g.getAttribute('color') !== undefined) continue;
    const n = g.getAttribute('position')?.count ?? 0;
    if (n > 0) g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3).fill(1), 3));
  }
  const wanted = Math.max(0, Math.floor(opts.count));
  const scaleRange = opts.scale ?? [1, 1];
  const lift = opts.lift ?? 0;
  const tilt = opts.tilt ?? 0;
  const jitter = opts.jitter ?? 0;

  const buckets: { x: number; y: number; z: number; s: number; yaw: number; tx: number; tz: number }[][] =
    variants.map(() => []);

  let tries = 0;
  let placed = 0;
  while (placed < wanted && tries < wanted * 12 + 64) {
    tries++;
    let x = 0;
    let y = 0;
    let z = 0;
    const src = opts.source;
    if (src.kind === 'rect') {
      const c = src.centre ?? [0, 0, 0];
      x = c[0] + (rand() * 2 - 1) * src.halfX;
      z = c[2] + (rand() * 2 - 1) * src.halfZ;
      y = c[1];
    } else if (src.kind === 'volume') {
      const c = src.centre ?? [0, 0, 0];
      x = c[0] + (rand() * 2 - 1) * src.halfX;
      y = c[1] + (rand() * 2 - 1) * src.halfY;
      z = c[2] + (rand() * 2 - 1) * src.halfZ;
    } else {
      src.at(placed / Math.max(1, wanted - 1), P);
      x = P.x + (rand() * 2 - 1) * jitter;
      y = P.y;
      z = P.z + (rand() * 2 - 1) * jitter;
    }
    if (opts.height) y += opts.height(x, z);
    y += lift;
    if (opts.reject?.(x, y, z)) continue;
    const v = Math.min(variants.length - 1, Math.floor(rand() * variants.length));
    buckets[v]?.push({
      x,
      y,
      z,
      s: scaleRange[0]! + rand() * (scaleRange[1]! - scaleRange[0]!),
      yaw: opts.yaw === false ? 0 : rand() * Math.PI * 2,
      tx: (rand() * 2 - 1) * tilt,
      tz: (rand() * 2 - 1) * tilt,
    });
    placed++;
  }

  const meshes: THREE.InstancedMesh[] = [];
  let index = 0;
  for (let v = 0; v < variants.length; v++) {
    const items = buckets[v];
    const geometry = variants[v];
    if (!items || !geometry || items.length === 0) continue;
    const mesh = new THREE.InstancedMesh(geometry, opts.material, items.length);
    mesh.name = `${group.name}:${v}`;
    if (opts.color) {
      mesh.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(items.length * 3).fill(1),
        3,
      );
    }
    for (let i = 0; i < items.length; i++) {
      const it = items[i]!;
      P.set(it.x, it.y, it.z);
      E.set(it.tx, it.yaw, it.tz);
      Q.setFromEuler(E);
      S.setScalar(it.s);
      M.compose(P, Q, S);
      mesh.setMatrixAt(i, M);
      if (opts.color) {
        opts.color(index, C);
        mesh.setColorAt(i, C);
      }
      index++;
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.computeBoundingSphere();
    group.add(mesh);
    meshes.push(mesh);
  }

  return {
    group,
    meshes,
    count: placed,
    setPhase(i, matrix) {
      let seen = 0;
      for (const mesh of meshes) {
        if (i < seen + mesh.count) {
          mesh.setMatrixAt(i - seen, matrix);
          mesh.instanceMatrix.needsUpdate = true;
          return;
        }
        seen += mesh.count;
      }
    },
    dispose() {
      for (const mesh of meshes) {
        mesh.count = 0;
        mesh.dispose();
      }
      meshes.length = 0;
      group.clear();
      group.removeFromParent();
    },
  };
}
