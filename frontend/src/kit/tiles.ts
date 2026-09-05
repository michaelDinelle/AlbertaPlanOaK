import * as THREE from 'three';
import { rngAt, type Rng } from './rng';
import { SpatialInstances } from './instances';

export type TileWriter = (
  x: number,
  y: number,
  z: number,
  yaw: number,
  scale: number,
  stretch: number,
  mix01: number,
  normal?: THREE.Vector3,
) => void;

export type TileFill = (tileX: number, tileZ: number, rng: Rng, write: TileWriter) => void;

export function bankKey(name: string): number {
  let h = 0x811c9dc5 | 0;
  for (let i = 0; i < name.length; i++) h = Math.imul(h ^ name.charCodeAt(i), 0x01000193) | 0;
  return h;
}

export interface TileBankHandle {
  mesh: THREE.Group;
  instances: SpatialInstances;
  setOrigin(tileX: number, tileZ: number, radius: number): void;
  dispose(): void;
}

export interface TileBankOptions {
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
  depth?: THREE.Material;
  perTile: number;
  colorA: THREE.ColorRepresentation;
  colorB: THREE.ColorRepresentation;
  fill: TileFill;
  castShadow?: boolean;
  receiveShadow?: boolean;
  seed: number;
  name: string;
}

const Z_STRIDE = 0x9e3779b1;

export function makeTileBank(opts: TileBankOptions): TileBankHandle {
  const batches = new SpatialInstances([{ ...opts, padding: 1 }]);
  const mesh = batches.group;
  mesh.name = opts.name;
  const matrices = new Float32Array(opts.perTile * 16), colors = new Float32Array(opts.perTile * 3);
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const groundRotation = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const euler = new THREE.Euler();
  const scaleVec = new THREE.Vector3();
  const colorA = new THREE.Color(opts.colorA);
  const colorB = new THREE.Color(opts.colorB);
  const color = new THREE.Color();

  let region = '';
  const tiles = new Map<string, { x: number; z: number }>();
  let pending: [number, number][] = [];
  let cursor = 0;
  let limit = 0;

  const write: TileWriter = (x, y, z, yaw, scale, stretch, mix01, normal) => {
    if (cursor >= limit) return;
    position.set(x, y, z);
    euler.set(0, yaw, 0);
    quaternion.setFromEuler(euler);
    if (normal) quaternion.premultiply(groundRotation.setFromUnitVectors(up, normal));
    scaleVec.set(scale, scale * stretch, scale);
    matrix.compose(position, quaternion, scaleVec);
    matrix.toArray(matrices, cursor * 16);
    color.copy(colorA).lerp(colorB, mix01 < 0 ? 0 : mix01 > 1 ? 1 : mix01);
    color.toArray(colors, cursor * 3);
    cursor++;
  };

  const bank = bankKey(opts.name);

  const fillTile = (ix: number, iz: number): void => {
    cursor = 0;
    limit = opts.perTile;
    opts.fill(ix, iz, rngAt(ix, (bank ^ Math.imul(iz, Z_STRIDE)) | 0, opts.seed), write);
    const key = `${ix}:${iz}`;
    tiles.set(key, { x: ix, z: iz });
    batches.set(key, matrices.subarray(0, cursor * 16), colors.subarray(0, cursor * 3));
  };

  return {
    mesh, instances: batches,
    setOrigin(tileX, tileZ, radius) {
      const next = `${tileX}:${tileZ}:${radius}`;
      if (next !== region) {
        region = next; pending = [];
        for (const [key, { x, z }] of tiles) if (Math.abs(x - tileX) > radius + 2 || Math.abs(z - tileZ) > radius + 2) { batches.delete(key); tiles.delete(key); }
        for (let z = tileZ - radius; z <= tileZ + radius; z++) for (let x = tileX - radius; x <= tileX + radius; x++) if (!tiles.has(`${x}:${z}`)) pending.push([x,z]);
        const distance = ([x,z]: [number,number]) => (x - tileX) ** 2 + (z - tileZ) ** 2;
        pending.sort((a,b) => distance(b) - distance(a));
      }
      const start = performance.now();
      while (pending.length && performance.now() - start < 1.5) fillTile(...pending.pop()!);
    },
    dispose() {
      batches.dispose();
    },
  };
}
