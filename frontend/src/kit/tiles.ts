import * as THREE from 'three';
import { rngAt, type Rng } from './rng';

export type TileWriter = (
  x: number,
  y: number,
  z: number,
  yaw: number,
  scale: number,
  stretch: number,
  mix01: number,
) => void;

export type TileFill = (tileX: number, tileZ: number, rng: Rng, write: TileWriter) => void;

export function bankKey(name: string): number {
  let h = 0x811c9dc5 | 0;
  for (let i = 0; i < name.length; i++) h = Math.imul(h ^ name.charCodeAt(i), 0x01000193) | 0;
  return h;
}

export interface TileBankHandle {
  mesh: THREE.InstancedMesh;
  setOrigin(tileX: number, tileZ: number): void;
  dispose(): void;
}

export interface TileBankOptions {
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
  tilesX: number;
  tilesZ: number;
  backX: number;
  backZ: number;
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
  const slots = Math.max(1, opts.tilesX * opts.tilesZ);
  const capacity = Math.max(1, slots * opts.perTile);
  const mesh = new THREE.InstancedMesh(opts.geometry, opts.material, capacity);
  mesh.name = opts.name;
  mesh.castShadow = opts.castShadow ?? false;
  mesh.receiveShadow = opts.receiveShadow ?? false;
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scaleVec = new THREE.Vector3();
  const colorA = new THREE.Color(opts.colorA);
  const colorB = new THREE.Color(opts.colorB);
  const color = new THREE.Color();
  const zero = new THREE.Matrix4().makeScale(0, 0, 0);

  let originX = Number.NaN;
  let originZ = Number.NaN;
  let cursor = 0;
  let limit = 0;

  const write: TileWriter = (x, y, z, yaw, scale, stretch, mix01) => {
    if (cursor >= limit) return;
    position.set(x, y, z);
    euler.set(0, yaw, 0);
    quaternion.setFromEuler(euler);
    scaleVec.set(scale, scale * stretch, scale);
    matrix.compose(position, quaternion, scaleVec);
    mesh.setMatrixAt(cursor, matrix);
    color.copy(colorA).lerp(colorB, mix01 < 0 ? 0 : mix01 > 1 ? 1 : mix01);
    mesh.setColorAt(cursor, color);
    cursor++;
  };

  const slotTileX = new Int32Array(slots).fill(0x7fffffff);
  const slotTileZ = new Int32Array(slots).fill(0x7fffffff);
  const bank = bankKey(opts.name);

  const fillSlot = (slot: number, ix: number, iz: number): void => {
    cursor = slot * opts.perTile;
    limit = cursor + opts.perTile;
    opts.fill(ix, iz, rngAt(ix, (bank ^ Math.imul(iz, Z_STRIDE)) | 0, opts.seed), write);
    for (let i = cursor; i < limit; i++) mesh.setMatrixAt(i, zero);
    slotTileX[slot] = ix;
    slotTileZ[slot] = iz;
  };

  const rebuild = (tileX: number, tileZ: number): void => {
    const firstX = tileX - opts.backX;
    const firstZ = tileZ - opts.backZ;
    let touched = false;
    for (let tz = 0; tz < opts.tilesZ; tz++) {
      const iz = firstZ + tz;
      const wz = ((iz % opts.tilesZ) + opts.tilesZ) % opts.tilesZ;
      for (let tx = 0; tx < opts.tilesX; tx++) {
        const ix = firstX + tx;
        const wx = ((ix % opts.tilesX) + opts.tilesX) % opts.tilesX;
        const slot = wz * opts.tilesX + wx;
        if (slotTileX[slot] === ix && slotTileZ[slot] === iz) continue;
        fillSlot(slot, ix, iz);
        touched = true;
      }
    }
    if (!touched) return;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  };

  return {
    mesh,
    setOrigin(tileX, tileZ) {
      if (tileX === originX && tileZ === originZ) return;
      originX = tileX;
      originZ = tileZ;
      rebuild(tileX, tileZ);
    },
    dispose() {
      mesh.dispose();
      mesh.removeFromParent();
    },
  };
}
