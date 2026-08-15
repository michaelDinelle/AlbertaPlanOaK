import * as THREE from 'three';
import { CELL, cellToWorld, type CellFieldHandle, type GridSize, type Vec3Tuple } from './grid';

export const TERRACE_DESCENT = 3.5;

export function rowDrop(grid: GridSize): number {
  return TERRACE_DESCENT / Math.max(1, grid.h - 1);
}

export function terraceDepth(row: number, grid: GridSize): number {
  return -rowDrop(grid) * row;
}

export function terraceDepthAt(z: number, grid: GridSize): number {
  return terraceDepth(z / CELL + (grid.h - 1) / 2, grid);
}

const SCRATCH: Vec3Tuple = [0, 0, 0];

export function cellToTerraceWorld(
  x: number,
  y: number,
  grid: GridSize,
  out?: Vec3Tuple,
): Vec3Tuple {
  const t = cellToWorld(x, y, grid, out);
  t[1] = terraceDepth(y, grid);
  return t;
}

export function cellToTerraceVec(
  x: number,
  y: number,
  grid: GridSize,
  out: THREE.Vector3,
): THREE.Vector3 {
  const w = cellToTerraceWorld(x, y, grid, SCRATCH);
  return out.set(w[0], w[1], w[2]);
}

export function terraceShear(grid: GridSize, out?: THREE.Matrix4): THREE.Matrix4 {
  const m = out ?? new THREE.Matrix4();
  const drop = rowDrop(grid);
  const k = drop / CELL;
  const lift = -drop * ((grid.h - 1) / 2);
  return m.set(1, 0, 0, 0, 0, 1, -k, lift, 0, 0, 1, 0, 0, 0, 0, 1);
}

export function terraceShelves(field: CellFieldHandle, grid: GridSize): void {
  const mesh = field.floorMeshes[0];
  if (!mesh) return;
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      const id = field.floorInstanceAt(x, y);
      if (id < 0) continue;
      mesh.getMatrixAt(id, matrix);
      matrix.decompose(position, quaternion, scale);
      position.y = terraceDepth(y, grid);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(id, matrix);
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.computeBoundingSphere();
}
