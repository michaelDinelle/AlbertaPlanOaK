
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { hash2f } from './rng';
import { emissiveMaterial, oakMaterial, oakUniforms, setEmissiveIntensity } from './materials';
import type { EmissiveMaterial, OakMaterial } from './materials';
import { TOKENS } from './tokens';


export const CELL = TOKENS.cellSize;

export interface GridSize {
  w: number;
  h: number;
}

export type Vec3Tuple = [number, number, number];

export function cellToWorld(x: number, y: number, grid: GridSize, out?: Vec3Tuple): Vec3Tuple {
  const t: Vec3Tuple = out ?? [0, 0, 0];
  t[0] = (x - (grid.w - 1) / 2) * CELL;
  t[1] = 0;
  t[2] = (y - (grid.h - 1) / 2) * CELL;
  return t;
}

export function cellToWorldVec(x: number, y: number, grid: GridSize, out: THREE.Vector3): THREE.Vector3 {
  return out.set((x - (grid.w - 1) / 2) * CELL, 0, (y - (grid.h - 1) / 2) * CELL);
}

export function clampCell(cell: GridCell, grid: GridSize): GridCell {
  cell.x = cell.x < 0 ? 0 : cell.x > grid.w - 1 ? grid.w - 1 : cell.x;
  cell.y = cell.y < 0 ? 0 : cell.y > grid.h - 1 ? grid.h - 1 : cell.y;
  return cell;
}

export function inBounds(x: number, y: number, grid: GridSize): boolean {
  return x >= 0 && y >= 0 && x < grid.w && y < grid.h;
}

export function cellIndex(x: number, y: number, grid: GridSize): number {
  return inBounds(x, y, grid) ? y * grid.w + x : -1;
}

export function isWall(grid: Pick<GridGeometry, 'w' | 'h' | 'walls'>, x: number, y: number): boolean {
  if (!inBounds(x, y, grid)) return true;
  return (grid.walls[y * grid.w + x] ?? 0) !== 0;
}

export function gridExtent(grid: GridSize): { width: number; depth: number } {
  return { width: grid.w * CELL, depth: grid.h * CELL };
}


export type CellTintFn = (x: number, y: number) => THREE.ColorRepresentation | null | undefined;
export type CellFilterFn = (x: number, y: number) => boolean;

export type WallShape = 'block' | 'hedge' | 'pillar';

export interface CellFieldStyle {
  floorColor: THREE.ColorRepresentation;
  wallColor: THREE.ColorRepresentation;
  floorTint?: CellTintFn;
  wallTint?: CellTintFn;
  floorFilter?: CellFilterFn;
  floorUnderWalls?: boolean;
  wallShape?: WallShape;
  wallHeight?: number;
  wallInset?: number;
  floorInset?: number;
  floorThickness?: number;
  jitter?: number;
  wind?: number;
  rimColor?: THREE.ColorRepresentation;
  rimStrength?: number;
  seed?: number;
  name?: string;
}

export interface CellFieldHandle {
  group: THREE.Group;
  floorMeshes: THREE.InstancedMesh[];
  wallMeshes: THREE.InstancedMesh[];
  floorMaterial: OakMaterial;
  wallMaterial: OakMaterial;
  floorCount: number;
  wallCount: number;
  floorInstanceAt(x: number, y: number): number;
  wallInstanceAt(x: number, y: number): number;
  setFloorColor(x: number, y: number, color: THREE.ColorRepresentation): void;
  setWallColor(x: number, y: number, color: THREE.ColorRepresentation): void;
  setTime(t: number): void;
  dispose(): void;
}

function wallGeometryFor(shape: WallShape, size: number, height: number): THREE.BufferGeometry {
  const detail = 1;
  if (shape === 'pillar') {
    const radial = Math.max(5, Math.round(12 * detail));
    const g = new THREE.CylinderGeometry(size * 0.42, size * 0.5, height, radial, 1);
    g.translate(0, height / 2, 0);
    return g;
  }
  if (shape === 'hedge') {
    const seg = Math.max(1, Math.round(3 * detail));
    const g = new THREE.BoxGeometry(size, height, size, seg, seg, seg);
    const pos = g.getAttribute('position');
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i);
      const t = (y + height / 2) / height;
      const k = 1 - 0.14 * t * t;
      pos.setX(i, pos.getX(i) * k);
      pos.setZ(i, pos.getZ(i) * k);
    }
    pos.needsUpdate = true;
    g.computeVertexNormals();
    g.translate(0, height / 2, 0);
    return g;
  }
  const g = new THREE.BoxGeometry(size, height, size);
  g.translate(0, height / 2, 0);
  return g;
}

export function buildCellField(
  grid: Pick<GridGeometry, 'w' | 'h' | 'walls'>,
  style: CellFieldStyle,
): CellFieldHandle {
  const group = new THREE.Group();
  group.name = style.name ?? 'oakCellField';

  const seed = style.seed ?? 0;
  const jitter = style.jitter ?? 0.06;
  const wallHeight = style.wallHeight ?? 0.9;
  const wallInset = style.wallInset ?? 0.02;
  const floorInset = style.floorInset ?? 0.03;
  const floorThickness = style.floorThickness ?? 0.12;
  const shape = style.wallShape ?? 'block';
  const floorUnderWalls = style.floorUnderWalls === true;

  const total = grid.w * grid.h;
  const floorIds = new Int32Array(total).fill(-1);
  const wallIds = new Int32Array(total).fill(-1);

  let floorCapacity = 0;
  let wallCapacity = 0;
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      const wall = (grid.walls[y * grid.w + x] ?? 0) !== 0;
      if (wall) wallCapacity++;
      const wantsFloor = (!wall || floorUnderWalls) && (style.floorFilter ? style.floorFilter(x, y) : true);
      if (wantsFloor) floorCapacity++;
    }
  }

  const floorMaterial = oakMaterial({
    color: style.floorColor,
    roughness: 0.95,
    flatShading: true,
    ...(style.rimColor !== undefined ? { rimColor: style.rimColor } : {}),
    rimStrength: style.rimStrength ?? (style.rimColor !== undefined ? 0.18 : 0),
    name: 'oakCellFloor',
  });
  const wallMaterial = oakMaterial({
    color: style.wallColor,
    roughness: 0.9,
    flatShading: true,
    ...(style.rimColor !== undefined ? { rimColor: style.rimColor } : {}),
    rimStrength: style.rimStrength ?? (style.rimColor !== undefined ? 0.25 : 0),
    ...(style.wind !== undefined && style.wind > 0
      ? { wind: { amplitude: style.wind, frequency: 1.05, height: wallHeight } }
      : {}),
    name: 'oakCellWall',
  });

  const floorGeometry = new THREE.BoxGeometry(CELL - floorInset * 2, floorThickness, CELL - floorInset * 2);
  floorGeometry.translate(0, -floorThickness / 2, 0);
  const wallGeometry = wallGeometryFor(shape, CELL - wallInset * 2, wallHeight);

  const floorMesh = new THREE.InstancedMesh(floorGeometry, floorMaterial, Math.max(1, floorCapacity));
  floorMesh.name = 'oakCellFloorMesh';
  floorMesh.castShadow = false;
  floorMesh.receiveShadow = true;
  floorMesh.count = 0;
  floorMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, floorCapacity) * 3), 3);

  const wallMesh = new THREE.InstancedMesh(wallGeometry, wallMaterial, Math.max(1, wallCapacity));
  wallMesh.name = 'oakCellWallMesh';
  wallMesh.castShadow = true;
  wallMesh.receiveShadow = true;
  wallMesh.count = 0;
  wallMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, wallCapacity) * 3), 3);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3(1, 1, 1);
  const color = new THREE.Color();
  const world: Vec3Tuple = [0, 0, 0];

  let floorCount = 0;
  let wallCount = 0;

  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      const idx = y * grid.w + x;
      const wall = (grid.walls[idx] ?? 0) !== 0;
      cellToWorld(x, y, grid, world);
      const n = hash2f(x, y, seed);

      const wantsFloor = (!wall || floorUnderWalls) && (style.floorFilter ? style.floorFilter(x, y) : true);
      if (wantsFloor) {
        position.set(world[0], 0, world[2]);
        scale.set(1, 1 + (n - 0.5) * jitter, 1);
        matrix.compose(position, quaternion, scale);
        floorMesh.setMatrixAt(floorCount, matrix);
        const tint = style.floorTint?.(x, y);
        color.set(tint ?? 0xffffff).multiplyScalar(1 + (n - 0.5) * jitter);
        floorMesh.setColorAt(floorCount, color);
        floorIds[idx] = floorCount;
        floorCount++;
      }

      if (wall) {
        const nw = hash2f(x, y, seed ^ 0x51ed270b);
        position.set(world[0], 0, world[2]);
        scale.set(1, 1 + (nw - 0.5) * jitter * 2, 1);
        matrix.compose(position, quaternion, scale);
        wallMesh.setMatrixAt(wallCount, matrix);
        const tint = style.wallTint?.(x, y);
        color.set(tint ?? 0xffffff).multiplyScalar(1 + (nw - 0.5) * jitter);
        wallMesh.setColorAt(wallCount, color);
        wallIds[idx] = wallCount;
        wallCount++;
      }
    }
  }

  floorMesh.count = floorCount;
  wallMesh.count = wallCount;
  floorMesh.instanceMatrix.needsUpdate = true;
  wallMesh.instanceMatrix.needsUpdate = true;
  if (floorMesh.instanceColor) floorMesh.instanceColor.needsUpdate = true;
  if (wallMesh.instanceColor) wallMesh.instanceColor.needsUpdate = true;
  floorMesh.computeBoundingSphere();
  wallMesh.computeBoundingSphere();

  const floorMeshes: THREE.InstancedMesh[] = [];
  const wallMeshes: THREE.InstancedMesh[] = [];
  if (floorCount > 0) {
    group.add(floorMesh);
    floorMeshes.push(floorMesh);
  }
  if (wallCount > 0) {
    group.add(wallMesh);
    wallMeshes.push(wallMesh);
  }

  const tmp = new THREE.Color();

  return {
    group,
    floorMeshes,
    wallMeshes,
    floorMaterial,
    wallMaterial,
    floorCount,
    wallCount,
    floorInstanceAt(x, y) {
      const i = cellIndex(x, y, grid);
      return i < 0 ? -1 : (floorIds[i] ?? -1);
    },
    wallInstanceAt(x, y) {
      const i = cellIndex(x, y, grid);
      return i < 0 ? -1 : (wallIds[i] ?? -1);
    },
    setFloorColor(x, y, c) {
      const i = cellIndex(x, y, grid);
      const id = i < 0 ? -1 : (floorIds[i] ?? -1);
      if (id < 0) return;
      floorMesh.setColorAt(id, tmp.set(c));
      if (floorMesh.instanceColor) floorMesh.instanceColor.needsUpdate = true;
    },
    setWallColor(x, y, c) {
      const i = cellIndex(x, y, grid);
      const id = i < 0 ? -1 : (wallIds[i] ?? -1);
      if (id < 0) return;
      wallMesh.setColorAt(id, tmp.set(c));
      if (wallMesh.instanceColor) wallMesh.instanceColor.needsUpdate = true;
    },
    setTime(t) {
      oakUniforms(floorMaterial)?.setTime(t);
      oakUniforms(wallMaterial)?.setTime(t);
    },
    dispose() {
      floorMesh.dispose();
      wallMesh.dispose();
      floorGeometry.dispose();
      wallGeometry.dispose();
      floorMaterial.dispose();
      wallMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}


interface CellPlaced {
  readonly group: THREE.Group;
  setCell(x: number, y: number, grid: GridSize): void;
  setVisible(v: boolean): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export interface GoalBeaconOptions {
  color: THREE.ColorRepresentation;
  height?: number;
  radius?: number;
  intensity?: number;
  name?: string;
}

export class GoalBeacon implements CellPlaced {
  readonly group = new THREE.Group();
  private readonly core: THREE.Mesh;
  private readonly disc: THREE.Mesh;
  private readonly material: EmissiveMaterial;
  private readonly discMaterial: EmissiveMaterial;
  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly baseIntensity: number;
  private flareAmount = 0;
  private readonly height: number;

  constructor(opts: GoalBeaconOptions) {
    const detail = 1;
    this.height = opts.height ?? 1.6;
    const radius = opts.radius ?? 0.16;
    this.baseIntensity = opts.intensity ?? 2.2;
    this.group.name = opts.name ?? 'oakGoalBeacon';

    const radial = Math.max(5, Math.round(14 * detail));
    const columnGeo = new THREE.CylinderGeometry(radius * 0.55, radius, this.height, radial, 1, true);
    columnGeo.translate(0, this.height / 2, 0);
    const discGeo = new THREE.CircleGeometry(radius * 3.2, Math.max(8, Math.round(24 * detail)));
    discGeo.rotateX(-Math.PI / 2);
    this.geometries.push(columnGeo, discGeo);

    this.material = emissiveMaterial(opts.color, this.baseIntensity, {
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      name: 'oakGoalBeaconCore',
    });
    this.discMaterial = emissiveMaterial(opts.color, this.baseIntensity * 0.6, {
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      name: 'oakGoalBeaconDisc',
    });

    this.core = new THREE.Mesh(columnGeo, this.material);
    this.core.name = 'oakGoalBeaconCore';
    this.disc = new THREE.Mesh(discGeo, this.discMaterial);
    this.disc.name = 'oakGoalBeaconDisc';
    this.disc.position.y = 0.015;
    this.group.add(this.core, this.disc);
  }

  setCell(x: number, y: number, grid: GridSize): void {
    cellToWorldVec(x, y, grid, this.group.position);
  }

  setVisible(v: boolean): void {
    this.group.visible = v;
  }

  flare(strength = 1): void {
    this.flareAmount = Math.max(this.flareAmount, strength);
  }

  update(dt: number, time: number): void {
    if (this.flareAmount > 0) {
      this.flareAmount = Math.max(0, this.flareAmount - dt / 0.6);
    }
    const breath = 1 + 0.12 * Math.sin(time * 1.7);
    const gain = breath * (1 + 2.5 * this.flareAmount);
    setEmissiveIntensity(this.material, this.baseIntensity * gain);
    setEmissiveIntensity(this.discMaterial, this.baseIntensity * 0.6 * gain);
    this.core.scale.setScalar(1 + 0.05 * (breath - 1) + 0.15 * this.flareAmount);
    this.disc.scale.setScalar(1 + 0.4 * this.flareAmount);
  }

  dispose(): void {
    for (const g of this.geometries) g.dispose();
    this.geometries.length = 0;
    this.material.dispose();
    this.discMaterial.dispose();
    this.group.removeFromParent();
    this.group.clear();
  }
}

export interface DoorGateOptions {
  color: THREE.ColorRepresentation;
  height?: number;
  thickness?: number;
  duration?: number;
  motion?: 'slide' | 'swing';
  name?: string;
}

export class DoorGate implements CellPlaced {
  readonly group = new THREE.Group();
  private readonly leaf: THREE.Mesh;
  private readonly pivot = new THREE.Group();
  private readonly material: OakMaterial;
  private readonly geometry: THREE.BufferGeometry;
  private readonly duration: number;
  private readonly motion: 'slide' | 'swing';
  private readonly height: number;
  private openness = 0;
  private targetOpenness = 0;

  constructor(opts: DoorGateOptions) {
    this.height = opts.height ?? 0.95;
    const thickness = opts.thickness ?? 0.18;
    this.duration = Math.max(0.001, opts.duration ?? 0.55);
    this.motion = opts.motion ?? 'slide';
    this.group.name = opts.name ?? 'oakDoorGate';

    this.geometry = new THREE.BoxGeometry(CELL * 0.96, this.height, thickness);
    this.geometry.translate(0, this.height / 2, 0);
    this.material = oakMaterial({
      color: opts.color,
      roughness: 0.8,
      flatShading: true,
      name: 'oakDoorGate',
    });
    this.leaf = new THREE.Mesh(this.geometry, this.material);
    this.leaf.name = 'oakDoorLeaf';
    this.leaf.castShadow = true;
    this.leaf.receiveShadow = true;
    this.pivot.add(this.leaf);
    this.group.add(this.pivot);
  }

  setCell(x: number, y: number, grid: GridSize): void {
    cellToWorldVec(x, y, grid, this.group.position);
  }

  setVisible(v: boolean): void {
    this.group.visible = v;
  }

  setOpen(open: boolean, immediate = false): void {
    this.targetOpenness = open ? 1 : 0;
    if (immediate) {
      this.openness = this.targetOpenness;
      this.apply();
    }
  }

  get open(): boolean {
    return this.targetOpenness >= 0.5;
  }

  get progress(): number {
    return this.openness;
  }

  snap(): void {
    this.openness = this.targetOpenness;
    this.apply();
  }

  update(dt: number, _time: number): void {
    if (this.openness !== this.targetOpenness) {
      const step = dt / this.duration;
      const d = this.targetOpenness - this.openness;
      this.openness += Math.sign(d) * Math.min(Math.abs(d), step);
      this.apply();
    }
  }

  private apply(): void {
    const t = this.openness;
    const eased = t * t * (3 - 2 * t);
    if (this.motion === 'swing') {
      this.pivot.rotation.y = eased * Math.PI * 0.5;
      this.leaf.position.x = CELL * 0.48 * eased;
    } else {
      this.leaf.position.y = -this.height * eased;
    }
    this.leaf.visible = eased < 0.999;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.group.removeFromParent();
    this.group.clear();
  }
}

export interface KeyTokenOptions {
  color: THREE.ColorRepresentation;
  radius?: number;
  hover?: number;
  intensity?: number;
  name?: string;
}

export class KeyToken implements CellPlaced {
  readonly group = new THREE.Group();
  private readonly mesh: THREE.Mesh;
  private readonly halo: THREE.Mesh;
  private readonly material: EmissiveMaterial;
  private readonly haloMaterial: EmissiveMaterial;
  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly hover: number;
  private readonly baseIntensity: number;

  constructor(opts: KeyTokenOptions) {
    const detail = 1;
    const radius = opts.radius ?? 0.11;
    this.hover = opts.hover ?? 0.42;
    this.baseIntensity = opts.intensity ?? 2;
    this.group.name = opts.name ?? 'oakKeyToken';

    const core = new THREE.OctahedronGeometry(radius, Math.round(detail));
    const halo = new THREE.SphereGeometry(radius * 2.1, Math.max(6, Math.round(12 * detail)), Math.max(4, Math.round(8 * detail)));
    this.geometries.push(core, halo);

    this.material = emissiveMaterial(opts.color, this.baseIntensity, { name: 'oakKeyTokenCore' });
    this.haloMaterial = emissiveMaterial(opts.color, this.baseIntensity * 0.35, {
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      name: 'oakKeyTokenHalo',
    });

    this.mesh = new THREE.Mesh(core, this.material);
    this.mesh.name = 'oakKeyTokenCore';
    this.halo = new THREE.Mesh(halo, this.haloMaterial);
    this.halo.name = 'oakKeyTokenHalo';
    this.group.add(this.mesh, this.halo);
    this.group.position.y = this.hover;
  }

  setCell(x: number, y: number, grid: GridSize): void {
    cellToWorldVec(x, y, grid, this.group.position);
    this.group.position.y = this.hover;
  }

  setVisible(v: boolean): void {
    this.group.visible = v;
  }

  update(_dt: number, time: number): void {
    this.mesh.rotation.y = time * 1.4;
    this.mesh.position.y = Math.sin(time * 2.1) * 0.045;
    this.halo.position.y = this.mesh.position.y;
    const breath = 1 + 0.18 * Math.sin(time * 2.6);
    setEmissiveIntensity(this.material, this.baseIntensity * breath);
  }

  dispose(): void {
    for (const g of this.geometries) g.dispose();
    this.geometries.length = 0;
    this.material.dispose();
    this.haloMaterial.dispose();
    this.group.removeFromParent();
    this.group.clear();
  }
}


export interface HopMotionOptions {
  duration?: number;
  height?: number;
}

export class HopMotion {
  readonly position = new THREE.Vector3();
  private grid: GridSize;
  private readonly duration: number;
  private readonly height: number;
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private _cellX = 0;
  private _cellY = 0;
  private startTime = 0;
  private t = 1;

  constructor(grid: GridSize, opts: HopMotionOptions = {}) {
    this.grid = grid;
    this.duration = Math.max(1e-4, opts.duration ?? 0.16);
    this.height = opts.height ?? 0.22;
    cellToWorldVec(0, 0, grid, this.to);
    this.from.copy(this.to);
    this.position.copy(this.to);
  }

  setGrid(grid: GridSize): void {
    if (grid.w === this.grid.w && grid.h === this.grid.h) {
      this.grid = grid;
      return;
    }
    const dx = ((this.grid.w - grid.w) / 2) * CELL;
    const dz = ((this.grid.h - grid.h) / 2) * CELL;
    this.grid = grid;
    this.from.x += dx;
    this.from.z += dz;
    this.position.x += dx;
    this.position.z += dz;
    cellToWorldVec(this._cellX, this._cellY, grid, this.to);
    if (this.t >= 1) this.position.copy(this.to);
  }

  get cellX(): number {
    return this._cellX;
  }

  get cellY(): number {
    return this._cellY;
  }

  get cell(): Readonly<GridCell> {
    return { x: this._cellX, y: this._cellY };
  }

  get progress(): number {
    return this.t;
  }

  get airborne(): boolean {
    return this.t < 1;
  }

  hopTo(x: number, y: number, now: number): void {
    if (x === this._cellX && y === this._cellY && this.t >= 1) return;
    this.from.copy(this.position);
    cellToWorldVec(x, y, this.grid, this.to);
    this._cellX = x;
    this._cellY = y;
    this.startTime = now;
    this.t = 0;
    this.update(now);
  }

  snap(x?: number, y?: number): void {
    if (x !== undefined && y !== undefined) {
      this._cellX = x;
      this._cellY = y;
      cellToWorldVec(x, y, this.grid, this.to);
    }
    this.from.copy(this.to);
    this.position.copy(this.to);
    this.t = 1;
  }

  update(now: number): void {
    if (this.t >= 1) {
      this.position.copy(this.to);
      return;
    }
    const raw = (now - this.startTime) / this.duration;
    const t = raw <= 0 ? 0 : raw >= 1 ? 1 : raw;
    this.t = t;
    const e = t * t * (3 - 2 * t);
    this.position.lerpVectors(this.from, this.to, e);
    this.position.y += this.height * 4 * t * (1 - t);
    if (t >= 1) this.position.copy(this.to);
  }
}
