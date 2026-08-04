// kit/grid — THE ACCURACY BACKBONE for all 10 grid environments.
//
// STYLE.md: "1 grid cell = 1 world unit, grid centered at origin, +x = grid-x (east),
// +z = grid-y (south), y up", "Ground/floor at y = 0", "kit/grid (cellToWorld,
// buildCellField) + the Wisp avatar for every grid env", "Path-plan between cells
// (straight hop only)" is a MUST NOT — hence HopMotion's straight-line arc.
//
// Everything geometric in a grid scene resolves through THIS file. `cellToWorld` is the
// single conversion; `buildCellField` places walls STRICTLY from `GridGeometry.walls`
// (no inference, no border assumptions, no "obviously it's a ring" shortcuts) so a
// procedurally regenerated maze is as correct as a hand-written one.
//
// Floor tiles sit with their TOP FACE at y = 0 (the style contract's ground plane) and
// wall blocks sit with their BASE at y = 0.

import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { hash2f } from './rng';
import { emissiveMaterial, oakMaterial, oakUniforms, setEmissiveIntensity } from './materials';
import type { EmissiveMaterial, OakMaterial } from './materials';
import { TOKENS } from './tokens';
import type { QualityTier } from './types';

// ---------------------------------------------------------------------------
// coordinates — the one conversion
// ---------------------------------------------------------------------------

/** 1 cell = 1 world unit (tokens.cellSize). */
export const CELL = TOKENS.cellSize;

/** The only part of GridGeometry the conversion needs. */
export interface GridSize {
  w: number;
  h: number;
}

export type Vec3Tuple = [number, number, number];

/**
 * Cell (x, y) -> world [wx, 0, wz]. The grid is centered at the origin, so cell
 * ((w-1)/2, (h-1)/2) is exactly the origin. +x = grid x (east), +z = grid y (south).
 *
 * Pass `out` to stay allocation-free inside update().
 */
export function cellToWorld(x: number, y: number, grid: GridSize, out?: Vec3Tuple): Vec3Tuple {
  const t: Vec3Tuple = out ?? [0, 0, 0];
  t[0] = (x - (grid.w - 1) / 2) * CELL;
  t[1] = 0;
  t[2] = (y - (grid.h - 1) / 2) * CELL;
  return t;
}

/** `cellToWorld` straight into a Vector3 (allocation-free with a preallocated `out`). */
export function cellToWorldVec(x: number, y: number, grid: GridSize, out: THREE.Vector3): THREE.Vector3 {
  return out.set((x - (grid.w - 1) / 2) * CELL, 0, (y - (grid.h - 1) / 2) * CELL);
}

/** Exact inverse of `cellToWorld` for cell centers; rounds elsewhere. Never clamps. */
export function worldToCell(wx: number, wz: number, grid: GridSize, out?: GridCell): GridCell {
  const c: GridCell = out ?? { x: 0, y: 0 };
  c.x = Math.round(wx / CELL + (grid.w - 1) / 2);
  c.y = Math.round(wz / CELL + (grid.h - 1) / 2);
  return c;
}

/** Clamp a cell into the grid (a separate, explicit step — worldToCell never hides drift). */
export function clampCell(cell: GridCell, grid: GridSize): GridCell {
  cell.x = cell.x < 0 ? 0 : cell.x > grid.w - 1 ? grid.w - 1 : cell.x;
  cell.y = cell.y < 0 ? 0 : cell.y > grid.h - 1 ? grid.h - 1 : cell.y;
  return cell;
}

export function inBounds(x: number, y: number, grid: GridSize): boolean {
  return x >= 0 && y >= 0 && x < grid.w && y < grid.h;
}

/** Row-major index of a cell. Returns -1 out of bounds. */
export function cellIndex(x: number, y: number, grid: GridSize): number {
  return inBounds(x, y, grid) ? y * grid.w + x : -1;
}

/** Wall occupancy read STRICTLY from the walls array. Out of bounds counts as wall. */
export function isWall(grid: Pick<GridGeometry, 'w' | 'h' | 'walls'>, x: number, y: number): boolean {
  if (!inBounds(x, y, grid)) return true;
  return (grid.walls[y * grid.w + x] ?? 0) !== 0;
}

/** World-space extent of the playfield: [width(x), depth(z)]. */
export function gridExtent(grid: GridSize): { width: number; depth: number } {
  return { width: grid.w * CELL, depth: grid.h * CELL };
}

/** Radius that covers the whole playfield (shadow focus, fog extent, camera framing). */
export function gridRadius(grid: GridSize): number {
  const { width, depth } = gridExtent(grid);
  return 0.5 * Math.sqrt(width * width + depth * depth);
}

/** Number of 1s in the walls array — the count `buildCellField` must place. */
export function wallCellCount(grid: Pick<GridGeometry, 'w' | 'h' | 'walls'>): number {
  let n = 0;
  const total = grid.w * grid.h;
  for (let i = 0; i < total; i++) if ((grid.walls[i] ?? 0) !== 0) n++;
  return n;
}

/** Iterate every cell in row-major order with its wall flag. */
export function forEachCell(
  grid: Pick<GridGeometry, 'w' | 'h' | 'walls'>,
  visit: (x: number, y: number, wall: boolean, index: number) => void,
): void {
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      const i = y * grid.w + x;
      visit(x, y, (grid.walls[i] ?? 0) !== 0, i);
    }
  }
}

// ---------------------------------------------------------------------------
// cell field — floors + walls, one draw call each
// ---------------------------------------------------------------------------

export type CellTintFn = (x: number, y: number) => THREE.ColorRepresentation | null | undefined;
export type CellFilterFn = (x: number, y: number) => boolean;

export type WallShape = 'block' | 'hedge' | 'pillar';

export interface CellFieldStyle {
  /** base floor color; per-cell variation comes from `floorTint` + instance jitter */
  floorColor: THREE.ColorRepresentation;
  /** base wall color */
  wallColor: THREE.ColorRepresentation;
  /**
   * Per-cell floor tint (rooms, seasons, reef shelves). The returned color MULTIPLIES
   * `floorColor` on the instance, so `0xffffff` means "leave it alone".
   */
  floorTint?: CellTintFn;
  /** per-cell wall tint, multiplied against `wallColor` the same way */
  wallTint?: CellTintFn;
  /** skip floor tiles where this returns false (DeepSea's open blue water) */
  floorFilter?: CellFilterFn;
  /** also place a floor tile under wall cells (default false) */
  floorUnderWalls?: boolean;
  wallShape?: WallShape;
  /** world height of a wall block (default 0.9) */
  wallHeight?: number;
  /** shrink of a wall block in x/z, world units per side (default 0.02) */
  wallInset?: number;
  /** shrink of a floor tile in x/z, world units per side (default 0.03) */
  floorInset?: number;
  /** floor slab thickness; the TOP face stays at y = 0 (default 0.12) */
  floorThickness?: number;
  /** deterministic per-cell height/tone jitter in [0,1] (default 0.06) */
  jitter?: number;
  /** vertex wind amplitude for hedge walls (default 0 for block/pillar) */
  wind?: number;
  /** sky-tinted fresnel rim color */
  rimColor?: THREE.ColorRepresentation;
  rimStrength?: number;
  /** lane seed — the only randomness source */
  seed?: number;
  name?: string;
}

export interface CellFieldHandle {
  /** add this ONE object to the scene */
  group: THREE.Group;
  /** one InstancedMesh per floor archetype (currently 1; array for future skins) */
  floorMeshes: THREE.InstancedMesh[];
  /** one InstancedMesh per wall archetype (currently 1) */
  wallMeshes: THREE.InstancedMesh[];
  floorMaterial: OakMaterial;
  wallMaterial: OakMaterial;
  /** instances actually placed */
  floorCount: number;
  wallCount: number;
  /** instance id of the floor tile at a cell, or -1 */
  floorInstanceAt(x: number, y: number): number;
  /** instance id of the wall block at a cell, or -1 */
  wallInstanceAt(x: number, y: number): number;
  /**
   * Per-instance tint for one floor tile (goal shrine glow, recently-passed stones).
   * This is a MULTIPLIER over `floorMaterial.color`, not an absolute color — pass
   * ~white to reset, a bright hue to push, a dark one to sink.
   */
  setFloorColor(x: number, y: number, color: THREE.ColorRepresentation): void;
  /** Per-instance tint MULTIPLIER for one wall block. */
  setWallColor(x: number, y: number, color: THREE.ColorRepresentation): void;
  setTime(t: number): void;
  dispose(): void;
}

/** hero 1.0 / medium 0.6 / thumb 0.3 — geometry detail, never cell coverage. */
export function cellFieldDetail(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 0.3 : tier === 'medium' ? 0.6 : 1;
}

function wallGeometryFor(shape: WallShape, tier: QualityTier | undefined, size: number, height: number): THREE.BufferGeometry {
  const detail = cellFieldDetail(tier);
  if (shape === 'pillar') {
    const radial = Math.max(5, Math.round(12 * detail));
    const g = new THREE.CylinderGeometry(size * 0.42, size * 0.5, height, radial, 1);
    g.translate(0, height / 2, 0);
    return g;
  }
  if (shape === 'hedge') {
    const seg = Math.max(1, Math.round(3 * detail));
    // a slightly tapered box reads as a clipped hedge and gives wind something to bend
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

/**
 * Build the playfield: instanced floor tiles + wall blocks, placed strictly from
 * `grid.walls`.
 *
 * ```ts
 * const field = buildCellField(ctx.geometry.grid!, { floorColor: p.groundA, wallColor: p.groundB }, ctx.tier);
 * scene.add(field.group);
 * ```
 */
export function buildCellField(
  grid: Pick<GridGeometry, 'w' | 'h' | 'walls'>,
  style: CellFieldStyle,
  tier?: QualityTier,
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
    tier,
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
    tier,
    name: 'oakCellWall',
  });

  const floorGeometry = new THREE.BoxGeometry(CELL - floorInset * 2, floorThickness, CELL - floorInset * 2);
  floorGeometry.translate(0, -floorThickness / 2, 0); // top face exactly at y = 0
  const wallGeometry = wallGeometryFor(shape, tier, CELL - wallInset * 2, wallHeight);

  const floorMesh = new THREE.InstancedMesh(floorGeometry, floorMaterial, Math.max(1, floorCapacity));
  floorMesh.name = 'oakCellFloorMesh';
  floorMesh.castShadow = false;
  floorMesh.receiveShadow = tier !== 'thumb';
  floorMesh.count = 0;
  floorMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, floorCapacity) * 3), 3);

  const wallMesh = new THREE.InstancedMesh(wallGeometry, wallMaterial, Math.max(1, wallCapacity));
  wallMesh.name = 'oakCellWallMesh';
  wallMesh.castShadow = tier !== 'thumb';
  wallMesh.receiveShadow = tier !== 'thumb';
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

// ---------------------------------------------------------------------------
// cell-placed components
// ---------------------------------------------------------------------------

interface CellPlaced {
  readonly group: THREE.Group;
  setCell(x: number, y: number, grid: GridSize): void;
  setVisible(v: boolean): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export interface GoalBeaconOptions {
  color: THREE.ColorRepresentation;
  tier?: QualityTier;
  /** column height in world units (default 1.6) */
  height?: number;
  radius?: number;
  intensity?: number;
  name?: string;
}

/**
 * The goal marker every grid scene reskins: a soft emissive column + a ground disc.
 * Position comes ONLY from `setCell`, so a respawning goal is exactly where sim-core
 * says it is.
 */
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
    const tier = opts.tier;
    const detail = cellFieldDetail(tier);
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

  /** Celebration hook: a bright flash that decays over ~0.6 s. */
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
  tier?: QualityTier;
  /** door leaf height (default 0.95) */
  height?: number;
  /** door thickness (default 0.18) */
  thickness?: number;
  /** seconds for a full open/close (default 0.55) */
  duration?: number;
  /** 'slide' sinks the leaf into the floor, 'swing' rotates it open */
  motion?: 'slide' | 'swing';
  name?: string;
}

/**
 * A door on an exact cell. `setOpen` starts an animation driven from the host clock via
 * `update(dt, ...)`; `setOpen(v, true)` snaps (scrub-safe).
 */
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
      tier: opts.tier,
      name: 'oakDoorGate',
    });
    this.leaf = new THREE.Mesh(this.geometry, this.material);
    this.leaf.name = 'oakDoorLeaf';
    this.leaf.castShadow = opts.tier !== 'thumb';
    this.leaf.receiveShadow = opts.tier !== 'thumb';
    this.pivot.add(this.leaf);
    this.group.add(this.pivot);
  }

  setCell(x: number, y: number, grid: GridSize): void {
    cellToWorldVec(x, y, grid, this.group.position);
  }

  setVisible(v: boolean): void {
    this.group.visible = v;
  }

  /** `immediate` (scrub / teleport) snaps with no easing. */
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

  /** 0 = shut, 1 = fully open. */
  get progress(): number {
    return this.openness;
  }

  /** Force the animation to its target (scene `snap()` / scrubbed frames). */
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
  tier?: QualityTier;
  /** token radius (default 0.11) */
  radius?: number;
  /** hover height above the floor (default 0.42) */
  hover?: number;
  intensity?: number;
  name?: string;
}

/**
 * The carryable token (seed / pearl / ember). It lives at a cell until the wisp picks it
 * up, at which point a scene reparents `group` under `Wisp.carry()`.
 */
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
    const detail = cellFieldDetail(opts.tier);
    const radius = opts.radius ?? 0.11;
    this.hover = opts.hover ?? 0.42;
    this.baseIntensity = opts.intensity ?? 2;
    this.group.name = opts.name ?? 'oakKeyToken';

    const core = new THREE.OctahedronGeometry(radius, opts.tier === 'thumb' ? 0 : Math.round(detail));
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

// ---------------------------------------------------------------------------
// HopMotion — the ONLY sanctioned cell-to-cell motion (STYLE.md: straight hop only)
// ---------------------------------------------------------------------------

export interface HopMotionOptions {
  /** seconds per hop (default 0.16 — one 8 Hz grid step) */
  duration?: number;
  /** arc apex height in world units (default 0.22) */
  height?: number;
}

/**
 * A straight-line cell-to-cell arc. NO path planning, NO extrapolation: the motion is a
 * lerp in xz plus a parabola in y between exactly two cells.
 *
 * ```ts
 * hop.hopTo(cell.x, cell.y, ctx.time());   // on a state change
 * hop.update(ctx.time());                  // every frame
 * hop.snap();                              // on `scrubbed` / `teleport`
 * ```
 */
export class HopMotion {
  /** current world position — read it, do not write it */
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
    // start landed on cell (0, 0) so position and cell agree before the first hop
    cellToWorldVec(0, 0, grid, this.to);
    this.from.copy(this.to);
    this.position.copy(this.to);
  }

  /**
   * Rebind the grid (ProceduralMaze regenerates its layout mid-stream, so the
   * playfield can change size under a hop that is still in the air).
   *
   * cellToWorld is a pure translation in w and h, so the in-flight arc is carried
   * across by that same translation and the DESTINATION is re-derived from the
   * cell — leaving `from`/`to` in the old grid's coordinates made the wisp land
   * off-cell and stay there until the next hop.
   */
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

  /** destination cell x (allocation-free — prefer this inside update()) */
  get cellX(): number {
    return this._cellX;
  }

  /** destination cell y */
  get cellY(): number {
    return this._cellY;
  }

  /** the DESTINATION cell as an object. ALLOCATES — never call it from update(). */
  get cell(): Readonly<GridCell> {
    return { x: this._cellX, y: this._cellY };
  }

  /** 0 at hop start, 1 when landed */
  get progress(): number {
    return this.t;
  }

  get airborne(): boolean {
    return this.t < 1;
  }

  /** Start a straight hop to a cell. A hop already in flight is replaced (no queueing). */
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

  /**
   * `snap(x, y)` teleports to a cell; `snap()` finishes the current hop instantly.
   * Both leave `airborne === false` and `progress === 1` — the scrub/teleport contract.
   */
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
