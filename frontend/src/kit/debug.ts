
import * as THREE from 'three';
import { CHEETAH_BONES, FK_FLOATS, FK_POINT_INDEX } from '@sim/fk/cheetah';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';
import { cellToWorld, type Vec3Tuple } from './grid';
import { TOKENS } from './tokens';
import type { FrameState } from './types';

export const DEBUG_LAYER = 31;

export function enableDebugLayer(camera: THREE.Camera): void {
  camera.layers.enable(DEBUG_LAYER);
}

export function disableDebugLayer(camera: THREE.Camera): void {
  camera.layers.disable(DEBUG_LAYER);
}

export function applyDebugLayer(object: THREE.Object3D): void {
  object.traverse((o) => o.layers.set(DEBUG_LAYER));
}

export interface GroundTruthOptions {
  color?: THREE.ColorRepresentation;
  goalColor?: THREE.ColorRepresentation;
  wallColor?: THREE.ColorRepresentation;
  camera?: THREE.Camera | null;
  agentCell?(state: FrameState): GridCell | null;
  goalCell?(state: FrameState): GridCell | null;
  keyCells?(state: FrameState): readonly GridCell[] | null;
  doorStates?(state: FrameState): readonly boolean[] | null;
  agentPos?(state: FrameState): readonly [number, number, number] | null;
  profileRange?: readonly [number, number];
  fkToWorld?(x: number, z: number, out: Vec3Tuple): Vec3Tuple;
  maxKeys?: number;
  name?: string;
}

const MAX_DOORS = 8;
const PROFILE_SAMPLES = 160;

function lineMaterial(color: THREE.ColorRepresentation, opacity = 0.9): THREE.LineBasicMaterial {
  const m = new THREE.LineBasicMaterial({
    color: new THREE.Color(color),
    transparent: true,
    opacity,
    depthTest: false,
    depthWrite: false,
    fog: false,
    toneMapped: false,
  });
  m.name = 'oakDebugLine';
  return m;
}

function segments(count: number, material: THREE.LineBasicMaterial, name: string): {
  object: THREE.LineSegments;
  positions: Float32Array;
  attr: THREE.BufferAttribute;
} {
  const positions = new Float32Array(count * 2 * 3);
  const attr = new THREE.BufferAttribute(positions, 3);
  attr.setUsage(THREE.DynamicDrawUsage);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', attr);
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
  const object = new THREE.LineSegments(geometry, material);
  object.name = name;
  object.frustumCulled = false;
  object.renderOrder = 999;
  return { object, positions, attr };
}

export class GroundTruthOverlay {
  readonly group = new THREE.Group();

  private readonly materials: THREE.LineBasicMaterial[] = [];
  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly camera: THREE.Camera | null;
  private readonly geometry: EnvGeometry;
  private readonly opts: GroundTruthOptions;

  private readonly grid: GridGeometry | null;
  private agentCross: THREE.LineSegments | null = null;
  private agentCrossPos: Float32Array | null = null;
  private agentCrossAttr: THREE.BufferAttribute | null = null;
  private goalMarker: THREE.LineSegments | null = null;
  private goalMarkerPos: Float32Array | null = null;
  private goalMarkerAttr: THREE.BufferAttribute | null = null;
  private keyMarker: THREE.LineSegments | null = null;
  private keyMarkerPos: Float32Array | null = null;
  private keyMarkerAttr: THREE.BufferAttribute | null = null;
  private doorMarker: THREE.LineSegments | null = null;
  private doorMarkerPos: Float32Array | null = null;
  private doorMarkerAttr: THREE.BufferAttribute | null = null;
  private readonly maxKeys: number;

  private skeletonPos: Float32Array | null = null;
  private skeletonAttr: THREE.BufferAttribute | null = null;
  private toesPos: Float32Array | null = null;
  private toesAttr: THREE.BufferAttribute | null = null;
  private readonly fkOut = new Float32Array(FK_FLOATS);

  private classicMarker: THREE.LineSegments | null = null;
  private classicMarkerPos: Float32Array | null = null;
  private classicMarkerAttr: THREE.BufferAttribute | null = null;

  private readonly scratch: Vec3Tuple = [0, 0, 0];
  private readonly scratchB: Vec3Tuple = [0, 0, 0];
  private _visible = false;

  constructor(geometry: EnvGeometry, opts: GroundTruthOptions = {}) {
    this.geometry = geometry;
    this.opts = opts;
    this.camera = opts.camera ?? null;
    this.maxKeys = Math.max(1, opts.maxKeys ?? 4);
    this.group.name = opts.name ?? 'oakGroundTruth';
    this.grid = geometry.grid ?? null;

    const main = lineMaterial(opts.color ?? TOKENS.uiCyan, 0.85);
    const wall = lineMaterial(opts.wallColor ?? 0x8fa6c4, 0.55);
    const goal = lineMaterial(opts.goalColor ?? TOKENS.celebrationGold, 0.95);
    this.materials.push(main, wall, goal);

    if (this.grid) this.buildGrid(this.grid, main, wall, goal);
    if (geometry.kind === 'cheetah' && geometry.cheetah) this.buildCheetah(main, goal);
    if (geometry.kind === 'classic' && geometry.classic) this.buildClassic(main, goal);

    applyDebugLayer(this.group);
    this.group.visible = false;
  }

  get visible(): boolean {
    return this._visible;
  }

  set visible(v: boolean) {
    this._visible = v;
    this.group.visible = v;
    if (this.camera) {
      if (v) enableDebugLayer(this.camera);
      else disableDebugLayer(this.camera);
    }
  }

  setState(state: FrameState): void {
    if (this.grid) this.updateGrid(state, this.grid);
    if (this.geometry.kind === 'cheetah') this.updateCheetah(state);
    if (this.geometry.kind === 'classic') this.updateClassic(state);
  }

  dispose(): void {
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
    this.geometries.length = 0;
    this.materials.length = 0;
    this.group.removeFromParent();
    this.group.clear();
  }


  private buildGrid(
    grid: GridGeometry,
    main: THREE.LineBasicMaterial,
    wallMat: THREE.LineBasicMaterial,
    goalMat: THREE.LineBasicMaterial,
  ): void {
    const y = 0.02;
    const half = 0.5;

    const latticeCount = grid.w + 1 + grid.h + 1;
    const lattice = segments(latticeCount, wallMat, 'oakDebugLattice');
    {
      const p = lattice.positions;
      let o = 0;
      const x0 = -grid.w / 2;
      const z0 = -grid.h / 2;
      for (let i = 0; i <= grid.w; i++) {
        const x = x0 + i;
        p[o++] = x; p[o++] = y; p[o++] = z0;
        p[o++] = x; p[o++] = y; p[o++] = z0 + grid.h;
      }
      for (let j = 0; j <= grid.h; j++) {
        const z = z0 + j;
        p[o++] = x0; p[o++] = y; p[o++] = z;
        p[o++] = x0 + grid.w; p[o++] = y; p[o++] = z;
      }
      lattice.attr.needsUpdate = true;
    }
    this.add(lattice.object);

    let walls = 0;
    for (let i = 0; i < grid.w * grid.h; i++) if ((grid.walls[i] ?? 0) !== 0) walls++;
    if (walls > 0) {
      const boxes = segments(walls * 12, wallMat, 'oakDebugWalls');
      const p = boxes.positions;
      let o = 0;
      const h = 0.9;
      for (let gy = 0; gy < grid.h; gy++) {
        for (let gx = 0; gx < grid.w; gx++) {
          if ((grid.walls[gy * grid.w + gx] ?? 0) === 0) continue;
          cellToWorld(gx, gy, grid, this.scratch);
          const cx = this.scratch[0];
          const cz = this.scratch[2];
          const x0 = cx - half;
          const x1 = cx + half;
          const z0 = cz - half;
          const z1 = cz + half;
          const edge = (ax: number, ay: number, az: number, bx: number, by: number, bz: number): void => {
            p[o++] = ax; p[o++] = ay; p[o++] = az;
            p[o++] = bx; p[o++] = by; p[o++] = bz;
          };
          edge(x0, 0, z0, x1, 0, z0);
          edge(x1, 0, z0, x1, 0, z1);
          edge(x1, 0, z1, x0, 0, z1);
          edge(x0, 0, z1, x0, 0, z0);
          edge(x0, h, z0, x1, h, z0);
          edge(x1, h, z0, x1, h, z1);
          edge(x1, h, z1, x0, h, z1);
          edge(x0, h, z1, x0, h, z0);
          edge(x0, 0, z0, x0, h, z0);
          edge(x1, 0, z0, x1, h, z0);
          edge(x1, 0, z1, x1, h, z1);
          edge(x0, 0, z1, x0, h, z1);
        }
      }
      boxes.attr.needsUpdate = true;
      this.add(boxes.object);
    }

    const mask = grid.actionMask;
    if (mask && mask.length >= grid.w * grid.h) {
      const arrows = segments(grid.w * grid.h * 3, main, 'oakDebugMaskArrows');
      const p = arrows.positions;
      let o = 0;
      for (let gy = 0; gy < grid.h; gy++) {
        for (let gx = 0; gx < grid.w; gx++) {
          const right = (mask[gy * grid.w + gx] ?? 0) !== 0;
          cellToWorld(gx, gy, grid, this.scratch);
          const cx = this.scratch[0];
          const cz = this.scratch[2];
          const d = right ? 0.28 : -0.28;
          p[o++] = cx - d; p[o++] = y; p[o++] = cz;
          p[o++] = cx + d; p[o++] = y; p[o++] = cz;
          p[o++] = cx + d; p[o++] = y; p[o++] = cz;
          p[o++] = cx + d * 0.55; p[o++] = y; p[o++] = cz - 0.12;
          p[o++] = cx + d; p[o++] = y; p[o++] = cz;
          p[o++] = cx + d * 0.55; p[o++] = y; p[o++] = cz + 0.12;
        }
      }
      arrows.attr.needsUpdate = true;
      this.add(arrows.object);
    }

    const carved = grid.carvedPath;
    if (carved && carved.length > 1) {
      const path = segments(carved.length - 1, goalMat, 'oakDebugCarvedPath');
      const p = path.positions;
      let o = 0;
      for (let i = 0; i < carved.length - 1; i++) {
        const a = carved[i] as GridCell;
        const b = carved[i + 1] as GridCell;
        cellToWorld(a.x, a.y, grid, this.scratch);
        cellToWorld(b.x, b.y, grid, this.scratchB);
        p[o++] = this.scratch[0]; p[o++] = y + 0.03; p[o++] = this.scratch[2];
        p[o++] = this.scratchB[0]; p[o++] = y + 0.03; p[o++] = this.scratchB[2];
      }
      path.attr.needsUpdate = true;
      this.add(path.object);
    }

    const cross = segments(3, main, 'oakDebugAgentCross');
    this.agentCross = cross.object;
    this.agentCrossPos = cross.positions;
    this.agentCrossAttr = cross.attr;
    this.add(cross.object);

    const goalDiamond = segments(4, goalMat, 'oakDebugGoal');
    this.goalMarker = goalDiamond.object;
    this.goalMarkerPos = goalDiamond.positions;
    this.goalMarkerAttr = goalDiamond.attr;
    this.add(goalDiamond.object);

    const keys = segments(this.maxKeys * 4, goalMat, 'oakDebugKeys');
    this.keyMarker = keys.object;
    this.keyMarkerPos = keys.positions;
    this.keyMarkerAttr = keys.attr;
    this.add(keys.object);

    const doors = segments(MAX_DOORS * 2, main, 'oakDebugDoors');
    this.doorMarker = doors.object;
    this.doorMarkerPos = doors.positions;
    this.doorMarkerAttr = doors.attr;
    this.add(doors.object);
  }

  private updateGrid(state: FrameState, grid: GridGeometry): void {
    const y = 0.03;

    const cell = this.opts.agentCell?.(state) ?? null;
    const cp = this.agentCrossPos;
    if (cp && this.agentCross && this.agentCrossAttr) {
      if (cell) {
        cellToWorld(cell.x, cell.y, grid, this.scratch);
        const cx = this.scratch[0];
        const cz = this.scratch[2];
        const r = 0.38;
        let o = 0;
        cp[o++] = cx - r; cp[o++] = y; cp[o++] = cz - r;
        cp[o++] = cx + r; cp[o++] = y; cp[o++] = cz + r;
        cp[o++] = cx - r; cp[o++] = y; cp[o++] = cz + r;
        cp[o++] = cx + r; cp[o++] = y; cp[o++] = cz - r;
        cp[o++] = cx; cp[o++] = y; cp[o++] = cz;
        cp[o++] = cx; cp[o++] = y + 0.9; cp[o++] = cz;
        this.agentCross.visible = true;
      } else {
        this.agentCross.visible = false;
      }
      this.agentCrossAttr.needsUpdate = true;
    }

    const goal = this.opts.goalCell?.(state) ?? grid?.goal ?? null;
    const gp = this.goalMarkerPos;
    if (gp && this.goalMarker && this.goalMarkerAttr) {
      if (goal) {
        cellToWorld(goal.x, goal.y, grid, this.scratch);
        const cx = this.scratch[0];
        const cz = this.scratch[2];
        const r = 0.42;
        let o = 0;
        const corner = (ax: number, az: number, bx: number, bz: number): void => {
          gp[o++] = cx + ax; gp[o++] = y; gp[o++] = cz + az;
          gp[o++] = cx + bx; gp[o++] = y; gp[o++] = cz + bz;
        };
        corner(0, -r, r, 0);
        corner(r, 0, 0, r);
        corner(0, r, -r, 0);
        corner(-r, 0, 0, -r);
        this.goalMarker.visible = true;
      } else {
        this.goalMarker.visible = false;
      }
      this.goalMarkerAttr.needsUpdate = true;
    }

    const keys = this.opts.keyCells?.(state) ?? grid?.keys ?? null;
    const kp = this.keyMarkerPos;
    if (kp && this.keyMarker && this.keyMarkerAttr) {
      kp.fill(0);
      let o = 0;
      const n = keys ? Math.min(keys.length, this.maxKeys) : 0;
      for (let i = 0; i < n; i++) {
        const k = keys?.[i];
        if (!k) continue;
        cellToWorld(k.x, k.y, grid, this.scratch);
        const cx = this.scratch[0];
        const cz = this.scratch[2];
        const r = 0.18;
        const edge = (ax: number, az: number, bx: number, bz: number): void => {
          kp[o++] = cx + ax; kp[o++] = y; kp[o++] = cz + az;
          kp[o++] = cx + bx; kp[o++] = y; kp[o++] = cz + bz;
        };
        edge(-r, -r, r, -r);
        edge(r, -r, r, r);
        edge(r, r, -r, r);
        edge(-r, r, -r, -r);
      }
      this.keyMarker.visible = n > 0;
      this.keyMarkerAttr.needsUpdate = true;
    }

    const doors = grid.doors;
    const dp = this.doorMarkerPos;
    if (dp && this.doorMarker && this.doorMarkerAttr) {
      dp.fill(0);
      const states = this.opts.doorStates?.(state) ?? null;
      const n = doors ? Math.min(doors.length, MAX_DOORS) : 0;
      let o = 0;
      for (let i = 0; i < n; i++) {
        const d = doors?.[i];
        if (!d) continue;
        cellToWorld(d.x, d.y, grid, this.scratch);
        const cx = this.scratch[0];
        const cz = this.scratch[2];
        const open = states?.[i] === true;
        const gap = open ? 0.3 : 0;
        dp[o++] = cx - 0.45; dp[o++] = y; dp[o++] = cz;
        dp[o++] = cx - gap; dp[o++] = y; dp[o++] = cz;
        dp[o++] = cx + gap; dp[o++] = y; dp[o++] = cz;
        dp[o++] = cx + 0.45; dp[o++] = y; dp[o++] = cz;
      }
      this.doorMarker.visible = n > 0;
      this.doorMarkerAttr.needsUpdate = true;
    }
  }


  private buildCheetah(main: THREE.LineBasicMaterial, goalMat: THREE.LineBasicMaterial): void {
    const bones = segments(CHEETAH_BONES.length, main, 'oakDebugSkeleton');
    this.skeletonPos = bones.positions;
    this.skeletonAttr = bones.attr;
    this.add(bones.object);

    const toe = segments(4, goalMat, 'oakDebugToes');
    this.toesPos = toe.positions;
    this.toesAttr = toe.attr;
    this.add(toe.object);
  }

  private updateCheetah(state: FrameState): void {
    const cheetah = this.geometry.cheetah;
    const sp = this.skeletonPos;
    if (!cheetah || !sp || !this.skeletonAttr) return;
    cheetah.fk(state.pose, this.fkOut);
    const toWorld = this.opts.fkToWorld ?? defaultFkToWorld;

    let o = 0;
    for (const bone of CHEETAH_BONES) {
      const a = bone[0];
      const b = bone[1];
      toWorld(this.fkOut[a * 2] ?? 0, this.fkOut[a * 2 + 1] ?? 0, this.scratch);
      toWorld(this.fkOut[b * 2] ?? 0, this.fkOut[b * 2 + 1] ?? 0, this.scratchB);
      sp[o++] = this.scratch[0]; sp[o++] = this.scratch[1]; sp[o++] = this.scratch[2];
      sp[o++] = this.scratchB[0]; sp[o++] = this.scratchB[1]; sp[o++] = this.scratchB[2];
    }
    this.skeletonAttr.needsUpdate = true;

    const tp = this.toesPos;
    if (tp && this.toesAttr) {
      let t = 0;
      const r = 0.06;
      for (const name of ['toe_b', 'toe_f'] as const) {
        const idx = FK_POINT_INDEX[name];
        toWorld(this.fkOut[idx * 2] ?? 0, this.fkOut[idx * 2 + 1] ?? 0, this.scratch);
        const x = this.scratch[0];
        const yy = this.scratch[1];
        const z = this.scratch[2];
        tp[t++] = x - r; tp[t++] = yy; tp[t++] = z;
        tp[t++] = x + r; tp[t++] = yy; tp[t++] = z;
        tp[t++] = x; tp[t++] = yy - r; tp[t++] = z;
        tp[t++] = x; tp[t++] = yy + r; tp[t++] = z;
      }
      this.toesAttr.needsUpdate = true;
    }
  }


  private buildClassic(main: THREE.LineBasicMaterial, goalMat: THREE.LineBasicMaterial): void {
    const classic = this.geometry.classic;
    if (!classic) return;

    const anchors = Object.keys(classic.anchors);
    let minX = Infinity;
    let maxX = -Infinity;
    for (const key of anchors) {
      const a = classic.anchors[key];
      if (!a) continue;
      if (a[0] < minX) minX = a[0];
      if (a[0] > maxX) maxX = a[0];
    }
    const range = this.opts.profileRange ?? [
      Number.isFinite(minX) ? minX : -1,
      Number.isFinite(maxX) ? maxX : 1,
    ];

    const heightAt = classic.heightAt;
    if (heightAt && range[1] > range[0]) {
      const profile = segments(PROFILE_SAMPLES, main, 'oakDebugProfile');
      const p = profile.positions;
      let o = 0;
      for (let i = 0; i < PROFILE_SAMPLES; i++) {
        const t0 = i / PROFILE_SAMPLES;
        const t1 = (i + 1) / PROFILE_SAMPLES;
        const x0 = range[0] + (range[1] - range[0]) * t0;
        const x1 = range[0] + (range[1] - range[0]) * t1;
        p[o++] = x0; p[o++] = heightAt(x0); p[o++] = 0;
        p[o++] = x1; p[o++] = heightAt(x1); p[o++] = 0;
      }
      profile.attr.needsUpdate = true;
      this.add(profile.object);
    }

    if (anchors.length > 0) {
      const marks = segments(anchors.length * 2, goalMat, 'oakDebugAnchors');
      const p = marks.positions;
      let o = 0;
      const r = 0.08;
      for (const key of anchors) {
        const a = classic.anchors[key];
        if (!a) continue;
        p[o++] = a[0] - r; p[o++] = a[1]; p[o++] = a[2];
        p[o++] = a[0] + r; p[o++] = a[1]; p[o++] = a[2];
        p[o++] = a[0]; p[o++] = a[1] - r; p[o++] = a[2];
        p[o++] = a[0]; p[o++] = a[1] + r; p[o++] = a[2];
      }
      marks.attr.needsUpdate = true;
      this.add(marks.object);
    }

    const cross = segments(2, main, 'oakDebugClassicAgent');
    this.classicMarker = cross.object;
    this.classicMarkerPos = cross.positions;
    this.classicMarkerAttr = cross.attr;
    cross.object.visible = false;
    this.add(cross.object);
  }

  private updateClassic(state: FrameState): void {
    const p = this.classicMarkerPos;
    if (!p || !this.classicMarker || !this.classicMarkerAttr) return;
    const pos = this.opts.agentPos?.(state) ?? null;
    if (!pos) {
      this.classicMarker.visible = false;
      return;
    }
    const r = 0.14;
    let o = 0;
    p[o++] = pos[0] - r; p[o++] = pos[1]; p[o++] = pos[2];
    p[o++] = pos[0] + r; p[o++] = pos[1]; p[o++] = pos[2];
    p[o++] = pos[0]; p[o++] = pos[1] - r; p[o++] = pos[2];
    p[o++] = pos[0]; p[o++] = pos[1] + r; p[o++] = pos[2];
    this.classicMarker.visible = true;
    this.classicMarkerAttr.needsUpdate = true;
  }

  private add(object: THREE.LineSegments): void {
    this.geometries.push(object.geometry);
    this.group.add(object);
  }
}

export function defaultFkToWorld(x: number, z: number, out: Vec3Tuple): Vec3Tuple {
  out[0] = x;
  out[1] = z;
  out[2] = 0;
  return out;
}
