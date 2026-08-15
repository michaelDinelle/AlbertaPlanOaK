
import * as THREE from 'three';
import type { EnvId } from '@shared/envs';
import type { EnvGeometry } from '@sim/types';
import { slotValue, type EnvEntry } from '@sim/catalog';
import { CHEETAH_BONES, FK_FLOATS, FK_POINT_INDEX, cheetahGeometry } from '@sim/fk/cheetah';
import { acrobotPoints } from '@sim/envs/acrobot';
import { cartPoleTip } from '@sim/envs/cartpole';
import { pendulumTip } from '@sim/envs/pendulum';
import { CELL } from './grid';
import { emissiveMaterial, type EmissiveMaterial } from './materials';
import type { DesignTokens } from './tokens';
import type { QualityTier } from './types';


export type GhostCategory = 'cheetah' | 'grid' | 'classic';

export type GhostEmphasis = 0 | 1 | 2;

export const GHOST_TRIANGLES_PER_GHOST: Readonly<
  Record<GhostCategory, Readonly<Record<QualityTier, number>>>
> = {
  cheetah: { hero: 1188, medium: 668, thumb: 360 },
  grid: { hero: 180, medium: 82, thumb: 82 },
  classic: { hero: 204, medium: 104, thumb: 104 },
};

export interface GhostLayerOptions {
  category: GhostCategory;
  envId: EnvId;
  entry: EnvEntry | null;
  geometry: EnvGeometry;
  maxGhosts: number;
  tokens: DesignTokens;
  accent: THREE.ColorRepresentation;
  tier: QualityTier;
  name?: string;
}

export interface GhostBank {
  readonly name: string;
  readonly mesh: THREE.InstancedMesh;
  readonly role: 'emitter' | 'shade';
}


export const GHOST_GROUP_NAME = 'oakGhostLayer';

export const GHOST_GLINT_SECONDS = 0.3;
const GLINT_TAU = 0.12;
const GLINT_BRIGHT = 1.1;
const GLINT_SCALE = 0.45;

export const GHOST_EMPHASIS_GAIN: readonly [number, number, number] = [0.55, 0.78, 1.05];
const GHOST_SHADE_SCALE: readonly [number, number, number] = [0.55, 0.8, 1];

export const GHOST_BLOOM_THRESHOLD = 0.85;

export const GHOST_CELL_OFFSET_MAX = 0.22;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

const RAMP_KNEE = 0.82;
const RAMP_VALUE_MIN = 0.35;

const GRID_GHOST_SCALE = 0.85;
const WISP_CORE_RADIUS = 0.16;
const SHELL_SIZE = 1.4;
const SHADE_BLEND = 0.93;
const DEAD_BLEND = 0.62;

const EMITTER_OPACITY = 1;
const SHELL_OPACITY = 0.55;

const CLASSIC_MARKER_RADIUS = 0.07;
const CLASSIC_LINK_THICKNESS = 0.03;

const AXIS_Z =  new THREE.Vector3(0, 0, 1);


function clamp01(x: number): number {
  return Number.isFinite(x) ? (x < 0 ? 0 : x > 1 ? 1 : x) : 0;
}

function bankTriangles(geometry: THREE.BufferGeometry): number {
  const index = geometry.getIndex();
  const count = index !== null ? index.count : (geometry.getAttribute('position')?.count ?? 0);
  return Math.floor(count / 3);
}

function tierDetail(tier: QualityTier): number {
  return tier === 'thumb' ? 0.5 : tier === 'medium' ? 0.75 : 1;
}

function icoDetail(base: number, detail: number): number {
  return Math.max(0, Math.floor(base * detail));
}

export function ghostCellOffset(i: number, n: number, out: Float32Array | number[]): void {
  const count = Math.max(1, n);
  const angle = i * GOLDEN_ANGLE;
  const radius = GHOST_CELL_OFFSET_MAX * Math.sqrt(Math.min(1, (i + 0.5) / count));
  out[0] = Math.cos(angle) * radius;
  out[1] = Math.sin(angle) * radius;
}

export function ghostGlintEnvelope(age: number): number {
  if (!(age >= 0) || age >= GHOST_GLINT_SECONDS) return 0;
  return Math.exp(-age / GLINT_TAU) * (1 - age / GHOST_GLINT_SECONDS);
}

type ClassicKind = 'mountain-car' | 'pendulum' | 'acrobot' | 'cart-pole';

const CLASSIC_KINDS: Partial<Record<EnvId, ClassicKind>> = {
  'MountainCar-cont-v0': 'mountain-car',
  'Pendulum-cont-v0': 'pendulum',
  'Acrobot-cont-v0': 'acrobot',
  'CartPole-cont-v0': 'cart-pole',
};

function poseNeedOf(category: GhostCategory, kind: ClassicKind): number {
  if (category === 'cheetah') return 9;
  if (category === 'grid') return 2;
  return kind === 'acrobot' ? 2 : kind === 'cart-pole' ? 3 : 1;
}

const POSE_STRIDE = 9;


export class GhostLayer {
  readonly group = new THREE.Group();
  readonly category: GhostCategory;
  readonly envId: EnvId;
  readonly tier: QualityTier;
  readonly maxGhosts: number;
  readonly banks: readonly GhostBank[];

  private readonly geometry: EnvGeometry;
  private readonly entry: EnvEntry | null;
  private readonly classicKind: ClassicKind;
  private readonly poseNeed: number;

  private readonly poses: Float32Array;
  private readonly posed: Uint8Array;
  private readonly aliveFlags: Uint8Array;
  private readonly rates: Float32Array;
  private readonly emphasis: Uint8Array;
  private readonly glintAt: Float64Array;
  private readonly glintList: Int32Array;
  private readonly glinting: Uint8Array;
  private glintCount = 0;

  private readonly densityGain: Float32Array;
  private readonly dTableKey: Float64Array;
  private readonly dTableCount: Int32Array;
  private readonly dTableGen: Int32Array;
  private readonly dTableMask: number;
  private dGen = 0;
  private densityDirty = false;

  private readonly poseScratch = new Float32Array(POSE_STRIDE);
  private readonly fkOut = new Float32Array(FK_FLOATS);
  private readonly pointOut = new Float32Array(4);
  private readonly offsetOut = new Float32Array(2);
  private readonly mat = new THREE.Matrix4();
  private readonly pos = new THREE.Vector3();
  private readonly quat = new THREE.Quaternion();
  private readonly scl = new THREE.Vector3();
  private readonly color = new THREE.Color();

  private readonly rampLow = new THREE.Color();
  private readonly rampMid = new THREE.Color();
  private readonly rampHigh = new THREE.Color();
  private readonly shadeColor = new THREE.Color();
  private readonly deadColor = new THREE.Color();
  private readonly deadShadeColor = new THREE.Color();
  private readonly white = new THREE.Color();

  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly materials: EmissiveMaterial[] = [];
  private readonly sprites: THREE.Texture[] = [];
  private readonly bankList: GhostBank[] = [];

  private readonly boneRadius: number;
  private readonly headRadius: number;
  private readonly gridRadiusScale: number;
  private readonly gridHeight: number;

  private time = 0;
  private disposed = false;

  constructor(opts: GhostLayerOptions) {
    this.category = opts.category;
    this.envId = opts.envId;
    this.tier = opts.tier;
    this.geometry = opts.geometry;
    this.entry = opts.entry;
    this.classicKind = CLASSIC_KINDS[opts.envId] ?? 'mountain-car';
    this.poseNeed = poseNeedOf(opts.category, this.classicKind);
    const n = Math.max(0, Math.floor(opts.maxGhosts));
    this.maxGhosts = n;
    const capacity = Math.max(1, n);

    this.group.name = opts.name ?? GHOST_GROUP_NAME;

    this.poses = new Float32Array(capacity * POSE_STRIDE);
    this.posed = new Uint8Array(capacity);
    this.aliveFlags = new Uint8Array(capacity).fill(1);
    this.rates = new Float32Array(capacity);
    this.emphasis = new Uint8Array(capacity);
    this.glintAt = new Float64Array(capacity).fill(-Infinity);
    this.glintList = new Int32Array(capacity);
    this.glinting = new Uint8Array(capacity);

    this.densityGain = new Float32Array(capacity).fill(1);
    let tableSize = 1;
    while (tableSize < capacity * 2) tableSize <<= 1;
    this.dTableKey = new Float64Array(tableSize);
    this.dTableCount = new Int32Array(tableSize);
    this.dTableGen = new Int32Array(tableSize);
    this.dTableMask = tableSize - 1;

    const tokens = opts.tokens;
    const accent = new THREE.Color(opts.accent);
    const night = new THREE.Color(tokens.palettes.night.skyTop);
    this.rampLow.copy(night);
    this.rampMid.copy(accent);
    this.rampHigh.set(tokens.celebrationGold);
    this.shadeColor.copy(accent).lerp(night, SHADE_BLEND);
    this.white.set(tokens.palettes.noon.sun);
    this.deadColor
      .set(tokens.accents['mountain-car'] ?? tokens.celebrationGold)
      .lerp(night, DEAD_BLEND);
    this.deadShadeColor.copy(this.deadColor).lerp(night, SHADE_BLEND);

    const cheetah = this.geometry.cheetah ?? cheetahGeometry;
    this.boneRadius = cheetah.radius;
    this.headRadius = cheetah.headRadius;
    this.gridRadiusScale = WISP_CORE_RADIUS * GRID_GHOST_SCALE;
    this.gridHeight = tokens.wispHeight * GRID_GHOST_SCALE;

    const detail = tierDetail(opts.tier);
    if (opts.category === 'cheetah') this.buildCheetahBanks(capacity, detail);
    else if (opts.category === 'grid') this.buildGridBanks(capacity, detail);
    else this.buildClassicBanks(capacity, detail);

    this.banks = this.bankList;

    for (let i = 0; i < capacity; i++) {
      this.hideGhost(i);
      this.applyTint(i);
    }
    for (const bank of this.bankList) {
      bank.mesh.count = n;
      bank.mesh.instanceMatrix.needsUpdate = true;
      if (bank.mesh.instanceColor) bank.mesh.instanceColor.needsUpdate = true;
    }
  }


  setGhostFrame(i: number, pose: Float32Array, alive: boolean): void {
    if (!this.valid(i)) return;
    const wasAlive = this.aliveFlags[i] === 1;
    this.aliveFlags[i] = alive ? 1 : 0;
    if (alive && this.finiteEnough(pose)) {
      const o = i * POSE_STRIDE;
      for (let k = 0; k < POSE_STRIDE; k++) this.poses[o + k] = pose[k] ?? 0;
      this.posed[i] = 1;
      this.applyGhost(i);
    } else if (this.posed[i] === 1) {
      this.applyGhost(i);
    } else {
      this.hideGhost(i);
    }
    if (wasAlive !== alive) this.applyTint(i);
    this.densityDirty = true;
  }

  setRate(i: number, rate01: number): void {
    if (!this.valid(i)) return;
    const next = clamp01(rate01);
    if (this.rates[i] === next) return;
    this.rates[i] = next;
    this.applyTint(i);
  }

  setEmphasis(i: number, level: GhostEmphasis): void {
    if (!this.valid(i)) return;
    const next = level === 2 ? 2 : level === 1 ? 1 : 0;
    if (this.emphasis[i] === next) return;
    this.emphasis[i] = next;
    this.applyGhost(i);
    this.applyTint(i);
  }

  glint(i: number): void {
    if (!this.valid(i)) return;
    if (this.aliveFlags[i] !== 1 || this.posed[i] !== 1) return;
    const age = this.time - (this.glintAt[i] ?? -Infinity);
    if (age < GHOST_GLINT_SECONDS) return;
    this.glintAt[i] = this.time;
    if (this.glinting[i] !== 1) {
      this.glinting[i] = 1;
      this.glintList[this.glintCount++] = i;
    }
    this.applyGhost(i);
    this.applyTint(i);
  }

  update(dt: number, time: number): void {
    void dt;
    this.time = Number.isFinite(time) ? time : this.time;
    if (this.densityDirty) {
      this.densityDirty = false;
      this.recomputeDensity();
    }
    if (this.glintCount === 0) return;
    let write = 0;
    for (let k = 0; k < this.glintCount; k++) {
      const i = this.glintList[k] ?? 0;
      const age = this.time - (this.glintAt[i] ?? -Infinity);
      this.applyGhost(i);
      this.applyTint(i);
      if (age >= 0 && age < GHOST_GLINT_SECONDS) {
        this.glintList[write++] = i;
      } else {
        this.glinting[i] = 0;
      }
    }
    this.glintCount = write;
  }

  draws(): number {
    if (!this.group.visible) return 0;
    let n = 0;
    for (const bank of this.bankList) if (bank.mesh.visible && bank.mesh.count > 0) n++;
    return n;
  }

  triangles(): number {
    if (!this.group.visible) return 0;
    let n = 0;
    for (const bank of this.bankList) {
      if (!bank.mesh.visible || bank.mesh.count <= 0) continue;
      n += bankTriangles(bank.mesh.geometry) * bank.mesh.count;
    }
    return n;
  }

  trianglesPerGhost(): number {
    let n = 0;
    for (const bank of this.bankList) n += bankTriangles(bank.mesh.geometry);
    return n;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const g of this.geometries) g.dispose();
    this.geometries.length = 0;
    for (const m of this.materials) m.dispose();
    this.materials.length = 0;
    for (const t of this.sprites) t.dispose();
    this.sprites.length = 0;
    for (const bank of this.bankList) {
      bank.mesh.count = 0;
      bank.mesh.dispose();
    }
    this.bankList.length = 0;
    this.group.removeFromParent();
    this.group.clear();
  }


  rampColor(t: number, out: THREE.Color): THREE.Color {
    const x = clamp01(t);
    if (x <= RAMP_KNEE) out.copy(this.rampLow).lerp(this.rampMid, RAMP_KNEE > 0 ? x / RAMP_KNEE : 1);
    else out.copy(this.rampMid).lerp(this.rampHigh, (x - RAMP_KNEE) / (1 - RAMP_KNEE));
    return out.multiplyScalar(RAMP_VALUE_MIN + (1 - RAMP_VALUE_MIN) * x);
  }

  emitterColor(i: number, out: THREE.Color): THREE.Color {
    const emph = (this.emphasis[i] ?? 0) as 0 | 1 | 2;
    const gain = GHOST_EMPHASIS_GAIN[emph] ?? GHOST_EMPHASIS_GAIN[0];
    if (this.aliveFlags[i] !== 1) return out.copy(this.deadColor).multiplyScalar(gain);
    this.rampColor(this.rates[i] ?? 0, out);
    const density = emph === 0 ? (this.densityGain[i] ?? 1) : 1;
    return out.multiplyScalar(gain * density * (1 + GLINT_BRIGHT * this.glintOf(i)));
  }

  private glintOf(i: number): number {
    if (this.glinting[i] !== 1) return 0;
    return ghostGlintEnvelope(this.time - (this.glintAt[i] ?? -Infinity));
  }

  private densityKey(i: number): number {
    const o = i * POSE_STRIDE;
    const x = this.poses[o] ?? 0;
    if (this.category === 'grid') {
      return Math.round(x) * 4096 + Math.round(this.poses[o + 1] ?? 0);
    }
    if (this.category === 'cheetah') return Math.round(x / 1.5);
    return Math.round(x / 0.15);
  }

  private recomputeDensity(): void {
    const gen = ++this.dGen;
    const mask = this.dTableMask;
    for (let i = 0; i < this.maxGhosts; i++) {
      if (this.posed[i] !== 1 || this.aliveFlags[i] !== 1) continue;
      const key = this.densityKey(i);
      let slot = (((key * 2654435761) | 0) >>> 0) & mask;
      for (;;) {
        if (this.dTableGen[slot] !== gen) {
          this.dTableGen[slot] = gen;
          this.dTableKey[slot] = key;
          this.dTableCount[slot] = 1;
          break;
        }
        if (this.dTableKey[slot] === key) {
          this.dTableCount[slot] = (this.dTableCount[slot] ?? 0) + 1;
          break;
        }
        slot = (slot + 1) & mask;
      }
    }
    for (let i = 0; i < this.maxGhosts; i++) {
      let gain = 1;
      if (this.posed[i] === 1 && this.aliveFlags[i] === 1) {
        const key = this.densityKey(i);
        let slot = (((key * 2654435761) | 0) >>> 0) & mask;
        for (;;) {
          if (this.dTableGen[slot] !== gen) break;
          if (this.dTableKey[slot] === key) {
            gain = 1 / Math.sqrt(this.dTableCount[slot] ?? 1);
            break;
          }
          slot = (slot + 1) & mask;
        }
      }
      if (Math.abs(gain - (this.densityGain[i] ?? 1)) > 1e-3) {
        this.densityGain[i] = gain;
        this.applyTint(i);
      }
    }
  }

  private applyTint(i: number): void {
    const emitter = this.emitterColor(i, this.color);
    for (const bank of this.bankList) {
      if (bank.role !== 'emitter') continue;
      bank.mesh.setColorAt(i, emitter);
      if (bank.mesh.instanceColor) bank.mesh.instanceColor.needsUpdate = true;
    }
    const shade = this.aliveFlags[i] === 1 ? this.shadeColor : this.deadShadeColor;
    for (const bank of this.bankList) {
      if (bank.role !== 'shade') continue;
      bank.mesh.setColorAt(i, shade);
      if (bank.mesh.instanceColor) bank.mesh.instanceColor.needsUpdate = true;
    }
  }


  private valid(i: number): boolean {
    return !this.disposed && Number.isInteger(i) && i >= 0 && i < this.maxGhosts;
  }

  private finiteEnough(pose: ArrayLike<number>): boolean {
    for (let k = 0; k < this.poseNeed; k++) {
      if (!Number.isFinite(pose[k] ?? Number.NaN)) return false;
    }
    return true;
  }

  private applyGhost(i: number): void {
    if (this.posed[i] !== 1) {
      this.hideGhost(i);
      return;
    }
    const o = i * POSE_STRIDE;
    for (let k = 0; k < POSE_STRIDE; k++) this.poseScratch[k] = this.poses[o + k] ?? 0;
    const puff = 1 + GLINT_SCALE * this.glintOf(i);
    if (this.category === 'cheetah') this.applyCheetah(i, puff);
    else if (this.category === 'grid') this.applyGrid(i, puff);
    else this.applyClassic(i, puff);
    for (const bank of this.bankList) bank.mesh.instanceMatrix.needsUpdate = true;
  }

  private applyCheetah(i: number, puff: number): void {
    const fk = (this.geometry.cheetah ?? cheetahGeometry).fk(this.poseScratch, this.fkOut);
    const r = this.boneRadius * puff;
    for (let b = 0; b < CHEETAH_BONES.length; b++) {
      const bone = CHEETAH_BONES[b];
      const mesh = this.bankList[b]?.mesh;
      if (!bone || !mesh) continue;
      const a0 = bone[0] * 2;
      const a1 = bone[1] * 2;
      this.setSegment(mesh, i, fk[a0] ?? 0, fk[a0 + 1] ?? 0, fk[a1] ?? 0, fk[a1 + 1] ?? 0, r);
    }
    const head = this.bankList[CHEETAH_BONES.length]?.mesh;
    if (head) {
      const h = FK_POINT_INDEX.head * 2;
      this.setPoint(head, i, fk[h] ?? 0, fk[h + 1] ?? 0, 0, this.headRadius * puff);
    }
  }

  private applyGrid(i: number, puff: number): void {
    const grid = this.geometry.grid;
    const w = grid?.w ?? 1;
    const h = grid?.h ?? 1;
    ghostCellOffset(i, this.maxGhosts, this.offsetOut);
    const entry = this.entry;
    const cx = entry === null ? (this.poseScratch[0] ?? 0) : slotValue(entry, this.poseScratch, 'x');
    const cy = entry === null ? (this.poseScratch[1] ?? 0) : slotValue(entry, this.poseScratch, 'y');
    const gx = Number.isFinite(cx) ? cx : 0;
    const gy = Number.isFinite(cy) ? cy : 0;
    const wx = (gx - (w - 1) / 2) * CELL + (this.offsetOut[0] ?? 0) * CELL;
    const wz = (gy - (h - 1) / 2) * CELL + (this.offsetOut[1] ?? 0) * CELL;
    const emph = (this.emphasis[i] ?? 0) as 0 | 1 | 2;
    const dark = GHOST_SHADE_SCALE[emph] ?? GHOST_SHADE_SCALE[0];

    const core = this.bankList[0]?.mesh;
    if (core) this.setOrb(core, i, wx, this.gridHeight, wz, this.gridRadiusScale * puff);
    const shell = this.bankList[1]?.mesh;
    if (shell) {
      const ring = 1 + (SHELL_SIZE - 1) * dark;
      this.setOrb(shell, i, wx, this.gridHeight, wz, this.gridRadiusScale * ring * puff);
    }
  }

  private applyClassic(i: number, puff: number): void {
    const marker = this.bankList[0]?.mesh;
    const link0 = this.bankList[1]?.mesh;
    const link1 = this.bankList[2]?.mesh;
    const r = CLASSIC_MARKER_RADIUS * puff;
    const t = CLASSIC_LINK_THICKNESS * puff;
    const anchors = this.geometry.classic?.anchors;
    const pivot = anchors?.['pivot'];
    const px = pivot?.[0] ?? 0;
    const py = pivot?.[1] ?? 0;
    const p = this.pointOut;

    switch (this.classicKind) {
      case 'mountain-car': {
        const x = this.poseScratch[0] ?? 0;
        const heightAt = this.geometry.classic?.heightAt;
        const y = heightAt ? heightAt(x) : 0;
        if (marker) this.setPoint(marker, i, x, y, 0, r);
        return;
      }
      case 'pendulum': {
        pendulumTip(this.poseScratch[0] ?? 0, p);
        const bx = p[0] ?? 0;
        const by = p[1] ?? 0;
        if (link0) this.setSegment(link0, i, px, py, bx, by, t);
        if (marker) this.setPoint(marker, i, bx, by, 0, r);
        return;
      }
      case 'cart-pole': {
        const x = this.poseScratch[0] ?? 0;
        cartPoleTip(x, this.poseScratch[2] ?? 0, p);
        const tx = p[0] ?? 0;
        const ty = p[1] ?? 0;
        if (link0) this.setSegment(link0, i, x, py, tx, ty, t);
        if (marker) this.setPoint(marker, i, x, py, 0, r);
        return;
      }
      case 'acrobot': {
        acrobotPoints(this.poseScratch[0] ?? 0, this.poseScratch[1] ?? 0, p);
        const ex = p[0] ?? 0;
        const ey = p[1] ?? 0;
        const tx = p[2] ?? 0;
        const ty = p[3] ?? 0;
        if (link0) this.setSegment(link0, i, px, py, ex, ey, t);
        if (link1) this.setSegment(link1, i, ex, ey, tx, ty, t);
        if (marker) this.setPoint(marker, i, tx, ty, 0, r);
        return;
      }
    }
  }

  private setSegment(
    mesh: THREE.InstancedMesh,
    i: number,
    ax: number,
    ay: number,
    bx: number,
    by: number,
    thickness: number,
  ): void {
    const dx = bx - ax;
    const dy = by - ay;
    const len = Math.sqrt(dx * dx + dy * dy);
    this.pos.set((ax + bx) * 0.5, (ay + by) * 0.5, 0);
    this.quat.setFromAxisAngle(AXIS_Z, Math.atan2(-dx, dy));
    this.scl.set(thickness, len > 1e-6 ? len : 1e-6, thickness);
    this.mat.compose(this.pos, this.quat, this.scl);
    mesh.setMatrixAt(i, this.mat);
  }

  private setPoint(
    mesh: THREE.InstancedMesh,
    i: number,
    x: number,
    y: number,
    z: number,
    radius: number,
  ): void {
    this.pos.set(x, y, z);
    this.quat.identity();
    this.scl.setScalar(radius);
    this.mat.compose(this.pos, this.quat, this.scl);
    mesh.setMatrixAt(i, this.mat);
  }

  private setOrb(
    mesh: THREE.InstancedMesh,
    i: number,
    x: number,
    y: number,
    z: number,
    radius: number,
  ): void {
    this.setPoint(mesh, i, x, y, z, radius);
  }

  private hideGhost(i: number): void {
    this.pos.set(0, 0, 0);
    this.quat.identity();
    this.scl.set(0, 0, 0);
    this.mat.compose(this.pos, this.quat, this.scl);
    for (const bank of this.bankList) {
      bank.mesh.setMatrixAt(i, this.mat);
      bank.mesh.instanceMatrix.needsUpdate = true;
    }
  }


  private addBank(
    name: string,
    role: 'emitter' | 'shade',
    geometry: THREE.BufferGeometry,
    material: EmissiveMaterial,
    capacity: number,
    renderOrder: number,
  ): GhostBank {
    const mesh = new THREE.InstancedMesh(geometry, material, capacity);
    mesh.name = `${GHOST_GROUP_NAME}:${name}`;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const colors = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3).fill(1), 3);
    colors.setUsage(THREE.DynamicDrawUsage);
    mesh.instanceColor = colors;
    mesh.frustumCulled = false;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.renderOrder = renderOrder;
    this.geometries.push(geometry);
    this.materials.push(material);
    this.group.add(mesh);
    const bank: GhostBank = { name, mesh, role };
    this.bankList.push(bank);
    return bank;
  }

  private emitterMaterial(name: string, map?: THREE.Texture): EmissiveMaterial {
    return emissiveMaterial(this.white, 1, {
      transparent: true,
      opacity: EMITTER_OPACITY,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      map: map ?? null,
      name: `${GHOST_GROUP_NAME}:${name}`,
    });
  }

  private shadeMaterial(name: string, opacity: number, map?: THREE.Texture): EmissiveMaterial {
    return emissiveMaterial(this.white, 1, {
      transparent: true,
      opacity,
      depthWrite: false,
      map: map ?? null,
      name: `${GHOST_GROUP_NAME}:${name}`,
    });
  }

  private buildCheetahBanks(capacity: number, detail: number): void {
    const capSeg = Math.max(2, Math.round(4 * detail));
    const radialSeg = Math.max(4, Math.round(8 * detail));
    for (let b = 0; b < CHEETAH_BONES.length; b++) {
      const geo = new THREE.CapsuleGeometry(1, 1, capSeg, radialSeg);
      geo.name = `${GHOST_GROUP_NAME}:bone${b}`;
      this.addBank(`bone${b}`, 'emitter', geo, this.emitterMaterial(`bone${b}`), capacity, -2);
    }
    const head = new THREE.IcosahedronGeometry(1, icoDetail(2, detail));
    head.name = `${GHOST_GROUP_NAME}:head`;
    this.addBank('head', 'emitter', head, this.emitterMaterial('head'), capacity, -2);
  }

  private buildGridBanks(capacity: number, detail: number): void {
    const core = new THREE.IcosahedronGeometry(1, icoDetail(2, detail));
    core.name = `${GHOST_GROUP_NAME}:core`;
    this.addBank('core', 'emitter', core, this.emitterMaterial('core'), capacity, -2);

    const shell = new THREE.IcosahedronGeometry(1, icoDetail(1, detail));
    shell.name = `${GHOST_GROUP_NAME}:shell`;
    this.addBank('shell', 'shade', shell, this.shadeMaterial('shell', SHELL_OPACITY), capacity, -3);
  }

  private buildClassicBanks(capacity: number, detail: number): void {
    const marker = new THREE.IcosahedronGeometry(1, icoDetail(2, detail));
    marker.name = `${GHOST_GROUP_NAME}:marker`;
    this.addBank('marker', 'emitter', marker, this.emitterMaterial('marker'), capacity, -2);

    const links = this.classicKind === 'acrobot' ? 2 : this.classicKind === 'mountain-car' ? 0 : 1;
    for (let k = 0; k < links; k++) {
      const geo = new THREE.BoxGeometry(1, 1, 1);
      geo.name = `${GHOST_GROUP_NAME}:link${k}`;
      this.addBank(`link${k}`, 'emitter', geo, this.emitterMaterial(`link${k}`), capacity, -2);
    }
  }
}

export function makeGhostLayer(opts: GhostLayerOptions): GhostLayer {
  return new GhostLayer(opts);
}
