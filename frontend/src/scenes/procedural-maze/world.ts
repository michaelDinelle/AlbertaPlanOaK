import * as THREE from 'three';
import {
  buildCellField,
  CELL,
  cellToWorld,
  collapseWallInstance,
  excludeCellCenters,
  excludeWallCells,
  isWall,
  type CellFieldHandle,
  type GridSize,
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms, type OakMaterial } from '@kit/materials';
import { blendTokens, clamp01, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f, rngAt, streamFor } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import {
  excludeAny,
  excludeRect,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
} from '@kit/vegetation';
import { makeWorld, type WorldCore } from '@kit/world';
import { PROCMAZE_H, PROCMAZE_W, proceduralMazeDecoder } from '@sim/envs/proceduralmaze';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';

export const RUIN_HALF = (PROCMAZE_W * CELL) / 2;
export const TERRAIN_SIZE = 44;
export const TERRAIN_SEGMENTS = 128;
export const RUIN_SINK = -0.07;
export const FOG_EXTENT = TERRAIN_SIZE * 12;
export const SUN_DIR: readonly [number, number, number] = [-0.36, 0.874, -0.325];
export const SHADOW_RADIUS = 12;

export const CAMERA_AZIMUTH = 0.62;
export const CAMERA_POLAR = 0.73;
export const CAMERA_TARGET_BIAS = 1.2;
export const CAMERA_TARGET_Y = 0.15;
export const CAMERA_DISTANCE = 19.0;
export const CAMERA_STANDOFF = CAMERA_TARGET_BIAS + CAMERA_DISTANCE * Math.sin(CAMERA_POLAR);

export const BASE_COURSE = 0.5;
export const BASE_INSET = 0.02;
export const UPPER_INSET = 0.11;
export const CAP_INSET = 0.21;

export function resolveGrid(geometry: EnvGeometry, seed: number): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === PROCMAZE_W && g.h === PROCMAZE_H && (g.carvedPath?.length ?? 0) > 0) return g;
  const st = proceduralMazeDecoder.initStatic(seed);
  const resolved = proceduralMazeDecoder.geometry(st, seed).grid;
  if (!resolved) throw new Error('procedural-maze: decoder produced no grid geometry');
  return resolved;
}

export function wallNeighbours(grid: GridGeometry, x: number, y: number): number {
  let n = 0;
  if (isWall(grid, x, y - 1)) n++;
  if (isWall(grid, x, y + 1)) n++;
  if (isWall(grid, x - 1, y)) n++;
  if (isWall(grid, x + 1, y)) n++;
  return n;
}

export function cornerCells(grid: GridSize): readonly GridCell[] {
  return [
    { x: 0, y: 0 },
    { x: grid.w - 1, y: 0 },
    { x: 0, y: grid.h - 1 },
    { x: grid.w - 1, y: grid.h - 1 },
  ];
}

export interface ArchCell {
  cell: GridCell;
  eastWest: boolean;
}

export function archCells(grid: GridGeometry): ArchCell[] {
  const out: ArchCell[] = [];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isWall(grid, x, y)) continue;
      const n = isWall(grid, x, y - 1);
      const s = isWall(grid, x, y + 1);
      const w = isWall(grid, x - 1, y);
      const e = isWall(grid, x + 1, y);
      if (n && s && !w && !e) out.push({ cell: { x, y }, eastWest: true });
      else if (w && e && !n && !s) out.push({ cell: { x, y }, eastWest: false });
    }
  }
  return out;
}

export interface WallSkin {
  upper: number;
  cap: number;
  crown: number;
  upperYaw: number;
  capYaw: number;
  capTilt: number;
  moss: number;
  cushion: boolean;
  ivy: number;
  ivyFace: number;
}

const UPPER_MIN = 0.16;
const UPPER_SPAN = 0.3;
const CAP_MIN = 0.09;
const CAP_SPAN = 0.1;
const IVY_MIN_CROWN = 0.86;

export const IVY_CLING = 0.035;
export const IVY_FLOOR_GAP = 0.06;

export interface IvyLedge {
  y: number;
  drop: number;
}

export function ivyLedge(skin: WallSkin, strand: number): IvyLedge {
  if (strand === 0 && skin.upper > 0) return { y: BASE_COURSE + skin.upper, drop: skin.upper };
  return { y: BASE_COURSE, drop: BASE_COURSE - IVY_FLOOR_GAP };
}

export function skinWallCell(grid: GridGeometry, x: number, y: number, seed: number): WallSkin {
  const r = rngAt(x, y, seed ^ 0x5ee1d0a7);
  const standing = r();
  const upper = standing < 0.16 ? 0 : UPPER_MIN + UPPER_SPAN * r();
  const capRoll = r();
  const cap = upper > 0 && capRoll < 0.55 ? CAP_MIN + CAP_SPAN * r() : 0;
  const crown = BASE_COURSE + upper + cap;
  const upperYaw = (r() - 0.5) * 0.24;
  const capYaw = (r() - 0.5) * 0.7;
  const capTilt = (r() - 0.5) * 0.18;
  const damp = wallNeighbours(grid, x, y) / 4;
  const moss = clamp01(0.12 + 0.55 * r() + 0.4 * damp - 0.2);
  const cushion = upper > 0 && r() < 0.34;
  const ivyRoll = r();
  const ivy = crown >= IVY_MIN_CROWN && ivyRoll < 0.5 ? 1 + (ivyRoll < 0.18 ? 1 : 0) : 0;
  const ivyFace = Math.min(3, Math.floor(r() * 4));
  return { upper, cap, crown, upperYaw, capYaw, capTilt, moss, cushion, ivy, ivyFace };
}

export function blockHalfWidthAt(skin: WallSkin, y: number): number {
  if (y <= BASE_COURSE) return CELL / 2 - BASE_INSET;
  if (y <= BASE_COURSE + skin.upper) return CELL / 2 - UPPER_INSET;
  if (y <= skin.crown) return CELL / 2 - CAP_INSET;
  return 0;
}

export interface RuinSkin {
  stoneLight: THREE.Color;
  stoneDark: THREE.Color;
  moss: THREE.Color;
  mossDeep: THREE.Color;
  flag: THREE.Color;
  flagDamp: THREE.Color;
  soil: THREE.Color;
  grass: THREE.Color;
  bloom: THREE.Color;
  canopy: THREE.Color;
  sunWarm: THREE.Color;
  bird: THREE.Color;
}

export function ruinSkin(palette: Palette, tokens: typeof TOKENS): RuinSkin {
  const accent = tokens.accents['procedural-maze'] ?? tokens.celebrationGold;
  const tan = tokens.palettes.dusk.groundA;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  return {
    stoneLight: shade(blendTokens(silver, tan, 0.86), 0.06),
    stoneDark: shade(blendTokens(tan, palette.hemiGround, 0.22), 0.4),
    moss: shade(blendTokens(accent, palette.groundA, 0.62), 0.24),
    mossDeep: shade(blendTokens(palette.groundA, accent, 0.12), 0.48),
    flag: shade(blendTokens(silver, tan, 0.72), 0.42),
    flagDamp: shade(blendTokens(palette.groundA, tan, 0.4), 0.46),
    soil: shade(blendTokens(palette.hemiGround, tan, 0.55), 0.44),
    grass: shade(blendTokens(palette.groundA, accent, 0.45), 0.12),
    bloom: shade(blendTokens(palette.groundB, tokens.celebrationGold, 0.4), 0.12),
    canopy: shade(blendTokens(palette.groundA, palette.groundB, 0.3), 0.24),
    sunWarm: blendTokens(palette.sun, tokens.celebrationGold, 0.42),
    bird: shade(blendTokens(palette.hemiGround, tan, 0.3), 0.72),
  };
}

const TONE_JITTER = 0.09;

function toneJitter(x: number, y: number, seed: number): number {
  return 1 + (hash2f(x, y, seed) - 0.5) * TONE_JITTER;
}

export function floorTintFor(
  grid: GridGeometry,
  skin: RuinSkin,
  x: number,
  y: number,
  seed: number,
  out?: THREE.Color,
): THREE.Color {
  const c = out ?? new THREE.Color();
  if (isWall(grid, x, y)) c.copy(skin.soil);
  else {
    const damp = wallNeighbours(grid, x, y) / 4;
    c.copy(skin.flag).lerp(skin.flagDamp, 0.25 + 0.6 * damp);
  }
  return c.multiplyScalar(toneJitter(x, y, seed));
}

export function wallTintFor(
  grid: GridGeometry,
  skin: RuinSkin,
  x: number,
  y: number,
  seed: number,
  out?: THREE.Color,
): THREE.Color {
  const s = skinWallCell(grid, x, y, seed);
  const c = out ?? new THREE.Color();
  return c
    .copy(skin.stoneDark)
    .lerp(skin.mossDeep, 0.2 + 0.55 * s.moss)
    .multiplyScalar(toneJitter(x, y, seed ^ 0x51ed270b));
}

export function makeForestFloorHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.max(Math.abs(x), Math.abs(z)) - RUIN_HALF;
    const w = smoothstep01(out / 5.5);
    if (w <= 0) return 0;
    const n1 = valueNoise2(x * 0.078 + 3.1, z * 0.078 + 9.4, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.2 + 6.7, z * 0.2 + 2.2, 64, seed ^ 0x51ed270b) - 0.5;
    return w * (n1 * 2.6 + n2 * 0.7);
  };
}

export interface IvyStrand {
  cell: GridCell;
  strand: number;
}

export interface CoursesHandle {
  group: THREE.Group;
  upperMesh: THREE.InstancedMesh;
  capMesh: THREE.InstancedMesh;
  cushionMesh: THREE.InstancedMesh;
  ivyMesh: THREE.InstancedMesh | null;
  upperCells: GridCell[];
  capCells: GridCell[];
  ivyCells: IvyStrand[];
  setTime(t: number): void;
  dispose(): void;
}

export function buildCourses(
  grid: GridGeometry,
  seed: number,
  skin: RuinSkin,
  palette: Palette,
  skip: (x: number, y: number) => boolean,
): CoursesHandle {
  const group = new THREE.Group();
  group.name = 'ruinCourses';
  const detail = 1;

  const upperCells: GridCell[] = [];
  const capCells: GridCell[] = [];
  const cushionCells: GridCell[] = [];
  const ivyCells: IvyStrand[] = [];
  const skins = new Map<number, WallSkin>();

  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (!isWall(grid, x, y) || skip(x, y)) continue;
      const s = skinWallCell(grid, x, y, seed);
      skins.set(y * grid.w + x, s);
      if (s.upper > 0) upperCells.push({ x, y });
      if (s.cap > 0) capCells.push({ x, y });
      if (s.cushion) cushionCells.push({ x, y });
      for (let i = 0; i < s.ivy; i++) ivyCells.push({ cell: { x, y }, strand: i });
    }
  }

  const skinAt = (x: number, y: number): WallSkin =>
    skins.get(y * grid.w + x) ?? skinWallCell(grid, x, y, seed);

  const white = palette.sun;
  const stoneMaterial = oakMaterial({
    color: white,
    roughness: 0.94,
    flatShading: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.22,
    toonSteps: 3,
    toonSoftness: 0.28,
    toonFloor: 0.42,
    name: 'ruinCourseStone',
  });
  const mossMaterial = oakMaterial({
    color: white,
    roughness: 0.98,
    flatShading: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.44,
    name: 'ruinMossCushion',
  });
  const ivyMaterial = oakMaterial({
    color: white,
    roughness: 0.95,
    flatShading: true,
    side: THREE.DoubleSide,
    vertexColors: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.36,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.58,
    wind: { amplitude: 0.045, frequency: 1.35, height: 0.62 },
    name: 'ruinIvy',
  });

  const upperGeometry = new THREE.BoxGeometry(CELL - UPPER_INSET * 2, 1, CELL - UPPER_INSET * 2);
  upperGeometry.translate(0, 0.5, 0);
  upperGeometry.name = 'ruinUpperCourse';
  const capGeometry = brokenSlabGeometry(detail);
  const cushionGeometry = new THREE.IcosahedronGeometry(0.34, 1);
  cushionGeometry.scale(1, 0.38, 1);
  cushionGeometry.name = 'ruinMossCushion';
  const ivyGeometry = ivyStrandGeometry(detail);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);
  const color = new THREE.Color();
  const world: Vec3Tuple = [0, 0, 0];

  const instanced = (
    geometry: THREE.BufferGeometry,
    material: OakMaterial,
    count: number,
    name: string,
    cast: boolean,
  ): THREE.InstancedMesh => {
    const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
    mesh.name = name;
    mesh.count = count;
    mesh.castShadow = cast;
    mesh.receiveShadow = true;
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3);
    return mesh;
  };

  const upperMesh = instanced(upperGeometry, stoneMaterial, upperCells.length, 'ruinUpperCourse', true);
  const capMesh = instanced(capGeometry, stoneMaterial, capCells.length, 'ruinCapstones', true);
  const cushionMesh = instanced(cushionGeometry, mossMaterial, cushionCells.length, 'ruinMossCushions', false);
  const ivyMesh =
    false || ivyCells.length === 0
      ? null
      : instanced(ivyGeometry, ivyMaterial, ivyCells.length, 'ruinIvy', false);

  for (let i = 0; i < upperCells.length; i++) {
    const c = upperCells[i] as GridCell;
    const s = skinAt(c.x, c.y);
    cellToWorld(c.x, c.y, grid, world);
    position.set(world[0], BASE_COURSE, world[2]);
    euler.set(0, s.upperYaw, 0);
    quaternion.setFromEuler(euler);
    scale.set(1, s.upper, 1);
    matrix.compose(position, quaternion, scale);
    upperMesh.setMatrixAt(i, matrix);
    color.copy(skin.stoneLight).lerp(skin.moss, s.moss * 0.3).multiplyScalar(0.86 + 0.28 * hash2f(c.x, c.y, seed));
    upperMesh.setColorAt(i, color);
  }

  for (let i = 0; i < capCells.length; i++) {
    const c = capCells[i] as GridCell;
    const s = skinAt(c.x, c.y);
    cellToWorld(c.x, c.y, grid, world);
    position.set(world[0], BASE_COURSE + s.upper, world[2]);
    euler.set(s.capTilt, s.capYaw, 0);
    quaternion.setFromEuler(euler);
    scale.set(1, s.cap, 1);
    matrix.compose(position, quaternion, scale);
    capMesh.setMatrixAt(i, matrix);
    color.copy(skin.stoneLight).lerp(skin.moss, s.moss * 0.16).multiplyScalar(0.98 + 0.22 * hash2f(c.y, c.x, seed));
    capMesh.setColorAt(i, color);
  }

  for (let i = 0; i < cushionCells.length; i++) {
    const c = cushionCells[i] as GridCell;
    const s = skinAt(c.x, c.y);
    cellToWorld(c.x, c.y, grid, world);
    const r = hash2f(c.x * 7 + 1, c.y * 13 + 5, seed);
    position.set(world[0] + (r - 0.5) * 0.22, s.crown - 0.04, world[2] + (hash2f(c.y, c.x, seed ^ 0x9e11) - 0.5) * 0.22);
    euler.set(0, r * 6.283, 0);
    quaternion.setFromEuler(euler);
    const k = 0.5 + 0.44 * s.moss;
    scale.set(k, 0.66 + 0.5 * r, k);
    matrix.compose(position, quaternion, scale);
    cushionMesh.setMatrixAt(i, matrix);
    color.copy(skin.moss).lerp(skin.mossDeep, 0.2 + 0.5 * r);
    cushionMesh.setColorAt(i, color);
  }

  if (ivyMesh) {
    for (let i = 0; i < ivyCells.length; i++) {
      const entry = ivyCells[i] as IvyStrand;
      const c = entry.cell;
      const s = skinAt(c.x, c.y);
      cellToWorld(c.x, c.y, grid, world);
      const face = (s.ivyFace + entry.strand * 2) % 4;
      const a = (face * Math.PI) / 2;
      const ledge = ivyLedge(s, entry.strand);
      const off = blockHalfWidthAt(s, ledge.y) + IVY_CLING;
      const jitter = hash2f(c.x * 3 + entry.strand, c.y * 5, seed ^ 0x1de2) - 0.5;
      position.set(
        world[0] + Math.sin(a) * off + Math.cos(a) * jitter * 0.5,
        ledge.y,
        world[2] + Math.cos(a) * off - Math.sin(a) * jitter * 0.5,
      );
      euler.set(0, a, 0);
      quaternion.setFromEuler(euler);
      scale.set(0.85 + 0.4 * (jitter + 0.5), ledge.drop, 0.85 + 0.4 * (jitter + 0.5));
      matrix.compose(position, quaternion, scale);
      ivyMesh.setMatrixAt(i, matrix);
      color.copy(skin.moss).lerp(skin.mossDeep, 0.12 + 0.34 * (jitter + 0.5));
      ivyMesh.setColorAt(i, color);
    }
  }

  const meshes: THREE.InstancedMesh[] = [upperMesh, capMesh, cushionMesh];
  if (ivyMesh) meshes.push(ivyMesh);
  for (const m of meshes) {
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    m.computeBoundingSphere();
    if (m.count > 0) group.add(m);
  }

  return {
    group,
    upperMesh,
    capMesh,
    cushionMesh,
    ivyMesh,
    upperCells,
    capCells,
    ivyCells,
    setTime(t) {
      oakUniforms(ivyMaterial)?.setTime(t);
      oakUniforms(stoneMaterial)?.setTime(t);
      oakUniforms(mossMaterial)?.setTime(t);
    },
    dispose() {
      upperMesh.dispose();
      capMesh.dispose();
      cushionMesh.dispose();
      ivyMesh?.dispose();
      upperGeometry.dispose();
      capGeometry.dispose();
      cushionGeometry.dispose();
      ivyGeometry.dispose();
      stoneMaterial.dispose();
      mossMaterial.dispose();
      ivyMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function brokenSlabGeometry(detail: number): THREE.BufferGeometry {
  const size = CELL - CAP_INSET * 2;
  const g = new THREE.BoxGeometry(size, 1, size, 1, 1, 1);
  g.translate(0, 0.5, 0);
  const pos = g.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    if (y > 0.5 && x > 0 && z > 0) {
      pos.setX(i, x * 0.72);
      pos.setZ(i, z * 0.72);
      pos.setY(i, y * (detail > 0.5 ? 0.88 : 0.94));
    }
  }
  pos.needsUpdate = true;
  g.computeVertexNormals();
  g.name = 'ruinCapstone';
  return g;
}

function ivyStrandGeometry(detail: number): THREE.BufferGeometry {
  const leaves = Math.max(3, Math.round(6 * detail));
  const positions = new Float32Array(leaves * 3 * 3);
  const colors = new Float32Array(leaves * 3 * 3);
  for (let i = 0; i < leaves; i++) {
    const t0 = i / leaves;
    const t1 = (i + 1) / leaves;
    const side = i % 2 === 0 ? 1 : -1;
    const w = 0.2 * (1 - 0.4 * t0);
    const o = i * 9;
    positions[o] = 0;
    positions[o + 1] = -t0;
    positions[o + 2] = 0;
    positions[o + 3] = 0;
    positions[o + 4] = -t1;
    positions[o + 5] = 0;
    positions[o + 6] = side * w * 1.7;
    positions[o + 7] = -(t0 + t1) * 0.5 + 0.02;
    positions[o + 8] = w * 0.6;
    for (let v = 0; v < 3; v++) {
      const k = v === 2 ? 1.15 - 0.35 * t0 : 0.72 - 0.28 * t0;
      colors[o + v * 3] = k;
      colors[o + v * 3 + 1] = k;
      colors[o + v * 3 + 2] = k;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'ruinIvyStrand';
  return g;
}

export const TOWER_HEIGHT_MIN = 1.12;
export const TOWER_HEIGHT_SPAN = 0.62;
export function towerGeometry(skin: RuinSkin, height: number, lean: number): THREE.BufferGeometry {
  const detail = 1;
  const radial = Math.max(6, Math.round(14 * detail));
  const parts: ColoredPart[] = [];

  const plinth = new THREE.BoxGeometry(CELL - BASE_INSET * 2, 0.26, CELL - BASE_INSET * 2);
  plinth.translate(0, 0.13, 0);
  parts.push({ geometry: plinth, color: skin.stoneDark });

  const drumH = (height - 0.26) / 3;
  for (let i = 0; i < 3; i++) {
    const r0 = 0.44 - i * 0.045;
    const r1 = 0.47 - i * 0.045;
    const drum = new THREE.CylinderGeometry(r0, r1, drumH, radial, 1);
    drum.translate(0, drumH * 0.5, 0);
    if (i === 2) {
      drum.scale(1, 0.62, 1);
      drum.rotateZ(lean);
    }
    drum.translate(0, 0.26 + i * drumH, 0);
    parts.push({ geometry: drum, color: i === 1 ? skin.stoneLight : skin.stoneDark });
  }

  const band = new THREE.CylinderGeometry(0.5, 0.5, 0.075, radial, 1);
  band.translate(0, 0.26 + drumH, 0);
  parts.push({ geometry: band, color: skin.stoneLight });

  return mergeColoredParts(parts, 'ruinTower');
}

export function archGeometry(skin: RuinSkin): THREE.BufferGeometry {
  const detail = 1;
  const parts: ColoredPart[] = [];
  const JAMB_H = 0.7;
  const HALF = 0.47;

  for (const sz of [-1, 1]) {
    const jamb = new THREE.BoxGeometry(0.3, JAMB_H, 0.16);
    jamb.translate(0, JAMB_H * 0.5, sz * HALF);
    parts.push({ geometry: jamb, color: skin.stoneDark });
    const corbel = new THREE.BoxGeometry(0.34, 0.1, 0.24);
    corbel.translate(0, JAMB_H + 0.05, sz * (HALF - 0.02));
    parts.push({ geometry: corbel, color: skin.stoneLight });
  }

  const spans: readonly (readonly [number, number, number])[] = [
    [-0.3, 0.855, 0],
    [0.0, 0.9, 0.16],
    [0.3, 0.855, 0],
  ];
  for (let i = 0; i < spans.length; i++) {
    const s = spans[i] as readonly [number, number, number];
    const voussoir = new THREE.BoxGeometry(0.2, 0.15, 0.34);
    voussoir.rotateX(s[2]);
    voussoir.translate(0, s[1], s[0]);
    parts.push({ geometry: voussoir, color: i === 1 ? skin.stoneLight : skin.stoneDark });
  }

  const cushion = new THREE.IcosahedronGeometry(0.16, 1);
  cushion.scale(1.3, 0.4, 1.1);
  cushion.translate(0.02, 0.985, -0.14);
  parts.push({ geometry: cushion, color: skin.moss });

  if (detail > 0.5) {
    const fallen = new THREE.BoxGeometry(0.22, 0.13, 0.3);
    fallen.rotateY(0.5);
    fallen.rotateZ(0.24);
    fallen.translate(0.3, 0.07, -0.2);
    parts.push({ geometry: fallen, color: skin.stoneDark });
  }

  return mergeColoredParts(parts, 'ruinArch');
}

const BIRD_COUNT = 7;

export interface BirdsHandle {
  mesh: THREE.InstancedMesh;
  setTime(t: number): void;
  dispose(): void;
}

export function makeBirds(seed: number, color: THREE.ColorRepresentation): BirdsHandle {
  const count = BIRD_COUNT;
  const rng = streamFor(seed, 'ruin-birds');
  const geometry = birdGeometry();
  const material = oakMaterial({
    color,
    roughness: 1,
    flatShading: true,
    side: THREE.DoubleSide,
    toonSteps: 2,
    toonFloor: 0.5,
    name: 'ruinBirds',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.name = 'ruinBirds';
  mesh.frustumCulled = false;
  mesh.castShadow = false;
  mesh.receiveShadow = false;

  const radii = new Float32Array(count);
  const heights = new Float32Array(count);
  const speeds = new Float32Array(count);
  const phases = new Float32Array(count);
  const sizes = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    radii[i] = 15 + rng() * 11;
    heights[i] = 8.5 + rng() * 5.5;
    speeds[i] = (0.055 + rng() * 0.05) * (rng() < 0.5 ? -1 : 1);
    phases[i] = rng() * Math.PI * 2;
    sizes[i] = 0.5 + rng() * 0.32;
  }

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);

  const setTime = (t: number): void => {
    for (let i = 0; i < count; i++) {
      const a = t * (speeds[i] ?? 0) + (phases[i] ?? 0);
      const r = radii[i] ?? 0;
      const s = sizes[i] ?? 1;
      position.set(Math.cos(a) * r, (heights[i] ?? 0) + Math.sin(t * 0.4 + a) * 0.5, Math.sin(a) * r);
      euler.set(0, -a + ((speeds[i] ?? 0) > 0 ? 0 : Math.PI), Math.sin(t * 1.6 + a * 3) * 0.3);
      quaternion.setFromEuler(euler);
      const flap = 0.7 + 0.4 * Math.abs(Math.sin(t * 3.1 + (phases[i] ?? 0)));
      scale.set(s, s * flap, s);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  };
  setTime(0);

  return {
    mesh,
    setTime,
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}

function birdGeometry(): THREE.BufferGeometry {
  const v: number[] = [];
  for (const side of [-1, 1]) {
    v.push(0, 0, 0.09, side * 0.34, 0.09, -0.04, side * 0.12, 0.0, -0.11);
  }
  const positions = new Float32Array(v);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'ruinBird';
  return g;
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  courses: CoursesHandle;
  towerRoots: THREE.Object3D[];
  archRoots: THREE.Object3D[];
  arches: ArchCell[];
  setGoalPosition(x: number, y: number, z: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.noon;
  const skin = ruinSkin(palette, ctx.tokens);
  const seed = ctx.seed;
  const w = makeWorld('overgrownRuins');

  const sky = skyFromPalette(palette, 'noon', {
    sunDir: SUN_DIR,
    sunSize: 0.032,
    sunHdr: 5.2,
    sunGlow: 0.24,
    sunGlowPower: 30,
    topSpread: 0.7,
    bottomSpread: 0.52,
    radius: 340,
  });
  w.add(sky.mesh);
  w.own(sky);
  w.time({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 3,
    pointDistance: 7,
    sunDistanceScale: 2.2,
  });
  w.add(rig.group);
  w.own(rig);
  rig.setShadowFocus({ x: 0, y: 0.4, z: 0 }, SHADOW_RADIUS);

  applyFog(scene, palette, FOG_EXTENT);

  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    seed,
    heightFn: makeForestFloorHeight(seed),
    jitter: 0,
    colorNoise: 0.06,
    slopeColor: skin.soil,
    slopeStrength: 0.55,
    slopeSharpness: 2.1,
    flattenMask: { x: 0, z: 0, halfX: RUIN_HALF, halfZ: RUIN_HALF, feather: 2.4 },
    flattenHeight: RUIN_SINK,
    colorRamp: [
      { t: 0, color: blendTokens(skin.soil, palette.groundA, 0.4) },
      { t: 0.45, color: shade(palette.groundA, 0.14) },
      { t: 1, color: shade(blendTokens(palette.groundA, palette.groundB, 0.42), 0.1) },
    ],
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.16,
      toonSteps: 3,
      toonSoftness: 0.3,
      toonFloor: 0.24,
    },
    receiveShadow: true,
    castShadow: false,
    name: 'ruinForestFloor',
  });
  w.add(terrain.mesh);
  w.own(terrain);

  const white = palette.sun;
  const field = buildCellField(
    grid,
    {
      floorColor: white,
      wallColor: white,
      floorUnderWalls: true,
      floorTint: (x, y) => floorTintFor(grid, skin, x, y, seed),
      wallTint: (x, y) => wallTintFor(grid, skin, x, y, seed),
      wallShape: 'block',
      wallHeight: BASE_COURSE,
      wallInset: BASE_INSET,
      floorInset: 0.055,
      floorThickness: 0.14,
      jitter: 0,
      rimColor: palette.skyHorizon,
      rimStrength: 0.2,
      seed,
      name: 'ruinCells',
    },
  );
  const wallUniforms = oakUniforms(field.wallMaterial);
  if (wallUniforms) wallUniforms.uniforms.uToonFloor.value = 0.42;
  const floorUniforms = oakUniforms(field.floorMaterial);
  if (floorUniforms) floorUniforms.uniforms.uToonFloor.value = 0.4;
  w.add(field.group);
  w.own(field);
  w.time({ setTime: (t) => field.setTime(t) });

  const corners = cornerCells(grid);
  const cornerKeys = new Set<number>();
  for (const c of corners) cornerKeys.add(c.y * grid.w + c.x);

  const courses = buildCourses(grid, seed, skin, palette, (x, y) => cornerKeys.has(y * grid.w + x));
  w.add(courses.group);
  w.time(w.own(courses));

  const towerRoots: THREE.Object3D[] = [];
  {
    const world: Vec3Tuple = [0, 0, 0];
    for (let i = 0; i < corners.length; i++) {
      const c = corners[i] as GridCell;
      collapseWallInstance(field, c.x, c.y);
      const root = new THREE.Object3D();
      root.name = `ruinTowerRoot${i}`;
      cellToWorld(c.x, c.y, grid, world);
      root.position.set(world[0], 0, world[2]);
      const r = rngAt(c.x, c.y, seed ^ 0x70e2)();
      const r2 = rngAt(c.y, c.x, seed ^ 0x70e2)();
      const geometry = towerGeometry(skin, TOWER_HEIGHT_MIN + TOWER_HEIGHT_SPAN * r, (r2 - 0.5) * 0.32);
      const material = oakMaterial({
        vertexColors: true,
        flatShading: true,
        roughness: 0.95,
        rimColor: palette.skyHorizon,
        rimStrength: 0.3,
        toonSteps: 3,
        toonSoftness: 0.28,
        toonFloor: 0.42,
        name: `ruinTowerMaterial${i}`,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `ruinTower${i}`;
      mesh.rotation.y = r * Math.PI * 0.5;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      root.add(mesh);
      w.add(root);
      towerRoots.push(root);
      w.own({
        dispose: () => {
          geometry.dispose();
          material.dispose();
        },
      });
    }
  }

  const allArches = archCells(grid);
  const arches = allArches.slice(0, 9);
  const archRoots: THREE.Object3D[] = [];
  if (arches.length > 0) {
    const archGeo = archGeometry(skin);
    const archMaterial = oakMaterial({
      vertexColors: true,
      flatShading: true,
      roughness: 0.94,
      rimColor: palette.skyHorizon,
      rimStrength: 0.32,
      toonSteps: 3,
      toonSoftness: 0.28,
      toonFloor: 0.42,
      name: 'ruinArchMaterial',
    });
    w.own({
      dispose: () => {
        archGeo.dispose();
        archMaterial.dispose();
      },
    });
    const world: Vec3Tuple = [0, 0, 0];
    for (let i = 0; i < arches.length; i++) {
      const a = arches[i] as ArchCell;
      const root = new THREE.Object3D();
      root.name = `ruinArchRoot${i}`;
      cellToWorld(a.cell.x, a.cell.y, grid, world);
      root.position.set(world[0], 0, world[2]);
      const mesh = new THREE.Mesh(archGeo, archMaterial);
      mesh.name = `ruinArch${i}`;
      mesh.rotation.y = a.eastWest ? 0 : Math.PI / 2;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      root.add(mesh);
      w.add(root);
      archRoots.push(root);
    }
  }

  const clearCenters = excludeCellCenters(grid, 0.3);
  const offWalls = excludeAny(excludeWallCells(grid), clearCenters);

  const tufts = scatterGrass({
    count: 620,
    area: { x: 0, z: 0, halfX: RUIN_HALF - 0.4, halfZ: RUIN_HALF - 0.4 },
    seed: seed ^ 0x6a11,
    exclude: offWalls,
    color: skin.grass,
    scale: [0.26, 0.55],
    stretch: [0.75, 1.5],
    hueJitter: 0.26,
    wind: 0.05,
    windFrequency: 1.6,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    name: 'ruinGrassTufts',
  });
  w.add(tufts.group);
  w.time(w.own(tufts));

  const wildflowers = scatterFlowers({
    count: 130,
    area: { x: 0, z: 0, halfX: RUIN_HALF - 0.5, halfZ: RUIN_HALF - 0.5 },
    seed: seed ^ 0x2f83,
    exclude: offWalls,
    color: skin.bloom,
    scale: [0.3, 0.6],
    stretch: [0.8, 1.2],
    hueJitter: 0.26,
    wind: 0.035,
    windFrequency: 1.4,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    name: 'ruinWildflowers',
  });
  w.add(wildflowers.group);
  w.time(w.own(wildflowers));

  const rubble = scatterRocks({
    count: 340,
    area: { x: 0, z: 0, halfX: RUIN_HALF - 0.2, halfZ: RUIN_HALF - 0.2 },
    seed: seed ^ 0x51c7,
    exclude: clearCenters,
    color: skin.stoneDark,
    scale: [0.1, 0.26],
    stretch: [0.22, 0.5],
    hueJitter: 0.3,
    rimColor: palette.skyHorizon,
    rimStrength: 0.24,
    castShadow: false,
    name: 'ruinRubble',
  });
  w.add(rubble.group);
  w.time(w.own(rubble));

  const offRuin = excludeRect(0, 0, RUIN_HALF, RUIN_HALF, 1.4);
  const nearCamera = excludeRect(
    Math.sin(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    Math.cos(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    4.6,
    4.6,
    0,
  );
  const forest = scatterTrees({
    archetypes: ['broadleaf', 'conifer'],
    count: 74,
    area: { x: 0, z: 0, halfX: 18, halfZ: 18, innerRadius: RUIN_HALF + 1.6 },
    seed: seed ^ 0x1a70,
    exclude: excludeAny(offRuin, nearCamera),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.72,
    color: skin.canopy,
    scale: [0.85, 1.55],
    stretch: [0.95, 1.5],
    hueJitter: 0.26,
    wind: 0.03,
    windFrequency: 0.8,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    castShadow: true,
    name: 'ruinForest',
  });
  w.add(forest.group);
  w.time(w.own(forest));

  const saplings = scatterTrees({
    archetypes: ['broadleaf'],
    count: 26,
    area: { x: 0, z: 0, halfX: RUIN_HALF + 2.2, halfZ: RUIN_HALF + 2.2, innerRadius: RUIN_HALF + 0.35 },
    seed: seed ^ 0x3c19,
    exclude: excludeAny(offRuin, nearCamera),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 1,
    color: skin.mossDeep,
    scale: [0.38, 0.66],
    stretch: [0.9, 1.3],
    hueJitter: 0.3,
    wind: 0.05,
    windFrequency: 1.05,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    castShadow: false,
    name: 'ruinSaplings',
  });
  w.add(saplings.group);
  w.time(w.own(saplings));

  const haze = makeAmbientParticles({
    preset: 'dust',
    count: 420,
    bounds: { x: 0, y: 1.15, z: 0, halfX: RUIN_HALF, halfY: 1.05, halfZ: RUIN_HALF },
    seed: seed ^ 0x7d21,
    color: blendTokens(skin.sunWarm, palette.hemiSky, 0.3),
    size: 0.9,
    intensity: 1.05,
    opacity: 0.4,
    rise: 0.03,
    swirl: 0.2,
    name: 'ruinDustHaze',
  });
  w.add(haze.points);
  w.size(w.time(w.own(haze)));

  const shaftDust = makeAmbientParticles({
    preset: 'dust',
    count: 150,
    bounds: { x: 0, y: 1.5, z: 0, halfX: 0.6, halfY: 1.45, halfZ: 0.6 },
    seed: seed ^ 0x2ba9,
    color: skin.sunWarm,
    size: 1.05,
    intensity: 0.9,
    opacity: 0.5,
    rise: 0.05,
    swirl: 0.1,
    name: 'ruinShaftDust',
  });
  w.add(shaftDust.points);
  w.size(w.time(w.own(shaftDust)));

  const pollen = makeAmbientParticles({
    preset: 'pollen',
    count: 300,
    bounds: { x: 0, y: 2.2, z: 0, halfX: 15, halfY: 2, halfZ: 15 },
    seed: seed ^ 0x11c4,
    color: blendTokens(palette.groundB, palette.sun, 0.35),
    size: 0.75,
    intensity: 1.35,
    opacity: 0.4,
    rise: 0.07,
    swirl: 0.26,
    name: 'ruinPollen',
  });
  w.add(pollen.points);
  w.size(w.time(w.own(pollen)));

  const birds = makeBirds(seed, skin.bird);
  w.add(birds.mesh);
  w.time(w.own(birds));

  const shaftDustPoints = shaftDust.points;

  return w.seal({
    sky,
    rig,
    terrain,
    field,
    courses,
    towerRoots,
    archRoots,
    arches,
    setGoalPosition(x, y, z) {
      shaftDustPoints.position.set(x, y, z);
    },
  }, {
    drop: () => {
      towerRoots.length = 0;
      archRoots.length = 0;
    },
  });
}
