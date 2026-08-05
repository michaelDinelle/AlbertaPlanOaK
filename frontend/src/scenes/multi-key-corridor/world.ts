
import * as THREE from 'three';
import {
  CELL,
  buildCellField,
  cellToWorld,
  isWall,
  type CellFieldHandle,
  type GridSize,
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { emissiveMaterial, oakMaterial, oakUniforms, type EmissiveMaterial, type OakMaterial } from '@kit/materials';
import { makeAmbientParticles, makeMistPlanes, type AmbientParticlesHandle } from '@kit/particles';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import {
  excludeAny,
  excludeRect,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
  type VegetationHandle,
} from '@kit/vegetation';
import {
  MKC_DOORS,
  MKC_GOAL,
  MKC_H,
  MKC_KEYS,
  MKC_ROOM_BASE,
  MKC_W,
  mkcRoomCells,
  mkcWalls,
} from '@sim/envs/multikeycorridor';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';


export const TEMPLE_HALF_X = (MKC_W * CELL) / 2;
export const TEMPLE_HALF_Z = (MKC_H * CELL) / 2;
export const TERRAIN_SIZE = 46;
export const TERRAIN_SEGMENTS = 128;
export const WALL_HEIGHT = 0.7;
export const COURT_SINK = -0.07;
export const FOG_EXTENT = TERRAIN_SIZE * 11;
export const SUN_DIR: readonly [number, number, number] = [0.56, 0.82, 0.25];
export const SHADOW_RADIUS = 13.5;
export const CAMERA_AZIMUTH = 0.14;
export const CAMERA_STANDOFF = 13.5;

export const CHAMBERS = 4;
export const CHAMBER_NAMES: readonly string[] = ['dawnwell', 'grove', 'ochre', 'sanctum'];


export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function blendTokens(
  a: THREE.ColorRepresentation,
  b: THREE.ColorRepresentation,
  k: number,
): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), k);
}

export function shade(color: THREE.ColorRepresentation, k: number): THREE.Color {
  return blendTokens(color, TOKENS.palettes.night.skyTop, k);
}

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

export function mergeColoredParts(parts: readonly ColoredPart[], name: string): THREE.BufferGeometry {
  let total = 0;
  for (const p of parts) {
    total += p.geometry.index ? p.geometry.index.count : p.geometry.getAttribute('position').count;
  }
  const positions = new Float32Array(total * 3);
  const normals = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);
  const c = new THREE.Color();
  let o = 0;
  for (const part of parts) {
    const geo = part.geometry;
    const pos = geo.getAttribute('position');
    const nor = geo.getAttribute('normal');
    const index = geo.index;
    const n = index ? index.count : pos.count;
    c.set(part.color);
    for (let i = 0; i < n; i++) {
      const vi = index ? index.getX(i) : i;
      positions[o * 3] = pos.getX(vi);
      positions[o * 3 + 1] = pos.getY(vi);
      positions[o * 3 + 2] = pos.getZ(vi);
      if (nor) {
        normals[o * 3] = nor.getX(vi);
        normals[o * 3 + 1] = nor.getY(vi);
        normals[o * 3 + 2] = nor.getZ(vi);
      }
      colors[o * 3] = c.r;
      colors[o * 3 + 1] = c.g;
      colors[o * 3 + 2] = c.b;
      o++;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeBoundingSphere();
  g.name = name;
  for (const p of parts) p.geometry.dispose();
  return g;
}


export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === MKC_W && g.h === MKC_H) return g;
  return {
    w: MKC_W,
    h: MKC_H,
    walls: mkcWalls(),
    goal: { ...MKC_GOAL },
    doors: MKC_DOORS.map((c) => ({ ...c })),
    keys: MKC_KEYS.map((c) => ({ ...c })),
    hallways: MKC_DOORS.map((c) => ({ ...c })),
  };
}

export function resolveDoors(grid: GridGeometry): readonly GridCell[] {
  const d = grid.doors && grid.doors.length > 0 ? grid.doors : grid.hallways;
  return d && d.length > 0 ? d : MKC_DOORS;
}

export function resolveKeys(grid: GridGeometry): readonly GridCell[] {
  const k = grid.keys;
  return k && k.length > 0 ? k : MKC_KEYS;
}

export function resolveGoal(grid: GridGeometry): GridCell {
  return grid.goal ?? MKC_GOAL;
}

const CHAMBER_LOOKUP: Int8Array = (() => {
  const table = new Int8Array(MKC_W * MKC_H).fill(-1);
  for (let r = 0; r < CHAMBERS; r++) {
    for (const cell of mkcRoomCells(r)) table[cell.y * MKC_W + cell.x] = r;
  }
  return table;
})();

export function chamberOf(x: number, y: number): number {
  if (x < 0 || y < 0 || x >= MKC_W || y >= MKC_H) return -1;
  return CHAMBER_LOOKUP[y * MKC_W + x] ?? -1;
}

export function processionT(x: number): number {
  return clamp01((x - 1) / (MKC_W - 3));
}

export interface ChamberBounds {
  cx: number;
  cz: number;
  halfX: number;
  halfZ: number;
  cells: number;
}

export function chamberBounds(grid: GridGeometry, chamber: number): ChamberBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  let cells = 0;
  const w: Vec3Tuple = [0, 0, 0];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isWall(grid, x, y)) continue;
      if (chamberOf(x, y) !== chamber) continue;
      cellToWorld(x, y, grid, w);
      if (w[0] < minX) minX = w[0];
      if (w[0] > maxX) maxX = w[0];
      if (w[2] < minZ) minZ = w[2];
      if (w[2] > maxZ) maxZ = w[2];
      cells++;
    }
  }
  if (cells === 0) return { cx: 0, cz: 0, halfX: 0, halfZ: 0, cells: 0 };
  const half = CELL * 0.5;
  return {
    cx: (minX + maxX) * 0.5,
    cz: (minZ + maxZ) * 0.5,
    halfX: (maxX - minX) * 0.5 + half,
    halfZ: (maxZ - minZ) * 0.5 + half,
    cells,
  };
}

export function unionBounds(all: readonly ChamberBounds[]): ChamberBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  let cells = 0;
  for (const b of all) {
    if (b.cells === 0) continue;
    minX = Math.min(minX, b.cx - b.halfX);
    maxX = Math.max(maxX, b.cx + b.halfX);
    minZ = Math.min(minZ, b.cz - b.halfZ);
    maxZ = Math.max(maxZ, b.cz + b.halfZ);
    cells += b.cells;
  }
  if (cells === 0) return { cx: 0, cz: 0, halfX: 0, halfZ: 0, cells: 0 };
  return {
    cx: (minX + maxX) * 0.5,
    cz: (minZ + maxZ) * 0.5,
    halfX: (maxX - minX) * 0.5,
    halfZ: (maxZ - minZ) * 0.5,
    cells,
  };
}

export function gateFacesX(grid: GridGeometry, cell: GridCell): boolean {
  return isWall(grid, cell.x, cell.y - 1) && isWall(grid, cell.x, cell.y + 1);
}


export interface ChamberSkin {
  floor: THREE.Color;
  wall: THREE.Color;
  litter: THREE.Color;
  bloom: THREE.Color;
  motes: THREE.Color;
}

export interface TempleSkin {
  chambers: readonly ChamberSkin[];
  threshold: THREE.Color;
  stoneLight: THREE.Color;
  stoneDark: THREE.Color;
  soil: THREE.Color;
  riverBed: THREE.Color;
  bramble: THREE.Color;
  brambleLeaf: THREE.Color;
  keys: readonly THREE.Color[];
  relic: THREE.Color;
  flame: THREE.Color;
}

export function templeSkin(palette: Palette, tokens: typeof TOKENS): TempleSkin {
  const gold = tokens.accents['multi-key-corridor'] ?? tokens.celebrationGold;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const garden = tokens.accents['four-rooms'] ?? palette.groundA;
  const terracotta = tokens.accents['high-dim-maze'] ?? tokens.celebrationGold;
  const moss = tokens.accents['procedural-maze'] ?? palette.groundA;
  const blossom = tokens.accents['chain'] ?? tokens.celebrationGold;
  const white = tokens.palettes.noon.sun;

  const chambers: ChamberSkin[] = [
    {
      floor: shade(blendTokens(palette.hemiSky, garden, 0.42), 0.64),
      litter: shade(blendTokens(moss, palette.hemiSky, 0.28), 0.42),
      bloom: shade(blendTokens(silver, white, 0.35), 0.3),
      motes: blendTokens(palette.hemiSky, white, 0.35),
      wall: shade(blendTokens(silver, palette.hemiGround, 0.55), 0.78),
    },
    {
      floor: shade(blendTokens(palette.groundA, garden, 0.34), 0.58),
      litter: shade(blendTokens(garden, palette.groundB, 0.34), 0.3),
      bloom: shade(blendTokens(blossom, white, 0.25), 0.2),
      motes: blendTokens(palette.groundB, white, 0.4),
      wall: shade(blendTokens(silver, palette.hemiGround, 0.5), 0.77),
    },
    {
      floor: shade(blendTokens(palette.groundB, terracotta, 0.42), 0.64),
      litter: shade(blendTokens(terracotta, gold, 0.42), 0.42),
      bloom: shade(blendTokens(gold, terracotta, 0.3), 0.24),
      motes: blendTokens(gold, white, 0.3),
      wall: shade(blendTokens(silver, terracotta, 0.2), 0.76),
    },
    {
      floor: shade(blendTokens(gold, tokens.celebrationGold, 0.4), 0.56),
      litter: shade(blendTokens(gold, white, 0.28), 0.38),
      bloom: shade(blendTokens(tokens.celebrationGold, white, 0.2), 0.18),
      motes: blendTokens(tokens.celebrationGold, white, 0.25),
      wall: shade(blendTokens(silver, terracotta, 0.26), 0.74),
    },
  ];

  return {
    chambers,
    threshold: shade(blendTokens(palette.hemiGround, silver, 0.24), 0.68),
    stoneLight: shade(blendTokens(silver, palette.hemiGround, 0.32), 0.3),
    stoneDark: shade(blendTokens(palette.hemiGround, silver, 0.3), 0.5),
    soil: shade(blendTokens(palette.hemiGround, TOKENS.palettes.dusk.groundA, 0.4), 0.5),
    riverBed: shade(blendTokens(TOKENS.palettes.underwater.skyBottom, palette.hemiGround, 0.3), 0.4),
    bramble: shade(blendTokens(palette.groundA, TOKENS.palettes.dusk.groundA, 0.5), 0.55),
    brambleLeaf: shade(blendTokens(palette.groundB, garden, 0.45), 0.34),
    keys: [
      new THREE.Color(tokens.accents['key-corridor'] ?? tokens.celebrationGold),
      blendTokens(white, palette.hemiSky, 0.16),
      new THREE.Color(tokens.accents['mountain-car'] ?? tokens.celebrationGold),
    ],
    relic: blendTokens(tokens.celebrationGold, palette.sun, 0.24),
    flame: blendTokens(tokens.celebrationGold, terracotta, 0.24),
  };
}


export function excludeCellCenters(grid: GridSize, radius: number): ExcludeFn {
  const r2 = radius * radius;
  const ox = (grid.w - 1) / 2;
  const oz = (grid.h - 1) / 2;
  return (x, z) => {
    const cx = Math.round(x / CELL + ox);
    const cz = Math.round(z / CELL + oz);
    const dx = x - (cx - ox) * CELL;
    const dz = z - (cz - oz) * CELL;
    return dx * dx + dz * dz < r2;
  };
}

export function excludeWallCells(grid: GridGeometry): ExcludeFn {
  const ox = (grid.w - 1) / 2;
  const oz = (grid.h - 1) / 2;
  return (x, z) => isWall(grid, Math.round(x / CELL + ox), Math.round(z / CELL + oz));
}

export function excludeProcession(grid: GridGeometry, doors: readonly GridCell[], halfWidth: number): ExcludeFn {
  const row = doors[0]?.y ?? (grid.h - 1) / 2;
  const oz = (grid.h - 1) / 2;
  const zWorld = (row - oz) * CELL;
  return (_x, z) => Math.abs(z - zWorld) < halfWidth;
}


export function makeGardenHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.max(Math.abs(x) - TEMPLE_HALF_X, Math.abs(z) - TEMPLE_HALF_Z);
    const w = smoothstep01(out / 6);
    if (w <= 0) return 0;
    const n1 = valueNoise2(x * 0.08 + 3.1, z * 0.08 + 9.4, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.19 + 1.7, z * 0.19 + 5.2, 64, seed ^ 0x51ed270b) - 0.5;
    return w * (n1 * 2.6 + n2 * 0.9);
  };
}


export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  portalRoots: THREE.Object3D[];
  doors: readonly GridCell[];
  stoneMaterial: OakMaterial;
  setTime(t: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

interface Disposable {
  dispose(): void;
}

interface Timed {
  setTime(t: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const tier: QualityTier = ctx.tier;
  const palette: Palette = ctx.tokens.palettes.morning;
  const skin = templeSkin(palette, ctx.tokens);
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'templeOfThreeGates';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const doors = resolveDoors(grid);

  const sky = skyFromPalette(palette, 'morning', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.04,
    sunHdr: 5.2,
    sunGlow: 0.3,
    sunGlowPower: 26,
    topSpread: 0.68,
    bottomSpread: 0.46,
    radius: 340,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 7.5,
    sunDistanceScale: 2.1,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0.5, y: 0.35, z: 0 }, SHADOW_RADIUS);

  applyFog(scene, palette, FOG_EXTENT);

  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    tier,
    seed: ctx.seed,
    heightFn: makeGardenHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.05,
    slopeColor: skin.soil,
    slopeStrength: 0.5,
    slopeSharpness: 2.2,
    flattenMask: { x: 0, z: 0, halfX: TEMPLE_HALF_X, halfZ: TEMPLE_HALF_Z, feather: 2.8 },
    flattenHeight: COURT_SINK,
    colorRamp: [
      { t: 0, color: shade(blendTokens(palette.groundA, skin.soil, 0.45), 0.3) },
      { t: 0.45, color: shade(palette.groundA, 0.26) },
      { t: 1, color: shade(blendTokens(palette.groundB, palette.groundA, 0.45), 0.2) },
    ],
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.16,
      toonSteps: 3,
      toonSoftness: 0.3,
      toonFloor: 0.26,
    },
    receiveShadow: shadows,
    castShadow: false,
    name: 'templeGarden',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  const white = palette.sun;
  const field = buildCellField(
    grid,
    {
      floorColor: white,
      wallColor: white,
      floorUnderWalls: true,
      floorTint: (x, y) => {
        if (isWall(grid, x, y)) return skin.soil;
        const chamber = chamberOf(x, y);
        return chamber < 0 ? skin.threshold : (skin.chambers[chamber]?.floor ?? skin.threshold);
      },
      wallTint: (x, y) => {
        const chamber = chamberOf(x, y);
        if (chamber >= 0) return skin.chambers[chamber]?.wall ?? skin.stoneDark;
        const t = processionT(x);
        const cool = skin.chambers[0]?.wall ?? skin.stoneDark;
        const warm = skin.chambers[CHAMBERS - 1]?.wall ?? skin.stoneLight;
        return new THREE.Color().copy(cool).lerp(warm, t);
      },
      wallShape: 'block',
      wallHeight: WALL_HEIGHT,
      wallInset: 0.015,
      floorInset: 0.03,
      floorThickness: 0.16,
      jitter: 0.08,
      rimColor: palette.skyHorizon,
      rimStrength: 0.22,
      seed: ctx.seed,
      name: 'templeCells',
    },
    tier,
  );
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  const stoneMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.93,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.26,
    tier,
    name: 'templeStone',
  });
  owned.push({ dispose: () => stoneMaterial.dispose() });

  const portalGeometry = makePortalGeometry(skin, tier);
  owned.push({ dispose: () => portalGeometry.dispose() });
  const portalRoots: THREE.Object3D[] = [];
  {
    const portals = new THREE.InstancedMesh(portalGeometry, stoneMaterial, Math.max(1, doors.length));
    portals.name = 'templePortals';
    portals.castShadow = shadows;
    portals.receiveShadow = shadows;
    portals.count = 0;
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    const world: Vec3Tuple = [0, 0, 0];
    let n = 0;
    for (let i = 0; i < doors.length; i++) {
      const cell = doors[i];
      if (!cell) continue;
      const root = new THREE.Object3D();
      root.name = `portalRoot${i}`;
      cellToWorld(cell.x, cell.y, grid, world);
      root.position.set(world[0], 0, world[2]);
      group.add(root);
      portalRoots.push(root);

      position.set(world[0], 0, world[2]);
      euler.set(0, gateFacesX(grid, cell) ? 0 : Math.PI / 2, 0);
      quaternion.setFromEuler(euler);
      matrix.compose(position, quaternion, scale);
      portals.setMatrixAt(n, matrix);
      n++;
    }
    portals.count = n;
    portals.instanceMatrix.needsUpdate = true;
    portals.computeBoundingSphere();
    if (n > 0) group.add(portals);
    owned.push({ dispose: () => portals.dispose() });
  }

  const lanternGeometry = makeLanternGeometry(skin, tier);
  owned.push({ dispose: () => lanternGeometry.dispose() });
  const lanternSpots: { x: number; z: number }[] = [];
  {
    const world: Vec3Tuple = [0, 0, 0];
    const row = doors[0]?.y ?? 3;
    for (let c = 0; c < CHAMBERS; c++) {
      const base = MKC_ROOM_BASE[c] ?? 1;
      for (const dz of [-1, 1]) {
        cellToWorld(base + 1, row + dz * 1.5, grid, world);
        lanternSpots.push({ x: world[0], z: world[2] });
      }
    }
  }
  {
    const lanterns = new THREE.InstancedMesh(lanternGeometry, stoneMaterial, Math.max(1, lanternSpots.length));
    lanterns.name = 'templeLanterns';
    lanterns.castShadow = shadows;
    lanterns.receiveShadow = shadows;
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    for (let i = 0; i < lanternSpots.length; i++) {
      const spot = lanternSpots[i] as { x: number; z: number };
      position.set(spot.x, 0, spot.z);
      euler.set(0, ((i * 37) % 12) * 0.09, 0);
      quaternion.setFromEuler(euler);
      matrix.compose(position, quaternion, scale);
      lanterns.setMatrixAt(i, matrix);
    }
    lanterns.count = lanternSpots.length;
    lanterns.instanceMatrix.needsUpdate = true;
    lanterns.computeBoundingSphere();
    group.add(lanterns);
    owned.push({ dispose: () => lanterns.dispose() });
  }
  if (tier !== 'thumb') {
    const flameGeometry = new THREE.IcosahedronGeometry(0.055, 1);
    const flameMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, 1.35, { name: 'templeLanternFlame' });
    const flames = new THREE.InstancedMesh(flameGeometry, flameMaterial, Math.max(1, lanternSpots.length));
    flames.name = 'templeLanternFlames';
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3(1, 1.35, 1);
    for (let i = 0; i < lanternSpots.length; i++) {
      const spot = lanternSpots[i] as { x: number; z: number };
      position.set(spot.x, LANTERN_FLAME_Y, spot.z);
      matrix.compose(position, quaternion, scale);
      flames.setMatrixAt(i, matrix);
    }
    flames.count = lanternSpots.length;
    flames.instanceMatrix.needsUpdate = true;
    flames.computeBoundingSphere();
    group.add(flames);
    owned.push({
      dispose: () => {
        flames.dispose();
        flameGeometry.dispose();
        flameMaterial.dispose();
      },
    });
  }

  const noPlanting = excludeAny(
    excludeWallCells(grid),
    excludeCellCenters(grid, 0.34),
    excludeProcession(grid, doors, 0.34),
  );
  const bounds: ChamberBounds[] = [];
  for (let c = 0; c < CHAMBERS; c++) bounds.push(chamberBounds(grid, c));

  const planted: VegetationHandle[] = [];
  const plantSpecs: readonly {
    chamber: number;
    kind: 'grass' | 'flowers' | 'rocks';
    count: number;
    scale: readonly [number, number];
    stretch: readonly [number, number];
    wind: number;
    name: string;
  }[] = [
    { chamber: 0, kind: 'grass', count: 105, scale: [0.26, 0.52], stretch: [0.7, 1.2], wind: 0.03, name: 'dawnwellMoss' },
    { chamber: 0, kind: 'flowers', count: 26, scale: [0.32, 0.62], stretch: [0.8, 1.2], wind: 0.028, name: 'dawnwellDewflowers' },
    { chamber: 1, kind: 'grass', count: 130, scale: [0.32, 0.64], stretch: [0.8, 1.4], wind: 0.042, name: 'groveGrass' },
    { chamber: 1, kind: 'flowers', count: 38, scale: [0.38, 0.72], stretch: [0.85, 1.3], wind: 0.03, name: 'groveBlossoms' },
    { chamber: 2, kind: 'rocks', count: 52, scale: [0.11, 0.24], stretch: [0.24, 0.42], wind: 0, name: 'ochreGravel' },
    { chamber: 2, kind: 'grass', count: 58, scale: [0.24, 0.46], stretch: [0.7, 1.1], wind: 0.034, name: 'ochreTufts' },
    { chamber: 3, kind: 'flowers', count: 34, scale: [0.34, 0.64], stretch: [0.9, 1.25], wind: 0.026, name: 'sanctumGoldflowers' },
    { chamber: 3, kind: 'rocks', count: 26, scale: [0.09, 0.18], stretch: [0.22, 0.36], wind: 0, name: 'sanctumGoldDust' },
  ];
  const gardenBounds = unionBounds(bounds);
  const thumbSpecs: typeof plantSpecs = [
    { chamber: 1, kind: 'grass', count: 293, scale: [0.28, 0.6], stretch: [0.75, 1.3], wind: 0.036, name: 'templeGroundcover' },
    { chamber: 1, kind: 'flowers', count: 98, scale: [0.34, 0.68], stretch: [0.85, 1.25], wind: 0.028, name: 'templeBlossoms' },
    { chamber: 2, kind: 'rocks', count: 78, scale: [0.1, 0.22], stretch: [0.23, 0.4], wind: 0, name: 'templeGravel' },
  ];
  const liveSpecs = tier === 'thumb' ? thumbSpecs : plantSpecs;
  for (let i = 0; i < liveSpecs.length; i++) {
    const spec = liveSpecs[i] as (typeof liveSpecs)[number];
    const b = (tier === 'thumb' ? gardenBounds : bounds[spec.chamber]) as ChamberBounds;
    const chamber = skin.chambers[spec.chamber] as ChamberSkin;
    const color = spec.kind === 'flowers' ? chamber.bloom : chamber.litter;
    const options = {
      count: spec.count,
      area: { x: b.cx, z: b.cz, halfX: b.halfX, halfZ: b.halfZ },
      seed: (ctx.seed ^ 0x5b10) + i * 977,
      tier,
      exclude: noPlanting,
      color,
      scale: spec.scale,
      stretch: spec.stretch,
      hueJitter: 0.24,
      ...(spec.wind > 0 ? { wind: spec.wind, windFrequency: 1.5 } : {}),
      rimColor: palette.skyHorizon,
      rimStrength: 0.3,
      castShadow: false,
      name: spec.name,
    };
    const handle =
      spec.kind === 'grass'
        ? scatterGrass(options)
        : spec.kind === 'flowers'
          ? scatterFlowers(options)
          : scatterRocks(options);
    group.add(handle.group);
    owned.push(handle);
    timed.push(handle);
    planted.push(handle);
  }

  const offTemple = excludeRect(0, 0, TEMPLE_HALF_X, TEMPLE_HALF_Z, 1.4);
  const nearCamera = excludeRect(
    Math.sin(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    Math.cos(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    6,
    5,
    0,
  );
  const trees: VegetationHandle[] = [];
  const treeSpecs: readonly {
    archetype: 'broadleaf' | 'conifer' | 'bare';
    count: number;
    area: { x: number; z: number; halfX: number; halfZ: number };
    color: THREE.Color;
    name: string;
  }[] = [
    {
      archetype: 'conifer',
      count: 22,
      area: { x: -14.5, z: 0, halfX: 6, halfZ: 11 },
      color: shade(blendTokens(palette.groundA, palette.hemiSky, 0.22), 0.42),
      name: 'westConifers',
    },
    {
      archetype: 'broadleaf',
      count: 20,
      area: { x: 0, z: -13.5, halfX: 14, halfZ: 5 },
      color: shade(blendTokens(palette.groundA, palette.groundB, 0.35), 0.3),
      name: 'northBroadleaf',
    },
    {
      archetype: 'broadleaf',
      count: 18,
      area: { x: 15, z: 0, halfX: 6, halfZ: 10 },
      color: shade(blendTokens(palette.groundB, skin.chambers[3]?.litter ?? palette.groundB, 0.4), 0.26),
      name: 'sanctumBroadleaf',
    },
  ];
  for (let i = 0; i < treeSpecs.length; i++) {
    const spec = treeSpecs[i] as (typeof treeSpecs)[number];
    if (tier === 'thumb' && spec.archetype === 'conifer') continue;
    const handle = scatterTrees({
      archetypes: [spec.archetype],
      count: spec.count,
      area: spec.area,
      seed: (ctx.seed ^ 0x1a70) + i * 613,
      tier,
      exclude: excludeAny(offTemple, nearCamera),
      heightAt: (x, z) => terrain.sampleHeight(x, z),
      maxSlope: 0.7,
      color: spec.color,
      scale: [0.75, 1.35],
      stretch: [0.9, 1.4],
      hueJitter: 0.24,
      wind: 0.03,
      windFrequency: 0.85,
      rimColor: palette.skyHorizon,
      rimStrength: 0.3,
      castShadow: true,
      name: spec.name,
    });
    group.add(handle.group);
    owned.push(handle);
    timed.push(handle);
    trees.push(handle);
  }

  const motes: AmbientParticlesHandle[] = [];
  const moteSpecs: readonly {
    chamber: number;
    preset: 'pollen' | 'dust' | 'snow';
    count: number;
    size: number;
    rise: number;
    swirl: number;
    opacity: number;
    intensity: number;
    y: number;
    name: string;
  }[] = [
    { chamber: 0, preset: 'snow', count: 70, size: 0.9, rise: -0.05, swirl: 0.14, opacity: 0.34, intensity: 0.9, y: 0.85, name: 'dawnwellDew' },
    { chamber: 1, preset: 'pollen', count: 120, size: 0.8, rise: 0.07, swirl: 0.26, opacity: 0.4, intensity: 1.35, y: 1.05, name: 'grovePollen' },
    { chamber: 2, preset: 'dust', count: 120, size: 0.76, rise: 0.03, swirl: 0.2, opacity: 0.38, intensity: 1.2, y: 1.0, name: 'ochreDust' },
    { chamber: 3, preset: 'pollen', count: 140, size: 0.92, rise: 0.05, swirl: 0.18, opacity: 0.46, intensity: 1.7, y: 1.15, name: 'sanctumMotes' },
  ];
  for (const spec of moteSpecs) {
    const b = bounds[spec.chamber] as ChamberBounds;
    const handle = makeAmbientParticles({
      preset: spec.preset,
      count: spec.count,
      bounds: { x: b.cx, y: spec.y, z: b.cz, halfX: b.halfX, halfY: 0.95, halfZ: b.halfZ },
      seed: (ctx.seed ^ 0x7d21) + spec.chamber * 613,
      tier,
      color: skin.chambers[spec.chamber]?.motes ?? white,
      size: spec.size,
      intensity: spec.intensity,
      opacity: spec.opacity,
      rise: spec.rise,
      swirl: spec.swirl,
      name: spec.name,
    });
    group.add(handle.points);
    owned.push(handle);
    timed.push(handle);
    motes.push(handle);
  }

  const haze = makeMistPlanes({
    size: [TERRAIN_SIZE * 0.9, TERRAIN_SIZE * 0.9],
    color: blendTokens(palette.fog, palette.hemiSky, 0.3),
    seed: ctx.seed ^ 0x3c91,
    tier,
    layers: 3,
    baseY: 1.1,
    spread: 1.6,
    opacity: 0.05,
    speed: 0.008,
    scale: 3.2,
    nearFade: 8,
    name: 'gardenHaze',
  });
  group.add(haze.group);
  owned.push(haze);
  timed.push(haze);

  return {
    group,
    sky,
    rig,
    terrain,
    field,
    portalRoots,
    doors,
    stoneMaterial,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
      oakUniforms(stoneMaterial)?.setTime(t);
    },
    setPixelRatio(dpr) {
      for (let i = 0; i < motes.length; i++) motes[i]?.setPixelRatio(dpr);
    },
    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      motes.length = 0;
      planted.length = 0;
      trees.length = 0;
      lanternSpots.length = 0;
      portalRoots.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}


export const LANTERN_FLAME_Y = 0.575;
export const PORTAL_LINTEL_Y = 1.42;

export function makePortalGeometry(skin: TempleSkin, tier: QualityTier): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const PIER_H = PORTAL_LINTEL_Y;
  const PIER_Z = 0.95;

  for (const sz of [-1, 1]) {
    const pier = new THREE.BoxGeometry(0.46, PIER_H, 0.42);
    pier.translate(0, PIER_H * 0.5, sz * PIER_Z);
    parts.push({ geometry: pier, color: skin.stoneLight });
    const plinth = new THREE.BoxGeometry(0.58, 0.14, 0.54);
    plinth.translate(0, 0.07, sz * PIER_Z);
    parts.push({ geometry: plinth, color: skin.stoneDark });
    const capital = new THREE.BoxGeometry(0.56, 0.13, 0.5);
    capital.translate(0, PIER_H - 0.065, sz * PIER_Z);
    parts.push({ geometry: capital, color: skin.stoneDark });
    const finial = new THREE.OctahedronGeometry(0.095, tier === 'thumb' ? 0 : 1);
    finial.scale(1, 1.7, 1);
    finial.translate(0, PIER_H + 0.16, sz * PIER_Z);
    parts.push({ geometry: finial, color: skin.stoneLight });
  }

  const lintel = new THREE.BoxGeometry(0.3, 0.16, 2.16);
  lintel.translate(0, PIER_H + 0.08, 0);
  parts.push({ geometry: lintel, color: skin.stoneLight });
  const band = new THREE.BoxGeometry(0.34, 0.05, 1.7);
  band.translate(0, PIER_H + 0.005, 0);
  parts.push({ geometry: band, color: skin.stoneDark });


  return mergeColoredParts(parts, 'templePortal');
}

export function makeLanternGeometry(skin: TempleSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = tier === 'thumb' ? 0.45 : tier === 'medium' ? 0.7 : 1;
  const radial = Math.max(5, Math.round(8 * detail));
  const parts: ColoredPart[] = [];

  const base = new THREE.CylinderGeometry(0.12, 0.16, 0.08, radial, 1);
  base.translate(0, 0.04, 0);
  parts.push({ geometry: base, color: skin.stoneDark });

  const shaft = new THREE.CylinderGeometry(0.05, 0.062, 0.36, radial, 1);
  shaft.translate(0, 0.26, 0);
  parts.push({ geometry: shaft, color: skin.stoneLight });

  const bowl = new THREE.CylinderGeometry(0.125, 0.082, 0.085, radial, 1);
  bowl.translate(0, 0.485, 0);
  parts.push({ geometry: bowl, color: skin.stoneDark });

  const cap = new THREE.ConeGeometry(0.155, 0.13, radial, 1);
  cap.translate(0, 0.69, 0);
  parts.push({ geometry: cap, color: skin.stoneLight });

  const finial = new THREE.OctahedronGeometry(0.042, 0);
  finial.translate(0, 0.78, 0);
  parts.push({ geometry: finial, color: skin.stoneDark });

  return mergeColoredParts(parts, 'templeLantern');
}
