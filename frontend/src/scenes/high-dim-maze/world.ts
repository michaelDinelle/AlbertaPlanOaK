
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
import { oakMaterial, type OakMaterial } from '@kit/materials';
import { makeAmbientParticles, makeMistPlanes, type AmbientParticlesHandle } from '@kit/particles';
import { mulberry32, hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import {
  excludeAny,
  excludeRect,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
} from '@kit/vegetation';
import {
  HIGHDIMMAZE_GAPS,
  HIGHDIMMAZE_H,
  HIGHDIMMAZE_W,
  highDimMazeWalls,
} from '@sim/envs/highdimmaze';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';


export const CANYON_HALF = (HIGHDIMMAZE_W * CELL) / 2;
export const TERRAIN_SIZE = 76;
export const TERRAIN_SEGMENTS = 120;
export const CANYON_SINK = -0.1;

export const FIN_HEIGHT_NORTH = 1.28;
export const FIN_HEIGHT_SOUTH = 1.0;
export const RIM_HEIGHT_NORTH = 1.24;
export const RIM_HEIGHT_SOUTH = 0.54;
export const WALL_BASE_HEIGHT = FIN_HEIGHT_NORTH;

export const ARCH_NOTCH: readonly number[] = Object.freeze([0.36, 0.74]);

export const ARCH_RISE = 0.14;
export const ARCH_OPENING = 1.25;
export const ARCH_BAND = 0.34;
export const ARCH_SPAN_HALF = 0.5;
export const ARCH_RIBBON = 0.36;
export const ARCH_LEG = 0.24;

export const FOG_EXTENT = 700;

export const SUN_DIR: readonly [number, number, number] = [-0.5, 0.8, 0.33];
export const SHADOW_RADIUS = 12.5;

export const CAMERA_AZIMUTH = 0.25;
export const CAMERA_STANDOFF = 15;


export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
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
  if (g && g.w === HIGHDIMMAZE_W && g.h === HIGHDIMMAZE_H) return g;
  return {
    w: HIGHDIMMAZE_W,
    h: HIGHDIMMAZE_H,
    walls: highDimMazeWalls(),
    hallways: HIGHDIMMAZE_GAPS.map((c) => ({ ...c })),
  };
}

export function resolveGaps(grid: GridGeometry): readonly GridCell[] {
  const h = grid.hallways;
  return h && h.length > 0 ? h : HIGHDIMMAZE_GAPS;
}

export function isRimCell(grid: GridSize, x: number, y: number): boolean {
  return x === 0 || y === 0 || x === grid.w - 1 || y === grid.h - 1;
}

export function wallNominalAt(grid: GridGeometry, x: number, y: number): number {
  if (!isWall(grid, x, y)) return 0;
  const t = grid.h > 1 ? clamp01(y / (grid.h - 1)) : 0;
  return isRimCell(grid, x, y)
    ? lerp(RIM_HEIGHT_NORTH, RIM_HEIGHT_SOUTH, t)
    : lerp(FIN_HEIGHT_NORTH, FIN_HEIGHT_SOUTH, t);
}

export function archNotch(grid: GridGeometry, x: number, y: number): number {
  if (!isWall(grid, x, y) || isRimCell(grid, x, y)) return 1;
  let k = 1;
  const gaps = resolveGaps(grid);
  for (let i = 0; i < gaps.length; i++) {
    const gap = gaps[i];
    if (!gap) continue;
    const alongZ = archAxis(grid, gap) === 'alongZ';
    const along = alongZ ? Math.abs(y - gap.y) : Math.abs(x - gap.x);
    const across = alongZ ? Math.abs(x - gap.x) : Math.abs(y - gap.y);
    if (across !== 0 || along < 1) continue;
    const f = ARCH_NOTCH[along - 1];
    if (f !== undefined && f < k) k = f;
  }
  return k;
}

export function wallHeightAt(grid: GridGeometry, x: number, y: number): number {
  return wallNominalAt(grid, x, y) * archNotch(grid, x, y);
}

export function wallHeightJitter(x: number, y: number, seed: number, amount = 0.09): number {
  return 1 + (hash2f(x, y, seed ^ 0x51ed270b) - 0.5) * 2 * amount;
}

export function wallDistance(grid: GridGeometry, x: number, y: number, cap = 3): number {
  if (isWall(grid, x, y)) return 0;
  for (let r = 1; r <= cap; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        if (isWall(grid, x + dx, y + dy)) return r;
      }
    }
  }
  return cap + 1;
}

export type ArchAxis = 'alongX' | 'alongZ';

export function archAxis(grid: GridGeometry, gap: GridCell): ArchAxis {
  const runsAlongZ = isWall(grid, gap.x, gap.y - 1) && isWall(grid, gap.x, gap.y + 1);
  return runsAlongZ ? 'alongZ' : 'alongX';
}

export function archYaw(axis: ArchAxis): number {
  return axis === 'alongZ' ? Math.PI / 2 : 0;
}

export function archJambs(grid: GridGeometry, gap: GridCell, heights?: Float32Array): [number, number] {
  const at = (x: number, y: number): number => {
    if (heights && x >= 0 && y >= 0 && x < grid.w && y < grid.h) {
      const h = heights[y * grid.w + x] ?? 0;
      if (h > 0) return h;
    }
    return wallHeightAt(grid, x, y);
  };
  return archAxis(grid, gap) === 'alongZ'
    ? [at(gap.x, gap.y + 1), at(gap.x, gap.y - 1)]
    : [at(gap.x - 1, gap.y), at(gap.x + 1, gap.y)];
}

export function archHeight(grid: GridGeometry, gap: GridCell, _heights?: Float32Array): number {
  const alongZ = archAxis(grid, gap) === 'alongZ';
  const a = alongZ ? wallNominalAt(grid, gap.x, gap.y + 1) : wallNominalAt(grid, gap.x - 1, gap.y);
  const b = alongZ ? wallNominalAt(grid, gap.x, gap.y - 1) : wallNominalAt(grid, gap.x + 1, gap.y);
  return Math.max(a, b) + ARCH_RISE;
}


export interface CanyonSkin {
  sandSun: THREE.Color;
  sandShade: THREE.Color;
  stone: THREE.Color;
  strata: readonly THREE.Color[];
  desert: THREE.Color;
  scrub: THREE.Color;
  farStone: THREE.Color;
  rim: THREE.Color;
  water: THREE.Color;
  waterDeep: THREE.Color;
  reed: THREE.Color;
  shaft: THREE.Color;
  dust: THREE.Color;
}

export function canyonSkin(palette: Palette, tokens: typeof TOKENS): CanyonSkin {
  const terracotta = tokens.accents['high-dim-maze'] ?? tokens.celebrationGold;
  const ember = tokens.accents['mountain-car'] ?? terracotta;
  const teal = tokens.accents['deep-sea'] ?? tokens.uiCyan;
  const leaf = tokens.accents['cart-pole'] ?? palette.groundA;
  const tan = TOKENS.palettes.dusk.groundA;
  const hot = TOKENS.palettes.dusk.skyHorizon;

  return {
    sandSun: shade(blendTokens(blendTokens(tan, terracotta, 0.62), palette.sun, 0.16), 0.08),
    sandShade: shade(blendTokens(blendTokens(tan, terracotta, 0.5), palette.hemiSky, 0.2), 0.44),
    stone: shade(blendTokens(terracotta, ember, 0.54), 0.63),
    strata: [
      shade(blendTokens(ember, terracotta, 0.2), 0.72),
      shade(blendTokens(terracotta, palette.sun, 0.42), 0.28),
      shade(blendTokens(terracotta, ember, 0.4), 0.55),
      shade(blendTokens(terracotta, palette.sun, 0.16), 0.3),
    ],
    desert: shade(blendTokens(blendTokens(palette.groundA, tan, 0.5), terracotta, 0.16), 0.22),
    scrub: shade(blendTokens(palette.groundA, palette.groundB, 0.4), 0.44),
    farStone: shade(blendTokens(terracotta, palette.skyHorizon, 0.48), 0.36),
    rim: blendTokens(palette.sun, hot, 0.42),
    water: blendTokens(teal, palette.sun, 0.1),
    waterDeep: shade(blendTokens(teal, TOKENS.palettes.underwater.skyBottom, 0.62), 0.34),
    reed: shade(blendTokens(leaf, palette.groundB, 0.18), 0.14),
    shaft: blendTokens(palette.sun, terracotta, 0.28),
    dust: blendTokens(blendTokens(terracotta, palette.sun, 0.55), palette.groundB, 0.2),
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


export const MESA_EDGE = 8.5;
export const MESA_EDGE_WIDTH = 3.5;
export const MESA_DROP = 10.5;

export function makeMesaHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.max(Math.abs(x), Math.abs(z)) - CANYON_HALF;
    const w = smoothstep01(out / 4);
    if (w <= 0) return 0;
    const n1 = valueNoise2(x * 0.038 + 3.7, z * 0.038 + 9.1, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.095 + 1.3, z * 0.095 + 4.7, 64, seed ^ 0x51ed270b) - 0.5;
    const n3 = valueNoise2(x * 0.24 + 7.9, z * 0.24 + 2.3, 64, seed ^ 0x2545f491) - 0.5;
    const dunes = n1 * 3.0 + n2 * 1.4 + n3 * 0.5;
    const wander = (valueNoise2(x * 0.06 + 21.3, z * 0.06 + 5.9, 64, seed ^ 0x7f4a7c15) - 0.5) * 4.5;
    const fall = smoothstep01((out - (MESA_EDGE + wander)) / MESA_EDGE_WIDTH);
    return w * dunes - fall * MESA_DROP;
  };
}


export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  archRoots: THREE.Object3D[];
  gaps: readonly GridCell[];
  wallHeights: Float32Array;
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
  const palette: Palette = ctx.tokens.palettes.afternoon;
  const skin = canyonSkin(palette, ctx.tokens);
  const group = new THREE.Group();
  group.name = 'slotCanyons';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];

  const sky = skyFromPalette(palette, 'afternoon', {
    sunDir: SUN_DIR,
    sunSize: 0.03,
    sunHdr: 5.2,
    sunGlow: 0.3,
    sunGlowPower: 26,
    topSpread: 0.68,
    bottomSpread: 0.42,
    radius: 340,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 3,
    pointDistance: 6,
    sunDistanceScale: 2.4,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.hemi.intensity = palette.hemiIntensity * 1.35;
  rig.setShadowFocus({ x: 0, y: 0.5, z: 0 }, SHADOW_RADIUS);

  applyFog(scene, palette, FOG_EXTENT);

  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    seed: ctx.seed,
    heightFn: makeMesaHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.06,
    slopeColor: skin.strata[0] ?? skin.stone,
    slopeStrength: 0.62,
    slopeSharpness: 1.8,
    flattenMask: { x: 0, z: 0, halfX: CANYON_HALF, halfZ: CANYON_HALF, feather: 3.2 },
    flattenHeight: CANYON_SINK,
    colorRamp: [
      { t: 0, color: blendTokens(skin.desert, skin.stone, 0.4) },
      { t: 0.45, color: skin.desert },
      { t: 1, color: blendTokens(skin.desert, skin.sandSun, 0.55) },
    ],
    materialOptions: {
      rimColor: skin.rim,
      rimStrength: 0.18,
      toonSteps: 3,
      toonSoftness: 0.32,
      toonFloor: 0.28,
    },
    receiveShadow: true,
    castShadow: false,
    name: 'canyonDesert',
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
        if (isWall(grid, x, y)) return skin.strata[0] ?? skin.stone;
        const open = clamp01((wallDistance(grid, x, y) - 1) / 1.5);
        return blendTokens(skin.sandShade, skin.sandSun, open);
      },
      wallTint: () => skin.stone,
      wallShape: 'block',
      wallHeight: WALL_BASE_HEIGHT,
      wallInset: 0,
      floorInset: 0.03,
      floorThickness: 0.16,
      jitter: 0.08,
      rimColor: skin.rim,
      rimStrength: 0.26,
      seed: ctx.seed,
      name: 'canyonCells',
    },
  );
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  const wallHeights = applyWallHeights(field, grid, ctx.seed);

  const strataGeometry = makeStrataGeometry(skin);
  const strataMaterial: OakMaterial = oakMaterial({
    color: white,
    vertexColors: true,
    flatShading: true,
    roughness: 0.96,
    rimColor: skin.rim,
    rimStrength: 0.5,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.26,
    name: 'canyonStrataMaterial',
  });
  owned.push({ dispose: () => strataMaterial.dispose() });
  owned.push({ dispose: () => strataGeometry.dispose() });
  {
    let count = 0;
    for (let i = 0; i < wallHeights.length; i++) if ((wallHeights[i] ?? 0) > 0) count++;
    const mesh = new THREE.InstancedMesh(strataGeometry, strataMaterial, Math.max(1, count));
    mesh.name = 'canyonStrata';
    mesh.castShadow = false;
    mesh.receiveShadow = true;
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3);
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scaleVec = new THREE.Vector3();
    const color = new THREE.Color();
    const world: Vec3Tuple = [0, 0, 0];
    let n = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        const h = wallHeights[y * grid.w + x] ?? 0;
        if (h <= 0) continue;
        cellToWorld(x, y, grid, world);
        position.set(world[0], 0, world[2]);
        scaleVec.set(1, h, 1);
        matrix.compose(position, quaternion, scaleVec);
        mesh.setMatrixAt(n, matrix);
        const k = 0.9 + 0.2 * hash2f(x, y, ctx.seed ^ 0x2c1b3a95);
        color.setRGB(k, k * (0.99 + 0.03 * hash2f(x, y, ctx.seed ^ 0x1a3)), k * 0.98);
        mesh.setColorAt(n, color);
        n++;
      }
    }
    mesh.count = n;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    group.add(mesh);
    owned.push({ dispose: () => mesh.dispose() });
  }

  const gaps = resolveGaps(grid);
  const archMaterial: OakMaterial = oakMaterial({
    color: white,
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: skin.rim,
    rimStrength: 0.55,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.42,
    name: 'canyonArchMaterial',
  });
  owned.push({ dispose: () => archMaterial.dispose() });
  const archRoots: THREE.Object3D[] = [];
  {
    const world: Vec3Tuple = [0, 0, 0];
    for (let i = 0; i < gaps.length; i++) {
      const gap = gaps[i];
      if (!gap) continue;
      const root = new THREE.Object3D();
      root.name = `canyonArchRoot${i}`;
      cellToWorld(gap.x, gap.y, grid, world);
      root.position.set(world[0], 0, world[2]);
      const jambs = archJambs(grid, gap, wallHeights);
      const geometry = makeArchGeometry(
        skin,
        archHeight(grid, gap, wallHeights),
        jambs,
        ctx.seed + i * 7919,
      );
      const mesh = new THREE.Mesh(geometry, archMaterial);
      mesh.name = `canyonArch${i}`;
      mesh.rotation.y = archYaw(archAxis(grid, gap));
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      root.add(mesh);
      group.add(root);
      archRoots.push(root);
      owned.push({ dispose: () => geometry.dispose() });
    }
  }

  {
    const geometry = makeMesaLineGeometry(skin, ctx.seed);
    const material: OakMaterial = oakMaterial({
      color: white,
      vertexColors: true,
      flatShading: true,
      roughness: 1,
      rimColor: palette.skyHorizon,
      rimStrength: 0.2,
      toonSteps: 2,
      toonSoftness: 0.4,
      toonFloor: 0.42,
      name: 'canyonMesaLineMaterial',
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'canyonMesaLine';
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    group.add(mesh);
    owned.push({ dispose: () => { geometry.dispose(); material.dispose(); } });
  }

  const offCanyon = excludeRect(0, 0, CANYON_HALF, CANYON_HALF, 0.7);
  const nearCamera = excludeRect(
    Math.sin(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    Math.cos(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    5.5,
    5.5,
    0,
  );
  const outside = excludeAny(offCanyon, nearCamera);

  const scrub = scatterGrass({
    count: 1500,
    area: { x: 0, z: 0, halfX: 24, halfZ: 24, innerRadius: CANYON_HALF + 0.9 },
    seed: ctx.seed ^ 0x5c1a,
    exclude: outside,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    color: skin.scrub,
    scale: [0.4, 0.95],
    stretch: [0.6, 1.1],
    hueJitter: 0.26,
    wind: 0.05,
    windFrequency: 1.9,
    rimColor: skin.rim,
    rimStrength: 0.36,
    name: 'desertScrub',
  });
  group.add(scrub.group);
  owned.push(scrub);
  timed.push(scrub);

  const boulders = scatterRocks({
    count: 340,
    area: { x: 0, z: 0, halfX: 28, halfZ: 28, innerRadius: CANYON_HALF + 1.1 },
    seed: ctx.seed ^ 0x77b3,
    exclude: outside,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.75,
    color: shade(skin.stone, 0.1),
    scale: [0.3, 1.5],
    stretch: [0.5, 1.1],
    hueJitter: 0.24,
    rimColor: skin.rim,
    rimStrength: 0.32,
    castShadow: true,
    name: 'desertBoulders',
  });
  group.add(boulders.group);
  owned.push(boulders);
  timed.push(boulders);

  const outcrops = scatterRocks({
    count: 30,
    area: { x: 0, z: 0, halfX: 19, halfZ: 19, innerRadius: CANYON_HALF + 4.2 },
    seed: ctx.seed ^ 0x18ce,
    exclude: outside,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.62,
    color: shade(blendTokens(skin.stone, skin.farStone, 0.3), 0.06),
    scale: [1.0, 2.5],
    stretch: [0.5, 1.05],
    hueJitter: 0.2,
    rimColor: skin.rim,
    rimStrength: 0.3,
    castShadow: true,
    name: 'desertOutcrops',
  });
  group.add(outcrops.group);
  owned.push(outcrops);
  timed.push(outcrops);

  const junipers = scatterTrees({
    archetypes: ['bare'],
    count: 26,
    area: { x: 0, z: 0, halfX: 22, halfZ: 22, innerRadius: CANYON_HALF + 2.4 },
    seed: ctx.seed ^ 0x31f7,
    exclude: outside,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.6,
    color: shade(blendTokens(skin.scrub, skin.stone, 0.42), 0.3),
    scale: [0.55, 1.05],
    stretch: [0.7, 1.15],
    hueJitter: 0.2,
    wind: 0.024,
    windFrequency: 0.9,
    rimColor: skin.rim,
    rimStrength: 0.3,
    castShadow: true,
    name: 'deadJunipers',
  });
  group.add(junipers.group);
  owned.push(junipers);
  timed.push(junipers);

  const scree = scatterRocks({
    count: 120,
    area: { x: 0, z: 0, halfX: CANYON_HALF - 0.6, halfZ: CANYON_HALF - 0.6 },
    seed: ctx.seed ^ 0x2d41,
    exclude: excludeAny(excludeWallCells(grid), excludeCellCenters(grid, 0.36)),
    color: shade(skin.strata[2] ?? skin.stone, 0.1),
    scale: [0.07, 0.2],
    stretch: [0.3, 0.7],
    hueJitter: 0.34,
    rimColor: skin.rim,
    rimStrength: 0.3,
    castShadow: false,
    name: 'canyonScree',
  });
  group.add(scree.group);
  owned.push(scree);
  timed.push(scree);

  const motes: AmbientParticlesHandle[] = [];
  const dust = makeAmbientParticles({
    preset: 'dust',
    count: 220,
    bounds: { x: 0, y: 1.25, z: 0, halfX: CANYON_HALF - 0.4, halfY: 0.55, halfZ: CANYON_HALF - 0.4 },
    seed: ctx.seed ^ 0x7d21,
    color: skin.dust,
    size: 0.34,
    intensity: 0.95,
    opacity: 0.1,
    rise: 0.035,
    swirl: 0.16,
    name: 'canyonDust',
  });
  group.add(dust.points);
  owned.push(dust);
  timed.push(dust);
  motes.push(dust);

  for (const spot of sandFallSpots(grid, ctx.seed, 4)) {
    const fall = makeAmbientParticles({
      preset: 'dust',
      count: 46,
      bounds: { x: spot.x, y: spot.top * 0.5, z: spot.z, halfX: 0.07, halfY: spot.top * 0.5, halfZ: 0.26 },
      seed: ctx.seed ^ (0x4a13 + spot.index * 977),
      color: blendTokens(skin.dust, skin.sandSun, 0.35),
      size: 0.42,
      intensity: 1.05,
      opacity: 0.22,
      rise: -0.42,
      swirl: 0.05,
      name: `canyonSandFall${spot.index}`,
    });
    group.add(fall.points);
    owned.push(fall);
    timed.push(fall);
    motes.push(fall);
  }

  const shimmer = makeMistPlanes({
    size: [HIGHDIMMAZE_W * CELL, HIGHDIMMAZE_H * CELL],
    color: blendTokens(skin.dust, palette.sun, 0.4),
    seed: ctx.seed ^ 0x6b21,
    layers: 2,
    baseY: 0.1,
    spread: 0.34,
    opacity: 0.045,
    speed: 0.05,
    scale: 3.4,
    nearFade: 2,
    name: 'canyonHeatShimmer',
  });
  group.add(shimmer.group);
  owned.push(shimmer);
  timed.push(shimmer);

  return {
    group,
    sky,
    rig,
    terrain,
    field,
    archRoots,
    gaps,
    wallHeights,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
    },
    setPixelRatio(dpr) {
      for (let i = 0; i < motes.length; i++) motes[i]?.setPixelRatio(dpr);
    },
    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      motes.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}


export function applyWallHeights(field: CellFieldHandle, grid: GridGeometry, seed: number): Float32Array {
  const heights = new Float32Array(grid.w * grid.h);
  const mesh = field.wallMeshes[0];
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scaleVec = new THREE.Vector3();
  const world: Vec3Tuple = [0, 0, 0];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      const base = wallHeightAt(grid, x, y);
      if (base <= 0) continue;
      const h = base * wallHeightJitter(x, y, seed);
      heights[y * grid.w + x] = h;
      if (!mesh) continue;
      const id = field.wallInstanceAt(x, y);
      if (id < 0) continue;
      cellToWorld(x, y, grid, world);
      position.set(world[0], 0, world[2]);
      scaleVec.set(1, h / WALL_BASE_HEIGHT, 1);
      matrix.compose(position, quaternion, scaleVec);
      mesh.setMatrixAt(id, matrix);
    }
  }
  if (mesh) {
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
  }
  return heights;
}

export function makeStrataGeometry(skin: CanyonSkin): THREE.BufferGeometry {
  const beds: readonly (readonly [number, number, number])[] =
    false
      ? [
          [0.46, 0.05, 1.03],
          [1.0, 0.032, 1.028],
        ]
      : [
          [0.17, 0.03, 1.042],
          [0.42, 0.025, 1.026],
          [0.67, 0.022, 1.046],
          [1.0, 0.032, 1.028],
        ];
  const parts: ColoredPart[] = [];
  for (let i = 0; i < beds.length; i++) {
    const bed = beds[i];
    if (!bed) continue;
    const [cy, half, spread] = bed;
    const g = new THREE.BoxGeometry(CELL * spread, half * 2, CELL * spread);
    g.translate(0, cy, 0);
    const tone = false ? (i === 0 ? 0 : 3) : i;
    parts.push({ geometry: g, color: skin.strata[tone] ?? skin.stone });
  }
  return mergeColoredParts(parts, 'canyonStrata');
}

export function makeArchGeometry(
  skin: CanyonSkin,
  height: number,
  jambs: readonly [number, number],
  seed: number,
): THREE.BufferGeometry {
  const spans = 20;
  const half = CELL * ARCH_SPAN_HALF;
  const depth = CELL * 0.5;
  const opening = Math.max(0.55, Math.min(ARCH_OPENING, height - ARCH_BAND));

  const [endA, endB] = jambs;
  const jambLine = (t: number): number => endA + (endB - endA) * (t + 1) * 0.5;

  const soffit = (t: number): number => {
    const a = Math.min(1, Math.abs(t));
    const shape = Math.pow(Math.max(0, 1 - Math.pow(a, 6)), 0.35);
    const wobble = 1 + 0.06 * Math.sin(t * 7.3 + seed * 0.017) + 0.035 * Math.sin(t * 15.1 - seed * 0.011);
    const base = jambLine(t);
    return base + Math.max(0, opening - base) * shape * wobble;
  };
  const crest = (t: number): number => {
    const base = jambLine(t);
    const hump = Math.max(0, height - base) * Math.pow(Math.max(0, 1 - t * t), 0.8);
    const erode = 0.035 * Math.sin(t * 5.1 + seed * 0.023) * (1 - t * t);
    return base + hump + erode;
  };

  const depthAt = (t: number): number => depth * (ARCH_RIBBON + (ARCH_LEG - ARCH_RIBBON) * Math.pow(Math.abs(t), 2.2));

  const xs = new Float32Array(spans + 1);
  const lo = new Float32Array(spans + 1);
  const hi = new Float32Array(spans + 1);
  const dz = new Float32Array(spans + 1);
  for (let i = 0; i <= spans; i++) {
    const t = (i / spans) * 2 - 1;
    xs[i] = t * half;
    const under = soffit(t);
    lo[i] = under;
    const taper = 1 - Math.min(1, Math.abs(t));
    hi[i] = Math.max(crest(t), under + ARCH_BAND * 0.5 * taper);
    dz[i] = depthAt(t);
  }

  const positions: number[] = [];
  const colors: number[] = [];
  const face = new THREE.Color(skin.strata[2] ?? skin.stone);
  const soffitColor = new THREE.Color(skin.strata[1] ?? skin.stone);
  const capColor = new THREE.Color(skin.strata[3] ?? skin.stone);
  const jambColor = new THREE.Color(skin.strata[2] ?? skin.stone);

  const push = (x: number, y: number, z: number, c: THREE.Color): void => {
    positions.push(x, y, z);
    colors.push(c.r, c.g, c.b);
  };
  const quad = (
    a: readonly [number, number, number],
    b: readonly [number, number, number],
    c: readonly [number, number, number],
    d: readonly [number, number, number],
    color: THREE.Color,
  ): void => {
    push(a[0], a[1], a[2], color);
    push(b[0], b[1], b[2], color);
    push(c[0], c[1], c[2], color);
    push(a[0], a[1], a[2], color);
    push(c[0], c[1], c[2], color);
    push(d[0], d[1], d[2], color);
  };

  for (let i = 0; i < spans; i++) {
    const x0 = xs[i] ?? 0;
    const x1 = xs[i + 1] ?? 0;
    const l0 = lo[i] ?? 0;
    const l1 = lo[i + 1] ?? 0;
    const h0 = hi[i] ?? 0;
    const h1 = hi[i + 1] ?? 0;
    const d0 = dz[i] ?? depth;
    const d1 = dz[i + 1] ?? depth;
    quad([x0, l0, d0], [x1, l1, d1], [x1, h1, d1], [x0, h0, d0], face);
    quad([x1, l1, -d1], [x0, l0, -d0], [x0, h0, -d0], [x1, h1, -d1], face);
    quad([x0, l0, -d0], [x1, l1, -d1], [x1, l1, d1], [x0, l0, d0], soffitColor);
    quad([x0, h0, d0], [x1, h1, d1], [x1, h1, -d1], [x0, h0, -d0], capColor);
  }
  {
    const xL = xs[0] ?? -half;
    const xR = xs[spans] ?? half;
    const dL = dz[0] ?? depth;
    const dR = dz[spans] ?? depth;
    quad([xL, lo[0] ?? 0, -dL], [xL, lo[0] ?? 0, dL], [xL, hi[0] ?? 0, dL], [xL, hi[0] ?? 0, -dL], jambColor);
    quad(
      [xR, lo[spans] ?? 0, dR],
      [xR, lo[spans] ?? 0, -dR],
      [xR, hi[spans] ?? 0, -dR],
      [xR, hi[spans] ?? 0, dR],
      jambColor,
    );
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'canyonArch';
  return g;
}

export function makeMesaLineGeometry(skin: CanyonSkin, seed: number): THREE.BufferGeometry {
  const rng = mulberry32((seed | 0) ^ 0x1d3b7c51);
  const count = 10;
  const parts: ColoredPart[] = [];
  for (let i = 0; i < count; i++) {
    const spread = Math.PI * 2 - 1.1;
    let angle = CAMERA_AZIMUTH + 0.55 + ((i + rng() * 0.55) / count) * spread;
    angle = ((angle + Math.PI) % (Math.PI * 2)) - Math.PI;
    const radius = 18 + rng() * 11;
    const w = 5 + rng() * 11;
    const d = 4 + rng() * 8;
    const h = 7.5 + rng() * 6;
    const cx = Math.sin(angle) * radius;
    const cz = Math.cos(angle) * radius;
    const base = -MESA_DROP - 1;
    const box = new THREE.BoxGeometry(w, h, d);
    box.translate(cx, base + h * 0.5, cz);
    parts.push({ geometry: box, color: skin.farStone });
    const cap = new THREE.BoxGeometry(w * 0.7, h * 0.3, d * 0.7);
    cap.translate(cx, base + h, cz);
    parts.push({ geometry: cap, color: blendTokens(skin.farStone, skin.strata[3] ?? skin.stone, 0.4) });
  }
  return mergeColoredParts(parts, 'canyonMesaLine');
}

export interface SandFallSpot {
  index: number;
  x: number;
  z: number;
  top: number;
}

export function sandFallSpots(grid: GridGeometry, seed: number, want: number): SandFallSpot[] {
  if (want <= 0) return [];
  interface Lip { x: number; z: number; top: number }
  const lips: Lip[] = [];
  const world: Vec3Tuple = [0, 0, 0];
  for (let y = 1; y < grid.h - 1; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (!isWall(grid, x, y)) continue;
      const left = !isWall(grid, x - 1, y);
      const right = !isWall(grid, x + 1, y);
      if (!left && !right) continue;
      const dir = right ? 1 : -1;
      cellToWorld(x, y, grid, world);
      lips.push({
        x: world[0] + dir * CELL * 0.52,
        z: world[2],
        top: wallHeightAt(grid, x, y),
      });
    }
  }
  if (lips.length === 0) return [];
  const rng = mulberry32((seed | 0) ^ 0x3f2a91b7);
  const out: SandFallSpot[] = [];
  const used = new Set<number>();
  for (let i = 0; i < want && used.size < lips.length; i++) {
    let pick = Math.min(lips.length - 1, (rng() * lips.length) | 0);
    let guard = 0;
    while (used.has(pick) && guard++ < lips.length) pick = (pick + 1) % lips.length;
    used.add(pick);
    const lip = lips[pick];
    if (!lip) continue;
    out.push({ index: i, x: lip.x, z: lip.z, top: lip.top });
  }
  return out;
}
