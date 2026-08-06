
import * as THREE from 'three';
import {
  CELL,
  buildCellField,
  cellToWorld,
  isWall,
  type CellFieldHandle,
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms, type OakMaterial } from '@kit/materials';
import { makeAmbientParticles, makeMistPlanes, type AmbientParticlesHandle } from '@kit/particles';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import {
  coniferTreeGeometry,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
} from '@kit/vegetation';
import {
  POMAZE_GAPS,
  POMAZE_H,
  POMAZE_W,
  poMazeWalls,
} from '@sim/envs/pomaze';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';


export const MAZE_HALF = (POMAZE_W * CELL) / 2;
export const TERRAIN_SIZE = 84;
export const TERRAIN_SEGMENTS = 120;
export const FLOOR_SINK = -0.05;

export const FOG_EXTENT = POMAZE_W * 2.6;

export const SUN_DIR: readonly [number, number, number] = [0.62, 0.55, -0.556];
export const SHADOW_RADIUS = 5;

export const CAMERA_AZIMUTH = -1.35;

export const BUBBLE_INNER = CELL;
export const BUBBLE_OUTER = CELL * 2;

export const REVEAL_RADIUS = 3.2;
export const FOG_LIFT = 0.55;

export const OUTER_FOREST_INNER = 14;
export const OUTER_FOREST_OUTER = 21;


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
  if (g && g.w === POMAZE_W && g.h === POMAZE_H) return g;
  return {
    w: POMAZE_W,
    h: POMAZE_H,
    walls: poMazeWalls(),
    hallways: POMAZE_GAPS.map((c) => ({ ...c })),
  };
}

export function resolveGaps(grid: GridGeometry): readonly GridCell[] {
  const h = grid.hallways;
  return h && h.length > 0 ? h : POMAZE_GAPS;
}


export function bubbleVisibility(wx: number, wz: number, cx: number, cz: number): number {
  const d = Math.max(Math.abs(cx - wx), Math.abs(cz - wz));
  if (d <= BUBBLE_INNER) return 1;
  if (d >= BUBBLE_OUTER) return 0;
  return 1 - smoothstep01((d - BUBBLE_INNER) / (BUBBLE_OUTER - BUBBLE_INNER));
}


export const MOTH_RIM = BUBBLE_OUTER * 0.88;
export const MOTH_BREATH = 0.16;
export const MOTH_SWING = 0.34;
export const MOTH_HEIGHT = 0.6;
export const MOTH_BOB = 0.12;

export function compassOffset(
  sx: number,
  sy: number,
  time: number,
  phase: number,
  out: THREE.Vector3,
): boolean {
  const dx = Math.sign(sx);
  const dz = Math.sign(sy);
  if (dx === 0 && dz === 0) {
    out.set(0, MOTH_HEIGHT, 0);
    return false;
  }
  const bearing = Math.atan2(dz, dx) + MOTH_SWING * Math.sin(time * 1.31 + phase * 1.7);
  const radius = MOTH_RIM + MOTH_BREATH * Math.sin(time * 0.9 + phase);
  out.set(
    Math.cos(bearing) * radius,
    MOTH_HEIGHT + MOTH_BOB * Math.sin(time * 2.1 + phase),
    Math.sin(bearing) * radius,
  );
  return true;
}

export function revealVisibility(rx: number, rz: number, cx: number, cz: number, radius: number): number {
  const d = Math.hypot(cx - rx, cz - rz);
  if (d >= radius) return 0;
  return 1 - smoothstep01((d - radius * 0.45) / (radius * 0.55));
}


export interface ForestSkin {
  litterA: THREE.Color;
  litterB: THREE.Color;
  spruce: THREE.Color;
  spruceFar: THREE.Color;
  fern: THREE.Color;
  stone: THREE.Color;
  stoneDark: THREE.Color;
  haze: THREE.Color;
  motes: THREE.Color;
  bloom: THREE.Color;
  moth: THREE.Color;
}

export function forestSkin(palette: Palette, tokens: typeof TOKENS): ForestSkin {
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const gold = tokens.celebrationGold;
  const haze = shade(blendTokens(palette.fog, palette.skyTop, 0.62), 0.62);
  return {
    litterA: shade(blendTokens(palette.groundA, haze, 0.55), 0.6),
    litterB: shade(blendTokens(palette.groundB, haze, 0.5), 0.6),
    spruce: shade(blendTokens(palette.groundA, silver, 0.1), 0.74),
    spruceFar: shade(blendTokens(palette.groundA, haze, 0.5), 0.5),
    fern: shade(blendTokens(palette.groundA, palette.groundB, 0.3), 0.62),
    stone: shade(blendTokens(silver, palette.hemiGround, 0.4), 0.44),
    stoneDark: shade(blendTokens(palette.hemiGround, silver, 0.2), 0.58),
    haze,
    motes: blendTokens(palette.sun, silver, 0.3),
    bloom: blendTokens(silver, palette.skyTop, 0.55),
    moth: blendTokens(gold, silver, 0.34),
  };
}


export function excludeCellCenters(radius: number): ExcludeFn {
  const r2 = radius * radius;
  return (x, z) => {
    const cx = Math.round(x / CELL);
    const cz = Math.round(z / CELL);
    const dx = x - cx * CELL;
    const dz = z - cz * CELL;
    return dx * dx + dz * dz < r2;
  };
}

export function excludeDisc(radius: number): ExcludeFn {
  const r2 = radius * radius;
  return (x, z) => x * x + z * z < r2;
}


export function makeForestHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.max(Math.abs(x), Math.abs(z)) - MAZE_HALF;
    const w = smoothstep01(out / 6);
    if (w <= 0) return 0;
    const n1 = valueNoise2(x * 0.07 + 3.1, z * 0.07 + 9.4, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.19 + 6.7, z * 0.19 + 2.2, 64, seed ^ 0x51ed270b) - 0.5;
    return w * (n1 * 1.5 + n2 * 0.5);
  };
}


export interface SpruceRanksOptions {
  grid: GridGeometry;
  tier: QualityTier;
  seed: number;
  color: THREE.ColorRepresentation;
  rimColor: THREE.ColorRepresentation;
  name?: string;
}

export interface SpruceRanksHandle {
  group: THREE.Group;
  mesh: THREE.InstancedMesh;
  material: OakMaterial;
  cellCount: number;
  treesPerCell: number;
  visibilityAt(x: number, y: number): number;
  setBubble(wx: number, wz: number, rx: number, rz: number, reveal: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildSpruceRanks(opts: SpruceRanksOptions): SpruceRanksHandle {
  const { grid, tier, seed } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'spruceRanks';
  const detail = tier === 'thumb' ? 0 : tier === 'medium' ? 1 : 2;
  const treesPerCell = tier === 'thumb' ? 1 : tier === 'medium' ? 2 : 3;

  const total = grid.w * grid.h;
  const index = new Int32Array(total).fill(-1);
  const cellX: number[] = [];
  const cellY: number[] = [];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (!isWall(grid, x, y)) continue;
      index[y * grid.w + x] = cellX.length;
      cellX.push(x);
      cellY.push(y);
    }
  }
  const cellCount = cellX.length;
  const instances = Math.max(1, cellCount * treesPerCell);

  const baseX = new Float32Array(instances);
  const baseZ = new Float32Array(instances);
  const baseScale = new Float32Array(instances);
  const baseStretch = new Float32Array(instances);
  const baseYaw = new Float32Array(instances);
  const baseTilt = new Float32Array(instances);
  const phase = new Float32Array(instances);
  const vis = new Float32Array(Math.max(1, cellCount)).fill(-1);

  const world: Vec3Tuple = [0, 0, 0];
  const colorAttr = new Float32Array(instances * 3);
  for (let i = 0; i < cellCount; i++) {
    const x = cellX[i] ?? 0;
    const y = cellY[i] ?? 0;
    cellToWorld(x, y, grid, world);
    for (let k = 0; k < treesPerCell; k++) {
      const id = i * treesPerCell + k;
      const h1 = hash2f(x * 7 + k, y * 13 + k * 3, seed ^ 0x2f6b);
      const h2 = hash2f(x * 3 + k * 5, y * 11 + k, seed ^ 0x91c3);
      const h3 = hash2f(x + k * 17, y + k * 29, seed ^ 0x5ab1);
      const ring = ((k + 0.5) / treesPerCell) * Math.PI * 2 + h1 * 1.1;
      const radius = treesPerCell > 1 ? 0.27 + h2 * 0.08 : 0.1;
      baseX[id] = world[0] + Math.cos(ring) * radius;
      baseZ[id] = world[2] + Math.sin(ring) * radius;
      baseScale[id] = 0.34 + h3 * 0.18 + (k === 0 ? 0.06 : 0);
      baseStretch[id] = 1.45 + h1 * 1.0;
      baseYaw[id] = h2 * Math.PI * 2;
      baseTilt[id] = (h3 - 0.5) * 0.14;
      phase[id] = k === 0 ? 0 : 0.18 + h1 * 0.3;
      const j = 1 + (h3 - 0.5) * 0.24;
      colorAttr[id * 3] = j;
      colorAttr[id * 3 + 1] = j * (1 + (h1 - 0.5) * 0.1);
      colorAttr[id * 3 + 2] = j * (1 + (h2 - 0.5) * 0.14);
    }
  }

  const geometry = coniferTreeGeometry(detail);
  geometry.name = 'spruceRank';
  const material = oakMaterial({
    color: opts.color,
    roughness: 0.95,
    flatShading: true,
    vertexColors: true,
    rimColor: opts.rimColor,
    rimStrength: 0.5,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.22,
    wind: { amplitude: 0.024, frequency: 0.75, height: 1.8 },
    tier,
    name: 'spruceRankMaterial',
  });

  const mesh = new THREE.InstancedMesh(geometry, material, instances);
  mesh.name = 'spruceRankMesh';
  mesh.castShadow = tier !== 'thumb';
  mesh.receiveShadow = false;
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(colorAttr, 3);
  mesh.instanceColor.needsUpdate = true;
  mesh.count = instances;
  group.add(mesh);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3();

  const writeCell = (i: number, v: number): void => {
    for (let k = 0; k < treesPerCell; k++) {
      const id = i * treesPerCell + k;
      const p = phase[id] ?? 0;
      const raw = p >= 1 ? 0 : (v - p) / (1 - p);
      const g = smoothstep01(raw);
      const s = baseScale[id] ?? 1;
      position.set(baseX[id] ?? 0, 0, baseZ[id] ?? 0);
      euler.set(baseTilt[id] ?? 0, baseYaw[id] ?? 0, (baseTilt[id] ?? 0) * 0.7);
      quaternion.setFromEuler(euler);
      const wide = s * g * (0.55 + 0.45 * g);
      scale.set(wide, s * (baseStretch[id] ?? 1) * g, wide);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(id, matrix);
    }
  };

  for (let i = 0; i < cellCount; i++) writeCell(i, 0);
  vis.fill(0);
  mesh.instanceMatrix.needsUpdate = true;

  return {
    group,
    mesh,
    material,
    cellCount,
    treesPerCell,
    visibilityAt(x, y) {
      if (x < 0 || y < 0 || x >= grid.w || y >= grid.h) return 0;
      const i = index[y * grid.w + x] ?? -1;
      return i < 0 ? 0 : (vis[i] ?? 0);
    },
    setBubble(wx, wz, rx, rz, reveal) {
      let dirty = false;
      for (let i = 0; i < cellCount; i++) {
        cellToWorld(cellX[i] ?? 0, cellY[i] ?? 0, grid, world);
        let v = bubbleVisibility(wx, wz, world[0], world[2]);
        if (reveal > 0) {
          const r = reveal * revealVisibility(rx, rz, world[0], world[2], REVEAL_RADIUS);
          if (r > v) v = r;
        }
        if (Math.abs(v - (vis[i] ?? -1)) < 1e-4) continue;
        vis[i] = v;
        writeCell(i, v);
        dirty = true;
      }
      if (dirty) mesh.instanceMatrix.needsUpdate = true;
    },
    setTime(t) {
      oakUniforms(material)?.setTime(t);
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}


export interface GapGateHandle {
  root: THREE.Object3D;
  body: THREE.Object3D;
  cell: GridCell;
}

export function buildGapGates(
  grid: GridGeometry,
  gaps: readonly GridCell[],
  material: THREE.Material,
  skin: ForestSkin,
  tier: QualityTier,
): { handles: GapGateHandle[]; geometry: THREE.BufferGeometry } {
  const geometry = makeGateStonesGeometry(skin, tier);
  const handles: GapGateHandle[] = [];
  const world: Vec3Tuple = [0, 0, 0];
  for (let i = 0; i < gaps.length; i++) {
    const g = gaps[i];
    if (!g) continue;
    const root = new THREE.Object3D();
    root.name = `gapGateRoot${i}`;
    cellToWorld(g.x, g.y, grid, world);
    root.position.set(world[0], 0, world[2]);
    const body = new THREE.Object3D();
    body.name = `gapGateBody${i}`;
    const bandRunsAlongX = isWall(grid, g.x - 1, g.y) && isWall(grid, g.x + 1, g.y);
    body.rotation.y = bandRunsAlongX ? 0 : Math.PI / 2;
    body.scale.setScalar(0.0001);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `gapGateStones${i}`;
    mesh.castShadow = tier !== 'thumb';
    mesh.receiveShadow = tier !== 'thumb';
    body.add(mesh);
    root.add(body);
    handles.push({ root, body, cell: { x: g.x, y: g.y } });
  }
  return { handles, geometry };
}

export function makeGateStonesGeometry(skin: ForestSkin, tier: QualityTier): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const segments = tier === 'thumb' ? 4 : tier === 'medium' ? 5 : 6;
  for (const sx of [-1, 1]) {
    const stone = new THREE.CylinderGeometry(0.1, 0.17, 0.74, segments, 1);
    stone.translate(0, 0.37, 0);
    stone.rotateZ(sx * -0.13);
    stone.translate(sx * 0.4, 0, 0);
    parts.push({ geometry: stone, color: skin.stone });
    const cap = new THREE.IcosahedronGeometry(0.11, 0);
    cap.scale(1, 0.55, 1);
    cap.translate(sx * 0.44, 0.74, 0);
    parts.push({ geometry: cap, color: skin.stoneDark });
  }
  return mergeColoredParts(parts, 'gapGateStones');
}


export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  fog: THREE.FogExp2;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  spruce: SpruceRanksHandle;
  gates: GapGateHandle[];
  gaps: readonly GridCell[];
  skin: ForestSkin;
  setTime(t: number): void;
  setPixelRatio(dpr: number): void;
  setShadowFocus(x: number, z: number): void;
  setFogLift(amount: number): void;
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
  const palette: Palette = ctx.tokens.palettes.dawn;
  const skin = forestSkin(palette, ctx.tokens);
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'fogboundForest';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];

  const sky = skyFromPalette(palette, 'dawn', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.045,
    sunHdr: 3.4,
    sunGlow: 0.5,
    sunGlowPower: 9,
    topSpread: 0.5,
    bottomSpread: 0.3,
    radius: 340,
  });
  sky.setColors(
    blendTokens(palette.skyTop, skin.haze, 0.18),
    blendTokens(palette.skyHorizon, skin.haze, 0.34),
    skin.haze,
  );
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 5.5,
    sunDistanceScale: 2.6,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, SHADOW_RADIUS);

  const fog = applyFog(scene, palette, FOG_EXTENT);
  fog.color.copy(skin.haze);
  const baseDensity = fog.density;

  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    tier,
    seed: ctx.seed,
    heightFn: makeForestHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.06,
    slopeColor: skin.stoneDark,
    slopeStrength: 0.45,
    slopeSharpness: 2.4,
    flattenMask: { x: 0, z: 0, halfX: MAZE_HALF, halfZ: MAZE_HALF, feather: 3.4 },
    flattenHeight: FLOOR_SINK,
    colorRamp: [
      { t: 0, color: blendTokens(skin.litterA, skin.stoneDark, 0.22) },
      { t: 0.45, color: skin.litterA },
      { t: 1, color: blendTokens(skin.litterB, skin.haze, 0.12) },
    ],
    materialOptions: {
      rimColor: skin.haze,
      rimStrength: 0.14,
      toonSteps: 3,
      toonSoftness: 0.32,
      toonFloor: 0.26,
    },
    receiveShadow: shadows,
    castShadow: false,
    name: 'forestFloor',
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
        const h = hash2f(x, y, ctx.seed ^ 0x3b71);
        return blendTokens(skin.litterA, skin.litterB, h * h);
      },
      wallTint: () => skin.spruce,
      wallShape: 'block',
      wallHeight: 0.6,
      wallInset: 0.04,
      floorInset: 0.03,
      floorThickness: 0.12,
      jitter: 0.14,
      rimColor: skin.haze,
      rimStrength: 0.16,
      seed: ctx.seed,
      name: 'forestFloorCells',
    },
    tier,
  );
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  collapseAllWalls(field, grid);

  const spruce = buildSpruceRanks({
    grid,
    tier,
    seed: ctx.seed,
    color: skin.spruce,
    rimColor: skin.haze,
    name: 'spruceRanks',
  });
  group.add(spruce.group);
  owned.push(spruce);
  timed.push(spruce);

  const stoneMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: skin.haze,
    rimStrength: 0.32,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.24,
    tier,
    name: 'forestStone',
  });
  owned.push({ dispose: () => stoneMaterial.dispose() });
  const gaps = resolveGaps(grid);
  const gates = buildGapGates(grid, gaps, stoneMaterial, skin, tier);
  owned.push({ dispose: () => gates.geometry.dispose() });
  for (const gate of gates.handles) group.add(gate.root);

  const clearCenters = excludeCellCenters(0.34);
  const ferns = scatterGrass({
    count: 5200,
    area: { halfX: MAZE_HALF + 2.5, halfZ: MAZE_HALF + 2.5 },
    seed: ctx.seed ^ 0x1f4d,
    tier,
    exclude: clearCenters,
    color: skin.fern,
    scale: [0.11, 0.27],
    stretch: [0.7, 1.6],
    hueJitter: 0.24,
    wind: 0.04,
    windFrequency: 1.1,
    rimColor: skin.haze,
    rimStrength: 0.34,
    name: 'forestFerns',
  });
  group.add(ferns.group);
  owned.push(ferns);
  timed.push(ferns);

  const stones = scatterRocks({
    count: 240,
    area: { halfX: MAZE_HALF + 6, halfZ: MAZE_HALF + 6 },
    seed: ctx.seed ^ 0x6b23,
    tier,
    exclude: clearCenters,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    color: skin.stone,
    scale: [0.12, 0.34],
    stretch: [0.45, 0.85],
    hueJitter: 0.22,
    rimColor: skin.haze,
    rimStrength: 0.3,
    castShadow: false,
    name: 'forestStones',
  });
  group.add(stones.group);
  owned.push(stones);
  timed.push(stones);

  const outer = scatterTrees({
    archetypes: ['conifer'],
    count: 1400,
    area: { halfX: OUTER_FOREST_OUTER, halfZ: OUTER_FOREST_OUTER, innerRadius: OUTER_FOREST_INNER },
    seed: ctx.seed ^ 0x77c1,
    tier,
    exclude: excludeDisc(OUTER_FOREST_INNER),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.8,
    color: skin.spruceFar,
    scale: [0.52, 1.22],
    stretch: [0.85, 1.3],
    hueJitter: 0.18,
    wind: 0.03,
    windFrequency: 0.7,
    rimColor: skin.haze,
    rimStrength: 0.4,
    castShadow: false,
    name: 'outerForest',
  });
  group.add(outer.group);
  owned.push(outer);
  timed.push(outer);

  const mist = makeMistPlanes({
    size: [18, 18],
    color: blendTokens(skin.haze, palette.sun, 0.28),
    seed: ctx.seed ^ 0x2ab7,
    tier,
    layers: 3,
    baseY: 0.06,
    spread: 0.5,
    opacity: 0.1,
    speed: 0.009,
    scale: 3.2,
    nearFade: 2.6,
    name: 'forestMist',
  });
  group.add(mist.group);
  owned.push(mist);
  timed.push(mist);

  const motes: AmbientParticlesHandle[] = [];
  const dust = makeAmbientParticles({
    preset: 'dust',
    count: 340,
    bounds: { x: 0, y: 0.85, z: 0, halfX: MAZE_HALF + 2, halfY: 0.8, halfZ: MAZE_HALF + 2 },
    seed: ctx.seed ^ 0x4e11,
    tier,
    color: skin.motes,
    size: 0.5,
    intensity: 1.3,
    opacity: 0.34,
    rise: 0.015,
    swirl: 0.16,
    name: 'forestDust',
  });
  group.add(dust.points);
  owned.push(dust);
  timed.push(dust);
  motes.push(dust);

  const spores = makeAmbientParticles({
    preset: 'pollen',
    count: 150,
    bounds: { x: 0, y: 1.9, z: 0, halfX: MAZE_HALF + 2, halfY: 1.1, halfZ: MAZE_HALF + 2 },
    seed: ctx.seed ^ 0x7d05,
    tier,
    color: blendTokens(skin.motes, skin.bloom, 0.3),
    size: 0.7,
    intensity: 1.5,
    opacity: 0.3,
    rise: -0.05,
    swirl: 0.24,
    name: 'forestSpores',
  });
  group.add(spores.points);
  owned.push(spores);
  timed.push(spores);
  motes.push(spores);

  return {
    group,
    sky,
    rig,
    fog,
    terrain,
    field,
    spruce,
    gates: gates.handles,
    gaps,
    skin,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
    },
    setPixelRatio(dpr) {
      for (let i = 0; i < motes.length; i++) motes[i]?.setPixelRatio(dpr);
    },
    setShadowFocus(x, z) {
      rig.setShadowFocus({ x, y: 0.3, z }, SHADOW_RADIUS);
    },
    setFogLift(amount) {
      const a = clamp01(amount);
      fog.density = baseDensity * (1 - FOG_LIFT * a);
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

export function collapseAllWalls(field: CellFieldHandle, grid: GridGeometry): void {
  const mesh = field.wallMeshes[0];
  if (!mesh) return;
  const m = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      const id = field.wallInstanceAt(x, y);
      if (id < 0) continue;
      mesh.getMatrixAt(id, m);
      m.decompose(p, q, s);
      m.compose(p, q, s.set(0, 0, 0));
      mesh.setMatrixAt(id, m);
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
}
