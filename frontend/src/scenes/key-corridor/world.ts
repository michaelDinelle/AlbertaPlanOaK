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
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, type OakMaterial } from '@kit/materials';
import { blendTokens, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles, makeMistPlanes } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f } from '@kit/rng';
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
  type VegetationHandle,
} from '@kit/vegetation';
import { makeWorld, type WorldCore } from '@kit/world';
import {
  KEYCORRIDOR_DOOR,
  KEYCORRIDOR_H,
  KEYCORRIDOR_W,
  keyCorridorWalls,
} from '@sim/envs/keycorridor';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';

export const GARDEN_HALF_X = (KEYCORRIDOR_W * CELL) / 2;
export const GARDEN_HALF_Z = (KEYCORRIDOR_H * CELL) / 2;
export const TERRAIN_SIZE = 40;
export const TERRAIN_SEGMENTS = 128;

export const KERB_HEIGHT = 0.5;
export const DIVIDER_HEIGHT = 1.02;
export const DIVIDER_CAP = 0.11;
export const DIVIDER_THICKNESS = 0.7;
export const GARDEN_SINK = -0.06;

export const FOG_EXTENT = TERRAIN_SIZE * 11;

export const SUN_DIR: readonly [number, number, number] = [-0.46, 0.78, -0.42];
export const SHADOW_RADIUS = 9.5;

export const CAMERA_AZIMUTH = -0.72;
export const CAMERA_STANDOFF = 12;

export function smoothBand(edge0: number, edge1: number, x: number): number {
  if (edge1 <= edge0) return x >= edge1 ? 1 : 0;
  return smoothstep01((x - edge0) / (edge1 - edge0));
}

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === KEYCORRIDOR_W && g.h === KEYCORRIDOR_H) return g;
  return {
    w: KEYCORRIDOR_W,
    h: KEYCORRIDOR_H,
    walls: keyCorridorWalls(),
    doors: [{ ...KEYCORRIDOR_DOOR }],
    hallways: [{ ...KEYCORRIDOR_DOOR }],
  };
}

export function resolveDoor(grid: GridGeometry): GridCell {
  const d = grid.doors?.[0] ?? grid.hallways?.[0];
  return d ? { x: d.x, y: d.y } : { ...KEYCORRIDOR_DOOR };
}

export const SIDE_WEST = -1;
export const SIDE_THRESHOLD = 0;
export const SIDE_EAST = 1;

export function gardenSide(door: GridCell, x: number): number {
  return x < door.x ? SIDE_WEST : x > door.x ? SIDE_EAST : SIDE_THRESHOLD;
}

export function isDividerWall(grid: GridGeometry, x: number, y: number): boolean {
  if (!isWall(grid, x, y)) return false;
  const openX = !isWall(grid, x - 1, y) && !isWall(grid, x + 1, y);
  const openZ = !isWall(grid, x, y - 1) && !isWall(grid, x, y + 1);
  return openX || openZ;
}

export function gateYaw(grid: GridGeometry, door: GridCell): number {
  const wallRunsAlongZ = isWall(grid, door.x, door.y - 1) && isWall(grid, door.x, door.y + 1);
  return wallRunsAlongZ ? Math.PI / 2 : 0;
}

export interface SideBounds {
  cx: number;
  cz: number;
  halfX: number;
  halfZ: number;
  cells: number;
}

export function sideBounds(grid: GridGeometry, door: GridCell, side: number): SideBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  let cells = 0;
  const w: Vec3Tuple = [0, 0, 0];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isWall(grid, x, y)) continue;
      if (gardenSide(door, x) !== side) continue;
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

export interface GardenSkin {
  wildFloor: THREE.Color;
  formalFloor: THREE.Color;
  threshold: THREE.Color;
  soil: THREE.Color;
  kerb: THREE.Color;
  divider: THREE.Color;
  dividerCap: THREE.Color;
  ivy: THREE.Color;
  ivyDark: THREE.Color;
  wood: THREE.Color;
  woodDark: THREE.Color;
  iron: THREE.Color;
  shrineStone: THREE.Color;
  shrineDark: THREE.Color;
  amber: THREE.Color;
  shrinePad: THREE.Color;
  wildGrass: THREE.Color;
  wildBloom: THREE.Color;
  gravel: THREE.Color;
  topiary: THREE.Color;
  meadow: THREE.Color;
  tree: THREE.Color;
  motesWild: THREE.Color;
  motesFormal: THREE.Color;
}

export function gardenSkin(palette: Palette, tokens: typeof TOKENS): GardenSkin {
  const amber = tokens.accents['key-corridor'] ?? tokens.celebrationGold;
  const leaf = tokens.accents['four-rooms'] ?? palette.groundB;
  const moss = tokens.accents['procedural-maze'] ?? palette.groundA;
  const gold = tokens.accents['multi-key-corridor'] ?? tokens.celebrationGold;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const terracotta = tokens.accents['high-dim-maze'] ?? tokens.celebrationGold;
  const earth = TOKENS.palettes.dusk.groundA;

  return {
    wildFloor: shade(blendTokens(palette.groundA, moss, 0.36), 0.34),
    formalFloor: shade(blendTokens(palette.hemiSky, palette.hemiGround, 0.36), 0.3),
    threshold: shade(blendTokens(gold, palette.hemiGround, 0.3), 0.34),
    soil: shade(blendTokens(palette.hemiGround, earth, 0.45), 0.42),
    kerb: shade(blendTokens(palette.skyBottom, terracotta, 0.34), 0.54),
    divider: shade(blendTokens(palette.groundA, silver, 0.3), 0.72),
    dividerCap: shade(blendTokens(silver, palette.hemiGround, 0.46), 0.52),
    ivy: shade(blendTokens(leaf, palette.groundA, 0.34), 0.26),
    ivyDark: shade(blendTokens(moss, palette.groundA, 0.55), 0.52),
    wood: shade(blendTokens(terracotta, earth, 0.16), 0.02),
    woodDark: shade(blendTokens(terracotta, earth, 0.38), 0.16),
    iron: shade(blendTokens(palette.hemiGround, silver, 0.18), 0.68),
    shrineStone: shade(blendTokens(silver, palette.hemiGround, 0.26), 0.2),
    shrineDark: shade(blendTokens(palette.hemiGround, silver, 0.3), 0.46),
    amber: new THREE.Color(amber),
    shrinePad: shade(blendTokens(gold, palette.hemiGround, 0.3), 0.4),
    wildGrass: shade(blendTokens(palette.groundB, moss, 0.42), 0.22),
    wildBloom: shade(blendTokens(amber, palette.sun, 0.18), 0.2),
    gravel: shade(blendTokens(palette.hemiSky, palette.hemiGround, 0.3), 0.46),
    topiary: shade(blendTokens(palette.groundA, leaf, 0.28), 0.4),
    meadow: shade(blendTokens(palette.groundA, palette.groundB, 0.3), 0.34),
    tree: shade(blendTokens(palette.groundA, moss, 0.26), 0.4),
    motesWild: blendTokens(amber, palette.sun, 0.4),
    motesFormal: blendTokens(palette.hemiSky, palette.sun, 0.4),
  };
}

export function makeMeadowHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.max(Math.abs(x) - GARDEN_HALF_X, Math.abs(z) - GARDEN_HALF_Z);
    const w = smoothstep01(out / 3.4);
    if (w <= 0) return 0;
    const n1 = valueNoise2(x * 0.062 + 3.7, z * 0.062 + 9.1, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.168 + 1.3, z * 0.168 + 6.7, 64, seed ^ 0x51ed270b) - 0.5;
    return w * (n1 * 3.4 + n2 * 1.1);
  };
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  door: GridCell;
  gateYaw: number;
  dividerCells: readonly GridCell[];
  paintFloor(x: number, y: number, color: THREE.ColorRepresentation): void;
  restoreFloorTint(x: number, y: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.morning;
  const skin = gardenSkin(palette, ctx.tokens);
  const w = makeWorld('ivyGateGarden');

  const door = resolveDoor(grid);
  const yaw = gateYaw(grid, door);

  const sky = skyFromPalette(palette, 'morning', {
    sunDir: SUN_DIR,
    sunSize: 0.04,
    sunHdr: 5,
    sunGlow: 0.3,
    sunGlowPower: 24,
    topSpread: 0.68,
    bottomSpread: 0.56,
    radius: 340,
  });
  w.add(sky.mesh);
  w.own(sky);
  w.time({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 5.5,
    sunDistanceScale: 2.4,
  });
  w.add(rig.group);
  w.own(rig);
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, SHADOW_RADIUS);

  applyFog(scene, palette, FOG_EXTENT);

  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    seed: ctx.seed,
    heightFn: makeMeadowHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.12,
    slopeColor: skin.soil,
    slopeStrength: 0.5,
    slopeSharpness: 2.2,
    flattenMask: { x: 0, z: 0, halfX: GARDEN_HALF_X + 0.25, halfZ: GARDEN_HALF_Z + 0.25, feather: 1.1 },
    flattenHeight: GARDEN_SINK,
    colorRamp: [
      { t: 0, color: blendTokens(skin.meadow, skin.soil, 0.55) },
      { t: 0.4, color: skin.meadow },
      { t: 0.75, color: blendTokens(skin.meadow, palette.groundB, 0.4) },
      { t: 1, color: blendTokens(skin.meadow, palette.groundB, 0.72) },
    ],
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.18,
      toonSteps: 3,
      toonSoftness: 0.3,
      toonFloor: 0.26,
    },
    receiveShadow: true,
    castShadow: false,
    name: 'ivyGateMeadow',
  });
  w.add(terrain.mesh);
  w.own(terrain);

  const FLOOR_JITTER = 0.17;
  const TINT_WILD_A = skin.wildFloor;
  const TINT_WILD_B = blendTokens(skin.wildFloor, skin.soil, 0.24);
  const TINT_FORMAL_A = skin.formalFloor;
  const TINT_FORMAL_B = blendTokens(skin.formalFloor, skin.divider, 0.2);
  const floorTintFor = (x: number, y: number): THREE.Color => {
    if (isWall(grid, x, y)) return skin.soil;
    const side = gardenSide(door, x);
    if (side === SIDE_THRESHOLD) return skin.threshold;
    if (side === SIDE_WEST) return ((x * 3 + y * 5) & 1) === 0 ? TINT_WILD_A : TINT_WILD_B;
    return (y & 1) === 0 ? TINT_FORMAL_A : TINT_FORMAL_B;
  };

  const white = palette.sun;
  const field = buildCellField(
    grid,
    {
      floorColor: white,
      wallColor: white,
      floorUnderWalls: true,
      floorTint: (x, y) => floorTintFor(x, y),
      wallTint: () => skin.kerb,
      wallShape: 'block',
      wallHeight: KERB_HEIGHT,
      wallInset: 0.02,
      floorInset: 0.035,
      floorThickness: 0.14,
      jitter: FLOOR_JITTER,
      rimColor: palette.skyHorizon,
      rimStrength: 0.09,
      seed: ctx.seed,
      name: 'ivyGateCells',
    },
  );
  w.add(field.group);
  w.own(field);
  w.time({ setTime: (t) => field.setTime(t) });

  const dividerCells: GridCell[] = [];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isDividerWall(grid, x, y)) dividerCells.push({ x, y });
    }
  }
  const dividerMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.93,
    rimColor: palette.skyHorizon,
    rimStrength: 0.13,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.22,
    name: 'ivyGateDividerMaterial',
  });
  w.own({ dispose: () => dividerMaterial.dispose() });
  if (dividerCells.length > 0) {
    const geometry = makeDividerGeometry(skin);
    w.own({ dispose: () => geometry.dispose() });
    const mesh = new THREE.InstancedMesh(geometry, dividerMaterial, dividerCells.length);
    mesh.name = 'ivyGateDivider';
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(dividerCells.length * 3),
      3,
    );
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    const tint = new THREE.Color();
    const world: Vec3Tuple = [0, 0, 0];
    for (let i = 0; i < dividerCells.length; i++) {
      const cell = dividerCells[i] as GridCell;
      collapseWallInstance(field, cell.x, cell.y);
      cellToWorld(cell.x, cell.y, grid, world);
      const n = hash2f(cell.x, cell.y, ctx.seed ^ 0x1d7a);
      position.set(world[0], 0, world[2]);
      euler.set(0, yaw + (n > 0.5 ? Math.PI : 0), 0);
      quaternion.setFromEuler(euler);
      scale.set(1, 1 + (n - 0.5) * 0.06, 1);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);
      const tone = 0.9 + hash2f(cell.x, cell.y, ctx.seed ^ 0x6b02) * 0.2;
      tint.setRGB(tone, tone, tone);
      mesh.setColorAt(i, tint);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    w.add(mesh);
    w.own({ dispose: () => mesh.dispose() });
  }

  const noPlanting = excludeAny(excludeWallCells(grid), excludeCellCenters(grid, 0.3));
  const wild = sideBounds(grid, door, SIDE_WEST);
  const formal = sideBounds(grid, door, SIDE_EAST);
  const planting: VegetationHandle[] = [];

  const wildGrass = scatterGrass({
    count: 420,
    area: { x: wild.cx, z: wild.cz, halfX: wild.halfX, halfZ: wild.halfZ },
    seed: ctx.seed ^ 0x4c17,
    exclude: noPlanting,
    color: skin.wildGrass,
    scale: [0.34, 0.7],
    stretch: [0.85, 1.5],
    hueJitter: 0.24,
    wind: 0.05,
    windFrequency: 1.6,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    name: 'wildGrass',
  });
  w.add(wildGrass.group);
  w.time(w.own(wildGrass));
  planting.push(wildGrass);

  const wildBlooms = scatterFlowers({
    count: 74,
    area: { x: wild.cx, z: wild.cz, halfX: wild.halfX, halfZ: wild.halfZ },
    seed: ctx.seed ^ 0x2b93,
    exclude: noPlanting,
    color: skin.wildBloom,
    scale: [0.45, 0.9],
    stretch: [0.8, 1.2],
    hueJitter: 0.24,
    wind: 0.032,
    windFrequency: 1.4,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    name: 'wildBlooms',
  });
  w.add(wildBlooms.group);
  w.time(w.own(wildBlooms));
  planting.push(wildBlooms);

  const gravel = scatterRocks({
    count: 130,
    area: { x: formal.cx, z: formal.cz, halfX: formal.halfX, halfZ: formal.halfZ },
    seed: ctx.seed ^ 0x7f41,
    exclude: noPlanting,
    color: shade(skin.gravel, 0.18),
    scale: [0.06, 0.13],
    stretch: [0.12, 0.24],
    hueJitter: 0.2,
    rimColor: palette.skyHorizon,
    rimStrength: 0.24,
    castShadow: false,
    name: 'formalGravel',
  });
  w.add(gravel.group);
  w.time(w.own(gravel));
  planting.push(gravel);

  const topiary = scatterRocks({
    count: 26,
    area: { x: formal.cx, z: formal.cz, halfX: formal.halfX, halfZ: formal.halfZ },
    seed: ctx.seed ^ 0x39d5,
    exclude: noPlanting,
    color: skin.topiary,
    scale: [0.16, 0.34],
    stretch: [0.75, 1.15],
    hueJitter: 0.16,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    castShadow: true,
    name: 'formalTopiary',
  });
  w.add(topiary.group);
  w.time(w.own(topiary));
  planting.push(topiary);

  const offGarden = excludeRect(0, 0, GARDEN_HALF_X, GARDEN_HALF_Z, 0.9);
  const nearCamera = excludeRect(
    Math.sin(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    Math.cos(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    4.6,
    4.6,
    0,
  );
  const outside = excludeAny(offGarden, nearCamera);

  const meadowGrass = scatterGrass({
    count: 380,
    area: { halfX: 17, halfZ: 17 },
    seed: ctx.seed ^ 0x6d0b,
    exclude: outside,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.85,
    color: blendTokens(palette.groundB, palette.sun, 0.22),
    scale: [0.42, 0.85],
    stretch: [0.7, 1.05],
    hueJitter: 0.3,
    wind: 0.07,
    windFrequency: 1.25,
    rimColor: palette.skyHorizon,
    rimStrength: 0.44,
    castShadow: false,
    name: 'meadowGrass',
  });
  w.add(meadowGrass.group);
  w.time(w.own(meadowGrass));
  planting.push(meadowGrass);

  const verge = scatterFlowers({
    count: 105,
    area: { halfX: GARDEN_HALF_X + 1.5, halfZ: GARDEN_HALF_Z + 1.5 },
    seed: ctx.seed ^ 0x58ba,
    exclude: excludeAny(excludeRect(0, 0, GARDEN_HALF_X, GARDEN_HALF_Z, 0.08), nearCamera),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    color: shade(blendTokens(skin.wildBloom, palette.groundB, 0.5), 0.2),
    scale: [0.34, 0.62],
    stretch: [0.9, 1.5],
    hueJitter: 0.34,
    wind: 0.05,
    windFrequency: 1.3,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    castShadow: false,
    name: 'wallVerge',
  });
  w.add(verge.group);
  w.time(w.own(verge));
  planting.push(verge);

  const treeLine = scatterTrees({
    archetypes: ['broadleaf', 'conifer'],
    count: 58,
    area: { halfX: 17.5, halfZ: 17.5, innerRadius: 7.6 },
    seed: ctx.seed ^ 0x1a70,
    exclude: outside,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.7,
    color: skin.tree,
    scale: [0.8, 1.5],
    stretch: [0.9, 1.4],
    hueJitter: 0.26,
    wind: 0.03,
    windFrequency: 0.8,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    castShadow: true,
    name: 'treeLine',
  });
  w.add(treeLine.group);
  w.time(w.own(treeLine));
  planting.push(treeLine);

  const wildMotes = makeAmbientParticles({
    preset: 'pollen',
    count: 190,
    bounds: { x: wild.cx, y: 0.95, z: wild.cz, halfX: wild.halfX, halfY: 0.85, halfZ: wild.halfZ },
    seed: ctx.seed ^ 0x7d21,
    color: skin.motesWild,
    size: 1.05,
    intensity: 1.7,
    opacity: 0.6,
    rise: 0.07,
    swirl: 0.26,
    name: 'wildPollen',
  });
  w.add(wildMotes.points);
  w.size(w.time(w.own(wildMotes)));

  const formalMotes = makeAmbientParticles({
    preset: 'dust',
    count: 150,
    bounds: { x: formal.cx, y: 0.8, z: formal.cz, halfX: formal.halfX, halfY: 0.72, halfZ: formal.halfZ },
    seed: ctx.seed ^ 0x51c3,
    color: skin.motesFormal,
    size: 0.72,
    intensity: 1.3,
    opacity: 0.44,
    rise: 0.03,
    swirl: 0.16,
    name: 'formalDust',
  });
  w.add(formalMotes.points);
  w.size(w.time(w.own(formalMotes)));

  const mist = makeMistPlanes({
    size: [38, 36],
    color: blendTokens(palette.fog, palette.sun, 0.3),
    seed: ctx.seed ^ 0x3ac1,
    layers: 2,
    baseY: 1.75,
    spread: 1.1,
    opacity: 0.05,
    speed: 0.008,
    scale: 2.0,
    nearFade: 9,
    name: 'morningAir',
  });
  w.add(mist.group);
  w.time(w.own(mist));

  const restoreColor = new THREE.Color();

  return w.seal({
    sky,
    rig,
    terrain,
    field,
    door,
    gateYaw: yaw,
    dividerCells,
    paintFloor(x, y, color) {
      field.setFloorColor(x, y, color);
    },
    restoreFloorTint(x, y) {
      const n = hash2f(x, y, ctx.seed);
      restoreColor.copy(floorTintFor(x, y)).multiplyScalar(1 + (n - 0.5) * FLOOR_JITTER);
      field.setFloorColor(x, y, restoreColor);
    },
  }, {
    drop: () => {
      planting.length = 0;
    },
  });
}

export function makeDividerGeometry(skin: GardenSkin): THREE.BufferGeometry {
  const detail = 1;
  const parts: ColoredPart[] = [];
  const size = CELL - 0.03;

  const courses = 3;
  for (let i = 0; i < courses; i++) {
    const y0 = (i / courses) * DIVIDER_HEIGHT;
    const y1 = ((i + 1) / courses) * DIVIDER_HEIGHT;
    const inset = i === courses - 1 ? 0.035 : i === 0 ? 0 : 0.018;
    const course = new THREE.BoxGeometry(size - inset * 2, y1 - y0 + 0.004, DIVIDER_THICKNESS - inset * 2);
    course.translate(0, (y0 + y1) * 0.5, 0);
    parts.push({
      geometry: course,
      color: i % 2 === 0 ? skin.divider : blendTokens(skin.divider, skin.dividerCap, 0.16),
    });
  }

  const cap = new THREE.BoxGeometry(size + 0.06, DIVIDER_CAP, DIVIDER_THICKNESS + 0.13);
  cap.translate(0, DIVIDER_HEIGHT + DIVIDER_CAP * 0.5, 0);
  parts.push({ geometry: cap, color: skin.dividerCap });

  const footing = new THREE.BoxGeometry(size, 0.13, DIVIDER_THICKNESS + 0.2);
  footing.translate(0, 0.065, 0);
  parts.push({ geometry: footing, color: skin.dividerCap });

  const rows = 2;
  const perRow = Math.max(5, Math.round(9 * detail));
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < perRow; i++) {
      const t = (i + (r % 2) * 0.5) / perRow;
      const lean = Math.sin(t * 7.3 + r * 2.1) * 0.34;
      const leaf = ivyLeafGeometry();
      const sc = 0.82 + 0.32 * Math.sin(t * 4.1 + r * 1.7);
      leaf.scale(sc, sc, sc);
      leaf.rotateX(-1.3 + Math.cos(t * 5.7 + r) * 0.4);
      leaf.rotateZ(lean);
      leaf.rotateY(lean * 2.2 + t * 4.1 + r);
      leaf.translate(
        (t - 0.5) * (size + 0.05),
        DIVIDER_HEIGHT + DIVIDER_CAP + 0.01 + 0.035 * Math.sin(t * 9.1 + r),
        -0.02 - r * 0.19 - 0.13 * Math.abs(Math.sin(t * 3.3 + r)),
      );
      parts.push({ geometry: leaf, color: (i + r) % 3 === 0 ? skin.ivyDark : skin.ivy });
    }
  }

  const runners = 2;
  const leavesPer = Math.max(3, Math.round(8 * detail));
  for (let r = 0; r < runners; r++) {
    const x0 = r === 0 ? -0.24 : 0.26;
    for (let i = 0; i < leavesPer; i++) {
      const t = i / (leavesPer - 1 || 1);
      const y = DIVIDER_HEIGHT + DIVIDER_CAP - 0.1 - t * (DIVIDER_HEIGHT * 0.78);
      const sway = Math.sin(t * 5.1 + r * 2.3) * 0.11;
      const s = 1 - 0.42 * t;
      const leaf = ivyLeafGeometry();
      leaf.scale(s, s, s);
      leaf.rotateX(-0.35 - t * 0.4);
      leaf.rotateZ(sway * 2.4);
      leaf.rotateY(sway * 3 + r * 1.7);
      leaf.translate(x0 + sway, y, -(DIVIDER_THICKNESS * 0.5) - 0.035);
      parts.push({ geometry: leaf, color: i % 3 === 0 ? skin.ivyDark : skin.ivy });
    }
  }

  return mergeColoredParts(parts, 'ivyWallSegment');
}

export function ivyLeafGeometry(): THREE.BufferGeometry {
  const w = 0.105;
  const l = 0.2;
  const base: readonly [number, number, number] = [0, 0, 0];
  const sh0: readonly [number, number, number] = [-w, l * 0.46, 0.012];
  const sh1: readonly [number, number, number] = [w, l * 0.46, 0.012];
  const lobe0: readonly [number, number, number] = [-w * 0.62, l * 0.84, 0.006];
  const lobe1: readonly [number, number, number] = [w * 0.62, l * 0.84, 0.006];
  const tip: readonly [number, number, number] = [0, l, 0];
  const tris: readonly (readonly (readonly [number, number, number])[])[] = [
    [base, sh1, sh0],
    [sh0, sh1, lobe1],
    [sh0, lobe1, lobe0],
    [lobe0, lobe1, tip],
  ];
  const grade: readonly number[] = [0.62, 0.86, 0.86, 0.86, 0.86, 1.02, 0.86, 1.02, 1.02, 1.02, 1.02, 1.18];
  const n = tris.length * 3;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  let o = 0;
  for (const tri of tris) {
    for (const v of tri) {
      positions[o * 3] = v[0];
      positions[o * 3 + 1] = v[1];
      positions[o * 3 + 2] = v[2];
      const k = grade[o] ?? 1;
      colors[o * 3] = k;
      colors[o * 3 + 1] = k;
      colors[o * 3 + 2] = k;
      o++;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'ivyLeaf';
  return g;
}
