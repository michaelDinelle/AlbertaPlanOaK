
import * as THREE from 'three';
import {
  CELL,
  buildCellField,
  cellToWorld,
  type CellFieldHandle,
  type GridSize,
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms, type OakMaterial } from '@kit/materials';
import { blendTokens, clamp01, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles, makeMistPlanes, type AmbientParticlesHandle, type MistPlanesHandle } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash1f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import {
  excludeAny,
  excludeBand,
  excludeRect,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
} from '@kit/vegetation';
import { makeWorld, type WorldCore } from '@kit/world';
import { DELAYEDCHAIN_END, DELAYEDCHAIN_L, delayedChainDecoder } from '@sim/envs/delayedchain';
import type { EnvGeometry, GridGeometry } from '@sim/types';
import { buildAurora, type AuroraHandle } from './aurora';


export const ROAD_HALF = (DELAYEDCHAIN_L * CELL) / 2;
export const LANTERN_Z = -0.62;
export const LAMP_REACH = 0.3;
export const ROAD_HALF_Z = 0.44;
export const ROAD_SINK = -0.05;

export const TERRAIN_SIZE: readonly [number, number] = [220, 140];
export const TERRAIN_SEGMENTS: readonly [number, number] = [176, 112];

export const FOG_EXTENT = 150;

export const MOON_DIR: readonly [number, number, number] = [-0.52, 0.62, 0.59];
export const SHADOW_RADIUS = 9;

export const AURORA_RADIUS = 26;
export const CAMERA_AZIMUTH = -1.16;
export const CAMERA_LANE_Z = 2.9;


export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === DELAYEDCHAIN_L && g.h === 1) return g;
  const own = delayedChainDecoder.geometry(delayedChainDecoder.initStatic(0), 0).grid;
  return (
    own ?? {
      w: DELAYEDCHAIN_L,
      h: 1,
      walls: new Uint8Array(DELAYEDCHAIN_L),
      goal: { x: DELAYEDCHAIN_END, y: 0 },
    }
  );
}

export function goalCellOf(grid: GridGeometry): number {
  const g = grid.goal;
  return g ? g.x : grid.w - 1;
}


export interface MoorSkin {
  peat: THREE.Color;
  moor: THREE.Color;
  flagstone: THREE.Color;
  flagstoneMoonlit: THREE.Color;
  verge: THREE.Color;
  grass: THREE.Color;
  heather: THREE.Color;
  stone: THREE.Color;
  stoneDark: THREE.Color;
  moss: THREE.Color;
  iron: THREE.Color;
  ironLight: THREE.Color;
  flame: THREE.Color;
  thorn: THREE.Color;
  mist: THREE.Color;
}

export function moorSkin(palette: Palette, tokens: typeof TOKENS): MoorSkin {
  const moon = palette.sun;
  const teal = tokens.accents['delayed-chain'] ?? tokens.uiCyan;
  const moss = tokens.accents['procedural-maze'] ?? palette.groundB;

  return {
    peat: shade(blendTokens(palette.hemiGround, palette.groundA, 0.45), 0.2),
    moor: shade(blendTokens(palette.groundA, palette.groundB, 0.45), 0.12),
    flagstone: blendTokens(palette.groundB, moon, 0.2),
    flagstoneMoonlit: blendTokens(palette.groundB, moon, 0.52),
    verge: shade(blendTokens(palette.groundA, palette.hemiGround, 0.4), 0.18),
    grass: shade(blendTokens(palette.groundB, palette.hemiSky, 0.3), 0.28),
    heather: shade(blendTokens(tokens.cueMagenta, palette.skyHorizon, 0.72), 0.14),
    stone: blendTokens(palette.groundB, moon, 0.26),
    stoneDark: shade(blendTokens(palette.groundB, palette.skyHorizon, 0.4), 0.24),
    moss: shade(blendTokens(palette.groundA, moss, 0.34), 0.3),
    iron: shade(blendTokens(palette.skyTop, palette.groundB, 0.42), 0.28),
    ironLight: blendTokens(shade(blendTokens(palette.skyTop, palette.groundB, 0.42), 0.28), palette.hemiSky, 0.4),
    flame: blendTokens(tokens.celebrationGold, moon, 0.16),
    thorn: shade(blendTokens(palette.groundA, palette.skyHorizon, 0.35), 0.42),
    mist: blendTokens(palette.fog, teal, 0.1),
  };
}


export interface MoonPool {
  cell: number;
  radius: number;
  strength: number;
}

export const MOON_POOL_COUNT = 5;

export function moonPools(seed: number, count = MOON_POOL_COUNT, length = DELAYEDCHAIN_L): MoonPool[] {
  const pools: MoonPool[] = [];
  const span = length / count;
  for (let i = 0; i < count; i++) {
    const j = hash1f(i, seed ^ 0x4d0f31);
    const k = hash1f(i, seed ^ 0x27b1a5);
    pools.push({
      cell: (i + 0.2 + 0.6 * j) * span,
      radius: 1.5 + 1.6 * k,
      strength: 0.55 + 0.45 * hash1f(i, seed ^ 0x6f2d19),
    });
  }
  return pools;
}

export function moonlightAt(cell: number, pools: readonly MoonPool[]): number {
  let m = 0;
  for (let i = 0; i < pools.length; i++) {
    const p = pools[i] as MoonPool;
    const d = Math.abs(cell - p.cell) / Math.max(0.001, p.radius);
    if (d >= 1) continue;
    const f = 1 - d * d;
    m += p.strength * f * f;
  }
  return clamp01(m);
}


export function makeMoorHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.abs(z) - (ROAD_HALF_Z + 0.55);
    const w = smoothstep01(out / 3.2);
    if (w <= 0) return 0;
    const n1 = valueNoise2(x * 0.035 + 3.7, z * 0.035 + 8.1, 128, seed) - 0.5;
    const n2 = valueNoise2(x * 0.11 + 1.3, z * 0.11 + 4.9, 128, seed ^ 0x51ed270b) - 0.5;
    const n3 = valueNoise2(x * 0.31 + 6.1, z * 0.31 + 2.3, 128, seed ^ 0x1b873593) - 0.5;
    const rolling = n1 * 3.4 + n2 * 1.1 + n3 * 0.28;
    const far = Math.max(0, (Math.max(Math.abs(x) - ROAD_HALF, Math.abs(z)) - 34) / 46);
    const ridge = smoothstep01(far) * (2.6 + 2.2 * (valueNoise2(x * 0.02 + 11.3, z * 0.02 + 5.5, 128, seed) - 0.5));
    return w * (rolling + ridge);
  };
}


export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  aurora: AuroraHandle;
  skin: MoorSkin;
  pools: readonly MoonPool[];
}


export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.night;
  const skin = moorSkin(palette, ctx.tokens);
  const w = makeWorld('longNightRoad');

  const sky = skyFromPalette(palette, 'night', {
    sunDir: MOON_DIR,
    sunSize: 0.022,
    sunHdr: 3.2,
    sunGlow: 0.16,
    sunGlowPower: 42,
    topSpread: 0.62,
    bottomSpread: 0.42,
    stars: { intensity: 2.6, density: 190, threshold: 0.972, twinkle: 0.45, color: palette.sun },
    radius: 420,
    name: 'nightSky',
  });
  w.add(sky.mesh);
  w.own(sky);
  w.time({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 9,
    sunDistanceScale: 2.2,
  });
  w.add(rig.group);
  w.own(rig);
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, SHADOW_RADIUS);

  applyFog(scene, palette, FOG_EXTENT);

  const terrain = makeTerrain({
    size: TERRAIN_SIZE,
    segments: TERRAIN_SEGMENTS,
    seed: ctx.seed,
    heightFn: makeMoorHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.07,
    slopeColor: skin.peat,
    slopeStrength: 0.55,
    slopeSharpness: 2.4,
    flattenMask: { x: 0, z: 0, halfX: ROAD_HALF + 6, halfZ: ROAD_HALF_Z + 0.55, feather: 3.2 },
    flattenHeight: ROAD_SINK,
    colorRamp: [
      { t: 0, color: shade(skin.peat, 0.18) },
      { t: 0.4, color: skin.peat },
      { t: 0.72, color: skin.moor },
      { t: 1, color: blendTokens(skin.moor, palette.hemiSky, 0.28) },
    ],
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.22,
      toonSteps: 3,
      toonSoftness: 0.32,
      toonFloor: 0.16,
    },
    receiveShadow: true,
    castShadow: false,
    name: 'moorTerrain',
  });
  w.add(terrain.mesh);
  w.own(terrain);

  const pools = moonPools(ctx.seed, MOON_POOL_COUNT, grid.w);
  const field = buildCellField(
    grid,
    {
      floorColor: palette.sun,
      wallColor: palette.sun,
      floorTint: (x) => {
        const m = moonlightAt(x, pools);
        const c = skin.flagstone.clone().lerp(skin.flagstoneMoonlit, m);
        return x % 8 === 3 ? c.lerp(skin.stoneDark, 0.22) : c;
      },
      wallShape: 'block',
      floorInset: 0.06,
      floorThickness: 0.1,
      jitter: 0.13,
      rimColor: palette.hemiSky,
      rimStrength: 0.24,
      seed: ctx.seed,
      name: 'roadFlagstones',
    },
  );
  w.add(field.group);
  w.own(field);
  w.time({ setTime: (t) => field.setTime(t) });

  {
    const parts: ColoredPart[] = [];
    const length = grid.w * CELL + 1.2;
    for (const side of [-1, 1]) {
      const strip = new THREE.BoxGeometry(length, 0.08, 0.34);
      strip.translate(0, -0.03, side * (ROAD_HALF_Z + 0.19));
      parts.push({ geometry: strip, color: side < 0 ? skin.verge : blendTokens(skin.verge, skin.peat, 0.3) });
    }
    const geometry = mergeColoredParts(parts, 'roadVerge');
    const material = oakMaterial({
      vertexColors: true,
      flatShading: true,
      roughness: 0.96,
      rimColor: palette.hemiSky,
      rimStrength: 0.16,
      toonSteps: 2,
      toonSoftness: 0.34,
      toonFloor: 0.2,
      name: 'roadVergeMaterial',
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'roadVerge';
    mesh.receiveShadow = true;
    w.add(mesh);
    w.own({
      dispose: () => {
        geometry.dispose();
        material.dispose();
      },
    });
  }

  const aurora = buildAurora({
    palette,
    tokens: ctx.tokens,
    center: cellToWorld(goalCellOf(grid), 0, grid)[0],
    radius: AURORA_RADIUS,
    name: 'endAurora',
  });
  w.add(aurora.group);
  w.own(aurora);
  w.time({ setTime: (t) => aurora.setTime(t) });

  const offRoad: ExcludeFn = excludeBand(0, ROAD_HALF_Z + 0.42, 0.24);
  const nearCamera: ExcludeFn = excludeRect(0, CAMERA_LANE_Z, ROAD_HALF + 12, 1.9, 0.4);

  const grassField = scatterGrass({
    count: 2600,
    area: { x: 0, z: 0, halfX: ROAD_HALF + 14, halfZ: 22 },
    seed: ctx.seed ^ 0x5c3a,
    exclude: offRoad,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.85,
    color: skin.grass,
    scale: [0.42, 0.95],
    stretch: [0.9, 1.7],
    hueJitter: 0.26,
    wind: 0.07,
    windFrequency: 1.15,
    rimColor: palette.hemiSky,
    rimStrength: 0.42,
    castShadow: false,
    name: 'moorGrass',
  });
  w.add(grassField.group);
  w.time(w.own(grassField));

  const heather = scatterFlowers({
    count: 760,
    area: { x: 0, z: 0, halfX: ROAD_HALF + 10, halfZ: 15 },
    seed: ctx.seed ^ 0x71bd,
    exclude: offRoad,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.8,
    color: skin.heather,
    scale: [0.3, 0.62],
    stretch: [0.6, 1.05],
    hueJitter: 0.3,
    wind: 0.045,
    windFrequency: 1.35,
    rimColor: palette.accentLight,
    rimStrength: 0.26,
    castShadow: false,
    name: 'moorHeather',
  });
  w.add(heather.group);
  w.time(w.own(heather));

  const stones = scatterRocks({
    count: 260,
    area: { x: 0, z: 0, halfX: ROAD_HALF + 16, halfZ: 24 },
    seed: ctx.seed ^ 0x3e91,
    exclude: excludeAny(offRoad, nearCamera),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.9,
    color: skin.stoneDark,
    scale: [0.16, 0.62],
    stretch: [0.35, 0.8],
    hueJitter: 0.22,
    rimColor: palette.hemiSky,
    rimStrength: 0.36,
    castShadow: true,
    name: 'moorStones',
  });
  w.add(stones.group);
  w.time(w.own(stones));

  const thorns = scatterTrees({
    archetypes: ['bare'],
    count: 90,
    area: { x: 0, z: 0, halfX: ROAD_HALF + 22, halfZ: 30, innerRadius: 0 },
    seed: ctx.seed ^ 0x18f7,
    exclude: excludeAny(excludeBand(0, 7.5, 1.6), nearCamera),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.7,
    color: skin.thorn,
    scale: [0.55, 1.15],
    stretch: [0.7, 1.25],
    hueJitter: 0.2,
    wind: 0.035,
    windFrequency: 0.7,
    rimColor: palette.skyHorizon,
    rimStrength: 0.4,
    castShadow: false,
    name: 'moorThorns',
  });
  w.add(thorns.group);
  w.time(w.own(thorns));

  const mist: MistPlanesHandle = makeMistPlanes({
    size: [110, 54],
    color: skin.mist,
    seed: ctx.seed ^ 0x2f18,
    layers: 5,
    baseY: 0.06,
    spread: 1.05,
    opacity: 0.15,
    speed: 0.006,
    scale: 3.2,
    nearFade: 6,
    name: 'moorMist',
  });
  w.add(mist.group);
  w.time(w.own(mist));

  const motes: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'dust',
    count: 1200,
    bounds: { x: 0, y: 1.15, z: -1.5, halfX: ROAD_HALF + 6, halfY: 1.1, halfZ: 5.2 },
    seed: ctx.seed ^ 0x6a21,
    color: blendTokens(palette.accentLight, skin.mist, 0.4),
    size: 0.42,
    intensity: 0.95,
    opacity: 0.34,
    rise: 0.03,
    swirl: 0.24,
    name: 'moorMotes',
  });
  w.add(motes.points);
  w.size(w.time(w.own(motes)));

  return w.seal({
    sky,
    rig,
    terrain,
    field,
    aurora,
    skin,
    pools,
  });
}


export function propMaterial(palette: Palette, name: string): OakMaterial {
  return oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.92,
    rimColor: palette.hemiSky,
    rimStrength: 0.42,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.18,
    name,
  });
}

export function setPropTime(material: THREE.Material | null, t: number): void {
  oakUniforms(material)?.setTime(t);
}

export function cellX(cell: number, grid: GridSize, out?: Vec3Tuple): number {
  return cellToWorld(cell, 0, grid, out)[0];
}
