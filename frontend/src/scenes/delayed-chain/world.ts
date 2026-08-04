// The Long Night Road — the static world (DelayedChain-cont-v0).
//
// THE ONE IDEA: DISTANCE IS THE SUBJECT. DelayedChain pays +1 at pos 31 and nowhere
// else, so the art has to make thirty-one dark cells FELT: a bare moorland path of 32
// worn flagstones, one unlit lantern post per cell, and an aurora that hangs over the
// far end alone. Everything bright in this scene is either the wisp (the agent) or the
// reward's own end of the road; the middle of the walk is deliberately, honestly empty.
//
// PATTERNS COPIED FROM THE TWO EXEMPLARS
//  * `buildCellField` is the substrate (four-rooms lesson 13): white base material +
//    per-cell `floorTint`, so the moonlight pools cost nothing but an instance color.
//  * All decorative height is multiplied by a weight that is IDENTICALLY ZERO over the
//    road band (mountain-car lesson 3) — the flagstones sit on a mathematically flat
//    strip, and `makeTerrain`'s flatten mask has nothing to fight.
//  * Nothing lives between the camera and the physics: the camera stands on the +z side
//    of the road, so the lantern posts are planted on the −z side and the tree line is
//    excluded from the camera's own lane.
//  * Night is a DARK palette, so emissive gain is palette-relative (four-rooms lesson
//    22): the lantern flame runs hotter here than the same flame would at noon.

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
import { makeAmbientParticles, makeMistPlanes, type AmbientParticlesHandle, type MistPlanesHandle } from '@kit/particles';
import { hash1f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import {
  excludeAny,
  excludeBand,
  excludeRect,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
  type VegetationHandle,
} from '@kit/vegetation';
import { DELAYEDCHAIN_END, DELAYEDCHAIN_L, delayedChainDecoder } from '@sim/envs/delayedchain';
import type { EnvGeometry, GridGeometry } from '@sim/types';
import { buildAurora, type AuroraHandle } from './aurora';

// ---------------------------------------------------------------------------
// world constants (1 cell = 1 world unit; the road really is 32 units long)
// ---------------------------------------------------------------------------

/** half the road's footprint along x, derived from the env's own corridor length */
export const ROAD_HALF = (DELAYEDCHAIN_L * CELL) / 2;
/** the lantern line: the FAR side of the road, so no post ever stands in front of the wisp */
export const LANTERN_Z = -0.62;
/** the lamp hangs this far back toward the road from its post */
export const LAMP_REACH = 0.3;
/** flagstone half-width; the verge starts here */
export const ROAD_HALF_Z = 0.44;
/** the moor sits a few centimetres under the flagstones, so the joints read as peat */
export const ROAD_SINK = -0.05;

export const TERRAIN_SIZE: readonly [number, number] = [220, 140];
export const TERRAIN_SEGMENTS: readonly [number, number] = [176, 112];

/**
 * Fog reach. `applyFog` turns extent into density through `fogDensityHint` (night =
 * 0.6), so 150 puts full density ~90 units out: the far end of the road, 32 units from
 * the start, sits about a third hazed — enough that the walk reads as LONG and the
 * terrain's own edge is invisible, while the lantern flames (emissive, unfogged) still
 * punch through the haze when the wave lights them.
 */
export const FOG_EXTENT = 150;

/**
 * The moon: high (38°) and BEHIND the camera's shoulder, north-west of the road. Two
 * reasons, both measured. A low night key rakes a 1.75-unit lantern post into a 4-unit
 * shadow that swallows three cells (mountain-car lesson 8), and a moon placed down-road
 * would backlight the flagstones into silhouette — this scene needs the path readable
 * and the SKY ahead free for the aurora.
 */
export const MOON_DIR: readonly [number, number, number] = [-0.52, 0.62, 0.59];
/** the shadow frustum follows the wisp; this is its radius (see index.ts) */
export const SHADOW_RADIUS = 9;

/**
 * The aurora's arc is centred on the road's END and stands this far beyond it. Brought in
 * from 34 with the arc narrowed to 0.6 rad (see aurora.ts): a narrow arc held at the old
 * radius is a bright STRIP on the horizon 50 units past the cairn, while the same arc at
 * 26 hangs over the terminus with the road running under it — the curtain is the far end's
 * own sky, so it has to be the size of a sky over that end and not a distant wall.
 */
export const AURORA_RADIUS = 26;
/** the first cell the aurora is allowed to reach over — the brief's "cells ~24-31" */
export const AURORA_FIRST_CELL = 24;

/** the camera's azimuth, shared with index.ts so planting and framing never disagree */
export const CAMERA_AZIMUTH = -1.16;
/** the +z lane the camera travels down — nothing tall may be planted in it */
export const CAMERA_LANE_Z = 2.9;

// ---------------------------------------------------------------------------
// small shared helpers (actors.ts and aurora.ts import these)
// ---------------------------------------------------------------------------

export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** A scene-local color: STYLE.md allows only BLENDS of token colors. */
export function blendTokens(
  a: THREE.ColorRepresentation,
  b: THREE.ColorRepresentation,
  k: number,
): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), k);
}

/**
 * Take a blend down into shade. `night.skyTop` is the darkest token there is and
 * STYLE.md names it as the legal way to reach near-black, so "shade" is a blend like
 * every other color here.
 */
export function shade(color: THREE.ColorRepresentation, k: number): THREE.Color {
  return blendTokens(color, TOKENS.palettes.night.skyTop, k);
}

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

/**
 * Merge pre-transformed parts into ONE geometry carrying a flat per-part vertex color,
 * disposing the sources. A lantern is five little primitives and one draw call.
 */
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

// ---------------------------------------------------------------------------
// the env's own truth
// ---------------------------------------------------------------------------

/**
 * The corridor the scene draws. The host hands us the decoder's geometry; the fallback
 * is the DECODER'S OWN geometry builder, never a scene re-derivation, so a placeholder
 * degrades to the correct 32×1 road instead of a guess.
 */
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

/** The paying cell, read from the env geometry (never the constant 31 typed inline). */
export function goalCellOf(grid: GridGeometry): number {
  const g = grid.goal;
  return g ? g.x : grid.w - 1;
}

// ---------------------------------------------------------------------------
// the skin (every color a blend of tokens — STYLE.md)
// ---------------------------------------------------------------------------

export interface MoorSkin {
  /** peat under the heather */
  peat: THREE.Color;
  /** the rolling moor top */
  moor: THREE.Color;
  /** worn flagstone, unlit */
  flagstone: THREE.Color;
  /** the same stone standing in a pool of moonlight */
  flagstoneMoonlit: THREE.Color;
  /** packed-earth verge either side of the path */
  verge: THREE.Color;
  /** wind-bleached moor grass */
  grass: THREE.Color;
  /** heather in flower — a dusky violet, the moor's only hue */
  heather: THREE.Color;
  /** wet granite: milestone, cairn, moor stones */
  stone: THREE.Color;
  stoneDark: THREE.Color;
  /** moss on the milestone */
  moss: THREE.Color;
  /** cold wrought iron: the lantern posts */
  iron: THREE.Color;
  ironLight: THREE.Color;
  /** the flame the wave lights — warm gold against a cold moor */
  flame: THREE.Color;
  /** bare windblown thorn trees on the horizon */
  thorn: THREE.Color;
  /** ground mist */
  mist: THREE.Color;
}

export function moorSkin(palette: Palette, tokens: typeof TOKENS): MoorSkin {
  const moon = palette.sun;
  const teal = tokens.accents['delayed-chain'] ?? tokens.uiCyan;
  const moss = tokens.accents['procedural-maze'] ?? palette.groundB;

  return {
    peat: shade(blendTokens(palette.hemiGround, palette.groundA, 0.45), 0.2),
    moor: shade(blendTokens(palette.groundA, palette.groundB, 0.45), 0.12),
    // Night ground albedo is the mirror of the noon albedo law (four-rooms lesson 19):
    // under a 0.5 key a stone shaded to 40 sRGB is a black hole on screen, so the
    // flagstones sit BRIGHT for a night palette and still never approach bloom.
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

// ---------------------------------------------------------------------------
// moonlight pools — the only bright thing on the road that is not the agent
// ---------------------------------------------------------------------------

export interface MoonPool {
  /** cell index the pool is centred on (fractional) */
  cell: number;
  /** reach in cells */
  radius: number;
  strength: number;
}

/** hero 5 pools / medium 4 / thumb 3 — a break in the cloud every seven-ish cells. */
export const MOON_POOL_COUNT = 5;

/**
 * Deterministic breaks in the cloud, spread down the road with a seeded offset. Pure
 * geometry of the LANE (a stationary function of cell index), never of the agent's
 * history — the same seed always lights the same stones.
 */
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

/** Moonlight at a cell in [0, 1] — a smooth sum over the pools, clamped. */
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

// ---------------------------------------------------------------------------
// the moor height field
// ---------------------------------------------------------------------------

/**
 * Rolling moor everywhere, identically flat over the road. `w` is 0 for every point
 * within one cell of the path, so the flagstones stand on an exactly level strip and
 * `sampleHeight` along the road is exactly `ROAD_SINK` (asserted in the test).
 */
export function makeMoorHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.abs(z) - (ROAD_HALF_Z + 0.55);
    const w = smoothstep01(out / 3.2);
    if (w <= 0) return 0;
    const n1 = valueNoise2(x * 0.035 + 3.7, z * 0.035 + 8.1, 128, seed) - 0.5;
    const n2 = valueNoise2(x * 0.11 + 1.3, z * 0.11 + 4.9, 128, seed ^ 0x51ed270b) - 0.5;
    const n3 = valueNoise2(x * 0.31 + 6.1, z * 0.31 + 2.3, 128, seed ^ 0x1b873593) - 0.5;
    const rolling = n1 * 3.4 + n2 * 1.1 + n3 * 0.28;
    // a far ridge line so the moor has a horizon instead of an edge
    const far = Math.max(0, (Math.max(Math.abs(x) - ROAD_HALF, Math.abs(z)) - 34) / 46);
    const ridge = smoothstep01(far) * (2.6 + 2.2 * (valueNoise2(x * 0.02 + 11.3, z * 0.02 + 5.5, 128, seed) - 0.5));
    return w * (rolling + ridge);
  };
}

// ---------------------------------------------------------------------------
// world handle
// ---------------------------------------------------------------------------

export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  aurora: AuroraHandle;
  skin: MoorSkin;
  pools: readonly MoonPool[];
  /** ctx.time() — mist, grass, aurora and flagstone rims all move on the one clock */
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
  const palette: Palette = ctx.tokens.palettes.night;
  const skin = moorSkin(palette, ctx.tokens);
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'longNightRoad';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];

  // ---- sky (exactly one): a night gradient with the token star field ----
  const sky = skyFromPalette(palette, 'night', tier, {
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
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the light budget: moon + hemi + 4 points (wisp / wave front / goal / aurora) ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 9,
    sunDistanceScale: 2.2,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, SHADOW_RADIUS);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the moor ----
  const terrain = makeTerrain({
    size: TERRAIN_SIZE,
    segments: TERRAIN_SEGMENTS,
    tier,
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
    receiveShadow: shadows,
    castShadow: false,
    name: 'moorTerrain',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  // ---- the path: 32 flagstones, one per cell, straight off `grid.walls` ----
  // White base + full-color tint (four-rooms lesson 13): the moonlight pools are an
  // instance color, so the whole road is ONE draw call however the cloud breaks.
  const pools = moonPools(ctx.seed, tier === 'thumb' ? 3 : tier === 'medium' ? 4 : MOON_POOL_COUNT, grid.w);
  const field = buildCellField(
    grid,
    {
      floorColor: palette.sun,
      wallColor: palette.sun,
      floorTint: (x) => {
        const m = moonlightAt(x, pools);
        const c = skin.flagstone.clone().lerp(skin.flagstoneMoonlit, m);
        // every eighth stone is a darker slab, so the path reads as laid stone, not tape
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
    tier,
  );
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  // ---- the verges: two packed-earth strips framing the path, one draw ----
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
      tier,
      name: 'roadVergeMaterial',
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'roadVerge';
    mesh.receiveShadow = shadows;
    group.add(mesh);
    owned.push({
      dispose: () => {
        geometry.dispose();
        material.dispose();
      },
    });
  }

  // ---- the aurora: the ONE bespoke ShaderMaterial, over the far end alone ----
  const aurora = buildAurora({
    tier,
    palette,
    tokens: ctx.tokens,
    center: cellToWorld(goalCellOf(grid), 0, grid)[0],
    radius: AURORA_RADIUS,
    name: 'endAurora',
  });
  group.add(aurora.group);
  owned.push(aurora);
  timed.push({ setTime: (t) => aurora.setTime(t) });

  // ---- planting: nothing on the path, nothing tall in the camera's lane ----
  const offRoad: ExcludeFn = excludeBand(0, ROAD_HALF_Z + 0.42, 0.24);
  const nearCamera: ExcludeFn = excludeRect(0, CAMERA_LANE_Z, ROAD_HALF + 12, 1.9, 0.4);

  const grassField = scatterGrass({
    count: 2600,
    area: { x: 0, z: 0, halfX: ROAD_HALF + 14, halfZ: 22 },
    seed: ctx.seed ^ 0x5c3a,
    tier,
    exclude: offRoad,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.85,
    color: skin.grass,
    scale: [0.42, 0.95],
    stretch: [0.9, 1.7],
    hueJitter: 0.26,
    // "breath of wind in the grass" — the moor is never still
    wind: 0.07,
    windFrequency: 1.15,
    rimColor: palette.hemiSky,
    rimStrength: 0.42,
    castShadow: false,
    name: 'moorGrass',
  });
  group.add(grassField.group);
  owned.push(grassField);
  timed.push(grassField);

  const heather = scatterFlowers({
    count: 760,
    area: { x: 0, z: 0, halfX: ROAD_HALF + 10, halfZ: 15 },
    seed: ctx.seed ^ 0x71bd,
    tier,
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
    // Under a 0.5 night key a rim-dominated flower is a PALE SPECK: at 0.5 the heather
    // read as a field of little white mushrooms rather than as the moor's one hue. 0.26
    // lets the dusky violet win and the rim only catch the tops.
    rimStrength: 0.26,
    castShadow: false,
    name: 'moorHeather',
  });
  group.add(heather.group);
  owned.push(heather);
  timed.push(heather);

  const stones = scatterRocks({
    count: 260,
    area: { x: 0, z: 0, halfX: ROAD_HALF + 16, halfZ: 24 },
    seed: ctx.seed ^ 0x3e91,
    tier,
    exclude: excludeAny(offRoad, nearCamera),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.9,
    color: skin.stoneDark,
    scale: [0.16, 0.62],
    stretch: [0.35, 0.8],
    hueJitter: 0.22,
    rimColor: palette.hemiSky,
    rimStrength: 0.36,
    castShadow: shadows,
    name: 'moorStones',
  });
  group.add(stones.group);
  owned.push(stones);
  timed.push(stones);

  // A thin line of wind-killed thorns on the horizon: the moor needs a silhouette, and
  // bare archetypes at night are pure shape. Kept out of the camera lane AND off the
  // near verge, so they never cross the road.
  const thorns = scatterTrees({
    archetypes: ['bare'],
    count: 90,
    area: { x: 0, z: 0, halfX: ROAD_HALF + 22, halfZ: 30, innerRadius: 0 },
    seed: ctx.seed ^ 0x18f7,
    tier,
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
  group.add(thorns.group);
  owned.push(thorns);
  timed.push(thorns);

  // ---- ground mist + night air ----
  const mist: MistPlanesHandle = makeMistPlanes({
    size: [110, 54],
    color: skin.mist,
    seed: ctx.seed ^ 0x2f18,
    tier,
    layers: 5,
    baseY: 0.06,
    spread: 1.05,
    opacity: 0.15,
    speed: 0.006,
    scale: 3.2,
    nearFade: 6,
    name: 'moorMist',
  });
  group.add(mist.group);
  owned.push(mist);
  timed.push(mist);

  // Dust motes, not fireflies: the night moor is empty, and the specks only read at all
  // because the wisp is dragging its own light through them.
  //
  // TWO MEASURED CORRECTIONS, both canon lesson 5 in its sharpest form. (a) SIZE. The kit
  // sizes a point as `uSize · 60/dist` screen px, so a field the CAMERA STANDS INSIDE
  // renders its nearest members at ~100 px: the first pass at 1.6 filled the road picture
  // with soft bokeh discs that read as a dirty lens, not as night air. 0.42 puts a mote at
  // 5 px a metre away and one pixel at the far end — dust. (b) PLACEMENT. The camera walks
  // the +z lane, so the field is pushed onto the −z side of the road: the motes drift
  // BETWEEN the wisp and the lantern line, where its light can find them, instead of
  // swimming across the lens.
  const motes: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'dust',
    count: 1200,
    bounds: { x: 0, y: 1.15, z: -1.5, halfX: ROAD_HALF + 6, halfY: 1.1, halfZ: 5.2 },
    seed: ctx.seed ^ 0x6a21,
    tier,
    color: blendTokens(palette.accentLight, skin.mist, 0.4),
    size: 0.42,
    intensity: 0.95,
    opacity: 0.34,
    rise: 0.03,
    swirl: 0.24,
    name: 'moorMotes',
  });
  group.add(motes.points);
  owned.push(motes);
  timed.push(motes);

  return {
    group,
    sky,
    rig,
    terrain,
    field,
    aurora,
    skin,
    pools,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
    },
    setPixelRatio(dpr) {
      motes.setPixelRatio(dpr);
    },
    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// small shared prop material (actors.ts)
// ---------------------------------------------------------------------------

/** The one stone/iron material shape every prop in this scene is cut from. */
export function propMaterial(palette: Palette, tier: QualityTier, name: string): OakMaterial {
  return oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.92,
    rimColor: palette.hemiSky,
    rimStrength: 0.42,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.18,
    tier,
    name,
  });
}

/** Drive one prop material's clock (kept here so actors.ts imports one thing). */
export function setPropTime(material: THREE.Material | null, t: number): void {
  oakUniforms(material)?.setTime(t);
}

/** World x of a (possibly fractional) cell index — the wave front rides this. */
export function cellX(cell: number, grid: GridSize, out?: Vec3Tuple): number {
  return cellToWorld(cell, 0, grid, out)[0];
}
