// The Fogbound Forest — the static world (POMaze-cont-v0).
//
// THE ONE IDEA OF THIS SCENE: THE FOG IS THE OBSERVATION. POMaze's agent sees eight wall
// bits — its immediate neighbours — plus four compass bits for the goal, and nothing
// else. So this forest is drawn the way the agent knows it: a continuous, uniform floor
// of needle litter under a fog so dense that the world ends five metres out, and dark
// spruce ranks that MATERIALIZE out of that fog only while they are inside the agent's
// own 3 × 3 window, then sink back as it walks on.
//
// THE HONESTY CONTRACT, made structural rather than remembered:
//  * `bubbleVisibility` is a Chebyshev falloff between 1.0 and 2.0 world units of the
//    wisp's CURRENT position. Sampled at cell centres with the wisp settled, it is
//    IDENTICALLY 1 on the eight neighbours and IDENTICALLY 0 everywhere else — the same
//    eight bits `poMazeView` hands the agent. During a hop it interpolates, which is the
//    dissolve; it is a pure function of the current position, so a scrub lands on the
//    exact bubble and never on a remembered one.
//  * The FLOOR is drawn under every one of the 121 cells (`floorUnderWalls`), tinted from
//    a hash that knows nothing about walls. If the floor were drawn only under free cells
//    the map would be readable from the air, which is the one thing this scene may not do.
//  * The interior of the maze holds no other structure at all. The only silhouettes past
//    the bubble are the OUTER forest — a ring that begins 14 units out, beyond the
//    playfield entirely, so it carries zero information about the layout.
//
// PATTERNS BORROWED FROM THE CANON
//  * `buildCellField` is the substrate (four-rooms): white base + `floorTint` instance
//    colors, one draw for the whole forest floor. Its wall blocks are COLLAPSED to zero
//    scale exactly like four-rooms' sundial cell — the walls array is untouched, the
//    overlay still draws every box, and the spruce ranks own those cells instead.
//  * Orientation is READ, never assumed (four-rooms' moon gates): a gap's stones flank
//    the axis its neighbouring wall cells run along, so the same code stands them up in
//    any layout.
//  * Terrain noise is multiplied by a weight that is identically 0 over the playfield
//    (mountain-car's exact-band law), so the forest floor is exactly y = 0 where the
//    accuracy lives and rolls only outside.

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

// ---------------------------------------------------------------------------
// world constants (1 cell = 1 world unit; the maze really is 11 × 11)
// ---------------------------------------------------------------------------

/** half the playfield footprint, derived from the env's own grid size */
export const MAZE_HALF = (POMAZE_W * CELL) / 2;
/**
 * The ground reaches 42 units in every direction, and the number is forced rather than
 * chosen: the sky dome is drawn UNFOGGED, so wherever the terrain's far edge falls inside
 * the frame the eye catches chips of bare sky through the gaps in the outer forest —
 * bright warm splinters hanging in the middle of a cold fog wall. At 52 units that edge
 * sat ~5° under the horizon and the artifact was visible on every frame; at 84 the edge
 * is 42 units out, where transmittance is e^(−0.1345·42) ≈ 0.35%, so there is nothing left
 * of it to see.
 */
export const TERRAIN_SIZE = 84;
export const TERRAIN_SEGMENTS = 120;
/** the litter sits a few centimetres under the cell tiles, so the joints read as soil */
export const FLOOR_SINK = -0.05;

/**
 * THE FOG. `applyFog` turns extent into density through `fogDensityHint` (dawn = 0.45):
 * density = 1.7308 / (0.45 · extent), i.e. 5% transmittance at 0.45 · extent. 2.6 grid
 * widths gives density 0.1345 — ten times four-rooms' 0.013 and the densest air in the
 * preserve by an order of magnitude, which is the brief.
 *
 * It is NOT the densest number that fits the brief, and the reason is measured. A pass at
 * 0.16 was shot and rejected: past ~0.15 the haze reaches the ADJACENT ranks, the spruce
 * stop being silhouettes and turn the same pale grey as the air, and the picture loses
 * the values it needs to be atmospheric ABOUT. At 0.1345 transmittance runs 0.87 at 1
 * unit and 0.76 at 2 (the bubble keeps its own values), 0.67 at 3, 0.45 at 6 (the ranks
 * read as dark shapes), 0.26 at 10, 0.15 at 14 (the far tree line is a rumour) and 0.003
 * at 42 (the terrain's own edge is gone) — the depth ladder the scene is made of.
 *
 * NOTHING is hidden by fog density here anyway: the bubble is what hides the maze, and it
 * hides it exactly, at every distance. The fog is the MOOD; the bubble is the honesty.
 */
export const FOG_EXTENT = POMAZE_W * 2.6;

/**
 * East and north, which with the westward camera (`CAMERA_AZIMUTH`) puts the dawn IN
 * FRONT of the eye: the spruce are back-lit, so they read as rim-lit silhouettes in a
 * glowing volume of air rather than as lit models, and their shadows run toward the
 * camera and give the litter its only relief. Elevation 0.55 (33°) stays above the ~25°
 * floor where a low sun smears a shadow across its own subject (canon lesson 8).
 */
export const SUN_DIR: readonly [number, number, number] = [0.62, 0.55, -0.556];
/**
 * The shadow frustum FOLLOWS the wisp (see `WorldHandle.setShadowFocus`): everything
 * past this radius is fog, so spending the 2048 map on the bubble is the whole budget
 * used where it is visible — about 0.005 world units per texel.
 */
export const SHADOW_RADIUS = 5;

/**
 * THE COMPOSITION AZIMUTH, shared with index.ts's OrbitRig so the two cannot disagree.
 *
 * It is read off the ENV, not chosen by eye. POMaze's walls are two full rows (y = 3 and
 * y = 7) plus the border, so the free space is three corridors that run EAST–WEST, and
 * the agent is next to a band almost every step of its life. A camera placed across that
 * axis — the obvious three-quarter aerial every other grid scene uses — therefore spends
 * most of the run looking at the back of a rank with the wisp hidden behind it; the first
 * cut of this scene did exactly that and buried its own subject.
 *
 * −1.35 rad puts the eye nearly due WEST (x = d·sin θ ≈ −0.98 d) looking east along the
 * corridor, so the ranks recede to left and right and frame the wisp instead of blocking
 * it, and the only cells that can ever stand between eye and agent are its east
 * neighbours — which in this maze are wall only at the border. It also turns the frame
 * INTO the dawn: `SUN_DIR` is east-and-north, so the spruce are back-lit, their fog is
 * lit from behind, and the warm end of the sky sits above the far tree line.
 */
export const CAMERA_AZIMUTH = -1.35;

/**
 * THE BUBBLE. Chebyshev distance in WORLD units from the wisp: 1 cell is full
 * visibility, 2 cells is nothing. Sampled at cell centres with the wisp settled that is
 * exactly the eight neighbours `poMazeView` reports and exactly nothing else.
 */
export const BUBBLE_INNER = CELL;
export const BUBBLE_OUTER = CELL * 2;

/** the goal's +1 buys one breath of sight, this far around the cell it happened on */
export const REVEAL_RADIUS = 3.2;
/** and lifts this fraction of the fog density while it lasts */
export const FOG_LIFT = 0.55;

/** the outer forest starts beyond every reachable camera position (11.5) — see index.ts */
export const OUTER_FOREST_INNER = 14;
export const OUTER_FOREST_OUTER = 21;

// ---------------------------------------------------------------------------
// small shared helpers (actors.ts imports these too)
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

/** Take a token blend down into shade; `night.skyTop` is STYLE.md's legal near-black. */
export function shade(color: THREE.ColorRepresentation, k: number): THREE.Color {
  return blendTokens(color, TOKENS.palettes.night.skyTop, k);
}

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

/**
 * Merge pre-transformed parts into ONE geometry carrying a flat per-part vertex color,
 * disposing the sources — one mesh and one material draw a whole hand-built prop.
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
 * The grid the scene draws. The host always hands us the decoder's geometry; the
 * fallback is the DECODER'S OWN wall builder, never a scene re-derivation, so a
 * placeholder geometry degrades to the correct maze instead of a guess.
 */
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

/** The two gaps that stitch the three bands together, from the env geometry when present. */
export function resolveGaps(grid: GridGeometry): readonly GridCell[] {
  const h = grid.hallways;
  return h && h.length > 0 ? h : POMAZE_GAPS;
}

// ---------------------------------------------------------------------------
// THE BUBBLE — the whole honesty contract in two pure functions
// ---------------------------------------------------------------------------

/**
 * How visible a cell centred at (cx, cz) is from a wisp standing at (wx, wz), both in
 * WORLD units. Chebyshev, because the agent's window is a square: full inside one cell,
 * gone at two, smooth in between so a hop dissolves instead of popping.
 *
 * With the wisp settled on a cell centre this returns EXACTLY 1 for the eight
 * neighbours (Chebyshev distance 1.0) and EXACTLY 0 for everything else (≥ 2.0).
 */
export function bubbleVisibility(wx: number, wz: number, cx: number, cz: number): number {
  const d = Math.max(Math.abs(cx - wx), Math.abs(cz - wz));
  if (d <= BUBBLE_INNER) return 1;
  if (d >= BUBBLE_OUTER) return 0;
  return 1 - smoothstep01((d - BUBBLE_INNER) / (BUBBLE_OUTER - BUBBLE_INNER));
}

// ---------------------------------------------------------------------------
// THE COMPASS — the second half of the honesty contract
// ---------------------------------------------------------------------------

/** the moth rides just inside the bubble's outer edge — the fog's edge, literally */
export const MOTH_RIM = BUBBLE_OUTER * 0.88;
/** how far the rim breathes in and out (world units) */
export const MOTH_BREATH = 0.16;
/**
 * How far the moth may wander off the true bearing, in radians. 0.34 ≈ 19.5°, comfortably
 * inside the 45° that would let an octant be misread — the creature is alive but the
 * direction it gives is never ambiguous.
 */
export const MOTH_SWING = 0.34;
/** its cruising height above the litter, and the bob it holds while it hovers */
export const MOTH_HEIGHT = 0.6;
export const MOTH_BOB = 0.12;

/**
 * THE COMPASS, as a pure function of the FOUR BITS POMaze actually publishes.
 *
 * The env's observation carries sign(gx − x) and sign(gy − y) and NOT the distance, so
 * this takes the two signs and the clock and nothing else: "the moth cannot leak the
 * distance" is then a property of the code — there is no distance in scope to leak —
 * rather than a promise made in a comment. Bearing is the sign vector's bearing plus a
 * bounded wander (|Δ| ≤ MOTH_SWING); the radius is a function of time alone, identical
 * for a goal one cell away and a goal across the maze.
 *
 * Returns false when the goal is on the agent's own cell: there is no direction, so the
 * compass has nothing honest to say and the caller hides it.
 */
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

/**
 * The goal breath: a radial (not Chebyshev — this is weather, not observation) falloff
 * around the cell the +1 happened on. Returns 0 outside `radius`, so a settled scene is
 * bit-identical to one that never scored.
 */
export function revealVisibility(rx: number, rz: number, cx: number, cz: number, radius: number): number {
  const d = Math.hypot(cx - rx, cz - rz);
  if (d >= radius) return 0;
  return 1 - smoothstep01((d - radius * 0.45) / (radius * 0.55));
}

// ---------------------------------------------------------------------------
// the palette (every color a blend of tokens — STYLE.md)
// ---------------------------------------------------------------------------

export interface ForestSkin {
  /** the needle-litter floor, two tones the per-cell hash picks between */
  litterA: THREE.Color;
  litterB: THREE.Color;
  /** the spruce ranks: the darkest thing in the forest */
  spruce: THREE.Color;
  /** the outer ring, one step further into the haze */
  spruceFar: THREE.Color;
  /** undergrowth ferns */
  fern: THREE.Color;
  /** mossy granite */
  stone: THREE.Color;
  stoneDark: THREE.Color;
  /** the air itself — fog color, mist planes and the grade tint all read from here */
  haze: THREE.Color;
  /** dust motes hanging in the dawn light */
  motes: THREE.Color;
  /** the goal's moonflower */
  bloom: THREE.Color;
  /** the compass moth */
  moth: THREE.Color;
}

/**
 * Dawn, cooled. The token dawn fog (0xdbc2a6) is a warm sand; blended a quarter of the
 * way into the dawn sky top it becomes the blue-grey haze a spruce forest actually has
 * at first light, and every surface color is pulled toward it so the whole frame reads as
 * one volume of air. The spruce are `dawn.groundA` taken deep into `night.skyTop` —
 * STYLE.md's legal near-black — because the ranks must read as SILHOUETTES: their job is
 * to be shapes in fog, not models.
 */
export function forestSkin(palette: Palette, tokens: typeof TOKENS): ForestSkin {
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const gold = tokens.celebrationGold;
  // THE MASTER VALUE, and the one number every other colour in this file is measured
  // against. Every distant surface converges to the haze, and — because the fog reaches
  // the near field too — it also sets the FLOOR of the frame: at 3 units a third of the
  // pixel is already air, so nothing can render much darker than that fraction of it.
  //
  // Two things are being bought here. 62% toward the dawn SKY TOP takes the token fog
  // (0xdbc2a6, a warm sand) off its warm axis, so the distance goes blue-grey while the
  // sun keeps the near litter warm — a picture with two temperatures instead of a sepia
  // wash. The 62% shade then buys VALUE: these blends happen in linear space, where a
  // half-and-half mix reads far brighter than the sRGB arithmetic suggests, and a lighter
  // haze pulls the spruce up to its own grey and flattens the whole frame.
  const haze = shade(blendTokens(palette.fog, palette.skyTop, 0.62), 0.62);
  return {
    // THE VALUE STRUCTURE of a misty forest, and the thing two passes of this scene got
    // wrong: the floor must be DARKER than the air. A litter bright enough to look like
    // ground in isolation lights up under the 2.2 dawn key until it out-values the haze,
    // and then the fog stops reading as fog and starts reading as a blown exposure.
    // ~0.20 albedo, pulled toward the haze's hue so the near floor and the far air are
    // the same weather.
    litterA: shade(blendTokens(palette.groundA, haze, 0.55), 0.6),
    litterB: shade(blendTokens(palette.groundB, haze, 0.5), 0.6),
    // the darkest thing in the preserve, and deliberately so: the ranks are SHAPES
    spruce: shade(blendTokens(palette.groundA, silver, 0.1), 0.74),
    spruceFar: shade(blendTokens(palette.groundA, haze, 0.5), 0.5),
    fern: shade(blendTokens(palette.groundA, palette.groundB, 0.3), 0.62),
    stone: shade(blendTokens(silver, palette.hemiGround, 0.4), 0.44),
    stoneDark: shade(blendTokens(palette.hemiGround, silver, 0.2), 0.58),
    haze,
    motes: blendTokens(palette.sun, silver, 0.3),
    // THE MOONFLOWER IS COOL, THE SPIRIT IS WARM, and the pair is the reason. Both are
    // small HDR emitters seen through the same fog a few metres apart, so if they share a
    // hue the frame reads as two identical lamps and the eye cannot tell the goal from the
    // agent. Moon-white against the wisp's amber separates them at a glance and gives the
    // one moment they are on screen together its picture.
    bloom: blendTokens(silver, palette.skyTop, 0.55),
    moth: blendTokens(gold, silver, 0.34),
  };
}

// ---------------------------------------------------------------------------
// scatter exclusions
// ---------------------------------------------------------------------------

/**
 * Keep a clear disc around every cell CENTER. The grid must stay legible under the
 * planting: undergrowth lives in the joints between cells, never on the square the agent
 * actually occupies. Note what this does NOT test — wall-ness. Undergrowth that thinned
 * out over free cells would draw the map on the floor.
 */
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

/** Nothing plants itself inside the disc the camera orbits in. */
export function excludeDisc(radius: number): ExcludeFn {
  const r2 = radius * radius;
  return (x, z) => x * x + z * z < r2;
}

// ---------------------------------------------------------------------------
// the forest floor height field
// ---------------------------------------------------------------------------

/**
 * Rolling ground OUTSIDE the maze, identically flat inside it. The weight is 0 for
 * everything within the playfield footprint, so the cell tiles sit on exactly
 * `FLOOR_SINK` and `sampleHeight` is exact where the accuracy lives.
 */
export function makeForestHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.max(Math.abs(x), Math.abs(z)) - MAZE_HALF;
    const w = smoothstep01(out / 6);
    if (w <= 0) return 0;
    // gently: the outer forest stands on this, and a metre of terrain roll out there is
    // a metre of tree crown pushed up through the skyline and out of the top of the frame
    const n1 = valueNoise2(x * 0.07 + 3.1, z * 0.07 + 9.4, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.19 + 6.7, z * 0.19 + 2.2, 64, seed ^ 0x51ed270b) - 0.5;
    return w * (n1 * 1.5 + n2 * 0.5);
  };
}

// ---------------------------------------------------------------------------
// the spruce ranks — the walls, and the only thing the bubble reveals
// ---------------------------------------------------------------------------

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
  /** wall cells the ranks stand on (= the count of 1s in `grid.walls`) */
  cellCount: number;
  treesPerCell: number;
  /** the visibility currently applied to a cell, 0 when it is not a wall */
  visibilityAt(x: number, y: number): number;
  /**
   * Rewrite the ranks for a wisp at world (wx, wz) plus an optional goal-breath reveal
   * centred at (rx, rz). Allocation-free, and it skips the GPU upload entirely when no
   * cell changed — a paused frame costs nothing.
   */
  setBubble(wx: number, wz: number, rx: number, rz: number, reveal: number): void;
  setTime(t: number): void;
  dispose(): void;
}

/**
 * One InstancedMesh for every wall cell in the maze, `treesPerCell` conifers apiece,
 * placed deterministically inside their own cell and SCALED by the bubble.
 *
 * Scale is the dissolve because it is the one channel an InstancedMesh gives per
 * instance for free (per-instance opacity would need a bespoke material, and STYLE.md
 * allows exactly one of those per scene — this one spends nothing). A tree at
 * visibility 0 is a degenerate triangle: no fragment, no fill cost, nothing to see.
 * Trees rise out of the litter as they widen, so entering adjacency reads as spruce
 * CONDENSING out of the fog rather than a mesh being switched on.
 */
export function buildSpruceRanks(opts: SpruceRanksOptions): SpruceRanksHandle {
  const { grid, tier, seed } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'spruceRanks';
  const detail = tier === 'thumb' ? 0 : tier === 'medium' ? 1 : 2;
  const treesPerCell = tier === 'thumb' ? 1 : tier === 'medium' ? 2 : 3;

  // wall cells, straight off the walls array — no border assumption, no band formula
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
      // A RING, not a scatter, and this is the difference between a forest and a hedge.
      // The trees of a cell sit on a circle around its centre at a fixed angular spacing
      // (jittered, so no two cells rhyme), which leaves the middle of every wall cell
      // OPEN. It matters because of the geometry this scene cannot escape: the agent's
      // window is one cell, so an adjacent rank stands 1 unit from a 0.35-high wisp while
      // the eye is 5 units back — no camera elevation short of a top-down shot clears a
      // 2-unit spruce at that range. Scattering filled the cell and the subject vanished
      // behind it; a ring leaves gaps at known angles, and the wisp's light comes through
      // them the way light comes through real trunks.
      const ring = ((k + 0.5) / treesPerCell) * Math.PI * 2 + h1 * 1.1;
      const radius = treesPerCell > 1 ? 0.27 + h2 * 0.08 : 0.1;
      baseX[id] = world[0] + Math.cos(ring) * radius;
      baseZ[id] = world[2] + Math.sin(ring) * radius;
      // SPIRES, not Christmas trees. `coniferTreeGeometry` is 1.8 units tall and 0.84
      // wide at unit scale; the first cut ran it at 0.52–0.98 wide, which put a 0.6-unit
      // canopy one cell from the wisp and closed the corridor into a wall. The width here
      // tops out near 0.45 of a cell while the height runs to ~2.2 — both what a
      // fog-drowned spruce actually looks like, and thin enough to be a stroke in the
      // picture rather than a mass.
      baseScale[id] = 0.34 + h3 * 0.18 + (k === 0 ? 0.06 : 0);
      baseStretch[id] = 1.45 + h1 * 1.0;
      baseYaw[id] = h2 * Math.PI * 2;
      baseTilt[id] = (h3 - 0.5) * 0.14;
      // a rank does not materialize in lockstep: the outer trees lag the nearest one
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

  // preallocated scratch — setBubble runs inside update()
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
      // Rise AND widen: a spruce condensing out of fog, never a mesh switched on. The
      // width carries a FACTOR OF g, and that is load-bearing rather than stylistic: a
      // conifer scaled to y = 0 with any width left is not gone, it is a flat pie-plate
      // of dark triangles lying in the litter at exactly the cell it was hiding. The
      // first cut of this scene held width at 0.3 and drew the ENTIRE MAZE as discs on
      // the floor — the one thing the scene may never do. At g = 0 every axis is
      // identically zero: no area, no fragment, no map.
      const wide = s * g * (0.55 + 0.45 * g);
      scale.set(wide, s * (baseStretch[id] ?? 1) * g, wide);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(id, matrix);
    }
  };

  // start settled and invisible; index.ts drives the first real bubble in update()
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

// ---------------------------------------------------------------------------
// the gap gates — the two cells that stitch the bands together
// ---------------------------------------------------------------------------

export interface GapGateHandle {
  /** probe object: exactly `cellToWorld(gap)`, POSITION ONLY */
  root: THREE.Object3D;
  /** the scaled child — the bubble drives this, never the root */
  body: THREE.Object3D;
  cell: GridCell;
}

/**
 * Two leaning granite stones flanking the pass, on the exact cell `grid.hallways` names.
 * Orientation is READ, never assumed (four-rooms' law): a gap whose EAST and WEST
 * neighbours are wall sits in a band that runs along x, so the passage — and the stones —
 * line up across it.
 *
 * These are bubble-gated like everything else, and honestly so: standing next to a gap,
 * the agent's own eight bits say "that cell is free while its neighbours are not". The
 * stones show exactly that and not one bit more.
 */
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
    // the band runs along x when the cells either side of the gap are wall
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

/**
 * A pair of standing stones with a mossy cap, opening along ±z (so the yaw above is the
 * only thing that ever changes between layouts).
 */
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

// ---------------------------------------------------------------------------
// world handle
// ---------------------------------------------------------------------------

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
  /** ctx.time() — pause freezes spruce, ferns, mist and motes together */
  setTime(t: number): void;
  setPixelRatio(dpr: number): void;
  /** the shadow map follows the only place shadows are visible: the bubble */
  setShadowFocus(x: number, z: number): void;
  /** 0 = the standing fog, 1 = the goal breath at full lift */
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

  // ---- sky (exactly one) ----
  // The horizon and the ground of the dome are pulled onto the haze color, so the far
  // fog wall and the sky are the same value: the world does not END at the fog, it
  // DISSOLVES into it. The sun disc stays small and only mildly HDR — a big bloomed disc
  // through this much fog reads as a blown highlight, not a dawn.
  const sky = skyFromPalette(palette, 'dawn', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.045,
    sunHdr: 3.4,
    sunGlow: 0.5,
    sunGlowPower: 9,
    // The only sky this camera ever sees is the ~5° band above the tree line, so the
    // warm horizon color has to live exactly there. A wide bottomSpread squeezed it out
    // of the frame entirely and the first passes had no dawn in them at all.
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

  // ---- the one light budget: sun + hemi + 4 points (wisp / breath / goal / moth) ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 5.5,
    sunDistanceScale: 2.6,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, SHADOW_RADIUS);

  // ---- THE FOG (see FOG_EXTENT) ----
  const fog = applyFog(scene, palette, FOG_EXTENT);
  fog.color.copy(skin.haze);
  const baseDensity = fog.density;

  // ---- the ground the maze sits in ----
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
    // The ramp is pulled tight around the two litter tones ON PURPOSE. The cell tiles are
    // separate boxes sitting a centimetre proud of this terrain, so any gap between the
    // two palettes draws a hard bright wedge along the playfield boundary wherever the
    // camera catches it — the maze stops being a clearing in a forest and becomes a rug
    // on a floor. Keeping the ends within a few percent of `litterA`/`litterB` lets the
    // edge read as one continuous forest floor.
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

  // ---- the playfield: ONE floor draw over ALL 121 cells ----
  // `floorUnderWalls` is the honesty rule made structural. Every cell gets a tile and the
  // tint is a hash of (x, y) that never asks whether the cell is a wall, so the litter
  // gives the eye a scale to read the fog against and gives the map away to nobody.
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

  // The kit placed a block on every wall cell (strictly from `grid.walls` — that is its
  // contract). Collapse ALL of them: the spruce ranks own those cells, and a block that
  // stood there would be visible through the fog from anywhere and hand over the map.
  // four-rooms collapses one cell for its sundial; this scene collapses the whole set.
  collapseAllWalls(field, grid);

  // ---- the spruce ranks: the walls, materialized only inside the bubble ----
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

  // ---- stone: the two gap gates share one material ----
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

  // ---- undergrowth: uniform over the whole floor, thinning outward ----
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

  // ---- the outer forest: silhouettes, and NOTHING else ----
  // It begins 14 units out — beyond the playfield, beyond every position the orbit rig
  // can reach (11.5, see index.ts's arithmetic) — so it never stands between the camera
  // and the physics and never says a word about the layout. In this fog it is a wall of
  // ghosts at 6–15% transmittance, which is exactly the job. Crowns are capped near 2.9
  // units: taller and the tree line stops being a line and fills the sky band that is
  // the only dawn in the frame.
  const outer = scatterTrees({
    archetypes: ['conifer'],
    // dense enough to close into a WALL. At 620 over the first pass's wide annulus it
    // read as scattered debris chopped by the top of the frame; a solid dark tree line
    // under a band of dawn sky is what frames the fog.
    count: 1400,
    area: { halfX: OUTER_FOREST_OUTER, halfZ: OUTER_FOREST_OUTER, innerRadius: OUTER_FOREST_INNER },
    seed: ctx.seed ^ 0x77c1,
    tier,
    exclude: excludeDisc(OUTER_FOREST_INNER),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.8,
    color: skin.spruceFar,
    // Tall enough to make a horizon and SHORT enough to stay under it. Crowns top out at
    // ~2.8 units against a 2.2-unit eye height, so at thirteen units the tree line sits
    // ~2.5° up and the frame's top edge is at 4.5° — a band of dawn sky survives above
    // the forest. The first pass's 5-unit crowns filled that band completely.
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

  // ---- the air ----
  // the banks are a touch BRIGHTER than the fog they hang in, so they read as drifting
  // wisps of light rather than as more of the same grey
  const mist = makeMistPlanes({
    size: [18, 18],
    color: blendTokens(skin.haze, palette.sun, 0.28),
    seed: ctx.seed ^ 0x2ab7,
    tier,
    // Three layers at 0.10, and ALL OF THEM BELOW THE EYE. Mist planes are horizontal
    // quads with `fog: false`, so a layer that reaches eye height is seen edge-on: its
    // noise compresses into hard bright horizontal streaks hanging in the middle of the
    // frame, which is exactly what the 0.1–2.0 stack did under this scene's low camera
    // (the eye sits at ~2.3). Capped at 0.56 they stay ground mist, read from above, and
    // do what they are for — drift over the litter.
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

  // Kit particle presets size their motes for a ~20-unit field; this camera stands ~5
  // units off its subject, so the sizes here are a fraction of the preset (canon lesson
  // 5). Dust hangs low in the litter, spores drift through the crowns.
  const motes: AmbientParticlesHandle[] = [];
  const dust = makeAmbientParticles({
    preset: 'dust',
    count: 340,
    bounds: { x: 0, y: 0.85, z: 0, halfX: MAZE_HALF + 2, halfY: 0.8, halfZ: MAZE_HALF + 2 },
    seed: ctx.seed ^ 0x4e11,
    tier,
    color: skin.motes,
    // Particles render UNFOGGED (kit contract), so a mote at fifteen units is as bright
    // as one at two and the eye reads the whole field as snow in front of the lens. In
    // the densest fog in the preserve they have to be small and faint or they own the
    // frame — the first pass at 0.9/0.50 did exactly that.
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

/**
 * Collapse every cell-field wall instance to nothing. The walls array is NOT touched:
 * each cell is still a wall to the env, to the overlay and to every accuracy check —
 * only the generic block that stood there is gone, and a spruce rank stands in its place.
 */
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
