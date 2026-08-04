// The Ivy Gate — the static world (KeyCorridor-cont-v0).
//
// THE ONE IDEA OF THIS SCENE: the env is a garden with ONE WALL AND ONE WAY THROUGH IT.
// Every visual decision serves that sentence. The outer wall is a LOW honey-stone kerb
// (0.5) so the plan view survives; the interior divider is a TALL, THIN ivy-clad wall
// (1.02 high, 0.7 thick) that dominates the frame without becoming a road; and the single
// free cell in it — the door — is the only thing in the garden the eye is asked to care
// about. West of the divider the planting is wild (moss, grass, the amber seed); east of
// it the ground is raked pale gravel around a lantern shrine on a warm flagstone. Wild →
// gate → formal, left to right, in one glance.
//
// NOTHING HERE RE-DERIVES THE LAYOUT (four-rooms canon lesson 12). Walls come from
// `grid.walls`; the door comes from `grid.doors` / `grid.hallways`; "is this cell the
// divider" is a STRUCTURAL read of the walls array (a wall with free cells on both sides
// along one axis), so the same code skins a regenerated layout. "West / east / threshold"
// is the sign of `x - door.x`, which is the env's own partition and not a typed-in range.
//
// PATTERNS COPIED FROM THE GRID CANON
//  * `buildCellField` is a SUBSTRATE: white base material + full-color `floorTint` /
//    `wallTint` instance colors, so the whole garden floor is ONE draw and the whole
//    kerb is one more.
//  * The divider cells COLLAPSE their cell-field instances (`wallInstanceAt` + zero
//    scale) and a taller bespoke wall segment owns them — the walls array is untouched,
//    so the ground-truth overlay still draws its boxes over the new prop.
//  * Orientation is READ (lesson 15): the gate faces the axis its flanking wall cells
//    run along, so `gateYaw` works on any layout that puts a door in a wall.
//
// LIGHT AND CAMERA ARE ONE DECISION. The door sits in a wall that runs north–south, so
// its faces look east and west: a camera due south sees the gate edge-on and the whole
// scene loses its subject. The frame is therefore a south-WEST 3/4 (azimuth −0.72, a 54.5°
// aerial — `index.ts` derives the elevation from the sightline census, not from taste),
// which reads the gate's west face at 45° while the divider hides only ~0.37 of a cell of
// the far room's floor. The sun then cannot sit where the camera is (a key light behind
// the lens is a flat light), so it stands ~89° off the view azimuth, west-by-north at 51°:
// the gate's visible face is lit, shadows rake across the frame toward the viewer, and the
// divider lays a soft bar of shade over the formal garden.

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
import { hash2f } from '@kit/rng';
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
  KEYCORRIDOR_DOOR,
  KEYCORRIDOR_H,
  KEYCORRIDOR_W,
  keyCorridorWalls,
} from '@sim/envs/keycorridor';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';

// ---------------------------------------------------------------------------
// world constants (1 cell = 1 world unit; the garden really is 9 × 7)
// ---------------------------------------------------------------------------

export const GARDEN_HALF_X = (KEYCORRIDOR_W * CELL) / 2;
export const GARDEN_HALF_Z = (KEYCORRIDOR_H * CELL) / 2;
export const TERRAIN_SIZE = 40;
export const TERRAIN_SEGMENTS = 128;

/**
 * The outer kerb. LOW on purpose: at the composed aerial a 0.9 wall hides 0.64 of the cell
 * behind it (h / tan 54.5°), and the near row of the garden stops reading as a grid. Half
 * that height hides a third of a cell, which the eye takes for depth rather than for loss.
 */
export const KERB_HEIGHT = 0.5;
/** the divider: the ONE thing in the garden you cannot walk through, so it stands tall */
export const DIVIDER_HEIGHT = 1.02;
export const DIVIDER_CAP = 0.11;
/**
 * How thick the divider is across the direction you would walk through it. A wall cell is
 * a whole cell to the ENV, but a full-cell stone prism reads from the air as a road, not
 * as a wall — the top face is then the widest, brightest surface in the frame and it wins
 * the composition. At 0.7 the wall keeps a wall's proportions, the cells beside it get
 * their floor back, and the overlay still draws the full cell box over it.
 */
export const DIVIDER_THICKNESS = 0.7;
/** the lawn sits a few centimetres under the flagstones so the joints read as soil */
export const GARDEN_SINK = -0.06;

/**
 * Fog reach. `applyFog` turns extent into density via `fogDensityHint` (morning = 0.35):
 * 11 × the terrain puts full density ~154 units out, which hazes the tree line at its
 * ~22-unit distance by about 6%. That is morning air. The first pass at 6 × put 18% of a
 * near-white fog over the tree line AND over the garden, and the whole frame went milky —
 * a grid scene cannot spend contrast on atmosphere, it has to spend it on cells.
 */
export const FOG_EXTENT = TERRAIN_SIZE * 11;

/**
 * Morning key light: west-by-north, elevation 51°. Two constraints pin it. It has to be
 * roughly side-on to the camera (~89° off the view azimuth) or the frame goes flat, and
 * it has to have a NEGATIVE x so the gate's visible west face is lit rather than in its
 * own wall's shadow. At 51° the divider throws 0.83 of a cell — a bar of shade across the
 * formal garden's near column. Dropping it to 30° threw 1.8 cells and swallowed a room.
 */
export const SUN_DIR: readonly [number, number, number] = [-0.46, 0.78, -0.42];
/** the shadow frustum must cover the FRAME (garden + near meadow), not just the cells */
export const SHADOW_RADIUS = 9.5;

/**
 * The composition azimuth, shared with `index.ts`'s OrbitRig so the planting, the sun and
 * the camera can never disagree. South-west: far enough round to read the gate's west
 * face at 45°, near enough to south that the divider only eats ~½ cell of the far room.
 */
export const CAMERA_AZIMUTH = -0.72;
/** how far out along that azimuth the "keep it clear" wedge is centred */
export const CAMERA_STANDOFF = 12;

// ---------------------------------------------------------------------------
// small shared helpers (actors.ts imports these too)
// ---------------------------------------------------------------------------

export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** smoothstep between two edges — the sequencing tool for the gate's two beats. */
export function smoothBand(edge0: number, edge1: number, x: number): number {
  if (edge1 <= edge0) return x >= edge1 ? 1 : 0;
  return smoothstep01((x - edge0) / (edge1 - edge0));
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
 * Take a token blend down into shade. `night.skyTop` is the darkest token there is and
 * STYLE.md names it as the legal way to reach near-black.
 *
 * The morning key is 2.6 — only a shade under noon's 3.0 — so the four-rooms albedo law
 * still binds here: ground albedo belongs around 140–175 sRGB, not 220. A pale gravel
 * courtyard is exactly the surface that wants to be typed in at 230 and exactly the one
 * that then crosses the bloom threshold and smears the east room into one white slab.
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
 * disposing the sources. A hand-built stone prop wants its SOURCE normals plus a per-part
 * palette baked in, so one mesh + one material draws the whole thing.
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
 * The grid the scene draws. The host always hands us the decoder's geometry; the fallback
 * is the DECODER'S OWN wall builder, never a scene re-derivation, so a placeholder
 * geometry degrades to the correct garden instead of a guess.
 */
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

/** The door cell, from the env geometry when present (doors, then hallways). */
export function resolveDoor(grid: GridGeometry): GridCell {
  const d = grid.doors?.[0] ?? grid.hallways?.[0];
  return d ? { x: d.x, y: d.y } : { ...KEYCORRIDOR_DOOR };
}

export const SIDE_WEST = -1;
export const SIDE_THRESHOLD = 0;
export const SIDE_EAST = 1;

/**
 * Which garden a cell belongs to, derived from the DOOR's own column: west of it is the
 * wild garden, east of it is the formal one, and the door's column is the threshold.
 * The env's rooms really are "left of the divider" and "right of the divider" — this is
 * that partition read off the geometry, not the decoder's hard-coded 1..3 / 5..7.
 */
export function gardenSide(door: GridCell, x: number): number {
  return x < door.x ? SIDE_WEST : x > door.x ? SIDE_EAST : SIDE_THRESHOLD;
}

/**
 * Is this wall cell part of an INTERIOR divider? A divider cell is a wall with free cells
 * on both sides along one axis — the structural definition of "a wall you have to get
 * through", true for KeyCorridor's x = 4 column and for any regenerated layout. Border
 * walls fail it because the outside of the grid counts as wall.
 */
export function isDividerWall(grid: GridGeometry, x: number, y: number): boolean {
  if (!isWall(grid, x, y)) return false;
  const openX = !isWall(grid, x - 1, y) && !isWall(grid, x + 1, y);
  const openZ = !isWall(grid, x, y - 1) && !isWall(grid, x, y + 1);
  return openX || openZ;
}

/**
 * The gate's yaw. Canon lesson 15: a gate faces the axis its FLANKING WALL CELLS run
 * along. A door whose north and south neighbours are wall sits in a wall running along z,
 * so the passage — and the door leaf — is turned 90° from the local build orientation
 * (which is built with the opening along ±z).
 */
export function gateYaw(grid: GridGeometry, door: GridCell): number {
  const wallRunsAlongZ = isWall(grid, door.x, door.y - 1) && isWall(grid, door.x, door.y + 1);
  return wallRunsAlongZ ? Math.PI / 2 : 0;
}

/** World-space bounding box of every FREE cell on one side of the divider. */
export interface SideBounds {
  cx: number;
  cz: number;
  halfX: number;
  halfZ: number;
  cells: number;
}

/**
 * Scan the walls array for one side's free cells and return their world extent (cell
 * centres ± ½ cell). Derived, never typed in: the per-room planting and mote fields are
 * clipped to the rooms the env actually has.
 */
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

// ---------------------------------------------------------------------------
// the palette (every color a blend of tokens — STYLE.md)
// ---------------------------------------------------------------------------

export interface GardenSkin {
  /** the wild west garden's floor: moss over dark earth */
  wildFloor: THREE.Color;
  /** the formal east garden's floor: raked pale gravel */
  formalFloor: THREE.Color;
  /** the door cell: a worn threshold stone */
  threshold: THREE.Color;
  /** soil under the walls */
  soil: THREE.Color;
  /** the low outer kerb: warm honey stone */
  kerb: THREE.Color;
  /** the tall divider: cool grey-green stone under its ivy */
  divider: THREE.Color;
  dividerCap: THREE.Color;
  /** the ivy itself — the scene's signature green */
  ivy: THREE.Color;
  ivyDark: THREE.Color;
  /** the door: warm weathered oak + its iron */
  wood: THREE.Color;
  woodDark: THREE.Color;
  iron: THREE.Color;
  /** the shrine's pale stone */
  shrineStone: THREE.Color;
  shrineDark: THREE.Color;
  /** the amber that the key, the shrine's ember and the celebration all share */
  amber: THREE.Color;
  /** the one floor tile the shrine stands on — repainted as the goal respawns */
  shrinePad: THREE.Color;
  /** wild planting */
  wildGrass: THREE.Color;
  wildBloom: THREE.Color;
  /** formal planting */
  gravel: THREE.Color;
  topiary: THREE.Color;
  /** the meadow outside the wall */
  meadow: THREE.Color;
  tree: THREE.Color;
  /** the two idle mote fields */
  motesWild: THREE.Color;
  motesFormal: THREE.Color;
}

/**
 * The palette. The amber is this scene's own accent (`accents['key-corridor']`), the ivy
 * borrows four-rooms' garden green and procedural-maze's moss, the stone borrows
 * multi-key-corridor's temple gold and po-maze's fog silver, and the oak borrows
 * high-dim-maze's terracotta warmed with the dusk earth. Every hue is on the shared
 * table, so the gallery still reads as one preserve.
 */
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
    // pale, but SHADED pale: the raked courtyard is the surface most likely to bloom out
    formalFloor: shade(blendTokens(palette.hemiSky, palette.hemiGround, 0.36), 0.3),
    threshold: shade(blendTokens(gold, palette.hemiGround, 0.3), 0.34),
    soil: shade(blendTokens(palette.hemiGround, earth, 0.45), 0.42),
    kerb: shade(blendTokens(palette.skyBottom, terracotta, 0.34), 0.54),
    // the darkest stone in the garden, so the one wall you cannot pass reads as a PLAN
    divider: shade(blendTokens(palette.groundA, silver, 0.3), 0.72),
    dividerCap: shade(blendTokens(silver, palette.hemiGround, 0.46), 0.52),
    ivy: shade(blendTokens(leaf, palette.groundA, 0.34), 0.26),
    ivyDark: shade(blendTokens(moss, palette.groundA, 0.55), 0.52),
    // THE OAK IS THE ONE WARM MASS IN A GREY WALL, and it has to win that argument on
    // VALUE, not only on hue. The leaf hangs on the divider's west face, which takes the
    // same 0.46 of the key light the leaf does, so the only separation the eye gets is
    // albedo: measured on the composed hero still, the shut leaf's median screen luma runs
    // ~34 levels over the wall's ~103, and the first pass (a wood shaded 0.12 toward night)
    // ran a third of that. Kept under the raked courtyard's 169 so the gate never blooms.
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

// ---------------------------------------------------------------------------
// scatter exclusions
// ---------------------------------------------------------------------------

/**
 * Keep a clear disc around every cell CENTER. The grid must stay legible under the
 * planting: props live in the joints between cells, never on the square the agent
 * actually occupies.
 */
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

/** Nothing plants itself on a wall cell (read from the walls array, never inferred). */
export function excludeWallCells(grid: GridGeometry): ExcludeFn {
  const ox = (grid.w - 1) / 2;
  const oz = (grid.h - 1) / 2;
  return (x, z) => isWall(grid, Math.round(x / CELL + ox), Math.round(z / CELL + oz));
}

// ---------------------------------------------------------------------------
// the meadow height field
// ---------------------------------------------------------------------------

/**
 * Rolling meadow OUTSIDE the garden, identically flat inside it. The weight is 0 for
 * everything within the garden footprint, so `makeTerrain`'s flatten mask has nothing to
 * fight and the ground under the flagstones is exactly `GARDEN_SINK`.
 */
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

// ---------------------------------------------------------------------------
// world handle
// ---------------------------------------------------------------------------

export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  /** the door cell, straight from the env geometry */
  door: GridCell;
  /** the yaw the gate (and its ivy) is built with */
  gateYaw: number;
  /** every divider cell the tall wall prop owns — the collapsed cell-field instances */
  dividerCells: readonly GridCell[];
  /** paint one floor tile (the shrine's pad follows the goal from cell to cell) */
  paintFloor(x: number, y: number, color: THREE.ColorRepresentation): void;
  /** put a painted tile back to exactly the tint + jitter `buildCellField` gave it */
  restoreFloorTint(x: number, y: number): void;
  /** ctx.time() — one clock for kerb, ivy wall, planting, motes and mist */
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
  const skin = gardenSkin(palette, ctx.tokens);
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'ivyGateGarden';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];

  const door = resolveDoor(grid);
  const yaw = gateYaw(grid, door);

  // ---- sky (exactly one) ----
  const sky = skyFromPalette(palette, 'morning', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.04,
    sunHdr: 5,
    sunGlow: 0.3,
    sunGlowPower: 24,
    topSpread: 0.68,
    bottomSpread: 0.56,
    radius: 340,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the one light budget: sun + hemi + 4 points (shrine / seed / wisp / bloom) ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 5.5,
    sunDistanceScale: 2.4,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, SHADOW_RADIUS);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the meadow the garden sits in ----
  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    tier,
    seed: ctx.seed,
    heightFn: makeMeadowHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.12,
    slopeColor: skin.soil,
    slopeStrength: 0.5,
    slopeSharpness: 2.2,
    flattenMask: { x: 0, z: 0, halfX: GARDEN_HALF_X + 0.25, halfZ: GARDEN_HALF_Z + 0.25, feather: 1.1 },
    flattenHeight: GARDEN_SINK,
    // real range top to bottom: a single-tone meadow made the garden look pasted onto a
    // billiard table, and no amount of scatter fixes a ground plane with no value in it
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
    receiveShadow: shadows,
    castShadow: false,
    name: 'ivyGateMeadow',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  // The per-cell floor tint, hoisted so `restoreFloorTint` can undo a state-driven repaint
  // BIT FOR BIT — including `buildCellField`'s own per-cell jitter, which the scene has to
  // reproduce because the field hands back no way to ask for it.
  const FLOOR_JITTER = 0.17;
  // the five tones a floor tile can take, built ONCE: `restoreFloorTint` runs from inside
  // update() on the frame the goal respawns, and update() may not allocate
  const TINT_WILD_A = skin.wildFloor;
  const TINT_WILD_B = blendTokens(skin.wildFloor, skin.soil, 0.24);
  const TINT_FORMAL_A = skin.formalFloor;
  const TINT_FORMAL_B = blendTokens(skin.formalFloor, skin.divider, 0.2);
  const floorTintFor = (x: number, y: number): THREE.Color => {
    if (isWall(grid, x, y)) return skin.soil;
    const side = gardenSide(door, x);
    if (side === SIDE_THRESHOLD) return skin.threshold;
    // the wild side is uneven ground, not a painted plane
    if (side === SIDE_WEST) return ((x * 3 + y * 5) & 1) === 0 ? TINT_WILD_A : TINT_WILD_B;
    // the formal court is RAKED: tone banded by row, so the ground carries parallel lines
    // across the whole courtyard. A chequer read as a chessboard and fought the grid; rows
    // read as a swept surface and agree with it.
    return (y & 1) === 0 ? TINT_FORMAL_A : TINT_FORMAL_B;
  };

  // ---- the garden: ONE floor draw, ONE kerb draw, straight off `grid.walls` ----
  // White base + full-color tint: the instance color MULTIPLIES the material color, so
  // this is how a cell field gets arbitrary per-cell hues without a second material.
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
      // 0.09, not the 0.22 a hero prop wants. Every wall in a plan-ish aerial is seen at a
      // grazing angle — precisely where the fresnel term peaks — so a normal rim strength
      // paints the whole cell field fog-white and the garden loses its stonework.
      rimStrength: 0.09,
      seed: ctx.seed,
      name: 'ivyGateCells',
    },
    tier,
  );
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  // ---- the tall divider: the cells the kerb gives up to a bespoke wall ----
  // Canon lesson 14: collapse the generic instance, keep the walls array untouched. The
  // overlay still draws a box on every one of these cells.
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
    tier,
    name: 'ivyGateDividerMaterial',
  });
  owned.push({ dispose: () => dividerMaterial.dispose() });
  if (dividerCells.length > 0) {
    const geometry = makeDividerGeometry(skin, tier);
    owned.push({ dispose: () => geometry.dispose() });
    const mesh = new THREE.InstancedMesh(geometry, dividerMaterial, dividerCells.length);
    mesh.name = 'ivyGateDivider';
    mesh.castShadow = shadows;
    mesh.receiveShadow = shadows;
    // a per-cell tone MULTIPLIER on top of the baked vertex colours: four identical
    // segments in a row look extruded, four subtly different ones look built
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
      // the segment is built with its ivy draping one face; a half-turn on half the
      // instances drapes the other, so the wall is green on both sides without a
      // second geometry
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
    group.add(mesh);
    owned.push({ dispose: () => mesh.dispose() });
  }

  // ---- planting: wild west, formal east, clipped to the rooms the env actually has ----
  const noPlanting = excludeAny(excludeWallCells(grid), excludeCellCenters(grid, 0.3));
  const wild = sideBounds(grid, door, SIDE_WEST);
  const formal = sideBounds(grid, door, SIDE_EAST);
  const planting: VegetationHandle[] = [];

  const wildGrass = scatterGrass({
    count: 420,
    area: { x: wild.cx, z: wild.cz, halfX: wild.halfX, halfZ: wild.halfZ },
    seed: ctx.seed ^ 0x4c17,
    tier,
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
  group.add(wildGrass.group);
  owned.push(wildGrass);
  timed.push(wildGrass);
  planting.push(wildGrass);

  const wildBlooms = scatterFlowers({
    count: 74,
    area: { x: wild.cx, z: wild.cz, halfX: wild.halfX, halfZ: wild.halfZ },
    seed: ctx.seed ^ 0x2b93,
    tier,
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
  group.add(wildBlooms.group);
  owned.push(wildBlooms);
  timed.push(wildBlooms);
  planting.push(wildBlooms);

  // the formal side: swept gravel (flattened rocks) and a handful of clipped box balls
  const gravel = scatterRocks({
    count: 130,
    area: { x: formal.cx, z: formal.cz, halfX: formal.halfX, halfZ: formal.halfZ },
    seed: ctx.seed ^ 0x7f41,
    tier,
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
  group.add(gravel.group);
  owned.push(gravel);
  timed.push(gravel);
  planting.push(gravel);

  const topiary = scatterRocks({
    count: 26,
    area: { x: formal.cx, z: formal.cz, halfX: formal.halfX, halfZ: formal.halfZ },
    seed: ctx.seed ^ 0x39d5,
    tier,
    exclude: noPlanting,
    color: skin.topiary,
    // round and squat: a rock archetype at stretch ~0.9 is a clipped box ball
    scale: [0.16, 0.34],
    stretch: [0.75, 1.15],
    hueJitter: 0.16,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    castShadow: shadows,
    name: 'formalTopiary',
  });
  group.add(topiary.group);
  owned.push(topiary);
  timed.push(topiary);
  planting.push(topiary);

  // ---- the meadow outside, and the tree line beyond it ----
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
    tier,
    exclude: outside,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.85,
    // lifted well above the ground tone: a blade seen from a steep aerial is edge-on, and a
    // grass colour picked to match the meadow renders as a scatter of dark shards
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
  group.add(meadowGrass.group);
  owned.push(meadowGrass);
  timed.push(meadowGrass);
  planting.push(meadowGrass);

  // THE VERGE. A garden wall standing in bare lawn looks pasted on; a band of taller
  // meadow flowers breaking against its foot is what makes it look BUILT there.
  const verge = scatterFlowers({
    count: 105,
    area: { halfX: GARDEN_HALF_X + 1.5, halfZ: GARDEN_HALF_Z + 1.5 },
    seed: ctx.seed ^ 0x58ba,
    tier,
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
  group.add(verge.group);
  owned.push(verge);
  timed.push(verge);
  planting.push(verge);

  const treeLine = scatterTrees({
    archetypes: ['broadleaf', 'conifer'],
    count: 58,
    area: { halfX: 17.5, halfZ: 17.5, innerRadius: 7.6 },
    seed: ctx.seed ^ 0x1a70,
    tier,
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
  group.add(treeLine.group);
  owned.push(treeLine);
  timed.push(treeLine);
  planting.push(treeLine);

  // ---- idle life ----
  // NOTE for every grid scene: the kit particle presets size their motes for a ~20-unit
  // field. This garden is 9 units across with the camera ~13 out, so `gl_PointSize`'s
  // 60/distance term wants sizes around 1, not the preset's 4–9.
  const motes: AmbientParticlesHandle[] = [];
  const wildMotes = makeAmbientParticles({
    preset: 'pollen',
    count: 190,
    bounds: { x: wild.cx, y: 0.95, z: wild.cz, halfX: wild.halfX, halfY: 0.85, halfZ: wild.halfZ },
    seed: ctx.seed ^ 0x7d21,
    tier,
    color: skin.motesWild,
    size: 1.05,
    intensity: 1.7,
    opacity: 0.6,
    rise: 0.07,
    swirl: 0.26,
    name: 'wildPollen',
  });
  group.add(wildMotes.points);
  owned.push(wildMotes);
  timed.push(wildMotes);
  motes.push(wildMotes);

  const formalMotes = makeAmbientParticles({
    preset: 'dust',
    count: 150,
    bounds: { x: formal.cx, y: 0.8, z: formal.cz, halfX: formal.halfX, halfY: 0.72, halfZ: formal.halfZ },
    seed: ctx.seed ^ 0x51c3,
    tier,
    color: skin.motesFormal,
    size: 0.72,
    intensity: 1.3,
    opacity: 0.44,
    rise: 0.03,
    swirl: 0.16,
    name: 'formalDust',
  });
  group.add(formalMotes.points);
  owned.push(formalMotes);
  timed.push(formalMotes);
  motes.push(formalMotes);

  // Morning air, banked ABOVE the kerb line. Any lower and it veils the cells, which is
  // the one thing a grid scene may not trade away; at 0.95 it hangs over the walls and
  // reads as the hour instead of as a filter.
  const mist = makeMistPlanes({
    size: [38, 36],
    color: blendTokens(palette.fog, palette.sun, 0.3),
    seed: ctx.seed ^ 0x3ac1,
    layers: 2,
    tier,
    // The camera looks DOWN through every mist plane at every cell, so a stack of them is
    // a stack of veils over the one thing the scene has to keep readable. Two layers at a
    // twentieth opacity, hung a metre above the wall line, read as morning air; the first
    // pass (four layers at 0.16) read as a dirty lens and cost the garden all its contrast.
    baseY: 1.75,
    spread: 1.1,
    opacity: 0.05,
    speed: 0.008,
    scale: 2.0,
    nearFade: 9,
    name: 'morningAir',
  });
  group.add(mist.group);
  owned.push(mist);
  timed.push(mist);

  const restoreColor = new THREE.Color();

  return {
    group,
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
      planting.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// props
// ---------------------------------------------------------------------------

/**
 * Collapse one cell-field wall instance to nothing so a bespoke prop can own that cell.
 * The walls array is NOT touched: the cell is still a wall to the env, to the overlay and
 * to every accuracy check — only the generic kerb block that stood there is gone.
 */
export function collapseWallInstance(field: CellFieldHandle, x: number, y: number): void {
  const mesh = field.wallMeshes[0];
  const id = field.wallInstanceAt(x, y);
  if (!mesh || id < 0) return;
  const m = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  mesh.getMatrixAt(id, m);
  m.decompose(p, q, s);
  m.compose(p, q, s.set(0, 0, 0));
  mesh.setMatrixAt(id, m);
  mesh.instanceMatrix.needsUpdate = true;
}

/**
 * One segment of the tall ivy wall: a stone block filling the cell, a capstone, and ivy
 * draping ONE face — a mat of leaves over the top course plus two runners down it. The
 * drape is BAKED, because a divider segment is not state-bearing: it never changes, so
 * paying a per-frame matrix for it would be pure cost. The gate's ivy — the one that
 * actually unwinds — is the animated system in actors.ts.
 *
 * Built in local space around the cell centre with the drape on −z, so `gateYaw` turns
 * the whole wall together with the door it contains.
 */
export function makeDividerGeometry(skin: GardenSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = tier === 'thumb' ? 0.35 : tier === 'medium' ? 0.7 : 1;
  const parts: ColoredPart[] = [];
  const size = CELL - 0.03;

  // three courses, not one block: a wall this size is the frame's darkest mass and a
  // single smooth prism reads as poured concrete. The tone alternates by course, which is
  // all the "stonework" a flat-shaded silhouette needs at this distance.
  const courses = tier === 'thumb' ? 2 : 3;
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

  // a spread footing, so the thinner wall still OCCUPIES the cell the env says it does
  const footing = new THREE.BoxGeometry(size, 0.13, DIVIDER_THICKNESS + 0.2);
  footing.translate(0, 0.065, 0);
  parts.push({ geometry: footing, color: skin.dividerCap });

  // A CUSHION of leaves over the top course. The first pass used one flat green box and
  // the wall read as a highway barrier with a stripe painted on it; ivy is a texture of
  // overlapping blades and nothing but real blades will say so.
  const rows = tier === 'thumb' ? 1 : 2;
  const perRow = tier === 'thumb' ? 5 : Math.max(5, Math.round(9 * detail));
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

  // runners down the face, each a short ladder of leaves that thins as it falls
  const runners = tier === 'thumb' ? 1 : 2;
  const leavesPer = Math.max(3, Math.round(8 * detail));
  for (let r = 0; r < runners; r++) {
    const x0 = runners === 1 ? 0 : (r === 0 ? -0.24 : 0.26);
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

/**
 * ONE ivy leaf: a five-vertex spade in the xy plane, stem at the origin, tip at +y, with
 * a vertex grade that darkens the stem end. Three triangles is all a leaf needs at this
 * scale, and it is the atom of both the baked wall drape and the animated gate vine.
 */
export function ivyLeafGeometry(): THREE.BufferGeometry {
  const w = 0.105;
  const l = 0.2;
  //      tip
  //   sh0   sh1     (shoulders)
  //      base
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
