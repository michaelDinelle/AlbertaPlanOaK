// Temple of Three Gates — the static world (MultiKeyCorridor-cont-v0).
//
// THE ONE IDEA OF THIS SCENE: the env is a PROCESSION. Four chambers, three locked
// gates, one relic at the far end — the agent must walk east collecting a key per
// chamber, and every gate it opens is a step deeper into the temple. So the art
// GRADUATES: chamber 0 is morning-cool stone in the shade, chamber 3 is sunlit gold,
// and the three gates between them are three different problems (a bramble you must
// cut, a ford you must stone, a mist you must part). Look at any frame and you know
// how far in the wisp has got.
//
// NOTHING HERE RE-DERIVES THE LAYOUT (the grid-canon law):
//  * walls come from `grid.walls` (the decoder's own `mkcWalls()` as the fallback),
//  * the three gate cells come from `grid.doors`,
//  * chamber membership comes from the decoder's own `mkcRoomCells(r)` — the same
//    function the env uses to draw a respawn cell — so a cell is chamber-3 gold if
//    and only if the ENV says the goal can respawn there.
//
// PATTERNS COPIED FROM THE GRID CANON (four-rooms), worth restating because this
// scene stresses them harder:
//  * `buildCellField` is a SUBSTRATE: white floor/wall material + full-color per-cell
//    `floorTint`/`wallTint` instance colors = one floor draw and one wall draw for a
//    17×7 temple with four different stones in it.
//  * A cell that wants a bespoke prop collapses its instance and a probe-rooted prop
//    owns it — here the river gate's cell, whose floor tile is skipped for a basin.
//  * Orientation is READ, never assumed: a gate faces the axis its flanking wall
//    cells run along, so the portals stand correctly without a hard-coded rotation.
//
// LIGHT: morning, from the east-south-east at ~53°, i.e. OVER the sanctum and down the
// processional toward the camera side. That is half the reason the graduation reads:
// the east end is lit and the west end sits in its own wall's shade. Shadows at 53°
// throw ~0.55 of a cell from a 0.7 wall — enough to model the gates and the lanterns,
// short enough that the grid never stops being a grid.

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

// ---------------------------------------------------------------------------
// world constants (1 cell = 1 world unit; the temple really is 17 × 7)
// ---------------------------------------------------------------------------

export const TEMPLE_HALF_X = (MKC_W * CELL) / 2;
export const TEMPLE_HALF_Z = (MKC_H * CELL) / 2;
export const TERRAIN_SIZE = 46;
export const TERRAIN_SEGMENTS = 128;
/**
 * Temple wall height. Over the wisp's orb, under the portal lintels — and no higher.
 * A 17 × 7 hall seen from a 3/4 aerial is a box canyon: at 0.92 the south wall ate the
 * front row of every chamber, and even at 0.74 it hid the floor of every GATE cell
 * (the one cell in each wall whose contents are the whole point). The camera's polar
 * and this number are one decision — see the frame note in index.ts.
 */
export const WALL_HEIGHT = 0.7;
/** the garden sits a few centimetres under the flagstones, so joints read as soil */
export const COURT_SINK = -0.07;
/**
 * Fog reach. `applyFog` turns extent into density via `fogDensityHint` (morning =
 * 0.35): this puts full density ~175 units out, which leaves the far tree line ~8%
 * hazed at its ~28-unit distance. Morning fog is nearly white — more than that reads
 * as a blown exposure, not as air, and the temple loses the contrast the graduation
 * is built on.
 */
export const FOG_EXTENT = TERRAIN_SIZE * 11;
/**
 * East-south-east, 53° up. The sanctum is at +x, so this is light coming in OVER the
 * relic and down the processional, with the shadows falling AWAY from the camera.
 * Lower than 40° doubles the shadow throw and the middle chambers go muddy; higher than
 * 60° flattens the portals into plan shapes.
 */
export const SUN_DIR: readonly [number, number, number] = [0.56, 0.82, 0.25];
/** the shadow frustum must cover the FRAME (temple + near planting), not the playfield */
export const SHADOW_RADIUS = 13.5;
/**
 * The composition azimuth, shared with `index.ts`'s OrbitRig so the planting and the
 * camera can never disagree. MountainCar's law — nothing lives between the camera and
 * the physics — needs a ground position to exclude, and this is where it comes from.
 */
export const CAMERA_AZIMUTH = 0.14;
/** how far out along that azimuth the "keep it clear" wedge is centred */
export const CAMERA_STANDOFF = 13.5;

/** chambers, in processional order: 0 is the entrance, 3 is the sanctum. */
export const CHAMBERS = 4;
export const CHAMBER_NAMES: readonly string[] = ['dawnwell', 'grove', 'ochre', 'sanctum'];

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

/**
 * Take a token blend down into shade. `night.skyTop` is the darkest token there is and
 * STYLE.md names it as the legal way to reach near-black, so "shade" is a blend like
 * every other color here.
 *
 * The morning key is 2.6 — only a shade under noon's 3.0 — so the noon albedo law
 * applies unchanged: a 220-sRGB flagstone renders past the bloom threshold and the
 * whole temple floor smears into one white slab. Ground albedo belongs at 140–170.
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
 * disposing the sources. Hand-built stone props (a portal, a lantern, a shrine) want
 * their SOURCE normals plus a per-part stone palette baked in, so one mesh — or one
 * InstancedMesh for all three portals — draws the whole thing.
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
 * placeholder geometry degrades to the correct temple instead of a guess.
 */
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

/** The three gate cells, from the env geometry when present (never guessed). */
export function resolveDoors(grid: GridGeometry): readonly GridCell[] {
  const d = grid.doors && grid.doors.length > 0 ? grid.doors : grid.hallways;
  return d && d.length > 0 ? d : MKC_DOORS;
}

/**
 * The three OPENING key cells — the layout the env starts every stream on. The live
 * cells are replayed per frame (`state.keys[j]`); this is only what the temple is built
 * showing before the first frame arrives, and it comes from the geometry the host
 * resolved, never from a cell written into the scene.
 */
export function resolveKeys(grid: GridGeometry): readonly GridCell[] {
  const k = grid.keys;
  return k && k.length > 0 ? k : MKC_KEYS;
}

/** The opening goal cell, same rule: read from the env, never a scene literal. */
export function resolveGoal(grid: GridGeometry): GridCell {
  return grid.goal ?? MKC_GOAL;
}

/**
 * Chamber membership, straight off the decoder's own room enumeration. `mkcRoomCells(r)`
 * is the exact function `gw_mkc_step` uses to place a respawned key, agent or goal, so
 * a cell is painted sanctum-gold if and only if the ENV can put the relic there.
 * −1 = threshold (the three gate cells) or wall.
 */
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

/**
 * A cell's position ALONG the procession in [0, 1]: 0 at the west wall, 1 at the east.
 * Used for the warm/cool graduation of everything that is not a chamber floor (walls,
 * gravel, motes), so the temple keeps warming even across the thresholds.
 */
export function processionT(x: number): number {
  return clamp01((x - 1) / (MKC_W - 3));
}

/** World-space bounding box of every FREE cell of a chamber, in world units. */
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

/**
 * The rectangle that covers every chamber — the whole planted garden. Used by the thumb
 * tier, which plants one bank per family across all of it instead of one per chamber.
 */
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

/**
 * True when the wall through this cell runs along z, i.e. the passage through it is
 * along x. Read from the walls array (grid canon lesson 15) so a portal stands
 * correctly whatever layout it is handed.
 */
export function gateFacesX(grid: GridGeometry, cell: GridCell): boolean {
  return isWall(grid, cell.x, cell.y - 1) && isWall(grid, cell.x, cell.y + 1);
}

// ---------------------------------------------------------------------------
// the temple palette (every color a blend of tokens — STYLE.md)
// ---------------------------------------------------------------------------

export interface ChamberSkin {
  /** flagstone tint */
  floor: THREE.Color;
  /** the chamber's own wall stone */
  wall: THREE.Color;
  /** ground scatter (moss / grass / gravel / gold dust) */
  litter: THREE.Color;
  /** the chamber's flowers */
  bloom: THREE.Color;
  /** the idle particle field */
  motes: THREE.Color;
}

export interface TempleSkin {
  chambers: readonly ChamberSkin[];
  /** the three gate cells: dark polished threshold stone */
  threshold: THREE.Color;
  /** the outer wall and the gate piers */
  stoneLight: THREE.Color;
  stoneDark: THREE.Color;
  /** the bed under every wall */
  soil: THREE.Color;
  /** the dark bed under gate 1's ford — the water is read against THIS, not the floor */
  riverBed: THREE.Color;
  /** the woody cane of the bramble that laces gate 0 */
  bramble: THREE.Color;
  /** its leaves — lighter, and the part an aerial camera can actually see */
  brambleLeaf: THREE.Color;
  /** the three key gems, in key order: seed / pearl / ember */
  keys: readonly THREE.Color[];
  /** the sanctum's relic flame */
  relic: THREE.Color;
  /** the lantern flames along the procession */
  flame: THREE.Color;
}

/**
 * The palette, and the single most important decision in this scene: THE WALLS ARE THE
 * DARKEST THING IN THE TEMPLE. The first pass gave the masonry the same warm tan as the
 * chamber floors and the whole 17 × 7 hall turned to porridge — no plan, no procession,
 * and a gold wisp invisible on a gold floor. Four-rooms learned the same law with its
 * yew hedges: a grid scene reads because its impassable cells are a dark lattice and
 * its floors are the light between them.
 *
 * So: walls sit around 60–90 sRGB, floors around 90–140 (the noon albedo law, and the
 * morning key is 2.6 — only a shade under noon's 3.0), and the saturated chamber colour
 * lives in the planting and the motes rather than in the stone. That leaves the whole
 * top of the range — the wisp, the three gems, the relic — to the things that MOVE.
 *
 * The four chambers still walk from a cool wet slate at the mouth through garden green
 * and terracotta to the sanctum's gold; every stop is a blend of shared tokens
 * (`po-maze` fog silver, `four-rooms` garden green, `high-dim-maze` terracotta, this
 * scene's own temple gold), so the gallery reads as one preserve and this scene reads
 * as one journey.
 */
export function templeSkin(palette: Palette, tokens: typeof TOKENS): TempleSkin {
  const gold = tokens.accents['multi-key-corridor'] ?? tokens.celebrationGold;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const garden = tokens.accents['four-rooms'] ?? palette.groundA;
  const terracotta = tokens.accents['high-dim-maze'] ?? tokens.celebrationGold;
  const moss = tokens.accents['procedural-maze'] ?? palette.groundA;
  const blossom = tokens.accents['chain'] ?? tokens.celebrationGold;
  const white = tokens.palettes.noon.sun; // the legal off-palette white (canon lesson 9)

  const chambers: ChamberSkin[] = [
    {
      // 0 — the dawnwell: cool wet slate, moss in the joints, the last of the night
      floor: shade(blendTokens(palette.hemiSky, garden, 0.42), 0.64),
      litter: shade(blendTokens(moss, palette.hemiSky, 0.28), 0.42),
      bloom: shade(blendTokens(silver, white, 0.35), 0.3),
      motes: blendTokens(palette.hemiSky, white, 0.35),
      wall: shade(blendTokens(silver, palette.hemiGround, 0.55), 0.78),
    },
    {
      // 1 — the grove: the garden proper, full morning green
      floor: shade(blendTokens(palette.groundA, garden, 0.34), 0.58),
      litter: shade(blendTokens(garden, palette.groundB, 0.34), 0.3),
      bloom: shade(blendTokens(blossom, white, 0.25), 0.2),
      motes: blendTokens(palette.groundB, white, 0.4),
      wall: shade(blendTokens(silver, palette.hemiGround, 0.5), 0.77),
    },
    {
      // 2 — the ochre court: dry stone, the garden giving way to temple
      floor: shade(blendTokens(palette.groundB, terracotta, 0.42), 0.64),
      litter: shade(blendTokens(terracotta, gold, 0.42), 0.42),
      bloom: shade(blendTokens(gold, terracotta, 0.3), 0.24),
      motes: blendTokens(gold, white, 0.3),
      wall: shade(blendTokens(silver, terracotta, 0.2), 0.76),
    },
    {
      // 3 — the sanctum: gold leaf under morning light, where the relic respawns
      floor: shade(blendTokens(gold, tokens.celebrationGold, 0.4), 0.56),
      litter: shade(blendTokens(gold, white, 0.28), 0.38),
      bloom: shade(blendTokens(tokens.celebrationGold, white, 0.2), 0.18),
      motes: blendTokens(tokens.celebrationGold, white, 0.25),
      wall: shade(blendTokens(silver, terracotta, 0.26), 0.74),
    },
  ];

  return {
    chambers,
    // the thresholds are the darkest floor of all: three dark bands across the hall
    // where the gates stand, so the temple's chapters are legible from the plan alone
    threshold: shade(blendTokens(palette.hemiGround, silver, 0.24), 0.68),
    // the ARCHITECTURE is lighter than the walls it stands in — portals, lanterns,
    // pedestals and the shrine are what the eye should find first after the wisp
    stoneLight: shade(blendTokens(silver, palette.hemiGround, 0.32), 0.3),
    stoneDark: shade(blendTokens(palette.hemiGround, silver, 0.3), 0.5),
    soil: shade(blendTokens(palette.hemiGround, TOKENS.palettes.dusk.groundA, 0.4), 0.5),
    riverBed: shade(blendTokens(TOKENS.palettes.underwater.skyBottom, palette.hemiGround, 0.3), 0.4),
    bramble: shade(blendTokens(palette.groundA, TOKENS.palettes.dusk.groundA, 0.5), 0.55),
    brambleLeaf: shade(blendTokens(palette.groundB, garden, 0.45), 0.34),
    keys: [
      // seed (amber), pearl (white — the noon.sun blend), ember (warm)
      new THREE.Color(tokens.accents['key-corridor'] ?? tokens.celebrationGold),
      blendTokens(white, palette.hemiSky, 0.16),
      new THREE.Color(tokens.accents['mountain-car'] ?? tokens.celebrationGold),
    ],
    relic: blendTokens(tokens.celebrationGold, palette.sun, 0.24),
    flame: blendTokens(tokens.celebrationGold, terracotta, 0.24),
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

/** The processional row (y = the gate row) stays walkable and clear of planting. */
export function excludeProcession(grid: GridGeometry, doors: readonly GridCell[], halfWidth: number): ExcludeFn {
  const row = doors[0]?.y ?? (grid.h - 1) / 2;
  const oz = (grid.h - 1) / 2;
  const zWorld = (row - oz) * CELL;
  return (_x, z) => Math.abs(z - zWorld) < halfWidth;
}

// ---------------------------------------------------------------------------
// the garden height field
// ---------------------------------------------------------------------------

/**
 * Rolling meadow OUTSIDE the temple, identically flat inside it. The weight is 0 for
 * everything within the temple footprint, so `makeTerrain`'s flatten mask has nothing
 * to fight and the ground under the flagstones is exactly `COURT_SINK`.
 */
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

// ---------------------------------------------------------------------------
// world handle
// ---------------------------------------------------------------------------

export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  field: CellFieldHandle;
  /** probe objects: the three portals, in `grid.doors` order. POSITION ONLY. */
  portalRoots: THREE.Object3D[];
  doors: readonly GridCell[];
  /** the shared temple stone material (the gates in actors.ts borrow it) */
  stoneMaterial: OakMaterial;
  /** ctx.time() — pause freezes the whole garden together */
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

  // ---- sky (exactly one) ----
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

  // ---- the one light budget: sun + hemi + 4 points (relic / wisp / event / sanctum) ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 7.5,
    sunDistanceScale: 2.1,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0.5, y: 0.35, z: 0 }, SHADOW_RADIUS);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the garden the temple stands in ----
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
    // shaded down: an unshaded morning groundA/groundB ramp under a 2.6 key renders as
    // a pale mint putting green, and the temple loses the ground it stands on
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

  // ---- the temple: ONE floor draw, ONE wall draw, straight off `grid.walls` ----
  const white = palette.sun;
  const field = buildCellField(
    grid,
    {
      floorColor: white,
      wallColor: white,
      floorUnderWalls: true,
      // EVERY free cell keeps its flagstone, the river gate's included. Skipping that one
      // tile (the obvious way to sink a basin into it) opened a hole straight through to
      // the garden terrain, which renders pale from above and read as a blank white slab
      // in the middle of the temple. The ford lies ON its flagstone instead.
      floorTint: (x, y) => {
        if (isWall(grid, x, y)) return skin.soil;
        const chamber = chamberOf(x, y);
        return chamber < 0 ? skin.threshold : (skin.chambers[chamber]?.floor ?? skin.threshold);
      },
      wallTint: (x, y) => {
        const chamber = chamberOf(x, y);
        if (chamber >= 0) return skin.chambers[chamber]?.wall ?? skin.stoneDark;
        // outer wall / gate piers: warm along the procession, so even the masonry
        // graduates from the cool entrance to the gold sanctum
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

  // ---- stone: portals + lanterns + copings share one material ----
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

  // ---- the three portals: one geometry, one InstancedMesh, three exact cells ----
  // Their ROOTS are probe objects (position only); the instanced mesh carries the art.
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

      // the portal is modelled with its passage along ±x; a gate in a wall that runs
      // along x is the same geometry yawed 90°
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

  // ---- stone lanterns down the procession, in the JOINTS between cells ----
  // Never on a cell center: the grid has to survive the decoration.
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
  // their flames: the one HDR emitter of the static world, warm and small. At thumb the
  // whole temple has 40 draw calls and a flame is under a pixel, so the bowls stay unlit.
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

  // ---- per-chamber planting, clipped to the chamber the env actually has ----
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
    // Counts are per CHAMBER, and a chamber is 15 cells with its centres and its
    // processional row cut out — about 7.6 usable world units². The first pass ran
    // 25 instances per unit² and the temple floor read as confetti; these are a third
    // of that, which is planting rather than litter.
    { chamber: 0, kind: 'grass', count: 105, scale: [0.26, 0.52], stretch: [0.7, 1.2], wind: 0.03, name: 'dawnwellMoss' },
    { chamber: 0, kind: 'flowers', count: 26, scale: [0.32, 0.62], stretch: [0.8, 1.2], wind: 0.028, name: 'dawnwellDewflowers' },
    { chamber: 1, kind: 'grass', count: 130, scale: [0.32, 0.64], stretch: [0.8, 1.4], wind: 0.042, name: 'groveGrass' },
    { chamber: 1, kind: 'flowers', count: 38, scale: [0.38, 0.72], stretch: [0.85, 1.3], wind: 0.03, name: 'groveBlossoms' },
    { chamber: 2, kind: 'rocks', count: 52, scale: [0.11, 0.24], stretch: [0.24, 0.42], wind: 0, name: 'ochreGravel' },
    { chamber: 2, kind: 'grass', count: 58, scale: [0.24, 0.46], stretch: [0.7, 1.1], wind: 0.034, name: 'ochreTufts' },
    { chamber: 3, kind: 'flowers', count: 34, scale: [0.34, 0.64], stretch: [0.9, 1.25], wind: 0.026, name: 'sanctumGoldflowers' },
    { chamber: 3, kind: 'rocks', count: 26, scale: [0.09, 0.18], stretch: [0.22, 0.36], wind: 0, name: 'sanctumGoldDust' },
  ];
  // THUMB: one bank per PLANT FAMILY instead of one per chamber. Eight per-chamber banks
  // are eight draw calls of the forty STYLE.md gives the whole wall-view frame, to plant
  // about sixty instances that are two pixels each — and at that size the per-chamber
  // hue they are paying for is not resolvable. So thumb plants grass, flowers and gravel
  // ONCE across the whole garden (the union of the four chamber bounds), carrying the
  // grove's and the ochre court's colours with the same hue jitter. Hero and medium keep
  // the four seasons of planting exactly as they were.
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

  // ---- the tree line outside the temple wall ----
  // Two exclusions, and the second is the one that matters: nothing lives between the
  // camera and the physics, and at this azimuth the camera stands off the south-east
  // corner of the garden.
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
    // thumb keeps only the belt behind the temple: two of the three tree lines are
    // outside a pulled-back square crop anyway, and each costs a draw call
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

  // ---- idle life ----
  // The kit presets size their motes for a ~20-unit field; this temple is 17 across
  // with the camera ~17 out, so sizes live around 1 (grid canon lesson 5).
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

  // ---- morning haze, OUTSIDE the temple only (thumb drops it) ----
  // A 3/4 aerial looks through every layer of a horizontal mist stack, so an opacity
  // that is gentle at eye level is a fog bank from above: the first pass at 4 × 0.13
  // put half a stop of milk over the whole hall. Three layers at 0.05, lifted to sit
  // ABOVE the wall tops, so the haze reads as morning air in the garden and never as a
  // veil over the playfield.
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

// ---------------------------------------------------------------------------
// props
// ---------------------------------------------------------------------------

/** the lantern bowl's flame height (the point light rides just over it) */
export const LANTERN_FLAME_Y = 0.575;
/** the portal lintel height — the gates in actors.ts hang their mechanisms under it */
export const PORTAL_LINTEL_Y = 1.42;

/**
 * A temple portal: two piers and a slender lintel, modelled in LOCAL space around the
 * gate cell's centre with the PASSAGE along ±x and the frame spanning ±z.
 *
 * THE SILHOUETTE HAS TO WORK FROM ABOVE. The first pass crowned it with a 0.56 × 1.3
 * stepped slab, which is handsome at eye level and, from this scene's 3/4 aerial, a
 * grey LID: it covered the doorway and hid the one thing a gate exists to show — the
 * bramble, the ford, the mist inside it. So the crown is gone and the lintel is narrow
 * in x. Seen from overhead the portal reads as two stone blocks flanking an OPEN gap
 * with a bar across it, and the mechanism inside is never occluded.
 */
export function makePortalGeometry(skin: TempleSkin, tier: QualityTier): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const PIER_H = PORTAL_LINTEL_Y;
  // THE PIERS STAND ON THE FLANKING WALL CELLS, NOT INSIDE THE GATE.
  // This is the single hardest-won number in the scene. A 1.42-tall pier standing at
  // z = ±0.46 (inside the gate cell, where a doorjamb belongs) occludes the ground from
  // its own z back to z − 0.9 at this camera's 57° elevation — which is the WHOLE gate
  // cell. Three gates rendered their bramble, their ford and their mist perfectly, into
  // a floor the camera could not see, and a magenta-water diagnostic was what finally
  // proved the mesh was there. At ±0.95 the piers are engaged columns rising off the
  // wall blocks: the portal reads bigger, the wall reads thicker, and the gate cell's
  // own floor — the mechanism — stays in frame.
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

  // narrow in x on purpose: from overhead this is a bar, not a roof
  const lintel = new THREE.BoxGeometry(0.3, 0.16, 2.16);
  lintel.translate(0, PIER_H + 0.08, 0);
  parts.push({ geometry: lintel, color: skin.stoneLight });
  const band = new THREE.BoxGeometry(0.34, 0.05, 1.7);
  band.translate(0, PIER_H + 0.005, 0);
  parts.push({ geometry: band, color: skin.stoneDark });

  // NO threshold slab. The first pass laid a 0.66 × 0.94 worn slab across the gate cell
  // and it covered the river gate's basin completely — the one gate whose mechanism is
  // horizontal, and therefore the one the aerial camera could actually have read. The
  // cell's own flagstone (tinted `skin.threshold`) is the threshold; the gates own the
  // rest of the cell.

  return mergeColoredParts(parts, 'templePortal');
}

/**
 * A stone lantern: base, shaft, bowl and a little pitched cap with the light slot
 * between them. Sixteen of these line the procession, all one InstancedMesh.
 */
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
