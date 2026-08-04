// The Slot Canyons — the static world (HighDimMaze-cont-v0).
//
// THE ONE IDEA: the maze's two interior wall columns ARE two sandstone fins, and the two
// holes punched in them ARE two natural rock arches. HighDimMaze is a serpentine — three
// north-south slots (x 1..2, x 4..6, x 8..9) stitched by the gaps at (3,9) and (7,1) —
// and that shape is the whole composition. Walk it and you walk a canyon system: south
// down the first slot, east under the low arch, north up the wide middle, east under the
// high arch, south to the oasis.
//
// NOTHING HERE RE-DERIVES THE LAYOUT. Wall cells come from `grid.walls`, the two arches
// from `grid.hallways` (the decoder's own gap list), the arch ORIENTATION from the wall
// cells flanking each gap, and the floor's sun/shade split from a scan of the walls
// array. Change the maze and the canyon follows.
//
// PATTERNS WORTH COPYING
//  * A cell field is a SUBSTRATE (grid canon): white base + full-color `floorTint` /
//    `wallTint` gives per-cell hue inside ONE floor draw and ONE wall draw.
//  * Per-cell wall HEIGHT is not in the kit's style, so the field's wall instances are
//    re-composed once at init from `wallHeightAt()` — a stationary function of the cell.
//    The canyon rim steps down toward the south, which is both how a drainage really
//    behaves and what keeps the near wall from eating the front row of the frame.
//  * The strata are ONE extra instanced draw: four proud ledge bands (two at thumb) built
//    in NORMALIZED height space (y ∈ [0,1]) and scaled per instance by that cell's own
//    height, so a taller cliff gets thicker beds for free.
//  * The sand-fall curtains are placed on cliff LIPS found by scanning the walls array
//    for wall cells with a free neighbour — never on typed-in coordinates.

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
import type { QualityTier, SceneContext } from '@kit/types';
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

// ---------------------------------------------------------------------------
// world constants (1 cell = 1 world unit; the canyon really is 11 × 11)
// ---------------------------------------------------------------------------

/** half the canyon footprint, derived from the env's own grid size */
export const CANYON_HALF = (HIGHDIMMAZE_W * CELL) / 2;
export const TERRAIN_SIZE = 76;
export const TERRAIN_SEGMENTS = 120;
/** the desert floor sits a hand's breadth under the sand slabs, so the joints read as rock */
export const CANYON_SINK = -0.1;

/**
 * Cliff heights. The two interior FINS are the tall walls that make the slots; the border
 * RIM is the outer lip of the mesa. Both taper from north to south, because a canyon
 * system drains one way and its rim really does step down along it — and because the near
 * (southern) rim stands between the camera and the playfield, so it is the ONE wall that
 * has to stay low (0.66) or it eats the front row of the grid.
 *
 * FIN HEIGHT AND CAMERA AZIMUTH ARE ONE DECISION, and it is measured, not chosen. A fin
 * viewed across its line throws occlusion into the slot beside it: the sight-line to a cell
 * one column west of the fin has to clear that fin's crown after crossing its whole 1-unit
 * thickness, which costs `0.5·(camY − 0.35)/(camX − cellX)` of height — a full cell of
 * sideways drift per 0.36 rad of yaw at this standoff. Ray-traced against the real cliff
 * columns, all 65 free cells, hero elevation (cells where the orb is fully hidden / cells
 * where under 35% of it shows):
 *
 *     fins         az 0.07   az 0.13   az 0.16   az 0.20   az 0.25   az 0.30
 *     1.92/1.44     0 / 0     3 / 0     3 / 0     3 / 1     4 / 1     5 / 1
 *     1.60/1.20     0 / 0     0 / 0     0 / 2     0 / 4     4 / 0     4 / 0
 *     1.45/1.10     0 / 0     0 / 0     0 / 0     0 / 0     0 / 4     3 / 1
 *     1.28/1.00     0 / 0     0 / 0     0 / 1     0 / 1     0 / 1     0 / 1
 *
 * The first pass took 1.92 with azimuth 0.36 on the theory that a camera there looks ALONG
 * the fins. It does not: the whole west corridor — seven cells, on the mandatory route out
 * of the start — hid the agent completely, and so did the high arch. Cliff height is what
 * buys the frame back. At 1.28 on a 1-unit cell the slots are still a canyon (a 2-cell slot
 * under 1.3 units of wall is a slot; the eye reads the RATIO, and the strata and the cast
 * shade sell the depth), and it is the line that pays for the 0.25 rad of yaw that makes
 * this a 3/4 view of a canyon system instead of a plan of one — with enough margin left
 * that every corner of the rig's limits still clears (12–35% of the orb at the worst one).
 */
export const FIN_HEIGHT_NORTH = 1.28;
export const FIN_HEIGHT_SOUTH = 1.0;
export const RIM_HEIGHT_NORTH = 1.24;
export const RIM_HEIGHT_SOUTH = 0.54;
/** the cell-field wall geometry is built at this height; per-cell scale.y is a fraction of it */
export const WALL_BASE_HEIGHT = FIN_HEIGHT_NORTH;

/**
 * THE NOTCH. An arch is not a hole punched through a full-height fin — it is what is LEFT
 * when the fin erodes down to a neck and the resistant bed on top survives as a bridge. So
 * the fin ramps down toward each gap: the cell touching it keeps 0.36 of its line, the next
 * 0.74, and the span stands over that saddle.
 *
 * This is also what makes the gap cells VISIBLE. A gap cell sits ON the fin's own line, so
 * the sight-line into it arrives ALONG the fin and grazes the crown of every cell between:
 * at full height the two cells south of (7,1) stand a metre of rock in front of a 0.35-unit
 * orb, and the agent in the high arch could not be seen at all. The ramp puts both crowns
 * 0.2–0.5 under the sight-line at every camera the rig allows (the test re-measures it),
 * and it is why the arch now reads as the thing the fin erodes DOWN to rather than a lump
 * sitting on top of it.
 *
 * It stays a stationary function of the walls array and the decoder's own hallway list —
 * the cells are still walls, the array is untouched, and the overlay still boxes them.
 */
export const ARCH_NOTCH: readonly number[] = Object.freeze([0.36, 0.74]);

/**
 * An arch stands a little proud of the fin it bridges, and no more. The first pass gave it
 * 0.34 over the taller jamb; with a 1.25 opening under it that left almost a unit of rock
 * on top and the arch rendered as a tower with a mouse hole in it. A natural arch is
 * SLENDER — the span is what you see, and the daylight under it is what says "arch".
 */
export const ARCH_RISE = 0.14;
/** headroom under an arch. The wisp orb rides at tokens.wispHeight = 0.35 with a 0.4 halo. */
export const ARCH_OPENING = 1.25;
/** the minimum band of rock left above the opening — below this it is a lintel, not a span */
export const ARCH_BAND = 0.34;
/**
 * Half the span, in cells: the arch bridges its own gap cell and no more. Wider spans were
 * measured and rejected — an abutment landing out on a notched shoulder puts a descending
 * haunch exactly where the sight-line into the gap arrives, which is the occlusion this
 * scene was fixing in the first place. What makes the gap READ is not a longer span but
 * where the soffit springs from (see `makeArchGeometry`).
 */
export const ARCH_SPAN_HALF = 0.5;
/**
 * How thick the span is across the passage, as a fraction of the fin's own thickness: a
 * RIBBON at the crown, and legs thinner still. This is the number that decides whether an
 * agent standing in an arch can be seen at all — see the note on `depthAt` in
 * `makeArchGeometry`. A full-thickness arch (the first pass) turns the gap cell into a
 * roofed tunnel whose mouths face east and west, i.e. away from every camera this scene
 * allows.
 */
export const ARCH_RIBBON = 0.36;
export const ARCH_LEG = 0.24;

/**
 * Fog reach. `applyFog` turns extent into density via `fogDensityHint` (afternoon = 0.3),
 * so full density lands ~210 units out: the playfield keeps ~96% of its colour at the
 * 17-unit standoff and the plateau still runs out into haze by 50. Aerial perspective is
 * the only depth cue an all-ground frame has — but the pass that set this to 330 laid a
 * pale green veil over the canyon itself and the terracotta went to soup. Fog belongs to
 * the distance, never to the subject.
 */
export const FOG_EXTENT = 700;

/**
 * The key light: west-south-west, 53° up. A slot canyon IS its light — an afternoon sun
 * that lights the west face of every fin and lays about ONE CELL of shade east across the
 * slot beside it, so a 2-cell corridor reads as one lit column and one shaded one.
 *
 * The first pass put it at 40°, which is what a canyon photograph looks like and what a
 * canyon RENDER cannot afford: a full cliff's worth of throw filled both narrow slots
 * edge to edge,
 * and a shadowed cell field lit by nothing but a 0.6 hemisphere is black. The grid canon's
 * lesson holds — a grid scene lives or dies on cell readability, and the shadow that eats
 * two cells has stopped being depth and started being a hole.
 */
export const SUN_DIR: readonly [number, number, number] = [-0.5, 0.8, 0.33];
/** the shadow frustum covers the FRAME (canyon + the near desert), not just the playfield */
export const SHADOW_RADIUS = 12.5;

/**
 * The composition azimuth, shared with `index.ts`'s OrbitRig so the planting and the
 * camera can never disagree. THE AGENT MUST BE VISIBLE ON EVERY CELL IT CAN STAND ON, and
 * on an 11-unit grid full of cliffs that is a hard constraint on this one number,
 * not a matter of taste. Ray-traced from the composed standoff against the real cliff
 * columns (orb at y = 0.35, all 65 free cells, `the wisp is visible on every free cell` in
 * the test file re-measures it):
 *
 *     azimuth   0.36   0.33   0.30   0.25   0.20   0.16   0.07
 *     hidden       3      1      0      0      0      1      0
 *
 * (at the fin height, notch and arch ribbon of this file — they are one decision; see
 * FIN_HEIGHT_NORTH for the height/azimuth table. The odd 1 at 0.16 is the high arch, whose
 * sight-line runs dead along its own fin when the camera sits at that cell's own x.)
 *
 * The first pass sat at 0.36 on the claim that a fin is "looked along" there — it is not:
 * 0.36 rad of lateral offset over this standoff drives the sight-line a full CELL sideways
 * while it crosses the fin's own thickness, so the x = 3 fin hid all seven cells of the
 * corridor west of it, on the mandatory route out of the start. 0.25 keeps the 3/4 view the
 * composition is built on with every cell readable, and the rig's yaw limits are clamped to
 * ±0.05 around it so the frame cannot be orbited back into the occluding wedge.
 */
export const CAMERA_AZIMUTH = 0.25;
/** how far out along that azimuth the "keep it clear" wedge is centred */
export const CAMERA_STANDOFF = 15;

// ---------------------------------------------------------------------------
// small shared helpers (actors.ts imports these too)
// ---------------------------------------------------------------------------

export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
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
 * STYLE.md names it as the legal route to near-black, so "shade" is a blend like every
 * other color here.
 *
 * The afternoon key is 2.7 — nearly the noon 3.0 that taught the grid canon its albedo
 * law. Terracotta is a saturated, bright hue; unshaded it renders past the 0.85 bloom
 * threshold and 56 cliff cells fuse into one orange slab with no strata in it. Every
 * rock surface in this file is shaded 0.2–0.5.
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
 * disposing the sources. A hand-built prop (an arch, a basin, a mesa line) wants its
 * SOURCE normals plus a per-part stone palette baked in, so one mesh draws the lot.
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
 * geometry degrades to the correct maze instead of a guess.
 */
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

/** The two gaps that stitch the three slots together, from the env geometry when present. */
export function resolveGaps(grid: GridGeometry): readonly GridCell[] {
  const h = grid.hallways;
  return h && h.length > 0 ? h : HIGHDIMMAZE_GAPS;
}

/** true when a wall cell sits on the grid boundary — the mesa rim rather than a fin. */
export function isRimCell(grid: GridSize, x: number, y: number): boolean {
  return x === 0 || y === 0 || x === grid.w - 1 || y === grid.h - 1;
}

/**
 * The cliff LINE at a cell: the height the fin or rim would stand at with no arch beside
 * it. 0 for free cells. This is what an arch's span carries over the gap — the bridge is
 * the fin's own top, not the top of the eroded neck it springs from.
 */
export function wallNominalAt(grid: GridGeometry, x: number, y: number): number {
  if (!isWall(grid, x, y)) return 0;
  const t = grid.h > 1 ? clamp01(y / (grid.h - 1)) : 0;
  return isRimCell(grid, x, y)
    ? lerp(RIM_HEIGHT_NORTH, RIM_HEIGHT_SOUTH, t)
    : lerp(FIN_HEIGHT_NORTH, FIN_HEIGHT_SOUTH, t);
}

/**
 * How far this cliff cell is stepped down because a gap sits beside it, in [0, 1].
 * 1 everywhere except the fin cells flanking a gap ALONG THE FIN (the axis `archAxis`
 * reads off the walls array, so it follows a regenerated layout), where the fin erodes into
 * the saddle the arch springs from. The outer rim keeps its line — its silhouette is the
 * canyon's edge against the desert, and it never stands between the camera and a cell.
 */
export function archNotch(grid: GridGeometry, x: number, y: number): number {
  if (!isWall(grid, x, y) || isRimCell(grid, x, y)) return 1;
  let k = 1;
  const gaps = resolveGaps(grid);
  for (let i = 0; i < gaps.length; i++) {
    const gap = gaps[i];
    if (!gap) continue;
    const alongZ = archAxis(grid, gap) === 'alongZ';
    // along the fin's own run, and on the fin's own line — never a diagonal neighbour
    const along = alongZ ? Math.abs(y - gap.y) : Math.abs(x - gap.x);
    const across = alongZ ? Math.abs(x - gap.x) : Math.abs(y - gap.y);
    if (across !== 0 || along < 1) continue;
    const f = ARCH_NOTCH[along - 1];
    if (f !== undefined && f < k) k = f;
  }
  return k;
}

/**
 * Cliff height at a cell, in world units. 0 for free cells. A stationary function of the
 * cell, the walls array and the hallway list — no state, no camera, no history — so it can
 * be baked into instance matrices once and trusted forever.
 */
export function wallHeightAt(grid: GridGeometry, x: number, y: number): number {
  return wallNominalAt(grid, x, y) * archNotch(grid, x, y);
}

/** deterministic ±`amount` height wobble so no two cliff cells crown at the same line */
export function wallHeightJitter(x: number, y: number, seed: number, amount = 0.09): number {
  return 1 + (hash2f(x, y, seed ^ 0x51ed270b) - 0.5) * 2 * amount;
}

/**
 * Chebyshev distance from a free cell to the nearest wall, capped. THE SUN/SHADE READ:
 * in this maze every cell of the two narrow slots is 1 (a wall is always beside you) and
 * only the middle column of the wide slot reaches 2 — so a scan of the walls array, and
 * nothing else, paints the one sunlit spine in the canyon system.
 */
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

/**
 * Which way a gap's arch spans. READ, never assumed (grid canon): the jambs are the wall
 * cells flanking the gap, so a gap with rock north and south of it sits in a cliff that
 * runs along z, and the passage — and the arch's own span — is the perpendicular. Works
 * unchanged on any layout, including a regenerated one.
 */
export function archAxis(grid: GridGeometry, gap: GridCell): ArchAxis {
  const runsAlongZ = isWall(grid, gap.x, gap.y - 1) && isWall(grid, gap.x, gap.y + 1);
  return runsAlongZ ? 'alongZ' : 'alongX';
}

/** Local-geometry yaw for an arch built spanning ±x with its passage along ±z. */
export function archYaw(axis: ArchAxis): number {
  return axis === 'alongZ' ? Math.PI / 2 : 0;
}

/**
 * The two jamb heights an arch springs from, in its OWN local order: [−x end, +x end].
 * Local −x maps to world +z (south) after the `alongZ` yaw and to world −x (west) without
 * it, which is the whole reason this is read off the axis rather than assumed.
 *
 * `heights` is the jittered per-cell array when the world has one, so the arch meets the
 * cliff it actually grows out of rather than the cliff's nominal height.
 */
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

/**
 * Cliff top over a gap: the taller of the two flanking cells' own cliff LINE (never their
 * notched tops), plus the arch's own rise. The span carries the fin across the hole, so it
 * stands where the fin would have stood — that is what makes it a bridge over a saddle
 * instead of a lintel across a doorway.
 */
export function archHeight(grid: GridGeometry, gap: GridCell, _heights?: Float32Array): number {
  const alongZ = archAxis(grid, gap) === 'alongZ';
  const a = alongZ ? wallNominalAt(grid, gap.x, gap.y + 1) : wallNominalAt(grid, gap.x - 1, gap.y);
  const b = alongZ ? wallNominalAt(grid, gap.x, gap.y - 1) : wallNominalAt(grid, gap.x + 1, gap.y);
  return Math.max(a, b) + ARCH_RISE;
}

// ---------------------------------------------------------------------------
// the palette (every color a blend of tokens — STYLE.md)
// ---------------------------------------------------------------------------

export interface CanyonSkin {
  /** open sand, the sunlit spine of the wide slot */
  sandSun: THREE.Color;
  /** sand at the foot of a cliff — the cool shade the slots live in */
  sandShade: THREE.Color;
  /** the cliff mass */
  stone: THREE.Color;
  /** the four sedimentary beds, bottom to top */
  strata: readonly THREE.Color[];
  /** the desert outside the canyon */
  desert: THREE.Color;
  /** dry scrub / dead juniper */
  scrub: THREE.Color;
  /** distant mesas, already half air */
  farStone: THREE.Color;
  /** the warm bounce that rims every crown — a canyon's own reflected light */
  rim: THREE.Color;
  /** the oasis: water, reed green, and the shaft that finds it */
  water: THREE.Color;
  waterDeep: THREE.Color;
  reed: THREE.Color;
  shaft: THREE.Color;
  /** the dust that hangs in the slots */
  dust: THREE.Color;
}

/**
 * Terracotta, built from `high-dim-maze`'s own accent plus `mountain-car`'s alpenglow
 * ember for the deep beds and dusk's warm tan for the sand — three entries from the
 * shared table, so the gallery still reads as one preserve while this location reads as
 * red rock. The one cool note in the whole canyon is the oasis, and that is the point.
 */
export function canyonSkin(palette: Palette, tokens: typeof TOKENS): CanyonSkin {
  const terracotta = tokens.accents['high-dim-maze'] ?? tokens.celebrationGold;
  const ember = tokens.accents['mountain-car'] ?? terracotta;
  const teal = tokens.accents['deep-sea'] ?? tokens.uiCyan;
  const leaf = tokens.accents['cart-pole'] ?? palette.groundA;
  const tan = TOKENS.palettes.dusk.groundA;
  const hot = TOKENS.palettes.dusk.skyHorizon;

  return {
    sandSun: shade(blendTokens(blendTokens(tan, terracotta, 0.62), palette.sun, 0.16), 0.08),
    // "cool shade" is a HUE shift, not a black level: the slots are lit by sky bounce off
    // their own walls, so the shaded sand goes bluer — never much darker, because the
    // sun's cast shadow is already stacked on top of it and albedo would double-count.
    sandShade: shade(blendTokens(blendTokens(tan, terracotta, 0.5), palette.hemiSky, 0.2), 0.44),
    // THE TWO BIG SURFACES AN AERIAL CAMERA SEES ARE THE SAND AND THE CLIFF TOPS, and if
    // they land on the same value the whole frame is one wash — which is exactly what the
    // first three passes produced. The structure is now explicit: dry olive desert (dark,
    // cool) frames deep terracotta rock (dark, saturated) framing warm ochre sand (light),
    // with the cast shadows cutting the slots cool. Every number below is authored against
    // that ladder, in LINEAR space (blendTokens works there, and `instanceColor` is a
    // linear multiplier over the material's own linear colour).
    stone: shade(blendTokens(terracotta, ember, 0.54), 0.63),
    strata: [
      // bottom bed: the deepest, wettest rock — ember shaded hard
      shade(blendTokens(ember, terracotta, 0.2), 0.72),
      // a bright bone bed: the ONE high line on a cliff face, and the reason the beds
      // read as sedimentary rather than as noise
      shade(blendTokens(terracotta, palette.sun, 0.42), 0.28),
      // mid terracotta
      shade(blendTokens(terracotta, ember, 0.4), 0.55),
      // the crown: sun-bleached caprock. It is the biggest surface an aerial camera sees,
      // so it is shaded HARDER than its hue suggests — the grid canon's albedo law with
      // an extra margin, because 56 lit cliff tops that cross the bloom threshold fuse
      // into one orange slab and the canyon loses its beds.
      shade(blendTokens(terracotta, palette.sun, 0.16), 0.3),
    ],
    // The desert must NOT be the same colour as the canyon, or the whole frame is one
    // tan wash and the playfield stops being a place. It keeps the plateau's dry olive
    // (afternoon's own ground token) with only a breath of terracotta in it, so the cut
    // reads as red rock opened in a scrub plain.
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

// ---------------------------------------------------------------------------
// scatter exclusions
// ---------------------------------------------------------------------------

/** Keep a clear disc around every cell CENTER — props live in the joints, never on a cell. */
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
// the desert height field
// ---------------------------------------------------------------------------

/**
 * Where the plateau ends and falls away, in units OUTSIDE the canyon footprint.
 *
 * This number is set by the FRAME, not by taste. The camera stands 12.1 up at a horizontal
 * radius of 13.3, and its top ray leaves at 20.7° of depression — which meets flat ground
 * about 18.7 units out from the origin on the far side. An edge beyond that is simply off
 * the top of the picture (the first attempt put it at 21.5 and nothing changed). At 8.5 the
 * lip lands ~8% down from the top and everything above it is sky.
 */
export const MESA_EDGE = 8.5;
export const MESA_EDGE_WIDTH = 3.5;
/** how far the country below the plateau sits under it */
export const MESA_DROP = 10.5;

/**
 * The country the canyon is cut into: dunes OUTSIDE the playfield, identically flat inside
 * it, and — the whole reason this function is more than noise — a PLATEAU EDGE.
 *
 * A grid scene at a 41° camera fills every pixel with ground: the plain runs out past the
 * top of the frame and there is no sky, no horizon and no depth in the upper third. Ending
 * the mesa at `MESA_EDGE` and dropping the country ~9 units below it puts the far lip at a
 * steeper depression than the frame's top ray, so everything above it is SKY — and the
 * distant buttes standing on that lower country become a skyline. This is the one move
 * that turned the upper third of the frame from filler into distance.
 *
 * The weight is identically 0 inside the footprint plus a margin, so `makeTerrain`'s
 * flatten mask has nothing to fight and the ground under the sand slabs is exactly
 * `CANYON_SINK`.
 */
export function makeMesaHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.max(Math.abs(x), Math.abs(z)) - CANYON_HALF;
    const w = smoothstep01(out / 4);
    if (w <= 0) return 0;
    const n1 = valueNoise2(x * 0.038 + 3.7, z * 0.038 + 9.1, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.095 + 1.3, z * 0.095 + 4.7, 64, seed ^ 0x51ed270b) - 0.5;
    const n3 = valueNoise2(x * 0.24 + 7.9, z * 0.24 + 2.3, 64, seed ^ 0x2545f491) - 0.5;
    const dunes = n1 * 3.0 + n2 * 1.4 + n3 * 0.5;
    // the edge wanders: a straight cliff line would read as the end of a diorama
    const wander = (valueNoise2(x * 0.06 + 21.3, z * 0.06 + 5.9, 64, seed ^ 0x7f4a7c15) - 0.5) * 4.5;
    const fall = smoothstep01((out - (MESA_EDGE + wander)) / MESA_EDGE_WIDTH);
    return w * dunes - fall * MESA_DROP;
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
  /** probe objects: the arches, in `grid.hallways` order. POSITION ONLY. */
  archRoots: THREE.Object3D[];
  gaps: readonly GridCell[];
  /** per-cell cliff height in world units, row-major (0 on free cells) */
  wallHeights: Float32Array;
  /** ctx.time() — pause freezes dust, sand-fall, shimmer and scrub together */
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
  const palette: Palette = ctx.tokens.palettes.afternoon;
  const skin = canyonSkin(palette, ctx.tokens);
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'slotCanyons';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];

  // ---- sky (exactly one) ----
  const sky = skyFromPalette(palette, 'afternoon', tier, {
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

  // ---- the one light budget: sun + hemi + 3 points (oasis / wisp / celebration) ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 3,
    pointDistance: 6,
    sunDistanceScale: 2.4,
  });
  group.add(rig.group);
  owned.push(rig);
  // A SLOT CANYON'S FLOOR IS LIT BY ITS OWN WALLS. The key light reaches maybe half of it
  // at any hour; everything else is sky and bounce, which in this kit is the hemisphere.
  // At the palette's own 0.6 the shadowed corridors rendered as black troughs and the grid
  // stopped reading — this is the one light value the scene tunes, and it buys back the
  // cells the cast shadows take. No extra lights: the budget is still sun + hemi + 3.
  rig.hemi.intensity = palette.hemiIntensity * 1.35;
  rig.setShadowFocus({ x: 0, y: 0.5, z: 0 }, SHADOW_RADIUS);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the desert the canyon is cut into ----
  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    tier,
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
    receiveShadow: shadows,
    castShadow: false,
    name: 'canyonDesert',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  // ---- the canyon: ONE sand draw, ONE cliff draw, straight off `grid.walls` ----
  // White base + full-color tint: the instance color MULTIPLIES the material color, so
  // this is how a cell field gets arbitrary per-cell hues without a second material.
  const white = palette.sun;
  const field = buildCellField(
    grid,
    {
      floorColor: white,
      wallColor: white,
      floorUnderWalls: true,
      floorTint: (x, y) => {
        if (isWall(grid, x, y)) return skin.strata[0] ?? skin.stone;
        // the sun/shade read: open cells warm, cliff-foot cells cool
        const open = clamp01((wallDistance(grid, x, y) - 1) / 1.5);
        return blendTokens(skin.sandShade, skin.sandSun, open);
      },
      wallTint: () => skin.stone,
      wallShape: 'block',
      wallHeight: WALL_BASE_HEIGHT,
      // exactly 0: adjacent cliff cells must FUSE into one continuous fin, not stand as a
      // row of separate blocks. The height jitter below is what keeps the crown line alive.
      wallInset: 0,
      // a hairline crack between sand slabs — dried-mud crazing, and the grid stays legible
      floorInset: 0.03,
      floorThickness: 0.16,
      jitter: 0.08,
      rimColor: skin.rim,
      rimStrength: 0.26,
      seed: ctx.seed,
      name: 'canyonCells',
    },
    tier,
  );
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  // ---- per-cell cliff heights (the kit has one global height; this is the canyon) ----
  const wallHeights = applyWallHeights(field, grid, ctx.seed);

  // ---- the strata: one extra instanced draw for all 56 cliff cells ----
  const strataGeometry = makeStrataGeometry(skin, tier);
  const strataMaterial: OakMaterial = oakMaterial({
    // the SAME white base the cell field uses, so a bed and the block it caps are graded
    // by identical maths and can never drift apart
    color: white,
    vertexColors: true,
    flatShading: true,
    roughness: 0.96,
    rimColor: skin.rim,
    rimStrength: 0.5,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.26,
    tier,
    name: 'canyonStrataMaterial',
  });
  owned.push({ dispose: () => strataMaterial.dispose() });
  owned.push({ dispose: () => strataGeometry.dispose() });
  {
    let count = 0;
    for (let i = 0; i < wallHeights.length; i++) if ((wallHeights[i] ?? 0) > 0) count++;
    const mesh = new THREE.InstancedMesh(strataGeometry, strataMaterial, Math.max(1, count));
    mesh.name = 'canyonStrata';
    mesh.castShadow = false; // the cliff mass already casts; doubling it only costs texels
    mesh.receiveShadow = shadows;
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

  // ---- the two arches, one per gap cell ----
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
    // a high floor so the soffit stays painterly instead of going to black
    toonFloor: 0.42,
    tier,
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
        tier,
        archHeight(grid, gap, wallHeights),
        jambs,
        ctx.seed + i * 7919,
      );
      const mesh = new THREE.Mesh(geometry, archMaterial);
      mesh.name = `canyonArch${i}`;
      // READ the orientation: the arch spans between the wall cells flanking the gap.
      mesh.rotation.y = archYaw(archAxis(grid, gap));
      mesh.castShadow = shadows;
      mesh.receiveShadow = shadows;
      root.add(mesh);
      group.add(root);
      archRoots.push(root);
      owned.push({ dispose: () => geometry.dispose() });
    }
  }

  // ---- the distant mesa line: one draw of pure silhouette ----
  {
    const geometry = makeMesaLineGeometry(skin, ctx.seed, tier);
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
      tier,
      name: 'canyonMesaLineMaterial',
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'canyonMesaLine';
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    group.add(mesh);
    owned.push({ dispose: () => { geometry.dispose(); material.dispose(); } });
  }

  // ---- desert planting, all of it OUTSIDE the canyon ----
  // MountainCar's law: nothing lives between the camera and the physics. At this azimuth
  // the camera stands over the southern rim, so the wedge under it stays bare.
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
    tier,
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
    tier,
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

  // proper boulders: the desert needs forms with a SILHOUETTE, not just pebble texture
  const outcrops = scatterRocks({
    count: 30,
    area: { x: 0, z: 0, halfX: 19, halfZ: 19, innerRadius: CANYON_HALF + 4.2 },
    seed: ctx.seed ^ 0x18ce,
    tier,
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
    tier,
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

  // a little scree at the cliff feet, INSIDE the canyon but never on a cell centre
  const scree = scatterRocks({
    count: 120,
    area: { x: 0, z: 0, halfX: CANYON_HALF - 0.6, halfZ: CANYON_HALF - 0.6 },
    seed: ctx.seed ^ 0x2d41,
    tier,
    exclude: excludeAny(excludeWallCells(grid), excludeCellCenters(grid, 0.36)),
    // the bone bed makes a beautiful line on a cliff and a snowfield on a canyon floor:
    // scree takes the MID terracotta instead, so it reads as fallen rock, not as litter
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

  // ---- idle life ----
  // The kit's particle presets size their motes for a ~20-unit field; this canyon is 11
  // units across with the camera ~19 out, so `gl_PointSize`'s 60/distance term wants
  // sizes around 1, not the preset's 3–9.
  const motes: AmbientParticlesHandle[] = [];
  const dust = makeAmbientParticles({
    preset: 'dust',
    count: 220,
    bounds: { x: 0, y: 1.25, z: 0, halfX: CANYON_HALF - 0.4, halfY: 0.55, halfZ: CANYON_HALF - 0.4 },
    seed: ctx.seed ^ 0x7d21,
    tier,
    color: skin.dust,
    // canon: kit presets size their motes for a ~20-unit field. At 1.05 in an 11-unit
    // canyon the dust read as confetti lying on the sand, not as air.
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

  // ---- sand-fall: thin curtains trickling off cliff LIPS found in the walls array ----
  for (const spot of sandFallSpots(grid, ctx.seed, tier === 'hero' ? 4 : tier === 'medium' ? 2 : 0)) {
    const fall = makeAmbientParticles({
      preset: 'dust',
      count: 46,
      bounds: { x: spot.x, y: spot.top * 0.5, z: spot.z, halfX: 0.07, halfY: spot.top * 0.5, halfZ: 0.26 },
      seed: ctx.seed ^ (0x4a13 + spot.index * 977),
      tier,
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

  // ---- heat shimmer: warm haze lying on the sand ----
  const shimmer = makeMistPlanes({
    size: [HIGHDIMMAZE_W * CELL, HIGHDIMMAZE_H * CELL],
    color: blendTokens(skin.dust, palette.sun, 0.4),
    seed: ctx.seed ^ 0x6b21,
    tier,
    layers: 2,
    baseY: 0.1,
    spread: 0.34,
    // a whisper. At 0.13 three warm planes lying over shadowed sand read as SMOG, not as
    // heat — the canyon looked like it was on fire in the wrong way.
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

// ---------------------------------------------------------------------------
// cliff shaping
// ---------------------------------------------------------------------------

/**
 * Re-compose every wall instance so each cliff cell stands at its own height. The kit
 * builds one wall geometry at `WALL_BASE_HEIGHT` with its base at y = 0, so a pure y
 * scale keeps the foot on the sand and moves only the crown.
 *
 * Returns the per-cell heights, row-major, 0 on free cells — the strata layer and the
 * arches read the same array, so a cliff and its beds can never disagree.
 */
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

/**
 * The sedimentary beds, built in NORMALIZED height space (y ∈ [0, 1]) so one geometry
 * serves every cliff cell: the instance's y scale is that cell's height, and a taller
 * cliff simply gets thicker beds — which is what a deeper cut through the same rock
 * actually looks like.
 *
 * Each bed is a slab slightly PROUD of the 1.0-unit cliff block in x/z, so it reads as
 * relief from any angle; the top one is the caprock, and it is the edge the afternoon sun
 * rims.
 */
export function makeStrataGeometry(skin: CanyonSkin, tier: QualityTier): THREE.BufferGeometry {
  // [center y, half thickness, xz spread]. THE CAP MUST CLEAR THE BLOCK TOP: the cell
  // field's wall geometry ends at exactly y = 1 in this normalized space, so a cap
  // centred at 0.972 with half 0.028 puts its top face EXACTLY there — two coplanar
  // quads, and the whole crown of the canyon z-fights into orange and cream triangles.
  // 1.0 ± 0.032 lifts the caprock a hair above the mass it caps, which is also what a
  // resistant caprock actually does.
  const beds: readonly (readonly [number, number, number])[] =
    tier === 'thumb'
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
    const tone = tier === 'thumb' ? (i === 0 ? 0 : 3) : i;
    parts.push({ geometry: g, color: skin.strata[tone] ?? skin.stone });
  }
  return mergeColoredParts(parts, 'canyonStrata');
}

/**
 * A natural rock arch: the cliff carried over the gap, with the opening eaten out from
 * underneath. Built in LOCAL space around the gap cell's centre spanning ±x with the
 * passage along ±z — the four-rooms gate convention — so a fin that runs along z is this
 * same geometry yawed 90°.
 *
 * THE SOFFIT SPRINGS FROM THE JAMB TOPS, NOT FROM THE FLOOR. This is the whole difference
 * between an arch you can see under and a wall with a hole in it. The first pass ran the
 * underside down to y = 0 at both ends, which put a solid wedge of rock inside the gap
 * cell's own footprint — from a camera looking along the fin (and every camera this scene
 * allows looks along the fin) that wedge is exactly what stands between the lens and an
 * agent standing in the arch, and the arch itself read as a rounded lump. Springing from
 * the notched shoulders instead, the underside runs `jamb + (opening − jamb)·(1 − |t|^6)^0.35`:
 * pinned on the rock it grows out of, flat across the middle of the cell — so the daylight
 * under the span is a full cell wide and the sight-line into the gap arrives under it.
 */
export function makeArchGeometry(
  skin: CanyonSkin,
  tier: QualityTier,
  height: number,
  jambs: readonly [number, number],
  seed: number,
): THREE.BufferGeometry {
  const spans = tier === 'thumb' ? 8 : tier === 'medium' ? 12 : 20;
  const half = CELL * ARCH_SPAN_HALF;
  const depth = CELL * 0.5;
  const opening = Math.max(0.55, Math.min(ARCH_OPENING, height - ARCH_BAND));

  // THE JAMB LINE: the straight run between the two shoulder tops. Both the crest and the
  // soffit are measured from it, so every part of the arch meets the rock it grows out of.
  const [endA, endB] = jambs;
  const jambLine = (t: number): number => endA + (endB - endA) * (t + 1) * 0.5;

  // A FLAT-TOPPED opening, not a semicircle: `(1 − |t|^6)^0.35` holds the full opening
  // across the whole gap cell and drops to the jamb only in the last few percent, which is
  // what wind carves and what keeps the daylight under the span a full cell wide. A
  // circular arch reads as masonry, and there is no masonry in a canyon.
  const soffit = (t: number): number => {
    const a = Math.min(1, Math.abs(t));
    const shape = Math.pow(Math.max(0, 1 - Math.pow(a, 6)), 0.35);
    const wobble = 1 + 0.06 * Math.sin(t * 7.3 + seed * 0.017) + 0.035 * Math.sin(t * 15.1 - seed * 0.011);
    const base = jambLine(t);
    return base + Math.max(0, opening - base) * shape * wobble;
  };
  // THE CREST MUST LAND ON THE ROCK IT SPRINGS FROM. A flat lintel at the crown height
  // leaves both end faces standing proud of their jambs — and since one jamb here is the
  // low southern rim, the first pass presented the camera with a bare bright slab where an
  // arch should be. The crest is a hump over the LINE BETWEEN THE TWO JAMB TOPS, so at
  // |t| = 1 it is exactly the neighbouring cliff and the ends are buried in it.
  const crest = (t: number): number => {
    const base = jambLine(t);
    const hump = Math.max(0, height - base) * Math.pow(Math.max(0, 1 - t * t), 0.8);
    const erode = 0.035 * Math.sin(t * 5.1 + seed * 0.023) * (1 - t * t);
    return base + hump + erode;
  };

  // THE SPAN IS A RIBBON, AND ITS LEGS ARE THE THINNEST PART OF IT. An arch as thick as the
  // fin it interrupts is a wall with a hole in it: from a camera looking along the fin —
  // which is every camera this scene allows — the gap cell becomes a roofed tunnel and the
  // agent standing in it cannot be seen at all. Erosion does the opposite: what survives is
  // a ribbon much thinner than the fin, standing on legs thinner still, and a sight-line
  // arriving along the fin passes BESIDE those legs to the cell floor behind them. Measured
  // from the composed camera, the leg half-width is what decides whether the orb in the
  // arch reads (0.5 = nothing, 0.15 = most of it).
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
    // The band never pinches shut over the opening — but it TAPERS TO NOTHING at the
    // springing, where the arch is already the shoulder it lands on. A constant minimum
    // band there leaves a thin lip of rock sticking out over the shoulder's crown, which is
    // exactly the height a grazing sight-line into the gap passes through.
    const taper = 1 - Math.min(1, Math.abs(t));
    hi[i] = Math.max(crest(t), under + ARCH_BAND * 0.5 * taper);
    dz[i] = depthAt(t);
  }

  const positions: number[] = [];
  const colors: number[] = [];
  const face = new THREE.Color(skin.strata[2] ?? skin.stone);
  // THE UNDERSIDE IS THE BRIGHTEST ROCK IN THE ARCH. A soffit faces down, so it never
  // sees the key light and is rendered by hemisphere alone; painted with the dark bed it
  // is a black slot and the span reads as a crack. Painted with the bone bed it reads as
  // the bounce-lit tunnel a real canyon arch has under it, and the daylight through the
  // gap is what tells you it is an arch at all from a camera looking at it edge-on.
  const soffitColor = new THREE.Color(skin.strata[1] ?? skin.stone);
  const capColor = new THREE.Color(skin.strata[3] ?? skin.stone);
  // the end quads are buried in the flanking cliff — they are closure, not decoration
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
    // the two big visible faces, looking down each corridor
    quad([x0, l0, d0], [x1, l1, d1], [x1, h1, d1], [x0, h0, d0], face);
    quad([x1, l1, -d1], [x0, l0, -d0], [x0, h0, -d0], [x1, h1, -d1], face);
    // the soffit — the eroded underside you walk beneath
    quad([x0, l0, -d0], [x1, l1, -d1], [x1, l1, d1], [x0, l0, d0], soffitColor);
    // the caprock
    quad([x0, h0, d0], [x1, h1, d1], [x1, h1, -d1], [x0, h0, -d0], capColor);
  }
  // the two ends, buried in the flanking cliff cells (they close the solid)
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

/**
 * The horizon: flat-topped mesas standing off in the haze. Pure silhouette — they exist so
 * the sky has something to end against and the canyon has a country to be in.
 */
export function makeMesaLineGeometry(skin: CanyonSkin, seed: number, tier: QualityTier): THREE.BufferGeometry {
  const rng = mulberry32((seed | 0) ^ 0x1d3b7c51);
  const count = tier === 'thumb' ? 6 : 10;
  const parts: ColoredPart[] = [];
  for (let i = 0; i < count; i++) {
    // the camera stands over the southern rim; a butte planted in that wedge would sit
    // between the lens and the playfield, which is the one thing the visual canon forbids
    const spread = Math.PI * 2 - 1.1;
    let angle = CAMERA_AZIMUTH + 0.55 + ((i + rng() * 0.55) / count) * spread;
    angle = ((angle + Math.PI) % (Math.PI * 2)) - Math.PI;
    // close enough that their TOPS clear the plateau lip and stand against the sky, far
    // enough that their bases stay hidden behind it — a skyline, not a wall
    const radius = 18 + rng() * 11;
    const w = 5 + rng() * 11;
    const d = 4 + rng() * 8;
    const h = 7.5 + rng() * 6;
    const cx = Math.sin(angle) * radius;
    const cz = Math.cos(angle) * radius;
    const base = -MESA_DROP - 1;
    const box = new THREE.BoxGeometry(w, h, d);
    // a stepped shoulder: two stacked blocks read as a mesa, one reads as a crate
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
  /** the cliff height at that lip */
  top: number;
}

/**
 * Cliff LIPS: wall cells with a free neighbour, so a curtain of sand always falls into a
 * corridor and never into solid rock. Found by scanning the walls array; the choice among
 * them is a seeded draw, so a lane's canyon trickles in the same four places forever.
 */
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
