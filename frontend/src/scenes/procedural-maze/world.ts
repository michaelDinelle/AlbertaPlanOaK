// The Overgrown Ruins — the static world (ProceduralMaze-cont-v0).
//
// THE ONE IDEA OF THIS SCENE: there is NO LAYOUT HERE. The env regenerates its 9×9
// interior from the lane seed at fleet init, so every lane is a different ruin, and the
// only thing an artisan can author is a MASONRY SYSTEM: a pure function
// `skinWallCell(grid, x, y, seed)` that turns "cell (x, y) is a 1 in `grid.walls`" into a
// specific ruined block — how many courses of it are still standing, how crooked the
// fallen lintel on top sits, how far the moss has climbed, whether ivy hangs off it.
// Feed it a different `walls` array and a different ruin comes out, correct by
// construction. Nothing in this file re-derives the maze; `grid.walls` is read and only
// read, and the carved solution path is never touched (it is `kit/debug`'s alone).
//
// PATTERNS WORTH COPYING
//  * THE RUIN IS BUILT IN COURSES, and the base course is the kit's. `buildCellField`
//    lays one uniform foundation block on every wall cell (jitter 0 — the ground line of
//    a ruin is the part that survives), and this file adds two more INSTANCED courses on
//    top plus moss cushions and ivy. Because the scene owns courses 2 and 3, it knows
//    every block's exact crown height, which is what lets moss, ivy and the corner towers
//    land on the stone instead of floating over it. Reading the kit's own internal
//    height jitter back out would have been a guess — and passing that jitter in was
//    WORSE than a guess: `CellFieldStyle.jitter` scales the wall block's HEIGHT as well
//    as its tone, so a foundation asked for at 0.5 was really 0.5 ± 0.045 and half the
//    second courses in the ruin floated while the other half sank (measured: 32 up, 33
//    down, max 0.045 — 4 px of daylight under a block at the composed standoff). The kit
//    now lays the foundation FLAT and the per-block tone variation the jitter also bought
//    is generated here, in `floorTintFor` / `wallTintFor`, where it cannot move stone.
//    `blockHalfWidthAt` is the other half of that bargain: the mason knows its own
//    silhouette, so anything that has to touch a block (ivy today) asks the block.
//  * ORIENTATION IS READ, NEVER ASSUMED (the four-rooms gate lesson, on a layout nobody
//    wrote): a free cell walled to its north AND south is a doorway in a wall that runs
//    east–west, so its arch spans north–south. `archCells()` is that sentence in code and
//    it is correct on all 2^81 possible mazes.
//  * THE FOUR CORNERS ARE THE ONE FIXED FACT. `procMazeIsWall` walls the whole border on
//    every seed, so (0,0), (10,0), (0,10) and (10,10) are guaranteed masonry — the only
//    cells a scene may count on. They become the four broken towers that give every ruin,
//    however random its middle, the same recognisable silhouette.
//
// LIGHT: noon, ~61° up and from the north-west, so the shadows are short enough to keep
// the grid readable but long enough that a 1.1-unit block throws a real edge across the
// flagstones. The sun direction is shared with the sky AND with the goal's sunbeam, so
// the shaft, the sun disc and the shadows can never disagree about where the sun is.

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
import { oakMaterial, oakUniforms, type OakMaterial } from '@kit/materials';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { hash2f, rngAt, streamFor } from '@kit/rng';
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
import { PROCMAZE_H, PROCMAZE_W, proceduralMazeDecoder } from '@sim/envs/proceduralmaze';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';

// ---------------------------------------------------------------------------
// world constants (1 cell = 1 world unit — the ruin really is 11 × 11)
// ---------------------------------------------------------------------------

/** half the ruin's footprint, derived from the env's own grid size */
export const RUIN_HALF = (PROCMAZE_W * CELL) / 2;
export const TERRAIN_SIZE = 44;
export const TERRAIN_SEGMENTS = 128;
/** the meadow sits a few centimetres under the flagstones, so the joints read as soil */
export const RUIN_SINK = -0.07;
/**
 * Fog reach. `applyFog` turns extent into density through `fogDensityHint` (noon = 0.25),
 * so this puts full density ~130 units out: the forest ring at ~20 units keeps its colour
 * and only the far tree line hazes. Noon fog is nearly white — over-fogging a noon scene
 * reads as a blown exposure, not as air.
 */
export const FOG_EXTENT = TERRAIN_SIZE * 12;
/**
 * The sun: ~61° up, from the north-west. High enough that a 1.1-unit block throws about
 * 0.6 of a cell (the grid stays legible), low enough that the goal's sunbeam leans a
 * visible 2.7 units across the ruin instead of dropping straight down like a spotlight.
 */
export const SUN_DIR: readonly [number, number, number] = [-0.36, 0.874, -0.325];
/** the shadow frustum covers the FRAME (ruin + the near forest), not just the playfield */
export const SHADOW_RADIUS = 12;

// ---------------------------------------------------------------------------
// THE COMPOSITION
//
// The hero orbit lives HERE, next to the planting it has to agree with, and the clear
// wedge the forest leaves for the camera is DERIVED from it rather than typed twice.
// index.ts reads these; nobody re-states them.
// ---------------------------------------------------------------------------

/** a 3/4 aerial from the south-east */
export const CAMERA_AZIMUTH = 0.62;
/** 0.73 rad off the zenith = 48.2° above the horizon */
export const CAMERA_POLAR = 0.73;
/**
 * The look target slides this far along the azimuth, TOWARD the near corner. The ruin's
 * projected diagonal is not symmetric about the orbit centre — the near corner is closer
 * to the camera, so it subtends more of the frame than the far one — and centring on the
 * grid origin therefore hangs the whole south-east tower off the bottom edge. Sliding the
 * pivot 1.2 units toward the camera re-centres the diagonal (measured: the far crown lands
 * at ndc.y +0.55, the near footing at −0.91, instead of +0.84 / −1.17).
 */
export const CAMERA_TARGET_BIAS = 1.2;
/** the pivot sits a little above the flagstones, at about the height of a fallen lintel */
export const CAMERA_TARGET_Y = 0.15;
/**
 * Distance is arithmetic, not taste — and the arithmetic has to include the CORNERS, not
 * just the footprint. Seen from azimuth a, an 11-cell square presents 11·(|cos a| + |sin a|)
 * across the frame (15.3 units here), and a 40° vertical FOV at 16:9 covers 1.29·d
 * horizontally, so d clears the ruin's WIDTH at 11.8. What actually binds is the vertical:
 * the four broken towers are 1.74 units of silhouette standing on the frame's own corners,
 * and the near one is a whole cell closer than the pivot. Projecting all four tower
 * footprints (±0.5 cell, base and crown) through this camera puts the worst |ndc| at 0.914
 * vertically and 0.890 horizontally at 4:3 — the ruin is framed with ~9% margin at 16:9,
 * 4:3 AND anywhere in the ±0.6 rad of yaw the rig allows. `procedural-maze.test.ts` holds
 * that measurement; 17.0 cropped the south-east tower clean out of the picture.
 */
export const CAMERA_DISTANCE = 19.0;
/**
 * Where the camera actually stands on the ground — the centre of the "keep it clear"
 * wedge in the forest. Derived, so the trees can never drift out of agreement with the
 * frame again.
 */
export const CAMERA_STANDOFF = CAMERA_TARGET_BIAS + CAMERA_DISTANCE * Math.sin(CAMERA_POLAR);

/**
 * The foundation course. Every wall cell has at least this much stone standing, which is
 * what makes the maze read as a maze from above; everything above it is weather.
 */
export const BASE_COURSE = 0.5;
/** the tallest a wall cell can finish (base + upper + capstone) */
export const MAX_CROWN = BASE_COURSE + 0.46 + 0.22;

/** the block plan: how far each course is inset from the cell edge, so it steps inward */
export const BASE_INSET = 0.02;
export const UPPER_INSET = 0.11;
export const CAP_INSET = 0.21;

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
 * STYLE.md names it as the legal route to near-black, so shading is a blend like every
 * other colour here.
 *
 * The noon key is 3.0. A limestone bright enough to look like limestone in isolation
 * renders past the bloom threshold under it and the whole ruin fuses into one white slab
 * with the cell grid smeared out of it. Shaded, the same hue reads as sunlit stone AND
 * keeps its joints: ground albedo belongs around 140–170 sRGB, never 220.
 */
export function shade(color: THREE.ColorRepresentation, k: number): THREE.Color {
  return blendTokens(color, TOKENS.palettes.night.skyTop, k);
}

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

/**
 * Merge pre-transformed parts into ONE geometry carrying a flat per-part vertex colour,
 * disposing the sources. A hand-built stone prop (an arch, a broken tower) wants its
 * SOURCE normals plus a per-part stone palette baked in, so one mesh draws the whole prop.
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
 * The grid the scene draws.
 *
 * The host always hands us the decoder's geometry, and the fallback is the DECODER'S OWN
 * generator seeded with the lane seed — never a scene re-derivation. The test for "is
 * this really a ProceduralMaze grid" is `carvedPath`: only this env's decoder emits one,
 * so the harness's generic 11×11 placeholder (a bare border ring) is rejected and
 * regenerated into the real maze instead of being drawn as an empty courtyard.
 */
export function resolveGrid(geometry: EnvGeometry, seed: number): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === PROCMAZE_W && g.h === PROCMAZE_H && (g.carvedPath?.length ?? 0) > 0) return g;
  const st = proceduralMazeDecoder.initStatic(seed);
  const resolved = proceduralMazeDecoder.geometry(st, seed).grid;
  if (!resolved) throw new Error('procedural-maze: decoder produced no grid geometry');
  return resolved;
}

/** Wall neighbours of a cell in the four cardinal directions (out of bounds counts). */
export function wallNeighbours(grid: GridGeometry, x: number, y: number): number {
  let n = 0;
  if (isWall(grid, x, y - 1)) n++;
  if (isWall(grid, x, y + 1)) n++;
  if (isWall(grid, x - 1, y)) n++;
  if (isWall(grid, x + 1, y)) n++;
  return n;
}

/** The four grid corners — the only cells `procMazeIsWall` guarantees on every seed. */
export function cornerCells(grid: GridSize): readonly GridCell[] {
  return [
    { x: 0, y: 0 },
    { x: grid.w - 1, y: 0 },
    { x: 0, y: grid.h - 1 },
    { x: grid.w - 1, y: grid.h - 1 },
  ];
}

/** An arch: the free cell it stands on, and the axis its opening faces. */
export interface ArchCell {
  cell: GridCell;
  /**
   * `true` when the passage runs east–west (the flanking walls are north and south), so
   * the arch spans north–south and you walk through it along x.
   */
  eastWest: boolean;
}

/**
 * Every free cell that is a DOORWAY: walled on two opposite sides and open on the other
 * two. Read straight off `grid.walls`, so it is right on any regenerated layout — this is
 * the four-rooms moon-gate lesson ("a gate faces the axis its flanking wall cells run
 * along") applied where nobody wrote the wall.
 */
export function archCells(grid: GridGeometry): ArchCell[] {
  const out: ArchCell[] = [];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isWall(grid, x, y)) continue;
      const n = isWall(grid, x, y - 1);
      const s = isWall(grid, x, y + 1);
      const w = isWall(grid, x - 1, y);
      const e = isWall(grid, x + 1, y);
      if (n && s && !w && !e) out.push({ cell: { x, y }, eastWest: true });
      else if (w && e && !n && !s) out.push({ cell: { x, y }, eastWest: false });
    }
  }
  return out;
}

/** arches actually built, per tier — decoration scales, the reading never changes */
export function archBudget(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 3 : tier === 'medium' ? 6 : 9;
}

// ---------------------------------------------------------------------------
// THE WALL-SKINNING SYSTEM
// ---------------------------------------------------------------------------

/**
 * What one wall cell looks like after a few centuries. Every field is a pure function of
 * `(walls, x, y, seed)`; nothing here remembers a visit order, so the same cell is the
 * same block whether it is built first or last, and the same seed rebuilds the same ruin
 * bit for bit after a scrub, a tier switch or a page reload.
 */
export interface WallSkin {
  /** second course height in world units; 0 = only the foundation survived */
  upper: number;
  /** fallen-lintel slab height; 0 = none (implies `upper` is the crown) */
  cap: number;
  /** finished crown height = BASE_COURSE + upper + cap */
  crown: number;
  /** yaw of the second course, radians — masonry settles crooked */
  upperYaw: number;
  /** yaw of the capstone: a slab that fell back into place never lines up */
  capYaw: number;
  /** capstone tilt about x, radians */
  capTilt: number;
  /** 0..1 moss coverage — damp, hemmed-in blocks go green, open ones bleach */
  moss: number;
  /** a moss cushion sits on the crown */
  cushion: boolean;
  /** ivy strands hanging off this block (0 when it is too low to hold any) */
  ivy: number;
  /**
   * Which face the ivy hangs on. The placement yaws by `face·π/2` and steps out along
   * (sin a, cos a), so: 0 = +z, 1 = +x, 2 = −z, 3 = −x. (It read "0 = −z … 2 = +z" until
   * the faces were measured against the matrices the loop actually writes.)
   */
  ivyFace: number;
}

const UPPER_MIN = 0.16;
const UPPER_SPAN = 0.3;
const CAP_MIN = 0.09;
const CAP_SPAN = 0.1;
/** below this crown a block has nothing for ivy to hang from */
const IVY_MIN_CROWN = 0.86;

/** how far a strand hangs clear of the stone it clings to */
export const IVY_CLING = 0.035;
/** a strand on the base course stops this far above the flagstones */
export const IVY_FLOOR_GAP = 0.06;

/** one place a strand can hold on: the ledge height, and the face height it then falls */
export interface IvyLedge {
  y: number;
  drop: number;
}

/**
 * WHERE IVY IS ROOTED. A ruin block is a stepped pyramid — each course is inset from the
 * one under it — so every course top leaves a LEDGE, and a ledge is the only thing a
 * creeper can both hold on to and hang free from. There are two on a standing block:
 *
 *   strand 0 → the top of the second course (under the capstone, which is narrower and
 *              never overhangs it): the strand runs down the standing stone's sunlit face
 *              and its tip lands exactly on the joint below;
 *   strand 1 → the joint where the foundation steps in: the widest stone on the block,
 *              and the strand trails down the base course, stopping above the flagstones.
 *
 * Hanging both at CROWN height was the bug: the crown is the capstone at 0.29 half-width,
 * the strand hung at the foundation's 0.515, and the whole rooted end was in open air.
 */
export function ivyLedge(skin: WallSkin, strand: number): IvyLedge {
  if (strand === 0 && skin.upper > 0) return { y: BASE_COURSE + skin.upper, drop: skin.upper };
  return { y: BASE_COURSE, drop: BASE_COURSE - IVY_FLOOR_GAP };
}

/**
 * The mason. One block, derived from the walls array and the lane seed and nothing else.
 *
 * The moss term is the part that makes a random maze look built rather than sprinkled: a
 * cell hemmed in by neighbours is damp and goes green, a block standing alone in the open
 * is sun-bleached. That is a pure read of `grid.walls` — the same information the debug
 * overlay draws its boxes from — so the ruin's colour tells you about the layout you can
 * already see, and never about the solution you cannot.
 */
export function skinWallCell(grid: GridGeometry, x: number, y: number, seed: number): WallSkin {
  const r = rngAt(x, y, seed ^ 0x5ee1d0a7);
  const standing = r();
  const upper = standing < 0.16 ? 0 : UPPER_MIN + UPPER_SPAN * r();
  const capRoll = r();
  const cap = upper > 0 && capRoll < 0.55 ? CAP_MIN + CAP_SPAN * r() : 0;
  const crown = BASE_COURSE + upper + cap;
  const upperYaw = (r() - 0.5) * 0.24;
  const capYaw = (r() - 0.5) * 0.7;
  const capTilt = (r() - 0.5) * 0.18;
  const damp = wallNeighbours(grid, x, y) / 4;
  const moss = clamp01(0.12 + 0.55 * r() + 0.4 * damp - 0.2);
  const cushion = upper > 0 && r() < 0.34;
  const ivyRoll = r();
  const ivy = crown >= IVY_MIN_CROWN && ivyRoll < 0.5 ? 1 + (ivyRoll < 0.18 ? 1 : 0) : 0;
  const ivyFace = Math.min(3, Math.floor(r() * 4));
  return { upper, cap, crown, upperYaw, capYaw, capTilt, moss, cushion, ivy, ivyFace };
}

/**
 * The finished crown height of a wall cell (0 for a free cell).
 *
 * This is a PROMISE about the stone that is actually drawn, not about the stone that was
 * ordered: the foundation is laid flat at exactly `BASE_COURSE` (see the `jitter: 0` note
 * at `buildCellField`), courses two and three are placed from this same skin, so
 * `crownHeightAt` is the real top of the real block to the last bit.
 */
export function crownHeightAt(grid: GridGeometry, x: number, y: number, seed: number): number {
  if (!isWall(grid, x, y)) return 0;
  return skinWallCell(grid, x, y, seed).crown;
}

/**
 * The block's own silhouette: half the width of whatever stone stands at height `y`, in
 * world units from the cell centre (0 above the crown). Each course is narrower than the
 * one under it, so a ruin block is a stepped pyramid and "how far out is the wall here"
 * depends on how high you ask.
 *
 * Anything that has to CLING to a block asks this instead of assuming the widest course:
 * the ivy hung at the foundation's radius but at CROWN height for two passes, which put
 * every strand's rooted end 0.13–0.23 units out in open air beside the block.
 */
export function blockHalfWidthAt(skin: WallSkin, y: number): number {
  if (y <= BASE_COURSE) return CELL / 2 - BASE_INSET;
  if (y <= BASE_COURSE + skin.upper) return CELL / 2 - UPPER_INSET;
  if (y <= skin.crown) return CELL / 2 - CAP_INSET;
  return 0;
}

// ---------------------------------------------------------------------------
// the ruin's palette (every colour a blend of tokens — STYLE.md)
// ---------------------------------------------------------------------------

export interface RuinSkin {
  /** sunlit limestone — the second and third courses */
  stoneLight: THREE.Color;
  /** the shaded foundation course and the underside of everything */
  stoneDark: THREE.Color;
  /** the moss that climbs the blocks */
  moss: THREE.Color;
  /** the deep green in the joints and under the ivy */
  mossDeep: THREE.Color;
  /** worn flagstones on the free cells */
  flag: THREE.Color;
  /** the damp flagstone of a cell hemmed in by walls */
  flagDamp: THREE.Color;
  /** soil showing under the blocks */
  soil: THREE.Color;
  /** the grass pushing through the joints */
  grass: THREE.Color;
  /** wildflowers reclaiming the ruin */
  bloom: THREE.Color;
  /** the canopy of the forest that has grown up around it */
  canopy: THREE.Color;
  /** warm sunlight — the beam, the pool, the celebration */
  sunWarm: THREE.Color;
  /** distant birds, near enough to black to be a silhouette */
  bird: THREE.Color;
}

/**
 * HONEY LIMESTONE, not slate. The first pass blended the po-maze silver and the dusk tan
 * half and half and shaded it a quarter of the way to black, and the whole ruin came out
 * a cold monochrome grey — eleven units of the same rock, with the moss reading as dirt.
 * Under the noon key the fix is not more light, it is more HUE: the stone is mostly tan
 * with the silver only cooling it, the moss is most of the way to this scene's own accent,
 * and the flagstones are sand rather than pavement. Every value stays inside the noon
 * albedo band (roughly 140–170 sRGB), so the ruin gains colour without gaining exposure.
 */
export function ruinSkin(palette: Palette, tokens: typeof TOKENS): RuinSkin {
  const accent = tokens.accents['procedural-maze'] ?? tokens.celebrationGold;
  const tan = tokens.palettes.dusk.groundA;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  return {
    // the masonry is TAN with the silver only cooling it — a half-and-half blend came out
    // neutral, and a neutral stone under a white key is concrete
    stoneLight: shade(blendTokens(silver, tan, 0.86), 0.06),
    stoneDark: shade(blendTokens(tan, palette.hemiGround, 0.22), 0.4),
    moss: shade(blendTokens(accent, palette.groundA, 0.62), 0.24),
    mossDeep: shade(blendTokens(palette.groundA, accent, 0.12), 0.48),
    // the flagstones must sit CLEARLY under the block crowns in value, or the maze stops
    // reading as a plan: floor and wall landed within 6 sRGB of each other on the pass
    // before this one and the whole ruin went flat
    flag: shade(blendTokens(silver, tan, 0.72), 0.42),
    flagDamp: shade(blendTokens(palette.groundA, tan, 0.4), 0.46),
    soil: shade(blendTokens(palette.hemiGround, tan, 0.55), 0.44),
    grass: shade(blendTokens(palette.groundA, accent, 0.45), 0.12),
    bloom: shade(blendTokens(palette.groundB, tokens.celebrationGold, 0.4), 0.12),
    canopy: shade(blendTokens(palette.groundA, palette.groundB, 0.3), 0.24),
    sunWarm: blendTokens(palette.sun, tokens.celebrationGold, 0.42),
    bird: shade(blendTokens(palette.hemiGround, tan, 0.3), 0.72),
  };
}

/**
 * PER-STONE TONE. The kit's `jitter` used to buy this for free (it multiplies each cell's
 * tint by 1 ± jitter/2), but it pays for it by scaling the block's HEIGHT with the same
 * number — which is exactly the thing this ruin may not have, because the scene stacks its
 * own courses on a foundation it has to KNOW the top of. So the tone is generated here and
 * the kit lays flat stone. Same hashes the kit used, so the ruin's colour is unchanged.
 */
const TONE_JITTER = 0.09;

function toneJitter(x: number, y: number, seed: number): number {
  return 1 + (hash2f(x, y, seed) - 0.5) * TONE_JITTER;
}

/**
 * The flagstone colour of one cell. A cell hemmed in by masonry never dries out, so the
 * damper it is the greener its stone — a pure read of `grid.walls`, the same fact the
 * debug overlay draws its boxes from, and never a word about the solution.
 *
 * Exported because `index.ts` re-tints the goal cell as the sunbeam lands on it and must
 * be able to put back EXACTLY what was there; a second, drifting copy of this rule is how
 * a restored tile ends up a shade off forever. It writes into `out` when one is given,
 * because that restore happens inside `update()`, where a scene may not allocate.
 */
export function floorTintFor(
  grid: GridGeometry,
  skin: RuinSkin,
  x: number,
  y: number,
  seed: number,
  out?: THREE.Color,
): THREE.Color {
  const c = out ?? new THREE.Color();
  if (isWall(grid, x, y)) c.copy(skin.soil);
  else {
    const damp = wallNeighbours(grid, x, y) / 4;
    c.copy(skin.flag).lerp(skin.flagDamp, 0.25 + 0.6 * damp);
  }
  return c.multiplyScalar(toneJitter(x, y, seed));
}

/** The foundation-block colour of one wall cell: moss climbs, so the base is greenest. */
export function wallTintFor(
  grid: GridGeometry,
  skin: RuinSkin,
  x: number,
  y: number,
  seed: number,
  out?: THREE.Color,
): THREE.Color {
  const s = skinWallCell(grid, x, y, seed);
  const c = out ?? new THREE.Color();
  return c
    .copy(skin.stoneDark)
    .lerp(skin.mossDeep, 0.2 + 0.55 * s.moss)
    .multiplyScalar(toneJitter(x, y, seed ^ 0x51ed270b));
}

// ---------------------------------------------------------------------------
// scatter exclusions
// ---------------------------------------------------------------------------

/**
 * Keep a clear disc around every cell CENTER. The grid must stay legible under the
 * overgrowth: tufts and rubble live in the joints between flagstones, never on the square
 * the agent actually occupies.
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

/** Nothing grows on a wall cell (read from the walls array, never inferred). */
export function excludeWallCells(grid: GridGeometry): ExcludeFn {
  const ox = (grid.w - 1) / 2;
  const oz = (grid.h - 1) / 2;
  return (x, z) => isWall(grid, Math.round(x / CELL + ox), Math.round(z / CELL + oz));
}

// ---------------------------------------------------------------------------
// the meadow height field
// ---------------------------------------------------------------------------

/**
 * Rolling forest floor OUTSIDE the ruin, identically flat under it. The weight is exactly
 * 0 everywhere within the footprint, so `makeTerrain`'s flatten mask has nothing to fight
 * and the ground under the flagstones is exactly `RUIN_SINK`.
 */
export function makeForestFloorHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const out = Math.max(Math.abs(x), Math.abs(z)) - RUIN_HALF;
    const w = smoothstep01(out / 5.5);
    if (w <= 0) return 0;
    const n1 = valueNoise2(x * 0.078 + 3.1, z * 0.078 + 9.4, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.2 + 6.7, z * 0.2 + 2.2, 64, seed ^ 0x51ed270b) - 0.5;
    return w * (n1 * 2.6 + n2 * 0.7);
  };
}

// ---------------------------------------------------------------------------
// the courses — instanced masonry, placed strictly from `grid.walls`
// ---------------------------------------------------------------------------

/** one hanging strand: the block it is rooted on, and which strand of that block it is */
export interface IvyStrand {
  cell: GridCell;
  strand: number;
}

export interface CoursesHandle {
  group: THREE.Group;
  /** second course; one instance per wall cell whose `upper > 0` */
  upperMesh: THREE.InstancedMesh;
  /** fallen lintels; one instance per wall cell whose `cap > 0` */
  capMesh: THREE.InstancedMesh;
  /** moss cushions on the crowns */
  cushionMesh: THREE.InstancedMesh;
  /** ivy strands (the one thing here that trembles) */
  ivyMesh: THREE.InstancedMesh | null;
  /** the wall cells that got each course, in placement order */
  upperCells: GridCell[];
  capCells: GridCell[];
  /** one entry per ivy INSTANCE, in placement order: the block it holds on to and which
   *  of that block's strands it is (the face is `(skin.ivyFace + strand * 2) % 4`) */
  ivyCells: IvyStrand[];
  setTime(t: number): void;
  dispose(): void;
}

/**
 * Courses two and three, plus the green that grows on them. Everything is instanced, so
 * the entire upper ruin — however many blocks this seed produced — costs four draws.
 *
 * `skip` is how the bespoke props claim their cells: the four corner towers hand in their
 * own cells and no generic course is laid there (the walls array is untouched, so the
 * overlay still draws its box and every accuracy check still passes).
 */
export function buildCourses(
  grid: GridGeometry,
  seed: number,
  skin: RuinSkin,
  palette: Palette,
  tier: QualityTier,
  skip: (x: number, y: number) => boolean,
): CoursesHandle {
  const group = new THREE.Group();
  group.name = 'ruinCourses';
  const detail = tier === 'thumb' ? 0.35 : tier === 'medium' ? 0.7 : 1;
  const shadows = tier !== 'thumb';

  const upperCells: GridCell[] = [];
  const capCells: GridCell[] = [];
  const cushionCells: GridCell[] = [];
  const ivyCells: IvyStrand[] = [];
  const skins = new Map<number, WallSkin>();

  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (!isWall(grid, x, y) || skip(x, y)) continue;
      const s = skinWallCell(grid, x, y, seed);
      skins.set(y * grid.w + x, s);
      if (s.upper > 0) upperCells.push({ x, y });
      if (s.cap > 0) capCells.push({ x, y });
      if (s.cushion) cushionCells.push({ x, y });
      for (let i = 0; i < s.ivy; i++) ivyCells.push({ cell: { x, y }, strand: i });
    }
  }

  const skinAt = (x: number, y: number): WallSkin =>
    skins.get(y * grid.w + x) ?? skinWallCell(grid, x, y, seed);

  // ---- materials ----
  // WHITE BASE, ABSOLUTE INSTANCE TINT (the four-rooms cell-field lesson, and the bug that
  // cost this scene its first two passes): `setColorAt` MULTIPLIES the material colour, so
  // a material tinted `stoneLight` carrying instance tints that are themselves built out
  // of `stoneLight` renders the stone SQUARED — the whole ruin came out a dark olive mass
  // with the moss reading as grime. The base is the token white and the instance colour is
  // the real hue, which also buys arbitrary per-block colour inside one draw.
  const white = palette.sun;
  const stoneMaterial = oakMaterial({
    color: white,
    roughness: 0.94,
    flatShading: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.22,
    toonSteps: 3,
    toonSoftness: 0.28,
    // a noon ruin's shaded faces are BOUNCE-LIT by the flagstones around them; at the
    // kit's default floor of 0 every north face went to mud and the maze lost its plan
    toonFloor: 0.42,
    tier,
    name: 'ruinCourseStone',
  });
  const mossMaterial = oakMaterial({
    color: white,
    roughness: 0.98,
    flatShading: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.44,
    tier,
    name: 'ruinMossCushion',
  });
  // per-instance wind needs an InstancedMesh: oakMaterial takes its wind phase from the
  // instance origin, so merged strands would tremble in lockstep (canon lesson 10)
  const ivyMaterial = oakMaterial({
    color: white,
    roughness: 0.95,
    flatShading: true,
    side: THREE.DoubleSide,
    vertexColors: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.36,
    toonSteps: 2,
    toonSoftness: 0.36,
    // a creeper in a joint is lit by the flagstones bouncing under it, not by the sun it
    // never sees: at the courses' own 0.42 floor the strands went to shadow with the face
    toonFloor: 0.58,
    // the ivy tremble: pinned at local y = 0 (the joint it is rooted in) so a strand
    // swings from where it holds on, and the wind phase is per-instance
    wind: { amplitude: 0.045, frequency: 1.35, height: 0.62 },
    tier,
    name: 'ruinIvy',
  });

  // ---- geometry ----
  const upperGeometry = new THREE.BoxGeometry(CELL - UPPER_INSET * 2, 1, CELL - UPPER_INSET * 2);
  upperGeometry.translate(0, 0.5, 0);
  upperGeometry.name = 'ruinUpperCourse';
  const capGeometry = brokenSlabGeometry(detail);
  const cushionGeometry = new THREE.IcosahedronGeometry(0.34, tier === 'thumb' ? 0 : 1);
  cushionGeometry.scale(1, 0.38, 1);
  cushionGeometry.name = 'ruinMossCushion';
  const ivyGeometry = ivyStrandGeometry(detail);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);
  const color = new THREE.Color();
  const world: Vec3Tuple = [0, 0, 0];

  const instanced = (
    geometry: THREE.BufferGeometry,
    material: OakMaterial,
    count: number,
    name: string,
    cast: boolean,
  ): THREE.InstancedMesh => {
    const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
    mesh.name = name;
    mesh.count = count;
    mesh.castShadow = cast && shadows;
    mesh.receiveShadow = shadows;
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3);
    return mesh;
  };

  const upperMesh = instanced(upperGeometry, stoneMaterial, upperCells.length, 'ruinUpperCourse', true);
  const capMesh = instanced(capGeometry, stoneMaterial, capCells.length, 'ruinCapstones', true);
  const cushionMesh = instanced(cushionGeometry, mossMaterial, cushionCells.length, 'ruinMossCushions', false);
  const ivyMesh =
    tier === 'thumb' || ivyCells.length === 0
      ? null
      : instanced(ivyGeometry, ivyMaterial, ivyCells.length, 'ruinIvy', false);

  // ---- second course ----
  for (let i = 0; i < upperCells.length; i++) {
    const c = upperCells[i] as GridCell;
    const s = skinAt(c.x, c.y);
    cellToWorld(c.x, c.y, grid, world);
    position.set(world[0], BASE_COURSE, world[2]);
    euler.set(0, s.upperYaw, 0);
    quaternion.setFromEuler(euler);
    scale.set(1, s.upper, 1);
    matrix.compose(position, quaternion, scale);
    upperMesh.setMatrixAt(i, matrix);
    // moss climbs from the ground, so the SECOND course is greener than the crown
    // ±14% of value per block, not ±6%: at the composed standoff a course of identical
    // instances reads as one extruded mass instead of as stones somebody stacked
    color.copy(skin.stoneLight).lerp(skin.moss, s.moss * 0.3).multiplyScalar(0.86 + 0.28 * hash2f(c.x, c.y, seed));
    upperMesh.setColorAt(i, color);
  }

  // ---- fallen lintels ----
  for (let i = 0; i < capCells.length; i++) {
    const c = capCells[i] as GridCell;
    const s = skinAt(c.x, c.y);
    cellToWorld(c.x, c.y, grid, world);
    position.set(world[0], BASE_COURSE + s.upper, world[2]);
    euler.set(s.capTilt, s.capYaw, 0);
    quaternion.setFromEuler(euler);
    scale.set(1, s.cap, 1);
    matrix.compose(position, quaternion, scale);
    capMesh.setMatrixAt(i, matrix);
    color.copy(skin.stoneLight).lerp(skin.moss, s.moss * 0.16).multiplyScalar(0.98 + 0.22 * hash2f(c.y, c.x, seed));
    capMesh.setColorAt(i, color);
  }

  // ---- moss cushions on the crowns ----
  for (let i = 0; i < cushionCells.length; i++) {
    const c = cushionCells[i] as GridCell;
    const s = skinAt(c.x, c.y);
    cellToWorld(c.x, c.y, grid, world);
    const r = hash2f(c.x * 7 + 1, c.y * 13 + 5, seed);
    position.set(world[0] + (r - 0.5) * 0.22, s.crown - 0.04, world[2] + (hash2f(c.y, c.x, seed ^ 0x9e11) - 0.5) * 0.22);
    euler.set(0, r * 6.283, 0);
    quaternion.setFromEuler(euler);
    const k = 0.5 + 0.44 * s.moss;
    scale.set(k, 0.66 + 0.5 * r, k);
    matrix.compose(position, quaternion, scale);
    cushionMesh.setMatrixAt(i, matrix);
    color.copy(skin.moss).lerp(skin.mossDeep, 0.2 + 0.5 * r);
    cushionMesh.setColorAt(i, color);
  }

  // ---- ivy: one strand per instance, hanging off a named face ----
  if (ivyMesh) {
    for (let i = 0; i < ivyCells.length; i++) {
      const entry = ivyCells[i] as IvyStrand;
      const c = entry.cell;
      const s = skinAt(c.x, c.y);
      cellToWorld(c.x, c.y, grid, world);
      const face = (s.ivyFace + entry.strand * 2) % 4;
      const a = (face * Math.PI) / 2;
      // ROOTED ON A LEDGE THAT EXISTS. Two passes got this wrong in opposite directions:
      // first at `0.5 - UPPER_INSET` (0.44), which buried every strand inside the
      // 0.48-wide foundation, then at the foundation's own radius but at CROWN height,
      // where the block has already stepped in twice — so the rooted end hung 0.13 (upper
      // course) to 0.23 (capstone) units out in open air. A ruin block is a stepped
      // pyramid, and every step is a ledge a creeper can hold: `ivyLedge` picks one, the
      // block's own `blockHalfWidthAt` says how far out its face is, and the strand falls
      // exactly the height of that face — so both ends touch stone.
      const ledge = ivyLedge(s, entry.strand);
      const off = blockHalfWidthAt(s, ledge.y) + IVY_CLING;
      const jitter = hash2f(c.x * 3 + entry.strand, c.y * 5, seed ^ 0x1de2) - 0.5;
      position.set(
        world[0] + Math.sin(a) * off + Math.cos(a) * jitter * 0.5,
        ledge.y,
        world[2] + Math.cos(a) * off - Math.sin(a) * jitter * 0.5,
      );
      euler.set(0, a, 0);
      quaternion.setFromEuler(euler);
      scale.set(0.85 + 0.4 * (jitter + 0.5), ledge.drop, 0.85 + 0.4 * (jitter + 0.5));
      matrix.compose(position, quaternion, scale);
      ivyMesh.setMatrixAt(i, matrix);
      // LIGHTER THAN THE MOSS IT GREW FROM. A strand used to hang in the open joint where
      // the sky lit it; rooted against the block it lies on the SHADED face all day, and
      // at the old mossDeep-led blend it disappeared into the shadow it sits in. Leading
      // with `moss` (and letting mossDeep only shade it) puts it back on the block.
      color.copy(skin.moss).lerp(skin.mossDeep, 0.12 + 0.34 * (jitter + 0.5));
      ivyMesh.setColorAt(i, color);
    }
  }

  const meshes: THREE.InstancedMesh[] = [upperMesh, capMesh, cushionMesh];
  if (ivyMesh) meshes.push(ivyMesh);
  for (const m of meshes) {
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    m.computeBoundingSphere();
    if (m.count > 0) group.add(m);
  }

  return {
    group,
    upperMesh,
    capMesh,
    cushionMesh,
    ivyMesh,
    upperCells,
    capCells,
    ivyCells,
    setTime(t) {
      oakUniforms(ivyMaterial)?.setTime(t);
      oakUniforms(stoneMaterial)?.setTime(t);
      oakUniforms(mossMaterial)?.setTime(t);
    },
    dispose() {
      upperMesh.dispose();
      capMesh.dispose();
      cushionMesh.dispose();
      ivyMesh?.dispose();
      upperGeometry.dispose();
      capGeometry.dispose();
      cushionGeometry.dispose();
      ivyGeometry.dispose();
      stoneMaterial.dispose();
      mossMaterial.dispose();
      ivyMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/** A capstone: a slab with one corner knocked off, so it never reads as a neat box. */
function brokenSlabGeometry(detail: number): THREE.BufferGeometry {
  const size = CELL - CAP_INSET * 2;
  const g = new THREE.BoxGeometry(size, 1, size, 1, 1, 1);
  g.translate(0, 0.5, 0);
  const pos = g.getAttribute('position');
  // Knock the +x/+z top corner in: three vertices, a whole silhouette. The first pass
  // pulled it to 0.42 and dropped the corner to two thirds height, which turned every
  // capstone into a big sloped face — from the composed aerial the ruin read as a village
  // of pitched ROOFS. A slab is chipped, not pitched.
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    if (y > 0.5 && x > 0 && z > 0) {
      pos.setX(i, x * 0.72);
      pos.setZ(i, z * 0.72);
      pos.setY(i, y * (detail > 0.5 ? 0.88 : 0.94));
    }
  }
  pos.needsUpdate = true;
  g.computeVertexNormals();
  g.name = 'ruinCapstone';
  return g;
}

/**
 * One ivy strand: a narrow ribbon of leaves hanging DOWN from local y = 0 (the course
 * joint it is rooted in), built in unit length so the instance scale sets how far it
 * falls. The vertex grade darkens the tip, which stops a flat ribbon reading as a plank.
 */
function ivyStrandGeometry(detail: number): THREE.BufferGeometry {
  const leaves = Math.max(3, Math.round(6 * detail));
  const positions = new Float32Array(leaves * 3 * 3);
  const colors = new Float32Array(leaves * 3 * 3);
  for (let i = 0; i < leaves; i++) {
    const t0 = i / leaves;
    const t1 = (i + 1) / leaves;
    const side = i % 2 === 0 ? 1 : -1;
    // a leaf is ~19 px across at the composed standoff: narrower than this and a strand
    // on a shaded face is a scratch rather than a plant
    const w = 0.2 * (1 - 0.4 * t0);
    const o = i * 9;
    // a leaf is a triangle: two points on the stem, one flicked out to the side
    positions[o] = 0;
    positions[o + 1] = -t0;
    positions[o + 2] = 0;
    positions[o + 3] = 0;
    positions[o + 4] = -t1;
    positions[o + 5] = 0;
    positions[o + 6] = side * w * 1.7;
    positions[o + 7] = -(t0 + t1) * 0.5 + 0.02;
    positions[o + 8] = w * 0.6;
    for (let v = 0; v < 3; v++) {
      const k = v === 2 ? 1.15 - 0.35 * t0 : 0.72 - 0.28 * t0;
      colors[o + v * 3] = k;
      colors[o + v * 3 + 1] = k;
      colors[o + v * 3 + 2] = k;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'ruinIvyStrand';
  return g;
}

// ---------------------------------------------------------------------------
// bespoke props
// ---------------------------------------------------------------------------

/**
 * How much of a corner tower the seed left standing. Exported because the FRAME is sized
 * against it: the towers are the tallest thing on the playfield's own corners, so they —
 * not the flagstones — are what a camera distance has to clear.
 */
export const TOWER_HEIGHT_MIN = 1.12;
export const TOWER_HEIGHT_SPAN = 0.62;
export const TOWER_HEIGHT_MAX = TOWER_HEIGHT_MIN + TOWER_HEIGHT_SPAN;

/**
 * A broken corner tower. Three drums, the top one shattered and leaning, on a plinth that
 * fills the cell footprint so the tower still READS as impassable masonry. Built in local
 * space around the cell centre; `height` is the seed's own verdict on how much of this
 * tower is left.
 */
export function towerGeometry(skin: RuinSkin, tier: QualityTier, height: number, lean: number): THREE.BufferGeometry {
  const detail = tier === 'thumb' ? 0.45 : tier === 'medium' ? 0.7 : 1;
  const radial = Math.max(6, Math.round(14 * detail));
  const parts: ColoredPart[] = [];

  const plinth = new THREE.BoxGeometry(CELL - BASE_INSET * 2, 0.26, CELL - BASE_INSET * 2);
  plinth.translate(0, 0.13, 0);
  parts.push({ geometry: plinth, color: skin.stoneDark });

  const drumH = (height - 0.26) / 3;
  for (let i = 0; i < 3; i++) {
    const r0 = 0.44 - i * 0.045;
    const r1 = 0.47 - i * 0.045;
    const drum = new THREE.CylinderGeometry(r0, r1, drumH, radial, 1);
    drum.translate(0, drumH * 0.5, 0);
    if (i === 2) {
      // the top drum is a stump: shorter, leaning, and sheared on one side
      drum.scale(1, 0.62, 1);
      drum.rotateZ(lean);
    }
    drum.translate(0, 0.26 + i * drumH, 0);
    parts.push({ geometry: drum, color: i === 1 ? skin.stoneLight : skin.stoneDark });
  }

  // a course band, so the tower reads as built rather than turned
  const band = new THREE.CylinderGeometry(0.5, 0.5, 0.075, radial, 1);
  band.translate(0, 0.26 + drumH, 0);
  parts.push({ geometry: band, color: skin.stoneLight });

  return mergeColoredParts(parts, 'ruinTower');
}

/**
 * A ruined doorway: two jambs at the cell's edges and a stepped lintel over them, with the
 * keystone displaced. Built with its OPENING along ±x, so a doorway in a wall that runs
 * north–south is the same geometry yawed 90° — the orientation is read from the walls, in
 * `archCells()`, and applied here.
 */
export function archGeometry(skin: RuinSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = tier === 'thumb' ? 0.45 : tier === 'medium' ? 0.7 : 1;
  const parts: ColoredPart[] = [];
  const JAMB_H = 0.7;
  const HALF = 0.47;

  for (const sz of [-1, 1]) {
    const jamb = new THREE.BoxGeometry(0.3, JAMB_H, 0.16);
    jamb.translate(0, JAMB_H * 0.5, sz * HALF);
    parts.push({ geometry: jamb, color: skin.stoneDark });
    const corbel = new THREE.BoxGeometry(0.34, 0.1, 0.24);
    corbel.translate(0, JAMB_H + 0.05, sz * (HALF - 0.02));
    parts.push({ geometry: corbel, color: skin.stoneLight });
  }

  // the lintel, in three voussoirs with the keystone shoved out of true
  const spans: readonly (readonly [number, number, number])[] = [
    [-0.3, 0.855, 0],
    [0.0, 0.9, 0.16],
    [0.3, 0.855, 0],
  ];
  for (let i = 0; i < spans.length; i++) {
    const s = spans[i] as readonly [number, number, number];
    const voussoir = new THREE.BoxGeometry(0.2, 0.15, 0.34);
    voussoir.rotateX(s[2]);
    voussoir.translate(0, s[1], s[0]);
    parts.push({ geometry: voussoir, color: i === 1 ? skin.stoneLight : skin.stoneDark });
  }

  // moss on the crown, because everything here is going back to the forest
  const cushion = new THREE.IcosahedronGeometry(0.16, tier === 'thumb' ? 0 : 1);
  cushion.scale(1.3, 0.4, 1.1);
  cushion.translate(0.02, 0.985, -0.14);
  parts.push({ geometry: cushion, color: skin.moss });

  if (detail > 0.5) {
    const fallen = new THREE.BoxGeometry(0.22, 0.13, 0.3);
    fallen.rotateY(0.5);
    fallen.rotateZ(0.24);
    fallen.translate(0.3, 0.07, -0.2);
    parts.push({ geometry: fallen, color: skin.stoneDark });
  }

  return mergeColoredParts(parts, 'ruinArch');
}

/**
 * Collapse one cell-field wall instance so a bespoke prop can own that cell. The walls
 * array is NOT touched: the cell is still a wall to the env, to the overlay and to every
 * accuracy check — only the generic foundation block that stood there is gone.
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

// ---------------------------------------------------------------------------
// distant birds
// ---------------------------------------------------------------------------

const BIRD_COUNT = 7;

export interface BirdsHandle {
  mesh: THREE.InstancedMesh;
  /** ctx.time() — the whole flock is a pure function of it */
  setTime(t: number): void;
  dispose(): void;
}

/**
 * Seven chevrons on lazy circles high over the forest. Position, bank and wingbeat are
 * pure functions of `ctx.time()`, so a paused frame holds the flock mid-glide and a
 * scrubbed one puts it exactly where a played one would.
 */
export function makeBirds(seed: number, color: THREE.ColorRepresentation, tier: QualityTier): BirdsHandle {
  const count = tier === 'thumb' ? 3 : BIRD_COUNT;
  const rng = streamFor(seed, 'ruin-birds');
  const geometry = birdGeometry();
  const material = oakMaterial({
    color,
    roughness: 1,
    flatShading: true,
    side: THREE.DoubleSide,
    toonSteps: 2,
    toonFloor: 0.5,
    tier,
    name: 'ruinBirds',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.name = 'ruinBirds';
  mesh.frustumCulled = false;
  mesh.castShadow = false;
  mesh.receiveShadow = false;

  const radii = new Float32Array(count);
  const heights = new Float32Array(count);
  const speeds = new Float32Array(count);
  const phases = new Float32Array(count);
  const sizes = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    radii[i] = 15 + rng() * 11;
    heights[i] = 8.5 + rng() * 5.5;
    speeds[i] = (0.055 + rng() * 0.05) * (rng() < 0.5 ? -1 : 1);
    phases[i] = rng() * Math.PI * 2;
    sizes[i] = 0.5 + rng() * 0.32;
  }

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);

  const setTime = (t: number): void => {
    for (let i = 0; i < count; i++) {
      const a = t * (speeds[i] ?? 0) + (phases[i] ?? 0);
      const r = radii[i] ?? 0;
      const s = sizes[i] ?? 1;
      position.set(Math.cos(a) * r, (heights[i] ?? 0) + Math.sin(t * 0.4 + a) * 0.5, Math.sin(a) * r);
      // Fly along the tangent, banking into the turn, wings beating slowly. The
      // parenthesis is load-bearing: `speeds[i] ?? 0 > 0` parses as `speeds[i] ?? (0 > 0)`,
      // which is truthy for EVERY non-zero speed, so the half of the flock that circles
      // the other way flew tail-first (measured: heading·travel = −1.000 on 2 of 7).
      euler.set(0, -a + ((speeds[i] ?? 0) > 0 ? 0 : Math.PI), Math.sin(t * 1.6 + a * 3) * 0.3);
      quaternion.setFromEuler(euler);
      const flap = 0.7 + 0.4 * Math.abs(Math.sin(t * 3.1 + (phases[i] ?? 0)));
      scale.set(s, s * flap, s);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  };
  setTime(0);

  return {
    mesh,
    setTime,
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}

/** Two swept triangles — at 20 units away a bird is a silhouette and nothing else. */
function birdGeometry(): THREE.BufferGeometry {
  const v: number[] = [];
  for (const side of [-1, 1]) {
    v.push(0, 0, 0.09, side * 0.34, 0.09, -0.04, side * 0.12, 0.0, -0.11);
  }
  const positions = new Float32Array(v);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'ruinBird';
  return g;
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
  courses: CoursesHandle;
  /** probe objects: the four corner towers, in `cornerCells` order. POSITION ONLY. */
  towerRoots: THREE.Object3D[];
  /** probe objects: the ruined doorways actually built. POSITION ONLY. */
  archRoots: THREE.Object3D[];
  /** the doorway cells those roots stand on, same order */
  arches: ArchCell[];
  /** move the shaft's dust column onto the live goal cell */
  setGoalPosition(x: number, y: number, z: number): void;
  /** ctx.time() — one clock for ivy, grass, dust, birds and the forest */
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
  const palette: Palette = ctx.tokens.palettes.noon;
  const skin = ruinSkin(palette, ctx.tokens);
  const shadows = tier !== 'thumb';
  const seed = ctx.seed;
  const group = new THREE.Group();
  group.name = 'overgrownRuins';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];

  // ---- sky (exactly one) ----
  const sky = skyFromPalette(palette, 'noon', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.032,
    sunHdr: 5.2,
    sunGlow: 0.24,
    sunGlowPower: 30,
    topSpread: 0.7,
    bottomSpread: 0.52,
    radius: 340,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the one light budget: sun + hemi + 3 points (shaft / wisp / celebration) ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 3,
    pointDistance: 7,
    sunDistanceScale: 2.2,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: 0.4, z: 0 }, SHADOW_RADIUS);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the forest floor the ruin sits in ----
  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    tier,
    seed,
    heightFn: makeForestFloorHeight(seed),
    jitter: 0,
    colorNoise: 0.06,
    slopeColor: skin.soil,
    slopeStrength: 0.55,
    slopeSharpness: 2.1,
    flattenMask: { x: 0, z: 0, halfX: RUIN_HALF, halfZ: RUIN_HALF, feather: 2.4 },
    flattenHeight: RUIN_SINK,
    colorRamp: [
      { t: 0, color: blendTokens(skin.soil, palette.groundA, 0.4) },
      { t: 0.45, color: shade(palette.groundA, 0.14) },
      { t: 1, color: shade(blendTokens(palette.groundA, palette.groundB, 0.42), 0.1) },
    ],
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.16,
      toonSteps: 3,
      toonSoftness: 0.3,
      toonFloor: 0.24,
    },
    receiveShadow: shadows,
    castShadow: false,
    name: 'ruinForestFloor',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  // ---- the foundation course + flagstones: ONE floor draw, ONE wall draw ----
  // White base + full-colour tint: the instance colour MULTIPLIES the material colour, so
  // this is how a cell field gets arbitrary per-cell hues inside a single draw.
  const white = palette.sun;
  const field = buildCellField(
    grid,
    {
      floorColor: white,
      wallColor: white,
      floorUnderWalls: true,
      floorTint: (x, y) => floorTintFor(grid, skin, x, y, seed),
      wallTint: (x, y) => wallTintFor(grid, skin, x, y, seed),
      wallShape: 'block',
      // the ruin's height variance is the SCENE's (courses 2 and 3), so the foundation is
      // laid flat: a known base is what lets moss, ivy and towers land on the stone
      wallHeight: BASE_COURSE,
      wallInset: BASE_INSET,
      floorInset: 0.055,
      floorThickness: 0.14,
      // JITTER 0, AND IT MATTERS. `CellFieldStyle.jitter` scales each wall block's y as
      // well as its tone (kit/grid: `scale.y = 1 + (nw - 0.5) * jitter * 2`), so 0.09 made
      // the "known base" 0.5 ± 0.045 and half the second courses in the ruin floated over
      // a foundation that had shrunk under them. The tone variation it also bought now
      // comes from `wallTintFor` / `floorTintFor`, which cannot move stone.
      jitter: 0,
      rimColor: palette.skyHorizon,
      rimStrength: 0.2,
      seed,
      name: 'ruinCells',
    },
    tier,
  );
  // `CellFieldStyle` exposes no toon controls, so the foundation course is lifted off the
  // kit's default floor of 0 through the public uniform handle. It matters more here than
  // in any other grid scene: 58% of this playfield is a north-facing block face, and at
  // floor 0 every one of them renders as mud and the maze stops reading as a plan.
  const wallUniforms = oakUniforms(field.wallMaterial);
  if (wallUniforms) wallUniforms.uniforms.uToonFloor.value = 0.42;
  const floorUniforms = oakUniforms(field.floorMaterial);
  if (floorUniforms) floorUniforms.uniforms.uToonFloor.value = 0.4;
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  // ---- the four corner towers claim their cells before the courses are laid ----
  const corners = cornerCells(grid);
  const cornerKeys = new Set<number>();
  for (const c of corners) cornerKeys.add(c.y * grid.w + c.x);

  const courses = buildCourses(grid, seed, skin, palette, tier, (x, y) => cornerKeys.has(y * grid.w + x));
  group.add(courses.group);
  owned.push(courses);
  timed.push(courses);

  const towerRoots: THREE.Object3D[] = [];
  {
    const world: Vec3Tuple = [0, 0, 0];
    for (let i = 0; i < corners.length; i++) {
      const c = corners[i] as GridCell;
      collapseWallInstance(field, c.x, c.y);
      const root = new THREE.Object3D();
      root.name = `ruinTowerRoot${i}`;
      cellToWorld(c.x, c.y, grid, world);
      root.position.set(world[0], 0, world[2]);
      const r = rngAt(c.x, c.y, seed ^ 0x70e2)();
      const r2 = rngAt(c.y, c.x, seed ^ 0x70e2)();
      const geometry = towerGeometry(skin, tier, TOWER_HEIGHT_MIN + TOWER_HEIGHT_SPAN * r, (r2 - 0.5) * 0.32);
      const material = oakMaterial({
        vertexColors: true,
        flatShading: true,
        roughness: 0.95,
        rimColor: palette.skyHorizon,
        rimStrength: 0.3,
        toonSteps: 3,
        toonSoftness: 0.28,
        toonFloor: 0.42,
        tier,
        name: `ruinTowerMaterial${i}`,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `ruinTower${i}`;
      mesh.rotation.y = r * Math.PI * 0.5;
      mesh.castShadow = shadows;
      mesh.receiveShadow = shadows;
      root.add(mesh);
      group.add(root);
      towerRoots.push(root);
      owned.push({
        dispose: () => {
          geometry.dispose();
          material.dispose();
        },
      });
    }
  }

  // ---- ruined doorways on the cells the walls themselves nominate ----
  const allArches = archCells(grid);
  const arches = allArches.slice(0, archBudget(tier));
  const archRoots: THREE.Object3D[] = [];
  if (arches.length > 0) {
    const archGeo = archGeometry(skin, tier);
    const archMaterial = oakMaterial({
      vertexColors: true,
      flatShading: true,
      roughness: 0.94,
      rimColor: palette.skyHorizon,
      rimStrength: 0.32,
      toonSteps: 3,
      toonSoftness: 0.28,
      toonFloor: 0.42,
      tier,
      name: 'ruinArchMaterial',
    });
    owned.push({
      dispose: () => {
        archGeo.dispose();
        archMaterial.dispose();
      },
    });
    const world: Vec3Tuple = [0, 0, 0];
    for (let i = 0; i < arches.length; i++) {
      const a = arches[i] as ArchCell;
      const root = new THREE.Object3D();
      root.name = `ruinArchRoot${i}`;
      cellToWorld(a.cell.x, a.cell.y, grid, world);
      root.position.set(world[0], 0, world[2]);
      const mesh = new THREE.Mesh(archGeo, archMaterial);
      mesh.name = `ruinArch${i}`;
      // the geometry's opening runs along ±x. A doorway whose flanking walls are north and
      // south is walked through along x, so it stands as built; the other case is yawed.
      mesh.rotation.y = a.eastWest ? 0 : Math.PI / 2;
      mesh.castShadow = shadows;
      mesh.receiveShadow = shadows;
      root.add(mesh);
      group.add(root);
      archRoots.push(root);
    }
  }

  // ---- what the forest has done to the floor ----
  const clearCenters = excludeCellCenters(grid, 0.3);
  const offWalls = excludeAny(excludeWallCells(grid), clearCenters);

  const tufts = scatterGrass({
    count: 620,
    area: { x: 0, z: 0, halfX: RUIN_HALF - 0.4, halfZ: RUIN_HALF - 0.4 },
    seed: seed ^ 0x6a11,
    tier,
    exclude: offWalls,
    color: skin.grass,
    scale: [0.26, 0.55],
    stretch: [0.75, 1.5],
    hueJitter: 0.26,
    wind: 0.05,
    windFrequency: 1.6,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    name: 'ruinGrassTufts',
  });
  group.add(tufts.group);
  owned.push(tufts);
  timed.push(tufts);

  const wildflowers = scatterFlowers({
    count: 130,
    area: { x: 0, z: 0, halfX: RUIN_HALF - 0.5, halfZ: RUIN_HALF - 0.5 },
    seed: seed ^ 0x2f83,
    tier,
    exclude: offWalls,
    color: skin.bloom,
    scale: [0.3, 0.6],
    stretch: [0.8, 1.2],
    hueJitter: 0.26,
    wind: 0.035,
    windFrequency: 1.4,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    name: 'ruinWildflowers',
  });
  group.add(wildflowers.group);
  owned.push(wildflowers);
  timed.push(wildflowers);

  // rubble spills off the blocks and gathers in the joints: the rock archetype squashed
  // flat is broken masonry, and it is the one scatter in the kit that never sways
  const rubble = scatterRocks({
    count: 340,
    area: { x: 0, z: 0, halfX: RUIN_HALF - 0.2, halfZ: RUIN_HALF - 0.2 },
    seed: seed ^ 0x51c7,
    tier,
    exclude: clearCenters,
    color: skin.stoneDark,
    scale: [0.1, 0.26],
    stretch: [0.22, 0.5],
    hueJitter: 0.3,
    rimColor: palette.skyHorizon,
    rimStrength: 0.24,
    castShadow: false,
    name: 'ruinRubble',
  });
  group.add(rubble.group);
  owned.push(rubble);
  timed.push(rubble);

  // ---- the forest that is taking the ruin back ----
  const offRuin = excludeRect(0, 0, RUIN_HALF, RUIN_HALF, 1.4);
  const nearCamera = excludeRect(
    Math.sin(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    Math.cos(CAMERA_AZIMUTH) * CAMERA_STANDOFF,
    4.6,
    4.6,
    0,
  );
  const forest = scatterTrees({
    archetypes: ['broadleaf', 'conifer'],
    count: 74,
    area: { x: 0, z: 0, halfX: 18, halfZ: 18, innerRadius: RUIN_HALF + 1.6 },
    seed: seed ^ 0x1a70,
    tier,
    exclude: excludeAny(offRuin, nearCamera),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.72,
    color: skin.canopy,
    scale: [0.85, 1.55],
    stretch: [0.95, 1.5],
    hueJitter: 0.26,
    wind: 0.03,
    windFrequency: 0.8,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    castShadow: true,
    name: 'ruinForest',
  });
  group.add(forest.group);
  owned.push(forest);
  timed.push(forest);

  // saplings pushing right up against the outside of the walls
  const saplings = scatterTrees({
    archetypes: ['broadleaf'],
    count: 26,
    area: { x: 0, z: 0, halfX: RUIN_HALF + 2.2, halfZ: RUIN_HALF + 2.2, innerRadius: RUIN_HALF + 0.35 },
    seed: seed ^ 0x3c19,
    tier,
    exclude: excludeAny(offRuin, nearCamera),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 1,
    color: skin.mossDeep,
    scale: [0.38, 0.66],
    stretch: [0.9, 1.3],
    hueJitter: 0.3,
    wind: 0.05,
    windFrequency: 1.05,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    castShadow: false,
    name: 'ruinSaplings',
  });
  group.add(saplings.group);
  owned.push(saplings);
  timed.push(saplings);

  // ---- idle life ----
  // Kit particle presets size their motes for a ~20-unit field; this ruin is 11 across
  // with the camera ~19.8 out, so `gl_PointSize`'s 60/distance term wants sizes near 1.
  const motes: AmbientParticlesHandle[] = [];
  const haze = makeAmbientParticles({
    preset: 'dust',
    count: 420,
    bounds: { x: 0, y: 1.15, z: 0, halfX: RUIN_HALF, halfY: 1.05, halfZ: RUIN_HALF },
    seed: seed ^ 0x7d21,
    tier,
    color: blendTokens(skin.sunWarm, palette.hemiSky, 0.3),
    size: 0.9,
    intensity: 1.05,
    opacity: 0.4,
    rise: 0.03,
    swirl: 0.2,
    name: 'ruinDustHaze',
  });
  group.add(haze.points);
  owned.push(haze);
  timed.push(haze);
  motes.push(haze);

  // the shaft's own column of dust: a tight field the scene walks onto the live goal cell
  const shaftDust = makeAmbientParticles({
    preset: 'dust',
    count: 150,
    bounds: { x: 0, y: 1.5, z: 0, halfX: 0.6, halfY: 1.45, halfZ: 0.6 },
    seed: seed ^ 0x2ba9,
    tier,
    color: skin.sunWarm,
    // 260 motes at size 1.5 and intensity 1.5 stacked additively into a white PLUME over
    // the atrium — the dust is meant to make the air visible, not to be a second beam
    size: 1.05,
    intensity: 0.9,
    opacity: 0.5,
    rise: 0.05,
    swirl: 0.1,
    name: 'ruinShaftDust',
  });
  group.add(shaftDust.points);
  owned.push(shaftDust);
  timed.push(shaftDust);
  motes.push(shaftDust);

  const pollen = makeAmbientParticles({
    preset: 'pollen',
    count: 300,
    bounds: { x: 0, y: 2.2, z: 0, halfX: 15, halfY: 2, halfZ: 15 },
    seed: seed ^ 0x11c4,
    tier,
    color: blendTokens(palette.groundB, palette.sun, 0.35),
    size: 0.75,
    intensity: 1.35,
    opacity: 0.4,
    rise: 0.07,
    swirl: 0.26,
    name: 'ruinPollen',
  });
  group.add(pollen.points);
  owned.push(pollen);
  timed.push(pollen);
  motes.push(pollen);

  const birds = makeBirds(seed, skin.bird, tier);
  group.add(birds.mesh);
  owned.push(birds);
  timed.push(birds);

  const shaftDustPoints = shaftDust.points;

  return {
    group,
    sky,
    rig,
    terrain,
    field,
    courses,
    towerRoots,
    archRoots,
    arches,
    setGoalPosition(x, y, z) {
      shaftDustPoints.position.set(x, y, z);
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
      towerRoots.length = 0;
      archRoots.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}
