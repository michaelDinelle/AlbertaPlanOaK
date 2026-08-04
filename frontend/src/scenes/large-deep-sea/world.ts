// The Abyss — the static world (LargeDeepSea-cont-v0).
//
// THE ONE IDEA: **the row index IS the light**. LargeDeepSea's state is (r, c) with
// r the depth and c the column, and the dynamics allow exactly one thing — c ≤ r. So the
// reachable set is a triangle, that triangle IS the terraced shelf, and every shelf cell's
// albedo is a pure function of its ROW: ocean-blue at r = 0, violet at the middle shelves,
// near-black on the trench floor at r = 15. Look at any frame for a quarter of a second
// and you know how deep the agent is, because the colour under it says so.
//
// The pair with DeepSea ("Sunlit Descent") is deliberate and structural, not stylistic:
// the same axis language (r = depth = +z, c = across = +x), the same reachable triangle,
// the same treasure-in-the-far-corner, the same resurface. This is that dive continued
// past the point where the sunlight gives out.
//
// NOTHING HERE RE-DERIVES THE ENV.
//  * the grid comes from `ctx.geometry.grid`; the fallback is the DECODER's own N.
//  * the per-cell action-flip mask comes from `grid.actionMask`; the fallback is the
//    DECODER's own `largeDeepSeaMask(largeDeepSeaMaskWords(seed))`.
//  * the shelf triangle is `c ≤ r`, which is the reachability the env's own step function
//    produces (r increments every step, c moves ±1 from 0) — not a drawn shape.
//  * the treasure sits on `grid.goal`, which the decoder puts at (15, 15).
//
// LIGHT: the abyss palette's key is 0.35 and its hemisphere 0.45 — a ninth of noon. Two
// consequences drive every colour choice in this file. (1) Surface colour is split into a
// unit-peak HUE (`hue()`, a token blend) and a per-channel BEER–LAMBERT brightness term
// (`attenuate`), because the palette cannot light a surface at all: a pure white shelf
// arrives at RGB(20, 45, 88). Canon lesson 19 in reverse, and then some. (2) Emissive gain
// has to be high to bloom (canon lesson 22): the wreck heart, the anglerfish lures and the
// jellies run 1.3–4.2 where a noon flame ran at 0.98.

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
import { makeAmbientParticles, makeMistPlanes, type AmbientParticlesHandle } from '@kit/particles';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import {
  LARGE_DEEPSEA_N,
  largeDeepSeaMask,
  largeDeepSeaMaskWords,
} from '@sim/envs/largedeepsea';
import type { EnvGeometry, GridGeometry } from '@sim/types';

// ---------------------------------------------------------------------------
// world constants (1 cell = 1 world unit; the trench really is 16 × 16)
// ---------------------------------------------------------------------------

export const ABYSS_N = LARGE_DEEPSEA_N;
/** half the playfield footprint, derived from the env's own grid size */
export const TRENCH_HALF = (ABYSS_N * CELL) / 2;

/**
 * Shelf slab thickness. The shelf tops are all at y = 0 (STYLE.md's ground plane — the
 * wisp walks there), so the ONLY place a "terrace" can exist is BELOW: each cell is a
 * column standing on the trench bed, which turns the triangle's hypotenuse into a real
 * cliff and every row boundary into a visible step. A thin slab (the kit default 0.12)
 * made the shelf a decal on the seabed and the dive read as flat.
 *
 * It is deliberately thicker than the deepest bed. The bed DROPS as the dive goes south
 * (BED_NORTH → BED_SOUTH ± relief), so a slab sized for the shallow end left the southern
 * terraces floating over their own shadow. Sizing past the deepest possible bed buries
 * every column and — the part that turned out to matter — makes the EXPOSED cliff grow
 * from 0.9 at the surface to 2.1 on the trench floor. The staircase gets taller as it
 * descends, which is the one place this scene is allowed to show depth as height.
 */
export const SHELF_THICKNESS = 3.2;
/** the trench bed under the north (shallow) edge of the footprint */
export const BED_NORTH = -0.9;
/** …and under the south (deep) edge: the trench itself drops as the dive goes on */
export const BED_SOUTH = -2.1;
/**
 * Terrain extent. It is NOT sized so the walls close the frame — it is sized so the
 * seabed runs past the far edge of the shot (the top of the frame lands ~39 horizontal
 * units from the eye) and the picture never shows the plane end. What closes the frame is
 * `WALL_RUN` below, and only on the right.
 */
export const TERRAIN_SIZE = 88;
export const TERRAIN_SEGMENTS = 160;
/**
 * How far outside the footprint the canyon walls reach full height, and how high.
 *
 * Set by the CAMERA and by a seed sweep, not by taste — and this comment is the corrected
 * one: the number that used to be here (11) put the walls OUTSIDE THE SHOT, which is the
 * thing it claimed to fix. Measured against the delivered hero camera (target
 * [−3.6, −3.1, 3.3], distance 30, azimuth −2.52, polar 0.74, 1600×900), the crest ridge at
 * run 11 projects to px(1568, 909) … (1839, 7): off the right edge at every point. The
 * right third of the frame came back flat fog — mean RGB(10.8, 20.6, 40.6), luminance sd
 * 8.05 — which is exactly what "the trench has no sides" looks like.
 *
 * At run 7 the same crest projects to px(1217, 717) → (1421, 348) → (1549, 116): a ridge
 * that crosses the right third diagonally, 16–28 units from the eye. That measures as
 * RGB(10.6, 22.3, 44.2) with sd 8.60 — the right third gains structure and a form, and the
 * near canyon reads as the trench's own side.
 *
 * It is NOT a hard silhouette, and this scene cannot have one there: at 16–28 units this
 * fog has already lifted everything toward the fog colour, so a wall can only ever be
 * BRIGHTER than the water behind it, never a black cut-out. What you get is a lit ridge
 * through water, which is what a canyon wall at 25 m actually looks like.
 *
 * Why not closer, where it would read harder: the wall relief is seeded
 * (`makeTrenchHeight`), so a nearer canyon starts poking into the camera's sightline for
 * SOME lanes. Swept over 160 seeds and both cameras (hero and the thumb rig), the worst
 * clearance between the canyon and the eye-to-cell sightline is −1.78 units at run 5 (seed
 * 24 blocks the start cell), +0.09 at run 6 (knife-edge), +1.78 at run 7. Run 7 is the
 * smallest run that can never eat the playfield, and the start cell — where the wisp
 * returns every fifteen steps — is not optional. `large-deep-sea.test.ts` asserts that
 * clearance so a future tweak cannot quietly re-break it.
 */
export const WALL_RUN = 7;
export const WALL_HEIGHT = 8;
/** the canyon rock's albedo gain — it is lit by the same nothing the shelf is */
export const ROCK_GAIN = 2.2;

/**
 * Fog reach — the single most important number in this scene, and the one that took the
 * longest to find.
 *
 * `applyFog` turns extent into density via `fogDensityHint` (abyss = 0.85). At extent 112
 * the trench bed and the canyon walls kept 60–70% of their colour out to 40 units, which
 * meant the terrain — a 88-unit plane that fills the entire frame from this camera — came
 * back as a uniform mid-blue wall behind everything. Measured: RGB(22, 36, 60) in every
 * corner of the shot. The scene had no blacks at all, so nothing in it could be bright.
 *
 * At 68 the near bed keeps three quarters of itself, 30 units out is half gone and 40 is
 * three quarters gone, so the far half of the trench sinks into the fog colour (0x061224 —
 * nearly black) and the frame gets its blacks back. An abyss needs somewhere for things to
 * disappear TO; that is what distinguishes it from a dark room.
 */
export const FOG_EXTENT = 68;

/**
 * The last of the surface light: steep (69° up), from the north-west. Two reasons it is
 * near-vertical. Down here it is the remembered sun, and remembered sun comes from
 * straight overhead; and a grid scene dies on long raking shadows (canon lesson 8) — at
 * 69° a 2-unit shelf lip throws 0.75 of a cell, which reads as a step instead of
 * swallowing its neighbour.
 */
export const SUN_DIR: readonly [number, number, number] = [-0.3, 0.93, -0.22];
/** the shadow frustum covers the FRAME (playfield + the near canyon), not the playfield */
export const SHADOW_RADIUS = 13.5;

/**
 * The composition azimuth, shared with `index.ts`'s OrbitRig so the planting, the
 * anglerfish and the camera can never disagree. −2.52 rad puts the camera in the
 * NORTH-NORTH-WEST, looking south-south-east: the shallow rows are near and bright, the
 * dive recedes into the dark, and the wreck is a far glimmer at the bottom of the frame.
 * That is the shot — you are looking DOWN the trench, not across it.
 */
export const CAMERA_AZIMUTH = -2.52;

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
 * Take a token blend down toward the dark. `night.skyTop` is the darkest token there is
 * and STYLE.md names it as the legal way to reach near-black (canon lesson 9), so "sink"
 * is a blend like every other colour here — not a magic multiplier.
 */
export function sink(color: THREE.ColorRepresentation, k: number): THREE.Color {
  return blendTokens(color, TOKENS.palettes.night.skyTop, k);
}

/** The legal white (canon lesson 9): the cell field's base colour, so tints carry the hue. */
export const TOKEN_WHITE = TOKENS.palettes.noon.sun;

/**
 * Normalise a token blend to unit peak channel — the colour's HUE with its brightness
 * factored out, so a depth term can own the brightness alone. Mutates and returns `c`.
 */
export function hue(c: THREE.Color): THREE.Color {
  const m = Math.max(c.r, c.g, c.b, 1e-6);
  return c.multiplyScalar(1 / m);
}

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

/**
 * Merge pre-transformed parts into ONE geometry carrying a flat per-part vertex colour,
 * disposing the sources. Hand-built props (a wreck, an anglerfish, a coral whip) want
 * their SOURCE normals plus a per-part palette baked in, so one mesh + one material draws
 * the whole prop.
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
 * is built from the DECODER's own N and goal, never a scene re-derivation, so a
 * placeholder geometry degrades to the correct trench instead of a guess.
 */
export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === ABYSS_N && g.h === ABYSS_N) return g;
  return {
    w: ABYSS_N,
    h: ABYSS_N,
    walls: new Uint8Array(ABYSS_N * ABYSS_N),
    goal: { x: ABYSS_N - 1, y: ABYSS_N - 1 },
  };
}

/**
 * The per-cell action-flip mask. The env draws it once per lane from the RESET stream and
 * keeps it forever; the decoder puts it in `grid.actionMask`. The fallback re-runs the
 * DECODER's own draw for the lane seed — same function the sim uses, not a lookalike.
 */
export function resolveMask(grid: GridGeometry, seed: number): Uint8Array {
  const m = grid.actionMask;
  if (m && m.length >= grid.w * grid.h) return m;
  return largeDeepSeaMask(largeDeepSeaMaskWords(seed));
}

/**
 * THE REACHABLE SET. The env starts every dive at (0, 0), adds 1 to r every single step
 * and moves c by ±1 clamped to [0, N-1] — so after r steps the column can be at most r.
 * `c ≤ r` is therefore the exact set of cells the agent can ever occupy, plus the treasure
 * row. That set is the shelf; everything above the diagonal is open trench water.
 */
export function isShelfCell(x: number, y: number): boolean {
  return x <= y;
}

/** Number of shelf cells in an N×N trench — N(N+1)/2, the triangle. */
export function shelfCellCount(n: number): number {
  return (n * (n + 1)) / 2;
}

/** `true` for a cell the AGENT can actually stand on (row N−1 wraps in the same step). */
export function isWalkableCell(x: number, y: number, n: number): boolean {
  return isShelfCell(x, y) && y <= n - 2;
}

// ---------------------------------------------------------------------------
// the skin (every colour a blend of tokens — STYLE.md)
// ---------------------------------------------------------------------------

export interface AbyssSkin {
  /** row 0: the last of the reef, still lit */
  shelfLit: THREE.Color;
  /** the middle shelves, where blue turns to violet */
  shelfMid: THREE.Color;
  /** the trench floor: the colour of nothing */
  shelfDeep: THREE.Color;
  /** the seabed under and beyond the shelf, at the shallow end */
  bed: THREE.Color;
  /** …and at the deep end: the bed obeys the same falloff law the shelf does */
  bedDeep: THREE.Color;
  /** the canyon walls */
  rock: THREE.Color;
  rockLip: THREE.Color;
  /** coral holdfast / stalk */
  coral: THREE.Color;
  /** coral tip — where the biolume lives */
  coralTip: THREE.Color;
  /** drowned timber */
  timber: THREE.Color;
  timberDark: THREE.Color;
  /** the fish that carry their own light */
  anglerBody: THREE.Color;
  /** the one warm colour in the whole trench: the wreck's heart */
  heart: THREE.Color;
  /** jelly bell / jelly trail */
  jellyBell: THREE.Color;
  jellyTrail: THREE.Color;
  /** suspended particulate */
  snow: THREE.Color;
  /** the sediment a right-move kicks off a terrace — the env's per-step cost, drawn */
  silt: THREE.Color;
}

/**
 * The palette. Everything is a blend of the `abyss` palette with its shallower sibling
 * `underwater` (which DeepSea is painted in) — that IS the pair: the top of this trench is
 * the bottom of that reef, and you can see the handover happen across rows 0–4. The two
 * biolume hues are the scene accent (`large-deep-sea`, biolume ice-blue) and the shared
 * `cueMagenta`, which is the only warm-ish token that survives at this depth.
 */
export function abyssSkin(tokens: typeof TOKENS): AbyssSkin {
  const abyss = tokens.palettes.abyss;
  const water = tokens.palettes.underwater;
  const accent = tokens.accents['large-deep-sea'] ?? tokens.uiCyan;
  const reef = tokens.accents['deep-sea'] ?? water.groundB;

  // The trench's own rock lip and the marine snow already hanging in the water, named here
  // because the SILT is a blend of the two: sediment kicked off a terrace has to read as
  // this floor getting up and moving, not as a second, unrelated particle system.
  const rockLip = hue(blendTokens(abyss.groundB, water.groundA, 0.45));
  const snow = blendTokens(water.accentLight, abyss.hemiSky, 0.4);

  // THE THREE SHELF STOPS ARE HUES, NOT BRIGHTNESSES. Each is normalised to unit peak
  // channel (`hue()`), so interpolating between them changes only the COLOUR of the water
  // — reef teal, trench blue, violet slate — and every bit of the brightness comes from
  // the Beer–Lambert term in `shelfTint`. Mixing the two (a dark paint AND a falloff)
  // double-counted the depth and left the middle rows a mud that was neither.
  return {
    // The top shelf is a REEF shelf: it belongs to the sibling scene's palette almost
    // entirely, so the two read as one continuous dive.
    shelfLit: hue(blendTokens(water.groundB, reef, 0.35)),
    shelfMid: hue(blendTokens(water.groundA, abyss.groundB, 0.4)),
    // the trench floor is VIOLET, not blue: six percent of the shared cue magenta into
    // the abyss sky is the whole difference between "dark water" and "the bottom"
    shelfDeep: hue(blendTokens(abyss.skyTop, tokens.cueMagenta, 0.06)),
    bed: hue(blendTokens(abyss.groundB, water.groundA, 0.32)),
    bedDeep: hue(blendTokens(abyss.groundA, abyss.skyBottom, 0.4)),
    rock: hue(blendTokens(abyss.groundB, abyss.skyTop, 0.3)),
    rockLip,
    coral: hue(blendTokens(abyss.groundB, water.groundA, 0.4)),
    coralTip: blendTokens(accent, water.groundB, 0.42),
    // The props are hue + explicit gain for the same reason the shelf is. Drowned timber
    // painted at its token value rendered as a black hole under its OWN heart light —
    // there is not enough key down here for a dark paint to survive being lit. The wreck
    // has to be able to catch the light it is carrying, so its albedo is high and its
    // darkness comes from having no other light source anywhere near it.
    timber: hue(blendTokens(tokens.palettes.dusk.groundA, abyss.groundB, 0.45)).multiplyScalar(0.85),
    timberDark: hue(blendTokens(abyss.groundA, tokens.palettes.dusk.groundA, 0.3)).multiplyScalar(0.45),
    // the fish stays a silhouette: it is meant to be a light with an animal behind it
    anglerBody: hue(blendTokens(abyss.groundA, abyss.skyBottom, 0.45)).multiplyScalar(0.34),
    heart: blendTokens(tokens.celebrationGold, accent, 0.3),
    jellyBell: blendTokens(accent, water.skyTop, 0.28),
    jellyTrail: blendTokens(tokens.cueMagenta, accent, 0.55),
    snow,
    silt: blendTokens(rockLip, snow, 0.34),
  };
}

/**
 * THE LIGHT FALLOFF, as a pure function of the row. This is the scene's whole thesis in
 * six lines: rows 0–3 hand over from the reef's blue to the trench's violet, and from
 * there down the albedo sinks toward black on a squared curve, so the last four rows are
 * genuinely almost invisible and finding the wreck in them is genuinely hard.
 *
 * The returned colour is used as an instanceColor MULTIPLIER over a white cell field
 * (canon lesson 13), so it is the shelf's actual albedo — the falloff is the lighting, not
 * a tint on top of lighting.
 */
export function shelfTint(row: number, n: number, skin: AbyssSkin, out?: THREE.Color): THREE.Color {
  const c = out ?? new THREE.Color();
  const t = n > 1 ? clamp01(row / (n - 1)) : 0;
  // Hue first: reef teal handing over to trench blue over the top seven rows, then sinking
  // toward the violet slate of the floor. The handover sits at 0.45 for an AREA reason,
  // not a taste one — row r holds r+1 cells, so 43% of the shelf is rows 12–15 and only 6%
  // is rows 0–3. At 0.22 the whole reef-blue phase landed in six cells at the apex and the
  // gradient stopped existing on screen.
  const handover = 0.45;
  if (t <= handover) {
    c.copy(skin.shelfLit).lerp(skin.shelfMid, smoothstep01(t / handover));
  } else {
    const k = (t - handover) / (1 - handover);
    c.copy(skin.shelfMid).lerp(skin.shelfDeep, k * k * (1.35 - 0.35 * k));
  }
  return attenuate(c, t);
}

/**
 * BEER–LAMBERT: the depth falloff, per channel, and the whole reason this scene reads.
 *
 * Two facts forced it. First, the abyss palette CANNOT light a surface. Measured through
 * the real chain — key 0.35, hemisphere 0.45, ACES at exposure 1.0 — a PURE WHITE shelf
 * under this palette arrives at RGB(20, 45, 88), and the reef-blue it is supposed to be
 * arrives at RGB(7, 25, 50). The brief's own instruction is the way out: "shelf
 * emissive/hemisphere influence fades with depth (drive a per-cell tint from r via the
 * cell-field instanceColor)". An instanceColor is a MULTIPLIER over a white field, so a
 * value above 1 is the shallow rows keeping some of the surface's light — the hue stays a
 * token blend at every row, only the amount of light in it changes.
 *
 * Second, once brightness is a free parameter the honest curve for it is the one real
 * water uses: transmitted light falls as exp(−k·depth) with k largest for red and smallest
 * for blue. That single line delivers the brief's "top rows still ocean-blue, the floor
 * near-black" AND the violet cast of the deep for free — by row 15 red has been attenuated
 * 81× and blue only 25×. It also keeps the falloff a RATIO, which is what lets the lure
 * and diving-bell pools still read on the deep shelves instead of multiplying into black.
 *
 * The first pass used a plain additive rim for the terrace edges instead, and learned the
 * hard way: a fresnel term is albedo-INDEPENDENT, so at rimStrength 0.5 it was seven times
 * the diffuse on the trench floor and the deepest rows rendered BRIGHTER than the surface.
 *
 * Measured off the hero frame, down the middle of the shelf: row 0 → RGB(0, 126, 168),
 * row 2 → RGB(0, 90, 140), row 7 → RGB(2, 33, 78), row 12 → RGB(10, 22, 49),
 * row 15 → RGB(4, 16, 41). The lower bound is the host grade's shadow lift, not black:
 * nothing in any OaK scene renders under about RGB(22, 26, 29).
 */
export const EXTINCTION: readonly [number, number, number] = [4.4, 3.9, 3.2];
/** albedo the surface row carries; the extinction takes it down from there */
export const SURFACE_GAIN = 11;

/** Attenuate a unit-peak hue to the light left at normalised depth `t`. Mutates `c`. */
export function attenuate(c: THREE.Color, t: number): THREE.Color {
  const d = clamp01(t);
  c.r *= SURFACE_GAIN * Math.exp(-EXTINCTION[0] * d);
  c.g *= SURFACE_GAIN * Math.exp(-EXTINCTION[1] * d);
  c.b *= SURFACE_GAIN * Math.exp(-EXTINCTION[2] * d);
  return c;
}

// ---------------------------------------------------------------------------
// the trench height field
// ---------------------------------------------------------------------------

/**
 * The seabed. Inside the playfield it is a gently undulating bed that DROPS as the dive
 * goes south (BED_NORTH → BED_SOUTH), so the open water above the diagonal is a real hole
 * with a real bottom rather than a gap in the floor; outside it rears up into canyon walls
 * over `WALL_RUN` units. The shelf slabs are `SHELF_THICKNESS` deep and their bottoms are
 * always below the bed, so the terrace columns are buried, not floating.
 */
export function makeTrenchHeight(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const outside = Math.max(Math.abs(x), Math.abs(z)) - TRENCH_HALF;
    const wall = smoothstep01(outside / WALL_RUN);
    const bedT = clamp01((z + TRENCH_HALF) / (2 * TRENCH_HALF));
    const bed = BED_NORTH + (BED_SOUTH - BED_NORTH) * bedT;
    const n1 = valueNoise2(x * 0.062 + 3.1, z * 0.062 + 8.7, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.173 + 1.7, z * 0.173 + 4.3, 64, seed ^ 0x51ed270b) - 0.5;
    const relief = n1 * 2.1 + n2 * 0.75;
    return bed + relief * (0.18 + 6.2 * wall) + wall * WALL_HEIGHT;
  };
}

/**
 * The seabed's colour, as a function of where it lies rather than how high it is: the bed
 * carries the SAME depth falloff the shelf does (north shallow → south dark) and only
 * turns to rock once it starts climbing the canyon wall.
 *
 * Returns a shared scratch Color — `makeTerrain` copies it per vertex, so this allocates
 * nothing across 25k samples.
 */
export function makeTrenchColor(skin: AbyssSkin): (h: number, slope: number, x: number, z: number) => THREE.Color {
  const scratch = new THREE.Color();
  const climb = new THREE.Color();
  return (h, _slope, _x, z) => {
    const depthT = clamp01((z + TRENCH_HALF) / (2 * TRENCH_HALF));
    const wallT = clamp01((h - BED_SOUTH) / (WALL_HEIGHT * 0.7));
    // the BED obeys the same Beer–Lambert law the shelf does, plus a shadow term — it is
    // a floor in the terraces' shade, so it keeps 62% of what the shelf above it keeps
    scratch.copy(skin.bed).lerp(skin.bedDeep, depthT * depthT * (1.4 - 0.4 * depthT));
    attenuate(scratch, depthT).multiplyScalar(0.62);
    if (wallT <= 0) return scratch;
    // The WALLS do not. They climb out of the trench toward whatever light is left, so
    // their falloff is linear in depth rather than exponential in it — and that is the
    // whole reason they are visible at all. Given the exponential treatment they went
    // black at every altitude and the trench lost its sides.
    climb
      .copy(skin.rock)
      .lerp(skin.rockLip, clamp01((wallT - 0.5) * 2))
      .multiplyScalar(ROCK_GAIN * (0.42 + 0.58 * (1 - depthT)));
    return scratch.lerp(climb, smoothstep01(wallT));
  };
}

// ---------------------------------------------------------------------------
// glow-jellies — the ONE bespoke ShaderMaterial this scene is allowed
// ---------------------------------------------------------------------------

/**
 * STYLE.md: "At most ONE bespoke ShaderMaterial per scene — it must consume token colors
 * and ctx.time()." This is it, and it is spent on the thing the brief calls for by name:
 * "glow-jellies drifting (instanced, vertex-shader motion)".
 *
 * Every jelly is an INSTANCE of one 70-triangle bell-and-ribbons geometry, and every bit
 * of its life happens in the vertex shader from `uTime`:
 *   * the bell contracts and relaxes (squash in y, swell in xz) on its own phase,
 *   * the ribbons trail the contraction with a lag, which is what makes it read as
 *     swimming rather than pulsing,
 *   * the whole animal drifts on a slow lissajous keyed to its own origin.
 * The CPU writes the instance matrices exactly once. One draw call, zero per-frame JS,
 * and a paused playback freezes the swarm mid-stroke like everything else.
 */
const JELLY_VERT = /* glsl */ `
#include <common>
uniform float uTime;
uniform float uDrift;
uniform float uPulse;
attribute vec2 aJelly;   // x = phase 0..1, y = stroke speed
attribute float aGrade;  // 1 at the crown, 0 at the ribbon tips
varying float vGrade;
varying vec3 vNormalV;
varying vec3 vViewDir;
void main() {
	float phase = aJelly.x * 6.2831853;
	float speed = aJelly.y;
	float stroke = sin( uTime * speed + phase );
	float lag = sin( uTime * speed - 1.05 + phase );

	vec3 p = position;
	float ribbon = 1.0 - aGrade;
	// the bell: squash vertically, swell radially, and inversely on the relax
	p.y *= 1.0 + uPulse * stroke * aGrade;
	p.xz *= 1.0 - uPulse * 0.55 * stroke * aGrade;
	// the ribbons: they follow, they never lead
	p.x += ribbon * ribbon * lag * 0.11;
	p.z += ribbon * ribbon * cos( uTime * speed * 0.73 + phase ) * 0.085;

	vec4 local = instanceMatrix * vec4( p, 1.0 );
	vec3 origin = instanceMatrix[ 3 ].xyz;
	local.x += sin( uTime * 0.19 + origin.z * 0.55 + phase ) * uDrift;
	local.y += sin( uTime * 0.27 + phase ) * uDrift * 0.55;
	local.z += cos( uTime * 0.23 + origin.x * 0.47 + phase ) * uDrift;

	vec4 mv = modelViewMatrix * local;
	vNormalV = normalize( normalMatrix * ( mat3( instanceMatrix ) * normal ) );
	vViewDir = - mv.xyz;
	vGrade = aGrade;
	gl_Position = projectionMatrix * mv;
}
`;

const JELLY_FRAG = /* glsl */ `
#include <common>
uniform vec3 uBell;
uniform vec3 uTrail;
uniform float uIntensity;
varying float vGrade;
varying vec3 vNormalV;
varying vec3 vViewDir;
void main() {
	float f = 1.0 - abs( dot( normalize( vNormalV ), normalize( vViewDir ) ) );
	float rim = pow( f, 1.9 );
	vec3 col = mix( uTrail, uBell, vGrade ) * ( 0.28 + 1.45 * rim ) * uIntensity;
	float a = clamp( ( 0.16 + 0.6 * rim ) * ( 0.22 + 0.78 * vGrade ), 0.0, 1.0 );
	if ( a <= 0.004 ) discard;
	gl_FragColor = vec4( col, a );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}
`;

export interface JellySwarmHandle {
  mesh: THREE.InstancedMesh;
  material: THREE.ShaderMaterial;
  count: number;
  /** ctx.time() — the ONLY clock */
  setTime(t: number): void;
  dispose(): void;
}

/**
 * One jelly: a scalloped bell of `rings × segments` plus `ribbons` tapered streamers.
 * `aGrade` runs 1 at the crown to 0 at the ribbon tips and carries BOTH the colour
 * gradient and the amount of swim deformation, so the geometry and the motion can never
 * disagree about which end is which.
 */
export function makeJellyGeometry(detail: number): THREE.BufferGeometry {
  const segments = Math.max(6, Math.round(10 * detail));
  const rings = Math.max(2, Math.round(3 * detail));
  const ribbons = Math.max(3, Math.round(5 * detail));
  // A jelly is a GRACE NOTE, not a subject. The first pass built them at cell scale
  // (0.23 bell, 0.62 tail) and 62 of them ate the entire frame: sixty white parachutes
  // over a trench you could no longer see. At 0.13/0.34 with 44 instances they are what
  // the brief asks for — drifting light in the water — and the dive stays the subject.
  const RAD = 0.13;
  const BELL = 0.17;
  const TAIL = 0.34;

  const positions: number[] = [];
  const grades: number[] = [];

  const bellPoint = (ring: number, seg: number): readonly [number, number, number] => {
    const t = ring / rings;
    const theta = t * Math.PI * 0.54;
    const a = (seg / segments) * Math.PI * 2;
    // scallop the rim so the silhouette is an animal, not a lampshade
    const scallop = 1 + 0.08 * Math.sin(a * segments * 0.5) * t;
    const r = Math.sin(theta) * RAD * scallop;
    const y = Math.cos(theta) * BELL;
    return [Math.cos(a) * r, y, Math.sin(a) * r];
  };

  const push = (p: readonly [number, number, number], g: number): void => {
    positions.push(p[0], p[1], p[2]);
    grades.push(g);
  };

  for (let ring = 0; ring < rings; ring++) {
    const g0 = 1 - 0.28 * (ring / rings);
    const g1 = 1 - 0.28 * ((ring + 1) / rings);
    for (let seg = 0; seg < segments; seg++) {
      const a = bellPoint(ring, seg);
      const b = bellPoint(ring, seg + 1);
      const c = bellPoint(ring + 1, seg + 1);
      const d = bellPoint(ring + 1, seg);
      push(a, g0);
      push(c, g1);
      push(b, g0);
      push(a, g0);
      push(d, g1);
      push(c, g1);
    }
  }

  const rimY = Math.cos(Math.PI * 0.54) * BELL;
  const rimR = Math.sin(Math.PI * 0.54) * RAD;
  for (let i = 0; i < ribbons; i++) {
    const a = (i / ribbons) * Math.PI * 2 + 0.31;
    const cx = Math.cos(a);
    const cz = Math.sin(a);
    const nx = -cz;
    const nz = cx;
    const w = 0.05;
    const len = TAIL * (0.72 + 0.28 * (((i * 5) % 3) / 2));
    const x0 = cx * rimR * 0.86;
    const z0 = cz * rimR * 0.86;
    const x1 = cx * rimR * 0.34;
    const z1 = cz * rimR * 0.34;
    const top: readonly [number, number, number] = [x0 - nx * w, rimY, z0 - nz * w];
    const top2: readonly [number, number, number] = [x0 + nx * w, rimY, z0 + nz * w];
    const mid: readonly [number, number, number] = [x1 - nx * w * 0.5, rimY - len * 0.55, z1 - nz * w * 0.5];
    const mid2: readonly [number, number, number] = [x1 + nx * w * 0.5, rimY - len * 0.55, z1 + nz * w * 0.5];
    const tip: readonly [number, number, number] = [x1 * 0.4, rimY - len, z1 * 0.4];
    push(top, 0.55);
    push(mid, 0.26);
    push(top2, 0.55);
    push(top2, 0.55);
    push(mid, 0.26);
    push(mid2, 0.26);
    push(mid, 0.26);
    push(tip, 0);
    push(mid2, 0.26);
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  g.setAttribute('aGrade', new THREE.BufferAttribute(new Float32Array(grades), 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'abyssJelly';
  return g;
}

export interface JellySwarmOptions {
  grid: GridSize;
  count: number;
  seed: number;
  tier: QualityTier;
  bell: THREE.ColorRepresentation;
  trail: THREE.ColorRepresentation;
  intensity: number;
  name?: string;
}

export function buildJellySwarm(opts: JellySwarmOptions): JellySwarmHandle {
  const detail = opts.tier === 'thumb' ? 0.55 : opts.tier === 'medium' ? 0.8 : 1;
  const count = Math.max(1, opts.count);
  const geometry = makeJellyGeometry(detail);

  const jelly = new Float32Array(count * 2);
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3();

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uDrift: { value: 0.42 },
      uPulse: { value: 0.2 },
      uBell: { value: new THREE.Color(opts.bell) },
      uTrail: { value: new THREE.Color(opts.trail) },
      uIntensity: { value: opts.intensity },
    },
    vertexShader: JELLY_VERT,
    fragmentShader: JELLY_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  material.name = opts.name ?? 'abyssJellySwarm';

  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.name = opts.name ?? 'abyssJellySwarm';
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.frustumCulled = false;
  mesh.renderOrder = 2;

  // Jellies drift over the WHOLE trench, shelf and open water alike — the empty half of
  // the grid is water too, and leaving it bare made a third of the frame a dead black
  // rectangle. They are still biased toward the shelf (two thirds of the area is shelf),
  // which is where the composition wants them.
  let placed = 0;
  let attempt = 0;
  while (placed < count && attempt < count * 40) {
    const h = hash2f(attempt, 17, opts.seed ^ 0x4a11ef);
    const h2 = hash2f(attempt, 41, opts.seed ^ 0x1d05b3);
    const h3 = hash2f(attempt, 73, opts.seed ^ 0x77c0de);
    const h4 = hash2f(attempt, 109, opts.seed ^ 0x2be115);
    attempt++;
    const cx = h * opts.grid.w;
    const cy = h2 * opts.grid.h;
    const world: Vec3Tuple = [0, 0, 0];
    cellToWorld(cx - 0.5, cy - 0.5, opts.grid, world);
    // they hang LOW over the shelf, not up in the camera's lap: at y > 4 the near ones
    // are a third of the frame each and the trench disappears behind them
    const depth = clamp01(cy / opts.grid.h);
    const open = isShelfCell(Math.floor(cx), Math.floor(cy)) ? 0 : 1;
    const y = 0.55 - open * 1.1 + h3 * (0.9 + 1.5 * depth);
    const s = 0.58 + h4 * 0.62;
    position.set(world[0], y, world[2]);
    euler.set((h3 - 0.5) * 0.5, h * Math.PI * 2, (h4 - 0.5) * 0.5);
    quaternion.setFromEuler(euler);
    scale.set(s, s * (0.85 + h3 * 0.4), s);
    matrix.compose(position, quaternion, scale);
    mesh.setMatrixAt(placed, matrix);
    jelly[placed * 2] = h4;
    jelly[placed * 2 + 1] = 0.55 + h3 * 0.75;
    placed++;
  }
  mesh.count = placed;
  mesh.instanceMatrix.needsUpdate = true;
  geometry.setAttribute('aJelly', new THREE.InstancedBufferAttribute(jelly, 2));

  return {
    mesh,
    material,
    count: placed,
    setTime(t) {
      (material.uniforms.uTime as { value: number }).value = t;
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}

// ---------------------------------------------------------------------------
// whisker coral — the action mask, told truthfully and quietly
// ---------------------------------------------------------------------------

/**
 * STYLE.md honesty rule: "Never hide what the env defines (… DeepSea mask currents — show
 * them, subtly)." LargeDeepSea's hidden per-cell map decides which of the two actions
 * moves the agent RIGHT, and it is fixed for the lane's whole life — so it is a property
 * of the PLACE, and a place shows what it is by what grows on it. One deep-water whisker
 * coral per shelf cell, leaning +x on a set bit and −x on a clear one.
 *
 * It is deliberately quiet: 0.32 rad of lean, off-centre so the cell the agent stands on
 * stays clear, and the same slow current sway on every whip. You can read the whole map if
 * you look for it, and you never notice it if you don't.
 */
export const CORAL_LEAN = 0.32;

/** the lean is a roll about +z, so it always tips the whip along ±x — the mask's axis */
const LEAN_AXIS = new THREE.Vector3(0, 0, 1);

export interface CoralFieldHandle {
  mesh: THREE.InstancedMesh;
  material: OakMaterial;
  count: number;
  setTime(t: number): void;
  dispose(): void;
}

/** One whip: a holdfast, a main stalk and two side whiskers, graded dark base → lit tip. */
export function makeCoralGeometry(detail: number): THREE.BufferGeometry {
  const radial = Math.max(3, Math.round(5 * detail));
  const parts: THREE.BufferGeometry[] = [];

  const whip = (
    height: number,
    base: number,
    tip: number,
    tx: number,
    tz: number,
    tilt: number,
    yaw: number,
  ): THREE.BufferGeometry => {
    const g = new THREE.CylinderGeometry(tip, base, height, radial, 1, true);
    g.translate(0, height / 2, 0);
    g.rotateZ(tilt);
    g.rotateY(yaw);
    g.translate(tx, 0.02, tz);
    return g;
  };

  parts.push(whip(0.52, 0.022, 0.005, 0, 0, 0, 0));
  parts.push(whip(0.33, 0.016, 0.004, 0.045, 0.02, 0.42, 0.7));
  parts.push(whip(0.26, 0.014, 0.004, -0.04, -0.03, -0.5, -0.4));
  const holdfast = new THREE.CylinderGeometry(0.045, 0.075, 0.05, Math.max(4, radial + 1), 1);
  holdfast.translate(0, 0.025, 0);
  parts.push(holdfast);

  // one geometry, with a vertical grade baked into the vertex colour: the tips are where
  // the biolume is, and instanceColor tints the whole whip by its row on top of that
  let total = 0;
  for (const p of parts) total += p.index ? p.index.count : p.getAttribute('position').count;
  const positions = new Float32Array(total * 3);
  const normals = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);
  let o = 0;
  for (const geo of parts) {
    const pos = geo.getAttribute('position');
    const nor = geo.getAttribute('normal');
    const index = geo.index;
    const n = index ? index.count : pos.count;
    for (let i = 0; i < n; i++) {
      const vi = index ? index.getX(i) : i;
      const y = pos.getY(vi);
      positions[o * 3] = pos.getX(vi);
      positions[o * 3 + 1] = y;
      positions[o * 3 + 2] = pos.getZ(vi);
      if (nor) {
        normals[o * 3] = nor.getX(vi);
        normals[o * 3 + 1] = nor.getY(vi);
        normals[o * 3 + 2] = nor.getZ(vi);
      }
      const k = 0.45 + 1.25 * clamp01(y / 0.52);
      colors[o * 3] = k;
      colors[o * 3 + 1] = k;
      colors[o * 3 + 2] = k;
      o++;
    }
  }
  for (const p of parts) p.dispose();
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeBoundingSphere();
  g.name = 'abyssCoral';
  return g;
}

export interface CoralFieldOptions {
  grid: GridSize;
  mask: Uint8Array;
  skin: AbyssSkin;
  seed: number;
  tier: QualityTier;
  name?: string;
}

export function buildCoralField(opts: CoralFieldOptions): CoralFieldHandle {
  const { grid, mask, skin } = opts;
  const detail = opts.tier === 'thumb' ? 0.5 : opts.tier === 'medium' ? 0.75 : 1;
  const geometry = makeCoralGeometry(detail);
  const capacity = shelfCellCount(Math.min(grid.w, grid.h));

  const material = oakMaterial({
    color: skin.coral,
    vertexColors: true,
    flatShading: true,
    roughness: 0.85,
    side: THREE.DoubleSide,
    rimColor: skin.coralTip,
    rimStrength: 0.65,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.3,
    // the current: one slow sway for every whip in the trench (canon lesson 10 — the
    // phase comes from the INSTANCE origin, which is why this is an InstancedMesh)
    wind: { amplitude: 0.028, frequency: 0.62, height: 0.5 },
    tier: opts.tier,
    name: 'abyssCoralMaterial',
  });

  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, capacity));
  mesh.name = opts.name ?? 'abyssCoral';
  mesh.castShadow = false;
  mesh.receiveShadow = opts.tier !== 'thumb';
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, capacity) * 3), 3);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const world: Vec3Tuple = [0, 0, 0];
  const tint = new THREE.Color();
  const biolume = new THREE.Color();
  const rowColor = new THREE.Color();

  let n = 0;
  for (let y = 0; y < grid.h; y++) {
    shelfTint(y, grid.h, skin, rowColor);
    for (let x = 0; x < grid.w; x++) {
      if (!isShelfCell(x, y)) continue;
      const bit = mask[y * grid.w + x] ?? 0;
      const h = hash2f(x, y, opts.seed ^ 0x0c02a1);
      const h2 = hash2f(x, y, opts.seed ^ 0x5eaf00);
      cellToWorld(x, y, grid, world);
      // off the cell CENTRE: the square the agent stands on stays clear
      const ox = (0.24 + 0.12 * h) * (h2 > 0.5 ? 1 : -1);
      const oz = (0.22 + 0.13 * h2) * (h > 0.5 ? 1 : -1);
      position.set(world[0] + ox, 0, world[2] + oz);
      // lean along ±x — the mask's own axis. Rotating about z by −dir·lean tips the
      // whip's local +y toward +dir·x, which is exactly "the current runs that way".
      const dir = bit !== 0 ? 1 : -1;
      quaternion.setFromAxisAngle(LEAN_AXIS, -dir * CORAL_LEAN * (0.8 + 0.4 * h));
      const s = 0.7 + 0.55 * h2;
      scale.set(s, s * (0.85 + 0.4 * h), s);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(n, matrix);
      // The whip obeys the same Beer–Lambert falloff the shelf under it does, and then
      // gets a small biolume floor ADDED on top — a floor, not a blend, so the shallow
      // coral is not dimmed by it and the deep coral is still faintly visible after the
      // rock it grows out of has gone black. That is what makes the mask readable all the
      // way down the trench, which is the honesty this whole organ exists for.
      tint.copy(rowColor).multiplyScalar(0.8);
      biolume.copy(skin.coralTip).multiplyScalar(0.06 + 0.2 * clamp01(y / (grid.h - 1)));
      tint.add(biolume);
      mesh.setColorAt(n, tint);
      n++;
    }
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();

  return {
    mesh,
    material,
    count: n,
    setTime(t) {
      oakUniforms(material)?.setTime(t);
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}

/** The lean the coral on a cell shows, in radians: > 0 leans +x (a set mask bit). */
export function coralLeanAt(mask: Uint8Array, grid: GridSize, x: number, y: number): number {
  const bit = mask[y * grid.w + x] ?? 0;
  return bit !== 0 ? CORAL_LEAN : -CORAL_LEAN;
}

// ---------------------------------------------------------------------------
// anglerfish sites
// ---------------------------------------------------------------------------

export interface AnglerSite {
  /** shelf cell the fish patrols over — always satisfies `isShelfCell` */
  x: number;
  y: number;
  hover: number;
  scale: number;
  phase: number;
}

/**
 * Where the two lures hang, deterministic from the lane seed (kit/rng, never Math.random).
 *
 * They are pushed toward the MIDDLE columns on purpose. This lane's policy hugs c ≤ 3 for
 * 96% of its steps, so a lure planted on the wisp's own track would sit on top of it every
 * fifteen steps; out at c ≈ 5–10 they light the part of the shelf the agent almost never
 * reaches — which is the half of the trench the composition needs lit and the story needs
 * empty.
 */
export function anglerSites(grid: GridSize, seed: number): AnglerSite[] {
  const a = hash2f(3, 11, seed ^ 0x104e);
  const b = hash2f(7, 23, seed ^ 0x2a71);
  const c = hash2f(13, 5, seed ^ 0x39b2);
  const d = hash2f(17, 29, seed ^ 0x4c08);
  const lastRow = Math.max(1, grid.h - 2);
  const rowA = Math.min(lastRow, 6 + Math.floor(a * 4));
  const rowB = Math.min(lastRow, 11 + Math.floor(c * 4));
  const colA = Math.max(2, Math.min(rowA - 1, rowA - 1 - Math.floor(b * 3)));
  const colB = Math.max(4, Math.min(rowB - 1, rowB - 2 - Math.floor(d * 5)));
  return [
    { x: colA, y: rowA, hover: 1.4, scale: 0.95, phase: 0.6 },
    { x: colB, y: rowB, hover: 1.75, scale: 1.25, phase: 3.1 },
  ];
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
  coral: CoralFieldHandle;
  jellies: JellySwarmHandle;
  /** the resurface set piece: a bubble column the scene flies back up the trench */
  rush: AmbientParticlesHandle;
  /**
   * The right-move cost, drawn: sediment the bell kicks off the terrace whenever the env
   * charges `LARGE_DEEPSEA_RIGHT_COST`. The SCENE owns its position (the wisp's true cell)
   * and its density (the eased action bit) — this handle is just the puff.
   */
  silt: AmbientParticlesHandle;
  skin: AbyssSkin;
  /** ctx.time() — pause freezes coral, jellies, marine snow and bubbles together */
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

export function buildWorld(
  ctx: SceneContext,
  scene: THREE.Scene,
  grid: GridGeometry,
  mask: Uint8Array,
): WorldHandle {
  const tier: QualityTier = ctx.tier;
  const palette: Palette = ctx.tokens.palettes.abyss;
  const water: Palette = ctx.tokens.palettes.underwater;
  const skin = abyssSkin(ctx.tokens);
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'theAbyss';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const sized: AmbientParticlesHandle[] = [];

  // ---- the depth column (exactly one sky; `abyss` compiles the sun disc out) ----
  const sky = skyFromPalette(palette, 'abyss', tier, {
    topSpread: 0.42,
    bottomSpread: 0.62,
    radius: 320,
    dither: 0.01,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the one light budget: sun + hemi + 4 points (wisp / wreck / two lures) ----
  const rig = makeLightRig(palette, tier, {
    sunDir: SUN_DIR,
    pointCount: 4,
    // Down here the point lights are not accents, they ARE the lighting — the remembered
    // sun contributes about a tenth of what a noon key does. But biolume is still a SMALL
    // light: a lure that reaches 20 units floods the whole trench and the darkness stops
    // being the subject. 9 units keeps every one of them a pool you can walk out of.
    pointDistance: 9,
    sunDistanceScale: 2.4,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: -0.3, z: 0 }, SHADOW_RADIUS);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the trench: bed under the shelf, canyon walls beyond it ----
  const terrain = makeTerrain({
    size: [TERRAIN_SIZE, TERRAIN_SIZE],
    segments: [TERRAIN_SEGMENTS, TERRAIN_SEGMENTS],
    tier,
    seed: ctx.seed,
    heightFn: makeTrenchHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.07,
    slopeColor: skin.rock,
    slopeStrength: 0.65,
    slopeSharpness: 1.8,
    // A ramp over normalised HEIGHT would spend 90% of its range on the canyon walls and
    // give the seabed one flat colour. A colour FUNCTION lets the bed obey the same law
    // the shelf does — the floor under the open water darkens with the row it lies in —
    // so the drop-off beyond the shelf edge is dark at the deep end and merely dim at the
    // shallow one, exactly like the terraces above it.
    colorRamp: makeTrenchColor(skin),
    materialOptions: {
      rimColor: water.skyHorizon,
      rimStrength: 0.22,
      toonSteps: 3,
      toonSoftness: 0.32,
      toonFloor: 0.3,
    },
    receiveShadow: shadows,
    castShadow: false,
    name: 'trenchBed',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  // ---- THE SHELF: one draw, straight off the reachable triangle ----
  // White base + full-colour tint (canon lesson 13): the instance colour MULTIPLIES the
  // material colour, so `shelfTint(row)` IS the shelf's albedo and the depth falloff is
  // one function of one integer.
  const field = buildCellField(
    grid,
    {
      floorColor: TOKEN_WHITE,
      wallColor: TOKEN_WHITE,
      floorFilter: isShelfCell,
      floorTint: (_x, y) => shelfTint(y, grid.h, skin),
      floorInset: 0.035,
      floorThickness: SHELF_THICKNESS,
      jitter: 0.08,
      // The rim draws the cliff faces (a fresnel term only fires at grazing angles), but
      // it is kept SMALL and it is the reason why. A rim is albedo-INDEPENDENT: at 0.5 it
      // was seven times the diffuse on the trench floor, so the deepest terraces rendered
      // brighter than the surface ones and the entire depth gradient ran backwards. At
      // 0.18 it is an edge on the steps and nothing else.
      rimColor: water.skyTop,
      rimStrength: 0.18,
      seed: ctx.seed,
      name: 'abyssShelves',
    },
    tier,
  );
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  // ---- the whisker coral: the action mask, one instance per shelf cell ----
  const coral = buildCoralField({ grid, mask, skin, seed: ctx.seed, tier });
  group.add(coral.mesh);
  owned.push(coral);
  timed.push(coral);

  // ---- the glow-jellies (the one bespoke shader) ----
  const jellyCount = tier === 'thumb' ? 12 : tier === 'medium' ? 26 : 46;
  const jellies = buildJellySwarm({
    grid,
    count: jellyCount,
    seed: ctx.seed,
    tier,
    bell: skin.jellyBell,
    trail: skin.jellyTrail,
    // canon lesson 22 cuts both ways: an emitter needs gain to BLOOM in a dark palette,
    // but on a black background it also needs restraint not to become the picture
    intensity: tier === 'thumb' ? 1.0 : 1.35,
    name: 'abyssJellies',
  });
  group.add(jellies.mesh);
  owned.push(jellies);
  timed.push(jellies);

  // ---- suspended life ----
  // Kit particle sizes are tuned for ~20-unit fields (canon lesson 5); this trench is 16
  // wide with the camera ~24 out, so everything here sits near 1, not the presets' 3–7.
  const snow = makeAmbientParticles({
    preset: 'snow',
    count: 330,
    bounds: { x: 0, y: 2.1, z: 0, halfX: TRENCH_HALF, halfY: 3.0, halfZ: TRENCH_HALF },
    seed: ctx.seed ^ 0x5f0a,
    tier,
    color: skin.snow,
    size: 0.8,
    intensity: 0.85,
    opacity: 0.32,
    rise: -0.11,
    swirl: 0.12,
    name: 'marineSnow',
  });
  group.add(snow.points);
  owned.push(snow);
  timed.push(snow);
  sized.push(snow);

  // the shallow rows still boil a little: bubbles only over the top of the dive
  const bubbles = makeAmbientParticles({
    preset: 'bubbles',
    count: 120,
    bounds: { x: -TRENCH_HALF * 0.45, y: 1.9, z: -TRENCH_HALF * 0.55, halfX: 3.4, halfY: 2.2, halfZ: 3.0 },
    seed: ctx.seed ^ 0x2b17,
    tier,
    color: water.accentLight,
    size: 1.0,
    intensity: 1.35,
    opacity: 0.4,
    rise: 0.5,
    swirl: 0.1,
    name: 'shelfBubbles',
  });
  group.add(bubbles.points);
  owned.push(bubbles);
  timed.push(bubbles);
  sized.push(bubbles);

  // ---- the right-move cost: the silt the bell stirs off a terrace ----
  // Small, low and close: a puff inside one cell, not a cloud around the diver. It is the
  // one particle system in this scene whose POSITION is state — the scene pins it to the
  // wisp's true cell every frame and the probe suite locks it there — so its bounds are
  // sized in cell units, not in trench units (canon lesson 5, at the small end).
  const silt = makeAmbientParticles({
    preset: 'dust',
    count: 120,
    // wider than the bell it hides under (radius 0.36) and still inside the 1-unit cell,
    // so the puff reads as a skirt around the slab rather than as fog inside the glass
    bounds: { x: 0, y: 0, z: 0, halfX: 0.44, halfY: 0.13, halfZ: 0.44 },
    seed: ctx.seed ^ 0x1d5e,
    tier,
    color: skin.silt,
    // sized like the rest of THIS scene, not like the kit preset: at a 28-unit standoff
    // the first pass (0.75 / 1.1 / 0.34) measured as a handful of stray pixels. The
    // resurface column next door runs 1.7 / 2.4 / 0.8 and reads; this sits under it.
    size: 1.1,
    intensity: 1.3,
    opacity: 0.42,
    // sediment SETTLES: it is the only particle field down here that falls
    rise: -0.09,
    swirl: 0.07,
    name: 'diveSilt',
  });
  silt.setDensity(0);
  group.add(silt.points);
  owned.push(silt);
  timed.push(silt);
  sized.push(silt);

  // ---- the resurface set piece ----
  // A tall, narrow bubble column that the SCENE flies from the cell the dive ended on back
  // to the surface cell. It is parked at density 0 (every particle culled in the vertex
  // shader) until a `resurface` event fires — see index.ts's `rushAt` envelope.
  const rush = makeAmbientParticles({
    preset: 'bubbles',
    count: 210,
    bounds: { x: 0, y: 0, z: 0, halfX: 0.32, halfY: 1.5, halfZ: 0.32 },
    seed: ctx.seed ^ 0x7a91,
    tier,
    color: water.accentLight,
    size: 1.7,
    intensity: 2.4,
    opacity: 0.8,
    rise: 1.7,
    swirl: 0.07,
    name: 'resurfaceRush',
  });
  rush.setDensity(0);
  group.add(rush.points);
  owned.push(rush);
  timed.push(rush);
  sized.push(rush);

  // ---- the last of the surface light, as suspended haze over the shallow rows ----
  const haze = makeMistPlanes({
    size: [TERRAIN_SIZE * 0.26, TERRAIN_SIZE * 0.26],
    color: blendTokens(water.skyHorizon, palette.skyTop, 0.45),
    seed: ctx.seed ^ 0x3c40,
    tier,
    layers: 4,
    baseY: 2.4,
    spread: 4.4,
    opacity: 0.07,
    speed: 0.006,
    scale: 1.8,
    nearFade: 6,
    name: 'abyssHaze',
  });
  // the haze sits over the SHALLOW corner and nowhere else: it is the last of the surface
  // light hanging in the water at the top of the dive, and it has no business at the bottom
  haze.group.position.set(-TRENCH_HALF * 0.55, 0, -TRENCH_HALF * 0.55);
  group.add(haze.group);
  owned.push(haze);
  timed.push(haze);

  return {
    group,
    sky,
    rig,
    terrain,
    field,
    coral,
    jellies,
    rush,
    silt,
    skin,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
    },
    setPixelRatio(dpr) {
      for (let i = 0; i < sized.length; i++) sized[i]?.setPixelRatio(dpr);
    },
    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      sized.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}
