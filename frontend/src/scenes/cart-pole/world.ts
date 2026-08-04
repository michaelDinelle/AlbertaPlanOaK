// The Boardwalk — the static world.
//
// THE ACCURACY CLAIM OF THIS FILE. Three numbers in this scene are the env, not a
// designer's taste, and all three are READ OUT OF `ctx.geometry.classic.anchors` rather
// than re-derived:
//
//   * the deck runs from `trackLeft.x` to `trackRight.x` and stops there. Those are the
//     |x| > 2.4 failure bounds, so the boardwalk simply ENDS over open water — walk off
//     the last plank and the episode is over.
//   * the surveyor's cord is strung at `failArcRight.y` = L·cos(12°) = 0.978147. That
//     height is the tilt bound in disguise: |th| > 12° ⟺ the pole TIP drops below
//     L·cos(12°), for EVERY cart position, because the tip's height is L·cos(th) alone.
//     One horizontal line therefore states the whole angular failure condition exactly,
//     and it keeps stating it as the trolley travels. (STYLE.md honesty rule: "never hide
//     what the env defines — CartPole failure arcs — show them, subtly.")
//   * the gauge board behind the deck carves the same bound the other way: two incised
//     radial lines leaving the pivot at ±`atan2(failArc.x, failArc.y)` and the arc of
//     radius `poleTipRest.y` between them, so the two carved arcs pass exactly through
//     the two failArc anchors the debug overlay crosses. It is a SIGHTING instrument:
//     `gaugeStation` plants it on the eye→pivot ray at its own depth and scales it by the
//     distance ratio, so its drawn cone is the exact perspective image of the cone in the
//     pole's plane rather than a same-shape object a metre behind it.
//
// THE VERTICAL READ IS SMALL, AND SAYING SO IS PART OF THE JOB. The tip's height is
// L·cos(th), so the entire legal band moves it by L(1 − cos 12°) = 2.2% of the pole —
// 6.1 px in a 1600×900 hero frame. That is why the cord is exact to the pixel (see
// CORD_Z), why nothing green may cross it (actors.ts WHORL_RADIUS), and why the carved
// cone carries the readable statement of the same bound: the tip swings ±56 px sideways
// over that same band, so the ANGLE is what a viewer can actually see.
//
// The env's y = 0 is the POLE HINGE plane, not the ground (pivot = [0,0,0],
// poleTipRest = [0,1,0]). So this world hangs BELOW zero: the deck top at DECK_Y, the
// lake at WATER_Y, and the trolley's planter box rises to exactly y = 0, which is where
// the sapling leaves the soil. Nothing is offset, nothing is scaled: 1 env unit is 1
// world unit and the anchors land where the anchors are.
//
// LIGHT: noon, high and a little behind-right, so the deck is top-lit, the sapling is
// rim-lit against the water, and the shadows are short enough never to smear across the
// physics (STYLE.md canon lesson 8).

import * as THREE from 'three';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, setMaterialsTime, type OakMaterial } from '@kit/materials';
import {
  makeAmbientParticles,
  makeMistPlanes,
  type AmbientParticlesHandle,
  type MistPlanesHandle,
} from '@kit/particles';
import { streamFor } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import {
  excludeAny,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
} from '@kit/vegetation';
import { makeWater, type WaterHandle } from '@kit/water';
import { CP_POLE_LENGTH, CP_TH_LIMIT, CP_X_LIMIT } from '@sim/envs/cartpole';
import type { EnvGeometry } from '@sim/types';

// ---------------------------------------------------------------------------
// world constants (1 env unit === 1 world unit)
// ---------------------------------------------------------------------------

/** deck top surface — the trolley's wheels rest here, 0.36 below the hinge plane */
export const DECK_Y = -0.36;
/** plank thickness */
export const DECK_T = 0.052;
/** half-width of the walkway in z */
export const DECK_HALF_Z = 0.46;
/** lake surface */
export const WATER_Y = -0.86;
/** lake bed depth below the surface inside the open-water corridor */
export const LAKE_DEPTH = 0.72;
/**
 * Mean crest of the NEAR shore above the lake — deliberately below DECK_Y, so the
 * mid-ground of the frame is water and a low reed spit, never a green wall behind the
 * sapling.
 */
export const BANK_HEIGHT = 0.38;
/** the distant ridge the treeline stands on, and how far out it takes to climb */
export const FAR_RISE = 2.4;
export const FAR_ONSET = 9;
export const FAR_SPAN = 12;
/**
 * Open water reaches this far either side of the deck centre-line before the banks lift.
 * It is deliberately FAR: the far shore is the whole background of this frame, and at
 * z = −3 it sits only 1.6× the subject's distance from the eye, so bank grass renders at
 * the sapling's own scale and swallows it. Pushed to z ≈ −7 the shore is 2.5× out, reads
 * at 40% scale, and picks up enough fog to sit behind the physics.
 */
export const BANK_INNER_Z = 5.4;
/** …and this far past the deck ends, so the lake reads as open water at both ends */
export const BANK_INNER_X = 6.5;
export const BANK_FEATHER = 2.6;

/**
 * THE CORD'S DEPTH, and it is worth two paragraphs because the whole scene turns on it.
 *
 * The cord states the height of a point that lives at z = 0, so any depth offset is a
 * systematic misread: measured through the real hero camera, the original z = −0.34 drew
 * the line 3.4 px ABOVE the height it claims — against a band whose entire legal travel
 * is 6.1 px, so the drawn cord called the pole dead at 7.9° instead of 12°. At 30 mm off
 * the pole's own plane the residual is 0.3 px, and the drawn crossing lands at 11.67°.
 *
 * BEHIND, not in front, and that is an occlusion decision rather than a geometric one.
 * The line is 11 mm thick, which at this camera is 3 px — half the legal band — so a cord
 * drawn in FRONT clips the top of the terminal bud exactly when the bud is winning, and
 * the alive pose renders as a bud with its tip cut off (shot, measured, rejected). Behind
 * the pole, the whole bud is always visible: above the line while the episode lives,
 * below it the moment the env ends it. 30 mm clears the bud's 21 mm radius.
 */
export const CORD_Z = -0.03;
/** the stakes that hold it stay behind the deck — the trolley owns the cord's own plane */
export const STAKE_Z = -0.34;
/** the carved gauge board */
export const GAUGE_Z = -0.92;

// ---------------------------------------------------------------------------
// the camera (index.ts builds the rig from these; the gauge is sighted on them)
// ---------------------------------------------------------------------------

// THE FRAME. Near-profile, eye a little above the cord: the reads this scene has to
// deliver are "where on the deck" (horizontal) and "crown above or below the line"
// (vertical), and both die in a 3/4 view. At 16:9 the 40° vFOV puts x ∈ [−3.1, 3.1] and
// y ∈ [−1.4, 2.0] in frame, i.e. both deck ends, the whole sapling, the waterline and a
// slice of the far bank — with just enough camera offset to keep the piles reading as
// depth rather than as a flat elevation drawing.
export const CAM_BASE: readonly [number, number, number] = [0.44, 1.16, 4.62];
export const CAM_LOOK: readonly [number, number, number] = [0, 0.32, -0.06];
/** thumb pulls back and centres so the whole 4.8-unit deck survives a square-ish crop */
export const CAM_BASE_THUMB: readonly [number, number, number] = [0, 1.02, 5.6];
export const CAM_LOOK_THUMB: readonly [number, number, number] = [0, 0.32, 0];

/**
 * THE GAUGE'S SIGHTING STATION. A protractor whose apex sits at the pivot's own x/y but
 * a metre further back does NOT draw the pole's failure cone: measured through the hero
 * camera, its apex projected 19 px right and 48 px above the hinge, which put the pole
 * tip 10.5 px inside the cone on one side and 28.6 px outside it on the other — the
 * instrument called the same angle safe on the right and dead on the left.
 *
 * A sighting instrument is set up on a line of sight. Put the apex on the eye→pivot ray
 * at the gauge's own depth and scale it by the distance ratio, and the whole gauge
 * becomes the exact perspective image of the cone in the pole's plane: every point Q of
 * that plane maps to C + t(Q − C), which is `station + t·Q`. The carved angle is still
 * the anchors' own ±atan2(failArc.x, failArc.y) — this only decides where the instrument
 * STANDS, the way a real one is planted so its scale lies over the thing it measures.
 */
export function gaugeStation(
  camera: readonly [number, number, number],
  z: number,
): { x: number; y: number; z: number; scale: number } {
  const cz = camera[2];
  // degenerate only if the gauge plane contains the eye; keep the literal station then
  if (!(Math.abs(cz - z) > 1e-6) || !(Math.abs(cz) > 1e-6)) return { x: 0, y: 0, z, scale: 1 };
  const t = (cz - z) / cz;
  return { x: camera[0] * (1 - t), y: camera[1] * (1 - t), z, scale: t };
}

export const TERRAIN_W = 34;
// Deep on purpose: the far edge of the ground must never cut a hard line across the
// sky. It sits at z = −22 where fog is ~47% AND where the ridge has climbed a metre
// above the eye, so the visible horizon is the ridge, not the mesh boundary.
export const TERRAIN_D = 44;
export const TERRAIN_SEG_X = 180;
export const TERRAIN_SEG_Z = 190;

/**
 * Fog reach, derived from the terrain footprint. The multiplier is large because noon
 * is the CLEAREST palette in the preserve (fogDensityHint 0.25 — half of dusk's) and
 * this diorama is only ~6.5 units across: at the palette's own hint over a bare
 * terrain-width extent the subject itself sits under 33% haze and the whole frame
 * whites out. Six terrain widths puts the sapling at ~3% and the far shore at ~50%.
 */
export const FOG_EXTENT = TERRAIN_W * 6.6;

/**
 * 44°, hard right, slightly in FRONT of the deck. Three constraints meet here:
 *  * below ~25° the sapling throws a 2-unit shadow straight down the boardwalk and
 *    swallows the physics (canon lesson 8);
 *  * above ~55° it throws almost none and the deck reads as a flat green ribbon;
 *  * behind the deck (negative z) every camera-facing surface — the trolley's front,
 *    the gauge, the planter — falls to hemisphere fill and the hero prop measures a
 *    muddy #3c5340.
 * At 44° from the front-right the sapling's shadow is 1.0 long and lands across the
 * planks to its left, the trolley keeps a lit face, and the crown still catches a rake.
 */
export const SUN_DIR: readonly [number, number, number] = [0.6, 0.7, 0.39];

// ---------------------------------------------------------------------------
// small shared helpers (actors.ts imports these too)
// ---------------------------------------------------------------------------

export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

export function clampRange(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** A scene-local color: STYLE.md allows only BLENDS of token colors. */
export function blendTokens(
  a: THREE.ColorRepresentation,
  b: THREE.ColorRepresentation,
  k: number,
): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), k);
}

/** `noon.sun` is the token white; `night.skyTop` the token near-black (canon lesson 9). */
export const TOKEN_WHITE = TOKENS.palettes.noon.sun;
export const TOKEN_BLACK = TOKENS.palettes.night.skyTop;

/**
 * KIT GAP WORKAROUND (filed in concerns). `kit/water` does not ship the colours it is
 * given: it bakes a fixed pull toward the palette's pale ends into its own uniforms —
 * `shallow.lerp(palette.groundB, 0.35)` and `deep.lerp(palette.skyBottom, 0.55)`. Under
 * NOON, whose groundB (0xd6e394) and skyBottom (0xeef7dc) are both near-white, that
 * turns any lake into a sheet of milk: measured #c2ccc8, brighter than the sky itself.
 *
 * So solve for the pre-image instead — the colour that, after the kit's pull, IS the
 * colour this scene wants. Still a linear combination of token colours (STYLE.md allows
 * blends; this is one with a coefficient outside [0, 1]), and the shader only ever sees
 * the corrected result. Components are computed in the linear working space, which is
 * where THREE.Color stores them and where `lerp` operates.
 */
export function unmixTokens(target: THREE.Color, pull: THREE.ColorRepresentation, k: number): THREE.Color {
  const p = new THREE.Color(pull);
  const inv = 1 / (1 - k);
  return new THREE.Color().setRGB(
    (target.r - k * p.r) * inv,
    (target.g - k * p.g) * inv,
    (target.b - k * p.b) * inv,
  );
}

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

/**
 * Merge pre-transformed parts into ONE geometry carrying a flat per-part vertex color,
 * disposing the sources — so a whole boardwalk (38 planks, 14 piles, 4 stakes) is one
 * draw call with per-plank wood tones.
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
// the env's own numbers, read from the anchors (never re-derived)
// ---------------------------------------------------------------------------

function anchor(geometry: EnvGeometry, name: string): readonly [number, number, number] | null {
  const a = geometry.classic?.anchors[name];
  if (!a) return null;
  return Number.isFinite(a[0]) && Number.isFinite(a[1]) && Number.isFinite(a[2]) ? a : null;
}

/** |x| failure bound — and therefore the exact half-length of the boardwalk. */
export function trackLimit(geometry: EnvGeometry): number {
  const a = anchor(geometry, 'trackRight');
  return a && a[0] > 0 ? a[0] : CP_X_LIMIT;
}

/** Drawn pole length (the `poleTipRest` anchor is the tip when upright). */
export function poleLength(geometry: EnvGeometry): number {
  const a = anchor(geometry, 'poleTipRest');
  return a && a[1] > 0 ? a[1] : CP_POLE_LENGTH;
}

/**
 * THE CORD HEIGHT. `failArcRight` is the tip at the +12° bound, so its y is
 * L·cos(θ_limit) — and the tip's height is L·cos(th) for every cart position, so
 * "tip below this line" IS "|th| beyond the bound", exactly, everywhere.
 */
export function tiltLineY(geometry: EnvGeometry): number {
  const a = anchor(geometry, 'failArcRight');
  return a && a[1] > 0 ? a[1] : CP_POLE_LENGTH * Math.cos(CP_TH_LIMIT);
}

/** Half-angle of the failure cone, recovered from the same anchor. */
export function tiltLimit(geometry: EnvGeometry): number {
  const a = anchor(geometry, 'failArcRight');
  return a && a[1] > 0 ? Math.atan2(a[0], a[1]) : CP_TH_LIMIT;
}

// ---------------------------------------------------------------------------
// the lake bed / bank height field
// ---------------------------------------------------------------------------

/**
 * Flat lake bed inside the corridor the boardwalk crosses, rising to grassy banks
 * outside it. The weight is identically 0 over the whole open-water corridor, so the
 * bed never breaks the surface anywhere the trolley can be seen against it, and the
 * noise that makes the banks read as land is multiplied by that same weight.
 */
export function makeShoreHeight(seed: number): (x: number, z: number) => number {
  const bed = WATER_Y - LAKE_DEPTH;
  return (x, z) => {
    const outZ = Math.abs(z) - BANK_INNER_Z;
    const outX = Math.abs(x) - BANK_INNER_X;
    const bank = Math.max(
      smoothstep01(outZ / BANK_FEATHER),
      smoothstep01(outX / BANK_FEATHER),
    );
    if (bank <= 0) return bed;
    // the far shore keeps climbing into a low hill; the near shore stays a flat spit
    const far = smoothstep01((-z - FAR_ONSET) / FAR_SPAN);
    const n1 = valueNoise2(x * 0.34 + 5.7, z * 0.34 + 2.3, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 1.05 + 13.1, z * 1.05 + 8.9, 64, seed ^ 0x5f3a91) - 0.5;
    const rise = LAKE_DEPTH + BANK_HEIGHT + far * FAR_RISE;
    return bed + bank * (rise + n1 * (0.34 + 0.7 * far) + n2 * (0.14 + 0.28 * far));
  };
}

// ---------------------------------------------------------------------------
// world handle
// ---------------------------------------------------------------------------

export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  water: WaterHandle;
  terrain: TerrainHandle;
  /** deck end at x = −trackLimit (a probe object: position only) */
  endLeftRoot: THREE.Object3D;
  /** deck end at x = +trackLimit */
  endRightRoot: THREE.Object3D;
  /** the carved failure-cone gauge, rooted on the pivot's x/y at its own z */
  gaugeRoot: THREE.Object3D;
  /** ctx.time() — pause freezes wind, water, mist and pollen together */
  setTime(t: number): void;
  /** signed pole lean in [-1, 1]: the gust that pushes the sapling combs the grass */
  setWind(lean: number): void;
  /** 0 = the balanced warm noon, 1 = the moment after a fall */
  setCool(env01: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

interface Disposable {
  dispose(): void;
}

interface Timed {
  setTime(t: number): void;
}

/** base grass sway; setWind rides on top of it */
const GRASS_WIND = 0.075;
const REED_WIND = 0.05;
const TREE_WIND = 0.028;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const tier: QualityTier = ctx.tier;
  const palette: Palette = ctx.tokens.palettes.noon;
  const accent = ctx.tokens.accents['cart-pole'] ?? ctx.tokens.celebrationGold;
  const xLimit = trackLimit(ctx.geometry);
  const cordY = tiltLineY(ctx.geometry);
  const coneHalf = tiltLimit(ctx.geometry);
  const armLength = poleLength(ctx.geometry);
  const group = new THREE.Group();
  group.name = 'boardwalkWorld';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const shadows = tier !== 'thumb';

  // ---- palette-derived surface colors (blends of tokens only) ----
  // Canon lesson 19 (the noon albedo law): under the 3.0 noon key, ground albedo stays
  // in the 140–170 sRGB band. Every wood tone below is blended toward night.skyTop to
  // hold it there — raw groundB at noon crosses the bloom threshold and smears.
  const woodPale = blendTokens(palette.groundB, TOKEN_BLACK, 0.34);
  const woodMid = blendTokens(palette.groundA, TOKEN_BLACK, 0.24);
  const woodDark = blendTokens(palette.groundA, TOKEN_BLACK, 0.52);
  const woodWet = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.42);
  const cordColor = blendTokens(TOKEN_WHITE, palette.hemiSky, 0.34);
  // the incised lines have to survive a 3-step toon ramp on a lit board: only a
  // near-black carries that far (canon lesson 9 — night.skyTop is the token near-black)
  const carve = blendTokens(TOKEN_BLACK, palette.groundA, 0.14);
  const bankGreen = blendTokens(palette.groundA, TOKEN_BLACK, 0.2);
  const meadowGreen = blendTokens(palette.groundB, palette.groundA, 0.6);
  const reedGreen = blendTokens(blendTokens(palette.groundA, palette.groundB, 0.22), TOKEN_BLACK, 0.16);
  const stone = blendTokens(palette.hemiGround, palette.skyTop, 0.34);
  /** bleached, salt-dried timber — the gauge has to read against a green bank */
  const driftwood = blendTokens(TOKEN_WHITE, palette.groundA, 0.56);
  const shallowMud = blendTokens(palette.hemiGround, palette.groundA, 0.4);
  /** what the lake should actually LOOK like (see unmixTokens) */
  const lakeShallow = blendTokens(blendTokens(palette.groundA, palette.skyTop, 0.35), TOKEN_BLACK, 0.42);
  const lakeDeep = blendTokens(TOKEN_BLACK, palette.skyTop, 0.14);

  // ---- sky (exactly one) ----
  const sky = skyFromPalette(palette, 'noon', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.038,
    sunHdr: 6.5,
    sunGlow: 0.3,
    sunGlowPower: 28,
    topSpread: 0.72,
    bottomSpread: 0.34,
    radius: 320,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the one light budget ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 2,
    pointDistance: 5,
    sunDistanceScale: 2.6,
  });
  group.add(rig.group);
  owned.push(rig);
  // The focus must COVER every shadow RECEIVER on screen, not just the deck: outside the
  // ortho frustum three clamps the lookup to the border texel and paints amoebas over the
  // banks (canon lesson 8). 2048 over 11 units is still 5 mm per texel — the trolley is
  // 0.4 wide, i.e. 74 texels.
  rig.setShadowFocus({ x: 0, y: -0.3, z: -1.4 }, 5.5);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the lake bed and its banks ----
  const shore = makeShoreHeight(ctx.seed);
  const rampColor = new THREE.Color();
  const terrain = makeTerrain({
    size: [TERRAIN_W, TERRAIN_D],
    segments: [TERRAIN_SEG_X, TERRAIN_SEG_Z],
    tier,
    seed: ctx.seed,
    heightFn: shore,
    jitter: 0,
    colorNoise: 0.05,
    slopeColor: stone,
    slopeStrength: 0.5,
    slopeSharpness: 2.4,
    colorRamp: (h) => {
      // deep bed -> wet mud at the waterline -> bank grass -> pale ridge crest.
      // The submerged half is pulled hard toward the token near-black: the water is
      // 86% opaque, so a noon-lit bed shows straight through and the lake reads as a
      // sheet of glass over a lawn.
      const depth = smoothstep01((WATER_Y - h) / 0.66);
      const grass = smoothstep01((h - (WATER_Y + 0.02)) / 0.34);
      const crest = smoothstep01((h - (WATER_Y + 0.9)) / 1.5);
      rampColor.copy(new THREE.Color(shallowMud));
      rampColor.lerp(new THREE.Color(bankGreen), grass);
      rampColor.lerp(new THREE.Color(meadowGreen), crest * 0.55);
      rampColor.lerp(new THREE.Color(TOKEN_BLACK), depth * 0.6);
      return rampColor;
    },
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.18,
      toonSteps: 3,
      toonSoftness: 0.3,
      toonFloor: 0.24,
    },
    receiveShadow: shadows,
    castShadow: false,
    name: 'lakeBed',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  // ---- the lake ----
  const water = makeWater({
    size: [TERRAIN_W - 2, TERRAIN_D - 2],
    palette,
    tier,
    segments: 108,
    level: WATER_Y,
    amplitude: 0.016,
    waveScale: 1.15,
    speed: 0.85,
    // this camera looks along the water at ~8°, where a physical fresnel term turns the
    // whole lake into a mirror of the pale noon sky; 0.28 keeps the rim on the wave
    // crests without bleaching the body of the water
    fresnel: 0.19,
    caustics: 0.26,
    depthFalloff: 5,
    opacity: 0.95,
    // the water is an UNLIT ShaderMaterial, so its uniforms are its final colour: pale
    // token blends read as ice under the noon grade. Shallow is a green-tinged sky,
    // deep is most of the way to the night sky-top.
    shallow: unmixTokens(lakeShallow, palette.groundB, 0.35),
    deep: unmixTokens(lakeDeep, palette.skyBottom, 0.55),
    sunDir: sky.sunDirection,
    name: 'boardwalkLake',
  });
  group.add(water.mesh);
  owned.push(water);
  timed.push(water);

  // ---- the boardwalk: EXACTLY [-xLimit, +xLimit] ----
  const deckParts: ColoredPart[] = [];
  const PLANK_PITCH = 0.126;
  const PLANK_GAP = 0.019;
  const plankCount = Math.max(2, Math.round((2 * xLimit) / PLANK_PITCH));
  const plankPitch = (2 * xLimit) / plankCount;
  const plankWidth = plankPitch - PLANK_GAP;
  {
    const rng = streamFor(ctx.seed, 'planks');
    for (let i = 0; i < plankCount; i++) {
      const cx = -xLimit + (i + 0.5) * plankPitch;
      const plank = new THREE.BoxGeometry(plankWidth, DECK_T, DECK_HALF_Z * 2);
      plank.translate(cx, DECK_Y - DECK_T * 0.5, 0);
      const k = rng();
      deckParts.push({
        geometry: plank,
        color: k < 0.34 ? woodPale : k < 0.72 ? woodMid : woodDark,
      });
    }
    // two stringers under the planks
    for (const z of [-DECK_HALF_Z + 0.12, DECK_HALF_Z - 0.12]) {
      const beam = new THREE.BoxGeometry(2 * xLimit, 0.1, 0.07);
      beam.translate(0, DECK_Y - DECK_T - 0.05, z);
      deckParts.push({ geometry: beam, color: woodDark });
    }
    // piles driven into the lake bed
    const pileCount = 7;
    for (let i = 0; i < pileCount; i++) {
      const px = -xLimit + (i * 2 * xLimit) / (pileCount - 1);
      for (const z of [-DECK_HALF_Z + 0.12, DECK_HALF_Z - 0.12]) {
        const h = DECK_Y - DECK_T - 0.1 - (WATER_Y - 0.5);
        const pile = new THREE.CylinderGeometry(0.045, 0.055, h, 6);
        pile.translate(px, WATER_Y - 0.5 + h * 0.5, z);
        deckParts.push({ geometry: pile, color: woodWet });
      }
    }
  }
  const deckGeometry = mergeColoredParts(deckParts, 'boardwalkDeck');
  const timberMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.88,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.24,
    tier,
    name: 'boardwalkTimber',
  });
  const deckMesh = new THREE.Mesh(deckGeometry, timberMaterial);
  deckMesh.name = 'boardwalkDeck';
  deckMesh.castShadow = shadows;
  deckMesh.receiveShadow = shadows;
  group.add(deckMesh);
  owned.push({
    dispose: () => {
      deckGeometry.dispose();
      timberMaterial.dispose();
    },
  });

  // ---- the two deck ENDS: the |x| failure bounds, made of wood ----
  // Each root carries POSITION ONLY and sits on the anchor; the end board and its two
  // corner posts are children, so deleting every decoration leaves the bound exact.
  const endRoots: THREE.Object3D[] = [];
  for (const sign of [-1, 1]) {
    const root = new THREE.Object3D();
    root.name = sign < 0 ? 'deckEndLeftRoot' : 'deckEndRightRoot';
    root.position.set(sign * xLimit, DECK_Y, 0);
    group.add(root);
    endRoots.push(root);

    const parts: ColoredPart[] = [];
    // the end board: a dark kerb capping the last plank, flush with the bound
    const board = new THREE.BoxGeometry(0.05, 0.075, DECK_HALF_Z * 2);
    board.translate(-sign * 0.025, -0.014, 0);
    parts.push({ geometry: board, color: woodDark });
    for (const z of [-DECK_HALF_Z + 0.075, DECK_HALF_Z - 0.075]) {
      const post = new THREE.CylinderGeometry(0.037, 0.045, 0.4, 7);
      post.translate(-sign * 0.045, 0.2, z);
      parts.push({ geometry: post, color: woodMid });
      const cap = new THREE.ConeGeometry(0.05, 0.07, 7);
      cap.translate(-sign * 0.045, 0.435, z);
      parts.push({ geometry: cap, color: woodPale });
      // a mooring ring hint: a dark band
      const band = new THREE.CylinderGeometry(0.048, 0.048, 0.024, 7);
      band.translate(-sign * 0.045, 0.315, z);
      parts.push({ geometry: band, color: woodWet });
    }
    const geo = mergeColoredParts(parts, root.name);
    const mesh = new THREE.Mesh(geo, timberMaterial);
    mesh.name = sign < 0 ? 'deckEndLeft' : 'deckEndRight';
    mesh.castShadow = shadows;
    mesh.receiveShadow = shadows;
    root.add(mesh);
    owned.push({ dispose: () => geo.dispose() });
  }

  // ---- THE SURVEYOR'S CORD: the tilt bound, strung at exactly cos(θlimit) ----
  const cordParts: ColoredPart[] = [];
  {
    // 8 mm, not 11: the line's own drawn thickness competes with the band it measures.
    // At this camera 11 mm is 3 px against a 6.1 px legal travel — the cord was eating
    // half its own read. 8 mm is 2.1 px, still an opaque box rather than a hairline.
    const span = xLimit + 0.06;
    const line = new THREE.BoxGeometry(span * 2, 0.008, 0.008);
    line.translate(0, cordY, CORD_Z);
    cordParts.push({ geometry: line, color: cordColor });
    // Four slim stakes, kept off the trolley's usual ground (|x| < 1.2). They stay in
    // their own plane behind the deck — a stake on the LINE's z would be in the trolley's
    // path — and each reaches the line on a short outrigger, which is how a survey line
    // gets held out over the thing it measures.
    for (const sx of [-1, 1]) {
      for (const sxAbs of [1.92, xLimit + 0.055]) {
        const px = sx * sxAbs;
        const h = cordY + 0.07 - DECK_Y;
        const stake = new THREE.CylinderGeometry(0.013, 0.018, h, 6);
        stake.translate(px, DECK_Y + h * 0.5, STAKE_Z);
        cordParts.push({ geometry: stake, color: woodPale });
        const tip = new THREE.ConeGeometry(0.022, 0.045, 6);
        tip.translate(px, DECK_Y + h + 0.018, STAKE_Z);
        cordParts.push({ geometry: tip, color: cordColor });
        const reach = CORD_Z - STAKE_Z;
        const arm = new THREE.BoxGeometry(0.012, 0.012, reach);
        arm.translate(px, cordY, STAKE_Z + reach * 0.5);
        cordParts.push({ geometry: arm, color: cordColor });
      }
    }
  }
  const cordGeometry = mergeColoredParts(cordParts, 'tiltCord');
  const cordMesh = new THREE.Mesh(cordGeometry, timberMaterial);
  cordMesh.name = 'tiltCord';
  // sub-texel cord + 16 mm stakes: casting them paints dashed streaks (canon lesson 8)
  cordMesh.castShadow = false;
  group.add(cordMesh);
  owned.push({ dispose: () => cordGeometry.dispose() });

  // ---- the carved failure-cone gauge, sighted on the pivot (see gaugeStation) ----
  const gaugeRoot = new THREE.Object3D();
  gaugeRoot.name = 'tiltGaugeRoot';
  const station = gaugeStation(tier === 'thumb' ? CAM_BASE_THUMB : CAM_BASE, GAUGE_Z);
  gaugeRoot.position.set(station.x, station.y, station.z);
  gaugeRoot.scale.setScalar(station.scale);
  group.add(gaugeRoot);
  {
    const parts: ColoredPart[] = [];
    // An OPEN protractor: the two incised radial lines ARE the object, held on one slim
    // upright. A backing board of any size sits directly behind the sapling and becomes
    // the subject of the frame — measured, it out-read the hero prop at every width
    // tried. Battens only, so the reeds and the water show through the failure cone.
    const FACE_Z = 0;
    // The stand stops AT the pivot. A full-height upright behind a leaning sapling reads
    // as a second pole — measured on a crop, it was the first thing the eye called "the
    // pole" — so the cone's apex is carried on a short post and braced by one chord.
    // The gauge is drawn in its SIGHTING frame (offset and scaled by `station`), so the
    // foot has to be authored back through it or the instrument floats: this is the local
    // y whose world height is the deck's own.
    const postTop = 0.06;
    const postBottom = (DECK_Y - 0.09 - station.y) / station.scale;
    const post = new THREE.CylinderGeometry(0.021, 0.032, postTop - postBottom, 6);
    post.translate(0, (postTop + postBottom) * 0.5, -0.05);
    parts.push({ geometry: post, color: woodMid });
    const foot = new THREE.BoxGeometry(0.2, 0.035, 0.11);
    foot.translate(0, postBottom + 0.02, -0.05);
    parts.push({ geometry: foot, color: woodDark });
    // The two radial battens, leaving the pivot at exactly ±coneHalf. Each is drawn
    // TWO-TONE — a pale bar over a slightly larger dark one — because the background
    // behind this gauge is a busy mid-value reed bank: a single pale batten dissolves
    // into it and a single dark batten reads as a stray twig among the reed shadows.
    // The pale/dark pair holds an edge against any background value.
    for (const sx of [-1, 1]) {
      const len = armLength * 1.05;
      for (const [w, dz, col] of [
        [0.036, -0.014, carve],
        [0.022, 0.004, driftwood],
      ] as const) {
        const arm = new THREE.BoxGeometry(w, len, w);
        arm.translate(0, len * 0.5, 0);
        arm.rotateZ(-sx * coneHalf);
        arm.translate(0, 0, FACE_Z + dz);
        parts.push({ geometry: arm, color: col });
      }
    }
    // the arc of radius `armLength` between them: the tip's failure locus at x = 0
    const ARC_SEGMENTS = 9;
    for (let i = 0; i < ARC_SEGMENTS; i++) {
      const a0 = -coneHalf + (2 * coneHalf * i) / ARC_SEGMENTS;
      const a1 = -coneHalf + (2 * coneHalf * (i + 1)) / ARC_SEGMENTS;
      const x0 = armLength * Math.sin(a0);
      const y0 = armLength * Math.cos(a0);
      const x1 = armLength * Math.sin(a1);
      const y1 = armLength * Math.cos(a1);
      const dx = x1 - x0;
      const dy = y1 - y0;
      const chord = Math.hypot(dx, dy) * 1.06;
      const ang = Math.atan2(dy, dx);
      for (const [w, dz, col] of [
        [0.03, -0.014, carve],
        [0.018, 0.004, driftwood],
      ] as const) {
        const seg = new THREE.BoxGeometry(chord, w, w);
        seg.rotateZ(ang);
        seg.translate((x0 + x1) * 0.5, (y0 + y1) * 0.5, FACE_Z + dz);
        parts.push({ geometry: seg, color: col });
      }
    }
    // one chord bracing the cone at mid-height — the brace that makes it an instrument
    {
      const r = armLength * 0.52;
      const halfSpan = r * Math.sin(coneHalf);
      const tie = new THREE.BoxGeometry(halfSpan * 2, 0.014, 0.014);
      tie.translate(0, r * Math.cos(coneHalf), FACE_Z);
      parts.push({ geometry: tie, color: driftwood });
    }
    // two ticks marking the failArc anchors themselves
    for (const sx of [-1, 1]) {
      const tick = new THREE.BoxGeometry(0.07, 0.026, 0.026);
      tick.rotateZ(-sx * coneHalf);
      tick.translate(sx * armLength * Math.sin(coneHalf), armLength * Math.cos(coneHalf), FACE_Z);
      parts.push({ geometry: tick, color: carve });
    }
    const geo = mergeColoredParts(parts, 'tiltGauge');
    const mesh = new THREE.Mesh(geo, timberMaterial);
    mesh.name = 'tiltGauge';
    mesh.castShadow = false;
    mesh.receiveShadow = shadows;
    gaugeRoot.add(mesh);
    owned.push({ dispose: () => geo.dispose() });
  }

  // ---- mooring posts out in the water (depth cues, never on the deck line) ----
  {
    const parts: ColoredPart[] = [];
    const rng = streamFor(ctx.seed, 'moorings');
    for (let i = 0; i < 7; i++) {
      const px = -5.2 + rng() * 10.4;
      const pz = -1.35 - rng() * 3.4;
      if (Math.abs(pz) < 0.95) continue;
      const h = 0.6 + rng() * 0.55;
      const post = new THREE.CylinderGeometry(0.035, 0.05, h, 6);
      post.translate(px, WATER_Y - 0.2 + h * 0.5, pz);
      parts.push({ geometry: post, color: woodWet });
    }
    if (parts.length > 0) {
      const geo = mergeColoredParts(parts, 'mooringPosts');
      const mesh = new THREE.Mesh(geo, timberMaterial);
      mesh.name = 'mooringPosts';
      mesh.castShadow = false;
      group.add(mesh);
      owned.push({ dispose: () => geo.dispose() });
    }
  }

  // ---- lily pads: one instanced disc field on the lake ----
  const padGeometry = new THREE.CircleGeometry(0.16, tier === 'thumb' ? 6 : 9, 0.35, Math.PI * 2 - 0.7);
  padGeometry.rotateX(-Math.PI / 2);
  const padMaterial = oakMaterial({
    color: blendTokens(palette.groundA, TOKEN_BLACK, 0.2),
    flatShading: true,
    roughness: 0.72,
    side: THREE.DoubleSide,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSteps: 2,
    toonSoftness: 0.35,
    toonFloor: 0.3,
    tier,
    name: 'lilyPadMaterial',
  });
  const PAD_COUNT = tier === 'thumb' ? 14 : tier === 'medium' ? 44 : 96;
  const pads = new THREE.InstancedMesh(padGeometry, padMaterial, PAD_COUNT);
  pads.name = 'lilyPads';
  pads.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(PAD_COUNT * 3), 3);
  {
    const rng = streamFor(ctx.seed, 'lilies');
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    const s = new THREE.Vector3();
    const col = new THREE.Color();
    let n = 0;
    for (let i = 0; i < PAD_COUNT * 6 && n < PAD_COUNT; i++) {
      const px = -7 + rng() * 14;
      const pz = -6.5 + rng() * 9;
      const scale = 0.6 + rng() * 0.75;
      const yaw = rng() * Math.PI * 2;
      const k = 0.86 + rng() * 0.3;
      // never under the walkway, never on dry land
      if (Math.abs(pz) < DECK_HALF_Z + 0.5 && Math.abs(px) < xLimit + 0.5) continue;
      if (shore(px, pz) > WATER_Y - 0.12) continue;
      p.set(px, WATER_Y + 0.006, pz);
      e.set(0, yaw, 0);
      q.setFromEuler(e);
      s.set(scale, 1, scale);
      m.compose(p, q, s);
      pads.setMatrixAt(n, m);
      col.setRGB(k, k * 1.03, k * 0.93);
      pads.setColorAt(n, col);
      n++;
    }
    pads.count = n;
    pads.instanceMatrix.needsUpdate = true;
    if (pads.instanceColor) pads.instanceColor.needsUpdate = true;
    pads.computeBoundingSphere();
  }
  group.add(pads);
  owned.push({
    dispose: () => {
      pads.dispose();
      padGeometry.dispose();
      padMaterial.dispose();
    },
  });

  // ---- vegetation: on the banks, never on the water, never over the physics ----
  const sampleAt = (x: number, z: number): number => terrain.sampleHeight(x, z);
  const underWater: ExcludeFn = (x, z) => terrain.sampleHeight(x, z) < WATER_Y + 0.05;
  /** the whole crossing corridor: deck, cord, gauge and the water they stand in */
  const overTheCrossing: ExcludeFn = (x, z) =>
    Math.abs(z) < BANK_INNER_Z - 0.35 && Math.abs(x) < xLimit + 1.6;
  /** the camera cone: nothing may stand between the eye and the boardwalk */
  const inFront: ExcludeFn = (x, z) => z > 2.0 && Math.abs(x) < 1.25 + (z - 2.0) * 0.62;

  const grass = scatterGrass({
    count: 3400,
    area: { x: 0, z: -9, halfX: 12, halfZ: 3.6 },
    seed: ctx.seed ^ 0x7f19,
    tier,
    exclude: excludeAny(underWater, overTheCrossing, inFront),
    heightAt: sampleAt,
    maxSlope: 0.8,
    // chest-high on the banks: grassClumpGeometry is 0.45 tall, so 1.3–2.4× reads as
    // 0.6–1.1 units — well over the 0.24-tall trolley it frames
    color: blendTokens(blendTokens(palette.groundA, palette.groundB, 0.34), TOKEN_BLACK, 0.12),
    scale: [0.95, 1.6],
    stretch: [0.9, 1.3],
    hueJitter: 0.22,
    wind: GRASS_WIND,
    windFrequency: 1.6,
    rimColor: palette.skyHorizon,
    rimStrength: 0.4,
    name: 'bankGrass',
  });
  group.add(grass.group);
  owned.push(grass);
  timed.push(grass);

  const reeds = scatterGrass({
    count: 620,
    area: { x: 0, z: -7.4, halfX: 11, halfZ: 2.6 },
    seed: ctx.seed ^ 0x3d51,
    tier,
    // reeds like the waterline: only the narrow strip either side of it
    exclude: excludeAny(
      overTheCrossing,
      inFront,
      (x, z) => {
        const h = terrain.sampleHeight(x, z);
        return h < WATER_Y - 0.08 || h > WATER_Y + 0.18;
      },
    ),
    heightAt: sampleAt,
    color: reedGreen,
    scale: [0.85, 1.45],
    stretch: [1.0, 1.4],
    hueJitter: 0.26,
    wind: REED_WIND,
    windFrequency: 1.15,
    rimColor: palette.skyHorizon,
    rimStrength: 0.5,
    name: 'lakeReeds',
  });
  group.add(reeds.group);
  owned.push(reeds);
  timed.push(reeds);

  // A foreground fringe that clips the bottom CORNERS and nothing else. The exclusion
  // is the camera cone widened by the near-plane fan: at z = 2.2 a reed inside
  // |x| < 1.4 is dead centre of frame and 2.4 units from the eye, i.e. a wall.
  const fringe = scatterGrass({
    count: 150,
    area: { x: 0, z: 2.15, halfX: 3.1, halfZ: 0.55 },
    seed: ctx.seed ^ 0x9ab3,
    tier,
    exclude: (x, z) => Math.abs(x) < 1.3 + (z - 1.6) * 0.7,
    heightAt: () => WATER_Y - 0.05,
    color: blendTokens(reedGreen, TOKEN_BLACK, 0.3),
    scale: [1.1, 1.8],
    stretch: [1.05, 1.5],
    hueJitter: 0.2,
    wind: REED_WIND * 1.4,
    windFrequency: 1.05,
    rimColor: palette.skyHorizon,
    rimStrength: 0.55,
    name: 'foregroundReeds',
  });
  group.add(fringe.group);
  owned.push(fringe);
  timed.push(fringe);

  const flowers = scatterFlowers({
    count: 340,
    area: { x: 0, z: -8.8, halfX: 11, halfZ: 3 },
    seed: ctx.seed ^ 0x2c71,
    tier,
    exclude: excludeAny(underWater, overTheCrossing, inFront),
    heightAt: sampleAt,
    maxSlope: 0.7,
    color: blendTokens(accent, TOKEN_WHITE, 0.42),
    scale: [0.8, 1.5],
    hueJitter: 0.3,
    wind: 0.05,
    windFrequency: 1.9,
    rimColor: palette.skyHorizon,
    rimStrength: 0.4,
    name: 'bankFlowers',
  });
  group.add(flowers.group);
  owned.push(flowers);
  timed.push(flowers);

  const rocks = scatterRocks({
    count: 220,
    area: { x: 0, z: -7.2, halfX: 11, halfZ: 2.4 },
    seed: ctx.seed ^ 0x2b17,
    tier,
    exclude: excludeAny(overTheCrossing, inFront, (x, z) => {
      const h = terrain.sampleHeight(x, z);
      return h < WATER_Y - 0.3 || h > WATER_Y + 0.7;
    }),
    heightAt: sampleAt,
    color: stone,
    scale: [0.14, 0.42],
    stretch: [0.5, 0.95],
    hueJitter: 0.24,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    // See farShoreTrees below for the shadow-budget rule. These sit 10–11 units out on
    // the far bank at 0.14–0.42 scale, behind chest-high grass and ~40% fog. Unlike the
    // trees, a few of them DO fall inside the shadow frustum — and it buys 53 pixels:
    // toggling this flag moves 53 of 1,440,000 pixels in a 1600×900 hero frame, all of
    // them on the far bank, for 17,600 triangles and 2 draws (8.5% of the hero total).
    castShadow: false,
    name: 'shoreStones',
  });
  group.add(rocks.group);
  owned.push(rocks);
  timed.push(rocks);

  const trees = scatterTrees({
    archetypes: ['broadleaf', 'conifer'],
    count: 120,
    area: { x: 0, z: -16, halfX: 15, halfZ: 6 },
    seed: ctx.seed ^ 0x5133,
    tier,
    exclude: excludeAny(underWater, (x, z) => z > -10.5),
    heightAt: sampleAt,
    maxSlope: 0.65,
    color: blendTokens(palette.groundA, TOKEN_BLACK, 0.3),
    scale: [0.5, 1.22],
    stretch: [0.9, 1.4],
    hueJitter: 0.24,
    wind: TREE_WIND,
    windFrequency: 0.85,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    // A CASTER THAT CANNOT REACH THE SHADOW MAP IS PURE COST. The rig's focus is a
    // radius-5.5 ortho box on (0, −0.3, −1.4); these stand at z ≤ −10.5, which puts them
    // 14 units off the light's axis — outside the frustum by 2.5×, so the shadow pass was
    // rasterizing 27,024 triangles (and 2 draw calls, 12% of the hero total) that could
    // never appear. Verified by pixel diff at 1600×900: toggling this flag changes NOTHING
    // in the 3D frame — the only pixels that move are the HUD's own draw/triangle counters.
    castShadow: false,
    name: 'farShoreTrees',
  });
  group.add(trees.group);
  owned.push(trees);
  timed.push(trees);

  // ---- atmosphere ----
  // NOTE (canon lesson 5): kit particle/mist sizes are tuned for a ~20-unit field. This
  // frame is ~6.5 units across with the camera 4.7 out, so every `size` below is the
  // preset scaled by roughly 1/7.
  const pollen: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'pollen',
    count: 190,
    bounds: { x: 0, y: 0.42, z: -1.1, halfX: 4.2, halfY: 1.05, halfZ: 3 },
    seed: ctx.seed,
    tier,
    color: blendTokens(palette.sun, accent, 0.4),
    size: 0.5,
    intensity: 0.85,
    opacity: 0.3,
    rise: 0.05,
    swirl: 0.12,
    name: 'lakePollen',
  });
  group.add(pollen.points);
  owned.push(pollen);
  timed.push(pollen);

  const spray: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'dust',
    count: 130,
    bounds: { x: 0, y: WATER_Y + 0.22, z: -3.2, halfX: 7, halfY: 0.3, halfZ: 3.4 },
    seed: ctx.seed ^ 0x51ab,
    tier,
    color: blendTokens(TOKEN_WHITE, palette.hemiSky, 0.3),
    size: 0.4,
    intensity: 0.9,
    opacity: 0.13,
    rise: 0.03,
    swirl: 0.18,
    name: 'lakeSpray',
  });
  group.add(spray.points);
  owned.push(spray);
  timed.push(spray);

  const haze: MistPlanesHandle = makeMistPlanes({
    size: [30, 14],
    color: blendTokens(palette.fog, palette.hemiSky, 0.45),
    seed: ctx.seed,
    tier,
    layers: 4,
    baseY: WATER_Y + 0.16,
    spread: 0.85,
    opacity: 0.055,
    speed: 0.008,
    scale: 2.2,
    nearFade: 3.4,
    name: 'lakeHaze',
  });
  haze.group.position.set(0, 0, -12);
  group.add(haze.group);
  owned.push(haze);
  timed.push(haze);

  const clouds: MistPlanesHandle = makeMistPlanes({
    size: [56, 20],
    color: blendTokens(TOKEN_WHITE, palette.skyHorizon, 0.12),
    seed: ctx.seed ^ 0x71c3,
    tier,
    layers: 3,
    baseY: 3.2,
    spread: 1.9,
    // near-white on a pale noon sky is an inherently low-contrast read; below ~0.35 the
    // bank is measurably invisible and only costs draws
    opacity: 0.42,
    speed: 0.004,
    scale: 1.5,
    nearFade: 9,
    name: 'noonClouds',
  });
  clouds.group.position.set(0, 0, -19);
  group.add(clouds.group);
  owned.push(clouds);
  timed.push(clouds);

  const animated: OakMaterial[] = [padMaterial, timberMaterial];
  timed.push({ setTime: (t) => setMaterialsTime(animated, t) });

  const sunBase = rig.sun.intensity;
  const hemiBase = rig.hemi.intensity;

  return {
    group,
    sky,
    rig,
    water,
    terrain,
    endLeftRoot: endRoots[0] as THREE.Object3D,
    endRightRoot: endRoots[1] as THREE.Object3D,
    gaugeRoot,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
    },
    setWind(lean) {
      const l = clampRange(lean, -1, 1);
      // A SIGNED amplitude flips the sway phase, so the grass combs with the sapling.
      // The offset is chosen so the flip happens while the amplitude is near zero —
      // i.e. where a phase inversion is invisible — instead of popping mid-sway.
      const k = 0.45 + 0.75 * l;
      grass.setWind(GRASS_WIND * k, 1.6 + 0.5 * Math.abs(l));
      reeds.setWind(REED_WIND * k, 1.15 + 0.4 * Math.abs(l));
      fringe.setWind(REED_WIND * 1.4 * k, 1.05 + 0.35 * Math.abs(l));
      trees.setWind(TREE_WIND * (0.7 + 0.4 * Math.abs(l)), 0.85);
    },
    setCool(env01) {
      const c = clampRange(env01, 0, 1);
      // the dense +1 IS the lighting: while the sapling stands the key light is full
      // noon; a fall dips the key and lets the cool hemisphere fill take over
      rig.sun.intensity = sunBase * (1 - 0.13 * c);
      rig.hemi.intensity = hemiBase * (1 + 0.34 * c);
    },
    setPixelRatio(dpr) {
      pollen.setPixelRatio(dpr);
      spray.setPixelRatio(dpr);
    },
    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      animated.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}
