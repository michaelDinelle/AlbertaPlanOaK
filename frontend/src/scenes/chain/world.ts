// Sixteen Stones — the static world (Chain-cont-v0).
//
// THE ONE IDEA: the env is a 16×1 grid and the art is a 16-stone ford across a creek.
// Cell i IS stone i, at `cellToWorld(i, 0)`, and nothing here re-derives that: the crossing
// is built by walking `grid.walls` (all zero — Chain has no walls) exactly like every other
// grid scene, so if the corridor length ever changed the ford would grow with it.
//
// THE CROSSING IS TWO OBJECTS, both placed by `cellToWorld`:
//  * `buildCellField` draws the SUBMERGED PLINTH under each stone — a slab whose top face
//    is the style contract's y = 0 ground plane and whose body drops 0.9 into the creek.
//    That is the kit substrate the grid canon requires, and under translucent water it is
//    what gives the stones their mass.
//  * a bespoke 7-sided river-stone CAP owns the dry top: one InstancedMesh, one instance
//    per cell, top face exactly at y = 0 so the wisp stands on the stone and not near it.
//    Its `instanceColor` is also the channel the recently-stepped-on glow rides (the
//    four-rooms lesson: a cell field is a substrate, and per-cell hue is instance color).
//
// CONTAINMENT IS DERIVED, NOT ASSERTED (review fix, and a lesson worth keeping). The first
// pass claimed the slab could never poke through because its half-diagonal (0.354) was
// inside "the cap's tightest radius (0.37)". Three things were wrong with that, and half
// the ford showed it: 0.37 is the CIRCUMradius of an n-gon, so the tightest radius is the
// APOTHEM `r·cos(π/n)` — and n drops to 5 at thumb; every cap is additionally xz-scaled by
// 0.9–1.08; and `buildCellField` composes the slab with an IDENTITY quaternion while each
// cap gets a random yaw, so the box corners sit at fixed world diagonals and are guaranteed
// to find an edge midpoint. Measured before the fix: 9/16 stones protruding at hero
// (worst 0.038), 11/16 at medium (0.051), 16/16 at thumb (0.069) — dark box corners
// breaking the silhouette of the scene's one state-bearing prop, worse the smaller the tier.
// `PLINTH_INSET` below is now COMPUTED from the cap's worst-case apothem, so the slab is
// contained by construction and the test measures the real instance matrices, not a claim.
//
// LIGHT: afternoon, low and from behind-left, so the stones carry a warm rim and the sun's
// specular path lies on the water between the camera and the far bank. The palette's own
// sun elevation (56°) is a noon angle and flattens the stones into pale squares; 38° is
// what makes sixteen identical rocks read as sixteen separate silhouettes.
//
// ALBEDO: the afternoon key is 2.7 — within a hair of noon — so the four-rooms albedo law
// applies unchanged. Every ground/stone color goes through `shade()` before it is used.

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
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
  type VegetationHandle,
} from '@kit/vegetation';
import { makeWater, type WaterHandle } from '@kit/water';
import { CHAIN_END, CHAIN_L, chainDecoder } from '@sim/envs/chain';
import type { EnvGeometry, GridGeometry } from '@sim/types';

// ---------------------------------------------------------------------------
// world constants (1 cell = 1 world unit; the ford really is 16 stones long)
// ---------------------------------------------------------------------------

/**
 * The creek surface. Low enough that a stone reads as a stone standing PROUD of the
 * water (the whole point of a stepping stone); high enough that the water still laps the
 * cap's flared waist instead of exposing the plinth.
 */
export const WATER_LEVEL = -0.17;
/** creek bed / bank heights — the ford is a broad shallow ford, not a gorge */
export const BED_Y = -1.05;
export const BANK_Y = 0.62;
/** flat bed inside |x| < this; the bank ramps up over BANK_RISE beyond it */
export const CREEK_HALF = 8.0;
export const BANK_RISE = 2.2;
/** the bank edge meanders by at most this, so it never crawls onto the crossing */
export const MEANDER = 0.55;

/**
 * The creek has to RUN somewhere. A ford is only a ford if the water arrives from upstream
 * and leaves downstream, so the ground is long in z (±40) and only as wide in x as the two
 * banks need — and the far end of it is where this scene gets its horizon.
 */
export const TERRAIN_SIZE: readonly [number, number] = [52, 80];
export const TERRAIN_SEGMENTS: readonly [number, number] = [120, 176];
/**
 * Fog reach. `applyFog` turns extent into density via `fogDensityHint` (afternoon = 0.3), so
 * 300 puts full density ~90 units out: the far bank (~20 units) sits at 13%, the wooded
 * spit downstream (~45 units) at 53%, and the ridge behind it at ~70%. That is what turns
 * the bend into a horizon instead of a cliff. The first pass used 384 (the creek ran to a
 * hard edge, unhazed) and the second 253 (the whole creek read as fog, not water).
 */
export const FOG_EXTENT = 300;
/**
 * Afternoon: 38° up, from behind and to the left. `paletteSunDirection(afternoon)` returns
 * 56° (the palette's key is nearly noon's), which lights sixteen identical stones straight
 * down and turns them into pale squares. The negative z puts the sun's specular path on
 * the water BETWEEN the camera and the far bank — the one bright thing in the frame that
 * is not the tree.
 */
export const SUN_DIR: readonly [number, number, number] = [-0.46, 0.62, -0.64];
/** the shadow frustum covers the frame (crossing + both banks), not just the stones */
export const SHADOW_RADIUS = 12.5;

/** The composition azimuth, shared with index.ts's OrbitRig so planting and camera agree. */
export const CAMERA_AZIMUTH = 0.35;
/** how far out along that azimuth the "keep it clear" wedge is centred */
export const CAMERA_STANDOFF = 13.5;

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
 * STYLE.md names it as the legal route to near-black, so shade is a blend like every other
 * color here. The afternoon key is 2.7 — a hair under noon's 3.0 — so the four-rooms
 * albedo law holds unchanged: ground/stone albedo belongs around 140–170 sRGB, not 220,
 * or the ford blooms into one white ribbon and sixteen stones become one.
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
 * disposing the sources. A hand-built prop (the blossom tree, a root boss) wants its SOURCE
 * normals plus a per-part palette baked in, so one mesh + one material draws the whole prop.
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

/** A tapered limb from `a` to `b` (bark, boughs, reed stems). Base radius `ra`, tip `rb`. */
export function tubeGeometry(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  ra: number,
  rb: number,
  radial: number,
): THREE.BufferGeometry {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-4;
  const g = new THREE.CylinderGeometry(rb, ra, len, Math.max(3, radial), 1);
  g.translate(0, len / 2, 0);
  const q = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(dx / len, dy / len, dz / len),
  );
  g.applyQuaternion(q);
  g.translate(a[0], a[1], a[2]);
  return g;
}

// ---------------------------------------------------------------------------
// the env's own truth
// ---------------------------------------------------------------------------

/**
 * The grid the scene draws. The host always hands us the decoder's geometry; the fallback
 * is the DECODER'S OWN geometry builder, never a scene re-derivation, so a placeholder
 * geometry degrades to the correct 16×1 corridor instead of a guess.
 */
export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === CHAIN_L && g.h === 1) return g;
  const fallback = chainDecoder.geometry(chainDecoder.initStatic(0), 0).grid;
  if (fallback) return fallback;
  return { w: CHAIN_L, h: 1, walls: new Uint8Array(CHAIN_L), goal: { x: CHAIN_END, y: 0 } };
}

/** World x of a stone's centre — the one conversion, wrapped for readability. */
export function stoneX(index: number, grid: GridSize): number {
  const out: Vec3Tuple = [0, 0, 0];
  return cellToWorld(index, 0, grid, out)[0];
}

// ---------------------------------------------------------------------------
// the palette (every color a blend of tokens — STYLE.md)
// ---------------------------------------------------------------------------

export interface CreekSkin {
  /** dry, sun-warmed stone top */
  stoneDry: THREE.Color;
  /** wet stone at the waterline */
  stoneWet: THREE.Color;
  /** the submerged plinth */
  stoneDeep: THREE.Color;
  /** the warm tint a recently-stepped-on stone glows toward */
  stoneGlow: THREE.Color;
  /** creek bed sand */
  bed: THREE.Color;
  bank: THREE.Color;
  bankHigh: THREE.Color;
  reed: THREE.Color;
  meadow: THREE.Color;
  flower: THREE.Color;
  bankTree: THREE.Color;
  bark: THREE.Color;
  moss: THREE.Color;
  blossom: THREE.Color;
  waterShallow: THREE.Color;
  waterDeep: THREE.Color;
  /** the fresnel/specular tint the kit water multiplies — the scene's brightest lever */
  waterSky: THREE.Color;
  waterCaustic: THREE.Color;
  motes: THREE.Color;
  dragonfly: THREE.Color;
  wing: THREE.Color;
}

/**
 * Afternoon over a creek. The blossom is `chain`'s own accent, the reeds borrow
 * `cart-pole`'s young-leaf green, the bank flowers borrow `acrobot`'s dawn blossom and the
 * stones are cooled with `po-maze` fog silver — four accents from the shared table, so the
 * gallery still reads as one preserve and this corner of it still reads as a creek.
 */
export function creekSkin(palette: Palette, tokens: typeof TOKENS): CreekSkin {
  const blossom = tokens.accents['chain'] ?? tokens.celebrationGold;
  const leaf = tokens.accents['cart-pole'] ?? palette.groundA;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const petalPink = tokens.accents['acrobot'] ?? blossom;
  const teal = tokens.accents['deep-sea'] ?? palette.hemiSky;
  const white = palette.sun;

  return {
    // warm river stone, not kerbstone: the first pass leaned on the cool silver accent and
    // sixteen grey pucks in green water read as concrete
    stoneDry: shade(blendTokens(palette.groundB, silver, 0.28), 0.46),
    stoneWet: shade(blendTokens(silver, palette.hemiGround, 0.52), 0.64),
    stoneDeep: shade(blendTokens(palette.hemiGround, silver, 0.2), 0.72),
    stoneGlow: blendTokens(blossom, tokens.celebrationGold, 0.42),
    bed: shade(blendTokens(palette.groundB, palette.groundA, 0.42), 0.56),
    bank: shade(blendTokens(palette.groundA, palette.groundB, 0.38), 0.48),
    bankHigh: shade(blendTokens(palette.groundB, leaf, 0.3), 0.44),
    reed: shade(blendTokens(leaf, palette.groundA, 0.34), 0.44),
    meadow: shade(blendTokens(palette.groundA, leaf, 0.3), 0.4),
    flower: shade(blendTokens(petalPink, white, 0.2), 0.3),
    bankTree: shade(blendTokens(palette.groundA, leaf, 0.34), 0.46),
    bark: shade(blendTokens(palette.hemiGround, TOKENS.palettes.dusk.groundA, 0.4), 0.5),
    moss: shade(blendTokens(leaf, palette.groundA, 0.5), 0.52),
    blossom: shade(blendTokens(blossom, white, 0.1), 0.2),
    // The creek is the biggest surface in the frame and it sits under a 2.7 key with an
    // additive caustic, an additive fresnel rim AND a 1.6× specular the kit hard-codes. The
    // first pass used the palette's pale horizon green for the shallows and the sky tint,
    // and every one of those terms stacked into a single blown white sheet that swallowed
    // eight of the sixteen stones. Both water colours are now deep, and `sky` (which the
    // kit multiplies into the specular) is the shaded sky TOP, not the pale horizon.
    // A CREEK, not a pool. The reef teal alone came out swimming-pool cyan and read
    // brighter than the sunlit banks it runs between, which inverts the whole frame: water
    // in a meadow is the DARK shape. Both stops are pulled most of the way to the palette's
    // own ground green and then shaded well past the banks.
    waterShallow: shade(blendTokens(blendTokens(teal, palette.groundA, 0.62), palette.hemiGround, 0.28), 0.44),
    waterDeep: shade(blendTokens(blendTokens(teal, palette.groundA, 0.5), palette.skyTop, 0.34), 0.7),
    waterSky: shade(palette.skyTop, 0.46),
    waterCaustic: shade(blendTokens(palette.sun, teal, 0.34), 0.46),
    motes: blendTokens(blossom, white, 0.42),
    dragonfly: shade(blendTokens(teal, palette.skyTop, 0.3), 0.62),
    wing: shade(blendTokens(silver, white, 0.4), 0.24),
  };
}

// ---------------------------------------------------------------------------
// the creek bed height field
// ---------------------------------------------------------------------------

/**
 * A broad shallow ford. The bed is flat-ish inside |x| < CREEK_HALF and ramps to the bank
 * over BANK_RISE; the edge MEANDERs with z so the banks wind instead of running ruler
 * straight, and the meander is clamped well outside the last stone's cap so land can never
 * crawl onto the crossing. Bed noise fades to nothing along the centre line, which is the
 * exact-band rule from MountainCar applied to a creek: the water reads clean exactly where
 * the sixteen stones are, and gets its texture everywhere else.
 */
export function makeCreekHeight(seed: number): (x: number, z: number) => number {
  const span = BANK_Y - BED_Y;
  return (x, z) => {
    const wander = (valueNoise2(z * 0.075 + 3.1, 7.7, 64, seed) - 0.5) * 2 * MEANDER;
    const s = smoothstep01((Math.abs(x) + wander - CREEK_HALF) / BANK_RISE);
    // DOWNSTREAM THE CREEK BENDS AWAY. Without this the water simply ran to the edge of the
    // world and the top half of every frame was a featureless pale sheet with no horizon in
    // it — the one thing STYLE.md says must always be visible. A wooded spit closing the
    // channel at z ≈ −25 gives the frame a far shore, and the rise behind it gives it a sky.
    const bend = smoothstep01((-z - 19) / 11);
    const land = s > bend ? s : bend;
    const far = smoothstep01((Math.abs(z) - 13) / 20) * 1.9 * land;
    const base = BED_Y + span * land + far;
    // bed/bank texture — identically 0 along the crossing's centre line
    const w = smoothstep01(Math.abs(z) / 2.4) * 0.55 + land * 0.85;
    if (w <= 0) return base;
    const n1 = valueNoise2(x * 0.12 + 5.3, z * 0.12 + 11.7, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.3 + 2.1, z * 0.3 + 7.3, 64, seed ^ 0x51ed270b) - 0.5;
    return base + w * (n1 * 0.9 + n2 * 0.34);
  };
}

// ---------------------------------------------------------------------------
// the stone caps — the dry top of every stone, one draw for the whole ford
// ---------------------------------------------------------------------------

export interface StoneCapsHandle {
  mesh: THREE.InstancedMesh;
  material: OakMaterial;
  /** how many stones were placed (= grid.w) */
  count: number;
  /**
   * Warm a stone the wisp has just stepped on, 0 = at rest. This is a FLOURISH channel,
   * not state: it is written from a decay clock the scene clears on every scrub.
   */
  setGlow(index: number, amount: number): void;
  /** drop every stone back to its resting tint (scrub) */
  clearGlow(): void;
  dispose(): void;
}

/** the cap's cross-section, in cap-local units before the per-instance xz scale */
export const CAP_R_TOP = 0.37;
export const CAP_R_WAIST = 0.42;
export const CAP_R_FOOT = 0.4;
/** the waist sits this fraction of the cap's height below the crown */
export const CAP_WAIST_FRAC = 0.42;
/** the cap's own height (its crown is y = 0, its foot y = −CAP_HEIGHT before y scale) */
export const CAP_HEIGHT = 0.5;
/** the per-instance xz scale range every stone is placed with */
export const CAP_SCALE_MIN = 0.9;
export const CAP_SCALE_MAX = 1.08;

/** How many sides a stone has at a tier. Thumb's pentagon is the tightest apothem there is. */
export function capSides(tier: QualityTier): number {
  return tier === 'thumb' ? 5 : tier === 'medium' ? 6 : 7;
}

/**
 * The cap's INSCRIBED radius (apothem) in world units — the honest containment bound. The
 * profile only ever widens below the crown (0.37 → 0.42 → 0.40), so the tightest circle the
 * cap presents anywhere at or under y = 0 is its crown ring, and the tightest POLYGON is
 * that ring's apothem at the smallest instance scale. Anything inside this radius is hidden
 * by the stone at every yaw, every tier and every instance.
 */
export function capApothem(sides: number, xzScale: number): number {
  return CAP_R_TOP * xzScale * Math.cos(Math.PI / Math.max(3, sides));
}

/** how far the plinth's top face sits under the cap's crown (z-fight relief, see buildWorld) */
export const PLINTH_DROP = 0.06;
/**
 * The submerged plinth's half-width, DERIVED. `buildCellField` gives every slab an identity
 * yaw and a square footprint, so its worst presentation is a corner at 45° — a half-diagonal
 * of `half·√2`. That has to live inside the tightest apothem any cap presents anywhere in
 * the gallery (thumb's pentagon at the smallest instance scale), with a margin for the
 * shading to survive too: a corner that only just clears reads as a crease.
 */
export const PLINTH_HALF = (capApothem(5, CAP_SCALE_MIN) * 0.94) / Math.SQRT2;
/** …and the inset the kit wants is the rest of the cell. */
export const PLINTH_INSET = CELL / 2 - PLINTH_HALF;

/**
 * One river stone: a 7-gon prism with its TOP FACE EXACTLY AT y = 0 (the style contract's
 * ground plane — the wisp stands on this, so it may not drift by a millimetre), a flared
 * waist, and a narrower wet foot. Vertex-graded pale-dry on top through wet-dark below, so
 * sixteen instances of one geometry read as sixteen weathered stones under one sun.
 */
export function stoneCapGeometry(sides: number, height: number): THREE.BufferGeometry {
  const n = Math.max(5, sides);
  const rTop = CAP_R_TOP;
  const rWaist = CAP_R_WAIST;
  const rFoot = CAP_R_FOOT;
  const yWaist = -height * CAP_WAIST_FRAC;
  const yFoot = -height;
  // grade values (multiplied into the instance tint): dry crown -> wet foot
  const gTop = 1.14;
  const gWaist = 0.9;
  const gFoot = 0.52;

  const tris = n /* top fan */ + n * 2 /* upper band */ + n * 2 /* lower band */;
  const positions = new Float32Array(tris * 3 * 3);
  const colors = new Float32Array(tris * 3 * 3);
  let o = 0;
  const push = (x: number, y: number, z: number, g: number): void => {
    positions[o * 3] = x;
    positions[o * 3 + 1] = y;
    positions[o * 3 + 2] = z;
    colors[o * 3] = g;
    colors[o * 3 + 1] = g;
    colors[o * 3 + 2] = g;
    o++;
  };
  const ring = (i: number, r: number): [number, number] => {
    const a = ((i % n) / n) * Math.PI * 2;
    return [Math.cos(a) * r, Math.sin(a) * r];
  };

  for (let i = 0; i < n; i++) {
    const [ax, az] = ring(i, rTop);
    const [bx, bz] = ring(i + 1, rTop);
    // top fan (CCW seen from +y)
    push(0, 0, 0, gTop);
    push(bx, 0, bz, gTop);
    push(ax, 0, az, gTop);

    const [wax, waz] = ring(i, rWaist);
    const [wbx, wbz] = ring(i + 1, rWaist);
    push(ax, 0, az, gTop);
    push(bx, 0, bz, gTop);
    push(wbx, yWaist, wbz, gWaist);
    push(ax, 0, az, gTop);
    push(wbx, yWaist, wbz, gWaist);
    push(wax, yWaist, waz, gWaist);

    const [fax, faz] = ring(i, rFoot);
    const [fbx, fbz] = ring(i + 1, rFoot);
    push(wax, yWaist, waz, gWaist);
    push(wbx, yWaist, wbz, gWaist);
    push(fbx, yFoot, fbz, gFoot);
    push(wax, yWaist, waz, gWaist);
    push(fbx, yFoot, fbz, gFoot);
    push(fax, yFoot, faz, gFoot);
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'chainStoneCap';
  return g;
}

function buildStoneCaps(
  grid: GridGeometry,
  skin: CreekSkin,
  palette: Palette,
  tier: QualityTier,
  seed: number,
): StoneCapsHandle {
  const sides = capSides(tier);
  const geometry = stoneCapGeometry(sides, CAP_HEIGHT);
  const material = oakMaterial({
    // white base + full-colour instance tint: the tint MULTIPLIES, so one material gives
    // every stone its own hue AND gives the glow somewhere to live
    color: 0xffffff,
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.22,
    tier,
    name: 'chainStoneCapMaterial',
  });

  const count = grid.w;
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
  mesh.name = 'chainStoneCaps';
  mesh.castShadow = tier !== 'thumb';
  // The stones cast but do not RECEIVE. A 0.8-unit faceted prism seen from a 29° camera
  // self-shadows through the depth bias, and the acne landed as a hard dark wedge across
  // every stone's crown — sixteen identical smudges, which is worse than the shadow is good.
  mesh.receiveShadow = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3);

  const base = new Float32Array(Math.max(1, count) * 3);
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3();
  const color = new THREE.Color();
  const world: Vec3Tuple = [0, 0, 0];

  for (let i = 0; i < count; i++) {
    cellToWorld(i, 0, grid, world);
    const n = hash2f(i, 0, seed);
    const m = hash2f(i, 1, seed ^ 0x51ed270b);
    position.set(world[0], 0, world[2]);
    euler.set(0, n * Math.PI * 2, 0);
    quaternion.setFromEuler(euler);
    // Y scales about the object origin, and the geometry's top face IS the origin plane,
    // so a taller or shorter stone still presents its crown at exactly y = 0. The xz range
    // is CAP_SCALE_MIN..CAP_SCALE_MAX and nothing else — the plinth's width is derived from
    // that minimum, so a wider spread here would have to be paid for there.
    const xz = CAP_SCALE_MIN + m * (CAP_SCALE_MAX - CAP_SCALE_MIN);
    scale.set(xz, 0.78 + n * 0.5, xz);
    matrix.compose(position, quaternion, scale);
    mesh.setMatrixAt(i, matrix);
    color.copy(skin.stoneDry).lerp(skin.stoneWet, m * 0.3).multiplyScalar(0.86 + n * 0.3);
    base[i * 3] = color.r;
    base[i * 3 + 1] = color.g;
    base[i * 3 + 2] = color.b;
    mesh.setColorAt(i, color);
  }
  mesh.count = count;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();

  const scratch = new THREE.Color();
  const glow = new THREE.Color().copy(skin.stoneGlow);

  return {
    mesh,
    material,
    count,
    setGlow(index, amount) {
      if (index < 0 || index >= count) return;
      const a = amount < 0 ? 0 : amount > 1 ? 1 : amount;
      scratch.setRGB(base[index * 3] ?? 1, base[index * 3 + 1] ?? 1, base[index * 3 + 2] ?? 1);
      if (a > 0) scratch.lerp(glow, a * 0.45).multiplyScalar(1 + a * 0.3);
      mesh.setColorAt(index, scratch);
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    },
    clearGlow() {
      for (let i = 0; i < count; i++) {
        scratch.setRGB(base[i * 3] ?? 1, base[i * 3 + 1] ?? 1, base[i * 3 + 2] ?? 1);
        mesh.setColorAt(i, scratch);
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
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
// dragonflies — three of them, and every one is a pure function of ctx.time()
// ---------------------------------------------------------------------------

export interface DragonfliesHandle {
  mesh: THREE.InstancedMesh;
  count: number;
  setTime(t: number): void;
  dispose(): void;
}

function dragonflyGeometry(skin: CreekSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const body = new THREE.CylinderGeometry(0.012, 0.02, 0.24, 5, 1);
  body.rotateX(Math.PI / 2);
  parts.push({ geometry: body, color: skin.dragonfly });
  const head = new THREE.IcosahedronGeometry(0.032, 0);
  head.translate(0, 0, 0.13);
  parts.push({ geometry: head, color: skin.dragonfly });
  for (const sx of [-1, 1]) {
    for (const oz of [0.05, -0.03]) {
      const wing = new THREE.PlaneGeometry(0.19, 0.055);
      wing.rotateY(Math.PI / 2);
      wing.rotateZ(sx > 0 ? -0.22 : 0.22);
      wing.translate(sx * 0.105, 0.018, oz);
      parts.push({ geometry: wing, color: skin.wing });
    }
  }
  return mergeColoredParts(parts, 'chainDragonfly');
}

function buildDragonflies(
  skin: CreekSkin,
  palette: Palette,
  tier: QualityTier,
  seed: number,
): DragonfliesHandle {
  const count = tier === 'thumb' ? 0 : 3;
  const geometry = dragonflyGeometry(skin);
  const material = oakMaterial({
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.5,
    metalness: 0.1,
    rimColor: palette.skyHorizon,
    rimStrength: 0.55,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.4,
    tier,
    name: 'chainDragonflyMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
  mesh.name = 'chainDragonflies';
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.count = count;
  mesh.visible = count > 0;
  mesh.frustumCulled = false;

  const phase = new Float32Array(Math.max(1, count) * 4);
  for (let i = 0; i < count; i++) {
    phase[i * 4] = hash2f(i, 11, seed) * Math.PI * 2;
    phase[i * 4 + 1] = 0.42 + hash2f(i, 12, seed) * 0.36;
    phase[i * 4 + 2] = (hash2f(i, 13, seed) - 0.5) * 2;
    phase[i * 4 + 3] = 0.72 + hash2f(i, 14, seed) * 0.5;
  }

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);

  return {
    mesh,
    count,
    setTime(t) {
      if (count === 0) return;
      for (let i = 0; i < count; i++) {
        const p0 = phase[i * 4] ?? 0;
        const rate = phase[i * 4 + 1] ?? 0.5;
        const lane = phase[i * 4 + 2] ?? 0;
        const size = phase[i * 4 + 3] ?? 1;
        const a = t * rate + p0;
        // a lazy Lissajous over the ford, and its analytic derivative for the heading
        const x = Math.sin(a) * 6.4 + Math.sin(a * 2.3 + p0) * 1.1;
        const z = lane * 2.1 + Math.cos(a * 1.7 + p0) * 1.35;
        const y = 0.62 + Math.sin(a * 2.9 + p0) * 0.22;
        const dx = Math.cos(a) * 6.4 * rate + Math.cos(a * 2.3 + p0) * 2.53 * rate;
        const dz = -Math.sin(a * 1.7 + p0) * 2.295 * rate;
        position.set(x, y, z);
        euler.set(Math.sin(a * 4.1) * 0.12, Math.atan2(dx, dz), Math.sin(a * 5.3 + p0) * 0.25);
        quaternion.setFromEuler(euler);
        scale.setScalar(size);
        matrix.compose(position, quaternion, scale);
        mesh.setMatrixAt(i, matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
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
// scatter exclusions
// ---------------------------------------------------------------------------

/** Nothing plants itself on the crossing corridor — the ford must stay legible. */
export function excludeCrossing(grid: GridSize, halfZ: number, margin: number): ExcludeFn {
  const halfX = ((grid.w - 1) / 2) * CELL + 0.5 + margin;
  return (x, z) => Math.abs(z) <= halfZ && Math.abs(x) <= halfX;
}

/** Keep a clear disc around every stone centre (creek rocks in the shallows). */
export function excludeStones(grid: GridSize, radius: number): ExcludeFn {
  const r2 = radius * radius;
  const ox = (grid.w - 1) / 2;
  return (x, z) => {
    const i = Math.round(x / CELL + ox);
    if (i < 0 || i > grid.w - 1) return false;
    const dx = x - (i - ox) * CELL;
    return dx * dx + z * z < r2;
  };
}

/** Nothing lives between the camera and the physics (the MountainCar law). */
export function excludeCameraWedge(halfX: number, halfZ: number): ExcludeFn {
  const cx = Math.sin(CAMERA_AZIMUTH) * CAMERA_STANDOFF;
  const cz = Math.cos(CAMERA_AZIMUTH) * CAMERA_STANDOFF;
  return (x, z) => Math.abs(x - cx) <= halfX && Math.abs(z - cz) <= halfZ;
}

// ---------------------------------------------------------------------------
// world handle
// ---------------------------------------------------------------------------

export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  water: WaterHandle;
  /** the submerged plinths — the kit substrate under the crossing */
  field: CellFieldHandle;
  /** the dry stone tops (and the visited-glow channel) */
  caps: StoneCapsHandle;
  /** blossom motes around the tree; density rides the bloom */
  petalMotes: AmbientParticlesHandle;
  /** ctx.time() — pause freezes water, reeds, motes and dragonflies together */
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
  const skin = creekSkin(palette, ctx.tokens);
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'sixteenStonesWorld';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];

  // ---- sky (exactly one) ----
  const sky = skyFromPalette(palette, 'afternoon', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.04,
    sunHdr: 5.2,
    sunGlow: 0.26,
    sunGlowPower: 26,
    topSpread: 0.68,
    bottomSpread: 0.46,
    radius: 340,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the one light budget: sun + hemi + 3 points (tree / wisp / celebration) ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 3,
    pointDistance: 7.5,
    sunDistanceScale: 2.1,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: 0.1, z: 0 }, SHADOW_RADIUS);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the creek bed and its two banks ----
  const terrain = makeTerrain({
    size: TERRAIN_SIZE,
    segments: TERRAIN_SEGMENTS,
    tier,
    seed: ctx.seed,
    heightFn: makeCreekHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.05,
    slopeColor: skin.bank,
    slopeStrength: 0.55,
    slopeSharpness: 2.2,
    // The height range spans the bed (−1.05) to the far ridge (~+2.5), so the stops are
    // placed where the LANDFORMS are, not evenly: the waterline sits at t ≈ 0.25 and the
    // bank crest at t ≈ 0.47. Evenly spaced stops put the entire visible bank inside one
    // band and the whole frame came out a single flat olive.
    colorRamp: [
      { t: 0, color: skin.bed },
      { t: 0.22, color: blendTokens(skin.bed, skin.bank, 0.55) },
      { t: 0.34, color: skin.bank },
      { t: 0.52, color: skin.bankHigh },
      { t: 1, color: blendTokens(skin.bankHigh, palette.hemiSky, 0.28) },
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
    name: 'creekBed',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  // ---- the creek. A KIT material, so the scene's one bespoke-shader budget stays unspent ----
  const water = makeWater({
    size: [TERRAIN_SIZE[0] + 8, TERRAIN_SIZE[1] + 10],
    palette,
    tier,
    segments: 128,
    level: WATER_LEVEL,
    shallow: skin.waterShallow,
    deep: skin.waterDeep,
    sky: skin.waterSky,
    caustic: skin.waterCaustic,
    amplitude: 0.024,
    waveScale: 1.45,
    speed: 0.8,
    // every one of these is a bloom feeder over the frame's largest surface: the creek gets
    // a rim and a shimmer, not a light source
    fresnel: 0.24,
    caustics: 0.14,
    depthFalloff: 9,
    // a shade translucent: the submerged plinths showing through are what sit the stones
    // IN the creek, and the kit water cannot receive a real shadow
    opacity: 0.8,
    sunDir: sky.sunDirection,
    name: 'creekWater',
  });
  // `makeWater` LERPS the constructor colours toward the palette (shallow 35% toward
  // groundB, deep 55% toward skyBottom), which is a sensible default for a pond under a
  // sky — and fatal here. The afternoon skyBottom is a bright cream, so a deep creek green
  // came out at 45% strength over 55% cream: the flattest, palest sheet in the gallery, with
  // no depth gradient left in it at all. `setColors` writes the live uniforms with no lerp,
  // so this line is what makes the water water.
  water.setColors(skin.waterShallow, skin.waterDeep);
  group.add(water.mesh);
  owned.push(water);
  timed.push({ setTime: (t) => water.setTime(t) });

  // ---- the crossing: sixteen plinths (kit substrate) + sixteen caps (the art) ----
  const field = buildCellField(
    grid,
    {
      floorColor: skin.stoneDeep,
      wallColor: skin.stoneDeep,
      // DERIVED, not chosen: PLINTH_INSET is whatever leaves the slab's half-diagonal inside
      // the cap's worst-case apothem (thumb pentagon × the smallest instance scale), so the
      // plinth is a column the stone completely covers and only its submerged shaft shows
      // through the creek. See the containment note at the top of this file.
      floorInset: PLINTH_INSET,
      floorThickness: 0.9,
      jitter: 0.12,
      rimColor: palette.skyHorizon,
      rimStrength: 0.12,
      seed: ctx.seed,
      name: 'stonePlinths',
    },
    tier,
  );
  // `buildCellField` puts a floor tile's TOP FACE at exactly y = 0 — the style contract's
  // ground plane — and so does the stone cap's crown. Two coplanar faces at the same depth
  // is z-fighting, and it landed as a dark angular smear across all sixteen stone tops.
  // The plinth is submerged mass, not an accuracy anchor (the cap and the wisp are), so it
  // sinks PLINTH_DROP and lives entirely inside the cap above the waterline.
  field.group.position.y = -PLINTH_DROP;
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  const caps = buildStoneCaps(grid, skin, palette, tier, ctx.seed);
  group.add(caps.mesh);
  owned.push(caps);

  // ---- the banks ----
  const offCrossing = excludeAny(
    excludeCrossing(grid, 1.15, 0.6),
    excludeStones(grid, 0.78),
    excludeCameraWedge(4.6, 4.6),
  );
  const heightAt = (x: number, z: number): number => terrain.sampleHeight(x, z);

  // reeds at the waterline: two bands, one per bank, sitting where the ground crosses the
  // water level rather than at a hand-typed z
  const reeds: VegetationHandle[] = [];
  for (const side of [-1, 1] as const) {
    const handle = scatterGrass({
      count: 900,
      area: { x: side * 9.4, z: 0, halfX: 2.1, halfZ: 26 },
      seed: (ctx.seed ^ 0x4ee1) + (side > 0 ? 977 : 0),
      tier,
      exclude: offCrossing,
      heightAt,
      color: skin.reed,
      scale: [0.85, 1.6],
      stretch: [1.2, 2.2],
      hueJitter: 0.26,
      wind: 0.05,
      windFrequency: 1.15,
      rimColor: palette.skyHorizon,
      rimStrength: 0.4,
      name: `creekReeds${side > 0 ? 'Far' : 'Near'}`,
    });
    group.add(handle.group);
    owned.push(handle);
    timed.push(handle);
    reeds.push(handle);
  }

  // A stand of reeds DOWNSTREAM of the ford, on the side away from the camera's own standoff.
  // The composed frame gives its bottom third to open water; this is what stops that third
  // being dead. They live at |z| ≈ 3–11, nowhere near a cell, so they can never be mistaken
  // for a seventeenth stone.
  // ROOTED, not floating (review fix). This was the one scatter that omitted `heightAt`, so
  // every blade started at y = 0 — 0.17 ABOVE the water — and the clump read as darts lying
  // on the surface with no base. The creek here is 0.7–1.1 deep, so being rooted on the bed
  // means being an EMERGENT reed: the sizes below put a blade at 0.7–1.5, which breaks the
  // surface by up to ~0.7 while its lower half stays visible through the translucent water.
  const shallowTufts = scatterGrass({
    count: 220,
    area: { x: -5.6, z: 6.8, halfX: 4.2, halfZ: 4.2 },
    seed: ctx.seed ^ 0x1c93,
    tier,
    exclude: excludeAny(excludeCrossing(grid, 2.2, 0.6), excludeCameraWedge(3.4, 3.4)),
    heightAt,
    color: skin.reed,
    scale: [0.85, 1.35],
    stretch: [1.9, 2.5],
    hueJitter: 0.28,
    wind: 0.055,
    windFrequency: 1.25,
    rimColor: palette.skyHorizon,
    rimStrength: 0.42,
    name: 'shallowReeds',
  });
  group.add(shallowTufts.group);
  owned.push(shallowTufts);
  timed.push(shallowTufts);
  reeds.push(shallowTufts);

  const meadow = scatterGrass({
    count: 1100,
    area: { x: 0, z: 0, halfX: 24, halfZ: 30 },
    seed: ctx.seed ^ 0x27a1,
    tier,
    exclude: excludeAny(offCrossing, (x) => Math.abs(x) < 10.4),
    heightAt,
    color: skin.meadow,
    scale: [0.32, 0.72],
    stretch: [0.8, 1.5],
    hueJitter: 0.24,
    wind: 0.045,
    windFrequency: 1.5,
    rimColor: palette.skyHorizon,
    rimStrength: 0.32,
    name: 'bankMeadow',
  });
  group.add(meadow.group);
  owned.push(meadow);
  timed.push(meadow);

  const flowers = scatterFlowers({
    count: 420,
    area: { x: 0, z: 0, halfX: 22, halfZ: 22 },
    seed: ctx.seed ^ 0x5b10,
    tier,
    exclude: excludeAny(offCrossing, (x) => Math.abs(x) < 10.0),
    heightAt,
    color: skin.flower,
    scale: [0.6, 1.15],
    stretch: [0.9, 1.5],
    hueJitter: 0.26,
    wind: 0.035,
    windFrequency: 1.6,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    name: 'bankFlowers',
  });
  group.add(flowers.group);
  owned.push(flowers);
  timed.push(flowers);

  // creek rocks: in the shallows between the stones and up the banks. `excludeStones`
  // keeps every one of them off a cell centre, so the ford is never ambiguous.
  const rocks = scatterRocks({
    count: 260,
    area: { x: 0, z: 0, halfX: 21, halfZ: 20 },
    seed: ctx.seed ^ 0x3f77,
    tier,
    // low boulders may stand in the near foreground: the MountainCar "nothing between the
    // camera and the physics" law is about OCCLUSION, and a 0.3-unit rock at the frame's
    // bottom edge occludes nothing while giving the empty near water something to be.
    exclude: excludeAny(excludeStones(grid, 0.95), excludeCameraWedge(2.6, 2.6)),
    heightAt,
    color: shade(blendTokens(skin.stoneWet, skin.bed, 0.3), 0.1),
    scale: [0.14, 0.5],
    stretch: [0.42, 0.86],
    hueJitter: 0.3,
    rimColor: palette.skyHorizon,
    rimStrength: 0.26,
    castShadow: false,
    name: 'creekRocks',
  });
  group.add(rocks.group);
  owned.push(rocks);
  timed.push(rocks);

  // the tree line, well back on both banks and out of the camera's wedge
  const bankTrees = scatterTrees({
    archetypes: ['broadleaf', 'conifer'],
    count: 100,
    area: { x: 0, z: 0, halfX: 24, halfZ: 34 },
    seed: ctx.seed ^ 0x1a70,
    tier,
    exclude: excludeAny(
      // the near bank stays open in front of the camera, the far bank carries the treeline
      // that becomes the horizon
      (x, z) => Math.abs(x) < 11.6 || (Math.abs(z) < 13 && Math.abs(x) < 16),
      excludeCameraWedge(7.5, 7.5),
    ),
    heightAt,
    maxSlope: 0.62,
    color: skin.bankTree,
    scale: [1.25, 2.4],
    stretch: [1.0, 1.6],
    hueJitter: 0.24,
    wind: 0.028,
    windFrequency: 0.85,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    castShadow: true,
    name: 'bankTrees',
  });
  group.add(bankTrees.group);
  owned.push(bankTrees);
  timed.push(bankTrees);

  // ---- idle life ----
  // Kit particle presets size their motes for a ~20-unit field; this ford is 16 across with
  // the camera ~15 out, so sizes land near 1, not the preset's 4–9.
  const motes: AmbientParticlesHandle[] = [];
  const pollen = makeAmbientParticles({
    preset: 'pollen',
    count: 240,
    bounds: { x: 0, y: 0.95, z: 0, halfX: 11, halfY: 1.0, halfZ: 7 },
    seed: ctx.seed ^ 0x7d21,
    tier,
    color: blendTokens(palette.sun, skin.motes, 0.3),
    size: 0.8,
    intensity: 1.15,
    opacity: 0.36,
    rise: 0.06,
    swirl: 0.3,
    name: 'creekPollen',
  });
  group.add(pollen.points);
  owned.push(pollen);
  timed.push(pollen);
  motes.push(pollen);

  const petalMotes = makeAmbientParticles({
    preset: 'snow',
    count: 220,
    bounds: { x: stoneX(CHAIN_END, grid) - 0.4, y: 1.5, z: 0, halfX: 2.4, halfY: 1.5, halfZ: 2.1 },
    seed: ctx.seed ^ 0x6c31,
    tier,
    color: skin.blossom,
    size: 1.25,
    intensity: 1.5,
    opacity: 0.78,
    rise: -0.22,
    swirl: 0.4,
    name: 'blossomPetalMotes',
  });
  petalMotes.setDensity(0);
  group.add(petalMotes.points);
  owned.push(petalMotes);
  timed.push(petalMotes);
  motes.push(petalMotes);

  // a thin warm haze downstream — afternoon air over water, not a fog bank
  const haze = makeMistPlanes({
    size: [50, 70],
    color: blendTokens(palette.fog, palette.sun, 0.25),
    seed: ctx.seed ^ 0x2c1b,
    tier,
    layers: 2,
    baseY: 0.02,
    spread: 0.7,
    // stacked planes multiply: three at 0.13 laid a 34% white veil over the whole creek
    opacity: 0.06,
    speed: 0.008,
    scale: 3.2,
    nearFade: 11,
    name: 'creekHaze',
  });
  group.add(haze.group);
  owned.push(haze);
  timed.push(haze);

  const dragonflies = buildDragonflies(skin, palette, tier, ctx.seed);
  group.add(dragonflies.mesh);
  owned.push(dragonflies);
  timed.push(dragonflies);

  return {
    group,
    sky,
    rig,
    terrain,
    water,
    field,
    caps,
    petalMotes,
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
      reeds.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}
