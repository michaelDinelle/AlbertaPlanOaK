// Gold Hour Savanna — the endless world the cheetah runs through.
//
// THE ACCURACY CLAIM OF THIS FILE: the ground is EXACTLY y = 0 wherever the animal can
// touch it. `savannaHeight` is identically 0 for |z| <= LANE_HALF_Z and only feathers
// into rolling ground far outside the running lane, so an FK toe at z = 0.046 (the geom
// radius) is resting on the visible surface, not near it. The debug overlay's FK stick
// figure is drawn from the same numbers, so "the skeleton stands on the ground" is true
// by construction.
//
// THE TREADMILL: the cheetah's rootx runs from 0 to ~1900 in a single recorded lane, so
// the world cannot be a fixed mesh. Instead everything here is a pure function of world
// x, re-windowed as the runner moves:
//
//   * the ground mesh's height field and vertex colors are PERIODIC in x with period
//     GROUND_PERIOD (built from integer harmonics, so the two edges agree to ~1e-9),
//     which makes snapping the mesh to a multiple of that period invisible — the surface
//     that leaves the back edge is the surface arriving at the front. h(x, z) is therefore
//     a global function of world position; props read it directly.
//   * grass / acacia / rock / cairn content is keyed by INTEGER TILE INDEX hashed with
//     the lane seed (`rngAt(tile, bankKey(name), ctx.seed)`), so tile 317 holds the same
//     three trees whether you walked there or scrubbed there, seed 7's plain is not seed
//     0's, and no two banks share a stream. Nothing is a history buffer.
//
// LIGHT: a 28° sun sits behind-left of the runner, so the animal is back-lit, its rim is
// the sun's own colour, and its shadow rakes forward-right toward the camera. The shadow
// focus is a tight box that TRACKS the cheetah: distant acacias fall outside the ortho
// frustum and cost nothing (three's frustum test leaves them unshadowed), which is what
// keeps a raking dusk sun from painting 20-unit tree streaks across the plain.

import * as THREE from 'three';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms, setMaterialsTime, type OakMaterial } from '@kit/materials';
import {
  makeAmbientParticles,
  makeMistPlanes,
  type AmbientParticlesHandle,
  type MistPlanesHandle,
} from '@kit/particles';
import { hash1f, hash2f, rngAt, streamFor, type Rng } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import { grassClumpGeometry, rockGeometry } from '@kit/vegetation';

// ---------------------------------------------------------------------------
// world constants (1 sim metre === 1 world unit — the cheetah torso really is 1.0 long)
// ---------------------------------------------------------------------------

/** |z| <= this is the EXACT band: savannaHeight === 0, so the toes strike y = 0. */
export const LANE_HALF_Z = 0.62;
/** world units over which the flat lane feathers into the rolling plain */
export const LANE_FEATHER = 3.1;
/** the ground height field repeats every this many units of x (see savannaHeight) */
export const GROUND_PERIOD = 48;
/** ground mesh footprint — 3 periods wide, so the window always covers rootx ± 48 */
export const GROUND_W = GROUND_PERIOD * 3;
export const GROUND_D = 104;
const GROUND_SEG_X = 192;
const GROUND_SEG_Z = 84;
/** scatter tile width; every prop is keyed by floor(x / TILE) hashed with the lane seed */
export const TILE = 6;
/** stone cairns every this many units — the distance ruler */
export const CAIRN_SPACING = 10;
/** fog reach is derived from the ground footprint, never hand-tuned per shot */
export const FOG_EXTENT = 250;
/**
 * Behind-left and 28° up. Two things fight: a dusk sun wants to be low (the whole point
 * of "gold hour"), and canon lesson 8 warns that a low sun smears long shadows. The
 * resolution is the tight, tracking shadow focus below — only the runner and the cairn
 * it is passing are inside the ortho box, so the raking light costs one long shadow of
 * the animal and nothing else. 28° is the floor of canon lesson 8, measured here: at 19° the
 * animal's own shadow stretched into two four-metre bands that read as mud rather than as
 * contact, and no amount of colour work fixed it.
 */
export const SUN_DIR: readonly [number, number, number] = [-0.62, 0.469, -0.63];
/** shadow box half-extent around the runner, sized to hold the animal AND its long shadow */
export const SHADOW_RADIUS = 4.2;

/** the plain's rolling amplitude only ramps in this far outside the lane */
const ROLL_RAMP = 22;
/** the far escarpment: distance, a broken horizon, and the frame's one cool value */
const RIDGE_Z = -72;
const RIDGE_PERIOD = 96;
const RIDGE_SEGMENTS = 288;

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
export function blendTokens(a: THREE.ColorRepresentation, b: THREE.ColorRepresentation, k: number): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), k);
}

/** `noon.sun` is the token white; `night.skyTop` the token near-black (canon lesson 9). */
export const TOKEN_WHITE = TOKENS.palettes.noon.sun;
export const TOKEN_BLACK = TOKENS.palettes.night.skyTop;

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

/**
 * Merge pre-transformed parts into ONE geometry carrying a flat per-part vertex color,
 * disposing the sources: a cairn, an acacia and a muzzle each become one mesh with a
 * stone/bark/nose palette baked in, drawn by the one shared material.
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
// the height field
// ---------------------------------------------------------------------------

const TWO_PI_OVER_PERIOD = (Math.PI * 2) / GROUND_PERIOD;

/**
 * The savanna floor: identically 0 across the running lane, gentle swells far out.
 *
 * Every x-dependent term is `sin(m · 2π x / GROUND_PERIOD + φ(z))` with integer m, so
 * `savannaHeight(x + GROUND_PERIOD, z)` and `savannaHeight(x, z)` agree to floating-point
 * evaluation error (~1e-9 — a nanometre of a metre-scale world). That is what lets the
 * ground mesh be snapped to period multiples and lets props read the height field in
 * absolute world coordinates.
 */
export function savannaHeight(x: number, z: number): number {
  const az = Math.abs(z);
  const lane = smoothstep01((az - LANE_HALF_Z) / LANE_FEATHER);
  if (lane <= 0) return 0;
  const amp = lane * Math.min(1, (az - LANE_HALF_Z) / ROLL_RAMP);
  if (amp <= 0) return 0;
  const k = TWO_PI_OVER_PERIOD * x;
  const roll =
    0.7 * Math.sin(k + 0.7 + 0.021 * z) * Math.cos(0.048 * z) +
    0.3 * Math.sin(2 * k + 2.3 + 0.075 * z) +
    0.14 * Math.sin(3 * k - 1.1) * Math.sin(0.062 * z + 0.5);
  // a shallow bowl far behind: the plain drops away instead of standing like a wall
  const away = z < 0 ? Math.min(1, -z / 60) : 0;
  return amp * (roll - 1.1 * away * away);
}

/** Ground patchiness, also exactly periodic in x (the vertex colors travel with the snap). */
function groundPatch(x: number, z: number, seed: number): number {
  // 6 lattice cells per period keeps the wrap exact: valueNoise2 tiles every `period`
  const u = (x / GROUND_PERIOD) * 6;
  return (
    0.62 * noiseTiled(u, z * 0.14, 6, seed) +
    0.38 * noiseTiled(u * 3, z * 0.42 + 11.7, 18, seed ^ 0x5f3a91)
  );
}

/** valueNoise2 with an x lattice that wraps every `period` cells (z is free). */
function noiseTiled(u: number, v: number, period: number, seed: number): number {
  const iu = Math.floor(u);
  const iv = Math.floor(v);
  const fu = fade(u - iu);
  const fv = fade(v - iv);
  const u0 = ((iu % period) + period) % period;
  const u1 = ((iu + 1) % period + period) % period;
  const n00 = hash2f(u0, iv, seed);
  const n10 = hash2f(u1, iv, seed);
  const n01 = hash2f(u0, iv + 1, seed);
  const n11 = hash2f(u1, iv + 1, seed);
  const a = n00 + (n10 - n00) * fu;
  const b = n01 + (n11 - n01) * fu;
  return a + (b - a) * fv;
}

function fade(t: number): number {
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------
// tile banks — the deterministic treadmill
// ---------------------------------------------------------------------------

/** write one instance; unwritten slots collapse to zero scale */
export type TileWriter = (
  x: number,
  y: number,
  z: number,
  yaw: number,
  scale: number,
  stretch: number,
  mix01: number,
) => void;

/** fill one tile's slots. Called ONLY with the tile's own deterministic stream. */
export type TileFill = (tile: number, rng: Rng, write: TileWriter) => void;

/**
 * A bank's IDENTITY as an integer — the `bank` coordinate of `rngAt(tile, bank, seed)`.
 *
 * It has to be a hash of the whole name: keyed by anything coarser (a name's LENGTH, say)
 * 'savannaGrass' and 'savannaRocks' would draw the identical per-tile stream, and "does
 * this tile have a rock" would become a deterministic function of where its first grass
 * clump sits. FNV-1a, the same mixing `kit/rng`'s own `streamFor` uses for named streams.
 */
export function bankKey(name: string): number {
  let h = 0x811c9dc5 | 0;
  for (let i = 0; i < name.length; i++) h = Math.imul(h ^ name.charCodeAt(i), 0x01000193) | 0;
  return h;
}

export interface TileBankHandle {
  mesh: THREE.InstancedMesh;
  /** re-window the bank around a tile index; a no-op when nothing moved */
  setOrigin(tile: number): void;
  dispose(): void;
}

export interface TileBankOptions {
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
  /** tiles kept alive (the window is [tile - back, tile - back + tiles)) */
  tiles: number;
  /** tiles of the window that sit BEHIND the runner */
  back: number;
  perTile: number;
  /** per-instance tint endpoints; mix01 lerps between them */
  colorA: THREE.ColorRepresentation;
  colorB: THREE.ColorRepresentation;
  fill: TileFill;
  castShadow?: boolean;
  receiveShadow?: boolean;
  /** THE LANE SEED (STYLE.md: kit/rng seeded with ctx.seed for ALL scatter) */
  seed: number;
  name: string;
}

/**
 * One InstancedMesh whose contents are a pure function of (tile index, bank, LANE SEED).
 * The rewrite happens ONLY when the window moves (a tile crossing, or a scrub), and it
 * re-derives every slot from `rngAt(tile, bank, seed)` — never from what was there before.
 * Two lanes of the gallery therefore grow different grass in different places, and no two
 * banks of one lane ever share a stream.
 */
export function makeTileBank(opts: TileBankOptions): TileBankHandle {
  const capacity = Math.max(1, opts.tiles * opts.perTile);
  const mesh = new THREE.InstancedMesh(opts.geometry, opts.material, capacity);
  mesh.name = opts.name;
  mesh.castShadow = opts.castShadow ?? false;
  mesh.receiveShadow = opts.receiveShadow ?? false;
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scaleVec = new THREE.Vector3();
  const colorA = new THREE.Color(opts.colorA);
  const colorB = new THREE.Color(opts.colorB);
  const color = new THREE.Color();
  const zero = new THREE.Matrix4().makeScale(0, 0, 0);

  let origin = Number.NaN;
  let cursor = 0;
  let limit = 0;

  const write: TileWriter = (x, y, z, yaw, scale, stretch, mix01) => {
    if (cursor >= limit) return;
    position.set(x, y, z);
    euler.set(0, yaw, 0);
    quaternion.setFromEuler(euler);
    scaleVec.set(scale, scale * stretch, scale);
    matrix.compose(position, quaternion, scaleVec);
    mesh.setMatrixAt(cursor, matrix);
    color.copy(colorA).lerp(colorB, mix01 < 0 ? 0 : mix01 > 1 ? 1 : mix01);
    mesh.setColorAt(cursor, color);
    cursor++;
  };

  // Slot assignment is `tile mod tiles`, which is what makes the treadmill cheap: walking
  // one tile forward rewrites ONE slot, not the whole bank. A scrub across the plain
  // rewrites everything, which is exactly right — and either way a slot's contents are a
  // pure function of the tile index that owns it.
  const slotTile = new Int32Array(opts.tiles).fill(0x7fffffff);
  const bank = bankKey(opts.name);

  const fillSlot = (slot: number, index: number): void => {
    cursor = slot * opts.perTile;
    limit = cursor + opts.perTile;
    opts.fill(index, rngAt(index, bank, opts.seed), write);
    for (let i = cursor; i < limit; i++) mesh.setMatrixAt(i, zero);
    slotTile[slot] = index;
  };

  const rebuild = (tile: number): void => {
    const first = tile - opts.back;
    let touched = false;
    for (let t = 0; t < opts.tiles; t++) {
      const index = first + t;
      const slot = ((index % opts.tiles) + opts.tiles) % opts.tiles;
      if (slotTile[slot] === index) continue;
      fillSlot(slot, index);
      touched = true;
    }
    if (!touched) return;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  };

  return {
    mesh,
    setOrigin(tile) {
      if (tile === origin) return;
      origin = tile;
      rebuild(tile);
    },
    dispose() {
      mesh.dispose();
      mesh.removeFromParent();
    },
  };
}

// ---------------------------------------------------------------------------
// bespoke geometry: the acacia and the cairn
// ---------------------------------------------------------------------------

/**
 * The savanna signature: a leaning trunk, three boughs, and a FLAT-TOPPED umbrella of
 * squashed canopy blobs. Read as a silhouette against a low sun, this shape is the
 * whole location — no kit archetype has it, so it is built here.
 */
export function acaciaGeometry(detail: number, bark: THREE.Color, leafLow: THREE.Color, leafHigh: THREE.Color): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const radial = detail > 0 ? 7 : 5;
  const trunk = new THREE.CylinderGeometry(0.075, 0.17, 1.75, radial, 1);
  trunk.translate(0, 0.87, 0);
  trunk.rotateZ(0.05);
  parts.push({ geometry: trunk, color: bark });
  const boughs: readonly (readonly [number, number, number])[] = [
    [0.62, 0.95, 1.15],
    [-0.72, 1.0, 2.5],
    [0.2, 0.8, 4.1],
  ];
  for (const [lean, len, yaw] of boughs) {
    const bough = new THREE.CylinderGeometry(0.045, 0.085, len, radial, 1);
    bough.translate(0, len * 0.5, 0);
    bough.rotateZ(lean > 0 ? -0.62 : 0.62);
    bough.rotateY(yaw);
    bough.translate(0, 1.6, 0);
    parts.push({ geometry: bough, color: bark });
  }
  // the umbrella: overlapping squashed blobs whose TOPS line up (that flat crown is
  // the acacia's whole silhouette), with the lower rim shaded
  const crown: readonly (readonly [number, number, number, number])[] = [
    [0, 2.36, 0, 1.72],
    [1.24, 2.22, 0.4, 1.06],
    [-1.16, 2.26, -0.5, 1.12],
    [0.24, 2.16, -1.16, 0.9],
    [-0.3, 2.14, 1.2, 0.84],
    [0.7, 2.44, -0.3, 0.72],
  ];
  for (const [x, y, z, r] of crown) {
    const blob = new THREE.IcosahedronGeometry(r, detail);
    blob.scale(1, 0.2, 1);
    blob.translate(x, y, z);
    parts.push({ geometry: blob, color: y > 2.3 ? leafHigh : leafLow });
  }
  return mergeColoredParts(parts, 'acacia');
}

/** A stacked stone cairn — the 10-unit distance ruler. */
export function cairnGeometry(seed: number, stone: THREE.Color, stoneLight: THREE.Color): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const rng = streamFor(seed, 'cairn');
  const tiers = 6;
  let stack = 0;
  for (let i = 0; i < tiers; i++) {
    const k = i / (tiers - 1);
    const r = 0.15 * (1 - k * 0.55);
    const h = 0.09 * (1 - k * 0.25);
    const g = new THREE.IcosahedronGeometry(r, 0);
    g.scale(1 + (rng() - 0.5) * 0.36, (h / r) * 1.2, 1 + (rng() - 0.5) * 0.36);
    g.rotateY(rng() * Math.PI * 2);
    g.translate((rng() - 0.5) * 0.05, stack + h * 0.5, (rng() - 0.5) * 0.05);
    parts.push({ geometry: g, color: i % 2 === 0 ? stone : stoneLight });
    stack += h;
  }
  return mergeColoredParts(parts, 'cairn');
}

/**
 * The far escarpment. Periodic in x exactly like the ground (integer harmonics of
 * 2π/RIDGE_PERIOD), so it snaps with the treadmill instead of sliding, and drawn as one
 * four-period strip 72 units back where the fog turns it to dusty mauve.
 */
export function ridgeHeight(x: number): number {
  const k = (Math.PI * 2 * x) / RIDGE_PERIOD;
  return (
    5.2 +
    2.4 * Math.sin(k + 0.4) +
    1.7 * Math.sin(2 * k + 2.1) +
    1.1 * Math.sin(3 * k - 0.7) +
    0.7 * Math.sin(5 * k + 1.9) +
    0.4 * Math.sin(8 * k - 2.4)
  );
}

function ridgeGeometry(low: THREE.Color, high: THREE.Color): THREE.BufferGeometry {
  const n = RIDGE_SEGMENTS;
  const span = RIDGE_PERIOD * 4;
  const positions = new Float32Array((n + 1) * 2 * 3);
  const colors = new Float32Array((n + 1) * 2 * 3);
  const indices = new Uint16Array(n * 6);
  for (let i = 0; i <= n; i++) {
    const x = -span / 2 + (span * i) / n;
    const h = ridgeHeight(x);
    const top = i * 2;
    const bottom = top + 1;
    positions[top * 3] = x;
    positions[top * 3 + 1] = h;
    positions[top * 3 + 2] = 0;
    positions[bottom * 3] = x;
    positions[bottom * 3 + 1] = -6;
    positions[bottom * 3 + 2] = 0;
    const k = Math.min(1, Math.max(0, (h - 3) / 6));
    colors[top * 3] = low.r + (high.r - low.r) * k;
    colors[top * 3 + 1] = low.g + (high.g - low.g) * k;
    colors[top * 3 + 2] = low.b + (high.b - low.b) * k;
    colors[bottom * 3] = low.r;
    colors[bottom * 3 + 1] = low.g;
    colors[bottom * 3 + 2] = low.b;
  }
  let o = 0;
  for (let i = 0; i < n; i++) {
    const a = i * 2;
    indices[o++] = a;
    indices[o++] = a + 1;
    indices[o++] = a + 2;
    indices[o++] = a + 2;
    indices[o++] = a + 1;
    indices[o++] = a + 3;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'savannaRidge';
  return g;
}

/** A distant bird: two swept wings and a body, flat in the x-y plane (canon lesson 11). */
function birdGeometry(): THREE.BufferGeometry {
  const v = new Float32Array([
    // left wing (two triangles, swept back and down at the tip)
    0, 0, 0, -0.62, 0.2, 0, -0.5, -0.02, 0,
    0, 0, 0, -0.5, -0.02, 0, -0.12, -0.05, 0,
    // right wing
    0, 0, 0, 0.5, -0.02, 0, 0.62, 0.2, 0,
    0, 0, 0, 0.12, -0.05, 0, 0.5, -0.02, 0,
  ]);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(v, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'savannaBird';
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
  /** re-window every treadmill element around the runner's world x */
  setOrigin(rootX: number): void;
  /** ctx.time() — pause freezes wind, haze, pollen and the birds together */
  setTime(t: number): void;
  /** 0..1 of the runner's top speed: grass lean, haze drift, bird urgency */
  setSpeed(speed01: number): void;
  /** aim the tight shadow box at the runner (and quantize it so texels stop swimming) */
  setShadowTarget(x: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

interface Disposable {
  dispose(): void;
}

interface Timed {
  setTime(t: number): void;
}

const BIRDS = 4;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const tier: QualityTier = ctx.tier;
  const palette: Palette = ctx.tokens.palettes.dusk;
  const group = new THREE.Group();
  group.name = 'goldHourSavanna';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const shadows = tier !== 'thumb';
  const veg = tier === 'thumb' ? 0.24 : tier === 'medium' ? 0.5 : 1;

  // ---- palette-derived surface colors (blends of tokens only) ----
  const cDust = blendTokens(blendTokens(palette.groundA, palette.hemiGround, 0.72), TOKEN_BLACK, 0.4);
  const cStraw = blendTokens(blendTokens(palette.groundB, palette.hemiGround, 0.56), TOKEN_BLACK, 0.2);
  const cGrassLow = blendTokens(blendTokens(palette.groundA, TOKENS.palettes.noon.groundA, 0.34), TOKEN_BLACK, 0.4);
  const cGrassHigh = blendTokens(blendTokens(palette.groundB, palette.sun, 0.45), TOKEN_WHITE, 0.26);
  const cRock = blendTokens(palette.hemiGround, palette.skyTop, 0.4);
  const cStone = blendTokens(palette.hemiGround, palette.skyTop, 0.52);
  const cStoneLight = blendTokens(cStone, palette.sun, 0.22);
  const cBark = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.62);
  const cLeafLow = blendTokens(TOKENS.palettes.noon.groundA, TOKEN_BLACK, 0.8);
  const cLeafHigh = blendTokens(TOKENS.palettes.noon.groundA, TOKEN_BLACK, 0.6);

  // ---- sky (exactly one) ----
  // A big, soft, low disc: this is the "gold hour" itself. The glow power is deliberately
  // low so the warm scatter spills a third of the way across the sky even when the disc
  // sits just outside the frame.
  const sky = skyFromPalette(palette, 'dusk', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.05,
    sunHdr: 4.6,
    sunGlow: 0.6,
    sunGlowPower: 3.4,
    topSpread: 0.2,
    bottomSpread: 0.3,
    radius: 320,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the one light budget ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 2,
    pointDistance: 16,
    sunDistanceScale: 3.2,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: 0.35, z: 0 }, SHADOW_RADIUS);

  // ---- fog, always ----
  // Real aerial perspective is BLUE-shifted: without this the amber fog painted every
  // distance the same hue as the foreground and the frame read as one wash.
  const fog = applyFog(scene, palette, FOG_EXTENT);
  fog.color.copy(blendTokens(palette.fog, palette.skyTop, 0.34));

  // ---- the ground: exactly flat where the toes land ----
  const rampColor = new THREE.Color();
  const terrain = makeTerrain({
    size: [GROUND_W, GROUND_D],
    segments: [GROUND_SEG_X, GROUND_SEG_Z],
    tier,
    seed: ctx.seed,
    heightFn: savannaHeight,
    jitter: 0, // accuracy first: the mesh must stay on savannaHeight
    colorNoise: 0.03,
    slopeColor: cDust,
    slopeStrength: 0.7,
    slopeSharpness: 2,
    colorRamp: (h, _slope, x, z) => {
      const patch = groundPatch(x, z, ctx.seed);
      // sun-bleached straw on the crests, warmer dust in the hollows and on the lane
      const dry = smoothstep01(patch * 1.5 - 0.28 + h * 0.35);
      rampColor.copy(cDust).lerp(cStraw, dry);
      // the lane itself is trodden earth, a shade darker than the plain
      const lane = 1 - smoothstep01((Math.abs(z) - LANE_HALF_Z * 0.6) / 1.5);
      if (lane > 0) rampColor.lerp(cDust, lane * 0.3);
      return rampColor;
    },
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.22,
      toonSteps: 3,
      toonSoftness: 0.34,
      toonFloor: 0.26,
    },
    receiveShadow: shadows,
    castShadow: false,
    name: 'savannaGround',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  // ---- one material per bank; every one of them is an oakMaterial ----
  const grassMaterial: OakMaterial = oakMaterial({
    color: TOKEN_WHITE, // the real hue rides on instanceColor (canon lesson 13)
    vertexColors: true,
    flatShading: true,
    roughness: 0.92,
    rimColor: palette.sun,
    rimStrength: 0.75,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.3,
    wind: { amplitude: 0.05, frequency: 1.5, height: 0.42 },
    side: THREE.DoubleSide,
    tier,
    name: 'savannaGrassMaterial',
  });
  const treeMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: palette.sun,
    rimStrength: 0.5,
    toonSteps: 2,
    toonSoftness: 0.3,
    toonFloor: 0.14,
    wind: { amplitude: 0.05, frequency: 0.62, height: 3.1 },
    tier,
    name: 'savannaAcaciaMaterial',
  });
  const stoneMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.96,
    rimColor: palette.sun,
    rimStrength: 0.62,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.2,
    tier,
    name: 'savannaStoneMaterial',
  });
  const animated: OakMaterial[] = [grassMaterial, treeMaterial];
  timed.push({ setTime: (t) => setMaterialsTime(animated, t) });
  owned.push({
    dispose: () => {
      grassMaterial.dispose();
      treeMaterial.dispose();
      stoneMaterial.dispose();
    },
  });

  // ---- grass: dense along the lane, thinning into the fog ----
  const grassGeometry = grassClumpGeometry(tier === 'thumb' ? 2 : 5, 0.42, 0.05);
  owned.push({ dispose: () => grassGeometry.dispose() });
  const grassPerTile = Math.max(8, Math.round(520 * veg));
  const grassBank = makeTileBank({
    geometry: grassGeometry,
    material: grassMaterial,
    tiles: 15,
    back: 5,
    perTile: grassPerTile,
    colorA: cGrassLow,
    colorB: cGrassHigh,
    seed: ctx.seed,
    name: 'savannaGrass',
    fill: (tile, rng, write) => {
      const x0 = tile * TILE;
      for (let i = 0; i < grassPerTile; i++) {
        const x = x0 + rng() * TILE;
        const u = rng();
        const roll = rng();
        // Three bands, and the split is a COMPOSITION decision, not decoration:
        //  * fringe  — the bottom edge of the frame, tall and back-lit, the foreground
        //              a wildlife shot is framed with (the camera clips it at z ≈ 1.8)
        //  * verge   — low tufts either side of the running line, never masking a leg
        //  * plain   — everything behind, thinning into the fog
        let z: number;
        let scale: number;
        let mix: number;
        if (roll < 0.16) {
          z = 1.5 + u * 1.2;
          scale = 0.62 * (0.55 + rng() * 0.8);
          mix = rng() * 0.3;
        } else if (roll < 0.34) {
          z = LANE_HALF_Z + 0.12 + u * 0.85;
          scale = 0.24 * (0.5 + rng() * 0.9);
          mix = 0.2 + rng() * 0.5;
        } else {
          z = -(LANE_HALF_Z + 0.08 + Math.pow(u, 1.6) * 32);
          scale = 0.5 * (0.5 + rng() * 0.85);
          mix = 0.25 + rng() * 0.75;
        }
        const near = 1 - Math.min(1, Math.abs(z) / 9);
        write(x, savannaHeight(x, z), z, rng() * Math.PI, scale * (0.75 + 0.4 * near), 0.7 + rng() * 0.8, mix);
      }
    },
  });
  group.add(grassBank.mesh);
  owned.push(grassBank);

  // ---- acacias: never near the lane, never a shadow caster (they are 20 units of
  // silhouette at this sun angle, and the shadow box belongs to the runner) ----
  const acaciaGeo = acaciaGeometry(tier === 'thumb' ? 0 : 1, cBark, cLeafLow, cLeafHigh);
  owned.push({ dispose: () => acaciaGeo.dispose() });
  const acaciaBank = makeTileBank({
    geometry: acaciaGeo,
    material: treeMaterial,
    tiles: 15,
    back: 5,
    perTile: 4,
    colorA: new THREE.Color(1, 1, 1),
    colorB: blendTokens(TOKEN_WHITE, palette.sun, 0.35),
    seed: ctx.seed,
    name: 'savannaAcacias',
    fill: (tile, rng, write) => {
      const x0 = tile * TILE;
      for (let i = 0; i < 4; i++) {
        if (rng() > 0.5) continue;
        const x = x0 + rng() * TILE;
        const u = rng();
        const z = -(13 + Math.pow(u, 1.35) * 40);
        write(x, savannaHeight(x, z), z, rng() * Math.PI * 2, 0.42 + rng() * 0.62, 0.85 + rng() * 0.6, rng());
      }
    },
  });
  group.add(acaciaBank.mesh);
  owned.push(acaciaBank);

  // ---- rocks and termite hummocks ----
  const rockGeo = rockGeometry(1.7, tier === 'hero' ? 1 : 0);
  owned.push({ dispose: () => rockGeo.dispose() });
  const rockBank = makeTileBank({
    geometry: rockGeo,
    material: stoneMaterial,
    tiles: 15,
    back: 5,
    perTile: 3,
    colorA: cRock,
    colorB: blendTokens(cRock, palette.sun, 0.3),
    castShadow: shadows,
    seed: ctx.seed,
    name: 'savannaRocks',
    fill: (tile, rng, write) => {
      const x0 = tile * TILE;
      for (let i = 0; i < 3; i++) {
        if (rng() > 0.62) continue;
        const x = x0 + rng() * TILE;
        const u = rng();
        const mound = rng() < 0.3;
        const scale = mound ? 0.7 + rng() * 0.9 : 0.3 + rng() * 0.8;
        const stretch = mound ? 1.7 + rng() * 1.1 : 0.5 + rng() * 0.5;
        // A prop may never stand INSIDE the runner's silhouette. The kit rock tops out at
        // ~0.24 of its unit size, so a full termite hummock is taller than the animal —
        // its distance from the lane therefore scales with its own height, and only
        // ankle-high stones are allowed on the camera side, where anything taller would
        // wipe the hero out of its own frame. Little stones still lie right by the verge.
        const height = 0.24 * scale * stretch;
        const side = height < 0.25 && rng() < 0.3 ? 1 : -1;
        const z = side * (LANE_HALF_Z + 0.6 + height * 3.6 + u * u * 22);
        write(x, savannaHeight(x, z) - 0.04, z, rng() * Math.PI * 2, scale, stretch, rng());
      }
    },
  });
  group.add(rockBank.mesh);
  owned.push(rockBank);

  // ---- cairns every CAIRN_SPACING units: the distance ruler ----
  const cairnGeo = cairnGeometry(ctx.seed, cStone, cStoneLight);
  owned.push({ dispose: () => cairnGeo.dispose() });
  const CAIRNS = 9;
  const cairns = new THREE.InstancedMesh(cairnGeo, stoneMaterial, CAIRNS);
  cairns.name = 'savannaCairns';
  cairns.castShadow = shadows;
  cairns.frustumCulled = false;
  cairns.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(CAIRNS * 3), 3);
  {
    const c = new THREE.Color();
    for (let i = 0; i < CAIRNS; i++) {
      c.setRGB(1, 1, 1);
      cairns.setColorAt(i, c);
    }
  }
  group.add(cairns);
  owned.push({
    dispose: () => {
      cairns.dispose();
      cairns.removeFromParent();
    },
  });

  // ---- the far escarpment ----
  const ridgeGeo = ridgeGeometry(
    blendTokens(palette.skyTop, TOKEN_BLACK, 0.55),
    blendTokens(palette.skyTop, palette.hemiGround, 0.2),
  );
  const ridgeMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 1,
    toonSteps: 1,
    toonFloor: 0.55,
    rimStrength: 0,
    side: THREE.DoubleSide,
    tier,
    name: 'savannaRidgeMaterial',
  });
  const ridge = new THREE.Mesh(ridgeGeo, ridgeMaterial);
  ridge.name = 'savannaRidge';
  ridge.position.z = RIDGE_Z;
  ridge.frustumCulled = false;
  group.add(ridge);
  owned.push({
    dispose: () => {
      ridgeGeo.dispose();
      ridgeMaterial.dispose();
      ridge.removeFromParent();
    },
  });

  // ---- the cairn markers' cloth: the one saturated accent in an amber frame, and a
  // deliberate echo of the exemplar's summit prayer flags ----
  const clothParts: ColoredPart[] = [];
  {
    const pole = new THREE.CylinderGeometry(0.016, 0.022, 0.46, 5);
    pole.translate(0, 0.46, 0);
    clothParts.push({ geometry: pole, color: blendTokens(palette.hemiGround, palette.groundB, 0.4) });
    const cloth = new THREE.PlaneGeometry(0.17, 0.12, 1, 1);
    cloth.translate(0.088, 0.62, 0);
    clothParts.push({ geometry: cloth, color: TOKEN_WHITE });
  }
  const clothGeo = mergeColoredParts(clothParts, 'savannaMarkerCloth');
  const clothMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.9,
    rimColor: palette.sun,
    rimStrength: 0.8,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.34,
    side: THREE.DoubleSide,
    wind: { amplitude: 0.02, frequency: 2.6, height: 0.6 },
    tier,
    name: 'savannaMarkerMaterial',
  });
  animated.push(clothMaterial);
  const markers = new THREE.InstancedMesh(clothGeo, clothMaterial, CAIRNS);
  markers.name = 'savannaMarkers';
  markers.frustumCulled = false;
  markers.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(CAIRNS * 3), 3);
  group.add(markers);
  owned.push({
    dispose: () => {
      markers.dispose();
      clothGeo.dispose();
      clothMaterial.dispose();
      markers.removeFromParent();
    },
  });
  // Prayer-flag cloth on the cairns. The cool flag used to be `tokens.uiCyan` — UI chrome
  // painted onto a world surface, and against a 174-sRGB amber plain that reads as an
  // artifact, not as cloth. It is now this scene's OWN sky: dusk skyTop carried a little
  // way into the dusk fog and then down toward `night.skyTop` (canon lesson 9's legal
  // near-black) until it sits under the plain's value. Cool, weathered, and part of the
  // palette. The warm flags (gold, garden green, dawn blossom) are unchanged — they are
  // what makes a cairn read as a marker at 60 m.
  const cairnCloth = blendTokens(blendTokens(palette.skyTop, palette.fog, 0.22), TOKEN_BLACK, 0.2);
  const MARKER_COLORS: readonly THREE.ColorRepresentation[] = [
    cairnCloth,
    ctx.tokens.celebrationGold,
    ctx.tokens.accents['four-rooms'] ?? ctx.tokens.celebrationGold,
    ctx.tokens.accents['acrobot'] ?? cairnCloth,
  ];

  // ---- atmosphere ----
  // Kit particle/mist presets are tuned for ~20-unit fields (canon lesson 5); this one is
  // a 40-unit plain seen from 5 units away, so pollen sits near the preset size and the
  // haze planes are big and very thin.
  const haze: MistPlanesHandle = makeMistPlanes({
    size: [96, 54],
    color: blendTokens(palette.fog, palette.sun, 0.35),
    seed: ctx.seed,
    tier,
    layers: 5,
    baseY: 0.25,
    spread: 3.4,
    opacity: 0.2,
    speed: 0.004,
    scale: 3.2,
    nearFade: 14,
    name: 'savannaHaze',
  });
  haze.group.position.z = -38;
  group.add(haze.group);
  owned.push(haze);
  timed.push(haze);

  const pollen: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'pollen',
    count: 460,
    bounds: { y: 1.25, halfX: 8, halfY: 1.5, halfZ: 6 },
    seed: ctx.seed,
    tier,
    color: blendTokens(palette.sun, TOKEN_WHITE, 0.3),
    size: 0.75,
    intensity: 0.9,
    opacity: 0.34,
    rise: 0.05,
    swirl: 0.22,
    name: 'savannaPollen',
  });
  group.add(pollen.points);
  owned.push(pollen);
  timed.push(pollen);

  // ---- distant birds: silhouettes on the warm side of the sky ----
  const birdGeo = birdGeometry();
  const birdMaterial = oakMaterial({
    color: blendTokens(TOKEN_BLACK, palette.skyTop, 0.3),
    flatShading: true,
    roughness: 1,
    rimColor: palette.sun,
    rimStrength: 0.3,
    toonSteps: 2,
    toonFloor: 0.12,
    side: THREE.DoubleSide,
    fog: false,
    tier,
    name: 'savannaBirdMaterial',
  });
  const birdGroup = new THREE.Group();
  birdGroup.name = 'savannaBirds';
  const birds: THREE.Mesh[] = [];
  const birdWings: THREE.Object3D[] = [];
  if (tier !== 'thumb') {
    for (let i = 0; i < BIRDS; i++) {
      const pivot = new THREE.Object3D();
      pivot.name = `savannaBirdPivot${i}`;
      const mesh = new THREE.Mesh(birdGeo, birdMaterial);
      mesh.name = `savannaBird${i}`;
      pivot.add(mesh);
      birdGroup.add(pivot);
      birds.push(mesh);
      birdWings.push(pivot);
    }
  }
  group.add(birdGroup);
  owned.push({
    dispose: () => {
      birdGeo.dispose();
      birdMaterial.dispose();
      birdGroup.clear();
    },
  });

  // ---- treadmill state (all preallocated; setOrigin allocates nothing) ----
  const cairnMatrix = new THREE.Matrix4();
  const cairnPos = new THREE.Vector3();
  const cairnQuat = new THREE.Quaternion();
  const cairnEuler = new THREE.Euler();
  const cairnScale = new THREE.Vector3();
  const markerColor = new THREE.Color();
  let cairnOrigin = Number.NaN;

  const placeCairns = (centre: number): void => {
    for (let i = 0; i < CAIRNS; i++) {
      const k = centre + i - ((CAIRNS - 1) >> 1);
      const x = k * CAIRN_SPACING;
      // two ranks, both BEHIND the running line: a rhythm you pass, never a fence you
      // look through, and never a prop between the camera and the animal
      const side = (k & 1) === 0 ? -1.5 : -3.1;
      const z = side + (hash1f(k, ctx.seed) - 0.5) * 0.5;
      cairnPos.set(x, savannaHeight(x, z) - 0.02, z);
      cairnEuler.set(0, hash1f(k, ctx.seed ^ 0x51ab) * Math.PI * 2, 0);
      cairnQuat.setFromEuler(cairnEuler);
      const s = 0.5 + hash1f(k, ctx.seed ^ 0x2b17) * 0.32;
      cairnScale.set(s, s * (0.85 + hash1f(k, ctx.seed ^ 0x77) * 0.5), s);
      cairnMatrix.compose(cairnPos, cairnQuat, cairnScale);
      cairns.setMatrixAt(i, cairnMatrix);

      cairnPos.set(x - 0.05, savannaHeight(x - 0.05, z - 0.06), z - 0.06);
      cairnEuler.set(0, hash1f(k, ctx.seed ^ 0x99) * Math.PI * 2, 0);
      cairnQuat.setFromEuler(cairnEuler);
      const ms = 0.8 + hash1f(k, ctx.seed ^ 0x1234) * 0.35;
      cairnScale.set(ms, ms, ms);
      cairnMatrix.compose(cairnPos, cairnQuat, cairnScale);
      markers.setMatrixAt(i, cairnMatrix);
      markerColor.set(MARKER_COLORS[((k % MARKER_COLORS.length) + MARKER_COLORS.length) % MARKER_COLORS.length] ?? TOKEN_WHITE);
      markers.setColorAt(i, markerColor);
    }
    cairns.instanceMatrix.needsUpdate = true;
    markers.instanceMatrix.needsUpdate = true;
    if (markers.instanceColor) markers.instanceColor.needsUpdate = true;
  };

  // The shadow box quantum: one 4-texel step in world units. Without it, a focus that
  // slides continuously makes every shadow edge crawl against the ground.
  const shadowQuantum = (2 * SHADOW_RADIUS * 4) / Math.max(256, tier === 'hero' ? 2048 : 1024);
  const shadowOffsetX = -sky.sunDirection.x * 1.1;
  const shadowOffsetZ = -sky.sunDirection.z * 1.1;

  const world: WorldHandle = {
    group,
    sky,
    rig,
    terrain,

    setOrigin(rootX) {
      terrain.mesh.position.x = Math.round(rootX / GROUND_PERIOD) * GROUND_PERIOD;
      const tile = Math.floor(rootX / TILE);
      grassBank.setOrigin(tile);
      acaciaBank.setOrigin(tile);
      rockBank.setOrigin(tile);
      const cairnCentre = Math.round(rootX / CAIRN_SPACING);
      if (cairnCentre !== cairnOrigin) {
        cairnOrigin = cairnCentre;
        placeCairns(cairnCentre);
      }
      ridge.position.x = Math.round(rootX / RIDGE_PERIOD) * RIDGE_PERIOD;
      haze.group.position.x = rootX;
      pollen.points.position.x = rootX;
      birdGroup.position.x = rootX;
    },

    setShadowTarget(x) {
      const q = Math.round(x / shadowQuantum) * shadowQuantum;
      rig.setShadowFocus({ x: q + shadowOffsetX, y: 0.3, z: shadowOffsetZ }, SHADOW_RADIUS);
    },

    setSpeed(v) {
      const s = clampRange(v, 0, 1);
      oakUniforms(grassMaterial)?.setWind(0.05 * (1 + 1.9 * s), 1.5 + 2.2 * s);
      oakUniforms(treeMaterial)?.setWind(0.05 * (1 + 0.9 * s), 0.62 + 0.5 * s);
      pollen.setDensity(0.45 + 0.55 * s);
    },

    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
      // birds: a pure function of (t, rootX) — they drift, they never accumulate
      for (let i = 0; i < birdWings.length; i++) {
        const pivot = birdWings[i];
        const mesh = birds[i];
        if (!pivot || !mesh) continue;
        const span = 96;
        const drift = (t * (0.5 + 0.22 * i) + i * 31.7 + hash1f(i, ctx.seed) * span) % span;
        const bx = drift - span * 0.5;
        pivot.position.set(bx, 8.5 + i * 2.1 + Math.sin(t * 0.21 + i) * 0.9, -34 - i * 4.5);
        const flap = Math.sin(t * (2.4 + 0.4 * i) + i * 2.1);
        pivot.rotation.z = flap * 0.26;
        mesh.scale.set(1.6 + 0.35 * i, 1 + 0.28 * flap, 1);
      }
    },

    setPixelRatio(dpr) {
      pollen.setPixelRatio(dpr);
    },

    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      birds.length = 0;
      birdWings.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };

  world.setOrigin(0);
  world.setShadowTarget(0);
  world.setSpeed(0);
  world.setTime(0);
  return world;
}
