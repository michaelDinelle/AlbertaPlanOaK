// The Firefly Cave — the static world (MemoryChain-cont-v0).
//
// THE ONE IDEA OF THIS SCENE: the cue is the only colour in the cave, and it exists only
// where the agent can actually see it. Everything built in this file is therefore
// deliberately COLOURLESS — wet grey-blue rock, black water, one green accent of
// fireflies. The cue crystal (actors.ts) and the two exit mouths (actors.ts) are the only
// objects allowed to carry `cueCyan` / `cueMagenta`, and the crystal only carries it while
// `state.pos === 0`. A tinted firefly, a tinted wisp ribbon or a lingering coloured pool
// would leak the hidden variable straight into the picture, which is the one thing this
// env is about.
//
// GEOMETRY: the env is a 32 × 1 grid, so `cellToWorld(pos, 0)` puts the passage on the x
// axis from −15.5 to +15.5 with z = 0. Every element here is placed against that axis and
// nothing re-derives it: the walkway is `buildCellField` straight off `grid.walls`, the
// cavern is swept along the same axis, and the water plane is sized from `gridExtent`.
//
// THE CUTAWAY IS FREE. The cavern is a closed arch swept along the passage whose triangle
// winding puts every normal INWARD. With `side: FrontSide`, the half of the shell between
// the camera and the passage is back-facing and disappears — so the scene is a genuine
// enclosed cave from the inside, and an open diorama from wherever the camera happens to
// stand. Orbit to the other bank and the cutaway follows you. No clipping planes, no
// second material, no camera-dependent geometry.

import * as THREE from 'three';
import {
  CELL,
  buildCellField,
  cellToWorld,
  gridExtent,
  type CellFieldHandle,
  type GridSize,
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms, type OakMaterial } from '@kit/materials';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { hash2f, streamFor } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import { excludeBand, scatterRocks } from '@kit/vegetation';
import { makeWater, type WaterHandle } from '@kit/water';
import { MEMORYCHAIN_END, MEMORYCHAIN_L, memoryChainDecoder } from '@sim/envs/memorychain';
import type { EnvGeometry, GridGeometry } from '@sim/types';

// ---------------------------------------------------------------------------
// world constants — every one of them derived from the 32 × 1 passage
// ---------------------------------------------------------------------------

/** half the passage footprint along x: 16 world units either side of the origin */
export const PASSAGE_HALF = (MEMORYCHAIN_L * CELL) / 2;
/** the cavern runs a little past both ends so neither end of the world is a raw edge */
export const CAVERN_X0 = -PASSAGE_HALF - 6.5;
export const CAVERN_X1 = PASSAGE_HALF + 5.5;
/** cavern half-width at the waterline (the water plane must stay inside this) */
export const CAVERN_HALF_W = 5.7;
/** vault height at the crown */
export const CAVERN_HEIGHT = 9.2;
/** how far the rock skirt continues below the waterline, so no edge is ever visible */
export const CAVERN_SKIRT = 1.6;
/** still water, a hand's breadth below the walkway stones */
export const WATER_LEVEL = -0.11;
/**
 * The water plane. It is deliberately WIDER than the cavern: on the far bank the rock wall
 * stands between the camera and the overshoot so none of it is ever seen, and on the near
 * bank — the half back-face culling opens up — the overshoot is what fills the bottom of
 * the frame with black water instead of the hard cut edge of a 9-unit plane.
 */
export const WATER_HALF_Z = 14;
/** stalagmites and the walkway stay inside the cavern proper, whatever the water does */
export const POOL_HALF_Z = 4.5;

/**
 * Fog reach. `applyFog` turns extent into density through the cave palette's
 * `fogDensityHint` (0.75), so this puts full density ~75 units out — which leaves the far
 * end of a 40-unit passage about two thirds swallowed. That is the point: the CHOICE is
 * supposed to be a pair of dim glows in the dark, not a signposted exit. The kit's
 * emissive and wisp materials are unfogged by construction, so the two mouths and the
 * avatar still punch through the dark at 40 units while every rock surface sinks.
 */
export const FOG_EXTENT = 100;

/**
 * The cave's one directional light: a cold raking glow from above and behind the crystal
 * end, as if a crack in the roof let a sliver of night sky in. 41° up (canon lesson 8:
 * below ~25° a light rakes shadows across the subject) and aimed down the passage, so the
 * far wall picks up a faint edge and the vault stays black.
 */
export const SUN_DIR: readonly [number, number, number] = [-0.42, 0.66, 0.62];
/** the shadow frustum covers the near half of the passage, where the light actually is */
export const SHADOW_CENTER: readonly [number, number, number] = [-6, 0.4, 0];
export const SHADOW_RADIUS = 13;

/**
 * The composition azimuth, shared with `index.ts`'s OrbitRig so the planting and the
 * camera can never disagree. The camera stands on the +z bank looking down the passage;
 * that is also which half of the cavern shell back-face culling opens up.
 */
export const CAMERA_AZIMUTH = -0.935;

// ---------------------------------------------------------------------------
// small shared helpers
// ---------------------------------------------------------------------------

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

/** A scene-local colour: STYLE.md allows only BLENDS of token colours. */
export function blendTokens(
  a: THREE.ColorRepresentation,
  b: THREE.ColorRepresentation,
  k: number,
): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), k);
}

/**
 * Take a token blend down into the dark. `night.skyTop` is the darkest token there is and
 * STYLE.md names it as the legal way to reach near-black.
 *
 * Canon lesson 19 in reverse: the cave key light is 0.2 — one fifteenth of noon — so the
 * failure mode here is the opposite of FourRooms'. Rock albedo has to sit HIGH (a wet
 * limestone around 90–120 sRGB) or the whole frame is one black rectangle; what must stay
 * dark is the LIGHTING, not the paint.
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
 * disposing the sources (the four-rooms `mergeColoredParts` pattern — a hand-built stone
 * prop wants its source normals plus a per-part palette baked in, so one mesh and one
 * material draw the whole thing).
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
 * is the DECODER'S OWN geometry builder, never a scene re-derivation, so a placeholder
 * geometry degrades to the correct 32-cell passage instead of a guess.
 */
export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === MEMORYCHAIN_L && g.h === 1) return g;
  const truth = memoryChainDecoder.geometry(memoryChainDecoder.initStatic(0), 0);
  return truth.grid ?? { w: MEMORYCHAIN_L, h: 1, walls: new Uint8Array(MEMORYCHAIN_L) };
}

/** The cell the cue crystal stands on: the env's own start of the corridor. */
export const CUE_CELL = 0;
/** The cell the two exit mouths stand on: the env's own choice cell. */
export const CHOICE_CELL = MEMORYCHAIN_END;

// ---------------------------------------------------------------------------
// the cave palette (every colour a blend of tokens — STYLE.md)
// ---------------------------------------------------------------------------

export interface CaveSkin {
  /** the walkway stones */
  stone: THREE.Color;
  /** the cavern wall low down, near the waterline */
  rockLow: THREE.Color;
  /** the cavern wall high up in the vault */
  rockHigh: THREE.Color;
  /** stalactites and stalagmites — wet mineral, a shade cooler than the wall */
  mineral: THREE.Color;
  /** still water */
  waterShallow: THREE.Color;
  /** the deep of the pools */
  waterDeep: THREE.Color;
  /** the firefly accent */
  firefly: THREE.Color;
  /** the neutral quartz of the cue crystal when it is DARK (never a cue colour) */
  quartz: THREE.Color;
  /** the cold sky-silver every rock surface catches on its fresnel rim */
  rim: THREE.Color;
}

/**
 * Wet grey-blue limestone, from the cave palette plus the fog-silver of `po-maze` for the
 * mineral highlights. Nothing here is warm and nothing here is cyan or magenta: the whole
 * cave is a neutral ground against which one crystal is allowed to be a colour.
 */
/**
 * How much extra albedo a tier needs to survive what the kit takes away from it.
 *
 * This is the one place the cave has to fight its own quality tiers. At medium the rig
 * drops to two point lights and the fresnel rim to 0.75×; at thumb there are NO point
 * lights, NO fresnel rim, no particles and no bloom — a palette whose key light is 0.2
 * then has literally nothing left to see by, and the first thumb render was a black
 * rectangle with a pink crystal in the corner. Brighter rock is the only lever a scene
 * owns, so the rock gets brighter exactly as fast as the lighting is taken away. The water
 * is deliberately NOT lifted: it is supposed to be the black the stones read against.
 */
export function tierLift(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 0.34 : tier === 'medium' ? 0.1 : 0;
}

/** Toon floor per tier — the same compensation, applied to the shadowed side. */
export function tierToonFloor(tier: QualityTier | undefined, base: number): number {
  return tier === 'thumb' ? Math.max(base, 0.62) : tier === 'medium' ? base + 0.06 : base;
}

/**
 * THE BOUNCE. A cave has no sky to bounce off, and this palette's key is 0.2 against a
 * 0.35 hemisphere — arithmetic that puts every rock face at roughly 0.003 linear before a
 * point light touches it. At hero that is fine, because four point lights and a cloud of
 * HDR fireflies carry the frame. At THUMB the tier contract takes every point light, every
 * particle, the fresnel rim and the bloom away at once, and no albedo short of pure white
 * can survive that: the first thumb render was a black rectangle with a pink crystal in it.
 *
 * So each rock material carries a small self-emission in its OWN colour — the standard way
 * to stand in for the indirect bounce a real cave gets off its own walls. It is far below
 * the 1.0 bloom threshold at every tier (STYLE.md: only HDR emissives bloom), it preserves
 * the value hierarchy because each surface emits its own albedo, and it scales with exactly
 * what the tier took away.
 */
export function tierBounce(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 0.085 : tier === 'medium' ? 0.05 : 0.03;
}

export function caveSkin(palette: Palette, tokens: typeof TOKENS, tier?: QualityTier): CaveSkin {
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const firefly = tokens.accents['memory-chain'] ?? palette.accentLight;
  const lift = tierLift(tier);
  const up = (c: THREE.Color): THREE.Color => (lift > 0 ? c.lerp(new THREE.Color(silver), lift) : c);
  return {
    // These albedos look absurdly pale written down (~0.3–0.5 linear, sRGB 150–190) and
    // they are the whole reason the cave is legible: multiplied by a 0.2 key and a 0.35
    // hemisphere they land at sRGB 60–110 on screen. The first pass used FourRooms-style
    // shaded rock and rendered one black rectangle — see `shade()`.
    // The WALKWAY is the brightest surface in the cave and the wall is a step below it:
    // the path the agent walks has to win the eye against 40 units of rock face.
    stone: up(blendTokens(palette.groundB, silver, 0.64)),
    rockLow: up(blendTokens(palette.groundB, silver, 0.38)),
    // the vault is the one place that IS allowed to fall away into black
    rockHigh: up(blendTokens(palette.groundA, silver, 0.09)),
    mineral: up(blendTokens(palette.groundB, silver, 0.76)),
    // ...and the water is the darkest thing in the cave, so the stones read as a ribbon
    waterShallow: shade(blendTokens(palette.groundA, palette.hemiSky, 0.45), 0.5),
    waterDeep: shade(palette.skyBottom, 0.68),
    firefly: new THREE.Color(firefly),
    quartz: up(blendTokens(palette.groundB, silver, 0.6)),
    /** the rim tint every rock surface catches: cold sky-silver, never a cue colour */
    rim: blendTokens(palette.hemiSky, silver, 0.55),
  };
}

// ---------------------------------------------------------------------------
// the cavern shell
// ---------------------------------------------------------------------------

/** ring samples across the arch, per tier */
function archSamples(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 7 : tier === 'medium' ? 13 : 19;
}

/** sections along the passage, per tier */
function sweepSamples(tier: QualityTier | undefined): number {
  return tier === 'thumb' ? 24 : tier === 'medium' ? 56 : 96;
}

/** The half-width of the cavern at a point along the passage — it breathes. */
export function cavernWidthAt(x: number, seed: number): number {
  const n = hash2f(Math.floor(x * 0.5) + 97, 3, seed) - 0.5;
  const m = Math.sin(x * 0.31 + seed * 0.0007) * 0.5;
  return CAVERN_HALF_W * (1 + 0.09 * m + 0.06 * n);
}

/** The crown height of the vault at a point along the passage. */
export function cavernHeightAt(x: number, seed: number): number {
  const n = hash2f(Math.floor(x * 0.4) + 13, 7, seed) - 0.5;
  const m = Math.sin(x * 0.22 + 1.7) * 0.5 + Math.sin(x * 0.51 - 0.4) * 0.22;
  return CAVERN_HEIGHT * (1 + 0.11 * m + 0.08 * n);
}

/**
 * The point on the cavern surface at sweep position `x` and arch parameter `t ∈ [0, 1]`
 * (0 = the far bank at z = −W, 0.5 = the crown, 1 = the near bank at z = +W), and the
 * INWARD unit normal there. Exported because the stalactites hang off exactly this
 * surface — a spike whose root is computed from the same function can never float.
 */
export function cavernSurface(
  x: number,
  t: number,
  seed: number,
  out: { z: number; y: number; nz: number; ny: number },
): void {
  const w = cavernWidthAt(x, seed);
  const h = cavernHeightAt(x, seed);
  const phi = Math.PI * t;
  const c = Math.cos(phi);
  const s = Math.sin(phi);
  out.z = -w * c;
  out.y = h * s;
  // inward normal of the ellipse (z = −w cos φ, y = h sin φ): (h cos φ, −w sin φ)
  const nz = h * c;
  const ny = -w * s;
  const len = Math.hypot(nz, ny) || 1;
  out.nz = nz / len;
  out.ny = ny / len;
}

/**
 * The cavern: one swept arch, one draw call, normals INWARD.
 *
 * The winding below is the load-bearing detail. Each quad is emitted as
 * `(i, j) → (i+1, j) → (i+1, j+1)` and `(i, j) → (i+1, j+1) → (i, j+1)`, which makes every
 * face's normal point into the cavity. Three culls back faces on `FrontSide`, so the half
 * of the shell nearer the camera than the passage simply is not drawn: an enclosed cave
 * from inside, an open cutaway from outside, from the same 2000 triangles.
 */
export function makeCavernGeometry(seed: number, tier: QualityTier, skin: CaveSkin): THREE.BufferGeometry {
  const rings = archSamples(tier);
  const sweeps = sweepSamples(tier);
  // two extra ring rows: a vertical rock skirt dropping below the waterline at each bank,
  // so no camera angle can ever see under the edge of the water
  const rowCount = rings + 2;
  const vertexCount = (sweeps + 1) * rowCount;
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const indices = new Uint32Array(sweeps * (rowCount - 1) * 6);

  const low = skin.rockLow;
  const high = skin.rockHigh;
  const c = new THREE.Color();
  const surf = { z: 0, y: 0, nz: 0, ny: 0 };

  let v = 0;
  for (let i = 0; i <= sweeps; i++) {
    const x = CAVERN_X0 + ((CAVERN_X1 - CAVERN_X0) * i) / sweeps;
    for (let j = 0; j < rowCount; j++) {
      let z: number;
      let y: number;
      if (j === 0 || j === rowCount - 1) {
        // the skirt rows: straight down from the bank into the water
        const t = j === 0 ? 0 : 1;
        cavernSurface(x, t, seed, surf);
        z = surf.z;
        y = -CAVERN_SKIRT;
      } else {
        const t = (j - 1) / (rings - 1);
        cavernSurface(x, t, seed, surf);
        z = surf.z;
        y = surf.y;
        // rock is not smooth: push each ring point along its own inward normal by a
        // deterministic amount, so the sweep reads as bedrock rather than a pipe
        const n = hash2f(i, j * 31 + 5, seed) - 0.5;
        const n2 = hash2f(i * 7 + 3, j, seed ^ 0x51ed270b) - 0.5;
        const push = (n * 0.62 + n2 * 0.3) * (0.24 + 0.5 * Math.sin(Math.PI * t));
        z += surf.nz * push;
        y += surf.ny * push;
      }
      positions[v * 3] = x;
      positions[v * 3 + 1] = y;
      positions[v * 3 + 2] = z;
      // vertical grade: pale wet limestone at the waterline, black in the vault
      const k = clamp01(y / CAVERN_HEIGHT);
      c.copy(low).lerp(high, smoothstep01(k * 1.15));
      const grain = 1 + (hash2f(i * 13 + 1, j * 5 + 2, seed) - 0.5) * 0.16;
      colors[v * 3] = c.r * grain;
      colors[v * 3 + 1] = c.g * grain;
      colors[v * 3 + 2] = c.b * grain;
      v++;
    }
  }

  let o = 0;
  for (let i = 0; i < sweeps; i++) {
    for (let j = 0; j < rowCount - 1; j++) {
      const a = i * rowCount + j;
      const b = (i + 1) * rowCount + j;
      indices[o++] = a;
      indices[o++] = b;
      indices[o++] = b + 1;
      indices[o++] = a;
      indices[o++] = b + 1;
      indices[o++] = a + 1;
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'cavernShell';
  return g;
}

// ---------------------------------------------------------------------------
// stalactites — instanced spikes rooted on the shell surface
// ---------------------------------------------------------------------------

/**
 * One spike: base at the local origin, tip at y = −1. Graded pale toward the tip.
 *
 * The base radius is the number that decides whether this is dripstone or grass. The first
 * pass used 0.12 against lengths up to 2.4 — a 20:1 needle — and the vault came out looking
 * like a hanging lawn. Real dripstone is stubby: 0.26 against a 0.7–2.2 length is 3–4:1,
 * which is what reads as rock.
 */
function spikeGeometry(tier: QualityTier): THREE.BufferGeometry {
  const radial = tier === 'thumb' ? 4 : tier === 'medium' ? 6 : 7;
  const g = new THREE.CylinderGeometry(0.02, 0.5, 1, radial, 2);
  g.rotateX(Math.PI); // apex down
  g.translate(0, -0.5, 0);
  const pos = g.getAttribute('position');
  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    // a shallow waist: enough to taper like dripstone, not enough to make a needle
    const y = pos.getY(i);
    const t = clamp01(-y);
    const k = 1 - 0.1 * Math.sin(Math.PI * t);
    pos.setX(i, pos.getX(i) * k);
    pos.setZ(i, pos.getZ(i) * k);
    const grade = 0.72 + 0.5 * t * t;
    colors[i * 3] = grade;
    colors[i * 3 + 1] = grade;
    colors[i * 3 + 2] = grade;
  }
  pos.needsUpdate = true;
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.name = 'stalactiteSpike';
  return g;
}

export interface StalactiteField {
  mesh: THREE.InstancedMesh;
  count: number;
  dispose(): void;
}

/**
 * The dripstone. Each spike's root is `cavernSurface(x, t)` — the same function the shell
 * is swept from — so a stalactite is attached to the rock, not floating near it, at every
 * tier and for every seed. Its axis leans halfway from straight down toward the local
 * inward normal, which is what makes a vault of them look grown rather than hung.
 */
export function makeStalactites(
  seed: number,
  tier: QualityTier,
  skin: CaveSkin,
  palette: Palette,
): StalactiteField {
  const wanted = tier === 'thumb' ? 20 : tier === 'medium' ? 34 : 54;
  const geometry = spikeGeometry(tier);
  const material = oakMaterial({
    // dripstone is WALL rock with a wet tip, not the pale mineral of a stalagmite lit
    // from below: at full `mineral` the vault came out brighter than the walkway
    color: blendTokens(skin.rockLow, skin.mineral, 0.22),
    vertexColors: true,
    flatShading: true,
    roughness: 0.86,
    rimColor: skin.rim,
    rimStrength: 0.42,
    emissive: blendTokens(skin.rockLow, skin.mineral, 0.22),
    emissiveIntensity: tierBounce(tier),
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: tierToonFloor(tier, 0.24),
    tier,
    name: 'stalactiteMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, wanted);
  mesh.name = 'stalactites';
  mesh.castShadow = tier !== 'thumb';
  mesh.receiveShadow = false;
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(wanted * 3), 3);

  const rng = streamFor(seed | 0, 'stalactites');
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const axis = new THREE.Vector3();
  const down = new THREE.Vector3(0, -1, 0);
  const color = new THREE.Color();
  const surf = { z: 0, y: 0, nz: 0, ny: 0 };

  for (let i = 0; i < wanted; i++) {
    const x = CAVERN_X0 + 1.5 + rng() * (CAVERN_X1 - CAVERN_X0 - 3);
    // Most of the dripstone lives on the half of the vault that back-face culling leaves
    // standing (t ≲ 0.55). The last sixth hangs from the culled near quarter: those are
    // the big foreground spikes that frame the top of the shot.
    const foreground = rng() < 0.06;
    // Dripstone grows on a CEILING. The first pass seeded from t = 0.05 — the vertical far
    // wall — and produced blades sticking sideways out of the rock face, which reads as
    // thorns, not as a cave. From t = 0.28 up, the surface is already leaning over the
    // passage and a spike hanging off it is unmistakably a stalactite.
    const t = foreground ? 0.6 + rng() * 0.16 : 0.28 + rng() * 0.3;
    cavernSurface(x, t, seed, surf);
    // SQUARED length: most dripstone is a stubby nub and a few are long, which is what
    // stops a ceiling of them reading as a lawn. A linear draw gives every spike roughly
    // the same length and the vault comes out looking combed.
    const shortness = rng();
    const len = foreground ? 1.2 + rng() * 1.1 : 0.42 + shortness * shortness * 1.5;
    const girth = (foreground ? 1.15 : 0.85 + rng() * 0.55) * (0.85 + 0.3 * rng());
    axis.set(0, -1, 0).lerp(new THREE.Vector3(0, surf.ny, surf.nz), 0.3).normalize();
    quaternion.setFromUnitVectors(down, axis);
    position.set(x, surf.y + 0.06, surf.z);
    scale.set(girth, len, girth);
    matrix.compose(position, quaternion, scale);
    mesh.setMatrixAt(i, matrix);
    const k = 0.82 + rng() * 0.34;
    color.setRGB(k, k, k);
    mesh.setColorAt(i, color);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();

  return {
    mesh,
    count: wanted,
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}

// ---------------------------------------------------------------------------
// world handle
// ---------------------------------------------------------------------------

export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  water: WaterHandle;
  skin: CaveSkin;
  /** the passage's own fireflies — the accent, and never tinted by the cue */
  fireflies: AmbientParticlesHandle;
  /**
   * The celebration swarm: a second firefly field that lives at density 0 until a correct
   * choice fires it, then sweeps the whole passage. It is a kit particle field whose GROUP
   * is moved, so the swarm costs one draw call and no bespoke shader.
   */
  flood: AmbientParticlesHandle;
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
  const palette: Palette = ctx.tokens.palettes.cave;
  const skin = caveSkin(palette, ctx.tokens, tier);
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'fireflyCave';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const sized: { setPixelRatio(dpr: number): void }[] = [];

  // ---- sky: the cave parameterization, a pure vertical depth column (no sun disc) ----
  const sky = skyFromPalette(palette, 'cave', tier, {
    topSpread: 0.5,
    bottomSpread: 0.45,
    dither: 0.01,
    radius: 320,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the one light budget: sun + hemi + 4 points ----
  // points[0] wisp · points[1] cue crystal · points[2] the choice · points[3] the flood
  const rig = makeLightRig(palette, tier, {
    sunDir: SUN_DIR,
    pointCount: 4,
    // a cave light has to REACH: the passage is 32 units long and the crystal is meant to
    // wash the near end of it, so the falloff is set to the width of the cavern, not to
    // the 12-unit default that was tuned for an open courtyard
    pointDistance: 15,
    pointDecay: 1.45,
    sunDistanceScale: 2.4,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: SHADOW_CENTER[0], y: SHADOW_CENTER[1], z: SHADOW_CENTER[2] }, SHADOW_RADIUS);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the cavern ----
  const cavernGeometry = makeCavernGeometry(ctx.seed, tier, skin);
  const cavernMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.97,
    rimColor: skin.rim,
    rimStrength: 0.4,
    emissive: blendTokens(skin.rockLow, skin.rockHigh, 0.72),
    emissiveIntensity: tierBounce(tier),
    toonSteps: 3,
    toonSoftness: 0.32,
    // a high toon floor is how a cave keeps its shape: the ramp still bands the lit
    // faces, but no face is ever allowed to go to black, because a black face in a cave
    // is not "shadow", it is "missing geometry"
    toonFloor: tierToonFloor(tier, 0.36),
    // FrontSide + inward normals = the free cutaway; see the file header
    side: THREE.FrontSide,
    tier,
    name: 'cavernMaterial',
  });
  const cavern = new THREE.Mesh(cavernGeometry, cavernMaterial);
  cavern.name = 'cavernShell';
  cavern.receiveShadow = shadows;
  cavern.castShadow = false;
  group.add(cavern);
  owned.push({
    dispose() {
      cavernGeometry.dispose();
      cavernMaterial.dispose();
      cavern.removeFromParent();
    },
  });

  // ---- the still water the walkway crosses ----
  const water = makeWater({
    // Longer than the cave in BOTH directions: the cavern is open at each end, so a water
    // plane that stopped at the rock would draw a hard bright seam across the far third of
    // the frame. Past the ends there is nothing but fog, which is the point.
    size: [(CAVERN_X1 - CAVERN_X0) + 34, WATER_HALF_Z * 2],
    palette,
    tier,
    segments: 96,
    shallow: skin.waterShallow,
    deep: skin.waterDeep,
    sky: blendTokens(palette.hemiSky, skin.rim, 0.35),
    caustic: blendTokens(palette.hemiSky, skin.firefly, 0.25),
    // STILL water: a cave pool has no wind on it. The waves are a tenth of the kit default
    // and slowed to a crawl, so what moves is the reflection, not the surface.
    amplitude: 0.0125,
    waveScale: 0.65,
    speed: 0.22,
    // The camera sees this plane at a grazing angle down its whole length, so fresnel is
    // the term that decides whether the pools are black water or a sheet of pale glare.
    // At the kit default of 0.55 the water was the BRIGHTEST thing in the cave and the
    // walkway read as dark holes punched in it — exactly backwards.
    fresnel: 0.24,
    caustics: 0.14,
    depthFalloff: 10,
    opacity: 0.97,
    sunDir: rig.sunDirection,
    level: WATER_LEVEL,
    name: 'cavePools',
  });
  water.mesh.position.x = (CAVERN_X0 + CAVERN_X1) * 0.5;
  // Biased toward the near bank, and far enough past it that the plane's own edge is
  // BEHIND the camera. The far overshoot only has to clear the widest section of the arch
  // (6.55) to leave no gap, and everything past that is hidden behind rock; the near
  // overshoot is what stops a hard bright seam cutting the bottom corner of every frame.
  water.mesh.position.z = 4;
  group.add(water.mesh);
  owned.push(water);
  timed.push(water);

  // ---- the walkway: ONE draw, straight off `grid.walls` (which is all-free here) ----
  // `floorInset` is doing real work: at 0.09 there is a 0.18-unit slot of black water
  // between every pair of stones, so the passage reads as 32 STEPPING STONES and the
  // agent's position along it can be counted off by eye.
  const field = buildCellField(
    grid,
    {
      floorColor: skin.stone,
      wallColor: skin.stone,
      floorTint: (x) => {
        // a slow wet-to-dry gradient down the passage — decoration only, and identical
        // for every seed and every cue, so it can never encode the hidden variable
        const k = x / Math.max(1, MEMORYCHAIN_L - 1);
        return blendTokens(0xffffff, 0x9aa6bb, 0.28 * k);
      },
      floorInset: 0.09,
      floorThickness: 0.34,
      jitter: 0.14,
      rimColor: skin.rim,
      rimStrength: 0.6,
      seed: ctx.seed,
      name: 'walkwayStones',
    },
    tier,
  );
  // `buildCellField` has no `toonFloor` option and its default is 0, which puts every
  // unlit face of every stone at pure black — fine under a noon key, fatal under 0.2. The
  // uniform handle is part of the material's public surface, so lift it here.
  {
    const floor = oakUniforms(field.floorMaterial);
    if (floor) floor.uniforms.uToonFloor.value = tierToonFloor(tier, 0.34);
    field.floorMaterial.emissive.copy(skin.stone);
    field.floorMaterial.emissiveIntensity = tierBounce(tier);
  }
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  // ---- stalagmites standing in the pools (never on the walkway) ----
  const extent = gridExtent(grid);
  const banks = scatterRocks({
    // Fewer and taller. At 168 the pools read as a gravel bed; a cave floor wants a
    // handful of real spires and a lot of empty black between them.
    count: tier === 'thumb' ? 26 : 82,
    area: { x: 0, z: 0.3, halfX: extent.width * 0.5 + 4.5, halfZ: POOL_HALF_Z },
    seed: ctx.seed ^ 0x2f19,
    tier,
    exclude: excludeBand(0, CELL * 0.5, 0.55),
    heightAt: () => WATER_LEVEL - 0.24,
    color: skin.mineral,
    // tall and narrow: a rock at stretch 3.4 is a stalagmite, at 0.4 it is a pebble
    scale: [0.42, 1.55],
    stretch: [1.3, 3.6],
    hueJitter: 0.22,
    rimColor: skin.rim,
    rimStrength: 0.8,
    castShadow: false,
    receiveShadow: false,
    name: 'stalagmites',
  });
  group.add(banks.group);
  owned.push(banks);
  timed.push(banks);

  // ---- dripstone ----
  const stalactites = makeStalactites(ctx.seed, tier, skin, palette);
  group.add(stalactites.mesh);
  owned.push(stalactites);

  // ---- the fireflies that line the passage (the accent) ----
  // Canon lesson 5: the kit presets size their motes for a ~20-unit field. This passage is
  // 40 units long and the camera stands 16–40 units off, so `gl_PointSize`'s 60/distance
  // term wants a size near 2, not the preset's 9.
  const fireflies = makeAmbientParticles({
    preset: 'fireflies',
    count: 640,
    bounds: { x: 0, y: 1.4, z: -0.1, halfX: PASSAGE_HALF + 3, halfY: 1.7, halfZ: 3.5 },
    seed: ctx.seed ^ 0x11e5,
    tier,
    color: skin.firefly,
    size: 2.35,
    intensity: 2.9,
    opacity: 0.95,
    rise: 0.05,
    swirl: 0.5,
    name: 'passageFireflies',
  });
  group.add(fireflies.points);
  owned.push(fireflies);
  timed.push(fireflies);
  sized.push(fireflies);

  // ---- a thin haze of cave dust high in the vault, for depth ----
  const vaultDust = makeAmbientParticles({
    preset: 'dust',
    count: 420,
    bounds: { x: 0, y: 4.6, z: -0.6, halfX: PASSAGE_HALF + 4, halfY: 3.1, halfZ: 3.4 },
    seed: ctx.seed ^ 0x7c03,
    tier,
    color: blendTokens(palette.hemiSky, skin.mineral, 0.4),
    size: 1.5,
    intensity: 0.85,
    opacity: 0.4,
    rise: 0.015,
    swirl: 0.24,
    name: 'vaultDust',
  });
  group.add(vaultDust.points);
  owned.push(vaultDust);
  timed.push(vaultDust);
  sized.push(vaultDust);

  // ---- the celebration swarm, parked at density 0 ----
  const flood = makeAmbientParticles({
    preset: 'fireflies',
    count: 520,
    bounds: { x: 0, y: 1.15, z: 0, halfX: 2.4, halfY: 1.3, halfZ: 1.5 },
    seed: ctx.seed ^ 0x63b1,
    tier,
    // Born NEUTRAL. The swarm is invisible until a correct choice fires it (density 0), and
    // the scene retints it to the winning cue on that same frame — but a field that sat
    // there holding a cue colour in its uniform, even unrendered, is exactly the kind of
    // thing this scene should not have anywhere in it.
    color: skin.firefly,
    size: 3.4,
    intensity: 3.4,
    opacity: 1,
    rise: 0.16,
    swirl: 0.62,
    name: 'choiceFlood',
  });
  flood.setDensity(0);
  group.add(flood.points);
  owned.push(flood);
  timed.push(flood);
  sized.push(flood);

  return {
    group,
    sky,
    rig,
    field,
    water,
    skin,
    fireflies,
    flood,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
      oakUniforms(cavernMaterial)?.setTime(t);
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

// ---------------------------------------------------------------------------
// state reading — the only two numbers this scene is made of
// ---------------------------------------------------------------------------

/** The passage cell the agent stands on, clamped into the corridor. */
export function readPos(state: { pose: Float32Array; derived: unknown }): number {
  const d = state.derived as { pos?: unknown } | null | undefined;
  const raw = d && typeof d === 'object' && typeof d.pos === 'number' ? d.pos : (state.pose[0] ?? 0);
  const v = Number.isFinite(raw) ? Math.round(raw) : 0;
  return v < 0 ? 0 : v > MEMORYCHAIN_END ? MEMORYCHAIN_END : v;
}

/**
 * The cue in force, 0 or 1. Reading it every frame is fine — SHOWING it is what the
 * honesty rule governs, and the only consumer is the crystal, which is dark unless the
 * agent is standing on cell 0 looking straight at it.
 */
export function readCue(state: { pose: Float32Array; derived: unknown }): 0 | 1 {
  const d = state.derived as { cue?: unknown } | null | undefined;
  const raw = d && typeof d === 'object' && typeof d.cue === 'number' ? d.cue : (state.pose[2] ?? 0);
  return Math.round(Number.isFinite(raw) ? raw : 0) === 1 ? 1 : 0;
}

/**
 * The action bit of the step that produced this frame. On the frame that carries a `goal`
 * event this IS the winning cue (the decoder fires `goal` exactly when
 * `action & 1 === prev.cue`), so the celebration colour comes from the current frame alone
 * — no history, no prev-frame lookup, scrub-safe by construction.
 */
export function readChoice(state: { ctrl: Float32Array }): 0 | 1 {
  const raw = state.ctrl[0] ?? 0;
  return (Math.trunc(Number.isFinite(raw) ? raw : 0) & 1) === 1 ? 1 : 0;
}

/** World x of a passage cell (the scene never converts cells any other way). */
export function cellX(cell: number, grid: GridSize, out: Vec3Tuple): number {
  cellToWorld(cell, 0, grid, out);
  return out[0];
}
