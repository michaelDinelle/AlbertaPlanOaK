// The Willow Swing — the static world.
//
// THE ACCURACY CLAIM OF THIS FILE: every world position that means something to the env
// is read from `geometry.classic.anchors` and nowhere else.
//   * `pivot`       -> the bough anchor the trapeze hangs from (the scene-graph root of
//                      the whole arm; see actors.ts)
//   * `successLine` -> the y the string of lights is strung at. The bulbs' centres ARE
//                      that surface: `tipHeight > 1` is true exactly when the lantern is
//                      above the bulbs, so the honesty rule ("never hide what the env
//                      defines — Acrobot success line") is satisfied by construction and
//                      not by a tuned decoration.
//   * `restTip`     -> where the lantern hangs at the reset, i.e. how far the pond has to
//                      sit below the pivot for the swing to skim it.
// Nothing in this file re-derives link lengths, the success height, or the reach; the
// decoder's `AB_L1`, `AB_L2` and the anchors are the only sources.
//
// LIGHT: the sun is low, left and BEHIND, so the willow is a backlit curtain, the pond
// carries a long specular path toward the camera, and the lantern — the one warm HDR
// thing on the near side — owns the frame. `toonFloor` is generous throughout: a dawn
// shadow side is blue, never black.

import * as THREE from 'three';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import {
  emissiveMaterial,
  oakMaterial,
  setMaterialsTime,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import {
  makeAmbientParticles,
  makeMistPlanes,
  type AmbientParticlesHandle,
  type MistPlanesHandle,
} from '@kit/particles';
import { streamFor } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { radialSprite, valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import { excludeAny, scatterGrass, scatterTrees, type ExcludeFn } from '@kit/vegetation';
import { makeWater, type WaterHandle } from '@kit/water';
import { AB_L1, AB_L2, AB_SUCCESS_HEIGHT } from '@sim/envs/acrobot';
import type { EnvGeometry } from '@sim/types';

// ---------------------------------------------------------------------------
// world constants (1 link === 1 world unit — the reach really is 2 units)
// ---------------------------------------------------------------------------

/** the arm's full reach from the pivot; nothing solid may enter this disc at z ≈ 0 */
export const REACH = AB_L1 + AB_L2;
/** a little clearance past the reach for the lantern's own body */
export const SWING_CLEARANCE = 0.3;

/**
 * The pond surface. The lantern hangs at `restTip` (y = −2) and is ~0.2 deep, so a
 * surface 0.42 under the rest tip lets the swing SKIM the water at the bottom of every
 * arc without ever touching it — the reflection meets the lantern and parts again.
 */
export const WATER_Y = -(AB_L1 + AB_L2) - 0.42;
export const POND_SIZE: readonly [number, number] = [30, 30];

export const TERRAIN_SIZE: readonly [number, number] = [46, 34];
export const TERRAIN_SEGMENTS: readonly [number, number] = [152, 112];

/** the string of lights spans the frame; the cords sit either side of the swing plane */
export const GARLAND_HALF_X = 3.42;
export const GARLAND_Z = 0.34;
export const GARLAND_BULBS = 13;

/** the willow stands off the left edge, its great bough arching in over the pond */
export const WILLOW_X = -4.95;
export const WILLOW_Z = -1.25;
/** the bough passes over the pivot at this height; a short hanger drops to the anchor */
export const BOUGH_Y = 0.34;
/**
 * ...and it passes over it BEHIND THE SWING PLANE. The load-bearing bough is the one
 * solid the arm could ever hit: the pivot hangs only 0.34 under it, and the env's whole
 * objective is to raise the free end a full link ABOVE the pivot, so a bough drawn at
 * z = 0 is crossed by the arm on every scoring swing (measured on the recorded stream:
 * link 1 or link 2 came within a mesh radius of the limb on 2.25% of frames, and passed
 * clean through its axis at the near-success poses). Everything inside the reach is
 * therefore offset by more than (bough radius 0.076 + lantern radius 0.127), and a
 * near-profile camera foreshortens the offset away.
 */
export const BOUGH_Z = -0.36;
/**
 * The minimum |z − pivot.z| any solid may have inside the swing disc. It is the clearance
 * the arm's own body needs, not a style choice: the lantern is 0.127 wide in z and the
 * bough 0.076, so 0.25 is the first round number that guarantees a gap.
 */
export const SWING_PLANE_CLEARANCE = 0.25;
/**
 * The hinge sleeve. The brass collar the arm turns in and the last stretch of the hanger
 * cord are the ONLY solids allowed to touch the swing plane, because they ARE the
 * attachment the arm rotates about — a hinge that cleared its own axle would be a lie.
 * Measured (locked by the test): the furthest vertex inside the plane sits at an in-plane
 * radius of 0.358 — a 0.02-radius cord and a 0.075 collar, 18% of the arm's reach.
 */
export const HINGE_R = 0.45;

/** fog reach is derived from the world footprint, never hand-tuned per shot */
export const FOG_EXTENT = 210;

/**
 * Low, left and behind: elevation 27° (canon lesson 8 — a lower sun smears the willow's
 * shadow across the whole pond), azimuth well behind the subject so the fronds are a
 * backlit curtain and the water carries a specular path back toward the camera.
 */
export const SUN_DIR: readonly [number, number, number] = [-0.578, 0.458, -0.677];

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

/** `noon.sun` is the token white and `night.skyTop` the token near-black (canon 9). */
export const TOKEN_WHITE = TOKENS.palettes.noon.sun;
export const TOKEN_BLACK = TOKENS.palettes.night.skyTop;

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

/**
 * Merge pre-transformed parts into ONE geometry carrying a flat per-part vertex color,
 * disposing the sources. Hand-built props (a willow, a trapeze, a garland) want their
 * SOURCE normals plus a bark/vine/brass palette baked per part, so one mesh + one
 * material draws the whole prop.
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

/** A capsule-free limb: one tapered cylinder laid between two points in the x-y plane. */
export function boneGeometry(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  r0: number,
  r1: number,
  radial = 6,
): THREE.BufferGeometry {
  const dx = bx - ax;
  const dy = by - ay;
  const len = Math.hypot(dx, dy);
  const g = new THREE.CylinderGeometry(r1, r0, len, radial, 1);
  // local +y runs a→b; rotate it onto the segment and drop it at the midpoint
  g.rotateZ(Math.atan2(dy, dx) - Math.PI / 2);
  g.translate((ax + bx) * 0.5, (ay + by) * 0.5, 0);
  return g;
}

/** floats written by `acrobotPointsExact`: [elbowX, elbowY, tipX, tipY] */
export const EXACT_POINT_FLOATS = 4;

/**
 * The f64 twin of the decoder's `acrobotPoints`. IDENTICAL formula — the decoder writes
 * into a `Float32Array`, which rounds every coordinate to ~2e-8, and a probe measured
 * against that would be grading the scene against a rounded truth. `out[3]` is bitwise
 * `acrobotTipHeight(th1, th2)`; the test pins both claims against the decoder itself.
 */
export function acrobotPointsExact(th1: number, th2: number, out: Float64Array): Float64Array {
  const ex = AB_L1 * Math.sin(th1);
  const ey = -AB_L1 * Math.cos(th1);
  out[0] = ex;
  out[1] = ey;
  out[2] = ex + AB_L2 * Math.sin(th1 + th2);
  out[3] = ey - AB_L2 * Math.cos(th1 + th2);
  return out;
}

/**
 * The env's own anchors, never a scene re-derivation (STYLE.md accuracy law).
 * A missing anchor falls back to the decoder's own literal, so a placeholder geometry
 * (the harness fixture loop) still builds a correct world.
 */
export function anchorOf(
  geometry: EnvGeometry,
  key: string,
  fallback: readonly [number, number, number],
): readonly [number, number, number] {
  return geometry.classic?.anchors?.[key] ?? fallback;
}

/** The bough anchor: the hinge the whole arm hangs from. */
export function pivotAnchor(geometry: EnvGeometry): readonly [number, number, number] {
  return anchorOf(geometry, 'pivot', [0, 0, 0]);
}

/** The success surface: `tipHeight > 1`, drawn as the string of lights. */
export function successAnchor(geometry: EnvGeometry): readonly [number, number, number] {
  return anchorOf(geometry, 'successLine', [0, AB_SUCCESS_HEIGHT, 0]);
}

// ---------------------------------------------------------------------------
// the bank height field
// ---------------------------------------------------------------------------

/**
 * A pond floor a metre under the surface everywhere the camera can see water, climbing
 * into a low far shore beyond z = −7.5 and into side banks past |x| = 9.5. The whole
 * near field (the swing's own water) is identically flat, so nothing the art invents can
 * ever poke through the mirror the lantern is reflected in.
 */
export function makeBankHeight(seed: number): (x: number, z: number) => number {
  const floor = WATER_Y - 1.05;
  return (x, z) => {
    const away = smoothstep01((-z - 7.5) / 9);
    const side = smoothstep01((Math.abs(x) - 9.5) / 6);
    const w = away > side ? away : side;
    if (w <= 0) return floor;
    const n1 = valueNoise2(x * 0.22 + 4.1, z * 0.22 + 9.3, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.63 + 1.7, z * 0.63 + 3.9, 64, seed ^ 0x2f19) - 0.5;
    return floor + w * (2.35 + n1 * 1.9 + n2 * 0.7);
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
  /** the string of lights — a probe object, world position === anchor `successLine` */
  garlandRoot: THREE.Object3D;
  /** pond surface height (the ripple plane and the lantern's glint pool) */
  waterY: number;
  /** ctx.time() — pause freezes wind, mist, fireflies and the dawn together */
  setTime(t: number): void;
  /** 0 = deep blue hour (fireflies out, mist thick), 1 = full dawn */
  setDawn(k01: number): void;
  /**
   * 0 → 1 celebration flare progress; bulbs light in sequence from the centre outward.
   * A NEGATIVE progress is "idle" (every bulb back to its resting gain).
   */
  setFlare(progress: number): void;
  /** 0..1 petal rain from the willow */
  setPetals(env01: number): void;
  /** the lantern's glint pool on the water: world x/z of the lantern + a 0..1 gain */
  setGlint(x: number, z: number, gain: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

interface Disposable {
  dispose(): void;
}

interface Timed {
  setTime(t: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const tier: QualityTier = ctx.tier;
  const palette: Palette = ctx.tokens.palettes.dawn;
  const accent = ctx.tokens.accents['acrobot'] ?? ctx.tokens.celebrationGold;
  const gold = ctx.tokens.celebrationGold;
  const group = new THREE.Group();
  group.name = 'willowSwing';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const animated: OakMaterial[] = [];
  const shadows = tier !== 'thumb';

  const pivot = pivotAnchor(ctx.geometry);
  const success = successAnchor(ctx.geometry);

  // ---- palette-derived surface colors (blends of tokens only) ----
  const bark = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.42);
  const barkLit = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.28);
  const frondA = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.3);
  const frondB = blendTokens(palette.groundA, palette.hemiGround, 0.55);
  const shore = blendTokens(palette.groundA, palette.hemiGround, 0.45);
  const shoreLit = blendTokens(palette.groundB, palette.skyHorizon, 0.28);
  const brass = blendTokens(palette.sun, palette.hemiGround, 0.42);

  // ---- sky (exactly one) ----
  const sky = skyFromPalette(palette, 'dawn', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.034,
    sunHdr: 3.6,
    sunGlow: 0.34,
    sunGlowPower: 20,
    // a small topSpread lets the dawn BLUE come all the way down to the treeline; the
    // kit default keeps the peach horizon so high that the whole sky reads as one cream
    topSpread: 0.28,
    bottomSpread: 0.42,
    radius: 320,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the one light budget ----
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 3,
    pointDistance: 7,
    sunDistanceScale: 3,
  });
  group.add(rig.group);
  owned.push(rig);
  // the focus must COVER what is on screen (canon lesson 8): the whole swing disc plus
  // the near bank, or the border texel paints amoebas across the far shore.
  rig.setShadowFocus({ x: 0, y: -0.45, z: -0.6 }, 4.2);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the pond floor and the banks ----
  const bankHeight = makeBankHeight(ctx.seed);
  const rampColor = new THREE.Color();
  const bedColor = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.78);
  const terrain = makeTerrain({
    size: TERRAIN_SIZE,
    segments: TERRAIN_SEGMENTS,
    tier,
    seed: ctx.seed,
    heightFn: bankHeight,
    jitter: 0,
    colorNoise: 0.05,
    slopeColor: blendTokens(shore, TOKEN_BLACK, 0.3),
    slopeStrength: 0.6,
    slopeSharpness: 2.1,
    colorRamp: (h) => {
      // under the water it is a dark bed; above it, damp shore into dawn-lit meadow
      const wet = smoothstep01((h - (WATER_Y - 0.55)) / 0.75);
      const dry = smoothstep01((h - (WATER_Y + 0.55)) / 1.5);
      rampColor.copy(bedColor).lerp(shore, wet);
      rampColor.lerp(shoreLit, dry);
      return rampColor;
    },
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.3,
      toonSteps: 3,
      toonSoftness: 0.34,
      toonFloor: 0.26,
    },
    receiveShadow: shadows,
    castShadow: false,
    name: 'pondBank',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  // ---- the mirror pond ----
  // MEASURED, not guessed: at this grazing angle the kit's palette-derived default water
  // renders around 80% white — it is tuned for a pond seen from ABOVE, where most of what
  // you see is the bed. Here the pond is a MIRROR at dawn, and a mirror is dark: the sky
  // glow arrives through the fresnel rim and the sun path, never through the body colour.
  const pondShallow = blendTokens(palette.skyTop, TOKEN_BLACK, 0.86);
  const pondDeep = blendTokens(TOKEN_BLACK, palette.skyTop, 0.12);
  const water = makeWater({
    size: POND_SIZE,
    palette,
    tier,
    segments: 96,
    level: WATER_Y,
    shallow: pondShallow,
    deep: pondDeep,
    sky: blendTokens(palette.skyHorizon, TOKEN_BLACK, 0.72),
    caustic: blendTokens(palette.sun, TOKEN_BLACK, 0.55),
    amplitude: 0.012,
    waveScale: 0.7,
    speed: 0.4,
    // a near-profile camera sees the fresnel term over the WHOLE surface (dot(n, view)
    // never gets far from grazing), so the kit default rim turns the mirror into a sheet
    // of horizon. The rim here is a hint at the far edge, nothing more.
    fresnel: 0.22,
    caustics: 0.06,
    depthFalloff: 11,
    opacity: 0.985,
    sunDir: sky.sunDirection,
    name: 'mirrorPond',
  });
  // KIT GAP (worked around, filed): `WaterOptions.shallow`/`deep` are not used as given —
  // makeWater lerps them 35% toward `palette.groundB` and 55% toward `palette.skyBottom`,
  // so on a warm palette NO override can reach a dark water. `setColors` writes the live
  // uniforms directly and is the only way to get a dawn mirror instead of a tan sheet.
  water.setColors(pondShallow, pondDeep);
  // the specular path is the ONE bloom feeder the water is allowed
  water.uniforms.uSpecular.value = tier === 'thumb' ? 0 : 1.15;
  group.add(water.mesh);
  owned.push(water);
  timed.push(water);

  // ---- the willow: trunk, boughs, and the hanger the trapeze swings from ----
  const boughPoints = willowBoughs();
  const willowParts: ColoredPart[] = [];
  {
    const rng = streamFor(ctx.seed, 'willow');
    // trunk: a stack of leaning segments so the silhouette is never a pipe
    const SEGS = 8;
    let x = WILLOW_X;
    let y = WATER_Y + 0.05;
    for (let i = 0; i < SEGS; i++) {
      const k = i / SEGS;
      const h = 0.31 + 0.06 * (1 - k);
      const lean = 0.045 + 0.13 * k * k + (rng() - 0.5) * 0.04;
      const nx = x + lean;
      const ny = y + h;
      willowParts.push({
        geometry: boneGeometry(x, y, nx, ny, 0.36 * (1 - k * 0.55), 0.36 * (1 - (k + 1 / SEGS) * 0.55), 7),
        color: i % 2 === 0 ? bark : barkLit,
      });
      x = nx;
      y = ny;
    }
    // root flare at the waterline
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2 + rng();
      const g = new THREE.IcosahedronGeometry(0.2 + rng() * 0.1, 0);
      g.scale(1.5, 0.55, 1.5);
      g.translate(WILLOW_X + Math.cos(a) * 0.26, WATER_Y + 0.1, WILLOW_Z + Math.sin(a) * 0.26);
      willowParts.push({ geometry: g, color: bark });
    }
    // the boughs
    for (const bough of boughPoints) {
      for (let i = 0; i < bough.length - 1; i++) {
        const a = bough[i] as readonly [number, number, number];
        const b = bough[i + 1] as readonly [number, number, number];
        const k0 = i / (bough.length - 1);
        const k1 = (i + 1) / (bough.length - 1);
        const g = new THREE.CylinderGeometry(
          0.115 * (1 - k1 * 0.8),
          0.115 * (1 - k0 * 0.8),
          Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]),
          6,
          1,
        );
        alignSegment(g, a, b);
        willowParts.push({ geometry: g, color: i % 2 === 0 ? barkLit : bark });
      }
    }
    // The hanger: a braided drop from the bough down to the env's pivot anchor. It leans
    // FORWARD out of the bough's plane (BOUGH_Z → the pivot) instead of dropping straight,
    // so the only part of it standing in the arm's own column is the tie at the hinge —
    // where the collar is, and where the arm is attached anyway. A leaning cord reads
    // exactly like a cord; a vertical rod through the swing does not.
    const hangerTop: readonly [number, number, number] = [pivot[0], BOUGH_Y, pivot[2] + BOUGH_Z];
    const hangerFoot: readonly [number, number, number] = [pivot[0], pivot[1] + 0.03, pivot[2]];
    const hanger = new THREE.CylinderGeometry(
      0.02,
      0.028,
      Math.hypot(
        hangerFoot[0] - hangerTop[0],
        hangerFoot[1] - hangerTop[1],
        hangerFoot[2] - hangerTop[2],
      ),
      5,
    );
    // local +y runs top → foot, so the FIRST radius is the one at the hinge
    alignSegment(hanger, hangerTop, hangerFoot);
    willowParts.push({ geometry: hanger, color: bark });
    const collar = new THREE.CylinderGeometry(0.062, 0.075, 0.075, 8);
    collar.rotateX(Math.PI / 2);
    collar.translate(pivot[0], pivot[1], pivot[2]);
    willowParts.push({ geometry: collar, color: brass });
  }
  const willowGeometry = mergeColoredParts(willowParts, 'willow');
  const woodMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.93,
    rimColor: palette.sun,
    rimStrength: 0.72,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.24,
    tier,
    name: 'willowWood',
  });
  const willowMesh = new THREE.Mesh(willowGeometry, woodMaterial);
  willowMesh.name = 'willow';
  willowMesh.castShadow = shadows;
  willowMesh.receiveShadow = shadows;
  group.add(willowMesh);
  owned.push({
    dispose: () => {
      willowGeometry.dispose();
      woodMaterial.dispose();
    },
  });

  // ---- the frond curtain: ONE InstancedMesh, per-instance wind phase (canon 10) ----
  const FROND_LEN = 1.5;
  const frondGeometry = makeFrondGeometry(FROND_LEN);
  const FROND_WIND = 0.075;
  const frondMaterial = oakMaterial({
    color: TOKEN_WHITE,
    vertexColors: true,
    side: THREE.DoubleSide,
    roughness: 0.88,
    rimColor: palette.sun,
    // a thin strand is edge-on almost everywhere, so a strong rim turns the whole
    // curtain into lit straw; the willow wants a dark silhouette with a lit EDGE
    rimStrength: 0.3,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.22,
    wind: { amplitude: FROND_WIND, frequency: 0.85, height: FROND_LEN },
    tier,
    name: 'willowFrond',
  });
  animated.push(frondMaterial);
  const fronds = buildFronds({
    geometry: frondGeometry,
    material: frondMaterial,
    boughs: boughPoints,
    seed: ctx.seed,
    tier,
    length: FROND_LEN,
    colors: [frondA, frondB, blendTokens(frondA, palette.groundB, 0.35)],
  });
  group.add(fronds);
  owned.push({
    dispose: () => {
      fronds.dispose();
      frondGeometry.dispose();
      frondMaterial.dispose();
    },
  });

  // ---- the string of lights: the success surface, physically placed ----
  const garlandRoot = new THREE.Object3D();
  garlandRoot.name = 'garlandRoot';
  garlandRoot.position.set(success[0], success[1], success[2]);
  group.add(garlandRoot);

  const rigParts: ColoredPart[] = [];
  {
    // two slender stakes driven into the pond hold the far end; the near end is tied
    // into the willow's own bough, so the line is a real span, not a floating rule.
    for (const z of [-GARLAND_Z, GARLAND_Z]) {
      const stakeH = success[1] + 0.42 - WATER_Y;
      const stake = new THREE.CylinderGeometry(0.02, 0.034, stakeH, 6);
      stake.rotateZ(-0.045);
      stake.translate(GARLAND_HALF_X + 0.05, (success[1] + 0.42 + WATER_Y) * 0.5, z);
      rigParts.push({ geometry: stake, color: bark });
      const cap = new THREE.ConeGeometry(0.045, 0.1, 6);
      cap.translate(GARLAND_HALF_X - 0.03, success[1] + 0.48, z);
      rigParts.push({ geometry: cap, color: barkLit });
      const tie = new THREE.TorusGeometry(0.038, 0.011, 4, 8);
      tie.rotateY(Math.PI / 2);
      tie.translate(GARLAND_HALF_X, success[1] + 0.02, z);
      rigParts.push({ geometry: tie, color: barkLit });
      // the cord itself: dead straight, exactly on the success height
      const cord = new THREE.BoxGeometry(GARLAND_HALF_X * 2, 0.011, 0.011);
      cord.translate(0, success[1], z);
      rigParts.push({ geometry: cord, color: bark });
    }
  }
  const garlandGeometry = mergeColoredParts(rigParts, 'garlandRig');
  const garlandMesh = new THREE.Mesh(garlandGeometry, woodMaterial);
  garlandMesh.name = 'garlandRig';
  // sub-texel props must not cast (canon lesson 8): an 11 mm cord shadows as dashes
  garlandMesh.castShadow = false;
  group.add(garlandMesh);
  owned.push({ dispose: () => garlandGeometry.dispose() });

  const BULB_BASE_HDR = 1.25;
  const bulbGeometry = new THREE.IcosahedronGeometry(0.05, tier === 'thumb' ? 0 : 1);
  bulbGeometry.scale(1, 1.25, 1);
  const bulbMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(palette.sun, gold, 0.45),
    BULB_BASE_HDR,
    { name: 'garlandBulb' },
  );
  const bulbCount = GARLAND_BULBS * 2;
  const bulbs = new THREE.InstancedMesh(bulbGeometry, bulbMaterial, bulbCount);
  bulbs.name = 'garlandBulbs';
  const bulbGain = new Float32Array(bulbCount * 3).fill(1);
  bulbs.instanceColor = new THREE.InstancedBufferAttribute(bulbGain, 3);
  /** |x| of each bulb, for the centre-outward flare sequence */
  const bulbSpan = new Float32Array(bulbCount);
  {
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3(1, 1, 1);
    let n = 0;
    for (const z of [-GARLAND_Z, GARLAND_Z]) {
      for (let i = 0; i < GARLAND_BULBS; i++) {
        const t = GARLAND_BULBS > 1 ? i / (GARLAND_BULBS - 1) : 0.5;
        const x = -GARLAND_HALF_X + t * GARLAND_HALF_X * 2;
        // the bulbs' CENTRES are the success surface: y === anchor successLine exactly
        p.set(x, success[1], z);
        m.compose(p, q, s);
        bulbs.setMatrixAt(n, m);
        bulbSpan[n] = Math.abs(x) / GARLAND_HALF_X;
        n++;
      }
    }
    bulbs.instanceMatrix.needsUpdate = true;
    bulbs.computeBoundingSphere();
  }
  group.add(bulbs);
  owned.push({
    dispose: () => {
      bulbs.dispose();
      bulbGeometry.dispose();
      bulbMaterial.dispose();
    },
  });

  // ---- reeds: a fringe in the true shallows, never a marsh over the mirror ----
  // The pond IS the scene: nothing stands in the open water the lantern is reflected in,
  // and nothing stands between the camera and the physics (the mountain-car lesson).
  const notShallow: ExcludeFn = (x, z) =>
    bankHeight(x, z) < WATER_Y - 0.85 || bankHeight(x, z) > WATER_Y - 0.02;
  const nearCamera: ExcludeFn = (_x, z) => z > -1.4;
  const shallowReeds = scatterGrass({
    count: 240,
    area: { x: 0, z: -8, halfX: 13, halfZ: 6.5 },
    seed: ctx.seed ^ 0x51c7,
    tier,
    exclude: excludeAny(notShallow, nearCamera),
    heightAt: () => WATER_Y - 0.05,
    color: blendTokens(palette.groundA, TOKEN_BLACK, 0.55),
    scale: [0.35, 0.85],
    stretch: [1.3, 2.3],
    hueJitter: 0.22,
    wind: 0.07,
    windFrequency: 1.15,
    rimColor: palette.skyHorizon,
    rimStrength: 0.7,
    name: 'pondReeds',
  });
  group.add(shallowReeds.group);
  owned.push(shallowReeds);
  timed.push(shallowReeds);

  const shoreGrass = scatterGrass({
    count: 4200,
    area: { x: 0, z: -13, halfX: 17, halfZ: 6.5 },
    seed: ctx.seed ^ 0x7f42,
    tier,
    exclude: (x, z) => terrain.sampleHeight(x, z) < WATER_Y + 0.08,
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.75,
    color: blendTokens(palette.groundB, palette.groundA, 0.5),
    scale: [0.18, 0.4],
    stretch: [0.8, 1.3],
    hueJitter: 0.2,
    wind: 0.05,
    windFrequency: 1.4,
    rimColor: palette.skyHorizon,
    rimStrength: 0.5,
    name: 'shoreGrass',
  });
  group.add(shoreGrass.group);
  owned.push(shoreGrass);
  timed.push(shoreGrass);

  const farTrees = scatterTrees({
    archetypes: ['broadleaf', 'conifer'],
    count: 64,
    area: { x: 0, z: -12.5, halfX: 18, halfZ: 5.5 },
    seed: ctx.seed ^ 0x1a3d,
    tier,
    exclude: (x, z) => terrain.sampleHeight(x, z) < WATER_Y + 0.5 || (Math.abs(x) < 4.5 && z > -7.5),
    heightAt: (x, z) => terrain.sampleHeight(x, z),
    maxSlope: 0.6,
    color: blendTokens(palette.hemiGround, TOKEN_BLACK, 0.35),
    scale: [0.5, 1.2],
    stretch: [0.9, 1.5],
    hueJitter: 0.2,
    wind: 0.03,
    windFrequency: 0.75,
    rimColor: palette.skyHorizon,
    rimStrength: 0.55,
    castShadow: false,
    name: 'farTrees',
  });
  group.add(farTrees.group);
  owned.push(farTrees);
  timed.push(farTrees);

  // ---- lily pads: the only thing allowed to float on the mirror ----
  // They give the water a scale and a near-field, and they are kept strictly out of the
  // swing's own column so the lantern's reflection always lands on open water.
  const padGeometry = makePadGeometry(tier);
  const padMaterial = oakMaterial({
    color: TOKEN_WHITE,
    vertexColors: true,
    side: THREE.DoubleSide,
    roughness: 0.72,
    rimColor: palette.skyHorizon,
    rimStrength: 0.5,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.3,
    tier,
    name: 'lilyPad',
  });
  const pads = buildPads({
    geometry: padGeometry,
    material: padMaterial,
    seed: ctx.seed,
    tier,
    colors: [
      blendTokens(palette.groundA, TOKEN_BLACK, 0.62),
      blendTokens(palette.groundB, palette.hemiGround, 0.78),
      blendTokens(palette.groundA, palette.skyTop, 0.45),
    ],
    bankHeight,
  });
  group.add(pads);
  owned.push({
    dispose: () => {
      pads.dispose();
      padGeometry.dispose();
      padMaterial.dispose();
    },
  });

  // ---- the lantern's glint pool: the mirror the swing skims ----
  // A horizontal streak ON the surface rather than a mirrored copy of the lantern: at
  // this near-profile camera the reflected image of a light IS a streak drawn toward the
  // viewer, and a streak costs one draw and can never be occluded wrongly.
  const glintSprite = radialSprite(64, 1.7);
  const glintMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(palette.sun, accent, 0.3), 1.6, {
    map: glintSprite,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'lanternGlint',
  });
  glintMaterial.opacity = 0;
  const glintGeometry = new THREE.PlaneGeometry(1.05, 3.4);
  glintGeometry.rotateX(-Math.PI / 2);
  const glint = new THREE.Mesh(glintGeometry, glintMaterial);
  glint.name = 'lanternGlint';
  glint.position.set(0, WATER_Y + 0.015, 0);
  glint.renderOrder = 3;
  glint.visible = false;
  group.add(glint);
  owned.push({
    dispose: () => {
      glintGeometry.dispose();
      glintMaterial.dispose();
      glintSprite.dispose();
    },
  });

  // ---- atmosphere ----
  // Kit particle/mist presets are tuned for ~20-unit fields (canon lesson 5). This
  // diorama is ~5 units of subject seen from 5.7, so sizes are scaled ~4× down.
  const mist: MistPlanesHandle = makeMistPlanes({
    size: [22, 12],
    color: blendTokens(palette.fog, palette.skyHorizon, 0.4),
    seed: ctx.seed,
    tier,
    layers: 3,
    baseY: WATER_Y + 0.1,
    spread: 0.42,
    opacity: 0.07,
    speed: 0.007,
    scale: 2.1,
    // the near half of the pond stays a clean mirror: mist is a DEPTH cue here, and a
    // horizontal plane seen at this grazing angle veils the whole surface if it is let
    nearFade: 6.5,
    name: 'pondMist',
  });
  mist.group.position.set(0, 0, -6);
  group.add(mist.group);
  owned.push(mist);
  timed.push(mist);

  const fireflies: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'fireflies',
    count: 260,
    bounds: { x: 0, y: WATER_Y + 0.75, z: -1.6, halfX: 6.4, halfY: 0.7, halfZ: 4.2 },
    seed: ctx.seed ^ 0x2ab9,
    tier,
    color: blendTokens(gold, palette.groundB, 0.32),
    size: 2.6,
    intensity: 2.4,
    opacity: 0.9,
    swirl: 0.22,
    name: 'pondFireflies',
  });
  group.add(fireflies.points);
  owned.push(fireflies);
  timed.push(fireflies);

  const pollen: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'pollen',
    count: 420,
    bounds: { x: -0.4, y: 0.35, z: -1.1, halfX: 4.4, halfY: 1.5, halfZ: 2.2 },
    seed: ctx.seed ^ 0x6d31,
    tier,
    color: blendTokens(palette.sun, palette.skyHorizon, 0.35),
    size: 1.4,
    intensity: 0.95,
    opacity: 0.26,
    rise: 0.05,
    swirl: 0.1,
    name: 'dawnPollen',
  });
  group.add(pollen.points);
  owned.push(pollen);
  timed.push(pollen);

  // petal rain — off until a success, then a slow blossom fall out of the canopy
  const petals: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'snow',
    count: 420,
    // the canopy the petals fall OUT of, not the whole sky (canon lesson 6: a small
    // frame needs half the amplitude of the default grammar, all of the beat)
    bounds: { x: WILLOW_X * 0.42, y: 0.55, z: -1.3, halfX: 2.9, halfY: 1.75, halfZ: 1.9 },
    seed: ctx.seed ^ 0x3ce1,
    tier,
    color: blendTokens(accent, TOKEN_WHITE, 0.15),
    size: 1.5,
    intensity: 1.15,
    opacity: 0.72,
    rise: -0.5,
    swirl: 0.32,
    name: 'willowPetals',
  });
  petals.setDensity(0);
  group.add(petals.points);
  owned.push(petals);
  timed.push(petals);

  timed.push({ setTime: (t) => setMaterialsTime(animated, t) });

  const bulbAttr = bulbs.instanceColor;
  let flareIdle = false;

  return {
    group,
    sky,
    rig,
    water,
    terrain,
    garlandRoot,
    waterY: WATER_Y,

    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
    },

    setDawn(k01) {
      const k = clampRange(k01, 0, 1);
      // the light coming up puts the fireflies out and burns the mist off
      fireflies.setIntensity(2.4 * (1 - k) + 0.15);
      fireflies.setDensity(0.25 + 0.75 * (1 - k));
      mist.setOpacity(0.035 + 0.075 * (1 - k));
      pollen.setIntensity(0.55 + 0.75 * k);
    },

    setFlare(progress) {
      if (!bulbAttr) return;
      const idle = !(progress >= 0);
      if (idle && flareIdle) return;
      flareIdle = idle;
      const e = idle ? -1 : clampRange(progress, 0, 1);
      for (let i = 0; i < bulbCount; i++) {
        // a bump that leaves the centre and travels out to both stakes, then settles
        const phase = e - (bulbSpan[i] ?? 0) * 0.6;
        const k = idle ? 0 : clampRange(1 - Math.abs(phase) / 0.3, 0, 1);
        const gain = 1 + 6.5 * k * k;
        bulbGain[i * 3] = gain;
        bulbGain[i * 3 + 1] = gain;
        bulbGain[i * 3 + 2] = gain;
      }
      bulbAttr.needsUpdate = true;
    },

    setPetals(env01) {
      petals.setDensity(clampRange(env01, 0, 1));
    },

    setGlint(x, z, gain) {
      const g = clampRange(gain, 0, 1);
      glint.visible = g > 0.004;
      if (!glint.visible) return;
      glint.position.set(x, WATER_Y + 0.015, z);
      glintMaterial.opacity = 0.62 * g;
      const spread = 1 + (1 - g) * 1.6;
      glint.scale.set(spread, 1, 0.75 + spread * 0.55);
    },

    setPixelRatio(dpr) {
      fireflies.setPixelRatio(dpr);
      pollen.setPixelRatio(dpr);
      petals.setPixelRatio(dpr);
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

// ---------------------------------------------------------------------------
// willow construction
// ---------------------------------------------------------------------------

/**
 * The bough skeleton, in world space. The FIRST bough is the load-bearing one: it leaves
 * the trunk, arches over the pond and passes over the env's pivot at `BOUGH_Y` — one
 * `BOUGH_Z` BEHIND the swing plane, so the trapeze's hanger is a real attachment rather
 * than a floating hinge AND the arm never has to pass through the limb it hangs from.
 * The whole stretch inside the reach keeps |z| ≥ `BOUGH_Z`; the tail only curls back
 * toward the camera once it is clear of the disc (x = 2.35 is already r = 2.48).
 */
export function willowBoughs(): readonly (readonly (readonly [number, number, number])[])[] {
  const trunkTop: readonly [number, number, number] = [WILLOW_X + 0.95, 1.72, WILLOW_Z];
  const main: (readonly [number, number, number])[] = [
    trunkTop,
    [WILLOW_X + 1.55, 1.36, WILLOW_Z * 0.72],
    [WILLOW_X + 2.35, 0.83, WILLOW_Z * 0.46],
    [-0.85, BOUGH_Y + 0.16, BOUGH_Z],
    [0, BOUGH_Y, BOUGH_Z],
    [1.15, BOUGH_Y + 0.1, BOUGH_Z],
    [2.35, BOUGH_Y + 0.46, BOUGH_Z + 0.06],
  ];
  const upper: (readonly [number, number, number])[] = [
    [WILLOW_X + 0.72, 1.5, WILLOW_Z],
    [WILLOW_X + 1.2, 2.05, WILLOW_Z - 0.35],
    [WILLOW_X + 2.0, 2.3, WILLOW_Z - 0.8],
    [WILLOW_X + 3.1, 2.24, WILLOW_Z - 1.1],
  ];
  const back: (readonly [number, number, number])[] = [
    [WILLOW_X + 0.55, 1.2, WILLOW_Z],
    [WILLOW_X + 1.1, 1.5, WILLOW_Z - 1.5],
    [WILLOW_X + 2.1, 1.62, WILLOW_Z - 2.6],
    [WILLOW_X + 3.4, 1.45, WILLOW_Z - 3.2],
  ];
  const front: (readonly [number, number, number])[] = [
    [WILLOW_X + 0.85, 1.62, WILLOW_Z],
    [WILLOW_X + 1.35, 2.15, WILLOW_Z + 1.4],
    [WILLOW_X + 2.35, 2.42, WILLOW_Z + 2.6],
    [WILLOW_X + 3.9, 2.5, WILLOW_Z + 3.4],
  ];
  return [main, upper, back, front];
}

/** Rotate + place a +y cylinder so it runs from `a` to `b` (world space). */
function alignSegment(
  g: THREE.BufferGeometry,
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): void {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  const len = Math.hypot(dx, dy, dz) || 1;
  const q = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(dx / len, dy / len, dz / len),
  );
  const m = new THREE.Matrix4().makeRotationFromQuaternion(q);
  m.setPosition((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5);
  g.applyMatrix4(m);
}

/**
 * One willow frond, built going UP from the origin. The kit's vertex wind reads OBJECT
 * space y (`transformed.y / uWindHeight`), so a strand modelled downward would never
 * sway; every instance is rotated π about z instead, which puts the high-sway tip at the
 * bottom — exactly what a willow does.
 */
export function makeFrondGeometry(length: number): THREE.BufferGeometry {
  const SEGMENTS = 4;
  const g = new THREE.PlaneGeometry(0.038, length, 1, SEGMENTS);
  g.translate(0, length * 0.5, 0);
  const pos = g.getAttribute('position');
  const cols = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const t = clampRange(pos.getY(i) / length, 0, 1);
    // taper to a point, and BOW: the strand leaves the bough almost vertically and curls
    // away at the tip, which is the whole silhouette of a willow
    pos.setX(i, pos.getX(i) * (0.26 + 0.74 * t) + t * t * 0.34);
    pos.setZ(i, pos.getZ(i) + (1 - t) * (1 - t) * 0.14);
    const k = 0.55 + 0.6 * t;
    cols[i * 3] = k;
    cols[i * 3 + 1] = k;
    cols[i * 3 + 2] = k;
  }
  pos.needsUpdate = true;
  g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'willowFrond';
  return g;
}

interface FrondOptions {
  geometry: THREE.BufferGeometry;
  material: OakMaterial;
  boughs: readonly (readonly (readonly [number, number, number])[])[];
  seed: number;
  tier: QualityTier;
  length: number;
  colors: readonly THREE.Color[];
}

/**
 * Hang fronds off every bough. The hard placement rules: nothing may enter the arm's
 * swing disc, nothing may hang down through the frame the arm and the lights live in,
 * and a foreground strand may only appear at the very edges of the shot.
 */
function buildFronds(opts: FrondOptions): THREE.InstancedMesh & { dispose(): void } {
  const perTier = opts.tier === 'thumb' ? 0.18 : opts.tier === 'medium' ? 0.5 : 1;
  const wanted = Math.max(8, Math.round(620 * perTier));
  const rng = streamFor(opts.seed, 'fronds');
  const mesh = new THREE.InstancedMesh(opts.geometry, opts.material, wanted);
  mesh.name = 'willowFronds';
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(wanted * 3), 3);
  mesh.castShadow = false;
  mesh.receiveShadow = false;

  const m = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const s = new THREE.Vector3();
  const col = new THREE.Color();
  let n = 0;
  let guard = 0;
  while (n < wanted && guard < wanted * 24) {
    guard++;
    const bough = opts.boughs[Math.min(opts.boughs.length - 1, (rng() * opts.boughs.length) | 0)];
    if (!bough) continue;
    const t = rng() * (bough.length - 1);
    const i = Math.min(bough.length - 2, Math.floor(t));
    const f = t - i;
    const a = bough[i] as readonly [number, number, number];
    const b = bough[i + 1] as readonly [number, number, number];
    const x = a[0] + (b[0] - a[0]) * f + (rng() - 0.5) * 0.3;
    const y = a[1] + (b[1] - a[1]) * f - 0.05;
    const z = a[2] + (b[2] - a[2]) * f + (rng() - 0.5) * 0.75;
    const scale = 0.55 + rng() * 0.95;
    const len = opts.length * scale;

    // the swing-disc rejection: only strands well out of the plane may hang low
    const inPlane = Math.abs(z) < 0.62;
    if (inPlane) {
      // closest approach of the vertical hang [y-len, y] to the pivot at the origin
      const yy = y - len > 0 ? y - len : y > 0 ? 0 : y;
      if (Math.hypot(x, yy) < REACH + SWING_CLEARANCE) continue;
    }
    // nothing at all between the camera and the physics: a foreground strand may only
    // hang at the very edges of the frame, high up
    if (z > 0.9 && (Math.abs(x) < 2.8 || y - len < 1.05)) continue;
    // ...nor may any strand hang down through the frame the arm and the lights live in
    if (Math.abs(z) < 3.4 && Math.abs(x) < 3.05 && y - len < 0.5) continue;

    p.set(x, y, z);
    // π about z hangs the strand tip-down; the yaw keeps the curtain from reading flat
    e.set(0, rng() * Math.PI * 2, Math.PI + (rng() - 0.5) * 0.5);
    q.setFromEuler(e);
    s.set(0.7 + rng() * 0.7, scale, 1);
    m.compose(p, q, s);
    mesh.setMatrixAt(n, m);
    col.copy(opts.colors[Math.min(opts.colors.length - 1, (rng() * opts.colors.length) | 0)] ?? opts.colors[0] ?? col);
    mesh.setColorAt(n, col);
    n++;
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();
  return mesh as THREE.InstancedMesh & { dispose(): void };
}


/** One lily pad: a notched disc lying flat, with a slight lift at the rim. */
export function makePadGeometry(tier: QualityTier): THREE.BufferGeometry {
  const seg = tier === 'thumb' ? 6 : 9;
  const g = new THREE.CircleGeometry(0.5, seg, 0.34, Math.PI * 2 - 0.68);
  g.rotateX(-Math.PI / 2);
  const pos = g.getAttribute('position');
  const cols = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const r = Math.hypot(pos.getX(i), pos.getZ(i));
    pos.setY(i, r * r * 0.16);
    const k = 0.75 + 0.5 * r;
    cols[i * 3] = k;
    cols[i * 3 + 1] = k;
    cols[i * 3 + 2] = k;
  }
  pos.needsUpdate = true;
  g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'lilyPad';
  return g;
}

interface PadOptions {
  geometry: THREE.BufferGeometry;
  material: OakMaterial;
  seed: number;
  tier: QualityTier;
  colors: readonly THREE.Color[];
  bankHeight(x: number, z: number): number;
}

function buildPads(opts: PadOptions): THREE.InstancedMesh & { dispose(): void } {
  const wanted = opts.tier === 'thumb' ? 8 : opts.tier === 'medium' ? 16 : 26;
  const rng = streamFor(opts.seed, 'lilies');
  const mesh = new THREE.InstancedMesh(opts.geometry, opts.material, wanted);
  mesh.name = 'lilyPads';
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(wanted * 3), 3);
  const m = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const s = new THREE.Vector3();
  const col = new THREE.Color();
  let n = 0;
  let guard = 0;
  while (n < wanted && guard < wanted * 40) {
    guard++;
    const x = (rng() * 2 - 1) * 7.5;
    const z = -5.5 + rng() * 10.5;
    // never under the swing, never on ground that is about to break the surface
    if (Math.abs(x) < 1.55 && z > -1.6 && z < 1.6) continue;
    if (opts.bankHeight(x, z) > WATER_Y - 0.35) continue;
    const scale = 0.38 + rng() * 0.5;
    p.set(x, WATER_Y + 0.012 + rng() * 0.006, z);
    e.set((rng() - 0.5) * 0.09, rng() * Math.PI * 2, (rng() - 0.5) * 0.09);
    q.setFromEuler(e);
    s.set(scale, scale, scale);
    m.compose(p, q, s);
    mesh.setMatrixAt(n, m);
    col.copy(opts.colors[Math.min(opts.colors.length - 1, (rng() * opts.colors.length) | 0)] ?? opts.colors[0] ?? col);
    mesh.setColorAt(n, col);
    n++;
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();
  return mesh as THREE.InstancedMesh & { dispose(): void };
}
