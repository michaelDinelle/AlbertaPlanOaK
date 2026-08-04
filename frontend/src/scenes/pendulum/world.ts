// The Lantern Bell — the static world (pond, torii, banks, koi, atmosphere).
//
// THE ACCURACY CLAIM OF THIS FILE: nothing here re-derives the pendulum. The torii is
// built AROUND the decoder's `pivot` anchor and the volume the LANTERN sweeps as it hangs
// from it — `dims.ts` derives that envelope from the very numbers `actors.ts` builds the
// lantern out of. The tie beam clears the tassel at the top of the arc, the hanger cables
// stand off the swing plane far enough that the paper barrel passes BETWEEN them, and the
// pillars sit outside the swept annulus. Change `PD_L` or any lantern dimension and the
// arch resizes with it.
//
// The first pass sized the arch against the bare rod (`ROD_LEN + 0.19`, a hand-written
// guess 0.13 short of the truth) and put the cables at ±0.105 inside a 0.135 barrel, so
// at |θ| ≳ 2.8 — the fleet init is θ = π — the tassel weight was buried in the beam and
// the cables cut a dark line across the lit paper. Nothing here is a hand-written
// clearance any more; every one is derived and the test re-measures the built meshes.
//
// The pond bed is a bowl whose SHORELINE is where the bed crosses `WATER_LEVEL`, so the
// reeds and rocks are scattered by asking the terrain where the water's edge is, never by
// a hand-authored ring. The one thing the bed must not do is rise above the water under
// the arch — the lantern's reflection lives there.
//
// LIGHT: the moon sits upper-left and BEHIND, so the arch is a rim-lit silhouette against
// the star field and the only warm light in the frame comes from the lantern itself. That
// is the whole point of the scene: the shaped reward is the lighting, so when the agent is
// far from its cost minimum the pond really does go cold and dark.

import * as THREE from 'three';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import {
  emissiveMaterial,
  oakMaterial,
  setEmissiveIntensity,
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
import { radialSprite } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import { valueNoise2 } from '@kit/textures';
import { excludeAny, scatterGrass, scatterRocks, type ExcludeFn } from '@kit/vegetation';
import { makeWater, type WaterHandle } from '@kit/water';
import type { EnvGeometry } from '@sim/types';
import {
  HANGER_PLATE_DROP,
  HANGER_PLATE_H,
  HANGER_PLATE_W,
  PIN_R,
  ROD_LEN,
  STRAP_HALF,
  STRAP_Z,
  SWING_CLEARANCE,
  SWING_REACH,
} from './dims';

// ---------------------------------------------------------------------------
// world constants (1 env unit === 1 world unit — the swing circle really is r = 1)
// ---------------------------------------------------------------------------

/** Still-water height. 0.62 below the bob's lowest point, so the lantern never dips. */
export const WATER_LEVEL = -1.62;
/** The lantern's reflection sits on the water, camera-ward of the swing plane. */
export const REFLECT_Z = 0.92;
export const REFLECT_LIFT = 0.014;

/** Pond bed: an elliptical bowl. These are the ellipse's semi-axes. */
export const POND_RX = 3.55;
export const POND_RZ = 2.85;
/** bank height (bed at the ellipse rim) and bowl depth below it */
const BANK_Y = -1.3;
const BOWL_DEPTH = 1.46;
const BANK_RISE = 0.62;

export const TERRAIN_W = 15;
export const TERRAIN_D = 12;
export const TERRAIN_SEG_X = 140;
export const TERRAIN_SEG_Z = 112;
export const WATER_W = 7.6;
export const WATER_D = 6;

/** fog reach derived from the pond footprint, never hand-tuned per shot */
export const FOG_EXTENT = TERRAIN_W * 2.3;

/**
 * Left and BEHIND, and deliberately LOW. A 40° moon backlights the arch beautifully and
 * is completely invisible: the camera's vertical FOV is fixed at 40°, so anything above
 * ~18° of elevation is out of frame. At 14° the disc sits just under the crown beam, the
 * arch is still a rim-lit silhouette, and the moon's own silver path lands on the water.
 */
export const MOON_DIR: readonly [number, number, number] = [-0.30, 0.175, -0.938];

// ---- torii geometry, all derived from the lantern's swept envelope ----------
/**
 * The tie beam the lantern hangs from. The binding surface is the UNDERSIDE OF THE HANGER
 * PLATE, not the beam — the plate is the lowest static thing inside the swing plane, so
 * it is what has to clear the tassel's apex at the top of the arc.
 */
const NUKI_Y = SWING_REACH + SWING_CLEARANCE + HANGER_PLATE_DROP + HANGER_PLATE_H * 0.5;
const SHIMAKI_Y = NUKI_Y + 0.21; // the thin beam under the crown
const KASAGI_Y = NUKI_Y + 0.34; // the crowning beam
const PILLAR_X = ROD_LEN + 0.62;
const PILLAR_TOP = NUKI_Y + 0.28;
const PILLAR_FOOT = -2.32;
const KASAGI_HALF = PILLAR_X + 0.63;
const NUKI_HALF = PILLAR_X + 0.33;
/** the tie beam is deep enough in ±z to carry the cables under it */
const NUKI_HALF_Z = STRAP_Z + 0.018;
/** …and the plate that bolts them on spans them without overhanging the beam */
const HANGER_PLATE_Z = STRAP_Z * 2 + 0.028;

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

/** `noon.sun` is the token white — the legal way for a night scene to say "moonlit". */
export const TOKEN_WHITE = TOKENS.palettes.noon.sun;

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

/**
 * Merge pre-transformed parts into ONE geometry carrying a flat per-part vertex color,
 * disposing the sources. A torii is a dozen boxes in three timbers' worth of colour; this
 * makes it one mesh and one material.
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

/** The env's own pivot anchor, never a scene re-derivation (STYLE.md accuracy law). */
export function pivotOf(geometry: EnvGeometry): readonly [number, number, number] {
  return geometry.classic?.anchors.pivot ?? [0, 0, 0];
}

// ---------------------------------------------------------------------------
// the pond bed
// ---------------------------------------------------------------------------

/**
 * An elliptical bowl: `BANK_Y - BOWL_DEPTH` at the centre, `BANK_Y` at the ellipse rim,
 * then a gentle rise into the surrounding moor. The value noise is weighted DOWN inside
 * the bowl (the bed under the lantern's reflection stays smooth) and up on the banks.
 */
export function makePondBed(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const r = Math.hypot(x / POND_RX, z / POND_RZ);
    const bowl = 1 - smoothstep01(r);
    const bank = smoothstep01((r - 1) / 1.25);
    const n1 = valueNoise2(x * 0.72 + 4.1, z * 0.72 + 2.3, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 2.05 + 9.7, z * 2.05 + 1.1, 64, seed ^ 0x4d2f13) - 0.5;
    // the shoreline gets the MOST break-up: a bowl with no noise at r ≈ 1 reads as a
    // stamped ellipse, and the eye finds a perfect ellipse instantly.
    const shore = 1 - smoothstep01(Math.abs(r - 1) / 0.5);
    const noiseW = 0.22 + 1.35 * bank + 0.75 * shore;
    return BANK_Y - BOWL_DEPTH * bowl + BANK_RISE * bank + (n1 * 0.15 + n2 * 0.055) * noiseW;
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
  /** the torii's hanging point — a probe object, position only */
  pivotRoot: THREE.Object3D;
  /** the lantern's warm reflection on the water — a probe object, position only */
  reflectionRoot: THREE.Object3D;
  /** the mote field that rides the bob (index.ts moves it with the lantern) */
  motes: THREE.Points;
  /** the rig's lantern light, or undefined below hero/medium tier */
  lanternLight: THREE.PointLight | undefined;
  /** ctx.time() — pause freezes reeds, koi, fireflies and mist together */
  setTime(t: number): void;
  /**
   * The one state channel of the whole world: `serenity` ∈ [0,1] is the shaped reward
   * rendered as light (see index.ts), `lanternX` is where the lantern hangs.
   */
  setLight(serenity: number, lanternX: number, time: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

interface Disposable {
  dispose(): void;
}

interface Timed {
  setTime(t: number): void;
}

const KOI_COUNT: Record<QualityTier, number> = { hero: 9, medium: 6, thumb: 3 };
const PAD_COUNT: Record<QualityTier, number> = { hero: 15, medium: 10, thumb: 5 };

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const tier: QualityTier = ctx.tier;
  const palette: Palette = ctx.tokens.palettes.night;
  const accent = ctx.tokens.accents['pendulum'] ?? ctx.tokens.celebrationGold;
  const pivot = pivotOf(ctx.geometry);
  const group = new THREE.Group();
  group.name = 'lanternBell';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const shadows = tier !== 'thumb';

  // ---- palette-derived surface colors (blends of tokens only) ----
  // The torii's vermilion is dusk's warm horizon dragged most of the way into night: a
  // weathered red that only wakes up when the lantern is lit.
  const vermilion = blendTokens(TOKENS.palettes.dusk.skyBottom, palette.skyTop, 0.44);
  const vermilionDark = blendTokens(TOKENS.palettes.dusk.skyBottom, palette.skyTop, 0.62);
  const timberShade = blendTokens(vermilion, palette.hemiGround, 0.4);
  const stone = blendTokens(palette.groundB, palette.hemiSky, 0.42);
  const iron = blendTokens(palette.hemiGround, palette.skyTop, 0.35);
  const bedDeep = blendTokens(palette.skyTop, palette.hemiGround, 0.5);
  const bedSilt = blendTokens(palette.skyTop, TOKENS.palettes.dusk.groundA, 0.3);
  const bankMoss = blendTokens(palette.groundA, TOKENS.accents['four-rooms'] ?? palette.groundB, 0.2);
  const bankStone = blendTokens(palette.groundB, palette.hemiSky, 0.3);

  // ---- sky (exactly one): the moon disc, its halo, and the star lattice ----
  const sky = skyFromPalette(palette, 'night', tier, {
    sunDir: MOON_DIR,
    // a crisp disc: at HDR 3.4 with a broad halo the bloom ate the whole upper-left
    // corner and the moon stopped reading as an object in the sky.
    sunSize: 0.038,
    sunHdr: 1.45,
    sunGlow: 0.06,
    sunGlowPower: 52,
    topSpread: 0.7,
    bottomSpread: 0.4,
    stars: {
      density: 225,
      size: 0.052,
      threshold: 0.9625,
      twinkle: 0.42,
      intensity: 2.7,
      color: blendTokens(palette.sun, TOKEN_WHITE, 0.45),
    },
    radius: 300,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  // ---- the one light budget ----
  // points[0] lantern, [1] pond wash, [2] the rim behind the arch — in priority order,
  // because `medium` only allocates two and `thumb` none.
  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 3,
    pointDistance: 7,
    sunDistanceScale: 2.6,
  });
  group.add(rig.group);
  owned.push(rig);
  // sized to the FRAME, not the arch: outside the ortho frustum three clamps the
  // shadow lookup to the border texel and paints amoebas across the pond.
  rig.setShadowFocus({ x: 0, y: -0.9, z: -0.2 }, 5.6);

  // ---- fog, always ----
  applyFog(scene, palette, FOG_EXTENT);

  // ---- the pond bed: the shoreline is where the bowl crosses the water ----
  const bed = makePondBed(ctx.seed);
  const rampColor = new THREE.Color();
  const terrain = makeTerrain({
    size: [TERRAIN_W, TERRAIN_D],
    segments: [TERRAIN_SEG_X, TERRAIN_SEG_Z],
    tier,
    seed: ctx.seed,
    heightFn: bed,
    jitter: 0,
    colorNoise: 0.05,
    slopeColor: bankStone,
    slopeStrength: 0.55,
    slopeSharpness: 2.4,
    colorRamp: (h) => {
      const submerged = smoothstep01((WATER_LEVEL - h) / 0.55);
      const dry = smoothstep01((h - WATER_LEVEL) / 0.5);
      rampColor.copy(bedSilt).lerp(bedDeep, submerged);
      rampColor.lerp(bankMoss, dry * 0.85);
      return rampColor;
    },
    materialOptions: {
      rimColor: palette.hemiSky,
      rimStrength: 0.18,
      toonSteps: 3,
      toonSoftness: 0.34,
      // the pond floor has to go BLACK where the lantern does not reach, or the pond
      // renders as bright as the sky and the whole night frame goes flat
      toonFloor: 0.05,
    },
    receiveShadow: shadows,
    castShadow: false,
    name: 'pondBed',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  const bedAt = (x: number, z: number): number => terrain.sampleHeight(x, z);

  // ---- the torii, straddling the swing circle -------------------------------
  // A probe root at the env's own pivot anchor. actors.ts hangs the rod from it; the
  // hanger straps below are pure decoration standing off in ±z.
  const pivotRoot = new THREE.Object3D();
  pivotRoot.name = 'toriiPivotRoot';
  pivotRoot.position.set(pivot[0], pivot[1], pivot[2]);
  group.add(pivotRoot);

  const toriiParts: ColoredPart[] = [];
  {
    const box = (
      w: number,
      h: number,
      d: number,
      x: number,
      y: number,
      z: number,
      color: THREE.ColorRepresentation,
      tiltZ = 0,
    ): void => {
      const g = new THREE.BoxGeometry(w, h, d);
      if (tiltZ !== 0) g.rotateZ(tiltZ);
      g.translate(x, y, z);
      toriiParts.push({ geometry: g, color });
    };

    // pillars — slightly battered (wider at the foot), sunk into the bed
    for (const sx of [-1, 1]) {
      const x = pivot[0] + sx * PILLAR_X;
      const h = PILLAR_TOP - PILLAR_FOOT;
      const pillar = new THREE.CylinderGeometry(0.098, 0.132, h, 9);
      pillar.translate(x, PILLAR_FOOT + h * 0.5, pivot[2]);
      toriiParts.push({ geometry: pillar, color: vermilion });
      // the collar band where the pillar meets the water
      const collar = new THREE.CylinderGeometry(0.148, 0.152, 0.07, 9);
      collar.translate(x, WATER_LEVEL + 0.03, pivot[2]);
      toriiParts.push({ geometry: collar, color: stone });
      // a stone base under the water
      const base = new THREE.CylinderGeometry(0.2, 0.28, 0.22, 9);
      base.translate(x, bedAt(x, pivot[2]) + 0.09, pivot[2]);
      toriiParts.push({ geometry: base, color: stone });
    }

    // nuki — the lower tie beam the lantern hangs from
    box(NUKI_HALF * 2, 0.1, NUKI_HALF_Z * 2, pivot[0], pivot[1] + NUKI_Y, pivot[2], timberShade);
    // shimaki — the thin beam under the crown
    box((KASAGI_HALF - 0.14) * 2, 0.072, NUKI_HALF_Z * 1.72, pivot[0], pivot[1] + SHIMAKI_Y, pivot[2], vermilion);
    // kasagi — the crown, and the DEEPEST timber in the arch (the classic profile: the
    // crown oversails everything). Three overlapping segments fake the upward sweep; the
    // ends must OVERLAP the centre span (the first pass left two floating slabs).
    box(KASAGI_HALF * 1.16, 0.125, NUKI_HALF_Z * 2.18, pivot[0], pivot[1] + KASAGI_Y, pivot[2], vermilionDark);
    for (const sx of [-1, 1]) {
      box(
        KASAGI_HALF * 0.78,
        0.115,
        NUKI_HALF_Z * 2.12,
        pivot[0] + sx * (KASAGI_HALF * 0.63),
        pivot[1] + KASAGI_Y + 0.031,
        pivot[2],
        vermilionDark,
        sx * -0.062,
      );
    }
    // gakuzuka — the little strut between nuki and shimaki, above the inverted bob
    box(0.1, SHIMAKI_Y - NUKI_Y - 0.09, 0.13, pivot[0], pivot[1] + (SHIMAKI_Y + NUKI_Y) * 0.5, pivot[2], vermilion);

    // The hanger: a plate under the nuki and two CABLES standing off in ±z, so the whole
    // LANTERN — not just the rod — swings between them and never intersects its own
    // support (`STRAP_Z` is derived from the barrel's half-width in dims.ts). They are
    // deliberately thin — the first pass made them thicker than the rod and the eye read
    // the support as the pendulum and the pendulum as a thread.
    box(
      HANGER_PLATE_W,
      HANGER_PLATE_H,
      HANGER_PLATE_Z,
      pivot[0],
      pivot[1] + NUKI_Y - HANGER_PLATE_DROP,
      pivot[2],
      iron,
    );
    for (const sz of [-1, 1]) {
      const top = pivot[1] + NUKI_Y - HANGER_PLATE_DROP - HANGER_PLATE_H * 0.6;
      const h = top - pivot[1];
      box(STRAP_HALF * 2, h, STRAP_HALF * 2, pivot[0], pivot[1] + h * 0.5, pivot[2] + sz * STRAP_Z, iron);
      // the bracket cheek that carries the pin
      const cheek = new THREE.CylinderGeometry(0.042, 0.042, 0.02, 8);
      cheek.rotateX(Math.PI / 2);
      cheek.translate(pivot[0], pivot[1], pivot[2] + sz * STRAP_Z);
      toriiParts.push({ geometry: cheek, color: iron });
    }
    // the pin itself, on the pivot, spanning the two cheeks — and thin enough to pass
    // through the rod's eye rather than through its wire (dims.ts sizes both)
    const pin = new THREE.CylinderGeometry(PIN_R, PIN_R, STRAP_Z * 2 + 0.04, 8);
    pin.rotateX(Math.PI / 2);
    pin.translate(pivot[0], pivot[1], pivot[2]);
    toriiParts.push({ geometry: pin, color: stone });
  }
  const toriiGeometry = mergeColoredParts(toriiParts, 'torii');
  const propMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.88,
    // moon-silver, and RESTRAINED: at 0.85 the fresnel repainted every beam's top face
    // chalk-white and the arch stopped reading as weathered timber at night.
    rimColor: blendTokens(palette.sun, palette.hemiSky, 0.45),
    rimStrength: 0.36,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.1,
    tier,
    name: 'lanternBellPropMaterial',
  });
  const toriiMesh = new THREE.Mesh(toriiGeometry, propMaterial);
  toriiMesh.name = 'torii';
  toriiMesh.castShadow = shadows;
  toriiMesh.receiveShadow = shadows;
  group.add(toriiMesh);
  owned.push({
    dispose: () => {
      toriiGeometry.dispose();
      propMaterial.dispose();
    },
  });

  // ---- the water ------------------------------------------------------------
  // Night water is DARKER than the night sky — that is the whole reason a lantern reads
  // on it. The first pass took the palette's sky colours straight and the pond rendered
  // as a pale lavender dish brighter than the horizon behind it.
  //
  // AND THE WARM BLENDS ARE TINY ON PURPOSE. `Color.lerp` runs in LINEAR space, where the
  // accent's red channel is 1.0 against this water's 0.006 — a 0.16 lerp toward it is a
  // 28× red lift, which is exactly how the pond ended up mauve for four screenshots. The
  // pond's warmth belongs to the caustics and the reflection, not to its albedo.
  const shallowCool = blendTokens(palette.skyBottom, palette.skyTop, 0.55);
  const shallowWarm = blendTokens(shallowCool, accent, 0.013);
  const deepCool = blendTokens(palette.skyTop, palette.fog, 0.35);
  const deepWarm = blendTokens(deepCool, accent, 0.006);
  const pond = makeWaterSurface(tier, palette, sky.sunDirection, accent);
  group.add(pond.handle.mesh);
  owned.push(pond.handle);
  const liveShallow = new THREE.Color();
  const liveDeep = new THREE.Color();

  // ---- koi: instanced, positions a pure function of (time, serenity, lanternX) ----
  const koi = buildKoi(KOI_COUNT[tier], ctx.seed, tier, palette, accent);
  group.add(koi.mesh);
  owned.push(koi);

  // ---- lotus pads floating on the rendered water surface ----
  const pads = buildLotusPads(PAD_COUNT[tier], ctx.seed, tier, palette, pond.handle.sampleHeight, bedAt);
  group.add(pads.mesh);
  owned.push(pads);

  // ---- the lantern's reflection: a warm smear on the water, under the lantern ----
  const reflectSprite = radialSprite(64, 1.7);
  const reflectMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(accent, palette.sun, 0.18), 1.5, {
    map: reflectSprite,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'lanternReflection',
  });
  reflectMaterial.opacity = 0;
  const reflectGeometry = new THREE.PlaneGeometry(0.56, 3.7);
  reflectGeometry.rotateX(-Math.PI / 2);
  const reflectionRoot = new THREE.Object3D();
  reflectionRoot.name = 'lanternReflectionRoot';
  reflectionRoot.position.set(pivot[0], WATER_LEVEL + REFLECT_LIFT, REFLECT_Z);
  const reflectMesh = new THREE.Mesh(reflectGeometry, reflectMaterial);
  reflectMesh.name = 'lanternReflection';
  reflectMesh.renderOrder = 3;
  reflectionRoot.add(reflectMesh);
  group.add(reflectionRoot);
  owned.push({
    dispose: () => {
      reflectGeometry.dispose();
      reflectMaterial.dispose();
      reflectSprite.dispose();
    },
  });

  // ---- the banks: reeds at the waterline, stones above it ----
  const submerged: ExcludeFn = (x, z) => bedAt(x, z) < WATER_LEVEL - 0.12;
  // NOTHING stands between the camera and the pond down the middle of the frame: the
  // lantern's reflection lives there. The near corners keep their reeds for depth.
  const nearCamera: ExcludeFn = (x, z) => z > 1.45 && Math.abs(x) < 2.6;
  const underArch: ExcludeFn = (x, z) => Math.abs(x) < PILLAR_X + 0.45 && Math.abs(z) < 0.5;

  const reeds = scatterGrass({
    count: 1700,
    area: { x: 0, z: -0.3, halfX: 5.4, halfZ: 4.1 },
    seed: ctx.seed ^ 0x51ab,
    tier,
    // the shoreline band, read off the terrain: never in open water, never up the moor
    exclude: excludeAny(submerged, nearCamera, underArch, (x, z) => bedAt(x, z) > WATER_LEVEL + 0.55),
    heightAt: bedAt,
    maxSlope: 0.95,
    color: blendTokens(palette.groundA, TOKENS.accents['four-rooms'] ?? palette.groundB, 0.3),
    // a 0.45-unit clump at stretch 2.2 is a ONE-METRE reed in a 2-metre frame: the first
    // pass grew a black palisade across the whole pond. This is a fringe, ankle-high.
    scale: [0.22, 0.5],
    stretch: [0.85, 1.3],
    hueJitter: 0.26,
    wind: 0.05,
    windFrequency: 0.85,
    rimColor: palette.hemiSky,
    rimStrength: 0.75,
    name: 'pondReeds',
  });
  group.add(reeds.group);
  owned.push(reeds);
  timed.push(reeds);

  const rocks = scatterRocks({
    count: 90,
    area: { x: 0, z: -0.4, halfX: 5.2, halfZ: 3.8 },
    seed: ctx.seed ^ 0x2b17,
    tier,
    exclude: excludeAny(nearCamera, underArch, (x, z) => {
      const h = bedAt(x, z);
      return h < WATER_LEVEL - 0.5 || h > WATER_LEVEL + 0.85;
    }),
    heightAt: bedAt,
    color: bankStone,
    scale: [0.14, 0.42],
    stretch: [0.45, 0.9],
    hueJitter: 0.24,
    rimColor: palette.hemiSky,
    rimStrength: 0.4,
    castShadow: shadows,
    name: 'pondStones',
  });
  group.add(rocks.group);
  owned.push(rocks);
  timed.push(rocks);

  // ---- atmosphere ----
  // Kit particle sizes are tuned for ~20-unit fields (canon lesson 5). This diorama is
  // ~8 units with the camera 6 out, so `gl_PointSize`'s 60/distance term wants roughly a
  // 25× reduction on the preset sizes.
  const fireflyColor = blendTokens(accent, TOKENS.accents['memory-chain'] ?? accent, 0.42);
  const fireflies: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'fireflies',
    count: 260,
    // ABOVE the water: the first band was centred on the surface and half the swarm
    // rendered as speckles inside the pond.
    bounds: { x: 0, y: WATER_LEVEL + 0.92, z: -0.6, halfX: 4.9, halfY: 0.5, halfZ: 3.2 },
    seed: ctx.seed ^ 0x7f19,
    tier,
    color: fireflyColor,
    size: 0.4,
    intensity: 2.3,
    opacity: 0.85,
    swirl: 0.3,
    name: 'reedFireflies',
  });
  group.add(fireflies.points);
  owned.push(fireflies);
  timed.push(fireflies);

  const mist: MistPlanesHandle = makeMistPlanes({
    size: [8.4, 6.4],
    color: blendTokens(palette.fog, palette.hemiSky, 0.45),
    seed: ctx.seed,
    tier,
    layers: 4,
    baseY: WATER_LEVEL + 0.06,
    spread: 0.5,
    opacity: 0.13,
    speed: 0.008,
    scale: 2.2,
    nearFade: 2.2,
    name: 'pondMist',
  });
  mist.group.position.set(0, 0, -0.5);
  group.add(mist.group);
  owned.push(mist);
  timed.push(mist);

  // motes rising off the lit paper — density follows the light, position follows the bob
  const embers: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'embers',
    count: 90,
    bounds: { halfX: 0.24, halfY: 0.42, halfZ: 0.2 },
    seed: ctx.seed ^ 0x3d91,
    tier,
    color: blendTokens(accent, palette.sun, 0.25),
    size: 0.24,
    intensity: 1.9,
    opacity: 0.7,
    rise: 0.22,
    swirl: 0.06,
    name: 'lanternMotes',
  });
  embers.setDensity(0);
  group.add(embers.points);
  owned.push(embers);

  // ---- the three point lights this scene claims from the rig ----
  const lanternLight = rig.points[0];
  if (lanternLight) {
    lanternLight.color.set(blendTokens(accent, palette.sun, 0.14));
    lanternLight.distance = 3.4;
  }
  const pondLight = rig.points[1];
  if (pondLight) {
    pondLight.color.set(blendTokens(accent, palette.hemiSky, 0.3));
    pondLight.position.set(pivot[0], WATER_LEVEL + 0.3, 0.35);
    pondLight.distance = 3.5;
  }
  const archLight = rig.points[2];
  if (archLight) {
    archLight.color.set(blendTokens(accent, TOKENS.palettes.dusk.skyBottom, 0.45));
    archLight.position.set(pivot[0], pivot[1] + 0.35, pivot[2] - 1.35);
    archLight.distance = 4;
  }

  return {
    group,
    sky,
    rig,
    terrain,
    pivotRoot,
    reflectionRoot,
    motes: embers.points,
    lanternLight,

    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
      pond.handle.setTime(t);
      pads.setTime(t);
      embers.setTime(t);
    },

    setLight(serenity, lanternX, time) {
      const s = clampRange(serenity, 0, 1);

      // the water: warm shallows and a warm floor when the lantern is lit
      liveShallow.copy(shallowCool).lerp(shallowWarm, s);
      liveDeep.copy(deepCool).lerp(deepWarm, s);
      pond.handle.setColors(liveShallow, liveDeep);
      pond.setGlow(s);

      // the reflection column, always under the lantern
      reflectionRoot.position.x = lanternX;
      reflectMaterial.opacity = 0.05 + 0.3 * s * s;
      reflectMesh.scale.set(0.6 + 0.6 * s, 1, 0.72 + 0.45 * s);
      setEmissiveIntensity(reflectMaterial, 0.7 + 0.85 * s);

      // the koi rise toward the light and gather under it
      koi.setState(time, s, lanternX);

      // motes off the paper (index.ts parks the field on the bob)
      embers.setDensity(s * s * 0.55);

      if (pondLight) pondLight.intensity = 0.05 + 1.15 * s;
      if (archLight) archLight.intensity = 0.02 + 0.5 * s;
      if (lanternLight) lanternLight.intensity = 0.08 + 1.7 * s;

      // the reeds lean a hair more in the cold: nothing but idle motion
      reeds.setWind(0.045 * (1 + 0.5 * (1 - s)), 0.85);
    },

    setPixelRatio(dpr) {
      fireflies.setPixelRatio(dpr);
      embers.setPixelRatio(dpr);
    },

    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// water
// ---------------------------------------------------------------------------

const BASE_CAUSTIC = 0.07;

interface PondHandle {
  handle: WaterHandle;
  /** 0..1 — the caustic web and the specular both wake up with the lantern */
  setGlow(s: number): void;
}

function makeWaterSurface(
  tier: QualityTier,
  palette: Palette,
  sunDir: THREE.Vector3,
  accent: number,
): PondHandle {
  const thumb = tier === 'thumb';
  const handle = makeWater({
    size: [WATER_W, WATER_D],
    palette,
    tier,
    segments: 88,
    level: WATER_LEVEL,
    amplitude: 0.045,
    // a COARSE lattice: `oakCaustic` raises its web to the third power, so at a fine
    // scale it stops being a web and becomes a field of white pinpricks.
    waveScale: 0.8,
    speed: 0.5,
    fresnel: 0.74,
    caustics: BASE_CAUSTIC,
    depthFalloff: 3.2,
    // deep enough to read as water, clear enough to keep the koi
    opacity: 0.74,
    sunDir,
    caustic: blendTokens(accent, palette.sun, 0.45),
    // The far shore is where the fresnel is strongest, so uSky is what puts a cool
    // sheen on the pond's far edge and keeps the near water dark.
    sky: blendTokens(palette.skyHorizon, palette.skyTop, 0.25),
    name: 'koiPond',
  });
  return {
    handle,
    setGlow(s) {
      // the caustic web is what makes the pond itself GLOW warm with the lantern; it
      // carries the "reward is the lighting" law onto the water, not a colour repaint
      handle.uniforms.uCausticStrength.value = thumb ? 0 : BASE_CAUSTIC * (0.08 + 0.92 * s);
      // a WHISPER of specular: the kit's 96-power glint on a 0.09-unit wave lattice
      // sprays the whole pond with white pinpricks at anything like full strength.
      handle.uniforms.uSpecular.value = thumb ? 0 : 0.45 + 0.85 * s;
    },
  };
}

// ---------------------------------------------------------------------------
// koi — instanced, and a pure function of (time, serenity, lanternX)
// ---------------------------------------------------------------------------

interface KoiHandle extends Disposable {
  mesh: THREE.InstancedMesh;
  setState(time: number, serenity: number, lanternX: number): void;
}

/** One koi: a tapered body plus a fan tail, nose along +x. ~44 triangles. */
function koiGeometry(): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const body = new THREE.IcosahedronGeometry(0.1, 1);
  body.scale(2.1, 0.62, 0.9);
  parts.push({ geometry: body, color: 0xffffff });
  const head = new THREE.ConeGeometry(0.062, 0.14, 6);
  head.rotateZ(-Math.PI / 2);
  head.translate(0.24, 0, 0);
  parts.push({ geometry: head, color: 0xffffff });
  const tail = new THREE.ConeGeometry(0.085, 0.17, 3);
  tail.rotateZ(Math.PI / 2);
  tail.scale(1, 1, 0.35);
  tail.translate(-0.27, 0.01, 0);
  parts.push({ geometry: tail, color: 0xdadada });
  const fin = new THREE.ConeGeometry(0.05, 0.1, 3);
  fin.rotateX(Math.PI / 2);
  fin.scale(1, 0.3, 1);
  fin.translate(0.02, 0.02, 0);
  parts.push({ geometry: fin, color: 0xdadada });
  return mergeColoredParts(parts, 'koi');
}

function buildKoi(
  count: number,
  seed: number,
  tier: QualityTier,
  palette: Palette,
  accent: number,
): KoiHandle {
  const geometry = koiGeometry();
  const material = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.55,
    rimColor: palette.hemiSky,
    rimStrength: 0.85,
    toonSteps: 2,
    toonSoftness: 0.45,
    // a high floor: these are pale fish under 74%-opaque water, and they have to hold
    // their shape on the unlit side or they vanish into the bed
    toonFloor: 0.55,
    tier,
    name: 'koiMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
  mesh.name = 'koiSchool';
  mesh.frustumCulled = false;
  mesh.count = count;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3);

  const white = blendTokens(TOKEN_WHITE, palette.hemiSky, 0.25);
  const gold = blendTokens(accent, TOKEN_WHITE, 0.25);
  const ember = blendTokens(accent, TOKENS.palettes.dusk.skyBottom, 0.4);
  const palettes = [white, gold, ember];

  interface Lane {
    radius: number;
    phase: number;
    speed: number;
    bobPhase: number;
    scale: number;
    squash: number;
  }
  const lanes: Lane[] = [];
  {
    const rng = streamFor(seed, 'koi');
    const c = new THREE.Color();
    for (let i = 0; i < count; i++) {
      lanes.push({
        radius: 0.55 + rng() * 1.55,
        phase: rng() * Math.PI * 2,
        speed: (0.11 + rng() * 0.16) * (rng() < 0.4 ? -1 : 1),
        bobPhase: rng() * Math.PI * 2,
        scale: 0.58 + rng() * 0.46,
        squash: 0.85 + rng() * 0.3,
      });
      c.copy(palettes[i % palettes.length] ?? white);
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  const m = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const s = new THREE.Vector3();

  return {
    mesh,
    setState(time, serenity, lanternX) {
      for (let i = 0; i < count; i++) {
        const lane = lanes[i] as Lane;
        const a = lane.phase + time * lane.speed;
        // the school tightens under the lantern as the light comes up
        const gather = 1 - 0.42 * serenity;
        const r = lane.radius * gather;
        const cx = lanternX * (0.15 + 0.45 * serenity);
        p.set(cx + Math.cos(a) * r * 1.5, 0, -0.15 + Math.sin(a) * r * 1.05);
        // and rises toward it
        const depth = 0.62 - 0.44 * serenity + 0.045 * Math.sin(time * 0.8 + lane.bobPhase);
        p.y = WATER_LEVEL - depth;
        // heading is the orbit tangent; a slow tail-driven roll rides on top
        const heading = lane.speed >= 0 ? a + Math.PI / 2 : a - Math.PI / 2;
        e.set(0, -heading, 0.14 * Math.sin(time * 1.7 + lane.bobPhase));
        q.setFromEuler(e);
        s.set(lane.scale, lane.scale * lane.squash, lane.scale);
        m.compose(p, q, s);
        mesh.setMatrixAt(i, m);
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
// lotus pads — they float on the RENDERED water surface, not a nominal level
// ---------------------------------------------------------------------------

interface PadHandle extends Disposable {
  mesh: THREE.InstancedMesh;
  setTime(t: number): void;
}

function buildLotusPads(
  count: number,
  seed: number,
  tier: QualityTier,
  palette: Palette,
  waterHeight: (x: number, z: number, t: number) => number,
  bedAt: (x: number, z: number) => number,
): PadHandle {
  // a notched disc: a circle with one wedge missing, the lotus-pad silhouette
  const geometry = new THREE.CircleGeometry(0.19, 11, 0.42, Math.PI * 2 - 0.84);
  geometry.rotateX(-Math.PI / 2);
  geometry.name = 'lotusPad';
  const material = oakMaterial({
    color: blendTokens(palette.groundA, TOKENS.accents['four-rooms'] ?? palette.groundB, 0.3),
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.72,
    rimColor: palette.hemiSky,
    rimStrength: 0.55,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.28,
    tier,
    name: 'lotusPadMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
  mesh.name = 'lotusPads';
  mesh.frustumCulled = false;
  mesh.count = 0;

  interface Pad {
    x: number;
    z: number;
    yaw: number;
    scale: number;
  }
  const pads: Pad[] = [];
  {
    const rng = streamFor(seed, 'lotus');
    for (let i = 0; i < count * 14 && pads.length < count; i++) {
      const x = (rng() * 2 - 1) * 3.1;
      const z = -0.4 + (rng() * 2 - 1) * 2.3;
      // only on real open water, and never right under the arch (the reflection lives there)
      if (bedAt(x, z) > WATER_LEVEL - 0.35) continue;
      if (Math.abs(x) < 0.85 && Math.abs(z) < 1.4) continue;
      pads.push({ x, z, yaw: rng() * Math.PI * 2, scale: 0.7 + rng() * 0.85 });
    }
    mesh.count = pads.length;
  }

  const m = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const s = new THREE.Vector3();

  return {
    mesh,
    setTime(t) {
      for (let i = 0; i < pads.length; i++) {
        const pad = pads[i] as Pad;
        p.set(pad.x, waterHeight(pad.x, pad.z, t) + 0.006, pad.z);
        // tilt with the swell: a two-sample slope of the same wave sum
        const dx = waterHeight(pad.x + 0.12, pad.z, t) - waterHeight(pad.x - 0.12, pad.z, t);
        const dz = waterHeight(pad.x, pad.z + 0.12, t) - waterHeight(pad.x, pad.z - 0.12, t);
        e.set(dz * 2.6, pad.yaw, -dx * 2.6);
        q.setFromEuler(e);
        s.setScalar(pad.scale);
        m.compose(p, q, s);
        mesh.setMatrixAt(i, m);
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
