// The Firefly Cave — the two state-bearing props (MemoryChain-cont-v0).
//
// THE HONESTY RULE LIVES IN THIS FILE. `cueCyan` and `cueMagenta` enter the scene graph in
// exactly two places, and nowhere else in the whole cave:
//
//   * `buildCueCrystal` — the crystal on cell 0. Its `setGlow(0)` is a HARD CUT, not a
//     fade: the emissive shell and the halo are `visible = false` AND their intensity is
//     zero, so there is no additive contribution, no bloom feed and no coloured point
//     light anywhere in the frame the instant the agent steps off cell 0. The crystal body
//     that remains is neutral quartz whose colour never depends on the cue — it is the
//     same dark stone whichever way the coin fell.
//   * `buildExitMouth` — the two mouths on cell 31. They are ALWAYS lit, ALWAYS equally,
//     one cyan and one magenta. They are the CHOICE, not the answer, so nothing about the
//     hidden cue may ever make one brighter than the other. The only asymmetry allowed is
//     the resolution itself: a correct choice flares the mouth the agent actually picked.
//     A mouth carries its colour on THREE channels — the throat lens, the arch stone's
//     fresnel rim, and (where the tier compiles the rim away) the stone's own albedo and
//     self-emission — because "always lit, always equally" is a claim about PIXELS, and a
//     carrier the tier contract deletes is a claim the viewer never sees. `mouthCarriers()`
//     is what the test asserts, per tier, for exactly that reason.
//
// ACCURACY. Both handles expose a `root` whose world position is EXACTLY `cellToWorld` of
// the env's own cell with y = 0 and nothing else. Every shard, arch, glow and halo hangs
// off a child, so deleting the art would leave a correctly-placed empty node.

import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveColor,
  setEmissiveIntensity,
  tierEffectScale,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import {
  blendTokens,
  clamp01,
  mergeColoredParts,
  tierBounce,
  tierToonFloor,
  type CaveSkin,
  type ColoredPart,
} from './world';

// ---------------------------------------------------------------------------
// the cue crystal
// ---------------------------------------------------------------------------

/** how far off the walkway the cluster stands, so the wisp on cell 0 is never hidden */
const CRYSTAL_OFFSET_Z = -0.82;
/** the tallest shard's tip, in world units above the walkway */
export const CRYSTAL_HEIGHT = 1.62;
/**
 * HDR gain of the lit crystal. Canon lesson 22: emissive gain is palette-relative, and the
 * cave key light is 0.2 — one fifteenth of noon. A gain that would be a lens flare at noon
 * is barely a glow here, so this sits well above anything in the FourRooms courtyard and
 * is still the only object in the frame that crosses the bloom threshold.
 */
const CRYSTAL_HDR = 1.7;

/**
 * Emissive gain factor per tier FOR THE CRYSTAL. Thumb has NO bloom pass, so an HDR emitter
 * there does not spill light — it just clips its brightest channels and comes out WHITE.
 * Cyan at 1.75 is (0.19, 1.30, 1.75) before tone mapping: two channels pinned, one dark,
 * i.e. a pale grey shard where a cyan one was intended. Less gain at thumb keeps the hue,
 * which at wall size is the only thing about this object that still has to be read.
 */
export function emissiveTierGain(tier: QualityTier): number {
  return tier === 'thumb' ? 0.72 : 1;
}

/**
 * ...and the MOUTHS' gain goes the other way, because the two jobs are opposites.
 *
 * The crystal at thumb is a big near object covering hundreds of pixels: its risk is
 * clipping to white, so its gain comes down. The mouths at thumb are forty units away in a
 * frame that has no bloom, no point lights and — this is the one that broke them — no
 * fresnel rim, because `tierEffectScale('thumb').rim === 0` (kit/materials). Every carrier
 * the tier contract leaves a mouth is one the mouth owns itself, so its gain goes UP.
 */
export function mouthTierGain(tier: QualityTier): number {
  return tier === 'thumb' ? 1.12 : 1;
}

/**
 * How much of a mouth's OWN colour is mixed into its arch stone, per tier.
 *
 * This is `tierLift` applied to hue instead of value, and for exactly the same reason: the
 * cave has to fight its own quality tiers. At hero and medium the arch stone catches its
 * mouth's colour through the fresnel rim, which is the prettiest carrier there is. At thumb
 * the kit sets `rim = 0` and the rim shader chunk is not even compiled, so the two
 * colour-coded choices — the thing this env IS — rendered as two identical grey arches.
 * The colour therefore moves into the one channel no tier can take away: the stone's own
 * albedo and self-emission. It is symmetric between the two mouths by construction (each
 * gets its OWN colour, never the cue's), so it reveals exactly nothing.
 */
export function mouthStoneTint(tier: QualityTier): number {
  return tier === 'thumb' ? 0.62 : 0;
}

/** The same compensation on the stone's self-emission (the cave's stand-in for bounce). */
export function mouthStoneEmissive(tier: QualityTier): number {
  return tier === 'thumb' ? 0.7 : 0;
}

/** The fresnel rim a mouth's stone catches at rest — zeroed by the kit at thumb. */
export const MOUTH_RIM = 0.5;

/**
 * What is actually painting a mouth's colour onto the screen, read off the live materials.
 *
 * The lock this replaced read `drawnIntensity('…Throat')` — an additive plane that (as the
 * review measured) contributed nothing to the frame at any tier — so the instrument could
 * not see what the viewer saw, and thirty-nine green tests sat on top of two grey arches.
 * Every field below is a channel that reaches the raster, and the test asserts them per
 * tier: equal between the two mouths, non-zero, and each within the cue tolerance of its
 * OWN token.
 */
export interface MouthCarriers {
  /** fresnel rim strength compiled into the arch stone (0 at thumb — kit law) */
  rim: number;
  rimColor: THREE.Color;
  /** HDR intensity of the throat glow currently drawn (0 when hidden) */
  glow: number;
  glowColor: THREE.Color;
  /** the mouth's colour baked into the arch stone's albedo */
  stone: THREE.Color;
  /** ...and into its self-emission, times the intensity actually set */
  stoneEmissive: number;
  stoneEmissiveColor: THREE.Color;
  /** the sum of every carrier's magnitude — zero means an invisible mouth */
  total: number;
}

/**
 * Read a mouth's carriers out of a live scene graph by name. Exported so the test asserts
 * the pixels' own channels instead of a prop's private field.
 */
export function mouthCarriers(root: THREE.Object3D, name: string): MouthCarriers {
  const frame = root.getObjectByName(`${name}Frame`) as THREE.Mesh | undefined;
  const throat = root.getObjectByName(`${name}Throat`) as THREE.Mesh | undefined;
  const frameMaterial = frame?.material as OakMaterial | undefined;
  const throatMaterial = throat?.material as EmissiveMaterial | undefined;
  const handle = oakUniforms(frameMaterial);
  // the EFFECTIVE rim: a material whose rim strength was 0 at construction never compiled
  // the rim chunk at all, so its `uRimStrength` uniform is a number nothing reads
  const rim = handle && handle.flags.rim ? handle.uniforms.uRimStrength.value : 0;
  const throatColor = throatMaterial ? throatMaterial.color : new THREE.Color(0, 0, 0);
  const drawn = throat?.visible === true;
  return {
    rim,
    rimColor: handle ? handle.uniforms.uRimColor.value : new THREE.Color(0, 0, 0),
    glow: drawn ? Math.max(throatColor.r, throatColor.g, throatColor.b) : 0,
    glowColor: throatColor,
    stone: frameMaterial ? frameMaterial.color : new THREE.Color(0, 0, 0),
    stoneEmissive: frameMaterial ? frameMaterial.emissiveIntensity : 0,
    stoneEmissiveColor: frameMaterial ? frameMaterial.emissive : new THREE.Color(0, 0, 0),
    total:
      rim +
      (drawn ? Math.max(throatColor.r, throatColor.g, throatColor.b) : 0) +
      (frameMaterial ? frameMaterial.emissiveIntensity : 0),
  };
}

export interface CueCrystalOptions {
  tier: QualityTier;
  skin: CaveSkin;
  /** the initial cue colour; `setCue` is what actually drives it */
  cueColors: readonly [THREE.ColorRepresentation, THREE.ColorRepresentation];
  name?: string;
}

export interface CueCrystalHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(0, 0), y = 0, position only */
  root: THREE.Object3D;
  /** local height of the glowing core — the scene lifts its point light to it */
  readonly coreHeight: number;
  /** world-space z offset of the visible cluster (used to aim the point light) */
  readonly offsetZ: number;
  /** THE ONLY place a cue colour is written into the scene graph */
  setCue(value: 0 | 1): void;
  /** 0 = utterly dark (emitters hidden, zero intensity, no colour anywhere) */
  setGlow(gain: number): void;
  /** ctx.time() — the slow inner drift of a lit crystal */
  setTime(t: number): void;
  /** true when any cue-coloured emitter is currently drawn (the honesty assertion) */
  readonly lit: boolean;
  dispose(): void;
}

/**
 * One shard: a thin, tall octahedron with its base at the local origin. Faceted by
 * construction, so the toon ramp breaks across it like cut quartz.
 */
function shardGeometry(detail: number): THREE.BufferGeometry {
  const g = new THREE.OctahedronGeometry(0.5, detail);
  const pos = g.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    // squash the lower pyramid into a stubby root and stretch the upper one into a spire
    pos.setY(i, y > 0 ? y * 1.85 : y * 0.42);
  }
  pos.needsUpdate = true;
  g.translate(0, 0.21, 0);
  g.computeVertexNormals();
  g.name = 'cueShard';
  return g;
}

/** The cluster: seven shards of falling height, deterministic and identical every build. */
function clusterParts(detail: number, color: THREE.ColorRepresentation): ColoredPart[] {
  const spec: readonly (readonly [number, number, number, number, number])[] = [
    // [x, z, height, girth, lean]
    [0.0, 0.0, 1.0, 0.3, 0.0],
    [-0.24, 0.13, 0.62, 0.21, -0.22],
    [0.22, -0.11, 0.7, 0.22, 0.19],
    [0.05, 0.26, 0.45, 0.17, 0.1],
    [-0.13, -0.24, 0.38, 0.16, -0.12],
    [0.33, 0.19, 0.3, 0.13, 0.26],
    [-0.35, -0.05, 0.26, 0.12, -0.3],
  ];
  const parts: ColoredPart[] = [];
  for (const s of spec) {
    const g = shardGeometry(detail);
    g.scale(s[3], s[2] * CRYSTAL_HEIGHT, s[3]);
    g.rotateZ(s[4]);
    g.rotateY(s[0] * 3.1 + s[1] * 1.7);
    g.translate(s[0], 0, s[1]);
    parts.push({ geometry: g, color });
  }
  return parts;
}

export function buildCueCrystal(opts: CueCrystalOptions): CueCrystalHandle {
  const { tier, skin } = opts;
  const detail = tier === 'thumb' ? 0 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'cueCrystal';

  // ---- the probe root: position only, forever ----
  const root = new THREE.Object3D();
  root.name = 'cueCrystalRoot';
  group.add(root);

  // ---- the cluster stands beside the walkway, on a decoration child ----
  const cluster = new THREE.Object3D();
  cluster.name = 'cueCrystalCluster';
  cluster.position.set(0, 0, CRYSTAL_OFFSET_Z);
  root.add(cluster);

  // ---- the plinth: plain cave rock, no cue colour, ever ----
  // It carries a submerged pier as well as the visible drum. The crystal stands beside the
  // walkway, i.e. over the pools, and a plinth that stopped at the waterline would read as
  // a prop floating on the water instead of a rock growing out of it.
  const plinthParts: ColoredPart[] = [];
  {
    const pier = new THREE.CylinderGeometry(0.66, 0.95, 1.5, tier === 'thumb' ? 6 : 9, 1);
    pier.translate(0, -0.72, 0);
    plinthParts.push({ geometry: pier, color: blendTokens(skin.rockLow, skin.rockHigh, 0.5) });
    const base = new THREE.CylinderGeometry(0.52, 0.66, 0.3, tier === 'thumb' ? 6 : 9, 1);
    base.translate(0, 0.13, 0);
    plinthParts.push({ geometry: base, color: skin.rockLow });
    const lip = new THREE.CylinderGeometry(0.38, 0.5, 0.13, tier === 'thumb' ? 6 : 9, 1);
    lip.translate(0, 0.31, 0);
    plinthParts.push({ geometry: lip, color: blendTokens(skin.rockLow, skin.mineral, 0.4) });
  }
  const plinthGeometry = mergeColoredParts(plinthParts, 'cueCrystalPlinth');
  const rockMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: skin.rim,
    rimStrength: 0.7,
    emissive: skin.rockLow,
    emissiveIntensity: tierBounce(tier),
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: tierToonFloor(tier, 0.34),
    tier,
    name: 'cueCrystalRock',
  });
  const plinth = new THREE.Mesh(plinthGeometry, rockMaterial);
  plinth.name = 'cueCrystalPlinth';
  plinth.castShadow = tier !== 'thumb';
  plinth.receiveShadow = tier !== 'thumb';
  cluster.add(plinth);

  // ---- the quartz body: NEUTRAL. This is what a dark crystal looks like. ----
  const quartzGeometry = mergeColoredParts(
    clusterParts(detail, skin.quartz),
    'cueCrystalQuartz',
  );
  const quartzMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.42,
    metalness: 0.05,
    rimColor: skin.rim,
    rimStrength: 1.25,
    rimPower: 2.1,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: tierToonFloor(tier, 0.42),
    tier,
    name: 'cueCrystalQuartz',
  });
  const quartz = new THREE.Mesh(quartzGeometry, quartzMaterial);
  quartz.name = 'cueCrystalQuartz';
  quartz.position.y = 0.3;
  quartz.castShadow = tier !== 'thumb';
  cluster.add(quartz);

  // ---- the light INSIDE the crystal: an additive shell hugging the same shards ----
  const glowGeometry = mergeColoredParts(clusterParts(detail, 0xffffff), 'cueCrystalGlow');
  const glowGain = CRYSTAL_HDR * emissiveTierGain(tier);
  const glowMaterial: EmissiveMaterial = emissiveMaterial(opts.cueColors[0], glowGain, {
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    name: 'cueCrystalGlow',
  });
  const glow = new THREE.Mesh(glowGeometry, glowMaterial);
  glow.name = 'cueCrystalGlow';
  glow.position.y = 0.3;
  glow.scale.setScalar(1.055);
  cluster.add(glow);

  // ---- and the soft bloom around it ----
  const haloGeometry = new THREE.IcosahedronGeometry(0.95, tier === 'thumb' ? 0 : 1);
  const haloMaterial: EmissiveMaterial = emissiveMaterial(opts.cueColors[0], glowGain * 0.24, {
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'cueCrystalHalo',
  });
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.name = 'cueCrystalHalo';
  halo.position.y = 0.92;
  halo.scale.set(1, 1.35, 1);
  cluster.add(halo);

  let glowLevel = 0;
  glow.visible = false;
  halo.visible = false;

  return {
    group,
    root,
    coreHeight: 0.95,
    offsetZ: CRYSTAL_OFFSET_Z,

    setCue(value) {
      const color = opts.cueColors[value === 1 ? 1 : 0];
      setEmissiveColor(glowMaterial, color);
      setEmissiveColor(haloMaterial, color);
    },

    setGlow(gain) {
      const g = clamp01(gain);
      glowLevel = g;
      // HARD gate. `visible = false` is belt and braces over the zeroed intensity: an
      // additive emitter at intensity 0 already contributes nothing, and a hidden one
      // cannot even be argued with.
      const on = g > 0;
      glow.visible = on;
      halo.visible = on;
      setEmissiveIntensity(glowMaterial, glowGain * g);
      setEmissiveIntensity(haloMaterial, glowGain * 0.24 * g);
      halo.scale.set(1 + 0.1 * g, 1.35 + 0.16 * g, 1 + 0.1 * g);
    },

    setTime(t) {
      oakUniforms(quartzMaterial)?.setTime(t);
      oakUniforms(rockMaterial)?.setTime(t);
      // the inner light drifts, the crystal itself never moves
      if (glowLevel > 0) {
        const breath = 1 + 0.09 * Math.sin(t * 1.35) + 0.05 * Math.sin(t * 2.9 + 1.1);
        glow.scale.setScalar(1.055 * breath);
        halo.rotation.y = t * 0.32;
      }
    },

    get lit() {
      return glowLevel > 0;
    },

    dispose() {
      plinthGeometry.dispose();
      quartzGeometry.dispose();
      glowGeometry.dispose();
      haloGeometry.dispose();
      rockMaterial.dispose();
      quartzMaterial.dispose();
      glowMaterial.dispose();
      haloMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// the exit mouths
// ---------------------------------------------------------------------------

/** how far the two mouths splay to either side of the passage axis */
const MOUTH_OFFSET_Z = 1.42;
/** how far past the last stone the arches stand */
const MOUTH_OFFSET_X = 1.15;
export const MOUTH_HEIGHT = 3.3;
/**
 * The mouths' resting HDR gain: DIM. They have to read as two possibilities at 40 units
 * through two thirds of fog, and they must never look like an answer. Below the bloom
 * threshold at rest; only the flare of a correct choice takes one of them over it.
 *
 * Measured, not guessed. On a 1600×900 hero still the brightest 5% of each mouth reads
 * (100,179,193) and (191,79,156) — hue distance 0.158 and 0.055 from `cueCyan` /
 * `cueMagenta`, i.e. comfortably inside the scene's own `isCueColored` tolerance of 0.22.
 * At 1.75 the cores clipped toward white and cyan measured 0.46 — a mouth so bright it had
 * lost the colour that is its entire job.
 */
const MOUTH_HDR = 0.72;

export interface ExitMouthOptions {
  tier: QualityTier;
  skin: CaveSkin;
  /** cueCyan or cueMagenta — the mouth's identity, fixed for the scene's whole life */
  color: THREE.ColorRepresentation;
  /** +1 = the near bank (+z), −1 = the far bank (−z) */
  side: 1 | -1;
  name?: string;
}

/**
 * The throat glow: a soft coloured lens standing in the arch's opening.
 *
 * It carries its own falloff in its VERTEX COLOURS — white at the centre, black at the rim
 * — over additive blending, so black contributes nothing and the edge dissolves. The first
 * build used a mapped `PlaneGeometry` instead and the review measured what that cost: the
 * arch interiors read (18,22,31) against an empty-cave (21,26,38), i.e. the throat painted
 * NOTHING at any tier while two draws and a shared 64² sprite paid for it, and every
 * honesty assertion in the suite was reading that invisible material. A vertex-coloured fan
 * has no texture to go wrong: the falloff is geometry, and geometry always rasterizes.
 *
 * The lens is dished AWAY from the camera (its centre sits deepest), so it reads as a lit
 * passage receding into the rock rather than a disc leaning on the arch.
 */
function throatLensGeometry(tier: QualityTier, halfZ: number, halfY: number, depth: number): THREE.BufferGeometry {
  const seg = tier === 'thumb' ? 12 : tier === 'medium' ? 18 : 26;
  // radius, brightness — a long soft shoulder, not a linear ramp and not a disc: a bright
  // core that has faded to a quarter by 60% of the way out reads as a passage the light is
  // coming FROM, where a flat ramp reads as a lamp hung in the doorway
  const rings: readonly (readonly [number, number])[] = [
    [0, 1],
    [0.28, 0.8],
    [0.55, 0.45],
    [0.8, 0.16],
    [1, 0],
  ];
  const count = 1 + seg * (rings.length - 1);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  // the centre vertex, deepest into the rock
  positions[0] = depth;
  colors[0] = colors[1] = colors[2] = 1;
  let v = 1;
  for (let r = 1; r < rings.length; r++) {
    const ring = rings[r] as readonly [number, number];
    const radius = ring[0];
    const bright = ring[1];
    for (let s = 0; s < seg; s++) {
      const a = (2 * Math.PI * s) / seg;
      positions[v * 3] = depth * (1 - radius * radius);
      positions[v * 3 + 1] = halfY * radius * Math.sin(a);
      positions[v * 3 + 2] = halfZ * radius * Math.cos(a);
      colors[v * 3] = bright;
      colors[v * 3 + 1] = bright;
      colors[v * 3 + 2] = bright;
      v++;
    }
  }
  const indices = new Uint16Array((seg + seg * 2 * (rings.length - 2)) * 3);
  let o = 0;
  for (let s = 0; s < seg; s++) {
    indices[o++] = 0;
    indices[o++] = 1 + s;
    indices[o++] = 1 + ((s + 1) % seg);
  }
  for (let r = 1; r < rings.length - 1; r++) {
    const inner = 1 + (r - 1) * seg;
    const outer = 1 + r * seg;
    for (let s = 0; s < seg; s++) {
      const s2 = (s + 1) % seg;
      indices[o++] = inner + s;
      indices[o++] = outer + s;
      indices[o++] = outer + s2;
      indices[o++] = inner + s;
      indices[o++] = outer + s2;
      indices[o++] = inner + s2;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeBoundingSphere();
  g.name = 'exitMouthThroatLens';
  return g;
}

export interface ExitMouthHandle {
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(31, 0), y = 0, position only */
  root: THREE.Object3D;
  /** local height of the throat glow — the scene lifts its point light to it */
  readonly glowHeight: number;
  readonly offsetZ: number;
  /** 1 = resting, 0 = doused (a wrong choice), > 1 = the flare of a correct one */
  setGlow(gain: number): void;
  setTime(t: number): void;
  dispose(): void;
}

/**
 * One mouth: a rough rock arch turned outward from the passage, with a soft coloured
 * throat behind it. Both mouths are the same geometry mirrored, so nothing about their
 * SHAPE could ever hint at which one is right.
 */
export function buildExitMouth(opts: ExitMouthOptions): ExitMouthHandle {
  const { tier, skin, side } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'exitMouth';

  const root = new THREE.Object3D();
  root.name = `${group.name}Root`;
  group.add(root);

  const arch = new THREE.Object3D();
  arch.name = `${group.name}Arch`;
  arch.position.set(MOUTH_OFFSET_X, 0, MOUTH_OFFSET_Z * side);
  // A SHALLOW splay. The two mouths are symmetric about the passage axis but the camera is
  // not: at ±0.46 rad the far mouth presented its glow almost face-on while the near one
  // was 47° foreshortened, and one of the two choices simply looked brighter than the
  // other — which is exactly the impression this scene must never give. At ±0.24 both
  // throats read the same from the composed frame, and the fork still forks.
  arch.rotation.y = -0.24 * side;
  root.add(arch);

  // ---- the rock frame ----
  /** the arch's inner radius and springing height — the throat is sized off both */
  const ringR = MOUTH_HEIGHT * 0.36;
  const ringY = MOUTH_HEIGHT * 0.58;
  const parts: ColoredPart[] = [];
  {
    const radial = Math.max(4, Math.round(9 * detail));
    const tubular = Math.max(8, Math.round(20 * detail));
    const ring = new THREE.TorusGeometry(ringR, 0.19, radial, tubular, Math.PI);
    ring.translate(0, ringY, 0);
    ring.rotateY(Math.PI / 2);
    parts.push({ geometry: ring, color: skin.rockLow });
    for (const sz of [-1, 1]) {
      const jamb = new THREE.BoxGeometry(0.3, ringY, 0.3);
      jamb.translate(0, ringY * 0.5, sz * ringR);
      parts.push({ geometry: jamb, color: blendTokens(skin.rockLow, skin.rockHigh, 0.45) });
    }
    const lintel = new THREE.BoxGeometry(0.34, 0.3, 0.4);
    lintel.translate(0, ringY + ringR + 0.06, 0);
    parts.push({ geometry: lintel, color: skin.mineral });
    // the landing, and the rock it stands on: the mouths are out over the pools, so a
    // sill that stopped at the waterline would leave two arches floating on black
    const sill = new THREE.BoxGeometry(1.5, 0.2, ringR * 2.5);
    sill.translate(-0.25, 0.05, 0);
    parts.push({ geometry: sill, color: skin.rockLow });
    const pier = new THREE.BoxGeometry(1.15, 1.7, ringR * 2.05);
    pier.translate(-0.25, -0.86, 0);
    parts.push({ geometry: pier, color: blendTokens(skin.rockLow, skin.rockHigh, 0.55) });
  }
  const frameGeometry = mergeColoredParts(parts, `${group.name}Frame`);
  // THE TIER-INDEPENDENT CARRIER. `color` multiplies the per-part vertex palette, so one
  // number tints the whole arch toward its own mouth's colour without disturbing the stone's
  // internal value hierarchy — and at hero/medium the tint is 0, i.e. the arch is the exact
  // neutral rock it has always been and the fresnel rim does the work. At thumb, where the
  // kit compiles the rim away entirely, this is the only channel left.
  const stoneTint = mouthStoneTint(tier);
  const emissiveTint = mouthStoneEmissive(tier);
  const frameMaterial: OakMaterial = oakMaterial({
    color: stoneTint > 0 ? blendTokens(0xffffff, opts.color, stoneTint) : 0xffffff,
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: opts.color,
    // the arch stone catches a little of its own mouth's colour on its rim. That is
    // symmetric between the two mouths by construction, so it reveals nothing.
    rimStrength: MOUTH_RIM,
    emissive:
      emissiveTint > 0 ? blendTokens(skin.rockLow, opts.color, emissiveTint) : skin.rockLow,
    emissiveIntensity: tierBounce(tier),
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: tierToonFloor(tier, 0.34),
    tier,
    name: `${group.name}Frame`,
  });
  const frame = new THREE.Mesh(frameGeometry, frameMaterial);
  frame.name = `${group.name}Frame`;
  frame.castShadow = tier !== 'thumb';
  frame.receiveShadow = tier !== 'thumb';
  arch.add(frame);

  // ---- the throat: the lit passage inside the arch ----
  // Sized to sit ENTIRELY inside the aperture: the jambs stand at ±ringR and the arc
  // springs at ringY, so a lens centred at 0.65·ringY with semi-axes (0.80·ringR, 0.55·
  // ringY) clears the sill at the bottom, the jambs at the sides and the arc at the top.
  // The first build ran the lens out past the sill and its lower lobe hung in front of the
  // landing as a lit tent — a glow that escapes its own arch stops reading as a passage.
  const throatGeometry = throatLensGeometry(tier, ringR * 0.8, ringY * 0.55, 0.62);
  const throatGain = MOUTH_HDR * mouthTierGain(tier);
  const throatMaterial: EmissiveMaterial = emissiveMaterial(opts.color, throatGain, {
    transparent: true,
    opacity: 1,
    depthWrite: false,
    // FRONT side only: the lens is a dish, wound so its faces look back down the passage,
    // and the far side of a solid rock arch is not a thing anyone can stand behind
    side: THREE.FrontSide,
    blending: THREE.AdditiveBlending,
    name: `${group.name}Throat`,
  });
  // the falloff lives in the mesh's vertex colours; additive black is additive nothing
  throatMaterial.vertexColors = true;
  const throat = new THREE.Mesh(throatGeometry, throatMaterial);
  throat.name = `${group.name}Throat`;
  throat.position.set(0.02, ringY * 0.65, 0);
  // THE ONE THAT KILLED THE FIRST BUILD. `kit/water` puts its plane at `renderOrder = 1`,
  // so every renderOrder-0 transparent is drawn BEFORE it — and the far pools are visible
  // straight through both apertures, because the cavern is open at that end and there is no
  // opaque geometry behind a mouth at all. A 0.97-opacity water plane then repainted the
  // throat out of the frame at every tier, which is why the arch interiors measured
  // (18,22,31) against an empty-cave (21,26,38). The glow lives in front of the water it is
  // seen against, so it is ordered in front of it.
  throat.renderOrder = 2;
  arch.add(throat);

  // the bounce the stone gets from its own throat: modulated by the glow so a doused mouth
  // dims its own stone and a flare warms it, at every tier
  const bounce = tierBounce(tier);
  const bounceLift = emissiveTint > 0 ? 1.15 : 0;
  // `setRim` writes the uniform RAW, so the tier scale `oakMaterial` applied at construction
  // has to be re-applied here or a medium mouth would silently run at the hero rim
  const rimScale = tierEffectScale(tier).rim;
  let gain = 1;

  return {
    group,
    root,
    glowHeight: ringY * 0.65,
    offsetZ: MOUTH_OFFSET_Z * side,

    setGlow(g) {
      gain = g < 0 ? 0 : g;
      throat.visible = gain > 0.001;
      setEmissiveIntensity(throatMaterial, throatGain * gain);
      oakUniforms(frameMaterial)?.setRim(MOUTH_RIM * rimScale * Math.min(1, gain));
      frameMaterial.emissiveIntensity = bounce * (1 + bounceLift * Math.min(1.6, gain));
    },

    setTime(t) {
      oakUniforms(frameMaterial)?.setTime(t);
      // the throat breathes very slowly — a cave draught, not a beacon
      const breath = 1 + 0.07 * Math.sin(t * 0.55 + (side > 0 ? 0 : 1.9));
      throat.scale.set(1, breath, breath);
    },

    dispose() {
      frameGeometry.dispose();
      throatGeometry.dispose();
      frameMaterial.dispose();
      throatMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
