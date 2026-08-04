// The Abyss — the three things down here that are alive (or were).
//
// ACCURACY. Every actor's `root` is an empty node whose world position is EXACTLY
// `cellToWorld(cell)` with y = 0, and nothing else — no hover, no tilt, no bob. The wreck's
// list, the anglerfish's swim and the bubble's wobble all live on CHILD nodes. Delete every
// mesh in this file and what remains is three correctly-placed empty nodes.
//
// The wreck carries a SECOND probe: `heartAnchor`, a bare node at a fixed local height over
// the goal cell. It is what the brief asks to be provable ("wreck heart at (15,15)"), and
// it is provable precisely because the glow that pulses on it is its child, not itself.
//
// CLOCKS. Nothing in this file owns a clock or integrates one. Every actor is driven by
// `setTime(ctx.time())` plus PURE WRITES of scene-computed scalars (`setFlare`, `setGain`),
// which is the pattern the sibling reef's clam uses (`deep-sea/actors.ts setState`). The
// wreck's heart flare used to decay internally on wall seconds, and that was a real bug in
// two directions: a PAUSED playback drained the celebration out of the mesh while the point
// light — driven by the scene's `ctx.time()` envelope — stayed pinned, and a SCRUB left the
// heart blazing for another wall second because a scrub carries no dt to decay. As a pure
// write of the scene's `treasure` envelope, both cases are correct by construction: pause
// freezes the beat mid-air, and a scrub zeroes `treasureAt` and therefore the flare.

import { emissiveMaterial, oakMaterial, oakUniforms, setEmissiveIntensity } from '@kit/materials';
import type { EmissiveMaterial, OakMaterial } from '@kit/materials';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import { blendTokens, clamp01, mergeColoredParts, type AbyssSkin, type ColoredPart } from './world';

// ---------------------------------------------------------------------------
// the sunken wreck — the treasure at (15, 15)
// ---------------------------------------------------------------------------

/** local height of the heart node over the goal cell — a probe constant, never animated */
export const HEART_Y = 0.62;
/**
 * The heart's HDR gain. Down here the key light is 0.35 and the hemisphere 0.45, so the
 * whole frame lives well under the 0.85 bloom threshold and an emitter has to be pushed
 * hard to become a light source rather than a bright pixel (canon lesson 22). At 0.98 —
 * the value the noon lantern-flower uses — this was a dull orange dot two cells wide.
 */
export const HEART_HDR = 2.6;
/**
 * The wreck is built at 1.35× a cell because of where it stands. (15, 15) is the FAR
 * corner from this scene's camera — 33 units out, past the fog, at the smallest end of the
 * perspective — and it is the thing the entire environment is about. At 1:1 it was a
 * six-pixel smudge under its own glow, which is a fine way to render a place nobody ever
 * reaches and a bad way to render a goal.
 */
export const WRECK_SCALE = 1.35;

export interface WreckOptions {
  tier: QualityTier;
  skin: AbyssSkin;
  /** the ice-blue scene accent, for the rim on the drowned timber */
  accent: THREE.ColorRepresentation;
  name?: string;
}

export interface WreckHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** probe object: world position === cellToWorld(goal), y = 0, always */
  root: THREE.Object3D;
  /** probe object: exactly HEART_Y above the goal cell — position only */
  heartAnchor: THREE.Object3D;
  /**
   * How hard the heart blazes, 0…1. A PURE WRITE, exactly like the sibling clam's
   * `setState`: the handle keeps no decay of its own, because the only honest source for
   * "how long ago was the treasure taken" is the scene's `ctx.time()` envelope.
   */
  setFlare(flare01: number): void;
  /** ctx.time() — the list, the heartbeat and the material clock */
  setTime(t: number): void;
  dispose(): void;
}

/**
 * A hull that broke its back: a keel lying across the corner cell, seven ribs arching out
 * of it (two of them snapped short), a stern block and a leaning mast stump. The heart —
 * whatever the ship was carrying — burns between the ribs, which is why the ribs are built
 * as an OPEN cage: the light has to get out through the gaps for the silhouette to read.
 */
export function buildWreck(opts: WreckOptions): WreckHandle {
  const { tier, skin } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'sunkenWreck';

  const root = new THREE.Object3D();
  root.name = 'wreckRoot';
  group.add(root);

  // the hull leans: a static pose on a CHILD, so the probe root stays pure
  const hull = new THREE.Object3D();
  hull.name = 'wreckHull';
  hull.rotation.set(0.1, -0.62, 0.13);
  hull.scale.setScalar(WRECK_SCALE);
  root.add(hull);

  const parts: ColoredPart[] = [];
  const radial = Math.max(4, Math.round(7 * detail));

  // keel
  {
    const keel = new THREE.BoxGeometry(1.58, 0.13, 0.3);
    keel.translate(0, 0.075, 0);
    parts.push({ geometry: keel, color: skin.timberDark });
    const spine = new THREE.BoxGeometry(1.34, 0.09, 0.11);
    spine.translate(-0.02, 0.2, 0);
    parts.push({ geometry: spine, color: skin.timber });
  }

  // ribs — an open cage, tallest amidships, two of them snapped
  const RIBS = tier === 'thumb' ? 4 : 7;
  for (let i = 0; i < RIBS; i++) {
    const t = RIBS > 1 ? i / (RIBS - 1) : 0.5;
    const x = -0.62 + t * 1.24;
    const fall = 1 - Math.pow(Math.abs(t - 0.46) * 2.1, 2);
    const height = 0.34 + 0.42 * Math.max(0.1, fall);
    const snapped = i === 1 || i === RIBS - 2;
    const arc = snapped ? Math.PI * 0.44 : Math.PI * 0.86;
    const rib = new THREE.TorusGeometry(height, 0.032, Math.max(3, Math.round(5 * detail)), Math.max(5, Math.round(11 * detail)), arc);
    rib.rotateZ(Math.PI - arc * 0.5 - (Math.PI - arc) * 0.5);
    rib.rotateY(Math.PI / 2);
    rib.translate(x, 0.12, 0);
    parts.push({ geometry: rib, color: i % 3 === 0 ? skin.timber : skin.timberDark });
  }

  // stern block + a mast stump snapped off at head height
  {
    const stern = new THREE.BoxGeometry(0.34, 0.46, 0.5);
    stern.rotateX(0.12);
    stern.translate(-0.66, 0.28, 0);
    parts.push({ geometry: stern, color: skin.timber });
    const mast = new THREE.CylinderGeometry(0.03, 0.055, 0.86, radial);
    mast.translate(0, 0.43, 0);
    mast.rotateZ(0.42);
    mast.rotateX(0.16);
    mast.translate(0.16, 0.18, 0.02);
    parts.push({ geometry: mast, color: skin.timberDark });
    const spar = new THREE.CylinderGeometry(0.022, 0.028, 0.52, Math.max(3, radial - 2));
    spar.rotateZ(Math.PI / 2);
    spar.translate(0.34, 0.66, -0.03);
    parts.push({ geometry: spar, color: skin.timber });
  }

  const hullGeometry = mergeColoredParts(parts, 'wreckHull');
  const hullMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.92,
    side: THREE.DoubleSide,
    rimColor: opts.accent,
    // the rim is doing real work here: it is most of what separates a black hull from a
    // black trench, and it is the only reason the ribs read as a cage at 35 units out
    rimStrength: 0.95,
    rimPower: 2.2,
    toonSteps: 2,
    toonSoftness: 0.35,
    toonFloor: 0.32,
    tier,
    name: 'wreckTimber',
  });
  const hullMesh = new THREE.Mesh(hullGeometry, hullMaterial);
  hullMesh.name = 'wreckTimberMesh';
  hullMesh.castShadow = tier !== 'thumb';
  hullMesh.receiveShadow = tier !== 'thumb';
  hull.add(hullMesh);

  // ---- the heart ----
  const heartAnchor = new THREE.Object3D();
  heartAnchor.name = 'wreckHeartAnchor';
  heartAnchor.position.set(0, HEART_Y, 0);
  root.add(heartAnchor);

  const heartDecor = new THREE.Object3D();
  heartDecor.name = 'wreckHeartDecor';
  heartDecor.scale.setScalar(WRECK_SCALE);
  heartAnchor.add(heartDecor);

  const coreGeometry = new THREE.IcosahedronGeometry(0.11, tier === 'thumb' ? 0 : 1);
  const coreMaterial: EmissiveMaterial = emissiveMaterial(skin.heart, HEART_HDR, { name: 'wreckHeartCore' });
  const core = new THREE.Mesh(coreGeometry, coreMaterial);
  core.name = 'wreckHeartCore';
  heartDecor.add(core);

  // the halo is deliberately small and thin. Bloom does the spreading; a fat additive
  // sphere on top of it turned the wreck into a streetlamp and ate its own ribs.
  const haloGeometry = new THREE.IcosahedronGeometry(0.23, tier === 'thumb' ? 0 : 1);
  const haloMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(skin.heart, opts.accent, 0.45), HEART_HDR * 0.24, {
    transparent: true,
    opacity: 0.26,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'wreckHeartHalo',
  });
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.name = 'wreckHeartHalo';
  heartDecor.add(halo);

  /** written ONLY by `setFlare`; never integrated, never decayed in here */
  let flareAmount = 0;

  return {
    group,
    root,
    heartAnchor,
    setFlare(flare01) {
      flareAmount = clamp01(flare01);
    },
    setTime(t) {
      // The wreck breathes on the swell; the heart beats faster than the hull moves,
      // which is the whole reason it reads as a heart and not as a lamp.
      hull.position.y = Math.sin(t * 0.42) * 0.012;
      hull.rotation.z = 0.13 + Math.sin(t * 0.33) * 0.014;
      const beat = 0.5 + 0.5 * Math.sin(t * 1.35);
      const pulse = 1 + 0.34 * beat + 2.6 * flareAmount;
      core.scale.setScalar(1 + 0.12 * beat + 0.5 * flareAmount);
      halo.scale.setScalar(1 + 0.2 * beat + 0.85 * flareAmount);
      heartDecor.position.y = Math.sin(t * 0.9) * 0.014;
      setEmissiveIntensity(coreMaterial, HEART_HDR * pulse);
      setEmissiveIntensity(haloMaterial, HEART_HDR * 0.24 * pulse);
      oakUniforms(hullMaterial)?.setTime(t);
    },
    dispose() {
      hullGeometry.dispose();
      hullMaterial.dispose();
      coreGeometry.dispose();
      coreMaterial.dispose();
      haloGeometry.dispose();
      haloMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// anglerfish — the lures that replace the sunbeams
// ---------------------------------------------------------------------------

export const LURE_HDR = 3.6;

export interface AnglerOptions {
  tier: QualityTier;
  skin: AbyssSkin;
  /** lure colour — this scene's accent, or its magenta counterpart */
  lure: THREE.ColorRepresentation;
  /** hover height of the fish over its shelf cell */
  hover: number;
  /** phase offset so two fish never swim in lockstep */
  phase: number;
  /** body scale */
  scale: number;
  name?: string;
}

export interface AnglerHandle {
  group: THREE.Group;
  /** probe object: world position === cellToWorld(cell), y = 0 */
  root: THREE.Object3D;
  /** the lure bulb's node — the scene reads its world position for the point light */
  lureAnchor: THREE.Object3D;
  setTime(t: number): void;
  dispose(): void;
}

/**
 * A blunt, ugly, beautiful fish: a fat tapered body, a tail fan, a jaw of teeth, and an
 * illicium — the fishing rod — arching forward off its skull with the esca (the bulb)
 * burning on the end. The body is nearly black on purpose: what you see from across the
 * trench is a moving light with a suggestion of an animal behind it, which is exactly what
 * you would see.
 */
export function buildAngler(opts: AnglerOptions): AnglerHandle {
  const { tier, skin } = opts;
  const detail = tier === 'thumb' ? 0.45 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'anglerfish';

  const root = new THREE.Object3D();
  root.name = `${opts.name ?? 'angler'}Root`;
  group.add(root);

  // everything that swims hangs here; the root never moves off its cell
  const swim = new THREE.Object3D();
  swim.name = `${opts.name ?? 'angler'}Swim`;
  swim.position.y = opts.hover;
  swim.scale.setScalar(opts.scale);
  root.add(swim);

  const parts: ColoredPart[] = [];
  const radial = Math.max(5, Math.round(9 * detail));
  const dark = skin.anglerBody;
  const belly = blendTokens(skin.anglerBody, skin.rockLip, 0.35);

  {
    const body = new THREE.SphereGeometry(0.23, radial, Math.max(4, Math.round(7 * detail)));
    body.scale(1.35, 1, 0.92);
    parts.push({ geometry: body, color: dark });
    const jaw = new THREE.SphereGeometry(0.15, Math.max(4, radial - 2), Math.max(3, Math.round(5 * detail)));
    jaw.scale(1.25, 0.55, 0.85);
    jaw.translate(0.19, -0.11, 0);
    parts.push({ geometry: jaw, color: belly });
    // tail fan
    const tail = new THREE.ConeGeometry(0.17, 0.3, Math.max(3, Math.round(5 * detail)), 1, true);
    tail.rotateZ(Math.PI / 2);
    tail.translate(-0.42, 0.02, 0);
    parts.push({ geometry: tail, color: dark });
    // dorsal
    const fin = new THREE.ConeGeometry(0.09, 0.2, 3, 1, true);
    fin.rotateX(-0.3);
    fin.translate(-0.06, 0.24, 0);
    parts.push({ geometry: fin, color: dark });
    // the rod, arching forward over the jaw
    const rod = new THREE.CylinderGeometry(0.011, 0.02, 0.5, Math.max(3, radial - 4));
    rod.translate(0, 0.25, 0);
    rod.rotateZ(-0.72);
    rod.translate(0.06, 0.16, 0);
    parts.push({ geometry: rod, color: dark });
    // teeth — four of them, just enough to be unsettling at hero tier
    if (tier !== 'thumb') {
      for (let i = 0; i < 4; i++) {
        const tooth = new THREE.ConeGeometry(0.014, 0.06, 3);
        tooth.rotateX(Math.PI);
        tooth.translate(0.22 + i * 0.045, -0.03, (i % 2 === 0 ? 1 : -1) * 0.055);
        parts.push({ geometry: tooth, color: skin.rockLip });
      }
    }
  }

  const bodyGeometry = mergeColoredParts(parts, 'anglerBody');
  const bodyMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.88,
    rimColor: opts.lure,
    rimStrength: 0.8,
    rimPower: 2.4,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.22,
    tier,
    name: 'anglerBodyMaterial',
  });
  const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
  bodyMesh.name = 'anglerBodyMesh';
  bodyMesh.castShadow = false;
  bodyMesh.receiveShadow = tier !== 'thumb';
  swim.add(bodyMesh);

  const lureAnchor = new THREE.Object3D();
  lureAnchor.name = `${opts.name ?? 'angler'}Lure`;
  lureAnchor.position.set(0.42, 0.44, 0.01);
  swim.add(lureAnchor);

  const escaGeometry = new THREE.IcosahedronGeometry(0.055, tier === 'thumb' ? 0 : 1);
  const escaMaterial: EmissiveMaterial = emissiveMaterial(opts.lure, LURE_HDR, { name: 'anglerEsca' });
  const esca = new THREE.Mesh(escaGeometry, escaMaterial);
  esca.name = 'anglerEsca';
  lureAnchor.add(esca);

  const glowGeometry = new THREE.IcosahedronGeometry(0.16, tier === 'thumb' ? 0 : 1);
  const glowMaterial: EmissiveMaterial = emissiveMaterial(opts.lure, LURE_HDR * 0.22, {
    transparent: true,
    opacity: 0.3,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'anglerEscaGlow',
  });
  const glow = new THREE.Mesh(glowGeometry, glowMaterial);
  glow.name = 'anglerEscaGlow';
  lureAnchor.add(glow);

  return {
    group,
    root,
    lureAnchor,
    setTime(t) {
      const p = opts.phase;
      // a slow hunting circle, plus the tail-driven yaw that makes it look propelled
      swim.position.x = Math.sin(t * 0.21 + p) * 0.55;
      swim.position.z = Math.cos(t * 0.17 + p * 1.7) * 0.42;
      swim.position.y = opts.hover + Math.sin(t * 0.29 + p * 0.6) * 0.17;
      swim.rotation.y = Math.PI * 0.15 + Math.sin(t * 0.21 + p) * 0.5;
      swim.rotation.z = Math.sin(t * 0.26 + p) * 0.08;
      const flicker = 0.72 + 0.28 * Math.sin(t * 1.9 + p * 3.1) + 0.1 * Math.sin(t * 5.3 + p);
      esca.scale.setScalar(0.9 + 0.16 * Math.sin(t * 1.9 + p * 3.1));
      glow.scale.setScalar(1 + 0.22 * Math.sin(t * 1.9 + p * 3.1));
      setEmissiveIntensity(escaMaterial, LURE_HDR * flicker);
      setEmissiveIntensity(glowMaterial, LURE_HDR * 0.22 * flicker);
      oakUniforms(bodyMaterial)?.setTime(t);
    },
    dispose() {
      bodyGeometry.dispose();
      bodyMaterial.dispose();
      escaGeometry.dispose();
      escaMaterial.dispose();
      glowGeometry.dispose();
      glowMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// the dive bubble — the wisp's diving bell
// ---------------------------------------------------------------------------

export interface DiveBubbleOptions {
  tier: QualityTier;
  /** the scene accent — the bubble's fresnel skin */
  accent: THREE.ColorRepresentation;
  /** the deep water the bubble is filled with */
  water: THREE.ColorRepresentation;
  radius?: number;
  name?: string;
}

export interface DiveBubbleHandle {
  /** add this ONE group to the scene; the scene positions it from the wisp every frame */
  group: THREE.Group;
  /**
   * How hard the bubble burns, 0…1. The brief's rule: it reads the ROW. A pure function
   * of state — no easing, no history, so a scrubbed frame shows exactly what a played one
   * would.
   */
  setGain(gain01: number): void;
  setTime(t: number): void;
  dispose(): void;
}

/**
 * A soft glass sphere the wisp rides inside. It is deliberately NOT an emissive ball: the
 * shell is an `oakMaterial` with a huge fresnel rim and almost no body, so it reads as a
 * curved surface catching what little light there is — and the rim gain is what the depth
 * drives. The inner shell is the only emissive part, and it stays under the bloom
 * threshold until the dive is well down the trench.
 */
export function buildDiveBubble(opts: DiveBubbleOptions): DiveBubbleHandle {
  const tier = opts.tier;
  const radius = opts.radius ?? 0.36;
  const group = new THREE.Group();
  group.name = opts.name ?? 'diveBubble';

  const decor = new THREE.Object3D();
  decor.name = 'diveBubbleDecor';
  group.add(decor);

  const shellGeometry = new THREE.IcosahedronGeometry(radius, tier === 'thumb' ? 1 : 2);
  const shellMaterial: OakMaterial = oakMaterial({
    color: opts.water,
    roughness: 0.15,
    metalness: 0,
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
    side: THREE.DoubleSide,
    toonSteps: 0,
    rimColor: opts.accent,
    rimStrength: 1.5,
    rimPower: 2.4,
    tier,
    name: 'diveBubbleShell',
  });
  const shell = new THREE.Mesh(shellGeometry, shellMaterial);
  shell.name = 'diveBubbleShell';
  shell.renderOrder = 3;
  decor.add(shell);

  const innerGeometry = new THREE.IcosahedronGeometry(radius * 0.82, tier === 'thumb' ? 0 : 1);
  const innerMaterial: EmissiveMaterial = emissiveMaterial(opts.accent, 0.5, {
    transparent: true,
    opacity: 0.13,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'diveBubbleInner',
  });
  const inner = new THREE.Mesh(innerGeometry, innerMaterial);
  inner.name = 'diveBubbleInner';
  decor.add(inner);

  let gain = 0;

  return {
    group,
    setGain(gain01) {
      gain = clamp01(gain01);
      // the fresnel skin brightens with depth: a soft shell up in the light, a lantern
      // on the floor. The base is low on purpose — a rim is additive and albedo-blind, so a
      // bright one over the surface rows' high-gain albedo bloomed the whole shallow end.
      oakUniforms(shellMaterial)?.setRim(0.5 + 3.4 * gain);
      shellMaterial.opacity = 0.16 + 0.16 * gain;
      setEmissiveIntensity(innerMaterial, 0.16 + 1.7 * gain);
      innerMaterial.opacity = 0.05 + 0.2 * gain;
    },
    setTime(t) {
      // a bubble is never quite still and never quite round
      const wobble = Math.sin(t * 1.7) * 0.035 + Math.sin(t * 2.9) * 0.018;
      shell.scale.set(1 + wobble, 1 - wobble * 0.8, 1 + wobble * 0.5);
      inner.scale.setScalar(1 - wobble * 0.4);
      decor.position.y = Math.sin(t * 1.15) * 0.012;
      decor.rotation.y = t * 0.28;
      oakUniforms(shellMaterial)?.setTime(t);
      inner.scale.multiplyScalar(1 + 0.05 * gain * Math.sin(t * 2.3));
    },
    dispose() {
      shellGeometry.dispose();
      shellMaterial.dispose();
      innerGeometry.dispose();
      innerMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
