// Sunlit Descent — the giant clam at the treasure cell, and the bubble the diver rides.
//
// ACCURACY: `clam.root` is the probe object. Its world position is EXACTLY
// `cellToReefWorld(goal.x, goal.y)` — the env's own goal cell, on its own terrace — with
// no hover, no sway and no lean. The pedestal, the two shells, the mantle and the pearl
// all hang off `body`, a child that carries the swell and the opening. Delete every
// decoration and what remains is a correctly-placed empty node.
//
// IDEMPOTENCE: `setState(open, flare)` is a pure write. Openness and flare are computed
// by the scene as pure functions of `(state, ctx.time(), the last goal event)`, so
// calling `setState` twice with the same arguments leaves the same world, and a scrub
// that snaps the clam to its resting gape lands exactly where a long-settled clam sits.

import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import { blendTokens, clamp01, mergeColoredParts, type ColoredPart, type ReefSkin } from './reef';

// ---------------------------------------------------------------------------
// the giant clam
// ---------------------------------------------------------------------------

/** shell radius — a giant clam fills most of its cell, which is how it reads as the goal */
const SHELL_R = 0.5;
/** how far the shells hinge apart at full gape, radians */
const GAPE_UPPER = 0.95;
const GAPE_LOWER = 0.34;
/** the hinge line, behind the shells' centre */
const HINGE_Z = -0.34;
const HINGE_Y = 0.17;
/** pearl centre above the terrace */
const PEARL_Y = 0.24;
/**
 * The pearl's HDR gain at rest. Canon lesson 22: emissive gain is palette-relative, and
 * the underwater key is 1.6 against a deep-blue column — a gain tuned on the noon
 * courtyard vanishes here. 1.3 keeps the pearl a POINT of light inside the shells; the
 * goal flare multiplies it, it does not replace it.
 */
const PEARL_HDR = 1.3;

export interface ClamOptions {
  tier: QualityTier;
  palette: Palette;
  skin: ReefSkin;
  /** yaw so the clam faces back up the reef, toward the approaching dive */
  facing: number;
  name?: string;
}

export interface ClamHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** the probe object: world position === the goal cell's shelf, always */
  root: THREE.Object3D;
  /** local height of the pearl — the scene lifts its point light to it */
  readonly pearlHeight: number;
  /** the whole visual state; both arguments are pure functions of the frame */
  setState(open01: number, flare01: number): void;
  /** ctx.time() — the swell and the material clocks */
  setTime(t: number): void;
  dispose(): void;
}

export function buildClam(opts: ClamOptions): ClamHandle {
  const { tier, palette, skin } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'giantClam';

  // ---- the probe root: position only, forever ----
  const root = new THREE.Object3D();
  root.name = 'clamRoot';
  group.add(root);

  const body = new THREE.Group();
  body.name = 'clamBody';
  body.rotation.y = opts.facing;
  root.add(body);

  const shellMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.72,
    metalness: 0.04,
    side: THREE.DoubleSide,
    rimColor: palette.hemiSky,
    rimStrength: 0.5,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.26,
    tier,
    name: 'clamShellMaterial',
  });

  // ---- the pedestal the clam is cemented to ----
  const baseParts: ColoredPart[] = [];
  {
    const radial = Math.max(6, Math.round(12 * detail));
    const rock = new THREE.CylinderGeometry(0.36, 0.46, 0.15, radial, 1);
    rock.translate(0, 0.075, 0);
    baseParts.push({ geometry: rock, color: blendTokens(skin.shelfDeep, skin.coral, 0.22) });
    for (let i = 0; i < (tier === 'thumb' ? 2 : 4); i++) {
      const a = (i / 4) * Math.PI * 2 + 0.6;
      const knob = new THREE.IcosahedronGeometry(0.06 + 0.025 * ((i % 3) / 2), 0);
      knob.scale(1, 0.7, 1);
      knob.translate(Math.cos(a) * 0.44, 0.05, Math.sin(a) * 0.44);
      baseParts.push({ geometry: knob, color: blendTokens(skin.coral, skin.anemone, 0.3) });
    }
  }
  const baseGeometry = mergeColoredParts(baseParts, 'clamBase');
  const baseMesh = new THREE.Mesh(baseGeometry, shellMaterial);
  baseMesh.name = 'clamBase';
  baseMesh.castShadow = tier !== 'thumb';
  baseMesh.receiveShadow = tier !== 'thumb';
  body.add(baseMesh);

  // ---- the mantle: the fleshy lip that shows once the shells part ----
  const mantleGeometry = makeMantleGeometry(detail, skin);
  const mantleMesh = new THREE.Mesh(mantleGeometry, shellMaterial);
  mantleMesh.name = 'clamMantle';
  mantleMesh.position.y = HINGE_Y;
  body.add(mantleMesh);

  // ---- the two shells, hinged at the back ----
  const shellGeometry = makeShellGeometry(detail, skin);
  const upperHinge = new THREE.Group();
  upperHinge.name = 'clamHingeUpper';
  upperHinge.position.set(0, HINGE_Y, HINGE_Z);
  const upperShell = new THREE.Mesh(shellGeometry, shellMaterial);
  upperShell.name = 'clamShellUpper';
  upperShell.position.z = -HINGE_Z;
  upperShell.castShadow = tier !== 'thumb';
  upperShell.receiveShadow = tier !== 'thumb';
  upperHinge.add(upperShell);
  body.add(upperHinge);

  const lowerHinge = new THREE.Group();
  lowerHinge.name = 'clamHingeLower';
  lowerHinge.position.set(0, HINGE_Y, HINGE_Z);
  const lowerShell = new THREE.Mesh(shellGeometry, shellMaterial);
  lowerShell.name = 'clamShellLower';
  lowerShell.position.z = -HINGE_Z;
  lowerShell.rotation.z = Math.PI; // the same shell, flipped into a bowl
  lowerShell.castShadow = tier !== 'thumb';
  lowerShell.receiveShadow = tier !== 'thumb';
  lowerHinge.add(lowerShell);
  body.add(lowerHinge);

  // ---- the pearl: the one thing in this reef that crosses the bloom threshold ----
  const pearlGeometry = new THREE.IcosahedronGeometry(0.115, tier === 'thumb' ? 1 : 2);
  const pearlMaterial: EmissiveMaterial = emissiveMaterial(skin.pearl, PEARL_HDR, { name: 'clamPearl' });
  const pearl = new THREE.Mesh(pearlGeometry, pearlMaterial);
  pearl.name = 'clamPearl';
  pearl.position.y = PEARL_Y;
  body.add(pearl);

  const haloGeometry = new THREE.IcosahedronGeometry(0.24, tier === 'thumb' ? 0 : 1);
  const haloMaterial: EmissiveMaterial = emissiveMaterial(skin.pearl, PEARL_HDR * 0.3, {
    transparent: true,
    opacity: 0.3,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'clamPearlHalo',
  });
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.name = 'clamPearlHalo';
  halo.position.y = PEARL_Y;
  body.add(halo);

  let openness = 0;

  return {
    group,
    root,
    pearlHeight: PEARL_Y,

    setState(open01, flare01) {
      const open = clamp01(open01);
      const flare = clamp01(flare01);
      openness = open;
      upperHinge.rotation.x = GAPE_UPPER * open;
      lowerHinge.rotation.x = -GAPE_LOWER * open;
      // the pearl swells and brightens with the gape; the flare rides on top of it
      const gain = (0.45 + 0.75 * open) * (1 + 3.4 * flare);
      pearl.scale.setScalar(0.8 + 0.34 * open + 0.5 * flare);
      halo.scale.setScalar(0.75 + 0.4 * open + 1.5 * flare);
      setEmissiveIntensity(pearlMaterial, PEARL_HDR * gain);
      setEmissiveIntensity(haloMaterial, PEARL_HDR * 0.3 * gain);
    },

    setTime(t) {
      // the swell rocks the whole clam; the amplitude grows as it gapes
      body.rotation.z = Math.sin(t * 0.62) * 0.035 * (0.5 + 0.5 * openness);
      body.rotation.x = Math.cos(t * 0.47) * 0.028 * (0.5 + 0.5 * openness);
      oakUniforms(shellMaterial)?.setTime(t);
    },

    dispose() {
      shellGeometry.dispose();
      baseGeometry.dispose();
      mantleGeometry.dispose();
      pearlGeometry.dispose();
      haloGeometry.dispose();
      shellMaterial.dispose();
      pearlMaterial.dispose();
      haloMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/**
 * One valve: a squashed hemisphere with radial ribs and a vertex grade that runs dark at
 * the hinge and pale at the rim. Both shells are the SAME geometry — the lower one is the
 * upper one rolled 180° — so a clam costs one buffer.
 */
export function makeShellGeometry(detail: number, skin: ReefSkin): THREE.BufferGeometry {
  const radial = Math.max(12, Math.round(32 * detail));
  const rings = Math.max(3, Math.round(7 * detail));
  // EVEN, and that is structural: the lower valve is the upper one rolled 180° about z,
  // which maps theta -> pi - theta, so cos(ribs·theta) only survives the flip for an even
  // rib count. Odd, and the two fluted rims interlock crest-into-trough and the shells
  // saw through each other every time the clam breathes.
  const ribs = 8;
  const g = new THREE.SphereGeometry(SHELL_R, radial, rings, 0, Math.PI * 2, 0, Math.PI / 2);
  const pos = g.getAttribute('position');
  const shell = new THREE.Color(skin.shell);
  const ribTone = new THREE.Color(skin.shellRib);
  const colors = new Float32Array(pos.count * 3);
  const tone = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const theta = Math.atan2(z, x);
    const wave = Math.cos(ribs * theta);
    // The flute has to DIE at the apex. A rib carried all the way to the pole makes eight
    // creases meeting in a point, and the valve reads as an origami flower instead of a
    // shell; weighted by (1 − t²) it is a deep corrugation at the rim that smooths out
    // over the dome, which is what a Tridacna valve actually does.
    const t = clamp01(y / SHELL_R);
    const flute = wave * (1 - t * t);
    const swell = 1 + 0.09 * flute;
    const rim = 1 - t;
    pos.setX(i, x * swell);
    pos.setZ(i, z * swell * 1.14);
    // flat and wide: a giant clam is a low mound with a long zigzag mouth, and the mouth
    // is the only part of it a 3/4 aerial can read
    pos.setY(i, y * (0.4 + 0.1 * flute) + 0.085 * wave * rim * rim);
    const flank = clamp01(Math.sqrt(x * x + z * z) / SHELL_R);
    tone.copy(shell).lerp(ribTone, 0.1 + 0.5 * (0.5 + 0.5 * flute)).multiplyScalar(0.58 + 0.4 * flank);
    colors[i * 3] = tone.r;
    colors[i * 3 + 1] = tone.g;
    colors[i * 3 + 2] = tone.b;
  }
  pos.needsUpdate = true;
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'clamShell';
  return g;
}

/** The mantle: a low ruffled disc that only shows once the valves part. */
export function makeMantleGeometry(detail: number, skin: ReefSkin): THREE.BufferGeometry {
  const radial = Math.max(10, Math.round(22 * detail));
  const g = new THREE.CircleGeometry(SHELL_R * 1.02, radial);
  g.rotateX(-Math.PI / 2);
  const pos = g.getAttribute('position');
  const colors = new Float32Array(pos.count * 3);
  const inner = new THREE.Color(skin.mantle);
  const edge = new THREE.Color(skin.anemone);
  const tone = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const r = Math.sqrt(x * x + z * z);
    const theta = Math.atan2(z, x);
    // the ruffle: the mantle's lip waves, which is what makes it read as alive
    const k = clamp01(r / (SHELL_R * 1.02));
    pos.setY(i, Math.sin(theta * 8) * 0.05 * k * k);
    tone.copy(inner).lerp(edge, k * k);
    colors[i * 3] = tone.r;
    colors[i * 3 + 1] = tone.g;
    colors[i * 3 + 2] = tone.b;
  }
  pos.needsUpdate = true;
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'clamMantle';
  return g;
}

// ---------------------------------------------------------------------------
// the diver's bubble
// ---------------------------------------------------------------------------

export interface BubbleOptions {
  tier: QualityTier;
  palette: Palette;
  skin: ReefSkin;
  accent: THREE.ColorRepresentation;
  /** the wisp's orb-centre height — the bubble is built around it */
  height: number;
  name?: string;
}

export interface BubbleHandle {
  /** the scene parents this under `wisp.avatar`, so it is exactly where the diver is */
  group: THREE.Group;
  /** ctx.time() — the wobble and the surface sheen */
  setTime(t: number): void;
  /** 0 = calm, 1 = the diver just took a hit of current (a pure read of the frame) */
  setStress(v: number): void;
  dispose(): void;
}

/**
 * The bubble the wisp rides. A thin double-sided shell with a strong sky-tinted fresnel
 * and a faint additive sheen — the two things that make a sphere read as a soap film
 * rather than a marble. It hangs off the avatar node, so it inherits the dive's exact
 * position and adds nothing to it.
 */
export function buildBubble(opts: BubbleOptions): BubbleHandle {
  const { tier, palette, skin } = opts;
  const detail = tier === 'thumb' ? 1 : 2;
  const group = new THREE.Group();
  group.name = opts.name ?? 'diverBubble';
  group.position.y = opts.height;

  const shellGeometry = new THREE.IcosahedronGeometry(0.33, detail);
  const shellMaterial: OakMaterial = oakMaterial({
    color: blendTokens(skin.bubble, palette.skyTop, 0.35),
    roughness: 0.08,
    metalness: 0,
    transparent: true,
    opacity: 0.15,
    depthWrite: false,
    side: THREE.DoubleSide,
    rimColor: opts.accent,
    rimStrength: 1.15,
    rimPower: 2.2,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.4,
    tier,
    name: 'diverBubbleShell',
  });
  const shell = new THREE.Mesh(shellGeometry, shellMaterial);
  shell.name = 'diverBubbleShell';
  group.add(shell);

  // a bright meniscus at the crown — where the light actually catches a bubble
  const sheenGeometry = new THREE.TorusGeometry(0.2, 0.018, 4, Math.max(10, 18 * detail));
  sheenGeometry.rotateX(-Math.PI / 2.6);
  sheenGeometry.translate(0, 0.15, 0.03);
  const sheenMaterial: EmissiveMaterial = emissiveMaterial(skin.bubble, 0.9, {
    transparent: true,
    opacity: 0.42,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'diverBubbleSheen',
  });
  const sheen = new THREE.Mesh(sheenGeometry, sheenMaterial);
  sheen.name = 'diverBubbleSheen';
  group.add(sheen);

  let stress = 0;

  return {
    group,
    setTime(t) {
      // a soap film never holds a perfect sphere: two out-of-phase breaths do the job
      const wobble = 0.05 + 0.05 * stress;
      shell.scale.set(
        1 + Math.sin(t * 2.3) * wobble,
        1 + Math.sin(t * 1.7 + 1.1) * wobble,
        1 + Math.cos(t * 2.05 + 0.4) * wobble,
      );
      sheen.rotation.y = t * 0.55;
      shellMaterial.opacity = 0.13 + 0.05 * (0.5 + 0.5 * Math.sin(t * 1.3));
      oakUniforms(shellMaterial)?.setTime(t);
    },
    setStress(v) {
      stress = clamp01(v);
      setEmissiveIntensity(sheenMaterial, 0.9 * (1 + 0.9 * stress));
    },
    dispose() {
      shellGeometry.dispose();
      shellMaterial.dispose();
      sheenGeometry.dispose();
      sheenMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
