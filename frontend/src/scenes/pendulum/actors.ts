// The Lantern Bell — the rod, the lantern and the torque curl.
//
// ACCURACY: `bobRoot` is the probe object. Its world position is EXACTLY
// pivot + (L·sin θ, −L·cos θ, 0) — written straight from `Math.sin`/`Math.cos` of the
// angle, with no bob, no lean and no lift on it. The lantern body, the tassel and the
// flame are children; delete every one of them and the rig is still correct.
//
// `swingRoot` sits ON the pivot and carries rotation only (`rotation.z = θ`), which is
// exactly the rotation that maps the local down-vector (0, −L) onto (L·sin θ, −L·cos θ).
// The rod is drawn from it, so rod and lantern can never disagree about the angle.
//
// THE TORQUE CURL. `curlRoot` hangs off `swingRoot` at (0, −L, 0), so its local +x is the
// +θ TANGENT of the swing circle and its local −y points radially outward, whatever the
// angle. A curl drawn once in that frame is therefore sign-correct at every θ for free:
// positive torque shows the +x family, negative torque shows the same geometry mirrored.
// Nothing here is a history buffer — every value below is a pure function of the frame.

import {
  emissiveMaterial,
  oakMaterial,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import {
  CAP_R,
  EYE_DROP,
  EYE_MAJOR,
  EYE_TUBE,
  FINIAL_H,
  LANTERN_H,
  LANTERN_R,
  RIB_HALF,
  RIB_INSET,
  RING_R,
  RING_TUBE,
  ROD_LEN,
  TASSEL_CORD_LEN,
  TASSEL_DROP,
  TASSEL_WEIGHT_H,
  TASSEL_WEIGHT_R,
} from './dims';
import { blendTokens, clampRange, mergeColoredParts, type ColoredPart } from './world';

// ---------------------------------------------------------------------------
// dimensions (world units — the lantern is 0.3 tall on a 1.0 rod)
//
// Every number that decides how far the assembly reaches lives in `dims.ts`, because
// `world.ts` places the torii's tie beam from that reach. Change a lantern dimension and
// the arch moves with it; the test measures these meshes and re-derives the envelope.
// ---------------------------------------------------------------------------

const ROD_TOP = -0.05;
const ROD_BOTTOM = -(ROD_LEN - 0.15);
const PAPER_BASE_HDR = 2.7;

export interface LanternRigHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** rotation-only node on the pivot: `rotation.z = θ`. A probe object. */
  swingRoot: THREE.Object3D;
  /** the bob: world position === pivot + (L sin θ, −L cos θ, 0), always. A probe object. */
  bobRoot: THREE.Object3D;
  /** world position of the lantern's flame core, refreshed by setState (preallocated) */
  readonly flameWorld: THREE.Vector3;
  /** the whole visual state, from the frame and nothing else */
  setState(theta: number, thetaDot: number, torque: number): void;
  /** 0..1 — how brightly the paper burns (the shaped reward, see index.ts) */
  setGlow(serenity: number): void;
  /** ctx.time() — the lantern's idle breath and the curl's drift */
  setTime(t: number): void;
  dispose(): void;
}

export interface LanternRigOptions {
  tier: QualityTier;
  palette: Palette;
  accent: THREE.ColorRepresentation;
  /** the decoder's own pivot anchor */
  pivot: readonly [number, number, number];
  /** |torque| that counts as full effort (PD_MAX_TORQUE) */
  maxTorque: number;
  /** |thdot| clip (PD_MAX_SPEED) */
  maxSpeed: number;
}

export function buildLanternRig(opts: LanternRigOptions): LanternRigHandle {
  const { tier, palette, accent, pivot, maxTorque, maxSpeed } = opts;
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'lanternRig';

  const iron = blendTokens(palette.hemiGround, palette.skyTop, 0.35);
  const ironLight = blendTokens(iron, palette.hemiSky, 0.35);
  const cord = blendTokens(palette.groundA, palette.hemiGround, 0.5);

  // ---- the rotation node, exactly on the pivot ----
  const swingRoot = new THREE.Object3D();
  swingRoot.name = 'swingRoot';
  swingRoot.position.set(pivot[0], pivot[1], pivot[2]);
  group.add(swingRoot);

  // ---- the rod: iron, drawn from the pivot down to the lantern's yoke ----
  const rodParts: ColoredPart[] = [];
  {
    // the eye: a forged eyelet whose HOLE the torii's pin passes through (dims.ts sizes
    // both, so the two files can never disagree about which one is inside the other)
    const eye = new THREE.TorusGeometry(EYE_MAJOR, EYE_TUBE, 5, 10);
    eye.translate(0, -EYE_DROP, 0);
    rodParts.push({ geometry: eye, color: ironLight });
    const len = ROD_TOP - ROD_BOTTOM;
    const shaft = new THREE.CylinderGeometry(0.024, 0.031, len, 6);
    shaft.translate(0, ROD_BOTTOM + len * 0.5, 0);
    rodParts.push({ geometry: shaft, color: iron });
    // the yoke that carries the lantern's top ring
    const yoke = new THREE.TorusGeometry(0.036, 0.011, 5, 10);
    yoke.translate(0, ROD_BOTTOM - 0.012, 0);
    rodParts.push({ geometry: yoke, color: ironLight });
    const hook = new THREE.CylinderGeometry(0.009, 0.009, 0.055, 5);
    hook.translate(0, ROD_BOTTOM - 0.042, 0);
    rodParts.push({ geometry: hook, color: iron });
  }
  const rodGeometry = mergeColoredParts(rodParts, 'lanternRod');
  const ironMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.68,
    metalness: 0.12,
    // moon-silver rim: the iron reads as a hard edge against the stars even unlit
    rimColor: blendTokens(palette.sun, palette.hemiSky, 0.35),
    rimStrength: 0.6,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.12,
    tier,
    name: 'lanternIronMaterial',
  });
  const rodMesh = new THREE.Mesh(rodGeometry, ironMaterial);
  rodMesh.name = 'lanternRod';
  rodMesh.castShadow = shadows;
  swingRoot.add(rodMesh);

  // ---- the bob: POSITION ONLY, forever ----
  const bobRoot = new THREE.Object3D();
  bobRoot.name = 'bobRoot';
  group.add(bobRoot);

  // decoration node: everything that is not the bob's position hangs off this
  const bobTilt = new THREE.Object3D();
  bobTilt.name = 'bobTilt';
  bobRoot.add(bobTilt);

  // ---- the lantern's iron frame (ribs, caps, finial, tassel) ----
  const frameParts: ColoredPart[] = [];
  {
    const topCap = new THREE.CylinderGeometry(CAP_R * 0.62, CAP_R, 0.05, 6);
    topCap.translate(0, LANTERN_H * 0.5 + 0.024, 0);
    frameParts.push({ geometry: topCap, color: ironLight });
    const bottomCap = new THREE.CylinderGeometry(CAP_R * 0.94, CAP_R * 0.66, 0.042, 6);
    bottomCap.translate(0, -LANTERN_H * 0.5 - 0.02, 0);
    frameParts.push({ geometry: bottomCap, color: ironLight });
    const finial = new THREE.ConeGeometry(0.026, FINIAL_H, 6);
    finial.translate(0, LANTERN_H * 0.5 + 0.076, 0);
    frameParts.push({ geometry: finial, color: iron });
    const ring = new THREE.TorusGeometry(RING_R, RING_TUBE, 4, 9);
    ring.translate(0, LANTERN_H * 0.5 + 0.113, 0);
    frameParts.push({ geometry: ring, color: iron });
    // six vertical ribs on the paper's corners
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      const rib = new THREE.BoxGeometry(RIB_HALF * 2, LANTERN_H + 0.02, RIB_HALF * 2);
      rib.translate(Math.cos(a) * LANTERN_R * RIB_INSET, 0, Math.sin(a) * LANTERN_R * RIB_INSET);
      frameParts.push({ geometry: rib, color: iron });
    }
    // The tassel: a short cord and a weight, the silhouette that says "hanging" — and
    // the FARTHEST thing from the bob, so `LANTERN_REACH` is measured to this cone's apex
    // and the tie beam is placed above it (dims.ts).
    const tasselCord = new THREE.CylinderGeometry(0.005, 0.005, TASSEL_CORD_LEN, 4);
    tasselCord.translate(0, -LANTERN_H * 0.5 - (TASSEL_DROP - TASSEL_WEIGHT_H * 0.92), 0);
    frameParts.push({ geometry: tasselCord, color: cord });
    const weight = new THREE.ConeGeometry(TASSEL_WEIGHT_R, TASSEL_WEIGHT_H, 6);
    weight.rotateZ(Math.PI);
    weight.translate(0, -LANTERN_H * 0.5 - TASSEL_DROP, 0);
    frameParts.push({ geometry: weight, color: cord });
  }
  const frameGeometry = mergeColoredParts(frameParts, 'lanternFrame');
  const frameMesh = new THREE.Mesh(frameGeometry, ironMaterial);
  frameMesh.name = 'lanternFrame';
  frameMesh.castShadow = shadows;
  bobTilt.add(frameMesh);

  // ---- the paper: the scene's one bloom feeder ----
  // A six-sided barrel bulging at the waist. `emissiveMaterial` is unlit and unfogged, so
  // its brightness is ENTIRELY the shaped reward — there is no ambient floor hiding in it.
  const paperGeometry = makePaperGeometry();
  const paperMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(accent, palette.sun, 0.22),
    PAPER_BASE_HDR,
    { transparent: true, opacity: 0.95, side: THREE.DoubleSide, name: 'lanternPaper' },
  );
  const paperMesh = new THREE.Mesh(paperGeometry, paperMaterial);
  paperMesh.name = 'lanternPaper';
  bobTilt.add(paperMesh);

  // the flame core inside the paper — small, very hot, the thing the bloom latches onto
  const flameGeometry = new THREE.IcosahedronGeometry(0.05, 1);
  flameGeometry.scale(1, 1.45, 1);
  const flameMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(palette.sun, accent, 0.62),
    PAPER_BASE_HDR * 1.5,
    { name: 'lanternFlameCore' },
  );
  const flameMesh = new THREE.Mesh(flameGeometry, flameMaterial);
  flameMesh.name = 'lanternFlame';
  flameMesh.position.y = -0.01;
  bobTilt.add(flameMesh);

  // ---- the torque curl: wind around the bob, in the swing-tangent frame ----
  const curlRoot = new THREE.Object3D();
  curlRoot.name = 'torqueCurlRoot';
  curlRoot.position.set(0, -ROD_LEN, 0);
  swingRoot.add(curlRoot);

  const curlGeometry = makeCurlGeometry();
  const curlMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(palette.hemiSky, accent, 0.55),
    2.3,
    {
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      name: 'torqueCurlMaterial',
    },
  );
  curlMaterial.vertexColors = true;
  curlMaterial.opacity = 0;
  // THUMB drops the curl with the particles and the mist: with no post chain there is
  // nothing for an HDR additive flourish to bloom into, and it reads as a smear.
  const curlTier = tier !== 'thumb';
  const curlPos = new THREE.Mesh(curlGeometry, curlMaterial);
  curlPos.name = 'torqueCurlPositive';
  curlPos.renderOrder = 4;
  curlPos.visible = false;
  const curlNeg = new THREE.Mesh(curlGeometry, curlMaterial);
  curlNeg.name = 'torqueCurlNegative';
  curlNeg.renderOrder = 4;
  curlNeg.scale.x = -1;
  curlNeg.visible = false;
  curlRoot.add(curlPos, curlNeg);

  const flameWorld = new THREE.Vector3();
  let glow = 1;
  let breath = 0;
  let spin = 0;

  const applyFlame = (): void => {
    // an idle flicker that never reaches zero, scaled by the light level so a dark
    // lantern is dark and still, not a dying candle
    const flicker = 1 + 0.07 * Math.sin(breath * 3.1) + 0.04 * Math.sin(breath * 7.7 + 1.1);
    const level = glow * flicker;
    setEmissiveIntensity(paperMaterial, PAPER_BASE_HDR * (0.035 + 0.965 * level));
    setEmissiveIntensity(flameMaterial, PAPER_BASE_HDR * 1.5 * (0.02 + 0.98 * level));
    paperMaterial.opacity = 0.55 + 0.42 * clampRange(level, 0, 1);
  };

  return {
    group,
    swingRoot,
    bobRoot,
    flameWorld,

    setState(theta, thetaDot, torque) {
      // THE ONE ACCURACY LINE: the bob is the env's own kinematics, nothing else.
      const sin = Math.sin(theta);
      const cos = Math.cos(theta);
      bobRoot.position.set(pivot[0] + ROD_LEN * sin, pivot[1] - ROD_LEN * cos, pivot[2]);
      swingRoot.rotation.z = theta;

      // The lantern is carried by the rod (rigid: at θ = ±π it really is upside down),
      // with the small give a hook has — a pure function of θ̇, so a scrub reproduces it.
      const lag = clampRange(thetaDot / maxSpeed, -1, 1);
      bobTilt.rotation.z = theta + lag * 0.22;

      // the curl: sign picks the family, |torque| drives scale and opacity
      const effort = clampRange(Math.abs(torque) / maxTorque, 0, 1);
      const live = curlTier && effort > 0.02;
      curlPos.visible = live && torque > 0;
      curlNeg.visible = live && torque < 0;
      curlMaterial.opacity = live ? 0.08 + 0.76 * effort : 0;
      const scale = 0.72 + 0.5 * effort;
      curlPos.scale.set(scale, scale, scale);
      curlNeg.scale.set(-scale, scale, scale);

      flameWorld.set(bobRoot.position.x, bobRoot.position.y - 0.01, bobRoot.position.z);
    },

    setGlow(serenity) {
      glow = clampRange(serenity, 0, 1);
      applyFlame();
    },

    setTime(t) {
      breath = t;
      spin = t;
      // the curl drifts outward as it fades — the wind is moving, not painted on
      const drift = 1 + 0.05 * Math.sin(spin * 2.3);
      curlRoot.rotation.z = 0.09 * Math.sin(spin * 1.7);
      curlRoot.scale.setScalar(drift);
      applyFlame();
    },

    dispose() {
      rodGeometry.dispose();
      frameGeometry.dispose();
      ironMaterial.dispose();
      paperGeometry.dispose();
      paperMaterial.dispose();
      flameGeometry.dispose();
      flameMaterial.dispose();
      curlGeometry.dispose();
      curlMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/** A six-sided paper barrel that bulges at the waist — three stacked frustums. */
function makePaperGeometry(): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const rings: readonly (readonly [number, number, number])[] = [
    // [yBottom, yTop, radiusBottom -> radiusTop is taken from the next entry]
    [-LANTERN_H * 0.5, -LANTERN_H * 0.18, CAP_R * 0.92],
    [-LANTERN_H * 0.18, LANTERN_H * 0.18, LANTERN_R],
    [LANTERN_H * 0.18, LANTERN_H * 0.5, LANTERN_R],
  ];
  const radii = [CAP_R * 0.92, LANTERN_R, LANTERN_R, CAP_R * 0.9];
  for (let i = 0; i < rings.length; i++) {
    const ring = rings[i] as readonly [number, number, number];
    const y0 = ring[0];
    const y1 = ring[1];
    const r0 = radii[i] as number;
    const r1 = radii[i + 1] as number;
    const g = new THREE.CylinderGeometry(r1, r0, y1 - y0, 6, 1, true);
    g.translate(0, (y0 + y1) * 0.5, 0);
    parts.push({ geometry: g, color: 0xffffff });
  }
  return mergeColoredParts(parts, 'lanternPaper');
}

/**
 * Three wind strokes swirling past the bob, drawn once in the tangent frame: local +x is
 * "the direction this torque pushes", local ±y is radially in/out.
 *
 * Each stroke is a SPIRAL — the radius grows along the sweep — so it comes in tight
 * behind the lantern, wraps it and is flung off wide toward +x, tapering to a needle. The
 * first pass used constant-radius arcs spanning 180° each and the family closed into a
 * halo: a ring reads as an aura, and an aura has no direction. A spiral that leaves the
 * frame on one side does.
 */
function makeCurlGeometry(): THREE.BufferGeometry {
  interface Stroke {
    /** radius at the tail and at the tip — the difference is what makes it a swirl */
    radius: number;
    radiusEnd: number;
    from: number;
    to: number;
    width: number;
    lift: number;
  }
  const strokes: readonly Stroke[] = [
    // ~120° each and all of them confined to the +x half. Three strokes of 180° close
    // into a halo however you phase them, and a halo has no direction.
    { radius: 0.21, radiusEnd: 0.42, from: 1.78, to: -0.38, width: 0.062, lift: 0.16 },
    { radius: 0.3, radiusEnd: 0.58, from: 1.42, to: -0.72, width: 0.078, lift: 0.19 },
    { radius: 0.25, radiusEnd: 0.47, from: 2.16, to: 0.18, width: 0.048, lift: 0.13 },
  ];
  const SEG = 26;
  const verts = strokes.length * (SEG + 1) * 2;
  const positions = new Float32Array(verts * 3);
  const colors = new Float32Array(verts * 4);
  const indices: number[] = [];
  let v = 0;
  for (const stroke of strokes) {
    const base = v;
    for (let i = 0; i <= SEG; i++) {
      const k = i / SEG;
      const a = stroke.from + (stroke.to - stroke.from) * k;
      const radius = stroke.radius + (stroke.radiusEnd - stroke.radius) * k * k;
      const cx = Math.cos(a) * radius;
      const cy = Math.sin(a) * radius;
      // taper: nothing at the tail, full at 40%, a needle at the tip
      const taper = Math.sin(Math.PI * Math.pow(k, 0.72)) * (1 - 0.55 * k);
      const half = stroke.width * taper;
      const nx = Math.cos(a);
      const ny = Math.sin(a);
      positions[v * 3] = cx - nx * half;
      positions[v * 3 + 1] = cy - ny * half;
      positions[v * 3 + 2] = stroke.lift;
      positions[(v + 1) * 3] = cx + nx * half;
      positions[(v + 1) * 3 + 1] = cy + ny * half;
      positions[(v + 1) * 3 + 2] = stroke.lift;
      const alpha = Math.pow(taper, 0.85) * (0.35 + 0.65 * k);
      const lum = 0.6 + 0.4 * k;
      for (let e = 0; e < 2; e++) {
        const o = (v + e) * 4;
        colors[o] = lum;
        colors[o + 1] = lum;
        colors[o + 2] = lum;
        colors[o + 3] = alpha;
      }
      v += 2;
    }
    for (let i = 0; i < SEG; i++) {
      const a = base + i * 2;
      indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 4));
  g.setIndex(indices);
  g.computeBoundingSphere();
  g.name = 'torqueCurl';
  return g;
}
