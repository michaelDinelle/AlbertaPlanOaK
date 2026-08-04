// Gold Hour Savanna — the animal.
//
// ACCURACY: `joints[i]` is an Object3D sitting at EXACTLY the i-th point of
// `geometry.cheetah.fk(pose)`, mapped (fkX, fkZ) -> world (x, y) with z = 0. Those nodes
// carry POSITION ONLY — no bob, no lean, no scale — and three of them (o1, toe_b, toe_f)
// are the scene's alignment probes. Every bone mesh is placed at the MIDPOINT of two
// joints and rotated to point at the second, so the coat is a rigid skin on the exact
// skeleton: delete every decoration and what remains is the FK stick figure the debug
// overlay draws.
//
// NOTHING THE SIM DEFINES IS TYPED IN HERE. The link lengths are MEASURED at build time
// from `fk(0^9)`; the capsule radius the coat must contain and the head sphere are read
// off `opts.cheetah` (`geometry.cheetah.radius` / `.headRadius`) and every facial feature
// is placed as a multiple of that head radius. Change the model and the whole animal
// follows — a coat that quietly stopped containing the physics capsule would still pass
// every joint probe, so the radii have to come from the same place the joints do.
//
// IDEMPOTENCE: everything in `setPose` is a pure function of the FK point array. The
// tail's swing phase is read off the back-foot bone angle (which oscillates with the
// gait), not off a clock or a history buffer, so a scrub reproduces the exact same
// animal — mid-stride, tail and all.
//
// CONTACT: the tail is the only part of this animal with no counterpart in the sim, so it
// is the only part that could lie about contact. It is SOLVED, not posed — see the tail
// section of `setPose`: each vertebra takes its art droop and then bends up by exactly
// the angle that keeps its far end on its own radius above y = 0. The scene's whole claim
// is that the ground is exactly y = 0 and the FK toes rest on it; a decoration hanging
// below the planted toe would destroy the one contact cue the frame has.

import {
  emissiveMaterial,
  oakMaterial,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { hash2f } from '@kit/rng';
import { radialSprite } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import { CHEETAH_BONES, FK_POINT_INDEX } from '@sim/fk/cheetah';
import type { CheetahGeometry } from '@sim/types';
import { blendTokens, clampRange, mergeColoredParts, smoothstep01, TOKEN_BLACK, TOKEN_WHITE, type ColoredPart } from './world';

// ---------------------------------------------------------------------------
// coat dimensions (world units — the torso really is 1.0 long, hip height ~0.7)
// ---------------------------------------------------------------------------

/** spots per world unit of coat, in both directions */
const SPOT_DENSITY = 7.5;
/** toe height (world y) below which a foot counts as planted */
const CONTACT_HIGH = 0.1;
const CONTACT_LOW = 0.036;
/** the runner's reference speed for dust / wind scaling, m/s */
export const REFERENCE_SPEED = 1.1;

/** per-bone coat: radius profile, ventral cream, distal sock, spot strength, section shape */
interface CoatSpec {
  r0: number;
  rMid: number;
  r1: number;
  belly: number;
  sock: number;
  spots: number;
  /** cross-section scale on the dorsal-ventral (local x) and lateral (local z) axes */
  deep: number;
  wide: number;
  rings: number;
  radial: number;
}

const TORSO: CoatSpec = { r0: 0.128, rMid: 0.088, r1: 0.138, belly: 0.8, sock: 0, spots: 1.15, deep: 1.2, wide: 0.82, rings: 44, radial: 22 };
const THIGH: CoatSpec = { r0: 0.09, rMid: 0.072, r1: 0.05, belly: 0.2, sock: 0, spots: 0.9, deep: 1.1, wide: 0.86, rings: 18, radial: 14 };
const SHIN: CoatSpec = { r0: 0.05, rMid: 0.042, r1: 0.036, belly: 0.15, sock: 0.45, spots: 0.4, deep: 1, wide: 0.95, rings: 14, radial: 12 };
const FOOT: CoatSpec = { r0: 0.036, rMid: 0.032, r1: 0.028, belly: 0, sock: 1, spots: 0, deep: 1, wide: 1, rings: 10, radial: 10 };

/** bone index -> coat spec, in CHEETAH_BONES order */
const COATS: readonly CoatSpec[] = [TORSO, THIGH, SHIN, FOOT, THIGH, SHIN, FOOT];
const BONE_NAMES: readonly string[] = ['torso', 'thighBack', 'shinBack', 'footBack', 'thighFront', 'shinFront', 'footFront'];

// ---------------------------------------------------------------------------
// the tail — the one part that is pure design, and therefore the one part held to the
// contact plane by a solver instead of by the sim (see `setPose`)
// ---------------------------------------------------------------------------

/** segment lengths, hip -> tip */
const TAIL_LEN: readonly number[] = [0.26, 0.23, 0.2, 0.17];
/** radius at each tail JOINT, tip cap included: TAIL_R[i+1] is segment i's far-end radius */
const TAIL_R: readonly number[] = [0.062, 0.055, 0.047, 0.04, 0.022];
/**
 * The rest curve, per vertebra. A cheetah carries its tail as a COUNTERWEIGHT streamed out
 * behind it — it never trails on the ground — so the whole chain (0.86 long, against a hip
 * that stands ~0.6 up) may only curve gently: the four droops plus TAIL_HANG drop the tip
 * ~0.45 below the hip at rest (a real hang), and a full-speed run turns that into ~0.12
 * ABOVE it. Measured over the shipped lane, the tail's lowest surface point never comes
 * within 4 cm of the ground; the solver in `setPose` is what makes that a guarantee.
 */
const TAIL_DROOP: readonly number[] = [0.17, 0.2, 0.22, 0.2];
const TAIL_SWING: readonly number[] = [0.14, 0.18, 0.22, 0.26];
/** fraction of the rest curve a full-speed run straightens out */
const TAIL_STREAM = 0.72;
/** the hang of the whole tail at the hip, and the lift a full-speed run adds to it */
const TAIL_HANG = 0.13;
const TAIL_RUN_LIFT = 0.4;

// ---------------------------------------------------------------------------
// coat geometry: a tapered tube of revolution with hemispherical ends
// ---------------------------------------------------------------------------

export type CoatColorFn = (t: number, theta: number, out: THREE.Color) => void;

/**
 * A limb: a body of revolution about local +y spanning y ∈ [-len/2, len/2] plus rounded
 * caps, so its two ENDS land exactly on the two joints it spans. `t` runs 0..1 from the
 * first joint to the second (caps included), `theta` runs 0..1 around the limb.
 *
 * `minRadius` is the honest collision radius the profile is floored at — pass the sim's
 * own `geometry.cheetah.radius` for a real bone (the coat must CONTAIN the capsule the
 * physics simulates) and 0 for the tail, which the physics does not have.
 */
export function limbGeometry(
  len: number,
  spec: CoatSpec,
  color: CoatColorFn,
  name: string,
  minRadius: number,
): THREE.BufferGeometry {
  const caps = 4;
  const rings = spec.rings;
  const radial = spec.radial;
  const profile = (u: number): number => {
    // quadratic through (0, r0), (0.5, rMid), (1, r1), floored at `minRadius`
    const a = spec.r0;
    const b = spec.rMid;
    const c = spec.r1;
    const m = 1 - u;
    return Math.max(minRadius, m * m * a + 2 * m * u * b + u * u * c);
  };

  // ---- spine: (y, radius) pairs from the bottom pole to the top pole ----
  const ys: number[] = [];
  const rs: number[] = [];
  const half = len * 0.5;
  for (let i = 0; i <= caps; i++) {
    const alpha = (i / caps) * (Math.PI / 2);
    ys.push(-half - spec.r0 * Math.cos(alpha));
    rs.push(spec.r0 * Math.sin(alpha));
  }
  for (let i = 1; i <= rings; i++) {
    const u = i / rings;
    ys.push(-half + u * len);
    rs.push(profile(u));
  }
  for (let i = 1; i <= caps; i++) {
    const alpha = (1 - i / caps) * (Math.PI / 2);
    ys.push(half + spec.r1 * Math.cos(alpha));
    rs.push(spec.r1 * Math.sin(alpha));
  }

  const nRing = ys.length;
  const nRad = radial + 1; // duplicated seam column keeps the coat colours continuous
  const count = nRing * nRad;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const c = new THREE.Color();
  const span = (ys[nRing - 1] ?? half) - (ys[0] ?? -half);
  for (let j = 0; j < nRing; j++) {
    const y = ys[j] ?? 0;
    const r = rs[j] ?? 0;
    const t = span > 1e-9 ? (y - (ys[0] ?? 0)) / span : 0;
    for (let i = 0; i < nRad; i++) {
      const theta = i / radial;
      const a = theta * Math.PI * 2;
      const idx = j * nRad + i;
      positions[idx * 3] = Math.cos(a) * r * spec.deep;
      positions[idx * 3 + 1] = y;
      positions[idx * 3 + 2] = Math.sin(a) * r * spec.wide;
      color(t, theta, c);
      colors[idx * 3] = c.r;
      colors[idx * 3 + 1] = c.g;
      colors[idx * 3 + 2] = c.b;
    }
  }

  const quads = (nRing - 1) * radial;
  const indices = count > 65535 ? new Uint32Array(quads * 6) : new Uint16Array(quads * 6);
  let o = 0;
  for (let j = 0; j < nRing - 1; j++) {
    for (let i = 0; i < radial; i++) {
      const a = j * nRad + i;
      const b = a + 1;
      const d = a + nRad;
      const e = d + 1;
      indices[o++] = a;
      indices[o++] = d;
      indices[o++] = b;
      indices[o++] = b;
      indices[o++] = d;
      indices[o++] = e;
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = name;
  return g;
}

/**
 * The coat pattern: a jittered lattice of spots that WRAPS around the limb (cell v is
 * taken modulo the ring count), so there is no seam and no image texture — STYLE.md
 * forbids texture maps for surface detail, and a cheetah without spots is a greyhound.
 */
export function spotMask(u: number, v: number, nu: number, nv: number, seed: number, radius: number): number {
  const cu = Math.floor(u * nu);
  const cv = Math.floor(v * nv);
  let best = 4;
  for (let du = -1; du <= 1; du++) {
    for (let dv = -1; dv <= 1; dv++) {
      const ci = cu + du;
      const cj = cv + dv;
      const wj = ((cj % nv) + nv) % nv;
      const jx = hash2f(ci, wj, seed);
      const jy = hash2f(ci, wj, seed ^ 0x9e3779b9);
      const px = (ci + 0.15 + jx * 0.7) / nu;
      const py = (cj + 0.15 + jy * 0.7) / nv;
      const dx = (u - px) * 1.05;
      const dy = (v - py) * 0.92;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < best) best = d;
    }
  }
  return 1 - smoothstep01(best / Math.max(1e-4, radius));
}

// ---------------------------------------------------------------------------
// the handle
// ---------------------------------------------------------------------------

export interface CheetahHandle {
  /** add this ONE group; it sits at the origin and never moves */
  group: THREE.Group;
  /** the FK joint nodes, in CHEETAH_FK_POINTS order — POSITION ONLY (the probes) */
  joints: THREE.Object3D[];
  /** the whole animal, from the FK points and nothing else */
  setPose(fk: Float32Array, speed01: number): void;
  /** ctx.time() — drives only the dust field; the animal itself is state-driven */
  setTime(t: number): void;
  /** 1 = normal, >1 = the teleport flash warming the coat */
  setGlow(gain: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

export interface CheetahOptions {
  tier: QualityTier;
  palette: Palette;
  accent: THREE.ColorRepresentation;
  seed: number;
  /** sim-core's own FK — the scene never re-derives the skeleton */
  cheetah: CheetahGeometry;
}

const ZERO_POSE = new Float32Array(9);

export function buildCheetah(opts: CheetahOptions): CheetahHandle {
  const { tier, palette, seed, cheetah } = opts;
  const shadows = tier !== 'thumb';
  const lod = tier === 'hero' ? 1 : tier === 'medium' ? 0.7 : 0.45;
  const group = new THREE.Group();
  group.name = 'cheetahRig';

  // ---- the coat palette (blends of tokens only) ----
  const cCoat = blendTokens(palette.groundB, palette.sun, 0.45);
  const cBack = blendTokens(cCoat, palette.groundA, 0.44);
  const cBelly = blendTokens(TOKEN_WHITE, palette.groundB, 0.42);
  const cSpot = blendTokens(TOKEN_BLACK, palette.hemiGround, 0.22);
  const cSock = blendTokens(TOKEN_BLACK, palette.groundA, 0.3);
  const cNose = blendTokens(TOKEN_BLACK, palette.hemiGround, 0.2);
  const cTip = blendTokens(TOKEN_WHITE, palette.sun, 0.25);

  // ---- the two sim dimensions, read from the geometry the host handed us (never typed
  //      in): the capsule radius the coat has to contain, and the head sphere ----
  const boneR = cheetah.radius;
  const headR = cheetah.headRadius;

  // ---- rest lengths MEASURED from sim-core's own FK ----
  const rest = cheetah.fk(ZERO_POSE);
  const boneLen: number[] = [];
  for (const bone of CHEETAH_BONES) {
    const a = bone[0];
    const b = bone[1];
    const dx = (rest[b * 2] ?? 0) - (rest[a * 2] ?? 0);
    const dz = (rest[b * 2 + 1] ?? 0) - (rest[a * 2 + 1] ?? 0);
    boneLen.push(Math.hypot(dx, dz));
  }

  // ---- joint nodes: the probes. Position only, forever. ----
  const joints: THREE.Object3D[] = [];
  for (let i = 0; i < 10; i++) {
    const node = new THREE.Object3D();
    node.name = `cheetahJoint${i}`;
    group.add(node);
    joints.push(node);
  }

  // ---- one material for the whole coat; the pattern rides on vertex colours ----
  const coatMaterial: OakMaterial = oakMaterial({
    color: TOKEN_WHITE,
    vertexColors: true,
    roughness: 0.82,
    rimColor: palette.sun,
    rimStrength: 1.4,
    rimPower: 2.4,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: tier === 'thumb' ? 0.46 : 0.22,
    tier,
    name: 'cheetahCoat',
  });

  const scratchColor = new THREE.Color();
  const spotColor = new THREE.Color(cSpot);

  const coatFor = (spec: CoatSpec, len: number, boneSeed: number): CoatColorFn => {
    const nu = Math.max(2, Math.round(len * SPOT_DENSITY));
    const nv = Math.max(3, Math.round(2 * Math.PI * spec.rMid * SPOT_DENSITY));
    return (t, theta, out) => {
      // local +x is the VENTRAL side of a forward-pointing bone (see the header note),
      // so cos(2π·theta) = +1 is the belly and -1 the spine.
      const ventral = Math.cos(theta * Math.PI * 2) * 0.5 + 0.5;
      out.copy(cBack).lerp(cCoat, smoothstep01(1 - Math.abs(ventral - 0.5) * 1.6));
      if (spec.belly > 0) out.lerp(cBelly, spec.belly * smoothstep01((ventral - 0.52) / 0.42));
      if (spec.spots > 0) {
        const mask = spotMask(t, theta, nu, nv, seed ^ boneSeed, 0.3);
        const fade = smoothstep01((0.86 - ventral) / 0.3) * smoothstep01((1 - Math.abs(t * 2 - 1)) / 0.22);
        out.lerp(spotColor, mask * spec.spots * fade);
      }
      if (spec.sock > 0) out.lerp(cSock, spec.sock * smoothstep01((t - 0.42) / 0.5));
    };
  };

  // ---- bones ----
  const boneMeshes: THREE.Mesh[] = [];
  const boneGeometries: THREE.BufferGeometry[] = [];
  for (let i = 0; i < CHEETAH_BONES.length; i++) {
    const base = COATS[i] ?? THIGH;
    const spec: CoatSpec = {
      ...base,
      rings: Math.max(6, Math.round(base.rings * lod)),
      radial: Math.max(6, Math.round(base.radial * lod)),
    };
    const len = boneLen[i] ?? 0.2;
    // the coat's profile floor is the SIM's capsule radius: a bone mesh always contains
    // the capsule the physics actually collides with
    const geo = limbGeometry(len, spec, coatFor(spec, len, i * 0x9e37), `cheetah_${BONE_NAMES[i] ?? i}`, boneR);
    boneGeometries.push(geo);
    const mesh = new THREE.Mesh(geo, coatMaterial);
    mesh.name = `cheetah_${BONE_NAMES[i] ?? i}`;
    mesh.castShadow = shadows;
    group.add(mesh);
    boneMeshes.push(mesh);
  }

  // ---- head: a child of the exact `head` joint, yawed by the torso's own angle ----
  const headNode = new THREE.Object3D();
  headNode.name = 'cheetahHeadNode';
  joints[FK_POINT_INDEX.head]?.add(headNode);

  // Every number below is a MULTIPLE of the sim's own head radius, not a world constant:
  // the debug overlay draws a circle of exactly `headRadius` at this joint, and the skull
  // has to be that ball however big the model says it is.
  const headParts: ColoredPart[] = [];
  {
    const skull = new THREE.SphereGeometry(headR, Math.max(10, Math.round(20 * lod)), Math.max(8, Math.round(14 * lod)));
    skull.scale(1.08, 0.94, 0.9);
    headParts.push({ geometry: skull, color: cCoat });
  }
  const headGeometry = mergeColoredParts(headParts, 'cheetahSkull');
  // repaint the skull per-vertex: crown spots, cream cheek, the tear line, a dark nose
  {
    const pos = headGeometry.getAttribute('position');
    const col = headGeometry.getAttribute('color');
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i) / headR;
      const y = pos.getY(i) / headR;
      const z = pos.getZ(i) / headR;
      const up = clampRange((y / 0.933) * 0.5 + 0.5, 0, 1);
      scratchColor.copy(cBack).lerp(cCoat, up);
      scratchColor.lerp(cBelly, smoothstep01((-y - 0.133) / 0.667) * 0.75);
      // spots over the crown
      const mask = spotMask(
        clampRange(x / 2.267 + 0.5, 0, 1),
        clampRange(Math.atan2(z, y) / (Math.PI * 2) + 0.5, 0, 1),
        4,
        6,
        seed ^ 0x51a5,
        0.34,
      );
      scratchColor.lerp(spotColor, mask * 0.5 * smoothstep01((y + 0.133) / 0.533));
      // the tear line: eye corner -> muzzle, the one marking that says "cheetah"
      const tear = Math.abs(z) > 0.367 ? smoothstep01(1 - Math.abs((y + 0.133) - (x - 0.4) * 0.55) / 0.187) : 0;
      scratchColor.lerp(spotColor, tear * clampRange((x + 0.133) / 0.667, 0, 1) * 0.9);
      col.setXYZ(i, scratchColor.r, scratchColor.g, scratchColor.b);
    }
    col.needsUpdate = true;
  }
  const headMesh = new THREE.Mesh(headGeometry, coatMaterial);
  headMesh.name = 'cheetahSkull';
  headMesh.castShadow = shadows;
  headNode.add(headMesh);

  const featureParts: ColoredPart[] = [];
  {
    const muzzle = new THREE.CylinderGeometry(0.367 * headR, 0.633 * headR, 0.6 * headR, Math.max(6, Math.round(10 * lod)));
    muzzle.rotateZ(-Math.PI / 2);
    muzzle.translate(0.9 * headR, -0.28 * headR, 0);
    featureParts.push({ geometry: muzzle, color: cBelly });
    const nose = new THREE.SphereGeometry(0.213 * headR, 8, 6);
    nose.scale(0.85, 0.8, 1);
    nose.translate(1.187 * headR, -0.213 * headR, 0);
    featureParts.push({ geometry: nose, color: cNose });
    for (const s of [-1, 1]) {
      const ear = new THREE.ConeGeometry(0.333 * headR, 0.467 * headR, 6);
      ear.rotateZ(s * 0.12);
      ear.rotateX(-s * 0.26);
      ear.translate(-0.187 * headR, 0.833 * headR, s * 0.48 * headR);
      featureParts.push({ geometry: ear, color: cBack });
      const eye = new THREE.SphereGeometry(0.127 * headR, 7, 5);
      eye.translate(0.587 * headR, 0.187 * headR, s * 0.587 * headR);
      featureParts.push({ geometry: eye, color: cNose });
    }
  }
  const featureGeometry = mergeColoredParts(featureParts, 'cheetahFace');
  const featureMesh = new THREE.Mesh(featureGeometry, coatMaterial);
  featureMesh.name = 'cheetahFace';
  featureMesh.castShadow = shadows;
  headNode.add(featureMesh);

  // ---- tail: a chain hung off the exact `bt` joint ----
  const tailRoot = new THREE.Object3D();
  tailRoot.name = 'cheetahTailRoot';
  joints[FK_POINT_INDEX.bt]?.add(tailRoot);
  const tailNodes: THREE.Object3D[] = [];
  const tailGeometries: THREE.BufferGeometry[] = [];
  {
    let parent: THREE.Object3D = tailRoot;
    for (let i = 0; i < TAIL_LEN.length; i++) {
      const node = new THREE.Object3D();
      node.name = `cheetahTail${i}`;
      node.position.y = i === 0 ? 0 : TAIL_LEN[i - 1] ?? 0;
      parent.add(node);
      const len = TAIL_LEN[i] ?? 0.2;
      const r0 = TAIL_R[i] ?? 0.04;
      const r1 = TAIL_R[i + 1] ?? 0.022;
      const last = i === TAIL_LEN.length - 1;
      const spec: CoatSpec = {
        r0,
        rMid: (r0 + r1) * 0.5,
        r1,
        belly: 0,
        sock: 0,
        spots: last ? 0 : 0.55,
        deep: 1,
        wide: 1,
        rings: Math.max(5, Math.round(10 * lod)),
        radial: Math.max(6, Math.round(10 * lod)),
      };
      const paint: CoatColorFn = (t, theta, out) => {
        out.copy(cCoat).lerp(cBack, 0.3);
        if (spec.spots > 0) {
          const mask = spotMask(t, theta, Math.max(2, Math.round(len * SPOT_DENSITY)), 5, seed ^ (0x77 + i), 0.4);
          out.lerp(spotColor, mask * 0.6);
        }
        // the white tail flag: every cheetah has one, and it is what you see at 200 m
        if (last) out.lerp(cTip, smoothstep01((t - 0.45) / 0.4));
      };
      // no radius floor: the physics model has no tail, so nothing here has to contain a
      // capsule — the tip really does taper to TAIL_R's last entry
      const geo = limbGeometry(len, spec, paint, `cheetahTailSeg${i}`, 0);
      geo.translate(0, len * 0.5, 0);
      tailGeometries.push(geo);
      const mesh = new THREE.Mesh(geo, coatMaterial);
      mesh.name = `cheetahTailMesh${i}`;
      mesh.castShadow = shadows;
      node.add(mesh);
      tailNodes.push(node);
      parent = node;
    }
  }

  // ---- ground contact: dust and a warm scuff under each planted toe ----
  const dustColor = blendTokens(palette.groundB, palette.sun, 0.35);
  const dust: AmbientParticlesHandle[] = [];
  for (let i = 0; i < 2; i++) {
    const handle = makeAmbientParticles({
      preset: 'dust',
      count: 130,
      bounds: { y: 0.08, halfX: 0.24, halfY: 0.11, halfZ: 0.13 },
      seed: seed ^ (0x3d91 + i * 977),
      tier,
      color: dustColor,
      size: 1.5,
      intensity: 0.75,
      opacity: 0.34,
      rise: 0.14,
      swirl: 0.05,
      name: `cheetahDust${i}`,
    });
    handle.setDensity(0);
    group.add(handle.points);
    dust.push(handle);
  }

  const scuffSprite = radialSprite(64, 2.2);
  const scuffGeometry = new THREE.PlaneGeometry(1, 1);
  scuffGeometry.rotateX(-Math.PI / 2);
  const scuffMaterials: EmissiveMaterial[] = [];
  const scuffs: THREE.Mesh[] = [];
  for (let i = 0; i < 2; i++) {
    const material = emissiveMaterial(blendTokens(dustColor, palette.sun, 0.3), 0.5, {
      map: scuffSprite,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      name: `cheetahScuffMaterial${i}`,
    });
    material.opacity = 0;
    const mesh = new THREE.Mesh(scuffGeometry, material);
    mesh.name = `cheetahScuff${i}`;
    mesh.position.y = 0.006;
    mesh.scale.set(0.5, 1, 0.34);
    mesh.renderOrder = 2;
    mesh.visible = false;
    group.add(mesh);
    scuffMaterials.push(material);
    scuffs.push(mesh);
  }

  // ---- preallocated scratch (setPose allocates nothing) ----
  const boneAngle = new Float64Array(CHEETAH_BONES.length);
  const TOE_INDEX: readonly number[] = [FK_POINT_INDEX.toe_b, FK_POINT_INDEX.toe_f];
  // the tail solver's working set: a parent frame, this vertebra's frame, the minimal
  // bend that lifts it, and its heading
  const Z_AXIS = new THREE.Vector3(0, 0, 1);
  const tailEuler = new THREE.Euler();
  const tailFrame = new THREE.Quaternion();
  const tailSegment = new THREE.Quaternion();
  const tailBend = new THREE.Quaternion();
  const tailDir = new THREE.Vector3();
  const tailLift = new THREE.Vector3();

  return {
    group,
    joints,

    setPose(fk, speed01) {
      // 1. the joints — exactly the FK points, nothing else
      for (let i = 0; i < joints.length; i++) {
        joints[i]?.position.set(fk[i * 2] ?? 0, fk[i * 2 + 1] ?? 0, 0);
      }

      // 2. the bones — midpoint + heading, both read off the joints
      for (let i = 0; i < CHEETAH_BONES.length; i++) {
        const bone = CHEETAH_BONES[i];
        const mesh = boneMeshes[i];
        if (!bone || !mesh) continue;
        const ax = fk[bone[0] * 2] ?? 0;
        const ay = fk[bone[0] * 2 + 1] ?? 0;
        const bx = fk[bone[1] * 2] ?? 0;
        const by = fk[bone[1] * 2 + 1] ?? 0;
        const dx = bx - ax;
        const dy = by - ay;
        mesh.position.set((ax + bx) * 0.5, (ay + by) * 0.5, 0);
        // R_z maps local +y onto (dx, dy)
        mesh.rotation.z = Math.atan2(-dx, dy);
        boneAngle[i] = Math.atan2(dy, dx);
      }

      // 3. head + tail hang off the torso's own heading (bone 0 is bt -> ft)
      const forward = boneAngle[0] ?? 0;
      headNode.rotation.z = forward;
      const gait = (boneAngle[3] ?? 0) - (boneAngle[6] ?? 0);
      const stream = clampRange(speed01, 0, 1);

      // THE TAIL IS SOLVED, NOT POSED. +z rotation from "straight back" is DOWN, so a
      // resting tail hangs and a running one lifts and streams out as a counterweight —
      // but the chain is 0.86 long against a hip that stands about 0.6 up, so the art
      // droop alone would sink the tip through the very plane this scene exists to prove.
      //
      // So each vertebra is walked in WORLD space: it takes its droop + swing, and if that
      // heading would put the far end of the segment below its own radius (plus one capsule
      // radius of daylight), it bends up by exactly the angle that lands it THERE and no
      // more. Both endpoints of every segment then clear the plane by at least their own
      // radius, and radius is linear along the segment, so no point of the tail's surface
      // is ever below the plane — in the shipped lane it never comes within 4 cm of it.
      // Every term is a pure function of (fk, speed01) — no ground query, no history, no
      // clock — so a scrub reproduces the identical tail.
      const rootAngle = forward + Math.PI * 0.5 + TAIL_HANG - TAIL_RUN_LIFT * stream;
      tailRoot.rotation.z = rootAngle;
      tailFrame.setFromAxisAngle(Z_AXIS, rootAngle);
      const backX = -Math.cos(forward);
      let endY = fk[FK_POINT_INDEX.bt * 2 + 1] ?? 0; // the tail base: the exact hip point
      for (let i = 0; i < tailNodes.length; i++) {
        const node = tailNodes[i];
        if (!node) continue;
        const amp = TAIL_SWING[i] ?? 0.2;
        tailEuler.set(
          amp * 0.9 * Math.sin(gait * 1.7 + i * 0.85 + 1.2),
          0,
          (TAIL_DROOP[i] ?? 0.2) * (1 - TAIL_STREAM * stream) + amp * Math.sin(gait * 1.7 + i * 0.6),
        );
        tailSegment.setFromEuler(tailEuler).premultiply(tailFrame);
        tailDir.set(0, 1, 0).applyQuaternion(tailSegment);

        const len = TAIL_LEN[i] ?? 0.2;
        // its own radius, plus one CAPSULE radius of daylight: the height at which a toe
        // counts as planted, so the tail can never be mistaken for a contact
        const floor = (TAIL_R[i + 1] ?? 0) + boneR;
        const climb = (floor - endY) / len; // the shallowest heading the ground allows
        if (tailDir.y < climb) {
          const y = climb > 1 ? 1 : climb;
          const flat = Math.hypot(tailDir.x, tailDir.z);
          const spread = Math.sqrt(1 - y * y);
          // keep the vertebra's azimuth (its swing out of the sagittal plane), raise only
          // its pitch; a heading that is exactly vertical keeps the tail's own "back"
          tailLift.set(
            (flat > 1e-6 ? tailDir.x / flat : backX >= 0 ? 1 : -1) * spread,
            y,
            (flat > 1e-6 ? tailDir.z / flat : 0) * spread,
          );
          tailBend.setFromUnitVectors(tailDir, tailLift);
          tailSegment.premultiply(tailBend);
          tailDir.copy(tailLift);
        }

        // local = parent⁻¹ · world, written straight onto the node (three keeps its Euler
        // in sync); the next vertebra inherits the frame this one ended up with
        node.quaternion.copy(tailFrame).invert().multiply(tailSegment);
        tailFrame.copy(tailSegment);
        endY += tailDir.y * len;
      }

      // 4. ground contact — a pure function of toe height and speed, so a scrub lands
      //    the dust exactly where the frame says the foot is
      for (let i = 0; i < 2; i++) {
        const idx = TOE_INDEX[i] ?? 0;
        const tx = fk[idx * 2] ?? 0;
        const ty = fk[idx * 2 + 1] ?? 0;
        const contact = 1 - smoothstep01((ty - CONTACT_LOW) / (CONTACT_HIGH - CONTACT_LOW));
        const kick = contact * stream;
        const puff = dust[i];
        if (puff) {
          puff.points.position.set(tx, 0.09, 0);
          puff.setDensity(kick * 0.95);
        }
        const scuff = scuffs[i];
        const material = scuffMaterials[i];
        if (scuff && material) {
          scuff.position.x = tx;
          const k = contact * (0.35 + 0.65 * stream);
          material.opacity = 0.26 * k;
          scuff.scale.set(0.34 + 0.5 * k, 1, 0.24 + 0.26 * k);
          scuff.visible = k > 0.02;
        }
      }
    },

    setTime(t) {
      for (const d of dust) d.setTime(t);
    },

    setGlow(gain) {
      const g = gain > 0 ? gain : 0;
      coatMaterial.color.setRGB(g, g, g);
    },

    setPixelRatio(dpr) {
      for (const d of dust) d.setPixelRatio(dpr);
    },

    dispose() {
      for (const g of boneGeometries) g.dispose();
      for (const g of tailGeometries) g.dispose();
      boneGeometries.length = 0;
      tailGeometries.length = 0;
      headGeometry.dispose();
      featureGeometry.dispose();
      coatMaterial.dispose();
      for (const d of dust) d.dispose();
      dust.length = 0;
      for (const m of scuffMaterials) m.dispose();
      scuffMaterials.length = 0;
      scuffGeometry.dispose();
      scuffSprite.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
