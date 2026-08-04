// The Ivy Gate — the three state-bearing props.
//
// ACCURACY: each handle exposes a `root` whose world position is EXACTLY `cellToWorld` of
// the cell the frame says it is on, with y = 0 and nothing else. Every swing, wrap,
// growth, breath and glow lives on children. Delete every decoration and what remains is
// three correctly-placed empty nodes.
//
// THE GATE IS THE SCENE. `doorOpen` is one bit of the frame, and this file spends its
// whole budget making that bit unmissable from across the garden: a locked gate is SEWN
// SHUT by an ivy vine that crosses it in an X, and an open one has that X gone and the oak
// leaf swung back against the wall. The two beats are sequenced off ONE scalar phase —
// ivy first, then the door — so the animation reads as cause and effect (the ivy lets go,
// therefore the door can move) and reverses correctly on the relock without a second
// state machine.
//
// THE VINE IS ONE DRAW. It is an InstancedMesh of ivy sprigs sampled along a fixed path
// through the gate's local space: up the near jamb as a helix, across the lintel, down the
// far jamb, then twice across the door face. Each sprig's position, tangent frame and tint
// are baked at build time; a frame only re-rolls it for the breeze and scales it by
// `clamp((tip − u)/feather)` — the vine's live TIP walking back along its own path — which
// is what makes it grow and retract like a vine instead of fading out like a decal.
//
// IDEMPOTENCE: nothing in this file integrates or remembers. `setState` is a pure function
// of (phase, growth, glow, ctx.time()), so calling it twice with the same arguments leaves
// the same world, and a scrub that snaps the phase lands exactly on the frame a settled
// gate would show.

import { GoalBeacon, KeyToken, cellFieldDetail } from '@kit/grid';
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { hash1f } from '@kit/rng';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import {
  DIVIDER_CAP,
  DIVIDER_HEIGHT,
  DIVIDER_THICKNESS,
  clamp01,
  ivyLeafGeometry,
  mergeColoredParts,
  smoothBand,
  type ColoredPart,
  type GardenSkin,
} from './world';

// ---------------------------------------------------------------------------
// the gate
// ---------------------------------------------------------------------------

/** underside of the lintel — the doorway's clear height */
const DOOR_HEIGHT = DIVIDER_HEIGHT - 0.02;
/** the leaf spans the opening minus the two jambs */
const DOOR_WIDTH = 0.9;
const DOOR_THICKNESS = 0.075;
/** hinge on the FAR jamb, so an open leaf swings away from the doorway we are looking at */
const HINGE_X = 0.45;
/**
 * The leaf hangs on the wall's OUTER face, not down the middle of the passage.
 *
 * This is the single change that made the gate readable. A wall 0.7 thick with a leaf
 * centred in it puts the whole door at the bottom of a 35 cm recess, and from the composed
 * aerial the near jamb simply covers it — the frame's subject was a dark slot. Hung on
 * the face (which is where a garden gate hangs anyway) the leaf, its ironwork and the ivy
 * crossing it are all in the open, and the swing still clears into the wild garden.
 */
export const DOOR_FACE_Z = -(DIVIDER_THICKNESS * 0.5) + 0.055;
/**
 * How far the leaf swings. Negative sends the free end west (into the wild garden) rather
 * than east; at −1.38 the open leaf clears the doorway and stops just short of the
 * neighbouring wall cell, so it never intersects the divider it hangs in.
 */
const DOOR_SWING = -1.38;

/** the vine's fraction that survives an open gate: the jambs and lintel keep their ivy */
export const IVY_FRAME_SPAN = 0.62;
/** how sharply the vine's tip fades in/out along `u` */
const IVY_FEATHER = 0.075;

/**
 * The two beats, in phase units: the ivy releases first, then the leaf moves.
 *
 * THE BEATS ARE FRONT-LOADED, AND THAT IS A MEASUREMENT, NOT A TASTE. The env opens the
 * door and steps the agent into the door cell in the SAME control step, so the wisp's hop
 * crosses the leaf's own plane about 0.07 s after the `door` event arrives (measured on
 * the demo lane at the harness's 60 fps / 8 Hz cadence). Beats that finish late render the
 * exact inverse of this scene's sentence: the orb walks through a sewn-shut gate and the
 * empty gate then opens by itself. So the ivy's X is gone by ~0.04 s and the leaf is clear
 * of the passage by ~0.07 s — while the ORDER survives, because that is the argument (the
 * ivy lets go, therefore the door can move), and the relock still runs the same scalar
 * backwards over its own long window, where there is no agent to overtake.
 */
const IVY_BEAT: readonly [number, number] = [0.0, 0.32];
const SWING_BEAT: readonly [number, number] = [0.06, 0.8];

export interface GateOptions {
  tier: QualityTier;
  palette: Palette;
  skin: GardenSkin;
  /** yaw that turns the local build (opening along ±z) onto the env's own wall axis */
  yaw: number;
  /** lane seed — STYLE.md: kit/rng seeded with ctx.seed is the ONLY randomness a scene has */
  seed: number;
  name?: string;
}

export interface GateHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(door), y = 0, always */
  root: THREE.Object3D;
  /**
   * The whole visual state from ONE scalar. 0 = locked and sewn shut, 1 = ivy released and
   * the leaf swung open. `time` drives only the breeze, never the pose.
   */
  setState(phase01: number, time: number): void;
  /** ctx.time() — the material clocks */
  setTime(t: number): void;
  dispose(): void;
}

interface VineSample {
  x: number;
  y: number;
  z: number;
  tx: number;
  ty: number;
  tz: number;
}

/**
 * The vine's path through the gate's LOCAL space (opening along ±z, jambs at x = ±0.47).
 *
 * u ∈ [0, IVY_FRAME_SPAN) is the frame ivy — a helix up the near jamb, a run across the
 * lintel, a short drop down the far jamb. u ∈ [IVY_FRAME_SPAN, 1] is the DOOR ivy: two
 * runners crossing the leaf in an X, at z just in front of its face. Retracting from u = 1
 * therefore unpicks the X first and leaves the architecture green, which is the whole
 * visual argument: the ivy was holding the door, and now it is not.
 */
export function vinePoint(u: number, out: THREE.Vector3): THREE.Vector3 {
  const t = clamp01(u);
  const top = DIVIDER_HEIGHT + DIVIDER_CAP + 0.02;
  if (t < 0.3) {
    // Up the near (−x) jamb, 2.4 turns of a flattened helix. The helix NARROWS as it
    // climbs, from the pier's full depth to its set-back head: a vine grips the stone it
    // is on, and — measured — a strand still swinging out at the old 0.27 above the
    // springing line is the last thing left hanging in the thumb's line to the doorway.
    const k = t / 0.3;
    const a = k * Math.PI * 2 * 2.4;
    return out.set(-0.45 + 0.13 * Math.cos(a), 0.05 + (top - 0.05) * k, (0.27 - 0.13 * k) * Math.sin(a));
  }
  if (t < 0.5) {
    // across the lintel, still weaving over and under it
    const k = (t - 0.3) / 0.2;
    const a = Math.PI * 2 * 2.4 + k * Math.PI * 2 * 1.6;
    return out.set(-0.45 + 0.9 * k, top + 0.13 + 0.04 * Math.sin(a * 1.7), 0.17 * Math.sin(a));
  }
  if (t < IVY_FRAME_SPAN) {
    // a short drop down the far (+x) jamb
    const k = (t - 0.5) / (IVY_FRAME_SPAN - 0.5);
    const a = Math.PI * 2 * 4 + k * Math.PI * 2 * 0.9;
    return out.set(0.45 + 0.12 * Math.cos(a), top + 0.13 - k * 0.6, 0.17 * Math.sin(a));
  }
  // the X across the door face
  const face = DOOR_FACE_Z - DOOR_THICKNESS * 0.5 - 0.035;
  if (t < 0.83) {
    const k = (t - IVY_FRAME_SPAN) / (0.83 - IVY_FRAME_SPAN);
    return out.set(0.42 - 0.86 * k, top - 0.45 - 0.28 * k, face + 0.012 * Math.sin(k * 9));
  }
  const k = (t - 0.83) / (1 - 0.83);
  return out.set(-0.4 + 0.78 * k, 0.14 + 0.5 * k, face + 0.012 * Math.sin(k * 7 + 2));
}

/** Sample the path plus its forward difference, so a sprig can be aimed along the vine. */
function sampleVine(u: number, du: number, a: THREE.Vector3, b: THREE.Vector3, out: VineSample): VineSample {
  vinePoint(u, a);
  vinePoint(Math.min(1, u + du), b);
  b.sub(a);
  if (b.lengthSq() < 1e-12) b.set(0, 1, 0);
  b.normalize();
  out.x = a.x;
  out.y = a.y;
  out.z = a.z;
  out.tx = b.x;
  out.ty = b.y;
  out.tz = b.z;
  return out;
}

export function buildGate(opts: GateOptions): GateHandle {
  const { tier, palette, skin, yaw, seed } = opts;
  const detail = cellFieldDetail(tier);
  const shadows = tier !== 'thumb';

  const group = new THREE.Group();
  group.name = opts.name ?? 'ivyGate';

  // ---- the probe root: position only, forever ----
  const root = new THREE.Object3D();
  root.name = 'ivyGateRoot';
  group.add(root);

  // the whole gate turns as one, so the local build never has to know the wall's axis
  const frame = new THREE.Object3D();
  frame.name = 'ivyGateFrame';
  frame.rotation.y = yaw;
  root.add(frame);

  // ---- the stone: two jambs, a cap course, a keystone, a worn threshold ----
  const stoneGeometry = makeGateStoneGeometry(skin, tier);
  const stoneMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.93,
    rimColor: palette.skyHorizon,
    rimStrength: 0.14,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.22,
    tier,
    name: 'ivyGateStoneMaterial',
  });
  const stone = new THREE.Mesh(stoneGeometry, stoneMaterial);
  stone.name = 'ivyGateStone';
  stone.castShadow = shadows;
  stone.receiveShadow = shadows;
  frame.add(stone);

  // ---- the leaf, on a hinge at the far jamb ----
  const pivot = new THREE.Object3D();
  pivot.name = 'ivyGateHinge';
  pivot.position.set(HINGE_X, 0.03, DOOR_FACE_Z);
  frame.add(pivot);

  const doorGeometry = makeDoorGeometry(skin, tier);
  const doorMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.78,
    rimColor: palette.skyHorizon,
    rimStrength: 0.18,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.28,
    tier,
    name: 'ivyGateDoorMaterial',
  });
  const door = new THREE.Mesh(doorGeometry, doorMaterial);
  door.name = 'ivyGateDoor';
  door.castShadow = shadows;
  door.receiveShadow = shadows;
  pivot.add(door);

  // ---- the vine ----
  const sprigCount = Math.max(14, Math.round(64 * detail));
  const sprigGeometry = makeIvySprigGeometry(skin, tier);
  const ivyMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.74,
    rimColor: palette.skyHorizon,
    rimStrength: 0.42,
    toonSteps: 3,
    toonSoftness: 0.34,
    toonFloor: 0.3,
    tier,
    name: 'ivyVineMaterial',
  });
  const vine = new THREE.InstancedMesh(sprigGeometry, ivyMaterial, sprigCount);
  vine.name = 'ivyVine';
  vine.castShadow = false;
  vine.receiveShadow = false;
  vine.frustumCulled = false;
  // The strand that holds the DOOR is young growth: a lighter tint, set once, so the X
  // across the dark oak reads as the thing being undone rather than as more wall ivy. One
  // instance-colour buffer, no second draw and no per-frame cost.
  vine.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(sprigCount * 3), 3);
  frame.add(vine);

  // ---- bake the vine's frame: position, tangent-aligned orientation, phase ----
  const us = new Float32Array(sprigCount);
  const basePos = new Float32Array(sprigCount * 3);
  const baseQuat = new Float32Array(sprigCount * 4);
  const axis = new Float32Array(sprigCount * 3);
  const phase = new Float32Array(sprigCount);
  const sizes = new Float32Array(sprigCount);
  {
    const UP = new THREE.Vector3(0, 1, 0);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const t = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const roll = new THREE.Quaternion();
    const sample: VineSample = { x: 0, y: 0, z: 0, tx: 0, ty: 1, tz: 0 };
    const tint = new THREE.Color();
    const du = 1 / (sprigCount * 4);
    for (let i = 0; i < sprigCount; i++) {
      const u = sprigCount === 1 ? 0 : i / (sprigCount - 1);
      sampleVine(u, du, a, b, sample);
      us[i] = u;
      basePos[i * 3] = sample.x;
      basePos[i * 3 + 1] = sample.y;
      basePos[i * 3 + 2] = sample.z;
      t.set(sample.tx, sample.ty, sample.tz);
      axis[i * 3] = t.x;
      axis[i * 3 + 1] = t.y;
      axis[i * 3 + 2] = t.z;
      q.setFromUnitVectors(UP, t);
      // a deterministic base roll so consecutive sprigs fan around the stem
      const r = hash1f(i, seed ^ 0x4b17) * Math.PI * 2 + i * 1.9;
      roll.setFromAxisAngle(t, r);
      q.premultiply(roll);
      baseQuat[i * 4] = q.x;
      baseQuat[i * 4 + 1] = q.y;
      baseQuat[i * 4 + 2] = q.z;
      baseQuat[i * 4 + 3] = q.w;
      phase[i] = hash1f(i, seed ^ 0x2c9d) * Math.PI * 2;
      // the door strand runs SMALLER as well as lighter: at full size it smothered the
      // leaf it is supposed to be holding, and a gate you cannot see is not a locked gate
      sizes[i] = (u >= IVY_FRAME_SPAN ? 0.7 : 0.88) + hash1f(i, seed ^ 0x77e1) * 0.36;
      const young = u >= IVY_FRAME_SPAN ? 1.22 : 0.98;
      tint.setRGB(young, young, young);
      vine.setColorAt(i, tint);
    }
    if (vine.instanceColor) vine.instanceColor.needsUpdate = true;
  }

  // ---- preallocated scratch: setState runs every frame and may not allocate ----
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const rollQuat = new THREE.Quaternion();
  const axisVec = new THREE.Vector3();
  const scale = new THREE.Vector3(1, 1, 1);

  return {
    group,
    root,

    setState(phase01, time) {
      const p = clamp01(phase01);

      // BEAT 2 — the leaf. It cannot start until the ivy has let go of it.
      pivot.rotation.y = DOOR_SWING * smoothBand(SWING_BEAT[0], SWING_BEAT[1], p);

      // BEAT 1 — the vine retracts from its tip back to the frame span (and no further:
      // an open gate is still a green gate). `tip` is the u the vine currently reaches;
      // it overshoots to 1 + feather when the gate is locked so the LAST sprig is fully
      // grown rather than sitting permanently inside its own fade window.
      const release = smoothBand(IVY_BEAT[0], IVY_BEAT[1], p);
      const tip = IVY_FRAME_SPAN + (1 - IVY_FRAME_SPAN + IVY_FEATHER) * (1 - release);

      for (let i = 0; i < sprigCount; i++) {
        const u = us[i] ?? 0;
        const grow = clamp01((tip - u) / IVY_FEATHER);
        if (grow <= 0) {
          matrix.makeScale(0, 0, 0);
          vine.setMatrixAt(i, matrix);
          continue;
        }
        const ph = phase[i] ?? 0;
        // the breeze is a ROLL about the vine's own tangent: the sprig rustles in place
        // and the stem never leaves the path it was baked onto
        const rustle = Math.sin(time * 1.15 + ph) * 0.14 + Math.sin(time * 2.3 + ph * 1.7) * 0.05;
        axisVec.set(axis[i * 3] ?? 0, axis[i * 3 + 1] ?? 1, axis[i * 3 + 2] ?? 0);
        rollQuat.setFromAxisAngle(axisVec, rustle);
        quaternion.set(
          baseQuat[i * 4] ?? 0,
          baseQuat[i * 4 + 1] ?? 0,
          baseQuat[i * 4 + 2] ?? 0,
          baseQuat[i * 4 + 3] ?? 1,
        );
        quaternion.premultiply(rollQuat);
        position.set(basePos[i * 3] ?? 0, basePos[i * 3 + 1] ?? 0, basePos[i * 3 + 2] ?? 0);
        const s = (sizes[i] ?? 1) * grow;
        scale.set(s, s, s);
        matrix.compose(position, quaternion, scale);
        vine.setMatrixAt(i, matrix);
      }
      vine.instanceMatrix.needsUpdate = true;
    },

    setTime(t) {
      oakUniforms(stoneMaterial)?.setTime(t);
      oakUniforms(doorMaterial)?.setTime(t);
      oakUniforms(ivyMaterial)?.setTime(t);
    },

    dispose() {
      vine.dispose();
      sprigGeometry.dispose();
      ivyMaterial.dispose();
      doorGeometry.dispose();
      doorMaterial.dispose();
      stoneGeometry.dispose();
      stoneMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/** top of the gate's standing posts — the crown of the whole garden */
export const GATE_RIDGE = DIVIDER_HEIGHT + DIVIDER_CAP + 0.42;

/**
 * Where the jambs STEP BACK, and how deep the head course above the step is.
 *
 * THIS IS THE SIGHTLINE, AND IT IS ARITHMETIC. The avatar in the doorway is an orb at
 * y = 0.35 sitting in a slot whose near cheek is 0.45 away and 1.02 tall, so any camera
 * below atan(0.67 / 0.45) = 56° of elevation has that cheek across the orb's centre. The
 * composed aerial is 54.5° at hero and the same at thumb: 1.5° of margin and not a degree
 * more, with the thumb's shallower ray to the door cell using every bit of it. A raycast
 * census from the composed camera to the orb centre over all 31 free cells found exactly
 * one occluded cell — the door cell, the ONE cell this whole scene exists to dramatise —
 * hidden by `ivyGateStone` at hero and by its coping again at thumb.
 *
 * So the pier is STEPPED: full wall depth up to the springing line, then a head course set
 * back to 0.4, and the coping narrowed to the head it stands on instead of oversailing the
 * wall by 0.16 — which is precisely the horizontal cover the crown comment below condemns,
 * one course lower down. Census after the change: zero occluded cells at either tier. The
 * step also reads as masonry, the same heavy-base / light-head grammar the shrine lantern
 * is built from.
 */
const JAMB_HEAD_Y = 0.66;
const JAMB_HEAD_D = 0.4;

/**
 * The gateway, built in local space around the door cell's centre with the opening along
 * ±z. Two PALE dressed jambs on plinths, a stepped head course, and a vertical crown.
 *
 * WHY A CROWN, AND WHY NOT A ROOF. A flush opening cut into a wall of one tone is not a
 * gate from 13 units up, it is a gap — silhouette is the only channel a prop this size
 * still has at that distance. The first attempt put a little pitched roof over the
 * passage and it was strictly worse: at a plan-ish aerial ANY horizontal cover sits directly
 * between the camera and the doorway, so the roof hid the leaf, the ivy and the threshold
 * all at once. The crown is therefore VERTICAL — two posts standing a head above the wall
 * with pyramid caps, tied by a beam thin enough to occlude nothing. Two bright marks over
 * a dark wall, visible from any angle a plan-view camera can take.
 *
 * The coping obeys the same law, which the first pass did not: a cap 0.86 deep over a wall
 * 0.7 thick is exactly the horizontal cover this comment condemns, one course lower. It is
 * now the width of the head it sits on.
 */
export function makeGateStoneGeometry(skin: GardenSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = cellFieldDetail(tier);
  const parts: ColoredPart[] = [];
  const jambW = 0.115;
  const jambD = DIVIDER_THICKNESS;
  const capTop = DIVIDER_HEIGHT + DIVIDER_CAP + 0.03;

  // DRESSED stone, a clear step paler than the wall it stands in: an opening cut in a
  // wall of ONE tone is a hole, and a hole is not a gate.
  for (const sx of [-1, 1]) {
    const x = sx * (0.5 - jambW * 0.5);
    const jamb = new THREE.BoxGeometry(jambW, JAMB_HEAD_Y, jambD);
    jamb.translate(x, JAMB_HEAD_Y * 0.5, 0);
    parts.push({ geometry: jamb, color: skin.shrineStone });
    // the head course, set back off the passage so the doorway can be SEEN into
    const head = new THREE.BoxGeometry(jambW, DIVIDER_HEIGHT - JAMB_HEAD_Y, JAMB_HEAD_D);
    head.translate(x, (JAMB_HEAD_Y + DIVIDER_HEIGHT) * 0.5, 0);
    parts.push({ geometry: head, color: skin.shrineStone });
    // a string course on the step itself: the setback has to read as built, not as a
    // missing block. It sits at the springing line, well below the sightline.
    const band = new THREE.BoxGeometry(jambW + 0.05, 0.055, jambD + 0.03);
    band.translate(x, JAMB_HEAD_Y - 0.0275, 0);
    parts.push({ geometry: band, color: skin.dividerCap });
    // a plinth block at the foot of each jamb, so the gate sits ON something
    const foot = new THREE.BoxGeometry(jambW + 0.08, 0.15, jambD + 0.08);
    foot.translate(x, 0.075, 0);
    parts.push({ geometry: foot, color: skin.dividerCap });
  }

  const cap = new THREE.BoxGeometry(1.12, DIVIDER_CAP + 0.03, JAMB_HEAD_D + 0.06);
  cap.translate(0, DIVIDER_HEIGHT + (DIVIDER_CAP + 0.03) * 0.5, 0);
  parts.push({ geometry: cap, color: skin.dividerCap });

  // the header the leaf shuts against
  const header = new THREE.BoxGeometry(1 - jambW * 2 + 0.02, 0.12, 0.34);
  header.translate(0, DIVIDER_HEIGHT - 0.06, 0);
  parts.push({ geometry: header, color: skin.shrineDark });

  // ---- the crown: two standing posts and a tie beam, nothing over the passage ----
  const postH = GATE_RIDGE - capTop - 0.1;
  for (const sx of [-1, 1]) {
    const post = new THREE.BoxGeometry(0.13, postH, 0.16);
    post.translate(sx * 0.435, capTop + postH * 0.5, 0);
    parts.push({ geometry: post, color: skin.shrineStone });
    const pyramid = new THREE.ConeGeometry(0.13, 0.14, 4, 1);
    pyramid.rotateY(Math.PI / 4);
    pyramid.translate(sx * 0.435, capTop + postH + 0.07, 0);
    parts.push({ geometry: pyramid, color: skin.dividerCap });
  }
  // the tie beam is 9 cm deep on purpose: it draws a line between the posts and casts a
  // hairline shadow, and it hides nothing of the leaf below it
  const beam = new THREE.BoxGeometry(1.06, 0.075, 0.09);
  beam.translate(0, capTop + postH * 0.72, 0);
  parts.push({ geometry: beam, color: skin.shrineDark });

  // a worn threshold slab, flush with the flagstones
  const slab = new THREE.BoxGeometry(0.98, 0.06, jambD * 0.86);
  slab.translate(0, 0.028, 0);
  parts.push({ geometry: slab, color: skin.threshold });

  // two mossy kerb stones marking the passage, one either side
  const stones = tier === 'thumb' ? 0 : 2;
  for (let i = 0; i < stones; i++) {
    const s = new THREE.IcosahedronGeometry(0.1, Math.round(detail));
    s.scale(1, 0.7, 1.3);
    s.translate(i === 0 ? -0.36 : 0.36, 0.05, jambD * 0.5 - 0.06);
    parts.push({ geometry: s, color: skin.shrineDark });
  }

  return mergeColoredParts(parts, 'ivyGateStone');
}

/**
 * The oak leaf: five boards with a hairline between them, two iron straps and a ring.
 * Built with its hinge edge at the local origin and growing toward −x, so the pivot node
 * places it and the rotation opens it.
 */
export function makeDoorGeometry(skin: GardenSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = cellFieldDetail(tier);
  const parts: ColoredPart[] = [];
  const boards = tier === 'thumb' ? 3 : 5;
  const h = DOOR_HEIGHT - 0.06;
  const boardW = DOOR_WIDTH / boards;

  for (let i = 0; i < boards; i++) {
    const board = new THREE.BoxGeometry(boardW - 0.012, h, DOOR_THICKNESS);
    // a plank silhouette: the tops are cut at slightly different heights
    const lift = ((i * 5) % 3) * 0.012;
    board.translate(-(i + 0.5) * boardW, h * 0.5 - lift, 0);
    parts.push({ geometry: board, color: i % 2 === 0 ? skin.wood : skin.woodDark });
  }

  for (const y of [h * 0.22, h * 0.78]) {
    const strap = new THREE.BoxGeometry(DOOR_WIDTH - 0.03, 0.055, DOOR_THICKNESS + 0.022);
    strap.translate(-DOOR_WIDTH * 0.5, y, 0);
    parts.push({ geometry: strap, color: skin.iron });
  }

  const ring = new THREE.TorusGeometry(0.062, 0.016, Math.max(4, Math.round(6 * detail)), Math.max(8, Math.round(14 * detail)));
  ring.rotateY(Math.PI / 2);
  ring.translate(-DOOR_WIDTH + 0.15, h * 0.52, -DOOR_THICKNESS * 0.5 - 0.02);
  parts.push({ geometry: ring, color: skin.iron });

  return mergeColoredParts(parts, 'ivyGateDoor');
}

/**
 * ONE ivy sprig: a short stem along +y with two leaves splayed off it. The stems chain
 * along the baked path — each is a little longer than the sample spacing — so the
 * instances read as one continuous vine rather than as beads.
 */
export function makeIvySprigGeometry(skin: GardenSkin, tier: QualityTier): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const stemLength = 0.13;
  const stem = new THREE.BoxGeometry(0.023, stemLength, 0.023);
  stem.translate(0, stemLength * 0.5, 0);
  parts.push({ geometry: stem, color: skin.ivyDark });

  const leaves: readonly { at: number; side: number }[] =
    tier === 'thumb' ? [{ at: 0.5, side: 1 }] : [{ at: 0.3, side: -1 }, { at: 0.74, side: 1 }];
  for (const spec of leaves) {
    const leaf = ivyLeafGeometry();
    leaf.rotateZ(spec.side * 0.95);
    leaf.rotateX(-0.42);
    leaf.translate(spec.side * 0.012, stemLength * spec.at, 0.012);
    parts.push({ geometry: leaf, color: spec.side < 0 ? skin.ivy : skin.ivyDark });
  }

  return mergeColoredParts(parts, 'ivySprig');
}

// ---------------------------------------------------------------------------
// the shrine (the goal)
// ---------------------------------------------------------------------------

/** where the ember sits — the scene lifts its point light to this local height */
const SHRINE_CROWN = 0.66;
/**
 * The ember's HDR gain. It is the ONE surface in the formal garden that crosses the bloom
 * threshold, and it is deliberately small: the shrine has to read as a STONE LANTERN
 * holding a light, not as a light where a lantern used to be.
 */
const EMBER_HDR = 1.3;

export interface ShrineOptions {
  tier: QualityTier;
  palette: Palette;
  skin: GardenSkin;
  name?: string;
}

export interface ShrineHandle {
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(gx, gy), y = 0, always */
  root: THREE.Object3D;
  readonly crownHeight: number;
  /** growth (a fresh shrine rises from its plinth) and glow (the ember's breath) */
  setState(grow01: number, glow01: number): void;
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildShrine(opts: ShrineOptions): ShrineHandle {
  const { tier, palette, skin } = opts;
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = opts.name ?? 'goalShrine';

  const root = new THREE.Object3D();
  root.name = 'shrineRoot';
  group.add(root);

  // the kit beacon, demoted to a decoration child at the local origin: parented under
  // `root` it inherits the exact goal cell, so scaling it is pure decoration
  const beacon = new GoalBeacon({
    color: skin.amber,
    tier,
    height: 0.85,
    radius: 0.1,
    intensity: 0.34,
    name: 'shrineGlow',
  });
  root.add(beacon.group);

  const body = new THREE.Object3D();
  body.name = 'shrineBody';
  root.add(body);

  const stoneGeometry = makeShrineGeometry(skin, tier);
  const stoneMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.9,
    rimColor: palette.skyHorizon,
    rimStrength: 0.2,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.24,
    tier,
    name: 'shrineStoneMaterial',
  });
  const stone = new THREE.Mesh(stoneGeometry, stoneMaterial);
  stone.name = 'shrineStone';
  stone.castShadow = shadows;
  stone.receiveShadow = shadows;
  body.add(stone);

  const emberGeometry = new THREE.IcosahedronGeometry(0.075, tier === 'thumb' ? 0 : 1);
  const emberMaterial: EmissiveMaterial = emissiveMaterial(skin.amber, EMBER_HDR, { name: 'shrineEmber' });
  const ember = new THREE.Mesh(emberGeometry, emberMaterial);
  ember.name = 'shrineEmber';
  ember.position.set(0, SHRINE_CROWN, 0);
  body.add(ember);

  return {
    group,
    root,
    crownHeight: SHRINE_CROWN,

    setState(grow01, glow01) {
      const grow = clamp01(grow01);
      const glow = clamp01(glow01);
      // growth is a SCALE on the decoration node — the root never moves, so a shrine that
      // is 3% grown is still exactly on its cell
      body.scale.setScalar(grow);
      beacon.group.scale.setScalar(grow);
      ember.scale.setScalar(0.72 + 0.4 * glow);
      setEmissiveIntensity(emberMaterial, EMBER_HDR * (0.55 + 0.7 * glow) * (0.25 + 0.75 * grow));
    },

    flare(strength = 1) {
      beacon.flare(strength);
    },

    setTime(t) {
      // the lantern settles on its plinth with a barely-there lean, pure ctx.time()
      body.rotation.z = Math.sin(t * 0.6) * 0.012;
      ember.rotation.y = t * 0.7;
      oakUniforms(stoneMaterial)?.setTime(t);
    },

    update(dt, time) {
      beacon.update(dt, time);
    },

    dispose() {
      beacon.dispose();
      emberGeometry.dispose();
      emberMaterial.dispose();
      stoneGeometry.dispose();
      stoneMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/** A stone garden lantern: plinth, shaft, platform, light-box, roof, finial. */
export function makeShrineGeometry(skin: GardenSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = cellFieldDetail(tier);
  const radial = Math.max(5, Math.round(12 * detail));
  const parts: ColoredPart[] = [];

  const plinth = new THREE.BoxGeometry(0.54, 0.13, 0.54);
  plinth.translate(0, 0.065, 0);
  parts.push({ geometry: plinth, color: skin.shrineDark });

  const shaft = new THREE.CylinderGeometry(0.085, 0.125, 0.34, radial, 1);
  shaft.translate(0, 0.13 + 0.17, 0);
  parts.push({ geometry: shaft, color: skin.shrineStone });

  const platform = new THREE.CylinderGeometry(0.2, 0.17, 0.055, radial, 1);
  platform.translate(0, 0.5, 0);
  parts.push({ geometry: platform, color: skin.shrineDark });

  // the light box: six posts around the ember, so the flame is SEEN THROUGH stone
  const posts = tier === 'thumb' ? 4 : 6;
  for (let i = 0; i < posts; i++) {
    const a = (i / posts) * Math.PI * 2;
    const post = new THREE.BoxGeometry(0.045, 0.26, 0.045);
    post.translate(0.155, 0.53 + 0.13, 0);
    post.rotateY(a);
    parts.push({ geometry: post, color: skin.shrineStone });
  }

  const lid = new THREE.CylinderGeometry(0.235, 0.205, 0.05, radial, 1);
  lid.translate(0, 0.79, 0);
  parts.push({ geometry: lid, color: skin.shrineDark });

  const roof = new THREE.ConeGeometry(0.27, 0.19, radial, 1);
  roof.translate(0, 0.905, 0);
  parts.push({ geometry: roof, color: skin.shrineStone });

  const finial = new THREE.IcosahedronGeometry(0.048, tier === 'thumb' ? 0 : 1);
  finial.translate(0, 1.02, 0);
  parts.push({ geometry: finial, color: skin.shrineDark });

  return mergeColoredParts(parts, 'goalShrine');
}

// ---------------------------------------------------------------------------
// the seed bed (the key)
// ---------------------------------------------------------------------------

/** how high the seed hovers over its bed */
export const SEED_HOVER = 0.34;

export interface SeedBedOptions {
  tier: QualityTier;
  palette: Palette;
  skin: GardenSkin;
  name?: string;
}

export interface SeedBedHandle {
  group: THREE.Group;
  /** the probe object: world position === cellToWorld(kx, ky), y = 0, always */
  root: THREE.Object3D;
  /** the carryable seed — the scene reparents THIS under `Wisp.carry()` */
  token: KeyToken;
  /** where the token hangs when it is home (a child of `root`) */
  cradle: THREE.Object3D;
  /** growth of the bed's moss + how present the seed is (1 = lying here, 0 = carried) */
  setState(grow01: number, present01: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildSeedBed(opts: SeedBedOptions): SeedBedHandle {
  const { tier, palette, skin } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'seedBed';

  const root = new THREE.Object3D();
  root.name = 'seedBedRoot';
  group.add(root);

  const bed = new THREE.Object3D();
  bed.name = 'seedBedMoss';
  root.add(bed);

  const bedGeometry = makeSeedBedGeometry(skin, tier);
  const bedMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.92,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.28,
    tier,
    name: 'seedBedMaterial',
  });
  const mesh = new THREE.Mesh(bedGeometry, bedMaterial);
  mesh.name = 'seedBedStone';
  mesh.castShadow = false;
  mesh.receiveShadow = tier !== 'thumb';
  bed.add(mesh);

  // The dish holds a pool of amber light while the seed is lying in it, and goes cold the
  // instant the wisp lifts it out — the ONE cue that says "the key is not here any more"
  // without moving anything the frame did not move.
  const poolGeometry = new THREE.CircleGeometry(0.2, tier === 'thumb' ? 8 : 20);
  poolGeometry.rotateX(-Math.PI / 2);
  poolGeometry.name = 'seedBedPool';
  const poolMaterial: EmissiveMaterial = emissiveMaterial(skin.amber, 0.5, {
    transparent: true,
    opacity: 0.62,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'seedBedPool',
  });
  const pool = new THREE.Mesh(poolGeometry, poolMaterial);
  pool.name = 'seedBedPool';
  pool.position.y = 0.075;
  bed.add(pool);

  // the cradle is where the seed rests when nobody is carrying it; the token is a CHILD
  // of it, so "at home" is exactly `cellToWorld(kx, ky) + SEED_HOVER`
  const cradle = new THREE.Object3D();
  cradle.name = 'seedCradle';
  cradle.position.set(0, SEED_HOVER, 0);
  root.add(cradle);

  const token = new KeyToken({
    color: skin.amber,
    tier,
    radius: 0.105,
    hover: 0,
    intensity: 0.85,
    name: 'amberSeed',
  });
  token.group.position.set(0, 0, 0);
  cradle.add(token.group);

  return {
    group,
    root,
    token,
    cradle,

    setState(grow01, present01) {
      const grow = clamp01(grow01);
      const present = clamp01(present01);
      bed.scale.setScalar(grow);
      pool.scale.setScalar(0.7 + 0.35 * present);
      setEmissiveIntensity(poolMaterial, 0.5 * present * grow);
      poolMaterial.opacity = 0.62 * present;
      pool.visible = present > 0.01;
    },

    setTime(t) {
      oakUniforms(bedMaterial)?.setTime(t);
    },

    update(dt, time) {
      token.update(dt, time);
    },

    dispose() {
      token.dispose();
      poolGeometry.dispose();
      poolMaterial.dispose();
      bedGeometry.dispose();
      bedMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/** A mossy stone dish: a low ring of kerb stones around a cushion of moss. */
export function makeSeedBedGeometry(skin: GardenSkin, tier: QualityTier): THREE.BufferGeometry {
  const detail = cellFieldDetail(tier);
  const parts: ColoredPart[] = [];

  const cushion = new THREE.IcosahedronGeometry(0.19, tier === 'thumb' ? 0 : 1);
  cushion.scale(1, 0.36, 1);
  cushion.translate(0, 0.045, 0);
  parts.push({ geometry: cushion, color: skin.ivy });

  const stones = tier === 'thumb' ? 4 : 7;
  for (let i = 0; i < stones; i++) {
    const a = (i / stones) * Math.PI * 2;
    const s = new THREE.IcosahedronGeometry(0.062, Math.round(detail));
    s.scale(1.25, 0.72, 0.9);
    s.translate(0.235, 0.035, 0);
    s.rotateY(a + i * 0.31);
    parts.push({ geometry: s, color: i % 3 === 0 ? skin.shrineDark : skin.shrineStone });
  }

  return mergeColoredParts(parts, 'seedBed');
}
