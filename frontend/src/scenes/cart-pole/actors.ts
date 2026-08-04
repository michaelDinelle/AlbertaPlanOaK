// The Boardwalk — the planter trolley and the sapling it carries.
//
// ACCURACY. Two probe objects, both carrying POSITION ONLY:
//   * `root`  — world (x, DECK_Y, 0). Nothing else: no bob, no lean, no roll.
//   * `tip`   — world (x + L·sin th, L·cos th, 0), i.e. exactly `cartPoleTip(x, th)`.
// The hinge sits at local y = −DECK_Y above the root, which puts it back on the env's
// y = 0 plane, and the planter's soil surface is that same plane — so the sapling really
// does leave the ground at the pivot the physics uses. The `tilt` node between them is a
// PURE ROTATION of −th about z (three rotates +y to (−sin a, cos a); we need
// (+sin th, cos th), hence −th), so deleting every decoration would leave a correct,
// invisible pole.
//
// IDEMPOTENCE. Everything in `setState` is a pure function of (x, th, xdot, force):
// wheel roll is −x/r, the push shimmer reads the recorded action, the crown's lean is
// the pole's own lean. The only event-driven visual is the shed/settle beat, and it
// lives on the `crown` node — a SIBLING of the tip probe — so no flourish can ever move
// a state-bearing object.

import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { streamFor } from '@kit/rng';
import { radialSprite } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import {
  CP_CART_HEIGHT,
  CP_CART_WIDTH,
  CP_POLE_WIDTH,
  CP_TH_LIMIT,
} from '@sim/envs/cartpole';
import {
  blendTokens,
  clampRange,
  DECK_Y,
  mergeColoredParts,
  TOKEN_BLACK,
  TOKEN_WHITE,
  type ColoredPart,
} from './world';

// ---------------------------------------------------------------------------
// dimensions (world units; the planter box IS the env's cart box)
// ---------------------------------------------------------------------------

/** the hinge, in root-local coordinates: exactly back up to the env's y = 0 */
export const HINGE_LOCAL_Y = -DECK_Y;
const WHEEL_R = 0.06;
const WHEEL_T = 0.028;
const WHEEL_Z = 0.115;
const AXLE_X = 0.132;
/** the planter's soil surface is the hinge plane */
const BOX_TOP = HINGE_LOCAL_Y;
const BOX_BOTTOM = BOX_TOP - CP_CART_HEIGHT;
const BOX_HALF_Z = 0.15;
/** where the crown of leaves hangs on the pole (tilt-local) */
export const CROWN_Y = 0.78;

const TRUNK_R_BASE = CP_POLE_WIDTH / 2;
const TRUNK_R_TIP = CP_POLE_WIDTH / 6;

/**
 * THE WHORL ENVELOPE — the fix for the crown/tip contradiction.
 *
 * The cord states the POLE TIP's bound, and the tip is at radius exactly `poleLength`
 * from the hinge. A leaf whorl that reaches FURTHER than that says the opposite of what
 * the env says: with the old free-growing whorl the visible green top sat at radius
 * 1.196·L, so it only crossed the cord at 35.1° — 2.9× the real 12° bound — and at the
 * exact failure pose the sapling still read SAFE while the env had already killed it.
 *
 * So every leaf blade is now clamped to land ON a sphere of this radius about the hinge,
 * and the topmost thing on the sapling is a terminal bud whose apex is at radius exactly
 * `poleLength`. The number is set so that even with the crown's full ±0.30 rad droop
 * (lean + shed, which peak together at a fall) no leaf vertex reaches the cord height
 * L·cos(12°) = 0.9781·L — checked off the built mesh in cart-pole.test.ts. The whorl
 * therefore states "below the line" and the bud states "the line", exactly.
 */
const WHORL_RADIUS = 0.9;
/** the lowest and highest leaf BASE on the trunk, as fractions of the pole length */
const WHORL_BASE = 0.36;
const WHORL_TOP = 0.82;
/** the leaf outline's own length (its furthest vertex from the blade's root) */
const LEAF_REACH = 0.215;
/**
 * THE TERMINAL BUD, and why it is a broad cap rather than a spire.
 *
 * Only ONE point can mark the tip without lying: anything with x-extent at radius L rises
 * by that extent times sin(th) when the pole leans, so a blunt top overshoots the cord.
 * A cone's apex is that point exactly — but a SLENDER cone is optically invisible, and
 * that is just as much of a lie: at r/h = 0.22 its top 8 mm is under a pixel wide, it
 * antialiases into the reeds, and the eye reads the sapling's top ~2.3 px low out of a
 * 6.1 px band (shot, measured, rejected — as was a z-aligned ridge, which is exact under
 * the pole's z-rotation but foreshortens to a point in this near-axial camera).
 *
 * The fix is the aspect ratio. At r/h = 0.71 the cap is a pixel wide 0.7 px below its
 * apex and 16 px wide at its base, so what a viewer calls "the top of the sapling" and
 * `cartPoleTip(x, th)` are the same point to within a pixel — with no overshoot at all,
 * because the apex is still a single vertex at radius exactly L.
 */
const BUD_HEIGHT = 0.042;
const BUD_RADIUS = 0.03;
const COLLAR_HEIGHT = 0.02;
const COLLAR_RADIUS = 0.024;

export interface TrolleyHandle {
  /** add this ONE group to the scene */
  group: THREE.Group;
  /** probe: world position === (x, DECK_Y, 0), always */
  root: THREE.Object3D;
  /** probe: world position === cartPoleTip(x, th) */
  tip: THREE.Object3D;
  /** world position of the leaf crown, refreshed by setState (preallocated) */
  readonly crownWorld: THREE.Vector3;
  /** the whole visual state, from the frame and nothing else */
  setState(x: number, th: number, xDot: number, force: number): void;
  /** ctx.time() — drives the shed motes; the trolley itself is state-driven */
  setTime(t: number): void;
  /** 0..1 fall beat, `dir` = the lean the pole fell with (−1 left, +1 right) */
  setShed(env01: number, dir: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

export interface TrolleyOptions {
  tier: QualityTier;
  palette: Palette;
  accent: THREE.ColorRepresentation;
  seed: number;
  /** drawn pole length, read from the env's `poleTipRest` anchor */
  poleLength: number;
  /** half-angle of the failure cone, read from the `failArc` anchors */
  tiltLimit: number;
}

export function buildTrolley(opts: TrolleyOptions): TrolleyHandle {
  const { tier, palette, accent, seed, poleLength } = opts;
  const tiltLimit = opts.tiltLimit > 1e-6 ? opts.tiltLimit : CP_TH_LIMIT;
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'trolleyRig';

  // Noon albedo law (canon lesson 19): every timber tone is pulled toward night.skyTop.
  const woodPale = blendTokens(palette.groundB, TOKEN_BLACK, 0.3);
  const woodMid = blendTokens(palette.groundA, TOKEN_BLACK, 0.2);
  const woodDark = blendTokens(palette.groundA, TOKEN_BLACK, 0.42);
  const iron = blendTokens(blendTokens(palette.hemiGround, palette.skyTop, 0.42), TOKEN_BLACK, 0.42);
  const soil = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.42);
  const moss = blendTokens(palette.groundA, palette.groundB, 0.4);
  // the trunk has to separate from the pale reed bank it stands against
  const bark = blendTokens(palette.groundA, TOKEN_BLACK, 0.54);
  // a young leaf is bright, but `accent` under a 3.0 noon key with a 0.32 toon floor
  // clips to white — both tones stay pulled toward the darker ground token
  const leafA = blendTokens(blendTokens(accent, palette.groundA, 0.5), TOKEN_BLACK, 0.14);
  const leafB = blendTokens(blendTokens(accent, palette.groundA, 0.18), TOKEN_BLACK, 0.1);
  // The terminal marker reads as SURVEYOR'S KIT, not foliage: the collar carries the
  // cord's own colour so the eye pairs them, and the bud is the palest thing on the
  // sapling so the one point the bound is about is also the one that catches the sky.
  const collarTone = blendTokens(TOKEN_WHITE, palette.hemiSky, 0.34);
  const budTone = blendTokens(leafB, TOKEN_WHITE, 0.32);

  // ---- the probe root: position only, forever ----
  const root = new THREE.Object3D();
  root.name = 'trolleyRoot';
  group.add(root);

  // ---- the planter body ----
  const bodyParts: ColoredPart[] = [];
  {
    const halfW = CP_CART_WIDTH / 2;
    // staved crate: 7 vertical boards per long side, 3 per end
    const STAVES = 7;
    for (let i = 0; i < STAVES; i++) {
      const k = (i + 0.5) / STAVES;
      const px = -halfW + k * CP_CART_WIDTH;
      for (const z of [-BOX_HALF_Z, BOX_HALF_Z]) {
        const stave = new THREE.BoxGeometry(CP_CART_WIDTH / STAVES - 0.006, CP_CART_HEIGHT, 0.022);
        stave.translate(px, BOX_BOTTOM + CP_CART_HEIGHT * 0.5, z);
        bodyParts.push({ geometry: stave, color: i % 2 === 0 ? woodPale : woodMid });
      }
    }
    for (let i = 0; i < 3; i++) {
      const k = (i + 0.5) / 3;
      const pz = -BOX_HALF_Z + k * BOX_HALF_Z * 2;
      for (const x of [-halfW, halfW]) {
        const end = new THREE.BoxGeometry(0.022, CP_CART_HEIGHT, (BOX_HALF_Z * 2) / 3 - 0.006);
        end.translate(x, BOX_BOTTOM + CP_CART_HEIGHT * 0.5, pz);
        bodyParts.push({ geometry: end, color: i % 2 === 0 ? woodMid : woodPale });
      }
    }
    // iron bands
    for (const y of [BOX_BOTTOM + 0.045, BOX_TOP - 0.04]) {
      for (const z of [-BOX_HALF_Z - 0.006, BOX_HALF_Z + 0.006]) {
        const band = new THREE.BoxGeometry(CP_CART_WIDTH + 0.018, 0.02, 0.012);
        band.translate(0, y, z);
        bodyParts.push({ geometry: band, color: iron });
      }
      for (const x of [-halfW - 0.006, halfW + 0.006]) {
        const band = new THREE.BoxGeometry(0.012, 0.02, BOX_HALF_Z * 2 + 0.012);
        band.translate(x, y, 0);
        bodyParts.push({ geometry: band, color: iron });
      }
    }
    // the chassis the axles hang from
    const chassis = new THREE.BoxGeometry(CP_CART_WIDTH - 0.03, 0.03, BOX_HALF_Z * 2 - 0.02);
    chassis.translate(0, BOX_BOTTOM - 0.012, 0);
    bodyParts.push({ geometry: chassis, color: woodDark });
    for (const x of [-AXLE_X, AXLE_X]) {
      const bracket = new THREE.BoxGeometry(0.028, 0.05, WHEEL_Z * 2 + 0.03);
      bracket.translate(x, WHEEL_R + 0.03, 0);
      bodyParts.push({ geometry: bracket, color: iron });
    }
    // the soil, mounded just under the rim, with a moss cap where the sapling stands
    const soilTop = new THREE.BoxGeometry(CP_CART_WIDTH - 0.03, 0.03, BOX_HALF_Z * 2 - 0.03);
    soilTop.translate(0, BOX_TOP - 0.016, 0);
    bodyParts.push({ geometry: soilTop, color: soil });
    const mound = new THREE.IcosahedronGeometry(0.11, 1);
    mound.scale(1.5, 0.34, 1.05);
    mound.translate(0, BOX_TOP - 0.006, 0);
    bodyParts.push({ geometry: mound, color: soil });
    const mossCap = new THREE.IcosahedronGeometry(0.062, 1);
    mossCap.scale(1.35, 0.3, 1.15);
    mossCap.translate(0, BOX_TOP + 0.006, 0);
    bodyParts.push({ geometry: mossCap, color: moss });
  }
  const bodyGeometry = mergeColoredParts(bodyParts, 'trolleyBody');
  const bodyMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.82,
    rimColor: palette.skyHorizon,
    rimStrength: 0.7,
    toonSteps: 3,
    toonSoftness: 0.28,
    // a shadow side on the HERO prop must stay painterly, never a hole in the frame
    toonFloor: 0.4,
    tier,
    name: 'trolleyBodyMaterial',
  });
  const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
  bodyMesh.name = 'trolleyBody';
  bodyMesh.castShadow = shadows;
  bodyMesh.receiveShadow = shadows;
  root.add(bodyMesh);

  // ---- wheels: one merged geometry per axle, rotating about its own hub ----
  const wheelGeometry = makeAxleGeometry(woodDark, iron);
  const wheelFront = new THREE.Mesh(wheelGeometry, bodyMaterial);
  wheelFront.name = 'trolleyAxleFront';
  wheelFront.position.set(AXLE_X, WHEEL_R, 0);
  wheelFront.castShadow = shadows;
  const wheelRear = new THREE.Mesh(wheelGeometry, bodyMaterial);
  wheelRear.name = 'trolleyAxleRear';
  wheelRear.position.set(-AXLE_X, WHEEL_R, 0);
  wheelRear.castShadow = shadows;
  root.add(wheelFront, wheelRear);

  // ---- the hinge: back on the env's y = 0 plane ----
  const hinge = new THREE.Object3D();
  hinge.name = 'poleHinge';
  hinge.position.set(0, HINGE_LOCAL_Y, 0);
  root.add(hinge);

  const tilt = new THREE.Object3D();
  tilt.name = 'poleTilt';
  hinge.add(tilt);

  /** the tip probe: pure position, no decoration ever */
  const tip = new THREE.Object3D();
  tip.name = 'poleTip';
  tip.position.set(0, poleLength, 0);
  tilt.add(tip);

  // ---- the sapling trunk ----
  const trunkParts: ColoredPart[] = [];
  {
    const radial = tier === 'thumb' ? 5 : 7;
    // The shaft stops INSIDE the bud. A cylinder that ran the full length would put its
    // top rim at radius √(L² + r²) > L, and a tilted rim corner would then sit 2.7 mm
    // above the cord at the exact bound — small, but it is the one measurement this
    // scene is about, so the only geometry out at radius L is the bud's apex vertex.
    const shaft = poleLength - BUD_HEIGHT * 0.6;
    const trunk = new THREE.CylinderGeometry(TRUNK_R_TIP, TRUNK_R_BASE, shaft, radial, 3);
    trunk.translate(0, shaft * 0.5, 0);
    trunkParts.push({ geometry: trunk, color: bark });
    // a flare where it meets the soil
    const flare = new THREE.ConeGeometry(TRUNK_R_BASE * 2.1, 0.075, radial);
    flare.translate(0, 0.03, 0);
    trunkParts.push({ geometry: flare, color: blendTokens(bark, soil, 0.4) });
    // three twigs, each ending where a leaf cluster starts
    const twigs: readonly (readonly [number, number])[] = [
      [0.5, 0.9],
      [0.64, -2.2],
      [0.76, 2.6],
    ];
    for (const [y, yaw] of twigs) {
      const len = 0.1;
      const twig = new THREE.CylinderGeometry(0.006, 0.011, len, 4);
      twig.translate(0, len * 0.5, 0);
      twig.rotateZ(-1.02);
      twig.rotateY(yaw);
      twig.translate(0, y * poleLength, 0);
      trunkParts.push({ geometry: twig, color: bark });
    }
    // THE TERMINAL MARKER, rigid on the pole (a sibling of the tip probe would be a
    // decoration; this is part of the trunk, so it can never drift off the tip). The
    // collar is a painted band that pairs the sapling with the cord's own colour; the
    // bud's RIDGE is the pole tip itself, at y = poleLength for every tilt.
    const collar = new THREE.CylinderGeometry(COLLAR_RADIUS, COLLAR_RADIUS + 0.004, COLLAR_HEIGHT, radial);
    collar.translate(0, poleLength - 0.07, 0);
    trunkParts.push({ geometry: collar, color: collarTone });
    // a cone's apex sits at +h/2, so this puts its topmost vertex at exactly poleLength
    const bud = new THREE.ConeGeometry(BUD_RADIUS, BUD_HEIGHT, radial + 2);
    bud.translate(0, poleLength - BUD_HEIGHT * 0.5, 0);
    trunkParts.push({ geometry: bud, color: budTone });
  }
  const trunkGeometry = mergeColoredParts(trunkParts, 'saplingTrunk');
  const trunkMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.9,
    rimColor: palette.skyHorizon,
    rimStrength: 0.6,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.24,
    tier,
    name: 'saplingTrunkMaterial',
  });
  const trunkMesh = new THREE.Mesh(trunkGeometry, trunkMaterial);
  trunkMesh.name = 'saplingTrunk';
  trunkMesh.castShadow = shadows;
  tilt.add(trunkMesh);

  // ---- the crown: a SIBLING of the probe, so flourishes can never move the tip ----
  const crown = new THREE.Object3D();
  crown.name = 'saplingCrown';
  crown.position.set(0, CROWN_Y, 0);
  tilt.add(crown);

  const leafParts: ColoredPart[] = [];
  {
    const rng = streamFor(seed, 'leaves');
    const count = tier === 'thumb' ? 11 : tier === 'medium' ? 18 : 26;
    const envelope = poleLength * WHORL_RADIUS;
    for (let i = 0; i < count; i++) {
      const k = i / (count - 1);
      // leaves ride the upper trunk, authored about the crown origin
      const base = poleLength * (WHORL_BASE + k * (WHORL_TOP - WHORL_BASE));
      const y = base - CROWN_Y;
      const yaw = i * 2.399963 + rng() * 0.5;
      // The whorl lifts and shortens as it climbs, so the crown tapers to the leader
      // instead of bursting at the tip. The pitch ramp is DELIBERATELY gentle: a blade
      // aimed steeply upward runs straight at the envelope, so a steep top whorl would
      // be clamped to stubs, while a lifted-but-not-vertical one keeps its length.
      const pitch = 0.2 + 0.55 * k * k + rng() * 0.16;
      // A blade rooted at (0, base, 0) pointing along a unit direction with vertical
      // component sin(pitch) meets the whorl envelope at exactly this length — solve
      // |base·ŷ + t·d| = envelope for t. Clamping the blade there is what keeps every
      // leaf under the cord while the bud alone states the bound.
      const reach =
        Math.sqrt(Math.max(0, envelope * envelope - base * base * Math.cos(pitch) ** 2)) -
        base * Math.sin(pitch);
      const scale = Math.min(1.5 - 0.78 * k + rng() * 0.24, reach / LEAF_REACH);
      if (!(scale > 0.02)) continue;
      leafParts.push({
        geometry: leafGeometry(y, yaw, pitch, scale),
        color: i % 3 === 0 ? leafA : leafB,
      });
    }
  }
  const leafGeo = mergeColoredParts(leafParts, 'saplingLeaves');
  const leafMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.72,
    side: THREE.DoubleSide,
    rimColor: palette.skyHorizon,
    rimStrength: 0.55,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.32,
    // the leaves flutter; the trunk NEVER does — it is the rigid pole
    wind: { amplitude: 0.009, frequency: 2.6, height: 0.32 },
    tier,
    name: 'saplingLeafMaterial',
  });
  const leafMesh = new THREE.Mesh(leafGeo, leafMaterial);
  leafMesh.name = 'saplingLeaves';
  leafMesh.castShadow = shadows;
  crown.add(leafMesh);

  // ---- the push shimmer: the recorded action, drawn on the deck ----
  const shimmerSprite = radialSprite(48, 2.6);
  const shimmerMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(accent, palette.sun, 0.5),
    1.35,
    {
      map: shimmerSprite,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      name: 'trolleyPushShimmer',
    },
  );
  shimmerMaterial.opacity = 0;
  const shimmerGeometry = new THREE.PlaneGeometry(0.3, 0.2);
  shimmerGeometry.rotateX(-Math.PI / 2);
  const shimmerLeft = new THREE.Mesh(shimmerGeometry, shimmerMaterial);
  shimmerLeft.name = 'trolleyShimmerLeft';
  shimmerLeft.position.set(-0.26, 0.006, 0);
  shimmerLeft.visible = false;
  const shimmerRight = new THREE.Mesh(shimmerGeometry, shimmerMaterial);
  shimmerRight.name = 'trolleyShimmerRight';
  shimmerRight.position.set(0.26, 0.006, 0);
  shimmerRight.visible = false;
  root.add(shimmerLeft, shimmerRight);

  // ---- the shed: leaf motes shaken loose by a fall ----
  const shed: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'snow',
    count: 120,
    bounds: { halfX: 0.3, halfY: 0.44, halfZ: 0.22 },
    seed: seed ^ 0x4d91,
    tier,
    color: blendTokens(leafB, TOKEN_WHITE, 0.2),
    size: 0.5,
    intensity: 1.25,
    opacity: 0.85,
    rise: -0.55,
    swirl: 0.09,
    name: 'saplingShed',
  });
  shed.setDensity(0);
  group.add(shed.points);

  const crownWorld = new THREE.Vector3();
  let shedEnv = 0;
  let shedDir = 1;

  return {
    group,
    root,
    tip,
    crownWorld,

    setState(x, th, xDot, force) {
      root.position.set(x, DECK_Y, 0);
      // three rotates local +y to (−sin a, cos a); the tip must land on
      // (+L sin th, L cos th), so the tilt angle is −th.
      tilt.rotation.z = -th;

      // the wheels roll as a pure function of position — a scrub reproduces the spokes
      const roll = -x / WHEEL_R;
      wheelFront.rotation.z = roll;
      wheelRear.rotation.z = roll;

      // the shove: shown behind the wheel it pushes from, and only while it bites
      const speed01 = clampRange(Math.abs(xDot) / 2.5, 0, 1);
      const pushing = force !== 0;
      shimmerLeft.visible = pushing && force > 0;
      shimmerRight.visible = pushing && force < 0;
      shimmerMaterial.opacity = pushing ? 0.1 + 0.34 * speed01 : 0;

      // the crown carries the fall beat; the settle is a lean, never a stretch, so the
      // drawn pole is always exactly `poleLength` long
      const lean = clampRange(th / tiltLimit, -1, 1);
      crown.rotation.z = -0.12 * lean - 0.18 * shedEnv * shedDir;
      const settle = 1 - 0.11 * shedEnv;
      crown.scale.set(settle, settle, settle);

      // crown + shed follow the pole without a matrix read (allocation-free)
      const s = Math.sin(th);
      const c = Math.cos(th);
      crownWorld.set(x + CROWN_Y * s, CROWN_Y * c, 0);
      shed.points.position.set(crownWorld.x, crownWorld.y - 0.1, 0);
    },

    setTime(t) {
      shed.setTime(t);
      oakUniforms(leafMaterial)?.setTime(t);
    },

    setShed(env01, dir) {
      shedEnv = clampRange(env01, 0, 1);
      shedDir = dir < 0 ? -1 : 1;
      shed.setDensity(shedEnv);
    },

    setPixelRatio(dpr) {
      shed.setPixelRatio(dpr);
    },

    dispose() {
      bodyGeometry.dispose();
      wheelGeometry.dispose();
      bodyMaterial.dispose();
      trunkGeometry.dispose();
      trunkMaterial.dispose();
      leafGeo.dispose();
      leafMaterial.dispose();
      shimmerGeometry.dispose();
      shimmerMaterial.dispose();
      shimmerSprite.dispose();
      shed.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

/**
 * One lanceolate leaf blade, built straight into its place on the crown: authored
 * pointing +x in the XY plane, pitched up by `pitch`, spun to `yaw`, and lifted to `y`
 * (all crown-local). Six vertices, four triangles — a leaf at this size is entirely its
 * outline.
 */
function leafGeometry(y: number, yaw: number, pitch: number, scale: number): THREE.BufferGeometry {
  const outline: readonly (readonly [number, number])[] = [
    [0, 0],
    [0.05, 0.028],
    [0.15, 0.018],
    [0.215, 0],
    [0.15, -0.018],
    [0.05, -0.028],
  ];
  const tris: readonly (readonly [number, number, number])[] = [
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    [0, 4, 5],
  ];
  const v = new Float32Array(tris.length * 9);
  let o = 0;
  for (const t of tris) {
    for (const i of t) {
      const p = outline[i] as readonly [number, number];
      v[o++] = p[0];
      v[o++] = p[1];
      v[o++] = 0;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(v, 3));
  g.scale(scale, scale, scale);
  // lie the blade flat, tip up by `pitch`, then swing it around the trunk
  g.rotateX(-Math.PI / 2);
  g.rotateZ(pitch);
  g.rotateY(yaw);
  g.translate(0, y, 0);
  g.computeVertexNormals();
  g.name = 'saplingLeaf';
  return g;
}

/** Two wheels + a hub + spokes, merged; the axle's local origin is the hub centre. */
function makeAxleGeometry(rim: THREE.Color, hub: THREE.Color): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const hubBar = new THREE.CylinderGeometry(0.009, 0.009, WHEEL_Z * 2 + WHEEL_T, 6);
  hubBar.rotateX(Math.PI / 2);
  parts.push({ geometry: hubBar, color: hub });
  for (const z of [-WHEEL_Z, WHEEL_Z]) {
    const tyre = new THREE.CylinderGeometry(WHEEL_R, WHEEL_R, WHEEL_T, 12, 1, true);
    tyre.rotateX(Math.PI / 2);
    tyre.translate(0, 0, z);
    parts.push({ geometry: tyre, color: hub });
    const disc = new THREE.CylinderGeometry(WHEEL_R * 0.88, WHEEL_R * 0.88, WHEEL_T * 0.55, 12);
    disc.rotateX(Math.PI / 2);
    disc.translate(0, 0, z);
    parts.push({ geometry: disc, color: rim });
    for (let s = 0; s < 3; s++) {
      const spoke = new THREE.BoxGeometry(WHEEL_R * 1.8, 0.009, WHEEL_T * 0.9);
      spoke.rotateZ((s / 3) * Math.PI);
      spoke.translate(0, 0, z);
      parts.push({ geometry: spoke, color: hub });
    }
  }
  return mergeColoredParts(parts, 'trolleyAxle');
}
