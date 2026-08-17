
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { blendTokens, clampRange, envelope, TOKEN_BLACK, TOKEN_WHITE } from '@kit/mix';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { streamFor } from '@kit/rng';
import { radialSprite } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import * as THREE from 'three';
import {
  CP_CART_HEIGHT,
  CP_CART_WIDTH,
  CP_POLE_WIDTH,
  CP_TH_LIMIT,
} from '@sim/envs/cartpole';
import { DECK_Y } from './world';


export const HINGE_LOCAL_Y = -DECK_Y;
const WHEEL_R = 0.06;
const WHEEL_T = 0.028;
const WHEEL_Z = 0.115;
const AXLE_X = 0.132;
const BOX_TOP = HINGE_LOCAL_Y;
const BOX_BOTTOM = BOX_TOP - CP_CART_HEIGHT;
const BOX_HALF_Z = 0.15;
export const CROWN_Y = 0.78;

const TRUNK_R_BASE = CP_POLE_WIDTH / 2;
const TRUNK_R_TIP = CP_POLE_WIDTH / 6;

const WHORL_RADIUS = 0.9;
const WHORL_BASE = 0.36;
const WHORL_TOP = 0.82;
const LEAF_REACH = 0.215;
const BUD_HEIGHT = 0.042;
const BUD_RADIUS = 0.03;
const COLLAR_HEIGHT = 0.02;
const COLLAR_RADIUS = 0.024;

export interface TrolleyHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  tip: THREE.Object3D;
  readonly crownWorld: THREE.Vector3;
  setState(x: number, th: number, xDot: number, force: number): void;
  setTime(t: number): void;
  setShed(env01: number, dir: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

export interface TrolleyOptions {
  palette: Palette;
  accent: THREE.ColorRepresentation;
  seed: number;
  poleLength: number;
  tiltLimit: number;
}

export function buildTrolley(opts: TrolleyOptions): TrolleyHandle {
  const { palette, accent, seed, poleLength } = opts;
  const tiltLimit = opts.tiltLimit > 1e-6 ? opts.tiltLimit : CP_TH_LIMIT;
  const group = new THREE.Group();
  group.name = 'trolleyRig';

  const woodPale = blendTokens(palette.groundB, TOKEN_BLACK, 0.3);
  const woodMid = blendTokens(palette.groundA, TOKEN_BLACK, 0.2);
  const woodDark = blendTokens(palette.groundA, TOKEN_BLACK, 0.42);
  const iron = blendTokens(blendTokens(palette.hemiGround, palette.skyTop, 0.42), TOKEN_BLACK, 0.42);
  const soil = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.42);
  const moss = blendTokens(palette.groundA, palette.groundB, 0.4);
  const bark = blendTokens(palette.groundA, TOKEN_BLACK, 0.54);
  const leafA = blendTokens(blendTokens(accent, palette.groundA, 0.5), TOKEN_BLACK, 0.14);
  const leafB = blendTokens(blendTokens(accent, palette.groundA, 0.18), TOKEN_BLACK, 0.1);
  const collarTone = blendTokens(TOKEN_WHITE, palette.hemiSky, 0.34);
  const budTone = blendTokens(leafB, TOKEN_WHITE, 0.32);

  const root = new THREE.Object3D();
  root.name = 'trolleyRoot';
  group.add(root);

  const bodyParts: ColoredPart[] = [];
  {
    const halfW = CP_CART_WIDTH / 2;
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
    const chassis = new THREE.BoxGeometry(CP_CART_WIDTH - 0.03, 0.03, BOX_HALF_Z * 2 - 0.02);
    chassis.translate(0, BOX_BOTTOM - 0.012, 0);
    bodyParts.push({ geometry: chassis, color: woodDark });
    for (const x of [-AXLE_X, AXLE_X]) {
      const bracket = new THREE.BoxGeometry(0.028, 0.05, WHEEL_Z * 2 + 0.03);
      bracket.translate(x, WHEEL_R + 0.03, 0);
      bodyParts.push({ geometry: bracket, color: iron });
    }
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
    toonFloor: 0.4,
    name: 'trolleyBodyMaterial',
  });
  const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
  bodyMesh.name = 'trolleyBody';
  bodyMesh.castShadow = true;
  bodyMesh.receiveShadow = true;
  root.add(bodyMesh);

  const wheelGeometry = makeAxleGeometry(woodDark, iron);
  const wheelFront = new THREE.Mesh(wheelGeometry, bodyMaterial);
  wheelFront.name = 'trolleyAxleFront';
  wheelFront.position.set(AXLE_X, WHEEL_R, 0);
  wheelFront.castShadow = true;
  const wheelRear = new THREE.Mesh(wheelGeometry, bodyMaterial);
  wheelRear.name = 'trolleyAxleRear';
  wheelRear.position.set(-AXLE_X, WHEEL_R, 0);
  wheelRear.castShadow = true;
  root.add(wheelFront, wheelRear);

  const hinge = new THREE.Object3D();
  hinge.name = 'poleHinge';
  hinge.position.set(0, HINGE_LOCAL_Y, 0);
  root.add(hinge);

  const tilt = new THREE.Object3D();
  tilt.name = 'poleTilt';
  hinge.add(tilt);

  const tip = new THREE.Object3D();
  tip.name = 'poleTip';
  tip.position.set(0, poleLength, 0);
  tilt.add(tip);

  const trunkParts: ColoredPart[] = [];
  {
    const radial = 7;
    const shaft = poleLength - BUD_HEIGHT * 0.6;
    const trunk = new THREE.CylinderGeometry(TRUNK_R_TIP, TRUNK_R_BASE, shaft, radial, 3);
    trunk.translate(0, shaft * 0.5, 0);
    trunkParts.push({ geometry: trunk, color: bark });
    const flare = new THREE.ConeGeometry(TRUNK_R_BASE * 2.1, 0.075, radial);
    flare.translate(0, 0.03, 0);
    trunkParts.push({ geometry: flare, color: blendTokens(bark, soil, 0.4) });
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
    const collar = new THREE.CylinderGeometry(COLLAR_RADIUS, COLLAR_RADIUS + 0.004, COLLAR_HEIGHT, radial);
    collar.translate(0, poleLength - 0.07, 0);
    trunkParts.push({ geometry: collar, color: collarTone });
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
    name: 'saplingTrunkMaterial',
  });
  const trunkMesh = new THREE.Mesh(trunkGeometry, trunkMaterial);
  trunkMesh.name = 'saplingTrunk';
  trunkMesh.castShadow = true;
  tilt.add(trunkMesh);

  const crown = new THREE.Object3D();
  crown.name = 'saplingCrown';
  crown.position.set(0, CROWN_Y, 0);
  tilt.add(crown);

  const leafParts: ColoredPart[] = [];
  {
    const rng = streamFor(seed, 'leaves');
    const count = 26;
    const envelope = poleLength * WHORL_RADIUS;
    for (let i = 0; i < count; i++) {
      const k = i / (count - 1);
      const base = poleLength * (WHORL_BASE + k * (WHORL_TOP - WHORL_BASE));
      const y = base - CROWN_Y;
      const yaw = i * 2.399963 + rng() * 0.5;
      const pitch = 0.2 + 0.55 * k * k + rng() * 0.16;
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
    wind: { amplitude: 0.009, frequency: 2.6, height: 0.32 },
    name: 'saplingLeafMaterial',
  });
  const leafMesh = new THREE.Mesh(leafGeo, leafMaterial);
  leafMesh.name = 'saplingLeaves';
  leafMesh.castShadow = true;
  crown.add(leafMesh);

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

  const shed: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'snow',
    count: 120,
    bounds: { halfX: 0.3, halfY: 0.44, halfZ: 0.22 },
    seed: seed ^ 0x4d91,
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
      tilt.rotation.z = -th;

      const roll = -x / WHEEL_R;
      wheelFront.rotation.z = roll;
      wheelRear.rotation.z = roll;

      const speed01 = clampRange(Math.abs(xDot) / 2.5, 0, 1);
      const pushing = force !== 0;
      shimmerLeft.visible = pushing && force > 0;
      shimmerRight.visible = pushing && force < 0;
      shimmerMaterial.opacity = pushing ? 0.1 + 0.34 * speed01 : 0;

      const lean = clampRange(th / tiltLimit, -1, 1);
      crown.rotation.z = -0.12 * lean - 0.18 * shedEnv * shedDir;
      const settle = 1 - 0.11 * shedEnv;
      crown.scale.set(settle, settle, settle);

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
  g.rotateX(-Math.PI / 2);
  g.rotateZ(pitch);
  g.rotateY(yaw);
  g.translate(0, y, 0);
  g.computeVertexNormals();
  g.name = 'saplingLeaf';
  return g;
}

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
