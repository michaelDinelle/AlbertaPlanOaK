import * as THREE from 'three/webgpu';
import { GoalBeacon, KeyToken } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import {
  emissiveMaterial,
  oakUniforms,
  sculptureMaterial,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { clamp01 } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash1f } from '@kit/rng';
import type { Palette } from '@kit/tokens';
import {
  DIVIDER_CAP,
  DIVIDER_HEIGHT,
  DIVIDER_THICKNESS,
  ivyLeafGeometry,
  smoothBand,
  type GardenSkin,
} from './world';

const DOOR_HEIGHT = DIVIDER_HEIGHT - 0.02;
const DOOR_WIDTH = 0.9;
const DOOR_THICKNESS = 0.075;
const HINGE_X = 0.45;
const DOOR_FACE_Z = -(DIVIDER_THICKNESS * 0.5) + 0.055;
const DOOR_SWING = -1.38;

const IVY_FRAME_SPAN = 0.62;
const IVY_FEATHER = 0.075;

const IVY_BEAT: readonly [number, number] = [0.0, 0.32];
const SWING_BEAT: readonly [number, number] = [0.06, 0.8];

interface GateOptions {
  palette: Palette;
  skin: GardenSkin;
  yaw: number;
  seed: number;
  name?: string;
}

export interface GateHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setState(phase01: number, time: number): void;
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

function vinePoint(u: number, out: THREE.Vector3): THREE.Vector3 {
  const t = clamp01(u);
  const top = DIVIDER_HEIGHT + DIVIDER_CAP + 0.02;
  if (t < 0.3) {
    const k = t / 0.3;
    const a = k * Math.PI * 2 * 2.4;
    return out.set(-0.45 + 0.13 * Math.cos(a), 0.05 + (top - 0.05) * k, (0.27 - 0.13 * k) * Math.sin(a));
  }
  if (t < 0.5) {
    const k = (t - 0.3) / 0.2;
    const a = Math.PI * 2 * 2.4 + k * Math.PI * 2 * 1.6;
    return out.set(-0.45 + 0.9 * k, top + 0.13 + 0.04 * Math.sin(a * 1.7), 0.17 * Math.sin(a));
  }
  if (t < IVY_FRAME_SPAN) {
    const k = (t - 0.5) / (IVY_FRAME_SPAN - 0.5);
    const a = Math.PI * 2 * 4 + k * Math.PI * 2 * 0.9;
    return out.set(0.45 + 0.12 * Math.cos(a), top + 0.13 - k * 0.6, 0.17 * Math.sin(a));
  }
  const face = DOOR_FACE_Z - DOOR_THICKNESS * 0.5 - 0.035;
  if (t < 0.83) {
    const k = (t - IVY_FRAME_SPAN) / (0.83 - IVY_FRAME_SPAN);
    return out.set(0.42 - 0.86 * k, top - 0.45 - 0.28 * k, face + 0.012 * Math.sin(k * 9));
  }
  const k = (t - 0.83) / (1 - 0.83);
  return out.set(-0.4 + 0.78 * k, 0.14 + 0.5 * k, face + 0.012 * Math.sin(k * 7 + 2));
}

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
  const { palette, skin, yaw, seed } = opts;

  const group = new THREE.Group();
  group.name = opts.name ?? 'ivyGate';

  const root = new THREE.Object3D();
  root.name = 'ivyGateRoot';
  group.add(root);

  const frame = new THREE.Object3D();
  frame.name = 'ivyGateFrame';
  frame.rotation.y = yaw;
  root.add(frame);

  const stoneGeometry = makeGateStoneGeometry(skin);
  const stoneMaterial: OakMaterial = sculptureMaterial({
    roughness: 0.93,
    rimColor: palette.skyHorizon,
    rimStrength: 0.14,
    toonSoftness: 0.3,
    toonFloor: 0.22,
    name: 'ivyGateStoneMaterial',
  });
  const stone = new THREE.Mesh(stoneGeometry, stoneMaterial);
  stone.name = 'ivyGateStone';
  stone.castShadow = true;
  stone.receiveShadow = true;
  frame.add(stone);

  const pivot = new THREE.Object3D();
  pivot.name = 'ivyGateHinge';
  pivot.position.set(HINGE_X, 0.03, DOOR_FACE_Z);
  frame.add(pivot);

  const doorGeometry = makeDoorGeometry(skin);
  const doorMaterial: OakMaterial = sculptureMaterial({
    roughness: 0.78,
    rimColor: palette.skyHorizon,
    rimStrength: 0.18,
    toonSoftness: 0.32,
    toonFloor: 0.28,
    name: 'ivyGateDoorMaterial',
  });
  const door = new THREE.Mesh(doorGeometry, doorMaterial);
  door.name = 'ivyGateDoor';
  door.castShadow = true;
  door.receiveShadow = true;
  pivot.add(door);

  const sprigCount = 64;
  const sprigGeometry = makeIvySprigGeometry(skin);
  const ivyMaterial: OakMaterial = sculptureMaterial({
    side: THREE.DoubleSide,
    roughness: 0.74,
    rimColor: palette.skyHorizon,
    rimStrength: 0.42,
    toonSoftness: 0.34,
    toonFloor: 0.3,
    name: 'ivyVineMaterial',
  });
  const vine = instancedMesh(sprigGeometry, ivyMaterial, sprigCount);
  vine.name = 'ivyVine';
  vine.castShadow = false;
  vine.receiveShadow = false;
  vine.frustumCulled = false;
  vine.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(sprigCount * 3), 3);
  frame.add(vine);

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
      const u = i / (sprigCount - 1);
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
      const r = hash1f(i, seed ^ 0x4b17) * Math.PI * 2 + i * 1.9;
      roll.setFromAxisAngle(t, r);
      q.premultiply(roll);
      baseQuat[i * 4] = q.x;
      baseQuat[i * 4 + 1] = q.y;
      baseQuat[i * 4 + 2] = q.z;
      baseQuat[i * 4 + 3] = q.w;
      phase[i] = hash1f(i, seed ^ 0x2c9d) * Math.PI * 2;
      sizes[i] = (u >= IVY_FRAME_SPAN ? 0.7 : 0.88) + hash1f(i, seed ^ 0x77e1) * 0.36;
      const young = u >= IVY_FRAME_SPAN ? 1.22 : 0.98;
      tint.setRGB(young, young, young);
      vine.setColorAt(i, tint);
    }
    vine.instanceColor.needsUpdate = true;
  }

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

      pivot.rotation.y = DOOR_SWING * smoothBand(SWING_BEAT[0], SWING_BEAT[1], p);

      const release = smoothBand(IVY_BEAT[0], IVY_BEAT[1], p);
      const tip = IVY_FRAME_SPAN + (1 - IVY_FRAME_SPAN + IVY_FEATHER) * (1 - release);

      for (let i = 0; i < sprigCount; i++) {
        const u = us[i]!;
        const grow = clamp01((tip - u) / IVY_FEATHER);
        if (grow <= 0) {
          matrix.makeScale(0, 0, 0);
          vine.setMatrixAt(i, matrix);
          continue;
        }
        const ph = phase[i]!;
        const rustle = Math.sin(time * 1.15 + ph) * 0.14 + Math.sin(time * 2.3 + ph * 1.7) * 0.05;
        axisVec.set(axis[i * 3]!, axis[i * 3 + 1]!, axis[i * 3 + 2]!);
        rollQuat.setFromAxisAngle(axisVec, rustle);
        quaternion.set(baseQuat[i * 4]!, baseQuat[i * 4 + 1]!, baseQuat[i * 4 + 2]!, baseQuat[i * 4 + 3]!);
        quaternion.premultiply(rollQuat);
        position.set(basePos[i * 3]!, basePos[i * 3 + 1]!, basePos[i * 3 + 2]!);
        const s = sizes[i]! * grow;
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

const GATE_RIDGE = DIVIDER_HEIGHT + DIVIDER_CAP + 0.42;

const JAMB_HEAD_Y = 0.66;
const JAMB_HEAD_D = 0.4;

function makeGateStoneGeometry(skin: GardenSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const jambW = 0.115;
  const jambD = DIVIDER_THICKNESS;
  const capTop = DIVIDER_HEIGHT + DIVIDER_CAP + 0.03;

  for (const sx of [-1, 1]) {
    const x = sx * (0.5 - jambW * 0.5);
    const jamb = new THREE.BoxGeometry(jambW, JAMB_HEAD_Y, jambD);
    jamb.translate(x, JAMB_HEAD_Y * 0.5, 0);
    parts.push({ geometry: jamb, color: skin.shrineStone });
    const head = new THREE.BoxGeometry(jambW, DIVIDER_HEIGHT - JAMB_HEAD_Y, JAMB_HEAD_D);
    head.translate(x, (JAMB_HEAD_Y + DIVIDER_HEIGHT) * 0.5, 0);
    parts.push({ geometry: head, color: skin.shrineStone });
    const band = new THREE.BoxGeometry(jambW + 0.05, 0.055, jambD + 0.03);
    band.translate(x, JAMB_HEAD_Y - 0.0275, 0);
    parts.push({ geometry: band, color: skin.dividerCap });
    const foot = new THREE.BoxGeometry(jambW + 0.08, 0.15, jambD + 0.08);
    foot.translate(x, 0.075, 0);
    parts.push({ geometry: foot, color: skin.dividerCap });
  }

  const cap = new THREE.BoxGeometry(1.12, DIVIDER_CAP + 0.03, JAMB_HEAD_D + 0.06);
  cap.translate(0, DIVIDER_HEIGHT + (DIVIDER_CAP + 0.03) * 0.5, 0);
  parts.push({ geometry: cap, color: skin.dividerCap });

  const header = new THREE.BoxGeometry(1 - jambW * 2 + 0.02, 0.12, 0.34);
  header.translate(0, DIVIDER_HEIGHT - 0.06, 0);
  parts.push({ geometry: header, color: skin.shrineDark });

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
  const beam = new THREE.BoxGeometry(1.06, 0.075, 0.09);
  beam.translate(0, capTop + postH * 0.72, 0);
  parts.push({ geometry: beam, color: skin.shrineDark });

  const slab = new THREE.BoxGeometry(0.98, 0.06, jambD * 0.86);
  slab.translate(0, 0.028, 0);
  parts.push({ geometry: slab, color: skin.threshold });

  for (let i = 0; i < 2; i++) {
    const s = new THREE.IcosahedronGeometry(0.1, 1);
    s.scale(1, 0.7, 1.3);
    s.translate(i === 0 ? -0.36 : 0.36, 0.05, jambD * 0.5 - 0.06);
    parts.push({ geometry: s, color: skin.shrineDark });
  }

  return mergeColoredParts(parts, 'ivyGateStone');
}

function makeDoorGeometry(skin: GardenSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const boards = 5;
  const h = DOOR_HEIGHT - 0.06;
  const boardW = DOOR_WIDTH / boards;

  for (let i = 0; i < boards; i++) {
    const board = new THREE.BoxGeometry(boardW - 0.012, h, DOOR_THICKNESS);
    const lift = ((i * 5) % 3) * 0.012;
    board.translate(-(i + 0.5) * boardW, h * 0.5 - lift, 0);
    parts.push({ geometry: board, color: i % 2 === 0 ? skin.wood : skin.woodDark });
  }

  for (const y of [h * 0.22, h * 0.78]) {
    const strap = new THREE.BoxGeometry(DOOR_WIDTH - 0.03, 0.055, DOOR_THICKNESS + 0.022);
    strap.translate(-DOOR_WIDTH * 0.5, y, 0);
    parts.push({ geometry: strap, color: skin.iron });
  }

  const ring = new THREE.TorusGeometry(0.062, 0.016, 6, 14);
  ring.rotateY(Math.PI / 2);
  ring.translate(-DOOR_WIDTH + 0.15, h * 0.52, -DOOR_THICKNESS * 0.5 - 0.02);
  parts.push({ geometry: ring, color: skin.iron });

  return mergeColoredParts(parts, 'ivyGateDoor');
}

function makeIvySprigGeometry(skin: GardenSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const stemLength = 0.13;
  const stem = new THREE.BoxGeometry(0.023, stemLength, 0.023);
  stem.translate(0, stemLength * 0.5, 0);
  parts.push({ geometry: stem, color: skin.ivyDark });

  const leaves: readonly { at: number; side: number }[] = [{ at: 0.3, side: -1 }, { at: 0.74, side: 1 }];
  for (const spec of leaves) {
    const leaf = ivyLeafGeometry();
    leaf.rotateZ(spec.side * 0.95);
    leaf.rotateX(-0.42);
    leaf.translate(spec.side * 0.012, stemLength * spec.at, 0.012);
    parts.push({ geometry: leaf, color: spec.side < 0 ? skin.ivy : skin.ivyDark });
  }

  return mergeColoredParts(parts, 'ivySprig');
}

const SHRINE_CROWN = 0.66;
const EMBER_HDR = 1.3;

interface ShrineOptions {
  palette: Palette;
  skin: GardenSkin;
  name?: string;
}

export interface ShrineHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly crownHeight: number;
  setState(grow01: number, glow01: number): void;
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildShrine(opts: ShrineOptions): ShrineHandle {
  const { palette, skin } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'goalShrine';

  const root = new THREE.Object3D();
  root.name = 'shrineRoot';
  group.add(root);

  const beacon = new GoalBeacon({
    color: skin.amber,
    height: 0.85,
    radius: 0.1,
    intensity: 0.34,
    name: 'shrineGlow',
  });
  root.add(beacon.group);

  const body = new THREE.Object3D();
  body.name = 'shrineBody';
  root.add(body);

  const stoneGeometry = makeShrineGeometry(skin);
  const stoneMaterial: OakMaterial = sculptureMaterial({
    roughness: 0.9,
    rimColor: palette.skyHorizon,
    rimStrength: 0.2,
    toonSoftness: 0.3,
    toonFloor: 0.24,
    name: 'shrineStoneMaterial',
  });
  const stone = new THREE.Mesh(stoneGeometry, stoneMaterial);
  stone.name = 'shrineStone';
  stone.castShadow = true;
  stone.receiveShadow = true;
  body.add(stone);

  const emberGeometry = new THREE.IcosahedronGeometry(0.075, 1);
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
      body.scale.setScalar(grow);
      beacon.group.scale.setScalar(grow);
      ember.scale.setScalar(0.72 + 0.4 * glow);
      setEmissiveIntensity(emberMaterial, EMBER_HDR * (0.55 + 0.7 * glow) * (0.25 + 0.75 * grow));
    },

    flare(strength = 1) {
      beacon.flare(strength);
    },

    setTime(t) {
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

function makeShrineGeometry(skin: GardenSkin): THREE.BufferGeometry {
  const radial = 12;
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

  const posts = 6;
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

  const finial = new THREE.IcosahedronGeometry(0.048, 1);
  finial.translate(0, 1.02, 0);
  parts.push({ geometry: finial, color: skin.shrineDark });

  return mergeColoredParts(parts, 'goalShrine');
}

export const SEED_HOVER = 0.34;

interface SeedBedOptions {
  palette: Palette;
  skin: GardenSkin;
  name?: string;
}

export interface SeedBedHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  token: KeyToken;
  cradle: THREE.Object3D;
  setState(grow01: number, present01: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildSeedBed(opts: SeedBedOptions): SeedBedHandle {
  const { palette, skin } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'seedBed';

  const root = new THREE.Object3D();
  root.name = 'seedBedRoot';
  group.add(root);

  const bed = new THREE.Object3D();
  bed.name = 'seedBedMoss';
  root.add(bed);

  const bedGeometry = makeSeedBedGeometry(skin);
  const bedMaterial: OakMaterial = sculptureMaterial({
    roughness: 0.92,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSoftness: 0.32,
    toonFloor: 0.28,
    name: 'seedBedMaterial',
  });
  const mesh = new THREE.Mesh(bedGeometry, bedMaterial);
  mesh.name = 'seedBedStone';
  mesh.castShadow = false;
  mesh.receiveShadow = true;
  bed.add(mesh);

  const poolGeometry = new THREE.CircleGeometry(0.2, 20);
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

  const cradle = new THREE.Object3D();
  cradle.name = 'seedCradle';
  cradle.position.set(0, SEED_HOVER, 0);
  root.add(cradle);

  const token = new KeyToken({
    color: skin.amber,
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

function makeSeedBedGeometry(skin: GardenSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];

  const cushion = new THREE.IcosahedronGeometry(0.19, 1);
  cushion.scale(1, 0.36, 1);
  cushion.translate(0, 0.045, 0);
  parts.push({ geometry: cushion, color: skin.ivy });

  const stones = 7;
  for (let i = 0; i < stones; i++) {
    const a = (i / stones) * Math.PI * 2;
    const s = new THREE.IcosahedronGeometry(0.062, 1);
    s.scale(1.25, 0.72, 0.9);
    s.translate(0.235, 0.035, 0);
    s.rotateY(a + i * 0.31);
    parts.push({ geometry: s, color: i % 3 === 0 ? skin.shrineDark : skin.shrineStone });
  }

  return mergeColoredParts(parts, 'seedBed');
}
