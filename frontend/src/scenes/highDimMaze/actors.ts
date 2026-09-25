import * as THREE from 'three/webgpu';
import { GoalBeacon } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  sculptureMaterial,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { clamp01 } from '@kit/mix';
import { gradedGeometry, mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { Palette } from '@kit/tokens';
import { makeWater, type WaterHandle } from '@kit/water';
import type { CanyonSkin } from './world';

const BASIN_R = 0.44;
const BASIN_RIM_Y = 0.13;
const POOL_R = 0.315;
const POOL_Y = 0.052;
const SHAFT_TOP = 1.62;
const SHAFT_HDR = 0.62;
const GROW_CEILING = 1.2;

type Vertex = readonly [number, number, number, number];

export interface OasisHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly crownHeight: number;
  setState(grow01: number, near01: number): void;
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildOasis(palette: Palette, skin: CanyonSkin): OasisHandle {
  const group = new THREE.Group();
  group.name = 'oasisAlcove';
  const root = new THREE.Object3D();
  root.name = 'oasisRoot';
  group.add(root);
  const alcove = new THREE.Object3D();
  alcove.name = 'oasisGrow';
  root.add(alcove);

  const beacon = new GoalBeacon({ color: skin.shaft, height: 0.46, radius: 0.08, intensity: 0.26, name: 'oasisGlow' });
  alcove.add(beacon.group);

  const basinParts: ColoredPart[] = [];
  const stones = 11;
  for (let i = 0; i < stones; i++) {
    const a = (i / stones) * Math.PI * 2;
    const wobble = 0.86 + 0.3 * ((i * 5) % 7) / 7;
    const stone = new THREE.BoxGeometry(0.2 * wobble, BASIN_RIM_Y * wobble, 0.13);
    stone.rotateY(-a);
    stone.translate(Math.cos(a) * BASIN_R * 0.92, BASIN_RIM_Y * 0.5 * wobble, Math.sin(a) * BASIN_R * 0.92);
    basinParts.push({ geometry: stone, color: i % 3 === 0 ? (skin.strata[1] ?? skin.stone) : (skin.strata[2] ?? skin.stone) });
  }
  const bed = new THREE.CylinderGeometry(POOL_R * 1.08, POOL_R * 0.8, 0.09, 16);
  bed.translate(0, 0.02, 0);
  basinParts.push({ geometry: bed, color: skin.waterDeep });
  const basinGeometry = mergeColoredParts(basinParts, 'oasisBasin');
  const basinMaterial: OakMaterial = sculptureMaterial({
    color: palette.sun, roughness: 0.7, rimColor: skin.rim, rimStrength: 0.45, toonSoftness: 0.3, toonFloor: 0.28, name: 'oasisBasinMaterial',
  });
  const basin = new THREE.Mesh(basinGeometry, basinMaterial);
  basin.name = 'oasisBasin';
  basin.castShadow = true;
  basin.receiveShadow = true;
  alcove.add(basin);

  const water: WaterHandle = makeWater({
    size: POOL_R * 2, palette, segments: 18,
    shallow: skin.water, deep: skin.waterDeep, sky: palette.skyHorizon, caustic: palette.sun,
    amplitude: 0.012, waveScale: 9, speed: 0.7, fresnel: 0.6, caustics: 0.5, depthFalloff: 1.2, opacity: 0.94,
    level: POOL_Y, name: 'oasisWater',
  });
  alcove.add(water.mesh);

  const reedGeometry = makeReedGeometry();
  const reedMaterial: OakMaterial = sculptureMaterial({
    color: skin.reed, side: THREE.DoubleSide, roughness: 0.78, rimColor: skin.rim, rimStrength: 0.4,
    toonSteps: 2, toonSoftness: 0.36, toonFloor: 0.3, wind: { amplitude: 0.05, frequency: 1.7, height: 0.34 },
    name: 'oasisReedMaterial',
  });
  const reedCount = 10;
  const reeds = instancedMesh(reedGeometry, reedMaterial, reedCount);
  reeds.name = 'oasisReeds';
  reeds.frustumCulled = false;
  reeds.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(reedCount * 3), 3);
  const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), e = new THREE.Euler(), s = new THREE.Vector3(), c = new THREE.Color();
  for (let i = 0; i < reedCount; i++) {
    const a = -2.7 + (i / (reedCount - 1)) * 3.6 + ((i * 3) % 5) * 0.06;
    const r = BASIN_R * (0.82 + 0.12 * (((i * 7) % 4) / 3));
    p.set(Math.cos(a) * r, 0.02, Math.sin(a) * r);
    q.setFromEuler(e.set(0.1 * Math.sin(i * 2.1), a + 1.5, 0.14 * Math.cos(i * 1.3)));
    const k = 0.95 + 0.6 * (((i * 5) % 6) / 5);
    s.set(k, k * (0.85 + 0.35 * (((i * 11) % 7) / 6)), k);
    reeds.setMatrixAt(i, m.compose(p, q, s));
    const tone = 0.82 + 0.3 * (((i * 13) % 5) / 4);
    reeds.setColorAt(i, c.setRGB(tone, tone, tone * 0.94));
  }
  reeds.instanceMatrix.needsUpdate = true;
  reeds.instanceColor.needsUpdate = true;
  alcove.add(reeds);

  const shaftGeometry = new THREE.CylinderGeometry(POOL_R * 2.5, POOL_R * 0.6, SHAFT_TOP, 18, 1, true);
  shaftGeometry.translate(0, SHAFT_TOP * 0.5 + POOL_Y, 0);
  const shaftMaterial: EmissiveMaterial = emissiveMaterial(skin.shaft, SHAFT_HDR, {
    transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, name: 'oasisShaft',
  });
  const shaft = new THREE.Mesh(shaftGeometry, shaftMaterial);
  shaft.name = 'oasisShaft';
  shaft.renderOrder = 2;
  alcove.add(shaft);

  return {
    group,
    root,
    crownHeight: 0.5,
    setState(grow01, near01) {
      const grow = grow01 > 0 ? (grow01 < GROW_CEILING ? grow01 : GROW_CEILING) : 0;
      const near = clamp01(near01);
      const lit = clamp01(grow);
      alcove.scale.setScalar(grow);
      setEmissiveIntensity(shaftMaterial, SHAFT_HDR * (0.55 + 0.7 * near) * (0.2 + 0.8 * lit));
      shaftMaterial.opacity = (0.115 + 0.09 * near) * lit;
    },
    flare(strength = 1) {
      beacon.flare(strength);
    },
    setTime(t) {
      water.setTime(t);
      oakUniforms(reedMaterial)?.setTime(t);
      const breath = 1 + 0.07 * Math.sin(t * 0.9) + 0.03 * Math.sin(t * 2.3);
      shaft.scale.set(breath, 1, breath);
      shaft.rotation.y = t * 0.11;
    },
    update(dt, time) {
      beacon.update(dt, time);
    },
    dispose() {
      beacon.dispose();
      water.dispose();
      reeds.dispose();
      reedGeometry.dispose();
      reedMaterial.dispose();
      basinGeometry.dispose();
      basinMaterial.dispose();
      shaftGeometry.dispose();
      shaftMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function makeReedGeometry(): THREE.BufferGeometry {
  const vertices: Vertex[] = [];
  for (let b = 0; b < 3; b++) {
    const a = (b / 3) * Math.PI * 1.1 - 0.4;
    const h = 0.3 + 0.14 * ((b * 5) % 3) / 2;
    const lean = 0.1 + 0.07 * ((b * 7) % 4) / 3;
    const dx = Math.cos(a) * 0.028;
    const dz = Math.sin(a) * 0.028;
    const tipX = Math.cos(a + 0.5) * lean;
    const tipZ = Math.sin(a + 0.5) * lean;
    const mid: Vertex = [tipX * 0.45 + dx * 0.5, h * 0.55, tipZ * 0.45 + dz * 0.5, 0.85];
    vertices.push([-dx, 0, -dz, 0.45], [dx, 0, dz, 0.45], mid, [-dx, 0, -dz, 0.45], mid, [tipX, h, tipZ, 1.2]);
  }
  return gradedGeometry('oasisReed', vertices);
}

export interface ScarfHandle {
  group: THREE.Group;
  setState(yaw: number, slack: number, speed: number, time: number): void;
  dispose(): void;
}

const SCARF_SEGMENTS = 7;
const SCARF_SEGMENT = 0.2;

export function buildScarf(tint: THREE.ColorRepresentation, height: number): ScarfHandle {
  const group = new THREE.Group();
  group.name = 'canyonScarf';
  group.position.y = height;
  const w0 = 0.105, w1 = 0.078, l = SCARF_SEGMENT * 0.5;
  const geometry = gradedGeometry('wispScarfPanel', [
    [-w0, 0, l, 1], [w0, 0, l, 1], [w1, 0, -l, 0.72],
    [-w0, 0, l, 1], [w1, 0, -l, 0.72], [-w1, 0, -l, 0.72],
  ]);
  const material: OakMaterial = sculptureMaterial({
    color: tint, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthWrite: false, roughness: 0.6,
    rimColor: tint, rimStrength: 0.55, toonSteps: 2, toonSoftness: 0.4, toonFloor: 0.42, name: 'wispScarfMaterial',
  });
  const mesh = instancedMesh(geometry, material, SCARF_SEGMENTS);
  mesh.name = 'wispScarfPanels';
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(SCARF_SEGMENTS * 3), 3);
  const c = new THREE.Color();
  for (let i = 0; i < SCARF_SEGMENTS; i++) mesh.setColorAt(i, c.setScalar(1 - (i / SCARF_SEGMENTS) * 0.45));
  mesh.instanceColor.needsUpdate = true;
  group.add(mesh);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);

  return {
    group,
    setState(yaw, slack, speed, time) {
      group.rotation.y = yaw;
      let x = 0, y = 0, z = 0, heading = 0;
      const lift = 0.42 * (1 - slack) + 0.1;
      const wag = 0.45 + 0.6 * speed;
      for (let i = 0; i < SCARF_SEGMENTS; i++) {
        const t = (i + 1) / SCARF_SEGMENTS;
        const phase = time * (1.9 + 1.6 * speed) - i * 0.52;
        const swing = Math.sin(phase) * (0.1 + 0.26 * t) * wag * (0.35 + 0.65 * (1 - slack));
        heading += swing * 0.3;
        const pitch = -(0.12 + 0.42 * t) * slack + lift * (0.5 - 0.5 * t) + 0.07 * Math.cos(phase * 0.8);
        const dx = Math.sin(heading) * SCARF_SEGMENT;
        const dz = -Math.cos(heading) * SCARF_SEGMENT;
        const dy = Math.sin(pitch) * SCARF_SEGMENT * 0.5;
        position.set(x + dx * 0.5, y + dy * 0.5, z + dz * 0.5);
        quaternion.setFromEuler(euler.set(pitch, heading, swing * 0.35));
        scale.set(1 - t * 0.62, 1, 1);
        mesh.setMatrixAt(i, matrix.compose(position, quaternion, scale));
        x += dx;
        y += dy;
        z += dz;
      }
      mesh.instanceMatrix.needsUpdate = true;
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

export interface GustHandle {
  group: THREE.Group;
  setLines(from: THREE.Vector3, to: THREE.Vector3, y: number, progress: number, amount: number): void;
  setWake(from: THREE.Vector3, to: THREE.Vector3, amount: number): void;
  setSettle(x: number, z: number, amount: number, time: number): void;
  clear(): void;
  dispose(): void;
}

const GUST_LINES = 6;

export function buildGust(skin: CanyonSkin, tint: THREE.ColorRepresentation): GustHandle {
  const group = new THREE.Group();
  group.name = 'canyonGust';

  const lineGeometry = new THREE.PlaneGeometry(1, 0.022);
  lineGeometry.rotateX(-Math.PI / 2);
  const lineMaterial: EmissiveMaterial = emissiveMaterial(tint, 1.1, {
    transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, name: 'wispSpeedLines',
  });
  const lineMesh = instancedMesh(lineGeometry, lineMaterial, GUST_LINES);
  lineMesh.name = 'wispSpeedLines';
  lineMesh.frustumCulled = false;
  lineMesh.visible = false;
  lineMesh.renderOrder = 3;
  group.add(lineMesh);

  const wakeGeometry = new THREE.PlaneGeometry(1, 1);
  wakeGeometry.rotateX(-Math.PI / 2);
  const wakeMaterial: OakMaterial = oakMaterial({
    color: skin.sandSun, transparent: true, opacity: 0, depthWrite: false, roughness: 1, toonSteps: 2, toonSoftness: 0.5, toonFloor: 0.6, name: 'wispSandWake',
  });
  const wake = new THREE.Mesh(wakeGeometry, wakeMaterial);
  wake.name = 'wispSandWake';
  wake.visible = false;
  wake.renderOrder = 1;
  group.add(wake);

  const ringGeometry = new THREE.RingGeometry(0.2, 0.36, 26);
  ringGeometry.rotateX(-Math.PI / 2);
  const ringMaterial: EmissiveMaterial = emissiveMaterial(skin.dust, 0.55, {
    transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, name: 'wispSettleRing',
  });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.name = 'wispSettleRing';
  ring.visible = false;
  ring.renderOrder = 1;
  group.add(ring);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);

  return {
    group,
    setLines(from, to, y, progress, amount) {
      lineMesh.visible = amount > 0.001;
      if (!lineMesh.visible) return;
      const dx = to.x - from.x;
      const dz = to.z - from.z;
      const span = Math.hypot(dx, dz);
      const yaw = Math.atan2(dx, dz);
      const headX = from.x + dx * progress;
      const headZ = from.z + dz * progress;
      quaternion.setFromEuler(euler.set(0, yaw, 0));
      for (let i = 0; i < GUST_LINES; i++) {
        const t = (i + 0.5) / GUST_LINES;
        const lag = 0.12 + 0.42 * t;
        const off = (i % 2 === 0 ? 1 : -1) * (0.05 + 0.09 * t);
        position.set(headX - dx * lag * 0.5 + Math.cos(yaw) * off, y + (i % 3) * 0.045 - 0.03, headZ - dz * lag * 0.5 - Math.sin(yaw) * off);
        scale.set(Math.max(0.12, span * (0.3 + 0.4 * t)), 1, 1 - t * 0.4);
        lineMesh.setMatrixAt(i, matrix.compose(position, quaternion, scale));
      }
      lineMesh.instanceMatrix.needsUpdate = true;
      setEmissiveIntensity(lineMaterial, 1.1 * amount);
      lineMaterial.opacity = 0.55 * amount;
    },
    setWake(from, to, amount) {
      wake.visible = amount > 0.001;
      if (!wake.visible) return;
      const dx = to.x - from.x;
      const dz = to.z - from.z;
      wake.position.set((from.x + to.x) * 0.5, 0.014, (from.z + to.z) * 0.5);
      wake.rotation.y = Math.atan2(dx, dz);
      wake.scale.set(0.44 + 0.16 * amount, 1, Math.max(0.35, Math.hypot(dx, dz)) + 0.5);
      wakeMaterial.opacity = 0.4 * amount;
    },
    setSettle(x, z, amount, time) {
      ring.visible = amount > 0.001;
      if (!ring.visible) return;
      const breath = 1 + 0.09 * Math.sin(time * 1.6);
      ring.position.set(x, 0.016, z);
      ring.scale.set(breath, 1, breath);
      ring.rotation.y = time * 0.25;
      setEmissiveIntensity(ringMaterial, 0.55 * amount);
      ringMaterial.opacity = 0.42 * amount;
    },
    clear() {
      lineMesh.visible = false;
      wake.visible = false;
      ring.visible = false;
    },
    dispose() {
      lineMesh.dispose();
      lineGeometry.dispose();
      lineMaterial.dispose();
      wakeGeometry.dispose();
      wakeMaterial.dispose();
      ringGeometry.dispose();
      ringMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
