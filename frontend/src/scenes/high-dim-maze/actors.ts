
import { GoalBeacon } from '@kit/grid';
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { clamp01 } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { Palette } from '@kit/tokens';
import { makeWater, type WaterHandle } from '@kit/water';
import * as THREE from 'three';
import type { CanyonSkin } from './world';


const BASIN_R = 0.44;
const BASIN_RIM_Y = 0.13;
const POOL_R = 0.315;
const POOL_Y = 0.052;
const SHAFT_TOP = 1.62;
const SHAFT_HDR = 0.62;
const GROW_CEILING = 1.2;

export interface OasisOptions {
  palette: Palette;
  skin: CanyonSkin;
  name?: string;
}

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

export function buildOasis(opts: OasisOptions): OasisHandle {
  const { palette, skin } = opts;
  const detail = 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'oasisAlcove';

  const root = new THREE.Object3D();
  root.name = 'oasisRoot';
  group.add(root);

  const alcove = new THREE.Object3D();
  alcove.name = 'oasisGrow';
  root.add(alcove);

  const beacon = new GoalBeacon({
    color: skin.shaft,
    height: 0.46,
    radius: 0.08,
    intensity: 0.26,
    name: 'oasisGlow',
  });
  alcove.add(beacon.group);

  const basinParts: ColoredPart[] = [];
  {
    const stones = 11;
    for (let i = 0; i < stones; i++) {
      const a = (i / stones) * Math.PI * 2;
      const wobble = 0.86 + 0.3 * ((i * 5) % 7) / 7;
      const stone = new THREE.BoxGeometry(0.2 * wobble, BASIN_RIM_Y * wobble, 0.13);
      stone.rotateY(-a);
      stone.translate(Math.cos(a) * BASIN_R * 0.92, BASIN_RIM_Y * 0.5 * wobble, Math.sin(a) * BASIN_R * 0.92);
      basinParts.push({
        geometry: stone,
        color: i % 3 === 0 ? (skin.strata[1] ?? skin.stone) : (skin.strata[2] ?? skin.stone),
      });
    }
    const bed = new THREE.CylinderGeometry(POOL_R * 1.08, POOL_R * 0.8, 0.09, Math.max(6, Math.round(16 * detail)));
    bed.translate(0, 0.02, 0);
    basinParts.push({ geometry: bed, color: skin.waterDeep });
  }
  const basinGeometry = mergeColoredParts(basinParts, 'oasisBasin');
  const basinMaterial: OakMaterial = oakMaterial({
    color: palette.sun,
    vertexColors: true,
    flatShading: true,
    roughness: 0.7,
    rimColor: skin.rim,
    rimStrength: 0.45,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.28,
    name: 'oasisBasinMaterial',
  });
  const basin = new THREE.Mesh(basinGeometry, basinMaterial);
  basin.name = 'oasisBasin';
  basin.castShadow = true;
  basin.receiveShadow = true;
  alcove.add(basin);

  const water: WaterHandle = makeWater({
    size: POOL_R * 2,
    palette,
    segments: 18,
    shallow: skin.water,
    deep: skin.waterDeep,
    sky: palette.skyHorizon,
    caustic: palette.sun,
    amplitude: 0.012,
    waveScale: 9,
    speed: 0.7,
    fresnel: 0.6,
    caustics: 0.5,
    depthFalloff: 1.2,
    opacity: 0.94,
    level: POOL_Y,
    name: 'oasisWater',
  });
  water.mesh.name = 'oasisWater';
  alcove.add(water.mesh);

  const reedGeometry = makeReedGeometry();
  const reedMaterial: OakMaterial = oakMaterial({
    color: skin.reed,
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.78,
    rimColor: skin.rim,
    rimStrength: 0.4,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.3,
    wind: { amplitude: 0.05, frequency: 1.7, height: 0.34 },
    name: 'oasisReedMaterial',
  });
  const reedCount = 10;
  const reeds = new THREE.InstancedMesh(reedGeometry, reedMaterial, reedCount);
  reeds.name = 'oasisReeds';
  reeds.castShadow = false;
  reeds.frustumCulled = false;
  reeds.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(reedCount * 3), 3);
  {
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    const s = new THREE.Vector3();
    const c = new THREE.Color();
    for (let i = 0; i < reedCount; i++) {
      const a = -2.7 + (i / Math.max(1, reedCount - 1)) * 3.6 + ((i * 3) % 5) * 0.06;
      const r = BASIN_R * (0.82 + 0.12 * (((i * 7) % 4) / 3));
      p.set(Math.cos(a) * r, 0.02, Math.sin(a) * r);
      e.set(0.1 * Math.sin(i * 2.1), a + 1.5, 0.14 * Math.cos(i * 1.3));
      q.setFromEuler(e);
      const k = 0.95 + 0.6 * (((i * 5) % 6) / 5);
      s.set(k, k * (0.85 + 0.35 * (((i * 11) % 7) / 6)), k);
      m.compose(p, q, s);
      reeds.setMatrixAt(i, m);
      const tone = 0.82 + 0.3 * (((i * 13) % 5) / 4);
      c.setRGB(tone, tone, tone * 0.94);
      reeds.setColorAt(i, c);
    }
    reeds.instanceMatrix.needsUpdate = true;
    if (reeds.instanceColor) reeds.instanceColor.needsUpdate = true;
  }
  alcove.add(reeds);

  const shaftGeometry = new THREE.CylinderGeometry(
    POOL_R * 2.5,
    POOL_R * 0.6,
    SHAFT_TOP,
    Math.max(6, Math.round(18 * detail)),
    1,
    true,
  );
  shaftGeometry.translate(0, SHAFT_TOP * 0.5 + POOL_Y, 0);
  const shaftMaterial: EmissiveMaterial = emissiveMaterial(skin.shaft, SHAFT_HDR, {
    transparent: true,
    opacity: 0.16,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    name: 'oasisShaft',
  });
  const shaft = new THREE.Mesh(shaftGeometry, shaftMaterial);
  shaft.name = 'oasisShaft';
  shaft.renderOrder = 2;
  alcove.add(shaft);

  const crownHeight = 0.5;

  return {
    group,
    root,
    crownHeight,

    setState(grow01, near01) {
      const grow = grow01 > 0 ? (grow01 < GROW_CEILING ? grow01 : GROW_CEILING) : 0;
      const near = clamp01(near01);
      alcove.scale.setScalar(grow);
      const lit = clamp01(grow);
      setEmissiveIntensity(shaftMaterial, SHAFT_HDR * (0.55 + 0.7 * near) * (0.2 + 0.8 * lit));
      shaftMaterial.opacity = (0.115 + 0.09 * near) * lit;
    },

    flare(strength = 1) {
      beacon.flare(strength);
    },

    setTime(t) {
      water.setTime(t);
      oakUniforms(reedMaterial)?.setTime(t);
      oakUniforms(basinMaterial)?.setTime(t);
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
  const blades = 3;
  const positions = new Float32Array(blades * 2 * 3 * 3);
  const colors = new Float32Array(blades * 2 * 3 * 3);
  let o = 0;
  for (let b = 0; b < blades; b++) {
    const a = (b / blades) * Math.PI * 1.1 - 0.4;
    const h = 0.3 + 0.14 * ((b * 5) % 3) / 2;
    const wide = 0.028;
    const lean = 0.1 + 0.07 * ((b * 7) % 4) / 3;
    const dx = Math.cos(a) * wide;
    const dz = Math.sin(a) * wide;
    const tipX = Math.cos(a + 0.5) * lean;
    const tipZ = Math.sin(a + 0.5) * lean;
    const midX = tipX * 0.45;
    const midZ = tipZ * 0.45;
    const pts: readonly (readonly [number, number, number, number])[] = [
      [-dx, 0, -dz, 0.45],
      [dx, 0, dz, 0.45],
      [midX + dx * 0.5, h * 0.55, midZ + dz * 0.5, 0.85],
      [-dx, 0, -dz, 0.45],
      [midX + dx * 0.5, h * 0.55, midZ + dz * 0.5, 0.85],
      [tipX, h, tipZ, 1.2],
    ];
    for (const [x, y, z, k] of pts) {
      positions[o * 3] = x;
      positions[o * 3 + 1] = y;
      positions[o * 3 + 2] = z;
      colors[o * 3] = k;
      colors[o * 3 + 1] = k;
      colors[o * 3 + 2] = k;
      o++;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'oasisReed';
  return g;
}


export interface ScarfOptions {
  color: THREE.ColorRepresentation;
  height: number;
  name?: string;
}

export interface ScarfHandle {
  group: THREE.Group;
  segments: number;
  setState(yaw: number, slack: number, speed: number, time: number): void;
  dispose(): void;
}

export function buildScarf(opts: ScarfOptions): ScarfHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'wispScarf';
  const segments = 7;
  group.position.y = opts.height;

  const SEG_LEN = 0.2;
  const geometry = makeScarfPanelGeometry(SEG_LEN);
  const material: OakMaterial = oakMaterial({
    color: opts.color,
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    roughness: 0.6,
    rimColor: opts.color,
    rimStrength: 0.55,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.42,
    name: 'wispScarfMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, segments);
  mesh.name = 'wispScarfPanels';
  mesh.castShadow = false;
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(segments * 3), 3);
  {
    const c = new THREE.Color();
    for (let i = 0; i < segments; i++) {
      const k = 1 - (i / segments) * 0.45;
      c.setRGB(k, k, k);
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
  group.add(mesh);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);

  return {
    group,
    segments,
    setState(yaw, slack, speed, time) {
      group.rotation.y = yaw;
      let x = 0;
      let y = 0;
      let z = 0;
      let heading = 0;
      const lift = 0.42 * (1 - slack) + 0.1;
      const wag = 0.45 + 0.6 * speed;
      for (let i = 0; i < segments; i++) {
        const t = (i + 1) / segments;
        const phase = time * (1.9 + 1.6 * speed) - i * 0.52;
        const swing = Math.sin(phase) * (0.1 + 0.26 * t) * wag * (0.35 + 0.65 * (1 - slack));
        const droop = -(0.12 + 0.42 * t) * slack;
        heading += swing * 0.3;
        const pitch = droop + lift * (0.5 - 0.5 * t) + 0.07 * Math.cos(phase * 0.8);
        const dx = Math.sin(heading) * SEG_LEN;
        const dz = -Math.cos(heading) * SEG_LEN;
        const dy = Math.sin(pitch) * SEG_LEN * 0.5;
        position.set(x + dx * 0.5, y + dy * 0.5, z + dz * 0.5);
        euler.set(pitch, heading, swing * 0.35);
        quaternion.setFromEuler(euler);
        const taper = 1 - t * 0.62;
        scale.set(taper, 1, 1);
        matrix.compose(position, quaternion, scale);
        mesh.setMatrixAt(i, matrix);
        x += dx;
        y += dy;
        z += dz;
      }
      mesh.instanceMatrix.needsUpdate = true;
      oakUniforms(material)?.setTime(time);
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

function makeScarfPanelGeometry(length: number): THREE.BufferGeometry {
  const w0 = 0.105;
  const w1 = 0.078;
  const verts: readonly (readonly [number, number, number, number])[] = [
    [-w0, 0, length * 0.5, 1],
    [w0, 0, length * 0.5, 1],
    [w1, 0, -length * 0.5, 0.72],
    [-w0, 0, length * 0.5, 1],
    [w1, 0, -length * 0.5, 0.72],
    [-w1, 0, -length * 0.5, 0.72],
  ];
  const positions = new Float32Array(verts.length * 3);
  const colors = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    const v = verts[i];
    if (!v) continue;
    positions[i * 3] = v[0];
    positions[i * 3 + 1] = v[1];
    positions[i * 3 + 2] = v[2];
    colors[i * 3] = v[3];
    colors[i * 3 + 1] = v[3];
    colors[i * 3 + 2] = v[3];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'wispScarfPanel';
  return g;
}


export interface GustOptions {
  skin: CanyonSkin;
  color: THREE.ColorRepresentation;
  name?: string;
}

export interface GustHandle {
  group: THREE.Group;
  setLines(
    fromX: number,
    fromZ: number,
    toX: number,
    toZ: number,
    y: number,
    progress: number,
    amount: number,
  ): void;
  setWake(fromX: number, fromZ: number, toX: number, toZ: number, amount: number): void;
  setSettle(x: number, z: number, amount: number, time: number): void;
  clear(): void;
  dispose(): void;
}

export function buildGust(opts: GustOptions): GustHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'wispGust';
  const lines = 6;

  const lineGeometry = new THREE.PlaneGeometry(1, 0.022);
  lineGeometry.rotateX(-Math.PI / 2);
  const lineMaterial: EmissiveMaterial = emissiveMaterial(opts.color, 1.1, {
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    name: 'wispSpeedLines',
  });
  const lineMesh = new THREE.InstancedMesh(lineGeometry, lineMaterial, Math.max(1, lines));
  lineMesh.name = 'wispSpeedLines';
  lineMesh.frustumCulled = false;
  lineMesh.visible = false;
  lineMesh.count = lines;
  lineMesh.renderOrder = 3;
  group.add(lineMesh);

  const wakeGeometry = new THREE.PlaneGeometry(1, 1, 1, 1);
  wakeGeometry.rotateX(-Math.PI / 2);
  const wakeMaterial: OakMaterial = oakMaterial({
    color: opts.skin.sandSun,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    roughness: 1,
    toonSteps: 2,
    toonSoftness: 0.5,
    toonFloor: 0.6,
    name: 'wispSandWake',
  });
  const wake = new THREE.Mesh(wakeGeometry, wakeMaterial);
  wake.name = 'wispSandWake';
  wake.visible = false;
  wake.renderOrder = 1;
  group.add(wake);

  const ringGeometry = new THREE.RingGeometry(0.2, 0.36, 26);
  ringGeometry.rotateX(-Math.PI / 2);
  const ringMaterial: EmissiveMaterial = emissiveMaterial(opts.skin.dust, 0.55, {
    transparent: true,
    opacity: 0,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    name: 'wispSettleRing',
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
    setLines(fromX, fromZ, toX, toZ, y, progress, amount) {
      if (amount <= 0.001) {
        lineMesh.visible = false;
        return;
      }
      lineMesh.visible = true;
      const dx = toX - fromX;
      const dz = toZ - fromZ;
      const span = Math.hypot(dx, dz);
      const yaw = Math.atan2(dx, dz);
      const headX = fromX + dx * progress;
      const headZ = fromZ + dz * progress;
      for (let i = 0; i < lines; i++) {
        const t = (i + 0.5) / lines;
        const lag = 0.12 + 0.42 * t;
        const cx = headX - dx * lag * 0.5;
        const cz = headZ - dz * lag * 0.5;
        const off = ((i % 2 === 0 ? 1 : -1) * (0.05 + 0.09 * t));
        position.set(cx + Math.cos(yaw) * off, y + (i % 3) * 0.045 - 0.03, cz - Math.sin(yaw) * off);
        euler.set(0, yaw, 0);
        quaternion.setFromEuler(euler);
        scale.set(Math.max(0.12, span * (0.3 + 0.4 * t)), 1, 1 - t * 0.4);
        matrix.compose(position, quaternion, scale);
        lineMesh.setMatrixAt(i, matrix);
      }
      lineMesh.instanceMatrix.needsUpdate = true;
      setEmissiveIntensity(lineMaterial, 1.1 * amount);
      lineMaterial.opacity = 0.55 * amount;
    },
    setWake(fromX, fromZ, toX, toZ, amount) {
      if (amount <= 0.001) {
        wake.visible = false;
        return;
      }
      wake.visible = true;
      const dx = toX - fromX;
      const dz = toZ - fromZ;
      const span = Math.max(0.35, Math.hypot(dx, dz));
      wake.position.set((fromX + toX) * 0.5, 0.014, (fromZ + toZ) * 0.5);
      wake.rotation.y = Math.atan2(dx, dz);
      wake.scale.set(0.44 + 0.16 * amount, 1, span + 0.5);
      wakeMaterial.opacity = 0.4 * amount;
    },
    setSettle(x, z, amount, time) {
      if (amount <= 0.001) {
        ring.visible = false;
        return;
      }
      ring.visible = true;
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
      wakeMaterial.opacity = 0;
      ringMaterial.opacity = 0;
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
