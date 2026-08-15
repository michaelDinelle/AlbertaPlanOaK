
import { GoalBeacon } from '@kit/grid';
import { cellToWorld, type GridSize } from '@kit/grid';
import {
  emissiveMaterial,
  oakMaterial,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import type { Palette } from '@kit/tokens';
import * as THREE from 'three';
import { hash1f } from '@kit/rng';
import { radialSprite } from '@kit/textures';
import {
  LAMP_REACH,
  LANTERN_Z,
  blendTokens,
  clamp01,
  mergeColoredParts,
  propMaterial,
  setPropTime,
  shade,
  type ColoredPart,
  type MoorSkin,
} from './world';

export const POST_HEIGHT = 1.78;
export const LAMP_Y = 1.5;
export const LAMP_HDR = 2.6;
const HALO_HDR = 0.75;
const POOL_HDR = 0.46;

export interface LanternRoadOptions {
  palette: Palette;
  skin: MoorSkin;
  grid: GridSize;
  goalCell: number;
  seed: number;
  accent: THREE.ColorRepresentation;
  name?: string;
}

export interface LanternRoadHandle {
  group: THREE.Group;
  endRoot: THREE.Object3D;
  readonly endLamp: THREE.Vector3;
  readonly count: number;
  lanternX(cell: number): number;
  setGlow(cell: number, v01: number): void;
  commit(): void;
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function lanternGeometry(skin: MoorSkin): THREE.BufferGeometry {
  const detail = 1;
  const radial = Math.max(4, Math.round(6 * detail));
  const parts: ColoredPart[] = [];

  const foot = new THREE.CylinderGeometry(0.075, 0.115, 0.11, radial);
  foot.translate(0, 0.055, 0);
  parts.push({ geometry: foot, color: skin.stoneDark });

  const shaft = new THREE.CylinderGeometry(0.03, 0.048, POST_HEIGHT, radial);
  shaft.translate(0, POST_HEIGHT * 0.5, 0);
  parts.push({ geometry: shaft, color: skin.iron });

  const collar = new THREE.CylinderGeometry(0.052, 0.052, 0.055, radial);
  collar.translate(0, 1.16, 0);
  parts.push({ geometry: collar, color: skin.ironLight });

  const arm = new THREE.BoxGeometry(0.038, 0.038, LAMP_REACH + 0.08);
  arm.translate(0, POST_HEIGHT - 0.08, (LAMP_REACH + 0.08) * 0.5 - 0.04);
  parts.push({ geometry: arm, color: skin.iron });

  const hook = new THREE.BoxGeometry(0.028, 0.13, 0.028);
  hook.translate(0, POST_HEIGHT - 0.15, LAMP_REACH);
  parts.push({ geometry: hook, color: skin.iron });

  const cap = new THREE.ConeGeometry(0.125, 0.1, radial);
  cap.translate(0, LAMP_Y + 0.15, LAMP_REACH);
  parts.push({ geometry: cap, color: skin.ironLight });

  const tray = new THREE.CylinderGeometry(0.092, 0.074, 0.035, radial);
  tray.translate(0, LAMP_Y - 0.11, LAMP_REACH);
  parts.push({ geometry: tray, color: skin.ironLight });

  const bars = 4;
  for (let i = 0; i < bars; i++) {
    const a = (i / bars) * Math.PI * 2 + Math.PI / 4;
    const bar = new THREE.BoxGeometry(0.017, 0.23, 0.017);
    bar.translate(Math.cos(a) * 0.075, LAMP_Y + 0.015, LAMP_REACH + Math.sin(a) * 0.075);
    parts.push({ geometry: bar, color: skin.iron });
  }

  return mergeColoredParts(parts, 'lanternPost');
}

function cairnGeometry(skin: MoorSkin, seed: number): THREE.BufferGeometry {
  const detail = 0;
  const parts: ColoredPart[] = [];
  const base = skin.stone;
  const dark = shade(skin.stoneDark, 0.1);

  const rings: readonly { y: number; r: number; n: number; s: number }[] = [
    { y: 0.16, r: 0.38, n: 5, s: 0.23 },
    { y: 0.42, r: 0.26, n: 4, s: 0.2 },
    { y: 0.64, r: 0.14, n: 3, s: 0.17 },
    { y: 0.8, r: 0, n: 1, s: 0.15 },
  ];
  let k = 0;
  for (const ring of rings) {
    for (let i = 0; i < ring.n; i++) {
      const h1 = hash1f(k, seed ^ 0x51a3d7);
      const h2 = hash1f(k, seed ^ 0x2ce19b);
      const a = (i / ring.n) * Math.PI * 2 + ring.y * 3.1 + h1 * 0.5;
      const rock = new THREE.IcosahedronGeometry(ring.s * (0.82 + 0.36 * h2), detail);
      rock.scale(1, 0.72 + 0.22 * h1, 0.88);
      rock.rotateY(h1 * 3.14);
      rock.rotateZ(h2 * 0.8 - 0.4);
      rock.translate(Math.cos(a) * ring.r, ring.y + 0.05 * h2, Math.sin(a) * ring.r * 0.8);
      parts.push({ geometry: rock, color: k % 3 === 1 ? dark : base });
      k++;
    }
  }

  const marker = new THREE.CylinderGeometry(0.16, 0.25, 1.16, 5);
  marker.scale(1, 1, 0.62);
  marker.rotateZ(0.15);
  marker.rotateY(-0.34);
  marker.translate(-0.42, 0.57, 0.1);
  parts.push({ geometry: marker, color: base });
  const wedge = new THREE.IcosahedronGeometry(0.19, detail);
  wedge.scale(1, 0.6, 0.9);
  wedge.translate(-0.3, 0.09, -0.24);
  parts.push({ geometry: wedge, color: dark });
  return mergeColoredParts(parts, 'endCairn');
}

function milestoneGeometry(skin: MoorSkin): THREE.BufferGeometry {
  const detail = 1;
  const radial = Math.max(5, Math.round(7 * detail));
  const parts: ColoredPart[] = [];

  const slab = new THREE.CylinderGeometry(0.1, 0.15, 0.52, 4);
  slab.rotateY(Math.PI * 0.25 + 0.16);
  slab.rotateZ(-0.12);
  slab.scale(1, 1, 0.72);
  slab.translate(0, 0.25, 0);
  parts.push({ geometry: slab, color: shade(skin.stone, 0.42) });

  const cap = new THREE.CylinderGeometry(0.085, 0.105, 0.05, 4);
  cap.rotateY(Math.PI * 0.25 + 0.16);
  cap.rotateZ(-0.12);
  cap.scale(1, 1, 0.72);
  cap.translate(0.033, 0.51, 0);
  parts.push({ geometry: cap, color: shade(skin.moss, 0.3) });

  const kerbA = new THREE.CylinderGeometry(0.1, 0.13, 0.08, radial);
  kerbA.translate(0.17, 0.03, 0.12);
  parts.push({ geometry: kerbA, color: shade(skin.stoneDark, 0.25) });
  const kerbB = new THREE.CylinderGeometry(0.08, 0.1, 0.06, radial);
  kerbB.translate(-0.14, 0.025, -0.15);
  parts.push({ geometry: kerbB, color: shade(skin.stoneDark, 0.25) });

  const bed = new THREE.CylinderGeometry(0.22, 0.26, 0.05, radial);
  bed.translate(0, 0.02, 0);
  parts.push({ geometry: bed, color: shade(skin.stoneDark, 0.25) });
  return mergeColoredParts(parts, 'startMilestone');
}

export function buildLanternRoad(opts: LanternRoadOptions): LanternRoadHandle {
  const { palette, skin, grid, goalCell, seed, accent: aurora } = opts;
  const detail = 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'lanternRoad';
  const count = grid.w;
  const instanceCount = Math.max(1, count - 1);

  const postGeometry = lanternGeometry(skin);
  const glassGeometry = new THREE.IcosahedronGeometry(0.058, 1);
  const haloGeometry = new THREE.IcosahedronGeometry(0.12, 2);
  const poolGeometry = new THREE.CircleGeometry(0.86, Math.max(10, Math.round(24 * detail)));
  poolGeometry.rotateX(-Math.PI / 2);
  const poolTexture = radialSprite(64, 1.7);

  const postMaterial: OakMaterial = propMaterial(palette, 'lanternIron');
  const glassMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, LAMP_HDR, { name: 'lanternFlame' });
  const haloMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, HALO_HDR, {
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'lanternHalo',
  });
  const poolColor = blendTokens(skin.flame, palette.accentLight, 0.1);
  const poolMaterial: EmissiveMaterial = emissiveMaterial(poolColor, POOL_HDR, {
    map: poolTexture,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'lanternPool',
  });

  const posts = new THREE.InstancedMesh(postGeometry, postMaterial, instanceCount);
  posts.name = 'lanternPosts';
  posts.castShadow = false;
  posts.receiveShadow = true;
  posts.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3).fill(1), 3);

  const glass = new THREE.InstancedMesh(glassGeometry, glassMaterial, instanceCount);
  glass.name = 'lanternFlames';
  glass.castShadow = false;
  glass.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3), 3);

  const halo = new THREE.InstancedMesh(haloGeometry, haloMaterial, instanceCount);
  halo.name = 'lanternHalos';
  halo.castShadow = false;
  halo.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3), 3);

  const pool = new THREE.InstancedMesh(poolGeometry, poolMaterial, instanceCount);
  pool.name = 'lanternPools';
  pool.castShadow = false;
  pool.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3), 3);

  const instanceOf = new Int32Array(count).fill(-1);
  {
    const matrix = new THREE.Matrix4();
    const lamp = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    const weather = new THREE.Color();
    const world: [number, number, number] = [0, 0, 0];
    let n = 0;
    for (let cell = 0; cell < count; cell++) {
      if (cell === goalCell) continue;
      cellToWorld(cell, 0, grid, world);
      const lean = hash1f(cell, seed ^ 0x2f5a17) - 0.5;
      const wear = hash1f(cell, seed ^ 0x7c1d93);
      position.set(world[0], 0, LANTERN_Z);
      euler.set(0, lean * 0.22, lean * 0.05, 'YXZ');
      quaternion.setFromEuler(euler);
      scale.setScalar(0.94 + 0.12 * wear);
      matrix.compose(position, quaternion, scale);
      posts.setMatrixAt(n, matrix);
      lamp.copy(matrix);
      glass.setMatrixAt(n, lamp);
      halo.setMatrixAt(n, lamp);
      pool.setMatrixAt(n, lamp);
      weather.setRGB(0.9 + 0.2 * wear, 0.9 + 0.2 * wear, 0.92 + 0.18 * wear);
      posts.setColorAt(n, weather);
      instanceOf[cell] = n;
      n++;
    }
    posts.count = n;
    glass.count = n;
    halo.count = n;
    pool.count = n;
  }
  glassGeometry.translate(0, LAMP_Y, LAMP_REACH);
  haloGeometry.translate(0, LAMP_Y, LAMP_REACH);
  poolGeometry.translate(0, 0.014, LAMP_REACH);

  for (const mesh of [posts, glass, halo, pool]) {
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    mesh.frustumCulled = false;
    group.add(mesh);
  }

  const endRoot = new THREE.Object3D();
  endRoot.name = 'endLanternRoot';
  {
    const world: [number, number, number] = [0, 0, 0];
    cellToWorld(goalCell, 0, grid, world);
    endRoot.position.set(world[0], 0, world[2]);
  }
  group.add(endRoot);

  const endPost = new THREE.Mesh(postGeometry, postMaterial);
  endPost.name = 'endLanternPost';
  endPost.position.set(0, 0, LANTERN_Z);
  endPost.castShadow = false;
  endPost.receiveShadow = true;
  endRoot.add(endPost);

  const endGlassMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, LAMP_HDR, { name: 'endLanternFlame' });
  const endGlass = new THREE.Mesh(glassGeometry, endGlassMaterial);
  endGlass.name = 'endLanternFlame';
  endGlass.position.copy(endPost.position);
  endRoot.add(endGlass);

  const endHaloMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, HALO_HDR, {
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'endLanternHalo',
  });
  const endHalo = new THREE.Mesh(haloGeometry, endHaloMaterial);
  endHalo.name = 'endLanternHalo';
  endHalo.position.copy(endPost.position);
  endRoot.add(endHalo);

  const endPoolMaterial: EmissiveMaterial = emissiveMaterial(poolColor, POOL_HDR, {
    map: poolTexture,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'endLanternPool',
  });
  const endPool = new THREE.Mesh(poolGeometry, endPoolMaterial);
  endPool.name = 'endLanternPool';
  endPool.position.copy(endPost.position);
  endRoot.add(endPool);

  const cairnGeo = cairnGeometry(skin, seed);
  const cairnMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.9,
    rimColor: blendTokens(palette.hemiSky, aurora, 0.45),
    rimStrength: 0.38,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.2,
    name: 'endCairnStone',
  });
  const cairn = new THREE.Mesh(cairnGeo, cairnMaterial);
  cairn.name = 'endCairn';
  cairn.position.set(0.74, 0, -0.14);
  cairn.rotation.y = 0.42;
  cairn.castShadow = true;
  cairn.receiveShadow = true;
  endRoot.add(cairn);

  const beacon = new GoalBeacon({
    color: blendTokens(palette.accentLight, aurora, 0.55),
    height: 0.2,
    radius: 0.085,
    intensity: 0.3,
    name: 'endBeacon',
  });
  endRoot.add(beacon.group);

  const glassColors = glass.instanceColor as THREE.InstancedBufferAttribute;
  const haloColors = halo.instanceColor as THREE.InstancedBufferAttribute;
  const poolColors = pool.instanceColor as THREE.InstancedBufferAttribute;
  const endLamp = new THREE.Vector3(endRoot.position.x, LAMP_Y, LANTERN_Z + LAMP_REACH);
  const worldScratch: [number, number, number] = [0, 0, 0];

  return {
    group,
    endRoot,
    endLamp,
    count,

    lanternX(cell) {
      return cellToWorld(cell, 0, grid, worldScratch)[0];
    },

    setGlow(cell, v01) {
      const v = clamp01(v01);
      const id = cell >= 0 && cell < count ? (instanceOf[cell] ?? -1) : -1;
      if (id >= 0) {
        const g = glassColors.array as Float32Array;
        const h = haloColors.array as Float32Array;
        const p = poolColors.array as Float32Array;
        const o = id * 3;
        g[o] = v;
        g[o + 1] = v;
        g[o + 2] = v;
        const hv = v * 0.62;
        h[o] = hv;
        h[o + 1] = hv;
        h[o + 2] = hv;
        const pv = v * 0.8;
        p[o] = pv;
        p[o + 1] = pv;
        p[o + 2] = pv;
        return;
      }
      if (cell === goalCell) {
        setEmissiveIntensity(endGlassMaterial, LAMP_HDR * v);
        setEmissiveIntensity(endHaloMaterial, HALO_HDR * v * 0.62);
        setEmissiveIntensity(endPoolMaterial, POOL_HDR * v * 0.8);
      }
    },

    commit() {
      glassColors.needsUpdate = true;
      haloColors.needsUpdate = true;
      poolColors.needsUpdate = true;
    },

    flare(strength = 1) {
      beacon.flare(strength);
    },

    setTime(t) {
      setPropTime(postMaterial, t);
      setPropTime(cairnMaterial, t);
    },

    update(dt, time) {
      beacon.update(dt, time);
    },

    dispose() {
      beacon.dispose();
      posts.dispose();
      glass.dispose();
      halo.dispose();
      pool.dispose();
      postGeometry.dispose();
      glassGeometry.dispose();
      haloGeometry.dispose();
      poolGeometry.dispose();
      poolTexture.dispose();
      cairnGeo.dispose();
      postMaterial.dispose();
      glassMaterial.dispose();
      haloMaterial.dispose();
      poolMaterial.dispose();
      endGlassMaterial.dispose();
      endHaloMaterial.dispose();
      endPoolMaterial.dispose();
      cairnMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}


export interface WispGlowHandle {
  mesh: THREE.Mesh;
  setGain(g: number): void;
  dispose(): void;
}

export function buildWispGlow(opts: {
  color: THREE.ColorRepresentation;
  radius?: number;
  intensity?: number;
  name?: string;
}): WispGlowHandle {
  const radius = opts.radius ?? 0.95;
  const intensity = opts.intensity ?? 0.5;
  const texture = radialSprite(64, 2.6);
  const geometry = new THREE.PlaneGeometry(radius * 2, radius * 2);
  const material = emissiveMaterial(opts.color, intensity, {
    map: texture,
    transparent: true,
    opacity: 1,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'wispGlowMaterial',
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = opts.name ?? 'wispGlow';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1;

  return {
    mesh,
    setGain(g) {
      setEmissiveIntensity(material, intensity * (g > 0 ? g : 0));
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      texture.dispose();
      mesh.removeFromParent();
    },
  };
}


export interface MilestoneOptions {
  palette: Palette;
  skin: MoorSkin;
  grid: GridSize;
  cell: number;
  name?: string;
}

export interface MilestoneHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setTime(t: number): void;
  dispose(): void;
}

export function buildMilestone(opts: MilestoneOptions): MilestoneHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'startMilestone';
  const root = new THREE.Object3D();
  root.name = 'milestoneRoot';
  {
    const world: [number, number, number] = [0, 0, 0];
    cellToWorld(opts.cell, 0, opts.grid, world);
    root.position.set(world[0], 0, world[2]);
  }
  group.add(root);

  const geometry = milestoneGeometry(opts.skin);
  const material = propMaterial(opts.palette, 'milestoneStone');
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'milestone';
  mesh.position.set(-0.34, 0, 1.32);
  mesh.rotation.y = 0.62;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  root.add(mesh);

  return {
    group,
    root,
    setTime(t) {
      setPropTime(material, t);
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
