
import { emissiveMaterial, oakMaterial, oakUniforms, setEmissiveIntensity } from '@kit/materials';
import type { EmissiveMaterial, OakMaterial } from '@kit/materials';
import { hash2f } from '@kit/rng';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import { blendTokens, clamp01, mergeColoredParts, smoothstep01, tubeGeometry, type ColoredPart } from './world';

type Vec3 = readonly [number, number, number];


interface Limb {
  a: Vec3;
  b: Vec3;
  ra: number;
  rb: number;
  twig: boolean;
}

const FORK: Vec3 = [0.22, 1.52, -0.08];

const TRUNK: readonly Limb[] = [
  { a: [0, 0, 0], b: [0.1, 0.78, -0.03], ra: 0.2, rb: 0.135, twig: false },
  { a: [0.1, 0.78, -0.03], b: FORK, ra: 0.135, rb: 0.092, twig: false },
];

const BOUGHS: readonly { mid: Vec3; tip: Vec3 }[] = [
  { mid: [-0.5, 1.95, 0.16], tip: [-1.12, 2.34, 0.36] },
  { mid: [0.78, 1.9, -0.22], tip: [1.46, 2.24, -0.46] },
  { mid: [0.36, 2.12, 0.5], tip: [0.4, 2.66, 1.06] },
  { mid: [0.38, 2.04, -0.56], tip: [0.58, 2.5, -1.14] },
];

const TWIGS: readonly Vec3[] = [
  [-0.42, 0.44, 0.32],
  [-0.3, 0.5, -0.3],
  [-0.5, 0.34, -0.08],
  [0.4, 0.42, 0.3],
  [0.18, 0.54, -0.34],
  [0.5, 0.36, 0.02],
  [-0.32, 0.4, 0.36],
  [0.3, 0.5, 0.26],
  [0.06, 0.46, 0.48],
  [0.36, 0.44, -0.28],
  [-0.24, 0.52, -0.26],
  [0.14, 0.38, -0.5],
];

function offset3(a: Vec3, d: Vec3): Vec3 {
  return [a[0] + d[0], a[1] + d[1], a[2] + d[2]];
}

function lerp3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export function blossomTreeLimbs(): Limb[] {
  const limbs: Limb[] = [...TRUNK];
  for (let i = 0; i < BOUGHS.length; i++) {
    const bough = BOUGHS[i] as { mid: Vec3; tip: Vec3 };
    limbs.push({ a: FORK, b: bough.mid, ra: 0.082, rb: 0.05, twig: false });
    limbs.push({ a: bough.mid, b: bough.tip, ra: 0.05, rb: 0.03, twig: false });
    const d0 = TWIGS[i * 3] as Vec3;
    const d1 = TWIGS[i * 3 + 1] as Vec3;
    const d2 = TWIGS[i * 3 + 2] as Vec3;
    limbs.push({ a: bough.mid, b: offset3(bough.mid, d0), ra: 0.03, rb: 0.014, twig: true });
    limbs.push({ a: bough.tip, b: offset3(bough.tip, d1), ra: 0.028, rb: 0.013, twig: true });
    limbs.push({ a: bough.tip, b: offset3(bough.tip, d2), ra: 0.026, rb: 0.012, twig: true });
  }
  return limbs;
}

export function blossomTreeCrownHeight(): number {
  let top = 0;
  for (const limb of blossomTreeLimbs()) {
    if (limb.b[1] > top) top = limb.b[1];
    if (limb.a[1] > top) top = limb.a[1];
  }
  return top;
}


export interface BlossomTreeOptions {
  tier: QualityTier;
  palette: Palette;
  blossom: THREE.ColorRepresentation;
  bark: THREE.ColorRepresentation;
  moss: THREE.ColorRepresentation;
  ember: THREE.ColorRepresentation;
  name?: string;
}

export interface BlossomTreeHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly crownHeight: number;
  readonly blossomCount: number;
  setState(bloom01: number, time: number): void;
  setTime(t: number): void;
  dispose(): void;
}

function blossomsPerTwig(tier: QualityTier): number {
  return tier === 'thumb' ? 2 : tier === 'medium' ? 4 : 6;
}

export function buildBlossomTree(opts: BlossomTreeOptions): BlossomTreeHandle {
  const { tier, palette, blossom, bark, moss, ember } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const radial = Math.max(4, Math.round(7 * detail));
  const group = new THREE.Group();
  group.name = opts.name ?? 'blossomTree';

  const root = new THREE.Object3D();
  root.name = 'blossomTreeRoot';
  group.add(root);

  const crown = new THREE.Object3D();
  crown.name = 'blossomTreeCrown';
  root.add(crown);

  const limbs = blossomTreeLimbs();
  const barkParts: ColoredPart[] = [];
  const barkLight = blendTokens(bark, palette.groundB, 0.22);
  for (const limb of limbs) {
    barkParts.push({
      geometry: tubeGeometry(limb.a, limb.b, limb.ra, limb.rb, limb.twig ? Math.max(3, radial - 3) : radial),
      color: limb.twig ? barkLight : bark,
    });
  }
  {
    const boss = new THREE.IcosahedronGeometry(0.33, tier === 'thumb' ? 0 : 1);
    boss.scale(1.05, 0.44, 0.98);
    boss.translate(0.02, 0.06, 0);
    barkParts.push({ geometry: boss, color: moss });
    for (let i = 0; i < (tier === 'thumb' ? 2 : 5); i++) {
      const a = (i / 5) * Math.PI * 2 + 0.4;
      const rootArm = tubeGeometry(
        [0.03, 0.2, 0],
        [Math.cos(a) * 0.31, 0.03, Math.sin(a) * 0.27],
        0.062,
        0.026,
        Math.max(3, radial - 3),
      );
      barkParts.push({ geometry: rootArm, color: blendTokens(bark, moss, 0.45) });
    }
  }
  const barkGeometry = mergeColoredParts(barkParts, 'blossomTreeBark');
  const barkMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.92,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.22,
    tier,
    name: 'blossomTreeBarkMaterial',
  });
  const barkMesh = new THREE.Mesh(barkGeometry, barkMaterial);
  barkMesh.name = 'blossomTreeBark';
  barkMesh.castShadow = tier !== 'thumb';
  barkMesh.receiveShadow = tier !== 'thumb';
  crown.add(barkMesh);

  const perTwig = blossomsPerTwig(tier);
  const twigs = limbs.filter((l) => l.twig);
  const blossomCount = twigs.length * perTwig;
  const blossomGeometry = blossomClusterGeometry(tier);
  const blossomMaterial: OakMaterial = oakMaterial({
    color: blossom,
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.72,
    rimColor: blendTokens(blossom, palette.sun, 0.5),
    rimStrength: 0.5,
    toonSteps: 2,
    toonSoftness: 0.38,
    toonFloor: 0.38,
    tier,
    name: 'blossomTreePetalMaterial',
  });
  const blossoms = new THREE.InstancedMesh(blossomGeometry, blossomMaterial, Math.max(1, blossomCount));
  blossoms.name = 'blossomTreePetals';
  blossoms.castShadow = false;
  blossoms.receiveShadow = false;
  blossoms.frustumCulled = false;
  blossoms.count = blossomCount;
  blossoms.visible = blossomCount > 0;
  blossoms.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, blossomCount) * 3), 3);
  crown.add(blossoms);

  const anchors = new Float32Array(Math.max(1, blossomCount) * 3);
  const quats = new Float32Array(Math.max(1, blossomCount) * 4);
  const sizes = new Float32Array(Math.max(1, blossomCount));
  const phases = new Float32Array(Math.max(1, blossomCount));
  {
    const q = new THREE.Quaternion();
    const e = new THREE.Euler(0, 0, 0, 'YXZ');
    const c = new THREE.Color();
    let k = 0;
    for (let t = 0; t < twigs.length; t++) {
      const twig = twigs[t] as Limb;
      for (let j = 0; j < perTwig; j++) {
        const h0 = hash2f(t, j, 0x1f83d9ab);
        const h1 = hash2f(t, j + 64, 0x5bd1e995);
        const h2 = hash2f(t, j + 128, 0x27d4eb2f);
        const u = 0.28 + ((j + h0) / perTwig) * 0.72;
        const p = lerp3(twig.a, twig.b, u);
        anchors[k * 3] = p[0] + (h1 - 0.5) * 0.1;
        anchors[k * 3 + 1] = p[1] + (h2 - 0.5) * 0.09;
        anchors[k * 3 + 2] = p[2] + (h0 - 0.5) * 0.1;
        e.set((h1 - 0.5) * 1.5, h0 * Math.PI * 2, (h2 - 0.5) * 1.5);
        q.setFromEuler(e);
        quats[k * 4] = q.x;
        quats[k * 4 + 1] = q.y;
        quats[k * 4 + 2] = q.z;
        quats[k * 4 + 3] = q.w;
        sizes[k] = 0.9 + h2 * 0.7;
        phases[k] = clamp01(u * 0.72 + h0 * 0.28);
        c.setRGB(0.9 + h0 * 0.2, 0.9 + h1 * 0.16, 0.92 + h2 * 0.16);
        blossoms.setColorAt(k, c);
        k++;
      }
    }
    if (blossoms.instanceColor) blossoms.instanceColor.needsUpdate = true;
  }

  const emberGeometry = new THREE.IcosahedronGeometry(0.1, tier === 'thumb' ? 0 : 1);
  const emberMaterial: EmissiveMaterial = emissiveMaterial(ember, 0.0, { name: 'blossomTreeHeart' });
  const emberMesh = new THREE.Mesh(emberGeometry, emberMaterial);
  emberMesh.name = 'blossomTreeHeart';
  emberMesh.position.set(FORK[0], FORK[1] + 0.08, FORK[2]);
  crown.add(emberMesh);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3(1, 1, 1);
  const crownHeight = blossomTreeCrownHeight();
  let bloom = 0;

  return {
    group,
    root,
    crownHeight,
    blossomCount,

    setState(bloom01, time) {
      bloom = clamp01(bloom01);
      if (blossomCount > 0) {
        for (let i = 0; i < blossomCount; i++) {
          const phase = phases[i] ?? 0;
          const open = clamp01((bloom - phase * 0.42) / 0.58);
          const eased = open * open * (3 - 2 * open);
          const breath = 1 + 0.07 * Math.sin(time * 1.35 + phase * 9.4);
          position.set(anchors[i * 3] ?? 0, anchors[i * 3 + 1] ?? 0, anchors[i * 3 + 2] ?? 0);
          quaternion.set(quats[i * 4] ?? 0, quats[i * 4 + 1] ?? 0, quats[i * 4 + 2] ?? 0, quats[i * 4 + 3] ?? 1);
          scale.setScalar((sizes[i] ?? 1) * eased * breath);
          matrix.compose(position, quaternion, scale);
          blossoms.setMatrixAt(i, matrix);
        }
        blossoms.instanceMatrix.needsUpdate = true;
      }
      setEmissiveIntensity(emberMaterial, 0.72 * bloom * bloom);
      emberMesh.scale.setScalar(0.42 + 0.44 * bloom);
    },

    setTime(t) {
      crown.rotation.z = Math.sin(t * 0.62) * 0.02 * (0.5 + 0.5 * bloom);
      crown.rotation.x = Math.cos(t * 0.47) * 0.016 * (0.5 + 0.5 * bloom);
      oakUniforms(barkMaterial)?.setTime(t);
      oakUniforms(blossomMaterial)?.setTime(t);
    },

    dispose() {
      blossoms.dispose();
      blossomGeometry.dispose();
      blossomMaterial.dispose();
      barkGeometry.dispose();
      barkMaterial.dispose();
      emberGeometry.dispose();
      emberMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

export function blossomClusterGeometry(tier: QualityTier): THREE.BufferGeometry {
  const petals = tier === 'thumb' ? 4 : 5;
  const heads: readonly { at: Vec3; tilt: number; yaw: number; scale: number }[] =
    tier === 'thumb'
      ? [{ at: [0, 0, 0], tilt: 0, yaw: 0, scale: 1.15 }]
      : [
          { at: [0, 0, 0], tilt: 0, yaw: 0, scale: 1 },
          { at: [0.072, 0.026, -0.03], tilt: 0.5, yaw: 1.1, scale: 0.86 },
          { at: [-0.052, 0.016, 0.062], tilt: -0.42, yaw: 2.4, scale: 0.78 },
        ];
  const inner = 0.02;
  const outer = 0.088;
  const cup = 0.028;

  const tris = heads.length * petals * 2;
  const positions = new Float32Array(tris * 3 * 3);
  const colors = new Float32Array(tris * 3 * 3);
  let o = 0;
  const v = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler(0, 0, 0, 'YXZ');

  for (const head of heads) {
    e.set(head.tilt, head.yaw, head.tilt * 0.6);
    q.setFromEuler(e);
    const place = (x: number, y: number, z: number, g: number): void => {
      v.set(x * head.scale, y * head.scale, z * head.scale).applyQuaternion(q);
      positions[o * 3] = v.x + head.at[0];
      positions[o * 3 + 1] = v.y + head.at[1];
      positions[o * 3 + 2] = v.z + head.at[2];
      colors[o * 3] = g;
      colors[o * 3 + 1] = g * (g > 1 ? 0.97 : 1);
      colors[o * 3 + 2] = g * (g > 1 ? 0.97 : 1);
      o++;
    };
    for (let i = 0; i < petals; i++) {
      const a0 = (i / petals) * Math.PI * 2;
      const a1 = ((i + 1) / petals) * Math.PI * 2;
      const am = (a0 + a1) * 0.5;
      const sx = Math.cos(a0) * inner;
      const sz = Math.sin(a0) * inner;
      const ex = Math.cos(a1) * inner;
      const ez = Math.sin(a1) * inner;
      const lx = Math.cos(a0 + (am - a0) * 0.25) * outer;
      const lz = Math.sin(a0 + (am - a0) * 0.25) * outer;
      const rx = Math.cos(a1 - (a1 - am) * 0.25) * outer;
      const rz = Math.sin(a1 - (a1 - am) * 0.25) * outer;
      place(sx, 0, sz, 1.24);
      place(lx, cup, lz, 0.86);
      place(rx, cup, rz, 0.86);
      place(sx, 0, sz, 1.24);
      place(rx, cup, rz, 0.86);
      place(ex, 0, ez, 1.24);
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'blossomCluster';
  return g;
}


export const FERRY_SECONDS = 1.55;
export const FERRY_STAGGER = 0.42;

export interface PetalFerryOptions {
  tier: QualityTier;
  palette: Palette;
  blossom: THREE.ColorRepresentation;
  from: readonly [number, number, number];
  to: readonly [number, number, number];
  name?: string;
}

export interface PetalFerryHandle {
  mesh: THREE.InstancedMesh;
  readonly count: number;
  fire(now: number): void;
  clear(): void;
  setTime(time: number): void;
  progress(time: number): number;
  dispose(): void;
}

export function buildPetalFerry(opts: PetalFerryOptions): PetalFerryHandle {
  const { tier, palette, blossom, from, to } = opts;
  const count = tier === 'thumb' ? 0 : tier === 'medium' ? 22 : 38;
  const geometry = petalGeometry();
  const material: OakMaterial = oakMaterial({
    color: blossom,
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.66,
    rimColor: blendTokens(blossom, palette.sun, 0.55),
    rimStrength: 0.6,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.45,
    tier,
    name: 'petalFerryMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
  mesh.name = opts.name ?? 'petalFerry';
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.frustumCulled = false;
  mesh.count = count;
  mesh.visible = count > 0;

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(0, 0, 0);
  const span = Math.max(0.05, FERRY_SECONDS - FERRY_STAGGER);
  let firedAt = -Infinity;

  const hideAll = (): void => {
    if (count === 0) return;
    scale.set(0, 0, 0);
    quaternion.set(0, 0, 0, 1);
    position.set(from[0], from[1], from[2]);
    matrix.compose(position, quaternion, scale);
    for (let i = 0; i < count; i++) mesh.setMatrixAt(i, matrix);
    mesh.instanceMatrix.needsUpdate = true;
  };
  hideAll();

  return {
    mesh,
    count,
    fire(now) {
      firedAt = now;
    },
    clear() {
      firedAt = -Infinity;
      hideAll();
    },
    progress(time) {
      const age = time - firedAt;
      if (!(age >= 0) || age > FERRY_SECONDS) return 0;
      return 1 - age / FERRY_SECONDS;
    },
    setTime(time) {
      if (count === 0) return;
      const age = time - firedAt;
      if (!(age >= 0) || age > FERRY_SECONDS + 0.001) {
        hideAll();
        return;
      }
      for (let i = 0; i < count; i++) {
        const u = count > 1 ? i / (count - 1) : 0;
        const h = hash2f(i, 7, 0x9e3779b9);
        const start = u * FERRY_STAGGER;
        const raw = (age - start) / span;
        const t = raw <= 0 ? 0 : raw >= 1 ? 1 : raw;
        const live = raw > 0 && raw < 1 ? 1 : 0;
        const e = Math.pow(t, 0.82);
        const arc = Math.sin(Math.PI * t);
        const px = from[0] + (to[0] - from[0]) * e;
        const py = from[1] + (to[1] - from[1]) * t + arc * (0.42 + h * 0.5) + Math.sin(t * 9 + h * 6.28) * 0.05;
        const pz = from[2] + (to[2] - from[2]) * t + Math.sin(t * 4.4 + u * 6.28) * (0.3 + 0.7 * h) * (0.25 + 0.75 * t);
        const fadeIn = smoothstep01(t / 0.14);
        const fadeOut = 1 - smoothstep01((t - 0.74) / 0.26);
        position.set(px, py, pz);
        euler.set(t * (5.5 + h * 4), t * (4.2 + h * 3), t * (6.1 + h * 2));
        quaternion.setFromEuler(euler);
        scale.setScalar(live * (0.75 + h * 0.6) * fadeIn * fadeOut);
        matrix.compose(position, quaternion, scale);
        mesh.setMatrixAt(i, matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}


export interface FordLanternOptions {
  tier: QualityTier;
  palette: Palette;
  stone: THREE.ColorRepresentation;
  moss: THREE.ColorRepresentation;
  offset?: readonly [number, number, number];
  name?: string;
}

export const LANTERN_ART_SCALE = 1.45;
const LANTERN_PLINTH_R = 0.3;
const LANTERN_PLINTH_H = 0.15;
export const LANTERN_BASE_RADIUS = LANTERN_PLINTH_R * LANTERN_ART_SCALE;
export const LANTERN_BASE_HEIGHT = LANTERN_PLINTH_H * LANTERN_ART_SCALE;
export const LANTERN_OFFSET: readonly [number, number, number] = [-0.45, 0, -2.35];

export interface FordLanternHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setTime(t: number): void;
  dispose(): void;
}

export function buildFordLantern(opts: FordLanternOptions): FordLanternHandle {
  const { tier, palette, stone, moss } = opts;
  const offset = opts.offset ?? LANTERN_OFFSET;
  const detail = tier === 'thumb' ? 0.5 : tier === 'medium' ? 0.75 : 1;
  const sides = Math.max(5, Math.round(6 * detail));
  const group = new THREE.Group();
  group.name = opts.name ?? 'fordLantern';

  const root = new THREE.Object3D();
  root.name = 'fordLanternRoot';
  group.add(root);

  const body = new THREE.Object3D();
  body.name = 'fordLanternBody';
  body.position.set(offset[0], offset[1], offset[2]);
  body.rotation.y = 0.38;
  body.scale.setScalar(LANTERN_ART_SCALE);
  root.add(body);

  const dark = blendTokens(stone, moss, 0.34);
  const parts: ColoredPart[] = [];
  const drum = (r0: number, r1: number, h: number, y: number, color: THREE.ColorRepresentation): void => {
    const g = new THREE.CylinderGeometry(r1, r0, h, sides, 1);
    g.translate(0, y + h / 2, 0);
    parts.push({ geometry: g, color });
  };
  drum(LANTERN_PLINTH_R, 0.26, LANTERN_PLINTH_H, 0, dark);
  drum(0.085, 0.075, 0.62, 0.15, stone);
  drum(0.24, 0.25, 0.08, 0.77, dark);
  drum(0.2, 0.21, 0.3, 0.85, stone);
  drum(0.35, 0.05, 0.27, 1.15, dark);
  {
    const finial = new THREE.IcosahedronGeometry(0.065, tier === 'thumb' ? 0 : 1);
    finial.translate(0, 1.46, 0);
    parts.push({ geometry: finial, color: stone });
    const collar = new THREE.CylinderGeometry(0.32, 0.34, 0.06, sides, 1);
    collar.translate(0, 0.02, 0);
    parts.push({ geometry: collar, color: moss });
  }

  const geometry = mergeColoredParts(parts, 'fordLantern');
  const material: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.96,
    rimColor: palette.skyHorizon,
    rimStrength: 0.36,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.22,
    tier,
    name: 'fordLanternMaterial',
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'fordLanternMesh';
  mesh.castShadow = tier !== 'thumb';
  mesh.receiveShadow = tier !== 'thumb';
  body.add(mesh);

  return {
    group,
    root,
    setTime(t) {
      oakUniforms(material)?.setTime(t);
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

export function petalGeometry(): THREE.BufferGeometry {
  const pts: readonly (readonly [number, number, number, number])[] = [
    [0, 0, -0.115, 1.2],
    [-0.068, 0.021, 0.018, 0.92],
    [0, 0, 0.125, 0.72],
    [0.068, 0.021, 0.018, 0.92],
  ];
  const tris: readonly (readonly [number, number, number])[] = [
    [0, 1, 2],
    [0, 2, 3],
  ];
  const positions = new Float32Array(2 * 3 * 3);
  const colors = new Float32Array(2 * 3 * 3);
  let o = 0;
  for (const tri of tris) {
    for (const idx of tri) {
      const p = pts[idx] as readonly [number, number, number, number];
      positions[o * 3] = p[0];
      positions[o * 3 + 1] = p[1];
      positions[o * 3 + 2] = p[2];
      colors[o * 3] = p[3];
      colors[o * 3 + 1] = p[3];
      colors[o * 3 + 2] = p[3];
      o++;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'chainPetal';
  return g;
}
