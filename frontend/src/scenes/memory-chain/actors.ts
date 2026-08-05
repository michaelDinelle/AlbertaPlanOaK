
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveColor,
  setEmissiveIntensity,
  tierEffectScale,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import {
  blendTokens,
  clamp01,
  mergeColoredParts,
  tierBounce,
  tierToonFloor,
  type CaveSkin,
  type ColoredPart,
} from './world';


const CRYSTAL_OFFSET_Z = -0.82;
export const CRYSTAL_HEIGHT = 1.62;
const CRYSTAL_HDR = 1.7;

export function emissiveTierGain(tier: QualityTier): number {
  return tier === 'thumb' ? 0.72 : 1;
}

export function mouthTierGain(tier: QualityTier): number {
  return tier === 'thumb' ? 1.12 : 1;
}

export function mouthStoneTint(tier: QualityTier): number {
  return tier === 'thumb' ? 0.62 : 0;
}

export function mouthStoneEmissive(tier: QualityTier): number {
  return tier === 'thumb' ? 0.7 : 0;
}

export const MOUTH_RIM = 0.5;

export interface MouthCarriers {
  rim: number;
  rimColor: THREE.Color;
  glow: number;
  glowColor: THREE.Color;
  stone: THREE.Color;
  stoneEmissive: number;
  stoneEmissiveColor: THREE.Color;
  total: number;
}

export function mouthCarriers(root: THREE.Object3D, name: string): MouthCarriers {
  const frame = root.getObjectByName(`${name}Frame`) as THREE.Mesh | undefined;
  const throat = root.getObjectByName(`${name}Throat`) as THREE.Mesh | undefined;
  const frameMaterial = frame?.material as OakMaterial | undefined;
  const throatMaterial = throat?.material as EmissiveMaterial | undefined;
  const handle = oakUniforms(frameMaterial);
  const rim = handle && handle.flags.rim ? handle.uniforms.uRimStrength.value : 0;
  const throatColor = throatMaterial ? throatMaterial.color : new THREE.Color(0, 0, 0);
  const drawn = throat?.visible === true;
  return {
    rim,
    rimColor: handle ? handle.uniforms.uRimColor.value : new THREE.Color(0, 0, 0),
    glow: drawn ? Math.max(throatColor.r, throatColor.g, throatColor.b) : 0,
    glowColor: throatColor,
    stone: frameMaterial ? frameMaterial.color : new THREE.Color(0, 0, 0),
    stoneEmissive: frameMaterial ? frameMaterial.emissiveIntensity : 0,
    stoneEmissiveColor: frameMaterial ? frameMaterial.emissive : new THREE.Color(0, 0, 0),
    total:
      rim +
      (drawn ? Math.max(throatColor.r, throatColor.g, throatColor.b) : 0) +
      (frameMaterial ? frameMaterial.emissiveIntensity : 0),
  };
}

export interface CueCrystalOptions {
  tier: QualityTier;
  skin: CaveSkin;
  cueColors: readonly [THREE.ColorRepresentation, THREE.ColorRepresentation];
  name?: string;
}

export interface CueCrystalHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly coreHeight: number;
  readonly offsetZ: number;
  setCue(value: 0 | 1): void;
  setGlow(gain: number): void;
  setTime(t: number): void;
  readonly lit: boolean;
  dispose(): void;
}

function shardGeometry(detail: number): THREE.BufferGeometry {
  const g = new THREE.OctahedronGeometry(0.5, detail);
  const pos = g.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    pos.setY(i, y > 0 ? y * 1.85 : y * 0.42);
  }
  pos.needsUpdate = true;
  g.translate(0, 0.21, 0);
  g.computeVertexNormals();
  g.name = 'cueShard';
  return g;
}

function clusterParts(detail: number, color: THREE.ColorRepresentation): ColoredPart[] {
  const spec: readonly (readonly [number, number, number, number, number])[] = [
    [0.0, 0.0, 1.0, 0.3, 0.0],
    [-0.24, 0.13, 0.62, 0.21, -0.22],
    [0.22, -0.11, 0.7, 0.22, 0.19],
    [0.05, 0.26, 0.45, 0.17, 0.1],
    [-0.13, -0.24, 0.38, 0.16, -0.12],
    [0.33, 0.19, 0.3, 0.13, 0.26],
    [-0.35, -0.05, 0.26, 0.12, -0.3],
  ];
  const parts: ColoredPart[] = [];
  for (const s of spec) {
    const g = shardGeometry(detail);
    g.scale(s[3], s[2] * CRYSTAL_HEIGHT, s[3]);
    g.rotateZ(s[4]);
    g.rotateY(s[0] * 3.1 + s[1] * 1.7);
    g.translate(s[0], 0, s[1]);
    parts.push({ geometry: g, color });
  }
  return parts;
}

export function buildCueCrystal(opts: CueCrystalOptions): CueCrystalHandle {
  const { tier, skin } = opts;
  const detail = tier === 'thumb' ? 0 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'cueCrystal';

  const root = new THREE.Object3D();
  root.name = 'cueCrystalRoot';
  group.add(root);

  const cluster = new THREE.Object3D();
  cluster.name = 'cueCrystalCluster';
  cluster.position.set(0, 0, CRYSTAL_OFFSET_Z);
  root.add(cluster);

  const plinthParts: ColoredPart[] = [];
  {
    const pier = new THREE.CylinderGeometry(0.66, 0.95, 1.5, tier === 'thumb' ? 6 : 9, 1);
    pier.translate(0, -0.72, 0);
    plinthParts.push({ geometry: pier, color: blendTokens(skin.rockLow, skin.rockHigh, 0.5) });
    const base = new THREE.CylinderGeometry(0.52, 0.66, 0.3, tier === 'thumb' ? 6 : 9, 1);
    base.translate(0, 0.13, 0);
    plinthParts.push({ geometry: base, color: skin.rockLow });
    const lip = new THREE.CylinderGeometry(0.38, 0.5, 0.13, tier === 'thumb' ? 6 : 9, 1);
    lip.translate(0, 0.31, 0);
    plinthParts.push({ geometry: lip, color: blendTokens(skin.rockLow, skin.mineral, 0.4) });
  }
  const plinthGeometry = mergeColoredParts(plinthParts, 'cueCrystalPlinth');
  const rockMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: skin.rim,
    rimStrength: 0.7,
    emissive: skin.rockLow,
    emissiveIntensity: tierBounce(tier),
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: tierToonFloor(tier, 0.34),
    tier,
    name: 'cueCrystalRock',
  });
  const plinth = new THREE.Mesh(plinthGeometry, rockMaterial);
  plinth.name = 'cueCrystalPlinth';
  plinth.castShadow = tier !== 'thumb';
  plinth.receiveShadow = tier !== 'thumb';
  cluster.add(plinth);

  const quartzGeometry = mergeColoredParts(
    clusterParts(detail, skin.quartz),
    'cueCrystalQuartz',
  );
  const quartzMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.42,
    metalness: 0.05,
    rimColor: skin.rim,
    rimStrength: 1.25,
    rimPower: 2.1,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: tierToonFloor(tier, 0.42),
    tier,
    name: 'cueCrystalQuartz',
  });
  const quartz = new THREE.Mesh(quartzGeometry, quartzMaterial);
  quartz.name = 'cueCrystalQuartz';
  quartz.position.y = 0.3;
  quartz.castShadow = tier !== 'thumb';
  cluster.add(quartz);

  const glowGeometry = mergeColoredParts(clusterParts(detail, 0xffffff), 'cueCrystalGlow');
  const glowGain = CRYSTAL_HDR * emissiveTierGain(tier);
  const glowMaterial: EmissiveMaterial = emissiveMaterial(opts.cueColors[0], glowGain, {
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    name: 'cueCrystalGlow',
  });
  const glow = new THREE.Mesh(glowGeometry, glowMaterial);
  glow.name = 'cueCrystalGlow';
  glow.position.y = 0.3;
  glow.scale.setScalar(1.055);
  cluster.add(glow);

  const haloGeometry = new THREE.IcosahedronGeometry(0.95, tier === 'thumb' ? 0 : 1);
  const haloMaterial: EmissiveMaterial = emissiveMaterial(opts.cueColors[0], glowGain * 0.24, {
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'cueCrystalHalo',
  });
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.name = 'cueCrystalHalo';
  halo.position.y = 0.92;
  halo.scale.set(1, 1.35, 1);
  cluster.add(halo);

  let glowLevel = 0;
  glow.visible = false;
  halo.visible = false;

  return {
    group,
    root,
    coreHeight: 0.95,
    offsetZ: CRYSTAL_OFFSET_Z,

    setCue(value) {
      const color = opts.cueColors[value === 1 ? 1 : 0];
      setEmissiveColor(glowMaterial, color);
      setEmissiveColor(haloMaterial, color);
    },

    setGlow(gain) {
      const g = clamp01(gain);
      glowLevel = g;
      const on = g > 0;
      glow.visible = on;
      halo.visible = on;
      setEmissiveIntensity(glowMaterial, glowGain * g);
      setEmissiveIntensity(haloMaterial, glowGain * 0.24 * g);
      halo.scale.set(1 + 0.1 * g, 1.35 + 0.16 * g, 1 + 0.1 * g);
    },

    setTime(t) {
      oakUniforms(quartzMaterial)?.setTime(t);
      oakUniforms(rockMaterial)?.setTime(t);
      if (glowLevel > 0) {
        const breath = 1 + 0.09 * Math.sin(t * 1.35) + 0.05 * Math.sin(t * 2.9 + 1.1);
        glow.scale.setScalar(1.055 * breath);
        halo.rotation.y = t * 0.32;
      }
    },

    get lit() {
      return glowLevel > 0;
    },

    dispose() {
      plinthGeometry.dispose();
      quartzGeometry.dispose();
      glowGeometry.dispose();
      haloGeometry.dispose();
      rockMaterial.dispose();
      quartzMaterial.dispose();
      glowMaterial.dispose();
      haloMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}


const MOUTH_OFFSET_Z = 1.42;
const MOUTH_OFFSET_X = 1.15;
export const MOUTH_HEIGHT = 3.3;
const MOUTH_HDR = 0.72;

export interface ExitMouthOptions {
  tier: QualityTier;
  skin: CaveSkin;
  color: THREE.ColorRepresentation;
  side: 1 | -1;
  name?: string;
}

function throatLensGeometry(tier: QualityTier, halfZ: number, halfY: number, depth: number): THREE.BufferGeometry {
  const seg = tier === 'thumb' ? 12 : tier === 'medium' ? 18 : 26;
  const rings: readonly (readonly [number, number])[] = [
    [0, 1],
    [0.28, 0.8],
    [0.55, 0.45],
    [0.8, 0.16],
    [1, 0],
  ];
  const count = 1 + seg * (rings.length - 1);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  positions[0] = depth;
  colors[0] = colors[1] = colors[2] = 1;
  let v = 1;
  for (let r = 1; r < rings.length; r++) {
    const ring = rings[r] as readonly [number, number];
    const radius = ring[0];
    const bright = ring[1];
    for (let s = 0; s < seg; s++) {
      const a = (2 * Math.PI * s) / seg;
      positions[v * 3] = depth * (1 - radius * radius);
      positions[v * 3 + 1] = halfY * radius * Math.sin(a);
      positions[v * 3 + 2] = halfZ * radius * Math.cos(a);
      colors[v * 3] = bright;
      colors[v * 3 + 1] = bright;
      colors[v * 3 + 2] = bright;
      v++;
    }
  }
  const indices = new Uint16Array((seg + seg * 2 * (rings.length - 2)) * 3);
  let o = 0;
  for (let s = 0; s < seg; s++) {
    indices[o++] = 0;
    indices[o++] = 1 + s;
    indices[o++] = 1 + ((s + 1) % seg);
  }
  for (let r = 1; r < rings.length - 1; r++) {
    const inner = 1 + (r - 1) * seg;
    const outer = 1 + r * seg;
    for (let s = 0; s < seg; s++) {
      const s2 = (s + 1) % seg;
      indices[o++] = inner + s;
      indices[o++] = outer + s;
      indices[o++] = outer + s2;
      indices[o++] = inner + s;
      indices[o++] = outer + s2;
      indices[o++] = inner + s2;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeBoundingSphere();
  g.name = 'exitMouthThroatLens';
  return g;
}

export interface ExitMouthHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly glowHeight: number;
  readonly offsetZ: number;
  setGlow(gain: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildExitMouth(opts: ExitMouthOptions): ExitMouthHandle {
  const { tier, skin, side } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'exitMouth';

  const root = new THREE.Object3D();
  root.name = `${group.name}Root`;
  group.add(root);

  const arch = new THREE.Object3D();
  arch.name = `${group.name}Arch`;
  arch.position.set(MOUTH_OFFSET_X, 0, MOUTH_OFFSET_Z * side);
  arch.rotation.y = -0.24 * side;
  root.add(arch);

  const ringR = MOUTH_HEIGHT * 0.36;
  const ringY = MOUTH_HEIGHT * 0.58;
  const parts: ColoredPart[] = [];
  {
    const radial = Math.max(4, Math.round(9 * detail));
    const tubular = Math.max(8, Math.round(20 * detail));
    const ring = new THREE.TorusGeometry(ringR, 0.19, radial, tubular, Math.PI);
    ring.translate(0, ringY, 0);
    ring.rotateY(Math.PI / 2);
    parts.push({ geometry: ring, color: skin.rockLow });
    for (const sz of [-1, 1]) {
      const jamb = new THREE.BoxGeometry(0.3, ringY, 0.3);
      jamb.translate(0, ringY * 0.5, sz * ringR);
      parts.push({ geometry: jamb, color: blendTokens(skin.rockLow, skin.rockHigh, 0.45) });
    }
    const lintel = new THREE.BoxGeometry(0.34, 0.3, 0.4);
    lintel.translate(0, ringY + ringR + 0.06, 0);
    parts.push({ geometry: lintel, color: skin.mineral });
    const sill = new THREE.BoxGeometry(1.5, 0.2, ringR * 2.5);
    sill.translate(-0.25, 0.05, 0);
    parts.push({ geometry: sill, color: skin.rockLow });
    const pier = new THREE.BoxGeometry(1.15, 1.7, ringR * 2.05);
    pier.translate(-0.25, -0.86, 0);
    parts.push({ geometry: pier, color: blendTokens(skin.rockLow, skin.rockHigh, 0.55) });
  }
  const frameGeometry = mergeColoredParts(parts, `${group.name}Frame`);
  const stoneTint = mouthStoneTint(tier);
  const emissiveTint = mouthStoneEmissive(tier);
  const frameMaterial: OakMaterial = oakMaterial({
    color: stoneTint > 0 ? blendTokens(0xffffff, opts.color, stoneTint) : 0xffffff,
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: opts.color,
    rimStrength: MOUTH_RIM,
    emissive:
      emissiveTint > 0 ? blendTokens(skin.rockLow, opts.color, emissiveTint) : skin.rockLow,
    emissiveIntensity: tierBounce(tier),
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: tierToonFloor(tier, 0.34),
    tier,
    name: `${group.name}Frame`,
  });
  const frame = new THREE.Mesh(frameGeometry, frameMaterial);
  frame.name = `${group.name}Frame`;
  frame.castShadow = tier !== 'thumb';
  frame.receiveShadow = tier !== 'thumb';
  arch.add(frame);

  const throatGeometry = throatLensGeometry(tier, ringR * 0.8, ringY * 0.55, 0.62);
  const throatGain = MOUTH_HDR * mouthTierGain(tier);
  const throatMaterial: EmissiveMaterial = emissiveMaterial(opts.color, throatGain, {
    transparent: true,
    opacity: 1,
    depthWrite: false,
    side: THREE.FrontSide,
    blending: THREE.AdditiveBlending,
    name: `${group.name}Throat`,
  });
  throatMaterial.vertexColors = true;
  const throat = new THREE.Mesh(throatGeometry, throatMaterial);
  throat.name = `${group.name}Throat`;
  throat.position.set(0.02, ringY * 0.65, 0);
  throat.renderOrder = 2;
  arch.add(throat);

  const bounce = tierBounce(tier);
  const bounceLift = emissiveTint > 0 ? 1.15 : 0;
  const rimScale = tierEffectScale(tier).rim;
  let gain = 1;

  return {
    group,
    root,
    glowHeight: ringY * 0.65,
    offsetZ: MOUTH_OFFSET_Z * side,

    setGlow(g) {
      gain = g < 0 ? 0 : g;
      throat.visible = gain > 0.001;
      setEmissiveIntensity(throatMaterial, throatGain * gain);
      oakUniforms(frameMaterial)?.setRim(MOUTH_RIM * rimScale * Math.min(1, gain));
      frameMaterial.emissiveIntensity = bounce * (1 + bounceLift * Math.min(1.6, gain));
    },

    setTime(t) {
      oakUniforms(frameMaterial)?.setTime(t);
      const breath = 1 + 0.07 * Math.sin(t * 0.55 + (side > 0 ? 0 : 1.9));
      throat.scale.set(1, breath, breath);
    },

    dispose() {
      frameGeometry.dispose();
      throatGeometry.dispose();
      frameMaterial.dispose();
      throatMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
