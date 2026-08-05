
import { emissiveMaterial, oakMaterial, oakUniforms, setEmissiveIntensity } from '@kit/materials';
import type { EmissiveMaterial, OakMaterial } from '@kit/materials';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import { blendTokens, clamp01, mergeColoredParts, type AbyssSkin, type ColoredPart } from './world';


export const HEART_Y = 0.62;
export const HEART_HDR = 2.6;
export const WRECK_SCALE = 1.35;

export interface WreckOptions {
  tier: QualityTier;
  skin: AbyssSkin;
  accent: THREE.ColorRepresentation;
  name?: string;
}

export interface WreckHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  heartAnchor: THREE.Object3D;
  setFlare(flare01: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildWreck(opts: WreckOptions): WreckHandle {
  const { tier, skin } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'sunkenWreck';

  const root = new THREE.Object3D();
  root.name = 'wreckRoot';
  group.add(root);

  const hull = new THREE.Object3D();
  hull.name = 'wreckHull';
  hull.rotation.set(0.1, -0.62, 0.13);
  hull.scale.setScalar(WRECK_SCALE);
  root.add(hull);

  const parts: ColoredPart[] = [];
  const radial = Math.max(4, Math.round(7 * detail));

  {
    const keel = new THREE.BoxGeometry(1.58, 0.13, 0.3);
    keel.translate(0, 0.075, 0);
    parts.push({ geometry: keel, color: skin.timberDark });
    const spine = new THREE.BoxGeometry(1.34, 0.09, 0.11);
    spine.translate(-0.02, 0.2, 0);
    parts.push({ geometry: spine, color: skin.timber });
  }

  const RIBS = tier === 'thumb' ? 4 : 7;
  for (let i = 0; i < RIBS; i++) {
    const t = RIBS > 1 ? i / (RIBS - 1) : 0.5;
    const x = -0.62 + t * 1.24;
    const fall = 1 - Math.pow(Math.abs(t - 0.46) * 2.1, 2);
    const height = 0.34 + 0.42 * Math.max(0.1, fall);
    const snapped = i === 1 || i === RIBS - 2;
    const arc = snapped ? Math.PI * 0.44 : Math.PI * 0.86;
    const rib = new THREE.TorusGeometry(height, 0.032, Math.max(3, Math.round(5 * detail)), Math.max(5, Math.round(11 * detail)), arc);
    rib.rotateZ(Math.PI - arc * 0.5 - (Math.PI - arc) * 0.5);
    rib.rotateY(Math.PI / 2);
    rib.translate(x, 0.12, 0);
    parts.push({ geometry: rib, color: i % 3 === 0 ? skin.timber : skin.timberDark });
  }

  {
    const stern = new THREE.BoxGeometry(0.34, 0.46, 0.5);
    stern.rotateX(0.12);
    stern.translate(-0.66, 0.28, 0);
    parts.push({ geometry: stern, color: skin.timber });
    const mast = new THREE.CylinderGeometry(0.03, 0.055, 0.86, radial);
    mast.translate(0, 0.43, 0);
    mast.rotateZ(0.42);
    mast.rotateX(0.16);
    mast.translate(0.16, 0.18, 0.02);
    parts.push({ geometry: mast, color: skin.timberDark });
    const spar = new THREE.CylinderGeometry(0.022, 0.028, 0.52, Math.max(3, radial - 2));
    spar.rotateZ(Math.PI / 2);
    spar.translate(0.34, 0.66, -0.03);
    parts.push({ geometry: spar, color: skin.timber });
  }

  const hullGeometry = mergeColoredParts(parts, 'wreckHull');
  const hullMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.92,
    side: THREE.DoubleSide,
    rimColor: opts.accent,
    rimStrength: 0.95,
    rimPower: 2.2,
    toonSteps: 2,
    toonSoftness: 0.35,
    toonFloor: 0.32,
    tier,
    name: 'wreckTimber',
  });
  const hullMesh = new THREE.Mesh(hullGeometry, hullMaterial);
  hullMesh.name = 'wreckTimberMesh';
  hullMesh.castShadow = tier !== 'thumb';
  hullMesh.receiveShadow = tier !== 'thumb';
  hull.add(hullMesh);

  const heartAnchor = new THREE.Object3D();
  heartAnchor.name = 'wreckHeartAnchor';
  heartAnchor.position.set(0, HEART_Y, 0);
  root.add(heartAnchor);

  const heartDecor = new THREE.Object3D();
  heartDecor.name = 'wreckHeartDecor';
  heartDecor.scale.setScalar(WRECK_SCALE);
  heartAnchor.add(heartDecor);

  const coreGeometry = new THREE.IcosahedronGeometry(0.11, tier === 'thumb' ? 0 : 1);
  const coreMaterial: EmissiveMaterial = emissiveMaterial(skin.heart, HEART_HDR, { name: 'wreckHeartCore' });
  const core = new THREE.Mesh(coreGeometry, coreMaterial);
  core.name = 'wreckHeartCore';
  heartDecor.add(core);

  const haloGeometry = new THREE.IcosahedronGeometry(0.23, tier === 'thumb' ? 0 : 1);
  const haloMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(skin.heart, opts.accent, 0.45), HEART_HDR * 0.24, {
    transparent: true,
    opacity: 0.26,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'wreckHeartHalo',
  });
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.name = 'wreckHeartHalo';
  heartDecor.add(halo);

  let flareAmount = 0;

  return {
    group,
    root,
    heartAnchor,
    setFlare(flare01) {
      flareAmount = clamp01(flare01);
    },
    setTime(t) {
      hull.position.y = Math.sin(t * 0.42) * 0.012;
      hull.rotation.z = 0.13 + Math.sin(t * 0.33) * 0.014;
      const beat = 0.5 + 0.5 * Math.sin(t * 1.35);
      const pulse = 1 + 0.34 * beat + 2.6 * flareAmount;
      core.scale.setScalar(1 + 0.12 * beat + 0.5 * flareAmount);
      halo.scale.setScalar(1 + 0.2 * beat + 0.85 * flareAmount);
      heartDecor.position.y = Math.sin(t * 0.9) * 0.014;
      setEmissiveIntensity(coreMaterial, HEART_HDR * pulse);
      setEmissiveIntensity(haloMaterial, HEART_HDR * 0.24 * pulse);
      oakUniforms(hullMaterial)?.setTime(t);
    },
    dispose() {
      hullGeometry.dispose();
      hullMaterial.dispose();
      coreGeometry.dispose();
      coreMaterial.dispose();
      haloGeometry.dispose();
      haloMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}


export const LURE_HDR = 3.6;

export interface AnglerOptions {
  tier: QualityTier;
  skin: AbyssSkin;
  lure: THREE.ColorRepresentation;
  hover: number;
  phase: number;
  scale: number;
  name?: string;
}

export interface AnglerHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  lureAnchor: THREE.Object3D;
  setTime(t: number): void;
  dispose(): void;
}

export function buildAngler(opts: AnglerOptions): AnglerHandle {
  const { tier, skin } = opts;
  const detail = tier === 'thumb' ? 0.45 : tier === 'medium' ? 0.7 : 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'anglerfish';

  const root = new THREE.Object3D();
  root.name = `${opts.name ?? 'angler'}Root`;
  group.add(root);

  const swim = new THREE.Object3D();
  swim.name = `${opts.name ?? 'angler'}Swim`;
  swim.position.y = opts.hover;
  swim.scale.setScalar(opts.scale);
  root.add(swim);

  const parts: ColoredPart[] = [];
  const radial = Math.max(5, Math.round(9 * detail));
  const dark = skin.anglerBody;
  const belly = blendTokens(skin.anglerBody, skin.rockLip, 0.35);

  {
    const body = new THREE.SphereGeometry(0.23, radial, Math.max(4, Math.round(7 * detail)));
    body.scale(1.35, 1, 0.92);
    parts.push({ geometry: body, color: dark });
    const jaw = new THREE.SphereGeometry(0.15, Math.max(4, radial - 2), Math.max(3, Math.round(5 * detail)));
    jaw.scale(1.25, 0.55, 0.85);
    jaw.translate(0.19, -0.11, 0);
    parts.push({ geometry: jaw, color: belly });
    const tail = new THREE.ConeGeometry(0.17, 0.3, Math.max(3, Math.round(5 * detail)), 1, true);
    tail.rotateZ(Math.PI / 2);
    tail.translate(-0.42, 0.02, 0);
    parts.push({ geometry: tail, color: dark });
    const fin = new THREE.ConeGeometry(0.09, 0.2, 3, 1, true);
    fin.rotateX(-0.3);
    fin.translate(-0.06, 0.24, 0);
    parts.push({ geometry: fin, color: dark });
    const rod = new THREE.CylinderGeometry(0.011, 0.02, 0.5, Math.max(3, radial - 4));
    rod.translate(0, 0.25, 0);
    rod.rotateZ(-0.72);
    rod.translate(0.06, 0.16, 0);
    parts.push({ geometry: rod, color: dark });
    if (tier !== 'thumb') {
      for (let i = 0; i < 4; i++) {
        const tooth = new THREE.ConeGeometry(0.014, 0.06, 3);
        tooth.rotateX(Math.PI);
        tooth.translate(0.22 + i * 0.045, -0.03, (i % 2 === 0 ? 1 : -1) * 0.055);
        parts.push({ geometry: tooth, color: skin.rockLip });
      }
    }
  }

  const bodyGeometry = mergeColoredParts(parts, 'anglerBody');
  const bodyMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.88,
    rimColor: opts.lure,
    rimStrength: 0.8,
    rimPower: 2.4,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.22,
    tier,
    name: 'anglerBodyMaterial',
  });
  const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
  bodyMesh.name = 'anglerBodyMesh';
  bodyMesh.castShadow = false;
  bodyMesh.receiveShadow = tier !== 'thumb';
  swim.add(bodyMesh);

  const lureAnchor = new THREE.Object3D();
  lureAnchor.name = `${opts.name ?? 'angler'}Lure`;
  lureAnchor.position.set(0.42, 0.44, 0.01);
  swim.add(lureAnchor);

  const escaGeometry = new THREE.IcosahedronGeometry(0.055, tier === 'thumb' ? 0 : 1);
  const escaMaterial: EmissiveMaterial = emissiveMaterial(opts.lure, LURE_HDR, { name: 'anglerEsca' });
  const esca = new THREE.Mesh(escaGeometry, escaMaterial);
  esca.name = 'anglerEsca';
  lureAnchor.add(esca);

  const glowGeometry = new THREE.IcosahedronGeometry(0.16, tier === 'thumb' ? 0 : 1);
  const glowMaterial: EmissiveMaterial = emissiveMaterial(opts.lure, LURE_HDR * 0.22, {
    transparent: true,
    opacity: 0.3,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'anglerEscaGlow',
  });
  const glow = new THREE.Mesh(glowGeometry, glowMaterial);
  glow.name = 'anglerEscaGlow';
  lureAnchor.add(glow);

  return {
    group,
    root,
    lureAnchor,
    setTime(t) {
      const p = opts.phase;
      swim.position.x = Math.sin(t * 0.21 + p) * 0.55;
      swim.position.z = Math.cos(t * 0.17 + p * 1.7) * 0.42;
      swim.position.y = opts.hover + Math.sin(t * 0.29 + p * 0.6) * 0.17;
      swim.rotation.y = Math.PI * 0.15 + Math.sin(t * 0.21 + p) * 0.5;
      swim.rotation.z = Math.sin(t * 0.26 + p) * 0.08;
      const flicker = 0.72 + 0.28 * Math.sin(t * 1.9 + p * 3.1) + 0.1 * Math.sin(t * 5.3 + p);
      esca.scale.setScalar(0.9 + 0.16 * Math.sin(t * 1.9 + p * 3.1));
      glow.scale.setScalar(1 + 0.22 * Math.sin(t * 1.9 + p * 3.1));
      setEmissiveIntensity(escaMaterial, LURE_HDR * flicker);
      setEmissiveIntensity(glowMaterial, LURE_HDR * 0.22 * flicker);
      oakUniforms(bodyMaterial)?.setTime(t);
    },
    dispose() {
      bodyGeometry.dispose();
      bodyMaterial.dispose();
      escaGeometry.dispose();
      escaMaterial.dispose();
      glowGeometry.dispose();
      glowMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}


export interface DiveBubbleOptions {
  tier: QualityTier;
  accent: THREE.ColorRepresentation;
  water: THREE.ColorRepresentation;
  radius?: number;
  name?: string;
}

export interface DiveBubbleHandle {
  group: THREE.Group;
  setGain(gain01: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildDiveBubble(opts: DiveBubbleOptions): DiveBubbleHandle {
  const tier = opts.tier;
  const radius = opts.radius ?? 0.36;
  const group = new THREE.Group();
  group.name = opts.name ?? 'diveBubble';

  const decor = new THREE.Object3D();
  decor.name = 'diveBubbleDecor';
  group.add(decor);

  const shellGeometry = new THREE.IcosahedronGeometry(radius, tier === 'thumb' ? 1 : 2);
  const shellMaterial: OakMaterial = oakMaterial({
    color: opts.water,
    roughness: 0.15,
    metalness: 0,
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
    side: THREE.DoubleSide,
    toonSteps: 0,
    rimColor: opts.accent,
    rimStrength: 1.5,
    rimPower: 2.4,
    tier,
    name: 'diveBubbleShell',
  });
  const shell = new THREE.Mesh(shellGeometry, shellMaterial);
  shell.name = 'diveBubbleShell';
  shell.renderOrder = 3;
  decor.add(shell);

  const innerGeometry = new THREE.IcosahedronGeometry(radius * 0.82, tier === 'thumb' ? 0 : 1);
  const innerMaterial: EmissiveMaterial = emissiveMaterial(opts.accent, 0.5, {
    transparent: true,
    opacity: 0.13,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'diveBubbleInner',
  });
  const inner = new THREE.Mesh(innerGeometry, innerMaterial);
  inner.name = 'diveBubbleInner';
  decor.add(inner);

  let gain = 0;

  return {
    group,
    setGain(gain01) {
      gain = clamp01(gain01);
      oakUniforms(shellMaterial)?.setRim(0.5 + 3.4 * gain);
      shellMaterial.opacity = 0.16 + 0.16 * gain;
      setEmissiveIntensity(innerMaterial, 0.16 + 1.7 * gain);
      innerMaterial.opacity = 0.05 + 0.2 * gain;
    },
    setTime(t) {
      const wobble = Math.sin(t * 1.7) * 0.035 + Math.sin(t * 2.9) * 0.018;
      shell.scale.set(1 + wobble, 1 - wobble * 0.8, 1 + wobble * 0.5);
      inner.scale.setScalar(1 - wobble * 0.4);
      decor.position.y = Math.sin(t * 1.15) * 0.012;
      decor.rotation.y = t * 0.28;
      oakUniforms(shellMaterial)?.setTime(t);
      inner.scale.multiplyScalar(1 + 0.05 * gain * Math.sin(t * 2.3));
    },
    dispose() {
      shellGeometry.dispose();
      shellMaterial.dispose();
      innerGeometry.dispose();
      innerMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
