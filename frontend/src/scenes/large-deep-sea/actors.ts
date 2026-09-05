import { emissiveMaterial, oakMaterial, oakUniforms, setEmissiveIntensity } from '@kit/materials';
import type { EmissiveMaterial, OakMaterial } from '@kit/materials';
import { blendTokens, clamp01 } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import * as THREE from 'three';
import type { AbyssSkin } from './world';

export const HEART_Y = 0.62;
export const HEART_HDR = 2.6;
export const WRECK_SCALE = 1.35;

export interface WreckOptions {
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
  const { skin } = opts;
  const detail = 1;
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

  const RIBS = 7;
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
    name: 'wreckTimber',
  });
  const hullMesh = new THREE.Mesh(hullGeometry, hullMaterial);
  hullMesh.name = 'wreckTimberMesh';
  hullMesh.castShadow = true;
  hullMesh.receiveShadow = true;
  hull.add(hullMesh);

  const heartAnchor = new THREE.Object3D();
  heartAnchor.name = 'wreckHeartAnchor';
  heartAnchor.position.set(0, HEART_Y, 0);
  root.add(heartAnchor);

  const heartDecor = new THREE.Object3D();
  heartDecor.name = 'wreckHeartDecor';
  heartDecor.scale.setScalar(WRECK_SCALE);
  heartAnchor.add(heartDecor);

  const coreGeometry = new THREE.IcosahedronGeometry(0.11, 1);
  const coreMaterial: EmissiveMaterial = emissiveMaterial(skin.heart, HEART_HDR, { name: 'wreckHeartCore' });
  const core = new THREE.Mesh(coreGeometry, coreMaterial);
  core.name = 'wreckHeartCore';
  heartDecor.add(core);

  const haloGeometry = new THREE.IcosahedronGeometry(0.23, 1);
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

export interface DiveBubbleOptions {
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
  const radius = opts.radius ?? 0.36;
  const group = new THREE.Group();
  group.name = opts.name ?? 'diveBubble';

  const decor = new THREE.Object3D();
  decor.name = 'diveBubbleDecor';
  group.add(decor);

  const shellGeometry = new THREE.IcosahedronGeometry(radius, 2);
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
    name: 'diveBubbleShell',
  });
  const shell = new THREE.Mesh(shellGeometry, shellMaterial);
  shell.name = 'diveBubbleShell';
  shell.renderOrder = 3;
  decor.add(shell);

  const innerGeometry = new THREE.IcosahedronGeometry(radius * 0.82, 1);
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
