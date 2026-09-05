import * as THREE from 'three';
import { emissiveMaterial, oakMaterial, setEmissiveIntensity } from '@kit/materials';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { CaveSkin } from './world';

function crystalGeometry(choice: 0 | 1): THREE.BufferGeometry {
  if (choice === 0) {
    const geometry = new THREE.OctahedronGeometry(0.53, 0);
    geometry.scale(0.8, 1.7, 0.8);
    return geometry;
  }
  const parts: ColoredPart[] = [];
  for (let i = 0; i < 3; i++) {
    const geometry = new THREE.OctahedronGeometry(0.32, 0);
    geometry.scale(0.8, i === 1 ? 2.3 : 1.5, 0.8);
    geometry.translate((i - 1) * 0.58, i === 1 ? 0.12 : -0.1, 0);
    parts.push({ geometry, color: 0xffffff });
  }
  return mergeColoredParts(parts, 'threePetalCrystal');
}

function stoneMaterial(skin: CaveSkin): THREE.MeshStandardMaterial {
  return oakMaterial({ color: skin.stone, roughness: 0.82, flatShading: true, rimColor: skin.rim, rimStrength: 0.3 });
}

export interface CueCrystalHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setCue(cue: 0 | 1): void;
  setGlow(gain: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildCueCrystal(opts: { skin: CaveSkin; cueColors: readonly [THREE.ColorRepresentation, THREE.ColorRepresentation] }): CueCrystalHandle {
  const group = new THREE.Group();
  group.name = 'startingCueLandmark';
  const plinthGeometry = new THREE.CylinderGeometry(0.78, 1.12, 0.64, 8);
  const stone = stoneMaterial(opts.skin);
  const plinth = new THREE.Mesh(plinthGeometry, stone);
  plinth.position.y = 0.22;
  plinth.castShadow = true;
  plinth.receiveShadow = true;
  group.add(plinth);
  const crystals = ([0, 1] as const).map(choice => {
    const material = emissiveMaterial(opts.cueColors[choice], 1.25, { name: `cueCrystal${choice}` });
    material.vertexColors = choice === 1;
    const mesh = new THREE.Mesh(crystalGeometry(choice), material);
    mesh.position.y = 1.45;
    mesh.scale.setScalar(1.15);
    mesh.visible = false;
    group.add(mesh);
    return mesh;
  });
  let cue: 0 | 1 = 0;
  let glow = 0;
  return {
    group,
    root: group,
    setCue(value) { cue = value; },
    setGlow(gain) {
      glow = gain;
      crystals.forEach((crystal, index) => { crystal.visible = gain > 0 && cue === index; });
      for (const crystal of crystals) setEmissiveIntensity(crystal.material, 1.25 * gain);
    },
    setTime(t) {
      for (const crystal of crystals) crystal.position.y = 1.45 + Math.sin(t * 1.1) * 0.055 * glow;
    },
    dispose() {
      for (const crystal of crystals) { crystal.geometry.dispose(); crystal.material.dispose(); }
      plinthGeometry.dispose();
      stone.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

export interface ExitMouthHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setGlow(gain: number, success?: boolean): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildExitMouth(opts: { skin: CaveSkin; color: THREE.ColorRepresentation; choice: 0 | 1 }): ExitMouthHandle {
  const group = new THREE.Group();
  group.name = opts.choice === 0 ? 'diamondAlcove' : 'tripletAlcove';
  const parts: ColoredPart[] = [];
  const base = new THREE.CylinderGeometry(1.35, 1.55, 0.36, 10);
  base.translate(0, 0.08, 0);
  parts.push({ geometry: base, color: 0xffffff });
  const arch = new THREE.TorusGeometry(1.15, 0.19, 5, 14, Math.PI);
  arch.translate(0, 1.72, -0.22);
  parts.push({ geometry: arch, color: 0xffffff });
  for (const side of [-1, 1]) {
    const pillar = new THREE.CylinderGeometry(0.19, 0.27, 1.75, 6);
    pillar.translate(side * 1.15, 0.86, -0.22);
    parts.push({ geometry: pillar, color: 0xffffff });
  }
  const architecture = mergeColoredParts(parts, 'limestoneAlcove');
  const stone = stoneMaterial(opts.skin);
  const frame = new THREE.Mesh(architecture, stone);
  frame.castShadow = true;
  frame.receiveShadow = true;
  const crystalMaterial = emissiveMaterial(opts.color, 1.25, { name: 'choiceCrystal' });
  crystalMaterial.vertexColors = opts.choice === 1;
  const crystal = new THREE.Mesh(crystalGeometry(opts.choice), crystalMaterial);
  crystal.position.y = 1.35;
  crystal.scale.setScalar(1.15);
  const ringGeometry = new THREE.TorusGeometry(0.94, 0.026, 4, 24);
  ringGeometry.rotateX(Math.PI / 2);
  const ringMaterial = emissiveMaterial(opts.color, 1.3);
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.position.y = 0.3;
  ring.visible = false;
  group.add(frame, crystal, ring);
  let gain = 1;
  return {
    group,
    root: group,
    setGlow(value, success = true) {
      gain = value;
      const selected = Math.max(0, value - 1);
      setEmissiveIntensity(crystalMaterial, success ? 1.25 * value : 1.25 * (1 - 0.9 * Math.min(1, selected / 0.8)));
      crystal.scale.setScalar(1.15 * (success ? 1 + selected * 0.08 : 1 - selected * 0.22));
      ring.visible = success && value > 1.01;
      setEmissiveIntensity(ringMaterial, (value - 1) * 1.3);
    },
    setTime(t) {
      crystal.position.y = 1.35 + Math.sin(t * 1.1) * 0.035;
      ring.scale.setScalar(1 + (gain - 1) * 0.08);
    },
    dispose() {
      architecture.dispose();
      stone.dispose();
      crystal.geometry.dispose();
      crystalMaterial.dispose();
      ringGeometry.dispose();
      ringMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

export function buildTraveler(): { group: THREE.Group; dispose(): void } {
  const group = new THREE.Group();
  group.name = 'neutralCaveLantern';
  group.position.y = 0.5;
  const shellGeometry = new THREE.CylinderGeometry(0.21, 0.17, 0.33, 8);
  const parts: ColoredPart[] = [];
  const capGeometry = new THREE.ConeGeometry(0.27, 0.14, 8);
  capGeometry.translate(0, 0.23, 0);
  parts.push({ geometry: capGeometry, color: 0xffffff });
  for (let i = 0; i < 4; i++) {
    const angle = Math.PI / 4 + i * Math.PI / 2;
    const bar = new THREE.BoxGeometry(0.028, 0.38, 0.028);
    bar.translate(Math.cos(angle) * 0.18, 0, Math.sin(angle) * 0.18);
    parts.push({ geometry: bar, color: 0xffffff });
  }
  const rim = new THREE.TorusGeometry(0.19, 0.026, 4, 12);
  rim.rotateX(Math.PI / 2);
  rim.translate(0, -0.18, 0);
  parts.push({ geometry: rim, color: 0xffffff });
  const handle = new THREE.TorusGeometry(0.11, 0.022, 4, 10);
  handle.translate(0, 0.39, 0);
  parts.push({ geometry: handle, color: 0xffffff });
  const frameGeometry = mergeColoredParts(parts, 'travelerLanternFrame');
  const frame = oakMaterial({ color: 0x9a7950, metalness: 0.48, roughness: 0.35 });
  const flame = emissiveMaterial(0xffe5b3, 1.7);
  group.scale.setScalar(1.35);
  group.add(new THREE.Mesh(shellGeometry, flame), new THREE.Mesh(frameGeometry, frame));
  return {
    group,
    dispose() {
      shellGeometry.dispose();
      frameGeometry.dispose();
      frame.dispose();
      flame.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
