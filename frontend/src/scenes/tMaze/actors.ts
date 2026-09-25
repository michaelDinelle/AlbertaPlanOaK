import * as THREE from 'three/webgpu';
import { emissiveMaterial, oakMaterial, setEmissiveIntensity } from '@kit/materials';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { CaveSkin } from './world';

export const CUE_NAMES = ['cyan spire', 'magenta triad', 'amber twins', 'green crown'] as const;
export const CUE_EXTRA_COLORS = [0xffb347, 0x8cf08a] as const;

function shard(radius: number, height: number, x: number, y: number, z: number, lean = 0, turn = 0): ColoredPart {
  const geometry = new THREE.OctahedronGeometry(radius, 0);
  geometry.scale(0.8, height, 0.8);
  geometry.rotateZ(lean);
  geometry.rotateY(turn);
  geometry.translate(x, y, z);
  return { geometry, color: 0xffffff };
}

function crystalGeometry(cue: number): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  if (cue === 0) parts.push(shard(0.53, 1.7, 0, 0, 0));
  else if (cue === 1) for (let i = 0; i < 3; i++) parts.push(shard(0.32, i === 1 ? 2.3 : 1.5, (i - 1) * 0.58, i === 1 ? 0.12 : -0.1, 0));
  else if (cue === 2) for (const side of [-1, 1]) parts.push(shard(0.4, 1.9, side * 0.3, 0, 0, -side * 0.3));
  else for (let i = 0; i < 4; i++) {
    const angle = Math.PI / 4 + i * Math.PI / 2;
    parts.push(shard(0.27, 1.55, Math.cos(angle) * 0.4, -0.08, Math.sin(angle) * 0.4, 0.28, Math.PI - angle));
  }
  return mergeColoredParts(parts, `cueCrystal${cue}`);
}

function stoneMaterial(skin: CaveSkin) {
  return oakMaterial({ color: skin.stone, roughness: 0.82, flatShading: true, rimColor: skin.rim, rimStrength: 0.3, name: 'limestone' });
}

export interface CueCrystalHandle {
  group: THREE.Group;
  setCue(cue: number, seen: boolean): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildCueCrystal(skin: CaveSkin, colors: readonly number[]): CueCrystalHandle {
  const group = new THREE.Group();
  group.name = 'startingCueLandmark';
  const plinthGeometry = new THREE.CylinderGeometry(0.78, 1.12, 0.64, 8);
  const stone = stoneMaterial(skin);
  const plinth = new THREE.Mesh(plinthGeometry, stone);
  plinth.position.y = 0.22;
  plinth.castShadow = true;
  plinth.receiveShadow = true;
  group.add(plinth);
  const crystals = colors.map((color, cue) => {
    const mesh = new THREE.Mesh(crystalGeometry(cue), emissiveMaterial(color, 1.25, { name: `cueCrystal${cue}` }));
    mesh.position.y = 1.45;
    mesh.scale.setScalar(1.15);
    mesh.visible = false;
    group.add(mesh);
    return mesh;
  });
  let glow = 0;
  return {
    group,
    setCue(cue, seen) {
      glow = seen ? 1 : 0;
      crystals.forEach((crystal, index) => { crystal.visible = seen && cue === index; });
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
  crystal: THREE.Object3D;
  setGlow(gain: number, success: boolean): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildExitMouth(skin: CaveSkin, color: number, cue: number): ExitMouthHandle {
  const group = new THREE.Group();
  group.name = `exitAlcove${cue}`;
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
  const stone = stoneMaterial(skin);
  const frame = new THREE.Mesh(architecture, stone);
  frame.castShadow = true;
  frame.receiveShadow = true;
  const crystalMaterial = emissiveMaterial(color, 1.25, { name: `choiceCrystal${cue}` });
  const crystal = new THREE.Mesh(crystalGeometry(cue), crystalMaterial);
  crystal.position.y = 1.35;
  crystal.scale.setScalar(1.15);
  const ringGeometry = new THREE.TorusGeometry(0.94, 0.026, 4, 24);
  ringGeometry.rotateX(Math.PI / 2);
  const ringMaterial = emissiveMaterial(color, 1.3, { name: `choiceRing${cue}` });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.position.y = 0.3;
  ring.visible = false;
  const light = new THREE.PointLight(color, 1.1, 12, 2);
  light.name = `alcoveLight${cue}`;
  light.position.y = 1.5;
  group.add(frame, crystal, ring, light);
  let gain = 1;
  return {
    group,
    crystal,
    setGlow(value, success) {
      gain = value;
      const selected = Math.max(0, value - 1);
      setEmissiveIntensity(crystalMaterial, success ? 1.25 * value : 1.25 * (1 - 0.9 * Math.min(1, selected / 0.8)));
      crystal.scale.setScalar(1.15 * (success ? 1 + selected * 0.08 : 1 - selected * 0.22));
      ring.visible = success && value > 1.01;
      setEmissiveIntensity(ringMaterial, (value - 1) * 1.3);
      light.intensity = 1.1 * (success ? value : 1 - 0.85 * Math.min(1, selected / 0.8));
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
      light.dispose();
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
  const cap = new THREE.ConeGeometry(0.27, 0.14, 8);
  cap.translate(0, 0.23, 0);
  parts.push({ geometry: cap, color: 0xffffff });
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
  const frame = oakMaterial({ color: 0x9a7950, metalness: 0.48, roughness: 0.35, name: 'lanternBrass' });
  const flame = emissiveMaterial(0xffe5b3, 1.7, { name: 'lanternFlame' });
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
