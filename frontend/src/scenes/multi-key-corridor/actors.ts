import { GoalBeacon } from '@kit/grid';
import { emissiveMaterial, oakMaterial, oakUniforms, setEmissiveIntensity, type EmissiveMaterial, type OakMaterial } from '@kit/materials';
import { blendTokens, clamp01 } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { Palette } from '@kit/tokens';
import * as THREE from 'three';
import { PORTAL_LINTEL_Y, type TempleSkin } from './world';

export const GATE_SECONDS = 0.5;
export const GATE_FLARE_SECONDS = 0.6;

export interface GateHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly progress: number;
  setProgress(progress: number): void;
  setFlare(amount: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface GateOptions {
  palette: Palette;
  skin: TempleSkin;
  facesX: boolean;
  stone: OakMaterial;
  name?: string;
}

function keyGeometry(index: number): THREE.BufferGeometry {
  if (index === 1) return new THREE.TorusGeometry(0.11, 0.035, 6, 12);
  const geometry = index === 0 ? new THREE.IcosahedronGeometry(0.12, 0) : new THREE.OctahedronGeometry(0.13, 0);
  geometry.scale(index === 0 ? 0.72 : 1, 1.55, 0.72);
  return geometry;
}

function gateLeaf(skin: TempleSkin, index: number): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const frame = new THREE.BoxGeometry(0.44, 1.26, 0.055);
  frame.translate(0, 0.67, 0);
  parts.push({ geometry: frame, color: index === 0 ? skin.bramble : index === 1 ? skin.riverBed : skin.stoneLight });
  for (let i = 0; i < 5; i++) {
    const y = 0.18 + i * 0.23;
    const rib = index === 0
      ? new THREE.CylinderGeometry(0.025, 0.035, 0.53, 5)
      : new THREE.IcosahedronGeometry(0.18, 0);
    if (index === 0) {
      rib.rotateZ(0.8 + (i % 2) * 0.75);
      rib.translate(0, y, 0.045);
    } else {
      rib.scale(1.22, 0.55, 0.45);
      rib.translate(0, y, 0.05);
    }
    parts.push({ geometry: rib, color: index === 0 ? skin.brambleLeaf : skin.stoneDark });
  }
  return mergeColoredParts(parts, 'templeGateLeaf');
}

function buildGate(opts: GateOptions, index: number): GateHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? `templeGate${index}`;
  const root = new THREE.Group();
  group.add(root);
  const pivot = new THREE.Group();
  pivot.rotation.y = opts.facesX ? Math.PI / 2 : 0;
  root.add(pivot);
  const color = opts.skin.keys[index] as THREE.Color;
  const geometry = gateLeaf(opts.skin, index);
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: index === 1 ? 0.58 : 0.86,
    metalness: index === 1 ? 0.12 : 0,
    transparent: index === 2,
    opacity: index === 2 ? 0.66 : 1,
    emissive: color,
    emissiveIntensity: index === 2 ? 0.16 : 0,
    depthWrite: index !== 2,
  });
  const clock = { value: 0 };
  if (index === 2) {
    material.onBeforeCompile = (shader) => {
      shader.uniforms.uMistTime = clock;
      shader.vertexShader = 'varying vec3 vGatePosition;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvGatePosition = position;');
      shader.fragmentShader = 'uniform float uMistTime; varying vec3 vGatePosition;\n' + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace('#include <alphamap_fragment>', `
        #include <alphamap_fragment>
        float fold = sin(vGatePosition.y * 14.0 + sin(vGatePosition.x * 22.0) + uMistTime * 0.7);
        diffuseColor.a *= 0.5 + 0.25 * fold;
      `);
    };
  }
  const leaves: THREE.Group[] = [];
  for (const side of [-1, 1]) {
    const hinge = new THREE.Group();
    hinge.position.x = side * 0.46;
    const leaf = new THREE.Mesh(geometry, material);
    leaf.position.x = -side * 0.22;
    leaf.castShadow = index !== 2;
    hinge.add(leaf);
    pivot.add(hinge);
    leaves.push(hinge);
  }
  const emblemGeometry = keyGeometry(index);
  const emblemMaterial = emissiveMaterial(color, 1.15);
  const emblem = new THREE.Mesh(emblemGeometry, emblemMaterial);
  emblem.position.y = PORTAL_LINTEL_Y + 0.09;
  emblem.scale.setScalar(1.1);
  pivot.add(emblem);
  let progress = 0;
  return {
    group,
    root,
    get progress() { return progress; },
    setProgress(value) {
      progress = clamp01(value);
      const open = progress * progress * (3 - 2 * progress);
      leaves.forEach((leaf, i) => { leaf.rotation.y = (i === 0 ? -1 : 1) * open * 1.5; });
    },
    setFlare(amount) {
      material.emissiveIntensity = (index === 2 ? 0.16 : 0) + amount * 0.5;
      setEmissiveIntensity(emblemMaterial, 1.15 + amount);
    },
    setTime(t) { clock.value = t; },
    dispose() {
      geometry.dispose();
      material.dispose();
      emblemGeometry.dispose();
      emblemMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

export const buildBrambleGate = (opts: GateOptions): GateHandle => buildGate(opts, 0);
export const buildRiverGate = (opts: GateOptions): GateHandle => buildGate(opts, 1);
export const buildMistGate = (opts: GateOptions): GateHandle => buildGate(opts, 2);

export interface KeyAltarOptions {
  palette: Palette;
  skin: TempleSkin;
  index: number;
  stone: OakMaterial;
  name?: string;
}

export interface KeyAltarHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly gemHeight: number;
  setCarried(carried: boolean): void;
  setGrow(grow: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildKeyAltar(opts: KeyAltarOptions): KeyAltarHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? `keyAltar${opts.index}`;
  const root = new THREE.Group();
  const decor = new THREE.Group();
  group.add(root);
  root.add(decor);
  const pedestalGeometry = new THREE.CylinderGeometry(0.16, 0.23, 0.19, 8);
  const pedestal = new THREE.Mesh(pedestalGeometry, opts.stone);
  pedestal.position.y = 0.095;
  pedestal.castShadow = true;
  decor.add(pedestal);
  const geometry = keyGeometry(opts.index);
  const material = emissiveMaterial(opts.skin.keys[opts.index] as THREE.Color, 1.5);
  const token = new THREE.Mesh(geometry, material);
  token.position.y = 0.42;
  decor.add(token);
  return {
    group,
    root,
    gemHeight: 0.42,
    setCarried(carried) { token.visible = !carried; },
    setGrow(grow) { decor.scale.setScalar(clamp01(grow)); },
    setTime(t) {
      token.rotation.y = t * 0.65;
      token.position.y = 0.42 + Math.sin(t * 1.7 + opts.index) * 0.025;
    },
    update() {},
    dispose() {
      geometry.dispose();
      material.dispose();
      pedestalGeometry.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

export interface CarryRackHandle {
  group: THREE.Group;
  setCarried(index: number, carried: boolean): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildCarryRack(skin: TempleSkin): CarryRackHandle {
  const group = new THREE.Group();
  group.name = 'keyCarryRack';
  const tokens = skin.keys.map((color, index) => {
    const token = new THREE.Mesh(keyGeometry(index), emissiveMaterial(color, 1.4));
    token.scale.setScalar(0.5);
    token.position.set((index - 1) * 0.17, 0.2, 0);
    token.visible = false;
    group.add(token);
    return token;
  });
  return {
    group,
    setCarried(index, carried) {
      const token = tokens[index];
      if (token) token.visible = carried;
    },
    setTime(t) { group.rotation.y = t * 0.3; },
    dispose() {
      for (const token of tokens) {
        token.geometry.dispose();
        token.material.dispose();
      }
      group.removeFromParent();
      group.clear();
    },
  };
}

export interface ShrineOptions {
  palette: Palette;
  skin: TempleSkin;
  stone: OakMaterial;
  name?: string;
}

export interface ShrineHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly crownHeight: number;
  setState(near01: number, grow01: number): void;
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildShrine(opts: ShrineOptions): ShrineHandle {
  const { palette, skin } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'sanctumShrine';
  const root = new THREE.Object3D();
  root.name = 'shrineRoot';
  group.add(root);
  const decor = new THREE.Group();
  decor.name = 'shrineDecor';
  root.add(decor);

  const plinthGeometry = makeShrineGeometry(skin);
  const plinth = new THREE.Mesh(plinthGeometry, opts.stone);
  plinth.name = 'shrinePlinth';
  plinth.castShadow = true;
  plinth.receiveShadow = true;
  decor.add(plinth);

  const beacon = new GoalBeacon({
    color: skin.relic,
    height: 1.0,
    radius: 0.11,
    intensity: 0.5,
    name: 'shrineGlow',
  });
  decor.add(beacon.group);

  const CROWN = 0.86;
  const relicGeometry = new THREE.OctahedronGeometry(0.125, 1);
  const RELIC_HDR = 1.3;
  const relicMaterial: EmissiveMaterial = emissiveMaterial(skin.relic, RELIC_HDR, { name: 'shrineRelic' });
  const relic = new THREE.Mesh(relicGeometry, relicMaterial);
  relic.name = 'shrineRelic';
  relic.position.y = CROWN;
  decor.add(relic);

  const bladeGeometry = new THREE.BoxGeometry(0.03, 0.2, 0.06);
  const bladeMaterial: OakMaterial = oakMaterial({
    color: blendTokens(skin.relic, palette.groundB, 0.25),
    roughness: 0.6,
    metalness: 0.15,
    flatShading: true,
    rimColor: skin.relic,
    rimStrength: 0.5,
    toonSteps: 2,
    toonSoftness: 0.36,
    toonFloor: 0.34,
    name: 'shrineBlades',
  });
  const blades = new THREE.InstancedMesh(bladeGeometry, bladeMaterial, 4);
  blades.name = 'shrineBlades';
  blades.castShadow = false;
  blades.frustumCulled = false;
  {
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2;
      position.set(Math.cos(a) * 0.17, CROWN, Math.sin(a) * 0.17);
      euler.set(0, -a, 0.5);
      quaternion.setFromEuler(euler);
      matrix.compose(position, quaternion, scale);
      blades.setMatrixAt(i, matrix);
    }
    blades.instanceMatrix.needsUpdate = true;
  }
  decor.add(blades);

  let near = 0;

  return {
    group,
    root,
    crownHeight: CROWN,
    setState(near01, grow01) {
      near = clamp01(near01);
      const grow = clamp01(grow01);
      decor.scale.setScalar(grow);
      relic.scale.setScalar(0.82 + 0.3 * near);
      setEmissiveIntensity(relicMaterial, RELIC_HDR * (0.6 + 0.7 * near) * (0.3 + 0.7 * grow));
    },
    flare(strength = 1) {
      beacon.flare(strength);
    },
    setTime(t) {
      relic.rotation.y = t * 0.8;
      relic.position.y = CROWN + Math.sin(t * 1.5) * 0.03;
      blades.rotation.y = -t * 0.35;
      decor.rotation.y = Math.sin(t * 0.3) * 0.04 * (0.3 + 0.7 * near);
      oakUniforms(bladeMaterial)?.setTime(t);
    },
    update(dt, time) {
      beacon.update(dt, time);
    },
    dispose() {
      beacon.dispose();
      blades.dispose();
      bladeGeometry.dispose();
      bladeMaterial.dispose();
      relicGeometry.dispose();
      relicMaterial.dispose();
      plinthGeometry.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function makeShrineGeometry(skin: TempleSkin): THREE.BufferGeometry {
  const radial = 8;
  const parts: ColoredPart[] = [];
  const step0 = new THREE.CylinderGeometry(0.33, 0.4, 0.11, radial, 1);
  step0.translate(0, 0.055, 0);
  parts.push({ geometry: step0, color: skin.stoneDark });
  const step1 = new THREE.CylinderGeometry(0.25, 0.31, 0.12, radial, 1);
  step1.translate(0, 0.17, 0);
  parts.push({ geometry: step1, color: skin.stoneLight });
  const drum = new THREE.CylinderGeometry(0.125, 0.18, 0.32, radial, 1);
  drum.translate(0, 0.39, 0);
  parts.push({ geometry: drum, color: skin.stoneLight });
  const lip = new THREE.CylinderGeometry(0.22, 0.14, 0.08, radial, 1);
  lip.translate(0, 0.59, 0);
  parts.push({ geometry: lip, color: skin.stoneDark });
  for (let i = 0; i < radial; i++) {
    const a = (i / radial) * Math.PI * 2;
    const petal = new THREE.ConeGeometry(0.06, 0.16, 4, 1);
    petal.rotateX(0.95);
    petal.rotateY(a);
    petal.translate(Math.cos(a) * 0.2, 0.64, Math.sin(a) * 0.2);
    parts.push({ geometry: petal, color: skin.stoneLight });
  }
  return mergeColoredParts(parts, 'shrinePlinth');
}
