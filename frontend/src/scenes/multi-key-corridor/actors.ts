
import { GoalBeacon, KeyToken } from '@kit/grid';
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { fbmField, makeDataTexture } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import { makeWater, type WaterHandle } from '@kit/water';
import * as THREE from 'three';
import {
  PORTAL_LINTEL_Y,
  blendTokens,
  clamp01,
  mergeColoredParts,
  type ColoredPart,
  type TempleSkin,
} from './world';

export const GATE_SECONDS = 0.5;

export const GATE_FLARE_SECONDS = 0.6;


export interface GateHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly progress: number;
  setProgress(progress01: number): void;
  setFlare(amount01: number): void;
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

function ease(t: number): number {
  const c = clamp01(t);
  return c * c * (3 - 2 * c);
}


export function buildBrambleGate(opts: GateOptions): GateHandle {
  const { palette, skin } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'brambleGate';
  const root = new THREE.Object3D();
  root.name = 'brambleGateRoot';
  group.add(root);

  const pivot = new THREE.Group();
  pivot.name = 'brambleGatePivot';
  pivot.rotation.y = opts.facesX ? 0 : Math.PI / 2;
  root.add(pivot);

  const canes = 8;
  const geometry = makeCaneGeometry( skin.bramble, skin.brambleLeaf);
  const material: OakMaterial = oakMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    roughness: 0.88,
    flatShading: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.3,
    wind: { amplitude: 0.012, frequency: 1.35, height: 1.1 },
    name: 'brambleCane',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, canes);
  mesh.name = 'brambleCanes';
  mesh.castShadow = true;
  mesh.receiveShadow = false;
  mesh.frustumCulled = false;
  pivot.add(mesh);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'ZYX');
  const scale = new THREE.Vector3(1, 1, 1);
  const flareColor = (skin.keys[0] ?? new THREE.Color(0xffffff)).clone();
  let progress = 0;
  let flareAmount = 0;

  const apply = (): void => {
    const e = ease(progress);
    for (let i = 0; i < canes; i++) {
      const t = canes > 1 ? i / (canes - 1) : 0.5;
      const lean = (i % 2 === 0 ? 1 : -1) * (0.12 + 0.1 * t);
      const y = 0.1 + t * (PORTAL_LINTEL_Y - 0.2);
      const side = i % 2 === 0 ? -1 : 1;
      position.set(0, y + e * 0.06 * side, side * e * 0.36);
      euler.set(lean * (1 - e) + e * side * 1.1, e * side * 0.5, 0);
      quaternion.setFromEuler(euler);
      const span = 1 - 0.86 * e;
      scale.set(1 - 0.3 * e, 1 - 0.3 * e, span);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  };
  apply();

  const applyFlare = (): void => {
    material.emissive.copy(flareColor).multiplyScalar(flareAmount * 0.6);
  };
  applyFlare();

  return {
    group,
    root,
    get progress() {
      return progress;
    },
    setProgress(p) {
      const v = clamp01(p);
      if (v === progress) return;
      progress = v;
      apply();
    },
    setFlare(amount) {
      const v = clamp01(amount);
      if (v === flareAmount) return;
      flareAmount = v;
      applyFlare();
    },
    setTime(t) {
      oakUniforms(material)?.setTime(t);
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

function makeCaneGeometry(
  dark: THREE.ColorRepresentation,
  light: THREE.ColorRepresentation,
): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const bar = new THREE.CylinderGeometry(0.03, 0.036, 0.94, 6, 1);
  bar.rotateX(Math.PI / 2);
  parts.push({ geometry: bar, color: dark });

  const thorns = 4;
  for (let i = 0; i < thorns; i++) {
    const t = (i + 0.5) / thorns;
    const thorn = new THREE.ConeGeometry(0.022, 0.075, 4, 1);
    thorn.rotateZ(i % 2 === 0 ? Math.PI / 2.4 : -Math.PI / 2.4);
    thorn.rotateY(i * 1.9);
    thorn.translate(0, 0, (t - 0.5) * 0.86);
    parts.push({ geometry: thorn, color: dark });
  }

  const leaves = 6;
  for (let i = 0; i < leaves; i++) {
    const t = (i + 0.5) / leaves;
    const side = i % 2 === 0 ? 1 : -1;
    const leaf = new THREE.CircleGeometry(0.085, 4);
    leaf.scale(1, 0.52, 1);
    leaf.rotateX(-Math.PI / 2 + side * 0.35);
    leaf.rotateY(side * 0.7 + i * 0.5);
    leaf.translate(side * 0.075, 0.02, (t - 0.5) * 0.82);
    parts.push({ geometry: leaf, color: i % 3 === 0 ? dark : light });
  }
  return mergeColoredParts(parts, 'brambleCane');
}


export function buildRiverGate(opts: GateOptions): GateHandle {
  const { palette, skin } = opts;
  const underwater = TOKENS.palettes.underwater;
  const group = new THREE.Group();
  group.name = opts.name ?? 'riverGate';
  const root = new THREE.Object3D();
  root.name = 'riverGateRoot';
  group.add(root);

  const pivot = new THREE.Group();
  pivot.name = 'riverGatePivot';
  pivot.rotation.y = opts.facesX ? 0 : Math.PI / 2;
  root.add(pivot);

  const CAUSTIC_IDLE = 0.26;
  const CAUSTIC_FLARE = 1.4;

  const basinGeometry = makeBasinGeometry(skin);
  const basinMesh = new THREE.Mesh(basinGeometry, opts.stone);
  basinMesh.name = 'riverBasin';
  basinMesh.castShadow = false;
  basinMesh.receiveShadow = true;
  pivot.add(basinMesh);

  const water: WaterHandle = makeWater({
    size: [0.99, 2.1],
    palette,
    segments: 26,
    sky: palette.skyHorizon,
    amplitude: 0.011,
    waveScale: 2.1,
    speed: 0.9,
    fresnel: 0,
    caustics: CAUSTIC_IDLE,
    depthFalloff: 45,
    opacity: 0.72,
    level: 0.035,
    name: 'riverWater',
  });
  water.setColors(
    blendTokens(underwater.skyTop, underwater.groundB, 0.32),
    underwater.skyBottom,
  );
  water.uniforms.uSpecular.value = 0;
  pivot.add(water.mesh);

  const stoneCount = 3;
  const stoneGeometry = new THREE.CylinderGeometry(0.19, 0.22, 0.14, 7, 1);
  const stoneMaterial: OakMaterial = oakMaterial({
    color: blendTokens(skin.stoneLight, skin.riverBed, 0.4),
    roughness: 0.95,
    flatShading: true,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.28,
    name: 'riverSteppingStone',
  });
  const stones = new THREE.InstancedMesh(stoneGeometry, stoneMaterial, stoneCount);
  stones.name = 'riverSteppingStones';
  stones.castShadow = true;
  stones.receiveShadow = true;
  stones.frustumCulled = false;
  pivot.add(stones);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3(1, 1, 1);
  let progress = 0;
  let flareAmount = 0;
  const SUNK_Y = -0.03;
  const SURFACE_Y = 0.1;

  const apply = (): void => {
    const e = ease(progress);
    for (let i = 0; i < stoneCount; i++) {
      const t = (i - (stoneCount - 1) / 2) / stoneCount;
      const stagger = (i % 2 === 0 ? 1 : -1) * 0.11;
      const local = clamp01((e - i * 0.12) / 0.76);
      position.set(t * 0.66, SUNK_Y + (SURFACE_Y - SUNK_Y) * ease(local), stagger);
      euler.set(0, i * 0.7, 0);
      quaternion.setFromEuler(euler);
      const s = 0.72 + 0.28 * ease(local);
      scale.set(s, 1, s);
      matrix.compose(position, quaternion, scale);
      stones.setMatrixAt(i, matrix);
    }
    stones.instanceMatrix.needsUpdate = true;
  };
  apply();

  const applyFlare = (): void => {
    water.uniforms.uCausticStrength.value = CAUSTIC_IDLE + CAUSTIC_FLARE * flareAmount;
  };
  applyFlare();

  return {
    group,
    root,
    get progress() {
      return progress;
    },
    setProgress(p) {
      const v = clamp01(p);
      if (v === progress) return;
      progress = v;
      apply();
    },
    setFlare(amount) {
      const v = clamp01(amount);
      if (v === flareAmount) return;
      flareAmount = v;
      applyFlare();
    },
    setTime(t) {
      water.setTime(t);
      oakUniforms(stoneMaterial)?.setTime(t);
    },
    dispose() {
      stones.dispose();
      stoneGeometry.dispose();
      stoneMaterial.dispose();
      water.dispose();
      basinGeometry.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function makeBasinGeometry(skin: TempleSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const bed = new THREE.BoxGeometry(0.99, 0.06, 2.1);
  bed.translate(0, 0.005, 0);
  parts.push({ geometry: bed, color: skin.riverBed });
  return mergeColoredParts(parts, 'riverBasin');
}


export function buildMistGate(opts: GateOptions): GateHandle {
  const { palette } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'mistGate';
  const root = new THREE.Object3D();
  root.name = 'mistGateRoot';
  group.add(root);

  const pivot = new THREE.Group();
  pivot.name = 'mistGatePivot';
  pivot.rotation.y = opts.facesX ? 0 : Math.PI / 2;
  root.add(pivot);

  const layers = 2;
  const halves = 2;
  const geometry = new THREE.PlaneGeometry(1.02, PORTAL_LINTEL_Y - 0.06);
  geometry.name = 'mistCurtain';
  geometry.rotateY(Math.PI / 2);

  const color = blendTokens(TOKENS.palettes.noon.sun, palette.hemiSky, 0.45);
  const textures: THREE.DataTexture[] = [];
  const materials: EmissiveMaterial[] = [];
  const meshes: THREE.Mesh[] = [];
  for (let l = 0; l < layers; l++) {
    const texture = makeMistTexture(0x51ed270b + l * 977);
    textures.push(texture);
    const material = emissiveMaterial(color, 0.95, {
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      side: THREE.DoubleSide,
      name: `mistCurtainLayer${l}`,
    });
    material.alphaMap = texture;
    materials.push(material);
    for (let h = 0; h < halves; h++) {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `mistCurtain${l}_${h}`;
      mesh.renderOrder = 2;
      mesh.frustumCulled = false;
      pivot.add(mesh);
      meshes.push(mesh);
    }
  }

  const poolGeometry = new THREE.PlaneGeometry(0.96, 0.86);
  poolGeometry.name = 'mistPool';
  poolGeometry.rotateX(-Math.PI / 2);
  const poolMaterial = emissiveMaterial(color, 0.9, {
    transparent: true,
    opacity: 0.45,
    depthWrite: false,
    side: THREE.DoubleSide,
    name: 'mistPool',
  });
  poolMaterial.alphaMap = textures[0] ?? null;
  const pool = new THREE.Mesh(poolGeometry, poolMaterial);
  pool.name = 'mistPool';
  pool.renderOrder = 2;
  pool.frustumCulled = false;
  pool.position.y = 0.06;
  pivot.add(pool);

  let progress = 0;
  let flareAmount = 0;
  let breath = 0;

  const apply = (): void => {
    const e = ease(progress);
    for (let i = 0; i < meshes.length; i++) {
      const mesh = meshes[i] as THREE.Mesh;
      const layer = Math.floor(i / halves);
      const side = i % halves === 0 ? -1 : 1;
      const drift = Math.sin(breath * (0.7 + 0.3 * layer) + layer * 2.1 + side) * 0.05;
      mesh.position.set(layer === 0 ? 0.03 : -0.03, (PORTAL_LINTEL_Y - 0.06) * 0.5 + drift * 0.6, side * (0.26 + 0.3 * e) + drift);
      mesh.scale.set(1, 1 + 0.04 * Math.sin(breath * 0.8 + layer), Math.max(0.001, 0.55 - 0.5 * e));
    }
    for (let l = 0; l < materials.length; l++) {
      const material = materials[l] as EmissiveMaterial;
      const fade = (1 - e) * (0.5 + 0.14 * Math.sin(breath * 0.9 + l * 1.7));
      material.opacity = Math.max(0, fade);
      setEmissiveIntensity(material, 0.95 * (1 + 1.6 * flareAmount));
    }
    pool.scale.set(1, 1, Math.max(0.001, 1 - 0.86 * e));
    pool.position.y = 0.06 + 0.012 * Math.sin(breath * 0.7);
    poolMaterial.opacity = Math.max(0, (1 - e) * (0.42 + 0.1 * Math.sin(breath * 0.6)));
    setEmissiveIntensity(poolMaterial, 0.9 * (1 + 1.6 * flareAmount));
  };
  apply();

  return {
    group,
    root,
    get progress() {
      return progress;
    },
    setProgress(p) {
      const v = clamp01(p);
      if (v === progress) return;
      progress = v;
      apply();
    },
    setFlare(amount) {
      const v = clamp01(amount);
      if (v === flareAmount) return;
      flareAmount = v;
      apply();
    },
    setTime(t) {
      breath = t;
      apply();
    },
    dispose() {
      geometry.dispose();
      poolGeometry.dispose();
      poolMaterial.dispose();
      for (const m of materials) m.dispose();
      for (const t of textures) t.dispose();
      materials.length = 0;
      textures.length = 0;
      meshes.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}

function makeMistTexture(seed: number): THREE.DataTexture {
  const size = 128;
  const field = fbmField(seed, { size, frequency: 3, octaves: 4, gain: 0.55 });
  const rgba = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const u = (x / (size - 1)) * 2 - 1;
      const v = (y / (size - 1)) * 2 - 1;
      const edge = Math.max(0, 1 - u * u * u * u) * Math.max(0, 1 - Math.pow(Math.max(0, v), 3));
      const n = Math.max(0, Math.min(1, (field.data[i] ?? 0) * 1.75 - 0.42));
      const a = Math.round(255 * Math.max(0, Math.min(1, n * edge)));
      rgba[i * 4] = 255;
      rgba[i * 4 + 1] = 255;
      rgba[i * 4 + 2] = 255;
      rgba[i * 4 + 3] = a;
    }
  }
  const texture = makeDataTexture(rgba, size, size, 'mistCurtainAlpha');
  return texture;
}




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
  setGrow(grow01: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildKeyAltar(opts: KeyAltarOptions): KeyAltarHandle {
  const { skin, index } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? `keyAltar${index}`;
  const root = new THREE.Object3D();
  root.name = `keyAltarRoot${index}`;
  group.add(root);

  const decor = new THREE.Group();
  decor.name = `keyAltarDecor${index}`;
  root.add(decor);

  const color = skin.keys[index] ?? new THREE.Color(0xffffff);
  const pedestalGeometry = makePedestalGeometry(skin);
  const pedestal = new THREE.Mesh(pedestalGeometry, opts.stone);
  pedestal.name = `keyPedestal${index}`;
  pedestal.castShadow = true;
  pedestal.receiveShadow = true;
  decor.add(pedestal);

  const token = new KeyToken({
    color,
    radius: 0.1,
    hover: 0.42,
    intensity: index === 1 ? 1.5 : 1.9,
    name: `keyToken${index}`,
  });
  decor.add(token.group);

  let carried = false;
  const gemHeight = 0.42;

  return {
    group,
    root,
    gemHeight,
    setCarried(v) {
      carried = v;
      token.setVisible(!v);
    },
    setGrow(grow01) {
      decor.scale.setScalar(clamp01(grow01));
    },
    setTime(t) {
      decor.rotation.y = Math.sin(t * 0.4 + index) * 0.05;
    },
    update(dt, time) {
      if (!carried) token.update(dt, time);
    },
    dispose() {
      token.dispose();
      pedestalGeometry.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function makePedestalGeometry(skin: TempleSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const base = new THREE.CylinderGeometry(0.18, 0.23, 0.09, 8, 1);
  base.translate(0, 0.045, 0);
  parts.push({ geometry: base, color: skin.stoneDark });
  const drum = new THREE.CylinderGeometry(0.1, 0.14, 0.13, 8, 1);
  drum.translate(0, 0.155, 0);
  parts.push({ geometry: drum, color: skin.stoneLight });
  for (let i = 0; i < 4; i++) {
    const petal = new THREE.ConeGeometry(0.07, 0.14, 4, 1);
    petal.rotateX(0.85);
    petal.rotateY((i / 4) * Math.PI * 2);
    petal.translate(Math.cos((i / 4) * Math.PI * 2) * 0.1, 0.25, Math.sin((i / 4) * Math.PI * 2) * 0.1);
    parts.push({ geometry: petal, color: skin.stoneLight });
  }
  return mergeColoredParts(parts, 'keyPedestal');
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
  const geometry = new THREE.OctahedronGeometry(0.062, 1);
  const material = emissiveMaterial(0xffffff, 1, { name: 'carriedKeys' });
  const mesh = new THREE.InstancedMesh(geometry, material, 3);
  mesh.name = 'carriedKeys';
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(9), 3);
  {
    const c = new THREE.Color();
    for (let i = 0; i < 3; i++) {
      c.copy(skin.keys[i] ?? new THREE.Color(0xffffff)).multiplyScalar(i === 1 ? 1.5 : 2.0);
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
  group.add(mesh);

  const carried = [false, false, false];
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3(1, 1, 1);

  const apply = (t: number): void => {
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * Math.PI * 2;
      const bob = Math.sin(t * 2.2 + i * 2.1) * 0.03;
      position.set(Math.cos(a) * 0.2, 0.02 * (i - 1) + bob, Math.sin(a) * 0.2);
      euler.set(t * 0.9 + i, t * 1.3, 0);
      quaternion.setFromEuler(euler);
      scale.setScalar(carried[i] ? 1 : 0);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  };
  apply(0);

  let time = 0;
  return {
    group,
    setCarried(index, on) {
      if (index < 0 || index > 2) return;
      if (carried[index] === on) return;
      carried[index] = on;
      apply(time);
    },
    setTime(t) {
      time = t;
      apply(t);
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

