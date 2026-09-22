import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { blendTokens, clamp01 } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { Palette } from '@kit/tokens';
import * as THREE from 'three';
import type { ReefSkin } from './reef';

const SHELL_R = 0.5;
const GAPE_UPPER = 0.95;
const GAPE_LOWER = 0.34;
const HINGE_Z = -0.34;
const HINGE_Y = 0.17;
const PEARL_Y = 0.24;
const PEARL_HDR = 1.3;

export interface ClamOptions {
  palette: Palette;
  skin: ReefSkin;
  facing: number;
  name?: string;
}

export interface ClamHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly pearlHeight: number;
  setState(open01: number, flare01: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildClam(opts: ClamOptions): ClamHandle {
  const { palette, skin } = opts;
  const detail = 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'giantClam';

  const root = new THREE.Object3D();
  root.name = 'clamRoot';
  group.add(root);

  const body = new THREE.Group();
  body.name = 'clamBody';
  body.rotation.y = opts.facing;
  root.add(body);

  const shellMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.72,
    metalness: 0.04,
    side: THREE.DoubleSide,
    rimColor: palette.hemiSky,
    rimStrength: 0.5,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.26,
    name: 'clamShellMaterial',
  });

  const baseParts: ColoredPart[] = [];
  {
    const radial = Math.max(6, Math.round(12 * detail));
    const rock = new THREE.CylinderGeometry(0.36, 0.46, 0.15, radial, 1);
    rock.translate(0, 0.075, 0);
    baseParts.push({ geometry: rock, color: blendTokens(skin.shelfDeep, skin.coral, 0.22) });
    for (let i = 0; i < (4); i++) {
      const a = (i / 4) * Math.PI * 2 + 0.6;
      const knob = new THREE.IcosahedronGeometry(0.06 + 0.025 * ((i % 3) / 2), 0);
      knob.scale(1, 0.7, 1);
      knob.translate(Math.cos(a) * 0.44, 0.05, Math.sin(a) * 0.44);
      baseParts.push({ geometry: knob, color: blendTokens(skin.coral, skin.anemone, 0.3) });
    }
  }
  const baseGeometry = mergeColoredParts(baseParts, 'clamBase');
  const baseMesh = new THREE.Mesh(baseGeometry, shellMaterial);
  baseMesh.name = 'clamBase';
  baseMesh.castShadow = true;
  baseMesh.receiveShadow = true;
  body.add(baseMesh);

  const mantleGeometry = makeMantleGeometry(detail, skin);
  const mantleMesh = new THREE.Mesh(mantleGeometry, shellMaterial);
  mantleMesh.name = 'clamMantle';
  mantleMesh.position.y = HINGE_Y;
  body.add(mantleMesh);

  const shellGeometry = makeShellGeometry(detail, skin);
  const upperHinge = new THREE.Group();
  upperHinge.name = 'clamHingeUpper';
  upperHinge.position.set(0, HINGE_Y, HINGE_Z);
  const upperShell = new THREE.Mesh(shellGeometry, shellMaterial);
  upperShell.name = 'clamShellUpper';
  upperShell.position.z = -HINGE_Z;
  upperShell.castShadow = true;
  upperShell.receiveShadow = true;
  upperHinge.add(upperShell);
  body.add(upperHinge);

  const lowerHinge = new THREE.Group();
  lowerHinge.name = 'clamHingeLower';
  lowerHinge.position.set(0, HINGE_Y, HINGE_Z);
  const lowerShell = new THREE.Mesh(shellGeometry, shellMaterial);
  lowerShell.name = 'clamShellLower';
  lowerShell.position.z = -HINGE_Z;
  lowerShell.rotation.z = Math.PI;
  lowerShell.castShadow = true;
  lowerShell.receiveShadow = true;
  lowerHinge.add(lowerShell);
  body.add(lowerHinge);

  const pearlGeometry = new THREE.IcosahedronGeometry(0.115, 2);
  const pearlMaterial: EmissiveMaterial = emissiveMaterial(skin.pearl, PEARL_HDR, { name: 'clamPearl' });
  const pearl = new THREE.Mesh(pearlGeometry, pearlMaterial);
  pearl.name = 'clamPearl';
  pearl.position.y = PEARL_Y;
  body.add(pearl);

  const haloGeometry = new THREE.IcosahedronGeometry(0.24, 1);
  const haloMaterial: EmissiveMaterial = emissiveMaterial(skin.pearl, PEARL_HDR * 0.16, {
    transparent: true,
    opacity: 0.18,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'clamPearlHalo',
  });
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.name = 'clamPearlHalo';
  halo.position.y = PEARL_Y;
  body.add(halo);

  let openness = 0;

  return {
    group,
    root,
    pearlHeight: PEARL_Y,

    setState(open01, flare01) {
      const open = clamp01(open01);
      const flare = clamp01(flare01);
      openness = open;
      upperHinge.rotation.x = GAPE_UPPER * open;
      lowerHinge.rotation.x = -GAPE_LOWER * open;
      const gain = (0.45 + 0.75 * open) * (1 + 1.5 * flare);
      pearl.scale.setScalar(0.8 + 0.34 * open + 0.2 * flare);
      halo.scale.setScalar(0.75 + 0.4 * open + 0.6 * flare);
      setEmissiveIntensity(pearlMaterial, PEARL_HDR * gain);
      setEmissiveIntensity(haloMaterial, PEARL_HDR * 0.16 * gain);
    },

    setTime(t) {
      body.rotation.z = Math.sin(t * 0.62) * 0.035 * (0.5 + 0.5 * openness);
      body.rotation.x = Math.cos(t * 0.47) * 0.028 * (0.5 + 0.5 * openness);
      oakUniforms(shellMaterial)?.setTime(t);
    },

    dispose() {
      shellGeometry.dispose();
      baseGeometry.dispose();
      mantleGeometry.dispose();
      pearlGeometry.dispose();
      haloGeometry.dispose();
      shellMaterial.dispose();
      pearlMaterial.dispose();
      haloMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

export function makeShellGeometry(detail: number, skin: ReefSkin): THREE.BufferGeometry {
  const radial = Math.max(12, Math.round(32 * detail));
  const rings = Math.max(3, Math.round(7 * detail));
  const ribs = 8;
  const g = new THREE.SphereGeometry(SHELL_R, radial, rings, 0, Math.PI * 2, 0, Math.PI / 2);
  const pos = g.getAttribute('position');
  const shell = new THREE.Color(skin.shell);
  const ribTone = new THREE.Color(skin.shellRib);
  const colors = new Float32Array(pos.count * 3);
  const tone = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const theta = Math.atan2(z, x);
    const wave = Math.cos(ribs * theta);
    const t = clamp01(y / SHELL_R);
    const flute = wave * (1 - t * t);
    const swell = 1 + 0.09 * flute;
    const rim = 1 - t;
    pos.setX(i, x * swell);
    pos.setZ(i, z * swell * 1.14);
    pos.setY(i, y * (0.4 + 0.1 * flute) + 0.085 * wave * rim * rim);
    const flank = clamp01(Math.sqrt(x * x + z * z) / SHELL_R);
    tone.copy(shell).lerp(ribTone, 0.1 + 0.5 * (0.5 + 0.5 * flute)).multiplyScalar(0.58 + 0.4 * flank);
    colors[i * 3] = tone.r;
    colors[i * 3 + 1] = tone.g;
    colors[i * 3 + 2] = tone.b;
  }
  pos.needsUpdate = true;
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'clamShell';
  return g;
}

export function makeMantleGeometry(detail: number, skin: ReefSkin): THREE.BufferGeometry {
  const radial = Math.max(10, Math.round(22 * detail));
  const g = new THREE.CircleGeometry(SHELL_R * 1.02, radial);
  g.rotateX(-Math.PI / 2);
  const pos = g.getAttribute('position');
  const colors = new Float32Array(pos.count * 3);
  const inner = new THREE.Color(skin.mantle);
  const edge = new THREE.Color(skin.anemone);
  const tone = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const r = Math.sqrt(x * x + z * z);
    const theta = Math.atan2(z, x);
    const k = clamp01(r / (SHELL_R * 1.02));
    pos.setY(i, Math.sin(theta * 8) * 0.05 * k * k);
    tone.copy(inner).lerp(edge, k * k);
    colors[i * 3] = tone.r;
    colors[i * 3 + 1] = tone.g;
    colors[i * 3 + 2] = tone.b;
  }
  pos.needsUpdate = true;
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'clamMantle';
  return g;
}

export interface BubbleOptions {
  palette: Palette;
  skin: ReefSkin;
  accent: THREE.ColorRepresentation;
  height: number;
  name?: string;
}

export interface BubbleHandle {
  group: THREE.Group;
  setTime(t: number): void;
  setStress(v: number): void;
  dispose(): void;
}

export function buildBubble(opts: BubbleOptions): BubbleHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'divingBell';
  group.position.y = opts.height;
  const brass = oakMaterial({ color: 0xb28f53, metalness: 0.68, roughness: 0.33, rimColor: opts.skin.bubble, rimStrength: 0.35 });
  const glass = emissiveMaterial(0xc5ffec, 1.5, { name: 'divingBellCore' });
  const bodyGeometry = new THREE.SphereGeometry(0.28, 12, 7, 0, Math.PI * 2, 0, Math.PI * 0.66);
  const rimGeometry = new THREE.TorusGeometry(0.246, 0.041, 5, 12);
  rimGeometry.rotateX(Math.PI / 2);
  rimGeometry.translate(0, -0.135, 0);
  const handleGeometry = new THREE.TorusGeometry(0.087, 0.018, 4, 10);
  handleGeometry.translate(0, 0.325, 0);
  const windowGeometry = new THREE.SphereGeometry(0.125, 10, 6);
  const body = new THREE.Mesh(bodyGeometry, brass);
  const rim = new THREE.Mesh(rimGeometry, brass);
  const handle = new THREE.Mesh(handleGeometry, brass);
  const window = new THREE.Mesh(windowGeometry, glass);
  window.scale.set(1, 0.86, 0.5);
  window.position.set(0, 0.01, 0.25);
  body.castShadow = true;
  group.add(body, rim, handle, window);
  let stress = 0;
  return {
    group,
    setTime(t) {
      group.rotation.z = Math.sin(t * 1.7) * 0.025 * (1 + stress);
      oakUniforms(brass)?.setTime(t);
    },
    setStress(value) {
      stress = value;
      setEmissiveIntensity(glass, 1.5 + stress * 0.7);
    },
    dispose() {
      for (const geometry of [bodyGeometry, rimGeometry, handleGeometry, windowGeometry]) geometry.dispose();
      brass.dispose();
      glass.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
