import { GoalBeacon } from '@kit/grid';
import {
  emissiveMaterial,
  oakMaterial,
  oakUniforms,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { blendTokens, clamp01 } from '@kit/mix';
import type { ColoredPart } from '@kit/parts';
import type { Palette } from '@kit/tokens';
import * as THREE from 'three';

const BUD_Y = 0.44;
const PETAL_SCALE = 1.55;
const PETALS = 6;
const PITCH_CLOSED = 0.1;
const PITCH_OPEN = 1.16;
const PISTIL_HDR = 0.98;

export interface LanternOptions {
  palette: Palette;
  blossom: THREE.ColorRepresentation;
  flame: THREE.ColorRepresentation;
  stem: THREE.ColorRepresentation;
  name?: string;
}

export interface LanternHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly crownHeight: number;
  setState(open01: number, grow01: number): void;
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildLantern(opts: LanternOptions): LanternHandle {
  const { palette, blossom, flame, stem } = opts;
  const detail = 1;
  const group = new THREE.Group();
  group.name = opts.name ?? 'lanternFlower';

  const root = new THREE.Object3D();
  root.name = 'lanternRoot';
  group.add(root);

  const beacon = new GoalBeacon({
    color: flame,
    height: 0.5,
    radius: 0.085,
    intensity: 0.3,
    name: 'lanternGlow',
  });
  root.add(beacon.group);

  const bud = new THREE.Object3D();
  bud.name = 'lanternBud';
  root.add(bud);

  const stemParts: ColoredPart[] = [];
  {
    const shaft = new THREE.CylinderGeometry(0.019, 0.038, BUD_Y, Math.max(4, Math.round(6 * detail)));
    shaft.translate(0, BUD_Y * 0.5, 0);
    stemParts.push({ geometry: shaft, color: stem });
    for (const side of [-1, 1]) {
      const leaf = new THREE.CircleGeometry(0.11, Math.max(4, Math.round(6 * detail)));
      leaf.scale(1, 0.5, 1);
      leaf.rotateX(-Math.PI / 2.3);
      leaf.rotateY(side > 0 ? 0.6 : Math.PI - 0.6);
      leaf.translate(side * 0.08, BUD_Y * 0.32, side * 0.04);
      stemParts.push({ geometry: leaf, color: blendTokens(stem, palette.groundB, 0.3) });
    }
  }
  const stemGeometry = mergeStem(stemParts);
  const stemMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.82,
    rimColor: palette.skyHorizon,
    rimStrength: 0.42,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.28,
    name: 'lanternStemMaterial',
  });
  const stemMesh = new THREE.Mesh(stemGeometry, stemMaterial);
  stemMesh.name = 'lanternStem';
  stemMesh.castShadow = true;
  bud.add(stemMesh);

  const petalGeometry = makePetalGeometry();
  const petalMaterial: OakMaterial = oakMaterial({
    color: blossom,
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.72,
    rimColor: flame,
    rimStrength: 0.42,
    toonSteps: 2,
    toonSoftness: 0.38,
    toonFloor: 0.35,
    name: 'lanternPetalMaterial',
  });
  const petals = new THREE.InstancedMesh(petalGeometry, petalMaterial, PETALS);
  petals.name = 'lanternPetals';
  petals.castShadow = false;
  petals.frustumCulled = false;
  petals.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(PETALS * 3), 3);
  {
    const c = new THREE.Color();
    for (let i = 0; i < PETALS; i++) {
      const k = 0.86 + 0.22 * (((i * 7) % 5) / 4);
      c.setRGB(k, k, k);
      petals.setColorAt(i, c);
    }
    if (petals.instanceColor) petals.instanceColor.needsUpdate = true;
  }
  bud.add(petals);

  const pistilGeometry = new THREE.IcosahedronGeometry(0.055, 1);
  const pistilMaterial: EmissiveMaterial = emissiveMaterial(flame, PISTIL_HDR, { name: 'lanternFlame' });
  const pistil = new THREE.Mesh(pistilGeometry, pistilMaterial);
  pistil.name = 'lanternFlame';
  pistil.position.set(0, BUD_Y + 0.065, 0);
  bud.add(pistil);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3(0, BUD_Y, 0);
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);
  let openness = 0;

  const crownHeight = BUD_Y + 0.12;

  return {
    group,
    root,
    crownHeight,

    setState(open01, grow01) {
      const open = clamp01(open01);
      const grow = clamp01(grow01);
      openness = open;

      bud.scale.setScalar(grow);
      beacon.group.scale.setScalar(grow);

      const pitch = PITCH_CLOSED + (PITCH_OPEN - PITCH_CLOSED) * open;
      for (let i = 0; i < PETALS; i++) {
        const yaw = (i / PETALS) * Math.PI * 2;
        const s = PETAL_SCALE * (0.94 + 0.16 * open + 0.06 * (((i * 3) % 4) / 3));
        euler.set(pitch, yaw, 0);
        quaternion.setFromEuler(euler);
        scale.set(s, PETAL_SCALE * (1.06 - 0.14 * open), s);
        matrix.compose(position, quaternion, scale);
        petals.setMatrixAt(i, matrix);
      }
      petals.instanceMatrix.needsUpdate = true;

      pistil.scale.setScalar(0.78 + 0.34 * open);
      setEmissiveIntensity(pistilMaterial, PISTIL_HDR * (0.55 + 0.75 * open) * (0.25 + 0.75 * grow));
    },

    flare(strength = 1) {
      beacon.flare(strength);
    },

    setTime(t) {
      bud.rotation.z = Math.sin(t * 0.85) * 0.055 * (0.4 + 0.6 * openness);
      bud.rotation.x = Math.cos(t * 0.63) * 0.042 * (0.4 + 0.6 * openness);
      oakUniforms(petalMaterial)?.setTime(t);
      oakUniforms(stemMaterial)?.setTime(t);
    },

    update(dt, time) {
      beacon.update(dt, time);
    },

    dispose() {
      beacon.dispose();
      petals.dispose();
      petalGeometry.dispose();
      petalMaterial.dispose();
      stemGeometry.dispose();
      stemMaterial.dispose();
      pistilGeometry.dispose();
      pistilMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function makePetalGeometry(): THREE.BufferGeometry {
  const b0: readonly [number, number, number] = [-0.042, 0, 0];
  const b1: readonly [number, number, number] = [0.042, 0, 0];
  const m0: readonly [number, number, number] = [-0.074, 0.105, 0.03];
  const m1: readonly [number, number, number] = [0.074, 0.105, 0.03];
  const tip: readonly [number, number, number] = [0, 0.238, 0.078];
  const tris: readonly (readonly (readonly [number, number, number])[])[] = [
    [b0, b1, m1],
    [b0, m1, m0],
    [m0, m1, tip],
  ];
  const grade: readonly number[] = [0.5, 0.5, 0.86, 0.5, 0.86, 0.86, 0.86, 0.86, 1.12];
  const positions = new Float32Array(9 * 3);
  const colors = new Float32Array(9 * 3);
  let o = 0;
  for (const tri of tris) {
    for (const v of tri) {
      positions[o * 3] = v[0];
      positions[o * 3 + 1] = v[1];
      positions[o * 3 + 2] = v[2];
      const k = grade[o] ?? 1;
      colors[o * 3] = k;
      colors[o * 3 + 1] = k;
      colors[o * 3 + 2] = k;
      o++;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'lanternPetal';
  return g;
}

function mergeStem(parts: readonly ColoredPart[]): THREE.BufferGeometry {
  let total = 0;
  for (const p of parts) {
    total += p.geometry.index ? p.geometry.index.count : p.geometry.getAttribute('position').count;
  }
  const positions = new Float32Array(total * 3);
  const normals = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);
  const c = new THREE.Color();
  let o = 0;
  for (const part of parts) {
    const geo = part.geometry;
    const pos = geo.getAttribute('position');
    const nor = geo.getAttribute('normal');
    const index = geo.index;
    const n = index ? index.count : pos.count;
    c.set(part.color);
    for (let i = 0; i < n; i++) {
      const vi = index ? index.getX(i) : i;
      positions[o * 3] = pos.getX(vi);
      positions[o * 3 + 1] = pos.getY(vi);
      positions[o * 3 + 2] = pos.getZ(vi);
      if (nor) {
        normals[o * 3] = nor.getX(vi);
        normals[o * 3 + 1] = nor.getY(vi);
        normals[o * 3 + 2] = nor.getZ(vi);
      }
      colors[o * 3] = c.r;
      colors[o * 3 + 1] = c.g;
      colors[o * 3 + 2] = c.b;
      o++;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeBoundingSphere();
  g.name = 'lanternStem';
  for (const p of parts) p.geometry.dispose();
  return g;
}
