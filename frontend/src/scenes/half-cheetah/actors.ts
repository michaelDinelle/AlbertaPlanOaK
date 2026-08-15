
import {
  emissiveMaterial,
  oakMaterial,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { hash2f } from '@kit/rng';
import { radialSprite } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import { CHEETAH_BONES, FK_POINT_INDEX } from '@sim/fk/cheetah';
import type { CheetahGeometry } from '@sim/types';
import { blendTokens, clampRange, mergeColoredParts, smoothstep01, TOKEN_BLACK, TOKEN_WHITE, type ColoredPart } from './world';


const SPOT_DENSITY = 7.5;
const CONTACT_HIGH = 0.1;
const CONTACT_LOW = 0.036;
export const REFERENCE_SPEED = 1.1;

interface CoatSpec {
  r0: number;
  rMid: number;
  r1: number;
  belly: number;
  sock: number;
  spots: number;
  deep: number;
  wide: number;
  rings: number;
  radial: number;
}

const TORSO: CoatSpec = { r0: 0.128, rMid: 0.088, r1: 0.138, belly: 0.8, sock: 0, spots: 1.15, deep: 1.2, wide: 0.82, rings: 44, radial: 22 };
const THIGH: CoatSpec = { r0: 0.09, rMid: 0.072, r1: 0.05, belly: 0.2, sock: 0, spots: 0.9, deep: 1.1, wide: 0.86, rings: 18, radial: 14 };
const SHIN: CoatSpec = { r0: 0.05, rMid: 0.042, r1: 0.036, belly: 0.15, sock: 0.45, spots: 0.4, deep: 1, wide: 0.95, rings: 14, radial: 12 };
const FOOT: CoatSpec = { r0: 0.036, rMid: 0.032, r1: 0.028, belly: 0, sock: 1, spots: 0, deep: 1, wide: 1, rings: 10, radial: 10 };

const COATS: readonly CoatSpec[] = [TORSO, THIGH, SHIN, FOOT, THIGH, SHIN, FOOT];
const BONE_NAMES: readonly string[] = ['torso', 'thighBack', 'shinBack', 'footBack', 'thighFront', 'shinFront', 'footFront'];


const TAIL_LEN: readonly number[] = [0.26, 0.23, 0.2, 0.17];
const TAIL_R: readonly number[] = [0.062, 0.055, 0.047, 0.04, 0.022];
const TAIL_DROOP: readonly number[] = [0.17, 0.2, 0.22, 0.2];
const TAIL_SWING: readonly number[] = [0.14, 0.18, 0.22, 0.26];
const TAIL_STREAM = 0.72;
const TAIL_HANG = 0.13;
const TAIL_RUN_LIFT = 0.4;


export type CoatColorFn = (t: number, theta: number, out: THREE.Color) => void;

export function limbGeometry(
  len: number,
  spec: CoatSpec,
  color: CoatColorFn,
  name: string,
  minRadius: number,
): THREE.BufferGeometry {
  const caps = 4;
  const rings = spec.rings;
  const radial = spec.radial;
  const profile = (u: number): number => {
    const a = spec.r0;
    const b = spec.rMid;
    const c = spec.r1;
    const m = 1 - u;
    return Math.max(minRadius, m * m * a + 2 * m * u * b + u * u * c);
  };

  const ys: number[] = [];
  const rs: number[] = [];
  const half = len * 0.5;
  for (let i = 0; i <= caps; i++) {
    const alpha = (i / caps) * (Math.PI / 2);
    ys.push(-half - spec.r0 * Math.cos(alpha));
    rs.push(spec.r0 * Math.sin(alpha));
  }
  for (let i = 1; i <= rings; i++) {
    const u = i / rings;
    ys.push(-half + u * len);
    rs.push(profile(u));
  }
  for (let i = 1; i <= caps; i++) {
    const alpha = (1 - i / caps) * (Math.PI / 2);
    ys.push(half + spec.r1 * Math.cos(alpha));
    rs.push(spec.r1 * Math.sin(alpha));
  }

  const nRing = ys.length;
  const nRad = radial + 1;
  const count = nRing * nRad;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const c = new THREE.Color();
  const span = (ys[nRing - 1] ?? half) - (ys[0] ?? -half);
  for (let j = 0; j < nRing; j++) {
    const y = ys[j] ?? 0;
    const r = rs[j] ?? 0;
    const t = span > 1e-9 ? (y - (ys[0] ?? 0)) / span : 0;
    for (let i = 0; i < nRad; i++) {
      const theta = i / radial;
      const a = theta * Math.PI * 2;
      const idx = j * nRad + i;
      positions[idx * 3] = Math.cos(a) * r * spec.deep;
      positions[idx * 3 + 1] = y;
      positions[idx * 3 + 2] = Math.sin(a) * r * spec.wide;
      color(t, theta, c);
      colors[idx * 3] = c.r;
      colors[idx * 3 + 1] = c.g;
      colors[idx * 3 + 2] = c.b;
    }
  }

  const quads = (nRing - 1) * radial;
  const indices = count > 65535 ? new Uint32Array(quads * 6) : new Uint16Array(quads * 6);
  let o = 0;
  for (let j = 0; j < nRing - 1; j++) {
    for (let i = 0; i < radial; i++) {
      const a = j * nRad + i;
      const b = a + 1;
      const d = a + nRad;
      const e = d + 1;
      indices[o++] = a;
      indices[o++] = d;
      indices[o++] = b;
      indices[o++] = b;
      indices[o++] = d;
      indices[o++] = e;
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = name;
  return g;
}

export function spotMask(u: number, v: number, nu: number, nv: number, seed: number, radius: number): number {
  const cu = Math.floor(u * nu);
  const cv = Math.floor(v * nv);
  let best = 4;
  for (let du = -1; du <= 1; du++) {
    for (let dv = -1; dv <= 1; dv++) {
      const ci = cu + du;
      const cj = cv + dv;
      const wj = ((cj % nv) + nv) % nv;
      const jx = hash2f(ci, wj, seed);
      const jy = hash2f(ci, wj, seed ^ 0x9e3779b9);
      const px = (ci + 0.15 + jx * 0.7) / nu;
      const py = (cj + 0.15 + jy * 0.7) / nv;
      const dx = (u - px) * 1.05;
      const dy = (v - py) * 0.92;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < best) best = d;
    }
  }
  return 1 - smoothstep01(best / Math.max(1e-4, radius));
}


export interface CheetahHandle {
  group: THREE.Group;
  joints: THREE.Object3D[];
  setPose(fk: Float32Array, speed01: number): void;
  setTime(t: number): void;
  setGlow(gain: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

export interface CheetahOptions {
  tier: QualityTier;
  palette: Palette;
  accent: THREE.ColorRepresentation;
  seed: number;
  cheetah: CheetahGeometry;
}

const ZERO_POSE = new Float32Array(9);

export function buildCheetah(opts: CheetahOptions): CheetahHandle {
  const { tier, palette, seed, cheetah } = opts;
  const shadows = tier !== 'thumb';
  const lod = tier === 'hero' ? 1 : tier === 'medium' ? 0.7 : 0.45;
  const group = new THREE.Group();
  group.name = 'cheetahRig';

  const cCoat = blendTokens(palette.groundB, palette.sun, 0.45);
  const cBack = blendTokens(cCoat, palette.groundA, 0.44);
  const cBelly = blendTokens(TOKEN_WHITE, palette.groundB, 0.42);
  const cSpot = blendTokens(TOKEN_BLACK, palette.hemiGround, 0.22);
  const cSock = blendTokens(TOKEN_BLACK, palette.groundA, 0.3);
  const cNose = blendTokens(TOKEN_BLACK, palette.hemiGround, 0.2);
  const cTip = blendTokens(TOKEN_WHITE, palette.sun, 0.25);

  const boneR = cheetah.radius;
  const headR = cheetah.headRadius;

  const rest = cheetah.fk(ZERO_POSE);
  const boneLen: number[] = [];
  for (const bone of CHEETAH_BONES) {
    const a = bone[0];
    const b = bone[1];
    const dx = (rest[b * 2] ?? 0) - (rest[a * 2] ?? 0);
    const dz = (rest[b * 2 + 1] ?? 0) - (rest[a * 2 + 1] ?? 0);
    boneLen.push(Math.hypot(dx, dz));
  }

  const joints: THREE.Object3D[] = [];
  for (let i = 0; i < 10; i++) {
    const node = new THREE.Object3D();
    node.name = `cheetahJoint${i}`;
    group.add(node);
    joints.push(node);
  }

  const spotColor = new THREE.Color(cSpot);

  const coatMaterial: OakMaterial = oakMaterial({
    color: TOKEN_WHITE,
    vertexColors: true,
    roughness: 0.82,
    rimColor: palette.sun,
    rimStrength: 1.4,
    rimPower: 2.4,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: tier === 'thumb' ? 0.46 : 0.22,
    spots: { color: cSpot, strength: 0.8, scale: 15, radius: 0.40, seed: (seed & 0xffff) / 64 },
    tier,
    name: 'cheetahCoat',
  });

  const scratchColor = new THREE.Color();

  const coatFor = (spec: CoatSpec, len: number, boneSeed: number): CoatColorFn => {
    return (t, theta, out) => {
      const ventral = Math.cos(theta * Math.PI * 2) * 0.5 + 0.5;
      out.copy(cBack).lerp(cCoat, smoothstep01(1 - Math.abs(ventral - 0.5) * 1.6));
      if (spec.belly > 0) out.lerp(cBelly, spec.belly * smoothstep01((ventral - 0.52) / 0.42));
      if (spec.sock > 0) out.lerp(cSock, spec.sock * smoothstep01((t - 0.42) / 0.5));
    };
  };

  const boneMeshes: THREE.Mesh[] = [];
  const boneGeometries: THREE.BufferGeometry[] = [];
  for (let i = 0; i < CHEETAH_BONES.length; i++) {
    const base = COATS[i] ?? THIGH;
    const spec: CoatSpec = {
      ...base,
      rings: Math.max(6, Math.round(base.rings * lod)),
      radial: Math.max(6, Math.round(base.radial * lod)),
    };
    const len = boneLen[i] ?? 0.2;
    const geo = limbGeometry(len, spec, coatFor(spec, len, i * 0x9e37), `cheetah_${BONE_NAMES[i] ?? i}`, boneR);
    boneGeometries.push(geo);
    const mesh = new THREE.Mesh(geo, coatMaterial);
    mesh.name = `cheetah_${BONE_NAMES[i] ?? i}`;
    mesh.castShadow = shadows;
    group.add(mesh);
    boneMeshes.push(mesh);
  }

  const headNode = new THREE.Object3D();
  headNode.name = 'cheetahHeadNode';
  joints[FK_POINT_INDEX.head]?.add(headNode);

  const headParts: ColoredPart[] = [];
  {
    const skull = new THREE.SphereGeometry(headR, Math.max(10, Math.round(20 * lod)), Math.max(8, Math.round(14 * lod)));
    skull.scale(1.08, 0.94, 0.9);
    headParts.push({ geometry: skull, color: cCoat });
  }
  const headGeometry = mergeColoredParts(headParts, 'cheetahSkull');
  {
    const pos = headGeometry.getAttribute('position');
    const col = headGeometry.getAttribute('color');
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i) / headR;
      const y = pos.getY(i) / headR;
      const z = pos.getZ(i) / headR;
      const up = clampRange((y / 0.933) * 0.5 + 0.5, 0, 1);
      scratchColor.copy(cBack).lerp(cCoat, up);
      scratchColor.lerp(cBelly, smoothstep01((-y - 0.133) / 0.667) * 0.75);
      const mask = spotMask(
        clampRange(x / 2.267 + 0.5, 0, 1),
        clampRange(Math.atan2(z, y) / (Math.PI * 2) + 0.5, 0, 1),
        4,
        6,
        seed ^ 0x51a5,
        0.34,
      );
      const tear = Math.abs(z) > 0.367 ? smoothstep01(1 - Math.abs((y + 0.133) - (x - 0.4) * 0.55) / 0.187) : 0;
      scratchColor.lerp(spotColor, tear * clampRange((x + 0.133) / 0.667, 0, 1) * 0.9);
      col.setXYZ(i, scratchColor.r, scratchColor.g, scratchColor.b);
    }
    col.needsUpdate = true;
  }
  const headMesh = new THREE.Mesh(headGeometry, coatMaterial);
  headMesh.name = 'cheetahSkull';
  headMesh.castShadow = shadows;
  headNode.add(headMesh);

  const featureParts: ColoredPart[] = [];
  {
    const muzzle = new THREE.CylinderGeometry(0.367 * headR, 0.633 * headR, 0.6 * headR, Math.max(6, Math.round(10 * lod)));
    muzzle.rotateZ(-Math.PI / 2);
    muzzle.translate(0.9 * headR, -0.28 * headR, 0);
    featureParts.push({ geometry: muzzle, color: cBelly });
    const nose = new THREE.SphereGeometry(0.213 * headR, 8, 6);
    nose.scale(0.85, 0.8, 1);
    nose.translate(1.187 * headR, -0.213 * headR, 0);
    featureParts.push({ geometry: nose, color: cNose });
    for (const s of [-1, 1]) {
      const ear = new THREE.ConeGeometry(0.333 * headR, 0.467 * headR, 6);
      ear.rotateZ(s * 0.12);
      ear.rotateX(-s * 0.26);
      ear.translate(-0.187 * headR, 0.833 * headR, s * 0.48 * headR);
      featureParts.push({ geometry: ear, color: cBack });
      const eye = new THREE.SphereGeometry(0.127 * headR, 7, 5);
      eye.translate(0.587 * headR, 0.187 * headR, s * 0.587 * headR);
      featureParts.push({ geometry: eye, color: cNose });
    }
  }
  const featureGeometry = mergeColoredParts(featureParts, 'cheetahFace');
  const featureMesh = new THREE.Mesh(featureGeometry, coatMaterial);
  featureMesh.name = 'cheetahFace';
  featureMesh.castShadow = shadows;
  headNode.add(featureMesh);

  const tailRoot = new THREE.Object3D();
  tailRoot.name = 'cheetahTailRoot';
  joints[FK_POINT_INDEX.bt]?.add(tailRoot);
  const tailNodes: THREE.Object3D[] = [];
  const tailGeometries: THREE.BufferGeometry[] = [];
  {
    let parent: THREE.Object3D = tailRoot;
    for (let i = 0; i < TAIL_LEN.length; i++) {
      const node = new THREE.Object3D();
      node.name = `cheetahTail${i}`;
      node.position.y = i === 0 ? 0 : TAIL_LEN[i - 1] ?? 0;
      parent.add(node);
      const len = TAIL_LEN[i] ?? 0.2;
      const r0 = TAIL_R[i] ?? 0.04;
      const r1 = TAIL_R[i + 1] ?? 0.022;
      const last = i === TAIL_LEN.length - 1;
      const spec: CoatSpec = {
        r0,
        rMid: (r0 + r1) * 0.5,
        r1,
        belly: 0,
        sock: 0,
        spots: last ? 0 : 0.55,
        deep: 1,
        wide: 1,
        rings: Math.max(5, Math.round(10 * lod)),
        radial: Math.max(6, Math.round(10 * lod)),
      };
      const paint: CoatColorFn = (t, theta, out) => {
        out.copy(cCoat).lerp(cBack, 0.3);
        if (spec.spots > 0) {
          const mask = spotMask(t, theta, Math.max(2, Math.round(len * SPOT_DENSITY)), 5, seed ^ (0x77 + i), 0.4);
          out.lerp(spotColor, mask * 0.6);
        }
        if (last) out.lerp(cTip, smoothstep01((t - 0.45) / 0.4));
      };
      const geo = limbGeometry(len, spec, paint, `cheetahTailSeg${i}`, 0);
      geo.translate(0, len * 0.5, 0);
      tailGeometries.push(geo);
      const mesh = new THREE.Mesh(geo, coatMaterial);
      mesh.name = `cheetahTailMesh${i}`;
      mesh.castShadow = shadows;
      node.add(mesh);
      tailNodes.push(node);
      parent = node;
    }
  }

  const dustColor = blendTokens(palette.groundB, palette.sun, 0.35);
  const dust: AmbientParticlesHandle[] = [];
  for (let i = 0; i < 2; i++) {
    const handle = makeAmbientParticles({
      preset: 'dust',
      count: 130,
      bounds: { y: 0.08, halfX: 0.24, halfY: 0.11, halfZ: 0.13 },
      seed: seed ^ (0x3d91 + i * 977),
      tier,
      color: dustColor,
      size: 1.5,
      intensity: 0.75,
      opacity: 0.34,
      rise: 0.14,
      swirl: 0.05,
      name: `cheetahDust${i}`,
    });
    handle.setDensity(0);
    group.add(handle.points);
    dust.push(handle);
  }

  const scuffSprite = radialSprite(64, 2.2);
  const scuffGeometry = new THREE.PlaneGeometry(1, 1);
  scuffGeometry.rotateX(-Math.PI / 2);
  const scuffMaterials: EmissiveMaterial[] = [];
  const scuffs: THREE.Mesh[] = [];
  for (let i = 0; i < 2; i++) {
    const material = emissiveMaterial(blendTokens(dustColor, palette.sun, 0.3), 0.5, {
      map: scuffSprite,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      name: `cheetahScuffMaterial${i}`,
    });
    material.opacity = 0;
    const mesh = new THREE.Mesh(scuffGeometry, material);
    mesh.name = `cheetahScuff${i}`;
    mesh.position.y = 0.006;
    mesh.scale.set(0.5, 1, 0.34);
    mesh.renderOrder = 2;
    mesh.visible = false;
    group.add(mesh);
    scuffMaterials.push(material);
    scuffs.push(mesh);
  }

  const boneAngle = new Float64Array(CHEETAH_BONES.length);
  const TOE_INDEX: readonly number[] = [FK_POINT_INDEX.toe_b, FK_POINT_INDEX.toe_f];
  const Z_AXIS = new THREE.Vector3(0, 0, 1);
  const tailEuler = new THREE.Euler();
  const tailFrame = new THREE.Quaternion();
  const tailSegment = new THREE.Quaternion();
  const tailBend = new THREE.Quaternion();
  const tailDir = new THREE.Vector3();
  const tailLift = new THREE.Vector3();

  return {
    group,
    joints,

    setPose(fk, speed01) {
      for (let i = 0; i < joints.length; i++) {
        joints[i]?.position.set(fk[i * 2] ?? 0, fk[i * 2 + 1] ?? 0, 0);
      }

      for (let i = 0; i < CHEETAH_BONES.length; i++) {
        const bone = CHEETAH_BONES[i];
        const mesh = boneMeshes[i];
        if (!bone || !mesh) continue;
        const ax = fk[bone[0] * 2] ?? 0;
        const ay = fk[bone[0] * 2 + 1] ?? 0;
        const bx = fk[bone[1] * 2] ?? 0;
        const by = fk[bone[1] * 2 + 1] ?? 0;
        const dx = bx - ax;
        const dy = by - ay;
        mesh.position.set((ax + bx) * 0.5, (ay + by) * 0.5, 0);
        mesh.rotation.z = Math.atan2(-dx, dy);
        boneAngle[i] = Math.atan2(dy, dx);
      }

      const forward = boneAngle[0] ?? 0;
      headNode.rotation.z = forward;
      const gait = (boneAngle[3] ?? 0) - (boneAngle[6] ?? 0);
      const stream = clampRange(speed01, 0, 1);

      const rootAngle = forward + Math.PI * 0.5 + TAIL_HANG - TAIL_RUN_LIFT * stream;
      tailRoot.rotation.z = rootAngle;
      tailFrame.setFromAxisAngle(Z_AXIS, rootAngle);
      const backX = -Math.cos(forward);
      let endY = fk[FK_POINT_INDEX.bt * 2 + 1] ?? 0;
      for (let i = 0; i < tailNodes.length; i++) {
        const node = tailNodes[i];
        if (!node) continue;
        const amp = TAIL_SWING[i] ?? 0.2;
        tailEuler.set(
          amp * 0.9 * Math.sin(gait * 1.7 + i * 0.85 + 1.2),
          0,
          (TAIL_DROOP[i] ?? 0.2) * (1 - TAIL_STREAM * stream) + amp * Math.sin(gait * 1.7 + i * 0.6),
        );
        tailSegment.setFromEuler(tailEuler).premultiply(tailFrame);
        tailDir.set(0, 1, 0).applyQuaternion(tailSegment);

        const len = TAIL_LEN[i] ?? 0.2;
        const floor = (TAIL_R[i + 1] ?? 0) + boneR;
        const climb = (floor - endY) / len;
        if (tailDir.y < climb) {
          const y = climb > 1 ? 1 : climb;
          const flat = Math.hypot(tailDir.x, tailDir.z);
          const spread = Math.sqrt(1 - y * y);
          tailLift.set(
            (flat > 1e-6 ? tailDir.x / flat : backX >= 0 ? 1 : -1) * spread,
            y,
            (flat > 1e-6 ? tailDir.z / flat : 0) * spread,
          );
          tailBend.setFromUnitVectors(tailDir, tailLift);
          tailSegment.premultiply(tailBend);
          tailDir.copy(tailLift);
        }

        node.quaternion.copy(tailFrame).invert().multiply(tailSegment);
        tailFrame.copy(tailSegment);
        endY += tailDir.y * len;
      }

      for (let i = 0; i < 2; i++) {
        const idx = TOE_INDEX[i] ?? 0;
        const tx = fk[idx * 2] ?? 0;
        const ty = fk[idx * 2 + 1] ?? 0;
        const contact = 1 - smoothstep01((ty - CONTACT_LOW) / (CONTACT_HIGH - CONTACT_LOW));
        const kick = contact * stream;
        const puff = dust[i];
        if (puff) {
          puff.points.position.set(tx, 0.09, 0);
          puff.setDensity(kick * 0.95);
        }
        const scuff = scuffs[i];
        const material = scuffMaterials[i];
        if (scuff && material) {
          scuff.position.x = tx;
          const k = contact * (0.35 + 0.65 * stream);
          material.opacity = 0.26 * k;
          scuff.scale.set(0.34 + 0.5 * k, 1, 0.24 + 0.26 * k);
          scuff.visible = k > 0.02;
        }
      }
    },

    setTime(t) {
      for (const d of dust) d.setTime(t);
    },

    setGlow(gain) {
      const g = gain > 0 ? gain : 0;
      coatMaterial.color.setRGB(g, g, g);
    },

    setPixelRatio(dpr) {
      for (const d of dust) d.setPixelRatio(dpr);
    },

    dispose() {
      for (const g of boneGeometries) g.dispose();
      for (const g of tailGeometries) g.dispose();
      boneGeometries.length = 0;
      tailGeometries.length = 0;
      headGeometry.dispose();
      featureGeometry.dispose();
      coatMaterial.dispose();
      for (const d of dust) d.dispose();
      dust.length = 0;
      for (const m of scuffMaterials) m.dispose();
      scuffMaterials.length = 0;
      scuffGeometry.dispose();
      scuffSprite.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}
