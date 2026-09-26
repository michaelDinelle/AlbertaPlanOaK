import {
  emissiveMaterial,
  oakMaterial,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { blendTokens, clampRange } from '@kit/mix';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { radialSprite } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import * as THREE from 'three/webgpu';
import { MC_MAX_SPEED } from '@sim/envs/mountainCar';

const WHEEL_R = 0.034;
const WHEEL_T = 0.012;
const WHEEL_Z = 0.05;
const AXLE_X = 0.07;
const DECK_Y = WHEEL_R + 0.012;
const FLOOR_Y = DECK_Y + 0.012;
const SLAT_H = 0.062;
const RAIL_Y = FLOOR_Y + SLAT_H;
const LANTERN_X = 0.116;
const LANTERN_Y = RAIL_Y + 0.024;
const LANTERN_FLAME_Y = LANTERN_Y + 0.006;

const RIBBON_SEGMENTS = 48;
const RIBBON_MAX_LEN = 0.58;
const RIBBON_LIFT = 0.048;
const LANTERN_BASE_HDR = 2.4;

export interface CartHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly lanternWorld: THREE.Vector3;
  setState(pos: number, vel: number, force: number): void;
  setTime(t: number): void;
  setDustKick(env01: number): void;
  setLanternGain(gain: number): void;
  dispose(): void;
}

export interface CartOptions {
  palette: Palette;
  accent: THREE.ColorRepresentation;
  seed: number;
  heightAt(x: number): number;
  slopeAt(x: number): number;
}

export function buildCart(opts: CartOptions): CartHandle {
  const { palette, accent, seed, heightAt, slopeAt } = opts;
  const group = new THREE.Group();
  group.name = 'cartRig';

  const woodLight = blendTokens(palette.groundB, palette.groundA, 0.3);
  const woodMid = blendTokens(palette.groundA, palette.groundB, 0.25);
  const woodDark = blendTokens(palette.groundA, palette.hemiGround, 0.7);
  const iron = blendTokens(palette.hemiGround, palette.skyTop, 0.55);

  const root = new THREE.Object3D();
  root.name = 'cartRoot';
  group.add(root);

  const tilt = new THREE.Object3D();
  tilt.name = 'cartTilt';
  root.add(tilt);

  const bodyParts: ColoredPart[] = [];
  {
    const beam = new THREE.BoxGeometry(0.196, 0.015, 0.116);
    beam.translate(0, DECK_Y, 0);
    bodyParts.push({ geometry: beam, color: woodDark });

    const floor = new THREE.BoxGeometry(0.172, 0.008, 0.1);
    floor.translate(0, FLOOR_Y, 0);
    bodyParts.push({ geometry: floor, color: woodMid });

    const SLATS = 6;
    for (let s = 0; s < SLATS; s++) {
      const k = s / (SLATS - 1);
      const x = -0.072 + k * 0.144;
      for (const z of [-0.05, 0.05]) {
        const slat = new THREE.BoxGeometry(0.015, SLAT_H, 0.008);
        slat.translate(x, FLOOR_Y + SLAT_H * 0.5, z);
        bodyParts.push({ geometry: slat, color: s % 2 === 0 ? woodLight : woodMid });
      }
    }
    for (const z of [-0.05, 0.05]) {
      const rail = new THREE.BoxGeometry(0.18, 0.009, 0.014);
      rail.translate(0, RAIL_Y, z);
      bodyParts.push({ geometry: rail, color: woodLight });
      const sill = new THREE.BoxGeometry(0.18, 0.008, 0.011);
      sill.translate(0, FLOOR_Y + 0.006, z);
      bodyParts.push({ geometry: sill, color: woodDark });
    }
    for (const x of [-0.086, 0.086]) {
      const board = new THREE.BoxGeometry(0.009, SLAT_H * 0.86, 0.1);
      board.translate(x, FLOOR_Y + SLAT_H * 0.45, 0);
      bodyParts.push({ geometry: board, color: woodMid });
    }
    const post = new THREE.BoxGeometry(0.009, 0.074, 0.009);
    post.translate(0.094, FLOOR_Y + 0.037, 0);
    bodyParts.push({ geometry: post, color: iron });
    const arm = new THREE.BoxGeometry(0.036, 0.007, 0.007);
    arm.translate(0.106, LANTERN_Y + 0.03, 0);
    bodyParts.push({ geometry: arm, color: iron });
    const ring = new THREE.CylinderGeometry(0.018, 0.021, 0.005, 6);
    ring.translate(LANTERN_X, LANTERN_Y - 0.014, 0);
    bodyParts.push({ geometry: ring, color: iron });
    const cap = new THREE.ConeGeometry(0.022, 0.016, 6);
    cap.translate(LANTERN_X, LANTERN_Y + 0.024, 0);
    bodyParts.push({ geometry: cap, color: iron });
  }
  const bodyGeometry = mergeColoredParts(bodyParts, 'cartBody');
  const bodyMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.78,
    rimColor: palette.sun,
    rimStrength: 0.9,
    toonSteps: 3,
    toonSoftness: 0.28,
    toonFloor: 0.22,
    name: 'cartBodyMaterial',
  });
  const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
  bodyMesh.name = 'cartBody';
  bodyMesh.castShadow = true;
  tilt.add(bodyMesh);

  const wheelGeometry = makeAxleGeometry(woodDark, iron);
  const wheelFront = new THREE.Mesh(wheelGeometry, bodyMaterial);
  wheelFront.name = 'cartAxleFront';
  wheelFront.position.set(AXLE_X, WHEEL_R, 0);
  wheelFront.castShadow = true;
  const wheelRear = new THREE.Mesh(wheelGeometry, bodyMaterial);
  wheelRear.name = 'cartAxleRear';
  wheelRear.position.set(-AXLE_X, WHEEL_R, 0);
  wheelRear.castShadow = true;
  tilt.add(wheelFront, wheelRear);

  const flameGeometry = new THREE.IcosahedronGeometry(0.013, 1);
  flameGeometry.scale(1, 1.4, 1);
  const flameMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(palette.sun, accent, 0.35),
    LANTERN_BASE_HDR,
    { name: 'cartLanternFlame' },
  );
  const flame = new THREE.Mesh(flameGeometry, flameMaterial);
  flame.name = 'cartLantern';
  flame.position.set(LANTERN_X, LANTERN_FLAME_Y, 0);
  tilt.add(flame);

  const shimmerSprite = radialSprite(48, 2.6);
  const shimmerMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(accent, palette.sun, 0.4), 1.5, {
    map: shimmerSprite,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'cartPushShimmer',
  });
  shimmerMaterial.opacity = 0;
  const shimmerGeometry = new THREE.PlaneGeometry(0.13, 0.062);
  shimmerGeometry.rotateX(-Math.PI / 2);
  const shimmerLeft = new THREE.Mesh(shimmerGeometry, shimmerMaterial);
  shimmerLeft.name = 'cartShimmerLeft';
  shimmerLeft.position.set(-0.1, 0.006, 0);
  shimmerLeft.visible = false;
  const shimmerRight = new THREE.Mesh(shimmerGeometry, shimmerMaterial);
  shimmerRight.name = 'cartShimmerRight';
  shimmerRight.position.set(0.1, 0.006, 0);
  shimmerRight.visible = false;
  tilt.add(shimmerLeft, shimmerRight);

  const ribbonVerts = (RIBBON_SEGMENTS + 1) * 2;
  const ribbonPos = new Float32Array(ribbonVerts * 3);
  const ribbonCol = new Float32Array(ribbonVerts * 4);
  const ribbonIdx = new Uint16Array(RIBBON_SEGMENTS * 6);
  for (let i = 0; i < RIBBON_SEGMENTS; i++) {
    const a = i * 2;
    ribbonIdx[i * 6] = a;
    ribbonIdx[i * 6 + 1] = a + 2;
    ribbonIdx[i * 6 + 2] = a + 1;
    ribbonIdx[i * 6 + 3] = a + 1;
    ribbonIdx[i * 6 + 4] = a + 2;
    ribbonIdx[i * 6 + 5] = a + 3;
  }
  const ribbonGeometry = new THREE.BufferGeometry();
  const ribbonPosAttr = new THREE.BufferAttribute(ribbonPos, 3);
  ribbonPosAttr.setUsage(THREE.DynamicDrawUsage);
  const ribbonColAttr = new THREE.BufferAttribute(ribbonCol, 4);
  ribbonColAttr.setUsage(THREE.DynamicDrawUsage);
  ribbonGeometry.setAttribute('position', ribbonPosAttr);
  ribbonGeometry.setAttribute('color', ribbonColAttr);
  ribbonGeometry.setIndex(new THREE.BufferAttribute(ribbonIdx, 1));
  ribbonGeometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(-0.3, 0.55, 0), 4);
  ribbonGeometry.name = 'cartMomentumTrail';
  const ribbonMaterial: EmissiveMaterial = emissiveMaterial(accent, 1.7, {
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    name: 'cartMomentumTrailMaterial',
  });
  ribbonMaterial.vertexColors = true;
  const ribbon = new THREE.Mesh(ribbonGeometry, ribbonMaterial);
  ribbon.name = 'cartMomentumTrail';
  ribbon.frustumCulled = false;
  ribbon.renderOrder = 3;
  group.add(ribbon);

  const dust: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'dust',
    count: 150,
    bounds: { halfX: 0.085, halfY: 0.05, halfZ: 0.055 },
    seed: seed ^ 0x3d91,
    color: blendTokens(palette.groundB, palette.sun, 0.3),
    size: 0.11,
    intensity: 0.95,
    opacity: 0.55,
    rise: 0.05,
    swirl: 0.03,
    name: 'cartScree',
  });
  dust.setDensity(0);
  group.add(dust.points);

  const lanternWorld = new THREE.Vector3();
  let dustKick = 0;

  return {
    group,
    root,
    lanternWorld,

    setState(pos, vel, force) {
      const height = heightAt(pos);
      const theta = Math.atan(slopeAt(pos));
      root.position.set(pos, height, 0);
      tilt.rotation.z = theta;

      const roll = -pos / WHEEL_R;
      wheelFront.rotation.z = roll;
      wheelRear.rotation.z = roll;

      const speed01 = clampRange(Math.abs(vel) / MC_MAX_SPEED, 0, 1);

      const pushing = force !== 0 && speed01 > 0.02;
      shimmerLeft.visible = pushing && force > 0;
      shimmerRight.visible = pushing && force < 0;
      shimmerMaterial.opacity = pushing ? 0.2 + 0.42 * speed01 : 0;

      const dir = vel >= 0 ? -1 : 1;
      const len = RIBBON_MAX_LEN * speed01;
      for (let i = 0; i <= RIBBON_SEGMENTS; i++) {
        const k = i / RIBBON_SEGMENTS;
        const x = pos + dir * len * k;
        const h = heightAt(x);
        const s = slopeAt(x);
        const inv = 1 / Math.sqrt(s * s + 1);
        const nx = -s * inv;
        const ny = inv;
        const halfW = 0.03 * Math.pow(1 - k, 0.55) * (0.35 + 0.65 * speed01);
        const cx = x + nx * RIBBON_LIFT;
        const cy = h + ny * RIBBON_LIFT;
        const a = i * 2;
        ribbonPos[a * 3] = cx - nx * halfW;
        ribbonPos[a * 3 + 1] = cy - ny * halfW;
        ribbonPos[a * 3 + 2] = 0;
        ribbonPos[(a + 1) * 3] = cx + nx * halfW;
        ribbonPos[(a + 1) * 3 + 1] = cy + ny * halfW;
        ribbonPos[(a + 1) * 3 + 2] = 0;
        const fade = Math.pow(1 - k, 1.5);
        const alpha = fade * Math.pow(speed01, 0.8);
        const lum = 0.55 + 0.45 * fade;
        for (let e = 0; e < 2; e++) {
          const v = (a + e) * 4;
          ribbonCol[v] = lum;
          ribbonCol[v + 1] = lum;
          ribbonCol[v + 2] = lum;
          ribbonCol[v + 3] = alpha;
        }
      }
      ribbonPosAttr.needsUpdate = true;
      ribbonColAttr.needsUpdate = true;
      ribbon.visible = speed01 > 0.015;

      dust.points.position.set(pos, height + 0.01, 0);
      dust.setDensity(clampRange(speed01 * 0.5 + dustKick * 0.85, 0, 1));

      const c = Math.cos(theta);
      const s2 = Math.sin(theta);
      lanternWorld.set(
        pos + LANTERN_X * c - LANTERN_FLAME_Y * s2,
        height + LANTERN_X * s2 + LANTERN_FLAME_Y * c,
        0,
      );
    },

    setTime(t) {
      dust.setTime(t);
    },

    setDustKick(env01) {
      dustKick = env01 < 0 ? 0 : env01 > 1 ? 1 : env01;
    },

    setLanternGain(gain) {
      setEmissiveIntensity(flameMaterial, LANTERN_BASE_HDR * (gain > 0 ? gain : 0));
    },

    dispose() {
      bodyGeometry.dispose();
      wheelGeometry.dispose();
      bodyMaterial.dispose();
      flameGeometry.dispose();
      flameMaterial.dispose();
      shimmerGeometry.dispose();
      shimmerMaterial.dispose();
      shimmerSprite.dispose();
      ribbonGeometry.dispose();
      ribbonMaterial.dispose();
      dust.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function makeAxleGeometry(rim: THREE.Color, hub: THREE.Color): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const hubBar = new THREE.CylinderGeometry(0.0055, 0.0055, WHEEL_Z * 2 + WHEEL_T, 6);
  hubBar.rotateX(Math.PI / 2);
  parts.push({ geometry: hubBar, color: hub });
  for (const z of [-WHEEL_Z, WHEEL_Z]) {
    const tyre = new THREE.CylinderGeometry(WHEEL_R, WHEEL_R, WHEEL_T, 14, 1, true);
    tyre.rotateX(Math.PI / 2);
    tyre.translate(0, 0, z);
    parts.push({ geometry: tyre, color: rim });
    const disc = new THREE.CylinderGeometry(WHEEL_R * 0.9, WHEEL_R * 0.9, WHEEL_T * 0.5, 14);
    disc.rotateX(Math.PI / 2);
    disc.translate(0, 0, z);
    parts.push({ geometry: disc, color: hub });
    for (let s = 0; s < 3; s++) {
      const spoke = new THREE.BoxGeometry(WHEEL_R * 1.86, 0.0055, WHEEL_T * 0.92);
      spoke.rotateZ((s / 3) * Math.PI);
      spoke.translate(0, 0, z);
      parts.push({ geometry: spoke, color: rim });
    }
  }
  return mergeColoredParts(parts, 'cartAxle');
}
