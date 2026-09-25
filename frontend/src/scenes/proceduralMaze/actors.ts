import * as THREE from 'three/webgpu';
import { abs, dot, mix, normalView, positionViewDirection, pow, sin, smoothstep, uniform, uv } from 'three/tsl';
import { emissiveMaterial, oakUniforms, sculptureMaterial, setEmissiveIntensity, type EmissiveMaterial, type OakMaterial } from '@kit/materials';
import { clamp01 } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { radialSprite } from '@kit/textures';
import type { Palette } from '@kit/tokens';
import type { RuinSkin } from './world';

export const BEAM_LENGTH = 4.4;
export const BEAM_BASE_RADIUS = 0.4;
export const BEAM_TOP_RADIUS = 0.62;
export const BEAM_HDR = 0.86;
export const POOL_HDR = 0.62;
export const HALO_HDR = 0.34;
const LIGHT_HEIGHT = 0.55;

export interface SunShaftOptions {
  palette: Palette;
  skin: RuinSkin;
  sunDir: readonly [number, number, number];
  name?: string;
}

export interface SunShaftHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  readonly lightHeight: number;
  setState(grow01: number): void;
  flare(strength?: number): void;
  setTime(t: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

export function buildSunShaft(opts: SunShaftOptions): SunShaftHandle {
  const { palette, skin, sunDir } = opts;
  const group = new THREE.Group();
  group.name = opts.name ?? 'sunShaft';

  const root = new THREE.Object3D();
  root.name = 'sunShaftRoot';
  group.add(root);

  const decor = new THREE.Group();
  decor.name = 'sunShaftDecor';
  root.add(decor);

  const beamGeometry = new THREE.CylinderGeometry(BEAM_TOP_RADIUS, BEAM_BASE_RADIUS, BEAM_LENGTH, 22, 1, true);
  beamGeometry.translate(0, BEAM_LENGTH / 2, 0);
  beamGeometry.name = 'sunShaftBeam';

  const beamIntensity = uniform(BEAM_HDR), beamTime = uniform(0);
  const beamMaterial = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  const axial = uv().y;
  beamMaterial.colorNode = uniform(new THREE.Color(skin.sunWarm)).mul(beamIntensity);
  beamMaterial.opacityNode = pow(abs(dot(normalView, positionViewDirection)), 1.35)
    .mul(mix(0.6, 1, axial))
    .mul(smoothstep(0.5, 1, axial).oneMinus())
    .mul(sin(axial.mul(11).sub(beamTime.mul(0.55))).mul(0.13).add(0.87))
    .mul(0.26);
  beamMaterial.alphaTest = 0.004;
  beamMaterial.name = 'sunShaftBeamMaterial';

  const beam = new THREE.Mesh(beamGeometry, beamMaterial);
  beam.name = 'sunShaftBeam';
  beam.frustumCulled = false;
  beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(sunDir[0], sunDir[1], sunDir[2]).normalize());
  decor.add(beam);

  const sprite = radialSprite(64, 2.1);
  sprite.name = 'sunShaftPoolSprite';
  const poolGeometry = new THREE.PlaneGeometry(1.15, 1.15);
  poolGeometry.rotateX(-Math.PI / 2);
  poolGeometry.name = 'sunShaftPool';
  const poolMaterial: EmissiveMaterial = emissiveMaterial(skin.sunWarm, POOL_HDR, {
    map: sprite,
    transparent: true,
    opacity: 0.42,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'sunShaftPoolMaterial',
  });
  const pool = new THREE.Mesh(poolGeometry, poolMaterial);
  pool.name = 'sunShaftPool';
  pool.position.y = 0.022;
  pool.renderOrder = 2;
  decor.add(pool);

  const haloGeometry = new THREE.PlaneGeometry(2.5, 2.5);
  haloGeometry.rotateX(-Math.PI / 2);
  haloGeometry.name = 'sunShaftHalo';
  const haloMaterial: EmissiveMaterial = emissiveMaterial(skin.sunWarm, HALO_HDR, {
    map: sprite,
    transparent: true,
    opacity: 0.26,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'sunShaftHaloMaterial',
  });
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.name = 'sunShaftHalo';
  halo.position.y = 0.014;
  halo.renderOrder = 1;
  decor.add(halo);

  const stubGeometry = atriumGeometry(skin);
  const stubMaterial: OakMaterial = sculptureMaterial({
    roughness: 0.94,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSoftness: 0.28,
    toonFloor: 0.42,
    name: 'sunShaftStubMaterial',
  });
  const stubs = new THREE.Mesh(stubGeometry, stubMaterial);
  stubs.name = 'sunShaftStubs';
  stubs.castShadow = true;
  stubs.receiveShadow = true;
  decor.add(stubs);

  let flareAmount = 0;
  let grow = 1;
  let breath = 1;

  const applyIntensity = (): void => {
    const gain = breath * (1 + 1.9 * flareAmount) * (0.25 + 0.75 * grow);
    beamIntensity.value = BEAM_HDR * gain;
    setEmissiveIntensity(poolMaterial, POOL_HDR * gain);
    setEmissiveIntensity(haloMaterial, HALO_HDR * gain);
  };
  applyIntensity();

  return {
    group,
    root,
    lightHeight: LIGHT_HEIGHT,

    setState(grow01) {
      grow = clamp01(grow01);
      decor.scale.setScalar(grow);
      applyIntensity();
    },

    flare(strength = 1) {
      flareAmount = Math.max(flareAmount, strength);
      applyIntensity();
    },

    setTime(t) {
      beamTime.value = t;
      breath = 1 + 0.09 * Math.sin(t * 0.73);
      applyIntensity();
      oakUniforms(stubMaterial)?.setTime(t);
    },

    update(dt) {
      if (flareAmount > 0) {
        flareAmount = Math.max(0, flareAmount - Math.max(0, dt) / 0.6);
        applyIntensity();
      }
    },

    dispose() {
      beamGeometry.dispose();
      beamMaterial.dispose();
      poolGeometry.dispose();
      poolMaterial.dispose();
      haloGeometry.dispose();
      haloMaterial.dispose();
      sprite.dispose();
      stubGeometry.dispose();
      stubMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

function atriumGeometry(skin: RuinSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const corners: readonly (readonly [number, number, number])[] = [
    [-0.33, -0.33, 0.21],
    [0.33, -0.33, 0.13],
    [0.33, 0.33, 0.25],
    [-0.33, 0.33, 0.16],
  ];
  corners.forEach(([x, z, height], i) => {
    const stub = new THREE.CylinderGeometry(0.085, 0.105, height, 10, 1);
    stub.translate(0, height * 0.5, 0);
    stub.rotateZ((i % 2 === 0 ? 1 : -1) * 0.08);
    stub.translate(x, 0.03, z);
    parts.push({ geometry: stub, color: i % 2 === 0 ? skin.stoneLight : skin.stoneDark });
  });

  const ring = new THREE.RingGeometry(0.3, 0.42, 24);
  ring.rotateX(-Math.PI / 2);
  ring.translate(0, 0.028, 0);
  parts.push({ geometry: ring, color: skin.stoneLight });

  const slab = new THREE.BoxGeometry(0.62, 0.055, 0.62);
  slab.translate(0, 0.028, 0);
  parts.push({ geometry: slab, color: skin.flag });

  return mergeColoredParts(parts, 'sunShaftAtrium');
}
