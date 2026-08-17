
import { emissiveMaterial, oakMaterial, setEmissiveIntensity, type EmissiveMaterial, type OakMaterial } from '@kit/materials';
import { clamp01, smoothstep01 } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { radialSprite } from '@kit/textures';
import * as THREE from 'three';
import type { ForestSkin } from './world';


export interface GoalBeaconOptions {
  glow: THREE.ColorRepresentation;
  name?: string;
}

export interface GoalBeaconHandle {
  group: THREE.Object3D;
  setTime(t: number): void;
  dispose(): void;
}

export function buildGoalBeacon(opts: GoalBeaconOptions): GoalBeaconHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'goalBeacon';
  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.Material[] = [];

  const BEAM_HEIGHT = 4.6;
  const beamGeo = new THREE.CylinderGeometry(0.045, 0.13, BEAM_HEIGHT, 8, 1, true);
  beamGeo.translate(0, BEAM_HEIGHT / 2, 0);
  geometries.push(beamGeo);
  const beamMaterial = emissiveMaterial(opts.glow, 0.9, {
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    fog: false,
    name: 'beacon:goalBeamMaterial',
  });
  materials.push(beamMaterial);
  const beam = new THREE.Mesh(beamGeo, beamMaterial);
  beam.name = 'beacon:goalBeam';
  group.add(beam);

  const ringTexture = radialSprite(64, 2.6);
  const ringGeo = new THREE.PlaneGeometry(1.6, 1.6);
  ringGeo.rotateX(-Math.PI / 2);
  ringGeo.translate(0, 0.03, 0);
  geometries.push(ringGeo);
  const ringMaterial = emissiveMaterial(opts.glow, 0.8, {
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    map: ringTexture,
    fog: false,
    name: 'beacon:goalRingMaterial',
  });
  materials.push(ringMaterial);
  const ring = new THREE.Mesh(ringGeo, ringMaterial);
  ring.name = 'beacon:goalRing';
  group.add(ring);

  const crownGeo = new THREE.IcosahedronGeometry(0.09, 1);
  geometries.push(crownGeo);
  const crownMaterial = emissiveMaterial(opts.glow, 1.6, { fog: false, name: 'beacon:goalCrownMaterial' });
  materials.push(crownMaterial);
  const crown = new THREE.Mesh(crownGeo, crownMaterial);
  crown.name = 'beacon:goalCrown';
  crown.position.y = BEAM_HEIGHT;
  group.add(crown);

  return {
    group,
    setTime(t) {
      const breath = 0.82 + 0.18 * Math.sin(t * 1.4);
      beamMaterial.opacity = 0.34 * breath;
      ringMaterial.opacity = 0.5 * breath;
      crown.position.y = BEAM_HEIGHT + 0.12 * Math.sin(t * 1.4);
    },
    dispose() {
      for (const g of geometries) g.dispose();
      for (const m of materials) m.dispose();
      ringTexture.dispose();
    },
  };
}


export interface MoonflowerOptions {
  skin: ForestSkin;
  glow: THREE.ColorRepresentation;
  name?: string;
}

export interface MoonflowerHandle {
  root: THREE.Object3D;
  bloom: THREE.Object3D;
  readonly crownHeight: number;
  setState(open: number, time: number): void;
  flare(strength?: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

const CROWN_HEIGHT = 0.3;

export function buildMoonflower(opts: MoonflowerOptions): MoonflowerHandle {
  const { skin } = opts;
  const detail = 1;
  const root = new THREE.Object3D();
  root.name = opts.name ?? 'moonflowerRoot';
  const body = new THREE.Object3D();
  body.name = 'moonflowerBloom';

  const geometries: THREE.BufferGeometry[] = [];

  const petals = Math.max(4, Math.round(6 * detail));
  const parts: ColoredPart[] = [];
  const stem = new THREE.CylinderGeometry(0.016, 0.026, CROWN_HEIGHT, Math.max(4, Math.round(6 * detail)), 1);
  stem.translate(0, CROWN_HEIGHT / 2, 0);
  parts.push({ geometry: stem, color: skin.fern });
  for (let i = 0; i < petals; i++) {
    const a = (i / petals) * Math.PI * 2;
    const petal = new THREE.ConeGeometry(0.075, 0.23, 4, 1);
    petal.scale(1, 1, 0.42);
    petal.rotateX(Math.PI * 0.5);
    petal.rotateZ(0.34);
    petal.rotateY(a);
    petal.translate(Math.sin(a) * 0.085, CROWN_HEIGHT + 0.015, Math.cos(a) * 0.085);
    parts.push({ geometry: petal, color: i % 2 === 0 ? skin.bloom : skin.stone });
  }
  const petalGeo = mergeColoredParts(parts, 'moonflowerPetals');
  geometries.push(petalGeo);
  const petalMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.72,
    rimColor: opts.glow,
    rimStrength: 0.65,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.3,
    name: 'moonflowerPetalMaterial',
  });
  const petalMesh = new THREE.Mesh(petalGeo, petalMaterial);
  petalMesh.name = 'moonflowerPetals';
  petalMesh.castShadow = true;
  body.add(petalMesh);

  const heartGeo = new THREE.IcosahedronGeometry(0.055, 1);
  geometries.push(heartGeo);
  const heartMaterial = emissiveMaterial(opts.glow, 1.15, { name: 'moonflowerHeart' });
  const heart = new THREE.Mesh(heartGeo, heartMaterial);
  heart.name = 'moonflowerHeart';
  heart.position.y = CROWN_HEIGHT + 0.05;
  body.add(heart);

  const haloGeo = new THREE.IcosahedronGeometry(0.115, 1);
  geometries.push(haloGeo);
  const haloMaterial = emissiveMaterial(opts.glow, 0.55, {
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'moonflowerHalo',
  });
  const halo = new THREE.Mesh(haloGeo, haloMaterial);
  halo.name = 'moonflowerHalo';
  halo.position.y = CROWN_HEIGHT + 0.05;
  body.add(halo);

  const poolTexture = radialSprite(64, 2.2);
  const poolGeo = new THREE.PlaneGeometry(1.5, 1.5);
  poolGeo.rotateX(-Math.PI / 2);
  geometries.push(poolGeo);
  const poolMaterial = emissiveMaterial(opts.glow, 0.55, {
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    map: poolTexture,
    name: 'moonflowerPool',
  });
  const pool = new THREE.Mesh(poolGeo, poolMaterial);
  pool.name = 'moonflowerPool';
  pool.position.y = 0.02;
  pool.renderOrder = 1;
  body.add(pool);

  let flareAmount = 0;
  let open = 0;

  return {
    root,
    bloom: body,
    crownHeight: CROWN_HEIGHT + 0.05,
    setState(o, time) {
      open = clamp01(o);
      const e = smoothstep01(open);
      body.scale.set(e, e * (0.35 + 0.65 * e), e);
      body.visible = e > 0.002;
      const breath = 1 + 0.1 * Math.sin(time * 1.6);
      petalMesh.rotation.y = time * 0.18;
      halo.scale.setScalar(breath * (1 + 0.5 * flareAmount));
      pool.scale.setScalar(breath * (0.9 + 0.5 * flareAmount));
    },
    flare(strength = 1) {
      flareAmount = Math.max(flareAmount, strength);
    },
    update(dt, time) {
      if (flareAmount > 0) flareAmount = Math.max(0, flareAmount - dt / 0.7);
      const breath = 1 + 0.16 * Math.sin(time * 1.9);
      const gain = breath * (1 + 2.2 * flareAmount) * (0.35 + 0.65 * open);
      setEmissiveIntensity(heartMaterial, 1.15 * gain);
      setEmissiveIntensity(haloMaterial, 0.55 * gain);
      setEmissiveIntensity(poolMaterial, 0.55 * gain);
    },
    dispose() {
      for (const g of geometries) g.dispose();
      geometries.length = 0;
      petalMaterial.dispose();
      heartMaterial.dispose();
      haloMaterial.dispose();
      poolMaterial.dispose();
      poolTexture.dispose();
      body.removeFromParent();
      body.clear();
      root.removeFromParent();
      root.clear();
    },
  };
}


export interface MothOptions {
  color: THREE.ColorRepresentation;
  name?: string;
}

export interface MothHandle {
  group: THREE.Group;
  readonly position: THREE.Vector3;
  setVisibility(v: number): void;
  setTime(time: number): void;
  flare(strength?: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}


export interface WispAuraHandle {
  mesh: THREE.Mesh;
  face(camera: THREE.Camera): void;
  setGain(gain: number): void;
  dispose(): void;
}

export function buildWispAura(color: THREE.ColorRepresentation, size: number): WispAuraHandle {
  const texture = radialSprite(128, 1.5);
  const geometry = new THREE.PlaneGeometry(size, size);
  geometry.name = 'wispAura';
  const material = emissiveMaterial(color, 0.95, {
    transparent: true,
    opacity: 0.38,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    map: texture,
    name: 'wispAura',
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'wispAura';
  mesh.renderOrder = 2;
  mesh.frustumCulled = false;
  return {
    mesh,
    face(camera) {
      mesh.quaternion.copy(camera.quaternion);
    },
    setGain(gain) {
      setEmissiveIntensity(material, 0.95 * Math.max(0, gain));
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      texture.dispose();
      mesh.removeFromParent();
    },
  };
}

export interface LanternPoolHandle {
  mesh: THREE.Mesh;
  material: EmissiveMaterial;
  readonly radius: number;
  setGain(gain: number): void;
  dispose(): void;
}

export function buildLanternPool(color: THREE.ColorRepresentation, radius: number): LanternPoolHandle {
  const texture = radialSprite(128, 1.7);
  const geometry = new THREE.PlaneGeometry(radius * 2, radius * 2);
  geometry.rotateX(-Math.PI / 2);
  geometry.name = 'lanternPool';
  const material = emissiveMaterial(color, 0.2, {
    transparent: true,
    opacity: 0.42,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    map: texture,
    name: 'lanternPool',
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'lanternPool';
  mesh.position.y = 0.015;
  mesh.renderOrder = 1;
  mesh.frustumCulled = false;
  return {
    mesh,
    material,
    radius,
    setGain(gain) {
      setEmissiveIntensity(material, 0.2 * Math.max(0, gain));
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      texture.dispose();
      mesh.removeFromParent();
    },
  };
}
