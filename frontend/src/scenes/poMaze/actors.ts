import * as THREE from 'three/webgpu';
import { emissiveMaterial, sculptureMaterial, setEmissiveIntensity } from '@kit/materials';
import { clamp01, smoothstep01 } from '@kit/mix';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { radialSprite } from '@kit/textures';
import type { ForestSkin } from './world';

const BEAM_HEIGHT = 4.6;
const CROWN_HEIGHT = 0.3;
const PETALS = 6;

const glowing = { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } as const;

export function buildGoalBeacon(glow: THREE.ColorRepresentation) {
  const group = new THREE.Group();
  group.name = 'goalBeacon';
  const beamGeometry = new THREE.CylinderGeometry(0.045, 0.13, BEAM_HEIGHT, 8, 1, true).translate(0, BEAM_HEIGHT / 2, 0);
  const beamMaterial = emissiveMaterial(glow, 0.9, { ...glowing, opacity: 0.34, side: THREE.DoubleSide, name: 'goalBeam' });
  const ringTexture = radialSprite(64, 2.6);
  const ringGeometry = new THREE.PlaneGeometry(1.6, 1.6).rotateX(-Math.PI / 2).translate(0, 0.03, 0);
  const ringMaterial = emissiveMaterial(glow, 0.8, { ...glowing, opacity: 0.5, map: ringTexture, name: 'goalRing' });
  const crownGeometry = new THREE.IcosahedronGeometry(0.09, 1);
  const crownMaterial = emissiveMaterial(glow, 1.6, { name: 'goalCrown' });
  const beam = new THREE.Mesh(beamGeometry, beamMaterial);
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  const crown = new THREE.Mesh(crownGeometry, crownMaterial);
  beam.name = 'goalBeam';
  ring.name = 'goalRing';
  crown.name = 'goalCrown';
  group.add(beam, ring, crown);
  return {
    group,
    setTime(t: number) {
      const breath = 0.82 + 0.18 * Math.sin(t * 1.4);
      beamMaterial.opacity = 0.34 * breath;
      ringMaterial.opacity = 0.5 * breath;
      crown.position.y = BEAM_HEIGHT + 0.12 * Math.sin(t * 1.4);
    },
    dispose() {
      for (const g of [beamGeometry, ringGeometry, crownGeometry]) g.dispose();
      for (const m of [beamMaterial, ringMaterial, crownMaterial]) m.dispose();
      ringTexture.dispose();
      group.removeFromParent();
    },
  };
}

export function buildMoonflower(skin: ForestSkin) {
  const body = new THREE.Group();
  body.name = 'moonflowerBloom';
  const parts: ColoredPart[] = [];
  parts.push({ geometry: new THREE.CylinderGeometry(0.016, 0.026, CROWN_HEIGHT, 6, 1).translate(0, CROWN_HEIGHT / 2, 0), color: skin.fern });
  for (let i = 0; i < PETALS; i++) {
    const a = (i / PETALS) * Math.PI * 2;
    const petal = new THREE.ConeGeometry(0.075, 0.23, 4, 1).scale(1, 1, 0.42).rotateX(Math.PI * 0.5).rotateZ(0.34).rotateY(a);
    petal.translate(Math.sin(a) * 0.085, CROWN_HEIGHT + 0.015, Math.cos(a) * 0.085);
    parts.push({ geometry: petal, color: i % 2 === 0 ? skin.bloom : skin.stone });
  }
  const petalGeometry = mergeColoredParts(parts, 'moonflowerPetals');
  const petalMaterial = sculptureMaterial({ roughness: 0.72, rimColor: skin.bloom, rimStrength: 0.65, toonSteps: 2, toonSoftness: 0.34, toonFloor: 0.3, name: 'moonflowerPetals' });
  const petals = new THREE.Mesh(petalGeometry, petalMaterial);
  petals.name = 'moonflowerPetals';
  petals.castShadow = true;

  const heartGeometry = new THREE.IcosahedronGeometry(0.055, 1);
  const heartMaterial = emissiveMaterial(skin.bloom, 1.15, { name: 'moonflowerHeart' });
  const heart = new THREE.Mesh(heartGeometry, heartMaterial);
  heart.name = 'moonflowerHeart';
  heart.position.y = CROWN_HEIGHT + 0.05;

  const haloGeometry = new THREE.IcosahedronGeometry(0.115, 1);
  const haloMaterial = emissiveMaterial(skin.bloom, 0.55, { ...glowing, opacity: 0.28, name: 'moonflowerHalo' });
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.name = 'moonflowerHalo';
  halo.position.y = CROWN_HEIGHT + 0.05;

  const poolTexture = radialSprite(64, 2.2);
  const poolGeometry = new THREE.PlaneGeometry(1.5, 1.5).rotateX(-Math.PI / 2);
  const poolMaterial = emissiveMaterial(skin.bloom, 0.55, { ...glowing, opacity: 0.5, map: poolTexture, name: 'moonflowerPool' });
  const pool = new THREE.Mesh(poolGeometry, poolMaterial);
  pool.name = 'moonflowerPool';
  pool.position.y = 0.02;
  pool.renderOrder = 1;

  body.add(petals, heart, halo, pool);
  let flareAmount = 0;
  let open = 0;
  return {
    body,
    crownHeight: CROWN_HEIGHT + 0.05,
    setState(amount: number, time: number) {
      open = clamp01(amount);
      const e = smoothstep01(open);
      body.scale.set(e, e * (0.35 + 0.65 * e), e);
      body.visible = e > 0.002;
      const breath = 1 + 0.1 * Math.sin(time * 1.6);
      petals.rotation.y = time * 0.18;
      halo.scale.setScalar(breath * (1 + 0.5 * flareAmount));
      pool.scale.setScalar(breath * (0.9 + 0.5 * flareAmount));
    },
    flare(strength: number) {
      flareAmount = Math.max(flareAmount, strength);
    },
    update(dt: number, time: number) {
      if (flareAmount > 0) flareAmount = Math.max(0, flareAmount - dt / 0.7);
      const gain = (1 + 0.16 * Math.sin(time * 1.9)) * (1 + 2.2 * flareAmount) * (0.35 + 0.65 * open);
      setEmissiveIntensity(heartMaterial, 1.15 * gain);
      setEmissiveIntensity(haloMaterial, 0.55 * gain);
      setEmissiveIntensity(poolMaterial, 0.55 * gain);
    },
    dispose() {
      for (const g of [petalGeometry, heartGeometry, haloGeometry, poolGeometry]) g.dispose();
      for (const m of [petalMaterial, heartMaterial, haloMaterial, poolMaterial]) m.dispose();
      poolTexture.dispose();
      body.removeFromParent();
    },
  };
}

function glowPlane(name: string, color: THREE.ColorRepresentation, size: number, falloff: number, intensity: number, opacity: number) {
  const texture = radialSprite(128, falloff);
  const geometry = new THREE.PlaneGeometry(size, size);
  const material = emissiveMaterial(color, intensity, { ...glowing, opacity, map: texture, name });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = name;
  mesh.frustumCulled = false;
  return {
    mesh,
    setGain(gain: number) {
      setEmissiveIntensity(material, intensity * Math.max(0, gain));
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      texture.dispose();
      mesh.removeFromParent();
    },
  };
}

export function buildWispAura(color: THREE.ColorRepresentation, size: number) {
  const aura = glowPlane('wispAura', color, size, 1.5, 0.95, 0.38);
  aura.mesh.renderOrder = 2;
  return aura;
}

export function buildLanternPool(color: THREE.ColorRepresentation, radius: number) {
  const pool = glowPlane('lanternPool', color, radius * 2, 1.7, 0.2, 0.42);
  pool.mesh.geometry.rotateX(-Math.PI / 2);
  pool.mesh.renderOrder = 1;
  return pool;
}
