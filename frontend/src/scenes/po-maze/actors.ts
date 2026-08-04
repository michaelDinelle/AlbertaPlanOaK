// The Fogbound Forest — the three things in it that are alive.
//
// THE MOONFLOWER is the goal, and it is built in TWO PIECES because the agent's knowledge
// of the goal comes in two pieces. `root` is an empty anchor on `cellToWorld(gx, gy)`
// carrying position only — the alignment probe, and the thing the ground-truth overlay
// agrees with; it never draws anything, because POMaze never tells the agent where the
// goal IS. `bloom` is the visible flower, and it is NOT a child of the root: the scene
// stands it on the cell a +1 was earned on, for the length of the goal breath. That is the
// one instant the location is a fact the agent owns — it just collected the reward there —
// so it is the one instant a flower may be drawn.
//
// THE MOTH is the compass. POMaze's observation carries four bits — gx>x, gx<x, gy>y,
// gy<y — and NOT the distance, so the moth flies on the rim of the visibility bubble in
// the direction those bits give, at a radius that is a function of the clock alone, and
// never one centimetre nearer for being close. It is decoration by construction (it has no
// probe) precisely because its position is not a state the env owns: it is the SIGN of a
// state.
//
// THE LANTERN POOL is the wisp's own light on the litter — the warm disc that makes the
// bubble a place rather than a radius. It follows the avatar and nothing else.

import { emissiveMaterial, oakMaterial, setEmissiveIntensity, type EmissiveMaterial, type OakMaterial } from '@kit/materials';
import { radialSprite } from '@kit/textures';
import type { QualityTier } from '@kit/types';
import * as THREE from 'three';
import { clamp01, mergeColoredParts, smoothstep01, type ColoredPart, type ForestSkin } from './world';

// ---------------------------------------------------------------------------
// the moonflower — the goal
// ---------------------------------------------------------------------------

export interface MoonflowerOptions {
  tier: QualityTier;
  skin: ForestSkin;
  /** the emissive heart color (accent-derived) */
  glow: THREE.ColorRepresentation;
  name?: string;
}

export interface MoonflowerHandle {
  /** PROBE OBJECT: the live goal cell, position only, y = 0 — and it never draws. */
  root: THREE.Object3D;
  /**
   * Everything visible, parented to the SCENE and not to the root: the scene stands it on
   * the clearing a +1 was earned on, which is the only goal position the agent is ever
   * given. `open` scales it; the root's exactness is untouched by any of that.
   */
  bloom: THREE.Object3D;
  /** world height of the flower's heart, for the point light */
  readonly crownHeight: number;
  /** `open` in [0,1] is the goal breath; `flare` is the celebration hook */
  setState(open: number, time: number): void;
  flare(strength?: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

/** the heart sits here above the litter */
const CROWN_HEIGHT = 0.3;

/**
 * A moonflower: a ring of pale petals cupped around a luminous heart, on a short stem,
 * with a glow pooled on the litter under it. The agent never sees it coming — it is drawn
 * only in the breath after a +1, on the cell where that +1 happened — so it is built to
 * open and read INSTANTLY through fog, not to reward inspection.
 */
export function buildMoonflower(opts: MoonflowerOptions): MoonflowerHandle {
  const { tier, skin } = opts;
  const detail = tier === 'thumb' ? 0.4 : tier === 'medium' ? 0.7 : 1;
  const root = new THREE.Object3D();
  root.name = opts.name ?? 'moonflowerRoot';
  // NOT a child of root: the anchor tracks the live goal, the flower stands on the
  // clearing. Parenting them would make the picture say where the next goal is.
  const body = new THREE.Object3D();
  body.name = 'moonflowerBloom';

  const geometries: THREE.BufferGeometry[] = [];

  // ---- petals + stem: one merged, vertex-colored prop ----
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
    tier,
    name: 'moonflowerPetalMaterial',
  });
  const petalMesh = new THREE.Mesh(petalGeo, petalMaterial);
  petalMesh.name = 'moonflowerPetals';
  petalMesh.castShadow = tier !== 'thumb';
  body.add(petalMesh);

  // ---- the heart: the only HDR emitter down here ----
  const heartGeo = new THREE.IcosahedronGeometry(0.055, tier === 'thumb' ? 0 : 1);
  geometries.push(heartGeo);
  const heartMaterial = emissiveMaterial(opts.glow, 1.15, { name: 'moonflowerHeart' });
  const heart = new THREE.Mesh(heartGeo, heartMaterial);
  heart.name = 'moonflowerHeart';
  heart.position.y = CROWN_HEIGHT + 0.05;
  body.add(heart);

  const haloGeo = new THREE.IcosahedronGeometry(0.115, tier === 'thumb' ? 0 : 1);
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

  // ---- the pool of light it lays on the litter ----
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
      // the flower does not just appear: it opens out of the litter as the fog lifts
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

// ---------------------------------------------------------------------------
// the moth — the compass, and the ONLY thing that knows where the goal is
// ---------------------------------------------------------------------------

export interface MothOptions {
  tier: QualityTier;
  color: THREE.ColorRepresentation;
  name?: string;
}

export interface MothHandle {
  group: THREE.Group;
  /** where the light rig should hang its point light */
  readonly position: THREE.Vector3;
  /** 0 hides it entirely (the goal is on the agent's own cell, or it has been found) */
  setVisibility(v: number): void;
  /** wings and body attitude, all on the playback clock */
  setTime(time: number): void;
  flare(strength?: number): void;
  update(dt: number, time: number): void;
  dispose(): void;
}

/**
 * A moth: a fat little body, two beating wings and a halo, all emissive. It is drawn
 * unlit and additively so it reads as a spark of light through fog rather than a model —
 * at two metres in this density a shaded insect would be a grey smudge.
 */
export function buildMoth(opts: MothOptions): MothHandle {
  const group = new THREE.Group();
  group.name = opts.name ?? 'compassMoth';
  const detail = opts.tier === 'thumb' ? 0 : 1;
  const geometries: THREE.BufferGeometry[] = [];

  const pivot = new THREE.Group();
  pivot.name = 'compassMothPivot';
  group.add(pivot);

  const bodyGeo = new THREE.IcosahedronGeometry(0.022, detail);
  bodyGeo.scale(0.7, 0.7, 1.5);
  geometries.push(bodyGeo);
  const bodyMaterial = emissiveMaterial(opts.color, 2.8, { name: 'compassMothBody' });
  const body = new THREE.Mesh(bodyGeo, bodyMaterial);
  body.name = 'compassMothBody';
  pivot.add(body);

  const wingGeo = new THREE.BufferGeometry();
  {
    // one triangle pair per wing, built in the xz plane hinged on x = 0. It is a MOTH,
    // not a bird: at 3 units through this fog anything larger reads as a lantern.
    const v = new Float32Array([
      0, 0, -0.012, 0.078, 0.009, -0.045, 0.06, 0.003, 0.033,
      0, 0, -0.012, 0.06, 0.003, 0.033, 0.021, 0, 0.027,
    ]);
    wingGeo.setAttribute('position', new THREE.BufferAttribute(v, 3));
    wingGeo.computeVertexNormals();
    wingGeo.name = 'compassMothWing';
  }
  geometries.push(wingGeo);
  const wingMaterial = emissiveMaterial(opts.color, 1.5, {
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    name: 'compassMothWing',
  });
  const wingL = new THREE.Mesh(wingGeo, wingMaterial);
  wingL.name = 'compassMothWingL';
  const wingR = new THREE.Mesh(wingGeo, wingMaterial);
  wingR.name = 'compassMothWingR';
  wingR.scale.x = -1;
  pivot.add(wingL, wingR);

  const haloGeo = new THREE.IcosahedronGeometry(0.058, detail);
  geometries.push(haloGeo);
  const haloMaterial = emissiveMaterial(opts.color, 0.9, {
    transparent: true,
    opacity: 0.3,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'compassMothHalo',
  });
  const halo = new THREE.Mesh(haloGeo, haloMaterial);
  halo.name = 'compassMothHalo';
  group.add(halo);

  let visibility = 0;
  let flareAmount = 0;

  return {
    group,
    position: group.position,
    setVisibility(v) {
      visibility = clamp01(v);
      const e = smoothstep01(visibility);
      group.scale.setScalar(e);
      group.visible = e > 0.002;
    },
    setTime(time) {
      // a beat, a roll and a slow yaw wander — every term on the playback clock, so a
      // paused frame freezes the moth mid-wingbeat instead of stalling it flat
      const beat = Math.sin(time * 11) * 0.6 + Math.sin(time * 23) * 0.12;
      wingL.rotation.z = -0.5 - beat;
      wingR.rotation.z = 0.5 + beat;
      pivot.rotation.y = Math.sin(time * 0.83) * 0.5;
      pivot.rotation.z = Math.sin(time * 1.7) * 0.16;
      halo.scale.setScalar(1 + 0.18 * Math.sin(time * 2.4) + 0.6 * flareAmount);
    },
    flare(strength = 1) {
      flareAmount = Math.max(flareAmount, strength);
    },
    update(dt, time) {
      if (flareAmount > 0) flareAmount = Math.max(0, flareAmount - dt / 0.6);
      const gain = (1 + 0.2 * Math.sin(time * 3.1)) * (1 + 2 * flareAmount);
      setEmissiveIntensity(bodyMaterial, 2.8 * gain);
      setEmissiveIntensity(wingMaterial, 1.5 * gain);
      setEmissiveIntensity(haloMaterial, 0.7 * gain * (0.5 + 0.5 * visibility));
    },
    dispose() {
      for (const g of geometries) g.dispose();
      geometries.length = 0;
      bodyMaterial.dispose();
      wingMaterial.dispose();
      haloMaterial.dispose();
      group.removeFromParent();
      group.clear();
    },
  };
}

// ---------------------------------------------------------------------------
// the lantern pool — the wisp's own light on the litter
// ---------------------------------------------------------------------------

export interface WispAuraHandle {
  mesh: THREE.Mesh;
  /** billboard it: the aura is scattering, and scattering has no orientation */
  face(camera: THREE.Camera): void;
  setGain(gain: number): void;
  dispose(): void;
}

/**
 * The halo of scattered light around the wisp.
 *
 * The kit avatar's own halo is a flat-shaded icosahedron, which at four metres in a
 * 40° frame renders as a visible faceted marble rather than a glow — this scene stands
 * closer to its avatar than any other, so it is the one scene where that shows. A
 * billboarded radial sprite in front of it does what fog actually does to a light source
 * and lets the orb stay small and hard-edged underneath.
 */
export function buildWispAura(color: THREE.ColorRepresentation, size: number, tier: QualityTier): WispAuraHandle {
  const texture = radialSprite(tier === 'thumb' ? 64 : 128, 1.5);
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
  /** radius in world units — the bubble made visible */
  readonly radius: number;
  setGain(gain: number): void;
  dispose(): void;
}

/**
 * The warm disc the wisp lays on the needle litter. Its radius is the bubble's OUTER
 * edge, so what the agent can see and what the frame lights up are the same circle.
 */
export function buildLanternPool(color: THREE.ColorRepresentation, radius: number, tier: QualityTier): LanternPoolHandle {
  const texture = radialSprite(tier === 'thumb' ? 64 : 128, 1.7);
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
