import * as THREE from 'three';
import { makeHeightField } from '@kit/heightfield';
import { makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { emissiveMaterial } from '@kit/materials';
import { blendTokens, TOKEN_WHITE } from '@kit/mix';
import { lamp, propMaterials } from '@kit/props';
import { radialSprite } from '@kit/textures';
import type { SceneContext } from '@kit/types';
import { scatterRocks, scatterTrees } from '@kit/vegetation';
import { makeWater } from '@kit/water';
import type { WorldCore } from '@kit/world';
import { parameter } from '../../data/envs';

export const TRACK_SCALE = 3.2;
export const LAKE_HALF_X = 13;
export const LAKE_HALF_Z = 6;
export const ICE_LEVEL = 0.02;
export const STAR_Y = 4.3;
export const STAR_Z = -2.6;
export const WISP_Z = 0.8;
const TRAIL = 40;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const palette = ctx.tokens.palettes.dawn, accent = ctx.tokens.accents['tracking'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const innovation = parameter(ctx.environment, 'innovation'), eta = parameter(ctx.environment, 'eta');
  const land = makeLandscape(scene, 'frozenLake', palette, 'dawn', {
    sky: { sunDir: [-0.82, 0.11, -0.56], sunSize: 0.05, sunHdr: 3, sunGlow: 0.18, sunGlowPower: 30, topSpread: 0.5, bottomSpread: 0.35, radius: 320 },
    light: { pointCount: 2, pointDistance: 6, sunDistanceScale: 3 },
    fogExtent: 120, fogDensity: 0.0038,
    weather: { preset: 'snow', count: Math.round(320 + innovation * 6200), seed: ctx.seed, color: 0xf4f8ff, size: 4.5, intensity: 1.1, opacity: 0.8, rise: -0.5 - innovation * 2.5, swirl: 0.25 + innovation * 1.6, bounds: { y: 3.2, halfX: LAKE_HALF_X + 3, halfY: 3.2, halfZ: LAKE_HALF_Z + 3 } },
  });
  const { world, sky, rig, fog } = land;
  sky.setColors(0x2f4f80, 0xb97f74, 0x6b6470);
  fog.color.set(0xa8969c);
  rig.sun.color.set(0xffc493); rig.sun.intensity = 1.25;
  rig.hemi.color.set(0x9fb8d8); rig.hemi.groundColor.set(0x5d6a7c); rig.hemi.intensity = 0.62;
  rig.setShadowFocus({ x: 0, y: 0, z: 0 }, 9);
  const field = makeHeightField({ base: 0, court: { halfX: LAKE_HALF_X + 0.8, halfZ: LAKE_HALF_Z + 0.8, rise: 0.42 }, octaves: [{ wavelength: 27, amplitude: 1.5, seed: ctx.seed }, { wavelength: 8, amplitude: 0.35, seed: ctx.seed ^ 0x77 }] });
  const heightAt = field.heightAt;
  const snowA = blendTokens(0xbccbdc, palette.skyHorizon, 0.15), snowB = blendTokens(0xe2eaf3, palette.hemiSky, 0.1);
  const terrain = world.own(world.time(makeSceneTerrain({
    follow: false, name: 'snowShore', seed: ctx.seed, colorA: snowA, colorB: snowB, segments: 128, radius: 380, height: field.glsl,
    detail: 'float drift = landNoise(p.xz * vec2(0.3, 0.12)); color *= 0.93 + 0.07 * drift + 0.04 * landNoise(p.xz * 2.1);',
  }, heightAt, { roughness: 1, rimColor: palette.sun, rimStrength: 0.2, toonSteps: 3, toonSoftness: 0.45, toonFloor: 0.3 })));
  world.add(terrain.mesh);
  const ice = world.own(world.time(makeWater({
    size: [LAKE_HALF_X * 2, LAKE_HALF_Z * 2], palette, segments: 48, level: ICE_LEVEL, bedHeight: () => -1.4, amplitude: 0.004, waveScale: 5, speed: 0.15, fresnel: 0.95, caustics: 0, depthFalloff: 0.9, opacity: 0.95,
    sunDir: sky.sunDirection, shallow: blendTokens(0x6d8fae, palette.skyHorizon, 0.08), deep: blendTokens(0x3a587a, palette.skyTop, 0.3), sky: blendTokens(palette.skyHorizon, 0x9fbcd8, 0.55), name: 'lakeIce',
  })));
  ice.uniforms.uSpecular.value = 0.45;
  world.add(ice.mesh);
  const shore = (x: number, z: number): boolean => Math.abs(x) < LAKE_HALF_X + 1.2 && Math.abs(z) < LAKE_HALF_Z + 1.2;
  const spruce = scatterTrees({
    archetypes: ['conifer'], count: 120, area: { x: 0, z: -4, halfX: 24, halfZ: 14 }, seed: ctx.seed ^ 0x51ab, exclude: (x, z) => shore(x, z) || (z > 2 && Math.abs(x) < 16),
    heightAt, maxSlope: 0.7, color: blendTokens(0x5f7f6c, palette.skyHorizon, 0.12), scale: [0.7, 1.5], stretch: [1, 1.6], hueJitter: 0.14, wind: 0.015, rimColor: palette.sun, rimStrength: 0.35, castShadow: true, name: 'shoreSpruce',
  });
  world.add(spruce.group); world.time(world.own(spruce));
  const boulders = scatterRocks({
    count: 70, area: { x: 0, z: 0, halfX: 20, halfZ: 11 }, seed: ctx.seed ^ 0x2b17, exclude: shore, heightAt, color: blendTokens(0xe6edf5, palette.hemiSky, 0.1),
    scale: [0.2, 0.7], stretch: [0.5, 0.9], hueJitter: 0.08, rimColor: palette.sun, rimStrength: 0.35, castShadow: true, name: 'snowBoulders',
  });
  world.add(boulders.group); world.time(world.own(boulders));
  const materials = propMaterials(world);
  const lamps = [-1, 1].map(side => {
    const l = lamp(world, accent, 1.7, materials);
    l.group.position.set(side * 7, heightAt(side * 7, LAKE_HALF_Z + 1.4), LAKE_HALF_Z + 1.4);
    world.add(l.group);
    l.lit(0.55);
    return l;
  });
  const starMaterial = world.own(emissiveMaterial(blendTokens(gold, TOKEN_WHITE, 0.45), 6, { name: 'driftingStar' }));
  const star = new THREE.Mesh(world.own(new THREE.IcosahedronGeometry(0.28, 1)), starMaterial);
  star.name = 'driftingStar';
  star.position.set(0, STAR_Y, STAR_Z);
  world.add(star);
  const sprite = world.own(radialSprite(64, 2.2));
  const haloMaterial = world.own(emissiveMaterial(blendTokens(gold, TOKEN_WHITE, 0.3), 1.3, { map: sprite, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, name: 'starHalo' }));
  const haloGeometry = world.own(new THREE.PlaneGeometry(1, 1));
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  halo.scale.setScalar(1.5 + innovation * 5);
  star.add(halo);
  const column = new THREE.Mesh(world.own(new THREE.CylinderGeometry(0.04, 0.04, STAR_Y - ICE_LEVEL, 8).translate(0, (STAR_Y + ICE_LEVEL) / 2, 0)), world.own(emissiveMaterial(blendTokens(gold, TOKEN_WHITE, 0.3), 1.4, { transparent: true, opacity: 0.4, depthWrite: false, blending: THREE.AdditiveBlending, name: 'starPlumb' })));
  column.name = 'starPlumb';
  column.position.set(0, 0, STAR_Z);
  world.add(column);
  const ghostMaterial = world.own(emissiveMaterial(blendTokens(gold, palette.skyHorizon, 0.5), 1.2, { map: sprite, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, name: 'observedStar' }));
  const ghost = new THREE.Mesh(haloGeometry, ghostMaterial);
  ghost.name = 'observedStar';
  ghost.position.set(0, STAR_Y, STAR_Z + 0.2);
  ghost.scale.setScalar(1.8);
  world.add(ghost);
  const trailPositions = new Float32Array(TRAIL * 3);
  const trailAttribute = new THREE.BufferAttribute(trailPositions, 3);
  trailAttribute.setUsage(THREE.DynamicDrawUsage);
  const trailGeometry = world.own(new THREE.BufferGeometry().setAttribute('position', trailAttribute));
  const trailMaterial = world.own(new THREE.LineBasicMaterial({ color: blendTokens(gold, TOKEN_WHITE, 0.2), transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
  const trail = new THREE.Line(trailGeometry, trailMaterial);
  trail.frustumCulled = false;
  world.add(trail);
  rig.points.forEach((point, i) => { point.color.set(i === 0 ? gold : accent); point.intensity = i === 0 ? 5 : 1.6; });
  const resetTrail = (x: number): void => { for (let i = 0; i < TRAIL; i++) { trailPositions[i * 3] = x; trailPositions[i * 3 + 1] = STAR_Y; trailPositions[i * 3 + 2] = STAR_Z; } trailAttribute.needsUpdate = true; };
  resetTrail(0);
  const extra = {
    accent, palette, heightAt, innovation, eta, lamps, star, ghost, ghostMaterial, halo, wispLight: rig.points[1],
    setView(coverage: Parameters<typeof land.setView>[0], camera: THREE.Camera) { land.setView(coverage, camera, terrain); halo.quaternion.copy(camera.quaternion); ghost.quaternion.copy(camera.quaternion); },
    placeStar(x: number, observed: number, snap: boolean) {
      star.position.x = x;
      column.position.x = x;
      ghost.position.x = observed;
      const light = rig.points[0]; if (light) light.position.set(x, STAR_Y - 1.2, STAR_Z + 0.6);
      if (snap) resetTrail(x);
      else { trailPositions.copyWithin(3, 0, (TRAIL - 1) * 3); trailPositions[0] = x; trailAttribute.needsUpdate = true; }
    },
  };
  return world.seal<WorldCore & typeof extra>(extra);
}

export type FrozenLakeWorld = ReturnType<typeof buildWorld>;
