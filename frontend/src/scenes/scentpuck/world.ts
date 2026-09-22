import * as THREE from 'three';
import { makeHeightField } from '@kit/heightfield';
import { makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { oakMaterial } from '@kit/materials';
import { blendTokens } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { hoodooGeometry, propMaterials } from '@kit/props';
import { engravedPanel, modelParts } from '@kit/robot';
import { streamFor } from '@kit/rng';
import type { SceneContext } from '@kit/types';
import { scatterGrass, scatterRocks } from '@kit/vegetation';
import type { WorldCore } from '@kit/world';
import { SP_DECOY_FALLOFF, SP_HALF, SP_SCENT_FALLOFF } from '@sim/envs/scentpuck';
import { parameter } from '../../data/envs';

export const SCALE = 4;
export const HALF = SP_HALF * SCALE;
export const DECOY_TINT = 0x9a6fe0;

export const toWorldX = (x: number): number => x * SCALE;
export const toWorldZ = (y: number): number => -y * SCALE;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const palette = ctx.tokens.palettes.afternoon, accent = ctx.tokens.accents['scentpuck'] ?? ctx.tokens.celebrationGold;
  const competing = parameter(ctx.environment, 'competing') !== 0;
  const land = makeLandscape(scene, 'badlandsSand', palette, 'afternoon', {
    sky: { sunDir: [-0.62, 0.36, -0.7], sunSize: 0.045, sunHdr: 6.5, sunGlow: 0.48, sunGlowPower: 16, topSpread: 0.55, bottomSpread: 0.3, radius: 320 },
    light: { pointCount: 2, pointDistance: 3.6, sunDistanceScale: 3 },
    fogExtent: 110,
    weather: { preset: 'dust', count: 600, seed: ctx.seed, color: 0xf0d2a6, size: 1.15, intensity: 1.1, opacity: 0.32, bounds: { y: 0.9, halfX: HALF + 2, halfY: 0.9, halfZ: HALF + 2 } },
  });
  const { world, sky, rig } = land;
  sky.setColors(0x6fa2cf, 0xf1d9b0, 0xd9b98a);
  rig.sun.color.set(0xffd9a6); rig.sun.intensity = 2.5;
  rig.hemi.color.set(0xd8e2ea); rig.hemi.groundColor.set(0x8c6a48); rig.hemi.intensity = 0.8;
  rig.setShadowFocus({ x: 0, y: 0, z: 0 }, 5.2);
  const field = makeHeightField({ base: 0, court: { halfX: HALF + 0.7, halfZ: HALF + 0.7, rise: 0.36 }, octaves: [{ wavelength: 24, amplitude: 1.9, seed: ctx.seed }, { wavelength: 7, amplitude: 0.45, seed: ctx.seed ^ 0x5a }] });
  const heightAt = field.heightAt;
  const goal = { value: new THREE.Vector2() }, decoy = { value: new THREE.Vector2(9, 9) }, decoyOn = { value: competing ? 1 : 0 };
  const sandA = blendTokens(0xd8b98a, palette.groundA, 0.25), sandB = blendTokens(0xefd8ad, palette.groundB, 0.2);
  const accentColor = new THREE.Color(accent), decoyColor = new THREE.Color(DECOY_TINT);
  const terrain = world.own(world.time(makeSceneTerrain({
    follow: false, name: 'badlandsGround', seed: ctx.seed, colorA: sandA, colorB: sandB, segments: 144, radius: 380, height: field.glsl,
    detail: 'float strata = smoothstep(0.35, 0.65, fract(p.y * 2.6 + landNoise(p.xz * 0.6) * 0.5)); color *= 0.9 + 0.1 * strata + 0.05 * landNoise(p.xz * 2.7);',
  }, heightAt, { roughness: 0.96, rimColor: palette.sun, rimStrength: 0.18, toonSteps: 4, toonSoftness: 0.32, toonFloor: 0.12 })));
  const compile = terrain.material.onBeforeCompile;
  terrain.material.onBeforeCompile = (shader, renderer) => {
    compile(shader, renderer);
    Object.assign(shader.uniforms, { uGoal: goal, uDecoy: decoy, uDecoyOn: decoyOn });
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform vec2 uGoal;\nuniform vec2 uDecoy;\nuniform float uDecoyOn;')
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
vec2 scentE = vec2(vLandPosition.x, -vLandPosition.z) / ${SCALE.toFixed(1)};
vec2 toGoal = scentE - uGoal, toDecoy = scentE - uDecoy;
float goalScent = exp(-${SP_SCENT_FALLOFF.toFixed(1)} * dot(toGoal, toGoal));
float decoyScent = exp(-${SP_SCENT_FALLOFF.toFixed(1)} * dot(toDecoy, toDecoy)) * uDecoyOn;
float decoyCore = exp(-${SP_DECOY_FALLOFF.toFixed(1)} * dot(toDecoy, toDecoy)) * uDecoyOn;
float court = step(abs(scentE.x), 1.02) * step(abs(scentE.y), 1.02);
float goalLevel = goalScent * 7.0, decoyLevel = decoyScent * 7.0;
float goalRing = 1.0 - smoothstep(0.02, 0.09, abs(fract(goalLevel + 0.5) - 0.5)) ;
float decoyRing = 1.0 - smoothstep(0.02, 0.09, abs(fract(decoyLevel + 0.5) - 0.5));
vec3 goalTint = vec3(${accentColor.r.toFixed(4)}, ${accentColor.g.toFixed(4)}, ${accentColor.b.toFixed(4)});
vec3 decoyTint = vec3(${decoyColor.r.toFixed(4)}, ${decoyColor.g.toFixed(4)}, ${decoyColor.b.toFixed(4)});
diffuseColor.rgb = mix(diffuseColor.rgb, goalTint * 0.7, goalScent * 0.45 * court);
diffuseColor.rgb = mix(diffuseColor.rgb, decoyTint * 0.5, decoyScent * 0.45 * court);
diffuseColor.rgb *= 1.0 - 0.45 * decoyCore * court;
totalEmissiveRadiance += court * (goalTint * goalScent * (0.12 + goalRing * 0.28 * smoothstep(0.06, 0.3, goalScent)) + decoyTint * decoyScent * (0.1 + decoyRing * 0.24 * smoothstep(0.06, 0.3, decoyScent)));
`);
  };
  world.add(terrain.mesh);
  const hoodoo = world.own(hoodooGeometry());
  const hoodooMaterial = world.own(oakMaterial({ vertexColors: true, flatShading: true, roughness: 0.96, rimColor: palette.sun, rimStrength: 0.26, toonSteps: 3, toonSoftness: 0.3, toonFloor: 0.18, name: 'badlandsHoodoo' }));
  const spots: readonly (readonly [number, number, number, number])[] = [[-8.5, -9, 1.5, 0.4], [3, -12, 2.2, -0.6], [10.5, -7, 1.6, 1.3], [-12, 1, 1.4, 2.2], [12.5, 3, 1.3, -1.4], [-5, 12, 1.7, 3.0]];
  const hoodoos = world.own(new THREE.InstancedMesh(hoodoo, hoodooMaterial, spots.length));
  hoodoos.castShadow = hoodoos.receiveShadow = true;
  {
    const rng = streamFor(ctx.seed, 'hoodoos'), dummy = new THREE.Object3D();
    spots.forEach(([x, z, s, yaw], i) => {
      dummy.position.set(x, heightAt(x, z) - 0.25, z);
      dummy.rotation.set(0, yaw + (rng() - 0.5) * 0.5, 0);
      dummy.scale.set(s, s * (0.85 + rng() * 0.35), s);
      dummy.updateMatrix();
      hoodoos.setMatrixAt(i, dummy.matrix);
    });
    hoodoos.computeBoundingSphere();
  }
  world.add(hoodoos);
  const inCourt = (x: number, z: number): boolean => Math.abs(x) < HALF + 0.3 && Math.abs(z) < HALF + 0.3;
  const rocks = scatterRocks({
    count: 160, area: { x: 0, z: 0, halfX: 15, halfZ: 15 }, seed: ctx.seed ^ 0x2b17, exclude: inCourt, heightAt, color: blendTokens(0xc9a074, palette.hemiGround, 0.25),
    scale: [0.12, 0.5], stretch: [0.45, 0.9], hueJitter: 0.22, rimColor: palette.sun, rimStrength: 0.28, castShadow: true, name: 'badlandsRubble',
  });
  world.add(rocks.group); world.time(world.own(rocks));
  const scrub = scatterGrass({
    count: 900, area: { x: 0, z: 0, halfX: 15, halfZ: 15 }, seed: ctx.seed ^ 0x7f19, exclude: (x, z) => inCourt(x, z) && Math.sin(x * 1.9) * Math.cos(z * 1.4) < 0.7, heightAt,
    color: blendTokens(0xbfa864, palette.groundB, 0.3), scale: [0.14, 0.3], stretch: [0.8, 1.4], hueJitter: 0.25, wind: 0.04, windFrequency: 0.9, rimColor: palette.sun, rimStrength: 0.35, name: 'badlandsScrub',
  });
  world.add(scrub.group); world.time(world.own(scrub));
  const materials = propMaterials(world), parts = modelParts(world);
  const puck = new THREE.Group();
  puck.name = 'scentPuck';
  parts.disc(puck, materials.brass, 0, 0.05, 0, 0.26, 0.1, 24);
  parts.disc(puck, materials.sandstone, 0, 0.005, 0, 0.34, 0.02, 24);
  parts.add(puck, engravedPanel(0.3, 0.3, 0.04, 0.02, 0.17).rotateX(-Math.PI / 2).translate(0, 0.11, 0), world.own(oakMaterial({ color: 0xe8d8b8, vertexColors: true, roughness: 0.5, metalness: 0.2, rimColor: palette.sun, rimStrength: 0.5, toonSteps: 3, name: 'puckArmor' })));
  parts.build();
  world.add(puck);
  const decoyPuck = new THREE.Group();
  decoyPuck.name = 'decoyPuck';
  decoyPuck.visible = competing;
  parts.disc(decoyPuck, materials.shale, 0, 0.05, 0, 0.26, 0.1, 24);
  parts.disc(decoyPuck, materials.glass(DECOY_TINT, 0.9), 0, 0.105, 0, 0.14, 0.03, 16);
  parts.build();
  world.add(decoyPuck);
  const plume = makeAmbientParticles({ preset: 'embers', count: 120, bounds: { halfX: 0.28, halfY: 0.55, halfZ: 0.28 }, seed: ctx.seed ^ 0x3d91, color: blendTokens(accent, palette.sun, 0.3), size: 0.18, intensity: 1.8, opacity: 0.75, rise: 0.32, swirl: 0.1, name: 'goalPlume' });
  world.add(plume.points); world.size(world.time(world.own(plume)));
  const decoyPlume = makeAmbientParticles({ preset: 'embers', count: competing ? 120 : 0, bounds: { halfX: 0.28, halfY: 0.55, halfZ: 0.28 }, seed: ctx.seed ^ 0x51ab, color: blendTokens(DECOY_TINT, palette.skyTop, 0.2), size: 0.18, intensity: 1.6, opacity: 0.7, rise: 0.32, swirl: 0.1, name: 'decoyPlume' });
  world.add(decoyPlume.points); world.size(world.time(world.own(decoyPlume)));
  rig.points.forEach((point, i) => { point.color.set(i === 0 ? accent : DECOY_TINT); point.intensity = i === 0 ? 1.4 : competing ? 1.1 : 0; });
  const extra = {
    accent, palette, heightAt, competing, puck, decoyPuck, plume, decoyPlume, goal, decoy,
    setView(coverage: Parameters<typeof land.setView>[0], camera: THREE.Camera) { land.setView(coverage, camera, terrain); },
    placePucks(gx: number, gy: number, bx: number, by: number) {
      goal.value.set(gx, gy); decoy.value.set(bx, by);
      const wx = toWorldX(gx), wz = toWorldZ(gy), y = heightAt(wx, wz);
      puck.position.set(wx, y, wz); plume.points.position.set(wx, y + 0.5, wz);
      const light = rig.points[0]; if (light) light.position.set(wx, y + 0.6, wz);
      const dx = toWorldX(bx), dz = toWorldZ(by), dy = heightAt(dx, dz);
      decoyPuck.position.set(dx, dy, dz); decoyPlume.points.position.set(dx, dy + 0.5, dz);
      const violet = rig.points[1]; if (violet) violet.position.set(dx, dy + 0.6, dz);
    },
  };
  return world.seal<WorldCore & typeof extra>(extra);
}

export type BadlandsWorld = ReturnType<typeof buildWorld>;
