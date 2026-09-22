import * as THREE from 'three';
import { makeHeightField } from '@kit/heightfield';
import { makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { blendTokens, TOKEN_WHITE } from '@kit/mix';
import { makeAmbientParticles, makeMistPlanes } from '@kit/particles';
import { banner, lamp, propMaterials } from '@kit/props';
import type { SceneContext } from '@kit/types';
import { excludeAny, scatterFlowers, scatterGrass, scatterRocks, scatterTrees } from '@kit/vegetation';
import { makeWater } from '@kit/water';
import type { WorldCore } from '@kit/world';
import { PUDDLE_ARMS, PUDDLE_GOAL, PUDDLE_RADIUS, puddleDepth } from '@sim/envs/puddle';

export const COURT = 8;
export const HALF = COURT / 2;
export const DIP = 4;
export const WATER_LEVEL = -0.035;

export const toWorldX = (x: number): number => x * COURT - HALF;
export const toWorldZ = (y: number): number => HALF - y * COURT;
export const fromWorldX = (x: number): number => (x + HALF) / COURT;
export const fromWorldZ = (z: number): number => (HALF - z) / COURT;

function puddleGlsl(): string {
  const arms = PUDDLE_ARMS.map(([ax, ay, bx, by]) => `{ vec2 q = e - vec2(${ax.toFixed(4)}, ${ay.toFixed(4)}); vec2 ab = vec2(${(bx - ax).toFixed(4)}, ${(by - ay).toFixed(4)}); float t = clamp(dot(q, ab) / dot(ab, ab), 0.0, 1.0); dip += max(0.0, ${PUDDLE_RADIUS.toFixed(4)} - length(q - ab * t)); }`);
  return `vec2 e = vec2((p.x + ${HALF.toFixed(4)}) / ${COURT.toFixed(4)}, (${HALF.toFixed(4)} - p.y) / ${COURT.toFixed(4)});\nfloat dip = 0.0;\n${arms.join('\n')}\nh -= dip * ${DIP.toFixed(4)};\nreturn h;`;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const palette = ctx.tokens.palettes.morning, accent = ctx.tokens.accents['puddle'] ?? ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, 'marshFlats', palette, 'morning', {
    sky: { sunDir: [0.62, 0.34, -0.71], sunSize: 0.05, sunHdr: 5.5, sunGlow: 0.5, sunGlowPower: 14, topSpread: 0.6, bottomSpread: 0.35, radius: 320 },
    light: { pointCount: 1, pointDistance: 3, sunDistanceScale: 3 },
    fogExtent: 140, fogDensity: 0.0055,
    weather: { preset: 'snow', count: 900, seed: ctx.seed, color: 0xbcd6ee, size: 0.95, intensity: 1, opacity: 0.24, rise: -1.9, swirl: 0.04, bounds: { y: 2.4, halfX: 7.5, halfY: 2.4, halfZ: 7.5 } },
  });
  const { world, sky, rig } = land;
  sky.setColors(0x8fb8d8, 0xe6e3d2, 0xd9dcc8);
  rig.sun.color.set(0xfff0d0); rig.sun.intensity = 2.1;
  rig.hemi.color.set(0xcfe0ea); rig.hemi.groundColor.set(0x6e7e5a); rig.hemi.intensity = 0.85;
  rig.setShadowFocus({ x: 0, y: 0, z: 0 }, 5.4);
  const field = makeHeightField({ base: 0, court: { halfX: HALF + 0.6, halfZ: HALF + 0.6, rise: 0.28 }, octaves: [{ wavelength: 30, amplitude: 0.7, seed: ctx.seed }, { wavelength: 9, amplitude: 0.2, seed: ctx.seed ^ 0x9d }] });
  const depthAt = (x: number, z: number): number => puddleDepth(fromWorldX(x), fromWorldZ(z));
  const heightAt = (x: number, z: number): number => field.heightAt(x, z) - depthAt(x, z) * DIP;
  const mud = blendTokens(palette.groundA, 0x4e5a3c, 0.45), moss = blendTokens(palette.groundB, 0x8fb06a, 0.4);
  const terrain = world.own(world.time(makeSceneTerrain({
    follow: false, name: 'marshGround', seed: ctx.seed, colorA: mud, colorB: moss, segments: 160, radius: 360, height: field.glsl.replace('return h;', puddleGlsl()),
    detail: `float wet = smoothstep(0.02, -0.25, p.y); color = mix(color, uLandA * 0.55, wet); color *= 0.95 + 0.08 * landNoise(p.xz * 2.3);`,
  }, heightAt, { roughness: 0.9, rimColor: palette.sun, rimStrength: 0.16, toonSteps: 4, toonSoftness: 0.35, toonFloor: 0.1 })));
  world.add(terrain.mesh);
  const water = world.own(world.time(makeWater({
    size: [COURT + 1.4, COURT + 1.4], palette, segments: 24, level: WATER_LEVEL, amplitude: 0.014, waveScale: 4.5, speed: 1.2, fresnel: 0.42, caustics: 0.1, depthFalloff: 30, opacity: 0.72,
    sunDir: sky.sunDirection, shallow: blendTokens(0x5f9bb4, accent, 0.35), deep: blendTokens(0x2f5b6d, palette.fog, 0.2), sky: blendTokens(palette.skyHorizon, 0x9fc4dc, 0.6), caustic: blendTokens(accent, palette.sun, 0.5), name: 'marshWater',
  })));
  water.uniforms.uSpecular.value = 0.55;
  world.add(water.mesh);
  const inWater = (x: number, z: number): boolean => depthAt(x, z) > 0.008;
  const outside = (x: number, z: number): boolean => Math.abs(x) > HALF || Math.abs(z) > HALF;
  const stones = scatterRocks({
    count: 120, area: { x: 0, z: 0, halfX: HALF, halfZ: HALF }, seed: ctx.seed ^ 0x2b17, exclude: excludeAny(inWater, (x, z) => depthAt(x, z) === 0 && Math.sin(x * 2.1) * Math.cos(z * 1.7) < 0.15),
    heightAt, color: blendTokens(0xb7b2a4, palette.skyHorizon, 0.2), scale: [0.24, 0.46], stretch: [0.22, 0.34], hueJitter: 0.12, rimColor: palette.sun, rimStrength: 0.3, castShadow: true, receiveShadow: true, name: 'steppingStones',
  });
  world.add(stones.group); world.time(world.own(stones));
  const reeds = scatterGrass({
    count: 1900, area: { x: 0, z: 0, halfX: HALF + 3, halfZ: HALF + 3 }, seed: ctx.seed ^ 0x7f19, exclude: excludeAny(inWater, (x, z) => !outside(x, z) && depthAt(x, z) === 0 && Math.sin(x * 1.3 + z * 0.9) > 0.55),
    heightAt, color: blendTokens(0x7d9a52, palette.groundB, 0.3), scale: [0.16, 0.36], stretch: [1, 1.9], hueJitter: 0.22, wind: 0.05, windFrequency: 1.1, rimColor: palette.sun, rimStrength: 0.4, name: 'marshReeds',
  });
  world.add(reeds.group); world.time(world.own(reeds));
  const blossoms = scatterFlowers({
    count: 260, area: { x: 0, z: 0, halfX: HALF + 2, halfZ: HALF + 2 }, seed: ctx.seed ^ 0x51ab, exclude: inWater, heightAt,
    color: blendTokens(0xf2c4d8, TOKEN_WHITE, 0.3), scale: [0.5, 0.9], hueJitter: 0.18, rimColor: palette.sun, rimStrength: 0.3, name: 'marshBlossoms',
  });
  world.add(blossoms.group); world.time(world.own(blossoms));
  const willows = scatterTrees({
    archetypes: ['broadleaf', 'bare'], count: 34, area: { x: 0, z: -3, halfX: 13, halfZ: 9 }, seed: ctx.seed ^ 0x33c1, exclude: (x, z) => Math.abs(x) < HALF + 1.2 && Math.abs(z) < HALF + 1.2,
    heightAt, maxSlope: 0.7, color: blendTokens(0x8aa862, palette.groundB, 0.35), scale: [0.55, 1.1], stretch: [0.9, 1.4], hueJitter: 0.2, wind: 0.03, rimColor: palette.sun, rimStrength: 0.3, castShadow: true, name: 'marshWillows',
  });
  world.add(willows.group); world.time(world.own(willows));
  const materials = propMaterials(world);
  const goalX = toWorldX(PUDDLE_GOAL + (1 - PUDDLE_GOAL) / 2), goalZ = toWorldZ(PUDDLE_GOAL + (1 - PUDDLE_GOAL) / 2);
  const beacon = lamp(world, accent, 1.3, materials);
  beacon.group.position.set(goalX + 0.35, heightAt(goalX + 0.35, goalZ - 0.35), goalZ - 0.35);
  beacon.group.scale.setScalar(0.8);
  world.add(beacon.group);
  const flag = banner(world, accent, 1.9, materials);
  flag.position.set(goalX - 0.45, heightAt(goalX - 0.45, goalZ - 0.5), goalZ - 0.5);
  flag.scale.setScalar(0.75);
  world.add(flag);
  const mist = makeMistPlanes({ size: [11, 11], color: blendTokens(palette.fog, TOKEN_WHITE, 0.4), seed: ctx.seed, layers: 3, baseY: 0.08, spread: 0.35, opacity: 0.035, speed: 0.008, scale: 2.2, nearFade: 2.5, name: 'marshMist' });
  world.add(mist.group); world.time(world.own(mist));
  const splash = makeAmbientParticles({ preset: 'bubbles', count: 70, bounds: { halfX: 0.25, halfY: 0.18, halfZ: 0.25 }, seed: ctx.seed ^ 0x3d91, color: blendTokens(palette.skyHorizon, TOKEN_WHITE, 0.5), size: 0.12, intensity: 1.2, opacity: 0.7, rise: 0.4, swirl: 0.05, name: 'wadeSplash' });
  splash.setDensity(0);
  world.add(splash.points); world.size(world.time(world.own(splash)));
  const glowLight = rig.points[0];
  if (glowLight) { glowLight.color.set(accent); glowLight.intensity = 0; }
  const goal = new THREE.Vector3(goalX, heightAt(goalX, goalZ), goalZ);
  const extra = { accent, palette, heightAt, depthAt, beacon, splash, glowLight, goal, setView(coverage: Parameters<typeof land.setView>[0], camera: THREE.Camera) { land.setView(coverage, camera, terrain); } };
  return world.seal<WorldCore & typeof extra>(extra);
}

export type MarshWorld = ReturnType<typeof buildWorld>;
