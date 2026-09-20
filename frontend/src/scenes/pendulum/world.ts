import * as THREE from 'three';
import type { LightRig } from '@kit/lighting';
import {
  emissiveMaterial,
  oakMaterial,
  setEmissiveIntensity,
  type EmissiveMaterial,
  type OakMaterial,
} from '@kit/materials';
import { blendTokens, clampRange, smoothstep01, TOKEN_WHITE } from '@kit/mix';
import {
  makeAmbientParticles,
  makeMistPlanes,
  type AmbientParticlesHandle,
  type MistPlanesHandle,
} from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { streamFor } from '@kit/rng';
import type { SkyHandle } from '@kit/sky';
import { makeLandscape, makeSceneTerrain, extendWater, extendVegetation, WORLD_NOISE } from '@kit/landscape';
import { plantInstances } from '@kit/vegetation';
import type { ViewCoverage } from '@kit/view';
import { radialSprite } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { valueNoise2 } from '@kit/textures';
import { excludeAny, scatterGrass, scatterRocks, type ExcludeFn } from '@kit/vegetation';
import { makeWater, type WaterHandle } from '@kit/water';
import type { Disposable, WorldBuilder, WorldCore } from '@kit/world';
import type { EnvGeometry } from '@sim/types';
import {
  HANGER_PLATE_DROP,
  HANGER_PLATE_H,
  HANGER_PLATE_W,
  PIN_R,
  ROD_LEN,
  STRAP_HALF,
  STRAP_Z,
  SWING_CLEARANCE,
  SWING_REACH,
} from './dims';

export const WATER_LEVEL = -1.62;
export const REFLECT_Z = 0.92;
export const REFLECT_LIFT = 0.014;

export const POND_RX = 3.55;
export const POND_RZ = 2.85;
const BANK_Y = -1.3;
const BOWL_DEPTH = 1.46;
const BANK_RISE = 0.62;

export const TERRAIN_W = 15;
export const WATER_W = 7.6;
export const WATER_D = 6;

export const FOG_EXTENT = TERRAIN_W * 2.3;

export const MOON_DIR: readonly [number, number, number] = [-0.30, 0.175, -0.938];

const NUKI_Y = SWING_REACH + SWING_CLEARANCE + HANGER_PLATE_DROP + HANGER_PLATE_H * 0.5;
const SHIMAKI_Y = NUKI_Y + 0.21;
const KASAGI_Y = NUKI_Y + 0.34;
const PILLAR_X = ROD_LEN + 0.62;
const PILLAR_TOP = NUKI_Y + 0.28;
const PILLAR_FOOT = -2.32;
const KASAGI_HALF = PILLAR_X + 0.63;
const NUKI_HALF = PILLAR_X + 0.33;
const NUKI_HALF_Z = STRAP_Z + 0.018;
const HANGER_PLATE_Z = STRAP_Z * 2 + 0.028;

export function pivotOf(geometry: EnvGeometry): readonly [number, number, number] {
  const pivot = geometry.classic?.anchors.pivot;
  if (!pivot) throw new Error('pendulum pivot anchor required');
  return pivot;
}

export function makePondBed(seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const r = Math.hypot(x / POND_RX, z / POND_RZ);
    const bowl = 1 - smoothstep01(r);
    const bank = smoothstep01((r - 1) / 1.25);
    const n1 = valueNoise2(x * 0.72 + 4.1, z * 0.72 + 2.3, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 2.05 + 9.7, z * 2.05 + 1.1, 64, seed ^ 0x4d2f13) - 0.5;
    const shore = 1 - smoothstep01(Math.abs(r - 1) / 0.5);
    const noiseW = 0.22 + 1.35 * bank + 0.75 * shore;
    const relief = bank * (Math.sin(x * 0.075 + Math.sin(z * 0.04)) * 0.22 + Math.sin(z * 0.056) * Math.cos(x * 0.036) * 0.28);
    const ground = BANK_Y - BOWL_DEPTH * bowl + BANK_RISE * bank + relief + (n1 * 0.03 + n2 * 0.009) * noiseW;
    const reach = smoothstep01((Math.abs(z) - 6) / 84);
    const channel = x - Math.sin(z * 0.058) * 2.5 - Math.sin(z * 0.018) * 3 - Math.sin(z * 0.008) * 8 * reach;
    const width = 1.45 + 0.45 * (0.5 + 0.5 * Math.cos(z * 0.047)) + reach * (5.5 + Math.abs(z) * 0.035);
    const bankWidth = 1.4 + reach * (4.5 + Math.abs(z) * 0.025);
    const inlet = smoothstep01((Math.abs(z) - 1.25) / 3.2) * (1 - smoothstep01((Math.abs(channel) - width) / bankWidth));
    return ground + (WATER_LEVEL - 0.65 - ground) * inlet;
  };
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  terrain: ReturnType<typeof makeSceneTerrain>;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
  pivotRoot: THREE.Object3D;
  reflectionRoot: THREE.Object3D;
  motes: THREE.Points;
  lanternLight: THREE.PointLight | undefined;
  setLight(serenity: number, lanternX: number, time: number): void;
}

const KOI_COUNT = 9;
const PAD_COUNT = 15;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.night;
  const accent = ctx.tokens.accents['pendulum'] ?? ctx.tokens.celebrationGold;
  const pivot = pivotOf(ctx.geometry);

  const vermilion = blendTokens(TOKENS.palettes.dusk.skyBottom, palette.skyTop, 0.3);
  const vermilionDark = blendTokens(TOKENS.palettes.dusk.skyBottom, palette.skyTop, 0.46);
  const timberShade = blendTokens(vermilion, palette.hemiGround, 0.26);
  const stone = blendTokens(palette.groundB, palette.hemiSky, 0.42);
  const iron = blendTokens(palette.hemiGround, palette.skyTop, 0.35);
  const bedDeep = blendTokens(palette.skyTop, palette.hemiGround, 0.5);
  const bedSilt = blendTokens(palette.skyTop, TOKENS.palettes.dusk.groundA, 0.3);
  const bankMoss = blendTokens(palette.groundA, 0x486450, 0.4);
  const bankStone = blendTokens(palette.groundB, 0x687c85, 0.4);

  const land = makeLandscape(scene, 'lanternBell', palette, 'night', { sky: {
    sunDir: MOON_DIR,
    sunSize: 0.038,
    sunHdr: 1.45,
    sunGlow: 0.06,
    sunGlowPower: 52,
    topSpread: 0.7,
    bottomSpread: 0.4,
    stars: {
      density: 225,
      size: 0.052,
      threshold: 0.9625,
      twinkle: 0.42,
      intensity: 2.7,
      color: blendTokens(palette.sun, TOKEN_WHITE, 0.45),
    },
    radius: 300,
  }, light: {
    pointCount: 3,
    pointDistance: 7,
    sunDistanceScale: 2.6,
  }, fogExtent: FOG_EXTENT });
  const { world: w, sky, rig } = land;
  rig.setShadowFocus({ x: 0, y: -0.9, z: -0.2 }, 5.6);
  rig.hemi.color.set(0x9aa8c4);
  rig.hemi.groundColor.set(0x5b6872);
  rig.hemi.intensity = 1.05;

  const bed = makePondBed(ctx.seed);
  const bedHeight = `
float r = length(p / vec2(${POND_RX}, ${POND_RZ}));
float bowl = 1.0 - smoothstep(0.0, 1.0, r);
float bank = smoothstep(1.0, 2.25, r);
float n1 = worldNoise(p * 0.72 + vec2(4.1, 2.3), 64.0, ${ctx.seed >>> 0}u) - 0.5;
float n2 = worldNoise(p * 2.05 + vec2(9.7, 1.1), 64.0, ${(ctx.seed ^ 0x4d2f13) >>> 0}u) - 0.5;
float shore = 1.0 - smoothstep(0.0, 0.5, abs(r - 1.0));
float relief = bank * (sin(p.x * 0.075 + sin(p.y * 0.04)) * 0.22 + sin(p.y * 0.056) * cos(p.x * 0.036) * 0.28);
float ground = ${BANK_Y} - ${BOWL_DEPTH} * bowl + ${BANK_RISE} * bank + relief + (n1 * 0.03 + n2 * 0.009) * (0.22 + 1.35 * bank + 0.75 * shore);
float reach = smoothstep(6.0, 90.0, abs(p.y));
float channel = p.x - sin(p.y * 0.058) * 2.5 - sin(p.y * 0.018) * 3.0 - sin(p.y * 0.008) * 8.0 * reach;
float width = 1.45 + 0.45 * (0.5 + 0.5 * cos(p.y * 0.047)) + reach * (5.5 + abs(p.y) * 0.035);
float bankWidth = 1.4 + reach * (4.5 + abs(p.y) * 0.025);
float inlet = smoothstep(1.25, 4.45, abs(p.y)) * (1.0 - smoothstep(width, width + bankWidth, abs(channel)));
return mix(ground, ${WATER_LEVEL - 0.65}, inlet);
`;
  const terrain = makeSceneTerrain({
    name: 'pondBed', seed: ctx.seed, colorA: bedSilt, colorB: bedDeep, segments: 160, height: bedHeight,
    detail: `float submerged = 1.0 - smoothstep(${WATER_LEVEL - 0.55}, ${WATER_LEVEL}, p.y); float dry = smoothstep(${WATER_LEVEL}, ${WATER_LEVEL + 0.5}, p.y); color = mix(mix(uLandA, uLandB, submerged), vec3(${bankMoss.r}, ${bankMoss.g}, ${bankMoss.b}), dry * 0.85);`,
  }, bed, { rimColor: palette.hemiSky, rimStrength: 0.18, toonSteps: 3, toonSoftness: 0.34, toonFloor: 0.18 });
  w.add(terrain.mesh);
  w.time(w.own(terrain));

  const bedAt = (x: number, z: number): number => terrain.sampleHeight(x, z);

  const pivotRoot = new THREE.Object3D();
  pivotRoot.name = 'toriiPivotRoot';
  pivotRoot.position.set(pivot[0], pivot[1], pivot[2]);
  w.add(pivotRoot);

  const toriiParts: ColoredPart[] = [];
  {
    const box = (
      w: number,
      h: number,
      d: number,
      x: number,
      y: number,
      z: number,
      color: THREE.ColorRepresentation,
      tiltZ = 0,
    ): void => {
      const g = new THREE.BoxGeometry(w, h, d);
      if (tiltZ !== 0) g.rotateZ(tiltZ);
      g.translate(x, y, z);
      toriiParts.push({ geometry: g, color });
    };

    for (const sx of [-1, 1]) {
      const x = pivot[0] + sx * PILLAR_X;
      const h = PILLAR_TOP - PILLAR_FOOT;
      const pillar = new THREE.CylinderGeometry(0.098, 0.132, h, 9);
      pillar.translate(x, PILLAR_FOOT + h * 0.5, pivot[2]);
      toriiParts.push({ geometry: pillar, color: vermilion });
      const collar = new THREE.CylinderGeometry(0.148, 0.152, 0.07, 9);
      collar.translate(x, WATER_LEVEL + 0.03, pivot[2]);
      toriiParts.push({ geometry: collar, color: stone });
      const base = new THREE.CylinderGeometry(0.2, 0.28, 0.22, 9);
      base.translate(x, bedAt(x, pivot[2]) + 0.09, pivot[2]);
      toriiParts.push({ geometry: base, color: stone });
    }

    box(NUKI_HALF * 2, 0.1, NUKI_HALF_Z * 2, pivot[0], pivot[1] + NUKI_Y, pivot[2], timberShade);
    box((KASAGI_HALF - 0.14) * 2, 0.072, NUKI_HALF_Z * 1.72, pivot[0], pivot[1] + SHIMAKI_Y, pivot[2], vermilion);
    box(KASAGI_HALF * 1.16, 0.125, NUKI_HALF_Z * 2.18, pivot[0], pivot[1] + KASAGI_Y, pivot[2], vermilionDark);
    for (const sx of [-1, 1]) {
      box(
        KASAGI_HALF * 0.78,
        0.115,
        NUKI_HALF_Z * 2.12,
        pivot[0] + sx * (KASAGI_HALF * 0.63),
        pivot[1] + KASAGI_Y + 0.031,
        pivot[2],
        vermilionDark,
        sx * -0.062,
      );
    }
    box(0.1, SHIMAKI_Y - NUKI_Y - 0.09, 0.13, pivot[0], pivot[1] + (SHIMAKI_Y + NUKI_Y) * 0.5, pivot[2], vermilion);

    box(
      HANGER_PLATE_W,
      HANGER_PLATE_H,
      HANGER_PLATE_Z,
      pivot[0],
      pivot[1] + NUKI_Y - HANGER_PLATE_DROP,
      pivot[2],
      iron,
    );
    for (const sz of [-1, 1]) {
      const top = pivot[1] + NUKI_Y - HANGER_PLATE_DROP - HANGER_PLATE_H * 0.6;
      const h = top - pivot[1];
      box(STRAP_HALF * 2, h, STRAP_HALF * 2, pivot[0], pivot[1] + h * 0.5, pivot[2] + sz * STRAP_Z, iron);
      const cheek = new THREE.CylinderGeometry(0.042, 0.042, 0.02, 8);
      cheek.rotateX(Math.PI / 2);
      cheek.translate(pivot[0], pivot[1], pivot[2] + sz * STRAP_Z);
      toriiParts.push({ geometry: cheek, color: iron });
    }
    const pin = new THREE.CylinderGeometry(PIN_R, PIN_R, STRAP_Z * 2 + 0.04, 8);
    pin.rotateX(Math.PI / 2);
    pin.translate(pivot[0], pivot[1], pivot[2]);
    toriiParts.push({ geometry: pin, color: stone });
  }
  const toriiGeometry = mergeColoredParts(toriiParts, 'torii');
  const propMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.88,
    rimColor: blendTokens(palette.sun, palette.hemiSky, 0.45),
    rimStrength: 0.36,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.18,
    name: 'lanternBellPropMaterial',
  });
  const toriiMesh = new THREE.Mesh(toriiGeometry, propMaterial);
  toriiMesh.name = 'torii';
  toriiMesh.castShadow = true;
  toriiMesh.receiveShadow = true;
  w.add(toriiMesh);
  w.own(propMaterial);
  w.own(toriiGeometry);

  const shallowCool = blendTokens(palette.skyBottom, palette.skyTop, 0.55);
  const shallowWarm = blendTokens(shallowCool, accent, 0.013);
  const deepCool = blendTokens(palette.skyTop, palette.fog, 0.35);
  const deepWarm = blendTokens(deepCool, accent, 0.006);
  const pond = makeWaterSurface(palette, sky.sunDirection, accent);
  pond.handle.material.fragmentShader = `${WORLD_NOISE}\nfloat moonlitPondBed(vec2 p) {${bedHeight}}\n${pond.handle.material.fragmentShader}`;
  w.add(pond.handle.mesh);
  w.own(pond.handle);
  const liveShallow = new THREE.Color();
  const liveDeep = new THREE.Color();

  const koi = buildKoi(KOI_COUNT, ctx.seed, palette, accent);
  w.add(koi.mesh);
  w.own(koi);

  const pads = buildLotusPads(w, PAD_COUNT, ctx.seed, palette, pond.handle.sampleHeight, bedAt);

  const reflectSprite = radialSprite(64, 1.7);
  const reflectMaterial: EmissiveMaterial = emissiveMaterial(blendTokens(accent, palette.sun, 0.18), 1.5, {
    map: reflectSprite,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    name: 'lanternReflection',
  });
  reflectMaterial.opacity = 0;
  const reflectGeometry = new THREE.PlaneGeometry(0.56, 3.7);
  reflectGeometry.rotateX(-Math.PI / 2);
  const reflectionRoot = new THREE.Object3D();
  reflectionRoot.name = 'lanternReflectionRoot';
  reflectionRoot.position.set(pivot[0], WATER_LEVEL + REFLECT_LIFT, REFLECT_Z);
  const reflectMesh = new THREE.Mesh(reflectGeometry, reflectMaterial);
  reflectMesh.name = 'lanternReflection';
  reflectMesh.renderOrder = 3;
  reflectionRoot.add(reflectMesh);
  w.add(reflectionRoot);
  w.own(reflectSprite);
  w.own(reflectMaterial);
  w.own(reflectGeometry);

  const submerged: ExcludeFn = (x, z) => bedAt(x, z) < WATER_LEVEL - 0.12;
  const nearCamera: ExcludeFn = (x, z) => z > 1.45 && Math.abs(x) < 2.6;
  const underArch: ExcludeFn = (x, z) => Math.abs(x) < PILLAR_X + 0.45 && Math.abs(z) < 0.5;

  const reeds = scatterGrass({
    count: 1700,
    area: { x: 0, z: -0.3, halfX: 5.4, halfZ: 4.1 },
    seed: ctx.seed ^ 0x51ab,
    exclude: excludeAny(submerged, nearCamera, underArch, (x, z) => bedAt(x, z) > WATER_LEVEL + 0.55),
    heightAt: bedAt,
    maxSlope: 0.95,
    color: blendTokens(bankMoss, palette.hemiSky, 0.15),
    scale: [0.22, 0.5],
    stretch: [0.85, 1.3],
    hueJitter: 0.26,
    wind: 0.05,
    windFrequency: 0.85,
    rimColor: palette.hemiSky,
    rimStrength: 0.75,
    name: 'pondReeds',
  });
  w.add(reeds.group);
  w.time(w.own(reeds));

  const rocks = scatterRocks({
    count: 90,
    area: { x: 0, z: -0.4, halfX: 5.2, halfZ: 3.8 },
    seed: ctx.seed ^ 0x2b17,
    exclude: excludeAny(nearCamera, underArch, (x, z) => {
      const h = bedAt(x, z);
      return h < WATER_LEVEL - 0.5 || h > WATER_LEVEL + 0.85;
    }),
    heightAt: bedAt,
    color: bankStone,
    scale: [0.14, 0.42],
    stretch: [0.45, 0.9],
    hueJitter: 0.24,
    rimColor: palette.hemiSky,
    rimStrength: 0.4,
    castShadow: true,
    name: 'pondStones',
  });
  w.add(rocks.group);
  w.time(w.own(rocks));

  const meadow = w.own(extendVegetation(reeds, {
    name: 'moonlitReedBanks', seed: ctx.seed ^ 0x51ab, tileSize: 24, perTile: 1024,
    scale: [0.18, 0.44], stretch: [0.8, 1.2], heightAt: bedAt,
    exclude: (x, z, height) => Math.abs(x) < 5.4 && Math.abs(z + 0.3) < 4.1 || height < WATER_LEVEL - 0.12 || height > WATER_LEVEL + 0.75 || Math.sin(x * 0.11 + z * 0.21) + Math.cos(z * 0.09 - x * 0.06) < -0.35,
  }));
  const stones = w.own(extendVegetation(rocks, {
    name: 'moonlitGardenStones', seed: ctx.seed ^ 0x2b17, tileSize: 24, perTile: 48,
    scale: [0.14, 0.42], stretch: [0.45, 0.9], heightAt: bedAt,
    exclude: (x, z, height) => Math.abs(x) < 5.2 && Math.abs(z + 0.4) < 3.8 || height < WATER_LEVEL - 0.5 || height > WATER_LEVEL + 0.9,
  }));
  for (const mesh of [...meadow.meshes, ...stones.meshes]) w.add(mesh);

  const fireflyColor = blendTokens(accent, TOKENS.accents['t-maze'] ?? accent, 0.42);
  const fireflies: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'fireflies',
    count: 260,
    bounds: { x: 0, y: WATER_LEVEL + 0.92, z: -0.6, halfX: 4.9, halfY: 0.5, halfZ: 3.2 },
    seed: ctx.seed ^ 0x7f19,
    color: fireflyColor,
    size: 0.4,
    intensity: 2.3,
    opacity: 0.85,
    swirl: 0.3,
    name: 'reedFireflies',
  });
  w.add(fireflies.points);
  w.size(w.time(w.own(fireflies)));

  const mist: MistPlanesHandle = makeMistPlanes({
    size: [8.4, 6.4],
    color: blendTokens(palette.fog, palette.hemiSky, 0.45),
    seed: ctx.seed,
    layers: 4,
    baseY: WATER_LEVEL + 0.06,
    spread: 0.5,
    opacity: 0.13,
    speed: 0.008,
    scale: 2.2,
    nearFade: 2.2,
    name: 'pondMist',
  });
  mist.group.position.set(0, 0, -0.5);
  w.add(mist.group);
  w.time(w.own(mist));

  const embers: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'embers',
    count: 90,
    bounds: { halfX: 0.24, halfY: 0.42, halfZ: 0.2 },
    seed: ctx.seed ^ 0x3d91,
    color: blendTokens(accent, palette.sun, 0.25),
    size: 0.24,
    intensity: 1.9,
    opacity: 0.7,
    rise: 0.22,
    swirl: 0.06,
    name: 'lanternMotes',
  });
  embers.setDensity(0);
  w.add(embers.points);
  w.size(w.own(embers));

  const lanternLight = rig.points[0];
  if (lanternLight) {
    lanternLight.color.set(blendTokens(accent, palette.sun, 0.14));
    lanternLight.distance = 3.4;
  }
  const pondLight = rig.points[1];
  if (pondLight) {
    pondLight.color.set(blendTokens(accent, palette.hemiSky, 0.3));
    pondLight.position.set(pivot[0], WATER_LEVEL + 0.3, 0.35);
    pondLight.distance = 3.5;
  }
  const archLight = rig.points[2];
  if (archLight) {
    archLight.color.set(blendTokens(accent, TOKENS.palettes.dusk.skyBottom, 0.45));
    archLight.position.set(pivot[0], pivot[1] + 0.35, pivot[2] - 1.35);
    archLight.distance = 4;
  }

  return w.seal({
    sky,
    rig,
    terrain,
    pivotRoot,
    reflectionRoot,
    motes: embers.points,
    lanternLight,
    setView(coverage, camera) {
      land.setView(coverage, camera, terrain, meadow, stones);
      extendWater(pond.handle.mesh, coverage, WATER_W, WATER_D, `${WATER_LEVEL} - moonlitPondBed(vWorld.xz)`);
    },

    setLight(serenity, lanternX, time) {
      const s = clampRange(serenity, 0, 1);

      liveShallow.copy(shallowCool).lerp(shallowWarm, s);
      liveDeep.copy(deepCool).lerp(deepWarm, s);
      pond.handle.setColors(liveShallow, liveDeep);
      pond.setGlow(s);

      reflectionRoot.position.x = lanternX;
      reflectMaterial.opacity = 0.05 + 0.3 * s * s;
      reflectMesh.scale.set(0.6 + 0.6 * s, 1, 0.72 + 0.45 * s);
      setEmissiveIntensity(reflectMaterial, 0.7 + 0.85 * s);

      koi.setState(time, s, lanternX);

      embers.setDensity(s * s * 0.55);

      if (pondLight) pondLight.intensity = 0.05 + 1.15 * s;
      if (archLight) archLight.intensity = 0.02 + 0.5 * s;
      if (lanternLight) lanternLight.intensity = 0.08 + 1.7 * s;

      reeds.setWind(0.045 * (1 + 0.5 * (1 - s)), 0.85);
    },

  }, {
    time: (t) => {
      pond.handle.setTime(t);
      pads.setTime(t);
      embers.setTime(t);
    },
  });
}

const BASE_CAUSTIC = 0.07;

interface PondHandle {
  handle: WaterHandle;
  setGlow(s: number): void;
}

function makeWaterSurface(
  palette: Palette,
  sunDir: THREE.Vector3,
  accent: number,
): PondHandle {
  const handle = makeWater({
    size: [WATER_W, WATER_D],
    palette,
    segments: 2,
    level: WATER_LEVEL,
    amplitude: 0.045,
    waveScale: 0.8,
    speed: 0.5,
    fresnel: 0.74,
    caustics: BASE_CAUSTIC,
    depthFalloff: 0.65,
    opacity: 0.74,
    sunDir,
    caustic: blendTokens(accent, palette.sun, 0.45),
    sky: blendTokens(palette.skyHorizon, palette.skyTop, 0.25),
    name: 'koiPond',
  });
  return {
    handle,
    setGlow(s) {
      handle.uniforms.uCausticStrength.value = BASE_CAUSTIC * (0.08 + 0.92 * s);
      handle.uniforms.uSpecular.value = 0.45 + 0.85 * s;
    },
  };
}

interface KoiHandle extends Disposable {
  mesh: THREE.InstancedMesh;
  setState(time: number, serenity: number, lanternX: number): void;
}

function koiGeometry(): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const body = new THREE.IcosahedronGeometry(0.1, 1);
  body.scale(2.1, 0.62, 0.9);
  parts.push({ geometry: body, color: 0xffffff });
  const head = new THREE.ConeGeometry(0.062, 0.14, 6);
  head.rotateZ(-Math.PI / 2);
  head.translate(0.24, 0, 0);
  parts.push({ geometry: head, color: 0xffffff });
  const tail = new THREE.ConeGeometry(0.085, 0.17, 3);
  tail.rotateZ(Math.PI / 2);
  tail.scale(1, 1, 0.35);
  tail.translate(-0.27, 0.01, 0);
  parts.push({ geometry: tail, color: 0xdadada });
  const fin = new THREE.ConeGeometry(0.05, 0.1, 3);
  fin.rotateX(Math.PI / 2);
  fin.scale(1, 0.3, 1);
  fin.translate(0.02, 0.02, 0);
  parts.push({ geometry: fin, color: 0xdadada });
  return mergeColoredParts(parts, 'koi');
}

function buildKoi(
  count: number,
  seed: number,
  palette: Palette,
  accent: number,
): KoiHandle {
  const geometry = koiGeometry();
  const material = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.55,
    rimColor: palette.hemiSky,
    rimStrength: 0.85,
    toonSteps: 2,
    toonSoftness: 0.45,
    toonFloor: 0.55,
    name: 'koiMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
  mesh.name = 'koiSchool';
  mesh.frustumCulled = false;
  mesh.count = count;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3);

  const white = blendTokens(TOKEN_WHITE, palette.hemiSky, 0.25);
  const gold = blendTokens(accent, TOKEN_WHITE, 0.25);
  const ember = blendTokens(accent, TOKENS.palettes.dusk.skyBottom, 0.4);
  const palettes = [white, gold, ember];

  interface Lane {
    radius: number;
    phase: number;
    speed: number;
    bobPhase: number;
    scale: number;
    squash: number;
  }
  const lanes: Lane[] = [];
  {
    const rng = streamFor(seed, 'koi');
    const c = new THREE.Color();
    for (let i = 0; i < count; i++) {
      lanes.push({
        radius: 0.55 + rng() * 1.55,
        phase: rng() * Math.PI * 2,
        speed: (0.11 + rng() * 0.16) * (rng() < 0.4 ? -1 : 1),
        bobPhase: rng() * Math.PI * 2,
        scale: 0.58 + rng() * 0.46,
        squash: 0.85 + rng() * 0.3,
      });
      c.copy(palettes[i % palettes.length] ?? white);
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  const m = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const s = new THREE.Vector3();

  return {
    mesh,
    setState(time, serenity, lanternX) {
      for (let i = 0; i < count; i++) {
        const lane = lanes[i] as Lane;
        const a = lane.phase + time * lane.speed;
        const gather = 1 - 0.42 * serenity;
        const r = lane.radius * gather;
        const cx = lanternX * (0.15 + 0.45 * serenity);
        p.set(cx + Math.cos(a) * r * 1.5, 0, -0.15 + Math.sin(a) * r * 1.05);
        const depth = 0.62 - 0.44 * serenity + 0.045 * Math.sin(time * 0.8 + lane.bobPhase);
        p.y = WATER_LEVEL - depth;
        const heading = lane.speed >= 0 ? a + Math.PI / 2 : a - Math.PI / 2;
        e.set(0, -heading, 0.14 * Math.sin(time * 1.7 + lane.bobPhase));
        q.setFromEuler(e);
        s.set(lane.scale, lane.scale * lane.squash, lane.scale);
        m.compose(p, q, s);
        mesh.setMatrixAt(i, m);
      }
      mesh.instanceMatrix.needsUpdate = true;
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}

interface PadHandle {
  mesh: THREE.InstancedMesh;
  setTime(t: number): void;
}

function buildLotusPads(
  world: WorldBuilder,
  count: number,
  seed: number,
  palette: Palette,
  waterHeight: (x: number, z: number, t: number) => number,
  bedAt: (x: number, z: number) => number,
): PadHandle {
  const geometry = new THREE.CircleGeometry(0.19, 11, 0.42, Math.PI * 2 - 0.84);
  geometry.rotateX(-Math.PI / 2);
  geometry.name = 'lotusPad';
  const material = oakMaterial({
    color: blendTokens(palette.groundA, TOKENS.accents['four-rooms'] ?? palette.groundB, 0.3),
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.72,
    rimColor: palette.hemiSky,
    rimStrength: 0.55,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.28,
    name: 'lotusPadMaterial',
  });
  const plants = plantInstances(world, 'lotusPads', geometry, material, Math.max(1, count));
  const { mesh } = plants;
  mesh.frustumCulled = false;

  interface Pad {
    x: number;
    z: number;
    yaw: number;
    scale: number;
  }
  const pads: Pad[] = [];
  {
    const rng = streamFor(seed, 'lotus');
    for (let i = 0; i < count * 14 && pads.length < count; i++) {
      const x = (rng() * 2 - 1) * 3.1;
      const z = -0.4 + (rng() * 2 - 1) * 2.3;
      if (bedAt(x, z) > WATER_LEVEL - 0.35) continue;
      if (Math.abs(x) < 0.85 && Math.abs(z) < 1.4) continue;
      pads.push({ x, z, yaw: rng() * Math.PI * 2, scale: 0.7 + rng() * 0.85 });
    }
    mesh.count = pads.length;
  }

  const { position: p, rotation: e, scale: s } = plants.transform;

  return {
    mesh,
    setTime(t) {
      for (let i = 0; i < pads.length; i++) {
        const pad = pads[i] as Pad;
        p.set(pad.x, waterHeight(pad.x, pad.z, t) + 0.006, pad.z);
        const dx = waterHeight(pad.x + 0.12, pad.z, t) - waterHeight(pad.x - 0.12, pad.z, t);
        const dz = waterHeight(pad.x, pad.z + 0.12, t) - waterHeight(pad.x, pad.z - 0.12, t);
        e.set(dz * 2.6, pad.yaw, -dx * 2.6);
        s.setScalar(pad.scale);
        plants.write(i);
      }
      plants.commit(pads.length);
    },
  };
}
