import * as THREE from 'three';
import type { LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms, setMaterialsTime, type OakMaterial } from '@kit/materials';
import { blendTokens, smoothstep01, TOKEN_WHITE } from '@kit/mix';
import {
  makeAmbientParticles,
  makeMistPlanes,
  type AmbientParticlesHandle,
  type MistPlanesHandle,
} from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { streamFor } from '@kit/rng';
import type { SkyHandle } from '@kit/sky';
import { makeTerrain } from '@kit/terrain';
import { mountainStyle } from '@kit/mountains';
import { extendVegetation, makeLandscape, makeSceneTerrain } from '../half-cheetah/world';
import type { ViewCoverage } from '@kit/view';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import {
  excludeAny,
  excludeBand,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
} from '@kit/vegetation';
import type { WorldCore } from '@kit/world';
import type { EnvGeometry } from '@sim/types';
import { MC_GOAL_POS, MC_MAX_POS, MC_MIN_POS, MC_START_POS } from '@sim/envs/mountaincar';

export const TERRAIN_W = 10.24;
export const BAND_HALF_Z = 0.34;
const BAND_FEATHER = 0.6;
export const SNOW_LINE = 0.7;
export const FOG_EXTENT = TERRAIN_W * 4;
export const SUN_DIR: readonly [number, number, number] = [0.798, 0.47, -0.379];
export const TRAILHEAD_Z = 0.24;
const EAGLE_DIR: readonly [number, number, number] = [0.277, 0.328, -0.904];
const EAGLE_RADIUS = 6.86;
const EAGLE_TRAVEL: readonly [number, number, number] = [0.35, -0.06, -0.935];
const EAGLE_SWEEP = 2;

export function heightResolver(geometry: EnvGeometry): (x: number) => number {
  const height = geometry.classic?.heightAt;
  if (!height) throw new Error('mountain car height profile required');
  return height;
}

export function makeRidgeHeight(
  heightAt: (x: number) => number,
  seed: number,
): (x: number, z: number) => number {
  return (x, z) => {
    const base = heightAt(x);
    const az = Math.abs(z);
    const lateral = smoothstep01((az - BAND_HALF_Z) / BAND_FEATHER);
    const outside = Math.max(0, Math.max(MC_MIN_POS - x, x - MC_MAX_POS));
    const longitudinal = smoothstep01(outside / 0.45);
    const w = lateral > longitudinal ? lateral : longitudinal;
    if (w <= 0) return base;
    const behind = -z;
    const front = behind <= 0;
    const swell = front ? behind * 0.46 : behind * 0.115 + behind * behind * 0.022;
    const amp = front ? 0.05 : 0.08 + 0.08 * Math.min(2.2, az);
    const n1 = valueNoise2(x * 0.62 + 11.3, z * 0.62 + 5.1, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 1.94 + 2.7, z * 1.94 + 9.4, 64, seed ^ 0x5f3a91) - 0.5;
    return base + w * (swell + n1 * 2 * amp + n2 * 0.8 * amp);
  };
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  terrain: ReturnType<typeof makeSceneTerrain>;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
  cairnRoot: THREE.Object3D;
  trailheadRoot: THREE.Object3D;
  summitBurst: THREE.Vector3;
  setGust(env01: number): void;
  setEagle(progress: number | null): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.dusk;
  const accent = ctx.tokens.accents['mountain-car'] ?? ctx.tokens.celebrationGold;
  const heightAt = heightResolver(ctx.geometry);

  const { valley: cValley, meadow: cMeadow, snow: cSnow, rock: cRock, material: mountainMaterial } = mountainStyle(palette);
  const woodLight = blendTokens(palette.groundB, palette.groundA, 0.4);
  const woodDark = blendTokens(palette.groundA, palette.hemiGround, 0.62);
  const stone = blendTokens(palette.hemiGround, palette.skyTop, 0.5);

  const land = makeLandscape(scene, 'alpineValley', palette, 'dusk', { sky: {
    sunDir: SUN_DIR,
    sunSize: 0.05,
    sunHdr: 7,
    sunGlow: 0.42,
    sunGlowPower: 20,
    topSpread: 0.62,
    bottomSpread: 0.3,
    radius: 320,
  }, light: {
    pointCount: 2,
    pointDistance: 2.6,
    sunDistanceScale: 3,
  }, fogExtent: FOG_EXTENT });
  const { world: w, sky, rig } = land;
  rig.setShadowFocus({ x: -0.1, y: 0.5, z: -0.6 }, 5.5);

  const ridge = makeRidgeHeight(heightAt, ctx.seed);
  const terrain = makeSceneTerrain({
    name: 'alpineRidge', seed: ctx.seed, colorA: cValley, colorB: cMeadow, segments: 160,
    height: `
float base = sin(p.x * 3.0) * 0.45 + 0.55;
float az = abs(p.y);
float lateral = smoothstep(0.34, 0.94, az);
float outside = max(0.0, max(-1.2 - p.x, p.x - 0.6));
float w = max(lateral, smoothstep(0.0, 0.45, outside));
float behind = -p.y;
float swell = behind <= 0.0 ? behind * 0.46 : behind * 0.115 + behind * behind * 0.022;
float amp = behind <= 0.0 ? 0.05 : 0.08 + 0.08 * min(2.2, az);
float n1 = worldNoise(p * 0.62 + vec2(11.3, 5.1), 64.0, ${ctx.seed >>> 0}u) - 0.5;
float n2 = worldNoise(p * 1.94 + vec2(2.7, 9.4), 64.0, ${(ctx.seed ^ 0x5f3a91) >>> 0}u) - 0.5;
float local = base + w * (swell + n1 * 2.0 * amp + n2 * 0.8 * amp);
float mountain = 0.3 + pow(abs(sin(p.x * 0.07 + sin(p.y * 0.039)) * cos(p.y * 0.073)), 1.2) * 14.0;
return mix(local, mountain, smoothstep(5.0, 14.0, length(p)));
`,
    detail: `float ragged = worldNoise(p.xz * 1.35 + vec2(3.1, 7.7), 64.0, ${ctx.seed >>> 0}u) - 0.5; float snow = smoothstep(0.65, 0.87, p.y + ragged * 0.16); color = mix(uLandA, uLandB, smoothstep(0.08, 0.54, p.y)); color = mix(color, vec3(${cSnow.r}, ${cSnow.g}, ${cSnow.b}), snow);`,
  }, (x, z) => {
    const t = smoothstep01((Math.hypot(x, z) - 5) / 9);
    const mountain = 0.3 + Math.pow(Math.abs(Math.sin(x * 0.07 + Math.sin(z * 0.039)) * Math.cos(z * 0.073)), 1.2) * 14;
    return ridge(x, z) * (1 - t) + mountain * t;
  }, mountainMaterial);
  w.add(terrain.mesh);
  w.time(w.own(terrain));
  const route = w.own(makeTerrain({
    size: [MC_MAX_POS - MC_MIN_POS, 0.42], segments: [256, 4],
    heightFn: (x) => heightAt(x + (MC_MIN_POS + MC_MAX_POS) * 0.5) + 0.003,
    colorRamp: (h) => new THREE.Color(cValley).lerp(cMeadow, smoothstep01((h - 0.08) / 0.46)).lerp(cSnow, smoothstep01((h - 0.65) / 0.22)),
    materialOptions: mountainMaterial,
    name: 'exactAlpineTrack',
  }));
  route.mesh.position.x = (MC_MIN_POS + MC_MAX_POS) * 0.5;
  w.add(route.mesh);

  const aboveSnow: ExcludeFn = (x, z) => terrain.sampleHeight(x, z) > SNOW_LINE + 0.02;
  const nearCamera: ExcludeFn = (_x, z) => z > 0.62;
  const valleyOpening: ExcludeFn = (x, z) => z > -3.2 && Math.abs(x + 0.45 + Math.sin(z * 1.4) * 0.22) < 0.5;
  const sampleAt = (x: number, z: number): number => terrain.sampleHeight(x, z);

  const trees = scatterTrees({
    archetypes: ['conifer'],
    count: 190,
    area: { x: -0.4, z: -2.2, halfX: 3.6, halfZ: 2 },
    seed: ctx.seed,
    exclude: excludeAny(excludeBand(0, 1.2, 0.15), nearCamera, aboveSnow, valleyOpening),
    heightAt: sampleAt,
    maxSlope: 0.62,
    color: blendTokens(TOKENS.palettes.noon.groundA, palette.skyTop, 0.45),
    scale: [0.16, 0.42],
    stretch: [0.9, 1.5],
    hueJitter: 0.2,
    wind: 0.02,
    windFrequency: 0.9,
    rimColor: palette.skyHorizon,
    rimStrength: 0.32,
    castShadow: true,
    name: 'alpineConifers',
  });
  w.add(trees.group);
  w.time(w.own(trees));

  const rocks = scatterRocks({
    count: 300,
    area: { x: -0.3, z: -0.9, halfX: 3.2, halfZ: 2.2 },
    seed: ctx.seed ^ 0x2b17,
    exclude: excludeAny(excludeBand(0, 0.26, 0.05), nearCamera),
    heightAt: sampleAt,
    color: cRock,
    scale: [0.05, 0.19],
    stretch: [0.6, 1.1],
    hueJitter: 0.22,
    rimColor: palette.skyHorizon,
    rimStrength: 0.28,
    castShadow: true,
    name: 'alpineScree',
  });
  w.add(rocks.group);
  w.time(w.own(rocks));

  const grass = scatterGrass({
    count: 3200,
    area: { x: -0.35, z: -0.7, halfX: 2.8, halfZ: 2.1 },
    seed: ctx.seed ^ 0x7f19,
    exclude: excludeAny(excludeBand(0, 0.17, 0.03), nearCamera, aboveSnow),
    heightAt: sampleAt,
    maxSlope: 0.78,
    color: blendTokens(TOKENS.palettes.noon.groundB, palette.groundA, 0.5),
    scale: [0.045, 0.13],
    stretch: [0.8, 1.5],
    hueJitter: 0.2,
    wind: 0.05,
    windFrequency: 1.5,
    rimColor: palette.skyHorizon,
    rimStrength: 0.38,
    name: 'alpineMeadow',
  });
  w.add(grass.group);
  w.time(w.own(grass));

  const woodland = w.own(extendVegetation(trees, {
    name: 'alpineWoodlands', seed: ctx.seed, tileSize: 4, perTile: 30,
    scale: [0.16, 0.42], stretch: [0.9, 1.5], heightAt: sampleAt, bark: 0x88735a,
    exclude: (x, z, height) => Math.abs(x + 0.4) < 3.6 && Math.abs(z + 2.2) < 2 || height > 0.72 || Math.abs(z) < 1.2,
  }));
  for (const mesh of woodland.meshes) w.add(mesh);

  const cairnRoot = new THREE.Object3D();
  cairnRoot.name = 'summitCairnRoot';
  cairnRoot.position.set(MC_GOAL_POS, heightAt(MC_GOAL_POS), 0);
  w.add(cairnRoot);

  const cairnParts: ColoredPart[] = [];
  let cairnStack = 0;
  {
    const rng = streamFor(ctx.seed, 'cairn');
    const tiers = 7;
    for (let i = 0; i < tiers; i++) {
      const k = i / (tiers - 1);
      const r = 0.078 * (1 - k * 0.62);
      const h = 0.036 * (1 - k * 0.3);
      const g = new THREE.IcosahedronGeometry(r, 0);
      g.scale(1 + (rng() - 0.5) * 0.3, (h / r) * 1.15, 1 + (rng() - 0.5) * 0.3);
      g.rotateY(rng() * Math.PI * 2);
      g.translate((rng() - 0.5) * 0.016, cairnStack + h * 0.5, (rng() - 0.5) * 0.016);
      cairnParts.push({ geometry: g, color: i % 2 === 0 ? stone : blendTokens(stone, cSnow, 0.32) });
      cairnStack += h;
    }
  }
  const POLE_H = 0.34;
  const cairnGeometry = mergeColoredParts(cairnParts, 'summitCairn');
  const propMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.92,
    rimColor: palette.sun,
    rimStrength: 0.6,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.18,
    name: 'alpinePropMaterial',
  });
  const cairnMesh = new THREE.Mesh(cairnGeometry, propMaterial);
  cairnMesh.name = 'summitCairn';
  cairnMesh.castShadow = true;
  cairnMesh.receiveShadow = true;
  cairnRoot.add(cairnMesh);
  w.own(propMaterial);
  w.own(cairnGeometry);

  const cordTopA = heightAt(MC_GOAL_POS) + cairnStack + POLE_H * 0.94;
  const summitBurst = new THREE.Vector3(MC_GOAL_POS, cordTopA - 0.05, 0);

  interface Guy {
    x: number;
    top: number;
  }
  const guys: Guy[] = [
    { x: 0.16, top: heightAt(0.16) + 0.33 },
    { x: 0.85, top: heightAt(0.85) + 0.3 },
  ];
  const buntingParts: ColoredPart[] = [];
  {
    const pole = new THREE.CylinderGeometry(0.005, 0.008, POLE_H, 5);
    pole.translate(MC_GOAL_POS, heightAt(MC_GOAL_POS) + cairnStack + POLE_H * 0.5, 0);
    buntingParts.push({ geometry: pole, color: woodDark });
    const tip = new THREE.ConeGeometry(0.013, 0.032, 6);
    tip.translate(MC_GOAL_POS, heightAt(MC_GOAL_POS) + cairnStack + POLE_H + 0.016, 0);
    buntingParts.push({ geometry: tip, color: woodLight });
  }
  for (const guy of guys) {
    const groundY = heightAt(guy.x);
    const h = guy.top - groundY;
    const pole = new THREE.CylinderGeometry(0.005, 0.008, h, 5);
    pole.translate(guy.x, groundY + h * 0.5, 0);
    buntingParts.push({ geometry: pole, color: woodDark });
    const dx = MC_GOAL_POS - guy.x;
    const dy = cordTopA - guy.top;
    const len = Math.hypot(dx, dy);
    const cord = new THREE.BoxGeometry(len, 0.0035, 0.0035);
    cord.rotateZ(Math.atan2(dy, dx));
    cord.translate((MC_GOAL_POS + guy.x) * 0.5, (cordTopA + guy.top) * 0.5, 0);
    buntingParts.push({ geometry: cord, color: woodDark });
  }
  const buntingGeometry = mergeColoredParts(buntingParts, 'buntingRig');
  const buntingMesh = new THREE.Mesh(buntingGeometry, propMaterial);
  buntingMesh.name = 'buntingRig';
  buntingMesh.castShadow = false;
  w.add(buntingMesh);
  w.own(buntingGeometry);

  const FLAG_W = 0.038;
  const FLAG_H = 0.052;
  const flagGeometry = new THREE.PlaneGeometry(FLAG_W, FLAG_H, 1, 2);
  flagGeometry.translate(0, FLAG_H * 0.5, 0);
  {
    const pos = flagGeometry.getAttribute('position');
    const cols = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const k = 0.76 + 0.4 * (pos.getY(i) / FLAG_H);
      cols[i * 3] = k;
      cols[i * 3 + 1] = k;
      cols[i * 3 + 2] = k;
    }
    flagGeometry.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  }
  const BASE_WIND = 0.013;
  const flagMaterial = oakMaterial({
    color: TOKEN_WHITE,
    vertexColors: true,
    side: THREE.DoubleSide,
    roughness: 0.85,
    rimColor: palette.sun,
    rimStrength: 0.55,
    toonSteps: 2,
    toonSoftness: 0.38,
    toonFloor: 0.34,
    wind: { amplitude: BASE_WIND, frequency: 2.4, height: FLAG_H },
    name: 'prayerFlagMaterial',
  });
  const FLAG_PALETTE: readonly THREE.ColorRepresentation[] = [
    ctx.tokens.uiCyan,
    TOKEN_WHITE,
    accent,
    ctx.tokens.accents['four-rooms'] ?? ctx.tokens.celebrationGold,
    ctx.tokens.celebrationGold,
  ];
  const PER_STRING = 11;
  const FLAG_COUNT = PER_STRING * guys.length;
  const flags = new THREE.InstancedMesh(flagGeometry, flagMaterial, FLAG_COUNT);
  flags.name = 'prayerFlags';
  flags.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(FLAG_COUNT * 3), 3);
  {
    const rng = streamFor(ctx.seed, 'flags');
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    const s = new THREE.Vector3();
    const col = new THREE.Color();
    let n = 0;
    for (const guy of guys) {
      for (let i = 0; i < PER_STRING; i++) {
        const t = (i + 0.8) / (PER_STRING + 0.6);
        const x = MC_GOAL_POS + (guy.x - MC_GOAL_POS) * t;
        const sag = Math.sin(Math.PI * t) * 0.045;
        const y = cordTopA + (guy.top - cordTopA) * t - sag;
        p.set(x, y, 0.004 * (n % 2 === 0 ? 1 : -1));
        e.set(Math.PI + (rng() - 0.5) * 0.34, (rng() - 0.5) * 0.55, (rng() - 0.5) * 0.4);
        q.setFromEuler(e);
        s.setScalar(0.82 + rng() * 0.45);
        m.compose(p, q, s);
        flags.setMatrixAt(n, m);
        col.set(FLAG_PALETTE[n % FLAG_PALETTE.length] ?? TOKEN_WHITE);
        flags.setColorAt(n, col);
        n++;
      }
    }
    flags.instanceMatrix.needsUpdate = true;
    if (flags.instanceColor) flags.instanceColor.needsUpdate = true;
    flags.computeBoundingSphere();
  }
  w.add(flags);
  w.own(flagMaterial);
  w.own(flagGeometry);
  w.own(flags);
  const animated: OakMaterial[] = [flagMaterial];
  w.time({ setTime: (t) => setMaterialsTime(animated, t) });

  const trailheadRoot = new THREE.Object3D();
  trailheadRoot.name = 'trailheadRoot';
  trailheadRoot.position.set(MC_START_POS, heightAt(MC_START_POS), TRAILHEAD_Z);
  w.add(trailheadRoot);
  const signParts: ColoredPart[] = [];
  {
    const post = new THREE.CylinderGeometry(0.006, 0.009, 0.17, 5);
    post.translate(0, 0.085, 0);
    signParts.push({ geometry: post, color: woodDark });
    const plank = new THREE.BoxGeometry(0.084, 0.028, 0.007);
    plank.rotateZ(-0.07);
    plank.translate(0.03, 0.148, 0);
    signParts.push({ geometry: plank, color: woodLight });
    const base = new THREE.IcosahedronGeometry(0.03, 0);
    base.scale(1.2, 0.4, 1.2);
    base.translate(0, 0.006, 0);
    signParts.push({ geometry: base, color: stone });
  }
  const signGeometry = mergeColoredParts(signParts, 'trailheadSign');
  const signMesh = new THREE.Mesh(signGeometry, propMaterial);
  signMesh.name = 'trailheadSign';
  signMesh.castShadow = true;
  trailheadRoot.add(signMesh);
  w.own(signGeometry);

  const mist: MistPlanesHandle = makeMistPlanes({
    size: [4.6, 3.2],
    color: blendTokens(palette.fog, palette.hemiSky, 0.4),
    seed: ctx.seed,
    layers: 5,
    baseY: 0.11,
    spread: 0.34,
    opacity: 0.1,
    speed: 0.006,
    scale: 2.4,
    nearFade: 0.9,
    name: 'valleyMist',
  });
  mist.group.position.set(-0.52, 0, -0.25);
  w.add(mist.group);
  w.time(w.own(mist));

  const pollen: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'pollen',
    count: 300,
    bounds: { x: -0.4, y: 0.5, z: -0.35, halfX: 1.7, halfY: 0.4, halfZ: 1 },
    seed: ctx.seed,
    color: blendTokens(palette.sun, palette.skyHorizon, 0.35),
    size: 0.16,
    intensity: 0.95,
    opacity: 0.5,
    swirl: 0.08,
    name: 'meadowPollen',
  });
  w.add(pollen.points);
  w.size(w.time(w.own(pollen)));

  const glitter: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'snow',
    count: 420,
    bounds: { x: 0.45, y: 1.16, z: -1, halfX: 1.4, halfY: 0.4, halfZ: 1.5 },
    seed: ctx.seed ^ 0x51ab,
    color: blendTokens(TOKEN_WHITE, palette.hemiSky, 0.25),
    size: 0.22,
    intensity: 1.15,
    opacity: 0.55,
    rise: -0.09,
    swirl: 0.05,
    name: 'snowGlitter',
  });
  w.add(glitter.points);
  w.size(w.time(w.own(glitter)));

  const eagle = new THREE.Group();
  eagle.name = 'eagle';
  eagle.visible = false;
  eagle.scale.setScalar(0.78);
  const eagleBank = new THREE.Group();
  eagleBank.name = 'eagleBank';
  eagleBank.rotation.x = -0.24;
  eagle.add(eagleBank);
  const eagleMaterial = oakMaterial({
    color: blendTokens(TOKENS.palettes.night.skyTop, palette.skyTop, 0.25),
    flatShading: true,
    roughness: 1,
    rimColor: palette.sun,
    rimStrength: 0.22,
    toonSteps: 2,
    toonFloor: 0.05,
    side: THREE.DoubleSide,
    fog: false,
    name: 'eagleMaterial',
  });
  const eagleBody = new THREE.ConeGeometry(0.032, 0.34, 4);
  eagleBody.rotateZ(-Math.PI / 2);
  eagleBody.translate(0.04, 0, 0);
  const bodyMesh = new THREE.Mesh(eagleBody, eagleMaterial);
  bodyMesh.name = 'eagleBody';
  const tailGeometry = makeTailGeometry();
  const tailMesh = new THREE.Mesh(tailGeometry, eagleMaterial);
  tailMesh.name = 'eagleTail';
  const wingGeometry = makeWingGeometry();
  const wingL = new THREE.Mesh(wingGeometry, eagleMaterial);
  wingL.name = 'eagleWingL';
  const wingR = new THREE.Mesh(wingGeometry, eagleMaterial);
  wingR.name = 'eagleWingR';
  wingR.scale.z = -1;
  eagleBank.add(bodyMesh, tailMesh, wingL, wingR);
  const eagleDir = new THREE.Vector3(EAGLE_DIR[0], EAGLE_DIR[1], EAGLE_DIR[2]).normalize();
  const eagleTangent = new THREE.Vector3(EAGLE_TRAVEL[0], EAGLE_TRAVEL[1], EAGLE_TRAVEL[2]).normalize();
  eagle.rotation.y = Math.atan2(-eagleTangent.z, eagleTangent.x);
  w.add(eagle);
  w.own(eagleMaterial);
  w.own(wingGeometry);
  w.own(tailGeometry);
  w.own(eagleBody);

  return w.seal({
    sky,
    rig,
    terrain,
    setView(coverage, camera) {
      land.setView(coverage, camera, terrain, woodland);
    },
    cairnRoot,
    trailheadRoot,
    summitBurst,
    setGust(env01) {
      const e = env01 < 0 ? 0 : env01 > 1 ? 1 : env01;
      oakUniforms(flagMaterial)?.setWind(BASE_WIND * (1 + 1.1 * e), 2.4 + 0.35 * e);
      grass.setWind(0.05 * (1 + 1.2 * e), 1.5 + 0.7 * e);
      trees.setWind(0.02 * (1 + 1.0 * e), 0.9 + 0.35 * e);
    },
    setEagle(progress) {
      if (progress === null) {
        eagle.visible = false;
        return;
      }
      const sweep = (progress * 2 - 1) * EAGLE_SWEEP;
      eagle.position.set(
        eagleDir.x * EAGLE_RADIUS + eagleTangent.x * sweep,
        eagleDir.y * EAGLE_RADIUS + eagleTangent.y * sweep + Math.sin(progress * Math.PI) * 0.26,
        eagleDir.z * EAGLE_RADIUS + eagleTangent.z * sweep,
      );
      const flap = Math.sin(progress * 11) * 0.34;
      wingL.rotation.x = flap;
      wingR.rotation.x = -flap;
      eagleBank.rotation.x = -0.24 + 0.18 * Math.sin(progress * Math.PI);
      eagle.visible = true;
    },
  });
}

function makeWingGeometry(): THREE.BufferGeometry {
  const p: readonly (readonly [number, number])[] = [
    [0.12, 0.015],
    [-0.13, 0.015],
    [0.08, 0.2],
    [-0.18, 0.2],
    [-0.02, 0.46],
    [-0.2, 0.4],
  ];
  const tris: readonly (readonly [number, number, number])[] = [
    [0, 1, 3],
    [0, 3, 2],
    [2, 3, 5],
    [2, 5, 4],
  ];
  const v = new Float32Array(tris.length * 9);
  let o = 0;
  for (const t of tris) {
    for (const i of t) {
      const q = p[i] as readonly [number, number];
      v[o++] = q[0];
      v[o++] = 0;
      v[o++] = q[1];
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(v, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'eagleWing';
  return g;
}

function makeTailGeometry(): THREE.BufferGeometry {
  const v = new Float32Array([
    -0.1, 0, 0.02, -0.27, 0, 0.075, -0.29, 0, 0,
    -0.1, 0, -0.02, -0.29, 0, 0, -0.27, 0, -0.075,
  ]);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(v, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'eagleTail';
  return g;
}
