
import * as THREE from 'three';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, setMaterialsTime, type OakMaterial } from '@kit/materials';
import {
  makeAmbientParticles,
  makeMistPlanes,
  type AmbientParticlesHandle,
  type MistPlanesHandle,
} from '@kit/particles';
import { streamFor } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import {
  excludeAny,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
} from '@kit/vegetation';
import { makeWater, type WaterHandle } from '@kit/water';
import { CP_POLE_LENGTH, CP_TH_LIMIT, CP_X_LIMIT } from '@sim/envs/cartpole';
import type { EnvGeometry } from '@sim/types';


export const DECK_Y = -0.36;
export const DECK_T = 0.052;
export const DECK_HALF_Z = 0.46;
export const WATER_Y = -0.86;
export const LAKE_DEPTH = 0.72;
export const BANK_HEIGHT = 0.38;
export const FAR_RISE = 2.4;
export const FAR_ONSET = 9;
export const FAR_SPAN = 12;
export const BANK_INNER_Z = 5.4;
export const BANK_INNER_X = 6.5;
export const BANK_FEATHER = 2.6;

export const CORD_Z = -0.03;
export const STAKE_Z = -0.34;
export const GAUGE_Z = -0.92;


export const CAM_BASE: readonly [number, number, number] = [0.44, 1.16, 4.62];
export const CAM_LOOK: readonly [number, number, number] = [0, 0.32, -0.06];

export function gaugeStation(
  camera: readonly [number, number, number],
  z: number,
): { x: number; y: number; z: number; scale: number } {
  const cz = camera[2];
  if (!(Math.abs(cz - z) > 1e-6) || !(Math.abs(cz) > 1e-6)) return { x: 0, y: 0, z, scale: 1 };
  const t = (cz - z) / cz;
  return { x: camera[0] * (1 - t), y: camera[1] * (1 - t), z, scale: t };
}

export const TERRAIN_W = 34;
export const TERRAIN_D = 44;
export const TERRAIN_SEG_X = 180;
export const TERRAIN_SEG_Z = 190;

export const FOG_EXTENT = TERRAIN_W * 6.6;

export const SUN_DIR: readonly [number, number, number] = [0.6, 0.7, 0.39];


export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

export function clampRange(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function blendTokens(
  a: THREE.ColorRepresentation,
  b: THREE.ColorRepresentation,
  k: number,
): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), k);
}

export const TOKEN_WHITE = TOKENS.palettes.noon.sun;
export const TOKEN_BLACK = TOKENS.palettes.night.skyTop;

export function unmixTokens(target: THREE.Color, pull: THREE.ColorRepresentation, k: number): THREE.Color {
  const p = new THREE.Color(pull);
  const inv = 1 / (1 - k);
  return new THREE.Color().setRGB(
    (target.r - k * p.r) * inv,
    (target.g - k * p.g) * inv,
    (target.b - k * p.b) * inv,
  );
}

export interface ColoredPart {
  geometry: THREE.BufferGeometry;
  color: THREE.ColorRepresentation;
}

export function mergeColoredParts(parts: readonly ColoredPart[], name: string): THREE.BufferGeometry {
  let total = 0;
  for (const p of parts) {
    total += p.geometry.index ? p.geometry.index.count : p.geometry.getAttribute('position').count;
  }
  const positions = new Float32Array(total * 3);
  const normals = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);
  const c = new THREE.Color();
  let o = 0;
  for (const part of parts) {
    const geo = part.geometry;
    const pos = geo.getAttribute('position');
    const nor = geo.getAttribute('normal');
    const index = geo.index;
    const n = index ? index.count : pos.count;
    c.set(part.color);
    for (let i = 0; i < n; i++) {
      const vi = index ? index.getX(i) : i;
      positions[o * 3] = pos.getX(vi);
      positions[o * 3 + 1] = pos.getY(vi);
      positions[o * 3 + 2] = pos.getZ(vi);
      if (nor) {
        normals[o * 3] = nor.getX(vi);
        normals[o * 3 + 1] = nor.getY(vi);
        normals[o * 3 + 2] = nor.getZ(vi);
      }
      colors[o * 3] = c.r;
      colors[o * 3 + 1] = c.g;
      colors[o * 3 + 2] = c.b;
      o++;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeBoundingSphere();
  g.name = name;
  for (const p of parts) p.geometry.dispose();
  return g;
}


function anchor(geometry: EnvGeometry, name: string): readonly [number, number, number] | null {
  const a = geometry.classic?.anchors[name];
  if (!a) return null;
  return Number.isFinite(a[0]) && Number.isFinite(a[1]) && Number.isFinite(a[2]) ? a : null;
}

export function trackLimit(geometry: EnvGeometry): number {
  const a = anchor(geometry, 'trackRight');
  return a && a[0] > 0 ? a[0] : CP_X_LIMIT;
}

export function poleLength(geometry: EnvGeometry): number {
  const a = anchor(geometry, 'poleTipRest');
  return a && a[1] > 0 ? a[1] : CP_POLE_LENGTH;
}

export function tiltLineY(geometry: EnvGeometry): number {
  const a = anchor(geometry, 'failArcRight');
  return a && a[1] > 0 ? a[1] : CP_POLE_LENGTH * Math.cos(CP_TH_LIMIT);
}

export function tiltLimit(geometry: EnvGeometry): number {
  const a = anchor(geometry, 'failArcRight');
  return a && a[1] > 0 ? Math.atan2(a[0], a[1]) : CP_TH_LIMIT;
}


export function makeShoreHeight(seed: number): (x: number, z: number) => number {
  const bed = WATER_Y - LAKE_DEPTH;
  return (x, z) => {
    const outZ = Math.abs(z) - BANK_INNER_Z;
    const outX = Math.abs(x) - BANK_INNER_X;
    const bank = Math.max(
      smoothstep01(outZ / BANK_FEATHER),
      smoothstep01(outX / BANK_FEATHER),
    );
    if (bank <= 0) return bed;
    const far = smoothstep01((-z - FAR_ONSET) / FAR_SPAN);
    const n1 = valueNoise2(x * 0.34 + 5.7, z * 0.34 + 2.3, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 1.05 + 13.1, z * 1.05 + 8.9, 64, seed ^ 0x5f3a91) - 0.5;
    const rise = LAKE_DEPTH + BANK_HEIGHT + far * FAR_RISE;
    return bed + bank * (rise + n1 * (0.34 + 0.7 * far) + n2 * (0.14 + 0.28 * far));
  };
}


export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  water: WaterHandle;
  terrain: TerrainHandle;
  endLeftRoot: THREE.Object3D;
  endRightRoot: THREE.Object3D;
  gaugeRoot: THREE.Object3D;
  setTime(t: number): void;
  setWind(lean: number): void;
  setCool(env01: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

interface Disposable {
  dispose(): void;
}

interface Timed {
  setTime(t: number): void;
}

const GRASS_WIND = 0.075;
const REED_WIND = 0.05;
const TREE_WIND = 0.028;

export function buildWorld(ctx: SceneContext, scene: THREE.Scene): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.noon;
  const accent = ctx.tokens.accents['cart-pole'] ?? ctx.tokens.celebrationGold;
  const xLimit = trackLimit(ctx.geometry);
  const cordY = tiltLineY(ctx.geometry);
  const coneHalf = tiltLimit(ctx.geometry);
  const armLength = poleLength(ctx.geometry);
  const group = new THREE.Group();
  group.name = 'boardwalkWorld';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];

  const woodPale = blendTokens(palette.groundB, TOKEN_BLACK, 0.34);
  const woodMid = blendTokens(palette.groundA, TOKEN_BLACK, 0.24);
  const woodDark = blendTokens(palette.groundA, TOKEN_BLACK, 0.52);
  const woodWet = blendTokens(palette.hemiGround, TOKEN_BLACK, 0.42);
  const cordColor = blendTokens(TOKEN_WHITE, palette.hemiSky, 0.34);
  const carve = blendTokens(TOKEN_BLACK, palette.groundA, 0.14);
  const bankGreen = blendTokens(palette.groundA, TOKEN_BLACK, 0.2);
  const meadowGreen = blendTokens(palette.groundB, palette.groundA, 0.6);
  const reedGreen = blendTokens(blendTokens(palette.groundA, palette.groundB, 0.22), TOKEN_BLACK, 0.16);
  const stone = blendTokens(palette.hemiGround, palette.skyTop, 0.34);
  const driftwood = blendTokens(TOKEN_WHITE, palette.groundA, 0.56);
  const shallowMud = blendTokens(palette.hemiGround, palette.groundA, 0.4);
  const lakeShallow = blendTokens(blendTokens(palette.groundA, palette.skyTop, 0.35), TOKEN_BLACK, 0.42);
  const lakeDeep = blendTokens(TOKEN_BLACK, palette.skyTop, 0.14);

  const sky = skyFromPalette(palette, 'noon', {
    sunDir: SUN_DIR,
    sunSize: 0.038,
    sunHdr: 6.5,
    sunGlow: 0.3,
    sunGlowPower: 28,
    topSpread: 0.72,
    bottomSpread: 0.34,
    radius: 320,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 2,
    pointDistance: 5,
    sunDistanceScale: 2.6,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: -0.3, z: -1.4 }, 5.5);

  applyFog(scene, palette, FOG_EXTENT);

  const shore = makeShoreHeight(ctx.seed);
  const rampColor = new THREE.Color();
  const terrain = makeTerrain({
    size: [TERRAIN_W, TERRAIN_D],
    segments: [TERRAIN_SEG_X, TERRAIN_SEG_Z],
    seed: ctx.seed,
    heightFn: shore,
    jitter: 0,
    colorNoise: 0.05,
    slopeColor: stone,
    slopeStrength: 0.5,
    slopeSharpness: 2.4,
    colorRamp: (h) => {
      const depth = smoothstep01((WATER_Y - h) / 0.66);
      const grass = smoothstep01((h - (WATER_Y + 0.02)) / 0.34);
      const crest = smoothstep01((h - (WATER_Y + 0.9)) / 1.5);
      rampColor.copy(new THREE.Color(shallowMud));
      rampColor.lerp(new THREE.Color(bankGreen), grass);
      rampColor.lerp(new THREE.Color(meadowGreen), crest * 0.55);
      rampColor.lerp(new THREE.Color(TOKEN_BLACK), depth * 0.6);
      return rampColor;
    },
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.18,
      toonSteps: 3,
      toonSoftness: 0.3,
      toonFloor: 0.24,
    },
    receiveShadow: true,
    castShadow: false,
    name: 'lakeBed',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  const water = makeWater({
    size: [TERRAIN_W - 2, TERRAIN_D - 2],
    palette,
    segments: 108,
    level: WATER_Y,
    amplitude: 0.016,
    waveScale: 1.15,
    speed: 0.85,
    fresnel: 0.19,
    caustics: 0.26,
    depthFalloff: 5,
    opacity: 0.95,
    shallow: unmixTokens(lakeShallow, palette.groundB, 0.35),
    deep: unmixTokens(lakeDeep, palette.skyBottom, 0.55),
    sunDir: sky.sunDirection,
    name: 'boardwalkLake',
  });
  group.add(water.mesh);
  owned.push(water);
  timed.push(water);

  const deckParts: ColoredPart[] = [];
  const PLANK_PITCH = 0.126;
  const PLANK_GAP = 0.019;
  const plankCount = Math.max(2, Math.round((2 * xLimit) / PLANK_PITCH));
  const plankPitch = (2 * xLimit) / plankCount;
  const plankWidth = plankPitch - PLANK_GAP;
  {
    const rng = streamFor(ctx.seed, 'planks');
    for (let i = 0; i < plankCount; i++) {
      const cx = -xLimit + (i + 0.5) * plankPitch;
      const plank = new THREE.BoxGeometry(plankWidth, DECK_T, DECK_HALF_Z * 2);
      plank.translate(cx, DECK_Y - DECK_T * 0.5, 0);
      const k = rng();
      deckParts.push({
        geometry: plank,
        color: k < 0.34 ? woodPale : k < 0.72 ? woodMid : woodDark,
      });
    }
    for (const z of [-DECK_HALF_Z + 0.12, DECK_HALF_Z - 0.12]) {
      const beam = new THREE.BoxGeometry(2 * xLimit, 0.1, 0.07);
      beam.translate(0, DECK_Y - DECK_T - 0.05, z);
      deckParts.push({ geometry: beam, color: woodDark });
    }
    const pileCount = 7;
    for (let i = 0; i < pileCount; i++) {
      const px = -xLimit + (i * 2 * xLimit) / (pileCount - 1);
      for (const z of [-DECK_HALF_Z + 0.12, DECK_HALF_Z - 0.12]) {
        const h = DECK_Y - DECK_T - 0.1 - (WATER_Y - 0.5);
        const pile = new THREE.CylinderGeometry(0.045, 0.055, h, 6);
        pile.translate(px, WATER_Y - 0.5 + h * 0.5, z);
        deckParts.push({ geometry: pile, color: woodWet });
      }
    }
  }
  const deckGeometry = mergeColoredParts(deckParts, 'boardwalkDeck');
  const timberMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.88,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.24,
    name: 'boardwalkTimber',
  });
  const deckMesh = new THREE.Mesh(deckGeometry, timberMaterial);
  deckMesh.name = 'boardwalkDeck';
  deckMesh.castShadow = true;
  deckMesh.receiveShadow = true;
  group.add(deckMesh);
  owned.push({
    dispose: () => {
      deckGeometry.dispose();
      timberMaterial.dispose();
    },
  });

  const endRoots: THREE.Object3D[] = [];
  for (const sign of [-1, 1]) {
    const root = new THREE.Object3D();
    root.name = sign < 0 ? 'deckEndLeftRoot' : 'deckEndRightRoot';
    root.position.set(sign * xLimit, DECK_Y, 0);
    group.add(root);
    endRoots.push(root);

    const parts: ColoredPart[] = [];
    const board = new THREE.BoxGeometry(0.05, 0.075, DECK_HALF_Z * 2);
    board.translate(-sign * 0.025, -0.014, 0);
    parts.push({ geometry: board, color: woodDark });
    for (const z of [-DECK_HALF_Z + 0.075, DECK_HALF_Z - 0.075]) {
      const post = new THREE.CylinderGeometry(0.037, 0.045, 0.4, 7);
      post.translate(-sign * 0.045, 0.2, z);
      parts.push({ geometry: post, color: woodMid });
      const cap = new THREE.ConeGeometry(0.05, 0.07, 7);
      cap.translate(-sign * 0.045, 0.435, z);
      parts.push({ geometry: cap, color: woodPale });
      const band = new THREE.CylinderGeometry(0.048, 0.048, 0.024, 7);
      band.translate(-sign * 0.045, 0.315, z);
      parts.push({ geometry: band, color: woodWet });
    }
    const geo = mergeColoredParts(parts, root.name);
    const mesh = new THREE.Mesh(geo, timberMaterial);
    mesh.name = sign < 0 ? 'deckEndLeft' : 'deckEndRight';
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    root.add(mesh);
    owned.push({ dispose: () => geo.dispose() });
  }

  const cordParts: ColoredPart[] = [];
  {
    const span = xLimit + 0.06;
    const line = new THREE.BoxGeometry(span * 2, 0.008, 0.008);
    line.translate(0, cordY, CORD_Z);
    cordParts.push({ geometry: line, color: cordColor });
    for (const sx of [-1, 1]) {
      for (const sxAbs of [1.92, xLimit + 0.055]) {
        const px = sx * sxAbs;
        const h = cordY + 0.07 - DECK_Y;
        const stake = new THREE.CylinderGeometry(0.013, 0.018, h, 6);
        stake.translate(px, DECK_Y + h * 0.5, STAKE_Z);
        cordParts.push({ geometry: stake, color: woodPale });
        const tip = new THREE.ConeGeometry(0.022, 0.045, 6);
        tip.translate(px, DECK_Y + h + 0.018, STAKE_Z);
        cordParts.push({ geometry: tip, color: cordColor });
        const reach = CORD_Z - STAKE_Z;
        const arm = new THREE.BoxGeometry(0.012, 0.012, reach);
        arm.translate(px, cordY, STAKE_Z + reach * 0.5);
        cordParts.push({ geometry: arm, color: cordColor });
      }
    }
  }
  const cordGeometry = mergeColoredParts(cordParts, 'tiltCord');
  const cordMesh = new THREE.Mesh(cordGeometry, timberMaterial);
  cordMesh.name = 'tiltCord';
  cordMesh.castShadow = false;
  group.add(cordMesh);
  owned.push({ dispose: () => cordGeometry.dispose() });

  const gaugeRoot = new THREE.Object3D();
  gaugeRoot.name = 'tiltGaugeRoot';
  const station = gaugeStation(CAM_BASE, GAUGE_Z);
  gaugeRoot.position.set(station.x, station.y, station.z);
  gaugeRoot.scale.setScalar(station.scale);
  group.add(gaugeRoot);
  {
    const parts: ColoredPart[] = [];
    const FACE_Z = 0;
    const postTop = 0.06;
    const postBottom = (DECK_Y - 0.09 - station.y) / station.scale;
    const post = new THREE.CylinderGeometry(0.021, 0.032, postTop - postBottom, 6);
    post.translate(0, (postTop + postBottom) * 0.5, -0.05);
    parts.push({ geometry: post, color: woodMid });
    const foot = new THREE.BoxGeometry(0.2, 0.035, 0.11);
    foot.translate(0, postBottom + 0.02, -0.05);
    parts.push({ geometry: foot, color: woodDark });
    for (const sx of [-1, 1]) {
      const len = armLength * 1.05;
      for (const [w, dz, col] of [
        [0.036, -0.014, carve],
        [0.022, 0.004, driftwood],
      ] as const) {
        const arm = new THREE.BoxGeometry(w, len, w);
        arm.translate(0, len * 0.5, 0);
        arm.rotateZ(-sx * coneHalf);
        arm.translate(0, 0, FACE_Z + dz);
        parts.push({ geometry: arm, color: col });
      }
    }
    const ARC_SEGMENTS = 9;
    for (let i = 0; i < ARC_SEGMENTS; i++) {
      const a0 = -coneHalf + (2 * coneHalf * i) / ARC_SEGMENTS;
      const a1 = -coneHalf + (2 * coneHalf * (i + 1)) / ARC_SEGMENTS;
      const x0 = armLength * Math.sin(a0);
      const y0 = armLength * Math.cos(a0);
      const x1 = armLength * Math.sin(a1);
      const y1 = armLength * Math.cos(a1);
      const dx = x1 - x0;
      const dy = y1 - y0;
      const chord = Math.hypot(dx, dy) * 1.06;
      const ang = Math.atan2(dy, dx);
      for (const [w, dz, col] of [
        [0.03, -0.014, carve],
        [0.018, 0.004, driftwood],
      ] as const) {
        const seg = new THREE.BoxGeometry(chord, w, w);
        seg.rotateZ(ang);
        seg.translate((x0 + x1) * 0.5, (y0 + y1) * 0.5, FACE_Z + dz);
        parts.push({ geometry: seg, color: col });
      }
    }
    {
      const r = armLength * 0.52;
      const halfSpan = r * Math.sin(coneHalf);
      const tie = new THREE.BoxGeometry(halfSpan * 2, 0.014, 0.014);
      tie.translate(0, r * Math.cos(coneHalf), FACE_Z);
      parts.push({ geometry: tie, color: driftwood });
    }
    for (const sx of [-1, 1]) {
      const tick = new THREE.BoxGeometry(0.07, 0.026, 0.026);
      tick.rotateZ(-sx * coneHalf);
      tick.translate(sx * armLength * Math.sin(coneHalf), armLength * Math.cos(coneHalf), FACE_Z);
      parts.push({ geometry: tick, color: carve });
    }
    const geo = mergeColoredParts(parts, 'tiltGauge');
    const mesh = new THREE.Mesh(geo, timberMaterial);
    mesh.name = 'tiltGauge';
    mesh.castShadow = false;
    mesh.receiveShadow = true;
    gaugeRoot.add(mesh);
    owned.push({ dispose: () => geo.dispose() });
  }

  {
    const parts: ColoredPart[] = [];
    const rng = streamFor(ctx.seed, 'moorings');
    for (let i = 0; i < 7; i++) {
      const px = -5.2 + rng() * 10.4;
      const pz = -1.35 - rng() * 3.4;
      if (Math.abs(pz) < 0.95) continue;
      const h = 0.6 + rng() * 0.55;
      const post = new THREE.CylinderGeometry(0.035, 0.05, h, 6);
      post.translate(px, WATER_Y - 0.2 + h * 0.5, pz);
      parts.push({ geometry: post, color: woodWet });
    }
    if (parts.length > 0) {
      const geo = mergeColoredParts(parts, 'mooringPosts');
      const mesh = new THREE.Mesh(geo, timberMaterial);
      mesh.name = 'mooringPosts';
      mesh.castShadow = false;
      group.add(mesh);
      owned.push({ dispose: () => geo.dispose() });
    }
  }

  const padGeometry = new THREE.CircleGeometry(0.16, 9, 0.35, Math.PI * 2 - 0.7);
  padGeometry.rotateX(-Math.PI / 2);
  const padMaterial = oakMaterial({
    color: blendTokens(palette.groundA, TOKEN_BLACK, 0.2),
    flatShading: true,
    roughness: 0.72,
    side: THREE.DoubleSide,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSteps: 2,
    toonSoftness: 0.35,
    toonFloor: 0.3,
    name: 'lilyPadMaterial',
  });
  const PAD_COUNT = 96;
  const pads = new THREE.InstancedMesh(padGeometry, padMaterial, PAD_COUNT);
  pads.name = 'lilyPads';
  pads.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(PAD_COUNT * 3), 3);
  {
    const rng = streamFor(ctx.seed, 'lilies');
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    const s = new THREE.Vector3();
    const col = new THREE.Color();
    let n = 0;
    for (let i = 0; i < PAD_COUNT * 6 && n < PAD_COUNT; i++) {
      const px = -7 + rng() * 14;
      const pz = -6.5 + rng() * 9;
      const scale = 0.6 + rng() * 0.75;
      const yaw = rng() * Math.PI * 2;
      const k = 0.86 + rng() * 0.3;
      if (Math.abs(pz) < DECK_HALF_Z + 0.5 && Math.abs(px) < xLimit + 0.5) continue;
      if (shore(px, pz) > WATER_Y - 0.12) continue;
      p.set(px, WATER_Y + 0.006, pz);
      e.set(0, yaw, 0);
      q.setFromEuler(e);
      s.set(scale, 1, scale);
      m.compose(p, q, s);
      pads.setMatrixAt(n, m);
      col.setRGB(k, k * 1.03, k * 0.93);
      pads.setColorAt(n, col);
      n++;
    }
    pads.count = n;
    pads.instanceMatrix.needsUpdate = true;
    if (pads.instanceColor) pads.instanceColor.needsUpdate = true;
    pads.computeBoundingSphere();
  }
  group.add(pads);
  owned.push({
    dispose: () => {
      pads.dispose();
      padGeometry.dispose();
      padMaterial.dispose();
    },
  });

  const sampleAt = (x: number, z: number): number => terrain.sampleHeight(x, z);
  const underWater: ExcludeFn = (x, z) => terrain.sampleHeight(x, z) < WATER_Y + 0.05;
  const overTheCrossing: ExcludeFn = (x, z) =>
    Math.abs(z) < BANK_INNER_Z - 0.35 && Math.abs(x) < xLimit + 1.6;
  const inFront: ExcludeFn = (x, z) => z > 2.0 && Math.abs(x) < 1.25 + (z - 2.0) * 0.62;

  const grass = scatterGrass({
    count: 3400,
    area: { x: 0, z: -9, halfX: 12, halfZ: 3.6 },
    seed: ctx.seed ^ 0x7f19,
    exclude: excludeAny(underWater, overTheCrossing, inFront),
    heightAt: sampleAt,
    maxSlope: 0.8,
    color: blendTokens(blendTokens(palette.groundA, palette.groundB, 0.34), TOKEN_BLACK, 0.12),
    scale: [0.95, 1.6],
    stretch: [0.9, 1.3],
    hueJitter: 0.22,
    wind: GRASS_WIND,
    windFrequency: 1.6,
    rimColor: palette.skyHorizon,
    rimStrength: 0.4,
    name: 'bankGrass',
  });
  group.add(grass.group);
  owned.push(grass);
  timed.push(grass);

  const reeds = scatterGrass({
    count: 620,
    area: { x: 0, z: -7.4, halfX: 11, halfZ: 2.6 },
    seed: ctx.seed ^ 0x3d51,
    exclude: excludeAny(
      overTheCrossing,
      inFront,
      (x, z) => {
        const h = terrain.sampleHeight(x, z);
        return h < WATER_Y - 0.08 || h > WATER_Y + 0.18;
      },
    ),
    heightAt: sampleAt,
    color: reedGreen,
    scale: [0.85, 1.45],
    stretch: [1.0, 1.4],
    hueJitter: 0.26,
    wind: REED_WIND,
    windFrequency: 1.15,
    rimColor: palette.skyHorizon,
    rimStrength: 0.5,
    name: 'lakeReeds',
  });
  group.add(reeds.group);
  owned.push(reeds);
  timed.push(reeds);

  const fringe = scatterGrass({
    count: 150,
    area: { x: 0, z: 2.15, halfX: 3.1, halfZ: 0.55 },
    seed: ctx.seed ^ 0x9ab3,
    exclude: (x, z) => Math.abs(x) < 1.3 + (z - 1.6) * 0.7,
    heightAt: () => WATER_Y - 0.05,
    color: blendTokens(reedGreen, TOKEN_BLACK, 0.3),
    scale: [1.1, 1.8],
    stretch: [1.05, 1.5],
    hueJitter: 0.2,
    wind: REED_WIND * 1.4,
    windFrequency: 1.05,
    rimColor: palette.skyHorizon,
    rimStrength: 0.55,
    name: 'foregroundReeds',
  });
  group.add(fringe.group);
  owned.push(fringe);
  timed.push(fringe);

  const flowers = scatterFlowers({
    count: 340,
    area: { x: 0, z: -8.8, halfX: 11, halfZ: 3 },
    seed: ctx.seed ^ 0x2c71,
    exclude: excludeAny(underWater, overTheCrossing, inFront),
    heightAt: sampleAt,
    maxSlope: 0.7,
    color: blendTokens(accent, TOKEN_WHITE, 0.42),
    scale: [0.8, 1.5],
    hueJitter: 0.3,
    wind: 0.05,
    windFrequency: 1.9,
    rimColor: palette.skyHorizon,
    rimStrength: 0.4,
    name: 'bankFlowers',
  });
  group.add(flowers.group);
  owned.push(flowers);
  timed.push(flowers);

  const rocks = scatterRocks({
    count: 220,
    area: { x: 0, z: -7.2, halfX: 11, halfZ: 2.4 },
    seed: ctx.seed ^ 0x2b17,
    exclude: excludeAny(overTheCrossing, inFront, (x, z) => {
      const h = terrain.sampleHeight(x, z);
      return h < WATER_Y - 0.3 || h > WATER_Y + 0.7;
    }),
    heightAt: sampleAt,
    color: stone,
    scale: [0.14, 0.42],
    stretch: [0.5, 0.95],
    hueJitter: 0.24,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    castShadow: false,
    name: 'shoreStones',
  });
  group.add(rocks.group);
  owned.push(rocks);
  timed.push(rocks);

  const trees = scatterTrees({
    archetypes: ['broadleaf', 'conifer'],
    count: 120,
    area: { x: 0, z: -16, halfX: 15, halfZ: 6 },
    seed: ctx.seed ^ 0x5133,
    exclude: excludeAny(underWater, (x, z) => z > -10.5),
    heightAt: sampleAt,
    maxSlope: 0.65,
    color: blendTokens(palette.groundA, TOKEN_BLACK, 0.3),
    scale: [0.5, 1.22],
    stretch: [0.9, 1.4],
    hueJitter: 0.24,
    wind: TREE_WIND,
    windFrequency: 0.85,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    castShadow: false,
    name: 'farShoreTrees',
  });
  group.add(trees.group);
  owned.push(trees);
  timed.push(trees);

  const pollen: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'pollen',
    count: 190,
    bounds: { x: 0, y: 0.42, z: -1.1, halfX: 4.2, halfY: 1.05, halfZ: 3 },
    seed: ctx.seed,
    color: blendTokens(palette.sun, accent, 0.4),
    size: 0.5,
    intensity: 0.85,
    opacity: 0.3,
    rise: 0.05,
    swirl: 0.12,
    name: 'lakePollen',
  });
  group.add(pollen.points);
  owned.push(pollen);
  timed.push(pollen);

  const spray: AmbientParticlesHandle = makeAmbientParticles({
    preset: 'dust',
    count: 130,
    bounds: { x: 0, y: WATER_Y + 0.22, z: -3.2, halfX: 7, halfY: 0.3, halfZ: 3.4 },
    seed: ctx.seed ^ 0x51ab,
    color: blendTokens(TOKEN_WHITE, palette.hemiSky, 0.3),
    size: 0.4,
    intensity: 0.9,
    opacity: 0.13,
    rise: 0.03,
    swirl: 0.18,
    name: 'lakeSpray',
  });
  group.add(spray.points);
  owned.push(spray);
  timed.push(spray);

  const haze: MistPlanesHandle = makeMistPlanes({
    size: [30, 14],
    color: blendTokens(palette.fog, palette.hemiSky, 0.45),
    seed: ctx.seed,
    layers: 4,
    baseY: WATER_Y + 0.16,
    spread: 0.85,
    opacity: 0.055,
    speed: 0.008,
    scale: 2.2,
    nearFade: 3.4,
    name: 'lakeHaze',
  });
  haze.group.position.set(0, 0, -12);
  group.add(haze.group);
  owned.push(haze);
  timed.push(haze);

  const clouds: MistPlanesHandle = makeMistPlanes({
    size: [56, 20],
    color: blendTokens(TOKEN_WHITE, palette.skyHorizon, 0.12),
    seed: ctx.seed ^ 0x71c3,
    layers: 3,
    baseY: 3.2,
    spread: 1.9,
    opacity: 0.42,
    speed: 0.004,
    scale: 1.5,
    nearFade: 9,
    name: 'noonClouds',
  });
  clouds.group.position.set(0, 0, -19);
  group.add(clouds.group);
  owned.push(clouds);
  timed.push(clouds);

  const animated: OakMaterial[] = [padMaterial, timberMaterial];
  timed.push({ setTime: (t) => setMaterialsTime(animated, t) });

  const sunBase = rig.sun.intensity;
  const hemiBase = rig.hemi.intensity;

  return {
    group,
    sky,
    rig,
    water,
    terrain,
    endLeftRoot: endRoots[0] as THREE.Object3D,
    endRightRoot: endRoots[1] as THREE.Object3D,
    gaugeRoot,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
    },
    setWind(lean) {
      const l = clampRange(lean, -1, 1);
      const k = 0.45 + 0.75 * l;
      grass.setWind(GRASS_WIND * k, 1.6 + 0.5 * Math.abs(l));
      reeds.setWind(REED_WIND * k, 1.15 + 0.4 * Math.abs(l));
      fringe.setWind(REED_WIND * 1.4 * k, 1.05 + 0.35 * Math.abs(l));
      trees.setWind(TREE_WIND * (0.7 + 0.4 * Math.abs(l)), 0.85);
    },
    setCool(env01) {
      const c = clampRange(env01, 0, 1);
      rig.sun.intensity = sunBase * (1 - 0.13 * c);
      rig.hemi.intensity = hemiBase * (1 + 0.34 * c);
    },
    setPixelRatio(dpr) {
      pollen.setPixelRatio(dpr);
      spray.setPixelRatio(dpr);
    },
    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      animated.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}
