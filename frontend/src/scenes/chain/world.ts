
import * as THREE from 'three';
import {
  CELL,
  buildCellField,
  cellToWorld,
  type CellFieldHandle,
  type GridSize,
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, type OakMaterial } from '@kit/materials';
import { makeAmbientParticles, makeMistPlanes, type AmbientParticlesHandle } from '@kit/particles';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { makeTerrain, type TerrainHandle } from '@kit/terrain';
import { valueNoise2 } from '@kit/textures';
import { TOKENS, type Palette } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import {
  excludeAny,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  scatterTrees,
  type ExcludeFn,
  type VegetationHandle,
} from '@kit/vegetation';
import { makeWater, type WaterHandle } from '@kit/water';
import { CHAIN_END, CHAIN_L, chainDecoder } from '@sim/envs/chain';
import type { EnvGeometry, GridGeometry } from '@sim/types';


export const WATER_LEVEL = -0.17;
export const BED_Y = -1.05;
export const BANK_Y = 0.62;
export const CREEK_HALF = 8.0;
export const BANK_RISE = 2.2;
export const MEANDER = 0.55;

export const TERRAIN_SIZE: readonly [number, number] = [52, 80];
export const TERRAIN_SEGMENTS: readonly [number, number] = [120, 176];
export const FOG_EXTENT = 300;
export const SUN_DIR: readonly [number, number, number] = [-0.46, 0.62, -0.64];
export const SHADOW_RADIUS = 12.5;

export const CAMERA_AZIMUTH = 0.35;
export const CAMERA_STANDOFF = 13.5;


export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function blendTokens(
  a: THREE.ColorRepresentation,
  b: THREE.ColorRepresentation,
  k: number,
): THREE.Color {
  return new THREE.Color(a).lerp(new THREE.Color(b), k);
}

export function shade(color: THREE.ColorRepresentation, k: number): THREE.Color {
  return blendTokens(color, TOKENS.palettes.night.skyTop, k);
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

export function tubeGeometry(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  ra: number,
  rb: number,
  radial: number,
): THREE.BufferGeometry {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-4;
  const g = new THREE.CylinderGeometry(rb, ra, len, Math.max(3, radial), 1);
  g.translate(0, len / 2, 0);
  const q = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(dx / len, dy / len, dz / len),
  );
  g.applyQuaternion(q);
  g.translate(a[0], a[1], a[2]);
  return g;
}


export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === CHAIN_L && g.h === 1) return g;
  const fallback = chainDecoder.geometry(chainDecoder.initStatic(0), 0).grid;
  if (fallback) return fallback;
  return { w: CHAIN_L, h: 1, walls: new Uint8Array(CHAIN_L), goal: { x: CHAIN_END, y: 0 } };
}

export function stoneX(index: number, grid: GridSize): number {
  const out: Vec3Tuple = [0, 0, 0];
  return cellToWorld(index, 0, grid, out)[0];
}


export interface CreekSkin {
  stoneDry: THREE.Color;
  stoneWet: THREE.Color;
  stoneDeep: THREE.Color;
  stoneGlow: THREE.Color;
  bed: THREE.Color;
  bank: THREE.Color;
  bankHigh: THREE.Color;
  reed: THREE.Color;
  meadow: THREE.Color;
  flower: THREE.Color;
  bankTree: THREE.Color;
  bark: THREE.Color;
  moss: THREE.Color;
  blossom: THREE.Color;
  waterShallow: THREE.Color;
  waterDeep: THREE.Color;
  waterSky: THREE.Color;
  waterCaustic: THREE.Color;
  motes: THREE.Color;
  dragonfly: THREE.Color;
  wing: THREE.Color;
}

export function creekSkin(palette: Palette, tokens: typeof TOKENS): CreekSkin {
  const blossom = tokens.accents['chain'] ?? tokens.celebrationGold;
  const leaf = tokens.accents['cart-pole'] ?? palette.groundA;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const petalPink = tokens.accents['acrobot'] ?? blossom;
  const teal = tokens.accents['deep-sea'] ?? palette.hemiSky;
  const white = palette.sun;

  return {
    stoneDry: shade(blendTokens(palette.groundB, silver, 0.28), 0.46),
    stoneWet: shade(blendTokens(silver, palette.hemiGround, 0.52), 0.64),
    stoneDeep: shade(blendTokens(palette.hemiGround, silver, 0.2), 0.72),
    stoneGlow: blendTokens(blossom, tokens.celebrationGold, 0.42),
    bed: shade(blendTokens(palette.groundB, palette.groundA, 0.42), 0.56),
    bank: shade(blendTokens(palette.groundA, palette.groundB, 0.38), 0.48),
    bankHigh: shade(blendTokens(palette.groundB, leaf, 0.3), 0.44),
    reed: shade(blendTokens(leaf, palette.groundA, 0.34), 0.44),
    meadow: shade(blendTokens(palette.groundA, leaf, 0.3), 0.4),
    flower: shade(blendTokens(petalPink, white, 0.2), 0.3),
    bankTree: shade(blendTokens(palette.groundA, leaf, 0.34), 0.46),
    bark: shade(blendTokens(palette.hemiGround, TOKENS.palettes.dusk.groundA, 0.4), 0.5),
    moss: shade(blendTokens(leaf, palette.groundA, 0.5), 0.52),
    blossom: shade(blendTokens(blossom, white, 0.1), 0.2),
    waterShallow: shade(blendTokens(blendTokens(teal, palette.groundA, 0.62), palette.hemiGround, 0.28), 0.44),
    waterDeep: shade(blendTokens(blendTokens(teal, palette.groundA, 0.5), palette.skyTop, 0.34), 0.7),
    waterSky: shade(palette.skyTop, 0.46),
    waterCaustic: shade(blendTokens(palette.sun, teal, 0.34), 0.46),
    motes: blendTokens(blossom, white, 0.42),
    dragonfly: shade(blendTokens(teal, palette.skyTop, 0.3), 0.62),
    wing: shade(blendTokens(silver, white, 0.4), 0.24),
  };
}


export function makeCreekHeight(seed: number): (x: number, z: number) => number {
  const span = BANK_Y - BED_Y;
  return (x, z) => {
    const wander = (valueNoise2(z * 0.075 + 3.1, 7.7, 64, seed) - 0.5) * 2 * MEANDER;
    const s = smoothstep01((Math.abs(x) + wander - CREEK_HALF) / BANK_RISE);
    const bend = smoothstep01((-z - 19) / 11);
    const land = s > bend ? s : bend;
    const far = smoothstep01((Math.abs(z) - 13) / 20) * 1.9 * land;
    const base = BED_Y + span * land + far;
    const w = smoothstep01(Math.abs(z) / 2.4) * 0.55 + land * 0.85;
    if (w <= 0) return base;
    const n1 = valueNoise2(x * 0.12 + 5.3, z * 0.12 + 11.7, 64, seed) - 0.5;
    const n2 = valueNoise2(x * 0.3 + 2.1, z * 0.3 + 7.3, 64, seed ^ 0x51ed270b) - 0.5;
    return base + w * (n1 * 0.9 + n2 * 0.34);
  };
}


export interface StoneCapsHandle {
  mesh: THREE.InstancedMesh;
  material: OakMaterial;
  count: number;
  setGlow(index: number, amount: number): void;
  clearGlow(): void;
  dispose(): void;
}

export const CAP_R_TOP = 0.37;
export const CAP_R_WAIST = 0.42;
export const CAP_R_FOOT = 0.4;
export const CAP_WAIST_FRAC = 0.42;
export const CAP_HEIGHT = 0.5;
export const CAP_SCALE_MIN = 0.9;
export const CAP_SCALE_MAX = 1.08;

export function capSides(tier: QualityTier): number {
  return tier === 'thumb' ? 5 : tier === 'medium' ? 6 : 7;
}

export function capApothem(sides: number, xzScale: number): number {
  return CAP_R_TOP * xzScale * Math.cos(Math.PI / Math.max(3, sides));
}

export const PLINTH_DROP = 0.06;
export const PLINTH_HALF = (capApothem(5, CAP_SCALE_MIN) * 0.94) / Math.SQRT2;
export const PLINTH_INSET = CELL / 2 - PLINTH_HALF;

export function stoneCapGeometry(sides: number, height: number): THREE.BufferGeometry {
  const n = Math.max(5, sides);
  const rTop = CAP_R_TOP;
  const rWaist = CAP_R_WAIST;
  const rFoot = CAP_R_FOOT;
  const yWaist = -height * CAP_WAIST_FRAC;
  const yFoot = -height;
  const gTop = 1.14;
  const gWaist = 0.9;
  const gFoot = 0.52;

  const tris = n  + n * 2  + n * 2 ;
  const positions = new Float32Array(tris * 3 * 3);
  const colors = new Float32Array(tris * 3 * 3);
  let o = 0;
  const push = (x: number, y: number, z: number, g: number): void => {
    positions[o * 3] = x;
    positions[o * 3 + 1] = y;
    positions[o * 3 + 2] = z;
    colors[o * 3] = g;
    colors[o * 3 + 1] = g;
    colors[o * 3 + 2] = g;
    o++;
  };
  const ring = (i: number, r: number): [number, number] => {
    const a = ((i % n) / n) * Math.PI * 2;
    return [Math.cos(a) * r, Math.sin(a) * r];
  };

  for (let i = 0; i < n; i++) {
    const [ax, az] = ring(i, rTop);
    const [bx, bz] = ring(i + 1, rTop);
    push(0, 0, 0, gTop);
    push(bx, 0, bz, gTop);
    push(ax, 0, az, gTop);

    const [wax, waz] = ring(i, rWaist);
    const [wbx, wbz] = ring(i + 1, rWaist);
    push(ax, 0, az, gTop);
    push(bx, 0, bz, gTop);
    push(wbx, yWaist, wbz, gWaist);
    push(ax, 0, az, gTop);
    push(wbx, yWaist, wbz, gWaist);
    push(wax, yWaist, waz, gWaist);

    const [fax, faz] = ring(i, rFoot);
    const [fbx, fbz] = ring(i + 1, rFoot);
    push(wax, yWaist, waz, gWaist);
    push(wbx, yWaist, wbz, gWaist);
    push(fbx, yFoot, fbz, gFoot);
    push(wax, yWaist, waz, gWaist);
    push(fbx, yFoot, fbz, gFoot);
    push(fax, yFoot, faz, gFoot);
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'chainStoneCap';
  return g;
}

function buildStoneCaps(
  grid: GridGeometry,
  skin: CreekSkin,
  palette: Palette,
  tier: QualityTier,
  seed: number,
): StoneCapsHandle {
  const sides = capSides(tier);
  const geometry = stoneCapGeometry(sides, CAP_HEIGHT);
  const material = oakMaterial({
    color: 0xffffff,
    vertexColors: true,
    flatShading: true,
    roughness: 0.95,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    toonSteps: 3,
    toonSoftness: 0.3,
    toonFloor: 0.22,
    tier,
    name: 'chainStoneCapMaterial',
  });

  const count = grid.w;
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
  mesh.name = 'chainStoneCaps';
  mesh.castShadow = tier !== 'thumb';
  mesh.receiveShadow = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * 3), 3);

  const base = new Float32Array(Math.max(1, count) * 3);
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3();
  const color = new THREE.Color();
  const world: Vec3Tuple = [0, 0, 0];

  for (let i = 0; i < count; i++) {
    cellToWorld(i, 0, grid, world);
    const n = hash2f(i, 0, seed);
    const m = hash2f(i, 1, seed ^ 0x51ed270b);
    position.set(world[0], 0, world[2]);
    euler.set(0, n * Math.PI * 2, 0);
    quaternion.setFromEuler(euler);
    const xz = CAP_SCALE_MIN + m * (CAP_SCALE_MAX - CAP_SCALE_MIN);
    scale.set(xz, 0.78 + n * 0.5, xz);
    matrix.compose(position, quaternion, scale);
    mesh.setMatrixAt(i, matrix);
    color.copy(skin.stoneDry).lerp(skin.stoneWet, m * 0.3).multiplyScalar(0.86 + n * 0.3);
    base[i * 3] = color.r;
    base[i * 3 + 1] = color.g;
    base[i * 3 + 2] = color.b;
    mesh.setColorAt(i, color);
  }
  mesh.count = count;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();

  const scratch = new THREE.Color();
  const glow = new THREE.Color().copy(skin.stoneGlow);

  return {
    mesh,
    material,
    count,
    setGlow(index, amount) {
      if (index < 0 || index >= count) return;
      const a = amount < 0 ? 0 : amount > 1 ? 1 : amount;
      scratch.setRGB(base[index * 3] ?? 1, base[index * 3 + 1] ?? 1, base[index * 3 + 2] ?? 1);
      if (a > 0) scratch.lerp(glow, a * 0.45).multiplyScalar(1 + a * 0.3);
      mesh.setColorAt(index, scratch);
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    },
    clearGlow() {
      for (let i = 0; i < count; i++) {
        scratch.setRGB(base[i * 3] ?? 1, base[i * 3 + 1] ?? 1, base[i * 3 + 2] ?? 1);
        mesh.setColorAt(i, scratch);
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    },
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}


export interface DragonfliesHandle {
  mesh: THREE.InstancedMesh;
  count: number;
  setTime(t: number): void;
  dispose(): void;
}

function dragonflyGeometry(skin: CreekSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const body = new THREE.CylinderGeometry(0.012, 0.02, 0.24, 5, 1);
  body.rotateX(Math.PI / 2);
  parts.push({ geometry: body, color: skin.dragonfly });
  const head = new THREE.IcosahedronGeometry(0.032, 0);
  head.translate(0, 0, 0.13);
  parts.push({ geometry: head, color: skin.dragonfly });
  for (const sx of [-1, 1]) {
    for (const oz of [0.05, -0.03]) {
      const wing = new THREE.PlaneGeometry(0.19, 0.055);
      wing.rotateY(Math.PI / 2);
      wing.rotateZ(sx > 0 ? -0.22 : 0.22);
      wing.translate(sx * 0.105, 0.018, oz);
      parts.push({ geometry: wing, color: skin.wing });
    }
  }
  return mergeColoredParts(parts, 'chainDragonfly');
}

function buildDragonflies(
  skin: CreekSkin,
  palette: Palette,
  tier: QualityTier,
  seed: number,
): DragonfliesHandle {
  const count = tier === 'thumb' ? 0 : 3;
  const geometry = dragonflyGeometry(skin);
  const material = oakMaterial({
    vertexColors: true,
    flatShading: true,
    side: THREE.DoubleSide,
    roughness: 0.5,
    metalness: 0.1,
    rimColor: palette.skyHorizon,
    rimStrength: 0.55,
    toonSteps: 2,
    toonSoftness: 0.4,
    toonFloor: 0.4,
    tier,
    name: 'chainDragonflyMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
  mesh.name = 'chainDragonflies';
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.count = count;
  mesh.visible = count > 0;
  mesh.frustumCulled = false;

  const phase = new Float32Array(Math.max(1, count) * 4);
  for (let i = 0; i < count; i++) {
    phase[i * 4] = hash2f(i, 11, seed) * Math.PI * 2;
    phase[i * 4 + 1] = 0.42 + hash2f(i, 12, seed) * 0.36;
    phase[i * 4 + 2] = (hash2f(i, 13, seed) - 0.5) * 2;
    phase[i * 4 + 3] = 0.72 + hash2f(i, 14, seed) * 0.5;
  }

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const scale = new THREE.Vector3(1, 1, 1);

  return {
    mesh,
    count,
    setTime(t) {
      if (count === 0) return;
      for (let i = 0; i < count; i++) {
        const p0 = phase[i * 4] ?? 0;
        const rate = phase[i * 4 + 1] ?? 0.5;
        const lane = phase[i * 4 + 2] ?? 0;
        const size = phase[i * 4 + 3] ?? 1;
        const a = t * rate + p0;
        const x = Math.sin(a) * 6.4 + Math.sin(a * 2.3 + p0) * 1.1;
        const z = lane * 2.1 + Math.cos(a * 1.7 + p0) * 1.35;
        const y = 0.62 + Math.sin(a * 2.9 + p0) * 0.22;
        const dx = Math.cos(a) * 6.4 * rate + Math.cos(a * 2.3 + p0) * 2.53 * rate;
        const dz = -Math.sin(a * 1.7 + p0) * 2.295 * rate;
        position.set(x, y, z);
        euler.set(Math.sin(a * 4.1) * 0.12, Math.atan2(dx, dz), Math.sin(a * 5.3 + p0) * 0.25);
        quaternion.setFromEuler(euler);
        scale.setScalar(size);
        matrix.compose(position, quaternion, scale);
        mesh.setMatrixAt(i, matrix);
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


export function excludeCrossing(grid: GridSize, halfZ: number, margin: number): ExcludeFn {
  const halfX = ((grid.w - 1) / 2) * CELL + 0.5 + margin;
  return (x, z) => Math.abs(z) <= halfZ && Math.abs(x) <= halfX;
}

export function excludeStones(grid: GridSize, radius: number): ExcludeFn {
  const r2 = radius * radius;
  const ox = (grid.w - 1) / 2;
  return (x, z) => {
    const i = Math.round(x / CELL + ox);
    if (i < 0 || i > grid.w - 1) return false;
    const dx = x - (i - ox) * CELL;
    return dx * dx + z * z < r2;
  };
}

export function excludeCameraWedge(halfX: number, halfZ: number): ExcludeFn {
  const cx = Math.sin(CAMERA_AZIMUTH) * CAMERA_STANDOFF;
  const cz = Math.cos(CAMERA_AZIMUTH) * CAMERA_STANDOFF;
  return (x, z) => Math.abs(x - cx) <= halfX && Math.abs(z - cz) <= halfZ;
}


export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  terrain: TerrainHandle;
  water: WaterHandle;
  field: CellFieldHandle;
  caps: StoneCapsHandle;
  petalMotes: AmbientParticlesHandle;
  setTime(t: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

interface Disposable {
  dispose(): void;
}

interface Timed {
  setTime(t: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const tier: QualityTier = ctx.tier;
  const palette: Palette = ctx.tokens.palettes.afternoon;
  const skin = creekSkin(palette, ctx.tokens);
  const shadows = tier !== 'thumb';
  const group = new THREE.Group();
  group.name = 'sixteenStonesWorld';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];

  const sky = skyFromPalette(palette, 'afternoon', tier, {
    sunDir: SUN_DIR,
    sunSize: 0.04,
    sunHdr: 5.2,
    sunGlow: 0.26,
    sunGlowPower: 26,
    topSpread: 0.68,
    bottomSpread: 0.46,
    radius: 340,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, tier, {
    sunDir: sky.sunDirection,
    pointCount: 3,
    pointDistance: 7.5,
    sunDistanceScale: 2.1,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: 0, y: 0.1, z: 0 }, SHADOW_RADIUS);

  applyFog(scene, palette, FOG_EXTENT);

  const terrain = makeTerrain({
    size: TERRAIN_SIZE,
    segments: TERRAIN_SEGMENTS,
    tier,
    seed: ctx.seed,
    heightFn: makeCreekHeight(ctx.seed),
    jitter: 0,
    colorNoise: 0.05,
    slopeColor: skin.bank,
    slopeStrength: 0.55,
    slopeSharpness: 2.2,
    colorRamp: [
      { t: 0, color: skin.bed },
      { t: 0.22, color: blendTokens(skin.bed, skin.bank, 0.55) },
      { t: 0.34, color: skin.bank },
      { t: 0.52, color: skin.bankHigh },
      { t: 1, color: blendTokens(skin.bankHigh, palette.hemiSky, 0.28) },
    ],
    materialOptions: {
      rimColor: palette.skyHorizon,
      rimStrength: 0.16,
      toonSteps: 3,
      toonSoftness: 0.3,
      toonFloor: 0.24,
    },
    receiveShadow: shadows,
    castShadow: false,
    name: 'creekBed',
  });
  group.add(terrain.mesh);
  owned.push(terrain);

  const water = makeWater({
    size: [TERRAIN_SIZE[0] + 8, TERRAIN_SIZE[1] + 10],
    palette,
    tier,
    segments: 128,
    level: WATER_LEVEL,
    shallow: skin.waterShallow,
    deep: skin.waterDeep,
    sky: skin.waterSky,
    caustic: skin.waterCaustic,
    amplitude: 0.024,
    waveScale: 1.45,
    speed: 0.8,
    fresnel: 0.24,
    caustics: 0.14,
    depthFalloff: 9,
    opacity: 0.8,
    sunDir: sky.sunDirection,
    name: 'creekWater',
  });
  water.setColors(skin.waterShallow, skin.waterDeep);
  group.add(water.mesh);
  owned.push(water);
  timed.push({ setTime: (t) => water.setTime(t) });

  const field = buildCellField(
    grid,
    {
      floorColor: skin.stoneDeep,
      wallColor: skin.stoneDeep,
      floorInset: PLINTH_INSET,
      floorThickness: 0.9,
      jitter: 0.12,
      rimColor: palette.skyHorizon,
      rimStrength: 0.12,
      seed: ctx.seed,
      name: 'stonePlinths',
    },
    tier,
  );
  field.group.position.y = -PLINTH_DROP;
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  const caps = buildStoneCaps(grid, skin, palette, tier, ctx.seed);
  group.add(caps.mesh);
  owned.push(caps);

  const offCrossing = excludeAny(
    excludeCrossing(grid, 1.15, 0.6),
    excludeStones(grid, 0.78),
    excludeCameraWedge(4.6, 4.6),
  );
  const heightAt = (x: number, z: number): number => terrain.sampleHeight(x, z);

  const reeds: VegetationHandle[] = [];
  for (const side of [-1, 1] as const) {
    const handle = scatterGrass({
      count: 900,
      area: { x: side * 9.4, z: 0, halfX: 2.1, halfZ: 26 },
      seed: (ctx.seed ^ 0x4ee1) + (side > 0 ? 977 : 0),
      tier,
      exclude: offCrossing,
      heightAt,
      color: skin.reed,
      scale: [0.85, 1.6],
      stretch: [1.2, 2.2],
      hueJitter: 0.26,
      wind: 0.05,
      windFrequency: 1.15,
      rimColor: palette.skyHorizon,
      rimStrength: 0.4,
      name: `creekReeds${side > 0 ? 'Far' : 'Near'}`,
    });
    group.add(handle.group);
    owned.push(handle);
    timed.push(handle);
    reeds.push(handle);
  }

  const shallowTufts = scatterGrass({
    count: 220,
    area: { x: -5.6, z: 6.8, halfX: 4.2, halfZ: 4.2 },
    seed: ctx.seed ^ 0x1c93,
    tier,
    exclude: excludeAny(excludeCrossing(grid, 2.2, 0.6), excludeCameraWedge(3.4, 3.4)),
    heightAt,
    color: skin.reed,
    scale: [0.85, 1.35],
    stretch: [1.9, 2.5],
    hueJitter: 0.28,
    wind: 0.055,
    windFrequency: 1.25,
    rimColor: palette.skyHorizon,
    rimStrength: 0.42,
    name: 'shallowReeds',
  });
  group.add(shallowTufts.group);
  owned.push(shallowTufts);
  timed.push(shallowTufts);
  reeds.push(shallowTufts);

  const meadow = scatterGrass({
    count: 1100,
    area: { x: 0, z: 0, halfX: 24, halfZ: 30 },
    seed: ctx.seed ^ 0x27a1,
    tier,
    exclude: excludeAny(offCrossing, (x) => Math.abs(x) < 10.4),
    heightAt,
    color: skin.meadow,
    scale: [0.32, 0.72],
    stretch: [0.8, 1.5],
    hueJitter: 0.24,
    wind: 0.045,
    windFrequency: 1.5,
    rimColor: palette.skyHorizon,
    rimStrength: 0.32,
    name: 'bankMeadow',
  });
  group.add(meadow.group);
  owned.push(meadow);
  timed.push(meadow);

  const flowers = scatterFlowers({
    count: 420,
    area: { x: 0, z: 0, halfX: 22, halfZ: 22 },
    seed: ctx.seed ^ 0x5b10,
    tier,
    exclude: excludeAny(offCrossing, (x) => Math.abs(x) < 10.0),
    heightAt,
    color: skin.flower,
    scale: [0.6, 1.15],
    stretch: [0.9, 1.5],
    hueJitter: 0.26,
    wind: 0.035,
    windFrequency: 1.6,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    name: 'bankFlowers',
  });
  group.add(flowers.group);
  owned.push(flowers);
  timed.push(flowers);

  const rocks = scatterRocks({
    count: 260,
    area: { x: 0, z: 0, halfX: 21, halfZ: 20 },
    seed: ctx.seed ^ 0x3f77,
    tier,
    exclude: excludeAny(excludeStones(grid, 0.95), excludeCameraWedge(2.6, 2.6)),
    heightAt,
    color: shade(blendTokens(skin.stoneWet, skin.bed, 0.3), 0.1),
    scale: [0.14, 0.5],
    stretch: [0.42, 0.86],
    hueJitter: 0.3,
    rimColor: palette.skyHorizon,
    rimStrength: 0.26,
    castShadow: false,
    name: 'creekRocks',
  });
  group.add(rocks.group);
  owned.push(rocks);
  timed.push(rocks);

  const bankTrees = scatterTrees({
    archetypes: ['broadleaf', 'conifer'],
    count: 100,
    area: { x: 0, z: 0, halfX: 24, halfZ: 34 },
    seed: ctx.seed ^ 0x1a70,
    tier,
    exclude: excludeAny(
      (x, z) => Math.abs(x) < 11.6 || (Math.abs(z) < 13 && Math.abs(x) < 16),
      excludeCameraWedge(7.5, 7.5),
    ),
    heightAt,
    maxSlope: 0.62,
    color: skin.bankTree,
    scale: [1.25, 2.4],
    stretch: [1.0, 1.6],
    hueJitter: 0.24,
    wind: 0.028,
    windFrequency: 0.85,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    castShadow: true,
    name: 'bankTrees',
  });
  group.add(bankTrees.group);
  owned.push(bankTrees);
  timed.push(bankTrees);

  const motes: AmbientParticlesHandle[] = [];
  const pollen = makeAmbientParticles({
    preset: 'pollen',
    count: 240,
    bounds: { x: 0, y: 0.95, z: 0, halfX: 11, halfY: 1.0, halfZ: 7 },
    seed: ctx.seed ^ 0x7d21,
    tier,
    color: blendTokens(palette.sun, skin.motes, 0.3),
    size: 0.8,
    intensity: 1.15,
    opacity: 0.36,
    rise: 0.06,
    swirl: 0.3,
    name: 'creekPollen',
  });
  group.add(pollen.points);
  owned.push(pollen);
  timed.push(pollen);
  motes.push(pollen);

  const petalMotes = makeAmbientParticles({
    preset: 'snow',
    count: 220,
    bounds: { x: stoneX(CHAIN_END, grid) - 0.4, y: 1.5, z: 0, halfX: 2.4, halfY: 1.5, halfZ: 2.1 },
    seed: ctx.seed ^ 0x6c31,
    tier,
    color: skin.blossom,
    size: 1.25,
    intensity: 1.5,
    opacity: 0.78,
    rise: -0.22,
    swirl: 0.4,
    name: 'blossomPetalMotes',
  });
  petalMotes.setDensity(0);
  group.add(petalMotes.points);
  owned.push(petalMotes);
  timed.push(petalMotes);
  motes.push(petalMotes);

  const haze = makeMistPlanes({
    size: [50, 70],
    color: blendTokens(palette.fog, palette.sun, 0.25),
    seed: ctx.seed ^ 0x2c1b,
    tier,
    layers: 2,
    baseY: 0.02,
    spread: 0.7,
    opacity: 0.06,
    speed: 0.008,
    scale: 3.2,
    nearFade: 11,
    name: 'creekHaze',
  });
  group.add(haze.group);
  owned.push(haze);
  timed.push(haze);

  const dragonflies = buildDragonflies(skin, palette, tier, ctx.seed);
  group.add(dragonflies.mesh);
  owned.push(dragonflies);
  timed.push(dragonflies);

  return {
    group,
    sky,
    rig,
    terrain,
    water,
    field,
    caps,
    petalMotes,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
    },
    setPixelRatio(dpr) {
      for (let i = 0; i < motes.length; i++) motes[i]?.setPixelRatio(dpr);
    },
    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      motes.length = 0;
      reeds.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}
