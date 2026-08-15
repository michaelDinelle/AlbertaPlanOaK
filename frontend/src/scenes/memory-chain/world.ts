
import * as THREE from 'three';
import {
  CELL,
  buildCellField,
  cellToWorld,
  gridExtent,
  type CellFieldHandle,
  type GridSize,
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms, type OakMaterial } from '@kit/materials';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { hash2f, streamFor } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { excludeBand, scatterRocks } from '@kit/vegetation';
import { makeWater, type WaterHandle } from '@kit/water';
import {
  MEMORYCHAIN_ACTION,
  MEMORYCHAIN_CUE,
  MEMORYCHAIN_END,
  MEMORYCHAIN_L,
  MEMORYCHAIN_POS,
  memoryChainDecoder,
} from '@sim/envs/memorychain';
import type { EnvGeometry, GridGeometry } from '@sim/types';


export const PASSAGE_HALF = (MEMORYCHAIN_L * CELL) / 2;
export const CAVERN_X0 = -PASSAGE_HALF - 6.5;
export const CAVERN_X1 = PASSAGE_HALF + 5.5;
export const CAVERN_HALF_W = 5.7;
export const CAVERN_HEIGHT = 9.2;
export const CAVERN_SKIRT = 1.6;
export const WATER_LEVEL = -0.11;
export const WATER_HALF_Z = 14;
export const POOL_HALF_Z = 4.5;

export const FOG_EXTENT = 100;

export const SUN_DIR: readonly [number, number, number] = [-0.42, 0.66, 0.62];
export const SHADOW_CENTER: readonly [number, number, number] = [-6, 0.4, 0];
export const SHADOW_RADIUS = 13;

export const CAMERA_AZIMUTH = -0.935;


export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function smoothstep01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
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


export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const g = geometry.grid;
  if (g && g.w === MEMORYCHAIN_L && g.h === 1) return g;
  const truth = memoryChainDecoder.geometry(memoryChainDecoder.initStatic(0), 0);
  return truth.grid ?? { w: MEMORYCHAIN_L, h: 1, walls: new Uint8Array(MEMORYCHAIN_L) };
}

export const CUE_CELL = 0;
export const CHOICE_CELL = MEMORYCHAIN_END;


export interface CaveSkin {
  stone: THREE.Color;
  rockLow: THREE.Color;
  rockHigh: THREE.Color;
  mineral: THREE.Color;
  waterShallow: THREE.Color;
  waterDeep: THREE.Color;
  firefly: THREE.Color;
  quartz: THREE.Color;
  rim: THREE.Color;
}

export function caveSkin(palette: Palette, tokens: typeof TOKENS): CaveSkin {
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const firefly = tokens.accents['memory-chain'] ?? palette.accentLight;
  const lift = 0;
  const up = (c: THREE.Color): THREE.Color => (lift > 0 ? c.lerp(new THREE.Color(silver), lift) : c);
  return {
    stone: up(blendTokens(palette.groundB, silver, 0.64)),
    rockLow: up(blendTokens(palette.groundB, silver, 0.38)),
    rockHigh: up(blendTokens(palette.groundA, silver, 0.09)),
    mineral: up(blendTokens(palette.groundB, silver, 0.76)),
    waterShallow: shade(blendTokens(palette.groundA, palette.hemiSky, 0.45), 0.5),
    waterDeep: shade(palette.skyBottom, 0.68),
    firefly: new THREE.Color(firefly),
    quartz: up(blendTokens(palette.groundB, silver, 0.6)),
    rim: blendTokens(palette.hemiSky, silver, 0.55),
  };
}


export function cavernWidthAt(x: number, seed: number): number {
  const n = hash2f(Math.floor(x * 0.5) + 97, 3, seed) - 0.5;
  const m = Math.sin(x * 0.31 + seed * 0.0007) * 0.5;
  return CAVERN_HALF_W * (1 + 0.09 * m + 0.06 * n);
}

export function cavernHeightAt(x: number, seed: number): number {
  const n = hash2f(Math.floor(x * 0.4) + 13, 7, seed) - 0.5;
  const m = Math.sin(x * 0.22 + 1.7) * 0.5 + Math.sin(x * 0.51 - 0.4) * 0.22;
  return CAVERN_HEIGHT * (1 + 0.11 * m + 0.08 * n);
}

export function cavernSurface(
  x: number,
  t: number,
  seed: number,
  out: { z: number; y: number; nz: number; ny: number },
): void {
  const w = cavernWidthAt(x, seed);
  const h = cavernHeightAt(x, seed);
  const phi = Math.PI * t;
  const c = Math.cos(phi);
  const s = Math.sin(phi);
  out.z = -w * c;
  out.y = h * s;
  const nz = h * c;
  const ny = -w * s;
  const len = Math.hypot(nz, ny) || 1;
  out.nz = nz / len;
  out.ny = ny / len;
}

export function makeCavernGeometry(seed: number, skin: CaveSkin): THREE.BufferGeometry {
  const rings = 19;
  const sweeps = 96;
  const rowCount = rings + 2;
  const vertexCount = (sweeps + 1) * rowCount;
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const indices = new Uint32Array(sweeps * (rowCount - 1) * 6);

  const low = skin.rockLow;
  const high = skin.rockHigh;
  const c = new THREE.Color();
  const surf = { z: 0, y: 0, nz: 0, ny: 0 };

  let v = 0;
  for (let i = 0; i <= sweeps; i++) {
    const x = CAVERN_X0 + ((CAVERN_X1 - CAVERN_X0) * i) / sweeps;
    for (let j = 0; j < rowCount; j++) {
      let z: number;
      let y: number;
      if (j === 0 || j === rowCount - 1) {
        const t = j === 0 ? 0 : 1;
        cavernSurface(x, t, seed, surf);
        z = surf.z;
        y = -CAVERN_SKIRT;
      } else {
        const t = (j - 1) / (rings - 1);
        cavernSurface(x, t, seed, surf);
        z = surf.z;
        y = surf.y;
        const n = hash2f(i, j * 31 + 5, seed) - 0.5;
        const n2 = hash2f(i * 7 + 3, j, seed ^ 0x51ed270b) - 0.5;
        const push = (n * 0.62 + n2 * 0.3) * (0.24 + 0.5 * Math.sin(Math.PI * t));
        z += surf.nz * push;
        y += surf.ny * push;
      }
      positions[v * 3] = x;
      positions[v * 3 + 1] = y;
      positions[v * 3 + 2] = z;
      const k = clamp01(y / CAVERN_HEIGHT);
      c.copy(low).lerp(high, smoothstep01(k * 1.15));
      const grain = 1 + (hash2f(i * 13 + 1, j * 5 + 2, seed) - 0.5) * 0.16;
      colors[v * 3] = c.r * grain;
      colors[v * 3 + 1] = c.g * grain;
      colors[v * 3 + 2] = c.b * grain;
      v++;
    }
  }

  let o = 0;
  for (let i = 0; i < sweeps; i++) {
    for (let j = 0; j < rowCount - 1; j++) {
      const a = i * rowCount + j;
      const b = (i + 1) * rowCount + j;
      indices[o++] = a;
      indices[o++] = b;
      indices[o++] = b + 1;
      indices[o++] = a;
      indices[o++] = b + 1;
      indices[o++] = a + 1;
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'cavernShell';
  return g;
}


function spikeGeometry(): THREE.BufferGeometry {
  const radial = 7;
  const g = new THREE.CylinderGeometry(0.02, 0.5, 1, radial, 2);
  g.rotateX(Math.PI);
  g.translate(0, -0.5, 0);
  const pos = g.getAttribute('position');
  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    const t = clamp01(-y);
    const k = 1 - 0.1 * Math.sin(Math.PI * t);
    pos.setX(i, pos.getX(i) * k);
    pos.setZ(i, pos.getZ(i) * k);
    const grade = 0.72 + 0.5 * t * t;
    colors[i * 3] = grade;
    colors[i * 3 + 1] = grade;
    colors[i * 3 + 2] = grade;
  }
  pos.needsUpdate = true;
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.computeVertexNormals();
  g.name = 'stalactiteSpike';
  return g;
}

export interface StalactiteField {
  mesh: THREE.InstancedMesh;
  count: number;
  dispose(): void;
}

export function makeStalactites(
  seed: number,
  skin: CaveSkin,
  palette: Palette,
): StalactiteField {
  const wanted = 54;
  const geometry = spikeGeometry();
  const material = oakMaterial({
    color: blendTokens(skin.rockLow, skin.mineral, 0.22),
    vertexColors: true,
    flatShading: true,
    roughness: 0.86,
    rimColor: skin.rim,
    rimStrength: 0.42,
    emissive: blendTokens(skin.rockLow, skin.mineral, 0.22),
    emissiveIntensity: 0.03,
    toonSteps: 2,
    toonSoftness: 0.34,
    toonFloor: 0.24,
    name: 'stalactiteMaterial',
  });
  const mesh = new THREE.InstancedMesh(geometry, material, wanted);
  mesh.name = 'stalactites';
  mesh.castShadow = true;
  mesh.receiveShadow = false;
  mesh.frustumCulled = false;
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(wanted * 3), 3);

  const rng = streamFor(seed | 0, 'stalactites');
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const axis = new THREE.Vector3();
  const down = new THREE.Vector3(0, -1, 0);
  const color = new THREE.Color();
  const surf = { z: 0, y: 0, nz: 0, ny: 0 };

  for (let i = 0; i < wanted; i++) {
    const x = CAVERN_X0 + 1.5 + rng() * (CAVERN_X1 - CAVERN_X0 - 3);
    const foreground = rng() < 0.06;
    const t = foreground ? 0.6 + rng() * 0.16 : 0.28 + rng() * 0.3;
    cavernSurface(x, t, seed, surf);
    const shortness = rng();
    const len = foreground ? 1.2 + rng() * 1.1 : 0.42 + shortness * shortness * 1.5;
    const girth = (foreground ? 1.15 : 0.85 + rng() * 0.55) * (0.85 + 0.3 * rng());
    axis.set(0, -1, 0).lerp(new THREE.Vector3(0, surf.ny, surf.nz), 0.3).normalize();
    quaternion.setFromUnitVectors(down, axis);
    position.set(x, surf.y + 0.06, surf.z);
    scale.set(girth, len, girth);
    matrix.compose(position, quaternion, scale);
    mesh.setMatrixAt(i, matrix);
    const k = 0.82 + rng() * 0.34;
    color.setRGB(k, k, k);
    mesh.setColorAt(i, color);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();

  return {
    mesh,
    count: wanted,
    dispose() {
      mesh.dispose();
      geometry.dispose();
      material.dispose();
      mesh.removeFromParent();
    },
  };
}


export interface WorldHandle {
  group: THREE.Group;
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  water: WaterHandle;
  skin: CaveSkin;
  fireflies: AmbientParticlesHandle;
  flood: AmbientParticlesHandle;
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
  const palette: Palette = ctx.tokens.palettes.cave;
  const skin = caveSkin(palette, ctx.tokens);
  const group = new THREE.Group();
  group.name = 'fireflyCave';
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const sized: { setPixelRatio(dpr: number): void }[] = [];

  const sky = skyFromPalette(palette, 'cave', {
    topSpread: 0.5,
    bottomSpread: 0.45,
    dither: 0.01,
    radius: 320,
  });
  group.add(sky.mesh);
  owned.push(sky);
  timed.push({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: SUN_DIR,
    pointCount: 4,
    pointDistance: 15,
    pointDecay: 1.45,
    sunDistanceScale: 2.4,
  });
  group.add(rig.group);
  owned.push(rig);
  rig.setShadowFocus({ x: SHADOW_CENTER[0], y: SHADOW_CENTER[1], z: SHADOW_CENTER[2] }, SHADOW_RADIUS);

  applyFog(scene, palette, FOG_EXTENT);

  const cavernGeometry = makeCavernGeometry(ctx.seed, skin);
  const cavernMaterial: OakMaterial = oakMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.97,
    rimColor: skin.rim,
    rimStrength: 0.4,
    emissive: blendTokens(skin.rockLow, skin.rockHigh, 0.72),
    emissiveIntensity: 0.03,
    toonSteps: 3,
    toonSoftness: 0.32,
    toonFloor: 0.36,
    side: THREE.FrontSide,
    name: 'cavernMaterial',
  });
  const cavern = new THREE.Mesh(cavernGeometry, cavernMaterial);
  cavern.name = 'cavernShell';
  cavern.receiveShadow = true;
  cavern.castShadow = false;
  group.add(cavern);
  owned.push({
    dispose() {
      cavernGeometry.dispose();
      cavernMaterial.dispose();
      cavern.removeFromParent();
    },
  });

  const water = makeWater({
    size: [(CAVERN_X1 - CAVERN_X0) + 34, WATER_HALF_Z * 2],
    palette,
    segments: 96,
    shallow: skin.waterShallow,
    deep: skin.waterDeep,
    sky: blendTokens(palette.hemiSky, skin.rim, 0.35),
    caustic: blendTokens(palette.hemiSky, skin.firefly, 0.25),
    amplitude: 0.0125,
    waveScale: 0.65,
    speed: 0.22,
    fresnel: 0.24,
    caustics: 0.14,
    depthFalloff: 10,
    opacity: 0.97,
    sunDir: rig.sunDirection,
    level: WATER_LEVEL,
    name: 'cavePools',
  });
  water.mesh.position.x = (CAVERN_X0 + CAVERN_X1) * 0.5;
  water.mesh.position.z = 4;
  group.add(water.mesh);
  owned.push(water);
  timed.push(water);

  const field = buildCellField(
    grid,
    {
      floorColor: skin.stone,
      wallColor: skin.stone,
      floorTint: (x) => {
        const k = x / Math.max(1, MEMORYCHAIN_L - 1);
        return blendTokens(0xffffff, 0x9aa6bb, 0.28 * k);
      },
      floorInset: 0.09,
      floorThickness: 0.34,
      jitter: 0.14,
      rimColor: skin.rim,
      rimStrength: 0.6,
      seed: ctx.seed,
      name: 'walkwayStones',
    },
  );
  {
    const floor = oakUniforms(field.floorMaterial);
    if (floor) floor.uniforms.uToonFloor.value = 0.34;
    field.floorMaterial.emissive.copy(skin.stone);
    field.floorMaterial.emissiveIntensity = 0.03;
  }
  group.add(field.group);
  owned.push(field);
  timed.push({ setTime: (t) => field.setTime(t) });

  const extent = gridExtent(grid);
  const banks = scatterRocks({
    count: 82,
    area: { x: 0, z: 0.3, halfX: extent.width * 0.5 + 4.5, halfZ: POOL_HALF_Z },
    seed: ctx.seed ^ 0x2f19,
    exclude: excludeBand(0, CELL * 0.5, 0.55),
    heightAt: () => WATER_LEVEL - 0.24,
    color: skin.mineral,
    scale: [0.42, 1.55],
    stretch: [1.3, 3.6],
    hueJitter: 0.22,
    rimColor: skin.rim,
    rimStrength: 0.8,
    castShadow: false,
    receiveShadow: false,
    name: 'stalagmites',
  });
  group.add(banks.group);
  owned.push(banks);
  timed.push(banks);

  const stalactites = makeStalactites(ctx.seed, skin, palette);
  group.add(stalactites.mesh);
  owned.push(stalactites);

  const fireflies = makeAmbientParticles({
    preset: 'fireflies',
    count: 640,
    bounds: { x: 0, y: 1.4, z: -0.1, halfX: PASSAGE_HALF + 3, halfY: 1.7, halfZ: 3.5 },
    seed: ctx.seed ^ 0x11e5,
    color: skin.firefly,
    size: 2.35,
    intensity: 2.9,
    opacity: 0.95,
    rise: 0.05,
    swirl: 0.5,
    name: 'passageFireflies',
  });
  group.add(fireflies.points);
  owned.push(fireflies);
  timed.push(fireflies);
  sized.push(fireflies);

  const vaultDust = makeAmbientParticles({
    preset: 'dust',
    count: 420,
    bounds: { x: 0, y: 4.6, z: -0.6, halfX: PASSAGE_HALF + 4, halfY: 3.1, halfZ: 3.4 },
    seed: ctx.seed ^ 0x7c03,
    color: blendTokens(palette.hemiSky, skin.mineral, 0.4),
    size: 1.5,
    intensity: 0.85,
    opacity: 0.4,
    rise: 0.015,
    swirl: 0.24,
    name: 'vaultDust',
  });
  group.add(vaultDust.points);
  owned.push(vaultDust);
  timed.push(vaultDust);
  sized.push(vaultDust);

  const flood = makeAmbientParticles({
    preset: 'fireflies',
    count: 520,
    bounds: { x: 0, y: 1.15, z: 0, halfX: 2.4, halfY: 1.3, halfZ: 1.5 },
    seed: ctx.seed ^ 0x63b1,
    color: skin.firefly,
    size: 3.4,
    intensity: 3.4,
    opacity: 1,
    rise: 0.16,
    swirl: 0.62,
    name: 'choiceFlood',
  });
  flood.setDensity(0);
  group.add(flood.points);
  owned.push(flood);
  timed.push(flood);
  sized.push(flood);

  return {
    group,
    sky,
    rig,
    field,
    water,
    skin,
    fireflies,
    flood,
    setTime(t) {
      for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
      oakUniforms(cavernMaterial)?.setTime(t);
    },
    setPixelRatio(dpr) {
      for (let i = 0; i < sized.length; i++) sized[i]?.setPixelRatio(dpr);
    },
    dispose() {
      for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
      owned.length = 0;
      timed.length = 0;
      sized.length = 0;
      group.removeFromParent();
      group.clear();
    },
  };
}


export function readPos(state: { pose: Float32Array }): number {
  const v = Math.round(state.pose[MEMORYCHAIN_POS] ?? 0);
  return v < 0 ? 0 : v > MEMORYCHAIN_END ? MEMORYCHAIN_END : v;
}

export function readCue(state: { pose: Float32Array }): 0 | 1 {
  return Math.round(state.pose[MEMORYCHAIN_CUE] ?? 0) === 1 ? 1 : 0;
}

export function readChoice(state: { pose: Float32Array }): 0 | 1 {
  return (Math.trunc(state.pose[MEMORYCHAIN_ACTION] ?? 0) & 1) === 1 ? 1 : 0;
}

export function cellX(cell: number, grid: GridSize, out: Vec3Tuple): number {
  cellToWorld(cell, 0, grid, out);
  return out[0];
}
