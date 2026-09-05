import * as THREE from 'three';
import { CELL, cellToWorld, type GridSize, type Vec3Tuple } from '@kit/grid';
import { makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, oakUniforms, type OakMaterial } from '@kit/materials';
import { blendTokens, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { broadleafTreeGeometry, coniferTreeGeometry, flowerGeometry, grassClumpGeometry } from '@kit/vegetation';
import { makeWater } from '@kit/water';
import { makeWorld, type WorldCore } from '@kit/world';
import { CHAIN_L } from '@sim/envs/chain';
import type { EnvGeometry, GridGeometry } from '@sim/types';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '../jelly-bean-world/view';
import { extendWater } from '../half-cheetah/world';

export const WATER_LEVEL = -0.17;

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
  const grid = geometry.grid;
  if (!grid || grid.w !== CHAIN_L || grid.h !== 1) throw new Error('Chain grid is missing or invalid');
  return grid;
}

export function stoneX(index: number, grid: GridSize): number {
  const out: Vec3Tuple = [0, 0, 0];
  return cellToWorld(index, 0, grid, out)[0];
}

export interface CreekSkin {
  stoneDry: THREE.Color;
  stoneWet: THREE.Color;
  stoneGlow: THREE.Color;
  reed: THREE.Color;
  meadow: THREE.Color;
  flower: THREE.Color;
  bark: THREE.Color;
  moss: THREE.Color;
  blossom: THREE.Color;
  waterShallow: THREE.Color;
  waterDeep: THREE.Color;
  waterSky: THREE.Color;
  waterCaustic: THREE.Color;
  motes: THREE.Color;
}

export function creekSkin(palette: Palette, tokens: typeof TOKENS): CreekSkin {
  const blossom = tokens.accents['chain'] ?? tokens.celebrationGold;
  const leaf = tokens.accents['foliage'] ?? palette.groundA;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const petalPink = tokens.accents['acrobot'] ?? blossom;
  const teal = tokens.accents['deep-sea'] ?? palette.hemiSky;
  const white = palette.sun;

  return {
    stoneDry: shade(blendTokens(palette.groundB, silver, 0.28), 0.46),
    stoneWet: shade(blendTokens(silver, palette.hemiGround, 0.52), 0.64),
    stoneGlow: blendTokens(blossom, tokens.celebrationGold, 0.42),
    reed: shade(blendTokens(leaf, palette.groundA, 0.34), 0.44),
    meadow: shade(blendTokens(palette.groundA, leaf, 0.3), 0.4),
    flower: shade(blendTokens(petalPink, white, 0.2), 0.3),
    bark: shade(blendTokens(palette.hemiGround, TOKENS.palettes.dusk.groundA, 0.4), 0.5),
    moss: shade(blendTokens(leaf, palette.groundA, 0.5), 0.52),
    blossom: shade(blendTokens(blossom, white, 0.1), 0.2),
    waterShallow: shade(blendTokens(blendTokens(teal, palette.groundA, 0.62), palette.hemiGround, 0.28), 0.44),
    waterDeep: shade(blendTokens(blendTokens(teal, palette.groundA, 0.5), palette.skyTop, 0.34), 0.7),
    waterSky: shade(palette.skyTop, 0.46),
    waterCaustic: shade(blendTokens(palette.sun, teal, 0.34), 0.46),
    motes: blendTokens(blossom, white, 0.42),
  };
}

export function makeCreekHeight(_seed: number): (x: number, z: number) => number {
  return (x, z) => {
    const shore = Math.abs(x - Math.sin(z * 0.018) * 10 - Math.sin(z * 0.043) * 2) + 1.8 * Math.exp(-z * z * 0.03);
    const bank = smoothstep01((shore - 7.8) / 3.2);
    return -1.05 + bank * (2.3 + Math.sin(x * 0.025) * Math.sin(z * 0.033) * 0.55 + Math.sin(x * 0.013 + z * 0.017) * 0.35);
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
export const CAP_HEIGHT = 1.1;
export const CAP_SCALE_MIN = 0.9;
export const CAP_SCALE_MAX = 1.08;

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
  seed: number,
): StoneCapsHandle {
  const sides = 7;
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
    name: 'chainStoneCapMaterial',
  });

  const count = grid.w;
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count));
  mesh.name = 'chainStoneCaps';
  mesh.castShadow = true;
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

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  terrain: { sampleHeight(x: number, z: number): number };
  caps: StoneCapsHandle;
  petalMotes: AmbientParticlesHandle;
  setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette = ctx.tokens.palettes.afternoon;
  const skin = creekSkin(palette, ctx.tokens);
  const w = makeWorld('riverValley');
  const sky = w.time(w.own(skyFromPalette(palette, 'afternoon', { sunDir: [-0.46, 0.62, -0.64], sunGlowPower: 24, topSpread: 0.75 })));
  w.add(sky.mesh);
  const rig = w.own(makeLightRig(palette, { sunDir: sky.sunDirection, pointCount: 3, pointDistance: 8 }));
  rig.setShadowFocus({ x: 0, y: 0.5, z: 0 }, 15);
  w.add(rig.group);
  const fog = new THREE.FogExp2(palette.fog, 0.005);
  scene.fog = fog;
  const heightAt = makeCreekHeight(ctx.seed);
  const groundHeight = 'float shore = abs(p.x-sin(p.y*.018)*10.-sin(p.y*.043)*2.) + 1.8*exp(-p.y*p.y*.03); float bank = smoothstep(7.8,11.,shore); return -1.05 + bank*(2.3 + sin(p.x*.025)*sin(p.y*.033)*.55 + sin(p.x*.013+p.y*.017)*.35);';
  const floor = w.time(w.own(makeContinuousSurface({
    name: 'floweringValley', colorA: 0x7b965f, colorB: 0x385e4d, segments: 128,
    height: groundHeight,
    detail: 'color = mix(color,vec3(.58,.59,.3),.18*landNoise(p.xz*.08));',
  })));
  w.add(floor.mesh);
  const water = w.time(w.own(makeWater({
    size: 2, segments: 2, palette, level: WATER_LEVEL,
    shallow: skin.waterShallow, deep: skin.waterDeep, sky: skin.waterSky, caustic: skin.waterCaustic,
    amplitude: 0.018, waveScale: 1.15, speed: 0.6, fresnel: 0.24, caustics: 0.07,
    depthFalloff: 1.1, opacity: 0.92, sunDir: rig.sunDirection, name: 'creekWater',
  })));
  water.setColors(skin.waterShallow, skin.waterDeep);
  w.add(water.mesh);
  water.material.fragmentShader = `float creekBed(vec2 p) {${groundHeight}}\n${water.material.fragmentShader}`;
  const caps = w.own(buildStoneCaps(grid, skin, palette, ctx.seed));
  w.add(caps.mesh);
  const vegetationMaterials: OakMaterial[] = [];
  const trees = [
    { geometry: broadleafTreeGeometry(0), trunkVertices: 48, scale: 1.4, leaf: 0x8fad79, name: 'riverBroadleaves' },
    { geometry: coniferTreeGeometry(0), trunkVertices: 60, scale: 1.6, leaf: 0x6c9780, name: 'riverConifers' },
  ].map((spec, index) => {
    const geometry = w.own(spec.geometry);
    const colors = geometry.getAttribute('color');
    const bark = new THREE.Color(0x87644c);
    const leaf = new THREE.Color(spec.leaf);
    for (let i = 0; i < colors.count; i++) {
      const tint = i < spec.trunkVertices ? bark : leaf;
      const shade = colors.getX(i);
      colors.setXYZ(i, tint.r * shade, tint.g * shade, tint.b * shade);
    }
    const material = w.own(oakMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.85, flatShading: true, wind: { amplitude: 0.025, frequency: 0.85, height: 2 }, rimStrength: 0.2 }));
    vegetationMaterials.push(material);
    const bank = w.own(makeSceneryBank({
      name: spec.name, seed: ctx.seed + index * 731, tileSize: 32, perTile: 16,
      geometry, material, colorA: 0xd6dfcf, colorB: 0xffffff,
      fill(tx, tz, rng, write) {
        for (let i = 0; i < 16; i++) {
          const x = (tx + rng()) * 32;
          const z = (tz + rng()) * 32;
          const y = heightAt(x, z);
          if (y < 0.6 || x * x / 289 + z * z / 196 < 1) continue;
          write(x, y, z, rng() * 6.28, spec.scale * (0.8 + rng() * 0.6), 0.9 + rng() * 0.4, rng());
        }
      },
    }));
    w.add(bank.mesh);
    return bank;
  });
  const groundCover = [
    { geometry: grassClumpGeometry(3), tile: 6, count: 36, scale: 0.9, colorA: skin.reed, colorB: skin.meadow, name: 'riverReeds' },
    { geometry: flowerGeometry(5), tile: 8, count: 18, scale: 0.8, colorA: skin.flower, colorB: 0xe6dbae, name: 'riverbankFlowers' },
  ].map((spec, index) => {
    const geometry = w.own(spec.geometry);
    const material = w.own(oakMaterial({ color: 0xffffff, vertexColors: true, side: THREE.DoubleSide, roughness: 0.8, wind: { amplitude: 0.045, frequency: 1.3, height: 0.5 }, rimStrength: 0.24 }));
    vegetationMaterials.push(material);
    const bank = w.own(makeSceneryBank({
      name: spec.name, seed: ctx.seed ^ (0x725a + index), tileSize: spec.tile, perTile: spec.count,
      geometry, material, colorA: spec.colorA, colorB: spec.colorB,
      fill(tx, tz, rng, write) {
        for (let i = 0; i < spec.count; i++) {
          const x = (tx + rng()) * spec.tile;
          const z = (tz + rng()) * spec.tile;
          const y = heightAt(x, z);
          if (y < (index === 0 ? -0.3 : 0) || (Math.abs(x) < 12 && Math.abs(z) < 2)) continue;
          write(x, y, z, rng() * 6.28, spec.scale * (0.65 + rng() * 0.65), index === 0 && Math.abs(x) < 11 ? 1.8 : 1, rng());
        }
      },
    }));
    w.add(bank.mesh);
    return bank;
  });
  const petalMotes = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'pollen', count: 120, bounds: { x: stoneX(grid.w - 1, grid), y: 2, halfX: 3.2, halfY: 2, halfZ: 3.2 },
    seed: ctx.seed ^ 0x1aa9, color: skin.motes, size: 1.8, intensity: 1.2, rise: -0.08, name: 'blossomPetals',
  }))));
  petalMotes.setDensity(0);
  w.add(petalMotes.points);
  return w.seal({
    sky, rig, terrain: { sampleHeight: heightAt }, caps, petalMotes,
    setView(coverage: ViewCoverage, camera: THREE.PerspectiveCamera) {
      floor.setView(coverage);
      extendWater(water.mesh, coverage, 2, 2, '-0.17 - creekBed(vWorld.xz)');
      for (const bank of trees) bank.setView(coverage);
      for (const bank of groundCover) bank.setView(coverage);
      fog.density = 2.8 / coverage.far;
      sky.mesh.position.copy(camera.position);
      sky.mesh.scale.setScalar(coverage.far * 0.85);
    },
  }, { time(t) { for (const material of vegetationMaterials) oakUniforms(material)?.setTime(t); } });
}
