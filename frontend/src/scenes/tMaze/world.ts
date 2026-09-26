import * as THREE from 'three/webgpu';
import { abs, cameraPosition, clamp, dot, float, length, min, mix, positionWorld, sin, smoothstep, vec2, vec3 } from 'three/tsl';
import { CELL, buildCellField } from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial } from '@kit/materials';
import { makeAmbientParticles } from '@kit/particles';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import type { SceneContext } from '@kit/types';
import { makeContinuousSurface, type ViewCoverage } from '@kit/view';
import { extendWater, makeWater } from '@kit/water';
import { makeWorld, type WorldCore } from '@kit/world';
import type { EnvGeometry, GridGeometry } from '@sim/types';

type N<T extends string> = THREE.Node<T>;

export const WATER_LEVEL = -0.22;
export const HERO_DISTANCE = 33.5;
const HERO_FOG = 0.0085;
const VAULT_TILE = 24;
const VAULT_TILES = 24;
const VAULT_RISE = 10.6;
const VAULT_SPAN = 12;

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w < 2 || grid.h !== 1 || grid.goal?.x !== grid.w - 1) throw new Error('T-maze corridor grid is missing or invalid');
  return grid;
}

export interface CaveSkin {
  stone: THREE.Color;
  rim: THREE.Color;
}

export interface WorldHandle extends WorldCore {
  rig: LightRig;
  skin: CaveSkin;
  setView(coverage: ViewCoverage): void;
}

const lakeScale = (half: number) => Math.min(1, (half + 0.6) / 8.1);
const rockHalf = (half: number) => Math.max(half, 7.5);

function lakeBed(half: number) {
  const scale = lakeScale(half), rock = rockHalf(half);
  return (p: N<'vec2'>): N<'float'> => {
    const width = float(8.2).add(sin(p.x.mul(0.14)).mul(1.1)).add(sin(p.x.mul(0.34)).mul(0.6)).mul(scale);
    const shore = p.y.add(sin(p.x.mul(0.1)).mul(0.8));
    const lake = length(vec2(p.x.div(half + 0.6), shore.div(width)));
    const basin = length(vec2(p.x.div(rock + 0.6), shore.div(width.div(scale))));
    const passage = abs(p.y.sub(sin(p.x.mul(0.12)).mul(0.45))).div(1.8);
    const bed = float(-1.1)
      .add(smoothstep(0.7, 0.96, min(lake, passage)).mul(1.15))
      .add(smoothstep(1.05, 2.4, basin).mul(smoothstep(1, 1.7, passage)).mul(sin(p.x.mul(0.11)).mul(sin(p.y.mul(0.07))).add(2.2)));
    const across = abs(p.x), along = abs(p.y);
    const wall = smoothstep(rock + 1, rock + 6, across).mul(smoothstep(8, 12, along).oneMinus()).mul(smoothstep(1.2, 2.8, along)).mul(6);
    const road = smoothstep(half + 0.6, half + 1.5, across).oneMinus().mul(smoothstep(0.4, 0.85, along).oneMinus());
    return mix(bed.add(wall), clamp(bed.add(wall), -0.4, -0.04), road);
  };
}

export function groundAt(half: number, x: number, z: number): number {
  const scale = lakeScale(half), rock = rockHalf(half), step = (a: number, b: number, v: number) => THREE.MathUtils.smoothstep(v, a, b);
  const width = (8.2 + Math.sin(x * 0.14) * 1.1 + Math.sin(x * 0.34) * 0.6) * scale;
  const shore = z + Math.sin(x * 0.1) * 0.8;
  const lake = Math.hypot(x / (half + 0.6), shore / width), basin = Math.hypot(x / (rock + 0.6), shore * scale / width);
  const passage = Math.abs(z - Math.sin(x * 0.12) * 0.45) / 1.8;
  const bed = -1.1 + step(0.7, 0.96, Math.min(lake, passage)) * 1.15 + step(1.05, 2.4, basin) * step(1, 1.7, passage) * (2.2 + Math.sin(x * 0.11) * Math.sin(z * 0.07));
  const across = Math.abs(x), along = Math.abs(z);
  const rough = bed + step(rock + 1, rock + 6, across) * (1 - step(8, 12, along)) * step(1.2, 2.8, along) * 6;
  const road = (1 - step(half + 0.6, half + 1.5, across)) * (1 - step(0.4, 0.85, along));
  return rough + (Math.min(-0.04, Math.max(-0.4, rough)) - rough) * road;
}

function vaultGeometry(): THREE.BufferGeometry {
  const positions: number[] = [], colors: number[] = [], indices: number[] = [];
  const stone = new THREE.Color(), low = new THREE.Color(0x97b4ad), high = new THREE.Color(0x647e86);
  for (let ring = 0; ring <= 24; ring++) {
    const x = ring - 12;
    for (let arc = 0; arc <= 32; arc++) {
      const angle = arc / 32 * Math.PI, rise = Math.sin(angle), phase = x * Math.PI / 12;
      const fold = Math.cos(phase * 2 + Math.sin(angle * 3) * 0.25) * 0.65 + Math.cos(phase * 5 + angle * 0.4) * 0.25;
      const width = VAULT_SPAN + fold * (0.5 + Math.pow(rise, 0.65) * 1.7);
      positions.push(x, -0.7 + rise * (VAULT_RISE + Math.cos(phase) * 0.7 + fold), -Math.cos(angle) * width);
      stone.copy(low).lerp(high, rise * 0.75);
      colors.push(stone.r, stone.g, stone.b);
      if (ring < 24 && arc < 32) {
        const a = ring * 33 + arc, b = a + 33;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry, distance: number): WorldHandle {
  const w = makeWorld('limestoneCavernLake');
  const palette = ctx.tokens.palettes.cave;
  const skin: CaveSkin = { stone: new THREE.Color(0xa8b7ae), rim: new THREE.Color(0xadcfd0) };
  const half = (grid.w - 1) * CELL / 2;
  const sky = w.time(w.own(skyFromPalette(palette, 'cave', { topSpread: 0.8, bottomSpread: 0.5 })));
  sky.setColors(0x192e3b, 0x2e5b64, 0x112b3a);
  w.add(sky.mesh);
  const rig = w.own(makeLightRig(palette, { sunDir: [-0.2, 0.85, 0.42], pointCount: 1, pointDistance: 12 }));
  rig.hemi.color.set(0xa6c7c8);
  rig.hemi.intensity = 1.6;
  rig.hemi.groundColor.set(0x466c73);
  rig.sun.color.set(0xd5e3ca);
  rig.sun.intensity = 1.45;
  rig.setShadowFocus({ x: 0, y: 0, z: 0 }, Math.max(grid.w * CELL * 0.6, 9));
  w.add(rig.group);
  scene.fog = new THREE.FogExp2(0x274853, HERO_FOG * HERO_DISTANCE / distance);
  const bed = lakeBed(half);
  const water = w.time(w.own(makeWater({
    size: 2, segments: 2, palette, level: WATER_LEVEL,
    shallow: 0x386576, deep: 0x102a43, sky: 0x6997a7, caustic: 0x41606a,
    amplitude: 0.012, waveScale: 0.6, speed: 0.2, fresnel: 0.26, caustics: 0.025,
    depthFalloff: 0.85, opacity: 0.96, sunDir: rig.sunDirection, name: 'cavernLake',
  })));
  water.setColors(0x386576, 0x102a43);
  w.add(water.mesh);
  const floor = w.time(w.own(makeContinuousSurface({ name: 'flowstoneLakeMargins', colorA: skin.stone, colorB: 0x66828a, segments: 160, height: bed })));
  w.add(floor.mesh);
  const field = w.own(buildCellField(grid, {
    floorColor: skin.stone, wallColor: skin.stone, floorInset: 0.025, floorThickness: 0.42, jitter: 0.015,
    rimColor: 0xa7ddd5, rimStrength: 0.25, seed: ctx.seed, name: 'luminousCauseway',
  }));
  field.floorMaterial.emissive.set(0x5a958e);
  field.floorMaterial.emissiveIntensity = 0.18;
  w.add(field.group);
  w.time(field);
  const vaultMaterial = w.own(oakMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.97, rimColor: skin.rim, rimStrength: 0.18, emissive: 0x283f48, emissiveIntensity: 0.04, toonSteps: 3, toonSoftness: 0.32, toonFloor: 0.36, side: THREE.FrontSide, name: 'limestoneVault' }));
  vaultMaterial.maskNode = dot(cameraPosition.sub(positionWorld), vec3(0, positionWorld.y.add(0.7).div(VAULT_RISE * VAULT_RISE), positionWorld.z.div(VAULT_SPAN * VAULT_SPAN))).lessThanEqual(0);
  const vault = w.own(instancedMesh(w.own(vaultGeometry()), vaultMaterial, VAULT_TILES));
  vault.name = 'continuousLimestoneVault';
  for (let i = 0; i < VAULT_TILES; i++) vault.setMatrixAt(i, new THREE.Matrix4().makeTranslation((i + 0.5 - VAULT_TILES / 2) * VAULT_TILE, 0, 0));
  vault.computeBoundingSphere();
  w.add(vault);
  const spread = Math.max(28, half + 8);
  const motes = w.time(w.own(makeAmbientParticles({
    preset: 'fireflies', count: Math.round(220 * spread / 28), bounds: { y: 2, halfX: spread, halfY: 3, halfZ: 8 },
    seed: ctx.seed, color: 0xaedbba, size: 1.7, intensity: 1.8, opacity: 0.75, rise: 0.02, name: 'caveMotes',
  })));
  w.add(motes.points);
  return w.seal<WorldHandle>({
    rig, skin,
    setView(coverage) {
      extendWater(water, coverage, 2, 2, (p) => float(WATER_LEVEL).sub(bed(p)));
      floor.setView(coverage);
      sky.fit(coverage.far);
    },
  });
}
