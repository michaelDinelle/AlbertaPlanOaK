import { sculptureMaterial } from '../key-corridor/world';
import * as THREE from 'three';
import { CELL, buildCellField, cellToWorldVec, isWall, type CellFieldHandle } from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, type OakMaterial } from '@kit/materials';
import { blendTokens, shade } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { excludeAny, rockGeometry, scatterFlowers, scatterGrass } from '@kit/vegetation';
import { excludeCellCenters, excludeWallCells } from '@kit/grid';
import { makeWorld, type WorldCore } from '@kit/world';
import { PROCMAZE_H, PROCMAZE_W } from '@sim/envs/proceduralmaze';
import type { EnvGeometry, GridGeometry } from '@sim/types';
import { ivyLeafGeometry, makeLandscape } from '../key-corridor/world';
import { makeSceneryBank, type ViewCoverage } from '../jelly-bean-world/view';

export const SUN_DIR: readonly [number, number, number] = [-0.36, 0.874, -0.325];

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w !== PROCMAZE_W || grid.h !== PROCMAZE_H || grid.walls.length !== grid.w * grid.h || !grid.carvedPath?.length) {
    throw new Error('Procedural Maze requires complete recorded grid geometry');
  }
  return grid;
}

export interface RuinSkin {
  stoneLight: THREE.Color;
  stoneDark: THREE.Color;
  moss: THREE.Color;
  mossDeep: THREE.Color;
  flag: THREE.Color;
  flagDamp: THREE.Color;
  soil: THREE.Color;
  grass: THREE.Color;
  bloom: THREE.Color;
  canopy: THREE.Color;
  sunWarm: THREE.Color;
  bird: THREE.Color;
}

export function ruinSkin(palette: Palette, tokens: typeof TOKENS): RuinSkin {
  const accent = tokens.accents['procedural-maze'] ?? tokens.celebrationGold;
  const tan = tokens.palettes.dusk.groundA;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  return {
    stoneLight: shade(blendTokens(silver, tan, 0.86), 0.06),
    stoneDark: shade(blendTokens(tan, palette.hemiGround, 0.22), 0.4),
    moss: shade(blendTokens(accent, palette.groundA, 0.62), 0.24),
    mossDeep: shade(blendTokens(palette.groundA, accent, 0.12), 0.48),
    flag: shade(blendTokens(silver, tan, 0.72), 0.42),
    flagDamp: shade(blendTokens(palette.groundA, tan, 0.4), 0.46),
    soil: shade(blendTokens(palette.hemiGround, tan, 0.55), 0.44),
    grass: shade(blendTokens(palette.groundA, accent, 0.45), 0.12),
    bloom: shade(blendTokens(palette.groundB, tokens.celebrationGold, 0.4), 0.12),
    canopy: shade(blendTokens(palette.groundA, palette.groundB, 0.3), 0.24),
    sunWarm: blendTokens(palette.sun, tokens.celebrationGold, 0.42),
    bird: shade(blendTokens(palette.hemiGround, tan, 0.3), 0.72),
  };
}

export function floorTintFor(grid: GridGeometry, skin: RuinSkin, x: number, y: number, seed: number, out = new THREE.Color()): THREE.Color {
  const shade = (isWall(grid, x - 1, y) ? 1 : 0) + (isWall(grid, x + 1, y) ? 1 : 0)
    + (isWall(grid, x, y - 1) ? 1 : 0) + (isWall(grid, x, y + 1) ? 1 : 0);
  return out.copy(isWall(grid, x, y) ? skin.soil : skin.flag)
    .lerp(skin.flagDamp, shade * 0.12)
    .multiplyScalar(0.94 + hash2f(x, y, seed) * 0.12);
}

function ruinWallGeometry(skin: RuinSkin, variant: number): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const base = new THREE.BoxGeometry(0.96, 0.5, 0.96);
  base.translate(0, 0.25, 0);
  parts.push({ geometry: base, color: skin.stoneDark });
  const upperHeight = 0.12 + (variant % 3) * 0.11;
  const upper = new THREE.BoxGeometry(0.78, upperHeight, 0.78);
  upper.rotateY((variant - 2) * 0.025);
  upper.translate(0, 0.5 + upperHeight / 2, 0);
  parts.push({ geometry: upper, color: skin.stoneLight });
  if (variant !== 1) {
    const cap = new THREE.BoxGeometry(0.58, 0.12, 0.58);
    cap.rotateY(variant * 0.11);
    cap.rotateZ((variant - 2) * 0.025);
    cap.translate(0.025, 0.53 + upperHeight, -0.025);
    parts.push({ geometry: cap, color: skin.stoneLight });
  }
  if (variant === 2) {
    const cushion = new THREE.IcosahedronGeometry(0.26, 1);
    cushion.scale(1, 0.38, 0.9);
    cushion.translate(-0.05, 0.65 + upperHeight, 0.03);
    parts.push({ geometry: cushion, color: skin.moss });
  }
  for (let i = 0; i < 5 + variant; i++) {
    const leaf = ivyLeafGeometry();
    leaf.scale(0.7, 0.7, 0.7);
    leaf.rotateY((i % 2) * 0.3);
    leaf.translate(Math.sin(i * 2.1) * 0.14, 0.74 - i * 0.08, -0.5);
    parts.push({ geometry: leaf, color: i % 2 ? skin.moss : skin.mossDeep });
  }
  return mergeColoredParts(parts, `weatheredRuin${variant}`);
}

export function makeRuinTower(skin: RuinSkin, height: number, lean: number): THREE.BufferGeometry {
  const radial = 14;
  const parts: ColoredPart[] = [];

  const plinth = new THREE.BoxGeometry(CELL - 0.04, 0.26, CELL - 0.04);
  plinth.translate(0, 0.13, 0);
  parts.push({ geometry: plinth, color: skin.stoneDark });

  const drumH = (height - 0.26) / 3;
  for (let i = 0; i < 3; i++) {
    const r0 = 0.44 - i * 0.045;
    const r1 = 0.47 - i * 0.045;
    const drum = new THREE.CylinderGeometry(r0, r1, drumH, radial, 1);
    drum.translate(0, drumH * 0.5, 0);
    if (i === 2) {
      drum.scale(1, 0.62, 1);
      drum.rotateZ(lean);
    }
    drum.translate(0, 0.26 + i * drumH, 0);
    parts.push({ geometry: drum, color: i === 1 ? skin.stoneLight : skin.stoneDark });
  }

  const band = new THREE.CylinderGeometry(0.5, 0.5, 0.075, radial, 1);
  band.translate(0, 0.26 + drumH, 0);
  parts.push({ geometry: band, color: skin.stoneLight });

  return mergeColoredParts(parts, 'ruinTower');
}

export function makeRuinArch(skin: RuinSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const JAMB_H = 0.7;
  const HALF = 0.47;

  for (const sz of [-1, 1]) {
    const jamb = new THREE.BoxGeometry(0.3, JAMB_H, 0.16);
    jamb.translate(0, JAMB_H * 0.5, sz * HALF);
    parts.push({ geometry: jamb, color: skin.stoneDark });
    const corbel = new THREE.BoxGeometry(0.34, 0.1, 0.24);
    corbel.translate(0, JAMB_H + 0.05, sz * (HALF - 0.02));
    parts.push({ geometry: corbel, color: skin.stoneLight });
  }

  const spans: readonly (readonly [number, number, number])[] = [
    [-0.3, 0.855, 0],
    [0.0, 0.9, 0.16],
    [0.3, 0.855, 0],
  ];
  for (let i = 0; i < spans.length; i++) {
    const s = spans[i] as readonly [number, number, number];
    const voussoir = new THREE.BoxGeometry(0.2, 0.15, 0.34);
    voussoir.rotateX(s[2]);
    voussoir.translate(0, s[1], s[0]);
    parts.push({ geometry: voussoir, color: i === 1 ? skin.stoneLight : skin.stoneDark });
  }

  const cushion = new THREE.IcosahedronGeometry(0.16, 1);
  cushion.scale(1.3, 0.4, 1.1);
  cushion.translate(0.02, 0.985, -0.14);
  parts.push({ geometry: cushion, color: skin.moss });

  {
    const fallen = new THREE.BoxGeometry(0.22, 0.13, 0.3);
    fallen.rotateY(0.5);
    fallen.rotateZ(0.24);
    fallen.translate(0.3, 0.07, -0.2);
    parts.push({ geometry: fallen, color: skin.stoneDark });
  }

  return mergeColoredParts(parts, 'ruinArch');
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  setGoalPosition(x: number, y: number, z: number): void;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette = ctx.tokens.palettes.noon;
  const skin = ruinSkin(palette, ctx.tokens);
  const w = makeWorld('overgrownRuins');
  const sky = w.time(w.own(skyFromPalette(palette, 'noon', { sunDir: SUN_DIR, sunGlow: 0.24, topSpread: 0.7 })));
  w.add(sky.mesh);
  const rig = w.own(makeLightRig(palette, { sunDir: sky.sunDirection, pointCount: 3, pointDistance: 7, sunDistanceScale: 2.2 }));
  w.add(rig.group);
  rig.setShadowFocus({ x: 0, y: 0.4, z: 0 }, 12);
  const fog = applyFog(scene, palette, 520);
  const halfX = grid.w * CELL / 2;
  const halfZ = grid.h * CELL / 2;
  const landscape = makeLandscape(w, ctx, {
    name: 'ruinWoodland', halfX, halfZ,
    ground: blendTokens(skin.soil, skin.flagDamp, 0.65),
    grass: skin.flagDamp,
    trees: [skin.canopy],
    amplitude: 5.5,
  });
  const field = w.time(w.own(buildCellField(grid, {
    floorColor: palette.sun,
    wallColor: palette.sun,
    floorUnderWalls: true,
    floorTint: (x, y) => floorTintFor(grid, skin, x, y, ctx.seed),
    wallTint: () => skin.stoneDark,
    wallHeight: 0,
    floorInset: 0.045,
    floorThickness: 0.14,
    jitter: 0,
    rimColor: palette.skyHorizon,
    seed: ctx.seed,
    name: 'ruinCells',
  })));
  w.add(field.group);
  for (const mesh of field.wallMeshes) mesh.visible = false;
  const stone: OakMaterial = w.own(sculptureMaterial({
    roughness: 0.95,
    rimColor: palette.skyHorizon,
    rimStrength: 0.25,
    toonFloor: 0.4,
    name: 'weatheredLimestone',
  }));
  const wallGeometries = Array.from({ length: 5 }, (_, i) => w.own(ruinWallGeometry(skin, i)));
  const archGeometry = w.own(makeRuinArch(skin));
  const towerGeometry = w.own(makeRuinTower(skin, 1.6, 0.12));
  const wallCells: [number, number][] = [];
  const archCells: [number, number, number][] = [];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isWall(grid, x, y)) wallCells.push([x, y]);
      else if (hash2f(x, y, ctx.seed ^ 0x671) < 0.08) {
        if (isWall(grid, x - 1, y) && isWall(grid, x + 1, y)) archCells.push([x, y, 0]);
        else if (isWall(grid, x, y - 1) && isWall(grid, x, y + 1)) archCells.push([x, y, Math.PI / 2]);
      }
    }
  }
  const transform = new THREE.Object3D();
  for (let variant = 0; variant < wallGeometries.length; variant++) {
    const selected = wallCells.filter(([x, y]) => Math.floor(hash2f(x, y, ctx.seed) * 5) === variant);
    const walls = w.own(new THREE.InstancedMesh(wallGeometries[variant]!, stone, selected.length));
    walls.name = `weatheredMazeWalls${variant}`;
    walls.castShadow = true;
    walls.receiveShadow = true;
    selected.forEach(([x, y], i) => {
      cellToWorldVec(x, y, grid, transform.position);
      transform.rotation.y = hash2f(x, y, ctx.seed ^ 0x14ad) > 0.5 ? Math.PI : 0;
      transform.scale.set(1, 0.9 + hash2f(y, x, ctx.seed) * 0.2, 1);
      transform.updateMatrix();
      walls.setMatrixAt(i, transform.matrix);
    });
    walls.instanceMatrix.needsUpdate = true;
    w.add(walls);
  }
  const arches = w.own(new THREE.InstancedMesh(archGeometry, stone, Math.max(1, archCells.length)));
  arches.count = archCells.length;
  arches.castShadow = true;
  archCells.forEach(([x, y, yaw], i) => {
    cellToWorldVec(x, y, grid, transform.position);
    transform.rotation.y = yaw + Math.PI / 2;
    transform.scale.setScalar(1);
    transform.updateMatrix();
    arches.setMatrixAt(i, transform.matrix);
  });
  arches.instanceMatrix.needsUpdate = true;
  w.add(arches);
  const towers = w.own(new THREE.InstancedMesh(towerGeometry, stone, 4));
  towers.castShadow = true;
  for (let i = 0; i < 4; i++) {
    cellToWorldVec((i & 1) ? grid.w - 1 : 0, i < 2 ? 0 : grid.h - 1, grid, transform.position);
    transform.rotation.y = i * 0.4;
    transform.scale.setScalar(0.92 + i * 0.08);
    transform.updateMatrix();
    towers.setMatrixAt(i, transform.matrix);
  }
  towers.instanceMatrix.needsUpdate = true;
  w.add(towers);
  const ruinBank = w.own(makeSceneryBank({
    geometry: w.own(rockGeometry(2.7, 1)), material: stone, tileSize: 12, perTile: 28,
    colorA: skin.stoneDark, colorB: skin.flagDamp, seed: ctx.seed, name: 'woodlandWallRemnants', receiveShadow: true,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 28; i++) {
        const x = Math.round((tx + rng()) * 12);
        const z = Math.round((tz + rng()) * 12);
        if (Math.abs(x) < halfX + 0.15 && Math.abs(z) < halfZ + 0.15) continue;
        if (Math.abs(Math.sin(x * 0.38 + Math.sin(z * 0.24))) < 0.7) continue;
        write(x, landscape.heightAt(x, z), z, rng() > 0.5 ? Math.PI : 0, 0.35 + rng() * 0.7, 0.15 + rng() * 0.3, rng());
      }
    },
  }));
  w.add(ruinBank.mesh);
  const rubbleBank = w.own(makeSceneryBank({
    geometry: w.own(rockGeometry(1, 0)), material: stone, tileSize: 12, perTile: 32,
    colorA: skin.flagDamp, colorB: skin.stoneLight, seed: ctx.seed ^ 0x5ba3, name: 'ruinRubble',
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 32; i++) {
        const x = (tx + rng()) * 12;
        const z = (tz + rng()) * 12;
        if (Math.abs(x) < halfX - 0.5 && Math.abs(z) < halfZ - 0.5) continue;
        write(x, landscape.heightAt(x, z), z, rng() * Math.PI * 2, 0.18 + rng() * 0.3, 0.22 + rng() * 0.4, rng());
      }
    },
  }));
  w.add(rubbleBank.mesh);
  const exclude = excludeAny(excludeWallCells(grid), excludeCellCenters(grid, 0.3));
  for (const flowering of [false, true]) for (const outside of [false, true]) {
    const plants = w.time(w.own((flowering ? scatterFlowers : scatterGrass)({
      count: (flowering ? 130 : 620) * (outside ? 2 : 1),
      area: { halfX: halfX + (outside ? 3.2 : 0), halfZ: halfZ + (outside ? 3.2 : 0) },
      seed: ctx.seed ^ (flowering ? 0x2f83 : 0x6a11) ^ (outside ? 917 : 0),
      exclude: outside ? (x, z) => Math.max(Math.abs(x) - halfX, Math.abs(z) - halfZ) < 0.04
        || Math.hypot(Math.max(0, Math.abs(x) - halfX), Math.max(0, Math.abs(z) - halfZ)) > 1.8 + Math.sin(x * 0.83 + Math.sin(z * 0.66) * 2) * 1.1 : exclude,
      heightAt: outside ? landscape.heightAt : undefined,
      color: flowering ? skin.bloom : skin.grass,
      scale: flowering ? [0.3, 0.6] : [0.26, 0.55],
      wind: 0.04,
      name: flowering ? 'ruinFlowers' : 'ruinGrass',
    })));
    w.add(plants.group);
  }
  const dust = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'dust', count: 360,
    bounds: { x: 0, y: 1.5, z: 0, halfX, halfY: 1.4, halfZ },
    seed: ctx.seed,
    color: skin.sunWarm,
    opacity: 0.4,
    size: 0.8,
    rise: 0.04,
    swirl: 0.14,
    name: 'ruinAir',
  }))));
  w.add(dust.points);
  const shaftDust = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'dust', count: 100,
    bounds: { x: 0, y: 1.5, z: 0, halfX: 0.5, halfY: 1.5, halfZ: 0.5 },
    seed: ctx.seed ^ 0x2ba9,
    color: skin.sunWarm,
    size: 1,
    opacity: 0.55,
    rise: 0.05,
    name: 'shaftDust',
  }))));
  w.add(shaftDust.points);
  return w.seal({
    sky,
    rig,
    field,
    setGoalPosition(x, y, z) { shaftDust.points.position.set(x, y, z); },
    setView(coverage, camera) {
      fog.density = 2.8 / coverage.far;
      landscape.setView(coverage, camera, sky);
      ruinBank.setView(coverage);
      rubbleBank.setView(coverage);
    },
  });
}
