import * as THREE from 'three/webgpu';
import {
  buildCellField,
  CELL,
  cellToWorld,
  excludeCellCenters,
  excludeWallCells,
  isWall,
  type CellFieldHandle,
  type Vec3Tuple,
} from '@kit/grid';
import { instancedMesh } from '@kit/instances';
import { makeLandscape } from '@kit/landscape';
import type { LightRig } from '@kit/lighting';
import { emissiveMaterial, oakUniforms, sculptureMaterial, type OakMaterial } from '@kit/materials';
import { makeMeadow } from '@kit/meadow';
import { blendTokens, clamp01, shade } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import type { SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { excludeAny, scatterFlowers, scatterGrass, scatterRocks, type ExcludeFn } from '@kit/vegetation';
import type { ViewCoverage } from '@kit/view';
import type { WorldCore } from '@kit/world';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';
import { accentNamed } from '../tape';

const WALL_HEIGHT = 0.7;
const TERRAIN_SIZE = 46;
const FOG_EXTENT = TERRAIN_SIZE * 11;
const SUN_DIR: readonly [number, number, number] = [0.56, 0.82, 0.25];
const SHADOW_RADIUS = 13.5;

export const CHAMBERS = 4;
const LANTERN_FLAME_Y = 0.575;
export const PORTAL_LINTEL_Y = 1.42;

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.walls.length !== grid.w * grid.h || grid.doors?.length !== CHAMBERS - 1) {
    throw new Error('Temple of Three Gates requires complete recorded grid geometry with its three doors');
  }
  return grid;
}

export function resolveDoors(grid: GridGeometry): readonly GridCell[] {
  const doors = grid.doors;
  if (doors?.length !== CHAMBERS - 1) throw new Error('Temple of Three Gates requires three recorded doors');
  return doors;
}

function chamberOf(grid: GridGeometry, doors: readonly GridCell[], x: number, y: number): number {
  if (isWall(grid, x, y) || doors.some((d) => d.x === x && d.y === y)) return -1;
  return doors.filter((d) => d.x < x).length;
}

function processionT(grid: GridGeometry, x: number): number {
  return clamp01((x - 1) / (grid.w - 3));
}

interface ChamberBounds {
  cx: number;
  cz: number;
  halfX: number;
  halfZ: number;
  cells: number;
}

export function chamberBounds(grid: GridGeometry, doors: readonly GridCell[], chamber: number): ChamberBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  let cells = 0;
  const w: Vec3Tuple = [0, 0, 0];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (chamberOf(grid, doors, x, y) !== chamber) continue;
      cellToWorld(x, y, grid, w);
      if (w[0] < minX) minX = w[0];
      if (w[0] > maxX) maxX = w[0];
      if (w[2] < minZ) minZ = w[2];
      if (w[2] > maxZ) maxZ = w[2];
      cells++;
    }
  }
  if (cells === 0) return { cx: 0, cz: 0, halfX: 0, halfZ: 0, cells: 0 };
  const half = CELL * 0.5;
  return {
    cx: (minX + maxX) * 0.5,
    cz: (minZ + maxZ) * 0.5,
    halfX: (maxX - minX) * 0.5 + half,
    halfZ: (maxZ - minZ) * 0.5 + half,
    cells,
  };
}

export function gateFacesX(grid: GridGeometry, cell: GridCell): boolean {
  return isWall(grid, cell.x, cell.y - 1) && isWall(grid, cell.x, cell.y + 1);
}

interface ChamberSkin {
  floor: THREE.Color;
  wall: THREE.Color;
  litter: THREE.Color;
  bloom: THREE.Color;
  motes: THREE.Color;
}

export interface TempleSkin {
  chambers: readonly ChamberSkin[];
  threshold: THREE.Color;
  stoneLight: THREE.Color;
  stoneDark: THREE.Color;
  soil: THREE.Color;
  riverBed: THREE.Color;
  bramble: THREE.Color;
  brambleLeaf: THREE.Color;
  keys: readonly THREE.Color[];
  relic: THREE.Color;
  flame: THREE.Color;
}

export function templeSkin(palette: Palette, tokens: typeof TOKENS): TempleSkin {
  const gold = tokens.celebrationGold;
  const silver = accentNamed(tokens.accents, 'POMaze');
  const garden = accentNamed(tokens.accents, 'FourRooms');
  const terracotta = accentNamed(tokens.accents, 'HighDimMaze');
  const moss = accentNamed(tokens.accents, 'ProceduralMaze');
  const blossom = accentNamed(tokens.accents, 'Chain');
  const white = tokens.palettes.noon.sun;

  const chambers: ChamberSkin[] = [
    {
      floor: shade(blendTokens(palette.hemiSky, garden, 0.42), 0.64),
      litter: shade(blendTokens(moss, palette.hemiSky, 0.28), 0.42),
      bloom: shade(blendTokens(silver, white, 0.35), 0.3),
      motes: blendTokens(palette.hemiSky, white, 0.35),
      wall: shade(blendTokens(silver, palette.hemiGround, 0.55), 0.78),
    },
    {
      floor: shade(blendTokens(palette.groundA, garden, 0.34), 0.58),
      litter: shade(blendTokens(garden, palette.groundB, 0.34), 0.3),
      bloom: shade(blendTokens(blossom, white, 0.25), 0.2),
      motes: blendTokens(palette.groundB, white, 0.4),
      wall: shade(blendTokens(silver, palette.hemiGround, 0.5), 0.77),
    },
    {
      floor: shade(blendTokens(palette.groundB, terracotta, 0.42), 0.64),
      litter: shade(blendTokens(terracotta, gold, 0.42), 0.42),
      bloom: shade(blendTokens(gold, terracotta, 0.3), 0.24),
      motes: blendTokens(gold, white, 0.3),
      wall: shade(blendTokens(silver, terracotta, 0.2), 0.76),
    },
    {
      floor: shade(blendTokens(gold, tokens.celebrationGold, 0.4), 0.56),
      litter: shade(blendTokens(gold, white, 0.28), 0.38),
      bloom: shade(blendTokens(tokens.celebrationGold, white, 0.2), 0.18),
      motes: blendTokens(tokens.celebrationGold, white, 0.25),
      wall: shade(blendTokens(silver, terracotta, 0.26), 0.74),
    },
  ];

  return {
    chambers,
    threshold: shade(blendTokens(palette.hemiGround, silver, 0.24), 0.68),
    stoneLight: shade(blendTokens(silver, palette.hemiGround, 0.32), 0.3),
    stoneDark: shade(blendTokens(palette.hemiGround, silver, 0.3), 0.5),
    soil: shade(blendTokens(palette.hemiGround, TOKENS.palettes.dusk.groundA, 0.4), 0.5),
    riverBed: shade(blendTokens(TOKENS.palettes.underwater.skyBottom, palette.hemiGround, 0.3), 0.4),
    bramble: shade(blendTokens(palette.groundA, TOKENS.palettes.dusk.groundA, 0.5), 0.55),
    brambleLeaf: shade(blendTokens(palette.groundB, garden, 0.45), 0.34),
    keys: [
      new THREE.Color(accentNamed(tokens.accents, 'KeyCorridor')),
      blendTokens(white, palette.hemiSky, 0.16),
      new THREE.Color(accentNamed(tokens.accents, 'MountainCar')),
    ],
    relic: blendTokens(tokens.celebrationGold, palette.sun, 0.24),
    flame: blendTokens(tokens.celebrationGold, terracotta, 0.24),
  };
}

function excludeProcession(grid: GridGeometry, doors: readonly GridCell[], halfWidth: number): ExcludeFn {
  const row = doors[0]!.y;
  const zWorld = (row - (grid.h - 1) / 2) * CELL;
  return (_x, z) => Math.abs(z - zWorld) < halfWidth;
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
  field: CellFieldHandle;
  stoneMaterial: OakMaterial;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.morning;
  const skin = templeSkin(palette, ctx.tokens);
  const doors = resolveDoors(grid);
  const halfX = grid.w * CELL / 2;
  const halfZ = grid.h * CELL / 2;
  const chamber = (x: number, y: number) => chamberOf(grid, doors, x, y);

  const land = makeLandscape(scene, 'templeOfThreeGates', palette, 'morning', {
    sky: { sunDir: SUN_DIR, sunSize: 0.04, sunHdr: 5.2, sunGlow: 0.3, sunGlowPower: 26, topSpread: 0.68, bottomSpread: 0.46 },
    light: { pointCount: 4, pointDistance: 7.5, sunDistanceScale: 2.1 },
    fogExtent: FOG_EXTENT,
  });
  const { world: w, sky, rig } = land;
  rig.setShadowFocus({ x: 0.5, y: 0.35, z: 0 }, SHADOW_RADIUS);

  const landscape = makeMeadow(w, ctx, {
    name: 'templeOrchards',
    halfX,
    halfZ,
    ground: blendTokens(palette.groundA, skin.soil, 0.35),
    grass: palette.groundB,
    trees: [skin.brambleLeaf],
    grounds: skin.chambers.map((c) => c.floor),
    archetypes: ['conifer', 'broadleaf'],
    treeKind: (x) => (x < -halfX ? 'conifer' : 'broadleaf'),
    amplitude: 4.5,
  });

  const white = palette.sun;
  const field = buildCellField(
    grid,
    {
      floorColor: white,
      wallColor: white,
      floorUnderWalls: true,
      floorTint: (x, y) => {
        if (isWall(grid, x, y)) return skin.soil;
        const c = chamber(x, y);
        return c < 0 ? skin.threshold : skin.chambers[c]!.floor;
      },
      wallTint: (x) => {
        const cool = skin.chambers[0]!.wall;
        const warm = skin.chambers[CHAMBERS - 1]!.wall;
        return new THREE.Color().copy(cool).lerp(warm, processionT(grid, x));
      },
      wallShape: 'block',
      wallHeight: WALL_HEIGHT,
      wallInset: 0.015,
      floorInset: 0.03,
      floorThickness: 0.16,
      jitter: 0.08,
      rimColor: palette.skyHorizon,
      rimStrength: 0.22,
      seed: ctx.seed,
      name: 'templeCells',
    },
  );
  w.add(field.group);
  w.own(field);
  w.time({ setTime: (t) => field.setTime(t) });

  const stoneMaterial: OakMaterial = w.own(sculptureMaterial({
    roughness: 0.93,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSoftness: 0.3,
    toonFloor: 0.26,
    name: 'templeStone',
  }));

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const euler = new THREE.Euler();
  const scale = new THREE.Vector3(1, 1, 1);
  const spot: Vec3Tuple = [0, 0, 0];

  const portals = w.own(instancedMesh(w.own(makePortalGeometry(skin)), stoneMaterial, doors.length));
  portals.name = 'templePortals';
  portals.castShadow = true;
  portals.receiveShadow = true;
  doors.forEach((cell, i) => {
    cellToWorld(cell.x, cell.y, grid, spot);
    position.set(spot[0], 0, spot[2]);
    euler.set(0, gateFacesX(grid, cell) ? 0 : Math.PI / 2, 0);
    quaternion.setFromEuler(euler);
    matrix.compose(position, quaternion, scale);
    portals.setMatrixAt(i, matrix);
  });
  portals.instanceMatrix.needsUpdate = true;
  portals.computeBoundingSphere();
  w.add(portals);

  const bounds: ChamberBounds[] = [];
  for (let c = 0; c < CHAMBERS; c++) bounds.push(chamberBounds(grid, doors, c));

  const lanternSpots: { x: number; z: number }[] = [];
  const row = doors[0]!.y;
  for (const b of bounds) {
    for (const dz of [-1, 1]) {
      cellToWorld(0, row + dz * 1.5, grid, spot);
      lanternSpots.push({ x: b.cx, z: spot[2] });
    }
  }
  const lanterns = w.own(instancedMesh(w.own(makeLanternGeometry(skin)), stoneMaterial, lanternSpots.length));
  lanterns.name = 'templeLanterns';
  lanterns.castShadow = true;
  lanterns.receiveShadow = true;
  lanternSpots.forEach((at, i) => {
    position.set(at.x, 0, at.z);
    euler.set(0, ((i * 37) % 12) * 0.09, 0);
    quaternion.setFromEuler(euler);
    matrix.compose(position, quaternion, scale);
    lanterns.setMatrixAt(i, matrix);
  });
  lanterns.instanceMatrix.needsUpdate = true;
  lanterns.computeBoundingSphere();
  w.add(lanterns);

  const flameMaterial = w.own(emissiveMaterial(skin.flame, 1.35, { name: 'templeLanternFlame' }));
  const flames = w.own(instancedMesh(w.own(new THREE.IcosahedronGeometry(0.055, 1)), flameMaterial, lanternSpots.length));
  flames.name = 'templeLanternFlames';
  quaternion.identity();
  scale.set(1, 1.35, 1);
  lanternSpots.forEach((at, i) => {
    position.set(at.x, LANTERN_FLAME_Y, at.z);
    matrix.compose(position, quaternion, scale);
    flames.setMatrixAt(i, matrix);
  });
  flames.instanceMatrix.needsUpdate = true;
  flames.computeBoundingSphere();
  w.add(flames);

  const noPlanting = excludeAny(
    excludeWallCells(grid),
    excludeCellCenters(grid, 0.34),
    excludeProcession(grid, doors, 0.34),
  );
  const plantSpecs: readonly {
    chamber: number;
    kind: 'grass' | 'flowers' | 'rocks';
    count: number;
    scale: readonly [number, number];
    stretch: readonly [number, number];
    wind: number;
    name: string;
  }[] = [
    { chamber: 0, kind: 'grass', count: 105, scale: [0.26, 0.52], stretch: [0.7, 1.2], wind: 0.03, name: 'dawnwellMoss' },
    { chamber: 0, kind: 'flowers', count: 26, scale: [0.32, 0.62], stretch: [0.8, 1.2], wind: 0.028, name: 'dawnwellDewflowers' },
    { chamber: 1, kind: 'grass', count: 130, scale: [0.32, 0.64], stretch: [0.8, 1.4], wind: 0.042, name: 'groveGrass' },
    { chamber: 1, kind: 'flowers', count: 38, scale: [0.38, 0.72], stretch: [0.85, 1.3], wind: 0.03, name: 'groveBlossoms' },
    { chamber: 2, kind: 'rocks', count: 52, scale: [0.11, 0.24], stretch: [0.24, 0.42], wind: 0, name: 'ochreGravel' },
    { chamber: 2, kind: 'grass', count: 58, scale: [0.24, 0.46], stretch: [0.7, 1.1], wind: 0.034, name: 'ochreTufts' },
    { chamber: 3, kind: 'flowers', count: 34, scale: [0.34, 0.64], stretch: [0.9, 1.25], wind: 0.026, name: 'sanctumGoldflowers' },
    { chamber: 3, kind: 'rocks', count: 26, scale: [0.09, 0.18], stretch: [0.22, 0.36], wind: 0, name: 'sanctumGoldDust' },
  ];
  plantSpecs.forEach((spec, i) => {
    const b = bounds[spec.chamber]!;
    const tones = skin.chambers[spec.chamber]!;
    const color = spec.kind === 'flowers' ? tones.bloom : tones.litter;
    for (const outside of [false, true]) {
      const options = {
        count: spec.count * (outside ? 2 : 1),
        area: { x: b.cx, z: b.cz, halfX: b.halfX + (outside ? 2.8 : 0), halfZ: b.halfZ + (outside ? 3.2 : 0) },
        seed: (ctx.seed ^ 0x5b10) + i * 977 + (outside ? 917 : 0),
        exclude: outside
          ? (x: number, z: number) => Math.max(Math.abs(x) - halfX, Math.abs(z) - halfZ) < 0.04
            || Math.hypot(Math.max(0, Math.abs(x - b.cx) - b.halfX), Math.max(0, Math.abs(z - b.cz) - b.halfZ)) > 1.8 + Math.sin(x * 0.83 + Math.sin(z * 0.66) * 2) * 1.1
          : noPlanting,
        heightAt: outside ? landscape.heightAt : undefined,
        color,
        scale: spec.scale,
        stretch: spec.stretch,
        hueJitter: 0.24,
        ...(spec.wind > 0 ? { wind: spec.wind, windFrequency: 1.5 } : {}),
        rimColor: palette.skyHorizon,
        rimStrength: 0.3,
        castShadow: false,
        name: spec.name,
      };
      const handle = spec.kind === 'grass' ? scatterGrass(options) : spec.kind === 'flowers' ? scatterFlowers(options) : scatterRocks(options);
      w.add(handle.group);
      w.time(w.own(handle));
    }
  });

  const moteSpecs: readonly {
    chamber: number;
    preset: 'pollen' | 'dust' | 'snow';
    count: number;
    size: number;
    rise: number;
    swirl: number;
    opacity: number;
    intensity: number;
    y: number;
    name: string;
  }[] = [
    { chamber: 0, preset: 'snow', count: 70, size: 0.9, rise: -0.05, swirl: 0.14, opacity: 0.34, intensity: 0.9, y: 0.85, name: 'dawnwellDew' },
    { chamber: 1, preset: 'pollen', count: 120, size: 0.8, rise: 0.07, swirl: 0.26, opacity: 0.4, intensity: 1.35, y: 1.05, name: 'grovePollen' },
    { chamber: 2, preset: 'dust', count: 120, size: 0.76, rise: 0.03, swirl: 0.2, opacity: 0.38, intensity: 1.2, y: 1.0, name: 'ochreDust' },
    { chamber: 3, preset: 'pollen', count: 140, size: 0.92, rise: 0.05, swirl: 0.18, opacity: 0.46, intensity: 1.7, y: 1.15, name: 'sanctumMotes' },
  ];
  for (const spec of moteSpecs) {
    const b = bounds[spec.chamber]!;
    const handle = makeAmbientParticles({
      preset: spec.preset,
      count: spec.count,
      bounds: { x: b.cx, y: spec.y, z: b.cz, halfX: b.halfX, halfY: 0.95, halfZ: b.halfZ },
      seed: (ctx.seed ^ 0x7d21) + spec.chamber * 613,
      color: skin.chambers[spec.chamber]!.motes,
      size: spec.size,
      intensity: spec.intensity,
      opacity: spec.opacity,
      rise: spec.rise,
      swirl: spec.swirl,
      name: spec.name,
    });
    w.add(handle.points);
    w.time(w.own(handle));
  }

  return w.seal<WorldHandle>({
    sky,
    rig,
    setView(coverage, camera) {
      land.setView(coverage, camera, landscape);
    },
    field,
    stoneMaterial,
  }, {
    time: (t) => {
      oakUniforms(stoneMaterial)?.setTime(t);
    },
  });
}

function makePortalGeometry(skin: TempleSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const PIER_H = PORTAL_LINTEL_Y;
  const PIER_Z = 0.95;
  const PLINTH_H = 0.14;
  const CAPITAL_H = 0.13;
  const SHAFT_H = PIER_H - PLINTH_H - CAPITAL_H;

  for (const sz of [-1, 1]) {
    const pier = new THREE.BoxGeometry(0.46, SHAFT_H, 0.42);
    pier.translate(0, PLINTH_H + SHAFT_H * 0.5, sz * PIER_Z);
    parts.push({ geometry: pier, color: skin.stoneLight });
    const plinth = new THREE.BoxGeometry(0.58, PLINTH_H, 0.54);
    plinth.translate(0, PLINTH_H * 0.5, sz * PIER_Z);
    parts.push({ geometry: plinth, color: skin.stoneDark });
    const capital = new THREE.BoxGeometry(0.56, CAPITAL_H, 0.5);
    capital.translate(0, PIER_H - CAPITAL_H * 0.5, sz * PIER_Z);
    parts.push({ geometry: capital, color: skin.stoneDark });
    const finial = new THREE.OctahedronGeometry(0.095, 1);
    finial.scale(1, 1.7, 1);
    finial.translate(0, PIER_H + 0.16, sz * PIER_Z);
    parts.push({ geometry: finial, color: skin.stoneLight });
  }

  const lintel = new THREE.BoxGeometry(0.3, 0.16, 2.16);
  lintel.translate(0, PIER_H + 0.08, 0);
  parts.push({ geometry: lintel, color: skin.stoneLight });
  const band = new THREE.BoxGeometry(0.34, 0.05, 1.7);
  band.translate(0, PIER_H + 0.005, 0);
  parts.push({ geometry: band, color: skin.stoneDark });

  return mergeColoredParts(parts, 'templePortal');
}

function makeLanternGeometry(skin: TempleSkin): THREE.BufferGeometry {
  const radial = 8;
  const parts: ColoredPart[] = [];

  const base = new THREE.CylinderGeometry(0.12, 0.16, 0.08, radial, 1);
  base.translate(0, 0.04, 0);
  parts.push({ geometry: base, color: skin.stoneDark });

  const shaft = new THREE.CylinderGeometry(0.05, 0.062, 0.36, radial, 1);
  shaft.translate(0, 0.26, 0);
  parts.push({ geometry: shaft, color: skin.stoneLight });

  const bowl = new THREE.CylinderGeometry(0.125, 0.082, 0.085, radial, 1);
  bowl.translate(0, 0.485, 0);
  parts.push({ geometry: bowl, color: skin.stoneDark });

  const cap = new THREE.ConeGeometry(0.155, 0.13, radial, 1);
  cap.translate(0, 0.69, 0);
  parts.push({ geometry: cap, color: skin.stoneLight });

  const finial = new THREE.OctahedronGeometry(0.042, 0);
  finial.translate(0, 0.78, 0);
  parts.push({ geometry: finial, color: skin.stoneDark });

  return mergeColoredParts(parts, 'templeLantern');
}
