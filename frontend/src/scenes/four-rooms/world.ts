import { sculptureMaterial } from '../key-corridor/world';
import { makeLandscape } from '../key-corridor/world';
import type { ViewCoverage } from '@kit/view';
import * as THREE from 'three';
import {
  buildCellField,
  CELL,
  cellToWorld,
  collapseWallInstance,
  excludeCellCenters,
  excludeWallCells,
  isWall,
  type CellFieldHandle,
  type Vec3Tuple,
} from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial, type OakMaterial } from '@kit/materials';
import { blendTokens, shade } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import {
  excludeAny,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
} from '@kit/vegetation';
import { makeWorld, type WorldCore } from '@kit/world';
import {
  FOURROOMS_H,
  FOURROOMS_W,
  fourRoomsRoom,
} from '@sim/envs/fourrooms';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';

export const COURT_HALF = (FOURROOMS_W * CELL) / 2;
export const TERRAIN_SIZE = 40;
export const TERRAIN_SEGMENTS = 128;
export const HEDGE_HEIGHT = 0.86;
export const COURT_SINK = -0.06;
export const FOG_EXTENT = TERRAIN_SIZE * 13;
export const SUN_DIR: readonly [number, number, number] = [-0.4, 0.866, -0.3];
export const SHADOW_RADIUS = 11.5;
export const CAMERA_AZIMUTH = 0.55;
export const CAMERA_STANDOFF = 12.5;

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w !== FOURROOMS_W || grid.h !== FOURROOMS_H || grid.walls.length !== grid.w * grid.h) {
    throw new Error('Four Rooms requires complete recorded grid geometry');
  }
  return grid;
}

export function resolveHallways(grid: GridGeometry): readonly GridCell[] {
  const hallways = grid.hallways;
  if (hallways?.length !== 4) throw new Error('Four Rooms requires four recorded hallways');
  return hallways;
}

export const SEASON_SPRING = 0;
export const SEASON_SUMMER = 1;
export const SEASON_AUTUMN = 2;
export const SEASON_WINTER = 3;
export const SEASON_NAMES: readonly string[] = ['spring', 'summer', 'autumn', 'winter'];

export interface RoomBounds {
  cx: number;
  cz: number;
  halfX: number;
  halfZ: number;
  cells: number;
}

export function roomBounds(grid: GridGeometry, room: number): RoomBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  let cells = 0;
  const w: Vec3Tuple = [0, 0, 0];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isWall(grid, x, y)) continue;
      if (fourRoomsRoom(x, y) !== room) continue;
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

export interface SeasonSkin {
  floor: THREE.Color;
  hedge: THREE.Color;
  litter: THREE.Color;
  tree: THREE.Color;
  motes: THREE.Color;
}

export interface CourtyardSkin {
  seasons: readonly SeasonSkin[];
  evergreen: THREE.Color;
  threshold: THREE.Color;
  soil: THREE.Color;
  stoneLight: THREE.Color;
  stoneDark: THREE.Color;
}

export function courtyardSkin(palette: Palette, tokens: typeof TOKENS): CourtyardSkin {
  const blossom = tokens.accents['chain'] ?? tokens.celebrationGold;
  const leaf = tokens.accents['foliage'] ?? palette.groundA;
  const gold = tokens.accents['multi-key-corridor'] ?? tokens.celebrationGold;
  const terracotta = tokens.accents['high-dim-maze'] ?? tokens.celebrationGold;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const white = palette.sun;

  const seasons: SeasonSkin[] = [
    {
      floor: shade(blendTokens(palette.groundB, blossom, 0.62), 0.32),
      hedge: shade(blendTokens(leaf, blossom, 0.42), 0.38),
      litter: shade(blendTokens(blossom, white, 0.2), 0.16),
      tree: shade(blendTokens(blossom, palette.groundB, 0.22), 0.2),
      motes: blendTokens(blossom, white, 0.35),
    },
    {
      floor: shade(blendTokens(palette.groundA, leaf, 0.42), 0.24),
      hedge: shade(blendTokens(palette.groundA, leaf, 0.22), 0.42),
      litter: shade(blendTokens(leaf, palette.groundB, 0.3), 0.2),
      tree: shade(blendTokens(palette.groundA, leaf, 0.3), 0.22),
      motes: blendTokens(palette.groundB, white, 0.4),
    },
    {
      floor: shade(blendTokens(gold, terracotta, 0.34), 0.32),
      hedge: shade(blendTokens(palette.groundA, terracotta, 0.55), 0.4),
      litter: shade(blendTokens(terracotta, gold, 0.35), 0.16),
      tree: shade(blendTokens(gold, terracotta, 0.42), 0.2),
      motes: blendTokens(gold, terracotta, 0.3),
    },
    {
      floor: shade(blendTokens(palette.hemiSky, white, 0.22), 0.44),
      hedge: shade(blendTokens(palette.groundA, silver, 0.6), 0.42),
      litter: shade(blendTokens(white, silver, 0.24), 0.2),
      tree: shade(blendTokens(silver, white, 0.25), 0.22),
      motes: blendTokens(white, silver, 0.2),
    },
  ];

  return {
    seasons,
    evergreen: shade(blendTokens(palette.groundA, leaf, 0.12), 0.58),
    threshold: shade(blendTokens(silver, palette.hemiGround, 0.4), 0.3),
    soil: shade(blendTokens(palette.hemiGround, TOKENS.palettes.dusk.groundA, 0.45), 0.4),
    stoneLight: shade(blendTokens(silver, palette.hemiGround, 0.3), 0.24),
    stoneDark: shade(blendTokens(palette.hemiGround, silver, 0.25), 0.42),
  };
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
  field: CellFieldHandle;
  sundialRoot: THREE.Object3D;
  gateRoots: THREE.Object3D[];
  hallways: readonly GridCell[];
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.noon;
  const skin = courtyardSkin(palette, ctx.tokens);
  const w = makeWorld('fourSeasonsCourtyard');

  const sky = skyFromPalette(palette, 'noon', {
    sunDir: SUN_DIR,
    sunSize: 0.035,
    sunHdr: 5.5,
    sunGlow: 0.22,
    sunGlowPower: 34,
    topSpread: 0.72,
    bottomSpread: 0.5,
    radius: 340,
  });
  w.add(sky.mesh);
  w.own(sky);
  w.time({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 3,
    pointDistance: 6.5,
    sunDistanceScale: 2.2,
  });
  w.add(rig.group);
  w.own(rig);
  rig.setShadowFocus({ x: 0, y: 0.35, z: 0 }, SHADOW_RADIUS);

  const fog = applyFog(scene, palette, FOG_EXTENT);

  const landscape = makeLandscape(w, ctx, {
    name: 'fourSeasons',
    halfX: COURT_HALF,
    halfZ: COURT_HALF,
    ground: blendTokens(palette.groundA, skin.soil, 0.2),
    grass: palette.groundB,
    trees: skin.seasons.map((season) => season.tree),
    grounds: skin.seasons.map((season) => season.floor),
    archetypes: ['broadleaf', 'bare'],
    treeKind: (x, z) => x >= 0 && z >= 0 ? 'bare' : 'broadleaf',
    amplitude: 3.8,
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
        const room = fourRoomsRoom(x, y);
        return room < 0 ? skin.threshold : (skin.seasons[room]?.floor ?? skin.threshold);
      },
      wallTint: (x, y) => {
        const room = fourRoomsRoom(x, y);
        return room < 0 ? skin.evergreen : (skin.seasons[room]?.hedge ?? skin.evergreen);
      },
      wallShape: 'hedge',
      wallHeight: HEDGE_HEIGHT,
      wallInset: 0.03,
      floorInset: 0.035,
      floorThickness: 0.14,
      jitter: 0.1,
      wind: 0.016,
      rimColor: palette.skyHorizon,
      rimStrength: 0.2,
      seed: ctx.seed,
      name: 'courtyardCells',
    },
  );
  w.add(field.group);
  w.own(field);
  w.time({ setTime: (t) => field.setTime(t) });

  const stoneMaterial: OakMaterial = sculptureMaterial({
    roughness: 0.94,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSoftness: 0.3,
    toonFloor: 0.24,
    name: 'courtyardStone',
  });
  w.own({ dispose: () => stoneMaterial.dispose() });

  const hallways = resolveHallways(grid);
  const gateGeometry = makeMoonGateGeometry(skin);
  w.own({ dispose: () => gateGeometry.dispose() });
  const gateRoots: THREE.Object3D[] = [];
  {
    const world: Vec3Tuple = [0, 0, 0];
    for (let i = 0; i < hallways.length; i++) {
      const h = hallways[i];
      if (!h) continue;
      const root = new THREE.Object3D();
      root.name = `moonGateRoot${i}`;
      cellToWorld(h.x, h.y, grid, world);
      root.position.set(world[0], 0, world[2]);
      const wallRunsAlongZ = isWall(grid, h.x, h.y - 1) && isWall(grid, h.x, h.y + 1);
      const mesh = new THREE.Mesh(gateGeometry, stoneMaterial);
      mesh.name = `moonGate${i}`;
      mesh.rotation.y = wallRunsAlongZ ? Math.PI / 2 : 0;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      root.add(mesh);
      w.add(root);
      gateRoots.push(root);
    }
  }

  collapseWallInstance(field, (grid.w / 2) | 0, (grid.h / 2) | 0);
  const sundialRoot = new THREE.Object3D();
  sundialRoot.name = 'sundialRoot';
  {
    const world: Vec3Tuple = [0, 0, 0];
    cellToWorld((grid.w / 2) | 0, (grid.h / 2) | 0, grid, world);
    sundialRoot.position.set(world[0], 0, world[2]);
  }
  w.add(sundialRoot);


  const sundialGeometry = makeSundialGeometry(skin);
  w.own({ dispose: () => sundialGeometry.dispose() });
  {
    const mesh = new THREE.Mesh(sundialGeometry, stoneMaterial);
    mesh.name = 'sundial';
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    sundialRoot.add(mesh);
  }

  const noPlanting = excludeAny(excludeWallCells(grid), excludeCellCenters(grid, 0.3));
  const bounds: RoomBounds[] = [];
  for (let room = 0; room < 4; room++) bounds.push(roomBounds(grid, room));

  const planting = [
    { scatter: scatterFlowers, count: 150, scale: [0.45, 0.85] as const, stretch: [0.8, 1.2] as const, wind: 0.03 },
    { scatter: scatterGrass, count: 520, scale: [0.3, 0.62] as const, stretch: [0.8, 1.4] as const, wind: 0.045 },
    { scatter: scatterRocks, count: 300, scale: [0.11, 0.26] as const, stretch: [0.1, 0.2] as const, wind: 0 },
    { scatter: scatterRocks, count: 190, scale: [0.08, 0.2] as const, stretch: [0.2, 0.36] as const, wind: 0 },
  ];
  planting.forEach((spec, season) => {
    const area = bounds[season] as RoomBounds;
    for (const outside of [false, true]) {
      const plants = spec.scatter({
        count: spec.count * (outside ? 2 : 1),
        area: { x: area.cx, z: area.cz, halfX: area.halfX + (outside ? 6 : 0), halfZ: area.halfZ + (outside ? 6 : 0) },
        seed: ctx.seed ^ (0x5b10 + season * 977 + (outside ? 917 : 0)),
        exclude: outside ? (x, z) => {
          const distance = Math.hypot(Math.max(0, Math.abs(x - area.cx) - area.halfX), Math.max(0, Math.abs(z - area.cz) - area.halfZ));
          return Math.max(Math.abs(x) - grid.w * CELL / 2, Math.abs(z) - grid.h * CELL / 2) < 0.04
            || hash2f(Math.floor(x * 1024), Math.floor(z * 1024), ctx.seed + season) > Math.exp(-distance * distance / 10) * (0.75 + Math.sin(x * 0.83 + Math.sin(z * 0.66) * 2) * 0.25);
        } : noPlanting,
        heightAt: outside ? landscape.heightAt : undefined,
        color: skin.seasons[season]?.litter ?? white,
        scale: spec.scale,
        stretch: spec.stretch,
        wind: spec.wind,
        hueJitter: 0.22,
        castShadow: false,
        name: `${SEASON_NAMES[season]}Garden`,
      });
      w.add(plants.group);
      w.time(w.own(plants));
    }
  });

  const moteSpecs: readonly {
    room: number;
    preset: 'pollen' | 'snow';
    count: number;
    size: number;
    rise: number;
    swirl: number;
    opacity: number;
    intensity: number;
    name: string;
  }[] = [
    { room: SEASON_SPRING, preset: 'pollen', count: 150, size: 1.15, rise: -0.09, swirl: 0.3, opacity: 0.62, intensity: 1.5, name: 'springPetals' },
    { room: SEASON_SUMMER, preset: 'pollen', count: 170, size: 0.75, rise: 0.08, swirl: 0.22, opacity: 0.5, intensity: 1.9, name: 'summerPollen' },
    { room: SEASON_AUTUMN, preset: 'snow', count: 120, size: 1.4, rise: -0.24, swirl: 0.3, opacity: 0.85, intensity: 1.1, name: 'autumnLeaffall' },
    { room: SEASON_WINTER, preset: 'snow', count: 220, size: 0.95, rise: -0.2, swirl: 0.16, opacity: 0.8, intensity: 1.2, name: 'winterSnow' },
  ];
  for (const spec of moteSpecs) {
    const b = bounds[spec.room] as RoomBounds;
    const handle = makeAmbientParticles({
      preset: spec.preset,
      count: spec.count,
      bounds: { x: b.cx, y: 1.05, z: b.cz, halfX: b.halfX, halfY: 0.95, halfZ: b.halfZ },
      seed: (ctx.seed ^ 0x7d21) + spec.room * 613,
      color: skin.seasons[spec.room]?.motes ?? white,
      size: spec.size,
      intensity: spec.intensity,
      opacity: spec.opacity,
      rise: spec.rise,
      swirl: spec.swirl,
      name: spec.name,
    });
    w.add(handle.points);
    w.size(w.time(w.own(handle)));
  }

  return w.seal({
    sky,
    rig,
    setView(coverage, camera) {
      fog.density = 2.8 / coverage.far;
      landscape.setView(coverage, camera, sky);
    },
    field,
    sundialRoot,
    gateRoots,
    hallways,
  });
}

export function makeMoonGateGeometry(skin: CourtyardSkin): THREE.BufferGeometry {
  const tubular = 26;
  const radial = 7;
  const RING_R = 0.455;
  const TUBE = 0.075;
  const RING_Y = 0.53;

  const parts: ColoredPart[] = [];
  const ring = new THREE.TorusGeometry(RING_R, TUBE, radial, tubular);
  ring.translate(0, RING_Y, 0);
  parts.push({ geometry: ring, color: skin.stoneLight });

  for (const sx of [-1, 1]) {
    const jamb = new THREE.BoxGeometry(0.15, RING_Y, 0.19);
    jamb.translate(sx * RING_R, RING_Y * 0.5, 0);
    parts.push({ geometry: jamb, color: skin.stoneDark });
  }
  const keystone = new THREE.BoxGeometry(0.19, 0.15, 0.21);
  keystone.translate(0, RING_Y + RING_R + 0.04, 0);
  parts.push({ geometry: keystone, color: skin.stoneLight });

  const slab = new THREE.BoxGeometry(0.86, 0.05, 0.44);
  slab.translate(0, 0.024, 0);
  parts.push({ geometry: slab, color: skin.stoneDark });

  return mergeColoredParts(parts, 'moonGate');
}

export function makeSundialGeometry(skin: CourtyardSkin): THREE.BufferGeometry {
  const radial = 14;
  const parts: ColoredPart[] = [];

  const kerb = new THREE.BoxGeometry(CELL - 0.06, 0.17, CELL - 0.06);
  kerb.translate(0, 0.085, 0);
  parts.push({ geometry: kerb, color: skin.stoneDark });

  const drum = new THREE.CylinderGeometry(0.31, 0.38, 0.46, radial, 1);
  drum.translate(0, 0.17 + 0.23, 0);
  parts.push({ geometry: drum, color: skin.stoneLight });

  const plate = new THREE.CylinderGeometry(0.42, 0.42, 0.07, radial, 1);
  plate.translate(0, 0.66, 0);
  parts.push({ geometry: plate, color: skin.stoneDark });

  const marks = 12;
  for (let i = 0; i < marks; i++) {
    const a = (i / marks) * Math.PI * 2;
    const mark = new THREE.BoxGeometry(0.1, 0.016, 0.028);
    mark.translate(0.3, 0.703, 0);
    mark.rotateY(a);
    parts.push({ geometry: mark, color: i % 3 === 0 ? skin.stoneLight : skin.threshold });
  }

  const blade = new THREE.BoxGeometry(0.03, 0.33, 0.26);
  blade.translate(0, 0.165, 0);
  blade.rotateX(0.62);
  blade.translate(0, 0.7, -0.02);
  parts.push({ geometry: blade, color: skin.stoneLight });

  return mergeColoredParts(parts, 'sundial');
}
