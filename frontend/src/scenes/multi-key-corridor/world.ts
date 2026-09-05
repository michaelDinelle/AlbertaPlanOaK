import { sculptureMaterial } from '../key-corridor/world';
import { makeLandscape } from '../key-corridor/world';
import type { ViewCoverage } from '@kit/view';
import * as THREE from 'three';
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
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { emissiveMaterial, oakMaterial, oakUniforms, type EmissiveMaterial, type OakMaterial } from '@kit/materials';
import { blendTokens, clamp01, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import {
  excludeAny,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  type ExcludeFn,
  type VegetationHandle,
} from '@kit/vegetation';
import { makeWorld, type WorldCore } from '@kit/world';
import {
  MKC_H,
  MKC_ROOM_BASE,
  MKC_W,
  mkcRoomCells,
} from '@sim/envs/multikeycorridor';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';

export const TEMPLE_HALF_X = (MKC_W * CELL) / 2;
export const TEMPLE_HALF_Z = (MKC_H * CELL) / 2;
export const TERRAIN_SIZE = 46;
export const TERRAIN_SEGMENTS = 128;
export const WALL_HEIGHT = 0.7;
export const COURT_SINK = -0.07;
export const FOG_EXTENT = TERRAIN_SIZE * 11;
export const SUN_DIR: readonly [number, number, number] = [0.56, 0.82, 0.25];
export const SHADOW_RADIUS = 13.5;
export const CAMERA_AZIMUTH = 0.14;
export const CAMERA_STANDOFF = 13.5;

export const CHAMBERS = 4;

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w !== MKC_W || grid.h !== MKC_H || grid.walls.length !== grid.w * grid.h) {
    throw new Error('Multi-Key Corridor requires complete recorded grid geometry');
  }
  return grid;
}

export function resolveDoors(grid: GridGeometry): readonly GridCell[] {
  const doors = grid.doors;
  if (doors?.length !== 3) throw new Error('Multi-Key Corridor requires three recorded doors');
  return doors;
}

export function resolveKeys(grid: GridGeometry): readonly GridCell[] {
  const keys = grid.keys;
  if (keys?.length !== 3) throw new Error('Multi-Key Corridor requires three recorded initial keys');
  return keys;
}

export function resolveGoal(grid: GridGeometry): GridCell {
  if (!grid.goal) throw new Error('Multi-Key Corridor requires its recorded initial goal');
  return grid.goal;
}

const CHAMBER_LOOKUP: Int8Array = (() => {
  const table = new Int8Array(MKC_W * MKC_H).fill(-1);
  for (let r = 0; r < CHAMBERS; r++) {
    for (const cell of mkcRoomCells(r)) table[cell.y * MKC_W + cell.x] = r;
  }
  return table;
})();

export function chamberOf(x: number, y: number): number {
  if (x < 0 || y < 0 || x >= MKC_W || y >= MKC_H) return -1;
  return CHAMBER_LOOKUP[y * MKC_W + x] ?? -1;
}

export function processionT(x: number): number {
  return clamp01((x - 1) / (MKC_W - 3));
}

export interface ChamberBounds {
  cx: number;
  cz: number;
  halfX: number;
  halfZ: number;
  cells: number;
}

export function chamberBounds(grid: GridGeometry, chamber: number): ChamberBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  let cells = 0;
  const w: Vec3Tuple = [0, 0, 0];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isWall(grid, x, y)) continue;
      if (chamberOf(x, y) !== chamber) continue;
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

export interface ChamberSkin {
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
  const gold = tokens.accents['multi-key-corridor'] ?? tokens.celebrationGold;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const garden = tokens.accents['four-rooms'] ?? palette.groundA;
  const terracotta = tokens.accents['high-dim-maze'] ?? tokens.celebrationGold;
  const moss = tokens.accents['procedural-maze'] ?? palette.groundA;
  const blossom = tokens.accents['chain'] ?? tokens.celebrationGold;
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
      new THREE.Color(tokens.accents['key-corridor'] ?? tokens.celebrationGold),
      blendTokens(white, palette.hemiSky, 0.16),
      new THREE.Color(tokens.accents['mountain-car'] ?? tokens.celebrationGold),
    ],
    relic: blendTokens(tokens.celebrationGold, palette.sun, 0.24),
    flame: blendTokens(tokens.celebrationGold, terracotta, 0.24),
  };
}

export function excludeProcession(grid: GridGeometry, doors: readonly GridCell[], halfWidth: number): ExcludeFn {
  const row = doors[0]?.y ?? (grid.h - 1) / 2;
  const oz = (grid.h - 1) / 2;
  const zWorld = (row - oz) * CELL;
  return (_x, z) => Math.abs(z - zWorld) < halfWidth;
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
  field: CellFieldHandle;
  portalRoots: THREE.Object3D[];
  doors: readonly GridCell[];
  stoneMaterial: OakMaterial;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.morning;
  const skin = templeSkin(palette, ctx.tokens);
  const w = makeWorld('templeOfThreeGates');
  const doors = resolveDoors(grid);

  const sky = skyFromPalette(palette, 'morning', {
    sunDir: SUN_DIR,
    sunSize: 0.04,
    sunHdr: 5.2,
    sunGlow: 0.3,
    sunGlowPower: 26,
    topSpread: 0.68,
    bottomSpread: 0.46,
    radius: 340,
  });
  w.add(sky.mesh);
  w.own(sky);
  w.time({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 7.5,
    sunDistanceScale: 2.1,
  });
  w.add(rig.group);
  w.own(rig);
  rig.setShadowFocus({ x: 0.5, y: 0.35, z: 0 }, SHADOW_RADIUS);

  const fog = applyFog(scene, palette, FOG_EXTENT);

  const landscape = makeLandscape(w, ctx, {
    name: 'templeOrchards',
    halfX: TEMPLE_HALF_X,
    halfZ: TEMPLE_HALF_Z,
    ground: blendTokens(palette.groundA, skin.soil, 0.35),
    grass: palette.groundB,
    trees: [skin.brambleLeaf],
    grounds: skin.chambers.map((chamber) => chamber.floor),
    archetypes: ['conifer', 'broadleaf'],
    treeKind: (x) => x < -TEMPLE_HALF_X ? 'conifer' : 'broadleaf',
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
        const chamber = chamberOf(x, y);
        return chamber < 0 ? skin.threshold : (skin.chambers[chamber]?.floor ?? skin.threshold);
      },
      wallTint: (x, y) => {
        const chamber = chamberOf(x, y);
        if (chamber >= 0) return skin.chambers[chamber]?.wall ?? skin.stoneDark;
        const t = processionT(x);
        const cool = skin.chambers[0]?.wall ?? skin.stoneDark;
        const warm = skin.chambers[CHAMBERS - 1]?.wall ?? skin.stoneLight;
        return new THREE.Color().copy(cool).lerp(warm, t);
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

  const stoneMaterial: OakMaterial = sculptureMaterial({
    roughness: 0.93,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    toonSoftness: 0.3,
    toonFloor: 0.26,
    name: 'templeStone',
  });
  w.own({ dispose: () => stoneMaterial.dispose() });

  const portalGeometry = makePortalGeometry(skin);
  w.own({ dispose: () => portalGeometry.dispose() });
  const portalRoots: THREE.Object3D[] = [];
  {
    const portals = new THREE.InstancedMesh(portalGeometry, stoneMaterial, Math.max(1, doors.length));
    portals.name = 'templePortals';
    portals.castShadow = true;
    portals.receiveShadow = true;
    portals.count = 0;
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    const world: Vec3Tuple = [0, 0, 0];
    let n = 0;
    for (let i = 0; i < doors.length; i++) {
      const cell = doors[i];
      if (!cell) continue;
      const root = new THREE.Object3D();
      root.name = `portalRoot${i}`;
      cellToWorld(cell.x, cell.y, grid, world);
      root.position.set(world[0], 0, world[2]);
      w.add(root);
      portalRoots.push(root);

      position.set(world[0], 0, world[2]);
      euler.set(0, gateFacesX(grid, cell) ? 0 : Math.PI / 2, 0);
      quaternion.setFromEuler(euler);
      matrix.compose(position, quaternion, scale);
      portals.setMatrixAt(n, matrix);
      n++;
    }
    portals.count = n;
    portals.instanceMatrix.needsUpdate = true;
    portals.computeBoundingSphere();
    if (n > 0) w.add(portals);
    w.own({ dispose: () => portals.dispose() });
  }

  const lanternGeometry = makeLanternGeometry(skin);
  w.own({ dispose: () => lanternGeometry.dispose() });

  const lanternSpots: { x: number; z: number }[] = [];
  {
    const world: Vec3Tuple = [0, 0, 0];
    const row = doors[0]?.y ?? 3;
    for (let c = 0; c < CHAMBERS; c++) {
      const base = MKC_ROOM_BASE[c] ?? 1;
      for (const dz of [-1, 1]) {
        cellToWorld(base + 1, row + dz * 1.5, grid, world);
        lanternSpots.push({ x: world[0], z: world[2] });
      }
    }
  }
  {
    const lanterns = new THREE.InstancedMesh(lanternGeometry, stoneMaterial, Math.max(1, lanternSpots.length));
    lanterns.name = 'templeLanterns';
    lanterns.castShadow = true;
    lanterns.receiveShadow = true;
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    for (let i = 0; i < lanternSpots.length; i++) {
      const spot = lanternSpots[i] as { x: number; z: number };
      position.set(spot.x, 0, spot.z);
      euler.set(0, ((i * 37) % 12) * 0.09, 0);
      quaternion.setFromEuler(euler);
      matrix.compose(position, quaternion, scale);
      lanterns.setMatrixAt(i, matrix);
    }
    lanterns.count = lanternSpots.length;
    lanterns.instanceMatrix.needsUpdate = true;
    lanterns.computeBoundingSphere();
    w.add(lanterns);
    w.own({ dispose: () => lanterns.dispose() });
  }
  {
    const flameGeometry = new THREE.IcosahedronGeometry(0.055, 1);
    const flameMaterial: EmissiveMaterial = emissiveMaterial(skin.flame, 1.35, { name: 'templeLanternFlame' });
    const flames = new THREE.InstancedMesh(flameGeometry, flameMaterial, Math.max(1, lanternSpots.length));
    flames.name = 'templeLanternFlames';
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3(1, 1.35, 1);
    for (let i = 0; i < lanternSpots.length; i++) {
      const spot = lanternSpots[i] as { x: number; z: number };
      position.set(spot.x, LANTERN_FLAME_Y, spot.z);
      matrix.compose(position, quaternion, scale);
      flames.setMatrixAt(i, matrix);
    }
    flames.count = lanternSpots.length;
    flames.instanceMatrix.needsUpdate = true;
    flames.computeBoundingSphere();
    w.add(flames);
    w.own({
      dispose: () => {
        flames.dispose();
        flameGeometry.dispose();
        flameMaterial.dispose();
      },
    });
  }

  const noPlanting = excludeAny(
    excludeWallCells(grid),
    excludeCellCenters(grid, 0.34),
    excludeProcession(grid, doors, 0.34),
  );
  const bounds: ChamberBounds[] = [];
  for (let c = 0; c < CHAMBERS; c++) bounds.push(chamberBounds(grid, c));

  const planted: VegetationHandle[] = [];
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
  for (let i = 0; i < plantSpecs.length; i++) {
    const spec = plantSpecs[i] as (typeof plantSpecs)[number];
    const b = (bounds[spec.chamber]) as ChamberBounds;
    const chamber = skin.chambers[spec.chamber] as ChamberSkin;
    const color = spec.kind === 'flowers' ? chamber.bloom : chamber.litter;
    for (const outside of [false, true]) {
      const options = {
        count: spec.count * (outside ? 2 : 1),
        area: { x: b.cx, z: b.cz, halfX: b.halfX + (outside ? 2.8 : 0), halfZ: b.halfZ + (outside ? 3.2 : 0) },
        seed: (ctx.seed ^ 0x5b10) + i * 977 + (outside ? 917 : 0),
        exclude: outside ? (x: number, z: number) => Math.max(Math.abs(x) - TEMPLE_HALF_X, Math.abs(z) - TEMPLE_HALF_Z) < 0.04
          || Math.hypot(Math.max(0, Math.abs(x - b.cx) - b.halfX), Math.max(0, Math.abs(z - b.cz) - b.halfZ)) > 1.8 + Math.sin(x * 0.83 + Math.sin(z * 0.66) * 2) * 1.1 : noPlanting,
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
      const handle =
        spec.kind === 'grass'
          ? scatterGrass(options)
          : spec.kind === 'flowers'
            ? scatterFlowers(options)
            : scatterRocks(options);
      w.add(handle.group);
      w.time(w.own(handle));
      planted.push(handle);
    }
  }

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
    const b = bounds[spec.chamber] as ChamberBounds;
    const handle = makeAmbientParticles({
      preset: spec.preset,
      count: spec.count,
      bounds: { x: b.cx, y: spec.y, z: b.cz, halfX: b.halfX, halfY: 0.95, halfZ: b.halfZ },
      seed: (ctx.seed ^ 0x7d21) + spec.chamber * 613,
      color: skin.chambers[spec.chamber]?.motes ?? white,
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
    portalRoots,
    doors,
    stoneMaterial,
  }, {
    time: (t) => {
      oakUniforms(stoneMaterial)?.setTime(t);
    },
    drop: () => {
      planted.length = 0;
      lanternSpots.length = 0;
      portalRoots.length = 0;
    },
  });
}

export const LANTERN_FLAME_Y = 0.575;
export const PORTAL_LINTEL_Y = 1.42;

export function makePortalGeometry(skin: TempleSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const PIER_H = PORTAL_LINTEL_Y;
  const PIER_Z = 0.95;

  for (const sz of [-1, 1]) {
    const pier = new THREE.BoxGeometry(0.46, PIER_H, 0.42);
    pier.translate(0, PIER_H * 0.5, sz * PIER_Z);
    parts.push({ geometry: pier, color: skin.stoneLight });
    const plinth = new THREE.BoxGeometry(0.58, 0.14, 0.54);
    plinth.translate(0, 0.07, sz * PIER_Z);
    parts.push({ geometry: plinth, color: skin.stoneDark });
    const capital = new THREE.BoxGeometry(0.56, 0.13, 0.5);
    capital.translate(0, PIER_H - 0.065, sz * PIER_Z);
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

export function makeLanternGeometry(skin: TempleSkin): THREE.BufferGeometry {
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
