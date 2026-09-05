import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '../jelly-bean-world/view';
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
import { oakMaterial, type OakMaterial, type OakMaterialOptions } from '@kit/materials';
import { blendTokens, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import {
  excludeAny,
  excludeRect,
  scatterFlowers,
  scatterGrass,
  scatterRocks,
  broadleafTreeGeometry,
  coniferTreeGeometry,
  bareTreeGeometry,
  grassClumpGeometry,
  type TreeArchetype,
  type VegetationHandle,
} from '@kit/vegetation';
import { makeWorld, type WorldCore, type WorldBuilder } from '@kit/world';
import {
  KEYCORRIDOR_H,
  KEYCORRIDOR_W,
} from '@sim/envs/keycorridor';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';

export const GARDEN_HALF_X = (KEYCORRIDOR_W * CELL) / 2;
export const GARDEN_HALF_Z = (KEYCORRIDOR_H * CELL) / 2;
export const TERRAIN_SIZE = 40;
export const TERRAIN_SEGMENTS = 128;

export const KERB_HEIGHT = 0.5;
export const DIVIDER_HEIGHT = 1.02;
export const DIVIDER_CAP = 0.11;
export const DIVIDER_THICKNESS = 0.7;
export const GARDEN_SINK = -0.06;

export const FOG_EXTENT = TERRAIN_SIZE * 11;

export const SUN_DIR: readonly [number, number, number] = [-0.46, 0.78, -0.42];
export const SHADOW_RADIUS = 9.5;

export const CAMERA_AZIMUTH = -0.72;
export const CAMERA_STANDOFF = 12;

export function smoothBand(edge0: number, edge1: number, x: number): number {
  if (edge1 <= edge0) return x >= edge1 ? 1 : 0;
  return smoothstep01((x - edge0) / (edge1 - edge0));
}

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w !== KEYCORRIDOR_W || grid.h !== KEYCORRIDOR_H || grid.walls.length !== grid.w * grid.h) {
    throw new Error('Key Corridor requires complete recorded grid geometry');
  }
  return grid;
}

export function resolveDoor(grid: GridGeometry): GridCell {
  const door = grid.doors?.[0];
  if (!door) throw new Error('Key Corridor requires its recorded door');
  return door;
}

export const SIDE_WEST = -1;
export const SIDE_THRESHOLD = 0;
export const SIDE_EAST = 1;

export function gardenSide(door: GridCell, x: number): number {
  return x < door.x ? SIDE_WEST : x > door.x ? SIDE_EAST : SIDE_THRESHOLD;
}

export function isDividerWall(grid: GridGeometry, x: number, y: number): boolean {
  if (!isWall(grid, x, y)) return false;
  const openX = !isWall(grid, x - 1, y) && !isWall(grid, x + 1, y);
  const openZ = !isWall(grid, x, y - 1) && !isWall(grid, x, y + 1);
  return openX || openZ;
}

export function gateYaw(grid: GridGeometry, door: GridCell): number {
  const wallRunsAlongZ = isWall(grid, door.x, door.y - 1) && isWall(grid, door.x, door.y + 1);
  return wallRunsAlongZ ? Math.PI / 2 : 0;
}

export interface SideBounds {
  cx: number;
  cz: number;
  halfX: number;
  halfZ: number;
  cells: number;
}

export function sideBounds(grid: GridGeometry, door: GridCell, side: number): SideBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  let cells = 0;
  const w: Vec3Tuple = [0, 0, 0];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isWall(grid, x, y)) continue;
      if (gardenSide(door, x) !== side) continue;
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

export interface GardenSkin {
  wildFloor: THREE.Color;
  formalFloor: THREE.Color;
  threshold: THREE.Color;
  soil: THREE.Color;
  kerb: THREE.Color;
  divider: THREE.Color;
  dividerCap: THREE.Color;
  ivy: THREE.Color;
  ivyDark: THREE.Color;
  wood: THREE.Color;
  woodDark: THREE.Color;
  iron: THREE.Color;
  shrineStone: THREE.Color;
  shrineDark: THREE.Color;
  amber: THREE.Color;
  shrinePad: THREE.Color;
  wildGrass: THREE.Color;
  wildBloom: THREE.Color;
  gravel: THREE.Color;
  topiary: THREE.Color;
  meadow: THREE.Color;
  tree: THREE.Color;
  motesWild: THREE.Color;
  motesFormal: THREE.Color;
}

export function gardenSkin(palette: Palette, tokens: typeof TOKENS): GardenSkin {
  const amber = tokens.accents['key-corridor'] ?? tokens.celebrationGold;
  const leaf = tokens.accents['four-rooms'] ?? palette.groundB;
  const moss = tokens.accents['procedural-maze'] ?? palette.groundA;
  const gold = tokens.accents['multi-key-corridor'] ?? tokens.celebrationGold;
  const silver = tokens.accents['po-maze'] ?? palette.hemiSky;
  const terracotta = tokens.accents['high-dim-maze'] ?? tokens.celebrationGold;
  const earth = TOKENS.palettes.dusk.groundA;

  return {
    wildFloor: shade(blendTokens(palette.groundA, moss, 0.36), 0.34),
    formalFloor: shade(blendTokens(palette.hemiSky, palette.hemiGround, 0.36), 0.3),
    threshold: shade(blendTokens(gold, palette.hemiGround, 0.3), 0.34),
    soil: shade(blendTokens(palette.hemiGround, earth, 0.45), 0.42),
    kerb: shade(blendTokens(palette.skyBottom, terracotta, 0.34), 0.54),
    divider: shade(blendTokens(palette.groundA, silver, 0.3), 0.72),
    dividerCap: shade(blendTokens(silver, palette.hemiGround, 0.46), 0.52),
    ivy: shade(blendTokens(leaf, palette.groundA, 0.34), 0.26),
    ivyDark: shade(blendTokens(moss, palette.groundA, 0.55), 0.52),
    wood: shade(blendTokens(terracotta, earth, 0.16), 0.02),
    woodDark: shade(blendTokens(terracotta, earth, 0.38), 0.16),
    iron: shade(blendTokens(palette.hemiGround, silver, 0.18), 0.68),
    shrineStone: shade(blendTokens(silver, palette.hemiGround, 0.26), 0.2),
    shrineDark: shade(blendTokens(palette.hemiGround, silver, 0.3), 0.46),
    amber: new THREE.Color(amber),
    shrinePad: shade(blendTokens(gold, palette.hemiGround, 0.3), 0.4),
    wildGrass: shade(blendTokens(palette.groundB, moss, 0.42), 0.22),
    wildBloom: shade(blendTokens(amber, palette.sun, 0.18), 0.2),
    gravel: shade(blendTokens(palette.hemiSky, palette.hemiGround, 0.3), 0.46),
    topiary: shade(blendTokens(palette.groundA, leaf, 0.28), 0.4),
    meadow: shade(blendTokens(palette.groundA, palette.groundB, 0.3), 0.34),
    tree: shade(blendTokens(palette.groundA, moss, 0.26), 0.4),
    motesWild: blendTokens(amber, palette.sun, 0.4),
    motesFormal: blendTokens(palette.hemiSky, palette.sun, 0.4),
  };
}

export interface LandscapeOptions {
  name: string;
  halfX: number;
  halfZ: number;
  ground: THREE.ColorRepresentation;
  grass: THREE.ColorRepresentation;
  trees: readonly THREE.ColorRepresentation[];
  archetypes?: readonly TreeArchetype[];
  treeKind?: (x: number, z: number) => TreeArchetype;
  grounds?: readonly THREE.ColorRepresentation[];
  amplitude?: number;
}

export function makeLandscape(w: WorldBuilder, ctx: SceneContext, opts: LandscapeOptions) {
  const amplitude = opts.amplitude ?? 2.6;
  const phase = ((ctx.seed >>> 0) % 4096) / 4096 * Math.PI * 2;
  const heightAt = (x: number, z: number): number => {
    const outer = Math.max(Math.abs(x) - opts.halfX, Math.abs(z) - opts.halfZ);
    const bank = smoothstep01((outer + 0.25) / 0.7) * Math.exp(-Math.max(0, outer) / (2.1 + Math.sin(x * 0.5 + z * 0.37) * 0.7));
    return -0.075 + bank * 0.34 + smoothstep01(outer / 10) * amplitude * (0.55 * Math.sin(x * 0.032 + phase) * Math.cos(z * 0.041) + 0.25 * Math.sin(x * 0.077 + z * 0.056 + phase));
  };
  const grounds = opts.grounds?.map((value) => {
    const c = new THREE.Color(value);
    return `vec3(${c.r.toFixed(5)}, ${c.g.toFixed(5)}, ${c.b.toFixed(5)})`;
  });
  const groundDetail = grounds ? `
    float seasonX = smoothstep(-4.0, 4.0, p.x + (landNoise(p.xz * 0.08 + 11.0) - 0.5) * 7.0);
    float seasonZ = smoothstep(-4.0, 4.0, p.z + (landNoise(p.xz * 0.07 + 23.0) - 0.5) * 7.0);
    vec3 season = mix(mix(${grounds[0]}, ${grounds[1]}, seasonX), mix(${grounds[2]}, ${grounds[3]}, seasonX), seasonZ);
    float garden = exp(-max(0.0, max(abs(p.x) - ${opts.halfX.toFixed(4)}, abs(p.z) - ${opts.halfZ.toFixed(4)})) * 0.07);
    color = mix(color, season, garden * 0.62);` : '';
  const surface = w.time(w.own(makeContinuousSurface({
    name: `${opts.name}Terrain`,
    seed: ctx.seed,
    colorA: opts.ground,
    colorB: opts.grass,
    height: `
      float outer = max(abs(p.x) - ${opts.halfX.toFixed(4)}, abs(p.y) - ${opts.halfZ.toFixed(4)});
      float roll = 0.55 * sin(p.x * 0.032 + ${phase.toFixed(6)}) * cos(p.y * 0.041)
        + 0.25 * sin(p.x * 0.077 + p.y * 0.056 + ${phase.toFixed(6)});
      float bank = smoothstep(-0.25, 0.45, outer) * exp(-max(0.0, outer) / (2.1 + sin(p.x * 0.5 + p.y * 0.37) * 0.7));
      return -0.075 + bank * 0.34 + smoothstep(0.0, 10.0, outer) * ${amplitude.toFixed(4)} * roll;
    `,
    detail: `
      float grass = sin(p.x * 1.7 + sin(p.z * 1.1)) * sin(p.z * 2.3);
      ${groundDetail}
      color *= 0.96 + 0.045 * grass;
    `,
  })));
  w.add(surface.mesh);
  const material = w.own(oakMaterial({ vertexColors: true, roughness: 0.9, flatShading: true,
    rimColor: opts.ground, rimStrength: 0.3, wind: { amplitude: 0.03, frequency: 0.8, height: 1.7 }, name: `${opts.name}Trees` }));
  material.customProgramCacheKey = () => `${opts.name}|grove-colors`;
  const compile = material.onBeforeCompile;
  material.onBeforeCompile = (shader, renderer) => {
    compile.call(material, shader, renderer);
    shader.uniforms.uGroveColors = { value: Array.from({ length: 4 }, (_, i) => new THREE.Color(opts.trees[i % opts.trees.length] as THREE.ColorRepresentation)) };
    shader.vertexShader = 'varying vec2 vGroveWorld; varying float vGroveCanopy;\n' + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `
      #include <begin_vertex>
      vGroveWorld = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xz;
      vGroveCanopy = step(color.r, color.g * 1.35);
    `);
    shader.fragmentShader = 'uniform vec3 uGroveColors[4]; varying vec2 vGroveWorld; varying float vGroveCanopy;\n' + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `
      #include <color_fragment>
      vec2 season = smoothstep(vec2(-6.0), vec2(6.0), vGroveWorld + sin(vGroveWorld.yx * 0.17) * 2.5);
      vec3 grove = mix(mix(uGroveColors[0], uGroveColors[1], season.x), mix(uGroveColors[2], uGroveColors[3], season.x), season.y);
      diffuseColor.rgb *= mix(vec3(1.0), grove, vGroveCanopy);
    `);
  };
  const kinds = opts.archetypes ?? ['broadleaf'];
  const treeGeometry = (kind: TreeArchetype, detail: number) => {
    const geometry = kind === 'conifer' ? coniferTreeGeometry(detail) : kind === 'bare' ? bareTreeGeometry(detail) : broadleafTreeGeometry(detail);
    const colors = geometry.getAttribute('color');
    const trunk = kind === 'bare' ? colors.count : (kind === 'conifer' ? Math.max(5, 5 + detail * 3) : Math.max(4, 4 + detail * 2)) * 12;
    for (let i = 0; i < trunk; i++) colors.setXYZ(i, 0.28, 0.15, 0.075);
    return geometry;
  };
  const banks = kinds.map((kind, variant) => {
    const geometry = w.own(treeGeometry(kind, 1));
    const bank = w.own(makeSceneryBank({
      name: `${opts.name}${kind}`, geometry, material, tileSize: 12, perTile: 16,
      seed: ctx.seed ^ (variant * 997), colorA: 0xffffff, colorB: 0xbdcbb2, receiveShadow: true,
      fill(tx, tz, rng, write) {
        for (let i = 0; i < 16; i++) {
          const x = (tx + (i % 4 + rng()) / 4) * 12;
          const z = (tz + (Math.floor(i / 4) + rng()) / 4) * 12;
          if (Math.abs(x) < opts.halfX + 0.7 && Math.abs(z) < opts.halfZ + 0.7) continue;
          if (opts.treeKind ? opts.treeKind(x, z) !== kind : rng() > 1 / kinds.length) continue;
          const grove = 0.5 + 0.5 * Math.sin(x * 0.24 + Math.cos(z * 0.21) * 1.8);
          if (rng() > 0.3 + grove * 0.7) continue;
          write(x, heightAt(x, z), z, rng() * Math.PI * 2, 0.7 + rng() * 0.65, 0.9 + rng() * 0.45, rng());
        }
      },
    }));
    w.add(bank.mesh);
    return bank;
  });
  const massParts: ColoredPart[] = [];
  for (let i = 0; i < 18; i++) {
    const kind = kinds[i % kinds.length] as TreeArchetype;
    const crown = treeGeometry(kind, 0);
    const x = Math.sin(i * 13.37) * 5.2;
    const z = Math.cos(i * 7.81) * 5.2;
    const size = 0.75 + hash2f(i, 31, ctx.seed) * 0.55;
    crown.scale(size, size * (0.9 + hash2f(i, 14, ctx.seed) * 0.45), size);
    crown.rotateY(i * 2.4);
    crown.translate(x, 0, z);
    massParts.push({ geometry: crown, color: new THREE.Color().setScalar(0.78 + (i % 4) * 0.06) });
  }
  const massColors = massParts.flatMap(({ geometry }) => Array.from(geometry.getAttribute('color').array));
  const massGeometry = w.own(mergeColoredParts(massParts, `${opts.name}Woodland`));
  massGeometry.setAttribute('color', new THREE.Float32BufferAttribute(massColors, 3));
  const masses = w.own(makeSceneryBank({
    name: `${opts.name}DistantGroves`, geometry: massGeometry, material,
    tileSize: 96, perTile: 8, seed: ctx.seed, colorA: 0xffffff, colorB: 0xc9d3d6,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 8; i++) {
        const x = (tx + rng()) * 96;
        const z = (tz + rng()) * 96;
        if (Math.abs(x) < opts.halfX + 70 && Math.abs(z) < opts.halfZ + 70) continue;
        write(x, heightAt(x, z) - 0.05, z, rng() * Math.PI * 2, 1, 1, rng());
      }
    },
  }));
  w.add(masses.mesh);
  const grassMaterial = w.own(oakMaterial({ color: opts.grass, vertexColors: true, roughness: 0.94,
    side: THREE.DoubleSide, wind: { amplitude: 0.04, frequency: 1.1, height: 0.5 }, name: `${opts.name}Understory` }));
  const grass = w.own(makeSceneryBank({
    name: `${opts.name}Meadow`, geometry: w.own(grassClumpGeometry(3)), material: grassMaterial,
    tileSize: 12, perTile: 256, seed: ctx.seed ^ 0x54ad, colorA: 0xe6efc4, colorB: 0xabb998,
    fill(tx, tz, rng, write) {
      for (let i = 0; i < 256; i++) {
        const x = (tx + rng()) * 12;
        const z = (tz + rng()) * 12;
        const outer = Math.max(Math.abs(x) - opts.halfX, Math.abs(z) - opts.halfZ);
        if (outer < 0.02) continue;
        const patch = 0.5 + 0.5 * Math.sin(x * 0.8 + Math.sin(z * 0.55) * 2.1);
        if (rng() > 0.24 + Math.max(Math.exp(-outer * 0.6), patch * 0.65) * 0.76) continue;
        write(x, heightAt(x, z), z, rng() * Math.PI * 2, 0.32 + rng() * 0.36, 0.8 + rng() * 0.7, rng());
      }
    },
  }));
  w.add(grass.mesh);
  w.time({ setTime(t) { material.oak.setTime(t); grassMaterial.oak.setTime(t); } });
  return {
    heightAt,
    setView(coverage: ViewCoverage, camera: THREE.Camera, sky: SkyHandle) {
      surface.setView(coverage);
      for (const bank of banks) bank.setView(coverage);
      grass.setView(coverage);
      masses.setView(coverage);
      sky.mesh.position.copy(camera.position);
      sky.mesh.scale.setScalar(coverage.far * 0.85);
    },
  };
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
  field: CellFieldHandle;
  door: GridCell;
  gateYaw: number;
  dividerCells: readonly GridCell[];
  paintFloor(x: number, y: number, color: THREE.ColorRepresentation): void;
  restoreFloorTint(x: number, y: number): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette: Palette = ctx.tokens.palettes.morning;
  const skin = gardenSkin(palette, ctx.tokens);
  const w = makeWorld('ivyGateGarden');

  const door = resolveDoor(grid);
  const yaw = gateYaw(grid, door);

  const sky = skyFromPalette(palette, 'morning', {
    sunDir: SUN_DIR,
    sunSize: 0.04,
    sunHdr: 5,
    sunGlow: 0.3,
    sunGlowPower: 24,
    topSpread: 0.68,
    bottomSpread: 0.56,
    radius: 340,
  });
  w.add(sky.mesh);
  w.own(sky);
  w.time({ setTime: (t) => sky.setTime(t) });

  const rig = makeLightRig(palette, {
    sunDir: sky.sunDirection,
    pointCount: 4,
    pointDistance: 5.5,
    sunDistanceScale: 2.4,
  });
  w.add(rig.group);
  w.own(rig);
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, SHADOW_RADIUS);

  const fog = applyFog(scene, palette, FOG_EXTENT);

  const landscape = makeLandscape(w, ctx, {
    name: 'ivyGarden',
    halfX: GARDEN_HALF_X,
    halfZ: GARDEN_HALF_Z,
    ground: skin.meadow,
    grass: blendTokens(skin.meadow, palette.groundB, 0.4),
    trees: [skin.tree],
    grounds: [skin.wildFloor, skin.formalFloor, skin.wildFloor, skin.formalFloor],
    archetypes: ['broadleaf', 'conifer'],
  });

  const FLOOR_JITTER = 0.17;
  const TINT_WILD_A = skin.wildFloor;
  const TINT_WILD_B = blendTokens(skin.wildFloor, skin.soil, 0.24);
  const TINT_FORMAL_A = skin.formalFloor;
  const TINT_FORMAL_B = blendTokens(skin.formalFloor, skin.divider, 0.2);
  const floorTintFor = (x: number, y: number): THREE.Color => {
    if (isWall(grid, x, y)) return skin.soil;
    const side = gardenSide(door, x);
    if (side === SIDE_THRESHOLD) return skin.threshold;
    if (side === SIDE_WEST) return ((x * 3 + y * 5) & 1) === 0 ? TINT_WILD_A : TINT_WILD_B;
    return (y & 1) === 0 ? TINT_FORMAL_A : TINT_FORMAL_B;
  };

  const white = palette.sun;
  const field = buildCellField(
    grid,
    {
      floorColor: white,
      wallColor: white,
      floorUnderWalls: true,
      floorTint: (x, y) => floorTintFor(x, y),
      wallTint: () => skin.kerb,
      wallShape: 'block',
      wallHeight: KERB_HEIGHT,
      wallInset: 0.02,
      floorInset: 0.035,
      floorThickness: 0.14,
      jitter: FLOOR_JITTER,
      rimColor: palette.skyHorizon,
      rimStrength: 0.09,
      seed: ctx.seed,
      name: 'ivyGateCells',
    },
  );
  w.add(field.group);
  w.own(field);
  w.time({ setTime: (t) => field.setTime(t) });

  const dividerCells: GridCell[] = [];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isDividerWall(grid, x, y)) dividerCells.push({ x, y });
    }
  }
  const dividerMaterial: OakMaterial = sculptureMaterial({
    roughness: 0.93,
    rimColor: palette.skyHorizon,
    rimStrength: 0.13,
    toonSoftness: 0.3,
    toonFloor: 0.22,
    name: 'ivyGateDividerMaterial',
  });
  w.own({ dispose: () => dividerMaterial.dispose() });
  const dividerGeometry = w.own(makeDividerGeometry(skin));
  if (dividerCells.length > 0) {
    const geometry = dividerGeometry;
    const mesh = new THREE.InstancedMesh(geometry, dividerMaterial, dividerCells.length);
    mesh.name = 'ivyGateDivider';
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(dividerCells.length * 3),
      3,
    );
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const euler = new THREE.Euler();
    const scale = new THREE.Vector3(1, 1, 1);
    const tint = new THREE.Color();
    const world: Vec3Tuple = [0, 0, 0];
    for (let i = 0; i < dividerCells.length; i++) {
      const cell = dividerCells[i] as GridCell;
      collapseWallInstance(field, cell.x, cell.y);
      cellToWorld(cell.x, cell.y, grid, world);
      const n = hash2f(cell.x, cell.y, ctx.seed ^ 0x1d7a);
      position.set(world[0], 0, world[2]);
      euler.set(0, yaw + (n > 0.5 ? Math.PI : 0), 0);
      quaternion.setFromEuler(euler);
      scale.set(1, 1 + (n - 0.5) * 0.06, 1);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);
      const tone = 0.9 + hash2f(cell.x, cell.y, ctx.seed ^ 0x6b02) * 0.2;
      tint.setRGB(tone, tone, tone);
      mesh.setColorAt(i, tint);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    w.add(mesh);
    w.own({ dispose: () => mesh.dispose() });
  }

  const noPlanting = excludeAny(excludeWallCells(grid), excludeCellCenters(grid, 0.3));
  const wild = sideBounds(grid, door, SIDE_WEST);
  const formal = sideBounds(grid, door, SIDE_EAST);
  const planting: VegetationHandle[] = [];

  const wildGrass = scatterGrass({
    count: 420,
    area: { x: wild.cx, z: wild.cz, halfX: wild.halfX, halfZ: wild.halfZ },
    seed: ctx.seed ^ 0x4c17,
    exclude: noPlanting,
    color: skin.wildGrass,
    scale: [0.34, 0.7],
    stretch: [0.85, 1.5],
    hueJitter: 0.24,
    wind: 0.05,
    windFrequency: 1.6,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    name: 'wildGrass',
  });
  w.add(wildGrass.group);
  w.time(w.own(wildGrass));
  planting.push(wildGrass);

  const wildBlooms = scatterFlowers({
    count: 74,
    area: { x: wild.cx, z: wild.cz, halfX: wild.halfX, halfZ: wild.halfZ },
    seed: ctx.seed ^ 0x2b93,
    exclude: noPlanting,
    color: skin.wildBloom,
    scale: [0.45, 0.9],
    stretch: [0.8, 1.2],
    hueJitter: 0.24,
    wind: 0.032,
    windFrequency: 1.4,
    rimColor: palette.skyHorizon,
    rimStrength: 0.3,
    name: 'wildBlooms',
  });
  w.add(wildBlooms.group);
  w.time(w.own(wildBlooms));
  planting.push(wildBlooms);

  const gravel = scatterRocks({
    count: 130,
    area: { x: formal.cx, z: formal.cz, halfX: formal.halfX, halfZ: formal.halfZ },
    seed: ctx.seed ^ 0x7f41,
    exclude: noPlanting,
    color: shade(skin.gravel, 0.18),
    scale: [0.06, 0.13],
    stretch: [0.12, 0.24],
    hueJitter: 0.2,
    rimColor: palette.skyHorizon,
    rimStrength: 0.24,
    castShadow: false,
    name: 'formalGravel',
  });
  w.add(gravel.group);
  w.time(w.own(gravel));
  planting.push(gravel);

  const topiary = scatterRocks({
    count: 26,
    area: { x: formal.cx, z: formal.cz, halfX: formal.halfX, halfZ: formal.halfZ },
    seed: ctx.seed ^ 0x39d5,
    exclude: noPlanting,
    color: skin.topiary,
    scale: [0.16, 0.34],
    stretch: [0.75, 1.15],
    hueJitter: 0.16,
    rimColor: palette.skyHorizon,
    rimStrength: 0.34,
    castShadow: true,
    name: 'formalTopiary',
  });
  w.add(topiary.group);
  w.time(w.own(topiary));
  planting.push(topiary);

  const verge = scatterFlowers({
    count: 450,
    area: { halfX: GARDEN_HALF_X + 3.2, halfZ: GARDEN_HALF_Z + 3.2 },
    seed: ctx.seed ^ 0x58ba,
    exclude: excludeRect(0, 0, GARDEN_HALF_X, GARDEN_HALF_Z, 0.08),
    heightAt: landscape.heightAt,
    color: skin.wildBloom,
    scale: [0.34, 0.62],
    wind: 0.05,
    name: 'gardenVerge',
  });
  w.add(verge.group);
  w.time(w.own(verge));

  const wildMotes = makeAmbientParticles({
    preset: 'pollen',
    count: 190,
    bounds: { x: wild.cx, y: 0.95, z: wild.cz, halfX: wild.halfX, halfY: 0.85, halfZ: wild.halfZ },
    seed: ctx.seed ^ 0x7d21,
    color: skin.motesWild,
    size: 1.05,
    intensity: 1.7,
    opacity: 0.6,
    rise: 0.07,
    swirl: 0.26,
    name: 'wildPollen',
  });
  w.add(wildMotes.points);
  w.size(w.time(w.own(wildMotes)));

  const formalMotes = makeAmbientParticles({
    preset: 'dust',
    count: 150,
    bounds: { x: formal.cx, y: 0.8, z: formal.cz, halfX: formal.halfX, halfY: 0.72, halfZ: formal.halfZ },
    seed: ctx.seed ^ 0x51c3,
    color: skin.motesFormal,
    size: 0.72,
    intensity: 1.3,
    opacity: 0.44,
    rise: 0.03,
    swirl: 0.16,
    name: 'formalDust',
  });
  w.add(formalMotes.points);
  w.size(w.time(w.own(formalMotes)));

  const restoreColor = new THREE.Color();

  return w.seal({
    sky,
    rig,
    setView(coverage, camera) {
      fog.density = 2.8 / coverage.far;
      landscape.setView(coverage, camera, sky);
    },
    field,
    door,
    gateYaw: yaw,
    dividerCells,
    paintFloor(x, y, color) {
      field.setFloorColor(x, y, color);
    },
    restoreFloorTint(x, y) {
      const n = hash2f(x, y, ctx.seed);
      restoreColor.copy(floorTintFor(x, y)).multiplyScalar(1 + (n - 0.5) * FLOOR_JITTER);
      field.setFloorColor(x, y, restoreColor);
    },
  }, {
    drop: () => {
      planting.length = 0;
    },
  });
}

export function makeDividerGeometry(skin: GardenSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const size = CELL - 0.03;

  const courses = 3;
  for (let i = 0; i < courses; i++) {
    const y0 = (i / courses) * DIVIDER_HEIGHT;
    const y1 = ((i + 1) / courses) * DIVIDER_HEIGHT;
    const inset = i === courses - 1 ? 0.035 : i === 0 ? 0 : 0.018;
    const course = new THREE.BoxGeometry(size - inset * 2, y1 - y0 + 0.004, DIVIDER_THICKNESS - inset * 2);
    course.translate(0, (y0 + y1) * 0.5, 0);
    parts.push({
      geometry: course,
      color: i % 2 === 0 ? skin.divider : blendTokens(skin.divider, skin.dividerCap, 0.16),
    });
  }

  const cap = new THREE.BoxGeometry(size + 0.06, DIVIDER_CAP, DIVIDER_THICKNESS + 0.13);
  cap.translate(0, DIVIDER_HEIGHT + DIVIDER_CAP * 0.5, 0);
  parts.push({ geometry: cap, color: skin.dividerCap });

  const footing = new THREE.BoxGeometry(size, 0.13, DIVIDER_THICKNESS + 0.2);
  footing.translate(0, 0.065, 0);
  parts.push({ geometry: footing, color: skin.dividerCap });

  const rows = 2;
  const perRow = 9;
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < perRow; i++) {
      const t = (i + (r % 2) * 0.5) / perRow;
      const lean = Math.sin(t * 7.3 + r * 2.1) * 0.34;
      const leaf = ivyLeafGeometry();
      const sc = 0.82 + 0.32 * Math.sin(t * 4.1 + r * 1.7);
      leaf.scale(sc, sc, sc);
      leaf.rotateX(-1.3 + Math.cos(t * 5.7 + r) * 0.4);
      leaf.rotateZ(lean);
      leaf.rotateY(lean * 2.2 + t * 4.1 + r);
      leaf.translate(
        (t - 0.5) * (size + 0.05),
        DIVIDER_HEIGHT + DIVIDER_CAP + 0.01 + 0.035 * Math.sin(t * 9.1 + r),
        -0.02 - r * 0.19 - 0.13 * Math.abs(Math.sin(t * 3.3 + r)),
      );
      parts.push({ geometry: leaf, color: (i + r) % 3 === 0 ? skin.ivyDark : skin.ivy });
    }
  }

  const runners = 2;
  const leavesPer = 8;
  for (let r = 0; r < runners; r++) {
    const x0 = r === 0 ? -0.24 : 0.26;
    for (let i = 0; i < leavesPer; i++) {
      const t = i / (leavesPer - 1 || 1);
      const y = DIVIDER_HEIGHT + DIVIDER_CAP - 0.1 - t * (DIVIDER_HEIGHT * 0.78);
      const sway = Math.sin(t * 5.1 + r * 2.3) * 0.11;
      const s = 1 - 0.42 * t;
      const leaf = ivyLeafGeometry();
      leaf.scale(s, s, s);
      leaf.rotateX(-0.35 - t * 0.4);
      leaf.rotateZ(sway * 2.4);
      leaf.rotateY(sway * 3 + r * 1.7);
      leaf.translate(x0 + sway, y, -(DIVIDER_THICKNESS * 0.5) - 0.035);
      parts.push({ geometry: leaf, color: i % 3 === 0 ? skin.ivyDark : skin.ivy });
    }
  }

  return mergeColoredParts(parts, 'ivyWallSegment');
}

export function sculptureMaterial(opts: OakMaterialOptions): OakMaterial {
  return oakMaterial({ vertexColors: true, flatShading: true, toonSteps: 3, ...opts });
}

export function gradedGeometry(name: string, vertices: readonly (readonly [number, number, number, number])[]): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices.flatMap(([x, y, z]) => [x, y, z]), 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(vertices.flatMap(([, , , grade]) => [grade, grade, grade]), 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  geometry.name = name;
  return geometry;
}

export function ivyLeafGeometry(): THREE.BufferGeometry {
  const w = 0.105;
  const l = 0.2;
  const base: readonly [number, number, number] = [0, 0, 0];
  const sh0: readonly [number, number, number] = [-w, l * 0.46, 0.012];
  const sh1: readonly [number, number, number] = [w, l * 0.46, 0.012];
  const lobe0: readonly [number, number, number] = [-w * 0.62, l * 0.84, 0.006];
  const lobe1: readonly [number, number, number] = [w * 0.62, l * 0.84, 0.006];
  const tip: readonly [number, number, number] = [0, l, 0];
  const tris: readonly (readonly (readonly [number, number, number])[])[] = [
    [base, sh1, sh0],
    [sh0, sh1, lobe1],
    [sh0, lobe1, lobe0],
    [lobe0, lobe1, tip],
  ];
  const grade: readonly number[] = [0.62, 0.86, 0.86, 0.86, 0.86, 1.02, 0.86, 1.02, 1.02, 1.02, 1.02, 1.18];
  return gradedGeometry('ivyLeaf', tris.flat().map((point, i) => [...point, grade[i]!]));
}
