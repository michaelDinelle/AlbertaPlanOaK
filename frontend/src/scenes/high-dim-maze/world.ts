import * as THREE from 'three';
import { CELL, buildCellField, cellToWorldVec, isWall, type CellFieldHandle } from '@kit/grid';
import { applyFog, makeLightRig, type LightRig } from '@kit/lighting';
import { oakMaterial } from '@kit/materials';
import { blendTokens, shade, smoothstep01 } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { mergeColoredParts, type ColoredPart } from '@kit/parts';
import { hash2f } from '@kit/rng';
import { skyFromPalette, type SkyHandle } from '@kit/sky';
import { TOKENS, type Palette } from '@kit/tokens';
import type { SceneContext } from '@kit/types';
import { grassClumpGeometry, rockGeometry } from '@kit/vegetation';
import { makeWorld, type WorldCore } from '@kit/world';
import { HIGHDIMMAZE_H, HIGHDIMMAZE_W } from '@sim/envs/highdimmaze';
import type { EnvGeometry, GridGeometry } from '@sim/types';
import { makeContinuousSurface, makeSceneryBank, type ViewCoverage } from '../jelly-bean-world/view';

export function resolveGrid(geometry: EnvGeometry): GridGeometry {
  const grid = geometry.grid;
  if (!grid || grid.w !== HIGHDIMMAZE_W || grid.h !== HIGHDIMMAZE_H || grid.walls.length !== grid.w * grid.h || !grid.hallways?.length) {
    throw new Error('High-Dim Maze requires complete recorded grid geometry');
  }
  return grid;
}

export interface CanyonSkin {
  sandSun: THREE.Color;
  sandShade: THREE.Color;
  stone: THREE.Color;
  strata: readonly THREE.Color[];
  desert: THREE.Color;
  scrub: THREE.Color;
  farStone: THREE.Color;
  rim: THREE.Color;
  water: THREE.Color;
  waterDeep: THREE.Color;
  reed: THREE.Color;
  shaft: THREE.Color;
  dust: THREE.Color;
}

export function canyonSkin(palette: Palette, tokens: typeof TOKENS): CanyonSkin {
  const terracotta = tokens.accents['high-dim-maze'] ?? tokens.celebrationGold;
  const ember = tokens.accents['mountain-car'] ?? terracotta;
  const teal = tokens.accents['deep-sea'] ?? tokens.uiCyan;
  const leaf = tokens.accents['foliage'] ?? palette.groundA;
  const tan = TOKENS.palettes.dusk.groundA;
  const hot = TOKENS.palettes.dusk.skyHorizon;

  return {
    sandSun: shade(blendTokens(blendTokens(tan, terracotta, 0.62), palette.sun, 0.16), 0.08),
    sandShade: shade(blendTokens(blendTokens(tan, terracotta, 0.5), palette.hemiSky, 0.2), 0.44),
    stone: shade(blendTokens(terracotta, ember, 0.54), 0.63),
    strata: [
      shade(blendTokens(ember, terracotta, 0.2), 0.72),
      shade(blendTokens(terracotta, palette.sun, 0.42), 0.28),
      shade(blendTokens(terracotta, ember, 0.4), 0.55),
      shade(blendTokens(terracotta, palette.sun, 0.16), 0.3),
    ],
    desert: shade(blendTokens(blendTokens(palette.groundA, tan, 0.5), terracotta, 0.16), 0.22),
    scrub: shade(blendTokens(palette.groundA, palette.groundB, 0.4), 0.44),
    farStone: shade(blendTokens(terracotta, palette.skyHorizon, 0.48), 0.36),
    rim: blendTokens(palette.sun, hot, 0.42),
    water: blendTokens(teal, palette.sun, 0.1),
    waterDeep: shade(blendTokens(teal, TOKENS.palettes.underwater.skyBottom, 0.62), 0.34),
    reed: shade(blendTokens(leaf, palette.groundB, 0.18), 0.14),
    shaft: blendTokens(palette.sun, terracotta, 0.28),
    dust: blendTokens(blendTokens(terracotta, palette.sun, 0.55), palette.groundB, 0.2),
  };
}

const ARCH_SPAN_HALF = 0.5;
const ARCH_OPENING = 1.25;
const ARCH_BAND = 0.34;
const ARCH_RIBBON = 0.36;
const ARCH_LEG = 0.24;

function finGeometry(skin: CanyonSkin): THREE.BufferGeometry {
  const parts: ColoredPart[] = [];
  const core = new THREE.BoxGeometry(0.985, 1, 0.985, 2, 4, 2);
  core.translate(0, 0.5, 0);
  const position = core.getAttribute('position');
  for (let i = 0; i < position.count; i++) {
    const y = position.getY(i);
    position.setX(i, position.getX(i) + Math.sin(y * 17 + position.getZ(i) * 3) * 0.018);
    position.setZ(i, position.getZ(i) + Math.cos(y * 13 + position.getX(i) * 4) * 0.016);
  }
  core.computeVertexNormals();
  parts.push({ geometry: core, color: skin.stone });
  for (const [i, cy] of [0.17, 0.42, 0.67, 1].entries()) {
    const spread = 1.026 + (i % 2) * 0.018;
    const bed = new THREE.BoxGeometry(spread, i === 3 ? 0.064 : 0.05, spread);
    bed.translate(0, cy, 0);
    parts.push({ geometry: bed, color: skin.strata[i]! });
  }
  return mergeColoredParts(parts, 'weatheredSandstoneFin');
}

export function makeArchGeometry(
  skin: CanyonSkin,
  height: number,
  jambs: readonly [number, number],
  seed: number,
): THREE.BufferGeometry {
  const spans = 20;
  const half = CELL * ARCH_SPAN_HALF;
  const depth = CELL * 0.5;
  const opening = Math.max(0.55, Math.min(ARCH_OPENING, height - ARCH_BAND));

  const [endA, endB] = jambs;
  const jambLine = (t: number): number => endA + (endB - endA) * (t + 1) * 0.5;

  const soffit = (t: number): number => {
    const a = Math.min(1, Math.abs(t));
    const shape = Math.pow(Math.max(0, 1 - Math.pow(a, 6)), 0.35);
    const wobble = 1 + 0.06 * Math.sin(t * 7.3 + seed * 0.017) + 0.035 * Math.sin(t * 15.1 - seed * 0.011);
    const base = jambLine(t);
    return base + Math.max(0, opening - base) * shape * wobble;
  };
  const crest = (t: number): number => {
    const base = jambLine(t);
    const hump = Math.max(0, height - base) * Math.pow(Math.max(0, 1 - t * t), 0.8);
    const erode = 0.035 * Math.sin(t * 5.1 + seed * 0.023) * (1 - t * t);
    return base + hump + erode;
  };

  const depthAt = (t: number): number => depth * (ARCH_RIBBON + (ARCH_LEG - ARCH_RIBBON) * Math.pow(Math.abs(t), 2.2));

  const xs = new Float32Array(spans + 1);
  const lo = new Float32Array(spans + 1);
  const hi = new Float32Array(spans + 1);
  const dz = new Float32Array(spans + 1);
  for (let i = 0; i <= spans; i++) {
    const t = (i / spans) * 2 - 1;
    xs[i] = t * half;
    const under = soffit(t);
    lo[i] = under;
    const taper = 1 - Math.min(1, Math.abs(t));
    hi[i] = Math.max(crest(t), under + ARCH_BAND * 0.5 * taper);
    dz[i] = depthAt(t);
  }

  const positions: number[] = [];
  const colors: number[] = [];
  const face = new THREE.Color(skin.strata[2] ?? skin.stone);
  const soffitColor = new THREE.Color(skin.strata[1] ?? skin.stone);
  const capColor = new THREE.Color(skin.strata[3] ?? skin.stone);
  const jambColor = new THREE.Color(skin.strata[2] ?? skin.stone);

  const push = (x: number, y: number, z: number, c: THREE.Color): void => {
    positions.push(x, y, z);
    colors.push(c.r, c.g, c.b);
  };
  const quad = (
    a: readonly [number, number, number],
    b: readonly [number, number, number],
    c: readonly [number, number, number],
    d: readonly [number, number, number],
    color: THREE.Color,
  ): void => {
    push(a[0], a[1], a[2], color);
    push(b[0], b[1], b[2], color);
    push(c[0], c[1], c[2], color);
    push(a[0], a[1], a[2], color);
    push(c[0], c[1], c[2], color);
    push(d[0], d[1], d[2], color);
  };

  for (let i = 0; i < spans; i++) {
    const x0 = xs[i] ?? 0;
    const x1 = xs[i + 1] ?? 0;
    const l0 = lo[i] ?? 0;
    const l1 = lo[i + 1] ?? 0;
    const h0 = hi[i] ?? 0;
    const h1 = hi[i + 1] ?? 0;
    const d0 = dz[i] ?? depth;
    const d1 = dz[i + 1] ?? depth;
    quad([x0, l0, d0], [x1, l1, d1], [x1, h1, d1], [x0, h0, d0], face);
    quad([x1, l1, -d1], [x0, l0, -d0], [x0, h0, -d0], [x1, h1, -d1], face);
    quad([x0, l0, -d0], [x1, l1, -d1], [x1, l1, d1], [x0, l0, d0], soffitColor);
    quad([x0, h0, d0], [x1, h1, d1], [x1, h1, -d1], [x0, h0, -d0], capColor);
  }
  {
    const xL = xs[0] ?? -half;
    const xR = xs[spans] ?? half;
    const dL = dz[0] ?? depth;
    const dR = dz[spans] ?? depth;
    quad([xL, lo[0] ?? 0, -dL], [xL, lo[0] ?? 0, dL], [xL, hi[0] ?? 0, dL], [xL, hi[0] ?? 0, -dL], jambColor);
    quad(
      [xR, lo[spans] ?? 0, dR],
      [xR, lo[spans] ?? 0, -dR],
      [xR, hi[spans] ?? 0, -dR],
      [xR, hi[spans] ?? 0, dR],
      jambColor,
    );
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.name = 'canyonArch';
  return g;
}

export interface WorldHandle extends WorldCore {
  sky: SkyHandle;
  rig: LightRig;
  field: CellFieldHandle;
  setView(coverage: ViewCoverage, camera: THREE.Camera): void;
}

export function buildWorld(ctx: SceneContext, scene: THREE.Scene, grid: GridGeometry): WorldHandle {
  const palette = ctx.tokens.palettes.afternoon;
  const skin = canyonSkin(palette, ctx.tokens);
  const w = makeWorld('terracottaCanyons');
  const sky = w.time(w.own(skyFromPalette(palette, 'afternoon', {
    sunDir: [-0.5, 0.8, 0.33],
    sunGlow: 0.3,
    sunHdr: 5.2,
    topSpread: 0.7,
  })));
  w.add(sky.mesh);
  const rig = w.own(makeLightRig(palette, { sunDir: sky.sunDirection, pointCount: 3, pointDistance: 6, sunDistanceScale: 2.4 }));
  w.add(rig.group);
  rig.setShadowFocus({ x: 0, y: 0.5, z: 0 }, 12.5);
  const fog = applyFog(scene, palette, 550);
  const half = Math.max(grid.w, grid.h) * CELL / 2;
  const phase = ((ctx.seed >>> 0) % 8192) / 8192 * 6.28;
  const heightAt = (x: number, z: number): number => {
    const band = 0.65 * Math.sin(x * 0.13 + Math.sin(z * 0.045) * 1.1) * Math.cos(z * 0.095 + Math.sin(x * 0.04))
      + 0.25 * Math.sin(x * 0.065 + z * 0.045 + phase);
    const ridge = smoothstep01((band + 0.1) / 0.68) * (2.3 + Math.sin(z * 0.035 + phase) * 0.7);
    return -0.1 + smoothstep01((Math.max(Math.abs(x), Math.abs(z)) - half) / 3.5)
      * (ridge + 0.5 * Math.sin(x * 0.055 + phase) * Math.cos(z * 0.07));
  };
  const surface = w.time(w.own(makeContinuousSurface({
    name: 'ochreDesert',
    seed: ctx.seed,
    colorA: skin.sandShade,
    colorB: skin.sandSun,
    height: `
      float blend = smoothstep(0.0, 3.5, max(abs(p.x), abs(p.y)) - ${half.toFixed(4)});
      float band = 0.65 * sin(p.x * 0.13 + sin(p.y * 0.045) * 1.1) * cos(p.y * 0.095 + sin(p.x * 0.04))
        + 0.25 * sin(p.x * 0.065 + p.y * 0.045 + ${phase.toFixed(6)});
      float ridge = smoothstep(-0.1, 0.58, band) * (2.3 + sin(p.y * 0.035 + ${phase.toFixed(6)}) * 0.7);
      return -0.1 + blend * (ridge + 0.5 * sin(p.x * 0.055 + ${phase.toFixed(6)}) * cos(p.y * 0.07));
    `,
    detail: `
      float wind = sin(p.x * 2.5 + p.z * 0.3 + sin(p.z * 0.22));
      color *= 0.98 + wind * 0.02;
      float ridgeFace = smoothstep(0.3, 1.5, p.y);
      float strata = 0.5 + 0.5 * sin(p.y * 12.0 + sin(p.x * 0.6 + p.z * 0.5) * 0.4);
      color = mix(color, mix(vec3(${skin.strata[0]!.r.toFixed(5)}, ${skin.strata[0]!.g.toFixed(5)}, ${skin.strata[0]!.b.toFixed(5)}),
        vec3(${skin.strata[3]!.r.toFixed(5)}, ${skin.strata[3]!.g.toFixed(5)}, ${skin.strata[3]!.b.toFixed(5)}), strata * 0.45), ridgeFace * 0.6);
    `,
  })));
  w.add(surface.mesh);
  const field = w.time(w.own(buildCellField(grid, {
    floorColor: palette.sun,
    wallColor: palette.sun,
    floorUnderWalls: true,
    floorTint: (x, y) => isWall(grid, x, y) ? skin.stone : blendTokens(skin.sandShade, skin.sandSun, 0.65),
    wallHeight: 0,
    floorThickness: 0.16,
    floorInset: 0.025,
    seed: ctx.seed,
    name: 'canyonRoutes',
  })));
  w.add(field.group);
  for (const mesh of [...field.wallMeshes, ...field.floorMeshes]) mesh.visible = false;
  const material = w.own(oakMaterial({
    vertexColors: true, roughness: 0.96, flatShading: true,
    rimColor: skin.rim, rimStrength: 0.32, toonSteps: 3, toonSoftness: 0.3, toonFloor: 0.3,
    name: 'layeredSandstone',
  }));
  const wallGeometry = w.own(finGeometry(skin));
  const cells: [number, number][] = [];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) if (isWall(grid, x, y)) cells.push([x, y]);
  }
  const walls = w.own(new THREE.InstancedMesh(wallGeometry, material, cells.length));
  walls.castShadow = true;
  walls.receiveShadow = true;
  walls.name = 'canyonFins';
  const transform = new THREE.Object3D();
  cells.forEach(([x, y], i) => {
    cellToWorldVec(x, y, grid, transform.position);
    transform.rotation.set(0, 0, 0);
    const t = y / (grid.h - 1);
    const rim = x === 0 || y === 0 || x === grid.w - 1 || y === grid.h - 1;
    let height = rim ? 1.24 - t * 0.7 : 1.28 - t * 0.28;
    if (!rim) for (const gap of grid.hallways!) {
      const alongZ = isWall(grid, gap.x, gap.y - 1) && isWall(grid, gap.x, gap.y + 1);
      const along = alongZ ? Math.abs(y - gap.y) : Math.abs(x - gap.x);
      const across = alongZ ? Math.abs(x - gap.x) : Math.abs(y - gap.y);
      if (across === 0 && (along === 1 || along === 2)) height *= along === 1 ? 0.36 : 0.74;
    }
    transform.scale.set(1, height * (0.91 + hash2f(x, y, ctx.seed ^ 0x51ed270b) * 0.18), 1);
    transform.updateMatrix();
    walls.setMatrixAt(i, transform.matrix);
  });
  walls.instanceMatrix.needsUpdate = true;
  w.add(walls);
  const arch = w.own(makeArchGeometry(skin, 1.4, [0.4, 0.4], ctx.seed));
  const gaps = grid.hallways!;
  const arches = w.own(new THREE.InstancedMesh(arch, material, gaps.length));
  arches.castShadow = true;
  arches.name = 'canyonPassages';
  gaps.forEach((gap, i) => {
    cellToWorldVec(gap.x, gap.y, grid, transform.position);
    transform.rotation.y = isWall(grid, gap.x - 1, gap.y) && isWall(grid, gap.x + 1, gap.y) ? 0 : Math.PI / 2;
    transform.scale.setScalar(1);
    transform.updateMatrix();
    arches.setMatrixAt(i, transform.matrix);
  });
  arches.instanceMatrix.needsUpdate = true;
  w.add(arches);
  const mesaParts: ColoredPart[] = [];
  for (let i = 0; i < 5; i++) {
    const fin = wallGeometry.clone();
    fin.scale(1.8 + (i % 3) * 0.7, 1.4 + i * 0.3, 1.5 + (i % 2) * 0.6);
    fin.translate(Math.sin(i * 2.3) * 1.7, 0, Math.cos(i * 2.3) * 1.2);
    mesaParts.push({ geometry: fin, color: skin.strata[i % skin.strata.length]! });
  }
  const mesa = w.own(mergeColoredParts(mesaParts, 'erodedSandstoneRidge'));
  const banks = [
    { name: 'farSandstoneRidges', geometry: mesa, tile: 128, perTile: 5, low: 2, span: 4 },
  ].map((spec) => {
    const bank = w.own(makeSceneryBank({
      name: spec.name, geometry: spec.geometry, material,
      tileSize: spec.tile, perTile: spec.perTile, seed: ctx.seed,
      colorA: 0xffffff, colorB: 0xd8bfa5, receiveShadow: true,
      fill(tx, tz, rng, write) {
        for (let i = 0; i < spec.perTile; i++) {
          const x = (tx + rng()) * spec.tile;
          const z = (tz + rng()) * spec.tile;
          const size = spec.low + rng() * spec.span;
          const edge = size * 2.8 + 38;
          if (Math.abs(x) < half + edge && Math.abs(z) < half + edge) continue;
          write(x, heightAt(x, z), z, rng() * Math.PI, size, 0.8 + rng() * 0.5, rng() * 0.4);
        }
      },
    }));
    w.add(bank.mesh);
    return bank;
  });
  for (const plants of [true, false]) {
    const scatter = w.own(makeSceneryBank({
      name: plants ? 'canyonScrub' : 'canyonScree',
      geometry: w.own(plants ? grassClumpGeometry(3) : rockGeometry(1, 0)),
      material: w.own(oakMaterial({ color: plants ? skin.scrub : skin.stone, vertexColors: true,
        roughness: 0.98, side: THREE.DoubleSide, rimColor: skin.rim, rimStrength: 0.16 })),
      tileSize: 12, perTile: plants ? 32 : 20, seed: ctx.seed ^ 0x69c,
      colorA: 0xffffff, colorB: 0xc5ae8d,
      fill(tx, tz, rng, write) {
        for (let i = 0; i < (plants ? 32 : 20); i++) {
          const x = (tx + rng()) * 12;
          const z = (tz + rng()) * 12;
          if (Math.abs(x) < half && Math.abs(z) < half) continue;
          write(x, heightAt(x, z), z, rng() * Math.PI, 0.12 + rng() * (plants ? 0.35 : 0.3), 0.6 + rng() * 0.6, rng());
        }
      },
    }));
    w.add(scatter.mesh);
    banks.push(scatter);
  }
  const dust = w.size(w.time(w.own(makeAmbientParticles({
    preset: 'dust', count: 260,
    bounds: { x: 0, y: 0.7, z: 0, halfX: half, halfY: 0.6, halfZ: half },
    seed: ctx.seed ^ 0x7d21,
    color: skin.dust,
    size: 0.55,
    opacity: 0.22,
    rise: 0.025,
    swirl: 0.1,
    name: 'canyonWind',
  }))));
  w.add(dust.points);
  return w.seal({
    sky,
    rig,
    field,
    setView(coverage, camera) {
      surface.setView(coverage);
      for (const bank of banks) bank.setView(coverage);
      fog.density = 2.8 / coverage.far;
      sky.mesh.position.copy(camera.position);
      sky.mesh.scale.setScalar(coverage.far * 0.85);
    },
  });
}
