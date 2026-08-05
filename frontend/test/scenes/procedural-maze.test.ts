/// <reference types="bun" />
// The Overgrown Ruins (ProceduralMaze-cont-v0) — the accuracy + lifecycle self-check.
//
// This mirrors four-rooms.test.ts's shape and adds what THIS env, alone among the ten
// grid scenes, has to prove: the maze is regenerated per lane, so the scene is not a
// layout but a wall-SKINNING system, and the only way to check a skinning system is to
// point it at more than one maze. Every structural assertion below runs over BOTH seed 96
// (the harness's default lane) and seed 97, and asserts the masonry against
// `procMazeIsWall` rather than against anything this scene believes.
//
// It also locks the two behaviours that separate this env from FourRooms:
//   * BOTH ends teleport on a completion, so the celebration must be read from the
//     PREVIOUS frame's goal cell — the only cell that still remembers where the +1 was —
//     and the wisp must hard-cut instead of gliding;
//   * `done` fires on that same successful batch, so the scene must never sigh.

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { DEBUG_LAYER } from '@kit/debug';
import { CELL, cellToWorld, isWall, wallCellCount } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  PROCMAZE_GOAL,
  PROCMAZE_H,
  PROCMAZE_START,
  PROCMAZE_W,
  procMazeIsWall,
  proceduralMazeDecoder,
} from '@sim/envs/proceduralmaze';
import type { EnvGeometry, GridCell, GridGeometry } from '@sim/types';

import createScene, { growth, envelope, readGoalCell } from '../../src/scenes/procedural-maze/index';
import { FIXTURE_SEED, fixtures } from '../../src/scenes/procedural-maze/fixtures';
import { manifest } from '../../src/scenes/procedural-maze/manifest';
import {
  BASE_COURSE,
  IVY_CLING,
  IVY_FLOOR_GAP,
  MAX_CROWN,
  RUIN_HALF,
  TOWER_HEIGHT_MAX,
  archCells,
  blockHalfWidthAt,
  buildCourses,
  cornerCells,
  crownHeightAt,
  floorTintFor,
  ivyLedge,
  makeBirds,
  makeForestFloorHeight,
  resolveGrid,
  ruinSkin,
  skinWallCell,
  wallNeighbours,
} from '../../src/scenes/procedural-maze/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameState,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

/** the harness's default ProceduralMaze lane, and a second maze to prove the system */
const SEED = FIXTURE_SEED;
const SEED_B = 97;
const SEEDS: readonly number[] = [SEED, SEED_B];

function geometryOf(seed: number): EnvGeometry {
  return proceduralMazeDecoder.geometry(proceduralMazeDecoder.initStatic(seed), seed);
}

function gridOf(seed: number): GridGeometry {
  const g = geometryOf(seed).grid;
  if (!g) throw new Error('decoder produced no grid');
  return g;
}

async function live(
  tier: QualityTier = 'hero',
  seed: number = SEED,
): Promise<{ module: SceneModule; ctx: StubSceneContext }> {
  const module = createScene();
  const ctx = makeSceneContext({ geometry: geometryOf(seed), seed, tier });
  await module.init(ctx);
  return { module, ctx };
}

function fixtureNamed(name: string): SceneFixture {
  const f = fixtures.find((x) => x.name === name);
  if (!f) throw new Error(`fixture "${name}" missing`);
  return f;
}

function poseFixture(name: string, x: number, y: number, gx: number, gy: number, step = 0): SceneFixture {
  return {
    name,
    pose: [x, y, gx, gy, 0, 0, 0, 0, 0],
    ctrl: [0, 0, 0, 0, 0, 0],
    derived: { x, y, gx, gy },
    step,
  };
}

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

/** Every instance translation of an InstancedMesh, keyed nowhere — just the raw list. */
function instanceTranslations(mesh: THREE.InstancedMesh): THREE.Vector3[] {
  const out: THREE.Vector3[] = [];
  const m = new THREE.Matrix4();
  for (let i = 0; i < mesh.count; i++) {
    mesh.getMatrixAt(i, m);
    out.push(new THREE.Vector3().setFromMatrixPosition(m));
  }
  return out;
}

/** The cell a world position sits on (exact inverse of cellToWorld for cell centres). */
function cellOfWorld(v: THREE.Vector3, grid: GridGeometry): GridCell {
  return {
    x: Math.round(v.x / CELL + (grid.w - 1) / 2),
    y: Math.round(v.z / CELL + (grid.h - 1) / 2),
  };
}

/** Every instance's full transform, in placement order. */
function instanceTransforms(mesh: THREE.InstancedMesh): {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  scale: THREE.Vector3;
}[] {
  const out = [];
  const m = new THREE.Matrix4();
  for (let i = 0; i < mesh.count; i++) {
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    mesh.getMatrixAt(i, m);
    m.decompose(position, quaternion, scale);
    out.push({ position, quaternion, scale });
  }
  return out;
}

/**
 * THE FRAMING PROBE. The worst |ndc| any corner tower reaches through the scene's own
 * camera at a given aspect — the tower's whole silhouette, not its centre: the ±0.5-cell
 * footprint at both the flagstones and the tallest crown this env can generate.
 */
function worstCornerNdc(module: SceneModule, grid: GridGeometry, aspect: number): { x: number; y: number } {
  module.resize({ width: 1600, height: Math.round(1600 / aspect), dpr: 1 });
  const cam = module.camera;
  cam.updateMatrixWorld(true);
  cam.updateProjectionMatrix();
  const v = new THREE.Vector3();
  const truth: [number, number, number] = [0, 0, 0];
  let x = 0;
  let y = 0;
  for (const c of cornerCells(grid)) {
    cellToWorld(c.x, c.y, grid, truth);
    for (const h of [0, TOWER_HEIGHT_MAX]) {
      for (const dx of [-0.5, 0, 0.5]) {
        for (const dz of [-0.5, 0, 0.5]) {
          v.set(truth[0] + dx * CELL, h, truth[2] + dz * CELL).project(cam);
          x = Math.max(x, Math.abs(v.x));
          y = Math.max(y, Math.abs(v.y));
        }
      }
    }
  }
  return { x, y };
}

// ---------------------------------------------------------------------------

describe('procedural-maze / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('ProceduralMaze-cont-v0');
    expect(manifest.slug).toBe('procedural-maze');
    expect(manifest.accent).toBe('procedural-maze');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame stands on the only cells legal on EVERY seed', () => {
    // the generator clears the 25 room cells at odd interior coords on every seed, and
    // (1,1) / (9,9) are the corner rooms — free on every maze the env can produce
    const p = manifest.thumbFrame.pose;
    expect(p[0]).toBe(PROCMAZE_START.x);
    expect(p[1]).toBe(PROCMAZE_START.y);
    expect(p[2]).toBe(PROCMAZE_GOAL.x);
    expect(p[3]).toBe(PROCMAZE_GOAL.y);
    for (let seed = 90; seed < 130; seed++) {
      const grid = gridOf(seed);
      expect(isWall(grid, PROCMAZE_START.x, PROCMAZE_START.y)).toBe(false);
      expect(isWall(grid, PROCMAZE_GOAL.x, PROCMAZE_GOAL.y)).toBe(false);
    }
  });

  test('ships at least three canonical poses, all on FREE cells of their own seed', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    const st = proceduralMazeDecoder.initStatic(FIXTURE_SEED);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      expect(procMazeIsWall(st.words, f.pose[0] ?? -1, f.pose[1] ?? -1)).toBe(false);
      expect(procMazeIsWall(st.words, f.pose[2] ?? -1, f.pose[3] ?? -1)).toBe(false);
    }
  });

  test('the fixtures really do cover a dead end and a doorway', () => {
    const grid = gridOf(FIXTURE_SEED);
    // dead-end: the GOAL is a free cell walled on three sides — the deepest cul-de-sac
    // shape a perfect maze produces; the art states the goal, never the route to it
    const deadEnd = fixtureNamed('dead-end');
    const gx = deadEnd.pose[2] ?? -1;
    const gy = deadEnd.pose[3] ?? -1;
    expect(isWall(grid, gx, gy)).toBe(false);
    expect(wallNeighbours(grid, gx, gy)).toBe(3);
    // under-the-arch: the AGENT stands on a doorway cell `archCells()` nominates
    const arch = fixtureNamed('under-the-arch');
    const ax = arch.pose[0] ?? -1;
    const ay = arch.pose[1] ?? -1;
    expect(archCells(grid).some((a) => a.cell.x === ax && a.cell.y === ay)).toBe(true);
  });
});

describe('procedural-maze / the layout is read, never re-derived', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    for (const seed of SEEDS) {
      const geometry = geometryOf(seed);
      expect(geometry.grid).toBeDefined();
      expect(resolveGrid(geometry, seed) === geometry.grid).toBe(true);
    }
  });

  test("a placeholder geometry falls back to the DECODER's own maze, not an empty room", () => {
    // the harness's generic grid placeholder is a bare border ring with no carvedPath;
    // an 11×11 check alone would accept it and draw an empty courtyard
    const placeholder: EnvGeometry = {
      kind: 'grid',
      grid: { w: PROCMAZE_W, h: PROCMAZE_H, walls: new Uint8Array(PROCMAZE_W * PROCMAZE_H) },
    };
    for (const seed of SEEDS) {
      const grid = resolveGrid(placeholder, seed);
      const st = proceduralMazeDecoder.initStatic(seed);
      expect(grid.w).toBe(PROCMAZE_W);
      expect(grid.h).toBe(PROCMAZE_H);
      expect((grid.carvedPath?.length ?? 0) > 0).toBe(true);
      for (let y = 0; y < grid.h; y++) {
        for (let x = 0; x < grid.w; x++) {
          expect(isWall(grid, x, y)).toBe(procMazeIsWall(st.words, x, y));
        }
      }
    }
  });

  test('two seeds really are two different mazes (the whole premise)', () => {
    const a = gridOf(SEED);
    const b = gridOf(SEED_B);
    let differing = 0;
    for (let i = 0; i < a.walls.length; i++) if ((a.walls[i] ?? 0) !== (b.walls[i] ?? 0)) differing++;
    expect(differing).toBeGreaterThan(5);
  });
});

describe('procedural-maze / the wall-skinning system', () => {
  test('skinWallCell is a pure function of (walls, cell, seed)', () => {
    const grid = gridOf(SEED);
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (!isWall(grid, x, y)) continue;
        expect(skinWallCell(grid, x, y, SEED)).toEqual(skinWallCell(grid, x, y, SEED));
      }
    }
  });

  test('every block stands between the foundation course and the declared crown cap', () => {
    for (const seed of SEEDS) {
      const grid = gridOf(seed);
      let stumps = 0;
      let tall = 0;
      let walls = 0;
      for (let y = 0; y < grid.h; y++) {
        for (let x = 0; x < grid.w; x++) {
          if (!isWall(grid, x, y)) {
            expect(crownHeightAt(grid, x, y, seed)).toBe(0);
            continue;
          }
          walls++;
          const s = skinWallCell(grid, x, y, seed);
          expect(s.crown).toBeCloseTo(BASE_COURSE + s.upper + s.cap, 12);
          expect(s.crown).toBeGreaterThanOrEqual(BASE_COURSE);
          expect(s.crown).toBeLessThanOrEqual(MAX_CROWN + 1e-9);
          expect(s.moss).toBeGreaterThanOrEqual(0);
          expect(s.moss).toBeLessThanOrEqual(1);
          // a capstone can only sit on a second course
          if (s.cap > 0) expect(s.upper).toBeGreaterThan(0);
          if (s.upper === 0) stumps++;
          if (s.crown > 0.9) tall++;
        }
      }
      // the ruin must actually be RUINED: some cells collapsed to their foundation, some
      // still stand tall. A skin that made every block identical would pass every
      // alignment check in this file and look like a hedge maze.
      expect(walls).toBeGreaterThan(40);
      expect(stumps).toBeGreaterThan(2);
      expect(tall).toBeGreaterThan(5);
    }
  });

  test('the same cell is skinned differently by different seeds', () => {
    const a = gridOf(SEED);
    const b = gridOf(SEED_B);
    let differing = 0;
    for (let y = 0; y < PROCMAZE_H; y++) {
      for (let x = 0; x < PROCMAZE_W; x++) {
        if (!isWall(a, x, y) || !isWall(b, x, y)) continue;
        if (skinWallCell(a, x, y, SEED).crown !== skinWallCell(b, x, y, SEED_B).crown) differing++;
      }
    }
    expect(differing).toBeGreaterThan(20);
  });

  test('moss reads the walls array: hemmed-in blocks are damper than exposed ones', () => {
    const grid = gridOf(SEED);
    let hemmed = 0;
    let hemmedSum = 0;
    let open = 0;
    let openSum = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (!isWall(grid, x, y)) continue;
        const s = skinWallCell(grid, x, y, SEED);
        const n = wallNeighbours(grid, x, y);
        if (n >= 3) {
          hemmed++;
          hemmedSum += s.moss;
        } else if (n <= 1) {
          open++;
          openSum += s.moss;
        }
      }
    }
    expect(hemmed).toBeGreaterThan(4);
    expect(open).toBeGreaterThan(2);
    expect(hemmedSum / hemmed).toBeGreaterThan(openSum / open);
  });

  test('the courses are laid on wall cells ONLY, and never on the corner towers', () => {
    for (const seed of SEEDS) {
      const grid = gridOf(seed);
      const skin = ruinSkin(TOKENS.palettes.noon, TOKENS);
      const corners = cornerCells(grid);
      const cornerKeys = new Set(corners.map((c) => c.y * grid.w + c.x));
      const courses = buildCourses(grid, seed, skin, TOKENS.palettes.noon, 'hero', (x, y) =>
        cornerKeys.has(y * grid.w + x),
      );
      expect(courses.upperCells.length).toBeGreaterThan(10);
      for (const list of [courses.upperCells, courses.capCells]) {
        for (const c of list) {
          expect(isWall(grid, c.x, c.y)).toBe(true);
          expect(cornerKeys.has(c.y * grid.w + c.x)).toBe(false);
        }
      }
      // and each instance really is standing on its own cell, at the right course height
      const upper = instanceTranslations(courses.upperMesh);
      expect(upper.length).toBe(courses.upperCells.length);
      const truth: [number, number, number] = [0, 0, 0];
      for (let i = 0; i < upper.length; i++) {
        const cell = courses.upperCells[i] as GridCell;
        const p = upper[i] as THREE.Vector3;
        cellToWorld(cell.x, cell.y, grid, truth);
        expect(p.x).toBeCloseTo(truth[0], 10);
        expect(p.z).toBeCloseTo(truth[2], 10);
        expect(p.y).toBeCloseTo(BASE_COURSE, 10);
      }
      const caps = instanceTranslations(courses.capMesh);
      for (let i = 0; i < caps.length; i++) {
        const cell = courses.capCells[i] as GridCell;
        const p = caps[i] as THREE.Vector3;
        const s = skinWallCell(grid, cell.x, cell.y, seed);
        cellToWorld(cell.x, cell.y, grid, truth);
        expect(p.x).toBeCloseTo(truth[0], 10);
        expect(p.z).toBeCloseTo(truth[2], 10);
        // instance matrices are f32: cell centres are exact halves, a course height is not
        expect(p.y).toBeCloseTo(BASE_COURSE + s.upper, 6);
      }
      courses.dispose();
    }
  });

  test('a doorway is READ from the walls, never assumed', () => {
    for (const seed of SEEDS) {
      const grid = gridOf(seed);
      for (const a of archCells(grid)) {
        expect(isWall(grid, a.cell.x, a.cell.y)).toBe(false);
        const n = isWall(grid, a.cell.x, a.cell.y - 1);
        const s = isWall(grid, a.cell.x, a.cell.y + 1);
        const w = isWall(grid, a.cell.x - 1, a.cell.y);
        const e = isWall(grid, a.cell.x + 1, a.cell.y);
        if (a.eastWest) {
          // walled north and south: the passage runs east–west, the arch spans north–south
          expect(n && s).toBe(true);
          expect(w || e).toBe(false);
        } else {
          // walled east and west: the passage runs north–south, the arch spans east–west
          expect(w && e).toBe(true);
          expect(n || s).toBe(false);
        }
      }
    }
  });

  test('the four corner cells are masonry on every seed (the one fixed fact)', () => {
    for (let seed = 90; seed < 130; seed++) {
      const grid = gridOf(seed);
      for (const c of cornerCells(grid)) expect(isWall(grid, c.x, c.y)).toBe(true);
    }
  });

  test('the forest floor is identically flat under the whole ruin', () => {
    const h = makeForestFloorHeight(SEED);
    for (let i = 0; i <= 40; i++) {
      for (let j = 0; j <= 40; j++) {
        const x = -RUIN_HALF + (2 * RUIN_HALF * i) / 40;
        const z = -RUIN_HALF + (2 * RUIN_HALF * j) / 40;
        expect(h(x, z)).toBe(0);
      }
    }
    let wander = 0;
    for (const p of [-18, -14, 14, 18]) wander = Math.max(wander, Math.abs(h(p, p)));
    expect(wander).toBeGreaterThan(0.05);
  });
});

describe('procedural-maze / the built scene matches the walls array, on both seeds', () => {
  test('one foundation block per wall cell, each exactly on its cell', async () => {
    for (const seed of SEEDS) {
      const { module } = await live('hero', seed);
      const grid = gridOf(seed);
      const field = (module.scene.getObjectByName('oakCellWallMesh') as THREE.InstancedMesh | undefined) ?? null;
      expect(field).not.toBeNull();
      if (!field) return;
      expect(field.count).toBe(wallCellCount(grid));

      const seen = new Set<number>();
      const truth: [number, number, number] = [0, 0, 0];
      for (const p of instanceTranslations(field)) {
        const cell = cellOfWorld(p, grid);
        // every block stands on a cell the ENV calls a wall...
        expect(isWall(grid, cell.x, cell.y)).toBe(true);
        cellToWorld(cell.x, cell.y, grid, truth);
        expect(p.x).toBeCloseTo(truth[0], 10);
        expect(p.z).toBeCloseTo(truth[2], 10);
        expect(p.y).toBeCloseTo(0, 10);
        seen.add(cell.y * grid.w + cell.x);
      }
      // ...and every wall cell got exactly one, the corner towers included (their generic
      // instance is collapsed to zero scale, never moved off its cell)
      expect(seen.size).toBe(wallCellCount(grid));
      for (let y = 0; y < grid.h; y++) {
        for (let x = 0; x < grid.w; x++) {
          if (isWall(grid, x, y)) expect(seen.has(y * grid.w + x)).toBe(true);
        }
      }
      module.dispose();
    }
  });

  test('the corner towers and the doorways stand on their own cells, on both seeds', async () => {
    for (const seed of SEEDS) {
      const { module } = await live('hero', seed);
      const grid = gridOf(seed);
      const truth: [number, number, number] = [0, 0, 0];
      const corners = cornerCells(grid);
      for (let i = 0; i < corners.length; i++) {
        const c = corners[i] as GridCell;
        const root = module.scene.getObjectByName(`ruinTowerRoot${i}`);
        expect(root).toBeDefined();
        if (!root) continue;
        root.updateWorldMatrix(true, false);
        const w = new THREE.Vector3().setFromMatrixPosition(root.matrixWorld);
        cellToWorld(c.x, c.y, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.y).toBeCloseTo(0, 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
      }

      const arches = archCells(grid);
      for (let i = 0; i < Math.min(arches.length, 9); i++) {
        const a = arches[i];
        const root = module.scene.getObjectByName(`ruinArchRoot${i}`);
        expect(root).toBeDefined();
        if (!root || !a) continue;
        root.updateWorldMatrix(true, false);
        const w = new THREE.Vector3().setFromMatrixPosition(root.matrixWorld);
        cellToWorld(a.cell.x, a.cell.y, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
        // the doorway faces the axis its flanking wall cells run along
        const mesh = module.scene.getObjectByName(`ruinArch${i}`);
        expect(mesh?.rotation.y ?? -1).toBeCloseTo(a.eastWest ? 0 : Math.PI / 2, 10);
      }
      module.dispose();
    }
  });

  test('alignment is exact on the SECOND seed too, with that seed\'s own cells', async () => {
    const grid = gridOf(SEED_B);
    const free: SceneFixture[] = [];
    for (let y = 0; y < grid.h && free.length < 5; y++) {
      for (let x = 0; x < grid.w && free.length < 5; x++) {
        if (isWall(grid, x, y)) continue;
        free.push(poseFixture(`b${x}-${y}`, x, y, PROCMAZE_GOAL.x, PROCMAZE_GOAL.y, free.length));
      }
    }
    const module = createScene();
    const report = await assertAligned(module, free, { geometry: geometryOf(SEED_B), seed: SEED_B, tier: 'hero' });
    for (const name of ['wisp', 'wisp-anchor', 'sun-shaft', 'corner-tower', 'ruin-arch']) {
      expect(report.worst.get(name) ?? 1).toBeLessThan(1e-9);
    }
    module.dispose();
  });
});

describe('procedural-maze / the frame holds the whole ruin', () => {
  // CANON LESSON 21, made into a gate. "Camera distance is arithmetic": the four broken
  // towers are the silhouette every seed shares AND the corner probe, and the first pass
  // hung the south-east one clean out of the picture (ndc.y −1.17 at 16:9 AND 4:3) under a
  // comment claiming the opposite. A composition claim nobody can measure is folklore.
  test('every corner tower is inside the hero frame at 16:9 and 4:3, on both seeds', async () => {
    for (const seed of SEEDS) {
      const { module } = await live('hero', seed);
      const grid = gridOf(seed);
      for (const aspect of [16 / 9, 4 / 3]) {
        const worst = worstCornerNdc(module, grid, aspect);
        // inside the frame, with margin — not "just on the edge"
        expect(worst.x).toBeLessThan(0.94);
        expect(worst.y).toBeLessThan(0.94);
      }
      module.dispose();
    }
  });

  test('the thumb holds all 11 × 11 at a SQUARE crop, towers included', async () => {
    // the wall-view contract: at aspect 1 the horizontal half-angle collapses to 20°, and
    // a distance that frames the field at 16:9 slices the west tower in half at 1:1
    for (const seed of SEEDS) {
      const { module } = await live('thumb', seed);
      const grid = gridOf(seed);
      for (const aspect of [1, 16 / 9, 4 / 3]) {
        const worst = worstCornerNdc(module, grid, aspect);
        expect(worst.x).toBeLessThan(0.94);
        expect(worst.y).toBeLessThan(0.94);
      }
      module.dispose();
    }
  });

  test('the ruin still FILLS its frame — a fix that just zoomed out would be a regression', async () => {
    const { module } = await live('hero', SEED);
    const grid = gridOf(SEED);
    const worst = worstCornerNdc(module, grid, 16 / 9);
    // the near/far corners must still reach most of the way to the edges
    expect(worst.y).toBeGreaterThan(0.72);
    module.dispose();
  });
});

describe('procedural-maze / the masonry stands on the foundation it was promised', () => {
  // The scene stacks courses 2 and 3 at heights it computes from `skinWallCell`, so the
  // kit's foundation MUST finish at exactly BASE_COURSE. `CellFieldStyle.jitter` scales
  // the wall block's height as well as its tone (kit/grid: scale.y = 1 + (nw-0.5)*jitter*2),
  // and passing 0.09 made half the ruin's second courses float and the other half sink.
  test('every foundation block is exactly BASE_COURSE tall, on both seeds', async () => {
    for (const seed of SEEDS) {
      const { module } = await live('hero', seed);
      const grid = gridOf(seed);
      const walls = module.scene.getObjectByName('oakCellWallMesh') as THREE.InstancedMesh;
      expect(walls.count).toBe(wallCellCount(grid));
      for (const t of instanceTransforms(walls)) {
        // unit y-scale on a wallHeight = BASE_COURSE block: the top face is the promise
        expect(t.scale.y).toBe(1);
        expect(t.position.y * 1 + BASE_COURSE * t.scale.y).toBeCloseTo(BASE_COURSE, 12);
      }
      module.dispose();
    }
  });

  test('the second course sits ON the foundation, not over or in it', async () => {
    for (const seed of SEEDS) {
      const { module } = await live('hero', seed);
      const grid = gridOf(seed);
      const walls = module.scene.getObjectByName('oakCellWallMesh') as THREE.InstancedMesh;
      const upper = module.scene.getObjectByName('ruinUpperCourse') as THREE.InstancedMesh;
      const foundationTop = new Map<number, number>();
      for (const t of instanceTransforms(walls)) {
        const cell = cellOfWorld(t.position, grid);
        foundationTop.set(cell.y * grid.w + cell.x, BASE_COURSE * t.scale.y);
      }
      let checked = 0;
      for (const t of instanceTransforms(upper)) {
        const cell = cellOfWorld(t.position, grid);
        const top = foundationTop.get(cell.y * grid.w + cell.x);
        expect(top).toBeDefined();
        // zero daylight, zero burial — the block's underside IS the foundation's top face
        expect(t.position.y).toBeCloseTo(top ?? -1, 6);
        checked++;
      }
      expect(checked).toBeGreaterThan(10);
      module.dispose();
    }
  });

  test('crownHeightAt is the height of the stone that is actually DRAWN', async () => {
    for (const seed of SEEDS) {
      const { module } = await live('hero', seed);
      const grid = gridOf(seed);
      const skin = ruinSkin(TOKENS.palettes.noon, TOKENS);
      const corners = new Set(cornerCells(grid).map((c) => c.y * grid.w + c.x));
      const courses = buildCourses(grid, seed, skin, TOKENS.palettes.noon, 'hero', (x, y) =>
        corners.has(y * grid.w + x),
      );
      const caps = instanceTransforms(courses.capMesh);
      for (let i = 0; i < caps.length; i++) {
        const c = courses.capCells[i] as GridCell;
        const t = caps[i] as { position: THREE.Vector3; scale: THREE.Vector3 };
        const s = skinWallCell(grid, c.x, c.y, seed);
        // the slab is a unit box scaled by its own thickness, so its top face is
        // position.y + scale.y — and THAT is the crown the rest of the scene is promised
        expect(t.scale.y).toBeCloseTo(s.cap, 6);
        expect(t.position.y + t.scale.y).toBeCloseTo(crownHeightAt(grid, c.x, c.y, seed), 6);
      }
      // and where no lintel fell, the second course IS the crown
      const uppers = instanceTransforms(courses.upperMesh);
      let bare = 0;
      for (let i = 0; i < uppers.length; i++) {
        const c = courses.upperCells[i] as GridCell;
        const s = skinWallCell(grid, c.x, c.y, seed);
        if (s.cap > 0) continue;
        const t = uppers[i] as { position: THREE.Vector3; scale: THREE.Vector3 };
        expect(t.position.y + t.scale.y).toBeCloseTo(crownHeightAt(grid, c.x, c.y, seed), 6);
        bare++;
      }
      expect(bare).toBeGreaterThan(3);
      courses.dispose();
      module.dispose();
    }
  });

  test('the sunbeam puts a flagstone back EXACTLY as `floorTintFor` painted it', async () => {
    const { module } = await live('hero', SEED);
    const grid = gridOf(SEED);
    const skin = ruinSkin(TOKENS.palettes.noon, TOKENS);
    const floor = module.scene.getObjectByName('oakCellFloorMesh') as THREE.InstancedMesh;
    const field = (module.scene.getObjectByName('ruinCells') as THREE.Group | undefined) ?? null;
    expect(field).not.toBeNull();
    const painted = new THREE.Color();
    const restored = new THREE.Color();
    // walk the beam over a run of free cells, then read the tile it left behind
    const visited: GridCell[] = [];
    for (let y = 1; y < grid.h - 1 && visited.length < 6; y++) {
      for (let x = 1; x < grid.w - 1 && visited.length < 6; x++) {
        if (!isWall(grid, x, y)) visited.push({ x, y });
      }
    }
    for (const c of visited) module.update(toFrameUpdate(poseFixture('beam', 1, 1, c.x, c.y)));
    // the last cell keeps the warm tint; every earlier one is back to its own colour
    const positions = instanceTranslations(floor);
    for (let i = 0; i < visited.length - 1; i++) {
      const c = visited[i] as GridCell;
      const index = positions.findIndex((p) => {
        const cell = cellOfWorld(p, grid);
        return cell.x === c.x && cell.y === c.y;
      });
      expect(index).toBeGreaterThanOrEqual(0);
      floor.getColorAt(index, restored);
      floorTintFor(grid, skin, c.x, c.y, SEED, painted);
      expect(restored.r).toBeCloseTo(painted.r, 6);
      expect(restored.g).toBeCloseTo(painted.g, 6);
      expect(restored.b).toBeCloseTo(painted.b, 6);
    }
    module.dispose();
  });
});

describe('procedural-maze / the ivy holds on to stone', () => {
  test('every strand is rooted on a real ledge and hangs down a real face', async () => {
    for (const seed of SEEDS) {
      const { module } = await live('hero', seed);
      const grid = gridOf(seed);
      const skin = ruinSkin(TOKENS.palettes.noon, TOKENS);
      const corners = new Set(cornerCells(grid).map((c) => c.y * grid.w + c.x));
      const courses = buildCourses(grid, seed, skin, TOKENS.palettes.noon, 'hero', (x, y) =>
        corners.has(y * grid.w + x),
      );
      const ivy = module.scene.getObjectByName('ruinIvy') as THREE.InstancedMesh;
      expect(ivy.count).toBeGreaterThan(5);
      expect(ivy.count).toBe(courses.ivyCells.length);
      const truth: [number, number, number] = [0, 0, 0];
      let onStandingCourse = 0;
      for (let i = 0; i < ivy.count; i++) {
        const t = instanceTransforms(ivy)[i] as {
          position: THREE.Vector3;
          scale: THREE.Vector3;
        };
        const entry = courses.ivyCells[i] as { cell: GridCell; strand: number };
        const s = skinWallCell(grid, entry.cell.x, entry.cell.y, seed);
        const ledge = ivyLedge(s, entry.strand);
        const a = (((s.ivyFace + entry.strand * 2) % 4) * Math.PI) / 2;
        cellToWorld(entry.cell.x, entry.cell.y, grid, truth);
        // the outward distance along the face normal (the lateral jitter is tangential)
        const radial =
          (t.position.x - truth[0]) * Math.sin(a) + (t.position.z - truth[2]) * Math.cos(a);
        expect(t.position.y).toBeCloseTo(ledge.y, 6);
        // ...and it is exactly one cling clear of the stone THAT EXISTS at that height
        expect(radial).toBeCloseTo(blockHalfWidthAt(s, ledge.y) + IVY_CLING, 6);
        // both ends stay on the block: never above its crown, never through the floor
        expect(ledge.y).toBeLessThanOrEqual(s.crown + 1e-9);
        expect(t.position.y - t.scale.y).toBeGreaterThanOrEqual(IVY_FLOOR_GAP - 1e-6);
        if (ledge.y > BASE_COURSE) onStandingCourse++;
      }
      // the ruin's ivy is not all swept down onto the base course: the standing stone
      // keeps its creepers, which is the whole reason the strands exist
      expect(onStandingCourse).toBeGreaterThan(ivy.count / 3);
      courses.dispose();
      module.dispose();
    }
  });

  test('a block only carries ivy when it has a ledge to carry it with', () => {
    for (const seed of SEEDS) {
      const grid = gridOf(seed);
      for (let y = 0; y < grid.h; y++) {
        for (let x = 0; x < grid.w; x++) {
          if (!isWall(grid, x, y)) continue;
          const s = skinWallCell(grid, x, y, seed);
          if (s.ivy === 0) continue;
          expect(s.upper).toBeGreaterThan(0);
          for (let strand = 0; strand < s.ivy; strand++) {
            const ledge = ivyLedge(s, strand);
            expect(blockHalfWidthAt(s, ledge.y)).toBeGreaterThan(0);
            expect(ledge.drop).toBeGreaterThan(0);
          }
        }
      }
    }
  });
});

describe('procedural-maze / idle life', () => {
  test('every bird flies nose-first — the whole flock, both directions round the circle', () => {
    const birds = makeBirds(SEED, TOKENS.palettes.noon.hemiGround, 'hero');
    const before = instanceTransforms(birds.mesh).map((t) => t.position.clone());
    birds.setTime(0.25);
    const after = instanceTransforms(birds.mesh);
    const heading = new THREE.Vector3();
    const travel = new THREE.Vector3();
    let checked = 0;
    for (let i = 0; i < after.length; i++) {
      const t = after[i] as { position: THREE.Vector3; quaternion: THREE.Quaternion };
      const from = before[i] as THREE.Vector3;
      // the bird geometry's nose is local +z
      heading.set(0, 0, 1).applyQuaternion(t.quaternion).setY(0).normalize();
      travel.copy(t.position).sub(from).setY(0);
      expect(travel.length()).toBeGreaterThan(1e-3);
      // > 0.9, not just > 0: half the flock circles the other way, and the operator
      // precedence bug that flew them tail-first scored exactly −1
      expect(heading.dot(travel.normalize())).toBeGreaterThan(0.9);
      checked++;
    }
    expect(checked).toBe(7);
    birds.dispose();
  });
});

describe('procedural-maze / alignment', () => {
  test('every probe lands on ground truth at hero tier — exactly, not within tolerance', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(SEED), seed: SEED, tier: 'hero' });
    // these roots carry POSITION ONLY, so the error is 0, not "small"
    expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('sun-shaft') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('corner-tower') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('ruin-arch') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(SEED), seed: SEED, tier });
      expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
      expect(report.worst.get('sun-shaft') ?? 1).toBeLessThan(1e-9);
      module.dispose();
    }
  });

  test('the wisp lands on EVERY free cell of this seed\'s maze', async () => {
    const { module } = await live();
    const grid = gridOf(SEED);
    const truth: [number, number, number] = [0, 0, 0];
    let visited = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (isWall(grid, x, y)) continue;
        module.update(toFrameUpdate(poseFixture(`cell${x}-${y}`, x, y, 9, 9, visited)));
        const w = worldOf(module, 'wisp-anchor');
        cellToWorld(x, y, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.y).toBeCloseTo(0, 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
        const orb = worldOf(module, 'wisp');
        expect(orb.y).toBeCloseTo(TOKENS.wispHeight, 10);
        visited++;
      }
    }
    expect(visited).toBeGreaterThan(30);
    module.dispose();
  });

  test('the sunbeam follows a respawning goal onto every free cell, dead ends included', async () => {
    const { module } = await live();
    const grid = gridOf(SEED);
    const truth: [number, number, number] = [0, 0, 0];
    let sealed = 0;
    let deadEnds = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (isWall(grid, x, y)) continue;
        module.update(toFrameUpdate(poseFixture('goalmove', 1, 1, x, y)));
        const w = worldOf(module, 'sun-shaft');
        cellToWorld(x, y, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.y).toBeCloseTo(0, 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
        if (wallNeighbours(grid, x, y) === 4) sealed++;
        if (wallNeighbours(grid, x, y) === 3) deadEnds++;
      }
    }
    // the generator is a perfect maze: every free cell is reachable, so a sealed
    // pocket (4 walls around a free cell) can never exist — but dead ends do, and
    // the beam lands on them the same as anywhere else
    expect(sealed).toBe(0);
    expect(deadEnds).toBeGreaterThan(0);
    module.dispose();
  });
});

describe('procedural-maze / the hop is bounded, and a teleport is a hard cut', () => {
  test('a hop stays inside the segment between two cells and lands on truth', async () => {
    const { module, ctx } = await live();
    const grid = gridOf(SEED);
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('a', 2, 2, 9, 9, 0)));
    cellToWorld(2, 2, grid, from);
    cellToWorld(3, 2, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('b', 3, 2, 9, 9, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(poseFixture('b', 3, 2, 9, 9, 1), { scrubbed: false }));
      const w = worldOf(module, 'wisp-anchor');
      expect(w.x).toBeGreaterThanOrEqual(Math.min(from[0], to[0]) - 1e-6);
      expect(w.x).toBeLessThanOrEqual(Math.max(from[0], to[0]) + 1e-6);
      expect(w.z).toBeCloseTo(from[2], 6);
      expect(w.y).toBeGreaterThanOrEqual(-1e-6);
    }
    const landed = worldOf(module, 'wisp-anchor');
    expect(landed.x).toBeCloseTo(to[0], 10);
    expect(landed.y).toBeCloseTo(0, 10);
    module.dispose();
  });

  test('a `teleport` event cuts the wisp across the maze with no glide at all', async () => {
    const { module, ctx } = await live();
    const grid = gridOf(SEED);
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(4);
    module.update(toFrameUpdate(poseFixture('a', 9, 8, 9, 9, 100), { scrubbed: false }));
    // the real completion batch: goal + teleport + done, with BOTH ends redrawn
    ctx.setTime(4.125);
    module.update(
      toFrameUpdate(poseFixture('respawned', 5, 2, 2, 5, 101), {
        scrubbed: false,
        prev: toFrameState(poseFixture('a', 9, 8, 9, 9, 100)),
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(5, 2, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(0, 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    module.dispose();
  });

  test('a scrub snaps the wisp with no glide at all', async () => {
    const { module, ctx } = await live();
    const grid = gridOf(SEED);
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(4);
    module.update(toFrameUpdate(poseFixture('a', 1, 1, 9, 9, 0), { scrubbed: false }));
    module.update(toFrameUpdate(poseFixture('far', 9, 9, 1, 1, 4000), { scrubbed: true }));
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(9, 9, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    expect(w.y).toBeCloseTo(0, 10);
    module.dispose();
  });
});

describe('procedural-maze / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('ruins-start')));
    let count = 0;
    module.scene.traverse(() => count++);
    for (let i = 0; i < 60; i++) {
      module.update(toFrameUpdate(fixtures[i % fixtures.length] as SceneFixture, { scrubbed: false }));
    }
    let after = 0;
    module.scene.traverse(() => after++);
    expect(after).toBe(count);
    module.dispose();
  });

  test('celebrations fire ONLY from frame.events', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    expect(pool).not.toBeNull();
    if (!pool) return;

    // standing exactly ON the goal cell with no event: nothing may fire
    ctx.setTime(1);
    module.update(toFrameUpdate(poseFixture('onGoal', 9, 9, 9, 9, 100), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    ctx.setTime(2);
    module.update(
      toFrameUpdate(poseFixture('scored', 5, 2, 2, 5, 101), {
        scrubbed: false,
        prev: toFrameState(poseFixture('before', 9, 8, 9, 9, 100)),
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    // `done` fired on that same batch and it is the SUCCESS: no sigh may have run
    expect(pool.pulse.sigh).toBe(0);
    module.dispose();
  });

  test("the bloom happens on the PREVIOUS frame's goal cell — where the +1 was earned", async () => {
    const { module, ctx } = await live();
    const grid = gridOf(SEED);
    ctx.setTime(3);
    module.update(toFrameUpdate(poseFixture('run', 9, 8, 9, 9, 200), { scrubbed: false }));
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(poseFixture('scored', 5, 2, 2, 5, 201), {
        scrubbed: false,
        prev: toFrameState(poseFixture('run', 9, 8, 9, 9, 200)),
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    // the celebration point light rides the bloom position: (9,9), NOT the new agent
    // cell (5,2) and NOT the new goal cell (2,5)
    const light = module.scene.getObjectByName('oakPoint2');
    expect(light).toBeDefined();
    if (!light) return;
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(9, 9, grid, truth);
    expect(light.position.x).toBeCloseTo(truth[0], 10);
    expect(light.position.z).toBeCloseTo(truth[2], 10);
    expect((light as THREE.PointLight).intensity).toBeGreaterThan(0);
    // and the sunbeam is already on its new cell, opening up from nothing
    const shaft = worldOf(module, 'sun-shaft');
    cellToWorld(2, 5, grid, truth);
    expect(shaft.x).toBeCloseTo(truth[0], 10);
    expect(shaft.z).toBeCloseTo(truth[2], 10);
    const decor = module.scene.getObjectByName('sunShaftDecor');
    expect(decor?.scale.x ?? 1).toBeLessThan(0.2);
    module.dispose();
  });

  test('a scrub clears every in-flight beat and settles the new atrium', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(poseFixture('scored', 5, 2, 2, 5, 300), {
        scrubbed: false,
        prev: toFrameState(poseFixture('before', 9, 8, 9, 9, 299)),
        events: [{ type: 'goal' }, { type: 'teleport' }],
      }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('ruins-start'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    const decor = module.scene.getObjectByName('sunShaftDecor');
    expect(decor?.scale.x ?? 0).toBeCloseTo(1, 6);
    module.dispose();
  });

  test('readGoalCell reads derived first and falls back to the raw pose', () => {
    const grid = gridOf(SEED);
    const out: GridCell = { x: 0, y: 0 };
    expect(readGoalCell(toFrameState(poseFixture('d', 1, 1, 4, 7)), grid, out)).toEqual({ x: 4, y: 7 });
    const bare = toFrameState({ name: 'bare', pose: [1, 1, 6, 3, 0, 0, 0, 0, 0] });
    expect(readGoalCell(bare, grid, out)).toEqual({ x: 6, y: 3 });
    // out-of-range poses clamp rather than escaping the playfield
    const wild = toFrameState({ name: 'wild', pose: [1, 1, 99, -4, 0, 0, 0, 0, 0] });
    expect(readGoalCell(wild, grid, out)).toEqual({ x: grid.w - 1, y: 0 });
  });

  test('the growth and envelope curves are pure, bounded and land on 1', () => {
    expect(growth(-Infinity)).toBe(1);
    expect(growth(Number.NaN)).toBe(1);
    expect(growth(0)).toBe(0);
    expect(growth(99)).toBe(1);
    for (let i = 0; i <= 50; i++) {
      const g = growth((i / 50) * 0.8);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(1.05);
    }
    expect(envelope(-1, 1.5, 2.4)).toBe(0);
    expect(envelope(99, 1.5, 2.4)).toBe(0);
    expect(envelope(0, 1.5, 2.4)).toBeCloseTo(1, 10);
    // the linear tail exists so no flourish ends on a pop
    expect(envelope(1.4999, 1.5, 2.4)).toBeLessThan(1e-4);
  });
});

describe('procedural-maze / honesty', () => {
  test('the carved solution path exists ONLY on the debug layer', async () => {
    const { module } = await live();
    let carved: THREE.Object3D | null = null;
    let carvedCount = 0;
    module.scene.traverse((o) => {
      if (o.name === 'oakDebugCarvedPath') {
        carved = o;
        carvedCount++;
      }
    });
    // exactly one, and it belongs to kit/debug — nothing in the art draws the solution
    expect(carvedCount).toBe(1);
    const path = carved as THREE.Object3D | null;
    expect(path).not.toBeNull();
    if (!path) return;
    const debugLayer = new THREE.Layers();
    debugLayer.set(DEBUG_LAYER);
    expect(path.layers.test(debugLayer)).toBe(true);
    // and while debug is off the camera cannot see that layer at all
    expect(module.camera.layers.test(debugLayer)).toBe(false);
    module.dispose();
  });

  test('setDebug toggles the ground-truth overlay onto the scene camera', async () => {
    const { module } = await live();
    module.setDebug(true);
    module.update(toFrameUpdate(fixtureNamed('dead-end')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'ruinGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(DEBUG_LAYER);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'ruinGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });
});

describe('procedural-maze / lifecycle', () => {
  test('init -> dispose round-trips clean three times', async () => {
    const report = await assertInitDisposeRoundTrip(createScene, {
      geometry: geometryOf(SEED),
      seed: SEED,
      tier: 'hero',
      fixtures,
      cycles: 3,
    });
    expect(report.census.length).toBe(3);
    expect((report.census[0]?.geometries.size ?? 0) > 0).toBe(true);
    expect((report.census[0]?.textures.size ?? 0) > 0).toBe(true);
  });

  test('every tier round-trips clean too', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      await assertInitDisposeRoundTrip(createScene, {
        geometry: geometryOf(SEED),
        seed: SEED,
        tier,
        fixtures,
        cycles: 2,
      });
    }
  });

  test('a second seed round-trips clean as well (a different maze is a different graph)', async () => {
    await assertInitDisposeRoundTrip(createScene, {
      geometry: geometryOf(SEED_B),
      seed: SEED_B,
      tier: 'hero',
      fixtures,
      cycles: 2,
    });
  });

  test('the camera stays a clamped 3/4 aerial over the ruin', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('ruins-start')));
    const cam = module.camera;
    expect(cam.position.y).toBeGreaterThan(6);
    expect(Math.hypot(cam.position.x, cam.position.z)).toBeGreaterThan(RUIN_HALF);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.y).toBeLessThan(-0.5); // a genuine aerial, never a horizon shot
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    module.dispose();
  });

  test('the start pose really is the env\'s deterministic init', () => {
    const start = fixtureNamed('ruins-start');
    expect(start.pose[0]).toBe(PROCMAZE_START.x);
    expect(start.pose[1]).toBe(PROCMAZE_START.y);
    expect(start.pose[2]).toBe(PROCMAZE_GOAL.x);
    expect(start.pose[3]).toBe(PROCMAZE_GOAL.y);
  });
});

describe('procedural-maze / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    // The wall view is the whole gallery on one page: every scene there is a thumb, and
    // the cap is what keeps 16 of them at 60 fps. Measured with the ground-truth overlay
    // ON, because that is the state review screenshots are taken in, and across every
    // fixture, because the busiest legal state is the one that has to fit.
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'procedural-maze (thumb, overlay on)');
    module.dispose();
  });
});
