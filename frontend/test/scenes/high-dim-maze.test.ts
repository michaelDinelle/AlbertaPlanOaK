/// <reference types="bun" />
// The Slot Canyons (HighDimMaze-cont-v0) — the accuracy + lifecycle self-check.
//
// This mirrors four-rooms.test.ts's shape (the grid canon) and adds what THIS env has to
// prove, none of which FourRooms exercises:
//
//   * the wisp probe is `wisp.avatar`, because this scene calls `Wisp.face()` — so the
//     probe must be exact AND idempotent while the body is mid-turn;
//   * `pose[4]` (lastDir) maps to a facing with no guessing;
//   * a DASH moves two cells in one 8 Hz step, and the hop must stay on the corridor
//     segment and land exactly on the true cell;
//   * the +1 blooms on the cell the agent SCORED on — the previous frame's goal — because
//     this env rejection-redraws BOTH the agent and the goal, so neither of their new
//     cells means anything;
//   * `done` here is always the +1, so the scene must never sigh;
//   * the two arches sit on the env's own gap cells, oriented off the flanking walls.

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { cellToWorld, isWall } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import { yawFromGridDelta } from '@kit/wisp';
import {
  HIGHDIMMAZE_DASH,
  HIGHDIMMAZE_GAPS,
  HIGHDIMMAZE_GOAL,
  HIGHDIMMAZE_H,
  HIGHDIMMAZE_STAY,
  HIGHDIMMAZE_START,
  HIGHDIMMAZE_W,
  highDimMazeDecoder,
  highDimMazeIsWall,
} from '@sim/envs/highdimmaze';
import type { EnvGeometry, GridCell } from '@sim/types';

import createScene, {
  dampAngle,
  envelope,
  facingDelta,
  facingYaw,
  growth,
  isDashSegment,
  nearness,
} from '../../src/scenes/high-dim-maze/index';
import { fixtures } from '../../src/scenes/high-dim-maze/fixtures';
import { manifest } from '../../src/scenes/high-dim-maze/manifest';
import {
  ARCH_NOTCH,
  ARCH_OPENING,
  CAMERA_AZIMUTH,
  CANYON_HALF,
  FIN_HEIGHT_NORTH,
  FIN_HEIGHT_SOUTH,
  RIM_HEIGHT_NORTH,
  RIM_HEIGHT_SOUTH,
  archAxis,
  archHeight,
  archNotch,
  archYaw,
  canyonSkin,
  isRimCell,
  makeMesaHeight,
  resolveGaps,
  resolveGrid,
  sandFallSpots,
  wallDistance,
  wallHeightAt,
  wallNominalAt,
} from '../../src/scenes/high-dim-maze/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 104; // the demo run's first HighDimMaze lane
/** 81 interior cells − 18 fin cells + the 2 gaps */
const FREE_CELLS = 65;

function geometryOf(): EnvGeometry {
  return highDimMazeDecoder.geometry(highDimMazeDecoder.initStatic(SEED), SEED);
}

async function live(tier: QualityTier = 'hero'): Promise<{ module: SceneModule; ctx: StubSceneContext }> {
  const module = createScene();
  const ctx = makeSceneContext({ geometry: geometryOf(), seed: SEED, tier });
  await module.init(ctx);
  return { module, ctx };
}

function fixtureNamed(name: string): SceneFixture {
  const f = fixtures.find((x) => x.name === name);
  if (!f) throw new Error(`fixture "${name}" missing`);
  return f;
}

function poseFixture(
  name: string,
  x: number,
  y: number,
  gx: number,
  gy: number,
  lastDir = 1,
  action = 0,
  step = 0,
): SceneFixture {
  return {
    name,
    pose: [x, y, gx, gy, lastDir, 0, 0, 0, 0],
    ctrl: [action, 0, 0, 0, 0, 0],
    derived: { x, y, gx, gy, lastDir },
    step,
  };
}

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

// ---------------------------------------------------------------------------

describe('high-dim-maze / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('HighDimMaze-cont-v0');
    expect(manifest.slug).toBe('high-dim-maze');
    expect(manifest.accent).toBe('high-dim-maze');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a legal canyon state', () => {
    const p = manifest.thumbFrame.pose;
    expect(highDimMazeIsWall(p[0] ?? -1, p[1] ?? -1)).toBe(false);
    expect(highDimMazeIsWall(p[2] ?? -1, p[3] ?? -1)).toBe(false);
    // lastDir is a cardinal action index, always
    expect(p[4]).toBeGreaterThanOrEqual(0);
    expect(p[4]).toBeLessThanOrEqual(3);
  });

  test('ships at least three canonical poses, all on FREE cells', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      expect(highDimMazeIsWall(f.pose[0] ?? -1, f.pose[1] ?? -1)).toBe(false);
      expect(highDimMazeIsWall(f.pose[2] ?? -1, f.pose[3] ?? -1)).toBe(false);
      const dir = f.pose[4] ?? -1;
      expect(dir).toBeGreaterThanOrEqual(0);
      expect(dir).toBeLessThanOrEqual(3);
    }
  });

  test('the fixtures stand on BOTH gap cells and cover stay + dash', () => {
    for (const gap of HIGHDIMMAZE_GAPS) {
      const found = fixtures.some((f) => f.pose[0] === gap.x && f.pose[1] === gap.y);
      expect(found).toBe(true);
    }
    const actions = new Set(fixtures.map((f) => f.ctrl?.[0] ?? -1));
    expect(actions.has(HIGHDIMMAZE_STAY)).toBe(true);
    expect(actions.has(HIGHDIMMAZE_DASH)).toBe(true);
  });

  test('the start fixture really is the env\'s deterministic init', () => {
    const start = fixtureNamed('canyon-start');
    expect(start.pose[0]).toBe(HIGHDIMMAZE_START.x);
    expect(start.pose[1]).toBe(HIGHDIMMAZE_START.y);
    expect(start.pose[2]).toBe(HIGHDIMMAZE_GOAL.x);
    expect(start.pose[3]).toBe(HIGHDIMMAZE_GOAL.y);
    expect(start.pose[4]).toBe(1); // oak_fleet_init sets lastDir = down
  });
});

describe('high-dim-maze / the layout is read, never re-derived', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry) === geometry.grid).toBe(true);
  });

  test('a placeholder geometry falls back to the DECODER\'s own walls', () => {
    const grid = resolveGrid({ kind: 'grid' });
    expect(grid.w).toBe(HIGHDIMMAZE_W);
    expect(grid.h).toBe(HIGHDIMMAZE_H);
    let free = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        expect(isWall(grid, x, y)).toBe(highDimMazeIsWall(x, y));
        if (!isWall(grid, x, y)) free++;
      }
    }
    expect(free).toBe(FREE_CELLS);
  });

  test('the gaps come from the decoder, and both are free cells inside a fin', () => {
    const grid = resolveGrid(geometryOf());
    const gaps = resolveGaps(grid);
    expect(gaps.length).toBe(HIGHDIMMAZE_GAPS.length);
    for (const gap of gaps) {
      expect(isWall(grid, gap.x, gap.y)).toBe(false);
      // a gap is a hole in a fin: rock on two opposite sides
      const vertical = isWall(grid, gap.x, gap.y - 1) && isWall(grid, gap.x, gap.y + 1);
      const horizontal = isWall(grid, gap.x - 1, gap.y) && isWall(grid, gap.x + 1, gap.y);
      expect(vertical || horizontal).toBe(true);
      expect(isRimCell(grid, gap.x, gap.y)).toBe(false);
    }
  });

  test('arch orientation is READ from the flanking walls, never assumed', () => {
    const grid = resolveGrid(geometryOf());
    for (const gap of resolveGaps(grid)) {
      // both of this maze's fins run north-south, so both arches span z and are yawed
      expect(archAxis(grid, gap)).toBe('alongZ');
      expect(archYaw(archAxis(grid, gap))).toBeCloseTo(Math.PI / 2, 12);
    }
    // and the OTHER branch is reachable: a synthetic east-west fin yields the other axis
    const w = 3;
    const h = 3;
    const walls = new Uint8Array(w * h);
    walls[1 * w + 0] = 1; // (0,1) wall
    walls[1 * w + 2] = 1; // (2,1) wall — a wall running along x with a hole at (1,1)
    const synthetic = { w, h, walls };
    expect(archAxis(synthetic, { x: 1, y: 1 })).toBe('alongX');
    expect(archYaw('alongX')).toBe(0);
  });

  test('cliff heights are a stationary function of the cell, tapering south', () => {
    const grid = resolveGrid(geometryOf());
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        const h = wallNominalAt(grid, x, y);
        if (!isWall(grid, x, y)) {
          expect(h).toBe(0);
          expect(wallHeightAt(grid, x, y)).toBe(0);
          continue;
        }
        expect(h).toBeGreaterThan(0);
        const lo = isRimCell(grid, x, y) ? RIM_HEIGHT_SOUTH : FIN_HEIGHT_SOUTH;
        const hi = isRimCell(grid, x, y) ? RIM_HEIGHT_NORTH : FIN_HEIGHT_NORTH;
        expect(h).toBeGreaterThanOrEqual(lo - 1e-9);
        expect(h).toBeLessThanOrEqual(hi + 1e-9);
        // the built height is that line times the notch, and nothing else
        expect(wallHeightAt(grid, x, y)).toBeCloseTo(h * archNotch(grid, x, y), 12);
      }
    }
    // the near (south) rim really is lower than the far one — that is what keeps the
    // front row of the grid out of its own occlusion
    expect(wallNominalAt(grid, 5, grid.h - 1)).toBeLessThan(wallNominalAt(grid, 5, 0));
    // and the interior fins stand taller than the rim beside them
    expect(wallNominalAt(grid, 3, 5)).toBeGreaterThan(wallNominalAt(grid, 0, 5));
  });

  test('the fin ramps DOWN into every gap, and only along its own line', () => {
    const grid = resolveGrid(geometryOf());
    const gaps = resolveGaps(grid);
    for (const gap of gaps) {
      const alongZ = archAxis(grid, gap) === 'alongZ';
      for (let k = 1; k <= ARCH_NOTCH.length; k++) {
        for (const dir of [-1, 1]) {
          const x = alongZ ? gap.x : gap.x + dir * k;
          const y = alongZ ? gap.y + dir * k : gap.y;
          if (!isWall(grid, x, y) || isRimCell(grid, x, y)) continue;
          // exactly the published factor, and monotonically rising away from the gap
          expect(archNotch(grid, x, y)).toBeCloseTo(ARCH_NOTCH[k - 1] ?? 1, 12);
          expect(wallHeightAt(grid, x, y)).toBeLessThan(wallNominalAt(grid, x, y));
        }
      }
      // the ramp is a NECK, not a crater: it never touches a cell off the fin's line
      const off = alongZ ? { x: gap.x + 1, y: gap.y } : { x: gap.x, y: gap.y + 1 };
      if (isWall(grid, off.x, off.y)) expect(archNotch(grid, off.x, off.y)).toBe(1);
    }
    // every cell three or more out along the fin keeps its full line
    expect(archNotch(grid, 7, 1 + ARCH_NOTCH.length + 1)).toBe(1);
    expect(archNotch(grid, 3, 5)).toBe(1);
    // the outer rim is never notched — its silhouette is the canyon's edge
    expect(archNotch(grid, 7, 0)).toBe(1);
  });

  test('every arch clears the wisp and stands proud of the fin it bridges', () => {
    const grid = resolveGrid(geometryOf());
    for (const gap of resolveGaps(grid)) {
      const top = archHeight(grid, gap);
      const jamb = Math.max(wallHeightAt(grid, gap.x, gap.y - 1), wallHeightAt(grid, gap.x, gap.y + 1));
      expect(top).toBeGreaterThan(jamb);
      // headroom: the orb rides at wispHeight and the opening must clear it
      expect(Math.min(ARCH_OPENING, top - 0.4)).toBeGreaterThan(TOKENS.wispHeight);
    }
  });

  test('the sun/shade read is a scan of the walls array: one sunlit spine', () => {
    const grid = resolveGrid(geometryOf());
    let open = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (isWall(grid, x, y)) {
          expect(wallDistance(grid, x, y)).toBe(0);
          continue;
        }
        const d = wallDistance(grid, x, y);
        expect(d).toBeGreaterThanOrEqual(1);
        // the two narrow slots are 2 cells wide, so a wall is ALWAYS beside you there
        if (d > 1) {
          open++;
          expect(x).toBe(5); // only the middle column of the wide slot gets clearance
        }
      }
    }
    expect(open).toBe(7); // (5,2) .. (5,8)
  });

  test('the sand-fall curtains land on cliff lips, deterministically', () => {
    const grid = resolveGrid(geometryOf());
    const a = sandFallSpots(grid, SEED, 4);
    const b = sandFallSpots(grid, SEED, 4);
    expect(a.length).toBe(4);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    expect(sandFallSpots(grid, SEED, 0).length).toBe(0);
    for (const spot of a) {
      expect(spot.top).toBeGreaterThan(0);
      // inside the canyon footprint, never out on the desert
      expect(Math.abs(spot.x)).toBeLessThanOrEqual(CANYON_HALF);
      expect(Math.abs(spot.z)).toBeLessThanOrEqual(CANYON_HALF);
    }
  });

  test('the desert is identically flat under the whole canyon', () => {
    const h = makeMesaHeight(SEED);
    for (let i = 0; i <= 40; i++) {
      for (let j = 0; j <= 40; j++) {
        const x = -CANYON_HALF + (2 * CANYON_HALF * i) / 40;
        const z = -CANYON_HALF + (2 * CANYON_HALF * j) / 40;
        expect(h(x, z)).toBe(0);
      }
    }
    let wander = 0;
    for (const p of [-24, -19, 19, 24]) wander = Math.max(wander, Math.abs(h(p, p)));
    expect(wander).toBeGreaterThan(0.05);
  });

  test('the skin is terracotta, and the oasis is the one cool note in it', () => {
    const skin = canyonSkin(TOKENS.palettes.afternoon, TOKENS);
    expect(skin.strata.length).toBe(4);
    const hexes = new Set(skin.strata.map((c) => c.getHex()));
    expect(hexes.size).toBe(4); // no two beds share a colour
    // the caprock is the brightest rock and the bottom bed the darkest
    const cap = skin.strata[3];
    const floorBed = skin.strata[0];
    expect(cap).toBeDefined();
    expect(floorBed).toBeDefined();
    if (cap && floorBed) {
      const lum = (c: THREE.Color): number => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
      expect(lum(cap)).toBeGreaterThan(lum(floorBed));
      // canon: under a 2.7 key a terracotta albedo must stay well under the bloom line
      expect(lum(cap)).toBeLessThan(0.72);
    }
    // water reads cool against every rock in the canyon
    expect(skin.water.b).toBeGreaterThan(skin.stone.b);
    expect(skin.stone.r).toBeGreaterThan(skin.stone.b);
  });
});

describe('high-dim-maze / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    // these roots carry POSITION ONLY — they should be exact, not merely within 1e-3
    expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('oasis') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('arch-0') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('arch-1') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
      expect(report.worst.get('oasis') ?? 1).toBeLessThan(1e-9);
      module.dispose();
    }
  });

  test('the wisp lands on EVERY free cell of the canyon', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    let visited = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (isWall(grid, x, y)) continue;
        module.update(toFrameUpdate(poseFixture(`cell${x}-${y}`, x, y, 9, 9, 1, 0, visited)));
        const w = worldOf(module, 'wisp');
        cellToWorld(x, y, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.y).toBeCloseTo(0, 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
        visited++;
      }
    }
    expect(visited).toBe(FREE_CELLS);
    module.dispose();
  });

  test('the oasis follows a respawning goal onto every free cell', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (isWall(grid, x, y)) continue;
        module.update(toFrameUpdate(poseFixture('goalmove', 1, 1, x, y)));
        const w = worldOf(module, 'oasis');
        cellToWorld(x, y, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.y).toBeCloseTo(0, 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
      }
    }
    module.dispose();
  });

  test('the arches sit on the env\'s own gap cells', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    const gaps = resolveGaps(grid);
    for (let i = 0; i < gaps.length; i++) {
      const gap = gaps[i];
      if (!gap) continue;
      const root = module.scene.getObjectByName(`canyonArchRoot${i}`);
      expect(root).toBeDefined();
      if (!root) continue;
      root.updateWorldMatrix(true, false);
      const w = new THREE.Vector3().setFromMatrixPosition(root.matrixWorld);
      cellToWorld(gap.x, gap.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
      // the arch mesh under it is yawed by the READ orientation, not by a constant
      const mesh = module.scene.getObjectByName(`canyonArch${i}`);
      expect(mesh?.rotation.y ?? -1).toBeCloseTo(archYaw(archAxis(grid, gap)), 12);
    }
    module.dispose();
  });
});

describe('high-dim-maze / the facing is pose[4], read not guessed', () => {
  test('lastDir maps to the env\'s own four cardinal deltas', () => {
    expect(facingDelta(0)).toEqual({ dx: 0, dy: -1 }); // up
    expect(facingDelta(1)).toEqual({ dx: 0, dy: 1 }); // down
    expect(facingDelta(2)).toEqual({ dx: -1, dy: 0 }); // left
    expect(facingDelta(3)).toEqual({ dx: 1, dy: 0 }); // right
    // out-of-range values wrap the way the kernel's `lastDir & 3` does
    expect(facingDelta(7)).toEqual(facingDelta(3));
    expect(facingDelta(-1)).toEqual(facingDelta(3));
  });

  test('the yaw is the kit\'s own grid convention', () => {
    for (let dir = 0; dir < 4; dir++) {
      const d = facingDelta(dir);
      expect(facingYaw(dir)).toBeCloseTo(yawFromGridDelta(d.dx, d.dy), 12);
    }
    expect(facingYaw(1)).toBeCloseTo(0, 12); // down = +z = yaw 0
    expect(facingYaw(3)).toBeCloseTo(Math.PI / 2, 12); // right = +x
  });

  test('the scarf turn damps on the shortest arc, exactly like the wisp body', () => {
    // a turn from just under +PI to just over -PI must go the SHORT way
    const out = dampAngle(Math.PI - 0.1, -Math.PI + 0.1, 1 / 60);
    expect(out).toBeGreaterThan(Math.PI - 0.1);
    expect(dampAngle(0, 0, 1 / 60)).toBe(0);
    // and it converges, never overshoots
    let a = 0;
    for (let i = 0; i < 400; i++) a = dampAngle(a, 1.2, 1 / 60);
    expect(a).toBeCloseTo(1.2, 6);
  });

  test('the wisp really is turned by the frame (and the probe survives it)', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    const body = module.scene.getObjectByName('canyonWisp');
    expect(body).toBeDefined();
    for (const dir of [0, 1, 2, 3]) {
      ctx.setTime(20 + dir);
      // a scrubbed frame snaps the facing, so the body yaw is the target exactly
      module.update(toFrameUpdate(poseFixture('turn', 5, 5, 9, 9, dir), { scrubbed: true }));
      const avatar = worldOf(module, 'wisp');
      cellToWorld(5, 5, grid, truth);
      expect(avatar.x).toBeCloseTo(truth[0], 10);
      expect(avatar.z).toBeCloseTo(truth[2], 10);
      // the scarf hangs off the avatar and carries the facing
      const scarf = module.scene.getObjectByName('canyonScarf');
      expect(scarf).toBeDefined();
      expect(scarf?.rotation.y ?? NaN).toBeCloseTo(facingYaw(dir), 6);
    }
    module.dispose();
  });
});

describe('high-dim-maze / the hop is bounded, never a path plan', () => {
  test('a single step stays inside the two cells and lands on truth', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('a', 1, 1, 9, 9, 1, 1, 0)));
    cellToWorld(1, 1, grid, from);
    cellToWorld(1, 2, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('b', 1, 2, 9, 9, 1, 1, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(poseFixture('b', 1, 2, 9, 9, 1, 1, 1), { scrubbed: false }));
      const w = worldOf(module, 'wisp');
      expect(w.z).toBeGreaterThanOrEqual(Math.min(from[2], to[2]) - 1e-6);
      expect(w.z).toBeLessThanOrEqual(Math.max(from[2], to[2]) + 1e-6);
      expect(w.x).toBeCloseTo(from[0], 6);
      expect(w.y).toBeGreaterThanOrEqual(-1e-6);
    }
    const landed = worldOf(module, 'wisp');
    expect(landed.z).toBeCloseTo(to[2], 10);
    expect(landed.y).toBeCloseTo(0, 10);
    module.dispose();
  });

  test('a DASH crosses two cells inside one control step, on the corridor line', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];
    cellToWorld(5, 5, grid, from);
    cellToWorld(5, 3, grid, to); // two cells north — every cell between them is free

    ctx.setTime(30);
    module.update(toFrameUpdate(poseFixture('pre', 5, 5, 2, 2, 0, 0, 0)));
    ctx.setTime(30);
    module.update(
      toFrameUpdate(poseFixture('dash', 5, 3, 2, 2, 0, HIGHDIMMAZE_DASH, 1), { scrubbed: false }),
    );
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(30 + (k / 8) * 0.125);
      module.update(
        toFrameUpdate(poseFixture('dash', 5, 3, 2, 2, 0, HIGHDIMMAZE_DASH, 1), { scrubbed: false }),
      );
      const w = worldOf(module, 'wisp');
      // the dash is collinear along lastDir, so the glide never leaves the corridor
      expect(w.x).toBeCloseTo(from[0], 6);
      expect(w.z).toBeGreaterThanOrEqual(Math.min(from[2], to[2]) - 1e-6);
      expect(w.z).toBeLessThanOrEqual(Math.max(from[2], to[2]) + 1e-6);
    }
    // one control step later the two-cell glide has landed EXACTLY on the true cell
    const landed = worldOf(module, 'wisp');
    expect(landed.x).toBeCloseTo(to[0], 10);
    expect(landed.z).toBeCloseTo(to[2], 10);
    expect(landed.y).toBeCloseTo(0, 10);
    // and the wake lies on the segment it crossed
    const wake = module.scene.getObjectByName('wispSandWake');
    expect(wake).toBeDefined();
    expect(wake?.visible ?? false).toBe(true);
    expect(wake?.position.x ?? NaN).toBeCloseTo(from[0], 10);
    expect(wake?.position.z ?? NaN).toBeCloseTo((from[2] + to[2]) / 2, 10);
    module.dispose();
  });

  test('a STAY settles: the ring shows, the wisp does not move', async () => {
    const { module, ctx } = await live();
    ctx.setTime(3);
    module.update(toFrameUpdate(poseFixture('stay', 8, 4, 1, 7, 1, HIGHDIMMAZE_STAY, 10)));
    const ring = module.scene.getObjectByName('wispSettleRing');
    expect(ring?.visible ?? false).toBe(true);
    const before = worldOf(module, 'wisp');
    ctx.setTime(3.5);
    module.update(toFrameUpdate(poseFixture('stay', 8, 4, 1, 7, 1, HIGHDIMMAZE_STAY, 11), { scrubbed: false }));
    const after = worldOf(module, 'wisp');
    expect(after.x).toBeCloseTo(before.x, 10);
    expect(after.z).toBeCloseTo(before.z, 10);
    // a normal step puts the ring away again — it is a pure function of the action
    module.update(toFrameUpdate(poseFixture('move', 8, 5, 1, 7, 1, 1, 12), { scrubbed: false }));
    expect(module.scene.getObjectByName('wispSettleRing')?.visible ?? true).toBe(false);
    module.dispose();
  });

  test('a scrub snaps the wisp with no glide at all', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(4);
    module.update(toFrameUpdate(poseFixture('a', 1, 1, 9, 9), { scrubbed: false }));
    module.update(toFrameUpdate(poseFixture('far', 9, 9, 1, 1, 2, 0, 4000), { scrubbed: true }));
    const w = worldOf(module, 'wisp');
    cellToWorld(9, 9, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    expect(w.y).toBeCloseTo(0, 10);
    module.dispose();
  });
});

describe('high-dim-maze / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('canyon-start')));
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
    module.update(toFrameUpdate(poseFixture('onGoal', 9, 9, 9, 9, 1, 0, 100), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    // the decoder's real batch on a +1: goal + teleport + done, all in one step
    ctx.setTime(2);
    module.update(
      toFrameUpdate(poseFixture('scored', 2, 4, 6, 7, 1, 0, 101), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    // `done` here is only ever the +1 — the scene must NEVER sigh
    expect(pool.pulse.sigh).toBe(0);
    module.dispose();
  });

  test('the bloom lands on the cell the agent SCORED on, not on either new cell', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    // step N: the agent is beside the oasis at (9,9)
    ctx.setTime(3);
    const before = poseFixture('run', 8, 9, 9, 9, 3, 3, 200);
    module.update(toFrameUpdate(before, { scrubbed: false }));
    // step N+1: it entered (9,9), scored, and BOTH it and the goal were redrawn
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(poseFixture('scored', 2, 4, 6, 7, 3, 3, 201), {
        scrubbed: false,
        prev: toFrameUpdate(before).state,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const light = module.scene.getObjectByName('oakPoint2');
    expect(light).toBeDefined();
    if (!light) return;
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(9, 9, grid, truth); // the OLD goal — where the +1 happened
    expect(light.position.x).toBeCloseTo(truth[0], 10);
    expect(light.position.z).toBeCloseTo(truth[2], 10);
    expect((light as THREE.PointLight).intensity).toBeGreaterThan(0);
    // the new oasis is already on its new cell, welling up from nothing
    const oasis = worldOf(module, 'oasis');
    cellToWorld(6, 7, grid, truth);
    expect(oasis.x).toBeCloseTo(truth[0], 10);
    expect(oasis.z).toBeCloseTo(truth[2], 10);
    const alcove = module.scene.getObjectByName('oasisGrow');
    expect(alcove?.scale.x ?? 1).toBeLessThan(0.2);
    module.dispose();
  });

  test('the teleport is a hard cut: the wisp never glides across the maze', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(5);
    module.update(toFrameUpdate(poseFixture('run', 8, 9, 9, 9, 3, 3, 300), { scrubbed: false }));
    ctx.setTime(5.125);
    module.update(
      toFrameUpdate(poseFixture('respawn', 1, 2, 6, 7, 3, 3, 301), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    // no easing may cross a teleport: the wisp is ON the redrawn cell in the same frame
    const w = worldOf(module, 'wisp');
    cellToWorld(1, 2, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    expect(w.y).toBeCloseTo(0, 10);
    // and the gust it might have been mid-dash through is gone
    expect(module.scene.getObjectByName('wispSpeedLines')?.visible ?? true).toBe(false);
    module.dispose();
  });

  test('the grow-in bounce actually reaches the alcove (it is not clamped away)', async () => {
    // The bug this locks: `growth()` overshoots 1 on purpose, and `Oasis.setState` ran it
    // through clamp01 — so the alcove could never scale past 1 and the documented "wells up
    // with a little bounce" beat rendered as a plain ease. Drive the real scene through the
    // real event and watch the real node.
    const { module, ctx } = await live();
    const alcove = module.scene.getObjectByName('oasisGrow');
    expect(alcove).toBeDefined();
    ctx.setTime(100);
    module.update(
      toFrameUpdate(poseFixture('scored', 2, 4, 6, 7, 1, 0, 500), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(alcove?.scale.x ?? 1).toBeLessThan(0.2); // the new oasis starts from nothing
    let peak = 0;
    for (let i = 1; i <= 40; i++) {
      ctx.setTime(100 + (i / 40) * 0.8);
      module.update(toFrameUpdate(poseFixture('after', 2, 4, 6, 7, 1, 0, 500 + i), { scrubbed: false }));
      peak = Math.max(peak, alcove?.scale.x ?? 0);
    }
    expect(peak).toBeGreaterThan(1.02); // it really overshoots on screen
    expect(peak).toBeLessThan(1.12); // and never becomes a cartoon
    // …and it settles to exactly its cell-sized self
    ctx.setTime(102);
    module.update(toFrameUpdate(poseFixture('settled', 2, 4, 6, 7, 1, 0, 560), { scrubbed: false }));
    expect(alcove?.scale.x ?? 0).toBeCloseTo(1, 9);
    module.dispose();
  });

  test('a scrub clears every in-flight beat and settles the new oasis', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(poseFixture('scored', 2, 4, 6, 7, 1, 0, 400), {
        scrubbed: false,
        events: [{ type: 'goal' }],
      }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('canyon-start'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    const alcove = module.scene.getObjectByName('oasisGrow');
    expect(alcove?.scale.x ?? 0).toBeCloseTo(1, 6);
    expect(module.scene.getObjectByName('wispSandWake')?.visible ?? true).toBe(false);
    module.dispose();
  });

  test('the envelopes are pure, bounded and land on their resting value', () => {
    expect(growth(-Infinity)).toBe(1);
    expect(growth(Number.NaN)).toBe(1);
    expect(growth(0)).toBe(0);
    expect(growth(99)).toBe(1);
    let peak = 0;
    for (let i = 0; i <= 50; i++) {
      const g = growth((i / 50) * 0.8);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(1.1);
      peak = Math.max(peak, g);
    }
    // THE BOUNCE IS REAL, NOT DECORATIVE MATHS. The first pass's overshoot peaked at 1.008
    // — under a thousandth of a cell on the alcove — and was then clamped away entirely by
    // the actor. A documented beat has to be big enough to see.
    expect(peak).toBeGreaterThan(1.03);
    // …and it still lands exactly on its resting value, with no step at the window's end
    expect(growth(0.8 - 1e-6)).toBeCloseTo(1, 3);
    expect(envelope(-Infinity, 1, 2)).toBe(0);
    expect(envelope(Number.NaN, 1, 2)).toBe(0);
    expect(envelope(0, 1, 2)).toBe(1);
    expect(envelope(2, 1, 2)).toBe(0);
    // nearness is bounded and rises as the wisp closes in
    const far = nearness({ x: 1, y: 1 }, { x: 9, y: 9 });
    const near = nearness({ x: 9, y: 8 }, { x: 9, y: 9 });
    expect(far).toBe(0);
    expect(near).toBeGreaterThan(far);
    expect(near).toBeLessThanOrEqual(1);
  });
});

describe('high-dim-maze / the dash draws only the step the env can make', () => {
  test('the segment guard accepts one- and two-cell axis hops and nothing else', () => {
    expect(isDashSegment(0, 0, 0, 1)).toBe(true);
    expect(isDashSegment(0, 0, 0, -2)).toBe(true);
    expect(isDashSegment(0, 0, 2, 0)).toBe(true);
    expect(isDashSegment(0, 0, 0, 0)).toBe(false); // a dash that moved nothing
    expect(isDashSegment(0, 0, 1, 1)).toBe(false); // diagonal — the env cannot make one
    expect(isDashSegment(0, 0, 0, 3)).toBe(false); // three cells — nor one of these
    expect(isDashSegment(-3, 2, -3, 4)).toBe(true);
  });

  test('a DASH that moves ZERO cells draws no speed lines (and no stale ones)', async () => {
    // `gw_hdm_step` stops a dash at the first wall, so action 5 at (1,1) with lastDir = up
    // is an ordinary legal transition that advances nothing. The bug this locks: the scene
    // refreshed the segment only when the cell changed, so the PREVIOUS dash's from/to were
    // still in the scratch vectors and this frame painted its speed lines across the maze.
    const { module, ctx } = await live();
    const lines = module.scene.getObjectByName('wispSpeedLines');
    const wake = module.scene.getObjectByName('wispSandWake');
    expect(lines).toBeDefined();

    // a real two-cell dash south→north up the middle slot, five cells from (1,1)
    ctx.setTime(20);
    module.update(toFrameUpdate(poseFixture('pre', 5, 5, 2, 2, 0, 0, 0)));
    ctx.setTime(20.05);
    module.update(
      toFrameUpdate(poseFixture('dash', 5, 3, 2, 2, 0, HIGHDIMMAZE_DASH, 1), { scrubbed: false }),
    );
    expect(lines?.visible ?? false).toBe(true);

    // walk away, then a BLOCKED dash at (1,1): action 5, cell unchanged
    ctx.setTime(20.4);
    module.update(toFrameUpdate(poseFixture('walk', 1, 1, 2, 2, 0, 0, 2), { scrubbed: false }));
    ctx.setTime(20.6);
    module.update(
      toFrameUpdate(poseFixture('blocked', 1, 1, 2, 2, 0, HIGHDIMMAZE_DASH, 3), { scrubbed: false }),
    );
    expect(lines?.visible ?? true).toBe(false);

    // the WAKE is allowed to outlive its dash — sand hangs — but it stays anchored on the
    // segment that dash actually crossed, never re-anchored onto the blocked one
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];
    cellToWorld(5, 5, grid, from);
    cellToWorld(5, 3, grid, to);
    if (wake?.visible) {
      expect(wake.position.x).toBeCloseTo(from[0], 6);
      expect(wake.position.z).toBeCloseTo((from[2] + to[2]) / 2, 6);
    }
    module.dispose();
  });

  test('a non-collinear "dash" from the fixture loop draws nothing at all', async () => {
    // the harness's fixture fallback plays canonical poses back to back, so two frames of
    // it can differ by a diagonal jump. The env cannot make that step; the scene must not
    // draw one either.
    const { module, ctx } = await live();
    const lines = module.scene.getObjectByName('wispSpeedLines');
    ctx.setTime(40);
    module.update(toFrameUpdate(poseFixture('a', 5, 5, 2, 2, 0, 0, 0)));
    ctx.setTime(40.05);
    module.update(
      toFrameUpdate(poseFixture('diagonal', 4, 4, 2, 2, 0, HIGHDIMMAZE_DASH, 1), { scrubbed: false }),
    );
    expect(lines?.visible ?? true).toBe(false);
    const wake = module.scene.getObjectByName('wispSandWake');
    expect(wake?.visible ?? true).toBe(false);
    module.dispose();
  });

  test('a scrub and a teleport both wipe the segment, not just the clock', async () => {
    const { module, ctx } = await live();
    const lines = module.scene.getObjectByName('wispSpeedLines');
    const wake = module.scene.getObjectByName('wispSandWake');
    ctx.setTime(60);
    module.update(toFrameUpdate(poseFixture('pre', 5, 5, 2, 2, 0, 0, 0)));
    ctx.setTime(60.05);
    module.update(
      toFrameUpdate(poseFixture('dash', 5, 3, 2, 2, 0, HIGHDIMMAZE_DASH, 1), { scrubbed: false }),
    );
    expect(wake?.visible ?? false).toBe(true);
    // a scrub across the maze: nothing in flight may survive it
    ctx.setTime(60.1);
    module.update(toFrameUpdate(poseFixture('elsewhere', 9, 2, 2, 2, 1, 0, 900), { scrubbed: true }));
    expect(lines?.visible ?? true).toBe(false);
    expect(wake?.visible ?? true).toBe(false);
    // and a dash on the very next frame draws its OWN segment, from the new cell
    ctx.setTime(60.2);
    module.update(
      toFrameUpdate(poseFixture('dash2', 9, 4, 2, 2, 1, HIGHDIMMAZE_DASH, 901), { scrubbed: false }),
    );
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];
    cellToWorld(9, 2, grid, from);
    cellToWorld(9, 4, grid, to);
    expect(wake?.position.x ?? NaN).toBeCloseTo(from[0], 6);
    expect(wake?.position.z ?? NaN).toBeCloseTo((from[2] + to[2]) / 2, 6);
    module.dispose();
  });
});

describe('high-dim-maze / the agent is visible on every cell it can stand on', () => {
  // THE SCENE'S HARDEST ACCURACY CLAIM, and the one a still cannot fake: a 1.3-unit cliff on
  // a 1-unit cell will hide a 0.35-unit orb from the wrong camera, and the frame that does
  // it looks perfectly composed. So the occlusion is MEASURED here — real Raycaster, real
  // wall instances, real strata, real arches — for every free cell, at the composed camera
  // and at every corner of the rig's own limits.
  const OCCLUDER = /oakCellWallMesh|canyonStrata|canyonArch\d|canyonDesert/;
  /** must mirror index.ts's OrbitRig limits + composed pose */
  const CAM = { azimuth: CAMERA_AZIMUTH, polar: 0.86, distance: 17, targetY: 0.62 };
  const LIMITS = { az: [CAMERA_AZIMUTH - 0.04, CAMERA_AZIMUTH + 0.04], polar: [0.74, 0.88], dist: [17, 24] };

  function camAt(azimuth: number, polar: number, distance: number): THREE.Vector3 {
    const sp = Math.sin(polar);
    return new THREE.Vector3(
      distance * sp * Math.sin(azimuth),
      CAM.targetY + distance * Math.cos(polar),
      distance * sp * Math.cos(azimuth),
    );
  }

  function occludersOf(module: SceneModule): THREE.Object3D[] {
    module.scene.updateMatrixWorld(true);
    const out: THREE.Object3D[] = [];
    module.scene.traverse((o) => {
      if ((o as THREE.Mesh).isMesh && OCCLUDER.test(o.name)) out.push(o);
    });
    return out;
  }

  /** what fraction of the wisp orb a camera at `cam` can see on this cell */
  function visibleFraction(
    occluders: THREE.Object3D[],
    cam: THREE.Vector3,
    cell: GridCell,
    grid: { w: number; h: number },
    ring = 8,
  ): number {
    const w: [number, number, number] = [0, 0, 0];
    cellToWorld(cell.x, cell.y, grid, w);
    const R = 0.165 * 0.8;
    const y = TOKENS.wispHeight;
    const pts: THREE.Vector3[] = [new THREE.Vector3(w[0], y, w[2])];
    for (let i = 0; i < ring; i++) {
      const a = (i / ring) * Math.PI * 2;
      pts.push(new THREE.Vector3(w[0] + Math.cos(a) * R, y + Math.sin(a) * R, w[2]));
      pts.push(new THREE.Vector3(w[0] + Math.cos(a) * R, y, w[2] + Math.sin(a) * R));
    }
    const ray = new THREE.Raycaster();
    const dir = new THREE.Vector3();
    let seen = 0;
    for (const p of pts) {
      dir.subVectors(p, cam);
      const dist = dir.length();
      ray.set(cam, dir.normalize());
      ray.near = 0.01;
      ray.far = dist - 0.02;
      if (ray.intersectObjects(occluders, false).length === 0) seen++;
    }
    return seen / pts.length;
  }

  function freeCells(grid: { w: number; h: number; walls: Uint8Array }): GridCell[] {
    const out: GridCell[] = [];
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) if (!isWall(grid, x, y)) out.push({ x, y });
    }
    return out;
  }

  test('no cliff, strata bed or arch hides the wisp from the composed camera', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('canyon-start')));
    const grid = resolveGrid(geometryOf());
    const occluders = occludersOf(module);
    expect(occluders.length).toBeGreaterThan(2); // walls + strata + both arches at least

    const cam = module.camera.position.clone();
    // the rig really did compose the frame this test is auditing
    expect(cam.distanceTo(new THREE.Vector3(0, CAM.targetY, 0))).toBeCloseTo(CAM.distance, 3);
    expect(Math.atan2(cam.x, cam.z)).toBeCloseTo(CAM.azimuth, 6);

    const cells = freeCells(grid);
    expect(cells.length).toBe(FREE_CELLS);
    let worst = 1;
    let worstCell = '';
    let total = 0;
    for (const cell of cells) {
      const f = visibleFraction(occluders, cam, cell, grid);
      total += f;
      if (f < worst) {
        worst = f;
        worstCell = `(${cell.x},${cell.y})`;
      }
    }
    // NOT ONE hidden cell. The first pass hid 8 of these 65 — the whole corridor west of
    // the x = 3 fin plus the high arch — and the source claimed the camera hid nothing.
    expect(`${worstCell}:${worst.toFixed(2)}`).not.toContain(':0.00');
    expect(worst).toBeGreaterThan(0.1);
    expect(total / cells.length).toBeGreaterThan(0.8);
    module.dispose();
  });

  test(
    'and none of the rig\'s own limits can orbit the agent out of sight',
    async () => {
      const { module } = await live();
      module.update(toFrameUpdate(fixtureNamed('canyon-start')));
      const grid = resolveGrid(geometryOf());
      const occluders = occludersOf(module);
      const cells = freeCells(grid);
      for (const azimuth of LIMITS.az) {
        for (const polar of LIMITS.polar) {
          for (const distance of LIMITS.dist) {
            const cam = camAt(azimuth, polar, distance);
            for (const cell of cells) {
              const f = visibleFraction(occluders, cam, cell, grid, 4);
              expect(`az${azimuth} pol${polar} d${distance} (${cell.x},${cell.y}) ${f.toFixed(2)}`).not.toContain(' 0.00');
            }
          }
        }
      }
      module.dispose();
    },
    60_000,
  );

  test('the thumb tier is framed by the same law', async () => {
    const { module } = await live('thumb');
    module.update(toFrameUpdate(fixtureNamed('canyon-start')));
    const grid = resolveGrid(geometryOf());
    const occluders = occludersOf(module);
    const cam = module.camera.position.clone();
    for (const cell of freeCells(grid)) {
      const f = visibleFraction(occluders, cam, cell, grid);
      expect(`thumb (${cell.x},${cell.y}) ${f.toFixed(2)}`).not.toContain(' 0.00');
    }
    module.dispose();
  });
});

describe('high-dim-maze / lifecycle', () => {
  test('init -> dispose round-trips clean three times', async () => {
    const report = await assertInitDisposeRoundTrip(createScene, {
      geometry: geometryOf(),
      seed: SEED,
      tier: 'hero',
      fixtures,
      cycles: 3,
    });
    expect(report.census.length).toBe(3);
    expect((report.census[0]?.geometries.size ?? 0) > 0).toBe(true);
  });

  test('every tier round-trips clean too', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      await assertInitDisposeRoundTrip(createScene, {
        geometry: geometryOf(),
        seed: SEED,
        tier,
        fixtures,
        cycles: 2,
      });
    }
  });

  test('setDebug toggles the ground-truth overlay onto the scene camera', async () => {
    const { module } = await live();
    module.setDebug(true);
    module.update(toFrameUpdate(fixtureNamed('low-arch')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'canyonGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'canyonGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the camera stays a clamped aerial over the canyon', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('canyon-start')));
    const cam = module.camera;
    expect(cam.position.y).toBeGreaterThan(8);
    expect(Math.hypot(cam.position.x, cam.position.z)).toBeGreaterThan(CANYON_HALF);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.y).toBeLessThan(-0.5); // a genuine aerial, never a horizon shot
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    module.dispose();
  });

  test('the decoder\'s own goal event batch is what this scene is built for', () => {
    const st = highDimMazeDecoder.initStatic(SEED);
    const init = highDimMazeDecoder.initFrame(SEED, st);
    const scored = { ...init, done: 1 };
    const kinds = highDimMazeDecoder.deriveEvents(init, scored, st).map((e) => e.type);
    expect(kinds).toEqual(['goal', 'teleport', 'done']);
    // and nothing fires on an ordinary step
    expect(highDimMazeDecoder.deriveEvents(init, init, st).length).toBe(0);
  });
});

describe('high-dim-maze / the canyon geometry is a function of the maze', () => {
  test('a synthetic maze with a different gap moves the arch with it', async () => {
    // the acid test for "read, never re-derive": hand the scene a DIFFERENT layout and the
    // arch must follow the hallway list, not the numbers this env happens to use
    const w = HIGHDIMMAZE_W;
    const h = HIGHDIMMAZE_H;
    const walls = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const border = x === 0 || y === 0 || x === w - 1 || y === h - 1;
        walls[y * w + x] = border || (x === 5 && y !== 4) ? 1 : 0;
      }
    }
    const gap: GridCell = { x: 5, y: 4 };
    const geometry: EnvGeometry = {
      kind: 'grid',
      grid: { w, h, walls, goal: { x: 8, y: 8 }, hallways: [gap] },
    };
    const module = createScene();
    await module.init(makeSceneContext({ geometry, seed: SEED, tier: 'hero' }));
    module.update(toFrameUpdate(poseFixture('synthetic', 2, 2, 8, 8)));
    const probe = module.probes().find((p) => p.name === 'arch-0');
    expect(probe).toBeDefined();
    if (probe) {
      probe.object.updateWorldMatrix(true, false);
      const world = new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
      const truth: [number, number, number] = [0, 0, 0];
      cellToWorld(gap.x, gap.y, { w, h }, truth);
      expect(world.x).toBeCloseTo(truth[0], 10);
      expect(world.z).toBeCloseTo(truth[2], 10);
    }
    // exactly one arch, because the layout has exactly one hallway
    expect(module.probes().filter((p) => p.name.startsWith('arch-')).length).toBe(1);
    module.dispose();
  });
});

describe('high-dim-maze / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    // The wall view is the whole gallery on one page: every scene there is a thumb, and
    // the cap is what keeps 16 of them at 60 fps. Measured with the ground-truth overlay
    // ON, because that is the state review screenshots are taken in, and across every
    // fixture, because the busiest legal state is the one that has to fit.
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'high-dim-maze (thumb, overlay on)');
    module.dispose();
  });
});
