/// <reference types="bun" />
// Sunlit Descent (DeepSea-cont-v0) — the accuracy + lifecycle self-check.
//
// This mirrors four-rooms.test.ts's shape and adds what THIS env has to prove:
//
//  * **r is depth, and it is not transposed.** `pose = [r, c, 7, 7]` — row FIRST. The
//    wisp for pose [3, 2] must stand at `cellToWorld(2, 3)`, i.e. x = −1.5 and z = −0.5,
//    NOT the other way round, and its y must be row 3's terrace. That single assertion is
//    the regression test for the bug the retired 2-D viewer shipped.
//  * **the reef is the reachable set.** `reefReachable` runs the decoder's own transition
//    forward; the test re-derives the same set independently (a plain BFS over
//    `deepSeaNextCol`) for all eight recorded lanes and checks it against the c ≤ r
//    triangle — so the art can never quietly grow a shelf the dive cannot stand on.
//  * **the kelp is the mask.** Every clump's lean direction is decoded back out of its
//    instance matrix and checked against `grid.actionMask`, cell for cell — the same bit
//    the ground-truth overlay draws its arrows from.
//  * **the wrap is a hard cut.** Leaving row N−2 always resurfaces the diver at (0, 0);
//    the scene must snap, never glide a straight line across six terraces.

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { CELL, cellToWorld } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  DEEPSEA_LAST_ROW,
  DEEPSEA_N,
  deepSeaDecoder,
  deepSeaNextCol,
  type DeepSeaStatic,
} from '@sim/envs/deepsea';
import type { EnvGeometry, GridGeometry } from '@sim/types';

import createScene, { gape } from '../../src/scenes/deep-sea/index';
import { fixtures } from '../../src/scenes/deep-sea/fixtures';
import { manifest, deepSeaPose } from '../../src/scenes/deep-sea/manifest';
import {
  ROW_DROP,
  SHELF_THICKNESS,
  cellToReefWorld,
  isReef,
  isReefLip,
  reefDepthY,
  reefDepthYAt,
  reefReachable,
  reefShear,
  resolveGrid,
} from '../../src/scenes/deep-sea/reef';
import { KELP_LEAN, treasureCell } from '../../src/scenes/deep-sea/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

/** the demo run's first DeepSea lane (seeds 16–23 are the eight recorded lanes) */
const SEED = 16;
const RECORDED_SEEDS = [16, 17, 18, 19, 20, 21, 22, 23] as const;

function geometryOf(seed = SEED): EnvGeometry {
  return deepSeaDecoder.geometry(deepSeaDecoder.initStatic(seed), seed);
}

function gridOf(seed = SEED): GridGeometry {
  return resolveGrid(geometryOf(seed), seed);
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

/** pose = [r, c, …] — ROW FIRST, exactly as the record packs it. */
function poseFixture(name: string, r: number, c: number, action = 0, step = 0): SceneFixture {
  return {
    name,
    pose: deepSeaPose(r, c),
    ctrl: [action, 0, 0, 0, 0, 0],
    derived: { r, c, lastA: action & 1 },
    step,
  };
}

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

/** An INDEPENDENT re-derivation of the dive's reachable set (not the scene's function). */
function bfsReachable(grid: GridGeometry): Uint8Array {
  const mask = grid.actionMask ?? new Uint8Array(grid.w * grid.h);
  const st: DeepSeaStatic = { seed: 0, maskLo: 0, maskHi: 0, mask };
  const reach = new Uint8Array(grid.w * grid.h);
  reach[0] = 1;
  for (let r = 0; r + 1 < grid.h; r++) {
    for (let c = 0; c < grid.w; c++) {
      if ((reach[r * grid.w + c] ?? 0) === 0) continue;
      reach[(r + 1) * grid.w + deepSeaNextCol(st, r, c, 0)] = 1;
      reach[(r + 1) * grid.w + deepSeaNextCol(st, r, c, 1)] = 1;
    }
  }
  return reach;
}

function reefCellCount(grid: GridGeometry, reach: Uint8Array): number {
  let n = 0;
  for (let r = 0; r < grid.h; r++) for (let c = 0; c < grid.w; c++) if (isReef(reach, grid, c, r)) n++;
  return n;
}

// ---------------------------------------------------------------------------

describe('deep-sea / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('DeepSea-cont-v0');
    expect(manifest.slug).toBe('deep-sea');
    expect(manifest.accent).toBe('deep-sea');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a legal dive state (row first, on the reef)', () => {
    const grid = gridOf();
    const reach = reefReachable(grid);
    const p = manifest.thumbFrame.pose;
    const r = p[0] ?? -1;
    const c = p[1] ?? -1;
    expect(r).toBeLessThanOrEqual(DEEPSEA_LAST_ROW);
    expect(isReef(reach, grid, c, r)).toBe(true);
    // the pose really does carry the treasure cell in slots 2 and 3
    expect(p[2]).toBe(DEEPSEA_N - 1);
    expect(p[3]).toBe(DEEPSEA_N - 1);
  });

  test('ships at least three canonical poses, all on cells a dive can occupy', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    const grid = gridOf();
    const reach = reefReachable(grid);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      const r = f.pose[0] ?? -1;
      const c = f.pose[1] ?? -1;
      // rows 0..N-2 are the only rows a dive is ever observed in
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(DEEPSEA_LAST_ROW);
      expect(isReef(reach, grid, c, r)).toBe(true);
    }
  });

  test('the fixtures cover the wall clamp, a shelf lip and the treasure approach', () => {
    const grid = gridOf();
    const reach = reefReachable(grid);
    expect(fixtures.some((f) => (f.pose[1] ?? -1) === 0 && (f.pose[0] ?? 0) > 0)).toBe(true);
    expect(fixtures.some((f) => isReefLip(reach, grid, f.pose[1] ?? -1, f.pose[0] ?? -1))).toBe(true);
    expect(
      fixtures.some((f) => (f.pose[0] ?? -1) === DEEPSEA_LAST_ROW && (f.pose[1] ?? -1) === DEEPSEA_N - 2),
    ).toBe(true);
  });
});

describe('deep-sea / the reef is the reachable set, never a typed-in triangle', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry, SEED) === geometry.grid).toBe(true);
  });

  test("a placeholder geometry falls back to the DECODER's own board + mask", () => {
    const grid = resolveGrid({ kind: 'grid' }, SEED);
    expect(grid.w).toBe(DEEPSEA_N);
    expect(grid.h).toBe(DEEPSEA_N);
    expect(grid.actionMask?.length).toBe(DEEPSEA_N * DEEPSEA_N);
    const truth = gridOf();
    for (let i = 0; i < DEEPSEA_N * DEEPSEA_N; i++) {
      expect(grid.actionMask?.[i]).toBe(truth.actionMask?.[i]);
    }
  });

  test('reefReachable === the c ≤ r triangle for every recorded lane', () => {
    for (const seed of RECORDED_SEEDS) {
      const grid = gridOf(seed);
      const reach = reefReachable(grid);
      const independent = bfsReachable(grid);
      for (let r = 0; r < grid.h; r++) {
        for (let c = 0; c < grid.w; c++) {
          const on = isReef(reach, grid, c, r);
          expect(`s${seed} ${c},${r}=${on}`).toBe(`s${seed} ${c},${r}=${c <= r}`);
          expect(on).toBe((independent[r * grid.w + c] ?? 0) !== 0);
        }
      }
      // 1 + 2 + … + 8 = 36 shelves; the other 28 cells are open blue water
      expect(reefCellCount(grid, reach)).toBe((DEEPSEA_N * (DEEPSEA_N + 1)) / 2);
    }
  });

  test('the lip of every row is its frontier column, and the open corner stays empty', () => {
    const grid = gridOf();
    const reach = reefReachable(grid);
    for (let r = 0; r < grid.h; r++) {
      for (let c = 0; c < grid.w; c++) {
        expect(isReefLip(reach, grid, c, r)).toBe(c === r);
      }
    }
    // the shallow / far-column corner is water: no shelf, no kelp, nothing to stand on
    expect(isReef(reach, grid, DEEPSEA_N - 1, 0)).toBe(false);
    expect(isReef(reach, grid, DEEPSEA_N - 1, DEEPSEA_N - 2)).toBe(false);
  });

  test('the treasure cell is the env geometry\'s own goal', () => {
    const grid = gridOf();
    const t = treasureCell(grid);
    expect(t.c).toBe(grid.goal?.x ?? -1);
    expect(t.r).toBe(grid.goal?.y ?? -1);
    expect(t.c).toBe(DEEPSEA_N - 1);
    expect(t.r).toBe(DEEPSEA_N - 1);
  });
});

describe('deep-sea / the depth axis', () => {
  test('every row sits strictly deeper than the one above it', () => {
    for (let r = 1; r < DEEPSEA_N; r++) {
      expect(reefDepthY(r)).toBeLessThan(reefDepthY(r - 1));
      expect(reefDepthY(r - 1) - reefDepthY(r)).toBeCloseTo(ROW_DROP, 12);
    }
    expect(reefDepthY(0)).toBeCloseTo(0, 12);
    // the slabs must overlap or daylight shows between the terraces
    expect(SHELF_THICKNESS).toBeGreaterThan(ROW_DROP);
  });

  test('reefDepthYAt is EXACT (not approximate) at every cell centre', () => {
    const grid = gridOf();
    const out: [number, number, number] = [0, 0, 0];
    for (let r = 0; r < grid.h; r++) {
      cellToWorld(0, r, grid, out);
      // bit-identical, which is what lets the alignment probes assert 0 error
      expect(reefDepthYAt(out[2], grid)).toBe(reefDepthY(r));
    }
  });

  test('cellToReefWorld is row-major-correct: pose [r, c] -> (x from c, z from r)', () => {
    const grid = gridOf();
    const out: [number, number, number] = [0, 0, 0];
    // the anti-transposition case: r = 3, c = 2 is NOT r = 2, c = 3
    cellToReefWorld(2, 3, grid, out);
    expect(out[0]).toBeCloseTo((2 - (DEEPSEA_N - 1) / 2) * CELL, 12);
    expect(out[2]).toBeCloseTo((3 - (DEEPSEA_N - 1) / 2) * CELL, 12);
    expect(out[1]).toBe(reefDepthY(3));
    const other: [number, number, number] = [0, 0, 0];
    cellToReefWorld(3, 2, grid, other);
    expect(other[0]).not.toBeCloseTo(out[0], 6);
    expect(other[1]).not.toBe(out[1]);
  });

  test('reefShear carries a flat grid point onto its own terrace', () => {
    const grid = gridOf();
    const m = reefShear(grid);
    const p = new THREE.Vector3();
    const truth: [number, number, number] = [0, 0, 0];
    for (let r = 0; r < grid.h; r++) {
      for (let c = 0; c < grid.w; c++) {
        cellToWorld(c, r, grid, truth);
        // a marker the overlay drew 0.03 above the flat plane...
        p.set(truth[0], 0.03, truth[2]).applyMatrix4(m);
        // ...lands 0.03 above the terrace, with x and z untouched
        expect(p.x).toBeCloseTo(truth[0], 12);
        expect(p.z).toBeCloseTo(truth[2], 12);
        expect(p.y).toBeCloseTo(reefDepthY(r) + 0.03, 12);
      }
    }
  });
});

describe('deep-sea / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    // these roots carry POSITION ONLY — they should be exact, not merely within 1e-3
    expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('clam') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('silt') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      module.dispose();
    }
  });

  test('the diver lands on EVERY cell a dive can occupy, on that row\'s terrace', async () => {
    const { module } = await live();
    const grid = gridOf();
    const reach = reefReachable(grid);
    const truth: [number, number, number] = [0, 0, 0];
    let visited = 0;
    for (let r = 0; r <= DEEPSEA_LAST_ROW; r++) {
      for (let c = 0; c < grid.w; c++) {
        if (!isReef(reach, grid, c, r)) continue;
        module.update(toFrameUpdate(poseFixture(`cell${r}-${c}`, r, c, 0, visited)));
        const w = worldOf(module, 'wisp-anchor');
        cellToReefWorld(c, r, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.y).toBeCloseTo(truth[1], 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
        // the orb centre rides exactly wispHeight above the shelf
        const orb = worldOf(module, 'wisp');
        expect(orb.y).toBeCloseTo(truth[1] + TOKENS.wispHeight, 10);
        visited++;
      }
    }
    // rows 0..N-2 hold 1 + 2 + … + (N-1) cells
    expect(visited).toBe((DEEPSEA_N * (DEEPSEA_N - 1)) / 2);
    module.dispose();
  });

  test('the clam sits on the deepest terrace, at the env\'s own goal cell', async () => {
    const { module } = await live();
    const grid = gridOf();
    const truth: [number, number, number] = [0, 0, 0];
    module.update(toFrameUpdate(fixtureNamed('treasure-approach')));
    const w = worldOf(module, 'clam');
    cellToReefWorld(DEEPSEA_N - 1, DEEPSEA_N - 1, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(truth[1], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    // and it really is the deepest thing in the reef
    expect(w.y).toBeCloseTo(reefDepthY(DEEPSEA_N - 1), 10);
    module.dispose();
  });

  test('one shelf per reef cell, each with its top face exactly on its terrace', async () => {
    const { module } = await live();
    const grid = gridOf();
    const reach = reefReachable(grid);
    const mesh = module.scene.getObjectByName('oakCellFloorMesh') as THREE.InstancedMesh | undefined;
    expect(mesh).toBeDefined();
    if (!mesh) return;
    expect(mesh.count).toBe(reefCellCount(grid, reach));
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const truth: [number, number, number] = [0, 0, 0];
    let checked = 0;
    for (let i = 0; i < mesh.count; i++) {
      mesh.getMatrixAt(i, m);
      m.decompose(p, q, s);
      const c = Math.round(p.x + (grid.w - 1) / 2);
      const r = Math.round(p.z + (grid.h - 1) / 2);
      expect(isReef(reach, grid, c, r)).toBe(true);
      cellToReefWorld(c, r, grid, truth);
      // instance matrices live in a Float32Array, so these are f32-exact, not f64-exact
      expect(p.x).toBeCloseTo(truth[0], 5);
      expect(p.z).toBeCloseTo(truth[2], 5);
      // the slab geometry's TOP FACE is at local y = 0 and jitter only scales it
      // downward, so the instance origin IS the shelf top
      expect(p.y).toBeCloseTo(reefDepthY(r), 5);
      checked++;
    }
    expect(checked).toBe((DEEPSEA_N * (DEEPSEA_N + 1)) / 2);
    module.dispose();
  });
});

describe('deep-sea / the kelp is the action mask', () => {
  test('every clump leans the way action 1 carries the diver at its cell', async () => {
    const { module } = await live();
    const grid = gridOf();
    const reach = reefReachable(grid);
    const mesh = module.scene.getObjectByName('reefKelp') as THREE.InstancedMesh | undefined;
    expect(mesh).toBeDefined();
    if (!mesh) return;
    expect(mesh.count).toBe(reefCellCount(grid, reach));

    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const up = new THREE.Vector3();
    const seen = new Set<string>();
    for (let i = 0; i < mesh.count; i++) {
      mesh.getMatrixAt(i, m);
      m.decompose(p, q, s);
      const c = Math.round(p.x + (grid.w - 1) / 2);
      const r = Math.round(p.z + (grid.h - 1) / 2);
      expect(isReef(reach, grid, c, r)).toBe(true);
      seen.add(`${c},${r}`);
      // a clump's own axis, rotated into the world: its x sign IS the current's direction
      up.set(0, 1, 0).applyQuaternion(q);
      const bit = (grid.actionMask?.[r * grid.w + c] ?? 0) !== 0;
      expect(`${c},${r} leansRight=${up.x > 0}`).toBe(`${c},${r} leansRight=${bit}`);
      // and it is a real, visible lean, not a rounding artefact
      expect(Math.abs(up.x)).toBeGreaterThan(Math.sin(KELP_LEAN * 0.7) - 1e-6);
      // it stands on its own terrace (f32 instance matrix)
      expect(p.y).toBeCloseTo(reefDepthY(r), 5);
    }
    expect(seen.size).toBe(mesh.count);
    module.dispose();
  });
});

describe('deep-sea / the dive glides, the wrap cuts', () => {
  test('one dive step stays inside the two cells and lands exactly on truth', async () => {
    const { module, ctx } = await live();
    const grid = gridOf();
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('a', 2, 1, 0, 0)));
    cellToReefWorld(1, 2, grid, from);
    cellToReefWorld(2, 3, grid, to);

    // step one row deeper and one column right; sample the glide over one control step
    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('b', 3, 2, 1, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(poseFixture('b', 3, 2, 1, 1), { scrubbed: false }));
      const w = worldOf(module, 'wisp-anchor');
      expect(w.x).toBeGreaterThanOrEqual(Math.min(from[0], to[0]) - 1e-6);
      expect(w.x).toBeLessThanOrEqual(Math.max(from[0], to[0]) + 1e-6);
      expect(w.z).toBeGreaterThanOrEqual(Math.min(from[2], to[2]) - 1e-6);
      expect(w.z).toBeLessThanOrEqual(Math.max(from[2], to[2]) + 1e-6);
      // and it is always DESCENDING through the terraces, never above the shallower one
      expect(w.y).toBeLessThanOrEqual(from[1] + 0.07);
      expect(w.y).toBeGreaterThanOrEqual(to[1] - 1e-6);
    }
    const landed = worldOf(module, 'wisp-anchor');
    expect(landed.x).toBeCloseTo(to[0], 10);
    expect(landed.y).toBeCloseTo(to[1], 10);
    expect(landed.z).toBeCloseTo(to[2], 10);
    module.dispose();
  });

  test('the resurface wrap is a HARD CUT, never a glide across the reef', async () => {
    const { module, ctx } = await live();
    const grid = gridOf();
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(4);
    module.update(toFrameUpdate(poseFixture('deep', DEEPSEA_LAST_ROW, 5, 1, 100), { scrubbed: false }));
    // the decoder fires [goal?, resurface] on the frame the diver is already at (0, 0)
    ctx.setTime(4.125);
    module.update(
      toFrameUpdate(poseFixture('surfaced', 0, 0, 1, 101), {
        scrubbed: false,
        events: [{ type: 'resurface' }],
      }),
    );
    const w = worldOf(module, 'wisp-anchor');
    cellToReefWorld(0, 0, grid, truth);
    // no interpolation at all: the very frame the event lands, the diver is at the surface
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(truth[1], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    module.dispose();
  });

  test('a wrap with the event dropped (a scrub) still cuts rather than glides', async () => {
    const { module, ctx } = await live();
    const grid = gridOf();
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(7);
    module.update(toFrameUpdate(poseFixture('deep', DEEPSEA_LAST_ROW, 3, 0, 200), { scrubbed: false }));
    ctx.setTime(7.125);
    module.update(toFrameUpdate(poseFixture('surfaced', 0, 0, 0, 201), { scrubbed: false }));
    const w = worldOf(module, 'wisp-anchor');
    cellToReefWorld(0, 0, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(truth[1], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    module.dispose();
  });

  test('a scrub snaps the diver with no glide at all', async () => {
    const { module, ctx } = await live();
    const grid = gridOf();
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(4);
    module.update(toFrameUpdate(poseFixture('a', 0, 0, 0, 0), { scrubbed: false }));
    module.update(toFrameUpdate(poseFixture('far', 6, 6, 1, 4000), { scrubbed: true }));
    const w = worldOf(module, 'wisp-anchor');
    cellToReefWorld(6, 6, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(truth[1], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    module.dispose();
  });
});

describe('deep-sea / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('surface-start')));
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

    // sitting one move from the treasure with no event: nothing may fire
    ctx.setTime(1);
    module.update(toFrameUpdate(fixtureNamed('treasure-approach'), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    // a resurface alone is not a reward: bubbles, but no celebration
    ctx.setTime(1.125);
    module.update(
      toFrameUpdate(poseFixture('surfaced', 0, 0, 1, 3), { scrubbed: false, events: [{ type: 'resurface' }] }),
    );
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);
    expect(pool.pulse.bloom).toBe(0);

    // the decoder's real batch on a successful dive: goal AND resurface, same frame
    ctx.setTime(2);
    module.update(
      toFrameUpdate(poseFixture('treasure', 0, 0, 1, 4), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'resurface' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    module.dispose();
  });

  test('the bloom happens at the CLAM, not where the wrapped diver is', async () => {
    const { module, ctx } = await live();
    const grid = gridOf();
    ctx.setTime(3);
    module.update(toFrameUpdate(poseFixture('deep', DEEPSEA_LAST_ROW, 6, 1, 300), { scrubbed: false }));
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(poseFixture('treasure', 0, 0, 1, 301), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'resurface' }],
      }),
    );
    const light = module.scene.getObjectByName('oakPoint2');
    expect(light).toBeDefined();
    if (!light) return;
    const truth: [number, number, number] = [0, 0, 0];
    cellToReefWorld(DEEPSEA_N - 1, DEEPSEA_N - 1, grid, truth);
    expect(light.position.x).toBeCloseTo(truth[0], 10);
    expect(light.position.z).toBeCloseTo(truth[2], 10);
    // above the shelf (the pearl sits inside the shells), and lit
    expect(light.position.y).toBeGreaterThan(truth[1]);
    expect((light as THREE.PointLight).intensity).toBeGreaterThan(0);
    // meanwhile the diver really is back at the surface
    const w = worldOf(module, 'wisp-anchor');
    cellToReefWorld(0, 0, grid, truth);
    expect(w.z).toBeCloseTo(truth[2], 10);
    module.dispose();
  });

  test('a scrub clears every in-flight beat', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(poseFixture('treasure', 0, 0, 1, 400), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'resurface' }],
      }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('mid-dive'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    module.dispose();
  });

  test('the gape envelope is pure, bounded, and opens as the dive closes in', () => {
    const grid = gridOf();
    for (let t = 0; t < 12; t += 0.37) {
      for (let r = 0; r <= DEEPSEA_LAST_ROW; r++) {
        const g = gape(t, { r, c: r, lastA: 0 }, grid);
        expect(g).toBeGreaterThanOrEqual(0);
        expect(g).toBeLessThanOrEqual(1);
      }
    }
    // deep AND near the treasure column opens wider than shallow and far
    const far = gape(0, { r: 0, c: 0, lastA: 0 }, grid);
    const near = gape(0, { r: DEEPSEA_LAST_ROW, c: DEEPSEA_N - 2, lastA: 0 }, grid);
    expect(near).toBeGreaterThan(far);
    // and it is a pure function: same arguments, same answer
    expect(gape(3.3, { r: 4, c: 6, lastA: 1 }, grid)).toBe(gape(3.3, { r: 4, c: 6, lastA: 1 }, grid));
  });
});

describe('deep-sea / lifecycle', () => {
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
    expect((report.census[0]?.textures.size ?? 0) > 0).toBe(true);
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
    module.update(toFrameUpdate(fixtureNamed('mid-dive')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'reefGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'reefGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the overlay carries the reef\'s own grid->world map, so it cannot drift', async () => {
    const { module } = await live();
    const grid = gridOf();
    const overlay = module.scene.getObjectByName('reefGroundTruth');
    expect(overlay).toBeDefined();
    if (!overlay) return;
    expect(overlay.matrixAutoUpdate).toBe(false);
    const expected = reefShear(grid);
    for (let i = 0; i < 16; i++) {
      expect(overlay.matrix.elements[i] ?? NaN).toBeCloseTo(expected.elements[i] ?? NaN, 12);
    }
    // the overlay's own agent cross for a mid-dive pose lands on that row's terrace
    overlay.updateWorldMatrix(true, false);
    const truth: [number, number, number] = [0, 0, 0];
    cellToReefWorld(2, 3, grid, truth);
    const p = new THREE.Vector3(truth[0], 0.03, truth[2]).applyMatrix4(overlay.matrixWorld);
    expect(p.y).toBeCloseTo(truth[1] + 0.03, 10);
    module.dispose();
  });

  test('the camera stands at the DEEP end, so the dive comes toward it', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('mid-dive')));
    const grid = gridOf();
    const cam = module.camera;
    // above the reef, outside the board, genuinely looking down into it
    expect(cam.position.y).toBeGreaterThan(2);
    expect(Math.hypot(cam.position.x, cam.position.z)).toBeGreaterThan((DEEPSEA_N * CELL) / 2);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.y).toBeLessThan(-0.25);
    // THE COMPOSITION INVARIANT. Both coordinates are positive, which puts the camera over
    // the treasure corner (+x = the far columns, +z = the DEEP rows). That is not taste:
    // with the dive receding, `spacing·sin(elev)` and `drop·cos(elev)` cancel exactly when
    // the shelf tops are visible, and the staircase renders flat. Coming toward the eye,
    // the two add. Flip either sign and the descent stops reading.
    expect(cam.position.x).toBeGreaterThan(0);
    expect(cam.position.z).toBeGreaterThan(0);
    // ...and the deepest row really is the nearest thing in the frame
    const near: [number, number, number] = [0, 0, 0];
    const far: [number, number, number] = [0, 0, 0];
    cellToReefWorld(0, DEEPSEA_N - 1, grid, near);
    cellToReefWorld(0, 0, grid, far);
    const dNear = cam.position.distanceTo(new THREE.Vector3(near[0], near[1], near[2]));
    const dFar = cam.position.distanceTo(new THREE.Vector3(far[0], far[1], far[2]));
    expect(dNear).toBeLessThan(dFar);
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    module.dispose();
  });

  test('the start pose really is the env\'s deterministic init', () => {
    const start = fixtureNamed('surface-start');
    expect(start.pose[0]).toBe(0);
    expect(start.pose[1]).toBe(0);
    const init = deepSeaDecoder.initFrame(SEED, deepSeaDecoder.initStatic(SEED));
    expect(init.state.r).toBe(start.pose[0] ?? -1);
    expect(init.state.c).toBe(start.pose[1] ?? -1);
  });
});

describe('deep-sea / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    // The wall view is the whole gallery on one page: every scene there is a thumb, and
    // the cap is what keeps 16 of them at 60 fps. Measured with the ground-truth overlay
    // ON, because that is the state review screenshots are taken in, and across every
    // fixture, because the busiest legal state is the one that has to fit.
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'deep-sea (thumb, overlay on)');
    module.dispose();
  });
});
