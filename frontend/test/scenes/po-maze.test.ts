/// <reference types="bun" />
// The Fogbound Forest (POMaze-cont-v0) — the accuracy + lifecycle + HONESTY self-check.
//
// This mirrors four-rooms.test.ts's shape and adds the thing this scene exists to prove:
// that what the art shows and what the AGENT observes are the same set, cell for cell.
// `poMazeView` publishes eight wall bits; `bubbleVisibility` decides which spruce are
// drawn; the tests below walk every free cell of the maze and assert those two agree
// EXACTLY — not approximately, not "mostly outside the bubble", exactly. Plus the
// compass: the moth may encode the four sign bits POMaze publishes and must be provably
// independent of the distance it is not told.

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { CELL, cellToWorld, isWall, wallCellCount } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  POMAZE_GAPS,
  POMAZE_GOAL,
  POMAZE_H,
  POMAZE_START,
  POMAZE_W,
  poMazeDecoder,
  poMazeIsWall,
  poMazeView,
} from '@sim/envs/pomaze';
import type { EnvGeometry, GridCell } from '@sim/types';

import createScene, { envelope, LINGER_SECONDS, readCells } from '../../src/scenes/po-maze/index';
import { fixtures } from '../../src/scenes/po-maze/fixtures';
import { manifest } from '../../src/scenes/po-maze/manifest';
import {
  BUBBLE_INNER,
  BUBBLE_OUTER,
  bubbleVisibility,
  buildSpruceRanks,
  compassOffset,
  makeForestHeight,
  MAZE_HALF,
  MOTH_BREATH,
  MOTH_RIM,
  MOTH_SWING,
  resolveGaps,
  resolveGrid,
  revealVisibility,
} from '../../src/scenes/po-maze/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 112; // the demo run's first POMaze lane

function geometryOf(): EnvGeometry {
  return poMazeDecoder.geometry(poMazeDecoder.initStatic(SEED), SEED);
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

function objectNamed(module: SceneModule, name: string): THREE.Object3D {
  const o = module.scene.getObjectByName(name);
  if (!o) throw new Error(`object "${name}" missing from the scene graph`);
  return o;
}

/** every FREE cell of the maze, in row-major order */
function freeCells(): GridCell[] {
  const out: GridCell[] = [];
  for (let y = 0; y < POMAZE_H; y++) {
    for (let x = 0; x < POMAZE_W; x++) if (!poMazeIsWall(x, y)) out.push({ x, y });
  }
  return out;
}

/**
 * Every spruce instance the scene is currently DRAWING, resolved back to the cell it
 * stands on. An instance whose basis vectors have zero length transforms every vertex of
 * its cone onto one point: no area, no fragment, nothing on screen. A cell is "shown"
 * when at least one of its trees still has extent.
 *
 * The lengths are measured off the raw matrix ELEMENTS rather than through
 * `Matrix4.decompose`, and that is not a style preference: three r185's decompose carries
 * a `det === 0` guard that returns scale (1, 1, 1) for exactly the degenerate matrices
 * this test exists to detect. Decomposing here would report a collapsed forest as a
 * fully-grown one and pass a scene that gives the whole maze away. This is the honesty
 * instrument, so it reads what the vertex shader reads.
 */
function shownWallCells(module: SceneModule): Set<string> {
  const mesh = objectNamed(module, 'spruceRankMesh') as THREE.InstancedMesh;
  const m = new THREE.Matrix4();
  const shown = new Set<string>();
  for (let i = 0; i < mesh.count; i++) {
    mesh.getMatrixAt(i, m);
    const e = m.elements;
    const sx = Math.hypot(e[0] ?? 0, e[1] ?? 0, e[2] ?? 0);
    const sy = Math.hypot(e[4] ?? 0, e[5] ?? 0, e[6] ?? 0);
    const sz = Math.hypot(e[8] ?? 0, e[9] ?? 0, e[10] ?? 0);
    if (sx <= 1e-4 && sy <= 1e-4 && sz <= 1e-4) continue;
    const cx = Math.round((e[12] ?? 0) / CELL + (POMAZE_W - 1) / 2);
    const cy = Math.round((e[14] ?? 0) / CELL + (POMAZE_H - 1) / 2);
    shown.add(`${cx},${cy}`);
  }
  return shown;
}

/** The 8 Chebyshev neighbours of a cell that are walls — the agent's own observation. */
function observedWallCells(x: number, y: number): Set<string> {
  const deltas: readonly (readonly [number, number])[] = [
    [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1],
  ];
  const bits = poMazeView(x, y);
  const out = new Set<string>();
  for (let i = 0; i < 8; i++) {
    const d = deltas[i] as readonly [number, number];
    if ((bits[i] ?? 0) !== 0) out.add(`${x + d[0]},${y + d[1]}`);
  }
  return out;
}

/**
 * A headless draw-call / triangle census of the scene graph — the harness HUD's numbers
 * without a GPU. One draw per visible mesh/points/line, plus one more for every shadow
 * caster (three renders the shadow map in its own pass); triangles from the geometries,
 * multiplied by live instance counts.
 */
function census(module: SceneModule, shadows: boolean): { draws: number; triangles: number } {
  let draws = 0;
  let triangles = 0;
  module.scene.traverse((o) => {
    if (!o.visible) return;
    const withGeo = o as THREE.Mesh & { isMesh?: boolean; isPoints?: boolean; isLine?: boolean };
    const geo = withGeo.geometry as THREE.BufferGeometry | undefined;
    if (!geo || !(withGeo.isMesh || withGeo.isPoints || withGeo.isLine)) return;
    let parentVisible = true;
    for (let p = o.parent; p; p = p.parent) if (!p.visible) parentVisible = false;
    if (!parentVisible) return;
    const instances = (o as THREE.InstancedMesh).isInstancedMesh ? (o as THREE.InstancedMesh).count : 1;
    if (instances <= 0) return;
    draws += 1;
    if (shadows && withGeo.castShadow) draws += 1;
    const pos = geo.getAttribute('position');
    const verts = geo.index ? geo.index.count : (pos?.count ?? 0);
    if (withGeo.isMesh) triangles += (verts / 3) * instances;
  });
  return { draws, triangles: Math.round(triangles) };
}

// ---------------------------------------------------------------------------

describe('po-maze / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('POMaze-cont-v0');
    expect(manifest.slug).toBe('po-maze');
    expect(manifest.accent).toBe('po-maze');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a legal maze state', () => {
    const p = manifest.thumbFrame.pose;
    expect(poMazeIsWall(p[0] ?? -1, p[1] ?? -1)).toBe(false);
    expect(poMazeIsWall(p[2] ?? -1, p[3] ?? -1)).toBe(false);
  });

  test('ships at least three canonical poses, all on FREE cells', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      expect(poMazeIsWall(f.pose[0] ?? -1, f.pose[1] ?? -1)).toBe(false);
      expect(poMazeIsWall(f.pose[2] ?? -1, f.pose[3] ?? -1)).toBe(false);
    }
  });

  test('the fixtures cover both gaps, an empty bubble and a goal inside the bubble', () => {
    const at = (c: GridCell): boolean =>
      fixtures.some((f) => (f.pose[0] ?? -1) === c.x && (f.pose[1] ?? -1) === c.y);
    for (const gap of POMAZE_GAPS) expect(at(gap)).toBe(true);
    // a cell with NO wall in its 3 × 3 window at all
    const empty = fixtures.some((f) => observedWallCells(f.pose[0] ?? 0, f.pose[1] ?? 0).size === 0);
    expect(empty).toBe(true);
    // and one where the goal is Chebyshev-adjacent — the state the frame must NOT betray
    // (the agent's four sign bits say "north-east" there and nothing more)
    const goalSeen = fixtures.some((f) => {
      const dx = Math.abs((f.pose[2] ?? 0) - (f.pose[0] ?? 0));
      const dy = Math.abs((f.pose[3] ?? 0) - (f.pose[1] ?? 0));
      return Math.max(dx, dy) <= 1;
    });
    expect(goalSeen).toBe(true);
  });
});

describe('po-maze / the layout is read, never re-derived', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry) === geometry.grid).toBe(true);
  });

  test("a placeholder geometry falls back to the DECODER's own walls", () => {
    const grid = resolveGrid({ kind: 'grid' });
    expect(grid.w).toBe(POMAZE_W);
    expect(grid.h).toBe(POMAZE_H);
    let free = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        expect(isWall(grid, x, y)).toBe(poMazeIsWall(x, y));
        if (!isWall(grid, x, y)) free++;
      }
    }
    // 9 × 9 interior, minus the two eight-cell bands, plus the two gaps
    expect(free).toBe(65);
    expect(freeCells().length).toBe(65);
  });

  test('the gaps come from the decoder, and both really are free cells in a wall band', () => {
    const grid = resolveGrid(geometryOf());
    const gaps = resolveGaps(grid);
    expect(gaps.length).toBe(POMAZE_GAPS.length);
    for (let i = 0; i < gaps.length; i++) {
      const g = gaps[i] as GridCell;
      const truth = POMAZE_GAPS[i] as GridCell;
      expect(g.x).toBe(truth.x);
      expect(g.y).toBe(truth.y);
      expect(isWall(grid, g.x, g.y)).toBe(false);
      // the band runs along x: east and west of a gap are wall, north and south are not
      expect(isWall(grid, g.x - 1, g.y)).toBe(true);
      expect(isWall(grid, g.x + 1, g.y)).toBe(true);
      expect(isWall(grid, g.x, g.y - 1)).toBe(false);
      expect(isWall(grid, g.x, g.y + 1)).toBe(false);
    }
  });

  test('the spruce ranks stand on exactly the wall cells the walls array names', () => {
    const grid = resolveGrid(geometryOf());
    const ranks = buildSpruceRanks({
      grid,
      tier: 'hero',
      seed: SEED,
      color: 0x203040,
      rimColor: 0x8090a0,
    });
    expect(ranks.cellCount).toBe(wallCellCount(grid));
    expect(ranks.cellCount).toBe(POMAZE_W * POMAZE_H - 65);
    ranks.dispose();
  });

  test('the forest floor is identically flat under the whole playfield', () => {
    const h = makeForestHeight(SEED);
    for (let i = 0; i <= 40; i++) {
      for (let j = 0; j <= 40; j++) {
        const x = -MAZE_HALF + (2 * MAZE_HALF * i) / 40;
        const z = -MAZE_HALF + (2 * MAZE_HALF * j) / 40;
        expect(h(x, z)).toBe(0);
      }
    }
    let wander = 0;
    for (const p of [-18, -14, 14, 18]) wander = Math.max(wander, Math.abs(h(p, p)));
    expect(wander).toBeGreaterThan(0.05);
  });
});

describe('po-maze / THE BUBBLE IS THE OBSERVATION', () => {
  test('bubbleVisibility is 1 inside one cell, 0 past two, and monotone between', () => {
    // the eight neighbours, sampled at their exact centres
    for (const [dx, dz] of [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]]) {
      expect(bubbleVisibility(0, 0, (dx ?? 0) * CELL, (dz ?? 0) * CELL)).toBe(1);
    }
    // everything at Chebyshev 2 or more, in every direction
    for (let dx = -3; dx <= 3; dx++) {
      for (let dz = -3; dz <= 3; dz++) {
        if (Math.max(Math.abs(dx), Math.abs(dz)) < 2) continue;
        expect(bubbleVisibility(0, 0, dx * CELL, dz * CELL)).toBe(0);
      }
    }
    // and it really is a falloff, not a step, so a hop dissolves
    let previous = 1;
    for (let k = 0; k <= 20; k++) {
      const d = BUBBLE_INNER + ((BUBBLE_OUTER - BUBBLE_INNER) * k) / 20;
      const v = bubbleVisibility(0, 0, d, 0);
      expect(v).toBeLessThanOrEqual(previous + 1e-12);
      expect(v).toBeGreaterThanOrEqual(0);
      previous = v;
    }
    expect(bubbleVisibility(0, 0, 1.5 * CELL, 0)).toBeCloseTo(0.5, 6);
  });

  test('the reveal falloff is 0 outside its radius and 1 at the centre', () => {
    expect(revealVisibility(0, 0, 0, 0, 3)).toBe(1);
    expect(revealVisibility(0, 0, 3, 0, 3)).toBe(0);
    expect(revealVisibility(0, 0, 9, 9, 3)).toBe(0);
    expect(revealVisibility(0, 0, 2, 0, 3)).toBeGreaterThan(0);
    expect(revealVisibility(0, 0, 2, 0, 3)).toBeLessThan(1);
  });

  test('the ranks drawn from EVERY free cell are exactly the agent\'s eight wall bits', async () => {
    const { module } = await live();
    let checked = 0;
    for (const cell of freeCells()) {
      module.update(toFrameUpdate(poseFixture('walk', cell.x, cell.y, 9, 9, checked)));
      const shown = shownWallCells(module);
      const observed = observedWallCells(cell.x, cell.y);
      expect([...shown].sort()).toEqual([...observed].sort());
      checked++;
    }
    expect(checked).toBe(65);
    module.dispose();
  });

  test('no wall outside the bubble is EVER drawn, at any tier', async () => {
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module } = await live(tier);
      for (const cell of [POMAZE_START, { x: 5, y: 5 }, POMAZE_GAPS[0] as GridCell, { x: 9, y: 9 }]) {
        module.update(toFrameUpdate(poseFixture('spot', cell.x, cell.y, 1, 1)));
        for (const key of shownWallCells(module)) {
          const parts = key.split(',');
          const wx = Number(parts[0]);
          const wy = Number(parts[1]);
          expect(Math.max(Math.abs(wx - cell.x), Math.abs(wy - cell.y))).toBeLessThanOrEqual(1);
        }
      }
      module.dispose();
    }
  });

  test('the dissolve is a pure function of position: a scrub lands on the exact bubble', async () => {
    const { module, ctx } = await live();
    // walk into a hop, then scrub somewhere else and back — the bubble must be identical
    ctx.setTime(3);
    module.update(toFrameUpdate(poseFixture('a', 4, 5, 9, 9), { scrubbed: false }));
    ctx.setTime(3.05);
    module.update(toFrameUpdate(poseFixture('b', 5, 5, 9, 9), { scrubbed: false }));
    const midHop = shownWallCells(module);
    ctx.setTime(40);
    module.update(toFrameUpdate(poseFixture('far', 1, 9, 9, 1), { scrubbed: true }));
    ctx.setTime(3.05);
    module.update(toFrameUpdate(poseFixture('b', 5, 5, 9, 9), { scrubbed: true }));
    const settled = shownWallCells(module);
    // mid-hop the bubble may straddle two cells; settled it is exactly the observation
    expect([...settled].sort()).toEqual([...observedWallCells(5, 5)].sort());
    expect(midHop.size).toBeGreaterThanOrEqual(0);
    module.dispose();
  });

  test('the gap gates appear only when the agent is adjacent to the gap', async () => {
    const { module } = await live();
    const gap = POMAZE_GAPS[0] as GridCell;
    // standing next to the north gap: its stones are up
    module.update(toFrameUpdate(poseFixture('near', gap.x, gap.y - 1, 1, 1)));
    expect(objectNamed(module, 'gapGateBody0').visible).toBe(true);
    // three cells away: gone
    module.update(toFrameUpdate(poseFixture('away', gap.x - 3, gap.y - 1, 1, 1)));
    expect(objectNamed(module, 'gapGateBody0').visible).toBe(false);
    module.dispose();
  });
});

/**
 * THE HONESTY DIGEST — every VISIBLE thing in the frame, plus every light.
 *
 * `gw_po_obs` (Sources/Gridworld.metal) publishes eight wall bits, the agent's own
 * position and FOUR SIGN BITS for the goal — no distance, no presence bit at any range.
 * So two states that differ only in how far away the goal is, with the same sign pair,
 * are the SAME observation, and the frame must be pixel-identical between them. This
 * digest is the instrument: it reads the world matrix of everything that draws (skipping
 * anything hidden, itself or by an ancestor) and the position + intensity of every light,
 * because a light parked on the goal cell would leak it just as loudly as a mesh.
 */
function frameDigest(module: SceneModule): string {
  const rows: string[] = [];
  module.scene.updateMatrixWorld(true);
  module.scene.traverse((o) => {
    if (!o.visible) return;
    for (let p = o.parent; p; p = p.parent) if (!p.visible) return;
    const drawable = o as THREE.Mesh & { isMesh?: boolean; isPoints?: boolean; isLine?: boolean };
    const light = o as THREE.PointLight & { isLight?: boolean };
    if (!drawable.isMesh && !drawable.isPoints && !drawable.isLine && !light.isLight) return;
    const m = o.matrixWorld.elements;
    let row = `${o.name}|`;
    for (let i = 0; i < 16; i++) row += `${(m[i] ?? 0).toFixed(9)},`;
    if (light.isLight) row += `I${light.intensity.toFixed(9)}`;
    rows.push(row);
  });
  return rows.sort().join('\n');
}

describe('po-maze / THE FRAME NEVER KNOWS MORE THAN THE AGENT', () => {
  test('two goals sharing a sign pair are INDISTINGUISHABLE in the frame', async () => {
    const { module, ctx } = await live();
    ctx.setTime(7.5);
    // group every free cell by the four sign bits it presents from (5,5) — the agent's
    // WHOLE knowledge of the goal — and demand one frame per group
    const bySign = new Map<string, GridCell[]>();
    for (const g of freeCells()) {
      if (g.x === 5 && g.y === 5) continue;
      const key = `${Math.sign(g.x - 5)},${Math.sign(g.y - 5)}`;
      const bucket = bySign.get(key);
      if (bucket) bucket.push(g);
      else bySign.set(key, [g]);
    }
    expect(bySign.size).toBe(8); // all eight octants really are exercised

    for (const [key, cells] of bySign) {
      let reference: string | null = null;
      let referenceCell: GridCell | null = null;
      for (const g of cells) {
        module.update(toFrameUpdate(poseFixture('leak', 5, 5, g.x, g.y)));
        const digest = frameDigest(module);
        if (reference === null) {
          reference = digest;
          referenceCell = g;
          continue;
        }
        if (digest !== reference) {
          const a = reference.split('\n');
          const b = digest.split('\n');
          const diff = b.find((line, i) => line !== a[i]) ?? '(row count differs)';
          throw new Error(
            `sign pair ${key}: goal (${g.x},${g.y}) renders differently from ` +
              `(${referenceCell?.x},${referenceCell?.y}) — the frame leaks the distance ` +
              `the agent is never told. First differing row:\n  ${diff}`,
          );
        }
      }
    }
    module.dispose();
  });
});

describe('po-maze / the goal is exact and invisible', () => {
  test('NO state ever draws the moonflower — not even the goal one cell away', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    const bloom = objectNamed(module, 'moonflowerBloom');

    // the whole state space that matters: every free agent cell against every free goal
    // cell in its own 3 × 3 window — the states the old build betrayed
    let adjacentChecked = 0;
    for (const cell of freeCells()) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const g = { x: cell.x + dx, y: cell.y + dy };
          if (poMazeIsWall(g.x, g.y)) continue;
          module.update(toFrameUpdate(poseFixture('adjacent', cell.x, cell.y, g.x, g.y)));
          expect(bloom.visible).toBe(false);
          // …and the anchor is on the goal to the last bit while the art says nothing
          const w = worldOf(module, 'goal-anchor');
          cellToWorld(g.x, g.y, grid, truth);
          expect(w.x).toBeCloseTo(truth[0], 10);
          expect(w.z).toBeCloseTo(truth[2], 10);
          adjacentChecked++;
        }
      }
    }
    expect(adjacentChecked).toBeGreaterThan(300);
    module.dispose();
  });

  test('a +1 opens the flower ON THE CLEARING, and only for the breath', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    const bloom = objectNamed(module, 'moonflowerBloom');

    ctx.setTime(1);
    const before = toFrameUpdate(poseFixture('run', 5, 5, 5, 6, 800), { scrubbed: false });
    module.update(before);
    expect(bloom.visible).toBe(false);

    // the agent walks into (5,6); POMaze throws it to (1,9) and hides a new goal at (8,2)
    ctx.setTime(1.05);
    module.update(
      toFrameUpdate(poseFixture('scored', 1, 9, 8, 2, 801), {
        scrubbed: false,
        prev: before.state,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(bloom.visible).toBe(true);
    bloom.updateWorldMatrix(true, false);
    const w = new THREE.Vector3().setFromMatrixPosition(bloom.matrixWorld);
    cellToWorld(5, 6, grid, truth); // the cell the reward was EARNED on
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    // and NOT on the new goal (8,2), which nothing in the frame may name
    cellToWorld(8, 2, grid, truth);
    expect(Math.hypot(w.x - truth[0], w.z - truth[2])).toBeGreaterThan(1);

    // the breath closes and takes the flower with it
    ctx.setTime(12);
    module.update(toFrameUpdate(poseFixture('later', 1, 9, 8, 2, 802), { scrubbed: false }));
    expect(bloom.visible).toBe(false);
    module.dispose();
  });

  test('the anchor follows a respawning goal onto every free cell', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (const g of freeCells()) {
      module.update(toFrameUpdate(poseFixture('goalmove', 5, 5, g.x, g.y)));
      const w = worldOf(module, 'goal-anchor');
      cellToWorld(g.x, g.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
    }
    module.dispose();
  });
});

describe('po-maze / THE COMPASS CARRIES SIGNS, NEVER DISTANCE', () => {
  test('compassOffset is a pure function of the four sign bits and the clock', () => {
    const out = new THREE.Vector3();
    // it takes SIGNS: every magnitude with the same sign pair gives the same offset
    const reference = new THREE.Vector3();
    compassOffset(1, -1, 2.5, 0.7, reference);
    for (const [dx, dy] of [[1, -1], [4, -1], [1, -9], [7, -7], [10, -3]]) {
      compassOffset(dx ?? 0, dy ?? 0, 2.5, 0.7, out);
      expect(out.x).toBe(reference.x);
      expect(out.y).toBe(reference.y);
      expect(out.z).toBe(reference.z);
    }
    // the radius never depends on the direction either: at a given instant EVERY bearing
    // sits on one circle, so no pairing of moth position and time can encode a distance
    const radii: number[] = [];
    for (const [dx, dy] of [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]]) {
      compassOffset(dx ?? 0, dy ?? 0, 2.5, 0.7, out);
      radii.push(Math.hypot(out.x, out.z));
    }
    expect(Math.max(...radii) - Math.min(...radii)).toBeLessThan(1e-12);
    // and the bearing wander is bounded well inside the 45° that would blur an octant
    for (let k = 0; k <= 200; k++) {
      const t = (k / 200) * 40;
      compassOffset(1, 0, t, 0.7, out);
      const bearing = Math.atan2(out.z, out.x);
      expect(Math.abs(bearing)).toBeLessThanOrEqual(MOTH_SWING + 1e-12);
      expect(MOTH_SWING).toBeLessThan(Math.PI / 4);
      expect(Math.hypot(out.x, out.z)).toBeGreaterThan(MOTH_RIM - MOTH_BREATH - 1e-12);
    }
    // a goal on the agent's own cell has no direction at all
    expect(compassOffset(0, 0, 2.5, 0.7, out)).toBe(false);
  });

  test('the moth sits in the goal\'s true direction at a constant radius', async () => {
    const { module, ctx } = await live();
    ctx.setTime(0);
    const grid = resolveGrid(geometryOf());
    const wispPos: [number, number, number] = [0, 0, 0];
    const radii: number[] = [];
    /** every moth position seen for a given sign pair */
    const bySign = new Map<string, THREE.Vector3[]>();

    // EVERY free cell, adjacent ones included. A moth that flew in and landed on a goal
    // one cell away would be reporting a distance the observation does not carry, so the
    // Chebyshev ≤ 1 states are the ones this test most needs to cover, not skip.
    for (const goal of freeCells()) {
      if (goal.x === 5 && goal.y === 5) continue;
      module.update(toFrameUpdate(poseFixture('compass', 5, 5, goal.x, goal.y)));
      const moth = objectNamed(module, 'compassMoth');
      moth.updateWorldMatrix(true, false);
      const p = new THREE.Vector3().setFromMatrixPosition(moth.matrixWorld);
      cellToWorld(5, 5, grid, wispPos);
      const dx = p.x - wispPos[0];
      const dz = p.z - wispPos[2];

      // 1. the bearing is the sign vector's bearing, wandering at most MOTH_SWING
      const want = Math.atan2(Math.sign(goal.y - 5), Math.sign(goal.x - 5));
      let d = Math.atan2(dz, dx) - want;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      expect(Math.abs(d)).toBeLessThanOrEqual(MOTH_SWING + 1e-9);
      // 2. …which means the sign bits themselves are never misreported
      if (goal.x !== 5) expect(Math.sign(dx)).toBe(Math.sign(goal.x - 5));
      if (goal.y !== 5) expect(Math.sign(dz)).toBe(Math.sign(goal.y - 5));

      radii.push(Math.hypot(dx, dz));
      const key = `${Math.sign(goal.x - 5)},${Math.sign(goal.y - 5)}`;
      const bucket = bySign.get(key);
      if (bucket) bucket.push(p);
      else bySign.set(key, [p]);
    }

    expect(radii.length).toBe(64); // all 65 free cells but the one under the agent
    // EVERY goal, at EVERY range, puts the moth at the same radius: a moth that drifted
    // outward for a far goal (or inward for a near one) would be leaking the distance the
    // agent is never told
    expect(Math.max(...radii) - Math.min(...radii)).toBeLessThan(1e-9);
    // and the strongest form of the same claim — two goals that share a sign pair are
    // INDISTINGUISHABLE in the frame, however far apart they are
    expect(bySign.size).toBeGreaterThanOrEqual(8);
    for (const [, positions] of bySign) {
      const first = positions[0] as THREE.Vector3;
      for (const p of positions) expect(p.distanceTo(first)).toBeLessThan(1e-9);
    }
    module.dispose();
  });

  test('a goal on the agent\'s own cell leaves the compass with nothing to say', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(poseFixture('onGoal', 5, 5, 5, 5)));
    expect(objectNamed(module, 'compassMoth').visible).toBe(false);
    module.update(toFrameUpdate(poseFixture('offGoal', 5, 5, 9, 9)));
    expect(objectNamed(module, 'compassMoth').visible).toBe(true);
    module.dispose();
  });
});

describe('po-maze / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    // these roots carry POSITION ONLY — they must be exact, not merely within 1e-3
    expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('goal-anchor') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('gap-gate0') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('gap-gate1') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
      expect(report.worst.get('goal-anchor') ?? 1).toBeLessThan(1e-9);
      module.dispose();
    }
  });

  test('the wisp lands on EVERY free cell of the maze', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    let visited = 0;
    for (const cell of freeCells()) {
      module.update(toFrameUpdate(poseFixture(`cell${cell.x}-${cell.y}`, cell.x, cell.y, 9, 9, visited)));
      const w = worldOf(module, 'wisp-anchor');
      cellToWorld(cell.x, cell.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
      visited++;
    }
    expect(visited).toBe(65);
    module.dispose();
  });

  test('the gate stones sit on the env\'s own gap cells', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (let i = 0; i < POMAZE_GAPS.length; i++) {
      const g = POMAZE_GAPS[i] as GridCell;
      const root = objectNamed(module, `gapGateRoot${i}`);
      root.updateWorldMatrix(true, false);
      const w = new THREE.Vector3().setFromMatrixPosition(root.matrixWorld);
      cellToWorld(g.x, g.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
    }
    module.dispose();
  });

  test('readCells prefers the decoded state and falls back to the raw pose', () => {
    const grid = { w: POMAZE_W, h: POMAZE_H };
    const agent: GridCell = { x: 0, y: 0 };
    const goal: GridCell = { x: 0, y: 0 };
    readCells(
      { pose: new Float32Array([1, 2, 3, 4, 0, 0, 0, 0, 0]), ctrl: new Float32Array(6), done: false, step: 0, simTime: 0, derived: null },
      grid, agent, goal,
    );
    expect([agent.x, agent.y, goal.x, goal.y]).toEqual([1, 2, 3, 4]);
    readCells(
      { pose: new Float32Array(9), ctrl: new Float32Array(6), done: false, step: 0, simTime: 0, derived: { x: 7, y: 8, gx: 2, gy: 1 } },
      grid, agent, goal,
    );
    expect([agent.x, agent.y, goal.x, goal.y]).toEqual([7, 8, 2, 1]);
  });
});

describe('po-maze / the hop is bounded, never a path plan', () => {
  test('a hop stays inside the segment between the two cells and lands on truth', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('a', 4, 5, 9, 9, 0)));
    cellToWorld(4, 5, grid, from);
    cellToWorld(5, 5, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('b', 5, 5, 9, 9, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(poseFixture('b', 5, 5, 9, 9, 1), { scrubbed: false }));
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

  test('a scrub snaps the wisp with no glide at all', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
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

  test('a teleport is a hard cut too — POMaze throws the agent across the maze on every +1', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(6);
    module.update(toFrameUpdate(poseFixture('before', 8, 9, 9, 9, 10), { scrubbed: false }));
    ctx.setTime(6.125);
    module.update(
      toFrameUpdate(poseFixture('after', 2, 1, 5, 6, 11), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(2, 1, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    module.dispose();
  });
});

describe('po-maze / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('forest-start')));
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
      toFrameUpdate(poseFixture('scored', 3, 4, 6, 2, 101), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    module.dispose();
  });

  test('the breath opens on the cell the +1 was EARNED on, not on the agent\'s new one', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    // the agent walks toward a goal at (5,6)…
    ctx.setTime(3);
    const before = toFrameUpdate(poseFixture('run', 5, 5, 5, 6, 200), { scrubbed: false });
    module.update(before);
    // …enters it, and POMaze redraws BOTH: the agent lands on (1,9), the goal on (8,2)
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(poseFixture('scored', 1, 9, 8, 2, 201), {
        scrubbed: false,
        prev: before.state,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    // the breath light stands over (5,6) — where the reward happened — not (1,9) or (8,2)
    const light = module.scene.getObjectByName('oakPoint1');
    expect(light).toBeDefined();
    if (!light) return;
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(5, 6, grid, truth);
    expect(light.position.x).toBeCloseTo(truth[0], 10);
    expect(light.position.z).toBeCloseTo(truth[2], 10);
    expect((light as THREE.PointLight).intensity).toBeGreaterThan(0);
    module.dispose();
  });

  test('the breath lifts the fog and gives it back', async () => {
    const { module, ctx } = await live();
    const fog = module.scene.fog as THREE.FogExp2;
    expect(fog).toBeDefined();
    ctx.setTime(1);
    const before = toFrameUpdate(poseFixture('run', 5, 5, 5, 6, 300), { scrubbed: false });
    module.update(before);
    const standing = fog.density;
    expect(standing).toBeGreaterThan(0);

    ctx.setTime(1.1);
    module.update(
      toFrameUpdate(poseFixture('scored', 1, 9, 8, 2, 301), {
        scrubbed: false,
        prev: before.state,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(fog.density).toBeLessThan(standing);

    // …and it closes again, all the way back, with no residue
    ctx.setTime(12);
    module.update(toFrameUpdate(poseFixture('later', 1, 9, 8, 2, 400), { scrubbed: false }));
    expect(fog.density).toBeCloseTo(standing, 12);
    module.dispose();
  });

  test('the breath shows walls, but only for a breath and only around the clearing', async () => {
    const { module, ctx } = await live();
    ctx.setTime(1);
    const before = toFrameUpdate(poseFixture('run', 8, 2, 8, 3, 500), { scrubbed: false });
    module.update(before);
    ctx.setTime(1.05);
    module.update(
      toFrameUpdate(poseFixture('scored', 1, 1, 9, 9, 501), {
        scrubbed: false,
        prev: before.state,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    // more than the eight bits at (1,1) are lit: the clearing is open
    const duringBreath = shownWallCells(module);
    expect(duringBreath.size).toBeGreaterThan(observedWallCells(1, 1).size);
    // and once it has closed, the forest is back to exactly the observation
    ctx.setTime(12);
    module.update(toFrameUpdate(poseFixture('after', 1, 1, 9, 9, 600), { scrubbed: false }));
    expect([...shownWallCells(module)].sort()).toEqual([...observedWallCells(1, 1)].sort());
    module.dispose();
  });

  test('a scrub clears every in-flight beat', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    const fog = module.scene.fog as THREE.FogExp2;
    ctx.setTime(1);
    module.update(
      toFrameUpdate(poseFixture('scored', 1, 3, 9, 9, 700), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    const lifted = fog.density;
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('forest-start'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    expect(fog.density).toBeGreaterThan(lifted);
    module.dispose();
  });

  test('the envelope is pure, bounded and lands on 0', () => {
    expect(envelope(-Infinity, 2, 1.6)).toBe(0);
    expect(envelope(Number.NaN, 2, 1.6)).toBe(0);
    expect(envelope(0, 2, 1.6)).toBe(1);
    expect(envelope(99, 2, 1.6)).toBe(0);
    expect(envelope(2, 2, 1.6)).toBe(0);
    for (let i = 0; i <= 50; i++) {
      const e = envelope((i / 50) * 2, 2, 1.6);
      expect(e).toBeGreaterThanOrEqual(0);
      expect(e).toBeLessThanOrEqual(1);
    }
  });
});

describe('po-maze / lifecycle', () => {
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
    module.update(toFrameUpdate(fixtureNamed('north-gap')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'forestGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'forestGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the camera is the lowest and closest in the preserve, and it rides the wisp', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    module.update(toFrameUpdate(fixtureNamed('forest-start')));
    const cam = module.camera;
    // four-rooms sits 12.5 units up and 17.8 out; this one is a spirit's-eye view
    expect(cam.position.y).toBeLessThan(4);
    expect(cam.position.y).toBeGreaterThan(1);
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(POMAZE_START.x, POMAZE_START.y, grid, truth);
    const standoff = Math.hypot(cam.position.x - truth[0], cam.position.z - truth[2]);
    expect(standoff).toBeLessThan(8);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.y).toBeLessThan(0); // still looking down at the litter
    expect(dir.y).toBeGreaterThan(-0.5); // but nowhere near an aerial
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);

    // and it follows: the same camera over a cell in the far corner
    module.update(toFrameUpdate(poseFixture('corner', 9, 9, 1, 1)));
    cellToWorld(9, 9, grid, truth);
    const standoffB = Math.hypot(cam.position.x - truth[0], cam.position.z - truth[2]);
    expect(standoffB).toBeLessThan(8);
    module.dispose();
  });

  test('the eye CUTS across a teleport and out of the linger — it never pans', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const cam = module.camera;
    const at: [number, number, number] = [0, 0, 0];
    /** how far the eye stands from the cell it is framing, once settled */
    const standoff = (x: number, y: number): number => {
      cellToWorld(x, y, grid, at);
      return Math.hypot(cam.position.x - at[0], cam.position.z - at[2]);
    };

    // settle on (8,9), one step short of the goal at (9,9)
    ctx.setTime(20);
    let before = toFrameUpdate(poseFixture('run', 8, 9, 9, 9, 900));
    module.update(before);
    for (let i = 0; i < 30; i++) {
      ctx.setTime(20 + i * 0.05);
      before = toFrameUpdate(poseFixture('run', 8, 9, 9, 9, 900), { scrubbed: false });
      module.update(before);
    }
    const settled = standoff(8, 9);
    expect(settled).toBeGreaterThan(1);

    // the +1: the agent is thrown to (1,1), a new goal is hidden at (5,6), and the eye
    // must be ON the clearing (9,9) on this very frame — not a frame of the way there
    const scoreTime = 21.6;
    ctx.setTime(scoreTime);
    module.update(
      toFrameUpdate(poseFixture('scored', 1, 1, 5, 6, 901), {
        scrubbed: false,
        prev: before.state,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(standoff(9, 9)).toBeCloseTo(settled, 9);
    const cutPos = cam.position.clone();

    // …and it HOLDS there: no drift while the fog breathes
    for (let i = 1; i <= 8; i++) {
      ctx.setTime(scoreTime + i * 0.05);
      module.update(toFrameUpdate(poseFixture('hold', 1, 1, 5, 6, 902), { scrubbed: false }));
      expect(cam.position.distanceTo(cutPos)).toBeLessThan(1e-9);
    }

    // the linger ends: ONE frame later the eye is framed on the wisp, eleven cells away.
    // An exponential drift at CAM_FOLLOW_RATE would have covered 3.6% of that in a frame.
    ctx.setTime(scoreTime + LINGER_SECONDS + 1e-3);
    module.update(toFrameUpdate(poseFixture('after', 1, 1, 5, 6, 903), { scrubbed: false }));
    expect(standoff(1, 1)).toBeCloseTo(settled, 9);
    module.dispose();
  });

  test('a scrub cuts the eye too', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const cam = module.camera;
    const at: [number, number, number] = [0, 0, 0];
    ctx.setTime(30);
    module.update(toFrameUpdate(poseFixture('a', 1, 1, 9, 9, 1000)));
    cellToWorld(1, 1, grid, at);
    const settled = Math.hypot(cam.position.x - at[0], cam.position.z - at[2]);
    module.update(toFrameUpdate(poseFixture('far', 9, 9, 1, 1, 5000), { scrubbed: true }));
    cellToWorld(9, 9, grid, at);
    expect(Math.hypot(cam.position.x - at[0], cam.position.z - at[2])).toBeCloseTo(settled, 9);
    module.dispose();
  });

  test('THE THUMB IS A HELD SHOT: the card never moves, and it frames the thumbFrame', async () => {
    const { module, ctx } = await live('thumb');
    const grid = resolveGrid(geometryOf());
    const cam = module.camera;
    module.update(toFrameUpdate(fixtureNamed('forest-start')));
    cam.updateMatrixWorld(true);
    const pose = cam.matrixWorld.clone();

    // every free cell, a hop chain, a scrub and a full +1 — the card holds all of it
    let frames = 0;
    for (const cell of freeCells()) {
      ctx.setTime(50 + frames * 0.05);
      const f = toFrameUpdate(poseFixture('walk', cell.x, cell.y, 9, 9, frames), { scrubbed: false });
      module.update(f);
      cam.updateMatrixWorld(true);
      expect(cam.matrixWorld.elements).toEqual(pose.elements);
      if (frames === 20) {
        module.update(
          toFrameUpdate(poseFixture('scored', 1, 1, 5, 6, frames), {
            scrubbed: false,
            prev: f.state,
            events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
          }),
        );
        cam.updateMatrixWorld(true);
        expect(cam.matrixWorld.elements).toEqual(pose.elements);
      }
      frames++;
    }
    expect(frames).toBe(65);

    // and the shot it holds is the manifest's own thumbFrame cell, dead ahead
    const at: [number, number, number] = [0, 0, 0];
    cellToWorld(manifest.thumbFrame.pose[0] ?? 0, manifest.thumbFrame.pose[1] ?? 0, grid, at);
    const toCell = new THREE.Vector2(at[0] - cam.position.x, at[2] - cam.position.z).normalize();
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    const look = new THREE.Vector2(dir.x, dir.z).normalize();
    expect(toCell.dot(look)).toBeGreaterThan(0.999);
    module.dispose();
  });

  test('the camera never wanders into the outer forest', async () => {
    const { module } = await live();
    let worst = 0;
    for (const cell of freeCells()) {
      module.update(toFrameUpdate(poseFixture('walk', cell.x, cell.y, 5, 5)));
      worst = Math.max(worst, Math.hypot(module.camera.position.x, module.camera.position.z));
    }
    // the ring starts at 13 (world.ts OUTER_FOREST_INNER): nothing between eye and physics
    expect(worst).toBeLessThan(13);
    module.dispose();
  });

  test('the start pose really is the env\'s deterministic init', () => {
    const start = fixtureNamed('forest-start');
    expect(start.pose[0]).toBe(POMAZE_START.x);
    expect(start.pose[1]).toBe(POMAZE_START.y);
    expect(start.pose[2]).toBe(POMAZE_GOAL.x);
    expect(start.pose[3]).toBe(POMAZE_GOAL.y);
  });

  test('every tier renders inside its declared budget', async () => {
    const heroLive = await live('hero');
    heroLive.module.update(toFrameUpdate(fixtureNamed('north-gap')));
    const hero = census(heroLive.module, true);
    expect(hero.draws).toBeLessThanOrEqual(manifest.budget.drawCalls);
    expect(hero.triangles).toBeLessThanOrEqual(manifest.budget.triangles);
    expect(hero.draws).toBeLessThanOrEqual(300);
    expect(hero.triangles).toBeLessThanOrEqual(500_000);
    heroLive.module.dispose();

    const thumbLive = await live('thumb');
    thumbLive.module.update(toFrameUpdate(fixtureNamed('north-gap')));
    const thumb = census(thumbLive.module, false);
    // STYLE.md's thumb contract
    expect(thumb.draws).toBeLessThanOrEqual(40);
    expect(thumb.triangles).toBeLessThanOrEqual(50_000);
    thumbLive.module.dispose();
  });

  test('the wisp height and cell size are the shared tokens, untouched', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('forest-start')));
    const core = module.scene.getObjectByName('oakWispCore');
    expect(core).toBeDefined();
    core?.updateWorldMatrix(true, false);
    const p = new THREE.Vector3().setFromMatrixPosition(core?.matrixWorld ?? new THREE.Matrix4());
    expect(p.y).toBeCloseTo(TOKENS.wispHeight, 10);
    expect(CELL).toBe(TOKENS.cellSize);
    module.dispose();
  });
});
