/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { CELL, cellToWorld, isWall } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  FOURROOMS_FREE_CELLS,
  FOURROOMS_GOAL,
  FOURROOMS_H,
  FOURROOMS_HALLWAYS,
  FOURROOMS_START,
  FOURROOMS_W,
  fourRoomsDecoder,
  fourRoomsIsWall,
  fourRoomsRoom,
} from '@sim/envs/fourrooms';
import type { EnvGeometry } from '@sim/types';

import createScene, { growth, opening } from '../../src/scenes/four-rooms/index';
import { fixtures } from '../../src/scenes/four-rooms/fixtures';
import { manifest } from '../../src/scenes/four-rooms/manifest';
import {
  COURT_HALF,
  courtyardSkin,
  makeMeadowHeight,
  resolveGrid,
  roomBounds,
} from '../../src/scenes/four-rooms/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 8;

function geometryOf(): EnvGeometry {
  return fourRoomsDecoder.geometry(fourRoomsDecoder.initStatic(SEED), SEED);
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
    derived: { x, y, gx, gy, room: fourRoomsRoom(x, y) },
    step,
  };
}

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}


describe('four-rooms / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('FourRooms-cont-v0');
    expect(manifest.slug).toBe('four-rooms');
    expect(manifest.accent).toBe('four-rooms');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a legal courtyard state', () => {
    const p = manifest.thumbFrame.pose;
    expect(fourRoomsIsWall(p[0] ?? -1, p[1] ?? -1)).toBe(false);
    expect(fourRoomsIsWall(p[2] ?? -1, p[3] ?? -1)).toBe(false);
  });

  test('ships at least three canonical poses, all on FREE cells', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      const x = f.pose[0] ?? -1;
      const y = f.pose[1] ?? -1;
      const gx = f.pose[2] ?? -1;
      const gy = f.pose[3] ?? -1;
      expect(fourRoomsIsWall(x, y)).toBe(false);
      expect(fourRoomsIsWall(gx, gy)).toBe(false);
    }
  });

  test('the fixtures cover a hallway agent and a hallway goal (room === -1)', () => {
    const inHallway = fixtures.some((f) => fourRoomsRoom(f.pose[0] ?? 0, f.pose[1] ?? 0) === -1);
    const goalInHallway = fixtures.some((f) => fourRoomsRoom(f.pose[2] ?? 0, f.pose[3] ?? 0) === -1);
    expect(inHallway).toBe(true);
    expect(goalInHallway).toBe(true);
  });
});

describe('four-rooms / the layout is read, never re-derived', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry) === geometry.grid).toBe(true);
  });

  test('a placeholder geometry falls back to the DECODER\'s own walls', () => {
    const grid = resolveGrid({ kind: 'grid' });
    expect(grid.w).toBe(FOURROOMS_W);
    expect(grid.h).toBe(FOURROOMS_H);
    let free = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        expect(isWall(grid, x, y)).toBe(fourRoomsIsWall(x, y));
        if (!isWall(grid, x, y)) free++;
      }
    }
    expect(free).toBe(FOURROOMS_FREE_CELLS);
  });

  test('roomBounds finds all four rooms, and only free cells of that room', () => {
    const grid = resolveGrid(geometryOf());
    let total = 0;
    for (let room = 0; room < 4; room++) {
      const b = roomBounds(grid, room);
      expect(b.cells).toBe(16);
      total += b.cells;
      expect(b.halfX).toBeCloseTo(2 * CELL, 10);
      expect(b.halfZ).toBeCloseTo(2 * CELL, 10);
      expect(Math.abs(b.cx)).toBeCloseTo(2.5 * CELL, 10);
      expect(Math.abs(b.cz)).toBeCloseTo(2.5 * CELL, 10);
      expect(Math.abs(b.cx) + b.halfX).toBeLessThanOrEqual(COURT_HALF + 1e-9);
      expect(Math.abs(b.cz) + b.halfZ).toBeLessThanOrEqual(COURT_HALF + 1e-9);
    }
    expect(total + FOURROOMS_HALLWAYS.length).toBe(FOURROOMS_FREE_CELLS);
  });

  test('the seasonal skin is a pure read of fourRoomsRoom: four rooms, four looks', () => {
    const skin = courtyardSkin(TOKENS.palettes.noon, TOKENS);
    expect(skin.seasons.length).toBe(4);
    const hexes = new Set(skin.seasons.map((s) => s.floor.getHex()));
    expect(hexes.size).toBe(4);
    for (const s of skin.seasons) expect(s.hedge.getHex()).not.toBe(skin.evergreen.getHex());
  });

  test('the meadow is identically flat under the whole courtyard', () => {
    const h = makeMeadowHeight(SEED);
    for (let i = 0; i <= 40; i++) {
      for (let j = 0; j <= 40; j++) {
        const x = -COURT_HALF + (2 * COURT_HALF * i) / 40;
        const z = -COURT_HALF + (2 * COURT_HALF * j) / 40;
        expect(h(x, z)).toBe(0);
      }
    }
    let wander = 0;
    for (const p of [-16, -13, 13, 16]) wander = Math.max(wander, Math.abs(h(p, p)));
    expect(wander).toBeGreaterThan(0.05);
  });
});

describe('four-rooms / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('goal-lantern') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('moon-gate') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('sundial') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      module.dispose();
    }
  });

  test('the wisp lands on EVERY free cell of the courtyard', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
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
    expect(visited).toBe(FOURROOMS_FREE_CELLS);
    module.dispose();
  });

  test('the lantern follows a respawning goal onto every free cell, hallways included', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (const g of [FOURROOMS_GOAL, ...FOURROOMS_HALLWAYS, { x: 1, y: 9 }, { x: 9, y: 1 }]) {
      module.update(toFrameUpdate(poseFixture('goalmove', 3, 3, g.x, g.y)));
      const w = worldOf(module, 'goal-lantern');
      cellToWorld(g.x, g.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
    }
    module.dispose();
  });

  test('the moon gates sit on the env\'s own hallway cells', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (let i = 0; i < FOURROOMS_HALLWAYS.length; i++) {
      const h = FOURROOMS_HALLWAYS[i];
      if (!h) continue;
      const root = module.scene.getObjectByName(`moonGateRoot${i}`);
      expect(root).toBeDefined();
      if (!root) continue;
      root.updateWorldMatrix(true, false);
      const w = new THREE.Vector3().setFromMatrixPosition(root.matrixWorld);
      cellToWorld(h.x, h.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
      expect(fourRoomsIsWall(h.x, h.y)).toBe(false);
      expect(fourRoomsRoom(h.x, h.y)).toBe(-1);
    }
    module.dispose();
  });
});

describe('four-rooms / the hop is bounded, never a path plan', () => {
  test('a hop stays inside the segment between the two cells and lands on truth', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('a', 1, 1, 9, 9, 0)));
    cellToWorld(1, 1, grid, from);
    cellToWorld(2, 1, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('b', 2, 1, 9, 9, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(poseFixture('b', 2, 1, 9, 9, 1), { scrubbed: false }));
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
});

describe('four-rooms / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('courtyard-start')));
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

    ctx.setTime(1);
    module.update(toFrameUpdate(poseFixture('onGoal', 9, 9, 9, 9, 100), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    ctx.setTime(2);
    module.update(
      toFrameUpdate(poseFixture('scored', 9, 9, 3, 7, 101), { scrubbed: false, events: [{ type: 'goal' }] }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    module.dispose();
  });

  test('the bloom happens where the agent is, not where the goal respawned', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    ctx.setTime(3);
    module.update(toFrameUpdate(poseFixture('run', 8, 8, 9, 9, 200), { scrubbed: false }));
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(poseFixture('scored', 9, 9, 1, 3, 201), { scrubbed: false, events: [{ type: 'goal' }] }),
    );
    const light = module.scene.getObjectByName('oakPoint2');
    expect(light).toBeDefined();
    if (!light) return;
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(9, 9, grid, truth);
    expect(light.position.x).toBeCloseTo(truth[0], 10);
    expect(light.position.z).toBeCloseTo(truth[2], 10);
    expect((light as THREE.PointLight).intensity).toBeGreaterThan(0);
    const lantern = worldOf(module, 'goal-lantern');
    cellToWorld(1, 3, grid, truth);
    expect(lantern.x).toBeCloseTo(truth[0], 10);
    expect(lantern.z).toBeCloseTo(truth[2], 10);
    const bud = module.scene.getObjectByName('lanternBud');
    expect(bud?.scale.x ?? 1).toBeLessThan(0.2);
    module.dispose();
  });

  test('a scrub clears every in-flight beat and settles the new lantern', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(poseFixture('scored', 9, 9, 1, 3, 300), { scrubbed: false, events: [{ type: 'goal' }] }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('courtyard-start'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    const bud = module.scene.getObjectByName('lanternBud');
    expect(bud?.scale.x ?? 0).toBeCloseTo(1, 6);
    module.dispose();
  });

  test('the growth and opening envelopes are pure, bounded and land on 1', () => {
    expect(growth(-Infinity)).toBe(1);
    expect(growth(Number.NaN)).toBe(1);
    expect(growth(0)).toBe(0);
    expect(growth(99)).toBe(1);
    for (let i = 0; i <= 50; i++) {
      const g = growth((i / 50) * 0.72);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(1.05);
    }
    const far = opening(0, { x: 1, y: 1 }, { x: 9, y: 9 });
    const near = opening(0, { x: 9, y: 8 }, { x: 9, y: 9 });
    expect(far).toBeGreaterThanOrEqual(0);
    expect(near).toBeLessThanOrEqual(1);
    expect(near).toBeGreaterThan(far);
  });
});

describe('four-rooms / lifecycle', () => {
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
    module.update(toFrameUpdate(fixtureNamed('winter-room')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'courtyardGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'courtyardGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the camera stays a clamped 3/4 aerial over the courtyard', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('courtyard-start')));
    const cam = module.camera;
    expect(cam.position.y).toBeGreaterThan(6);
    expect(Math.hypot(cam.position.x, cam.position.z)).toBeGreaterThan(COURT_HALF);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.y).toBeLessThan(-0.5);
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    module.dispose();
  });

  test('the start pose really is the env\'s deterministic init', () => {
    const start = fixtureNamed('courtyard-start');
    expect(start.pose[0]).toBe(FOURROOMS_START.x);
    expect(start.pose[1]).toBe(FOURROOMS_START.y);
    expect(start.pose[2]).toBe(FOURROOMS_GOAL.x);
    expect(start.pose[3]).toBe(FOURROOMS_GOAL.y);
  });
});

describe('four-rooms / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'four-rooms (thumb, overlay on)');
    module.dispose();
  });
});
