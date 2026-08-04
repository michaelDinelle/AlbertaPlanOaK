/// <reference types="bun" />
// The Ivy Gate (KeyCorridor-cont-v0) — the accuracy + lifecycle self-check.
//
// This mirrors four-rooms.test.ts's shape and adds what THIS env has that FourRooms did
// not: a live key cell whose token is carried, a live door bit that drives a two-beat
// animation, and a goal that teleports the agent across the garden in the same frame it
// pays out. Each of those is a place a scene can silently lie, so each has its own lock:
//
//   * every probe is EXACT (< 1e-9), at all three tiers, on every free cell;
//   * the seed's dish tracks `(kx, ky)` whether the seed is in it or on the wisp, and the
//     token is provably reparented in both directions;
//   * the gate's phase is a pure function of `(doorOpen, age)` that lands on 0 or 1;
//   * the goal bloom lands on the PREVIOUS frame's goal cell, not the respawned one;
//   * a `teleport` hard-cuts the wisp instead of gliding it through the wall.

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { cellToWorld, isWall } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  KEYCORRIDOR_DOOR,
  KEYCORRIDOR_GOAL,
  KEYCORRIDOR_H,
  KEYCORRIDOR_KEY,
  KEYCORRIDOR_START,
  KEYCORRIDOR_W,
  keyCorridorDecoder,
  keyCorridorIsWall,
  keyCorridorRoomCells,
} from '@sim/envs/keycorridor';
import type { EnvGeometry, GridCell } from '@sim/types';

import createScene, {
  GATE_OPEN_SECONDS,
  GATE_RELOCK_SECONDS,
  gatePhase,
  growth,
  readState,
  makeGardenState,
  shrineGlow,
} from '../../src/scenes/key-corridor/index';
import { fixtures } from '../../src/scenes/key-corridor/fixtures';
import { manifest } from '../../src/scenes/key-corridor/manifest';
import { DOOR_FACE_Z, IVY_FRAME_SPAN, SEED_HOVER, vinePoint } from '../../src/scenes/key-corridor/actors';
import {
  GARDEN_HALF_X,
  GARDEN_HALF_Z,
  SIDE_EAST,
  SIDE_WEST,
  gardenSide,
  gardenSkin,
  gateYaw,
  isDividerWall,
  makeMeadowHeight,
  resolveDoor,
  resolveGrid,
  sideBounds,
} from '../../src/scenes/key-corridor/world';
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

const SEED = 24; // the demo run's first KeyCorridor lane

function geometryOf(): EnvGeometry {
  return keyCorridorDecoder.geometry(keyCorridorDecoder.initStatic(SEED), SEED);
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

/** pose = [x, y, gx, gy, kx, ky, hasKey, doorOpen, 0] */
function poseFixture(
  name: string,
  x: number,
  y: number,
  gx: number,
  gy: number,
  kx: number,
  ky: number,
  hasKey = false,
  doorOpen = false,
  step = 0,
): SceneFixture {
  return {
    name,
    pose: [x, y, gx, gy, kx, ky, hasKey ? 1 : 0, doorOpen ? 1 : 0, 0],
    ctrl: [0, 0, 0, 0, 0, 0],
    derived: { x, y, gx, gy, kx, ky, hasKey, doorOpen },
    step,
  };
}

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

function objectWorld(module: SceneModule, name: string): THREE.Vector3 {
  const object = module.scene.getObjectByName(name);
  if (!object) throw new Error(`object "${name}" missing`);
  object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(object.matrixWorld);
}

/** how many ivy sprigs the vine currently draws (a retracted sprig is scaled to zero) */
function liveSprigCount(vine: THREE.InstancedMesh): () => number {
  const m = new THREE.Matrix4();
  const s = new THREE.Vector3();
  return () => {
    let n = 0;
    for (let i = 0; i < vine.count; i++) {
      vine.getMatrixAt(i, m);
      s.setFromMatrixScale(m);
      if (s.x > 1e-6) n++;
    }
    return n;
  };
}

/** how much vine is drawn in total — a continuous measure of the wrap */
function vineExtent(vine: THREE.InstancedMesh): () => number {
  const m = new THREE.Matrix4();
  const s = new THREE.Vector3();
  return () => {
    let total = 0;
    for (let i = 0; i < vine.count; i++) {
      vine.getMatrixAt(i, m);
      s.setFromMatrixScale(m);
      total += s.x;
    }
    return total;
  };
}

function isDescendantOf(object: THREE.Object3D, ancestorName: string): boolean {
  let node: THREE.Object3D | null = object.parent;
  while (node) {
    if (node.name === ancestorName) return true;
    node = node.parent;
  }
  return false;
}

// ---------------------------------------------------------------------------

describe('key-corridor / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('KeyCorridor-cont-v0');
    expect(manifest.slug).toBe('key-corridor');
    expect(manifest.accent).toBe('key-corridor');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a legal garden state: free cells, key west, goal east', () => {
    const p = manifest.thumbFrame.pose;
    const door = KEYCORRIDOR_DOOR;
    expect(keyCorridorIsWall(p[0] ?? -1, p[1] ?? -1)).toBe(false);
    expect(keyCorridorIsWall(p[2] ?? -1, p[3] ?? -1)).toBe(false);
    expect(keyCorridorIsWall(p[4] ?? -1, p[5] ?? -1)).toBe(false);
    // the agent and the seed start on the wild side, the shrine on the formal one
    expect(gardenSide(door, p[0] ?? 0)).toBe(SIDE_WEST);
    expect(gardenSide(door, p[4] ?? 0)).toBe(SIDE_WEST);
    expect(gardenSide(door, p[2] ?? 0)).toBe(SIDE_EAST);
    // locked and unpicked: the state the gate is most itself in
    expect(p[6]).toBe(0);
    expect(p[7]).toBe(0);
  });

  test('ships at least three canonical poses, every cell legal and on the right side', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    const door = KEYCORRIDOR_DOOR;
    const west = new Set(keyCorridorRoomCells(1).map((c) => `${c.x},${c.y}`));
    const east = new Set(keyCorridorRoomCells(5).map((c) => `${c.x},${c.y}`));
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      const [x, y, gx, gy, kx, ky] = [f.pose[0] ?? -1, f.pose[1] ?? -1, f.pose[2] ?? -1, f.pose[3] ?? -1, f.pose[4] ?? -1, f.pose[5] ?? -1];
      expect(keyCorridorIsWall(x, y)).toBe(false);
      expect(keyCorridorIsWall(gx, gy)).toBe(false);
      expect(keyCorridorIsWall(kx, ky)).toBe(false);
      // the env draws the key from the LEFT room list and the goal from the RIGHT one
      expect(west.has(`${kx},${ky}`)).toBe(true);
      expect(east.has(`${gx},${gy}`)).toBe(true);
      void door;
    }
  });

  test('the fixtures cover carried / uncarried and open / shut, plus the door cell itself', () => {
    const carried = fixtures.some((f) => (f.pose[6] ?? 0) !== 0);
    const uncarried = fixtures.some((f) => (f.pose[6] ?? 0) === 0);
    const open = fixtures.some((f) => (f.pose[7] ?? 0) !== 0);
    const shut = fixtures.some((f) => (f.pose[7] ?? 0) === 0);
    const inDoor = fixtures.some((f) => f.pose[0] === KEYCORRIDOR_DOOR.x && f.pose[1] === KEYCORRIDOR_DOOR.y);
    expect(carried && uncarried && open && shut && inDoor).toBe(true);
  });

  test("the start pose really is the env's deterministic init", () => {
    const start = fixtureNamed('garden-start');
    expect(start.pose[0]).toBe(KEYCORRIDOR_START.x);
    expect(start.pose[1]).toBe(KEYCORRIDOR_START.y);
    expect(start.pose[2]).toBe(KEYCORRIDOR_GOAL.x);
    expect(start.pose[3]).toBe(KEYCORRIDOR_GOAL.y);
    expect(start.pose[4]).toBe(KEYCORRIDOR_KEY.x);
    expect(start.pose[5]).toBe(KEYCORRIDOR_KEY.y);
  });
});

describe('key-corridor / the layout is read, never re-derived', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry) === geometry.grid).toBe(true);
  });

  test("a placeholder geometry falls back to the DECODER's own walls", () => {
    const grid = resolveGrid({ kind: 'grid' });
    expect(grid.w).toBe(KEYCORRIDOR_W);
    expect(grid.h).toBe(KEYCORRIDOR_H);
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        expect(isWall(grid, x, y)).toBe(keyCorridorIsWall(x, y));
      }
    }
    expect(resolveDoor(grid)).toEqual({ x: KEYCORRIDOR_DOOR.x, y: KEYCORRIDOR_DOOR.y });
  });

  test('the door comes from the env geometry, and it is a FREE cell inside a wall', () => {
    const grid = resolveGrid(geometryOf());
    const door = resolveDoor(grid);
    expect(door).toEqual({ x: KEYCORRIDOR_DOOR.x, y: KEYCORRIDOR_DOOR.y });
    // canon lesson 14: the lock lives in the STATE, not in the walls array
    expect(isWall(grid, door.x, door.y)).toBe(false);
    expect(keyCorridorIsWall(door.x, door.y)).toBe(false);
    // and its north/south neighbours are wall, which is what makes it a gate
    expect(isWall(grid, door.x, door.y - 1)).toBe(true);
    expect(isWall(grid, door.x, door.y + 1)).toBe(true);
  });

  test('the gate yaw is READ off the flanking wall cells (lesson 15)', () => {
    const grid = resolveGrid(geometryOf());
    const door = resolveDoor(grid);
    // this wall runs along z, so the gate is turned a quarter turn from the local build
    expect(gateYaw(grid, door)).toBeCloseTo(Math.PI / 2, 12);
    // a synthetic door flanked EAST and WEST sits in a wall running along x, and must
    // NOT be turned — the same code has to orient a regenerated layout
    const flat = { w: 3, h: 3, walls: Uint8Array.from([0, 0, 0, 1, 0, 1, 0, 0, 0]) };
    expect(gateYaw(flat, { x: 1, y: 1 })).toBe(0);
  });

  test('isDividerWall finds the interior column and nothing else', () => {
    const grid = resolveGrid(geometryOf());
    const cells: GridCell[] = [];
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (isDividerWall(grid, x, y)) cells.push({ x, y });
      }
    }
    // x = 4 for y = 1, 2, 4, 5 — the door (4,3) is free and (4,0)/(4,6) are border
    expect(cells).toEqual([
      { x: 4, y: 1 },
      { x: 4, y: 2 },
      { x: 4, y: 4 },
      { x: 4, y: 5 },
    ]);
    for (const c of cells) expect(isWall(grid, c.x, c.y)).toBe(true);
  });

  test('gardenSide partitions exactly the two rooms the env draws from', () => {
    const grid = resolveGrid(geometryOf());
    const door = resolveDoor(grid);
    for (const c of keyCorridorRoomCells(1)) expect(gardenSide(door, c.x)).toBe(SIDE_WEST);
    for (const c of keyCorridorRoomCells(5)) expect(gardenSide(door, c.x)).toBe(SIDE_EAST);
    expect(gardenSide(door, door.x)).toBe(0);

    const west = sideBounds(grid, door, SIDE_WEST);
    const east = sideBounds(grid, door, SIDE_EAST);
    expect(west.cells).toBe(15);
    expect(east.cells).toBe(15);
    // each room is 3 x 5 cells, so ±1.5 / ±2.5 world units plus half a cell of margin
    for (const b of [west, east]) {
      expect(b.halfX).toBeCloseTo(1.5, 10);
      expect(b.halfZ).toBeCloseTo(2.5, 10);
      expect(Math.abs(b.cx) + b.halfX).toBeLessThanOrEqual(GARDEN_HALF_X + 1e-9);
      expect(Math.abs(b.cz) + b.halfZ).toBeLessThanOrEqual(GARDEN_HALF_Z + 1e-9);
    }
    // and they really are on opposite sides of the origin
    expect(west.cx).toBeLessThan(0);
    expect(east.cx).toBeGreaterThan(0);
  });

  test('the skin gives the two gardens, the threshold and the ivy four distinct looks', () => {
    const skin = gardenSkin(TOKENS.palettes.morning, TOKENS);
    const hexes = new Set([
      skin.wildFloor.getHex(),
      skin.formalFloor.getHex(),
      skin.threshold.getHex(),
      skin.ivy.getHex(),
    ]);
    expect(hexes.size).toBe(4);
    // the divider is the darkest stone in the garden — that is how the plan reads
    expect(luma(skin.divider)).toBeLessThan(luma(skin.kerb));
    // and every ground albedo respects the bright-key law (~140–175 sRGB, never 220)
    for (const c of [skin.wildFloor, skin.formalFloor, skin.threshold]) {
      expect(luma(c)).toBeLessThan(0.72);
    }
  });

  test('the meadow is identically flat under the whole garden', () => {
    const h = makeMeadowHeight(SEED);
    for (let i = 0; i <= 30; i++) {
      for (let j = 0; j <= 30; j++) {
        const x = -GARDEN_HALF_X + (2 * GARDEN_HALF_X * i) / 30;
        const z = -GARDEN_HALF_Z + (2 * GARDEN_HALF_Z * j) / 30;
        expect(h(x, z)).toBe(0);
      }
    }
    let wander = 0;
    for (const p of [-16, -13, 13, 16]) wander = Math.max(wander, Math.abs(h(p, p)));
    expect(wander).toBeGreaterThan(0.05);
  });
});

function luma(c: THREE.Color): number {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

describe('key-corridor / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    // these roots carry POSITION ONLY — they should be exact, not merely within 1e-3
    for (const name of ['wisp', 'wisp-anchor', 'shrine', 'seed-bed', 'gate']) {
      expect(report.worst.get(name) ?? 1).toBeLessThan(1e-9);
    }
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      for (const name of ['wisp', 'wisp-anchor', 'shrine', 'seed-bed', 'gate']) {
        expect(report.worst.get(name) ?? 1).toBeLessThan(1e-9);
      }
      module.dispose();
    }
  });

  test('the wisp lands on EVERY free cell of the garden', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    let visited = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (isWall(grid, x, y)) continue;
        module.update(toFrameUpdate(poseFixture(`cell${x}-${y}`, x, y, 7, 1, 3, 5, false, false, visited)));
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
    // 15 + 15 room cells + the door cell
    expect(visited).toBe(31);
    module.dispose();
  });

  test('the shrine follows a respawning goal onto every cell of the formal garden', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (const g of keyCorridorRoomCells(5)) {
      module.update(toFrameUpdate(poseFixture('goalmove', 2, 2, g.x, g.y, 3, 5)));
      const w = worldOf(module, 'shrine');
      cellToWorld(g.x, g.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
    }
    module.dispose();
  });

  test('the seed dish follows a respawning key onto every cell of the wild garden', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (const k of keyCorridorRoomCells(1)) {
      module.update(toFrameUpdate(poseFixture('keymove', 6, 2, 7, 1, k.x, k.y)));
      const w = worldOf(module, 'seed-bed');
      cellToWorld(k.x, k.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
    }
    module.dispose();
  });

  test("the gate sits on the env's own door cell at every tier", async () => {
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(KEYCORRIDOR_DOOR.x, KEYCORRIDOR_DOOR.y, grid, truth);
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module } = await live(tier);
      module.update(toFrameUpdate(fixtureNamed('garden-start')));
      const w = worldOf(module, 'gate');
      expect(w.x).toBeCloseTo(truth[0], 12);
      expect(w.y).toBeCloseTo(0, 12);
      expect(w.z).toBeCloseTo(truth[2], 12);
      module.dispose();
    }
  });
});

describe('key-corridor / the seed is carried, not copied', () => {
  test('uncarried, the token hovers EXACTLY over the key cell', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    for (const k of [{ x: 1, y: 1 }, { x: 3, y: 5 }, { x: 2, y: 4 }]) {
      module.update(toFrameUpdate(poseFixture('drop', 6, 2, 7, 1, k.x, k.y, false, true)));
      const w = objectWorld(module, 'amberSeed');
      cellToWorld(k.x, k.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
      expect(w.y).toBeCloseTo(SEED_HOVER, 10);
    }
    module.dispose();
  });

  test('carried, the ONE token is reparented under the wisp and comes home again', async () => {
    const { module } = await live();
    const token = module.scene.getObjectByName('amberSeed');
    expect(token).toBeDefined();
    if (!token) return;

    module.update(toFrameUpdate(poseFixture('home', 1, 1, 7, 1, 3, 5, false, false)));
    expect(isDescendantOf(token, 'seedCradle')).toBe(true);
    expect(isDescendantOf(token, 'gardenWisp')).toBe(false);

    module.update(toFrameUpdate(poseFixture('lifted', 3, 5, 7, 1, 3, 5, true, false)));
    expect(isDescendantOf(token, 'gardenWisp')).toBe(true);
    expect(isDescendantOf(token, 'seedCradle')).toBe(false);
    // a carried seed rides the wisp: same cell, and off the floor
    const carried = objectWorld(module, 'amberSeed');
    const orb = worldOf(module, 'wisp');
    expect(carried.distanceTo(orb)).toBeLessThan(0.5);

    // ...and the respawn puts it back in a dish, on the new key cell
    module.update(toFrameUpdate(poseFixture('respawn', 2, 3, 7, 4, 2, 4, false, false)));
    expect(isDescendantOf(token, 'seedCradle')).toBe(true);
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(2, 4, grid, truth);
    const home = objectWorld(module, 'amberSeed');
    expect(home.x).toBeCloseTo(truth[0], 10);
    expect(home.z).toBeCloseTo(truth[2], 10);
    module.dispose();
  });

  test('the dish goes cold when the seed leaves it and warm when it returns', async () => {
    const { module } = await live();
    const pool = module.scene.getObjectByName('seedBedPool');
    expect(pool).toBeDefined();
    if (!pool) return;
    module.update(toFrameUpdate(poseFixture('home', 1, 1, 7, 1, 3, 5, false, false)));
    expect(pool.visible).toBe(true);
    module.update(toFrameUpdate(poseFixture('lifted', 3, 5, 7, 1, 3, 5, true, false)));
    expect(pool.visible).toBe(false);
    module.update(toFrameUpdate(poseFixture('home2', 1, 1, 7, 1, 3, 5, false, false)));
    expect(pool.visible).toBe(true);
    module.dispose();
  });
});

describe('key-corridor / the gate is one bit, sequenced', () => {
  test('the phase is a pure function of (doorOpen, age) and lands on its target', () => {
    // a settled clock returns the target exactly, in both directions
    expect(gatePhase(false, -Infinity)).toBe(0);
    expect(gatePhase(true, -Infinity)).toBe(1);
    expect(gatePhase(false, Number.NaN)).toBe(0);
    expect(gatePhase(true, 99)).toBe(1);
    // a fresh clock starts from the OTHER end and runs monotonically to the target
    expect(gatePhase(true, 0)).toBeCloseTo(0, 12);
    expect(gatePhase(false, 0)).toBeCloseTo(1, 12);
    let last = 0;
    for (let i = 0; i <= 40; i++) {
      const v = gatePhase(true, (i / 40) * GATE_OPEN_SECONDS);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
      expect(v).toBeGreaterThanOrEqual(last - 1e-12);
      last = v;
    }
    expect(last).toBeCloseTo(1, 6);
    let lastDown = 1;
    for (let i = 0; i <= 40; i++) {
      const v = gatePhase(false, (i / 40) * GATE_RELOCK_SECONDS);
      expect(v).toBeLessThanOrEqual(lastDown + 1e-12);
      lastDown = v;
    }
    expect(lastDown).toBeCloseTo(0, 6);
  });

  test('THE TWO CLOCKS ARE NOT THE SAME LENGTH: the opening is one hop, the relock is not', () => {
    // the agent is walking THROUGH the opening in the same control step, so it finishes
    // inside a hop; the relock rides the teleport and may take its time
    expect(GATE_OPEN_SECONDS).toBeLessThanOrEqual(0.125); // one 8 Hz control step
    expect(GATE_RELOCK_SECONDS).toBeGreaterThan(4 * GATE_OPEN_SECONDS);
    // ...and the asymmetry is really in the function, not just in the constants
    expect(gatePhase(true, GATE_OPEN_SECONDS)).toBe(1);
    expect(gatePhase(false, GATE_OPEN_SECONDS)).toBeGreaterThan(0.5);
    expect(gatePhase(false, GATE_RELOCK_SECONDS)).toBe(0);
  });

  test('the ivy retracts to the frame span and never further', async () => {
    const { module, ctx } = await live();
    const vine = module.scene.getObjectByName('ivyVine');
    expect(vine).toBeDefined();
    if (!(vine instanceof THREE.InstancedMesh)) throw new Error('the vine is not instanced');

    const liveCount = (): number => {
      const m = new THREE.Matrix4();
      const s = new THREE.Vector3();
      let n = 0;
      for (let i = 0; i < vine.count; i++) {
        vine.getMatrixAt(i, m);
        s.setFromMatrixScale(m);
        if (s.x > 1e-6) n++;
      }
      return n;
    };

    ctx.setTime(20);
    module.update(toFrameUpdate(poseFixture('locked', 1, 1, 7, 1, 3, 5, false, false)));
    const wrapped = liveCount();
    module.update(toFrameUpdate(poseFixture('open', 4, 3, 7, 1, 3, 5, true, true)));
    const released = liveCount();

    // a locked gate is fully sewn; an open one keeps its architecture green
    expect(wrapped).toBe(vine.count);
    expect(released).toBeLessThan(wrapped);
    expect(released).toBeGreaterThan(0);
    // the survivors are exactly the frame strand, within one feathered sprig
    expect(Math.abs(released / vine.count - IVY_FRAME_SPAN)).toBeLessThan(0.12);
    module.dispose();
  });

  test("the vine's door strand really lies across the leaf, and its frame strand does not", () => {
    const p = new THREE.Vector3();
    // the frame strand climbs and crosses the arch: it is deep in z and high in y
    vinePoint(0.15, p);
    expect(p.y).toBeGreaterThan(0.3);
    vinePoint(0.4, p);
    expect(p.y).toBeGreaterThan(0.9);
    // the door strand hugs the leaf's OUTER face plane, just proud of it
    for (const u of [0.7, 0.85, 1]) {
      vinePoint(u, p);
      expect(p.z).toBeLessThan(DOOR_FACE_Z);
      expect(p.z).toBeGreaterThan(DOOR_FACE_Z - 0.12);
      expect(Math.abs(p.x)).toBeLessThan(0.5);
    }
  });

  test('the leaf swings only after the ivy has let go', async () => {
    const { module, ctx } = await live();
    const hinge = module.scene.getObjectByName('ivyGateHinge');
    const vine = module.scene.getObjectByName('ivyVine');
    expect(hinge).toBeDefined();
    if (!hinge || !(vine instanceof THREE.InstancedMesh)) throw new Error('no gate');
    const sprigs = liveSprigCount(vine);

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('locked', 1, 1, 7, 1, 3, 5, false, false), { scrubbed: true }));
    expect(hinge.rotation.y).toBeCloseTo(0, 10);
    const sewn = sprigs();
    expect(sewn).toBe(vine.count);

    // the door event starts the clock; the leaf has not moved on the event's own frame
    module.update(
      toFrameUpdate(poseFixture('opening', 4, 3, 7, 1, 3, 5, true, true), {
        scrubbed: false,
        events: [{ type: 'door', index: 0, open: true }],
      }),
    );
    expect(hinge.rotation.y).toBeCloseTo(0, 10);

    // CAUSE BEFORE EFFECT, measured on the clock rather than asserted in a comment: walk
    // the window and record when the ivy's door strand has fully let go and when the leaf
    // reaches full swing. The release must complete first.
    ctx.setTime(12);
    module.update(toFrameUpdate(poseFixture('open', 5, 3, 7, 1, 3, 5, true, true), { scrubbed: false }));
    const settledAngle = Math.abs(hinge.rotation.y);
    const released = sprigs();
    expect(settledAngle).toBeGreaterThan(1.3);
    expect(released).toBeLessThan(sewn);

    let ivyFreeAt = Infinity;
    let leafOpenAt = Infinity;
    let sixtyPercentAt = Infinity;
    for (let i = 0; i <= 240; i++) {
      const age = (i / 240) * GATE_OPEN_SECONDS;
      ctx.setTime(10 + age);
      module.update(toFrameUpdate(poseFixture('opening', 4, 3, 7, 1, 3, 5, true, true), { scrubbed: false }));
      const swung = Math.abs(hinge.rotation.y) / settledAngle;
      if (sprigs() <= released && age < ivyFreeAt) ivyFreeAt = age;
      if (swung >= 0.6 && age < sixtyPercentAt) sixtyPercentAt = age;
      if (swung >= 0.999 && age < leafOpenAt) leafOpenAt = age;
    }
    expect(ivyFreeAt).toBeLessThan(leafOpenAt);
    // and the ivy is off the leaf before the leaf is even half-way round
    expect(ivyFreeAt).toBeLessThanOrEqual(sixtyPercentAt);
    // both beats land inside the opening window (a settled gate, not a half-swung one)
    expect(leafOpenAt).toBeLessThanOrEqual(GATE_OPEN_SECONDS);
    module.dispose();
  });

  test('THE BLOCKER: driven at the harness cadence, the gate is open BEFORE the wisp reaches it', async () => {
    // The env opens the door and moves the agent into the door cell in the SAME control
    // step, so this is the only instrument that can see the failure the fixture loop
    // cannot: fixtures default to `scrubbed: true`, which snaps the gate to its settled
    // pose and hides every timing lie. Here the clock runs at 60 fps with a control step
    // every 0.125 s, exactly as the harness plays the recording (demo lane seed 24, rows
    // 116-122: (3,3) -> door opens + agent enters (4,3) -> (5,3)).
    const { module, ctx } = await live();
    const hinge = module.scene.getObjectByName('ivyGateHinge');
    const vine = module.scene.getObjectByName('ivyVine');
    if (!hinge || !(vine instanceof THREE.InstancedMesh)) throw new Error('no gate');
    const sprigs = liveSprigCount(vine);

    // the leaf's own plane, read off the scene (the hinge IS the shut leaf's plane)
    const leafPlaneX = objectWorld(module, 'ivyGateHinge').x;
    // what a settled open gate looks like — the bar every driven frame is measured against
    ctx.setTime(0);
    module.update(toFrameUpdate(poseFixture('settled', 5, 3, 7, 1, 3, 5, true, true), { scrubbed: true }));
    const settledAngle = Math.abs(hinge.rotation.y);
    const frameStrand = sprigs();
    // ...and now put the garden back the way the recording has it one step earlier: the
    // wisp standing on (3,3), the gate shut and sewn, every clock settled
    ctx.setTime(1);
    module.update(toFrameUpdate(poseFixture('before', 3, 3, 7, 1, 3, 5, true, false), { scrubbed: true }));
    expect(worldOf(module, 'wisp-anchor').x).toBeLessThan(leafPlaneX);

    const dt = 1 / 60;
    let now = 1;
    const steps: [number, number, boolean, boolean][] = [
      [3, 3, false, false],
      [4, 3, true, true],
      [5, 3, true, false],
      [5, 3, true, false],
    ];
    let crossedAngle = -1;
    let crossedSprigs = -1;
    let inDoorAngle = -1;
    for (let s = 0; s < steps.length; s++) {
      const [x, y, open, fires] = steps[s] as [number, number, boolean, boolean];
      for (let f = 0; f < 8; f++) {
        ctx.setTime(now);
        module.update(
          toFrameUpdate(poseFixture('drive', x, y, 7, 1, 3, 5, true, open, 116 + s), {
            scrubbed: false,
            dtReal: dt,
            events: f === 0 && fires ? [{ type: 'door', index: 0, open: true }] : [],
          }),
        );
        const wisp = worldOf(module, 'wisp-anchor');
        if (crossedAngle < 0 && wisp.x >= leafPlaneX) {
          crossedAngle = Math.abs(hinge.rotation.y);
          crossedSprigs = sprigs();
        }
        if (inDoorAngle < 0 && s === 1 && f === 7) inDoorAngle = Math.abs(hinge.rotation.y);
        now += dt;
      }
    }

    // THE ASSERTION THE REVIEW BOUGHT: the frame the orb crosses the leaf's plane, the
    // leaf is out of the way and the ivy that was holding it is already gone.
    expect(crossedAngle / settledAngle).toBeGreaterThan(0.9);
    expect(crossedSprigs).toBe(frameStrand);
    expect(crossedSprigs).toBeLessThan(vine.count);
    // ...and by the time it is standing in the doorway the gate is fully open
    expect(inDoorAngle).toBeCloseTo(settledAngle, 6);
    module.dispose();
  });

  test('the relock runs the same scalar backwards, slowly, leaf before ivy', async () => {
    const { module, ctx } = await live();
    const hinge = module.scene.getObjectByName('ivyGateHinge');
    const vine = module.scene.getObjectByName('ivyVine');
    if (!hinge || !(vine instanceof THREE.InstancedMesh)) throw new Error('no gate');
    const sprigs = liveSprigCount(vine);

    // how much vine is actually DRAWN — the sprig count only says a sprig exists, and the
    // last one spends the whole tail of the relock growing back to full size
    const extent = vineExtent(vine);
    ctx.setTime(20);
    module.update(toFrameUpdate(poseFixture('open', 5, 3, 7, 1, 3, 5, true, true), { scrubbed: true }));
    const settledAngle = Math.abs(hinge.rotation.y);
    const openExtent = extent();
    module.update(toFrameUpdate(poseFixture('locked', 2, 3, 7, 4, 2, 4, false, false), { scrubbed: true }));
    const sewnExtent = extent();
    expect(sewnExtent).toBeGreaterThan(openExtent);
    expect(sprigs()).toBe(vine.count);

    // the goal frame: relock + teleport, the wisp already back in the wild garden
    ctx.setTime(20);
    module.update(toFrameUpdate(poseFixture('open', 5, 3, 7, 1, 3, 5, true, true), { scrubbed: true }));
    module.update(
      toFrameUpdate(poseFixture('scored', 2, 3, 7, 4, 2, 4, false, false, 264), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'door', index: 0, open: false }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    let leafShutAt = Infinity;
    let ivySewnAt = Infinity;
    let extentAtShut = -1;
    for (let i = 0; i <= 360; i++) {
      const age = (i / 360) * GATE_RELOCK_SECONDS;
      ctx.setTime(20 + age);
      module.update(toFrameUpdate(poseFixture('relock', 2, 3, 7, 4, 2, 4, false, false, 264), { scrubbed: false }));
      if (Math.abs(hinge.rotation.y) / settledAngle <= 1e-6 && age < leafShutAt) {
        leafShutAt = age;
        extentAtShut = extent();
      }
      if (extent() >= sewnExtent - 1e-6 && age < ivySewnAt) ivySewnAt = age;
    }
    // the reverse order: the leaf shuts, and only then does the ivy finish sewing over it
    expect(leafShutAt).toBeLessThan(ivySewnAt);
    expect(extentAtShut).toBeLessThan(sewnExtent);
    // and it really is the long window — this beat has no agent to overtake
    expect(ivySewnAt).toBeGreaterThan(4 * GATE_OPEN_SECONDS);
    expect(ivySewnAt).toBeLessThanOrEqual(GATE_RELOCK_SECONDS + 1e-9);
    module.dispose();
  });

  test('a scrub lands on a settled gate, never a half-swung one', async () => {
    const { module, ctx } = await live();
    const hinge = module.scene.getObjectByName('ivyGateHinge');
    if (!hinge) throw new Error('no hinge');
    ctx.setTime(3);
    module.update(
      toFrameUpdate(poseFixture('opening', 4, 3, 7, 1, 3, 5, true, true), {
        scrubbed: false,
        events: [{ type: 'door', index: 0, open: true }],
      }),
    );
    expect(hinge.rotation.y).toBeCloseTo(0, 10); // mid-transition, still shut
    ctx.setTime(3.02);
    module.update(toFrameUpdate(poseFixture('scrub', 6, 2, 7, 1, 3, 5, true, true), { scrubbed: true }));
    expect(Math.abs(hinge.rotation.y)).toBeGreaterThan(1.3); // snapped fully open
    module.dispose();
  });
});

describe('key-corridor / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph, carry included', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('garden-start')));
    let count = 0;
    module.scene.traverse(() => count++);
    for (let i = 0; i < 80; i++) {
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

    // standing exactly ON the shrine cell with no event: nothing may fire
    ctx.setTime(1);
    module.update(toFrameUpdate(poseFixture('onGoal', 7, 1, 7, 1, 3, 5, true, true, 100), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    // carrying the seed with no key event: still nothing
    module.update(toFrameUpdate(poseFixture('carrying', 6, 1, 7, 1, 3, 5, true, true, 101), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);

    // the decoder's real batch for a goal frame
    ctx.setTime(2);
    module.update(
      toFrameUpdate(poseFixture('scored', 2, 3, 7, 4, 2, 4, false, false, 102), {
        scrubbed: false,
        events: [
          { type: 'goal' },
          { type: 'door', index: 0, open: false },
          { type: 'teleport' },
          { type: 'done' },
        ],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    // a `done` that came WITH the reward must never sigh
    expect(pool.pulse.sigh).toBe(0);
    module.dispose();
  });

  test('mid-tier events burst but never ripple (STYLE.md celebration grammar)', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(4);
    module.update(
      toFrameUpdate(poseFixture('key', 3, 5, 7, 1, 3, 5, true, false, 10), {
        scrubbed: false,
        events: [{ type: 'key', index: 0 }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(0);
    ctx.setTime(5);
    module.update(
      toFrameUpdate(poseFixture('door', 4, 3, 7, 1, 3, 5, true, true, 11), {
        scrubbed: false,
        events: [{ type: 'door', index: 0, open: true }],
      }),
    );
    expect(pool.stats.bursts).toBe(2);
    expect(pool.stats.ripples).toBe(0);
    // a relock is part of the respawn, not a beat of its own
    ctx.setTime(6);
    module.update(
      toFrameUpdate(poseFixture('relock', 2, 3, 7, 4, 2, 4, false, false, 12), {
        scrubbed: false,
        events: [{ type: 'door', index: 0, open: false }],
      }),
    );
    expect(pool.stats.bursts).toBe(2);
    module.dispose();
  });

  test('the goal blooms on the SHRINE the wisp reached, not on the respawned one', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());

    // the frame before: the wisp is one cell short of the shrine on (7,1)
    const before = poseFixture('run', 7, 2, 7, 1, 3, 5, true, true, 200);
    ctx.setTime(3);
    module.update(toFrameUpdate(before, { scrubbed: false }));

    // ...and now the goal frame: everything has already respawned
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(poseFixture('scored', 2, 3, 7, 4, 2, 4, false, false, 201), {
        scrubbed: false,
        prev: toFrameState(before),
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );

    // the celebration point light rides the bloom position: the OLD shrine cell
    const light = module.scene.getObjectByName('oakPoint3');
    expect(light).toBeDefined();
    if (!light) return;
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(7, 1, grid, truth);
    expect(light.position.x).toBeCloseTo(truth[0], 10);
    expect(light.position.z).toBeCloseTo(truth[2], 10);
    expect((light as THREE.PointLight).intensity).toBeGreaterThan(0);

    // and the new shrine is already on its new cell, growing in from nothing
    const shrine = worldOf(module, 'shrine');
    cellToWorld(7, 4, grid, truth);
    expect(shrine.x).toBeCloseTo(truth[0], 10);
    expect(shrine.z).toBeCloseTo(truth[2], 10);
    const body = module.scene.getObjectByName('shrineBody');
    expect(body?.scale.x ?? 1).toBeLessThan(0.25);
    module.dispose();
  });

  test('a teleport hard-cuts the wisp instead of gliding it through the wall', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];

    ctx.setTime(8);
    module.update(toFrameUpdate(poseFixture('at-goal', 7, 2, 7, 1, 3, 5, true, true, 300), { scrubbed: false }));

    // the respawn frame — same playback instant, so a HOP would still be at its start
    module.update(
      toFrameUpdate(poseFixture('respawned', 1, 5, 7, 4, 2, 4, false, false, 301), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(1, 5, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.y).toBeCloseTo(0, 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    module.dispose();
  });

  test('a plain step still glides, and lands inside one cell within a control step', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('a', 2, 2, 7, 1, 3, 5, false, false, 0)));
    cellToWorld(2, 2, grid, from);
    cellToWorld(3, 2, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(poseFixture('b', 3, 2, 7, 1, 3, 5, false, false, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(poseFixture('b', 3, 2, 7, 1, 3, 5, false, false, 1), { scrubbed: false }));
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

  test('a scrub clears every in-flight beat and settles the new shrine', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(poseFixture('scored', 2, 3, 7, 4, 2, 4, false, false, 400), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('garden-start'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    const body = module.scene.getObjectByName('shrineBody');
    expect(body?.scale.x ?? 0).toBeCloseTo(1, 6);
    module.dispose();
  });

  test('the growth and glow envelopes are pure, bounded and land on 1', () => {
    expect(growth(-Infinity)).toBe(1);
    expect(growth(Number.NaN)).toBe(1);
    expect(growth(0)).toBe(0);
    expect(growth(99)).toBe(1);
    for (let i = 0; i <= 50; i++) {
      const g = growth((i / 50) * 0.75);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(1.05);
    }
    const far = shrineGlow(0, { x: 1, y: 1 }, { x: 7, y: 5 });
    const near = shrineGlow(0, { x: 6, y: 5 }, { x: 7, y: 5 });
    expect(far).toBeGreaterThanOrEqual(0);
    expect(near).toBeLessThanOrEqual(1);
    expect(near).toBeGreaterThan(far);
  });

  test('readState prefers the decoder state and falls back to the raw pose', () => {
    const grid = resolveGrid(geometryOf());
    const out = makeGardenState();
    // no `derived`: the packing is read straight off the 9 floats
    const bare = toFrameState({ name: 'bare', pose: [3, 4, 7, 2, 1, 5, 1, 1, 0] });
    readState(bare, grid, out);
    expect(out.agent).toEqual({ x: 3, y: 4 });
    expect(out.goal).toEqual({ x: 7, y: 2 });
    expect(out.key).toEqual({ x: 1, y: 5 });
    expect(out.hasKey).toBe(true);
    expect(out.doorOpen).toBe(true);
    // with `derived`, sim-core wins
    const decoded = toFrameState({
      name: 'decoded',
      pose: [0, 0, 0, 0, 0, 0, 0, 0, 0],
      derived: { x: 2, y: 1, gx: 5, gy: 3, kx: 3, ky: 2, hasKey: false, doorOpen: false },
    });
    readState(decoded, grid, out);
    expect(out.agent).toEqual({ x: 2, y: 1 });
    expect(out.goal).toEqual({ x: 5, y: 3 });
    expect(out.key).toEqual({ x: 3, y: 2 });
    expect(out.hasKey).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// THE COMPOSED FRAME — the two instruments a still cannot argue with.
//
// A scene can be perfectly aligned and still be unwatchable: the avatar can sit behind
// the very prop the scene is about, and the playfield can run off the bottom edge. Both
// were filed against this scene and both are now measured here rather than eyeballed.
// ---------------------------------------------------------------------------

/** every opaque mesh that is NOT the avatar (or the debug overlay) */
function occluderMeshes(module: SceneModule): THREE.Object3D[] {
  const out: THREE.Object3D[] = [];
  module.scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh && !(o as THREE.InstancedMesh).isInstancedMesh) return;
    let node: THREE.Object3D | null = o;
    while (node) {
      if (node.name === 'gardenWisp' || node.name === 'gardenGroundTruth') return;
      node = node.parent;
    }
    const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
    const mats = Array.isArray(mat) ? mat : mat ? [mat] : [];
    // a veil is not an occluder: mist planes and additive pools are see-through by design
    if (mats.length > 0 && mats.every((m) => m.transparent || m.depthWrite === false)) return;
    out.push(o);
  });
  return out;
}

/**
 * Cast from the live camera to the avatar's orb centre on every free cell and report the
 * cells where something opaque gets in the way. Each probe uses a REACHABLE state with the
 * agent on that cell — the seed is carried, the gate is open, and the shrine and the dish
 * stand somewhere else — so the only things that can be in the way are the world's props.
 */
function occludedCells(module: SceneModule): string[] {
  const grid = resolveGrid(geometryOf());
  const camera = module.camera;
  const raycaster = new THREE.Raycaster();
  const dir = new THREE.Vector3();
  const orb = new THREE.Vector3();
  const truth: [number, number, number] = [0, 0, 0];
  const bad: string[] = [];
  for (let y = 0; y < grid.h; y++) {
    for (let x = 0; x < grid.w; x++) {
      if (isWall(grid, x, y)) continue;
      const goal = x === 7 && y === 5 ? { x: 5, y: 1 } : { x: 7, y: 5 };
      const key = x === 1 && y === 5 ? { x: 3, y: 1 } : { x: 1, y: 5 };
      module.update(
        toFrameUpdate(poseFixture('census', x, y, goal.x, goal.y, key.x, key.y, true, true), { scrubbed: true }),
      );
      module.scene.updateMatrixWorld(true);
      camera.updateMatrixWorld(true);
      cellToWorld(x, y, grid, truth);
      orb.set(truth[0] ?? 0, TOKENS.wispHeight, truth[2] ?? 0);
      dir.copy(orb).sub(camera.position);
      const distance = dir.length();
      dir.normalize();
      raycaster.set(camera.position, dir);
      raycaster.near = 0;
      raycaster.far = distance - 1e-3;
      const names = new Set<string>();
      for (const hit of raycaster.intersectObjects(occluderMeshes(module), false)) {
        if (hit.distance < distance - 1e-3) names.add(hit.object.name || '(unnamed)');
      }
      if (names.size > 0) bad.push(`(${x},${y}) <- ${[...names].join(', ')}`);
    }
  }
  return bad;
}

describe('key-corridor / the composed frame', () => {
  test('THE MAJOR: nothing stands between the hero camera and the avatar, on ANY free cell', async () => {
    const module = createScene();
    const ctx = makeSceneContext({
      geometry: geometryOf(),
      seed: SEED,
      tier: 'hero',
      size: { width: 1600, height: 900, dpr: 1 },
    });
    await module.init(ctx);
    module.resize({ width: 1600, height: 900, dpr: 1 });
    // the door cell is the one this scene exists to dramatise, and it was the one the
    // gate's own crown was hiding — for the four control steps the agent spends there
    expect(occludedCells(module)).toEqual([]);
    module.dispose();
  });

  test('...and the same holds for the fixed thumb camera', async () => {
    const module = createScene();
    const ctx = makeSceneContext({
      geometry: geometryOf(),
      seed: SEED,
      tier: 'thumb',
      size: { width: 400, height: 400, dpr: 1 },
    });
    await module.init(ctx);
    module.resize({ width: 400, height: 400, dpr: 1 });
    expect(occludedCells(module)).toEqual([]);
    module.dispose();
  });

  test('the WHOLE garden — kerb corners included — is inside the hero frame', async () => {
    const grid = resolveGrid(geometryOf());
    for (const [width, height] of [
      [1600, 900],
      [1600, 813],
      [1280, 720],
    ] as const) {
      const module = createScene();
      const ctx = makeSceneContext({ geometry: geometryOf(), seed: SEED, tier: 'hero', size: { width, height, dpr: 1 } });
      await module.init(ctx);
      module.update(toFrameUpdate(fixtureNamed('garden-start')));
      module.resize({ width, height, dpr: 1 });
      const camera = module.camera;
      camera.updateMatrixWorld(true);
      const p = new THREE.Vector3();
      for (const cx of [-GARDEN_HALF_X, GARDEN_HALF_X]) {
        for (const cz of [-GARDEN_HALF_Z, GARDEN_HALF_Z]) {
          for (const cy of [0, 0.5]) {
            p.set(cx, cy, cz).project(camera);
            // vFOV is fixed at 40° whatever the aspect (canon lesson 11), so the vertical
            // fit is aspect-independent and this must hold at every framing
            expect(Math.abs(p.y)).toBeLessThan(1);
          }
        }
      }
      // and the playfield keeps real margin, not a hairline
      let worst = 0;
      for (let y = 0; y < grid.h; y++) {
        for (let x = 0; x < grid.w; x++) {
          if (isWall(grid, x, y)) continue;
          const truth: [number, number, number] = [0, 0, 0];
          cellToWorld(x, y, grid, truth);
          for (const ox of [-0.5, 0.5]) {
            for (const oz of [-0.5, 0.5]) {
              p.set((truth[0] ?? 0) + ox, 0, (truth[2] ?? 0) + oz).project(camera);
              worst = Math.max(worst, Math.abs(p.x), Math.abs(p.y));
            }
          }
        }
      }
      expect(worst).toBeLessThan(0.9);
      module.dispose();
    }
  });

  test('the thumb square crop holds every free cell AND the kerb', async () => {
    const grid = resolveGrid(geometryOf());
    const module = createScene();
    const ctx = makeSceneContext({ geometry: geometryOf(), seed: SEED, tier: 'thumb', size: { width: 400, height: 400, dpr: 1 } });
    await module.init(ctx);
    module.update(toFrameUpdate(fixtureNamed('garden-start')));
    module.resize({ width: 400, height: 400, dpr: 1 });
    const camera = module.camera;
    camera.updateMatrixWorld(true);
    const p = new THREE.Vector3();
    let worstCell = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (isWall(grid, x, y)) continue;
        const truth: [number, number, number] = [0, 0, 0];
        cellToWorld(x, y, grid, truth);
        for (const ox of [-0.5, 0.5]) {
          for (const oz of [-0.5, 0.5]) {
            p.set((truth[0] ?? 0) + ox, 0, (truth[2] ?? 0) + oz).project(camera);
            worstCell = Math.max(worstCell, Math.abs(p.x), Math.abs(p.y));
          }
        }
      }
    }
    expect(worstCell).toBeLessThan(0.85);
    for (const cx of [-GARDEN_HALF_X, GARDEN_HALF_X]) {
      for (const cz of [-GARDEN_HALF_Z, GARDEN_HALF_Z]) {
        p.set(cx, 0, cz).project(camera);
        expect(Math.abs(p.x)).toBeLessThan(1);
        expect(Math.abs(p.y)).toBeLessThan(1);
      }
    }
    module.dispose();
  });

  test('the shut leaf separates from the wall it hangs on by albedo, not by hope', () => {
    const skin = gardenSkin(TOKENS.palettes.morning, TOKENS);
    // the leaf and the divider's west face take the same key light, so the only separation
    // the eye gets is the albedo step between them
    expect(luma(skin.wood) - luma(skin.divider)).toBeGreaterThan(0.2);
    expect(luma(skin.woodDark) - luma(skin.divider)).toBeGreaterThan(0.12);
    // ...and it still sits under the raked courtyard, which is the scene's bloom ceiling
    expect(luma(skin.wood)).toBeLessThan(luma(skin.formalFloor));
  });
});

describe('key-corridor / lifecycle', () => {
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
    module.update(toFrameUpdate(fixtureNamed('in-the-gate')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'gardenGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'gardenGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the camera reads the gate face-on-ish and stays a clamped aerial', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('garden-start')));
    const cam = module.camera;
    expect(cam.position.y).toBeGreaterThan(6);
    expect(Math.hypot(cam.position.x, cam.position.z)).toBeGreaterThan(GARDEN_HALF_X);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.y).toBeLessThan(-0.5); // a genuine aerial, never a horizon shot
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    // the whole point of the azimuth: the camera stands WEST of the wall, so the door's
    // west face is turned toward it rather than edge-on
    expect(cam.position.x).toBeLessThan(-3);
    const horiz = Math.hypot(dir.x, dir.z);
    expect(dir.x / horiz).toBeGreaterThan(0.4);
    module.dispose();
  });
});

describe('key-corridor / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    // The wall view is the whole gallery on one page: every scene there is a thumb, and
    // the cap is what keeps 16 of them at 60 fps. Measured with the ground-truth overlay
    // ON, because that is the state review screenshots are taken in, and across every
    // fixture, because the busiest legal state is the one that has to fit.
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'key-corridor (thumb, overlay on)');
    module.dispose();
  });
});
