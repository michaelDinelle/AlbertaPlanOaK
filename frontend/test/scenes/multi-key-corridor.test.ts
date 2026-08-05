/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { CELL, cellToWorld, isWall } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  MKC_DOORS,
  MKC_GOAL,
  MKC_H,
  MKC_KEYS,
  MKC_ROOM_CELLS,
  MKC_START,
  MKC_W,
  mkcIsWall,
  mkcRoomCells,
  mkcWalls,
  multiKeyCorridorDecoder,
} from '@sim/envs/multikeycorridor';
import type { MultiKeyCorridorState } from '@sim/envs/multikeycorridor';
import { mkcStep } from '@sim/envs/multikeycorridor';
import type { EnvGeometry, GridCell } from '@sim/types';

import createScene, { gatePhase, growth, proximity } from '../../src/scenes/multi-key-corridor/index';
import { GATE_SECONDS } from '../../src/scenes/multi-key-corridor/actors';
import { fixtures } from '../../src/scenes/multi-key-corridor/fixtures';
import { manifest } from '../../src/scenes/multi-key-corridor/manifest';
import {
  CHAMBERS,
  TEMPLE_HALF_X,
  TEMPLE_HALF_Z,
  blendTokens,
  chamberBounds,
  chamberOf,
  gateFacesX,
  makeGardenHeight,
  resolveDoors,
  resolveGoal,
  resolveGrid,
  resolveKeys,
  templeSkin,
} from '../../src/scenes/multi-key-corridor/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 80;
const FREE_CELLS = 63;

function geometryOf(): EnvGeometry {
  return multiKeyCorridorDecoder.geometry(multiKeyCorridorDecoder.initStatic(SEED), SEED);
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

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

function stateOf(
  x: number,
  y: number,
  gx: number,
  gy: number,
  keys: readonly GridCell[],
  flags: number,
  respawnIdx = 0,
): MultiKeyCorridorState {
  return {
    x,
    y,
    gx,
    gy,
    keys: keys.map((k) => ({ ...k })),
    hasKey: [(flags & 1) !== 0, (flags & 2) !== 0, (flags & 4) !== 0],
    doorOpen: [(flags & 8) !== 0, (flags & 16) !== 0, (flags & 32) !== 0],
    flags,
    respawnIdx,
    poseAgrees: true,
  };
}

function frameOf(name: string, s: MultiKeyCorridorState, step = 0): SceneFixture {
  return {
    name,
    pose: [s.x, s.y, s.gx, s.gy, s.flags & 7, (s.flags >> 3) & 7, 0, 0, 0],
    ctrl: [0, 0, 0, 0, 0, 0],
    derived: s,
    step,
  };
}


describe('multi-key-corridor / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('MultiKeyCorridor-cont-v0');
    expect(manifest.slug).toBe('multi-key-corridor');
    expect(manifest.accent).toBe('multi-key-corridor');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a legal temple state', () => {
    const p = manifest.thumbFrame.pose;
    expect(mkcIsWall(p[0] ?? -1, p[1] ?? -1)).toBe(false);
    expect(mkcIsWall(p[2] ?? -1, p[3] ?? -1)).toBe(false);
    const onGate = MKC_DOORS.some((d) => d.x === p[0] && d.y === p[1]);
    expect(onGate).toBe(true);
  });

  test('ships at least three canonical poses, all on FREE cells', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      expect(mkcIsWall(f.pose[0] ?? -1, f.pose[1] ?? -1)).toBe(false);
      expect(mkcIsWall(f.pose[2] ?? -1, f.pose[3] ?? -1)).toBe(false);
      const s = f.derived as MultiKeyCorridorState;
      expect(chamberOf(s.gx, s.gy)).toBe(3);
      for (let j = 0; j < 3; j++) {
        const k = s.keys[j] as GridCell;
        expect(mkcIsWall(k.x, k.y)).toBe(false);
        expect(chamberOf(k.x, k.y)).toBe(j);
      }
    }
  });

  test('the fixtures cover the start, a threshold, a full key ring and a respawn', () => {
    const states = fixtures.map((f) => f.derived as MultiKeyCorridorState);
    expect(states.some((s) => s.x === MKC_START.x && s.y === MKC_START.y && s.flags === 0)).toBe(true);
    expect(states.some((s) => MKC_DOORS.some((d) => d.x === s.x && d.y === s.y))).toBe(true);
    expect(states.some((s) => s.hasKey.every(Boolean) && s.doorOpen.every(Boolean))).toBe(true);
    const respawn = states.find((s) => s.respawnIdx > 0);
    expect(respawn).toBeDefined();
    expect(respawn?.flags).toBe(0);
    expect(respawn?.keys).not.toEqual(MKC_KEYS.map((k) => ({ ...k })));
  });
});

describe('multi-key-corridor / the layout is read, never re-derived', () => {
  test('resolveGrid returns the decoder geometry untouched', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry) === geometry.grid).toBe(true);
  });

  test("a placeholder geometry falls back to the DECODER's own walls", () => {
    const grid = resolveGrid({ kind: 'grid' });
    expect(grid.w).toBe(MKC_W);
    expect(grid.h).toBe(MKC_H);
    let free = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        expect(isWall(grid, x, y)).toBe(mkcIsWall(x, y));
        if (!isWall(grid, x, y)) free++;
      }
    }
    expect(free).toBe(FREE_CELLS);
  });

  test('the three gate cells come from the env, are free, and are flanked by wall', () => {
    const grid = resolveGrid(geometryOf());
    const doors = resolveDoors(grid);
    expect(doors.length).toBe(3);
    for (let j = 0; j < doors.length; j++) {
      const d = doors[j] as GridCell;
      const truth = MKC_DOORS[j] as GridCell;
      expect(d.x).toBe(truth.x);
      expect(d.y).toBe(truth.y);
      expect(isWall(grid, d.x, d.y)).toBe(false);
      expect(chamberOf(d.x, d.y)).toBe(-1);
      expect(gateFacesX(grid, d)).toBe(true);
    }
  });

  test("chamberOf is a pure read of the decoder's own room enumeration", () => {
    let painted = 0;
    for (let r = 0; r < CHAMBERS; r++) {
      const cells = mkcRoomCells(r);
      expect(cells.length).toBe(MKC_ROOM_CELLS);
      for (const c of cells) {
        expect(chamberOf(c.x, c.y)).toBe(r);
        expect(mkcIsWall(c.x, c.y)).toBe(false);
        painted++;
      }
    }
    expect(painted + MKC_DOORS.length).toBe(FREE_CELLS);
    expect(chamberOf(-1, 3)).toBe(-1);
    expect(chamberOf(0, 0)).toBe(-1);
  });

  test('chamberBounds finds all four chambers, and only free cells of that chamber', () => {
    const grid = resolveGrid(geometryOf());
    for (let c = 0; c < CHAMBERS; c++) {
      const b = chamberBounds(grid, c);
      expect(b.cells).toBe(MKC_ROOM_CELLS);
      expect(b.halfX).toBeCloseTo(1.5 * CELL, 10);
      expect(b.halfZ).toBeCloseTo(2.5 * CELL, 10);
      expect(Math.abs(b.cx) + b.halfX).toBeLessThanOrEqual(TEMPLE_HALF_X + 1e-9);
      expect(Math.abs(b.cz) + b.halfZ).toBeLessThanOrEqual(TEMPLE_HALF_Z + 1e-9);
    }
    for (let c = 1; c < CHAMBERS; c++) {
      expect(chamberBounds(grid, c).cx).toBeGreaterThan(chamberBounds(grid, c - 1).cx);
    }
  });

  test('the chamber skin graduates cool to warm, four distinct stones', () => {
    const skin = templeSkin(TOKENS.palettes.morning, TOKENS);
    expect(skin.chambers.length).toBe(CHAMBERS);
    const hexes = new Set(skin.chambers.map((c) => c.floor.getHex()));
    expect(hexes.size).toBe(CHAMBERS);
    let previous = -Infinity;
    for (const c of skin.chambers) {
      const warmth = c.floor.r - c.floor.b;
      expect(warmth).toBeGreaterThan(previous);
      previous = warmth;
    }
    for (const c of skin.chambers) {
      expect(Math.max(c.floor.r, c.floor.g, c.floor.b)).toBeLessThan(0.8);
    }
    expect(new Set(skin.keys.map((k) => k.getHex())).size).toBe(3);
    const pearl = skin.keys[1] as THREE.Color;
    expect(pearl.r + pearl.g + pearl.b).toBeGreaterThan(2.4);
  });

  test('the garden is identically flat under the whole temple', () => {
    const h = makeGardenHeight(SEED);
    for (let i = 0; i <= 34; i++) {
      for (let j = 0; j <= 14; j++) {
        const x = -TEMPLE_HALF_X + (2 * TEMPLE_HALF_X * i) / 34;
        const z = -TEMPLE_HALF_Z + (2 * TEMPLE_HALF_Z * j) / 14;
        expect(h(x, z)).toBe(0);
      }
    }
    let wander = 0;
    for (const p of [-19, -16, 16, 19]) wander = Math.max(wander, Math.abs(h(p, p)));
    expect(wander).toBeGreaterThan(0.05);
  });
});

describe('multi-key-corridor / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    for (const name of ['wisp', 'wisp-anchor', 'shrine', 'key-altar-0', 'key-altar-1', 'key-altar-2']) {
      expect(report.worst.get(name) ?? 1).toBeLessThan(1e-9);
    }
    for (let j = 0; j < 3; j++) {
      expect(report.worst.get(`gate-${j}`) ?? 1).toBeLessThan(1e-9);
      expect(report.worst.get(`portal-${j}`) ?? 1).toBeLessThan(1e-9);
    }
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      module.dispose();
    }
  });

  test('the wisp lands on EVERY free cell of the temple, gate cells included', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    let visited = 0;
    for (let y = 0; y < grid.h; y++) {
      for (let x = 0; x < grid.w; x++) {
        if (isWall(grid, x, y)) continue;
        module.update(toFrameUpdate(frameOf(`cell${x}-${y}`, stateOf(x, y, 15, 1, MKC_KEYS, 0), visited)));
        const w = worldOf(module, 'wisp-anchor');
        cellToWorld(x, y, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.y).toBeCloseTo(0, 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
        expect(worldOf(module, 'wisp').y).toBeCloseTo(TOKENS.wispHeight, 10);
        visited++;
      }
    }
    expect(visited).toBe(FREE_CELLS);
    module.dispose();
  });

  test('the shrine follows the relic onto every cell the sanctum can respawn it to', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    const cells = mkcRoomCells(3);
    expect(cells.length).toBe(MKC_ROOM_CELLS);
    for (const g of cells) {
      module.update(toFrameUpdate(frameOf('goalmove', stateOf(2, 2, g.x, g.y, MKC_KEYS, 0, 1))));
      const w = worldOf(module, 'shrine');
      cellToWorld(g.x, g.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
    }
    module.dispose();
  });

  test('the altars follow the REPLAYED key cells through real dynamics, over respawns', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    const replay = {
      x: MKC_START.x,
      y: MKC_START.y,
      keys: MKC_KEYS.map((k) => ({ ...k })),
      gx: MKC_GOAL.x,
      gy: MKC_GOAL.y,
      flags: 0,
      respawnIdx: 0,
    };
    let rng = 0x9e3779b9;
    const nextAction = (): number => {
      rng = (rng * 1664525 + 1013904223) >>> 0;
      const r = rng / 0x100000000;
      return r < 0.45 ? 3 : r < 0.6 ? 2 : r < 0.8 ? 0 : 1;
    };
    const seenKeyLayouts = new Set<string>();
    for (let step = 0; step < 600; step++) {
      mkcStep(SEED, replay, nextAction());
      const s = stateOf(replay.x, replay.y, replay.gx, replay.gy, replay.keys, replay.flags, replay.respawnIdx);
      module.update(toFrameUpdate(frameOf('replay', s, step)));
      seenKeyLayouts.add(replay.keys.map((k) => `${k.x},${k.y}`).join('|'));
      for (let j = 0; j < 3; j++) {
        const key = replay.keys[j] as GridCell;
        const w = worldOf(module, `key-altar-${j}`);
        cellToWorld(key.x, key.y, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.y).toBeCloseTo(0, 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
      }
      const agent = worldOf(module, 'wisp-anchor');
      cellToWorld(replay.x, replay.y, grid, truth);
      expect(agent.x).toBeCloseTo(truth[0], 10);
      expect(agent.z).toBeCloseTo(truth[2], 10);
    }
    expect(replay.respawnIdx).toBeGreaterThanOrEqual(1);
    expect(seenKeyLayouts.size).toBeGreaterThanOrEqual(2);
    module.dispose();
  });

  test('a carried key leaves its pedestal behind at the cell it came from', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    module.update(toFrameUpdate(frameOf('carrying', stateOf(6, 3, 15, 1, MKC_KEYS, 0b010))));
    const w = worldOf(module, 'key-altar-1');
    const cell = MKC_KEYS[1] as GridCell;
    cellToWorld(cell.x, cell.y, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    expect(module.scene.getObjectByName('keyToken1')?.visible).toBe(false);
    expect(module.scene.getObjectByName('keyToken0')?.visible).toBe(true);
    const rack = module.scene.getObjectByName('carriedKeys') as THREE.InstancedMesh;
    expect(rack).toBeDefined();
    const m = new THREE.Matrix4();
    const basis = (i: number): number => {
      rack.getMatrixAt(i, m);
      const e = m.elements;
      return Math.hypot(e[0] ?? 0, e[1] ?? 0, e[2] ?? 0);
    };
    expect(basis(1)).toBeCloseTo(1, 6);
    expect(basis(0)).toBe(0);
    expect(basis(2)).toBe(0);
    module.dispose();
  });
});

describe('multi-key-corridor / the gates are a pure read of the frame', () => {
  function stoneY(module: SceneModule): number {
    const stones = module.scene.getObjectByName('riverSteppingStones') as THREE.InstancedMesh;
    expect(stones).toBeDefined();
    const m = new THREE.Matrix4();
    stones.getMatrixAt(1, m);
    return m.elements[13] ?? 0;
  }

  test('doorOpen drives each gate, and a scrub lands on it with no easing', async () => {
    const { module, ctx } = await live();
    ctx.setTime(5);
    module.update(toFrameUpdate(frameOf('shut', stateOf(1, 1, 15, 1, MKC_KEYS, 0)), { scrubbed: true }));
    const sunk = stoneY(module);
    module.update(
      toFrameUpdate(frameOf('open', stateOf(9, 3, 15, 1, MKC_KEYS, 0b111000)), { scrubbed: true }),
    );
    const risen = stoneY(module);
    expect(risen).toBeGreaterThan(sunk);
    module.update(toFrameUpdate(frameOf('shut', stateOf(1, 1, 15, 1, MKC_KEYS, 0)), { scrubbed: true }));
    expect(stoneY(module)).toBe(sunk);
    module.dispose();
  });

  test('a respawn relocks all three gates in the same frame as the teleport', async () => {
    const { module, ctx } = await live();
    ctx.setTime(2);
    module.update(toFrameUpdate(frameOf('shut', stateOf(1, 1, 15, 1, MKC_KEYS, 0)), { scrubbed: true }));
    const sunk = stoneY(module);
    module.update(toFrameUpdate(frameOf('open', stateOf(14, 1, 15, 1, MKC_KEYS, 0b111111)), { scrubbed: true }));
    const open = stoneY(module);
    expect(open).toBeGreaterThan(sunk);

    ctx.setTime(2.125);
    module.update(
      toFrameUpdate(frameOf('scored', stateOf(2, 4, 13, 5, MKC_KEYS, 0, 1), 1), {
        scrubbed: false,
        prev: null,
        events: [
          { type: 'goal' },
          { type: 'door', index: 0, open: false },
          { type: 'door', index: 1, open: false },
          { type: 'door', index: 2, open: false },
          { type: 'teleport' },
          { type: 'done' },
        ],
      }),
    );
    expect(stoneY(module)).toBe(sunk);
    module.dispose();
  });
});

describe('multi-key-corridor / the gates run on the PLAYBACK clock', () => {
  interface Mechanism {
    ford(): number;
    mist(): number;
    bramble(): number;
    brambleGlow(): number;
    caustics(): number;
  }

  function mechanism(module: SceneModule): Mechanism {
    const stones = module.scene.getObjectByName('riverSteppingStones') as THREE.InstancedMesh;
    const canes = module.scene.getObjectByName('brambleCanes') as THREE.InstancedMesh;
    const curtain = module.scene.getObjectByName('mistCurtain0_0') as THREE.Mesh;
    const water = module.scene.getObjectByName('riverWater') as THREE.Mesh;
    expect(stones).toBeDefined();
    expect(canes).toBeDefined();
    expect(curtain).toBeDefined();
    expect(water).toBeDefined();
    const m = new THREE.Matrix4();
    const caneMaterial = canes.material as THREE.MeshStandardMaterial;
    const waterUniforms = (water.material as THREE.ShaderMaterial).uniforms;
    return {
      ford: () => {
        stones.getMatrixAt(1, m);
        return m.elements[13] ?? 0;
      },
      mist: () => curtain.position.z,
      bramble: () => {
        canes.getMatrixAt(0, m);
        return Math.hypot(m.elements[8] ?? 0, m.elements[9] ?? 0, m.elements[10] ?? 0);
      },
      brambleGlow: () => caneMaterial.emissive.r + caneMaterial.emissive.g + caneMaterial.emissive.b,
      caustics: () => (waterUniforms.uCausticStrength?.value as number) ?? -1,
    };
  }

  function opening(j: number, step: number, flags = (1 << j) | (8 << j)) {
    return toFrameUpdate(
      frameOf('opening', stateOf(4 + 4 * j, 3, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, flags), step),
      { scrubbed: false, events: [{ type: 'door', index: j, open: true }] },
    );
  }

  test('a paused transport freezes all three mechanisms mid-swing', async () => {
    const { module, ctx } = await live();
    const gate = mechanism(module);

    ctx.setTime(5);
    module.update(toFrameUpdate(frameOf('shut', stateOf(1, 1, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, 0)), { scrubbed: true }));
    const shut = { ford: gate.ford(), mist: gate.mist(), bramble: gate.bramble() };

    const openFlags = 0b111111;
    for (let j = 0; j < 3; j++) module.update(opening(j, 100 + j, openFlags));
    module.update(
      toFrameUpdate(frameOf('paused', stateOf(9, 3, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, openFlags), 103), {
        scrubbed: false,
      }),
    );
    const atPause = { ford: gate.ford(), mist: gate.mist(), bramble: gate.bramble(), glow: gate.brambleGlow() };
    expect(atPause.ford).toBe(shut.ford);
    expect(atPause.mist).toBe(shut.mist);
    expect(atPause.bramble).toBe(shut.bramble);

    for (let i = 0; i < 40; i++) {
      module.update(
        toFrameUpdate(frameOf('paused', stateOf(9, 3, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, openFlags), 103), {
          scrubbed: false,
          dtReal: 1 / 60,
        }),
      );
    }
    expect(gate.ford()).toBe(atPause.ford);
    expect(gate.mist()).toBe(atPause.mist);
    expect(gate.bramble()).toBe(atPause.bramble);
    expect(gate.brambleGlow()).toBe(atPause.glow);

    ctx.setTime(5 + GATE_SECONDS);
    module.update(
      toFrameUpdate(frameOf('open', stateOf(9, 3, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, openFlags), 103), {
        scrubbed: false,
        dtReal: 0,
      }),
    );
    expect(gate.ford()).toBeGreaterThan(shut.ford);
    expect(Math.abs(gate.mist() - shut.mist)).toBeGreaterThan(0.2);
    expect(gate.bramble()).toBeLessThan(shut.bramble - 0.5);
    module.dispose();
  });

  test('openness follows the playback age, and ignores how fast frames arrive', async () => {
    async function swing(dtReal: number, frames: number): Promise<number[]> {
      const { module, ctx } = await live();
      const gate = mechanism(module);
      ctx.setTime(2);
      module.update(toFrameUpdate(frameOf('shut', stateOf(1, 1, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, 0)), { scrubbed: true }));
      module.update(opening(1, 10));
      const out: number[] = [];
      for (let i = 1; i <= frames; i++) {
        ctx.setTime(2 + (i / frames) * GATE_SECONDS);
        module.update(
          toFrameUpdate(frameOf('swing', stateOf(8, 3, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, 0b010010), 11), {
            scrubbed: false,
            dtReal,
          }),
        );
        out.push(gate.ford());
      }
      module.dispose();
      return out;
    }
    const slow = await swing(1 / 240, 12);
    const fast = await swing(4 / 60, 12);
    expect(fast).toEqual(slow);
    expect(slow[0] ?? 1).toBeLessThan(slow[slow.length - 1] ?? 0);
    for (let i = 1; i < slow.length; i++) expect(slow[i] ?? 0).toBeGreaterThanOrEqual(slow[i - 1] ?? 0);
  });

  test('gatePhase is pure, bounded, and settles on the frame\'s own bit', () => {
    expect(gatePhase(true, -Infinity)).toBe(1);
    expect(gatePhase(false, -Infinity)).toBe(0);
    expect(gatePhase(true, Number.NaN)).toBe(1);
    expect(gatePhase(false, Number.NaN)).toBe(0);
    expect(gatePhase(true, 99)).toBe(1);
    expect(gatePhase(false, 99)).toBe(0);
    expect(gatePhase(true, 0)).toBe(0);
    expect(gatePhase(false, 0)).toBe(1);
    expect(gatePhase(true, GATE_SECONDS / 2)).toBeCloseTo(0.5, 12);
    expect(gatePhase(false, GATE_SECONDS / 2)).toBeCloseTo(0.5, 12);
    for (let i = 0; i <= 40; i++) {
      const age = (i / 40) * GATE_SECONDS * 1.5;
      const up = gatePhase(true, age);
      expect(up).toBeGreaterThanOrEqual(0);
      expect(up).toBeLessThanOrEqual(1);
      expect(gatePhase(false, age)).toBeCloseTo(1 - up, 12);
    }
  });

  test('the eased mechanism is idempotent mid-swing, not just the probe roots', async () => {
    const { module, ctx } = await live();
    const gate = mechanism(module);
    ctx.setTime(7);
    module.update(toFrameUpdate(frameOf('shut', stateOf(1, 1, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, 0)), { scrubbed: true }));
    for (let j = 0; j < 3; j++) module.update(opening(j, 200 + j));
    ctx.setTime(7 + GATE_SECONDS * 0.5);
    const frame = frameOf('mid', stateOf(9, 3, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, 0b111111), 210);
    module.update(toFrameUpdate(frame, { scrubbed: false }));
    const a = [gate.ford(), gate.mist(), gate.bramble(), gate.brambleGlow(), gate.caustics()];
    for (let i = 0; i < 5; i++) module.update(toFrameUpdate(frame, { scrubbed: false }));
    expect([gate.ford(), gate.mist(), gate.bramble(), gate.brambleGlow(), gate.caustics()]).toEqual(a);
    expect(a[0]).toBeGreaterThan(-0.03);
    expect(a[0]).toBeLessThan(0.1);
    module.dispose();
  });

  test('every flourish the gates carry settles back to its built-in idle', async () => {
    const { module, ctx } = await live();
    const gate = mechanism(module);
    const IDLE_CAUSTICS = 0.26;
    expect(gate.caustics()).toBeCloseTo(IDLE_CAUSTICS, 12);
    expect(gate.brambleGlow()).toBe(0);

    ctx.setTime(3);
    module.update(opening(1, 300, 0b011011));
    expect(gate.caustics()).toBeGreaterThan(IDLE_CAUSTICS + 1);
    module.update(opening(0, 301, 0b011011));
    expect(gate.brambleGlow()).toBeGreaterThan(0);

    ctx.setTime(3 + 2);
    module.update(
      toFrameUpdate(frameOf('after', stateOf(9, 3, MKC_GOAL.x, MKC_GOAL.y, MKC_KEYS, 0b011011), 320), {
        scrubbed: false,
      }),
    );
    expect(gate.caustics()).toBeCloseTo(IDLE_CAUSTICS, 12);
    expect(gate.brambleGlow()).toBe(0);

    ctx.setTime(3.05);
    module.update(opening(1, 300, 0b011011));
    expect(gate.caustics()).toBeGreaterThan(IDLE_CAUSTICS);
    ctx.setTime(0.5);
    module.update(toFrameUpdate(fixtureNamed('temple-start'), { scrubbed: true }));
    expect(gate.caustics()).toBeCloseTo(IDLE_CAUSTICS, 12);
    expect(gate.brambleGlow()).toBe(0);
    module.dispose();
  });

  test('the ford is water-coloured — the kit palette default is not honest here', async () => {
    const { module } = await live();
    const water = module.scene.getObjectByName('riverWater') as THREE.Mesh;
    const uniforms = (water.material as THREE.ShaderMaterial).uniforms;
    const shallow = uniforms.uShallow?.value as THREE.Color;
    const deep = uniforms.uDeep?.value as THREE.Color;
    const underwater = TOKENS.palettes.underwater;

    const wantShallow = blendTokens(underwater.skyTop, underwater.groundB, 0.32);
    const wantDeep = new THREE.Color(underwater.skyBottom);
    expect(shallow.getHex()).toBe(wantShallow.getHex());
    expect(deep.getHex()).toBe(wantDeep.getHex());

    for (const c of [shallow, deep]) {
      expect(c.b).toBeGreaterThan(c.r * 3);
      expect(c.b).toBeGreaterThan(c.g);
    }
    const morning = TOKENS.palettes.morning;
    const kitDefault = new THREE.Color(morning.skyHorizon).lerp(new THREE.Color(morning.groundB), 0.35);
    expect(shallow.getHex()).not.toBe(kitDefault.getHex());
    expect(kitDefault.b).toBeLessThan(kitDefault.r * 3);

    const curtain = module.scene.getObjectByName('mistCurtain0_0') as THREE.Mesh;
    const mistColor = (curtain.material as THREE.MeshStandardMaterial).color;
    const distance = Math.hypot(mistColor.r - shallow.r, mistColor.g - shallow.g, mistColor.b - shallow.b);
    expect(distance).toBeGreaterThan(0.5);
    module.dispose();
  });
});

describe('multi-key-corridor / the temple is built from the geometry it is handed', () => {
  test('the opening key cells and relic cell come from ctx.geometry, never a literal', async () => {
    const keys: GridCell[] = [
      { x: 2, y: 2 },
      { x: 6, y: 4 },
      { x: 10, y: 2 },
    ];
    const goal: GridCell = { x: 13, y: 5 };
    const geometry: EnvGeometry = {
      kind: 'grid',
      grid: {
        w: MKC_W,
        h: MKC_H,
        walls: mkcWalls(),
        goal,
        doors: MKC_DOORS.map((c) => ({ ...c })),
        keys: keys.map((c) => ({ ...c })),
        hallways: MKC_DOORS.map((c) => ({ ...c })),
      },
    };
    expect(keys).not.toEqual(MKC_KEYS.map((k) => ({ ...k })));
    expect(goal).not.toEqual({ ...MKC_GOAL });

    const module = createScene();
    await module.init(makeSceneContext({ geometry, seed: SEED, tier: 'hero' }));
    const grid = resolveGrid(geometry);
    const truth: [number, number, number] = [0, 0, 0];
    for (let j = 0; j < 3; j++) {
      const w = worldOf(module, `key-altar-${j}`);
      const k = keys[j] as GridCell;
      cellToWorld(k.x, k.y, grid, truth);
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
    }
    const shrine = worldOf(module, 'shrine');
    cellToWorld(goal.x, goal.y, grid, truth);
    expect(shrine.x).toBeCloseTo(truth[0], 10);
    expect(shrine.z).toBeCloseTo(truth[2], 10);
    module.dispose();
  });

  test('the resolvers fall back to the DECODER, and never to a scene-owned cell', () => {
    const real = resolveGrid(geometryOf());
    expect(real.keys).toBeDefined();
    expect(real.goal).toBeDefined();
    expect(resolveKeys(real) === real.keys).toBe(true);
    expect(resolveGoal(real) === real.goal).toBe(true);
    const placeholder = resolveGrid({ kind: 'grid' });
    expect(resolveKeys(placeholder).map((k) => ({ x: k.x, y: k.y }))).toEqual(MKC_KEYS.map((k) => ({ ...k })));
    expect({ ...resolveGoal(placeholder) }).toEqual({ ...MKC_GOAL });
    expect(resolveKeys({ w: MKC_W, h: MKC_H, walls: mkcWalls() }).length).toBe(3);
    expect({ ...resolveGoal({ w: MKC_W, h: MKC_H, walls: mkcWalls() }) }).toEqual({ ...MKC_GOAL });
  });
});

describe('multi-key-corridor / the hop is bounded, never a path plan', () => {
  test('a hop stays inside the segment between the two cells and lands on truth', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(frameOf('a', stateOf(1, 3, 15, 1, MKC_KEYS, 0), 0)));
    cellToWorld(1, 3, grid, from);
    cellToWorld(2, 3, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(frameOf('b', stateOf(2, 3, 15, 1, MKC_KEYS, 0), 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(frameOf('b', stateOf(2, 3, 15, 1, MKC_KEYS, 0), 1), { scrubbed: false }));
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

  test('a teleport hard-cuts the wisp across the temple with no glide', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(4);
    module.update(toFrameUpdate(frameOf('a', stateOf(14, 1, 15, 1, MKC_KEYS, 0b111111)), { scrubbed: false }));
    module.update(
      toFrameUpdate(frameOf('respawn', stateOf(2, 4, 13, 5, MKC_KEYS, 0, 1), 1), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(2, 4, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    expect(w.y).toBeCloseTo(0, 10);
    module.dispose();
  });

  test('a scrub snaps the wisp with no glide at all', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(4);
    module.update(toFrameUpdate(frameOf('a', stateOf(1, 1, 15, 1, MKC_KEYS, 0)), { scrubbed: false }));
    module.update(toFrameUpdate(frameOf('far', stateOf(15, 5, 15, 1, MKC_KEYS, 0b111111), 4000), { scrubbed: true }));
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(15, 5, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    expect(w.y).toBeCloseTo(0, 10);
    module.dispose();
  });
});

describe('multi-key-corridor / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('temple-start')));
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
    module.update(
      toFrameUpdate(frameOf('onGoal', stateOf(15, 1, 15, 1, MKC_KEYS, 0b111111), 100), { scrubbed: false }),
    );
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);
    expect(pool.pulse.bloom).toBe(0);

    ctx.setTime(2);
    module.update(
      toFrameUpdate(frameOf('scored', stateOf(2, 4, 13, 5, MKC_KEYS, 0, 1), 101), {
        scrubbed: false,
        prev: toFrameUpdate(frameOf('prev', stateOf(14, 1, 15, 1, MKC_KEYS, 0b111111), 100)).state,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    expect(pool.pulse.sigh).toBe(0);
    module.dispose();
  });

  test('the +1 blooms on the relic that was reached, NOT on the respawn cell', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    ctx.setTime(3);
    const prev = toFrameUpdate(frameOf('run', stateOf(14, 1, 15, 1, MKC_KEYS, 0b111111), 200)).state;
    module.update(toFrameUpdate(frameOf('run', stateOf(14, 1, 15, 1, MKC_KEYS, 0b111111), 200), { scrubbed: false }));
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(frameOf('scored', stateOf(2, 4, 13, 5, MKC_KEYS, 0, 1), 201), {
        scrubbed: false,
        prev,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    const light = module.scene.getObjectByName('oakPoint2') as THREE.PointLight;
    expect(light).toBeDefined();
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(15, 1, grid, truth);
    expect(light.position.x).toBeCloseTo(truth[0], 10);
    expect(light.position.z).toBeCloseTo(truth[2], 10);
    expect(light.intensity).toBeGreaterThan(0);
    const agent = worldOf(module, 'wisp-anchor');
    cellToWorld(2, 4, grid, truth);
    expect(agent.x).toBeCloseTo(truth[0], 10);
    expect(Math.abs(light.position.x - agent.x)).toBeGreaterThan(10);
    const shrine = worldOf(module, 'shrine');
    cellToWorld(13, 5, grid, truth);
    expect(shrine.x).toBeCloseTo(truth[0], 10);
    expect(shrine.z).toBeCloseTo(truth[2], 10);
    expect(module.scene.getObjectByName('shrineDecor')?.scale.x ?? 1).toBeLessThan(0.2);
    module.dispose();
  });

  test('key and door events use the mid-tier grammar: a burst, never a ripple', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];

    ctx.setTime(1);
    module.update(
      toFrameUpdate(frameOf('gotKey', stateOf(3, 5, 15, 1, MKC_KEYS, 0b001), 10), {
        scrubbed: false,
        events: [{ type: 'key', index: 0 }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(0);
    let light = module.scene.getObjectByName('oakPoint2') as THREE.PointLight;
    const key0 = MKC_KEYS[0] as GridCell;
    cellToWorld(key0.x, key0.y, grid, truth);
    expect(light.position.x).toBeCloseTo(truth[0], 10);
    expect(light.position.z).toBeCloseTo(truth[2], 10);

    ctx.setTime(1.5);
    module.update(
      toFrameUpdate(frameOf('opened', stateOf(4, 3, 15, 1, MKC_KEYS, 0b1001), 11), {
        scrubbed: false,
        events: [{ type: 'door', index: 0, open: true }],
      }),
    );
    expect(pool.stats.bursts).toBe(2);
    expect(pool.stats.ripples).toBe(0);
    light = module.scene.getObjectByName('oakPoint2') as THREE.PointLight;
    const door0 = MKC_DOORS[0] as GridCell;
    cellToWorld(door0.x, door0.y, grid, truth);
    expect(light.position.x).toBeCloseTo(truth[0], 10);

    ctx.setTime(2);
    module.update(
      toFrameUpdate(frameOf('relocked', stateOf(2, 2, 13, 5, MKC_KEYS, 0, 1), 12), {
        scrubbed: false,
        events: [{ type: 'door', index: 0, open: false }],
      }),
    );
    expect(pool.stats.bursts).toBe(2);
    module.dispose();
  });

  test('a scrub clears every in-flight beat and settles the new keys and relic', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(frameOf('scored', stateOf(2, 4, 13, 5, MKC_KEYS, 0, 1), 300), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'teleport' }, { type: 'done' }],
      }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('temple-start'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    expect(module.scene.getObjectByName('shrineDecor')?.scale.x ?? 0).toBeCloseTo(1, 6);
    expect(module.scene.getObjectByName('keyAltarDecor0')?.scale.x ?? 0).toBeCloseTo(1, 6);
    module.dispose();
  });

  test('the growth and proximity envelopes are pure, bounded and land on 1', () => {
    expect(growth(-Infinity)).toBe(1);
    expect(growth(Number.NaN)).toBe(1);
    expect(growth(0)).toBe(0);
    expect(growth(99)).toBe(1);
    for (let i = 0; i <= 50; i++) {
      const g = growth((i / 50) * 0.72);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(1.05);
    }
    const far = proximity({ x: 1, y: 1, gx: 15, gy: 1 });
    const near = proximity({ x: 14, y: 1, gx: 15, gy: 1 });
    expect(far).toBe(0);
    expect(near).toBeGreaterThan(0);
    expect(near).toBeLessThanOrEqual(1);
  });
});

describe('multi-key-corridor / lifecycle', () => {
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
    module.update(toFrameUpdate(fixtureNamed('on-the-ford')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'templeGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'templeGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the camera stays a clamped 3/4 aerial over the whole hall', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('temple-start')));
    const cam = module.camera;
    expect(cam.position.y).toBeGreaterThan(6);
    expect(Math.hypot(cam.position.x, cam.position.z)).toBeGreaterThan(TEMPLE_HALF_Z);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.y).toBeLessThan(-0.5);
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    cam.updateMatrixWorld();
    const frustum = new THREE.Frustum().setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse),
    );
    const grid = resolveGrid(geometryOf());
    const corner: [number, number, number] = [0, 0, 0];
    for (const c of [
      { x: 0, y: 0 },
      { x: MKC_W - 1, y: 0 },
      { x: 0, y: MKC_H - 1 },
      { x: MKC_W - 1, y: MKC_H - 1 },
    ]) {
      cellToWorld(c.x, c.y, grid, corner);
      expect(frustum.containsPoint(new THREE.Vector3(corner[0], 0.4, corner[2]))).toBe(true);
    }
    module.dispose();
  });

  test("the start fixture really is the env's deterministic init", () => {
    const start = fixtureNamed('temple-start');
    expect(start.pose[0]).toBe(MKC_START.x);
    expect(start.pose[1]).toBe(MKC_START.y);
    expect(start.pose[2]).toBe(MKC_GOAL.x);
    expect(start.pose[3]).toBe(MKC_GOAL.y);
    expect(start.pose[4]).toBe(0);
    expect(start.pose[5]).toBe(0);
    const init = multiKeyCorridorDecoder.initFrame(SEED, multiKeyCorridorDecoder.initStatic(SEED));
    expect(Array.from(init.pose.subarray(0, 6))).toEqual([
      MKC_START.x,
      MKC_START.y,
      MKC_GOAL.x,
      MKC_GOAL.y,
      0,
      0,
    ]);
  });
});

describe('multi-key-corridor / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'multi-key-corridor (thumb, overlay on)');
    module.dispose();
  });
});
