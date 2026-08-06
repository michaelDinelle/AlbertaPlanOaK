/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import { cellToWorld, isWall, worldToCell } from '@kit/grid';
import { TOKENS } from '@kit/tokens';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  LARGE_DEEPSEA_LAST_ROW,
  LARGE_DEEPSEA_N,
  LARGE_DEEPSEA_RIGHT_COST,
  largeDeepSeaBit,
  largeDeepSeaDecoder,
  largeDeepSeaNextCol,
} from '@sim/envs/largedeepsea';
import type { EnvGeometry, GridCell } from '@sim/types';

import createScene, {
  bubbleGain,
  envelope,
  isDiveStep,
  readRightMove,
  siltLift,
} from '../../src/scenes/large-deep-sea/index';
import { fixtures } from '../../src/scenes/large-deep-sea/fixtures';
import { abyssPose, manifest } from '../../src/scenes/large-deep-sea/manifest';
import { HEART_Y, WRECK_SCALE } from '../../src/scenes/large-deep-sea/actors';
import {
  CORAL_LEAN,
  EXTINCTION,
  SURFACE_GAIN,
  abyssSkin,
  anglerSites,
  attenuate,
  coralLeanAt,
  hue,
  isShelfCell,
  isWalkableCell,
  makeTrenchHeight,
  resolveGrid,
  resolveMask,
  shelfCellCount,
  shelfTint,
  BED_SOUTH,
  SHELF_THICKNESS,
  TRENCH_HALF,
  WALL_HEIGHT,
  WALL_RUN,
} from '../../src/scenes/large-deep-sea/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 72;

function staticOf(): ReturnType<typeof largeDeepSeaDecoder.initStatic> {
  return largeDeepSeaDecoder.initStatic(SEED);
}

function geometryOf(): EnvGeometry {
  return largeDeepSeaDecoder.geometry(staticOf(), SEED);
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

function diveFixture(name: string, r: number, c: number, step = 0, action = 0): SceneFixture {
  return {
    name,
    pose: abyssPose(r, c),
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

function luminance(c: THREE.Color): number {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

function objectNamed(module: SceneModule, name: string): THREE.Object3D {
  const o = module.scene.getObjectByName(name);
  if (!o) throw new Error(`object "${name}" missing`);
  return o;
}

function densityOf(module: SceneModule, name: string): number {
  const points = objectNamed(module, name) as THREE.Points;
  return ((points.material as THREE.ShaderMaterial).uniforms.uDensity as { value: number }).value;
}

function scoringFrame(step: number): ReturnType<typeof toFrameUpdate> {
  return toFrameUpdate(diveFixture('scored', 0, 0, step), {
    scrubbed: false,
    events: [{ type: 'goal' }, { type: 'resurface' }],
  });
}


describe('large-deep-sea / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('LargeDeepSea-cont-v0');
    expect(manifest.slug).toBe('large-deep-sea');
    expect(manifest.accent).toBe('large-deep-sea');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a state the agent can actually be in', () => {
    const p = manifest.thumbFrame.pose;
    const r = p[0] ?? -1;
    const c = p[1] ?? -1;
    expect(isWalkableCell(c, r, LARGE_DEEPSEA_N)).toBe(true);
    expect(p[2]).toBe(LARGE_DEEPSEA_N - 1);
    expect(p[3]).toBe(LARGE_DEEPSEA_N - 1);
  });

  test('ships at least three canonical poses, every one of them reachable', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      const r = f.pose[0] ?? -1;
      const c = f.pose[1] ?? -1;
      expect(isWalkableCell(c, r, LARGE_DEEPSEA_N)).toBe(true);
      expect(c).toBeLessThanOrEqual(r);
    }
  });

  test('the fixtures cover the surface, the hypotenuse and the last row', () => {
    expect(fixtures.some((f) => f.pose[0] === 0 && f.pose[1] === 0)).toBe(true);
    expect(fixtures.some((f) => f.pose[0] === f.pose[1] && (f.pose[0] ?? 0) > 0)).toBe(true);
    expect(fixtures.some((f) => f.pose[0] === LARGE_DEEPSEA_LAST_ROW)).toBe(true);
  });

  test('the treasure-reach fixture really is one right-move from the wreck', () => {
    const f = fixtureNamed('treasure-reach');
    const r = f.pose[0] ?? 0;
    const c = f.pose[1] ?? 0;
    expect(r).toBe(LARGE_DEEPSEA_LAST_ROW);
    const st = staticOf();
    const landed = [0, 1].map((a) => largeDeepSeaNextCol(st, r, c, a));
    expect(landed).toContain(LARGE_DEEPSEA_N - 1);
  });
});

describe('large-deep-sea / the shelf IS the reachable set', () => {
  test('simulating the env from (0,0) reaches exactly the cells `isShelfCell` claims', () => {
    const st = staticOf();
    let row = new Set<number>([0]);
    for (let r = 0; r <= LARGE_DEEPSEA_LAST_ROW; r++) {
      for (const c of row) {
        expect(isShelfCell(c, r)).toBe(true);
      }
      for (let c = 0; c <= r; c++) expect(row.has(c)).toBe(true);
      expect(row.size).toBe(r + 1);
      if (r === LARGE_DEEPSEA_LAST_ROW) break;
      const next = new Set<number>();
      for (const c of row) for (const a of [0, 1]) next.add(largeDeepSeaNextCol(st, r, c, a));
      row = next;
    }
  });

  test('the shelf covers the triangle and nothing else', () => {
    let n = 0;
    for (let y = 0; y < LARGE_DEEPSEA_N; y++) {
      for (let x = 0; x < LARGE_DEEPSEA_N; x++) {
        expect(isShelfCell(x, y)).toBe(x <= y);
        if (isShelfCell(x, y)) n++;
      }
    }
    expect(n).toBe(shelfCellCount(LARGE_DEEPSEA_N));
    expect(n).toBe(136);
    expect(isShelfCell(LARGE_DEEPSEA_N - 1, LARGE_DEEPSEA_N - 1)).toBe(true);
    expect(isWalkableCell(LARGE_DEEPSEA_N - 1, LARGE_DEEPSEA_N - 1, LARGE_DEEPSEA_N)).toBe(false);
  });

  test('resolveGrid returns the decoder geometry untouched; the fallback is the decoder\'s own N', () => {
    const geometry = geometryOf();
    expect(geometry.grid).toBeDefined();
    expect(resolveGrid(geometry) === geometry.grid).toBe(true);
    const fallback = resolveGrid({ kind: 'grid' });
    expect(fallback.w).toBe(LARGE_DEEPSEA_N);
    expect(fallback.h).toBe(LARGE_DEEPSEA_N);
    expect(fallback.goal).toEqual({ x: LARGE_DEEPSEA_N - 1, y: LARGE_DEEPSEA_N - 1 });
    for (let y = 0; y < fallback.h; y++) {
      for (let x = 0; x < fallback.w; x++) expect(isWall(fallback, x, y)).toBe(false);
    }
  });

  test('resolveMask uses the decoder\'s mask, and reproduces it from the seed when absent', () => {
    const grid = resolveGrid(geometryOf());
    const fromGeometry = resolveMask(grid, SEED);
    expect(fromGeometry === grid.actionMask).toBe(true);
    const regenerated = resolveMask(resolveGrid({ kind: 'grid' }), SEED);
    expect(regenerated.length).toBe(LARGE_DEEPSEA_N * LARGE_DEEPSEA_N);
    for (let i = 0; i < regenerated.length; i++) expect(regenerated[i]).toBe(fromGeometry[i]);
  });
});

describe('large-deep-sea / the depth falloff is the light', () => {
  const skin = abyssSkin(TOKENS);

  test('shelfTint darkens strictly monotonically from the surface to the floor', () => {
    let previous = Infinity;
    for (let r = 0; r < LARGE_DEEPSEA_N; r++) {
      const l = luminance(shelfTint(r, LARGE_DEEPSEA_N, skin));
      expect(l).toBeLessThan(previous);
      previous = l;
    }
  });

  test('the surface row is ocean-bright and the floor row is near-black', () => {
    const top = shelfTint(0, LARGE_DEEPSEA_N, skin);
    const floor = shelfTint(LARGE_DEEPSEA_N - 1, LARGE_DEEPSEA_N, skin);
    expect(luminance(top)).toBeGreaterThan(1);
    expect(luminance(floor)).toBeLessThan(luminance(top) / 20);
    expect(floor.r).toBeLessThan(top.r / 10);
    expect(floor.g / floor.b).toBeLessThan((top.g / top.b) / 3);
    expect(floor.r / floor.b).toBeGreaterThan(top.r / top.b);
  });

  test('attenuate is Beer–Lambert per channel, red first', () => {
    expect(EXTINCTION[0]).toBeGreaterThan(EXTINCTION[1]);
    expect(EXTINCTION[1]).toBeGreaterThan(EXTINCTION[2]);
    const c = attenuate(new THREE.Color(1, 1, 1), 0);
    expect(c.r).toBeCloseTo(SURFACE_GAIN, 10);
    const half = attenuate(new THREE.Color(1, 1, 1), 0.5);
    expect(half.r).toBeCloseTo(SURFACE_GAIN * Math.exp(-EXTINCTION[0] * 0.5), 10);
    expect(attenuate(new THREE.Color(1, 1, 1), -3).r).toBeCloseTo(SURFACE_GAIN, 10);
    expect(attenuate(new THREE.Color(1, 1, 1), 9).b).toBeCloseTo(SURFACE_GAIN * Math.exp(-EXTINCTION[2]), 10);
  });

  test('hue() factors brightness out and leaves the ratios alone', () => {
    const raw = new THREE.Color(0.2, 0.5, 0.8);
    const normalized = hue(raw.clone());
    expect(Math.max(normalized.r, normalized.g, normalized.b)).toBeCloseTo(1, 10);
    expect(normalized.r / normalized.b).toBeCloseTo(0.2 / 0.8, 10);
  });

  test('the trench bed is always below the shelf slabs, so no terrace floats', () => {
    const h = makeTrenchHeight(SEED);
    const slabBottom = -SHELF_THICKNESS * 0.96;
    for (let i = 0; i <= 24; i++) {
      for (let j = 0; j <= 24; j++) {
        const x = -TRENCH_HALF + (2 * TRENCH_HALF * i) / 24;
        const z = -TRENCH_HALF + (2 * TRENCH_HALF * j) / 24;
        const bed = h(x, z);
        expect(bed).toBeGreaterThan(slabBottom);
        expect(bed).toBeLessThan(0);
      }
    }
    expect(h(0, TRENCH_HALF - 0.5)).toBeLessThan(h(0, -TRENCH_HALF + 0.5));
    expect(BED_SOUTH).toBeLessThan(0);
  });
});

describe('large-deep-sea / the coral tells the truth about the mask', () => {
  test('every whisker leans the way its cell\'s mask bit says', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const mask = resolveMask(grid, SEED);
    const coral = module.scene.getObjectByName('abyssCoral') as THREE.InstancedMesh | undefined;
    expect(coral).toBeDefined();
    if (!coral) return;
    expect(coral.count).toBe(shelfCellCount(LARGE_DEEPSEA_N));

    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const up = new THREE.Vector3();
    const cell: GridCell = { x: 0, y: 0 };
    const truth: [number, number, number] = [0, 0, 0];
    const seen = new Set<number>();

    for (let i = 0; i < coral.count; i++) {
      coral.getMatrixAt(i, m);
      m.decompose(p, q, s);
      worldToCell(p.x, p.z, grid, cell);
      expect(isShelfCell(cell.x, cell.y)).toBe(true);
      cellToWorld(cell.x, cell.y, grid, truth);
      expect(Math.abs(p.x - truth[0])).toBeLessThan(0.5);
      expect(Math.abs(p.z - truth[2])).toBeLessThan(0.5);
      expect(Math.abs(p.x - truth[0])).toBeGreaterThan(0.15);
      expect(p.y).toBeCloseTo(0, 10);

      up.set(0, 1, 0).applyQuaternion(q);
      expect(Math.abs(up.z)).toBeLessThan(1e-9);
      const bit = mask[cell.y * grid.w + cell.x] ?? 0;
      expect(Math.sign(up.x)).toBe(bit !== 0 ? 1 : -1);
      expect(Math.sign(coralLeanAt(mask, grid, cell.x, cell.y))).toBe(Math.sign(up.x));
      expect(Math.abs(up.x)).toBeLessThanOrEqual(Math.sin(CORAL_LEAN * 1.2) + 1e-9);

      const key = cell.y * grid.w + cell.x;
      expect(seen.has(key)).toBe(false);
      seen.add(key);
    }
    expect(seen.size).toBe(shelfCellCount(LARGE_DEEPSEA_N));
    module.dispose();
  });

  test('the anglerfish patrol real shelf cells, never the wisp\'s own column', () => {
    const grid = resolveGrid(geometryOf());
    for (const seed of [0, 1, 72, 73, 79, 12345]) {
      const sites = anglerSites(grid, seed);
      expect(sites.length).toBe(2);
      for (const site of sites) {
        expect(isWalkableCell(site.x, site.y, LARGE_DEEPSEA_N)).toBe(true);
        expect(site.x).toBeGreaterThanOrEqual(2);
        expect(site.x).toBeLessThan(site.y);
        expect(site.hover).toBeGreaterThan(0);
      }
      expect((sites[1]?.y ?? 0) > (sites[0]?.y ?? 0)).toBe(true);
    }
  });
});

describe('large-deep-sea / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    expect(report.worst.get('wisp') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('wreck') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('wreck-heart') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('angler-site') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('silt') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      expect(report.worst.get('wisp-anchor') ?? 1).toBeLessThan(1e-9);
      expect(report.worst.get('wreck-heart') ?? 1).toBeLessThan(1e-9);
      expect(report.worst.get('silt') ?? 1).toBeLessThan(1e-9);
      module.dispose();
    }
  });

  test('the wisp lands on EVERY cell the agent can occupy', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    let visited = 0;
    for (let r = 0; r <= LARGE_DEEPSEA_LAST_ROW; r++) {
      for (let c = 0; c <= r; c++) {
        module.update(toFrameUpdate(diveFixture(`cell${c}-${r}`, r, c, visited)));
        const w = worldOf(module, 'wisp-anchor');
        cellToWorld(c, r, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 10);
        expect(w.y).toBeCloseTo(0, 10);
        expect(w.z).toBeCloseTo(truth[2], 10);
        expect(worldOf(module, 'wisp').y).toBeCloseTo(TOKENS.wispHeight, 10);
        visited++;
      }
    }
    expect(visited).toBe(shelfCellCount(LARGE_DEEPSEA_N - 1));
    module.dispose();
  });

  test('the wreck sits on the treasure cell and its heart exactly HEART_Y above it', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    module.update(toFrameUpdate(fixtureNamed('deep-dive')));
    cellToWorld(LARGE_DEEPSEA_N - 1, LARGE_DEEPSEA_N - 1, grid, truth);
    const root = worldOf(module, 'wreck');
    expect(root.x).toBeCloseTo(truth[0], 10);
    expect(root.y).toBeCloseTo(0, 10);
    expect(root.z).toBeCloseTo(truth[2], 10);
    const heart = worldOf(module, 'wreck-heart');
    expect(heart.x).toBeCloseTo(truth[0], 10);
    expect(heart.y).toBeCloseTo(HEART_Y, 10);
    expect(heart.z).toBeCloseTo(truth[2], 10);
    const hull = module.scene.getObjectByName('wreckHull');
    expect(hull?.scale.x ?? 0).toBeCloseTo(WRECK_SCALE, 10);
    expect(hull?.rotation.y ?? 0).not.toBe(0);
    module.dispose();
  });
});

describe('large-deep-sea / the wrap is a cut, never a glide', () => {
  test('a dive step is a bounded hop that lands on truth', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const from: [number, number, number] = [0, 0, 0];
    const to: [number, number, number] = [0, 0, 0];

    ctx.setTime(10);
    module.update(toFrameUpdate(diveFixture('a', 6, 2, 0)));
    cellToWorld(2, 6, grid, from);
    cellToWorld(3, 7, grid, to);

    ctx.setTime(10);
    module.update(toFrameUpdate(diveFixture('b', 7, 3, 1), { scrubbed: false }));
    for (let k = 0; k <= 8; k++) {
      ctx.setTime(10 + (k / 8) * 0.125);
      module.update(toFrameUpdate(diveFixture('b', 7, 3, 1), { scrubbed: false }));
      const w = worldOf(module, 'wisp-anchor');
      expect(w.x).toBeGreaterThanOrEqual(Math.min(from[0], to[0]) - 1e-6);
      expect(w.x).toBeLessThanOrEqual(Math.max(from[0], to[0]) + 1e-6);
      expect(w.z).toBeGreaterThanOrEqual(Math.min(from[2], to[2]) - 1e-6);
      expect(w.z).toBeLessThanOrEqual(Math.max(from[2], to[2]) + 1e-6);
      expect(w.y).toBeGreaterThanOrEqual(-1e-6);
    }
    const landed = worldOf(module, 'wisp-anchor');
    expect(landed.x).toBeCloseTo(to[0], 10);
    expect(landed.z).toBeCloseTo(to[2], 10);
    expect(landed.y).toBeCloseTo(0, 10);
    module.dispose();
  });

  test('the resurface wrap snaps instantly — with the event AND without it', async () => {
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(0, 0, grid, truth);

    for (const withEvent of [true, false]) {
      const { module, ctx } = await live();
      ctx.setTime(5);
      module.update(toFrameUpdate(diveFixture('deep', 14, 12, 100), { scrubbed: false }));
      ctx.setTime(5.125);
      module.update(
        toFrameUpdate(diveFixture('wrapped', 0, 0, 101), {
          scrubbed: false,
          ...(withEvent ? { events: [{ type: 'resurface' as const }] } : {}),
        }),
      );
      const w = worldOf(module, 'wisp-anchor');
      expect(w.x).toBeCloseTo(truth[0], 10);
      expect(w.y).toBeCloseTo(0, 10);
      expect(w.z).toBeCloseTo(truth[2], 10);
      module.dispose();
    }
  });

  test('isDiveStep accepts exactly the transitions the env can produce', () => {
    expect(isDiveStep(3, 7, 4, 8)).toBe(true);
    expect(isDiveStep(3, 7, 2, 8)).toBe(true);
    expect(isDiveStep(0, 7, 0, 8)).toBe(true);
    expect(isDiveStep(3, 7, 3, 8)).toBe(true);
    expect(isDiveStep(12, 14, 0, 0)).toBe(false);
    expect(isDiveStep(3, 7, 5, 8)).toBe(false);
    expect(isDiveStep(3, 7, 4, 9)).toBe(false);
    expect(isDiveStep(3, 7, 4, 7)).toBe(false);
  });

  test('the bubble rush flies from the cell the dive ended on, then parks', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const rush = module.scene.getObjectByName('resurfaceRush') as THREE.Points | undefined;
    expect(rush).toBeDefined();
    if (!rush) return;
    const density = (): number =>
      ((rush.material as THREE.ShaderMaterial).uniforms.uDensity as { value: number }).value;

    ctx.setTime(20);
    module.update(toFrameUpdate(diveFixture('deep', 14, 9, 300), { scrubbed: false }));
    expect(density()).toBe(0);

    ctx.setTime(20.125);
    const prev = toFrameUpdate(diveFixture('deep', 14, 9, 300)).state;
    module.update(
      toFrameUpdate(diveFixture('wrapped', 0, 0, 301), {
        scrubbed: false,
        prev,
        events: [{ type: 'resurface' }],
      }),
    );
    expect(density()).toBe(0);

    const start: [number, number, number] = [0, 0, 0];
    const surface: [number, number, number] = [0, 0, 0];
    cellToWorld(9, 14, grid, start);
    cellToWorld(0, 0, grid, surface);

    ctx.setTime(20.125 + 0.1);
    module.update(toFrameUpdate(diveFixture('wrapped', 0, 0, 301), { scrubbed: false }));
    expect(density()).toBeGreaterThan(0);
    const toStart = Math.hypot(rush.position.x - start[0], rush.position.z - start[2]);
    const toSurface = Math.hypot(rush.position.x - surface[0], rush.position.z - surface[2]);
    expect(toStart).toBeLessThan(toSurface);

    ctx.setTime(20.125 + 1.1);
    module.update(toFrameUpdate(diveFixture('wrapped', 0, 0, 301), { scrubbed: false }));
    expect(Math.hypot(rush.position.x - surface[0], rush.position.z - surface[2])).toBeLessThan(1.5);

    ctx.setTime(20.125 + 3);
    module.update(toFrameUpdate(diveFixture('wrapped', 0, 0, 301), { scrubbed: false }));
    expect(density()).toBe(0);
    module.dispose();
  });
});

describe('large-deep-sea / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('abyss-start')));
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

  test('the dive bubble reads the ROW and nothing else', async () => {
    const { module, ctx } = await live();
    const inner = module.scene.getObjectByName('diveBubbleInner') as THREE.Mesh | undefined;
    expect(inner).toBeDefined();
    const gainOf = (): number => {
      const m = (inner as THREE.Mesh).material as THREE.MeshBasicMaterial;
      return m.opacity;
    };
    ctx.setTime(3);
    module.update(toFrameUpdate(diveFixture('surface', 0, 0, 1)));
    const shallow = gainOf();
    module.update(toFrameUpdate(diveFixture('floor', 14, 3, 2)));
    const deep = gainOf();
    expect(deep).toBeGreaterThan(shallow);
    module.update(toFrameUpdate(diveFixture('floor', 14, 3, 2), { scrubbed: false }));
    expect(gainOf()).toBeCloseTo(deep, 12);

    const grid = { w: LARGE_DEEPSEA_N, h: LARGE_DEEPSEA_N };
    expect(bubbleGain(0, grid)).toBe(0);
    expect(bubbleGain(LARGE_DEEPSEA_LAST_ROW, grid)).toBe(1);
    expect(bubbleGain(7, grid)).toBeCloseTo(0.5, 10);
    module.dispose();
  });

  test('celebrations fire ONLY from frame.events', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    expect(pool).not.toBeNull();
    if (!pool) return;

    ctx.setTime(1);
    module.update(toFrameUpdate(diveFixture('brink', 14, 14, 100), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);
    expect(pool.pulse.bloom).toBe(0);

    ctx.setTime(2);
    module.update(
      toFrameUpdate(diveFixture('scored', 0, 0, 101), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'resurface' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    module.dispose();
  });

  test('the bloom happens at the WRECK, not where the agent landed', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    ctx.setTime(3);
    module.update(toFrameUpdate(diveFixture('brink', 14, 14, 200), { scrubbed: false }));
    ctx.setTime(3.125);
    module.update(
      toFrameUpdate(diveFixture('scored', 0, 0, 201), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'resurface' }],
      }),
    );
    const light = module.scene.getObjectByName('oakPoint1') as THREE.PointLight | undefined;
    expect(light).toBeDefined();
    if (!light) return;
    const truth: [number, number, number] = [0, 0, 0];
    cellToWorld(LARGE_DEEPSEA_N - 1, LARGE_DEEPSEA_N - 1, grid, truth);
    expect(light.position.x).toBeCloseTo(truth[0], 10);
    expect(light.position.y).toBeCloseTo(HEART_Y, 10);
    expect(light.position.z).toBeCloseTo(truth[2], 10);
    expect(light.intensity).toBeGreaterThan(4);
    module.dispose();
  });

  test('LargeDeepSea never sighs: there is no failure in this environment', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(diveFixture('everything', 0, 0, 400), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'resurface' }],
      }),
    );
    expect(pool.pulse.sigh).toBe(0);
    module.dispose();
  });

  test('a scrub clears every in-flight beat', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    const core = objectNamed(module, 'wreckHeartCore');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(diveFixture('scored', 0, 0, 500), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'resurface' }],
      }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    const blazing = core.scale.x;
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('shallow-shelf'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    expect(densityOf(module, 'resurfaceRush')).toBe(0);
    const idle = 1 + 0.12 * (0.5 + 0.5 * Math.sin(1.01 * 1.35));
    expect(blazing).toBeGreaterThan(idle + 0.3);
    expect(core.scale.x).toBeCloseTo(idle, 12);
    module.dispose();
  });

  test('a scrub across the whole trench snaps the wisp with no glide at all', async () => {
    const { module, ctx } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    ctx.setTime(4);
    module.update(toFrameUpdate(diveFixture('a', 1, 1, 0), { scrubbed: false }));
    module.update(toFrameUpdate(diveFixture('far', 14, 12, 4000), { scrubbed: true }));
    const w = worldOf(module, 'wisp-anchor');
    cellToWorld(12, 14, grid, truth);
    expect(w.x).toBeCloseTo(truth[0], 10);
    expect(w.z).toBeCloseTo(truth[2], 10);
    expect(w.y).toBeCloseTo(0, 10);
    module.dispose();
  });

  test('the envelope is pure, bounded, and lands on zero', () => {
    expect(envelope(-Infinity, 2, 2)).toBe(0);
    expect(envelope(Number.NaN, 2, 2)).toBe(0);
    expect(envelope(0, 2, 2)).toBe(1);
    expect(envelope(2, 2, 2)).toBe(0);
    expect(envelope(99, 2, 2)).toBe(0);
    for (let i = 0; i <= 50; i++) {
      const e = envelope((i / 50) * 2, 2, 2);
      expect(e).toBeGreaterThanOrEqual(0);
      expect(e).toBeLessThanOrEqual(1);
    }
  });
});

describe('large-deep-sea / every flourish is on the playback clock', () => {

  test('a paused playback freezes the blaze — mesh and light alike', async () => {
    const { module, ctx } = await live();
    const core = objectNamed(module, 'wreckHeartCore');
    const halo = objectNamed(module, 'wreckHeartHalo');
    const light = objectNamed(module, 'oakPoint1') as THREE.PointLight;

    ctx.setTime(9);
    module.update(toFrameUpdate(diveFixture('brink', 14, 14, 900), { scrubbed: false }));
    const idleScale = core.scale.x;

    ctx.setTime(9.125);
    module.update(scoringFrame(901));
    const blazeCore = core.scale.x;
    const blazeHalo = halo.scale.x;
    const blazeLight = light.intensity;
    expect(blazeCore).toBeGreaterThan(idleScale + 0.3);

    for (let i = 0; i < 40; i++) {
      module.update(toFrameUpdate(diveFixture('scored', 0, 0, 901), { scrubbed: false, dtReal: 1 / 60 }));
    }
    expect(core.scale.x).toBeCloseTo(blazeCore, 12);
    expect(halo.scale.x).toBeCloseTo(blazeHalo, 12);
    expect(light.intensity).toBeCloseTo(blazeLight, 12);
    module.dispose();
  });

  test('the blaze is a pure function of ctx.time(): the wall clock cannot move it', async () => {
    const readAt = async (dt: number, steps: number): Promise<number> => {
      const { module, ctx } = await live();
      const core = objectNamed(module, 'wreckHeartCore');
      ctx.setTime(4);
      module.update(toFrameUpdate(diveFixture('brink', 14, 14, 10), { scrubbed: false }));
      module.update(scoringFrame(11));
      for (let i = 1; i <= steps; i++) {
        ctx.setTime(4 + (0.4 * i) / steps);
        module.update(toFrameUpdate(diveFixture('scored', 0, 0, 11), { scrubbed: false, dtReal: dt }));
      }
      const v = core.scale.x;
      module.dispose();
      return v;
    };
    const oneSlowFrame = await readAt(0.4, 1);
    const manyFatFrames = await readAt(0.5, 30);
    expect(manyFatFrames).toBeCloseTo(oneSlowFrame, 12);
    expect(oneSlowFrame).toBeGreaterThan(1.2);
  });

  test('a scrub puts the heart back to idle instead of leaving it burning', async () => {
    const { module, ctx } = await live();
    const core = objectNamed(module, 'wreckHeartCore');
    const light = objectNamed(module, 'oakPoint1') as THREE.PointLight;
    ctx.setTime(5);
    module.update(toFrameUpdate(diveFixture('brink', 14, 14, 20), { scrubbed: false }));
    module.update(scoringFrame(21));
    const blazing = core.scale.x;

    ctx.setTime(5.01);
    module.update(toFrameUpdate(fixtureNamed('shallow-shelf'), { scrubbed: true }));
    const scrubbed = core.scale.x;
    const scrubbedLight = light.intensity;

    const fresh = await live();
    fresh.ctx.setTime(5.01);
    fresh.module.update(toFrameUpdate(fixtureNamed('shallow-shelf'), { scrubbed: true }));
    const idle = objectNamed(fresh.module, 'wreckHeartCore').scale.x;
    const idleLight = (objectNamed(fresh.module, 'oakPoint1') as THREE.PointLight).intensity;

    expect(blazing).toBeGreaterThan(idle + 0.3);
    expect(scrubbed).toBeCloseTo(idle, 12);
    expect(scrubbedLight).toBeCloseTo(idleLight, 12);
    fresh.module.dispose();
    module.dispose();
  });
});

describe('large-deep-sea / the right-move cost is drawn', () => {

  test('the decoder really does charge for the action bit', () => {
    expect(LARGE_DEEPSEA_RIGHT_COST).toBeLessThan(0);
    expect(LARGE_DEEPSEA_RIGHT_COST).toBeCloseTo(-0.01 / LARGE_DEEPSEA_N, 12);
  });

  test('readRightMove prefers the decoder, falls back to ctrl[0], and never returns junk', () => {
    const withDerived = toFrameUpdate(diveFixture('a', 4, 2, 0, 1)).state;
    expect(readRightMove(withDerived)).toBe(1);
    const quiet = toFrameUpdate(diveFixture('b', 4, 2, 0, 0)).state;
    expect(readRightMove(quiet)).toBe(0);
    const rawOn = toFrameUpdate({ name: 'raw', pose: abyssPose(4, 2), ctrl: [3, 0, 0, 0, 0, 0] }).state;
    expect(readRightMove(rawOn)).toBe(1);
    const rawOff = toFrameUpdate({ name: 'raw', pose: abyssPose(4, 2), ctrl: [2, 0, 0, 0, 0, 0] }).state;
    expect(readRightMove(rawOff)).toBe(0);
    const nan = toFrameUpdate({ name: 'nan', pose: abyssPose(4, 2), ctrl: [Number.NaN, 0, 0, 0, 0, 0] }).state;
    expect(readRightMove(nan)).toBe(0);
  });

  test('the silt fires on the charged action and settles on the free one', async () => {
    const { module, ctx } = await live();
    ctx.setTime(2);
    module.update(toFrameUpdate(diveFixture('free', 6, 3, 40, 0)));
    const quiet = densityOf(module, 'diveSilt');
    module.update(toFrameUpdate(diveFixture('charged', 7, 4, 41, 1)));
    const kicked = densityOf(module, 'diveSilt');
    expect(kicked).toBeGreaterThan(quiet + 0.5);
    expect(quiet).toBeGreaterThanOrEqual(0);
    expect(kicked).toBeLessThanOrEqual(1);
    module.dispose();
  });

  test('it is the COST, not the direction — action 1 kicks silt even where the mask sends it left', async () => {
    const st = staticOf();
    let right: GridCell | null = null;
    let left: GridCell | null = null;
    for (let r = 2; r <= LARGE_DEEPSEA_LAST_ROW && (!right || !left); r++) {
      for (let c = 1; c < r && (!right || !left); c++) {
        const bit = largeDeepSeaBit(st, r, c);
        if (bit === 1 && !right) right = { x: c, y: r };
        if (bit === 0 && !left) left = { x: c, y: r };
      }
    }
    expect(right).not.toBeNull();
    expect(left).not.toBeNull();
    if (!right || !left) return;
    expect(largeDeepSeaNextCol(st, right.y, right.x, 1)).toBe(right.x + 1);
    expect(largeDeepSeaNextCol(st, left.y, left.x, 1)).toBe(left.x - 1);

    const { module, ctx } = await live();
    ctx.setTime(3);
    module.update(toFrameUpdate(diveFixture('goesRight', right.y, right.x, 50, 1)));
    const kickedRight = densityOf(module, 'diveSilt');
    module.update(toFrameUpdate(diveFixture('goesLeft', left.y, left.x, 51, 1)));
    const kickedLeft = densityOf(module, 'diveSilt');
    expect(kickedLeft).toBeCloseTo(kickedRight, 12);
    module.update(toFrameUpdate(diveFixture('freeThere', left.y, left.x, 52, 0)));
    expect(densityOf(module, 'diveSilt')).toBeLessThan(kickedLeft - 0.5);
    module.dispose();
  });

  test('the stir HARD SNAPS on a scrub — no ease crosses a discontinuity', async () => {
    const { module, ctx } = await live();
    ctx.setTime(6);
    module.update(toFrameUpdate(diveFixture('charged', 8, 4, 60, 1), { scrubbed: false }));
    for (let i = 0; i < 200; i++) {
      module.update(toFrameUpdate(diveFixture('charged', 8, 4, 60, 1), { scrubbed: false, dtReal: 1 / 60 }));
    }
    const settled = densityOf(module, 'diveSilt');

    const other = await live();
    other.ctx.setTime(6);
    other.module.update(toFrameUpdate(diveFixture('free', 2, 0, 1, 0), { scrubbed: false }));
    other.module.update(toFrameUpdate(diveFixture('charged', 8, 4, 60, 1), { scrubbed: true }));
    expect(densityOf(other.module, 'diveSilt')).toBeCloseTo(settled, 12);
    other.module.dispose();
    module.dispose();
  });

  test('the silt rides the true cell over the whole walkable triangle', async () => {
    const { module } = await live();
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    const lift = siltLift(TOKENS.wispHeight);
    expect(lift).toBeGreaterThan(0);
    for (let r = 0; r <= LARGE_DEEPSEA_LAST_ROW; r += 3) {
      for (let c = 0; c <= r; c += 2) {
        module.update(toFrameUpdate(diveFixture(`silt${c}-${r}`, r, c, 0, r % 2)));
        const w = worldOf(module, 'silt');
        cellToWorld(c, r, grid, truth);
        expect(w.x).toBeCloseTo(truth[0], 12);
        expect(w.y).toBeCloseTo(lift, 12);
        expect(w.z).toBeCloseTo(truth[2], 12);
      }
    }
    module.dispose();
  });
});

describe('large-deep-sea / the canyon is in the shot, and never in the way', () => {
  async function heroCamera(): Promise<{ module: SceneModule; camera: THREE.PerspectiveCamera }> {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('deep-dive')));
    module.camera.updateMatrixWorld(true);
    return { module, camera: module.camera as THREE.PerspectiveCamera };
  }

  test('the crest ridge projects INSIDE the frame (at WALL_RUN 11 it did not)', async () => {
    const { module, camera } = await heroCamera();
    const height = makeTrenchHeight(SEED);
    const x = -(TRENCH_HALF + WALL_RUN);
    let inside = 0;
    for (const z of [-4, 4, 12]) {
      const p = new THREE.Vector3(x, height(x, z), z).project(camera);
      if (Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1 && p.z <= 1) inside++;
    }
    expect(inside).toBeGreaterThanOrEqual(2);

    const oldX = -(TRENCH_HALF + 11);
    let oldInside = 0;
    for (const z of [-4, 4, 12]) {
      const p = new THREE.Vector3(oldX, WALL_HEIGHT + BED_SOUTH, z).project(camera);
      if (Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1 && p.z <= 1) oldInside++;
    }
    expect(oldInside).toBe(0);
    module.dispose();
  });

  test('the canyon never crosses the sightline to a playfield cell, for any lane seed', async () => {
    const { module, camera } = await heroCamera();
    const eye = camera.position.clone();
    module.dispose();

    const seeds = [72, 73, 74, 75, 76, 77, 78, 79, 24, 25];
    const grid = resolveGrid(geometryOf());
    const truth: [number, number, number] = [0, 0, 0];
    let worst = Infinity;
    for (const seed of seeds) {
      const height = makeTrenchHeight(seed);
      for (let r = 0; r < LARGE_DEEPSEA_N; r++) {
        for (let c = 0; c <= r; c++) {
          cellToWorld(c, r, grid, truth);
          for (let t = 0.02; t < 1; t += 0.02) {
            const x = eye.x + (truth[0] - eye.x) * t;
            const y = eye.y * (1 - t);
            const z = eye.z + (truth[2] - eye.z) * t;
            if (Math.max(Math.abs(x), Math.abs(z)) <= TRENCH_HALF) continue;
            worst = Math.min(worst, y - height(x, z));
          }
        }
      }
    }
    expect(worst).toBeGreaterThan(1);
  });
});

describe('large-deep-sea / lifecycle', () => {
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
    expect((report.census[0]?.materials.size ?? 0) > 0).toBe(true);
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
    module.update(toFrameUpdate(fixtureNamed('deep-dive')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'abyssGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'abyssGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the camera stays a clamped aerial looking down the trench', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('abyss-start')));
    const cam = module.camera;
    expect(cam.position.y).toBeGreaterThan(8);
    expect(cam.position.x).toBeLessThan(-TRENCH_HALF);
    expect(cam.position.z).toBeLessThan(-TRENCH_HALF / 2);
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    expect(dir.y).toBeLessThan(-0.5);
    expect(dir.z).toBeGreaterThan(0);
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    module.dispose();
  });

  test('the scene is deterministic in the lane seed', async () => {
    const positions: number[][] = [];
    for (let pass = 0; pass < 2; pass++) {
      const module = createScene();
      await module.init(makeSceneContext({ geometry: geometryOf(), seed: SEED, tier: 'hero' }));
      const coral = module.scene.getObjectByName('abyssCoral') as THREE.InstancedMesh;
      const out: number[] = [];
      const m = new THREE.Matrix4();
      for (let i = 0; i < coral.count; i++) {
        coral.getMatrixAt(i, m);
        for (const v of m.elements) out.push(v);
      }
      positions.push(out);
      module.dispose();
    }
    expect(positions[1]).toEqual(positions[0] as number[]);
  });
});

describe('large-deep-sea / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'large-deep-sea (thumb, overlay on)');
    module.dispose();
  });
});
