/// <reference types="bun" />

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  MC_GOAL_POS,
  MC_MAX_POS,
  MC_MIN_POS,
  MC_START_POS,
  mountainCarDecoder,
  mountainCarHeight,
  mountainCarSlope,
} from '@sim/envs/mountaincar';
import type { EnvGeometry } from '@sim/types';

import createScene from '../../src/scenes/mountain-car/index';
import { fixtures } from '../../src/scenes/mountain-car/fixtures';
import { manifest } from '../../src/scenes/mountain-car/manifest';
import { BAND_HALF_Z, makeRidgeHeight } from '../../src/scenes/mountain-car/world';
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

const SEED = 40;

function geometryOf(): EnvGeometry {
  return mountainCarDecoder.geometry(mountainCarDecoder.initStatic(SEED), SEED);
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


describe('mountain-car / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('MountainCar-cont-v0');
    expect(manifest.slug).toBe('mountain-car');
    expect(manifest.accent).toBe('mountain-car');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('ships at least three canonical poses, all inside the env range', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      const pos = f.pose[0] ?? 0;
      expect(pos).toBeGreaterThanOrEqual(MC_MIN_POS);
      expect(pos).toBeLessThanOrEqual(MC_MAX_POS);
      expect(f.pose.length).toBe(9);
      expect(f.pose[2]).toBe(MC_GOAL_POS);
    }
  });
});

describe('mountain-car / the terrain IS the physics', () => {
  test('the height field on the track band is EXACTLY heightAt(x)', () => {
    const ridge = makeRidgeHeight(mountainCarHeight, SEED);
    for (let i = 0; i <= 400; i++) {
      const x = MC_MIN_POS + ((MC_MAX_POS - MC_MIN_POS) * i) / 400;
      for (const z of [0, BAND_HALF_Z * 0.5, -BAND_HALF_Z * 0.5, BAND_HALF_Z, -BAND_HALF_Z]) {
        expect(ridge(x, z)).toBe(mountainCarHeight(x));
      }
    }
  });

  test('past the track ends the profile is free to wander (the exact zone is the playfield)', () => {
    const ridge = makeRidgeHeight(mountainCarHeight, SEED);
    expect(ridge(MC_MAX_POS + 0.001, 0)).toBeCloseTo(mountainCarHeight(MC_MAX_POS + 0.001), 5);
    let wander = 0;
    for (const x of [-2.6, -1.9, 1.4, 2.8]) {
      wander = Math.max(wander, Math.abs(ridge(x, 0) - mountainCarHeight(x)));
    }
    expect(wander).toBeGreaterThan(0.01);
  });

  test('but the ridge really does swell away from the band (it is not a flat extrusion)', () => {
    const ridge = makeRidgeHeight(mountainCarHeight, SEED);
    let deviation = 0;
    for (let i = 0; i <= 40; i++) {
      const x = -2 + i * 0.1;
      deviation = Math.max(deviation, Math.abs(ridge(x, -2.4) - mountainCarHeight(x)));
    }
    expect(deviation).toBeGreaterThan(0.2);
  });
});

describe('mountain-car / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    expect(report.worst.get('cart') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('summit-cairn') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('trailhead') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      module.dispose();
    }
  });

  test('the cart probe tracks a swept position continuously', async () => {
    const { module } = await live();
    const world = new THREE.Vector3();
    for (let i = 0; i <= 90; i++) {
      const pos = MC_MIN_POS + ((MC_MAX_POS - MC_MIN_POS) * i) / 90;
      const fixture: SceneFixture = {
        name: `sweep${i}`,
        pose: [pos, 0.02, MC_GOAL_POS, 0, 0, 0, 0, 0, 0],
        ctrl: [2, 0, 0, 0, 0, 0],
        step: i,
      };
      module.update(toFrameUpdate(fixture));
      const probe = module.probes()[0];
      expect(probe).toBeDefined();
      if (!probe) continue;
      probe.object.updateWorldMatrix(true, false);
      world.setFromMatrixPosition(probe.object.matrixWorld);
      const truth = Math.fround(pos);
      expect(world.x).toBeCloseTo(truth, 10);
      expect(world.y).toBeCloseTo(mountainCarHeight(truth), 10);
      expect(world.z).toBeCloseTo(0, 10);
    }
    module.dispose();
  });
});

describe('mountain-car / the cart rides the tangent', () => {
  test('the tilt node maps local +x onto the analytic surface tangent', async () => {
    const { module } = await live();
    const tilt = module.scene.getObjectByName('cartTilt');
    expect(tilt).toBeDefined();
    if (!tilt) return;
    const forward = new THREE.Vector3();
    const expected = new THREE.Vector3();
    for (let i = 0; i <= 60; i++) {
      const pos = MC_MIN_POS + ((MC_MAX_POS - MC_MIN_POS) * i) / 60;
      module.update(
        toFrameUpdate({
          name: `tilt${i}`,
          pose: [pos, 0.01, MC_GOAL_POS, 0, 0, 0, 0, 0, 0],
          ctrl: [1, 0, 0, 0, 0, 0],
          step: i,
        }),
      );
      tilt.updateWorldMatrix(true, false);
      forward.set(1, 0, 0).transformDirection(tilt.matrixWorld);
      const x = Math.fround(pos);
      expected.set(1, mountainCarSlope(x), 0).normalize();
      expect(forward.x).toBeCloseTo(expected.x, 9);
      expect(forward.y).toBeCloseTo(expected.y, 9);
      expect(forward.z).toBeCloseTo(0, 9);
      expect(tilt.position.lengthSq()).toBe(0);
    }
    module.dispose();
  });
});

describe('mountain-car / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('valley-rest')));
    let count = 0;
    module.scene.traverse(() => count++);
    for (let i = 0; i < 50; i++) {
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
    module.update(toFrameUpdate(fixtureNamed('summit'), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    ctx.setTime(2);
    module.update(
      toFrameUpdate(fixtureNamed('summit'), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    module.dispose();
  });

  test('a scrub clears every in-flight beat', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(fixtureNamed('summit'), { scrubbed: false, events: [{ type: 'goal' }] }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('valley-rest'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    module.dispose();
  });

  test('teleport is a hard snap: the cart is at the respawn on the event frame', async () => {
    const { module, ctx } = await live();
    const probeOf = (): THREE.Vector3 => {
      const p = module.probes()[0];
      if (!p) throw new Error('no cart probe');
      p.object.updateWorldMatrix(true, false);
      return new THREE.Vector3().setFromMatrixPosition(p.object.matrixWorld);
    };

    ctx.setTime(1);
    const atSummit = toFrameUpdate(fixtureNamed('summit'), { scrubbed: false });
    module.update(atSummit);
    expect(probeOf().x).toBeCloseTo(MC_GOAL_POS, 10);

    ctx.setTime(1.033);
    const respawn: SceneFixture = {
      name: 'respawn',
      pose: [MC_START_POS, 0, MC_GOAL_POS, 0, 0, 0, 0, 0, 0],
      ctrl: [1, 0, 0, 0, 0, 0],
      step: (fixtureNamed('summit').step ?? 0) + 1,
    };
    module.update(
      toFrameUpdate(respawn, {
        scrubbed: false,
        prev: toFrameState(fixtureNamed('summit')),
        events: [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }],
      }),
    );
    const after = probeOf();
    expect(after.x).toBeCloseTo(MC_START_POS, 10);
    expect(after.y).toBeCloseTo(mountainCarHeight(MC_START_POS), 10);
    module.dispose();
  });

  test('the superluminal guard rejects a host-interpolated mid-teleport pose', async () => {
    const { module, ctx } = await live();
    const cartX = (): number => {
      const p = module.probes()[0];
      if (!p) throw new Error('no cart probe');
      p.object.updateWorldMatrix(true, false);
      return new THREE.Vector3().setFromMatrixPosition(p.object.matrixWorld).x;
    };
    ctx.setTime(1);
    module.update(toFrameUpdate(fixtureNamed('summit'), { scrubbed: false }));
    expect(cartX()).toBeCloseTo(MC_GOAL_POS, 10);

    const midSlide: SceneFixture = {
      name: 'mid-slide',
      pose: [0, 0.02, MC_GOAL_POS, 0, 0, 0, 0, 0, 0],
      ctrl: [1, 0, 0, 0, 0, 0],
      step: fixtureNamed('summit').step ?? 0,
    };
    module.update(toFrameUpdate(midSlide, { scrubbed: false }));
    expect(cartX()).toBeCloseTo(MC_GOAL_POS, 10);
    module.dispose();
  });
});

describe('mountain-car / lifecycle', () => {
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
    module.update(toFrameUpdate(fixtureNamed('mid-slope-climb')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'alpineGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'alpineGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });
});

describe('mountain-car / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'mountain-car (thumb, overlay on)');
    module.dispose();
  });
});
