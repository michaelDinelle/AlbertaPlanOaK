/// <reference types="bun" />
// The Boardwalk (CartPole-cont-v0) — the accuracy + lifecycle self-check.
//
// Mirrors mountain-car.test.ts's shape and adds what THIS env has to prove:
//   * the deck really does run from anchor to anchor and stop there,
//   * the surveyor's cord is at L·cos(θlimit) — and therefore that "crown below the
//     cord" is the tilt failure condition at EVERY cart position, not just at x = 0,
//   * the pole tip is `cartPoleTip` exactly, over a full sweep of both coordinates,
//   * the reset (which the env performs in the SAME step as the failure) is a hard snap
//     and the host's lerp toward it is caught by the superluminal guard,
//   * the sigh fires only from `done` — never from a pose that merely looks doomed.

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  CP_POLE_LENGTH,
  CP_TAU,
  CP_TH_LIMIT,
  CP_X_LIMIT,
  cartPoleDecoder,
  cartPoleTip,
} from '@sim/envs/cartpole';
import type { EnvGeometry } from '@sim/types';

import createScene, { envelope, unreachable } from '../../src/scenes/cart-pole/index';
import { fixtures } from '../../src/scenes/cart-pole/fixtures';
import { manifest } from '../../src/scenes/cart-pole/manifest';
import {
  BANK_INNER_X,
  BANK_INNER_Z,
  CAM_BASE,
  CORD_Z,
  DECK_Y,
  GAUGE_Z,
  TERRAIN_D,
  WATER_Y,
  gaugeStation,
  makeShoreHeight,
  poleLength,
  tiltLimit,
  tiltLineY,
  trackLimit,
} from '../../src/scenes/cart-pole/world';
import { CROWN_Y, HINGE_LOCAL_Y } from '../../src/scenes/cart-pole/actors';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameState,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 48; // the demo run's first CartPole lane

/**
 * REAL RECORDED ROWS — frontend-demo-16env-20260803 / CartPole-cont-v0 / seed 48,
 * `actions.bin` rows 16-23 as [x, xd, th, thd]. Row 23 is the last live row of that
 * episode; row 24 is the all-zero reset carrying done = 1, which is what the host lerps
 * toward and what the superluminal guard has to refuse to draw.
 */
const REAL_ROWS: readonly (readonly number[])[] = [
  [-0.019524, -0.390904, 0.029527, 0.600096],
  [-0.027342, -0.586426, 0.041529, 0.901931],
  [-0.03907, -0.782086, 0.059567, 1.207373],
  [-0.054712, -0.587782, 0.083715, 0.933936],
  [-0.066468, -0.783927, 0.102394, 1.251707],
  [-0.082146, -0.980201, 0.127428, 1.574627],
  [-0.10175036, -1.17659116, 0.15892021, 1.90418649],
  [-0.12528218, -1.37303424, 0.19700395, 2.24166417],
];
const ROW22 = REAL_ROWS[6] as readonly number[];
const ROW23 = REAL_ROWS[7] as readonly number[];

function geometryOf(): EnvGeometry {
  return cartPoleDecoder.geometry(cartPoleDecoder.initStatic(SEED), SEED);
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

function worldOf(module: SceneModule, name: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === name);
  if (!probe) throw new Error(`probe "${name}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

// ---------------------------------------------------------------------------

describe('cart-pole / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('CartPole-cont-v0');
    expect(manifest.slug).toBe('cart-pole');
    expect(manifest.accent).toBe('cart-pole');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a physically legal, non-trivial pose', () => {
    const p = manifest.thumbFrame.pose;
    expect(Math.abs(p[0] ?? 0)).toBeLessThan(CP_X_LIMIT);
    expect(Math.abs(p[2] ?? 0)).toBeLessThan(CP_TH_LIMIT);
    // not a parked cart: it is leaning and moving
    expect(Math.abs(p[2] ?? 0)).toBeGreaterThan(0.02);
    expect(Math.abs(p[1] ?? 0)).toBeGreaterThan(0.1);
  });

  test('ships at least three canonical poses, all inside the env bounds', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      expect(Math.abs(f.pose[0] ?? 0)).toBeLessThanOrEqual(CP_X_LIMIT);
      expect(Math.abs(f.pose[2] ?? 0)).toBeLessThanOrEqual(CP_TH_LIMIT);
      // pose slots 4..8 are unused by this env and must stay zero
      for (let i = 4; i < 9; i++) expect(f.pose[i] ?? 0).toBe(0);
    }
    // the reset state the env snaps back to on every failure is one of them
    expect(fixtures.some((f) => f.pose.every((v) => v === 0))).toBe(true);
    // and so is the exact tilt bound
    expect(fixtures.some((f) => Math.abs(f.pose[2] ?? 0) === CP_TH_LIMIT)).toBe(true);
  });
});

describe('cart-pole / the env numbers are READ, never re-derived', () => {
  test('the deck length, pole length and cone come out of the anchors', () => {
    const g = geometryOf();
    expect(trackLimit(g)).toBe(CP_X_LIMIT);
    expect(poleLength(g)).toBe(CP_POLE_LENGTH);
    expect(tiltLimit(g)).toBeCloseTo(CP_TH_LIMIT, 12);
    expect(tiltLineY(g)).toBeCloseTo(CP_POLE_LENGTH * Math.cos(CP_TH_LIMIT), 12);
  });

  test('the cord height IS the tilt failure condition, at EVERY cart position', () => {
    // |th| > θlimit  ⟺  tip.y < L·cos(θlimit), for all x — this is the whole reason
    // one horizontal line can state the bound honestly while the trolley travels.
    const g = geometryOf();
    const cord = tiltLineY(g);
    const out = new Float32Array(2);
    for (const x of [-CP_X_LIMIT, -1.19, -0.34, 0, 0.5, 1.19, CP_X_LIMIT]) {
      for (let i = -40; i <= 40; i++) {
        const th = (i / 40) * (CP_TH_LIMIT * 1.6);
        // `cartPoleTip` writes f32, so skip the samples sitting ON the bound where a
        // 1e-8 storage wobble decides the strict inequality
        if (Math.abs(Math.abs(th) - CP_TH_LIMIT) < 1e-5) continue;
        const tip = cartPoleTip(x, th, out);
        const fails = Math.abs(th) > CP_TH_LIMIT;
        const below = (tip[1] as number) < cord;
        expect(below).toBe(fails);
      }
    }
  });

  test('the carved arcs pass exactly through the failArc anchors', () => {
    const g = geometryOf();
    const arm = poleLength(g);
    const half = tiltLimit(g);
    const left = g.classic?.anchors.failArcLeft;
    const right = g.classic?.anchors.failArcRight;
    expect(left).toBeDefined();
    expect(right).toBeDefined();
    if (!left || !right) return;
    expect(arm * Math.sin(half)).toBeCloseTo(right[0], 12);
    expect(arm * Math.cos(half)).toBeCloseTo(right[1], 12);
    expect(-arm * Math.sin(half)).toBeCloseTo(left[0], 12);
  });
});

describe('cart-pole / the lake never breaks the surface under the crossing', () => {
  test('the bed stays submerged over the whole open-water corridor', () => {
    const shore = makeShoreHeight(SEED);
    for (let i = 0; i <= 120; i++) {
      const x = -8 + (i * 16) / 120;
      for (const z of [-BANK_INNER_Z + 0.4, -1, -0.46, 0, 0.46, 1, BANK_INNER_Z - 0.4]) {
        if (Math.abs(x) > BANK_INNER_X - 0.05) continue;
        expect(shore(x, z)).toBeLessThan(WATER_Y - 0.2);
      }
    }
  });

  test('the near spit lifts out of the water but stays BELOW the deck', () => {
    // the mid-ground of this frame is water and a low reed bank; a shore that reaches
    // deck height becomes a green wall directly behind the sapling
    const shore = makeShoreHeight(SEED);
    let highest = -Infinity;
    let lowest = Infinity;
    for (let i = 0; i <= 40; i++) {
      const x = -9 + (i * 18) / 40;
      const h = shore(x, -8);
      highest = Math.max(highest, h);
      lowest = Math.min(lowest, h);
    }
    expect(highest).toBeGreaterThan(WATER_Y + 0.15);
    // never up into the sapling's own band: the crest stays around plank height
    expect(highest).toBeLessThan(DECK_Y + 0.25);
    expect(lowest).toBeLessThan(highest); // a wavy shoreline, not a step
  });

  test('the distant ridge climbs above the eye, so no mesh edge cuts the sky', () => {
    // the terrain's far boundary sits at z = −TERRAIN_D/2; the ridge must have passed
    // the camera's own height (1.16) by then or the ground ends in a hard line
    const shore = makeShoreHeight(SEED);
    const edge = -TERRAIN_D / 2 + 1;
    let lowestOnRidge = Infinity;
    for (let i = 0; i <= 40; i++) {
      const x = -12 + (i * 24) / 40;
      lowestOnRidge = Math.min(lowestOnRidge, shore(x, edge));
    }
    expect(lowestOnRidge).toBeGreaterThan(1.16);
  });
});

describe('cart-pole / alignment', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    // probe roots carry POSITION ONLY — they should be exact, not merely within 1e-3
    expect(report.worst.get('trolley') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('pole-tip') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('deck-end-left') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('deck-end-right') ?? 1).toBeLessThan(1e-9);
    expect(report.worst.get('tilt-gauge') ?? 1).toBeLessThan(1e-9);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      expect(report.worst.get('pole-tip') ?? 1).toBeLessThan(1e-9);
      module.dispose();
    }
  });

  test('the tip tracks cartPoleTip over a full sweep of BOTH coordinates', async () => {
    const { module } = await live();
    const out = new Float32Array(2);
    for (let i = 0; i <= 24; i++) {
      const x = -CP_X_LIMIT + (2 * CP_X_LIMIT * i) / 24;
      for (let j = 0; j <= 16; j++) {
        const th = -CP_TH_LIMIT + (2 * CP_TH_LIMIT * j) / 16;
        module.update(
          toFrameUpdate({ name: `sweep${i}_${j}`, pose: [x, 0, th, 0, 0, 0, 0, 0, 0], step: i * 17 + j }),
        );
        // the pose arrives as f32, so ground truth is the f32 round-trip of the state;
        // `cartPoleTip` then STORES its answer in f32 too, hence the 1e-6 band here
        // (the f64 formula itself is checked exactly by the probe assertions above)
        const fx = Math.fround(x);
        const fth = Math.fround(th);
        const truth = cartPoleTip(fx, fth, out);
        const tip = worldOf(module, 'pole-tip');
        expect(tip.x).toBeCloseTo(truth[0] as number, 6);
        expect(tip.y).toBeCloseTo(truth[1] as number, 6);
        expect(tip.z).toBeCloseTo(0, 9);
        expect(tip.x).toBeCloseTo(fx + CP_POLE_LENGTH * Math.sin(fth), 12);
        expect(tip.y).toBeCloseTo(CP_POLE_LENGTH * Math.cos(fth), 12);
        const cart = worldOf(module, 'trolley');
        expect(cart.x).toBeCloseTo(fx, 9);
        expect(cart.y).toBeCloseTo(DECK_Y, 12);
        expect(cart.z).toBeCloseTo(0, 12);
      }
    }
    module.dispose();
  });

  test('the hinge sits back on the env plane and the tilt node is a PURE rotation', async () => {
    const { module } = await live();
    const hinge = module.scene.getObjectByName('poleHinge');
    const tilt = module.scene.getObjectByName('poleTilt');
    expect(hinge).toBeDefined();
    expect(tilt).toBeDefined();
    if (!hinge || !tilt) return;
    // the planter's soil surface IS the pole pivot: DECK_Y + HINGE_LOCAL_Y === 0
    expect(DECK_Y + HINGE_LOCAL_Y).toBeCloseTo(0, 12);
    const world = new THREE.Vector3();
    for (const f of fixtures) {
      module.update(toFrameUpdate(f));
      hinge.updateWorldMatrix(true, false);
      world.setFromMatrixPosition(hinge.matrixWorld);
      expect(world.x).toBeCloseTo(Math.fround(f.pose[0] ?? 0), 9);
      expect(world.y).toBeCloseTo(0, 12);
      // a rotation only — never a translation, or the tip probe would drift
      expect(tilt.position.lengthSq()).toBe(0);
      expect(tilt.scale.x).toBe(1);
      expect(tilt.rotation.z).toBeCloseTo(-Math.fround(f.pose[2] ?? 0), 12);
    }
    module.dispose();
  });

  test('the fall beat lives on the crown, so it can never move the tip', async () => {
    const { module, ctx } = await live();
    const crown = module.scene.getObjectByName('saplingCrown');
    expect(crown).toBeDefined();
    if (!crown) return;
    expect(crown.position.y).toBeCloseTo(CROWN_Y, 12);

    const at = fixtureNamed('tilt-limit-left');
    ctx.setTime(10);
    module.update(toFrameUpdate(at, { scrubbed: false }));
    const quietTip = worldOf(module, 'pole-tip').clone();
    const quietCrown = crown.scale.x;

    // the same pose, one frame into a fall beat
    ctx.setTime(10.02);
    module.update(
      toFrameUpdate(at, { scrubbed: false, events: [{ type: 'done' }, { type: 'teleport' }] }),
    );
    // the crown really did react…
    expect(crown.scale.x).toBeLessThan(quietCrown);
    // …and the tip did not move a micron
    const beatTip = worldOf(module, 'pole-tip');
    expect(beatTip.distanceTo(quietTip)).toBeLessThan(1e-9);
    module.dispose();
  });
});

// ---------------------------------------------------------------------------
// THE READ ITSELF: what a viewer actually sees vs what the env actually says.
// These are the assertions the scene's headline claim lives or dies on.
// ---------------------------------------------------------------------------

/** every vertex of a mesh, in world space */
function worldVertices(mesh: THREE.Mesh): THREE.Vector3[] {
  mesh.updateWorldMatrix(true, false);
  const pos = mesh.geometry.getAttribute('position');
  const out: THREE.Vector3[] = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.set(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(mesh.matrixWorld);
    out.push(v.clone());
  }
  return out;
}

/** the highest point of anything a viewer would call "the sapling" */
function saplingTop(module: SceneModule): number {
  let top = -Infinity;
  for (const name of ['saplingTrunk', 'saplingLeaves']) {
    const mesh = module.scene.getObjectByName(name) as THREE.Mesh | undefined;
    if (!mesh) throw new Error(`mesh "${name}" missing`);
    for (const v of worldVertices(mesh)) if (v.y > top) top = v.y;
  }
  return top;
}

describe('cart-pole / the drawn crown states the TIP bound, not a bigger one', () => {
  test('the visible top of the sapling IS the pole tip, at every tier', async () => {
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module } = await live(tier);
      for (let i = -8; i <= 8; i++) {
        const th = (i / 8) * CP_TH_LIMIT;
        module.update(toFrameUpdate({ name: `t${i}`, pose: [0, 0, th, 0, 0, 0, 0, 0, 0] }));
        const tip = worldOf(module, 'pole-tip');
        // exactly the tip: the terminal bud's apex is the topmost vertex on the whole
        // sapling, so "the drawn top" and "cartPoleTip" are the same point
        expect(saplingTop(module)).toBeCloseTo(tip.y, 9);
      }
      module.dispose();
    }
  });

  test('the drawn top crosses the cord at EXACTLY the env bound (was 35.1°)', async () => {
    const { module } = await live();
    const cord = tiltLineY(geometryOf());
    const above = (th: number): boolean => {
      module.update(toFrameUpdate({ name: 'x', pose: [0, 0, th, 0, 0, 0, 0, 0, 0] }));
      return saplingTop(module) > cord;
    };
    let lo = 0;
    let hi = CP_TH_LIMIT * 3;
    for (let i = 0; i < 60; i++) {
      const mid = 0.5 * (lo + hi);
      if (above(mid)) lo = mid;
      else hi = mid;
    }
    expect((lo + hi) * 0.5).toBeCloseTo(CP_TH_LIMIT, 6);
    module.dispose();
  });

  test('nothing green may reach the cord, even under the crown’s full droop', async () => {
    // The crown carries lean (±0.12) and the shed beat (±0.18), which peak TOGETHER at a
    // fall — so the worst case is exactly the frame a viewer is reading. Measured in the
    // hinge frame, where a rotation about the crown cannot be argued away.
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module } = await live(tier);
      module.update(toFrameUpdate({ name: 'up', pose: [0, 0, 0, 0, 0, 0, 0, 0, 0] }));
      const crown = module.scene.getObjectByName('saplingCrown') as THREE.Object3D;
      const leaves = module.scene.getObjectByName('saplingLeaves') as THREE.Mesh;
      const toCrown = new THREE.Matrix4();
      crown.updateWorldMatrix(true, true);
      leaves.updateWorldMatrix(true, false);
      toCrown.copy(crown.matrixWorld).invert().multiply(leaves.matrixWorld);
      const pos = leaves.geometry.getAttribute('position');
      const p = new THREE.Vector3();
      let worst = 0;
      for (let s = -12; s <= 12; s++) {
        const phi = (s / 12) * 0.3;
        const cs = Math.cos(phi);
        const sn = Math.sin(phi);
        for (let i = 0; i < pos.count; i++) {
          p.set(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(toCrown);
          const rx = p.x * cs - p.y * sn;
          const ry = p.x * sn + p.y * cs;
          worst = Math.max(worst, Math.hypot(rx, ry + CROWN_Y, p.z));
        }
      }
      // world y never exceeds the distance from the hinge, so this bounds the read at
      // every tilt at once
      expect(worst).toBeLessThan(tiltLineY(geometryOf()));
      module.dispose();
    }
  });

  test('the cord is at the anchor height, in the pole’s own plane to 0.4 px', async () => {
    const { module } = await live();
    const cord = module.scene.getObjectByName('tiltCord') as THREE.Mesh;
    const verts = worldVertices(cord);
    const cordY = tiltLineY(geometryOf());
    // the line itself: the widest span in x, centred on the anchor height
    let top = -Infinity;
    let bottom = Infinity;
    for (const v of verts) {
      if (Math.abs(v.y - cordY) < 0.02 && Math.abs(v.z - CORD_Z) < 0.02) {
        top = Math.max(top, v.y);
        bottom = Math.min(bottom, v.y);
      }
    }
    // f32 vertex storage, so 1e-7 — five orders of magnitude under a pixel (0.0037/px)
    expect((top + bottom) * 0.5).toBeCloseTo(cordY, 7);
    // and it lives essentially in the tip's own plane, so the drawn line is the drawn
    // bound (at z = −0.34 the same cord projected 3.4 px high — 55% of the whole band)
    expect(Math.abs(CORD_Z)).toBeLessThan(0.05);
    module.dispose();
  });
});

describe('cart-pole / the carved cone is a SIGHTING instrument', () => {
  test('the station is the eye→pivot ray at the gauge’s own depth', () => {
    const s = gaugeStation(CAM_BASE, GAUGE_Z);
    expect(s.z).toBe(GAUGE_Z);
    // colinear with the camera and the pivot: station = C + t(0 − C)
    const t = s.scale;
    expect(s.x).toBeCloseTo(CAM_BASE[0] * (1 - t), 12);
    expect(s.y).toBeCloseTo(CAM_BASE[1] * (1 - t), 12);
    expect(t).toBeGreaterThan(1); // a gauge set BACK is scaled UP
  });

  test('the drawn cone lands on the true failure cone, to a hundredth of a pixel', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('upright-reset')));
    const camera = module.camera;
    camera.aspect = 16 / 9;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    const project = (x: number, y: number, z: number): [number, number] => {
      const v = new THREE.Vector3(x, y, z).project(camera);
      return [(v.x * 0.5 + 0.5) * 1600, (1 - (v.y * 0.5 + 0.5)) * 900];
    };
    const gauge = module.probes().find((p) => p.name === 'tilt-gauge');
    if (!gauge) throw new Error('gauge probe missing');
    gauge.object.updateWorldMatrix(true, false);
    const root = new THREE.Vector3().setFromMatrixPosition(gauge.object.matrixWorld);
    const scale = new THREE.Vector3().setFromMatrixScale(gauge.object.matrixWorld);

    // the apex must project onto the pivot the pole actually turns about
    const pivot = project(0, 0, 0);
    const apex = project(root.x, root.y, root.z);
    expect(apex[0]).toBeCloseTo(pivot[0], 2);
    expect(apex[1]).toBeCloseTo(pivot[1], 2);

    // …and every point of the carved cone onto the same point of the real cone
    const anchors = geometryOf().classic?.anchors;
    if (!anchors) throw new Error('anchors missing');
    for (const name of ['failArcLeft', 'failArcRight', 'poleTipRest']) {
      const a = anchors[name];
      if (!a) throw new Error(`anchor ${name} missing`);
      const truth = project(a[0], a[1], 0);
      const drawn = project(root.x + scale.x * a[0], root.y + scale.y * a[1], root.z);
      expect(drawn[0]).toBeCloseTo(truth[0], 2);
      expect(drawn[1]).toBeCloseTo(truth[1], 2);
    }
    module.dispose();
  });

  test('the gauge still stands ON the deck under its sighting transform', async () => {
    const { module } = await live();
    const mesh = module.scene.getObjectByName('tiltGauge') as THREE.Mesh;
    let lowest = Infinity;
    for (const v of worldVertices(mesh)) lowest = Math.min(lowest, v.y);
    // its foot reaches the planks (a floating instrument reads as a wire arch)
    expect(lowest).toBeLessThan(DECK_Y);
    expect(lowest).toBeGreaterThan(DECK_Y - 0.3);
    module.dispose();
  });
});

describe('cart-pole / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('upright-reset')));
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

  test('the sigh fires ONLY from frame.events — never from a doomed-looking pose', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    expect(pool).not.toBeNull();
    if (!pool) return;

    // sitting EXACTLY on the tilt bound with no event: nothing may fire
    ctx.setTime(1);
    module.update(toFrameUpdate(fixtureNamed('tilt-limit-left'), { scrubbed: false }));
    expect(pool.pulse.sigh).toBe(0);
    // and CartPole has no +1 event at all, so a burst/ripple must never happen
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    // the same pose with the decoder's real event batch: exactly one sigh, still no burst
    ctx.setTime(2);
    module.update(
      toFrameUpdate(fixtureNamed('upright-reset'), {
        scrubbed: false,
        events: [{ type: 'done' }, { type: 'teleport' }],
      }),
    );
    expect(pool.pulse.sigh).toBeGreaterThan(0);
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);
    module.dispose();
  });

  test('the decoder really does emit done+teleport together and nothing else', () => {
    const prev = cartPoleDecoder.initFrame(SEED, cartPoleDecoder.initStatic(SEED));
    const raw = new Float32Array(16);
    raw[6] = 1; // done, with the all-zero reset pose the kernel writes in the same step
    const next = cartPoleDecoder.decodeFrame(0, raw, cartPoleDecoder.initStatic(SEED), prev);
    const events = cartPoleDecoder.deriveEvents(prev, next, cartPoleDecoder.initStatic(SEED));
    expect(events.map((e) => e.type).sort()).toEqual(['done', 'teleport']);
    expect(events.some((e) => e.type === 'goal')).toBe(false);
  });

  test('a scrub clears every in-flight beat', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    ctx.setTime(1);
    module.update(
      toFrameUpdate(fixtureNamed('upright-reset'), { scrubbed: false, events: [{ type: 'done' }] }),
    );
    expect(pool.pulse.sigh).toBeGreaterThan(0);
    ctx.setTime(1.01);
    module.update(toFrameUpdate(fixtureNamed('balancing-right'), { scrubbed: true }));
    expect(pool.pulse.sigh).toBe(0);
    expect(pool.activeRipples).toBe(0);
    module.dispose();
  });

  test('teleport is a hard snap: the trolley is upright at the origin on the event frame', async () => {
    const { module, ctx } = await live();
    ctx.setTime(1);
    const falling = fixtureNamed('tilt-limit-left');
    module.update(toFrameUpdate(falling, { scrubbed: false }));
    expect(worldOf(module, 'trolley').x).toBeCloseTo(Math.fround(falling.pose[0] ?? 0), 9);

    // the recorded failing row IS the respawn: all four coordinates zero with done = 1
    ctx.setTime(1.02);
    const respawn: SceneFixture = {
      name: 'respawn',
      pose: [0, 0, 0, 0, 0, 0, 0, 0, 0],
      ctrl: [0, 0, 0, 0, 0, 0],
      step: (falling.step ?? 0) + 1,
    };
    module.update(
      toFrameUpdate(respawn, {
        scrubbed: false,
        prev: toFrameState(falling),
        events: [{ type: 'done' }, { type: 'teleport' }],
      }),
    );
    const cart = worldOf(module, 'trolley');
    const tip = worldOf(module, 'pole-tip');
    expect(cart.x).toBeCloseTo(0, 12);
    expect(cart.y).toBeCloseTo(DECK_Y, 12);
    expect(tip.x).toBeCloseTo(0, 12);
    expect(tip.y).toBeCloseTo(CP_POLE_LENGTH, 12);
    module.dispose();
  });

  test('the superluminal guard holds the REAL recorded slide (seed 48, rows 22-24)', async () => {
    // The fabricated x = 1.1 pose this test used to fly only exercised the CART channel —
    // the one channel this agent never uses (|x| max 1.158 over 20k steps) — so it passed
    // while 76% of real slide frames drew an un-falling pole. These are the actual rows.
    const { module, ctx } = await live();
    ctx.setTime(1);
    const lerp = (a: readonly number[], b: readonly number[], t: number): number[] =>
      a.map((v, i) => v + ((b[i] ?? 0) - v) * t);

    // frame A: playback position 22.5 — an ordinary interpolated frame, drawn as given
    const a = lerp(ROW22, ROW23, 0.5);
    module.update(toFrameUpdate({ name: 'preA', pose: [...(a as number[]), 0, 0, 0, 0, 0], step: 21 }));
    module.update(
      toFrameUpdate({ name: 'preB', pose: [...(a as number[]), 0, 0, 0, 0, 0], step: 21 }, { scrubbed: false }),
    );
    const drawnA = worldOf(module, 'pole-tip');
    expect(drawnA.x).toBeCloseTo(Math.fround(a[0] ?? 0) + Math.sin(Math.fround(a[2] ?? 0)), 6);

    // frame B: position 23.333 — the host is a third of the way to the ALL-ZERO reset
    // row, so it hands over row 23 scaled by 2/3 (th 0.19700 -> 0.13134) and the event
    // has not arrived yet. The pole must NOT un-fall.
    const slide = ROW23.map((v) => v * (2 / 3));
    module.update(
      toFrameUpdate({ name: 'slide', pose: [...slide, 0, 0, 0, 0, 0], step: 22 }, { scrubbed: false }),
    );
    const heldTilt = module.scene.getObjectByName('poleTilt') as THREE.Object3D;
    const heldTh = -heldTilt.rotation.z;
    expect(heldTh).toBeCloseTo(Math.fround(a[2] ?? 0), 6); // frozen on the last legal pose
    expect(Math.abs(heldTh - (slide[2] ?? 0))).toBeGreaterThan(0.04); // ≥ 19% of the band

    // frame C: the event lands with the recorded reset row — a hard cut
    module.update(
      toFrameUpdate(
        { name: 'reset', pose: [0, 0, 0, 0, 0, 0, 0, 0, 0], step: 23 },
        { scrubbed: false, events: [{ type: 'done' }, { type: 'teleport' }] },
      ),
    );
    expect(worldOf(module, 'trolley').x).toBeCloseTo(0, 12);
    expect(worldOf(module, 'pole-tip').y).toBeCloseTo(CP_POLE_LENGTH, 12);
    module.dispose();
  });

  test('the guard never holds a LEGAL step — on the real recorded rows', async () => {
    const { module, ctx } = await live();
    ctx.setTime(1);
    // Rows 16-23 of the recording, played one per frame: every one is a legal Euler step
    // of a pole falling hard (thd climbs 0.60 -> 2.24), which is exactly the regime a
    // magnitude-only guard confuses with a reset.
    for (let i = 0; i < REAL_ROWS.length; i++) {
      const r = REAL_ROWS[i] as readonly number[];
      module.update(
        toFrameUpdate({ name: `row${i}`, pose: [...r, 0, 0, 0, 0, 0], step: 16 + i }, { scrubbed: false }),
      );
      expect(worldOf(module, 'trolley').x).toBeCloseTo(Math.fround(r[0] ?? 0), 6);
      const tilt = module.scene.getObjectByName('poleTilt') as THREE.Object3D;
      expect(-tilt.rotation.z).toBeCloseTo(Math.fround(r[2] ?? 0), 6);
    }
    module.dispose();
  });

  test('the guard never holds a LEGAL step (synthetic hard accelerations)', async () => {
    const { module, ctx } = await live();
    ctx.setTime(1);
    // walk the exact integrator: th' = th + tau*thd, x' = x + tau*xd
    let x = -0.4;
    let xd = 1.1;
    let th = -0.06;
    let thd = 1.9;
    for (let i = 0; i < 60; i++) {
      module.update(
        toFrameUpdate({ name: `euler${i}`, pose: [x, xd, th, thd, 0, 0, 0, 0, 0], step: 100 + i }, { scrubbed: false }),
      );
      const cart = worldOf(module, 'trolley');
      expect(cart.x).toBeCloseTo(Math.fround(x), 6);
      x += 0.02 * xd;
      th += 0.02 * thd;
      // a hard, legal acceleration on every step
      xd += i % 2 === 0 ? 0.19 : -0.19;
      thd += i % 3 === 0 ? -0.34 : 0.17;
    }
    module.dispose();
  });

  test('unreachable() is SIGNED: a fast fall widens the band only where the pole is going', () => {
    // the real slide: th ran 0.17796 -> 0.13134 (backwards) while thd read +1.49
    expect(unreachable(0.13134 - 0.17796, 2.073, 1.494, 0.4, 0, 2)).toBe(true);
    // the same |Δth| in the direction thd points is perfectly legal
    expect(unreachable(0.17796 - 0.13134, 2.073, 1.494, 0.4, 0, 2)).toBe(false);
    // the OLD symmetric rule: |Δth| ≤ τ·seg·(|thd| + acc·seg/2) — it accepted the slide
    const symmetric = CP_TAU * 2 * (Math.max(2.073, 1.494) + 0.4) + 0.003;
    expect(Math.abs(0.13134 - 0.17796)).toBeLessThan(symmetric);
    // a standing pose with zero velocity may not jump at all
    expect(unreachable(0.05, 0, 0, 0.4, 0, 1)).toBe(true);
    // …and a still pose is always reachable
    expect(unreachable(0, 2.4, 2.4, 0.4, 0, 1)).toBe(false);
    // the band grows with the segment count, never shrinks
    expect(unreachable(0.09, 2.4, 2.3, 0.4, 0, 1)).toBe(true);
    expect(unreachable(0.09, 2.4, 2.3, 0.4, 0, 3)).toBe(false);
  });
});

describe('cart-pole / envelopes + lighting', () => {
  test('every flourish envelope ends at exactly zero (no one-frame pop)', () => {
    expect(envelope(-1, 0.5, 3)).toBe(0);
    expect(envelope(0, 0.5, 3)).toBeCloseTo(1, 12);
    expect(envelope(0.5, 0.5, 3)).toBe(0);
    expect(envelope(0.51, 0.5, 3)).toBe(0);
    for (let i = 1; i < 40; i++) {
      const a = envelope((i - 1) / 80, 0.5, 3);
      const b = envelope(i / 80, 0.5, 3);
      expect(b).toBeLessThanOrEqual(a + 1e-12);
    }
  });

  test('the dense +1 IS the light: a safe pose is warmer than a doomed one', async () => {
    const { module } = await live();
    const warm = module.scene.getObjectByName('oakPoint0') as THREE.PointLight | undefined;
    expect(warm).toBeDefined();
    if (!warm) return;
    module.update(toFrameUpdate(fixtureNamed('upright-reset'), { scrubbed: false }));
    const safe = warm.intensity;
    module.update(toFrameUpdate(fixtureNamed('tilt-limit-left'), { scrubbed: false }));
    const doomed = warm.intensity;
    expect(safe).toBeGreaterThan(doomed);
    // and the warm light rides the crown, not the origin
    expect(warm.position.y).toBeGreaterThan(0.5);
    module.dispose();
  });

  test('the grass wind leans with the sapling (cosmetic, but signed by the state)', async () => {
    const { module } = await live();
    const windOf = (): number => {
      let amp = Number.NaN;
      module.scene.traverse((o) => {
        if (o.name !== 'bankGrass:mesh0') return;
        const mat = (o as THREE.Mesh).material as THREE.Material;
        const oak = (mat as unknown as { oak?: { uniforms: { uWindAmp: { value: number } } } }).oak;
        if (oak) amp = oak.uniforms.uWindAmp.value;
      });
      return amp;
    };
    module.update(toFrameUpdate({ name: 'lean-right', pose: [0, 0, CP_TH_LIMIT, 0, 0, 0, 0, 0, 0] }));
    const right = windOf();
    module.update(toFrameUpdate({ name: 'lean-left', pose: [0, 0, -CP_TH_LIMIT, 0, 0, 0, 0, 0, 0] }));
    const left = windOf();
    expect(Number.isFinite(right)).toBe(true);
    expect(right).toBeGreaterThan(left);
    module.dispose();
  });
});

/**
 * Reproduce `renderer.info.render` off the live scene graph: one draw per visible
 * renderable, one MORE for each shadow caster (the shadow pass) — and, the part this
 * function used to drop, that caster's triangles a SECOND time, because the shadow pass
 * rasterizes them again. Counting them once under-reported the hero frame by 25% (186.1k
 * against a real 233.1k) and would have kept passing while the shipped HUD showed the
 * over-budget warning.
 *
 * `tier` matters: kit meshes force castShadow off at thumb, and the host skips the
 * shadow pass entirely there.
 *
 * Cross-checked against a live HUD reading of "45/70 draws · 215.6k tris" on the
 * intermediate build whose figure here was 45 / 215,648.
 */
function measure(scene: THREE.Scene, tier: QualityTier): { draws: number; triangles: number } {
  let draws = 0;
  let triangles = 0;
  const shadows = tier !== 'thumb';
  scene.traverse((o) => {
    const r = o as THREE.Object3D & {
      isMesh?: boolean;
      isPoints?: boolean;
      isLine?: boolean;
      isLineSegments?: boolean;
      isInstancedMesh?: boolean;
      count?: number;
      geometry?: THREE.BufferGeometry;
      castShadow?: boolean;
    };
    if (!(r.isMesh || r.isPoints || r.isLine || r.isLineSegments)) return;
    for (let p: THREE.Object3D | null = o; p; p = p.parent) if (!p.visible) return;
    const instances = r.isInstancedMesh ? (r.count ?? 0) : 1;
    if (instances === 0) return;
    const casts = !!r.castShadow && shadows;
    draws++;
    if (casts) draws++;
    const g = r.geometry;
    if (g && r.isMesh) {
      const n = g.index ? g.index.count : (g.getAttribute('position')?.count ?? 0);
      const tris = (n / 3) * instances;
      triangles += tris;
      if (casts) triangles += tris; // the shadow pass draws them AGAIN
    }
  });
  return { draws, triangles: Math.round(triangles) };
}

describe('cart-pole / budget', () => {
  test('the declared budget is real at every tier, and thumb fits the wall-view cap', async () => {
    const caps = {
      hero: { draws: manifest.budget.drawCalls, tris: manifest.budget.triangles },
      medium: { draws: manifest.budget.drawCalls, tris: manifest.budget.triangles },
      thumb: { draws: 40, tris: 50_000 }, // STYLE.md's thumb contract
    } as const;
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module } = await live(tier);
      for (const f of fixtures) module.update(toFrameUpdate(f));
      const m = measure(module.scene, tier);
      const cap = caps[tier];
      if (m.draws > cap.draws || m.triangles > cap.tris) {
        throw new Error(`${tier}: ${m.draws} draws / ${m.triangles} tris exceeds ${cap.draws} / ${cap.tris}`);
      }
      // and the overlay must not blow the budget either — review shots run with it on
      module.setDebug(true);
      module.update(toFrameUpdate(fixtures[1] as SceneFixture));
      expect(measure(module.scene, tier).draws).toBeLessThanOrEqual(cap.draws);
      module.dispose();
    }
  });

  test('the declared budget carries REAL headroom over the measured frame', async () => {
    // A budget the frame sits at 97% of is a tripwire, not a budget: the HUD flags the
    // manifest on any drift. These are the numbers quoted in manifest.ts, ±0 — if a
    // change moves them, the comment is stale and this fails.
    const expected = {
      hero: { draws: 43, tris: 188_628 },
      medium: { draws: 40, tris: 63_238 },
      thumb: { draws: 24, tris: 16_846 },
    } as const;
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const { module } = await live(tier);
      for (const f of fixtures) module.update(toFrameUpdate(f));
      const m = measure(module.scene, tier);
      expect(`${tier}:${m.draws}/${m.triangles}`).toBe(
        `${tier}:${expected[tier].draws}/${expected[tier].tris}`,
      );
      module.dispose();
    }
    // ≥15% headroom on both axes at the tier the budget is declared for
    expect(expected.hero.draws / manifest.budget.drawCalls).toBeLessThan(0.85);
    expect(expected.hero.tris / manifest.budget.triangles).toBeLessThan(0.85);
  });
});

describe('cart-pole / lifecycle', () => {
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
    module.update(toFrameUpdate(fixtureNamed('balancing-right')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'boardwalkGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'boardwalkGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the gauge sits on its sighting station and never moves', async () => {
    const { module } = await live();
    const station = gaugeStation(CAM_BASE, GAUGE_Z);
    module.update(toFrameUpdate(fixtureNamed('far-right-drift')));
    const gauge = worldOf(module, 'tilt-gauge');
    expect(gauge.x).toBeCloseTo(station.x, 12);
    expect(gauge.y).toBeCloseTo(station.y, 12);
    expect(gauge.z).toBeCloseTo(GAUGE_Z, 12);
    // and it stays put while the trolley travels — the anchors do not move
    module.update(toFrameUpdate(fixtureNamed('near-track-end')));
    expect(worldOf(module, 'tilt-gauge').x).toBeCloseTo(station.x, 12);
    module.dispose();
  });
});
