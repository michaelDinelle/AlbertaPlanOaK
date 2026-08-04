/// <reference types="bun" />
// The Willow Swing (Acrobot-cont-v0) — the accuracy + lifecycle self-check.
//
// This mirrors four-rooms.test.ts's shape and adds what a TWO-BONE CHAIN has to prove:
// the elbow and the lantern land on the decoder's own `acrobotPoints` over a dense sweep
// of both angles, the links are exactly 1.0 long at every pose, the string of lights sits
// exactly on the `successLine` anchor (so "lantern above the bulbs" and "the env scores"
// are the same statement), the elbow torque glow is on the correct side of the joint, and
// — the one thing a screenshot can never catch — the arc guard turns the host's linear
// lerp across a ±π wrap and across the post-success respawn into physically reachable
// motion instead of a whip.

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  AB_DT,
  AB_L1,
  AB_L2,
  AB_SUCCESS_HEIGHT,
  acrobotDecoder,
  acrobotPoints,
  acrobotTipHeight,
} from '@sim/envs/acrobot';
import type { EnvGeometry } from '@sim/types';

import createScene, { acrobotTipSpeed, arcCorrupt, envelope, torqueOf, wrapPi } from '../../src/scenes/acrobot/index';
import { fixtures, th2ForSuccess } from '../../src/scenes/acrobot/fixtures';
import { manifest } from '../../src/scenes/acrobot/manifest';
import {
  EXACT_POINT_FLOATS,
  HINGE_R,
  REACH,
  SWING_PLANE_CLEARANCE,
  WATER_Y,
  acrobotPointsExact,
  makeBankHeight,
  pivotAnchor,
  successAnchor,
  willowBoughs,
} from '../../src/scenes/acrobot/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  probeSnapshot,
  toFrameState,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 32; // the demo run's first Acrobot lane

/** one rendered frame at 60 fps — WALL seconds, exactly what the host reports */
const DT_WALL = 1 / 60;
/**
 * The transport speeds the harness ships. EVERY guard test runs all of them: the pose
 * timeline advances in PLAYBACK seconds (`dtReal · speed`), so a guard that reads
 * `dtReal` is right at 1× by coincidence and wrong everywhere else.
 */
const SPEEDS = [0.25, 0.5, 1, 2, 4] as const;

function geometryOf(): EnvGeometry {
  return acrobotDecoder.geometry(acrobotDecoder.initStatic(SEED), SEED);
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

/**
 * A frame as the HOST assembles it: the pose is interpolated (so it may be anywhere
 * between two control rows) while `derived` stays the exact floor-frame state, which is
 * precisely the asymmetry the arc guard reads.
 */
function hostFrame(
  name: string,
  pose: readonly [number, number, number, number],
  anchor: readonly [number, number, number, number],
  action = 1,
  step = 0,
): SceneFixture {
  return {
    name,
    pose: [pose[0], pose[1], pose[2], pose[3], 0, 0, 0, 0, 0],
    ctrl: [action, 0, 0, 0, 0, 0],
    derived: {
      th1: anchor[0],
      th2: anchor[1],
      w1: anchor[2],
      w2: anchor[3],
      tipHeight: acrobotTipHeight(anchor[0], anchor[1]),
      action,
      torque: (action % 3) - 1,
    },
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
  const o = module.scene.getObjectByName(name);
  if (!o) throw new Error(`object "${name}" missing`);
  o.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(o.matrixWorld);
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Closest distance between two 3D segments (Ericson, *Real-Time Collision Detection*
 * §5.1.9). The swing-clearance question is exactly this: a link is a segment, a bough is
 * a segment, and "the arm passes through the branch" is their distance falling under the
 * sum of the two mesh radii.
 */
function segmentDistance(
  p1: readonly [number, number, number],
  q1: readonly [number, number, number],
  p2: readonly [number, number, number],
  q2: readonly [number, number, number],
): number {
  const d1x = q1[0] - p1[0];
  const d1y = q1[1] - p1[1];
  const d1z = q1[2] - p1[2];
  const d2x = q2[0] - p2[0];
  const d2y = q2[1] - p2[1];
  const d2z = q2[2] - p2[2];
  const rx = p1[0] - p2[0];
  const ry = p1[1] - p2[1];
  const rz = p1[2] - p2[2];
  const a = d1x * d1x + d1y * d1y + d1z * d1z;
  const e = d2x * d2x + d2y * d2y + d2z * d2z;
  const f = d2x * rx + d2y * ry + d2z * rz;
  const EPS = 1e-12;
  let s = 0;
  let t = 0;
  if (a <= EPS && e <= EPS) {
    // two points
  } else if (a <= EPS) {
    t = clamp01(f / e);
  } else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= EPS) {
      s = clamp01(-c / a);
    } else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z;
      const denom = a * e - b * b;
      s = denom > EPS ? clamp01((b * f - c * e) / denom) : 0;
      t = (b * s + f) / e;
      if (t < 0) {
        t = 0;
        s = clamp01(-c / a);
      } else if (t > 1) {
        t = 1;
        s = clamp01((b - c) / a);
      }
    }
  }
  return Math.hypot(
    p1[0] + d1x * s - (p2[0] + d2x * t),
    p1[1] + d1y * s - (p2[1] + d2y * t),
    p1[2] + d1z * s - (p2[2] + d2z * t),
  );
}

export interface StepPlayback {
  /** frames delivered */
  frames: number;
  /** PLAYBACK seconds actually elapsed across them */
  elapsedPlay: number;
  /** total distance the lantern travelled, summed frame to frame */
  path: number;
  /** the worst single-frame jump of the lantern */
  worstJump: number;
}

/**
 * Play ONE control step exactly the way `kit/harness` does at transport `speed`:
 *
 *   * the host advances `ctx.time()` by `dtReal · clock.speed` (PLAYBACK seconds) and
 *     hands the scene the WALL `dtReal`;
 *   * the pose is lerped by the playback fraction of the step, while `derived` stays the
 *     exact floor frame — the asymmetry the arc guard exists to read.
 *
 * `make(alpha)` builds the interpolated frame. Returns what the lantern actually did, so
 * a test can bound the rendered motion against the physics at every speed.
 */
function playControlStep(
  module: SceneModule,
  ctx: StubSceneContext,
  speed: number,
  t0: number,
  make: (alpha: number) => SceneFixture,
): StepPlayback {
  const dtPlay = DT_WALL * speed;
  const frames = Math.max(1, Math.ceil(AB_DT / dtPlay - 1e-9));
  let previous = worldOf(module, 'lantern');
  let path = 0;
  let worstJump = 0;
  let elapsedPlay = 0;
  for (let k = 1; k <= frames; k++) {
    elapsedPlay = Math.min(AB_DT, k * dtPlay);
    ctx.setTime(t0 + elapsedPlay);
    module.update(
      toFrameUpdate(make(elapsedPlay / AB_DT), { scrubbed: false, dtReal: DT_WALL }),
    );
    const now = worldOf(module, 'lantern');
    const jump = previous.distanceTo(now);
    path += jump;
    if (jump > worstJump) worstJump = jump;
    previous = now;
  }
  return { frames, elapsedPlay, path, worstJump };
}

// ---------------------------------------------------------------------------

describe('acrobot / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('Acrobot-cont-v0');
    expect(manifest.slug).toBe('acrobot');
    expect(manifest.accent).toBe('acrobot');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a reachable, un-scored pose', () => {
    const p = manifest.thumbFrame.pose;
    const h = acrobotTipHeight(p[0] ?? 0, p[1] ?? 0);
    expect(h).toBeLessThan(AB_SUCCESS_HEIGHT); // the env would have reset it otherwise
    expect(h).toBeGreaterThan(0.9); // ...but it is a hand's width from scoring
  });

  test('ships at least three canonical poses, all with a consistent derived state', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      const d = f.derived as { th1: number; th2: number; w1: number; w2: number; tipHeight: number };
      expect(d.th1).toBe(f.pose[0] ?? Number.NaN);
      expect(d.th2).toBe(f.pose[1] ?? Number.NaN);
      expect(d.w1).toBe(f.pose[2] ?? Number.NaN);
      expect(d.w2).toBe(f.pose[3] ?? Number.NaN);
      expect(d.tipHeight).toBeCloseTo(acrobotTipHeight(d.th1, d.th2), 12);
    }
  });

  test('the fixtures cover the reset, the success surface and a wrapped pair', () => {
    const hang = fixtureNamed('dawn-hang');
    expect(hang.pose.slice(0, 4)).toEqual([0, 0, 0, 0]);

    const crossing = fixtureNamed('crossing-the-lights');
    expect(acrobotTipHeight(crossing.pose[0] ?? 0, crossing.pose[1] ?? 0)).toBeCloseTo(AB_SUCCESS_HEIGHT, 12);

    const wrapped = fixtureNamed('wrapped-fold');
    expect(Math.abs(wrapped.pose[0] ?? 0)).toBeGreaterThan(Math.PI - 0.2);
    expect(Math.abs(wrapped.pose[1] ?? 0)).toBeGreaterThan(Math.PI - 0.35);
  });

  test('th2ForSuccess really puts the free end on the surface', () => {
    for (const th1 of [1.6, 1.9, 2.2, 2.6, -2.1]) {
      const th2 = th2ForSuccess(th1);
      expect(Number.isFinite(th2)).toBe(true);
      expect(acrobotTipHeight(th1, th2)).toBeCloseTo(AB_SUCCESS_HEIGHT, 12);
    }
  });
});

describe('acrobot / the FK is the scene graph', () => {
  test('every probe lands on ground truth at hero tier', async () => {
    const module = createScene();
    const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier: 'hero' });
    // these roots carry POSITION ONLY — they are exact, not merely within 1e-3
    expect(report.worst.get('pivot') ?? 1).toBe(0);
    expect(report.worst.get('success-line') ?? 1).toBe(0);
    expect(report.worst.get('elbow') ?? 1).toBeLessThan(1e-12);
    expect(report.worst.get('lantern') ?? 1).toBeLessThan(1e-12);
    module.dispose();
  });

  test('alignment holds at medium and thumb (tiers may not move state-bearing meshes)', async () => {
    for (const tier of ['medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      expect(report.worst.get('lantern') ?? 1).toBeLessThan(1e-12);
      module.dispose();
    }
  });

  test('the chain reproduces acrobotPoints over a dense sweep of BOTH angles', async () => {
    const { module } = await live();
    const pivot = pivotAnchor(geometryOf());
    const points = new Float64Array(EXACT_POINT_FLOATS);
    let worstElbow = 0;
    let worstTip = 0;
    for (let i = 0; i < 24; i++) {
      const th1 = -Math.PI + (i / 24) * 2 * Math.PI;
      for (let j = 0; j < 24; j++) {
        const th2 = -Math.PI + (j / 24) * 2 * Math.PI;
        module.update(toFrameUpdate(hostFrame(`s${i}-${j}`, [th1, th2, 0, 0], [th1, th2, 0, 0], 1, i * 24 + j)));
        // the pose crosses the host as f32, so ground truth is its f32 round-trip
        acrobotPointsExact(Math.fround(th1), Math.fround(th2), points);
        const elbow = worldOf(module, 'elbow');
        const tip = worldOf(module, 'lantern');
        worstElbow = Math.max(
          worstElbow,
          Math.hypot(elbow.x - (pivot[0] + (points[0] ?? 0)), elbow.y - (pivot[1] + (points[1] ?? 0)), elbow.z - pivot[2]),
        );
        worstTip = Math.max(
          worstTip,
          Math.hypot(tip.x - (pivot[0] + (points[2] ?? 0)), tip.y - (pivot[1] + (points[3] ?? 0)), tip.z - pivot[2]),
        );
      }
    }
    expect(worstElbow).toBeLessThan(1e-12);
    expect(worstTip).toBeLessThan(1e-12);
    module.dispose();
  });

  test('the scene\'s f64 point twin IS the decoder\'s acrobotPoints', () => {
    const exact = new Float64Array(EXACT_POINT_FLOATS);
    const decoder = new Float32Array(4);
    for (let i = 0; i < 37; i++) {
      const th1 = -Math.PI + (i / 37) * 2 * Math.PI;
      for (let j = 0; j < 37; j++) {
        const th2 = -Math.PI + (j / 37) * 2 * Math.PI;
        acrobotPointsExact(th1, th2, exact);
        acrobotPoints(th1, th2, decoder);
        for (let k = 0; k < 4; k++) {
          // the ONLY difference is the decoder's Float32Array storage
          expect(Math.abs((exact[k] ?? 0) - (decoder[k] ?? 0))).toBeLessThan(2e-7);
        }
        // ...and the free end's height is the env's own success quantity, bitwise
        expect(exact[3]).toBe(acrobotTipHeight(th1, th2));
      }
    }
  });

  test('both links are exactly 1.0 world unit long at every pose', async () => {
    const { module } = await live();
    for (let i = 0; i <= 40; i++) {
      const th1 = -3 + (i / 40) * 6;
      const th2 = 2.9 - (i / 40) * 5.6;
      module.update(toFrameUpdate(hostFrame(`len${i}`, [th1, th2, 0, 0], [th1, th2, 0, 0], 1, i)));
      const pivot = worldOf(module, 'pivot');
      const elbow = worldOf(module, 'elbow');
      const tip = worldOf(module, 'lantern');
      expect(pivot.distanceTo(elbow)).toBeCloseTo(AB_L1, 12);
      expect(elbow.distanceTo(tip)).toBeCloseTo(AB_L2, 12);
      // the swing is planar: everything stays on the pivot's own z
      expect(elbow.z).toBeCloseTo(pivot.z, 12);
      expect(tip.z).toBeCloseTo(pivot.z, 12);
    }
    module.dispose();
  });

  test('th1 is measured from straight DOWN and th2 relative to link 1', async () => {
    const { module } = await live();
    // hang: the whole arm points straight down from the pivot
    module.update(toFrameUpdate(hostFrame('hang', [0, 0, 0, 0], [0, 0, 0, 0])));
    expect(worldOf(module, 'elbow').y).toBeCloseTo(pivotAnchor(geometryOf())[1] - AB_L1, 12);
    expect(worldOf(module, 'lantern').y).toBeCloseTo(pivotAnchor(geometryOf())[1] - AB_L1 - AB_L2, 12);
    expect(worldOf(module, 'elbow').x).toBeCloseTo(pivotAnchor(geometryOf())[0], 12);

    // th1 = π/2 swings link 1 to +x; th2 = 0 keeps link 2 in line with it
    // (the pose crosses the wire as f32, so π/2 arrives rounded: 1e-7, not 1e-12)
    module.update(toFrameUpdate(hostFrame('right', [Math.PI / 2, 0, 0, 0], [Math.PI / 2, 0, 0, 0])));
    const pivot = pivotAnchor(geometryOf());
    expect(worldOf(module, 'elbow').x).toBeCloseTo(pivot[0] + AB_L1, 7);
    expect(worldOf(module, 'lantern').x).toBeCloseTo(pivot[0] + AB_L1 + AB_L2, 7);

    // th2 = −π/2 folds link 2 back to straight down — RELATIVE to link 1, not the world
    module.update(toFrameUpdate(hostFrame('folded', [Math.PI / 2, -Math.PI / 2, 0, 0], [Math.PI / 2, -Math.PI / 2, 0, 0])));
    expect(worldOf(module, 'lantern').x).toBeCloseTo(pivot[0] + AB_L1, 7);
    expect(worldOf(module, 'lantern').y).toBeCloseTo(pivot[1] - AB_L2, 7);
    module.dispose();
  });
});

describe('acrobot / the success surface is physically placed', () => {
  test('the garland root and EVERY bulb centre sit exactly on the successLine anchor', async () => {
    const { module } = await live();
    const success = successAnchor(geometryOf());
    expect(successAnchor(geometryOf())[1]).toBe(AB_SUCCESS_HEIGHT);
    expect(worldOf(module, 'success-line').y).toBe(success[1]);

    const bulbs = module.scene.getObjectByName('garlandBulbs') as THREE.InstancedMesh | undefined;
    expect(bulbs).toBeDefined();
    if (!bulbs) return;
    expect(bulbs.count).toBeGreaterThan(6);
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    for (let i = 0; i < bulbs.count; i++) {
      bulbs.getMatrixAt(i, m);
      p.setFromMatrixPosition(m);
      expect(p.y).toBe(success[1]);
    }
    module.dispose();
  });

  test('"the lantern is above the bulbs" IS "tipHeight > 1"', async () => {
    const { module } = await live();
    const success = successAnchor(geometryOf());
    const th1 = 2;
    for (const delta of [-0.25, -0.05, 0, 0.05, 0.3]) {
      // walk th2 either side of the exact crossing solution
      const th2 = th2ForSuccess(th1) + delta * 0.35;
      const height = acrobotTipHeight(th1, th2);
      module.update(toFrameUpdate(hostFrame('cross', [th1, th2, 0, 0], [th1, th2, 0, 0])));
      const tipY = worldOf(module, 'lantern').y;
      expect(height > AB_SUCCESS_HEIGHT).toBe(tipY > success[1] + 1e-12);
    }
    module.dispose();
  });

  test('no willow strand and no lily pad may enter the swing volume', async () => {
    const { module } = await live();
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const sc = new THREE.Vector3();

    const fronds = module.scene.getObjectByName('willowFronds') as THREE.InstancedMesh | undefined;
    expect(fronds).toBeDefined();
    expect(fronds?.count ?? 0).toBeGreaterThan(40);
    for (let i = 0; i < (fronds?.count ?? 0); i++) {
      fronds?.getMatrixAt(i, m);
      m.decompose(p, q, sc);
      // a strand hangs from p down by FROND_LEN * scale.y; only strands well out of the
      // swing plane are allowed to hang low at all
      if (Math.abs(p.z) >= 0.62) continue;
      const bottom = p.y - 1.5 * sc.y;
      const closestY = bottom > 0 ? bottom : p.y > 0 ? 0 : p.y;
      expect(Math.hypot(p.x, closestY)).toBeGreaterThanOrEqual(REACH);
    }

    const pads = module.scene.getObjectByName('lilyPads') as THREE.InstancedMesh | undefined;
    expect(pads).toBeDefined();
    expect(pads?.count ?? 0).toBeGreaterThan(4);
    for (let i = 0; i < (pads?.count ?? 0); i++) {
      pads?.getMatrixAt(i, m);
      p.setFromMatrixPosition(m);
      // a pad floats ON the surface and never under the arm's own column
      expect(p.y).toBeGreaterThan(WATER_Y);
      expect(p.y).toBeLessThan(WATER_Y + 0.05);
      expect(Math.abs(p.x) >= 1.55 || p.z <= -1.6 || p.z >= 1.6).toBe(true);
    }
    module.dispose();
  });

  test('the willow BOUGHS keep out of the swing plane wherever they are inside the reach', () => {
    // THE REGRESSION THIS LOCKS. The frond/pad test below never looked at the limb the
    // fronds hang from: the main bough ran through [0, BOUGH_Y, 0] — 0.34 from the pivot,
    // dead centre of the disc, in the plane the arm swings in — and the recorded stream
    // put a link within a mesh radius of it on 2.25% of frames. The env's objective is to
    // raise the tip a full link ABOVE the pivot, so this is not a rare pose: the arm
    // crosses the bough's line on every scoring swing.
    const pivot = pivotAnchor(geometryOf());
    let worstInPlane = Infinity;
    for (const bough of willowBoughs()) {
      for (let i = 0; i < bough.length - 1; i++) {
        const a = bough[i] as readonly [number, number, number];
        const b = bough[i + 1] as readonly [number, number, number];
        for (let s = 0; s <= 128; s++) {
          const t = s / 128;
          const x = a[0] + (b[0] - a[0]) * t - pivot[0];
          const y = a[1] + (b[1] - a[1]) * t - pivot[1];
          const z = a[2] + (b[2] - a[2]) * t - pivot[2];
          // the arm sweeps the disc r ≤ REACH at z = pivot.z; anything inside it must be
          // out of the plane, and there is no "close enough" — 0.007 was measured
          if (Math.hypot(x, y) > REACH) continue;
          worstInPlane = Math.min(worstInPlane, Math.abs(z));
        }
      }
    }
    expect(worstInPlane).toBeGreaterThanOrEqual(SWING_PLANE_CLEARANCE);
  });

  test('no willow or garland VERTEX stands in the swing volume, bar the hinge sleeve', async () => {
    // the polyline test above is the skeleton; this is the MESH the reviewer measured —
    // every triangle of the willow and of the garland rig, in world space
    const { module } = await live();
    const pivot = pivotAnchor(geometryOf());
    const v = new THREE.Vector3();
    for (const name of ['willow', 'garlandRig']) {
      const mesh = module.scene.getObjectByName(name) as THREE.Mesh | undefined;
      expect(mesh).toBeDefined();
      if (!mesh) continue;
      mesh.updateWorldMatrix(true, false);
      const pos = mesh.geometry.getAttribute('position');
      let checked = 0;
      let hinge = 0;
      let worstHingeR = 0;
      for (let i = 0; i < pos.count; i++) {
        v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
        const r = Math.hypot(v.x - pivot[0], v.y - pivot[1]);
        if (r > REACH) continue; // outside the arm's reach: it may be anywhere
        if (Math.abs(v.z - pivot[2]) >= SWING_PLANE_CLEARANCE) {
          checked++;
          continue; // inside the reach but out of the plane: clear
        }
        // Inside the reach AND in the plane. Only the hinge sleeve may be here — the
        // brass collar and the tie at the foot of the cord, i.e. the attachment the arm
        // turns about. A hinge that cleared its own axle would be a lie.
        hinge++;
        worstHingeR = Math.max(worstHingeR, r);
        if (!(r <= HINGE_R)) {
          throw new Error(
            `${name} vertex ${i} stands in the swing volume: r=${r.toFixed(3)} ` +
              `z=${(v.z - pivot[2]).toFixed(3)} (needs |z| ≥ ${SWING_PLANE_CLEARANCE} ` +
              `or r ≤ ${HINGE_R})`,
          );
        }
      }
      if (name === 'willow') {
        // the bough really does pass through the disc — this test would be vacuous if it
        // did not — and the hinge exemption really is only the hinge
        expect(checked).toBeGreaterThan(120);
        expect(hinge).toBeGreaterThan(0);
        // The exemption is not a loophole. Two independent bounds: no in-plane vertex
        // reaches past 0.36 (the number world.ts's HINGE_R comment claims), and the set is
        // small enough to be only the collar + the cord's tie — a limb could not hide in it.
        expect(worstHingeR).toBeLessThan(0.36);
        expect(hinge).toBeLessThan(400);
      }
    }
    module.dispose();
  });

  test('the swept arm clears every bough over a dense sweep of both angles', async () => {
    // the end-to-end statement: no pose of the two links comes within a mesh radius of a
    // limb. Segment-to-segment, over 33 × 33 poses, against every bough of the willow.
    const pivot = pivotAnchor(geometryOf());
    const points = new Float64Array(EXACT_POINT_FLOATS);
    const boughs = willowBoughs();
    /** bough radius (0.115 at the trunk) + the thicker vine (0.055) + the lantern's body */
    const MESH_RADII = 0.115 + 0.055;
    let worst = Infinity;
    for (let i = 0; i <= 32; i++) {
      const th1 = -Math.PI + (i / 32) * 2 * Math.PI;
      for (let j = 0; j <= 32; j++) {
        const th2 = -Math.PI + (j / 32) * 2 * Math.PI;
        acrobotPointsExact(th1, th2, points);
        const p0: [number, number, number] = [pivot[0], pivot[1], pivot[2]];
        const p1: [number, number, number] = [pivot[0] + (points[0] ?? 0), pivot[1] + (points[1] ?? 0), pivot[2]];
        const p2: [number, number, number] = [pivot[0] + (points[2] ?? 0), pivot[1] + (points[3] ?? 0), pivot[2]];
        for (const bough of boughs) {
          for (let k = 0; k < bough.length - 1; k++) {
            const a = bough[k] as readonly [number, number, number];
            const b = bough[k + 1] as readonly [number, number, number];
            worst = Math.min(worst, segmentDistance(p0, p1, a, b), segmentDistance(p1, p2, a, b));
          }
        }
      }
    }
    expect(worst).toBeGreaterThan(MESH_RADII);
  });

  test('nothing solid is planted inside the swing disc', () => {
    // the willow's bank sits well clear of the arm's reach, and the pond is below it
    const bank = makeBankHeight(SEED);
    expect(WATER_Y).toBeLessThan(-(AB_L1 + AB_L2));
    for (let i = 0; i <= 40; i++) {
      const x = -REACH + (i / 40) * REACH * 2;
      // under the whole swing the bed is flat and submerged: no lump can breach the mirror
      expect(bank(x, 0)).toBeLessThan(WATER_Y);
      expect(bank(x, 0)).toBe(bank(0, 0));
    }
  });
});

describe('acrobot / the torque curl is sign-correct', () => {
  test('+1 curls toward +x at the elbow, −1 toward −x, 0 is invisible', async () => {
    const { module } = await live();
    const curl = module.scene.getObjectByName('torqueCurl');
    expect(curl).toBeDefined();
    if (!curl) return;

    // the arm hanging: increasing th2 (what +1 torque does) drives the free end toward +x
    module.update(toFrameUpdate(hostFrame('push', [0, 0, 0, 0], [0, 0, 0, 0], 2)));
    expect(curl.visible).toBe(true);
    const elbow = worldOf(module, 'elbow');
    const headPlus = objectWorld(module, 'torqueHead');
    expect(headPlus.x - elbow.x).toBeGreaterThan(0.05);

    module.update(toFrameUpdate(hostFrame('pull', [0, 0, 0, 0], [0, 0, 0, 0], 0)));
    expect(curl.visible).toBe(true);
    const headMinus = objectWorld(module, 'torqueHead');
    expect(headMinus.x - elbow.x).toBeLessThan(-0.05);
    // the two sides are mirror images about the joint
    expect(headMinus.x - elbow.x).toBeCloseTo(-(headPlus.x - elbow.x), 12);
    expect(headMinus.y).toBeCloseTo(headPlus.y, 12);

    module.update(toFrameUpdate(hostFrame('coast', [0, 0, 0, 0], [0, 0, 0, 0], 1)));
    expect(curl.visible).toBe(false);
    module.dispose();
  });

  test('torqueOf decodes the kernel action, and never a torque on a missing derived', () => {
    const base = toFrameState({ name: 'x', pose: [0, 0, 0, 0, 0, 0, 0, 0, 0], ctrl: [2, 0, 0, 0, 0, 0] });
    expect(torqueOf(base)).toBe(1);
    base.ctrl[0] = 0;
    expect(torqueOf(base)).toBe(-1);
    base.ctrl[0] = 1;
    expect(torqueOf(base)).toBe(0);
    base.derived = { torque: -1 };
    base.ctrl[0] = 2;
    expect(torqueOf(base)).toBe(-1); // sim-core wins over the raw ctrl
  });
});

describe('acrobot / the arc guard', () => {
  test('arcCorrupt fires on a wrap and on a respawn slide, never on plausible motion', () => {
    // a th1 wrap: the lerp sets off backwards while the true motion is forwards
    expect(arcCorrupt(-3.0, 0.4, 0.4)).toBe(true);
    expect(arcCorrupt(-0.2, 3.0, 3.0)).toBe(true); // caught in the first frames, by direction
    // the respawn slide: the whole arm sliding home inside one control step
    expect(arcCorrupt(-1.0, 1.6, 0.8)).toBe(true);
    // ordinary fast swinging is NOT corrupt
    expect(arcCorrupt(0.05, 2.0, 2.1)).toBe(false);
    expect(arcCorrupt(0.9, 4.0, 4.4)).toBe(false);
    // a turning point inside a step: a small reversal must stay legal
    expect(arcCorrupt(-0.02, 1.0, 0.4)).toBe(false);
    // and a still joint is never tripped by noise
    expect(arcCorrupt(0.001, 0, 0)).toBe(false);
  });

  test('wrapPi is a real wrap and the identity inside (−π, π)', () => {
    expect(wrapPi(0)).toBe(0);
    expect(wrapPi(1.25)).toBe(1.25);
    expect(wrapPi(-1.25)).toBe(-1.25);
    expect(wrapPi(Math.PI * 2)).toBeCloseTo(0, 12);
    expect(wrapPi(3.2)).toBeCloseTo(3.2 - Math.PI * 2, 12);
    expect(wrapPi(-3.2)).toBeCloseTo(Math.PI * 2 - 3.2, 12);
  });

  test('a host lerp across a ±π wrap never whips the forearm — at EVERY transport speed', async () => {
    // THE REGRESSION THIS LOCKS. The guard integrates and budgets in PLAYBACK seconds, so
    // every claim below must hold identically at 0.25× and at 4×. Driven on wall time it
    // passed at 1× and rendered a 1.6-unit single-frame teleport of the lantern at 0.25×
    // (the one-control-step budget burnt 4× early and released mid-wrap) — worse than no
    // guard at all, and invisible to a 1×-only test.
    for (const speed of SPEEDS) {
      const { module, ctx } = await live();
      // the real shape of the artefact: th2 = 3.10 -> −3.10 in one control step, i.e. a
      // TRUE displacement of +0.083 that the linear lerp renders as −6.20.
      const anchor: readonly [number, number, number, number] = [0.2, 3.1, 0.5, 0.4];
      const target = -3.1;
      ctx.setTime(10);
      module.update(toFrameUpdate(hostFrame('pre', [0.2, 3.1, 0.5, 0.4], anchor, 1, 100), { scrubbed: false }));

      const played = playControlStep(module, ctx, speed, 10, (alpha) =>
        hostFrame(`mid@${speed}`, [0.2 + 0.1 * alpha, 3.1 + (target - 3.1) * alpha, 0.5, 0.4], anchor, 1, 100),
      );

      // an unguarded lerp sweeps the forearm most of a half circle (> 2.5 units of tip
      // path); the physical bound here is (|w1| + |w1 + w2|) · AB_DT = 0.28
      expect(`${speed}x path ${played.path < 0.6}`).toBe(`${speed}x path true`);
      // ...and no single frame may move the tip further than it can physically travel in
      // one frame of PLAYBACK time — the teleport test, which is what actually broke
      const reach = 1.4 * DT_WALL * speed + 0.02;
      expect(`${speed}x jump ${played.worstJump < reach}`).toBe(`${speed}x jump true`);

      // ...and the forearm is left within a few degrees of the TRUE continuous angle
      const link2 = module.scene.getObjectByName('link2Node');
      expect(link2).toBeDefined();
      const trueTh2 = 3.1 + 0.083 * (played.elapsedPlay / AB_DT);
      expect(Math.abs(wrapPi((link2?.rotation.z ?? 0) - trueTh2))).toBeLessThan(0.25);
      module.dispose();
    }
  });

  test('a paused host freezes the guard: repeated update() is bit-identical', async () => {
    const { module, ctx } = await live();
    const anchor: readonly [number, number, number, number] = [0.2, 3.1, 0.5, 0.4];
    ctx.setTime(10);
    module.update(toFrameUpdate(hostFrame('pre', [0.2, 3.1, 0.5, 0.4], anchor, 1, 100), { scrubbed: false }));
    // Engage the guard mid-wrap: three frames of a corrupt lerp leave `held` well inside
    // its one-control-step budget, i.e. the guard is actively CARRYING the angle on the
    // velocity channel — the branch that used to integrate on wall time.
    const rawTh2 = (k: number): number => 3.1 - 6.2 * (k / 12);
    for (let k = 1; k <= 3; k++) {
      ctx.setTime(10 + k * DT_WALL);
      module.update(
        toFrameUpdate(hostFrame(`mid${k}`, [0.2, rawTh2(k), 0.5, 0.4], anchor, 1, 100), {
          scrubbed: false,
          dtReal: DT_WALL,
        }),
      );
    }
    const link2 = module.scene.getObjectByName('link2Node');
    // proof the carrying branch is live: the render is nowhere near the pose it was handed
    expect(Math.abs(wrapPi((link2?.rotation.z ?? 0) - rawTh2(3)))).toBeGreaterThan(1);

    // ...and now PAUSE. The host keeps calling update() with a real dtReal and re-delivers
    // the SAME frame with ctx.time() frozen (kit/harness: `clock.advance` is a no-op while
    // paused, the rAF loop is not). A guard on wall time integrates right through it —
    // measured 1.229 world units of travel and then a 0.614 snap back. STYLE.md: pause
    // must freeze the world coherently.
    const paused = toFrameUpdate(hostFrame('paused', [0.2, rawTh2(3), 0.5, 0.4], anchor, 1, 100), {
      scrubbed: false,
      dtReal: DT_WALL,
    });
    module.update(paused);
    const before = probeSnapshot(module);
    const tip = worldOf(module, 'lantern');
    for (let i = 0; i < 30; i++) module.update(paused);
    expect(probeSnapshot(module)).toEqual(before);
    const after = worldOf(module, 'lantern');
    expect(after.x).toBe(tip.x);
    expect(after.y).toBe(tip.y);
    expect(after.z).toBe(tip.z);
    module.dispose();
  });

  test('the guard releases onto the truth once the control step is over', async () => {
    const { module, ctx } = await live();
    const anchor: readonly [number, number, number, number] = [0.2, 3.1, 0.5, 0.4];
    ctx.setTime(0);
    module.update(toFrameUpdate(hostFrame('pre', [0.2, 3.1, 0.5, 0.4], anchor, 1, 100), { scrubbed: false }));
    // 20 frames of a corrupt lerp: the guard's budget is ONE control step, after which it
    // must take the pose at its word rather than integrate forever
    for (let k = 1; k <= 20; k++) {
      ctx.setTime(k / 60);
      module.update(
        toFrameUpdate(hostFrame(`mid${k}`, [0.2, -3.0, 0.5, 0.4], anchor, 1, 100), {
          scrubbed: false,
          dtReal: 1 / 60,
        }),
      );
    }
    const link2 = module.scene.getObjectByName('link2Node');
    expect(Math.abs(wrapPi((link2?.rotation.z ?? 0) - -3.0))).toBeLessThan(1e-9);
    module.dispose();
  });

  test('acrobotTipSpeed is the exact derivative of acrobotPoints', () => {
    const h = 1e-5;
    const a = new Float64Array(EXACT_POINT_FLOATS);
    const b = new Float64Array(EXACT_POINT_FLOATS);
    for (const [th1, th2, w1, w2] of [
      [0.4, 1.1, 2.0, -3.0],
      [-2.2, 0.3, -1.4, 5.5],
      [1.9, -2.7, 0.2, 0.1],
    ] as const) {
      acrobotPointsExact(th1 - w1 * h, th2 - w2 * h, a);
      acrobotPointsExact(th1 + w1 * h, th2 + w2 * h, b);
      const numeric = Math.hypot((b[2] ?? 0) - (a[2] ?? 0), (b[3] ?? 0) - (a[3] ?? 0)) / (2 * h);
      expect(acrobotTipSpeed(th1, th2, w1, w2)).toBeCloseTo(numeric, 7);
    }
  });
});

describe('acrobot / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('dawn-hang')));
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

    // hanging exactly on the success surface with no event: nothing may fire
    ctx.setTime(1);
    module.update(toFrameUpdate(fixtureNamed('crossing-the-lights'), { scrubbed: false }));
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);

    // the decoder's real batch (goal + done + teleport arrive together): one celebration
    ctx.setTime(2);
    module.update(
      toFrameUpdate(fixtureNamed('dawn-hang'), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }],
      }),
    );
    expect(pool.stats.bursts).toBe(1);
    expect(pool.stats.ripples).toBe(1);
    expect(pool.pulse.bloom).toBeGreaterThan(0);
    module.dispose();
  });

  test('the bloom happens at the APEX the lantern reached — at EVERY transport speed', async () => {
    // The recorded goal row IS the all-zero hang, so the host slides the whole arm home
    // for a full control step BEFORE the event arrives. On wall time the guard's budget
    // ran out mid-slide at every speed below 1×, and the celebration then fired at
    // y = −2.000 — the reset hang, 3 units under the success it was celebrating. This is
    // the same defect as the wrap test, seen through the celebration.
    const success = successAnchor(geometryOf());
    const crossing = fixtureNamed('crossing-the-lights');
    const th1 = crossing.pose[0] ?? 0;
    const th2 = crossing.pose[1] ?? 0;

    for (const speed of SPEEDS) {
      const { module, ctx } = await live();
      ctx.setTime(3);
      module.update(toFrameUpdate(crossing, { scrubbed: false }));
      const apex = worldOf(module, 'lantern');
      expect(apex.y).toBeCloseTo(success[1], 6);

      // the host lerps the arm all the way into the all-zero reset row over one control
      // step; `derived` stays the crossing, which is the exact truth the guard reads
      const played = playControlStep(module, ctx, speed, 3, (alpha) =>
        hostFrame(
          'slide',
          [th1 * (1 - alpha), th2 * (1 - alpha), 0.8, 0.45],
          [th1, th2, 1.6, 0.9],
          2,
          crossing.step ?? 0,
        ),
      );
      const carried = worldOf(module, 'lantern');
      // the guard carried the arm ON ITS OWN VELOCITY instead of sliding it home: it is
      // still up at the lights (the slide would have taken it to y = −2)...
      expect(`${speed}x apexY ${carried.y > success[1] - 0.2}`).toBe(`${speed}x apexY true`);
      // ...and it never moved further than the tip can physically travel in that time
      // (|d(tip)/dt| = 2.05 at this pose), which is what "held at the apex" really means
      const reach = 2.05 * played.elapsedPlay + 0.05;
      expect(`${speed}x carried ${carried.distanceTo(apex) < reach}`).toBe(`${speed}x carried true`);

      // now the event frame: the recorded row IS the hang
      ctx.setTime(3 + AB_DT + DT_WALL * speed);
      module.update(
        toFrameUpdate(hostFrame('scored', [0, 0, 0, 0], [0, 0, 0, 0], 1, (crossing.step ?? 0) + 1), {
          scrubbed: false,
          prev: toFrameState(crossing),
          events: [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }],
        }),
      );
      // the celebration light rides the apex, ~3 units above the hang it fired on
      const light = module.scene.getObjectByName('oakPoint2') as THREE.PointLight | undefined;
      expect(light).toBeDefined();
      if (!light) return;
      expect(`${speed}x lightY ${light.position.y > success[1] - 0.3}`).toBe(`${speed}x lightY true`);
      expect(light.intensity).toBeGreaterThan(0);
      // and the teleport is a HARD SNAP: the arm is already back at the hang
      expect(worldOf(module, 'lantern').y).toBeCloseTo(pivotAnchor(geometryOf())[1] - AB_L1 - AB_L2, 9);
      module.dispose();
    }
  });

  test('a scrub clears every in-flight beat AND the tip ribbon', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    if (!pool) throw new Error('no pool');
    // build a real ribbon first
    for (let k = 0; k <= 12; k++) {
      ctx.setTime(k * 0.05);
      const th1 = 0.4 + k * 0.12;
      module.update(
        toFrameUpdate(hostFrame(`sw${k}`, [th1, 0.2, 3, 2], [th1, 0.2, 3, 2], 1, k), { scrubbed: false }),
      );
    }
    const ribbon = module.scene.getObjectByName('tipRibbon');
    expect(ribbon).toBeDefined();
    expect(ribbon?.visible).toBe(true);

    ctx.setTime(0.7);
    module.update(
      toFrameUpdate(fixtureNamed('swing-up'), { scrubbed: false, events: [{ type: 'goal' }] }),
    );
    expect(pool.pulse.bloom).toBeGreaterThan(0);

    ctx.setTime(0.71);
    module.update(toFrameUpdate(fixtureNamed('dawn-hang'), { scrubbed: true }));
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    expect(ribbon?.visible).toBe(false);
    module.dispose();
  });

  test('a teleport clears the ribbon too — no streak may cross a respawn', async () => {
    const { module, ctx } = await live();
    for (let k = 0; k <= 12; k++) {
      ctx.setTime(k * 0.05);
      const th1 = 0.4 + k * 0.12;
      module.update(
        toFrameUpdate(hostFrame(`sw${k}`, [th1, 0.2, 3, 2], [th1, 0.2, 3, 2], 1, k), { scrubbed: false }),
      );
    }
    const ribbon = module.scene.getObjectByName('tipRibbon');
    expect(ribbon?.visible).toBe(true);
    ctx.setTime(0.7);
    module.update(
      toFrameUpdate(hostFrame('reset', [0, 0, 0, 0], [0, 0, 0, 0], 1, 13), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }],
      }),
    );
    expect(ribbon?.visible).toBe(false);
    module.dispose();
  });

  test('the flourish envelope is bounded and never ends on a pop', () => {
    expect(envelope(-1, 2, 1)).toBe(0);
    expect(envelope(3, 2, 1)).toBe(0);
    expect(envelope(0, 2, 1)).toBe(1);
    expect(envelope(2, 2, 1)).toBe(0);
    for (let i = 0; i <= 60; i++) {
      const e = envelope((i / 60) * 2, 2, 1.6);
      expect(e).toBeGreaterThanOrEqual(0);
      expect(e).toBeLessThanOrEqual(1);
    }
  });
});

describe('acrobot / lifecycle', () => {
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
    module.update(toFrameUpdate(fixtureNamed('swing-up')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'willowGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'willowGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the camera is a near-profile frame that holds the whole reach', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('swing-up')));
    const cam = module.camera;
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    // near-profile: it looks down −z at the swing plane, never from above or along x
    expect(dir.z).toBeLessThan(-0.9);
    expect(Math.abs(dir.y)).toBeLessThan(0.25);
    expect(cam.fov).toBeGreaterThanOrEqual(35);
    expect(cam.fov).toBeLessThanOrEqual(45);
    // far enough back that the 40° vertical FOV contains the lights AND the pond
    const halfH = Math.tan((cam.fov * Math.PI) / 360) * cam.position.z;
    expect(halfH).toBeGreaterThan((AB_SUCCESS_HEIGHT - WATER_Y) / 2);
    module.dispose();
  });
});

describe('acrobot / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    // The wall view is the whole gallery on one page: every scene there is a thumb, and
    // the cap is what keeps 16 of them at 60 fps. Measured with the ground-truth overlay
    // ON, because that is the state review screenshots are taken in, and across every
    // fixture, because the busiest legal state is the one that has to fit.
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'acrobot (thumb, overlay on)');
    module.dispose();
  });
});
