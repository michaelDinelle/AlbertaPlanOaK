/// <reference types="bun" />
// The Lantern Bell (Pendulum-cont-v0) — the accuracy + lifecycle self-check.
//
// This mirrors four-rooms.test.ts's shape and adds what THIS env has to prove:
//   * the bob is exactly pivot + (L sin θ, −L cos θ) on the whole circle, at every tier;
//   * the torii clears the volume the LANTERN sweeps — measured off the built meshes, at
//     every z, with no exemptions (the earlier version of this test used a hand-written
//     clearance 0.13 short of the truth and skipped exactly the z band the hanger lives
//     in, so it passed while the cables ran through the paper barrel);
//   * the ANGLE-WRAP TRAP is handled AT EVERY SAMPLING DENSITY. The host advances
//     `controlHz·speed·dtReal` rows per render frame, so one record row can be 1 frame or
//     48; the earlier version tested only the 3-frames-per-row case, which is the one
//     density where the shipped guard happened to work. Here the env's own integrator
//     generates the trajectories and the host's own sampling law replays them at
//     0.75 … 48 frames per row;
//   * the lighting IS the shaped reward — a state the env scores BETTER is never rendered
//     darker, checked over a (θ, θ̇, u) grid rather than along the one slice where the old
//     product law could not fail;
//   * the torque curl is sign-correct and magnitude-scaled;
//   * and this scene NEVER celebrates, not even when handed an event batch.

import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';

import { existingCelebrations } from '@kit/celebrate';
import type { EmissiveMaterial } from '@kit/materials';
import type { QualityTier, SceneFixture, SceneModule } from '@kit/types';
import {
  PD_DT,
  PD_G,
  PD_L,
  PD_M,
  PD_MAX_SPEED,
  PD_MAX_TORQUE,
  pendulumDecoder,
  pendulumTip,
  wrapPi,
} from '@sim/envs/pendulum';
import type { EnvGeometry } from '@sim/types';

import createScene, {
  COST_MAX,
  MAX_HOLD_STEPS,
  PD_MAX_DTHETA,
  PD_MAX_DTHETADOT,
  pendulumCost,
  serenityOf,
  wrapReach,
} from '../../src/scenes/pendulum/index';
import { fixtures } from '../../src/scenes/pendulum/fixtures';
import { manifest, pendulumDerived, pendulumPose } from '../../src/scenes/pendulum/manifest';
import {
  EYE_MAJOR,
  EYE_TUBE,
  LANTERN_HALF_Z,
  LANTERN_REACH,
  PIN_R,
  ROD_LEN,
  STRAP_HALF,
  STRAP_Z,
  SWING_CLEARANCE,
  SWING_INNER,
  SWING_REACH,
} from '../../src/scenes/pendulum/dims';
import {
  makePondBed,
  POND_RX,
  REFLECT_LIFT,
  REFLECT_Z,
  WATER_LEVEL,
} from '../../src/scenes/pendulum/world';
import {
  assertAligned,
  assertInitDisposeRoundTrip,
  assertThumbBudget,
  assertUpdateIdempotent,
  makeSceneContext,
  toFrameUpdate,
  type StubSceneContext,
} from './aligned';

const SEED = 56; // the demo run's first Pendulum lane

function geometryOf(): EnvGeometry {
  return pendulumDecoder.geometry(pendulumDecoder.initStatic(SEED), SEED);
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

function poseFixture(name: string, th: number, thDot: number, torque: number, step: number): SceneFixture {
  return {
    name,
    pose: pendulumPose(th, thDot, torque),
    ctrl: [torque / PD_MAX_TORQUE, 0, 0, 0, 0, 0],
    derived: pendulumDerived(th, thDot, torque, torque / PD_MAX_TORQUE),
    step,
  };
}

function worldOf(module: SceneModule, probeName: string): THREE.Vector3 {
  const probe = module.probes().find((p) => p.name === probeName);
  if (!probe) throw new Error(`probe "${probeName}" missing`);
  probe.object.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(probe.object.matrixWorld);
}

function meshNamed(module: SceneModule, name: string): THREE.Mesh {
  const o = module.scene.getObjectByName(name);
  if (!(o instanceof THREE.Mesh)) throw new Error(`mesh "${name}" missing`);
  return o;
}

function emissiveGain(material: THREE.Material): number {
  const v = (material as EmissiveMaterial).userData.oakEmissiveIntensity;
  return typeof v === 'number' ? v : Number.NaN;
}

/** The rig's point lights, in the order world.ts claims them. */
function pointLights(module: SceneModule): THREE.PointLight[] {
  const out: THREE.PointLight[] = [];
  module.scene.traverse((o) => {
    if (o instanceof THREE.PointLight) out.push(o);
  });
  return out;
}

/** the guard's own "I am holding this frame" flag */
function isHolding(module: SceneModule): boolean {
  return (module as unknown as { wrapHeld: boolean }).wrapHeld;
}

/** the paper's emissive gain the scene WOULD show for a state, via a scrubbed frame */
function emissiveGainOf(module: SceneModule, th: number, thd: number, u: number): number {
  module.update(toFrameUpdate(poseFixture('probe', th, thd, u, 0)));
  return emissiveGain(meshNamed(module, 'lanternPaper').material as THREE.Material);
}

/** the angle the scene actually RENDERED, read back off the bob probe */
function renderedTheta(module: SceneModule): number {
  const bob = worldOf(module, 'bob');
  return Math.atan2(bob.x, -bob.y);
}

// ---------------------------------------------------------------------------
// the env, transcribed — `gw_pend_step`, Sources/ClassicControl.metal L189-200
// ---------------------------------------------------------------------------

interface Row {
  th: number;
  thd: number;
  u: number;
}

/** one control step, in the kernel's own order (θ integrates the PRE-clamp velocity) */
function pendStep(th: number, thd: number, ctrl0: number): Row {
  const u = Math.max(-PD_MAX_TORQUE, Math.min(PD_MAX_TORQUE, 2 * ctrl0));
  let nthd = thd + (((1.5 * PD_G) / PD_L) * Math.sin(th + Math.PI) + (3 * u) / (PD_M * PD_L * PD_L)) * PD_DT;
  const next = wrapPi(th + nthd * PD_DT);
  nthd = Math.max(-PD_MAX_SPEED, Math.min(PD_MAX_SPEED, nthd));
  // the recorder writes f32, and the host hands the scene a Float32Array
  return { th: Math.fround(next), thd: Math.fround(nthd), u: Math.fround(u) };
}

function simulate(th0: number, thd0: number, ctrl: (i: number) => number, n: number): Row[] {
  const rows: Row[] = [{ th: Math.fround(wrapPi(th0)), thd: Math.fround(thd0), u: 0 }];
  for (let i = 1; i <= n; i++) rows.push(pendStep(rows[i - 1]!.th, rows[i - 1]!.thd, ctrl(i)));
  return rows;
}

/** rows that contain a ±π crossing (the host's linear lerp fabricates a revolution here) */
function wrapRows(rows: readonly Row[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < rows.length; i++) if (Math.abs(rows[i]!.th - rows[i - 1]!.th) > Math.PI) out.push(i - 1);
  return out;
}

interface ReplayReport {
  frames: number;
  /** worst |rendered − physically-correct short-arc angle|, in radians */
  maxError: number;
  /** the highest cos-distance the render ever put between itself and the truth */
  worstAt: number;
  holdFrames: number;
  holdRuns: number;
  maxHoldRun: number;
  /** lowest rendered bob y — a false revolution dives to −1 */
  minY: number;
  /** lowest rendered bob y over the frames inside a genuine wrap step */
  minYAtWrap: number;
}

/**
 * Replay `rows` through the scene the way the HOST does: `kit/harness.ts` advances a
 * fractional frame position by `controlHz · speed · dtReal` rows per render frame, samples
 * the 9-float pose by LINEAR interpolation between the two neighbouring rows, and reports
 * `state.step` as the FLOOR of that position. `rowsPerFrame` is the whole sampling
 * regime: 1/3 is 60 fps at 1×, 1/24 is 120 fps at 0.25×, 1.33 is 60 fps at 4×.
 */
function replay(module: SceneModule, rows: readonly Row[], rowsPerFrame: number): ReplayReport {
  const wraps = new Set(wrapRows(rows));
  const last = rows.length - 1;
  const report: ReplayReport = {
    frames: 0,
    maxError: 0,
    worstAt: 0,
    holdFrames: 0,
    holdRuns: 0,
    maxHoldRun: 0,
    minY: Infinity,
    minYAtWrap: Infinity,
  };
  let run = 0;
  let prevHeld = false;
  for (let pos = 0; pos <= last; pos += rowsPerFrame) {
    const i = Math.min(last, Math.floor(pos));
    const j = Math.min(last, i + 1);
    const a = pos - i;
    const A = rows[i] as Row;
    const B = rows[j] as Row;
    // the host's LINEAR lerp — the bug the guard exists to refuse
    const th = A.th + (B.th - A.th) * a;
    const thd = A.thd + (B.thd - A.thd) * a;
    const u = A.u + (B.u - A.u) * a;
    module.update(toFrameUpdate(poseFixture(`f${report.frames}`, th, thd, u, i), { scrubbed: report.frames === 0 }));
    report.frames++;

    const held = isHolding(module);
    if (held) {
      report.holdFrames++;
      run++;
      if (!prevHeld) report.holdRuns++;
      report.maxHoldRun = Math.max(report.maxHoldRun, run);
    } else run = 0;
    prevHeld = held;

    // the physically correct sub-step angle: the SHORT ARC on the circle
    const truth = wrapPi(A.th + wrapPi(B.th - A.th) * a);
    const err = Math.abs(wrapPi(renderedTheta(module) - truth));
    if (err > report.maxError) {
      report.maxError = err;
      report.worstAt = pos;
    }
    const y = worldOf(module, 'bob').y;
    report.minY = Math.min(report.minY, y);
    if (wraps.has(i)) report.minYAtWrap = Math.min(report.minYAtWrap, y);
  }
  return report;
}

/** every sampling density the transport can produce: 0.25×…4× across 30…240 Hz */
const DENSITIES: readonly number[] = [1 / 48, 1 / 24, 1 / 12, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 1, 4 / 3, 8 / 3];

/**
 * The lowest bob y a frame inside a WRAP row may render. A wrap row's own angle is within
 * one control step of ±π, and the guard keeps the render within 2·PD_MAX_DTHETA + slack of
 * the truth, so the bob is pinned to a cap around the top of its arc. The fabricated
 * revolution reaches −1.
 */
const TOP_CAP_Y = -Math.cos(Math.PI - 3 * PD_MAX_DTHETA - 0.02);

// ---------------------------------------------------------------------------

describe('pendulum / manifest + fixtures', () => {
  test('manifest declares the frozen contract fields', () => {
    expect(manifest.envId).toBe('Pendulum-cont-v0');
    expect(manifest.slug).toBe('pendulum');
    expect(manifest.accent).toBe('pendulum');
    expect([...manifest.tiers].sort()).toEqual(['hero', 'medium', 'thumb']);
    expect(manifest.thumbFrame.pose.length).toBe(9);
    expect(manifest.postConfig.bloomStrength).toBeGreaterThanOrEqual(0.25);
    expect(manifest.postConfig.bloomStrength).toBeLessThanOrEqual(0.9);
    expect(manifest.budget.drawCalls).toBeLessThanOrEqual(300);
    expect(manifest.budget.triangles).toBeLessThanOrEqual(500_000);
  });

  test('the thumbFrame is a legal, reachable pendulum state', () => {
    const p = manifest.thumbFrame.pose;
    expect(Math.abs(p[0] ?? 9)).toBeLessThanOrEqual(Math.PI);
    expect(Math.abs(p[1] ?? 99)).toBeLessThanOrEqual(PD_MAX_SPEED);
    expect(Math.abs(p[2] ?? 99)).toBeLessThanOrEqual(PD_MAX_TORQUE);
    // and it is a handsome one: the lantern lit, but visibly off the vertical
    expect(serenityOf(p[0] ?? 0, p[1] ?? 0, p[2] ?? 0)).toBeGreaterThan(0.8);
    expect(Math.abs(p[0] ?? 0)).toBeGreaterThan(0.2);
  });

  test('ships at least three canonical poses, all inside the env limits', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(3);
    for (const f of fixtures) {
      expect(f.pose.length).toBe(9);
      expect(Math.abs(f.pose[0] ?? 9)).toBeLessThanOrEqual(Math.PI);
      expect(Math.abs(f.pose[1] ?? 99)).toBeLessThanOrEqual(PD_MAX_SPEED);
      expect(Math.abs(f.pose[2] ?? 99)).toBeLessThanOrEqual(PD_MAX_TORQUE);
    }
  });

  test('the fixtures cover the cost minimum, the inverted init, both torque signs, and both wraps', () => {
    expect(fixtures.some((f) => (f.pose[0] ?? 1) === 0)).toBe(true);
    expect(fixtures.some((f) => Math.abs(f.pose[0] ?? 0) === Math.PI)).toBe(true);
    expect(fixtures.some((f) => (f.pose[2] ?? 0) > 0.5)).toBe(true);
    expect(fixtures.some((f) => (f.pose[2] ?? 0) < -0.5)).toBe(true);
    // the two wrap edges are the guard's two different regimes: one caught by the
    // direction test (|θ̇| well above the sign-flip bound), one only by the reach bound
    const fast = fixtureNamed('wrap-edge');
    const slow = fixtureNamed('wrap-edge-slow');
    expect(Math.abs(fast.pose[1] ?? 0)).toBeGreaterThan(PD_MAX_DTHETADOT);
    expect(Math.abs(slow.pose[1] ?? 9)).toBeLessThan(PD_MAX_DTHETADOT);
    expect(Math.abs(fast.pose[0] ?? 0)).toBeGreaterThan(3);
    expect(Math.abs(slow.pose[0] ?? 0)).toBeGreaterThan(3);
  });
});

describe("pendulum / the kinematics are the decoder's, never re-derived", () => {
  test('the scene rod length IS the env rod length', () => {
    expect(ROD_LEN).toBe(PD_L);
  });

  test('the still-water plane sits below the bob at every angle', () => {
    // the lantern must never dip into the pond, at any θ
    expect(WATER_LEVEL).toBeLessThan(-PD_L - 0.25);
  });

  test("the pond bed really is a bowl, and its shoreline is inside the arch's span", () => {
    const bed = makePondBed(SEED);
    expect(bed(0, 0)).toBeLessThan(WATER_LEVEL - 0.6); // deep water under the lantern
    expect(bed(POND_RX * 1.6, 0)).toBeGreaterThan(WATER_LEVEL); // dry bank beyond the rim
    // the water's edge is somewhere between half and one ellipse-radius out
    let crossing = Number.NaN;
    for (let i = 1; i <= 400; i++) {
      const x = (POND_RX * 1.6 * i) / 400;
      if (bed(x, 0) > WATER_LEVEL) {
        crossing = x;
        break;
      }
    }
    expect(crossing).toBeGreaterThan(POND_RX * 0.5);
    expect(crossing).toBeLessThan(POND_RX * 1.25);
  });
});

describe('pendulum / alignment', () => {
  test('every probe lands on ground truth at every tier', async () => {
    for (const tier of ['hero', 'medium', 'thumb'] as const) {
      const module = createScene();
      const report = await assertAligned(module, fixtures, { geometry: geometryOf(), seed: SEED, tier });
      // probe roots carry POSITION ONLY — these should be exact, not merely within 1e-3
      expect(report.worst.get('bob') ?? 1).toBeLessThan(1e-12);
      expect(report.worst.get('swing-root') ?? 1).toBe(0);
      expect(report.worst.get('pivot') ?? 1).toBe(0);
      expect(report.worst.get('lantern-reflection') ?? 1).toBeLessThan(1e-12);
      module.dispose();
    }
  });

  test("the bob probe traces the env's own unit circle, all the way round", async () => {
    const { module } = await live();
    const tip = new Float32Array(2);
    for (let i = 0; i <= 180; i++) {
      const th = -Math.PI + (2 * Math.PI * i) / 180;
      module.update(toFrameUpdate(poseFixture(`sweep${i}`, th, 0, 0, i)));
      const world = worldOf(module, 'bob');
      // the pose arrives as f32, so ground truth is the f32 round-trip of θ
      const truth = Math.fround(th);
      expect(world.x).toBeCloseTo(PD_L * Math.sin(truth), 12);
      expect(world.y).toBeCloseTo(-PD_L * Math.cos(truth), 12);
      expect(world.z).toBeCloseTo(0, 12);
      // and it is the decoder's own kinematics (which rounds its output to f32)
      pendulumTip(truth, tip);
      expect(world.x).toBeCloseTo(tip[0] ?? 0, 6);
      expect(world.y).toBeCloseTo(tip[1] ?? 0, 6);
      // and the radius is exactly the rod length, everywhere
      expect(Math.hypot(world.x, world.y)).toBeCloseTo(PD_L, 12);
    }
    module.dispose();
  });

  test('bobRoot is position-only; the lantern tilt lives on a child', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('swing-right')));
    const bob = module.scene.getObjectByName('bobRoot');
    const tilt = module.scene.getObjectByName('bobTilt');
    expect(bob).toBeDefined();
    expect(tilt).toBeDefined();
    if (!bob || !tilt) return;
    expect(bob.rotation.x).toBe(0);
    expect(bob.rotation.y).toBe(0);
    expect(bob.rotation.z).toBe(0);
    expect(bob.scale.x).toBe(1);
    expect(tilt.position.lengthSq()).toBe(0);
    expect(Math.abs(tilt.rotation.z)).toBeGreaterThan(0.5); // the tilt really is carried here
    module.dispose();
  });

  test('the swing node maps its local down-vector exactly onto the bob', async () => {
    const { module } = await live();
    const down = new THREE.Vector3();
    for (let i = 0; i <= 72; i++) {
      const th = -Math.PI + (2 * Math.PI * i) / 72;
      module.update(toFrameUpdate(poseFixture(`swing${i}`, th, 0, 0, i)));
      const swing = module.scene.getObjectByName('swingRoot');
      expect(swing).toBeDefined();
      if (!swing) return;
      swing.updateWorldMatrix(true, false);
      // rotation only: the node itself never leaves the pivot
      expect(swing.scale.x).toBe(1);
      down.set(0, -PD_L, 0).applyMatrix4(swing.matrixWorld);
      const bob = worldOf(module, 'bob');
      expect(down.x).toBeCloseTo(bob.x, 9);
      expect(down.y).toBeCloseTo(bob.y, 9);
      expect(down.z).toBeCloseTo(bob.z, 9);
    }
    module.dispose();
  });

  test("the reflection rides the lantern's abscissa on the still-water plane", async () => {
    const { module } = await live();
    for (const f of fixtures) {
      module.update(toFrameUpdate(f));
      const bob = worldOf(module, 'bob');
      const reflection = worldOf(module, 'lantern-reflection');
      expect(reflection.x).toBe(bob.x);
      expect(reflection.y).toBe(WATER_LEVEL + REFLECT_LIFT);
      expect(reflection.z).toBe(REFLECT_Z);
    }
    module.dispose();
  });
});

// ---------------------------------------------------------------------------
// the arch is built around what the LANTERN sweeps — not around the bare rod
// ---------------------------------------------------------------------------

interface Envelope {
  /** closest and farthest material from the rotation centre, in the x-y plane */
  rIn: number;
  rOut: number;
  /** half-thickness across the swing plane (rotation about z never changes z) */
  halfZ: number;
}

/** the true swept envelope of a set of meshes about a local origin, measured, not declared */
function envelopeOf(module: SceneModule, names: readonly string[], origin: THREE.Vector3): Envelope {
  const v = new THREE.Vector3();
  let rIn = Infinity;
  let rOut = 0;
  let halfZ = 0;
  for (const name of names) {
    const mesh = meshNamed(module, name);
    mesh.updateWorldMatrix(true, false);
    const pos = mesh.geometry.getAttribute('position');
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld).sub(origin);
      const r = Math.hypot(v.x, v.y);
      rIn = Math.min(rIn, r);
      rOut = Math.max(rOut, r);
      halfZ = Math.max(halfZ, Math.abs(v.z));
    }
  }
  return { rIn, rOut, halfZ };
}

/**
 * Triangles of `mesh` that intersect the annulus (rIn, rOut) × (−halfZ, halfZ) swept about
 * the pivot's z-axis, tested by AABB.
 *
 * A VERTEX test is not enough and that is not a detail: a box face can cross the swept
 * volume with all eight of its corners outside it. The hanger plate is exactly that shape
 * — it spans ±z past the barrel while its underside sits inside the lantern's reach.
 */
function sweptHits(mesh: THREE.Mesh, halfZ: number, rIn: number, rOut: number): number {
  const pos = mesh.geometry.getAttribute('position');
  let hits = 0;
  for (let t = 0; t + 2 < pos.count; t += 3) {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    let z0 = Infinity;
    let z1 = -Infinity;
    for (let k = 0; k < 3; k++) {
      const x = pos.getX(t + k);
      const y = pos.getY(t + k);
      const z = pos.getZ(t + k);
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
      z0 = Math.min(z0, z);
      z1 = Math.max(z1, z);
    }
    if (!(z0 < halfZ && z1 > -halfZ)) continue;
    const rmin = Math.hypot(Math.max(x0, 0, -x1), Math.max(y0, 0, -y1));
    const rmax = Math.max(Math.hypot(x0, y0), Math.hypot(x0, y1), Math.hypot(x1, y0), Math.hypot(x1, y1));
    if (rmin < rOut && rmax > rIn) hits++;
  }
  return hits;
}

/** the same AABB test for one axis-aligned box, so the test can prove it has teeth */
function boxHitsSwept(
  w: number,
  h: number,
  d: number,
  x: number,
  y: number,
  z: number,
  halfZ: number,
  rIn: number,
  rOut: number,
): boolean {
  if (!(z - d / 2 < halfZ && z + d / 2 > -halfZ)) return false;
  const x0 = x - w / 2;
  const x1 = x + w / 2;
  const y0 = y - h / 2;
  const y1 = y + h / 2;
  const rmin = Math.hypot(Math.max(x0, 0, -x1), Math.max(y0, 0, -y1));
  const rmax = Math.max(Math.hypot(x0, y0), Math.hypot(x0, y1), Math.hypot(x1, y0), Math.hypot(x1, y1));
  return rmin < rOut && rmax > rIn;
}

describe('pendulum / the arch is built around the swept lantern', () => {
  test("dims.ts describes the lantern that actors.ts actually builds", async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('at-rest')));
    const lantern = envelopeOf(module, ['lanternFrame', 'lanternPaper', 'lanternFlame'], new THREE.Vector3(0, -ROD_LEN, 0));
    // the declared envelope is the built one — this is what stops the arch and the
    // lantern drifting apart again
    expect(lantern.rOut).toBeCloseTo(LANTERN_REACH, 6);
    expect(lantern.halfZ).toBeCloseTo(LANTERN_HALF_Z, 6);
    expect(SWING_REACH).toBeCloseTo(ROD_LEN + LANTERN_REACH, 12);
    expect(SWING_INNER).toBeCloseTo(ROD_LEN - LANTERN_REACH, 12);
    module.dispose();
  });

  test('no torii geometry sits inside the volume the lantern sweeps', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('at-rest')));
    const torii = meshNamed(module, 'torii');
    const lantern = envelopeOf(module, ['lanternFrame', 'lanternPaper', 'lanternFlame'], new THREE.Vector3(0, -ROD_LEN, 0));
    // measured envelope, every z, no exemptions
    expect(sweptHits(torii, lantern.halfZ, ROD_LEN - lantern.rOut, ROD_LEN + lantern.rOut)).toBe(0);
    module.dispose();
  });

  test('no torii geometry sits inside the volume the ROD sweeps either', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('at-rest')));
    const torii = meshNamed(module, 'torii');
    const rod = envelopeOf(module, ['lanternRod'], new THREE.Vector3(0, 0, 0));
    expect(sweptHits(torii, rod.halfZ, rod.rIn, rod.rOut)).toBe(0);
    // …which is only possible because the pin passes through the eye's HOLE
    expect(PIN_R).toBeLessThan(rod.rIn);
    expect(EYE_MAJOR - EYE_TUBE).toBeGreaterThan(PIN_R);
    module.dispose();
  });

  test('this clearance test has teeth: the pre-fix hanger would fail it', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('at-rest')));
    const lantern = envelopeOf(module, ['lanternFrame', 'lanternPaper', 'lanternFlame'], new THREE.Vector3(0, -ROD_LEN, 0));
    const rIn = ROD_LEN - lantern.rOut;
    const rOut = ROD_LEN + lantern.rOut;
    // the shipped hanger plate: 0.24 × 0.03 × 0.26 under a nuki at y = 1.30
    expect(boxHitsSwept(0.24, 0.03, 0.26, 0, 1.238, 0, lantern.halfZ, rIn, rOut)).toBe(true);
    // the shipped cables: z = ±0.105, inside a 0.135 barrel
    expect(boxHitsSwept(0.016, 1.22, 0.016, 0, 0.61, 0.105, lantern.halfZ, rIn, rOut)).toBe(true);
    // the shipped tie beam itself
    expect(boxHitsSwept(3.9, 0.1, 0.3, 0, 1.3, 0, lantern.halfZ, rIn, rOut)).toBe(true);
    module.dispose();
  });

  test('the hanger cables straddle the barrel, and the beam clears the tassel', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('inverted-init')));
    // the near face of a cable is outside the widest part of the paper barrel
    expect(STRAP_Z - STRAP_HALF).toBeGreaterThan(LANTERN_HALF_Z);
    // and the lowest static thing in the swing plane really does clear the tassel's apex
    const torii = meshNamed(module, 'torii');
    const pos = torii.geometry.getAttribute('position');
    let lowestInPlane = Infinity;
    for (let i = 0; i < pos.count; i++) {
      if (Math.abs(pos.getZ(i)) >= LANTERN_HALF_Z) continue;
      const x = pos.getX(i);
      const y = pos.getY(i);
      if (y <= 0) continue; // the pillars run all the way down; only the crown is overhead
      lowestInPlane = Math.min(lowestInPlane, Math.hypot(x, y));
    }
    expect(lowestInPlane).toBeGreaterThanOrEqual(SWING_REACH + SWING_CLEARANCE);
    // and the bob at θ = π really is up there, under it
    expect(worldOf(module, 'bob').y).toBeCloseTo(ROD_LEN, 6);
    module.dispose();
  });

  test('the hanger straps straddle the swing plane in ±z', async () => {
    const { module } = await live();
    const torii = meshNamed(module, 'torii');
    const pos = torii.geometry.getAttribute('position');
    let front = 0;
    let back = 0;
    for (let i = 0; i < pos.count; i++) {
      // near the pivot in x/y, but off the plane in z: that is the hanger
      if (Math.hypot(pos.getX(i), pos.getY(i)) > 0.6) continue;
      const z = pos.getZ(i);
      if (z > STRAP_Z * 0.8) front++;
      else if (z < -STRAP_Z * 0.8) back++;
    }
    expect(front).toBeGreaterThan(0);
    expect(back).toBeGreaterThan(0);
    module.dispose();
  });
});

// ---------------------------------------------------------------------------
// the angle-wrap trap, at every sampling density the transport can produce
// ---------------------------------------------------------------------------

describe('pendulum / the angle-wrap trap', () => {
  test("the guard's reach constants are the env's own, and they bound it", () => {
    // brute-force the integrator over the whole reachable state space
    let worstDTheta = 0;
    let worstDThetaDot = 0;
    for (let ti = 0; ti <= 64; ti++) {
      const th = -Math.PI + (2 * Math.PI * ti) / 64;
      for (let vi = 0; vi <= 64; vi++) {
        const thd = -PD_MAX_SPEED + (2 * PD_MAX_SPEED * vi) / 64;
        for (const c of [-1, 0, 1]) {
          const next = pendStep(th, thd, c);
          worstDTheta = Math.max(worstDTheta, Math.abs(wrapPi(next.th - th)));
          worstDThetaDot = Math.max(worstDThetaDot, Math.abs(next.thd - thd));
        }
      }
    }
    // (the recorder writes f32, so the bound holds to a float32 epsilon, not exactly —
    // which is a thousand times inside the guard's own slack)
    expect(worstDTheta).toBeLessThanOrEqual(PD_MAX_DTHETA * (1 + 1e-6));
    expect(worstDThetaDot).toBeLessThanOrEqual(PD_MAX_DTHETADOT * (1 + 1e-6));
    // and they are TIGHT — a bound nowhere near the truth would hide real jumps
    expect(worstDTheta).toBeGreaterThan(PD_MAX_DTHETA * 0.85);
    // the shipped constant (PD_MAX_SPEED · PD_DT) was BELOW the env's real reach
    expect(PD_MAX_SPEED * PD_DT).toBeLessThan(worstDTheta);
    // the reach is monotone in both arguments and never over-claims
    expect(wrapReach(0, 1)).toBeLessThan(wrapReach(PD_MAX_SPEED, 1));
    expect(wrapReach(PD_MAX_SPEED, 1)).toBeLessThanOrEqual(PD_MAX_DTHETA + 0.02 + 1e-12);
    expect(wrapReach(PD_MAX_SPEED, 2)).toBeGreaterThan(wrapReach(PD_MAX_SPEED, 1));
  });

  test('the fabricated revolution is refused at EVERY sampling density', async () => {
    // a real swing-up that rotates through ±π four times under full positive torque
    const rows = simulate(2.4, 3.2, () => 1, 60);
    expect(wrapRows(rows).length).toBeGreaterThanOrEqual(4);
    for (const rowsPerFrame of DENSITIES) {
      const { module } = await live();
      const report = replay(module, rows, rowsPerFrame);
      // the render never leaves the neighbourhood the env could actually have reached
      expect(report.maxError).toBeLessThanOrEqual(2 * PD_MAX_DTHETA + 0.02);
      // and above all: the lantern stays at the TOP through the crossing — it never
      // swings down through the bottom of its arc, which is the fabricated revolution.
      // A wrap row's true angle is within one step of ±π, and the render is within the
      // bound above of that, so the bob cannot leave this cap. (Measured: 0.849, i.e.
      // four times better than the bound; the shipped guard reached −1.000.)
      expect(report.minYAtWrap).toBeGreaterThan(TOP_CAP_Y);
      // no more hold runs than there are wraps to refuse: the shipped guard produced 6
      // where the lane has 2, freezing honest motion
      expect(report.holdRuns).toBeLessThanOrEqual(wrapRows(rows).length);
      module.dispose();
    }
  });

  test('the SLOW crossing (lane 56 row 3) is refused too — reach alone catches it', async () => {
    // tipping over the top at |θ̇| ≈ 0.06: far below the speed the direction test needs,
    // so the reach bound is the only thing standing between this and a false revolution
    const rows = simulate(3.1377, 0.0133, () => 0.15, 40);
    const first = wrapRows(rows)[0] ?? -1;
    expect(first).toBeGreaterThanOrEqual(0);
    expect(Math.abs((rows[first] as Row).thd)).toBeLessThan(PD_MAX_DTHETADOT);
    for (const rowsPerFrame of [1 / 24, 1 / 12, 1 / 3]) {
      const { module } = await live();
      const report = replay(module, rows, rowsPerFrame);
      expect(report.maxError).toBeLessThanOrEqual(2 * PD_MAX_DTHETA + 0.02);
      expect(report.minYAtWrap).toBeGreaterThan(TOP_CAP_Y);
      expect(report.holdRuns).toBeLessThanOrEqual(wrapRows(rows).length);
      module.dispose();
    }
  });

  test('honest motion is NEVER held, at any density — including fast transport', async () => {
    // the fastest legal sweep that stays off the branch cut: full torque from the bottom
    const rows = simulate(0, 0, () => 1, 30);
    expect(wrapRows(rows).length).toBe(0);
    for (const rowsPerFrame of DENSITIES) {
      const { module } = await live();
      const report = replay(module, rows, rowsPerFrame);
      expect(report.holdFrames).toBe(0);
      expect(report.maxError).toBeLessThan(1e-6);
      module.dispose();
    }
    // and a long free swing through the bottom at the speed clip
    const fast = simulate(-2.5, 7.5, () => 0, 40);
    for (const rowsPerFrame of DENSITIES) {
      const { module } = await live();
      const report = replay(module, fast, rowsPerFrame);
      expect(report.maxError).toBeLessThanOrEqual(2 * PD_MAX_DTHETA + 0.02);
      expect(report.holdRuns).toBeLessThanOrEqual(wrapRows(fast).length);
      // the bob really does pass through the bottom on a free swing — the guard is not
      // just refusing everything below the horizontal
      expect(report.minY).toBeLessThan(-0.9);
      module.dispose();
    }
  });

  test('a hold never outlasts MAX_HOLD_STEPS control steps', async () => {
    const { module } = await live();
    // an angle that is unreachable forever: the guard must still let go
    module.update(toFrameUpdate(poseFixture('anchor', 0, 0, 0, 0)));
    let held = 0;
    for (let step = 1; step <= MAX_HOLD_STEPS + 4; step++) {
      module.update(toFrameUpdate(poseFixture(`far${step}`, Math.PI, 0, 0, step), { scrubbed: false }));
      if (isHolding(module)) held++;
    }
    expect(held).toBeLessThanOrEqual(MAX_HOLD_STEPS);
    expect(isHolding(module)).toBe(false);
    // …and it landed on the truth, not on the anchor
    expect(worldOf(module, 'bob').y).toBeCloseTo(PD_L, 6);
    module.dispose();
  });

  test('the recorded lane-56 wrap holds at the apex, then releases onto the truth', async () => {
    const { module } = await live();
    const A = 3.088365; // row 39
    const B = -3.06844; // row 40 — +0.127 rad THROUGH the top
    module.update(toFrameUpdate(poseFixture('wrapA', A, 2.621751, -0.723766, 39)));
    const apex = worldOf(module, 'bob');
    expect(apex.y).toBeGreaterThan(0.9); // it really is near the top

    // every frame the host can produce inside that one control step, at 24 per row
    for (let k = 1; k < 24; k++) {
      const a = k / 24;
      module.update(
        toFrameUpdate(poseFixture(`lerp${k}`, A + (B - A) * a, 2.62 - 0.09 * a, -0.72 + 0.36 * a, 39), {
          scrubbed: false,
        }),
      );
      const held = worldOf(module, 'bob');
      expect(held.y).toBeGreaterThan(0.9);
      expect(Math.abs(wrapPi(renderedTheta(module) - Math.fround(A)))).toBeLessThanOrEqual(PD_MAX_DTHETA + 0.02);
    }

    // the lerp lands on record row 40: reachable on the circle (0.127 rad), so released
    module.update(toFrameUpdate(poseFixture('wrapB', B, 2.52761, -0.361593, 40), { scrubbed: false }));
    expect(isHolding(module)).toBe(false);
    const after = worldOf(module, 'bob');
    expect(after.x).toBeCloseTo(PD_L * Math.sin(Math.fround(B)), 9);
    expect(after.y).toBeCloseTo(-PD_L * Math.cos(Math.fround(B)), 9);
    module.dispose();
  });

  test('a physically legal crossing of ±π is NEVER held', async () => {
    const { module } = await live();
    // walk the circle at the env's own maximum rate, straight through the branch cut
    const rate = PD_MAX_SPEED * PD_DT;
    let th = 2.0;
    module.update(toFrameUpdate(poseFixture('start', th, PD_MAX_SPEED, 0, 0)));
    for (let i = 1; i <= 40; i++) {
      th = wrapPi(th + rate);
      module.update(toFrameUpdate(poseFixture(`walk${i}`, th, PD_MAX_SPEED, 0, i), { scrubbed: false }));
      expect(isHolding(module)).toBe(false);
      const bob = worldOf(module, 'bob');
      const truth = Math.fround(th);
      expect(bob.x).toBeCloseTo(PD_L * Math.sin(truth), 6);
      expect(bob.y).toBeCloseTo(-PD_L * Math.cos(truth), 6);
    }
    module.dispose();
  });

  test('a scrub releases the guard on the spot (hard snap, never an ease)', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(poseFixture('wrapA', 3.088, 2.62, 0, 39)));
    // an unreachable angle on a SEQUENTIAL frame is held...
    module.update(toFrameUpdate(poseFixture('mid', 0.1, 2.6, 0, 39), { scrubbed: false }));
    expect(isHolding(module)).toBe(true);
    expect(worldOf(module, 'bob').y).toBeGreaterThan(0.9);
    // ...and the same angle on a SCRUBBED frame snaps straight there
    module.update(toFrameUpdate(poseFixture('mid', 0.1, 2.6, 0, 39), { scrubbed: true }));
    expect(isHolding(module)).toBe(false);
    const snapped = worldOf(module, 'bob');
    expect(snapped.y).toBeCloseTo(-PD_L * Math.cos(Math.fround(0.1)), 9);
    expect(snapped.y).toBeLessThan(-0.9);
    // and the scrub re-anchored: the next sequential frame is judged from HERE
    module.update(toFrameUpdate(poseFixture('after', 0.2, 2.6, 0, 40), { scrubbed: false }));
    expect(isHolding(module)).toBe(false);
    module.dispose();
  });

  test('the light follows the HELD angle, so the flame never flashes through a wrap', async () => {
    const { module } = await live();
    const paper = (): number => emissiveGain(meshNamed(module, 'lanternPaper').material as THREE.Material);
    module.update(toFrameUpdate(poseFixture('apex', 3.088365, 2.621751, -0.723766, 39)));
    const atApex = paper();
    // the lerp's midpoint is θ ≈ 0 — the COST MINIMUM. If the scene lit the raw angle the
    // pond would flare to full brightness in the middle of a wrap.
    module.update(toFrameUpdate(poseFixture('mid', 0.01, 2.57, -0.54, 39), { scrubbed: false }));
    expect(isHolding(module)).toBe(true);
    // the flame tracks the HELD angle: it drifts only by the honest θ̇/torque lerp, and
    // stays nowhere near the blaze the raw θ ≈ 0 would have produced
    expect(Math.abs(paper() - atApex)).toBeLessThan(0.05 * atApex);
    expect(paper()).toBeLessThan(0.25 * emissiveGainOf(module, 0, 0, 0));
    module.dispose();
  });
});

// ---------------------------------------------------------------------------
// the shaped reward IS the lighting
// ---------------------------------------------------------------------------

describe('pendulum / the shaped reward IS the lighting', () => {
  test("pendulumCost is the env's own expression", () => {
    // an independent transcription of `gw_pend_step`'s cost line
    for (let i = 0; i <= 40; i++) {
      const th = -Math.PI + (2 * Math.PI * i) / 40;
      for (const thd of [-8, -3.3, 0, 1.7, 8]) {
        for (const u of [-2, 0, 0.9, 2]) {
          const thn = wrapPi(th);
          expect(pendulumCost(th, thd, u)).toBeCloseTo(thn * thn + 0.1 * thd * thd + 0.001 * u * u, 12);
        }
      }
    }
    // it is a function ON THE CIRCLE: θ and θ ± 2π are the same state
    expect(pendulumCost(3, 1, 0)).toBeCloseTo(pendulumCost(3 - 2 * Math.PI, 1, 0), 12);
    expect(COST_MAX).toBeCloseTo(Math.PI ** 2 + 0.1 * PD_MAX_SPEED ** 2 + 0.001 * PD_MAX_TORQUE ** 2, 12);
  });

  test('A BETTER-SCORED STATE IS NEVER DARKER — over the whole state space', () => {
    interface Sample {
      cost: number;
      serenity: number;
      label: string;
    }
    const samples: Sample[] = [];
    for (let i = 0; i <= 24; i++) {
      const th = -Math.PI + (2 * Math.PI * i) / 24;
      for (let v = 0; v <= 16; v++) {
        const thd = -PD_MAX_SPEED + (2 * PD_MAX_SPEED * v) / 16;
        for (const u of [-PD_MAX_TORQUE, 0, PD_MAX_TORQUE]) {
          samples.push({
            cost: pendulumCost(th, thd, u),
            serenity: serenityOf(th, thd, u),
            label: `θ=${th.toFixed(2)} θ̇=${thd.toFixed(2)} u=${u}`,
          });
        }
      }
    }
    samples.sort((a, b) => a.cost - b.cost);
    for (let i = 1; i < samples.length; i++) {
      const lo = samples[i - 1] as Sample;
      const hi = samples[i] as Sample;
      if (!(lo.serenity >= hi.serenity - 1e-12)) {
        throw new Error(
          `inversion: ${lo.label} costs ${lo.cost.toFixed(3)} and renders ${lo.serenity.toFixed(3)}, ` +
            `while ${hi.label} costs ${hi.cost.toFixed(3)} and renders ${hi.serenity.toFixed(3)}`,
        );
      }
    }
    // and it is not degenerate: the range really is used
    expect((samples[0] as Sample).serenity).toBeGreaterThan(0.99);
    expect((samples[samples.length - 1] as Sample).serenity).toBeLessThan(0.01);
  });

  test('the specific inversions the old product law rendered are gone', () => {
    // measured on the shipped law: (θ=0, θ̇=8) cost 6.40 rendered 0.620, while
    // (θ=2.4, θ̇=0) cost 5.76 — a BETTER state — rendered 0.131, i.e. 4.7× darker
    expect(pendulumCost(0, 8, 0)).toBeGreaterThan(pendulumCost(2.4, 0, 0));
    expect(serenityOf(0, 8, 0)).toBeLessThan(serenityOf(2.4, 0, 0));
    expect(pendulumCost(0, 8, 0)).toBeGreaterThan(pendulumCost(2.0, 0, 0));
    expect(serenityOf(0, 8, 0)).toBeLessThan(serenityOf(2.0, 0, 0));
    // the cost minimum is the brightest frame in the scene, exactly
    expect(serenityOf(0, 0, 0)).toBe(1);
    // …and the darkest is not merely "inverted": it is inverted AND spinning AND pushing
    expect(serenityOf(Math.PI, PD_MAX_SPEED, PD_MAX_TORQUE)).toBe(0);
    expect(serenityOf(Math.PI, 0, 0)).toBeGreaterThan(0);
    expect(serenityOf(Math.PI, 0, 0)).toBeLessThan(serenityOf(0, PD_MAX_SPEED, 0));
    // the velocity term carries real weight — a third of the mean cost on the real lane
    expect(serenityOf(0, PD_MAX_SPEED, 0)).toBeLessThan(0.4);
  });

  test('the flame, the point lights and the reflection all follow it', async () => {
    const { module } = await live();
    const read = (): { paper: number; flame: number; lights: number; reflection: number } => {
      const paper = meshNamed(module, 'lanternPaper');
      const flame = meshNamed(module, 'lanternFlame');
      const reflection = meshNamed(module, 'lanternReflection');
      let lights = 0;
      for (const l of pointLights(module)) lights += l.intensity;
      return {
        paper: emissiveGain(paper.material as THREE.Material),
        flame: emissiveGain(flame.material as THREE.Material),
        lights,
        reflection: (reflection.material as THREE.Material).opacity,
      };
    };

    module.update(toFrameUpdate(fixtureNamed('at-rest')));
    const lit = read();
    module.update(toFrameUpdate(fixtureNamed('inverted-init')));
    const dark = read();

    expect(lit.paper).toBeGreaterThan(dark.paper * 4);
    expect(lit.flame).toBeGreaterThan(dark.flame * 4);
    expect(lit.lights).toBeGreaterThan(dark.lights * 4);
    expect(lit.reflection).toBeGreaterThan(dark.reflection * 2);
    // the inverted pose is a dim ember, well under the scene's bloom threshold
    expect(dark.paper).toBeLessThan(manifest.postConfig.bloomThreshold);
    // and the ordering is the reward's, everywhere in between
    let prev = Infinity;
    for (let i = 0; i <= 16; i++) {
      module.update(toFrameUpdate(poseFixture(`ramp${i}`, (Math.PI * i) / 16, 0, 0, i)));
      const gain = read().paper;
      expect(gain).toBeLessThanOrEqual(prev + 1e-9);
      prev = gain;
    }
    module.dispose();
  });

  test('the koi rise toward the light and never break the surface', async () => {
    const { module } = await live();
    const koi = module.scene.getObjectByName('koiSchool');
    expect(koi instanceof THREE.InstancedMesh).toBe(true);
    if (!(koi instanceof THREE.InstancedMesh)) return;
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const meanY = (): number => {
      let sum = 0;
      for (let i = 0; i < koi.count; i++) {
        koi.getMatrixAt(i, m);
        p.setFromMatrixPosition(m);
        expect(p.y).toBeLessThan(WATER_LEVEL - 0.04); // always under the water
        sum += p.y;
      }
      return sum / koi.count;
    };

    module.update(toFrameUpdate(fixtureNamed('inverted-init')));
    const deep = meanY();
    module.update(toFrameUpdate(fixtureNamed('at-rest')));
    const risen = meanY();
    expect(risen).toBeGreaterThan(deep + 0.2);
    module.dispose();
  });
});

describe('pendulum / the torque curl', () => {
  test('the curl picks its side from the SIGN and its weight from the magnitude', async () => {
    const { module } = await live();
    const pos = module.scene.getObjectByName('torqueCurlPositive');
    const neg = module.scene.getObjectByName('torqueCurlNegative');
    expect(pos).toBeDefined();
    expect(neg).toBeDefined();
    if (!(pos instanceof THREE.Mesh) || !(neg instanceof THREE.Mesh)) return;

    module.update(toFrameUpdate(poseFixture('push+', 0.3, 0, PD_MAX_TORQUE, 1)));
    expect(pos.visible).toBe(true);
    expect(neg.visible).toBe(false);
    const full = (pos.material as THREE.Material).opacity;

    module.update(toFrameUpdate(poseFixture('push-', 0.3, 0, -PD_MAX_TORQUE, 2)));
    expect(pos.visible).toBe(false);
    expect(neg.visible).toBe(true);
    // the negative family is the same geometry, mirrored across the tangent
    expect(neg.scale.x).toBeLessThan(0);

    module.update(toFrameUpdate(poseFixture('nudge', 0.3, 0, PD_MAX_TORQUE * 0.25, 3)));
    expect((pos.material as THREE.Material).opacity).toBeLessThan(full);

    module.update(toFrameUpdate(poseFixture('coast', 0.3, 0, 0, 4)));
    expect(pos.visible).toBe(false);
    expect(neg.visible).toBe(false);
    module.dispose();
  });

  test('thumb drops the curl with the particles and the mist', async () => {
    const { module } = await live('thumb');
    const pos = module.scene.getObjectByName('torqueCurlPositive');
    const neg = module.scene.getObjectByName('torqueCurlNegative');
    expect(pos).toBeDefined();
    expect(neg).toBeDefined();
    for (const torque of [PD_MAX_TORQUE, -PD_MAX_TORQUE, 0]) {
      module.update(toFrameUpdate(poseFixture(`t${torque}`, 0.3, 0, torque, 1)));
      expect(pos?.visible).toBe(false);
      expect(neg?.visible).toBe(false);
    }
    module.dispose();
  });

  test('the curl lives in the swing-tangent frame, so it is sign-correct at every angle', async () => {
    const { module } = await live();
    const curl = module.scene.getObjectByName('torqueCurlRoot');
    expect(curl).toBeDefined();
    if (!curl) return;
    const world = new THREE.Vector3();
    const tangent = new THREE.Vector3();
    for (let i = 0; i <= 24; i++) {
      const th = -Math.PI + (2 * Math.PI * i) / 24;
      module.update(toFrameUpdate(poseFixture(`curl${i}`, th, 0, 1, i)));
      curl.updateWorldMatrix(true, false);
      // it sits on the bob…
      world.setFromMatrixPosition(curl.matrixWorld);
      const bob = worldOf(module, 'bob');
      expect(world.distanceTo(bob)).toBeLessThan(1e-9);
      // …and its local +x is the +θ tangent of the swing circle: d/dθ (sin θ, −cos θ)
      tangent.set(1, 0, 0).transformDirection(curl.matrixWorld);
      const t = Math.fround(th);
      expect(tangent.x).toBeCloseTo(Math.cos(t), 6);
      expect(tangent.y).toBeCloseTo(Math.sin(t), 6);
    }
    module.dispose();
  });
});

describe('pendulum / update contract', () => {
  test('update is idempotent on the same state', async () => {
    const { module } = await live();
    for (const f of fixtures) assertUpdateIdempotent(module, f);
    module.dispose();
  });

  test('update is idempotent while the guard is HOLDING, too', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(poseFixture('apex', 3.088365, 2.621751, 0, 39)));
    const frame = toFrameUpdate(poseFixture('mid', 0.01, 2.6, 0, 39), { scrubbed: false });
    module.update(frame);
    expect(isHolding(module)).toBe(true);
    const first = worldOf(module, 'bob').clone();
    for (let i = 0; i < 5; i++) module.update(frame);
    expect(isHolding(module)).toBe(true);
    expect(worldOf(module, 'bob').distanceTo(first)).toBe(0);
    module.dispose();
  });

  test('update never grows the scene graph', async () => {
    const { module } = await live();
    module.update(toFrameUpdate(fixtureNamed('at-rest')));
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

  test('this scene NEVER celebrates — not on any pose, not on any event', async () => {
    const { module, ctx } = await live();
    const pool = existingCelebrations(module.scene);
    expect(pool).not.toBeNull();
    if (!pool) return;

    // every canonical pose, played sequentially: a meditation piece has no beats
    let t = 0;
    for (const f of fixtures) {
      ctx.setTime((t += 1));
      module.update(toFrameUpdate(f, { scrubbed: false }));
    }
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);
    expect(pool.pulse.bloom).toBe(0);

    // and even handed the full event grammar (which this env can never emit), nothing
    ctx.setTime(99);
    module.update(
      toFrameUpdate(fixtureNamed('at-rest'), {
        scrubbed: false,
        events: [{ type: 'goal' }, { type: 'done' }, { type: 'teleport' }],
      }),
    );
    expect(pool.stats.bursts).toBe(0);
    expect(pool.stats.ripples).toBe(0);
    expect(pool.pulse.bloom).toBe(0);
    expect(pool.activeRipples).toBe(0);
    module.dispose();
  });

  test('the decoder really does emit no events for this env', () => {
    const st = pendulumDecoder.initStatic(SEED);
    const a = pendulumDecoder.initFrame(SEED, st);
    const b = pendulumDecoder.initFrame(SEED, st);
    expect(pendulumDecoder.deriveEvents(a, b, st).length).toBe(0);
  });
});

describe('pendulum / lifecycle', () => {
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
    module.update(toFrameUpdate(fixtureNamed('swing-right')));
    let overlayVisible = false;
    module.scene.traverse((o) => {
      if (o.name === 'lanternBellGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(true);
    const debugLayer = new THREE.Layers();
    debugLayer.set(31);
    expect(module.camera.layers.test(debugLayer)).toBe(true);
    module.setDebug(false);
    module.scene.traverse((o) => {
      if (o.name === 'lanternBellGroundTruth') overlayVisible = o.visible;
    });
    expect(overlayVisible).toBe(false);
    module.dispose();
  });

  test('the OVERLAY always shows the raw recorded angle, even while the art is held', async () => {
    const { module } = await live();
    module.setDebug(true);
    module.update(toFrameUpdate(poseFixture('apex', 3.088365, 2.621751, 0, 39)));
    module.update(toFrameUpdate(poseFixture('mid', 0.01, 2.6, 0, 39), { scrubbed: false }));
    expect(isHolding(module)).toBe(true);
    // the art holds the apex…
    expect(worldOf(module, 'bob').y).toBeGreaterThan(0.9);
    // …and the probe's own ground truth is still the raw pose, so the review overlay
    // would draw the disagreement rather than hide it
    const probe = module.probes().find((p) => p.name === 'bob');
    expect(probe).toBeDefined();
    const truth = probe?.expectWorldPos({
      pose: new Float32Array([0.01, 2.6, 0, 0, 0, 0, 0, 0, 0]),
      ctrl: new Float32Array(6),
      done: false,
      step: 39,
      simTime: 39 / 20,
      derived: null,
    });
    expect(truth?.[1] ?? 0).toBeLessThan(-0.9);
    module.dispose();
  });
});

describe('pendulum / the thumb wall-view cap', () => {
  test('the thumb frame fits STYLE.md\u2019s 40 draws / 50k triangles, overlay included', async () => {
    // The wall view is the whole gallery on one page: every scene there is a thumb, and
    // the cap is what keeps 16 of them at 60 fps. Measured with the ground-truth overlay
    // ON, because that is the state review screenshots are taken in, and across every
    // fixture, because the busiest legal state is the one that has to fit.
    const { module } = await live('thumb');
    module.setDebug(true);
    for (const f of fixtures) module.update(toFrameUpdate(f));
    assertThumbBudget(module, 'pendulum (thumb, overlay on)');
    module.dispose();
  });
});
