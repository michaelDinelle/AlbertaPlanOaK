// The Lantern Bell — Pendulum-cont-v0. A meditation piece.
//
// THE ONE IDEA: the shaped reward IS the lighting. The env's cost is
// `θ² + 0.1·θ̇² + 0.001·u²` (`gw_pend_step`, Sources/ClassicControl.metal), and
// `serenityOf` below is that expression — the whole of it, the same weights, wrapped the
// same way — mapped through ONE decreasing curve into [0, 1]. The lantern's flame, the
// three warm point lights, the water's caustics and colour, the reflection on the pond
// and how high the koi rise are all that single number. Nothing in this scene is lit
// because it looks nice; when the agent is far from its cost minimum the pond really does
// go cold, and the only light left is the moon and the stars.
//
// BECAUSE IT IS A DECREASING FUNCTION OF THE COST ITSELF, a state the agent is scored
// BETTER on can never render darker than a worse one — which is the property the first
// pass got wrong. That one multiplied a `cos θ` term by a `θ̇` term, and a product of two
// terms is not monotone in their sum: (θ=0, θ̇=8) costs 6.40 and rendered 2.1× BRIGHTER
// than (θ=2.0, θ̇=0), which costs 4.00. Its rank correlation with the true reward was
// 0.47 on the real seed-56 lane. This one's is 1.000, by construction, and the test locks
// it over a grid instead of along the one slice where a product cannot fail.
//
// The velocity term is not a garnish, either: over that lane the mean cost is 0.051 of θ²
// against 0.031 of 0.1·θ̇², so roughly a third of what the agent is being charged for is
// SPEED. A lantern whipping through the bottom of its arc is not at the cost minimum and
// the pond must not pretend it is.
//
// WHICH POSE IS THE COST MINIMUM. Read `sim/envs/pendulum.ts` before touching this:
// the kernel integrates `θ̈ = −15 sin θ`, so **θ = 0 is the bob HANGING** (the stable
// equilibrium and the cost minimum) and θ = ±π is inverted (the fleet init, which falls).
// The decoder documents that the Metal comments have those two labels backwards. The
// brief asks for a `cos θ` law; the cost's θ² term plays exactly that role — brightest
// hanging, darkest inverted — so the brief's intent survives, with the velocity and
// torque terms the brief did not know about folded in. The bright pose is the lantern at
// REST at the bottom of its arc, which is exactly where the agent is being rewarded.
// Lighting the inverted pose instead would be a lie about the reward.
//
// THE ANGLE-WRAP TRAP (the superluminal guard, canon lesson 4). θ is wrapped into [−π, π)
// every step, and the host lerps the 9-float pose LINEARLY. So the single control step
// that contains a wrap renders the lantern spinning all the way backwards through the
// bottom of its arc — a full revolution in 50 ms. The guard below HOLDS the last
// physically-reachable angle for the length of that control step; the lantern hangs at
// the top for a breath and goes over, which is what a pendulum barely clearing the top
// actually does.
//
// THE GUARD IS ANCHORED PER CONTROL STEP, NEVER PER FRAME — that is the whole trick, and
// the first pass got it wrong in both directions. It compared each RENDER frame against
// the previous RENDER frame using a whole control step's budget, so:
//
//   * at fine sampling it never fired. The host advances `controlHz·speed·dtReal` rows
//     per frame, so 0.25× on a 120 Hz display is 24 frames inside one row; each of those
//     frames moves only 1/24 of the wrap, which is well under any one-step budget. The
//     real seed-56 lane (2 genuine wraps in 100k steps) rendered the fabricated
//     revolution in full — bob y reaching −1.000, the lantern swinging DOWN through the
//     bottom of its arc and back up over ~0.3 s. Zero holds fired.
//   * at coarse sampling it fired on honest motion. At 4× the host advances 1.33 rows per
//     frame while `state.step` advances 1, so the budget under-stated the interval and
//     legal fast swinging was frozen: 6 hold-runs where only 2 wraps exist, saturating a
//     16-frame cap into a quarter-second freeze followed by a snap.
//
// Anchoring on the record step instead makes the test sampling-rate independent, because
// the physics bound is per control step and now so is the comparison. Three exact bounds,
// all read off the env's own integrator, decide it (see PD_MAX_DTHETA below):
//
//   1. REACH — across `steps` control steps θ cannot move further than the integrator
//      allows at this frame's own θ̇, summed over the speed ramp (`wrapReach`).
//   2. DIRECTION — θ̇ cannot change sign inside `steps` control steps unless |θ̇| is small
//      enough to reach zero (≤ MAX_DTHETADOT per step). When it cannot, a displacement
//      that OPPOSES θ̇ is impossible — and a linear lerp across the branch cut always
//      opposes θ̇, because the true motion crosses the cut while the lerp takes the long
//      way round. That signature is independent of how finely the host samples.
//   3. The record row itself is never doubted: at the moment `state.step` advances, the
//      incoming pose is an exact record row (α = 0), so the guard re-anchors on it — and
//      a hold never outlives the step it began in (MAX_HOLD_STEPS), because the artifact
//      cannot either, and a stale anchor is worse than the glitch.
//
// Measured on the REAL seed-56 lane, replayed through the host's own sampler at 0.38 …
// 48 frames per row and 0.25× … 4×: exactly 2 hold runs (the lane has exactly 2 wraps) at
// every density, the rendered angle never further than 0.27 rad from the physically
// correct short-arc angle (it was π), the bob never below y = +0.96 through a crossing
// (it reached −1.000), and zero frozen frames during honest motion (there were 96). On a
// synthetic stream that rotates continuously through ±π four times, the worst case rises
// to 0.61 rad — still inside the structural bound the test asserts, 2·PD_MAX_DTHETA.
//
// AND THERE IS NEVER A CELEBRATION. Pendulum never sets `done`, never resets and never
// scores: `deriveEvents` returns nothing, forever. The pool is attached so the host's
// per-scene bloom channel resolves like every other scene's — but this scene fires no
// burst, no ripple and no sigh, ever, and the test locks that even against an injected
// event batch.

import { GroundTruthOverlay } from '@kit/debug';
import { makeFrameRig, type FrameRig } from '@kit/rigs';
import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import type {
  AlignmentProbe,
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import * as THREE from 'three';
import { PD_DT, PD_G, PD_L, PD_M, PD_MAX_SPEED, PD_MAX_TORQUE, wrapPi } from '@sim/envs/pendulum';
import { buildLanternRig, type LanternRigHandle } from './actors';
import { manifest } from './manifest';
import {
  buildWorld,
  clampRange,
  pivotOf,
  REFLECT_LIFT,
  REFLECT_Z,
  WATER_LEVEL,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. Near-profile, because the ANGLE is the physics here: swing the camera round
// and θ stops reading off the silhouette. 6° of offset is enough to separate the two
// torii pillars in depth without foreshortening the arc.
//
// The distance is pinned by TWO edges at once, and both are unforgiving at a fixed 40°
// vFOV (canon lesson 11 — vFOV never changes with aspect, so a wide window does not help).
// Closer than this and the crown's swept ends leave the top of the frame; a torii with a
// cropped crown stops reading as a torii. Farther and the bottom edge drops past the
// pond's near shoreline at z ≈ 2.1 and a bank of reeds climbs into the foreground,
// between the camera and the water the lantern is reflecting in.
//
// The tie beam moved UP when the arch was rebuilt around the lantern's true swept reach
// (world.ts / dims.ts), taking the crown to y = 1.93, so this frame pulls back 0.25 and
// lifts the look-at by the same 0.07 the arch gained: headroom over the crown is back to
// ~0.38 while the bottom edge stays where it was, on open water. Pulling back ALONE would
// have dropped the near shoreline into the foreground.
const CAM_BASE: readonly [number, number, number] = [0.62, 0.41, 6.55];
const CAM_LOOK: readonly [number, number, number] = [0, -0.09, 0];
/** thumb pulls back for a square-ish crop; the top edge lands on the same y as hero's */
const CAM_BASE_THUMB: readonly [number, number, number] = [0.34, 0.36, 6.95];
const CAM_LOOK_THUMB: readonly [number, number, number] = [0, -0.24, 0];

// ---------------------------------------------------------------------------
// the env's own one-step bounds — read off the integrator, not guessed
// ---------------------------------------------------------------------------
//
// `gw_pend_step` (Sources/ClassicControl.metal) integrates
//
//     nthd = thd + (1.5·g/l · sin(θ+π) + 3·u/(m·l²)) · dt      |the term| ≤ 21
//     th  += nthd · dt          ← BEFORE nthd is clamped to ±PD_MAX_SPEED
//     nthd = clamp(nthd, ±PD_MAX_SPEED)
//
// so one control step moves θ̇ by at most PD_MAX_DTHETADOT, and θ by at most
// (PD_MAX_SPEED + PD_MAX_DTHETADOT)·dt = 0.4525 — NOT PD_MAX_SPEED·dt = 0.4. The shipped
// guard used the smaller figure and was therefore below the env's true single-step reach
// (the largest step on the seed-56 lane is 0.3995, so it was latent, not yet fatal).

/** |Δθ̇| the gravity + torque term can produce in one control step */
export const PD_MAX_DTHETADOT = ((1.5 * PD_G) / PD_L + (3 * PD_MAX_TORQUE) / (PD_M * PD_L * PD_L)) * PD_DT;
/** |Δθ| one control step can produce, worst case over every reachable state */
export const PD_MAX_DTHETA = (PD_MAX_SPEED + PD_MAX_DTHETADOT) * PD_DT;
/** f32 pose round-trip + one frame of lerp noise; never a physical allowance */
const WRAP_SLACK = 0.02;
/**
 * A HOLD LIVES INSIDE EXACTLY ONE CONTROL STEP, and that is a correctness bound, not a
 * safety valve. The lerp artifact cannot outlive the row that contains the wrap: two
 * consecutive rows can never both wrap (a second crossing needs another 2π − 0.45 rad of
 * travel), so once `state.step` advances the incoming pose is honest again. Meanwhile a
 * hold whose anchor has gone stale is WORSE than the artifact it replaces — at 4× on a
 * 30 Hz display the anchor ages 2.7 rows per frame, and an unbounded hold froze the
 * lantern for three frames at up to 2.88 rad of error to prevent one frame of glitch.
 */
export const MAX_HOLD_STEPS = 1;

/**
 * How far θ can travel across `steps` control steps, given the θ̇ this frame reports.
 *
 * The frame's θ̇ is honest (it never wraps, so the host's lerp of it is a real velocity
 * between two real velocities). One control step moves θ by |θ̇ₖ₊₁|·dt, and the speed can
 * only ramp by PD_MAX_DTHETADOT per step — so across `steps` steps the bound is the SUM
 * of a ramp, `Σᵢ (|θ̇| + (i+1)·Δv)·dt`, not the final speed applied to the whole interval.
 * (Using the final speed throughout is loose by a factor that grows with `steps`, and at
 * 4× transport that slack was wide enough to let a mid-wrap sample through.)
 *
 * At the slow angles where wraps actually happen — the seed-56 lane's first wrap crosses
 * ±π at |θ̇| = 0.013 — this is far tighter than the global PD_MAX_DTHETA, and it is capped
 * by that global bound, so it is never an over-claim.
 */
export function wrapReach(thetaDot: number, steps: number): number {
  const ramp = PD_MAX_DTHETADOT * ((steps * (steps + 1)) / 2 + steps);
  const travel = Math.min(
    steps * Math.abs(thetaDot) + ramp,
    steps * (PD_MAX_SPEED + PD_MAX_DTHETADOT),
  );
  return travel * PD_DT + WRAP_SLACK;
}

interface PendulumLike {
  torque?: unknown;
}

/** sim-core's decode when present, the raw pose otherwise (both are the same float). */
function torqueOf(state: FrameState): number {
  const d = state.derived as PendulumLike | null | undefined;
  if (d && typeof d === 'object' && typeof d.torque === 'number') return d.torque;
  const raw = state.pose[2] ?? 0;
  return Number.isFinite(raw) ? raw : 0;
}

/**
 * THE ENV'S OWN COST, transcribed. `gw_pend_step` computes
 * `thn² + 0.1·thd² + 0.001·u²` with `thn = gw_wrap_pi(th)` and `u` the clamped torque —
 * which is exactly `pose[0..2]` as the recorder writes them. Reward is its negative.
 */
export function pendulumCost(theta: number, thetaDot: number, torque: number): number {
  const th = wrapPi(theta);
  return th * th + 0.1 * thetaDot * thetaDot + 0.001 * torque * torque;
}

/** the cost at the worst reachable state: inverted, at the speed clip, on full torque */
export const COST_MAX = pendulumCost(Math.PI, PD_MAX_SPEED, PD_MAX_TORQUE);

/**
 * The render curve. Any monotone increasing map of the normalised reward keeps the
 * honesty claim — this one is a contrast curve, nothing more: it drops the inverted pose
 * to a dim ember instead of a mid-grey so a night frame reads as night. The test locks
 * MONOTONICITY (a better-scored state is never darker), not this exponent.
 */
const SERENITY_GAMMA = 3;

/**
 * The shaped reward, rendered as one number in [0, 1] — strictly decreasing in the cost
 * above, so brightness and reward can never disagree about which state is better.
 *
 * θ = 0 hanging at rest is the cost minimum and renders at 1. The darkest frame is not
 * merely "inverted": it is inverted AND spinning AND pushing, which is the honest reading
 * of a cost that charges for all three. (The old law made every inverted pose equally
 * black, which flattened 6.4 units of velocity cost into nothing.)
 */
export function serenityOf(theta: number, thetaDot: number, torque = 0): number {
  const reward = 1 - pendulumCost(theta, thetaDot, torque) / COST_MAX;
  return clampRange(reward, 0, 1) ** SERENITY_GAMMA;
}

class LanternBellScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private lantern: LanternRigHandle | null = null;
  private rig: FrameRig | null = null;
  private pool: CelebrationPool | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private pivot: readonly [number, number, number] = [0, 0, 0];

  // ---- the angle-wrap guard (see the header note) ----
  // The anchor is the last angle accepted at a RECORD-STEP boundary, and the step it came
  // from. Both, together, are the only history this scene keeps.
  private anchorTheta = Number.NaN;
  private anchorStep = 0;
  // the previous frame's RAW incoming angle — the second reference for the direction
  // test, because at fast transport the anchor can be several steps stale
  private prevRaw = Number.NaN;
  private prevStep = 0;
  /** the record step a hold began in — a hold never outlives it */
  private holdStep = Number.NaN;
  /** true while the guard is holding this frame — the test reads it, the art never does */
  wrapHeld = false;

  private debugOn = false;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'lanternBellScene';
    this.pivot = pivotOf(ctx.geometry);

    const palette = ctx.tokens.palettes.night;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene);
    this.world = world;
    this.scene.add(world.group);

    const lantern = buildLanternRig({
      tier: ctx.tier,
      palette,
      accent,
      pivot: this.pivot,
      maxTorque: PD_MAX_TORQUE,
      maxSpeed: PD_MAX_SPEED,
    });
    this.lantern = lantern;
    this.scene.add(lantern.group);

    const thumb = ctx.tier === 'thumb';
    const rig = makeFrameRig({
      position: thumb ? CAM_BASE_THUMB : CAM_BASE,
      lookAt: thumb ? CAM_LOOK_THUMB : CAM_LOOK,
      fov: ctx.tokens.fovHero,
      near: 0.05,
      far: 400,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      canvas: thumb ? null : ctx.canvas,
      parallax: 0.09,
      parallaxY: 0.5,
      damping: 2.6,
      name: 'lanternFrameRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // Attached so `celebrationPulse(scene)` resolves this scene's own channel like every
    // other scene's. NOTHING in this file ever fires into it — see the header note.
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      name: 'lanternBellCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentPos: (state) => {
        // GROUND TRUTH: the raw recorded angle, never the guard's held value — if the
        // art and the overlay ever disagree, the art is what is wrong.
        const th = state.pose[0] ?? 0;
        this.overlayPos[0] = this.pivot[0] + PD_L * Math.sin(th);
        this.overlayPos[1] = this.pivot[1] - PD_L * Math.cos(th);
        this.overlayPos[2] = this.pivot[2];
        return this.overlayPos;
      },
      name: 'lanternBellGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    world.setPixelRatio(ctx.size.dpr);
    // the fleet init: inverted at θ = π, the darkest the frame ever gets
    this.applyState(Math.PI, 0, 0, 0);
    world.setTime(0);
    lantern.setTime(0);
  }

  /** Everything state-driven, in one place: the scene's whole visual is these lines. */
  private applyState(theta: number, thetaDot: number, torque: number, time: number): void {
    const world = this.world;
    const lantern = this.lantern;
    if (!world || !lantern) return;

    lantern.setState(theta, thetaDot, torque);
    const serenity = serenityOf(theta, thetaDot, torque);
    lantern.setGlow(serenity);
    world.setLight(serenity, lantern.bobRoot.position.x, time);

    // the lantern's own light and its rising motes ride the flame
    const flame = lantern.flameWorld;
    world.motes.position.copy(flame);
    const light = world.lanternLight;
    if (light) light.position.copy(flame);
  }

  /**
   * True when a displacement `delta` from some reference is impossible for a pendulum
   * moving at `thetaDot` across `steps` control steps — i.e. it runs the wrong way and
   * θ̇ is too fast to have turned round inside that interval.
   */
  private opposes(delta: number, thetaDot: number, steps: number): boolean {
    if (!Number.isFinite(delta)) return false;
    return Math.abs(thetaDot) > PD_MAX_DTHETADOT * steps && delta * thetaDot < 0;
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const lantern = this.lantern;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !lantern || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;

    // the pool still adopts the host clock every frame; it simply never has anything in it
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      rig.snap();
    }

    // `frame.events` is deliberately UNREAD. Pendulum has no terminal state, no goal and
    // no respawn, so sim-core emits nothing here — and a `done` that somehow arrived would
    // be a recorder bug, not a failure the agent suffered, so there is nothing to sigh
    // about either. Test-locked: an injected {goal, done, teleport} batch fires nothing.

    // ---- the angle-wrap guard ----
    // θ̇ and the torque never wrap, so the host's lerp is honest for both — which is what
    // makes θ̇ usable as the arbiter of what θ was allowed to do.
    const rawTheta = state.pose[0] ?? 0;
    const thetaDot = state.pose[1] ?? 0;
    let theta = rawTheta;
    this.wrapHeld = false;
    if (frame.scrubbed || !Number.isFinite(this.anchorTheta)) {
      this.anchorTheta = rawTheta;
      this.anchorStep = state.step;
      this.holdStep = Number.NaN;
    } else {
      // the interval since the anchor spans fewer than this many control steps: the
      // anchor sits somewhere inside its own step, this frame inside `state.step`
      const steps = Math.max(0, state.step - this.anchorStep) + 1;
      // measured ON THE CIRCLE: a wrap is continuous there, the host's lerp is not
      const delta = wrapPi(rawTheta - this.anchorTheta);
      const unreachable = Math.abs(delta) > wrapReach(thetaDot, steps);
      // θ̇ cannot reach zero — so cannot change sign — inside `steps` control steps, so
      // above that speed a displacement OPPOSING θ̇ is impossible, and the lerp across the
      // branch cut is always exactly that. Checked from the anchor AND from the previous
      // frame: at 4× transport the anchor may be three steps stale (which raises the
      // speed this test needs) while the previous frame is never more than one.
      const backwards =
        this.opposes(delta, thetaDot, steps) ||
        this.opposes(
          wrapPi(rawTheta - this.prevRaw),
          thetaDot,
          Math.max(0, state.step - this.prevStep) + 1,
        );
      // a hold may run only inside the step it began in (MAX_HOLD_STEPS, above)
      const mayHold = !Number.isFinite(this.holdStep) || this.holdStep === state.step;
      if ((unreachable || backwards) && mayHold) {
        theta = this.anchorTheta;
        this.wrapHeld = true;
        this.holdStep = state.step;
      } else {
        this.holdStep = Number.NaN;
        if (state.step !== this.anchorStep) {
          // a step boundary: the row this pose came from has moved on, so re-anchor —
          // an anchor is only ever allowed to be stale within its own control step
          this.anchorTheta = rawTheta;
          this.anchorStep = state.step;
        }
      }
    }
    this.prevRaw = rawTheta;
    this.prevStep = state.step;

    this.applyState(theta, thetaDot, torqueOf(state), time);

    // one clock for the whole world
    world.setTime(time);
    lantern.setTime(time);

    // camera easing runs in WALL time so a paused frame is still explorable
    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.world?.setPixelRatio(size.dpr);
    this.pool?.setPixelRatio(size.dpr);
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
  }

  probes(): AlignmentProbe[] {
    const pivot = this.pivot;
    const swing = this.lantern?.swingRoot ?? new THREE.Object3D();
    const bob = this.lantern?.bobRoot ?? new THREE.Object3D();
    const pivotRoot = this.world?.pivotRoot ?? new THREE.Object3D();
    const reflection = this.world?.reflectionRoot ?? new THREE.Object3D();
    return [
      {
        // the bob: the whole accuracy claim of the scene in one line
        name: 'bob',
        object: bob,
        expectWorldPos: (state) => {
          const th = state.pose[0] ?? 0;
          return [pivot[0] + PD_L * Math.sin(th), pivot[1] - PD_L * Math.cos(th), pivot[2]] as const;
        },
      },
      {
        // the rod's rotation node sits ON the env's pivot anchor, undecorated
        name: 'swing-root',
        object: swing,
        expectWorldPos: () => pivot,
      },
      {
        // the torii's hanging point is that same anchor — the arch is built around it
        name: 'pivot',
        object: pivotRoot,
        expectWorldPos: () => pivot,
      },
      {
        // the reflection tracks the lantern's abscissa exactly, on the still-water plane
        name: 'lantern-reflection',
        object: reflection,
        expectWorldPos: (state) =>
          [pivot[0] + PD_L * Math.sin(state.pose[0] ?? 0), WATER_LEVEL + REFLECT_LIFT, REFLECT_Z] as const,
      },
    ];
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.lantern?.dispose();
    this.lantern = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.anchorTheta = Number.NaN;
    this.anchorStep = 0;
    this.prevRaw = Number.NaN;
    this.prevStep = 0;
    this.holdStep = Number.NaN;
    this.wrapHeld = false;
  }
}

const createScene: SceneFactory = () => new LanternBellScene();
export default createScene;
