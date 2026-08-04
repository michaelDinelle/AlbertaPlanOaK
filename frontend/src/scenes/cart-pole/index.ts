// The Boardwalk — CartPole-cont-v0.
//
// THE ONE IDEA: a pole TIP's height is L·cos(th) and nothing else, so the whole angular
// failure condition — |th| > 12° — is the single sentence "the tip must stay above
// y = L·cos(12°)", true at every cart position. This scene strings that sentence across
// the lake as a surveyor's cord at exactly the `failArcRight` anchor height.
//
// It is the TIP, not "the crown", and the difference is the whole reason the sapling is
// built the way it is. A free-growing leaf whorl reached radius 1.196·L and therefore
// only crossed the cord at 35°, so at the exact failure pose the art still read SAFE
// while the env had already killed the episode. So the whorl is now clamped inside a
// sphere of 0.9·L about the hinge and the topmost thing on the sapling is a pale
// terminal bud whose apex IS `cartPoleTip(x, th)`. Measured off the built mesh at every
// tier: the drawn top of the sapling crosses the cord at 12.00°, and no leaf vertex ever
// reaches the cord even under the crown's full ±0.30 rad droop. Measured in SCREEN space
// through the hero camera it crosses at 11.67° — the residue is the cord's own 0.3 px of
// depth parallax, against 7.9° before this pass.
//
// The cord is honest but SMALL: L(1 − cos 12°) is 2.2% of the pole, 6.1 px in a 1600×900
// hero frame, which is why the carved cone behind the deck carries the read a viewer can
// actually see (the tip swings ±56 px sideways over that same band) and why it is planted
// as a sighting instrument — see `gaugeStation` in world.ts. The other bound needs no
// gauge at all: the boardwalk runs from `trackLeft` to `trackRight` and stops over water.
//
// WHAT THE RECORDING ACTUALLY CONTAINS, and why the beats are sized the way they are:
// this agent falls every 13–24 control steps (median 12 on seed 48), i.e. roughly four
// times per second of playback. Every celebration weight here is chosen for THAT rate.
// A full-strength grammar at 4 Hz is a strobe; what the frame wants is a rhythm. So the
// fall beat is a dozen cool motes, a shed of leaves and a half-second dip of the key
// light — and the reward, which is a dense +1 on every surviving step and never an
// event, is simply the fact that the world stays warm while the sapling stands.
//
// There is no `goal` event in CartPole and therefore no burst, no ripple and no bloom
// pulse anywhere in this file. Firing one would be a lie about what the agent earned.
//
// THE SUPERLUMINAL GUARD. A fall scores its +1, sets done = 1 and resets all four
// coordinates to zero IN THE SAME STEP, so the host's pose lerp renders the trolley
// sliding home from the fall BEFORE the event arrives. The env integrates with explicit
// Euler on the OLD velocity (`th' = th + τ·thd`, verified to f32 noise across all eight
// demo lanes), so the displacement over a segment is τ·∫thd — and the sign of that
// integral is the whole test.
//
// A SYMMETRIC |Δth| ≤ τ·seg·|thd| bound does NOT work here, and the failure is
// structural rather than a matter of slack: |thd| is MAXIMAL exactly at a fall, so the
// licence it grants is widest exactly where the slide has to be caught. Measured by
// driving THIS module with the real lane through the host's own sampling loop (seed 48,
// 20k steps, 50 Hz record → 60 fps, `guardprobe`): the symmetric form held only 355 of
// 1492 slide frames at 1×, and drew the pole un-falling by up to 0.1155 rad — 55% of the
// entire legal tilt band — before the event landed. Its x channel could not help: this
// agent barely moves the cart (|x| max 1.158 over 20k steps).
//
// `unreachable()` below replaces it with the signed reachability interval, and the same
// probe now reports, over 0.25× / 0.5× / 1× / 2× / 4×: the guard holds 91–98% of all
// slide frames, the worst frame-to-frame UN-fall is 0.0027 rad at 1× (1.3% of the band,
// 10 frames in 24001; the residue is at 4×, where one rendered frame already spans 3.3
// control steps), and it wrongly held a legal step ZERO times at every speed.

import { GroundTruthOverlay } from '@kit/debug';
import { makeFrameRig, type FrameRig } from '@kit/rigs';
import {
  CelebrationPool,
  attachCelebrations,
  disposeCelebrations,
} from '@kit/celebrate';
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
import { CP_TAU, cartPoleTip } from '@sim/envs/cartpole';
import { buildTrolley, type TrolleyHandle } from './actors';
import { manifest } from './manifest';
import {
  CAM_BASE,
  CAM_BASE_THUMB,
  CAM_LOOK,
  CAM_LOOK_THUMB,
  DECK_Y,
  GAUGE_Z,
  buildWorld,
  clampRange,
  gaugeStation,
  poleLength,
  tiltLimit,
  trackLimit,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// flourish envelopes, all measured on ctx.time()
const FALL_SECONDS = 0.42;
const COOL_SECONDS = 0.58;
/** the failure grammar, sized for a 4 Hz failure rate (see the header note) */
const SIGH_STRENGTH = 0.5;
const SIGH_PARTICLES = 34;
/** how far in FRONT of the sapling the two claimed point lights sit (see update) */
const LIGHT_STANDOFF = 1.15;

// ---- the superluminal guard ----
/**
 * Max |Δthd| and |Δxd| a single legal control step produced across all eight demo
 * lanes (0.351 rad/s and 0.197 m/s), rounded up. gym's own numbers agree: at the 12°
 * bound under a ±10 N shove, τ·|θ̈| ≤ 0.35 and τ·|ẍ| ≤ 0.19. This is how far a velocity
 * can have drifted INSIDE the segment a rendered frame straddles.
 */
const TH_ACC = 0.4;
const X_ACC = 0.25;
/** f32 + interpolation noise */
const POSE_EPS = 0.003;
/**
 * How long a hold may last, in PLAYBACK seconds — not rendered frames. One record row
 * is exactly one integrator step of playback (this env records at its own 50 Hz) and the
 * slide is exactly one row long, so this cap carries 100% margin at EVERY transport
 * speed. A frame count does not: at 0.25× one row spans 4.8 rendered frames, and the old
 * 4-frame cap dumped the pole back onto the slide before the cut ever landed (measured:
 * 1021 un-falling frames at 0.25×). Playback seconds also stop while paused, which is
 * the behaviour a paused mid-slide frame wants.
 */
const MAX_HELD_SECONDS = 2 * CP_TAU;

/** τ·∫thd for the extremal velocity profile: a·mean + curve·a²/4. */
function displacementBound(a: number, mean: number, curve: number): number {
  return a * mean + 0.25 * curve * a * a;
}

/**
 * SIGNED REACHABILITY — the guard's whole test.
 *
 * Over `a` control steps whose recorded velocity ran from v0 to v1 with |Δv| ≤ acc per
 * step, the displacement is τ·∫thd, and the extremal profiles (accelerate hard to a
 * peak, then decelerate hard) bound that integral by a·m ± acc·a²/4 with m = (v0+v1)/2.
 * The endpoints are themselves host-interpolated — each within `acc` of a recorded
 * velocity — which widens m by ±acc. `a` is known only to lie in [aLo, aHi]; both
 * bounding curves are quadratics whose extremes over that range sit at its endpoints.
 *
 * DIRECTION is the point. |thd| is maximal exactly at a fall, so a symmetric
 * |Δth| ≤ τ·a·|thd| bound grants its widest licence precisely where the reset slide has
 * to be caught — measured, it let 76% of slide frames through. Here a large falling thd
 * widens the band only in the direction the pole is falling; the slide runs the other
 * way, back toward upright, and lands outside it.
 */
export function unreachable(
  d: number,
  v0: number,
  v1: number,
  acc: number,
  aLo: number,
  aHi: number,
): boolean {
  const mean = 0.5 * (v0 + v1);
  const lo = Math.min(
    displacementBound(aLo, mean - acc, -acc),
    displacementBound(aHi, mean - acc, -acc),
  );
  const hi = Math.max(
    displacementBound(aLo, mean + acc, acc),
    displacementBound(aHi, mean + acc, acc),
  );
  return d < CP_TAU * lo - POSE_EPS || d > CP_TAU * hi + POSE_EPS;
}

interface CartPoleLike {
  force?: unknown;
}

/** force ∈ {−10, +10}: sim-core's decode when present, the raw ctrl bit otherwise. */
function forceOf(state: FrameState): number {
  const d = state.derived as CartPoleLike | null | undefined;
  if (d && typeof d === 'object' && typeof d.force === 'number') return d.force;
  const raw = state.ctrl[0] ?? 0;
  if (!Number.isFinite(raw)) return 0;
  return ((Math.trunc(raw) >>> 0) & 1) !== 0 ? 10 : -10;
}

class BoardwalkScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private trolley: TrolleyHandle | null = null;
  private rig: FrameRig | null = null;
  private pool: CelebrationPool | null = null;
  private overlay: GroundTruthOverlay | null = null;

  private xLimit = 2.4;
  private thLimit = 0.2094395;
  private armLength = 1;

  // ---- flourish clocks (playback seconds; -Infinity = idle) ----
  private fallAt = -Infinity;
  private fallDir = 1;

  // ---- the superluminal guard ----
  private heldX = Number.NaN;
  private heldTh = 0;
  private heldXd = 0;
  private heldThd = 0;
  private heldStep = Number.NaN;
  private heldFrames = 0;
  /** playback seconds at which the current hold began (-Infinity = not holding) */
  private heldAt = -Infinity;

  private debugOn = false;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];
  private readonly tipScratch = new Float32Array(2);

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'boardwalkScene';

    const palette = ctx.tokens.palettes.noon;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.xLimit = trackLimit(ctx.geometry);
    this.thLimit = tiltLimit(ctx.geometry);
    this.armLength = poleLength(ctx.geometry);

    const world = buildWorld(ctx, this.scene);
    this.world = world;
    this.scene.add(world.group);

    // This scene claims both of the rig's point lights. Their colours are STATIC — only
    // intensity is state-driven, so update() never touches a Color. points[] is empty at
    // thumb tier; the guards are the tier policy.
    const warm = world.rig.points[0];
    if (warm) warm.color.set(palette.accentLight);
    const cool = world.rig.points[1];
    if (cool) cool.color.set(ctx.tokens.failureBlue);

    const trolley = buildTrolley({
      tier: ctx.tier,
      palette,
      accent,
      seed: ctx.seed,
      poleLength: this.armLength,
      tiltLimit: this.thLimit,
    });
    this.trolley = trolley;
    this.scene.add(trolley.group);

    const thumb = ctx.tier === 'thumb';
    const rig = makeFrameRig({
      position: thumb ? CAM_BASE_THUMB : CAM_BASE,
      lookAt: thumb ? CAM_LOOK_THUMB : CAM_LOOK,
      fov: ctx.tokens.fovHero,
      near: 0.05,
      far: 420,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      canvas: thumb ? null : ctx.canvas,
      parallax: 0.1,
      parallaxY: 0.4,
      damping: 2.4,
      name: 'boardwalkFrameRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      sighSize: SIGH_PARTICLES,
      sighRadius: 0.34,
      sighHeight: 0.9,
      name: 'boardwalkCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      // the pole TIP is the one point that encodes both state coordinates at once, so
      // it is the marker worth checking against the art
      agentPos: (state) => {
        const t = cartPoleTip(state.pose[0] ?? 0, state.pose[2] ?? 0, this.tipScratch);
        this.overlayPos[0] = t[0] ?? 0;
        this.overlayPos[1] = t[1] ?? 0;
        this.overlayPos[2] = 0;
        return this.overlayPos;
      },
      name: 'boardwalkGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    trolley.setPixelRatio(ctx.size.dpr);
    world.setPixelRatio(ctx.size.dpr);
    trolley.setShed(0, 1);
    trolley.setState(0, 0, 0, 0);
    world.setWind(0);
    world.setCool(0);
    world.setTime(0);
    trolley.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const trolley = this.trolley;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !trolley || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. a scrub wipes every in-flight beat: nothing may survive a discontinuity
    if (frame.scrubbed) {
      pool.clear();
      this.fallAt = -Infinity;
      this.heldX = Number.NaN;
      this.heldFrames = 0;
      this.heldAt = -Infinity;
      rig.snap();
    }

    // 3. events — the sigh is NEVER inferred from the state stream. `done` here is
    //    always a FAILURE (CartPole's +1 is the dense per-step reward, not the done),
    //    and it always arrives with its own `teleport`.
    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'done') {
        // Fire where the sapling WAS. The reset row this event arrives with is already
        // all-zero, so both the position and the direction come from the frame BEFORE
        // it: `crownWorld` still holds the pre-reset crown, and `heldTh` is still the
        // angle the pole fell at (the guard below has not run yet this frame). That is
        // the one piece of event-latched state in this scene, and a scrub clears it.
        pool.sigh(SIGH_STRENGTH, trolley.crownWorld, time);
        this.fallAt = time;
        this.fallDir = this.heldTh < 0 ? -1 : 1;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    // 4. the superluminal guard (see the header note)
    const rawX = state.pose[0] ?? 0;
    const rawXd = state.pose[1] ?? 0;
    const rawTh = state.pose[2] ?? 0;
    const rawThd = state.pose[3] ?? 0;
    let x = rawX;
    let th = rawTh;
    if (teleported || frame.scrubbed || !Number.isFinite(this.heldX)) {
      this.heldFrames = 0;
    } else {
      // How many control steps the record head advanced since the held pose. Two sampled
      // positions in the same sequential window differ by Δstep plus a fraction whose
      // magnitude is under one, and time only runs forward — so it is in [Δstep−1, Δstep+1]
      // intersected with [0, ∞).
      const drift = Math.abs(state.step - this.heldStep);
      const aLo = drift > 1 ? drift - 1 : 0;
      const aHi = drift + 1;
      const jumped =
        unreachable(rawTh - this.heldTh, this.heldThd, rawThd, TH_ACC, aLo, aHi) ||
        unreachable(rawX - this.heldX, this.heldXd, rawXd, X_ACC, aLo, aHi);
      if (jumped && (this.heldFrames === 0 || time - this.heldAt <= MAX_HELD_SECONDS)) {
        if (this.heldFrames === 0) this.heldAt = time;
        x = this.heldX;
        th = this.heldTh;
        this.heldFrames++;
      } else {
        this.heldFrames = 0;
      }
    }
    // While the hold is up the whole reference frame freezes — pose, velocities AND the
    // step it came from — so `seg` keeps measuring the real span back to it.
    if (this.heldFrames === 0) {
      this.heldX = rawX;
      this.heldTh = rawTh;
      this.heldXd = rawXd;
      this.heldThd = rawThd;
      this.heldStep = state.step;
    }

    // 5. envelopes — every one of them measured on the playback clock
    const fall = envelope(time - this.fallAt, FALL_SECONDS, 3.4);
    const cool = envelope(time - this.fallAt, COOL_SECONDS, 2.6);

    // 6. the state visuals
    trolley.setShed(fall, this.fallDir);
    trolley.setState(x, th, rawXd, forceOf(state));
    // the gust that leans the sapling combs the grass (cosmetic; state, never history)
    world.setWind(clampRange(th / this.thLimit, -1, 1));
    world.setCool(cool);

    // 7. THE DENSE REWARD IS THE LIGHT. Every surviving step scores +1, so "how safe is
    //    this pose" is literally how much reward the next steps are worth — and it is
    //    the warmth of the frame. A fall takes it away for half a second.
    const safety =
      1 -
      clampRange(
        Math.max(Math.abs(th) / this.thLimit, Math.abs(x) / this.xLimit),
        0,
        1,
      );
    // A point light STANDS OFF the crown. Parked on it, the leaves sit 5 cm from the
    // source and a decay-2 falloff delivers I/d² ≈ 14 — the crown measured as a blown
    // white blob. From 1.1 units in front it is a soft fill on the camera-facing side,
    // which is exactly the face the key light rakes away from.
    const warm = world.rig.points[0];
    if (warm) {
      warm.position.set(
        trolley.crownWorld.x - 0.25,
        trolley.crownWorld.y + 0.3,
        LIGHT_STANDOFF,
      );
      warm.intensity = (0.35 + 1.15 * safety * safety) * (1 - 0.75 * cool);
    }
    const coolLight = world.rig.points[1];
    if (coolLight) {
      coolLight.position.set(
        trolley.crownWorld.x + 0.3,
        trolley.crownWorld.y + 0.15,
        LIGHT_STANDOFF,
      );
      coolLight.intensity = 1.5 * cool;
    }

    // 8. one clock for the whole world
    world.setTime(time);
    trolley.setTime(time);

    // 9. camera easing runs in WALL time so a paused frame is still explorable
    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.trolley?.setPixelRatio(size.dpr);
    this.world?.setPixelRatio(size.dpr);
    this.pool?.setPixelRatio(size.dpr);
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
  }

  probes(): AlignmentProbe[] {
    const trolley = this.trolley;
    const world = this.world;
    const xLimit = this.xLimit;
    const arm = this.armLength;
    const root = trolley?.root ?? new THREE.Object3D();
    const tip = trolley?.tip ?? new THREE.Object3D();
    const endLeft = world?.endLeftRoot ?? new THREE.Object3D();
    const endRight = world?.endRightRoot ?? new THREE.Object3D();
    const gauge = world?.gaugeRoot ?? new THREE.Object3D();
    const station = gaugeStation(
      this.ctx?.tier === 'thumb' ? CAM_BASE_THUMB : CAM_BASE,
      GAUGE_Z,
    );
    return [
      {
        name: 'trolley',
        object: root,
        expectWorldPos: (state) => [state.pose[0] ?? 0, DECK_Y, 0] as const,
      },
      {
        // exactly `cartPoleTip`: (x + L sin th, L cos th)
        name: 'pole-tip',
        object: tip,
        expectWorldPos: (state) => {
          const x = state.pose[0] ?? 0;
          const th = state.pose[2] ?? 0;
          return [x + arm * Math.sin(th), arm * Math.cos(th), 0] as const;
        },
      },
      {
        name: 'deck-end-left',
        object: endLeft,
        expectWorldPos: () => [-xLimit, DECK_Y, 0] as const,
      },
      {
        name: 'deck-end-right',
        object: endRight,
        expectWorldPos: () => [xLimit, DECK_Y, 0] as const,
      },
      {
        // The carved failure cone is a SIGHTING instrument: its apex stands on the
        // eye→pivot ray at the gauge's own depth (world.ts gaugeStation), which is what
        // makes its drawn cone the exact perspective image of the pole's failure cone.
        name: 'tilt-gauge',
        object: gauge,
        expectWorldPos: () => [station.x, station.y, station.z] as const,
      },
    ];
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.trolley?.dispose();
    this.trolley = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.fallAt = -Infinity;
    this.heldX = Number.NaN;
    this.heldFrames = 0;
    this.heldAt = -Infinity;
  }
}

/**
 * Exponential decay in [0, 1] over `span` playback seconds, 0 outside the window.
 * The linear tail term is not cosmetic: a bare exp() still has real amplitude when the
 * window closes, so every flourish would end on a one-frame pop.
 */
export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

const createScene: SceneFactory = () => new BoardwalkScene();
export default createScene;
