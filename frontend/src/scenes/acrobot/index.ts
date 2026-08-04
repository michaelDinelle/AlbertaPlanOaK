// The Willow Swing — Acrobot-cont-v0.
//
// THE ONE IDEA: the two links ARE the scene graph. `th1` and `th2` are applied as the
// only two rotations in a chain whose bone lengths are the env's own `AB_L1`/`AB_L2`, so
// the elbow and the lantern land on `acrobotPoints` by construction (actors.ts). The
// success surface — `tipHeight > 1` — is not drawn as a hint: it is a string of lights
// whose bulb centres sit exactly on the `successLine` anchor, so "the lantern is above
// the lights" and "the env scores" are the same sentence.
//
// TWO THINGS THIS SCENE HAS TO SOLVE THAT THE EXEMPLARS DID NOT
//
//  1. THE ARC GUARD (the wrap). The kernel wraps both angles into [−π, π), and the host
//     lerps the 9-float pose linearly. When a step crosses the wrap (th2 does it on ~1.5%
//     of recorded steps, th1 on ~0.03%), the linear lerp of 3.10 → −3.10 sweeps the link
//     the LONG way round in 200 ms: a whip once every ~13 s of playback. The position
//     channel is genuinely destroyed by that lerp, but the VELOCITY channels are not —
//     `w1`/`w2` are never wrapped. So the guard watches the interpolated angle against
//     the exact floor-frame anchor (`frame.state.derived`, which the host does NOT
//     interpolate) and, whenever the drift is unreachable or points the wrong way, hands
//     the render over to the pose's own angular velocity for at most one control step.
//     Same machinery, same frame, covers the post-success respawn: the recorded goal row
//     is the all-zero hang, so the host slides the whole arm home for 200 ms BEFORE the
//     teleport event arrives (the mountain-car superluminal lesson, canon 4). The guard
//     holds the arm at its apex instead, which is where the celebration then fires.
//
//     THE GUARD'S CLOCK IS THE PLAYBACK CLOCK, NEVER `dtReal`. The timeline it is
//     correcting — the pose lerp — advances in PLAYBACK seconds (`dtReal · clock.speed`;
//     the transport ships 0.25/0.5/1/2/4×), so a guard integrating on WALL seconds is
//     right only at 1× and is worse than no guard at 0.25× (the one-control-step budget
//     burns 4× early and releases mid-wrap: a measured 1.6-unit single-frame teleport of
//     the lantern, and the apex the celebration reads collapses to the reset hang). At
//     Acrobot's controlHz 5 one playback second IS one sim second (5 × AB_DT = 1), so
//     `dtPlay` integrates `w` in the env's own units, exactly, at every transport speed —
//     and is identically 0 while paused, which makes update() a literal no-op (canon:
//     `ctx.time()` for ALL animation time; pause must freeze the world coherently).
//
//  2. THE TIP RIBBON is the one visual in this scene that is NOT a pure function of
//     state (canon lesson 2). An acrobot's tip path cannot be recovered from one pose the
//     way MountainCar's momentum trail is recovered from the terrain curve, so actors.ts
//     keeps a PREALLOCATED ring buffer of (x, y, playbackTime). It is cleared on every
//     `scrubbed` frame and every `teleport`, its samples are stamped with `ctx.time()` so
//     a pause freezes it rather than draining it, and it allocates nothing per frame.
//
// Everything else obeys the canon: celebrations fire ONLY from `frame.events`, every
// flourish clock is `ctx.time()`, and `done` here always accompanies `goal` (the kernel
// only ever raises done on `tipHeight > 1`), so this scene never sighs.

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
import { AB_DT, AB_L1, AB_L2 } from '@sim/envs/acrobot';
import { buildArm, type ArmHandle } from './actors';
import { manifest } from './manifest';
import {
  EXACT_POINT_FLOATS,
  acrobotPointsExact,
  buildWorld,
  clampRange,
  pivotAnchor,
  successAnchor,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. Near-profile, because the physics is planar: a 3/4 view turns a 2-link
// pendulum into "some sticks". The camera sits 5.4 units out on the swing plane's own
// axis, a touch right and just above the water, which at 16:9 frames x ∈ [−3.5, +3.5]
// and y ∈ [−2.5, +1.45] — the whole 2-unit reach, the string of lights at y = 1, and the
// pond it skims, with a few degrees of offset so the water reads as a plane.
const CAM_BASE: readonly [number, number, number] = [0.58, -0.24, 5.7];
const CAM_LOOK: readonly [number, number, number] = [-0.06, -0.5, 0];
/** thumb pulls back so the full reach survives a square-ish crop */
const CAM_BASE_THUMB: readonly [number, number, number] = [0.28, -0.34, 6.9];
const CAM_LOOK_THUMB: readonly [number, number, number] = [-0.04, -0.55, 0];

// celebration/flourish envelopes, all measured on ctx.time()
const FLARE_SECONDS = 1.9;
const PETAL_SECONDS = 5;
const FLASH_SECONDS = 0.3;
/** one slow breath of the morning: fireflies out, mist burning off, and back again */
const DAWN_PERIOD = 96;

// Celebration weights (canon lesson 6): this diorama is ~5 units of subject seen from
// 5.4, so the default full-strength grammar whites the pond out. Half the amplitude, all
// three beats.
const BURST_SCALE = 0.5;
const RIPPLE_SCALE = 0.34;
const BURST_PARTICLES = 110;
const PULSE_STRENGTH = 0.42;

// ---- the arc guard's thresholds ----
/**
 * How far past `|w| · AB_DT` a true displacement may run inside one control step. The
 * integrator is RK4, so the angular rate at the END of a step can be well past the rate
 * at its start; 1.7 covers the acceleration without ever admitting a 2π sweep.
 */
const REACH_SLACK = 1.7;
/** a floor on the reach so a nearly-still joint is not tripped by fp noise (≈3°) */
const REACH_FLOOR = 0.05;
/** the wrong-way test needs a real displacement before it fires (≈3°) */
const BACKLASH = 0.05;

const TAU = Math.PI * 2;

/** Wrap an angle into [−π, π). */
export function wrapPi(a: number): number {
  const t = (a + Math.PI) % TAU;
  return (t < 0 ? t + TAU : t) - Math.PI;
}

/**
 * Is the interpolated angle `raw` reachable from the exact floor-frame anchor?
 * `wAnchor` is the anchor's own angular velocity (exact), `wPose` the interpolated one.
 * Exported for the test: the wrap and the respawn slide must both come back true.
 */
export function arcCorrupt(drift: number, wAnchor: number, wPose: number): boolean {
  const speed = Math.max(Math.abs(wAnchor), Math.abs(wPose));
  const reach = speed * AB_DT * REACH_SLACK + REACH_FLOOR;
  if (Math.abs(drift) > reach) return true;
  // the long way round always sets off in the opposite direction to the true motion
  return drift * wAnchor < 0 && Math.abs(drift) > BACKLASH;
}

/**
 * Exponential decay in [0, 1] over `span` playback seconds, 0 outside the window. The
 * linear tail is not cosmetic: a bare exp() still has amplitude when the window closes,
 * so every flourish would end on a one-frame pop (canon lesson 6).
 */
export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

interface AcrobotLike {
  th1?: unknown;
  th2?: unknown;
  w1?: unknown;
  w2?: unknown;
  torque?: unknown;
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/** torque ∈ {−1, 0, +1}: sim-core's decode when present, the raw ctrl otherwise. */
export function torqueOf(state: FrameState): number {
  const d = state.derived as AcrobotLike | null | undefined;
  if (d && typeof d === 'object' && typeof d.torque === 'number') return d.torque;
  const raw = state.ctrl[0] ?? 0;
  if (!Number.isFinite(raw)) return 0;
  return ((Math.trunc(raw) >>> 0) % 3) - 1;
}

class WillowSwingScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private arm: ArmHandle | null = null;
  private rig: FrameRig | null = null;
  private pool: CelebrationPool | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private accent = 0xf2a7c3;
  private gold = 0xffd166;
  private pivot: readonly [number, number, number] = [0, 0, 0];
  private success: readonly [number, number, number] = [0, 1, 0];

  // ---- flourish clocks (playback seconds; -Infinity = idle) ----
  private flareAt = -Infinity;
  private petalAt = -Infinity;
  private flashAt = -Infinity;

  // ---- the arc guard ----
  /** the CONTINUOUS rendered angles; equal to the pose (mod 2π) whenever inactive */
  private readonly render = new Float32Array(2);
  /** PLAYBACK seconds each angle has been carried by its velocity channel */
  private readonly held = new Float32Array(2);
  /** previous `ctx.time()`; NaN = no previous frame, so the first dtPlay is 0 */
  private lastTime = Number.NaN;
  private primed = false;

  // ---- preallocated scratch (update() may not allocate) ----
  private readonly points = new Float64Array(EXACT_POINT_FLOATS);
  private readonly overlayPos: [number, number, number] = [0, 0, 0];
  private readonly ripplePos = new THREE.Vector3();
  private readonly celebrateAt = new THREE.Vector3();

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'willowSwingScene';

    const palette = ctx.tokens.palettes.dawn;
    this.accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.gold = ctx.tokens.celebrationGold;
    this.pivot = pivotAnchor(ctx.geometry);
    this.success = successAnchor(ctx.geometry);
    const success = this.success;

    const world = buildWorld(ctx, this.scene);
    this.world = world;
    this.scene.add(world.group);

    const arm = buildArm({
      tier: ctx.tier,
      palette,
      accent: this.accent,
      gold: this.gold,
      seed: ctx.seed,
      pivot: this.pivot,
    });
    this.arm = arm;
    this.scene.add(arm.group);

    // This scene claims three of the rig's point lights. Colours are STATIC and the
    // garland's position never moves, so update() never touches a Color. points[] is
    // empty at thumb; the guards are the tier policy.
    const lantern = world.rig.points[0];
    if (lantern) lantern.color.set(palette.accentLight);
    const garlandLight = world.rig.points[1];
    if (garlandLight) {
      garlandLight.color.set(this.gold);
      garlandLight.position.set(success[0], success[1] + 0.16, success[2]);
    }
    const celebration = world.rig.points[2];
    if (celebration) celebration.color.set(this.accent);

    const thumb = ctx.tier === 'thumb';
    const rig = makeFrameRig({
      position: thumb ? CAM_BASE_THUMB : CAM_BASE,
      lookAt: thumb ? CAM_LOOK_THUMB : CAM_LOOK,
      fov: ctx.tokens.fovHero,
      near: 0.05,
      far: 400,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      canvas: thumb ? null : ctx.canvas,
      parallax: 0.12,
      parallaxY: 0.5,
      damping: 2.4,
      name: 'willowFrameRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'willowCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      // the cross rides the TRUTH (the raw pose), never the guarded render — that is
      // the whole point of the instrument
      agentPos: (state) => {
        acrobotPointsExact(state.pose[0] ?? 0, state.pose[1] ?? 0, this.points);
        this.overlayPos[0] = this.pivot[0] + (this.points[2] ?? 0);
        this.overlayPos[1] = this.pivot[1] + (this.points[3] ?? 0);
        this.overlayPos[2] = this.pivot[2];
        return this.overlayPos;
      },
      name: 'willowGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    arm.setPixelRatio(ctx.size.dpr);
    world.setPixelRatio(ctx.size.dpr);
    world.setDawn(0.45);
    world.setFlare(-1);
    world.setPetals(0);
    world.setGlint(0, 0, 0);
    arm.setLanternGain(1);
    arm.setState(0, 0, 0, 0);
    arm.setTime(0);
    world.setTime(0);
    this.celebrateAt.copy(arm.tipWorld);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const arm = this.arm;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !arm || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;

    // 0. PLAYBACK seconds since the previous frame — the arc guard's only clock. This is
    //    what the pose timeline advances on, so it is the only dt that can be compared
    //    with a control step or integrated against an angular velocity. It is 0 on the
    //    first frame, 0 while paused (a frozen clock re-delivering the same frame), and
    //    0 across a wrap of the transport (time running backwards), which is exactly the
    //    "no motion happened" the guard should see in all three cases.
    const prevTime = this.lastTime;
    const dtPlay = Number.isFinite(prevTime) && time > prevTime ? time - prevTime : 0;
    this.lastTime = time;

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. a scrub wipes every in-flight beat: nothing may survive a discontinuity
    if (frame.scrubbed) {
      pool.clear();
      this.flareAt = -Infinity;
      this.petalAt = -Infinity;
      this.flashAt = -Infinity;
      this.primed = false;
      arm.clearTrail();
      rig.snap();
    }

    // 3. events — celebrations are NEVER inferred from the state stream.
    //    `celebrateAt` is the PREVIOUS frame's rendered tip, which under the arc guard is
    //    the apex the lantern actually reached: the recorded goal row is already the
    //    all-zero hang, so reading the tip out of this frame's pose would bloom at the
    //    bottom of the swing, a full 3 units from where the success happened.
    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        pool.burst(this.celebrateAt, this.accent, BURST_SCALE, time);
        this.ripplePos.set(this.celebrateAt.x, world.waterY, this.celebrateAt.z);
        pool.ripple(this.ripplePos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        this.flareAt = time;
        this.petalAt = time;
        this.flashAt = time;
        const light = world.rig.points[2];
        if (light) light.position.copy(this.celebrateAt);
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
      // `done` here IS the +1 (the kernel only raises it on tipHeight > 1): never a sigh.
    }

    // 4. the arc guard (see the header note)
    const rawTh1 = state.pose[0] ?? 0;
    const rawTh2 = state.pose[1] ?? 0;
    const wPose1 = state.pose[2] ?? 0;
    const wPose2 = state.pose[3] ?? 0;
    const d = state.derived as AcrobotLike | null | undefined;
    const anchor1 = num(d?.th1, rawTh1);
    const anchor2 = num(d?.th2, rawTh2);
    const wAnchor1 = num(d?.w1, wPose1);
    const wAnchor2 = num(d?.w2, wPose2);

    if (teleported || !this.primed) {
      this.render[0] = rawTh1;
      this.render[1] = rawTh2;
      this.held[0] = 0;
      this.held[1] = 0;
      this.primed = true;
      // a respawn is a hard cut: the ribbon may not stretch across it
      arm.clearTrail();
    } else {
      this.follow(0, rawTh1, anchor1, wAnchor1, wPose1, dtPlay);
      this.follow(1, rawTh2, anchor2, wAnchor2, wPose2, dtPlay);
    }
    const th1 = this.render[0] ?? 0;
    const th2 = this.render[1] ?? 0;

    // 5. envelopes — every one of them measured on the playback clock
    const flareAge = time - this.flareAt;
    const flare = flareAge >= 0 && flareAge <= FLARE_SECONDS ? flareAge / FLARE_SECONDS : -1;
    const petals = envelope(time - this.petalAt, PETAL_SECONDS, 0.9);
    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.2);

    // 6. the state visuals
    const tipSpeed = acrobotTipSpeed(th1, th2, wPose1, wPose2);
    arm.setState(th1, th2, torqueOf(state), tipSpeed);
    arm.setLanternGain(1 + 2.4 * flash);
    if (!frame.scrubbed) arm.pushTrail(time);
    arm.refreshTrail(time);

    // the lantern's glint pools on the water and spreads as it climbs away from it
    const height01 = clampRange((arm.tipWorld.y - world.waterY) / 3.2, 0, 1);
    const glint = (1 - height01) * (1 - height01) * (0.55 + 0.45 * flash);
    world.setGlint(arm.tipWorld.x, arm.tipWorld.z, glint);

    world.setFlare(flare);
    world.setPetals(petals);
    // one slow breath of the morning — pauses coherently with everything else
    world.setDawn(0.45 + 0.4 * Math.sin((time * TAU) / DAWN_PERIOD));

    // 7. the three point lights this scene claims from the rig
    const lantern = world.rig.points[0];
    if (lantern) {
      lantern.position.copy(arm.lanternWorld);
      lantern.intensity = 0.5 + 2.6 * flash;
    }
    const garlandLight = world.rig.points[1];
    if (garlandLight) {
      garlandLight.intensity = 0.05 + 1.5 * (flare >= 0 ? 1 - flare : 0);
    }
    const celebration = world.rig.points[2];
    if (celebration) {
      celebration.intensity = 2.4 * envelope(time - this.flareAt, FLARE_SECONDS, 2.4);
    }

    // 8. one clock for the whole world
    world.setTime(time);
    arm.setTime(time);

    // 9. camera easing runs in WALL time so a paused frame is still explorable
    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);

    // 10. remember where the lantern ended up — the next `goal` blooms there
    this.celebrateAt.copy(arm.tipWorld);
  }

  /**
   * Advance one rendered angle toward the interpolated pose, or — when that
   * interpolation is crossing a ±π wrap or a respawn — carry it on the pose's own
   * angular-velocity channel for at most one control step.
   *
   * `dt` is PLAYBACK seconds (see update() step 0), which at controlHz 5 are the env's
   * own seconds: both the integration and the `AB_DT` budget are therefore exact at every
   * transport speed. `dt === 0` (paused, or the first frame) makes this a literal no-op
   * rather than "integrate by zero" — `wrapPi` is not bit-exactly idempotent near 0, and
   * a paused host re-delivers the same frame indefinitely.
   */
  private follow(
    k: number,
    raw: number,
    anchor: number,
    wAnchor: number,
    wPose: number,
    dt: number,
  ): void {
    if (arcCorrupt(raw - anchor, wAnchor, wPose)) {
      if ((this.held[k] ?? 0) < AB_DT) {
        if (dt > 0) {
          this.held[k] = (this.held[k] ?? 0) + dt;
          this.render[k] = wrapPi((this.render[k] ?? 0) + wPose * dt);
        }
        return;
      }
      // The budget is ONE control step. Past it the pose is taken at its word and the
      // budget is deliberately NOT refunded while the drift stays unreachable, or the
      // guard would oscillate between carrying and releasing forever.
      this.render[k] = raw;
      return;
    }
    // Straight onto the truth. Nothing is eased and nothing is unwrapped: a rotation is
    // 2π-periodic, so releasing the guard onto `raw` — even from the far side of a wrap —
    // is visually continuous, it is a fixed point (update() must be idempotent), and it
    // makes every probe EXACT rather than merely close.
    this.held[k] = 0;
    this.render[k] = raw;
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.arm?.setPixelRatio(size.dpr);
    this.world?.setPixelRatio(size.dpr);
    this.pool?.setPixelRatio(size.dpr);
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
  }

  probes(): AlignmentProbe[] {
    const pivot = this.pivot;
    const arm = this.arm;
    const empty = new THREE.Object3D();
    const points = this.points;
    const success = this.success;
    return [
      {
        name: 'pivot',
        object: arm?.pivotRoot ?? empty,
        expectWorldPos: () => [pivot[0], pivot[1], pivot[2]] as const,
      },
      {
        name: 'elbow',
        object: arm?.elbowRoot ?? empty,
        expectWorldPos: (state) => {
          acrobotPointsExact(state.pose[0] ?? 0, state.pose[1] ?? 0, points);
          return [pivot[0] + (points[0] ?? 0), pivot[1] + (points[1] ?? 0), pivot[2]] as const;
        },
      },
      {
        name: 'lantern',
        object: arm?.tipRoot ?? empty,
        expectWorldPos: (state) => {
          acrobotPointsExact(state.pose[0] ?? 0, state.pose[1] ?? 0, points);
          return [pivot[0] + (points[2] ?? 0), pivot[1] + (points[3] ?? 0), pivot[2]] as const;
        },
      },
      {
        name: 'success-line',
        object: this.world?.garlandRoot ?? empty,
        expectWorldPos: () => [success[0], success[1], success[2]] as const,
      },
    ];
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.arm?.dispose();
    this.arm = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.flareAt = -Infinity;
    this.petalAt = -Infinity;
    this.flashAt = -Infinity;
    this.primed = false;
    this.lastTime = Number.NaN;
    this.held[0] = 0;
    this.held[1] = 0;
  }
}

/**
 * |d(tip)/dt| from the pose alone: the free end's velocity is
 * `w1 · (cos th1, sin th1) · L1 + (w1 + w2) · (cos(th1+th2), sin(th1+th2)) · L2`,
 * which is the exact derivative of `acrobotPoints`. Used only for how HARD the ribbon
 * and the flame read, never for where anything is.
 */
export function acrobotTipSpeed(th1: number, th2: number, w1: number, w2: number): number {
  const a = th1;
  const b = th1 + th2;
  const wb = w1 + w2;
  const vx = w1 * Math.cos(a) * AB_L1 + wb * Math.cos(b) * AB_L2;
  const vy = w1 * Math.sin(a) * AB_L1 + wb * Math.sin(b) * AB_L2;
  return Math.hypot(vx, vy);
}

const createScene: SceneFactory = () => new WillowSwingScene();
export default createScene;
