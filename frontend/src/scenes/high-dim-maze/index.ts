// The Slot Canyons — HighDimMaze-cont-v0.
//
// THE ONE IDEA: the maze's SHAPE is the location. HighDimMaze is three north-south
// corridors cut by two solid fins, with one hole punched in each — so it is three slot
// canyons joined by two natural stone arches, and the serpentine the agent has to learn
// is a walk you can see. Nothing is re-derived: cliffs come from `grid.walls`, the arches
// from `grid.hallways`, their ORIENTATION from the wall cells flanking each gap, and the
// sun/shade split on the sand from a scan of the walls array.
//
// WHAT THIS SCENE ADDS TO THE GRID CANON
//  * A FACING. HighDimMaze is the only grid env that records one: `pose[4]` is `lastDir`,
//    the last cardinal action, and it is not cosmetic — it is the direction a `dash`
//    would travel. The wisp turns to it and wears a scarf that streams behind, so a
//    single frame tells you which way the agent is pointed and where a dash would go.
//    Because `Wisp.face()` yaws the body the orb hangs from, this scene probes
//    `wisp.avatar` — the exact-cell node — and never `core` (THE WISP PROBE RULE).
//  * SIX ACTIONS. 4 = stay (the wisp settles; the scarf goes slack, disturbed sand rings
//    under it, dust sifts down) and 5 = dash (up to two cells along `lastDir` inside one
//    8 Hz step: speed lines strung along the segment and a sand wake lying on the cells
//    crossed). `HopMotion` needs no help for the two-cell case — a dash is collinear by
//    construction, so the straight hop IS the path, travelled twice as fast.
//  * A TELEPORTING AGENT. Unlike FourRooms, this env redraws BOTH the agent and the goal
//    on a +1. So the celebration cannot bloom where the agent is — by the time the event
//    arrives the agent is somewhere else entirely. It blooms on the cell the agent
//    SCORED on, which is the goal cell of the previous frame, and the wisp hard-cuts to
//    its new cell on the `teleport` that rides along with it.
//  * `done` here is only ever the +1 (the env has no failure), so this scene never sighs.

import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { CELL, cellToWorld, cellToWorldVec, clampCell, type GridSize, type Vec3Tuple } from '@kit/grid';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { makeOrbitRig, type OrbitRig } from '@kit/rigs';
import { TOKENS } from '@kit/tokens';
import type {
  AlignmentProbe,
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import { Wisp, yawFromGridDelta } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { HIGHDIMMAZE_DASH, HIGHDIMMAZE_STAY } from '@sim/envs/highdimmaze';
import { buildGust, buildOasis, buildScarf, type GustHandle, type OasisHandle, type ScarfHandle } from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  blendTokens,
  buildWorld,
  canyonSkin,
  clamp01,
  resolveGrid,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. A 3/4 view from the south-south-east, 41° above the horizon. THE AZIMUTH IS
// NOT A MATTER OF TASTE: the fins run north-south, so every degree of yaw drives the
// sight-line sideways while it crosses a fin's own thickness, and past a measured limit the
// fin hides the whole corridor beside it. The first pass sat at 0.36 rad believing a camera
// there "looks along" the fins; ray-traced against the real cliffs, it hid the agent
// completely on 8 of the 65 cells it can stand on, including the entire west corridor on
// the mandatory route out of the start. `world.ts` carries the measured table; the frame
// below is the most oblique corner of it that leaves EVERY cell readable, and the rig's
// yaw limits are clamped to that corner so it cannot be orbited away.
//
// The ELEVATION is the second half of the bargain. The first pass sat at 52°, where the
// biggest surface in the frame is cliff TOPS and the scene renders as a low-walled maze —
// accurate, and not a canyon. At 41° the faces win, the strata read, and the terracotta
// that this location is named for is finally what you look at.
//
// Distance is arithmetic, not taste: at azimuth a an 11-unit grid presents
// 11·(|cos a| + |sin a|) = 13.4 units across and 13.4·cos(polar) + finHeight·sin(polar) ≈
// 9.7 down the frame, and a 40° vertical FOV covers 2·d·tan20° = 0.728·d — so d must clear
// 13.3 to touch the edges. 17.0 leaves the margin that makes it a composition.
const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.86;
const CAM_DISTANCE = 17;
const CAM_TARGET_Y = 0.62;
/**
 * How far the user may orbit either side of the composed azimuth. Measured, like the
 * azimuth itself: across [0.21, 0.29] every free cell still shows the agent from every
 * distance and elevation the rig allows, and past that the corridor west of the x = 3 fin
 * begins to close over the cells behind it.
 */
const CAM_YAW_RANGE = 0.04;
/** thumb pulls back and stops moving: a square-ish crop must still hold 11 × 11 cells */
const CAM_DISTANCE_THUMB = 20;
const CAM_AZIMUTH_THUMB = CAMERA_AZIMUTH;
const CAM_POLAR_THUMB = 0.82;

// Celebration weights. The canyon is 11 units of sunlit terracotta — already the warmest
// palette in the preserve — so the kit's full-strength grammar (a 3.2-unit ripple and a
// 1.0 bloom pulse) washes three cells of grid out at once. Two thirds of the amplitude,
// all three beats (canon: celebration amplitude scales with scene size).
/** the burst travels ~0.76 world units — a little under one cell */
const BURST_SCALE = 0.84;
const RIPPLE_SCALE = 0.52;
const BURST_PARTICLES = 110;
const PULSE_STRENGTH = 0.5;
/** the spray leaps from the pool's surface, not off the sand */
const BLOOM_Y = 0.34;
const BLOOM_SECONDS = 1.5;
/** a new oasis takes this long to well up at its new cell (never a teleport-slide) */
const GROW_SECONDS = 0.8;
/** the overshoot at the top of that grow-in: peaks at scale 1.06 around t = 2/3 */
const GROW_BOUNCE = 0.65;
/** the sand a dash kicked up hangs this long */
const WAKE_SECONDS = 0.62;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;

/** the wisp's facing damping — mirrored by the scarf so the two can never disagree */
const FACE_RATE = 12;

/** how many cells out the oasis starts pulling the eye */
const PROXIMITY_CELLS = 4;

interface HighDimMazeLike {
  x?: unknown;
  y?: unknown;
  gx?: unknown;
  gy?: unknown;
  lastDir?: unknown;
}

/** The four numbers a frame of this env is made of, snapped to cells. */
interface Reading {
  agent: GridCell;
  goal: GridCell;
  lastDir: number;
  action: number;
}

function number(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/**
 * Agent cell, goal cell, facing and action for a frame. sim-core's decoded state is the
 * source; the raw pose (`[x, y, gx, gy, lastDir, …]`) is the fallback so the scene still
 * stands up under the harness's fixture loop, which carries no `derived`.
 */
function read(state: FrameState, grid: GridSize, out: Reading): Reading {
  const d = state.derived as HighDimMazeLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  out.agent.x = Math.round(number(has ? d.x : undefined, state.pose[0] ?? 0));
  out.agent.y = Math.round(number(has ? d.y : undefined, state.pose[1] ?? 0));
  out.goal.x = Math.round(number(has ? d.gx : undefined, state.pose[2] ?? 0));
  out.goal.y = Math.round(number(has ? d.gy : undefined, state.pose[3] ?? 0));
  clampCell(out.agent, grid);
  clampCell(out.goal, grid);
  const dir = Math.round(number(has ? d.lastDir : undefined, state.pose[4] ?? 0));
  out.lastDir = ((dir % 4) + 4) % 4;
  const a = Math.trunc(number(state.ctrl[0], 0));
  out.action = ((a % 6) + 6) % 6;
  return out;
}

class SlotCanyonsScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private scarf: ScarfHandle | null = null;
  private gust: GustHandle | null = null;
  private oasis: OasisHandle | null = null;
  private settleDust: AmbientParticlesHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private growAt = -Infinity;
  private bloomAt = -Infinity;
  private dashAt = -Infinity;

  // ---- preallocated scratch: update() may not allocate ----
  private readonly now: Reading = { agent: { x: 0, y: 0 }, goal: { x: 0, y: 0 }, lastDir: 1, action: 0 };
  private readonly before: Reading = { agent: { x: 0, y: 0 }, goal: { x: 0, y: 0 }, lastDir: 1, action: 0 };
  private readonly lastGoal: GridCell = { x: -1, y: -1 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  /** the segment THIS frame's dash is travelling (collapsed when a dash moved nothing) */
  private readonly dashFrom = new THREE.Vector3();
  private readonly dashTo = new THREE.Vector3();
  /** the segment the last dash that actually MOVED crossed — the wake outlives its dash */
  private readonly wakeFrom = new THREE.Vector3();
  private readonly wakeTo = new THREE.Vector3();
  private scarfYaw = 0;
  private spray = 0x5ee0c8;
  private gold = 0xffd166;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'slotCanyonsScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;

    const palette = ctx.tokens.palettes.afternoon;
    const skin = canyonSkin(palette, ctx.tokens);
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    // THE WISP IS AN EMBER, NOT A FLASHBULB. `wispMaterial` is additive at 1.9 gain with
    // an additive halo on top, so pushing the accent toward the sun (the first pass blended
    // 0.42) just saturates every channel and the avatar renders as a white hole in the
    // sand. Kept at the raw accent it stays ORANGE against a warm floor — and the scarf,
    // which is a lit surface rather than a glow, is what gives it a silhouette.
    const ember = accent;
    // the +1 is WATER LEAPING OUT OF THE POOL — the only cool colour the canyon owns.
    // The burst is additive at 2.4 gain, so a pale teal saturates to white and the beat
    // loses the one thing that made it belong to this scene; the deep pool colour is what
    // survives the gain.
    this.spray = blendTokens(skin.water, skin.waterDeep, 0.4).getHex();
    this.gold = ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    // This scene claims three of the rig's point lights. Their COLORS are static — only
    // position and intensity are state-driven — so update() never touches a Color.
    const oasisLight = world.rig.points[0];
    if (oasisLight) oasisLight.color.set(skin.shaft);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const bloomLight = world.rig.points[2];
    if (bloomLight) bloomLight.color.set(this.spray);

    // ---- the shared avatar ----
    const wisp = new Wisp(
      {
        color: ember,
        tier: ctx.tier,
        // a shade over the 0.16 default: from the composed 19.4-unit standoff the default
        // orb is ~13 px, and 13 px of warm glow on warm sand is not an avatar
        radius: 0.165,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'canyonWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    // the scarf hangs off `avatar` — the exact-cell node — so the probe stays bit-exact
    const scarf = buildScarf({
      tier: ctx.tier,
      color: blendTokens(accent, TOKENS.palettes.dusk.skyHorizon, 0.3),
      height: ctx.tokens.wispHeight * 0.92,
      name: 'canyonScarf',
    });
    this.scarf = scarf;
    wisp.avatar.add(scarf.group);

    // ---- dash + stay trappings (world space, pooled, invisible at rest) ----
    const gust = buildGust({
      tier: ctx.tier,
      skin,
      color: blendTokens(accent, palette.sun, 0.5),
      name: 'canyonGust',
    });
    this.gust = gust;
    this.scene.add(gust.group);

    // dust that sifts down around a wisp holding station — a field that follows the
    // avatar and whose density is a PURE function of the action, not of a clock
    const settleDust = makeAmbientParticles({
      preset: 'dust',
      count: 110,
      bounds: { x: 0, y: 0, z: 0, halfX: 0.4, halfY: 0.28, halfZ: 0.4 },
      seed: ctx.seed ^ 0x5e77,
      tier: ctx.tier,
      color: skin.dust,
      size: 0.9,
      intensity: 1.4,
      opacity: 0.55,
      rise: -0.2,
      swirl: 0.06,
      name: 'canyonSettleDust',
    });
    settleDust.setDensity(0);
    this.settleDust = settleDust;
    this.scene.add(settleDust.points);

    // ---- the goal: an oasis on whatever cell the goal has respawned to ----
    const oasis = buildOasis({ tier: ctx.tier, palette, skin, name: 'oasisAlcove' });
    this.oasis = oasis;
    this.scene.add(oasis.group);

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [0, CAM_TARGET_Y, 0],
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: thumb ? CAM_AZIMUTH_THUMB : CAM_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      limits: {
        // THE WHOLE ENVELOPE IS AUDITED, not just the composed pose: every corner of these
        // limits was ray-traced against the real cliffs and keeps all 65 free cells showing
        // the agent (the test re-measures the corners). Never so low that the near rim hides
        // the front row, never so high the fins lose their faces and the canyon flattens
        // into a plan, and never yawed far enough for a fin to swallow its own corridor.
        minPolar: 0.74,
        maxPolar: 0.88,
        minAzimuth: CAM_AZIMUTH - CAM_YAW_RANGE,
        maxAzimuth: CAM_AZIMUTH + CAM_YAW_RANGE,
        minDistance: 17,
        maxDistance: 24,
      },
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0012,
      name: 'canyonOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'canyonCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
    const overlayReading: Reading = {
      agent: { x: 0, y: 0 },
      goal: { x: 0, y: 0 },
      lastDir: 1,
      action: 0,
    };
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => read(state, grid, overlayReading).agent,
      goalCell: (state) => read(state, grid, overlayReading).goal,
      name: 'canyonGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    oasis.setState(1, 0);
    oasis.setTime(0);
    scarf.setState(0, 1, 0, 0);
    gust.clear();
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const scarf = this.scarf;
    const gust = this.gust;
    const oasis = this.oasis;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !scarf || !gust || !oasis || !pool || !rig) return;

    const time = ctx.time();
    const grid = this.grid;
    const now = read(frame.state, grid, this.now);
    const agent = now.agent;
    const goal = now.goal;

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. a scrub wipes every in-flight beat: nothing may survive a discontinuity
    if (frame.scrubbed) {
      pool.clear();
      gust.clear();
      this.bloomAt = -Infinity;
      this.growAt = -Infinity;
      this.clearDash(agent);
      wisp.setCell(agent.x, agent.y, grid);
      this.scarfYaw = facingYaw(now.lastDir);
      wisp.face(this.scarfYaw);
      wisp.snap();
      rig.snap();
      this.lastGoal.x = goal.x;
      this.lastGoal.y = goal.y;
    }

    // 3. events — celebrations are NEVER inferred from the state stream
    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        // THE +1 HAPPENED WHERE THE GOAL WAS, NOT WHERE THE AGENT IS. This env redraws
        // BOTH on completion, so by the time the event lands the agent has already been
        // rejection-sampled somewhere else entirely and the goal is on a new cell. The
        // only cell that means anything is the previous frame's goal — the alcove the
        // agent actually reached.
        this.bloomCell(frame, this.groundPos);
        this.bloomPos.copy(this.groundPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.spray, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        oasis.flare(0.8);
        this.bloomAt = time;
        this.growAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
      // `done` in this env is only ever the +1 — there is no failure here, so this scene
      // must never sigh. Sighing would be a lie about what the agent experienced.
    }

    // 4. a teleport is a HARD CUT: the agent was rejection-redrawn across the maze and
    //    must never be seen gliding there
    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      this.scarfYaw = facingYaw(now.lastDir);
      wisp.face(this.scarfYaw);
      wisp.snap();
      gust.clear();
      this.clearDash(agent);
      rig.snap();
    }

    // 5. the avatar: one straight hop per cell change, never a path plan. A dash is
    //    collinear along `lastDir` by construction, so its two-cell hop IS the path the
    //    agent took — the same straight segment, travelled twice as fast.
    const moved = agent.x !== wisp.cellX || agent.y !== wisp.cellY;
    const dashing = now.action === HIGHDIMMAZE_DASH ? 1 : 0;
    if (dashing > 0) {
      // A DASH THAT MOVES NOTHING IS AN ORDINARY, LEGAL TRANSITION. `gw_hdm_step` walks up
      // to two cells along `lastDir` and stops at the first wall, so action 5 with a wall
      // straight ahead (5 at (1,1) with lastDir = up) advances zero cells. The segment is
      // therefore refreshed on EVERY dash frame, not only when the cell changed: refreshing
      // it only on a move left the PREVIOUS dash's from/to sitting in these vectors, and
      // this frame's speed lines were drawn along that stale segment — cells away from the
      // wisp, across a different corridor. When the dash moved nothing the segment collapses
      // onto the wisp's own cell, which draws as nothing at all.
      cellToWorldVec(wisp.cellX, wisp.cellY, grid, this.dashFrom);
      cellToWorldVec(agent.x, agent.y, grid, this.dashTo);
      if (moved && isDashSegment(this.dashFrom.x, this.dashFrom.z, this.dashTo.x, this.dashTo.z)) {
        // the wake outlives the dash that laid it, so it keeps its own copy of the segment
        this.wakeFrom.copy(this.dashFrom);
        this.wakeTo.copy(this.dashTo);
        this.dashAt = time;
      }
    }
    if (moved) wisp.hopTo(agent, time);
    wisp.face(facingDelta(now.lastDir));
    wisp.update(frame.dtReal, time);

    // 6. the scarf mirrors the wisp's own damped turn (identical law, identical inputs),
    //    goes slack when the wisp is settled and snaps taut in a gust
    this.scarfYaw = dampAngle(this.scarfYaw, facingYaw(now.lastDir), frame.dtReal);
    const staying = now.action === HIGHDIMMAZE_STAY ? 1 : 0;
    const airborne = wisp.motion.airborne ? 1 - wisp.motion.progress : 0;
    const speed = clamp01(airborne * (0.55 + 0.45 * dashing) + 0.35 * dashing);
    scarf.setState(this.scarfYaw, staying, speed, time);

    // 7. the gust: speed lines while a dash is in the air, a wake on the cells it crossed.
    //    Both are drawn ONLY on a segment the env could actually have produced — one or two
    //    cells, collinear. The env guarantees that; the inputs this code accepts do not
    //    (the harness's fixture loop plays unrelated poses back to back), and a diagonal
    //    streak with a ground wake laid through solid rock is a lie about the step.
    const wake = envelope(time - this.dashAt, WAKE_SECONDS, 2.6);
    if (dashing > 0 && isDashSegment(this.dashFrom.x, this.dashFrom.z, this.dashTo.x, this.dashTo.z)) {
      gust.setLines(
        this.dashFrom.x,
        this.dashFrom.z,
        this.dashTo.x,
        this.dashTo.z,
        ctx.tokens.wispHeight * 0.8,
        wisp.motion.progress,
        clamp01(0.35 + 0.65 * airborne),
      );
    } else {
      gust.setLines(0, 0, 0, 0, 0, 1, 0);
    }
    gust.setWake(this.wakeFrom.x, this.wakeFrom.z, this.wakeTo.x, this.wakeTo.z, wake);
    gust.setSettle(wisp.position.x, wisp.position.z, staying, time);

    // 8. the settled wisp's dust: density is a PURE function of the action
    const settleDust = this.settleDust;
    if (settleDust) {
      settleDust.points.position.set(wisp.position.x, ctx.tokens.wispHeight * 0.7, wisp.position.z);
      settleDust.setDensity(staying);
      settleDust.setTime(time);
    }

    // 9. the oasis: position is pure state, growth is pure event clock
    cellToWorldVec(goal.x, goal.y, grid, oasis.root.position);
    const grow = growth(time - this.growAt);
    oasis.setState(grow, nearness(agent, goal));
    oasis.setTime(time);
    oasis.update(frame.dtReal, time);

    // 10. the three point lights this scene claims from the rig
    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const oasisLight = world.rig.points[0];
    if (oasisLight) {
      oasisLight.position.set(oasis.root.position.x, oasis.crownHeight, oasis.root.position.z);
      oasisLight.intensity = (0.2 + 0.5 * nearness(agent, goal)) * grow;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.18 + 0.9 * bloom;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 2.4 * bloom;
    }

    // 11. one clock for the whole canyon
    world.setTime(time);

    // 12. camera easing runs in WALL time so a paused frame is still explorable
    rig.update(frame.dtReal);

    // 13. remember this frame's truth — the +1 fires one step after the cell that earned it
    this.lastGoal.x = goal.x;
    this.lastGoal.y = goal.y;

    if (this.debugOn && this.overlay) this.overlay.setState(frame.state);
  }

  /**
   * Kill the dash trappings with no easing: the clock stops and BOTH segments collapse onto
   * the wisp's own cell, so nothing can be left over to draw across a discontinuity. A
   * scrub or a teleport is a cut — the sand a dash kicked up three cells ago did not
   * survive it.
   */
  private clearDash(agent: GridCell): void {
    this.dashAt = -Infinity;
    cellToWorldVec(agent.x, agent.y, this.grid, this.dashFrom);
    this.dashTo.copy(this.dashFrom);
    this.wakeFrom.copy(this.dashFrom);
    this.wakeTo.copy(this.dashFrom);
  }

  /**
   * The world position of the cell the +1 was earned on: the PREVIOUS frame's goal cell.
   * `frame.prev` is the pure source; the tracked cell covers a host that hands us an
   * event without a previous frame, and the current goal is the last resort.
   */
  private bloomCell(frame: FrameUpdate, out: THREE.Vector3): void {
    const grid = this.grid;
    if (frame.prev) {
      const before = read(frame.prev, grid, this.before);
      cellToWorldVec(before.goal.x, before.goal.y, grid, out);
      return;
    }
    if (this.lastGoal.x >= 0) {
      cellToWorldVec(this.lastGoal.x, this.lastGoal.y, grid, out);
      return;
    }
    cellToWorldVec(this.now.goal.x, this.now.goal.y, grid, out);
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.world?.setPixelRatio(size.dpr);
    this.settleDust?.setPixelRatio(size.dpr);
    this.pool?.setPixelRatio(size.dpr);
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
  }

  probes(): AlignmentProbe[] {
    const grid = this.grid;
    const wisp = this.wisp;
    const oasis = this.oasis;
    const world = this.world;

    const wispReading: Reading = { agent: { x: 0, y: 0 }, goal: { x: 0, y: 0 }, lastDir: 1, action: 0 };
    const goalReading: Reading = { agent: { x: 0, y: 0 }, goal: { x: 0, y: 0 }, lastDir: 1, action: 0 };
    const wispOut: Vec3Tuple = [0, 0, 0];
    const goalOut: Vec3Tuple = [0, 0, 0];

    // THE WISP PROBE RULE. `assertUpdateIdempotent` compares probe objects' whole WORLD
    // MATRIX, not just their translation, so a probe may carry no eased value of any kind
    // — position OR rotation OR scale. This scene calls `Wisp.face()` every frame (the
    // whole point of `pose[4]`), and `face()` damps the body yaw toward its target over
    // `dt`; `wisp.probeObject` (the orb core) is a child of that body. So the probe lives
    // on `wisp.avatar`, which carries the cell and nothing else — exactly `cellToWorld`,
    // y = 0 — and the scarf hangs off it as a child, which cannot move it.
    const probes: AlignmentProbe[] = [
      {
        name: 'wisp',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          const r = read(state, grid, wispReading);
          return cellToWorld(r.agent.x, r.agent.y, grid, wispOut);
        },
      },
      {
        // the goal alcove — it must land on the RESPAWNED cell, every time
        name: 'oasis',
        object: oasis?.root ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          const r = read(state, grid, goalReading);
          return cellToWorld(r.goal.x, r.goal.y, grid, goalOut);
        },
      },
    ];

    // the arches sit on the env's OWN gap cells, never on guessed holes
    const gaps = world?.gaps ?? [];
    for (let i = 0; i < gaps.length; i++) {
      const gap = gaps[i];
      const root = world?.archRoots[i];
      if (!gap || !root) continue;
      const out: Vec3Tuple = [0, 0, 0];
      probes.push({
        name: `arch-${i}`,
        object: root,
        expectWorldPos: () => cellToWorld(gap.x, gap.y, grid, out),
      });
    }
    return probes;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.oasis?.dispose();
    this.oasis = null;
    this.settleDust?.dispose();
    this.settleDust = null;
    this.gust?.dispose();
    this.gust = null;
    this.scarf?.dispose();
    this.scarf = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
    this.dashAt = -Infinity;
    this.scarfYaw = 0;
    this.lastGoal.x = -1;
    this.lastGoal.y = -1;
  }
}

// ---------------------------------------------------------------------------
// pure state functions — every one of them testable without a GPU
// ---------------------------------------------------------------------------

/** The grid delta `lastDir` names: 0 = up (−y), 1 = down (+y), 2 = left (−x), 3 = right (+x). */
const FACING: readonly { dx: number; dy: number }[] = Object.freeze([
  { dx: 0, dy: -1 },
  { dx: 0, dy: 1 },
  { dx: -1, dy: 0 },
  { dx: 1, dy: 0 },
]);

/** `pose[4]` as a grid delta — the direction the agent last moved, and a dash's direction. */
export function facingDelta(lastDir: number): { dx: number; dy: number } {
  return FACING[((lastDir % 4) + 4) % 4] ?? { dx: 0, dy: 1 };
}

/** `pose[4]` as a yaw about +y, in the kit's grid convention (`atan2(dx, dy)`). */
export function facingYaw(lastDir: number): number {
  const d = facingDelta(lastDir);
  return yawFromGridDelta(d.dx, d.dy);
}

/**
 * Is this world-space segment the one- or two-cell collinear hop a `dash` actually makes?
 *
 * The ENV guarantees it — `gw_hdm_step` walks up to two cells along `lastDir` and stops at
 * the first wall — but this scene does not get to assume its inputs came from the env: the
 * harness's fixture loop plays canonical poses back to back, and a "dash" between two of
 * them can be any pair of cells at all. Speed lines strung diagonally across the canyon and
 * a ground wake laid through solid rock are a lie about the step, so both are gated on this.
 */
export function isDashSegment(fromX: number, fromZ: number, toX: number, toZ: number): boolean {
  const dx = Math.abs(toX - fromX) / CELL;
  const dz = Math.abs(toZ - fromZ) / CELL;
  const eps = 1e-6;
  if (dx > eps && dz > eps) return false; // never diagonal: a dash follows one axis
  const cells = dx > dz ? dx : dz;
  return cells >= 1 - eps && cells <= 2 + eps;
}

/** Shortest-arc damped turn — the exact law `Wisp.update` uses, so the scarf never lags it. */
export function dampAngle(current: number, target: number, dt: number): number {
  const k = 1 - Math.exp(-FACE_RATE * Math.max(0, dt));
  let d = target - current;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return current + d * k;
}

/**
 * Exponential decay in [0, 1] over `span` playback seconds, 0 outside the window. The
 * linear tail is not cosmetic: a bare exp() still has real amplitude when the window
 * closes, so every flourish would end on a one-frame pop.
 */
export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

/**
 * The new oasis welling up, with the little overshoot that makes water arriving read as
 * water arriving. A never-fired clock (`-Infinity`) and any age past the window both return
 * exactly 1, which is what makes a scrub land on a settled canyon instead of replaying a
 * growth it never saw.
 *
 * The bounce is `GROW_BOUNCE · t²(1 − t)`, which peaks at t = 2/3 — LATE, where the cubic
 * ease is already at 0.96, so the curve actually crosses 1 and settles back. The first pass
 * multiplied the ease by `1 + 0.14·sin(πt)(1 − t)`, whose overshoot peaks at 1.008: eight
 * thousandths of a scale, invisible on a 1-unit alcove even before `Oasis.setState` clamped
 * it away. A documented beat that renders as nothing is not a beat.
 */
export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  return 1 - Math.pow(1 - t, 3) + GROW_BOUNCE * t * t * (1 - t);
}

/**
 * How close the wisp is to the oasis, in [0, 1]. Chebyshev, because that is the metric the
 * maze's own four-neighbour walk makes meaningful at a glance. It shows nothing the agent
 * lacks: its own cell and the goal's cell are two of the four numbers the frame is made of.
 */
export function nearness(agent: GridCell, goal: GridCell): number {
  const d = Math.max(Math.abs(agent.x - goal.x), Math.abs(agent.y - goal.y));
  return d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
}

const createScene: SceneFactory = () => new SlotCanyonsScene();
export default createScene;
