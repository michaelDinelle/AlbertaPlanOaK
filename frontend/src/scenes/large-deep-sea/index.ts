// The Abyss — LargeDeepSea-cont-v0. The dive continued past the reef.
//
// THE ONE IDEA: `state.r` IS THE LIGHT. LargeDeepSea's dynamics allow exactly one thing —
// r increases by 1 every single step and c moves ±1 — so the reachable set is the triangle
// c ≤ r, that triangle is the terraced shelf, and every shelf cell's albedo is a pure
// function of its row (world.ts `shelfTint`). Row 0 is the last of the reef's blue; row 15
// is the colour of nothing. Look at one frame and the colour under the wisp tells you how
// deep this dive got, which is the only number this environment is about.
//
// WHAT THE RECORDINGS ACTUALLY CONTAIN, and why the scene is built the way it is:
// across all eight recorded lanes × 100k steps, the treasure at (15, 15) is taken TWICE
// (seed 78, steps 8879 and 95444) and seed 72 — the lane the harness loads — never takes
// it at all. The goal celebration is therefore real, exact and essentially never seen.
// The event that DOES fire, 6,667 times per lane, is `resurface`: every fifteenth step the
// dive hits row 14, wraps to (0, 0), and the trench answers with a column of bubbles
// rushing back up to the surface. That is this scene's recurring set piece, and it is
// built with the same care a +1 would get somewhere else.
//
// PATTERNS COPIED FROM THE GRID CANON (four-rooms), and the two that are new here:
//  * the avatar is `kit/wisp`, both anchors probed; `face()` is never called, so
//    `wisp.probeObject` stays bit-exact (canon lesson 16).
//  * MOTION is `HopMotion`, one straight arc per state change, hop 0.11 s — shorter than
//    one 8 Hz control step, so the glide always lands before the next truth (lesson 17).
//  * NEW — THE WRAP GUARD. A legal LargeDeepSea transition is exactly (r+1, c±1). A frame
//    that is not that is a resurface, and a resurface must be a HARD CUT: hopping 14 rows
//    back to the surface would be a path plan, and a very slow one. The test asserts the
//    wisp never travels more than one cell in a step. This is a pure function of the two
//    states, so it holds with or without the event.
//  * NEW — the celebration cannot be exercised by the data, so the scene is built to be
//    beautiful without it and the +1 path is proven in the test suite from constructed
//    `FrameUpdate` events instead. See `large-deep-sea.test.ts`.
//  * every flourish is measured on `ctx.time()` — the heart's beat AND its treasure blaze,
//    the lure flicker, the jellies' stroke, the bubble rush — so a paused playback freezes
//    the trench coherently and a scrub wipes all of it at once. No actor in this scene owns
//    a decaying clock: `wreck.setFlare()` is a pure write of the `treasure` envelope, so the
//    heart's mesh and the heart's point light are the same number by construction. The two
//    values that DO ease in wall seconds are the ones that must stay alive while the
//    playback is paused — the orbit camera (you can still look around) and the silt stir,
//    which converges to the frame's own truth and then stops.
//  * THE RIGHT-MOVE COST IS DRAWN. Half of this env's reward function is `-0.01/N` charged
//    whenever `action & 1` (`gw_ldeep_step`, `LARGE_DEEPSEA_RIGHT_COST`), and that fires on
//    about half of all steps. The wisp kicks a silt puff off the terrace exactly when the
//    env charges it — keyed to `ctrl[0] & 1`, the COST, never the direction the mask then
//    pushes. This is the same beat the sibling reef renders, for the same reason.

import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { cellToWorld, cellToWorldVec, clampCell, type GridSize, type Vec3Tuple } from '@kit/grid';
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
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { HEART_Y, buildAngler, buildDiveBubble, buildWreck, type AnglerHandle, type DiveBubbleHandle, type WreckHandle } from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  anglerSites,
  blendTokens,
  buildWorld,
  clamp01,
  resolveGrid,
  resolveMask,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. The camera stands in the NORTH-NORTH-WEST and looks south-south-east, which
// means it looks DOWN the dive: the shallow rows are near and still blue, the trench
// recedes away into the dark, and the wreck at (15, 15) is a far glimmer near the bottom
// of the frame. Looking across the trench instead (the four-rooms azimuth, +0.55) framed
// the same geometry as a flat diagonal and the descent stopped existing.
//
// THE TARGET is not the middle of the grid. Half of a LargeDeepSea grid is open water
// above the diagonal and nothing lives there, so the camera aims at the CENTROID OF THE
// REACHABLE TRIANGLE — cell (N/3, 2N/3) ≈ (5, 10) — and lets the empty half take the
// crop, then shifts ~1.3 units along the screen-x axis to centre the shelf inside the
// frame. Aiming at the world origin put the shelf in a corner with a third of the picture
// spent on black water.
//
// THE DISTANCE is set by the NEAR corner, not the far one. This aerial puts cell (0, 0)
// 21 units from the eye and the wreck 35 — a 1.7:1 perspective spread — so at 21 units the
// apex subtends 27° of a 20° half-FOV while the wreck subtends 13, and every framing that
// centred on the shelf's middle threw the START CELL out of the bottom of the frame. The
// start cell is where the wisp returns every fifteen steps: it is not optional. At 30
// units, with the target dropped below the floor plane to tip the view down, the span
// lands at −16° … +15°, which is the whole dive with room for the transport bar.
const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.74;
const CAM_DISTANCE = 30;
const CAM_TARGET: readonly [number, number, number] = [-3.6, -3.1, 3.3];
const CAM_YAW_RANGE = 0.55;
/** thumb pulls back and stops moving: a square-ish crop must still hold the whole dive */
const CAM_DISTANCE_THUMB = 34;
const CAM_POLAR_THUMB = 0.72;

// Celebration weights. The trench is 16 units of near-black, so the calculus is the
// opposite of the daylight scenes: the frame can TAKE a bright burst (nothing else is
// near the bloom threshold), but the burst must not out-shine the wreck heart it is
// supposed to be celebrating. Two thirds of the kit's default travel, all three beats.
const BURST_SCALE = 0.8;
const RIPPLE_SCALE = 0.55;
const BURST_PARTICLES = 130;
const PULSE_STRENGTH = 0.62;
const TREASURE_SECONDS = 2.2;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;

/** how far over the diving bell its lamp hangs — see the note at the point light */
const WISP_LAMP_LIFT = 0.75;

/** the resurface rush: bubbles sweep from the cell the dive ended on back to the surface */
const RUSH_SECONDS = 1.15;
const RUSH_Y = 1.25;

/**
 * How high over the terrace the stirred silt hangs, as a fraction of the wisp's height.
 * LOW, and the number was measured, not chosen: the diving bell is a 0.36-radius sphere
 * centred at `wispHeight`, so a puff at 0.45·wispHeight sat entirely inside the bell's own
 * glow and was invisible at 28 units. At 0.28 it hugs the slab and spreads out from under
 * the bell, which is also what sediment does.
 */
const SILT_LIFT = 0.28;
/** how fast the water agrees with the frame (per wall second) — see the note in update() */
const SILT_RATE = 9;
/** the puff always breathes a little; the right-move is what makes it a puff */
const SILT_IDLE = 0.04;
const SILT_KICK = 0.85;

interface LargeDeepSeaLike {
  r?: unknown;
  c?: unknown;
  lastA?: unknown;
}

/**
 * The agent and treasure cells for a frame. sim-core's decoded state is the source; the
 * raw pose (`[r, c, gx, gy, …]`) is the fallback so the scene still stands up under the
 * harness's fixture loop, which carries no `derived`.
 */
function readCells(
  state: FrameState,
  grid: GridGeometry,
  agent: GridCell,
  goal: GridCell,
): void {
  const d = state.derived as LargeDeepSeaLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const r = has && typeof d.r === 'number' ? d.r : (state.pose[0] ?? 0);
  const c = has && typeof d.c === 'number' ? d.c : (state.pose[1] ?? 0);
  const gx = state.pose[2];
  const gy = state.pose[3];
  // grid x is the COLUMN and grid y is the ROW: the dive runs +z (south), exactly the
  // axis language DeepSea uses, so the pair reads as one descent.
  agent.x = Number.isFinite(c) ? Math.round(c) : 0;
  agent.y = Number.isFinite(r) ? Math.round(r) : 0;
  goal.x = typeof gx === 'number' && Number.isFinite(gx) ? Math.round(gx) : (grid.goal?.x ?? grid.w - 1);
  goal.y = typeof gy === 'number' && Number.isFinite(gy) ? Math.round(gy) : (grid.goal?.y ?? grid.h - 1);
  clampCell(agent, grid);
  clampCell(goal, grid);
}

/**
 * THE HALF OF THE REWARD FUNCTION THAT IS NOT THE TREASURE. `gw_ldeep_step` charges
 * `LARGE_DEEPSEA_RIGHT_COST` (−0.01/N = −0.000625) whenever `action & 1`, whatever the
 * hidden mask then does with it — so the cost is keyed to the ACTION BIT, not to the
 * direction the wisp ends up moving. `ctrl[0]` is that action as the recording stores it;
 * the decoder's `derived.lastA` is the same number already masked, and is preferred when
 * the host supplies it. Returns 1 on a charged step, 0 otherwise.
 */
export function readRightMove(state: FrameState): number {
  const d = state.derived as LargeDeepSeaLike | null | undefined;
  if (d !== null && typeof d === 'object' && typeof d.lastA === 'number' && Number.isFinite(d.lastA)) {
    return Math.trunc(d.lastA) & 1;
  }
  const a = state.ctrl[0];
  return typeof a === 'number' && Number.isFinite(a) ? Math.trunc(a) & 1 : 0;
}

/**
 * The env's own step relation: r ALWAYS advances by exactly one and c moves by exactly one
 * (clamped at the walls, which is why |Δc| = 0 is legal at c = 0 and c = N−1). Anything
 * else is the wrap back to the surface, and the wrap is a hard cut, never a glide.
 */
export function isDiveStep(fromX: number, fromY: number, toX: number, toY: number): boolean {
  return toY === fromY + 1 && Math.abs(toX - fromX) <= 1;
}

class TheAbyssScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private bubble: DiveBubbleHandle | null = null;
  private wreck: WreckHandle | null = null;
  private anglers: AnglerHandle[] = [];
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private treasureAt = -Infinity;
  private rushAt = -Infinity;
  /**
   * Eased stir of the silt: the frame says 0 or 1, the water takes a moment to agree.
   * NOT a flourish clock — it is a damped follower of the CURRENT frame's own truth (the
   * sibling reef's `stir`), so it converges and stops instead of decaying to nothing, it
   * can never disagree with the frame it is showing, and a scrub snaps it to the target.
   */
  private stir = 0;

  // ---- preallocated scratch: update() may not allocate ----
  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly heartPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly bubblePos = new THREE.Vector3();
  private readonly siltPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly lureWorld = new THREE.Vector3();
  private readonly rushFrom = new THREE.Vector3();
  private readonly rushTo = new THREE.Vector3();
  private accent = 0x7fd4e8;
  private gold = 0xffd166;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'theAbyssScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    const mask = resolveMask(grid, ctx.seed);

    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.uiCyan;
    this.accent = accent;
    this.gold = ctx.tokens.celebrationGold;
    const water = ctx.tokens.palettes.underwater;

    const world = buildWorld(ctx, this.scene, grid, mask);
    this.world = world;
    this.scene.add(world.group);
    const skin = world.skin;

    // This scene claims all four of the rig's point lights, and their COLOURS are static —
    // only position and intensity are state-driven, so update() never touches a Color.
    // The order is the priority order: at medium tier only two lights exist, and the two
    // that must survive are the one the agent carries and the one it is looking for.
    const wispLight = world.rig.points[0];
    if (wispLight) wispLight.color.set(accent);
    const heartLight = world.rig.points[1];
    if (heartLight) heartLight.color.set(skin.heart);

    // ---- the shared avatar, riding inside a bubble ----
    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        // bigger than the courtyard's 0.21: from a 24.5-unit standoff in near-total
        // darkness the orb is the eye's only anchor, and the bubble around it needs
        // something with mass inside to refract
        radius: 0.19,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'abyssWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const bubble = buildDiveBubble({
      tier: ctx.tier,
      accent,
      water: blendTokens(water.groundA, ctx.tokens.palettes.abyss.skyTop, 0.5),
      name: 'diveBubble',
    });
    this.bubble = bubble;
    this.scene.add(bubble.group);

    // ---- the treasure: a wreck on whatever cell the env calls the goal ----
    const wreck = buildWreck({ tier: ctx.tier, skin, accent, name: 'sunkenWreck' });
    this.wreck = wreck;
    this.scene.add(wreck.group);

    // ---- the anglerfish, on real shelf cells ----
    const sites = anglerSites(grid, ctx.seed);
    const lureColors = [accent, blendTokens(ctx.tokens.cueMagenta, accent, 0.42).getHex()];
    for (let i = 0; i < sites.length; i++) {
      const site = sites[i];
      if (!site) continue;
      const angler = buildAngler({
        tier: ctx.tier,
        skin,
        lure: lureColors[i] ?? accent,
        hover: site.hover,
        phase: site.phase,
        scale: site.scale,
        name: `angler${i}`,
      });
      cellToWorldVec(site.x, site.y, grid, angler.root.position);
      this.scene.add(angler.group);
      this.anglers.push(angler);
      const light = world.rig.points[2 + i];
      if (light) light.color.set(lureColors[i] ?? accent);
    }

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [CAM_TARGET[0], CAM_TARGET[1], CAM_TARGET[2]],
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: CAM_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      limits: {
        minPolar: 0.42,
        maxPolar: 1.02,
        minAzimuth: CAM_AZIMUTH - CAM_YAW_RANGE,
        maxAzimuth: CAM_AZIMUTH + CAM_YAW_RANGE,
        // close enough to swim down to a single shelf, never close enough to lose the dive
        minDistance: 13,
        maxDistance: 34,
      },
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0012,
      name: 'abyssOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'abyssCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
    // It draws the mask arrows for the whole 16×16 lattice, which is the check the coral
    // has to survive: every whip must lean the way its arrow points.
    const overlayAgent: GridCell = { x: 0, y: 0 };
    const overlayGoal: GridCell = { x: 0, y: 0 };
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayAgent;
      },
      goalCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayGoal;
      },
      name: 'abyssGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    cellToWorldVec(grid.goal?.x ?? grid.w - 1, grid.goal?.y ?? grid.h - 1, grid, wreck.root.position);
    cellToWorldVec(0, 0, grid, this.rushTo);
    this.rushTo.y = RUSH_Y;
    this.rushFrom.copy(this.rushTo);
    wreck.setFlare(0);
    wreck.setTime(0);
    bubble.setGain(0);
    bubble.setTime(0);
    // the silt starts settled on the start cell: no cost has been charged yet
    cellToWorldVec(0, 0, grid, world.silt.points.position);
    world.silt.points.position.y += siltLift(ctx.tokens.wispHeight);
    world.silt.setDensity(0);
    for (const angler of this.anglers) angler.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const bubble = this.bubble;
    const wreck = this.wreck;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !bubble || !wreck || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const agent = this.agentCell;
    const goal = this.goalCell;

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. the frame's truth, snapped to cells
    readCells(state, grid, agent, goal);

    // 2b. THE DEPTH, the one number this whole scene is a function of. It drives the
    //     bubble's gain, the avatar's brightness and its point light, and it is a pure
    //     read of the row — no easing, no history, so a scrubbed frame and a played frame
    //     at the same row are the same picture.
    const depth = bubbleGain(agent.y, grid);
    // The avatar itself dims toward the surface. Not decoration — a NECESSITY: the top
    // rows carry an ~11× surface-light gain (world.ts `attenuate`), so a full-strength wisp
    // standing on row 0 put its own pool over the bloom threshold and blew a 300 px white
    // hole through the bottom of the frame every fifteen steps. Dim up top where there is
    // already light, blazing on the floor where there is none — which is also exactly what
    // the brief asks the diving bell to do. Set BEFORE the scrub branch so `wisp.snap()`
    // lands on the right brightness instead of easing into it from the last frame's.
    wisp.dim(0.2 + 0.8 * depth);

    // 2c. THE COST — the other half of this env's reward function: −0.01/N every time the
    //     agent picks action 1, whichever way the hidden mask then sends it. A pure read of
    //     the frame, no history, so the only thing that eases is how fast the water agrees.
    const charged = readRightMove(state);

    // 3. the treasure sits where the env says the treasure is
    cellToWorldVec(goal.x, goal.y, grid, wreck.root.position);
    this.groundPos.copy(wreck.root.position);
    this.heartPos.copy(wreck.root.position);
    this.heartPos.y = HEART_Y;

    // 4. a scrub wipes every in-flight beat: nothing may survive a discontinuity. That
    //    includes the wreck's blaze, which is now a pure function of `treasureAt` (the
    //    handle carries no decay of its own), and the silt stir, which is snapped to the
    //    frame's own truth rather than eased across the discontinuity.
    if (frame.scrubbed) {
      pool.clear();
      this.treasureAt = -Infinity;
      this.rushAt = -Infinity;
      this.stir = charged;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      rig.snap();
    }

    // 5. events — celebrations are NEVER inferred from the state stream
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        // The +1 is collected ON the wreck: the env scores it for arriving at column 15
        // out of row 14, and then wraps — so unlike FourRooms, the celebration belongs to
        // the TREASURE's cell, not to the cell the agent is standing in when it lands.
        pool.burst(this.heartPos, this.accent, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        this.treasureAt = time;
      } else if (ev.type === 'resurface' || ev.type === 'teleport') {
        // The bubble rush starts where the dive ENDED. `frame.prev` is the honest source;
        // the wisp's own (pre-hop) cell is the fallback for a first frame with no prev.
        const fromX = frame.prev ? Math.round(frame.prev.pose[1] ?? wisp.cellX) : wisp.cellX;
        const fromY = frame.prev ? Math.round(frame.prev.pose[0] ?? wisp.cellY) : wisp.cellY;
        cellToWorldVec(
          fromX < 0 ? 0 : fromX > grid.w - 1 ? grid.w - 1 : fromX,
          fromY < 0 ? 0 : fromY > grid.h - 1 ? grid.h - 1 : fromY,
          grid,
          this.rushFrom,
        );
        this.rushFrom.y = RUSH_Y;
        cellToWorldVec(agent.x, agent.y, grid, this.rushTo);
        this.rushTo.y = RUSH_Y;
        this.rushAt = time;
      }
      // LargeDeepSea emits no `done` and no failure: this scene must never sigh.
    }

    // 6. the avatar. A dive step is a hop; anything else is the wrap, and the wrap is a
    //    hard cut — never a fourteen-cell glide back to the surface.
    if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      if (isDiveStep(wisp.cellX, wisp.cellY, agent.x, agent.y)) {
        wisp.hopTo(agent, time);
      } else {
        wisp.setCell(agent.x, agent.y, grid);
        wisp.snap();
      }
    }
    wisp.update(frame.dtReal, time);

    // 7. the dive bubble: position is the wisp's, brightness is the ROW. Both are pure
    //    functions of the frame — the bubble on a scrubbed frame is the bubble on a
    //    played one.
    this.bubblePos.copy(wisp.position);
    this.bubblePos.y += TOKENS.wispHeight;
    bubble.group.position.copy(this.bubblePos);
    bubble.setGain(depth);
    bubble.setTime(time);

    // 7b. THE SILT — the right-move cost, drawn. Position is the wisp's cell exactly (it
    //     is a state-bearing object: where it is answers "where is the dive", and the
    //     probe suite locks it there); density is the eased cost. The ease runs in WALL
    //     seconds like the camera's damping, and unlike a flourish it CONVERGES: a paused
    //     frame settles onto that frame's own truth instead of draining away from it.
    this.stir += (charged - this.stir) * (1 - Math.exp(-SILT_RATE * Math.max(0, frame.dtReal)));
    this.siltPos.copy(wisp.position);
    this.siltPos.y += siltLift(ctx.tokens.wispHeight);
    world.silt.points.position.copy(this.siltPos);
    world.silt.setDensity(SILT_IDLE + SILT_KICK * this.stir);
    // the puff is lit by the wisp's own lamp, so down in the dark it has to carry more of
    // its own light or the cost stops being visible exactly where the trench is blackest
    world.silt.setIntensity(0.45 + 0.85 * depth);

    // 8. the flourish envelopes, all measured on the playback clock
    const treasure = envelope(time - this.treasureAt, TREASURE_SECONDS, 2.1);
    const rushAge = time - this.rushAt;
    const rushing = rushAge >= 0 && rushAge <= RUSH_SECONDS;
    const rushT = rushing ? rushAge / RUSH_SECONDS : 0;
    // ease-out travel and a sine envelope: the column blooms as it leaves and thins as it
    // reaches the surface, so the beat ends on nothing instead of on a pop
    const rushE = 1 - Math.pow(1 - rushT, 2.2);
    world.rush.points.position.lerpVectors(this.rushFrom, this.rushTo, rushing ? rushE : 0);
    world.rush.setDensity(rushing ? Math.sin(Math.PI * rushT) : 0);

    // 9. the wreck and the fish. `setFlare` is a pure write of the SAME envelope the heart's
    //    point light below is driven by, so the mesh and the light can never disagree —
    //    which is what a paused frame and a scrubbed frame both used to prove they did.
    wreck.setFlare(treasure);
    wreck.setTime(time);
    for (let i = 0; i < this.anglers.length; i++) this.anglers[i]?.setTime(time);

    // 10. the four point lights this scene claims from the rig
    const wispLight = world.rig.points[0];
    if (wispLight) {
      // The lamp rides ABOVE the bell, not inside it, and the reason is arithmetic. A
      // decay-2 point light 0.35 units off the floor has a distance attenuation of 8.2×,
      // and this scene's shallow terraces run at ~10 albedo (the Beer–Lambert surface
      // gain), so the product was 7.5 linear — nine times the bloom threshold — and the
      // wisp burned a hole through the shelf every time it surfaced. At 1.1 units the
      // attenuation is 0.83 and the pool is a pool.
      this.lightPos.copy(this.bubblePos);
      this.lightPos.y += WISP_LAMP_LIFT;
      wispLight.position.copy(this.lightPos);
      // …and it burns as the SQUARE of the depth: near zero where the shelf still has its
      // own light to lose, full where there is nothing left to add to.
      wispLight.intensity =
        0.06 + 3.4 * depth * depth + 5.5 * treasure + 1.2 * (rushing ? Math.sin(Math.PI * rushT) : 0);
    }
    const heartLight = world.rig.points[1];
    if (heartLight) {
      heartLight.position.copy(this.heartPos);
      heartLight.intensity = 1.05 + 0.35 * Math.sin(time * 1.35) + 8 * treasure;
    }
    for (let i = 0; i < this.anglers.length; i++) {
      const light = world.rig.points[2 + i];
      const angler = this.anglers[i];
      if (!light || !angler) continue;
      angler.lureAnchor.getWorldPosition(this.lureWorld);
      light.position.copy(this.lureWorld);
      light.intensity = 1.6 + 0.55 * Math.sin(time * 1.9 + i * 3.1);
    }

    // 11. one clock for the whole trench
    world.setTime(time);

    // 12. camera easing runs in WALL time so a paused frame is still explorable
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
    const grid = this.grid;
    const wisp = this.wisp;
    const wreck = this.wreck;
    const height = this.ctx?.tokens.wispHeight ?? TOKENS.wispHeight;
    const seed = this.ctx?.seed ?? 0;

    const wispCell: GridCell = { x: 0, y: 0 };
    const wispGoal: GridCell = { x: 0, y: 0 };
    const wispOut: Vec3Tuple = [0, 0, 0];
    const anchorCell: GridCell = { x: 0, y: 0 };
    const anchorGoal: GridCell = { x: 0, y: 0 };
    const anchorOut: Vec3Tuple = [0, 0, 0];
    const wreckCell: GridCell = { x: 0, y: 0 };
    const wreckGoal: GridCell = { x: 0, y: 0 };
    const wreckOut: Vec3Tuple = [0, 0, 0];
    const heartCell: GridCell = { x: 0, y: 0 };
    const heartGoal: GridCell = { x: 0, y: 0 };
    const heartOut: Vec3Tuple = [0, 0, 0];
    const anglerOut: Vec3Tuple = [0, 0, 0];

    // THE WISP PROBE RULE (canon lesson 16): this scene never calls `wisp.face()` — the
    // dive is symmetric and a diving bell has no front — so BOTH anchors stay pure and
    // both are probed. If a future revision turns the wisp, `wisp` must move to
    // `wisp.avatar` or the yaw damping will show up as probe drift.
    const probes: AlignmentProbe[] = [
      {
        // the orb centre: exactly `wispHeight` over the true cell (column, row)
        name: 'wisp',
        object: wisp?.probeObject ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readCells(state, grid, wispCell, wispGoal);
          cellToWorld(wispCell.x, wispCell.y, grid, wispOut);
          wispOut[1] = height;
          return wispOut;
        },
      },
      {
        // and the avatar node under it: the cell itself, y = 0, position only
        name: 'wisp-anchor',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readCells(state, grid, anchorCell, anchorGoal);
          return cellToWorld(anchorCell.x, anchorCell.y, grid, anchorOut);
        },
      },
      {
        // the wreck root: the treasure cell the env declares, y = 0
        name: 'wreck',
        object: wreck?.root ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readCells(state, grid, wreckCell, wreckGoal);
          return cellToWorld(wreckGoal.x, wreckGoal.y, grid, wreckOut);
        },
      },
      {
        // the heart: a fixed height over the same cell, with the beat on its child
        name: 'wreck-heart',
        object: wreck?.heartAnchor ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readCells(state, grid, heartCell, heartGoal);
          cellToWorld(heartGoal.x, heartGoal.y, grid, heartOut);
          heartOut[1] = HEART_Y;
          return heartOut;
        },
      },
    ];

    const world = this.world;
    if (world) {
      const siltCell: GridCell = { x: 0, y: 0 };
      const siltGoal: GridCell = { x: 0, y: 0 };
      const siltOut: Vec3Tuple = [0, 0, 0];
      probes.push({
        // the stirred silt rides the wisp's TRUE cell — a state-bearing object, not
        // decoration: it is the env's per-right-move cost, drawn where the cost was paid
        name: 'silt',
        object: world.silt.points,
        expectWorldPos: (state) => {
          readCells(state, grid, siltCell, siltGoal);
          cellToWorld(siltCell.x, siltCell.y, grid, siltOut);
          siltOut[1] += siltLift(height);
          return siltOut;
        },
      });
    }

    const site = anglerSites(grid, seed)[0];
    const angler = this.anglers[0];
    if (site && angler) {
      probes.push({
        // an anglerfish patrols a REAL shelf cell — the swim is on its child
        name: 'angler-site',
        object: angler.root,
        expectWorldPos: () => cellToWorld(site.x, site.y, grid, anglerOut),
      });
    }
    return probes;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    for (const angler of this.anglers) angler.dispose();
    this.anglers.length = 0;
    this.wreck?.dispose();
    this.wreck = null;
    this.bubble?.dispose();
    this.bubble = null;
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
    this.treasureAt = -Infinity;
    this.rushAt = -Infinity;
    this.stir = 0;
  }
}

// ---------------------------------------------------------------------------
// envelopes — every one of them measured on the playback clock
// ---------------------------------------------------------------------------

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

/** How hard the dive bubble burns at a row: 0 at the surface, 1 on the last walkable row. */
export function bubbleGain(row: number, grid: GridSize): number {
  return clamp01(row / Math.max(1, grid.h - 2));
}

/** Height of the silt puff over the terrace it was kicked off — the probe's own constant. */
export function siltLift(wispHeight: number): number {
  return wispHeight * SILT_LIFT;
}

const createScene: SceneFactory = () => new TheAbyssScene();
export default createScene;
