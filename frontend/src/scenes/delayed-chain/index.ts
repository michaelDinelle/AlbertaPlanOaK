// The Long Night Road — DelayedChain-cont-v0.
//
// THE ONE IDEA: DELAYED CREDIT, STAGED LITERALLY. This env pays +1 for arriving at pos
// 31 and nothing anywhere else, so the whole walk is credit that has not arrived yet.
// The road is therefore built of thirty-two UNLIT lanterns, and the reward's only
// physical act is to light them — starting at the end that paid and running BACKWARD,
// cell by cell, to the start the wisp is already walking out of again. You watch the
// credit propagate down the road. That is the algorithm, drawn.
//
// WHAT THIS SCENE HAD TO GET RIGHT, AND WHY IT DIFFERS FROM THE GRID CANON
//  * `dchain_step` wraps to pos 0 in the SAME step that pays, so the frame carrying the
//    `goal` event already reads pos 0. FourRooms' rule ("the goal blooms where the AGENT
//    is") would bloom the start of the road — the exact opposite of the truth. Here the
//    celebration is nailed to the GOAL CELL from `ctx.geometry`, and the test locks it.
//  * The same wrap is a 30-cell jump, so the `teleport` event that rides with it is a
//    HARD CUT: the wisp is snapped, the follow camera is snapped, and only then does the
//    ignition wave roll home. A hop across the whole moor would be a lie about the env.
//  * The wave is driven PURELY by time-since-event with a hard cutoff (`lanternGlow`) —
//    no integrator, no per-lantern state machine. A scrub sets the clock to −Infinity and
//    every lantern is cold on the very next frame, with no easing to leak across.
//  * `done` never arrives without `goal` in this env (they are the same step), so this
//    scene never sighs. Sighing would claim a failure the stream does not contain.
//  * No superluminal guard: DelayedChain is `discretePose`, so the host SNAPS poses
//    rather than interpolating them, and the respawn can never render as a slide.
//  * The avatar is the kit `Wisp` with its HALO MESH SWITCHED OFF and a radial-sprite
//    billboard hung in its place (`buildWispGlow`). That is a NIGHT-SCENE requirement, not
//    a preference: this camera stands 8.4 units from the orb, close enough that the kit
//    halo's detail-1 icosphere shows its facets as a flat green decagon. Everything else
//    about the avatar — core, skirt, ribbon — stays at kit brightness, so the wisp really
//    is the brightest thing on the road; only its edge is ours. (See the `oakWispHalo`
//    block in init(): `Wisp.dim()` CANNOT do this job, and the first pass's `dim(0.1)`
//    was a 10× dim of the whole avatar.)

import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { cellToWorld, cellToWorldVec, type GridSize, type Vec3Tuple } from '@kit/grid';
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
import {
  LAMP_Y,
  buildLanternRoad,
  buildMilestone,
  buildWispGlow,
  type LanternRoadHandle,
  type MilestoneHandle,
  type WispGlowHandle,
} from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  SHADOW_RADIUS,
  buildWorld,
  clamp01,
  goalCellOf,
  resolveGrid,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. A road picture, not a plan view: the camera walks with the wisp at 2.5 m
// above the flagstones and pitches down only 8°, so the horizon sits ~30% from the top
// and the aurora gets a real piece of sky. That angle is the whole composition — at the
// 27° aerial a grid scene normally wants, the frame is 100% ground, the vanishing point
// is off the top edge, and the aurora (the reason this env has a far end at all) is not
// in the picture. It also puts the lantern lamps at 1.5 m, just under camera height, so
// the ignition wave runs across the frame's middle instead of under its chin.
// The 8.4-unit standoff is the second pass. At 7.1 the wisp's halo (radius · 2.4, a
// detail-1 icosphere in the kit) subtended nearly 8° and read as a faceted green disc
// pasted over the road; a quarter more distance and a slightly smaller orb put it back to
// a lamp-sized spirit without touching how BRIGHT it is — which is what "the wisp is the
// brightest thing on the road" actually asks for.
const CAM_DISTANCE = 8.4;
const CAM_POLAR = 1.43;
const CAM_TARGET_Y = 1.3;
/** the camera looks a cell ahead of the wisp: the road, not the walker, is the subject */
const CAM_LOOK_AHEAD = 0.9;
/** follow damping, per second, in WALL time — snapped on every discontinuity */
const CAM_FOLLOW_RATE = 5.5;
// THUMB IS THE WALL VIEW, AND THE WALL VIEW OF THIS ENV IS THE WHOLE ROAD. The first pass
// framed the last third — cairn, aurora, end lanterns — which is handsome and wrong: the
// agent spends nearly all of its 100k steps somewhere in the other twenty-five cells, so
// that crop shows an empty road most of the time. Backed off far enough to hold all 32
// cells, the thumbnail answers the only question a wall of streams asks — HOW FAR ALONG IS
// THIS ONE — with the wisp's own spark of light against a line of dark lanterns.
// The frame is the HERO ANGLE PULLED BACK, and that is arithmetic rather than taste. A
// side-on framing holds all 32 cells (four-rooms lesson 21) but throws the aurora — which
// stands 30+ units EAST of the road — clean out of the picture, and an aurora-less thumb
// of this scene is any dark road. So the camera keeps the road's own axis, and the road
// recedes from the near corner to the curtain: identity and progress in one crop.
//
// THE FRAME IS SIZED AT 1:1, NOT AT 16:9, and that is canon lesson 11 taken seriously.
// vFOV is fixed at 40° whatever the canvas is, so a square wall cell has a 40° HORIZONTAL
// field where a 16:9 one has 66°: the second pass (d 31, target [2.5, 1.05, 0]) put cell 0
// — the cell the stream respawns on after every single pay-off, and where a random walker
// spends most of its life — at NDC x −0.74 on a 16:9 canvas and −1.31 on a square one, i.e.
// OFF-FRAME on the wall the thumbnail exists for. Pulling the target back down the road to
// x = −4 and dropping the camera to 7.3 units (polar 1.38) holds BOTH ends of the corridor
// and both lantern tops inside |NDC| ≤ 0.78 at 1:1, 4:3 and 16:9 alike, with the curtain's
// axis still in the crop — asserted over three aspects in the test, at ≤ 0.9. The standoff
// is then the LARGEST that still fits the square: 28 spends the leftover margin on the
// road itself (its span across a square frame goes 1.19 → 1.29 of half-width), because a
// wall cell that holds the whole road but draws it as a hairline has answered nothing.
// 1:1 is the narrowest crop this frame is sized for — the road runs off a PORTRAIT card
// (3:4 puts cell 0 at −1.03), which is a deliberate bound, not an oversight: a portrait
// wall cell would cost another ~25% of standoff and shrink the road for every landscape
// and square one. If the gallery ever ships portrait cells, re-solve here first.
const CAM_DISTANCE_THUMB = 28;
const CAM_AZIMUTH_THUMB = -1.12;
const CAM_POLAR_THUMB = 1.38;
const CAM_TARGET_THUMB: readonly [number, number, number] = [-4, 2, 0];

// ---- the ignition wave (playback seconds) ----
/** how long the wave takes to run the whole road, end to start */
export const WAVE_TRAVEL = 1.35;
/** a lantern kindles in this long once the wave reaches it */
export const IGNITE_ATTACK = 0.14;
/** …holds full for this long… */
export const IGNITE_HOLD = 0.85;
/** …and gutters out over this long. The hard cutoff is the sum. */
export const IGNITE_GUTTER = 2.9;
/** one lantern's whole life after the wave arrives */
export const LANTERN_SPAN = IGNITE_ATTACK + IGNITE_HOLD + IGNITE_GUTTER;
/** the wave is completely over this long after the goal — nothing may glow past it */
export const WAVE_SPAN = WAVE_TRAVEL + LANTERN_SPAN;

// ---- celebration weights ----
// The pay-off happens at the far END of a 32-unit road and is usually watched from 20–30
// units away, so this scene needs MORE amplitude than the two exemplars, not less (canon
// lesson 6 cuts both ways: scale with the scene). The ripple is 2.9 units across — three
// cells of flagstone — and still reads as a ring rather than a flash.
const BURST_SCALE = 1.25;
const RIPPLE_SCALE = 0.9;
const BURST_PARTICLES = 150;
/**
 * BEAT 3 — the bloom pulse the host feeds to the composer. It is deliberately NOT scaled
 * up with the burst: the particles are 30 units away and want amplitude, while the pulse
 * is a full-screen post effect on the joint-darkest palette in the gallery (bloomStrength
 * 0.62), where 1.0 clips the wisp's core to a white hole. Reaching this number takes one
 * extra line at the fire site — see the `pulse.reset()` in update().
 */
export const PULSE_STRENGTH = 0.6;
/** the aurora answers the pay-off it has been advertising all along */
const SURGE_SECONDS = 2.6;
/**
 * The avatar's own answer to the +1. RETUNED, because it used to be unreachable: the flare
 * was fired inside the event loop and then wiped by the wrap's `Wisp.snap()` on the same
 * frame, so 0.85 had never actually been seen. Fired for real on an UNDIMMED orb it drove
 * the core rim to 4.8 and turned the whole foreground into a white plate — on the one frame
 * where the eye belongs at the FAR END, watching the credit arrive. 0.32 reads as the wisp
 * answering without competing with the thing it earned. Measured on the pay-off frame at
 * 1600×900 (with the glow multiplier below): pixels ≥ 250/255 fell 1292 → 94, and the idle
 * frames never clip at all (max 242).
 */
const WISP_FLARE = 0.32;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;
/** how far the wisp leans into its direction of travel (pose[2] = lastA) */
const WISP_LEAN = 0.15;

interface DelayedChainLike {
  pos?: unknown;
  lastA?: unknown;
}

/**
 * The agent's cell and its last action for a frame. sim-core's decoded state is the
 * source; the raw pose (`[pos, 31, lastA, …]`) is the fallback so the scene still stands
 * up under the harness's fixture loop, which carries no `derived`.
 */
function readState(state: FrameState, grid: GridSize, out: GridCell): number {
  const d = state.derived as DelayedChainLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const raw = has && typeof d.pos === 'number' ? d.pos : (state.pose[0] ?? 0);
  const lastA = has && typeof d.lastA === 'number' ? d.lastA : (state.pose[2] ?? 0);
  const pos = Number.isFinite(raw) ? Math.round(raw) : 0;
  out.x = pos < 0 ? 0 : pos > grid.w - 1 ? grid.w - 1 : pos;
  out.y = 0;
  return Number.isFinite(lastA) && lastA >= 0.5 ? 1 : 0;
}

class LongNightRoadScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private road: LanternRoadHandle | null = null;
  private milestone: MilestoneHandle | null = null;
  private wisp: Wisp | null = null;
  private glow: WispGlowHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });
  private goalCell = 0;

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private waveAt = -Infinity;
  private surgeAt = -Infinity;
  private burstAt = -Infinity;
  /** true while any lantern is still writing a non-zero colour */
  private waveLit = false;

  // ---- preallocated scratch: update() may not allocate ----
  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly overlayCell: GridCell = { x: 0, y: 0 };
  private readonly overlayGoal: GridCell = { x: 0, y: 0 };
  private readonly burstPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly shadowCenter = { x: 0, y: 0.3, z: 0 };
  private shadowCell = Number.NaN;
  private camX = 0;
  private accent = 0x7fe0c4;
  private gold = 0xffd166;
  private flame = 0xffd166;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'longNightRoadScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.goalCell = goalCellOf(grid);

    const palette = ctx.tokens.palettes.night;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.uiCyan;
    this.accent = accent;
    this.gold = ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);
    this.flame = world.skin.flame.getHex();

    // The four point lights this scene claims from the rig. COLORS are static — only
    // position and intensity are state-driven — so update() never touches a Color.
    //
    // THE CLAIM ORDER IS THE PRIORITY ORDER, because the rig hands out four at hero, TWO at
    // medium and none at thumb (and every use below is guarded). The two that survive the
    // cut are the two the scene cannot be read without: the agent, and the address of the
    // reward. The wave-front and burst lights are celebration gravy — the ignition itself
    // is instanced emissive lamps and their ground pools, so the wave plays in full at
    // every tier with no lights at all.
    const wispLight = world.rig.points[0];
    if (wispLight) wispLight.color.set(accent);
    // A STANDING TEAL FILL over the last few cells — the single most load-bearing light in
    // the scene. DelayedChain's reward lives at pos 31 and nowhere else, so the end of the
    // road must be legible as a DESTINATION from every cell of the walk; without this the
    // cairn and the end lantern are black shapes on a black moor and the road just fades
    // out. Hung 3.4 above cell 29 with the rig's 9-unit falloff, it washes about six cells
    // and dies well before the middle of the walk — the reward's address, lit by its own
    // sky, and nothing else on the moor lit at all.
    const auroraLight = world.rig.points[1];
    if (auroraLight) {
      auroraLight.color.set(accent);
      auroraLight.position.set(cellToWorld(this.goalCell - 2, 0, grid)[0], 3.4, -1.5);
      auroraLight.intensity = 3.6;
    }
    const waveLight = world.rig.points[2];
    if (waveLight) waveLight.color.set(this.flame);
    const goalLight = world.rig.points[3];
    if (goalLight) {
      goalLight.color.set(this.gold);
      goalLight.position.set(cellToWorld(this.goalCell, 0, grid)[0], LAMP_Y, 0);
    }

    // ---- the road of lanterns ----
    const road = buildLanternRoad({
      tier: ctx.tier,
      palette,
      skin: world.skin,
      grid,
      goalCell: this.goalCell,
      seed: ctx.seed,
      accent,
      name: 'lanternRoad',
    });
    this.road = road;
    this.scene.add(road.group);
    for (let i = 0; i < road.count; i++) road.setGlow(i, 0);
    road.commit();

    const milestone = buildMilestone({
      tier: ctx.tier,
      palette,
      skin: world.skin,
      grid,
      cell: 0,
      name: 'startMilestone',
    });
    this.milestone = milestone;
    this.scene.add(milestone.group);

    // ---- the shared avatar: the brightest thing on the road ----
    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        // essentially the kit default: the road camera is CLOSE (8.4 units), so the orb
        // holds the frame on brightness, not on size
        radius: 0.165,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'roadWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);
    // THE HALO IS SWITCHED OFF; NOTHING ELSE IS TOUCHED. The kit halo is a detail-1
    // icosphere and this camera is close enough to count its facets, so the job is handed
    // to a real radial falloff (see buildWispGlow). The obvious call — `wisp.dim(0.1)`,
    // which this scene shipped first — is NOT a halo control: kit/wisp's applyBrightness()
    // multiplies that number into the CORE rim (1.9 → 0.19), the ribbon opacity (0.4 →
    // 0.04) and the skirt opacity (0.85 → 0.085) as well, i.e. it dimmed the entire avatar
    // ten-fold and deleted the motion ribbon, on the one scene whose whole claim is that
    // the agent is the brightest thing in the frame. The halo is a NAMED CHILD of the kit
    // rig, so hiding that one object is exact, costs a draw call rather than a fade, and
    // leaves every other part of the avatar at the brightness the kit shipped.
    // (Kit gap, filed in concerns: `Wisp` has no halo-only control, e.g. `haloGain(g)`.)
    const kitHalo = wisp.group.getObjectByName('oakWispHalo');
    if (kitHalo) kitHalo.visible = false;
    wisp.snap();

    const glow = buildWispGlow({
      color: accent,
      tier: ctx.tier,
      radius: ctx.tier === 'thumb' ? 0.8 : 1.05,
      intensity: 0.72,
      name: 'wispGlow',
    });
    this.glow = glow;
    this.scene.add(glow.mesh);

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: thumb ? CAM_TARGET_THUMB : [0, CAM_TARGET_Y, 0],
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: thumb ? CAM_AZIMUTH_THUMB : CAMERA_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      // THE LIMITS ARE PER-TIER, and that is not a detail. `makeOrbitRig` CLAMPS the
      // opening azimuth/polar/distance into the limits it is given, so a single hero-shaped
      // limit box silently dragged the thumb camera from (−0.18, 27.5) to (−0.71, 16) — the
      // fixed wall-view frame was a bug, not a taste call, and it took a projection
      // calculation that disagreed with the screenshot to find it. Thumb has no canvas and
      // no user input, so its "limits" only have to contain its own one frame.
      limits: thumb
        ? {
            minPolar: CAM_POLAR_THUMB,
            maxPolar: CAM_POLAR_THUMB,
            minAzimuth: CAM_AZIMUTH_THUMB,
            maxAzimuth: CAM_AZIMUTH_THUMB,
            minDistance: CAM_DISTANCE_THUMB,
            maxDistance: CAM_DISTANCE_THUMB,
          }
        : {
            // never under the moor, never a plan view: this scene lives on the horizon line
            minPolar: 1.15,
            maxPolar: 1.52,
            minAzimuth: CAMERA_AZIMUTH - 0.45,
            maxAzimuth: CAMERA_AZIMUTH + 0.45,
            minDistance: 4.5,
            maxDistance: 16,
          },
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.0035,
      zoomSpeed: 0.0012,
      name: 'roadOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'roadCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        readState(state, grid, this.overlayCell);
        return this.overlayCell;
      },
      goalCell: () => {
        this.overlayGoal.x = this.goalCell;
        this.overlayGoal.y = 0;
        return this.overlayGoal;
      },
      name: 'roadGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    world.aurora.setGain(1);
    road.setTime(0);
    milestone.setTime(0);
    wisp.setCell(0, 0, grid);
    this.camX = wisp.position.x;
    if (!thumb) rig.setTarget(this.camX + CAM_LOOK_AHEAD, CAM_TARGET_Y, 0);
    rig.snap();
    glow.mesh.position.copy(wisp.position);
    glow.mesh.position.y += ctx.tokens.wispHeight;
    glow.mesh.quaternion.copy(this.camera.quaternion);
    glow.setGain(1);
    this.applyShadowFocus(wisp.position.x, world);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const road = this.road;
    const wisp = this.wisp;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !road || !wisp || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const agent = this.agentCell;
    const thumb = ctx.tier === 'thumb';

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. the frame's truth, snapped to a cell
    const lastA = readState(state, grid, agent);

    // 3. a scrub wipes every in-flight beat: nothing may survive a discontinuity, and
    //    that explicitly includes the ignition wave (the clock goes to −Infinity, so the
    //    very next sweep writes cold iron on all thirty-two lanterns)
    let snapCamera = false;
    if (frame.scrubbed) {
      pool.clear();
      this.waveAt = -Infinity;
      this.surgeAt = -Infinity;
      this.burstAt = -Infinity;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      snapCamera = true;
    }

    // 4. events — celebrations are NEVER inferred from the state stream
    let teleported = false;
    let paid = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        // THE PAY-OFF IS AT CELL 31, NOT WHERE THE AGENT IS. The env wrapped the agent
        // to pos 0 in the same step it paid, so reading the agent's cell here would
        // bloom the milestone at the start of the road.
        this.burstPos.copy(road.endLamp);
        cellToWorldVec(this.goalCell, 0, grid, this.groundPos);
        pool.burst(this.burstPos, this.flame, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        // BEAT 3, AND IT MUST BE TAKEN BACK OFF THE POOL BEFORE IT CAN BE SET. `burst()`
        // fires the bloom channel itself at `min(1, scale)`, and `PulseChannel.fire` is a
        // max() — so on the one scene in the gallery whose burst scale is above 1, a bare
        // `fire(PULSE_STRENGTH)` was dead code and the pay-off shipped a FULL-strength
        // bloom pulse over a night palette (a clipped white core on the wisp, measured).
        // `reset()` is the only public way back down; it is exact, and it is safe here
        // because nothing else writes this channel in the same frame and this scene never
        // sighs (`done` always rides with `goal`).
        pool.pulse.reset();
        pool.pulse.fire(PULSE_STRENGTH);
        road.flare(1);
        paid = true;
        this.waveAt = time;
        this.surgeAt = time;
        this.burstAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
      // `done` in this env ALWAYS rides with `goal` — the +1 and the wrap are one step.
      // There is no failure in the stream, so this scene must never sigh.
    }

    // 5. the wrap is a 30-cell jump: hard cut, never a glide across the moor.
    //    The celebration fired above survives it — only the avatar and camera snap.
    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      snapCamera = true;
    }
    // ...and the avatar's own flash is armed AFTER that snap, never inside the event loop.
    // `Wisp.snap()` drops every eased value to its target — the flare decay included — and
    // in THIS env `goal` and `teleport` are always the same step, so a flare fired before
    // the snap is extinguished by it on the very frame it was earned (measured: the orb
    // rim never left 1.9). A scrub still kills it, which is correct: a scrub is not an event.
    if (paid) wisp.flare(WISP_FLARE);

    // 6. the avatar: one straight hop per cell change, leaning into the way it stepped
    if (agent.x !== wisp.cellX) wisp.hopTo(agent, time);
    wisp.avatar.rotation.z = lastA === 1 ? -WISP_LEAN : WISP_LEAN;
    wisp.update(frame.dtReal, time);

    // 7. THE WAVE. A pure function of (time − the goal event) with a hard cutoff, so it
    //    is idempotent, scrub-safe and free of per-lantern state. `waveLit` only exists
    //    to stop writing colours once the road is cold again.
    const age = time - this.waveAt;
    const active = age >= 0 && age <= WAVE_SPAN;
    if (active || this.waveLit) {
      const count = road.count;
      for (let i = 0; i < count; i++) {
        const g = active ? lanternGlow(age, i, count) : 0;
        road.setGlow(i, g > 0 ? g * flicker(time, i) : 0);
      }
      road.commit();
      this.waveLit = active;
    }

    // 8. the point lights this scene claims from the rig
    const burst = envelope(time - this.burstAt, 1.6, 2.2);
    const surge = envelope(time - this.surgeAt, SURGE_SECONDS, 1.5);

    // the wisp's own lantern-glow: a billboard, so it carries no state of its own —
    // exactly the avatar's position, exactly the camera's facing, every frame
    const glow = this.glow;
    if (glow) {
      glow.mesh.position.copy(wisp.position);
      glow.mesh.position.y += TOKENS.wispHeight;
      glow.mesh.quaternion.copy(this.camera.quaternion);
      // …and the same retune as WISP_FLARE: this multiplier was set against a kit avatar
      // that was being dimmed 10×, so at 2.4 it now blooms a 2-unit disc of white over the
      // near foreground on the pay-off frame. The glow lifts; the orb is the thing that
      // flares.
      glow.setGain(1 + 1.1 * burst);
    }

    const wispLight = world.rig.points[0];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.62 + 1.4 * burst;
    }
    // points[1] is the standing aurora fill: set once at init, never state-driven.
    const waveLight = world.rig.points[2];
    if (waveLight) {
      // a warm light travelling with the wave front, end to start
      const front = active ? 1 - clamp01(age / WAVE_TRAVEL) : 0;
      waveLight.position.set(road.lanternX(front * (road.count - 1)), 1.15, -0.2);
      waveLight.intensity = active ? 2.6 * envelope(age, WAVE_TRAVEL + 0.45, 1.1) : 0;
    }
    const goalLight = world.rig.points[3];
    if (goalLight) goalLight.intensity = 3.2 * burst;

    // 9. one clock for the whole moor
    world.aurora.setGain(1 + 1.35 * surge);
    world.setTime(time);
    road.setTime(time);
    road.update(frame.dtReal, time);
    this.milestone?.setTime(time);

    // 10. the camera walks with the wisp; the shadow frustum follows the same walk so a
    //     2048 map covers 18 units of road instead of 32 (canon lesson 8)
    if (!thumb) {
      const wispX = wisp.position.x;
      if (snapCamera) this.camX = wispX;
      else this.camX += (wispX - this.camX) * (1 - Math.exp(-CAM_FOLLOW_RATE * Math.max(0, frame.dtReal)));
      rig.setTarget(this.camX + CAM_LOOK_AHEAD, CAM_TARGET_Y, 0);
    }
    if (snapCamera) rig.snap();
    else rig.update(frame.dtReal);
    this.applyShadowFocus(wisp.position.x, world);

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
    const road = this.road;
    const milestone = this.milestone;
    const goalCell = this.goalCell;

    const wispCell: GridCell = { x: 0, y: 0 };
    const wispOut: Vec3Tuple = [0, 0, 0];
    const endOut: Vec3Tuple = [0, 0, 0];
    const startOut: Vec3Tuple = [0, 0, 0];

    // THE WISP PROBE RULE (four-rooms lesson 16): this scene LEANS its wisp, which is a
    // rotation on `avatar`, so `avatar` is the one anchor that is still pure translation
    // — and it is exactly `cellToWorld(pos, 0)`. The orb core hangs under that lean and
    // is deliberately NOT probed.
    return [
      {
        name: 'wisp',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readState(state, grid, wispCell);
          return cellToWorld(wispCell.x, 0, grid, wispOut);
        },
      },
      {
        // the paying cell's own lantern: the one place in this env reward exists
        name: 'end-lantern',
        object: road?.endRoot ?? new THREE.Object3D(),
        expectWorldPos: () => cellToWorld(goalCell, 0, grid, endOut),
      },
      {
        // and the stone the wrap always lands on
        name: 'start-milestone',
        object: milestone?.root ?? new THREE.Object3D(),
        expectWorldPos: () => cellToWorld(0, 0, grid, startOut),
      },
    ];
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.milestone?.dispose();
    this.milestone = null;
    this.road?.dispose();
    this.road = null;
    this.glow?.dispose();
    this.glow = null;
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
    this.waveAt = -Infinity;
    this.surgeAt = -Infinity;
    this.burstAt = -Infinity;
    this.waveLit = false;
    this.shadowCell = Number.NaN;
  }

  /** Re-fit the shadow frustum around the wisp, quantized to a cell to stop it shimmering. */
  private applyShadowFocus(x: number, world: WorldHandle): void {
    const cell = Math.round(x);
    if (cell === this.shadowCell) return;
    this.shadowCell = cell;
    this.shadowCenter.x = cell;
    world.rig.setShadowFocus(this.shadowCenter, SHADOW_RADIUS);
  }
}

// ---------------------------------------------------------------------------
// the wave — pure functions, measured on the playback clock, tested directly
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

/**
 * THE IGNITION WAVE. Brightness of the lantern on cell `index` at `age` seconds after
 * the +1, on a road of `count` cells.
 *
 * The credit starts where it was earned — the LAST cell — and walks back to the first,
 * which is the shape of the update this env is famous for being slow at. Everything is
 * read off `age`: a lantern that the wave has not reached yet is 0, a lantern the wave
 * has finished with is 0, and `age < 0` (a scrub, or no goal yet) is 0 everywhere. There
 * is no state to leak, so a scrubbed frame lands on a cold road with no easing at all.
 */
export function lanternGlow(age: number, index: number, count: number): number {
  if (!(age >= 0)) return 0;
  const n = Math.max(1, count - 1);
  const i = index < 0 ? 0 : index > n ? n : index;
  /** 0 at the paying end, 1 at the start — the wave's own delay */
  const back = (n - i) / n;
  const local = age - back * WAVE_TRAVEL;
  if (local < 0 || local > LANTERN_SPAN) return 0;
  if (local < IGNITE_ATTACK) {
    const t = local / IGNITE_ATTACK;
    return t * t * (3 - 2 * t);
  }
  const fall = local - IGNITE_ATTACK;
  if (fall < IGNITE_HOLD) return 1;
  const g = (fall - IGNITE_HOLD) / IGNITE_GUTTER;
  const e = Math.exp(-2.1 * g) * (1 - g);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

/** A lantern's own flame-wobble: pure in (time, index), so it is scrub- and pause-safe. */
export function flicker(time: number, index: number): number {
  return 0.87 + 0.13 * Math.sin(time * 9.3 + index * 2.3994);
}

const createScene: SceneFactory = () => new LongNightRoadScene();
export default createScene;
