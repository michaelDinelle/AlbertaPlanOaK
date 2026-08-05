// Four Seasons Courtyard — FourRooms-cont-v0. The GRID CANON of the OaK Preserve.
//
// THE ONE IDEA: `state.room` IS the art. The env already partitions its 11×11 courtyard
// into four rooms plus a cross of hallways; this scene paints room 0 spring, 1 summer,
// 2 autumn, 3 winter and −1 evergreen-and-stone, so "which room is the agent in" is
// answered by looking at the frame for a quarter of a second. Nothing is re-derived:
// walls come from `grid.walls`, hallways from `grid.hallways`, rooms from the decoder's
// own `fourRoomsRoom`.
//
// PATTERNS THE NINE OTHER GRID SCENES SHOULD COPY
//  * The AVATAR is `kit/wisp`, and its two anchors are both probed here: `wisp.avatar`
//    is the cell (y = 0, exactly `cellToWorld`) and `wisp.probeObject` is the orb centre
//    (y = `tokens.wispHeight`). Every bob, breath, skirt spin and ribbon hangs off
//    `decor`, so both anchors stay bit-exact while the wisp is alive.
//  * MOTION is `HopMotion` through `wisp.hopTo`, one straight arc per state change and
//    never a path plan. Its duration (0.11 s) is deliberately SHORTER than one control
//    step at 8 Hz (0.125 s), so the glide always lands before the next truth arrives:
//    the wisp is at most one cell — and less than one step — behind the snapped state,
//    which is the whole latency budget a grid scene is allowed.
//  * CELEBRATIONS fire ONLY from `frame.events`. FourRooms' `goal` fires on the frame
//    the agent ENTERS the old goal cell, and the goal has ALREADY respawned by then, so
//    the burst belongs to the agent's own cell and the new lantern grows in elsewhere.
//    Reading it off the goal's new position would bloom the wrong end of the courtyard.
//  * There is no `done`, no failure and no agent teleport in this env (the agent keeps
//    its cell; only the goal moves), so this scene never sighs and never hard-cuts. The
//    `teleport` branch below exists for the contract, not for this stream.
//  * Every flourish is measured on `ctx.time()` — growth, bloom, petal breath, hedge
//    wind, the four seasonal mote fields — so a paused playback freezes the courtyard
//    coherently, mid-snowflake.

import { cellToWorld, cellToWorldVec, clampCell, type GridSize, type Vec3Tuple } from '@kit/grid';
import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
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
import { buildLantern, type LanternHandle } from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  blendTokens,
  buildWorld,
  courtyardSkin,
  resolveGrid,
  shade,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. A 3/4 aerial from the south-east at 47° above the horizon: high enough that
// all four rooms and the cross of hedges read as a GRID (the whole point), low enough
// that the hedges, the moon gates and the sundial keep their silhouettes. The user may
// orbit ±0.6 rad and tilt inside a 30° band — enough to look around a hedge, never
// enough to lose the plan view or drop under the lawn.
// The distance is NOT a taste call. Seen from azimuth a, a square courtyard presents
// 11·(|cos a| + |sin a|) across the frame — 15.0 units at a = 0.55 — and a 47° aerial
// lays that out over 11.0 units of screen height. A 40° vertical FOV covers 2·d·tan20°,
// so d must clear 15.1 just to touch the edges; 18.5 leaves the margin that makes it a
// composition instead of a crop. The first pass at 15.6 cut both far corners off.
const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.75;
const CAM_DISTANCE = 17.8;
const CAM_TARGET_Y = 0.5;
const CAM_YAW_RANGE = 0.6;
/** thumb pulls back and stops moving: a square-ish crop must still hold 11 × 11 cells */
const CAM_DISTANCE_THUMB = 21.5;
const CAM_AZIMUTH_THUMB = 0.42;
const CAM_POLAR_THUMB = 0.72;

// Celebration weights, tuned against a REAL goal (seed 8). The courtyard is 11 units of
// bright noon lawn: the kit's full-strength grammar (a 3.2-unit ripple and a 1.0 bloom
// pulse) washes three cells of the grid out at once, which is exactly the readability
// this scene is built to protect. Two thirds of the amplitude, all of the beat.
/** the burst travels ~0.77 world units — a little under one cell */
const BURST_SCALE = 0.85;
const RIPPLE_SCALE = 0.5;
const BURST_PARTICLES = 110;
const PULSE_STRENGTH = 0.5;
/** the burst blooms at the flower's crown height, not on the flagstones */
const BLOOM_Y = 0.42;
const BLOOM_SECONDS = 1.5;
/** a new lantern takes this long to grow in at its new cell (no teleport-slide) */
const GROW_SECONDS = 0.72;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.17;

// the lantern's idle life: a bud that breathes, and opens as the wisp closes in
const OPEN_BASE = 0.3;
const OPEN_BREATH = 0.18;
const OPEN_PROXIMITY = 0.4;
const PROXIMITY_CELLS = 4;

interface FourRoomsLike {
  x?: unknown;
  y?: unknown;
  gx?: unknown;
  gy?: unknown;
}

/**
 * The agent and goal cells for a frame. sim-core's decoded state is the source; the raw
 * pose (`[x, y, gx, gy, …]`) is the fallback so the scene still stands up under the
 * harness's fixture loop, which carries no `derived`.
 */
function readCells(state: FrameState, grid: GridSize, agent: GridCell, goal: GridCell): void {
  const d = state.derived as FourRoomsLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const ax = has && typeof d.x === 'number' ? d.x : (state.pose[0] ?? 0);
  const ay = has && typeof d.y === 'number' ? d.y : (state.pose[1] ?? 0);
  const gx = has && typeof d.gx === 'number' ? d.gx : (state.pose[2] ?? 0);
  const gy = has && typeof d.gy === 'number' ? d.gy : (state.pose[3] ?? 0);
  agent.x = Number.isFinite(ax) ? Math.round(ax) : 0;
  agent.y = Number.isFinite(ay) ? Math.round(ay) : 0;
  goal.x = Number.isFinite(gx) ? Math.round(gx) : 0;
  goal.y = Number.isFinite(gy) ? Math.round(gy) : 0;
  clampCell(agent, grid);
  clampCell(goal, grid);
}

class FourSeasonsCourtyardScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private lantern: LanternHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private growAt = -Infinity;
  private bloomAt = -Infinity;

  // ---- preallocated scratch: update() may not allocate ----
  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private blossom = 0xf7a8b8;
  private gold = 0xffd166;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'fourSeasonsCourtyardScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;

    const palette = ctx.tokens.palettes.noon;
    const skin = courtyardSkin(palette, ctx.tokens);
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.blossom = ctx.tokens.accents['chain'] ?? ctx.tokens.celebrationGold;
    this.gold = ctx.tokens.celebrationGold;
    const flame = blendTokens(ctx.tokens.celebrationGold, palette.sun, 0.22).getHex();

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    // This scene claims three of the rig's point lights. Their COLORS are static — only
    // position and intensity are state-driven — so update() never touches a Color.
    const lanternLight = world.rig.points[0];
    if (lanternLight) lanternLight.color.set(flame);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const bloomLight = world.rig.points[2];
    if (bloomLight) bloomLight.color.set(this.blossom);

    // ---- the shared avatar ----
    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        // a shade over the 0.16 default: from the composed 17.8-unit standoff the
        // default orb is ~14 px of pale green on a pale floor and the eye loses it
        radius: 0.21,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'courtyardWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    // ---- the goal: a lantern-flower on whatever cell the goal has respawned to ----
    const lantern = buildLantern({
      tier: ctx.tier,
      palette,
      // a DEEP rose, not the spring bed's pale one: the petals sit under a 3.0 noon key
      // with a flame rimming them from inside, and a pale pink there renders white
      blossom: shade(blendTokens(this.blossom, palette.groundB, 0.12), 0.3),
      flame,
      stem: skin.seasons[1]?.hedge ?? palette.groundA,
      name: 'lanternFlower',
    });
    this.lantern = lantern;
    this.scene.add(lantern.group);

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [0, CAM_TARGET_Y, 0],
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: thumb ? CAM_AZIMUTH_THUMB : CAM_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0012,
      name: 'courtyardOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'courtyardCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
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
      name: 'courtyardGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    lantern.setState(OPEN_BASE, 1);
    lantern.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const lantern = this.lantern;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !lantern || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const agent = this.agentCell;
    const goal = this.goalCell;

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. the frame's truth, snapped to cells
    readCells(state, grid, agent, goal);

    // 3. a scrub wipes every in-flight beat: nothing may survive a discontinuity
    if (frame.scrubbed) {
      pool.clear();
      this.bloomAt = -Infinity;
      this.growAt = -Infinity;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      rig.snap();
    }

    // 4. events — celebrations are NEVER inferred from the state stream
    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        // The +1 fires as the agent ENTERS the old goal cell and the goal respawns in
        // the same step, so the bloom belongs to the agent's cell — the flower that was
        // standing there has just burst into petals.
        cellToWorldVec(agent.x, agent.y, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.blossom, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        lantern.flare(0.7);
        this.bloomAt = time;
        this.growAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
      // FourRooms emits no `done` and no failure: this scene must never sigh.
    }
    // Contract completeness only — this env never moves the agent discontinuously.
    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      rig.snap();
    }

    // 5. the avatar: one straight hop per cell change, never a path plan.
    //    `wisp.face()` is deliberately NOT called — see the probe note in `probes()`.
    if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      wisp.hopTo(agent, time);
    }
    wisp.update(frame.dtReal, time);

    // 6. the lantern: position is pure state, growth is pure event clock
    cellToWorldVec(goal.x, goal.y, grid, lantern.root.position);
    const grow = growth(time - this.growAt);
    const open = opening(time, agent, goal);
    lantern.setState(open, grow);
    lantern.setTime(time);
    lantern.update(frame.dtReal, time);

    // 7. the three point lights this scene claims from the rig
    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const lanternLight = world.rig.points[0];
    if (lanternLight) {
      lanternLight.position.set(
        lantern.root.position.x,
        lantern.crownHeight,
        lantern.root.position.z,
      );
      lanternLight.intensity = (0.14 + 0.46 * open) * grow;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.13 + 0.8 * bloom;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 2.2 * bloom;
    }

    // 8. one clock for the whole courtyard
    world.setTime(time);

    // 9. camera easing runs in WALL time so a paused frame is still explorable
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
    const lantern = this.lantern;
    const world = this.world;
    const height = this.ctx?.tokens.wispHeight ?? TOKENS.wispHeight;

    const cellFor = (state: FrameState, agent: GridCell, goal: GridCell): void => {
      readCells(state, grid, agent, goal);
    };

    const wispCell: GridCell = { x: 0, y: 0 };
    const wispGoal: GridCell = { x: 0, y: 0 };
    const wispOut: Vec3Tuple = [0, 0, 0];
    const anchorCell: GridCell = { x: 0, y: 0 };
    const anchorGoal: GridCell = { x: 0, y: 0 };
    const anchorOut: Vec3Tuple = [0, 0, 0];
    const goalCell: GridCell = { x: 0, y: 0 };
    const goalGoal: GridCell = { x: 0, y: 0 };
    const goalOut: Vec3Tuple = [0, 0, 0];
    const gateOut: Vec3Tuple = [0, 0, 0];
    const sundialOut: Vec3Tuple = [0, 0, 0];

    // THE PROBE RULE FOR EVERY GRID SCENE, learned here: `assertUpdateIdempotent`
    // compares probe objects' whole WORLD MATRIX, not just their translation, so a probe
    // may carry no eased value of any kind — position OR rotation OR scale. `Wisp.face()`
    // damps the body yaw toward its target over `dt`, and `wisp.probeObject` (the orb
    // core) is a child of that body, so a scene that turns its wisp MUST move this probe
    // down to `wisp.avatar`, which is pure. This courtyard's wisp is a radially symmetric
    // spirit whose skirt already spins on its own clock, so facing buys it nothing — it
    // never calls `face()`, and both anchors stay exact and probeable.
    const probes: AlignmentProbe[] = [
      {
        // the orb centre: exactly `wispHeight` over the true cell, bob and breath and
        // skirt and ribbon all excluded (they live on `decor`)
        name: 'wisp',
        object: wisp?.probeObject ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          cellFor(state, wispCell, wispGoal);
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
          cellFor(state, anchorCell, anchorGoal);
          return cellToWorld(anchorCell.x, anchorCell.y, grid, anchorOut);
        },
      },
      {
        // the goal beacon root — it must land on the RESPAWNED cell, every time
        name: 'goal-lantern',
        object: lantern?.root ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          cellFor(state, goalCell, goalGoal);
          return cellToWorld(goalGoal.x, goalGoal.y, grid, goalOut);
        },
      },
    ];

    const hallway = world?.hallways[0];
    const gateRoot = world?.gateRoots[0];
    if (hallway && gateRoot) {
      probes.push({
        // a moon gate sits on the env's own hallway cell, never on a guessed gap
        name: 'moon-gate',
        object: gateRoot,
        expectWorldPos: () => cellToWorld(hallway.x, hallway.y, grid, gateOut),
      });
    }
    if (world) {
      probes.push({
        // the sundial owns the centre wall cell
        name: 'sundial',
        object: world.sundialRoot,
        expectWorldPos: () =>
          cellToWorld((grid.w / 2) | 0, (grid.h / 2) | 0, grid, sundialOut),
      });
    }
    return probes;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.lantern?.dispose();
    this.lantern = null;
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

/**
 * The new lantern's scale-in. A never-fired clock (`-Infinity`) and any age past the
 * window both return 1, which is what makes a scrub land on a settled courtyard instead
 * of replaying a growth it never saw.
 */
export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.16 * Math.sin(Math.PI * t) * (1 - t));
}

/**
 * How wide the lantern-flower stands open: a slow breath plus a term that rises as the
 * wisp closes in. Both are pure functions of `(time, state)` — no history, no easing —
 * so the flower is identical on a scrubbed frame and on a played one. The proximity term
 * shows nothing the agent does not have: it is the agent's own cell and the goal's own
 * cell, the two numbers the frame is made of.
 */
export function opening(time: number, agent: GridCell, goal: GridCell): number {
  const d = Math.max(Math.abs(agent.x - goal.x), Math.abs(agent.y - goal.y));
  const near = d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
  const breath = 0.5 + 0.5 * Math.sin(time * 0.72);
  const v = OPEN_BASE + OPEN_BREATH * breath + OPEN_PROXIMITY * near;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

const createScene: SceneFactory = () => new FourSeasonsCourtyardScene();
export default createScene;
