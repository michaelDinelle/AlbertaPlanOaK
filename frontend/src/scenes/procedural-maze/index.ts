// The Overgrown Ruins — ProceduralMaze-cont-v0.
//
// THE ONE IDEA: this env has no layout to illustrate. It regenerates its 9×9 interior
// from the lane seed at fleet init, so what an artisan can author is not a place but a
// MASONRY SYSTEM — `world.ts`'s `skinWallCell`, a pure function from "cell (x, y) is a 1
// in `grid.walls`" to a specific ruined block. Point it at another seed and another ruin
// comes out, correct by construction, with the same hands visible in the stonework.
//
// WHAT THIS SCENE ADDS TO THE GRID CANON
//  * BOTH ENDS TELEPORT. In FourRooms only the goal respawns; here `gw_pmaze_step`
//    redraws the AGENT as well on a completion, so the frame carrying the `goal` event
//    already holds a new agent cell somewhere else entirely. The wisp therefore hard-cuts
//    on `teleport` (it must never glide across half a maze), and the celebration is read
//    from `frame.prev`'s GOAL cell — the square the +1 was actually earned on, which by
//    then is the only place in the frame that still remembers where anything happened.
//  * `done` IS NOT A FAILURE HERE. The decoder fires `goal`, `teleport` and `done`
//    together on the same completion, so this scene celebrates and never sighs; a sigh on
//    that batch would be the shared failure grammar firing on a success.
//  * THE ROUTE IS NEVER SHOWN. A respawn draws uniformly from the free cells of a
//    perfect maze — always reachable, but the way there is the agent's problem. The
//    sunbeam marks the goal cell and says nothing else — no trail, no arrow, no glow
//    along the way. `geometry.grid.carvedPath` exists, and it belongs to `kit/debug`
//    alone.
//
// Every flourish is measured on `ctx.time()` — the grow-in, the bloom, the beam's dust
// banding, the ivy tremble, the wheeling birds — so a paused playback freezes the ruin
// coherently, mid-mote.

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
import { buildSunShaft, type SunShaftHandle } from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  CAMERA_DISTANCE,
  CAMERA_POLAR,
  CAMERA_TARGET_BIAS,
  CAMERA_TARGET_Y,
  SUN_DIR,
  blendTokens,
  buildWorld,
  floorTintFor,
  resolveGrid,
  ruinSkin,
  type RuinSkin,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. A 3/4 aerial from the south-east, 48° above the horizon — a shade higher
// than the four-rooms courtyard, because this playfield is nearly 60% masonry and a
// flatter angle turns the passages into slots you cannot see the floor of.
//
// The numbers live in `world.ts`, next to the tree planting that has to agree with them,
// and they are MEASURED, not chosen: `procedural-maze.test.ts` projects all four corner
// towers (±0.5 cell, footing and 1.74-unit crown) through this very camera at 16:9, 4:3
// and 1:1 and holds every one of them inside |ndc| ≤ 0.94. The first pass framed on the
// grid origin at d = 17 and hung the whole south-east tower off the bottom edge
// (ndc.y −1.17) at BOTH aspects — a canon-lesson-21 miss with a comment claiming the
// opposite. Pulling back to 19 and sliding the pivot 1.2 units toward the near corner is
// what fixed it; raising the polar or the target only made it worse (both measured).
const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = CAMERA_POLAR;
const CAM_DISTANCE = CAMERA_DISTANCE;
const CAM_TARGET_Y = CAMERA_TARGET_Y;
const CAM_TARGET_BIAS = CAMERA_TARGET_BIAS;
const CAM_YAW_RANGE = 0.6;
/**
 * Thumb pulls back and stops moving, and it pulls back FURTHER than the hero because the
 * wall view is a square-ish crop: at aspect 1 the horizontal half-angle drops to 20°, and
 * 21.8 sliced the west tower in half against the left edge (|ndc.x| 1.02 on its
 * silhouette, 0.96 on its cell centre). 25.0 holds every corner tower inside 0.92 at
 * aspect 1, and 0.90 at 16:9 — the whole 11 × 11, towers included, with ~8% to spare.
 */
const CAM_DISTANCE_THUMB = 25.0;
const CAM_AZIMUTH_THUMB = 0.5;
const CAM_POLAR_THUMB = 0.66;

// Celebration weights. The ruin is 11 units of bright noon stone: the kit's full-strength
// grammar (a 3.2-unit ripple, a 1.0 bloom pulse) washes three cells of maze out at once,
// which is exactly the readability this scene exists to protect. Two thirds of the
// amplitude, all three beats.
const BURST_SCALE = 0.82;
const RIPPLE_SCALE = 0.48;
const BURST_PARTICLES = 110;
const PULSE_STRENGTH = 0.5;
/** the burst blooms at about the height of a standing block, not on the flagstones */
const BLOOM_Y = 0.46;
const BLOOM_SECONDS = 1.5;
/** the sun takes this long to find its way into a newly opened atrium */
const GROW_SECONDS = 0.8;

/** one hop is shorter than one 8 Hz control step (0.125 s), so the glide lands on time */
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;

interface ProceduralMazeLike {
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
  const d = state.derived as ProceduralMazeLike | null | undefined;
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

/** The goal cell of a frame only — used on `prev`, where nothing else matters. */
export function readGoalCell(state: FrameState, grid: GridSize, out: GridCell): GridCell {
  const d = state.derived as ProceduralMazeLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const gx = has && typeof d.gx === 'number' ? d.gx : (state.pose[2] ?? 0);
  const gy = has && typeof d.gy === 'number' ? d.gy : (state.pose[3] ?? 0);
  out.x = Number.isFinite(gx) ? Math.round(gx) : 0;
  out.y = Number.isFinite(gy) ? Math.round(gy) : 0;
  return clampCell(out, grid);
}

class OvergrownRuinsScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private shaft: SunShaftHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry | null = null;
  private skin: RuinSkin | null = null;
  /** the lane seed — the flagstone tints are a pure function of it, restore included */
  private seed = 0;

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private growAt = -Infinity;
  private bloomAt = -Infinity;

  // ---- preallocated scratch: update() may not allocate ----
  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly prevGoalCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly litTint = new THREE.Color();
  private readonly restoreTint = new THREE.Color();
  /** the cell whose flagstone the beam is currently warming (−1 = none) */
  private litX = -1;
  private litY = -1;
  private gold = TOKENS.celebrationGold;
  private moteColor = TOKENS.celebrationGold;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'overgrownRuinsScene';

    const grid = resolveGrid(ctx.geometry, ctx.seed);
    this.grid = grid;
    this.seed = ctx.seed;

    const palette = ctx.tokens.palettes.noon;
    const skin = ruinSkin(palette, ctx.tokens);
    this.skin = skin;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.gold = ctx.tokens.celebrationGold;
    // A puff of GILDED DUST knocked out of old stone, not a firework. The kit fires bursts
    // at a fixed 2.4 HDR, so a near-white mote colour clips to a white star and the beat
    // stops belonging to this scene; taken down toward the dusk tan it blooms amber.
    this.moteColor = blendTokens(
      ctx.tokens.celebrationGold,
      ctx.tokens.palettes.dusk.groundA,
      0.3,
    ).getHex();
    // the sunlit flagstone: the same worn stone, lifted toward the beam's own warmth
    this.litTint.copy(blendTokens(skin.flag, skin.sunWarm, 0.5));

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    // This scene claims three of the rig's point lights. Their COLORS are static — only
    // position and intensity are state-driven — so update() never touches a Color.
    const beamLight = world.rig.points[0];
    if (beamLight) beamLight.color.set(skin.sunWarm);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const bloomLight = world.rig.points[2];
    if (bloomLight) bloomLight.color.set(this.moteColor);

    // ---- the shared avatar ----
    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        // a shade over the 0.16 default: from the composed ~19.8-unit standoff the
        // default orb is ~13 px of moss green against sunlit stone and the eye loses it
        radius: 0.22,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'ruinWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);
    // `wisp.face()` is deliberately never called — see the probe note in `probes()`.

    // ---- the goal: a shaft of sunlight into whatever cell the goal respawned to ----
    const shaft = buildSunShaft({
      tier: ctx.tier,
      palette,
      skin,
      sunDir: SUN_DIR,
      name: 'sunShaft',
    });
    this.shaft = shaft;
    this.scene.add(shaft.group);

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    // the hero pivot slides along its own azimuth, toward the near corner: that is what
    // re-centres the ruin's projected diagonal instead of hanging it off the bottom edge.
    // The thumb is a plan view of the whole field and orbits the middle of it.
    const bias = thumb ? 0 : CAM_TARGET_BIAS;
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [Math.sin(CAM_AZIMUTH) * bias, CAM_TARGET_Y, Math.cos(CAM_AZIMUTH) * bias],
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
      name: 'ruinOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'ruinCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
    // It draws `grid.carvedPath` on its own, which is the ONLY place that path may ever
    // appear (STYLE.md honesty rule) — nothing in the art above reads it.
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
      name: 'ruinGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    shaft.setState(1);
    shaft.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const shaft = this.shaft;
    const pool = this.pool;
    const rig = this.rig;
    const grid = this.grid;
    if (!ctx || !world || !wisp || !shaft || !pool || !rig || !grid) return;

    const time = ctx.time();
    const state = frame.state;
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
        // The +1 is earned by STEPPING ONTO the goal, and by the time this frame exists
        // both the agent and the goal have been redrawn elsewhere. The one cell in the
        // batch that still remembers where it happened is the PREVIOUS frame's goal.
        const prev = frame.prev;
        if (prev) readGoalCell(prev, grid, this.prevGoalCell);
        else {
          // contract completeness only: the host always supplies `prev` alongside events
          this.prevGoalCell.x = agent.x;
          this.prevGoalCell.y = agent.y;
        }
        cellToWorldVec(this.prevGoalCell.x, this.prevGoalCell.y, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.moteColor, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        shaft.flare(0.9);
        this.bloomAt = time;
        this.growAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
      // `done` fires on this same batch and is the SUCCESS, not a failure: never sigh.
    }

    // 5. the avatar. Both ends of this env teleport on a completion, so a `teleport` is a
    //    hard cut — a glide would slide the wisp straight through half the ruin's walls.
    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
    } else if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      wisp.hopTo(agent, time);
    }
    wisp.update(frame.dtReal, time);

    // 6. the sunbeam: position is pure state, the grow-in is pure event clock
    cellToWorldVec(goal.x, goal.y, grid, shaft.root.position);
    const grow = growth(time - this.growAt);
    shaft.setState(grow);
    shaft.setTime(time);
    shaft.update(frame.dtReal, time);
    // the flagstone the beam lands on warms up; the one it left goes back to exactly what
    // `floorTintFor` says it is, never to an approximation of it
    this.relightFloor(goal.x, goal.y);
    world.setGoalPosition(shaft.root.position.x, 1.5, shaft.root.position.z);

    // 7. the three point lights this scene claims from the rig
    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const beamLight = world.rig.points[0];
    if (beamLight) {
      beamLight.position.set(shaft.root.position.x, shaft.lightHeight, shaft.root.position.z);
      // 1.75 at rest blew the atrium's own stubs to white: the beam is daylight arriving,
      // and daylight does not come with a lamp under it
      beamLight.intensity = (0.22 + 0.6 * grow) * (1 + 0.9 * bloom);
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.16 + 0.8 * bloom;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 2.2 * bloom;
    }

    // 8. one clock for the whole ruin
    world.setTime(time);

    // 9. camera easing runs in WALL time so a paused frame is still explorable
    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
  }

  /**
   * Warm the goal cell's flagstone and cool the one the beam left. Idempotent by
   * construction: it does nothing at all unless the cell changed, and the colour it
   * restores comes from `floorTintFor` — the same function that painted the tile in the
   * first place — so a tile can never end up a shade off after a hundred respawns.
   */
  private relightFloor(x: number, y: number): void {
    if (x === this.litX && y === this.litY) return;
    const world = this.world;
    const grid = this.grid;
    const skin = this.skin;
    if (!world || !grid || !skin) return;
    if (this.litX >= 0) {
      // written into the scene's own Color: a respawn happens inside update(), where a
      // scene may not allocate, and `floorTintFor` would otherwise mint two per event
      floorTintFor(grid, skin, this.litX, this.litY, this.seed, this.restoreTint);
      world.field.setFloorColor(this.litX, this.litY, this.restoreTint);
    }
    world.field.setFloorColor(x, y, this.litTint);
    this.litX = x;
    this.litY = y;
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
    const grid = this.grid ?? resolveGrid({ kind: 'grid' }, this.ctx?.seed ?? 0);
    const wisp = this.wisp;
    const shaft = this.shaft;
    const world = this.world;
    const height = this.ctx?.tokens.wispHeight ?? TOKENS.wispHeight;

    const wispCell: GridCell = { x: 0, y: 0 };
    const wispGoal: GridCell = { x: 0, y: 0 };
    const wispOut: Vec3Tuple = [0, 0, 0];
    const anchorCell: GridCell = { x: 0, y: 0 };
    const anchorGoal: GridCell = { x: 0, y: 0 };
    const anchorOut: Vec3Tuple = [0, 0, 0];
    const shaftCell: GridCell = { x: 0, y: 0 };
    const shaftGoal: GridCell = { x: 0, y: 0 };
    const shaftOut: Vec3Tuple = [0, 0, 0];
    const towerOut: Vec3Tuple = [0, 0, 0];
    const archOut: Vec3Tuple = [0, 0, 0];

    // THE PROBE RULE FOR EVERY GRID SCENE: `assertUpdateIdempotent` compares probe
    // objects' whole WORLD MATRIX, not just their translation, so a probe may carry no
    // eased value of any kind — position OR rotation OR scale. `Wisp.face()` damps the
    // body yaw toward its target over `dt` and `wisp.probeObject` (the orb core) hangs off
    // that body, so a scene that turns its wisp must move this probe down to `wisp.avatar`.
    // This ruin's wisp is a radially symmetric spirit whose skirt already spins on its own
    // clock — facing buys it nothing, it never calls `face()`, and both anchors stay exact.
    const probes: AlignmentProbe[] = [
      {
        // the orb centre: exactly `wispHeight` over the true cell, with the bob, breath,
        // skirt and ribbon all excluded (they live on `decor`)
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
        // the sunbeam's base — it must land on the RESPAWNED goal cell, every time,
        // including the ones nothing can walk to
        name: 'sun-shaft',
        object: shaft?.root ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readCells(state, grid, shaftCell, shaftGoal);
          return cellToWorld(shaftGoal.x, shaftGoal.y, grid, shaftOut);
        },
      },
    ];

    // the north-west corner tower: the one masonry cell EVERY seed is guaranteed to have
    const tower = world?.towerRoots[0];
    if (tower) {
      probes.push({
        name: 'corner-tower',
        object: tower,
        expectWorldPos: () => cellToWorld(0, 0, grid, towerOut),
      });
    }
    // and a ruined doorway, standing on a cell the walls themselves nominated
    const arch = world?.arches[0];
    const archRoot = world?.archRoots[0];
    if (arch && archRoot) {
      probes.push({
        name: 'ruin-arch',
        object: archRoot,
        expectWorldPos: () => cellToWorld(arch.cell.x, arch.cell.y, grid, archOut),
      });
    }
    return probes;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.shaft?.dispose();
    this.shaft = null;
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
    this.grid = null;
    this.skin = null;
    this.seed = 0;
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
    this.litX = -1;
    this.litY = -1;
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
 * The new atrium's opening-up. A never-fired clock (`-Infinity`) and any age past the
 * window both return 1, which is what makes a scrub land on a settled ruin instead of
 * replaying a sunrise it never saw.
 */
export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.13 * Math.sin(Math.PI * t) * (1 - t));
}

const createScene: SceneFactory = () => new OvergrownRuinsScene();
export default createScene;
