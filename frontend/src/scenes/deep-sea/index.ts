// Sunlit Descent — DeepSea-cont-v0. The dive, rendered DOWNWARD.
//
// THE ONE IDEA: **r is depth.** DeepSea's pose is `[r, c, 7, 7]` with the ROW FIRST, and
// every step of the stream increments r, so this scene spends its whole y axis on that
// one fact: row r's shelf top is exactly `reefDepthY(r)`, the diver rides down the
// staircase one terrace per control step, and the treasure sits on the bottom one. (The
// retired 2-D viewer read the pose as `[x, y]` and drew the dive sideways. Every
// conversion in this scene goes through `cellToReefWorld(c, r)`, which cannot be
// transposed by accident.)
//
// WHAT THE ART IS NOT ALLOWED TO INVENT
//  * The reef is the REACHABLE SET, computed by running the decoder's own transition
//    forward from the env's init (`reefReachable`). It comes out as the triangle c ≤ r;
//    everything else is open blue water with no floor at all. Nothing is typed in.
//  * The kelp leans the way action 1 carries the diver at that cell — straight off
//    `grid.actionMask`. STYLE.md's honesty rule names "DeepSea mask currents" as
//    something to SHOW, subtly, and the ground-truth overlay's arrows confirm it cell for
//    cell.
//  * The silt the diver stirs is the COST, not the direction: the env charges
//    `-0.01/N` whenever `action & 1`, whichever way the mask then pushes, so the puff is
//    keyed to `ctrl[0]` exactly as the reward is.
//
// EVENTS. `deriveEvents` fires `resurface` on every wrap (every 7th step — leaving row
// N−2 always wraps) and `goal` alongside it on the ~3% of dives that land in column N−1.
//  * `goal` blooms at the CLAM, not at the agent. Four-rooms' lesson 18 says the bloom
//    belongs to the agent's cell because ITS goal respawns underneath the agent; here the
//    treasure cell never moves and the diver has already been swept back to (0, 0) by the
//    time the event arrives, so the pearl is the only honest place for it.
//  * `resurface` is a HARD CUT. The wisp snaps to the surface and leaves a column of
//    bubbles behind at the exact spot it left — a hop from row 6 to row 0 would be a
//    six-terrace flight the env never performed.
//  * DeepSea emits no `done` and no `teleport` (`emitsDone: false`), so this scene never
//    sighs. The `teleport` branch below exists for the contract, not for this stream.

import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import type { GridSize } from '@kit/grid';
import { makeOrbitRig, type OrbitRig } from '@kit/rigs';
import { TOKENS } from '@kit/tokens';
import type {
  AlignmentProbe,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { buildBubble, buildClam, type BubbleHandle, type ClamHandle } from './actors';
import { manifest } from './manifest';
import {
  cellToReefVec,
  cellToReefWorld,
  clamp01,
  envelope,
  isDiveStep,
  readDive,
  reefDepthYAt,
  reefReachable,
  reefShear,
  resolveGrid,
  type DiveCell,
} from './reef';
import { CAMERA_AZIMUTH, buildWorld, treasureCell, type WorldHandle } from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME, and why it faces the way it does. A dive can be shot from the shallow end,
// looking DOWN the staircase — and it must not be. Receding steps gain
// `spacing·sin(elev)` of screen height and lose `drop·cos(elev)`, and a shelf top is
// visible over the one in front of it precisely when those two are in the ratio that
// cancels them: a receding descent renders either as a flat slab or as a wall of risers.
// (Three passes of this scene were spent proving it.)
//
// So the camera stands at the DEEP end, over the corner where the open water begins, and
// the dive comes toward it. Now each next step is lower AND nearer — the two terms add,
// the staircase marches down the frame, and every shelf top stays in view because
// everything behind it is higher. The reachable triangle becomes a funnel: row 7, eight
// cells wide, fills the foreground with the giant clam on its corner, and row 0's single
// cell is the far apex at the top of the frame where the surface light comes in.
// The distance is arithmetic, not taste. An 8-unit board at azimuth 0.78 presents
// 8·(|cos a| + |sin a|) = 11.3 units across; at this pitch each row costs
// `1·cos45°·sin(elev) + 0.5·cos(elev)` ≈ 0.84 units of frame height, so eight rows plus
// the near shelf's 1.35-unit cliff need ~8.1. A 40° vertical FOV covers 2·d·tan20° =
// 0.728·d, so d must clear ~11.2 for the reef alone; 17.6 also keeps the water column
// (where the god rays live) above it and the clam clear of the frame edge.
const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.82;
const CAM_DISTANCE = 17.6;
const CAM_TARGET: readonly [number, number, number] = [-0.85, -2.08, 1.05];
const CAM_YAW_RANGE = 0.55;
/** thumb pulls back and stops moving: a square-ish crop must still hold the whole reef */
const CAM_DISTANCE_THUMB = 21.0;
const CAM_AZIMUTH_THUMB = 0.74;
const CAM_POLAR_THUMB = 0.82;

// Celebration weights. The reef is an 8-unit field in a dark palette, so the kit's
// full-strength grammar (a 3.2-unit ripple and a 1.0 bloom pulse) floods three terraces
// at once and the staircase stops reading. Two thirds of the amplitude, all of the beat.
/** the burst travels ~0.7 world units — a little under one cell */
const BURST_SCALE = 0.78;
const RIPPLE_SCALE = 0.44;
const BURST_PARTICLES = 96;
const PULSE_STRENGTH = 0.52;
/** the pearl's flash: short and bright */
const FLARE_SECONDS = 1.5;
/**
 * How long the clam holds its gape after the treasure is taken. Deliberately long: a
 * successful dive is ~3% of dives, and the clam gaping for a beat afterwards is the only
 * lasting evidence in the frame that one just happened.
 */
const GAPE_SECONDS = 2.8;
/** the bubble column a resurfacing diver leaves behind */
const RUSH_SECONDS = 0.85;
/** centre of the rush column, so it spans the water above the reef */
const RUSH_CENTER_Y = 0.4;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
/** the diver swims DOWN a terrace each step; a tall arc would fight the descent */
const HOP_HEIGHT = 0.06;

// the clam's idle life: a resting gape that breathes, opening as the dive closes in
const GAPE_BASE = 0.3;
const GAPE_BREATH = 0.12;
const GAPE_PROXIMITY = 0.52;
/** how many columns from the treasure the clam starts to notice the diver */
const PROXIMITY_COLS = 3;

class SunlitDescentScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private bubble: BubbleHandle | null = null;
  private clam: ClamHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  /** the node that carries the diver DOWN the staircase (see `update` step 6) */
  private readonly diver = new THREE.Group();
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });
  private reach: Uint8Array = new Uint8Array(0);
  private treasure: GridCell = { x: 0, y: 0 };

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private goalAt = -Infinity;
  private resurfaceAt = -Infinity;
  /** eased stir of silt: the frame says 0 or 1, the water takes a moment to agree */
  private stir = 0;

  // ---- preallocated scratch: update() may not allocate ----
  private readonly dive: DiveCell = { r: 0, c: 0, lastA: 0 };
  private readonly overlayDive: DiveCell = { r: 0, c: 0, lastA: 0 };
  private readonly overlayCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly diverPos = new THREE.Vector3();
  private pearlHue = 0xffd166;
  private gold = 0xffd166;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'sunlitDescentScene';

    const grid = resolveGrid(ctx.geometry, ctx.seed);
    this.grid = grid;
    // the reef, derived from the decoder's own transition — never a typed-in triangle
    this.reach = reefReachable(grid);
    const goal = treasureCell(grid);
    this.treasure = { x: goal.c, y: goal.r };

    const palette = ctx.tokens.palettes.underwater;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.uiCyan;
    this.gold = ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid, this.reach);
    this.world = world;
    this.scene.add(world.group);
    this.pearlHue = world.skin.pearl.getHex();

    // This scene claims three of the rig's point lights. Their COLORS are static — only
    // position and intensity are state-driven — so update() never touches a Color.
    const pearlLight = world.rig.points[0];
    if (pearlLight) pearlLight.color.set(world.skin.pearl);
    const diverLight = world.rig.points[1];
    if (diverLight) diverLight.color.set(accent);
    const bloomLight = world.rig.points[2];
    if (bloomLight) bloomLight.color.set(world.skin.pearl);

    // ---- the shared avatar, riding the terraces ----
    // The wisp lives under `diver`, a node whose y IS the reef's depth map. Everything
    // about the wisp stays in the kit's flat grid space (HopMotion has no idea the world
    // has a y axis); this one node turns that flat glide into a descent, and because the
    // depth map is affine in z it stays exact at every landed cell.
    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        // a shade over the 0.16 default: from the composed 14.2-unit standoff, inside a
        // bubble, in a dark blue column, the default orb is a dim speck
        radius: 0.19,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        // The trail lives in the wisp's OWN local space, and this scene moves that space
        // vertically every frame to ride the staircase — a ribbon would smear along the
        // current terrace instead of trailing back up the dive. The bubble wake and the
        // stirred silt do the job honestly instead.
        ribbon: false,
        name: 'diverWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.diver.name = 'diverColumn';
    this.diver.add(wisp.group);
    this.scene.add(this.diver);

    // ---- the bubble the diver rides: a child of the exact avatar node ----
    const bubble = buildBubble({
      tier: ctx.tier,
      palette,
      skin: world.skin,
      accent,
      height: ctx.tokens.wispHeight,
      name: 'diverBubble',
    });
    this.bubble = bubble;
    wisp.avatar.add(bubble.group);

    // ---- the treasure: a giant clam on the env's own goal cell ----
    const clam = buildClam({
      tier: ctx.tier,
      palette,
      skin: world.skin,
      // A clam gapes toward the open water and the light, which here is also the side the
      // frame watches from — so the mantle and the pearl are visible instead of a shell back.
      facing: Math.PI * 0.26,
      name: 'giantClam',
    });
    this.clam = clam;
    cellToReefVec(this.treasure.x, this.treasure.y, grid, clam.root.position);
    this.scene.add(clam.group);

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [CAM_TARGET[0], CAM_TARGET[1], CAM_TARGET[2]],
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
      name: 'reefOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'reefCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
    // The overlay draws in GRID space, flat, as it must. This scene's grid→world map is a
    // shear (`reefShear`), so handing the overlay group that exact matrix carries the
    // lattice, the mask arrows, the agent cross and the goal diamond down the staircase
    // with the art: every cell-centred marker lands on its own terrace, and x and z are
    // untouched. If the art and the overlay ever disagree, the art is wrong.
    const overlayDive = this.overlayDive;
    const overlayCell = this.overlayCell;
    const overlayGrid: GridSize = grid;
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        readDive(state, overlayGrid, overlayDive);
        overlayCell.x = overlayDive.c;
        overlayCell.y = overlayDive.r;
        return overlayCell;
      },
      goalCell: () => this.treasure,
      name: 'reefGroundTruth',
    });
    reefShear(grid, this.overlay.group.matrix);
    this.overlay.group.matrixAutoUpdate = false;
    this.overlay.group.matrixWorldNeedsUpdate = true;
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    this.diver.position.y = reefDepthYAt(wisp.position.z, grid);
    bubble.setTime(0);
    bubble.setStress(0);
    clam.setState(GAPE_BASE, 0);
    clam.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const bubble = this.bubble;
    const clam = this.clam;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !bubble || !clam || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const dive = this.dive;

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. the frame's truth: row FIRST, then column, then the action that got us here
    readDive(state, grid, dive);

    // 3. a scrub wipes every in-flight beat: nothing may survive a discontinuity
    if (frame.scrubbed) {
      pool.clear();
      this.goalAt = -Infinity;
      this.resurfaceAt = -Infinity;
      this.stir = dive.lastA;
      wisp.setCell(dive.c, dive.r, grid);
      wisp.snap();
      rig.snap();
    }

    // 4. events — celebrations are NEVER inferred from the state stream
    let hardCut = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        // The treasure cell never moves and the diver has already been swept back to the
        // surface by the time this arrives, so the bloom belongs to the PEARL.
        cellToReefVec(this.treasure.x, this.treasure.y, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y += clam.pearlHeight;
        pool.burst(this.bloomPos, this.pearlHue, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(0.85);
        this.goalAt = time;
      } else if (ev.type === 'resurface') {
        // the column of bubbles is left at the cell the diver LEFT, captured before the
        // snap moves it back to the surface
        world.rush.points.position.set(wisp.position.x, RUSH_CENTER_Y, wisp.position.z);
        this.resurfaceAt = time;
        hardCut = true;
      } else if (ev.type === 'teleport') {
        hardCut = true;
      }
      // DeepSea emits no `done`: this scene must never sigh.
    }

    // 5. the avatar. One straight hop per dive step — and a HARD CUT for anything that is
    //    not one, which is exactly the wrap: the env never carried the diver across six
    //    terraces, so the art must not either.
    if (hardCut || !isDiveStep(wisp.cellX, wisp.cellY, dive)) {
      if (hardCut || dive.c !== wisp.cellX || dive.r !== wisp.cellY) {
        wisp.setCell(dive.c, dive.r, grid);
        wisp.snap();
      }
    } else {
      wisp.hopTo({ x: dive.c, y: dive.r }, time);
    }
    // `wisp.face()` is deliberately NOT called — see the probe note in `probes()`.
    wisp.update(frame.dtReal, time);

    // 6. the descent. `HopMotion` glides in the flat grid plane; this one line turns that
    //    plane into the staircase. At a landed cell `reefDepthYAt(z)` is bit-identical to
    //    `reefDepthY(row)`, which is why the alignment probes read exactly 0 error.
    this.diver.position.y = reefDepthYAt(wisp.position.z, grid);
    this.diverPos.set(wisp.position.x, wisp.position.y + this.diver.position.y, wisp.position.z);

    // 7. the bubble and the silt: pure reads of `ctrl[0] & 1`, the action the env charges
    //    its -0.01/N right-move cost for, eased in WALL time like any ambient value.
    const target = dive.lastA === 1 ? 1 : 0;
    this.stir += (target - this.stir) * (1 - Math.exp(-9 * Math.max(0, frame.dtReal)));
    bubble.setStress(this.stir);
    bubble.setTime(time);
    world.silt.points.position.set(
      this.diverPos.x,
      this.diverPos.y + ctx.tokens.wispHeight * 0.6,
      this.diverPos.z,
    );
    world.silt.setDensity(0.05 + 0.75 * this.stir);

    // 8. the resurfacing column
    const rush = envelope(time - this.resurfaceAt, RUSH_SECONDS, 2.6);
    world.rush.setDensity(rush);
    world.rush.setIntensity(1.1 + 1.6 * rush);

    // 9. the clam: gape is a pure function of (time, state), flare is a pure function of
    //    the last goal event's clock. Nothing here integrates or remembers a position.
    const flare = envelope(time - this.goalAt, FLARE_SECONDS, 2.2);
    const held = envelope(time - this.goalAt, GAPE_SECONDS, 1.1);
    const open = clamp01(gape(time, dive, grid) + 0.8 * held);
    clam.setState(open, flare);
    clam.setTime(time);

    // 10. the three point lights this scene claims from the rig
    const pearlLight = world.rig.points[0];
    if (pearlLight) {
      pearlLight.position.set(
        clam.root.position.x,
        clam.root.position.y + clam.pearlHeight,
        clam.root.position.z,
      );
      pearlLight.intensity = 0.16 + 0.5 * open;
    }
    const diverLight = world.rig.points[1];
    if (diverLight) {
      diverLight.position.set(
        this.diverPos.x,
        this.diverPos.y + ctx.tokens.wispHeight,
        this.diverPos.z,
      );
      diverLight.intensity = 0.5 + 0.7 * rush;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 3.2 * flare;
    }

    // 11. one clock for the whole reef
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
    const clam = this.clam;
    const world = this.world;
    const height = this.ctx?.tokens.wispHeight ?? TOKENS.wispHeight;
    const treasure = this.treasure;

    const orbDive: DiveCell = { r: 0, c: 0, lastA: 0 };
    const orbOut: [number, number, number] = [0, 0, 0];
    const anchorDive: DiveCell = { r: 0, c: 0, lastA: 0 };
    const anchorOut: [number, number, number] = [0, 0, 0];
    const siltDive: DiveCell = { r: 0, c: 0, lastA: 0 };
    const siltOut: [number, number, number] = [0, 0, 0];
    const clamOut: [number, number, number] = [0, 0, 0];

    // THE PROBE RULE FOR EVERY GRID SCENE: `assertUpdateIdempotent` compares probe
    // objects' whole WORLD MATRIX, so a probe may carry no eased value of any kind.
    // `Wisp.face()` damps the body yaw over `dt` and `wisp.probeObject` (the orb core)
    // hangs off that body — this scene never calls `face()` (a diver in a bubble has no
    // heading to show), so both wisp anchors stay pure and probeable.
    const probes: AlignmentProbe[] = [
      {
        // the orb centre: exactly `wispHeight` over the true cell's SHELF
        name: 'wisp',
        object: wisp?.probeObject ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readDive(state, grid, orbDive);
          cellToReefWorld(orbDive.c, orbDive.r, grid, orbOut);
          orbOut[1] += height;
          return orbOut;
        },
      },
      {
        // and the avatar node under it: the cell itself, on its terrace, position only
        name: 'wisp-anchor',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readDive(state, grid, anchorDive);
          return cellToReefWorld(anchorDive.c, anchorDive.r, grid, anchorOut);
        },
      },
      {
        // the treasure: the env's OWN goal cell, on the deepest terrace, forever
        name: 'clam',
        object: clam?.root ?? new THREE.Object3D(),
        expectWorldPos: () => cellToReefWorld(treasure.x, treasure.y, grid, clamOut),
      },
    ];

    if (world) {
      probes.push({
        // the stirred silt rides the diver exactly — it is a state-bearing object, not
        // decoration, because WHERE it is is the answer to "where is the diver"
        name: 'silt',
        object: world.silt.points,
        expectWorldPos: (state) => {
          readDive(state, grid, siltDive);
          cellToReefWorld(siltDive.c, siltDive.r, grid, siltOut);
          siltOut[1] += height * 0.6;
          return siltOut;
        },
      });
    }
    return probes;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.clam?.dispose();
    this.clam = null;
    // the bubble hangs off the wisp's avatar node, so it must be released FIRST: the
    // wisp's own dispose() clears its group and would leave the bubble's buffers orphaned
    this.bubble?.dispose();
    this.bubble = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.diver.removeFromParent();
    this.diver.clear();
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.goalAt = -Infinity;
    this.resurfaceAt = -Infinity;
    this.stir = 0;
  }
}

// ---------------------------------------------------------------------------
// envelopes — every one of them measured on the playback clock
// ---------------------------------------------------------------------------

/**
 * How wide the giant clam stands open: a slow swell, plus a term that rises as the dive
 * gets DEEP and CLOSE to the treasure column. Both are pure functions of `(time, state)`
 * — no history, no easing — so the clam is identical on a scrubbed frame and a played
 * one. It shows nothing the agent does not have: the row and the column are the two
 * numbers the frame is made of.
 */
export function gape(time: number, dive: DiveCell, grid: GridSize): number {
  const depth = grid.h > 1 ? clamp01(dive.r / (grid.h - 1)) : 0;
  const away = Math.abs(dive.c - (grid.w - 1));
  const near = away >= PROXIMITY_COLS ? 0 : 1 - away / PROXIMITY_COLS;
  const breath = 0.5 + 0.5 * Math.sin(time * 0.55);
  return clamp01(GAPE_BASE + GAPE_BREATH * breath + GAPE_PROXIMITY * depth * near);
}

const createScene: SceneFactory = () => new SunlitDescentScene();
export default createScene;
