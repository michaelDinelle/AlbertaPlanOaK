// Temple of Three Gates — MultiKeyCorridor-cont-v0.
//
// THE ONE IDEA: this env is a PROCESSION, so the scene is a hall. Four chambers walk
// from dewy morning blue-green at the mouth to gold in the sanctum; three gates stand
// between them and each one is a different problem — a bramble laced across the way, a
// ford that has to be stoned, a mist that has to be parted. Where the wisp has got to,
// what it is carrying and which gates have given way are all answerable from one glance
// at the frame, which is the whole job of an art pass over a state stream.
//
// WHAT THIS SCENE ADDS TO THE GRID CANON (four-rooms), and why:
//
//  * THE KEY CELLS ARE REPLAYED, NOT RECORDED. The 9-float record row carries the
//    agent, the goal and the two bitmasks — the key positions are nowhere in it. The
//    decoder recovers them by re-running the exact `gw_mkc_step` dynamics over the
//    ctrl stream, so `state.keys[j]` is ground truth and this scene reads it and NEVER
//    guesses. (`DecodedRecording.replay` reports the health of that replay: on all
//    eight demo lanes it is 0 disagreements over 20k steps.) The pose fallback below
//    can rebuild everything EXCEPT the key cells — for those it falls back to the env's
//    opening positions, which is the honest best a poseless frame allows.
//
//  * THE +1 DOES NOT HAPPEN WHERE THE AGENT IS. Canon lesson 18 says the goal bloom
//    belongs at the agent's cell, because in FourRooms the agent is standing on the old
//    goal when it scores. Here the agent scores AND IS THROWN BACK to chamber 0 in the
//    same step (that is what the `teleport` in the same event batch means), so its cell
//    on the goal frame is four chambers away from where the reward happened. The
//    celebration is read from `frame.prev`'s GOAL cell — the relic that was just
//    reached — and the wisp hard-cuts to its respawn cell. Measured on seed 80 frame
//    3094: prev goal (15,1), prev agent (14,1), new agent (2,4). Blooming the agent's
//    cell would fire the whole grammar at the wrong end of the temple.
//
//  * MID-TIER EVENTS ARE REAL HERE. `key` and `door` fire ~44 and ~85 times per 20k
//    steps on one lane, so they get STYLE.md's mid-tier grammar exactly: a reduced
//    burst, no ripple, and the gate/altar's own flare. Only the +1 gets all three beats.
//
//  * THERE IS NO FAILURE. MKC's `done` is emitted only alongside a `goal`, so this
//    scene never sighs. The cool-blue grammar would be a lie about the stream.
//
//  * GATE STATE IS A PURE READ — OF THE FRAME AND THE PLAYBACK CLOCK, NOTHING ELSE.
//    Each gate is posed every frame by `gatePhase(state.doorOpen[j], time - doorAt[j])`:
//    the frame's own bit says WHERE the mechanism is headed, and the age of that gate's
//    last door event — measured on `ctx.time()`, the same clock the wisp hops on — says
//    how far along it is. The gates therefore freeze when the transport pauses, run at
//    the transport's speed, and land exactly on the frame's bit whenever the clock is
//    settled, which makes a scrub and a respawn hard cuts by construction rather than by
//    a snap flag. (An earlier pass eased them on `frame.dtReal`; a paused frame then
//    showed a wisp stopped in mid-hop while the gates kept swinging, and the mechanism
//    was history rather than a function of state.)

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
import type { GridCell, GridGeometry } from '@sim/types';
import * as THREE from 'three';
import {
  GATE_FLARE_SECONDS,
  GATE_SECONDS,
  buildBrambleGate,
  buildCarryRack,
  buildKeyAltar,
  buildMistGate,
  buildRiverGate,
  buildShrine,
  buildThumbGateBank,
  type CarryRackHandle,
  type GateHandle,
  type KeyAltarHandle,
  type ShrineHandle,
  type ThumbGateBank,
} from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  buildWorld,
  chamberBounds,
  gateFacesX,
  resolveDoors,
  resolveGoal,
  resolveGrid,
  resolveKeys,
  templeSkin,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. A high 3/4 aerial from the south-south-east. The temple is 17 × 7 — a
// hall, not a courtyard — so the camera sits nearly square onto it and lets the
// procession run across the frame.
//
// The distance is arithmetic, not taste (grid canon lesson 21): seen from azimuth a, a
// 17 × 7 playfield presents 17·|cos a| + 7·|sin a| across the frame — 17.8 units at
// a = 0.14 — and a 40° vertical FOV at 16:9 covers 2·d·tan(32.9°) horizontally, so the
// bare fit is d = 13.7 and 15.5 is the margin that makes it a composition, not a crop.
//
// THE POLAR IS NOT TASTE EITHER, and it cost a whole review pass to find out. Every
// gate cell has a wall cell immediately SOUTH of it — that is what a door in a wall
// means — and at 50° of elevation a 0.74 wall half a cell away hides everything below
// y = 0.33 at the gate's own centre: the ford, its stepping stones and the mist pool
// were all rendering perfectly, into a sliver the camera could not see. Clearing a
// wall of height h at half a cell needs tan(elevation) > 2h, i.e. > 55.9° for these
// walls; 0.58 rad of polar puts the gate row at 57.4° with the wall shortened to 0.70,
// and the whole gate cell — mechanism included — comes back into frame.
const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.58;
const CAM_DISTANCE = 15.5;
const CAM_TARGET_Y = 0.35;
const CAM_YAW_RANGE = 0.55;
/** thumb pulls back and stops moving: the card may be squarer than 16:9 */
const CAM_DISTANCE_THUMB = 20.0;
const CAM_AZIMUTH_THUMB = 0.12;
const CAM_POLAR_THUMB = 0.56;

// Celebration weights. The temple is 17 units of morning stone at a 15.5-unit standoff:
// the kit's full-strength grammar (a 3.2-unit ripple, a 1.0 pulse) whites out a third of
// the hall, which is exactly the readability this scene is built on. Canon lesson 6 —
// keep all three beats, halve every weight.
const BURST_SCALE = 0.85;
const RIPPLE_SCALE = 0.55;
const BURST_PARTICLES = 120;
const PULSE_STRENGTH = 0.5;
/** the +1 blooms at the relic's crown, not on the flagstones */
const BLOOM_Y = 0.62;
const BLOOM_SECONDS = 1.5;
/** mid-tier grammar: a reduced burst and no ripple (STYLE.md) */
const KEY_BURST_SCALE = 0.34;
const KEY_BURST_Y = 0.42;
const DOOR_BURST_SCALE = 0.42;
const DOOR_BURST_Y = 0.6;
const MID_SECONDS = 0.9;
/** a respawned key / relic takes this long to grow in (never a slide from the old cell) */
const GROW_SECONDS = 0.72;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.17;

/** how close the wisp must be before the relic opens up (Chebyshev cells) */
const PROXIMITY_CELLS = 5;

const KEYS = 3;

/** Everything this scene draws, read out of one frame. Preallocated — never allocated. */
interface TempleView {
  x: number;
  y: number;
  gx: number;
  gy: number;
  hasKey: boolean[];
  doorOpen: boolean[];
  keys: GridCell[];
}

function makeView(): TempleView {
  return {
    x: 0,
    y: 0,
    gx: 0,
    gy: 0,
    hasKey: [false, false, false],
    doorOpen: [false, false, false],
    // filled from the ENV's opening layout in init(); the live cells arrive with the
    // first frame's replayed `derived.keys`
    keys: [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 0, y: 0 },
    ],
  };
}

/**
 * Seed a view's key cells from the geometry the host resolved. A poseless frame keeps
 * the last cells it knew, so the temple has to OPEN on the env's own layout — and that
 * layout is read from `grid.keys`, never written into the scene.
 */
function seedKeys(view: TempleView, grid: GridGeometry): void {
  const keys = resolveKeys(grid);
  for (let j = 0; j < KEYS; j++) {
    const cell = view.keys[j] as GridCell;
    const source = keys[j];
    if (!source) continue;
    cell.x = source.x;
    cell.y = source.y;
    clampCell(cell, grid);
  }
}

interface MkcLike {
  x?: unknown;
  y?: unknown;
  gx?: unknown;
  gy?: unknown;
  hasKey?: unknown;
  doorOpen?: unknown;
  keys?: unknown;
}

function bit(mask: number, i: number): boolean {
  return ((mask >> i) & 1) !== 0;
}

function cellOf(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : fallback;
}

/**
 * The frame's truth. sim-core's decoded state is the source; the raw pose
 * (`[x, y, gx, gy, keyMask, doorMask, …]`) rebuilds everything except the key CELLS,
 * which live only in the replay — a poseless frame keeps the last cells it knew, which
 * for the fixture loop is the env's own opening layout.
 */
function readState(state: FrameState, grid: GridSize, out: TempleView): void {
  const d = state.derived as MkcLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  out.x = cellOf(has ? d.x : undefined, Math.round(state.pose[0] ?? 0));
  out.y = cellOf(has ? d.y : undefined, Math.round(state.pose[1] ?? 0));
  out.gx = cellOf(has ? d.gx : undefined, Math.round(state.pose[2] ?? 0));
  out.gy = cellOf(has ? d.gy : undefined, Math.round(state.pose[3] ?? 0));

  const keyMask = Math.round(state.pose[4] ?? 0);
  const doorMask = Math.round(state.pose[5] ?? 0);
  const hasKey = has && Array.isArray(d.hasKey) ? (d.hasKey as unknown[]) : null;
  const doorOpen = has && Array.isArray(d.doorOpen) ? (d.doorOpen as unknown[]) : null;
  for (let j = 0; j < KEYS; j++) {
    out.hasKey[j] = hasKey ? hasKey[j] === true : bit(keyMask, j);
    out.doorOpen[j] = doorOpen ? doorOpen[j] === true : bit(doorMask, j);
  }

  const keys = has && Array.isArray(d.keys) ? (d.keys as unknown[]) : null;
  for (let j = 0; j < KEYS; j++) {
    const cell = out.keys[j] as GridCell;
    const source = keys ? (keys[j] as { x?: unknown; y?: unknown } | undefined) : undefined;
    if (source) {
      cell.x = cellOf(source.x, cell.x);
      cell.y = cellOf(source.y, cell.y);
    }
    clampCell(cell, grid);
  }

  out.x = clampInt(out.x, grid.w);
  out.y = clampInt(out.y, grid.h);
  out.gx = clampInt(out.gx, grid.w);
  out.gy = clampInt(out.gy, grid.h);
}

/** clamp a cell coordinate into [0, extent - 1] (the scalar twin of clampCell) */
function clampInt(v: number, extent: number): number {
  return v < 0 ? 0 : v > extent - 1 ? extent - 1 : v;
}

/** The goal cell of a frame — used on the +1 to find the relic that was just reached. */
export function readGoalCell(state: FrameState, grid: GridSize, out: GridCell): void {
  const d = state.derived as MkcLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  out.x = cellOf(has ? d.gx : undefined, Math.round(state.pose[2] ?? 0));
  out.y = cellOf(has ? d.gy : undefined, Math.round(state.pose[3] ?? 0));
  clampCell(out, grid);
}

class TempleOfThreeGatesScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private gates: GateHandle[] = [];
  /** thumb only: the three gates collapsed into one instanced doorway */
  private gateBank: ThumbGateBank | null = null;
  private altars: KeyAltarHandle[] = [];
  private rack: CarryRackHandle | null = null;
  private shrine: ShrineHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });
  private doors: readonly GridCell[] = [];

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private growAt = -Infinity;
  private bloomAt = -Infinity;
  private midAt = -Infinity;
  /** one per gate: when this gate's door event landed. -Infinity = settled on its bit. */
  private readonly doorAt: number[] = [-Infinity, -Infinity, -Infinity];

  // ---- preallocated scratch: update() may not allocate ----
  private readonly view = makeView();
  private readonly probeView = makeView();
  private readonly prevGoal: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly eventPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private keyColors: THREE.Color[] = [];
  private relicColor = new THREE.Color(0xffd166);
  private accentColor = new THREE.Color(0xd9b36a);

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'templeOfThreeGatesScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    const doors = resolveDoors(grid);
    this.doors = doors;
    // the opening layout comes from the resolved geometry — the scene owns no cell
    seedKeys(this.view, grid);
    seedKeys(this.probeView, grid);
    const openingGoal = resolveGoal(grid);

    const palette = ctx.tokens.palettes.morning;
    const skin = templeSkin(palette, ctx.tokens);
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.accentColor = new THREE.Color(accent);
    this.keyColors = skin.keys.map((c) => c.clone());
    this.relicColor = skin.relic.clone();

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    // ---- the three gates, in the env's own door order ----
    // At hero and medium these are three separate mechanisms — the three chapters of the
    // env. At thumb they are ONE instanced doorway (`buildThumbGateBank`): seven of the
    // forty draw calls STYLE.md gives the whole wall-view frame were being spent on three
    // ~10 px thresholds. Same `GateHandle` contract either way, so nothing downstream —
    // update, probes, celebrations — knows which one it is holding.
    const gateCount = Math.min(doors.length, 3);
    const builders = [buildBrambleGate, buildRiverGate, buildMistGate];
    const bank =
      ctx.tier === 'thumb'
        ? buildThumbGateBank({
            tier: ctx.tier,
            skin,
            facesX: doors.slice(0, gateCount).map((cell) => gateFacesX(grid, cell)),
            count: gateCount,
            name: 'thumbGateBank',
          })
        : null;
    this.gateBank = bank;
    if (bank) this.scene.add(bank.group);
    for (let j = 0; j < gateCount; j++) {
      const cell = doors[j] as GridCell;
      const build = builders[j] as (typeof builders)[number];
      const gate =
        bank?.gates[j] ??
        build({
          tier: ctx.tier,
          palette,
          skin,
          facesX: gateFacesX(grid, cell),
          stone: world.stoneMaterial,
          name: `gate${j}`,
        });
      cellToWorldVec(cell.x, cell.y, grid, gate.root.position);
      // the bank poses every leaf from its root, so the leaf follows the cell it was
      // just placed on rather than sitting at the origin until the first door event
      gate.setProgress(gate.progress);
      this.scene.add(gate.group);
      this.gates.push(gate);
    }

    // ---- the three key altars ----
    for (let j = 0; j < KEYS; j++) {
      const altar = buildKeyAltar({
        tier: ctx.tier,
        palette,
        skin,
        index: j,
        stone: world.stoneMaterial,
        name: `keyAltar${j}`,
      });
      const cell = this.view.keys[j] as GridCell;
      cellToWorldVec(cell.x, cell.y, grid, altar.root.position);
      this.scene.add(altar.group);
      this.altars.push(altar);
    }

    // ---- the shrine ----
    const shrine = buildShrine({ tier: ctx.tier, palette, skin, stone: world.stoneMaterial });
    this.shrine = shrine;
    cellToWorldVec(openingGoal.x, openingGoal.y, grid, shrine.root.position);
    this.scene.add(shrine.group);

    // ---- the shared avatar, carrying its rack of keys ----
    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        // Well over the 0.16 default. This hall is 17 units wide at a 15.5-unit
        // standoff — nearly twice the ground four-rooms covers — so the default orb is
        // ~12 px of pale gold on pale stone and the eye simply loses it. At 0.26 the
        // wisp is a quarter of its cell and unmistakable from the composed frame.
        radius: 0.26,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'templeWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const rack = buildCarryRack(skin, ctx.tier);
    this.rack = rack;
    wisp.carry(rack.group);

    // This scene claims all four of the rig's point lights. Their colors are set here
    // (and, for the event light, once per event) — update() never allocates one.
    const relicLight = world.rig.points[0];
    if (relicLight) relicLight.color.copy(this.relicColor);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.copy(this.accentColor);
    const eventLight = world.rig.points[2];
    if (eventLight) eventLight.color.set(ctx.tokens.celebrationGold);
    const sanctumLight = world.rig.points[3];
    if (sanctumLight) {
      // a static warm fill in the sanctum: the east end of a 17-unit hall needs its own
      // depth or the gold flattens into a painted floor
      const b = chamberBounds(grid, 3);
      sanctumLight.color.set(skin.chambers[3]?.bloom ?? accent);
      sanctumLight.position.set(b.cx, 1.9, b.cz);
      sanctumLight.intensity = 0.85;
    }

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [0, CAM_TARGET_Y, 0],
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: thumb ? CAM_AZIMUTH_THUMB : CAM_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      limits: {
        minPolar: 0.42,
        maxPolar: 1.0,
        minAzimuth: CAM_AZIMUTH - CAM_YAW_RANGE,
        maxAzimuth: CAM_AZIMUTH + CAM_YAW_RANGE,
        // close enough to lean into one chamber, never close enough to lose the hall
        minDistance: 10,
        maxDistance: 26,
      },
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0012,
      name: 'templeOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'templeCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
    // Every resolver is explicit: the overlay draws the keys the REPLAY found, not the
    // opening layout its `derived` sniffer would otherwise fall back to.
    const overlayView = makeView();
    seedKeys(overlayView, grid);
    const readOverlay = (state: FrameState): TempleView => {
      readState(state, grid, overlayView);
      return overlayView;
    };
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        const v = readOverlay(state);
        return { x: v.x, y: v.y };
      },
      goalCell: (state) => {
        const v = readOverlay(state);
        return { x: v.gx, y: v.gy };
      },
      keyCells: (state) => {
        const v = readOverlay(state);
        return v.keys;
      },
      doorStates: (state) => readOverlay(state).doorOpen,
      maxKeys: KEYS,
      name: 'templeGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    for (const gate of this.gates) {
      // shut, settled, unlit: the temple opens on the env's own locked layout
      gate.setProgress(0);
      gate.setFlare(0);
      gate.setTime(0);
    }
    for (const altar of this.altars) {
      altar.setGrow(1);
      altar.setCarried(false);
      altar.setTime(0);
    }
    rack.setTime(0);
    shrine.setState(0, 1);
    shrine.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const shrine = this.shrine;
    const rack = this.rack;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !shrine || !rack || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const view = this.view;

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. the frame's truth, snapped to cells
    readState(state, grid, view);

    // 3. a scrub wipes every in-flight beat: nothing may survive a discontinuity
    if (frame.scrubbed) {
      pool.clear();
      this.bloomAt = -Infinity;
      this.midAt = -Infinity;
      this.growAt = -Infinity;
      // a settled gate clock IS the hard snap: step 7 then reads the frame's bit exactly
      for (let j = 0; j < KEYS; j++) this.doorAt[j] = -Infinity;
      wisp.setCell(view.x, view.y, grid);
      wisp.snap();
      rig.snap();
    }

    // 4. events — celebrations are NEVER inferred from the state stream
    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        // THE +1 IS NOT WHERE THE AGENT IS. The agent has already been thrown back to
        // chamber 0 by the same step; the relic it reached is the goal cell of the
        // PREVIOUS frame. (`prev` is null only on a non-sequential frame, which carries
        // no events — the fallback is for contract completeness.)
        if (frame.prev) readGoalCell(frame.prev, grid, this.prevGoal);
        else {
          this.prevGoal.x = view.gx;
          this.prevGoal.y = view.gy;
        }
        cellToWorldVec(this.prevGoal.x, this.prevGoal.y, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.relicColor, BURST_SCALE, time);
        pool.ripple(this.groundPos, ctx.tokens.celebrationGold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        shrine.flare(1);
        this.bloomAt = time;
        this.growAt = time;
        const light = world.rig.points[2];
        if (light) light.color.copy(this.relicColor);
      } else if (ev.type === 'key') {
        // auto-pickup is co-location, so the key's own cell IS where it happened — and
        // `state.keys[j]` still holds it (the array only changes on a respawn)
        const cell = view.keys[ev.index];
        if (cell) {
          cellToWorldVec(cell.x, cell.y, grid, this.eventPos);
          this.eventPos.y = KEY_BURST_Y;
          pool.burst(this.eventPos, this.keyColors[ev.index] ?? this.accentColor, KEY_BURST_SCALE, time);
          this.midAt = time;
          const light = world.rig.points[2];
          if (light) light.color.copy(this.keyColors[ev.index] ?? this.accentColor);
        }
        wisp.flare(0.55);
      } else if (ev.type === 'door') {
        // THE GATE'S CLOCK. Both directions start it — an opening swings 0 → 1 over
        // GATE_SECONDS of playback time, a relock without a respawn runs 1 → 0 — and
        // step 7 reads the direction off the frame's own bit, so the clock never has to
        // remember which way the gate was going.
        if (ev.index >= 0 && ev.index < KEYS) this.doorAt[ev.index] = time;
        // opening is a mid-tier beat; the relock that comes with a respawn is silent —
        // the teleport below snaps all three gates shut in the same frame
        const cell = this.doors[ev.index];
        if (ev.open && cell) {
          cellToWorldVec(cell.x, cell.y, grid, this.eventPos);
          this.eventPos.y = DOOR_BURST_Y;
          pool.burst(this.eventPos, this.keyColors[ev.index] ?? this.accentColor, DOOR_BURST_SCALE, time);
          this.midAt = time;
          const light = world.rig.points[2];
          if (light) light.color.copy(this.keyColors[ev.index] ?? this.accentColor);
        }
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
      // MKC emits `done` only alongside `goal` (a +1, never a failure), so this scene
      // has no sigh. The cool-blue grammar would be a lie about the stream.
    }

    // 5. the respawn: a hard cut, then everything grows back in on its new cell
    if (teleported) {
      wisp.setCell(view.x, view.y, grid);
      wisp.snap();
      // the relock is instant: settling the clocks makes step 7 read the frame's shut
      // bits with no ease at all, in the teleport's own frame
      for (let j = 0; j < KEYS; j++) this.doorAt[j] = -Infinity;
      this.growAt = time;
    }

    // 6. the avatar: one straight hop per cell change, never a path plan.
    //    `wisp.face()` is deliberately NOT called — see the probe note in `probes()`.
    if (view.x !== wisp.cellX || view.y !== wisp.cellY) {
      wisp.hopTo(view, time);
    }
    wisp.update(frame.dtReal, time);

    // 7. the gates: pose and flare are both PURE READS of (the frame's bit, the playback
    //    clock). No dt, no remembered target, nothing to snap — a settled clock lands on
    //    the bit exactly, which is what a scrub and a respawn leave behind.
    for (let j = 0; j < this.gates.length; j++) {
      const gate = this.gates[j];
      if (!gate) continue;
      const open = view.doorOpen[j] === true;
      const age = time - (this.doorAt[j] ?? -Infinity);
      gate.setProgress(gatePhase(open, age));
      // only a gate that IS open can be glowing from having just given way; a relock
      // therefore goes dark in the same frame it starts closing
      gate.setFlare(open ? envelope(age, GATE_FLARE_SECONDS, 2.6) : 0);
      gate.setTime(time);
    }

    // 8. the keys: pedestal on the replayed cell, gem on the altar or on the wisp
    const grow = growth(time - this.growAt);
    for (let j = 0; j < this.altars.length; j++) {
      const altar = this.altars[j];
      const cell = view.keys[j];
      if (!altar || !cell) continue;
      cellToWorldVec(cell.x, cell.y, grid, altar.root.position);
      const carried = view.hasKey[j] === true;
      altar.setCarried(carried);
      altar.setGrow(grow);
      altar.setTime(time);
      altar.update(frame.dtReal, time);
      rack.setCarried(j, carried);
    }
    rack.setTime(time);

    // 9. the shrine: position is pure state, growth is pure event clock
    cellToWorldVec(view.gx, view.gy, grid, shrine.root.position);
    shrine.setState(proximity(view), grow);
    shrine.setTime(time);
    shrine.update(frame.dtReal, time);

    // 10. the point lights this scene claims from the rig
    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const mid = envelope(time - this.midAt, MID_SECONDS, 3.2);
    const relicLight = world.rig.points[0];
    if (relicLight) {
      relicLight.position.set(shrine.root.position.x, shrine.crownHeight + 0.1, shrine.root.position.z);
      relicLight.intensity = (0.18 + 0.5 * proximity(view)) * grow + 1.6 * bloom;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.16 + 0.7 * bloom + 0.35 * mid;
    }
    const eventLight = world.rig.points[2];
    if (eventLight) {
      // the flash rides whichever beat fired last: the relic on a +1, the key or the
      // gate on a mid-tier event
      if (bloom >= mid) eventLight.position.copy(this.bloomPos);
      else eventLight.position.copy(this.eventPos);
      eventLight.intensity = 2.4 * bloom + 1.5 * mid;
    }

    // 11. one clock for the whole temple
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
    const shrine = this.shrine;
    const world = this.world;
    const height = this.ctx?.tokens.wispHeight ?? TOKENS.wispHeight;
    const view = this.probeView;

    const wispOut: Vec3Tuple = [0, 0, 0];
    const anchorOut: Vec3Tuple = [0, 0, 0];
    const shrineOut: Vec3Tuple = [0, 0, 0];

    // THE PROBE RULE FOR EVERY GRID SCENE: `assertUpdateIdempotent` compares probe
    // objects' whole WORLD MATRIX, so a probe may carry no eased value of any kind —
    // position OR rotation OR scale. `Wisp.face()` damps a body yaw that `core` hangs
    // from, so a scene that turns its wisp must probe `wisp.avatar` instead. This
    // temple's wisp is a radially symmetric spirit with three keys already orbiting it;
    // facing buys it nothing, so it never calls `face()` and BOTH anchors stay exact.
    const probes: AlignmentProbe[] = [
      {
        // the orb centre: exactly `wispHeight` over the true cell — bob, breath, skirt,
        // ribbon and the carried keys all live on `decor`
        name: 'wisp',
        object: wisp?.probeObject ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readState(state, grid, view);
          cellToWorld(view.x, view.y, grid, wispOut);
          wispOut[1] = height;
          return wispOut;
        },
      },
      {
        // and the avatar node under it: the cell itself, y = 0, position only
        name: 'wisp-anchor',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readState(state, grid, view);
          return cellToWorld(view.x, view.y, grid, anchorOut);
        },
      },
      {
        // the relic — it must land on the RESPAWNED sanctum cell, every time
        name: 'shrine',
        object: shrine?.root ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readState(state, grid, view);
          return cellToWorld(view.gx, view.gy, grid, shrineOut);
        },
      },
    ];

    // the three key altars: each on the cell the DECODER'S REPLAY recovered, carried or
    // not (the pedestal never follows the key onto the wisp)
    for (let j = 0; j < this.altars.length; j++) {
      const altar = this.altars[j];
      if (!altar) continue;
      const out: Vec3Tuple = [0, 0, 0];
      probes.push({
        name: `key-altar-${j}`,
        object: altar.root,
        expectWorldPos: (state) => {
          readState(state, grid, view);
          const cell = view.keys[j] as GridCell;
          return cellToWorld(cell.x, cell.y, grid, out);
        },
      });
    }

    // the three gates and the three portals they hang in: fixed cells, read from the
    // env's own `grid.doors` — a gate that drifted off its cell would be a lie about
    // where the wall opens
    for (let j = 0; j < this.gates.length; j++) {
      const gate = this.gates[j];
      const cell = this.doors[j];
      if (!gate || !cell) continue;
      const out: Vec3Tuple = [0, 0, 0];
      probes.push({
        name: `gate-${j}`,
        object: gate.root,
        expectWorldPos: () => cellToWorld(cell.x, cell.y, grid, out),
      });
    }
    if (world) {
      for (let j = 0; j < world.portalRoots.length; j++) {
        const root = world.portalRoots[j];
        const cell = this.doors[j];
        if (!root || !cell) continue;
        const out: Vec3Tuple = [0, 0, 0];
        probes.push({
          name: `portal-${j}`,
          object: root,
          expectWorldPos: () => cellToWorld(cell.x, cell.y, grid, out),
        });
      }
    }

    return probes;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.shrine?.dispose();
    this.shrine = null;
    for (const gate of this.gates) gate.dispose();
    this.gates = [];
    this.gateBank?.dispose();
    this.gateBank = null;
    for (const altar of this.altars) altar.dispose();
    this.altars = [];
    this.wisp?.carry(null);
    this.rack?.dispose();
    this.rack = null;
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
    this.doors = [];
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
    this.midAt = -Infinity;
    for (let j = 0; j < KEYS; j++) this.doorAt[j] = -Infinity;
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
 * A GATE'S POSE, from the frame's ONE bit plus the age of that gate's last door event —
 * both of which the frame hands over. 0 = shut, 1 = fully open.
 *
 * The direction is READ, never remembered: with `open` true the ramp runs 0 → 1 as the
 * clock matures, and with it false the same ramp runs 1 → 0. A settled clock
 * (`-Infinity`, or any age past the window) returns the bit exactly, so a scrub, a
 * teleport and a relock all land on a fully-shut or fully-open gate and never on a
 * half-swung one — no snap flag required.
 *
 * The ramp is deliberately LINEAR: each mechanism applies its own smoothstep when it
 * poses itself, so the eye sees exactly the ease the gates always had, now measured on
 * `ctx.time()` instead of on wall seconds.
 */
export function gatePhase(open: boolean, age: number): number {
  const p = !(age >= 0) || age >= GATE_SECONDS ? 1 : age / GATE_SECONDS;
  return open ? p : 1 - p;
}

/**
 * The scale-in a freshly drawn key or relic plays. A never-fired clock (`-Infinity`)
 * and any age past the window both return 1, which is what makes a scrub land on a
 * settled temple instead of replaying a growth it never saw.
 */
export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.16 * Math.sin(Math.PI * t) * (1 - t));
}

/**
 * How near the wisp has come to the relic, in [0, 1] — a pure function of the frame,
 * with no history and no easing, so a scrubbed frame and a played one show the same
 * shrine. It shows nothing the agent does not have: its own cell and the goal's cell
 * are two of the six numbers the frame is made of.
 */
export function proximity(view: { x: number; y: number; gx: number; gy: number }): number {
  const d = Math.max(Math.abs(view.x - view.gx), Math.abs(view.y - view.gy));
  return d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
}

const createScene: SceneFactory = () => new TempleOfThreeGatesScene();
export default createScene;
