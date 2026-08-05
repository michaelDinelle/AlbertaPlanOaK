// The Ivy Gate — KeyCorridor-cont-v0.
//
// THE ONE IDEA: this env is a SENTENCE — take the seed, open the gate, reach the shrine —
// and the scene's whole job is to make the sentence readable at a glance from any frame.
// So the garden is split into a wild half and a formal half by the one wall the agent
// cannot cross, the amber seed lies in a stone dish on the wild side, the shrine burns on
// the formal side, and the single door between them is visibly SEWN SHUT with ivy that
// unpicks itself the moment `doorOpen` turns true and sews itself back the moment the
// shrine is reached.
//
// WHAT THIS SCENE ADDS TO THE GRID CANON (four-rooms is the pattern; these are the parts
// FourRooms never had to solve)
//
//  * THE AGENT REALLY TELEPORTS. Reaching the shrine respawns everything at once — wisp
//    to the wild garden, seed to a new dish, shrine to a new cell, gate re-locked — so the
//    `teleport` branch is load-bearing here, not contract completeness. Without it the
//    wisp would hop in a straight line straight THROUGH the wall it just came round.
//  * THE +1 DOES NOT HAPPEN WHERE THE AGENT IS. FourRooms' canon lesson 18 says the goal
//    bloom belongs to the agent's cell because its agent stays put. Here the agent is
//    already across the garden by the time we see the frame, and the cell that earned the
//    reward is `prev`'s GOAL cell — the shrine the wisp walked into. That is where the
//    petals go, and the test locks it.
//  * ONE BIT DRIVES A TWO-BEAT ANIMATION. `doorOpen` is a single boolean, but a door that
//    just pops open reads as a glitch. `actors.ts` sequences it: the ivy lets go, then the
//    leaf swings — and the same scalar run backwards closes the leaf before the ivy
//    re-wraps it. One scalar, no state machine, and a scrub lands on the settled pose.
//    The two DIRECTIONS get different windows, because the agent is only present for one
//    of them: the opening finishes inside the hop that carries the wisp through the
//    doorway, the relock takes its time behind the teleport (see GATE_OPEN_SECONDS).
//  * THE KEY IS CARRIED, NOT COPIED. `hasKey` reparents the ONE KeyToken under
//    `Wisp.carry()`, and the dish it came out of goes cold. The token's cell anchor stays
//    exactly on `(kx, ky)` the whole time, which is what the probe measures.
//
// Every flourish is measured on `ctx.time()` — the gate phase, the shrine's growth, the
// seed's fade-in, the ivy breeze — so a paused playback freezes the garden coherently,
// mid-unwind.

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
import {
  SEED_HOVER,
  buildGate,
  buildSeedBed,
  buildShrine,
  type GateHandle,
  type SeedBedHandle,
  type ShrineHandle,
} from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  buildWorld,
  clamp01,
  gardenSkin,
  resolveDoor,
  resolveGrid,
  smoothstep01,
  type GardenSkin,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. A south-west 3/4, 54.5° above the horizon, aimed a stride short of the
// garden's centre. Every one of those three numbers is a measurement.
//
// AZIMUTH is not a taste call: the door sits in a wall that runs north–south, so its faces
// look east and west, and a camera due south would see the scene's entire subject edge-on
// as a sliver. At −0.72 rad the gate's west face is 45° off the view axis (fully readable)
// while the divider still hides only ~0.37 of a cell of the far room's floor.
//
// POLAR is the sightline into the doorway. A raycast census — camera → the avatar's orb
// centre (y = 0.35) over all 31 free cells — is the acceptance test this scene is graded
// on, and it has two knees: too shallow and the wall's own coping covers the orb on the
// cells EAST of the gate; too steep and the frame turns into a plan. 0.62 rad is the
// middle, and it reports zero occluded cells (the door cell included, once the gate's
// piers step back — see JAMB_HEAD_Y in actors.ts).
//
// DISTANCE + TARGET are one decision, and the arithmetic that used to live here was wrong.
// Seen from azimuth a the 9 × 7 garden presents 9·|cos a| + 7·|sin a| = 11.38 units across
// the frame and 11.20 units of depth, and a 54.5° aerial lays that depth over
// 11.20·sin 54.5° = 9.1 units of screen height (NOT 11.20·cos, which is what the first
// pass computed and is why it thought 12.4 had margin). A 40° vertical FOV covers
// 2·d·tan 20° = 9.46 units at d = 13. That would just fit — except the near corner is 1.6×
// closer to the lens than the far one, so a frame centred on the garden's plan centre
// throws it well past the bottom edge (measured: NDC y −1.379 at the old 12.4, i.e. a
// third of a half-frame off-screen, with three border cells sliced and their overlay boxes
// with them). Sliding the orbit target 1.2 units along the view azimuth TOWARD the lens
// re-centres what is projected rather than what is planned: measured, the four kerb corners
// then land at NDC y −0.98 … +0.90 — the whole garden, kerb included, inside the frame,
// with the worst free-cell corner at 0.68 and the gate only 10% smaller than the crop was.
const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.62;
const CAM_DISTANCE = 13.0;
const CAM_TARGET_Y = 0.5;
/** how far the orbit target slides along the view azimuth, toward the camera */
const CAM_TARGET_BIAS = 1.2;
const CAM_TARGET_X = CAM_TARGET_BIAS * Math.sin(CAM_AZIMUTH);
const CAM_TARGET_Z = CAM_TARGET_BIAS * Math.cos(CAM_AZIMUTH);
const CAM_YAW_RANGE = 0.55;
/**
 * The thumbnail is the SAME composition pulled back until a square crop holds it: same
 * azimuth, same aerial, same biased target, distance 15.4. Measured at 1:1 the whole
 * garden fits (worst kerb corner |NDC| 0.96, worst free-cell corner 0.72) — the old thumb
 * swung round to −0.6 rad, which put the wall's coping across the door cell.
 */
const CAM_DISTANCE_THUMB = 15.4;
const CAM_AZIMUTH_THUMB = CAM_AZIMUTH;
const CAM_POLAR_THUMB = CAM_POLAR;

// Celebration weights. The garden is 9 units of bright morning ground: the kit's
// full-strength grammar (a 3.2-unit ripple and a 1.0 bloom pulse) washes out three cells
// at once, which is exactly the readability this scene exists to protect. Four fifths of
// the amplitude, all three of the beats.
const BURST_SCALE = 0.8;
const RIPPLE_SCALE = 0.46;
const BURST_PARTICLES = 110;
const PULSE_STRENGTH = 0.5;
/** the goal burst blooms at the shrine's ember height, not on the gravel */
const BLOOM_Y = 0.6;
const BLOOM_SECONDS = 1.5;

/** mid-tier beats (STYLE.md: burst at reduced scale, no ripple) */
const KEY_BURST_SCALE = 0.34;
const IVY_BURST_SCALE = 0.42;
/** the ivy releases at the height of the leaf it was holding */
const IVY_BURST_Y = 0.62;

/** a new shrine and a new seed take this long to grow in (no teleport-slide) */
const GROW_SECONDS = 0.75;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.17;

/**
 * THE GATE'S TWO CLOCKS, AND WHY THEY ARE NOT THE SAME LENGTH.
 *
 * The env opens the door and moves the agent into the door cell in the SAME control step:
 * the `door` event and the hop that carries the wisp through the doorway arrive together.
 * At 8 Hz that hop crosses the leaf's own plane about 0.07 s later — so an unwind that
 * takes any longer than a hop renders the exact opposite of what happened: the orb walks
 * through a sewn-shut gate, and the empty gate opens behind it. The opening therefore gets
 * EXACTLY ONE HOP (canon lesson 17's constant, reused: the two motions are the same event
 * seen from two sides, and they must finish together).
 *
 * The relock is the mirror image and gets its own, far longer window. It rides the goal
 * respawn, which teleports the wisp back across the garden — there is nobody at the gate to
 * overtake, so the ivy can take its time sewing itself shut, which is the beat the scene's
 * logline promises. One scalar, one direction-selected span, still no state machine.
 */
export const GATE_OPEN_SECONDS = HOP_SECONDS;
export const GATE_RELOCK_SECONDS = 0.9;

/** the shrine breathes and brightens as the wisp closes in — pure (state, time) */
const GLOW_BASE = 0.3;
const GLOW_BREATH = 0.16;
const GLOW_PROXIMITY = 0.44;
const PROXIMITY_CELLS = 4;

interface KeyCorridorLike {
  x?: unknown;
  y?: unknown;
  gx?: unknown;
  gy?: unknown;
  kx?: unknown;
  ky?: unknown;
  hasKey?: unknown;
  doorOpen?: unknown;
}

/** The frame's truth, snapped to cells and booleans. */
export interface GardenState {
  agent: GridCell;
  goal: GridCell;
  key: GridCell;
  hasKey: boolean;
  doorOpen: boolean;
}

export function makeGardenState(): GardenState {
  return {
    agent: { x: 0, y: 0 },
    goal: { x: 0, y: 0 },
    key: { x: 0, y: 0 },
    hasKey: false,
    doorOpen: false,
  };
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/**
 * Read one frame. sim-core's decoded state is the source; the raw pose
 * (`[x, y, gx, gy, kx, ky, hasKey, doorOpen, 0]`) is the fallback so the scene still
 * stands up under the harness's fixture loop, which carries no `derived`.
 */
export function readState(state: FrameState, grid: GridSize, out: GardenState): GardenState {
  const d = state.derived as KeyCorridorLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const p = state.pose;
  out.agent.x = Math.round(num(has ? d.x : undefined, p[0] ?? 0));
  out.agent.y = Math.round(num(has ? d.y : undefined, p[1] ?? 0));
  out.goal.x = Math.round(num(has ? d.gx : undefined, p[2] ?? 0));
  out.goal.y = Math.round(num(has ? d.gy : undefined, p[3] ?? 0));
  out.key.x = Math.round(num(has ? d.kx : undefined, p[4] ?? 0));
  out.key.y = Math.round(num(has ? d.ky : undefined, p[5] ?? 0));
  out.hasKey = has && typeof d.hasKey === 'boolean' ? d.hasKey : (p[6] ?? 0) !== 0;
  out.doorOpen = has && typeof d.doorOpen === 'boolean' ? d.doorOpen : (p[7] ?? 0) !== 0;
  clampCell(out.agent, grid);
  clampCell(out.goal, grid);
  clampCell(out.key, grid);
  return out;
}

class IvyGateScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private gate: GateHandle | null = null;
  private shrine: ShrineHandle | null = null;
  private seed: SeedBedHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });
  private door: GridCell = { x: 0, y: 0 };
  private skin: GardenSkin | null = null;

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private gateAt = -Infinity;
  private shrineGrowAt = -Infinity;
  private seedGrowAt = -Infinity;
  private bloomAt = -Infinity;

  /** whether the ONE KeyToken currently hangs off the wisp */
  private carried = false;
  /** the cell currently painted as the shrine's pad (-1 = none) */
  private readonly paddedGoal: GridCell = { x: -1, y: -1 };

  // ---- preallocated scratch: update() may not allocate ----
  private readonly cur: GardenState = makeGardenState();
  private readonly prevState: GardenState = makeGardenState();
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly bloomCell: GridCell = { x: 0, y: 0 };

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'ivyGateScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.door = resolveDoor(grid);

    const palette = ctx.tokens.palettes.morning;
    const skin = gardenSkin(palette, ctx.tokens);
    this.skin = skin;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    // This scene claims all four of the rig's point lights. Their COLORS are static — only
    // position and intensity are state-driven — so update() never touches a Color.
    const shrineLight = world.rig.points[0];
    if (shrineLight) shrineLight.color.set(skin.amber);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const seedLight = world.rig.points[2];
    if (seedLight) seedLight.color.set(skin.amber);
    const bloomLight = world.rig.points[3];
    if (bloomLight) bloomLight.color.set(ctx.tokens.celebrationGold);

    // ---- the shared avatar ----
    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        // a shade over the 0.16 default: from the composed 13-unit standoff the default
        // orb is ~16 px of pale amber on a pale floor and the eye loses it
        radius: 0.17,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'gardenWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    // ---- the gate: the scene's subject ----
    const gate = buildGate({ tier: ctx.tier, palette, skin, yaw: world.gateYaw, seed: ctx.seed, name: 'ivyGate' });
    this.gate = gate;
    this.scene.add(gate.group);
    cellToWorldVec(this.door.x, this.door.y, grid, gate.root.position);

    // ---- the shrine (goal) and the seed bed (key) ----
    const shrine = buildShrine({ tier: ctx.tier, palette, skin, name: 'goalShrine' });
    this.shrine = shrine;
    this.scene.add(shrine.group);

    const seed = buildSeedBed({ tier: ctx.tier, palette, skin, name: 'seedBed' });
    this.seed = seed;
    this.scene.add(seed.group);

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [CAM_TARGET_X, CAM_TARGET_Y, CAM_TARGET_Z],
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
      name: 'gardenOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'gardenCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay (the QA instrument) ----
    // Explicit resolvers, never the sniffers: this env carries a live KEY cell and a live
    // DOOR flag, and the overlay has to draw both off the frame or it is not ground truth.
    const overlayState = makeGardenState();
    const overlayKeys: GridCell[] = [{ x: 0, y: 0 }];
    const overlayDoors: boolean[] = [false];
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => readState(state, grid, overlayState).agent,
      goalCell: (state) => readState(state, grid, overlayState).goal,
      keyCells: (state) => {
        const st = readState(state, grid, overlayState);
        const cell = overlayKeys[0];
        if (cell) {
          cell.x = st.key.x;
          cell.y = st.key.y;
        }
        return overlayKeys;
      },
      doorStates: (state) => {
        overlayDoors[0] = readState(state, grid, overlayState).doorOpen;
        return overlayDoors;
      },
      name: 'gardenGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    gate.setState(0, 0);
    gate.setTime(0);
    shrine.setState(1, GLOW_BASE);
    shrine.setTime(0);
    seed.setState(1, 1);
    seed.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const gate = this.gate;
    const shrine = this.shrine;
    const seed = this.seed;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !gate || !shrine || !seed || !pool || !rig) return;

    const time = ctx.time();
    const grid = this.grid;
    const s = readState(frame.state, grid, this.cur);

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. a scrub wipes every in-flight beat: nothing may survive a discontinuity
    if (frame.scrubbed) {
      pool.clear();
      this.gateAt = -Infinity;
      this.shrineGrowAt = -Infinity;
      this.seedGrowAt = -Infinity;
      this.bloomAt = -Infinity;
      wisp.setCell(s.agent.x, s.agent.y, grid);
      wisp.snap();
      rig.snap();
    }

    // 3. events — celebrations are NEVER inferred from the state stream
    let teleported = false;
    let sawGoal = false;
    let sawDone = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        sawGoal = true;
        // The +1 was earned by walking INTO the shrine, and by the time this frame is
        // drawn the shrine has already respawned across the garden — so the cell that
        // deserves the petals is the goal cell of the PREVIOUS frame, not this one's.
        this.readBloomCell(frame, wisp);
        cellToWorldVec(this.bloomCell.x, this.bloomCell.y, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.skin?.amber ?? ctx.tokens.celebrationGold, BURST_SCALE, time);
        pool.ripple(this.groundPos, ctx.tokens.celebrationGold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        shrine.flare(0.8);
        this.bloomAt = time;
        // everything the respawn moved grows back in where the frame now says it is
        this.shrineGrowAt = time;
        this.seedGrowAt = time;
      } else if (ev.type === 'key') {
        // mid-tier: a small amber puff where the seed was lifted out of its dish
        cellToWorldVec(s.key.x, s.key.y, grid, this.bloomPos);
        this.bloomPos.y = SEED_HOVER;
        pool.burst(this.bloomPos, this.skin?.amber ?? ctx.tokens.celebrationGold, KEY_BURST_SCALE, time);
        wisp.flare(0.6);
      } else if (ev.type === 'door') {
        // ONE clock for both beats; the phase function reads the direction off the frame
        this.gateAt = time;
        if (ev.open) {
          // mid-tier: the ivy letting go, in ivy green, at the height of the leaf
          cellToWorldVec(this.door.x, this.door.y, grid, this.bloomPos);
          this.bloomPos.y = IVY_BURST_Y;
          pool.burst(this.bloomPos, this.skin?.ivy ?? ctx.tokens.celebrationGold, IVY_BURST_SCALE, time);
        }
      } else if (ev.type === 'teleport') {
        teleported = true;
      } else if (ev.type === 'done') {
        sawDone = true;
      }
    }
    // KeyCorridor's `done` IS the reward — it always rides with the goal — so the sigh is
    // reserved for a done that earned nothing. It never fires in this stream; it fires if
    // the env ever grows a failure, and that is exactly what the shared grammar is for.
    if (sawDone && !sawGoal) pool.sigh(0.7, this.groundPos, time);

    // 4. the respawn really does move the agent across the garden. A hop would glide it
    //    straight through the wall it just walked round: hard cut instead.
    if (teleported) {
      wisp.setCell(s.agent.x, s.agent.y, grid);
      wisp.snap();
      rig.snap();
    }

    // 5. the avatar: one straight hop per cell change, never a path plan.
    //    `wisp.face()` is deliberately NOT called — see the probe note in `probes()`.
    if (s.agent.x !== wisp.cellX || s.agent.y !== wisp.cellY) {
      wisp.hopTo(s.agent, time);
    }
    wisp.update(frame.dtReal, time);

    // 6. the seed: its ANCHOR is always the frame's key cell; only the token moves house
    cellToWorldVec(s.key.x, s.key.y, grid, seed.root.position);
    if (s.hasKey !== this.carried) {
      if (s.hasKey) {
        wisp.carry(seed.token.group);
      } else {
        wisp.carry(null);
        seed.cradle.add(seed.token.group);
        seed.token.group.position.set(0, 0, 0);
      }
      this.carried = s.hasKey;
    }
    const seedGrow = growth(time - this.seedGrowAt);
    seed.setState(seedGrow, s.hasKey ? 0 : 1);
    seed.setTime(time);
    seed.update(frame.dtReal, time);

    // 7. the shrine: position is pure state, growth is pure event clock. Its CELL is
    //    repainted too — the flagstone under the lantern goes warm — so the goal reads
    //    from a thumbnail, where the lantern itself is only a few pixels tall.
    cellToWorldVec(s.goal.x, s.goal.y, grid, shrine.root.position);
    if (s.goal.x !== this.paddedGoal.x || s.goal.y !== this.paddedGoal.y) {
      if (this.paddedGoal.x >= 0) world.restoreFloorTint(this.paddedGoal.x, this.paddedGoal.y);
      world.paintFloor(s.goal.x, s.goal.y, this.skin?.shrinePad ?? ctx.tokens.celebrationGold);
      this.paddedGoal.x = s.goal.x;
      this.paddedGoal.y = s.goal.y;
    }
    const shrineGrow = growth(time - this.shrineGrowAt);
    const glow = shrineGlow(time, s.agent, s.goal);
    shrine.setState(shrineGrow, glow);
    shrine.setTime(time);
    shrine.update(frame.dtReal, time);

    // 8. the gate: one bit of state plus one clock, sequenced into two beats
    gate.setState(gatePhase(s.doorOpen, time - this.gateAt), time);
    gate.setTime(time);

    // 9. the four point lights this scene claims from the rig
    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const shrineLight = world.rig.points[0];
    if (shrineLight) {
      shrineLight.position.set(
        shrine.root.position.x,
        shrine.crownHeight * shrineGrow,
        shrine.root.position.z,
      );
      shrineLight.intensity = (0.2 + 0.7 * glow) * shrineGrow;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.14 + (s.hasKey ? 0.3 : 0) + 0.7 * bloom;
    }
    const seedLight = world.rig.points[2];
    if (seedLight) {
      seedLight.position.set(seed.root.position.x, SEED_HOVER, seed.root.position.z);
      seedLight.intensity = s.hasKey ? 0 : 0.55 * seedGrow;
    }
    const bloomLight = world.rig.points[3];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.position.y = BLOOM_Y;
      bloomLight.intensity = 2.1 * bloom;
    }

    // 10. one clock for the whole garden
    world.setTime(time);

    // 11. camera easing runs in WALL time so a paused frame is still explorable
    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(frame.state);
  }

  /**
   * Where the +1 happened. `prev` is the frame the agent was on when it stepped into the
   * shrine, so ITS goal cell is the one that just paid out. Without a `prev` (a lone
   * fixture, the very first frame) the wisp's last known cell is the best remaining
   * evidence — it is at least in the room the reward came from.
   */
  private readBloomCell(frame: FrameUpdate, wisp: Wisp): void {
    const prev = frame.prev;
    if (prev) {
      const p = readState(prev, this.grid, this.prevState);
      this.bloomCell.x = p.goal.x;
      this.bloomCell.y = p.goal.y;
      return;
    }
    this.bloomCell.x = wisp.cellX;
    this.bloomCell.y = wisp.cellY;
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
    const door = this.door;
    const wisp = this.wisp;
    const gate = this.gate;
    const shrine = this.shrine;
    const seed = this.seed;
    const height = this.ctx?.tokens.wispHeight ?? TOKENS.wispHeight;

    const wispState = makeGardenState();
    const anchorState = makeGardenState();
    const goalState = makeGardenState();
    const keyState = makeGardenState();
    const wispOut: Vec3Tuple = [0, 0, 0];
    const anchorOut: Vec3Tuple = [0, 0, 0];
    const goalOut: Vec3Tuple = [0, 0, 0];
    const keyOut: Vec3Tuple = [0, 0, 0];
    const gateOut: Vec3Tuple = [0, 0, 0];

    // THE PROBE RULE FOR EVERY GRID SCENE: `assertUpdateIdempotent` compares probe
    // objects' whole WORLD MATRIX, not just their translation, so a probe may carry no
    // eased value of any kind — position OR rotation OR scale. `Wisp.face()` damps the
    // body yaw over `dt` and `wisp.probeObject` hangs off that body, so a scene that turns
    // its wisp must probe `wisp.avatar` instead. This garden's wisp is a radially
    // symmetric spirit whose skirt already spins on its own clock, so facing buys it
    // nothing — it never calls `face()`, and both anchors stay exact and probeable.
    return [
      {
        // the orb centre: exactly `wispHeight` over the true cell, bob and breath and
        // skirt and ribbon all excluded (they live on `decor`)
        name: 'wisp',
        object: wisp?.probeObject ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          const st = readState(state, grid, wispState);
          cellToWorld(st.agent.x, st.agent.y, grid, wispOut);
          wispOut[1] = height;
          return wispOut;
        },
      },
      {
        // and the avatar node under it: the cell itself, y = 0, position only
        name: 'wisp-anchor',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          const st = readState(state, grid, anchorState);
          return cellToWorld(st.agent.x, st.agent.y, grid, anchorOut);
        },
      },
      {
        // the shrine root — it must land on the RESPAWNED goal cell, every time
        name: 'shrine',
        object: shrine?.root ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          const st = readState(state, grid, goalState);
          return cellToWorld(st.goal.x, st.goal.y, grid, goalOut);
        },
      },
      {
        // the seed's dish: exactly the frame's key cell, carried or not. The token itself
        // moves house when `hasKey` flips; this anchor never does, which is precisely why
        // it is the thing worth measuring.
        name: 'seed-bed',
        object: seed?.root ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          const st = readState(state, grid, keyState);
          return cellToWorld(st.key.x, st.key.y, grid, keyOut);
        },
      },
      {
        // the gate owns the env's own door cell — read from `grid.doors`, never guessed
        name: 'gate',
        object: gate?.root ?? new THREE.Object3D(),
        expectWorldPos: () => cellToWorld(door.x, door.y, grid, gateOut),
      },
    ];
  }

  dispose(): void {
    // the token may be hanging off the wisp; let go before either side is torn down
    this.wisp?.carry(null);
    this.carried = false;
    this.overlay?.dispose();
    this.overlay = null;
    this.seed?.dispose();
    this.seed = null;
    this.shrine?.dispose();
    this.shrine = null;
    this.gate?.dispose();
    this.gate = null;
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
    this.skin = null;
    this.gateAt = -Infinity;
    this.shrineGrowAt = -Infinity;
    this.seedGrowAt = -Infinity;
    this.bloomAt = -Infinity;
    this.paddedGoal.x = -1;
    this.paddedGoal.y = -1;
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
 * A respawned shrine's (or seed's) scale-in. A never-fired clock (`-Infinity`) and any age
 * past the window both return 1, which is what makes a scrub land on a settled garden
 * instead of replaying a growth it never saw.
 */
export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.15 * Math.sin(Math.PI * t) * (1 - t));
}

/**
 * The gate's phase from the frame's ONE bit plus the age of the last door event.
 * 0 = locked and sewn shut, 1 = ivy released and the leaf swung back.
 *
 * The direction is READ OFF THE FRAME, never remembered: when `open` is true the phase
 * runs 0 → 1 over a hop's worth of seconds (the agent is coming THROUGH — see
 * GATE_OPEN_SECONDS), and when it is false the same scalar runs 1 → 0 over the long relock
 * window. A settled clock (`-Infinity`, or any age past the window) returns the target
 * exactly, so a scrub lands on a fully-shut or fully-open gate and never on a half-swung
 * one.
 */
export function gatePhase(open: boolean, age: number): number {
  const span = open ? GATE_OPEN_SECONDS : GATE_RELOCK_SECONDS;
  const p = !(age >= 0) || age >= span ? 1 : smoothstep01(age / span);
  const target = open ? 1 : 0;
  return clamp01(target * p + (1 - target) * (1 - p));
}

/**
 * How hot the shrine's ember burns: a slow breath plus a term that rises as the wisp
 * closes in. Both are pure functions of `(time, state)` — no history, no easing — so the
 * shrine is identical on a scrubbed frame and on a played one. The proximity term shows
 * nothing the agent lacks: it is the agent's own cell and the goal's own cell, two of the
 * eight numbers the frame is made of.
 */
export function shrineGlow(time: number, agent: GridCell, goal: GridCell): number {
  const d = Math.max(Math.abs(agent.x - goal.x), Math.abs(agent.y - goal.y));
  const near = d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
  const breath = 0.5 + 0.5 * Math.sin(time * 0.64);
  return clamp01(GLOW_BASE + GLOW_BREATH * breath + GLOW_PROXIMITY * near);
}

const createScene: SceneFactory = () => new IvyGateScene();
export default createScene;
