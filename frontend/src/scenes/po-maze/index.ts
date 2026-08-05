// The Fogbound Forest — POMaze-cont-v0.
//
// THE ONE IDEA: PARTIAL OBSERVABILITY IS THE SCENE. POMaze's agent gets eight wall bits
// (its own 3 × 3 window) and four compass bits (the sign of the goal's offset, never the
// distance). So the forest is drawn as the agent knows it: dense fog, a warm bubble of
// light exactly one cell wide in every direction, and spruce ranks that condense out of
// the haze the moment they enter that window and dissolve the moment they leave it. A
// slim shaft of light stands on the live goal cell for the AUDIENCE — the agent never
// sees it. Press `g` and the ground-truth overlay draws the whole maze at once: that
// toggle, between "what it knows" and "what is there", is the drama of the scene.
//
// WHAT MAKES IT HONEST, structurally rather than by discipline
//  * `bubbleVisibility` is a pure function of the wisp's CURRENT world position, so the
//    visible set is exactly `poMazeView`'s eight bits when settled, and a scrub lands on
//    the true bubble rather than a remembered one.
//  * The floor is drawn under all 121 cells with a wall-blind tint, the interior holds no
//    other structure, and the only silhouettes past the bubble live outside the playfield.
//  * THE AGENT'S FRAME NEVER KNOWS MORE THAN THE AGENT — but the VIEWER is allowed one
//    thing the agent is not: a closed moonflower bud stands on the live goal cell, dim in
//    the fog, so the audience always knows where the prize is while the wisp does not
//    (owner's ruling, 2026-08-05 — it replaced an absolute no-draw rule). The bud is
//    VIEWER knowledge by construction: nothing the agent-knowledge machinery renders (the
//    bubble, the moth, the lights) reads it, and the indistinguishability test still holds
//    for every drawable OUTSIDE the beacon subtree. The moonflower's root keeps tracking
//    the live goal exactly for the probes; the moth holds the rim of the bubble at a
//    radius that is a function of the clock alone.
//  * The one exception is bought with a +1, and it is the one moment the agent DOES know
//    where the goal was — it just collected the reward standing on it. So the fog lifts
//    for a breath around that cell and the moonflower opens there, on the clearing, as it
//    is found. Reward buys sight, and nothing else does.
//
// WHAT THE STREAM ACTUALLY DOES, and why the camera behaves the way it does: POMaze
// redraws BOTH the agent and the goal on a goal entry, so on the frame `done` fires the
// wisp is already somewhere else and the +1 belongs to the PREVIOUS frame's goal cell.
// The camera therefore CUTS to that clearing and holds it for a beat — that is the
// "revealing where you actually were" the brief asks for — then CUTS again to find the
// spirit wherever the maze threw it. Both are hard cuts: every goal carries a `teleport`,
// and STYLE.md forbids easing across one (an eased return also panned the eye straight
// through the fog wall for a second and a half). Reading the celebration off the agent's
// new cell would bloom a random corner.

import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { cellToWorld, cellToWorldVec, clampCell, type GridSize, type Vec3Tuple } from '@kit/grid';
import { hash1f } from '@kit/rng';
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
  buildGoalBeacon,
  buildLanternPool,
  buildMoonflower,
  buildWispAura,
  type GoalBeaconHandle,
  type LanternPoolHandle,
  type MoonflowerHandle,
  type WispAuraHandle,
} from './actors';
import { manifest } from './manifest';
import {
  BUBBLE_OUTER,
  CAMERA_AZIMUTH,
  blendTokens,
  bubbleVisibility,
  buildWorld,
  clamp01,
  compassOffset,
  forestSkin,
  resolveGrid,
  revealVisibility,
  REVEAL_RADIUS,
  type WorldHandle,
} from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME, and it is the lowest and closest of any grid scene in the preserve — which
// is the brief, and also the only framing this fog permits. FogExp2 at density 0.1345
// leaves 87% of a surface at 1 unit, 67% at 3, 45% at 6 and 15% at 14: a four-rooms-style
// 17.8-unit aerial would render this maze as one flat sheet of haze. At 4.7 units the
// wisp and its bubble keep their own values, the ranks two cells out are silhouettes and
// the far tree line is a rumour — the exact depth ladder the scene is about.
//
// Polar 1.32 puts the eye 1.99 units off the litter (4.55 out, pitched down 14.4°), which
// is low enough that the dawn sky sits above the crowns — the horizon requirement met by
// an actual horizon rather than by argument — and shallow enough that the ranks stand
// UP in the frame instead of being read from above like a map.
const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 1.32;
const CAM_DISTANCE = 4.7;
const CAM_TARGET_Y = 0.82;
/**
 * THE THUMB IS A HELD SHOT (STYLE.md's thumb contract: "fixed camera"). It pulls back and
 * lifts a touch so a square-ish crop holds the whole bubble, and — unlike hero and medium
 * — its target is nailed to the manifest's own `thumbFrame` cell in `init` and never
 * touched again, so on the wall this card sits as still as four-rooms' and the framing the
 * manifest composed is the framing every viewer gets. The cost is real and accepted: a
 * stream that walks its wisp into a far corner walks it out of this frame, which in this
 * fog it would have done at any standoff a card can afford.
 */
const CAM_DISTANCE_THUMB = 6.9;
const CAM_AZIMUTH_THUMB = CAMERA_AZIMUTH + 0.22;
const CAM_POLAR_THUMB = 1.25;
/**
 * The orbit is clamped TIGHT. Distance ≤ 6.6 keeps the camera inside
 * 5 + 6.6·sin(1.36) = 11.5 units of the origin at the very worst (the wisp on a border
 * cell, fully zoomed out, dragged to the end of the azimuth arc), and the outer forest
 * starts at 14 — so nothing can ever stand between this camera and the physics. Polar
 * stays under the horizon and above a ground-grazing shot; the azimuth arc lets you look
 * around a rank, never around the maze.
 */
const CAM_YAW_RANGE = 0.55;
const CAM_MIN_POLAR = 1.04;
const CAM_MAX_POLAR = 1.36;
const CAM_MIN_DISTANCE = 3.6;
const CAM_MAX_DISTANCE = 6.6;
/**
 * How fast the eye keeps up with a WALKING wisp (per second, wall time). It smooths the
 * 8 Hz hop chain and nothing else: every discontinuity — first frame, scrub, teleport, and
 * the cut out of the post-goal linger — bypasses it and snaps.
 */
const CAM_FOLLOW_RATE = 2.2;

// Celebration weights. This frame is small (a 3-unit bubble at 4.7 units) and the air is
// bright, so the kit's full-strength grammar washes the whole clearing out: under half
// the amplitude in a finer spray (the particle SIZE scales with the burst scale, so a
// heavy burst here reads as cotton wool rather than sparks), all three beats.
const BURST_SCALE = 0.45;
const RIPPLE_SCALE = 0.52;
const BURST_PARTICLES = 120;
const PULSE_STRENGTH = 0.5;
/** the burst blooms at moth height over the clearing, not down in the litter */
const BLOOM_Y = 0.55;

/** the goal breath: how long the fog stays lifted, and how long the eye holds on it */
export const REVEAL_SECONDS = 1.9;
export const LINGER_SECONDS = 1.35;

/** one hop is shorter than one 8 Hz control step, so the glide always lands on time */
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.14;

/**
 * The scene's own default bloom hue, derived from the token table at module load exactly
 * as `forestSkin` derives it in `init` — so the field initializer below is a token blend
 * rather than a hex copy of one. (manifest.ts takes its grade tint the same way.)
 */
const DEFAULT_BLOOM = forestSkin(TOKENS.palettes.dawn, TOKENS).bloom.getHex();

interface POMazeLike {
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
export function readCells(state: FrameState, grid: GridSize, agent: GridCell, goal: GridCell): void {
  const d = state.derived as POMazeLike | null | undefined;
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

class FogboundForestScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private flower: MoonflowerHandle | null = null;
  private beacon: GoalBeaconHandle | null = null;
  private lantern: LanternPoolHandle | null = null;
  private aura: WispAuraHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });

  // ---- flourish clocks (playback seconds; -Infinity = settled/idle) ----
  private revealAt = -Infinity;
  private lingerAt = -Infinity;
  private revealX = 0;
  private revealZ = 0;

  // ---- preallocated scratch: update() may not allocate ----
  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly prevGoalCell: GridCell = { x: 0, y: 0 };
  private readonly prevAgentCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly goalWorld = new THREE.Vector3();
  private readonly camAnchor = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private camPrimed = false;
  /** thumb only: the target is fixed at init and update() never moves the eye */
  private camFixed = false;
  /** was the eye holding the clearing last frame? its END is a cut, not a drift */
  private lingering = false;
  // Defaults are TOKEN values, not literals: these are overwritten from `ctx.tokens` in
  // init, and a hex here would be a copy of a token that stops being one the day the
  // token moves (STYLE.md: tokens for every color, scene-local colors are token blends).
  private gold = TOKENS.celebrationGold;
  private bloomColor = DEFAULT_BLOOM;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'fogboundForestScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;

    const palette = ctx.tokens.palettes.dawn;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.gold = ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);
    this.bloomColor = world.skin.bloom.getHex();

    // This scene claims all four of the rig's point lights. Their COLORS are static —
    // only position and intensity are state-driven — so update() never touches a Color.
    // At medium the rig allocates two and the guards below simply skip the rest; the
    // ordering is the priority order (the wisp's own lantern first, the goal breath next).
    const lanternLight = world.rig.points[0];
    if (lanternLight) lanternLight.color.set(blendTokens(accent, palette.accentLight, 0.62));
    const breathLight = world.rig.points[1];
    if (breathLight) breathLight.color.set(this.gold);
    const flowerLight = world.rig.points[2];
    if (flowerLight) flowerLight.color.set(world.skin.bloom);

    // ---- the shared avatar ----
    // The spirit is the one WARM thing in a cold volume of air. `accents['po-maze']` is
    // fog silver — pushed a third of the way into the celebration gold it becomes a
    // lantern rather than a hole in the picture, and it is then the only hue in the frame
    // that is not the haze, which is what makes the eye find it through the ranks.
    const wispColor = blendTokens(accent, this.gold, 0.34);
    const wisp = new Wisp(
      {
        color: wispColor,
        tier: ctx.tier,
        // WELL under the 0.16 default. The kit's halo is 2.4 × the orb and it is a
        // flat-shaded icosahedron, so at this camera it renders as a visible faceted
        // bubble rather than a glow — the passes at 0.19, 0.13 and 0.11 all did. At 0.09
        // (plus the `dim` below) the halo has shrunk back to a rim inside the soft aura
        // sprite; the SIZE of the wisp's presence is carried by the lantern pool and the
        // aura instead, both radial sprites, which read as light rather than as geometry.
        radius: 0.09,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'forestWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    // the warm pool it lays on the litter — the bubble made a place
    const lantern = buildLanternPool(
      blendTokens(accent, palette.accentLight, 0.55),
      BUBBLE_OUTER * 1.15,
      ctx.tier,
    );
    this.lantern = lantern;
    this.scene.add(lantern.mesh);

    const aura = buildWispAura(blendTokens(wispColor, palette.sun, 0.4), 1.05, ctx.tier);
    this.aura = aura;
    this.scene.add(aura.mesh);

    // ---- the goal: an exact anchor that never draws, and a flower that only a +1 opens ----
    // Both go on the SCENE, not on each other: the anchor rides the live goal for the
    // probes and the overlay, the bloom stands wherever the last reward was collected.
    const flower = buildMoonflower({
      tier: ctx.tier,
      skin: world.skin,
      glow: world.skin.bloom,
      name: 'moonflowerRoot',
    });
    this.flower = flower;
    this.scene.add(flower.root);
    this.scene.add(flower.bloom);

    // ---- the goal BEACON: viewer knowledge, never the agent's ----
    // A slim fog-proof shaft of light on the live goal cell. Its whole subtree
    // carries the beacon name so the honesty tests can hold the rest of the
    // frame to the sign-bits contract while the audience keeps the prize.
    const beacon = buildGoalBeacon({
      tier: ctx.tier,
      glow: world.skin.bloom,
      name: 'goalBeacon',
    });
    this.beacon = beacon;
    this.scene.add(beacon.group);

    // ---- camera ----
    const thumb = ctx.tier === 'thumb';
    this.camFixed = thumb;
    // THE HELD THUMB SHOT is composed on the manifest's OWN thumbFrame cell, read out of
    // the manifest rather than restated here, so the card and the pose the gallery ships
    // with it cannot drift apart. Hero and medium start on the maze centre and are primed
    // onto the wisp by the first update.
    if (thumb) {
      const thumbCell: GridCell = {
        x: Math.round(manifest.thumbFrame.pose[0] ?? 0),
        y: Math.round(manifest.thumbFrame.pose[1] ?? 0),
      };
      clampCell(thumbCell, grid);
      cellToWorldVec(thumbCell.x, thumbCell.y, grid, this.camAnchor);
    }
    this.camAnchor.y = CAM_TARGET_Y;
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [this.camAnchor.x, CAM_TARGET_Y, this.camAnchor.z],
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
      name: 'forestOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'forestCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    // ---- the ground-truth overlay: the whole maze, and the point of the scene ----
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
      name: 'forestGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    // ---- a settled opening frame ----
    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    world.setFogLift(0);
    // THE FOG DIM. `Wisp.dim` exists in the kit for exactly this scene ("fade toward a
    // resting glow (POMaze fog…)"), and at this camera it is not optional: the avatar's
    // halo is a flat-shaded icosahedron, so at full brightness two units from the eye it
    // renders as a hard-edged translucent POLYGON around a blown-out core — a faceted
    // marble, not a spirit. Halved, the halo sinks into the soft radial aura sprite in
    // front of it and the orb reads as a lantern in mist. Celebration flares multiply
    // through `flare()` on top of this, so the +1 still punches.
    wisp.dim(0.5);
    wisp.setCell(0, 0, grid);
    flower.setState(0, 0);
    lantern.setGain(1);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const flower = this.flower;
    const lantern = this.lantern;
    const aura = this.aura;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !flower || !lantern || !aura || !pool || !rig) return;

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
      this.revealAt = -Infinity;
      this.lingerAt = -Infinity;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      // the eye's own snap is in step 12, where `frame.scrubbed` is one of the four cuts
      this.camPrimed = false;
      this.lingering = false;
    }

    // 4. events — celebrations are NEVER inferred from the state stream
    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        // The +1 was earned on the cell the agent WALKED INTO — the previous frame's
        // goal — and by the time this frame renders the env has already thrown the agent
        // somewhere else. Read the clearing off `prev`; the agent's own cell is a
        // fallback for a host that hands us no previous frame.
        const prev = frame.prev;
        if (prev) {
          readCells(prev, grid, this.prevAgentCell, this.prevGoalCell);
          cellToWorldVec(this.prevGoalCell.x, this.prevGoalCell.y, grid, this.bloomPos);
        } else {
          cellToWorldVec(agent.x, agent.y, grid, this.bloomPos);
        }
        this.groundPos.copy(this.bloomPos);
        this.revealX = this.bloomPos.x;
        this.revealZ = this.bloomPos.z;
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.bloomColor, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        flower.flare(0.9);
        this.revealAt = time;
        this.lingerAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
      // POMaze's `done` IS the +1 — it never arrives without a `goal` — so this forest
      // has no failure and never sighs. A sigh here would be a lie about the env.
    }

    // 5. the avatar: one straight hop per cell change, never a path plan; a teleport
    //    (which every goal carries) is a hard cut, exactly like a scrub.
    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
    } else if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      wisp.face({ dx: agent.x - wisp.cellX, dy: agent.y - wisp.cellY });
      wisp.hopTo(agent, time);
    }
    wisp.update(frame.dtReal, time);

    const wx = wisp.position.x;
    const wz = wisp.position.z;

    // 6. THE BUBBLE. One pure function of where the wisp is, plus the goal breath.
    const reveal = envelope(time - this.revealAt, REVEAL_SECONDS, 1.6);
    // The AIR and the FOREST run off the same envelope on different curves. The air is
    // linear in it (fog lifts and settles), but the ranks are driven by a saturated
    // version: the whole point of the breath is to SHOW you the clearing you just earned,
    // and an envelope that peaks at 1 and decays means the trees would only ever reach
    // whatever fraction of full size the decay had left — half-grown stumps, which reads
    // as a rendering glitch rather than a revelation. Saturating at 2.2× holds them at
    // full height for the first ~0.4 s, then lets them sink back with the fog.
    const breath = clamp01(reveal * 2.2);
    world.spruce.setBubble(wx, wz, this.revealX, this.revealZ, breath);
    world.setFogLift(reveal);
    world.setShadowFocus(wx, wz);

    // the gap gates live on the same rule as the ranks around them
    for (let i = 0; i < world.gates.length; i++) {
      const gate = world.gates[i];
      if (!gate) continue;
      const gx = gate.root.position.x;
      const gz = gate.root.position.z;
      let v = bubbleVisibility(wx, wz, gx, gz);
      if (breath > 0) {
        const r = breath * revealVisibility(this.revealX, this.revealZ, gx, gz, REVEAL_RADIUS);
        if (r > v) v = r;
      }
      const e = v <= 0 ? 0.0001 : v;
      gate.body.scale.set(e, e * (0.4 + 0.6 * e), e);
      gate.body.visible = v > 0.002;
    }

    // 7. THE GOAL, in its two honest halves. The ROOT rides the live goal cell exactly on
    //    every frame and draws NOTHING — POMaze publishes four sign bits and no distance,
    //    so a flower standing on the goal would be telling the frame something the agent
    //    is never told, at every range, adjacent most of all. The BLOOM stands on the
    //    clearing instead and opens with the breath: the instant of the +1 is the one
    //    instant that cell is a fact the agent owns.
    cellToWorldVec(goal.x, goal.y, grid, this.goalWorld);
    flower.root.position.copy(this.goalWorld);
    flower.bloom.position.set(this.revealX, 0, this.revealZ);
    flower.setState(breath, time);
    flower.update(frame.dtReal, time);

    //    …and the VIEWER'S half: the light shaft on the live goal — fog-proof,
    //    dim next to the wisp's warm bubble, an omen rather than a lamp
    const beacon = this.beacon;
    if (beacon) {
      beacon.group.position.set(this.goalWorld.x, 0, this.goalWorld.z);
      beacon.setTime(time);
    }

    // 8. (the compass moth is retired — the beacon is the audience's direction
    //    now, and a second glowing creature beside the wisp read as a second
    //    agent. `compassOffset` survives in world.ts as a tested pure function.)

    // 9. the wisp's light: a pool on the litter and a billboarded aura of scattering,
    //    both following the avatar and nothing else
    lantern.mesh.position.set(wx, 0.015, wz);
    lantern.setGain(1 + 0.55 * reveal);
    aura.mesh.position.set(wx, wisp.position.y + TOKENS.wispHeight, wz);
    aura.face(this.camera);
    aura.setGain(0.72 + 1.4 * reveal);

    // 10. the four point lights this scene claims from the rig
    const lanternLight = world.rig.points[0];
    if (lanternLight) {
      this.lightPos.set(wx, TOKENS.wispHeight + 0.12, wz);
      lanternLight.position.copy(this.lightPos);
      lanternLight.intensity = 0.62 + 0.5 * reveal;
    }
    const breathLight = world.rig.points[1];
    if (breathLight) {
      breathLight.position.set(this.revealX, 0.75, this.revealZ);
      breathLight.intensity = 2.8 * reveal;
    }
    // the flower's own cool light stands with the bloom, on the clearing — never on the
    // live goal, where even an intensity-0 lamp would be a state-bearing marker
    const flowerLight = world.rig.points[2];
    if (flowerLight) {
      flowerLight.position.set(this.revealX, flower.crownHeight, this.revealZ);
      flowerLight.intensity = 0.75 * breath;
    }
    const beaconLight = world.rig.points[3];
    if (beaconLight) {
      beaconLight.name = 'beacon:goalLight';
      beaconLight.color.set(world.skin.bloom);
      beaconLight.position.set(this.goalWorld.x, 1.4, this.goalWorld.z);
      beaconLight.intensity = 0.3;
    }

    // 11. one clock for the whole forest
    world.setTime(time);

    // 12. THE EYE, cut like film. It rides the wisp; after a +1 it holds the clearing for
    //     a beat — the shot the brief asks for, and the only way to see it at all given
    //     the agent has already been thrown across the maze — and then it CUTS to wherever
    //     that was. Every discontinuity is a cut and never a pan: the first frame, a
    //     scrub, the `teleport` every goal carries (STYLE.md MUST NOT: "Ease across
    //     `scrubbed` or `teleport` — hard snap"), and the end of the linger that teleport
    //     opened. Only walking is smoothed, in WALL time, so a paused frame is explorable.
    //     At thumb the eye does not move at all (see CAM_DISTANCE_THUMB).
    if (!this.camFixed) {
      const linger = time - this.lingerAt >= 0 && time - this.lingerAt < LINGER_SECONDS;
      const tx = linger ? this.revealX : wx;
      const tz = linger ? this.revealZ : wz;
      const cut = !this.camPrimed || frame.scrubbed || teleported || (this.lingering && !linger);
      if (cut) {
        this.camAnchor.set(tx, CAM_TARGET_Y, tz);
        this.camPrimed = true;
        rig.setTarget(tx, CAM_TARGET_Y, tz);
        rig.snap();
      } else {
        const k = 1 - Math.exp(-CAM_FOLLOW_RATE * Math.max(0, frame.dtReal));
        this.camAnchor.x += (tx - this.camAnchor.x) * k;
        this.camAnchor.z += (tz - this.camAnchor.z) * k;
        rig.setTarget(this.camAnchor.x, CAM_TARGET_Y, this.camAnchor.z);
      }
      this.lingering = linger;
    }
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
    const flower = this.flower;
    const world = this.world;

    const wispCell: GridCell = { x: 0, y: 0 };
    const wispGoal: GridCell = { x: 0, y: 0 };
    const wispOut: Vec3Tuple = [0, 0, 0];
    const goalCell: GridCell = { x: 0, y: 0 };
    const goalGoal: GridCell = { x: 0, y: 0 };
    const goalOut: Vec3Tuple = [0, 0, 0];

    // THE WISP PROBE RULE (four-rooms canon lesson 16): this scene calls `wisp.face()`,
    // which damps the body yaw the orb core hangs from. `assertUpdateIdempotent` compares
    // whole world MATRICES, so the core carries an eased rotation and may not be probed.
    // `wisp.avatar` is pure `HopMotion.position` — exactly `cellToWorld`, y = 0.
    const probes: AlignmentProbe[] = [
      {
        name: 'wisp-anchor',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readCells(state, grid, wispCell, wispGoal);
          return cellToWorld(wispCell.x, wispCell.y, grid, wispOut);
        },
      },
      {
        // The goal anchor, which draws nothing at all — that is the scene — while
        // tracking the live, respawning goal exactly. It is what makes "never drawn" a
        // rendering decision instead of an excuse: the position is right on every frame,
        // it agrees with the ground-truth overlay cell for cell, and the art declines to
        // use it. (The flower the viewer does see hangs off no probe on purpose: it
        // stands on the cell a +1 was earned on, which is an event, not a state.)
        name: 'goal-anchor',
        object: flower?.root ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readCells(state, grid, goalCell, goalGoal);
          return cellToWorld(goalGoal.x, goalGoal.y, grid, goalOut);
        },
      },
    ];

    // the gate stones stand on the env's OWN gap cells (grid.hallways), never on a
    // guessed gap — one probe each
    if (world) {
      for (let i = 0; i < world.gates.length; i++) {
        const gate = world.gates[i];
        if (!gate) continue;
        const out: Vec3Tuple = [0, 0, 0];
        const cell = gate.cell;
        probes.push({
          name: `gap-gate${i}`,
          object: gate.root,
          expectWorldPos: () => cellToWorld(cell.x, cell.y, grid, out),
        });
      }
    }
    return probes;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.flower?.dispose();
    this.flower = null;
    this.beacon?.dispose();
    this.beacon = null;
    this.aura?.dispose();
    this.aura = null;
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
    this.revealAt = -Infinity;
    this.lingerAt = -Infinity;
    this.camPrimed = false;
    this.camFixed = false;
    this.lingering = false;
    // init reads this back as the hero/medium starting target, so it must go home
    this.camAnchor.set(0, 0, 0);
  }
}

// ---------------------------------------------------------------------------
// envelopes — measured on the playback clock, so a pause freezes them mid-breath
// ---------------------------------------------------------------------------

/**
 * Exponential decay in [0, 1] over `span` playback seconds, 0 outside the window. The
 * linear tail is not cosmetic: a bare exp() still has real amplitude when the window
 * closes, so the fog would snap back on instead of closing.
 */
export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

const createScene: SceneFactory = () => new FogboundForestScene();
export default createScene;
