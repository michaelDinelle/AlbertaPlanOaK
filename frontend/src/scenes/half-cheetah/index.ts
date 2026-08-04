// Gold Hour Savanna — HalfCheetah-v4. The one non-wisp hero of the preserve.
//
// THE ONE IDEA: the animal IS its forward kinematics. `geometry.cheetah.fk(qpos)` returns
// ten planar points; this scene maps them
//
//        FK x -> world x      FK z -> world y      scene depth -> world z (the animal
//                                                  lives at z = 0, exactly)
//
// and hangs a coat on them. Joint nodes carry POSITION ONLY, bones sit at joint midpoints
// rotated to point at the next joint, the head and the tail are children of exact joints,
// and the link lengths are measured from `fk(0^9)` rather than typed in. Turn on the
// ground-truth overlay and the cyan stick figure runs INSIDE the cheetah, because it is
// the same ten numbers.
//
// The second idea is the treadmill. rootx runs to 431 m in the recorded lane (1871 m in
// seed 1), so the world is generated around the runner: a ground mesh whose height field
// is periodic in x (snapping it by one period is invisible), and grass / acacias / rocks
// keyed by integer tile index hashed with the lane seed. Scrub to step 90000 and
// the same three trees stand in the same places they would have if you had walked there.
//
// The third is honesty about contact. The savanna floor is identically y = 0 wherever the
// animal can reach it, so an FK toe at y = 0.046 (the geom radius) is standing on the
// visible ground. Dust and scuff are pure functions of toe height × speed — no emitters,
// no history, nothing to desynchronize on a scrub. The one part of the animal the sim does
// NOT define — the tail — is held to that same plane by a solver (actors.ts `setPose`), so
// the lowest thing in the frame is always a foot: a decoration that dipped through the
// contact plane would destroy the only contact cue the frame has.
//
// Celebrations: none. HalfCheetah has no goal, no done and no +1 — its reward is the
// forward velocity you are already watching. The only event the decoder can raise is
// `teleport` (a gym-1000 episode reset), which gets the contract's hard cut: hold the
// last reachable pose while the host's interpolation slides, snap the camera, flash once.

import {
  CelebrationPool,
  attachCelebrations,
  disposeCelebrations,
} from '@kit/celebrate';
import { GroundTruthOverlay, defaultFkToWorld } from '@kit/debug';
import { makeTrackRig, type TrackRig } from '@kit/rigs';
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
import { CHEETAH_DT, CHEETAH_TELEPORT_DX } from '@sim/envs/halfcheetah';
import { FK_FLOATS, FK_POINT_INDEX, cheetahGeometry } from '@sim/fk/cheetah';
import type { CheetahGeometry } from '@sim/types';
import { REFERENCE_SPEED, buildCheetah, type CheetahHandle } from './actors';
import { manifest } from './manifest';
import { buildWorld, clampRange, type WorldHandle } from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

/**
 * THE FRAME, and it took the whole shoot to find it. Roughly the animal's own eye height
 * and a little ahead of it, pitched down only ~4°: high enough that the ground plane still
 * reads (a truly level camera on a flat plain gives no contact cue at all and the cheetah
 * appears to hover), low enough that the frame keeps half a sky — which is where every
 * non-amber colour in this scene lives. Looking AHEAD of the runner puts it on the left
 * third with the plain to run into.
 *
 * The camera tracks a FIXED height, not the bobbing root: track the root's y and the whole
 * world pumps up and down with the gait.
 */
export const TRACK_Y = 0.52;
export const CAM_OFFSET: readonly [number, number, number] = [0.95, 0.33, 3.5];
/** looking AHEAD of the runner puts it on the left third and leaves the plain to run into */
const CAM_LOOK: readonly [number, number, number] = [0.68, 0.06, 0];
/** thumb: closer, centred, square-safe — the wall view crops hard */
const CAM_OFFSET_THUMB: readonly [number, number, number] = [0.12, 0.42, 4.35];
const CAM_LOOK_THUMB: readonly [number, number, number] = [0.22, 0.1, 0];

/** one-frame warm flash after a teleport (playback seconds) */
const FLASH_SECONDS = 0.34;

/**
 * THE SUPERLUMINAL GUARD (canon lesson 4). The host lerps continuous poses, so the single
 * control step that contains an episode reset renders the animal SLIDING hundreds of
 * metres back down the plain. A real control step moves rootx by well under a metre; 1.0
 * (= 20 m/s) is twenty times anything the model can do and still a tenth of the decoder's
 * own 10 m teleport threshold, so the window catches every interpolated respawn and never
 * a real stride.
 */
const MAX_STEP_DX = 1.0;
const SUPERLUMINAL_SLACK = 0.05;
const MAX_HELD_FRAMES = 4;

interface CheetahLike {
  velX?: unknown;
}

/** the env's own forward-velocity term, from sim-core when it is there. */
function velocityOf(state: FrameState, prev: FrameState | null): number {
  const d = state.derived as CheetahLike | null | undefined;
  if (d && typeof d === 'object' && typeof d.velX === 'number' && Number.isFinite(d.velX)) {
    return Math.abs(d.velX) > CHEETAH_TELEPORT_DX ? 0 : d.velX;
  }
  if (prev) {
    const dSteps = state.step - prev.step;
    const dx = (state.pose[0] ?? 0) - (prev.pose[0] ?? 0);
    if (dSteps > 0 && Math.abs(dx) < CHEETAH_TELEPORT_DX) return dx / (dSteps * CHEETAH_DT);
  }
  return 0;
}

class GoldHourSavannaScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private cheetah: CheetahHandle | null = null;
  private rig: TrackRig | null = null;
  private pool: CelebrationPool | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private fkSource: CheetahGeometry = cheetahGeometry;

  // ---- preallocated per-frame scratch (update() must never allocate) ----
  private readonly fkOut = new Float32Array(FK_FLOATS);
  private readonly probeFk = new Float32Array(FK_FLOATS);
  private readonly probeOut: [number, number, number] = [0, 0, 0];
  private readonly renderPose = new Float32Array(9);
  private readonly heldPose = new Float32Array(9);

  // ---- the superluminal guard ----
  private heldValid = false;
  private heldStep = Number.NaN;
  private heldFrames = 0;

  // ---- flourish clock (playback seconds; -Infinity = idle) ----
  private flashAt = -Infinity;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'goldHourSavannaScene';
    this.fkSource = ctx.geometry.cheetah ?? cheetahGeometry;

    const palette = ctx.tokens.palettes.dusk;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene);
    this.world = world;
    this.scene.add(world.group);

    // This scene claims both of the rig's point lights and their COLOUR is static — only
    // position and intensity are state-driven, so update() never touches a Color.
    const bounce = world.rig.points[0];
    if (bounce) bounce.color.set(palette.accentLight);
    const kicker = world.rig.points[1];
    if (kicker) kicker.color.set(palette.sun);

    const cheetah = buildCheetah({
      tier: ctx.tier,
      palette,
      accent,
      seed: ctx.seed,
      cheetah: this.fkSource,
    });
    this.cheetah = cheetah;
    this.scene.add(cheetah.group);

    const thumb = ctx.tier === 'thumb';
    const rig = makeTrackRig({
      offset: thumb ? CAM_OFFSET_THUMB : CAM_OFFSET,
      lookOffset: thumb ? CAM_LOOK_THUMB : CAM_LOOK,
      fov: ctx.tokens.fovHero,
      near: 0.08,
      far: 420,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      lookAhead: thumb ? 0 : 0.34,
      maxLookAhead: 1.1,
      damping: 3.4,
      lookDamping: 5,
      velocityDamping: 2.6,
      name: 'savannaTrackRig',
    });
    this.rig = rig;
    this.camera = rig.camera;

    // the host reads this pool's own channel through celebrationPulse(scene); this scene
    // never bursts, but it owns a channel so one lane's flash cannot bloom another's frame
    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      name: 'savannaCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      // the scene's planar mapping, written down once: FK x -> world x, FK z -> world y
      fkToWorld: defaultFkToWorld,
      name: 'savannaGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    cheetah.setPixelRatio(ctx.size.dpr);
    world.setPixelRatio(ctx.size.dpr);

    // stand the animal up on its own init pose so the very first rendered frame is right
    this.applyPose(this.renderPose, 0);
    world.setTime(0);
    cheetah.setTime(0);
    cheetah.setGlow(1);
    rig.setTarget(0, TRACK_Y, 0);
    rig.snap();
  }

  /** the one place the world is driven from a pose — shared by init and update */
  private applyPose(pose: Float32Array, speed01: number): void {
    const cheetah = this.cheetah;
    const world = this.world;
    if (!cheetah || !world) return;
    this.fkSource.fk(pose, this.fkOut);
    cheetah.setPose(this.fkOut, speed01);
    const rootX = this.fkOut[FK_POINT_INDEX.o1 * 2] ?? 0;
    world.setOrigin(rootX);
    world.setShadowTarget(rootX);
    world.setSpeed(speed01);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const cheetah = this.cheetah;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !cheetah || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;

    // 1. the pool adopts the host clock BEFORE anything could fire into it
    pool.update(frame.dtReal, time);

    // 2. a scrub wipes every in-flight beat and every guard
    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
      this.heldValid = false;
      this.heldFrames = 0;
    }

    // 3. events — this scene celebrates NOTHING (there is no +1 in HalfCheetah). The only
    //    event the decoder can raise is a gym-1000 respawn.
    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'teleport') teleported = true;
    }

    // 4. the superluminal guard: hold the last reachable pose through the slide
    const rawX = state.pose[0] ?? 0;
    let held = false;
    if (!teleported && !frame.scrubbed && this.heldValid && this.heldFrames < MAX_HELD_FRAMES) {
      const dSteps = Math.max(1, Math.abs(state.step - this.heldStep));
      const reach = MAX_STEP_DX * dSteps + SUPERLUMINAL_SLACK;
      if (Math.abs(rawX - (this.heldPose[0] ?? 0)) > reach) held = true;
    }
    if (held) {
      this.renderPose.set(this.heldPose);
      this.heldFrames++;
    } else {
      for (let i = 0; i < 9; i++) this.renderPose[i] = state.pose[i] ?? 0;
      this.heldFrames = 0;
    }
    this.heldPose.set(this.renderPose);
    this.heldStep = state.step;
    this.heldValid = true;
    if (teleported) this.flashAt = time;

    // 5. the animal and the world it re-generates around itself
    const velX = held ? 0 : velocityOf(state, frame.prev);
    const speed01 = clampRange(Math.abs(velX) / REFERENCE_SPEED, 0, 1);
    this.applyPose(this.renderPose, speed01);

    // 6. envelopes — measured on the playback clock, so a pause freezes them mid-air
    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.6);
    cheetah.setGlow(1 + 0.85 * flash);
    if (flash > 0) pool.pulse.fire(0.45 * flash);

    // 7. the two point lights this scene claims from the rig, riding with the runner
    const rootX = this.renderPose[0] ?? 0;
    const bounce = world.rig.points[0];
    if (bounce) {
      bounce.position.set(rootX + 1.4, 1.95, 4.7);
      bounce.intensity = 3.4 + 24 * flash;
    }
    const kicker = world.rig.points[1];
    if (kicker) {
      kicker.position.set(rootX - 1.5, 1.15, -1.9);
      kicker.intensity = 1.6 + 6 * flash;
    }

    // 8. one clock for the whole world
    world.setTime(time);
    cheetah.setTime(time);

    // 9. the camera: a hard cut across a teleport or a scrub, damped in WALL time
    //    otherwise, so a paused frame still settles
    rig.setTarget(rootX, TRACK_Y, 0);
    if (teleported || frame.scrubbed || ctx.tier === 'thumb') rig.snap();
    else rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.cheetah?.setPixelRatio(size.dpr);
    this.world?.setPixelRatio(size.dpr);
    this.pool?.setPixelRatio(size.dpr);
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
  }

  probes(): AlignmentProbe[] {
    const joints = this.cheetah?.joints ?? [];
    const source = this.fkSource;
    const buffer = this.probeFk;
    const out = this.probeOut;
    const at =
      (index: number) =>
      (state: FrameState): readonly [number, number, number] => {
        source.fk(state.pose, buffer);
        out[0] = buffer[index * 2] ?? 0;
        out[1] = buffer[index * 2 + 1] ?? 0;
        out[2] = 0;
        return out;
      };
    const node = (index: number): THREE.Object3D => joints[index] ?? new THREE.Object3D();
    return [
      // the torso anchor: (rootx, ROOT_Z0 + rootz) — the point every other bone hangs from
      { name: 'torso', object: node(FK_POINT_INDEX.o1), expectWorldPos: at(FK_POINT_INDEX.o1) },
      // both toes: the two points that decide whether the animal is standing on the ground
      { name: 'toe-back', object: node(FK_POINT_INDEX.toe_b), expectWorldPos: at(FK_POINT_INDEX.toe_b) },
      { name: 'toe-front', object: node(FK_POINT_INDEX.toe_f), expectWorldPos: at(FK_POINT_INDEX.toe_f) },
      // the head, the one joint whose skin carries a rotation of its own
      { name: 'head', object: node(FK_POINT_INDEX.head), expectWorldPos: at(FK_POINT_INDEX.head) },
    ];
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.cheetah?.dispose();
    this.cheetah = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.flashAt = -Infinity;
    this.heldValid = false;
    this.heldFrames = 0;
  }
}

/**
 * Exponential decay in [0, 1] over `span` playback seconds, 0 outside the window. The
 * linear tail is not cosmetic: a bare exp() still has real amplitude when the window
 * closes, so the flourish would end on a one-frame pop (canon lesson 6).
 */
export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

const createScene: SceneFactory = () => new GoldHourSavannaScene();
export default createScene;
