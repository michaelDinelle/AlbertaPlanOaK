
import {
  CelebrationPool,
  attachCelebrations,
  disposeCelebrations,
} from '@kit/celebrate';
import { GroundTruthOverlay, defaultFkToWorld } from '@kit/debug';
import { makeOrbitRig, orbitFromPose, type OrbitRig } from '@kit/rigs';
import type {
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import * as THREE from 'three';
import { CHEETAH_DT, CHEETAH_ROOT_X, CHEETAH_TELEPORT_DX } from '@sim/envs/halfcheetah';
import { FK_FLOATS, FK_POINT_INDEX, cheetahGeometry } from '@sim/fk/cheetah';
import type { CheetahGeometry } from '@sim/types';
import { REFERENCE_SPEED, buildCheetah, type CheetahHandle } from './actors';
import { manifest } from './manifest';
import { buildWorld, clampRange, type WorldHandle } from './world';


export const TRACK_Y = 0.52;
export const CAM_OFFSET: readonly [number, number, number] = [0.95, 0.33, 3.5];

const FLASH_SECONDS = 0.34;

const MAX_STEP_DX = 1.0;
const SUPERLUMINAL_SLACK = 0.05;
const MAX_HELD_FRAMES = 4;

function velocityOf(state: FrameState, prev: FrameState | null): number {
  if (prev) {
    const dSteps = state.step - prev.step;
    const dx = (state.pose[CHEETAH_ROOT_X] ?? 0) - (prev.pose[0] ?? 0);
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
  private rig: OrbitRig | null = null;
  private pool: CelebrationPool | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private fkSource: CheetahGeometry = cheetahGeometry;

  private readonly fkOut = new Float32Array(FK_FLOATS);
  private readonly renderPose = new Float32Array(9);
  private readonly heldPose = new Float32Array(9);

  private heldValid = false;
  private heldStep = Number.NaN;
  private heldFrames = 0;

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

    const bounce = world.rig.points[0];
    if (bounce) bounce.color.set(palette.accentLight);
    const kicker = world.rig.points[1];
    if (kicker) kicker.color.set(palette.sun);

    const cheetah = buildCheetah({
      palette,
      accent,
      seed: ctx.seed,
      cheetah: this.fkSource,
    });
    this.cheetah = cheetah;
    this.scene.add(cheetah.group);

    const offset = CAM_OFFSET;
    const home = orbitFromPose(offset, [0, 0, 0]);
    const rig = makeOrbitRig({
      canvas: ctx.canvas,
      target: [0, TRACK_Y, 0],
      azimuth: home.azimuth,
      polar: home.polar,
      distance: home.distance,
      fov: ctx.tokens.fovHero,
      near: 0.08,
      far: 420,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 6,
      follow: 3.4,
      name: 'savannaOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;

    const pool = new CelebrationPool({
      seed: ctx.seed,
      name: 'savannaCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      fkToWorld: defaultFkToWorld,
      name: 'savannaGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    cheetah.setPixelRatio(ctx.size.dpr);
    world.setPixelRatio(ctx.size.dpr);

    this.applyPose(this.renderPose, 0);
    world.setTime(0);
    cheetah.setTime(0);
    cheetah.setGlow(1);
    rig.setTarget(0, TRACK_Y, 0);
    rig.snap();
  }

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

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
      this.heldValid = false;
      this.heldFrames = 0;
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'teleport') teleported = true;
    }

    const rawX = state.pose[CHEETAH_ROOT_X] ?? 0;
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

    const velX = held ? 0 : velocityOf(state, frame.prev);
    const speed01 = clampRange(Math.abs(velX) / REFERENCE_SPEED, 0, 1);
    this.applyPose(this.renderPose, speed01);

    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.6);
    cheetah.setGlow(1 + 0.85 * flash);
    if (flash > 0) pool.pulse.fire(0.45 * flash);

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

    world.setTime(time);
    cheetah.setTime(time);

    rig.follow(rootX, TRACK_Y, 0);
    if (teleported || frame.scrubbed) rig.snap();
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

export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

const createScene: SceneFactory = () => new GoldHourSavannaScene();
export default createScene;
