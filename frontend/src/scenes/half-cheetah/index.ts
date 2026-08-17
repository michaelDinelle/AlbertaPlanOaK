
import type { CelebrationPoolOptions } from '@kit/celebrate';
import { defaultFkToWorld, type GroundTruthOptions } from '@kit/debug';
import { clampRange, envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene } from '@kit/stage';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { CHEETAH_DT, CHEETAH_ROOT_X } from '@sim/envs/halfcheetah';
import { FK_FLOATS, FK_POINT_INDEX, cheetahGeometry } from '@sim/fk/cheetah';
import type { CheetahGeometry } from '@sim/types';
import { REFERENCE_SPEED, buildCheetah, type CheetahHandle } from './actors';
import { manifest } from './manifest';
import { buildWorld, type WorldHandle } from './world';


export const TRACK_Y = 0.52;
export const CAM_OFFSET: readonly [number, number, number] = [0.95, 0.33, 3.5];

const FLASH_SECONDS = 0.34;

function velocityOf(state: FrameState, prev: FrameState | null): number {
  if (prev && !prev.done) {
    const dSteps = state.step - prev.step;
    const dx = (state.pose[CHEETAH_ROOT_X] ?? 0) - (prev.pose[0] ?? 0);
    if (dSteps > 0) return dx / (dSteps * CHEETAH_DT);
  }
  return 0;
}

class GoldHourSavannaScene extends StageScene<WorldHandle> {
  readonly manifest = manifest;
  protected readonly stageName = 'goldHourSavanna';

  private cheetah: CheetahHandle | null = null;
  private fkSource: CheetahGeometry = cheetahGeometry;

  private readonly fkOut = new Float32Array(FK_FLOATS);
  private readonly renderPose = new Float32Array(9);

  private flashAt = -Infinity;

  protected buildWorld(ctx: SceneContext): WorldHandle {
    this.fkSource = ctx.geometry.cheetah ?? cheetahGeometry;
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const palette = ctx.tokens.palettes.dusk;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;

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
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    const offset = CAM_OFFSET;
    const home = orbitFromPose(offset, [0, 0, 0]);
    return {
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
    };
  }

  protected poolOptions(ctx: SceneContext): CelebrationPoolOptions {
    return {
      seed: ctx.seed,
      name: 'savannaCelebrations',
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      camera: this.camera,
      fkToWorld: defaultFkToWorld,
      name: 'savannaGroundTruth',
    };
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    this.cheetah?.setPixelRatio(ctx.size.dpr);

    this.applyPose(this.renderPose, 0);
    world.setTime(0);
    this.cheetah?.setTime(0);
    this.cheetah?.setGlow(1);
    this.rig?.setTarget(0, TRACK_Y, 0);
    this.rig?.snap();
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

  protected tick(frame: FrameUpdate): void {
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
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'teleport') teleported = true;
    }

    for (let i = 0; i < 9; i++) this.renderPose[i] = state.pose[i] ?? 0;
    if (teleported) this.flashAt = time;

    const velX = velocityOf(state, frame.prev);
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
  }

  resize(size: SceneSize): void {
    super.resize(size);
    this.cheetah?.setPixelRatio(size.dpr);
  }

  protected drop(): void {
    this.cheetah?.dispose();
    this.cheetah = null;
    this.flashAt = -Infinity;
  }
}

const createScene: SceneFactory = () => new GoldHourSavannaScene();
export default createScene;
