import { POSE_LEN } from '@sim/record';
import type { GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { HU_X, HU_Y, HU_Z } from '@sim/envs/humanoid';
import { buildHumanoid, type HumanoidHandle } from './actors';
import { buildWorld, type WorldHandle } from './world';

export const TRACK_Y = 1;
const CAM_OFFSET: readonly [number, number, number] = [1.8, 0.7, 4];
const FLASH_SECONDS = 0.34;

class StrideFieldScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'strideField';

  private humanoid: HumanoidHandle | null = null;
  private flashAt = -Infinity;
  private readonly renderPose = new Float32Array(POSE_LEN);
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const humanoid = buildHumanoid({
      palette: ctx.tokens.palettes.noon,
      accent: this.accentOf(ctx),
      name: 'fieldHumanoid',
    });
    this.humanoid = humanoid;
    this.scene.add(humanoid.group);
  }

  protected rigOptions(): OrbitRigOptions {
    const home = orbitFromPose(CAM_OFFSET, [0, 0, 0]);
    return {
      target: [0, TRACK_Y, 0],
      azimuth: home.azimuth,
      polar: home.polar,
      distance: home.distance,
      near: 0.05,
      far: 420,
      rotateSpeed: 0.006,
      zoomSpeed: 0.0016,
      damping: 5,
      follow: 3,
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentPos: (state) => {
        this.overlayPos[0] = state.pose[HU_X] ?? 0;
        this.overlayPos[1] = state.pose[HU_Z] ?? 0;
        this.overlayPos[2] = -(state.pose[HU_Y] ?? 0);
        return this.overlayPos;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.renderPose.fill(0);
    this.renderPose[HU_Z] = 1.4;
    this.renderPose[3] = 1;
    this.humanoid?.setPose(this.renderPose);
    world.setOrigin(0, 0);
    world.setTime(0);
    this.rig?.snap();
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const humanoid = this.humanoid;
    if (!humanoid) return;

    const state = frame.state;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
    }

    let teleported = false;
    for (const ev of frame.events) if (ev.type === 'teleport') teleported = true;
    if (teleported) this.flashAt = time;

    for (let i = 0; i < POSE_LEN; i++) this.renderPose[i] = state.pose[i] ?? 0;
    humanoid.setPose(this.renderPose);

    const rootX = this.renderPose[HU_X] ?? 0;
    const rootZ = -(this.renderPose[HU_Y] ?? 0);
    world.setOrigin(rootX, rootZ);

    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.6);
    humanoid.setGlow(flash);
    if (flash > 0) pool.pulse.fire(0.4 * flash);

    world.setTime(time);
    humanoid.setTime(time);

    rig.follow(rootX, TRACK_Y, rootZ);
    if (teleported || frame.scrubbed) rig.snap();
    else rig.update(frame.dtReal);
  }

  protected drop(): void {
    this.humanoid?.dispose();
    this.humanoid = null;
    this.flashAt = -Infinity;
  }
}

export default ((() => new StrideFieldScene()) satisfies SceneFactory);
