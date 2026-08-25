import type { GroundTruthOptions } from '@kit/debug';
import { clampRange, envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { PU_GOAL_X, PU_GOAL_Y, PU_OBJ_X, PU_OBJ_Y } from '@sim/envs/pusher';
import { buildPusher, type PusherHandle } from './actors';
import { buildWorld, PUSH_SCALE, TABLE_Y, type WorldHandle } from './world';

const CAM_BASE: readonly [number, number, number] = [1.4, 3.4, 3.2];
const CAM_LOOK: readonly [number, number, number] = [0.3, TABLE_Y, 0];
const FLASH_SECONDS = 0.34;

class PushBenchScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'pushBench';

  private pusher: PusherHandle | null = null;
  private flashAt = -Infinity;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const pusher = buildPusher({
      palette: ctx.tokens.palettes.afternoon,
      accent: this.accentOf(ctx),
    });
    this.pusher = pusher;
    this.scene.add(pusher.group);
  }

  protected rigOptions(): OrbitRigOptions {
    const home = orbitFromPose(CAM_BASE, CAM_LOOK);
    return {
      target: CAM_LOOK,
      azimuth: home.azimuth,
      polar: home.polar,
      distance: home.distance,
      near: 0.05,
      far: 420,
      rotateSpeed: 0.006,
      zoomSpeed: 0.0016,
      damping: 4,
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentPos: (state) => {
        this.overlayPos[0] = (state.pose[PU_OBJ_X] ?? 0) * PUSH_SCALE;
        this.overlayPos[1] = TABLE_Y + 0.06;
        this.overlayPos[2] = -(state.pose[PU_OBJ_Y] ?? 0) * PUSH_SCALE;
        return this.overlayPos;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.pusher?.setState(new Float32Array(11));
    world.setTime(0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const pusher = this.pusher;
    if (!pusher) return;

    const state = frame.state;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
    }

    let teleported = false;
    for (const ev of frame.events) if (ev.type === 'teleport') teleported = true;
    if (teleported) this.flashAt = time;

    pusher.setState(state.pose);

    const dx = (state.pose[PU_OBJ_X] ?? 0) - (state.pose[PU_GOAL_X] ?? 0);
    const dy = (state.pose[PU_OBJ_Y] ?? 0) - (state.pose[PU_GOAL_Y] ?? 0);
    const near = 1 - clampRange(Math.hypot(dx, dy) / 0.35, 0, 1);

    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.6);
    pusher.setGlow(near + flash);
    if (flash > 0) pool.pulse.fire(0.4 * flash);

    world.setTime(time);
    pusher.setTime(time);
    rig.update(frame.dtReal);
  }

  protected drop(): void {
    this.pusher?.dispose();
    this.pusher = null;
    this.flashAt = -Infinity;
  }
}

export default ((() => new PushBenchScene()) satisfies SceneFactory);
