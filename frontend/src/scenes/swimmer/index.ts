import type { GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { SW_JOINT1, SW_JOINT2, SW_ROOT_X, SW_ROOT_Y, SW_YAW } from '@sim/envs/swimmer';
import { buildSwimmer, type SwimmerHandle } from './actors';
import { buildWorld, SWIM_Y, type WorldHandle } from './world';

const CAM_OFFSET: readonly [number, number, number] = [1.6, 4.2, 4.6];
const FLASH_SECONDS = 0.34;

class SwimLagoonScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'swimLagoon';

  private swimmer: SwimmerHandle | null = null;
  private flashAt = -Infinity;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const swimmer = buildSwimmer({
      palette: ctx.tokens.palettes.underwater,
      accent: this.accentOf(ctx),
    });
    this.swimmer = swimmer;
    this.scene.add(swimmer.group);
  }

  protected rigOptions(): OrbitRigOptions {
    const home = orbitFromPose(CAM_OFFSET, [0, SWIM_Y, 0]);
    return {
      target: [0, SWIM_Y, 0],
      azimuth: home.azimuth,
      polar: home.polar,
      distance: home.distance,
      near: 0.05,
      far: 420,
      rotateSpeed: 0.006,
      zoomSpeed: 0.0016,
      damping: 5,
      follow: 3.2,
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentPos: (state) => {
        this.overlayPos[0] = state.pose[SW_ROOT_X] ?? 0;
        this.overlayPos[1] = SWIM_Y;
        this.overlayPos[2] = -(state.pose[SW_ROOT_Y] ?? 0);
        return this.overlayPos;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.swimmer?.setState(0, 0, 0, 0, 0);
    world.setOrigin(0, 0);
    world.setTime(0);
    this.rig?.snap();
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const swimmer = this.swimmer;
    if (!swimmer) return;

    const state = frame.state;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
    }

    let teleported = false;
    for (const ev of frame.events) if (ev.type === 'teleport') teleported = true;
    if (teleported) this.flashAt = time;

    const rootX = state.pose[SW_ROOT_X] ?? 0;
    const rootZ = -(state.pose[SW_ROOT_Y] ?? 0);
    swimmer.setState(
      rootX,
      state.pose[SW_ROOT_Y] ?? 0,
      state.pose[SW_YAW] ?? 0,
      state.pose[SW_JOINT1] ?? 0,
      state.pose[SW_JOINT2] ?? 0,
    );
    world.setOrigin(rootX, rootZ);

    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.6);
    swimmer.setGlow(flash);
    if (flash > 0) pool.pulse.fire(0.4 * flash);

    world.setTime(time);
    swimmer.setTime(time);

    rig.follow(rootX, SWIM_Y, rootZ);
    if (teleported || frame.scrubbed) rig.snap();
    else rig.update(frame.dtReal);
  }

  protected drop(): void {
    this.swimmer?.dispose();
    this.swimmer = null;
    this.flashAt = -Infinity;
  }
}

export default ((() => new SwimLagoonScene()) satisfies SceneFactory);
