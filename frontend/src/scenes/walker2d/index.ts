import type { GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import {
  WK_L_FOOT,
  WK_L_LEG,
  WK_L_THIGH,
  WK_R_FOOT,
  WK_R_LEG,
  WK_R_THIGH,
  WK_ROOT_X,
  WK_ROOT_Z,
  WK_TORSO_TH,
} from '@sim/envs/walker2d';
import { buildWalker, type WalkerHandle } from './actors';
import { buildWorld, type WorldHandle } from './world';

export const TRACK_Y = 0.95;
const CAM_OFFSET: readonly [number, number, number] = [0.9, 0.35, 3.8];
const FLASH_SECONDS = 0.34;

class StrideFlatsScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'strideFlats';

  private walker: WalkerHandle | null = null;
  private flashAt = -Infinity;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const walker = buildWalker({
      palette: ctx.tokens.palettes.dusk,
      accent: this.accentOf(ctx),
    });
    this.walker = walker;
    this.scene.add(walker.group);
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
      damping: 6,
      follow: 3.4,
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentPos: (state) => {
        this.overlayPos[0] = state.pose[WK_ROOT_X] ?? 0;
        this.overlayPos[1] = state.pose[WK_ROOT_Z] ?? 0;
        this.overlayPos[2] = 0;
        return this.overlayPos;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.walker?.setState(0, 1.25, 0, [0, 0, 0], [0, 0, 0]);
    world.setOrigin(0);
    world.setTime(0);
    this.rig?.snap();
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const walker = this.walker;
    if (!walker) return;

    const state = frame.state;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
    }

    let teleported = false;
    for (const ev of frame.events) if (ev.type === 'teleport') teleported = true;
    if (teleported) this.flashAt = time;

    const rootX = state.pose[WK_ROOT_X] ?? 0;
    walker.setState(
      rootX,
      state.pose[WK_ROOT_Z] ?? 0,
      state.pose[WK_TORSO_TH] ?? 0,
      [state.pose[WK_R_THIGH] ?? 0, state.pose[WK_R_LEG] ?? 0, state.pose[WK_R_FOOT] ?? 0],
      [state.pose[WK_L_THIGH] ?? 0, state.pose[WK_L_LEG] ?? 0, state.pose[WK_L_FOOT] ?? 0],
    );
    world.setOrigin(rootX);

    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.6);
    walker.setGlow(flash);
    if (flash > 0) pool.pulse.fire(0.4 * flash);

    world.setTime(time);
    walker.setTime(time);

    rig.follow(rootX, TRACK_Y, 0);
    if (teleported || frame.scrubbed) rig.snap();
    else rig.update(frame.dtReal);
  }

  protected drop(): void {
    this.walker?.dispose();
    this.walker = null;
    this.flashAt = -Infinity;
  }
}

export default ((() => new StrideFlatsScene()) satisfies SceneFactory);
