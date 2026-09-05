import type { GroundTruthOptions } from '@kit/debug';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { ACROBOT_TH1, ACROBOT_TH2 } from '@sim/envs/acrobot';
import { makeSceneView, type SceneView } from '@kit/view';
import { buildArm, type ArmHandle } from './actors';
import { buildWorld, pivotAnchor, type WorldHandle } from './world';

class AlpineLakeScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'alpineLake';
  private view: SceneView | null = null;
  private arm: ArmHandle | null = null;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): Promise<WorldHandle> {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    this.arm = buildArm(pivotAnchor(ctx.geometry));
    this.scene.add(this.arm.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-2.85, -2.5, -0.9], max: [2.85, 2.2, 0.3] }, [0, -0.6, 0], { distance: 7.2, azimuth: 0.12, polar: 1.48 });
    return { ...this.view.options, far: 5000, limits: { ...this.view.options.limits, maxPolar: 1.48 } };
  }

  protected overlayOptions(): GroundTruthOptions {
    return { agentPos: () => {
      if (this.arm) this.arm.tipWorld.toArray(this.overlayPos);
      return this.overlayPos;
    } };
  }

  protected start(): void {
    this.arm?.setState(0, 0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const arm = this.arm;
    if (!arm) return;
    arm.setState(frame.state.pose[ACROBOT_TH1] ?? 0, frame.state.pose[ACROBOT_TH2] ?? 0);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    for (const event of frame.events) if (event.type === 'goal') pool.burst(arm.tipWorld, 0x98e1cd, 0.35, time);
    world.setTime(time);
    if (this.view) world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  protected drop(): void {
    this.arm?.dispose();
    this.arm = null;
  }
}

export default ((() => new AlpineLakeScene()) satisfies SceneFactory);
