import type { GroundTruthOptions } from '@kit/debug';
import { clampRange } from '@kit/mix';
import { type OrbitRigOptions } from '@kit/rigs';
import { makeSceneView, type SceneView } from '@kit/view';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { PD_L, PD_MAX_SPEED, PD_MAX_TORQUE, PENDULUM_TH, PENDULUM_THDOT, PENDULUM_TORQUE, wrapPi } from '@sim/envs/pendulum';
import { buildLanternRig, type LanternRigHandle } from './actors';
import { buildWorld, pivotOf, type WorldHandle } from './world';


function torqueOf(state: FrameState): number {
  return state.pose[PENDULUM_TORQUE] ?? 0;
}

export function pendulumCost(theta: number, thetaDot: number, torque: number): number {
  const th = wrapPi(theta);
  return th * th + 0.1 * thetaDot * thetaDot + 0.001 * torque * torque;
}

export const COST_MAX = pendulumCost(Math.PI, PD_MAX_SPEED, PD_MAX_TORQUE);

const SERENITY_GAMMA = 3;

export function serenityOf(theta: number, thetaDot: number, torque = 0): number {
  const reward = 1 - pendulumCost(theta, thetaDot, torque) / COST_MAX;
  return clampRange(reward, 0, 1) ** SERENITY_GAMMA;
}

class LanternBellScene extends StageScene<WorldHandle> {
  private view: SceneView | null = null;
  protected readonly stageName = 'lanternBell';

  private lantern: LanternRigHandle | null = null;
  private pivot: readonly [number, number, number] = [0, 0, 0];
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    this.pivot = pivotOf(ctx.geometry);
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const palette = ctx.tokens.palettes.night;
    const accent = this.accentOf(ctx);

    const lantern = buildLanternRig({
      palette,
      accent,
      pivot: this.pivot,
      maxTorque: PD_MAX_TORQUE,
      maxSpeed: PD_MAX_SPEED,
    });
    this.lantern = lantern;
    this.scene.add(lantern.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-2.4, -2.1, -0.55], max: [2.4, 2.1, 0.55] }, [0, -0.09, 0], { distance: 6.598249768, azimuth: 0.094375296, polar: 1.360000000 });
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentPos: (state) => {
        const th = state.pose[PENDULUM_TH] ?? 0;
        this.overlayPos[0] = this.pivot[0] + PD_L * Math.sin(th);
        this.overlayPos[1] = this.pivot[1] - PD_L * Math.cos(th);
        this.overlayPos[2] = this.pivot[2];
        return this.overlayPos;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.applyState(Math.PI, 0, 0, 0);
    world.setTime(0);
    this.lantern?.setTime(0);
  }

  private applyState(theta: number, thetaDot: number, torque: number, time: number): void {
    const world = this.world;
    const lantern = this.lantern;
    if (!world || !lantern) return;

    lantern.setState(theta, thetaDot, torque);
    const serenity = serenityOf(theta, thetaDot, torque);
    lantern.setGlow(serenity);
    world.setLight(serenity, lantern.bobRoot.position.x, time);

    const flame = lantern.flameWorld;
    world.motes.position.copy(flame);
    const light = world.lanternLight;
    if (light) light.position.copy(flame);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const lantern = this.lantern;
    if (!lantern) return;

    const state = frame.state;

    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
    }

    const theta = state.pose[PENDULUM_TH] ?? 0;
    const thetaDot = state.pose[PENDULUM_THDOT] ?? 0;

    this.applyState(theta, thetaDot, torqueOf(state), time);

    world.setTime(time);
    lantern.setTime(time);

    if (this.view) world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  protected drop(): void {
    this.lantern?.dispose();
    this.lantern = null;
  }
}

export default ((() => new LanternBellScene()) satisfies SceneFactory);
