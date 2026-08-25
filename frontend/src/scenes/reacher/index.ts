import type { GroundTruthOptions } from '@kit/debug';
import { clampRange, envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { RE_JOINT0, RE_JOINT1, RE_TARGET_X, RE_TARGET_Y } from '@sim/envs/reacher';
import { buildArm, LINK0, LINK1, type ArmHandle } from './actors';
import { buildWorld, REACH_SCALE, TABLE_Y, type WorldHandle } from './world';

const CAM_BASE: readonly [number, number, number] = [0.9, 4.4, 3.1];
const CAM_LOOK: readonly [number, number, number] = [0, TABLE_Y, 0];
const FLASH_SECONDS = 0.34;

class ReachTableScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'reachTable';

  private arm: ArmHandle | null = null;
  private flashAt = -Infinity;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const arm = buildArm({
      palette: ctx.tokens.palettes.night,
      accent: this.accentOf(ctx),
    });
    this.arm = arm;
    this.scene.add(arm.group);
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
        const q0 = state.pose[RE_JOINT0] ?? 0;
        const q1 = q0 + (state.pose[RE_JOINT1] ?? 0);
        this.overlayPos[0] = LINK0 * Math.cos(q0) + LINK1 * Math.cos(q1);
        this.overlayPos[1] = TABLE_Y + 0.08;
        this.overlayPos[2] = -(LINK0 * Math.sin(q0) + LINK1 * Math.sin(q1));
        return this.overlayPos;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.arm?.setState(0, 0, 0.1, 0.1);
    world.setRingGlow(0);
    world.setTime(0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const arm = this.arm;
    if (!arm) return;

    const state = frame.state;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
    }

    let teleported = false;
    for (const ev of frame.events) if (ev.type === 'teleport') teleported = true;
    if (teleported) this.flashAt = time;

    arm.setState(
      state.pose[RE_JOINT0] ?? 0,
      state.pose[RE_JOINT1] ?? 0,
      state.pose[RE_TARGET_X] ?? 0,
      state.pose[RE_TARGET_Y] ?? 0,
    );

    const dx = arm.fingertip.x - arm.target.position.x;
    const dz = arm.fingertip.z - arm.target.position.z;
    const near = 1 - clampRange(Math.hypot(dx, dz) / (0.1 * REACH_SCALE), 0, 1);

    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.6);
    arm.setGlow(near + flash);
    world.setRingGlow(near);
    if (flash > 0) pool.pulse.fire(0.4 * flash);

    world.setTime(time);
    arm.setTime(time);
    rig.update(frame.dtReal);
  }

  protected drop(): void {
    this.arm?.dispose();
    this.arm = null;
    this.flashAt = -Infinity;
  }
}

export default ((() => new ReachTableScene()) satisfies SceneFactory);
