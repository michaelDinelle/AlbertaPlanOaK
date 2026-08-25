import type { GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { ANT_HIP0, ANT_QW, ANT_QX, ANT_QY, ANT_QZ, ANT_X, ANT_Y, ANT_Z } from '@sim/envs/ant';
import { buildAnt, type AntHandle } from './actors';
import { buildWorld, type WorldHandle } from './world';

export const TRACK_Y = 0.5;
const CAM_OFFSET: readonly [number, number, number] = [1.9, 2.1, 3.6];
const FLASH_SECONDS = 0.34;

class AntBasinScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'antBasin';

  private ant: AntHandle | null = null;
  private flashAt = -Infinity;
  private readonly jointScratch = new Float32Array(8);
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const ant = buildAnt({
      palette: ctx.tokens.palettes.dawn,
      accent: this.accentOf(ctx),
    });
    this.ant = ant;
    this.scene.add(ant.group);
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
        this.overlayPos[0] = state.pose[ANT_X] ?? 0;
        this.overlayPos[1] = state.pose[ANT_Z] ?? 0;
        this.overlayPos[2] = -(state.pose[ANT_Y] ?? 0);
        return this.overlayPos;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.ant?.setState(0, 0, 0.55, 1, 0, 0, 0, this.jointScratch);
    world.setOrigin(0, 0);
    world.setTime(0);
    this.rig?.snap();
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const ant = this.ant;
    if (!ant) return;

    const state = frame.state;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      this.flashAt = -Infinity;
    }

    let teleported = false;
    for (const ev of frame.events) if (ev.type === 'teleport') teleported = true;
    if (teleported) this.flashAt = time;

    for (let i = 0; i < 8; i++) this.jointScratch[i] = state.pose[ANT_HIP0 + i] ?? 0;
    const rootX = state.pose[ANT_X] ?? 0;
    const rootZ = -(state.pose[ANT_Y] ?? 0);
    ant.setState(
      rootX,
      state.pose[ANT_Y] ?? 0,
      state.pose[ANT_Z] ?? 0,
      state.pose[ANT_QW] ?? 1,
      state.pose[ANT_QX] ?? 0,
      state.pose[ANT_QY] ?? 0,
      state.pose[ANT_QZ] ?? 0,
      this.jointScratch,
    );
    world.setOrigin(rootX, rootZ);

    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.6);
    ant.setGlow(flash);
    if (flash > 0) pool.pulse.fire(0.4 * flash);

    world.setTime(time);
    ant.setTime(time);

    rig.follow(rootX, TRACK_Y, rootZ);
    if (teleported || frame.scrubbed) rig.snap();
    else rig.update(frame.dtReal);
  }

  protected drop(): void {
    this.ant?.dispose();
    this.ant = null;
    this.flashAt = -Infinity;
  }
}

export default ((() => new AntBasinScene()) satisfies SceneFactory);
