import { Matrix4 } from 'three';
import { defaultFkToWorld, type GroundTruthOptions } from '@kit/debug';
import { type OrbitRigOptions } from '@kit/rigs';
import { makeSceneView, trackPose, type SceneView } from '@kit/view';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { FK_FLOATS, FK_POINT_INDEX } from '@sim/fk/cheetah';
import type { CheetahGeometry } from '@sim/types';
import { buildCheetah, type CheetahHandle } from './actors';
import { prairieHeight, buildWorld, type WorldHandle } from './world';

export const TRACK_Y = 0.52;

class AlbertaPrairieScene extends StageScene<WorldHandle> {
  private view: SceneView | null = null;
  protected readonly stageName = 'albertaPrairieSunrise';

  private cheetah: CheetahHandle | null = null;
  private fkSource!: CheetahGeometry;

  private readonly projection = new Matrix4();
  private readonly fkOut = new Float32Array(FK_FLOATS);
  private readonly renderPose = new Float32Array(9);

  protected buildWorld(ctx: SceneContext): Promise<WorldHandle> {
    if (!ctx.geometry.cheetah) throw new Error('half cheetah FK geometry required');
    this.fkSource = ctx.geometry.cheetah;
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(): void {
    this.cheetah = buildCheetah(this.fkSource);
    this.scene.add(this.cheetah.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-2.6, 0, -1.7], max: [2.6, 1.6, 1.7] }, [0, TRACK_Y, 0], { distance: 4.4, azimuth: 0.265, polar: 1.44 });
    return { ...this.view.options, limits: { ...this.view.options.limits, maxPolar: 1.48 } };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      fkToWorld: defaultFkToWorld,
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.applyPose(this.renderPose);
    world.setTime(0);
    this.rig?.setTarget(0, TRACK_Y, 0);
    this.rig?.snap();
  }

  private applyPose(pose: Float32Array): void {
    const cheetah = this.cheetah;
    const world = this.world;
    if (!cheetah || !world) return;
    this.fkSource.fk(pose, this.fkOut);
    cheetah.setPose(this.fkOut);
    const rootX = this.fkOut[FK_POINT_INDEX.o1 * 2] ?? 0;
    world.setShadowTarget(rootX);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    for (let i = 0; i < this.renderPose.length; i++) this.renderPose[i] = frame.state.pose[i] ?? 0;
    this.applyPose(this.renderPose);
    world.setTime(time);
    const followX = trackPose(frame, 0);
    if (this.view) {
      const coverage = this.view.update(rig, frame.dtReal, [followX, TRACK_Y, 0]);
      this.camera.position.y = Math.max(this.camera.position.y, prairieHeight(this.camera.position.x, this.camera.position.z) + 1.2);
      this.camera.lookAt(rig.target);
      this.camera.far = coverage.far = Math.max(coverage.far, 5000);
      this.camera.updateProjectionMatrix();
      this.camera.updateMatrixWorld();
      coverage.frustum.setFromProjectionMatrix(this.projection.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse));
      world.setView(coverage, this.camera);
    }
  }

  protected drop(): void {
    this.cheetah?.dispose();
    this.cheetah = null;
  }
}

export default ((() => new AlbertaPrairieScene()) satisfies SceneFactory);
