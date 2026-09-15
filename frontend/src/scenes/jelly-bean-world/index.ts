import * as THREE from 'three';
import type { GroundTruthOptions } from '@kit/debug';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { jbwCollectible, jbwItemAt, JBW_ENERGY, JBW_HEADING, JBW_X, JBW_Y } from '@sim/envs/jellybeanworld';
import { buildForager } from './actors';
import { CollectionLog } from './collectionlog';
import { scentAtCell } from './fieldmath';
import { buildWorld, type ParklandWorld } from './world';
import { makeSceneView, trackPose, type SceneView } from '@kit/view';

class WinterCandyScene extends StageScene<ParklandWorld> {
  protected readonly stageName = 'winterCandyLand';
  private readonly collections = new CollectionLog();
  private readonly position = new THREE.Vector3();
  private readonly groundNormal = new THREE.Vector3();
  private readonly groundTilt = new THREE.Quaternion();
  private readonly eventPosition = new THREE.Vector3();
  private readonly noCone = new Float32Array(0);
  private view!: SceneView;
  private robot!: ReturnType<typeof buildForager>;
  private x = NaN;
  private y = NaN;
  private heading = 0;
  private sensorStep = -1;

  protected buildWorld(ctx: SceneContext): ParklandWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void { this.robot = buildForager(); this.scene.add(this.robot.group); }
  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-6, -1, -6], max: [6, 5, 6] }, [0, 0.4, 0], { distance: 36, azimuth: 0.62, polar: 1.50 }, 2048);
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return { frame: () => ({ originX: this.x, originY: this.position.y, originZ: this.y, agent: [0, 0, 0], cone: this.noCone, coneCount: 0 }) };
  }
  protected start(ctx: SceneContext): void { this.resize(ctx.size); }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<ParklandWorld>): void {
    const step = frame.state.step, x = Math.round(frame.state.pose[JBW_X]!), y = Math.round(frame.state.pose[JBW_Y]!);
    const heading = Math.round(frame.state.pose[JBW_HEADING]!) & 3;
    if (this.collections.sync(frame.eventLog)) world.scent.sync(this.collections.log, this.collections.version);
    this.x = x; this.y = y;
    this.position.set(trackPose(frame, JBW_X), 0, trackPose(frame, JBW_Y));
    this.position.y = world.heightAt(this.position.x, this.position.z);
    const travel = frame.scrubbed ? 0 : this.robot.group.position.distanceTo(this.position);
    this.robot.group.position.copy(this.position);
    const angle = Math.PI / 2 - heading * Math.PI / 2;
    const turn = Math.atan2(Math.sin(angle - this.heading), Math.cos(angle - this.heading)) * (frame.scrubbed ? 1 : 1 - Math.exp(-frame.dtReal * 20));
    this.heading += turn; this.robot.roll(travel, turn);
    this.robot.group.rotation.set(0, this.heading, 0);
    this.groundNormal.set(world.heightAt(this.position.x - 0.25, this.position.z) - world.heightAt(this.position.x + 0.25, this.position.z), 0.5, world.heightAt(this.position.x, this.position.z - 0.25) - world.heightAt(this.position.x, this.position.z + 0.25)).normalize();
    this.robot.group.quaternion.premultiply(this.groundTilt.setFromUnitVectors(THREE.Object3D.DEFAULT_UP, this.groundNormal));
    if (step !== this.sensorStep) {
      this.sensorStep = step;
      const signals = scentAtCell(ctx.seed, this.collections.collected, step, x, y, world.scent.kernel);
      world.agent.value.set(x, y); world.vision.value.fill(0);
      const directions = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
      for (let i = 0; i < 8; i++) {
        const direction = directions[(2 * heading + i) & 7]!, sx = x + direction[0]!, sy = y + direction[1]!, item = jbwItemAt(ctx.seed, sx, sy);
        const energy = jbwCollectible(item) && this.collections.depleted(sx, sy, step) ? 0 : JBW_ENERGY[item]!;
        signals.push(energy); world.vision.value[(direction[1]! + 1) * 3 + direction[0]! + 1] = energy;
      }
      this.robot.sense(signals);
    }
    world.debug.value = this.debugOn ? 1 : 0;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    for (const event of frame.events) if (event.type === 'collection') {
      this.eventPosition.set(event.x, world.heightAt(event.x, event.y) + 0.3, event.y);
      if (event.reward > 0) pool.burst(this.eventPosition, 0xefb7cf, 0.25, time);
      else if (event.reward < 0) pool.sigh(0.25, this.eventPosition, time);
    }
    const coverage = this.view.update(rig, frame.dtReal, [this.position.x, this.position.y + 0.4, this.position.z]);
    world.scent.frame(step, Math.round(coverage.x), Math.round(coverage.z));
    world.setTime(time); world.setView(coverage, this.camera);
    world.items.update(coverage, this.camera, step, this.collections);
  }

  protected drop(): void { this.robot.dispose(); }
}

export default ((() => new WinterCandyScene()) satisfies SceneFactory);
