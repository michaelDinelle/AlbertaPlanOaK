import * as THREE from 'three';
import type { GroundTruthOptions } from '@kit/debug';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, SceneSize, ScenePost } from '@kit/types';
import { JBW_BANANA, JBW_COLOR, JBW_HEADING, JBW_JELLY, JBW_ONION, JBW_TRUFFLE, JBW_X, JBW_Y } from '@sim/envs/jbw';
import { parameter } from '../../data/envs';
import { formatCount } from '../../format';
import { buildCourier } from '@kit/cast/courier';
import { CollectionLog } from './collectionlog';
import { buildWorld, type ParklandWorld } from './world';
import { makeSceneView, type SceneView } from '@kit/view';

const itemTint = (item: number): number => { const c = JBW_COLOR[item]!; return new THREE.Color().setRGB(c[0], c[1], c[2], THREE.LinearSRGBColorSpace).getHex(); };

class WinterCandyScene extends StageScene<ParklandWorld> {
  readonly post: ScenePost = { bloom: 0.38, fog: { from: ['afternoon', 'fog'], blend: [[['dusk', 'fog'], 0.45]] } };
  readonly controlHz = 8;
  protected override poolOptions() { return { burstSize: 108 }; }
  protected readonly stageName = 'winterCandyLand';
  private readonly collections = new CollectionLog();
  private readonly position = new THREE.Vector3();
  private readonly eventPosition = new THREE.Vector3();
  private readonly noCone = new Float32Array(0);
  private readonly cellOrigin = new THREE.Vector3();
  private view!: SceneView;
  private robot!: ReturnType<typeof buildCourier>;
  private x = NaN;
  private y = NaN;
  private sensorStep = -1;
  private readonly from = { x: 0, y: 0, heading: 0 };
  private readonly to = { x: 0, y: 0, heading: 0 };
  private step = 0;
  private period = 1;
  private polarity(): number { return (Math.floor(this.step / this.period) & 1) === 0 ? 1 : -1; }
  override legend: LegendSpec = {
    title: 'REWARD',
    rows: [
      { key: 'jelly', label: 'jelly', tint: itemTint(JBW_JELLY), value: () => (this.polarity() > 0 ? '+1' : '−1') },
      { key: 'onion', label: 'onion', tint: itemTint(JBW_ONION), value: () => (this.polarity() > 0 ? '−1' : '+1') },
      { key: 'banana', label: 'banana', tint: itemTint(JBW_BANANA), value: () => '0' },
      { key: 'truffle', label: 'truffle', tint: itemTint(JBW_TRUFFLE), value: () => 'scent only' },
    ],
    note: () => `flips in ${formatCount(this.period - this.step % this.period)} steps`,
    bar: () => (this.step % this.period) / this.period,
  };

  protected buildWorld(ctx: SceneContext): ParklandWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(ctx: SceneContext): void { this.robot = buildCourier({ name: 'parklandCourier', tint: this.accentOf(ctx), overgrowth: 0.4 }); this.scene.add(this.robot.group); }
  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-6, -1, -6], max: [6, 5, 6] }, [0, 0.4, 0], { distance: 36, azimuth: 0.62, polar: 1.50 }, 2048);
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return { floor: [9, 9], frame: () => ({ originX: this.x, originY: this.position.y, originZ: this.y, agent: this.cellOrigin, cone: this.noCone, coneCount: 0 }) };
  }
  protected start(ctx: SceneContext): void { this.period = parameter(ctx.environment, 'polarityPeriod'); this.resize(ctx.size); }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<ParklandWorld>): void {
    const step = frame.state.step, x = Math.round(frame.state.pose[JBW_X]!), y = Math.round(frame.state.pose[JBW_Y]!);
    this.step = step;
    const heading = Math.round(frame.state.pose[JBW_HEADING]!) & 3;
    if (this.collections.sync(frame.eventLog)) world.scent.sync(this.collections.log, this.collections.version);
    this.x = x; this.y = y;
    if (frame.scrubbed || frame.state.reset || step === 0) { this.from.x = x; this.from.y = y; this.from.heading = heading; }
    else { const before = frame.poseAt(step - 1); this.from.x = Math.round(before[JBW_X]!); this.from.y = Math.round(before[JBW_Y]!); this.from.heading = Math.round(before[JBW_HEADING]!) & 3; }
    this.to.x = x; this.to.y = y; this.to.heading = heading;
    this.robot.drive(this.from, this.to, frame.position - Math.floor(frame.position), (px, pz) => world.heightAt(px, pz), frame.scrubbed);
    this.position.copy(this.robot.group.position);
    world.lights.setShadowFocus({ x: Math.round(this.position.x), y: this.position.y, z: Math.round(this.position.z) }, 16);
    if (frame.scrubbed || step !== this.sensorStep) {
      this.sensorStep = step;
      const observations = frame.state.observations;
      const signals = Array.from(observations.subarray(8, 11));
      world.agent.value.set(x, y); world.vision.value.fill(0);
      const directions = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
      for (let i = 0; i < 8; i++) {
        const direction = directions[(2 * heading + i) & 7]!;
        const energy = observations[i]!;
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

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.robot.dispose(); }
}

export default ((() => new WinterCandyScene()) satisfies SceneFactory);
