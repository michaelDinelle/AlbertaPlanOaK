import * as THREE from 'three';
import { gridHeading } from '@kit/cast/cellwalker';
import type { CourierPose } from '@kit/cast/courier';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, SceneSize, ScenePost } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { FORAGER_X, FORAGER_Y } from '@sim/envs/forager';
import { countEvents } from '@sim/events';
import { wordInt } from '@sim/words';
import { formatCount } from '../../format';
import { BERRY_TINT, BURR_TINT, buildWorld, MEMORY_STEPS, TRAIL, type PrairieWorld } from './world';

const towards = (from: number, to: number, side: number): number => {
  const d = (to - from) % side;
  return d > side / 2 ? d - side : d < -side / 2 ? d + side : d;
};

class PrairieScene extends StageScene<PrairieWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 8;
  protected override poolOptions() { return { burstSize: 90 }; }
  protected readonly stageName = 'foragerPrairie';
  private view!: SceneView;
  private readonly burstAt = new THREE.Vector3();
  private readonly cellOrigin = new THREE.Vector3();
  private readonly noCone = new Float32Array(0);
  private readonly from: CourierPose = { x: 0, y: 0, heading: 0 };
  private readonly turned: CourierPose = { x: 0, y: 0, heading: 0 };
  private readonly to: CourierPose = { x: 0, y: 0, heading: 0 };
  private readonly trailX = new Float64Array(TRAIL);
  private readonly trailZ = new Float64Array(TRAIL);
  private readonly trailStep = new Int32Array(TRAIL).fill(-1);
  private ux = NaN;
  private uz = NaN;
  private rx = 0;
  private rz = 0;
  private shown = -1;
  private step = 0;
  private tallyStep = -1;
  private gathered = 0;
  private stung = 0;

  protected buildWorld(ctx: SceneContext): PrairieWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-8, -1, -8], max: [8, 4, 8] }, [0, 0.3, 0], { distance: 14, azimuth: 0.55, polar: 1.0 }, 768);
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return { floor: [9, 9], frame: () => ({ originX: this.ux, originY: this.world!.heightAt(this.ux, this.uz), originZ: this.uz, agent: this.cellOrigin, cone: this.noCone, coneCount: 0 }) };
  }
  protected start(_ctx: SceneContext, world: PrairieWorld): void {
    const spec: LegendSpec = {
      title: 'FORAGE',
      rows: [
        { key: 'berries', label: 'berries', tint: BERRY_TINT, value: () => (world.goodAt(this.step) === 1 ? '+1' : '−1') },
        { key: 'burrs', label: 'burrs', tint: BURR_TINT, value: () => (world.goodAt(this.step) === 2 ? '+1' : '−1') },
        { key: 'gathered', label: 'gathered', tint: BERRY_TINT, value: () => formatCount(this.gathered) },
        { key: 'stung', label: 'stung', tint: BURR_TINT, value: () => formatCount(this.stung) },
      ],
    };
    if (world.switching) {
      spec.note = () => `swaps in ${formatCount(world.period - this.step % world.period)} steps`;
      spec.bar = () => (this.step % world.period) / world.period;
    }
    this.legend = spec;
  }

  private remember(x: number, z: number, step: number): void {
    const slot = ((step % TRAIL) + TRAIL) % TRAIL;
    this.trailX[slot] = x; this.trailZ[slot] = z; this.trailStep[slot] = step;
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<PrairieWorld>): void {
    const step = frame.state.step, side = world.side, x = frame.state.pose[FORAGER_X]!, y = frame.state.pose[FORAGER_Y]!;
    this.step = step;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed || Number.isNaN(this.ux)) {
      pool.clear();
      this.ux = Number.isNaN(this.ux) ? x : x + side * Math.round((this.ux - x) / side);
      this.uz = Number.isNaN(this.uz) ? y : y + side * Math.round((this.uz - y) / side);
      this.rx = x; this.rz = y;
      this.from.x = this.to.x = this.ux; this.from.y = this.to.y = this.uz; this.from.heading = this.to.heading;
      world.forget();
      this.trailStep.fill(-1);
      let px = this.ux, pz = this.uz;
      for (let i = step; i >= Math.max(0, step - TRAIL); i--) {
        const f = frame.frameAt(i);
        this.remember(px, pz, i);
        if (i > 0) { const before = frame.frameAt(i - 1); px -= towards(wordInt(before.words[0]!), wordInt(f.words[0]!), side); pz -= towards(wordInt(before.words[1]!), wordInt(f.words[1]!), side); }
      }
      for (let i = Math.max(0, step - MEMORY_STEPS); i <= step; i++) {
        const f = frame.frameAt(i);
        world.remember(f.step, wordInt(f.words[0]!), wordInt(f.words[1]!), f.observations);
      }
      this.shown = -1;
    } else {
      for (let i = this.shown + 1; i <= step; i++) {
        const f = frame.frameAt(i), nx = wordInt(f.words[0]!), nz = wordInt(f.words[1]!);
        const dx = towards(this.rx, nx, side), dz = towards(this.rz, nz, side);
        this.from.x = this.ux; this.from.y = this.uz; this.from.heading = this.to.heading;
        this.ux += dx; this.uz += dz; this.rx = nx; this.rz = nz;
        this.to.x = this.ux; this.to.y = this.uz; this.to.heading = gridHeading(dx, dz, this.from.heading);
        this.remember(this.ux, this.uz, i);
        world.remember(f.step, nx, nz, f.observations);
        if (f.reward !== 0) {
          this.burstAt.set(this.ux, world.heightAt(this.ux, this.uz) + 0.45, this.uz);
          if (f.reward > 0) pool.burst(this.burstAt, BERRY_TINT, 0.5, time);
          else { pool.sigh(0.5, this.burstAt, time); pool.ripple(this.burstAt, BURR_TINT, 0.4, time); }
        }
        if (world.goodAt(f.step) !== world.goodAt(f.step - 1)) { pool.pulse.fire(1); pool.ripple(this.burstAt.set(this.ux, world.heightAt(this.ux, this.uz) + 0.2, this.uz), ctx.tokens.celebrationGold, 1, time); }
      }
    }
    if (step < this.tallyStep) { this.tallyStep = -1; this.gathered = 0; this.stung = 0; }
    if (step > this.tallyStep) {
      this.gathered += countEvents(frame.eventLog, 'goal', this.tallyStep, step);
      this.stung += countEvents(frame.eventLog, 'harm', this.tallyStep, step);
      this.tallyStep = step;
    }
    if (this.shown !== step) world.show(step, this.ux, this.uz);
    this.shown = step;
    const alpha = frame.position - Math.floor(frame.position), turning = this.from.heading !== this.to.heading;
    const ground = (px: number, pz: number): number => world.heightAt(px, pz);
    if (turning) { this.turned.x = this.from.x; this.turned.y = this.from.y; this.turned.heading = this.to.heading; }
    if (turning && alpha < 0.5) world.courier.drive(this.from, this.turned, alpha * 2, ground, frame.scrubbed);
    else world.courier.drive(turning ? this.turned : this.from, this.to, turning ? alpha * 2 - 1 : alpha, ground, frame.scrubbed);
    const bx = world.courier.group.position.x, bz = world.courier.group.position.z, by = world.courier.group.position.y;
    world.lens.position.set(bx, by + 0.1, bz);
    world.lamp.position.set(bx, by + 2.6, bz);
    let shownPrints = 0;
    for (let slot = 0; slot < TRAIL; slot++) {
      const at = this.trailStep[slot]!;
      if (at < 0 || at > step) continue;
      const age = (step - at) / TRAIL;
      if (age >= 1) continue;
      world.trail.transform.position.set(this.trailX[slot]!, world.heightAt(this.trailX[slot]!, this.trailZ[slot]!), this.trailZ[slot]!);
      world.trail.transform.scale.setScalar(1 - age);
      world.trail.write(shownPrints++);
    }
    world.trail.commit(shownPrints);
    world.lights.setShadowFocus({ x: Math.round(bx), y: by, z: Math.round(bz) }, 16);
    world.setTime(time);
    world.setView(this.view.update(rig, frame.dtReal, [bx, by + 0.3, bz]), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.ux = NaN; this.uz = NaN; this.shown = -1; this.to.heading = 0; this.tallyStep = -1; this.gathered = 0; this.stung = 0; }
}

export default ((() => new PrairieScene()) satisfies SceneFactory);
