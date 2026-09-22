import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { brazier, relief } from '@kit/props';
import { flameMaterial, makeTerrace, yawToward } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { BANDIT_ACTION, banditMean } from '@sim/envs/bandit';
import { parameter } from '../../data/envs';
import { formatSigned } from '../../format';

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const arms = parameter(ctx.environment, 'arms'), drift = parameter(ctx.environment, 'drift');
  const t = makeTerrace(ctx, scene, { mood: 'dusk', halfX: 6.5, halfZ: 4.5, ground: [0x6b6a43, 0x9c9a5c], landscape: {
    sky: { sunSize: 0.05, sunHdr: 7, sunGlow: 0.3, sunGlowPower: 24, topSpread: 0.62, bottomSpread: 0.3 },
    light: { pointCount: 2, pointDistance: 7 },
    weather: { preset: 'embers', count: 120 + arms * 12, seed: ctx.seed, size: 1.3, intensity: 2.2, bounds: { y: 1.6, halfX: 6, halfY: 1.6, halfZ: 3.5 } },
  } });
  const { world, rig, m, accent } = t;
  rig.sun.intensity = 1.9; rig.hemi.intensity = 0.55;
  rig.points.forEach((point, i) => { point.color.set(i === 0 ? 0xffa34d : accent); point.intensity = i === 0 ? 4 : 3; point.position.set(i === 0 ? 0 : 0, 1.6, i === 0 ? -2.5 : 1.5); });
  const wall = relief(world, 14, 2.2, 0.7, m);
  wall.position.set(0, 0, -4.7);
  world.add(wall);
  const flameGeometry = world.own(new THREE.SphereGeometry(0.17, 12, 10).scale(1, 2.4, 1).translate(0, 0.32, 0));
  const pitch = arms > 4 ? 1.9 : 2.8;
  const braziers = Array.from({ length: arms }, (_, i) => {
    const x = (i - (arms - 1) / 2) * pitch, z = -2.6 + Math.abs(i - (arms - 1) / 2) * 0.35;
    const b = brazier(world, accent, m);
    b.group.position.set(x, 0, z);
    world.add(b.group);
    const material = world.own(flameMaterial(0xff8a3d, 1.5));
    const flame = new THREE.Mesh(flameGeometry, material);
    flame.position.set(x, 0.92, z);
    world.add(flame);
    return { ...b, flame, material, x, z, top: new THREE.Vector3(x, 1.4, z) };
  });
  const sentinel = world.own(buildSentinel({ name: 'banditPuller', tint: accent }));
  sentinel.place(0, 0, 2);
  world.add(sentinel.group);
  const extra = { land: t.land, lights: rig, braziers, sentinel, drift, accent };
  return world.seal<WorldCore & typeof extra>(extra);
}

type HallWorld = ReturnType<typeof buildWorld>;

class BrazierHallScene extends StageScene<HallWorld> {
  readonly post: ScenePost = { bloom: 0.36, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'brazierHall';
  private view!: SceneView;
  private lastStep = -1;
  private pullAt = -Infinity;
  private pulled = 0;
  private readonly ledger = { pulled: 0, best: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'BRAZIERS',
    rows: [
      { key: 'pulled', label: 'pulled', tint: 0xffc857, value: () => `brazier ${this.ledger.pulled + 1} of ${this.world!.braziers.length}` },
      { key: 'best', label: 'hottest', tint: 0xff8a3d, value: () => `brazier ${this.ledger.best + 1}` },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): HallWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-7, -0.5, -5], max: [7, 3, 4] }, [0, 0.8, -0.6], { distance: 12, azimuth: 0.32, polar: 1.12 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [14, 10], agentPos: () => this.world!.sentinel.group.position }; }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<HallWorld>): void {
    const step = frame.state.step, raw = frame.frameAt(step), arms = world.braziers.length;
    let lo = Infinity, hi = -Infinity, best = 0;
    for (let i = 0; i < arms; i++) { const mean = banditMean(raw, i); if (mean > hi) best = i; lo = Math.min(lo, mean); hi = Math.max(hi, mean); }
    const span = hi - lo || 1;
    const pulled = Math.min(arms - 1, Math.max(0, Math.round(frame.state.pose[BANDIT_ACTION]!)));
    this.ledger.pulled = pulled; this.ledger.best = best; this.ledger.reward = frame.state.reward;
    if (step !== this.lastStep) {
      this.lastStep = step; this.pulled = pulled; this.pullAt = time;
      const target = world.braziers[pulled]!;
      if (!frame.scrubbed) {
        if (frame.state.reward > 0.8) pool.burst(target.top, world.accent, Math.min(1, frame.state.reward * 0.4), time);
        else if (frame.state.reward < -0.8) pool.sigh(Math.min(1, -frame.state.reward * 0.3), target.top, time);
      }
    }
    const flare = envelope(time - this.pullAt, 1.2, 2.4);
    world.braziers.forEach((b, i) => {
      const height = 0.3 + 0.7 * (banditMean(raw, i) - lo) / span;
      const flicker = 1 + 0.08 * Math.sin(time * 9 + i * 1.7) + (world.drift > 0 ? 0.12 * Math.sin(time * 0.6 + i) : 0);
      b.flame.scale.set(0.8 + height * 0.5, height * flicker + (i === this.pulled ? flare * 0.6 : 0), 0.8 + height * 0.5);
      b.flame.rotation.z = world.drift > 0 ? 0.18 * Math.sin(time * 0.7 + i * 2.1) : 0.04 * Math.sin(time * 5 + i);
      b.lit(height * 0.6 + (i === this.pulled ? flare * 1.5 : 0));
    });
    const target = world.braziers[this.pulled]!;
    const light = world.lights.points[0];
    if (light) { light.position.set(target.x, 1.3, target.z); light.intensity = 2 + 6 * flare; }
    world.sentinel.idle(time);
    world.sentinel.face(yawToward(0, 2, target.x, target.z));
    world.sentinel.glow(flare);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.pullAt = -Infinity; }
}

export default ((() => new BrazierHallScene()) satisfies SceneFactory);
