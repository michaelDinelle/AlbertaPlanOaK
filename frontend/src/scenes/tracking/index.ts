import * as THREE from 'three';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import { setEmissiveIntensity } from '@kit/materials';
import { TRACK_ESTIMATE, TRACK_OBSERVED, TRACK_TARGET } from '@sim/envs/tracking';
import { formatSigned } from '../../format';
import { buildWorld, ICE_LEVEL, LAKE_HALF_X, LAKE_HALF_Z, STAR_Y, TRACK_SCALE, WISP_Z, type FrozenLakeWorld } from './world';

class FrozenLakeScene extends StageScene<FrozenLakeWorld> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['dawn', 'fog'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'frozenLake';
  private view!: SceneView;
  private wisp!: Wisp;
  private readonly ledger = { target: 0, observed: 0, estimate: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'TRACK',
    rows: [
      { key: 'target', label: 'star', tint: 0xffd166, value: () => formatSigned(this.ledger.target) },
      { key: 'observed', label: 'seen through snow', tint: 0xd9a8ff, value: () => formatSigned(this.ledger.observed) },
      { key: 'estimate', label: 'lantern', tint: 0x5ee0ff, value: () => formatSigned(this.ledger.estimate) },
      { key: 'miss', label: 'miss', tint: 0x7a95c4, value: () => formatSigned(this.ledger.estimate - this.ledger.target) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): FrozenLakeWorld { return buildWorld(ctx, this.scene); }

  protected buildActors(_ctx: SceneContext, world: FrozenLakeWorld): void {
    this.wisp = new Wisp({ color: world.accent, radius: 0.42, height: 0.9, skirt: true, ribbon: false, silhouette: false, name: 'lanternWisp' });
    this.wisp.group.position.z = WISP_Z;
    this.scene.add(this.wisp.group);
  }

  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-LAKE_HALF_X, -0.5, -LAKE_HALF_Z], max: [LAKE_HALF_X, STAR_Y + 1, LAKE_HALF_Z] }, [0, 2, -0.8], { distance: 12, azimuth: 0.08, polar: 1.22 });
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const at = new THREE.Vector3();
    return {
      floor: [LAKE_HALF_X * 2, LAKE_HALF_Z * 2], profileRange: [-LAKE_HALF_X, LAKE_HALF_X], agentPos: (state) => at.set((state.pose[TRACK_ESTIMATE] ?? 0) * TRACK_SCALE, ICE_LEVEL, WISP_Z),
      debug: (_frame, kit) => {
        const gold = this.ctx!.tokens.celebrationGold, cyan = this.ctx!.tokens.uiCyan;
        const target = this.ledger.target, estimate = this.ledger.estimate;
        kit.marks('track', 3).set(target * TRACK_SCALE, ICE_LEVEL + 0.35, WISP_Z - 1.2, 0.12, gold);
        kit.marks('track', 3).set(this.ledger.observed * TRACK_SCALE, ICE_LEVEL + 0.35, WISP_Z - 0.6, 0.09, this.ctx!.tokens.cueMagenta);
        kit.marks('track', 3).set(estimate * TRACK_SCALE, ICE_LEVEL + 0.35, WISP_Z, 0.1, cyan);
        const line = kit.curve('miss', 2, cyan);
        line.point(estimate * TRACK_SCALE, ICE_LEVEL + 0.35, WISP_Z); line.point(target * TRACK_SCALE, ICE_LEVEL + 0.35, WISP_Z - 1.2);
        kit.trail('star', 120, gold).push(target * TRACK_SCALE, ICE_LEVEL + 0.35, WISP_Z - 1.2);
      },
    };
  }

  protected start(_ctx: SceneContext, world: FrozenLakeWorld): void {
    world.placeStar(0, 0, true);
    this.wisp.snap();
    this.wisp.place(0, 0.4);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<FrozenLakeWorld>): void {
    const state = frame.state, target = state.pose[TRACK_TARGET] ?? 0, estimate = state.pose[TRACK_ESTIMATE] ?? 0, observed = state.pose[TRACK_OBSERVED] ?? 0;
    this.ledger.target = target; this.ledger.observed = observed; this.ledger.estimate = estimate; this.ledger.reward = state.reward;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.placeStar(target * TRACK_SCALE, observed * TRACK_SCALE, frame.scrubbed);
    const miss = estimate - target;
    const lock = 1 / (1 + miss * miss * 1.5);
    this.wisp.group.position.x = estimate * TRACK_SCALE;
    this.wisp.face(Math.atan2(-miss * TRACK_SCALE, -WISP_Z - 3));
    this.wisp.dim(0.6 + 0.4 * lock);
    if (lock > 0.85) this.wisp.flare(0.35);
    if (frame.scrubbed) this.wisp.snap();
    this.wisp.update(frame.dtReal, time);
    this.wisp.place(0, 0.4 + 0.12 * Math.sin(time * 1.7));
    const jitter = 0.7 + 0.3 * Math.sin(time * 23 + state.step);
    setEmissiveIntensity(world.ghostMaterial, 1.1 * jitter);
    world.halo.scale.setScalar((1.5 + world.innovation * 5) * (1 + 0.06 * Math.sin(time * 3.1)));
    world.lamps.forEach((lamp, i) => lamp.lit(0.45 + 0.35 * lock + 0.08 * Math.sin(time * 1.9 + i)));
    const glow = world.wispLight;
    if (glow) { glow.position.set(estimate * TRACK_SCALE, 1.1, WISP_Z); glow.intensity = 0.8 + 2 * lock; }
    world.setTime(time);
    world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.wisp.dispose(); }
}

export default ((() => new FrozenLakeScene()) satisfies SceneFactory);
