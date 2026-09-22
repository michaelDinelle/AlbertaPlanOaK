import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import type { GroundTruthOptions } from '@kit/debug';
import { emissiveMaterial, setEmissiveIntensity } from '@kit/materials';
import { blendTokens } from '@kit/mix';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { SCENT_ACTION, SCENT_DECOY_X, SCENT_DECOY_Y, SCENT_GOAL_X, SCENT_GOAL_Y, SCENT_X, SCENT_Y, SP_STEP, scentAt } from '@sim/envs/scentpuck';
import { formatPercent, formatRate, formatSigned } from '../../format';
import { buildWorld, DECOY_TINT, HALF, SCALE, toWorldX, toWorldZ, type BadlandsWorld } from './world';

const SENTINEL_SCALE = 0.5;
const MOVE = ['↑ north', 'east →', '↓ south', '← west', '· stay'] as const;

class BadlandsScene extends StageScene<BadlandsWorld> {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'badlandsSand';
  protected override poolOptions() { return { burstSize: 90 }; }
  private view!: SceneView;
  private sentinel!: ReturnType<typeof buildSentinel>;
  private antennae: ReturnType<typeof emissiveMaterial>[] = [];
  private tip: THREE.BufferGeometry | null = null;
  private readonly at = new THREE.Vector3();
  private readonly last = new THREE.Vector3();
  private yaw = 0;
  private readonly ledger = { scent: 0, decoy: 0, distance: 0, move: 4, reward: 0 };
  override legend: LegendSpec = {
    title: 'SCENT',
    rows: [
      { key: 'scent', label: 'goal scent', tint: 0xe89a5e, value: () => formatPercent(this.ledger.scent, 0) },
      { key: 'decoy', label: 'decoy scent', tint: DECOY_TINT, value: () => (this.world!.competing ? formatPercent(this.ledger.decoy, 0) : 'none') },
      { key: 'distance', label: 'to goal', tint: 0xffd166, value: () => formatRate(this.ledger.distance, 2) },
      { key: 'move', label: 'action', tint: 0xdcf4ff, value: () => MOVE[this.ledger.move]! },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): BadlandsWorld { return buildWorld(ctx, this.scene); }

  protected buildActors(_ctx: SceneContext, world: BadlandsWorld): void {
    this.sentinel = buildSentinel({ name: 'scentSentinel', tint: world.accent, scale: SENTINEL_SCALE });
    const tip = this.tip = new THREE.SphereGeometry(0.06, 10, 8);
    this.antennae = [-1, 1].map(side => {
      const material = emissiveMaterial(side < 0 ? blendTokens(world.accent, world.palette.sun, 0.3) : DECOY_TINT, 0.2, { name: 'antennaTip' });
      const mesh = new THREE.Mesh(tip, material);
      mesh.position.set(side * 0.12, 1.9, 0.1);
      this.sentinel.group.add(mesh);
      return material;
    });
    this.scene.add(this.sentinel.group);
  }

  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-HALF - 0.5, -0.5, -HALF - 0.5], max: [HALF + 0.5, 1.5, HALF + 0.5] }, [0, 0.2, 0], { distance: 10, azimuth: 0.52, polar: 1.02 });
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const samples = 17;
    return {
      floor: [HALF * 2 + 1, HALF * 2 + 1], agentPos: () => this.at,
      debug: (frame, kit) => {
        const world = this.world!, pose = frame.state.pose, gold = this.ctx!.tokens.celebrationGold;
        const gx = pose[SCENT_GOAL_X] ?? 0, gy = pose[SCENT_GOAL_Y] ?? 0, field = kit.marks('scent', samples * samples);
        for (let j = 0; j < samples; j++) for (let i = 0; i < samples; i++) {
          const sx = -1 + 2 * i / (samples - 1), sy = -1 + 2 * j / (samples - 1), scent = scentAt(sx - gx, sy - gy);
          if (scent > 0.02) field.set(toWorldX(sx), 0.12 + scent * 0.8, toWorldZ(sy), 0.02 + scent * 0.12, gold);
        }
        kit.marks('goal', 2).set(toWorldX(gx), 1.1, toWorldZ(gy), 0.16, gold);
        if (world.competing) kit.marks('goal', 2).set(toWorldX(pose[SCENT_DECOY_X] ?? 0), 1.1, toWorldZ(pose[SCENT_DECOY_Y] ?? 0), 0.13, DECOY_TINT);
        kit.trail('walked', 120, this.ctx!.tokens.uiCyan).push(this.at.x, this.at.y + 0.08, this.at.z);
        const line = kit.curve('bearing', 2, gold);
        line.point(this.at.x, this.at.y + 0.3, this.at.z); line.point(toWorldX(gx), 0.3, toWorldZ(gy));
      },
    };
  }

  protected start(_ctx: SceneContext, world: BadlandsWorld): void {
    world.placePucks(0, 0, 9, 9);
    this.sentinel.place(0, world.heightAt(0, 0), 0);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<BadlandsWorld>): void {
    const state = frame.state, pose = state.pose;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.placePucks(pose[SCENT_GOAL_X] ?? 0, pose[SCENT_GOAL_Y] ?? 0, world.competing ? pose[SCENT_DECOY_X] ?? 9 : 9, world.competing ? pose[SCENT_DECOY_Y] ?? 9 : 9);
    const wx = toWorldX(pose[SCENT_X] ?? 0), wz = toWorldZ(pose[SCENT_Y] ?? 0);
    this.at.set(wx, world.heightAt(wx, wz), wz);
    const jump = frame.scrubbed || this.last.distanceTo(this.at) > SP_STEP * SCALE * 3;
    const dx = this.at.x - this.last.x, dz = this.at.z - this.last.z, travel = Math.hypot(dx, dz);
    if (!jump && travel > 1e-4) this.yaw = Math.atan2(dx, dz);
    const speed = jump || frame.dtReal <= 0 ? 0 : Math.min(2, travel / frame.dtReal / (SP_STEP * SCALE * frame.controlHz) * 1.3);
    this.sentinel.place(this.at.x, this.at.y, this.at.z);
    this.sentinel.face(this.yaw);
    if (speed > 0.05) this.sentinel.stride(speed, frame.dtReal); else this.sentinel.idle(time);
    this.last.copy(this.at);
    const sniff = state.observations;
    this.ledger.scent = Math.max(0, sniff[0] ?? 0); this.ledger.decoy = Math.max(0, sniff[1] ?? 0); this.ledger.move = Math.min(4, Math.max(0, Math.trunc(pose[SCENT_ACTION] ?? 4))); this.ledger.reward = state.reward;
    this.ledger.distance = Math.hypot((pose[SCENT_X] ?? 0) - (pose[SCENT_GOAL_X] ?? 0), (pose[SCENT_Y] ?? 0) - (pose[SCENT_GOAL_Y] ?? 0));
    setEmissiveIntensity(this.antennae[0]!, 0.2 + 3 * Math.max(0, sniff[0] ?? 0));
    setEmissiveIntensity(this.antennae[1]!, 0.05 + 3 * Math.max(0, sniff[1] ?? 0));
    for (const event of frame.events) if (event.type === 'goal') { pool.burst(this.at, world.accent, 0.5, time); pool.ripple(this.at, ctx.tokens.celebrationGold, 0.35, time); }
    world.setTime(time);
    world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { for (const material of this.antennae) material.dispose(); this.antennae = []; this.tip?.dispose(); this.tip = null; this.sentinel.dispose(); this.last.set(0, 0, 0); }
}

export default ((() => new BadlandsScene()) satisfies SceneFactory);
