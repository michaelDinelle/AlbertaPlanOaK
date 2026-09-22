import type { GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import { slipDraw } from '@kit/props/ice';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { AB_SUCCESS_HEIGHT, ACROBOT_ACTION, ACROBOT_TH1, ACROBOT_TH2, ACROBOT_W1, ACROBOT_W2 } from '@sim/envs/acrobot';
import { formatPercent, formatSigned } from '../../format';
import { makeSceneView, type SceneView } from '@kit/view';
import { constant } from '../../data/envs';
import { buildArm, type ArmHandle } from './actors';
import { buildWorld, lakeHeight, pivotAnchor, WATER_Y, type WorldHandle } from './world';

const SKID_SECONDS = 0.8;
const GOAL_SECONDS = 1.4;
const TORQUE = ['\u2190 swing back', '\u00b7 coast', 'swing forward \u2192'] as const;

class AlpineLakeScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.46, fog: { from: ['dawn', 'fog'] } };
  readonly controlHz = 5;
  protected override poolOptions() { return { burstSize: 110 }; }
  protected readonly stageName = 'alpineLake';
  private view: SceneView | null = null;
  private arm: ArmHandle | null = null;
  private omegaMax = 1;
  private lastStep = -1;
  private skidAt = -Infinity;
  private goalAt = -Infinity;
  private readonly ledger = { height: 0, torque: 1, spin: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'ALPINE LAKE',
    rows: [
      { key: 'height', label: 'tip height', tint: 0xf2a7c3, value: () => `${formatSigned(this.ledger.height)} (bar at ${formatSigned(AB_SUCCESS_HEIGHT)})` },
      { key: 'torque', label: 'action', tint: 0xdcf4ff, value: () => TORQUE[this.ledger.torque]! },
      { key: 'spin', label: 'spin', tint: 0x98e1cd, value: () => formatPercent(this.ledger.spin, 0) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): Promise<WorldHandle> {
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    this.omegaMax = constant(ctx.environment, 'ACROBOT_OMEGA1_MAX');
    this.arm = buildArm(pivotAnchor(ctx.geometry), { warm: 0xffc98a, status: ctx.tokens.uiCyan, frost: world.slip > 0, blind: world.positionOnly });
    this.scene.add(this.arm.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-2.85, -2.5, -0.9], max: [2.85, 2.2, 0.3] }, [0, -0.6, 0], { distance: 7.2, azimuth: 0.12, polar: 1.48 }, 5000);
    return { ...this.view.options, near: 1, floor: (x, z) => Math.max(lakeHeight(x, z) + 0.6, WATER_Y + 0.2), limits: { ...this.view.options.limits, maxPolar: 1.48 } };
  }

  protected overlayOptions(): GroundTruthOptions {
    const env = this.ctx!.environment, top = constant(env, 'ACROBOT_ENERGY_TOP'), pivot = pivotAnchor(this.ctx!.geometry);
    return {
      agentPos: () => this.arm?.tipWorld ?? null,
      debug: (frame, kit) => {
        const arm = this.arm;
        if (!arm) return;
        const pose = frame.state.pose, th1 = pose[ACROBOT_TH1] ?? 0, th2 = pose[ACROBOT_TH2] ?? 0, w1 = pose[ACROBOT_W1] ?? 0, w2 = pose[ACROBOT_W2] ?? 0;
        const c2 = Math.cos(th2), energy = 0.5 * ((3.5 + c2) * w1 * w1 + 2 * (1.25 + 0.5 * c2) * w1 * w2 + 1.25 * w2 * w2) - 14.7 * Math.cos(th1) - 4.9 * Math.cos(th1 + th2);
        const gold = this.ctx!.tokens.celebrationGold, cyan = this.ctx!.tokens.uiCyan, red = 0xff6b57;
        kit.bars('energy', 2).set(pivot[0] + 2.5, pivot[1] - 2.2, pivot[2], Math.max(0, energy + 19.6) / (2 * top) * 2.2, energy > top ? red : cyan, 0.14);
        kit.marks('energyTop', 1).set(pivot[0] + 2.5, pivot[1] - 2.2 + 2.2, pivot[2], 0.1, gold);
        kit.trail('tip', 90, cyan).push(arm.tipWorld.x, arm.tipWorld.y, arm.tipWorld.z + 0.02);
      },
    };
  }

  protected start(): void {
    this.arm?.setState(0, 0);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const arm = this.arm;
    if (!arm) return;
    const state = frame.state;
    arm.setState(state.pose[ACROBOT_TH1] ?? 0, state.pose[ACROBOT_TH2] ?? 0);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.skidAt = -Infinity; this.goalAt = -Infinity; }
    if (state.step !== this.lastStep) {
      this.lastStep = state.step;
      if (!frame.scrubbed && slipDraw(ctx.seed, state.step, world.slip) !== null) { this.skidAt = time; pool.burst(arm.elbowWorld, 0xdcf4ff, 0.3, time); }
    }
    for (const event of frame.events) if (event.type === 'goal') { this.goalAt = time; pool.burst(arm.tipWorld, 0x98e1cd, 0.35, time); pool.ripple(arm.elbowWorld, ctx.tokens.accents['acrobot'] ?? ctx.tokens.celebrationGold, 0.3, time); }
    const skid = envelope(time - this.skidAt, SKID_SECONDS, 2.2), goal = envelope(time - this.goalAt, GOAL_SECONDS, 1.6);
    const spin = Math.min(1, Math.hypot(state.pose[ACROBOT_W1] ?? 0, state.pose[ACROBOT_W2] ?? 0) / this.omegaMax);
    const th1 = state.pose[ACROBOT_TH1] ?? 0, th2 = state.pose[ACROBOT_TH2] ?? 0;
    this.ledger.height = -Math.cos(th1) - Math.cos(th1 + th2); this.ledger.torque = Math.min(2, Math.max(0, Math.round(state.pose[ACROBOT_ACTION] ?? 1))); this.ledger.spin = spin; this.ledger.reward = state.reward;
    arm.setStatus(spin + goal, skid);
    world.lamps.forEach((lamp, i) => lamp.lit(world.positionOnly ? 0 : 0.55 + 0.8 * goal + 0.08 * Math.sin(time * 1.4 + i)));
    world.rimLight.intensity = 2.2 + 1.2 * goal;
    world.accentLight.intensity = 1.5 + 0.8 * spin + 3 * skid;
    world.accentLight.position.set(arm.tipWorld.x * 0.5 - 1.2, WATER_Y + 0.5, 1.6);
    world.setTime(time);
    if (this.view) world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }

  protected drop(): void {
    this.arm?.dispose();
    this.arm = null;
    this.lastStep = -1;
    this.skidAt = -Infinity;
    this.goalAt = -Infinity;
  }
}

export default ((() => new AlpineLakeScene()) satisfies SceneFactory);
