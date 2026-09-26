import * as THREE from 'three/webgpu';
import { formatSigned } from '../../format';
import type { GroundTruthOptions } from '@kit/debug';
import { clampRange, envelope } from '@kit/mix';
import { slipDraw } from '@kit/props/ice';
import { type OrbitRigOptions } from '@kit/rigs';
import { makeSceneView, type SceneView } from '@kit/view';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameState, FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { PD_L, PD_MAX_SPEED, PD_MAX_TORQUE, PENDULUM_TH, PENDULUM_THDOT, PENDULUM_TORQUE, wrapPi } from '@sim/envs/pendulum';
import { parameter } from '../../data/envs';
import { buildLanternRig, type LanternRigHandle } from './actors';
import { buildWorld, pivotOf, type WorldHandle } from './world';

const CHILL_SECONDS = 0.9;

function torqueOf(state: FrameState): number {
  return state.pose[PENDULUM_TORQUE] ?? 0;
}

export function pendulumCost(theta: number, thetaDot: number, torque: number): number {
  const th = wrapPi(theta);
  return th * th + 0.1 * thetaDot * thetaDot + 0.001 * torque * torque;
}

export const COST_MAX = pendulumCost(Math.PI, PD_MAX_SPEED, PD_MAX_TORQUE);

const SERENITY_GAMMA = 3;

export function serenityOf(theta: number, thetaDot: number, torque = 0): number {
  const reward = 1 - pendulumCost(theta, thetaDot, torque) / COST_MAX;
  return clampRange(reward, 0, 1) ** SERENITY_GAMMA;
}

class LanternBellScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.54, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 20;
  private view: SceneView | null = null;
  protected readonly stageName = 'lanternBell';

  private lantern: LanternRigHandle | null = null;
  private pivot: readonly [number, number, number] = [0, 0, 0];
  private slip = 0;
  private positionOnly = false;
  private lastStep = -1;
  private chillAt = -Infinity;
  private readonly overlayPos = new THREE.Vector3();
  private readonly ledger = { angle: 0, spin: 0, torque: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'LANTERN BELL',
    rows: [
      { key: 'angle', label: 'from upright', tint: 0xffb35e, value: () => `${formatSigned(this.ledger.angle)}\u00b0` },
      { key: 'spin', label: 'spin', tint: 0xdcf4ff, value: () => formatSigned(this.ledger.spin) },
      { key: 'torque', label: 'torque', tint: 0xffd166, value: () => formatSigned(this.ledger.torque) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): WorldHandle {
    this.pivot = pivotOf(ctx.geometry);
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    const palette = ctx.tokens.palettes.night;
    const accent = this.accentOf(ctx);
    this.slip = Math.fround(parameter(ctx.environment, 'slip'));
    this.positionOnly = parameter(ctx.environment, 'positionOnly') !== 0;

    const lantern = buildLanternRig({
      palette,
      accent,
      pivot: this.pivot,
      maxTorque: PD_MAX_TORQUE,
      maxSpeed: PD_MAX_SPEED,
    });
    this.lantern = lantern;
    this.scene.add(lantern.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-2.4, -2.1, -0.55], max: [2.4, 2.1, 0.55] }, [0, -0.09, 0], { distance: 6.598249768, azimuth: 0.094375296, polar: 1.360000000 });
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentPos: (state) => {
        const th = state.pose[PENDULUM_TH] ?? 0;
        return this.overlayPos.set(this.pivot[0] + PD_L * Math.sin(th), this.pivot[1] - PD_L * Math.cos(th), this.pivot[2]);
      },
      debug: (frame, kit) => {
        const pose = frame.state.pose, th = wrapPi(pose[PENDULUM_TH] ?? 0), thDot = pose[PENDULUM_THDOT] ?? 0, torque = torqueOf(frame.state), gold = this.ctx!.tokens.celebrationGold, cyan = this.ctx!.tokens.uiCyan;
        const [px, py, pz] = this.pivot;
        kit.marks('upright', 1).set(px, py + PD_L, pz, 0.08, gold);
        kit.bars('spin', 2).set(px + 1.7, py - 1.2, pz, thDot / PD_MAX_SPEED * 1.2, cyan, 0.08);
        kit.bars('spin', 2).set(px + 2.0, py - 1.2, pz, torque * 2 / PD_MAX_TORQUE * 1.2, gold, 0.08);
        kit.trail('bob', 90, cyan).push(px + PD_L * Math.sin(th), py - PD_L * Math.cos(th), pz + 0.02);
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.applyState(Math.PI, 0, 0, 0, 0);
    world.setTime(0);
    this.lantern?.setTime(0);
  }

  private applyState(theta: number, thetaDot: number, torque: number, time: number, chill: number): void {
    const world = this.world;
    const lantern = this.lantern;
    if (!world || !lantern) return;

    lantern.setState(theta, thetaDot, torque);
    const serenity = serenityOf(theta, thetaDot, torque);
    lantern.setGlow(this.positionOnly ? 0.04 : serenity);
    lantern.setChill(chill);
    world.setLight(this.positionOnly ? serenity * 0.45 : serenity, lantern.bobRoot.position.x, time);

    const flame = lantern.flameWorld;
    world.motes.position.copy(flame);
    const light = world.lanternLight;
    if (light) { light.position.copy(flame); if (this.positionOnly) light.intensity = 0.05 + 0.6 * chill; }
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const lantern = this.lantern;
    if (!lantern) return;

    const state = frame.state;
    const th = state.pose[PENDULUM_TH] ?? 0;
    this.ledger.angle = Math.atan2(Math.sin(th), Math.cos(th)) * 180 / Math.PI; this.ledger.spin = state.pose[PENDULUM_THDOT] ?? 0; this.ledger.torque = state.pose[PENDULUM_TORQUE] ?? 0; this.ledger.reward = state.reward;

    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      this.chillAt = -Infinity;
    }

    const theta = state.pose[PENDULUM_TH] ?? 0;
    const thetaDot = state.pose[PENDULUM_THDOT] ?? 0;

    if (state.step !== this.lastStep) {
      this.lastStep = state.step;
      if (!frame.scrubbed && slipDraw(ctx.seed, state.step, this.slip) !== null) {
        this.chillAt = time;
        pool.burst(lantern.flameWorld, 0xbfefff, 0.2, time);
      }
    }

    this.applyState(theta, thetaDot, torqueOf(state), time, envelope(time - this.chillAt, CHILL_SECONDS, 2));

    world.setTime(time);
    lantern.setTime(time);

    if (this.view) world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }

  protected drop(): void {
    this.lantern?.dispose();
    this.lantern = null;
    this.lastStep = -1;
    this.chillAt = -Infinity;
  }
}

export default ((() => new LanternBellScene()) satisfies SceneFactory);
