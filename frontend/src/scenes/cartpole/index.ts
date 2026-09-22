import type { GroundTruthOptions } from '@kit/debug';
import * as THREE from 'three';
import { envelope } from '@kit/mix';
import { slipDraw } from '@kit/props/ice';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { CARTPOLE_ACTION, CARTPOLE_THETA, CARTPOLE_THETADOT, CARTPOLE_X, CARTPOLE_XDOT, cartpoleForce, CP_SPEED_SCALE, CP_TRACK_HALF } from '@sim/envs/cartpole';
import { constant, parameter } from '../../data/envs';
import { formatPercent, formatSigned } from '../../format';
import { buildCart } from './actors';
import { buildWorld, COURT_HALF_X, COURT_HALF_Z, RAIL_Y, type PrairieWorld } from './world';

const SKID_SECONDS = 0.7;
const FALL_SECONDS = 1.1;
const PUSH = ['← push left', '· coast', 'push right →'] as const;

class PrairieRailScene extends StageScene<PrairieWorld> {
  readonly post: ScenePost = { bloom: 0.34, fog: { from: ['noon', 'fog'] } };
  readonly controlHz = 30;
  protected readonly stageName = 'prairieRail';
  protected override poolOptions() { return { burstSize: 70 }; }
  private view!: SceneView;
  private cart!: ReturnType<typeof buildCart>;
  private slip = 0;
  private readonly skidAtPos = new THREE.Vector3();
  private lastStep = -1;
  private skidAt = -Infinity;
  private fallAt = -Infinity;
  private readonly overlayPos = new THREE.Vector3();
  private readonly ledger = { push: 1, upright: 0, x: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'SWING-UP',
    rows: [
      { key: 'push', label: 'action', tint: 0xf0c25e, value: () => PUSH[this.ledger.push]! },
      { key: 'upright', label: 'upright', tint: 0xffd166, value: () => formatPercent(this.ledger.upright, 0) },
      { key: 'x', label: 'cart', tint: 0xdcf4ff, value: () => formatSigned(this.ledger.x) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): PrairieWorld { return buildWorld(ctx, this.scene); }

  protected buildActors(ctx: SceneContext, world: PrairieWorld): void {
    this.slip = Math.fround(parameter(ctx.environment, 'slip'));
    this.cart = buildCart({ palette: world.palette, accent: world.accent, materials: world.materials, poleLength: constant(ctx.environment, 'CARTPOLE_HALF') * 2, seed: ctx.seed, slip: world.slip });
    this.scene.add(this.cart.group);
  }

  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-COURT_HALF_X, -0.2, -COURT_HALF_Z], max: [COURT_HALF_X, RAIL_Y + 1.7, COURT_HALF_Z] }, [0, RAIL_Y + 0.15, 0], { distance: 6.4, azimuth: 0.3, polar: 1.3 });
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [COURT_HALF_X * 2, COURT_HALF_Z * 2],
      profileRange: [-CP_TRACK_HALF, CP_TRACK_HALF],
      agentPos: (state) => this.overlayPos.set(state.pose[CARTPOLE_X] ?? 0, RAIL_Y, 0),
      debug: (frame, kit) => {
        const pose = frame.state.pose, x = pose[CARTPOLE_X] ?? 0, xDot = pose[CARTPOLE_XDOT] ?? 0, thetaDot = pose[CARTPOLE_THETADOT] ?? 0;
        const gold = this.ctx!.tokens.celebrationGold, cyan = this.ctx!.tokens.uiCyan, red = 0xff6b57;
        kit.bars('motion', 2).set(x - 0.16, RAIL_Y + 0.06, 0.25, xDot * 0.25, cyan, 0.05);
        kit.bars('motion', 2).set(x + 0.16, RAIL_Y + 0.06, 0.25, thetaDot * 0.08, gold, 0.05);
        kit.marks('edges', 2).set(-CP_TRACK_HALF, RAIL_Y, 0, 0.08, red);
        kit.marks('edges', 2).set(CP_TRACK_HALF, RAIL_Y, 0, 0.08, red);
        kit.trail('tip', 120, gold).push(this.cart.tipWorld.x, this.cart.tipWorld.y, this.cart.tipWorld.z + 0.02);
      },
    };
  }

  protected start(ctx: SceneContext): void { this.cart.setState(0, Math.PI, 0, 0, 0); this.cart.setPixelRatio(ctx.size.dpr); }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<PrairieWorld>): void {
    const state = frame.state, cart = this.cart;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.skidAt = -Infinity; this.fallAt = -Infinity; }
    const x = state.pose[CARTPOLE_X] ?? 0, theta = state.pose[CARTPOLE_THETA] ?? 0, xDot = state.pose[CARTPOLE_XDOT] ?? 0;
    this.ledger.push = cartpoleForce(state.pose[CARTPOLE_ACTION] ?? 1) + 1; this.ledger.upright = (Math.cos(theta) + 1) / 2; this.ledger.x = x; this.ledger.reward = state.reward;
    if (state.step !== this.lastStep) {
      this.lastStep = state.step;
      if (!frame.scrubbed && slipDraw(ctx.seed, state.step, this.slip) !== null) { this.skidAt = time; pool.burst(this.skidAtPos.set(x, RAIL_Y + 0.05, 0), 0xdcf4ff, 0.22, time); }
    }
    for (const event of frame.events) if (event.type === 'done') { this.fallAt = time; pool.sigh(0.5, cart.tipWorld, time); }
    const skid = envelope(time - this.skidAt, SKID_SECONDS, 2.4), fall = envelope(time - this.fallAt, FALL_SECONDS, 1.8);
    cart.setState(x, theta, cartpoleForce(state.pose[CARTPOLE_ACTION] ?? 1), Math.cos(theta) - 0.01 * x * x, skid + Math.min(1, Math.abs(xDot) / CP_SPEED_SCALE) * 0.35);
    world.lamps.forEach((lamp, i) => lamp.lit(0.3 + 0.6 * fall + (Math.abs(x) > CP_TRACK_HALF * 0.8 && (x < 0) === (i === 0) ? 0.5 : 0)));
    world.rig.points.forEach((point, i) => { point.position.x = x + (i === 0 ? -0.4 : 0.4); point.position.y = RAIL_Y + 0.35; point.intensity = 0.25 + 0.5 * Math.max(0, Math.cos(theta)) + 1.5 * skid; });
    world.setTime(time); cart.setTime(time);
    world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); this.cart?.setPixelRatio(size.dpr); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.cart.dispose(); this.lastStep = -1; this.skidAt = -Infinity; this.fallAt = -Infinity; }
}

export default ((() => new PrairieRailScene()) satisfies SceneFactory);
