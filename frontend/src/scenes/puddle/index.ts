import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import type { GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { PUDDLE_ACTION, PUDDLE_DELTA, PUDDLE_PENALTY, PUDDLE_STEP, PUDDLE_X, PUDDLE_Y, puddleBestAction, puddleDepth, puddlePlan } from '@sim/envs/puddle';
import { constant } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { buildWorld, COURT, HALF, toWorldX, toWorldZ, type MarshWorld } from './world';

const SENTINEL_SCALE = 0.7;
const GOAL_SECONDS = 1.6;
const MOVE = ['↑ north', 'east →', '↓ south', '← west'] as const;

class MarshFlatsScene extends StageScene<MarshWorld> {
  readonly post: ScenePost = { bloom: 0.36, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'marshFlats';
  protected override poolOptions() { return { burstSize: 90 }; }
  private view!: SceneView;
  private sentinel!: ReturnType<typeof buildSentinel>;
  private readonly at = new THREE.Vector3();
  private readonly last = new THREE.Vector3();
  private yaw = 0;
  private wading = false;
  private goalAt = -Infinity;
  private readonly overlayPos = new THREE.Vector3();
  private readonly ledger = { x: 0, y: 0, move: 0, depth: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'MARSH',
    rows: [
      { key: 'at', label: 'position', tint: 0x7fc8e8, value: () => `${formatCount(this.ledger.x * 100)}% east · ${formatCount(this.ledger.y * 100)}% north` },
      { key: 'move', label: 'action', tint: 0xdcf4ff, value: () => MOVE[this.ledger.move]! },
      { key: 'depth', label: 'puddle', tint: 0x5f9bb4, value: () => (this.ledger.depth > 0 ? formatSigned(-PUDDLE_PENALTY * this.ledger.depth) : 'dry') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): MarshWorld { return buildWorld(ctx, this.scene); }

  protected buildActors(_ctx: SceneContext, world: MarshWorld): void {
    this.sentinel = buildSentinel({ name: 'marshSentinel', tint: world.accent, scale: SENTINEL_SCALE });
    this.scene.add(this.sentinel.group);
  }

  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-HALF - 0.6, -0.6, -HALF - 0.6], max: [HALF + 0.6, 2, HALF + 0.6] }, [0.2, 0.25, -0.2], { distance: 10.2, azimuth: 0.38, polar: 1.08 });
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const env = this.ctx!.environment, side = constant(env, 'PUDDLE_PLAN_SIDE'), goal = constant(env, 'PUDDLE_GOAL'), penalty = constant(env, 'PUDDLE_PENALTY');
    const plan = puddlePlan(side, goal, penalty, constant(env, 'PUDDLE_PLAN_SWEEPS'));
    const samples = 21;
    return {
      floor: [HALF * 2 + 1, HALF * 2 + 1], agentPos: (state) => this.overlayPos.set(toWorldX(state.pose[PUDDLE_X] ?? 0), 0, toWorldZ(state.pose[PUDDLE_Y] ?? 0)),
      debug: (frame, kit) => {
        const gold = this.ctx!.tokens.celebrationGold, depths = kit.marks('depths', samples * samples);
        for (let j = 0; j < samples; j++) for (let i = 0; i < samples; i++) {
          const x = i / (samples - 1), y = j / (samples - 1), depth = puddleDepth(x, y);
          if (depth > 0) depths.set(toWorldX(x), 0.15 + depth * 6, toWorldZ(y), 0.03 + depth * 0.8, 0x5f9bb4);
        }
        const gx = toWorldX(goal), gz = toWorldZ(goal), corner = kit.curve('goal', 5, gold);
        corner.point(gx, 0.08, gz); corner.point(toWorldX(1), 0.08, gz); corner.point(toWorldX(1), 0.08, toWorldZ(1)); corner.point(gx, 0.08, toWorldZ(1)); corner.point(gx, 0.08, gz);
        let x = frame.state.pose[PUDDLE_X] ?? 0, y = frame.state.pose[PUDDLE_Y] ?? 0;
        const path = kit.curve('plan', 80, this.ctx!.tokens.uiCyan);
        path.point(toWorldX(x), 0.25, toWorldZ(y));
        for (let k = 0; k < 79 && !(x >= goal && y >= goal); k++) {
          const [dx, dy] = PUDDLE_DELTA[puddleBestAction(plan, side, goal, penalty, x, y)]!;
          x = Math.min(1, Math.max(0, x + PUDDLE_STEP * dx)); y = Math.min(1, Math.max(0, y - PUDDLE_STEP * dy));
          path.point(toWorldX(x), 0.25, toWorldZ(y));
        }
        kit.trail('walked', 120, 0xffffff).push(this.at.x, this.at.y + 0.1, this.at.z);
      },
    };
  }

  protected start(_ctx: SceneContext, world: MarshWorld): void {
    world.beacon.lit(0.7);
    this.sentinel.place(0, world.heightAt(0, 0), 0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<MarshWorld>): void {
    const state = frame.state, x = state.pose[PUDDLE_X] ?? 0, y = state.pose[PUDDLE_Y] ?? 0;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.goalAt = -Infinity; }
    const wx = toWorldX(x), wz = toWorldZ(y), depth = world.depthAt(wx, wz), ground = world.heightAt(wx, wz);
    this.ledger.x = x; this.ledger.y = y; this.ledger.move = Math.trunc(state.pose[PUDDLE_ACTION] ?? 0) & 3; this.ledger.depth = depth; this.ledger.reward = state.reward;
    this.at.set(wx, Math.min(ground, world.heightAt(wx, wz) * 0.55), wz);
    const jump = frame.scrubbed || state.reset || this.last.distanceTo(this.at) > PUDDLE_STEP * COURT * 3;
    const dx = this.at.x - this.last.x, dz = this.at.z - this.last.z, travel = Math.hypot(dx, dz);
    if (!jump && travel > 1e-4) this.yaw = Math.atan2(dx, dz);
    const speed = jump || frame.dtReal <= 0 ? 0 : Math.min(2.2, travel / frame.dtReal / (PUDDLE_STEP * COURT * frame.controlHz) * 1.4);
    this.sentinel.place(this.at.x, this.at.y, this.at.z);
    this.sentinel.face(this.yaw);
    if (speed > 0.05) this.sentinel.stride(speed, frame.dtReal); else this.sentinel.idle(time);
    this.last.copy(this.at);
    const wading = depth > 0;
    if (wading && !this.wading && !jump) pool.sigh(0.35, this.at, time);
    this.wading = wading;
    world.splash.points.position.set(wx, ground + depth * 2.2, wz);
    world.splash.setDensity(wading && speed > 0.05 ? Math.min(1, depth * 9) : 0);
    this.sentinel.glow(Math.max(0, 1 - depth * 7));
    for (const event of frame.events) if (event.type === 'goal') { this.goalAt = time; pool.burst(world.goal, world.accent, 0.5, time); pool.ripple(world.goal, world.accent, 0.4, time); }
    const flare = envelope(time - this.goalAt, GOAL_SECONDS, 1.6);
    world.beacon.lit(0.7 + 1.6 * flare);
    if (world.glowLight) { world.glowLight.position.set(wx, this.at.y + 0.9, wz); world.glowLight.intensity = 0.35 * Math.max(0, 1 - depth * 7); }
    world.setTime(time);
    world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.sentinel.dispose(); this.wading = false; this.goalAt = -Infinity; this.last.set(0, 0, 0); }
}

export default ((() => new MarshFlatsScene()) satisfies SceneFactory);
