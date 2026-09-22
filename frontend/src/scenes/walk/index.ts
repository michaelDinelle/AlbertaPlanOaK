import { lastEvent } from '@sim/events';
import * as THREE from 'three';
import { PULSE_SECONDS } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import type { GridGeometry } from '@sim/types';
import { CHAIN_POS } from '@sim/envs/walk';
import { makeSceneView, trackPose, type SceneView } from '@kit/view';
import { buildCrossing, buildRover, chainX, deckHeight } from './actors';
import { buildWorld, riverHeight, WATER_LEVEL, type WorldHandle } from './world';
import { formatCount, formatSigned } from '../../format';

class RiverCrossingScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.36, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 8;
  protected override poolOptions() { return { burstSize: 120 }; }
  protected readonly stageName = 'riverCrossing';
  private view: SceneView | null = null;
  private rover: ReturnType<typeof buildRover> | null = null;
  private crossing: ReturnType<typeof buildCrossing> | null = null;
  private grid!: GridGeometry;
  private goalAt = -Infinity;
  private readonly goal = new THREE.Vector3();
  private readonly cell = { x: 0, y: 0 };
  private readonly ledger = { pos: 0, forward: true, reward: 0, crossed: '\u2014' };
  override legend: LegendSpec = {
    title: 'CROSSING',
    rows: [
      { key: 'at', label: 'plank', tint: 0xf7a8b8, value: () => `${this.ledger.pos + 1} of ${this.grid.w}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.forward ? 'forward \u2192' : '\u2190 back') },
      { key: 'crossed', label: 'last crossing', tint: 0xffd166, value: () => this.ledger.crossed },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): Promise<WorldHandle> {
    const grid = ctx.geometry.grid;
    if (!grid || grid.w < 2 || grid.h !== 1 || !grid.goal) throw new Error('Chain grid is missing or invalid');
    this.grid = grid;
    const end = chainX(grid.goal.x, grid);
    this.goal.set(end, deckHeight(end) + 0.8, 0);
    this.view = makeSceneView(ctx, { min: [-11.8, -2, -2], max: [11.8, 3.2, 2] }, [0, 0.35, 0], { distance: 16, azimuth: 0.22, polar: 1.38 }, 5000);
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    this.rover = buildRover();
    this.crossing = buildCrossing(this.grid, this.accentOf(ctx));
    this.scene.add(this.rover.group, this.crossing.group);
  }

  protected rigOptions(): OrbitRigOptions {
    return { ...this.view!.options, near: 1, floor: (x, z) => Math.max(riverHeight(x, z) + 0.65, WATER_LEVEL + 0.25), limits: { ...this.view!.options.limits, maxPolar: 1.48 } };
  }

  protected overlayOptions(): GroundTruthOptions {
    return { agentCell: state => { this.cell.x = Math.round(state.pose[CHAIN_POS]!); return this.cell; }, goalCell: () => this.grid.goal! };
  }

  protected start(): void {
    this.rover!.setPosition(chainX(0, this.grid), true);
    this.crossing!.setState(0, 0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const rover = this.rover!, position = trackPose(frame, CHAIN_POS), step = frame.state.step, crossedAt = lastEvent(frame.eventLog, 'goal', step);
    this.ledger.pos = Math.round(frame.state.pose[CHAIN_POS]!); this.ledger.forward = (Math.trunc(frame.frameAt(step).actions[0]!) & 1) === 1; this.ledger.reward = frame.state.reward;
    this.ledger.crossed = crossedAt === undefined ? '\u2014' : `${formatCount(step - crossedAt)} steps ago`;
    let achievedAt: number | null = null;
    pool.update(frame.dtReal, time);
    rover.setPosition(chainX(position, this.grid), frame.scrubbed || frame.state.reset);
    if (frame.scrubbed) {
      pool.clear(); this.goalAt = -Infinity;
      const step = lastEvent(frame.eventLog, 'goal', frame.state.step);
      if (step !== undefined) achievedAt = step / frame.controlHz;
    }
    for (const event of frame.events) if (event.type === 'goal') achievedAt = time;
    if (achievedAt !== null) {
      this.goalAt = achievedAt;
      pool.burst(this.goal, 0xb8ecd8, 0.4, achievedAt);
      pool.pulse.fire(Math.max(0, 0.16 - (time - achievedAt) / PULSE_SECONDS));
      pool.update(0, time);
    }
    this.crossing!.setState(Math.round(position), Math.exp(-(time - this.goalAt) / 1.4));
    world.setTime(time);
    world.setView(this.view!.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }

  protected drop(): void {
    this.rover?.dispose(); this.rover = null;
    this.crossing?.dispose(); this.crossing = null;
  }
}

export default ((() => new RiverCrossingScene()) satisfies SceneFactory);
