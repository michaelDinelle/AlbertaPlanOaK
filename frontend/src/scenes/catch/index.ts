import * as THREE from 'three';
import type { GroundTruthOptions } from '@kit/debug';
import { setEmissiveIntensity } from '@kit/materials';
import { envelope } from '@kit/mix';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { CATCH_ACTION, CATCH_COLUMN, CATCH_PADDLE, CATCH_ROW } from '@sim/envs/catch';
import { lastEvent } from '@sim/events';
import { formatSigned } from '../../format';
import { wordInt } from '@sim/words';
import { buildWorld, COURT_DEPTH, PADDLE_Y, type PergolaWorld } from './world';

const CATCH_SECONDS = 1.2;
const MOVE = ['← left', '· stay', 'right →'] as const;

class PergolaCourtScene extends StageScene<PergolaWorld> {
  readonly post: ScenePost = { bloom: 0.46, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'pergolaCourt';
  protected override poolOptions() { return { burstSize: 90 }; }
  private view!: SceneView;
  private caughtAt = -Infinity;
  private readonly landing = new THREE.Vector3();
  private readonly agentCell = { x: 0, y: 0 };
  private readonly seedCell = { x: 0, y: 0 };
  private readonly ledger = { column: 0, row: 0, paddle: 0, move: 1, reward: 0, last: '—' };
  override legend: LegendSpec = {
    title: 'CATCH',
    rows: [
      { key: 'seed', label: 'seed', tint: 0xffd166, value: () => `column ${this.ledger.column + 1} · row ${this.ledger.row + 1} of ${this.world!.height}` },
      { key: 'paddle', label: 'paddle', tint: 0xffd166, value: () => `column ${this.ledger.paddle + 1}` },
      { key: 'move', label: 'action', tint: 0xdcf4ff, value: () => MOVE[this.ledger.move]! },
      { key: 'last', label: 'last drop', tint: 0xffd166, value: () => this.ledger.last },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): PergolaWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}

  protected rigOptions(ctx: SceneContext) {
    const world = this.world!, halfW = world.courtW / 2 + 1.6;
    this.view = makeSceneView(ctx, { min: [-halfW, -0.4, -COURT_DEPTH], max: [halfW, world.top + 1.4, COURT_DEPTH] }, [0, world.top * 0.5, 0.2], { distance: Math.max(8, world.top * 1.55, world.courtW * 1.5 + 2.5), azimuth: 0.24, polar: 1.12 });
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    return { agentCell: (state) => { this.agentCell.x = Math.round(state.pose[CATCH_PADDLE] ?? 0); this.agentCell.y = this.world!.height - 1; return this.agentCell; }, goalCell: (state) => { this.seedCell.x = Math.round(state.pose[CATCH_COLUMN] ?? 0); this.seedCell.y = Math.round(state.pose[CATCH_ROW] ?? 0); return this.seedCell; } };
  }

  protected start(_ctx: SceneContext, world: PergolaWorld): void {
    world.paddle.position.x = world.colX(Math.floor(world.width / 2));
    world.seed.position.set(0, world.rowY(0), 0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<PergolaWorld>): void {
    const state = frame.state, step = state.step;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.caughtAt = -Infinity; }
    const column = state.pose[CATCH_COLUMN] ?? 0, paddle = state.pose[CATCH_PADDLE] ?? 0;
    let row = state.pose[CATCH_ROW] ?? 0;
    const next = frame.frameAt(step + 1);
    if (next.step === step + 1 && next.reset && !state.reset) row += frame.position - step;
    this.ledger.column = Math.round(column); this.ledger.row = Math.round(state.pose[CATCH_ROW] ?? 0); this.ledger.paddle = Math.round(paddle);
    this.ledger.move = Math.min(2, Math.max(0, Math.round(state.pose[CATCH_ACTION] ?? 1))); this.ledger.reward = state.reward;
    const dropAt = lastEvent(frame.eventLog, 'done', step), caughtAt = lastEvent(frame.eventLog, 'goal', step);
    this.ledger.last = dropAt === undefined ? '—' : caughtAt === dropAt ? 'caught' : 'missed';
    world.paddle.position.x = world.colX(paddle);
    world.seed.position.set(world.colX(column), world.rowY(row), 0);
    world.seed.rotation.y = time * 1.6;
    if (world.seedLight) world.seedLight.position.set(world.seed.position.x, world.seed.position.y + 0.2, 0.35);
    for (const event of frame.events) {
      if (event.type === 'goal') { this.caughtAt = time; this.landing.set(world.paddle.position.x, PADDLE_Y + 0.15, 0); pool.burst(this.landing, world.gold, 0.55, time); pool.ripple(this.landing, world.accent, 0.3, time); }
      else if (event.type === 'done' && step > 0) { const before = frame.frameAt(step - 1); this.landing.set(world.colX(wordInt(before.words[0]!)), PADDLE_Y - 0.1, 0); if (!frame.events.some(other => other.type === 'goal')) pool.sigh(0.4, this.landing, time); }
    }
    const caught = envelope(time - this.caughtAt, CATCH_SECONDS, 1.8);
    const shimmer = world.noise > 0 ? Math.min(1, Math.abs(state.reward) / world.scale) : 0;
    world.paddleGlass.emissiveIntensity = 0.7 + 2.2 * caught + 1.4 * shimmer;
    setEmissiveIntensity(world.haloMaterial, 1.1 + 0.8 * shimmer * Math.sin(time * 27) ** 2);
    world.glitter.setIntensity(1.6 + 1.6 * shimmer);
    world.lamps.forEach((lamp, i) => lamp.lit(0.45 + 0.6 * caught + 0.06 * Math.sin(time * 1.7 + i)));
    world.setTime(time);
    world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.caughtAt = -Infinity; }
}

export default ((() => new PergolaCourtScene()) satisfies SceneFactory);
