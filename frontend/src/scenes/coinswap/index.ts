import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { lamp, plinth, relief } from '@kit/props';
import { makeTerrace, yawToward } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { COINSWAP_ACTION, COINSWAP_COIN } from '@sim/envs/coinswap';
import { parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const LAMP_X = 2.2;
const LAMP_Z = -1.2;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const swapRate = parameter(ctx.environment, 'swapRate'), storm = Math.min(1, Math.max(0, (Math.log10(swapRate) + 4) / 2));
  const t = makeTerrace(ctx, scene, { mood: 'night', halfX: 5, halfZ: 4, ground: [0xd7dfe8, 0xeef3f8], amplitude: 1.2, landscape: {
    sky: { stars: { intensity: 2 - storm } }, light: { pointCount: 0 },
    weather: { preset: 'snow', count: 300 + Math.round(700 * storm), seed: ctx.seed, size: 1.6, opacity: 0.7, swirl: 0.26 + 0.5 * storm, rise: -0.55 - 0.6 * storm, bounds: { y: 2.5, halfX: 8, halfY: 2.5, halfZ: 6 } },
  } });
  const { world, rig, m, accent, parts } = t;
  rig.sun.intensity = 0.5; rig.hemi.intensity = 0.55;
  const wall = relief(world, 9, 1.8, 0.2, m);
  wall.position.set(0, 0, -3.6);
  world.add(wall);
  const lamps = [-1, 1].map(side => {
    const base = plinth(world, 0.9, 0.5, m);
    base.position.set(side * LAMP_X, 0, LAMP_Z);
    world.add(base);
    const l = lamp(world, accent, 1.6, m);
    l.group.position.set(side * LAMP_X, 0.5, LAMP_Z);
    world.add(l.group);
    return { ...l, x: side * LAMP_X, top: new THREE.Vector3(side * LAMP_X, 2.3, LAMP_Z) };
  });
  const coin = new THREE.Group();
  world.add(coin);
  parts.disc(coin, m.glass(ctx.tokens.celebrationGold, 0.9), 0, 0, 0, 0.28, 0.05, 20);
  parts.build();
  coin.rotation.x = Math.PI / 2;
  const sentinel = world.own(buildSentinel({ name: 'coinswapGuesser', tint: accent }));
  sentinel.place(0, 0, 1.8);
  world.add(sentinel.group);
  const extra = { land: t.land, lights: rig, lamps, coin, sentinel, storm, accent, gold: ctx.tokens.celebrationGold };
  return world.seal<WorldCore & typeof extra>(extra);
}

type LanternWorld = ReturnType<typeof buildWorld>;

class TwoLanternsScene extends StageScene<LanternWorld> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'twoLanterns';
  private view!: SceneView;
  private lastStep = -1;
  private guessAt = -Infinity;
  private guess = 0;
  private swapAt = -Infinity;
  private coinX = 0;
  private readonly ledger = { coin: 0, guess: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'LANTERNS',
    rows: [
      { key: 'coin', label: 'coin hides under', tint: 0xffd166, value: () => (this.ledger.coin ? 'the right lantern' : 'the left lantern') },
      { key: 'guess', label: 'guess', tint: 0xdcf4ff, value: () => (this.ledger.guess ? 'right' : 'left') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): LanternWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-5, -0.5, -4], max: [5, 3.5, 3] }, [0, 1, -0.4], { distance: 9, azimuth: 0.3, polar: 1.1 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [10, 8], agentPos: () => this.world!.sentinel.group.position }; }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, time }: Stage<LanternWorld>): void {
    const step = frame.state.step, coin = Math.round(frame.state.pose[COINSWAP_COIN]!) & 1, guess = Math.round(frame.state.pose[COINSWAP_ACTION]!) & 1;
    this.ledger.coin = coin; this.ledger.guess = guess; this.ledger.reward = frame.state.reward;
    if (step !== this.lastStep) {
      this.lastStep = step; this.guessAt = time; this.guess = guess;
      if (!frame.scrubbed && frame.state.reward > 0) pool.burst(world.lamps[guess]!.top, world.gold, 0.5, time);
    }
    for (const event of frame.events) if (event.type === 'cue') this.swapAt = time;
    if (frame.scrubbed) { this.coinX = world.lamps[coin]!.x; this.swapAt = -Infinity; }
    const swap = envelope(time - this.swapAt, 0.8, 1.5);
    this.coinX += (world.lamps[coin]!.x - this.coinX) * (1 - Math.exp(-8 * frame.dtReal));
    world.coin.position.set(this.coinX, 0.08 + swap * 0.3, LAMP_Z + 0.9);
    world.coin.rotation.y = swap * Math.PI * 4;
    const flare = envelope(time - this.guessAt, 1, 2), flicker = swap * (0.5 + 0.5 * Math.sin(time * 40));
    world.lamps.forEach((l, side) => l.lit(0.45 + (side === this.guess ? flare * (frame.state.reward > 0 ? 1.4 : 0.3) : 0) - flicker * 0.4 + 0.04 * Math.sin(time * 2 + side)));
    world.sentinel.idle(time);
    world.sentinel.face(yawToward(0, 1.8, world.lamps[this.guess]!.x, LAMP_Z));
    world.sentinel.glow(frame.state.reward > 0 ? flare : 0);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(this.rig!, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.guessAt = this.swapAt = -Infinity; }
}

export default ((() => new TwoLanternsScene()) satisfies SceneFactory);
