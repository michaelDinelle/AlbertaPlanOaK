import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { KeyToken } from '@kit/grid';
import { gate, lamp, plinth, relief } from '@kit/props';
import { memoryCourt, outcomeGlow, stoneRoad } from '@kit/props/memory-court';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { KEYDOOR_KEY, KEYDOOR_PHASE } from '@sim/envs/keydoor';
import { lastEvent } from '@sim/events';
import { parameter } from '../../data/envs';
import { formatSigned } from '../../format';

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, delay = parameter(env, 'delay');
  if (delay < 2 || env.actions !== 2) throw new Error('KeyDoor geometry does not match its recorded configuration');
  const length = 4 * Math.log2(delay) + 2, spacing = length / (delay - 1), halfL = length / 2;
  const court = memoryCourt(ctx, scene, {
    name: 'keyDoorGatehouse', palette: 'dawn', fogExtent: 120 + length, points: 0, sun: 1.6, hemi: 0.7,
    weather: { preset: 'snow', count: 700, seed: ctx.seed, size: 2.4, bounds: { y: 3, halfX: 10, halfY: 3.5, halfZ: halfL + 7 } },
    focus: { x: 0, y: 0.5, z: 0, radius: halfL + 5 },
    meadow: { halfX: 6.5, halfZ: halfL + 5, ground: 0xe4ebf2, grass: 0xd6dfe8, trees: [0xffffff, 0xf4f8ff, 0xffffff, 0xeef3fa], archetypes: ['bare', 'conifer'], amplitude: 1.4 },
  });
  const { world, materials: m, accent } = court;
  const road = stoneRoad(world, m.sandstone, delay, spacing, halfL, 'keyDoorPath');
  const start = plinth(world, 0.5, 0.8, m);
  start.position.set(1.3, 0, halfL + 0.4);
  world.add(start);
  const key = world.own(new KeyToken({ color: accent, radius: 0.13, name: 'keyDoorKey' }));
  world.add(key.group);
  const door = gate(world, 1.4, 2.2, accent, m);
  door.group.position.set(0, 0, -halfL - 1.3);
  world.add(door.group);
  for (const side of [-1, 1]) { const wall = relief(world, 3.6, 2.2, 0.3, m); wall.position.set(side * 3, 0, -halfL - 1.3); world.add(wall); }
  const lamps = [-1.5, 1.5].map(x => { const post = lamp(world, accent, 1.6, m); post.group.position.set(x, 0, -halfL - 0.5); world.add(post.group); return post; });
  const sentinel = world.own(buildSentinel({ name: 'keyDoorSentinel', tint: accent }));
  sentinel.face(Math.PI);
  world.add(sentinel.group);
  const doorAt = new THREE.Vector3(0, 1.2, -halfL - 1.3);
  const extra = { court, delay, length, halfL, spacing, accent, road, key, door, doorAt, lamps, sentinel };
  return world.seal<WorldCore & typeof extra>(extra);
}

type KeyDoorWorld = ReturnType<typeof buildWorld>;

class KeyDoorScene extends StageScene<KeyDoorWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['dawn', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'keyDoorGatehouse';
  private view!: SceneView;
  private z = 0;
  private phaseSeen = -1;
  private outcomeStep: number | undefined;
  private readonly ledger = { phase: 0, hasKey: false, action: 0, reward: 0, gate: '\u2014' };
  override legend: LegendSpec = {
    title: 'GATEHOUSE',
    rows: [
      { key: 'step', label: 'stone', tint: 0xe8c66e, value: () => `${this.ledger.phase + 1} of ${this.world!.delay}` },
      { key: 'key', label: 'key', tint: 0xffd166, value: () => (this.ledger.hasKey ? 'in hand' : 'on the plinth') },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.phase === 0 ? (this.ledger.action ? 'take the key' : 'leave the key') : 'walk on') },
      { key: 'gate', label: 'last gate', tint: 0xffd166, value: () => this.ledger.gate },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): KeyDoorWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const world = this.world!;
    this.view = makeSceneView(ctx, { min: [-6, -0.5, -world.halfL - 4], max: [6, 3.5, world.halfL + 3] }, [0, 0.6, -world.halfL * 0.1], { distance: 12, azimuth: 0.5, polar: 1.05 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [12, this.world!.halfL * 2 + 8], agentPos: () => this.world!.sentinel.group.position }; }
  protected start(_ctx: SceneContext, world: KeyDoorWorld): void { this.z = world.halfL; }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<KeyDoorWorld>): void {
    const phase = Math.round(frame.state.pose[KEYDOOR_PHASE]!), hasKey = frame.state.pose[KEYDOOR_KEY]! > 0.5;
    if (phase < 0 || phase >= world.delay) throw new Error('KeyDoor frame is outside its recorded geometry');
    const targetZ = world.halfL - phase * world.spacing;
    if (frame.scrubbed || frame.state.reset) this.z = targetZ; else this.z += (targetZ - this.z) * (1 - Math.exp(-7 * frame.dtReal));
    const speed = Math.min(2, Math.abs(targetZ - this.z) * 6);
    if (speed > 0.05) world.sentinel.stride(speed, frame.dtReal); else world.sentinel.idle(time);
    world.sentinel.place(0, 0, this.z);
    world.sentinel.glow(hasKey ? 0.8 : 0.1);
    if (hasKey) world.key.group.position.set(0.36, 1.25, this.z + 0.1); else world.key.group.position.set(1.3, 0.85, world.halfL + 0.4);
    world.key.update(frame.dtReal, time);
    if (this.phaseSeen !== phase) {
      for (let i = 0; i < world.delay; i++) world.road.paint(i, i < phase ? 0xbcc4cc : i === phase ? 0xfff0c0 : 0xffffff);
      world.road.commit(false);
      this.phaseSeen = phase;
    }
    const step = lastEvent(frame.eventLog, 'done', frame.state.step);
    const outcome = step === undefined ? null : frame.frameAt(step);
    const hit = outcome !== null && outcome.reward > 0;
    this.ledger.phase = phase; this.ledger.hasKey = hasKey; this.ledger.action = Math.trunc(frame.frameAt(frame.state.step).actions[0]!) & 1; this.ledger.reward = frame.state.reward; this.ledger.gate = outcome === null ? '\u2014' : hit ? 'opened' : 'stayed shut';
    const age = step === undefined ? Infinity : time - step / frame.controlHz;
    const glow = outcomeGlow(age), opening = age >= 0 ? Math.min(1, age * 2) * Math.exp(-age * 0.4) : 0;
    world.door.lit(hit ? glow * 1.3 : -0.27 * glow);
    world.door.open(hit ? opening : 0);
    world.lamps.forEach((post, i) => post.lit(0.55 + 0.2 * Math.sin(time * 1.7 + i) + (hit ? glow : 0)));
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    if (!frame.scrubbed && step !== this.outcomeStep && outcome) {
      if (hit) pool.burst(world.doorAt, world.accent, 0.7, time); else pool.sigh(0.4, world.doorAt, time);
    }
    this.outcomeStep = step;
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal, [0, 0.6, this.z - 1.2]), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.phaseSeen = -1; this.outcomeStep = undefined; }
}

export default ((() => new KeyDoorScene()) satisfies SceneFactory);
