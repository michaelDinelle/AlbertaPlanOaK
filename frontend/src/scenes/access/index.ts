import * as THREE from 'three';
import { buildCourier } from '@kit/cast/courier';
import { envelope } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { banner, gate, relief, signalPole } from '@kit/props';
import { makeTerrace } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { ACCESS_ACTION, ACCESS_FREE, ACCESS_PRIORITY } from '@sim/envs/access';
import { constant, constantList, parameter } from '../../data/envs';
import { formatCount, formatPercent, formatSigned } from '../../format';

const PRIORITY_TINTS = [0xb9c2c9, 0x8fd47a, 0x6fb8e8, 0xffd166] as const;
const QUEUE_Z = 1.6;
const GATE_Z = -2.6;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const drift = parameter(ctx.environment, 'drift'), servers = constant(ctx.environment, 'ACCESS_SERVERS'), priorities = constant(ctx.environment, 'ACCESS_PRIORITIES');
  const rewards = constantList(ctx.environment, 'ACCESS_REWARD'), period = constant(ctx.environment, 'ACCESS_DRIFT_PERIOD');
  if (rewards.length !== priorities || priorities > PRIORITY_TINTS.length) throw new Error('AccessControl priorities do not match the recorded configuration');
  const t = makeTerrace(ctx, scene, { mood: 'afternoon', halfX: 7, halfZ: 4.5, ground: [0xdfe6ec, 0xf4f7fa], amplitude: 1.1, landscape: {
    sky: { sunHdr: 4, sunGlow: 0.3 }, light: { pointCount: 1, pointDistance: 6 },
  } });
  const { world, rig, m, accent } = t;
  rig.sun.color.set(0xfff0d8); rig.sun.intensity = 2.2; rig.hemi.intensity = 0.7;
  const snow = world.size(world.time(world.own(makeAmbientParticles({ preset: 'snow', name: 'accessSnow', count: 900, seed: ctx.seed, size: 1.6, opacity: 0.7, bounds: { y: 2.5, halfX: 9, halfY: 2.5, halfZ: 7 } }))));
  world.add(snow.points);
  for (const side of [-1, 1]) { const wall = relief(world, 5.2, 2.4, 0.3, m); wall.position.set(side * 4.2, 0, GATE_Z); world.add(wall); }
  const door = gate(world, 2, 2.4, accent, m);
  door.group.position.set(0, 0, GATE_Z);
  world.add(door.group);
  const pole = signalPole(world, servers, 0xffa34d, m);
  pole.group.position.set(-5.4, 0, GATE_Z + 0.9);
  world.add(pole.group);
  const flags = PRIORITY_TINTS.slice(0, priorities).map((tint, p) => { const f = banner(world, tint, 2.6, m); f.position.set(2.2 + p * 0.9, 0, GATE_Z + 0.9); world.add(f); return f; });
  const point = rig.points[0];
  if (point) { point.color.set(accent); point.position.set(0, 2.2, GATE_Z + 0.4); }
  const courier = world.own(buildCourier({ name: 'accessCustomer', tint: accent, overgrowth: 0.2, scale: 1.4 }));
  world.add(courier.group);
  const ticket = new THREE.Mesh(world.own(new THREE.PlaneGeometry(0.5, 0.7, 2, 4).translate(0, 1.2, 0)), m.cloth(PRIORITY_TINTS[0]));
  ticket.castShadow = true;
  world.add(ticket);
  const ticketCloths = PRIORITY_TINTS.slice(0, priorities).map(tint => m.cloth(tint));
  const extra = { land: t.land, lights: rig, snow, door, pole, flags, courier, ticket, ticketCloths, rewards, servers, drift, period, accent };
  return world.seal<WorldCore & typeof extra>(extra);
}

type GatehouseWorld = ReturnType<typeof buildWorld>;

class GatehouseQueueScene extends StageScene<GatehouseWorld> {
  readonly post: ScenePost = { bloom: 0.36, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'gatehouseQueue';
  private view!: SceneView;
  private lastStep = -1;
  private acceptedAt = -Infinity;
  private readonly from = { x: 0, y: QUEUE_Z, heading: 0 };
  private readonly to = { x: 0, y: GATE_Z - 1.4, heading: 0 };
  private readonly burstAt = new THREE.Vector3();
  private readonly skyTop = new THREE.Color();
  private readonly skyHorizon = new THREE.Color();
  private readonly tint = new THREE.Color();
  private readonly ledger = { priority: 0, free: 0, accept: 0, reward: 0, tide: 0 };
  override legend: LegendSpec = {
    title: 'QUEUE',
    rows: [
      { key: 'customer', label: 'at the gate', tint: 0xe0b878, value: () => `priority ${this.ledger.priority + 1} \u00b7 pays ${formatSigned(this.world!.rewards[this.ledger.priority] ?? 0)}` },
      { key: 'servers', label: 'free rooms', tint: 0xffa34d, value: () => `${formatCount(this.ledger.free)} of ${this.world!.servers}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.accept ? 'admit' : 'turn away') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): GatehouseWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-7, -0.5, -4], max: [7, 3.5, 4] }, [0, 0.9, -0.6], { distance: 12, azimuth: -0.5, polar: 1.1 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [14, 8], agentPos: () => this.world!.courier.group.position }; }
  protected start(_ctx: SceneContext, world: GatehouseWorld): void { if (world.drift) { this.legend.note = () => `snow tide ${formatPercent(0.5 + 0.5 * this.ledger.tide, 0)}`; this.legend.bar = () => 0.5 + 0.5 * this.ledger.tide; } }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<GatehouseWorld>): void {
    const step = frame.state.step, free = Math.round(frame.state.pose[ACCESS_FREE]!), priority = Math.round(frame.state.pose[ACCESS_PRIORITY]!);
    const accept = Math.round(frame.state.pose[ACCESS_ACTION]!) & 1, admitted = accept === 1 && frame.state.reward > 0;
    if (priority < 0 || priority >= world.ticketCloths.length) throw new Error('AccessControl priority is outside its recorded range');
    this.ledger.priority = priority; this.ledger.free = free; this.ledger.accept = accept; this.ledger.reward = frame.state.reward;
    if (step !== this.lastStep) {
      this.lastStep = step;
      if (admitted) {
        this.acceptedAt = time;
        if (!frame.scrubbed) pool.burst(this.burstAt.set(0, 1.6, GATE_Z), PRIORITY_TINTS[Math.min(PRIORITY_TINTS.length - 1, priority)]!, 0.3 + 0.7 * frame.state.reward / 8, time);
      } else if (accept === 1 && !frame.scrubbed) pool.sigh(0.2, this.burstAt.set(0, 0.8, QUEUE_Z), time);
    }
    const passage = Math.min(1, (time - this.acceptedAt) / 0.7);
    world.door.open(envelope(time - this.acceptedAt, 1.3, 1.2) * 1.4);
    world.door.lit(envelope(time - this.acceptedAt, 1.2, 2));
    if (passage >= 1) world.courier.drive(this.from, this.from, 0, () => 0, true);
    else world.courier.drive(this.from, this.to, passage, () => 0, false);
    world.ticket.position.set(world.courier.group.position.x + 0.45, 0, world.courier.group.position.z + 0.2);
    world.ticket.material = world.ticketCloths[priority]!;
    world.pole.set((1 << (world.servers - free)) - 1);
    world.flags.forEach((flag, p) => { flag.scale.setScalar(p === priority ? 1.15 : 0.9); flag.position.y = p === priority ? 0.15 : 0; });
    const phase = world.drift ? Math.sin(2 * Math.PI * (step % world.period) / world.period) : 0;
    this.ledger.tide = phase;
    world.snow.setDensity(0.55 + 0.45 * phase);
    world.land.sky.setColors(this.skyTop.set(0x5da2d6).lerp(this.tint.set(0x9fc3e0), 0.5 + 0.5 * phase), this.skyHorizon.set(0xc6dfae).lerp(this.tint.set(0xe8e0d0), 0.5 + 0.5 * phase), 0xe8dfb0);
    const point = world.lights.points[0];
    if (point) point.intensity = 1 + 6 * envelope(time - this.acceptedAt, 1.2, 2);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.acceptedAt = -Infinity; }
}

export default ((() => new GatehouseQueueScene()) satisfies SceneFactory);
