import * as THREE from 'three';
import { CELL } from '@kit/grid';
import { tokenRing } from '@kit/props';
import { glyphSet, glyphStones, instanced, memoryCourt, outcomeGlow } from '@kit/props/memory-court';
import { modelParts } from '@kit/robot';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import type { WorldCore } from '@kit/world';
import { COUNT_ACTION, COUNT_COUNT, COUNT_EVENT, COUNT_PHASE, COUNT_QUERY } from '@sim/envs/count';
import { lastEvent } from '@sim/events';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const GLYPH = ['cyan cube', 'magenta pyramid', 'gold sphere', 'green octahedron'] as const;

const RAIL_Z = -0.6;
const RING_Z = -2.6;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, period = parameter(env, 'period'), modulus = constant(env, 'COUNT_MODULUS');
  if (modulus !== env.actions || modulus > 4 || period < 2) throw new Error('CountRecall geometry does not match its recorded configuration');
  const grid = { w: period, h: 1 }, halfW = (period - 1) * CELL / 2;
  const court = memoryCourt(ctx, scene, {
    name: 'countTerrace', palette: 'morning', fogExtent: 130 + halfW * 2, points: 2, pointDistance: 6,
    weather: { preset: 'pollen', count: 360, seed: ctx.seed, color: 0xfff1c8, size: 1.1, intensity: 0.8, bounds: { y: 1.4, halfX: halfW + 5, halfY: 1.2, halfZ: 6 } },
    focus: { x: 0, y: 0.5, z: 0, radius: halfW + 6 },
    meadow: { halfX: halfW + 5, halfZ: 7, ground: 0xb7a26a, grass: 0xc4a95a, trees: [0xe0a24a, 0xc9713a, 0xd8b35a, 0xa9843c], amplitude: 1.6 },
  });
  const { world, materials: m, accent, rig } = court;
  rig.points.forEach((point, i) => { point.color.set(accent); point.intensity = 5; point.position.set(i === 0 ? -halfW - 2 : halfW + 2, 1.8, 0); });
  const parts = modelParts(world), terrace = new THREE.Group();
  parts.box(terrace, m.sandstone, 0, -0.16, -0.4, period * CELL + 3, 0.32, 6.4, 0.04);
  parts.rod(terrace, m.brass, new THREE.Vector3(-halfW - 0.9, 0.5, RAIL_Z), new THREE.Vector3(halfW + 0.9, 0.5, RAIL_Z), 0.035);
  for (const x of [-halfW - 0.9, halfW + 0.9]) parts.box(terrace, m.sandstone, x, 0.3, RAIL_Z, 0.3, 0.6, 0.3, 0.03);
  parts.build();
  world.add(terrace);
  const beads = instanced(world, new THREE.SphereGeometry(0.24, 14, 10), world.own(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55, metalness: 0.2 })), period, 'countBeads');
  const cells = instanced(world, new THREE.BoxGeometry(0.8, 0.06, 0.6), m.sandstone, period, 'countPhaseCells');
  for (let i = 0; i < period; i++) { beads.place(i, -halfW + i * CELL, 0.5, RAIL_Z); beads.paint(i, 0x8f8672); cells.place(i, -halfW + i * CELL, 0.03, 0); cells.paint(i, 0xffffff); }
  beads.commit();
  cells.commit();
  const ring = tokenRing(world, modulus, 0.9, accent, m);
  ring.group.position.set(0, 0.02, RING_Z);
  world.add(ring.group);
  const glyphs = glyphSet(world, ctx.tokens);
  const stones = glyphStones(world, glyphs, modulus, (i, mesh) => { const at = ring.tokens[i]!.position; mesh.position.set(at.x, 0.78, at.z + RING_Z); mesh.scale.setScalar(0.55); });
  const wisp = world.own(new Wisp({ color: 0xfff1cf, radius: 0.3, height: 0.7, skirt: false, ribbon: false, silhouette: false, hop: { duration: 0.12, height: 0.14 }, name: 'countTally' }, grid));
  world.add(wisp.group);
  const extra = { court, grid, period, modulus, halfW, accent, beads, cells, ring, stones, wisp };
  return world.seal<WorldCore & typeof extra>(extra);
}

type CountWorld = ReturnType<typeof buildWorld>;

class CountScene extends StageScene<CountWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'countTerrace';
  private view!: SceneView;
  private lift = new Float32Array(0);
  private step = -1;
  private outcomeStep: number | undefined;
  private readonly cell = { x: 0, y: 0 };
  private readonly at = new THREE.Vector3();
  private readonly ledger = { count: 0, phase: 0, event: false, query: false, action: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'TALLY',
    rows: [
      { key: 'beads', label: 'beads counted', tint: 0xc99cf0, value: () => `${formatCount(this.ledger.count)} (mod ${this.world!.modulus} = ${GLYPH[this.ledger.count % this.world!.modulus]!})` },
      { key: 'phase', label: 'cell', tint: 0xdcf4ff, value: () => `${this.ledger.phase + 1} of ${this.world!.period}` },
      { key: 'event', label: 'this step', tint: 0xffd166, value: () => (this.ledger.query ? `asked \u2192 answered ${GLYPH[this.ledger.action]!}` : this.ledger.event ? 'a bead drops' : 'quiet') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): CountWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const world = this.world!;
    this.view = makeSceneView(ctx, { min: [-world.halfW - 4, -0.5, -5], max: [world.halfW + 4, 3, 5] }, [0, 0.5, -0.4], { distance: Math.max(10, world.period * 0.7), azimuth: 0.25, polar: 1.0 });
    return this.view.options;
  }
  protected overlayOptions() { return { agentCell: () => this.cell }; }
  protected start(_ctx: SceneContext, world: CountWorld): void { this.lift = new Float32Array(world.period); world.wisp.setCell(0, 0, world.grid); }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<CountWorld>): void {
    const count = Math.round(frame.state.pose[COUNT_COUNT]!), phase = Math.round(frame.state.pose[COUNT_PHASE]!), event = frame.state.pose[COUNT_EVENT]! > 0.5;
    if (phase < 0 || phase >= world.period || count < 0 || count > world.period) throw new Error('CountRecall frame is outside its recorded geometry');
    this.cell.x = phase;
    this.ledger.count = count; this.ledger.phase = phase; this.ledger.event = event; this.ledger.query = frame.state.pose[COUNT_QUERY]! > 0.5; this.ledger.action = Math.trunc(frame.state.pose[COUNT_ACTION]!) % world.modulus; this.ledger.reward = frame.state.reward;
    if (frame.scrubbed || Math.abs(world.wisp.cellX - phase) > 1) { world.wisp.setCell(phase, 0, world.grid); world.wisp.snap(); }
    else if (world.wisp.cellX !== phase) world.wisp.hopTo(this.cell, time);
    world.wisp.update(frame.dtReal, time);
    const k = frame.scrubbed ? 1 : 1 - Math.exp(-9 * frame.dtReal);
    for (let i = 0; i < world.period; i++) {
      const lift = this.lift[i]! + ((i < count ? 1 : 0) - this.lift[i]!) * k;
      this.lift[i] = lift;
      world.beads.place(i, -world.halfW + i * CELL, 0.5 + 0.55 * lift, RAIL_Z, 1 + 0.15 * lift);
      world.beads.paint(i, lift > 0.5 ? world.accent : 0x8f8672, 1 + lift * 0.6);
    }
    world.beads.commit(false);
    if (frame.state.step !== this.step) {
      if (!frame.scrubbed && event && count > 0) pool.burst(this.at.set(-world.halfW + (count - 1) * CELL, 1.1, RAIL_Z), world.accent, 0.3, time);
      this.step = frame.state.step;
    }
    const step = lastEvent(frame.eventLog, 'done', frame.state.step);
    const outcome = step === undefined || step < 1 ? null : frame.frameAt(step);
    const answer = outcome === null ? -1 : Math.trunc(outcome.actions[0]!) % world.modulus;
    const truth = step === undefined || step < 1 ? -1 : frame.frameAt(step - 1).words[0]! % world.modulus;
    const hit = outcome !== null && outcome.reward > 0;
    const age = step === undefined ? Infinity : time - step / frame.controlHz;
    const glow = outcomeGlow(age);
    world.ring.lift(answer, 0.3 * glow);
    world.stones.forEach((stone, i) => {
      stone.material.emissiveIntensity = 0.5 + (i === answer ? (hit ? 2.4 : -0.4) * glow : 0) + (i === truth && !hit ? 1.2 * glow * (0.5 + 0.5 * Math.sin(time * 8)) : 0);
      stone.mesh.rotation.y = time * 0.5;
    });
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    if (!frame.scrubbed && step !== this.outcomeStep && answer >= 0) {
      const stone = world.stones[answer]!;
      if (hit) pool.burst(stone.mesh.position, stone.tint, 0.55, time); else pool.sigh(0.3, stone.mesh.position, time);
    }
    this.outcomeStep = step;
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.outcomeStep = undefined; this.lift = new Float32Array(0); }
}

export default ((() => new CountScene()) satisfies SceneFactory);
