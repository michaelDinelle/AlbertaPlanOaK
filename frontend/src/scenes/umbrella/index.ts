import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { makeAmbientParticles } from '@kit/particles';
import { brazier, doorway, lamp, plinth, relief } from '@kit/props';
import { instanced, lampRow, memoryCourt, outcomeGlow, stoneRoad } from '@kit/props/memory-court';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { UMBRELLA_DISTRACTOR, UMBRELLA_HAVE, UMBRELLA_MAX_DISTRACTORS, UMBRELLA_NEED, UMBRELLA_PHASE } from '@sim/envs/umbrella';
import { lastEvent } from '@sim/events';
import { parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const CLEAR = [0x5b6b80, 0x9aa5b1, 0x76828f] as const;
const STORM = [0x2b3342, 0x5a636f, 0x454d59] as const;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, length = parameter(env, 'length'), distractors = parameter(env, 'distractors');
  if (length < 2 || env.actions !== 2 || distractors > UMBRELLA_MAX_DISTRACTORS) throw new Error('Umbrella geometry does not match its recorded configuration');
  const spacing = length <= 16 ? 1 : 0.5, road = (length - 1) * spacing, halfL = road / 2;
  const court = memoryCourt(ctx, scene, {
    name: 'umbrellaRoad', palette: 'afternoon', fogExtent: 90 + road, points: 0, sun: 1.0, hemi: 0.9, sky: { sunHdr: 1.5, sunGlow: 0.2 },
    focus: { x: 0, y: 0.5, z: 0, radius: halfL + 5 },
    meadow: { halfX: 5.5, halfZ: halfL + 5, ground: 0x7f9a62, grass: 0x6f9a5a, trees: [0xa8d47a, 0x8fc46e, 0xc6e08a, 0x7fb066], amplitude: 1.3 },
  });
  const { world, materials: m, accent, rig } = court;
  rig.sun.color.set(0xd8e2ee);
  const rain = makeAmbientParticles({ preset: 'snow', count: 1400, bounds: { y: 3.2, halfX: 8, halfY: 3.4, halfZ: halfL + 6 }, seed: ctx.seed ^ 0x77, color: 0xbcd0e4, size: 1.3, rise: -3.2, swirl: 0.04, opacity: 0.55, name: 'umbrellaRain' });
  world.add(rain.points);
  world.size(world.time(world.own(rain)));
  const stones = stoneRoad(world, m.sandstone, length, spacing, halfL, 'umbrellaRoad');
  const shrine = doorway(world, 1.0, 1.7, m);
  shrine.position.set(0, 0, -halfL - 1.5);
  world.add(shrine);
  const fire = brazier(world, accent, m);
  fire.group.position.set(0, 0, -halfL - 2.3);
  world.add(fire.group);
  const back = relief(world, 3.6, 1.7, 0.8, m);
  back.position.set(0, 0, -halfL - 3);
  world.add(back);
  const post = lamp(world, accent, 1.5, m);
  post.group.position.set(1.2, 0, halfL + 0.5);
  world.add(post.group);
  const stand = plinth(world, 0.45, 0.6, m);
  stand.position.set(-1.2, 0, halfL + 0.5);
  world.add(stand);
  const cloth = m.cloth(accent), parts = modelParts(world);
  world.time(cloth.oak);
  const carried = new THREE.Group(), stowed = new THREE.Group();
  for (const holder of [carried, stowed]) {
    parts.rod(holder, m.brass, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1.05, 0), 0.02);
    parts.add(holder, new THREE.ConeGeometry(0.6, 0.24, 8, 1, true).translate(0, 1, 0), cloth);
    parts.add(holder, new THREE.SphereGeometry(0.04, 8, 6).translate(0, 1.14, 0), m.brass);
  }
  parts.build();
  stowed.position.set(-1.2, 0.6, halfL + 0.5);
  world.add(stowed);
  world.add(carried);
  const poles = distractors ? instanced(world, new THREE.CylinderGeometry(0.03, 0.04, 1.3, 8).translate(0, 0.65, 0), m.brass, distractors, 'umbrellaPoles') : null;
  const bulbs = distractors ? lampRow(world, distractors, 0.09, 'umbrellaDistractors') : null;
  for (let d = 0; d < distractors; d++) {
    const x = (d % 2 ? 1 : -1) * 1.15, z = halfL - (d + 0.5) * road / distractors;
    poles!.place(d, x, 0, z); poles!.paint(d, 0xffffff);
    bulbs!.place(d, x, 1.36, z); bulbs!.paint(d, accent, 0.1);
  }
  poles?.commit();
  bulbs?.commit();
  const sentinel = world.own(buildSentinel({ name: 'umbrellaWalker', tint: accent }));
  sentinel.face(Math.PI);
  world.add(sentinel.group);
  const fireAt = new THREE.Vector3(0, 1.1, -halfL - 2.3);
  const extra = { court, length, spacing, halfL, distractors, accent, rain, stones, fire, fireAt, post, carried, stowed, bulbs, sentinel };
  return world.seal<WorldCore & typeof extra>(extra);
}

type UmbrellaWorld = ReturnType<typeof buildWorld>;

class UmbrellaScene extends StageScene<UmbrellaWorld> {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['afternoon', 'fog'], blend: [[['night', 'fog'], 0.55]] } };
  readonly controlHz = 8;
  protected readonly stageName = 'umbrellaRoad';
  private view!: SceneView;
  private z = 0;
  private storm = 0.3;
  private step = -1;
  private outcomeStep: number | undefined;
  private readonly sky = [new THREE.Color(), new THREE.Color(), new THREE.Color()] as const;
  private readonly clear = CLEAR.map(hex => new THREE.Color(hex));
  private readonly dark = STORM.map(hex => new THREE.Color(hex));
  private readonly ledger = { phase: 0, need: false, have: false, action: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'ROAD',
    rows: [
      { key: 'step', label: 'stone', tint: 0x8fb8ff, value: () => `${this.ledger.phase + 1} of ${this.world!.length}` },
      { key: 'sky', label: 'sky at the shrine', tint: 0x5b6b80, value: () => (this.ledger.need ? 'storm' : 'clear') },
      { key: 'umbrella', label: 'umbrella', tint: 0x8fb8ff, value: () => (this.ledger.have ? 'carried' : 'stowed') },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.phase === 0 ? (this.ledger.action ? 'take the umbrella' : 'leave it') : 'walk on') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): UmbrellaWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const world = this.world!;
    this.view = makeSceneView(ctx, { min: [-5, -0.5, -world.halfL - 4], max: [5, 3.5, world.halfL + 3] }, [0, 0.6, -world.halfL * 0.1], { distance: 11, azimuth: 0.45, polar: 1.05 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [10, this.world!.halfL * 2 + 8], agentPos: () => this.world!.sentinel.group.position,
      debug: (_frame, kit) => {
        kit.bars('phase', 1).set(1.6, 0.05, this.z, 0.04 + 1.0 * this.ledger.phase / (this.world!.length - 1), this.ctx!.tokens.uiCyan, 0.1);
      },
    };
  }
  protected start(_ctx: SceneContext, world: UmbrellaWorld): void { this.z = world.halfL; world.post.lit(0.7); }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<UmbrellaWorld>): void {
    const pose = frame.state.pose, phase = Math.round(pose[UMBRELLA_PHASE]!), need = pose[UMBRELLA_NEED]! > 0.5, have = pose[UMBRELLA_HAVE]! > 0.5;
    if (phase < 0 || phase >= world.length) throw new Error('Umbrella frame is outside its recorded geometry');
    this.ledger.phase = phase; this.ledger.need = need; this.ledger.have = have; this.ledger.action = Math.trunc(frame.frameAt(frame.state.step).actions[0]!) & 1; this.ledger.reward = frame.state.reward;
    const targetZ = world.halfL - phase * world.spacing;
    if (frame.scrubbed || frame.state.reset) this.z = targetZ; else this.z += (targetZ - this.z) * (1 - Math.exp(-7 * frame.dtReal));
    const speed = Math.min(2, Math.abs(targetZ - this.z) * 6);
    if (speed > 0.05) world.sentinel.stride(speed, frame.dtReal); else world.sentinel.idle(time);
    world.sentinel.place(0, 0, this.z);
    world.sentinel.glow(have ? 0.7 : 0.1);
    world.carried.visible = have;
    world.stowed.visible = !have;
    world.carried.position.set(0.42, 1.05, this.z);
    const storm = phase === 0 || phase >= world.length - 2 ? (need ? 1 : 0) : 0.3;
    this.storm += (storm - this.storm) * (frame.scrubbed ? 1 : 1 - Math.exp(-4 * frame.dtReal));
    for (let i = 0; i < 3; i++) this.sky[i]!.copy(this.clear[i]!).lerp(this.dark[i]!, this.storm);
    world.court.land.sky.setColors(this.sky[0], this.sky[1], this.sky[2]);
    world.court.land.fog.color.copy(this.sky[1]);
    world.rain.setDensity(0.12 + 0.88 * this.storm);
    world.court.rig.sun.intensity = 1.1 - 0.7 * this.storm;
    if (frame.state.step !== this.step) {
      if (world.bulbs) {
        for (let d = 0; d < world.distractors; d++) world.bulbs.paint(d, world.accent, pose[UMBRELLA_DISTRACTOR + d]! > 0.5 ? 2.2 : 0.08);
        world.bulbs.commit(false);
      }
      for (let i = 0; i < world.length; i++) world.stones.paint(i, i === phase ? 0xfff0c0 : 0xffffff);
      world.stones.commit(false);
      this.step = frame.state.step;
    }
    const step = lastEvent(frame.eventLog, 'done', frame.state.step);
    const outcome = step === undefined ? null : frame.frameAt(step);
    const hit = outcome !== null && outcome.reward > 0;
    const age = step === undefined ? Infinity : time - step / frame.controlHz;
    const glow = outcomeGlow(age);
    world.fire.lit(0.6 + (hit ? 2 : -0.5) * glow);
    world.post.lit(0.7 + 0.1 * Math.sin(time * 2.1));
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    if (!frame.scrubbed && step !== this.outcomeStep && outcome) {
      if (hit) pool.burst(world.fireAt, world.accent, 0.6, time); else pool.sigh(0.4, world.fireAt, time);
    }
    this.outcomeStep = step;
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal, [0, 0.6, this.z - 1.2]), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.outcomeStep = undefined; this.storm = 0.3; }
}

export default ((() => new UmbrellaScene()) satisfies SceneFactory);
