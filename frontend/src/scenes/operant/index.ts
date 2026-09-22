import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { lamp, plinth, relief } from '@kit/props';
import { makeTerrace } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { OPERANT_ACTION, OPERANT_CUE } from '@sim/envs/operant';
import { parameter } from '../../data/envs';
import { formatPercent, formatSigned } from '../../format';

const TOWER_Z = -2.6;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const cueRate = parameter(ctx.environment, 'cueRate'), tiers = Math.round(Math.log2(1 / cueRate));
  const t = makeTerrace(ctx, scene, { mood: 'afternoon', halfX: 5.5, halfZ: 4.5, ground: [0x7fa356, 0xc9cf7a], landscape: {
    sky: { sunHdr: 5.5, sunGlow: 0.3 }, light: { pointCount: 1, pointDistance: 6 },
    weather: { preset: 'pollen', count: 260, seed: ctx.seed, size: 1.2, bounds: { y: 1.5, halfX: 6, halfY: 1.5, halfZ: 5 } },
  } });
  const { world, rig, m, accent, parts } = t;
  rig.sun.intensity = 2.4; rig.hemi.intensity = 0.7;
  const wall = relief(world, 12, 1.4, 0.5, m);
  wall.position.set(0, 0, -4.4);
  world.add(wall);
  const tower = new THREE.Group();
  tower.position.set(0, 0, TOWER_Z);
  world.add(tower);
  const height = 1.2 + tiers * 0.55;
  for (let tier = 0; tier < tiers; tier++) {
    const width = 1.9 - tier * 0.12;
    parts.box(tower, tier % 2 ? m.shale : m.sandstone, 0, 0.6 + tier * 0.55 + 0.275, 0, width, 0.55, width, 0.04);
  }
  parts.box(tower, m.sandstone, 0, 0.3, 0, 2.4, 0.6, 2.4, 0.05);
  for (const side of [-1, 1]) parts.box(tower, m.sandstone, side * 0.7, height + 0.45, 0, 0.3, 0.9, 0.3, 0.04);
  parts.box(tower, m.brass, 0, height + 0.95, 0, 1.8, 0.12, 0.4, 0.02);
  parts.box(tower, m.sandstone, 0, height + 1.12, 0, 2, 0.26, 0.5, 0.05);
  parts.build();
  const bellPivot = new THREE.Group();
  bellPivot.position.set(0, height + 0.9, 0);
  tower.add(bellPivot);
  parts.add(bellPivot, new THREE.CylinderGeometry(0.2, 0.44, 0.6, 14, 1, true).translate(0, -0.4, 0), m.brass);
  parts.add(bellPivot, new THREE.SphereGeometry(0.21, 12, 8).scale(1, 0.5, 1).translate(0, -0.1, 0), m.brass);
  parts.add(bellPivot, new THREE.SphereGeometry(0.08, 8, 6).translate(0, -0.7, 0), m.shale);
  parts.build();
  const cue = lamp(world, ctx.tokens.cueCyan, 0.9, m);
  cue.group.position.set(-1.4, 0, TOWER_Z + 1.6);
  world.add(cue.group);
  const stand = plinth(world, 0.7, 0.8, m);
  stand.position.set(1.6, 0, 0.4);
  world.add(stand);
  const lever = new THREE.Group();
  lever.position.set(1.6, 0.85, 0.4);
  world.add(lever);
  parts.rod(lever, m.brass, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0.9, 0), 0.035);
  parts.add(lever, new THREE.SphereGeometry(0.09, 10, 8).translate(0, 0.92, 0), m.glass(accent, 0.6));
  parts.build();
  const sentinel = world.own(buildSentinel({ name: 'operantSubject', tint: accent }));
  sentinel.place(0.5, 0, 1.3);
  sentinel.face(Math.atan2(1.1, -0.9));
  world.add(sentinel.group);
  const point = rig.points[0];
  if (point) { point.color.set(ctx.tokens.celebrationGold); point.position.set(0, height + 0.9, TOWER_Z + 0.8); }
  const extra = { land: t.land, lights: rig, bellPivot, bellTop: new THREE.Vector3(0, height + 0.9, TOWER_Z), height, cue, lever, sentinel, accent };
  return world.seal<WorldCore & typeof extra>(extra);
}

type TowerWorld = ReturnType<typeof buildWorld>;

class BellTowerScene extends StageScene<TowerWorld> {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'bellTower';
  private view!: SceneView;
  private lastStep = -1;
  private rungAt = -Infinity;
  private pressAt = -Infinity;
  private cueAt = -Infinity;
  private readonly ledger = { cue: false, press: false, reward: 0 };
  override legend: LegendSpec = {
    title: 'BELL TOWER',
    rows: [
      { key: 'cue', label: 'cue lamp', tint: 0x5ee0ff, value: () => (this.ledger.cue ? 'lit' : 'dark') },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.press ? 'pull the lever' : 'wait') },
      { key: 'rate', label: 'cue rate', tint: 0xf7b26b, value: () => formatPercent(parameter(this.ctx!.environment, 'cueRate'), 1) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): TowerWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const top = this.world!.height + 1.3;
    this.view = makeSceneView(ctx, { min: [-5, -0.5, -4], max: [5, top, 4] }, [0.3, top * 0.45, -0.8], { distance: 8.5 + top * 1.1, azimuth: 0.6, polar: 1.12 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [10, 8], agentPos: () => this.world!.sentinel.group.position }; }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<TowerWorld>): void {
    const step = frame.state.step, cue = Math.round(frame.state.pose[OPERANT_CUE]!) === 1, press = Math.round(frame.state.pose[OPERANT_ACTION]!) === 1;
    this.ledger.cue = cue; this.ledger.press = press; this.ledger.reward = frame.state.reward;
    if (step !== this.lastStep) {
      this.lastStep = step;
      if (press) this.pressAt = time;
      if (cue) this.cueAt = time;
      if (frame.state.reward > 0) { this.rungAt = time; if (!frame.scrubbed) pool.burst(world.bellTop, world.accent, 0.7, time); }
      else if (frame.state.reward < 0 && !frame.scrubbed) pool.sigh(0.25, world.lever.position, time);
    }
    const ring = envelope(time - this.rungAt, 2.2, 1.1), cueGlow = envelope(time - this.cueAt, 0.9, 1.5);
    world.bellPivot.rotation.x = Math.sin((time - this.rungAt) * 9) * 0.45 * ring;
    world.lever.rotation.x = -0.9 * envelope(time - this.pressAt, 0.5, 1.5) * 2;
    world.cue.lit(cue ? 1.4 : cueGlow * 0.8);
    world.sentinel.idle(time);
    world.sentinel.glow(ring);
    const point = world.lights.points[0];
    if (point) point.intensity = 0.5 + 7 * ring;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.rungAt = this.pressAt = this.cueAt = -Infinity; }
}

export default ((() => new BellTowerScene()) satisfies SceneFactory);
