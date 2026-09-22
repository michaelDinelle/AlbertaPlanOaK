import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { makeMistPlanes } from '@kit/particles';
import { doorway, relief } from '@kit/props';
import { flameMaterial, makeTerrace, yawToward } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { scatterTrees } from '@kit/vegetation';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { TIGER_ACTION, TIGER_HEARD, TIGER_HINT, TIGER_LISTEN, TIGER_SIDE, TIGER_TALLY } from '@sim/envs/tiger';
import { parameter } from '../../data/envs';
import { formatPercent, formatSigned } from '../../format';

const DOOR_X = 2.1;
const WALL_Z = -2.8;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const accuracy = parameter(ctx.environment, 'accuracy'), murk = 1 - accuracy;
  const t = makeTerrace(ctx, scene, { mood: 'night', halfX: 6, halfZ: 4.5, ground: [0x1e3324, 0x2f4a2c], amplitude: 1.8, landscape: {
    sky: { stars: { intensity: 1.6 } }, light: { pointCount: 1, pointDistance: 7 }, fogDensity: 0.03 + 0.06 * murk,
    weather: { preset: 'fireflies', count: 140 + Math.round(200 * murk), seed: ctx.seed, size: 2.4, bounds: { y: 1.3, halfX: 8, halfY: 1.2, halfZ: 6 } },
  } });
  const { world, rig, m, accent, parts, heightAt } = t;
  rig.sun.intensity = 0.4; rig.hemi.intensity = 0.55;
  const mist = world.time(world.own(makeMistPlanes({ size: [18, 14], color: 0x8aa39a, seed: ctx.seed, layers: 4, baseY: 0.15, spread: 0.9, opacity: 0.12 + 0.3 * murk, name: 'tigerMist' })));
  world.add(mist.group);
  const trees = world.time(world.own(scatterTrees({ archetypes: ['broadleaf'], count: 70, area: { halfX: 22, halfZ: 18 }, seed: ctx.seed, exclude: (x, z) => Math.abs(x) < 9 && z > -6 && z < 13, heightAt, color: 0x2c5a37, scale: [1.2, 2.2], stretch: [1, 1.5], wind: 0.03, rimColor: 0x6fa08a, rimStrength: 0.3, castShadow: true, name: 'jungleCanopy' })));
  world.add(trees.group);
  for (const [x, width] of [[-4.9, 3.6], [0, 1.6], [4.9, 3.6]] as const) { const wall = relief(world, width, 2.6, 1.1, m); wall.position.set(x, 0, WALL_Z); world.add(wall); }
  const leafGeometry = world.own(new THREE.BoxGeometry(1.1, 2.1, 0.08).translate(0.55, 1.05, 0));
  const doors = [-1, 1].map(side => {
    const frame = doorway(world, 1.2, 2.2, m);
    frame.position.set(side * DOOR_X, 0, WALL_Z);
    world.add(frame);
    const hinge = new THREE.Group();
    hinge.position.set(side * DOOR_X - 0.55, 0, WALL_Z);
    world.add(hinge);
    const leaf = new THREE.Mesh(leafGeometry, m.shale);
    leaf.castShadow = true;
    hinge.add(leaf);
    const eyes = new THREE.Group();
    eyes.position.set(side * DOOR_X, 1.1, WALL_Z - 1.2);
    world.add(eyes);
    const eyeMaterial = world.own(flameMaterial(0xffb347, 2));
    for (const dx of [-0.12, 0.12]) { const eye = new THREE.Mesh(world.own(new THREE.SphereGeometry(0.055, 8, 6)), eyeMaterial); eye.position.x = dx; eyes.add(eye); }
    const glow = new THREE.PointLight(0xffa040, 0, 6, 1.6);
    glow.position.set(0, 0.3, 0.4);
    eyes.add(glow);
    return { hinge, eyes, eyeMaterial, glow, x: side * DOOR_X, sill: new THREE.Vector3(side * DOOR_X, 0.05, WALL_Z + 0.6), top: new THREE.Vector3(side * DOOR_X, 2.4, WALL_Z) };
  });
  parts.build();
  const sentinel = world.own(buildSentinel({ name: 'tigerListener', tint: accent }));
  sentinel.place(0, 0, 1.4);
  world.add(sentinel.group);
  const point = rig.points[0];
  if (point) { point.color.set(accent); point.position.set(0, 2, 0.6); point.intensity = 2; }
  const extra = { land: t.land, lights: rig, doors, sentinel, accent, cueCyan: ctx.tokens.cueCyan, cueMagenta: ctx.tokens.cueMagenta };
  return world.seal<WorldCore & typeof extra>(extra);
}

type JungleWorld = ReturnType<typeof buildWorld>;

class JungleDoorsScene extends StageScene<JungleWorld> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'jungleDoors';
  private view!: SceneView;
  private lastStep = -1;
  private openedAt = -Infinity;
  private opened = 0;
  private outcome = 0;
  private listenAt = -Infinity;
  private readonly ledger = { tiger: 0, action: 0, heard: false, hint: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'JUNGLE DOORS',
    rows: [
      { key: 'tiger', label: 'tiger waits behind', tint: 0xffb347, value: () => (this.ledger.tiger ? 'the right door' : 'the left door') },
      { key: 'heard', label: 'last growl', tint: 0x5ee0ff, value: () => (this.ledger.heard ? (this.ledger.hint ? 'from the right' : 'from the left') : 'silence') },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.action === TIGER_LISTEN ? 'listen' : this.ledger.action & 1 ? 'open the right door' : 'open the left door') },
      { key: 'accuracy', label: 'growls are right', tint: 0xffb347, value: () => formatPercent(parameter(this.ctx!.environment, 'accuracy'), 0) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): JungleWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-6, -0.5, -4], max: [6, 3.5, 4] }, [0, 1, -0.8], { distance: 10.5, azimuth: 0.2, polar: 1.15 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [12, 8], agentPos: () => this.world!.sentinel.group.position,
      debug: (frame, kit) => {
        const tiger = this.world!.doors[this.ledger.tiger]!, tally = Math.round(frame.state.pose[TIGER_TALLY]!);
        kit.marks('tiger', 1).set(tiger.x, 3.0, WALL_Z, 0.16, 0xff6b57);
        kit.bars('tally', 1).set(0, 0.05, 1.4, tally * 0.25, tally > 0 ? this.ctx!.tokens.uiCyan : this.ctx!.tokens.celebrationGold, 0.14);
      },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<JungleWorld>): void {
    const step = frame.state.step, action = Math.round(frame.state.pose[TIGER_ACTION]!);
    const tigerNow = Math.round(frame.state.pose[TIGER_SIDE]!) & 1;
    const before = frame.frameAt(step - 1);
    const tigerBefore = step > 0 ? before.words[0]! & 1 : tigerNow;
    const heard = Math.round(frame.state.pose[TIGER_HEARD]!) === 1, hint = Math.round(frame.state.pose[TIGER_HINT]!) & 1;
    this.ledger.tiger = tigerNow; this.ledger.action = action; this.ledger.heard = heard; this.ledger.hint = hint; this.ledger.reward = frame.state.reward;
    if (step !== this.lastStep) {
      this.lastStep = step;
      if (action === TIGER_LISTEN) {
        this.listenAt = time;
        if (heard && !frame.scrubbed) pool.ripple(world.doors[hint]!.sill, hint === tigerBefore ? world.cueCyan : world.cueMagenta, 0.5, time);
      } else if (step > 0) {
        this.openedAt = time; this.opened = action & 1; this.outcome = frame.state.reward;
        if (!frame.scrubbed) {
          if (frame.state.reward > 0) pool.burst(world.doors[this.opened]!.top, world.accent, 0.8, time);
          else pool.sigh(1, world.doors[this.opened]!.sill, time);
        }
      }
    }
    const swing = envelope(time - this.openedAt, 1.6, 1.2), listen = envelope(time - this.listenAt, 0.8, 2);
    world.doors.forEach((door, side) => {
      const tiger = time - this.openedAt < 1.6 ? (side === this.opened ? this.outcome < 0 : this.outcome > 0 && side !== this.opened) : side === tigerNow;
      door.hinge.rotation.y = side === this.opened ? -swing * 1.3 * (side === 0 ? 1 : -1) : 0;
      const roar = side === this.opened && this.outcome < 0 ? swing * 5 : 0;
      door.glow.intensity = tiger ? 1.4 + roar + 0.3 * Math.sin(time * 7) : 0;
      door.eyeMaterial.opacity = tiger ? 0.6 + 0.4 * Math.abs(Math.sin(time * 1.3)) : 0;
    });
    world.sentinel.idle(time);
    world.sentinel.face(action === TIGER_LISTEN ? Math.sin(time * 2) * 0.4 : yawToward(0, 1.4, world.doors[this.opened]!.x, WALL_Z));
    world.sentinel.glow(listen + (this.outcome > 0 ? swing : 0));
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.openedAt = this.listenAt = -Infinity; }
}

export default ((() => new JungleDoorsScene()) satisfies SceneFactory);
