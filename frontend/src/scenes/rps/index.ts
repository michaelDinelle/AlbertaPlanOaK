import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { relief, tokenRing } from '@kit/props';
import { flameMaterial, makeTerrace } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { plantInstances } from '@kit/vegetation';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { RPS_ACTION, RPS_COUNTS, RPS_FILLED, RPS_OPPONENT, rpsCounter } from '@sim/envs/rps';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const MOVE = ['rock', 'paper', 'scissors'] as const;

const RING_Z = 1.7;
const RING_RADIUS = 1.3;
const NOTCH_RADIUS = 5.6;

function glyph(move: number): THREE.BufferGeometry {
  if (move === 0) return new THREE.IcosahedronGeometry(0.16, 1).translate(0, 0.22, 0);
  if (move === 1) return new THREE.BoxGeometry(0.34, 0.03, 0.42).translate(0, 0.12, 0);
  const a = new THREE.CylinderGeometry(0.025, 0.025, 0.42, 6).rotateZ(0.6).translate(0, 0.28, 0);
  const b = new THREE.CylinderGeometry(0.025, 0.025, 0.42, 6).rotateZ(-0.6).translate(0, 0.28, 0);
  const merged = new THREE.BufferGeometry();
  const positions = [...a.toNonIndexed().getAttribute('position').array, ...b.toNonIndexed().getAttribute('position').array];
  merged.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  merged.computeVertexNormals();
  a.dispose(); b.dispose();
  return merged;
}

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const window = parameter(ctx.environment, 'window');
  const t = makeTerrace(ctx, scene, { mood: 'dusk', halfX: 6.5, halfZ: 6.5, ground: [0x8a6b3f, 0xb8955a], landscape: {
    sky: { sunSize: 0.05, sunHdr: 6, sunGlow: 0.25, sunGlowPower: 30 }, light: { pointCount: 2, pointDistance: 6 },
    weather: { preset: 'dust', count: 500, seed: ctx.seed, size: 0.9, bounds: { y: 1, halfX: 7, halfY: 1, halfZ: 7 } },
  } });
  const { world, rig, m, accent, parts } = t;
  rig.sun.intensity = 1.8; rig.hemi.intensity = 0.55;
  const wall = relief(world, 10, 1.6, 0.4, m);
  wall.position.set(0, 0, -6.2);
  world.add(wall);
  const agent = tokenRing(world, 3, RING_RADIUS, accent, m);
  agent.group.position.set(0, 0, RING_Z);
  world.add(agent.group);
  const shade = m.glass(0x2a2438, 0.15);
  const opponent = tokenRing(world, 3, RING_RADIUS, 0x2a2438, m);
  opponent.group.position.set(0, 0, -RING_Z);
  opponent.group.rotation.y = Math.PI;
  world.add(opponent.group);
  for (const ring of [agent, opponent]) ring.tokens.forEach((token, move) => { parts.add(token, glyph(move), ring === agent ? m.brass : shade); });
  const tallies = [0, 1, 2].map(move => {
    const pillar = new THREE.Group();
    pillar.position.set((move - 1) * 1.6, 0, -4.6);
    world.add(pillar);
    parts.box(pillar, m.sandstone, 0, 0.1, 0, 0.6, 0.2, 0.6, 0.03);
    parts.add(pillar, glyph(move).translate(0, 0.2, 0), m.brass);
    const column = new THREE.Mesh(world.own(new THREE.BoxGeometry(0.34, 1, 0.34).translate(0, 0.5, 0)), m.glass(accent, 0.5));
    column.position.y = 0.2;
    column.castShadow = true;
    pillar.add(column);
    return column;
  });
  parts.build();
  const notches = plantInstances(world, 'rpsNotches', new THREE.BoxGeometry(0.14, 0.2, 0.08), m.sandstone.clone(), window);
  for (let i = 0; i < window; i++) {
    const angle = i / window * Math.PI * 2;
    notches.transform.position.set(Math.cos(angle) * NOTCH_RADIUS, 0.1, Math.sin(angle) * NOTCH_RADIUS);
    notches.transform.rotation.set(0, -angle, 0);
    notches.write(i);
  }
  notches.commit(window);
  notches.mesh.castShadow = true;
  const shadow = new THREE.Mesh(world.own(new THREE.CircleGeometry(0.45, 20).rotateX(-Math.PI / 2)), world.own(flameMaterial(0x1a1428, 1)));
  world.add(shadow);
  const sentinel = world.own(buildSentinel({ name: 'rpsPlayer', tint: accent }));
  sentinel.place(0, 0, RING_Z + 2.4);
  sentinel.face(Math.PI);
  world.add(sentinel.group);
  rig.points.forEach((point, i) => { point.color.set(i === 0 ? accent : 0x6a5a9a); point.position.set(0, 1.6, i === 0 ? RING_Z : -RING_Z); point.intensity = 2; });
  const extra = { land: t.land, lights: rig, agent, opponent, tallies, shadow, sentinel, window, accent, failure: ctx.tokens.failureBlue };
  return world.seal<WorldCore & typeof extra>(extra);
}

type ArenaWorld = ReturnType<typeof buildWorld>;

class ArenaScene extends StageScene<ArenaWorld> {
  readonly post: ScenePost = { bloom: 0.36, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'tokenArena';
  private view!: SceneView;
  private lastStep = -1;
  private playedAt = -Infinity;
  private readonly at = new THREE.Vector3();
  private readonly ledger = { move: 0, opponent: 0, reward: 0, counts: [0, 0, 0] };
  override legend: LegendSpec = {
    title: 'ARENA',
    rows: [
      { key: 'move', label: 'played', tint: 0xe89ac2, value: () => MOVE[this.ledger.move]! },
      { key: 'opponent', label: 'opponent played', tint: 0x6a5a9a, value: () => MOVE[this.ledger.opponent]! },
      { key: 'counts', label: 'opponent lately', tint: 0xc9a27a, value: () => `rock ${formatCount(this.ledger.counts[0])} \u00b7 paper ${formatCount(this.ledger.counts[1])} \u00b7 scissors ${formatCount(this.ledger.counts[2])}` },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): ArenaWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-6, -0.5, -6], max: [6, 3, 6] }, [0, 0.6, -0.4], { distance: 12.5, azimuth: 0.7, polar: 1.02 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [12, 12], agentPos: () => this.world!.sentinel.group.position,
      debug: (frame, kit) => {
        const counts = this.ledger.counts;
        let frequent = 0;
        for (let m = 1; m < 3; m++) if (counts[m]! > counts[frequent]!) frequent = m;
        const token = this.world!.agent.tokens[rpsCounter(frame.state.pose[RPS_FILLED]! === 0 ? 0 : rpsCounter(frequent))]!;
        kit.marks('beat', 1).set(token.position.x, 0.8, token.position.z + RING_Z, 0.1, this.ctx!.tokens.celebrationGold);
      },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<ArenaWorld>): void {
    const step = frame.state.step, move = Math.round(frame.state.pose[RPS_ACTION]!) % 3, opponent = Math.round(frame.state.pose[RPS_OPPONENT]!) % 3, filled = frame.state.pose[RPS_FILLED]!;
    this.ledger.move = move; this.ledger.opponent = opponent; this.ledger.reward = frame.state.reward;
    for (let m = 0; m < 3; m++) this.ledger.counts[m] = frame.state.pose[RPS_COUNTS + m]!;
    if (step !== this.lastStep) {
      this.lastStep = step; this.playedAt = time;
      const token = world.agent.tokens[move]!;
      this.at.set(token.position.x, 0.6, token.position.z + RING_Z);
      if (!frame.scrubbed) {
        if (frame.state.reward > 0) pool.burst(this.at, world.accent, 0.6, time);
        else if (frame.state.reward < 0) pool.sigh(0.5, this.at, time);
      }
    }
    const play = envelope(time - this.playedAt, 1, 1.6);
    world.agent.lift(move, 0.25 * play + 0.08);
    world.opponent.lift(opponent, 0.25 * play + 0.08);
    const shadowToken = world.opponent.tokens[rpsCounter(opponent)]!;
    world.shadow.position.set(-shadowToken.position.x, 0.03, -shadowToken.position.z - RING_Z);
    world.shadow.scale.setScalar(0.6 + 0.5 * play);
    world.agent.glass.emissiveIntensity = 0.6 + (frame.state.reward > 0 ? 2 * play : frame.state.reward < 0 ? -0.4 * play : 0);
    world.tallies.forEach((column, m) => { column.scale.y = Math.max(0.02, frame.state.pose[RPS_COUNTS + m]! / world.window * 1.6); });
    world.sentinel.idle(time);
    world.sentinel.glow(frame.state.reward > 0 ? play : 0);
    const point = world.lights.points[0];
    if (point) point.intensity = 2 + 4 * play * Math.max(0, frame.state.reward);
    world.opponent.group.rotation.y = Math.PI + 0.03 * Math.sin(time * 0.8) * Math.min(1, filled / world.window);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.playedAt = -Infinity; }
}

export default ((() => new ArenaScene()) satisfies SceneFactory);
