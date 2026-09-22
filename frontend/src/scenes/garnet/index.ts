import * as THREE from 'three';
import { buildAurora } from '@kit/aurora';
import { envelope } from '@kit/mix';
import { flameMaterial, makeTerrace } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { plantInstances } from '@kit/vegetation';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import type { WorldCore } from '@kit/world';
import { GARNET_ACTION, GARNET_STATE, garnetModelOf, type GarnetModel } from '@sim/envs/garnet';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const ARC_SEGMENTS = 6;
const ACTION_TINTS = [0x5ee0ff, 0xff5ec8, 0xffd166, 0x9fea9a] as const;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const states = parameter(ctx.environment, 'states'), actions = parameter(ctx.environment, 'actions'), branching = parameter(ctx.environment, 'branching');
  if (actions > ACTION_TINTS.length) throw new Error('Garnet action count exceeds the scene palette');
  const radius = Math.max(3.4, states * 0.16), half = radius + 2;
  const t = makeTerrace(ctx, scene, { mood: 'night', halfX: half, halfZ: half, ground: [0x4a4634, 0x6e6a4c], amplitude: 1.6, paving: false, landscape: {
    sky: { stars: { intensity: 2.2 } }, light: { pointCount: 1, pointDistance: 5 },
    weather: { preset: 'pollen', count: 140, seed: ctx.seed, color: 0xc9a86a, size: 0.8, bounds: { y: 1.2, halfX: half, halfY: 1, halfZ: half } },
  } });
  const { world, rig, m, accent } = t;
  rig.sun.color.set(0xbfd4ff); rig.sun.intensity = 0.9; rig.hemi.color.set(0x55688f); rig.hemi.intensity = 0.9;
  const aurora = world.own(buildAurora(t.land.sky, 0x66e0a8, 0.6));
  const stoneGeometry = new THREE.CylinderGeometry(0.22, 0.32, 1.5, 7).translate(0, 0.75, 0);
  const stones = plantInstances(world, 'garnetStones', stoneGeometry, m.sandstone.clone(), states);
  stones.mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(states * 3), 3);
  const positions: THREE.Vector3[] = [], color = new THREE.Color(0xffffff);
  for (let s = 0; s < states; s++) {
    const angle = s / states * Math.PI * 2, ring = states > 20 && s % 2 ? radius - 1.1 : radius;
    positions.push(new THREE.Vector3(Math.cos(angle) * ring, 0, Math.sin(angle) * ring));
    stones.transform.position.copy(positions[s]!);
    stones.transform.rotation.set(0, -angle + 0.3 * Math.sin(s * 3.1), 0.05 * Math.sin(s * 1.7));
    stones.transform.scale.set(1, 0.85 + 0.3 * Math.abs(Math.sin(s * 2.3)), 1);
    stones.write(s, color);
  }
  stones.commit(states);
  stones.mesh.castShadow = stones.mesh.receiveShadow = true;
  const edgeCount = states * actions * branching, vertexCount = edgeCount * ARC_SEGMENTS * 2;
  const edgeGeometry = world.own(new THREE.BufferGeometry());
  edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3), 3));
  edgeGeometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(vertexCount * 3), 3));
  const edges = new THREE.LineSegments(edgeGeometry, world.own(new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false })));
  edges.frustumCulled = false;
  world.add(edges);
  const runes = plantInstances(world, 'garnetRunes', new THREE.OctahedronGeometry(0.11, 0), new THREE.MeshBasicMaterial({ toneMapped: false }), states);
  runes.mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(states * 3), 3);
  const wisp = world.own(new Wisp({ color: accent, radius: 0.3, height: 0.8, hop: { duration: 0.16, height: 0.4 }, name: 'garnetWalker' }));
  world.add(wisp.group);
  const beacon = new THREE.Mesh(world.own(new THREE.CylinderGeometry(0.34, 0.4, 0.05, 7)), world.own(flameMaterial(accent, 1.2)));
  world.add(beacon);
  const point = rig.points[0];
  if (point) { point.color.set(0xffd7a8); point.intensity = 1.5; }
  const extra = { land: t.land, lights: rig, aurora, stones, positions, edges, runes, wisp, beacon, states, actions, branching, accent, gold: ctx.tokens.celebrationGold, period: constant(ctx.environment, 'GARNET_PERIOD'), resample: parameter(ctx.environment, 'resample') };
  return world.seal<WorldCore & typeof extra>(extra);
}

type StoneWorld = ReturnType<typeof buildWorld>;

class StandingStonesScene extends StageScene<StoneWorld> {
  readonly post: ScenePost = { bloom: 0.52, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'standingStones';
  private view!: SceneView;
  private epoch = -1;
  private model: GarnetModel | null = null;
  private lastStep = -1;
  private state = 0;
  private arrivedAt = -Infinity;
  private rebuiltAt = -Infinity;
  private readonly color = new THREE.Color();
  private readonly at = new THREE.Vector3();
  private step = 0;
  private readonly ledger = { state: 0, action: 0, greedy: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'STANDING STONES',
    rows: [
      { key: 'state', label: 'stone', tint: 0xe0666e, value: () => `${this.ledger.state + 1} of ${this.world!.states}` },
      { key: 'action', label: 'path taken', tint: 0xdcf4ff, value: () => `${this.ledger.action + 1} of ${this.world!.actions}` },
      { key: 'greedy', label: 'best path here', tint: 0xffd166, value: () => `${this.ledger.greedy + 1}` },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): StoneWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const half = Math.max(3.4, this.world!.states * 0.16) + 1;
    this.view = makeSceneView(ctx, { min: [-half, -0.5, -half], max: [half, 4, half] }, [0, 0.8, 0], { distance: half * 2.9, azimuth: 0.4, polar: 1.3 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [this.world!.states * 0.32 + 8, this.world!.states * 0.32 + 8], agentPos: () => this.world!.wisp.group.position,
      debug: (_frame, kit) => {
        const world = this.world!, model = this.model, state = this.ledger.state, gold = this.ctx!.tokens.celebrationGold, cyan = this.ctx!.tokens.uiCyan;
        if (!model) return;
        const here = world.positions[state]!, marks = kit.marks('next', world.actions * world.branching);
        for (let a = 0; a < world.actions; a++) {
          const pair = state * world.actions + a, greedy = model.greedy[state] === a, tint = greedy ? gold : ACTION_TINTS[a]!;
          for (let b = 0; b < world.branching; b++) { const to = world.positions[model.next[pair * world.branching + b]!]!; marks.set(to.x, 1.9, to.z, 0.05 + 0.14 * model.prob[pair * world.branching + b]!, tint); }
        }
        kit.trail('walk', 40, cyan).push(here.x, 1.65, here.z);
      },
    };
  }
  protected start(_ctx: SceneContext, world: StoneWorld): void {
    world.wisp.snap();
    if (world.resample) { this.legend.note = () => `stones reshuffle in ${formatCount(world.period - this.step % world.period)} steps`; this.legend.bar = () => (this.step % world.period) / world.period; }
  }

  private rebuild(ctx: SceneContext, world: StoneWorld, step: number): void {
    const epoch = world.resample ? Math.floor(step / world.period) : 0;
    if (epoch === this.epoch) return;
    this.epoch = epoch;
    const model = this.model = garnetModelOf(ctx.environment, ctx.seed, step);
    const positions = world.edges.geometry.getAttribute('position') as THREE.BufferAttribute, colors = world.edges.geometry.getAttribute('color') as THREE.BufferAttribute;
    let v = 0;
    for (let s = 0; s < world.states; s++) for (let a = 0; a < world.actions; a++) {
      const pair = s * world.actions + a, greedy = model.greedy[s] === a;
      this.color.set(ACTION_TINTS[a]!).multiplyScalar(greedy ? 1 : 0.35);
      for (let b = 0; b < world.branching; b++) {
        const from = world.positions[s]!, to = world.positions[model.next[pair * world.branching + b]!]!, weight = model.prob[pair * world.branching + b]!;
        const lift = 0.9 + from.distanceTo(to) * 0.35 + a * 0.18;
        for (let k = 0; k < ARC_SEGMENTS; k++) for (const end of [k, k + 1]) {
          const u = end / ARC_SEGMENTS;
          positions.setXYZ(v, from.x + (to.x - from.x) * u, 1.1 + 4 * u * (1 - u) * lift, from.z + (to.z - from.z) * u);
          colors.setXYZ(v, this.color.r * weight, this.color.g * weight, this.color.b * weight);
          v++;
        }
      }
    }
    positions.needsUpdate = true; colors.needsUpdate = true;
    for (let s = 0; s < world.states; s++) {
      let best = 0;
      for (let a = 0; a < world.actions; a++) best = Math.abs(model.reward[s * world.actions + a]!) > Math.abs(best) ? model.reward[s * world.actions + a]! : best;
      world.runes.transform.position.copy(world.positions[s]!).setY(1.75);
      world.runes.transform.scale.setScalar(best === 0 ? 0 : 0.7 + Math.min(1.5, Math.abs(best)));
      world.runes.write(s, this.color.set(best > 0 ? world.gold : 0x7a95c4).multiplyScalar(2));
    }
    world.runes.commit(world.states);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, time }: Stage<StoneWorld>): void {
    const step = frame.state.step, state = Math.round(frame.state.pose[GARNET_STATE]!), action = Math.round(frame.state.pose[GARNET_ACTION]!);
    if (state < 0 || state >= world.states) throw new Error('Garnet state is outside its recorded model');
    const before = this.epoch;
    this.rebuild(ctx, world, step);
    if (before !== -1 && before !== this.epoch) { this.rebuiltAt = time; world.stones.mesh.instanceColor!.needsUpdate = true; }
    const model = this.model!;
    this.step = step; this.ledger.state = state; this.ledger.action = action; this.ledger.greedy = model.greedy[state] ?? 0; this.ledger.reward = frame.state.reward;
    if (step !== this.lastStep) {
      this.lastStep = step; this.arrivedAt = time;
      if (frame.scrubbed || frame.prev === null) { world.wisp.group.position.copy(world.positions[state]!); world.wisp.snap(); }
      else if (state !== this.state) world.wisp.group.position.copy(world.positions[state]!);
      this.state = state;
      if (!frame.scrubbed && frame.state.reward !== 0) {
        this.at.copy(world.positions[state]!).setY(1.6);
        if (frame.state.reward > 0) pool.burst(this.at, world.gold, Math.min(1, frame.state.reward), time);
        else pool.sigh(Math.min(1, -frame.state.reward), this.at, time);
      }
    }
    const arrive = envelope(time - this.arrivedAt, 0.8, 2), flash = envelope(time - this.rebuiltAt, 2.5, 1);
    for (let s = 0; s < world.states; s++) world.stones.mesh.setColorAt(s, this.color.set(s === state ? world.accent : 0xffffff).multiplyScalar(s === state ? 1.2 + 0.4 * arrive : 1));
    world.stones.mesh.instanceColor!.needsUpdate = true;
    world.beacon.position.copy(world.positions[state]!).setY(0.03);
    world.beacon.scale.setScalar(1 + 0.6 * arrive);
    (world.beacon.material as THREE.MeshBasicMaterial).opacity = 0.5 + 0.5 * arrive;
    world.wisp.update(frame.dtReal, time);
    world.wisp.flare(model.greedy[state] === action ? arrive * 0.5 : 0);
    const point = world.lights.points[0];
    if (point) { point.position.copy(world.positions[state]!).setY(1.8); point.intensity = 1.2 + 2 * arrive; }
    world.aurora.setGain(1 + 3 * flash);
    world.aurora.setTime(time);
    world.edges.material.opacity = 0.4 + 0.5 * flash;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(this.rig!, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.epoch = -1; this.model = null; this.arrivedAt = this.rebuiltAt = -Infinity; }
}

export default ((() => new StandingStonesScene()) satisfies SceneFactory);
