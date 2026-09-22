import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { lamp, plinth } from '@kit/props';
import { flameMaterial, makeTerrace, yawToward } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { CONTEXTUAL_ACTION, contextualDrift, contextualWeights } from '@sim/envs/contextual';
import { constant, parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const CHECKPOINT = 32768;
const CATCHUP_STEPS = 3000;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const inputs = parameter(ctx.environment, 'inputs'), drift = parameter(ctx.environment, 'drift'), arms = constant(ctx.environment, 'CONTEXTUAL_ARMS');
  const t = makeTerrace(ctx, scene, { mood: 'night', halfX: 5.5, halfZ: 5.5, ground: [0x3a3524, 0x5b4d2c], landscape: {
    sky: { stars: { intensity: 3, density: 220 }, sunHdr: 1.5 },
    light: { pointCount: 0 },
    weather: { preset: 'fireflies', count: 160, seed: ctx.seed, color: 0xd9b36a, size: 2.2, bounds: { y: 1.2, halfX: 7, halfY: 1, halfZ: 7 } },
  } });
  const { world, rig, m, accent } = t;
  rig.sun.intensity = 0.55; rig.hemi.intensity = 0.5;
  const dais = plinth(world, 3.4, 0.35, m);
  world.add(dais);
  const lamps = Array.from({ length: arms }, (_, a) => {
    const angle = a / arms * Math.PI * 2 + Math.PI / 4, x = Math.cos(angle) * 4.4, z = Math.sin(angle) * 4.4;
    const l = lamp(world, accent, 1.7, m);
    l.group.position.set(x, 0, z);
    world.add(l.group);
    return { ...l, x, z, top: new THREE.Vector3(x, 1.95, z) };
  });
  const sentinel = world.own(buildSentinel({ name: 'contextualAstronomer', tint: accent }));
  sentinel.place(0, 0.35, 0);
  world.add(sentinel.group);
  const beamMaterial = world.own(flameMaterial(accent, 1.6));
  const beam = new THREE.Mesh(world.own(new THREE.CylinderGeometry(0.012, 0.03, 1, 6).rotateX(Math.PI / 2).translate(0, 0, 0.5)), beamMaterial);
  beam.position.set(0, 1.9, 0);
  world.add(beam);
  const constellation = new THREE.Group();
  constellation.position.y = 5;
  world.add(constellation);
  const stars = world.own(new THREE.InstancedMesh(world.own(new THREE.SphereGeometry(1, 8, 6)), world.own(new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false })), inputs));
  stars.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(inputs * 3), 3);
  constellation.add(stars);
  const threads = new THREE.LineSegments(world.own(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(inputs * 6), 3))), world.own(new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.35 })));
  threads.frustumCulled = false;
  constellation.add(threads);
  const weights = contextualWeights(ctx.seed, arms, inputs);
  const extra = { land: t.land, lamps, sentinel, beam, beamMaterial, constellation, stars, threads, weights, drift, inputs, arms, accent };
  return world.seal<WorldCore & typeof extra>(extra);
}

type ObservatoryWorld = ReturnType<typeof buildWorld>;

class LampObservatoryScene extends StageScene<ObservatoryWorld> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'lampObservatory';
  private view!: SceneView;
  private weightStep = 0;
  private readonly checkpoints = new Map<number, Float32Array>();
  private lastStep = -1;
  private pullAt = -Infinity;
  private pulled = 0;
  private readonly star = new THREE.Object3D();
  private readonly color = new THREE.Color();
  private readonly values = new Float32Array(4);
  private readonly ledger = { pulled: 0, best: 0, expected: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'OBSERVATORY',
    rows: [
      { key: 'pulled', label: 'lamp lit', tint: 0xffa76e, value: () => `lamp ${this.ledger.pulled + 1} of ${this.world!.arms}` },
      { key: 'best', label: 'best lamp tonight', tint: 0xffe2a0, value: () => `lamp ${this.ledger.best + 1}` },
      { key: 'expected', label: 'worth of the lit lamp', tint: 0xffd166, value: () => formatSigned(this.ledger.expected) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): ObservatoryWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-6, -0.5, -6], max: [6, 7, 6] }, [0, 1.6, 0], { distance: 13, azimuth: 0.55, polar: 1.05 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [12, 12], agentPos: () => this.world!.sentinel.group.position }; }
  protected start(_ctx: SceneContext, world: ObservatoryWorld): void { this.checkpoints.set(0, world.weights.slice()); }

  private advanceWeights(world: ObservatoryWorld, seed: number, target: number): void {
    if (world.drift === 0) return;
    if (target < this.weightStep) {
      const base = Math.floor(target / CHECKPOINT) * CHECKPOINT;
      let at = base;
      while (!this.checkpoints.has(at)) at -= CHECKPOINT;
      world.weights.set(this.checkpoints.get(at)!);
      this.weightStep = at;
    }
    const until = Math.min(target, this.weightStep + CATCHUP_STEPS);
    while (this.weightStep < until) {
      this.weightStep++;
      contextualDrift(seed, this.weightStep, world.drift, world.weights);
      if (this.weightStep % CHECKPOINT === 0 && !this.checkpoints.has(this.weightStep)) this.checkpoints.set(this.weightStep, world.weights.slice());
    }
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<ObservatoryWorld>): void {
    const step = frame.state.step, obs = frame.state.observations;
    this.advanceWeights(world, ctx.seed, step);
    const pulled = Math.min(world.arms - 1, Math.max(0, Math.round(frame.state.pose[CONTEXTUAL_ACTION]!)));
    if (step !== this.lastStep) {
      this.lastStep = step; this.pulled = pulled; this.pullAt = time;
      const target = world.lamps[pulled]!;
      if (!frame.scrubbed) {
        if (frame.state.reward > 0.3) pool.burst(target.top, world.accent, Math.min(1, frame.state.reward), time);
        else if (frame.state.reward < -0.3) pool.sigh(Math.min(1, -frame.state.reward), target.top, time);
      }
      const positions = world.threads.geometry.getAttribute('position') as THREE.BufferAttribute;
      for (let j = 0; j < world.inputs; j++) {
        const angle = j / world.inputs * Math.PI * 2, value = obs[j] ?? 0, radius = 3.6 + 1.2 * Math.sin(j * 2.3);
        this.star.position.set(Math.cos(angle) * radius, 0.8 * Math.sin(j * 1.7), Math.sin(angle) * radius);
        this.star.scale.setScalar(0.035 + 0.045 * Math.min(2.5, Math.abs(value)));
        this.star.updateMatrix();
        world.stars.setMatrixAt(j, this.star.matrix);
        world.stars.setColorAt(j, this.color.set(value >= 0 ? 0xffe2a0 : 0x8fb8ff).multiplyScalar(1.2 + Math.min(2.5, Math.abs(value))));
        positions.setXYZ(j * 2, this.star.position.x, this.star.position.y, this.star.position.z);
        positions.setXYZ(j * 2 + 1, 0, -3.1, 0);
      }
      world.stars.instanceMatrix.needsUpdate = true;
      world.stars.instanceColor!.needsUpdate = true;
      positions.needsUpdate = true;
    }
    let lo = Infinity, hi = -Infinity, best = 0;
    for (let a = 0; a < world.arms; a++) {
      let value = 0;
      for (let j = 0; j < world.inputs; j++) value += world.weights[a * world.inputs + j]! * (obs[j] ?? 0);
      this.values[a] = value; if (value > hi) best = a; lo = Math.min(lo, value); hi = Math.max(hi, value);
    }
    this.ledger.pulled = pulled; this.ledger.best = best; this.ledger.expected = this.values[pulled]!; this.ledger.reward = frame.state.reward;
    const span = hi - lo || 1, flare = envelope(time - this.pullAt, 1, 2.2);
    world.lamps.forEach((l, a) => l.lit(0.15 + 0.85 * (this.values[a]! - lo) / span + (a === this.pulled ? flare * 1.2 : 0)));
    const target = world.lamps[this.pulled]!, yaw = yawToward(0, 0, target.x, target.z);
    world.beam.rotation.y = yaw;
    world.beam.scale.set(1 + flare, 1 + flare, Math.hypot(target.x, target.z));
    world.beamMaterial.opacity = 0.12 + 0.5 * flare;
    world.sentinel.idle(time);
    world.sentinel.face(yaw);
    world.sentinel.glow(flare);
    world.constellation.rotation.y = world.drift > 0 ? time * 0.05 : 0;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.pullAt = -Infinity; this.weightStep = 0; this.checkpoints.clear(); }
}

export default ((() => new LampObservatoryScene()) satisfies SceneFactory);
