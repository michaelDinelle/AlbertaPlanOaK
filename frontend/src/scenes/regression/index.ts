import * as THREE from 'three';
import { makeLandscape } from '@kit/landscape';
import { makeMeadow } from '@kit/meadow';
import { envelope } from '@kit/mix';
import { banner, lamp, propMaterials, relief } from '@kit/props';
import { dial } from '@kit/props/dial';
import { lampBank } from '@kit/props/lampbank';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { REGRESSION_ACTION, REGRESSION_EDGE, REGRESSION_WEIGHTS, regressionBin, regressionSum } from '@sim/envs/regression';
import { constant, parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const NEEDLE_REACH = 1.55;
const NEEDLE_SWEEP = Math.PI * 0.42;
const BEAM_Y = 2.1;
const DIAL_RADIUS = 1.15;
const DIAL_Z = 1.6;

const needleAngle = (y: number): number => -Math.max(-1, Math.min(1, y / NEEDLE_REACH)) * NEEDLE_SWEEP;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, inputs = parameter(env, 'inputs'), bins = constant(env, 'REGRESSION_BINS');
  if (bins !== REGRESSION_EDGE.length + 1) throw new Error('Regression bin count does not match its edges');
  const palette = ctx.tokens.palettes.morning, accent = ctx.tokens.accents['regression'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, 'regressionHall', palette, 'morning', { sky: { sunDir: [0.55, 0.42, 0.72], sunSize: 0.05, sunHdr: 6, sunGlow: 0.4, sunGlowPower: 18 }, light: { pointCount: 2, pointDistance: 6 }, fogExtent: 110, weather: { preset: 'pollen', count: 200, seed: ctx.seed, color: 0xffe3a8, size: 0.7, intensity: 0.8, bounds: { y: 1.5, halfX: 7, halfY: 1.3, halfZ: 5 } } });
  const { world, rig } = land;
  rig.sun.intensity = 2.3; rig.sun.color.set(0xfff0cf); rig.hemi.intensity = 0.75;
  rig.setShadowFocus({ x: 0, y: 1, z: -0.5 }, 8);
  const meadow = makeMeadow(world, ctx, { name: 'regressionHall', halfX: 6.5, halfZ: 4.5, ground: 0x9a7b46, grass: 0xc9a352, trees: [0xd98a3a, 0xe8b84a, 0xc25c2e, 0xf0cf6a], amplitude: 1.3 });
  const m = propMaterials(world), parts = modelParts(world);
  const wall = relief(world, 11, 2.4, 0.6, m);
  wall.position.set(0, 0, -3.6);
  world.add(wall);
  for (const x of [-5.2, 5.2]) { const b = banner(world, accent, 2.8, m); b.position.set(x, 0, -3); world.add(b); }
  const lamps = [-3.6, 3.6].map(x => { const l = lamp(world, accent, 1.6, m); l.group.position.set(x, 0, -2.4); world.add(l.group); l.lit(0.5); return l; });
  const span = inputs * 0.42 + 0.6, beam = new THREE.Group();
  world.add(beam);
  for (const side of [-1, 1]) {
    parts.box(beam, m.sandstone, side * (span / 2 + 0.25), BEAM_Y / 2, -1.2, 0.42, BEAM_Y, 0.42, 0.04);
    parts.box(beam, m.brass, side * (span / 2 + 0.25), BEAM_Y + 0.03, -1.2, 0.5, 0.06, 0.5, 0.01);
  }
  parts.box(beam, m.brass, 0, BEAM_Y + 0.08, -1.2, span + 0.6, 0.1, 0.16, 0.02);
  parts.box(beam, m.sandstone, 0, 0.22, -1.2, span + 0.9, 0.44, 0.9, 0.04);
  parts.box(beam, m.brass, 0, 0.47, -1.2, span + 0.7, 0.05, 0.7, 0.01);
  parts.build();
  const slot = (j: number): number => (j - (inputs - 1) / 2) * 0.42;
  const weights = lampBank(world, new THREE.CylinderGeometry(0.1, 0.12, 0.26, 12).translate(0, -0.13, 0), inputs, { emissive: 0.5, roughness: 0.4, metalness: 0.5, castShadow: true });
  world.add(weights.mesh);
  const strings = lampBank(world, new THREE.CylinderGeometry(0.012, 0.012, 1, 6).translate(0, -0.5, 0), inputs, { emissive: 0.15, roughness: 0.6, metalness: 0.4 });
  world.add(strings.mesh);
  const readings = lampBank(world, new THREE.SphereGeometry(0.11, 12, 10), inputs, { emissive: 1.8 });
  world.add(readings.mesh);
  for (let j = 0; j < inputs; j++) { readings.place(j, slot(j), 0.62, -1.2); readings.tint(j, accent); readings.glow(j, 0.2); }
  readings.commit();
  const scale = dial(world, DIAL_RADIUS, 20, accent, m, 5);
  scale.group.position.set(0, 0, DIAL_Z);
  world.add(scale.group);
  const sectorMaterials = Array.from({ length: bins }, () => m.glass(accent, 0.35));
  const sectors = sectorMaterials.map((material, bin) => {
    const lo = bin === 0 ? NEEDLE_SWEEP * 1.18 : needleAngle(REGRESSION_EDGE[bin - 1]!), hi = bin === bins - 1 ? -NEEDLE_SWEEP * 1.18 : needleAngle(REGRESSION_EDGE[bin]!);
    const a0 = Math.min(lo, hi), a1 = Math.max(lo, hi);
    const mesh = new THREE.Mesh(world.own(new THREE.RingGeometry(DIAL_RADIUS * 0.52, DIAL_RADIUS * 0.86, 12, 1, -a1 - Math.PI / 2, a1 - a0).rotateX(-Math.PI / 2)), material);
    mesh.position.set(0, 0.135, DIAL_Z);
    world.add(mesh);
    return mesh;
  });
  const extra = { land, meadow, inputs, bins, weights, strings, readings, scale, sectorMaterials, sectors, lamps, accent, gold, slot, dialCenter: new THREE.Vector3(0, 0.2, DIAL_Z) };
  return world.seal<WorldCore & typeof extra>(extra);
}

type HallWorld = ReturnType<typeof buildWorld>;

class WeighingHallScene extends StageScene<HallWorld> {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'regression';
  private view!: SceneView;
  private step = -1;
  private verdictAt = -Infinity;
  private chosen = -1;
  private correct = false;
  private readonly warm = new THREE.Color(0xe0b46a);
  private readonly cool = new THREE.Color(0x7fa7c9);
  private readonly ledger = { sum: 0, chosen: -1, truth: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'WEIGHING HALL',
    rows: [
      { key: 'sum', label: 'the scale reads', tint: 0x86c7f0, value: () => formatSigned(this.ledger.sum) },
      { key: 'truth', label: 'true bin', tint: 0xffd166, value: () => `${this.ledger.truth + 1} of ${this.world!.bins}` },
      { key: 'chosen', label: 'answered bin', tint: 0xdcf4ff, value: () => (this.ledger.chosen < 0 ? '\u2014' : `${this.ledger.chosen + 1}`) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): HallWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const span = this.world!.inputs * 0.42 + 2;
    this.view = makeSceneView(ctx, { min: [-span / 2 - 1, -0.5, -4], max: [span / 2 + 1, 3.5, 3] }, [0, 0.9, -0.2], { distance: Math.max(9.5, span * 1.15), azimuth: 0.32, polar: 1.05 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [this.world!.slot(this.world!.inputs - 1) * 2 + 4, 8] }; }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<HallWorld>): void {
    const { state } = frame;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.verdictAt = -Infinity; }
    if (state.step !== this.step) {
      this.step = state.step;
      const asked = frame.frameAt(state.step - 1), y = regressionSum(asked, world.inputs);
      this.chosen = state.step === 0 ? -1 : Math.trunc(state.pose[REGRESSION_ACTION]!) % world.bins;
      this.correct = state.reward > 0;
      if (this.chosen >= 0 && regressionBin(y) !== this.chosen && this.correct) throw new Error('Regression reward disagrees with its bin edges');
      this.verdictAt = time;
      this.ledger.sum = y; this.ledger.chosen = this.chosen; this.ledger.truth = regressionBin(y); this.ledger.reward = state.reward;
      world.scale.setAngle(needleAngle(y));
      for (let j = 0; j < world.inputs; j++) {
        const w = state.pose[REGRESSION_WEIGHTS + j]!, x = asked.observations[j]!;
        world.weights.place(j, world.slot(j), BEAM_Y - 0.25 - w * 0.5, -1.2, 0, 0.7 + Math.min(1.2, Math.abs(w)) * 0.6);
        world.weights.tint(j, w >= 0 ? this.warm : this.cool);
        world.strings.place(j, world.slot(j), BEAM_Y + 0.03, -1.2, 0, 1, 0.28 - w * 0.5);
        world.strings.tint(j, this.warm, 0.6);
        world.readings.tint(j, x >= 0 ? this.warm : this.cool);
        world.readings.glow(j, 0.15 + Math.min(2.4, Math.abs(x)) * 0.9);
      }
      world.weights.commit(); world.strings.commit();
    }
    const verdict = envelope(time - this.verdictAt, 1 / frame.controlHz + 0.35, 1.6);
    world.sectorMaterials.forEach((material, bin) => {
      const chosen = bin === this.chosen;
      material.color.set(chosen && this.correct ? world.gold : chosen ? this.ctx!.tokens.failureBlue : world.accent);
      material.emissive.copy(material.color);
      material.emissiveIntensity = chosen ? 0.6 + (this.correct ? 2.4 : 1.2) * verdict : 0.3;
    });
    world.scale.glass.emissiveIntensity = 1 + 1.5 * verdict * (this.correct ? 1 : 0.3);
    world.lamps.forEach((l, i) => l.lit(0.45 + 0.2 * Math.sin(time * 1.2 + i * 2.1)));
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera, world.meadow);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.verdictAt = -Infinity; this.chosen = -1; }
}

export default ((() => new WeighingHallScene()) satisfies SceneFactory);
