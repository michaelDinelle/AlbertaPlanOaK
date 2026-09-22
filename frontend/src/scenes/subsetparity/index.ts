import * as THREE from 'three';
import { makeLandscape } from '@kit/landscape';
import { envelope } from '@kit/mix';
import { lamp, propMaterials } from '@kit/props';
import { court, groves } from '@kit/props/court';
import { lampBank } from '@kit/props/lampbank';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { SUBSETPARITY_ACTION, SUBSETPARITY_MASK } from '@sim/envs/subsetparity';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const popcount = (v: number): number => { let n = 0; for (let x = v >>> 0; x !== 0; x >>>= 1) n += x & 1; return n; };

const BEAM_Y = 2.7;
const LAMP_Y = 1.55;
const PITCH = 0.46;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, inputs = parameter(env, 'inputs'), subset = parameter(env, 'subset');
  const palette = ctx.tokens.palettes.night, accent = ctx.tokens.accents['subsetparity'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, 'subsetparityLoom', palette, 'night', { sky: { sunDir: [0.45, 0.5, 0.74], moon: true, sunSize: 0.03, sunHdr: 2.4, sunGlow: 0.14, sunGlowPower: 36, stars: { intensity: 2.4 } }, light: { pointCount: 3, pointDistance: 7 }, fogExtent: 90, weather: { preset: 'fireflies', count: 180, seed: ctx.seed, color: 0xbff59a, size: 1.4, intensity: 1.8, bounds: { y: 1.4, halfX: 8, halfY: 1.3, halfZ: 6 } } });
  const { world, rig } = land;
  rig.sun.intensity = 0.7; rig.sun.color.set(0xc4d4ff); rig.hemi.intensity = 0.5;
  rig.setShadowFocus({ x: 0, y: 1.2, z: 0 }, 7);
  const ground = court(land, { name: 'subsetparity', seed: ctx.seed, halfX: 5, halfZ: 4, colorA: 0x3d573a, colorB: 0x6b8a4a, palette, amplitude: 0.9 });
  groves(land, ctx, ['#3f6b3a', '#5d8a46', '#4f7c40', '#6f9a52'], 8, 12, ground.heightAt, 4);
  const m = propMaterials(world), parts = modelParts(world);
  const span = inputs * PITCH, half = span / 2 + 0.6, slot = (j: number): number => (j - (inputs - 1) / 2) * PITCH;
  for (const side of [-1, 1]) {
    parts.box(world.group, m.sandstone, side * half, 0.2, 0, 0.7, 0.4, 0.9, 0.04);
    parts.box(world.group, m.sandstone, side * half, BEAM_Y / 2 + 0.2, 0, 0.42, BEAM_Y - 0.4, 0.42, 0.04);
    parts.box(world.group, m.brass, side * half, BEAM_Y + 0.03, 0, 0.52, 0.06, 0.52, 0.01);
  }
  parts.box(world.group, m.brass, 0, BEAM_Y + 0.1, 0, span + 1.5, 0.12, 0.14, 0.02);
  parts.box(world.group, m.brass, 0, 0.5, 0, span + 1.4, 0.08, 0.1, 0.02);
  for (let j = 0; j < inputs; j++) parts.rod(world.group, m.brass, new THREE.Vector3(slot(j), 0.54, 0), new THREE.Vector3(slot(j), BEAM_Y + 0.04, 0), 0.012);
  parts.build();
  const lamps = lampBank(world, new THREE.SphereGeometry(0.12, 12, 10), inputs, { emissive: 2 });
  world.add(lamps.mesh);
  for (let j = 0; j < inputs; j++) { lamps.place(j, slot(j), LAMP_Y, 0); lamps.tint(j, accent); lamps.glow(j, 0.1); }
  lamps.commit();
  const ringGlass = m.glass(gold, 0.5);
  const rings = lampBank(world, new THREE.TorusGeometry(0.22, 0.03, 8, 28), subset, { emissive: 1.4, metalness: 0.4, roughness: 0.4 });
  world.add(rings.mesh);
  const answers = lampBank(world, new THREE.SphereGeometry(0.14, 12, 10), 2, { emissive: 2 });
  world.add(answers.mesh);
  const answerZ = 2.4;
  [-0.9, 0.9].forEach((x, i) => {
    parts.box(world.group, m.sandstone, x, 0.15, answerZ, 0.4, 0.3, 0.4, 0.03);
    parts.rod(world.group, m.brass, new THREE.Vector3(x, 0.3, answerZ), new THREE.Vector3(x, 0.72, answerZ), 0.025);
    answers.place(i, x, 0.84, answerZ); answers.tint(i, i === 0 ? ctx.tokens.cueCyan : ctx.tokens.cueMagenta); answers.glow(i, 0.1);
  });
  answers.commit();
  parts.build();
  const poleLamps = [-half - 1.4, half + 1.4].map(x => { const l = lamp(world, accent, 1.6, m); l.group.position.set(x, 0, -1.4); world.add(l.group); l.lit(0.45); return l; });
  rig.points.forEach((point, i) => { point.color.set(i === 2 ? gold : accent); point.position.set(i === 2 ? 0 : (i === 0 ? -half * 0.5 : half * 0.5), i === 2 ? 1.4 : LAMP_Y + 0.3, i === 2 ? answerZ : 0.4); point.intensity = i === 2 ? 0 : 2.5; });
  const extra = { land, ground, inputs, subset, slot, lamps, rings, ringGlass, answers, answerLight: rig.points[2]!, poleLamps, accent, gold, answerZ, half };
  return world.seal<WorldCore & typeof extra>(extra);
}

type LoomWorld = ReturnType<typeof buildWorld>;

class LampLoomScene extends StageScene<LoomWorld> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'subsetparity';
  private view!: SceneView;
  private step = -1;
  private verdictAt = -Infinity;
  private answer = -1;
  private correct = false;
  private readonly ledger = { ringed: 0, answer: -1, correct: false, reward: 0 };
  override legend: LegendSpec = {
    title: 'LAMP LOOM',
    rows: [
      { key: 'ringed', label: 'ringed lamps lit', tint: 0xffd166, value: () => `${formatCount(this.ledger.ringed)} \u2192 ${this.ledger.ringed & 1 ? 'odd' : 'even'}` },
      { key: 'answer', label: 'answer', tint: 0xdcf4ff, value: () => (this.ledger.answer < 0 ? '\u2014' : `${this.ledger.answer ? 'odd' : 'even'} \u00b7 ${this.ledger.correct ? 'right' : 'wrong'}`) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): LoomWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const half = this.world!.half;
    this.view = makeSceneView(ctx, { min: [-half - 2, -0.5, -3], max: [half + 2, 3.4, 3.5] }, [0, 1.3, 0.4], { distance: Math.max(9, half * 2.6), azimuth: 0.3, polar: 1.15 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [this.world!.half * 2 + 4, 8],
      debug: (frame, kit) => {
        const world = this.world!, mask = Math.trunc(frame.state.pose[SUBSETPARITY_MASK]!), marks = kit.marks('mask', world.inputs);
        for (let j = 0; j < world.inputs; j++) if ((mask >>> j) & 1) marks.set(world.slot(j), LAMP_Y + 0.5, 0, 0.1, this.ctx!.tokens.celebrationGold);
      },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<LoomWorld>): void {
    const { state } = frame;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.verdictAt = -Infinity; }
    if (state.step !== this.step) {
      this.step = state.step;
      const asked = frame.frameAt(state.step - 1), bits = asked.words[1]!, mask = Math.trunc(state.pose[SUBSETPARITY_MASK]!);
      this.answer = state.step === 0 ? -1 : Math.trunc(state.pose[SUBSETPARITY_ACTION]!) & 1;
      this.correct = state.reward > 0;
      this.verdictAt = time;
      this.ledger.ringed = popcount(bits & mask); this.ledger.answer = this.answer; this.ledger.correct = this.correct; this.ledger.reward = state.reward;
      for (let j = 0; j < world.inputs; j++) world.lamps.glow(j, (bits >>> j) & 1 ? 1.7 : 0.08);
      let ring = 0;
      for (let j = 0; j < world.inputs && ring < world.subset; j++) if ((mask >>> j) & 1) { world.rings.place(ring, world.slot(j), LAMP_Y, 0); world.rings.tint(ring, world.gold); ring++; }
      world.rings.commit();
      for (let i = 0; i < 2; i++) world.answers.glow(i, i === this.answer ? 1.8 : 0.1);
    }
    const verdict = envelope(time - this.verdictAt, 1 / frame.controlHz + 0.4, 1.6);
    for (let r = 0; r < world.subset; r++) world.rings.glow(r, 0.25 + (this.correct ? 2.2 : 0) * verdict);
    world.answerLight.color.set(this.correct ? world.gold : this.ctx!.tokens.failureBlue);
    world.answerLight.intensity = (this.correct ? 5 : 2) * verdict;
    world.answerLight.position.x = this.answer < 0 ? 0 : this.answer === 0 ? -0.9 : 0.9;
    world.poleLamps.forEach((l, i) => l.lit(0.4 + 0.2 * Math.sin(time * 1.2 + i * 1.9)));
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.verdictAt = -Infinity; this.answer = -1; }
}

export default ((() => new LampLoomScene()) satisfies SceneFactory);
