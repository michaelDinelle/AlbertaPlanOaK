import * as THREE from 'three';
import { makeLandscape } from '@kit/landscape';
import { envelope } from '@kit/mix';
import { lamp, plinth, propMaterials } from '@kit/props';
import { court } from '@kit/props/court';
import { lampBank } from '@kit/props/lampbank';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { PERMUTED_ACTION } from '@sim/envs/permuted';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const INNER = 0.95;
const OUTER = 1.7;
const WHEEL_Y = 0.62;

const codeOf = (observations: Float32Array, states: number): number => { let code = -1; for (let i = 0; i < states; i++) if (observations[i]! > 0.5) code = i; return code; };

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, states = constant(env, 'PERMUTED_STATES'), actions = constant(env, 'PERMUTED_ACTIONS'), recodePeriod = parameter(env, 'recodePeriod');
  const palette = ctx.tokens.palettes.night, accent = ctx.tokens.accents['permuted'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, 'permutedWheel', palette, 'night', { sky: { sunDir: [-0.5, 0.55, 0.67], moon: true, sunSize: 0.035, sunHdr: 2.2, sunGlow: 0.12, sunGlowPower: 40, stars: { intensity: 2.6, density: 180 } }, light: { pointCount: 3, pointDistance: 7 }, fogExtent: 90, weather: { preset: 'snow', count: 800, seed: ctx.seed, size: 1.3, intensity: 0.9, bounds: { y: 2.2, halfX: 9, halfY: 2.4, halfZ: 9 } } });
  const { world, rig } = land;
  rig.sun.intensity = 0.8; rig.sun.color.set(0xb9ccff); rig.hemi.intensity = 0.55;
  rig.setShadowFocus({ x: 0, y: 0.6, z: 0 }, 6);
  const ground = court(land, { name: 'permuted', seed: ctx.seed, halfX: 4.5, halfZ: 4.5, colorA: 0x3b4a5c, colorB: 0x6b7c8c, palette, amplitude: 1.2, snow: 1 });
  const m = propMaterials(world), parts = modelParts(world);
  const base = plinth(world, 3.6, 0.5, m);
  world.add(base);
  const wheel = new THREE.Group();
  wheel.position.y = 0.5;
  world.add(wheel);
  parts.add(wheel, new THREE.TorusGeometry(INNER, 0.04, 8, 64).rotateX(Math.PI / 2).translate(0, 0.1, 0), m.brass);
  parts.add(wheel, new THREE.CylinderGeometry(0.35, 0.4, 0.16, 24).translate(0, 0.08, 0), m.brass);
  parts.box(wheel, m.brass, 0, 0.18, 0, 0.5, 0.05, 0.08, 0.01);
  parts.box(wheel, m.brass, 0, 0.18, 0, 0.08, 0.05, 0.5, 0.01);
  const outer = new THREE.Group();
  wheel.add(outer);
  parts.add(outer, new THREE.TorusGeometry(OUTER, 0.05, 8, 80).rotateX(Math.PI / 2).translate(0, 0.1, 0), m.brass);
  parts.add(outer, new THREE.TorusGeometry(OUTER - 0.3, 0.025, 6, 80).rotateX(Math.PI / 2).translate(0, 0.1, 0), m.shale);
  if (recodePeriod > 0) for (let i = 0; i < states; i++) {
    const angle = i / states * Math.PI * 2 + Math.PI / states, x = Math.cos(angle) * (OUTER + 0.22), z = Math.sin(angle) * (OUTER + 0.22);
    parts.rod(outer, m.brass, new THREE.Vector3(Math.cos(angle) * OUTER, 0.1, Math.sin(angle) * OUTER), new THREE.Vector3(x, 0.1, z), 0.03);
    parts.add(outer, new THREE.SphereGeometry(0.07, 10, 8).translate(x, 0.1, z), m.brass);
  }
  parts.build();
  const stateStones = lampBank(world, new THREE.OctahedronGeometry(0.17, 0), states, { emissive: 1.9, roughness: 0.4 });
  wheel.add(stateStones.mesh);
  const codeGlyphs = lampBank(world, new THREE.IcosahedronGeometry(0.16, 0), states, { emissive: 1.9, roughness: 0.4 });
  outer.add(codeGlyphs.mesh);
  for (let i = 0; i < states; i++) {
    const angle = i / states * Math.PI * 2;
    stateStones.place(i, Math.cos(angle) * INNER, WHEEL_Y - 0.5, Math.sin(angle) * INNER, angle);
    stateStones.tint(i, accent); stateStones.glow(i, 0.12);
    codeGlyphs.place(i, Math.cos(angle) * OUTER, WHEEL_Y - 0.5, Math.sin(angle) * OUTER, angle);
    codeGlyphs.tint(i, ctx.tokens.cueCyan); codeGlyphs.glow(i, 0.12);
  }
  stateStones.commit(); codeGlyphs.commit();
  const linkGlass = m.glass(0xffffff, 1.6);
  const link = new THREE.Mesh(world.own(new THREE.BoxGeometry(0.04, 0.02, 1)), linkGlass);
  link.position.y = WHEEL_Y + 0.08;
  world.add(link);
  const answers = lampBank(world, new THREE.SphereGeometry(0.13, 12, 10), actions, { emissive: 2 });
  world.add(answers.mesh);
  const answerZ = 3.1;
  for (let a = 0; a < actions; a++) {
    const x = (a - (actions - 1) / 2) * 0.7;
    parts.box(world.group, m.sandstone, x, 0.15, answerZ, 0.34, 0.3, 0.34, 0.03);
    parts.rod(world.group, m.brass, new THREE.Vector3(x, 0.3, answerZ), new THREE.Vector3(x, 0.75, answerZ), 0.025);
    answers.place(a, x, 0.85, answerZ); answers.tint(a, accent); answers.glow(a, 0.1);
  }
  answers.commit();
  parts.build();
  const lamps = [-3.4, 3.4].map(x => { const l = lamp(world, accent, 1.5, m); l.group.position.set(x, 0, -2.2); world.add(l.group); l.lit(0.5); return l; });
  rig.points.forEach((point, i) => { point.color.set(i === 2 ? gold : accent); point.position.set(i === 2 ? 0 : (i === 0 ? -1.6 : 1.6), i === 2 ? 1.6 : 1.4, i === 2 ? answerZ : 0); point.intensity = i === 2 ? 0 : 3; });
  const extra = { land, ground, states, actions, recodePeriod, wheel, outer, stateStones, codeGlyphs, link, linkGlass, answers, answerLight: rig.points[2]!, lamps, accent, gold, answerZ };
  return world.seal<WorldCore & typeof extra>(extra);
}

type WheelWorld = ReturnType<typeof buildWorld>;

class CipherWheelScene extends StageScene<WheelWorld> {
  readonly post: ScenePost = { bloom: 0.52, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'permuted';
  private view!: SceneView;
  private step = -1;
  private verdictAt = -Infinity;
  private answer = -1;
  private correct = false;
  private spin = 0;
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly ledger = { state: 0, code: 0, answer: -1, correct: false, reward: 0 };
  override legend: LegendSpec = {
    title: 'CIPHER WHEEL',
    rows: [
      { key: 'state', label: 'inner stone', tint: 0xb0c8ff, value: () => `${this.ledger.state + 1} of ${this.world!.states}` },
      { key: 'code', label: 'outer glyph shown', tint: 0x5ee0ff, value: () => `${this.ledger.code + 1}` },
      { key: 'answer', label: 'answer', tint: 0xdcf4ff, value: () => (this.ledger.answer < 0 ? '\u2014' : `${this.ledger.answer + 1} of ${this.world!.actions} \u00b7 ${this.ledger.correct ? 'right' : 'wrong'}`) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): WheelWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-4, -0.5, -3.5], max: [4, 2.5, 4] }, [0, 0.6, 0.6], { distance: 9.5, azimuth: 0.2, polar: 0.95 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [8, 8] }; }
  protected start(_ctx: SceneContext, world: WheelWorld): void {
    if (world.recodePeriod > 0) { this.legend.note = () => `wheel recodes in ${formatCount(world.recodePeriod - this.step % world.recodePeriod)} steps`; this.legend.bar = () => (Math.max(0, this.step) % world.recodePeriod) / world.recodePeriod; }
  }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WheelWorld>): void {
    const { state } = frame;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.verdictAt = -Infinity; }
    if (state.step !== this.step) {
      this.step = state.step;
      const asked = frame.frameAt(state.step - 1), stateIndex = asked.words[0]!, code = codeOf(asked.observations, world.states);
      if (stateIndex >= world.states || code < 0) throw new Error('Permuted frame is outside its recorded state space');
      this.answer = state.step === 0 ? -1 : Math.trunc(state.pose[PERMUTED_ACTION]!) % world.actions;
      this.correct = state.reward > 0;
      this.verdictAt = time;
      this.ledger.state = stateIndex; this.ledger.code = code; this.ledger.answer = this.answer; this.ledger.correct = this.correct; this.ledger.reward = state.reward;
      for (let i = 0; i < world.states; i++) { world.stateStones.glow(i, i === stateIndex ? 1.6 : 0.12); world.codeGlyphs.glow(i, i === code ? 1.6 : 0.12); }
      const a = stateIndex / world.states * Math.PI * 2;
      this.from.set(Math.cos(a) * INNER, 0, Math.sin(a) * INNER);
      world.answers.tint(this.answer < 0 ? 0 : this.answer, this.correct ? world.gold : this.ctx!.tokens.failureBlue);
      for (let i = 0; i < world.actions; i++) { if (i !== this.answer) world.answers.tint(i, world.accent); world.answers.glow(i, i === this.answer ? 1.8 : 0.1); }
    }
    const epoch = world.recodePeriod > 0 ? Math.floor(state.step / world.recodePeriod) : 0;
    const target = epoch * Math.PI * 2 * (1 + 1 / world.states);
    this.spin = frame.scrubbed ? target : this.spin + (target - this.spin) * (1 - Math.exp(-2.2 * frame.dtReal));
    world.outer.rotation.y = -this.spin;
    const code = this.step < 0 ? 0 : codeOf(frame.frameAt(this.step - 1).observations, world.states);
    const c = code / world.states * Math.PI * 2 + this.spin;
    this.to.set(Math.cos(c) * OUTER, 0, Math.sin(c) * OUTER);
    world.link.position.set((this.from.x + this.to.x) / 2, WHEEL_Y + 0.1, (this.from.z + this.to.z) / 2);
    world.link.scale.z = Math.max(0.01, this.from.distanceTo(this.to));
    world.link.rotation.y = Math.atan2(this.to.x - this.from.x, this.to.z - this.from.z);
    const verdict = envelope(time - this.verdictAt, 1 / frame.controlHz + 0.4, 1.6);
    world.linkGlass.emissiveIntensity = 0.8 + 1.4 * verdict;
    world.answerLight.color.set(this.correct ? world.gold : this.ctx!.tokens.failureBlue);
    world.answerLight.intensity = (this.correct ? 6 : 2) * verdict;
    world.answerLight.position.x = this.answer < 0 ? 0 : (this.answer - (world.actions - 1) / 2) * 0.7;
    world.lamps.forEach((l, i) => l.lit(0.45 + 0.2 * Math.sin(time * 1.1 + i * 1.7)));
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.verdictAt = -Infinity; this.answer = -1; this.spin = 0; }
}

export default ((() => new CipherWheelScene()) satisfies SceneFactory);
