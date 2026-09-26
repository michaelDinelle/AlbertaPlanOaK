import * as THREE from 'three/webgpu';
import type { DebugKit, GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, trackPose, type SceneView } from '@kit/view';
import { CHEETAH_ROOT_X } from '@sim/envs/planar';
import { constant, parameter } from '../../data/envs';
import { formatPercent, formatSigned, formatStep } from '../../format';
import { buildCheetah, type CheetahHandle } from './actors';
import { bodyPoint, capsuleEnd, cheetahFrames, cheetahModel, type CheetahModel } from './body';
import { buildWorld, prairieHeight, type WorldHandle } from './world';

const TRACK_Y = 0.52;
const ARC = 10;
const SHIFT_SECONDS = 2.4;

class AlbertaPrairieScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['dawn', 'fog'], blend: [[['dusk', 'skyTop'], 0.25]] } };
  controlHz = 1;
  protected override poolOptions() { return { burstSize: 90, sighSize: 46, sighRadius: 1.3, sighHeight: 1.5 }; }
  protected readonly stageName = 'albertaPrairieSunrise';
  private view: SceneView | null = null;
  private cheetah: CheetahHandle | null = null;
  private model!: CheetahModel;
  private period = 0;
  private grip = 0;
  private gripRange = '';
  private shiftAt = -Infinity;
  private readonly scratch = new Float32Array(9);
  private readonly overlayFrames = new Float32Array(21);
  private readonly point = new THREE.Vector3();
  private readonly other = new THREE.Vector3();
  private readonly ledger = { speed: 0, distance: 0, effort: 0, reward: 0, step: 0 };
  override legend: LegendSpec = { title: 'PRAIRIE SUNRISE', rows: [] };

  protected buildWorld(ctx: SceneContext): Promise<WorldHandle> {
    const env = ctx.environment;
    this.controlHz = 1 / (constant(env, 'PLANAR_SUBSTEPS') * constant(env, 'PLANAR_TIMESTEP'));
    this.model = cheetahModel(env);
    this.period = parameter(env, 'frictionPeriod');
    this.grip = constant(env, 'PLANAR_FRICTION');
    this.gripRange = `${+constant(env, 'PLANAR_FRICTION_LOW').toPrecision(3)}–${+constant(env, 'PLANAR_FRICTION_HIGH').toPrecision(3)}`;
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext): void {
    this.cheetah = buildCheetah(this.model, this.accentOf(ctx));
    this.scene.add(this.cheetah.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-2.6, 0, -1.7], max: [2.6, 1.6, 1.7] }, [0, TRACK_Y, 0], { distance: 4.4, azimuth: 0.265, polar: 1.44 }, 5000);
    return { ...this.view.options, near: 0.25, floor: (x, z) => prairieHeight(x, z) + 1.2, limits: { ...this.view.options.limits, maxPolar: 1.48 } };
  }

  protected overlayOptions(ctx: SceneContext): GroundTruthOptions {
    return { debug: (frame, kit) => this.drawTruth(frame, kit, ctx) };
  }

  private drawTruth(frame: FrameUpdate, kit: DebugKit, ctx: SceneContext): void {
    const model = this.model, frames = cheetahFrames(model, frame.state.pose, this.overlayFrames), r = model.radius;
    const cyan = ctx.tokens.uiCyan, gold = ctx.tokens.celebrationGold, fatal = 0xff6b57, a = this.point, b = this.other;
    const contacts = kit.marks('contacts', model.capsules.length * 2);
    model.capsules.forEach((capsule, i) => {
      bodyPoint(frames, capsule.body, ...capsuleEnd(capsule, 0), a);
      bodyPoint(frames, capsule.body, ...capsuleEnd(capsule, 1), b);
      const ux = (b.x - a.x) / (2 * capsule.half), uy = (b.y - a.y) / (2 * capsule.half), outline = kit.curve(`capsule${i}`, 2 * ARC + 3, capsule.body === 0 ? fatal : cyan);
      for (const [cx, cy, from] of [[b.x, b.y, Math.atan2(ux, -uy)], [a.x, a.y, Math.atan2(-ux, uy)]] as const) {
        for (let k = 0; k <= ARC; k++) outline.point(cx + r * Math.cos(from - Math.PI * k / ARC), cy + r * Math.sin(from - Math.PI * k / ARC), 0);
      }
      outline.point(b.x + r * Math.cos(Math.atan2(ux, -uy)), b.y + r * Math.sin(Math.atan2(ux, -uy)), 0);
      for (const end of [a, b]) if (end.y - r <= 0) contacts.set(end.x, end.y - r, 0, 0.035, capsule.body === 0 ? fatal : gold);
    });
    const x = frames[0]!, ground = kit.curve('ground', 2, gold);
    ground.point(x - 6, 0, 0);
    ground.point(x + 6, 0, 0);
    const joints = kit.marks('joints', model.parent.length);
    for (let k = 0; k < model.parent.length; k++) joints.set(frames[k * 3]!, frames[k * 3 + 1]!, 0, 0.022, cyan);
    const actions = frame.frameAt(Math.floor(frame.position) + 1).actions, bars = kit.bars('torque', actions.length);
    for (let j = 0; j < actions.length; j++) bars.set(x - 0.3 + j * 0.12, 1.75, 0, actions[j]! * 0.22, actions[j]! >= 0 ? gold : cyan, 0.07);
    const trail = kit.trail('torso', 240, cyan);
    if (frame.scrubbed || frame.state.reset) trail.clear();
    trail.push(x, frames[1]!, 0);
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    this.cheetah?.setPose(this.scratch);
    world.setShadowTarget(this.cheetah?.frames[0] ?? 0);
    world.setTime(0);
    this.legend.rows = [
      { key: 'speed', label: 'speed', tint: 0x98e1cd, value: () => `${formatSigned(this.ledger.speed)} m/s` },
      { key: 'distance', label: 'distance', tint: 0xf1dfa9, value: () => `${(Math.round(this.ledger.distance * 10) / 10 || 0).toFixed(1)} m` },
      { key: 'effort', label: 'torque', tint: this.accentOf(ctx), value: () => formatPercent(this.ledger.effort, 0) },
      { key: 'reward', label: 'reward', tint: ctx.tokens.celebrationGold, value: () => formatSigned(this.ledger.reward) },
      ...(this.period > 0 ? [{ key: 'grip', label: 'ground grip', tint: 0xdcf4ff, value: () => this.gripLabel() }] : []),
    ];
    if (this.period > 0) {
      this.legend.note = () => `new ground in ${formatStep(this.period - this.ledger.step % this.period)} steps`;
      this.legend.bar = () => (this.ledger.step % this.period) / this.period;
    }
  }

  private gripLabel(): string {
    const regime = Math.floor(this.ledger.step / this.period);
    return regime === 0 ? `μ ${this.grip.toFixed(2)}` : `draw #${regime} · μ ${this.gripRange}`;
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const cheetah = this.cheetah;
    if (!cheetah) return;
    const state = frame.state, index = Math.floor(frame.position);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.shiftAt = -Infinity; }
    cheetah.setPose(state.pose);
    const actions = frame.frameAt(index + 1).actions;
    cheetah.setEffort(actions);
    const torso = cheetah.frames;
    world.setShadowTarget(torso[0]!);
    for (const event of frame.events) if (event.type === 'done') {
      this.point.set(torso[0]!, 0.1, 0);
      pool.burst(this.point, ctx.tokens.failureBlue, 0.4, time);
      pool.sigh(1, this.point, time);
    }
    if (this.period > 0 && frame.prev && !frame.scrubbed && Math.floor(state.step / this.period) !== Math.floor(frame.prev.step / this.period)) {
      this.shiftAt = time;
      this.point.set(torso[0]!, 0.05, 0);
      pool.burst(this.point, 0xdcf4ff, 0.3, time);
    }
    world.setGroundShift(envelope(time - this.shiftAt, SHIFT_SECONDS, 1.4));
    const x = state.pose[CHEETAH_ROOT_X]!, before = frame.frameAt(state.step).reset ? x : frame.poseAt(state.step - 1)[CHEETAH_ROOT_X]!;
    const applied = frame.frameAt(state.step).actions;
    let effort = 0;
    for (let j = 0; j < applied.length; j++) effort += Math.min(1, Math.abs(applied[j]!)) / applied.length;
    this.ledger.speed = (frame.poseAt(state.step)[CHEETAH_ROOT_X]! - before) * this.controlHz;
    this.ledger.distance = x;
    this.ledger.effort = effort;
    this.ledger.reward = state.reward;
    this.ledger.step = state.step;
    world.setTime(time);
    if (this.view) world.setView(this.view.update(rig, frame.dtReal, [trackPose(frame, CHEETAH_ROOT_X), TRACK_Y, 0]), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }

  protected drop(): void {
    this.cheetah?.dispose();
    this.cheetah = null;
  }
}

export default ((() => new AlbertaPrairieScene()) satisfies SceneFactory);
