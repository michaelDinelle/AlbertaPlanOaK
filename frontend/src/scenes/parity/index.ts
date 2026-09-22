import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { instanced, memoryCourt } from '@kit/props/memory-court';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { PARITY_ACTION, PARITY_REGISTER } from '@sim/envs/parity';
import { parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const SPACING = 0.1;
const THREAD_Y = 1.15;
const THREADS = [0x2f3550, 0xf4e6b8] as const;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, delay = parameter(env, 'delay');
  if (env.actions !== 2 || delay < 1 || delay + 1 >= PARITY_REGISTER || env.words * 32 !== PARITY_REGISTER) throw new Error('Parity geometry does not match its recorded configuration');
  const x0 = -(PARITY_REGISTER - 1) * SPACING / 2, xAt = (age: number) => x0 + age * SPACING, width = PARITY_REGISTER * SPACING + 1;
  const court = memoryCourt(ctx, scene, {
    name: 'parityLoom', palette: 'morning', fogExtent: 120, points: 2, pointDistance: 6,
    weather: { preset: 'pollen', count: 360, seed: ctx.seed, color: 0xfff1c8, size: 1.1, intensity: 0.8, bounds: { y: 1.4, halfX: 8, halfY: 1.2, halfZ: 5 } },
    focus: { x: 0, y: 1, z: 0, radius: 8 },
    meadow: { halfX: 8, halfZ: 6, ground: 0x8fa85e, grass: 0xa6c46a, trees: [0xa8d47a, 0x8fc46e, 0xc6e08a, 0x7fb066], amplitude: 1.4 },
  });
  const { world, materials: m, accent, rig } = court;
  rig.points.forEach((point, i) => { point.color.set(accent); point.intensity = 4; point.position.set(i === 0 ? -4 : 4, 2.6, 1.5); });
  const parts = modelParts(world), loom = new THREE.Group();
  parts.box(loom, m.sandstone, 0, -0.1, 0, width + 2, 0.2, 4, 0.05);
  for (const side of [-1, 1]) parts.box(loom, m.sandstone, side * (width / 2 + 0.3), 1.15, 0, 0.22, 2.3, 0.22, 0.03);
  parts.box(loom, m.brass, 0, 2.25, 0, width + 0.6, 0.16, 0.24, 0.02);
  parts.box(loom, m.shale, 0, 0.12, 0, width + 0.6, 0.16, 0.24, 0.02);
  for (const age of [delay, delay + 1]) parts.box(loom, m.brass, xAt(age), 2.42, 0, 0.05, 0.22, 0.3, 0.01);
  parts.build();
  world.add(loom);
  const threads = instanced(world, new THREE.BoxGeometry(0.035, 1.9, 0.035), world.own(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8 })), PARITY_REGISTER, 'parityThreads', false);
  for (let age = 0; age < PARITY_REGISTER; age++) { threads.place(age, xAt(age), THREAD_Y, 0); threads.paint(age, THREADS[0]); }
  threads.commit();
  const mark = new THREE.Group(), markGlass = m.glass(accent, 0.6);
  parts.add(mark, new THREE.BoxGeometry(0.3, 2, 0.03).translate(xAt(delay) + SPACING / 2, THREAD_Y, -0.06), markGlass);
  parts.build();
  world.add(mark);
  const shuttles = THREADS.map((tint, bit) => {
    const material = world.own(new THREE.MeshStandardMaterial({ color: tint, emissive: tint, emissiveIntensity: 0.3, roughness: 0.5 }));
    const mesh = world.add(new THREE.Mesh(world.own(new THREE.CylinderGeometry(0.22, 0.22, 0.1, 16)), material));
    mesh.position.set(xAt(delay) + (bit - 0.5) * 0.6, 0.5, 1.5);
    mesh.castShadow = true;
    return { mesh, material, tint };
  });
  parts.box(loom, m.sandstone, xAt(delay), 0.22, 1.5, 1.4, 0.44, 0.6, 0.03);
  parts.build();
  const sentinel = world.own(buildSentinel({ name: 'parityWeaver', tint: accent }));
  sentinel.place(xAt(delay) + 1.2, 0, 2.3);
  sentinel.face(Math.atan2(-1.2, -2.3));
  world.add(sentinel.group);
  const markAt = new THREE.Vector3(xAt(delay) + SPACING / 2, THREAD_Y, 0);
  const extra = { court, delay, threads, markGlass, markAt, shuttles, sentinel };
  return world.seal<WorldCore & typeof extra>(extra);
}

type LoomWorld = ReturnType<typeof buildWorld>;

class ParityScene extends StageScene<LoomWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'parityLoom';
  private view!: SceneView;
  private step = -1;
  private scrollFrom = -Infinity;
  private flashAt = -Infinity;
  private flashBit = -1;
  private flashHit = false;
  private readonly ledger = { now: 0, a: 0, b: 0, answer: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'LOOM',
    rows: [
      { key: 'now', label: 'thread now', tint: 0xd6b3ff, value: () => (this.ledger.now ? 'light' : 'dark') },
      { key: 'pair', label: 'threads at the delay', tint: 0xf4e6b8, value: () => `${this.ledger.a ? 'light' : 'dark'} \u00b7 ${this.ledger.b ? 'light' : 'dark'} \u2192 ${this.ledger.a !== this.ledger.b ? 'odd' : 'even'}` },
      { key: 'answer', label: 'answer', tint: 0xdcf4ff, value: () => (this.ledger.answer ? 'odd' : 'even') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): LoomWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-6.5, -0.5, -3], max: [6.5, 3.5, 3.5] }, [0, 1.1, 0.3], { distance: 12, azimuth: 0.25, polar: 1.15 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [14, 8], agentPos: () => this.world!.sentinel.group.position,
      debug: (frame, kit) => {
        const world = this.world!, pose = frame.state.pose, gold = this.ctx!.tokens.celebrationGold, cyan = this.ctx!.tokens.uiCyan, x0 = -(PARITY_REGISTER - 1) * SPACING / 2;
        const a = pose[world.delay]! > 0.5 ? 1 : 0, b = pose[world.delay + 1]! > 0.5 ? 1 : 0, marks = kit.marks('pair', 3);
        marks.set(x0 + world.delay * SPACING, THREAD_Y + 1.25, 0, 0.09, a ? gold : 0x33404f);
        marks.set(x0 + (world.delay + 1) * SPACING, THREAD_Y + 1.25, 0, 0.09, b ? gold : 0x33404f);
        marks.set(x0, THREAD_Y + 1.25, 0, 0.08, pose[0]! > 0.5 ? cyan : 0x33404f);
      },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<LoomWorld>): void {
    const pose = frame.state.pose, action = Math.trunc(pose[PARITY_ACTION]!) & 1;
    this.ledger.now = pose[0]! > 0.5 ? 1 : 0; this.ledger.a = pose[world.delay]! > 0.5 ? 1 : 0; this.ledger.b = pose[world.delay + 1]! > 0.5 ? 1 : 0; this.ledger.answer = action; this.ledger.reward = frame.state.reward;
    if (frame.scrubbed) { this.flashAt = -Infinity; this.scrollFrom = -Infinity; }
    if (frame.state.step !== this.step) {
      const advanced = !frame.scrubbed && frame.state.step === this.step + 1;
      this.scrollFrom = advanced ? time : -Infinity;
      for (let age = 0; age < PARITY_REGISTER; age++) world.threads.paint(age, THREADS[pose[age]! > 0.5 ? 1 : 0], age === world.delay || age === world.delay + 1 ? 1.5 : 1);
      world.threads.commit(false);
      if (advanced) {
        this.flashAt = time; this.flashBit = action; this.flashHit = frame.state.reward > 0;
        if (this.flashHit) pool.burst(world.markAt, world.court.accent, 0.4, time); else pool.sigh(0.25, world.markAt, time);
      }
      this.step = frame.state.step;
    }
    const scroll = Math.min(1, (time - this.scrollFrom) * frame.controlHz * 1.6);
    world.threads.mesh.position.x = -SPACING * (1 - scroll);
    const flash = envelope(time - this.flashAt, 1.1, 2.2);
    world.markGlass.emissiveIntensity = 0.6 + (this.flashHit ? 3 : -0.5) * flash;
    world.shuttles.forEach((shuttle, bit) => { shuttle.material.emissiveIntensity = 0.3 + (bit === this.flashBit ? (this.flashHit ? 2.2 : -0.28) * flash : 0); shuttle.mesh.position.y = 0.5 + (bit === this.flashBit ? 0.25 * flash : 0); });
    world.sentinel.idle(time);
    world.sentinel.glow(this.flashHit ? flash : 0.1);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.scrollFrom = -Infinity; this.flashAt = -Infinity; this.flashBit = -1; }
}

export default ((() => new ParityScene()) satisfies SceneFactory);
