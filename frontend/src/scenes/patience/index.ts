import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { emissiveMaterial, oakMaterial, setEmissiveIntensity } from '@kit/materials';
import { plinth } from '@kit/props';
import { instanced, memoryCourt, outcomeGlow } from '@kit/props/memory-court';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { PATIENCE_POS } from '@sim/envs/patience';
import { lastEvent } from '@sim/events';
import { parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const LEAVES = 10;
const STEM = 1.7;
const RING = 2.1;
const BLOSSOM = 0xf2a7c3;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, length = parameter(env, 'length');
  if (length < 1 || env.actions !== 2) throw new Error('Patience geometry does not match its recorded configuration');
  const court = memoryCourt(ctx, scene, {
    name: 'patienceSeedling', palette: 'dawn', fogExtent: 100, points: 1, pointDistance: 5,
    weather: { preset: 'pollen', count: 320, seed: ctx.seed, color: 0xfff1c8, size: 1.1, intensity: 0.8, bounds: { y: 1.4, halfX: 5, halfY: 1.2, halfZ: 5 } },
    focus: { x: 0, y: 0.8, z: 0, radius: 5 },
    meadow: { halfX: 5, halfZ: 5, ground: 0x8fa85e, grass: 0xa2c26a, trees: [0xf2b8cf, 0xaed17a, 0xd8e39a, 0x8fbf74], amplitude: 1.3 },
  });
  const { world, materials: m, accent, rig } = court;
  rig.points.forEach(point => { point.color.set(BLOSSOM); point.intensity = 0; point.position.set(0, 2.8, 0); });
  world.add(plinth(world, 1.1, 0.8, m));
  const parts = modelParts(world), soil = new THREE.Group();
  parts.add(soil, new THREE.CylinderGeometry(0.5, 0.5, 0.06, 24).translate(0, 0.86, 0), m.shale);
  parts.build();
  world.add(soil);
  const plant = new THREE.Group();
  plant.position.y = 0.88;
  world.add(plant);
  const stemMaterial = world.own(oakMaterial({ color: 0x5b8a3c, roughness: 0.9, toonSteps: 3 }));
  const stem = new THREE.Mesh(world.own(new THREE.CylinderGeometry(0.03, 0.06, 1, 8).translate(0, 0.5, 0)), stemMaterial);
  stem.castShadow = true;
  plant.add(stem);
  const leafGeometry = new THREE.CircleGeometry(0.16, 6).scale(1, 1.8, 1).rotateX(-Math.PI / 2 + 0.7).rotateY(-Math.PI / 2).translate(0.18, 0, 0);
  const leaves = instanced(world, leafGeometry, world.own(oakMaterial({ color: 0x7fb352, roughness: 0.95, side: THREE.DoubleSide, toonSteps: 3 })), LEAVES, 'patienceLeaves');
  leaves.mesh.count = 0;
  plant.add(leaves.mesh);
  const blossomMaterial = world.own(emissiveMaterial(BLOSSOM, 1.4));
  const blossom = new THREE.Mesh(world.own(new THREE.IcosahedronGeometry(0.17, 1)), blossomMaterial);
  plant.add(blossom);
  const ticks = instanced(world, new THREE.BoxGeometry(1, 0.06, 0.18), m.sandstone, length, 'patienceTicks');
  const width = Math.min(0.3, 2 * Math.PI * RING / length * 0.72);
  for (let i = 0; i < length; i++) { const angle = i / length * 2 * Math.PI; ticks.place(i, RING * Math.cos(angle), 0.03, RING * Math.sin(angle), width, 1, 1, -angle + Math.PI / 2); ticks.paint(i, 0xffffff); }
  ticks.commit();
  const sentinel = world.own(buildSentinel({ name: 'patienceGardener', tint: accent }));
  sentinel.place(-1.8, 0, 0.7);
  sentinel.face(Math.atan2(1.8, -0.7));
  world.add(sentinel.group);
  const blossomAt = new THREE.Vector3(0, 0.88 + STEM + 0.15, 0);
  const extra = { court, length, accent, plant, stem, leaves, blossom, blossomMaterial, blossomAt, ticks, sentinel };
  return world.seal<WorldCore & typeof extra>(extra);
}

type SeedlingWorld = ReturnType<typeof buildWorld>;

class PatienceScene extends StageScene<SeedlingWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['dawn', 'fog'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'patienceSeedling';
  private view!: SceneView;
  private growth = 0;
  private posSeen = -1;
  private outcomeStep: number | undefined;
  private readonly ledger = { pos: 0, wait: true, reward: 0, harvest: '\u2014' };
  override legend: LegendSpec = {
    title: 'SEEDLING',
    rows: [
      { key: 'growth', label: 'growth', tint: 0xc6e07a, value: () => `${this.ledger.pos} of ${this.world!.length}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.wait ? 'tend and wait' : 'cash out now') },
      { key: 'harvest', label: 'last harvest', tint: 0xf2a7c3, value: () => this.ledger.harvest },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): SeedlingWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-3, -0.5, -3], max: [3, 3.5, 3] }, [0, 1.1, 0], { distance: 7.5, azimuth: 0.5, polar: 1.1 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [6, 6], agentPos: () => this.world!.sentinel.group.position,
      debug: (_frame, kit) => {
        kit.bars('growth', 1).set(1.4, 0.88, 0, 0.05 + 1.7 * this.ledger.pos / this.world!.length, this.ctx!.tokens.uiCyan, 0.14);
      },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<SeedlingWorld>): void {
    const pos = Math.round(frame.state.pose[PATIENCE_POS]!);
    if (pos < 0 || pos >= world.length) throw new Error('Patience frame is outside its recorded geometry');
    const target = pos / world.length;
    const k = frame.scrubbed || target < this.growth - 0.5 ? 1 : 1 - Math.exp(-6 * frame.dtReal);
    this.growth += (target - this.growth) * k;
    const height = 0.06 + this.growth * STEM;
    world.stem.scale.set(0.6 + this.growth * 0.8, height, 0.6 + this.growth * 0.8);
    let shown = 0;
    for (let leaf = 0; leaf < LEAVES; leaf++) {
      const threshold = (leaf + 1) / (LEAVES + 1);
      if (this.growth <= threshold) continue;
      const size = Math.min(1, (this.growth - threshold) * 10), angle = leaf * 2.4;
      world.leaves.place(shown++, Math.cos(angle) * 0.05, height * threshold, Math.sin(angle) * 0.05, size, size, size, angle);
    }
    world.leaves.mesh.count = shown;
    world.leaves.commit(false);
    const step = lastEvent(frame.eventLog, 'done', frame.state.step);
    const outcome = step === undefined ? null : frame.frameAt(step);
    const harvest = outcome !== null && outcome.reward >= 1;
    this.ledger.pos = pos; this.ledger.wait = (Math.trunc(frame.frameAt(frame.state.step).actions[0]!) & 1) === 1; this.ledger.reward = frame.state.reward; this.ledger.harvest = outcome === null ? '\u2014' : harvest ? `bloomed for ${formatSigned(outcome.reward)}` : `cashed out for ${formatSigned(outcome.reward)}`;
    const age = step === undefined ? Infinity : time - step / frame.controlHz;
    const glow = outcomeGlow(age, 0.9);
    const bloom = Math.max(0.001, Math.min(1, (this.growth - 0.75) / 0.25) * (1 + 0.15 * Math.sin(time * 3)) + (harvest ? glow * 1.5 : 0));
    world.blossom.scale.setScalar(bloom);
    world.blossom.position.y = height + 0.12;
    setEmissiveIntensity(world.blossomMaterial, 1.4 + (harvest ? 3 * glow : 0));
    const bloomLight = world.court.rig.points[0];
    if (bloomLight) bloomLight.intensity = harvest ? 8 * glow : 0;
    if (this.posSeen !== pos) {
      for (let i = 0; i < world.length; i++) world.ticks.paint(i, i < pos ? world.accent : 0xffffff, i < pos ? 1.4 : 1);
      world.ticks.commit(false);
      this.posSeen = pos;
    }
    world.sentinel.idle(time);
    world.sentinel.glow(harvest ? glow : outcome !== null ? -0.15 * glow : 0);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    if (!frame.scrubbed && step !== this.outcomeStep && outcome) {
      if (harvest) pool.burst(world.blossomAt, BLOSSOM, 0.7, time); else pool.sigh(0.35, world.blossomAt, time);
    }
    this.outcomeStep = step;
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.growth = 0; this.posSeen = -1; this.outcomeStep = undefined; }
}

export default ((() => new PatienceScene()) satisfies SceneFactory);
