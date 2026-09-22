import * as THREE from 'three';
import { emissiveMaterial, setEmissiveColor, setEmissiveIntensity } from '@kit/materials';
import { hoodooGeometry } from '@kit/props';
import { instanced, memoryCourt } from '@kit/props/memory-court';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import type { WorldCore } from '@kit/world';
import { ALIASED_ACTION, ALIASED_POS } from '@sim/envs/aliased';
import { philox4x32_10, STREAM_RESET } from '@sim/philox';
import { parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const HOP_SECONDS = 0.14;
const STONE_TINTS = [0x7f6f58, 0xe2cfa6] as const;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, cells = parameter(env, 'cells');
  if (cells < 2 || env.actions !== 2) throw new Error('AliasedRing geometry does not match its recorded configuration');
  const radius = Math.max(2.6, cells * 0.62 / (2 * Math.PI));
  const at = (cell: number, out: THREE.Vector3) => out.set(radius * Math.cos(cell / cells * 2 * Math.PI), 0, radius * Math.sin(cell / cells * 2 * Math.PI));
  const court = memoryCourt(ctx, scene, {
    name: 'aliasedRingRoad', palette: 'afternoon', fogExtent: 110 + radius * 2, points: 0, sun: 2.1, hemi: 0.55,
    weather: { preset: 'dust', count: 500, seed: ctx.seed, size: 1.6, bounds: { y: 0.8, halfX: radius + 3, halfY: 0.8, halfZ: radius + 3 } },
    focus: { x: 0, y: 0.5, z: 0, radius: radius + 3 },
    meadow: { halfX: radius + 3, halfZ: radius + 3, ground: 0xc7a46a, grass: 0xb99a55, trees: [0xd98a3a, 0xc46a2a, 0xe0b04a, 0x9a6a3a], amplitude: 2.2 },
  });
  const { world, materials: m } = court;
  const parts = modelParts(world), road = new THREE.Group();
  parts.add(road, new THREE.TorusGeometry(radius, 0.5, 6, Math.max(48, cells * 2)).rotateX(Math.PI / 2).scale(1, 0.1, 1), m.shale);
  parts.build();
  world.add(road);
  const scale = Math.min(0.95, 2 * Math.PI * radius / cells / 0.9);
  const stones = instanced(world, new THREE.CylinderGeometry(0.32, 0.38, 0.14, 10).translate(0, 0.07, 0), world.own(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 })), cells, 'aliasedStones');
  const spot = new THREE.Vector3();
  for (let cell = 0; cell < cells; cell++) {
    at(cell, spot);
    stones.place(cell, spot.x, 0.02, spot.z, scale, 1, scale);
    stones.paint(cell, STONE_TINTS[philox4x32_10(ctx.seed, STREAM_RESET, cell, 0, 0, 0)[0] & 1]!);
  }
  stones.commit();
  const bluff = world.add(new THREE.Mesh(world.own(hoodooGeometry()), m.sandstone));
  bluff.scale.setScalar(radius / 6);
  bluff.position.y = -0.05;
  bluff.castShadow = bluff.receiveShadow = true;
  const wisp = world.own(new Wisp({ color: 0xfff1cf, radius: 0.36, height: 0.8, skirt: false, ribbon: false, silhouette: false, name: 'aliasedWalker' }));
  world.add(wisp.group);
  const beamMaterial = world.own(emissiveMaterial(ctx.tokens.celebrationGold, 1.5, { transparent: true, opacity: 0.7, depthWrite: false }));
  const beam = world.add(new THREE.Mesh(world.own(new THREE.CylinderGeometry(0.1, 0.16, 2.4, 10).translate(0, 1.2, 0)), beamMaterial));
  beam.visible = false;
  const extra = { court, cells, radius, at, stones, wisp, beam, beamMaterial };
  return world.seal<WorldCore & typeof extra>(extra);
}

type RingWorld = ReturnType<typeof buildWorld>;

class AliasedScene extends StageScene<RingWorld> {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'aliasedRingRoad';
  private view!: SceneView;
  private pos = -1;
  private step = -1;
  private hopStart = -Infinity;
  private claimAt = -Infinity;
  private claimHit = false;
  private claimIndex = 0;
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly ledger = { pos: 0, target: 0, claim: false, reward: 0 };
  override legend: LegendSpec = {
    title: 'RING ROAD',
    rows: [
      { key: 'at', label: 'stone', tint: 0xff9ab8, value: () => `${this.ledger.pos + 1} of ${this.world!.cells}` },
      { key: 'target', label: 'secret stone', tint: 0xffd166, value: () => `${this.ledger.target + 1}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.claim ? 'claim this stone' : 'step on') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): RingWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const world = this.world!, r = world.radius + 2;
    this.view = makeSceneView(ctx, { min: [-r, -0.5, -r], max: [r, 3, r] }, [0, 0.3, 0], { distance: Math.max(9, world.radius * 2), azimuth: 0.4, polar: 0.8 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    const at = new THREE.Vector3();
    return {
      floor: [this.world!.radius * 2 + 4, this.world!.radius * 2 + 4], agentPos: () => this.world!.wisp.group.position,
      debug: (_frame, kit) => {
        const world = this.world!;
        world.at(this.ledger.target, at);
        kit.marks('target', 1).set(at.x, 0.9, at.z, 0.16, this.ctx!.tokens.celebrationGold);
        kit.trail('path', 64, this.ctx!.tokens.uiCyan).push(world.wisp.group.position.x, 0.5, world.wisp.group.position.z);
      },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<RingWorld>): void {
    const pos = Math.round(frame.state.pose[ALIASED_POS]!), claim = (Math.trunc(frame.state.pose[ALIASED_ACTION]!) & 1) === 1;
    if (pos < 0 || pos >= world.cells) throw new Error('AliasedRing frame is outside its recorded geometry');
    this.ledger.pos = pos; this.ledger.target = frame.frameAt(frame.state.step).words[1]!; this.ledger.claim = claim; this.ledger.reward = frame.state.reward;
    if (pos !== this.pos) {
      this.from.copy(this.to);
      world.at(pos, this.to);
      this.hopStart = frame.scrubbed || this.pos < 0 ? -Infinity : time;
      this.pos = pos;
    }
    const t = Math.min(1, (time - this.hopStart) / HOP_SECONDS);
    world.wisp.group.position.lerpVectors(this.from, this.to, t);
    world.wisp.group.position.y = Math.sin(Math.PI * t) * 0.16;
    world.wisp.update(frame.dtReal, time);
    if (frame.scrubbed) this.claimAt = -Infinity;
    if (frame.state.step !== this.step) {
      if (!frame.scrubbed && this.step >= 0 && claim) {
        this.claimAt = time; this.claimHit = frame.state.reward > 0; this.claimIndex = pos;
        setEmissiveColor(world.beamMaterial, this.claimHit ? ctx.tokens.celebrationGold : ctx.tokens.failureBlue);
        if (this.claimHit) { pool.burst(this.to, ctx.tokens.celebrationGold, 0.6, time); world.wisp.flare(1); } else pool.sigh(0.3, this.to, time);
      }
      this.step = frame.state.step;
    }
    const age = time - this.claimAt;
    world.beam.visible = age >= 0 && age < 1.4;
    if (world.beam.visible) { world.at(this.claimIndex, world.beam.position); setEmissiveIntensity(world.beamMaterial, (this.claimHit ? 2.2 : 0.8) * Math.exp(-age * 2)); }
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.pos = -1; this.step = -1; this.hopStart = -Infinity; this.claimAt = -Infinity; }
}

export default ((() => new AliasedScene()) satisfies SceneFactory);
