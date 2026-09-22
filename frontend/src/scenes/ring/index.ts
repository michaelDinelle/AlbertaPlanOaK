import * as THREE from 'three';
import { makeLandscape } from '@kit/landscape';
import { makeMeadow } from '@kit/meadow';
import { plinth, propMaterials } from '@kit/props';
import { lampBank } from '@kit/props/lampbank';
import { engravedPanel, modelParts, robotMaterials } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import type { WorldCore } from '@kit/world';
import { RING_ACTION, RING_COLOUR, RING_POSITION } from '@sim/envs/ring';
import { constant, parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const STONE_PITCH = 0.64;
const HOP_LIFT = 0.14;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, cells = constant(env, 'RING_CELLS'), arcLength = parameter(env, 'arcLength');
  const palette = ctx.tokens.palettes.noon, accent = ctx.tokens.accents['ring'] ?? ctx.tokens.celebrationGold, blossom = ctx.tokens.accents['acrobot'] ?? ctx.tokens.cueMagenta;
  const radius = cells * STONE_PITCH / (Math.PI * 2);
  const land = makeLandscape(scene, 'ringRoad', palette, 'noon', { sky: { sunDir: [0.4, 0.78, 0.48], sunSize: 0.045, sunHdr: 6, sunGlow: 0.3, sunGlowPower: 30 }, light: { pointCount: 1, pointDistance: 4 }, fogExtent: 150 });
  const { world, rig } = land;
  rig.sun.intensity = 2.8; rig.hemi.intensity = 0.75;
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0 }, radius + 2);
  const meadow = makeMeadow(world, ctx, { name: 'ringRoad', halfX: radius + 1.6, halfZ: radius + 1.6, ground: 0x86a55a, grass: 0xb9d96c, trees: [0xf2c4d6, 0xa9d954, 0xf6dce6, 0xc8e39a], amplitude: 1.6 });
  const m = propMaterials(world), parts = modelParts(world);
  parts.add(world.group, new THREE.TorusGeometry(radius + 0.5, 0.05, 8, 96).rotateX(Math.PI / 2).translate(0, 0.03, 0), m.brass);
  parts.add(world.group, new THREE.TorusGeometry(radius - 0.5, 0.05, 8, 96).rotateX(Math.PI / 2).translate(0, 0.03, 0), m.brass);
  parts.add(world.group, new THREE.RingGeometry(radius - 0.5, radius + 0.5, 96, 1).rotateX(-Math.PI / 2).translate(0, 0.012, 0), m.shale);
  parts.build();
  const stones = lampBank(world, new THREE.CylinderGeometry(0.24, 0.27, 0.1, 10).translate(0, 0.05, 0), cells, { emissive: 1.2, roughness: 0.7 });
  world.add(stones.mesh);
  const colours = [new THREE.Color(accent).multiplyScalar(0.6), new THREE.Color(blossom).multiplyScalar(0.75)] as const;
  const stoneAt = (i: number, out: THREE.Vector3): THREE.Vector3 => { const angle = i / cells * Math.PI * 2; return out.set(Math.sin(angle) * radius, 0, -Math.cos(angle) * radius); };
  const scratch = new THREE.Vector3();
  for (let i = 0; i < cells; i++) { stoneAt(i, scratch); stones.place(i, scratch.x, 0.02, scratch.z, i / cells * Math.PI * 2); stones.tint(i, colours[Math.floor(i / arcLength) & 1]!); stones.glow(i, 0.1); }
  stones.commit();
  const hub = plinth(world, 1.1, 0.7, m);
  world.add(hub);
  const armor = world.own(robotMaterials().armor);
  const emblem = new THREE.Mesh(world.own(engravedPanel(0.8, 0.8, 0.06, 0.06, 0.5).rotateX(-Math.PI / 2)), armor);
  emblem.position.y = 0.79;
  emblem.castShadow = true;
  world.add(emblem);
  const wisp = world.own(new Wisp({ color: 0xfff2d0, radius: 0.3, height: 0.7, skirt: true, ribbon: false, hop: { duration: 0.1, height: 0.1 }, name: 'ringWalker' }));
  world.add(wisp.group);
  const guessGlass = m.glass(accent, 1.8);
  const guess = new THREE.Mesh(world.own(new THREE.SphereGeometry(0.07, 10, 8)), guessGlass);
  wisp.carry(guess);
  rig.points[0]!.color.set(accent);
  const extra = { land, meadow, cells, arcLength, radius, stones, colours, stoneAt, wisp, guessGlass, walkerLight: rig.points[0]! };
  return world.seal<WorldCore & typeof extra>(extra);
}

type RoadWorld = ReturnType<typeof buildWorld>;

class RingRoadScene extends StageScene<RoadWorld> {
  readonly post: ScenePost = { bloom: 0.34, fog: { from: ['noon', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'ring';
  private view!: SceneView;
  private position = -1;
  private hopAt = -Infinity;
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();
  private readonly ledger = { position: 0, colour: 0, guess: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'STONE RING',
    rows: [
      { key: 'at', label: 'stone', tint: 0x9ad8c0, value: () => `${this.ledger.position + 1} of ${this.world!.cells} \u00b7 arcs of ${this.world!.arcLength}` },
      { key: 'colour', label: 'next stone is', tint: 0xf2a7c3, value: () => (this.ledger.colour ? 'pink' : 'green') },
      { key: 'guess', label: 'guessed', tint: 0xdcf4ff, value: () => (this.ledger.guess ? 'pink' : 'green') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): RoadWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const r = this.world!.radius + 1.5;
    this.view = makeSceneView(ctx, { min: [-r, -0.5, -r], max: [r, 2.5, r] }, [0, 0.5, 0.4], { distance: r * 2.15, azimuth: 0.2, polar: 0.82 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [this.world!.radius * 2 + 4, this.world!.radius * 2 + 4], agentPos: () => this.world!.wisp.group.position }; }
  protected start(_ctx: SceneContext, world: RoadWorld): void { world.stoneAt(0, this.to); this.from.copy(this.to); }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<RoadWorld>): void {
    const { state } = frame, position = Math.trunc(state.pose[RING_POSITION]!), colour = Math.trunc(state.pose[RING_COLOUR]!) & 1, guess = Math.trunc(state.pose[RING_ACTION]!) & 1;
    if (position < 0 || position >= world.cells) throw new Error('Ring position is outside its recorded road');
    this.ledger.position = position; this.ledger.colour = colour; this.ledger.guess = guess; this.ledger.reward = state.reward;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    if (position !== this.position) {
      if (this.position >= 0) world.stones.glow(this.position, 0.1);
      world.stones.glow(position, 1.6);
      this.from.copy(frame.scrubbed || this.position < 0 ? world.stoneAt(position, this.to) : this.to);
      world.stoneAt(position, this.to);
      this.hopAt = frame.scrubbed ? -Infinity : time;
      this.position = position;
      world.guessGlass.color.set(world.colours[guess]!);
      world.guessGlass.emissive.copy(world.guessGlass.color);
      if (state.step > 0 && !frame.scrubbed) {
        if (state.reward > 0) { world.wisp.flare(0.8); world.wisp.undim(); }
        else world.wisp.dim(0.45);
      }
      world.wisp.face(Math.atan2(this.to.x - this.from.x, this.to.z - this.from.z));
    }
    const t = Math.min(1, (time - this.hopAt) * frame.controlHz), eased = t * t * (3 - 2 * t);
    world.wisp.group.position.lerpVectors(this.from, this.to, eased);
    world.wisp.group.position.y = HOP_LIFT * 4 * t * (1 - t);
    world.wisp.update(frame.dtReal, time);
    world.walkerLight.position.set(world.wisp.group.position.x, 0.9, world.wisp.group.position.z);
    world.walkerLight.intensity = state.reward > 0 ? 2.2 : 0.6;
    world.walkerLight.color.set(colour === guess ? world.colours[colour]! : this.ctx!.tokens.failureBlue);
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera, world.meadow);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.position = -1; this.hopAt = -Infinity; }
}

export default ((() => new RingRoadScene()) satisfies SceneFactory);
