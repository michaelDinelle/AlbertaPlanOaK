import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { brazier } from '@kit/props';
import { glyphSet, glyphStones, instanced, lampRow, memoryCourt } from '@kit/props/memory-court';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { RECITE_ACTION, RECITE_PHASE, RECITE_SEQUENCE } from '@sim/envs/recite';
import { philox4x32_10 } from '@sim/philox';
import { constant, parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const GLYPH = ['cyan cube', 'magenta pyramid', 'gold sphere', 'green octahedron'] as const;

const ARC = 3.5;
const ENV_STREAM = 3;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, length = parameter(env, 'length'), symbols = constant(env, 'RECITE_SYMBOLS');
  if (symbols !== env.actions || symbols > 4 || length < 1) throw new Error('Recite geometry does not match its recorded configuration');
  const radius = Math.max(3.4, length * 0.5);
  const court = memoryCourt(ctx, scene, {
    name: 'reciteAmphitheatre', palette: 'dusk', fogExtent: 110 + radius, points: 2, pointDistance: 6,
    weather: { preset: 'embers', count: 200, seed: ctx.seed, size: 4, bounds: { y: 1.5, halfX: radius + 1, halfY: 1.4, halfZ: radius + 1 } },
    focus: { x: 0, y: 0.6, z: -radius * 0.4, radius: radius + 3 },
    meadow: { halfX: radius + 3, halfZ: radius + 3, ground: 0x9a7a4a, grass: 0xb08a4a, trees: [0xd9822b, 0xc45a2a, 0xe0a63a, 0xa4552b], amplitude: 1.8 },
  });
  const { world, materials: m, accent, rig } = court;
  rig.points.forEach((point, i) => { point.color.set(0xff9a4d); point.intensity = 6; point.position.set((i === 0 ? -1 : 1) * radius * 0.7, 1.4, -radius * 0.5); });
  const seat = (r: number, index: number, n: number) => { const angle = -ARC / 2 + ARC * (index + 0.5) / n; return { x: r * Math.sin(angle), z: -r * Math.cos(angle), angle }; };
  const tierCounts = [0, 1].map(t => Math.round(ARC * (radius + 0.9 + t * 0.8) / 0.62));
  const tiers = instanced(world, new THREE.BoxGeometry(0.6, 1, 0.7), m.sandstone, tierCounts[0]! + tierCounts[1]!, 'reciteTiers');
  let block = 0;
  tierCounts.forEach((n, t) => { for (let i = 0; i < n; i++) { const at = seat(radius + 0.9 + t * 0.8, i, n); tiers.place(block, at.x, 0.18 + t * 0.18, at.z, 1, 0.36 + t * 0.36, 1, -at.angle); tiers.paint(block++, 0xffffff); } });
  tiers.commit();
  const pedestals = instanced(world, new THREE.BoxGeometry(0.36, 0.9, 0.36), m.sandstone, length, 'recitePedestals');
  const lamps = lampRow(world, length, 0.16, 'reciteChorus');
  const seats = Array.from({ length }, (_, i) => { const at = seat(radius, i, length); return new THREE.Vector3(at.x, 1.15, at.z); });
  seats.forEach((at, i) => { pedestals.place(i, at.x, 0.45, at.z); pedestals.paint(i, 0xffffff); lamps.place(i, at.x, at.y, at.z); lamps.paint(i, accent, 0.06); });
  pedestals.commit();
  lamps.commit();
  const parts = modelParts(world), slab = new THREE.Group();
  parts.box(slab, m.sandstone, 0, 0.15, 1.2, 2.8, 0.3, 0.8, 0.03);
  parts.box(slab, m.sandstone, 0, -0.1, -radius * 0.3, radius * 2 + 3, 0.2, radius + 4, 0.05);
  parts.build();
  world.add(slab);
  const glyphs = glyphSet(world, ctx.tokens);
  const stones = glyphStones(world, glyphs, symbols, (i, mesh) => { mesh.position.set((i - 1.5) * 0.6, 0.55, 1.2); mesh.scale.setScalar(0.5); });
  const braziers = [-2.2, 2.2].map(x => { const fire = brazier(world, 0xff9a4d, m); fire.group.position.set(x, 0, 0.6); world.add(fire.group); return fire; });
  const sentinel = world.own(buildSentinel({ name: 'reciteSentinel', tint: accent }));
  sentinel.face(Math.PI);
  world.add(sentinel.group);
  const extra = { court, length, symbols, radius, glyphs, stones, lamps, seats, braziers, sentinel };
  return world.seal<WorldCore & typeof extra>(extra);
}

type ReciteWorld = ReturnType<typeof buildWorld>;

class ReciteScene extends StageScene<ReciteWorld> {
  readonly post: ScenePost = { bloom: 0.46, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'reciteAmphitheatre';
  private view!: SceneView;
  private symbols = new Uint8Array(0);
  private sequence = -1;
  private step = -1;
  private flashAt = -Infinity;
  private flashGlyph = -1;
  private flashHit = false;
  private judged = -1;
  private readonly ledger = { phase: '', expected: '\u2014', answer: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'RECITE',
    rows: [
      { key: 'phase', label: 'phase', tint: 0xb8a4ff, value: () => this.ledger.phase },
      { key: 'expected', label: 'chorus shows', tint: 0xffd166, value: () => this.ledger.expected },
      { key: 'answer', label: 'answer', tint: 0xdcf4ff, value: () => GLYPH[this.ledger.answer]! },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): ReciteWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const world = this.world!;
    this.view = makeSceneView(ctx, { min: [-world.radius - 2, -0.5, -world.radius - 3], max: [world.radius + 2, 3, 3] }, [0, 0.8, -world.radius * 0.35], { distance: Math.max(11, world.radius * 2.6), azimuth: 0.05, polar: 1.0 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [this.world!.radius * 2 + 4, this.world!.radius + 6], agentPos: () => this.world!.sentinel.group.position,
      debug: (frame, kit) => {
        const world = this.world!, phase = Math.round(frame.state.pose[RECITE_PHASE]!), index = phase < world.length ? phase : phase - world.length;
        if (index >= 0 && index < world.length) kit.marks('seat', 1).set(world.seats[index]!.x, world.seats[index]!.y + 0.3, world.seats[index]!.z, 0.1, this.ctx!.tokens.celebrationGold);
      },
    };
  }
  protected start(_ctx: SceneContext, world: ReciteWorld): void { this.symbols = new Uint8Array(world.length); }
  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<ReciteWorld>): void {
    const sequence = Math.round(frame.state.pose[RECITE_SEQUENCE]!), phase = Math.round(frame.state.pose[RECITE_PHASE]!), action = Math.trunc(frame.state.pose[RECITE_ACTION]!) % world.symbols;
    if (phase < 0 || phase >= 2 * world.length || sequence < 0) throw new Error('Recite frame is outside its recorded geometry');
    if (this.sequence !== sequence) {
      for (let i = 0; i < world.length; i++) this.symbols[i] = philox4x32_10(ctx.seed, ENV_STREAM, sequence, i, 0, 0)[0] & 3;
      this.sequence = sequence;
    }
    const watching = phase < world.length, shown = watching ? phase : -1, next = watching ? -1 : phase - world.length;
    const judged = frame.state.done ? world.length - 1 : phase > world.length ? phase - world.length - 1 : -1;
    this.ledger.phase = watching ? `watching ${shown + 1} of ${world.length}` : `reciting ${next + 1} of ${world.length}`;
    this.ledger.expected = watching ? GLYPH[this.symbols[shown]!]! : `remember ${GLYPH[this.symbols[next]!]!}`; this.ledger.answer = action; this.ledger.reward = frame.state.reward;
    if (frame.scrubbed) this.flashAt = -Infinity;
    if (frame.state.step !== this.step) {
      if (!frame.scrubbed && this.step >= 0 && judged >= 0) {
        this.flashAt = time; this.flashGlyph = action; this.flashHit = frame.state.reward > 0; this.judged = judged;
        const at = world.seats[judged]!;
        if (this.flashHit) pool.burst(at, world.glyphs.tints[this.symbols[judged]!]!, 0.4, time); else pool.sigh(0.25, at, time);
      }
      this.step = frame.state.step;
    }
    const flash = envelope(time - this.flashAt, 1.1, 2.2);
    for (let i = 0; i < world.length; i++) {
      const base = watching ? (i === shown ? 2.8 : i < shown ? 0.5 : 0.06) : (i < next ? 0.55 : i === next ? 0.9 + 0.35 * Math.sin(time * 7) : 0.3);
      const boost = i === this.judged ? (this.flashHit ? 2.4 : -0.5) * flash : 0;
      world.lamps.paint(i, world.glyphs.tints[this.symbols[i]!]!, Math.max(0.03, base + boost));
    }
    world.lamps.commit(false);
    world.stones.forEach((stone, i) => { stone.material.emissiveIntensity = 0.5 + (i === this.flashGlyph ? (this.flashHit ? 2.4 : -0.42) * flash : 0); stone.mesh.rotation.y = time * 0.4; });
    world.braziers.forEach((fire, i) => fire.lit(0.8 + 0.2 * Math.sin(time * 3.1 + i * 2)));
    world.sentinel.idle(time);
    world.sentinel.glow(watching ? 0.15 : 0.8);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.sequence = -1; this.step = -1; this.flashAt = -Infinity; this.flashGlyph = -1; this.judged = -1; }
}

export default ((() => new ReciteScene()) satisfies SceneFactory);
