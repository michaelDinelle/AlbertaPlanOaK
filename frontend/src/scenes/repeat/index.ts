import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { glyphSet, glyphStones, instanced, lampRow, memoryCourt } from '@kit/props/memory-court';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { REPEAT_ACTION, REPEAT_REGISTER } from '@sim/envs/repeat';
import { constant, parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const GLYPH = ['cyan cube', 'magenta pyramid', 'gold sphere', 'green octahedron'] as const;

const SPACING = 0.42;
const RAIL_Y = 1.5;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, lag = parameter(env, 'lag'), symbols = constant(env, 'REPEAT_SYMBOLS');
  if (symbols !== env.actions || symbols > 4 || lag < 1 || lag >= REPEAT_REGISTER || env.words * 16 !== REPEAT_REGISTER) throw new Error('Repeat geometry does not match its recorded configuration');
  const x0 = -(REPEAT_REGISTER - 1) * SPACING / 2, xAt = (age: number) => x0 + age * SPACING;
  const court = memoryCourt(ctx, scene, {
    name: 'repeatEchoHall', palette: 'night', fogExtent: 110, points: 2, pointDistance: 8, sun: 0.45, hemi: 0.5,
    weather: { preset: 'snow', count: 600, seed: ctx.seed, size: 2.4, bounds: { y: 3.5, halfX: 16, halfY: 4, halfZ: 9 } },
    focus: { x: 0, y: 1, z: 0, radius: 13 },
    meadow: { halfX: 13, halfZ: 6, ground: 0xdfe6ee, grass: 0xd0d9e3, trees: [0xffffff, 0xf2f6fc, 0xffffff, 0xeaf0f8], archetypes: ['bare', 'conifer'], amplitude: 1.6 },
  });
  const { world, materials: m, accent, rig } = court;
  rig.points.forEach((point, i) => { point.color.set(accent); point.intensity = 6; point.position.set(i === 0 ? -6 : 6, 2.6, 0.5); });
  const parts = modelParts(world), hall = new THREE.Group();
  parts.box(hall, m.sandstone, 0, -0.16, 0, 24, 0.32, 7, 0.05);
  parts.rod(hall, m.brass, new THREE.Vector3(x0 - 0.6, RAIL_Y, 0), new THREE.Vector3(xAt(REPEAT_REGISTER - 1) + 0.6, RAIL_Y, 0), 0.03);
  parts.box(hall, m.sandstone, xAt(lag), 0.15, 1.6, 2.4, 0.3, 0.7, 0.03);
  parts.build();
  world.add(hall);
  const columns = instanced(world, new THREE.CylinderGeometry(0.2, 0.24, 3.2, 12).translate(0, 1.6, 0), m.sandstone, 24, 'repeatColumns');
  const caps = instanced(world, new THREE.BoxGeometry(0.6, 0.12, 0.6), m.brass, 24, 'repeatCaps');
  for (let c = 0; c < 24; c++) { const x = -9.9 + (c % 12) * 1.8, z = c < 12 ? -2.8 : 2.8; columns.place(c, x, 0, z); columns.paint(c, 0xffffff); caps.place(c, x, 3.26, z); caps.paint(c, 0xffffff); }
  columns.commit();
  caps.commit();
  const lamps = lampRow(world, REPEAT_REGISTER, 0.13, 'repeatRibbon');
  for (let age = 0; age < REPEAT_REGISTER; age++) { lamps.place(age, xAt(age), RAIL_Y, 0); lamps.paint(age, accent, 0.1); }
  lamps.commit();
  const marker = new THREE.Group(), markerGlass = m.glass(accent, 0.8);
  parts.add(marker, new THREE.TorusGeometry(0.28, 0.035, 8, 32), markerGlass);
  parts.add(marker, new THREE.ConeGeometry(0.08, 0.2, 6).rotateX(Math.PI).translate(0, 0.5, 0), m.brass);
  parts.build();
  marker.position.set(xAt(lag), RAIL_Y, 0);
  world.add(marker);
  const glyphs = glyphSet(world, ctx.tokens);
  const stones = glyphStones(world, glyphs, symbols, (i, mesh) => { mesh.position.set(xAt(lag) + (i - 1.5) * 0.5, 0.55, 1.6); mesh.scale.setScalar(0.45); });
  const sentinel = world.own(buildSentinel({ name: 'repeatEcho', tint: accent }));
  sentinel.place(xAt(lag), 0, 2.5);
  sentinel.face(Math.PI);
  world.add(sentinel.group);
  const markerAt = new THREE.Vector3(xAt(lag), RAIL_Y, 0);
  const extra = { court, lag, symbols, glyphs, stones, lamps, markerGlass, markerAt, sentinel };
  return world.seal<WorldCore & typeof extra>(extra);
}

type EchoWorld = ReturnType<typeof buildWorld>;

class RepeatScene extends StageScene<EchoWorld> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'repeatEchoHall';
  private view!: SceneView;
  private step = -1;
  private scrollFrom = -Infinity;
  private flashAt = -Infinity;
  private flashGlyph = -1;
  private flashHit = false;
  private readonly ledger = { now: 0, due: 0, answer: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'ECHO',
    rows: [
      { key: 'now', label: 'glyph now', tint: 0xa8d8ff, value: () => GLYPH[this.ledger.now]! },
      { key: 'due', label: 'due from the lag', tint: 0xffd166, value: () => GLYPH[this.ledger.due]! },
      { key: 'answer', label: 'answer', tint: 0xdcf4ff, value: () => GLYPH[this.ledger.answer]! },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): EchoWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-11, -0.5, -4], max: [11, 4, 4] }, [0, 1.2, 0.3], { distance: 20, azimuth: 0.25, polar: 1.15 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    const x0 = -(REPEAT_REGISTER - 1) * SPACING / 2;
    return {
      floor: [22, 8], agentPos: () => this.world!.sentinel.group.position,
      debug: (_frame, kit) => {
        const world = this.world!;
        kit.marks('due', 2).set(x0 + world.lag * SPACING, RAIL_Y + 0.6, 0, 0.1, world.glyphs.tints[this.ledger.due]!);
        kit.marks('due', 2).set(x0, RAIL_Y + 0.6, 0, 0.08, this.ctx!.tokens.uiCyan);
      },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<EchoWorld>): void {
    const pose = frame.state.pose, action = Math.trunc(pose[REPEAT_ACTION]!) % world.symbols;
    this.ledger.now = Math.round(pose[0]!) & 3; this.ledger.due = Math.round(pose[world.lag]!) & 3; this.ledger.answer = action; this.ledger.reward = frame.state.reward;
    if (frame.scrubbed) { this.flashAt = -Infinity; this.scrollFrom = -Infinity; }
    if (frame.state.step !== this.step) {
      const advanced = !frame.scrubbed && frame.state.step === this.step + 1;
      this.scrollFrom = advanced ? time : -Infinity;
      for (let age = 0; age < REPEAT_REGISTER; age++) world.lamps.paint(age, world.glyphs.tints[Math.round(pose[age]!) & 3]!, age === world.lag ? 2.2 : 0.85 * (1 - 0.65 * age / REPEAT_REGISTER));
      world.lamps.commit(false);
      if (advanced) {
        this.flashAt = time; this.flashGlyph = action; this.flashHit = frame.state.reward > 0;
        if (this.flashHit) pool.burst(world.markerAt, world.court.accent, 0.45, time); else pool.sigh(0.25, world.markerAt, time);
      }
      this.step = frame.state.step;
    }
    const scroll = Math.min(1, (time - this.scrollFrom) * frame.controlHz * 1.6);
    world.lamps.mesh.position.x = -SPACING * (1 - scroll);
    const flash = envelope(time - this.flashAt, 1.1, 2.2);
    world.markerGlass.emissiveIntensity = 0.8 + (this.flashHit ? 3 : -0.6) * flash;
    world.stones.forEach((stone, i) => { stone.material.emissiveIntensity = 0.5 + (i === this.flashGlyph ? (this.flashHit ? 2.4 : -0.42) * flash : 0); stone.mesh.rotation.y = time * 0.4; });
    world.sentinel.idle(time);
    world.sentinel.glow(this.flashHit ? flash : 0.1);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.scrollFrom = -Infinity; this.flashAt = -Infinity; this.flashGlyph = -1; }
}

export default ((() => new RepeatScene()) satisfies SceneFactory);
