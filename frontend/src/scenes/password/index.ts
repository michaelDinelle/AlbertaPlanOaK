import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { plinth, relief } from '@kit/props';
import { glyphSet, glyphStones, lampRow, memoryCourt, outcomeGlow } from '@kit/props/memory-court';
import { engravedPanel, modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { PASSWORD_ACTION, PASSWORD_PHASE } from '@sim/envs/password';
import { lastEvent } from '@sim/events';
import { philox4x32_10, STREAM_RESET } from '@sim/philox';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const GLYPH = ['cyan cube', 'magenta pyramid', 'gold sphere', 'green octahedron'] as const;

const DOOR_Z = -2.05;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, length = parameter(env, 'length'), base = constant(env, 'PASSWORD_BASE');
  if (base !== env.actions || base > 4 || length < 1) throw new Error('Password geometry does not match its recorded configuration');
  const digits = Uint8Array.from({ length }, (_, p) => philox4x32_10(ctx.seed, STREAM_RESET, p, 0, 0, 0)[0] % base);
  const court = memoryCourt(ctx, scene, {
    name: 'passwordVault', palette: 'night', fogExtent: 90, points: 2, pointDistance: 7, sun: 0.5, hemi: 0.55,
    weather: { preset: 'fireflies', count: 140, seed: ctx.seed, size: 4, bounds: { y: 1.4, halfX: 7, halfY: 1.2, halfZ: 6 } },
    focus: { x: 0, y: 1, z: -0.5, radius: 7 },
    meadow: { halfX: 6, halfZ: 5, ground: 0x2b3a2e, grass: 0x3a5a3a, trees: [0x3f6b48, 0x4d7a4a, 0x5a8a55, 0x35603e], amplitude: 1.4 },
  });
  const { world, materials: m, accent, rig } = court;
  rig.points.forEach((point, i) => { point.color.set(ctx.tokens.celebrationGold); point.intensity = 3; point.position.set(i === 0 ? -2.6 : 2.6, 2.2, -1.2); });
  const wall = relief(world, 8, 3.4, 0.45, m);
  wall.position.set(0, 0, -2.4);
  world.add(wall);
  const parts = modelParts(world), door = new THREE.Group(), ring = m.glass(accent, 0.8);
  const armor = world.own(new THREE.MeshStandardMaterial({ color: 0xe3d6b8, vertexColors: true, roughness: 0.85 }));
  parts.disc(door, m.brass, 0, 0, 0, 1.15, 0.26, 40);
  for (let s = 0; s < 4; s++) parts.add(door, new THREE.BoxGeometry(0.1, 2, 0.06).rotateZ(s * Math.PI / 4).translate(0, 0, 0.15), m.shale);
  parts.add(door, new THREE.TorusGeometry(0.62, 0.05, 8, 40).translate(0, 0, 0.16), ring);
  parts.add(door, engravedPanel(0.5, 0.56, 0.08, 0.04, 0.3).translate(0, 0, 0.2), armor);
  parts.build();
  door.position.set(0, 1.5, DOOR_Z);
  world.add(door);
  const pad = plinth(world, 1.2, 0.7, m);
  pad.position.set(0, 0, 0.4);
  world.add(pad);
  const glyphs = glyphSet(world, ctx.tokens);
  const stones = glyphStones(world, glyphs, base, (i, mesh) => { mesh.position.set((i % 2 - 0.5) * 0.6, 0.95, 0.4 + (Math.floor(i / 2) - 0.5) * 0.6); mesh.scale.setScalar(0.5); });
  const lamps = lampRow(world, length, 0.13, 'passwordProgress');
  for (let p = 0; p < length; p++) { lamps.place(p, (p - (length - 1) / 2) * 0.5, 3.1, -1.95); lamps.paint(p, glyphs.tints[digits[p]!]!, 0.12); }
  lamps.commit();
  const sentinel = world.own(buildSentinel({ name: 'passwordKeeper', tint: accent }));
  sentinel.place(0, 0, 1.8);
  sentinel.face(Math.PI);
  world.add(sentinel.group);
  const doorAt = new THREE.Vector3(0, 1.5, DOOR_Z + 0.3), padAt = new THREE.Vector3(0, 1, 0.4);
  const extra = { court, length, base, digits, glyphs, stones, lamps, door, ring, sentinel, doorAt, padAt };
  return world.seal<WorldCore & typeof extra>(extra);
}

type VaultWorld = ReturnType<typeof buildWorld>;

class PasswordScene extends StageScene<VaultWorld> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 4;
  protected readonly stageName = 'passwordVault';
  private view!: SceneView;
  private step = -1;
  private flashAt = -Infinity;
  private flashGlyph = -1;
  private flashHit = false;
  private outcomeStep: number | undefined;
  private readonly ledger = { phase: 0, pressed: 0, reward: 0, opened: '\u2014' };
  override legend: LegendSpec = {
    title: 'VAULT',
    rows: [
      { key: 'digit', label: 'digit', tint: 0xff9ecf, value: () => `${this.ledger.phase + 1} of ${this.world!.length}` },
      { key: 'secret', label: 'secret glyph', tint: 0xffd166, value: () => GLYPH[this.world!.digits[this.ledger.phase]!]! },
      { key: 'pressed', label: 'pressed', tint: 0xdcf4ff, value: () => GLYPH[this.ledger.pressed]! },
      { key: 'opened', label: 'last opening', tint: 0xffd166, value: () => this.ledger.opened },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): VaultWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-5, -0.5, -3.5], max: [5, 4, 3] }, [0, 1.3, -0.8], { distance: 9, azimuth: 0.3, polar: 1.05 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [10, 8], agentPos: () => this.world!.sentinel.group.position }; }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<VaultWorld>): void {
    const phase = Math.round(frame.state.pose[PASSWORD_PHASE]!), action = Math.trunc(frame.state.pose[PASSWORD_ACTION]!) % world.base;
    if (phase < 0 || phase >= world.length) throw new Error('Password frame is outside its recorded geometry');
    if (frame.scrubbed) this.flashAt = -Infinity;
    if (frame.state.step !== this.step) {
      if (!frame.scrubbed && this.step >= 0) {
        const before = Math.round(frame.poseAt(frame.state.step - 1)[PASSWORD_PHASE]!);
        this.flashAt = time; this.flashGlyph = action; this.flashHit = phase > before || frame.state.done;
        if (!this.flashHit) pool.sigh(0.2, world.padAt, time);
      }
      this.step = frame.state.step;
    }
    const flash = envelope(time - this.flashAt, 0.9, 2.5);
    world.stones.forEach((stone, i) => { stone.material.emissiveIntensity = 0.5 + (i === this.flashGlyph ? (this.flashHit ? 2.6 : -0.42) * flash : 0); stone.mesh.rotation.y = time * 0.4; });
    const step = lastEvent(frame.eventLog, 'done', frame.state.step);
    this.ledger.phase = phase; this.ledger.pressed = action; this.ledger.reward = frame.state.reward; this.ledger.opened = step === undefined ? '\u2014' : `${formatCount(frame.state.step - step)} steps ago`;
    const age = step === undefined ? Infinity : time - step / frame.controlHz;
    const glow = outcomeGlow(age, 0.8);
    for (let p = 0; p < world.length; p++) world.lamps.paint(p, glow > 0.02 ? world.court.accent : world.glyphs.tints[world.digits[p]!]!, (p < phase ? 2.2 : 0.12) + 3 * glow);
    world.lamps.commit(false);
    world.door.rotation.z = age >= 0 ? -(1 - Math.exp(-age * 1.5)) * Math.PI * 0.5 : 0;
    world.ring.emissiveIntensity = 0.8 + 3 * glow;
    world.sentinel.idle(time);
    world.sentinel.glow(this.flashHit ? flash : 0);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    if (!frame.scrubbed && step !== this.outcomeStep && step !== undefined) pool.burst(world.doorAt, world.court.accent, 0.8, time);
    this.outcomeStep = step;
    world.setTime(time);
    world.court.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.flashAt = -Infinity; this.flashGlyph = -1; this.outcomeStep = undefined; }
}

export default ((() => new PasswordScene()) satisfies SceneFactory);
