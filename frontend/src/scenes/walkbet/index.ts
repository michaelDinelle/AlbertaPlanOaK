import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { makeLandscape } from '@kit/landscape';
import { oakMaterial, oakUniforms } from '@kit/materials';
import { envelope } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { banner, gate, hoodooGeometry, lamp, propMaterials } from '@kit/props';
import { court } from '@kit/props/court';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { WALKBET_ACTION, WALKBET_POSITION } from '@sim/envs/walkbet';
import { constant, parameter } from '../../data/envs';
import { formatSigned } from '../../format';

const STEP_X = 0.7;
const DECK_Y = 0.36;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, cells = constant(env, 'WALKBET_CELLS'), period = constant(env, 'WALKBET_PERIOD'), drift = parameter(env, 'drift') !== 0;
  const palette = ctx.tokens.palettes.afternoon, accent = ctx.tokens.accents['walkbet'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const half = cells / 2, xOf = (position: number): number => (position - half) * STEP_X, edge = half * STEP_X;
  const land = makeLandscape(scene, 'walkbetCanyon', palette, 'afternoon', { sky: { sunDir: [0.3, 0.42, 0.85], sunSize: 0.055, sunHdr: 6, sunGlow: 0.45, sunGlowPower: 16, topSpread: 0.5, bottomSpread: 0.35 }, light: { pointCount: 3, pointDistance: 6 }, fogExtent: 120 });
  const { world, rig, sky } = land;
  sky.setColors(0x5b9ad0, 0xe9c99a, 0xd9a877);
  rig.sun.intensity = 2.5; rig.sun.color.set(0xffe2b8); rig.hemi.intensity = 0.65;
  rig.setShadowFocus({ x: 0, y: 0.6, z: 0 }, edge + 2);
  const ground = court(land, { name: 'walkbet', seed: ctx.seed, halfX: edge + 2.5, halfZ: 3, colorA: 0xb0834a, colorB: 0xe0c08a, palette, amplitude: 1.8, rise: 0.4 });
  const dust = world.size(world.time(world.own(makeAmbientParticles({ preset: 'dust', count: 600, seed: ctx.seed, size: 0.9, intensity: 1, bounds: { y: 1.2, halfX: edge + 3, halfY: 1.4, halfZ: 4 }, name: 'walkbetDust' }))));
  world.add(dust.points);
  const rock = world.own(oakMaterial({ vertexColors: true, flatShading: true, roughness: 0.95, rimColor: palette.sun, rimStrength: 0.3, toonSteps: 4 }));
  const bluff = world.own(hoodooGeometry());
  for (const [x, z, yaw, scale] of [[-edge - 5, -1.5, 0.25, 1.6], [edge + 5, -1.5, -0.25, 1.6], [-edge - 2, -9, 0.9, 2.2], [edge + 3, -10, -1.1, 2.4], [0, -13, 0.3, 2.6]] as const) {
    const mesh = new THREE.Mesh(bluff, rock);
    mesh.position.set(x, ground.heightAt(x, z) - 0.2, z); mesh.rotation.y = yaw; mesh.scale.setScalar(scale);
    mesh.castShadow = mesh.receiveShadow = true;
    world.add(mesh);
  }
  const m = propMaterials(world), parts = modelParts(world);
  parts.box(world.group, m.sandstone, 0, DECK_Y / 2 - 0.05, 0, edge * 2 + 0.6, DECK_Y + 0.1, 1.5, 0.04);
  for (let i = 1; i < cells; i++) parts.box(world.group, i % 2 ? m.sandstone : m.shale, xOf(i), DECK_Y + 0.02, 0, STEP_X - 0.06, 0.04, 1.2, 0.01);
  for (const z of [-0.72, 0.72]) parts.box(world.group, m.brass, 0, DECK_Y + 0.05, z, edge * 2 + 0.4, 0.05, 0.06, 0.01);
  parts.build();
  const gates = [-1, 1].map(side => { const g = gate(world, 1.1, 1.7, accent, m); g.group.position.set(side * (edge + 0.2), DECK_Y, 0); g.group.rotation.y = side * Math.PI / 2; world.add(g.group); g.lit(0); return g; });
  const cloths = [-1, 1].map(side => { const b = banner(world, accent, 2.8, m); b.position.set(side * (edge + 1.3), DECK_Y, -1.1); world.add(b); return oakUniforms((b.children[1] as THREE.Mesh).material as THREE.Material); });
  const bets = [-1, 1].map(side => { const l = lamp(world, gold, 1.2, m); l.group.position.set(side * (edge + 1.3), DECK_Y, 1.1); world.add(l.group); return l; });
  const sentinel = world.own(buildSentinel({ name: 'walkbetWalker', tint: accent }));
  sentinel.place(xOf(half), DECK_Y, 0);
  world.add(sentinel.group);
  rig.points.forEach((point, i) => { point.color.set(i === 2 ? accent : gold); point.intensity = 0; point.position.set(i === 2 ? 0 : (i === 0 ? -1 : 1) * (edge + 1.3), i === 2 ? DECK_Y + 3 : DECK_Y + 1.5, i === 2 ? 0.6 : 1.1); });
  const extra = { land, ground, cells, period, drift, half, xOf, edge, gates, cloths, bets, sentinel, dust, betLights: [rig.points[0]!, rig.points[1]!] as const, walkerLight: rig.points[2]!, accent, gold, sunBase: new THREE.Vector3(0.3, 0.42, 0.85) };
  return world.seal<WorldCore & typeof extra>(extra);
}

type CanyonWorld = ReturnType<typeof buildWorld>;

class CanyonWalkwayScene extends StageScene<CanyonWorld> {
  readonly post: ScenePost = { bloom: 0.38, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'walkbet';
  private view!: SceneView;
  private step = -1;
  private position = -1;
  private moveAt = -Infinity;
  private from = 0;
  private to = 0;
  private bet = -1;
  private correct = false;
  private verdictAt = -Infinity;
  private absorbedAt = -Infinity;
  private absorbedGate = 0;
  private readonly sun = new THREE.Vector3();
  private readonly ledger = { position: 0, bet: -1, wind: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'WALKWAY',
    rows: [
      { key: 'at', label: 'plank', tint: 0xffc7a0, value: () => `${this.ledger.position} of ${this.world!.cells}` },
      { key: 'bet', label: 'bet on', tint: 0xdcf4ff, value: () => (this.ledger.bet < 0 ? '\u2014' : this.ledger.bet === 0 ? 'the left gate' : 'the right gate') },
      { key: 'wind', label: 'wind', tint: 0xe0c08a, value: () => (!this.world!.drift ? 'still' : this.ledger.wind > 0.2 ? 'blowing right \u2192' : this.ledger.wind < -0.2 ? '\u2190 blowing left' : 'lull') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): CanyonWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const edge = this.world!.edge;
    this.view = makeSceneView(ctx, { min: [-edge - 3, -0.5, -4], max: [edge + 3, 3.5, 3] }, [0, 1, -0.2], { distance: edge * 1.9, azimuth: 0.12, polar: 1.12 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [this.world!.edge * 2 + 6, 8], agentPos: () => this.world!.sentinel.group.position }; }
  protected start(_ctx: SceneContext, world: CanyonWorld): void { this.from = this.to = world.xOf(world.half); }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<CanyonWorld>): void {
    const { state } = frame, position = Math.trunc(state.pose[WALKBET_POSITION]!);
    if (position <= 0 || position >= world.cells) throw new Error('WalkBet position is outside its walkway');
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.verdictAt = -Infinity; this.absorbedAt = -Infinity; this.moveAt = -Infinity; }
    let teleported = false;
    for (const event of frame.events) if (event.type === 'teleport') teleported = true;
    if (state.step !== this.step) {
      this.step = state.step;
      this.bet = state.step === 0 ? -1 : Math.trunc(state.pose[WALKBET_ACTION]!) & 1;
      this.correct = state.reward > 0;
      this.verdictAt = time;
      if (teleported) {
        const before = frame.frameAt(state.step - 1).words[0]!;
        this.absorbedGate = before < world.half ? 0 : 1;
        this.absorbedAt = time;
        pool.ripple(world.sentinel.group.position, world.accent, 0.5, time);
        this.from = this.to = world.xOf(position);
        this.moveAt = -Infinity;
      } else if (position !== this.position) {
        this.from = frame.scrubbed || this.position < 0 ? world.xOf(position) : this.to;
        this.to = world.xOf(position);
        this.moveAt = frame.scrubbed ? -Infinity : time;
      }
      this.position = position;
    }
    const t = Math.min(1, (time - this.moveAt) * frame.controlHz), eased = t * t * (3 - 2 * t), x = this.from + (this.to - this.from) * eased;
    world.sentinel.place(x, DECK_Y, 0);
    world.sentinel.face(this.to >= this.from ? Math.PI / 2 : -Math.PI / 2);
    if (t < 1) world.sentinel.stride(1.4, frame.dtReal); else world.sentinel.idle(time);
    const phase = world.drift ? Math.PI * 2 * (state.step % world.period) / world.period : 0, wind = world.drift ? Math.sin(phase) : 0;
    this.ledger.position = position; this.ledger.bet = this.bet; this.ledger.wind = wind; this.ledger.reward = state.reward;
    world.cloths.forEach((cloth, side) => cloth?.setWind(0.05 + 0.1 * Math.abs(wind) + (side === this.bet ? 0.06 : 0), 1.4 + Math.abs(wind)));
    (world.dust.material.uniforms.uSwirl as { value: number }).value = 0.18 + 0.9 * Math.abs(wind);
    world.dust.setIntensity(1.2 + 0.8 * Math.abs(wind));
    if (world.drift) {
      this.sun.copy(world.sunBase).applyAxisAngle(THREE.Object3D.DEFAULT_UP, 0.7 * Math.sin(phase));
      world.land.sky.setSunDirection(this.sun.x, this.sun.y, this.sun.z);
      world.land.rig.setSunDirection(this.sun.x, this.sun.y, this.sun.z);
    }
    const verdict = envelope(time - this.verdictAt, 1 / frame.controlHz + 0.3, 1.8), absorbed = envelope(time - this.absorbedAt, 1.6, 1.4);
    world.bets.forEach((l, side) => l.lit(side === this.bet ? 0.6 + 1.2 * verdict : 0.05));
    world.betLights.forEach((light, side) => { light.color.set(this.correct ? world.gold : this.ctx!.tokens.failureBlue); light.intensity = side === this.bet ? (this.correct ? 5 : 2) * verdict : 0; });
    world.gates.forEach((g, side) => { const hit = side === this.absorbedGate ? absorbed : 0; g.open(hit); g.lit(hit * 1.5); });
    world.walkerLight.position.set(x, DECK_Y + 3, 0.6);
    world.walkerLight.intensity = 0.5 + 5 * absorbed;
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.position = -1; this.moveAt = -Infinity; this.bet = -1; this.verdictAt = -Infinity; this.absorbedAt = -Infinity; }
}

export default ((() => new CanyonWalkwayScene()) satisfies SceneFactory);
