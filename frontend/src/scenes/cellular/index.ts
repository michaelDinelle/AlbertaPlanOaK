import * as THREE from 'three';
import { makeLandscape } from '@kit/landscape';
import { envelope } from '@kit/mix';
import { makeAmbientParticles } from '@kit/particles';
import { lamp, plinth, propMaterials } from '@kit/props';
import { court, groves } from '@kit/props/court';
import { lampBank } from '@kit/props/lampbank';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { CELLULAR_ACTION, CELLULAR_STATE, cellularNext } from '@sim/envs/cellular';
import { constant } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const popcount = (v: number): number => { let n = 0; for (let x = v >>> 0; x !== 0; x >>>= 1) n += x & 1; return n; };

const RING = 2.8;
const LAMP_Y = 1.05;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, cells = constant(env, 'CELLULAR_CELLS'), rule = constant(env, 'CELLULAR_RULE');
  const palette = ctx.tokens.palettes.afternoon, accent = ctx.tokens.accents['cellular'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, 'cellularRibbon', palette, 'afternoon', { sky: { sunDir: [0.62, 0.34, 0.7], sunSize: 0.05, sunHdr: 5, sunGlow: 0.38, sunGlowPower: 20 }, light: { pointCount: 3, pointDistance: 6 }, fogExtent: 80 });
  const { world, rig, sky } = land;
  sky.setColors(0x4a80b8, 0xc9d8e4, 0xdde4ea);
  rig.sun.intensity = 1.8; rig.sun.color.set(0xfff1dc); rig.hemi.intensity = 0.6; rig.hemi.color.set(0xc4d8ec);
  rig.setShadowFocus({ x: 0, y: 0.6, z: 0 }, 6);
  const ground = court(land, { name: 'cellular', seed: ctx.seed, halfX: 4.5, halfZ: 4.5, colorA: 0x5a6a70, colorB: 0x8a9aa0, palette, amplitude: 1.3, snow: 1 });
  groves(land, ctx, [], 7, 11, ground.heightAt, 3);
  const snow = world.size(world.time(world.own(makeAmbientParticles({ preset: 'snow', count: 900, seed: ctx.seed, size: 1.3, intensity: 0.9, bounds: { y: 2.2, halfX: 9, halfY: 2.4, halfZ: 9 }, name: 'cellularSnow' }))));
  world.add(snow.points);
  const m = propMaterials(world), parts = modelParts(world);
  const cellAt = (i: number, out: THREE.Vector3, radius = RING): THREE.Vector3 => { const angle = i / cells * Math.PI * 2; return out.set(Math.sin(angle) * radius, 0, -Math.cos(angle) * radius); };
  const at = new THREE.Vector3();
  for (let i = 0; i < cells; i++) {
    cellAt(i, at);
    parts.box(world.group, m.sandstone, at.x, 0.14, at.z, 0.34, 0.28, 0.34, 0.03);
    parts.rod(world.group, m.brass, new THREE.Vector3(at.x, 0.28, at.z), new THREE.Vector3(at.x, LAMP_Y - 0.16, at.z), 0.025);
    parts.add(world.group, new THREE.TorusGeometry(i === 1 ? 0.3 : 0.2, 0.025, 6, 24).rotateX(Math.PI / 2).translate(at.x, LAMP_Y - 0.14, at.z), i === 1 ? m.brass : m.shale);
  }
  parts.add(world.group, new THREE.TorusGeometry(RING, 0.04, 8, 96).rotateX(Math.PI / 2).translate(0, 0.05, 0), m.brass);
  const lever = new THREE.Group();
  cellAt(0, at, RING + 0.9);
  lever.position.set(at.x, 0.5, at.z);
  world.add(lever);
  parts.box(world.group, m.sandstone, at.x, 0.25, at.z, 0.5, 0.5, 0.5, 0.04);
  parts.rod(lever, m.brass, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0.7, 0), 0.03);
  parts.add(lever, new THREE.SphereGeometry(0.09, 10, 8).translate(0, 0.72, 0), m.brass);
  parts.build();
  const lamps = lampBank(world, new THREE.SphereGeometry(0.16, 14, 10), cells, { emissive: 2 });
  world.add(lamps.mesh);
  for (let i = 0; i < cells; i++) { cellAt(i, at); lamps.place(i, at.x, LAMP_Y, at.z, 0, i === 1 ? 1.35 : 1); lamps.tint(i, accent); lamps.glow(i, 0.08); }
  lamps.commit();
  const hub = plinth(world, 1.2, 0.6, m);
  world.add(hub);
  const ruleLamps = lampBank(world, new THREE.SphereGeometry(0.055, 8, 6), 8, { emissive: 1.8 });
  world.add(ruleLamps.mesh);
  for (let b = 0; b < 8; b++) { const angle = b / 8 * Math.PI * 2; ruleLamps.place(b, Math.sin(angle) * 0.42, 0.72, -Math.cos(angle) * 0.42); ruleLamps.tint(b, gold); ruleLamps.glow(b, (rule >>> (7 - b)) & 1 ? 1.4 : 0.05); }
  ruleLamps.commit();
  const poleLamps = [-3.6, 3.6].map(x => { const l = lamp(world, accent, 1.5, m); l.group.position.set(x, 0, -3.2); world.add(l.group); l.lit(0.45); return l; });
  cellAt(1, at);
  rig.points[0]!.color.set(gold); rig.points[0]!.position.set(at.x, LAMP_Y + 0.3, at.z); rig.points[0]!.intensity = 0;
  cellAt(0, at);
  rig.points[1]!.color.set(accent); rig.points[1]!.position.set(at.x, LAMP_Y + 0.3, at.z); rig.points[1]!.intensity = 0;
  cellAt(cells - 1, at);
  rig.points[2]!.color.set(0xffffff); rig.points[2]!.position.set(at.x, LAMP_Y + 0.3, at.z); rig.points[2]!.intensity = 0;
  const extra = { land, ground, cells, rule, lamps, lever, cellAt, snow, poleLamps, rewardLight: rig.points[0]!, leverLight: rig.points[1]!, gustLight: rig.points[2]!, accent, gold, rewardBase: cellAt(1, new THREE.Vector3()) };
  return world.seal<WorldCore & typeof extra>(extra);
}

type RibbonWorld = ReturnType<typeof buildWorld>;

class LampRibbonScene extends StageScene<RibbonWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'cellular';
  private view!: SceneView;
  private step = -1;
  private flipAt = -Infinity;
  private gustAt = -Infinity;
  private rewardAt = -Infinity;
  private leverTarget = 0;
  private leverAngle = 0;
  private rewarded = false;
  private readonly ledger = { lit: 0, flip: false, reward: 0 };
  override legend: LegendSpec = {
    title: 'LAMP RING',
    rows: [
      { key: 'rule', label: 'rule', tint: 0xffd166, value: () => `${formatCount(this.world!.rule)}` },
      { key: 'lit', label: 'lamps lit', tint: 0xb5f0a0, value: () => `${formatCount(this.ledger.lit)} of ${this.world!.cells}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.flip ? 'throw the lever' : 'let it run') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): RibbonWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-4.5, -0.5, -4.5], max: [4.5, 2.5, 4.5] }, [0, 0.7, 0], { distance: 10, azimuth: 0.15, polar: 0.95 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [10, 10] }; }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<RibbonWorld>): void {
    const { state } = frame, cells = Math.trunc(state.pose[CELLULAR_STATE]!) >>> 0;
    this.ledger.lit = popcount(cells); this.ledger.flip = state.step > 0 && (Math.trunc(state.pose[CELLULAR_ACTION]!) & 1) === 1; this.ledger.reward = state.reward;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.flipAt = -Infinity; this.gustAt = -Infinity; this.rewardAt = -Infinity; }
    if (state.step !== this.step) {
      this.step = state.step;
      const before = frame.frameAt(state.step - 1).words[0]!, flip = state.step > 0 && (Math.trunc(state.pose[CELLULAR_ACTION]!) & 1) === 1;
      const expected = cellularNext(flip ? before ^ 1 : before, world.cells, world.rule);
      const top = world.cells - 1;
      if (state.step > 0 && ((expected >>> top) & 1) !== ((cells >>> top) & 1)) this.gustAt = time;
      if (flip) { this.flipAt = time; this.leverTarget = this.leverTarget === 0 ? 1 : 0; }
      const rewarded = (cells >>> 1) & 1 ? true : false;
      if (rewarded && !this.rewarded && !frame.scrubbed) { this.rewardAt = time; pool.ripple(world.rewardBase, world.gold, 0.45, time); }
      this.rewarded = rewarded;
      for (let i = 0; i < world.cells; i++) { world.lamps.glow(i, (cells >>> i) & 1 ? 1.7 : 0.08); world.lamps.tint(i, i === 1 && rewarded ? world.gold : world.accent); }
    }
    this.leverAngle += ((this.leverTarget === 0 ? 0.55 : -0.55) - this.leverAngle) * (1 - Math.exp(-14 * frame.dtReal));
    world.lever.rotation.x = this.leverAngle;
    const flip = envelope(time - this.flipAt, 0.5, 3), gust = envelope(time - this.gustAt, 1.8, 1.6), reward = envelope(time - this.rewardAt, 0.9, 2);
    world.leverLight.intensity = 5 * flip;
    world.gustLight.intensity = 6 * gust;
    world.rewardLight.intensity = 6 * reward;
    if (gust > 0) world.lamps.glow(world.cells - 1, Math.max(world.lamps.glowOf(world.cells - 1), 2.4 * gust));
    world.snow.setIntensity(1.1 + 1.6 * gust);
    (world.snow.material.uniforms.uSwirl as { value: number }).value = 0.26 + 1.4 * gust;
    world.poleLamps.forEach((l, i) => l.lit(0.4 + 0.2 * Math.sin(time * 1.2 + i * 2)));
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.flipAt = -Infinity; this.gustAt = -Infinity; this.rewardAt = -Infinity; this.leverTarget = 0; this.leverAngle = 0; this.rewarded = false; }
}

export default ((() => new LampRibbonScene()) satisfies SceneFactory);
