import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { makeLandscape } from '@kit/landscape';
import { envelope } from '@kit/mix';
import { lamp, plinth, propMaterials, relief } from '@kit/props';
import { court, groves } from '@kit/props/court';
import { lampBank } from '@kit/props/lampbank';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { CONDITIONING_ACTION, CONDITIONING_COUNTDOWN } from '@sim/envs/conditioning';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const POLE = new THREE.Vector3(-2.6, 0, -0.8);
const STAND = new THREE.Vector3(2.6, 0, 0.6);
const MAX_BELLS = 3;

function ribbonPoint(k: number, count: number, out: THREE.Vector3): THREE.Vector3 {
  const t = (k + 0.5) / (count + 1);
  if (count <= 16) {
    const angle = Math.PI * (1 - t), radius = (STAND.x - POLE.x) / 2;
    return out.set((POLE.x + STAND.x) / 2 + Math.cos(angle) * radius, 0, (POLE.z + STAND.z) / 2 - Math.sin(angle) * radius * 0.85);
  }
  const turns = 2.5, angle = Math.PI * (1 - t) - turns * Math.PI * 2 * (1 - t), radius = 1.1 + 2.7 * (1 - t);
  return out.set(Math.cos(angle) * radius, 0, -Math.sin(angle) * radius * 0.8 + 0.2);
}

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, interval = parameter(env, 'interval'), pattern = parameter(env, 'pattern') !== 0;
  const palette = ctx.tokens.palettes.dusk, accent = ctx.tokens.accents['conditioning'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, 'conditioningBell', palette, 'dusk', { sky: { sunDir: [0.72, 0.2, 0.66], sunSize: 0.055, sunHdr: 7, sunGlow: 0.45, sunGlowPower: 18, topSpread: 0.62, bottomSpread: 0.3 }, light: { pointCount: 3, pointDistance: 6 }, fogExtent: 90, weather: { preset: 'embers', count: 200, seed: ctx.seed, size: 0.9, intensity: 2, bounds: { y: 1.6, halfX: 6, halfY: 1.6, halfZ: 5 } } });
  const { world, rig } = land;
  rig.sun.intensity = 1.9; rig.hemi.intensity = 0.55;
  rig.setShadowFocus({ x: 0, y: 0.8, z: 0 }, 7);
  const ground = court(land, { name: 'conditioning', seed: ctx.seed, halfX: 5, halfZ: 4, colorA: 0x7a5a36, colorB: 0xb8843c, palette, amplitude: 1.2 });
  groves(land, ctx, ['#d98a3a', '#c25c2e', '#e8b84a', '#a8552c'], 7, 11, ground.heightAt, 4);
  const m = propMaterials(world), parts = modelParts(world);
  const wall = relief(world, 8, 1.5, 0.6, m);
  wall.position.set(0, 0, -3.6);
  world.add(wall);
  const pole = new THREE.Group();
  pole.position.copy(POLE);
  world.add(pole);
  parts.box(pole, m.sandstone, 0, 0.2, 0, 0.7, 0.4, 0.7, 0.04);
  parts.rod(pole, m.brass, new THREE.Vector3(0, 0.4, 0), new THREE.Vector3(0, 3.3, 0), 0.05);
  const bellCount = pattern ? MAX_BELLS : 1, armHalf = (bellCount - 1) * 0.45 + 0.2;
  parts.box(pole, m.brass, 0, 3.25, 0, armHalf * 2 + 0.3, 0.07, 0.07, 0.01);
  const bellGlass = Array.from({ length: bellCount }, () => m.glass(gold, 0.2));
  const bells = bellGlass.map((glass, b) => {
    const bell = new THREE.Group();
    bell.position.set((b - (bellCount - 1) / 2) * 0.9, 3.2, 0);
    pole.add(bell);
    parts.rod(bell, m.brass, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -0.3, 0), 0.015);
    parts.add(bell, new THREE.CylinderGeometry(0.12, 0.28, 0.4, 16, 1, true).translate(0, -0.5, 0), m.brass);
    parts.add(bell, new THREE.SphereGeometry(0.13, 12, 8).scale(1, 0.5, 1).translate(0, -0.3, 0), m.brass);
    parts.add(bell, new THREE.SphereGeometry(0.06, 10, 8).translate(0, -0.7, 0), glass);
    return bell;
  });
  parts.build();
  const stand = plinth(world, 1, 0.5, m);
  stand.position.copy(STAND);
  world.add(stand);
  const sentinel = world.own(buildSentinel({ name: 'conditioningWatcher', tint: accent }));
  sentinel.place(STAND.x, 0.5, STAND.z);
  sentinel.face(-Math.PI * 0.62);
  world.add(sentinel.group);
  const countdown = lampBank(world, new THREE.SphereGeometry(interval > 16 ? 0.07 : 0.11, 10, 8), interval + 1, { emissive: 2 });
  world.add(countdown.mesh);
  const at = new THREE.Vector3();
  for (let k = 0; k <= interval; k++) {
    ribbonPoint(k, interval + 1, at);
    countdown.place(k, at.x, 0.34, at.z); countdown.tint(k, accent); countdown.glow(k, 0.06);
    parts.add(world.group, new THREE.CylinderGeometry(0.06, 0.08, 0.26, 8).translate(at.x, 0.13, at.z), m.shale);
  }
  countdown.commit();
  parts.build();
  const lamps = [-4, 4].map(x => { const l = lamp(world, accent, 1.4, m); l.group.position.set(x, 0, -2.8); world.add(l.group); l.lit(0.5); return l; });
  rig.points.forEach((point, i) => { point.color.set(i === 0 ? gold : i === 1 ? accent : gold); point.intensity = 0; });
  rig.points[0]!.position.set(POLE.x, 2.9, POLE.z);
  rig.points[1]!.position.set(STAND.x, 1.9, STAND.z);
  rig.points[2]!.position.set(STAND.x, 2.2, STAND.z + 0.6);
  const extra = { land, ground, interval, pattern, bells, bellGlass, countdown, sentinel, lamps, bellLight: rig.points[0]!, blinkLight: rig.points[1]!, verdictLight: rig.points[2]!, accent, gold, ribbonHead: new THREE.Vector3() };
  return world.seal<WorldCore & typeof extra>(extra);
}

type BellWorld = ReturnType<typeof buildWorld>;

class BellPoleScene extends StageScene<BellWorld> {
  readonly post: ScenePost = { bloom: 0.44, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'conditioning';
  private view!: SceneView;
  private step = -1;
  private readonly rungAt = new Float64Array(MAX_BELLS).fill(-Infinity);
  private blinkAt = -Infinity;
  private verdictAt = -Infinity;
  private reward = 0;
  private readonly ledger = { remaining: 0, blink: false, reward: 0 };
  override legend: LegendSpec = {
    title: 'BELL POLE',
    rows: [
      { key: 'countdown', label: 'after the bell', tint: 0xf0a8d8, value: () => (this.ledger.remaining > 0 ? `${formatCount(this.ledger.remaining)} steps to the reward` : 'waiting for a bell') },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.blink ? 'blink now' : 'hold') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): BellWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-5, -0.5, -4], max: [5, 3.8, 3.5] }, [0, 1.1, -0.2], { distance: 11, azimuth: 0.28, polar: 1.05 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [10, 8], agentPos: () => this.world!.sentinel.group.position }; }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<BellWorld>): void {
    const { state } = frame, remaining = Math.trunc(state.pose[CONDITIONING_COUNTDOWN]!);
    this.ledger.remaining = remaining; this.ledger.blink = state.step > 0 && (Math.trunc(state.pose[CONDITIONING_ACTION]!) & 1) === 1; this.ledger.reward = state.reward;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.rungAt.fill(-Infinity); this.blinkAt = -Infinity; this.verdictAt = -Infinity; }
    for (const event of frame.events) if (event.type === 'cue') for (let b = 0; b < world.bells.length; b++) if ((event.value >>> b) & 1) { this.rungAt[b] = time; pool.ripple(world.bells[b]!.getWorldPosition(world.ribbonHead), world.gold, 0.35, time); }
    if (state.step !== this.step) {
      this.step = state.step;
      const blink = state.step > 0 && (Math.trunc(state.pose[CONDITIONING_ACTION]!) & 1) === 1;
      this.reward = state.reward;
      if (blink) { this.blinkAt = time; this.verdictAt = time; }
      if (!frame.scrubbed && blink && state.reward > 0) { pool.burst(world.sentinel.group.position, world.gold, 0.5, time); pool.pulse.fire(0.5); }
      else if (!frame.scrubbed && blink && state.reward < 0) pool.sigh(0.3, world.sentinel.group.position, time);
      const head = remaining > 0 ? world.interval + 1 - remaining : -1;
      for (let k = 0; k <= world.interval; k++) world.countdown.glow(k, k === head ? 1.9 : k < head ? 0.35 : 0.06);
    }
    let loudest = 0;
    world.bells.forEach((bell, b) => { const age = time - this.rungAt[b]!, ring = envelope(age, 2.4, 1.3); bell.rotation.z = Math.sin(age * 11) * 0.45 * ring; world.bellGlass[b]!.emissiveIntensity = 0.2 + 3 * ring; loudest = Math.max(loudest, ring); });
    world.bellLight.intensity = 7 * loudest;
    const blink = envelope(time - this.blinkAt, 0.6, 3), verdict = envelope(time - this.verdictAt, 1.1, 1.8);
    world.sentinel.idle(time);
    world.sentinel.glow(blink * 1.6);
    world.blinkLight.intensity = 5 * blink;
    world.verdictLight.color.set(this.reward > 0 ? world.gold : this.ctx!.tokens.failureBlue);
    world.verdictLight.intensity = (this.reward > 0 ? 6 : this.reward < 0 ? 2.5 : 0) * verdict;
    world.lamps.forEach((l, i) => l.lit(0.45 + 0.2 * Math.sin(time * 1.2 + i * 2.2)));
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.rungAt.fill(-Infinity); this.blinkAt = -Infinity; this.verdictAt = -Infinity; }
}

export default ((() => new BellPoleScene()) satisfies SceneFactory);
