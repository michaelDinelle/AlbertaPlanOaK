import * as THREE from 'three';
import { makeLandscape } from '@kit/landscape';
import { envelope } from '@kit/mix';
import { makeMistPlanes } from '@kit/particles';
import { lamp, propMaterials, relief } from '@kit/props';
import { court, groves } from '@kit/props/court';
import { dial } from '@kit/props/dial';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { NEXTING_ACTION, NEXTING_PHASE } from '@sim/envs/nexting';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const DIAL_RADIUS = 1.5;
const BEACON_Y = 4.25;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, period = constant(env, 'NEXTING_PERIOD'), horizon = parameter(env, 'horizon');
  const palette = ctx.tokens.palettes.dawn, accent = ctx.tokens.accents['nexting'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, 'nextingTower', palette, 'dawn', { sky: { sunDir: [0.62, 0.24, 0.75], sunSize: 0.06, sunHdr: 6, sunGlow: 0.5, sunGlowPower: 14, topSpread: 0.6, bottomSpread: 0.3 }, light: { pointCount: 2, pointDistance: 7 }, fogExtent: 90 });
  const { world, rig } = land;
  rig.sun.intensity = 1.9; rig.sun.color.set(0xffd2a0); rig.hemi.intensity = 0.7;
  rig.setShadowFocus({ x: 0, y: 1.2, z: -1 }, 7);
  const ground = court(land, { name: 'nexting', seed: ctx.seed, halfX: 4.2, halfZ: 4.2, colorA: 0x5b7a46, colorB: 0xa3b562, palette, amplitude: 1.1 });
  groves(land, ctx, ['#7fa34a', '#a9c25e', '#8fb85c', '#c4cf7a'], 7.5, 11, ground.heightAt, 3);
  const m = propMaterials(world), parts = modelParts(world);
  const tower = new THREE.Group();
  tower.position.set(0, 0, -2.7);
  world.add(tower);
  for (let row = 0; row < 9; row++) {
    const side = 1.6 - row * 0.07, offset = (row % 2) * 0.03;
    parts.box(tower, row % 3 === 2 ? m.shale : m.sandstone, offset, 0.2 + row * 0.4, 0, side, 0.38, side, 0.04);
  }
  parts.box(tower, m.brass, 0, 3.83, 0, 1.2, 0.06, 1.2, 0.01);
  for (const [x, z] of [[-0.42, -0.42], [0.42, -0.42], [-0.42, 0.42], [0.42, 0.42]] as const) parts.rod(tower, m.brass, new THREE.Vector3(x, 3.86, z), new THREE.Vector3(x, 4.6, z), 0.03);
  parts.add(tower, new THREE.ConeGeometry(0.85, 0.5, 4).rotateY(Math.PI / 4).translate(0, 4.85, 0), m.brass);
  parts.build();
  const beaconGlass = m.glass(accent, 0.3);
  const beacon = new THREE.Mesh(world.own(new THREE.SphereGeometry(0.24, 18, 14)), beaconGlass);
  beacon.position.set(0, BEACON_Y, 0);
  tower.add(beacon);
  const verdictGlass = m.glass(gold, 0);
  const verdict = new THREE.Mesh(world.own(new THREE.TorusGeometry(0.4, 0.035, 8, 40).rotateX(Math.PI / 2)), verdictGlass);
  verdict.position.set(0, BEACON_Y - 0.05, 0);
  tower.add(verdict);
  const beaconLight = rig.points[0]!;
  beaconLight.color.set(accent); beaconLight.position.set(0, BEACON_Y, -2.7);
  const pulseLight = rig.points[1]!;
  pulseLight.color.set(gold); pulseLight.position.set(0, 0.9, 0.9);
  const sundial = dial(world, DIAL_RADIUS, period, accent, m, 10);
  sundial.group.position.set(0, 0, 0.9);
  world.add(sundial.group);
  const arcStart = Math.PI * 2 * (period - horizon) / period, arcLength = Math.PI * 2 * horizon / period;
  const arcGlass = m.glass(accent, 0.8);
  const arc = new THREE.Mesh(world.own(new THREE.TorusGeometry(DIAL_RADIUS - 0.22, 0.045, 6, Math.max(6, horizon * 3), arcLength).rotateX(Math.PI / 2).rotateY(-(arcStart + Math.PI / 2))), arcGlass);
  arc.position.set(0, 0.16, 0.9);
  world.add(arc);
  const pulseGlass = m.glass(gold, 0.4);
  const pulseMark = new THREE.Mesh(world.own(new THREE.SphereGeometry(0.09, 12, 8)), pulseGlass);
  pulseMark.position.set(0, 0.2, 0.9 + DIAL_RADIUS - 0.1);
  world.add(pulseMark);
  const beadGlass = m.glass(0xfff4d8, 1.2);
  const bead = new THREE.Mesh(world.own(new THREE.SphereGeometry(0.06, 10, 8)), beadGlass);
  world.add(bead);
  const wall = relief(world, 7, 1.3, 0.7, m);
  wall.position.set(0, 0, -4.6);
  world.add(wall);
  const lamps = [-3, 3].map(x => { const l = lamp(world, accent, 1.4, m); l.group.position.set(x, 0, -1.2); world.add(l.group); l.lit(0.5); return l; });
  const mist = world.time(world.own(makeMistPlanes({ size: [16, 14], color: palette.fog, seed: ctx.seed, layers: 3, baseY: 0.1, spread: 0.4, opacity: 0.07, speed: 0.008, scale: 2.2, nearFade: 3, name: 'nextingMist' })));
  world.add(mist.group);
  const extra = { land, ground, period, horizon, sundial, beaconGlass, verdictGlass, arcGlass, pulseGlass, bead, lamps, beaconLight, pulseLight, accent, gold, dialCenter: new THREE.Vector3(0, 0.2, 0.9) };
  return world.seal<WorldCore & typeof extra>(extra);
}

type TowerWorld = ReturnType<typeof buildWorld>;

class SignalTowerScene extends StageScene<TowerWorld> {
  readonly post: ScenePost = { bloom: 0.46, fog: { from: ['dawn', 'fog'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'nexting';
  private view!: SceneView;
  private pulseAt = -Infinity;
  private verdictAt = -Infinity;
  private verdictStep = -1;
  private readonly ledger = { phase: 0, within: false, predicted: false, reward: 0 };
  override legend: LegendSpec = {
    title: 'SIGNAL TOWER',
    rows: [
      { key: 'dial', label: 'dial', tint: 0x7fd6e8, value: () => `${formatCount(this.ledger.phase)} of ${this.world!.period}` },
      { key: 'pulse', label: 'pulse', tint: 0xffd166, value: () => (this.ledger.within ? `due within ${this.world!.horizon} steps` : 'not yet') },
      { key: 'beacon', label: 'beacon says', tint: 0xdcf4ff, value: () => (this.ledger.predicted ? 'pulse coming' : 'quiet') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): TowerWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-4.5, -0.5, -5], max: [4.5, 5.2, 4] }, [0, 1.9, -0.9], { distance: 12.5, azimuth: 0.5, polar: 1.12 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [10, 10],
      debug: (frame, kit) => {
        const world = this.world!, phase = frame.state.pose[NEXTING_PHASE]!, gold = this.ctx!.tokens.celebrationGold;
        const angle = Math.PI * 2 * phase / world.period, observed = Math.atan2(frame.state.observations[0] ?? 0, frame.state.observations[1] ?? 1);
        const marks = kit.marks('dial', 2 + world.horizon);
        marks.set(-Math.sin(angle) * DIAL_RADIUS, 0.55, 0.9 + Math.cos(angle) * DIAL_RADIUS, 0.12, gold);
        marks.set(-Math.sin(observed) * (DIAL_RADIUS - 0.42), 0.55, 0.9 + Math.cos(observed) * (DIAL_RADIUS - 0.42), 0.09, this.ctx!.tokens.cueMagenta);
        for (let k = 1; k <= world.horizon; k++) { const a = Math.PI * 2 * ((world.period - k) / world.period); marks.set(-Math.sin(a) * (DIAL_RADIUS + 0.3), 0.4, 0.9 + Math.cos(a) * (DIAL_RADIUS + 0.3), 0.07, gold); }
      },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<TowerWorld>): void {
    const { state } = frame, phase = state.pose[NEXTING_PHASE]!, predicted = state.pose[NEXTING_ACTION]! >= 0.5;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.pulseAt = -Infinity; this.verdictAt = -Infinity; this.verdictStep = -1; }
    for (const event of frame.events) if (event.type === 'cue') { this.pulseAt = time; pool.ripple(world.dialCenter, world.gold, 0.6, time); pool.pulse.fire(0.35); }
    if (state.step !== this.verdictStep && state.step > 0) { this.verdictStep = state.step; this.verdictAt = time; }
    const pulse = envelope(time - this.pulseAt, 1.6, 2.4), verdict = envelope(time - this.verdictAt, 0.9, 2.2);
    world.sundial.setAngle(Math.PI * 2 * phase / world.period);
    const observed = Math.atan2(state.observations[0] ?? 0, state.observations[1] ?? 1);
    world.bead.position.set(-Math.sin(observed) * (DIAL_RADIUS - 0.42), 0.24, 0.9 + Math.cos(observed) * (DIAL_RADIUS - 0.42));
    const within = (phase + world.horizon) % world.period < world.horizon;
    this.ledger.phase = Math.round(phase); this.ledger.within = within; this.ledger.predicted = predicted; this.ledger.reward = state.reward;
    world.arcGlass.emissiveIntensity = within ? 2.2 + 0.4 * Math.sin(time * 6) : 0.7;
    world.pulseGlass.emissiveIntensity = 0.4 + 3 * pulse;
    world.pulseLight.intensity = 9 * pulse;
    world.beaconGlass.emissiveIntensity = predicted ? 2.6 : 0.22;
    world.beaconLight.intensity = predicted ? 5 : 0.4;
    world.verdictGlass.color.set(state.reward > 0 ? world.gold : this.ctx!.tokens.failureBlue);
    world.verdictGlass.emissive.copy(world.verdictGlass.color);
    world.verdictGlass.emissiveIntensity = (state.reward > 0 ? 2.4 : 0.9) * verdict;
    world.lamps.forEach((l, i) => l.lit(0.45 + 0.2 * Math.sin(time * 1.1 + i * 2)));
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.pulseAt = -Infinity; this.verdictAt = -Infinity; this.verdictStep = -1; }
}

export default ((() => new SignalTowerScene()) satisfies SceneFactory);
