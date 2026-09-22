import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { makeMistPlanes } from '@kit/particles';
import { lamp, plinth } from '@kit/props';
import { flameMaterial, makeTerrace } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { CALIBRATION_ACTION, CALIBRATION_FREQUENCY } from '@sim/envs/calibration';
import { parameter } from '../../data/envs';
import { formatPercent, formatSigned } from '../../format';

const BASIN_HEIGHT = 2.2;
const BASIN_RADIUS = 1.1;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const target = parameter(ctx.environment, 'target');
  const t = makeTerrace(ctx, scene, { mood: 'morning', halfX: 5, halfZ: 4.5, ground: [0x6f8f4e, 0xa9c46a], landscape: {
    sky: { sunHdr: 5, sunGlow: 0.32 }, light: { pointCount: 1, pointDistance: 6 }, fogExtent: 70,
    weather: { preset: 'pollen', count: 120, seed: ctx.seed, color: 0xf6f0d8, size: 0.9, bounds: { y: 1.4, halfX: 6, halfY: 1.2, halfZ: 5 } },
  } });
  const { world, rig, m, accent, parts } = t;
  rig.sun.intensity = 2.3; rig.hemi.intensity = 0.8;
  const mist = world.time(world.own(makeMistPlanes({ size: [16, 14], color: 0xe6eef0, seed: ctx.seed, layers: 4, baseY: 0.1, spread: 0.6, opacity: 0.18, name: 'calibrationMist' })));
  world.add(mist.group);
  const base = plinth(world, 3.2, 0.4, m);
  world.add(base);
  const clock = new THREE.Group();
  clock.position.y = 0.4;
  world.add(clock);
    parts.add(clock, new THREE.CylinderGeometry(BASIN_RADIUS + 0.3, BASIN_RADIUS + 0.3, 0.1, 24).translate(0, 0.05, 0), m.shale);
  parts.add(clock, new THREE.TorusGeometry(BASIN_RADIUS + 0.22, 0.05, 8, 40).rotateX(Math.PI / 2).translate(0, BASIN_HEIGHT, 0), m.brass);
  parts.add(clock, new THREE.TorusGeometry(BASIN_RADIUS + 0.24, 0.035, 8, 40).rotateX(Math.PI / 2).translate(0, target * BASIN_HEIGHT, 0), m.brass);
  parts.rod(clock, m.brass, new THREE.Vector3(0, BASIN_HEIGHT + 0.6, -1.9), new THREE.Vector3(0, BASIN_HEIGHT + 0.6, 0), 0.04);
  parts.rod(clock, m.brass, new THREE.Vector3(0, 0.5, -1.9), new THREE.Vector3(0, BASIN_HEIGHT + 0.6, -1.9), 0.05);
  parts.rod(clock, m.brass, new THREE.Vector3(BASIN_RADIUS + 0.2, 0.25, 0), new THREE.Vector3(BASIN_RADIUS + 0.9, 0.12, 0), 0.04);
  parts.build();
  const alabaster = new THREE.Mesh(world.own(new THREE.CylinderGeometry(BASIN_RADIUS + 0.16, BASIN_RADIUS + 0.28, BASIN_HEIGHT, 24, 1, true).translate(0, BASIN_HEIGHT / 2, 0)), world.own(new THREE.MeshStandardMaterial({ color: 0xf1e6cf, roughness: 0.35, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false })));
  alabaster.position.y = 0.4;
  alabaster.renderOrder = 2;
  world.add(alabaster);
  const water = new THREE.Mesh(world.own(new THREE.CylinderGeometry(BASIN_RADIUS + 0.12, BASIN_RADIUS + 0.12, 1, 24).translate(0, 0.5, 0)), world.own(new THREE.MeshStandardMaterial({ color: 0x5fb8d8, emissive: 0x2f7fa8, emissiveIntensity: 0.7, roughness: 0.2 })));
  water.position.y = 0.5;
  world.add(water);
  const targetGlow = new THREE.Mesh(world.own(new THREE.TorusGeometry(BASIN_RADIUS + 0.24, 0.05, 8, 40).rotateX(Math.PI / 2)), world.own(flameMaterial(accent, 1)));
  targetGlow.position.y = 0.4 + target * BASIN_HEIGHT;
  world.add(targetGlow);
  const drop = new THREE.Mesh(world.own(new THREE.SphereGeometry(0.07, 8, 6)), world.own(flameMaterial(0xbfe8ff, 1.4)));
  world.add(drop);
  const lamps = [-3.2, 3.2].map(x => { const l = lamp(world, accent, 1.5, m); l.group.position.set(x, 0, -2.6); world.add(l.group); l.lit(0.5); return l; });
  const sentinel = world.own(buildSentinel({ name: 'calibrationKeeper', tint: accent }));
  sentinel.place(2.6, 0, 1.2);
  sentinel.face(Math.atan2(-2.6, -1.2));
  world.add(sentinel.group);
  const point = rig.points[0];
  if (point) { point.color.set(accent); point.position.set(0, 0.5 + target * BASIN_HEIGHT, 1.6); }
  const extra = { land: t.land, lights: rig, water, targetGlow, drop, lamps, sentinel, target, accent };
  return world.seal<WorldCore & typeof extra>(extra);
}

type ClockWorld = ReturnType<typeof buildWorld>;

class WaterClockScene extends StageScene<ClockWorld> {
  readonly post: ScenePost = { bloom: 0.38, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'waterClock';
  private view!: SceneView;
  private lastStep = -1;
  private dropAt = -Infinity;
  private dropUp = true;
  private readonly ledger = { level: 0, up: true, reward: 0 };
  override legend: LegendSpec = {
    title: 'WATER CLOCK',
    rows: [
      { key: 'level', label: 'level', tint: 0x5fb8d8, value: () => formatPercent(this.ledger.level, 0) },
      { key: 'target', label: 'target ring', tint: 0xf2d38a, value: () => formatPercent(this.world!.target, 0) },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.up ? 'pour in \u2191' : 'drain \u2193') },
      { key: 'miss', label: 'off by', tint: 0x7a95c4, value: () => formatSigned(this.ledger.level - this.world!.target) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): ClockWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-5, -0.5, -4], max: [5, 4, 4] }, [0, 1.5, 0], { distance: 9.5, azimuth: 0.5, polar: 1.15 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [10, 8], agentPos: () => this.world!.sentinel.group.position,
      debug: (_frame, kit) => {
        const world = this.world!, level = this.ledger.level, target = world.target, cyan = this.ctx!.tokens.uiCyan, gold = this.ctx!.tokens.celebrationGold;
        kit.bars('level', 2).set(-1.9, 0.4, 0, level * BASIN_HEIGHT, cyan, 0.16);
        kit.marks('target', 1).set(-1.9, 0.4 + target * BASIN_HEIGHT, 0, 0.12, gold);
      },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<ClockWorld>): void {
    const step = frame.state.step, level = Math.max(0.01, frame.state.pose[CALIBRATION_FREQUENCY]!), up = Math.round(frame.state.pose[CALIBRATION_ACTION]!) === 1;
    this.ledger.level = level; this.ledger.up = up; this.ledger.reward = frame.state.reward;
    if (step !== this.lastStep) { this.lastStep = step; this.dropAt = time; this.dropUp = up; }
    world.water.scale.y = level * BASIN_HEIGHT;
    const fall = Math.min(1, (time - this.dropAt) * 3), y = 0.4 + (this.dropUp ? BASIN_HEIGHT + 0.5 - fall * (BASIN_HEIGHT + 0.5 - level * BASIN_HEIGHT) : 0.2 - fall * 0.5);
    world.drop.position.set(this.dropUp ? 0 : BASIN_RADIUS + 0.9, y, 0);
    world.drop.visible = fall < 1;
    const miss = Math.abs(level - world.target), closeness = Math.exp(-miss * 9);
    (world.targetGlow.material as THREE.MeshBasicMaterial).opacity = 0.2 + 0.7 * closeness;
    world.targetGlow.scale.setScalar(1 + 0.03 * Math.sin(time * 3));
    const point = world.lights.points[0];
    if (point) point.intensity = 1 + 5 * closeness;
    world.lamps.forEach((l, i) => l.lit(0.4 + 0.2 * Math.sin(time * 1.1 + i)));
    world.sentinel.idle(time);
    world.sentinel.glow(closeness);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.dropAt = -Infinity; }
}

export default ((() => new WaterClockScene()) satisfies SceneFactory);
