import * as THREE from 'three';
import { makeLandscape } from '@kit/landscape';
import { oakUniforms } from '@kit/materials';
import { makeMeadow } from '@kit/meadow';
import { envelope } from '@kit/mix';
import { banner, lamp, plinth, propMaterials, relief } from '@kit/props';
import { dial } from '@kit/props/dial';
import { modelParts } from '@kit/robot';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { COVARIATE_ACTION, COVARIATE_WEIGHTS } from '@sim/envs/covariate';
import { constant, parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const STATION_Y = 0.9;
const STATION_Z = 0.4;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, inputs = constant(env, 'COVARIATE_INPUTS'), period = constant(env, 'COVARIATE_PERIOD'), shift = parameter(env, 'shift');
  if (inputs !== 4) throw new Error('The weather station has four instruments');
  const palette = ctx.tokens.palettes.afternoon, accent = ctx.tokens.accents['covariate'] ?? ctx.tokens.celebrationGold, gold = ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, 'covariateStation', palette, 'afternoon', { sky: { sunDir: [0.6, 0.5, 0.62], sunSize: 0.05, sunHdr: 6, sunGlow: 0.36, sunGlowPower: 22 }, light: { pointCount: 2, pointDistance: 6 }, fogExtent: 120 });
  const { world, rig, sky } = land;
  rig.sun.intensity = 2.4; rig.hemi.intensity = 0.7;
  rig.setShadowFocus({ x: 0, y: 0.8, z: 0 }, 7);
  const meadow = makeMeadow(world, ctx, { name: 'covariateStation', halfX: 6, halfZ: 6.5, ground: 0x7f9a52, grass: 0xa8d067, trees: [0xf2c4d6, 0xa9d954, 0xe9f0a4, 0xd4e8b0], amplitude: 1.2 });
  const m = propMaterials(world), parts = modelParts(world);
  const wall = relief(world, 9, 1.4, 0.8, m);
  wall.position.set(0, 0, -3.2);
  world.add(wall);
  const cloths: ReturnType<typeof oakUniforms>[] = [];
  for (const x of [-4.4, 4.4]) { const b = banner(world, accent, 2.6, m); b.position.set(x, 0, -2.6); world.add(b); cloths.push(oakUniforms((b.children[1] as THREE.Mesh).material as THREE.Material)); }
  const columns = [-3, -1, 1, 3].map(x => { const p = plinth(world, 0.6, STATION_Y, m); p.position.set(x, 0, STATION_Z); world.add(p); return p; });
  const vane = new THREE.Group();
  vane.position.set(-3, STATION_Y + 0.55, STATION_Z);
  world.add(vane);
  parts.rod(world.group, m.brass, new THREE.Vector3(-3, STATION_Y, STATION_Z), new THREE.Vector3(-3, STATION_Y + 0.55, STATION_Z), 0.025);
  parts.box(vane, m.brass, 0, 0, 0, 0.9, 0.03, 0.03, 0.01);
  parts.add(vane, new THREE.ConeGeometry(0.09, 0.22, 4).rotateZ(-Math.PI / 2).translate(0.5, 0, 0), m.brass);
  parts.box(vane, m.brass, -0.42, 0, 0, 0.16, 0.22, 0.02, 0.01);
  const barometer = dial(world, 0.42, 16, accent, m, 4);
  barometer.group.position.set(-1, STATION_Y, STATION_Z);
  world.add(barometer.group);
  const columnGlass = m.glass(accent, 1.2);
  const column = new THREE.Mesh(world.own(new THREE.CylinderGeometry(0.08, 0.08, 1, 12).translate(0, 0.5, 0)), columnGlass);
  column.position.set(1, STATION_Y, STATION_Z);
  world.add(column);
  parts.add(world.group, new THREE.CylinderGeometry(0.11, 0.11, 1.05, 12, 1, true).translate(1, STATION_Y + 0.525, STATION_Z), m.glass(0xffffff, 0));
  parts.box(world.group, m.brass, 1, STATION_Y + 1.06, STATION_Z, 0.3, 0.04, 0.3, 0.01);
  const cups = new THREE.Group();
  cups.position.set(3, STATION_Y + 0.6, STATION_Z);
  world.add(cups);
  parts.rod(world.group, m.brass, new THREE.Vector3(3, STATION_Y, STATION_Z), new THREE.Vector3(3, STATION_Y + 0.6, STATION_Z), 0.025);
  for (let i = 0; i < 3; i++) {
    const angle = i / 3 * Math.PI * 2, end = new THREE.Vector3(Math.cos(angle) * 0.32, 0, Math.sin(angle) * 0.32);
    parts.rod(cups, m.brass, new THREE.Vector3(0, 0, 0), end, 0.015);
    parts.add(cups, new THREE.SphereGeometry(0.075, 10, 8, 0, Math.PI * 2, 0, Math.PI / 2).rotateZ(Math.PI / 2).rotateY(-angle).translate(end.x, end.y, end.z), m.brass);
  }
  const rack = new THREE.Group();
  rack.position.set(0, 0, -2.4);
  world.add(rack);
  parts.box(rack, m.sandstone, 0, 0.2, 0, 2.6, 0.4, 0.7, 0.04);
  const weightRods = Array.from({ length: inputs + 1 }, (_, j) => {
    const rod = new THREE.Group();
    rod.position.set((j - inputs / 2) * 0.45, 0.4, 0);
    rack.add(rod);
    parts.add(rod, new THREE.CylinderGeometry(0.035, 0.045, 1, 8).translate(0, 0.5, 0), j === inputs ? m.shale : m.brass);
    return rod;
  });
  parts.build();
  const answers = [-0.8, 0.8].map((x, i) => { const l = lamp(world, i === 0 ? ctx.tokens.cueCyan : gold, 1.1, m); l.group.position.set(x, 0, 2.3); world.add(l.group); return l; });
  const verdictGlass = m.glass(gold, 0);
  const verdict = new THREE.Mesh(world.own(new THREE.TorusGeometry(0.32, 0.035, 8, 36).rotateX(Math.PI / 2)), verdictGlass);
  verdict.position.set(0, 0.7, 2.3);
  world.add(verdict);
  parts.rod(world.group, m.brass, new THREE.Vector3(0, 0, 2.3), new THREE.Vector3(0, 0.7, 2.3), 0.025);
  parts.build();
  const lamps = [-2.2, 2.2].map(x => { const l = lamp(world, accent, 1.5, m); l.group.position.set(x, 0, -2.6); world.add(l.group); l.lit(0.45); return l; });
  const extra = { land, meadow, sky, period, shift, vane, barometer, column, columnGlass, cups, weightRods, answers, verdictGlass, cloths, lamps, accent, gold, warmSky: [0x5da2d6, 0xe8c9a0, 0xe8dfb0] as const, coolSky: [0x3f7ab8, 0xb9d3c8, 0xc7d6c2] as const };
  return world.seal<WorldCore & typeof extra>(extra);
}

type StationWorld = ReturnType<typeof buildWorld>;

class WeatherStationScene extends StageScene<StationWorld> {
  readonly post: ScenePost = { bloom: 0.36, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 5;
  protected readonly stageName = 'covariate';
  private view!: SceneView;
  private step = -1;
  private verdictAt = -Infinity;
  private spin = 0;
  private spinRate = 0;
  private answer = -1;
  private correct = false;
  private readonly tint = new THREE.Color();
  private readonly ledger = { answer: -1, correct: false, shifted: false, reward: 0 };
  override legend: LegendSpec = {
    title: 'WEATHER STATION',
    rows: [
      { key: 'front', label: 'front', tint: 0x9ed9d0, value: () => (this.ledger.shifted ? 'cold air (shifted readings)' : 'warm air') },
      { key: 'answer', label: 'forecast', tint: 0xdcf4ff, value: () => (this.ledger.answer < 0 ? '\u2014' : this.ledger.answer === 0 ? 'cyan lamp' : 'gold lamp') },
      { key: 'verdict', label: 'verdict', tint: 0xffd166, value: () => (this.ledger.answer < 0 ? '\u2014' : this.ledger.correct ? 'right' : 'wrong') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): StationWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-5, -0.5, -3.5], max: [5, 3, 3.5] }, [0, 0.9, 0], { distance: 10.5, azimuth: 0.36, polar: 1.1 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      floor: [10, 8],
      debug: (frame, kit) => {
        const pose = frame.state.pose, obs = frame.state.observations, gold = this.ctx!.tokens.celebrationGold, cyan = this.ctx!.tokens.cueCyan;
        const inputs = kit.bars('inputs', 4), weights = kit.bars('weights', 5);
        let sum = pose[COVARIATE_WEIGHTS + 4]!;
        for (let j = 0; j < 4; j++) {
          const x = [-3, -1, 1, 3][j]!, value = obs[j] ?? 0, w = pose[COVARIATE_WEIGHTS + j]!;
          sum += w * value;
          inputs.set(x - 0.35, STATION_Y + 1.2, STATION_Z, value * 0.35, 0xdcf4ff, 0.1);
          weights.set((j - 2) * 0.45, 0.4, -2.4, w * 0.45, w >= 0 ? gold : cyan, 0.1);
        }
        weights.set(2 * 0.45, 0.4, -2.4, pose[COVARIATE_WEIGHTS + 4]! * 0.45, 0xffffff, 0.1);
        kit.marks('truth', 1).set(sum > 0 ? 0.8 : -0.8, 1.6, 2.3, 0.14, sum > 0 ? gold : cyan);
      },
    };
  }
  protected start(_ctx: SceneContext, world: StationWorld): void {
    world.weightRods.forEach(rod => { rod.scale.y = 1; });
    if (world.shift !== 0) { this.legend.note = () => `front turns in ${formatCount(world.period - Math.max(0, this.step) % world.period)} steps`; this.legend.bar = () => (Math.max(0, this.step) % world.period) / world.period; }
  }
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<StationWorld>): void {
    const { state } = frame;
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.verdictAt = -Infinity; }
    if (state.step !== this.step) {
      this.step = state.step;
      const asked = frame.frameAt(state.step - 1).observations;
      this.answer = state.step === 0 ? -1 : Math.trunc(state.pose[COVARIATE_ACTION]!) & 1;
      this.correct = state.reward > 0;
      this.verdictAt = time;
      world.vane.rotation.y = -asked[0]! * 0.7;
      world.barometer.setAngle(asked[1]! * 0.75);
      const heat = asked[2]!;
      world.column.scale.y = Math.max(0.04, 0.5 + heat * 0.22);
      world.columnGlass.color.set(heat >= 0 ? 0xffb066 : 0x7fc8ff);
      world.columnGlass.emissive.copy(world.columnGlass.color);
      this.spinRate = asked[3]! * 5;
      world.weightRods.forEach((rod, j) => { const w = state.pose[COVARIATE_WEIGHTS + j]!; rod.scale.y = 0.15 + Math.min(2.6, Math.abs(w)) * 0.45; rod.rotation.z = w >= 0 ? 0 : Math.PI; rod.position.y = w >= 0 ? 0.4 : 0.4 + rod.scale.y; });
    }
    this.spin += this.spinRate * frame.dtReal;
    world.cups.rotation.y = this.spin;
    const shifted = world.shift !== 0 && ((Math.floor(state.step / world.period)) & 1) === 1;
    this.ledger.answer = this.answer; this.ledger.correct = this.correct; this.ledger.shifted = shifted; this.ledger.reward = state.reward;
    const [top, horizon, bottom] = shifted ? world.coolSky : world.warmSky;
    world.sky.setColors(top, horizon, bottom);
    world.land.rig.hemi.color.set(shifted ? 0x9fc4e8 : 0xc9e2ea);
    world.land.rig.sun.color.set(shifted ? 0xd9e8ff : 0xfff3d6);
    world.cloths.forEach((cloth, i) => cloth?.setWind(shifted ? 0.12 : 0.05, shifted ? 2.4 : 1.4 + i * 0.1));
    const verdict = envelope(time - this.verdictAt, 1 / frame.controlHz + 0.4, 1.6);
    world.answers.forEach((l, i) => l.lit(i === this.answer ? 0.8 + 1.2 * verdict : 0.08));
    world.verdictGlass.color.set(this.correct ? world.gold : this.ctx!.tokens.failureBlue);
    world.verdictGlass.emissive.copy(world.verdictGlass.color);
    world.verdictGlass.emissiveIntensity = (this.correct ? 2.6 : 1) * verdict;
    world.lamps.forEach((l, i) => l.lit(0.4 + 0.2 * Math.sin(time * 1.3 + i * 2)));
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera, world.meadow);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.step = -1; this.verdictAt = -Infinity; this.spin = 0; this.spinRate = 0; this.answer = -1; }
}

export default ((() => new WeatherStationScene()) satisfies SceneFactory);
