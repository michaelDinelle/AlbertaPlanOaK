import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { envelope } from '@kit/mix';
import { banner, plinth } from '@kit/props';
import { flameMaterial, makeTerrace } from '@kit/props/decision-terrace';
import type { GroundTruthOptions } from '@kit/debug';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { HIDDENMODE_ACTION, HIDDENMODE_MODE } from '@sim/envs/hiddenmode';
import { parameter } from '../../data/envs';
import { formatSigned } from '../../format';

function vane(parts: ReturnType<typeof makeTerrace>['parts'], m: ReturnType<typeof makeTerrace>['m'], size: number) {
  const group = new THREE.Group();
  parts.rod(group, m.brass, new THREE.Vector3(-size, 0, 0), new THREE.Vector3(size, 0, 0), 0.03 * size);
  parts.add(group, new THREE.ConeGeometry(0.12 * size, 0.4 * size, 8).rotateZ(-Math.PI / 2).translate(size + 0.15 * size, 0, 0), m.brass);
  parts.box(group, m.brass, -size + 0.2 * size, 0, 0, 0.5 * size, 0.42 * size, 0.03 * size, 0.01);
  return group;
}

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const switchRate = parameter(ctx.environment, 'switchRate'), gale = Math.min(1, Math.max(0, (Math.log10(switchRate) + 3) / 1.5));
  const t = makeTerrace(ctx, scene, { mood: 'afternoon', halfX: 5, halfZ: 4.5, ground: [0x5b6d4a, 0x7f8f5c], amplitude: 1.6, landscape: {
    sky: { sunHdr: 1.2, sunGlow: 0.2 }, light: { pointCount: 1, pointDistance: 6 }, fogDensity: 0.04,
    weather: { preset: 'snow', count: 500 + Math.round(500 * gale), seed: ctx.seed, color: 0xc4d4e2, size: 1.1, opacity: 0.5, rise: -3.2 - 1.5 * gale, swirl: 0.08 + 0.2 * gale, bounds: { y: 3, halfX: 9, halfY: 3, halfZ: 7 } },
  } });
  const { world, rig, m, accent, parts } = t;
  t.land.sky.setColors(0x4f5c6c, 0x8b95a1, 0x6b7681);
  t.land.fog.color.set(0x8a939e);
  rig.sun.color.set(0xcfd8e2); rig.sun.intensity = 1.1; rig.hemi.color.set(0x9aa6b4); rig.hemi.intensity = 0.7;
  const tower = plinth(world, 0.9, 2.8, m);
  tower.position.set(0, 0, -2);
  world.add(tower);
  const mast = new THREE.Group();
  mast.position.set(0, 2.8, -2);
  world.add(mast);
  parts.rod(mast, m.brass, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1.2, 0), 0.04);
  for (const [dx, dz] of [[0.35, 0], [-0.35, 0], [0, 0.35], [0, -0.35]] as const) parts.rod(mast, m.brass, new THREE.Vector3(0, 0.55, 0), new THREE.Vector3(dx, 0.55, dz), 0.015);
  const worldVane = vane(parts, m, 0.7);
  worldVane.position.y = 1.2;
  mast.add(worldVane);
  const stand = plinth(world, 0.6, 1.1, m);
  stand.position.set(2.4, 0, 0.8);
  world.add(stand);
  const guessMast = new THREE.Group();
  guessMast.position.set(2.4, 1.1, 0.8);
  world.add(guessMast);
  parts.rod(guessMast, m.brass, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0.5, 0), 0.03);
  const guessVane = vane(parts, m, 0.4);
  guessVane.position.y = 0.5;
  guessMast.add(guessVane);
  parts.build();
  const halo = new THREE.Mesh(world.own(new THREE.TorusGeometry(0.55, 0.04, 8, 32).rotateX(Math.PI / 2)), world.own(flameMaterial(accent, 1.4)));
  halo.position.set(2.4, 1.62, 0.8);
  world.add(halo);
  const banners = [-3.6, 3.6].map(x => { const b = banner(world, accent, 2.6, m); b.position.set(x, 0, -3); world.add(b); return b; });
  const cloth = banners[0]!.children.find(child => child instanceof THREE.Mesh && 'oak' in child.material) as THREE.Mesh | undefined;
  const clothMaterial = cloth?.material as (THREE.Material & { oak: { setWind(a: number, f?: number): void } }) | undefined;
  clothMaterial?.oak.setWind(0.05 + 0.2 * gale, 1.4 + 2 * gale);
  const sentinel = world.own(buildSentinel({ name: 'hiddenmodeWatcher', tint: accent }));
  sentinel.place(0.6, 0, 1.4);
  sentinel.face(Math.atan2(1.8, -0.6));
  world.add(sentinel.group);
  const point = rig.points[0];
  if (point) { point.color.set(0xdfe8ff); point.position.set(0, 5, -1); }
  const extra = { land: t.land, lights: rig, worldVane, guessVane, halo, sentinel, gale, accent };
  return world.seal<WorldCore & typeof extra>(extra);
}

type VaneWorld = ReturnType<typeof buildWorld>;

class WeatherVaneScene extends StageScene<VaneWorld> {
  readonly post: ScenePost = { bloom: 0.34, fog: { from: ['afternoon', 'fog'], blend: [[['night', 'fog'], 0.35]] } };
  readonly controlHz = 5;
  protected readonly stageName = 'weatherVane';
  private view!: SceneView;
  private lastStep = -1;
  private matchAt = -Infinity;
  private lightningAt = -Infinity;
  private worldYaw = 0;
  private guessYaw = 0;
  private readonly ledger = { mode: 0, guess: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'WEATHER VANE',
    rows: [
      { key: 'wind', label: 'hidden wind', tint: 0xcfd8e2, value: () => (this.ledger.mode ? 'from the west' : 'from the east') },
      { key: 'guess', label: 'vane set to', tint: 0xdcf4ff, value: () => (this.ledger.guess ? 'west' : 'east') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): VaneWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-5, -0.5, -4], max: [5, 5, 4] }, [0.4, 1.8, -0.6], { distance: 10, azimuth: 0.45, polar: 1.1 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions { return { floor: [10, 8], agentPos: () => this.world!.sentinel.group.position }; }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, time }: Stage<VaneWorld>): void {
    const step = frame.state.step, mode = Math.round(frame.state.pose[HIDDENMODE_MODE]!) & 1, guess = Math.round(frame.state.pose[HIDDENMODE_ACTION]!) & 1;
    this.ledger.mode = mode; this.ledger.guess = guess; this.ledger.reward = frame.state.reward;
    if (step !== this.lastStep) {
      this.lastStep = step;
      if (frame.state.reward > 0) { this.matchAt = time; if (!frame.scrubbed) pool.burst(world.halo.position, world.accent, 0.35, time); }
    }
    for (const event of frame.events) if (event.type === 'cue') this.lightningAt = time;
    const k = frame.scrubbed ? 1 : 1 - Math.exp(-6 * frame.dtReal), gust = Math.sin(time * 2.3) * 0.08 * (0.3 + world.gale) + Math.sin(time * 7.1) * 0.03 * world.gale;
    this.worldYaw += (mode * Math.PI - this.worldYaw) * k;
    this.guessYaw += (guess * Math.PI - this.guessYaw) * k;
    world.worldVane.rotation.y = this.worldYaw + gust;
    world.guessVane.rotation.y = this.guessYaw + gust * 0.5;
    const match = envelope(time - this.matchAt, 0.9, 1.6), flash = envelope(time - this.lightningAt, 0.5, 9);
    (world.halo.material as THREE.MeshBasicMaterial).opacity = 0.15 + 0.8 * match;
    world.lights.sun.intensity = 1.1 + 6 * flash;
    world.lights.hemi.intensity = 0.7 + 2 * flash;
    const point = world.lights.points[0];
    if (point) point.intensity = 12 * flash;
    world.sentinel.idle(time);
    world.sentinel.glow(match);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(this.rig!, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.lastStep = -1; this.matchAt = this.lightningAt = -Infinity; }
}

export default ((() => new WeatherVaneScene()) satisfies SceneFactory);
