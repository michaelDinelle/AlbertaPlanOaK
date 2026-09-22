import * as THREE from 'three';
import { buildSentinel } from '@kit/cast/sentinel';
import { makeHeightField } from '@kit/heightfield';
import { makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { lamp, relief } from '@kit/props';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const palette = ctx.tokens.palettes.dusk, accent = ctx.tokens.accents[ctx.environment.family] ?? ctx.tokens.celebrationGold;
  const land = makeLandscape(scene, `${ctx.environment.family}Stub`, palette, 'dusk', { sky: { sunSize: 0.05, sunHdr: 7, sunGlow: 0.42, sunGlowPower: 20, topSpread: 0.62, bottomSpread: 0.3 }, light: { pointCount: 2, pointDistance: 4 }, fogExtent: 90, weather: { preset: 'pollen', count: 160, seed: ctx.seed, color: 0xfff1c8, size: 1.1, intensity: 0.8, bounds: { y: 1.2, halfX: 9, halfY: 1, halfZ: 9 } } });
  const { world, rig } = land;
  rig.sun.intensity = 1.9; rig.hemi.intensity = 0.55;
  rig.points.forEach((point, i) => { point.color.set(accent); point.intensity = 6; point.position.set(i === 0 ? -3.2 : 3.2, 1.7, -2.4); });
  rig.setShadowFocus({ x: 0, y: 0.5, z: 0 }, 9);
  const field = makeHeightField({ base: -0.05, court: { halfX: 4, halfZ: 4, rise: 0.25 }, octaves: [{ wavelength: 27, amplitude: 1.4, seed: ctx.seed }, { wavelength: 9, amplitude: 0.35, seed: ctx.seed ^ 0x7 }] });
  const terrain = world.own(world.time(makeSceneTerrain({ follow: false, name: `${ctx.environment.family}StubGround`, seed: ctx.seed, colorA: 0x5f6b45, colorB: 0x8f9a5e, segments: 112, radius: 420, height: field.glsl, detail: 'color *= 0.94 + 0.06 * landNoise(p.xz * 1.3);' }, field.heightAt, { roughness: 0.95, rimColor: palette.skyHorizon, rimStrength: 0.14, toonSteps: 4 })));
  world.add(terrain.mesh);
  const wall = relief(world, 6.5, 1.6, 0.7);
  wall.position.set(0, 0, -3.6);
  world.add(wall);
  const lamps = [-3.2, 3.2].map(x => { const l = lamp(world, accent, 1.5); l.group.position.set(x, 0, -2.4); world.add(l.group); l.lit(0.6); return l; });
  const sentinel = world.own(buildSentinel({ name: `${ctx.environment.family}StubSentinel`, tint: accent }));
  sentinel.place(0, 0, 0);
  world.add(sentinel.group);
  const extra = { land, sentinel, lamps, heightAt: field.heightAt };
  return world.seal<WorldCore & typeof extra>(extra);
}

type StubWorld = ReturnType<typeof buildWorld>;

class StubScene extends StageScene<StubWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'stub';
  private view!: SceneView;
  private glow = 0;

  protected buildWorld(ctx: SceneContext): StubWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    this.view = makeSceneView(ctx, { min: [-5, -0.5, -5], max: [5, 3, 5] }, [0, 0.9, -0.6], { distance: 11, azimuth: 0.42, polar: 1.15 });
    return this.view.options;
  }
  protected overlayOptions() { return {}; }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<StubWorld>): void {
    this.glow = Math.max(0, this.glow - frame.dtReal * 1.5, Math.min(1.5, frame.state.reward));
    world.sentinel.idle(time);
    world.sentinel.glow(this.glow);
    world.lamps.forEach((l, i) => l.lit(0.55 + 0.25 * Math.sin(time * 1.3 + i)));
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) pool.clear();
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.glow = 0; }
}

export default ((() => new StubScene()) satisfies SceneFactory);
