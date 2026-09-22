import * as THREE from 'three';
import { parameter } from '../../data/envs';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec } from '@kit/grid';
import { makeHeightField, worldNoise } from '@kit/heightfield';
import { fieldstoneGeometry, makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { oakMaterial } from '@kit/materials';
import { blendTokens, envelope } from '@kit/mix';
import { makeAmbientParticles, makeMistPlanes } from '@kit/particles';
import { lamp, propMaterials } from '@kit/props';
import { hash2f } from '@kit/rng';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { plantInstances, scatterGrass, scatterRocks, scatterTrees } from '@kit/vegetation';
import { makeSceneView, type SceneView } from '@kit/view';
import { makeWater } from '@kit/water';
import { Wisp } from '@kit/wisp';
import type { WorldCore } from '@kit/world';
import { RIVERSWIM_POS } from '@sim/envs/riverswim';
import { lastEvent } from '@sim/events';
import { formatCount, formatSigned } from '../../format';

const smooth = (a: number, b: number, v: number): number => { const t = Math.min(1, Math.max(0, (v - a) / (b - a))); return t * t * (3 - 2 * t); };

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, grid = ctx.geometry.grid;
  if (!grid || grid.h !== 1 || grid.w !== parameter(env, 'states')) throw new Error('RiverSwim grid is missing');
  const states = grid.w, span = (states - 1) * CELL, half = span / 2, accent = ctx.tokens.accents[env.family] ?? ctx.tokens.celebrationGold, palette = ctx.tokens.palettes.morning, seed = ctx.seed >>> 0;
  const land = makeLandscape(scene, 'riverFord', palette, 'morning', {
    sky: { sunDir: [0.35, 0.5, 0.79], sunSize: 0.045, sunHdr: 5.5, sunGlow: 0.34, sunGlowPower: 28, topSpread: 0.62, bottomSpread: 0.4 },
    light: { pointCount: 2, pointDistance: 6 }, fogExtent: 180,
  });
  const { world, rig, sky } = land;
  rig.sun.color.set(0xfff3d6); rig.sun.intensity = 2.4;
  rig.hemi.color.set(0xc8e3f5); rig.hemi.groundColor.set(0x6f8c5a); rig.hemi.intensity = 1;
  rig.setShadowFocus({ x: 0, y: 0, z: 0 }, half + 4);
  const banks = makeHeightField({ base: 0.15, octaves: [{ wavelength: 23, amplitude: 0.9, seed: ctx.seed }, { wavelength: 7, amplitude: 0.25, seed: ctx.seed ^ 0x33 }] });
  const edgeSeed = (seed ^ 0x5d1) >>> 0, bedSeed = (seed ^ 0x2a7) >>> 0;
  const heightAt = (x: number, z: number) => {
    const channel = smooth(2.4, 5.2, Math.abs(z) + (worldNoise(x / 9, z / 9, 4096, edgeSeed) - 0.5) * 1.4);
    const bed = -1.7 + (worldNoise(x / 5, z / 5, 4096, bedSeed) - 0.5) * 0.5;
    return bed + (banks.heightAt(x, z) - bed) * channel;
  };
  const terrain = world.own(world.time(makeSceneTerrain({ follow: false, name: 'fordBanks', seed: ctx.seed, colorA: 0x8d9a6c, colorB: 0xb9b58a, segments: 128, radius: 420,
    height: `${banks.glsl.replace('return h;', '')}
float channel = smoothstep(2.4, 5.2, abs(p.y) + (worldNoise(p / 9.0, 4096.0, ${edgeSeed}u) - 0.5) * 1.4);
float bed = -1.7 + (worldNoise(p / 5.0, 4096.0, ${bedSeed}u) - 0.5) * 0.5;
return mix(bed, h, channel);`,
    detail: 'color = mix(color, vec3(0.42, 0.5, 0.45), 1.0 - smoothstep(-0.9, 0.0, p.y)); color *= 0.94 + 0.06 * landNoise(p.xz * 1.6);' }, heightAt, { roughness: 0.95, rimColor: palette.sun, rimStrength: 0.18, toonSteps: 4 })));
  world.add(terrain.mesh);
  const water = world.time(world.own(makeWater({ size: [span + 90, 11], palette, level: -0.42, amplitude: 0.05, waveScale: 1.6, speed: 1.4, sunDir: sky.sunDirection, fresnel: 0.35, caustics: 0.25, name: 'fordWater' })));
  water.setColors(0x62b3b6, 0x1f6a78);
  water.uniforms.uSpecular.value = 0.3;
  world.add(water.mesh);
  const stones = plantInstances(world, 'steppingStones', fieldstoneGeometry(), oakMaterial({ vertexColors: true, roughness: 0.95, rimColor: palette.sun, rimStrength: 0.3, toonSteps: 4, name: 'steppingStone' }), states + 1);
  for (let i = 0; i < states; i++) {
    stones.transform.position.set((i - (states - 1) / 2) * CELL, -0.34, Math.sin(i * 1.7) * 0.16);
    stones.transform.rotation.set(0, hash2f(i, 3, ctx.seed) * Math.PI * 2, 0);
    stones.transform.scale.setScalar(1.05 + hash2f(i, 7, ctx.seed) * 0.25);
    stones.write(i);
  }
  stones.transform.position.set(half + 1.5, -0.55, 0);
  stones.transform.rotation.set(0, 0.6, 0);
  stones.transform.scale.set(2.4, 2.2, 2.4);
  stones.write(states);
  stones.commit(states + 1);
  stones.mesh.castShadow = stones.mesh.receiveShadow = true;
  const m = propMaterials(world);
  const shrine = lamp(world, accent, 1.1, m);
  shrine.group.position.set(half + 1.5, 0.45, 0);
  shrine.lit(0.6);
  world.add(shrine.group);
  const foam = world.size(world.time(world.own(makeAmbientParticles({ preset: 'bubbles', count: 320, seed: ctx.seed ^ 0x1f, color: 0xf4fbff, size: 0.75, intensity: 1.4, rise: 1.6, swirl: 0.12, bounds: { halfX: 0.08, halfY: half + 9, halfZ: 2.2 }, name: 'fordFoam' }))));
  foam.points.rotation.z = Math.PI / 2;
  foam.points.position.set(0, -0.36, 0);
  world.add(foam.points);
  const mist = world.time(world.own(makeMistPlanes({ size: [span + 30, 12], color: blendTokens(palette.fog, 0xffffff, 0.3), seed: ctx.seed, layers: 3, baseY: 0.05, spread: 1.1, opacity: 0.08, speed: 0.009, scale: 2.6, nearFade: 2, name: 'fordMist' })));
  world.add(mist.group);
  const channel = (x: number, z: number) => Math.abs(z) < 3.2 || (Math.abs(z) < 4.2 && Math.abs(x) < half + 3);
  const grass = scatterGrass({ count: 1600, area: { halfX: half + 22, halfZ: 14 }, seed: ctx.seed ^ 0x61, exclude: channel, heightAt, maxSlope: 0.8, color: 0x9fb56d, scale: [0.35, 0.75], stretch: [0.8, 1.5], hueJitter: 0.2, rimColor: palette.sun, rimStrength: 0.34, name: 'fordGrass' });
  world.add(grass.group); world.time(world.own(grass));
  const trees = scatterTrees({ archetypes: ['broadleaf'], count: 70, area: { halfX: half + 22, halfZ: 14 }, seed: ctx.seed ^ 0x5a, exclude: (x, z) => Math.abs(z) < 5.5, heightAt, maxSlope: 0.7, color: 0x6f9a52, scale: [0.8, 1.4], stretch: [1, 1.4], hueJitter: 0.2, rimColor: palette.sun, rimStrength: 0.3, castShadow: true, name: 'fordAspens' });
  world.add(trees.group); world.time(world.own(trees));
  const rocks = scatterRocks({ count: 140, area: { halfX: half + 20, halfZ: 6 }, seed: ctx.seed ^ 0x2b, exclude: (x, z) => Math.abs(z) < 1.4 && Math.abs(x) < half + 3, heightAt, color: 0xa8a89c, scale: [0.15, 0.5], stretch: [0.5, 1.1], hueJitter: 0.2, rimColor: palette.sun, rimStrength: 0.28, castShadow: true, name: 'fordBoulders' });
  world.add(rocks.group); world.time(world.own(rocks));
  const wisp = world.own(new Wisp({ color: accent, radius: 0.3, height: 0.8, skirt: true, ribbon: true, silhouette: false, hop: { duration: 0.14, height: 0.28 }, name: 'fordWisp' }, grid));
  world.add(wisp.group);
  const extra = { land, terrain, grid, states, half, accent, wisp, shrine, water, shrinePos: new THREE.Vector3(half + 1.5, 1.7, 0) };
  return world.seal<WorldCore & typeof extra>(extra);
}

type FordWorld = ReturnType<typeof buildWorld>;

class RiverFordScene extends StageScene<FordWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 6;
  protected readonly stageName = 'riverFord';
  private view!: SceneView;
  private readonly cell = { x: 0, y: 0 };
  private readonly splashAt = new THREE.Vector3();
  private lastStep = -1;
  private goalAt = -Infinity;
  private readonly ledger = { pos: 0, upstream: false, reward: 0, caught: '\u2014' };
  override legend: LegendSpec = {
    title: 'RIVER',
    rows: [
      { key: 'at', label: 'stone', tint: 0x6fb8e8, value: () => `${this.ledger.pos + 1} of ${this.world!.states}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => (this.ledger.upstream ? 'swim upstream \u2192' : '\u2190 drift down') },
      { key: 'caught', label: 'last catch', tint: 0xffd166, value: () => this.ledger.caught },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): FordWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const { half } = this.world!;
    this.view = makeSceneView(ctx, { min: [-half - 4, -2, -6], max: [half + 4, 2, 6] }, [0.6, 0.1, 0], { distance: 11, azimuth: 0.25, polar: 1.05 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return { agentCell: state => { this.cell.x = Math.round(state.pose[RIVERSWIM_POS]!); return this.cell; }, goalCell: () => this.world!.grid.goal! };
  }
  protected start(_ctx: SceneContext, world: FordWorld): void { world.wisp.setCell(0, 0, world.grid); }
  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<FordWorld>): void {
    const step = frame.state.step, pos = Math.min(world.states - 1, Math.max(0, Math.round(frame.state.pose[RIVERSWIM_POS]!)));
    this.cell.x = pos;
    const caughtAt = lastEvent(frame.eventLog, 'goal', step);
    this.ledger.pos = pos; this.ledger.upstream = (Math.trunc(frame.frameAt(step).actions[0]!) & 1) === 1; this.ledger.reward = frame.state.reward; this.ledger.caught = caughtAt === undefined ? '\u2014' : `${formatCount(step - caughtAt)} steps ago`;
    if (frame.scrubbed || Math.abs(pos - world.wisp.cellX) > 1) { world.wisp.setCell(pos, 0, world.grid); world.wisp.snap(); }
    else if (pos !== world.wisp.cellX) world.wisp.hopTo(this.cell, time);
    world.wisp.update(frame.dtReal, time);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.goalAt = -Infinity; this.lastStep = step; }
    if (step !== this.lastStep && step > 0) {
      const upstream = (Math.trunc(frame.frameAt(step).actions[0]!) & 1) === 1, before = Math.round(frame.poseAt(step - 1)[RIVERSWIM_POS]!);
      cellToWorldVec(before, 0, world.grid, this.splashAt);
      if (upstream && pos < before) { pool.sigh(0.3, this.splashAt, time); pool.ripple(this.splashAt, 0xbfe8ff, 0.35, time); }
      else if (upstream && pos === before && pos < world.states - 1) pool.ripple(this.splashAt, 0xbfe8ff, 0.18, time);
    }
    this.lastStep = step;
    for (const event of frame.events) if (event.type === 'goal') {
      pool.burst(world.shrinePos, ctx.tokens.celebrationGold, 0.7, time);
      pool.pulse.fire(0.5);
      world.wisp.flare(1);
      this.goalAt = time;
    }
    const flare = envelope(time - this.goalAt, 1.4, 2);
    world.shrine.lit(0.6 + 1.8 * flare);
    const follow = world.land.rig.points[0], glow = world.land.rig.points[1];
    if (follow) { follow.color.set(world.accent); follow.position.copy(world.wisp.position).setY(0.7); follow.intensity = 0.3; }
    if (glow) { glow.color.set(ctx.tokens.celebrationGold); glow.position.copy(world.shrinePos); glow.intensity = 2.5 * flare; }
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal, [world.wisp.position.x * 0.8, 0.1, 0]), this.camera, world.terrain);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.goalAt = -Infinity; this.lastStep = -1; }
}

export default ((() => new RiverFordScene()) satisfies SceneFactory);
