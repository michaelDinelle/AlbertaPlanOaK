import * as THREE from 'three';
import { parameter } from '../../data/envs';
import { cellWalk, gridHeading, walkCells } from '@kit/cast/cellwalker';
import { buildCourier } from '@kit/cast/courier';
import type { GroundTruthOptions } from '@kit/debug';
import { buildCellField, CELL, cellToWorldVec } from '@kit/grid';
import { makeHeightField, worldNoise } from '@kit/heightfield';
import { makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { blendTokens, envelope } from '@kit/mix';
import { makeAmbientParticles, makeMistPlanes } from '@kit/particles';
import { banner, lamp, propMaterials, relief } from '@kit/props';
import { hash2f } from '@kit/rng';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { excludeAny, excludeRect, scatterGrass, scatterRocks, scatterTrees } from '@kit/vegetation';
import { makeSceneView, type SceneView } from '@kit/view';
import { makeWater } from '@kit/water';
import type { WorldCore } from '@kit/world';
import { CLIFF_X, CLIFF_Y, isCliffCell } from '@sim/envs/cliff';
import { lastEvent } from '@sim/events';
import { formatPercent, formatSigned } from '../../format';

const GROUND = () => 0;
const DIR = ['\u2191 north', 'east \u2192', '\u2193 south', '\u2190 west'] as const;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, grid = ctx.geometry.grid;
  if (!grid || !grid.goal) throw new Error('Cliff grid is missing');
  const slip = parameter(env, 'slip'), accent = ctx.tokens.accents[env.family] ?? ctx.tokens.celebrationGold, palette = ctx.tokens.palettes.dusk;
  const halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2, edgeZ = -halfZ + CELL, seed = ctx.seed >>> 0;
  const land = makeLandscape(scene, 'canyonRim', palette, 'dusk', {
    sky: { sunDir: [-0.62, 0.3, -0.72], sunSize: 0.05, sunHdr: 7, sunGlow: 0.42, sunGlowPower: 20, topSpread: 0.62, bottomSpread: 0.3 },
    light: { pointCount: 2, pointDistance: 5 }, fogExtent: 420,
    weather: { preset: 'dust', count: 420, seed: ctx.seed, color: 0xffe0b8, size: 0.6, intensity: 0.8, opacity: 0.45, swirl: 1.1, rise: 0.01, bounds: { y: 1.2, z: edgeZ - 3, halfX: halfX + 6, halfY: 1.6, halfZ: 6 } },
  });
  const { world, rig, sky } = land;
  rig.sun.color.set(0xffb877); rig.sun.intensity = 2.2;
  rig.hemi.color.set(0x8e9fd6); rig.hemi.groundColor.set(0x6b5040); rig.hemi.intensity = 0.8;
  rig.setShadowFocus({ x: 0, y: 0.3, z: 0.5 }, halfX + 2.5);
  const plateau = makeHeightField({ base: -0.05, court: { halfX: halfX + 0.5, halfZ: halfZ + 0.5, rise: 0.22 }, octaves: [{ wavelength: 27, amplitude: 1.1, seed: ctx.seed }, { wavelength: 8, amplitude: 0.28, seed: ctx.seed ^ 0x11 }] });
  const rimSeed = (seed ^ 0x77) >>> 0, bedSeed = (seed ^ 0x3c1) >>> 0;
  const canyonAt = (x: number, z: number, top: number) => {
    const ragged = (worldNoise(x / 6, z / 6, 4096, rimSeed) - 0.5) * 0.9 * smooth(halfX, halfX + 3, Math.abs(x));
    const over = smooth(-1.1, 0, z - edgeZ + ragged);
    const canyon = -7.5 + (worldNoise(x / 13, z / 13, 4096, bedSeed) - 0.5) * 2.2 + (1 - smooth(-32, -14, z - edgeZ)) * 14;
    return canyon + (top - canyon) * over;
  };
  const terrain = world.own(world.time(makeSceneTerrain({ follow: false, name: 'canyonRimGround', seed: ctx.seed, colorA: 0x8c6a4c, colorB: 0xc2a06e, segments: 128, radius: 420,
    height: `${plateau.glsl.replace('return h;', '')}
float ragged = (worldNoise(p / 6.0, 4096.0, ${rimSeed}u) - 0.5) * 0.9 * smoothstep(${halfX.toFixed(3)}, ${(halfX + 3).toFixed(3)}, abs(p.x));
float over = smoothstep(-1.1, 0.0, p.y - ${edgeZ.toFixed(3)} + ragged);
float canyon = -7.5 + (worldNoise(p / 13.0, 4096.0, ${bedSeed}u) - 0.5) * 2.2 + (1.0 - smoothstep(-32.0, -14.0, p.y - ${edgeZ.toFixed(3)})) * 14.0;
return mix(canyon, h, over);`,
    detail: 'float strata = 0.5 + 0.5 * sin(p.y * 6.0 + landNoise(p.xz * 0.5) * 3.0); color = mix(color, vec3(0.55, 0.32, 0.24), 0.4 * strata * (1.0 - smoothstep(-2.5, -0.3, p.y))); color *= 0.93 + 0.07 * landNoise(p.xz * 1.9);' },
  (x, z) => canyonAt(x, z, plateau.heightAt(x, z)), { roughness: 0.95, rimColor: palette.sun, rimStrength: 0.2, toonSteps: 4 })));
  world.add(terrain.mesh);
  const flag = new THREE.Color(0xc9a27a), flagDark = new THREE.Color(0x9a7452), ice = new THREE.Color(0xd8ecff), tint = new THREE.Color();
  const cells = world.own(world.time(buildCellField(grid, {
    floorColor: 0xffffff, wallColor: 0xffffff, floorFilter: (x, y) => !isCliffCell(grid, x, y), floorThickness: 0.26, floorInset: 0.03, jitter: 0.08, seed: ctx.seed,
    floorTint: (x, y) => { tint.copy(flag).lerp(flagDark, hash2f(x, y, ctx.seed)); return slip > 0 && y <= 1 ? tint.lerp(ice, 0.8 - 0.3 * y) : tint; },
    rimColor: palette.sun, rimStrength: 0.3, name: 'rimFlagstones',
  })));
  if (slip > 0) { cells.floorMaterial.roughness = 0.22; cells.floorMaterial.metalness = 0.15; }
  world.add(cells.group);
  const water = world.time(world.own(makeWater({ size: [420, 70], palette, level: -6.1, amplitude: 0.05, waveScale: 1.5, speed: 1.2, sunDir: sky.sunDirection, sky: 0xb9a3c4, fresnel: 0.35, caustics: 0.15, name: 'canyonRiver' })));
  water.setColors(0x3f8a8a, 0x1d4f5e);
  water.uniforms.uSpecular.value = 0.25;
  water.mesh.position.set(0, -6.1, edgeZ - 22);
  world.add(water.mesh);
  const mist = world.time(world.own(makeMistPlanes({ size: [48, 30], color: blendTokens(palette.fog, palette.hemiSky, 0.4), seed: ctx.seed, layers: 4, baseY: -4.2, spread: 2.6, opacity: 0.16, speed: 0.01, scale: 2.2, nearFade: 2, name: 'canyonMist' })));
  mist.group.position.set(0, 0, edgeZ - 14);
  world.add(mist.group);
  const m = propMaterials(world);
  const wall = relief(world, grid.w * CELL + 1.6, 1.25, 0.8, m);
  wall.position.set(0, 0, halfZ + 0.6);
  world.add(wall);
  const goalPos = cellToWorldVec(grid.goal.x, grid.goal.y, grid, new THREE.Vector3()), startPos = cellToWorldVec(0, 0, grid, new THREE.Vector3());
  const goalBanner = banner(world, accent, 2, m);
  goalBanner.position.set(goalPos.x + 0.38, 0, goalPos.z - 0.32);
  world.add(goalBanner);
  const goalLamp = lamp(world, ctx.tokens.celebrationGold, 1.2, m);
  goalLamp.group.position.set(goalPos.x + 0.38, 0, goalPos.z + 0.36);
  goalLamp.lit(0.5);
  world.add(goalLamp.group);
  const startLamp = lamp(world, palette.accentLight, 1.2, m);
  startLamp.group.position.set(startPos.x - 0.38, 0, startPos.z - 0.32);
  startLamp.lit(0.5);
  world.add(startLamp.group);
  const court = excludeRect(0, 0, halfX + 0.9, halfZ + 0.9, 0.2), heightAt = terrain.sampleHeight;
  const trees = scatterTrees({ archetypes: ['conifer'], count: 90, area: { z: halfZ + 7, halfX: 24, halfZ: 5 }, seed: ctx.seed ^ 0x5a, exclude: court, heightAt, maxSlope: 0.7, color: 0x4f6f4a, scale: [0.7, 1.3], stretch: [1, 1.5], hueJitter: 0.2, rimColor: palette.sun, rimStrength: 0.32, castShadow: true, name: 'rimConifers' });
  world.add(trees.group); world.time(world.own(trees));
  const grass = scatterGrass({ count: 1300, area: { z: halfZ + 5, halfX: 24, halfZ: 9 }, seed: ctx.seed ^ 0x7f, exclude: court, heightAt, maxSlope: 0.8, color: 0xa89a5c, scale: [0.35, 0.75], stretch: [0.8, 1.5], hueJitter: 0.2, rimColor: palette.sun, rimStrength: 0.36, name: 'rimGrass' });
  world.add(grass.group); world.time(world.own(grass));
  const rocks = scatterRocks({ count: 110, area: { z: edgeZ + 0.35, halfX: 26, halfZ: 0.7 }, seed: ctx.seed ^ 0x2b, exclude: excludeAny(court, (x, z) => canyonAt(x, z, 0) < -0.4), heightAt, color: 0x9a7b62, scale: [0.12, 0.42], stretch: [0.6, 1.1], hueJitter: 0.2, rimColor: palette.sun, rimStrength: 0.28, castShadow: true, name: 'rimScree' });
  world.add(rocks.group); world.time(world.own(rocks));
  if (slip > 0) {
    const glitter = world.size(world.time(world.own(makeAmbientParticles({ preset: 'snow', count: 220, seed: ctx.seed ^ 0x9c, color: 0xe6f4ff, size: 0.9, intensity: 1.2, opacity: 0.6, rise: -0.03, swirl: 0.05, bounds: { y: 0.35, z: edgeZ + CELL, halfX, halfY: 0.3, halfZ: CELL }, name: 'rimIceGlitter' }))));
    world.add(glitter.points);
  }
  const robot = world.own(buildCourier({ name: 'rimCourier', tint: accent, scale: 1.25, overgrowth: 0.35 }));
  world.add(robot.group);
  const extra = { land, terrain, grid, slip, accent, robot, goalLamp, goalPos, cliffDrop: edgeZ - 1 };
  return world.seal<WorldCore & typeof extra>(extra);
}

const smooth = (a: number, b: number, v: number): number => { const t = Math.min(1, Math.max(0, (v - a) / (b - a))); return t * t * (3 - 2 * t); };

type RimWorld = ReturnType<typeof buildWorld>;

class CanyonRimScene extends StageScene<RimWorld> {
  readonly post: ScenePost = { bloom: 0.44, fog: { from: ['dusk', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'canyonRim';
  private view!: SceneView;
  private readonly walk = cellWalk();
  private readonly from = { x: 0, y: 0, heading: 1 };
  private readonly to = { x: 0, y: 0, heading: 1 };
  private readonly cell = { x: 0, y: 0 };
  private readonly last = new THREE.Vector3();
  private readonly burstAt = new THREE.Vector3();
  private heading = 1;
  private goalAt = -Infinity;
  private fallAt = -Infinity;
  private readonly ledger = { x: 0, y: 0, action: 0, reward: 0, last: '\u2014' };
  override legend: LegendSpec = {
    title: 'CLIFF',
    rows: [
      { key: 'at', label: 'position', tint: 0xd97b6c, value: () => `column ${this.ledger.x + 1} \u00b7 row ${this.ledger.y + 1}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => DIR[this.ledger.action]! },
      { key: 'slip', label: 'slip', tint: 0xd8ecff, value: () => formatPercent(this.world!.slip, 0) },
      { key: 'last', label: 'last run', tint: 0xffd166, value: () => this.ledger.last },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): RimWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const { grid, cliffDrop } = this.world!, halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2;
    this.view = makeSceneView(ctx, { min: [-halfX - 2, -8, cliffDrop - 10], max: [halfX + 2, 2.5, halfZ + 3] }, [0, -0.4, -1.2], { distance: halfX * 2.0, azimuth: 1.9, polar: 1.12 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return { agentCell: state => { this.cell.x = Math.round(state.pose[CLIFF_X]!); this.cell.y = Math.round(state.pose[CLIFF_Y]!); return this.cell; }, goalCell: () => this.world!.grid.goal! };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<RimWorld>): void {
    const walk = walkCells(frame, CLIFF_X, CLIFF_Y, world.grid, this.walk), ox = (world.grid.w - 1) / 2, oz = (world.grid.h - 1) / 2;
    this.last.copy(world.robot.group.position);
    if (walk.moving) this.heading = gridHeading(walk.toX - walk.fromX, walk.toY - walk.fromY, this.heading);
    this.from.x = (walk.fromX - ox) * CELL; this.from.y = (walk.fromY - oz) * CELL; this.from.heading = this.heading;
    this.to.x = (walk.toX - ox) * CELL; this.to.y = (walk.toY - oz) * CELL; this.to.heading = this.heading;
    world.robot.drive(this.from, this.to, frame.position - frame.state.step, GROUND, frame.scrubbed || frame.state.reset);
    if (world.slip > 0 && walk.moving) world.robot.group.rotation.y += Math.sin(time * 11) * world.slip * 0.9;
    const step = frame.state.step, fellAt = lastEvent(frame.eventLog, 'teleport', step), arrivedAt = lastEvent(frame.eventLog, 'goal', step);
    this.ledger.x = walk.toX; this.ledger.y = walk.toY; this.ledger.action = Math.trunc(frame.frameAt(step).actions[0]!) & 3; this.ledger.reward = frame.state.reward;
    this.ledger.last = fellAt === undefined && arrivedAt === undefined ? '\u2014' : (arrivedAt ?? -1) >= (fellAt ?? -1) ? 'reached the gate' : 'fell';
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.goalAt = -Infinity; this.fallAt = -Infinity; }
    let goal = false, teleport = false;
    for (const event of frame.events) { if (event.type === 'goal') goal = true; else if (event.type === 'teleport') teleport = true; }
    if (goal) {
      this.burstAt.copy(world.goalPos); this.burstAt.y = 1.3;
      pool.burst(this.burstAt, world.accent, 0.7, time);
      pool.ripple(world.goalPos, ctx.tokens.celebrationGold, 0.4, time);
      pool.pulse.fire(0.5);
      this.goalAt = time;
    } else if (teleport) {
      this.burstAt.copy(this.last); this.burstAt.y -= 0.6;
      pool.sigh(0.6, this.burstAt, time);
      this.fallAt = time;
    }
    const flare = envelope(time - this.goalAt, 1.4, 2), fall = envelope(time - this.fallAt, 1, 2.5);
    world.goalLamp.lit(0.5 + 1.6 * flare);
    const follow = world.land.rig.points[0], glow = world.land.rig.points[1], at = world.robot.group.position;
    if (follow) { follow.color.set(world.accent); follow.position.set(at.x, 0.8, at.z); follow.intensity = 0.3 + 0.6 * fall; }
    if (glow) { glow.color.set(ctx.tokens.celebrationGold); glow.position.set(world.goalPos.x, 1.1, world.goalPos.z); glow.intensity = 2.5 * flare; }
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera, world.terrain);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.goalAt = -Infinity; this.fallAt = -Infinity; }
}

export default ((() => new CanyonRimScene()) satisfies SceneFactory);
