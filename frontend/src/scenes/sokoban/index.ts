import * as THREE from 'three';
import { cellWalk, walkCells } from '@kit/cast/cellwalker';
import { buildSentinel } from '@kit/cast/sentinel';
import type { GroundTruthOptions } from '@kit/debug';
import { buildCellField, CELL, cellToWorldVec } from '@kit/grid';
import { makeHeightField } from '@kit/heightfield';
import { makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { oakMaterial } from '@kit/materials';
import { envelope } from '@kit/mix';
import { doorway, lamp, propMaterials, relief } from '@kit/props';
import { hash2f } from '@kit/rng';
import { modelParts } from '@kit/robot';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { excludeRect, scatterRocks, scatterTrees } from '@kit/vegetation';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { SOKOBAN_BOX_X, SOKOBAN_BOX_Y, SOKOBAN_TARGET_X, SOKOBAN_TARGET_Y, SOKOBAN_X, SOKOBAN_Y } from '@sim/envs/sokoban';
import { constant } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const DIR = ['\u2191 north', 'east \u2192', '\u2193 south', '\u2190 west'] as const;


function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, grid = ctx.geometry.grid;
  if (!grid) throw new Error('Sokoban grid is missing');
  const accent = ctx.tokens.accents[env.family] ?? ctx.tokens.celebrationGold, palette = ctx.tokens.palettes.afternoon;
  const halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2, span = grid.w * CELL + 1.6;
  const land = makeLandscape(scene, 'storehouseCourt', palette, 'afternoon', {
    sky: { sunDir: [0.42, 0.36, 0.83], sunSize: 0.04, sunHdr: 5, sunGlow: 0.3, sunGlowPower: 26, topSpread: 0.55, bottomSpread: 0.4 },
    light: { pointCount: 2, pointDistance: 6 }, fogExtent: 140,
    weather: { preset: 'snow', count: 700, seed: ctx.seed, size: 1.1, bounds: { y: 1.6, halfX: halfX + 5, halfY: 1.6, halfZ: halfZ + 5 } },
  });
  const { world, rig, sky, fog } = land;
  sky.setColors(0x7fa6cf, 0xe4e9ef, 0xc9d3dc);
  fog.color.set(0xd6dde6);
  rig.sun.color.set(0xfff1de); rig.sun.intensity = 2.2;
  rig.hemi.color.set(0xc4d8ee); rig.hemi.groundColor.set(0x6f7d8a); rig.hemi.intensity = 1;
  rig.setShadowFocus({ x: 0, y: 0.4, z: 0 }, halfX + 3);
  const field = makeHeightField({ base: -0.06, court: { halfX: halfX + 1.2, halfZ: halfZ + 1.2, rise: 0.3 }, octaves: [{ wavelength: 25, amplitude: 1.1, seed: ctx.seed }, { wavelength: 7, amplitude: 0.22, seed: ctx.seed ^ 0x45 }] });
  const terrain = world.own(world.time(makeSceneTerrain({ follow: false, name: 'snowfield', seed: ctx.seed, colorA: 0xe4ebf4, colorB: 0xf8fafd, segments: 112, radius: 420, height: field.glsl, detail: 'color *= 0.95 + 0.05 * landNoise(p.xz * 2.3); color = mix(color, vec3(0.78, 0.84, 0.92), 0.4 * smoothstep(0.5, 0.8, landNoise(p.xz * 0.16)));' }, field.heightAt, { roughness: 1, rimColor: palette.skyHorizon, rimStrength: 0.2, toonSteps: 4 })));
  world.add(terrain.mesh);
  const flagA = new THREE.Color(0xbfb6a5), flagB = new THREE.Color(0xe6e2da), tint = new THREE.Color();
  const cells = world.own(world.time(buildCellField(grid, { floorColor: 0xffffff, wallColor: 0xffffff, floorThickness: 0.14, floorInset: 0.035, jitter: 0.08, seed: ctx.seed, floorTint: (x, y) => tint.copy(flagA).lerp(flagB, hash2f(x, y, ctx.seed)), rimColor: palette.skyHorizon, rimStrength: 0.25, name: 'storehouseFlagstones' })));
  world.add(cells.group);
  const m = propMaterials(world);
  const back = relief(world, span, 1.7, 0.15, m);
  back.position.set(0, 0, -halfZ - 0.5);
  world.add(back);
  for (const side of [-1, 1]) {
    const wall = relief(world, span, 1.7, 0.15, m);
    wall.rotation.y = Math.PI / 2;
    wall.position.set(side * (halfX + 0.5), 0, 0);
    world.add(wall);
  }
  const door = doorway(world, 1.3, 1.5, m), doorSpan = 1.3 + 0.64, wing = (span - doorSpan) / 2;
  door.position.set(0, 0, halfZ + 0.5);
  world.add(door);
  for (const side of [-1, 1]) {
    const wall = relief(world, wing, 1.7, 0.15, m);
    wall.position.set(side * (doorSpan / 2 + wing / 2), 0, halfZ + 0.5);
    world.add(wall);
  }
  const lamps = [-1, 1].map(side => {
    const post = lamp(world, 0xffd9a0, 1.5, m);
    post.group.position.set(side * 1.35, 0, halfZ + 1);
    post.lit(0.4);
    world.add(post.group);
    return post;
  });
  const parts = modelParts(world), target = new THREE.Group(), targetGlass = m.glass(accent, 0.5);
  parts.disc(target, m.brass, 0, 0.03, 0, 0.36, 0.06, 24);
  parts.disc(target, targetGlass, 0, 0.065, 0, 0.24, 0.02, 24);
  world.add(target);
  const crate = new THREE.Group(), wood = world.own(oakMaterial({ color: 0x8b6a45, roughness: 0.85, rimColor: palette.sun, rimStrength: 0.3, toonSteps: 3, name: 'crateWood' })), snow = world.own(oakMaterial({ color: 0xf4f7fb, roughness: 1, rimColor: palette.skyHorizon, rimStrength: 0.3, toonSteps: 3, name: 'crateSnow' }));
  parts.box(crate, wood, 0, 0.36, 0, 0.72, 0.72, 0.72, 0.04);
  for (const x of [-0.36, 0.36]) for (const z of [-0.36, 0.36]) parts.rod(crate, m.brass, new THREE.Vector3(x, 0.01, z), new THREE.Vector3(x, 0.71, z), 0.022);
  for (const y of [0.02, 0.7]) for (const side of [-0.36, 0.36]) {
    parts.rod(crate, m.brass, new THREE.Vector3(-0.36, y, side), new THREE.Vector3(0.36, y, side), 0.022);
    parts.rod(crate, m.brass, new THREE.Vector3(side, y, -0.36), new THREE.Vector3(side, y, 0.36), 0.022);
  }
  parts.box(crate, snow, 0, 0.745, 0, 0.6, 0.05, 0.6, 0.02);
  world.add(crate);
  parts.build();
  const court = excludeRect(0, 0, halfX + 1.4, halfZ + 1.6, 0.3);
  const trees = scatterTrees({ archetypes: ['conifer', 'bare'], count: 140, area: { halfX: halfX + 20, halfZ: halfZ + 18 }, seed: ctx.seed ^ 0x5c, exclude: court, heightAt: field.heightAt, color: 0x9fb4ad, scale: [0.7, 1.4], stretch: [1, 1.5], hueJitter: 0.15, rimColor: palette.skyHorizon, rimStrength: 0.35, castShadow: true, name: 'winterSpruce' });
  world.add(trees.group); world.time(world.own(trees));
  const rocks = scatterRocks({ count: 90, area: { halfX: halfX + 18, halfZ: halfZ + 16 }, seed: ctx.seed ^ 0x9e, exclude: court, heightAt: field.heightAt, color: 0xd3d9e2, scale: [0.15, 0.5], stretch: [0.5, 1], rimColor: palette.skyHorizon, rimStrength: 0.3, castShadow: true, name: 'snowBoulders' });
  world.add(rocks.group); world.time(world.own(rocks));
  const sentinel = world.own(buildSentinel({ name: 'storehouseSentinel', tint: accent, scale: 0.9 }));
  world.add(sentinel.group);
  const extra = { land, terrain, grid, accent, sentinel, crate, target, targetGlass, lamps };
  return world.seal<WorldCore & typeof extra>(extra);
}

type CourtWorld = ReturnType<typeof buildWorld>;

class StorehouseCourtScene extends StageScene<CourtWorld> {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['afternoon', 'fog'], blend: [[['noon', 'skyHorizon'], 0.4]] } };
  readonly controlHz = 8;
  protected readonly stageName = 'storehouseCourt';
  private view!: SceneView;
  private readonly walk = cellWalk();
  private readonly push = cellWalk();
  private readonly cell = { x: 0, y: 0 };
  private readonly goal = { x: 0, y: 0 };
  private readonly burstAt = new THREE.Vector3();
  private goalAt = -Infinity;
  private readonly ledger = { keeper: '', crate: '', target: '', action: 0, patience: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'STOREHOUSE',
    rows: [
      { key: 'keeper', label: 'keeper', tint: 0xc9a27a, value: () => this.ledger.keeper },
      { key: 'crate', label: 'crate', tint: 0x8b6a45, value: () => this.ledger.crate },
      { key: 'target', label: 'target', tint: 0xffd166, value: () => this.ledger.target },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => DIR[this.ledger.action]! },
      { key: 'patience', label: 'patience', tint: 0xdcf4ff, value: () => `${formatCount(this.ledger.patience)} steps` },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): CourtWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const { grid } = this.world!, halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2;
    this.view = makeSceneView(ctx, { min: [-halfX - 1.5, -0.5, -halfZ - 1.5], max: [halfX + 1.5, 2.5, halfZ + 2] }, [0, 0.4, 0.2], { distance: Math.max(halfX, halfZ) * 3.0, azimuth: 0.38, polar: 0.95 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      agentCell: state => { this.cell.x = Math.round(state.pose[SOKOBAN_X]!); this.cell.y = Math.round(state.pose[SOKOBAN_Y]!); return this.cell; },
      goalCell: state => { this.goal.x = Math.round(state.pose[SOKOBAN_TARGET_X]!); this.goal.y = Math.round(state.pose[SOKOBAN_TARGET_Y]!); return this.goal; },
    };
  }
  protected start(): void {}
  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<CourtWorld>): void {
    const walk = walkCells(frame, SOKOBAN_X, SOKOBAN_Y, world.grid, this.walk), push = walkCells(frame, SOKOBAN_BOX_X, SOKOBAN_BOX_Y, world.grid, this.push);
    const step = frame.state.step, raw = frame.frameAt(step), tx = Math.round(frame.state.pose[SOKOBAN_TARGET_X]!), ty = Math.round(frame.state.pose[SOKOBAN_TARGET_Y]!);
    this.ledger.keeper = `${walk.toX + 1}, ${walk.toY + 1}`; this.ledger.crate = `${push.toX + 1}, ${push.toY + 1}`; this.ledger.target = `${tx + 1}, ${ty + 1}`;
    this.ledger.action = Math.trunc(raw.actions[0]!) & 3; this.ledger.patience = Math.max(0, constant(ctx.environment, 'SOKOBAN_PATIENCE') - raw.words[6]!); this.ledger.reward = frame.state.reward;
    world.sentinel.place(walk.x, 0, walk.z);
    if (walk.moving) { world.sentinel.face(walk.yaw); world.sentinel.stride(push.moving ? 0.7 : 1, frame.dtReal); } else world.sentinel.idle(time);
    world.crate.position.set(push.x, 0, push.z);
    world.crate.rotation.set(Math.cos(push.yaw) * 0.07 * push.moving, 0, -Math.sin(push.yaw) * 0.07 * push.moving);
    cellToWorldVec(Math.round(frame.state.pose[SOKOBAN_TARGET_X]!), Math.round(frame.state.pose[SOKOBAN_TARGET_Y]!), world.grid, world.target.position);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.goalAt = -Infinity; }
    let goal = false, teleport = false;
    for (const event of frame.events) { if (event.type === 'goal') goal = true; else if (event.type === 'teleport') teleport = true; }
    if (goal) {
      const previous = frame.prev;
      if (previous) cellToWorldVec(Math.round(previous.pose[SOKOBAN_TARGET_X]!), Math.round(previous.pose[SOKOBAN_TARGET_Y]!), world.grid, this.burstAt);
      else this.burstAt.copy(world.target.position);
      pool.ripple(this.burstAt, ctx.tokens.celebrationGold, 0.45, time);
      this.burstAt.y = 0.9;
      pool.burst(this.burstAt, world.accent, 0.8, time);
      pool.pulse.fire(0.55);
      this.goalAt = time;
    } else if (teleport) {
      this.burstAt.copy(world.crate.position).setY(0.6);
      pool.sigh(0.4, this.burstAt, time);
    }
    const flare = envelope(time - this.goalAt, 1.5, 2), near = push.toX === Math.round(frame.state.pose[SOKOBAN_TARGET_X]!) && push.toY === Math.round(frame.state.pose[SOKOBAN_TARGET_Y]!) ? 1 : 0;
    world.targetGlass.emissiveIntensity = 0.4 + 0.3 * Math.sin(time * 2.4) + 1.6 * near + 2 * flare;
    world.sentinel.glow(flare + 0.3 * push.moving);
    world.lamps.forEach((post, i) => post.lit(0.38 + 0.08 * Math.sin(time * 1.3 + i * 2)));
    const follow = world.land.rig.points[0], glow = world.land.rig.points[1];
    if (follow) { follow.color.set(world.accent); follow.position.set(walk.x, 0.9, walk.z); follow.intensity = 0.3 + 0.6 * push.moving; }
    if (glow) { glow.color.set(ctx.tokens.celebrationGold); glow.position.copy(world.target.position).setY(0.8); glow.intensity = 0.4 + 2.5 * flare; }
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera, world.terrain);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.goalAt = -Infinity; }
}

export default ((() => new StorehouseCourtScene()) satisfies SceneFactory);
