import * as THREE from 'three';
import { cellWalk, walkCells } from '@kit/cast/cellwalker';
import { buildSentinel } from '@kit/cast/sentinel';
import type { GroundTruthOptions } from '@kit/debug';
import { buildCellField, CELL, cellToWorldVec, isWall } from '@kit/grid';
import { makeHeightField } from '@kit/heightfield';
import { makeLandscape, makeSceneTerrain } from '@kit/landscape';
import { oakMaterial } from '@kit/materials';
import { envelope } from '@kit/mix';
import { banner, gate, hoodooGeometry, lamp, propMaterials, relief } from '@kit/props';
import { hash2f } from '@kit/rng';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { excludeRect, plantInstances, scatterGrass, scatterRocks } from '@kit/vegetation';
import { makeSceneView, type SceneView } from '@kit/view';
import type { WorldCore } from '@kit/world';
import { BLOCKING_X, BLOCKING_Y, blockingPhase, blockingPhases } from '@sim/envs/blocking';
import { constant } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const DIR = ['\u2191 north', 'east \u2192', '\u2193 south', '\u2190 west'] as const;


function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, phases = blockingPhases(env), grid = phases[0], accent = ctx.tokens.accents[env.family] ?? ctx.tokens.celebrationGold;
  const palette = ctx.tokens.palettes.afternoon, halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2;
  const land = makeLandscape(scene, 'sandstoneMaze', palette, 'afternoon', {
    sky: { sunDir: [0.55, 0.42, -0.72], sunSize: 0.045, sunHdr: 6, sunGlow: 0.36, sunGlowPower: 24, topSpread: 0.6, bottomSpread: 0.35 },
    light: { pointCount: 2, pointDistance: 6 }, fogExtent: 150,
    weather: { preset: 'dust', count: 520, seed: ctx.seed, color: 0xffd9a8, size: 1.2, intensity: 1.1, swirl: 0.5, bounds: { y: 0.9, halfX: halfX + 3, halfY: 0.9, halfZ: halfZ + 3 } },
  });
  const { world, rig, sky } = land;
  sky.setColors(0x6aa3d8, 0xf0c98e, 0xd9b27a);
  rig.sun.color.set(0xffd9a0); rig.sun.intensity = 2.4;
  rig.hemi.color.set(0x9fc3e6); rig.hemi.groundColor.set(0x8a6a44); rig.hemi.intensity = 0.9;
  rig.setShadowFocus({ x: 0, y: 0.4, z: 0 }, halfX + 3);
  const field = makeHeightField({ base: -0.06, court: { halfX: halfX + 0.6, halfZ: halfZ + 0.6, rise: 0.28 }, octaves: [{ wavelength: 29, amplitude: 1.3, seed: ctx.seed }, { wavelength: 9, amplitude: 0.32, seed: ctx.seed ^ 0x2b }] });
  const terrain = world.own(world.time(makeSceneTerrain({ follow: false, name: 'badlandsSand', seed: ctx.seed, colorA: 0xb8834f, colorB: 0xd9b784, segments: 112, radius: 420, height: field.glsl,
    detail: 'color *= 0.92 + 0.08 * landNoise(p.xz * 1.7); color = mix(color, vec3(0.62, 0.42, 0.28), 0.35 * smoothstep(0.35, 0.65, landNoise(p.xz * 0.21)));' }, field.heightAt, { roughness: 0.95, rimColor: palette.sun, rimStrength: 0.16, toonSteps: 4 })));
  world.add(terrain.mesh);
  const both = new Uint8Array(grid.w * grid.h), differing: number[] = [];
  for (let i = 0; i < both.length; i++) {
    const a = phases[0].walls[i]! !== 0, b = phases[1].walls[i]! !== 0;
    both[i] = a && b ? 1 : 0;
    if (a !== b) differing.push(i);
  }
  const constantWalls = { w: grid.w, h: grid.h, walls: both }, flagA = new THREE.Color(0xd7b98c), flagB = new THREE.Color(0xb48f62), tint = new THREE.Color();
  const cells = world.own(world.time(buildCellField(constantWalls, {
    floorColor: 0xffffff, wallColor: 0xffffff, wallShape: 'block', wallHeight: 0.85, wallInset: 0.02, floorInset: 0.035, floorThickness: 0.14, jitter: 0.09, seed: ctx.seed,
    floorTint: (x, y) => tint.copy(flagA).lerp(flagB, hash2f(x, y, ctx.seed)),
    wallTint: (x, y) => tint.set(hash2f(x, y, ctx.seed ^ 0x9d) < 0.3 ? 0x8d7a66 : 0xdcc6a2),
    rimColor: palette.sun, rimStrength: 0.28, name: 'mazeCells',
  })));
  world.add(cells.group);
  const m = propMaterials(world);
  const backdrop = relief(world, grid.w * CELL + 1.6, 1.5, 0.8, m);
  backdrop.position.set(0, 0, -halfZ - 0.55);
  world.add(backdrop);
  const gates = differing.map(index => {
    const x = index % grid.w, y = Math.floor(index / grid.w), alongX = isWall(constantWalls, x - 1, y) || isWall(constantWalls, x + 1, y);
    const door = gate(world, CELL - 0.44, 0.8, accent, m);
    cellToWorldVec(x, y, grid, door.group.position);
    door.group.rotation.y = alongX ? 0 : Math.PI / 2;
    world.add(door.group);
    return { door, index };
  });
  const goal = grid.goal!, goalPos = cellToWorldVec(goal.x, goal.y, grid, new THREE.Vector3());
  const flag = banner(world, accent, 2.2, m);
  flag.position.set(goalPos.x + 0.36, 0, goalPos.z - 0.38);
  world.add(flag);
  const goalLamp = lamp(world, ctx.tokens.celebrationGold, 1.3, m);
  goalLamp.group.position.set(goalPos.x - 0.38, 0, goalPos.z - 0.38);
  goalLamp.lit(0.6);
  world.add(goalLamp.group);
  const hoodoos = plantInstances(world, 'badlandsHoodoos', hoodooGeometry(), oakMaterial({ vertexColors: true, flatShading: true, roughness: 0.95, rimColor: palette.sun, rimStrength: 0.3, toonSteps: 4, name: 'hoodooMaterial' }), 3);
  [[-13, -13, 1.4, 0.4], [11, -16, 1.8, 2.6], [1, -21, 1.6, 1.4]].forEach(([x, z, size, yaw], i) => {
    hoodoos.transform.position.set(x!, field.heightAt(x!, z!) - 0.25, z!);
    hoodoos.transform.rotation.set(0, yaw!, 0);
    hoodoos.transform.scale.setScalar(size!);
    hoodoos.write(i);
  });
  hoodoos.commit(3);
  hoodoos.mesh.castShadow = hoodoos.mesh.receiveShadow = true;
  const court = excludeRect(0, 0, halfX + 0.8, halfZ + 0.8, 0.2);
  const grass = scatterGrass({ count: 1100, area: { halfX: halfX + 16, halfZ: halfZ + 14 }, seed: ctx.seed ^ 0x61, exclude: court, heightAt: field.heightAt, color: 0xc9b478, scale: [0.4, 0.8], stretch: [0.8, 1.4], hueJitter: 0.2, rimColor: palette.sun, rimStrength: 0.3, name: 'badlandsGrass' });
  world.add(grass.group); world.time(world.own(grass));
  const rocks = scatterRocks({ count: 140, area: { halfX: halfX + 16, halfZ: halfZ + 14 }, seed: ctx.seed ^ 0x77, exclude: court, heightAt: field.heightAt, color: 0xb59a7a, scale: [0.12, 0.4], stretch: [0.5, 1], rimColor: palette.sun, rimStrength: 0.28, castShadow: true, name: 'badlandsScree' });
  world.add(rocks.group); world.time(world.own(rocks));
  const sentinel = world.own(buildSentinel({ name: 'mazeSentinel', tint: accent, scale: 0.9 }));
  world.add(sentinel.group);
  const extra = { land, terrain, grid, phases, gates, openness: new Float32Array(gates.length), sentinel, goalLamp, goalPos, accent };
  return world.seal<WorldCore & typeof extra>(extra);
}

type MazeWorld = ReturnType<typeof buildWorld>;

class SandstoneMazeScene extends StageScene<MazeWorld> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['afternoon', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'sandstoneMaze';
  private view!: SceneView;
  private readonly walk = cellWalk();
  private readonly cell = { x: 0, y: 0 };
  private readonly burstAt = new THREE.Vector3();
  private goalAt = -Infinity;
  private step = 0;
  private period = 1;
  private readonly ledger = { x: 0, y: 0, action: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'MAZE',
    rows: [
      { key: 'at', label: 'position', tint: 0xb9c27a, value: () => `column ${this.ledger.x + 1} \u00b7 row ${this.ledger.y + 1}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => DIR[this.ledger.action]! },
      { key: 'walls', label: 'walls', tint: 0xdcc6a2, value: () => (blockingPhase(this.ctx!.environment, this.step) === 0 ? 'first layout' : 'second layout') },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
    note: () => `walls shift in ${formatCount(this.period - this.step % this.period)} steps`,
    bar: () => (this.step % this.period) / this.period,
  };

  protected buildWorld(ctx: SceneContext): MazeWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const { grid } = this.world!, halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2;
    this.view = makeSceneView(ctx, { min: [-halfX - 1, -0.5, -halfZ - 1], max: [halfX + 1, 2.5, halfZ + 1] }, [0, 0.35, -0.3], { distance: Math.max(halfX, halfZ) * 2.6, azimuth: 0.5, polar: 1.0 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return { agentCell: state => { this.cell.x = Math.round(state.pose[BLOCKING_X]!); this.cell.y = Math.round(state.pose[BLOCKING_Y]!); return this.cell; }, goalCell: () => this.world!.grid.goal! };
  }
  protected start(ctx: SceneContext): void { this.period = constant(ctx.environment, 'BLOCKING_PERIOD'); }
  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<MazeWorld>): void {
    const walk = walkCells(frame, BLOCKING_X, BLOCKING_Y, world.grid, this.walk);
    this.step = frame.state.step; this.ledger.x = walk.toX; this.ledger.y = walk.toY; this.ledger.action = Math.trunc(frame.frameAt(frame.state.step).actions[0]!) & 3; this.ledger.reward = frame.state.reward;
    world.sentinel.place(walk.x, 0, walk.z);
    if (walk.moving) { world.sentinel.face(walk.yaw); world.sentinel.stride(1, frame.dtReal); } else world.sentinel.idle(time);
    const phase = blockingPhase(ctx.environment, frame.state.step), k = frame.scrubbed ? 1 : 1 - Math.exp(-4 * frame.dtReal);
    world.gates.forEach(({ door, index }, i) => {
      const target = world.phases[phase]!.walls[index] ? 0 : 1, open = world.openness[i]! + (target - world.openness[i]!) * k;
      world.openness[i] = open;
      door.open(open); door.lit(open);
    });
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.goalAt = -Infinity; }
    for (const event of frame.events) if (event.type === 'goal') {
      this.burstAt.copy(world.goalPos); this.burstAt.y = 1.4;
      pool.burst(this.burstAt, world.accent, 0.7, time);
      pool.ripple(world.goalPos, ctx.tokens.celebrationGold, 0.4, time);
      pool.pulse.fire(0.5);
      this.goalAt = time;
    }
    const flare = envelope(time - this.goalAt, 1.4, 2);
    world.goalLamp.lit(0.6 + 1.6 * flare);
    world.sentinel.glow(flare);
    const follow = world.land.rig.points[0], glow = world.land.rig.points[1];
    if (follow) { follow.color.set(world.accent); follow.position.set(walk.x, 0.9, walk.z); follow.intensity = 0.35 + flare; }
    if (glow) { glow.color.set(ctx.tokens.celebrationGold); glow.position.set(world.goalPos.x, 1.2, world.goalPos.z); glow.intensity = 2.5 * flare; }
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera, world.terrain);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.goalAt = -Infinity; }
}

export default ((() => new SandstoneMazeScene()) satisfies SceneFactory);
