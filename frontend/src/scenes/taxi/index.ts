import * as THREE from 'three';
import { constant } from '../../data/envs';
import { cellWalk, gridHeading, walkCells } from '@kit/cast/cellwalker';
import { buildCourier } from '@kit/cast/courier';
import type { GroundTruthOptions } from '@kit/debug';
import { buildCellField, CELL, cellToWorldVec } from '@kit/grid';
import { makeLandscape } from '@kit/landscape';
import { makeMeadow } from '@kit/meadow';
import { envelope } from '@kit/mix';
import { banner, lamp, plinth, propMaterials, relief } from '@kit/props';
import { hash2f } from '@kit/rng';
import { modelParts } from '@kit/robot';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import type { WorldCore } from '@kit/world';
import { TAXI_DESTINATION, TAXI_PASSENGER, TAXI_X, TAXI_Y, taxiBlocked, taxiDepots } from '@sim/envs/taxi';
import { formatSigned } from '../../format';

const GROUND = () => 0;
const DEPOT = ['cyan', 'magenta', 'gold', 'green'] as const;
const ACTION = ['\u2191 north', 'east \u2192', '\u2193 south', '\u2190 west', 'pick up', 'drop off'] as const;

function buildWorld(ctx: SceneContext, scene: THREE.Scene) {
  const env = ctx.environment, grid = ctx.geometry.grid;
  if (!grid) throw new Error('Taxi grid is missing');
  const side = constant(env, 'TAXI_SIDE'), depots = taxiDepots(env), accent = ctx.tokens.accents[env.family] ?? ctx.tokens.celebrationGold, palette = ctx.tokens.palettes.morning;
  const halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2;
  const land = makeLandscape(scene, 'townSquare', palette, 'morning', {
    sky: { sunDir: [0.5, 0.55, 0.67], sunSize: 0.04, sunHdr: 5.5, sunGlow: 0.3, sunGlowPower: 30, topSpread: 0.65, bottomSpread: 0.4 },
    light: { pointCount: 2, pointDistance: 6 }, fogExtent: 200,
  });
  const { world, rig } = land;
  rig.sun.color.set(0xfff0cf); rig.sun.intensity = 2.5;
  rig.hemi.color.set(0xbfe0f5); rig.hemi.groundColor.set(0x7e8f5c); rig.hemi.intensity = 1;
  rig.setShadowFocus({ x: 0, y: 0.4, z: 0 }, halfX + 3);
  const meadow = makeMeadow(world, ctx, { name: 'springSquare', halfX: halfX + 0.9, halfZ: halfZ + 0.9, ground: 0x8ea666, grass: 0xa9c47a, trees: [0xf2b7cf, 0xa9d954, 0xe8c8dc, 0x9fcf6a], archetypes: ['broadleaf'], amplitude: 1.3 });
  const flagA = new THREE.Color(0xe9dcc2), flagB = new THREE.Color(0xc9b596), tint = new THREE.Color();
  const cells = world.own(world.time(buildCellField(grid, { floorColor: 0xffffff, wallColor: 0xffffff, floorThickness: 0.16, floorInset: 0.035, jitter: 0.08, seed: ctx.seed, floorTint: (x, y) => tint.copy(flagA).lerp(flagB, hash2f(x, y, ctx.seed)), rimColor: palette.sun, rimStrength: 0.22, name: 'squareFlagstones' })));
  world.add(cells.group);
  const m = propMaterials(world);
  for (let x = 0; x < side - 1; x++) for (let y0 = 0; y0 < side; y0++) {
    if (!taxiBlocked(side, x, y0, x + 1, y0) || (y0 > 0 && taxiBlocked(side, x, y0 - 1, x + 1, y0 - 1))) continue;
    let y1 = y0;
    while (y1 + 1 < side && taxiBlocked(side, x, y1 + 1, x + 1, y1 + 1)) y1++;
    const wall = relief(world, (y1 - y0 + 1) * CELL, 0.72, 0.4, m);
    wall.rotation.y = Math.PI / 2;
    wall.position.set((x + 0.5 - (side - 1) / 2) * CELL, 0, ((y0 + y1) / 2 - (side - 1) / 2) * CELL);
    world.add(wall);
  }
  const colors = [ctx.tokens.cueCyan, ctx.tokens.cueMagenta, ctx.tokens.celebrationGold, ctx.tokens.accents['hall'] ?? ctx.tokens.uiCyan];
  const parts = modelParts(world), stalls = new THREE.Group();
  const depotPos = depots.map((depot, i) => {
    const at = cellToWorldVec(depot.x, depot.y, grid, new THREE.Vector3()), sx = Math.sign(at.x) || 1, sz = Math.sign(at.z) || 1, color = colors[i % colors.length]!;
    const flag = banner(world, color, 2.1, m);
    flag.position.set(at.x + sx * 0.4, 0, at.z + sz * 0.4);
    world.add(flag);
    const stand = plinth(world, 0.24, 0.5, m);
    stand.position.set(at.x + sx * 0.4, 0, at.z - sz * 0.36);
    world.add(stand);
    parts.disc(stalls, m.glass(color, 0.12), at.x, 0.012, at.z, 0.3, 0.02, 24);
    return at;
  });
  parts.build();
  world.add(stalls);
  const tiles = stalls.children.map(child => (child as THREE.Mesh).material as THREE.MeshStandardMaterial);
  const lamps = [[-halfX - 0.7, -halfZ - 0.7], [halfX + 0.7, halfZ + 0.7]].map(([x, z]) => {
    const post = lamp(world, 0xffe0b0, 1.6, m);
    post.group.position.set(x!, meadow.heightAt(x!, z!), z!);
    post.lit(0.5);
    world.add(post.group);
    return post;
  });
  const taxi = world.own(buildCourier({ name: 'squareTaxi', tint: accent, scale: 1.15, overgrowth: 0.2 }));
  world.add(taxi.group);
  const passenger = world.own(new Wisp({ color: 0xfff1c8, radius: 0.24, height: 0.6, skirt: true, ribbon: false, silhouette: false, name: 'passengerWisp' }));
  world.add(passenger.group);
  const extra = { land, meadow, grid, depots, depotPos, tiles, colors, lamps, taxi, passenger, accent };
  return world.seal<WorldCore & typeof extra>(extra);
}

type SquareWorld = ReturnType<typeof buildWorld>;

class TownSquareScene extends StageScene<SquareWorld> {
  readonly post: ScenePost = { bloom: 0.4, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'townSquare';
  private view!: SceneView;
  private readonly walk = cellWalk();
  private readonly from = { x: 0, y: 0, heading: 0 };
  private readonly to = { x: 0, y: 0, heading: 0 };
  private readonly cell = { x: 0, y: 0 };
  private readonly target = new THREE.Vector3();
  private readonly burstAt = new THREE.Vector3();
  private heading = 0;
  private lastStep = -1;
  private goalAt = -Infinity;
  private readonly ledger = { passenger: 0, destination: 0, action: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'TAXI',
    rows: [
      { key: 'passenger', label: 'passenger', tint: 0xfff1c8, value: () => (this.ledger.passenger >= DEPOT.length ? 'aboard' : `waiting at ${DEPOT[this.ledger.passenger]!}`) },
      { key: 'destination', label: 'destination', tint: 0xf5d35c, value: () => `${DEPOT[this.ledger.destination]!} stall` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => ACTION[this.ledger.action]! },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): SquareWorld { return buildWorld(ctx, this.scene); }
  protected buildActors(): void {}
  protected rigOptions(ctx: SceneContext) {
    const { grid } = this.world!, halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2;
    this.view = makeSceneView(ctx, { min: [-halfX - 1.5, -0.5, -halfZ - 1.5], max: [halfX + 1.5, 2.5, halfZ + 1.5] }, [0, 0.4, 0], { distance: Math.max(halfX, halfZ) * 3.0, azimuth: 0.62, polar: 0.95 });
    return this.view.options;
  }
  protected overlayOptions(): GroundTruthOptions {
    return {
      agentCell: state => { this.cell.x = Math.round(state.pose[TAXI_X]!); this.cell.y = Math.round(state.pose[TAXI_Y]!); return this.cell; },
      goalCell: state => this.world!.depots[Math.round(state.pose[TAXI_DESTINATION]!)] ?? null,
      keyCells: () => this.world!.grid.keys ?? null,
      maxKeys: 4,
      debug: (_frame, kit) => {
        const world = this.world!, gold = this.ctx!.tokens.celebrationGold, cyan = this.ctx!.tokens.uiCyan, riding = this.ledger.passenger >= world.depots.length;
        const pickup = riding ? world.taxi.group.position : world.depotPos[this.ledger.passenger]!, drop = world.depotPos[this.ledger.destination]!;
        kit.marks('fare', 2).set(pickup.x, 1.3, pickup.z, 0.11, 0xfff1c8);
        kit.marks('fare', 2).set(drop.x, 1.3, drop.z, 0.12, gold);
        kit.trail('taxi', 60, cyan).push(world.taxi.group.position.x, 0.4, world.taxi.group.position.z);
      },
    };
  }
  protected start(): void { this.passengerTo(this.world!, true); }
  private passengerTo(world: SquareWorld, snap: boolean, k = 1): void {
    world.passenger.group.position.lerp(this.target, snap ? 1 : k);
  }
  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<SquareWorld>): void {
    const walk = walkCells(frame, TAXI_X, TAXI_Y, world.grid, this.walk), ox = (world.grid.w - 1) / 2, oz = (world.grid.h - 1) / 2;
    const passengerAt = Math.round(frame.state.pose[TAXI_PASSENGER]!), destination = Math.round(frame.state.pose[TAXI_DESTINATION]!), riding = passengerAt >= world.depots.length;
    if (walk.moving) this.heading = gridHeading(walk.toX - walk.fromX, walk.toY - walk.fromY, this.heading);
    this.from.x = (walk.fromX - ox) * CELL; this.from.y = (walk.fromY - oz) * CELL; this.from.heading = this.heading;
    this.to.x = (walk.toX - ox) * CELL; this.to.y = (walk.toY - oz) * CELL; this.to.heading = this.heading;
    this.ledger.passenger = passengerAt; this.ledger.destination = destination; this.ledger.action = Math.min(5, Math.max(0, Math.trunc(frame.frameAt(frame.state.step).actions[0]!))); this.ledger.reward = frame.state.reward;
    const snap = frame.scrubbed || frame.state.reset;
    world.taxi.drive(this.from, this.to, frame.position - frame.state.step, GROUND, snap);
    if (riding) this.target.copy(world.taxi.group.position).setY(0.42);
    else { const depot = world.depotPos[passengerAt]; if (depot) this.target.copy(depot).setY(0); }
    this.passengerTo(world, snap || riding, 1 - Math.exp(-10 * frame.dtReal));
    world.passenger.update(frame.dtReal, time);
    pool.update(frame.dtReal, time);
    if (frame.scrubbed) { pool.clear(); this.goalAt = -Infinity; this.lastStep = frame.state.step; }
    for (const event of frame.events) {
      if (event.type === 'goal') {
        this.burstAt.copy(world.taxi.group.position).setY(1.1);
        pool.burst(this.burstAt, ctx.tokens.celebrationGold, 0.8, time);
        pool.ripple(world.taxi.group.position, ctx.tokens.celebrationGold, 0.45, time);
        pool.pulse.fire(0.55);
        world.passenger.flare(1);
        this.goalAt = time;
      } else if (event.type === 'key') {
        this.burstAt.copy(world.taxi.group.position).setY(0.7);
        pool.burst(this.burstAt, world.colors[event.index % world.colors.length]!, 0.35, time);
        world.passenger.flare(0.7);
      }
    }
    if (!frame.scrubbed && frame.state.step !== this.lastStep && frame.state.reward <= -10) { this.burstAt.copy(world.taxi.group.position).setY(0.5); pool.sigh(0.35, this.burstAt, time); }
    this.lastStep = frame.state.step;
    const flare = envelope(time - this.goalAt, 1.4, 2), pulse = 0.5 + 0.5 * Math.sin(time * 3);
    world.tiles.forEach((tile, i) => { tile.emissiveIntensity = i === destination ? 1 + 0.7 * pulse + 2 * flare : 0.12; });
    world.lamps.forEach((post, i) => post.lit(0.45 + 0.15 * Math.sin(time * 1.1 + i * 2)));
    const headlamp = world.land.rig.points[0], glow = world.land.rig.points[1], depot = world.depotPos[destination];
    if (headlamp) { headlamp.color.set(world.accent); headlamp.position.copy(world.taxi.group.position).setY(0.7); headlamp.intensity = 0.35 + 0.6 * flare; }
    if (glow && depot) { glow.color.set(world.colors[destination % world.colors.length]!); glow.position.copy(depot).setY(1); glow.intensity = 0.6 + 0.4 * pulse + 2.4 * flare; }
    world.setTime(time);
    world.land.setView(this.view.update(rig, frame.dtReal), this.camera, world.meadow);
  }
  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.view) this.view.resize(size, this.rig); }
  protected drop(): void { this.goalAt = -Infinity; this.lastStep = -1; }
}

export default ((() => new TownSquareScene()) satisfies SceneFactory);
