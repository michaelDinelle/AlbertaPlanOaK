import * as THREE from 'three';
import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { clamp01, envelope, smoothstep01 } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { Wisp } from '@kit/wisp';
import { clearOrchardReadout, publishOrchardReadout } from '../../state/orchard';
import type { GridSize } from '@kit/grid';
import type { GridCell } from '@sim/types';
import {
  jbwCandidate,
  jbwItemAt,
  jbwPolarity,
  jbwWallLine,
  JBW_BANANAS,
  JBW_BLOCK,
  JBW_EMPTY,
  JBW_HEADING,
  JBW_JELLIES,
  JBW_ONIONS,
  JBW_TRUFFLES,
  JBW_WALL,
  JBW_X,
  JBW_Y,
  jbwCollectible,
  jbwEnergy,
  jbwBlocks,
} from '@sim/envs/jellybeanworld';
import { AGENT_SPAN, ITEM_KINDS } from './actors';
import { buildScentKernel, ScentField } from './scent';
import {
  buildWorld,
  CAM_AZIMUTH,
  CAM_DISTANCE,
  CAM_FOLLOW,
  CAM_FAR,
  CAM_MAX_DISTANCE,
  MAP_FULL,
  MAP_START,
  CAM_NEAR,
  CAM_POLAR,
  OCCLUDER_LIMIT,
  orchardHeight,
  DRAW_RADIUS,
  type OrchardWorld,
} from './world';

const UNIT_GRID: GridSize = { w: 1, h: 1 };

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.16;

const BURST_SCALE = 0.92;
const RIPPLE_SCALE = 0.62;
const PULSE_STRENGTH = 0.6;
const BLOOM_Y = 0.62;
const BLOOM_SECONDS = 1.4;
const SIGH_STRENGTH = 0.85;

const HARVEST_TAU = 4096;
const HARVEST_FULL = HARVEST_TAU / 512;
const HARVEST_CHASE = 0.9;

const FRONT_SECONDS = 1.35;
const FRONT_SPAN = 220;
const BLOCK_BIAS = 1 << 18;
const BLOCK_SPAN = 1 << 19;
const BLOCK_CACHE_LIMIT = 40000;
const DEBUG_CELLS = 33;

const INVENTORY_SLOTS: readonly number[] = [JBW_BANANAS, JBW_ONIONS, JBW_JELLIES, JBW_TRUFFLES];

const READOUT_SECONDS = 1 / 15;

const FORWARD: readonly (readonly [number, number])[] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];

const SENSE_DIRS: readonly (readonly [number, number])[] = [
  [0, -1],
  [1, -1],
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1],
];

class WindfallOrchardScene extends StageScene<OrchardWorld> {
  protected readonly stageName = 'windfallOrchard';

  private wisp: Wisp | null = null;
  private field: ScentField | null = null;
  private seed = 0;

  private readonly inventory = new Float32Array(INVENTORY_SLOTS.length);
  private inventoryReady = false;
  private rate = 0;
  private rateStep = 0;
  private harvest = 0.35;
  private polarity: 1 | -1 = 1;
  private ripeness = 1;
  private frontAt = -Infinity;
  private frontFrom = 1;
  private bloomAt = -Infinity;

  private readonly cell: GridCell = { x: 0, y: 0 };
  private readonly matrix = new THREE.Matrix4();
  private readonly place = new THREE.Vector3();
  private readonly spin = new THREE.Quaternion();
  private readonly tilt = new THREE.Euler();
  private readonly unit = new THREE.Vector3(1, 1, 1);
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly counts = new Map<number, number>();
  private sightX = Number.NaN;
  private sightY = Number.NaN;
  private sightHeading = -1;
  private readonly blocks = new Map<number, Int32Array>();
  private blockX = Number.NaN;
  private blockY = Number.NaN;
  private itemsDirty = true;
  private readonly debugWalls = new Uint8Array(DEBUG_CELLS * DEBUG_CELLS);
  private readonly debugCell: GridCell = { x: 0, y: 0 };
  private readonly sensed = new Float32Array(8);
  private readonly scentOut = new Float32Array(3);
  private readonly sight = new Float32Array(OCCLUDER_LIMIT * 3);
  private readoutAt = -Infinity;
  private moved = false;
  private accent = TOKENS.celebrationGold;
  private gold = TOKENS.celebrationGold;

  protected buildWorld(ctx: SceneContext): OrchardWorld {
    this.seed = ctx.seed;
    const field = new ScentField(
      ctx.seed,
      buildScentKernel(),
      ctx.renderer.capabilities.getMaxAnisotropy(),
    );
    this.field = field;
    return buildWorld(ctx, this.scene, field);
  }

  protected buildActors(ctx: SceneContext, world: OrchardWorld): void {
    this.accent = this.accentOf(ctx);
    this.gold = ctx.tokens.celebrationGold;
    const wisp = new Wisp(
      {
        color: this.accent,
        radius: AGENT_SPAN / 4.8,
        height: AGENT_SPAN * 0.24,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'orchardWisp',
      },
      UNIT_GRID,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);
    const agentLight = world.rig.points[0];
    if (agentLight) agentLight.color.set(this.accent);
    const bloomLight = world.rig.points[1];
    if (bloomLight) bloomLight.color.set(this.gold);
  }

  protected rigOptions(): OrbitRigOptions {
    return {
      target: [0, 0.4, 0],
      distance: CAM_DISTANCE,
      azimuth: CAM_AZIMUTH,
      polar: CAM_POLAR,
      follow: CAM_FOLLOW,
      limits: { maxDistance: CAM_MAX_DISTANCE },
    };
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 108, sighSize: 96 };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      windowCells: DEBUG_CELLS,
      gridWindow: (state) => {
        const ax = Math.round(state.pose[JBW_X] ?? 0);
        const ay = Math.round(state.pose[JBW_Y] ?? 0);
        const half = (DEBUG_CELLS - 1) / 2;
        const walls = this.debugWalls;
        walls.fill(0);
        for (let j = 0; j < DEBUG_CELLS; j++) {
          for (let i = 0; i < DEBUG_CELLS; i++) {
            const wx = ax - half + i;
            const wy = ay - half + j;
            if (jbwItemAt(this.seed, wx, wy) === JBW_WALL) walls[j * DEBUG_CELLS + i] = 1;
          }
        }
        return { walls, originX: ax - half, originY: ay - half, height: orchardHeight(this.seed, ax, ay) };
      },
      agentCell: () => {
        this.debugCell.x = (DEBUG_CELLS - 1) / 2;
        this.debugCell.y = (DEBUG_CELLS - 1) / 2;
        return this.debugCell;
      },
    };
  }

  protected start(_ctx: SceneContext, world: OrchardWorld): void {
    this.camera.near = CAM_NEAR;
    this.camera.far = CAM_FAR;
    this.camera.updateProjectionMatrix();
    world.setTime(0);
    world.setOrigin(0, 0);
    this.wisp?.setCell(0, 0, UNIT_GRID);
  }

  private readInventory(pose: Float32Array): number {
    let gained = 0;
    for (let i = 0; i < INVENTORY_SLOTS.length; i++) {
      const slot = INVENTORY_SLOTS[i] ?? 0;
      const now = pose[slot] ?? 0;
      const was = this.inventory[i] ?? 0;
      if (this.inventoryReady && now > was) gained += now - was;
      this.inventory[i] = now;
    }
    this.inventoryReady = true;
    return gained;
  }

  private advanceRate(step: number, credit: number): void {
    const elapsed = step - this.rateStep;
    if (elapsed > 0) {
      this.rate *= Math.exp(-elapsed / HARVEST_TAU);
      this.rateStep = step;
    }
    this.rate += credit;
    if (this.rate < 0) this.rate = 0;
  }

  private blockAt(bx: number, by: number): Int32Array {
    const key = (bx + BLOCK_BIAS) * BLOCK_SPAN + (by + BLOCK_BIAS);
    const cached = this.blocks.get(key);
    if (cached) return cached;
    const out: number[] = [];
    const line = jbwWallLine(this.seed, bx, by);
    if (line) {
      for (let i = 0; i < JBW_BLOCK; i++) {
        out.push(
          line[0] ? bx * JBW_BLOCK + i : line[1],
          line[0] ? line[1] : by * JBW_BLOCK + i,
          JBW_WALL,
        );
      }
    }
    const at = jbwCandidate(this.seed, bx, by);
    const item = jbwItemAt(this.seed, at[0], at[1]);
    if (item !== JBW_EMPTY && item !== JBW_WALL) out.push(at[0], at[1], item);
    const packed = Int32Array.from(out);
    if (this.blocks.size >= BLOCK_CACHE_LIMIT) this.blocks.clear();
    this.blocks.set(key, packed);
    return packed;
  }

  private rebuildItems(world: OrchardWorld, ax: number, ay: number, field: ScentField): void {
    const counts = this.counts;
    for (const kind of ITEM_KINDS) counts.set(kind, 0);
    const firstX = Math.floor((ax - DRAW_RADIUS) / JBW_BLOCK);
    const lastX = Math.floor((ax + DRAW_RADIUS) / JBW_BLOCK);
    const firstY = Math.floor((ay - DRAW_RADIUS) / JBW_BLOCK);
    const lastY = Math.floor((ay + DRAW_RADIUS) / JBW_BLOCK);
    for (let by = firstY; by <= lastY; by++) {
      for (let bx = firstX; bx <= lastX; bx++) {
        const packed = this.blockAt(bx, by);
        for (let i = 0; i < packed.length; i += 3) {
          const wx = packed[i] ?? 0;
          const wy = packed[i + 1] ?? 0;
          const item = packed[i + 2] ?? 0;
          if (jbwCollectible(item) && field.depleted(wx, wy)) continue;
          this.write(world, counts, item, wx, wy);
        }
      }
    }
    for (const kind of ITEM_KINDS) {
      const mesh = world.items.get(kind);
      if (!mesh) continue;
      mesh.count = counts.get(kind) ?? 0;
      mesh.instanceMatrix.needsUpdate = true;
    }
  }

  private write(
    world: OrchardWorld,
    counts: Map<number, number>,
    item: number,
    wx: number,
    wy: number,
  ): void {
    const mesh = world.items.get(item);
    if (!mesh) throw new Error(`no orchard mesh for item ${item}`);
    const used = counts.get(item) ?? 0;
    if (used >= mesh.instanceMatrix.count) {
      throw new Error(`orchard item ${item} exceeded capacity ${mesh.instanceMatrix.count}`);
    }
    this.place.set(wx, orchardHeight(this.seed, wx, wy), wy);
    this.tilt.set(0, wx * 0.71 + wy * 1.13, 0);
    this.spin.setFromEuler(this.tilt);
    this.matrix.compose(this.place, this.spin, this.unit);
    mesh.setMatrixAt(used, this.matrix);
    counts.set(item, used + 1);
  }

  private publish(field: ScentField, ax: number, ay: number, heading: number, step: number): void {
    for (let i = 0; i < 8; i++) {
      const d = SENSE_DIRS[(2 * heading + i) & 7];
      if (!d) continue;
      const wx = ax + d[0];
      const wy = ay + d[1];
      const item = jbwItemAt(this.seed, wx, wy);
      const absent = !jbwBlocks(item) && field.depleted(wx, wy);
      this.sensed[i] = absent ? 0 : jbwEnergy(item);
    }
    field.sample(ax, ay, step, this.scentOut);
    publishOrchardReadout({
      live: true,
      sensed: Array.from(this.sensed),
      scent: [this.scentOut[0] ?? 0, this.scentOut[1] ?? 0, this.scentOut[2] ?? 0],
      inventory: [
        this.inventory[0] ?? 0,
        this.inventory[1] ?? 0,
        this.inventory[2] ?? 0,
        this.inventory[3] ?? 0,
      ],
      moved: this.moved,
      polarity: this.polarity,
      harvest: this.harvest,
    });
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<OrchardWorld>): void {
    const wisp = this.wisp;
    const field = this.field;
    if (!wisp || !field) return;

    const pose = frame.state.pose;
    const step = frame.state.step;
    const ax = Math.round(pose[JBW_X] ?? 0);
    const ay = Math.round(pose[JBW_Y] ?? 0);
    const heading = Math.round(pose[JBW_HEADING] ?? 0) & 3;

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      this.bloomAt = -Infinity;
      this.frontAt = -Infinity;
      this.rate = 0;
      this.rateStep = step;
      this.sightX = Number.NaN;
      this.itemsDirty = true;
      this.inventoryReady = false;
      this.harvest = 0.35;
      field.resetScars();
      wisp.setCell(ax, ay, UNIT_GRID);
      wisp.snap();
      rig.snap();
    }

    const gained = this.readInventory(pose);
    const polarity = jbwPolarity(step);
    if (polarity !== this.polarity) {
      this.polarity = polarity;
      this.frontFrom = this.ripeness;
      this.frontAt = time;
    }

    let credit = 0;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev || ev.type !== 'collection') continue;
      credit += ev.reward;
      this.bloomPos.set(ax, orchardHeight(this.seed, ax, ay) + BLOOM_Y, ay);
      this.groundPos.set(this.bloomPos.x, this.bloomPos.y - BLOOM_Y, this.bloomPos.z);
      if (ev.reward >= 0) {
        pool.burst(this.bloomPos, this.accent, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        this.bloomAt = time;
      } else {
        pool.sigh(SIGH_STRENGTH, this.bloomPos, time);
      }
    }
    if (gained > 0) {
      field.scar(ax, ay, step);
      this.itemsDirty = true;
    }

    this.advanceRate(step, credit);
    const target = clamp01(this.rate / HARVEST_FULL);
    this.harvest += (target - this.harvest) * (1 - Math.exp(-HARVEST_CHASE * frame.dtReal));

    const frontAge = time - this.frontAt;
    const frontT = frontAge >= 0 && frontAge <= FRONT_SECONDS ? frontAge / FRONT_SECONDS : 1;
    const wanted = polarity > 0 ? 1 : 0;
    this.ripeness = this.frontFrom + (wanted - this.frontFrom) * smoothstep01(frontT);
    world.setRipeness(this.ripeness);
    world.setFront(
      ax + ay + (frontT * 2 - 1) * FRONT_SPAN,
      envelope(frontAge, FRONT_SECONDS, 1.4) * 1.6,
    );

    field.update(ax, ay, step);
    const origin = field.origin;
    world.setScentOrigin(origin[0], origin[1]);
    if (ax !== this.sightX || ay !== this.sightY || heading !== this.sightHeading) {
      this.sightX = ax;
      this.sightY = ay;
      this.sightHeading = heading;
      const forward = FORWARD[heading] ?? FORWARD[0];
      const count = field.occluders(ax, ay, this.sight);
      world.paint.uOccluders.value.set(this.sight);
      world.setSight(ax, ay, Math.atan2(forward?.[1] ?? 0, forward?.[0] ?? 1), count);
    }

    world.setOrigin(ax, ay);
    world.setHour(this.harvest);
    const zoom = rig.state.distance;
    world.setMap(smoothstep01((zoom - MAP_START) / (MAP_FULL - MAP_START)), zoom);
    world.setBreeze(0.25 + 0.75 * this.harvest);
    const blockX = Math.floor(ax / JBW_BLOCK);
    const blockY = Math.floor(ay / JBW_BLOCK);
    if (this.itemsDirty || blockX !== this.blockX || blockY !== this.blockY) {
      this.itemsDirty = false;
      this.blockX = blockX;
      this.blockY = blockY;
      this.rebuildItems(world, ax, ay, field);
    }

    this.moved = ax !== wisp.cellX || ay !== wisp.cellY;
    if (this.moved) {
      this.cell.x = ax;
      this.cell.y = ay;
      wisp.hopTo(this.cell, time);
    }
    wisp.face(heading === 0 ? { dx: 0, dy: -1 } : heading === 1 ? { dx: 1, dy: 0 } : heading === 2 ? { dx: 0, dy: 1 } : { dx: -1, dy: 0 });
    wisp.update(frame.dtReal, time);
    wisp.group.position.y = orchardHeight(this.seed, ax, ay);

    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.2);
    const agentLight = world.rig.points[0];
    if (agentLight) {
      agentLight.position.set(wisp.position.x, wisp.position.y + TOKENS.wispHeight + 0.6, wisp.position.z);
      agentLight.intensity = 1.15 + 1.4 * bloom + 0.5 * this.harvest;
    }
    const bloomLight = world.rig.points[1];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 2.4 * bloom;
    }

    if (time - this.readoutAt >= READOUT_SECONDS) {
      this.readoutAt = time;
      this.publish(field, ax, ay, heading, step);
    }

    world.setTime(time);
    rig.follow(wisp.position.x, wisp.position.y + 0.4, wisp.position.z);
    rig.update(frame.dtReal);
  }

  protected drop(): void {
    clearOrchardReadout();
    this.readoutAt = -Infinity;
    this.wisp?.dispose();
    this.wisp = null;
    this.field?.dispose();
    this.field = null;
    this.counts.clear();
    this.blocks.clear();
    this.inventoryReady = false;
    this.bloomAt = -Infinity;
    this.frontAt = -Infinity;
  }
}

export default ((() => new WindfallOrchardScene()) satisfies SceneFactory);
