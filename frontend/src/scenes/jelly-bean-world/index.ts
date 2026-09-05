import * as THREE from 'three';
import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { CellBoxes, GroundTruthOptions } from '@kit/debug';
import { clamp01, envelope, smoothstep01 } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import type { Wisp } from '@kit/wisp';
import type { GridSize, Vec3Tuple } from '@kit/grid';
import type { GridCell } from '@sim/types';
import {
  jbwBlockField,
  jbwCandidate,
  jbwItemAt,
  jbwPolarity,
  jbwWallLine,
  JBW_BLOCK,
  JBW_BLOCK_ITEMS,
  JBW_EMPTY,
  JBW_ENERGY,
  JBW_HEADING,
  JBW_TRUFFLE,
  JBW_TREE,
  JBW_WALL,
  JBW_X,
  JBW_Y,
  jbwCollectible,
} from '@sim/envs/jellybeanworld';
import {
  buildSeedpod,
  DEBUG_CELL_CAPACITY,
  DECAL_CAPACITY,
  DRAW_RADIUS,
  ITEM_KINDS,
  ITEM_MOTION,
} from './actors';
import { CollectionLog } from './collectionlog';
import { scentAtCell } from './fieldmath';
import { SenseRing, SENSE_DIRECTIONS } from './sense';
import { buildWorld, SHADOW_SPAN, type OrchardWorld } from './world';
import { makeSceneView, type SceneView, type ViewCoverage } from './view';

const UNIT_GRID: GridSize = { w: 1, h: 1 };

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
const BLOCK_CACHE_LIMIT = 131072;
const GRID_GAIN = 0.34;
const ENERGY_PEAK = Math.max(...JBW_ENERGY);
const IDENTITY = new THREE.Quaternion();
const DECAL_LIFT = 0.05;

class WindfallOrchardScene extends StageScene<OrchardWorld> {
  protected readonly stageName = 'windfallOrchard';

  private wisp: Wisp | null = null;
  private readonly clog = new CollectionLog();
  private view: SceneView | null = null;
  private seed = 0;

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
  private surface: (x: number, z: number) => number = () => 0;
  private decalCount = 0;
  private beaconCount = 0;

  private readonly decalScale = new THREE.Vector3(1, 1, 1);
  private readonly decalKind = new Uint8Array(DECAL_CAPACITY);
  private readonly decalColor = new THREE.Color();
  private decalRipeness = Number.NaN;
  private readonly place = new THREE.Vector3();
  private readonly unit = new THREE.Vector3(1, 1, 1);
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly counts = new Map<number, number>();
  private readonly blocks = new Map<number, Float64Array>();
  private blockX = Number.NaN;
  private blockY = Number.NaN;
  private itemsDirty = true;
  private fieldStamp = '';
  private pixelScale = 1;
  private readonly fieldProjection = new THREE.Matrix4();
  private readonly fieldFrustum = new THREE.Frustum();
  private readonly fieldSphere = new THREE.Sphere(new THREE.Vector3(), 1);
  private sense: SenseRing | null = null;
  private readonly energies = new Float32Array(8);
  private readonly debugCone = new Float32Array(0);
  private readonly debugAgent: Vec3Tuple = [0, 0, 0];
  private boxes: CellBoxes | null = null;
  private debugWas = false;
  private frameX = 0;
  private frameY = 0;
  private frameZ = 0;
  private accent = TOKENS.celebrationGold;
  private gold = TOKENS.celebrationGold;

  protected buildWorld(ctx: SceneContext): OrchardWorld {
    this.seed = ctx.seed;
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext, world: OrchardWorld): void {
    this.accent = this.accentOf(ctx);
    this.gold = ctx.tokens.celebrationGold;
    const wisp = buildSeedpod(this.accent);
    this.wisp = wisp;
    this.scene.add(wisp.group);
    const sense = new SenseRing(this.accent);
    this.sense = sense;
    this.scene.add(sense.group);
    const agentLight = world.rig.points[0];
    if (agentLight) agentLight.color.set(this.accent);
    const bloomLight = world.rig.points[1];
    if (bloomLight) bloomLight.color.set(this.gold);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-24, -6, -24], max: [24, 7, 24] }, [0, 0.4, 0], { distance: 78, azimuth: 0.68, polar: 1.2 }, 720);
    return this.view.options;
  }

  protected override poolOptions(): CelebrationPoolOptions {
    return { burstSize: 108, sighSize: 96 };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      cellCapacity: DEBUG_CELL_CAPACITY,
      frame: (state) => {
        const ax = Math.round(state.pose[JBW_X] ?? 0);
        const ay = Math.round(state.pose[JBW_Y] ?? 0);
        this.debugAgent[0] = ax - this.frameX;
        this.debugAgent[1] = this.surface(ax, ay) - this.frameY;
        this.debugAgent[2] = ay - this.frameZ;
        return {
          originX: this.frameX,
          originY: this.frameY,
          originZ: this.frameZ,
          agent: this.debugAgent,
          cone: this.debugCone,
          coneCount: 0,
        };
      },
    };
  }

  protected start(ctx: SceneContext, world: OrchardWorld): void {
    this.pixelScale = ctx.size.height / (2 * Math.tan(this.camera.fov * Math.PI / 360));
    this.surface = (x, z) => world.heightAt(x, z);
    world.setTime(0);
    world.setOrigin(0, 0);
    this.wisp?.setCell(0, 0, UNIT_GRID);
  }

  override resize(size: Parameters<StageScene<OrchardWorld>['resize']>[0]): void {
    super.resize(size);
    this.pixelScale = size.height / (2 * Math.tan(this.camera.fov * Math.PI / 360));
    this.itemsDirty = true;
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

  private blockAt(bx: number, by: number): Float64Array {
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
          this.surface(line[0] ? bx * JBW_BLOCK + i : line[1], line[0] ? line[1] : by * JBW_BLOCK + i),
        );
      }
    }
    const block = jbwBlockField(this.seed, bx, by);
    for (let slot = 0; slot < JBW_BLOCK_ITEMS; slot++) {
      const at = jbwCandidate(block, bx, by, slot);
      const item = jbwItemAt(this.seed, at[0], at[1]);
      if (item !== JBW_EMPTY && item !== JBW_WALL) out.push(at[0], at[1], item, this.surface(at[0], at[1]));
    }
    const packed = Float64Array.from(out);
    if (this.blocks.size >= BLOCK_CACHE_LIMIT) {
      const oldest = this.blocks.keys().next().value;
      if (oldest !== undefined) this.blocks.delete(oldest);
    }
    this.blocks.set(key, packed);
    return packed;
  }

  private rebuildItems(world: OrchardWorld, ax: number, ay: number, step: number, coverage: ViewCoverage): void {
    const counts = this.counts;
    for (const kind of ITEM_KINDS) counts.set(kind, 0);
    this.decalCount = 0;
    this.beaconCount = 0;
    this.frameX = ax;
    this.frameY = this.surface(ax, ay);
    this.frameZ = ay;
    const boxes = this.debugOn ? this.overlay?.cellBoxes() ?? null : null;
    this.boxes = boxes;
    boxes?.begin();
    this.fieldProjection.copy(this.camera.projectionMatrix);
    this.fieldProjection.elements[0] = (this.fieldProjection.elements[0] ?? 1) / 1.15;
    this.fieldProjection.elements[5] = (this.fieldProjection.elements[5] ?? 1) / 1.15;
    this.fieldProjection.multiply(this.camera.matrixWorldInverse);
    this.fieldFrustum.setFromProjectionMatrix(this.fieldProjection);
    const reach = Math.min(DRAW_RADIUS, this.camera.far);
    const firstX = Math.floor((coverage.x - reach) / JBW_BLOCK);
    const lastX = Math.floor((coverage.x + reach) / JBW_BLOCK);
    const firstY = Math.floor((coverage.z - reach) / JBW_BLOCK);
    const lastY = Math.floor((coverage.z + reach) / JBW_BLOCK);
    for (let by = firstY; by <= lastY; by++) {
      for (let bx = firstX; bx <= lastX; bx++) {
        this.fieldSphere.center.set((bx + 0.5) * JBW_BLOCK, 2, (by + 0.5) * JBW_BLOCK);
        this.fieldSphere.radius = JBW_BLOCK + 7;
        if (!this.fieldFrustum.intersectsSphere(this.fieldSphere)) continue;
        const packed = this.blockAt(bx, by);
        for (let i = 0; i < packed.length; i += 4) {
          const wx = packed[i] ?? 0;
          const wy = packed[i + 1] ?? 0;
          const item = packed[i + 2] ?? 0;
          const distance = this.fieldSphere.center.set(wx, 2, wy).distanceToSquared(this.camera.position);
          const area = item === JBW_TREE ? 12 : item === JBW_WALL ? 1.6 : 4;
          if (distance > 520 * 520 && area * this.pixelScale * this.pixelScale / distance < 0.7) continue;
          if (jbwCollectible(item) && this.clog.depleted(wx, wy, step)) continue;
          this.write(world, counts, item, wx, wy, packed[i + 3] ?? 0);
        }
      }
    }
    for (const kind of ITEM_KINDS) {
      const mesh = world.items.get(kind);
      if (!mesh) continue;
      mesh.count = counts.get(kind) ?? 0;
      mesh.instanceMatrix.clearUpdateRanges();
      mesh.instanceMatrix.addUpdateRange(0, mesh.count * 16);
      mesh.instanceMatrix.needsUpdate = true;
    }
    world.beacon.mesh.count = this.beaconCount;
    world.beacon.mesh.instanceMatrix.needsUpdate = true;
    world.beacon.pool.count = this.beaconCount;
    world.beacon.pool.instanceMatrix.needsUpdate = true;
    world.decals.count = this.decalCount;
    world.decals.instanceMatrix.needsUpdate = true;
    this.decalRipeness = Number.NaN;
    this.paintDecals(world);
    for (const [kind, shell] of world.shells) {
      shell.count = counts.get(kind) ?? 0;
      shell.instanceMatrix.clearUpdateRanges();
      shell.instanceMatrix.addUpdateRange(0, shell.count * 16);
      shell.instanceMatrix.needsUpdate = true;
    }
    boxes?.commit();
  }

  private paintDecals(world: OrchardWorld): void {
    if (this.ripeness === this.decalRipeness) return;
    this.decalRipeness = this.ripeness;
    const decals = world.decals;
    for (let i = 0; i < this.decalCount; i++) {
      decals.setColorAt(i, world.decalTint(this.decalKind[i] ?? 0, this.decalColor));
    }
    const tint = decals.instanceColor;
    if (tint) tint.needsUpdate = true;
  }

  private write(
    world: OrchardWorld,
    counts: Map<number, number>,
    item: number,
    wx: number,
    wy: number,
    ground: number,
  ): void {
    const mesh = world.items.get(item);
    if (!mesh) throw new Error(`no orchard mesh for item ${item}`);
    const used = counts.get(item) ?? 0;
    if (used >= mesh.instanceMatrix.count) {
      throw new Error(`orchard item ${item} exceeded capacity ${mesh.instanceMatrix.count}`);
    }
    this.place.set(wx, ground, wy);
    const at = used * 16;
    const matrices = mesh.instanceMatrix.array;
    matrices[at + 12] = wx;
    matrices[at + 13] = ground;
    matrices[at + 14] = wy;
    const shell = world.shells.get(item);
    if (shell) {
      const shells = shell.instanceMatrix.array;
      shells[at + 12] = wx;
      shells[at + 13] = ground;
      shells[at + 14] = wy;
    }
    counts.set(item, used + 1);
    const boxes = this.boxes;
    if (boxes) {
      const tint = world.cellColor;
      const t = item * 3;
      boxes.box(
        wx - this.frameX,
        wy - this.frameZ,
        this.surface(wx - 0.5, wy - 0.5) - this.frameY,
        this.surface(wx + 0.5, wy - 0.5) - this.frameY,
        this.surface(wx - 0.5, wy + 0.5) - this.frameY,
        this.surface(wx + 0.5, wy + 0.5) - this.frameY,
        tint[t] ?? 1,
        tint[t + 1] ?? 1,
        tint[t + 2] ?? 1,
      );
    }
    const distance = this.place.distanceToSquared(this.camera.position);
    if (distance < this.pixelScale * this.pixelScale / 2 && (ITEM_MOTION[item]?.hover ?? 0) > 0) this.markCell(world, item, wx, ground, wy);
    if (item === JBW_TRUFFLE && distance < 520 * 520) this.markScentColumn(world, wx, ground, wy);
  }

  private markCell(world: OrchardWorld, item: number, wx: number, ground: number, wy: number): void {
    const decals = world.decals;
    if (this.decalCount >= decals.instanceMatrix.count) {
      throw new Error(`orchard decals exceeded capacity ${decals.instanceMatrix.count}`);
    }
    this.place.set(wx, ground + DECAL_LIFT, wy);
    this.matrix.compose(this.place, IDENTITY, this.decalScale);
    decals.setMatrixAt(this.decalCount, this.matrix);
    this.decalKind[this.decalCount] = item;
    this.decalCount++;
  }

  private markScentColumn(world: OrchardWorld, wx: number, ground: number, wy: number): void {
    const beacon = world.beacon;
    if (this.beaconCount >= beacon.mesh.instanceMatrix.count) return;
    this.place.set(wx, ground, wy);
    this.matrix.compose(this.place, IDENTITY, this.unit);
    beacon.mesh.setMatrixAt(this.beaconCount, this.matrix);
    this.place.set(wx, ground + DECAL_LIFT, wy);
    this.matrix.compose(this.place, IDENTITY, this.unit);
    beacon.pool.setMatrixAt(this.beaconCount, this.matrix);
    this.beaconCount++;
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<OrchardWorld>): void {
    const wisp = this.wisp;
    if (!wisp) return;

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
      this.itemsDirty = true;
      this.harvest = 0.35;
      wisp.setCell(ax, ay, UNIT_GRID);
      wisp.snap();
    }

    if (this.clog.sync(frame.eventLog)) {
      world.scentField.sync(this.clog.log, this.clog.version);
      this.itemsDirty = true;
    }
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
      this.itemsDirty = true;
      this.bloomPos.set(ev.x, this.surface(ev.x, ev.y) + BLOOM_Y, ev.y);
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

    this.advanceRate(step, credit);
    const target = clamp01(this.rate / HARVEST_FULL);
    this.harvest += (target - this.harvest) * (1 - Math.exp(-HARVEST_CHASE * frame.dtReal));

    const frontAge = time - this.frontAt;
    const frontT = frontAge >= 0 && frontAge <= FRONT_SECONDS ? frontAge / FRONT_SECONDS : 1;
    const wanted = polarity > 0 ? 1 : 0;
    this.ripeness = this.frontFrom + (wanted - this.frontFrom) * smoothstep01(frontT);
    world.setRipeness(this.ripeness);
    this.paintDecals(world);
    world.setFront(
      ax + ay + (frontT * 2 - 1) * FRONT_SPAN,
      envelope(frontAge, FRONT_SECONDS, 1.4) * 1.6,
    );

    world.scentField.frame(step, ax, ay);
    world.setOrigin(ax, ay);
    world.setHour(this.harvest);
    world.setZoom(SHADOW_SPAN);
    world.setBreeze(0.25 + 0.75 * this.harvest);
    if (this.debugOn !== this.debugWas) {
      this.debugWas = this.debugOn;
      this.itemsDirty = true;
      world.paint.uGridGain.value = this.debugOn ? GRID_GAIN : 0;
    }
    if (ax !== wisp.cellX || ay !== wisp.cellY) {
      this.cell.x = ax;
      this.cell.y = ay;
      wisp.hopTo(this.cell, time);
    }
    wisp.face(heading === 0 ? { dx: 0, dy: -1 } : heading === 1 ? { dx: 1, dy: 0 } : heading === 2 ? { dx: 0, dy: 1 } : { dx: -1, dy: 0 });
    wisp.update(frame.dtReal, time);
    wisp.group.position.y = this.surface(wisp.position.x, wisp.position.z);

    const sense = this.sense;
    if (sense) {
      for (let i = 0; i < 8; i++) {
        const dir = SENSE_DIRECTIONS[(2 * heading + i) & 7] ?? [0, 0];
        const wx = ax + dir[0];
        const wy = ay + dir[1];
        const item = jbwItemAt(this.seed, wx, wy);
        const absent = jbwCollectible(item) && this.clog.depleted(wx, wy, step);
        this.energies[i] = absent ? 0 : JBW_ENERGY[item] ?? 0;
      }
      const scent = scentAtCell(
        this.seed,
        this.clog.collected,
        step,
        ax,
        ay,
        world.scentField.kernel,
      );
      sense.update(ax, ay, heading, this.surface, this.energies, ENERGY_PEAK, scent);
    }

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

    world.setTime(time);
    if (this.view) {
      const coverage = this.view.update(rig, frame.dtReal, [wisp.position.x, 0.4, wisp.position.z]);
      world.setView(coverage, this.camera);
      const blockX = Math.floor(coverage.x / JBW_BLOCK);
      const blockY = Math.floor(coverage.z / JBW_BLOCK);
      const stamp = `${Math.floor(rig.state.azimuth * 24)}:${Math.floor(rig.state.polar * 24)}:${Math.floor(Math.log2(rig.state.distance) * 16)}:${Math.floor(this.camera.aspect * 20)}`;
      if (this.itemsDirty || blockX !== this.blockX || blockY !== this.blockY || stamp !== this.fieldStamp) {
        this.rebuildItems(world, ax, ay, step, coverage);
        this.itemsDirty = false;
        this.blockX = blockX;
        this.blockY = blockY;
        this.fieldStamp = stamp;
      }
    }
  }

  protected drop(): void {
    this.wisp?.dispose();
    this.wisp = null;
    this.sense?.dispose();
    this.sense = null;
    this.counts.clear();
    this.blocks.clear();
    this.fieldStamp = '';
    this.bloomAt = -Infinity;
    this.frontAt = -Infinity;
  }
}

export default ((() => new WindfallOrchardScene()) satisfies SceneFactory);
