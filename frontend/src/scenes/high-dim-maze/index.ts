
import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec, clampCell, type GridSize } from '@kit/grid';
import { blendTokens, clamp01, envelope } from '@kit/mix';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { Wisp, yawFromGridDelta } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { HIGHDIMMAZE_ACTION, HIGHDIMMAZE_ACTIONS, HIGHDIMMAZE_DASH, HIGHDIMMAZE_GX, HIGHDIMMAZE_GY, HIGHDIMMAZE_HEADING, HIGHDIMMAZE_STAY, HIGHDIMMAZE_X, HIGHDIMMAZE_Y } from '@sim/envs/highdimmaze';
import { buildGust, buildOasis, buildScarf, type GustHandle, type OasisHandle, type ScarfHandle } from './actors';
import { manifest } from './manifest';
import { CAMERA_AZIMUTH, buildWorld, canyonSkin, resolveGrid, type WorldHandle } from './world';


const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.86;
const CAM_DISTANCE = 17;
const CAM_TARGET_Y = 0.62;

const BURST_SCALE = 0.84;
const RIPPLE_SCALE = 0.52;
const BURST_PARTICLES = 110;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.34;
const BLOOM_SECONDS = 1.5;
const GROW_SECONDS = 0.8;
const GROW_BOUNCE = 0.65;
const WAKE_SECONDS = 0.62;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;

const FACE_RATE = 12;

const PROXIMITY_CELLS = 4;

interface Reading {
  agent: GridCell;
  goal: GridCell;
  lastDir: number;
  action: number;
}

function read(state: FrameState, grid: GridSize, out: Reading): Reading {
  out.agent.x = Math.round(state.pose[HIGHDIMMAZE_X] ?? 0);
  out.agent.y = Math.round(state.pose[HIGHDIMMAZE_Y] ?? 0);
  out.goal.x = Math.round(state.pose[HIGHDIMMAZE_GX] ?? 0);
  out.goal.y = Math.round(state.pose[HIGHDIMMAZE_GY] ?? 0);
  clampCell(out.agent, grid);
  clampCell(out.goal, grid);
  const dir = Math.round(state.pose[HIGHDIMMAZE_HEADING] ?? 0);
  out.lastDir = ((dir % 4) + 4) % 4;
  const a = Math.trunc(state.pose[HIGHDIMMAZE_ACTION] ?? 0);
  out.action = ((a % HIGHDIMMAZE_ACTIONS) + HIGHDIMMAZE_ACTIONS) % HIGHDIMMAZE_ACTIONS;
  return out;
}

class SlotCanyonsScene extends StageScene<WorldHandle> {
  readonly manifest = manifest;
  protected readonly stageName = 'slotCanyons';

  private wisp: Wisp | null = null;
  private scarf: ScarfHandle | null = null;
  private gust: GustHandle | null = null;
  private oasis: OasisHandle | null = null;
  private settleDust: AmbientParticlesHandle | null = null;
  private grid: GridGeometry = resolveGrid({});

  private growAt = -Infinity;
  private bloomAt = -Infinity;
  private dashAt = -Infinity;

  private readonly now: Reading = { agent: { x: 0, y: 0 }, goal: { x: 0, y: 0 }, lastDir: 1, action: 0 };
  private readonly before: Reading = { agent: { x: 0, y: 0 }, goal: { x: 0, y: 0 }, lastDir: 1, action: 0 };
  private readonly lastGoal: GridCell = { x: -1, y: -1 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly dashFrom = new THREE.Vector3();
  private readonly dashTo = new THREE.Vector3();
  private readonly wakeFrom = new THREE.Vector3();
  private readonly wakeTo = new THREE.Vector3();
  private scarfYaw = 0;
  private spray = 0x5ee0c8;
  private gold = 0xffd166;

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const palette = ctx.tokens.palettes.afternoon;
    const skin = canyonSkin(palette, ctx.tokens);
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    const ember = accent;
    this.spray = blendTokens(skin.water, skin.waterDeep, 0.4).getHex();
    this.gold = ctx.tokens.celebrationGold;

    const oasisLight = world.rig.points[0];
    if (oasisLight) oasisLight.color.set(skin.shaft);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const bloomLight = world.rig.points[2];
    if (bloomLight) bloomLight.color.set(this.spray);

    const wisp = new Wisp(
      {
        color: ember,
        radius: 0.165,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'canyonWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const scarf = buildScarf({
      color: blendTokens(accent, TOKENS.palettes.dusk.skyHorizon, 0.3),
      height: ctx.tokens.wispHeight * 0.92,
      name: 'canyonScarf',
    });
    this.scarf = scarf;
    wisp.avatar.add(scarf.group);

    const gust = buildGust({
      skin,
      color: blendTokens(accent, palette.sun, 0.5),
      name: 'canyonGust',
    });
    this.gust = gust;
    this.scene.add(gust.group);

    const settleDust = makeAmbientParticles({
      preset: 'dust',
      count: 110,
      bounds: { x: 0, y: 0, z: 0, halfX: 0.4, halfY: 0.28, halfZ: 0.4 },
      seed: ctx.seed ^ 0x5e77,
      color: skin.dust,
      size: 0.9,
      intensity: 1.4,
      opacity: 0.55,
      rise: -0.2,
      swirl: 0.06,
      name: 'canyonSettleDust',
    });
    settleDust.setDensity(0);
    this.settleDust = settleDust;
    this.scene.add(settleDust.points);

    const oasis = buildOasis({ palette, skin, name: 'oasisAlcove' });
    this.oasis = oasis;
    this.scene.add(oasis.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    return {
      canvas: ctx.canvas,
      target: [0, CAM_TARGET_Y, 0],
      distance: CAM_DISTANCE,
      azimuth: CAM_AZIMUTH,
      polar: CAM_POLAR,
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0012,
      name: 'canyonOrbitRig',
    };
  }

  protected poolOptions(ctx: SceneContext): CelebrationPoolOptions {
    return {
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'canyonCelebrations',
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    const grid = this.grid;
    const overlayReading: Reading = {
      agent: { x: 0, y: 0 },
      goal: { x: 0, y: 0 },
      lastDir: 1,
      action: 0,
    };
    return {
      camera: this.camera,
      agentCell: (state) => read(state, grid, overlayReading).agent,
      goalCell: (state) => read(state, grid, overlayReading).goal,
      name: 'canyonGroundTruth',
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    this.wisp?.setCell(0, 0, this.grid);
    this.oasis?.setState(1, 0);
    this.oasis?.setTime(0);
    this.scarf?.setState(0, 1, 0, 0);
    this.gust?.clear();
  }

  protected tick(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const scarf = this.scarf;
    const gust = this.gust;
    const oasis = this.oasis;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !scarf || !gust || !oasis || !pool || !rig) return;

    const time = ctx.time();
    const grid = this.grid;
    const now = read(frame.state, grid, this.now);
    const agent = now.agent;
    const goal = now.goal;

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      gust.clear();
      this.bloomAt = -Infinity;
      this.growAt = -Infinity;
      this.clearDash(agent);
      wisp.setCell(agent.x, agent.y, grid);
      this.scarfYaw = facingYaw(now.lastDir);
      wisp.face(this.scarfYaw);
      wisp.snap();
      rig.snap();
      this.lastGoal.x = goal.x;
      this.lastGoal.y = goal.y;
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        this.bloomCell(frame, this.groundPos);
        this.bloomPos.copy(this.groundPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.spray, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        oasis.flare(0.8);
        this.bloomAt = time;
        this.growAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      this.scarfYaw = facingYaw(now.lastDir);
      wisp.face(this.scarfYaw);
      wisp.snap();
      gust.clear();
      this.clearDash(agent);
      rig.snap();
    }

    const moved = agent.x !== wisp.cellX || agent.y !== wisp.cellY;
    const dashing = now.action === HIGHDIMMAZE_DASH ? 1 : 0;
    if (dashing > 0) {
      cellToWorldVec(wisp.cellX, wisp.cellY, grid, this.dashFrom);
      cellToWorldVec(agent.x, agent.y, grid, this.dashTo);
      if (moved && isDashSegment(this.dashFrom.x, this.dashFrom.z, this.dashTo.x, this.dashTo.z)) {
        this.wakeFrom.copy(this.dashFrom);
        this.wakeTo.copy(this.dashTo);
        this.dashAt = time;
      }
    }
    if (moved) wisp.hopTo(agent, time);
    wisp.face(facingDelta(now.lastDir));
    wisp.update(frame.dtReal, time);

    this.scarfYaw = dampAngle(this.scarfYaw, facingYaw(now.lastDir), frame.dtReal);
    const staying = now.action === HIGHDIMMAZE_STAY ? 1 : 0;
    const airborne = wisp.motion.airborne ? 1 - wisp.motion.progress : 0;
    const speed = clamp01(airborne * (0.55 + 0.45 * dashing) + 0.35 * dashing);
    scarf.setState(this.scarfYaw, staying, speed, time);

    const wake = envelope(time - this.dashAt, WAKE_SECONDS, 2.6);
    if (dashing > 0 && isDashSegment(this.dashFrom.x, this.dashFrom.z, this.dashTo.x, this.dashTo.z)) {
      gust.setLines(
        this.dashFrom.x,
        this.dashFrom.z,
        this.dashTo.x,
        this.dashTo.z,
        ctx.tokens.wispHeight * 0.8,
        wisp.motion.progress,
        clamp01(0.35 + 0.65 * airborne),
      );
    } else {
      gust.setLines(0, 0, 0, 0, 0, 1, 0);
    }
    gust.setWake(this.wakeFrom.x, this.wakeFrom.z, this.wakeTo.x, this.wakeTo.z, wake);
    gust.setSettle(wisp.position.x, wisp.position.z, staying, time);

    const settleDust = this.settleDust;
    if (settleDust) {
      settleDust.points.position.set(wisp.position.x, ctx.tokens.wispHeight * 0.7, wisp.position.z);
      settleDust.setDensity(staying);
      settleDust.setTime(time);
    }

    cellToWorldVec(goal.x, goal.y, grid, oasis.root.position);
    const grow = growth(time - this.growAt);
    oasis.setState(grow, nearness(agent, goal));
    oasis.setTime(time);
    oasis.update(frame.dtReal, time);

    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const oasisLight = world.rig.points[0];
    if (oasisLight) {
      oasisLight.position.set(oasis.root.position.x, oasis.crownHeight, oasis.root.position.z);
      oasisLight.intensity = (0.2 + 0.5 * nearness(agent, goal)) * grow;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.18 + 0.9 * bloom;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 2.4 * bloom;
    }

    world.setTime(time);

    rig.update(frame.dtReal);

    this.lastGoal.x = goal.x;
    this.lastGoal.y = goal.y;
  }

  private clearDash(agent: GridCell): void {
    this.dashAt = -Infinity;
    cellToWorldVec(agent.x, agent.y, this.grid, this.dashFrom);
    this.dashTo.copy(this.dashFrom);
    this.wakeFrom.copy(this.dashFrom);
    this.wakeTo.copy(this.dashFrom);
  }

  private bloomCell(frame: FrameUpdate, out: THREE.Vector3): void {
    const grid = this.grid;
    if (frame.prev) {
      const before = read(frame.prev, grid, this.before);
      cellToWorldVec(before.goal.x, before.goal.y, grid, out);
      return;
    }
    if (this.lastGoal.x >= 0) {
      cellToWorldVec(this.lastGoal.x, this.lastGoal.y, grid, out);
      return;
    }
    cellToWorldVec(this.now.goal.x, this.now.goal.y, grid, out);
  }

  resize(size: SceneSize): void {
    super.resize(size);
    this.settleDust?.setPixelRatio(size.dpr);
  }

  protected drop(): void {
    this.oasis?.dispose();
    this.oasis = null;
    this.settleDust?.dispose();
    this.settleDust = null;
    this.gust?.dispose();
    this.gust = null;
    this.scarf?.dispose();
    this.scarf = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
    this.dashAt = -Infinity;
    this.scarfYaw = 0;
    this.lastGoal.x = -1;
    this.lastGoal.y = -1;
  }
}


const FACING: readonly { dx: number; dy: number }[] = Object.freeze([
  { dx: 0, dy: -1 },
  { dx: 0, dy: 1 },
  { dx: -1, dy: 0 },
  { dx: 1, dy: 0 },
]);

export function facingDelta(lastDir: number): { dx: number; dy: number } {
  return FACING[((lastDir % 4) + 4) % 4] ?? { dx: 0, dy: 1 };
}

export function facingYaw(lastDir: number): number {
  const d = facingDelta(lastDir);
  return yawFromGridDelta(d.dx, d.dy);
}

export function isDashSegment(fromX: number, fromZ: number, toX: number, toZ: number): boolean {
  const dx = Math.abs(toX - fromX) / CELL;
  const dz = Math.abs(toZ - fromZ) / CELL;
  const eps = 1e-6;
  if (dx > eps && dz > eps) return false;
  const cells = dx > dz ? dx : dz;
  return cells >= 1 - eps && cells <= 2 + eps;
}

export function dampAngle(current: number, target: number, dt: number): number {
  const k = 1 - Math.exp(-FACE_RATE * Math.max(0, dt));
  let d = target - current;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return current + d * k;
}

export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  return 1 - Math.pow(1 - t, 3) + GROW_BOUNCE * t * t * (1 - t);
}

export function nearness(agent: GridCell, goal: GridCell): number {
  const d = Math.max(Math.abs(agent.x - goal.x), Math.abs(agent.y - goal.y));
  return d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
}

const createScene: SceneFactory = () => new SlotCanyonsScene();
export default createScene;
