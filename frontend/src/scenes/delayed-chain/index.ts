
import { DELAYEDCHAIN_ACTION, DELAYEDCHAIN_POS } from '@sim/envs/delayedchain';
import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { cellToWorld, cellToWorldVec, type GridSize } from '@kit/grid';
import { makeOrbitRig, type OrbitRig } from '@kit/rigs';
import { TOKENS } from '@kit/tokens';
import type {
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import {
  LAMP_Y,
  buildLanternRoad,
  buildMilestone,
  buildWispGlow,
  type LanternRoadHandle,
  type MilestoneHandle,
  type WispGlowHandle,
} from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  SHADOW_RADIUS,
  buildWorld,
  clamp01,
  goalCellOf,
  resolveGrid,
  type WorldHandle,
} from './world';


const CAM_DISTANCE = 8.4;
const CAM_POLAR = 1.43;
const CAM_TARGET_Y = 1.3;
const CAM_LOOK_AHEAD = 0.9;
const CAM_FOLLOW_RATE = 5.5;

export const WAVE_TRAVEL = 1.35;
export const IGNITE_ATTACK = 0.14;
export const IGNITE_HOLD = 0.85;
export const IGNITE_GUTTER = 2.9;
export const LANTERN_SPAN = IGNITE_ATTACK + IGNITE_HOLD + IGNITE_GUTTER;
export const WAVE_SPAN = WAVE_TRAVEL + LANTERN_SPAN;

const BURST_SCALE = 1.25;
const RIPPLE_SCALE = 0.9;
const BURST_PARTICLES = 150;
export const PULSE_STRENGTH = 0.6;
const SURGE_SECONDS = 2.6;
const WISP_FLARE = 0.32;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;
const WISP_LEAN = 0.15;

function readState(state: FrameState, grid: GridSize, out: GridCell): number {
  const pos = Math.round(state.pose[DELAYEDCHAIN_POS] ?? 0);
  out.x = pos < 0 ? 0 : pos > grid.w - 1 ? grid.w - 1 : pos;
  out.y = 0;
  return (state.pose[DELAYEDCHAIN_ACTION] ?? 0) >= 0.5 ? 1 : 0;
}

class LongNightRoadScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private road: LanternRoadHandle | null = null;
  private milestone: MilestoneHandle | null = null;
  private wisp: Wisp | null = null;
  private glow: WispGlowHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({});
  private goalCell = 0;

  private waveAt = -Infinity;
  private surgeAt = -Infinity;
  private burstAt = -Infinity;
  private waveLit = false;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly overlayCell: GridCell = { x: 0, y: 0 };
  private readonly overlayGoal: GridCell = { x: 0, y: 0 };
  private readonly burstPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly shadowCenter = { x: 0, y: 0.3, z: 0 };
  private shadowCell = Number.NaN;
  private camX = 0;
  private gold = 0xffd166;
  private flame = 0xffd166;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'longNightRoadScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.goalCell = goalCellOf(grid);

    const palette = ctx.tokens.palettes.night;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.uiCyan;
    this.gold = ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);
    this.flame = world.skin.flame.getHex();

    const wispLight = world.rig.points[0];
    if (wispLight) wispLight.color.set(accent);
    const auroraLight = world.rig.points[1];
    if (auroraLight) {
      auroraLight.color.set(accent);
      auroraLight.position.set(cellToWorld(this.goalCell - 2, 0, grid)[0], 3.4, -1.5);
      auroraLight.intensity = 3.6;
    }
    const waveLight = world.rig.points[2];
    if (waveLight) waveLight.color.set(this.flame);
    const goalLight = world.rig.points[3];
    if (goalLight) {
      goalLight.color.set(this.gold);
      goalLight.position.set(cellToWorld(this.goalCell, 0, grid)[0], LAMP_Y, 0);
    }

    const road = buildLanternRoad({
      palette,
      skin: world.skin,
      grid,
      goalCell: this.goalCell,
      seed: ctx.seed,
      accent,
      name: 'lanternRoad',
    });
    this.road = road;
    this.scene.add(road.group);
    for (let i = 0; i < road.count; i++) road.setGlow(i, 0);
    road.commit();

    const milestone = buildMilestone({
      palette,
      skin: world.skin,
      grid,
      cell: 0,
      name: 'startMilestone',
    });
    this.milestone = milestone;
    this.scene.add(milestone.group);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.165,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'roadWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);
    const kitHalo = wisp.group.getObjectByName('oakWispHalo');
    if (kitHalo) kitHalo.visible = false;
    wisp.snap();

    const glow = buildWispGlow({
      color: accent,
      radius: 1.05,
      intensity: 0.72,
      name: 'wispGlow',
    });
    this.glow = glow;
    this.scene.add(glow.mesh);

    const rig = makeOrbitRig({
      canvas: ctx.canvas,
      target: [0, CAM_TARGET_Y, 0],
      distance: CAM_DISTANCE,
      azimuth: CAMERA_AZIMUTH,
      polar: CAM_POLAR,
      limits: {
            minPolar: 1.15,
            maxPolar: 1.52,
            minAzimuth: CAMERA_AZIMUTH - 0.45,
            maxAzimuth: CAMERA_AZIMUTH + 0.45,
            minDistance: 4.5,
            maxDistance: 16,
          },
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.0035,
      zoomSpeed: 0.0012,
      name: 'roadOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'roadCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        readState(state, grid, this.overlayCell);
        return this.overlayCell;
      },
      goalCell: () => {
        this.overlayGoal.x = this.goalCell;
        this.overlayGoal.y = 0;
        return this.overlayGoal;
      },
      name: 'roadGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    world.aurora.setGain(1);
    road.setTime(0);
    milestone.setTime(0);
    wisp.setCell(0, 0, grid);
    this.camX = wisp.position.x;
    rig.setTarget(this.camX + CAM_LOOK_AHEAD, CAM_TARGET_Y, 0);
    rig.snap();
    glow.mesh.position.copy(wisp.position);
    glow.mesh.position.y += ctx.tokens.wispHeight;
    glow.mesh.quaternion.copy(this.camera.quaternion);
    glow.setGain(1);
    this.applyShadowFocus(wisp.position.x, world);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const road = this.road;
    const wisp = this.wisp;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !road || !wisp || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const agent = this.agentCell;

    pool.update(frame.dtReal, time);

    const lastA = readState(state, grid, agent);

    let snapCamera = false;
    if (frame.scrubbed) {
      pool.clear();
      this.waveAt = -Infinity;
      this.surgeAt = -Infinity;
      this.burstAt = -Infinity;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      snapCamera = true;
    }

    let teleported = false;
    let paid = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        this.burstPos.copy(road.endLamp);
        cellToWorldVec(this.goalCell, 0, grid, this.groundPos);
        pool.burst(this.burstPos, this.flame, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.reset();
        pool.pulse.fire(PULSE_STRENGTH);
        road.flare(1);
        paid = true;
        this.waveAt = time;
        this.surgeAt = time;
        this.burstAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      snapCamera = true;
    }
    if (paid) wisp.flare(WISP_FLARE);

    if (agent.x !== wisp.cellX) wisp.hopTo(agent, time);
    wisp.avatar.rotation.z = lastA === 1 ? -WISP_LEAN : WISP_LEAN;
    wisp.update(frame.dtReal, time);

    const age = time - this.waveAt;
    const active = age >= 0 && age <= WAVE_SPAN;
    if (active || this.waveLit) {
      const count = road.count;
      for (let i = 0; i < count; i++) {
        const g = active ? lanternGlow(age, i, count) : 0;
        road.setGlow(i, g > 0 ? g * flicker(time, i) : 0);
      }
      road.commit();
      this.waveLit = active;
    }

    const burst = envelope(time - this.burstAt, 1.6, 2.2);
    const surge = envelope(time - this.surgeAt, SURGE_SECONDS, 1.5);

    const glow = this.glow;
    if (glow) {
      glow.mesh.position.copy(wisp.position);
      glow.mesh.position.y += TOKENS.wispHeight;
      glow.mesh.quaternion.copy(this.camera.quaternion);
      glow.setGain(1 + 1.1 * burst);
    }

    const wispLight = world.rig.points[0];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.62 + 1.4 * burst;
    }
    const waveLight = world.rig.points[2];
    if (waveLight) {
      const front = active ? 1 - clamp01(age / WAVE_TRAVEL) : 0;
      waveLight.position.set(road.lanternX(front * (road.count - 1)), 1.15, -0.2);
      waveLight.intensity = active ? 2.6 * envelope(age, WAVE_TRAVEL + 0.45, 1.1) : 0;
    }
    const goalLight = world.rig.points[3];
    if (goalLight) goalLight.intensity = 3.2 * burst;

    world.aurora.setGain(1 + 1.35 * surge);
    world.setTime(time);
    road.setTime(time);
    road.update(frame.dtReal, time);
    this.milestone?.setTime(time);

    {
      const wispX = wisp.position.x;
      if (snapCamera) this.camX = wispX;
      else this.camX += (wispX - this.camX) * (1 - Math.exp(-CAM_FOLLOW_RATE * Math.max(0, frame.dtReal)));
      rig.setTarget(this.camX + CAM_LOOK_AHEAD, CAM_TARGET_Y, 0);
    }
    if (snapCamera) rig.snap();
    else rig.update(frame.dtReal);
    this.applyShadowFocus(wisp.position.x, world);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.world?.setPixelRatio(size.dpr);
    this.pool?.setPixelRatio(size.dpr);
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.milestone?.dispose();
    this.milestone = null;
    this.road?.dispose();
    this.road = null;
    this.glow?.dispose();
    this.glow = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.waveAt = -Infinity;
    this.surgeAt = -Infinity;
    this.burstAt = -Infinity;
    this.waveLit = false;
    this.shadowCell = Number.NaN;
  }

  private applyShadowFocus(x: number, world: WorldHandle): void {
    const cell = Math.round(x);
    if (cell === this.shadowCell) return;
    this.shadowCell = cell;
    this.shadowCenter.x = cell;
    world.rig.setShadowFocus(this.shadowCenter, SHADOW_RADIUS);
  }
}


export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

export function lanternGlow(age: number, index: number, count: number): number {
  if (!(age >= 0)) return 0;
  const n = Math.max(1, count - 1);
  const i = index < 0 ? 0 : index > n ? n : index;
  const back = (n - i) / n;
  const local = age - back * WAVE_TRAVEL;
  if (local < 0 || local > LANTERN_SPAN) return 0;
  if (local < IGNITE_ATTACK) {
    const t = local / IGNITE_ATTACK;
    return t * t * (3 - 2 * t);
  }
  const fall = local - IGNITE_ATTACK;
  if (fall < IGNITE_HOLD) return 1;
  const g = (fall - IGNITE_HOLD) / IGNITE_GUTTER;
  const e = Math.exp(-2.1 * g) * (1 - g);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

export function flicker(time: number, index: number): number {
  return 0.87 + 0.13 * Math.sin(time * 9.3 + index * 2.3994);
}

const createScene: SceneFactory = () => new LongNightRoadScene();
export default createScene;
