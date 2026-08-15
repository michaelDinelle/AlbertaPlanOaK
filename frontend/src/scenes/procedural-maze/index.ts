
import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { cellToWorldVec, clampCell, type GridSize } from '@kit/grid';
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
import { buildSunShaft, type SunShaftHandle } from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  CAMERA_DISTANCE,
  CAMERA_POLAR,
  CAMERA_TARGET_BIAS,
  CAMERA_TARGET_Y,
  SUN_DIR,
  blendTokens,
  buildWorld,
  floorTintFor,
  resolveGrid,
  ruinSkin,
  type RuinSkin,
  type WorldHandle,
} from './world';


const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = CAMERA_POLAR;
const CAM_DISTANCE = CAMERA_DISTANCE;
const CAM_TARGET_Y = CAMERA_TARGET_Y;
const CAM_TARGET_BIAS = CAMERA_TARGET_BIAS;
const CAM_YAW_RANGE = 0.6;
const CAM_DISTANCE_THUMB = 25.0;
const CAM_AZIMUTH_THUMB = 0.5;
const CAM_POLAR_THUMB = 0.66;

const BURST_SCALE = 0.82;
const RIPPLE_SCALE = 0.48;
const BURST_PARTICLES = 110;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.46;
const BLOOM_SECONDS = 1.5;
const GROW_SECONDS = 0.8;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;

interface ProceduralMazeLike {
  x?: unknown;
  y?: unknown;
  gx?: unknown;
  gy?: unknown;
}

function readCells(state: FrameState, grid: GridSize, agent: GridCell, goal: GridCell): void {
  const d = state.derived as ProceduralMazeLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const ax = has && typeof d.x === 'number' ? d.x : (state.pose[0] ?? 0);
  const ay = has && typeof d.y === 'number' ? d.y : (state.pose[1] ?? 0);
  const gx = has && typeof d.gx === 'number' ? d.gx : (state.pose[2] ?? 0);
  const gy = has && typeof d.gy === 'number' ? d.gy : (state.pose[3] ?? 0);
  agent.x = Number.isFinite(ax) ? Math.round(ax) : 0;
  agent.y = Number.isFinite(ay) ? Math.round(ay) : 0;
  goal.x = Number.isFinite(gx) ? Math.round(gx) : 0;
  goal.y = Number.isFinite(gy) ? Math.round(gy) : 0;
  clampCell(agent, grid);
  clampCell(goal, grid);
}

export function readGoalCell(state: FrameState, grid: GridSize, out: GridCell): GridCell {
  const d = state.derived as ProceduralMazeLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const gx = has && typeof d.gx === 'number' ? d.gx : (state.pose[2] ?? 0);
  const gy = has && typeof d.gy === 'number' ? d.gy : (state.pose[3] ?? 0);
  out.x = Number.isFinite(gx) ? Math.round(gx) : 0;
  out.y = Number.isFinite(gy) ? Math.round(gy) : 0;
  return clampCell(out, grid);
}

class OvergrownRuinsScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private shaft: SunShaftHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry | null = null;
  private skin: RuinSkin | null = null;
  private seed = 0;

  private growAt = -Infinity;
  private bloomAt = -Infinity;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly prevGoalCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly litTint = new THREE.Color();
  private readonly restoreTint = new THREE.Color();
  private litX = -1;
  private litY = -1;
  private gold = TOKENS.celebrationGold;
  private moteColor = TOKENS.celebrationGold;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'overgrownRuinsScene';

    const grid = resolveGrid(ctx.geometry, ctx.seed);
    this.grid = grid;
    this.seed = ctx.seed;

    const palette = ctx.tokens.palettes.noon;
    const skin = ruinSkin(palette, ctx.tokens);
    this.skin = skin;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.gold = ctx.tokens.celebrationGold;
    this.moteColor = blendTokens(
      ctx.tokens.celebrationGold,
      ctx.tokens.palettes.dusk.groundA,
      0.3,
    ).getHex();
    this.litTint.copy(blendTokens(skin.flag, skin.sunWarm, 0.5));

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    const beamLight = world.rig.points[0];
    if (beamLight) beamLight.color.set(skin.sunWarm);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const bloomLight = world.rig.points[2];
    if (bloomLight) bloomLight.color.set(this.moteColor);

    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        radius: 0.22,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'ruinWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const shaft = buildSunShaft({
      tier: ctx.tier,
      palette,
      skin,
      sunDir: SUN_DIR,
      name: 'sunShaft',
    });
    this.shaft = shaft;
    this.scene.add(shaft.group);

    const thumb = ctx.tier === 'thumb';
    const bias = thumb ? 0 : CAM_TARGET_BIAS;
    const rig = makeOrbitRig({
      canvas: ctx.canvas,
      target: [Math.sin(CAM_AZIMUTH) * bias, CAM_TARGET_Y, Math.cos(CAM_AZIMUTH) * bias],
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: thumb ? CAM_AZIMUTH_THUMB : CAM_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0012,
      name: 'ruinOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'ruinCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    const overlayAgent: GridCell = { x: 0, y: 0 };
    const overlayGoal: GridCell = { x: 0, y: 0 };
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayAgent;
      },
      goalCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayGoal;
      },
      name: 'ruinGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    shaft.setState(1);
    shaft.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const shaft = this.shaft;
    const pool = this.pool;
    const rig = this.rig;
    const grid = this.grid;
    if (!ctx || !world || !wisp || !shaft || !pool || !rig || !grid) return;

    const time = ctx.time();
    const state = frame.state;
    const agent = this.agentCell;
    const goal = this.goalCell;

    pool.update(frame.dtReal, time);

    readCells(state, grid, agent, goal);

    if (frame.scrubbed) {
      pool.clear();
      this.bloomAt = -Infinity;
      this.growAt = -Infinity;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      rig.snap();
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        const prev = frame.prev;
        if (prev) readGoalCell(prev, grid, this.prevGoalCell);
        else {
          this.prevGoalCell.x = agent.x;
          this.prevGoalCell.y = agent.y;
        }
        cellToWorldVec(this.prevGoalCell.x, this.prevGoalCell.y, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.moteColor, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        shaft.flare(0.9);
        this.bloomAt = time;
        this.growAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
    } else if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      wisp.hopTo(agent, time);
    }
    wisp.update(frame.dtReal, time);

    cellToWorldVec(goal.x, goal.y, grid, shaft.root.position);
    const grow = growth(time - this.growAt);
    shaft.setState(grow);
    shaft.setTime(time);
    shaft.update(frame.dtReal, time);
    this.relightFloor(goal.x, goal.y);
    world.setGoalPosition(shaft.root.position.x, 1.5, shaft.root.position.z);

    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const beamLight = world.rig.points[0];
    if (beamLight) {
      beamLight.position.set(shaft.root.position.x, shaft.lightHeight, shaft.root.position.z);
      beamLight.intensity = (0.22 + 0.6 * grow) * (1 + 0.9 * bloom);
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.16 + 0.8 * bloom;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 2.2 * bloom;
    }

    world.setTime(time);

    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
  }

  private relightFloor(x: number, y: number): void {
    if (x === this.litX && y === this.litY) return;
    const world = this.world;
    const grid = this.grid;
    const skin = this.skin;
    if (!world || !grid || !skin) return;
    if (this.litX >= 0) {
      floorTintFor(grid, skin, this.litX, this.litY, this.seed, this.restoreTint);
      world.field.setFloorColor(this.litX, this.litY, this.restoreTint);
    }
    world.field.setFloorColor(x, y, this.litTint);
    this.litX = x;
    this.litY = y;
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
    this.shaft?.dispose();
    this.shaft = null;
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
    this.grid = null;
    this.skin = null;
    this.seed = 0;
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
    this.litX = -1;
    this.litY = -1;
  }
}


export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.13 * Math.sin(Math.PI * t) * (1 - t));
}

const createScene: SceneFactory = () => new OvergrownRuinsScene();
export default createScene;
