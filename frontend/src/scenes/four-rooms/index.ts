
import { FOURROOMS_GX, FOURROOMS_GY, FOURROOMS_X, FOURROOMS_Y } from '@sim/envs/fourrooms';
import { cellToWorldVec, clampCell, type GridSize } from '@kit/grid';
import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { blendTokens, envelope, shade } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { buildLantern, type LanternHandle } from './actors';
import { manifest } from './manifest';
import { CAMERA_AZIMUTH, buildWorld, courtyardSkin, resolveGrid, type WorldHandle } from './world';


const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.75;
const CAM_DISTANCE = 17.8;
const CAM_TARGET_Y = 0.5;

const BURST_SCALE = 0.85;
const RIPPLE_SCALE = 0.5;
const BURST_PARTICLES = 110;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.42;
const BLOOM_SECONDS = 1.5;
const GROW_SECONDS = 0.72;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.17;

const OPEN_BASE = 0.3;
const OPEN_BREATH = 0.18;
const OPEN_PROXIMITY = 0.4;
const PROXIMITY_CELLS = 4;

function readCells(state: FrameState, grid: GridSize, agent: GridCell, goal: GridCell): void {
  agent.x = Math.round(state.pose[FOURROOMS_X] ?? 0);
  agent.y = Math.round(state.pose[FOURROOMS_Y] ?? 0);
  goal.x = Math.round(state.pose[FOURROOMS_GX] ?? 0);
  goal.y = Math.round(state.pose[FOURROOMS_GY] ?? 0);
  clampCell(agent, grid);
  clampCell(goal, grid);
}

class FourSeasonsCourtyardScene extends StageScene<WorldHandle> {
  readonly manifest = manifest;
  protected readonly stageName = 'fourSeasonsCourtyard';

  private wisp: Wisp | null = null;
  private lantern: LanternHandle | null = null;
  private grid: GridGeometry = resolveGrid({});

  private growAt = -Infinity;
  private bloomAt = -Infinity;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private blossom = 0xf7a8b8;
  private gold = 0xffd166;

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const palette = ctx.tokens.palettes.noon;
    const skin = courtyardSkin(palette, ctx.tokens);
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.blossom = ctx.tokens.accents['chain'] ?? ctx.tokens.celebrationGold;
    this.gold = ctx.tokens.celebrationGold;
    const flame = blendTokens(ctx.tokens.celebrationGold, palette.sun, 0.22).getHex();

    const lanternLight = world.rig.points[0];
    if (lanternLight) lanternLight.color.set(flame);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const bloomLight = world.rig.points[2];
    if (bloomLight) bloomLight.color.set(this.blossom);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.21,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'courtyardWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const lantern = buildLantern({
      palette,
      blossom: shade(blendTokens(this.blossom, palette.groundB, 0.12), 0.3),
      flame,
      stem: skin.seasons[1]?.hedge ?? palette.groundA,
      name: 'lanternFlower',
    });
    this.lantern = lantern;
    this.scene.add(lantern.group);
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
      name: 'courtyardOrbitRig',
    };
  }

  protected poolOptions(ctx: SceneContext): CelebrationPoolOptions {
    return {
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'courtyardCelebrations',
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    const grid = this.grid;
    const overlayAgent: GridCell = { x: 0, y: 0 };
    const overlayGoal: GridCell = { x: 0, y: 0 };
    return {
      camera: this.camera,
      agentCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayAgent;
      },
      goalCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayGoal;
      },
      name: 'courtyardGroundTruth',
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    this.wisp?.setCell(0, 0, this.grid);
    this.lantern?.setState(OPEN_BASE, 1);
    this.lantern?.setTime(0);
  }

  protected tick(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const lantern = this.lantern;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !lantern || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
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
        cellToWorldVec(agent.x, agent.y, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.blossom, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        lantern.flare(0.7);
        this.bloomAt = time;
        this.growAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }
    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      rig.snap();
    }

    if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      wisp.hopTo(agent, time);
    }
    wisp.update(frame.dtReal, time);

    cellToWorldVec(goal.x, goal.y, grid, lantern.root.position);
    const grow = growth(time - this.growAt);
    const open = opening(time, agent, goal);
    lantern.setState(open, grow);
    lantern.setTime(time);
    lantern.update(frame.dtReal, time);

    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const lanternLight = world.rig.points[0];
    if (lanternLight) {
      lanternLight.position.set(
        lantern.root.position.x,
        lantern.crownHeight,
        lantern.root.position.z,
      );
      lanternLight.intensity = (0.14 + 0.46 * open) * grow;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.13 + 0.8 * bloom;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 2.2 * bloom;
    }

    world.setTime(time);

    rig.update(frame.dtReal);
  }

  protected drop(): void {
    this.lantern?.dispose();
    this.lantern = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
  }
}


export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.16 * Math.sin(Math.PI * t) * (1 - t));
}

export function opening(time: number, agent: GridCell, goal: GridCell): number {
  const d = Math.max(Math.abs(agent.x - goal.x), Math.abs(agent.y - goal.y));
  const near = d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
  const breath = 0.5 + 0.5 * Math.sin(time * 0.72);
  const v = OPEN_BASE + OPEN_BREATH * breath + OPEN_PROXIMITY * near;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

const createScene: SceneFactory = () => new FourSeasonsCourtyardScene();
export default createScene;
