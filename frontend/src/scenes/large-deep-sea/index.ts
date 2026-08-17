
import { LARGE_DEEPSEA_ACTION, LARGE_DEEPSEA_COL, LARGE_DEEPSEA_GX, LARGE_DEEPSEA_GY, LARGE_DEEPSEA_ROW } from '@sim/envs/largedeepsea';
import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { clampCell, type GridSize } from '@kit/grid';
import { blendTokens, clamp01, envelope } from '@kit/mix';
import { cellToTerraceVec as cellToWorldVec, terraceDepthAt } from '@kit/terrace';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { HEART_Y, buildAngler, buildDiveBubble, buildWreck, type AnglerHandle, type DiveBubbleHandle, type WreckHandle } from './actors';
import { manifest } from './manifest';
import { CAMERA_AZIMUTH, anglerSites, buildWorld, resolveGrid, resolveMask, type WorldHandle } from './world';


const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.74;
const CAM_DISTANCE = 24;
const CAM_TARGET: readonly [number, number, number] = [-1.2, -1.75, 1.1];

const BURST_SCALE = 0.8;
const RIPPLE_SCALE = 0.55;
const BURST_PARTICLES = 130;
const PULSE_STRENGTH = 0.62;
const TREASURE_SECONDS = 2.2;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;

const WISP_LAMP_LIFT = 0.75;

const RUSH_SECONDS = 1.15;
const RUSH_Y = 1.25;

const SILT_LIFT = 0.28;
const SILT_RATE = 9;
const SILT_IDLE = 0.04;
const SILT_KICK = 0.85;

function readCells(
  state: FrameState,
  grid: GridGeometry,
  agent: GridCell,
  goal: GridCell,
): void {
  agent.x = Math.round(state.pose[LARGE_DEEPSEA_COL] ?? 0);
  agent.y = Math.round(state.pose[LARGE_DEEPSEA_ROW] ?? 0);
  goal.x = Math.round(state.pose[LARGE_DEEPSEA_GX] ?? 0);
  goal.y = Math.round(state.pose[LARGE_DEEPSEA_GY] ?? 0);
  clampCell(agent, grid);
  clampCell(goal, grid);
}

export function readRightMove(state: FrameState): number {
  return Math.trunc(state.pose[LARGE_DEEPSEA_ACTION] ?? 0) & 1;
}

export function isDiveStep(fromX: number, fromY: number, toX: number, toY: number): boolean {
  return toY === fromY + 1 && Math.abs(toX - fromX) <= 1;
}

class TheAbyssScene extends StageScene<WorldHandle> {
  readonly manifest = manifest;
  protected readonly stageName = 'theAbyss';

  private wisp: Wisp | null = null;
  private readonly diver = new THREE.Group();
  private bubble: DiveBubbleHandle | null = null;
  private wreck: WreckHandle | null = null;
  private anglers: AnglerHandle[] = [];
  private grid: GridGeometry = resolveGrid({});

  private treasureAt = -Infinity;
  private rushAt = -Infinity;
  private stir = 0;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly heartPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly bubblePos = new THREE.Vector3();
  private readonly siltPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly lureWorld = new THREE.Vector3();
  private readonly rushFrom = new THREE.Vector3();
  private readonly rushTo = new THREE.Vector3();
  private accent = 0x7fd4e8;
  private gold = 0xffd166;

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    const mask = resolveMask(grid, ctx.seed);
    return buildWorld(ctx, this.scene, grid, mask);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.uiCyan;
    this.accent = accent;
    this.gold = ctx.tokens.celebrationGold;
    const water = ctx.tokens.palettes.underwater;
    const skin = world.skin;

    const wispLight = world.rig.points[0];
    if (wispLight) wispLight.color.set(accent);
    const heartLight = world.rig.points[1];
    if (heartLight) heartLight.color.set(skin.heart);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.19,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'abyssWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.diver.add(wisp.group);
    this.scene.add(this.diver);

    const bubble = buildDiveBubble({
      accent,
      water: blendTokens(water.groundA, ctx.tokens.palettes.abyss.skyTop, 0.5),
      name: 'diveBubble',
    });
    this.bubble = bubble;
    this.scene.add(bubble.group);

    const wreck = buildWreck({ skin, accent, name: 'sunkenWreck' });
    this.wreck = wreck;
    this.scene.add(wreck.group);

    const sites = anglerSites(grid, ctx.seed);
    const lureColors = [accent, blendTokens(ctx.tokens.cueMagenta, accent, 0.42).getHex()];
    for (let i = 0; i < sites.length; i++) {
      const site = sites[i];
      if (!site) continue;
      const angler = buildAngler({
        skin,
        lure: lureColors[i] ?? accent,
        hover: site.hover,
        phase: site.phase,
        scale: site.scale,
        name: `angler${i}`,
      });
      cellToWorldVec(site.x, site.y, grid, angler.root.position);
      this.scene.add(angler.group);
      this.anglers.push(angler);
      const light = world.rig.points[2 + i];
      if (light) light.color.set(lureColors[i] ?? accent);
    }
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    return {
      canvas: ctx.canvas,
      target: [CAM_TARGET[0], CAM_TARGET[1], CAM_TARGET[2]],
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
      name: 'abyssOrbitRig',
    };
  }

  protected poolOptions(ctx: SceneContext): CelebrationPoolOptions {
    return {
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'abyssCelebrations',
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
      name: 'abyssGroundTruth',
    };
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const wisp = this.wisp;
    const wreck = this.wreck;
    const bubble = this.bubble;
    if (!wisp || !wreck || !bubble) return;
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    cellToWorldVec(grid.goal?.x ?? grid.w - 1, grid.goal?.y ?? grid.h - 1, grid, wreck.root.position);
    cellToWorldVec(0, 0, grid, this.rushTo);
    this.rushTo.y = RUSH_Y;
    this.rushFrom.copy(this.rushTo);
    wreck.setFlare(0);
    wreck.setTime(0);
    bubble.setGain(0);
    bubble.setTime(0);
    cellToWorldVec(0, 0, grid, world.silt.points.position);
    world.silt.points.position.y += siltLift(ctx.tokens.wispHeight);
    world.silt.setDensity(0);
    for (const angler of this.anglers) angler.setTime(0);
  }

  protected tick(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const bubble = this.bubble;
    const wreck = this.wreck;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !bubble || !wreck || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const agent = this.agentCell;
    const goal = this.goalCell;

    pool.update(frame.dtReal, time);

    readCells(state, grid, agent, goal);

    const depth = bubbleGain(agent.y, grid);
    wisp.dim(0.2 + 0.8 * depth);

    const charged = readRightMove(state);

    cellToWorldVec(goal.x, goal.y, grid, wreck.root.position);
    this.groundPos.copy(wreck.root.position);
    this.heartPos.copy(wreck.root.position);
    this.heartPos.y = HEART_Y;

    if (frame.scrubbed) {
      pool.clear();
      this.treasureAt = -Infinity;
      this.rushAt = -Infinity;
      this.stir = charged;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      rig.snap();
    }

    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        pool.burst(this.heartPos, this.accent, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        this.treasureAt = time;
      } else if (ev.type === 'resurface' || ev.type === 'teleport') {
        const fromX = frame.prev ? Math.round(frame.prev.pose[1] ?? wisp.cellX) : wisp.cellX;
        const fromY = frame.prev ? Math.round(frame.prev.pose[0] ?? wisp.cellY) : wisp.cellY;
        cellToWorldVec(
          fromX < 0 ? 0 : fromX > grid.w - 1 ? grid.w - 1 : fromX,
          fromY < 0 ? 0 : fromY > grid.h - 1 ? grid.h - 1 : fromY,
          grid,
          this.rushFrom,
        );
        this.rushFrom.y = RUSH_Y;
        cellToWorldVec(agent.x, agent.y, grid, this.rushTo);
        this.rushTo.y = RUSH_Y;
        this.rushAt = time;
      }
    }

    if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      if (isDiveStep(wisp.cellX, wisp.cellY, agent.x, agent.y)) {
        wisp.hopTo(agent, time);
      } else {
        wisp.setCell(agent.x, agent.y, grid);
        wisp.snap();
      }
    }
    wisp.update(frame.dtReal, time);
    this.diver.position.y = terraceDepthAt(wisp.position.z, grid);

    this.bubblePos.copy(wisp.position);
    this.bubblePos.y += TOKENS.wispHeight;
    bubble.group.position.copy(this.bubblePos);
    bubble.setGain(depth);
    bubble.setTime(time);

    this.stir += (charged - this.stir) * (1 - Math.exp(-SILT_RATE * Math.max(0, frame.dtReal)));
    this.siltPos.copy(wisp.position);
    this.siltPos.y += siltLift(ctx.tokens.wispHeight);
    world.silt.points.position.copy(this.siltPos);
    world.silt.setDensity(SILT_IDLE + SILT_KICK * this.stir);
    world.silt.setIntensity(0.45 + 0.85 * depth);

    const treasure = envelope(time - this.treasureAt, TREASURE_SECONDS, 2.1);
    const rushAge = time - this.rushAt;
    const rushing = rushAge >= 0 && rushAge <= RUSH_SECONDS;
    const rushT = rushing ? rushAge / RUSH_SECONDS : 0;
    const rushE = 1 - Math.pow(1 - rushT, 2.2);
    world.rush.points.position.lerpVectors(this.rushFrom, this.rushTo, rushing ? rushE : 0);
    world.rush.setDensity(rushing ? Math.sin(Math.PI * rushT) : 0);

    wreck.setFlare(treasure);
    wreck.setTime(time);
    for (let i = 0; i < this.anglers.length; i++) this.anglers[i]?.setTime(time);

    const wispLight = world.rig.points[0];
    if (wispLight) {
      this.lightPos.copy(this.bubblePos);
      this.lightPos.y += WISP_LAMP_LIFT;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity =
        0.06 + 3.4 * depth * depth + 5.5 * treasure + 1.2 * (rushing ? Math.sin(Math.PI * rushT) : 0);
    }
    const heartLight = world.rig.points[1];
    if (heartLight) {
      heartLight.position.copy(this.heartPos);
      heartLight.intensity = 1.05 + 0.35 * Math.sin(time * 1.35) + 8 * treasure;
    }
    for (let i = 0; i < this.anglers.length; i++) {
      const light = world.rig.points[2 + i];
      const angler = this.anglers[i];
      if (!light || !angler) continue;
      angler.lureAnchor.getWorldPosition(this.lureWorld);
      light.position.copy(this.lureWorld);
      light.intensity = 1.6 + 0.55 * Math.sin(time * 1.9 + i * 3.1);
    }

    world.setTime(time);

    rig.update(frame.dtReal);
  }

  protected drop(): void {
    for (const angler of this.anglers) angler.dispose();
    this.anglers.length = 0;
    this.wreck?.dispose();
    this.wreck = null;
    this.bubble?.dispose();
    this.bubble = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.treasureAt = -Infinity;
    this.rushAt = -Infinity;
    this.stir = 0;
  }
}


export function bubbleGain(row: number, grid: GridSize): number {
  return clamp01(row / Math.max(1, grid.h - 2));
}

export function siltLift(wispHeight: number): number {
  return wispHeight * SILT_LIFT;
}

const createScene: SceneFactory = () => new TheAbyssScene();
export default createScene;
