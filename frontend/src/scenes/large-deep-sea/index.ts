import { PULSE_SECONDS } from '@kit/celebrate';
import { CELL } from '@kit/grid';
import { makeSceneView, type SceneView } from '@kit/view';
import { LARGE_DEEPSEA_ACTION, LARGE_DEEPSEA_COL, LARGE_DEEPSEA_GX, LARGE_DEEPSEA_GY, LARGE_DEEPSEA_ROW } from '@sim/envs/largedeepsea';
import type { GroundTruthOptions } from '@kit/debug';
import { clampCell, type GridSize } from '@kit/grid';
import { blendTokens, clamp01, envelope } from '@kit/mix';
import { cellToTerraceVec as cellToWorldVec, terraceDepthAt, terraceShear } from '@kit/terrace';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { HEART_Y, buildDiveBubble, buildWreck, type DiveBubbleHandle, type WreckHandle } from './actors';
import { buildWorld, resolveGrid, resolveMask, type WorldHandle } from './world';


const BURST_SCALE = 0.8;
const RIPPLE_SCALE = 0.55;
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
  protected readonly stageName = 'theAbyss';
  private view: SceneView | null = null;

  private wisp: Wisp | null = null;
  private readonly diver = new THREE.Group();
  private bubble: DiveBubbleHandle | null = null;
  private wreck: WreckHandle | null = null;
  private grid!: GridGeometry;

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
  private readonly rushFrom = new THREE.Vector3();
  private readonly rushTo = new THREE.Vector3();
  private accent = 0x7fd4e8;
  private gold = 0xffd166;

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    const half = Math.max(grid.w, grid.h) * CELL * 0.5 + 0.8;
    this.view = makeSceneView(ctx, { min: [-half, -4.8, -half], max: [half, 2, half] }, [-2, -1.75, 2.3], { distance: 26.4, azimuth: -2.52, polar: 0.74 });
    const mask = resolveMask(grid);
    return buildWorld(ctx, this.scene, grid, mask);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const accent = this.accentOf(ctx);
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

  }

  protected rigOptions(_ctx: SceneContext): OrbitRigOptions {
    if (!this.view) throw new Error('Scene view is missing');
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const grid = this.grid;
    const overlayAgent: GridCell = { x: 0, y: 0 };
    const overlayGoal: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayAgent;
      },
      goalCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayGoal;
      },
    };
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const wisp = this.wisp;
    const wreck = this.wreck;
    const bubble = this.bubble;
    if (!wisp || !wreck || !bubble) return;
    if (this.overlay) {
      terraceShear(grid, this.overlay.group.matrix);
      this.overlay.group.matrixAutoUpdate = false;
      this.overlay.group.matrixWorldNeedsUpdate = true;
    }
    world.setTime(0);
    if (this.view && this.rig) world.setView(this.view.update(this.rig, 0), this.camera);
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
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const bubble = this.bubble;
    const wreck = this.wreck;
    if (!wisp || !bubble || !wreck) return;

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
    this.heartPos.y += HEART_Y;

    if (frame.scrubbed) {
      pool.clear();
      this.treasureAt = -Infinity;
      this.rushAt = -Infinity;
      this.stir = charged;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      const log = frame.eventLog;
      let low = 0;
      let high = log.count;
      while (low < high) {
        const mid = (low + high) >>> 1;
        if ((log.steps[mid] ?? Infinity) <= state.step) low = mid + 1;
        else high = mid;
      }
      const seconds = state.step > 0 ? state.simTime / state.step : 0;
      for (let i = low - 1; i >= 0; i--) {
        const at = ((log.steps[i] ?? -1) + 1) * seconds;
        if (time - at > TREASURE_SECONDS) break;
        if (log.types[log.kinds[i] ?? 0] === 'goal') {
          this.treasureAt = at;
          pool.burst(this.heartPos, this.accent, BURST_SCALE, at);
          pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, at);
          pool.pulse.reset();
          pool.pulse.fire(Math.max(0, PULSE_STRENGTH - (time - at) / PULSE_SECONDS));
          pool.update(0, time);
          break;
        }
      }
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
    this.bubblePos.y += TOKENS.wispHeight + this.diver.position.y;
    bubble.group.position.copy(this.bubblePos);
    bubble.setGain(depth);
    bubble.setTime(time);

    this.stir += (charged - this.stir) * (1 - Math.exp(-SILT_RATE * Math.max(0, frame.dtReal)));
    this.siltPos.copy(wisp.position);
    this.siltPos.y += siltLift(ctx.tokens.wispHeight) + this.diver.position.y;
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

    world.setTime(time);

    if (this.view) world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.view && this.rig && this.world) {
      this.view.resize(size, this.rig);
      this.world.setView(this.view.update(this.rig, 0), this.camera);
    }
  }

  protected drop(): void {
    this.wreck?.dispose();
    this.wreck = null;
    this.bubble?.dispose();
    this.bubble = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.diver.removeFromParent();
    this.diver.clear();
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

export default ((() => new TheAbyssScene()) satisfies SceneFactory);
