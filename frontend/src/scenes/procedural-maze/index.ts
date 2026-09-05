import { makeSceneView, type SceneView } from '../jelly-bean-world/view';
import { PROCMAZE_GX, PROCMAZE_GY, PROCMAZE_X, PROCMAZE_Y } from '@sim/envs/proceduralmaze';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec, clampCell, type GridSize } from '@kit/grid';
import { blendTokens, envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { buildSunShaft, type SunShaftHandle } from './actors';
import {
  SUN_DIR,
  buildWorld,
  floorTintFor,
  resolveGrid,
  ruinSkin,
  type RuinSkin,
  type WorldHandle,
} from './world';

const BURST_SCALE = 0.82;
const RIPPLE_SCALE = 0.48;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.46;
const BLOOM_SECONDS = 1.5;
const GROW_SECONDS = 0.8;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;

function readCells(state: FrameState, grid: GridSize, agent: GridCell, goal: GridCell): void {
  agent.x = Math.round(state.pose[PROCMAZE_X] ?? 0);
  agent.y = Math.round(state.pose[PROCMAZE_Y] ?? 0);
  goal.x = Math.round(state.pose[PROCMAZE_GX] ?? 0);
  goal.y = Math.round(state.pose[PROCMAZE_GY] ?? 0);
  clampCell(agent, grid);
  clampCell(goal, grid);
}

export function readGoalCell(state: FrameState, grid: GridSize, out: GridCell): GridCell {
  out.x = Math.round(state.pose[PROCMAZE_GX] ?? 0);
  out.y = Math.round(state.pose[PROCMAZE_GY] ?? 0);
  return clampCell(out, grid);
}

class OvergrownRuinsScene extends StageScene<WorldHandle> {
  private sceneView: SceneView | null = null;

  protected readonly stageName = 'overgrownRuins';

  private wisp: Wisp | null = null;
  private shaft: SunShaftHandle | null = null;
  private gridGeom!: GridGeometry;
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

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.gridGeom = grid;
    this.seed = ctx.seed;
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.gridGeom;
    const palette = ctx.tokens.palettes.noon;
    const skin = ruinSkin(palette, ctx.tokens);
    this.skin = skin;
    const accent = this.accentOf(ctx);
    this.gold = ctx.tokens.celebrationGold;
    this.moteColor = blendTokens(
      ctx.tokens.celebrationGold,
      ctx.tokens.palettes.dusk.groundA,
      0.3,
    ).getHex();
    this.litTint.copy(blendTokens(skin.flag, skin.sunWarm, 0.5));

    const beamLight = world.rig.points[0];
    if (beamLight) beamLight.color.set(skin.sunWarm);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const bloomLight = world.rig.points[2];
    if (bloomLight) bloomLight.color.set(this.moteColor);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.22,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'ruinWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const shaft = buildSunShaft({
      palette,
      skin,
      sunDir: SUN_DIR,
      name: 'sunShaft',
    });
    this.shaft = shaft;
    this.scene.add(shaft.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    const grid = this.gridGeom;
    const halfX = grid.w * CELL / 2;
    const halfZ = grid.h * CELL / 2;
    this.sceneView = makeSceneView(ctx, { min: [-halfX, 0, -halfZ], max: [halfX, 1.8, halfZ] }, [Math.sin(0.62) * 1.2, 0.15, Math.cos(0.62) * 1.2], { distance: 19, azimuth: 0.62, polar: 0.73 });
    return this.sceneView.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const grid = this.gridGeom;
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

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    this.wisp?.setCell(0, 0, this.gridGeom);
    this.shaft?.setState(1);
    this.shaft?.setTime(0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const shaft = this.shaft;
    const grid = this.gridGeom;
    if (!wisp || !shaft) return;

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

    if (this.sceneView) world.setView(this.sceneView.update(rig, frame.dtReal), this.camera);
  }

  private relightFloor(x: number, y: number): void {
    if (x === this.litX && y === this.litY) return;
    const world = this.world;
    const grid = this.gridGeom;
    const skin = this.skin;
    if (!world || !skin) return;
    if (this.litX >= 0) {
      floorTintFor(grid, skin, this.litX, this.litY, this.seed, this.restoreTint);
      world.field.setFloorColor(this.litX, this.litY, this.restoreTint);
    }
    world.field.setFloorColor(x, y, this.litTint);
    this.litX = x;
    this.litY = y;
  }

  protected drop(): void {
    this.shaft?.dispose();
    this.shaft = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.skin = null;
    this.seed = 0;
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
    this.litX = -1;
    this.litY = -1;
  }
}

export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.13 * Math.sin(Math.PI * t) * (1 - t));
}

export default ((() => new OvergrownRuinsScene()) satisfies SceneFactory);
