import { makeSceneView, type SceneView } from '../jelly-bean-world/view';
import { POMAZE_GX, POMAZE_GY, POMAZE_X, POMAZE_Y } from '@sim/envs/pomaze';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec, clampCell, type GridSize } from '@kit/grid';
import { blendTokens, clamp01, envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import {
  buildGoalBeacon,
  buildLanternPool,
  buildMoonflower,
  buildWispAura,
  type GoalBeaconHandle,
  type LanternPoolHandle,
  type MoonflowerHandle,
  type WispAuraHandle,
} from './actors';
import {
  BUBBLE_OUTER,
  buildWorld,
  forestSkin,
  resolveGrid,
  type WorldHandle,
} from './world';


const BURST_SCALE = 0.45;
const RIPPLE_SCALE = 0.52;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.55;

export const REVEAL_SECONDS = 1.9;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.14;

const DEFAULT_BLOOM = forestSkin(TOKENS.palettes.dawn, TOKENS).bloom.getHex();

export function readCells(state: FrameState, grid: GridSize, agent: GridCell, goal: GridCell): void {
  agent.x = Math.round(state.pose[POMAZE_X] ?? 0);
  agent.y = Math.round(state.pose[POMAZE_Y] ?? 0);
  goal.x = Math.round(state.pose[POMAZE_GX] ?? 0);
  goal.y = Math.round(state.pose[POMAZE_GY] ?? 0);
  clampCell(agent, grid);
  clampCell(goal, grid);
}

class FogboundForestScene extends StageScene<WorldHandle> {
  private sceneView: SceneView | null = null;

  protected readonly stageName = 'fogboundForest';

  private wisp: Wisp | null = null;
  private flower: MoonflowerHandle | null = null;
  private beacon: GoalBeaconHandle | null = null;
  private lantern: LanternPoolHandle | null = null;
  private aura: WispAuraHandle | null = null;
  private grid!: GridGeometry;

  private revealAt = -Infinity;
  private revealX = 0;
  private revealZ = 0;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly prevGoalCell: GridCell = { x: 0, y: 0 };
  private readonly prevAgentCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly goalWorld = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private gold = TOKENS.celebrationGold;
  private bloomColor = DEFAULT_BLOOM;

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const palette = ctx.tokens.palettes.dawn;
    const accent = this.accentOf(ctx);
    this.gold = ctx.tokens.celebrationGold;
    this.bloomColor = world.skin.bloom.getHex();

    const lanternLight = world.rig.points[0];
    if (lanternLight) lanternLight.color.set(blendTokens(accent, palette.accentLight, 0.62));
    const breathLight = world.rig.points[1];
    if (breathLight) breathLight.color.set(this.gold);
    const flowerLight = world.rig.points[2];
    if (flowerLight) flowerLight.color.set(world.skin.bloom);

    const wispColor = blendTokens(accent, this.gold, 0.34);
    const wisp = new Wisp(
      {
        color: wispColor,
        radius: 0.09,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'forestWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const lantern = buildLanternPool(
      blendTokens(accent, palette.accentLight, 0.55),
      BUBBLE_OUTER * 1.15,
    );
    this.lantern = lantern;
    this.scene.add(lantern.mesh);

    const aura = buildWispAura(blendTokens(wispColor, palette.sun, 0.4), 1.05);
    this.aura = aura;
    this.scene.add(aura.mesh);

    const flower = buildMoonflower({
      skin: world.skin,
      glow: world.skin.bloom,
      name: 'moonflowerRoot',
    });
    this.flower = flower;
    this.scene.add(flower.root);
    this.scene.add(flower.bloom);

    const beacon = buildGoalBeacon({
      glow: world.skin.bloom,
      name: 'goalBeacon',
    });
    this.beacon = beacon;
    this.scene.add(beacon.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    const grid = this.grid;
    const halfX = grid.w * CELL / 2;
    const halfZ = grid.h * CELL / 2;
    this.sceneView = makeSceneView(ctx, { min: [-halfX, 0, -halfZ], max: [halfX, 1.8, halfZ] }, [0, 0.82, 0], { distance: 4.7, azimuth: -1.35, polar: 1.32 });
    return this.sceneView.options;
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

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    world.setFogLift(0);
    this.wisp?.dim(0.5);
    this.wisp?.setCell(0, 0, this.grid);
    this.flower?.setState(0, 0);
    this.lantern?.setGain(1);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const flower = this.flower;
    const lantern = this.lantern;
    const aura = this.aura;
    if (!wisp || !flower || !lantern || !aura) return;

    const state = frame.state;
    const grid = this.grid;
    const agent = this.agentCell;
    const goal = this.goalCell;

    pool.update(frame.dtReal, time);

    readCells(state, grid, agent, goal);

    if (frame.scrubbed) {
      pool.clear();
      this.revealAt = -Infinity;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        const prev = frame.prev;
        if (prev) {
          readCells(prev, grid, this.prevAgentCell, this.prevGoalCell);
          cellToWorldVec(this.prevGoalCell.x, this.prevGoalCell.y, grid, this.bloomPos);
        } else {
          cellToWorldVec(agent.x, agent.y, grid, this.bloomPos);
        }
        this.groundPos.copy(this.bloomPos);
        this.revealX = this.bloomPos.x;
        this.revealZ = this.bloomPos.z;
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.bloomColor, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        flower.flare(0.9);
        this.revealAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
    } else if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      wisp.face({ dx: agent.x - wisp.cellX, dy: agent.y - wisp.cellY });
      wisp.hopTo(agent, time);
    }
    wisp.update(frame.dtReal, time);

    const wx = wisp.position.x;
    const wz = wisp.position.z;

    const reveal = envelope(time - this.revealAt, REVEAL_SECONDS, 1.6);
    const breath = clamp01(reveal * 2.2);
    world.spruce.setBubble(wx, wz, this.revealX, this.revealZ, breath);
    world.setFogLift(reveal);

    cellToWorldVec(goal.x, goal.y, grid, this.goalWorld);
    flower.root.position.copy(this.goalWorld);
    flower.bloom.position.set(this.revealX, 0, this.revealZ);
    flower.setState(breath, time);
    flower.update(frame.dtReal, time);

    const beacon = this.beacon;
    if (beacon) {
      beacon.group.position.set(this.goalWorld.x, 0, this.goalWorld.z);
      beacon.setTime(time);
    }

    lantern.mesh.position.set(wx, 0.015, wz);
    lantern.setGain(1 + 0.55 * reveal);
    aura.mesh.position.set(wx, wisp.position.y + TOKENS.wispHeight, wz);
    aura.face(this.camera);
    aura.setGain(0.72 + 1.4 * reveal);

    const lanternLight = world.rig.points[0];
    if (lanternLight) {
      this.lightPos.set(wx, TOKENS.wispHeight + 0.12, wz);
      lanternLight.position.copy(this.lightPos);
      lanternLight.intensity = 0.62 + 0.5 * reveal;
    }
    const breathLight = world.rig.points[1];
    if (breathLight) {
      breathLight.position.set(this.revealX, 0.75, this.revealZ);
      breathLight.intensity = 2.8 * reveal;
    }
    const flowerLight = world.rig.points[2];
    if (flowerLight) {
      flowerLight.position.set(this.revealX, flower.crownHeight, this.revealZ);
      flowerLight.intensity = 0.75 * breath;
    }
    const beaconLight = world.rig.points[3];
    if (beaconLight) {
      beaconLight.name = 'beacon:goalLight';
      beaconLight.color.set(world.skin.bloom);
      beaconLight.position.set(this.goalWorld.x, 1.4, this.goalWorld.z);
      beaconLight.intensity = 0.3;
    }

    world.setTime(time);

    if (this.sceneView) world.setView(this.sceneView.update(rig, frame.dtReal, [wx, 0.82, wz]), this.camera);
  }

  protected drop(): void {
    this.flower?.dispose();
    this.flower = null;
    this.beacon?.dispose();
    this.beacon = null;
    this.aura?.dispose();
    this.aura = null;
    this.lantern?.dispose();
    this.lantern = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.revealAt = -Infinity;
  }
}

export default ((() => new FogboundForestScene()) satisfies SceneFactory);
