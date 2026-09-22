import { makeSceneView, type SceneView } from '@kit/view';
import { FOURROOMS_GX, FOURROOMS_GY, FOURROOMS_X, FOURROOMS_Y } from '@sim/envs/hall';
import { CELL, cellToWorldVec, clampCell, type GridSize } from '@kit/grid';
import type { GroundTruthOptions } from '@kit/debug';
import { blendTokens, envelope, shade } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { buildLantern, buildTravelerLantern, type LanternHandle } from './actors';
import { buildWorld, courtyardSkin, resolveGrid, type WorldHandle } from './world';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';

const DIR = ['\u2191 north', 'east \u2192', '\u2193 south', '\u2190 west'] as const;


const BURST_SCALE = 0.85;
const RIPPLE_SCALE = 0.5;
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
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['noon', 'fog'] } };
  readonly controlHz = 8;
  protected override poolOptions() { return { burstSize: 110 }; }
  private sceneView: SceneView | null = null;

  protected readonly stageName = 'fourSeasonsCourtyard';

  private wisp: Wisp | null = null;
  private traveler: ReturnType<typeof buildTravelerLantern> | null = null;
  private lantern: LanternHandle | null = null;
  private grid!: GridGeometry;

  private growAt = -Infinity;
  private bloomAt = -Infinity;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private blossom = 0xf7a8b8;
  private gold = 0xffd166;
  private step = 0;
  private period = 0;
  private readonly ledger = { action: 0, reward: 0 };
  override legend: LegendSpec = {
    title: 'COURTYARD',
    rows: [
      { key: 'at', label: 'traveler', tint: 0xffd166, value: () => `column ${this.agentCell.x + 1} \u00b7 row ${this.agentCell.y + 1}` },
      { key: 'goal', label: 'lantern flower', tint: 0xf7a8b8, value: () => `column ${this.goalCell.x + 1} \u00b7 row ${this.goalCell.y + 1}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => DIR[this.ledger.action]! },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
  };

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const palette = ctx.tokens.palettes.noon;
    const skin = courtyardSkin(palette, ctx.tokens);
    const accent = this.accentOf(ctx);
    this.blossom = ctx.tokens.accents['walk'] ?? ctx.tokens.celebrationGold;
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
        color: flame,
        radius: 0.3,
        height: 0.75,
        skirt: false,
        ribbon: false,
        silhouette: false,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'courtyardWisp',
      },
      grid,
    );
    this.traveler = buildTravelerLantern(0xffffff);
    wisp.avatar.add(this.traveler.mesh);
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
    const grid = this.grid;
    const halfX = grid.w * CELL / 2;
    const halfZ = grid.h * CELL / 2;
    this.sceneView = makeSceneView(ctx, { min: [-halfX, 0, -halfZ], max: [halfX, 1.8, halfZ] }, [0, 0.5, 0], { distance: 14.5, azimuth: 0.55, polar: 0.92 });
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
      debug: (_frame, kit) => {
        const wisp = this.wisp;
        if (wisp) kit.trail('walk', 120, this.ctx!.tokens.uiCyan).push(wisp.position.x, 0.35, wisp.position.z);
      },
    };
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    this.period = parameter(ctx.environment, 'goalPeriod');
    if (this.period > 0) {
      this.legend.note = () => `lantern moves in ${formatCount(this.period - this.step % this.period)} steps`;
      this.legend.bar = () => (this.step % this.period) / this.period;
    }
    world.setTime(0);
    this.wisp?.setCell(0, 0, this.grid);
    this.lantern?.setState(OPEN_BASE, 1);
    this.lantern?.setTime(0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const lantern = this.lantern;
    if (!wisp || !lantern) return;

    const state = frame.state;
    const grid = this.grid;
    const agent = this.agentCell;
    const goal = this.goalCell;

    pool.update(frame.dtReal, time);

    readCells(state, grid, agent, goal);
    this.step = state.step; this.ledger.action = Math.trunc(frame.frameAt(state.step).actions[0]!) & 3; this.ledger.reward = state.reward;

    if (frame.scrubbed) {
      pool.clear();
      this.bloomAt = -Infinity;
      this.growAt = -Infinity;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
    }

    let teleported = state.reset;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        cellToWorldVec(goal.x, goal.y, grid, this.bloomPos);
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
      lanternLight.intensity = (0.5 + 0.9 * open) * grow;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.7 + 1.2 * bloom;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 3 * bloom;
    }

    world.brazierFlame.emissiveIntensity = 1.2 + 0.35 * Math.sin(time * 2.3);
    world.setTime(time);

    if (this.sceneView) world.setView(this.sceneView.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void { super.resize(size); if (this.rig && this.sceneView) this.sceneView.resize(size, this.rig); }

  protected drop(): void {
    this.traveler?.dispose();
    this.traveler = null;
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

export default ((() => new FourSeasonsCourtyardScene()) satisfies SceneFactory);
