import * as THREE from 'three/webgpu';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec } from '@kit/grid';
import { clamp01, envelope, smoothstep01 } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import { CORRIDOR_GOALS, CORRIDOR_GX, CORRIDOR_GY, CORRIDOR_HELD, CORRIDOR_OPEN } from '@sim/envs/corridor';
import type { GridCell, GridGeometry } from '@sim/types';
import { formatCount, formatSigned } from '../../../format';
import { ACTIONS, accentNamed, agentCell, eventSteps, keyCell } from '../tape';
import {
  SEED_HOVER,
  buildGate,
  buildSeedBed,
  buildShrine,
  type GateHandle,
  type SeedBedHandle,
  type ShrineHandle,
} from './actors';
import { buildWorld, gardenSkin, resolveDoor, resolveGrid, type GardenSkin, type WorldHandle } from './world';

const BURST_SCALE = 0.8;
const RIPPLE_SCALE = 0.46;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.6;
const BLOOM_SECONDS = 1.5;

const KEY_BURST_SCALE = 0.34;
const IVY_BURST_SCALE = 0.42;
const IVY_BURST_Y = 0.62;

const GROW_SECONDS = 0.75;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.17;

const GATE_OPEN_SECONDS = HOP_SECONDS;
const GATE_RELOCK_SECONDS = 0.9;

const GLOW_BASE = 0.3;
const GLOW_BREATH = 0.16;
const GLOW_PROXIMITY = 0.44;
const PROXIMITY_CELLS = 4;

const AMBER = accentNamed(TOKENS.accents, 'KeyCorridor');
const LEAF = accentNamed(TOKENS.accents, 'FourRooms');

interface GardenState {
  agent: GridCell;
  goal: GridCell;
  key: GridCell;
  hasKey: boolean;
  doorOpen: boolean;
}

function makeGardenState(): GardenState {
  return { agent: { x: 0, y: 0 }, goal: { x: 0, y: 0 }, key: { x: 0, y: 0 }, hasKey: false, doorOpen: false };
}

function readState(pose: Float32Array, out: GardenState): GardenState {
  agentCell(pose, out.agent);
  out.goal.x = pose[CORRIDOR_GX]!;
  out.goal.y = pose[CORRIDOR_GY]!;
  keyCell(pose, 0, out.key);
  out.hasKey = (pose[CORRIDOR_HELD]! & 1) !== 0;
  out.doorOpen = (pose[CORRIDOR_OPEN]! & 1) !== 0;
  return out;
}

export class IvyGateScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 8;
  private sceneView: SceneView | null = null;

  protected readonly stageName = 'ivyGate';

  private wisp: Wisp | null = null;
  private gate: GateHandle | null = null;
  private shrine: ShrineHandle | null = null;
  private seed: SeedBedHandle | null = null;
  private grid!: GridGeometry;
  private door: GridCell = { x: 0, y: 0 };
  private skin: GardenSkin | null = null;

  private gateAt = -Infinity;
  private shrineGrowAt = -Infinity;
  private seedGrowAt = -Infinity;
  private bloomAt = -Infinity;

  private carried = false;
  private readonly paddedGoal: GridCell = { x: -1, y: -1 };

  private readonly cur: GardenState = makeGardenState();
  private readonly steps: number[] = [];
  private readonly eventCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly ledger = { action: 0, reward: 0, goals: 0 };

  override legend: LegendSpec = {
    title: 'IVY GATE',
    rows: [
      { key: 'at', label: 'wisp', tint: AMBER, value: () => `column ${this.cur.agent.x + 1} · row ${this.cur.agent.y + 1}` },
      { key: 'seed', label: 'amber seed', tint: AMBER, value: () => (this.cur.hasKey ? 'carried' : `column ${this.cur.key.x + 1} · row ${this.cur.key.y + 1}`) },
      { key: 'gate', label: 'ivy gate', tint: LEAF, value: () => (this.cur.doorOpen ? 'open' : 'locked') },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => ACTIONS[this.ledger.action]! },
      { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => formatSigned(this.ledger.reward) },
    ],
    note: () => `${formatCount(this.ledger.goals)} blooms`,
  };

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.door = resolveDoor(grid);
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const palette = ctx.tokens.palettes.morning;
    const skin = gardenSkin(palette, ctx.tokens);
    this.skin = skin;
    const accent = this.accentOf(ctx);

    world.rig.points[0]?.color.set(skin.amber);
    world.rig.points[1]?.color.set(accent);
    world.rig.points[2]?.color.set(skin.amber);
    world.rig.points[3]?.color.set(ctx.tokens.celebrationGold);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.17,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'gardenWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const gate = buildGate({ palette, skin, yaw: world.gateYaw, seed: ctx.seed, name: 'ivyGate' });
    this.gate = gate;
    this.scene.add(gate.group);
    cellToWorldVec(this.door.x, this.door.y, grid, gate.root.position);

    const shrine = buildShrine({ palette, skin, name: 'goalShrine' });
    this.shrine = shrine;
    this.scene.add(shrine.group);

    const seed = buildSeedBed({ palette, skin, name: 'seedBed' });
    this.seed = seed;
    this.scene.add(seed.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    const grid = this.grid;
    const halfX = grid.w * CELL / 2;
    const halfZ = grid.h * CELL / 2;
    this.sceneView = makeSceneView(ctx, { min: [-halfX, 0, -halfZ], max: [halfX, 1.8, halfZ] }, [Math.sin(-0.72) * 1.2, 0.5, Math.cos(-0.72) * 1.2], { distance: 13, azimuth: -0.72, polar: 0.62 });
    return this.sceneView.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const overlay = makeGardenState();
    const keys: GridCell[] = [{ x: 0, y: 0 }];
    const doors: boolean[] = [false];
    return {
      agentCell: (state) => readState(state.pose, overlay).agent,
      goalCell: (state) => readState(state.pose, overlay).goal,
      keyCells: (state) => {
        const st = readState(state.pose, overlay);
        const at = st.hasKey ? st.agent : st.key;
        keys[0]!.x = at.x;
        keys[0]!.y = at.y;
        return keys;
      },
      doorStates: (state) => {
        doors[0] = readState(state.pose, overlay).doorOpen;
        return doors;
      },
      maxKeys: 1,
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    this.wisp?.setCell(0, 0, this.grid);
    this.gate?.setState(0, 0);
    this.gate?.setTime(0);
    this.shrine?.setState(1, GLOW_BASE);
    this.shrine?.setTime(0);
    this.seed?.setState(1, 1);
    this.seed?.setTime(0);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const gate = this.gate;
    const shrine = this.shrine;
    const seed = this.seed;
    if (!wisp || !gate || !shrine || !seed) return;

    const grid = this.grid;
    const s = readState(frame.state.pose, this.cur);
    this.ledger.action = Math.trunc(frame.frameAt(frame.state.step).actions[0]!) & 3;
    this.ledger.reward = frame.state.reward;
    this.ledger.goals = frame.state.pose[CORRIDOR_GOALS]!;

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      this.gateAt = -Infinity;
      this.shrineGrowAt = -Infinity;
      this.seedGrowAt = -Infinity;
      this.bloomAt = -Infinity;
      wisp.setCell(s.agent.x, s.agent.y, grid);
      wisp.snap();
    }

    const steps = eventSteps(frame, this.steps);
    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i]!;
      if (ev.type === 'goal') {
        const before = frame.poseAt(steps[i]! - 1);
        cellToWorldVec(before[CORRIDOR_GX]!, before[CORRIDOR_GY]!, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.skin?.amber ?? ctx.tokens.celebrationGold, BURST_SCALE, time);
        pool.ripple(this.groundPos, ctx.tokens.celebrationGold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        shrine.flare(0.8);
        this.bloomAt = time;
        this.shrineGrowAt = time;
        this.seedGrowAt = time;
      } else if (ev.type === 'key') {
        const at = keyCell(frame.poseAt(steps[i]!), ev.index, this.eventCell);
        cellToWorldVec(at.x, at.y, grid, this.bloomPos);
        this.bloomPos.y = SEED_HOVER;
        pool.burst(this.bloomPos, this.skin?.amber ?? ctx.tokens.celebrationGold, KEY_BURST_SCALE, time);
        wisp.flare(0.6);
      } else if (ev.type === 'door') {
        this.gateAt = time;
        if (ev.open) {
          cellToWorldVec(this.door.x, this.door.y, grid, this.bloomPos);
          this.bloomPos.y = IVY_BURST_Y;
          pool.burst(this.bloomPos, this.skin?.ivy ?? ctx.tokens.celebrationGold, IVY_BURST_SCALE, time);
        }
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    if (teleported) {
      wisp.setCell(s.agent.x, s.agent.y, grid);
      wisp.snap();
    }

    if (s.agent.x !== wisp.cellX || s.agent.y !== wisp.cellY) {
      wisp.hopTo(s.agent, time);
    }
    wisp.update(frame.dtReal, time);

    cellToWorldVec(s.key.x, s.key.y, grid, seed.root.position);
    if (s.hasKey !== this.carried) {
      if (s.hasKey) {
        wisp.carry(seed.token.group);
      } else {
        wisp.carry(null);
        seed.cradle.add(seed.token.group);
        seed.token.group.position.set(0, 0, 0);
      }
      this.carried = s.hasKey;
    }
    const seedGrow = growth(time - this.seedGrowAt);
    seed.setState(seedGrow, s.hasKey ? 0 : 1);
    seed.setTime(time);
    seed.update(frame.dtReal, time);

    cellToWorldVec(s.goal.x, s.goal.y, grid, shrine.root.position);
    if (s.goal.x !== this.paddedGoal.x || s.goal.y !== this.paddedGoal.y) {
      if (this.paddedGoal.x >= 0) world.restoreFloorTint(this.paddedGoal.x, this.paddedGoal.y);
      world.paintFloor(s.goal.x, s.goal.y, this.skin?.shrinePad ?? ctx.tokens.celebrationGold);
      this.paddedGoal.x = s.goal.x;
      this.paddedGoal.y = s.goal.y;
    }
    const shrineGrow = growth(time - this.shrineGrowAt);
    const glow = shrineGlow(time, s.agent, s.goal);
    shrine.setState(shrineGrow, glow);
    shrine.setTime(time);
    shrine.update(frame.dtReal, time);

    gate.setState(gatePhase(s.doorOpen, time - this.gateAt), time);
    gate.setTime(time);

    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const shrineLight = world.rig.points[0];
    if (shrineLight) {
      shrineLight.position.set(shrine.root.position.x, shrine.crownHeight * shrineGrow, shrine.root.position.z);
      shrineLight.intensity = (0.2 + 0.7 * glow) * shrineGrow;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.14 + (s.hasKey ? 0.3 : 0) + 0.7 * bloom;
    }
    const seedLight = world.rig.points[2];
    if (seedLight) {
      seedLight.position.set(seed.root.position.x, SEED_HOVER, seed.root.position.z);
      seedLight.intensity = s.hasKey ? 0 : 0.55 * seedGrow;
    }
    const bloomLight = world.rig.points[3];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.position.y = BLOOM_Y;
      bloomLight.intensity = 2.1 * bloom;
    }

    world.setTime(time);

    if (this.sceneView) world.setView(this.sceneView.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.rig && this.sceneView) this.sceneView.resize(size, this.rig);
  }

  protected drop(): void {
    this.wisp?.carry(null);
    this.carried = false;
    this.seed?.dispose();
    this.seed = null;
    this.shrine?.dispose();
    this.shrine = null;
    this.gate?.dispose();
    this.gate = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.skin = null;
    this.gateAt = -Infinity;
    this.shrineGrowAt = -Infinity;
    this.seedGrowAt = -Infinity;
    this.bloomAt = -Infinity;
    this.paddedGoal.x = -1;
    this.paddedGoal.y = -1;
  }
}

function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.15 * Math.sin(Math.PI * t) * (1 - t));
}

function gatePhase(open: boolean, age: number): number {
  const span = open ? GATE_OPEN_SECONDS : GATE_RELOCK_SECONDS;
  const p = !(age >= 0) || age >= span ? 1 : smoothstep01(age / span);
  const target = open ? 1 : 0;
  return clamp01(target * p + (1 - target) * (1 - p));
}

function shrineGlow(time: number, agent: GridCell, goal: GridCell): number {
  const d = Math.max(Math.abs(agent.x - goal.x), Math.abs(agent.y - goal.y));
  const near = d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
  const breath = 0.5 + 0.5 * Math.sin(time * 0.64);
  return clamp01(GLOW_BASE + GLOW_BREATH * breath + GLOW_PROXIMITY * near);
}
