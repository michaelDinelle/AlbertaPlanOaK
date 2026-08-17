
import { KEYCORRIDOR_CARRY, KEYCORRIDOR_GATE, KEYCORRIDOR_GX, KEYCORRIDOR_GY, KEYCORRIDOR_KX, KEYCORRIDOR_KY, KEYCORRIDOR_X, KEYCORRIDOR_Y } from '@sim/envs/keycorridor';
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
import {
  SEED_HOVER,
  buildGate,
  buildSeedBed,
  buildShrine,
  type GateHandle,
  type SeedBedHandle,
  type ShrineHandle,
} from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
  buildWorld,
  clamp01,
  gardenSkin,
  resolveDoor,
  resolveGrid,
  smoothstep01,
  type GardenSkin,
  type WorldHandle,
} from './world';


const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.62;
const CAM_DISTANCE = 13.0;
const CAM_TARGET_Y = 0.5;
const CAM_TARGET_BIAS = 1.2;
const CAM_TARGET_X = CAM_TARGET_BIAS * Math.sin(CAM_AZIMUTH);
const CAM_TARGET_Z = CAM_TARGET_BIAS * Math.cos(CAM_AZIMUTH);

const BURST_SCALE = 0.8;
const RIPPLE_SCALE = 0.46;
const BURST_PARTICLES = 110;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.6;
const BLOOM_SECONDS = 1.5;

const KEY_BURST_SCALE = 0.34;
const IVY_BURST_SCALE = 0.42;
const IVY_BURST_Y = 0.62;

const GROW_SECONDS = 0.75;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.17;

export const GATE_OPEN_SECONDS = HOP_SECONDS;
export const GATE_RELOCK_SECONDS = 0.9;

const GLOW_BASE = 0.3;
const GLOW_BREATH = 0.16;
const GLOW_PROXIMITY = 0.44;
const PROXIMITY_CELLS = 4;

export interface GardenState {
  agent: GridCell;
  goal: GridCell;
  key: GridCell;
  hasKey: boolean;
  doorOpen: boolean;
}

export function makeGardenState(): GardenState {
  return {
    agent: { x: 0, y: 0 },
    goal: { x: 0, y: 0 },
    key: { x: 0, y: 0 },
    hasKey: false,
    doorOpen: false,
  };
}

export function readState(state: FrameState, grid: GridSize, out: GardenState): GardenState {
  const p = state.pose;
  out.agent.x = Math.round(p[KEYCORRIDOR_X] ?? 0);
  out.agent.y = Math.round(p[KEYCORRIDOR_Y] ?? 0);
  out.goal.x = Math.round(p[KEYCORRIDOR_GX] ?? 0);
  out.goal.y = Math.round(p[KEYCORRIDOR_GY] ?? 0);
  out.key.x = Math.round(p[KEYCORRIDOR_KX] ?? 0);
  out.key.y = Math.round(p[KEYCORRIDOR_KY] ?? 0);
  out.hasKey = (p[KEYCORRIDOR_CARRY] ?? 0) !== 0;
  out.doorOpen = (p[KEYCORRIDOR_GATE] ?? 0) !== 0;
  clampCell(out.agent, grid);
  clampCell(out.goal, grid);
  clampCell(out.key, grid);
  return out;
}

class IvyGateScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private gate: GateHandle | null = null;
  private shrine: ShrineHandle | null = null;
  private seed: SeedBedHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({});
  private door: GridCell = { x: 0, y: 0 };
  private skin: GardenSkin | null = null;

  private gateAt = -Infinity;
  private shrineGrowAt = -Infinity;
  private seedGrowAt = -Infinity;
  private bloomAt = -Infinity;

  private carried = false;
  private readonly paddedGoal: GridCell = { x: -1, y: -1 };

  private readonly cur: GardenState = makeGardenState();
  private readonly prevState: GardenState = makeGardenState();
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly bloomCell: GridCell = { x: 0, y: 0 };

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'ivyGateScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.door = resolveDoor(grid);

    const palette = ctx.tokens.palettes.morning;
    const skin = gardenSkin(palette, ctx.tokens);
    this.skin = skin;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    const shrineLight = world.rig.points[0];
    if (shrineLight) shrineLight.color.set(skin.amber);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.set(accent);
    const seedLight = world.rig.points[2];
    if (seedLight) seedLight.color.set(skin.amber);
    const bloomLight = world.rig.points[3];
    if (bloomLight) bloomLight.color.set(ctx.tokens.celebrationGold);

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

    const rig = makeOrbitRig({
      canvas: ctx.canvas,
      target: [CAM_TARGET_X, CAM_TARGET_Y, CAM_TARGET_Z],
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
      name: 'gardenOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'gardenCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    const overlayState = makeGardenState();
    const overlayKeys: GridCell[] = [{ x: 0, y: 0 }];
    const overlayDoors: boolean[] = [false];
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => readState(state, grid, overlayState).agent,
      goalCell: (state) => readState(state, grid, overlayState).goal,
      keyCells: (state) => {
        const st = readState(state, grid, overlayState);
        const cell = overlayKeys[0];
        if (cell) {
          cell.x = st.key.x;
          cell.y = st.key.y;
        }
        return overlayKeys;
      },
      doorStates: (state) => {
        overlayDoors[0] = readState(state, grid, overlayState).doorOpen;
        return overlayDoors;
      },
      name: 'gardenGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    gate.setState(0, 0);
    gate.setTime(0);
    shrine.setState(1, GLOW_BASE);
    shrine.setTime(0);
    seed.setState(1, 1);
    seed.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const gate = this.gate;
    const shrine = this.shrine;
    const seed = this.seed;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !gate || !shrine || !seed || !pool || !rig) return;

    const time = ctx.time();
    const grid = this.grid;
    const s = readState(frame.state, grid, this.cur);

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      this.gateAt = -Infinity;
      this.shrineGrowAt = -Infinity;
      this.seedGrowAt = -Infinity;
      this.bloomAt = -Infinity;
      wisp.setCell(s.agent.x, s.agent.y, grid);
      wisp.snap();
      rig.snap();
    }

    let teleported = false;
    let sawGoal = false;
    let sawDone = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        sawGoal = true;
        this.readBloomCell(frame, wisp);
        cellToWorldVec(this.bloomCell.x, this.bloomCell.y, grid, this.bloomPos);
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
        cellToWorldVec(s.key.x, s.key.y, grid, this.bloomPos);
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
      } else if (ev.type === 'done') {
        sawDone = true;
      }
    }
    if (sawDone && !sawGoal) pool.sigh(0.7, this.groundPos, time);

    if (teleported) {
      wisp.setCell(s.agent.x, s.agent.y, grid);
      wisp.snap();
      rig.snap();
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
      shrineLight.position.set(
        shrine.root.position.x,
        shrine.crownHeight * shrineGrow,
        shrine.root.position.z,
      );
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

    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(frame.state);
  }

  private readBloomCell(frame: FrameUpdate, wisp: Wisp): void {
    const prev = frame.prev;
    if (prev) {
      const p = readState(prev, this.grid, this.prevState);
      this.bloomCell.x = p.goal.x;
      this.bloomCell.y = p.goal.y;
      return;
    }
    this.bloomCell.x = wisp.cellX;
    this.bloomCell.y = wisp.cellY;
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
    this.wisp?.carry(null);
    this.carried = false;
    this.overlay?.dispose();
    this.overlay = null;
    this.seed?.dispose();
    this.seed = null;
    this.shrine?.dispose();
    this.shrine = null;
    this.gate?.dispose();
    this.gate = null;
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
    this.skin = null;
    this.gateAt = -Infinity;
    this.shrineGrowAt = -Infinity;
    this.seedGrowAt = -Infinity;
    this.bloomAt = -Infinity;
    this.paddedGoal.x = -1;
    this.paddedGoal.y = -1;
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
  return e * (1 + 0.15 * Math.sin(Math.PI * t) * (1 - t));
}

export function gatePhase(open: boolean, age: number): number {
  const span = open ? GATE_OPEN_SECONDS : GATE_RELOCK_SECONDS;
  const p = !(age >= 0) || age >= span ? 1 : smoothstep01(age / span);
  const target = open ? 1 : 0;
  return clamp01(target * p + (1 - target) * (1 - p));
}

export function shrineGlow(time: number, agent: GridCell, goal: GridCell): number {
  const d = Math.max(Math.abs(agent.x - goal.x), Math.abs(agent.y - goal.y));
  const near = d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
  const breath = 0.5 + 0.5 * Math.sin(time * 0.64);
  return clamp01(GLOW_BASE + GLOW_BREATH * breath + GLOW_PROXIMITY * near);
}

const createScene: SceneFactory = () => new IvyGateScene();
export default createScene;
