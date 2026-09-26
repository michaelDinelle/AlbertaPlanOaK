import * as THREE from 'three/webgpu';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec } from '@kit/grid';
import { blendTokens, envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, LegendSpec, SceneContext, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import { CORRIDOR_GOALS, CORRIDOR_GX, CORRIDOR_GY, CORRIDOR_HELD, CORRIDOR_OPEN, corridorBit } from '@sim/envs/corridor';
import type { GridCell, GridGeometry } from '@sim/types';
import { formatCount, formatSigned } from '../../../format';
import { ACTIONS, accentNamed, agentCell, eventSteps, keyCell } from '../tape';
import {
  GATE_FLARE_SECONDS,
  GATE_SECONDS,
  buildCarryRack,
  buildGate,
  buildKeyAltar,
  buildShrine,
  type CarryRackHandle,
  type GateHandle,
  type KeyAltarHandle,
  type ShrineHandle,
} from './actors';
import { CHAMBERS, buildWorld, chamberBounds, gateFacesX, resolveDoors, resolveGrid, templeSkin, type WorldHandle } from './world';

const BURST_SCALE = 0.85;
const RIPPLE_SCALE = 0.55;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.62;
const BLOOM_SECONDS = 1.5;
const KEY_BURST_SCALE = 0.34;
const KEY_BURST_Y = 0.42;
const DOOR_BURST_SCALE = 0.42;
const DOOR_BURST_Y = 0.6;
const MID_SECONDS = 0.9;
const GROW_SECONDS = 0.72;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.17;

const PROXIMITY_CELLS = 5;

const KEYS = CHAMBERS - 1;
const GATE_NAMES = ['bramble gate', 'river gate', 'mist gate'] as const;
const KEY_TINTS = [
  accentNamed(TOKENS.accents, 'KeyCorridor'),
  blendTokens(TOKENS.palettes.noon.sun, TOKENS.palettes.morning.hemiSky, 0.16).getHex(),
  accentNamed(TOKENS.accents, 'MountainCar'),
] as const;

interface TempleView {
  agent: GridCell;
  goal: GridCell;
  hasKey: boolean[];
  doorOpen: boolean[];
  keys: GridCell[];
}

function makeView(): TempleView {
  return {
    agent: { x: 0, y: 0 },
    goal: { x: 0, y: 0 },
    hasKey: [false, false, false],
    doorOpen: [false, false, false],
    keys: [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }],
  };
}

function readState(pose: Float32Array, out: TempleView): TempleView {
  agentCell(pose, out.agent);
  out.goal.x = pose[CORRIDOR_GX]!;
  out.goal.y = pose[CORRIDOR_GY]!;
  for (let j = 0; j < KEYS; j++) {
    out.hasKey[j] = corridorBit(pose[CORRIDOR_HELD]!, j);
    out.doorOpen[j] = corridorBit(pose[CORRIDOR_OPEN]!, j);
    keyCell(pose, j, out.keys[j]!);
  }
  return out;
}

function gateStatus(view: TempleView, j: number): string {
  return view.doorOpen[j] ? 'open' : view.hasKey[j] ? 'key carried' : 'locked';
}

export class TempleOfThreeGatesScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['morning', 'fog'] } };
  readonly controlHz = 8;
  private sceneView: SceneView | null = null;

  protected readonly stageName = 'templeOfThreeGates';

  private wisp: Wisp | null = null;
  private gates: GateHandle[] = [];
  private altars: KeyAltarHandle[] = [];
  private rack: CarryRackHandle | null = null;
  private shrine: ShrineHandle | null = null;
  private grid!: GridGeometry;
  private doors: readonly GridCell[] = [];

  private growAt = -Infinity;
  private bloomAt = -Infinity;
  private midAt = -Infinity;
  private readonly doorAt: number[] = [-Infinity, -Infinity, -Infinity];

  private readonly view = makeView();
  private readonly steps: number[] = [];
  private readonly eventCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly eventPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private keyColors: THREE.Color[] = [];
  private relicColor = new THREE.Color(TOKENS.celebrationGold);
  private accentColor = new THREE.Color(TOKENS.celebrationGold);
  private readonly ledger = { action: 0, reward: 0, goals: 0 };

  override legend: LegendSpec = {
    title: 'TEMPLE',
    rows: [
      { key: 'at', label: 'wisp', tint: KEY_TINTS[0], value: () => `column ${this.view.agent.x + 1} · row ${this.view.agent.y + 1}` },
      ...GATE_NAMES.map((label, j) => ({ key: `gate${j}`, label, tint: KEY_TINTS[j]!, value: () => gateStatus(this.view, j) })),
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => ACTIONS[this.ledger.action]! },
      { key: 'reward', label: 'reward', tint: TOKENS.celebrationGold, value: () => formatSigned(this.ledger.reward) },
    ],
    note: () => `${formatCount(this.ledger.goals)} relics`,
  };

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.doors = resolveDoors(grid);
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const doors = this.doors;

    const palette = ctx.tokens.palettes.morning;
    const skin = templeSkin(palette, ctx.tokens);
    const accent = this.accentOf(ctx);
    this.accentColor = new THREE.Color(accent);
    this.keyColors = skin.keys.map((c) => c.clone());
    this.relicColor = skin.relic.clone();

    doors.forEach((cell, j) => {
      const gate = buildGate({ skin, facesX: gateFacesX(grid, cell), name: `gate${j}` }, j);
      cellToWorldVec(cell.x, cell.y, grid, gate.root.position);
      gate.setProgress(0);
      this.scene.add(gate.group);
      this.gates.push(gate);
    });

    for (let j = 0; j < KEYS; j++) {
      const altar = buildKeyAltar({ skin, index: j, stone: world.stoneMaterial, name: `keyAltar${j}` });
      this.scene.add(altar.group);
      this.altars.push(altar);
    }

    const shrine = buildShrine({ palette, skin, stone: world.stoneMaterial });
    this.shrine = shrine;
    this.scene.add(shrine.group);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.26,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'templeWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const rack = buildCarryRack(skin);
    this.rack = rack;
    wisp.carry(rack.group);

    world.rig.points[0]?.color.copy(this.relicColor);
    world.rig.points[1]?.color.copy(this.accentColor);
    world.rig.points[2]?.color.set(ctx.tokens.celebrationGold);
    const sanctumLight = world.rig.points[3];
    if (sanctumLight) {
      const b = chamberBounds(grid, doors, CHAMBERS - 1);
      sanctumLight.color.set(skin.chambers[CHAMBERS - 1]!.bloom);
      sanctumLight.position.set(b.cx, 1.9, b.cz);
      sanctumLight.intensity = 0.85;
    }
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    const grid = this.grid;
    const halfX = grid.w * CELL / 2;
    const halfZ = grid.h * CELL / 2;
    this.sceneView = makeSceneView(ctx, { min: [-halfX, 0, -halfZ], max: [halfX, 1.8, halfZ] }, [0, 0.35, 0], { distance: 15.5, azimuth: 0.14, polar: 0.58 });
    return this.sceneView.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const overlay = makeView();
    const keys: GridCell[] = [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }];
    const read = (state: FrameState) => readState(state.pose, overlay);
    return {
      agentCell: (state) => read(state).agent,
      goalCell: (state) => read(state).goal,
      keyCells: (state) => {
        const v = read(state);
        for (let j = 0; j < KEYS; j++) {
          const at = v.hasKey[j] ? v.agent : v.keys[j]!;
          keys[j]!.x = at.x;
          keys[j]!.y = at.y;
        }
        return keys;
      },
      doorStates: (state) => read(state).doorOpen,
      maxKeys: KEYS,
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    this.wisp?.setCell(0, 0, this.grid);
    for (const gate of this.gates) {
      gate.setProgress(0);
      gate.setFlare(0);
      gate.setTime(0);
    }
    for (const altar of this.altars) {
      altar.setGrow(1);
      altar.setCarried(false);
      altar.setTime(0);
    }
    this.rack?.setTime(0);
    this.shrine?.setState(0, 1);
    this.shrine?.setTime(0);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const shrine = this.shrine;
    const rack = this.rack;
    if (!wisp || !shrine || !rack) return;

    const grid = this.grid;
    const view = readState(frame.state.pose, this.view);
    this.ledger.action = Math.trunc(frame.frameAt(frame.state.step).actions[0]!) & 3;
    this.ledger.reward = frame.state.reward;
    this.ledger.goals = frame.state.pose[CORRIDOR_GOALS]!;

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      this.bloomAt = -Infinity;
      this.midAt = -Infinity;
      this.growAt = -Infinity;
      this.doorAt.fill(-Infinity);
      wisp.setCell(view.agent.x, view.agent.y, grid);
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
        pool.burst(this.bloomPos, this.relicColor, BURST_SCALE, time);
        pool.ripple(this.groundPos, ctx.tokens.celebrationGold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        shrine.flare(1);
        this.bloomAt = time;
        this.growAt = time;
        world.rig.points[2]?.color.copy(this.relicColor);
      } else if (ev.type === 'key') {
        const cell = keyCell(frame.poseAt(steps[i]!), ev.index, this.eventCell);
        cellToWorldVec(cell.x, cell.y, grid, this.eventPos);
        this.eventPos.y = KEY_BURST_Y;
        pool.burst(this.eventPos, this.keyColors[ev.index] ?? this.accentColor, KEY_BURST_SCALE, time);
        this.midAt = time;
        world.rig.points[2]?.color.copy(this.keyColors[ev.index] ?? this.accentColor);
        wisp.flare(0.55);
      } else if (ev.type === 'door') {
        this.doorAt[ev.index] = time;
        const cell = this.doors[ev.index];
        if (ev.open && cell) {
          cellToWorldVec(cell.x, cell.y, grid, this.eventPos);
          this.eventPos.y = DOOR_BURST_Y;
          pool.burst(this.eventPos, this.keyColors[ev.index] ?? this.accentColor, DOOR_BURST_SCALE, time);
          this.midAt = time;
          world.rig.points[2]?.color.copy(this.keyColors[ev.index] ?? this.accentColor);
        }
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    if (teleported) {
      wisp.setCell(view.agent.x, view.agent.y, grid);
      wisp.snap();
      this.doorAt.fill(-Infinity);
      this.growAt = time;
    }

    if (view.agent.x !== wisp.cellX || view.agent.y !== wisp.cellY) {
      wisp.hopTo(view.agent, time);
    }
    wisp.update(frame.dtReal, time);

    for (let j = 0; j < this.gates.length; j++) {
      const gate = this.gates[j]!;
      const open = view.doorOpen[j] === true;
      const age = time - this.doorAt[j]!;
      gate.setProgress(gatePhase(open, age));
      gate.setFlare(open ? envelope(age, GATE_FLARE_SECONDS, 2.6) : 0);
      gate.setTime(time);
    }

    const grow = growth(time - this.growAt);
    for (let j = 0; j < this.altars.length; j++) {
      const altar = this.altars[j]!;
      const cell = view.keys[j]!;
      cellToWorldVec(cell.x, cell.y, grid, altar.root.position);
      const carried = view.hasKey[j] === true;
      altar.setCarried(carried);
      altar.setGrow(grow);
      altar.setTime(time);
      rack.setCarried(j, carried);
    }
    rack.setTime(time);

    cellToWorldVec(view.goal.x, view.goal.y, grid, shrine.root.position);
    const near = proximity(view.agent, view.goal);
    shrine.setState(near, grow);
    shrine.setTime(time);
    shrine.update(frame.dtReal, time);

    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const mid = envelope(time - this.midAt, MID_SECONDS, 3.2);
    const relicLight = world.rig.points[0];
    if (relicLight) {
      relicLight.position.set(shrine.root.position.x, shrine.crownHeight + 0.1, shrine.root.position.z);
      relicLight.intensity = (0.18 + 0.5 * near) * grow + 1.6 * bloom;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.16 + 0.7 * bloom + 0.35 * mid;
    }
    const eventLight = world.rig.points[2];
    if (eventLight) {
      eventLight.position.copy(bloom >= mid ? this.bloomPos : this.eventPos);
      eventLight.intensity = 2.4 * bloom + 1.5 * mid;
    }

    world.setTime(time);

    if (this.sceneView) world.setView(this.sceneView.update(rig, frame.dtReal), this.camera);
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.rig && this.sceneView) this.sceneView.resize(size, this.rig);
  }

  protected drop(): void {
    this.shrine?.dispose();
    this.shrine = null;
    for (const gate of this.gates) gate.dispose();
    this.gates = [];
    for (const altar of this.altars) altar.dispose();
    this.altars = [];
    this.wisp?.carry(null);
    this.rack?.dispose();
    this.rack = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.doors = [];
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
    this.midAt = -Infinity;
    this.doorAt.fill(-Infinity);
  }
}

function gatePhase(open: boolean, age: number): number {
  const p = !(age >= 0) || age >= GATE_SECONDS ? 1 : age / GATE_SECONDS;
  return open ? p : 1 - p;
}

function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.16 * Math.sin(Math.PI * t) * (1 - t));
}

function proximity(agent: GridCell, goal: GridCell): number {
  const d = Math.max(Math.abs(agent.x - goal.x), Math.abs(agent.y - goal.y));
  return d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
}
