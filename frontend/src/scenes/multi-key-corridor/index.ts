import { philoxDraw } from '@sim/philox';
import { makeSceneView, type SceneView } from '../jelly-bean-world/view';
import { MKC_CARRY, MKC_GATES, MKC_GX, MKC_GY, MKC_X, MKC_Y, mkcBit } from '@sim/envs/multikeycorridor';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec, clampCell, type GridSize } from '@kit/grid';
import { envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { Wisp } from '@kit/wisp';
import type { EventStream, GridCell, GridGeometry } from '@sim/types';
import * as THREE from 'three';
import {
  GATE_FLARE_SECONDS,
  GATE_SECONDS,
  buildBrambleGate,
  buildCarryRack,
  buildKeyAltar,
  buildMistGate,
  buildRiverGate,
  buildShrine,
  type CarryRackHandle,
  type GateHandle,
  type KeyAltarHandle,
  type ShrineHandle,
} from './actors';
import {
  buildWorld,
  chamberBounds,
  gateFacesX,
  resolveDoors,
  resolveGoal,
  resolveGrid,
  resolveKeys,
  templeSkin,
  type WorldHandle,
} from './world';


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

const KEYS = 3;

interface TempleView {
  x: number;
  y: number;
  gx: number;
  gy: number;
  hasKey: boolean[];
  doorOpen: boolean[];
  keys: GridCell[];
}

function makeView(): TempleView {
  return {
    x: 0,
    y: 0,
    gx: 0,
    gy: 0,
    hasKey: [false, false, false],
    doorOpen: [false, false, false],
    keys: [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 0, y: 0 },
    ],
  };
}

export class KeyHistory {
  private stream: EventStream | null = null;
  private taken = 0;
  private readonly goals: number[] = [];

  sync(stream: EventStream): void {
    if (this.stream !== stream || stream.count < this.taken) {
      this.stream = stream;
      this.taken = 0;
      this.goals.length = 0;
    }
    const goal = stream.types.indexOf('goal');
    while (this.taken < stream.count) {
      const at = this.taken++;
      if (stream.kinds[at] === goal) this.goals.push(stream.steps[at] as number);
    }
  }

  generationAt(step: number): number {
    let lo = 0;
    let hi = this.goals.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if ((this.goals[mid] as number) <= step) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
}

export function keyCells(seed: number, generation: number, grid: GridGeometry, out: GridCell[]): void {
  const initial = resolveKeys(grid);
  const words = generation > 0 ? philoxDraw(seed >>> 0, 3, generation - 1, 0) : null;
  for (let j = 0; j < KEYS; j++) {
    const cell = out[j] as GridCell;
    if (words) {
      const index = Math.trunc(Math.fround(((words[j + 1] as number) >>> 8) * 2 ** -24 * 15));
      cell.x = 1 + j * 4 + index % 3;
      cell.y = 1 + Math.floor(index / 3);
    } else {
      const source = initial[j];
      if (!source) throw new Error('Multi-Key Corridor has no initial key cell');
      cell.x = source.x;
      cell.y = source.y;
    }
  }
}

function readState(state: FrameState, grid: GridSize, out: TempleView): void {
  out.x = Math.round(state.pose[MKC_X] ?? 0);
  out.y = Math.round(state.pose[MKC_Y] ?? 0);
  out.gx = Math.round(state.pose[MKC_GX] ?? 0);
  out.gy = Math.round(state.pose[MKC_GY] ?? 0);

  const keyMask = Math.round(state.pose[MKC_CARRY] ?? 0);
  const doorMask = Math.round(state.pose[MKC_GATES] ?? 0);
  for (let j = 0; j < KEYS; j++) {
    out.hasKey[j] = mkcBit(keyMask, j);
    out.doorOpen[j] = mkcBit(doorMask, j);
  }
  for (let j = 0; j < KEYS; j++) clampCell(out.keys[j] as GridCell, grid);

  out.x = clampInt(out.x, grid.w);
  out.y = clampInt(out.y, grid.h);
  out.gx = clampInt(out.gx, grid.w);
  out.gy = clampInt(out.gy, grid.h);
}

function clampInt(v: number, extent: number): number {
  return v < 0 ? 0 : v > extent - 1 ? extent - 1 : v;
}

export function readGoalCell(state: FrameState, grid: GridSize, out: GridCell): void {
  out.x = Math.round(state.pose[MKC_GX] ?? 0);
  out.y = Math.round(state.pose[MKC_GY] ?? 0);
  clampCell(out, grid);
}

class TempleOfThreeGatesScene extends StageScene<WorldHandle> {
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
  private readonly history = new KeyHistory();
  private readonly eventKeys: GridCell[] = [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }];
  private readonly prevGoal: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly eventPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private keyColors: THREE.Color[] = [];
  private relicColor = new THREE.Color(0xffd166);
  private accentColor = new THREE.Color(0xd9b36a);

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.doors = resolveDoors(grid);
    keyCells(ctx.seed, 0, grid, this.view.keys);
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const doors = this.doors;
    const openingGoal = resolveGoal(grid);

    const palette = ctx.tokens.palettes.morning;
    const skin = templeSkin(palette, ctx.tokens);
    const accent = this.accentOf(ctx);
    this.accentColor = new THREE.Color(accent);
    this.keyColors = skin.keys.map((c) => c.clone());
    this.relicColor = skin.relic.clone();

    const gateCount = Math.min(doors.length, 3);
    const builders = [buildBrambleGate, buildRiverGate, buildMistGate];
    for (let j = 0; j < gateCount; j++) {
      const cell = doors[j] as GridCell;
      const build = builders[j] as (typeof builders)[number];
      const gate = build({
          palette,
          skin,
          facesX: gateFacesX(grid, cell),
          stone: world.stoneMaterial,
          name: `gate${j}`,
        });
      cellToWorldVec(cell.x, cell.y, grid, gate.root.position);
      gate.setProgress(gate.progress);
      this.scene.add(gate.group);
      this.gates.push(gate);
    }

    for (let j = 0; j < KEYS; j++) {
      const altar = buildKeyAltar({
        palette,
        skin,
        index: j,
        stone: world.stoneMaterial,
        name: `keyAltar${j}`,
      });
      const cell = this.view.keys[j] as GridCell;
      cellToWorldVec(cell.x, cell.y, grid, altar.root.position);
      this.scene.add(altar.group);
      this.altars.push(altar);
    }

    const shrine = buildShrine({ palette, skin, stone: world.stoneMaterial });
    this.shrine = shrine;
    cellToWorldVec(openingGoal.x, openingGoal.y, grid, shrine.root.position);
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

    const relicLight = world.rig.points[0];
    if (relicLight) relicLight.color.copy(this.relicColor);
    const wispLight = world.rig.points[1];
    if (wispLight) wispLight.color.copy(this.accentColor);
    const eventLight = world.rig.points[2];
    if (eventLight) eventLight.color.set(ctx.tokens.celebrationGold);
    const sanctumLight = world.rig.points[3];
    if (sanctumLight) {
      const b = chamberBounds(grid, 3);
      sanctumLight.color.set(skin.chambers[3]?.bloom ?? accent);
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
    const grid = this.grid;
    const overlayView = makeView();

    const readOverlay = (state: FrameState): TempleView => {
      readState(state, grid, overlayView);
      keyCells(this.ctx?.seed ?? 0, this.history.generationAt(state.step), grid, overlayView.keys);
      return overlayView;
    };
    return {
      agentCell: (state) => {
        const v = readOverlay(state);
        return { x: v.x, y: v.y };
      },
      goalCell: (state) => {
        const v = readOverlay(state);
        return { x: v.gx, y: v.gy };
      },
      keyCells: (state) => {
        const v = readOverlay(state);
        return v.keys;
      },
      doorStates: (state) => readOverlay(state).doorOpen,
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

    const state = frame.state;
    const grid = this.grid;
    const view = this.view;

    pool.update(frame.dtReal, time);

    this.history.sync(frame.eventLog);
    readState(state, grid, view);
    const generation = this.history.generationAt(state.step);
    keyCells(ctx.seed, generation, grid, view.keys);
    let eventGeneration = generation;
    for (const event of frame.events) if (event.type === 'goal') eventGeneration--;

    if (frame.scrubbed) {
      pool.clear();
      this.bloomAt = -Infinity;
      this.midAt = -Infinity;
      this.growAt = -Infinity;
      for (let j = 0; j < KEYS; j++) this.doorAt[j] = -Infinity;
      wisp.setCell(view.x, view.y, grid);
      wisp.snap();
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        eventGeneration++;
        if (frame.prev) readGoalCell(frame.prev, grid, this.prevGoal);
        else {
          this.prevGoal.x = view.gx;
          this.prevGoal.y = view.gy;
        }
        cellToWorldVec(this.prevGoal.x, this.prevGoal.y, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.relicColor, BURST_SCALE, time);
        pool.ripple(this.groundPos, ctx.tokens.celebrationGold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        shrine.flare(1);
        this.bloomAt = time;
        this.growAt = time;
        const light = world.rig.points[2];
        if (light) light.color.copy(this.relicColor);
      } else if (ev.type === 'key') {
        keyCells(ctx.seed, eventGeneration, grid, this.eventKeys);
        const cell = this.eventKeys[ev.index];
        if (cell) {
          cellToWorldVec(cell.x, cell.y, grid, this.eventPos);
          this.eventPos.y = KEY_BURST_Y;
          pool.burst(this.eventPos, this.keyColors[ev.index] ?? this.accentColor, KEY_BURST_SCALE, time);
          this.midAt = time;
          const light = world.rig.points[2];
          if (light) light.color.copy(this.keyColors[ev.index] ?? this.accentColor);
        }
        wisp.flare(0.55);
      } else if (ev.type === 'door') {
        if (ev.index >= 0 && ev.index < KEYS) this.doorAt[ev.index] = time;
        const cell = this.doors[ev.index];
        if (ev.open && cell) {
          cellToWorldVec(cell.x, cell.y, grid, this.eventPos);
          this.eventPos.y = DOOR_BURST_Y;
          pool.burst(this.eventPos, this.keyColors[ev.index] ?? this.accentColor, DOOR_BURST_SCALE, time);
          this.midAt = time;
          const light = world.rig.points[2];
          if (light) light.color.copy(this.keyColors[ev.index] ?? this.accentColor);
        }
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    if (teleported) {
      wisp.setCell(view.x, view.y, grid);
      wisp.snap();
      for (let j = 0; j < KEYS; j++) this.doorAt[j] = -Infinity;
      this.growAt = time;
    }

    if (view.x !== wisp.cellX || view.y !== wisp.cellY) {
      wisp.hopTo(view, time);
    }
    wisp.update(frame.dtReal, time);

    for (let j = 0; j < this.gates.length; j++) {
      const gate = this.gates[j];
      if (!gate) continue;
      const open = view.doorOpen[j] === true;
      const age = time - (this.doorAt[j] ?? -Infinity);
      gate.setProgress(gatePhase(open, age));
      gate.setFlare(open ? envelope(age, GATE_FLARE_SECONDS, 2.6) : 0);
      gate.setTime(time);
    }

    const grow = growth(time - this.growAt);
    for (let j = 0; j < this.altars.length; j++) {
      const altar = this.altars[j];
      const cell = view.keys[j];
      if (!altar || !cell) continue;
      cellToWorldVec(cell.x, cell.y, grid, altar.root.position);
      const carried = view.hasKey[j] === true;
      altar.setCarried(carried);
      altar.setGrow(grow);
      altar.setTime(time);
      altar.update(frame.dtReal, time);
      rack.setCarried(j, carried);
    }
    rack.setTime(time);

    cellToWorldVec(view.gx, view.gy, grid, shrine.root.position);
    shrine.setState(proximity(view), grow);
    shrine.setTime(time);
    shrine.update(frame.dtReal, time);

    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const mid = envelope(time - this.midAt, MID_SECONDS, 3.2);
    const relicLight = world.rig.points[0];
    if (relicLight) {
      relicLight.position.set(shrine.root.position.x, shrine.crownHeight + 0.1, shrine.root.position.z);
      relicLight.intensity = (0.18 + 0.5 * proximity(view)) * grow + 1.6 * bloom;
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
      if (bloom >= mid) eventLight.position.copy(this.bloomPos);
      else eventLight.position.copy(this.eventPos);
      eventLight.intensity = 2.4 * bloom + 1.5 * mid;
    }

    world.setTime(time);

    if (this.sceneView) world.setView(this.sceneView.update(rig, frame.dtReal), this.camera);
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
    for (let j = 0; j < KEYS; j++) this.doorAt[j] = -Infinity;
  }
}

export function gatePhase(open: boolean, age: number): number {
  const p = !(age >= 0) || age >= GATE_SECONDS ? 1 : age / GATE_SECONDS;
  return open ? p : 1 - p;
}

export function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.16 * Math.sin(Math.PI * t) * (1 - t));
}

export function proximity(view: { x: number; y: number; gx: number; gy: number }): number {
  const d = Math.max(Math.abs(view.x - view.gx), Math.abs(view.y - view.gy));
  return d >= PROXIMITY_CELLS ? 0 : 1 - d / PROXIMITY_CELLS;
}

export default ((() => new TempleOfThreeGatesScene()) satisfies SceneFactory);
