
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
import type { GridCell, GridGeometry } from '@sim/types';
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
  buildThumbGateBank,
  type CarryRackHandle,
  type GateHandle,
  type KeyAltarHandle,
  type ShrineHandle,
  type ThumbGateBank,
} from './actors';
import { manifest } from './manifest';
import {
  CAMERA_AZIMUTH,
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


const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.58;
const CAM_DISTANCE = 15.5;
const CAM_TARGET_Y = 0.35;
const CAM_YAW_RANGE = 0.55;
const CAM_DISTANCE_THUMB = 20.0;
const CAM_AZIMUTH_THUMB = 0.12;
const CAM_POLAR_THUMB = 0.56;

const BURST_SCALE = 0.85;
const RIPPLE_SCALE = 0.55;
const BURST_PARTICLES = 120;
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

function seedKeys(view: TempleView, grid: GridGeometry): void {
  const keys = resolveKeys(grid);
  for (let j = 0; j < KEYS; j++) {
    const cell = view.keys[j] as GridCell;
    const source = keys[j];
    if (!source) continue;
    cell.x = source.x;
    cell.y = source.y;
    clampCell(cell, grid);
  }
}

interface MkcLike {
  x?: unknown;
  y?: unknown;
  gx?: unknown;
  gy?: unknown;
  hasKey?: unknown;
  doorOpen?: unknown;
  keys?: unknown;
}

function bit(mask: number, i: number): boolean {
  return ((mask >> i) & 1) !== 0;
}

function cellOf(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : fallback;
}

function readState(state: FrameState, grid: GridSize, out: TempleView): void {
  const d = state.derived as MkcLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  out.x = cellOf(has ? d.x : undefined, Math.round(state.pose[0] ?? 0));
  out.y = cellOf(has ? d.y : undefined, Math.round(state.pose[1] ?? 0));
  out.gx = cellOf(has ? d.gx : undefined, Math.round(state.pose[2] ?? 0));
  out.gy = cellOf(has ? d.gy : undefined, Math.round(state.pose[3] ?? 0));

  const keyMask = Math.round(state.pose[4] ?? 0);
  const doorMask = Math.round(state.pose[5] ?? 0);
  const hasKey = has && Array.isArray(d.hasKey) ? (d.hasKey as unknown[]) : null;
  const doorOpen = has && Array.isArray(d.doorOpen) ? (d.doorOpen as unknown[]) : null;
  for (let j = 0; j < KEYS; j++) {
    out.hasKey[j] = hasKey ? hasKey[j] === true : bit(keyMask, j);
    out.doorOpen[j] = doorOpen ? doorOpen[j] === true : bit(doorMask, j);
  }

  const keys = has && Array.isArray(d.keys) ? (d.keys as unknown[]) : null;
  for (let j = 0; j < KEYS; j++) {
    const cell = out.keys[j] as GridCell;
    const source = keys ? (keys[j] as { x?: unknown; y?: unknown } | undefined) : undefined;
    if (source) {
      cell.x = cellOf(source.x, cell.x);
      cell.y = cellOf(source.y, cell.y);
    }
    clampCell(cell, grid);
  }

  out.x = clampInt(out.x, grid.w);
  out.y = clampInt(out.y, grid.h);
  out.gx = clampInt(out.gx, grid.w);
  out.gy = clampInt(out.gy, grid.h);
}

function clampInt(v: number, extent: number): number {
  return v < 0 ? 0 : v > extent - 1 ? extent - 1 : v;
}

export function readGoalCell(state: FrameState, grid: GridSize, out: GridCell): void {
  const d = state.derived as MkcLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  out.x = cellOf(has ? d.gx : undefined, Math.round(state.pose[2] ?? 0));
  out.y = cellOf(has ? d.gy : undefined, Math.round(state.pose[3] ?? 0));
  clampCell(out, grid);
}

class TempleOfThreeGatesScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private gates: GateHandle[] = [];
  private gateBank: ThumbGateBank | null = null;
  private altars: KeyAltarHandle[] = [];
  private rack: CarryRackHandle | null = null;
  private shrine: ShrineHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });
  private doors: readonly GridCell[] = [];

  private growAt = -Infinity;
  private bloomAt = -Infinity;
  private midAt = -Infinity;
  private readonly doorAt: number[] = [-Infinity, -Infinity, -Infinity];

  private readonly view = makeView();
  private readonly prevGoal: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly eventPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private keyColors: THREE.Color[] = [];
  private relicColor = new THREE.Color(0xffd166);
  private accentColor = new THREE.Color(0xd9b36a);

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'templeOfThreeGatesScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    const doors = resolveDoors(grid);
    this.doors = doors;
    seedKeys(this.view, grid);
    const openingGoal = resolveGoal(grid);

    const palette = ctx.tokens.palettes.morning;
    const skin = templeSkin(palette, ctx.tokens);
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.accentColor = new THREE.Color(accent);
    this.keyColors = skin.keys.map((c) => c.clone());
    this.relicColor = skin.relic.clone();

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);

    const gateCount = Math.min(doors.length, 3);
    const builders = [buildBrambleGate, buildRiverGate, buildMistGate];
    const bank =
      ctx.tier === 'thumb'
        ? buildThumbGateBank({
            tier: ctx.tier,
            skin,
            facesX: doors.slice(0, gateCount).map((cell) => gateFacesX(grid, cell)),
            count: gateCount,
            name: 'thumbGateBank',
          })
        : null;
    this.gateBank = bank;
    if (bank) this.scene.add(bank.group);
    for (let j = 0; j < gateCount; j++) {
      const cell = doors[j] as GridCell;
      const build = builders[j] as (typeof builders)[number];
      const gate =
        bank?.gates[j] ??
        build({
          tier: ctx.tier,
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
        tier: ctx.tier,
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

    const shrine = buildShrine({ tier: ctx.tier, palette, skin, stone: world.stoneMaterial });
    this.shrine = shrine;
    cellToWorldVec(openingGoal.x, openingGoal.y, grid, shrine.root.position);
    this.scene.add(shrine.group);

    const wisp = new Wisp(
      {
        color: accent,
        tier: ctx.tier,
        radius: 0.26,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'templeWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const rack = buildCarryRack(skin, ctx.tier);
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

    const thumb = ctx.tier === 'thumb';
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [0, CAM_TARGET_Y, 0],
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: thumb ? CAM_AZIMUTH_THUMB : CAM_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0012,
      name: 'templeOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'templeCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    const overlayView = makeView();
    seedKeys(overlayView, grid);
    const readOverlay = (state: FrameState): TempleView => {
      readState(state, grid, overlayView);
      return overlayView;
    };
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
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
      name: 'templeGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
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
    rack.setTime(0);
    shrine.setState(0, 1);
    shrine.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const shrine = this.shrine;
    const rack = this.rack;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !shrine || !rack || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const view = this.view;

    pool.update(frame.dtReal, time);

    readState(state, grid, view);

    if (frame.scrubbed) {
      pool.clear();
      this.bloomAt = -Infinity;
      this.midAt = -Infinity;
      this.growAt = -Infinity;
      for (let j = 0; j < KEYS; j++) this.doorAt[j] = -Infinity;
      wisp.setCell(view.x, view.y, grid);
      wisp.snap();
      rig.snap();
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
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
        const cell = view.keys[ev.index];
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

    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
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
    this.overlay?.dispose();
    this.overlay = null;
    this.shrine?.dispose();
    this.shrine = null;
    for (const gate of this.gates) gate.dispose();
    this.gates = [];
    this.gateBank?.dispose();
    this.gateBank = null;
    for (const altar of this.altars) altar.dispose();
    this.altars = [];
    this.wisp?.carry(null);
    this.rack?.dispose();
    this.rack = null;
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
    this.doors = [];
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
    this.midAt = -Infinity;
    for (let j = 0; j < KEYS; j++) this.doorAt[j] = -Infinity;
  }
}


export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
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

const createScene: SceneFactory = () => new TempleOfThreeGatesScene();
export default createScene;
