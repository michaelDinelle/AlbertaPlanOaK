
import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import type { GridSize } from '@kit/grid';
import { makeOrbitRig, type OrbitRig } from '@kit/rigs';
import { TOKENS } from '@kit/tokens';
import type {
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { buildBubble, buildClam, type BubbleHandle, type ClamHandle } from './actors';
import { manifest } from './manifest';
import {
  cellToReefVec,
  clamp01,
  envelope,
  isDiveStep,
  readDive,
  reefDepthYAt,
  reefReachable,
  reefShear,
  resolveGrid,
  type DiveCell,
} from './reef';
import { CAMERA_AZIMUTH, buildWorld, treasureCell, type WorldHandle } from './world';


const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 0.82;
const CAM_DISTANCE = 17.6;
const CAM_TARGET: readonly [number, number, number] = [-0.85, -2.08, 1.05];

const BURST_SCALE = 0.78;
const RIPPLE_SCALE = 0.44;
const BURST_PARTICLES = 96;
const PULSE_STRENGTH = 0.52;
const FLARE_SECONDS = 1.5;
const GAPE_SECONDS = 2.8;
const RUSH_SECONDS = 0.85;
const RUSH_CENTER_Y = 0.4;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.06;

const GAPE_BASE = 0.3;
const GAPE_BREATH = 0.12;
const GAPE_PROXIMITY = 0.52;
const PROXIMITY_COLS = 3;

class SunlitDescentScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private bubble: BubbleHandle | null = null;
  private clam: ClamHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private readonly diver = new THREE.Group();
  private grid: GridGeometry = resolveGrid({});
  private reach: Uint8Array = new Uint8Array(0);
  private treasure: GridCell = { x: 0, y: 0 };

  private goalAt = -Infinity;
  private resurfaceAt = -Infinity;
  private stir = 0;

  private readonly dive: DiveCell = { r: 0, c: 0, lastA: 0 };
  private readonly overlayDive: DiveCell = { r: 0, c: 0, lastA: 0 };
  private readonly overlayCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly diverPos = new THREE.Vector3();
  private pearlHue = 0xffd166;
  private gold = 0xffd166;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'sunlitDescentScene';

    const grid = resolveGrid(ctx.geometry, ctx.seed);
    this.grid = grid;
    this.reach = reefReachable(grid);
    const goal = treasureCell(grid);
    this.treasure = { x: goal.c, y: goal.r };

    const palette = ctx.tokens.palettes.underwater;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.uiCyan;
    this.gold = ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid, this.reach);
    this.world = world;
    this.scene.add(world.group);
    this.pearlHue = world.skin.pearl.getHex();

    const pearlLight = world.rig.points[0];
    if (pearlLight) pearlLight.color.set(world.skin.pearl);
    const diverLight = world.rig.points[1];
    if (diverLight) diverLight.color.set(accent);
    const bloomLight = world.rig.points[2];
    if (bloomLight) bloomLight.color.set(world.skin.pearl);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.19,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        ribbon: false,
        name: 'diverWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.diver.name = 'diverColumn';
    this.diver.add(wisp.group);
    this.scene.add(this.diver);

    const bubble = buildBubble({
      palette,
      skin: world.skin,
      accent,
      height: ctx.tokens.wispHeight,
      name: 'diverBubble',
    });
    this.bubble = bubble;
    wisp.avatar.add(bubble.group);

    const clam = buildClam({
      palette,
      skin: world.skin,
      facing: Math.PI * 0.26,
      name: 'giantClam',
    });
    this.clam = clam;
    cellToReefVec(this.treasure.x, this.treasure.y, grid, clam.root.position);
    this.scene.add(clam.group);

    const rig = makeOrbitRig({
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
      name: 'reefOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'reefCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    const overlayDive = this.overlayDive;
    const overlayCell = this.overlayCell;
    const overlayGrid: GridSize = grid;
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        readDive(state, overlayGrid, overlayDive);
        overlayCell.x = overlayDive.c;
        overlayCell.y = overlayDive.r;
        return overlayCell;
      },
      goalCell: () => this.treasure,
      name: 'reefGroundTruth',
    });
    reefShear(grid, this.overlay.group.matrix);
    this.overlay.group.matrixAutoUpdate = false;
    this.overlay.group.matrixWorldNeedsUpdate = true;
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    wisp.setCell(0, 0, grid);
    this.diver.position.y = reefDepthYAt(wisp.position.z, grid);
    bubble.setTime(0);
    bubble.setStress(0);
    clam.setState(GAPE_BASE, 0);
    clam.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const bubble = this.bubble;
    const clam = this.clam;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !bubble || !clam || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const dive = this.dive;

    pool.update(frame.dtReal, time);

    readDive(state, grid, dive);

    if (frame.scrubbed) {
      pool.clear();
      this.goalAt = -Infinity;
      this.resurfaceAt = -Infinity;
      this.stir = dive.lastA;
      wisp.setCell(dive.c, dive.r, grid);
      wisp.snap();
      rig.snap();
    }

    let hardCut = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        cellToReefVec(this.treasure.x, this.treasure.y, grid, this.bloomPos);
        this.groundPos.copy(this.bloomPos);
        this.bloomPos.y += clam.pearlHeight;
        pool.burst(this.bloomPos, this.pearlHue, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(0.85);
        this.goalAt = time;
      } else if (ev.type === 'resurface') {
        world.rush.points.position.set(wisp.position.x, RUSH_CENTER_Y, wisp.position.z);
        this.resurfaceAt = time;
        hardCut = true;
      } else if (ev.type === 'teleport') {
        hardCut = true;
      }
    }

    if (hardCut || !isDiveStep(wisp.cellX, wisp.cellY, dive)) {
      if (hardCut || dive.c !== wisp.cellX || dive.r !== wisp.cellY) {
        wisp.setCell(dive.c, dive.r, grid);
        wisp.snap();
      }
    } else {
      wisp.hopTo({ x: dive.c, y: dive.r }, time);
    }
    wisp.update(frame.dtReal, time);

    this.diver.position.y = reefDepthYAt(wisp.position.z, grid);
    this.diverPos.set(wisp.position.x, wisp.position.y + this.diver.position.y, wisp.position.z);

    const target = dive.lastA === 1 ? 1 : 0;
    this.stir += (target - this.stir) * (1 - Math.exp(-9 * Math.max(0, frame.dtReal)));
    bubble.setStress(this.stir);
    bubble.setTime(time);
    world.silt.points.position.set(
      this.diverPos.x,
      this.diverPos.y + ctx.tokens.wispHeight * 0.6,
      this.diverPos.z,
    );
    world.silt.setDensity(0.05 + 0.75 * this.stir);

    const rush = envelope(time - this.resurfaceAt, RUSH_SECONDS, 2.6);
    world.rush.setDensity(rush);
    world.rush.setIntensity(1.1 + 1.6 * rush);

    const flare = envelope(time - this.goalAt, FLARE_SECONDS, 2.2);
    const held = envelope(time - this.goalAt, GAPE_SECONDS, 1.1);
    const open = clamp01(gape(time, dive, grid) + 0.8 * held);
    clam.setState(open, flare);
    clam.setTime(time);

    const pearlLight = world.rig.points[0];
    if (pearlLight) {
      pearlLight.position.set(
        clam.root.position.x,
        clam.root.position.y + clam.pearlHeight,
        clam.root.position.z,
      );
      pearlLight.intensity = 0.16 + 0.5 * open;
    }
    const diverLight = world.rig.points[1];
    if (diverLight) {
      diverLight.position.set(
        this.diverPos.x,
        this.diverPos.y + ctx.tokens.wispHeight,
        this.diverPos.z,
      );
      diverLight.intensity = 0.5 + 0.7 * rush;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 3.2 * flare;
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
    this.clam?.dispose();
    this.clam = null;
    this.bubble?.dispose();
    this.bubble = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.diver.removeFromParent();
    this.diver.clear();
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.goalAt = -Infinity;
    this.resurfaceAt = -Infinity;
    this.stir = 0;
  }
}


export function gape(time: number, dive: DiveCell, grid: GridSize): number {
  const depth = grid.h > 1 ? clamp01(dive.r / (grid.h - 1)) : 0;
  const away = Math.abs(dive.c - (grid.w - 1));
  const near = away >= PROXIMITY_COLS ? 0 : 1 - away / PROXIMITY_COLS;
  const breath = 0.5 + 0.5 * Math.sin(time * 0.55);
  return clamp01(GAPE_BASE + GAPE_BREATH * breath + GAPE_PROXIMITY * depth * near);
}

const createScene: SceneFactory = () => new SunlitDescentScene();
export default createScene;
