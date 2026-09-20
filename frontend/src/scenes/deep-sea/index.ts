import { lastEvent } from '@sim/events';
import { PULSE_SECONDS } from '@kit/celebrate';
import { CELL } from '@kit/grid';
import { makeSceneView, type SceneView } from '@kit/view';
import type { GroundTruthOptions } from '@kit/debug';
import type { GridSize } from '@kit/grid';
import { clamp01, envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import { buildBubble, buildClam, type BubbleHandle, type ClamHandle } from './actors';
import {
  cellToReefVec,
  isDiveStep,
  readDive,
  reefDepthYAt,
  reefReachable,
  reefShear,
  resolveGrid,
  type DiveCell,
} from './reef';
import { buildWorld, treasureCell, type WorldHandle } from './world';


const BURST_SCALE = 0.78;
const RIPPLE_SCALE = 0.44;
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

class SunlitDescentScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'sunlitDescent';
  private view: SceneView | null = null;

  private wisp: Wisp | null = null;
  private bubble: BubbleHandle | null = null;
  private clam: ClamHandle | null = null;
  private readonly diver = new THREE.Group();
  private grid!: GridGeometry;
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

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    const half = Math.max(grid.w, grid.h) * CELL * 0.5 + 0.8;
    this.view = makeSceneView(ctx, { min: [-half, -4.8, -half], max: [half, 2, half] }, [0.05, -2, 1.65], { distance: half * 3, azimuth: 0.78, polar: 0.82 });
    this.reach = reefReachable(grid);
    const goal = treasureCell(grid);
    this.treasure = { x: goal.c, y: goal.r };
    return buildWorld(ctx, this.scene, grid, this.reach);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const palette = ctx.tokens.palettes.underwater;
    const accent = this.accentOf(ctx);
    this.gold = ctx.tokens.celebrationGold;
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
        skirt: false,
        silhouette: false,
        name: 'diverWisp',
      },
      grid,
    );
    this.wisp = wisp;
    wisp.core.visible = false;
    const halo = wisp.group.getObjectByName('oakWispHalo');
    if (halo) halo.visible = false;
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
  }

  protected rigOptions(_ctx: SceneContext): OrbitRigOptions {
    if (!this.view) throw new Error('Scene view is missing');
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const overlayDive = this.overlayDive;
    const overlayCell = this.overlayCell;
    const overlayGrid: GridSize = this.grid;
    return {
      agentCell: (state) => {
        readDive(state, overlayGrid, overlayDive);
        overlayCell.x = overlayDive.c;
        overlayCell.y = overlayDive.r;
        return overlayCell;
      },
      goalCell: () => this.treasure,
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    const wisp = this.wisp;
    const overlay = this.overlay;
    if (!wisp || !overlay) return;
    reefShear(this.grid, overlay.group.matrix);
    overlay.group.matrixAutoUpdate = false;
    overlay.group.matrixWorldNeedsUpdate = true;

    world.setTime(0);
    if (this.view && this.rig) world.setView(this.view.update(this.rig, 0), this.camera);
    wisp.setCell(0, 0, this.grid);
    this.diver.position.y = reefDepthYAt(wisp.position.z, this.grid);
    this.bubble?.setTime(0);
    this.bubble?.setStress(0);
    this.clam?.setState(GAPE_BASE, 0);
    this.clam?.setTime(0);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const bubble = this.bubble;
    const clam = this.clam;
    if (!wisp || !bubble || !clam) return;

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
      const step = lastEvent(frame.eventLog, 'goal', state.step);
      if (step !== undefined) {
        const at = step / frame.controlHz;
        if (time - at <= GAPE_SECONDS) {
          this.goalAt = at;
          cellToReefVec(this.treasure.x, this.treasure.y, grid, this.bloomPos);
          this.groundPos.copy(this.bloomPos); this.bloomPos.y += clam.pearlHeight;
          pool.burst(this.bloomPos, this.pearlHue, BURST_SCALE, at);
          pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, at);
          pool.pulse.reset(); pool.pulse.fire(Math.max(0, PULSE_STRENGTH - (time - at) / PULSE_SECONDS));
          pool.update(0, time);
        }
      }
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
    this.clam?.dispose();
    this.clam = null;
    this.bubble?.dispose();
    this.bubble = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.diver.removeFromParent();
    this.diver.clear();
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

export default ((() => new SunlitDescentScene()) satisfies SceneFactory);
