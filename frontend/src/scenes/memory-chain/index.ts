import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorld, cellToWorldVec, type Vec3Tuple } from '@kit/grid';
import { clamp01, envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, SceneContext, SceneFactory } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import {
  buildCueCrystal,
  buildExitMouth,
  type CueCrystalHandle,
  type ExitMouthHandle,
} from './actors';
import {
  CAMERA_AZIMUTH,
  CHOICE_CELL,
  CUE_CELL,
  buildWorld,
  readChoice,
  readCue,
  readPos,
  resolveGrid,
  type WorldHandle,
} from './world';

const CAM_TARGET: readonly [number, number, number] = [-4, 1.0, 0];
const CAM_DISTANCE = 24.8;
const CAM_POLAR = 1.263;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.14;

const BURST_SCALE = 0.72;
const RIPPLE_SCALE = 0.62;
const PULSE_STRENGTH = 0.3;
const SIGH_STRENGTH = 0.85;

const BLOOM_Y = 1.25;

const FLARE_SECONDS = 1.35;
const DOUSE_SECONDS = 1.15;
const KINDLE_SECONDS = 0.8;
const FLOOD_SECONDS = 1.9;

const MOUTH_REST = 1;
const MOUTH_FLARE = 3.4;

class FireflyCaveScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'fireflyCave';

  private wisp: Wisp | null = null;
  private crystal: CueCrystalHandle | null = null;
  private mouthCyan: ExitMouthHandle | null = null;
  private mouthMagenta: ExitMouthHandle | null = null;
  private grid: GridGeometry = resolveGrid({});

  private kindleAt = -Infinity;
  private flareAt = -Infinity;
  private douseAt = -Infinity;
  private floodAt = -Infinity;
  private winner: 0 | 1 = 0;

  private readonly cell: GridCell = { x: 0, y: 0 };
  private readonly overlayCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: CHOICE_CELL, y: 0 };
  private readonly choicePos = new THREE.Vector3();
  private readonly bloomPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly cueColors: [THREE.Color, THREE.Color] = [
    new THREE.Color(TOKENS.cueCyan),
    new THREE.Color(TOKENS.cueMagenta),
  ];
  private cueX = 0;
  private choiceX = 0;

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.cueColors[0].set(ctx.tokens.cueCyan);
    this.cueColors[1].set(ctx.tokens.cueMagenta);
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const accent = this.accentOf(ctx);

    const anchor: Vec3Tuple = [0, 0, 0];
    this.cueX = cellToWorld(CUE_CELL, 0, grid, anchor)[0];
    this.choiceX = cellToWorld(CHOICE_CELL, 0, grid, anchor)[0];
    this.choicePos.set(this.choiceX, 0, 0);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.21,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'caveWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const crystal = buildCueCrystal({
      skin: world.skin,
      cueColors: [ctx.tokens.cueCyan, ctx.tokens.cueMagenta],
      name: 'cueCrystal',
    });
    this.crystal = crystal;
    this.scene.add(crystal.group);
    cellToWorldVec(CUE_CELL, 0, grid, crystal.root.position);

    const mouthCyan = buildExitMouth({
      skin: world.skin,
      color: ctx.tokens.cueCyan,
      side: -1,
      name: 'exitMouthCyan',
    });
    const mouthMagenta = buildExitMouth({
      skin: world.skin,
      color: ctx.tokens.cueMagenta,
      side: 1,
      name: 'exitMouthMagenta',
    });
    this.mouthCyan = mouthCyan;
    this.mouthMagenta = mouthMagenta;
    this.scene.add(mouthCyan.group, mouthMagenta.group);
    cellToWorldVec(CHOICE_CELL, 0, grid, mouthCyan.root.position);
    cellToWorldVec(CHOICE_CELL, 0, grid, mouthMagenta.root.position);

    const wispLight = world.rig.points[0];
    if (wispLight) {
      wispLight.color.set(accent);
      wispLight.intensity = 0;
    }
    const crystalLight = world.rig.points[1];
    if (crystalLight) {
      crystalLight.color.copy(this.cueColors[0]);
      crystalLight.intensity = 0;
      crystalLight.position.set(this.cueX, crystal.coreHeight, crystal.offsetZ);
    }
    const choiceLight = world.rig.points[2];
    if (choiceLight) {
      choiceLight.color.copy(this.cueColors[0]).lerp(this.cueColors[1], 0.5);
      choiceLight.intensity = 0;
      choiceLight.position.set(this.choiceX + 0.4, mouthCyan.glowHeight, 0);
    }
    const floodLight = world.rig.points[3];
    if (floodLight) {
      floodLight.color.copy(this.cueColors[0]);
      floodLight.intensity = 0;
      floodLight.position.set(this.choiceX, 1.2, 0);
    }
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    return {
      target: CAM_TARGET,
      distance: CAM_DISTANCE,
      azimuth: CAMERA_AZIMUTH,
      polar: CAM_POLAR,
      zoomSpeed: 0.0014,
    };
  }

  protected poolOptions(ctx: SceneContext): CelebrationPoolOptions {
    return {
      sighRadius: 3.4,
      sighHeight: 4,
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    const overlayCell = this.overlayCell;
    const goalCell = this.goalCell;
    return {
      agentCell: (state) => {
        overlayCell.x = readPos(state);
        overlayCell.y = 0;
        return overlayCell;
      },
      goalCell: () => goalCell,
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    this.wisp?.setCell(CUE_CELL, 0, this.grid);
    this.crystal?.setCue(0);
    this.crystal?.setGlow(1);
    this.crystal?.setTime(0);
    this.mouthCyan?.setGlow(MOUTH_REST);
    this.mouthMagenta?.setGlow(MOUTH_REST);
    this.mouthCyan?.setTime(0);
    this.mouthMagenta?.setTime(0);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const crystal = this.crystal;
    const cyan = this.mouthCyan;
    const magenta = this.mouthMagenta;
    if (!wisp || !crystal || !cyan || !magenta) return;

    const state = frame.state;
    const grid = this.grid;

    pool.update(frame.dtReal, time);

    const pos = readPos(state);
    const cue = readCue(state);
    this.cell.x = pos;
    this.cell.y = 0;

    if (frame.scrubbed) {
      pool.clear();
      this.kindleAt = -Infinity;
      this.flareAt = -Infinity;
      this.douseAt = -Infinity;
      this.floodAt = -Infinity;
      wisp.setCell(pos, 0, grid);
      wisp.snap();
      rig.snap();
    }

    let teleported = false;
    let scored = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        scored = true;
        this.winner = readChoice(state);
        const color = this.cueColors[this.winner];
        this.flareAt = time;
        this.floodAt = time;
        world.flood.setColor(color);
        this.bloomPos.set(this.choiceX, BLOOM_Y, 0);
        pool.burst(this.bloomPos, color, BURST_SCALE, time);
        pool.ripple(this.choicePos, color, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(0.6);
        const floodLight = world.rig.points[3];
        if (floodLight) floodLight.color.copy(color);
      } else if (ev.type === 'cue') {
        this.kindleAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      } else if (ev.type === 'done') {
        if (!scored) {
          this.douseAt = time;
          this.bloomPos.set(this.choiceX, BLOOM_Y, 0);
          pool.sigh(SIGH_STRENGTH, this.bloomPos, time);
        }
        scored = false;
      }
    }

    if (teleported || Math.abs(pos - wisp.cellX) > 1) {
      wisp.setCell(pos, 0, grid);
      wisp.snap();
    } else if (pos !== wisp.cellX) {
      wisp.hopTo(this.cell, time);
    }
    wisp.update(frame.dtReal, time);

    const kindle = envelope(time - this.kindleAt, KINDLE_SECONDS, 3.4);
    const atCue = pos === CUE_CELL;
    crystal.setCue(cue);
    crystal.setGlow(atCue ? 1.3 + 0.45 * kindle + 0.18 * Math.sin(time * 2.1) : 0);
    crystal.setTime(time);

    const flare = envelope(time - this.flareAt, FLARE_SECONDS, 2.3);
    const douse = envelope(time - this.douseAt, DOUSE_SECONDS, 1.6);
    const asking = pos === CHOICE_CELL ? 0.8 : 0;
    const rest = MOUTH_REST * (1 + asking) * (1 - 0.94 * douse);
    cyan.setGlow(rest + (this.winner === 0 ? MOUTH_FLARE * flare : 0));
    magenta.setGlow(rest + (this.winner === 1 ? MOUTH_FLARE * flare : 0));
    cyan.setTime(time);
    magenta.setTime(time);

    const floodAge = time - this.floodAt;
    const flooding = floodRunning(floodAge);
    const flood = floodProgress(floodAge);
    if (flooding) {
      const eased = 1 - (1 - flood) * (1 - flood);
      world.flood.points.position.x = this.choiceX + (this.cueX - this.choiceX) * eased;
      world.flood.setDensity(1);
      world.flood.setIntensity(2.4 * (1 - flood * flood));
    } else {
      world.flood.points.position.x = this.choiceX;
      world.flood.setDensity(0);
    }

    const wispLight = world.rig.points[0];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += ctx.tokens.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 2.4 + 2.6 * flare;
    }
    const crystalLight = world.rig.points[1];
    if (crystalLight) {
      crystalLight.color.copy(this.cueColors[cue]);
      crystalLight.intensity = atCue ? 4.2 + 2.2 * kindle : 0;
    }
    const choiceLight = world.rig.points[2];
    if (choiceLight) {
      choiceLight.intensity = 2.4 * (1 - 0.94 * douse) + 3.5 * flare;
    }
    const floodLight = world.rig.points[3];
    if (floodLight) {
      floodLight.position.x = world.flood.points.position.x;
      floodLight.intensity = flooding ? 3.4 * (1 - flood * flood) : 0;
    }

    world.setTime(time);

    rig.update(frame.dtReal);
  }

  protected drop(): void {
    this.mouthCyan?.dispose();
    this.mouthCyan = null;
    this.mouthMagenta?.dispose();
    this.mouthMagenta = null;
    this.crystal?.dispose();
    this.crystal = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.kindleAt = -Infinity;
    this.flareAt = -Infinity;
    this.douseAt = -Infinity;
    this.floodAt = -Infinity;
  }
}

export function floodRunning(age: number): boolean {
  return age >= 0 && age < FLOOD_SECONDS;
}

export function floodProgress(age: number): number {
  if (!(age >= 0) || age >= FLOOD_SECONDS) return 0;
  return clamp01(age / FLOOD_SECONDS);
}

export default ((() => new FireflyCaveScene()) satisfies SceneFactory);
