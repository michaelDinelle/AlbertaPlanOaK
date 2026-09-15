import { PULSE_SECONDS } from '@kit/celebrate';
import { makeSceneView, trackPose, type SceneView } from '@kit/view';
import { DELAYEDCHAIN_ACTION, DELAYEDCHAIN_POS } from '@sim/envs/delayedchain';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorld, cellToWorldVec, type GridSize } from '@kit/grid';
import { clamp01, envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import {
  LAMP_Y,
  buildLanternRoad,
  buildMilestone,
  buildWispGlow,
  type LanternRoadHandle,
  type MilestoneHandle,
  type WispGlowHandle,
} from './actors';
import {
  buildWorld,
  goalCellOf,
  resolveGrid,
  type WorldHandle,
} from './world';


export const WAVE_TRAVEL = 1.35;
export const IGNITE_ATTACK = 0.14;
export const IGNITE_HOLD = 0.85;
export const IGNITE_GUTTER = 2.9;
export const LANTERN_SPAN = IGNITE_ATTACK + IGNITE_HOLD + IGNITE_GUTTER;
export const WAVE_SPAN = WAVE_TRAVEL + LANTERN_SPAN;

const BURST_SCALE = 1.25;
const RIPPLE_SCALE = 0.9;
export const PULSE_STRENGTH = 0.6;
const SURGE_SECONDS = 2.6;
const WISP_FLARE = 0.32;

const WISP_LEAN = 0.15;
const HOP_HEIGHT = 0.15;

function readState(state: FrameState, grid: GridSize, out: GridCell): number {
  const pos = Math.round(state.pose[DELAYEDCHAIN_POS] ?? 0);
  out.x = pos < 0 ? 0 : pos > grid.w - 1 ? grid.w - 1 : pos;
  out.y = 0;
  return (state.pose[DELAYEDCHAIN_ACTION] ?? 0) >= 0.5 ? 1 : 0;
}

class LongNightRoadScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'longNightRoad';
  private view: SceneView | null = null;

  private road: LanternRoadHandle | null = null;
  private milestone: MilestoneHandle | null = null;
  private wisp: Wisp | null = null;
  private glow: WispGlowHandle | null = null;
  private grid!: GridGeometry;
  private goalCell = 0;

  private waveAt = -Infinity;
  private surgeAt = -Infinity;
  private burstAt = -Infinity;
  private waveLit = false;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly overlayCell: GridCell = { x: 0, y: 0 };
  private readonly overlayGoal: GridCell = { x: 0, y: 0 };
  private readonly burstPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private gold = 0xffd166;
  private flame = 0xffd166;

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    const half = Math.abs(cellToWorld(0, 0, grid)[0]) + 2;
    this.view = makeSceneView(ctx, { min: [-half, -0.5, -3], max: [half, 3, 3] }, [0, 1.3, 0], { distance: 8.4, azimuth: -1.16, polar: 1.36 });
    this.goalCell = goalCellOf(grid);
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const palette = ctx.tokens.palettes.night;
    const accent = this.accentOf(ctx);
    this.gold = ctx.tokens.celebrationGold;
    this.flame = world.skin.flame.getHex();

    const wispLight = world.rig.points[0];
    if (wispLight) wispLight.color.set(accent);
    const auroraLight = world.rig.points[1];
    if (auroraLight) {
      auroraLight.color.set(accent);
      auroraLight.position.set(cellToWorld(this.goalCell - 2, 0, grid)[0], 3.4, -1.5);
      auroraLight.intensity = 3.6;
    }
    const waveLight = world.rig.points[2];
    if (waveLight) waveLight.color.set(this.flame);
    const goalLight = world.rig.points[3];
    if (goalLight) {
      goalLight.color.set(this.gold);
      goalLight.position.set(cellToWorld(this.goalCell, 0, grid)[0], LAMP_Y, 0);
    }

    const road = buildLanternRoad({
      palette,
      skin: world.skin,
      grid,
      goalCell: this.goalCell,
      seed: ctx.seed,
      accent,
      name: 'lanternRoad',
    });
    this.road = road;
    this.scene.add(road.group);
    for (let i = 0; i < road.count; i++) road.setGlow(i, 0);
    road.commit();

    const milestone = buildMilestone({
      palette,
      skin: world.skin,
      grid,
      cell: 0,
      name: 'startMilestone',
    });
    this.milestone = milestone;
    this.scene.add(milestone.group);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.165,
        name: 'roadWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);
    const kitHalo = wisp.group.getObjectByName('oakWispHalo');
    if (kitHalo) kitHalo.visible = false;
    wisp.snap();

    const glow = buildWispGlow({
      color: accent,
      radius: 1.05,
      intensity: 0.72,
      name: 'wispGlow',
    });
    this.glow = glow;
    this.scene.add(glow.mesh);
  }

  protected rigOptions(_ctx: SceneContext): OrbitRigOptions {
    if (!this.view) throw new Error('Scene view is missing');
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const grid = this.grid;
    return {
      agentCell: (state) => {
        readState(state, grid, this.overlayCell);
        return this.overlayCell;
      },
      goalCell: () => {
        this.overlayGoal.x = this.goalCell;
        this.overlayGoal.y = 0;
        return this.overlayGoal;
      },
    };
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    const road = this.road;
    const wisp = this.wisp;
    const glow = this.glow;
    const rig = this.rig;
    if (!road || !wisp || !glow || !rig) return;
    world.setTime(0);
    if (this.view && this.rig) world.setView(this.view.update(this.rig, 0, [(this.wisp?.position.x ?? 0) + 0.9, 1.3, 0]), this.camera);
    world.aurora.setGain(1);
    road.setTime(0);
    this.milestone?.setTime(0);
    wisp.setCell(0, 0, this.grid);
    glow.mesh.position.copy(wisp.position);
    glow.mesh.position.y += ctx.tokens.wispHeight;
    glow.mesh.quaternion.copy(this.camera.quaternion);
    glow.setGain(1);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const road = this.road;
    const wisp = this.wisp;
    if (!road || !wisp) return;

    const state = frame.state;
    const grid = this.grid;
    const agent = this.agentCell;

    pool.update(frame.dtReal, time);

    const lastA = readState(state, grid, agent);

    if (frame.scrubbed) {
      pool.clear();
      this.waveAt = -Infinity;
      this.surgeAt = -Infinity;
      this.burstAt = -Infinity;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      const log = frame.eventLog;
      let low = 0;
      let high = log.count;
      while (low < high) {
        const mid = (low + high) >>> 1;
        if ((log.steps[mid] ?? Infinity) <= state.step) low = mid + 1;
        else high = mid;
      }
      const seconds = state.step > 0 ? state.simTime / state.step : 0;
      for (let i = low - 1; i >= 0; i--) {
        const at = ((log.steps[i] ?? -1) + 1) * seconds;
        if (time - at > WAVE_SPAN) break;
        if (log.types[log.kinds[i] ?? 0] === 'goal') {
          this.waveAt = at;
          this.surgeAt = at;
          this.burstAt = at;
          this.burstPos.copy(road.endLamp);
          cellToWorldVec(this.goalCell, 0, grid, this.groundPos);
          pool.burst(this.burstPos, this.flame, BURST_SCALE, at);
          pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, at);
          pool.pulse.reset();
          pool.pulse.fire(Math.max(0, PULSE_STRENGTH - (time - at) / PULSE_SECONDS));
          pool.update(0, time);
          break;
        }
      }
    }

    let paid = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        this.burstPos.copy(road.endLamp);
        cellToWorldVec(this.goalCell, 0, grid, this.groundPos);
        pool.burst(this.burstPos, this.flame, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.reset();
        pool.pulse.fire(PULSE_STRENGTH);
        road.flare(1);
        paid = true;
        this.waveAt = time;
        this.surgeAt = time;
        this.burstAt = time;
      }
    }

    if (paid) wisp.flare(WISP_FLARE);

    wisp.avatar.rotation.z = lastA === 1 ? -WISP_LEAN : WISP_LEAN;
    wisp.update(frame.dtReal, time);
    const step = Math.floor(frame.position);
    const hopping = frame.frameAt(step + 1).pose[DELAYEDCHAIN_POS] !== frame.frameAt(step).pose[DELAYEDCHAIN_POS];
    wisp.place((trackPose(frame, DELAYEDCHAIN_POS) - (grid.w - 1) / 2) * CELL, hopping ? HOP_HEIGHT * Math.sin(Math.PI * (frame.position - step)) : 0);

    const age = time - this.waveAt;
    const active = age >= 0 && age <= WAVE_SPAN;
    if (active || this.waveLit) {
      const count = road.count;
      for (let i = 0; i < count; i++) {
        const g = active ? lanternGlow(age, i, count) : 0;
        road.setGlow(i, g > 0 ? g * flicker(time, i) : 0);
      }
      road.commit();
      this.waveLit = active;
    }

    const burst = envelope(time - this.burstAt, 1.6, 2.2);
    const surge = envelope(time - this.surgeAt, SURGE_SECONDS, 1.5);

    const glow = this.glow;
    if (glow) {
      glow.mesh.position.copy(wisp.position);
      glow.mesh.position.y += TOKENS.wispHeight;
      glow.mesh.quaternion.copy(this.camera.quaternion);
      glow.setGain(1 + 1.1 * burst);
    }

    const wispLight = world.rig.points[0];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.62 + 1.4 * burst;
    }
    const waveLight = world.rig.points[2];
    if (waveLight) {
      const front = active ? 1 - clamp01(age / WAVE_TRAVEL) : 0;
      waveLight.position.set(road.lanternX(front * (road.count - 1)), 1.15, -0.2);
      waveLight.intensity = active ? 2.6 * envelope(age, WAVE_TRAVEL + 0.45, 1.1) : 0;
    }
    const goalLight = world.rig.points[3];
    if (goalLight) goalLight.intensity = 3.2 * burst;

    world.aurora.setGain(1 + 1.35 * surge);
    world.setTime(time);
    road.setTime(time);
    road.update(frame.dtReal, time);
    this.milestone?.setTime(time);

    if (this.view) world.setView(this.view.update(rig, frame.dtReal, [wisp.position.x + 0.9, 1.3, 0]), this.camera);
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.view && this.rig && this.world) {
      this.view.resize(size, this.rig);
      this.world.setView(this.view.update(this.rig, 0, [(this.wisp?.position.x ?? 0) + 0.9, 1.3, 0]), this.camera);
    }
  }

  protected drop(): void {
    this.milestone?.dispose();
    this.milestone = null;
    this.road?.dispose();
    this.road = null;
    this.glow?.dispose();
    this.glow = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.waveAt = -Infinity;
    this.surgeAt = -Infinity;
    this.burstAt = -Infinity;
    this.waveLit = false;
  }


}

export function lanternGlow(age: number, index: number, count: number): number {
  if (!(age >= 0)) return 0;
  const n = Math.max(1, count - 1);
  const i = index < 0 ? 0 : index > n ? n : index;
  const back = (n - i) / n;
  const local = age - back * WAVE_TRAVEL;
  if (local < 0 || local > LANTERN_SPAN) return 0;
  if (local < IGNITE_ATTACK) {
    const t = local / IGNITE_ATTACK;
    return t * t * (3 - 2 * t);
  }
  const fall = local - IGNITE_ATTACK;
  if (fall < IGNITE_HOLD) return 1;
  const g = (fall - IGNITE_HOLD) / IGNITE_GUTTER;
  const e = Math.exp(-2.1 * g) * (1 - g);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

export function flicker(time: number, index: number): number {
  return 0.87 + 0.13 * Math.sin(time * 9.3 + index * 2.3994);
}

export default ((() => new LongNightRoadScene()) satisfies SceneFactory);
