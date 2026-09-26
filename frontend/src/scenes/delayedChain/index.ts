import * as THREE from 'three/webgpu';
import { PULSE_SECONDS } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorld, cellToWorldVec } from '@kit/grid';
import { clamp01, envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost } from '@kit/types';
import { makeSceneView, trackPose, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import type { WorldCore } from '@kit/world';
import { lastEvent } from '@sim/events';
import { CHAIN_POS } from '@sim/envs/walk';
import type { GridGeometry } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { LAMP_Y, buildLanternRoad, buildMilestone, buildWispGlow, type LanternRoad, type WispGlow } from './actors';
import { buildWorld, type WorldHandle } from './world';

const LAST_MOVE_WORD = 1;

const WAVE_TRAVEL = 1.35;
const IGNITE_ATTACK = 0.14;
const IGNITE_HOLD = 0.85;
const IGNITE_GUTTER = 2.9;
const LANTERN_SPAN = IGNITE_ATTACK + IGNITE_HOLD + IGNITE_GUTTER;
const WAVE_SPAN = WAVE_TRAVEL + LANTERN_SPAN;

const BURST_SCALE = 1.25;
const RIPPLE_SCALE = 0.9;
const PULSE_STRENGTH = 0.6;
const SURGE_SECONDS = 2.6;
const WISP_FLARE = 0.32;
const WISP_LEAN = 0.15;
const HOP_HEIGHT = 0.15;

class LongNightRoadScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['night', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'longNightRoad';
  private view!: SceneView;
  private grid!: GridGeometry;
  private goal = 0;
  private road: LanternRoad | null = null;
  private milestone: WorldCore | null = null;
  private wisp: Wisp | null = null;
  private glow: WispGlow | null = null;
  private latestAt = -Infinity;
  private earlierAt = -Infinity;
  private waveLit = false;
  private readonly goalGround = new THREE.Vector3();
  private readonly overlayCell = { x: 0, y: 0 };
  private readonly ledger = { pos: 0, forward: false, reward: 0, step: 0, litAt: undefined as number | undefined };
  override legend: LegendSpec = {
    title: 'LONG NIGHT ROAD',
    rows: [
      { key: 'at', label: 'wisp', tint: TOKENS.accents.DelayedChain!, value: () => `stone ${this.ledger.pos + 1} of ${this.grid.w}` },
      { key: 'move', label: 'last move', tint: 0xdcf4ff, value: () => (this.ledger.forward ? 'forward →' : '← back') },
      { key: 'lit', label: 'lanterns lit', tint: 0xffd166, value: () => (this.ledger.litAt === undefined ? 'not yet' : `${formatCount(this.ledger.step - this.ledger.litAt)} steps ago`) },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
    note: () => {
      const left = this.goal - this.ledger.pos;
      return `${formatCount(left)} step${left === 1 ? '' : 's'} to the end lantern`;
    },
    bar: () => this.ledger.pos / this.goal,
  };

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = ctx.geometry.grid;
    if (!grid?.goal || grid.w < 2 || grid.h !== 1) throw new Error('Delayed Chain grid is missing or invalid');
    this.grid = grid;
    this.goal = grid.goal.x;
    const half = Math.abs(cellToWorld(0, 0, grid)[0]) + 2;
    this.view = makeSceneView(ctx, { min: [-half, -0.5, -3], max: [half, 3, 3] }, [0, 1.3, 0], { distance: 8.4, azimuth: -1.16, polar: 1.36 });
    cellToWorldVec(this.goal, 0, grid, this.goalGround);
    return buildWorld(ctx, this.scene, grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const palette = ctx.tokens.palettes.night, accent = this.accentOf(ctx), grid = this.grid;
    const road = buildLanternRoad(palette, world.skin, grid, this.goal, ctx.seed, accent);
    this.road = road;
    this.milestone = buildMilestone(palette, world.skin, grid);
    const wisp = new Wisp({ color: accent, radius: 0.165, name: 'roadWisp' }, grid);
    wisp.group.getObjectByName('oakWispHalo')!.visible = false;
    this.wisp = wisp;
    this.glow = buildWispGlow(accent);
    this.scene.add(road.group, this.milestone.group, wisp.group, this.glow.group);

    const [wispLight, auroraLight, waveLight, goalLight] = world.rig.points;
    wispLight!.color.set(accent);
    auroraLight!.color.set(accent);
    auroraLight!.position.set(road.lanternX(this.goal - 2), 3.4, -1.5);
    auroraLight!.intensity = 3.6;
    waveLight!.color.set(world.skin.flame);
    goalLight!.color.set(ctx.tokens.celebrationGold);
    goalLight!.position.set(road.lanternX(this.goal), LAMP_Y, 0);
  }

  protected rigOptions(): OrbitRigOptions {
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentCell: (state) => {
        this.overlayCell.x = Math.round(state.pose[CHAIN_POS]!);
        return this.overlayCell;
      },
      debug: (frame, kit) => {
        const x = this.road!.lanternX(Math.round(frame.state.pose[CHAIN_POS]!)), forward = frame.frameAt(frame.state.step).words[LAST_MOVE_WORD] === 1;
        kit.marks('lastMove', 1).set(x + (forward ? 0.36 : -0.36), 0.06, 0, 0.08, TOKENS.uiCyan);
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    world.setView(this.view.update(this.rig!, 0, [this.road!.lanternX(0) + 0.9, 1.3, 0]), this.camera);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const road = this.road!, wisp = this.wisp!, glow = this.glow!, state = frame.state, ledger = this.ledger;
    ledger.pos = Math.round(state.pose[CHAIN_POS]!);
    ledger.forward = frame.frameAt(state.step).words[LAST_MOVE_WORD] === 1;
    ledger.reward = state.reward;
    ledger.step = state.step;
    ledger.litAt = lastEvent(frame.eventLog, 'goal', state.step);

    pool.update(frame.dtReal, time);
    let litAt: number | null = null;
    if (frame.scrubbed) {
      pool.clear();
      wisp.setCell(ledger.pos, 0, this.grid);
      wisp.snap();
      const earlier = ledger.litAt === undefined ? undefined : lastEvent(frame.eventLog, 'goal', ledger.litAt - 1);
      this.latestAt = ledger.litAt === undefined ? -Infinity : ledger.litAt / frame.controlHz;
      this.earlierAt = earlier === undefined ? -Infinity : earlier / frame.controlHz;
      if (time - this.latestAt <= WAVE_SPAN) litAt = this.latestAt;
    }
    for (const event of frame.events) {
      if (event.type !== 'goal') continue;
      this.earlierAt = this.latestAt;
      this.latestAt = litAt = time;
      road.flare(1);
      wisp.flare(WISP_FLARE);
    }
    if (litAt !== null) {
      pool.burst(road.endLamp, world.skin.flame, BURST_SCALE, litAt);
      pool.ripple(this.goalGround, ctx.tokens.celebrationGold, RIPPLE_SCALE, litAt);
      pool.pulse.reset();
      pool.pulse.fire(Math.max(0, PULSE_STRENGTH - (time - litAt) / PULSE_SECONDS));
      pool.update(0, time);
    }

    wisp.avatar.rotation.z = ledger.forward ? -WISP_LEAN : WISP_LEAN;
    wisp.update(frame.dtReal, time);
    const at = Math.floor(frame.position), from = frame.poseAt(at)[CHAIN_POS];
    wisp.place(road.lanternX(trackPose(frame, CHAIN_POS)), frame.poseAt(at + 1)[CHAIN_POS] !== from ? HOP_HEIGHT * Math.sin(Math.PI * (frame.position - at)) : 0);

    const age = time - this.latestAt, earlier = time - this.earlierAt, active = age <= WAVE_SPAN || earlier <= WAVE_SPAN;
    if (active || this.waveLit) {
      for (let i = 0; i < road.count; i++) road.setGlow(i, Math.max(lanternGlow(age, i, road.count), lanternGlow(earlier, i, road.count)) * flicker(time, i));
      road.commit();
      this.waveLit = active;
    }
    const burst = envelope(age, 1.6, 2.2);

    world.setView(this.view.update(rig, frame.dtReal, [wisp.position.x + 0.9, 1.3, 0]), this.camera);
    glow.mesh.position.copy(wisp.position);
    glow.mesh.position.y += ctx.tokens.wispHeight;
    glow.mesh.quaternion.copy(this.camera.quaternion);
    glow.setGain(1 + 1.1 * burst);

    const [wispLight, , waveLight, goalLight] = world.rig.points;
    wispLight!.position.copy(glow.mesh.position);
    wispLight!.intensity = 0.62 + 1.4 * burst;
    waveLight!.position.set(road.lanternX((1 - clamp01(age / WAVE_TRAVEL)) * (road.count - 1)), 1.15, -0.2);
    waveLight!.intensity = 2.6 * envelope(age, WAVE_TRAVEL + 0.45, 1.1);
    goalLight!.intensity = 3.2 * burst;

    world.aurora.setGain(1 + 1.35 * envelope(age, SURGE_SECONDS, 1.5));
    world.setTime(time);
    road.update(frame.dtReal, time);
  }

  protected drop(): void {
    this.road?.dispose();
    this.milestone?.dispose();
    this.glow?.dispose();
    this.wisp?.dispose();
    this.road = null;
    this.milestone = null;
    this.glow = null;
    this.wisp = null;
    this.latestAt = -Infinity;
    this.earlierAt = -Infinity;
    this.waveLit = false;
  }
}

function lanternGlow(age: number, index: number, count: number): number {
  const n = Math.max(1, count - 1);
  const local = age - ((n - index) / n) * WAVE_TRAVEL;
  if (local < 0 || local > LANTERN_SPAN) return 0;
  if (local < IGNITE_ATTACK) {
    const t = local / IGNITE_ATTACK;
    return t * t * (3 - 2 * t);
  }
  const fall = local - IGNITE_ATTACK;
  if (fall < IGNITE_HOLD) return 1;
  const g = (fall - IGNITE_HOLD) / IGNITE_GUTTER;
  return clamp01(Math.exp(-2.1 * g) * (1 - g));
}

function flicker(time: number, index: number): number {
  return 0.87 + 0.13 * Math.sin(time * 9.3 + index * 2.3994);
}

export default ((() => new LongNightRoadScene()) satisfies SceneFactory);
