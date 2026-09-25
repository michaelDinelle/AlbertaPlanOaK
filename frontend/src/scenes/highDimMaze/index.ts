import * as THREE from 'three/webgpu';
import { PULSE_SECONDS, type CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec } from '@kit/grid';
import { blendTokens, clamp01, envelope } from '@kit/mix';
import { makeAmbientParticles, type AmbientParticlesHandle } from '@kit/particles';
import { damp, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameUpdate, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp, yawFromGridDelta } from '@kit/wisp';
import { lastEvent } from '@sim/events';
import { MAZE_GX, MAZE_GY, MAZE_HEADING, MAZE_X, MAZE_Y } from '@sim/envs/maze';
import type { GridCell, GridGeometry } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { buildGust, buildOasis, buildScarf, type GustHandle, type OasisHandle, type ScarfHandle } from './actors';
import { buildWorld, canyonSkin, resolveGrid, type WorldHandle } from './world';

const BURST_SCALE = 0.84;
const RIPPLE_SCALE = 0.52;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.34;
const BLOOM_SECONDS = 1.5;
const GROW_SECONDS = 0.8;
const GROW_BOUNCE = 0.65;
const WAKE_SECONDS = 0.62;
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;
const TURN_RATE = 12;
const SETTLE_RATE = 6;
const PROXIMITY_CELLS = 4;
const TRAIL = 24;
const STAY = 4;
const DASH = 5;
const FOUND_WORD = 5;

const HEADINGS = [{ dx: 0, dy: -1 }, { dx: 0, dy: 1 }, { dx: -1, dy: 0 }, { dx: 1, dy: 0 }] as const;
const HEADING_NAMES = ['↑ north', '↓ south', '← west', 'east →'] as const;

interface Reading { x: number; y: number; gx: number; gy: number; heading: number }

function read(pose: Float32Array, out: Reading): Reading {
  out.x = pose[MAZE_X]!;
  out.y = pose[MAZE_Y]!;
  out.gx = pose[MAZE_GX]!;
  out.gy = pose[MAZE_GY]!;
  out.heading = pose[MAZE_HEADING]! & 3;
  return out;
}

const headingYaw = (heading: number): number => yawFromGridDelta(HEADINGS[heading]!.dx, HEADINGS[heading]!.dy);

class SlotCanyonsScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.42, fog: { from: ['afternoon', 'fog'], blend: [[['dusk', 'fog'], 0.3]] } };
  readonly controlHz = 8;
  protected override poolOptions() { return { burstSize: 110 }; }
  protected readonly stageName = 'slotCanyons';

  private view: SceneView | null = null;
  private wisp: Wisp | null = null;
  private scarf: ScarfHandle | null = null;
  private gust: GustHandle | null = null;
  private oasis: OasisHandle | null = null;
  private settleDust: AmbientParticlesHandle | null = null;
  private travelerMaterial: THREE.MeshStandardNodeMaterial | null = null;
  private grid!: GridGeometry;
  private actions = 1;

  private growAt = -Infinity;
  private bloomAt = -Infinity;
  private dashAt = -Infinity;
  private dashStep = -1;
  private seenStep = -1;
  private yaw = 0;
  private slack = 1;
  private spray = 0x5ee0c8;
  private gold = 0xffd166;

  private readonly now: Reading = { x: 0, y: 0, gx: 0, gy: 0, heading: 1 };
  private readonly ledger = { action: -1, reward: 0, found: 0 };
  private readonly from: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly dashFrom = new THREE.Vector3();
  private readonly dashTo = new THREE.Vector3();
  private readonly mark = new THREE.Vector3();

  protected buildWorld(ctx: SceneContext): WorldHandle {
    this.grid = resolveGrid(ctx.geometry);
    this.actions = ctx.environment.actions;
    return buildWorld(ctx, this.scene, this.grid);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const palette = ctx.tokens.palettes.afternoon;
    const skin = canyonSkin(palette, ctx.tokens);
    const accent = this.accentOf(ctx);
    const scarfTint = blendTokens(accent, TOKENS.palettes.dusk.skyHorizon, 0.3);
    this.spray = blendTokens(skin.water, skin.waterDeep, 0.4).getHex();
    this.gold = ctx.tokens.celebrationGold;

    world.rig.points[0]?.color.set(skin.shaft);
    world.rig.points[1]?.color.set(accent);
    world.rig.points[2]?.color.set(this.spray);

    const wisp = new Wisp({
      color: skin.sandSun,
      radius: 0.165,
      hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
      name: 'sandTraveler',
      skirt: false,
      ribbon: false,
      silhouette: false,
    }, this.grid);
    this.travelerMaterial = new THREE.MeshStandardNodeMaterial({ color: skin.sandSun, roughness: 0.84 });
    wisp.core.material = this.travelerMaterial;
    const halo = wisp.group.getObjectByName('oakWispHalo');
    if (halo) halo.visible = false;
    this.wisp = wisp;
    this.scene.add(wisp.group);

    this.scarf = buildScarf(scarfTint, ctx.tokens.wispHeight * 0.92);
    wisp.avatar.add(this.scarf.group);

    this.gust = buildGust(skin, blendTokens(accent, palette.sun, 0.5));
    this.scene.add(this.gust.group);

    this.settleDust = makeAmbientParticles({
      preset: 'dust',
      count: 110,
      bounds: { x: 0, y: 0, z: 0, halfX: 0.4, halfY: 0.28, halfZ: 0.4 },
      seed: ctx.seed ^ 0x5e77,
      color: skin.dust,
      size: 0.9,
      intensity: 1.4,
      opacity: 0.55,
      rise: -0.2,
      swirl: 0.06,
      name: 'canyonSettleDust',
    });
    this.scene.add(this.settleDust.points);

    this.oasis = buildOasis(palette, skin);
    this.scene.add(this.oasis.group);

    this.legend = {
      title: 'CANYONS',
      rows: [
        { key: 'at', label: 'traveler', tint: scarfTint.getHex(), value: () => `column ${this.now.x} · row ${this.now.y}` },
        { key: 'goal', label: 'oasis', tint: skin.water.getHex(), value: () => `column ${this.now.gx} · row ${this.now.gy}` },
        { key: 'facing', label: 'facing', tint: 0xdcf4ff, value: () => HEADING_NAMES[this.now.heading]! },
        { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.actionName() },
        { key: 'reward', label: 'reward', tint: this.gold, value: () => formatSigned(this.ledger.reward) },
      ],
      note: () => `${formatCount(this.ledger.found)} ${this.ledger.found === 1 ? 'oasis' : 'oases'} reached`,
    };
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    const halfX = this.grid.w * CELL / 2;
    const halfZ = this.grid.h * CELL / 2;
    this.view = makeSceneView(ctx, { min: [-halfX, 0, -halfZ], max: [halfX, 1.8, halfZ] }, [0, 0.62, 0], { distance: 17, azimuth: 0.25, polar: 0.86 });
    return this.view.options;
  }

  protected overlayOptions(ctx: SceneContext): GroundTruthOptions {
    const reading: Reading = { x: 0, y: 0, gx: 0, gy: 0, heading: 1 };
    const agent: GridCell = { x: 0, y: 0 };
    const goal: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        read(state.pose, reading);
        agent.x = reading.x;
        agent.y = reading.y;
        return agent;
      },
      goalCell: (state) => {
        read(state.pose, reading);
        goal.x = reading.gx;
        goal.y = reading.gy;
        return goal;
      },
      debug: (frame, kit) => {
        const step = frame.state.step;
        read(frame.state.pose, reading);
        const path = kit.curve('path', TRAIL, ctx.tokens.uiCyan);
        let first = step;
        while (first > 0 && step - first < TRAIL - 1 && !frame.frameAt(first).reset) first--;
        for (let k = first; k <= step; k++) {
          const pose = frame.poseAt(k);
          cellToWorldVec(pose[MAZE_X]!, pose[MAZE_Y]!, this.grid, this.mark);
          path.point(this.mark.x, 0.06, this.mark.z);
        }
        const d = HEADINGS[reading.heading]!;
        cellToWorldVec(reading.x, reading.y, this.grid, this.mark);
        const tipX = this.mark.x + d.dx * 0.46, tipZ = this.mark.z + d.dy * 0.46;
        const arrow = kit.curve('heading', 5, ctx.tokens.celebrationGold);
        arrow.point(this.mark.x, 0.08, this.mark.z);
        arrow.point(tipX, 0.08, tipZ);
        arrow.point(tipX - d.dx * 0.16 + d.dy * 0.12, 0.08, tipZ - d.dy * 0.16 - d.dx * 0.12);
        arrow.point(tipX, 0.08, tipZ);
        arrow.point(tipX - d.dx * 0.16 - d.dy * 0.12, 0.08, tipZ - d.dy * 0.16 + d.dx * 0.12);
      },
    };
  }

  protected start(): void {}

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp!, scarf = this.scarf!, gust = this.gust!, oasis = this.oasis!, settleDust = this.settleDust!;
    const grid = this.grid, state = frame.state, step = state.step, now = read(state.pose, this.now);
    const action = this.actionAt(frame, step);
    this.ledger.action = action;
    this.ledger.reward = state.reward;
    this.ledger.found = frame.frameAt(step).words[FOUND_WORD]!;

    pool.update(frame.dtReal, time);

    const still = step === 0 || state.reset || !this.movedAt(frame, step);
    const target = headingYaw(now.heading);
    wisp.face(target);

    if (frame.scrubbed) {
      pool.clear();
      gust.clear();
      this.bloomAt = -Infinity;
      this.growAt = -Infinity;
      this.dashAt = -Infinity;
      this.dashStep = -1;
      const goalStep = lastEvent(frame.eventLog, 'goal', step);
      if (goalStep !== undefined && time - goalStep / frame.controlHz <= BLOOM_SECONDS) {
        this.celebrate(frame, goalStep, pool);
        pool.pulse.reset();
        pool.pulse.fire(Math.max(0, PULSE_STRENGTH - (time - this.bloomAt) / PULSE_SECONDS));
        pool.update(0, time);
      }
      for (let k = step; k > 0 && step - k < Math.ceil(WAKE_SECONDS * frame.controlHz) && !frame.frameAt(k).reset; k--) {
        if (this.actionAt(frame, k) !== DASH || !this.movedAt(frame, k)) continue;
        this.dash(frame, k);
        break;
      }
      wisp.setCell(now.x, now.y, grid);
      wisp.snap();
      this.yaw = target;
      this.slack = still ? 1 : 0;
    } else {
      if (frame.events.some((event) => event.type === 'goal')) {
        this.celebrate(frame, lastEvent(frame.eventLog, 'goal', step) ?? step, pool);
        pool.pulse.fire(PULSE_STRENGTH);
        oasis.flare(0.8);
      }
      if (step !== this.seenStep) {
        if (frame.events.some((event) => event.type === 'teleport')) {
          wisp.setCell(now.x, now.y, grid);
          wisp.snap();
          gust.clear();
          this.yaw = target;
          this.dashAt = -Infinity;
          this.dashStep = -1;
        } else {
          if (action === DASH && !still) this.dash(frame, step);
          if (now.x !== wisp.cellX || now.y !== wisp.cellY) wisp.hopTo(now, time);
        }
      }
    }
    this.seenStep = step;

    wisp.update(frame.dtReal, time);
    wisp.core.scale.set(0.82, 1.2, 0.82);

    let turn = target - this.yaw;
    turn -= 2 * Math.PI * Math.round(turn / (2 * Math.PI));
    this.yaw += turn * damp(TURN_RATE, frame.dtReal);
    this.slack += ((still ? 1 : 0) - this.slack) * damp(SETTLE_RATE, frame.dtReal);

    const airborne = wisp.motion.airborne ? 1 - wisp.motion.progress : 0;
    const dashing = this.dashStep === step && airborne > 0 ? 1 : 0;
    scarf.setState(this.yaw, this.slack, clamp01(airborne * (0.55 + 0.45 * dashing) + 0.35 * dashing), time);
    gust.setLines(this.dashFrom, this.dashTo, ctx.tokens.wispHeight * 0.8, wisp.motion.progress, dashing * clamp01(0.35 + 0.65 * airborne));
    gust.setWake(this.dashFrom, this.dashTo, envelope(time - this.dashAt, WAKE_SECONDS, 2.6));
    gust.setSettle(wisp.position.x, wisp.position.z, this.slack, time);

    settleDust.points.position.set(wisp.position.x, ctx.tokens.wispHeight * 0.7, wisp.position.z);
    settleDust.setDensity(this.slack);
    settleDust.setTime(time);

    cellToWorldVec(now.gx, now.gy, grid, oasis.root.position);
    const grow = growth(time - this.growAt);
    const near = nearness(now, PROXIMITY_CELLS);
    oasis.setState(grow, near);
    oasis.setTime(time);
    oasis.update(frame.dtReal, time);

    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const oasisLight = world.rig.points[0];
    if (oasisLight) {
      oasisLight.position.set(oasis.root.position.x, oasis.crownHeight, oasis.root.position.z);
      oasisLight.intensity = (0.2 + 0.5 * near) * grow;
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      wispLight.position.copy(this.lightPos.copy(wisp.position).setY(wisp.position.y + ctx.tokens.wispHeight));
      wispLight.intensity = 0.18 + 0.9 * bloom;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 2.4 * bloom;
    }

    world.setTime(time);
    if (this.view) world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  private actionAt(frame: FrameUpdate, step: number): number {
    return step > 0 ? Math.trunc(frame.frameAt(step).actions[0]!) % this.actions : -1;
  }

  private movedAt(frame: FrameUpdate, step: number): boolean {
    const before = frame.poseAt(step - 1);
    this.from.x = before[MAZE_X]!;
    this.from.y = before[MAZE_Y]!;
    const after = frame.poseAt(step);
    return after[MAZE_X] !== this.from.x || after[MAZE_Y] !== this.from.y;
  }

  private dash(frame: FrameUpdate, step: number): void {
    const before = frame.poseAt(step - 1);
    cellToWorldVec(before[MAZE_X]!, before[MAZE_Y]!, this.grid, this.dashFrom);
    const after = frame.poseAt(step);
    cellToWorldVec(after[MAZE_X]!, after[MAZE_Y]!, this.grid, this.dashTo);
    this.dashAt = step / frame.controlHz;
    this.dashStep = step;
  }

  private celebrate(frame: FrameUpdate, goalStep: number, pool: CelebrationPool): void {
    const at = goalStep / frame.controlHz;
    const before = frame.poseAt(goalStep - 1);
    cellToWorldVec(before[MAZE_GX]!, before[MAZE_GY]!, this.grid, this.groundPos);
    this.bloomPos.copy(this.groundPos).setY(BLOOM_Y);
    pool.burst(this.bloomPos, this.spray, BURST_SCALE, at);
    pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, at);
    this.bloomAt = at;
    this.growAt = at;
  }

  private actionName(): string {
    const action = this.ledger.action;
    if (action < 0) return '—';
    if (action < STAY) return HEADING_NAMES[action]!;
    return action === STAY ? 'stay' : `dash ${HEADING_NAMES[this.now.heading]}`;
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.rig && this.view) this.view.resize(size, this.rig);
  }

  protected drop(): void {
    this.travelerMaterial?.dispose();
    this.travelerMaterial = null;
    this.oasis?.dispose();
    this.oasis = null;
    this.settleDust?.dispose();
    this.settleDust = null;
    this.gust?.dispose();
    this.gust = null;
    this.scarf?.dispose();
    this.scarf = null;
    this.wisp?.dispose();
    this.wisp = null;
  }
}

function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  return 1 - Math.pow(1 - t, 3) + GROW_BOUNCE * t * t * (1 - t);
}

function nearness(at: Reading, cells: number): number {
  const d = Math.max(Math.abs(at.x - at.gx), Math.abs(at.y - at.gy));
  return d >= cells ? 0 : 1 - d / cells;
}

export default ((() => new SlotCanyonsScene()) satisfies SceneFactory);
