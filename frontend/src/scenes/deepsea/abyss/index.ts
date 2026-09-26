import { PULSE_SECONDS } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL } from '@kit/grid';
import { blendTokens, clamp01, envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { cellToTerraceVec, rowDrop, terraceDepthAt, terraceShear } from '@kit/terrace';
import type { FrameUpdate, SceneContext, SceneSize, ScenePost } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import { DEEPSEA_COL, DEEPSEA_ROW } from '@sim/envs/deepsea';
import { lastEvent } from '@sim/events';
import type { GridCell, GridGeometry } from '@sim/types';
import * as THREE from 'three/webgpu';
import { parameter } from '../../../data/envs';
import { formatRate, formatSigned } from '../../../format';
import { isDiveStep, readDive, reefReachable, resolveGrid, type DiveCell } from '../reef';
import { HEART_Y, buildDiveBubble, buildWreck, type DiveBubbleHandle, type WreckHandle } from './actors';
import { buildWorld, type WorldHandle } from './world';

const BURST_SCALE = 0.8;
const RIPPLE_SCALE = 0.55;
const PULSE_STRENGTH = 0.62;
const TREASURE_SECONDS = 2.2;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;

const WISP_LAMP_LIFT = 0.75;

const RUSH_SECONDS = 1.15;
const RUSH_Y = 1.25;

const SILT_LIFT = 0.28;
const SILT_RATE = 9;
const SILT_IDLE = 0.04;
const SILT_KICK = 0.85;

export class TheAbyssScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.6, fog: { from: ['abyss', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'theAbyss';
  private view: SceneView | null = null;

  private wisp: Wisp | null = null;
  private readonly diver = new THREE.Group();
  private trail: THREE.Object3D | null = null;
  private bubble: DiveBubbleHandle | null = null;
  private wreck: WreckHandle | null = null;
  private grid!: GridGeometry;
  private goal!: GridCell;
  private moveCost = 0;

  private treasureAt = -Infinity;
  private rushAt = -Infinity;
  private stir = 0;

  private readonly dive: DiveCell = { r: 0, c: 0, lastA: 0 };
  private readonly overlayDive: DiveCell = { r: 0, c: 0, lastA: 0 };
  private readonly overlayCell: GridCell = { x: 0, y: 0 };
  private readonly heartPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly bubblePos = new THREE.Vector3();
  private readonly siltPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly rushFrom = new THREE.Vector3();
  private readonly rushTo = new THREE.Vector3();
  private accent = 0;
  private gold = 0;
  private readonly ledger = { row: 0, col: 0, action: 0, moved: '—', reward: 0 };

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    if (!grid.goal) throw new Error('Deep Sea goal is missing');
    this.grid = grid;
    this.goal = grid.goal;
    this.moveCost = parameter(ctx.environment, 'moveCost');
    const half = Math.max(grid.w, grid.h) * CELL * 0.5 + 0.8;
    this.view = makeSceneView(ctx, { min: [-half, -4.8, -half], max: [half, 2, half] }, [-3.8, -1.6, 0.9], { distance: 31, azimuth: -2.52, polar: 0.74 });
    return buildWorld(ctx, this.scene, grid, reefReachable(grid), this.accentOf(ctx));
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const accent = this.accentOf(ctx);
    this.accent = accent;
    this.gold = ctx.tokens.celebrationGold;
    const water = ctx.tokens.palettes.underwater;
    const skin = world.skin;

    const wispLight = world.rig.points[0];
    if (wispLight) wispLight.color.set(accent);
    const heartLight = world.rig.points[1];
    if (heartLight) heartLight.color.set(skin.heart);

    const wisp = new Wisp(
      {
        color: accent,
        radius: 0.19,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'abyssWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.trail = wisp.group.getObjectByName('oakWispRibbon') ?? null;
    if (this.trail) this.trail.matrixAutoUpdate = false;
    this.diver.add(wisp.group);
    this.scene.add(this.diver);

    const bubble = buildDiveBubble({
      accent,
      water: blendTokens(water.groundA, ctx.tokens.palettes.abyss.skyTop, 0.5),
      name: 'diveBubble',
    });
    this.bubble = bubble;
    this.scene.add(bubble.group);

    const wreck = buildWreck({ skin, accent, name: 'sunkenWreck' });
    this.wreck = wreck;
    this.scene.add(wreck.group);

    const ledger = this.ledger;
    this.legend = {
      title: 'THE ABYSS',
      rows: [
        { key: 'depth', label: 'depth', tint: accent, value: () => `row ${ledger.row + 1} of ${grid.h}` },
        { key: 'column', label: 'column', tint: accent, value: () => `${ledger.col + 1} of ${grid.w}` },
        { key: 'action', label: 'action', tint: skin.snow.getHex(), value: () => ledger.action === 1 ? `1 · costs ${formatRate(-this.moveCost)}` : '0 · free' },
        { key: 'moved', label: 'moved', tint: skin.snow.getHex(), value: () => ledger.moved },
        { key: 'wreck', label: 'wreck', tint: skin.heart.getHex(), value: () => `column ${this.goal.x + 1}, row ${this.goal.y + 1}` },
        { key: 'reward', label: 'reward', tint: this.gold, value: () => formatSigned(ledger.reward) },
      ],
    };
  }

  protected rigOptions(_ctx: SceneContext): OrbitRigOptions {
    if (!this.view) throw new Error('Scene view is missing');
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentCell: (state) => {
        readDive(state, this.grid, this.overlayDive);
        this.overlayCell.x = this.overlayDive.c;
        this.overlayCell.y = this.overlayDive.r;
        return this.overlayCell;
      },
    };
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const wisp = this.wisp;
    const wreck = this.wreck;
    const bubble = this.bubble;
    if (!wisp || !wreck || !bubble) return;
    if (this.overlay) {
      terraceShear(grid, this.overlay.group.matrix);
      this.overlay.group.matrixAutoUpdate = false;
      this.overlay.group.matrixWorldNeedsUpdate = true;
    }
    world.setTime(0);
    if (this.view && this.rig) world.setView(this.view.update(this.rig, 0), this.camera);
    wisp.setCell(0, 0, grid);
    cellToTerraceVec(this.goal.x, this.goal.y, grid, wreck.root.position);
    this.groundPos.copy(wreck.root.position);
    this.heartPos.copy(wreck.root.position);
    this.heartPos.y += HEART_Y;
    cellToTerraceVec(0, 0, grid, this.rushTo);
    this.rushTo.y = RUSH_Y;
    this.rushFrom.copy(this.rushTo);
    wreck.setFlare(0);
    wreck.setTime(0);
    bubble.setGain(0);
    bubble.setTime(0);
    cellToTerraceVec(0, 0, grid, world.silt.points.position);
    world.silt.points.position.y += ctx.tokens.wispHeight * SILT_LIFT;
    world.silt.setDensity(0);
  }

  protected tick(frame: FrameUpdate, { ctx, world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const bubble = this.bubble;
    const wreck = this.wreck;
    if (!wisp || !bubble || !wreck) return;

    const state = frame.state;
    const grid = this.grid;
    const dive = readDive(state, grid, this.dive);

    pool.update(frame.dtReal, time);

    const depth = clamp01(dive.r / Math.max(1, grid.h - 2));
    wisp.dim(0.2 + 0.8 * depth);

    const known = state.step > 0 && frame.frameAt(state.step - 1).step === state.step - 1;
    const before = frame.poseAt(state.step - 1);
    const fromRow = before[DEEPSEA_ROW] ?? dive.r;
    const fromCol = before[DEEPSEA_COL] ?? dive.c;
    this.ledger.row = dive.r;
    this.ledger.col = dive.c;
    this.ledger.action = dive.lastA;
    this.ledger.reward = state.reward;
    this.ledger.moved = !known ? '—' : dive.r < fromRow ? 'resurfaced' : dive.c > fromCol ? 'right →' : dive.c < fromCol ? '← left' : '← left (edge)';

    if (frame.scrubbed) {
      pool.clear();
      this.treasureAt = -Infinity;
      this.rushAt = -Infinity;
      this.stir = dive.lastA;
      wisp.setCell(dive.c, dive.r, grid);
      wisp.snap();
      const step = lastEvent(frame.eventLog, 'goal', state.step);
      if (step !== undefined) {
        const at = step / frame.controlHz;
        if (time - at <= TREASURE_SECONDS) {
          this.treasureAt = at;
          pool.burst(this.heartPos, this.accent, BURST_SCALE, at);
          pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, at);
          pool.pulse.reset();
          pool.pulse.fire(Math.max(0, PULSE_STRENGTH - (time - at) / PULSE_SECONDS));
          pool.update(0, time);
        }
      }
    }

    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        pool.burst(this.heartPos, this.accent, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        this.treasureAt = time;
      } else if (ev.type === 'resurface') {
        const at = lastEvent(frame.eventLog, 'resurface', state.step) ?? state.step;
        const from = frame.poseAt(at - 1);
        cellToTerraceVec(from[DEEPSEA_COL] ?? 0, from[DEEPSEA_ROW] ?? 0, grid, this.rushFrom);
        this.rushFrom.y = RUSH_Y;
        const to = frame.poseAt(at);
        cellToTerraceVec(to[DEEPSEA_COL] ?? 0, to[DEEPSEA_ROW] ?? 0, grid, this.rushTo);
        this.rushTo.y = RUSH_Y;
        this.rushAt = time;
      }
    }

    if (dive.c !== wisp.cellX || dive.r !== wisp.cellY) {
      if (isDiveStep(wisp.cellX, wisp.cellY, dive)) {
        wisp.hopTo({ x: dive.c, y: dive.r }, time);
      } else {
        wisp.setCell(dive.c, dive.r, grid);
        wisp.snap();
      }
    }
    wisp.update(frame.dtReal, time);
    this.diver.position.y = terraceDepthAt(wisp.position.z, grid);
    if (this.trail) {
      const slope = rowDrop(grid) / CELL;
      this.trail.matrix.set(1, 0, 0, 0, 0, 1, -slope, slope * wisp.position.z, 0, 0, 1, 0, 0, 0, 0, 1);
      this.trail.matrixWorldNeedsUpdate = true;
    }

    this.bubblePos.copy(wisp.position);
    this.bubblePos.y += ctx.tokens.wispHeight + this.diver.position.y;
    bubble.group.position.copy(this.bubblePos);
    bubble.setGain(depth);
    bubble.setTime(time);

    this.stir += (dive.lastA - this.stir) * (1 - Math.exp(-SILT_RATE * Math.max(0, frame.dtReal)));
    this.siltPos.copy(wisp.position);
    this.siltPos.y += ctx.tokens.wispHeight * SILT_LIFT + this.diver.position.y;
    world.silt.points.position.copy(this.siltPos);
    world.silt.setDensity(SILT_IDLE + SILT_KICK * this.stir);
    world.silt.setIntensity(0.45 + 0.85 * depth);

    const treasure = envelope(time - this.treasureAt, TREASURE_SECONDS, 2.1);
    const rushAge = time - this.rushAt;
    const rushing = rushAge >= 0 && rushAge <= RUSH_SECONDS;
    const rushT = rushing ? rushAge / RUSH_SECONDS : 0;
    const rushE = 1 - Math.pow(1 - rushT, 2.2);
    world.rush.points.position.lerpVectors(this.rushFrom, this.rushTo, rushing ? rushE : 0);
    world.rush.setDensity(rushing ? Math.sin(Math.PI * rushT) : 0);

    wreck.setFlare(treasure);
    wreck.setTime(time);

    const wispLight = world.rig.points[0];
    if (wispLight) {
      this.lightPos.copy(this.bubblePos);
      this.lightPos.y += WISP_LAMP_LIFT;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity =
        0.06 + 3.4 * depth * depth + 5.5 * treasure + 1.2 * (rushing ? Math.sin(Math.PI * rushT) : 0);
    }
    const heartLight = world.rig.points[1];
    if (heartLight) {
      heartLight.position.copy(this.heartPos);
      heartLight.intensity = 1.05 + 0.35 * Math.sin(time * 1.35) + 8 * treasure;
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
    this.wreck?.dispose();
    this.wreck = null;
    this.bubble?.dispose();
    this.bubble = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.trail = null;
    this.diver.removeFromParent();
    this.diver.clear();
    this.treasureAt = -Infinity;
    this.rushAt = -Infinity;
    this.stir = 0;
  }
}
