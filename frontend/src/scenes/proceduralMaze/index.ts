import * as THREE from 'three/webgpu';
import { PULSE_SECONDS, type CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorldVec } from '@kit/grid';
import { blendTokens, envelope } from '@kit/mix';
import { damp, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, trackPose, type SceneView, type ViewAnchor } from '@kit/view';
import { Wisp } from '@kit/wisp';
import { lastEvent } from '@sim/events';
import { MAZE_GX, MAZE_GY, MAZE_X, MAZE_Y } from '@sim/envs/maze';
import type { GridCell, GridGeometry } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { buildSunShaft, type SunShaftHandle } from './actors';
import { FRAMED_SIDE, SUN_DIR, buildWorld, floorTintFor, resolveGrid, ruinSkin, spanOf, type RuinSkin, type WorldHandle } from './world';

const ACTIONS = ['↑ north', '↓ south', '← west', 'east →'] as const;
const CLOCK_WORD = 4;
const ARRIVALS_WORD = 5;

const BURST_SCALE = 0.82;
const RIPPLE_SCALE = 0.48;
const BLOOM_Y = 0.46;
const BLOOM_SECONDS = 1.5;
const GROW_SECONDS = 0.8;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.15;

const DISTANCE = 19;
const AZIMUTH = 0.62;
const POLAR = 0.73;
const LEAD = 1.2;
const FOLLOW_RATE = 6;

function readCells(state: FrameState, agent: GridCell, goal: GridCell): void {
  agent.x = Math.round(state.pose[MAZE_X]!);
  agent.y = Math.round(state.pose[MAZE_Y]!);
  goal.x = Math.round(state.pose[MAZE_GX]!);
  goal.y = Math.round(state.pose[MAZE_GY]!);
}

class OvergrownRuinsScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.45, fog: { from: ['noon', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'overgrownRuins';

  private sceneView: SceneView | null = null;
  private follow = false;
  private lead = LEAD;
  private readonly focus = new THREE.Vector3();
  private readonly aim = new THREE.Vector3();
  private wisp: Wisp | null = null;
  private shaft: SunShaftHandle | null = null;
  private grid!: GridGeometry;
  private skin!: RuinSkin;

  private growAt = -Infinity;
  private bloomAt = -Infinity;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private readonly litTint = new THREE.Color();
  private readonly restoreTint = new THREE.Color();
  private litX = -1;
  private litY = -1;
  private gold = TOKENS.celebrationGold;
  private moteColor = TOKENS.celebrationGold;
  private readonly ledger = { action: 0, reward: 0, clock: 0, arrivals: 0 };

  protected buildWorld(ctx: SceneContext): WorldHandle {
    this.grid = resolveGrid(ctx.geometry);
    this.skin = ruinSkin(ctx.tokens.palettes.noon, ctx.tokens, this.accentOf(ctx));
    return buildWorld(ctx, this.scene, this.grid, this.skin);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    const skin = this.skin;
    const accent = this.accentOf(ctx);
    this.gold = ctx.tokens.celebrationGold;
    this.moteColor = blendTokens(ctx.tokens.celebrationGold, ctx.tokens.palettes.dusk.groundA, 0.3).getHex();
    this.litTint.copy(blendTokens(skin.flag, skin.sunWarm, 0.5));

    world.rig.points[0]?.color.set(skin.sunWarm);
    world.rig.points[1]?.color.set(accent);
    world.rig.points[2]?.color.set(this.moteColor);

    const wisp = new Wisp({ color: accent, radius: 0.22, hop: { duration: HOP_SECONDS, height: HOP_HEIGHT }, name: 'ruinWisp' }, grid);
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const shaft = buildSunShaft({ palette: ctx.tokens.palettes.noon, skin, sunDir: SUN_DIR, name: 'sunShaft' });
    this.shaft = shaft;
    this.scene.add(shaft.group);

    const agent = this.agentCell, goal = this.goalCell, ledger = this.ledger;
    this.legend = {
      title: 'RUINS',
      rows: [
        { key: 'at', label: 'wisp', tint: accent, value: () => `column ${agent.x} · row ${agent.y}` },
        { key: 'goal', label: 'sun shaft', tint: skin.sunWarm.getHex(), value: () => `column ${goal.x} · row ${goal.y}` },
        { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => ACTIONS[ledger.action]! },
        { key: 'reward', label: 'reward', tint: this.gold, value: () => formatSigned(ledger.reward) },
        { key: 'reached', label: 'shafts reached', tint: this.gold, value: () => formatCount(ledger.arrivals) },
      ],
      note: () => `${formatCount(ledger.clock)} ${ledger.clock === 1 ? 'step' : 'steps'} into this search`,
    };
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    const grid = this.grid;
    const span = spanOf(grid);
    const reach = Math.sqrt(span);
    const halfX = grid.w * CELL / 2;
    const halfZ = grid.h * CELL / 2;
    this.follow = span > 1;
    this.lead = LEAD * reach;
    this.sceneView = makeSceneView(
      ctx,
      { min: [-halfX, 0, -halfZ], max: [halfX, 1.8, halfZ] },
      [Math.sin(AZIMUTH) * this.lead, 0.15, Math.cos(AZIMUTH) * this.lead],
      { distance: DISTANCE * reach, azimuth: AZIMUTH, polar: POLAR },
    );
    return this.sceneView.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 };
    const goal: GridCell = { x: 0, y: 0 };
    return {
      agentCell: (state) => {
        readCells(state, agent, goal);
        return agent;
      },
      goalCell: (state) => {
        readCells(state, agent, goal);
        return goal;
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    this.shaft?.setState(1);
    this.shaft?.setTime(0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp;
    const shaft = this.shaft;
    if (!wisp || !shaft) return;
    const grid = this.grid;
    const state = frame.state;
    const agent = this.agentCell;
    const goal = this.goalCell;

    pool.update(frame.dtReal, time);
    readCells(state, agent, goal);
    const now = frame.frameAt(state.step);
    this.ledger.action = Math.trunc(now.actions[0]!) & 3;
    this.ledger.reward = state.reward;
    this.ledger.clock = now.words[CLOCK_WORD]!;
    this.ledger.arrivals = now.words[ARRIVALS_WORD]!;

    let teleported = state.reset;
    if (frame.scrubbed) {
      pool.clear();
      this.bloomAt = -Infinity;
      this.growAt = -Infinity;
      const last = lastEvent(frame.eventLog, 'goal', state.step);
      if (last !== undefined) this.arrive(frame, last, last / frame.controlHz, time, pool);
      teleported = true;
    } else {
      let arrived = false;
      for (const event of frame.events) {
        if (event.type === 'teleport') teleported = true;
        else if (event.type === 'goal') arrived = true;
      }
      if (arrived) {
        for (let i = frame.prev!.step + 1; i <= state.step; i++) if (frame.frameAt(i).reset) this.arrive(frame, i, time, time, pool);
        wisp.flare(1);
        shaft.flare(0.9);
      }
    }

    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
    } else if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      wisp.hopTo(agent, time);
    }
    wisp.update(frame.dtReal, time);

    cellToWorldVec(goal.x, goal.y, grid, shaft.root.position);
    const grow = growth(time - this.growAt);
    shaft.setState(grow);
    shaft.setTime(time);
    shaft.update(frame.dtReal, time);
    this.relightFloor(goal.x, goal.y);
    world.setGoalPosition(shaft.root.position.x, 1.5, shaft.root.position.z);

    const bloom = envelope(time - this.bloomAt, BLOOM_SECONDS, 2.4);
    const beamLight = world.rig.points[0];
    if (beamLight) {
      beamLight.position.set(shaft.root.position.x, shaft.lightHeight, shaft.root.position.z);
      beamLight.intensity = (0.22 + 0.6 * grow) * (1 + 0.9 * bloom);
    }
    const wispLight = world.rig.points[1];
    if (wispLight) {
      this.lightPos.copy(wisp.position);
      this.lightPos.y += TOKENS.wispHeight;
      wispLight.position.copy(this.lightPos);
      wispLight.intensity = 0.16 + 0.8 * bloom;
    }
    const bloomLight = world.rig.points[2];
    if (bloomLight) {
      bloomLight.position.copy(this.bloomPos);
      bloomLight.intensity = 2.2 * bloom;
    }

    world.setTime(time);
    if (this.sceneView) world.setView(this.sceneView.update(rig, frame.dtReal, this.follow ? this.track(frame) : undefined), this.camera);
  }

  private track(frame: FrameUpdate): ViewAnchor {
    const grid = this.grid, inset = FRAMED_SIDE * CELL / 2, x = (grid.w * CELL) / 2 - inset, z = (grid.h * CELL) / 2 - inset;
    cellToWorldVec(trackPose(frame, MAZE_X), trackPose(frame, MAZE_Y), grid, this.aim);
    this.aim.set(Math.min(x, Math.max(-x, this.aim.x)), 0, Math.min(z, Math.max(-z, this.aim.z)));
    if (frame.scrubbed) this.focus.copy(this.aim);
    else this.focus.lerp(this.aim, damp(FOLLOW_RATE, frame.dtReal));
    return [this.focus.x + Math.sin(AZIMUTH) * this.lead, 0.15, this.focus.z + Math.cos(AZIMUTH) * this.lead];
  }

  private arrive(frame: FrameUpdate, step: number, at: number, time: number, pool: CelebrationPool): void {
    this.bloomAt = at;
    this.growAt = at;
    const age = time - at;
    if (age > BLOOM_SECONDS) return;
    const reached = frame.poseAt(step - 1);
    cellToWorldVec(reached[MAZE_GX]!, reached[MAZE_GY]!, this.grid, this.bloomPos);
    this.groundPos.copy(this.bloomPos);
    this.bloomPos.y = BLOOM_Y;
    pool.burst(this.bloomPos, this.moteColor, BURST_SCALE, at);
    pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, at);
    pool.pulse.reset();
    pool.pulse.fire(Math.max(0, BURST_SCALE - age / PULSE_SECONDS));
    pool.update(0, time);
  }

  private relightFloor(x: number, y: number): void {
    if (x === this.litX && y === this.litY) return;
    const world = this.world;
    if (!world) return;
    if (this.litX >= 0) world.field.setFloorColor(this.litX, this.litY, floorTintFor(this.grid, this.skin, this.litX, this.litY, this.ctx!.seed, this.restoreTint));
    world.field.setFloorColor(x, y, this.litTint);
    this.litX = x;
    this.litY = y;
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.rig && this.sceneView) this.sceneView.resize(size, this.rig);
  }

  protected drop(): void {
    this.shaft?.dispose();
    this.shaft = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.growAt = -Infinity;
    this.bloomAt = -Infinity;
    this.litX = -1;
    this.litY = -1;
  }
}

function growth(age: number): number {
  if (!(age >= 0) || age >= GROW_SECONDS) return 1;
  const t = age / GROW_SECONDS;
  const e = 1 - Math.pow(1 - t, 3);
  return e * (1 + 0.13 * Math.sin(Math.PI * t) * (1 - t));
}

export default ((() => new OvergrownRuinsScene()) satisfies SceneFactory);
