import * as THREE from 'three/webgpu';
import { PULSE_SECONDS, type CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { CELL, cellToWorld, cellToWorldVec, type Vec3Tuple } from '@kit/grid';
import { blendTokens, clamp01, envelope } from '@kit/mix';
import type { OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import { lastEvent } from '@sim/events';
import { MAZE_GX, MAZE_GY, MAZE_X, MAZE_Y } from '@sim/envs/maze';
import type { GridCell, GridGeometry } from '@sim/types';
import { formatCount, formatSigned } from '../../format';
import { buildGoalBeacon, buildLanternPool, buildMoonflower, buildWispAura } from './actors';
import { BUBBLE_OUTER, buildWorld, type WorldHandle } from './world';

const ACTIONS = ['↑ north', '↓ south', '← west', 'east →'] as const;
const NEIGHBOURS = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]] as const;
const BEARINGS = ['↖ NW', '↑ N', '↗ NE', '← W', '·', '→ E', '↙ SW', '↓ S', '↘ SE'] as const;
const EAST = 10, WEST = 11, SOUTH = 12, NORTH = 13;

const BURST_SCALE = 0.45;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.55;
const REVEAL_SECONDS = 1.9;
const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.14;

function readCells(state: FrameState, agent: GridCell, goal: GridCell): void {
  agent.x = state.pose[MAZE_X]!;
  agent.y = state.pose[MAZE_Y]!;
  goal.x = state.pose[MAZE_GX]!;
  goal.y = state.pose[MAZE_GY]!;
}

class FogboundForestScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['dawn', 'fog'] } };
  readonly controlHz = 8;
  protected readonly stageName = 'fogboundForest';

  private view: SceneView | null = null;
  private grid!: GridGeometry;
  private wisp: Wisp | null = null;
  private flower: ReturnType<typeof buildMoonflower> | null = null;
  private beacon: ReturnType<typeof buildGoalBeacon> | null = null;
  private lantern: ReturnType<typeof buildLanternPool> | null = null;
  private aura: ReturnType<typeof buildWispAura> | null = null;
  private lights: THREE.PointLight[] = [];
  private revealAt = -Infinity;
  private bloomColor = 0;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly goalPos = new THREE.Vector3();
  private readonly ledger = { trees: 0, bearing: 4, action: 0, reward: 0, since: null as number | null };

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = ctx.geometry.grid;
    if (!grid) throw new Error(`${ctx.envId} has no recorded maze grid`);
    this.grid = grid;
    const halfX = grid.w * CELL / 2, halfZ = grid.h * CELL / 2;
    this.view = makeSceneView(ctx, { min: [-halfX, 0, -halfZ], max: [halfX, 1.8, halfZ] }, [0, 0.82, 0], { distance: 17, azimuth: -1.35, polar: 0.73 });
    return buildWorld(ctx, this.scene, grid, this.accentOf(ctx));
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const palette = ctx.tokens.palettes.dawn, accent = this.accentOf(ctx), skin = world.skin, gold = ctx.tokens.celebrationGold;
    this.bloomColor = skin.bloom.getHex();
    this.lights = world.rig.points;
    const [lanternLight, breathLight, flowerLight, beaconLight] = this.lights;
    lanternLight!.color.set(blendTokens(accent, palette.accentLight, 0.62));
    breathLight!.color.set(gold);
    flowerLight!.color.set(skin.bloom);
    beaconLight!.color.set(skin.bloom);
    beaconLight!.intensity = 0.3;

    const wispColor = blendTokens(accent, gold, 0.34);
    this.wisp = new Wisp({ color: wispColor, radius: 0.09, hop: { duration: HOP_SECONDS, height: HOP_HEIGHT }, name: 'forestWisp' }, this.grid);
    this.lantern = buildLanternPool(blendTokens(accent, palette.accentLight, 0.55), BUBBLE_OUTER * 1.15);
    this.aura = buildWispAura(blendTokens(wispColor, palette.sun, 0.4), 1.05);
    this.flower = buildMoonflower(skin);
    this.beacon = buildGoalBeacon(skin.bloom);
    this.scene.add(this.wisp.group, this.lantern.mesh, this.aura.mesh, this.flower.body, this.beacon.group);

    this.legend = {
      title: 'FOGBOUND FOREST',
      rows: [
        { key: 'at', label: 'wisp', tint: wispColor.getHex(), value: () => `column ${this.agentCell.x} · row ${this.agentCell.y}` },
        { key: 'goal', label: 'beacon', tint: this.bloomColor, value: () => `column ${this.goalCell.x} · row ${this.goalCell.y}` },
        { key: 'senses', label: 'senses', tint: accent, value: () => `${this.ledger.trees} ${this.ledger.trees === 1 ? 'tree' : 'trees'} · beacon ${BEARINGS[this.ledger.bearing]}` },
        { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => ACTIONS[this.ledger.action] ?? '—' },
        { key: 'reward', label: 'reward', tint: gold, value: () => formatSigned(this.ledger.reward) },
        { key: 'bloom', label: 'last bloom', tint: gold, value: () => this.ledger.since === null ? 'none yet' : `${formatCount(this.ledger.since)} ${this.ledger.since === 1 ? 'step' : 'steps'} ago` },
      ],
    };
  }

  protected rigOptions(): OrbitRigOptions {
    return this.view!.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const agent: GridCell = { x: 0, y: 0 }, goal: GridCell = { x: 0, y: 0 }, at: Vec3Tuple = [0, 0, 0];
    return {
      agentCell: (state) => { readCells(state, agent, goal); return agent; },
      goalCell: (state) => { readCells(state, agent, goal); return goal; },
      debug: (frame, kit) => {
        const obs = frame.state.observations, sensed = kit.marks('sensed', NEIGHBOURS.length), bearing = kit.curve('bearing', 2, TOKENS.celebrationGold);
        readCells(frame.state, agent, goal);
        NEIGHBOURS.forEach(([dx, dy], i) => {
          if (obs[i] !== 1) return;
          cellToWorld(agent.x + dx, agent.y + dy, this.grid, at);
          sensed.set(at[0], 0.9, at[2], 0.09, TOKENS.uiCyan);
        });
        cellToWorld(agent.x, agent.y, this.grid, at);
        bearing.point(at[0], 0.9, at[2]);
        bearing.point(at[0] + (obs[EAST]! - obs[WEST]!) * 0.8 * CELL, 0.9, at[2] + (obs[SOUTH]! - obs[NORTH]!) * 0.8 * CELL);
        if (frame.state.simTime - this.revealAt <= REVEAL_SECONDS) kit.marks('bloom', 1).set(this.bloomPos.x, 0.9, this.bloomPos.z, 0.14, TOKENS.celebrationGold);
        const wisp = this.wisp;
        if (wisp) kit.trail('walk', 120, TOKENS.uiCyan).push(wisp.position.x, 0.35, wisp.position.z);
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    world.setTime(0);
    world.setFogLift(0);
    this.wisp!.dim(0.5);
  }

  private bloom(frame: FrameUpdate, at: number | undefined, pool: CelebrationPool, time: number): void {
    if (at === undefined) return;
    const born = at / frame.controlHz, age = time - born;
    if (!(age <= REVEAL_SECONDS) || frame.frameAt(at - 1).step !== at - 1) return;
    const before = frame.poseAt(at - 1);
    cellToWorldVec(before[MAZE_GX]!, before[MAZE_GY]!, this.grid, this.bloomPos);
    this.revealAt = born;
    this.bloomPos.y = BLOOM_Y;
    pool.burst(this.bloomPos, this.bloomColor, BURST_SCALE, born);
    this.bloomPos.y = 0;
    pool.update(0, time);
    const fresh = Math.max(0, 1 - age / PULSE_SECONDS);
    pool.pulse.fire(PULSE_STRENGTH * fresh);
    this.flower!.flare(0.9 * fresh);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp!, flower = this.flower!, lantern = this.lantern!, aura = this.aura!, beacon = this.beacon!;
    const state = frame.state, grid = this.grid, agent = this.agentCell, goal = this.goalCell, obs = state.observations;

    pool.update(frame.dtReal, time);
    readCells(state, agent, goal);
    let trees = 0;
    for (let i = 0; i < NEIGHBOURS.length; i++) trees += obs[i]!;
    const bloomedAt = lastEvent(frame.eventLog, 'goal', state.step);
    this.ledger.trees = trees;
    this.ledger.bearing = 1 + obs[EAST]! - obs[WEST]! + 3 * (1 + obs[SOUTH]! - obs[NORTH]!);
    this.ledger.action = state.step === 0 ? -1 : Math.trunc(frame.frameAt(state.step).actions[0]!) & 3;
    this.ledger.reward = state.reward;
    this.ledger.since = bloomedAt === undefined ? null : state.step - bloomedAt;

    let bloomed = false, teleported = frame.scrubbed;
    for (const event of frame.events) {
      if (event.type === 'goal') bloomed = true;
      else if (event.type === 'teleport') teleported = true;
    }
    if (frame.scrubbed) {
      pool.clear();
      this.revealAt = -Infinity;
    }
    if (frame.scrubbed || bloomed) this.bloom(frame, bloomedAt, pool, time);
    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
    } else if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      wisp.face({ dx: agent.x - wisp.cellX, dy: agent.y - wisp.cellY });
      wisp.hopTo(agent, time);
    }
    if (bloomed) wisp.flare(1);
    wisp.update(frame.dtReal, time);

    const wx = wisp.position.x, wz = wisp.position.z, bx = this.bloomPos.x, bz = this.bloomPos.z;
    const reveal = envelope(time - this.revealAt, REVEAL_SECONDS, 1.6);
    const breath = clamp01(reveal * 2.2);
    world.setBubble(wx, wz, bx, bz, breath);
    world.setFogLift(reveal);

    cellToWorldVec(goal.x, goal.y, grid, this.goalPos);
    flower.body.position.set(bx, 0, bz);
    flower.setState(breath, time);
    flower.update(frame.dtReal, time);
    beacon.group.position.copy(this.goalPos);
    beacon.setTime(time);
    lantern.mesh.position.set(wx, 0.015, wz);
    lantern.setGain(1 + 0.55 * reveal);
    aura.mesh.position.set(wx, wisp.position.y + TOKENS.wispHeight, wz);
    aura.setGain(0.72 + 1.4 * reveal);

    const [lanternLight, breathLight, flowerLight, beaconLight] = this.lights;
    lanternLight!.position.set(wx, TOKENS.wispHeight + 0.12, wz);
    lanternLight!.intensity = 2.2 + 1.2 * reveal;
    breathLight!.position.set(bx, 1.6, bz);
    breathLight!.intensity = 2.8 * reveal;
    flowerLight!.position.set(bx, flower.crownHeight + 0.6, bz);
    flowerLight!.intensity = 0.75 * breath;
    beaconLight!.position.set(this.goalPos.x, 1.4, this.goalPos.z);

    world.setTime(time);
    world.setView(this.view!.update(rig, frame.dtReal), this.camera);
    aura.mesh.quaternion.copy(this.camera.quaternion);
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.rig && this.view) this.view.resize(size, this.rig);
  }

  protected drop(): void {
    this.flower?.dispose();
    this.beacon?.dispose();
    this.aura?.dispose();
    this.lantern?.dispose();
    this.wisp?.dispose();
    this.flower = this.beacon = this.aura = this.lantern = this.wisp = null;
    this.lights = [];
    this.revealAt = -Infinity;
  }
}

export default ((() => new FogboundForestScene()) satisfies SceneFactory);
