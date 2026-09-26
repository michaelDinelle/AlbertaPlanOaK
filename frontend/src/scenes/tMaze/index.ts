import * as THREE from 'three/webgpu';
import { PULSE_SECONDS, type CelebrationPool } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { cellToWorld, cellToWorldVec } from '@kit/grid';
import { envelope } from '@kit/mix';
import { damp, orbitPosition, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameUpdate, LegendSpec, SceneContext, SceneFactory, ScenePost, SceneSize } from '@kit/types';
import { makeSceneView, type SceneView } from '@kit/view';
import { Wisp } from '@kit/wisp';
import { lastEvent } from '@sim/events';
import { TMAZE_CUE, TMAZE_POS, TMAZE_TRIAL, tMazeArm } from '@sim/envs/tmaze';
import type { GridCell, GridGeometry } from '@sim/types';
import { parameter } from '../../data/envs';
import { formatCount, formatSigned } from '../../format';
import { buildCueCrystal, buildExitMouth, buildTraveler, CUE_EXTRA_COLORS, CUE_NAMES, type CueCrystalHandle, type ExitMouthHandle } from './actors';
import { buildWorld, groundAt, HERO_DISTANCE, resolveGrid, type WorldHandle } from './world';

const LANTERN = 0xffe4b0;
const ARM_REACH = 2.6;
const ARM_STRIDE = 3.2;
const HOLD_SECONDS = 0.6;
const FLARE_SECONDS = 2.2;
const BURST_SCALE = 0.6;
const SIGH_STRENGTH = 0.55;
const REPLAY_SECONDS = 2;
const HERO_POLAR = 1.36;
const HERO_EXTENT = 35.1;
const AZIMUTH = -0.45;
const CLOSEST = 16;
const EDGE = 0.91;
const SEARCH = 30;
const FOLLOW_RATE = 2.5;

interface Outcome {
  step: number;
  arm: number;
  success: boolean;
}

function armPosition(arm: number, end: number, out: THREE.Vector3): THREE.Vector3 {
  return out.set(end + 0.85, 0, (arm & 1 ? 1 : -1) * (ARM_REACH + (arm >> 1) * ARM_STRIDE));
}

const polarAt = (distance: number) => HERO_POLAR - 0.16 * Math.min(1, (HERO_DISTANCE - distance) / (HERO_DISTANCE - CLOSEST));

function anchorRange(fov: number, distance: number, start: THREE.Vector3[], finish: THREE.Vector3[]): [number, number] {
  const camera = new THREE.PerspectiveCamera(fov, 16 / 9, 0.1, 1000), target = new THREE.Vector3(), point = new THREE.Vector3(), polar = polarAt(distance);
  const reach = (anchor: number, points: THREE.Vector3[], side: number) => {
    orbitPosition({ azimuth: AZIMUTH, polar, distance }, target.set(anchor, 1, 0), camera.position);
    camera.lookAt(target);
    camera.updateMatrixWorld();
    return Math.max(...points.map(p => side * point.copy(p).project(camera).x));
  };
  const solve = (points: THREE.Vector3[], side: number) => {
    let low = start[0]!.x - SEARCH, high = finish[0]!.x + SEARCH;
    for (let i = 0; i < 40; i++) {
      const mid = (low + high) / 2;
      if ((reach(mid, points, side) > EDGE) === (side < 0)) high = mid; else low = mid;
    }
    return low;
  };
  return [solve(start, -1), solve(finish, 1)];
}

function framedDistance(fov: number, least: number, start: THREE.Vector3[], finish: THREE.Vector3[]): number {
  const fits = (distance: number) => { const [low, high] = anchorRange(fov, distance, start, finish); return low >= high; };
  if (fits(least)) return least;
  if (!fits(HERO_DISTANCE)) return HERO_DISTANCE;
  let low = least, high = HERO_DISTANCE;
  for (let i = 0; i < 24; i++) {
    const mid = (low + high) / 2;
    if (fits(mid)) high = mid; else low = mid;
  }
  return high;
}

const corners = (x: number, ys: readonly number[], zs: readonly number[]) => ys.flatMap(y => zs.map(z => new THREE.Vector3(x, y, z)));

class LimestoneLakeScene extends StageScene<WorldHandle> {
  readonly post: ScenePost = { bloom: 0.5, fog: { from: ['cave', 'fog'] } };
  readonly controlHz = 8;
  protected override poolOptions() { return { burstSize: 90 }; }
  protected readonly stageName = 'limestoneLake';
  private view: SceneView | null = null;
  private grid!: GridGeometry;
  private cueBits = 1;
  private colors: number[] = [];
  private wisp: Wisp | null = null;
  private traveler: ReturnType<typeof buildTraveler> | null = null;
  private cue: CueCrystalHandle | null = null;
  private readonly arms: ExitMouthHandle[] = [];
  private outcome: Outcome | null = null;
  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly effect = new THREE.Vector3();
  private readonly anchor: [number, number, number] = [0, 1, 0];
  private span: [number, number] | null = null;
  private readonly ledger = { pos: 0, cue: 0, seen: true, trial: 0, action: '—', last: '—', reward: 0 };
  override legend: LegendSpec = {
    title: 'CAUSEWAY',
    rows: [
      { key: 'at', label: 'stone', tint: LANTERN, value: () => `${this.ledger.pos + 1} of ${this.grid.w}` },
      { key: 'cue', label: 'cue', tint: 0x5ee0ff, value: () => `${CUE_NAMES[this.ledger.cue]} · ${this.ledger.seen ? 'in view' : 'out of sight'}` },
      { key: 'action', label: 'action', tint: 0xdcf4ff, value: () => this.ledger.action },
      { key: 'last', label: 'last choice', tint: 0xffd166, value: () => this.ledger.last },
      { key: 'reward', label: 'reward', tint: 0xffd166, value: () => formatSigned(this.ledger.reward) },
    ],
    note: () => `trial ${formatCount(this.ledger.trial + 1)}`,
    bar: () => this.ledger.pos / (this.grid.w - 1),
  };

  protected buildWorld(ctx: SceneContext): WorldHandle {
    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;
    this.cueBits = parameter(ctx.environment, 'cueBits');
    const first = cellToWorld(0, 0, grid)[0], end = cellToWorld(grid.w - 1, 0, grid)[0];
    const reach = ARM_REACH + ((1 << this.cueBits) - 1 >> 1) * ARM_STRIDE + 1.6;
    const left = first - 1.5, right = end + 2.6;
    const start = corners(first - 1.4, [-0.2, 2.6], [-2.8, 0.6]), finish = corners(end + 2.5, [-0.2, 3.3], [-reach, reach]);
    const distance = framedDistance(ctx.tokens.fovHero, Math.min(HERO_DISTANCE, Math.max(CLOSEST, HERO_DISTANCE * (right - left) / HERO_EXTENT)), start, finish);
    const polar = polarAt(distance), [low, high] = anchorRange(ctx.tokens.fovHero, distance, start, finish);
    this.span = low < high ? [low, high] : null;
    this.anchor[0] = Math.min(low, Math.max(high, 0));
    this.view = makeSceneView(ctx, { min: [left, -0.4, -reach], max: [right, 3.5, reach] }, this.anchor, { distance, azimuth: AZIMUTH, polar });
    return buildWorld(ctx, this.scene, grid, distance);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const grid = this.grid;
    this.colors = [ctx.tokens.cueCyan, ctx.tokens.cueMagenta, ...CUE_EXTRA_COLORS].slice(0, 1 << this.cueBits);
    const wisp = new Wisp({ color: LANTERN, radius: 0.2, skirt: false, ribbon: false, silhouette: false, hop: { duration: 0.11, height: 0.1 }, name: 'lanternWisp' }, grid);
    wisp.core.visible = false;
    const halo = wisp.group.getObjectByName('oakWispHalo');
    if (halo) halo.visible = false;
    this.wisp = wisp;
    this.traveler = buildTraveler();
    wisp.avatar.add(this.traveler.group);
    this.scene.add(wisp.group);
    const half = cellToWorld(grid.w - 1, 0, grid)[0], rest = (at: THREE.Vector3) => { at.y = Math.max(0, groundAt(half, at.x, at.z)); };
    this.cue = buildCueCrystal(world.skin, this.colors);
    cellToWorldVec(0, 0, grid, this.cue.group.position).z = -1.6;
    rest(this.cue.group.position);
    this.scene.add(this.cue.group);
    this.colors.forEach((color, arm) => {
      const mouth = buildExitMouth(world.skin, color, arm);
      rest(armPosition(arm, half, mouth.group.position));
      this.arms.push(mouth);
      this.scene.add(mouth.group);
    });
    world.rig.points[0]?.color.set(LANTERN);
  }

  protected rigOptions(): OrbitRigOptions {
    if (!this.view) throw new Error('Cavern view is missing');
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    const cell: GridCell = { x: 0, y: 0 };
    return {
      agentCell: state => { cell.x = Math.round(state.pose[TMAZE_POS]!); return cell; },
      goalCell: () => this.grid.goal!,
      debug: (frame, kit) => {
        const cue = Math.round(frame.state.pose[TMAZE_CUE]!), mouth = this.arms[cue], plinth = this.cue;
        if (!mouth || !plinth) return;
        const marks = kit.marks('trueCue', 2), color = this.colors[cue]!;
        marks.set(mouth.group.position.x, mouth.group.position.y + 3.4, mouth.group.position.z, 0.2, color);
        marks.set(plinth.group.position.x, plinth.group.position.y + 2.7, plinth.group.position.z, 0.14, color);
      },
    };
  }

  protected start(_ctx: SceneContext, world: WorldHandle): void {
    this.wisp?.setCell(0, 0, this.grid);
    world.setTime(0);
    if (this.rig && this.view) world.setView(this.view.update(this.rig, 0));
  }

  private resolve(frame: FrameUpdate): Outcome | null {
    const step = lastEvent(frame.eventLog, 'done', frame.state.step);
    if (step === undefined) return null;
    if (this.outcome?.step === step) return this.outcome;
    const at = frame.frameAt(step);
    if (at.step !== step) return null;
    return { step, arm: tMazeArm(at.actions[0]!, this.cueBits), success: at.reward > 0 };
  }

  private celebrate(outcome: Outcome, at: number, time: number, pool: CelebrationPool): void {
    if (time - at > REPLAY_SECONDS) return;
    this.effect.copy(this.arms[outcome.arm]!.group.position);
    this.effect.y += 1.4;
    if (outcome.success) pool.burst(this.effect, this.colors[outcome.arm]!, BURST_SCALE, at);
    else pool.sigh(SIGH_STRENGTH, this.effect, at);
    pool.pulse.reset();
    if (outcome.success) pool.pulse.fire(Math.max(0, BURST_SCALE - (time - at) / PULSE_SECONDS));
    else pool.pulse.fireSigh(Math.max(0, SIGH_STRENGTH - (time - at) / PULSE_SECONDS));
    pool.update(0, time);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const wisp = this.wisp, cue = this.cue, view = this.view;
    if (!wisp || !cue || !view) return;
    const state = frame.state, grid = this.grid, pos = Math.round(state.pose[TMAZE_POS]!);
    pool.update(frame.dtReal, time);
    const outcome = this.resolve(frame);
    this.outcome = outcome;
    if (frame.scrubbed) {
      pool.clear();
      if (outcome) this.celebrate(outcome, outcome.step / frame.controlHz, time, pool);
    }
    let teleported = false;
    for (const event of frame.events) {
      if (event.type === 'teleport') teleported = true;
      else if ((event.type === 'goal' || event.type === 'harm') && outcome) this.celebrate(outcome, time, time, pool);
    }
    if (frame.scrubbed || teleported || Math.abs(pos - wisp.cellX) > 1) {
      wisp.setCell(pos, 0, grid);
      wisp.snap();
    } else if (pos !== wisp.cellX) {
      this.agentCell.x = pos;
      wisp.hopTo(this.agentCell, time);
    }
    wisp.update(frame.dtReal, time);
    const age = outcome ? time - outcome.step / frame.controlHz : -1;
    if (this.span) {
      const goal = age >= 0 && age < HOLD_SECONDS ? this.span[1] : Math.min(this.span[1], Math.max(this.span[0], wisp.position.x));
      this.anchor[0] = frame.scrubbed ? goal : this.anchor[0] + (goal - this.anchor[0]) * damp(FOLLOW_RATE, frame.dtReal);
    }
    const obs = state.observations, seen = obs[2] === 1;
    cue.setCue((obs[1]! > 0.5 ? 1 : 0) | ((obs[6] ?? 0) > 0.5 ? 2 : 0), seen);
    cue.setTime(time);
    const flare = age >= 0 ? envelope(Math.max(0, age - HOLD_SECONDS), FLARE_SECONDS, 1.4) : 0;
    this.arms.forEach((mouth, arm) => {
      const chosen = outcome?.arm === arm, success = outcome?.success ?? true;
      mouth.setGlow(1 + (chosen ? flare * (success ? 2.5 : 0.8) : 0), !chosen || success);
      mouth.setTime(time);
    });
    const light = world.rig.points[0];
    if (light) {
      light.position.copy(wisp.position);
      light.position.y += 0.6;
      light.intensity = 1.6;
    }
    const now = frame.frameAt(state.step), ledger = this.ledger;
    ledger.pos = pos;
    ledger.cue = Math.round(state.pose[TMAZE_CUE]!);
    ledger.seen = seen;
    ledger.trial = Math.round(state.pose[TMAZE_TRIAL]!);
    ledger.reward = state.reward;
    ledger.action = state.step === 0 ? '—' : now.done ? `chose ${CUE_NAMES[tMazeArm(now.actions[0]!, this.cueBits)]}` : (Math.trunc(now.actions[0]!) & 1) === 1 ? 'forward →' : '← back';
    ledger.last = outcome ? `${CUE_NAMES[outcome.arm]} ${outcome.success ? '✓' : '✗'}` : '—';
    world.setTime(time);
    world.setView(view.update(rig, frame.dtReal, this.anchor));
  }

  override resize(size: SceneSize): void {
    super.resize(size);
    if (this.view && this.rig && this.world) {
      this.view.resize(size, this.rig);
      this.world.setView(this.view.update(this.rig, 0));
    }
  }

  protected drop(): void {
    for (const arm of this.arms) arm.dispose();
    this.arms.length = 0;
    this.cue?.dispose();
    this.cue = null;
    this.traveler?.dispose();
    this.traveler = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.outcome = null;
  }
}

export default ((() => new LimestoneLakeScene()) satisfies SceneFactory);
