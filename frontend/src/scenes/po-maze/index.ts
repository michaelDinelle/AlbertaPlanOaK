
import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import { GroundTruthOverlay } from '@kit/debug';
import { cellToWorld, cellToWorldVec, clampCell, type GridSize, type Vec3Tuple } from '@kit/grid';
import { hash1f } from '@kit/rng';
import { makeOrbitRig, type OrbitRig } from '@kit/rigs';
import { TOKENS } from '@kit/tokens';
import type {
  AlignmentProbe,
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import { Wisp } from '@kit/wisp';
import * as THREE from 'three';
import type { GridCell, GridGeometry } from '@sim/types';
import {
  buildGoalBeacon,
  buildLanternPool,
  buildMoonflower,
  buildWispAura,
  type GoalBeaconHandle,
  type LanternPoolHandle,
  type MoonflowerHandle,
  type WispAuraHandle,
} from './actors';
import { manifest } from './manifest';
import {
  BUBBLE_OUTER,
  CAMERA_AZIMUTH,
  blendTokens,
  bubbleVisibility,
  buildWorld,
  clamp01,
  compassOffset,
  forestSkin,
  resolveGrid,
  revealVisibility,
  REVEAL_RADIUS,
  type WorldHandle,
} from './world';


const CAM_AZIMUTH = CAMERA_AZIMUTH;
const CAM_POLAR = 1.32;
const CAM_DISTANCE = 4.7;
const CAM_TARGET_Y = 0.82;
const CAM_DISTANCE_THUMB = 6.9;
const CAM_AZIMUTH_THUMB = CAMERA_AZIMUTH + 0.22;
const CAM_POLAR_THUMB = 1.25;
const CAM_YAW_RANGE = 0.55;
const CAM_MIN_POLAR = 1.04;
const CAM_MAX_POLAR = 1.36;
const CAM_MIN_DISTANCE = 3.6;
const CAM_MAX_DISTANCE = 6.6;
const CAM_FOLLOW_RATE = 2.2;

const BURST_SCALE = 0.45;
const RIPPLE_SCALE = 0.52;
const BURST_PARTICLES = 120;
const PULSE_STRENGTH = 0.5;
const BLOOM_Y = 0.55;

export const REVEAL_SECONDS = 1.9;
export const LINGER_SECONDS = 1.35;

const HOP_SECONDS = 0.11;
const HOP_HEIGHT = 0.14;

const DEFAULT_BLOOM = forestSkin(TOKENS.palettes.dawn, TOKENS).bloom.getHex();

interface POMazeLike {
  x?: unknown;
  y?: unknown;
  gx?: unknown;
  gy?: unknown;
}

export function readCells(state: FrameState, grid: GridSize, agent: GridCell, goal: GridCell): void {
  const d = state.derived as POMazeLike | null | undefined;
  const has = d !== null && typeof d === 'object';
  const ax = has && typeof d.x === 'number' ? d.x : (state.pose[0] ?? 0);
  const ay = has && typeof d.y === 'number' ? d.y : (state.pose[1] ?? 0);
  const gx = has && typeof d.gx === 'number' ? d.gx : (state.pose[2] ?? 0);
  const gy = has && typeof d.gy === 'number' ? d.gy : (state.pose[3] ?? 0);
  agent.x = Number.isFinite(ax) ? Math.round(ax) : 0;
  agent.y = Number.isFinite(ay) ? Math.round(ay) : 0;
  goal.x = Number.isFinite(gx) ? Math.round(gx) : 0;
  goal.y = Number.isFinite(gy) ? Math.round(gy) : 0;
  clampCell(agent, grid);
  clampCell(goal, grid);
}

class FogboundForestScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(TOKENS.fovHero, 16 / 9, 0.1, 900);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private wisp: Wisp | null = null;
  private flower: MoonflowerHandle | null = null;
  private beacon: GoalBeaconHandle | null = null;
  private lantern: LanternPoolHandle | null = null;
  private aura: WispAuraHandle | null = null;
  private pool: CelebrationPool | null = null;
  private rig: OrbitRig | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private grid: GridGeometry = resolveGrid({ kind: 'grid' });

  private revealAt = -Infinity;
  private lingerAt = -Infinity;
  private revealX = 0;
  private revealZ = 0;

  private readonly agentCell: GridCell = { x: 0, y: 0 };
  private readonly goalCell: GridCell = { x: 0, y: 0 };
  private readonly prevGoalCell: GridCell = { x: 0, y: 0 };
  private readonly prevAgentCell: GridCell = { x: 0, y: 0 };
  private readonly bloomPos = new THREE.Vector3();
  private readonly groundPos = new THREE.Vector3();
  private readonly goalWorld = new THREE.Vector3();
  private readonly camAnchor = new THREE.Vector3();
  private readonly lightPos = new THREE.Vector3();
  private camPrimed = false;
  private camFixed = false;
  private lingering = false;
  private gold = TOKENS.celebrationGold;
  private bloomColor = DEFAULT_BLOOM;

  private debugOn = false;

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'fogboundForestScene';

    const grid = resolveGrid(ctx.geometry);
    this.grid = grid;

    const palette = ctx.tokens.palettes.dawn;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.gold = ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene, grid);
    this.world = world;
    this.scene.add(world.group);
    this.bloomColor = world.skin.bloom.getHex();

    const lanternLight = world.rig.points[0];
    if (lanternLight) lanternLight.color.set(blendTokens(accent, palette.accentLight, 0.62));
    const breathLight = world.rig.points[1];
    if (breathLight) breathLight.color.set(this.gold);
    const flowerLight = world.rig.points[2];
    if (flowerLight) flowerLight.color.set(world.skin.bloom);

    const wispColor = blendTokens(accent, this.gold, 0.34);
    const wisp = new Wisp(
      {
        color: wispColor,
        tier: ctx.tier,
        radius: 0.09,
        hop: { duration: HOP_SECONDS, height: HOP_HEIGHT },
        name: 'forestWisp',
      },
      grid,
    );
    this.wisp = wisp;
    this.scene.add(wisp.group);

    const lantern = buildLanternPool(
      blendTokens(accent, palette.accentLight, 0.55),
      BUBBLE_OUTER * 1.15,
      ctx.tier,
    );
    this.lantern = lantern;
    this.scene.add(lantern.mesh);

    const aura = buildWispAura(blendTokens(wispColor, palette.sun, 0.4), 1.05, ctx.tier);
    this.aura = aura;
    this.scene.add(aura.mesh);

    const flower = buildMoonflower({
      tier: ctx.tier,
      skin: world.skin,
      glow: world.skin.bloom,
      name: 'moonflowerRoot',
    });
    this.flower = flower;
    this.scene.add(flower.root);
    this.scene.add(flower.bloom);

    const beacon = buildGoalBeacon({
      tier: ctx.tier,
      glow: world.skin.bloom,
      name: 'goalBeacon',
    });
    this.beacon = beacon;
    this.scene.add(beacon.group);

    const thumb = ctx.tier === 'thumb';
    this.camFixed = thumb;
    if (thumb) {
      const thumbCell: GridCell = {
        x: Math.round(manifest.thumbFrame.pose[0] ?? 0),
        y: Math.round(manifest.thumbFrame.pose[1] ?? 0),
      };
      clampCell(thumbCell, grid);
      cellToWorldVec(thumbCell.x, thumbCell.y, grid, this.camAnchor);
    }
    this.camAnchor.y = CAM_TARGET_Y;
    const rig = makeOrbitRig({
      canvas: thumb ? null : ctx.canvas,
      target: [this.camAnchor.x, CAM_TARGET_Y, this.camAnchor.z],
      distance: thumb ? CAM_DISTANCE_THUMB : CAM_DISTANCE,
      azimuth: thumb ? CAM_AZIMUTH_THUMB : CAM_AZIMUTH,
      polar: thumb ? CAM_POLAR_THUMB : CAM_POLAR,
      fov: ctx.tokens.fovHero,
      near: 0.1,
      far: 900,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 5,
      rotateSpeed: 0.004,
      zoomSpeed: 0.0012,
      name: 'forestOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      tier: ctx.tier,
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'forestCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    const overlayAgent: GridCell = { x: 0, y: 0 };
    const overlayGoal: GridCell = { x: 0, y: 0 };
    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayAgent;
      },
      goalCell: (state) => {
        readCells(state, grid, overlayAgent, overlayGoal);
        return overlayGoal;
      },
      name: 'forestGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    world.setPixelRatio(ctx.size.dpr);
    world.setTime(0);
    world.setFogLift(0);
    wisp.dim(0.5);
    wisp.setCell(0, 0, grid);
    flower.setState(0, 0);
    lantern.setGain(1);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const wisp = this.wisp;
    const flower = this.flower;
    const lantern = this.lantern;
    const aura = this.aura;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !wisp || !flower || !lantern || !aura || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;
    const grid = this.grid;
    const agent = this.agentCell;
    const goal = this.goalCell;

    pool.update(frame.dtReal, time);

    readCells(state, grid, agent, goal);

    if (frame.scrubbed) {
      pool.clear();
      this.revealAt = -Infinity;
      this.lingerAt = -Infinity;
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
      this.camPrimed = false;
      this.lingering = false;
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        const prev = frame.prev;
        if (prev) {
          readCells(prev, grid, this.prevAgentCell, this.prevGoalCell);
          cellToWorldVec(this.prevGoalCell.x, this.prevGoalCell.y, grid, this.bloomPos);
        } else {
          cellToWorldVec(agent.x, agent.y, grid, this.bloomPos);
        }
        this.groundPos.copy(this.bloomPos);
        this.revealX = this.bloomPos.x;
        this.revealZ = this.bloomPos.z;
        this.bloomPos.y = BLOOM_Y;
        pool.burst(this.bloomPos, this.bloomColor, BURST_SCALE, time);
        pool.ripple(this.groundPos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        wisp.flare(1);
        flower.flare(0.9);
        this.revealAt = time;
        this.lingerAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    if (teleported) {
      wisp.setCell(agent.x, agent.y, grid);
      wisp.snap();
    } else if (agent.x !== wisp.cellX || agent.y !== wisp.cellY) {
      wisp.face({ dx: agent.x - wisp.cellX, dy: agent.y - wisp.cellY });
      wisp.hopTo(agent, time);
    }
    wisp.update(frame.dtReal, time);

    const wx = wisp.position.x;
    const wz = wisp.position.z;

    const reveal = envelope(time - this.revealAt, REVEAL_SECONDS, 1.6);
    const breath = clamp01(reveal * 2.2);
    world.spruce.setBubble(wx, wz, this.revealX, this.revealZ, breath);
    world.setFogLift(reveal);
    world.setShadowFocus(wx, wz);

    for (let i = 0; i < world.gates.length; i++) {
      const gate = world.gates[i];
      if (!gate) continue;
      const gx = gate.root.position.x;
      const gz = gate.root.position.z;
      let v = bubbleVisibility(wx, wz, gx, gz);
      if (breath > 0) {
        const r = breath * revealVisibility(this.revealX, this.revealZ, gx, gz, REVEAL_RADIUS);
        if (r > v) v = r;
      }
      const e = v <= 0 ? 0.0001 : v;
      gate.body.scale.set(e, e * (0.4 + 0.6 * e), e);
      gate.body.visible = v > 0.002;
    }

    cellToWorldVec(goal.x, goal.y, grid, this.goalWorld);
    flower.root.position.copy(this.goalWorld);
    flower.bloom.position.set(this.revealX, 0, this.revealZ);
    flower.setState(breath, time);
    flower.update(frame.dtReal, time);

    const beacon = this.beacon;
    if (beacon) {
      beacon.group.position.set(this.goalWorld.x, 0, this.goalWorld.z);
      beacon.setTime(time);
    }


    lantern.mesh.position.set(wx, 0.015, wz);
    lantern.setGain(1 + 0.55 * reveal);
    aura.mesh.position.set(wx, wisp.position.y + TOKENS.wispHeight, wz);
    aura.face(this.camera);
    aura.setGain(0.72 + 1.4 * reveal);

    const lanternLight = world.rig.points[0];
    if (lanternLight) {
      this.lightPos.set(wx, TOKENS.wispHeight + 0.12, wz);
      lanternLight.position.copy(this.lightPos);
      lanternLight.intensity = 0.62 + 0.5 * reveal;
    }
    const breathLight = world.rig.points[1];
    if (breathLight) {
      breathLight.position.set(this.revealX, 0.75, this.revealZ);
      breathLight.intensity = 2.8 * reveal;
    }
    const flowerLight = world.rig.points[2];
    if (flowerLight) {
      flowerLight.position.set(this.revealX, flower.crownHeight, this.revealZ);
      flowerLight.intensity = 0.75 * breath;
    }
    const beaconLight = world.rig.points[3];
    if (beaconLight) {
      beaconLight.name = 'beacon:goalLight';
      beaconLight.color.set(world.skin.bloom);
      beaconLight.position.set(this.goalWorld.x, 1.4, this.goalWorld.z);
      beaconLight.intensity = 0.3;
    }

    world.setTime(time);

    if (!this.camFixed) {
      const linger = time - this.lingerAt >= 0 && time - this.lingerAt < LINGER_SECONDS;
      const tx = linger ? this.revealX : wx;
      const tz = linger ? this.revealZ : wz;
      const cut = !this.camPrimed || frame.scrubbed || teleported || (this.lingering && !linger);
      if (cut) {
        this.camAnchor.set(tx, CAM_TARGET_Y, tz);
        this.camPrimed = true;
        rig.setTarget(tx, CAM_TARGET_Y, tz);
        rig.snap();
      } else {
        const k = 1 - Math.exp(-CAM_FOLLOW_RATE * Math.max(0, frame.dtReal));
        this.camAnchor.x += (tx - this.camAnchor.x) * k;
        this.camAnchor.z += (tz - this.camAnchor.z) * k;
        rig.setTarget(this.camAnchor.x, CAM_TARGET_Y, this.camAnchor.z);
      }
      this.lingering = linger;
    }
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

  probes(): AlignmentProbe[] {
    const grid = this.grid;
    const wisp = this.wisp;
    const flower = this.flower;
    const world = this.world;

    const wispCell: GridCell = { x: 0, y: 0 };
    const wispGoal: GridCell = { x: 0, y: 0 };
    const wispOut: Vec3Tuple = [0, 0, 0];
    const goalCell: GridCell = { x: 0, y: 0 };
    const goalGoal: GridCell = { x: 0, y: 0 };
    const goalOut: Vec3Tuple = [0, 0, 0];

    const probes: AlignmentProbe[] = [
      {
        name: 'wisp-anchor',
        object: wisp?.avatar ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readCells(state, grid, wispCell, wispGoal);
          return cellToWorld(wispCell.x, wispCell.y, grid, wispOut);
        },
      },
      {
        name: 'goal-anchor',
        object: flower?.root ?? new THREE.Object3D(),
        expectWorldPos: (state) => {
          readCells(state, grid, goalCell, goalGoal);
          return cellToWorld(goalGoal.x, goalGoal.y, grid, goalOut);
        },
      },
    ];

    if (world) {
      for (let i = 0; i < world.gates.length; i++) {
        const gate = world.gates[i];
        if (!gate) continue;
        const out: Vec3Tuple = [0, 0, 0];
        const cell = gate.cell;
        probes.push({
          name: `gap-gate${i}`,
          object: gate.root,
          expectWorldPos: () => cellToWorld(cell.x, cell.y, grid, out),
        });
      }
    }
    return probes;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.flower?.dispose();
    this.flower = null;
    this.beacon?.dispose();
    this.beacon = null;
    this.aura?.dispose();
    this.aura = null;
    this.lantern?.dispose();
    this.lantern = null;
    this.wisp?.dispose();
    this.wisp = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.revealAt = -Infinity;
    this.lingerAt = -Infinity;
    this.camPrimed = false;
    this.camFixed = false;
    this.lingering = false;
    this.camAnchor.set(0, 0, 0);
  }
}


export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

const createScene: SceneFactory = () => new FogboundForestScene();
export default createScene;
