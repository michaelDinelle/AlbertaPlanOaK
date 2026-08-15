
import { GroundTruthOverlay } from '@kit/debug';
import { makeOrbitRig, orbitFromPose, type OrbitRig } from '@kit/rigs';
import {
  CelebrationPool,
  attachCelebrations,
  disposeCelebrations,
} from '@kit/celebrate';
import type {
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import * as THREE from 'three';
import { CARTPOLE_ACTION, CARTPOLE_TH, CARTPOLE_THDOT, CARTPOLE_X, CARTPOLE_XDOT, CP_TAU, cartPoleForce, cartPoleTip } from '@sim/envs/cartpole';
import { buildTrolley, type TrolleyHandle } from './actors';
import { manifest } from './manifest';
import {
  CAM_BASE,
  CAM_LOOK,
  buildWorld,
  clampRange,
  poleLength,
  tiltLimit,
  trackLimit,
  type WorldHandle,
} from './world';


const FALL_SECONDS = 0.42;
const COOL_SECONDS = 0.58;
const SIGH_STRENGTH = 0.5;
const SIGH_PARTICLES = 34;
const LIGHT_STANDOFF = 1.15;

const TH_ACC = 0.4;
const X_ACC = 0.25;
const POSE_EPS = 0.003;
const MAX_HELD_SECONDS = 2 * CP_TAU;

function displacementBound(a: number, mean: number, curve: number): number {
  return a * mean + 0.25 * curve * a * a;
}

export function unreachable(
  d: number,
  v0: number,
  v1: number,
  acc: number,
  aLo: number,
  aHi: number,
): boolean {
  const mean = 0.5 * (v0 + v1);
  const lo = Math.min(
    displacementBound(aLo, mean - acc, -acc),
    displacementBound(aHi, mean - acc, -acc),
  );
  const hi = Math.max(
    displacementBound(aLo, mean + acc, acc),
    displacementBound(aHi, mean + acc, acc),
  );
  return d < CP_TAU * lo - POSE_EPS || d > CP_TAU * hi + POSE_EPS;
}

function forceOf(state: FrameState): number {
  return cartPoleForce(state.pose[CARTPOLE_ACTION] ?? 0);
}

class BoardwalkScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private trolley: TrolleyHandle | null = null;
  private rig: OrbitRig | null = null;
  private pool: CelebrationPool | null = null;
  private overlay: GroundTruthOverlay | null = null;

  private xLimit = 2.4;
  private thLimit = 0.2094395;
  private armLength = 1;

  private fallAt = -Infinity;
  private fallDir = 1;

  private heldX = Number.NaN;
  private heldTh = 0;
  private heldXd = 0;
  private heldThd = 0;
  private heldStep = Number.NaN;
  private heldFrames = 0;
  private heldAt = -Infinity;

  private debugOn = false;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];
  private readonly tipScratch = new Float32Array(2);

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'boardwalkScene';

    const palette = ctx.tokens.palettes.noon;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.xLimit = trackLimit(ctx.geometry);
    this.thLimit = tiltLimit(ctx.geometry);
    this.armLength = poleLength(ctx.geometry);

    const world = buildWorld(ctx, this.scene);
    this.world = world;
    this.scene.add(world.group);

    const warm = world.rig.points[0];
    if (warm) warm.color.set(palette.accentLight);
    const cool = world.rig.points[1];
    if (cool) cool.color.set(ctx.tokens.failureBlue);

    const trolley = buildTrolley({
      palette,
      accent,
      seed: ctx.seed,
      poleLength: this.armLength,
      tiltLimit: this.thLimit,
    });
    this.trolley = trolley;
    this.scene.add(trolley.group);

    const home = orbitFromPose(
      CAM_BASE,
      CAM_LOOK,
    );
    const rig = makeOrbitRig({
      canvas: ctx.canvas,
      target: CAM_LOOK,
      azimuth: home.azimuth,
      polar: home.polar,
      distance: home.distance,
      fov: ctx.tokens.fovHero,
      near: 0.05,
      far: 420,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      damping: 2.4,
      name: 'boardwalkOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      seed: ctx.seed,
      sighSize: SIGH_PARTICLES,
      sighRadius: 0.34,
      sighHeight: 0.9,
      name: 'boardwalkCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentPos: (state) => {
        const t = cartPoleTip(state.pose[CARTPOLE_X] ?? 0, state.pose[CARTPOLE_TH] ?? 0, this.tipScratch);
        this.overlayPos[0] = t[0] ?? 0;
        this.overlayPos[1] = t[1] ?? 0;
        this.overlayPos[2] = 0;
        return this.overlayPos;
      },
      name: 'boardwalkGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    trolley.setPixelRatio(ctx.size.dpr);
    world.setPixelRatio(ctx.size.dpr);
    trolley.setShed(0, 1);
    trolley.setState(0, 0, 0, 0);
    world.setWind(0);
    world.setCool(0);
    world.setTime(0);
    trolley.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const trolley = this.trolley;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !trolley || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      this.fallAt = -Infinity;
      this.heldX = Number.NaN;
      this.heldFrames = 0;
      this.heldAt = -Infinity;
      rig.snap();
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'done') {
        pool.sigh(SIGH_STRENGTH, trolley.crownWorld, time);
        this.fallAt = time;
        this.fallDir = this.heldTh < 0 ? -1 : 1;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    const rawX = state.pose[CARTPOLE_X] ?? 0;
    const rawXd = state.pose[CARTPOLE_XDOT] ?? 0;
    const rawTh = state.pose[CARTPOLE_TH] ?? 0;
    const rawThd = state.pose[CARTPOLE_THDOT] ?? 0;
    let x = rawX;
    let th = rawTh;
    if (teleported || frame.scrubbed || !Number.isFinite(this.heldX)) {
      this.heldFrames = 0;
    } else {
      const drift = Math.abs(state.step - this.heldStep);
      const aLo = drift > 1 ? drift - 1 : 0;
      const aHi = drift + 1;
      const jumped =
        unreachable(rawTh - this.heldTh, this.heldThd, rawThd, TH_ACC, aLo, aHi) ||
        unreachable(rawX - this.heldX, this.heldXd, rawXd, X_ACC, aLo, aHi);
      if (jumped && (this.heldFrames === 0 || time - this.heldAt <= MAX_HELD_SECONDS)) {
        if (this.heldFrames === 0) this.heldAt = time;
        x = this.heldX;
        th = this.heldTh;
        this.heldFrames++;
      } else {
        this.heldFrames = 0;
      }
    }
    if (this.heldFrames === 0) {
      this.heldX = rawX;
      this.heldTh = rawTh;
      this.heldXd = rawXd;
      this.heldThd = rawThd;
      this.heldStep = state.step;
    }

    const fall = envelope(time - this.fallAt, FALL_SECONDS, 3.4);
    const cool = envelope(time - this.fallAt, COOL_SECONDS, 2.6);

    trolley.setShed(fall, this.fallDir);
    trolley.setState(x, th, rawXd, forceOf(state));
    world.setWind(clampRange(th / this.thLimit, -1, 1));
    world.setCool(cool);

    const safety =
      1 -
      clampRange(
        Math.max(Math.abs(th) / this.thLimit, Math.abs(x) / this.xLimit),
        0,
        1,
      );
    const warm = world.rig.points[0];
    if (warm) {
      warm.position.set(
        trolley.crownWorld.x - 0.25,
        trolley.crownWorld.y + 0.3,
        LIGHT_STANDOFF,
      );
      warm.intensity = (0.35 + 1.15 * safety * safety) * (1 - 0.75 * cool);
    }
    const coolLight = world.rig.points[1];
    if (coolLight) {
      coolLight.position.set(
        trolley.crownWorld.x + 0.3,
        trolley.crownWorld.y + 0.15,
        LIGHT_STANDOFF,
      );
      coolLight.intensity = 1.5 * cool;
    }

    world.setTime(time);
    trolley.setTime(time);

    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.trolley?.setPixelRatio(size.dpr);
    this.world?.setPixelRatio(size.dpr);
    this.pool?.setPixelRatio(size.dpr);
  }

  setDebug(on: boolean): void {
    this.debugOn = on;
    if (this.overlay) this.overlay.visible = on;
  }

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.trolley?.dispose();
    this.trolley = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.fallAt = -Infinity;
    this.heldX = Number.NaN;
    this.heldFrames = 0;
    this.heldAt = -Infinity;
  }
}

export function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

const createScene: SceneFactory = () => new BoardwalkScene();
export default createScene;
