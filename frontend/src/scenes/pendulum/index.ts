
import { GroundTruthOverlay } from '@kit/debug';
import { makeOrbitRig, orbitFromPose, type OrbitRig } from '@kit/rigs';
import { CelebrationPool, attachCelebrations, disposeCelebrations } from '@kit/celebrate';
import type {
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import * as THREE from 'three';
import { PD_DT, PD_G, PD_L, PD_M, PD_MAX_SPEED, PD_MAX_TORQUE, PENDULUM_TH, PENDULUM_THDOT, PENDULUM_TORQUE, wrapPi } from '@sim/envs/pendulum';
import { buildLanternRig, type LanternRigHandle } from './actors';
import { manifest } from './manifest';
import {
  buildWorld,
  clampRange,
  pivotOf,
  type WorldHandle,
} from './world';


const CAM_BASE: readonly [number, number, number] = [0.62, 0.41, 6.55];
const CAM_LOOK: readonly [number, number, number] = [0, -0.09, 0];


export const PD_MAX_DTHETADOT = ((1.5 * PD_G) / PD_L + (3 * PD_MAX_TORQUE) / (PD_M * PD_L * PD_L)) * PD_DT;
const WRAP_SLACK = 0.02;
export function wrapReach(thetaDot: number, steps: number): number {
  const ramp = PD_MAX_DTHETADOT * ((steps * (steps + 1)) / 2 + steps);
  const travel = Math.min(
    steps * Math.abs(thetaDot) + ramp,
    steps * (PD_MAX_SPEED + PD_MAX_DTHETADOT),
  );
  return travel * PD_DT + WRAP_SLACK;
}

function torqueOf(state: FrameState): number {
  return state.pose[PENDULUM_TORQUE] ?? 0;
}

export function pendulumCost(theta: number, thetaDot: number, torque: number): number {
  const th = wrapPi(theta);
  return th * th + 0.1 * thetaDot * thetaDot + 0.001 * torque * torque;
}

export const COST_MAX = pendulumCost(Math.PI, PD_MAX_SPEED, PD_MAX_TORQUE);

const SERENITY_GAMMA = 3;

export function serenityOf(theta: number, thetaDot: number, torque = 0): number {
  const reward = 1 - pendulumCost(theta, thetaDot, torque) / COST_MAX;
  return clampRange(reward, 0, 1) ** SERENITY_GAMMA;
}

class LanternBellScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private lantern: LanternRigHandle | null = null;
  private rig: OrbitRig | null = null;
  private pool: CelebrationPool | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private pivot: readonly [number, number, number] = [0, 0, 0];

  private anchorTheta = Number.NaN;
  private anchorStep = 0;
  private prevRaw = Number.NaN;
  private prevStep = 0;
  private holdStep = Number.NaN;
  wrapHeld = false;

  private debugOn = false;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'lanternBellScene';
    this.pivot = pivotOf(ctx.geometry);

    const palette = ctx.tokens.palettes.night;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene);
    this.world = world;
    this.scene.add(world.group);

    const lantern = buildLanternRig({
      palette,
      accent,
      pivot: this.pivot,
      maxTorque: PD_MAX_TORQUE,
      maxSpeed: PD_MAX_SPEED,
    });
    this.lantern = lantern;
    this.scene.add(lantern.group);

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
      damping: 2.6,
      name: 'lanternOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      seed: ctx.seed,
      name: 'lanternBellCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      agentPos: (state) => {
        const th = state.pose[PENDULUM_TH] ?? 0;
        this.overlayPos[0] = this.pivot[0] + PD_L * Math.sin(th);
        this.overlayPos[1] = this.pivot[1] - PD_L * Math.cos(th);
        this.overlayPos[2] = this.pivot[2];
        return this.overlayPos;
      },
      name: 'lanternBellGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    world.setPixelRatio(ctx.size.dpr);
    this.applyState(Math.PI, 0, 0, 0);
    world.setTime(0);
    lantern.setTime(0);
  }

  private applyState(theta: number, thetaDot: number, torque: number, time: number): void {
    const world = this.world;
    const lantern = this.lantern;
    if (!world || !lantern) return;

    lantern.setState(theta, thetaDot, torque);
    const serenity = serenityOf(theta, thetaDot, torque);
    lantern.setGlow(serenity);
    world.setLight(serenity, lantern.bobRoot.position.x, time);

    const flame = lantern.flameWorld;
    world.motes.position.copy(flame);
    const light = world.lanternLight;
    if (light) light.position.copy(flame);
  }

  private opposes(delta: number, thetaDot: number, steps: number): boolean {
    if (!Number.isFinite(delta)) return false;
    return Math.abs(thetaDot) > PD_MAX_DTHETADOT * steps && delta * thetaDot < 0;
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const lantern = this.lantern;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !lantern || !pool || !rig) return;

    const time = ctx.time();
    const state = frame.state;

    pool.update(frame.dtReal, time);
    if (frame.scrubbed) {
      pool.clear();
      rig.snap();
    }


    const rawTheta = state.pose[PENDULUM_TH] ?? 0;
    const thetaDot = state.pose[PENDULUM_THDOT] ?? 0;
    let theta = rawTheta;
    this.wrapHeld = false;
    if (frame.scrubbed || !Number.isFinite(this.anchorTheta)) {
      this.anchorTheta = rawTheta;
      this.anchorStep = state.step;
      this.holdStep = Number.NaN;
    } else {
      const steps = Math.max(0, state.step - this.anchorStep) + 1;
      const delta = wrapPi(rawTheta - this.anchorTheta);
      const unreachable = Math.abs(delta) > wrapReach(thetaDot, steps);
      const backwards =
        this.opposes(delta, thetaDot, steps) ||
        this.opposes(
          wrapPi(rawTheta - this.prevRaw),
          thetaDot,
          Math.max(0, state.step - this.prevStep) + 1,
        );
      const mayHold = !Number.isFinite(this.holdStep) || this.holdStep === state.step;
      if ((unreachable || backwards) && mayHold) {
        theta = this.anchorTheta;
        this.wrapHeld = true;
        this.holdStep = state.step;
      } else {
        this.holdStep = Number.NaN;
        if (state.step !== this.anchorStep) {
          this.anchorTheta = rawTheta;
          this.anchorStep = state.step;
        }
      }
    }
    this.prevRaw = rawTheta;
    this.prevStep = state.step;

    this.applyState(theta, thetaDot, torqueOf(state), time);

    world.setTime(time);
    lantern.setTime(time);

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

  dispose(): void {
    this.overlay?.dispose();
    this.overlay = null;
    this.lantern?.dispose();
    this.lantern = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.anchorTheta = Number.NaN;
    this.anchorStep = 0;
    this.prevRaw = Number.NaN;
    this.prevStep = 0;
    this.holdStep = Number.NaN;
    this.wrapHeld = false;
  }
}

const createScene: SceneFactory = () => new LanternBellScene();
export default createScene;
