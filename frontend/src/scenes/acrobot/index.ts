import type { GroundTruthOptions } from '@kit/debug';
import { clampRange, envelope } from '@kit/mix';
import { type OrbitRigOptions } from '@kit/rigs';
import { makeSceneView, type SceneView } from '../jelly-bean-world/view';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import * as THREE from 'three';
import { AB_L1, AB_L2, ACROBOT_ACTION, ACROBOT_TH1, ACROBOT_TH2, ACROBOT_W1, ACROBOT_W2, acrobotTorque } from '@sim/envs/acrobot';
import { buildArm, type ArmHandle } from './actors';
import {
  EXACT_POINT_FLOATS,
  acrobotPointsExact,
  buildWorld,
  pivotAnchor,
  successAnchor,
  type WorldHandle,
} from './world';


const FLARE_SECONDS = 1.9;
const PETAL_SECONDS = 5;
const FLASH_SECONDS = 0.3;
const DAWN_PERIOD = 96;

const BURST_SCALE = 0.5;
const RIPPLE_SCALE = 0.34;
const PULSE_STRENGTH = 0.42;

const TAU = Math.PI * 2;

export function torqueOf(state: FrameState): number {
  return acrobotTorque(state.pose[ACROBOT_ACTION] ?? 0);
}

class WillowSwingScene extends StageScene<WorldHandle> {
  private view: SceneView | null = null;
  protected readonly stageName = 'willowSwing';

  private arm: ArmHandle | null = null;
  private accent = 0xf2a7c3;
  private gold = 0xffd166;
  private pivot: readonly [number, number, number] = [0, 0, 0];
  private success: readonly [number, number, number] = [0, 1, 0];

  private flareAt = -Infinity;
  private petalAt = -Infinity;
  private flashAt = -Infinity;

  private readonly points = new Float64Array(EXACT_POINT_FLOATS);
  private readonly overlayPos: [number, number, number] = [0, 0, 0];
  private readonly ripplePos = new THREE.Vector3();
  private readonly celebrateAt = new THREE.Vector3();

  protected buildWorld(ctx: SceneContext): WorldHandle {
    this.accent = this.accentOf(ctx);
    this.gold = ctx.tokens.celebrationGold;
    this.pivot = pivotAnchor(ctx.geometry);
    this.success = successAnchor(ctx.geometry);
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const palette = ctx.tokens.palettes.dawn;
    const success = this.success;

    const arm = buildArm({
      palette,
      accent: this.accent,
      gold: this.gold,
      seed: ctx.seed,
      pivot: this.pivot,
    });
    this.arm = arm;
    this.scene.add(arm.group);

    const lantern = world.rig.points[0];
    if (lantern) lantern.color.set(palette.accentLight);
    const garlandLight = world.rig.points[1];
    if (garlandLight) {
      garlandLight.color.set(this.gold);
      garlandLight.position.set(success[0], success[1] + 0.16, success[2]);
    }
    const celebration = world.rig.points[2];
    if (celebration) celebration.color.set(this.accent);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-2.35, -2.5, -0.45], max: [2.35, 2.1, 0.45] }, [-0.06, -0.5, 0], { distance: 5.741707063, azimuth: 0.111812400, polar: 1.360000000 });
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentPos: (state) => {
        acrobotPointsExact(state.pose[ACROBOT_TH1] ?? 0, state.pose[ACROBOT_TH2] ?? 0, this.points);
        this.overlayPos[0] = this.pivot[0] + (this.points[2] ?? 0);
        this.overlayPos[1] = this.pivot[1] + (this.points[3] ?? 0);
        this.overlayPos[2] = this.pivot[2];
        return this.overlayPos;
      },
    };
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    const arm = this.arm;
    if (!arm) return;
    arm.setPixelRatio(ctx.size.dpr);
    world.setDawn(0.45);
    world.setFlare(-1);
    world.setPetals(0);
    world.setGlint(0, 0, 0);
    arm.setLanternGain(1);
    arm.setState(0, 0, 0, 0);
    arm.setTime(0);
    world.setTime(0);
    this.celebrateAt.copy(arm.tipWorld);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const arm = this.arm;
    if (!arm) return;

    const state = frame.state;

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      this.flareAt = -Infinity;
      this.petalAt = -Infinity;
      this.flashAt = -Infinity;
      arm.clearTrail();
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        pool.burst(this.celebrateAt, this.accent, BURST_SCALE, time);
        this.ripplePos.set(this.celebrateAt.x, world.waterY, this.celebrateAt.z);
        pool.ripple(this.ripplePos, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        this.flareAt = time;
        this.petalAt = time;
        this.flashAt = time;
        const light = world.rig.points[2];
        if (light) light.position.copy(this.celebrateAt);
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    const th1 = state.pose[ACROBOT_TH1] ?? 0;
    const th2 = state.pose[ACROBOT_TH2] ?? 0;
    const w1 = state.pose[ACROBOT_W1] ?? 0;
    const w2 = state.pose[ACROBOT_W2] ?? 0;
    if (teleported) arm.clearTrail();

    const flareAge = time - this.flareAt;
    const flare = flareAge >= 0 && flareAge <= FLARE_SECONDS ? flareAge / FLARE_SECONDS : -1;
    const petals = envelope(time - this.petalAt, PETAL_SECONDS, 0.9);
    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 2.2);

    const tipSpeed = acrobotTipSpeed(th1, th2, w1, w2);
    arm.setState(th1, th2, torqueOf(state), tipSpeed);
    arm.setLanternGain(1 + 2.4 * flash);
    if (!frame.scrubbed) arm.pushTrail(time);
    arm.refreshTrail(time);

    const height01 = clampRange((arm.tipWorld.y - world.waterY) / 3.2, 0, 1);
    const glint = (1 - height01) * (1 - height01) * (0.55 + 0.45 * flash);
    world.setGlint(arm.tipWorld.x, arm.tipWorld.z, glint);

    world.setFlare(flare);
    world.setPetals(petals);
    world.setDawn(0.45 + 0.4 * Math.sin((time * TAU) / DAWN_PERIOD));

    const lantern = world.rig.points[0];
    if (lantern) {
      lantern.position.copy(arm.lanternWorld);
      lantern.intensity = 0.5 + 2.6 * flash;
    }
    const garlandLight = world.rig.points[1];
    if (garlandLight) {
      garlandLight.intensity = 0.05 + 1.5 * (flare >= 0 ? 1 - flare : 0);
    }
    const celebration = world.rig.points[2];
    if (celebration) {
      celebration.intensity = 2.4 * envelope(time - this.flareAt, FLARE_SECONDS, 2.4);
    }

    world.setTime(time);
    arm.setTime(time);

    if (this.view) world.setView(this.view.update(rig, frame.dtReal), this.camera);

    this.celebrateAt.copy(arm.tipWorld);
  }

  resize(size: SceneSize): void {
    super.resize(size);
    this.arm?.setPixelRatio(size.dpr);
  }

  protected drop(): void {
    this.arm?.dispose();
    this.arm = null;
    this.flareAt = -Infinity;
    this.petalAt = -Infinity;
    this.flashAt = -Infinity;
  }
}

export function acrobotTipSpeed(th1: number, th2: number, w1: number, w2: number): number {
  const a = th1;
  const b = th1 + th2;
  const wb = w1 + w2;
  const vx = w1 * Math.cos(a) * AB_L1 + wb * Math.cos(b) * AB_L2;
  const vy = w1 * Math.sin(a) * AB_L1 + wb * Math.sin(b) * AB_L2;
  return Math.hypot(vx, vy);
}

export default ((() => new WillowSwingScene()) satisfies SceneFactory);
