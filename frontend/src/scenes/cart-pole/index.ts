import type { CelebrationPoolOptions } from '@kit/celebrate';
import type { GroundTruthOptions } from '@kit/debug';
import { clampRange, envelope } from '@kit/mix';
import { orbitFromPose, type OrbitRigOptions } from '@kit/rigs';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { CARTPOLE_ACTION, CARTPOLE_TH, CARTPOLE_X, CARTPOLE_XDOT, cartPoleForce, cartPoleTip } from '@sim/envs/cartpole';
import { buildTrolley, type TrolleyHandle } from './actors';
import { CAM_BASE, CAM_LOOK, buildWorld, poleLength, tiltLimit, trackLimit, type WorldHandle } from './world';

const FALL_SECONDS = 0.42;
const COOL_SECONDS = 0.58;
const SIGH_STRENGTH = 0.5;
const SIGH_PARTICLES = 34;
const LIGHT_STANDOFF = 1.15;

function forceOf(state: FrameState): number {
  return cartPoleForce(state.pose[CARTPOLE_ACTION] ?? 0);
}

class BoardwalkScene extends StageScene<WorldHandle> {
  protected readonly stageName = 'boardwalk';

  private trolley: TrolleyHandle | null = null;

  private xLimit = 2.4;
  private thLimit = 0.2094395;
  private armLength = 1;

  private fallAt = -Infinity;
  private fallDir = 1;

  private readonly overlayPos: [number, number, number] = [0, 0, 0];
  private readonly tipScratch = new Float32Array(2);

  protected buildWorld(ctx: SceneContext): WorldHandle {
    this.xLimit = trackLimit(ctx.geometry);
    this.thLimit = tiltLimit(ctx.geometry);
    this.armLength = poleLength(ctx.geometry);
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const palette = ctx.tokens.palettes.noon;
    const accent = this.accentOf(ctx);

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
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    const home = orbitFromPose(
      CAM_BASE,
      CAM_LOOK,
    );
    return {
      target: CAM_LOOK,
      azimuth: home.azimuth,
      polar: home.polar,
      distance: home.distance,
      near: 0.05,
      far: 420,
      rotateSpeed: 0.006,
      zoomSpeed: 0.0016,
      damping: 2.4,
    };
  }

  protected poolOptions(ctx: SceneContext): CelebrationPoolOptions {
    return {
      sighSize: SIGH_PARTICLES,
      sighRadius: 0.34,
      sighHeight: 0.9,
    };
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      agentPos: (state) => {
        const t = cartPoleTip(state.pose[CARTPOLE_X] ?? 0, state.pose[CARTPOLE_TH] ?? 0, this.tipScratch);
        this.overlayPos[0] = t[0] ?? 0;
        this.overlayPos[1] = t[1] ?? 0;
        this.overlayPos[2] = 0;
        return this.overlayPos;
      },
    };
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    const trolley = this.trolley;
    if (!trolley) return;
    trolley.setPixelRatio(ctx.size.dpr);
    trolley.setShed(0, 1);
    trolley.setState(0, 0, 0, 0);
    world.setWind(0);
    world.setCool(0);
    world.setTime(0);
    trolley.setTime(0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const trolley = this.trolley;
    if (!trolley) return;

    const state = frame.state;

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      this.fallAt = -Infinity;
      rig.snap();
    }

    const x = state.pose[CARTPOLE_X] ?? 0;
    const xd = state.pose[CARTPOLE_XDOT] ?? 0;
    const th = state.pose[CARTPOLE_TH] ?? 0;

    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'done') {
        pool.sigh(SIGH_STRENGTH, trolley.crownWorld, time);
        this.fallAt = time;
        this.fallDir = th < 0 ? -1 : 1;
      }
    }

    const fall = envelope(time - this.fallAt, FALL_SECONDS, 3.4);
    const cool = envelope(time - this.fallAt, COOL_SECONDS, 2.6);

    trolley.setShed(fall, this.fallDir);
    trolley.setState(x, th, xd, forceOf(state));
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
  }

  resize(size: SceneSize): void {
    super.resize(size);
    this.trolley?.setPixelRatio(size.dpr);
  }

  protected drop(): void {
    this.trolley?.dispose();
    this.trolley = null;
    this.fallAt = -Infinity;
  }
}

export default ((() => new BoardwalkScene()) satisfies SceneFactory);
