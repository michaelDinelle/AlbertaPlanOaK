import type { GroundTruthOptions } from '@kit/debug';
import { envelope } from '@kit/mix';
import { type OrbitRigOptions } from '@kit/rigs';
import { makeSceneView, type SceneView } from '../jelly-bean-world/view';
import { StageScene, type Stage } from '@kit/stage';
import type { FrameState, FrameUpdate, SceneContext, SceneFactory, SceneSize } from '@kit/types';
import { MC_GOAL_POS, MC_START_POS, MOUNTAINCAR_ACTION, MOUNTAINCAR_POS, MOUNTAINCAR_VEL, mountainCarForce, mountainCarSlope } from '@sim/envs/mountaincar';
import { buildCart, type CartHandle } from './actors';
import { buildWorld, heightResolver, type WorldHandle } from './world';


const GUST_SECONDS = 3.2;
const EAGLE_SECONDS = 6.5;
const FLASH_SECONDS = 0.24;
const SUMMIT_FLARE_SECONDS = 1.3;
const SCREE_SECONDS = 0.85;
const BURST_SCALE = 0.38;
const RIPPLE_SCALE = 0.22;
const PULSE_STRENGTH = 0.6;

function forceOf(state: FrameState): number {
  return mountainCarForce(state.pose[MOUNTAINCAR_ACTION] ?? 0);
}

class AlpineValleyScene extends StageScene<WorldHandle> {
  private view: SceneView | null = null;
  protected readonly stageName = 'alpineValley';

  private cart: CartHandle | null = null;
  private heightAt: (x: number) => number = () => 0;
  private accent: number = 0xe86e4d;
  private gold: number = 0xffd166;

  private gustAt = -Infinity;
  private eagleAt = -Infinity;
  private flashAt = -Infinity;
  private flareAt = -Infinity;
  private screeAt = -Infinity;
  private screeStep = Number.NaN;

  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  protected buildWorld(ctx: SceneContext): WorldHandle {
    this.heightAt = heightResolver(ctx.geometry);
    return buildWorld(ctx, this.scene);
  }

  protected buildActors(ctx: SceneContext, world: WorldHandle): void {
    const palette = ctx.tokens.palettes.dusk;
    const accent = this.accentOf(ctx);
    this.accent = accent;
    this.gold = ctx.tokens.celebrationGold;

    const lanternLight = world.rig.points[0];
    if (lanternLight) lanternLight.color.set(palette.accentLight);
    const summitLight = world.rig.points[1];
    if (summitLight) {
      summitLight.color.set(this.gold);
      summitLight.position.set(MC_GOAL_POS, world.cairnRoot.position.y + 0.28, 0);
    }

    const cart = buildCart({
      palette,
      accent,
      seed: ctx.seed,
      heightAt: this.heightAt,
      slopeAt: mountainCarSlope,
    });
    this.cart = cart;
    this.scene.add(cart.group);
  }

  protected rigOptions(ctx: SceneContext): OrbitRigOptions {
    this.view = makeSceneView(ctx, { min: [-1.35, 0, -0.4], max: [0.75, 1.55, 0.4] }, [-0.31, 0.75, -0.05], { distance: 2.632622267, azimuth: 0.057408115, polar: 1.360000000 });
    return this.view.options;
  }

  protected overlayOptions(): GroundTruthOptions {
    return {
      profileRange: [-1.2, 0.6],
      agentPos: (state) => {
        const x = state.pose[MOUNTAINCAR_POS] ?? 0;
        this.overlayPos[0] = x;
        this.overlayPos[1] = this.heightAt(x);
        this.overlayPos[2] = 0;
        return this.overlayPos;
      },
    };
  }

  protected start(ctx: SceneContext, world: WorldHandle): void {
    const cart = this.cart;
    if (!cart) return;
    cart.setPixelRatio(ctx.size.dpr);
    world.setGust(0);
    world.setEagle(null);
    cart.setLanternGain(1);
    cart.setState(MC_START_POS, 0, 0);
    world.setTime(0);
    cart.setTime(0);
  }

  protected tick(frame: FrameUpdate, { world, pool, rig, time }: Stage<WorldHandle>): void {
    const cart = this.cart;
    if (!cart) return;

    const state = frame.state;

    pool.update(frame.dtReal, time);

    if (frame.scrubbed) {
      pool.clear();
      this.gustAt = -Infinity;
      this.eagleAt = -Infinity;
      this.flashAt = -Infinity;
      this.flareAt = -Infinity;
      this.screeAt = -Infinity;
      this.screeStep = Number.NaN;
    }

    let teleported = false;
    for (let i = 0; i < frame.events.length; i++) {
      const ev = frame.events[i];
      if (!ev) continue;
      if (ev.type === 'goal') {
        pool.burst(world.summitBurst, this.accent, BURST_SCALE, time);
        pool.ripple(world.cairnRoot.position, this.gold, RIPPLE_SCALE, time);
        pool.pulse.fire(PULSE_STRENGTH);
        this.gustAt = time;
        this.eagleAt = time;
        this.flareAt = time;
      } else if (ev.type === 'teleport') {
        teleported = true;
      }
    }

    const pos = state.pose[MOUNTAINCAR_POS] ?? 0;
    const vel = state.pose[MOUNTAINCAR_VEL] ?? 0;
    if (teleported) this.flashAt = time;

    const prev = frame.prev;
    if (!frame.scrubbed && !teleported && prev) {
      const prevVel = prev.pose[1] ?? 0;
      const turned = prevVel > 0 !== vel > 0 && Math.abs(prevVel) > 0.004;
      if (turned && this.heightAt(pos) < 0.82 && this.screeStep !== state.step) {
        this.screeAt = time;
        this.screeStep = state.step;
      }
    }

    const gust = envelope(time - this.gustAt, GUST_SECONDS, 1.7);
    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 1);
    const flare = envelope(time - this.flareAt, SUMMIT_FLARE_SECONDS, 2.2);
    const scree = envelope(time - this.screeAt, SCREE_SECONDS, 2.6);
    const eagleAge = time - this.eagleAt;
    const eagle = eagleAge >= 0 && eagleAge <= EAGLE_SECONDS ? eagleAge / EAGLE_SECONDS : null;

    cart.setDustKick(scree);
    cart.setState(pos, vel, forceOf(state));
    cart.setLanternGain(1 + 1.4 * flash);
    world.setGust(gust);
    world.setEagle(eagle);

    const lantern = world.rig.points[0];
    if (lantern) {
      lantern.position.copy(cart.lanternWorld);
      lantern.intensity = 0.07 + 0.5 * flash;
    }
    const summit = world.rig.points[1];
    if (summit) {
      summit.intensity = 0.04 + 0.7 * flare;
    }

    world.setTime(time);
    cart.setTime(time);

    if (this.view) world.setView(this.view.update(rig, frame.dtReal), this.camera);
  }

  resize(size: SceneSize): void {
    super.resize(size);
    this.cart?.setPixelRatio(size.dpr);
  }

  protected drop(): void {
    this.cart?.dispose();
    this.cart = null;
    this.gustAt = -Infinity;
    this.eagleAt = -Infinity;
    this.flashAt = -Infinity;
    this.flareAt = -Infinity;
    this.screeAt = -Infinity;
  }
}

export default ((() => new AlpineValleyScene()) satisfies SceneFactory);
