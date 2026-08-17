
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
import { MC_GOAL_POS, MC_START_POS, MOUNTAINCAR_ACTION, MOUNTAINCAR_POS, MOUNTAINCAR_VEL, mountainCarForce, mountainCarSlope } from '@sim/envs/mountaincar';
import { buildCart, type CartHandle } from './actors';
import { manifest } from './manifest';
import { buildWorld, heightResolver, type WorldHandle } from './world';


const CAM_BASE: readonly [number, number, number] = [-0.16, 1.06, 2.56];
const CAM_LOOK: readonly [number, number, number] = [-0.31, 0.75, -0.05];

const GUST_SECONDS = 3.2;
const EAGLE_SECONDS = 6.5;
const FLASH_SECONDS = 0.24;
const SUMMIT_FLARE_SECONDS = 1.3;
const SCREE_SECONDS = 0.85;
const BURST_SCALE = 0.38;
const RIPPLE_SCALE = 0.22;
const BURST_PARTICLES = 84;
const PULSE_STRENGTH = 0.6;

function forceOf(state: FrameState): number {
  return mountainCarForce(state.pose[MOUNTAINCAR_ACTION] ?? 0);
}

class AlpineValleyScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private cart: CartHandle | null = null;
  private rig: OrbitRig | null = null;
  private pool: CelebrationPool | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private heightAt: (x: number) => number = () => 0;
  private accent: number = 0xe86e4d;
  private gold: number = 0xffd166;

  private gustAt = -Infinity;
  private eagleAt = -Infinity;
  private flashAt = -Infinity;
  private flareAt = -Infinity;
  private screeAt = -Infinity;
  private screeStep = Number.NaN;

  private debugOn = false;
  private readonly overlayPos: [number, number, number] = [0, 0, 0];

  async init(ctx: SceneContext): Promise<void> {
    this.ctx = ctx;
    this.scene.name = 'alpineValleyScene';
    this.heightAt = heightResolver(ctx.geometry);

    const palette = ctx.tokens.palettes.dusk;
    const accent = ctx.tokens.accents[manifest.accent] ?? ctx.tokens.celebrationGold;
    this.accent = accent;
    this.gold = ctx.tokens.celebrationGold;

    const world = buildWorld(ctx, this.scene);
    this.world = world;
    this.scene.add(world.group);

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
      name: 'alpineOrbitRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    const pool = new CelebrationPool({
      seed: ctx.seed,
      burstSize: BURST_PARTICLES,
      name: 'alpineCelebrations',
    });
    attachCelebrations(this.scene, pool);
    pool.setPixelRatio(ctx.size.dpr);
    this.pool = pool;

    this.overlay = new GroundTruthOverlay(ctx.geometry, {
      camera: this.camera,
      profileRange: [-1.2, 0.6],
      agentPos: (state) => {
        const x = state.pose[MOUNTAINCAR_POS] ?? 0;
        this.overlayPos[0] = x;
        this.overlayPos[1] = this.heightAt(x);
        this.overlayPos[2] = 0;
        return this.overlayPos;
      },
      name: 'alpineGroundTruth',
    });
    this.scene.add(this.overlay.group);
    this.overlay.visible = this.debugOn;

    cart.setPixelRatio(ctx.size.dpr);
    world.setPixelRatio(ctx.size.dpr);
    world.setGust(0);
    world.setEagle(null);
    cart.setLanternGain(1);
    cart.setState(MC_START_POS, 0, 0);
    world.setTime(0);
    cart.setTime(0);
  }

  update(frame: FrameUpdate): void {
    const ctx = this.ctx;
    const world = this.world;
    const cart = this.cart;
    const pool = this.pool;
    const rig = this.rig;
    if (!ctx || !world || !cart || !pool || !rig) return;

    const time = ctx.time();
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
      rig.snap();
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

    rig.update(frame.dtReal);

    if (this.debugOn && this.overlay) this.overlay.setState(state);
  }

  resize(size: SceneSize): void {
    const aspect = size.height > 0 ? size.width / size.height : 16 / 9;
    this.rig?.resize(aspect);
    this.cart?.setPixelRatio(size.dpr);
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
    this.cart?.dispose();
    this.cart = null;
    this.world?.dispose();
    this.world = null;
    this.rig?.dispose();
    this.rig = null;
    disposeCelebrations(this.scene);
    this.pool = null;
    this.scene.clear();
    this.scene.fog = null;
    this.ctx = null;
    this.gustAt = -Infinity;
    this.eagleAt = -Infinity;
    this.flashAt = -Infinity;
    this.flareAt = -Infinity;
    this.screeAt = -Infinity;
  }
}

function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

const createScene: SceneFactory = () => new AlpineValleyScene();
export default createScene;
