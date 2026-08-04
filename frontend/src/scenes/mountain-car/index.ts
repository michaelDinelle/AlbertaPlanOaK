// Alpine Valley — MountainCar-cont-v0. The visual canon for the OaK Preserve.
//
// THE ONE IDEA: the ground IS the physics. `mountainCarHeight(x) = sin(3x)·0.45+0.55`
// over x ∈ [-1.2, 0.6] is not "inspiration for" the terrain — it is the terrain, at 1:1
// world scale, and the cart sits on it at (pose[0], heightAt(pose[0]), 0) with the tilt
// read off the analytic slope. Every other element is placed from an env anchor: the
// cairn on the goal line at 0.5, the signpost on the respawn at -0.5.
//
// PATTERNS THE OTHER FIFTEEN SCENES SHOULD COPY
//  * probe roots carry POSITION ONLY. Tilt lives on `cartTilt`, lift lives in the part
//    geometry. Deleting every decoration would leave a correct scene.
//  * state visuals are PURE FUNCTIONS of the frame. The momentum trail is the terrain
//    curve traced backwards over `|vel|` world units, not a history buffer, so it is
//    idempotent, scrub-safe and free on a teleport.
//  * every non-state animation is keyed off `ctx.time()` — including the goal gust, the
//    eagle crossing and the teleport flash — so a paused playback freezes the whole
//    world coherently, celebrations included.
//  * celebrations fire ONLY from `frame.events`. `done` here always accompanies `goal`
//    (MountainCar's done IS the +1), so this scene never sighs — that would be a lie.
//  * the SUPERLUMINAL GUARD (see `update`): the host linearly interpolates continuous
//    poses, so the single control step that contains a respawn renders the cart sliding
//    from the summit back down the map. Holding the last physically-reachable pose for
//    at most a few frames turns that ease into the hard cut the contract demands.

import { GroundTruthOverlay } from '@kit/debug';
import { makeFrameRig, type FrameRig } from '@kit/rigs';
import {
  CelebrationPool,
  attachCelebrations,
  disposeCelebrations,
} from '@kit/celebrate';
import type {
  AlignmentProbe,
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneFactory,
  SceneModule,
  SceneSize,
} from '@kit/types';
import * as THREE from 'three';
import {
  MC_GOAL_POS,
  MC_MAX_SPEED,
  MC_START_POS,
  mountainCarSlope,
} from '@sim/envs/mountaincar';
import { buildCart, type CartHandle } from './actors';
import { manifest } from './manifest';
import { buildWorld, heightResolver, TRAILHEAD_Z, type WorldHandle } from './world';

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

// THE FRAME. Near-profile and close: the whole point of this scene is that the ground is
// the physics, and a 3/4 view turns `sin(3x)` into "some mountains". The camera sits at
// the track's own centre (x = -0.3), 2.1 units out, high enough to see the trough floor —
// at 16:9 that frames x ∈ [-1.6, 1.0] and y ∈ [0.2, 1.7], i.e. the left wall, the whole
// basin and the summit cairn, with a few degrees of offset for depth.
const CAM_BASE: readonly [number, number, number] = [-0.16, 1.06, 2.56];
const CAM_LOOK: readonly [number, number, number] = [-0.31, 0.75, -0.05];
/** thumb pulls back so the whole 1.8-unit playfield survives a square-ish crop */
const CAM_BASE_THUMB: readonly [number, number, number] = [-0.28, 1.1, 3.05];
const CAM_LOOK_THUMB: readonly [number, number, number] = [-0.3, 0.72, 0];

// celebration/flourish envelopes, all measured on ctx.time()
const GUST_SECONDS = 3.2;
const EAGLE_SECONDS = 6.5;
const FLASH_SECONDS = 0.24;
const SUMMIT_FLARE_SECONDS = 1.3;
const SCREE_SECONDS = 0.85;
// Celebration weights, tuned against a REAL goal (seed 40, step 44984). Everything in
// this scene is small and the snow is already near the bloom threshold, so the default
// full-strength grammar whites the frame out: a 2.4-intensity point light on a snow
// crest plus a 1.0 bloom pulse smears the whole ridge. Half the amplitude, all of the
// beat.
/** the burst travels ~0.3 world units in a 1.8-unit playfield */
const BURST_SCALE = 0.38;
const RIPPLE_SCALE = 0.22;
const BURST_PARTICLES = 84;
const PULSE_STRENGTH = 0.6;

/** a control step can move the cart at most MC_MAX_SPEED; anything more is a respawn */
const SUPERLUMINAL_SLACK = 0.025;
const MAX_HELD_FRAMES = 4;

interface MountainCarLike {
  pos?: unknown;
  vel?: unknown;
  force?: unknown;
}

/** force ∈ {-1, 0, +1}: sim-core's decode when present, the raw ctrl otherwise. */
function forceOf(state: FrameState): number {
  const d = state.derived as MountainCarLike | null | undefined;
  if (d && typeof d === 'object' && typeof d.force === 'number') return d.force;
  const raw = state.ctrl[0] ?? 0;
  if (!Number.isFinite(raw)) return 0;
  return ((Math.trunc(raw) >>> 0) % 3) - 1;
}

class AlpineValleyScene implements SceneModule {
  readonly manifest = manifest;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 500);

  private ctx: SceneContext | null = null;
  private world: WorldHandle | null = null;
  private cart: CartHandle | null = null;
  private rig: FrameRig | null = null;
  private pool: CelebrationPool | null = null;
  private overlay: GroundTruthOverlay | null = null;
  private heightAt: (x: number) => number = () => 0;
  private accent: number = 0xe86e4d;
  private gold: number = 0xffd166;

  // ---- flourish clocks (playback seconds; -Infinity = idle) ----
  private gustAt = -Infinity;
  private eagleAt = -Infinity;
  private flashAt = -Infinity;
  private flareAt = -Infinity;
  private screeAt = -Infinity;
  private screeStep = Number.NaN;

  // ---- the superluminal guard ----
  private heldPos = Number.NaN;
  private heldStep = Number.NaN;
  private heldFrames = 0;

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

    // This scene claims exactly two of the rig's point lights, and their colour and
    // (for the summit) position are STATIC — only intensity is state-driven, so update()
    // never touches a Color. points[] is empty at thumb; the guards are the tier policy.
    const lanternLight = world.rig.points[0];
    if (lanternLight) lanternLight.color.set(palette.accentLight);
    const summitLight = world.rig.points[1];
    if (summitLight) {
      summitLight.color.set(this.gold);
      summitLight.position.set(MC_GOAL_POS, world.cairnRoot.position.y + 0.28, 0);
    }

    const cart = buildCart({
      tier: ctx.tier,
      palette,
      accent,
      seed: ctx.seed,
      heightAt: this.heightAt,
      slopeAt: mountainCarSlope,
    });
    this.cart = cart;
    this.scene.add(cart.group);

    const thumb = ctx.tier === 'thumb';
    const rig = makeFrameRig({
      position: thumb ? CAM_BASE_THUMB : CAM_BASE,
      lookAt: thumb ? CAM_LOOK_THUMB : CAM_LOOK,
      fov: ctx.tokens.fovHero,
      near: 0.05,
      far: 400,
      aspect: ctx.size.height > 0 ? ctx.size.width / ctx.size.height : 16 / 9,
      canvas: thumb ? null : ctx.canvas,
      parallax: 0.06,
      parallaxY: 0.45,
      damping: 2.6,
      name: 'alpineFrameRig',
    });
    this.rig = rig;
    this.camera = rig.camera;
    rig.snap();

    // the host reads this pool's own channel through celebrationPulse(scene)
    const pool = new CelebrationPool({
      tier: ctx.tier,
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
        const x = state.pose[0] ?? 0;
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

    // 1. the pool adopts the host clock BEFORE anything fires into it
    pool.update(frame.dtReal, time);

    // 2. a scrub wipes every in-flight beat: nothing may survive a discontinuity
    if (frame.scrubbed) {
      pool.clear();
      this.gustAt = -Infinity;
      this.eagleAt = -Infinity;
      this.flashAt = -Infinity;
      this.flareAt = -Infinity;
      this.screeAt = -Infinity;
      this.screeStep = Number.NaN;
      this.heldPos = Number.NaN;
      this.heldFrames = 0;
      rig.snap();
    }

    // 3. events — celebrations are NEVER inferred from the state stream
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
      // `done` is MountainCar's +1, never a failure: this scene must not sigh.
    }

    // 4. the superluminal guard (see the header note)
    const rawPos = state.pose[0] ?? 0;
    const vel = state.pose[1] ?? 0;
    let pos = rawPos;
    if (teleported || frame.scrubbed || !Number.isFinite(this.heldPos)) {
      this.heldFrames = 0;
    } else {
      const dSteps = Math.max(1, Math.abs(state.step - this.heldStep));
      const reach = MC_MAX_SPEED * dSteps + SUPERLUMINAL_SLACK;
      if (Math.abs(rawPos - this.heldPos) > reach && this.heldFrames < MAX_HELD_FRAMES) {
        pos = this.heldPos;
        this.heldFrames++;
      } else {
        this.heldFrames = 0;
      }
    }
    // a teleport gets a one-frame warm flash, but NEVER a pool.clear(): the goal burst
    // fired in the same event batch and must survive the snap it caused.
    if (teleported) this.flashAt = time;
    this.heldPos = pos;
    this.heldStep = state.step;

    // 5. a failed climb: the velocity turns over below the summit and kicks scree
    const prev = frame.prev;
    if (!frame.scrubbed && !teleported && prev) {
      const prevVel = prev.pose[1] ?? 0;
      const turned = prevVel > 0 !== vel > 0 && Math.abs(prevVel) > 0.004;
      if (turned && this.heightAt(pos) < 0.82 && this.screeStep !== state.step) {
        this.screeAt = time;
        this.screeStep = state.step;
      }
    }

    // 6. envelopes — every one of them measured on the playback clock
    const gust = envelope(time - this.gustAt, GUST_SECONDS, 1.7);
    const flash = envelope(time - this.flashAt, FLASH_SECONDS, 1);
    const flare = envelope(time - this.flareAt, SUMMIT_FLARE_SECONDS, 2.2);
    const scree = envelope(time - this.screeAt, SCREE_SECONDS, 2.6);
    const eagleAge = time - this.eagleAt;
    const eagle = eagleAge >= 0 && eagleAge <= EAGLE_SECONDS ? eagleAge / EAGLE_SECONDS : null;

    // 7. the state visuals
    cart.setDustKick(scree);
    cart.setState(pos, vel, forceOf(state));
    cart.setLanternGain(1 + 1.4 * flash);
    world.setGust(gust);
    world.setEagle(eagle);

    // 8. the two point lights this scene claims from the rig
    const lantern = world.rig.points[0];
    if (lantern) {
      lantern.position.copy(cart.lanternWorld);
      lantern.intensity = 0.07 + 0.5 * flash;
    }
    const summit = world.rig.points[1];
    if (summit) {
      summit.intensity = 0.04 + 0.7 * flare;
    }

    // 9. one clock for the whole world (setMaterialsTime under the hood)
    world.setTime(time);
    cart.setTime(time);

    // 10. camera easing runs in WALL time so a paused frame is still explorable
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

  probes(): AlignmentProbe[] {
    const heightAt = this.heightAt;
    const cartRoot = this.cart?.root ?? new THREE.Object3D();
    const cairn = this.world?.cairnRoot ?? new THREE.Object3D();
    const trailhead = this.world?.trailheadRoot ?? new THREE.Object3D();
    return [
      {
        name: 'cart',
        object: cartRoot,
        expectWorldPos: (state) => {
          const x = state.pose[0] ?? 0;
          return [x, heightAt(x), 0] as const;
        },
      },
      {
        name: 'summit-cairn',
        object: cairn,
        expectWorldPos: () => [MC_GOAL_POS, heightAt(MC_GOAL_POS), 0] as const,
      },
      {
        // planted on the respawn abscissa, offset in z inside the exact-profile band
        name: 'trailhead',
        object: trailhead,
        expectWorldPos: () => [MC_START_POS, heightAt(MC_START_POS), TRAILHEAD_Z] as const,
      },
    ];
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
    this.heldPos = Number.NaN;
    this.heldFrames = 0;
  }
}

/**
 * Exponential decay in [0, 1] over `span` playback seconds, 0 outside the window.
 * The linear tail term is not cosmetic: a bare exp() still has real amplitude when the
 * window closes, so every flourish would end on a one-frame pop.
 */
function envelope(age: number, span: number, rate: number): number {
  if (!(age >= 0) || age > span) return 0;
  const e = Math.exp(-rate * age) * (1 - age / span);
  return e < 0 ? 0 : e > 1 ? 1 : e;
}

const createScene: SceneFactory = () => new AlpineValleyScene();
export default createScene;
