// theater/sceneHost — the WebGL half of the Theater.
//
// The grown-up sibling of `kit/harness`: same host contract, same frame
// assembly, same composer wiring, same celebration-pulse hook — so a scene that
// looks right on the artisan page looks right here. What it adds is what a hero
// view needs and a dev page does not:
//
//   * the TIER GOVERNOR (STYLE.md makes tier selection a host duty): p95 frame
//     time over budget for two sustained seconds drops a tier and rebuilds.
//   * a UI-free surface — every readout is pushed to the view through callbacks,
//     so the chrome is ordinary DOM the rest of the app can style.
//   * frame-accurate transport (step buttons, seek, speed) driven from outside.
//
// Ownership (kit/types.ts): this file owns the ONE WebGLRenderer, the canvas and
// its resize, the playback clock, the EffectComposer chain, tier selection and
// the debug toggle. The scene owns everything inside its THREE.Scene.

import * as THREE from 'three';
import type { EnvId } from '@shared/envs';
// The clock and the frame sampler are the harness's, unchanged — the Theater
// renders the same SceneModule through the same host contract, so it must assemble
// frames the same way (`player.advanceCursor` carries the catch-up rule).
import { makePlaybackClock, sampleFrame, type HarnessSource, type PlaybackClock } from '@kit/harness';
import { celebrationPulse } from '@kit/celebrate';
import type { GhostLayer } from '@kit/ghosts';
import { configureRendererShadows } from '@kit/lighting';
import { buildComposer, type ComposerHandle } from '@kit/post';
import { TOKENS } from '@kit/tokens';
import type {
  FrameState,
  FrameUpdate,
  QualityTier,
  SceneContext,
  SceneManifest,
  SceneModule,
  SceneSize,
} from '@kit/types';
import type { SimEvent } from '@sim/types';
import { sceneRegistry } from '../scenes/registry';
import { advanceCursor, collectEvents, makeFrameCursor, makeTierGovernor, type TierGovernor } from './player';

/** Per-frame readout pushed to the view. The SAME object every frame — copy what you keep. */
export interface HostFrameInfo {
  /** fractional frame position */
  position: number;
  /** integer frame index */
  index: number;
  /** record step of that frame (-1 for the synthesized init frame) */
  step: number;
  simTime: number;
  frameCount: number;
  playing: boolean;
  speed: number;
  tier: QualityTier;
  fps: number;
  frameMs: number;
  /** governor's p95 frame time in ms; 0 until its window is warm */
  p95: number;
  /** the ms p95 must exceed to count as over budget; 0 until warm */
  p95Threshold: number;
  /** draws/triangles latched at the scene's onAfterRender, before the post chain */
  draws: number;
  triangles: number;
  /**
   * How many of `draws` the ghost layer accounts for (0 without one). The ghost group
   * is a child of the scene, so it is already inside the latch — this is the split, so
   * a view can say "34 draws, 8 of them the population" instead of guessing.
   */
  ghostDraws: number;
  /**
   * How many of `triangles` the ghost layer accounts for (0 without one).
   *
   * This is the split that MATTERS. Instancing pins the population's draw count to the
   * body plan — 8 for cheetah at 64 ghosts and at 8192 — but its triangle count is
   * strictly linear in the ghost count (kit/ghosts `GHOST_TRIANGLES_PER_GHOST`: a hero
   * cheetah ghost is 1188, so 1024 seeds are 1.22M triangles, several times a whole
   * scene's budget). Without the split, `triangles` reads over budget and no caller can
   * tell an expensive SCENE from a large POPULATION.
   */
  ghostTriangles: number;
  /**
   * The SCENE's manifest budget — it governs `draws − ghostDraws` and
   * `triangles − ghostTriangles`. The population is a separate, declared cost: it is
   * chosen by `maxGhosts` against `GhostLayer.trianglesPerGhost()`, not by this.
   */
  budget: { drawCalls: number; triangles: number };
  /** true while the last jump was a scrub (no events delivered) */
  scrubbed: boolean;
}

/**
 * The population layer, mounted alongside the hero scene (kit/ghosts).
 *
 * The host attaches `layer.group` (named `oakGhostLayer`) to the scene AFTER `init()`
 * and detaches it before every rebuild, so the layer survives tier changes and no scene
 * ever sees it — the boundary in kit/types.ts. The host also ticks the layer with the
 * SAME dt and playback clock it hands the scene, so a glint freezes on pause like every
 * other celebration envelope (canon lesson 7).
 */
export interface SceneHostGhosts {
  /** the caller builds and OWNS this one; the host never disposes it */
  layer: GhostLayer;
  /**
   * Optional: rebuild the layer at a new quality tier (instance detail scales with tier
   * exactly like a scene's). The host disposes the layer it replaces and owns whatever
   * this returns. Without it, one layer is carried across every tier change.
   */
  rebuild?(tier: QualityTier): GhostLayer;
}

export interface SceneHostOptions {
  envId: EnvId;
  source: HarnessSource;
  tier?: QualityTier;
  debug?: boolean;
  paused?: boolean;
  /** OPTIONAL population layer — omit and the theater behaves exactly as before */
  ghosts?: SceneHostGhosts;
  /** initial frame position */
  at?: number;
  speed?: number;
  /** every rendered frame; the argument is REUSED */
  onFrame?(info: HostFrameInfo): void;
  /**
   * Sim events delivered on a sequential step. Both arrays are REUSED and are
   * parallel: `steps[i]` is the record step `events[i]` fired on (a catch-up
   * tick can span several).
   */
  onEvents?(events: readonly SimEvent[], steps: readonly number[]): void;
  /** a tier change that has finished rebuilding */
  onTier?(tier: QualityTier, reason: 'auto' | 'manual'): void;
  /** the scene module is built and the first frame is up */
  onReady?(manifest: SceneManifest): void;
}

export interface SceneHostHandle {
  readonly manifest: SceneManifest;
  readonly frameCount: number;
  readonly clock: PlaybackClock;
  readonly tier: QualityTier;
  readonly debug: boolean;
  readonly canvas: HTMLCanvasElement;
  /** the LIVE population layer (a tier rebuild can replace it), or null */
  readonly ghosts: GhostLayer | null;
  play(): void;
  pause(): void;
  /** returns the new playing state */
  toggle(): boolean;
  setSpeed(speed: number): void;
  seek(position: number): void;
  setTier(tier: QualityTier): Promise<void>;
  setDebug(on: boolean): void;
  destroy(): void;
}

export class SceneHostError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SceneHostError';
  }
}

function makeFrameState(): FrameState {
  return {
    pose: new Float32Array(9),
    ctrl: new Float32Array(6),
    done: false,
    step: 0,
    simTime: 0,
    derived: null,
  };
}

function copyFrameState(from: FrameState, to: FrameState): void {
  to.pose.set(from.pose);
  to.ctrl.set(from.ctrl);
  to.done = from.done;
  to.step = from.step;
  to.simTime = from.simTime;
  to.derived = from.derived;
}

/** DPR ceiling per tier — the same ladder the artisan page uses. */
function tierDpr(tier: QualityTier): number {
  return tier === 'thumb' ? 1 : tier === 'medium' ? 1.5 : 2;
}

/**
 * Mount one scene against one decoded lane.
 *
 * Rejects (never silently degrades) when the env has no scene or WebGL is
 * unavailable: the Theater's whole job is showing real experience, so a
 * placeholder standing in for it would be a lie the view must be free to report.
 */
export async function mountSceneHost(
  container: HTMLElement,
  options: SceneHostOptions,
): Promise<SceneHostHandle> {
  const { envId, source } = options;
  const loader = sceneRegistry[envId];
  if (!loader) throw new SceneHostError(`no scene registered for ${envId}`);

  const canvas = document.createElement('canvas');
  canvas.className = 'theater-canvas';
  container.appendChild(canvas);

  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance',
    });
  } catch (cause) {
    canvas.remove();
    throw new SceneHostError(`WebGL unavailable: ${String(cause)}`);
  }
  renderer.setClearColor(0x05070c, 1);
  // The manifest budget is a SCENE budget, but the composer issues ~15 more
  // draws afterwards — so reset once per frame and latch at onAfterRender.
  renderer.info.autoReset = false;

  const factory = (await loader()).default;
  // Build ONE module up front for its manifest; the first buildScene() consumes
  // this same instance, so nothing is constructed and thrown away.
  let pendingModule: SceneModule | null = factory();
  const manifest = pendingModule.manifest;

  const clock = makePlaybackClock(source.frames.length, source.controlHz);
  clock.speed = options.speed && options.speed > 0 ? options.speed : 1;
  if (options.at !== undefined && Number.isFinite(options.at)) clock.seek(options.at);
  if (options.paused) clock.playing = false;

  const governor: TierGovernor = makeTierGovernor({ tier: options.tier ?? 'hero' });
  let tier: QualityTier = options.tier ?? 'hero';
  const ghostConfig = options.ghosts ?? null;
  let ghostLayer: GhostLayer | null = ghostConfig?.layer ?? null;
  /** the tier the CURRENT ghost layer was built for */
  let ghostTier: QualityTier = tier;
  /** true once the host built the live layer itself (and therefore must dispose it) */
  let ghostOwned = false;
  let debugOn = options.debug === true;
  let scene: SceneModule | null = null;
  let post: ComposerHandle | null = null;
  let sceneCalls = 0;
  let sceneTriangles = 0;
  let destroyed = false;
  let building = true;

  const state = makeFrameState();
  const previous = makeFrameState();
  const cursor = makeFrameCursor();
  const eventBuffer: SimEvent[] = [];
  const eventFrames: number[] = [];
  const eventSteps: number[] = [];
  const frameUpdate: FrameUpdate = {
    state,
    prev: null,
    events: eventBuffer,
    dtReal: 0,
    scrubbed: true,
  };

  const info: HostFrameInfo = {
    position: 0,
    index: 0,
    step: 0,
    simTime: 0,
    frameCount: source.frames.length,
    playing: clock.playing,
    speed: clock.speed,
    tier,
    fps: 0,
    frameMs: 0,
    p95: 0,
    p95Threshold: 0,
    draws: 0,
    triangles: 0,
    ghostDraws: 0,
    ghostTriangles: 0,
    budget: manifest.budget,
    scrubbed: true,
  };

  const sizeOf = (): SceneSize => ({
    width: Math.max(1, container.clientWidth || canvas.clientWidth || 1),
    height: Math.max(1, container.clientHeight || canvas.clientHeight || 1),
    dpr: Math.min(tierDpr(tier), window.devicePixelRatio || 1),
  });

  const applySize = (): void => {
    const size = sizeOf();
    renderer.setPixelRatio(size.dpr);
    renderer.setSize(size.width, size.height, false);
    scene?.resize(size);
    post?.resize(size.width, size.height, size.dpr);
  };

  // Scene builds are async, so two quick tier switches can interleave: build A is
  // still awaiting init when build B disposes and starts over, and A then installs
  // itself over B. A generation counter makes the superseded build clean up and return.
  let buildGeneration = 0;

  const buildScene = async (): Promise<void> => {
    const generation = ++buildGeneration;
    building = true;
    // The ghost layer outlives scene rebuilds, so detach it BEFORE the scene that is
    // holding it is disposed — a scene must never dispose (or even see) this group.
    ghostLayer?.group.removeFromParent();
    scene?.dispose();
    post?.dispose();
    scene = null;
    post = null;

    configureRendererShadows(renderer, tier);
    const size = sizeOf();
    const module = pendingModule ?? factory();
    pendingModule = null;
    const context: SceneContext = {
      renderer,
      canvas,
      size,
      tier,
      tokens: TOKENS,
      geometry: source.geometry,
      seed: source.seed,
      time: () => clock.time(),
    };
    await module.init(context);
    if (destroyed || generation !== buildGeneration) {
      module.dispose();
      return;
    }
    scene = module;
    // chain, never clobber: the scene owns its THREE.Scene and may use this hook
    type AfterRender = THREE.Scene['onAfterRender'];
    const sceneHook: AfterRender = module.scene.onAfterRender;
    module.scene.onAfterRender = function (this: THREE.Scene, ...args: Parameters<AfterRender>) {
      sceneCalls = renderer.info.render.calls;
      sceneTriangles = renderer.info.render.triangles;
      sceneHook.apply(this, args);
    };
    // Ghosts: rebuild at the new tier when the caller supplied a factory, guarded by the
    // same generation counter — a superseded build disposes what it just made and leaves.
    if (ghostConfig?.rebuild && ghostTier !== tier) {
      const next = ghostConfig.rebuild(tier);
      if (destroyed || generation !== buildGeneration) {
        next.dispose();
      } else {
        if (ghostOwned) ghostLayer?.dispose();
        ghostLayer = next;
        ghostOwned = true;
        ghostTier = tier;
      }
    }
    if (ghostLayer) module.scene.add(ghostLayer.group);

    post = buildComposer(renderer, module.scene, module.camera, module.manifest.postConfig, tier, size);
    module.setDebug(debugOn);
    applySize();
    // force a snap on the first frame after any rebuild
    cursor.lastIndex = -1;
    cursor.hasPrevious = false;
    governor.reset();
    building = false;
  };

  // Nothing can have destroyed us yet: the handle that owns `destroy()` does not
  // exist until this function returns, and the caller's own teardown path
  // disposes the resolved handle.
  await buildScene();

  const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => applySize());
  observer?.observe(container);
  const onWindowResize = (): void => applySize();
  window.addEventListener('resize', onWindowResize);

  let raf = 0;
  let lastWall = 0;
  let fps = 0;

  const tick = (now: number): void => {
    raf = requestAnimationFrame(tick);
    if (!scene || !post) return;
    const frameMs = lastWall === 0 ? 0 : now - lastWall;
    const dtReal = lastWall === 0 ? 0 : Math.min(0.1, frameMs / 1000);
    lastWall = now;

    const position = clock.advance(dtReal);

    // `hasPrevious` must be read BEFORE the cursor advances — it is the answer to
    // "was there a frame before this one", and advanceCursor sets it to true.
    const hadPrevious = cursor.hasPrevious;
    if (hadPrevious) copyFrameState(state, previous);
    sampleFrame(source, position, state);

    const step = advanceCursor(cursor, position, source.frames.length);
    if (step.sequential) collectEvents(source.events, step.from, step.to, eventBuffer, eventFrames);
    else eventBuffer.length = 0;

    frameUpdate.prev = hadPrevious ? previous : null;
    frameUpdate.dtReal = dtReal;
    frameUpdate.scrubbed = !step.sequential;
    scene.update(frameUpdate);
    // same dt, same playback clock as the scene: a ghost glint freezes on pause
    ghostLayer?.update(dtReal, clock.time());

    renderer.info.reset();
    // the scene's OWN channel (celebration pools own theirs), so one lane's +1
    // can never bloom another scene's frame
    post.setBloomPulse(celebrationPulse(scene.scene).bloom);
    post.render(dtReal);

    if (eventBuffer.length > 0 && options.onEvents) {
      eventSteps.length = 0;
      // frames[i].step is the RECORD row; frames[0] is the synthesized step -1.
      for (const frame of eventFrames) eventSteps.push(source.frames[frame]?.step ?? frame - 1);
      options.onEvents(eventBuffer, eventSteps);
    }

    // The governor only judges steady-state frames: a rebuild's shader compiles
    // and texture uploads are not a budget failure.
    if (!building && frameMs > 0) {
      const drop = governor.sample(frameMs);
      if (drop && drop !== tier) {
        tier = drop;
        void buildScene().then(() => {
          if (!destroyed) options.onTier?.(tier, 'auto');
        });
      }
    }

    fps = dtReal > 0 ? (fps === 0 ? 1 / dtReal : fps * 0.9 + (1 / dtReal) * 0.1) : fps;
    info.position = position;
    info.index = step.index;
    info.step = state.step;
    info.simTime = state.simTime;
    info.playing = clock.playing;
    info.speed = clock.speed;
    info.tier = tier;
    info.fps = fps;
    info.frameMs = frameMs;
    info.p95 = governor.p95;
    info.p95Threshold = governor.threshold;
    info.draws = sceneCalls;
    info.triangles = sceneTriangles;
    info.ghostDraws = ghostLayer ? ghostLayer.draws() : 0;
    info.ghostTriangles = ghostLayer ? ghostLayer.triangles() : 0;
    info.scrubbed = !step.sequential;
    options.onFrame?.(info);
  };
  raf = requestAnimationFrame(tick);
  options.onReady?.(manifest);

  const handle: SceneHostHandle = {
    manifest,
    frameCount: source.frames.length,
    clock,
    get tier() {
      return tier;
    },
    get debug() {
      return debugOn;
    },
    get ghosts() {
      return ghostLayer;
    },
    canvas,
    play() {
      clock.playing = true;
    },
    pause() {
      clock.playing = false;
    },
    toggle() {
      clock.playing = !clock.playing;
      return clock.playing;
    },
    setSpeed(speed) {
      clock.speed = speed > 0 ? speed : 1;
    },
    seek(position) {
      clock.seek(position);
    },
    async setTier(next) {
      if (next === tier) return;
      tier = next;
      governor.adopt(next);
      await buildScene();
      if (!destroyed) options.onTier?.(tier, 'manual');
    },
    setDebug(on) {
      debugOn = on;
      scene?.setDebug(on);
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      cancelAnimationFrame(raf);
      observer?.disconnect();
      window.removeEventListener('resize', onWindowResize);
      // the caller's own layer is theirs to dispose; anything the host built is the
      // host's to release
      ghostLayer?.group.removeFromParent();
      if (ghostOwned) ghostLayer?.dispose();
      scene?.dispose();
      post?.dispose();
      pendingModule?.dispose();
      pendingModule = null;
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
  return handle;
}
