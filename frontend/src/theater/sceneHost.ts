
import * as THREE from 'three';
import type { EnvId } from '@shared/envs';
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

export interface HostFrameInfo {
  position: number;
  index: number;
  step: number;
  simTime: number;
  frameCount: number;
  playing: boolean;
  speed: number;
  tier: QualityTier;
  fps: number;
  frameMs: number;
  p95: number;
  p95Threshold: number;
  draws: number;
  triangles: number;
  ghostDraws: number;
  ghostTriangles: number;
  budget: { drawCalls: number; triangles: number };
  scrubbed: boolean;
}

export interface SceneHostGhosts {
  layer: GhostLayer;
  rebuild?(tier: QualityTier): GhostLayer;
}

export interface SceneHostOptions {
  envId: EnvId;
  source: HarnessSource;
  tier?: QualityTier;
  debug?: boolean;
  paused?: boolean;
  ghosts?: SceneHostGhosts;
  at?: number;
  speed?: number;
  onFrame?(info: HostFrameInfo): void;
  onEvents?(events: readonly SimEvent[], steps: readonly number[]): void;
  onTier?(tier: QualityTier, reason: 'auto' | 'manual'): void;
  onReady?(manifest: SceneManifest): void;
}

export interface SceneHostHandle {
  readonly manifest: SceneManifest;
  readonly frameCount: number;
  readonly clock: PlaybackClock;
  readonly tier: QualityTier;
  readonly debug: boolean;
  readonly canvas: HTMLCanvasElement;
  readonly ghosts: GhostLayer | null;
  play(): void;
  pause(): void;
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

function tierDpr(tier: QualityTier): number {
  return tier === 'thumb' ? 1 : tier === 'medium' ? 1.5 : 2;
}

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
  renderer.info.autoReset = false;

  const factory = (await loader()).default;
  let pendingModule: SceneModule | null = factory();
  const manifest = pendingModule.manifest;

  const clock = makePlaybackClock(source.frameCount, source.controlHz);
  clock.speed = options.speed && options.speed > 0 ? options.speed : 1;
  if (options.at !== undefined && Number.isFinite(options.at)) clock.seek(options.at);
  if (options.paused) clock.playing = false;

  const governor: TierGovernor = makeTierGovernor({ tier: options.tier ?? 'hero' });
  let tier: QualityTier = options.tier ?? 'hero';
  const ghostConfig = options.ghosts ?? null;
  let ghostLayer: GhostLayer | null = ghostConfig?.layer ?? null;
  let ghostTier: QualityTier = tier;
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
    frameCount: source.frameCount,
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

  let buildGeneration = 0;

  const buildScene = async (): Promise<void> => {
    const generation = ++buildGeneration;
    building = true;
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
    type AfterRender = THREE.Scene['onAfterRender'];
    const sceneHook: AfterRender = module.scene.onAfterRender;
    module.scene.onAfterRender = function (this: THREE.Scene, ...args: Parameters<AfterRender>) {
      sceneCalls = renderer.info.render.calls;
      sceneTriangles = renderer.info.render.triangles;
      sceneHook.apply(this, args);
    };
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
    cursor.lastIndex = -1;
    cursor.hasPrevious = false;
    governor.reset();
    building = false;
  };

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

    const hadPrevious = cursor.hasPrevious;
    if (hadPrevious) copyFrameState(state, previous);
    sampleFrame(source, position, state);

    const step = advanceCursor(cursor, position, source.frameCount);
    if (step.sequential) collectEvents(source.eventsAt, step.from, step.to, eventBuffer, eventFrames);
    else eventBuffer.length = 0;

    frameUpdate.prev = hadPrevious ? previous : null;
    frameUpdate.dtReal = dtReal;
    frameUpdate.scrubbed = !step.sequential;
    scene.update(frameUpdate);
    ghostLayer?.update(dtReal, clock.time());

    renderer.info.reset();
    post.setBloomPulse(celebrationPulse(scene.scene).bloom);
    post.render(dtReal);

    if (eventBuffer.length > 0 && options.onEvents) {
      eventSteps.length = 0;
      for (const frame of eventFrames) eventSteps.push(frame - 1);
      options.onEvents(eventBuffer, eventSteps);
    }

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
    frameCount: source.frameCount,
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
