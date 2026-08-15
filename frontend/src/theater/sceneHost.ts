import * as THREE from 'three';
import type { EnvId } from '@shared/envs';
import { makePlaybackClock, sampleFrame, type PlaybackSource, type PlaybackClock } from '@kit/playback';
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
import type { EnvGeometry, SimEvent } from '@sim/types';
import type { EnvCatalog } from '@sim/catalog';
import { sceneRegistry } from '../scenes/registry';
import { specFactory } from '../specs/registry';
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
  scrubbed: boolean;
}

export interface SceneHostGhosts {
  layer: GhostLayer;
  rebuild?(tier: QualityTier): GhostLayer;
}

export interface SceneHostOptions {
  envId: EnvId;
  catalog?: EnvCatalog;
  source: PlaybackSource;
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
  readonly manifest: SceneManifest | null;
  readonly clock: PlaybackClock;
  readonly tier: QualityTier;
  readonly debug: boolean;
  readonly canvas: HTMLCanvasElement;
  readonly ghosts: GhostLayer | null;
  readonly envId: EnvId;
  play(): void;
  pause(): void;
  toggle(): boolean;
  setSpeed(speed: number): void;
  seek(position: number): void;
  setSource(source: PlaybackSource, opts?: { at?: number }): Promise<void>;
  setEnv(envId: EnvId, source: PlaybackSource, ghosts: SceneHostGhosts | null, opts?: { at?: number }): Promise<void>;
  setGhosts(ghosts: SceneHostGhosts | null): void;
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

export function sameGeometry(a: EnvGeometry, b: EnvGeometry): boolean {
  if (a === b) return true;
  if (a.kind !== b.kind) return false;
  if (a.kind !== 'grid') return true;
  const ga = a.grid;
  const gb = b.grid;
  if (ga === undefined || gb === undefined) return ga === gb;
  if (ga.w !== gb.w || ga.h !== gb.h) return false;
  if (ga.walls.byteLength !== gb.walls.byteLength) return false;
  for (let i = 0; i < ga.walls.byteLength; i++) {
    if (ga.walls[i] !== gb.walls[i]) return false;
  }
  return ga.goal?.x === gb.goal?.x && ga.goal?.y === gb.goal?.y;
}

export async function mountSceneHost(
  container: HTMLElement,
  options: SceneHostOptions,
): Promise<SceneHostHandle> {
  let envId = options.envId;
  let source = options.source;
  let loader = sceneRegistry[envId];
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

  const clockSource = {
    get frameCount() {
      return source.frameCount;
    },
    get loop() {
      return source.complete;
    },
  };
  let clock = makePlaybackClock(clockSource, source.controlHz);
  clock.speed = options.speed && options.speed > 0 ? options.speed : 1;
  if (options.at !== undefined && Number.isFinite(options.at)) clock.seek(options.at);
  if (options.paused) clock.playing = false;

  const governor: TierGovernor = makeTierGovernor({ tier: options.tier ?? 'hero' });
  let tier: QualityTier = options.tier ?? 'hero';
  let ghostConfig = options.ghosts ?? null;
  let ghostLayer: GhostLayer | null = ghostConfig?.layer ?? null;
  let ghostTier: QualityTier = tier;
  let ghostOwned = false;
  let debugOn = options.debug === true;
  let scene: SceneModule | null = null;
  let post: ComposerHandle | null = null;
  let manifest: SceneManifest | null = null;
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

  const resetCursor = (): void => {
    cursor.lastIndex = -1;
    cursor.hasPrevious = false;
    eventBuffer.length = 0;
    eventFrames.length = 0;
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
    const currentLoader = loader;
    if (!currentLoader) throw new SceneHostError(`no scene registered for ${envId}`);
    const factory = (options.catalog ? await specFactory(envId, options.catalog) : null)
      ?? (await currentLoader()).default;
    if (destroyed || generation !== buildGeneration) return;
    const module = factory();
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
    manifest = module.manifest;
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
    resetCursor();
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
    info.frameCount = source.frameCount;
    info.playing = clock.playing;
    info.speed = clock.speed;
    info.tier = tier;
    info.fps = fps;
    info.frameMs = frameMs;
    info.p95 = governor.p95;
    info.p95Threshold = governor.threshold;
    info.scrubbed = !step.sequential;
    options.onFrame?.(info);
  };
  raf = requestAnimationFrame(tick);
  if (manifest !== null) options.onReady?.(manifest);

  const adoptGhosts = (ghosts: SceneHostGhosts | null): void => {
    ghostLayer?.group.removeFromParent();
    if (ghostOwned) ghostLayer?.dispose();
    ghostConfig = ghosts;
    ghostLayer = ghosts?.layer ?? null;
    ghostOwned = false;
    ghostTier = tier;
    if (ghostLayer && scene) scene.scene.add(ghostLayer.group);
  };

  const swapSource = (next: PlaybackSource, at: number | undefined): void => {
    const speed = clock.speed;
    const playing = clock.playing;
    source = next;
    clock = makePlaybackClock(clockSource, next.controlHz);
    clock.speed = speed;
    clock.playing = playing;
    if (at !== undefined && Number.isFinite(at)) clock.seek(at);
    resetCursor();
  };

  const handle: SceneHostHandle = {
    get manifest() {
      return manifest;
    },
    get clock() {
      return clock;
    },
    get tier() {
      return tier;
    },
    get debug() {
      return debugOn;
    },
    get ghosts() {
      return ghostLayer;
    },
    get envId() {
      return envId;
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
    async setSource(next, opts = {}) {
      if (next.envId !== envId) throw new SceneHostError('setSource cannot change env — use setEnv');
      const rebuild = !sameGeometry(source.geometry, next.geometry);
      swapSource(next, opts.at ?? clock.position());
      if (rebuild) await buildScene();
    },
    async setEnv(nextEnv, nextSource, ghosts, opts = {}) {
      const nextLoader = sceneRegistry[nextEnv];
      if (!nextLoader) throw new SceneHostError(`no scene registered for ${nextEnv}`);
      envId = nextEnv;
      loader = nextLoader;
      adoptGhosts(null);
      swapSource(nextSource, opts.at ?? 0);
      await buildScene();
      adoptGhosts(ghosts);
      if (!destroyed && manifest !== null) options.onReady?.(manifest);
    },
    setGhosts(ghosts) {
      adoptGhosts(ghosts);
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
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
  return handle;
}
