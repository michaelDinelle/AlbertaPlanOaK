import * as THREE from 'three';
import type { EnvId } from '@shared/envs';
import { makePlaybackClock, sampleFrame, type PlaybackSource, type PlaybackClock } from './playbackSource';
import { celebrationPulse } from '@kit/celebrate';
import { buildComposer, type ComposerHandle } from '@kit/post';
import { TOKENS } from '@kit/tokens';
import type {
  FrameState,
  FrameUpdate,
  SceneContext,
  SceneModule,
  SceneSize,
} from '@kit/types';
import type { SimEvent } from '@sim/types';
import { decoders } from '@sim/registry';
import { sceneRegistry } from '../scenes/registry';
import { SCENES } from '../scenes/table';
import { advanceCursor, collectEvents, makeFrameCursor } from './player';

export interface HostFrameInfo {
  position: number;
  step: number;
  simTime: number;
  frameMs: number;
}

export interface SceneHostOptions {
  envId: EnvId;
  source: PlaybackSource;
  debug?: boolean;
  paused?: boolean;
  at?: number;
  speed?: number;
  onFrame?(info: HostFrameInfo): void;
  onEvents?(events: readonly SimEvent[], steps: readonly number[]): void;
}

export interface SceneHostHandle {
  readonly clock: PlaybackClock;
  readonly debug: boolean;
  readonly canvas: HTMLCanvasElement;
  readonly envId: EnvId;
  play(): void;
  pause(): void;
  toggle(): boolean;
  setSpeed(speed: number): void;
  seek(position: number): void;
  setSource(source: PlaybackSource, opts?: { at?: number }): Promise<void>;
  setEnv(envId: EnvId, source: PlaybackSource, opts?: { at?: number }): Promise<void>;
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
    done: false,
    step: 0,
    simTime: 0,
  };
}

function copyFrameState(from: FrameState, to: FrameState): void {
  to.pose.set(from.pose);
  to.done = from.done;
  to.step = from.step;
  to.simTime = from.simTime;
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
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;

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

  let debugOn = options.debug === true;
  let scene: SceneModule | null = null;
  let post: ComposerHandle | null = null;
  let destroyed = false;

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
    step: 0,
    simTime: 0,
    frameMs: 0,
  };

  const sizeOf = (): SceneSize => ({
    width: Math.max(1, container.clientWidth || canvas.clientWidth || 1),
    height: Math.max(1, container.clientHeight || canvas.clientHeight || 1),
    dpr: window.devicePixelRatio || 1,
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
    scene?.dispose();
    post?.dispose();
    scene = null;
    post = null;

    const size = sizeOf();
    const currentLoader = loader;
    if (!currentLoader) throw new SceneHostError(`no scene registered for ${envId}`);
    const factory = (await currentLoader()).default;
    if (destroyed || generation !== buildGeneration) return;
    const module = factory();
    const context: SceneContext = {
      renderer,
      canvas,
      size,
      tokens: TOKENS,
      envId,
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

    renderer.compile(module.scene, module.camera);
    post = buildComposer(renderer, module.scene, module.camera, SCENES[envId], size);
    module.setDebug(debugOn);
    applySize();
    resetCursor();
  };

  await buildScene();

  let sizeFrame = 0;
  const scheduleSize = (): void => {
    if (sizeFrame !== 0) return;
    sizeFrame = requestAnimationFrame(() => {
      sizeFrame = 0;
      applySize();
    });
  };
  const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(scheduleSize);
  observer?.observe(container);
  const onWindowResize = (): void => scheduleSize();
  window.addEventListener('resize', onWindowResize);

  let raf = 0;
  let lastWall = 0;

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

    post.setBloomPulse(celebrationPulse(scene.scene).bloom);
    post.render(dtReal);

    if (eventBuffer.length > 0 && options.onEvents) {
      eventSteps.length = 0;
      for (const frame of eventFrames) eventSteps.push(frame - 1);
      options.onEvents(eventBuffer, eventSteps);
    }

    info.position = position;
    info.step = state.step;
    info.simTime = state.simTime;
    info.frameMs = frameMs;
    options.onFrame?.(info);
  };
  raf = requestAnimationFrame(tick);

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
    get clock() {
      return clock;
    },
    get debug() {
      return debugOn;
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
      const rebuild = decoders[envId].seedGeometry && next.seed !== source.seed;
      swapSource(next, opts.at ?? clock.position());
      if (rebuild) await buildScene();
    },
    async setEnv(nextEnv, nextSource, opts = {}) {
      const nextLoader = sceneRegistry[nextEnv];
      if (!nextLoader) throw new SceneHostError(`no scene registered for ${nextEnv}`);
      envId = nextEnv;
      loader = nextLoader;
      swapSource(nextSource, opts.at ?? 0);
      await buildScene();
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
      scene?.dispose();
      post?.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
  return handle;
}
