import * as THREE from 'three';
import { MeshoptSimplifier } from 'meshoptimizer/simplifier';
import { celebrationPulse } from '@kit/celebrate';
import { buildComposer, GLASS_RECTS, type ComposerHandle, type GlassRect } from '@kit/post';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneModule, SceneSize } from '@kit/types';
import { EMPTY_EVENT_STREAM, type SimEvent } from '@sim/types';
import type { EnvDescriptor } from '../data/envs';
import { SCENES } from '../scenes/table';
import { EMPTY_STAGE, patch, setStage } from '../store';
import { openTape, type Tape } from './tape';

export interface Target {
  runId: string;
  environment: EnvDescriptor;
  lane: number;
  seed: number;
  horizon: number;
  hash: bigint;
}

export interface Frame {
  step: number;
  simTime: number;
  requestedStep: number;
}

export interface Stage {
  setTarget(target: Target | null): void;
  play(): void;
  pause(): void;
  setSpeed(speed: number): void;
  seekStep(step: number): void;
  stepBy(delta: number): void;
  setDebug(on: boolean): void;
  resetCamera(): void;
  destroy(): void;
}

export const frameListeners = new Set<(frame: Frame) => void>();

const MAX_CATCHUP_FRAMES = 12;

const frameState = (): FrameState => ({ pose: new Float32Array(9), observations: new Float32Array(0), reward: 0, reset: false, done: false, step: 0, simTime: 0 });

function sample(source: Tape, position: number, out: FrameState): void {
  const i = Math.floor(position), frame = source.frameAt(i);
  const next = source.hasFrame(i + 1) ? source.frameAt(i + 1) : frame;
  const alpha = source.discretePose || next.reset ? 0 : position - i;
  out.pose.fill(0);
  for (let k = 0; k < frame.pose.length; k++) {
    const a = frame.pose[k]!;
    let d = next.pose[k]! - a;
    if (((source.wrapMask >>> k) & 1) !== 0) d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
    out.pose[k] = a + d * alpha;
  }
  out.observations = frame.observations;
  out.reward = frame.reward;
  out.reset = frame.reset;
  out.done = frame.done !== 0;
  out.step = frame.step;
  out.simTime = frame.step / source.controlHz;
}

const radii = new WeakMap<Element, number>();

function glassRects(container: HTMLElement): GlassRect[] {
  const dpr = window.devicePixelRatio || 1;
  const height = container.clientHeight;
  const out: GlassRect[] = [];
  for (const el of document.querySelectorAll('.glass')) {
    if (out.length === GLASS_RECTS) break;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;
    let radius = radii.get(el);
    if (radius === undefined) radii.set(el, (radius = parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0));
    out.push({ x: rect.left * dpr, y: (height - rect.bottom) * dpr, width: rect.width * dpr, height: rect.height * dpr, radius: radius * dpr });
  }
  return out;
}

export function mountStage(container: HTMLElement): Stage {
  const canvas = document.createElement('canvas');
  canvas.className = 'theater-canvas';
  container.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: true });
  renderer.setClearColor(0x05070c, 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;

  let target: Target | null = null;
  let recording: Tape | null = null;
  let scene: SceneModule | null = null;
  let post: ComposerHandle | null = null;
  let generation = 0;
  let destroyed = false;
  let playing = true;
  let speed = 1;
  let debug = false;
  let seconds = 0;
  let position = 0;
  let sceneKey: string | null = null;
  let buildingKey: string | null = null;
  let sizeDirty = true;
  let lastWall = 0;
  let lastIndex = -1;
  let hasPrevious = false;
  let awaitingFrame = true;
  let broken: string | null = null;
  const state = frameState();
  const previous = frameState();
  const events: SimEvent[] = [];
  const update: FrameUpdate = { state, prev: null, events, eventLog: EMPTY_EVENT_STREAM, dtReal: 0, scrubbed: true, position: 0, controlHz: 1, frameAt: () => { throw new Error('No replay frame selected'); } };
  const frame: Frame = { step: -1, simTime: 0, requestedStep: 0 };

  const sizeOf = (): SceneSize => ({
    width: Math.max(1, container.clientWidth),
    height: Math.max(1, container.clientHeight),
    dpr: window.devicePixelRatio || 1,
  });
  const applySize = (): void => {
    const size = sizeOf();
    renderer.setPixelRatio(size.dpr);
    renderer.setSize(size.width, size.height, false);
    scene?.resize(size);
    post?.resize(size.width, size.height, size.dpr);
  };
  const observer = new ResizeObserver(() => { sizeDirty = true; });
  observer.observe(container);

  const resetCursor = (): void => {
    lastIndex = -1;
    hasPrevious = false;
    events.length = 0;
  };
  const seekPosition = (next: number): void => {
    if (recording === null || target === null) return;
    position = Math.min(target.horizon, Math.max(0, next));
    recording.seek(position);
    awaitingFrame = true;
    resetCursor();
    publish();
  };
  const publish = (): void => {
    if (recording === null || target === null) return;
    const error = broken ?? recording.error?.message ?? null;
    setStage({
      status: error !== null ? 'error' : scene === null || recording.geometry === null ? 'loading' : recording.hasFrame(position) ? 'ready' : 'buffering',
      note: error,
      frameCount: recording.frameCount,
      totalSteps: target.horizon,
      requestedStep: Math.floor(position),
      ticks: recording.ticks,
      identical: recording.identical,
    });
  };
  const failStage = (what: string, error: unknown): void => {
    broken = `${what}: ${error instanceof Error ? error.message : String(error)}`;
    setStage({ status: 'error', note: broken });
  };

  const keyOf = (next: Target): string => `${next.environment.envId}/${next.seed}`;
  const buildScene = async (key: string, source: Tape): Promise<void> => {
    if (source.geometry === null || buildingKey === key || sceneKey === key) return;
    const gen = ++generation, environment = source.environment, spec = SCENES[environment.family]!;
    buildingKey = key;
    scene?.dispose(); post?.dispose(); scene = null; post = null; sceneKey = null;
    let module: SceneModule | null = null;
    try {
      const [loaded] = await Promise.all([spec.load(), MeshoptSimplifier.ready]);
      if (destroyed || gen !== generation) return;
      module = loaded.default();
      const context: SceneContext = { renderer, canvas, size: sizeOf(), tokens: TOKENS, envId: environment.envId, environment, geometry: source.geometry, seed: source.seed, time: () => seconds };
      await module.init(context);
      if (destroyed || gen !== generation || target === null || keyOf(target) !== key) { module.dispose(); return; }
      scene = module;
      renderer.compile(module.scene, module.camera);
      post = buildComposer(renderer, module.scene, module.camera, spec, sizeOf());
      sceneKey = key; buildingKey = null;
      module.setDebug(debug); applySize(); resetCursor(); publish();
    } catch (error) {
      module?.dispose();
      if (gen === generation) { scene = null; post?.dispose(); post = null; sceneKey = null; buildingKey = null; failStage(`Could not build ${environment.envId}`, error); }
    }
  };

  let raf = 0;
  const tick = (now: number): void => {
    raf = requestAnimationFrame(tick);
    if (scene === null || post === null || recording === null || target === null) {
      lastWall = 0;
      return;
    }
    if (sizeDirty) {
      sizeDirty = false;
      applySize();
    }
    const dt = lastWall === 0 ? 0 : Math.min(0.1, (now - lastWall) / 1000);
    lastWall = now;
    const source = recording;
    if (playing && hasPrevious && !awaitingFrame && source.hasFrame(position)) {
      position = Math.min(target.horizon, position + dt * speed * source.controlHz);
      if (!source.hasFrame(position)) { source.seek(position); publish(); }
      if (position === target.horizon) patch({ playing: false });
    }
    if (!source.hasFrame(position)) {
      awaitingFrame = true;
      post.render(0);
      frame.requestedStep = Math.floor(position);
      for (const listener of frameListeners) listener(frame);
      return;
    }
    awaitingFrame = false;
    seconds = position / source.controlHz;
    if (hasPrevious) {
      previous.pose.set(state.pose);
      previous.done = state.done;
      previous.step = state.step;
      previous.simTime = state.simTime;
      previous.observations = state.observations; previous.reward = state.reward; previous.reset = state.reset;
    }
    sample(source, position, state);
    const index = Math.floor(position);
    const sequential = hasPrevious && index >= lastIndex && index - lastIndex <= MAX_CATCHUP_FRAMES;
    events.length = 0;
    if (sequential) for (let i = lastIndex + 1; i <= index; i++) events.push(...source.eventsAt(i));
    update.prev = hasPrevious ? previous : null;
    update.eventLog = source.events;
    update.dtReal = dt;
    update.scrubbed = !sequential;
    update.position = position;
    update.controlHz = source.controlHz;
    update.frameAt = (i) => { const step = Math.max(0, Math.min(target!.horizon, Math.floor(i))); return source.frameAt(source.hasFrame(step) ? step : index); };
    lastIndex = index;
    hasPrevious = true;
    scene.update(update);
    post.setBloomPulse(celebrationPulse(scene.scene).bloom);
    post.setGlass(glassRects(container));
    post.render(dt);
    frame.step = state.step;
    frame.simTime = state.simTime;
    frame.requestedStep = Math.floor(position);
    for (const listener of frameListeners) listener(frame);
  };
  raf = requestAnimationFrame(tick);

  return {
    setTarget(next) {
      if (target !== null && next !== null && target.runId === next.runId && target.lane === next.lane && target.hash === next.hash && recording?.error === null && broken === null) return;
      const sameEnvironment = target?.environment.envId === next?.environment.envId;
      const key = next === null ? null : keyOf(next);
      if (recording !== null) { recording.onChange = () => {}; recording.drop(); recording = null; }
      if (next === null || key !== sceneKey && key !== buildingKey) {
        generation++; buildingKey = null; sceneKey = null;
        scene?.dispose(); post?.dispose(); scene = null; post = null; renderer.clear();
      }
      target = next; broken = null; resetCursor(); lastWall = 0;
      if (!sameEnvironment) { position = 0; seconds = 0; frame.step = -1; }
      if (next === null) { setStage(EMPTY_STAGE); return; }
      if (!SCENES[next.environment.family] || !next.environment.replay) {
        setStage({ ...EMPTY_STAGE, status: 'unsupported', note: 'Visualization unavailable', totalSteps: next.horizon });
        return;
      }
      const opened = openTape(next.runId, next.lane, next.environment, next.seed, next.horizon, next.hash);
      recording = opened;
      opened.onChange = () => {
        if (recording !== opened) return;
        publish();
        if (opened.error === null) void buildScene(key!, opened).catch(error => { if (recording === opened) { buildingKey = null; failStage('Scene initialization failed', error); } });
      };
      seekPosition(position);
      publish();
    },
    play() { playing = true; },
    pause() { playing = false; },
    setSpeed(next) { speed = next; },
    seekStep: (step) => seekPosition(step),
    stepBy: (delta) => seekPosition(Math.floor(position) + delta),
    setDebug(on) {
      debug = on;
      scene?.setDebug(on);
    },
    resetCamera: () => scene?.resetCamera(),
    destroy() {
      destroyed = true;
      generation++;
      cancelAnimationFrame(raf);
      observer.disconnect();
      recording?.drop();
      scene?.dispose();
      post?.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
}
