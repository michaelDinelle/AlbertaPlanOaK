import * as THREE from 'three';
import { MeshoptSimplifier } from 'meshoptimizer/simplifier';
import { celebrationPulse } from '@kit/celebrate';
import { buildComposer, GLASS_RECTS, type ComposerHandle, type GlassRect } from '@kit/post';
import { TOKENS } from '@kit/tokens';
import type { FrameState, FrameUpdate, SceneContext, SceneModule, SceneSize } from '@kit/types';
import type { EnvEntry } from '@sim/catalog';
import { decoders } from '@sim/registry';
import { EMPTY_EVENT_STREAM, type SimEvent } from '@sim/types';
import type { EnvId } from '../data/envs';
import { SCENES } from '../scenes/table';
import { EMPTY_STAGE, patch, setStage } from '../store';
import { ticksOf } from './recording';
import { keepTapes, tapeFor, type Tape } from './tape';

export interface Alternate {
  runId: string;
  lane: number;
  seed: number;
  entry: EnvEntry;
  horizon: number;
}

export interface Target extends Alternate {
  envId: EnvId;
  live: boolean;
  startStep: number | null;
  alternates: Alternate[];
}

export interface Frame {
  step: number;
  simTime: number;
}

export interface Stage {
  setTarget(target: Target | null): void;
  pull(): void;
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

const frameState = (): FrameState => ({ pose: new Float32Array(9), done: false, step: 0, simTime: 0 });

function sample(source: Tape, position: number, out: FrameState): void {
  const last = source.frameCount - 1;
  const p = position < 0 ? 0 : position > last ? last : position;
  const i = Math.min(last, Math.floor(p));
  const frame = source.frameAt(i);
  const next = source.frameAt(Math.min(last, i + 1));
  const alpha = source.discretePose || frame.done !== 0 ? 0 : p - i;
  if (alpha > 0 && next !== frame) {
    for (let k = 0; k < 9; k++) {
      const a = frame.pose[k] ?? 0;
      let d = (next.pose[k] ?? 0) - a;
      if (((source.wrapMask >>> k) & 1) !== 0) d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      out.pose[k] = a + d * alpha;
    }
  } else {
    out.pose.set(frame.pose.subarray(0, 9));
  }
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
  let sizeDirty = true;
  let lastWall = 0;
  let lastIndex = -1;
  let hasPrevious = false;
  let broken: string | null = null;
  const state = frameState();
  const previous = frameState();
  const events: SimEvent[] = [];
  const update: FrameUpdate = { state, prev: null, events, eventLog: EMPTY_EVENT_STREAM, dtReal: 0, scrubbed: true };
  const frame: Frame = { step: -1, simTime: 0 };

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

  const lastFrame = (): number => (recording === null ? 0 : recording.frameCount - 1);
  const resetCursor = (): void => {
    lastIndex = -1;
    hasPrevious = false;
    events.length = 0;
  };
  const seekPosition = (position: number): void => {
    if (recording === null || target === null) return;
    seconds = Math.min(target.horizon, Math.max(0, position)) / recording.controlHz;
    resetCursor();
  };
  const publish = (): void => {
    if (recording === null || target === null) return;
    setStage({
      status: broken !== null ? 'error' : recording.frameCount > 1 ? 'ready' : 'loading',
      note: broken,
      frameCount: recording.frameCount,
      totalSteps: target.horizon,
      ticks: ticksOf(recording.events, target.horizon),
    });
  };
  const failStage = (what: string, error: unknown): void => {
    broken = `${what}: ${error instanceof Error ? error.message : String(error)}`;
    setStage({ status: 'error', note: broken });
  };

  const buildScene = async (gen: number, envId: EnvId, source: Tape): Promise<void> => {
    scene?.dispose();
    post?.dispose();
    scene = null;
    post = null;
    const [loaded] = await Promise.all([SCENES[envId].load(), MeshoptSimplifier.ready]);
    if (destroyed || gen !== generation) return;
    const module = loaded.default();
    const context: SceneContext = {
      renderer,
      canvas,
      size: sizeOf(),
      tokens: TOKENS,
      envId,
      geometry: source.geometry,
      seed: source.seed,
      time: () => seconds,
    };
    await module.init(context);
    if (destroyed || gen !== generation) {
      module.dispose();
      return;
    }
    scene = module;
    renderer.compile(module.scene, module.camera);
    post = buildComposer(renderer, module.scene, module.camera, SCENES[envId], sizeOf());
    module.setDebug(debug);
    applySize();
    resetCursor();
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
    const duration = lastFrame() / source.controlHz;
    if (playing && duration > 0) {
      seconds += dt * speed;
      if (seconds > duration) seconds = target.live ? duration : seconds % duration;
    }
    const position = seconds * source.controlHz;
    if (hasPrevious) {
      previous.pose.set(state.pose);
      previous.done = state.done;
      previous.step = state.step;
      previous.simTime = state.simTime;
    }
    sample(source, position, state);
    const index = Math.min(lastFrame(), Math.max(0, Math.floor(position)));
    const sequential = hasPrevious && index >= lastIndex && index - lastIndex <= MAX_CATCHUP_FRAMES;
    events.length = 0;
    if (sequential) for (let i = lastIndex + 1; i <= index; i++) events.push(...source.eventsAt(i));
    update.prev = hasPrevious ? previous : null;
    update.eventLog = source.events;
    update.dtReal = dt;
    update.scrubbed = !sequential;
    lastIndex = index;
    hasPrevious = true;
    scene.update(update);
    post.setBloomPulse(celebrationPulse(scene.scene).bloom);
    post.setGlass(glassRects(container));
    post.render(dt);
    frame.step = state.step;
    frame.simTime = state.simTime;
    for (const listener of frameListeners) listener(frame);
  };
  raf = requestAnimationFrame(tick);

  return {
    setTarget(next) {
      if (recording !== null) recording.onChange = () => {};
      if (next === null) {
        recording = null;
        target = null;
        setStage(EMPTY_STAGE);
        return;
      }
      const same = target !== null && target.runId === next.runId && target.envId === next.envId && target.seed === next.seed && target.lane === next.lane;
      const sameEnv = target?.envId === next.envId;
      const carry = sameEnv ? Math.max(0, frame.step) : 0;
      const rebuild = !sameEnv || (decoders[next.envId].seedGeometry && target?.seed !== next.seed);
      target = next;
      const gen = same ? generation : ++generation;
      const opened = tapeFor(next.runId, next.lane, next.envId, next.seed, next.entry, next.horizon);
      opened.onChange = (error) => {
        if (gen !== generation) return;
        if (error !== undefined) failStage(`Could not play seed ${next.seed}`, error);
        else publish();
      };
      recording = opened;
      keepTapes([opened, ...next.alternates.map((a) => tapeFor(a.runId, a.lane, next.envId, a.seed, a.entry, a.horizon))]);
      if (same) {
        publish();
        return;
      }
      broken = null;
      seekPosition((next.startStep ?? carry) + 1);
      if (next.startStep !== null) patch({ playing: true });
      frame.step = -1;
      publish();
      if (rebuild) buildScene(gen, next.envId, opened).catch((error: unknown) => { if (gen === generation) failStage(`Could not build ${next.envId}`, error); });
    },
    pull: () => recording?.pull(),
    play() { playing = true; },
    pause() { playing = false; },
    setSpeed(next) { speed = next; },
    seekStep: (step) => seekPosition(step + 1),
    stepBy: (delta) => seekPosition(frame.step + delta + 1),
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
      keepTapes([]);
      scene?.dispose();
      post?.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
}
