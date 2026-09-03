import type { EnvId } from '@shared/envs';
import type { Policy, RecordingRef, RunDetail } from '@shared/types';
import { entryFor, parseCatalog } from '@sim/catalog';
import { nameOf } from '../scenes/table';
import { createStore } from '../state/store';
import { frameFromStep, heroSeed, stepFromFrame } from '../lib/selection';
import { publishPosition, setTransport, subscribeTransport, transportSnapshot } from '../playback/store';
import {
  consumePendingSeek,
  selectionSnapshot,
  setPendingSeek,
  subscribeSelection,
} from '../state/selectionStore';
import {
  loadHero,
  makeInitialSource,
  recordingRef,
  seedsFor,
  seekTargetForTick,
  type HeroHandle,
  type TimelineTick,
} from './loader';
import { clampPosition } from './player';
import { liveEdgeBufferFrames, type PlaybackSource } from './playbackSource';
import { mountSceneHost, type HostFrameInfo, type SceneHostHandle } from './sceneHost';

export interface TheaterInputs {
  active: RunDetail | null;
  policy: Policy;
  baselineId: string | undefined;
  baselinePending: boolean;
  live: boolean;
}

export type StageStatus = 'empty' | 'loading' | 'ready' | 'error';

export interface StageMeta {
  status: StageStatus;
  note: string | null;
  noteDetail: string | null;
  envName: string | null;
  seed: number | null;
  promoted: boolean;
  policy: Policy;
  frameCount: number;
  totalSteps: number;
  loadedSteps: number;
  complete: boolean;
  ticks: TimelineTick[];
}

const EMPTY_META: StageMeta = {
  status: 'empty',
  note: null,
  noteDetail: null,
  envName: null,
  seed: null,
  promoted: false,
  policy: 'oak',
  frameCount: 1,
  totalSteps: 0,
  loadedSteps: 0,
  complete: false,
  ticks: [],
};

const meta = createStore<StageMeta>(EMPTY_META);

export const stageMetaSnapshot = meta.get;
export const useStageMeta = meta.use;

const setStageMeta = meta.patch;

export interface StageFrame {
  position: number;
  frameCount: number;
  step: number;
  totalSteps: number;
  simTime: number;
  rate: number;
}

const frameListeners = new Set<(frame: StageFrame) => void>();

export function subscribeStageFrame(listener: (frame: StageFrame) => void): () => void {
  frameListeners.add(listener);
  return () => {
    frameListeners.delete(listener);
  };
}

const frameScratch: StageFrame = { position: 0, frameCount: 1, step: 0, totalSteps: 0, simTime: 0, rate: 0 };

let container: HTMLElement | null = null;
let disposeQueued = false;
let subscribed = false;
let unsubscribeSelection: (() => void) | null = null;
let unsubscribeTransport: (() => void) | null = null;
let sceneHost: SceneHostHandle | null = null;
let stageQueue: Promise<void> = Promise.resolve();
let hero: HeroHandle | null = null;
const heroCache = new Map<string, HeroHandle>();
const HERO_CACHE_MAX = 6;
let generation = 0;
let attached = false;
let inputs: TheaterInputs | null = null;
let targetKey = '';
let appliedDebug = false;

function degradedReason(recordings: readonly RecordingRef[], envId: EnvId): string | null {
  if (recordings.length === 0) return 'no recordings on this run';
  if (seedsFor(recordings, envId).length === 0) return `no recordings for ${nameOf(envId)} in this run`;
  return null;
}

function baselineReason(
  recordings: readonly RecordingRef[],
  envId: EnvId,
  seed: number | null,
  policy: 'ppo' | 'random',
): string | null {
  if (recordings.length === 0) return `no ${policy} recordings on this baseline`;
  if (seedsFor(recordings, envId).length === 0) return `no ${policy} recordings for ${nameOf(envId)}`;
  if (seed !== null && recordingRef(recordings, envId, seed) === null) {
    return `no ${policy} recording for ${nameOf(envId)} seed ${seed}`;
  }
  return null;
}

function totalSteps(): number {
  return hero?.totalSteps ?? Math.max(1, frameCount() - 1);
}

function frameCount(): number {
  return attached && hero !== null ? hero.source.frameCount : 1;
}

function queueStage(work: () => Promise<void>): void {
  stageQueue = stageQueue.then(work).catch((error: unknown) => {
    if (container === null) return;
    setStageMeta({
      status: 'error',
      note: 'Could not play this seed',
      noteDetail: error instanceof Error ? error.message : String(error),
    });
  });
}

function applyTransport(): void {
  if (sceneHost === null) return;
  const transport = transportSnapshot();
  sceneHost.setSpeed(transport.speed);
  if (transport.playing) sceneHost.play();
  else sceneHost.pause();
}

function applyDebug(): void {
  const debug = selectionSnapshot().debug;
  if (debug === appliedDebug) return;
  appliedDebug = debug;
  sceneHost?.setDebug(debug);
}

function liveStartFrame(source: PlaybackSource): number {
  const edge = source.edge;
  const head = source.frameCount - 1;
  return edge === null ? head : Math.max(0, head - Math.round(liveEdgeBufferFrames(edge)));
}

function resolvePendingFrame(handle: HeroHandle, source: PlaybackSource): number {
  const pending = selectionSnapshot().pendingSeekStep;
  if (pending === null) return 0;
  if (source.frameCount <= 1) return 0;
  const total = handle.totalSteps;
  if (pending > total && !source.complete) return liveStartFrame(source);
  consumePendingSeek(pending);
  return clampPosition(frameFromStep(Math.min(pending, total), total, source.frameCount), source.frameCount);
}

function publishHeroMeta(handle: HeroHandle): void {
  setStageMeta({
    frameCount: handle.source.frameCount,
    totalSteps: handle.totalSteps,
    loadedSteps: handle.loadedSteps,
    complete: handle.source.complete,
    ticks: handle.ticks(),
  });
}

function onFrame(info: HostFrameInfo): void {
  frameScratch.position = info.position;
  frameScratch.frameCount = frameCount();
  frameScratch.step = info.step;
  frameScratch.totalSteps = totalSteps();
  frameScratch.simTime = info.simTime;
  frameScratch.rate = info.rate;
  for (const listener of frameListeners) listener(frameScratch);
  publishPosition(stepFromFrame(info.position, frameScratch.totalSteps, frameScratch.frameCount));
}

function computeTargetKey(): string {
  const sel = selectionSnapshot();
  const inp = inputs;
  const pending = inp === null || sel.policy !== inp.policy || inp.baselinePending;
  return [
    container === null ? '' : 'mounted',
    inp?.active?.id ?? '',
    sel.envId ?? '',
    String(heroSeed(sel.selection) ?? ''),
    sel.policy,
    pending ? 'pending' : 'settled',
    inp !== null && inp.active === null ? `missing:${inp.baselineId ?? ''}` : '',
  ].join('|');
}

function retarget(): void {
  const key = computeTargetKey();
  if (key === targetKey) {
    applyDebug();
    applyPendingSeek();
    return;
  }
  targetKey = key;
  mountHero();
}

let liveStarted = false;

function applyPendingSeek(): void {
  const pending = selectionSnapshot().pendingSeekStep;
  if (pending === null || !attached || hero === null || sceneHost === null) return;
  if (pending > hero.totalSteps && !hero.source.complete) {
    if (liveStarted || !hero.fetched) return;
    liveStarted = true;
    sceneHost.seek(liveStartFrame(hero.source));
    return;
  }
  sceneHost.seek(resolvePendingFrame(hero, hero.source));
}

function mountHero(): void {
  const gen = ++generation;
  const sel = selectionSnapshot();
  const inp = inputs;
  if (hero !== null) {
    hero.setCallbacks(null, null);
    hero.setLive(false);
  }
  hero = null;
  attached = false;
  liveStarted = false;
  if (container === null || inp === null || sel.envId === null) {
    setStageMeta({ ...EMPTY_META, policy: sel.policy });
    return;
  }
  const env = sel.envId;
  if (sel.policy !== inp.policy || inp.baselinePending) {
    setStageMeta({
      status: 'loading',
      note: `Loading the ${sel.policy} fleet`,
      noteDetail: null,
      policy: sel.policy,
      ticks: [],
      loadedSteps: 0,
    });
    return;
  }
  const run = inp.active;
  if (run === null) {
    if (sel.policy === 'oak') {
      setStageMeta({ ...EMPTY_META, policy: sel.policy });
      return;
    }
    setStageMeta({
      status: 'error',
      note: `No ${sel.policy} baseline to play`,
      noteDetail:
        inp.baselineId === undefined
          ? `no ${sel.policy} baseline for this fleet`
          : `could not load the ${sel.policy} baseline run ${inp.baselineId}`,
      policy: sel.policy,
      ticks: [],
    });
    return;
  }
  const seed = heroSeed(sel.selection);
  const reason =
    sel.policy === 'oak'
      ? degradedReason(run.recordings, env)
      : baselineReason(run.recordings, env, seed, sel.policy);
  if (reason !== null || seed === null) {
    setStageMeta({
      status: 'error',
      note: 'Nothing to replay here',
      noteDetail: reason ?? 'no seed to play',
      envName: nameOf(env),
      seed,
      policy: sel.policy,
      ticks: [],
    });
    return;
  }

  setStageMeta({
    status: 'ready',
    note: null,
    noteDetail: null,
    envName: nameOf(env),
    seed,
    promoted: sel.selection.promoted !== null,
    policy: sel.policy,
    frameCount: 1,
    totalSteps: 0,
    loadedSteps: 0,
    complete: false,
    ticks: [],
  });

  const entry = entryFor(parseCatalog(run.meta.envCatalog), env);
  const cacheKey = `${run.id}|${env}|${seed}`;
  let handle = heroCache.get(cacheKey);
  if (handle === undefined) {
    handle = loadHero({
      entry,
      runId: run.id,
      envId: env,
      seed,
      horizonSteps: run.meta.steps,
      live: inp.live,
      startedAt: run.meta.startedAt === undefined ? null : Date.parse(run.meta.startedAt),
    });
    heroCache.set(cacheKey, handle);
    for (const [oldKey, old] of heroCache) {
      if (heroCache.size <= HERO_CACHE_MAX) break;
      if (old === handle) continue;
      old.destroy();
      heroCache.delete(oldKey);
    }
  } else {
    heroCache.delete(cacheKey);
    heroCache.set(cacheKey, handle);
    handle.setLive(inp.live);
  }
  hero = handle;
  const heroHandle = handle;

  queueStage(async () => {
    if (container === null || gen !== generation) return;
    const attachSource = heroHandle.source.frameCount > 1 ? heroHandle.source : makeInitialSource(entry, env, seed);
    if (attachSource === heroHandle.source) {
      attached = true;
      liveStarted = heroHandle.fetched;
    }
    const at = resolvePendingFrame(heroHandle, attachSource);
    if (sceneHost === null) {
      const transport = transportSnapshot();
      appliedDebug = selectionSnapshot().debug;
      const host = await mountSceneHost(container, {
        envId: env,
        source: attachSource,
        at,
        speed: transport.speed,
        paused: !transport.playing,
        debug: appliedDebug,
        onFrame,
      });
      if (container === null) {
        host.destroy();
        return;
      }
      sceneHost = host;
    } else if (sceneHost.envId !== env) {
      await sceneHost.setEnv(env, attachSource, { at });
    } else {
      await sceneHost.setSource(attachSource, { at });
    }
    applyTransport();
    if (attached) publishHeroMeta(heroHandle);
  });

  heroHandle.setCallbacks(
    () => {
      if (container === null || gen !== generation || hero !== heroHandle) return;
      queueStage(async () => {
        if (container === null || gen !== generation || hero !== heroHandle || sceneHost === null) return;
        if (!attached) {
          attached = true;
          liveStarted = heroHandle.fetched;
          const at = resolvePendingFrame(heroHandle, heroHandle.source);
          await sceneHost.setSource(heroHandle.source, { at });
          applyTransport();
        } else {
          applyPendingSeek();
        }
        publishHeroMeta(heroHandle);
      });
    },
    (error) => {
      if (container === null || gen !== generation) return;
      setStageMeta({ status: 'error', note: 'Could not play this seed', noteDetail: error.message });
    },
  );
}

function ensureSubscribed(): void {
  if (subscribed) return;
  subscribed = true;
  unsubscribeSelection = subscribeSelection(retarget);
  unsubscribeTransport = subscribeTransport(applyTransport);
}

export function setTheaterInputs(next: TheaterInputs): void {
  inputs = next;
  hero?.setLive(next.live);
  retarget();
}

export function mountStage(node: HTMLElement): void {
  disposeQueued = false;
  if (container === node) return;
  if (container !== null) hardDetach();
  container = node;
  ensureSubscribed();
  targetKey = '';
  retarget();
}

export function unmountStage(): void {
  disposeQueued = true;
  queueMicrotask(() => {
    if (!disposeQueued) return;
    disposeQueued = false;
    hardDetach();
  });
}

function hardDetach(): void {
  generation++;
  unsubscribeSelection?.();
  unsubscribeTransport?.();
  unsubscribeSelection = null;
  unsubscribeTransport = null;
  subscribed = false;
  if (hero !== null) {
    hero.setCallbacks(null, null);
    hero.setLive(false);
    hero = null;
  }
  for (const cached of heroCache.values()) cached.destroy();
  heroCache.clear();
  sceneHost?.destroy();
  sceneHost = null;
  attached = false;
  container = null;
  targetKey = '';
  meta.set(EMPTY_META);
}

export function scrubToStep(step: number): void {
  const target = Math.max(0, Math.floor(step));
  if (sceneHost === null || hero === null || !attached) {
    setPendingSeek(target > 0 ? target : null);
    return;
  }
  const source = hero.source;
  const total = hero.totalSteps;
  if (target > total && !source.complete) {
    setPendingSeek(target);
    sceneHost.seek(liveStartFrame(source));
    return;
  }
  setPendingSeek(null);
  sceneHost.seek(clampPosition(frameFromStep(Math.min(target, total), total, source.frameCount), source.frameCount));
}

export function stepBySteps(delta: number): void {
  if (sceneHost === null || hero === null || !attached) return;
  setTransport({ playing: false });
  const current = stepFromFrame(sceneHost.clock.position(), hero.totalSteps, hero.source.frameCount);
  scrubToStep(current + delta);
}

export function currentStep(): number {
  if (sceneHost === null || hero === null || !attached) return selectionSnapshot().pendingSeekStep ?? 0;
  return stepFromFrame(sceneHost.clock.position(), hero.totalSteps, hero.source.frameCount);
}

export function seekToTick(tick: TimelineTick): void {
  if (sceneHost === null || hero === null || !attached) return;
  setPendingSeek(null);
  sceneHost.seek(clampPosition(seekTargetForTick(tick), hero.source.frameCount));
  setTransport({ playing: true });
}

export function scrubToFrame(frame: number): void {
  if (hero === null || !attached) return;
  scrubToStep(stepFromFrame(frame, hero.totalSteps, hero.source.frameCount));
}
