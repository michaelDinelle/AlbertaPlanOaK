
import * as THREE from 'three';
import { ENVS, envBySlug, type EnvId } from '@shared/envs';
import type { RecordingMeta, RecordingRef, RunDetail } from '@shared/types';
import { NO_EVENTS, lensRecording } from '@sim/decode';
import { parseActionsBin } from '@sim/record';
import { RECORD_V4_VERSION, parseActionsBinV4 } from '@sim/record4';
import { decoders } from '@sim/registry';
import type { EnvDecoder, EnvFrame, EnvGeometry, SimEvent } from '@sim/types';
import { sceneRegistry } from '../scenes/registry';
import { celebrationPulse, pulseChannel } from './celebrate';
import { configureRendererShadows } from './lighting';
import { buildComposer, type ComposerHandle } from './post';
import { TOKENS } from './tokens';
import type {
  FrameState,
  FrameUpdate,
  QualityTier,
  SceneContext,
  SceneFixture,
  SceneModule,
  SceneSize,
} from './types';

export const DEFAULT_HARNESS_RUN = 'frontend-demo-16env-20260803';

export const TIERS: readonly QualityTier[] = ['hero', 'medium', 'thumb'];
const SPEEDS: readonly number[] = [0.25, 0.5, 1, 2, 4];


export interface PlaybackClock {
  time(): number;
  position(): number;
  playing: boolean;
  speed: number;
  advance(dtReal: number): number;
  seek(position: number): void;
  readonly duration: number;
}

export function makePlaybackClock(frameCount: number, controlHz: number): PlaybackClock {
  const hz = controlHz > 0 ? controlHz : 8;
  const last = Math.max(0, frameCount - 1);
  let t = 0;
  const clock: PlaybackClock = {
    playing: true,
    speed: 1,
    duration: last / hz,
    time: () => t,
    position: () => t * hz,
    advance(dtReal) {
      if (clock.playing) {
        t += dtReal * clock.speed;
        if (t > clock.duration) t = clock.duration > 0 ? t % clock.duration : 0;
        if (t < 0) t = 0;
      }
      return t * hz;
    },
    seek(position) {
      const p = position < 0 ? 0 : position > last ? last : position;
      t = p / hz;
    },
  };
  return clock;
}


export interface HarnessSource {
  kind: 'recording' | 'fixture';
  envId: EnvId;
  seed: number;
  controlHz: number;
  discretePose: boolean;
  geometry: EnvGeometry;
  frameCount: number;
  frameAt(i: number): EnvFrame<unknown>;
  eventsAt(i: number): SimEvent[];
  eventEntries(): IterableIterator<[number, SimEvent[]]>;
  label: string;
  note?: string;
  replay?: { firstDisagreeStep: number | null; disagreeCount: number };
  nonFiniteRows?: number;
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { headers: { accept: 'application/json' }, ...(signal ? { signal } : {}) });
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return (await res.json()) as T;
}

export function firstRecordingFor(recordings: readonly RecordingRef[], envId: EnvId): RecordingRef | null {
  let best: RecordingRef | null = null;
  for (const rec of recordings) {
    if (rec.envId !== envId) continue;
    if (!best || rec.seed < best.seed) best = rec;
  }
  return best;
}

function fixtureFrame(step: number, fixture: SceneFixture | null, thumbPose: readonly number[]): EnvFrame<unknown> {
  const pose = new Float32Array(9);
  const source = fixture?.pose ?? thumbPose;
  for (let i = 0; i < 9; i++) pose[i] = source[i] ?? 0;
  const ctrl = new Float32Array(6);
  if (fixture?.ctrl) for (let i = 0; i < 6; i++) ctrl[i] = fixture.ctrl[i] ?? 0;
  return { step, ctrl, done: 0, pose, state: fixture?.derived ?? null };
}

function placeholderGeometry(envId: EnvId): EnvGeometry {
  const info = ENVS[envId];
  if (info.category === 'grid') {
    const w = 11;
    const h = 11;
    const walls = new Uint8Array(w * h);
    for (let x = 0; x < w; x++) {
      walls[x] = 1;
      walls[(h - 1) * w + x] = 1;
    }
    for (let y = 0; y < h; y++) {
      walls[y * w] = 1;
      walls[y * w + w - 1] = 1;
    }
    return { kind: 'grid', grid: { w, h, walls, goal: { x: w - 2, y: h - 2 } } };
  }
  if (info.category === 'classic') {
    return { kind: 'classic', classic: { anchors: { origin: [0, 0, 0] } } };
  }
  return { kind: 'cheetah' };
}

export interface LoadSourceOptions {
  runId?: string;
  maxSteps?: number;
  signal?: AbortSignal;
  fixtures?: readonly SceneFixture[];
  thumbPose?: readonly number[];
}

export async function loadHarnessSource(envId: EnvId, opts: LoadSourceOptions = {}): Promise<HarnessSource> {
  const info = ENVS[envId];
  const runId = opts.runId ?? DEFAULT_HARNESS_RUN;
  const decoder = decoders[envId] as EnvDecoder<unknown, unknown> | undefined;

  const fallback = (note: string, geometry?: EnvGeometry, seed = 0): HarnessSource => {
    const fixtures = opts.fixtures ?? [];
    const thumbPose = opts.thumbPose ?? [];
    const frames: EnvFrame<unknown>[] = [];
    const count = Math.max(2, fixtures.length);
    for (let i = 0; i < count; i++) {
      frames.push(fixtureFrame(i - 1, fixtures[i % Math.max(1, fixtures.length)] ?? null, thumbPose));
    }
    let resolved = geometry;
    if (!resolved && decoder) {
      try {
        const st = decoder.initStatic(seed);
        resolved = decoder.geometry(st, seed);
      } catch {
        resolved = undefined;
      }
    }
    return {
      kind: 'fixture',
      envId,
      seed,
      controlHz: Math.max(1, Math.min(8, info.controlHz)),
      discretePose: info.discretePose,
      geometry: resolved ?? placeholderGeometry(envId),
      frameCount: frames.length,
      frameAt: (i) => frames[Math.max(0, Math.min(frames.length - 1, Math.floor(i)))] as EnvFrame<unknown>,
      eventsAt: () => NO_EVENTS,
      eventEntries: () => new Map<number, SimEvent[]>().entries(),
      label: `fixtures ×${count}`,
      note,
    };
  };

  if (!decoder) return fallback(`no @sim decoder registered for ${envId} — fixture loop`);

  let detail: RunDetail;
  try {
    detail = await getJson<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`, opts.signal);
  } catch (cause) {
    return fallback(`run ${runId} unavailable (${String(cause)}) — fixture loop`);
  }

  const ref = firstRecordingFor(detail.recordings ?? [], envId);
  if (!ref) return fallback(`no ${envId} recording in ${runId} — fixture loop`);

  const base = `/api/runs/${encodeURIComponent(runId)}/recordings/${encodeURIComponent(envId)}/${ref.seed}`;
  try {
    const meta = await getJson<RecordingMeta>(`${base}/meta.json`, opts.signal);
    const res = await fetch(`${base}/actions.bin`, opts.signal ? { signal: opts.signal } : {});
    if (!res.ok) throw new Error(`${res.status} actions.bin`);
    const buffer = await res.arrayBuffer();
    const isV4 = meta.recordVersion === RECORD_V4_VERSION;
    const raw = isV4 ? await parseActionsBinV4(buffer, meta) : parseActionsBin(buffer, meta);
    const lens = lensRecording(raw, decoder);
    return {
      kind: 'recording',
      envId,
      seed: lens.seed,
      controlHz: lens.controlHz,
      discretePose: lens.discretePose,
      geometry: lens.geometry,
      frameCount: lens.frameCount,
      frameAt: lens.frameAt,
      eventsAt: lens.eventsAt,
      eventEntries: lens.eventEntries,
      label: `${runId} · seed ${lens.seed} · ${lens.steps} steps`,
      ...(lens.replay ? { replay: lens.replay } : {}),
      nonFiniteRows: lens.nonFiniteRows,
    };
  } catch (cause) {
    return fallback(`decode failed for ${envId} seed ${ref.seed} (${String(cause)}) — fixture loop`, undefined, ref.seed);
  }
}

export async function loadSceneFixtures(slug: string): Promise<SceneFixture[]> {
  try {
    const modules = import.meta.glob('../scenes/*/fixtures.ts');
    const key = Object.keys(modules).find((k) => k.includes(`/${slug}/`));
    if (key === undefined) return [];
    const loader = modules[key];
    if (!loader) return [];
    const mod = (await loader()) as { fixtures?: SceneFixture[]; default?: SceneFixture[] };
    const list = mod.fixtures ?? mod.default;
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
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

export function sampleFrame(source: HarnessSource, position: number, out: FrameState): void {
  const n = source.frameCount;
  if (n === 0) return;
  const last = n - 1;
  const p = position < 0 ? 0 : position > last ? last : position;
  const i = Math.min(last, Math.floor(p));
  const frame = source.frameAt(i);
  const next = source.frameAt(Math.min(last, i + 1));
  const alpha = source.discretePose ? 0 : p - i;

  if (alpha > 0 && next !== frame) {
    for (let k = 0; k < 9; k++) {
      const a = frame.pose[k] ?? 0;
      const b = next.pose[k] ?? 0;
      out.pose[k] = a + (b - a) * alpha;
    }
  } else {
    out.pose.set(frame.pose.subarray(0, 9));
  }
  out.ctrl.set(frame.ctrl.subarray(0, 6));
  out.done = frame.done !== 0;
  out.step = frame.step;
  out.simTime = frame.step / (source.controlHz > 0 ? source.controlHz : 1);
  out.derived = frame.state;
}

export const MAX_CATCHUP_FRAMES = 12;


export interface HarnessOptions {
  runId?: string;
  tier?: QualityTier;
  maxSteps?: number;
  fixtures?: readonly SceneFixture[];
  at?: number;
  paused?: boolean;
  debug?: boolean;
}

export interface HarnessHandle {
  destroy(): void;
}

const CSS = `
.oak-harness { position:absolute; inset:0; background:#05070c; overflow:hidden; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; color:#cfe0f5; }
.oak-harness canvas { position:absolute; inset:0; width:100%; height:100%; display:block; }
.oak-harness .oak-hud { position:absolute; padding:8px 10px; background:rgba(6,10,18,.72); border:1px solid rgba(120,160,210,.18); border-radius:6px; backdrop-filter:blur(6px); pointer-events:none; }
.oak-harness .oak-hud.tl { top:12px; left:12px; max-width:46ch; }
.oak-harness .oak-hud.tr { top:12px; right:12px; text-align:right; pointer-events:auto; }
.oak-harness .oak-title { color:#eaf3ff; font-size:13px; letter-spacing:.02em; }
.oak-harness .oak-dim { color:#7e93ad; }
.oak-harness .oak-warn { color:#ffbe5c; }
.oak-harness .oak-transport { position:absolute; left:12px; right:12px; bottom:12px; display:flex; align-items:center; gap:10px; padding:8px 10px; background:rgba(6,10,18,.78); border:1px solid rgba(120,160,210,.18); border-radius:6px; }
.oak-harness button, .oak-harness select { font:inherit; color:#cfe0f5; background:#101724; border:1px solid rgba(120,160,210,.28); border-radius:4px; padding:3px 9px; cursor:pointer; }
.oak-harness button:hover, .oak-harness select:hover { border-color:rgba(150,200,255,.55); }
.oak-harness input[type=range] { flex:1; accent-color:#5ec1ff; min-width:80px; }
.oak-harness .oak-readout { min-width:20ch; }
.oak-harness .oak-err { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; padding:32px; text-align:center; color:#ffbe5c; }
`;

function ensureStyle(): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById('oak-harness-style')) return;
  const style = document.createElement('style');
  style.id = 'oak-harness-style';
  style.textContent = CSS;
  document.head.appendChild(style);
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export async function mountHarness(
  container: HTMLElement,
  slug: string,
  opts: HarnessOptions = {},
): Promise<HarnessHandle> {
  ensureStyle();
  const root = el('div', 'oak-harness');
  container.appendChild(root);

  const info = envBySlug(slug);
  if (!info) {
    root.appendChild(el('div', 'oak-err', `unknown scene slug "${slug}" — expected one of the 16 env slugs`));
    return { destroy: () => root.remove() };
  }

  const loader = sceneRegistry[info.envId];
  if (!loader) {
    root.appendChild(
      el(
        'div',
        'oak-err',
        `no scene registered for ${info.envId} yet.\nAdd it to src/scenes/registry.ts, then reload #/harness/${slug}.`,
      ),
    );
    return { destroy: () => root.remove() };
  }

  const abort = new AbortController();
  let destroyed = false;

  const [fixtures, mod] = await Promise.all([
    opts.fixtures ? Promise.resolve([...opts.fixtures]) : loadSceneFixtures(slug),
    loader(),
  ]);
  if (destroyed) return { destroy: () => root.remove() };

  const factory = mod.default;
  let pendingModule: SceneModule | null = factory();
  const probeManifest = pendingModule.manifest;
  const source = await loadHarnessSource(info.envId, {
    ...(opts.runId !== undefined ? { runId: opts.runId } : {}),
    ...(opts.maxSteps !== undefined ? { maxSteps: opts.maxSteps } : {}),
    fixtures,
    thumbPose: probeManifest.thumbFrame.pose,
    signal: abort.signal,
  });
  if (destroyed) {
    pendingModule?.dispose();
    pendingModule = null;
    return { destroy: () => root.remove() };
  }

  const canvas = el('canvas');
  root.appendChild(canvas);
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance' });
  } catch (cause) {
    root.appendChild(el('div', 'oak-err', `WebGL unavailable: ${String(cause)}`));
    return { destroy: () => root.remove() };
  }
  renderer.setClearColor(0x05070c, 1);

  const hudLeft = el('div', 'oak-hud tl');
  const titleLine = el('div', 'oak-title');
  const sourceLine = el('div', 'oak-dim');
  const noteLine = el('div', 'oak-warn');
  hudLeft.append(titleLine, sourceLine, noteLine);
  root.appendChild(hudLeft);

  const hudRight = el('div', 'oak-hud tr');
  const statsLine = el('div');
  const tierSelect = el('select');
  for (const t of TIERS) {
    const option = el('option');
    option.value = t;
    option.textContent = t;
    tierSelect.appendChild(option);
  }
  hudRight.append(statsLine, tierSelect);
  root.appendChild(hudRight);

  const transport = el('div', 'oak-transport');
  const playButton = el('button', undefined, 'pause');
  const speedSelect = el('select');
  for (const s of SPEEDS) {
    const option = el('option');
    option.value = String(s);
    option.textContent = `${s}×`;
    speedSelect.appendChild(option);
  }
  speedSelect.value = '1';
  const scrub = el('input');
  scrub.type = 'range';
  scrub.min = '0';
  scrub.max = String(Math.max(1, source.frameCount - 1));
  scrub.step = '0.01';
  scrub.value = '0';
  const readout = el('div', 'oak-readout');
  const hint = el('div', 'oak-dim', 'g: ground truth');
  transport.append(playButton, speedSelect, scrub, readout, hint);
  root.appendChild(transport);

  titleLine.textContent = `${probeManifest.title} — ${info.name}`;
  sourceLine.textContent = `${source.kind} · ${source.label}`;
  noteLine.textContent = source.note ?? '';

  let tier: QualityTier = opts.tier ?? 'hero';
  tierSelect.value = tier;
  let scene: SceneModule | null = null;
  let post: ComposerHandle | null = null;
  let debugOn = false;
  renderer.info.autoReset = false;
  let sceneCalls = 0;
  let sceneTriangles = 0;
  let statsPrimed = false;

  const clock = makePlaybackClock(source.frameCount, source.controlHz);
  const state = makeFrameState();
  const previous = makeFrameState();
  let hasPrevious = false;
  let lastFrameIndex = -1;
  const eventBuffer: SimEvent[] = [];
  const frameUpdate: FrameUpdate = {
    state,
    prev: null,
    events: eventBuffer,
    dtReal: 0,
    scrubbed: true,
  };

  const sizeOf = (): SceneSize => ({
    width: Math.max(1, root.clientWidth || 1),
    height: Math.max(1, root.clientHeight || 1),
    dpr: Math.min(tier === 'thumb' ? 1 : tier === 'medium' ? 1.5 : 2, window.devicePixelRatio || 1),
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
    scene?.dispose();
    post?.dispose();
    scene = null;
    post = null;
    pulseChannel.reset();

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
    post = buildComposer(renderer, module.scene, module.camera, module.manifest.postConfig, tier, size);
    module.setDebug(debugOn);
    applySize();
    hasPrevious = false;
    lastFrameIndex = -1;
    statsPrimed = false;
  };

  if (opts.debug) debugOn = true;

  await buildScene();

  if (opts.at !== undefined && Number.isFinite(opts.at)) {
    const target = Math.max(0, Math.min(source.frameCount - 1, opts.at));
    clock.seek(target);
    scrub.value = String(target);
  }
  if (opts.paused) clock.playing = false;

  const syncPlayButton = (): void => {
    playButton.textContent = clock.playing ? 'pause' : 'play';
  };
  syncPlayButton();
  playButton.addEventListener('click', () => {
    clock.playing = !clock.playing;
    syncPlayButton();
  });
  speedSelect.addEventListener('change', () => {
    clock.speed = Number(speedSelect.value) || 1;
  });
  let scrubbing = false;
  scrub.addEventListener('pointerdown', () => {
    scrubbing = true;
  });
  scrub.addEventListener('pointerup', () => {
    scrubbing = false;
  });
  scrub.addEventListener('input', () => {
    clock.seek(Number(scrub.value));
  });
  tierSelect.addEventListener('change', () => {
    const next = tierSelect.value as QualityTier;
    if (next === tier) return;
    tier = next;
    void buildScene();
  });

  const onKey = (e: KeyboardEvent): void => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    if (e.key === 'g' || e.key === 'G') {
      debugOn = !debugOn;
      scene?.setDebug(debugOn);
      hint.textContent = debugOn ? 'g: ground truth ON' : 'g: ground truth';
    } else if (e.key === ' ') {
      e.preventDefault();
      clock.playing = !clock.playing;
      syncPlayButton();
    }
  };
  window.addEventListener('keydown', onKey);

  const onResize = (): void => applySize();
  window.addEventListener('resize', onResize);

  let raf = 0;
  let lastWall = 0;
  let fps = 0;
  let statsTimer = 0;

  const tick = (now: number): void => {
    raf = requestAnimationFrame(tick);
    if (!scene || !post) return;
    const dtReal = lastWall === 0 ? 0 : Math.min(0.1, (now - lastWall) / 1000);
    lastWall = now;

    const position = clock.advance(dtReal);
    if (!scrubbing) scrub.value = String(position);

    if (hasPrevious) copyFrameState(state, previous);
    sampleFrame(source, position, state);

    const index = Math.min(source.frameCount - 1, Math.max(0, Math.floor(position)));
    const delta = index - lastFrameIndex;
    const sequential = hasPrevious && delta >= 0 && delta <= MAX_CATCHUP_FRAMES;
    eventBuffer.length = 0;
    if (sequential) {
      for (let i = lastFrameIndex + 1; i <= index; i++) {
        const evs = source.eventsAt(i);
        if (evs !== NO_EVENTS) for (const ev of evs) eventBuffer.push(ev);
      }
    }
    frameUpdate.prev = hasPrevious ? previous : null;
    frameUpdate.dtReal = dtReal;
    frameUpdate.scrubbed = !sequential;
    scene.update(frameUpdate);
    lastFrameIndex = index;
    hasPrevious = true;

    renderer.info.reset();
    post.setBloomPulse(celebrationPulse(scene.scene).bloom);
    post.render(dtReal);

    statsTimer += dtReal;
    fps = dtReal > 0 ? (fps > 0 ? fps * 0.9 + (1 / dtReal) * 0.1 : 1 / dtReal) : fps;
    if (!statsPrimed || statsTimer > 0.25) {
      statsPrimed = true;
      statsTimer = 0;
      const budget = scene.manifest.budget;
      const over = sceneCalls > budget.drawCalls || sceneTriangles > budget.triangles ? ' ⚠' : '';
      statsLine.textContent =
        `${fps > 0 ? fps.toFixed(0) : '--'} fps · ${sceneCalls}/${budget.drawCalls} draws · ` +
        `${(sceneTriangles / 1000).toFixed(1)}k/${(budget.triangles / 1000).toFixed(0)}k tris` +
        `${over} · ${renderer.info.render.calls} total`;
      readout.textContent = `step ${state.step} · ${state.simTime.toFixed(2)}s · f${index}/${source.frameCount - 1}`;
    }
  };
  raf = requestAnimationFrame(tick);
  syncPlayButton();

  return {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      abort.abort();
      cancelAnimationFrame(raf);
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', onResize);
      scene?.dispose();
      post?.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      root.remove();
    },
  };
}
