// kit/harness — the artisan dev page.
//
// One scene, one lane, full transport, on-screen budget readout, ground-truth toggle.
// This is where a scene is BUILT: the theater renders the same SceneModule through the
// same host contract, so anything that looks right here looks right there.
//
// Host responsibilities it implements (kit/types.ts ownership boundary):
//   * the ONE WebGLRenderer + canvas + resize
//   * the playback clock and transport (play / pause / speed / scrub)
//   * the EffectComposer chain via kit/post (scenes only parameterize it)
//   * the celebration bloom pulse, resolved PER SCENE via `celebrationPulse(scene)` —
//     pools own their channel now, so a scene must attach its pool
//     (`attachCelebrations` / `celebrationsFor`) for the host to see its beats
//   * quality-tier selection (a switch re-inits the scene, as the contract requires)
//   * the debug toggle (`g`)
//   * frame assembly: pose interpolation for continuous envs, snap for grids, event
//     delivery ONLY on sequential steps, `scrubbed` on every jump
//
// Data: the same-origin API's recordings for the scene's env (first available seed),
// parsed by @sim/record and decoded by @sim/decode + @sim/registry. If the env has no
// decoder registered yet, the harness falls back to a static fixture loop built from the
// scene manifest's thumbFrame plus `scenes/<slug>/fixtures.ts` if it exists, so art work
// is never blocked on sim-core.

import * as THREE from 'three';
import { ENVS, envBySlug, type EnvId } from '@shared/envs';
import type { RecordingMeta, RecordingRef, RunDetail } from '@shared/types';
import { decodeRecording } from '@sim/decode';
import { RECORD_BYTES, parseActionsBin } from '@sim/record';
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

/** The demo run every artisan page reads from. */
export const DEFAULT_HARNESS_RUN = 'frontend-demo-16env-20260803';

/** Frames decoded for the dev page. A 100k-step lane is far more than art work needs. */
export const DEFAULT_MAX_STEPS = 20000;

export const TIERS: readonly QualityTier[] = ['hero', 'medium', 'thumb'];
const SPEEDS: readonly number[] = [0.25, 0.5, 1, 2, 4];

// ---------------------------------------------------------------------------
// playback clock
// ---------------------------------------------------------------------------

export interface PlaybackClock {
  /** playback seconds — SceneContext.time() */
  time(): number;
  /** frame position in [0, frameCount-1] */
  position(): number;
  playing: boolean;
  speed: number;
  /** advance by wall seconds; returns the new position */
  advance(dtReal: number): number;
  /** jump to a frame position (a scrub) */
  seek(position: number): void;
  readonly duration: number;
}

/** Frame position <-> playback seconds is exactly `controlHz`. */
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

// ---------------------------------------------------------------------------
// data source
// ---------------------------------------------------------------------------

export interface HarnessSource {
  kind: 'recording' | 'fixture';
  envId: EnvId;
  seed: number;
  controlHz: number;
  discretePose: boolean;
  geometry: EnvGeometry;
  frames: EnvFrame<unknown>[];
  events: SimEvent[][];
  /** short human label for the HUD */
  label: string;
  /** why we fell back, when we did */
  note?: string;
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { headers: { accept: 'application/json' }, ...(signal ? { signal } : {}) });
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return (await res.json()) as T;
}

/** The first recorded lane for an env in a run, or null. */
export function firstRecordingFor(recordings: readonly RecordingRef[], envId: EnvId): RecordingRef | null {
  let best: RecordingRef | null = null;
  for (const rec of recordings) {
    if (rec.envId !== envId) continue;
    if (!best || rec.seed < best.seed) best = rec;
  }
  return best;
}

/** A synthetic frame the fixture fallback plays back. */
function fixtureFrame(step: number, fixture: SceneFixture | null, thumbPose: readonly number[]): EnvFrame<unknown> {
  const pose = new Float32Array(9);
  const source = fixture?.pose ?? thumbPose;
  for (let i = 0; i < 9; i++) pose[i] = source[i] ?? 0;
  const ctrl = new Float32Array(6);
  if (fixture?.ctrl) for (let i = 0; i < 6; i++) ctrl[i] = fixture.ctrl[i] ?? 0;
  return { step, ctrl, done: 0, pose, state: fixture?.derived ?? null };
}

/** Placeholder geometry so art work can proceed before a decoder lands. */
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
  /** fixtures for the fallback loop */
  fixtures?: readonly SceneFixture[];
  /** manifest thumbFrame pose for the fallback loop */
  thumbPose?: readonly number[];
}

/**
 * Load one lane of real experience for an env, or fall back to a fixture loop.
 * Never throws for a missing run / missing decoder — the dev page must always mount.
 */
export async function loadHarnessSource(envId: EnvId, opts: LoadSourceOptions = {}): Promise<HarnessSource> {
  const info = ENVS[envId];
  const runId = opts.runId ?? DEFAULT_HARNESS_RUN;
  const decoder = decoders[envId] as EnvDecoder<unknown, unknown> | undefined;

  const fallback = (note: string, geometry?: EnvGeometry, seed = 0): HarnessSource => {
    const fixtures = opts.fixtures ?? [];
    const thumbPose = opts.thumbPose ?? [];
    const frames: EnvFrame<unknown>[] = [];
    const events: SimEvent[][] = [];
    const count = Math.max(2, fixtures.length);
    for (let i = 0; i < count; i++) {
      frames.push(fixtureFrame(i - 1, fixtures[i % Math.max(1, fixtures.length)] ?? null, thumbPose));
      events.push([]);
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
      frames,
      events,
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
  const maxSteps = opts.maxSteps ?? DEFAULT_MAX_STEPS;
  try {
    const meta = await getJson<RecordingMeta>(`${base}/meta.json`, opts.signal);
    // A 100k-step lane is 6.4 MB; the dev page decodes `maxSteps` of it. The API serves
    // byte ranges, so ask for exactly the rows we are going to decode. A server that
    // ignores the header just sends the whole file, which still parses.
    const wantSteps = Math.max(1, Math.min(ref.steps || maxSteps, maxSteps));
    const res = await fetch(`${base}/actions.bin`, {
      headers: { range: `bytes=0-${wantSteps * RECORD_BYTES - 1}` },
      ...(opts.signal ? { signal: opts.signal } : {}),
    });
    if (!res.ok && res.status !== 206) throw new Error(`${res.status} actions.bin`);
    const raw = parseActionsBin(await res.arrayBuffer(), meta);
    const decoded = decodeRecording(raw, decoder, { maxSteps });
    return {
      kind: 'recording',
      envId,
      seed: decoded.seed,
      controlHz: decoded.controlHz,
      discretePose: decoded.discretePose,
      geometry: decoded.geometry,
      frames: decoded.frames,
      events: decoded.events,
      label: `${runId} · seed ${decoded.seed} · ${decoded.frames.length - 1} steps`,
    };
  } catch (cause) {
    return fallback(`decode failed for ${envId} seed ${ref.seed} (${String(cause)}) — fixture loop`, undefined, ref.seed);
  }
}

/** `scenes/<slug>/fixtures.ts` if the artisan has written one yet. */
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

// ---------------------------------------------------------------------------
// frame assembly
// ---------------------------------------------------------------------------

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

/**
 * Fill `out` from the source at a fractional frame position.
 * Continuous envs interpolate the 9-float pose; `discretePose` envs snap (a grid agent
 * is never between cells — that is what HopMotion is for).
 */
export function sampleFrame(source: HarnessSource, position: number, out: FrameState): void {
  const n = source.frames.length;
  if (n === 0) return;
  const last = n - 1;
  const p = position < 0 ? 0 : position > last ? last : position;
  const i = Math.min(last, Math.floor(p));
  const frame = source.frames[i] as EnvFrame<unknown>;
  const next = source.frames[Math.min(last, i + 1)] as EnvFrame<unknown>;
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

/** Max frames we will walk in one tick before declaring the jump a scrub. */
export const MAX_CATCHUP_FRAMES = 12;

// ---------------------------------------------------------------------------
// the page
// ---------------------------------------------------------------------------

export interface HarnessOptions {
  runId?: string;
  /** initial tier (default hero) */
  tier?: QualityTier;
  maxSteps?: number;
  /** override the fixture list (tests / previews) */
  fixtures?: readonly SceneFixture[];
  /** start scrubbed to this frame position (deterministic stills / headless screenshots) */
  at?: number;
  /** start paused (deterministic stills / headless screenshots) */
  paused?: boolean;
  /** start with the ground-truth overlay on */
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

/**
 * Mount the artisan page for one scene slug.
 * Route wiring lives in the router (orchestrator-owned); `views/harness.ts` is the thin
 * ViewMount adapter over this function.
 */
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

  // ---- data + scene factory in parallel ----
  const [fixtures, mod] = await Promise.all([
    opts.fixtures ? Promise.resolve([...opts.fixtures]) : loadSceneFixtures(slug),
    loader(),
  ]);
  if (destroyed) return { destroy: () => root.remove() };

  const factory = mod.default;
  // Build ONE module up front for its manifest; buildScene() consumes this same
  // instance on its first call, so nothing is constructed and thrown away.
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

  // ---- renderer + canvas (host-owned, one for the page's whole life) ----
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

  // ---- HUD ----
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
  scrub.max = String(Math.max(1, source.frames.length - 1));
  scrub.step = '0.01';
  scrub.value = '0';
  const readout = el('div', 'oak-readout');
  const hint = el('div', 'oak-dim', 'g: ground truth');
  transport.append(playButton, speedSelect, scrub, readout, hint);
  root.appendChild(transport);

  titleLine.textContent = `${probeManifest.title} — ${info.name}`;
  sourceLine.textContent = `${source.kind} · ${source.label}`;
  noteLine.textContent = source.note ?? '';

  // ---- mutable per-tier state ----
  let tier: QualityTier = opts.tier ?? 'hero';
  tierSelect.value = tier;
  let scene: SceneModule | null = null;
  let post: ComposerHandle | null = null;
  let debugOn = false;
  // The manifest budget is a SCENE budget, but a composer chain issues ~15 more draws
  // after the scene is done. `info.autoReset` would also reset the counters at every
  // pass, leaving only the last one. So: reset once per frame, and latch the counts the
  // instant the scene finishes rendering (Scene.onAfterRender fires after its draws are
  // submitted, shadow pass included).
  renderer.info.autoReset = false;
  let sceneCalls = 0;
  let sceneTriangles = 0;
  // The stats line is written on a 0.25 s cadence, which is right for a live page and
  // wrong for a headless one: `--virtual-time-budget` screenshots only ever tick rAF a
  // couple of times, so the gate never opened and EVERY review still shipped with an
  // empty HUD. The first PRESENTED frame now writes the line immediately (the counters
  // are latched by then — `Scene.onAfterRender` fired inside `post.render`), and the
  // cadence takes over from there. Reset on every rebuild, so a tier switch reports its
  // own numbers at once.
  let statsPrimed = false;

  const clock = makePlaybackClock(source.frames.length, source.controlHz);
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

  // Scene builds are async (module.init awaits textures/geometry), so two quick tier
  // switches can interleave: build A is still awaiting init when build B disposes and
  // starts over, and A then installs itself over B. A generation counter makes the
  // superseded build dispose what it made and return.
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
    // chain, never clobber: the scene owns its THREE.Scene and may use this hook itself
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
    // force a snap on the first frame after any rebuild
    hasPrevious = false;
    lastFrameIndex = -1;
    // a tier switch is a new scene with new counts: show them on its first frame too
    statsPrimed = false;
  };

  // Deterministic-still options (?at=&pause=1&debug=1 — headless screenshot loop).
  // debugOn is applied by buildScene itself (it restores overlay state on every rebuild).
  if (opts.debug) debugOn = true;

  await buildScene();

  if (opts.at !== undefined && Number.isFinite(opts.at)) {
    const target = Math.max(0, Math.min(source.frames.length - 1, opts.at));
    clock.seek(target);
    scrub.value = String(target);
  }
  if (opts.paused) clock.playing = false;

  // ---- transport wiring ----
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

  // ---- the loop ----
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

    const index = Math.min(source.frames.length - 1, Math.max(0, Math.floor(position)));
    const delta = index - lastFrameIndex;
    const sequential = hasPrevious && delta >= 0 && delta <= MAX_CATCHUP_FRAMES;
    eventBuffer.length = 0;
    if (sequential) {
      for (let i = lastFrameIndex + 1; i <= index; i++) {
        const evs = source.events[i];
        if (evs) for (const ev of evs) eventBuffer.push(ev);
      }
    }
    frameUpdate.prev = hasPrevious ? previous : null;
    frameUpdate.dtReal = dtReal;
    frameUpdate.scrubbed = !sequential;
    scene.update(frameUpdate);
    lastFrameIndex = index;
    hasPrevious = true;

    renderer.info.reset();
    // the scene's OWN channel (celebrate pools no longer share a global one);
    // falls back to the module channel for a scene that attached no pool
    post.setBloomPulse(celebrationPulse(scene.scene).bloom);
    post.render(dtReal);

    // HUD
    statsTimer += dtReal;
    // seed the EMA with the first real sample instead of ramping up from zero — the ramp
    // made the first second of every session report an fps nobody was running at
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
      readout.textContent = `step ${state.step} · ${state.simTime.toFixed(2)}s · f${index}/${source.frames.length - 1}`;
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
