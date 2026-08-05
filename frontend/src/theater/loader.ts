// theater/loader — the Theater's data layer.
//
// Everything here is pure logic plus one network function, so the picker, the
// deep link, the progress bar, the event timeline and the damage banner are all
// unit-testable without a DOM or a GPU.
//
// It deliberately does NOT call `kit/harness.loadHarnessSource`: that helper
// picks "the first recorded lane for an env" and swallows failures into a
// fixture loop, which is exactly right for the artisan dev page and exactly
// wrong for a hero view addressed by (run, env, seed). So this module reuses the
// layer underneath it — the same `api` + `@sim/record` + `@sim/decode` path the
// harness uses — and produces the very same `HarnessSource` shape, so
// `kit/harness.sampleFrame` consumes it unchanged.
//
// The one thing added on top: a STREAMED download. A 100k-step lane is 6.4 MB;
// `Response.arrayBuffer()` gives no progress, so the body is read chunk by
// chunk against `content-length` and the caller gets real bytes.

import { ENVS, ENV_IDS, isEnvId, type EnvId } from '@shared/envs';
import type { RecordingMeta, RecordingRef, RunIndexEntry } from '@shared/types';
import { DEFAULT_HARNESS_RUN, type HarnessSource } from '@kit/harness';
import type { QualityTier } from '@kit/types';
import { eventTimeline, lensRecording, type RecordingLens } from '@sim/decode';
import { RECORD_BYTES, parseActionsBin } from '@sim/record';
import {
  FILE_HEADER_BYTES as RECORD_V4_HEADER_BYTES,
  RECORD_V4_VERSION,
  chunkTableFromMeta,
  parseActionsBinV4,
  planPrefixV4,
} from '@sim/record4';
import { decoders } from '@sim/registry';
import type { DecodedRecording, EnvDecoder, SimEvent } from '@sim/types';
import { sortRuns } from '../state';

/** The demo run the picker highlights, shared with the artisan page. */
export const DEFAULT_THEATER_RUN = DEFAULT_HARNESS_RUN;

/**
 * Frames decoded for one visit. A lane is 100k steps ≈ 6.4 MB ≈ 40 ms of decode
 * (measured), so the whole thing is affordable; the cap only exists so a future
 * 1M-step recording cannot silently allocate a gigabyte of frame objects.
 */

const TIER_NAMES: readonly string[] = ['hero', 'medium', 'thumb'];

// ---------------------------------------------------------------------------
// the deep link — #/theater/:id/:envId/:seed
// ---------------------------------------------------------------------------

export interface TheaterTarget {
  runId: string;
  envId: EnvId;
  seed: number;
}

/**
 * Router params -> a target, or null for the picker.
 *
 * Strict on purpose: a bad env id or a non-integer seed is a broken link, and
 * silently "correcting" it would play a lane the URL never asked for.
 */
export function parseTheaterParams(params: Readonly<Record<string, string>>): TheaterTarget | null {
  const runId = params['id'];
  const envId = params['envId'];
  const rawSeed = params['seed'];
  if (!runId || !envId || rawSeed === undefined) return null;
  if (!isEnvId(envId)) return null;
  if (!/^\d+$/.test(rawSeed.trim())) return null;
  const seed = Number(rawSeed.trim());
  if (!Number.isSafeInteger(seed) || seed < 0) return null;
  return { runId, envId, seed };
}

/** The canonical link shape — `views/run.ts` links here, the picker links here. */
export function theaterHref(target: TheaterTarget): string {
  return `#/theater/${encodeURIComponent(target.runId)}/${encodeURIComponent(target.envId)}/${target.seed}`;
}

export interface TheaterQuery {
  tier?: QualityTier;
  steps?: number;
  at?: number;
  paused?: boolean;
  debug?: boolean;
  speed?: number;
}

/**
 * `#/theater/<run>/<env>/<seed>?tier=medium&at=1200&pause=1&debug=1&steps=4000`
 * — the deterministic-still switches, same vocabulary as the artisan page.
 */
export function parseTheaterQuery(hash: string): TheaterQuery {
  const out: TheaterQuery = {};
  const q = hash.indexOf('?');
  if (q < 0) return out;
  const params = new URLSearchParams(hash.slice(q + 1));

  const tier = params.get('tier');
  if (tier && TIER_NAMES.includes(tier)) out.tier = tier as QualityTier;

  const steps = Number(params.get('steps'));
  if (Number.isFinite(steps) && steps > 0) out.steps = Math.floor(steps);

  const at = Number(params.get('at'));
  if (Number.isFinite(at) && at >= 0) out.at = at;

  const speed = Number(params.get('speed'));
  if (Number.isFinite(speed) && speed > 0) out.speed = speed;

  if (params.get('pause') === '1') out.paused = true;
  if (params.get('debug') === '1') out.debug = true;
  return out;
}

// ---------------------------------------------------------------------------
// the picker
// ---------------------------------------------------------------------------

/**
 * Runs the Theater can actually play, in picker order: the demo run first (it
 * is the one with all 16 envs recorded), then the gallery's own ranking.
 */
export function pickerRuns(
  entries: readonly RunIndexEntry[],
  featuredId: string = DEFAULT_THEATER_RUN,
): RunIndexEntry[] {
  const playable = entries.filter((entry) => entry.hasRecordings);
  const ordered = sortRuns(playable);
  const featured = ordered.filter((entry) => entry.id === featuredId);
  if (featured.length === 0) return ordered;
  return [...featured, ...ordered.filter((entry) => entry.id !== featuredId)];
}

export interface EnvChip {
  envId: EnvId;
  name: string;
  slug: string;
  /** recorded lanes for this env in this run */
  count: number;
  seeds: number[];
}

/** One chip per env WITH a recording, in the canonical env order (never sorted by count). */
export function envChips(recordings: readonly RecordingRef[]): EnvChip[] {
  const bySeed = new Map<EnvId, Set<number>>();
  for (const rec of recordings) {
    if (!isEnvId(rec.envId)) continue;
    let set = bySeed.get(rec.envId);
    if (!set) {
      set = new Set<number>();
      bySeed.set(rec.envId, set);
    }
    set.add(rec.seed);
  }
  const out: EnvChip[] = [];
  for (const envId of ENV_IDS) {
    const set = bySeed.get(envId);
    if (!set || set.size === 0) continue;
    const info = ENVS[envId];
    const seeds = [...set].sort((a, b) => a - b);
    out.push({ envId, name: info.name, slug: info.slug, count: seeds.length, seeds });
  }
  return out;
}

/** Recorded seeds for one env, ascending and deduped. */
export function seedsFor(recordings: readonly RecordingRef[], envId: EnvId): number[] {
  const set = new Set<number>();
  for (const rec of recordings) if (rec.envId === envId) set.add(rec.seed);
  return [...set].sort((a, b) => a - b);
}

/** The recording row for one (env, seed), or null. */
export function recordingRef(
  recordings: readonly RecordingRef[],
  envId: EnvId,
  seed: number,
): RecordingRef | null {
  for (const rec of recordings) if (rec.envId === envId && rec.seed === seed) return rec;
  return null;
}

// ---------------------------------------------------------------------------
// load progress
// ---------------------------------------------------------------------------

export type LoadPhase = 'meta' | 'download' | 'decode' | 'scene' | 'ready';

/**
 * Phase bands on the progress bar. The download owns three quarters of it
 * because it owns essentially all of the wall clock on a real network; the
 * decode and the scene build are ~40 ms each but must still visibly advance so
 * a stalled build is distinguishable from a stalled fetch.
 */
const PHASE_BAND: Record<LoadPhase, readonly [number, number]> = {
  meta: [0, 0.04],
  download: [0.04, 0.78],
  decode: [0.78, 0.9],
  scene: [0.9, 0.99],
  ready: [1, 1],
};

export interface LoadProgress {
  phase: LoadPhase;
  /** bytes received (download phase only) */
  received: number;
  /** bytes expected; 0 when the server sent no length */
  total: number;
  /** 0..1 across the whole load, monotone in phase order */
  fraction: number;
  label: string;
}

/** 0..1 for a phase + byte counts. Monotone: every phase starts where the last ended. */
export function progressFraction(phase: LoadPhase, received = 0, total = 0): number {
  const band = PHASE_BAND[phase];
  const lo = band[0];
  const hi = band[1];
  if (phase !== 'download') return phase === 'ready' ? 1 : lo;
  if (!(total > 0)) return lo;
  const inner = received <= 0 ? 0 : received >= total ? 1 : received / total;
  return lo + (hi - lo) * inner;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} kB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** A LoadProgress with its human label — the single place progress text is written. */
export function makeProgress(phase: LoadPhase, received = 0, total = 0): LoadProgress {
  const fraction = progressFraction(phase, received, total);
  let label: string;
  switch (phase) {
    case 'meta':
      label = 'reading recording meta';
      break;
    case 'download':
      label = total > 0
        ? `downloading ${formatBytes(received)} / ${formatBytes(total)}`
        : `downloading ${formatBytes(received)}`;
      break;
    case 'decode':
      label = 'decoding experience';
      break;
    case 'scene':
      label = 'building the scene';
      break;
    default:
      label = 'ready';
      break;
  }
  return { phase, received, total, fraction, label };
}

// ---------------------------------------------------------------------------
// replay health — the "recording damaged" banner
// ---------------------------------------------------------------------------

export interface ReplayHealth {
  damaged: boolean;
  nonFiniteRows: number;
  firstDisagreeStep: number | null;
  disagreeCount: number;
  /** banner text, or null when the recording is clean */
  message: string | null;
}

/**
 * Two independent damage signals, both from sim-core:
 *   * `nonFiniteRows` — the RECORD carries NaN/Inf. `Math.trunc(NaN) & 3 === 0`
 *     decodes as a plausible-looking action, so a silent one is the dangerous case.
 *   * `replay.disagreeCount` — a `needsReplay` decoder's hidden state no longer
 *     matches the recorded pose, so everything the replay owns (key cells) is
 *     suspect from `firstDisagreeStep` on.
 */
export function replayHealth(
  decoded: Pick<DecodedRecording<unknown>, 'replay' | 'nonFiniteRows'>,
): ReplayHealth {
  const nonFiniteRows = decoded.nonFiniteRows ?? 0;
  const disagreeCount = decoded.replay?.disagreeCount ?? 0;
  const firstDisagreeStep = decoded.replay?.firstDisagreeStep ?? null;
  const parts: string[] = [];
  if (nonFiniteRows > 0) {
    parts.push(`${nonFiniteRows} record row${nonFiniteRows === 1 ? ' carries' : 's carry'} non-finite values`);
  }
  if (disagreeCount > 0) {
    parts.push(
      firstDisagreeStep === null
        ? `hidden-state replay desynced on ${disagreeCount} frames`
        : `hidden-state replay desynced from step ${firstDisagreeStep} (${disagreeCount} frames)`,
    );
  }
  return {
    damaged: parts.length > 0,
    nonFiniteRows,
    firstDisagreeStep,
    disagreeCount,
    message: parts.length > 0 ? `Recording damaged — ${parts.join('; ')}.` : null,
  };
}

// ---------------------------------------------------------------------------
// the event timeline — scrub-bar tick marks
// ---------------------------------------------------------------------------

export const TICK_KINDS = ['goal', 'done', 'key', 'door', 'cue'] as const;
export type TickKind = (typeof TICK_KINDS)[number];

/**
 * Which kind wins when several events land in one tick bucket. `goal` first
 * because that is the celebration the tick exists to let you replay.
 */
const TICK_RANK: Record<TickKind, number> = { goal: 0, door: 1, key: 2, cue: 3, done: 4 };

/** The five marked kinds; `teleport`/`resurface` are continuity, not landmarks. */
export function tickKindOf(event: SimEvent): TickKind | null {
  switch (event.type) {
    case 'goal':
    case 'done':
    case 'key':
    case 'door':
    case 'cue':
      return event.type;
    default:
      return null;
  }
}

/** Short human label for one event — used by ticks AND by the side-strip ticker. */
export function describeEvent(event: SimEvent): string {
  switch (event.type) {
    case 'key':
      return `key ${event.index}`;
    case 'door':
      return `door ${event.index} ${event.open ? 'open' : 'shut'}`;
    case 'cue':
      return `cue ${event.value}`;
    default:
      return event.type;
  }
}

export interface TimelineEntry {
  frame: number;
  step: number;
  event: SimEvent;
}

export interface TimelineTick {
  /** the dominant kind in the bucket — drives the tick's color */
  kind: TickKind;
  /** earliest frame in the bucket; the seek anchor */
  frame: number;
  step: number;
  /** 0..1 across the scrub range */
  position: number;
  /** events merged into this tick */
  count: number;
  label: string;
}

export interface TimelineTickOptions {
  /** bucket resolution; the output never exceeds this. Default 180. */
  maxTicks?: number;
}

/**
 * Flatten a timeline into scrub-bar ticks.
 *
 * A 100k-step MemoryChain lane fires ~4400 events — one DOM node each would be
 * both unreadable and expensive, so events are bucketed by scrub POSITION (not
 * by index: the bar is what the user is aiming at) and each bucket keeps its
 * earliest frame, its highest-ranked kind, and the full mix in its label.
 */
export function timelineTicks(
  timeline: readonly TimelineEntry[],
  frameCount: number,
  options: TimelineTickOptions = {},
): TimelineTick[] {
  const buckets = Math.max(1, Math.floor(options.maxTicks ?? 180));
  const last = Math.max(1, frameCount - 1);
  const byBucket = new Map<number, { frame: number; step: number; counts: Map<TickKind, number> }>();

  for (const entry of timeline) {
    const kind = tickKindOf(entry.event);
    if (!kind) continue;
    const position = entry.frame <= 0 ? 0 : entry.frame >= last ? 1 : entry.frame / last;
    const slot = Math.min(buckets - 1, Math.max(0, Math.round(position * (buckets - 1))));
    let bucket = byBucket.get(slot);
    if (!bucket) {
      bucket = { frame: entry.frame, step: entry.step, counts: new Map<TickKind, number>() };
      byBucket.set(slot, bucket);
    } else if (entry.frame < bucket.frame) {
      bucket.frame = entry.frame;
      bucket.step = entry.step;
    }
    bucket.counts.set(kind, (bucket.counts.get(kind) ?? 0) + 1);
  }

  const out: TimelineTick[] = [];
  for (const [slot, bucket] of byBucket) {
    const kinds = [...bucket.counts.entries()].sort((a, b) => TICK_RANK[a[0]] - TICK_RANK[b[0]]);
    const head = kinds[0];
    if (!head) continue;
    let count = 0;
    for (const [, n] of kinds) count += n;
    const mix = kinds.map(([kind, n]) => (n > 1 ? `${n}× ${kind}` : kind)).join(' · ');
    out.push({
      kind: head[0],
      frame: bucket.frame,
      step: bucket.step,
      position: buckets === 1 ? 0 : slot / (buckets - 1),
      count,
      label: `${mix} — step ${bucket.step}`,
    });
  }
  out.sort((a, b) => a.frame - b.frame);
  return out;
}

/** Ticks straight off a decoded recording. */
export function ticksFromRecording(
  decoded: DecodedRecording<unknown>,
  options?: TimelineTickOptions,
): TimelineTick[] {
  return timelineTicks(eventTimeline(decoded), decoded.frames.length, options);
}

/** Same ticks, from a lens: the sparse event map IS the timeline. */
export function ticksFromLens(
  lens: Pick<RecordingLens, 'eventEntries' | 'frameCount'>,
  options?: TimelineTickOptions,
): TimelineTick[] {
  const timeline: { frame: number; step: number; event: SimEvent }[] = [];
  for (const [frame, events] of lens.eventEntries()) {
    for (const event of events) timeline.push({ frame, step: frame - 1, event });
  }
  return timelineTicks(timeline, lens.frameCount, options);
}

/**
 * Clicking a tick seeks JUST BEFORE the event so the celebration actually
 * plays: the host fires events only on sequential steps, so landing exactly on
 * the event frame would arrive as a scrub and swallow the beat.
 */
export function seekTargetForTick(tick: Pick<TimelineTick, 'frame'>, lead = 3): number {
  return Math.max(0, tick.frame - Math.max(1, Math.floor(lead)));
}

// ---------------------------------------------------------------------------
// loading one lane
// ---------------------------------------------------------------------------

export interface TheaterRecording {
  /** the `kit/harness` source shape — `sampleFrame` consumes this unchanged */
  source: HarnessSource;
  health: ReplayHealth;
  ticks: TimelineTick[];
  /** bytes actually downloaded */
  bytes: number;
  /** record rows decoded (frames = steps + 1) */
  steps: number;
  /** record rows on the server, before any maxSteps clamp */
  availableSteps: number;
}

export interface LoadRecordingOptions extends TheaterTarget {
  /** rows the API index says the lane has — the progress total before headers land */
  expectedSteps?: number;
  signal?: AbortSignal;
  onProgress?(progress: LoadProgress): void;
  /** injected in tests; defaults to global fetch */
  fetchImpl?: typeof fetch;
}

export class TheaterLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TheaterLoadError';
  }
}

function recordingBase(runId: string, envId: EnvId, seed: number): string {
  return `/api/runs/${encodeURIComponent(runId)}/recordings/${encodeURIComponent(envId)}/${seed}`;
}

/**
 * Concatenate streamed chunks into one buffer, truncated down to a multiple of
 * `align`.
 *
 * v3 aligns to 64 because `parseActionsBin` rejects a non-row-aligned length
 * outright, so a stream cut mid-row would fail the whole lane. v4 aligns to 1:
 * its own chunk scan is the torn-tail authority and a partial trailing chunk is
 * dropped there, with the crc as the backstop.
 */
function joinBytes(chunks: readonly Uint8Array[], received: number, align: number): Uint8Array {
  const usable = align > 1 ? Math.floor(received / align) * align : received;
  const out = new Uint8Array(usable);
  let offset = 0;
  for (const chunk of chunks) {
    if (offset >= usable) break;
    const take = Math.min(chunk.byteLength, usable - offset);
    out.set(take === chunk.byteLength ? chunk : chunk.subarray(0, take), offset);
    offset += take;
  }
  return out;
}

/** `content-length`, or the total from a `content-range`, or 0. */
export function expectedBytes(headers: Headers, fallbackSteps = 0): number {
  const range = headers.get('content-range');
  if (range) {
    const slash = range.lastIndexOf('/');
    const total = slash >= 0 ? Number(range.slice(slash + 1)) : NaN;
    if (Number.isFinite(total) && total > 0) return total;
  }
  const length = Number(headers.get('content-length'));
  if (Number.isFinite(length) && length > 0) return length;
  return fallbackSteps > 0 ? fallbackSteps * RECORD_BYTES : 0;
}

/**
 * Fetch + decode one recorded lane, reporting real byte progress.
 *
 * Unlike the harness loader this THROWS on failure: the Theater is addressed by
 * an explicit (run, env, seed), so a missing lane is a broken link the view must
 * show, never a fixture loop quietly standing in for real experience.
 */
export async function loadTheaterRecording(options: LoadRecordingOptions): Promise<TheaterRecording> {
  const { runId, envId, seed } = options;
  const doFetch = options.fetchImpl ?? fetch;
  const signal = options.signal;
  const report = (progress: LoadProgress): void => options.onProgress?.(progress);
  const decoder = decoders[envId] as EnvDecoder<unknown, unknown> | undefined;
  if (!decoder) throw new TheaterLoadError(`no @sim decoder registered for ${envId}`);

  const base = recordingBase(runId, envId, seed);
  report(makeProgress('meta'));

  const metaRes = await doFetch(`${base}/meta.json`, {
    headers: { accept: 'application/json' },
    ...(signal ? { signal } : {}),
  });
  if (!metaRes.ok) {
    throw new TheaterLoadError(`${metaRes.status} — no recording for ${envId} seed ${seed} in ${runId}`);
  }
  const meta = (await metaRes.json()) as RecordingMeta;

  const expected = Math.max(0, Math.floor(options.expectedSteps ?? 0));
  // The FULL lane, always — the lens gives full-horizon random access at bounded
  // memory, so there is no decode cap and no prefix-range planning. A 1M-step v4
  // lane is ~2-19 MB on the wire.
  const isV4 = meta.recordVersion === RECORD_V4_VERSION;
  const wantBytes = isV4 ? 0 : Math.max(0, expected) * RECORD_BYTES;

  report(makeProgress('download', 0, wantBytes));
  const binRes = await doFetch(`${base}/actions.bin`, {
    // The API serves byte ranges, so ask for exactly the bytes we will decode; a
    // server that ignores the header sends the whole file, which still parses.
    ...(wantBytes > 0 ? { headers: { range: `bytes=0-${wantBytes - 1}` } } : {}),
    ...(signal ? { signal } : {}),
  });
  if (!binRes.ok && binRes.status !== 206) {
    throw new TheaterLoadError(`${binRes.status} — actions.bin unavailable for ${envId} seed ${seed}`);
  }

  const total = expectedBytes(binRes.headers, isV4 ? 0 : expected) || wantBytes;
  const align = isV4 ? 1 : RECORD_BYTES;
  let bytes: Uint8Array;
  const body = binRes.body;
  if (body && typeof body.getReader === 'function') {
    const reader = body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    try {
      for (;;) {
        const step = await reader.read();
        if (step.done) break;
        const chunk = step.value;
        if (!chunk) continue;
        chunks.push(chunk);
        received += chunk.byteLength;
        report(makeProgress('download', received, total));
      }
    } finally {
      reader.releaseLock();
    }
    bytes = joinBytes(chunks, received, align);
  } else {
    // No streaming body (older runtimes, some test doubles) — one shot, no progress.
    const buffer = await binRes.arrayBuffer();
    const whole = new Uint8Array(buffer);
    bytes = joinBytes([whole], whole.byteLength, align);
    report(makeProgress('download', whole.byteLength, total || whole.byteLength));
  }

  if (bytes.byteLength < (isV4 ? RECORD_V4_HEADER_BYTES : RECORD_BYTES)) {
    throw new TheaterLoadError(`actions.bin for ${envId} seed ${seed} is empty`);
  }

  report(makeProgress('decode', bytes.byteLength, total));
  const raw = isV4 ? await parseActionsBinV4(bytes, meta) : parseActionsBin(bytes, meta);
  if (raw.steps === 0) {
    throw new TheaterLoadError(`actions.bin for ${envId} seed ${seed} is empty`);
  }
  const lens = lensRecording(raw, decoder);
  const info = ENVS[envId];
  const steps = lens.steps;

  const source: HarnessSource = {
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
    label: `${runId} · ${info.name} · seed ${lens.seed} · ${steps} steps`,
    ...(lens.replay ? { replay: lens.replay } : {}),
    nonFiniteRows: lens.nonFiniteRows,
  };

  return {
    source,
    health: replayHealth(lens),
    ticks: ticksFromLens(lens),
    bytes: bytes.byteLength,
    steps,
    availableSteps: expected > 0 ? expected : raw.steps,
  };
}
