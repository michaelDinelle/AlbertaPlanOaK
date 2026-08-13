
import { ENVS, ENV_IDS, isEnvId, type EnvId } from '@shared/envs';
import type { RecordingMeta, RecordingRef, RunIndexEntry } from '@shared/types';
import { DEFAULT_HARNESS_RUN, type HarnessSource } from '@kit/harness';
import type { QualityTier } from '@kit/types';
import { eventTimeline, lensRecording, type RecordingLens } from '@sim/decode';
import {
  FILE_HEADER_BYTES as RECORD_HEADER_BYTES,
  parseActionsBin,
} from '@sim/record';
import { decoders } from '@sim/registry';
import type { DecodedRecording, EnvDecoder, SimEvent } from '@sim/types';
import { sortRuns } from '../state';

export const DEFAULT_THEATER_RUN = DEFAULT_HARNESS_RUN;


const TIER_NAMES: readonly string[] = ['hero', 'medium', 'thumb'];


export interface TheaterTarget {
  runId: string;
  envId: EnvId;
  seed: number;
}

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
  count: number;
  seeds: number[];
}

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

export function seedsFor(recordings: readonly RecordingRef[], envId: EnvId): number[] {
  const set = new Set<number>();
  for (const rec of recordings) if (rec.envId === envId) set.add(rec.seed);
  return [...set].sort((a, b) => a - b);
}

export function recordingRef(
  recordings: readonly RecordingRef[],
  envId: EnvId,
  seed: number,
): RecordingRef | null {
  for (const rec of recordings) if (rec.envId === envId && rec.seed === seed) return rec;
  return null;
}


export type LoadPhase = 'meta' | 'download' | 'decode' | 'scene' | 'ready';

const PHASE_BAND: Record<LoadPhase, readonly [number, number]> = {
  meta: [0, 0.04],
  download: [0.04, 0.78],
  decode: [0.78, 0.9],
  scene: [0.9, 0.99],
  ready: [1, 1],
};

export interface LoadProgress {
  phase: LoadPhase;
  received: number;
  total: number;
  fraction: number;
  label: string;
}

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


export interface ReplayHealth {
  damaged: boolean;
  nonFiniteRows: number;
  firstDisagreeStep: number | null;
  disagreeCount: number;
  message: string | null;
}

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


export const TICK_KINDS = ['goal', 'done', 'key', 'door', 'cue'] as const;
export type TickKind = (typeof TICK_KINDS)[number];

const TICK_RANK: Record<TickKind, number> = { goal: 0, door: 1, key: 2, cue: 3, done: 4 };

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
  kind: TickKind;
  frame: number;
  step: number;
  position: number;
  count: number;
  label: string;
}

export interface TimelineTickOptions {
  maxTicks?: number;
}

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

export function ticksFromRecording(
  decoded: DecodedRecording<unknown>,
  options?: TimelineTickOptions,
): TimelineTick[] {
  return timelineTicks(eventTimeline(decoded), decoded.frames.length, options);
}

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

export function seekTargetForTick(tick: Pick<TimelineTick, 'frame'>, lead = 3): number {
  return Math.max(0, tick.frame - Math.max(1, Math.floor(lead)));
}


export interface TheaterRecording {
  source: HarnessSource;
  health: ReplayHealth;
  ticks: TimelineTick[];
  bytes: number;
  steps: number;
  availableSteps: number;
}

export interface LoadRecordingOptions extends TheaterTarget {
  expectedSteps?: number;
  signal?: AbortSignal;
  onProgress?(progress: LoadProgress): void;
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

export function expectedBytes(headers: Headers): number {
  const range = headers.get('content-range');
  if (range) {
    const slash = range.lastIndexOf('/');
    const total = slash >= 0 ? Number(range.slice(slash + 1)) : NaN;
    if (Number.isFinite(total) && total > 0) return total;
  }
  const length = Number(headers.get('content-length'));
  if (Number.isFinite(length) && length > 0) return length;
  return 0;
}

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

  report(makeProgress('download', 0, 0));
  const binRes = await doFetch(`${base}/actions.bin`, {
    ...(signal ? { signal } : {}),
  });
  if (!binRes.ok && binRes.status !== 206) {
    throw new TheaterLoadError(`${binRes.status} — actions.bin unavailable for ${envId} seed ${seed}`);
  }

  const total = expectedBytes(binRes.headers);
  const align = 1;
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
    const buffer = await binRes.arrayBuffer();
    const whole = new Uint8Array(buffer);
    bytes = joinBytes([whole], whole.byteLength, align);
    report(makeProgress('download', whole.byteLength, total || whole.byteLength));
  }

  if (bytes.byteLength < RECORD_HEADER_BYTES) {
    throw new TheaterLoadError(`actions.bin for ${envId} seed ${seed} is empty`);
  }

  report(makeProgress('decode', bytes.byteLength, total));
  const raw = await parseActionsBin(bytes, meta);
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
