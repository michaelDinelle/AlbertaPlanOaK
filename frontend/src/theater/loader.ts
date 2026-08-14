import { ENVS, type EnvId } from '@shared/envs';
import type { RecordingRef } from '@shared/types';
import type { PlaybackSource } from '@kit/playback';
import { recordingUrl } from '../api';
import { NO_EVENTS, lensRecording, type RecordingLens } from '@sim/decode';
import {
  FILE_HEADER_BYTES,
  RECORD_FLOATS,
  decodeChunkPayload,
  inflateChunk,
  readRecordHeader,
  scanNextChunk,
  type CodecDescriptor,
} from '@sim/record';
import { decoders } from '@sim/registry';
import type { EnvDecoder, EnvFrame, SimEvent } from '@sim/types';

export interface TheaterTarget {
  runId: string;
  envId: EnvId;
  seed: number;
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

export class TheaterLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TheaterLoadError';
  }
}

const UPDATE_MIN_MS = 150;
const LIVE_POLL_MS = 2_000;

export interface HeroOptions extends TheaterTarget {
  steps: number;
  horizonSteps: number;
  live: boolean;
  onUpdate?(): void;
  onError?(error: Error): void;
}

export interface HeroHandle {
  readonly source: PlaybackSource;
  readonly loadedSteps: number;
  readonly totalSteps: number;
  readonly bytes: number;
  ticks(): TimelineTick[];
  setLive(live: boolean): void;
  setCallbacks(onUpdate: (() => void) | null, onError: ((error: Error) => void) | null): void;
  destroy(): void;
}

export function makeInitialSource(envId: EnvId, seed: number): PlaybackSource {
  const decoder = decoders[envId] as EnvDecoder<unknown, unknown>;
  const st = decoder.initStatic(seed);
  const init = decoder.initFrame(seed, st);
  const info = ENVS[envId];
  return {
    envId,
    seed,
    controlHz: info.controlHz,
    discretePose: info.discretePose,
    geometry: decoder.geometry(st, seed),
    frameCount: 1,
    complete: false,
    frameAt: () => init as EnvFrame<unknown>,
    eventsAt: () => NO_EVENTS,
    eventEntries: () => new Map<number, SimEvent[]>().entries(),
  };
}

export function loadHero(options: HeroOptions): HeroHandle {
  const { runId, envId, seed } = options;
  const decoder = decoders[envId] as EnvDecoder<unknown, unknown>;
  if (decoder === undefined) throw new TheaterLoadError(`no @sim decoder registered for ${envId}`);
  const info = ENVS[envId];
  const st = decoder.initStatic(seed);
  const geometry = decoder.geometry(st, seed);
  const initFrame = decoder.initFrame(seed, st);

  let live = options.live;
  const capacitySteps = Math.max(options.steps, live ? options.horizonSteps : options.steps, 1);
  const raw = {
    envId,
    seed,
    data: new Float32Array(capacitySteps * RECORD_FLOATS),
    steps: 0,
  };

  let bytes = new Uint8Array(64 * 1024);
  let byteLength = 0;
  let scanOffset = FILE_HEADER_BYTES;
  let decodedSteps = 0;
  let desc: CodecDescriptor | null = null;
  let fullyFetched = false;
  let destroyed = false;
  let lens: RecordingLens | null = null;
  let updateTimer: ReturnType<typeof setTimeout> | null = null;
  let onUpdate = options.onUpdate ?? null;
  let onError = options.onError ?? null;
  let etag: string | null = null;
  const abort = new AbortController();

  const source: PlaybackSource = {
    envId,
    seed,
    controlHz: info.controlHz,
    discretePose: info.discretePose,
    geometry,
    get frameCount() {
      return lens !== null ? lens.frameCount : 1;
    },
    get complete() {
      return fullyFetched && !live;
    },
    frameAt: (i) => (lens !== null ? lens.frameAt(i) : initFrame),
    eventsAt: (i) => (lens !== null ? lens.eventsAt(i) : NO_EVENTS),
    eventEntries: () =>
      lens !== null ? lens.eventEntries() : new Map<number, SimEvent[]>().entries(),
  };

  const grow = (): void => {
    if (destroyed || decodedSteps === 0) return;
    raw.steps = decodedSteps;
    if (lens === null) {
      lens = lensRecording(raw, decoder);
      onUpdate?.();
      return;
    }
    lens.extend();
    if (updateTimer === null) {
      updateTimer = setTimeout(() => {
        updateTimer = null;
        if (!destroyed) onUpdate?.();
      }, UPDATE_MIN_MS);
    }
  };

  const append = async (chunk: Uint8Array): Promise<void> => {
    if (byteLength + chunk.byteLength > bytes.byteLength) {
      let capacity = bytes.byteLength;
      while (capacity < byteLength + chunk.byteLength) capacity *= 2;
      const grown = new Uint8Array(capacity);
      grown.set(bytes.subarray(0, byteLength));
      bytes = grown;
    }
    bytes.set(chunk, byteLength);
    byteLength += chunk.byteLength;

    if (desc === null && byteLength >= FILE_HEADER_BYTES) {
      const header = readRecordHeader(bytes.subarray(0, byteLength));
      if (header.envId !== envId) {
        throw new TheaterLoadError(
          `recording header says ${header.envId}, expected ${envId}`,
        );
      }
      desc = header.desc;
    }
    if (desc === null) return;

    let decodedAny = false;
    for (;;) {
      const view = bytes.subarray(0, byteLength);
      const next = scanNextChunk(view, scanOffset, decodedSteps, 0);
      if (next === null) break;
      if (decodedSteps + next.stepCount > capacitySteps) {
        throw new TheaterLoadError(
          `recording exceeds its declared ${capacitySteps} steps — refusing to decode past the horizon`,
        );
      }
      const payload = await inflateChunk(view, next);
      decodeChunkPayload(payload, desc, next.stepCount, raw.data, decodedSteps * RECORD_FLOATS);
      decodedSteps += next.stepCount;
      scanOffset = next.byteOffset + next.byteLength;
      decodedAny = true;
    }
    if (decodedAny) grow();
  };

  const fetchFrom = async (offset: number): Promise<'grew' | 'restart' | 'done'> => {
    const headers: Record<string, string> = {};
    if (offset > 0) {
      headers['range'] = `bytes=${offset}-`;
      if (etag !== null) headers['if-range'] = etag;
    }
    const res = await fetch(recordingUrl(runId, envId, seed), {
      headers,
      signal: abort.signal,
    });
    if (res.status === 416) return 'done';
    if (!res.ok && res.status !== 206) {
      throw new TheaterLoadError(`${res.status} — recording unavailable for ${envId} seed ${seed}`);
    }
    if (offset > 0 && res.status === 200) return 'restart';
    etag = res.headers.get('etag');
    const body = res.body;
    if (body === null) {
      await append(new Uint8Array(await res.arrayBuffer()));
      return 'grew';
    }
    const reader = body.getReader();
    try {
      for (;;) {
        const step = await reader.read();
        if (step.done) break;
        if (step.value) await append(step.value);
      }
    } finally {
      reader.releaseLock();
    }
    return 'grew';
  };

  const run = async (): Promise<void> => {
    await fetchFrom(0);
    fullyFetched = true;
    onUpdate?.();
    while (live && !destroyed) {
      await new Promise((res) => setTimeout(res, LIVE_POLL_MS));
      if (!live || destroyed) break;
      const result = await fetchFrom(byteLength);
      if (result === 'restart') {
        throw new TheaterLoadError('the recording was rewritten underneath the player — reload the run');
      }
    }
    onUpdate?.();
  };

  void run().catch((error: unknown) => {
    if (destroyed || (error instanceof DOMException && error.name === 'AbortError')) return;
    onError?.(error instanceof Error ? error : new Error(String(error)));
  });

  return {
    source,
    get loadedSteps() {
      return decodedSteps;
    },
    get totalSteps() {
      return Math.max(options.steps, decodedSteps);
    },
    get bytes() {
      return byteLength;
    },
    ticks() {
      return lens !== null ? ticksFromLens(lens) : [];
    },
    setLive(next) {
      live = next;
    },
    setCallbacks(update, error) {
      onUpdate = update;
      onError = error;
    },
    destroy() {
      destroyed = true;
      abort.abort();
      if (updateTimer !== null) clearTimeout(updateTimer);
    },
  };
}
