import type { EnvEntry } from '@sim/catalog';
import type { EnvId } from '@shared/envs';
import type { RecordingRef } from '@shared/types';
import type { PlaybackSource } from './playbackSource';
import { recordingUrl } from '../data/api';
import { lensRecording, type RecordingLens } from '@sim/decode';
import {
  RECORD_FLOATS,
  decodeChunkPayload,
  inflateChunk,
  codecFromEntry,
  scanNextChunk,
} from '@sim/record';
import { decoders } from '@sim/registry';
import { NO_EVENTS, type EnvDecoder } from '@sim/types';

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

export const TICK_KINDS = ['goal', 'door', 'key', 'cue', 'done'] as const;
export type TickKind = (typeof TICK_KINDS)[number];

export interface TimelineTick {
  kind: TickKind;
  frame: number;
  step: number;
  position: number;
  count: number;
  label: string;
}

export function ticksFromLens(
  lens: Pick<RecordingLens, 'events' | 'frameCount'>,
  buckets = 180,
): TimelineTick[] {
  const last = Math.max(1, lens.frameCount - 1);
  const ranks = TICK_KINDS.length;
  const first = new Int32Array(buckets).fill(-1);
  const tally = new Int32Array(buckets * ranks);
  const { steps, kinds, count } = lens.events;
  const rankOf = lens.events.types.map((type) => TICK_KINDS.indexOf(type as TickKind));

  for (let i = 0; i < count; i++) {
    const rank = rankOf[kinds[i] ?? 0] ?? -1;
    if (rank < 0) continue;
    const frame = (steps[i] ?? 0) + 1;
    const position = frame >= last ? 1 : frame / last;
    const slot = Math.min(buckets - 1, Math.max(0, Math.round(position * (buckets - 1))));
    if (first[slot] === -1) first[slot] = frame;
    const cell = slot * ranks + rank;
    tally[cell] = (tally[cell] ?? 0) + 1;
  }

  const out: TimelineTick[] = [];
  for (let slot = 0; slot < buckets; slot++) {
    const frame = first[slot] ?? -1;
    if (frame < 0) continue;
    const parts: string[] = [];
    let count = 0;
    let head: TickKind | null = null;
    for (let r = 0; r < ranks; r++) {
      const n = tally[slot * ranks + r] ?? 0;
      if (n === 0) continue;
      const kind = TICK_KINDS[r] as TickKind;
      if (head === null) head = kind;
      count += n;
      parts.push(n > 1 ? `${n}× ${kind}` : kind);
    }
    if (head === null) continue;
    out.push({
      kind: head,
      frame,
      step: frame - 1,
      position: buckets === 1 ? 0 : slot / (buckets - 1),
      count,
      label: `${parts.join(' · ')} — step ${frame - 1}`,
    });
  }
  return out;
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

export interface HeroOptions extends TheaterTarget {
  entry: EnvEntry;
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

export function makeInitialSource(entry: EnvEntry, envId: EnvId, seed: number): PlaybackSource {
  const decoder = decoders[envId] as EnvDecoder;
  const st = decoder.initStatic(seed);
  const init = decoder.initFrame(seed, st);
  return {
    envId,
    seed,
    controlHz: entry.controlHz,
    discretePose: entry.discretePose,
    wrapMask: 0,
    geometry: decoder.geometry(st, seed),
    frameCount: 1,
    complete: false,
    frameAt: () => init,
    eventsAt: () => NO_EVENTS,
  };
}

export function loadHero(options: HeroOptions): HeroHandle {
  const { runId, envId, seed } = options;
  const decoder = decoders[envId] as EnvDecoder;
  if (decoder === undefined) throw new TheaterLoadError(`no @sim decoder registered for ${envId}`);
  const st = decoder.initStatic(seed);
  const geometry = decoder.geometry(st, seed);
  const initFrame = decoder.initFrame(seed, st);

  let live = options.live;
  const capacitySteps = Math.max(options.horizonSteps, 1);
  const raw = {
    envId,
    seed,
    data: new Float32Array(capacitySteps * RECORD_FLOATS),
    steps: 0,
  };

  let bytes = new Uint8Array(64 * 1024);
  let byteLength = 0;
  let scanOffset = 0;
  let decodedSteps = 0;
  const desc = codecFromEntry(options.entry);
  let fullyFetched = false;
  let destroyed = false;
  let lens: RecordingLens | null = null;
  let updateTimer: number | null = null;
  let onUpdate = options.onUpdate ?? null;
  let onError = options.onError ?? null;
  const abort = new AbortController();

  const wrapMask = options.entry.fields.reduce(
    (mask, field) => (field.wrap ? mask | (1 << field.slot) : mask),
    0,
  );

  const source: PlaybackSource = {
    envId,
    seed,
    controlHz: options.entry.controlHz,
    discretePose: options.entry.discretePose,
    wrapMask,
    geometry,
    get frameCount() {
      return lens !== null ? lens.frameCount : 1;
    },
    get complete() {
      return fullyFetched && !live;
    },
    frameAt: (i) => (lens !== null ? lens.frameAt(i) : initFrame),
    eventsAt: (i) => (lens !== null ? lens.eventsAt(i) : NO_EVENTS),
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
      updateTimer = requestAnimationFrame(() => {
        updateTimer = null;
        if (!destroyed) onUpdate?.();
      });
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

  const fetchFrom = async (offset: number): Promise<'grew' | 'done'> => {
    const headers: Record<string, string> = {};
    if (offset > 0) headers['range'] = `bytes=${offset}-`;
    const res = await fetch(recordingUrl(runId, envId, seed), {
      headers,
      signal: abort.signal,
    });
    if (res.status === 416) return 'done';
    if (!res.ok && res.status !== 206) {
      throw new TheaterLoadError(`${res.status} — recording unavailable for ${envId} seed ${seed}`);
    }
    if (offset > 0 && res.status === 200) {
      throw new TheaterLoadError(`the server ignored a range request for ${envId} seed ${seed}`);
    }
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
    let lastGrewAt = performance.now();
    let intervalMs = 0;
    let backoffMs = 0;
    while (live && !destroyed) {
      const dueIn = intervalMs > 0
        ? Math.max(0, lastGrewAt + intervalMs - performance.now())
        : 0;
      await new Promise((res) => setTimeout(res, Math.max(dueIn, backoffMs)));
      if (!live || destroyed) break;
      const startedAt = performance.now();
      const result = await fetchFrom(byteLength);
      if (result === 'grew') {
        const at = performance.now();
        intervalMs = at - lastGrewAt;
        lastGrewAt = at;
        backoffMs = 0;
      } else {
        backoffMs = Math.max(performance.now() - startedAt, backoffMs * 2);
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
      return Math.max(decodedSteps, 1);
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
      if (update !== null && lens !== null) update();
    },
    destroy() {
      destroyed = true;
      abort.abort();
      if (updateTimer !== null) cancelAnimationFrame(updateTimer);
    },
  };
}
