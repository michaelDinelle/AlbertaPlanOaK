import type { EnvEntry } from '@sim/catalog';
import type { EnvId } from '../data/envs';
import type { LiveEdge, PlaybackSource } from './playbackSource';
import { FRAME_FLOATS, fetchFrames } from '../data/oak';
import { lensRecording, type RecordingLens } from '@sim/decode';
import { decoders } from '@sim/registry';
import { EMPTY_EVENT_STREAM, NO_EVENTS, type EnvDecoder } from '@sim/types';

export interface TheaterTarget {
  runId: string;
  envId: EnvId;
  seed: number;
}

export const TICK_KINDS = [
  'goal',
  'collection',
  'door',
  'key',
  'cue',
  'teleport',
  'resurface',
  'done',
] as const;
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
  lane: number;
  horizonSteps: number;
  live: boolean;
  startedAt: number;
  onUpdate?(): void;
  onError?(error: Error): void;
}

export interface HeroHandle {
  readonly source: PlaybackSource;
  readonly fetched: boolean;
  readonly loadedSteps: number;
  readonly totalSteps: number;
  readonly bytes: number;
  ticks(): TimelineTick[];
  pull(): void;
  setLive(live: boolean): void;
  setCallbacks(onUpdate: (() => void) | null, onError: ((error: Error) => void) | null): void;
  destroy(): void;
}

const PAGE_STEPS = 65536;

export function loadHero(options: HeroOptions): HeroHandle {
  const { runId, envId, seed, lane } = options;
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
    data: new Float32Array(capacitySteps * FRAME_FLOATS),
    steps: 0,
  };

  let decodedSteps = 0;
  let fullyFetched = false;
  let destroyed = false;
  let lens: RecordingLens | null = null;
  let updateTimer: number | null = null;
  const edge: LiveEdge = { framesPerSecond: 0, chunkFrames: 0 };
  let edgeAt = 0;
  let edgeSteps = 0;
  let arrivals = 0;
  const observeArrival = (): void => {
    const now = performance.now();
    const seconds = (now - edgeAt) / 1000;
    const steps = decodedSteps - edgeSteps;
    if (seconds < 0.5 || steps <= 0) return;
    arrivals++;
    if (arrivals > 1) {
      const instant = steps / seconds;
      edge.framesPerSecond = edge.framesPerSecond > 0 ? (edge.framesPerSecond + instant) / 2 : instant;
      edge.chunkFrames = (edge.chunkFrames + steps) / 2;
    }
    edgeAt = now;
    edgeSteps = decodedSteps;
  };
  let onUpdate = options.onUpdate ?? null;
  let onError = options.onError ?? null;
  const abort = new AbortController();
  let pulling = false;

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
    get edge() {
      return live && edge.framesPerSecond > 0 ? edge : null;
    },
    get events() {
      return lens !== null ? lens.events : EMPTY_EVENT_STREAM;
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
    if (fullyFetched) observeArrival();
    if (updateTimer === null) {
      updateTimer = requestAnimationFrame(() => {
        updateTimer = null;
        if (!destroyed) onUpdate?.();
      });
    }
  };

  const append = (from: number, values: Float32Array): boolean => {
    if (from !== decodedSteps) return false;
    const rows = Math.floor(values.length / FRAME_FLOATS);
    if (rows === 0) return false;
    if (decodedSteps + rows > capacitySteps) {
      throw new TheaterLoadError(
        `recording exceeds its declared ${capacitySteps} steps — refusing to decode past the horizon`,
      );
    }
    raw.data.set(values.subarray(0, rows * FRAME_FLOATS), decodedSteps * FRAME_FLOATS);
    decodedSteps += rows;
    if (rows > edge.chunkFrames) edge.chunkFrames = rows;
    grow();
    return true;
  };

  const pull = async (): Promise<void> => {
    if (pulling || destroyed) return;
    pulling = true;
    try {
      while (!destroyed && decodedSteps < capacitySteps) {
        const from = decodedSteps;
        const values = await fetchFrames(runId, lane, from, Math.min(PAGE_STEPS, capacitySteps - from), abort.signal);
        if (!append(from, values) || values.length < PAGE_STEPS * FRAME_FLOATS) break;
      }
    } finally {
      pulling = false;
    }
  };

  const report = (error: unknown): void => {
    if (destroyed || (error instanceof DOMException && error.name === 'AbortError')) return;
    onError?.(error instanceof Error ? error : new Error(String(error)));
  };

  const run = async (): Promise<void> => {
    await pull();
    fullyFetched = true;
    edgeAt = performance.now();
    edgeSteps = decodedSteps;
    if (decodedSteps > 0) edge.framesPerSecond = decodedSteps / Math.max(1, (Date.now() - options.startedAt) / 1000);
    onUpdate?.();
  };

  void run().catch(report);

  return {
    source,
    get fetched() {
      return fullyFetched;
    },
    get loadedSteps() {
      return decodedSteps;
    },
    get totalSteps() {
      return Math.max(source.frameCount - 1, 1);
    },
    get bytes() {
      return decodedSteps * FRAME_FLOATS * 4;
    },
    ticks() {
      return lens !== null ? ticksFromLens(lens) : [];
    },
    pull() {
      if (fullyFetched) pull().catch(report);
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
