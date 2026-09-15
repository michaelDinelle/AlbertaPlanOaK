import type { EnvEntry } from '@sim/catalog';
import { decodeChunk, indexChunks, type Chunk } from '@sim/chunks';
import { deriveStream, frameOf } from '@sim/events';
import { decoders } from '@sim/registry';
import { emptyStream, NO_EVENTS, type EnvFrame, type EnvGeometry, type EventStream, type SimEvent } from '@sim/types';
import type { EnvId } from '../data/envs';
import { FRAME_FLOATS } from '../data/oak';
import type { TapeJob, TapeResult } from './tape.worker';

export interface Tape {
  envId: EnvId;
  seed: number;
  controlHz: number;
  discretePose: boolean;
  wrapMask: number;
  geometry: EnvGeometry;
  readonly frameCount: number;
  readonly events: EventStream;
  frameAt(i: number): EnvFrame;
  eventsAt(i: number): readonly SimEvent[];
  onChange: (error?: Error) => void;
  pull(): void;
}

type Slot = Tape & { job(): TapeJob; land(result: TapeResult): void; drop(): void };

const cache = new Map<string, Slot>();
const queue: Slot[] = [];
let worker: Worker | null = null;
let running: Slot | null = null;

function pump(): void {
  if (running !== null || queue.length === 0) return;
  running = queue.shift()!;
  if (worker === null) {
    worker = new Worker('/tape.worker.js', { type: 'module' });
    worker.onmessage = (event: MessageEvent<TapeResult>) => {
      const slot = running!;
      running = null;
      slot.land(event.data);
      pump();
    };
    worker.onerror = (event) => {
      const slot = running!;
      running = null;
      worker?.terminate();
      worker = null;
      slot.onChange(new Error(`tape worker: ${event.message}`));
      pump();
    };
  }
  worker.postMessage(running.job());
}

function open(runId: string, lane: number, envId: EnvId, seed: number, entry: EnvEntry, horizon: number): Slot {
  const decoder = decoders[envId];
  const st = decoder.initStatic(seed);
  const init = decoder.initFrame(seed, st);
  const fields = entry.fields;
  const abort = new AbortController();
  const chunks: Chunk[] = [];
  const windows = new Map<number, Float32Array>();
  const stream = emptyStream();
  let bytes = new Uint8Array(0);
  let indexed = 0;
  let frames: Float32Array | null = null;
  let steps = 0;
  let decoded = 0;
  let derived = 0;
  let sent = 0;
  let queued = false;
  let pulling = false;

  const frameAt = (i: number): EnvFrame => {
    const c = Math.max(0, Math.min(steps, Math.floor(i)));
    if (c === 0) return init;
    if (frames !== null) return frameOf(frames, c - 1, c - 1);
    let lo = 0;
    let hi = chunks.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >>> 1;
      if (chunks[mid]!.start <= c - 1) lo = mid;
      else hi = mid - 1;
    }
    const chunk = chunks[lo]!;
    let window = windows.get(lo);
    if (window === undefined) {
      window = new Float32Array(chunk.live * FRAME_FLOATS);
      decodeChunk(bytes, chunk, fields, window, 0);
      windows.set(lo, window);
    }
    return frameOf(window, c - 1 - chunk.start, c - 1);
  };

  const settle = (): void => {
    for (; decoded < chunks.length; decoded++) decodeChunk(bytes, chunks[decoded]!, fields, frames!, chunks[decoded]!.start);
    deriveStream(frames!, derived, steps, decoder, init, st, stream);
    derived = steps;
  };

  const pull = async (): Promise<void> => {
    if (pulling) return;
    pulling = true;
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/poses/${lane}`, { headers: { Range: `bytes=${bytes.length}-` }, signal: abort.signal });
      if (!response.ok) throw new Error(`${response.status} poses/${lane}: ${await response.text()}`);
      const more = new Uint8Array(await response.arrayBuffer());
      if (more.length === 0) return;
      const next = new Uint8Array(bytes.length + more.length);
      next.set(bytes);
      next.set(more, bytes.length);
      bytes = next;
      indexed = indexChunks(bytes, indexed, chunks);
      const last = chunks[chunks.length - 1];
      steps = last === undefined ? 0 : last.start + last.live;
      if (steps > horizon) throw new Error(`lane ${lane} holds ${steps} steps, past the horizon ${horizon}`);
      if (frames !== null) settle();
      else if (!queued && chunks.length > 0) {
        queued = true;
        queue.push(slot);
        pump();
      }
      slot.onChange();
    } catch (error) {
      if (!abort.signal.aborted) slot.onChange(error instanceof Error ? error : new Error(String(error)));
    } finally {
      pulling = false;
    }
  };

  const slot: Slot = {
    envId,
    seed,
    controlHz: entry.controlHz,
    discretePose: entry.discretePose,
    wrapMask: fields.reduce((mask, field) => (field.wrap ? mask | (1 << field.slot) : mask), 0),
    geometry: decoder.geometry(st, seed),
    get frameCount() { return steps + 1; },
    get events() { return stream; },
    frameAt,
    eventsAt(i) {
      const c = Math.floor(i);
      return c <= 0 || c > steps ? NO_EVENTS : decoder.deriveEvents(frameAt(c - 1), frameAt(c), st);
    },
    onChange: () => {},
    pull: () => void pull(),
    job() {
      sent = chunks.length;
      return { bytes, chunks: chunks.slice(), fields, envId, seed, horizon };
    },
    land(result) {
      frames = result.frames;
      decoded = sent;
      derived = result.rows;
      Object.assign(stream, result.stream);
      windows.clear();
      settle();
      slot.onChange();
    },
    drop() {
      abort.abort();
      if (queue.includes(slot)) queue.splice(queue.indexOf(slot), 1);
      if (running === slot) {
        worker?.terminate();
        worker = null;
        running = null;
        pump();
      }
    },
  };
  void pull();
  return slot;
}

export function tapeFor(runId: string, lane: number, envId: EnvId, seed: number, entry: EnvEntry, horizon: number): Tape {
  const key = `${runId}/${lane}`;
  if (!cache.has(key)) cache.set(key, open(runId, lane, envId, seed, entry, horizon));
  return cache.get(key)!;
}

export function keepTapes(keep: Tape[]): void {
  for (const [key, slot] of cache) {
    if (keep.includes(slot)) continue;
    cache.delete(key);
    slot.drop();
  }
}
