import { ENVS, isEnvId } from '@shared/envs';
import { NO_EVENTS } from './events';
import { RECORD_FLOATS } from './record';
import type { DecodedRecording, EnvDecoder, EnvFrame, RawRecording, SimEvent } from './types';

export { NO_EVENTS };

export class DecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DecodeError';
  }
}

function checkRow(row: Float32Array, i: number): void {
  for (let j = 0; j < RECORD_FLOATS; j++) {
    if (!Number.isFinite(row[j])) {
      throw new DecodeError(
        `row ${i} carries a non-finite value — recordings end at the last finite row, so this file is corrupt`,
      );
    }
  }
}

function checkIds<S, St>(raw: RawRecording, decoder: EnvDecoder<S, St>): void {
  if (!isEnvId(raw.envId)) {
    throw new DecodeError(`unknown recording envId ${JSON.stringify(raw.envId)}`);
  }
  if (decoder.envId !== raw.envId) {
    throw new DecodeError(
      `decoder envId ${decoder.envId} does not match recording envId ${raw.envId}`,
    );
  }
  const available = Math.floor(raw.data.length / RECORD_FLOATS);
  if (available < raw.steps) {
    throw new DecodeError(
      `recording claims ${raw.steps} steps but data holds only ${available}`,
    );
  }
}

export function decodeRecording<S, St>(
  raw: RawRecording,
  decoder: EnvDecoder<S, St>,
): DecodedRecording<S> {
  checkIds(raw, decoder);
  const info = ENVS[raw.envId as keyof typeof ENVS];
  const st = decoder.initStatic(raw.seed);
  const geometry = decoder.geometry(st, raw.seed);

  const frames: EnvFrame<S>[] = new Array<EnvFrame<S>>(raw.steps + 1);
  const events: SimEvent[][] = new Array<SimEvent[]>(raw.steps + 1);
  const init = decoder.initFrame(raw.seed, st);
  frames[0] = init;
  events[0] = NO_EVENTS;

  let prev = init;
  for (let i = 0; i < raw.steps; i++) {
    const row = raw.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS);
    checkRow(row, i);
    const frame = decoder.decodeFrame(i, row, st, prev);
    frames[i + 1] = frame;
    events[i + 1] = decoder.deriveEvents(prev, frame, st);
    prev = frame;
  }

  return {
    envId: raw.envId,
    seed: raw.seed,
    controlHz: info.controlHz,
    discretePose: info.discretePose,
    geometry,
    frames,
    events,
  };
}

export const LENS_BLOCK = 4096;

export interface RecordingLens<S = unknown> {
  envId: RawRecording['envId'];
  seed: number;
  controlHz: number;
  discretePose: boolean;
  geometry: ReturnType<EnvDecoder<S, never>['geometry']>;
  readonly frameCount: number;
  readonly steps: number;
  frameAt(i: number): EnvFrame<S>;
  eventsAt(i: number): SimEvent[];
  eventEntries(): IterableIterator<[number, SimEvent[]]>;
  extend(): boolean;
}

export function lensRecording<S, St>(
  raw: RawRecording,
  decoder: EnvDecoder<S, St>,
): RecordingLens<S> {
  checkIds(raw, decoder);
  const info = ENVS[raw.envId as keyof typeof ENVS];
  const st = decoder.initStatic(raw.seed);
  const geometry = decoder.geometry(st, raw.seed);
  const init = decoder.initFrame(raw.seed, st);

  const eventsMap = new Map<number, SimEvent[]>();
  const eventList: [number, SimEvent[]][] = [];
  const checkpoints: EnvFrame<S>[] = [];
  let processed = 0;
  let prev = init;

  const advance = (): void => {
    const limit = raw.steps;
    for (; processed < limit; processed++) {
      if (processed % LENS_BLOCK === 0) checkpoints.push(prev);
      const row = raw.data.subarray(processed * RECORD_FLOATS, (processed + 1) * RECORD_FLOATS);
      checkRow(row, processed);
      const frame = decoder.decodeFrame(processed, row, st, prev);
      const events = decoder.deriveEvents(prev, frame, st);
      if (events !== NO_EVENTS && events.length > 0) {
        eventsMap.set(processed + 1, events);
        eventList.push([processed + 1, events]);
      }
      prev = frame;
    }
  };
  advance();

  const cache = new Map<number, { count: number; frames: EnvFrame<S>[] }>();
  const decodeBlock = (b: number): EnvFrame<S>[] => {
    const start = b * LENS_BLOCK;
    const count = Math.min(LENS_BLOCK, processed - start);
    const hit = cache.get(b);
    if (hit !== undefined && hit.count === count) return hit.frames;
    const ck = checkpoints[b];
    if (ck === undefined) throw new DecodeError(`lens block ${b} out of range`);
    const frames: EnvFrame<S>[] = new Array<EnvFrame<S>>(count);
    let p = ck;
    for (let k = 0; k < count; k++) {
      const i = start + k;
      const row = raw.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS);
      p = decoder.decodeFrame(i, row, st, p);
      frames[k] = p;
    }
    cache.set(b, { count, frames });
    if (cache.size > 3) {
      for (const key of cache.keys()) {
        if (Math.abs(key - b) > 1) cache.delete(key);
        if (cache.size <= 3) break;
      }
    }
    return frames;
  };

  const frameAt = (i: number): EnvFrame<S> => {
    const clamped = Math.max(0, Math.min(processed, Math.floor(i)));
    if (clamped === 0) return init;
    const row = clamped - 1;
    const b = Math.floor(row / LENS_BLOCK);
    const block = decodeBlock(b);
    const frame = block[row - b * LENS_BLOCK];
    if (frame === undefined) throw new DecodeError(`lens frame ${clamped} missing`);
    return frame;
  };

  return {
    envId: raw.envId,
    seed: raw.seed,
    controlHz: info.controlHz,
    discretePose: info.discretePose,
    geometry,
    get frameCount() {
      return processed + 1;
    },
    get steps() {
      return processed;
    },
    frameAt,
    eventsAt: (i) => eventsMap.get(Math.floor(i)) ?? NO_EVENTS,
    eventEntries: () => eventList.values(),
    extend() {
      const before = processed;
      advance();
      return processed > before;
    },
  };
}

export function eventTimeline<S>(
  dec: DecodedRecording<S>,
): { frame: number; step: number; event: SimEvent }[] {
  const out: { frame: number; step: number; event: SimEvent }[] = [];
  for (let i = 0; i < dec.events.length; i++) {
    const evs = dec.events[i];
    if (!evs || evs.length === 0) continue;
    const step = dec.frames[i]?.step ?? i - 1;
    for (const event of evs) out.push({ frame: i, step, event });
  }
  return out;
}
