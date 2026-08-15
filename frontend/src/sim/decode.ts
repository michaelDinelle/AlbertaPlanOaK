import { ENVS, isEnvId } from '@shared/envs';
import { NO_EVENTS } from './events';
import { RECORD_FLOATS } from './record';
import type { EnvDecoder, EnvFrame, RawRecording, SimEvent } from './types';

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

export const LENS_BLOCK = 4096;

export interface EventStream {
  steps: Int32Array;
  kinds: Uint16Array;
  types: readonly SimEvent['type'][];
  count: number;
}

export interface RecordingLens<S = unknown> {
  envId: RawRecording['envId'];
  seed: number;
  controlHz: number;
  discretePose: boolean;
  geometry: ReturnType<EnvDecoder<S, never>['geometry']>;
  readonly frameCount: number;
  frameAt(i: number): EnvFrame<S>;
  eventsAt(i: number): readonly SimEvent[];
  readonly events: EventStream;
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

  let eventSteps = new Int32Array(1024);
  let eventKinds = new Uint16Array(1024);
  let eventCount = 0;
  const typeIds = new Map<SimEvent['type'], number>();
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
      for (const event of events) {
        if (eventCount === eventSteps.length) {
          const steps = new Int32Array(eventCount * 2);
          const kinds = new Uint16Array(eventCount * 2);
          steps.set(eventSteps);
          kinds.set(eventKinds);
          eventSteps = steps;
          eventKinds = kinds;
        }
        let id = typeIds.get(event.type);
        if (id === undefined) {
          id = typeIds.size;
          typeIds.set(event.type, id);
        }
        eventSteps[eventCount] = processed;
        eventKinds[eventCount] = id;
        eventCount += 1;
      }
      prev = frame;
    }
  };
  advance();

  type Block = { count: number; frames: EnvFrame<S>[]; events: SimEvent[][] };
  const cache = new Map<number, Block>();
  const decodeBlock = (b: number): Block => {
    const start = b * LENS_BLOCK;
    const count = Math.min(LENS_BLOCK, processed - start);
    const hit = cache.get(b);
    if (hit !== undefined && hit.count === count) return hit;
    const ck = checkpoints[b];
    if (ck === undefined) throw new DecodeError(`lens block ${b} out of range`);
    const frames: EnvFrame<S>[] = new Array<EnvFrame<S>>(count);
    const events: SimEvent[][] = new Array<SimEvent[]>(count);
    let p = ck;
    for (let k = 0; k < count; k++) {
      const i = start + k;
      const row = raw.data.subarray(i * RECORD_FLOATS, (i + 1) * RECORD_FLOATS);
      const next = decoder.decodeFrame(i, row, st, p);
      events[k] = decoder.deriveEvents(p, next, st) as SimEvent[];
      p = next;
      frames[k] = p;
    }
    cache.set(b, { count, frames, events });
    if (cache.size > 3) {
      for (const key of cache.keys()) {
        if (Math.abs(key - b) > 1) cache.delete(key);
        if (cache.size <= 3) break;
      }
    }
    return cache.get(b) as Block;
  };

  const frameAt = (i: number): EnvFrame<S> => {
    const clamped = Math.max(0, Math.min(processed, Math.floor(i)));
    if (clamped === 0) return init;
    const row = clamped - 1;
    const b = Math.floor(row / LENS_BLOCK);
    const frame = decodeBlock(b).frames[row - b * LENS_BLOCK];
    if (frame === undefined) throw new DecodeError(`lens frame ${clamped} missing`);
    return frame;
  };

  const eventsAt = (i: number): readonly SimEvent[] => {
    const clamped = Math.floor(i);
    if (clamped <= 0 || clamped > processed) return NO_EVENTS;
    const row = clamped - 1;
    const b = Math.floor(row / LENS_BLOCK);
    return decodeBlock(b).events[row - b * LENS_BLOCK] ?? NO_EVENTS;
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
    frameAt,
    eventsAt,
    get events() {
      return { steps: eventSteps, kinds: eventKinds, types: [...typeIds.keys()], count: eventCount };
    },
    extend() {
      const before = processed;
      advance();
      return processed > before;
    },
  };
}
