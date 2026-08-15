import { ENVS, isEnvId } from '@shared/envs';
import { NO_EVENTS } from './events';
import { OFF_DONE, OFF_POSE, POSE_LEN, RECORD_FLOATS } from './record';
import type { EnvDecoder, EnvFrame, EnvGeometry, RawRecording, SimEvent } from './types';

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

function checkIds<St>(raw: RawRecording, decoder: EnvDecoder<St>): void {
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

export interface EventStream {
  steps: Int32Array;
  kinds: Uint16Array;
  types: readonly SimEvent['type'][];
  count: number;
}

export interface RecordingLens {
  envId: RawRecording['envId'];
  seed: number;
  controlHz: number;
  discretePose: boolean;
  geometry: EnvGeometry;
  readonly frameCount: number;
  frameAt(i: number): EnvFrame;
  eventsAt(i: number): readonly SimEvent[];
  readonly events: EventStream;
  extend(): boolean;
}

export function lensRecording<St>(
  raw: RawRecording,
  decoder: EnvDecoder<St>,
): RecordingLens {
  checkIds(raw, decoder);
  const info = ENVS[raw.envId as keyof typeof ENVS];
  const st = decoder.initStatic(raw.seed);
  const geometry = decoder.geometry(st, raw.seed);
  const init = decoder.initFrame(raw.seed, st);

  let eventSteps = new Int32Array(1024);
  let eventKinds = new Uint16Array(1024);
  let eventCount = 0;
  const typeIds = new Map<SimEvent['type'], number>();
  let processed = 0;

  const frameAt = (i: number): EnvFrame => {
    const clamped = Math.max(0, Math.min(processed, Math.floor(i)));
    if (clamped === 0) return init;
    const at = (clamped - 1) * RECORD_FLOATS;
    return {
      step: clamped - 1,
      done: raw.data[at + OFF_DONE] ?? 0,
      pose: raw.data.subarray(at + OFF_POSE, at + OFF_POSE + POSE_LEN),
    };
  };

  const eventsAt = (i: number): readonly SimEvent[] => {
    const clamped = Math.floor(i);
    if (clamped <= 0 || clamped > processed) return NO_EVENTS;
    return decoder.deriveEvents(frameAt(clamped - 1), frameAt(clamped), st);
  };

  const advance = (): void => {
    for (; processed < raw.steps; processed++) {
      checkRow(raw.data.subarray(processed * RECORD_FLOATS, (processed + 1) * RECORD_FLOATS), processed);
      for (const event of eventsAt(processed + 1)) {
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
    }
  };
  advance();

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
