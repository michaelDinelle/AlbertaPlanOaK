import { isEnvId } from '@shared/envs';
import { OFF_DONE, OFF_POSE, POSE_LEN, RECORD_FLOATS } from './record';
import {
  NO_EVENTS,
  type EnvDecoder,
  type EnvFrame,
  type EventStream,
  type RawRecording,
  type SimEvent,
} from './types';


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

export interface RecordingLens {
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
  const st = decoder.initStatic(raw.seed);
  const init = decoder.initFrame(raw.seed, st);

  const typeNames: SimEvent['type'][] = [];
  const stream: EventStream = {
    steps: new Int32Array(1024),
    kinds: new Uint16Array(1024),
    cellX: new Int32Array(1024),
    cellY: new Int32Array(1024),
    types: typeNames,
    count: 0,
  };
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
    while (processed < raw.steps) {
      const row = processed;
      checkRow(raw.data.subarray(row * RECORD_FLOATS, (row + 1) * RECORD_FLOATS), row);
      processed = row + 1;
      for (const event of eventsAt(processed)) {
        if (stream.count === stream.steps.length) {
          const steps = new Int32Array(stream.count * 2);
          const kinds = new Uint16Array(stream.count * 2);
          const cellX = new Int32Array(stream.count * 2);
          const cellY = new Int32Array(stream.count * 2);
          steps.set(stream.steps);
          kinds.set(stream.kinds);
          cellX.set(stream.cellX);
          cellY.set(stream.cellY);
          stream.steps = steps;
          stream.kinds = kinds;
          stream.cellX = cellX;
          stream.cellY = cellY;
        }
        let id = typeIds.get(event.type);
        if (id === undefined) {
          id = typeIds.size;
          typeIds.set(event.type, id);
          typeNames.push(event.type);
        }
        stream.steps[stream.count] = row;
        stream.kinds[stream.count] = id;
        stream.cellX[stream.count] = event.type === 'collection' ? event.x : 0;
        stream.cellY[stream.count] = event.type === 'collection' ? event.y : 0;
        stream.count += 1;
      }
    }
  };
  advance();

  return {
    get frameCount() {
      return processed + 1;
    },
    frameAt,
    eventsAt,
    get events() {
      return stream;
    },
    extend() {
      const before = processed;
      advance();
      return processed > before;
    },
  };
}
