import { FRAME_FLOATS, OFF_DONE, OFF_POSE, POSE_LEN } from '../data/oak';
import type { EnvDecoder, EnvFrame, EventStream } from './types';

export function frameOf(data: Float32Array, row: number, step: number): EnvFrame {
  const at = row * FRAME_FLOATS;
  return { step, done: data[at + OFF_DONE] ?? 0, pose: data.subarray(at + OFF_POSE, at + OFF_POSE + POSE_LEN) };
}

export function deriveStream(frames: Float32Array, from: number, to: number, decoder: EnvDecoder, init: EnvFrame, st: unknown, stream: EventStream): void {
  const frame = (i: number): EnvFrame => (i === 0 ? init : frameOf(frames, i - 1, i - 1));
  for (let row = from; row < to; row++) {
    for (const event of decoder.deriveEvents(frame(row), frame(row + 1), st)) {
      if (stream.count === stream.steps.length) {
        const grow = <T extends Int32Array | Uint16Array>(old: T): T => { const next = new (old.constructor as new (n: number) => T)(old.length * 2); next.set(old); return next; };
        stream.steps = grow(stream.steps);
        stream.kinds = grow(stream.kinds);
        stream.cellX = grow(stream.cellX);
        stream.cellY = grow(stream.cellY);
      }
      let id = stream.types.indexOf(event.type);
      if (id < 0) id = stream.types.push(event.type) - 1;
      stream.steps[stream.count] = row;
      stream.kinds[stream.count] = id;
      stream.cellX[stream.count] = event.type === 'collection' ? event.x : 0;
      stream.cellY[stream.count] = event.type === 'collection' ? event.y : 0;
      stream.count += 1;
    }
  }
}
