import type { RecordFieldSpec } from '@sim/catalog';
import { decodeChunk, type Chunk } from '@sim/chunks';
import { deriveStream } from '@sim/events';
import { decoders } from '@sim/registry';
import { emptyStream, type EventStream } from '@sim/types';
import type { EnvId } from '../data/envs';
import { FRAME_FLOATS } from '../data/oak';

export interface TapeJob { bytes: Uint8Array; chunks: Chunk[]; fields: RecordFieldSpec[]; envId: EnvId; seed: number; horizon: number }
export interface TapeResult { frames: Float32Array; stream: EventStream; rows: number }

onmessage = (event: MessageEvent<TapeJob>) => {
  const { bytes, chunks, fields, envId, seed, horizon } = event.data;
  const frames = new Float32Array(horizon * FRAME_FLOATS);
  let rows = 0;
  for (const chunk of chunks) {
    decodeChunk(bytes, chunk, fields, frames, chunk.start);
    rows = chunk.start + chunk.live;
  }
  const decoder = decoders[envId];
  const st = decoder.initStatic(seed);
  const stream = emptyStream();
  deriveStream(frames, 0, rows, decoder, decoder.initFrame(seed, st), st, stream);
  postMessage({ frames, stream, rows } satisfies TapeResult, { transfer: [frames.buffer, stream.steps.buffer, stream.kinds.buffer, stream.cellX.buffer, stream.cellY.buffer] });
};
