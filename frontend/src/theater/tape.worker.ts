import { deriveBlock, frameOf } from '@sim/events';
import type { EnvFrame, EventBlock } from '@sim/types';
import { readHeader, readVerdict, VERDICT_BYTES, type TapeHeader } from '../data/replay';
import { SCENES } from '../scenes/table';
import { tickAccumulator, type Tick } from './recording';

export type TapeMessage =
  | { type: 'header'; header: TapeHeader; initial: ArrayBuffer }
  | { type: 'block'; index: number; data: ArrayBuffer; events: EventBlock; through: number }
  | { type: 'progress'; ticks: Tick[]; through: number }
  | { type: 'verdict'; identical: boolean }
  | { type: 'error'; message: string };
export type TapeRequest = { type: 'open'; url: string; envId: string; seed: number; horizon: number; hash: string } | { type: 'seek'; step: number };

let request: Extract<TapeRequest, { type: 'open' }>;
let header: TapeHeader | null = null, pending = new Uint8Array(0), nextFrame = 0, initialSent = false;
let previous: EnvFrame | null = null, cached = false, totalBytes = 0, cursor = 0, lastProgress = 0;
let accumulator: ReturnType<typeof tickAccumulator>;
let seeking = false, wanted = 0, stopped = false, verified = false;
const loaded = new Set<number>();
const abort = new AbortController();

const send = (message: TapeMessage, transfer: Transferable[] = []) => postMessage(message, { transfer });
const fail = (error: unknown) => { if (stopped) return; stopped = true; abort.abort(); send({ type: 'error', message: error instanceof Error ? error.message : String(error) }); };

function verdict(bytes: Uint8Array) {
  const result = readVerdict(bytes, header!);
  const identical = result.identical && result.recorded === BigInt(request.hash);
  verified = true;
  send({ type: 'verdict', identical });
  if (!identical) throw new Error('Replay reward hash differs from the recording');
}

function emitBlock(bytes: Uint8Array, from: number, prev: EnvFrame | null, through: number) {
  const h = header!, index = Math.floor(from / h.blockFrames);
  if (loaded.has(index)) return;
  const data = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const decoder = SCENES[h.descriptor.environment.family]?.decoder;
  if (!decoder) throw new Error('Visualization unavailable');
  const events = deriveBlock(data, from, h, decoder, prev);
  accumulator.add(events);
  loaded.add(index);
  send({ type: 'block', index, data: data.buffer as ArrayBuffer, events, through }, [data.buffer, events.steps.buffer, events.kinds.buffer, events.cellX.buffer, events.cellY.buffer, events.values.buffer]);
}

function progress(force = false) {
  const now = performance.now();
  if (force || now - lastProgress > 80) {
    lastProgress = now;
    send({ type: 'progress', ticks: accumulator.ticks(), through: nextFrame - 1 });
  }
}

function consume(bytes: Uint8Array) {
  const joined = new Uint8Array(pending.length + bytes.length);
  joined.set(pending); joined.set(bytes, pending.length); pending = joined;
  if (!header) {
    header = readHeader(pending);
    if (!header) return;
    const d = header.descriptor;
    if (d.environment.envId !== request.envId || d.seed !== request.seed || d.horizon !== request.horizon) throw new Error('Replay does not match the selected lane');
    accumulator = tickAccumulator(d.horizon);
    pending = pending.subarray(header.offset);
  }
  const rowBytes = header.width * 4;
  if (!initialSent && pending.length >= rowBytes) {
    const initial = pending.slice(0, rowBytes).buffer;
    initialSent = true;
    send({ type: 'header', header, initial }, [initial]);
  }
  while (nextFrame < header.frames) {
    const count = Math.min(header.blockFrames, header.frames - nextFrame), size = count * rowBytes;
    if (pending.length < size) break;
    const block = pending.slice(0, size), data = new Float32Array(block.buffer);
    const last = frameOf(data.slice((count - 1) * header.width), 0, nextFrame + count - 1, header);
    emitBlock(block, nextFrame, previous, nextFrame + count - 1);
    previous = last; nextFrame += count;
    pending = pending.subarray(size);
  }
  if (nextFrame === header.frames && pending.length >= VERDICT_BYTES) {
    if (pending.length !== VERDICT_BYTES) throw new Error('Unexpected data after replay verdict');
    verdict(pending); pending = new Uint8Array(0);
  }
  progress();
}

async function range(first: number, last: number): Promise<Uint8Array> {
  const response = await fetch(request.url, { headers: { Range: `bytes=${first}-${last}` }, signal: abort.signal });
  if (response.status !== 206) { await response.body?.cancel(); throw new Error('Completed replay cache changed; retry this selection'); }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length !== last - first + 1) throw new Error('Incomplete replay range');
  return bytes;
}

async function seek() {
  if (seeking || !cached || !header || stopped) return;
  seeking = true;
  try {
    for (;;) {
      const index = Math.floor(wanted / header.blockFrames);
      if (loaded.has(index)) break;
      const from = index * header.blockFrames, start = Math.max(0, from - 1), end = Math.min(header.frames, from + header.blockFrames);
      const bytes = await range(header.offset + start * header.width * 4, header.offset + end * header.width * 4 - 1);
      const prev = from ? frameOf(new Float32Array(bytes.buffer, 0, header.width), 0, start, header) : null;
      emitBlock(bytes.slice((from - start) * header.width * 4), from, prev, nextFrame - 1);
      progress(true);
      if (Math.floor(wanted / header.blockFrames) === index) break;
    }
  } catch (error) { fail(error); }
  finally { seeking = false; }
}

async function load() {
  const response = await fetch(request.url, { headers: { Range: 'bytes=0-32767' }, signal: abort.signal });
  if (!response.ok || !response.body) throw new Error(`${response.status} replay: ${await response.text()}`);
  cached = response.status === 206;
  if (cached) {
    const match = /^bytes \d+-\d+\/(\d+)$/.exec(response.headers.get('Content-Range') ?? '');
    if (!match) throw new Error('Replay range length is missing');
    totalBytes = Number(match[1]);
  }
  for await (const chunk of response.body) { cursor += chunk.length; consume(chunk); }
  if (cached) {
    if (!header) throw new Error('Replay descriptor is incomplete');
    verdict(await range(totalBytes - VERDICT_BYTES, totalBytes - 1));
    void seek();
    while (cursor < totalBytes && !stopped) {
      const last = Math.min(totalBytes - 1, cursor + 1024 * 1024 - 1);
      const bytes = await range(cursor, last); cursor = last + 1; consume(bytes);
    }
  }
  if (!verified || !header || nextFrame !== header.frames || pending.length !== 0) throw new Error('Replay generation interrupted');
  progress(true);
}

onmessage = (event: MessageEvent<TapeRequest>) => {
  if (event.data.type === 'open') { request = event.data; void load().catch(fail); }
  else { wanted = event.data.step; void seek(); }
};
