import { blockOf, deriveBlock, frameOf } from '@sim/events';
import type { EnvDecoder, EnvFrame, EventBlock } from '@sim/types';
import { readHeader, readVerdict, VERDICT_BYTES, type TapeHeader } from '../data/replay';
import { sceneOf } from '../scenes/table';
import { tickAccumulator, type Tick } from './recording';

export type TapeMessage =
  | { type: 'header'; header: TapeHeader; initial: ArrayBuffer }
  | { type: 'block'; index: number; data: ArrayBuffer; events: EventBlock | null; through: number }
  | { type: 'events'; index: number; events: EventBlock; through: number }
  | { type: 'progress'; ticks: Tick[]; through: number }
  | { type: 'error'; message: string };
export type TapeRequest =
  | { type: 'open'; url: string; envId: string; seed: number; horizon: number; hash: string }
  | { type: 'seek'; step: number }
  | { type: 'forget'; indices: number[] };

const PROGRESS_MS = 250;
const PROBE_BYTES = 32768;
const FETCH_BYTES = 1024 * 1024;
const HISTORY_BLOCKS = 16;

let request: Extract<TapeRequest, { type: 'open' }>;
let header: TapeHeader | null = null, decoder: EnvDecoder;
let pending: Uint8Array = new Uint8Array(0), tail: Uint8Array = new Uint8Array(0);
let block: Uint8Array | null = null, fill = 0, nextFrame = 0, initialSent = false;
let previous: EnvFrame | null = null, cached = false, totalBytes = 0, cursor = 0, lastProgress = 0;
let eventsThrough = -1, wanted = 0, seeking = false, stopped = false, verified = false;
let accumulator: ReturnType<typeof tickAccumulator>;
const loaded = new Set<number>(), derived = new Set<number>();
const abort = new AbortController();

const send = (message: TapeMessage, transfer: Transferable[] = []) => postMessage(message, { transfer });
const fail = (error: unknown) => { if (stopped) return; stopped = true; abort.abort(); send({ type: 'error', message: error instanceof Error ? error.message : String(error) }); };
const concat = (a: Uint8Array, b: Uint8Array): Uint8Array<ArrayBuffer> => { const out = new Uint8Array(a.length + b.length); out.set(a); out.set(b, a.length); return out; };
const rowBytes = (): number => header!.width * 4;
const blockIndex = (step: number): number => Math.floor(step / header!.blockFrames);
const blockStart = (index: number): number => index * header!.blockFrames;
const blockCount = (from: number): number => Math.min(header!.blockFrames, header!.frames - from);

function verdict(bytes: Uint8Array): void {
  const result = readVerdict(bytes, header!);
  verified = true;
  if (!result.identical || result.recorded !== BigInt(request.hash)) throw new Error('Replay reward hash differs from the recording');
}

function extendThrough(): void {
  for (;;) {
    const next = eventsThrough + 1;
    if (next >= header!.frames || !derived.has(blockIndex(next))) return;
    eventsThrough = blockStart(blockIndex(next)) + blockCount(blockStart(blockIndex(next))) - 1;
  }
}

function emitBlock(data: ArrayBuffer, from: number, prev: EnvFrame | null, frames: boolean): void {
  const index = blockIndex(from), view = blockOf(data);
  let events: EventBlock | null = null;
  if (!derived.has(index)) {
    events = deriveBlock(view, from, header!, decoder, prev);
    accumulator.add(events);
    derived.add(index);
    extendThrough();
  }
  const transfer: Transferable[] = events ? [events.steps.buffer, events.kinds.buffer, events.cellX.buffer, events.cellY.buffer, events.values.buffer] : [];
  if (frames) { loaded.add(index); send({ type: 'block', index, data, events, through: eventsThrough }, [data, ...transfer]); }
  else if (events) send({ type: 'events', index, events, through: eventsThrough }, transfer);
}

function progress(force = false): void {
  const now = performance.now();
  if (force || now - lastProgress > PROGRESS_MS) {
    lastProgress = now;
    send({ type: 'progress', ticks: accumulator.ticks(), through: eventsThrough });
  }
}

function consume(chunk: Uint8Array): void {
  let bytes: Uint8Array = chunk;
  if (header === null) {
    pending = concat(pending, bytes);
    header = readHeader(pending);
    if (header === null) return;
    const d = header.descriptor;
    if (d.environment.envId !== request.envId || d.seed !== request.seed || d.horizon !== request.horizon) throw new Error('Replay does not match the selected lane');
    decoder = sceneOf(d.environment).decoder;
    accumulator = tickAccumulator(d.horizon);
    bytes = pending.subarray(header.offset);
    pending = new Uint8Array(0);
  }
  let at = 0;
  while (at < bytes.length) {
    if (nextFrame < header.frames) {
      if (block === null) { block = new Uint8Array(blockCount(nextFrame) * rowBytes()); fill = 0; }
      const take = Math.min(block.length - fill, bytes.length - at);
      block.set(bytes.subarray(at, at + take), fill); fill += take; at += take;
      if (!initialSent && nextFrame === 0 && fill >= rowBytes()) {
        initialSent = true;
        const initial = block.slice(0, rowBytes()).buffer as ArrayBuffer;
        send({ type: 'header', header, initial }, [initial]);
      }
      if (fill === block.length) {
        const count = block.length / rowBytes();
        const lastRow = block.slice((count - 1) * rowBytes()).buffer as ArrayBuffer;
        const last = frameOf(blockOf(lastRow), 0, nextFrame + count - 1, header);
        if (!loaded.has(blockIndex(nextFrame))) emitBlock(block.buffer as ArrayBuffer, nextFrame, previous, true);
        else extendThrough();
        previous = last; nextFrame += count; block = null;
      }
    } else {
      tail = concat(tail, bytes.subarray(at)); at = bytes.length;
      if (tail.length > VERDICT_BYTES) throw new Error('Unexpected data after replay verdict');
      if (tail.length === VERDICT_BYTES) verdict(tail);
    }
  }
  progress();
}

async function range(first: number, last: number): Promise<ArrayBuffer> {
  const response = await fetch(request.url, { headers: { Range: `bytes=${first}-${last}` }, signal: abort.signal });
  if (response.status !== 206) { await response.body?.cancel(); throw new Error('Completed replay cache changed; retry this selection'); }
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== last - first + 1) throw new Error('Incomplete replay range');
  return bytes;
}

async function fetchBlocks(first: number, last: number, frames: boolean): Promise<void> {
  const start = Math.max(0, blockStart(first) - 1), end = blockStart(last) + blockCount(blockStart(last));
  const bytes = await range(header!.offset + start * rowBytes(), header!.offset + end * rowBytes() - 1);
  for (let index = first; index <= last; index++) {
    const from = blockStart(index), at = from - start;
    const prev = from > 0 ? frameOf(blockOf(bytes.slice((at - 1) * rowBytes(), at * rowBytes())), 0, from - 1, header!) : null;
    emitBlock(bytes.slice(at * rowBytes(), (at + blockCount(from)) * rowBytes()), from, prev, frames);
  }
}

async function seek(): Promise<void> {
  if (seeking || !cached || header === null || stopped) return;
  seeking = true;
  try {
    for (;;) {
      const target = blockIndex(Math.min(wanted, header.frames - 1));
      if (decoder.sequential) {
        while (!stopped && eventsThrough + 1 < header.frames && blockIndex(eventsThrough + 1) < target) { await fetchBlocks(blockIndex(eventsThrough + 1), blockIndex(eventsThrough + 1), false); progress(); }
      }
      const first = Math.max(0, target - HISTORY_BLOCKS);
      if (!loaded.has(target) || !loaded.has(first)) await fetchBlocks(first, target, true);
      progress(true);
      if (blockIndex(Math.min(wanted, header.frames - 1)) === target) break;
    }
  } catch (error) { fail(error); }
  finally { seeking = false; }
}

async function load(): Promise<void> {
  const response = await fetch(request.url, { headers: { Range: `bytes=0-${PROBE_BYTES - 1}` }, signal: abort.signal });
  if (!response.ok || !response.body) throw new Error(`${response.status} replay: ${await response.text()}`);
  cached = response.status === 206;
  if (cached) {
    const match = /^bytes \d+-\d+\/(\d+)$/.exec(response.headers.get('Content-Range') ?? '');
    if (!match) throw new Error('Replay range length is missing');
    totalBytes = Number(match[1]);
  }
  for await (const chunk of response.body) { cursor += chunk.length; consume(chunk); }
  if (cached) {
    while (header === null && cursor < totalBytes) {
      const last = Math.min(totalBytes - 1, cursor + PROBE_BYTES - 1);
      const bytes = await range(cursor, last); cursor = last + 1; consume(new Uint8Array(bytes));
    }
    if (header === null) throw new Error('Replay descriptor is incomplete');
    verdict(new Uint8Array(await range(totalBytes - VERDICT_BYTES, totalBytes - 1)));
    void seek();
    while (cursor < totalBytes && !stopped) {
      const last = Math.min(totalBytes - 1, cursor + FETCH_BYTES - 1);
      const bytes = await range(cursor, last); cursor = last + 1; consume(new Uint8Array(bytes));
    }
  } else { cached = true; totalBytes = cursor; }
  if (!verified || header === null || nextFrame !== header.frames) throw new Error('Replay generation interrupted');
  progress(true);
  void seek();
}

onmessage = (event: MessageEvent<TapeRequest>) => {
  const message = event.data;
  if (message.type === 'open') { request = message; void load().catch(fail); }
  else if (message.type === 'seek') { wanted = message.step; void seek(); }
  else for (const index of message.indices) loaded.delete(index);
};
