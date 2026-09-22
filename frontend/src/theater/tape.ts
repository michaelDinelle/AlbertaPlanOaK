import { blockOf, frameOf, type Block } from '@sim/events';
import { emptyStream, NO_EVENTS, SIM_EVENT_TYPES, type EnvDecoder, type EnvFrame, type EnvGeometry, type EventStream, type SimEvent } from '@sim/types';
import type { EnvDescriptor } from '../data/envs';
import type { TapeHeader } from '../data/replay';
import { SCENES } from '../scenes/table';
import type { Tick } from './recording';
import type { TapeMessage, TapeRequest } from './tape.worker';

export const WINDOW_BLOCKS = 32;
const NOTIFY_MS = 250;

export interface Tape {
  environment: EnvDescriptor;
  seed: number;
  decoder: EnvDecoder;
  geometry: EnvGeometry | null;
  readonly frameCount: number;
  readonly events: EventStream;
  readonly ticks: Tick[];
  readonly error: Error | null;
  hasFrame(i: number): boolean;
  frameAt(i: number): EnvFrame;
  eventsAt(i: number): readonly SimEvent[];
  follow(i: number): void;
  seek(i: number): void;
  onChange: () => void;
  drop(): void;
}

export function openTape(runId: string, lane: number, environment: EnvDescriptor, seed: number, horizon: number, hash: bigint): Tape {
  const row = SCENES[environment.family];
  if (!row) throw new Error(`No scene is registered for the ${environment.family} family`);
  const decoder = row.decoder;
  const worker = new Worker('/workers/tape.worker.js', { type: 'module' });
  const blocks = new Map<number, Block>();
  const events = emptyStream();
  let header: TapeHeader | null = null, initial: EnvFrame | null = null, through = -1, cursor = 0;
  let error: Error | null = null, ticks: Tick[] = [];
  let timer: ReturnType<typeof setTimeout> | null = null;
  const post = (message: TapeRequest) => worker.postMessage(message);
  const notify = (immediate = false) => {
    if (immediate) { if (timer !== null) { clearTimeout(timer); timer = null; } tape.onChange(); return; }
    if (timer === null) timer = setTimeout(() => { timer = null; tape.onChange(); }, NOTIFY_MS);
  };
  const blockIndex = (step: number): number => Math.floor(step / header!.blockFrames);
  const evict = (): void => {
    if (header === null) return;
    const centre = blockIndex(cursor), gone: number[] = [];
    for (const index of blocks.keys()) if (Math.abs(index - centre) > WINDOW_BLOCKS) { blocks.delete(index); gone.push(index); }
    if (gone.length > 0) post({ type: 'forget', indices: gone });
  };
  const hasFrame = (i: number): boolean => {
    const step = Math.floor(i);
    return header !== null && step >= 0 && step < header.frames && (step === 0 && initial !== null || blocks.has(blockIndex(step))) && (!decoder.sequential || step <= through);
  };
  const frameAt = (i: number): EnvFrame => {
    const step = Math.floor(i);
    if (!hasFrame(step)) throw new Error(`Replay frame ${step} is not available`);
    if (step === 0) return initial!;
    return frameOf(blocks.get(blockIndex(step))!, step % header!.blockFrames, step, header!);
  };
  const tape: Tape = {
    environment, seed, decoder, geometry: null,
    get frameCount() { return Math.max(initial ? 1 : 0, through + 1); },
    get events() { return events; }, get ticks() { return ticks; }, get error() { return error; },
    hasFrame, frameAt,
    eventsAt(i) {
      if (!header || !hasFrame(i)) return NO_EVENTS;
      const step = Math.floor(i), block = events.blocks[blockIndex(step)];
      if (!block) return NO_EVENTS;
      let lo = 0, hi = block.steps.length;
      while (lo < hi) { const mid = (lo + hi) >>> 1; if (block.steps[mid]! < step) lo = mid + 1; else hi = mid; }
      const out: SimEvent[] = [];
      for (let at = lo; at < block.steps.length && block.steps[at] === step; at++) {
        const type = SIM_EVENT_TYPES[block.kinds[at]!]!;
        if (type === 'collection') out.push({ type, item: (['banana', 'onion', 'jellyBean', 'truffle'] as const)[block.values[at]!]!, reward: frameAt(step).reward as -1 | 0 | 1, x: block.cellX[at]!, y: block.cellY[at]! });
        else if (type === 'cue') out.push({ type, value: block.values[at]! });
        else if (type === 'key' || type === 'door') throw new Error('Unexpected replay event');
        else out.push({ type });
      }
      return out;
    },
    follow(i) { cursor = Math.floor(i); },
    seek(step) { cursor = Math.floor(step); evict(); post({ type: 'seek', step: cursor }); },
    onChange: () => {},
    drop() { worker.terminate(); if (timer !== null) clearTimeout(timer); blocks.clear(); events.blocks.length = 0; },
  };
  worker.onmessage = (event: MessageEvent<TapeMessage>) => {
    const message = event.data;
    try {
      if (message.type === 'header') {
        header = message.header; events.blockFrames = header.blockFrames;
        initial = frameOf(blockOf(message.initial), 0, 0, header); through = Math.max(through, 0);
        tape.geometry = decoder.geometry(header.descriptor, initial);
        notify(true);
        return;
      }
      if (message.type === 'block') {
        blocks.set(message.index, blockOf(message.data));
        if (message.events) { events.blocks[message.index] = message.events; events.count += message.events.steps.length; }
        through = Math.max(through, message.through);
        evict();
      } else if (message.type === 'events') {
        events.blocks[message.index] = message.events; events.count += message.events.steps.length;
        through = Math.max(through, message.through);
      } else if (message.type === 'progress') { ticks = message.ticks; through = Math.max(through, message.through); }
      else { error = new Error(message.message); notify(true); return; }
    } catch (caught) { error = caught instanceof Error ? caught : new Error(String(caught)); worker.terminate(); notify(true); return; }
    notify();
  };
  worker.onerror = event => { error = new Error(event.message || 'Replay worker failed to load'); notify(true); };
  post({ type: 'open', url: `/api/runs/${encodeURIComponent(runId)}/replay/${lane}`, envId: environment.envId, seed, horizon, hash: hash.toString() });
  return tape;
}
