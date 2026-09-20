import { frameOf } from '@sim/events';
import { emptyStream, NO_EVENTS, SIM_EVENT_TYPES, type EnvFrame, type EnvGeometry, type EventStream, type SimEvent } from '@sim/types';
import type { EnvDescriptor } from '../data/envs';
import type { TapeHeader } from '../data/replay';
import { SCENES } from '../scenes/table';
import type { Tick } from './recording';
import type { TapeMessage, TapeRequest } from './tape.worker';

export interface Tape {
  environment: EnvDescriptor;
  seed: number;
  controlHz: number;
  discretePose: boolean;
  wrapMask: number;
  geometry: EnvGeometry | null;
  readonly frameCount: number;
  readonly events: EventStream;
  readonly ticks: Tick[];
  readonly identical: boolean | null;
  readonly error: Error | null;
  hasFrame(i: number): boolean;
  frameAt(i: number): EnvFrame;
  eventsAt(i: number): readonly SimEvent[];
  seek(i: number): void;
  onChange: () => void;
  drop(): void;
}

export function openTape(runId: string, lane: number, environment: EnvDescriptor, seed: number, horizon: number, hash: bigint): Tape {
  const family = SCENES[environment.family];
  if (!family) throw new Error('Visualization unavailable');
  const worker = new Worker('/workers/tape.worker.js', { type: 'module' });
  const blocks = new Map<number, Float32Array>();
  const events = emptyStream();
  let header: TapeHeader | null = null, initial: EnvFrame | null = null, through = -1;
  let identical: boolean | null = null, error: Error | null = null, ticks: Tick[] = [];
  let timer: ReturnType<typeof setTimeout> | null = null;
  const post = (message: TapeRequest) => worker.postMessage(message);
  const notify = () => { if (timer === null) timer = setTimeout(() => { timer = null; tape.onChange(); }, 50); };
  const hasFrame = (i: number): boolean => {
    const step = Math.floor(i);
    return header !== null && step >= 0 && step < header.frames && (step === 0 && initial !== null || blocks.has(Math.floor(step / header.blockFrames))) && (environment.family !== 'jbw' || step <= through);
  };
  const frameAt = (i: number): EnvFrame => {
    const step = Math.floor(i);
    if (!hasFrame(step)) throw new Error(`Replay frame ${step} is not available`);
    if (step === 0) return initial!;
    return frameOf(blocks.get(Math.floor(step / header!.blockFrames))!, step % header!.blockFrames, step, header!);
  };
  const tape: Tape = {
    environment, seed, controlHz: family.controlHz, discretePose: family.discretePose, wrapMask: family.wrapMask, geometry: null,
    get frameCount() { return Math.max(initial ? 1 : 0, through + 1); },
    get events() { return events; }, get ticks() { return ticks; }, get identical() { return identical; }, get error() { return error; },
    hasFrame, frameAt,
    eventsAt(i) {
      if (!header || !hasFrame(i)) return NO_EVENTS;
      const step = Math.floor(i), block = events.blocks[Math.floor(step / header.blockFrames)];
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
    seek: step => post({ type: 'seek', step: Math.floor(step) }),
    onChange: () => {},
    drop() { worker.terminate(); if (timer !== null) clearTimeout(timer); blocks.clear(); events.blocks.length = 0; },
  };
  worker.onmessage = (event: MessageEvent<TapeMessage>) => {
    const message = event.data;
    try {
      if (message.type === 'header') {
        header = message.header; events.blockFrames = header.blockFrames;
        initial = frameOf(new Float32Array(message.initial), 0, 0, header); through = 0;
        tape.geometry = family.decoder.geometry(header.descriptor, initial);
        tape.onChange();
      } else if (message.type === 'block') {
        blocks.set(message.index, new Float32Array(message.data));
        events.blocks[message.index] = message.events; events.count += message.events.steps.length;
        through = Math.max(through, message.through);
      } else if (message.type === 'progress') { ticks = message.ticks; through = Math.max(through, message.through); }
      else if (message.type === 'verdict') identical = message.identical;
      else error = new Error(message.message);
    } catch (caught) { error = caught instanceof Error ? caught : new Error(String(caught)); worker.terminate(); }
    notify();
  };
  worker.onerror = event => { error = new Error(event.message || 'Replay worker failed to load'); notify(); };
  post({ type: 'open', url: `/api/runs/${encodeURIComponent(runId)}/replay/${lane}`, envId: environment.envId, seed, horizon, hash: hash.toString() });
  return tape;
}
