import type { TapeHeader } from '../data/replay';
import { SIM_EVENT_TYPES, type EnvDecoder, type EnvFrame, type EventBlock, type EventStream, type SimEvent } from './types';

export interface Block { u32: Uint32Array; f32: Float32Array }

export const blockOf = (buffer: ArrayBuffer): Block => ({ u32: new Uint32Array(buffer), f32: new Float32Array(buffer) });

export function frameOf(block: Block, row: number, step: number, header: TapeHeader): EnvFrame {
  const env = header.descriptor.environment, at = row * header.width, viewAt = at + env.words, obsAt = viewAt + env.view, actAt = obsAt + env.obs;
  const done = block.f32[at + header.width - 1]!;
  return { step, words: block.u32.subarray(at, viewAt), view: block.u32.subarray(viewAt, obsAt), observations: block.f32.subarray(obsAt, actAt), actions: block.f32.subarray(actAt, at + header.width - 2), reward: block.f32[at + header.width - 2]!, done, reset: done !== 0 };
}

export function deriveBlock(block: Block, from: number, header: TapeHeader, decoder: EnvDecoder, previous: EnvFrame | null): EventBlock {
  const steps: number[] = [], kinds: number[] = [], xs: number[] = [], ys: number[] = [], values: number[] = [];
  const count = block.f32.length / header.width;
  for (let row = 0; row < count; row++) {
    const next = frameOf(block, row, from + row, header);
    if (previous !== null) for (const event of decoder.deriveEvents(previous, next)) {
      steps.push(next.step); kinds.push(SIM_EVENT_TYPES.indexOf(event.type));
      values.push(event.type === 'cue' ? event.value : event.type === 'collection' ? ['banana', 'onion', 'jellyBean', 'truffle'].indexOf(event.item) : event.type === 'key' ? event.index : event.type === 'door' ? event.index | (event.open ? 1 << 16 : 0) : 0);
      xs.push(event.type === 'collection' ? event.x : 0); ys.push(event.type === 'collection' ? event.y : 0);
    }
    previous = next;
  }
  return { from, to: from + count - 1, steps: Int32Array.from(steps), kinds: Uint16Array.from(kinds), cellX: Int32Array.from(xs), cellY: Int32Array.from(ys), values: Int32Array.from(values) };
}

export function countEvents(stream: EventStream, type: SimEvent['type'], after: number, upTo: number): number {
  const kind = SIM_EVENT_TYPES.indexOf(type);
  let count = 0;
  for (let b = Math.max(0, Math.floor((after + 1) / stream.blockFrames)); b <= Math.floor(upTo / stream.blockFrames); b++) {
    const block = stream.blocks[b];
    if (!block) continue;
    for (let i = 0; i < block.steps.length; i++) if (block.kinds[i] === kind && block.steps[i]! > after && block.steps[i]! <= upTo) count++;
  }
  return count;
}

export function lastEvent(stream: EventStream, type: SimEvent['type'], step: number): number | undefined {
  const kind = SIM_EVENT_TYPES.indexOf(type);
  for (let b = Math.min(stream.blocks.length - 1, Math.floor(step / stream.blockFrames)); b >= 0; b--) {
    const block = stream.blocks[b];
    if (!block) continue;
    for (let i = block.steps.length - 1; i >= 0; i--) if (block.steps[i]! <= step && block.kinds[i] === kind) return block.steps[i];
  }
  return undefined;
}
