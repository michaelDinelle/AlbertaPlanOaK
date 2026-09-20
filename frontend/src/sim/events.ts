import type { TapeHeader } from '../data/replay';
import { SIM_EVENT_TYPES, type EnvDecoder, type EnvFrame, type EventBlock, type EventStream, type SimEvent } from './types';

export function frameOf(data: Float32Array, row: number, step: number, header: TapeHeader): EnvFrame {
  const at = row * header.width, env = header.descriptor.environment;
  const pose = data.subarray(at, at + env.pose), observations = data.subarray(at + env.pose, at + env.pose + env.obs);
  const actions = data.subarray(at + env.pose + env.obs, at + header.width - 2);
  const done = data[at + header.width - 1]!;
  return { step, pose, observations, actions, reward: data[at + header.width - 2]!, done, reset: done !== 0 || (env.family === 'deepsea' && step > 0 && pose[0] === 0) };
}

export function deriveBlock(data: Float32Array, from: number, header: TapeHeader, decoder: EnvDecoder, previous: EnvFrame | null): EventBlock {
  const steps: number[] = [], kinds: number[] = [], xs: number[] = [], ys: number[] = [], values: number[] = [];
  const count = data.length / header.width;
  for (let row = 0; row < count; row++) {
    const next = frameOf(data, row, from + row, header);
    if (previous !== null) for (const event of decoder.deriveEvents(previous, next)) {
      steps.push(next.step); kinds.push(SIM_EVENT_TYPES.indexOf(event.type));
      values.push(event.type === 'cue' ? event.value : event.type === 'collection' ? ['banana', 'onion', 'jellyBean', 'truffle'].indexOf(event.item) : 0);
      xs.push(event.type === 'collection' ? event.x : 0); ys.push(event.type === 'collection' ? event.y : 0);
    }
    previous = next;
  }
  return { from, to: from + count - 1, steps: Int32Array.from(steps), kinds: Uint16Array.from(kinds), cellX: Int32Array.from(xs), cellY: Int32Array.from(ys), values: Int32Array.from(values) };
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
