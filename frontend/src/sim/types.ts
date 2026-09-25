import type { ReplayDescriptor } from '../data/replay';

export type SimEvent =
  | { type: 'goal' }
  | { type: 'harm' }
  | { type: 'done' }
  | { type: 'key'; index: number }
  | { type: 'door'; index: number; open: boolean }
  | { type: 'cue'; value: number }
  | { type: 'teleport' }
  | { type: 'resurface' }
  | {
      type: 'collection';
      item: 'banana' | 'onion' | 'jellyBean' | 'truffle';
      reward: -1 | 0 | 1;
      x: number;
      y: number;
    };

export const NO_EVENTS: SimEvent[] = Object.freeze([]) as unknown as SimEvent[];

export const SIM_EVENT_TYPES = [
  'goal',
  'harm',
  'done',
  'key',
  'door',
  'cue',
  'teleport',
  'resurface',
  'collection',
] as const satisfies readonly SimEvent['type'][];

type CoversAllSimEvents = Exclude<SimEvent['type'], (typeof SIM_EVENT_TYPES)[number]>;
export const SIM_EVENT_TYPES_COMPLETE: CoversAllSimEvents extends never ? true : never = true;

export interface EventBlock {
  from: number;
  to: number;
  steps: Int32Array;
  kinds: Uint16Array;
  cellX: Int32Array;
  cellY: Int32Array;
  values: Int32Array;
}

export interface EventStream {
  blocks: (EventBlock | undefined)[];
  blockFrames: number;
  count: number;
}

export const emptyStream = (blockFrames = 256): EventStream => ({ blocks: [], blockFrames, count: 0 });
export const EMPTY_EVENT_STREAM = emptyStream();

export interface EnvFrame {
  step: number;
  done: number;
  words: Uint32Array;
  view: Uint32Array;
  observations: Float32Array;
  actions: Float32Array;
  reward: number;
  reset: boolean;
}

export interface GridCell { x: number; y: number }

export interface GridGeometry {
  w: number;
  h: number;
  walls: Uint8Array;
  goal?: GridCell;
  doors?: GridCell[];
  keys?: GridCell[];
  hallways?: GridCell[];
  actionMask?: Uint8Array;
  carvedPath?: GridCell[];
}

export interface ClassicGeometry {
  heightAt?(x: number): number;
  anchors: Record<string, readonly [number, number, number]>;
}

export interface EnvGeometry {
  grid?: GridGeometry;
  classic?: ClassicGeometry;
}

export interface EnvDecoder {
  slots: number;
  wrap?: number;
  discrete?: boolean;
  sequential?: boolean;
  decode(frame: EnvFrame, out: Float32Array): void;
  geometry(descriptor: ReplayDescriptor, initial: EnvFrame): EnvGeometry;
  deriveEvents(prev: EnvFrame, next: EnvFrame): SimEvent[];
}
