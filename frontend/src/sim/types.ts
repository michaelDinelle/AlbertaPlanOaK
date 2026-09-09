import type { EnvId } from '../data/envs';

export type SimEvent =
  | { type: 'goal' }
  | { type: 'done' }
  | { type: 'key'; index: number }
  | { type: 'door'; index: number; open: boolean }
  | { type: 'cue'; value: 0 | 1 }
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

export interface EventStream {
  steps: Int32Array;
  kinds: Uint16Array;
  cellX: Int32Array;
  cellY: Int32Array;
  types: readonly SimEvent['type'][];
  count: number;
}

export const EMPTY_EVENT_STREAM: EventStream = {
  steps: new Int32Array(0),
  kinds: new Uint16Array(0),
  cellX: new Int32Array(0),
  cellY: new Int32Array(0),
  types: [],
  count: 0,
};

export interface EnvFrame {
  step: number;
  done: number;
  pose: Float32Array;
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

export const CHEETAH_FK_POINTS = [
  'o1', 'bt', 'ft', 'head', 'bs', 'bf', 'toe_b', 'fs', 'ff', 'toe_f',
] as const;

export interface CheetahGeometry {
  fk(qpos: ArrayLike<number>, out?: Float32Array): Float32Array;
  radius: number;
  headRadius: number;
}

export interface ClassicGeometry {
  heightAt?(x: number): number;
  anchors: Record<string, readonly [number, number, number]>;
}

export interface EnvGeometry {
  grid?: GridGeometry;
  cheetah?: CheetahGeometry;
  classic?: ClassicGeometry;
}

export interface EnvDecoder<St = unknown> {
  envId: EnvId;
  seedGeometry: boolean;
  initStatic(seed: number): St;
  geometry(st: St, seed: number): EnvGeometry;
  initFrame(seed: number, st: St): EnvFrame;
  deriveEvents(prev: EnvFrame, next: EnvFrame, st: St): SimEvent[];
}

export interface RawRecording {
  envId: EnvId;
  seed: number;
  data: Float32Array;
  steps: number;
}
