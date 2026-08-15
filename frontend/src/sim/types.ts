
import type { EnvId } from '@shared/envs';


export type SimEvent =
  | { type: 'goal' }
  | { type: 'done' }
  | { type: 'key'; index: number }
  | { type: 'door'; index: number; open: boolean }
  | { type: 'cue'; value: 0 | 1 }
  | { type: 'teleport' }
  | { type: 'resurface' };


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
  kind: 'grid' | 'cheetah' | 'classic';
  grid?: GridGeometry;
  cheetah?: CheetahGeometry;
  classic?: ClassicGeometry;
}


export interface EnvDecoder<St = unknown> {
  envId: EnvId;
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
