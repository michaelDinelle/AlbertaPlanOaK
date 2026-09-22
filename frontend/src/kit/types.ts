import type * as THREE from 'three';
import type { EnvDescriptor } from '../data/envs';
import type { EnvFrame, EnvGeometry, EventStream, SimEvent } from '@sim/types';
import type { DesignTokens, Palette, PaletteName } from './tokens';

export type PaletteToken = readonly [PaletteName, keyof Palette];

export interface FogRecipe {
  from: PaletteToken;
  blend?: readonly (readonly [PaletteToken, number])[];
}

export interface ScenePost {
  bloom: number;
  fog: FogRecipe;
}

export type SceneEvent = SimEvent;

export interface FrameState {
  pose: Float32Array;
  observations: Float32Array;
  reward: number;
  reset: boolean;
  done: boolean;
  step: number;
  simTime: number;
}

export interface FrameUpdate {
  state: FrameState;
  prev: FrameState | null;
  events: readonly SceneEvent[];
  eventLog: EventStream;
  dtReal: number;
  scrubbed: boolean;
  position: number;
  controlHz: number;
  frameAt(i: number): EnvFrame;
  poseAt(i: number): Float32Array;
}

export interface SceneSize { width: number; height: number; dpr: number }

export interface SceneContext {
  renderer: THREE.WebGLRenderer;
  canvas: HTMLCanvasElement;
  envId: string;
  environment: EnvDescriptor;
  size: SceneSize;
  tokens: DesignTokens;
  geometry: EnvGeometry;
  seed: number;
  time(): number;
}

export interface LegendRow {
  key: string;
  label: string;
  tint: number;
  value(): string;
}

export interface LegendSpec {
  title: string;
  rows: readonly LegendRow[];
  note?(): string;
  bar?(): number;
}

export interface SceneModule {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly post: ScenePost;
  readonly controlHz: number;
  readonly legend: LegendSpec | null;
  init(ctx: SceneContext): Promise<void>;
  update(frame: FrameUpdate): void;
  resize(size: SceneSize): void;
  resetCamera(): void;
  setDebug(on: boolean): void;
  dispose(): void;
}

export type SceneFactory = () => SceneModule;
