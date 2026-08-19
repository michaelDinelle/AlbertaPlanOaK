import type * as THREE from 'three';
import type { EnvId } from '@shared/envs';
import type { EnvGeometry, SimEvent } from '@sim/types';
import type { DesignTokens } from './tokens';

export type SceneEvent = SimEvent;

export interface FrameState {
  pose: Float32Array;
  done: boolean;
  step: number;
  simTime: number;
}

export interface FrameUpdate {
  state: FrameState;
  prev: FrameState | null;
  events: readonly SceneEvent[];
  dtReal: number;
  scrubbed: boolean;
}

export interface SceneSize { width: number; height: number; dpr: number }

export interface SceneContext {
  renderer: THREE.WebGLRenderer;
  canvas: HTMLCanvasElement;
  envId: EnvId;
  size: SceneSize;
  tokens: DesignTokens;
  geometry: EnvGeometry;
  seed: number;
  time(): number;
}

export interface SceneModule {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  init(ctx: SceneContext): Promise<void>;
  update(frame: FrameUpdate): void;
  resize(size: SceneSize): void;
  setDebug(on: boolean): void;
  dispose(): void;
}

export type SceneFactory = () => SceneModule;
export type SceneRegistry = Record<EnvId, () => Promise<{ default: SceneFactory }>>;
