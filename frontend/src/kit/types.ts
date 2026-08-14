
import type * as THREE from 'three';
import type { EnvId } from '@shared/envs';
import type { EnvGeometry, SimEvent } from '@sim/types';
import type { DesignTokens } from './tokens';

export type SceneEvent = SimEvent;
export type QualityTier = 'hero' | 'medium' | 'thumb';

export interface SceneManifest {
  envId: EnvId;
  slug: string;
  title: string;
  logline: string;
  accent: string;
  tiers: readonly QualityTier[];
  thumbFrame: { pose: readonly number[]; derived?: unknown };
  postConfig: {
    bloomStrength: number;
    bloomThreshold: number;
    fogTint: number;
  };
  budget: { drawCalls: number; triangles: number };
}

export interface FrameState {
  pose: Float32Array;
  ctrl: Float32Array;
  done: boolean;
  step: number;
  simTime: number;
  derived: unknown;
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
  size: SceneSize;
  tier: QualityTier;
  tokens: DesignTokens;
  geometry: EnvGeometry;
  seed: number;
  time(): number;
}

export interface SceneModule {
  readonly manifest: SceneManifest;
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
