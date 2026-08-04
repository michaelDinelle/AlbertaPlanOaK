// FROZEN Phase-0 contract: the SceneModule interface every artisan scene implements,
// and the host/scene ownership boundary. Additive changes only after Phase 0.
//
// HOST OWNS: the one WebGLRenderer, canvas + resize, playback clock + transport,
// the EffectComposer chain (scenes parameterize it via manifest.postConfig only),
// quality-tier selection + frame governor, thumbnails, the debug toggle.
// SCENE OWNS: its THREE.Scene, fog, camera (built from kit rigs), lights (kit rig
// presets only), all meshes, ambient motion.

import type * as THREE from 'three';
import type { EnvId } from '@shared/envs';
import type { EnvGeometry, SimEvent } from '@sim/types';
import type { DesignTokens } from './tokens';

export type SceneEvent = SimEvent;
export type QualityTier = 'hero' | 'medium' | 'thumb';

export interface SceneManifest {
  envId: EnvId;
  /** must equal shared/envs.ts slug (scene dir name) */
  slug: string;
  /** scene display title, e.g. "Gold Hour Savanna" */
  title: string;
  /** one-sentence art brief */
  logline: string;
  /** key into DesignTokens.accents (normally === slug) */
  accent: string;
  /** all three tiers are mandatory */
  tiers: readonly QualityTier[];
  /** canonical pose (9 floats) + derived state for host-side thumbnail render */
  thumbFrame: { pose: readonly number[]; derived?: unknown };
  postConfig: {
    bloomStrength: number;   // 0.25–0.9
    bloomThreshold: number;  // ~0.85
    fogTint: number;         // hex, grade pass tint
  };
  /** self-declared hero-tier budget, checked in review against renderer.info */
  budget: { drawCalls: number; triangles: number };
}

export interface FrameState {
  /** 9 floats — interpolated by the host for continuous envs, snapped for grids */
  pose: Float32Array;
  ctrl: Float32Array;
  done: boolean;
  /** integer record step of the underlying frame (floor when interpolating) */
  step: number;
  /** step / controlHz, seconds */
  simTime: number;
  /** env-specific decoded semantic state from sim-core (cast at use site) */
  derived: unknown;
}

export interface FrameUpdate {
  state: FrameState;
  prev: FrameState | null;
  /** ONLY from sim-core, ONLY when stepping sequentially — never synthesized by scenes */
  events: readonly SceneEvent[];
  /** wall-clock seconds since last render — ambient animation only */
  dtReal: number;
  /** true on any non-sequential jump: SNAP all easing, fire no celebrations */
  scrubbed: boolean;
}

export interface SceneSize { width: number; height: number; dpr: number }

export interface SceneContext {
  /** read capabilities only — never reconfigure (autoClear, clear color, tone mapping) */
  renderer: THREE.WebGLRenderer;
  canvas: HTMLCanvasElement;
  size: SceneSize;
  tier: QualityTier;
  tokens: DesignTokens;
  /** ground truth from sim-core — scenes NEVER re-derive geometry */
  geometry: EnvGeometry;
  /** lane seed — the ONLY seed allowed for scene-local deterministic scatter */
  seed: number;
  /** host playback clock, seconds — the ONLY clock for shader/ambient animation */
  time(): number;
}

/** Alignment probe for the headless accuracy self-check (kit assertAligned). */
export interface AlignmentProbe {
  name: string;
  /** the state-bearing object whose world position is checked */
  object: THREE.Object3D;
  /** ground-truth world position for a given state */
  expectWorldPos(state: FrameState): readonly [number, number, number];
  /** world-unit tolerance (default 1e-3) */
  tolerance?: number;
}

/** Canonical poses each scene ships in fixtures.ts (≥3) for assertAligned + review. */
export interface SceneFixture {
  name: string;
  pose: readonly number[];      // 9
  ctrl?: readonly number[];     // 6
  derived?: unknown;
  step?: number;
}

export interface SceneModule {
  readonly manifest: SceneManifest;
  /** valid after init() resolves */
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  init(ctx: SceneContext): Promise<void>;
  /** ZERO allocation; idempotent w.r.t. state (scrub-safe); snap on scrubbed/teleport */
  update(frame: FrameUpdate): void;
  resize(size: SceneSize): void;
  /** ground-truth overlay (kit/debug GroundTruthOverlay) — the QA instrument */
  setDebug(on: boolean): void;
  /** state-bearing objects for the headless accuracy check */
  probes(): AlignmentProbe[];
  /** full GPU release; init/dispose must round-trip clean (checked ×3 in review) */
  dispose(): void;
}

export type SceneFactory = () => SceneModule;
export type SceneRegistry = Partial<Record<EnvId, () => Promise<{ default: SceneFactory }>>>;
