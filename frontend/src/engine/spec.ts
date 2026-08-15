import type * as THREE from 'three';
import type { PaletteName } from '@kit/tokens';
import type { QualityTier, SceneContext } from '@kit/types';
import type { SemanticFrame } from './frame';

export interface Detail {
  counts: number;
  segments: number;
  shadows: boolean;
  shadowMap: number;
  renderScale: number;
}

export function detailFor(tier: QualityTier): Detail {
  switch (tier) {
    case 'thumb':
      return { counts: 0.28, segments: 0.5, shadows: false, shadowMap: 0, renderScale: 1 };
    case 'medium':
      return { counts: 0.6, segments: 0.75, shadows: true, shadowMap: 1024, renderScale: 1.5 };
    default:
      return { counts: 1, segments: 1, shadows: true, shadowMap: 2048, renderScale: 2 };
  }
}

export interface GroundSpec {
  kind: 'cells' | 'profile' | 'none';
  terraced?: boolean;
  extent?: number;
  height?(x: number, z: number): number;
  color?: number;
  floorTint?(x: number, y: number): number;
  wallTint?(x: number, y: number): number;
}

export interface CameraSpec {
  target: readonly [number, number, number];
  azimuth: number;
  polar: number;
  distance: number;
  follow?: number;
}

export interface StageContext extends SceneContext {
  detail: Detail;
  group: THREE.Group;
  scene: THREE.Scene;
}

export interface Stage {
  update?(frame: SemanticFrame, dt: number, time: number): void;
  resize?(dpr: number): void;
  dispose?(): void;
}

export type StageBuilder = (ctx: StageContext) => Stage | void;

export interface EnvSpec {
  envId: string;
  title: string;
  logline: string;
  palette: PaletteName;
  accent: string;
  ground: GroundSpec;
  camera: CameraSpec;
  bloom: number;
  fogExtent: number;
  build: StageBuilder;
}
