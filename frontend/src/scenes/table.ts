import { ENV_IDS, isEnvId, type EnvId } from '../data/envs';
import type { SceneRegistry } from '@kit/types';
import type { Palette, PaletteName } from '@kit/tokens';

export type PaletteToken = readonly [PaletteName, keyof Palette];

export interface FogRecipe {
  from: PaletteToken;
  blend?: readonly (readonly [PaletteToken, number])[];
}

export interface SceneRow {
  slug: string;
  bloom: number;
  burst?: number;
  fog: FogRecipe;
  load: SceneRegistry[EnvId];
}

export const SCENES: Record<EnvId, SceneRow> = {
  'FourRoomsHall-cont-v0': { slug: 'four-rooms', bloom: 0.45, burst: 110, fog: { from: ['noon', 'fog'] }, load: () => import('./four-rooms/index') },
  'FourRoomsHall-moving-cont-v0': { slug: 'four-rooms', bloom: 0.45, burst: 110, fog: { from: ['noon', 'fog'] }, load: () => import('./four-rooms/index') },
  'DeepSea-cont-v0': { slug: 'deep-sea', bloom: 0.52, burst: 96, fog: { from: ['underwater', 'fog'] }, load: () => import('./deep-sea/index') },
  'Acrobot-cont-v0': { slug: 'acrobot', bloom: 0.46, burst: 110, fog: { from: ['dawn', 'fog'] }, load: () => import('./acrobot/index') },
  'MountainCar-cont-v0': { slug: 'mountain-car', bloom: 0.44, burst: 84, fog: { from: ['dusk', 'fog'] }, load: () => import('./mountain-car/index') },
  'Pendulum-cont-v0': { slug: 'pendulum', bloom: 0.54, fog: { from: ['night', 'fog'] }, load: () => import('./pendulum/index') },
  'Chain-cont-v0': { slug: 'chain', bloom: 0.34, burst: 120, fog: { from: ['afternoon', 'fog'] }, load: () => import('./chain/index') },
  'LargeDeepSea-cont-v0': { slug: 'large-deep-sea', bloom: 0.48, burst: 130, fog: { from: ['abyss', 'fog'] }, load: () => import('./large-deep-sea/index') },
  'MemoryChain-cont-v0': { slug: 'memory-chain', bloom: 0.62, burst: 130, fog: { from: ['cave', 'fog'] }, load: () => import('./memory-chain/index') },
  'JellyBeanWorld-cont-v0': { slug: 'jelly-bean-world', bloom: 0.38, burst: 108, fog: { from: ['afternoon', 'fog'], blend: [[['dusk', 'fog'], 0.45]] }, load: () => import('./jelly-bean-world/index') },
  'JellyBeanWorld-8-cont-v0': { slug: 'jelly-bean-world', bloom: 0.38, burst: 108, fog: { from: ['afternoon', 'fog'], blend: [[['dusk', 'fog'], 0.45]] }, load: () => import('./jelly-bean-world/index') },
};

export const sceneRegistry = Object.fromEntries(
  ENV_IDS.map((envId) => [envId, SCENES[envId].load]),
) as SceneRegistry;

export function slugOf(envId: EnvId): string {
  return SCENES[envId].slug;
}

export function nameOf(envId: string): string {
  return envId.replace(/-cont-v0$/, '');
}

export const sceneOf = (envId: string): SceneRow | null => (isEnvId(envId) ? SCENES[envId] : null);
