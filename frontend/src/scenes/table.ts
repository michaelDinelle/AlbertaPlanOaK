import type { EnvId } from '@shared/envs';
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
}

export const SCENES: Record<EnvId, SceneRow> = {
  'HalfCheetah-v4': { slug: 'half-cheetah', bloom: 0.42, fog: { from: ['dusk', 'fog'], blend: [[['dusk', 'skyTop'], 0.55]] } },
  'FourRooms-cont-v0': { slug: 'four-rooms', bloom: 0.45, burst: 110, fog: { from: ['noon', 'fog'] } },
  'DeepSea-cont-v0': { slug: 'deep-sea', bloom: 0.52, burst: 96, fog: { from: ['underwater', 'fog'] } },
  'KeyCorridor-cont-v0': { slug: 'key-corridor', bloom: 0.3, burst: 110, fog: { from: ['morning', 'fog'] } },
  'Acrobot-cont-v0': { slug: 'acrobot', bloom: 0.46, burst: 110, fog: { from: ['dawn', 'fog'] } },
  'MountainCar-cont-v0': { slug: 'mountain-car', bloom: 0.44, burst: 84, fog: { from: ['dusk', 'fog'] } },
  'CartPole-cont-v0': { slug: 'cart-pole', bloom: 0.34, fog: { from: ['noon', 'fog'] } },
  'Pendulum-cont-v0': { slug: 'pendulum', bloom: 0.54, fog: { from: ['night', 'fog'] } },
  'Chain-cont-v0': { slug: 'chain', bloom: 0.34, burst: 120, fog: { from: ['afternoon', 'fog'] } },
  'LargeDeepSea-cont-v0': { slug: 'large-deep-sea', bloom: 0.48, burst: 130, fog: { from: ['abyss', 'fog'] } },
  'MultiKeyCorridor-cont-v0': { slug: 'multi-key-corridor', bloom: 0.42, burst: 120, fog: { from: ['morning', 'fog'] } },
  'MemoryChain-cont-v0': { slug: 'memory-chain', bloom: 0.62, burst: 130, fog: { from: ['cave', 'fog'] } },
  'ProceduralMaze-cont-v0': { slug: 'procedural-maze', bloom: 0.44, burst: 110, fog: { from: ['noon', 'fog'] } },
  'HighDimMaze-cont-v0': { slug: 'high-dim-maze', bloom: 0.42, burst: 110, fog: { from: ['afternoon', 'fog'] } },
  'POMaze-cont-v0': { slug: 'po-maze', bloom: 0.38, burst: 120, fog: { from: ['dawn', 'fog'], blend: [[['dawn', 'skyTop'], 0.62], [['night', 'skyTop'], 0.62]] } },
  'DelayedChain-cont-v0': { slug: 'delayed-chain', bloom: 0.62, burst: 150, fog: { from: ['night', 'fog'] } },
  'JellyBeanWorld-cont-v0': { slug: 'jelly-bean-world', bloom: 0.38, burst: 36, fog: { from: ['afternoon', 'fog'] } },
};

export function slugOf(envId: EnvId): string {
  return SCENES[envId].slug;
}

export function nameOf(envId: EnvId): string {
  return envId.slice(0, envId.indexOf('-'));
}
