import type { EnvId } from '@shared/envs';
import type { Palette, PaletteName } from '@kit/tokens';

export type PaletteToken = readonly [PaletteName, keyof Palette];

export interface FogRecipe {
  from: PaletteToken;
  blend?: readonly (readonly [PaletteToken, number])[];
}

export interface SceneRow {
  slug: string;
  title: string;
  bloom: number;
  burst?: number;
  fog: FogRecipe;
}

export const SCENES: Record<EnvId, SceneRow> = {
  'HalfCheetah-v4': {
    slug: 'half-cheetah', title: 'Gold Hour Savanna', bloom: 0.42,
    fog: { from: ['dusk', 'fog'], blend: [[['dusk', 'skyTop'], 0.55]] },
  },
  'FourRooms-cont-v0': {
    slug: 'four-rooms', title: 'Four Seasons Courtyard', bloom: 0.45, burst: 110,
    fog: { from: ['noon', 'fog'] },
  },
  'DeepSea-cont-v0': {
    slug: 'deep-sea', title: 'Sunlit Descent', bloom: 0.52, burst: 96,
    fog: { from: ['underwater', 'fog'] },
  },
  'KeyCorridor-cont-v0': {
    slug: 'key-corridor', title: 'The Ivy Gate', bloom: 0.3, burst: 110,
    fog: { from: ['morning', 'fog'] },
  },
  'Acrobot-cont-v0': {
    slug: 'acrobot', title: 'The Willow Swing', bloom: 0.46, burst: 110,
    fog: { from: ['dawn', 'fog'] },
  },
  'MountainCar-cont-v0': {
    slug: 'mountain-car', title: 'Alpine Valley', bloom: 0.44, burst: 84,
    fog: { from: ['dusk', 'fog'] },
  },
  'CartPole-cont-v0': {
    slug: 'cart-pole', title: 'The Boardwalk', bloom: 0.34,
    fog: { from: ['noon', 'fog'] },
  },
  'Pendulum-cont-v0': {
    slug: 'pendulum', title: 'The Lantern Bell', bloom: 0.54,
    fog: { from: ['night', 'fog'] },
  },
  'Chain-cont-v0': {
    slug: 'chain', title: 'Sixteen Stones', bloom: 0.34, burst: 120,
    fog: { from: ['afternoon', 'fog'] },
  },
  'LargeDeepSea-cont-v0': {
    slug: 'large-deep-sea', title: 'The Abyss', bloom: 0.48, burst: 130,
    fog: { from: ['abyss', 'fog'] },
  },
  'MultiKeyCorridor-cont-v0': {
    slug: 'multi-key-corridor', title: 'Temple of Three Gates', bloom: 0.42, burst: 120,
    fog: { from: ['morning', 'fog'] },
  },
  'MemoryChain-cont-v0': {
    slug: 'memory-chain', title: 'The Firefly Cave', bloom: 0.62, burst: 130,
    fog: { from: ['cave', 'fog'] },
  },
  'ProceduralMaze-cont-v0': {
    slug: 'procedural-maze', title: 'The Overgrown Ruins', bloom: 0.44, burst: 110,
    fog: { from: ['noon', 'fog'] },
  },
  'HighDimMaze-cont-v0': {
    slug: 'high-dim-maze', title: 'The Slot Canyons', bloom: 0.42, burst: 110,
    fog: { from: ['afternoon', 'fog'] },
  },
  'POMaze-cont-v0': {
    slug: 'po-maze', title: 'The Fogbound Forest', bloom: 0.38, burst: 120,
    fog: { from: ['dawn', 'fog'], blend: [[['dawn', 'skyTop'], 0.62], [['night', 'skyTop'], 0.62]] },
  },
  'DelayedChain-cont-v0': {
    slug: 'delayed-chain', title: 'The Long Night Road', bloom: 0.62, burst: 150,
    fog: { from: ['night', 'fog'] },
  },
};

export function slugOf(envId: EnvId): string {
  return SCENES[envId].slug;
}

export function nameOf(envId: EnvId): string {
  return envId.slice(0, envId.indexOf('-'));
}
