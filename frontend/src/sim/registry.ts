import type { EnvId } from '../data/envs';
import type { EnvDecoder } from './types';

import { fourRoomsDecoder } from './envs/fourrooms';
import { deepSeaDecoder } from './envs/deepsea';
import { acrobotDecoder } from './envs/acrobot';
import { mountainCarDecoder } from './envs/mountaincar';
import { pendulumDecoder } from './envs/pendulum';
import { chainDecoder } from './envs/chain';
import { largeDeepSeaDecoder } from './envs/largedeepsea';
import { memoryChainDecoder } from './envs/memorychain';
import { jellyBeanWorldDecoder } from './envs/jellybeanworld';

const TABLE = {
  'FourRoomsHall-cont-v0': fourRoomsDecoder,
  'FourRoomsHall-moving-cont-v0': fourRoomsDecoder,
  'DeepSea-cont-v0': deepSeaDecoder,
  'Acrobot-cont-v0': acrobotDecoder,
  'MountainCar-cont-v0': mountainCarDecoder,
  'Pendulum-cont-v0': pendulumDecoder,
  'Chain-cont-v0': chainDecoder,
  'LargeDeepSea-cont-v0': largeDeepSeaDecoder,
  'MemoryChain-cont-v0': memoryChainDecoder,
  'JellyBeanWorld-cont-v0': jellyBeanWorldDecoder,
  'JellyBeanWorld-8-cont-v0': jellyBeanWorldDecoder,
} satisfies Record<EnvId, unknown>;

export const decoders = TABLE as Record<EnvId, EnvDecoder>;

export interface Playback {
  controlHz: number;
  discretePose: boolean;
  wrapMask: number;
}

const grid: Playback = { controlHz: 8, discretePose: true, wrapMask: 0 };

export const PLAYBACK: Record<EnvId, Playback> = {
  'FourRoomsHall-cont-v0': grid,
  'FourRoomsHall-moving-cont-v0': grid,
  'DeepSea-cont-v0': grid,
  'Acrobot-cont-v0': { controlHz: 5, discretePose: false, wrapMask: 0b11 },
  'MountainCar-cont-v0': { controlHz: 30, discretePose: false, wrapMask: 0 },
  'Pendulum-cont-v0': { controlHz: 20, discretePose: false, wrapMask: 0b1 },
  'Chain-cont-v0': grid,
  'LargeDeepSea-cont-v0': grid,
  'MemoryChain-cont-v0': grid,
  'JellyBeanWorld-cont-v0': grid,
  'JellyBeanWorld-8-cont-v0': grid,
};
