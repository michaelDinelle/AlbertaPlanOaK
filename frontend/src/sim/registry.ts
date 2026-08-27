import type { EnvId } from '@shared/envs';
import type { EnvDecoder } from './types';

import { halfCheetahDecoder } from './envs/halfcheetah';
import { fourRoomsDecoder } from './envs/fourrooms';
import { deepSeaDecoder } from './envs/deepsea';
import { keyCorridorDecoder } from './envs/keycorridor';
import { acrobotDecoder } from './envs/acrobot';
import { mountainCarDecoder } from './envs/mountaincar';
import { cartPoleDecoder } from './envs/cartpole';
import { pendulumDecoder } from './envs/pendulum';
import { chainDecoder } from './envs/chain';
import { largeDeepSeaDecoder } from './envs/largedeepsea';
import { multiKeyCorridorDecoder } from './envs/multikeycorridor';
import { memoryChainDecoder } from './envs/memorychain';
import { proceduralMazeDecoder } from './envs/proceduralmaze';
import { highDimMazeDecoder } from './envs/highdimmaze';
import { poMazeDecoder } from './envs/pomaze';
import { delayedChainDecoder } from './envs/delayedchain';

const TABLE = {
  'HalfCheetah-v4': halfCheetahDecoder,
  'FourRooms-cont-v0': fourRoomsDecoder,
  'DeepSea-cont-v0': deepSeaDecoder,
  'KeyCorridor-cont-v0': keyCorridorDecoder,
  'Acrobot-cont-v0': acrobotDecoder,
  'MountainCar-cont-v0': mountainCarDecoder,
  'CartPole-cont-v0': cartPoleDecoder,
  'Pendulum-cont-v0': pendulumDecoder,
  'Chain-cont-v0': chainDecoder,
  'LargeDeepSea-cont-v0': largeDeepSeaDecoder,
  'MultiKeyCorridor-cont-v0': multiKeyCorridorDecoder,
  'MemoryChain-cont-v0': memoryChainDecoder,
  'ProceduralMaze-cont-v0': proceduralMazeDecoder,
  'HighDimMaze-cont-v0': highDimMazeDecoder,
  'POMaze-cont-v0': poMazeDecoder,
  'DelayedChain-cont-v0': delayedChainDecoder,
} satisfies Record<EnvId, unknown>;

export const decoders = TABLE as Record<EnvId, EnvDecoder>;
