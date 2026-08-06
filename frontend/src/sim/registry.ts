
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

export const decoders: Record<EnvId, EnvDecoder<unknown, unknown>> = {
  'HalfCheetah-v4': halfCheetahDecoder as EnvDecoder<unknown, unknown>,
  'FourRooms-cont-v0': fourRoomsDecoder as EnvDecoder<unknown, unknown>,
  'DeepSea-cont-v0': deepSeaDecoder as EnvDecoder<unknown, unknown>,
  'KeyCorridor-cont-v0': keyCorridorDecoder as EnvDecoder<unknown, unknown>,
  'Acrobot-cont-v0': acrobotDecoder as EnvDecoder<unknown, unknown>,
  'MountainCar-cont-v0': mountainCarDecoder as EnvDecoder<unknown, unknown>,
  'CartPole-cont-v0': cartPoleDecoder as EnvDecoder<unknown, unknown>,
  'Pendulum-cont-v0': pendulumDecoder as EnvDecoder<unknown, unknown>,
  'Chain-cont-v0': chainDecoder as EnvDecoder<unknown, unknown>,
  'LargeDeepSea-cont-v0': largeDeepSeaDecoder as EnvDecoder<unknown, unknown>,
  'MultiKeyCorridor-cont-v0': multiKeyCorridorDecoder as EnvDecoder<unknown, unknown>,
  'MemoryChain-cont-v0': memoryChainDecoder as EnvDecoder<unknown, unknown>,
  'ProceduralMaze-cont-v0': proceduralMazeDecoder as EnvDecoder<unknown, unknown>,
  'HighDimMaze-cont-v0': highDimMazeDecoder as EnvDecoder<unknown, unknown>,
  'POMaze-cont-v0': poMazeDecoder as EnvDecoder<unknown, unknown>,
  'DelayedChain-cont-v0': delayedChainDecoder as EnvDecoder<unknown, unknown>,
};

export function decoderFor(envId: EnvId): EnvDecoder<unknown, unknown> {
  const d = decoders[envId];
  if (!d) throw new Error(`no decoder registered for ${envId}`);
  return d;
}
