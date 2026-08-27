import type { SceneRegistry } from '@kit/types';

export const sceneRegistry: SceneRegistry = {
  'HalfCheetah-v4': () => import('./half-cheetah/index'),
  'FourRooms-cont-v0': () => import('./four-rooms/index'),
  'DeepSea-cont-v0': () => import('./deep-sea/index'),
  'KeyCorridor-cont-v0': () => import('./key-corridor/index'),
  'Acrobot-cont-v0': () => import('./acrobot/index'),
  'MountainCar-cont-v0': () => import('./mountain-car/index'),
  'CartPole-cont-v0': () => import('./cart-pole/index'),
  'Pendulum-cont-v0': () => import('./pendulum/index'),
  'Chain-cont-v0': () => import('./chain/index'),
  'LargeDeepSea-cont-v0': () => import('./large-deep-sea/index'),
  'MultiKeyCorridor-cont-v0': () => import('./multi-key-corridor/index'),
  'MemoryChain-cont-v0': () => import('./memory-chain/index'),
  'ProceduralMaze-cont-v0': () => import('./procedural-maze/index'),
  'HighDimMaze-cont-v0': () => import('./high-dim-maze/index'),
  'POMaze-cont-v0': () => import('./po-maze/index'),
  'DelayedChain-cont-v0': () => import('./delayed-chain/index'),
  'JellyBeanWorld-cont-v0': () => import('./jelly-bean-world/index'),
};
