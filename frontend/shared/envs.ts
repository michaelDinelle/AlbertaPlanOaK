
export const ENV_IDS = [
  'HalfCheetah-v4',
  'FourRooms-cont-v0',
  'DeepSea-cont-v0',
  'KeyCorridor-cont-v0',
  'Acrobot-cont-v0',
  'MountainCar-cont-v0',
  'CartPole-cont-v0',
  'Pendulum-cont-v0',
  'Chain-cont-v0',
  'LargeDeepSea-cont-v0',
  'MultiKeyCorridor-cont-v0',
  'MemoryChain-cont-v0',
  'ProceduralMaze-cont-v0',
  'HighDimMaze-cont-v0',
  'POMaze-cont-v0',
  'DelayedChain-cont-v0',
] as const;

export type EnvId = (typeof ENV_IDS)[number];

export interface EnvInfo {
  slug: string;
  name: string;
  controlHz: number;
  discretePose: boolean;
}

export const ENVS: Record<EnvId, EnvInfo> = {
  'HalfCheetah-v4': {
    slug: 'half-cheetah', name: 'HalfCheetah',
    controlHz: 20, discretePose: false,
  },
  'FourRooms-cont-v0': {
    slug: 'four-rooms', name: 'FourRooms',
    controlHz: 8, discretePose: true,
  },
  'DeepSea-cont-v0': {
    slug: 'deep-sea', name: 'DeepSea',
    controlHz: 8, discretePose: true,
  },
  'KeyCorridor-cont-v0': {
    slug: 'key-corridor', name: 'KeyCorridor',
    controlHz: 8, discretePose: true,
  },
  'Acrobot-cont-v0': {
    slug: 'acrobot', name: 'Acrobot',
    controlHz: 5, discretePose: false,
  },
  'MountainCar-cont-v0': {
    slug: 'mountain-car', name: 'MountainCar',
    controlHz: 30, discretePose: false,
  },
  'CartPole-cont-v0': {
    slug: 'cart-pole', name: 'CartPole',
    controlHz: 50, discretePose: false,
  },
  'Pendulum-cont-v0': {
    slug: 'pendulum', name: 'Pendulum',
    controlHz: 20, discretePose: false,
  },
  'Chain-cont-v0': {
    slug: 'chain', name: 'Chain',
    controlHz: 8, discretePose: true,
  },
  'LargeDeepSea-cont-v0': {
    slug: 'large-deep-sea', name: 'LargeDeepSea',
    controlHz: 8, discretePose: true,
  },
  'MultiKeyCorridor-cont-v0': {
    slug: 'multi-key-corridor', name: 'MultiKeyCorridor',
    controlHz: 8, discretePose: true,
  },
  'MemoryChain-cont-v0': {
    slug: 'memory-chain', name: 'MemoryChain',
    controlHz: 8, discretePose: true,
  },
  'ProceduralMaze-cont-v0': {
    slug: 'procedural-maze', name: 'ProceduralMaze',
    controlHz: 8, discretePose: true,
  },
  'HighDimMaze-cont-v0': {
    slug: 'high-dim-maze', name: 'HighDimMaze',
    controlHz: 8, discretePose: true,
  },
  'POMaze-cont-v0': {
    slug: 'po-maze', name: 'POMaze',
    controlHz: 8, discretePose: true,
  },
  'DelayedChain-cont-v0': {
    slug: 'delayed-chain', name: 'DelayedChain',
    controlHz: 8, discretePose: true,
  },
};

export function isEnvId(s: string): s is EnvId {
  return (ENV_IDS as readonly string[]).includes(s);
}
