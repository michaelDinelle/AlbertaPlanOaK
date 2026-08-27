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
  'JellyBeanWorld-cont-v0',
] as const;

export type EnvId = (typeof ENV_IDS)[number];

export function isEnvId(s: string): s is EnvId {
  return (ENV_IDS as readonly string[]).includes(s);
}
