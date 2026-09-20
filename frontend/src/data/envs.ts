export const ENV_IDS = [
  'FourRoomsHall-cont-v0',
  'FourRoomsHall-moving-cont-v0',
  'DeepSea-cont-v0',
  'Acrobot-cont-v0',
  'MountainCar-cont-v0',
  'Pendulum-cont-v0',
  'Chain-cont-v0',
  'LargeDeepSea-cont-v0',
  'MemoryChain-cont-v0',
  'JellyBeanWorld-cont-v0',
  'JellyBeanWorld-8-cont-v0',
] as const;

export type EnvId = (typeof ENV_IDS)[number];

export function isEnvId(s: string): s is EnvId {
  return (ENV_IDS as readonly string[]).includes(s);
}
