// The 16 OaK environments — the single client/server-shared env table.
// Semantics verified against Sources/Gridworld.metal, Sources/ClassicControl.metal,
// Sources/OakKernel.metal (oak_stage_fleet_env pose packing, L2650–2979).
// FROZEN in Phase 0: additive changes only.

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

export type EnvCategory = 'cheetah' | 'classic' | 'grid';

export interface EnvInfo {
  envId: EnvId;
  /** kebab-case directory/route slug */
  slug: string;
  /** display name (env identity, not the scene title — scenes name themselves) */
  name: string;
  category: EnvCategory;
  /** 'box' N means N continuous means; 'discrete' N means N-way categorical */
  action: { kind: 'box' | 'discrete'; n: number };
  /**
   * Steps per simulated second for realtime playback. Gridworlds are unitless;
   * they carry a readable default tempo (8 Hz).
   */
  controlHz: number;
  /** true = integer/cell state, playback snaps (no pose interpolation) */
  discretePose: boolean;
  /** whether the env ever emits done=1 in the record stream */
  emitsDone: boolean;
}

export const ENVS: Record<EnvId, EnvInfo> = {
  'HalfCheetah-v4': {
    envId: 'HalfCheetah-v4', slug: 'half-cheetah', name: 'HalfCheetah',
    category: 'cheetah', action: { kind: 'box', n: 6 }, controlHz: 20,
    discretePose: false, emitsDone: false,
  },
  'FourRooms-cont-v0': {
    envId: 'FourRooms-cont-v0', slug: 'four-rooms', name: 'FourRooms',
    category: 'grid', action: { kind: 'discrete', n: 4 }, controlHz: 8,
    discretePose: true, emitsDone: false,
  },
  'DeepSea-cont-v0': {
    envId: 'DeepSea-cont-v0', slug: 'deep-sea', name: 'DeepSea',
    category: 'grid', action: { kind: 'discrete', n: 2 }, controlHz: 8,
    discretePose: true, emitsDone: false,
  },
  'KeyCorridor-cont-v0': {
    envId: 'KeyCorridor-cont-v0', slug: 'key-corridor', name: 'KeyCorridor',
    category: 'grid', action: { kind: 'discrete', n: 4 }, controlHz: 8,
    discretePose: true, emitsDone: true,
  },
  'Acrobot-cont-v0': {
    envId: 'Acrobot-cont-v0', slug: 'acrobot', name: 'Acrobot',
    category: 'classic', action: { kind: 'discrete', n: 3 }, controlHz: 5,
    discretePose: false, emitsDone: true,
  },
  'MountainCar-cont-v0': {
    envId: 'MountainCar-cont-v0', slug: 'mountain-car', name: 'MountainCar',
    category: 'classic', action: { kind: 'discrete', n: 3 }, controlHz: 30,
    discretePose: false, emitsDone: true,
  },
  'CartPole-cont-v0': {
    envId: 'CartPole-cont-v0', slug: 'cart-pole', name: 'CartPole',
    category: 'classic', action: { kind: 'discrete', n: 2 }, controlHz: 50,
    discretePose: false, emitsDone: true,
  },
  'Pendulum-cont-v0': {
    envId: 'Pendulum-cont-v0', slug: 'pendulum', name: 'Pendulum',
    category: 'classic', action: { kind: 'box', n: 1 }, controlHz: 20,
    discretePose: false, emitsDone: false,
  },
  'Chain-cont-v0': {
    envId: 'Chain-cont-v0', slug: 'chain', name: 'Chain',
    category: 'grid', action: { kind: 'discrete', n: 2 }, controlHz: 8,
    discretePose: true, emitsDone: true,
  },
  'LargeDeepSea-cont-v0': {
    envId: 'LargeDeepSea-cont-v0', slug: 'large-deep-sea', name: 'LargeDeepSea',
    category: 'grid', action: { kind: 'discrete', n: 2 }, controlHz: 8,
    discretePose: true, emitsDone: false,
  },
  'MultiKeyCorridor-cont-v0': {
    envId: 'MultiKeyCorridor-cont-v0', slug: 'multi-key-corridor', name: 'MultiKeyCorridor',
    category: 'grid', action: { kind: 'discrete', n: 4 }, controlHz: 8,
    discretePose: true, emitsDone: true,
  },
  'MemoryChain-cont-v0': {
    envId: 'MemoryChain-cont-v0', slug: 'memory-chain', name: 'MemoryChain',
    category: 'grid', action: { kind: 'discrete', n: 2 }, controlHz: 8,
    discretePose: true, emitsDone: true,
  },
  'ProceduralMaze-cont-v0': {
    envId: 'ProceduralMaze-cont-v0', slug: 'procedural-maze', name: 'ProceduralMaze',
    category: 'grid', action: { kind: 'discrete', n: 4 }, controlHz: 8,
    discretePose: true, emitsDone: true,
  },
  'HighDimMaze-cont-v0': {
    envId: 'HighDimMaze-cont-v0', slug: 'high-dim-maze', name: 'HighDimMaze',
    category: 'grid', action: { kind: 'discrete', n: 6 }, controlHz: 8,
    discretePose: true, emitsDone: true,
  },
  'POMaze-cont-v0': {
    envId: 'POMaze-cont-v0', slug: 'po-maze', name: 'POMaze',
    category: 'grid', action: { kind: 'discrete', n: 4 }, controlHz: 8,
    discretePose: true, emitsDone: true,
  },
  'DelayedChain-cont-v0': {
    envId: 'DelayedChain-cont-v0', slug: 'delayed-chain', name: 'DelayedChain',
    category: 'grid', action: { kind: 'discrete', n: 2 }, controlHz: 8,
    discretePose: true, emitsDone: true,
  },
};

export function isEnvId(s: string): s is EnvId {
  return (ENV_IDS as readonly string[]).includes(s);
}

export function envBySlug(slug: string): EnvInfo | undefined {
  return Object.values(ENVS).find((e) => e.slug === slug);
}
