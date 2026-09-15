export interface EnvScheduleEntry {
  envId: string;
  seedStart: number;
  seedEnd: number;
}

export type Policy = 'oak' | 'ppo' | 'random';

export type BaselinePolicy = 'ppo' | 'random';

export interface RunMeta {
  seeds: number[];
  steps: number;
  batch: number;
  policy: Policy;
  preset: string;
  startedAt: string;
  envSchedule: EnvScheduleEntry[];
  envCatalog: unknown[];
}

export const DEATH_DETECTORS: readonly string[] = [
  'alive', 'envReward', 'policyMean', 'envObs', 'paramNonFinite',
  'paramExplosion', 'kernelRho', 'envError',
];

export interface RunIndexEntry {
  id: string;
  envIds: string[];
  steps: number;
  startedAt: string;
  complete: boolean;
  inFlight: boolean;
  policy: Policy;
  preset: string;
  role: 'best' | null;
}

export interface RunDetail {
  id: string;
  meta: RunMeta;
  baselines: Partial<Record<BaselinePolicy, string>>;
}
