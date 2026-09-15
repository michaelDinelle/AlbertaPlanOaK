import type { EnvId } from './envs';

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
  policy: Policy;
  preset: string;
  startedAt?: string;
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
  startedAt: string | null;
  complete: boolean;
  inFlight: boolean;
  hasRecordings: boolean;
  policy: Policy;
  preset: string;
  role: 'best' | null;
}

export interface RecordingRef {
  envId: EnvId;
  seed: number;
}

export interface RunDetail {
  id: string;
  meta: RunMeta;
  recordings: RecordingRef[];
  baselines: Partial<Record<BaselinePolicy, string>>;
}
