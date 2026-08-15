import type { EnvId } from './envs';

export interface EnvScheduleEntry {
  envId: string;
  seedStart: number;
  seedEnd: number;
}

export interface RunMeta {
  batch: number;
  seeds: number[];
  steps: number;
  ablationLabel: string;
  startedAt?: string;
  envSchedule: EnvScheduleEntry[];
  envCatalog: unknown[];
}

export const TAPE_MAGIC = 0x544b414f;
export const TAPE_FIELDS = ['rewardSum', 'rewardRate', 'deathStep', 'deathDetector'] as const;
export const DEATH_DETECTORS: readonly string[] = [
  'alive', 'envReward', 'policyMean', 'envObs', 'paramNonFinite',
  'paramExplosion', 'kernelRho', 'envError',
];

export interface EnvResult {
  seeds?: number;
  horizon?: number;
  maxStep?: number;
  complete?: boolean;
  rewardRateIQM?: number;
  positiveFraction?: number;
  divergence?: { diverged?: number };
  solvedSeeds?: number;
  retentionFraction?: number | null;
}

export interface RunSummary {
  envSlices: Record<string, EnvResult>;
}

export interface RunIndexEntry {
  id: string;
  envIds: string[];
  batch: number;
  steps: number;
  ablationLabel: string;
  startedAt: string | null;
  complete: boolean;
  maxStep: number;
  inFlight: boolean;
  hasSummary: boolean;
  hasRecordings: boolean;
  role: 'keeper' | null;
}

export interface RecordingRef {
  envId: EnvId;
  seed: number;
  bytes: number;
}

export interface RunDetail {
  id: string;
  meta: RunMeta;
  summary: RunSummary | null;
  recordings: RecordingRef[];
}

