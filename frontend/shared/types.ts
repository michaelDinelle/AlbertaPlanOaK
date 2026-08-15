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
export type TapeField = (typeof TAPE_FIELDS)[number];

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
  recentRewardRateIQM?: number;
  recentRewardRateMax?: number;
  bestSeed?: number;
  bestSeedRewardRate?: number;
  bestSeedRewardPerGFLOP?: number;
  positiveSeeds?: number;
  positiveFraction?: number;
  rewardRatePercentiles?: Record<string, number>;
  thresholdFractions?: Record<string, number>;
  divergence?: { diverged?: number; rate?: number };
  rmst?: number;
  solvedSeeds?: number;
  solvedSeedsRetained?: number;
  decayedSeeds?: number;
  decayFraction?: number | null;
  retentionMarginP50?: number | null;
  retentionFraction?: number | null;
  rewardRateBySeed?: Record<string, number>;
  retentionBySeed?: Record<string, number | null>;
  aggStepsPerSecond?: number;
  [key: string]: unknown;
}

export interface RunSummary {
  solvedLaw?: string;
  retainedLaw?: string;
  meta?: Partial<RunMeta>;
  overall: EnvResult;
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
  steps: number;
  bytes: number;
}

export interface RunDetail {
  id: string;
  meta: RunMeta;
  summary: RunSummary | null;
  recordings: RecordingRef[];
}

