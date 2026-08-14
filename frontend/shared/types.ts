
import type { EnvId } from './envs';


export interface EnvScheduleEntry {
  envId: string;
  seedStart: number;
  seedEnd: number;
  flopsPerLaneStep: number;
}

export interface RunMeta {
  fleetSchemaVersion: number;
  command?: string;
  envId: string;
  batch: number;
  seeds: number[];
  steps: number;
  policies: string[];
  ablation?: string[];
  ablationLabel?: string;
  flopsPerLaneStep?: number;
  organFlops?: Record<string, number>;
  organCostVersion?: number;
  startedAt?: string;
  envSchedule?: EnvScheduleEntry[];
}


export interface EnvInfoChannel {
  count: number;
  finiteCount: number;
  nonFiniteCount: number;
  sum: number | null;
  mean: number | null;
  last: number | null;
  delta: number | null;
}

export interface FleetHealth {
  rewardRate: number | null;
  actionDeltaL2Mean?: number;
  actionDeltaPeak?: number;
  continuingRestartCount?: number;
  envInfo?: Record<string, EnvInfoChannel>;
  nonFiniteByPhase?: { encode: number; learn: number; plan: number; tdPsi: number };
  actorSupportEntropy?: number | null;
  terminationRate?: number | null;
  optionDurationMean?: number | null;
  keyboardL1?: number | null;
  mainTd?: number | null;
  planTd?: number | null;
  [key: string]: unknown;
}

export type DeathDetector =
  | 'envReward' | 'policyMean' | 'envObs' | 'paramNonFinite'
  | 'paramExplosion' | 'kernelRho' | 'envError';

export interface FleetRow {
  v: number;
  kind: 'hb' | 'final';
  envId: string;
  policy: string;
  ablationLabel?: string;
  lane: number;
  seed: number;
  step: number;
  wallSeconds: number;
  stepsPerSecond?: number;
  rewardSum: number;
  doneCount: number;
  kernelRho?: number | null;
  paramAbsMax?: number | null;
  firstNonFiniteStep?: number | null;
  firstNonFiniteDetector?: DeathDetector | null;
  health?: FleetHealth;
}


export interface AggregateRow {
  step: number;
  kind: 'hb' | 'final';
  policy: string;
  alive: number;
  dead: number;
  deliveredRateP10: number;
  deliveredRateP50: number;
  deliveredRateP90: number;
  recentRateP10: number;
  recentRateP50: number;
  recentRateP90: number;
  recentRateMax: number;
  bestLaneSeed: number;
  bestLaneDeliveredRate: number;
  bestLaneRewardSum: number;
  positiveRateLaneCount: number;
  positiveRateFraction: number;
  aggStepsPerSecond: number;
  wallSeconds: number;
}


export interface DeathRow {
  seed: number;
  lane: number;
  step: number;
  detector: DeathDetector;
  policy: string;
  rewardSumAtDeath: number;
  lastGoodRewardRate: number | null;
  nfByPhase?: { encode: number; learn: number; plan: number; tdPsi: number };
  actorSupportEntropy?: number | null;
  terminationRate?: number | null;
  optionDurationMean?: number | null;
  rhoAtDeath?: number | null;
  paramAbsMax?: number | null;
  explosionStep?: number | null;
  explosionElem?: number | null;
}


export interface SummaryDivergence {
  diverged?: number;
  rate?: number;
  wilsonLo?: number;
  wilsonHi?: number;
  byDetector?: Record<string, number>;
}

export interface SummaryPeak {
  peakStepQ?: string;
  peakRewardRateQ?: string;
  retentionQ?: string;
}

export interface SummaryPolicyResult {
  policy: string;
  seeds?: number;
  horizon?: number;
  rewardRateIQM?: number;
  rewardRateIQMCI95?: number[];
  rewardRateIQMNonDiverged?: number | null;
  recentRewardRateIQM?: number;
  recentRewardRateMax?: number;
  bestSeed?: number;
  bestSeedRewardRate?: number;
  bestSeedTotalReward?: number;
  bestSeedRewardPerGFLOP?: number;
  positiveSeeds?: number;
  positiveFraction?: number;
  divergence?: SummaryDivergence;
  rmst?: number;
  survival?: Record<string, number>;
  rewardRatePercentiles?: Record<string, number>;
  thresholdFractions?: Record<string, number>;
  peak?: SummaryPeak;
  solvedSeeds?: number;
  solvedSeedsRetained?: number;
  solvedRetentionFraction?: number | null;
  retentionFraction?: number | null;
  rewardRateBySeed?: Record<string, number>;
  recentRewardRateBySeed?: Record<string, number>;
  retentionBySeed?: Record<string, number>;
  rewardPerGFLOP?: number;
  aggStepsPerSecond?: number;
  seedStepsTotal?: number;
  entropyCollapse?: { count?: number; rate?: number; wilsonLo?: number; wilsonHi?: number };
  complete?: boolean;
  maxStep?: number;
  [key: string]: unknown;
}

export interface RunSummary {
  summarySchemaVersion?: number;
  solvedThreshold?: number;
  meta?: Partial<RunMeta>;
  results?: SummaryPolicyResult[];
  envSlices?: Record<string, SummaryPolicyResult[]>;
  [key: string]: unknown;
}


export interface RecordingMeta {
  envId: string;
  seed: number;
  actDim: number;
  codec?: unknown;
  codecVersion?: number;
  chunks?: [number, number, number, number][];
  steps?: number;
}


export interface RunIndexEntry {
  id: string;
  envIds: string[];
  batch: number;
  steps: number;
  policies: string[];
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
}

export interface RunDetail {
  id: string;
  meta: RunMeta;
  summary: RunSummary | null;
  summaryError: 'unreadable' | null;
  aggregate: AggregateRow[];
  deathsCount: number;
  recordings: RecordingRef[];
}

export interface FleetColumnar {
  steps: number[];
  lanes: { lane: number; seed: number; policy: string; envId: string }[];
  series: Record<string, (number | null)[][]>;
}


export interface WallLaneHeader {
  seed: number;
  steps: number;
  samples: number;
  firstNonFiniteRow: number | null;
  scannedRows?: number;
  truncated?: true;
  error?: string;
}

export interface WallChunkEvents {
  seed: number;
  stepStart: number[];
  stepCount: number[];
  doneCount: number[];
  rewardDelta: (number | null)[];
}

export interface WallResponse {
  v: 1;
  runId: string;
  envId: string;
  stride: number;
  headerBytes: number;
  binaryBytes: number;
  lanes: WallLaneHeader[];
  chunkEvents?: WallChunkEvents[];
}
