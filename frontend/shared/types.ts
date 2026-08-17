import type { EnvId } from './envs';

export interface EnvScheduleEntry {
  envId: string;
  seedStart: number;
  seedEnd: number;
}

export type Policy = 'oak' | 'ppo' | 'random';

export type BaselinePolicy = 'ppo' | 'random';

export interface RunMeta {
  batch: number;
  seeds: number[];
  steps: number;
  ablationLabel: string;
  policy: Policy;
  startedAt?: string;
  finishedAt?: string;
  envSchedule: EnvScheduleEntry[];
  envCatalog: unknown[];
  flopsPerStep: number;
  envFlopsPerStep: Record<string, number>;
  gpuSeconds?: number;
}

export const TAPE_MAGIC = 0x544b414f;
export const DEATH_DETECTORS: readonly string[] = [
  'alive', 'envReward', 'policyMean', 'envObs', 'paramNonFinite',
  'paramExplosion', 'kernelRho', 'envError',
];

export type Ci95 = [number, number];

export interface RateSummary {
  iqm: number;
  ci95: Ci95;
  iqmZ: number;
  ci95Z: Ci95;
}

export interface ZSummary {
  iqm: number;
  ci95: Ci95;
}

export interface ForgettingSummary {
  phiIQM: number;
  ci95: Ci95;
}

export interface DivergenceSummary {
  diverged: number;
  rate: number;
  expectedAliveSteps: number;
}

export interface PercentileCurves {
  percentileSeeds: { p5: number; p50: number; p95: number };
  rateZ: { p5: (number | null)[]; p50: (number | null)[]; p95: (number | null)[] };
}

export interface EnvPerLane {
  seeds: number[];
  finalZ: number[];
  lifetimeZ: number[];
  phi: number[];
  deathStep: (number | null)[];
}

export interface PlasticityTrack {
  final: number | null;
  iqm: (number | null)[];
}

export interface EnvSummary {
  floor: number;
  ceiling: number;
  ceilingSource: string;
  finalRate: RateSummary;
  lifetimeRate: RateSummary;
  forgetting: ForgettingSummary;
  divergence: DivergenceSummary;
  iprZ90: number;
  curves: PercentileCurves;
  perLane: EnvPerLane;
  plasticity?: Record<string, PlasticityTrack>;
}

export interface ProfileSummary {
  tau: number[];
  fraction: number[];
  ci95: [number[], number[]];
}

export interface FrontierSummary {
  cumFlopsPerLane: number[];
  zIQM: (number | null)[];
}

export interface AggregateSummary {
  finalZ: ZSummary;
  lifetimeZ: ZSummary;
  optimalityGap: { mean: number; ci95: Ci95 };
  iprZ90: number;
  forgetting: ForgettingSummary;
  divergence: DivergenceSummary;
  profile: ProfileSummary;
  frontier: FrontierSummary;
  computeToThreshold: Record<string, number | null>;
}

export interface ComputeSummary {
  flopsPerStep: number;
  envFlopsPerStep: Record<string, number>;
  gpuSeconds?: number;
  omega: number;
}

export interface ProbImprovement {
  mean: number;
  envs: Record<string, { p: number; ci95: Ci95 }>;
}

export interface BaselineComparison {
  run: string;
  probImprovement: ProbImprovement;
  computeRatioAt: Record<string, number | null>;
}

export interface BootstrapSpec {
  baseSeed: number;
  reps: number;
  curveReps: number;
  method: string;
}

export interface RunSummary {
  complete: boolean;
  generatedAt: string;
  scale: string;
  horizon: number;
  maxStep: number;
  seedsPerEnv: number;
  curveSteps: number[];
  plasticitySteps: number[];
  envs: Record<string, EnvSummary>;
  aggregate: AggregateSummary;
  compute: ComputeSummary;
  bootstrap: BootstrapSpec;
  vsBaselines?: Partial<Record<BaselinePolicy, BaselineComparison>>;
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
  policy: Policy;
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
  baselines: Partial<Record<BaselinePolicy, string>>;
}
