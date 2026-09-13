import type { EnvId } from './envs';

export interface EnvScheduleEntry {
  envId: string;
  seedStart: number;
  seedEnd: number;
}

export type Policy = 'oak' | 'ppo' | 'random';

export type BaselinePolicy = 'ppo' | 'random';

export interface AgentShape {
  percepts: number;
  features: number;
  activeFeaturesPerStep: number;
  nursery: number;
  options: number;
  optionsPerThread: number;
  traceLevels: number;
  featureTimeScale: number;
  actorStepDenominator: number;
}

export interface RunGeometry {
  envIds: string[];
  shape: AgentShape;
}

export interface RunMeta {
  batch: number;
  seeds: number[];
  steps: number;
  policy: Policy;
  preset: string;
  startedAt?: string;
  envSchedule: EnvScheduleEntry[];
  envCatalog: unknown[];
  geometry?: RunGeometry[];
}

export const DEATH_DETECTORS: readonly string[] = [
  'alive', 'envReward', 'policyMean', 'envObs', 'paramNonFinite',
  'paramExplosion', 'kernelRho', 'envError',
];

export type Ci95 = [number, number];

export interface RateSummary {
  iqm: number;
  ci95: Ci95;
}

export interface KmCurve {
  steps: number[];
  survival: number[];
  ci95: [(number | null)[], (number | null)[]];
}

export interface DivergenceSummary {
  diverged: number;
  rate: number;
  rateCi95: Ci95;
  expectedAliveSteps: number;
  expectedAliveCi95: Ci95;
  km: KmCurve;
}

export interface EnvCurves {
  iqm: (number | null)[];
}

export interface EnvSummary {
  finalRate: RateSummary;
  forgetting: RateSummary;
  divergence: DivergenceSummary;
  curves: EnvCurves;
}

export interface AggregateSummary {
  divergence: DivergenceSummary;
}

export interface ProbImprovement {
  mean: number;
  ci95: Ci95;
  envs: Record<string, { p: number; ci95: Ci95 }>;
}

export interface BaselineComparison {
  probImprovement: ProbImprovement;
}

export interface RunSummary {
  seedsPerEnv: number;
  curveSteps: number[];
  curveSeconds: number[];
  envs: Record<string, EnvSummary>;
  aggregate: AggregateSummary;
  vsBaselines?: Partial<Record<BaselinePolicy, BaselineComparison>>;
}

export interface RunIndexEntry {
  id: string;
  envIds: string[];
  steps: number;
  startedAt: string | null;
  complete: boolean;
  maxStep: number;
  inFlight: boolean;
  hasSummary: boolean;
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
  summary: RunSummary | null;
  recordings: RecordingRef[];
  baselines: Partial<Record<BaselinePolicy, string>>;
}
