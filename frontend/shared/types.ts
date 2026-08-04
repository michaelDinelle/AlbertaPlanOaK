// FROZEN Phase-0 contract: on-disk fleet schemas + API DTOs shared by server and client.
// Additive changes only after Phase 0. On-disk shapes mirror Sources/Main.swift writers
// (fleetSchemaVersion 1) — do not "fix" field names here; fix readers instead.

import type { EnvId } from './envs';

// ---------- on-disk: .oak-benchmarks/<run>/meta.json ----------

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
  /** present only for mixed-env runs; contiguous, covers all lanes */
  envSchedule?: EnvScheduleEntry[];
}

// ---------- on-disk: fleet.jsonl rows ----------

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
  /** channel names are env-conditional per the lane's own env */
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

// ---------- on-disk: aggregate.jsonl rows ----------

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

// ---------- on-disk: deaths.jsonl rows ----------

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

// ---------- on-disk: summary.json (fleet.py report output; partial typing) ----------

export interface SummaryPolicyResult {
  policy: string;
  lanes?: number;
  iqmDeliveredAll?: number;
  iqmDeliveredSurvivors?: number;
  divergenceRate?: number;
  positiveFraction?: number;
  sustainingFraction?: number;
  survival?: Record<string, number>;
  rateDistribution?: Record<string, number>;
  tierFractions?: Record<string, number>;
  rewardPerGFLOP?: number;
  complete?: boolean;
  [key: string]: unknown;
}

export interface RunSummary {
  meta?: Partial<RunMeta>;
  results?: SummaryPolicyResult[];
  envSlices?: Record<string, SummaryPolicyResult[]>;
  [key: string]: unknown;
}

// ---------- on-disk: per-lane recording meta.json ----------

export interface RecordingMeta {
  envId: string;
  seed: number;
  actDim: number;
  recordVersion: number; // only 3 is servable
}

// ---------- API DTOs ----------

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
  role: 'candidate' | 'parent' | 'keeper' | null;
}

export interface RecordingRef {
  envId: EnvId;
  seed: number;
  steps: number; // byteLength / 64
}

export interface RunDetail {
  id: string;
  meta: RunMeta;
  summary: RunSummary | null;
  aggregate: AggregateRow[];
  deathsCount: number;
  recordings: RecordingRef[];
}

/** columnar per-lane series derived from fleet.jsonl (server-side parse only) */
export interface FleetColumnar {
  steps: number[];
  lanes: { lane: number; seed: number; policy: string; envId: string }[];
  /** series[field][laneIndex][heartbeatIndex]; null = missing/non-finite */
  series: Record<string, (number | null)[][]>;
}

export interface RolesResponse {
  candidate: string | null;
  parent: string | null;
  keeper: string | null;
}

// ---------- SSE /api/runs/:id/live event payloads ----------

export interface LiveHello { rows: AggregateRow[] }
export interface LiveStatus { maxStep: number; inFlight: boolean; complete: boolean }
