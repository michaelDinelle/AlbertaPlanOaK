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

// ---------- summary.json v2 (fleet.py report output, normalized by server/summaryAdapter) ----------
//
// Vocabulary law (summarySchemaVersion 2): a fleet LANE is a SEED — one agent, one
// continuing stream — so every population figure counts seeds. "Delivered rate" is
// the AVERAGE REWARD RATE (reward per step); "zenith/decay" is PEAK/RETENTION; an
// "igniter" is a seed that crossed `solvedThreshold`. v1 roots on disk keep the old
// key names; `adaptSummary` maps them to these before anything else sees them, and
// fields v1 never wrote stay ABSENT (nothing is synthesized).

export interface SummaryDivergence {
  /** seeds whose stream went non-finite */
  diverged?: number;
  rate?: number;
  wilsonLo?: number;
  wilsonHi?: number;
  byDetector?: Record<string, number>;
}

/** quartile strings ("q25 / q50 / q75") over the per-seed peak & retention pass */
export interface SummaryPeak {
  peakStepQ?: string;
  peakRewardRateQ?: string;
  retentionQ?: string;
}

export interface SummaryPolicyResult {
  policy: string;
  /** population size — seeds, not "lanes" */
  seeds?: number;
  horizon?: number;
  rewardRateIQM?: number;
  /** [lo, hi] bootstrap 95% CI of rewardRateIQM */
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
  /** seeds whose peak windowed reward rate crossed summary.solvedThreshold */
  solvedSeeds?: number;
  solvedSeedsRetained?: number;
  solvedRetentionFraction?: number | null;
  /** fraction of seeds still holding ≥ ½ their peak reward rate */
  retentionFraction?: number | null;
  /** per-seed vectors — v2 persists them; absent on v1 roots */
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
  /** 2 once fleet.py wrote it; absent on v1 roots (the adapter fills it in) */
  summarySchemaVersion?: number;
  /** peak windowed reward rate above which a seed counts as solved */
  solvedThreshold?: number;
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
  recordVersion: number; // 3 and 4 are servable
  /** v4 only: codec descriptor per src/sim/RECORD_V4_SPEC.md (verbatim static-table entry) */
  codec?: unknown;
  /** v4 only: codecVersion from the spec */
  codecVersion?: number;
  /** v4 only, finalize-time hint: [stepStart, stepCount, byteOffset, byteLength][] */
  chunks?: [number, number, number, number][];
  /** v4 only, finalize-time: total decoded steps */
  steps?: number;
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
  role: 'bestKernel' | 'candidate' | 'parent' | null;
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
  /**
   * WHY `summary` is null, so the UI never asserts something false about the file.
   * `null` here = the run genuinely has no summary.json (fleet.py report was never
   * run) — the normal state. `'unreadable'` = the file EXISTS but did not parse
   * (invalid JSON: a truncated/half-flushed write, a disk error, or a writer that
   * emitted bare NaN). Those need opposite advice: the first says "run the tool",
   * the second must say the file was REJECTED — telling the user to re-run the tool
   * that produced it is a doom loop.
   */
  summaryError: 'unreadable' | null;
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

/**
 * Learning curve over one run's heartbeats, derived from fleet.jsonl alone.
 *
 * Every series is a population statistic of the MEASURED windowed reward rate —
 * (rewardSum_i − rewardSum_{i−1}) / (step_i − step_{i−1}) per seed — never the
 * agent's own ρ̄ estimate. `iqm` is the interquartile mean (tools/fleet.py's
 * headline statistic); `ciLo`/`ciHi` are its percentile-bootstrap 95% CI.
 * A heartbeat no seed could be measured at is `null`, never 0.
 */
export interface CurvesResponse {
  steps: number[];
  iqm: (number | null)[];
  ciLo: (number | null)[];
  ciHi: (number | null)[];
  p10: (number | null)[];
  p50: (number | null)[];
  p90: (number | null)[];
  /** seeds behind the curve: the lane set left after the policy + env filter */
  seeds: number;
}

/**
 * ROLES.json v2. `bestKernel` replaced `keeper`; readers accept BOTH shapes on
 * disk for one migration cycle. `featured` is reserved for a showcase run and is
 * null until something claims it.
 */
export interface RolesResponse {
  bestKernel: string | null;
  candidate: string | null;
  parent: string | null;
  featured: string | null;
}

// ---------- SSE /api/runs/:id/live event payloads ----------

export interface LiveHello { rows: AggregateRow[] }
export interface LiveStatus { maxStep: number; inFlight: boolean; complete: boolean }

// ---------- GET /api/runs/:id/wall/:envId — every recorded seed of one env ----------
//
// THE FRAMING, exactly (one response body, two sections):
//
//   <UTF-8 JSON `WallResponse`>  0x0A  <deflate-raw binary section>
//
// The FIRST 0x0A byte in the body is the separator and the only one that can appear
// before the binary: `JSON.stringify` escapes every control character inside strings,
// so the header text can never contain a raw newline. `headerBytes` is the byte length
// of that JSON text WITHOUT the newline (so the binary starts at `headerBytes + 1`),
// and `binaryBytes` is the length of the COMPRESSED section — the two together
// reconstruct the split without scanning, and disagree with the scan only if the body
// was corrupted in transit.
//
// The binary section inflates (`DecompressionStream('deflate-raw')`) to the lanes'
// samples concatenated IN `lanes` ORDER, each lane contributing `samples` records of
// 37 TIGHTLY PACKED bytes:
//
//   bytes  0..35  pose[9]  little-endian f32, verbatim record bytes (NaN preserved)
//   byte      36  done     1 iff the recorded done float is finite and non-zero
//
// 37 is not a multiple of 4, so pose floats sit at unaligned offsets: read them with a
// DataView (or de-interleave into an aligned Float32Array, which is what `fetchWall`
// does). When every lane is empty the section is omitted entirely and `binaryBytes` is 0.
//
// Sampling: record rows `0, stride, 2·stride, …` PLUS the final row (`steps − 1`) when
// the stride missed it. So `samples` is never 0 for a non-empty lane, and the last
// sample is always the last row the wall COVERS.
//
// `stride` is the stride the response was BUILT at, which is the requested one unless
// that would have exceeded the server's payload or decode budget (a fully-recorded
// Tier-L is 1024 seeds × 1M steps). The request is a floor; read the stride back from
// here rather than assuming it.

export interface WallLaneHeader {
  seed: number;
  /**
   * Rows this lane's samples COVER: `lastSampledRow + 1`. A torn v4 tail, a short v3
   * file, or a `truncated` budget stop all yield fewer than the lane meta claims.
   */
  steps: number;
  /** records this lane contributes to the binary section */
  samples: number;
  /**
   * Index of the first RECORD ROW — not sample — whose projected payload (any of
   * pose[9], or done) is non-finite; null when no such row was SCANNED. This is the
   * wall's own death certificate: the row at which the tile has nothing true left to
   * draw. It is exact over the rows the server decoded — see `scannedRows`.
   */
  firstNonFiniteRow: number | null;
  /**
   * Rows the server actually decoded, and therefore scanned for non-finite values.
   * Present ONLY when it is smaller than `steps`: a v4 lane whose chunks are shorter
   * than the stride is decoded chunk-by-chunk, and a chunk holding no sampled row is
   * never inflated. `firstNonFiniteRow` is then the first non-finite row among the
   * SCANNED ones — the death is dated to within one chunk, not to the row.
   */
  scannedRows?: number;
  /**
   * The wall stopped this lane early to stay inside its own budget (a lane that grew
   * past what run discovery reported, or a run so large that decoding every lane in
   * full would block the server). `steps` is then what the wall COVERS, not what the
   * seed recorded.
   */
  truncated?: true;
  /** present ONLY when the lane could not be decoded; it then contributes 0 samples */
  error?: string;
}

/**
 * Per-chunk event hints for one v4 lane. Chunks are the recorder's drain intervals —
 * contiguous and step-ordered — so these are a free, already-computed coarse timeline
 * of where a seed finished episodes and what it was paid, at chunk resolution.
 *
 * They are read from the chunk HEADERS, so they cover every chunk the wall covers —
 * including the ones a coarse stride let it skip decoding (`scannedRows`). The rows
 * they describe end at `steps`.
 */
export interface WallChunkEvents {
  seed: number;
  stepStart: number[];
  stepCount: number[];
  /** done bits set inside the chunk */
  doneCount: number[];
  /** lane rewardSum delta over the chunk; null where the host had no snapshot (NaN on disk) */
  rewardDelta: (number | null)[];
}

export interface WallResponse {
  /** framing + projection version; bumping it invalidates every cached wall */
  v: 1;
  runId: string;
  envId: string;
  /** rows per sample ACTUALLY used — never finer than requested, sometimes coarser */
  stride: number;
  /** byte length of this JSON text, newline excluded */
  headerBytes: number;
  /** byte length of the deflate-raw section that follows the newline */
  binaryBytes: number;
  /** recorded seeds of `envId`, ascending; empty for a run with no recordings */
  lanes: WallLaneHeader[];
  /** v4 lanes only, in `lanes` order; absent when no lane carries chunk framing */
  chunkEvents?: WallChunkEvents[];
}
