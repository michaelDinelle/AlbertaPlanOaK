// summary.json v1 -> v2 normalization.
//
// `tools/fleet.py report` writes summarySchemaVersion 2 today, but ~600 run roots on
// disk carry the v1 shape and are NOT going to be regenerated (several are hours of
// GPU time whose fleet.jsonl still exists but whose value is historical). Rather than
// teach every view two vocabularies, the whole app speaks v2 and this module is the
// single place the old names die.
//
// Two rules make it safe to trust the output:
//
//   1. PASSTHROUGH at v2. A summary that already declares summarySchemaVersion >= 2
//      is returned as-is (same object identity), so a regenerated root is never
//      reinterpreted by a mapping table.
//   2. SYNTHESIZE NOTHING. v1 never wrote the per-seed vectors, so adapt() does not
//      invent them — an absent field stays absent and the UI renders "—" rather than
//      a number no run produced. The only field adapt() adds is the schema version
//      itself (marking "this has been normalized"), never a measurement.
//
// Pure: no fs, no fetch, no clock. The golden test in test/server/api.test.ts feeds it
// the v1 summary of a fixture root and compares against the v2 summary fleet.py writes
// for that same root.

import type { RunSummary, SummaryPolicyResult } from '@shared/types';

export const SUMMARY_SCHEMA_VERSION = 2;

/** The on-disk v1 result shape. Private — nothing outside this module speaks it. */
interface LegacySummaryResult {
  policy?: unknown;
  lanes?: unknown;
  iqmDeliveredAll?: unknown;
  iqmDeliveredAllCI95?: unknown;
  iqmDeliveredSurvivors?: unknown;
  iqmRecent?: unknown;
  recentMax?: unknown;
  bestLaneSeed?: unknown;
  bestLaneRate?: unknown;
  bestLaneRewardSum?: unknown;
  bestRewardPerGFLOP?: unknown;
  positiveLanes?: unknown;
  rateDistribution?: unknown;
  tierFractions?: unknown;
  plasticity?: unknown;
  igniters?: unknown;
  ignitersSustained?: unknown;
  igniterSustainFraction?: unknown;
  sustainingFraction?: unknown;
  laneStepsTotal?: unknown;
  divergence?: unknown;
  [key: string]: unknown;
}

interface LegacySummary {
  summarySchemaVersion?: unknown;
  meta?: unknown;
  results?: unknown;
  envSlices?: unknown;
  [key: string]: unknown;
}

/** v1 key -> v2 key, one entry per rename. Everything not listed keeps its name. */
const RESULT_KEYS: readonly (readonly [string, string])[] = [
  ['lanes', 'seeds'],
  ['iqmDeliveredAll', 'rewardRateIQM'],
  ['iqmDeliveredAllCI95', 'rewardRateIQMCI95'],
  ['iqmDeliveredSurvivors', 'rewardRateIQMNonDiverged'],
  ['iqmRecent', 'recentRewardRateIQM'],
  ['recentMax', 'recentRewardRateMax'],
  ['bestLaneSeed', 'bestSeed'],
  ['bestLaneRate', 'bestSeedRewardRate'],
  ['bestLaneRewardSum', 'bestSeedTotalReward'],
  ['bestRewardPerGFLOP', 'bestSeedRewardPerGFLOP'],
  ['positiveLanes', 'positiveSeeds'],
  ['rateDistribution', 'rewardRatePercentiles'],
  ['tierFractions', 'thresholdFractions'],
  ['igniters', 'solvedSeeds'],
  ['ignitersSustained', 'solvedSeedsRetained'],
  ['igniterSustainFraction', 'solvedRetentionFraction'],
  ['sustainingFraction', 'retentionFraction'],
  ['laneStepsTotal', 'seedStepsTotal'],
  ['deliveredBySeed', 'rewardRateBySeed'],
  ['recentRates', 'recentRewardRateBySeed'],
];

/** v1 `plasticity` sub-keys -> v2 `peak` sub-keys. */
const PEAK_KEYS: readonly (readonly [string, string])[] = [
  ['zenithStepQ', 'peakStepQ'],
  ['zenithRateQ', 'peakRewardRateQ'],
  ['decayRatioQ', 'retentionQ'],
];

/** v1 `divergence.dead` -> v2 `divergence.diverged`; the rest of the block is unchanged. */
const DIVERGENCE_KEYS: readonly (readonly [string, string])[] = [['dead', 'diverged']];

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Copy `source` renaming the listed keys. Keys absent from the source are absent from
 * the result (no undefined placeholders — `'x' in out` stays a truthful question), and
 * a v2 name already present in the source wins over the v1 name it would be renamed
 * from, so a half-migrated file cannot lose its newer value.
 */
function rename(
  source: Record<string, unknown>,
  pairs: readonly (readonly [string, string])[],
): Record<string, unknown> {
  const map = new Map(pairs);
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(source)) {
    const renamed = map.get(key);
    if (renamed === undefined) {
      out[key] = source[key];
      continue;
    }
    // `key` is a v1 name. Only write the v2 slot if the source did not already carry it.
    if (!Object.prototype.hasOwnProperty.call(source, renamed)) out[renamed] = source[key];
  }
  return out;
}

function adaptResult(raw: unknown): SummaryPolicyResult {
  if (!isRecord(raw)) return { policy: 'oak' };
  const legacy = raw as LegacySummaryResult;
  const out = rename(legacy, RESULT_KEYS);

  const plasticity = legacy['plasticity'];
  if (isRecord(plasticity)) {
    delete out['plasticity'];
    if (!isRecord(out['peak'])) out['peak'] = rename(plasticity, PEAK_KEYS);
  }

  const divergence = legacy['divergence'];
  if (isRecord(divergence)) out['divergence'] = rename(divergence, DIVERGENCE_KEYS);

  if (typeof out['policy'] !== 'string') out['policy'] = 'oak';
  return out as SummaryPolicyResult;
}

function adaptSlices(raw: unknown): Record<string, SummaryPolicyResult[]> | undefined {
  if (!isRecord(raw)) return undefined;
  const out: Record<string, SummaryPolicyResult[]> = {};
  for (const [envId, list] of Object.entries(raw)) {
    out[envId] = Array.isArray(list) ? list.map(adaptResult) : [];
  }
  return out;
}

/**
 * Normalize a parsed summary.json to the v2 DTO.
 *
 * `null` in -> `null` out (a run with no summary.json is a normal state, not an error).
 * A v2 summary is returned unchanged. A v1 summary is renamed key-for-key and stamped
 * with the schema version; no measurement is invented.
 */
export function adaptSummary(raw: RunSummary | null): RunSummary | null {
  if (raw === null || !isRecord(raw)) return null;
  const legacy = raw as LegacySummary;
  const version = legacy.summarySchemaVersion;
  if (typeof version === 'number' && version >= SUMMARY_SCHEMA_VERSION) return raw;

  const out: Record<string, unknown> = { ...legacy, summarySchemaVersion: SUMMARY_SCHEMA_VERSION };
  if (Array.isArray(legacy.results)) out['results'] = legacy.results.map(adaptResult);
  const slices = adaptSlices(legacy.envSlices);
  if (slices !== undefined) out['envSlices'] = slices;
  return out as RunSummary;
}
