
import type { RunSummary, SummaryPolicyResult } from '@shared/types';

export const SUMMARY_SCHEMA_VERSION = 2;

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

const PEAK_KEYS: readonly (readonly [string, string])[] = [
  ['zenithStepQ', 'peakStepQ'],
  ['zenithRateQ', 'peakRewardRateQ'],
  ['decayRatioQ', 'retentionQ'],
];

const DIVERGENCE_KEYS: readonly (readonly [string, string])[] = [['dead', 'diverged']];

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

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
