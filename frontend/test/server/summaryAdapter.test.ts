// summary.json v1 -> v2 adapter: the golden test.
//
// The adapter's contract is not "these names map to those names" — it is "a v1 root
// and a freshly-reported v2 root describe the SAME run identically". So the fixture is
// one mini fleet root (test/fixtures/server/summary-adapter/run, built deterministically
// by make_fixture.py) reported TWICE: once by the pre-rename fleet.py (summary-v1.json)
// and once by the current one (summary-v2.json). If a future rename lands in fleet.py
// without landing here, these deep-equals fail.

import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { RunSummary } from '../../shared/types';
import { SUMMARY_SCHEMA_VERSION, adaptSummary } from '../../server/summaryAdapter';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, '..', 'fixtures', 'server', 'summary-adapter');

function fixture(name: string): RunSummary {
  return JSON.parse(readFileSync(join(FIXTURES, name), 'utf8')) as RunSummary;
}

/**
 * Fields v2 ADDS rather than renames. v1 never wrote them, and the adapter's second
 * rule is "synthesize nothing", so adapt(v1) legitimately lacks them. Listing them
 * explicitly means a new v2 field cannot quietly slip past the golden comparison.
 */
const V2_ONLY_TOP = ['solvedThreshold'] as const;
const V2_ONLY_RESULT = ['rewardRateBySeed', 'recentRewardRateBySeed', 'retentionBySeed'] as const;

function stripAdditive(summary: RunSummary): RunSummary {
  const clone = JSON.parse(JSON.stringify(summary)) as Record<string, unknown>;
  for (const key of V2_ONLY_TOP) delete clone[key];
  const scrub = (list: unknown): void => {
    if (!Array.isArray(list)) return;
    for (const entry of list) {
      if (entry === null || typeof entry !== 'object') continue;
      for (const key of V2_ONLY_RESULT) delete (entry as Record<string, unknown>)[key];
    }
  };
  scrub(clone['results']);
  const slices = clone['envSlices'];
  if (slices !== null && typeof slices === 'object') {
    for (const list of Object.values(slices as Record<string, unknown>)) scrub(list);
  }
  return clone as RunSummary;
}

describe('summaryAdapter — v1 root vs a freshly-reported v2 root', () => {
  const v1 = fixture('summary-v1.json');
  const v2 = fixture('summary-v2.json');

  it('the fixtures really are the two schemas of one run', () => {
    expect(v1['summarySchemaVersion']).toBeUndefined();
    expect(v2['summarySchemaVersion']).toBe(SUMMARY_SCHEMA_VERSION);
    expect(v2['solvedThreshold']).toBe(0.1);
    // same root => same meta, same policies, same env slices
    expect(v1['meta']).toEqual(v2['meta']);
    expect(Object.keys(v1['envSlices'] as object).sort()).toEqual(Object.keys(v2['envSlices'] as object).sort());
  });

  it('adapt(v1) equals the v2 report of the same root', () => {
    const adapted = adaptSummary(v1);
    expect(adapted).not.toBeNull();
    expect(stripAdditive(adapted as RunSummary)).toEqual(stripAdditive(v2));
  });

  it('every renamed field carries the v1 VALUE, not just the v2 name', () => {
    const legacy = (v1['results'] as Record<string, unknown>[])[0] as Record<string, unknown>;
    const out = ((adaptSummary(v1) as RunSummary)['results'] as Record<string, unknown>[])[0] as Record<string, unknown>;
    expect(out['seeds']).toBe(legacy['lanes']);
    expect(out['rewardRateIQM']).toBe(legacy['iqmDeliveredAll']);
    expect(out['rewardRateIQMCI95']).toEqual(legacy['iqmDeliveredAllCI95']);
    expect(out['rewardRateIQMNonDiverged']).toBe(legacy['iqmDeliveredSurvivors']);
    expect(out['recentRewardRateIQM']).toBe(legacy['iqmRecent']);
    expect(out['recentRewardRateMax']).toBe(legacy['recentMax']);
    expect(out['bestSeed']).toBe(legacy['bestLaneSeed']);
    expect(out['bestSeedRewardRate']).toBe(legacy['bestLaneRate']);
    expect(out['bestSeedTotalReward']).toBe(legacy['bestLaneRewardSum']);
    expect(out['bestSeedRewardPerGFLOP']).toBe(legacy['bestRewardPerGFLOP']);
    expect(out['positiveSeeds']).toBe(legacy['positiveLanes']);
    expect(out['rewardRatePercentiles']).toEqual(legacy['rateDistribution']);
    expect(out['thresholdFractions']).toEqual(legacy['tierFractions']);
    expect(out['solvedSeeds']).toBe(legacy['igniters']);
    expect(out['solvedSeedsRetained']).toBe(legacy['ignitersSustained']);
    expect(out['solvedRetentionFraction']).toBe(legacy['igniterSustainFraction']);
    expect(out['retentionFraction']).toBe(legacy['sustainingFraction']);
    expect(out['seedStepsTotal']).toBe(legacy['laneStepsTotal']);
    const plasticity = legacy['plasticity'] as Record<string, unknown>;
    expect(out['peak']).toEqual({
      peakStepQ: plasticity['zenithStepQ'],
      peakRewardRateQ: plasticity['zenithRateQ'],
      retentionQ: plasticity['decayRatioQ'],
    });
    expect((out['divergence'] as Record<string, unknown>)['diverged'])
      .toBe((legacy['divergence'] as Record<string, unknown>)['dead']);
  });

  it('drops no v1 field on the floor: every old key has a new home or its own name', () => {
    const legacy = (v1['results'] as Record<string, unknown>[])[0] as Record<string, unknown>;
    const out = ((adaptSummary(v1) as RunSummary)['results'] as Record<string, unknown>[])[0] as Record<string, unknown>;
    // v1 keys that keep their name must still be present and equal. `divergence`
    // keeps its name but renames a member (dead -> diverged), asserted above.
    for (const key of Object.keys(legacy)) {
      if (key === 'divergence') continue;
      if (Object.prototype.hasOwnProperty.call(out, key)) expect(out[key]).toEqual(legacy[key]);
    }
    // and the count is preserved: 21 renames + `plasticity` -> `peak` is 1-for-1
    expect(Object.keys(out).length).toBe(Object.keys(legacy).length);
  });

  it('renames inside envSlices too — a per-env slice is a full result', () => {
    const adapted = adaptSummary(v1) as RunSummary;
    for (const [envId, list] of Object.entries(adapted.envSlices ?? {})) {
      const slice = list[0] as Record<string, unknown> | undefined;
      expect(slice, envId).toBeDefined();
      expect(slice?.['rewardRateIQM']).toBeTypeOf('number');
      expect(slice?.['iqmDeliveredAll']).toBeUndefined();
      // the slice CI is the slice's OWN population, never gated on the pooled n
      expect(slice?.['rewardRateIQMCI95']).toBeInstanceOf(Array);
    }
  });
});

describe('summaryAdapter — the two safety rules', () => {
  it('passes a v2 summary through untouched, same object identity', () => {
    const v2 = fixture('summary-v2.json');
    expect(adaptSummary(v2)).toBe(v2);
  });

  it('answers null for null — a run with no summary.json is a normal state', () => {
    expect(adaptSummary(null)).toBeNull();
  });

  it('synthesizes nothing: an absent v1 field stays absent', () => {
    const out = adaptSummary({ results: [{ policy: 'oak', lanes: 4 }] }) as RunSummary;
    const result = (out.results ?? [])[0] as Record<string, unknown>;
    expect(result['seeds']).toBe(4);
    expect('rewardRateIQM' in result).toBe(false);
    expect('retentionFraction' in result).toBe(false);
    expect('peak' in result).toBe(false);
    expect('solvedThreshold' in out).toBe(false);
    // the ONE thing it adds is the marker that it ran
    expect(out.summarySchemaVersion).toBe(SUMMARY_SCHEMA_VERSION);
  });

  it('a half-migrated file keeps its newer value', () => {
    const out = adaptSummary({
      results: [{ policy: 'oak', lanes: 4, seeds: 8, iqmDeliveredAll: 1, rewardRateIQM: 2 }],
    }) as RunSummary;
    const result = (out.results ?? [])[0] as Record<string, unknown>;
    expect(result['seeds']).toBe(8);
    expect(result['rewardRateIQM']).toBe(2);
  });

  it('survives a malformed results array without throwing', () => {
    const out = adaptSummary({ results: [null, 7, { policy: 'zero' }] as never }) as RunSummary;
    expect((out.results ?? []).map((r) => r.policy)).toEqual(['oak', 'oak', 'zero']);
  });
});
