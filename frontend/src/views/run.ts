// #/run/:id — the run dashboard.
//
// House rule this view enforces in its layout: metrics are NEVER pooled across
// environments. The overview shows the fleet-wide aggregate (which the fleet
// itself computes) and the per-env tab gives every env its own section on its
// own scale. The only cross-env chart is positivity, because "fraction of seeds
// above zero" is the one quantity that means the same thing in CartPole and in
// HalfCheetah.

import type { ViewMount } from '../router';
import type { FleetColumnar, RunDetail, SummaryPolicyResult } from '@shared/types';
import { ENVS, isEnvId } from '@shared/envs';
import { ApiError, api } from '../api';
import {
  aggregateSeries,
  capLanes,
  envName,
  envShares,
  envSliceResult,
  envSlug,
  formatAgo,
  formatCount,
  formatPercent,
  formatRate,
  formatSigned,
  formatStep,
  formatWall,
  kmCurve,
  laneSeries,
  loadRunDetail,
  policiesIn,
  policyView,
  rankLanes,
  toWindowRates,
  type LaneSeries,
  type PolicyView,
} from '../state';
import {
  badge,
  el,
  emptyState,
  errorState,
  eyebrow,
  linkButton,
  spinner,
  statTile,
  viewShell,
} from '../components/table';
import {
  VIZ,
  band,
  columns,
  learningCurve,
  spaghetti,
  steppedLine,
  timeSeries,
  type ChartHandle,
} from '../components/chart';
import { bars, parseQuartiles, rangeBars, type RangeRow } from '../components/bars';
import { accentCss, envSpectrum } from './gallery';

/**
 * The Wall link for this run — `#/?run=<id>&env=<envId>`.
 *
 * Built here rather than imported from `views/wall` on purpose: the wall module
 * pulls the scene host (and Three) into whatever chunk imports it, and this is a
 * two-parameter string. `wallHref` in `views/wall.ts` is the canonical builder
 * and `parseWallParams` is what reads it back; keep the two in step.
 */
function wallLink(runId: string, envId?: string): string {
  const run = `run=${encodeURIComponent(runId)}`;
  return envId === undefined ? `#/?${run}` : `#/?${run}&env=${encodeURIComponent(envId)}`;
}

interface Disposable {
  destroy(): void;
}

const LANE_CAP = 140;

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function tone(value: number | null): 'pos' | 'neg' | 'dim' {
  if (value === null) return 'dim';
  return value > 0 ? 'pos' : value < 0 ? 'neg' : 'dim';
}

/** the seed cap for the per-seed total-reward profile (one bar per column) */
const COLUMN_CAP = 256;

/**
 * Reverse ECDF of a per-seed ratio: for every observed value x, the fraction of seeds
 * at or above it. Stepped, it reads as "how much of the population kept how much of
 * its peak" — and its height at x = ½ is exactly the retention-rate tile, which makes
 * the two cross-check each other instead of being two unrelated numbers.
 *
 * Ties collapse onto one step (a value shared by 40 seeds is one drop, not 40).
 */
function retentionCurve(values: readonly number[]): { steps: number[]; fraction: number[] } {
  const sorted = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const n = sorted.length;
  const steps: number[] = [];
  const fraction: number[] = [];
  for (let i = 0; i < n; i++) {
    const value = sorted[i];
    if (value === undefined) continue;
    if (i > 0 && value === sorted[i - 1]) continue;
    steps.push(value);
    fraction.push((n - i) / n);
  }
  return { steps, fraction };
}

// ---------------------------------------------------------------------------
// the science module
//
// This file no longer owns a page. `#/run/:id` redirects into the wall, and the
// wall embeds `mountRunScience` below the stage: the CURRENT environment's
// science first (stats, learning curve, and a collapsible per-seed detail whose
// total-reward columns are click-to-watch), then the whole-run statistics in a
// collapsed block. Nothing was deleted from the science — it moved to where the
// environment is.
// ---------------------------------------------------------------------------

export const mount: ViewMount = (el_, params) => {
  const id = params['id'];
  window.location.replace(id ? `#/?run=${encodeURIComponent(id)}` : '#/');
  void el_;
  return { destroy() {} };
};

export interface RunScienceOptions {
  runId: string;
  /** the environment whose section leads; null = first in the schedule */
  envId: string | null;
  /** click-to-watch: a seed picked in the total-reward columns (or curve) */
  onWatchSeed?(envId: string, seed: number): void;
}

export interface RunScienceHandle {
  destroy(): void;
  /** swap the leading environment section (the wall's env chips call this) */
  setEnv(envId: string): void;
}

export function mountRunScience(host: HTMLElement, opts: RunScienceOptions): RunScienceHandle {
  const id = opts.runId;
  const body = el('div', { class: 'science' });
  host.append(body);

  const disposables: Disposable[] = [];
  let destroyed = false;
  let envSwap: ((envId: string) => void) | null = null;

  function track<T extends Disposable>(item: T): T {
    disposables.push(item);
    return item;
  }

  const status = el('div', { class: 'view-status' }, [spinner(`Loading ${id} statistics`)]);
  body.append(status);

  loadRunDetail(id).then(
    (detail) => {
      if (destroyed) return;
      status.remove();
      render(detail);
    },
    (error: unknown) => {
      if (destroyed) return;
      status.replaceChildren(
        errorState(error instanceof Error ? error.message : String(error), undefined, {
          title: `Could not load ${id}`,
        }),
      );
    },
  );

  function render(detail: RunDetail): void {
    const meta = detail.meta;
    const policies = policiesIn(detail.aggregate);
    let policy = policies[0] ?? meta.policies[0] ?? 'oak';
    const shares = envShares(meta);
    const mixed = shares.length > 1;
    const horizon = meta.steps;

    /**
     * `summary` is null for two OPPOSITE reasons and the advice differs. Missing =
     * fleet.py report was never run here, so run it. Unreadable = the file is on
     * disk and was REJECTED as invalid JSON — telling the user to re-run the tool
     * that just wrote it is a doom loop, and claiming the file is absent (or that
     * it "predates" a field it actually contains) is simply false. Every empty
     * state below asks this first.
     */
    const summaryUnreadable = detail.summaryError === 'unreadable';
    const SUMMARY_REJECTED =
      'summary.json is present on this root but could not be parsed, so it was rejected as '
      + 'invalid JSON — no figure below can be read from it. Usual causes: a torn or '
      + 'half-flushed write, or a non-finite number serialized as bare NaN (JSON has no NaN). '
      + 'Re-running tools/fleet.py report helps only once the writer emits valid JSON.';

    /**
     * EVERY policy-dependent number resolves through this (see state.policyView).
     * A policy switch must RE-DERIVE the figures — redrawing the charts while the
     * hero IQM, positivity, survival and completeness still describe the previous
     * policy is the worst kind of wrong: confidently mislabelled.
     */
    const forPolicy = (name: string): PolicyView => policyView(detail, name);

    // ---------------- header ----------------
    // `slot` is display:contents — the badges it holds stay direct flex items of
    // the row, so a policy switch can swap them without a layout seam.
    const completeSlot = el('span', { class: 'slot' });
    const policySlot = el('span', { class: 'slot' });
    const bannerSlot = el('div', { class: 'slot' });

    // Embedded below the wall: the wall names the run, so the header here is
    // just the schedule strip, the chips, and the banners.
    body.append(
      el('section', { class: 'run-head' }, [
        envSpectrum(shares),
        el('div', { class: 'run-head-body' }, [
          el('div', {
            class: 'run-envs',
            text: shares.map((s) => `${envName(s.envId)} ×${Math.round(s.lanes)}`).join('   ·   '),
          }),
          el('div', { class: 'run-chips' }, [
            completeSlot,
            badge(`B ${meta.batch}`, 'quiet', 'seeds in the fleet'),
            badge(`${formatStep(horizon)} steps`, 'quiet', 'declared horizon'),
            badge(
              `ablate ${meta.ablationLabel ?? 'none'}`,
              meta.ablationLabel && meta.ablationLabel !== 'none' ? 'warn' : 'quiet',
              'OAK_ABLATE for this run',
            ),
            policySlot,
            badge(formatAgo(meta.startedAt ?? null), 'quiet', meta.startedAt ?? undefined),
          ]),
        ]),
      ]),
      bannerSlot,
    );

    function syncPolicyHeader(): void {
      const view = forPolicy(policy);
      completeSlot.replaceChildren(
        view.complete
          ? badge('complete', 'good')
          : badge('incomplete', 'warn', 'Did not reach its declared horizon'),
      );
      policySlot.replaceChildren(badge(policy, 'quiet', 'policy'));
      bannerSlot.replaceChildren(
        ...(view.complete
          ? []
          : [
              el('div', { class: 'banner banner-warn' }, [
                el('strong', { text: 'Incomplete run. ' }),
                el('span', {
                  text: `Reached ${formatStep(view.maxStep)} of ${formatStep(horizon)} declared steps. fleet.py gate refuses these — read it, do not promote on it.`,
                }),
              ]),
            ]),
      );
    }
    syncPolicyHeader();

    // ---------------- seed-tab state (shared with the per-env tab) ----------------
    let currentEnv: string =
      opts.envId !== null && shares.some((sh) => sh.envId === opts.envId)
        ? opts.envId
        : shares[0]?.envId ?? meta.envId;
    let selectedEnv: string = currentEnv;

    // ---------------- learning curves ----------------
    /**
     * A self-loading learning-curve block.
     *
     * The curve is derived server-side from fleet.jsonl (never summary.json), so it
     * draws on legacy roots too, and it is asked for PER ENV: pooling a CartPole seed
     * that caps at 1.0/step with a maze seed that caps at 0.002 would produce a
     * confident line describing no environment at all. Every env panel of a 16-env
     * fleet asks separately and the server answers them all from one parse.
     *
     * Returns immediately with a spinner; `stale` is the panel-rebuild guard, because a
     * policy switch tears the panel down while these fetches are still in the air.
     */
    function learningCurveBlock(
      scope: PanelScope,
      options: { envId?: string; height?: number } = {},
    ): HTMLElement {
      const host = el('div', { class: 'view-status' }, [spinner('Deriving the learning curve')]);
      let stale = false;
      scope.hold({
        destroy() {
          stale = true;
        },
      });
      api.curves(id, { policy, ...(options.envId === undefined ? {} : { envId: options.envId }) }).then(
        (curve) => {
          if (destroyed || stale) return;
          if (curve.steps.length < 2) {
            host.replaceChildren(
              emptyState(
                'Not enough heartbeats',
                'A curve needs at least two heartbeats in fleet.jsonl; this selection has fewer.',
              ),
            );
            return;
          }
          const handle = learningCurve({
            steps: curve.steps,
            iqm: curve.iqm,
            ciLo: curve.ciLo,
            ciHi: curve.ciHi,
            p10: curve.p10,
            p90: curve.p90,
            label: `IQM · ${formatCount(curve.seeds)} seeds`,
            height: options.height ?? 220,
            zeroLine: true,
            yUnit: 'reward / step',
            xUnit: 'steps',
          });
          host.replaceChildren(scope.chart(handle));
          host.className = '';
          handle.refresh();
        },
        (error: unknown) => {
          if (destroyed || stale) return;
          host.replaceChildren(errorState(error instanceof Error ? error.message : String(error)));
        },
      );
      return host;
    }

    // ---------------- tab machinery ----------------
    /**
     * A panel owns everything it built. Charts are kept apart because they also
     * need refresh() on tab show; `handles` is every other disposable (bars,
     * rangeBars, tables, fetch-scoped state) — all of them hold ResizeObservers,
     * so leaving them in the VIEW-level list leaked one observer per policy
     * switch, still watching a detached node.
     */
    interface Panel {
      node: HTMLElement;
      built: boolean;
      charts: ChartHandle[];
      handles: Disposable[];
    }
    interface PanelScope {
      /** register a chart and return its root (refreshed on show, destroyed on rebuild) */
      chart(handle: ChartHandle): HTMLElement;
      /** register any other disposable this panel owns */
      hold<T extends Disposable>(item: T): T;
    }
    const tabs: { key: string; label: string; build(panel: HTMLElement, scope: PanelScope): void }[] = [];
    const panels = new Map<string, Panel>();
    const tabBar = el('div', { class: 'tabs', role: 'tablist' });
    const panelHost = el('div', { class: 'panels' });

    function addTab(key: string, label: string, build: (panel: HTMLElement, scope: PanelScope) => void): void {
      tabs.push({ key, label, build });
      panels.set(key, { node: el('div', { class: 'panel' }), built: false, charts: [], handles: [] });
    }

    function scopeFor(panel: Panel): PanelScope {
      return {
        chart(handle) {
          panel.charts.push(handle);
          return handle.root;
        },
        hold(item) {
          panel.handles.push(item);
          return item;
        },
      };
    }

    /** Tear every panel back down to unbuilt — the policy switch and view destroy. */
    function resetPanels(): void {
      for (const panel of panels.values()) {
        for (const chart of panel.charts) chart.destroy();
        panel.charts.length = 0;
        for (const item of panel.handles) item.destroy();
        panel.handles.length = 0;
        panel.built = false;
        panel.node.replaceChildren();
      }
    }
    track({ destroy: resetPanels });

    // ---------------- overview ----------------
    addTab('env', 'This environment', (panel, scope) => {
      const slice = envSliceResult(detail.summary, currentEnv, policy);
      const info = isEnvId(currentEnv) ? ENVS[currentEnv] : null;
      const section = el('section', { class: 'env-section' }, [
        el('div', { class: 'env-head' }, [
          el('h2', { class: 'env-name', text: envName(currentEnv) }),
          el('span', { class: 'env-sub', text: info ? `${info.action.kind} ${info.action.n} · ${info.category}` : '' }),
        ]),
        el('div', { class: 'stat-row' }, [
          statTile('IQM', formatSigned(num(slice?.rewardRateIQM), 4), {
            tone: tone(num(slice?.rewardRateIQM)), hint: 'all seeds',
          }),
          statTile('recent IQM', formatSigned(num(slice?.recentRewardRateIQM), 4), { hint: 'last window' }),
          statTile('positive', formatPercent(num(slice?.positiveFraction), 0), {
            hint: `${formatCount(num(slice?.seeds))} seeds`,
          }),
          statTile('best seed', formatRate(num(slice?.bestSeedRewardRate)), {
            hint: `seed ${formatCount(num(slice?.bestSeed))}`,
          }),
          statTile('solved-threshold seeds', formatCount(num(slice?.solvedSeeds)), {
            hint: `${formatCount(num(slice?.solvedSeedsRetained))} retained`,
          }),
          statTile('reward / GFLOP', formatRate(num(slice?.bestSeedRewardPerGFLOP)), { hint: 'best seed' }),
        ]),
      ]);
      // NO learning curve here: THE RATE strip above the science IS this
      // environment's learning curve (same statistic, plus the playhead).
      section.style.setProperty('--section-accent', accentCss(envSlug(currentEnv)));
      panel.append(section);

      // Per-seed detail, collapsed by default: valuable, env-specific, optional.
      const detailBlock = el('details', { class: 'block science-details' }, [
        el('summary', { text: `Per-seed detail — ${envName(currentEnv)}` }),
      ]);
      const detailHost = el('div');
      detailBlock.append(detailHost);
      let seedsBuilt = false;
      const buildSeeds = (): void => {
        if (seedsBuilt) return;
        seedsBuilt = true;
        buildSeedsDetail(detailHost, scope);
      };
      detailBlock.addEventListener('toggle', () => {
        if ((detailBlock as HTMLDetailsElement).open) {
          buildSeeds();
          for (const chart of panels.get('env')?.charts ?? []) chart.refresh();
        }
      });
      panel.append(detailBlock);
    });

    addTab('overview', 'Overview', (panel, scope) => {
      const chart = (handle: ChartHandle): HTMLElement => scope.chart(handle);
      // derived HERE, not in render(): this closure re-runs on every policy switch
      const { result } = forPolicy(policy);
      const series = aggregateSeries(detail.aggregate, policy);
      const iqm = num(result?.rewardRateIQM);
      const deaths = detail.deathsCount;
      const div = result?.divergence;
      const divergence = num(div?.rate);
      const divLo = num(div?.wilsonLo);
      const divHi = num(div?.wilsonHi);
      const seeds = num(result?.seeds) ?? meta.batch;
      // The CI belongs next to the point estimate, not in a footnote: an IQM of
      // +0.4433 [0.41, 0.47] and one of +0.4433 [-0.2, 1.1] are different claims.
      const ci = result?.rewardRateIQMCI95;
      const ciText = Array.isArray(ci) && ci.length === 2 && num(ci[0]) !== null && num(ci[1]) !== null
        ? ` [${formatRate(ci[0], 3)}, ${formatRate(ci[1], 3)}]`
        : '';

      panel.append(
        el('div', { class: 'hero' }, [
          el('div', { class: 'hero-figure' }, [
            el('div', { class: 'hero-label', text: 'Average reward rate — IQM ± 95% CI, all seeds' }),
            el('div', {
              class: `hero-value tone-${tone(iqm)}`,
              text: iqm === null ? 'n/a' : `${formatSigned(iqm, 4)}${ciText}`,
            }),
            el('div', {
              class: 'hero-hint',
              text: iqm === null
                ? (summaryUnreadable
                    ? 'summary.json is present but could not be parsed — it is not valid JSON.'
                    : 'No summary.json on this root — run tools/fleet.py report.')
                : `reward per step · interquartile mean over ${formatCount(seeds)} seeds`,
            }),
          ]),
          el('div', { class: 'hero-stats' }, [
            statTile('% seeds > 0', formatPercent(num(result?.positiveFraction), 1), {
              hint: `${formatCount(num(result?.positiveSeeds))} above zero`,
            }),
            statTile('retention rate', formatPercent(num(result?.retentionFraction), 1), {
              hint: 'seeds holding ≥ ½ peak reward rate',
            }),
            statTile('divergence', divergence === null ? '—' : formatPercent(divergence, 2), {
              hint: divLo !== null && divHi !== null
                ? `numerically diverged seeds · 95% CI ${formatPercent(divLo, 2)}–${formatPercent(divHi, 2)}`
                : `${formatCount(deaths)} numerically diverged ${deaths === 1 ? 'seed' : 'seeds'} (Wilson 95% CI shown)`,
              tone: deaths > 0 ? 'neg' : 'neutral',
            }),
            statTile('best seed', formatRate(num(result?.bestSeedRewardRate)), {
              hint: `seed ${formatCount(num(result?.bestSeed))}`,
            }),
            statTile('reward / GFLOP', formatRate(num(result?.rewardPerGFLOP)), { hint: 'the SOTA-target metric' }),
            statTile(
              'throughput',
              num(result?.aggStepsPerSecond) === null ? '—' : `${formatStep(num(result?.aggStepsPerSecond))}/s`,
              { hint: formatWall(series.wallSeconds.at(-1) ?? null) },
            ),
          ]),
        ]),
      );

      if (mixed) {
        panel.append(
          el('p', {
            class: 'note',
            text: `This fleet mixes ${shares.length} environments, so every rate below pools scales that are not comparable — a CartPole seed caps at 1.0/step and a maze seed at 0.002. Read them as the fleet's own aggregate; the Per env tab keeps each environment on its own scale.`,
          }),
        );
      }

      const bandBlock = el('section', { class: 'block' }, [
        eyebrow('Population progress', el('span', { class: 'eyebrow-note', text: 'average reward rate · P10 · median · P90' })),
      ]);
      if (series.steps.length >= 2) {
        bandBlock.append(
          chart(band({
            steps: series.steps,
            p10: series.p10,
            p50: series.p50,
            p90: series.p90,
            height: 240,
            zeroLine: true,
            yUnit: 'reward / step',
            xUnit: 'steps',
          })),
        );
      } else {
        bandBlock.append(emptyState('No aggregate heartbeats', 'aggregate.jsonl is empty or missing for this root.'));
      }
      panel.append(bandBlock);

      // The learning curve belongs to ONE scale. A mixed fleet gets one per env in the
      // Per env tab instead of a pooled line that describes no environment.
      if (!mixed) {
        panel.append(
          el('section', { class: 'block' }, [
            eyebrow(
              'Learning curve',
              el('span', {
                class: 'eyebrow-note',
                text: 'measured reward rate per heartbeat window · IQM ± bootstrap 95% CI · P10–P90',
              }),
            ),
            learningCurveBlock(scope, { height: 240 }),
          ]),
        );
      }

      if (series.steps.length >= 2) {
        panel.append(
          el('div', { class: 'block-grid' }, [
            el('section', { class: 'block' }, [
              eyebrow('Seeds above zero'),
              chart(timeSeries({
                steps: series.steps,
                lines: [{ label: 'positive fraction', values: series.positivity }],
                height: 150,
                yFormat: (v) => formatPercent(v, 0),
                yRange: () => [0, 1],
                yUnit: 'fraction of seeds',
              })),
            ]),
            el('section', { class: 'block' }, [
              eyebrow('Non-diverged seeds'),
              chart(timeSeries({
                steps: series.steps,
                lines: [
                  { label: 'non-diverged', values: series.alive, stroke: VIZ.series },
                  { label: 'diverged', values: series.dead, stroke: VIZ.crit },
                ],
                height: 150,
                yFormat: (v) => formatCount(v),
                yUnit: 'seeds',
                legend: true,
              })),
            ]),
          ]),
        );
      }

      const grid = el('div', { class: 'block-grid' });

      const tiers = result?.thresholdFractions as Record<string, number> | undefined;
      if (tiers) {
        const items = [
          { key: 'gt0', label: '> 0   walking' },
          { key: 'gt1', label: '> 1   milestone' },
          { key: 'gt5', label: '> 5   ~SAC' },
          { key: 'gt10', label: '> 10  SOTA' },
        ]
          .filter((entry) => typeof tiers[entry.key] === 'number')
          .map((entry) => ({ label: entry.label, value: tiers[entry.key] ?? 0 }));
        if (items.length > 0) {
          grid.append(
            el('section', { class: 'block' }, [
              eyebrow('Threshold fractions', el('span', { class: 'eyebrow-note', text: 'fraction of seeds' })),
              scope.hold(bars({ items, domain: [0, 1], ramp: true, format: (v) => formatPercent(v, 1), labelWidth: 130 })).root,
            ]),
          );
        }
      }

      const distribution = result?.rewardRatePercentiles as Record<string, number> | undefined;
      if (distribution) {
        const items = ['p10', 'p25', 'p50', 'p75', 'p90', 'p99']
          .filter((key) => typeof distribution[key] === 'number')
          .map((key) => ({
            label: key.toUpperCase(),
            value: distribution[key] ?? 0,
            display: formatSigned(distribution[key] ?? 0),
          }));
        if (items.length > 0) {
          grid.append(
            el('section', { class: 'block' }, [
              eyebrow('Reward-rate percentiles', el('span', { class: 'eyebrow-note', text: 'reward / step' })),
              scope.hold(bars({ items, labelWidth: 56, format: (v) => formatSigned(v) })).root,
            ]),
          );
        }
      }

      const peak = result?.peak as Record<string, unknown> | undefined;
      if (peak) {
        const rows: RangeRow[] = [];
        const peakStep = parseQuartiles(peak['peakStepQ']);
        const peakRate = parseQuartiles(peak['peakRewardRateQ']);
        const retention = parseQuartiles(peak['retentionQ']);
        if (peakStep) {
          rows.push({
            label: 'peak step', q1: peakStep[0], q2: peakStep[1], q3: peakStep[2],
            domain: [0, Math.max(horizon, peakStep[2])], format: formatStep, hint: 'when each seed peaked',
          });
        }
        if (peakRate) {
          rows.push({
            label: 'peak reward rate', q1: peakRate[0], q2: peakRate[1], q3: peakRate[2],
            domain: [0, Math.max(peakRate[2] * 1.15, 1e-9)], format: (v) => formatRate(v), hint: 'peak reward rate per seed',
          });
        }
        if (retention) {
          rows.push({
            label: 'retention', q1: retention[0], q2: retention[1], q3: retention[2],
            domain: [0, 1], format: (v) => formatRate(v, 3), hint: 'final ÷ peak — 1 means nothing was lost after the peak',
          });
        }
        if (rows.length > 0) {
          grid.append(
            el('section', { class: 'block' }, [
              eyebrow('Peak & retention', el('span', { class: 'eyebrow-note', text: 'quartiles · each row on its own scale' })),
              scope.hold(rangeBars({ rows })).root,
            ]),
          );
        }
      }
      if (grid.childElementCount > 0) panel.append(grid);

      // Retention distribution. Retention is final ÷ peak — a RATIO, so unlike a reward
      // rate it means the same thing in CartPole and HalfCheetah and can be shown for
      // the whole fleet at once. The quartile strip above gives three numbers; this
      // gives the shape, which is where a bimodal population (seeds that held their
      // gait vs seeds that lost it) stops looking like a mediocre median.
      const retentionBySeed = result?.retentionBySeed;
      const retentionValues = retentionBySeed ? Object.values(retentionBySeed).filter((v) => typeof v === 'number') : [];
      const retentionBlock = el('section', { class: 'block' }, [
        eyebrow(
          'Retention distribution',
          el('span', { class: 'eyebrow-note', text: 'fraction of seeds holding at least x of their peak' }),
        ),
      ]);
      if (retentionValues.length > 0) {
        const curve = retentionCurve(retentionValues);
        // The x domain is PINNED to [0, 1] because that is what the ratio can mean:
        // peak is a max over the same heartbeats final is drawn from, so final ÷ peak
        // tops out at 1, and 0 is "kept none of it". A seed whose final rate went
        // negative lands below 0 — on p93 three of them sit near −17, and on a
        // min/max domain those three outliers squeeze the entire informative range
        // into the last 3% of the width, which is how a correct chart becomes an
        // unreadable one. They are clipped from the axis and COUNTED in the caption;
        // the curve still steps down through them, so the height at 0 is honest.
        const belowZero = retentionValues.filter((v) => Number.isFinite(v) && v < 0).length;
        const worst = retentionValues.filter((v) => Number.isFinite(v)).reduce((a, b) => Math.min(a, b), Infinity);
        retentionBlock.append(
          chart(steppedLine({
            steps: curve.steps,
            values: curve.fraction,
            label: 'seeds at or above',
            height: 170,
            yFormat: (v) => formatPercent(v, 0),
            xFormat: (v) => formatRate(v, 2),
            yRange: () => [0, 1],
            xRange: () => [0, 1],
            yUnit: 'fraction of seeds',
            xUnit: 'retention',
          })),
          el('p', {
            class: 'note',
            text: `Read at 0.5 it is the retention rate tile: ${formatPercent(num(result?.retentionFraction), 1)} of `
              + `${formatCount(retentionValues.length)} seeds still hold ≥ ½ their peak reward rate.`
              + (belowZero > 0
                ? ` The axis stops at 0: ${formatCount(belowZero)} ${belowZero === 1 ? 'seed ended' : 'seeds ended'} `
                  + `at a negative reward rate (lowest ${formatRate(worst, 3)}) and ${belowZero === 1 ? 'sits' : 'sit'} `
                  + 'off the left edge, counted in the curve but not drawn.'
                : ''),
          }),
        );
      } else {
        retentionBlock.append(
          summaryUnreadable
            ? emptyState('No per-seed retention on this root', SUMMARY_REJECTED)
            : emptyState(
                'No per-seed retention on this root',
                'summary.json predates the per-seed vectors (summarySchemaVersion 2). '
                  + 'Re-run tools/fleet.py report on this root to write retentionBySeed; nothing is derived from thin air here.',
              ),
        );
      }
      panel.append(retentionBlock);

      const survival = result?.survival;
      if (survival && Object.keys(survival).length > 0) {
        const curve = kmCurve(survival, horizon);
        panel.append(
          el('section', { class: 'block' }, [
            eyebrow('Survival', el('span', {
              class: 'eyebrow-note',
              text: `Kaplan-Meier · restricted mean survival time (RMST) ${formatStep(num(result?.rmst))}`,
            })),
            chart(steppedLine({
              steps: curve.steps,
              values: curve.fraction,
              height: 160,
              yFormat: (v) => formatPercent(v, 2),
              yRange: (min) => [Math.min(min, 0.98), 1],
              yUnit: 'seeds surviving',
            })),
          ]),
        );
      } else {
        panel.append(
          el('section', { class: 'block' }, [
            eyebrow('Survival'),
            emptyState('No divergence', 'Every seed reached the horizon — there is no curve to draw.'),
          ]),
        );
      }
    });

    // ---------------- per env ----------------
    if (mixed) {
      addTab('envs', 'Per env', (panel, scope) => {
        // envSlices[env] is one result PER POLICY, same as summary.results
        const sliceFor = (envId: string): SummaryPolicyResult | null =>
          envSliceResult(detail.summary, envId, policy);
        panel.append(
          el('p', {
            class: 'note',
            text: 'Reward rates are not comparable between environments — CartPole tops out at 1.0/step and HalfCheetah does not. Only positivity is shown side by side; every other number stays inside its own section.',
          }),
        );

        // The env LIST comes from meta.json's envSchedule (via `shares`), never from
        // summary.json. Only the stat tiles and the positivity comparison are summary-
        // derived; the learning curve is computed from fleet.jsonl by /curves and needs
        // no summary at all. Gating the whole tab on envSlices used to blank all 16
        // curves on any root whose summary is missing or unreadable — which is most of
        // them, since ~600 historical roots predate envSlices entirely.
        const withData = shares.filter((share) => sliceFor(share.envId));

        if (withData.length > 0) {
          panel.append(
            el('section', { class: 'block' }, [
              eyebrow('Positive seeds by environment', el('span', { class: 'eyebrow-note', text: 'the one comparable quantity' })),
              scope.hold(bars({
                items: withData.map((share) => {
                  const slice = sliceFor(share.envId);
                  return {
                    label: envName(share.envId),
                    value: num(slice?.positiveFraction) ?? 0,
                    display: formatPercent(num(slice?.positiveFraction), 0),
                    hint: `${formatCount(num(slice?.seeds))} seeds`,
                    onClick: () => document.getElementById(`env-${envSlug(share.envId)}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
                  };
                }),
                domain: [0, 1],
                format: (v) => formatPercent(v, 0),
              })).root,
            ]),
          );
        } else {
          panel.append(
            el('p', {
              class: 'note',
              text: (summaryUnreadable
                ? `${SUMMARY_REJECTED} The per-environment tiles below therefore read “—”. `
                : 'No envSlices in this root’s summary.json, so the per-environment tiles below read “—”. ')
                + 'The curves are unaffected: they are derived from fleet.jsonl, not from the summary.',
            }),
          );
        }

        for (const share of shares) {
          const slice = sliceFor(share.envId);
          const info = isEnvId(share.envId) ? ENVS[share.envId] : null;
          const inspect = el('a', {
            class: 'btn btn-quiet',
            text: 'Watch on the wall',
            href: wallLink(id, share.envId),
          });
          const section = el('section', { class: 'env-section' }, [
            el('div', { class: 'env-head' }, [
              el('h2', { class: 'env-name', text: envName(share.envId) }),
              el('span', { class: 'env-sub', text: info ? `${info.action.kind} ${info.action.n} · ${info.category}` : '' }),
              inspect,
            ]),
            el('div', { class: 'stat-row' }, [
              statTile('IQM', formatSigned(num(slice?.rewardRateIQM), 4), {
                tone: tone(num(slice?.rewardRateIQM)), hint: 'all seeds',
              }),
              statTile('recent IQM', formatSigned(num(slice?.recentRewardRateIQM), 4), { hint: 'last window' }),
              statTile('positive', formatPercent(num(slice?.positiveFraction), 0), {
                hint: `${formatCount(num(slice?.seeds))} seeds`,
              }),
              statTile('best seed', formatRate(num(slice?.bestSeedRewardRate)), {
                hint: `seed ${formatCount(num(slice?.bestSeed))}`,
              }),
              statTile('solved-threshold seeds', formatCount(num(slice?.solvedSeeds)), {
                hint: `${formatCount(num(slice?.solvedSeedsRetained))} retained`,
              }),
              statTile('reward / GFLOP', formatRate(num(slice?.bestSeedRewardPerGFLOP)), { hint: 'best seed' }),
            ]),
            eyebrow(
              'Learning curve',
              el('span', {
                class: 'eyebrow-note',
                text: 'measured reward rate · IQM ± bootstrap 95% CI · P10–P90 · this environment only',
              }),
            ),
            learningCurveBlock(scope, { envId: share.envId, height: 200 }),
          ]);
          section.id = `env-${envSlug(share.envId)}`;
          section.style.setProperty('--section-accent', accentCss(envSlug(share.envId)));
          panel.append(section);
        }
      });
    }

    // ---------------- per-seed detail (env-scoped, lives under the env section) ----------------
    function buildSeedsDetail(panel: HTMLElement, scope: PanelScope): void {
      let columnar: FleetColumnar | null = null;
      let laneChart: ChartHandle | null = null;
      let totalsChart: ChartHandle | null = null;
      let shown: LaneSeries[] = [];
      let stale = false;
      scope.hold({
        destroy() {
          stale = true; // a policy switch rebuilds this panel; drop the in-flight fetch
          laneChart?.destroy();
          totalsChart?.destroy();
        },
      });

      const controls = el('div', { class: 'controls controls-inline' });
      const laneStatus = el('div', { class: 'view-status' }, [spinner('Reading fleet.jsonl')]);
      const chartHost = el('section', { class: 'block' });
      const totalsHost = el('section', { class: 'block' });

      /**
       * The default series is MEASURED: rewardSum differenced over each heartbeat
       * window. `health.rewardRate` is the agent's own internal average-reward
       * estimate ρ̄ (Sources/Main.swift) — a learned parameter that is also a
       * divergence diagnostic, not reward the environment paid. It used to be
       * plotted here under a reward-rate label, which quietly showed the agent's
       * self-assessment instead of its results; it now needs asking for by name.
       */
      const SERIES_MODES = {
        measured: { field: 'rewardSum', label: 'measured reward rate (heartbeat window)', windowed: true },
        rho: { field: 'health.rewardRate', label: 'agent estimated average reward (ρ̄)', windowed: false },
      } as const;
      type SeriesMode = keyof typeof SERIES_MODES;
      let mode: SeriesMode = 'measured';

      const seriesSelect = el('select', { class: 'input input-select', ariaLabel: 'Series' });
      for (const key of Object.keys(SERIES_MODES) as SeriesMode[]) {
        seriesSelect.append(el('option', { value: key, text: SERIES_MODES[key].label }));
      }
      seriesSelect.value = mode;
      seriesSelect.addEventListener('change', () => {
        mode = (seriesSelect.value as SeriesMode) in SERIES_MODES ? (seriesSelect.value as SeriesMode) : 'measured';
        paintLanes();
      });
      controls.append(el('span', { class: 'control-label', text: 'series' }), seriesSelect);

      panel.append(controls, laneStatus, chartHost, totalsHost);

      /**
       * Total reward by seed: the FINAL cumulative `rewardSum` of every seed, sorted.
       *
       * Same fetch as the curves above — no second request — and deliberately the total
       * rather than a rate, because it is the population objective (`fleet total` in
       * tools/table.py) and because sorting it exposes the shape the IQM hides: whether
       * the fleet is one mode or a handful of rockets over a flat negative field.
       */
      function paintTotals(): void {
        if (!columnar) return;
        const finals = laneSeries(columnar, 'rewardSum', mixed ? selectedEnv : undefined)
          .filter((lane) => lane.last !== null);
        totalsChart?.destroy();
        totalsChart = null;
        if (finals.length === 0) {
          totalsHost.replaceChildren();
          return;
        }
        const picked = rankLanes(capLanes(finals, COLUMN_CAP));
        const capped = picked.length < finals.length;
        const recordedSeeds = new Set(
          detail.recordings.filter((r) => r.envId === selectedEnv).map((r) => r.seed),
        );
        totalsChart = columns({
          values: picked.map((lane) => lane.last ?? 0),
          labels: picked.map((lane) =>
            `seed ${lane.seed}${recordedSeeds.has(lane.seed) ? ' · click to watch' : ''}`),
          height: 170,
          yFormat: (v) => formatSigned(v, 3),
          yUnit: 'total reward',
          onColumnClick: (index) => {
            const lane = picked[index];
            if (lane && recordedSeeds.has(lane.seed)) opts.onWatchSeed?.(selectedEnv, lane.seed);
          },
        });
        totalsHost.replaceChildren(
          eyebrow('Total reward by seed', el('span', {
            class: 'eyebrow-note',
            text: (capped
              ? `${picked.length} of ${finals.length} seeds · sorted · extremes and median always kept`
              : `${finals.length} seeds · sorted`) + ' · click a column to watch that seed',
          })),
          totalsChart.root,
        );
        totalsChart.refresh();
      }

      function paintLanes(): void {
        if (!columnar) return;
        laneStatus.hidden = false;
        const { field, label, windowed } = SERIES_MODES[mode];
        const raw = laneSeries(columnar, field, mixed ? selectedEnv : undefined);
        if (raw.length === 0) {
          laneStatus.replaceChildren(
            emptyState(
              mixed ? 'No seeds for this environment' : 'No seed series',
              `The fleet endpoint returned no “${field}” column for this run.`,
            ),
          );
          chartHost.replaceChildren();
          totalsChart?.destroy();
          totalsChart = null;
          totalsHost.replaceChildren();
          return;
        }
        const all = windowed ? toWindowRates(columnar.steps, raw) : raw;
        laneStatus.replaceChildren();
        laneStatus.hidden = true;
        shown = capLanes(all, LANE_CAP);
        const capped = shown.length < all.length;

        laneChart?.destroy();
        laneChart = spaghetti({
          steps: columnar.steps,
          lanes: shown.map((lane) => ({ key: String(lane.seed), label: `seed ${lane.seed}`, values: lane.values })),
          height: 280,
          zeroLine: true,
          yUnit: label,
          xUnit: 'steps',
        });
        chartHost.replaceChildren(
          eyebrow('Per-seed reward-rate curves', el('span', {
            class: 'eyebrow-note',
            text: capped
              ? `${shown.length} of ${all.length} seeds · extremes and median always kept`
              : `${all.length} seeds`,
          })),
          laneChart.root,
        );
        laneChart.refresh();

        paintTotals();

      }

      // Both series arrive in ONE request so the toggle never refetches. The server
      // filters fleet.jsonl by policy (defaulting to `oak`), so the selected policy
      // has to travel with it or a zero/random view silently reads oak's seeds.
      api.fleet(id, { fields: [SERIES_MODES.measured.field, SERIES_MODES.rho.field], policy }).then(
        (data) => {
          if (destroyed || stale) return;
          columnar = data;
          paintLanes();
        },
        (error: unknown) => {
          if (destroyed || stale) return;
          laneStatus.hidden = false;
          laneStatus.replaceChildren(errorState(error instanceof Error ? error.message : String(error)));
        },
      );
    }


    // ---------------- assemble: ONE scrolling page, no tabs ----------------
    // Every section builds eagerly and stacks in reading order. Old ?tab= links
    // (and the retired 'lanes' spelling) scroll to their section instead.
    function buildAll(): void {
      for (const tab of tabs) {
        const panel = panels.get(tab.key);
        if (!panel || panel.built) continue;
        panel.built = true;
        tab.build(panel.node, scopeFor(panel));
        for (const chart of panel.charts) chart.refresh();
      }
    }
    // Assembly order: the CURRENT environment leads; the whole-run statistics
    // (fleet-wide aggregates + the 16-env comparison) sit in one collapsed block.
    const envPanel = panels.get('env');
    if (envPanel) panelHost.append(envPanel.node);
    const runDetails = el('details', { class: 'block science-details science-run' }, [
      el('summary', { text: 'Whole-run statistics — every task, every seed' }),
    ]);
    for (const key of ['overview', 'envs']) {
      const panel = panels.get(key);
      if (panel) runDetails.append(panel.node);
    }
    runDetails.addEventListener('toggle', () => {
      if (!(runDetails as HTMLDetailsElement).open) return;
      for (const key of ['overview', 'envs']) {
        for (const chart of panels.get(key)?.charts ?? []) chart.refresh();
      }
    });
    panelHost.append(runDetails);

    // The wall's env chips land here: rebuild ONLY the env section for the new env.
    envSwap = (envId: string) => {
      if (envId === currentEnv) return;
      currentEnv = envId;
      selectedEnv = envId;
      const panel = panels.get('env');
      if (!panel) return;
      for (const chart of panel.charts) chart.destroy();
      panel.charts.length = 0;
      for (const item of panel.handles) item.destroy();
      panel.handles.length = 0;
      panel.node.replaceChildren();
      panel.built = true;
      tabs.find((tab) => tab.key === 'env')?.build(panel.node, scopeFor(panel));
      for (const chart of panel.charts) chart.refresh();
    };

    if (policies.length > 1) {
      const policySelect = el('select', { class: 'input input-select', ariaLabel: 'Policy' });
      for (const name of policies) policySelect.append(el('option', { value: name, text: name }));
      policySelect.value = policy;
      policySelect.addEventListener('change', () => {
        policy = policySelect.value;
        // numbers AND charts AND the header rebuild together — nothing survives
        // a policy switch except the run's own identity.
        resetPanels();
        syncPolicyHeader();
        buildAll();
      });
      tabBar.append(
        el('div', { class: 'tabs-trailing' }, [el('span', { class: 'control-label', text: 'policy' }), policySelect]),
      );
      body.append(tabBar);
    }

    body.append(panelHost);
    buildAll();
  }

  return {
    setEnv(envId: string) {
      envSwap?.(envId);
    },
    destroy() {
      destroyed = true;
      for (const item of disposables) item.destroy();
      disposables.length = 0;
      body.remove();
    },
  };
}
