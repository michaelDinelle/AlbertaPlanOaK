// The run-science module the wall embeds (`#/run/:id` itself just redirects).
//
// House rule: metrics are NEVER pooled across environments, and nothing is
// shown twice. The wall already carries the run identity (header bar), every
// per-env reward figure (the learning curve + the rail), and the env switcher
// (the chips) — so this module adds ONLY the two things with no other surface:
// the per-seed detail for the current environment, and fleet-wide population
// health (divergence, survival, retention, positivity), which is population-
// typed or scale-free by construction. The only cross-env chart is positivity,
// because "fraction of seeds above zero" is the one quantity that means the
// same thing in CartPole and in HalfCheetah.

import type { ViewMount } from '../router';
import type { CurvesResponse, FleetColumnar, RunDetail } from '@shared/types';
import { api } from '../api';
import {
  aggregateSeries,
  capLanes,
  envName,
  envShares,
  envSlug,
  formatCount,
  formatPercent,
  formatRate,
  formatSigned,
  formatStep,
  laneSeries,
  loadRunDetail,
  policiesIn,
  policyView,
  rankLanes,
  toWindowRates,
  windowMidpoints,
  windowRates,
  type LaneSeries,
  type PolicyView,
} from '../state';
import {
  accentCss,
  el,
  emptyState,
  errorState,
  eyebrow,
  spinner,
} from '../components/table';
import {
  columns,
  populationChart,
  steppedLine,
  timeSeries,
  type ChartHandle,
} from '../components/chart';

interface Disposable {
  destroy(): void;
}

const LANE_CAP = 140;

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
function lastFinite(values: readonly (number | null)[]): { value: number; index: number } | null {
  for (let i = values.length - 1; i >= 0; i--) {
    const value = values[i];
    if (value !== null && value !== undefined && Number.isFinite(value)) return { value, index: i };
  }
  return null;
}

/** final ÷ peak of a windowed-rate series — the rail's own retention semantics */
function retentionOf(window: readonly (number | null)[]): number | null {
  let peak = -Infinity;
  for (const value of window) {
    if (value !== null && value !== undefined && Number.isFinite(value) && value > peak) peak = value;
  }
  const final = lastFinite(window);
  if (final === null || !Number.isFinite(peak) || peak <= 0) return null;
  return final.value / peak;
}

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
  // Anchor the FULL [0, 1] domain. A reverse ECDF is defined everywhere:
  // left of the lowest observed value the fraction is "everything at or above
  // 0", and right of the highest it is 0 — without these anchors the curve
  // simply started at the first observation and looked like missing data.
  const first = steps[0];
  if (n > 0 && first !== undefined && first > 0) {
    const atZero = sorted.filter((v) => v >= 0).length / n;
    steps.unshift(0);
    fraction.unshift(atZero);
  }
  const last = steps[steps.length - 1];
  if (n > 0 && last !== undefined && last < 1) {
    steps.push(1);
    fraction.push(0);
  }
  return { steps, fraction };
}

// ---------------------------------------------------------------------------
// the science module
//
// This file no longer owns a page. `#/run/:id` redirects into the wall, and the
// wall embeds `mountRunScience` below its learning-curve strip: the CURRENT
// environment's per-seed detail (collapsible; total-reward columns are
// click-to-watch), then population health in a second collapsed block.
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
  /** click-to-watch: a seed picked in the totals columns or a population lane */
  onWatchSeed?(envId: string, seed: number, step?: number): void;
  /** scrub from a science chart's timeline; commit=false while dragging */
  onSeek?(step: number, commit: boolean): void;
}

export interface RunScienceHandle {
  destroy(): void;
  /** swap the leading environment section (the wall's env chips call this) */
  setEnv(envId: string): void;
  /** ride the wall's clock: the gold playhead on every step-axis chart */
  setPlayhead(step: number | null): void;
  /** the hero seed — its lane is lifted in the population chart */
  setHighlight(seed: number | null): void;
}

export function mountRunScience(host: HTMLElement, opts: RunScienceOptions): RunScienceHandle {
  const id = opts.runId;
  const body = el('div', { class: 'science' });
  host.append(body);

  const disposables: Disposable[] = [];
  let destroyed = false;
  let envSwap: ((envId: string) => void) | null = null;
  /** every chart whose x axis is the run's own timeline */
  const playheadCharts = new Set<ChartHandle>();
  let playheadStep: number | null = null;
  let heroSeed: number | null = null;
  let repaintLanes: (() => void) | null = null;

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
    // The wall's single bar already carries the run's identity: name, schedule
    // line, ablation, age, best-kernel/incomplete badges, and the env chips name
    // every environment with its recorded-seed count. The science therefore adds
    // NO identity of its own — only the loud incomplete-run banner, because that
    // one changes how every number below must be read.
    const bannerSlot = el('div', { class: 'slot' });
    body.append(bannerSlot);

    function syncBanner(): void {
      const view = forPolicy(policy);
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
    syncBanner();

    // ---------------- seed-tab state (shared with the per-env tab) ----------------
    let currentEnv: string =
      opts.envId !== null && shares.some((sh) => sh.envId === opts.envId)
        ? opts.envId
        : shares[0]?.envId ?? meta.envId;
    let selectedEnv: string = currentEnv;

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
    const panelHost = el('div', { class: 'panels' });

    function addTab(key: string, label: string, build: (panel: HTMLElement, scope: PanelScope) => void): void {
      tabs.push({ key, label, build });
      panels.set(key, { node: el('div', { class: 'slot' }), built: false, charts: [], handles: [] });
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

    /** register a step-axis chart for the playhead + scrub grammar */
    function trackChart(scope: PanelScope, handle: ChartHandle): ChartHandle {
      playheadCharts.add(handle);
      handle.setMarker(playheadStep);
      scope.hold({ destroy() { playheadCharts.delete(handle); } });
      return handle;
    }
    const scrubStep = (step: number, phase: 'drag' | 'end' | 'tap'): void => {
      opts.onSeek?.(Math.round(step), phase !== 'drag');
    };

    // ---------------- this environment: the per-seed detail ----------------
    // Deliberately NOTHING else, and NOTHING folded. The env's stat tiles used
    // to sit here and every one of them restated an always-visible surface: IQM
    // and recent IQM are the learning curve's line and endpoint, positive is
    // the rail's "seeds > 0", best seed is the rail's row #1, and solved /
    // reward-per-GFLOP moved INTO the rail. Per-seed curves and totals are the
    // only env science with no other home — two cells beside the wall's curve.
    addTab('env', 'Per-seed detail', (panel, scope) => {
      buildSeedsDetail(panel, scope);
    });

    // ---------------- this environment: the per-seed detail ----------------
    // Deliberately NOTHING else, and NOTHING folded. The env's stat tiles used
    // to sit here and every one of them restated an always-visible surface: IQM
    // and recent IQM are the learning curve's line and endpoint, positive is
    // the rail's "seeds > 0", best seed is the rail's row #1, and solved /
    // reward-per-GFLOP moved INTO the rail. Per-seed curves and totals are the
    // only env science with no other home — two cells beside the wall's curve.
    addTab('env', 'Per-seed detail', (panel, scope) => {
      buildSeedsDetail(panel, scope);
    });

    // ---------------- per-seed detail (env-scoped) ----------------
    // The columnar fetch is cached per POLICY: an env switch rebuilds this panel
    // but filters the same fleet columnar client-side — refetching megabytes of
    // per-lane series sixteen times for one run would be self-inflicted latency.
    const fleetCache = new Map<string, Promise<FleetColumnar>>();
    const fleetFor = (name: string): Promise<FleetColumnar> => {
      let hit = fleetCache.get(name);
      if (!hit) {
        hit = api.fleet(id, { fields: ['rewardSum', 'health.rewardRate', 'firstNonFiniteStep'], policy: name });
        fleetCache.set(name, hit);
      }
      return hit;
    };
    const curvesCache = new Map<string, Promise<CurvesResponse>>();
    const curvesFor = (envId: string, name: string): Promise<CurvesResponse> => {
      const key = `${name}|${envId}`;
      let hit = curvesCache.get(key);
      if (!hit) {
        hit = api.curves(id, { policy: name, ...(mixed ? { envId } : {}) });
        curvesCache.set(key, hit);
      }
      return hit;
    };

    function buildSeedsDetail(panel: HTMLElement, scope: PanelScope): void {
      let columnar: FleetColumnar | null = null;
      let curves: CurvesResponse | null = null;
      const recordedSeeds = new Set(
        detail.recordings.filter((r) => r.envId === selectedEnv).map((r) => r.seed),
      );
      let laneChart: ChartHandle | null = null;
      let totalsChart: ChartHandle | null = null;
      let shown: LaneSeries[] = [];
      let stale = false;
      scope.hold({
        destroy() {
          stale = true; // a policy/env switch rebuilds this panel; drop the in-flight fetch
          if (laneChart) playheadCharts.delete(laneChart);
          if (repaintLanes) repaintLanes = null;
          if (seedsChart) playheadCharts.delete(seedsChart);
          if (throughputChart) playheadCharts.delete(throughputChart);
          laneChart?.destroy();
          totalsChart?.destroy();
          seedsChart?.destroy();
          throughputChart?.destroy();
          retentionChart?.destroy();
        },
      });

      const laneStatus = el('div', { class: 'view-status' }, [spinner('Reading fleet.jsonl')]);
      const chartHost = el('section', { class: 'science-cell' });
      const seedsHost = el('section', { class: 'science-cell' });
      const throughputHost = el('section', { class: 'science-cell' });
      const retentionHost = el('section', { class: 'science-cell' });
      const totalsHost = el('section', { class: 'science-cell' });
      let seedsChart: ChartHandle | null = null;
      let throughputChart: ChartHandle | null = null;
      let retentionChart: ChartHandle | null = null;

      /**
       * The default series is MEASURED: rewardSum differenced over each heartbeat
       * window. `health.rewardRate` is the agent's own internal average-reward
       * estimate ρ̄ (Sources/Main.swift) — a learned parameter that is also a
       * divergence diagnostic, not reward the environment paid. It used to be
       * plotted here under a reward-rate label, which quietly showed the agent's
       * self-assessment instead of its results; it now needs asking for by name.
       */
      const SERIES_MODES = {
        measured: { field: 'rewardSum', label: 'measured rate', windowed: true },
        rho: { field: 'health.rewardRate', label: 'agent ρ̄', windowed: false },
      } as const;
      type SeriesMode = keyof typeof SERIES_MODES;
      let mode: SeriesMode = 'measured';

      const seriesSelect = el('select', {
        class: 'input input-select input-mini',
        ariaLabel: 'Series',
        title: 'measured = reward the environment paid, per heartbeat window · ρ̄ = the agent’s own average-reward estimate (a learned parameter and divergence diagnostic)',
      });
      for (const key of Object.keys(SERIES_MODES) as SeriesMode[]) {
        seriesSelect.append(el('option', { value: key, text: SERIES_MODES[key].label }));
      }
      seriesSelect.value = mode;
      seriesSelect.addEventListener('change', () => {
        mode = (seriesSelect.value as SeriesMode) in SERIES_MODES ? (seriesSelect.value as SeriesMode) : 'measured';
        paintLanes();
      });

      panel.append(chartHost, seedsHost, throughputHost, retentionHost, totalsHost);
      chartHost.append(laneStatus);

      /**
       * Total reward by seed: the FINAL cumulative `rewardSum` of every seed, sorted.
       *
       * Same fetch as the curves — no second request — and deliberately the total
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
        const heroIndex = heroSeed === null ? -1 : picked.findIndex((lane) => lane.seed === heroSeed);
        totalsChart = columns({
          ...(heroIndex >= 0 ? { highlight: { index: heroIndex, color: accentCss(envSlug(selectedEnv)) } } : {}),
          values: picked.map((lane) => lane.last ?? 0),
          labels: picked.map((lane) =>
            `seed ${lane.seed}${recordedSeeds.has(lane.seed) ? ' · click to watch' : ''}`),
          fillHeight: true,
          yFormat: (v) => formatSigned(v, 3),
          yUnit: 'total reward',
          onColumnClick: (index) => {
            const lane = picked[index];
            if (lane && recordedSeeds.has(lane.seed)) opts.onWatchSeed?.(selectedEnv, lane.seed);
          },
        });
        totalsHost.replaceChildren(
          eyebrow('Totals by seed', el('span', {
            class: 'eyebrow-note',
            text: (capped ? `top ${picked.length} · ` : '') + 'click to watch',
            title: 'final cumulative reward of every seed, sorted — the population objective · click a column to watch that seed',
          })),
          totalsChart.root,
        );
        totalsChart.refresh();
      }

      /**
       * Survival, positivity, retention and throughput for THIS environment
       * (throughput excepted — all lanes step in lockstep on one GPU, so it is
       * fleet-wide and says so). Survival and positivity are both fractions of
       * this env's seeds over the same timeline, so they share ONE tile.
       * Semantics match the rail exactly: divergedAt = last value of
       * firstNonFiniteStep, retention = final ÷ peak of the windowed rate.
       */
      function paintEnvHealth(): void {
        if (!columnar) return;
        const steps = columnar.steps;
        const cum = laneSeries(columnar, 'rewardSum', mixed ? selectedEnv : undefined);
        const n = cum.length;
        if (seedsChart) playheadCharts.delete(seedsChart);
        if (throughputChart) playheadCharts.delete(throughputChart);
        seedsChart?.destroy(); seedsChart = null;
        throughputChart?.destroy(); throughputChart = null;
        retentionChart?.destroy(); retentionChart = null;
        if (n === 0 || steps.length < 2) {
          seedsHost.replaceChildren();
          throughputHost.replaceChildren();
          retentionHost.replaceChildren();
          return;
        }

        const deathSteps: number[] = [];
        for (const lane of laneSeries(columnar, 'firstNonFiniteStep', mixed ? selectedEnv : undefined)) {
          const death = lastFinite(lane.values);
          if (death !== null) deathSteps.push(death.value);
        }
        deathSteps.sort((a, b) => a - b);
        const surviving = steps.map((step) => {
          let dead = 0;
          while (dead < deathSteps.length && (deathSteps[dead] ?? Infinity) <= step) dead++;
          return (n - dead) / n;
        });
        let rmst = 0;
        let prev = 0;
        for (let i = 0; i < steps.length; i++) {
          rmst += (surviving[i] ?? 1) * ((steps[i] ?? prev) - prev);
          prev = steps[i] ?? prev;
        }
        const positive = steps.map((_, i) => {
          let count = 0;
          for (const lane of cum) {
            const v = lane.values[i];
            if (v !== null && v !== undefined && Number.isFinite(v) && v > 0) count++;
          }
          return count / n;
        });
        seedsChart = trackChart(scope, timeSeries({
          steps: [0, ...steps],
          lines: [
            { label: 'surviving', values: [1, ...surviving], stroke: '#7fd47a' },
            { label: '> 0', values: [0, ...positive], stroke: '#2b98d9' },
          ],
          fillHeight: true,
          legend: true,
          yFormat: (v) => formatPercent(v, 0),
          yRange: () => [0, 1],
          xRange: () => [0, horizon],
          yUnit: 'fraction of seeds',
          onSeek: scrubStep,
        }));
        seedsHost.replaceChildren(
          eyebrow('Seeds', el('span', {
            class: 'eyebrow-note',
            text: `${formatCount(deathSteps.length)} div · RMST ${formatStep(rmst)}`,
            title: 'this environment\u2019s own lanes · surviving = non-diverged fraction · > 0 = fraction with positive cumulative reward · RMST = restricted mean survival time',
          })),
          seedsChart.root,
        );
        seedsChart.refresh();

        const agg = aggregateSeries(detail.aggregate, policy);
        if (agg.steps.length >= 2) {
          // windowed average -> window centers; step-hold anchors at 0 and the horizon
          const aggLast = agg.steps[agg.steps.length - 1] ?? 0;
          throughputChart = trackChart(scope, timeSeries({
            steps: [0, ...windowMidpoints(agg.steps), aggLast],
            lines: [{ label: 'steps/s', values: [agg.sps[0] ?? null, ...agg.sps, agg.sps[agg.sps.length - 1] ?? null] }],
            fillHeight: true,
            yFormat: (v) => `${formatStep(v)}/s`,
            xRange: () => [0, horizon],
            yUnit: 'agg steps / second',
            onSeek: scrubStep,
          }));
          throughputHost.replaceChildren(
            eyebrow('Throughput', el('span', { class: 'eyebrow-note', text: 'all tasks · agg steps/s' })),
            throughputChart.root,
          );
          throughputChart.refresh();
        } else {
          throughputHost.replaceChildren();
        }

        const retentionValues: number[] = [];
        for (const lane of cum) {
          const r = retentionOf(windowRates(steps, lane.values));
          if (r !== null) retentionValues.push(r);
        }
        if (retentionValues.length > 0) {
          const curve = retentionCurve(retentionValues);
          const belowZero = retentionValues.filter((v) => v < 0).length;
          retentionChart = steppedLine({
            steps: curve.steps,
            values: curve.fraction,
            label: 'seeds at or above',
            fillHeight: true,
            yFormat: (v) => formatPercent(v, 0),
            xFormat: (v) => formatRate(v, 2),
            yRange: () => [0, 1],
            xRange: () => [0, 1],
            yUnit: 'fraction of seeds',
            xUnit: 'retention',
          });
          retentionHost.replaceChildren(
            eyebrow('Retention', el('span', {
              class: 'eyebrow-note',
              text: belowZero > 0 ? `final ÷ peak · ${formatCount(belowZero)} < 0 off-axis` : 'final ÷ peak',
              title: 'this environment\u2019s own lanes · fraction of seeds holding at least x of their peak windowed reward rate',
            })),
            retentionChart.root,
          );
          retentionChart.refresh();
        } else {
          retentionHost.replaceChildren(
            eyebrow('Retention'),
            emptyState('No positive peaks', 'No seed of this environment ever held a positive windowed rate.'),
          );
        }
      }

      function paintLanes(): void {
        if (!columnar) return;
        laneStatus.hidden = false;
        const { field, windowed } = SERIES_MODES[mode];
        const raw = laneSeries(columnar, field, mixed ? selectedEnv : undefined);
        if (raw.length === 0) {
          laneStatus.replaceChildren(
            emptyState(
              mixed ? 'No seeds for this environment' : 'No seed series',
              `The fleet endpoint returned no “${field}” column for this run.`,
            ),
          );
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

        // The population statistics overlay ONLY when the server curve shares the
        // lanes' exact heartbeat grid (same fleet.jsonl, so it always should) and
        // the lanes are in MEASURED mode — ρ̄ is a different quantity than the
        // measured band and overlaying them would be confidently mislabelled.
        const grid = curves;
        const sameGrid = grid !== null
          && grid.steps.length === columnar.steps.length
          && grid.steps[0] === columnar.steps[0]
          && grid.steps.at(-1) === columnar.steps.at(-1);
        const band = mode === 'measured' && grid !== null && sameGrid
          ? { p10: grid.p10, p90: grid.p90, iqm: grid.iqm, ciLo: grid.ciLo, ciHi: grid.ciHi }
          : undefined;

        if (laneChart) playheadCharts.delete(laneChart);
        laneChart?.destroy();
        // TIME PLACEMENT IS THE CONTRACT. Windowed averages (measured mode,
        // and the IQM/CI band) are plotted at their window CENTERS — plotting
        // window (a, b] at b rendered a seed's peak up to one full window
        // after the scene showed it dying. The exact anchors close the edges:
        // t = 0 is truly 0 delivered, and the final window's value holds to
        // the horizon. ρ̄ is a point-in-time estimate read AT each heartbeat,
        // so it stays on the raw heartbeat grid (with its true 0 init).
        const centered = mode === 'measured';
        const lastStep = columnar.steps[columnar.steps.length - 1] ?? 0;
        const steps0 = centered
          ? [0, ...windowMidpoints(columnar.steps), lastStep]
          : [0, ...columnar.steps];
        const seriesOf = (values: readonly (number | null)[]): (number | null)[] => (
          centered ? [0, ...values, values[values.length - 1] ?? null] : [0, ...values]
        );
        const anchored = band
          ? {
              p10: seriesOf(band.p10),
              p90: seriesOf(band.p90),
              iqm: seriesOf(band.iqm),
              ciLo: seriesOf(band.ciLo ?? []),
              ciHi: seriesOf(band.ciHi ?? []),
            }
          : undefined;
        laneChart = populationChart({
          steps: steps0,
          lanes: shown.map((lane) => ({ key: String(lane.seed), label: `seed ${lane.seed}`, values: seriesOf(lane.values) })),
          ...(anchored ? { band: anchored } : {}),
          highlightKey: heroSeed === null ? null : String(heroSeed),
          highlightStroke: accentCss(envSlug(selectedEnv)),
          fillHeight: true,
          zeroLine: true,
          xRange: () => [0, horizon],
          yUnit: SERIES_MODES[mode].label,
          xUnit: 'steps',
          onLaneClick: (key, step) => {
            const seed = Number(key);
            if (recordedSeeds.has(seed)) opts.onWatchSeed?.(selectedEnv, seed, Math.round(step));
            else opts.onSeek?.(Math.round(step), true);
          },
          onSeek: (step, phase) => opts.onSeek?.(Math.round(step), phase !== 'drag'),
        });
        playheadCharts.add(laneChart);
        laneChart.setMarker(playheadStep);
        chartHost.replaceChildren(
          laneStatus,
          eyebrow('Reward rate', el('span', { class: 'slot' }, [
            el('span', {
              class: 'eyebrow-note',
              text: (capped ? `${shown.length}/${all.length} seeds · ` : 'every seed · ') + 'drag to seek',
              title: 'IQM ± bootstrap 95% CI and P10–P90 over every seed, each seed\u2019s own curve ghosted, the hero lifted in the env accent · click a highlighted curve to watch that seed at that step · press and drag to scrub playback',
            }),
            seriesSelect,
          ])),
          laneChart.root,
        );
        laneChart.refresh();

        paintTotals();
        paintEnvHealth();

      }
      repaintLanes = paintLanes;

      // Both series arrive in ONE request so the toggle never refetches. The server
      // filters fleet.jsonl by policy (defaulting to `oak`), so the selected policy
      // has to travel with it or a zero/random view silently reads oak's seeds.
      // The population curve rides the same paint: one tile, one source of truth.
      Promise.all([
        fleetFor(policy),
        curvesFor(selectedEnv, policy).catch(() => null),
      ]).then(
        ([data, curve]) => {
          if (destroyed || stale) return;
          columnar = data;
          curves = curve;
          paintLanes();
        },
        (error: unknown) => {
          if (destroyed || stale) return;
          laneStatus.hidden = false;
          laneStatus.replaceChildren(errorState(error instanceof Error ? error.message : String(error)));
        },
      );
    }


    // ---------------- assemble: ONE fixed row of six cells ----------------
    // Learning curve (the wall's cell, adopted — the wall keeps painting it),
    // per-seed curves, totals, then survival, positivity, retention. One fixed
    // row is what lets the stage above flex to a real video shape.
    function buildAll(): void {
      for (const tab of tabs) {
        const panel = panels.get(tab.key);
        if (!panel || panel.built) continue;
        panel.built = true;
        tab.build(panel.node, scopeFor(panel));
        for (const chart of panel.charts) chart.refresh();
      }
    }
    const chartsRow = el('div', { class: 'science-row' });
    const envPanel = panels.get('env');
    if (envPanel) chartsRow.append(envPanel.node);
    panelHost.append(chartsRow);

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
        // numbers AND charts AND the banner rebuild together — nothing survives
        // a policy switch except the run's own identity.
        resetPanels();
        syncBanner();
        buildAll();
      });
      body.append(
        el('div', { class: 'controls controls-inline' }, [el('span', { class: 'control-label', text: 'policy' }), policySelect]),
      );
    }

    body.append(panelHost);
    buildAll();
  }

  return {
    setEnv(envId: string) {
      envSwap?.(envId);
    },
    setPlayhead(step: number | null) {
      playheadStep = step;
      for (const chart of playheadCharts) chart.setMarker(step);
    },
    setHighlight(seed: number | null) {
      if (seed === heroSeed) return;
      heroSeed = seed;
      repaintLanes?.();
    },
    destroy() {
      destroyed = true;
      for (const item of disposables) item.destroy();
      disposables.length = 0;
      body.remove();
    },
  };
}
