
import type { ViewMount } from '../router';
import type { FleetColumnar, RunDetail } from '@shared/types';
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
  policyView,
  populationBand,
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

const COLUMN_CAP = 256;

function lastFinite(values: readonly (number | null)[]): { value: number; index: number } | null {
  for (let i = values.length - 1; i >= 0; i--) {
    const value = values[i];
    if (value !== null && value !== undefined && Number.isFinite(value)) return { value, index: i };
  }
  return null;
}

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


export const mount: ViewMount = (el_, params) => {
  const id = params['id'];
  window.location.replace(id ? `#/?run=${encodeURIComponent(id)}` : '#/');
  void el_;
  return { destroy() {} };
};

export interface RunScienceOptions {
  runId: string;
  envId: string | null;
  onWatchSeed?(envId: string, seed: number, step?: number): void;
  onSeek?(step: number, commit: boolean): void;
}

export interface RunScienceHandle {
  destroy(): void;
  setEnv(envId: string): void;
  setPlayhead(step: number | null): void;
  setHighlight(seed: number | null): void;
}

export function mountRunScience(host: HTMLElement, opts: RunScienceOptions): RunScienceHandle {
  const id = opts.runId;
  const body = el('div', { class: 'science' });
  host.append(body);

  const disposables: Disposable[] = [];
  let destroyed = false;
  let envSwap: ((envId: string) => void) | null = null;
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
    const policy = 'oak';
    const shares = envShares(meta);
    const mixed = shares.length > 1;
    const horizon = meta.steps;

    const summaryUnreadable = detail.summaryError === 'unreadable';
    const SUMMARY_REJECTED =
      'summary.json is present on this root but could not be parsed, so it was rejected as '
      + 'invalid JSON — no figure below can be read from it. Usual causes: a torn or '
      + 'half-flushed write, or a non-finite number serialized as bare NaN (JSON has no NaN). '
      + 'Re-running tools/fleet.py report helps only once the writer emits valid JSON.';

    const forPolicy = (name: string): PolicyView => policyView(detail, name);

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

    let currentEnv: string =
      opts.envId !== null && shares.some((sh) => sh.envId === opts.envId)
        ? opts.envId
        : shares[0]?.envId ?? '';
    let selectedEnv: string = currentEnv;

    interface Panel {
      node: HTMLElement;
      built: boolean;
      charts: ChartHandle[];
      handles: Disposable[];
    }
    interface PanelScope {
      chart(handle: ChartHandle): HTMLElement;
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

    function trackChart(scope: PanelScope, handle: ChartHandle): ChartHandle {
      playheadCharts.add(handle);
      handle.setMarker(playheadStep);
      scope.hold({ destroy() { playheadCharts.delete(handle); } });
      return handle;
    }
    const scrubStep = (step: number, phase: 'drag' | 'end' | 'tap'): void => {
      opts.onSeek?.(Math.round(step), phase !== 'drag');
    };

    addTab('env', 'Per-seed detail', (panel, scope) => {
      buildSeedsDetail(panel, scope);
    });

    const fleetCache = new Map<string, Promise<FleetColumnar>>();
    const fleetFor = (name: string): Promise<FleetColumnar> => {
      let hit = fleetCache.get(name);
      if (!hit) {
        hit = api.fleet(id, { fields: ['rewardSum', 'health.rewardRate', 'firstNonFiniteStep'], policy: name });
        fleetCache.set(name, hit);
      }
      return hit;
    };

    function buildSeedsDetail(panel: HTMLElement, scope: PanelScope): void {
      let columnar: FleetColumnar | null = null;
      const recordedSeeds = new Set(
        detail.recordings.filter((r) => r.envId === selectedEnv).map((r) => r.seed),
      );
      let laneChart: ChartHandle | null = null;
      let totalsChart: ChartHandle | null = null;
      let shown: LaneSeries[] = [];
      let stale = false;
      scope.hold({
        destroy() {
          stale = true;
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
        const band = mode === 'measured' ? populationBand(columnar.steps, all) : undefined;

        if (laneChart) playheadCharts.delete(laneChart);
        laneChart?.destroy();
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
              title: 'IQM and P10–P90 over every seed, each seed\u2019s own curve ghosted, the hero lifted in the env accent · click a highlighted curve to watch that seed at that step · press and drag to scrub playback',
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

      fleetFor(policy).then(
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
