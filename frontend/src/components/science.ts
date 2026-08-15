import type { RunDetail } from '@shared/types';
import type { Tape } from '../tape';
import {
  binnedRates,
  deliveredRate,
  envLanes,
  envShares,
  envSlug,
  formatCount,
  formatPercent,
  formatRate,
  formatSigned,
  formatStep,
  laneDeath,
  laneRetention,
  loadRunDetail,
  loadTape,
  survivalSeries,
  throughputSeries,
} from '../state';
import { accentCss, el, emptyState, errorState, eyebrow, spinner } from './table';
import { columns, populationChart, steppedLine, timeSeries, type ChartHandle } from './chart';

interface Disposable {
  destroy(): void;
}

export const LANE_CAP = 140;
export const COLUMN_CAP = 256;
export const CHART_BINS = 240;

interface LaneSeries {
  seed: number;
  values: (number | null)[];
  last: number | null;
}

function lastFinite(values: readonly (number | null)[]): number | null {
  for (let i = values.length - 1; i >= 0; i--) {
    const v = values[i];
    if (v !== null && v !== undefined && Number.isFinite(v)) return v;
  }
  return null;
}

export function capSeries(series: readonly LaneSeries[], cap: number): LaneSeries[] {
  if (series.length <= cap) return [...series];
  const ranked = [...series].sort((a, b) => (a.last ?? -Infinity) - (b.last ?? -Infinity));
  const keep = new Set<number>([0, ranked.length - 1, Math.floor(ranked.length / 2)]);
  const stride = (ranked.length - 1) / Math.max(1, cap - keep.size);
  for (let i = 0; keep.size < cap && i < ranked.length; i++) {
    keep.add(Math.min(ranked.length - 1, Math.round(i * stride)));
  }
  return [...keep].sort((a, b) => a - b)
    .map((index) => ranked[index])
    .filter((entry): entry is LaneSeries => entry !== undefined);
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
    steps.unshift(0);
    fraction.unshift(sorted.filter((v) => v >= 0).length / n);
  }
  const last = steps[steps.length - 1];
  if (n > 0 && last !== undefined && last < 1) {
    steps.push(1);
    fraction.push(0);
  }
  return { steps, fraction };
}

export interface RunScienceOptions {
  runId: string;
  envId: string | null;
  onWatchSeed?(envId: string, seed: number, step?: number): void;
  onSeek?(step: number, commit: boolean): void;
}

export interface RunScienceHandle {
  destroy(): void;
  setEnv(envId: string): void;
  setTape(tape: Tape): void;
  setPlayhead(step: number | null): void;
  setHighlight(seed: number | null): void;
}

export function mountRunScience(host: HTMLElement, opts: RunScienceOptions): RunScienceHandle {
  const id = opts.runId;
  const body = el('div', { class: 'science' });
  host.append(body);

  let destroyed = false;
  let detail: RunDetail | null = null;
  let tape: Tape | null = null;
  let selectedEnv: string | null = opts.envId;
  let heroSeed: number | null = null;
  let playheadStep: number | null = null;
  const playheadCharts = new Set<ChartHandle>();
  const charts: ChartHandle[] = [];
  const disposables: Disposable[] = [];
  let totalsChart: ChartHandle | null = null;
  let repaintTotals: (() => void) | null = null;

  const status = el('div', { class: 'view-status' }, [spinner(`Loading ${id} statistics`)]);
  body.append(status);

  const bannerSlot = el('div', { class: 'slot' });
  const rateHost = el('section', { class: 'science-cell' });
  const seedsHost = el('section', { class: 'science-cell' });
  const throughputHost = el('section', { class: 'science-cell' });
  const retentionHost = el('section', { class: 'science-cell' });
  const totalsHost = el('section', { class: 'science-cell' });
  const row = el('div', { class: 'science-row' }, [
    rateHost, seedsHost, throughputHost, retentionHost, totalsHost,
  ]);

  const SERIES_MODES = {
    measured: { label: 'measured rate' },
    rho: { label: 'agent ρ̄' },
  } as const;
  type SeriesMode = keyof typeof SERIES_MODES;
  let mode: SeriesMode = 'measured';
  const seriesSelect = el('select', {
    class: 'input input-select input-mini',
    ariaLabel: 'Series',
    title: 'measured = reward the environment paid, averaged exactly over each chart bin · ρ̄ = the agent’s own average-reward estimate (a learned parameter and divergence diagnostic)',
  });
  for (const key of Object.keys(SERIES_MODES) as SeriesMode[]) {
    seriesSelect.append(el('option', { value: key, text: SERIES_MODES[key].label }));
  }
  seriesSelect.value = mode;
  seriesSelect.addEventListener('change', () => {
    mode = (seriesSelect.value as SeriesMode) in SERIES_MODES ? (seriesSelect.value as SeriesMode) : 'measured';
    paint();
  });

  const scrubStep = (step: number, phase: 'drag' | 'end' | 'tap'): void => {
    opts.onSeek?.(Math.round(step), phase !== 'drag');
  };

  function clearCharts(): void {
    for (const chart of charts) {
      playheadCharts.delete(chart);
      chart.destroy();
    }
    charts.length = 0;
  }

  function trackChart(handle: ChartHandle, seekable = true): ChartHandle {
    charts.push(handle);
    if (seekable) {
      playheadCharts.add(handle);
      handle.setMarker(playheadStep);
    }
    return handle;
  }

  function rhoSeries(t: Tape, lanes: readonly number[], bins: number): {
    x: number[]; perLane: (number | null)[][];
  } {
    const rateIdx = t.fields.indexOf('rewardRate');
    const edgeRows: number[] = [];
    const want = Math.max(1, Math.min(bins, t.rows));
    let prev = -1;
    for (let b = 1; b <= want; b++) {
      const r = Math.round((b * t.rows) / want) - 1;
      if (r > prev) {
        edgeRows.push(r);
        prev = r;
      }
    }
    const x = edgeRows.map((r) => t.stepEnd[r] ?? 0);
    const perLane = lanes.map((lane) =>
      edgeRows.map((r) => {
        const v = t.data[(r * t.fields.length + rateIdx) * t.laneCount + lane] ?? NaN;
        return Number.isFinite(v) ? v : null;
      }));
    return { x, perLane };
  }

  interface RewardBuilt {
    env: string;
    mode: SeriesMode;
    chart: ChartHandle;
    seeds: number[];
    x0: number[];
    cols: (number | null)[][];
    bandCount: number;
  }
  let reward: RewardBuilt | null = null;

  function heroColumn(built: RewardBuilt): (number | null)[] {
    const li = heroSeed === null ? -1 : built.seeds.indexOf(heroSeed);
    if (li < 0) return built.x0.map(() => null);
    return built.cols[built.bandCount + li] ?? built.x0.map(() => null);
  }

  function rewardData(built: RewardBuilt): Parameters<ChartHandle['setData']>[0] {
    return [
      built.x0,
      ...built.cols.slice(0, built.bandCount),
      heroColumn(built),
      ...built.cols.slice(built.bandCount),
    ] as unknown as Parameters<ChartHandle['setData']>[0];
  }

  function dropReward(): void {
    if (reward === null) return;
    playheadCharts.delete(reward.chart);
    const at = charts.indexOf(reward.chart);
    if (at >= 0) charts.splice(at, 1);
    reward.chart.destroy();
    reward = null;
  }

  function ensureReward(
    t: Tape,
    lanes: readonly number[],
    seeds: readonly number[],
    horizon: number,
    accent: string,
    recordedSeeds: ReadonlySet<number>,
  ): void {
    const isRho = mode === 'rho';
    const binned = isRho ? null : binnedRates(t, lanes, CHART_BINS);
    const rho = isRho ? rhoSeries(t, lanes, CHART_BINS) : null;
    const x = isRho ? rho?.x ?? [] : binned?.x ?? [];
    const perLane = isRho ? rho?.perLane ?? [] : binned?.perLane ?? [];
    const x0 = [0, ...x];
    const bandCols: (number | null)[][] = binned !== null
      ? [[null, ...binned.p90], [null, ...binned.p10], [null, ...binned.iqm]]
      : [];
    const laneCols = perLane.map((values) => [null, ...values]);
    const cols = [...bandCols, ...laneCols];

    if (
      reward !== null
      && reward.env === selectedEnv
      && reward.mode === mode
      && reward.seeds.length === seeds.length
      && lanes.length <= LANE_CAP
    ) {
      reward.x0 = x0;
      reward.cols = cols;
      reward.chart.setData(rewardData(reward));
      return;
    }

    dropReward();
    const series: LaneSeries[] = perLane.map((values, li) => ({
      seed: seeds[li] ?? 0,
      values,
      last: lastFinite(values),
    }));
    const shown = capSeries(series, LANE_CAP);
    const capped = shown.length < series.length;
    const built: RewardBuilt = {
      env: selectedEnv ?? '',
      mode,
      chart: null as unknown as ChartHandle,
      seeds: shown.map((lane) => lane.seed),
      x0,
      cols: capped
        ? [...bandCols, ...shown.map((lane) => [null, ...lane.values] as (number | null)[])]
        : cols,
      bandCount: bandCols.length,
    };
    built.chart = populationChart({
      steps: x0,
      lanes: shown.map((lane) => ({
        key: String(lane.seed),
        label: `seed ${lane.seed}`,
        values: [null, ...lane.values],
      })),
      ...(binned !== null
        ? {
            band: {
              p10: [null, ...binned.p10],
              p90: [null, ...binned.p90],
              iqm: [null, ...binned.iqm],
            },
          }
        : {}),
      hero: { label: 'hero seed', stroke: accent, values: heroColumn(built) },
      fillHeight: true,
      zeroLine: true,
      xRange: () => [0, horizon],
      yUnit: SERIES_MODES[mode].label,
      xUnit: 'steps',
      onLaneClick: (key, step) => {
        const seed = Number(key);
        if (recordedSeeds.has(seed) && selectedEnv !== null) {
          opts.onWatchSeed?.(selectedEnv, seed, Math.round(step));
        } else {
          opts.onSeek?.(Math.round(step), true);
        }
      },
      onSeek: scrubStep,
    });
    reward = built;
    playheadCharts.add(built.chart);
    built.chart.setMarker(playheadStep);
    built.chart.setData(rewardData(built));
    rateHost.replaceChildren(
      eyebrow('Reward rate', el('span', { class: 'slot' }, [
        el('span', {
          class: 'eyebrow-note',
          text: (capped ? `${shown.length}/${series.length} seeds · ` : 'every seed · ') + 'drag to seek',
          title: 'each point is the exact mean reward per step over its bin — IQM and P10–P90 across seeds, the hero lifted in the env accent',
        }),
        seriesSelect,
      ])),
      built.chart.root,
    );
    built.chart.refresh();
  }

  function paint(): void {
    if (destroyed || detail === null || tape === null || selectedEnv === null) return;
    const meta = detail.meta;
    const horizon = meta.steps;
    const t = tape;
    clearCharts();
    status.remove();

    const maxStep = t.rows > 0 ? t.stepEnd[t.rows - 1] ?? 0 : 0;
    bannerSlot.replaceChildren(
      ...(maxStep >= horizon
        ? []
        : [
            el('div', { class: 'banner banner-warn' }, [
              el('strong', { text: 'Incomplete run. ' }),
              el('span', {
                text: `Reached ${formatStep(maxStep)} of ${formatStep(horizon)} declared steps. fleet.py gate refuses these — read it, do not promote on it.`,
              }),
            ]),
          ]),
    );

    const { lanes, seeds } = envLanes(meta, selectedEnv);
    if (lanes.length === 0) {
      rateHost.replaceChildren(emptyState('No seeds for this environment', 'The run schedule assigns it no lanes.'));
      seedsHost.replaceChildren();
      throughputHost.replaceChildren();
      retentionHost.replaceChildren();
      totalsHost.replaceChildren();
      return;
    }
    const recordedSeeds = new Set(
      detail.recordings.filter((r) => r.envId === selectedEnv).map((r) => r.seed),
    );
    const accent = accentCss(envSlug(selectedEnv));

    ensureReward(t, lanes, seeds, horizon, accent, recordedSeeds);

    {
      const surv = survivalSeries(t, lanes, CHART_BINS);
      const diverged = lanes.filter((lane) => laneDeath(t, lane) !== null).length;
      const chart = trackChart(timeSeries({
        steps: [0, ...surv.x],
        lines: [
          { label: 'surviving', values: [1, ...surv.surviving], stroke: '#7fd47a' },
          { label: '> 0', values: [0, ...surv.positive], stroke: '#2b98d9' },
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
          text: `${formatCount(diverged)} div`,
          title: 'surviving = non-diverged fraction · > 0 = fraction with positive cumulative reward',
        })),
        chart.root,
      );
      chart.refresh();
    }

    {
      const tp = throughputSeries(t, CHART_BINS);
      if (tp.x.some((_, i) => tp.stepsPerSecond[i] !== null)) {
        const chart = trackChart(timeSeries({
          steps: tp.x,
          lines: [{ label: 'steps/s', values: tp.stepsPerSecond }],
          fillHeight: true,
          yFormat: (v) => `${formatStep(v)}/s`,
          xRange: () => [0, horizon],
          yUnit: 'agg steps / second',
          onSeek: scrubStep,
        }));
        throughputHost.replaceChildren(
          eyebrow('Throughput', el('span', { class: 'eyebrow-note', text: 'all tasks · agg steps/s' })),
          chart.root,
        );
        chart.refresh();
      } else {
        throughputHost.replaceChildren();
      }
    }

    {
      const retentions: number[] = [];
      for (const lane of lanes) {
        const r = laneRetention(t, lane, horizon);
        if (r !== null) retentions.push(r.retention);
      }
      if (retentions.length > 0) {
        const curve = retentionCurve(retentions);
        const belowZero = retentions.filter((v) => v < 0).length;
        const chart = steppedLine({
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
        trackChart(chart, false);
        retentionHost.replaceChildren(
          eyebrow('Retention', el('span', {
            class: 'eyebrow-note',
            text: belowZero > 0 ? `final ÷ peak · ${formatCount(belowZero)} < 0 off-axis` : 'final ÷ peak',
            title: 'fraction of seeds holding at least x of their peak windowed reward rate — the same law fleet.py applies',
          })),
          chart.root,
        );
        chart.refresh();
      } else {
        retentionHost.replaceChildren(
          eyebrow('Retention'),
          emptyState('No positive peaks', 'No seed of this environment ever held a positive windowed rate.'),
        );
      }
    }

    repaintTotals = () => {
      totalsChart?.destroy();
      totalsChart = null;
      const totals = lanes
        .map((lane, li) => ({
          seed: seeds[li] ?? 0,
          total: (() => {
            const rate = deliveredRate(t, lane);
            const last = t.rows > 0 ? t.stepEnd[t.rows - 1] ?? 0 : 0;
            return rate === null ? null : rate * last;
          })(),
        }))
        .filter((entry): entry is { seed: number; total: number } => entry.total !== null)
        .sort((a, b) => b.total - a.total)
        .slice(0, COLUMN_CAP);
      if (totals.length === 0) {
        totalsHost.replaceChildren();
        return;
      }
      const heroIndex = heroSeed === null ? -1 : totals.findIndex((e) => e.seed === heroSeed);
      totalsChart = columns({
        ...(heroIndex >= 0 ? { highlight: { index: heroIndex, color: accent } } : {}),
        values: totals.map((e) => e.total),
        labels: totals.map((e) =>
          `seed ${e.seed}${recordedSeeds.has(e.seed) ? ' · click to watch' : ''}`),
        fillHeight: true,
        yFormat: (v) => formatSigned(v, 3),
        yUnit: 'total reward',
        onColumnClick: (index) => {
          const entry = totals[index];
          if (entry && recordedSeeds.has(entry.seed) && selectedEnv !== null) {
            opts.onWatchSeed?.(selectedEnv, entry.seed);
          }
        },
      });
      totalsHost.replaceChildren(
        eyebrow('Totals by seed', el('span', {
          class: 'eyebrow-note',
          text: 'click to watch',
          title: 'final cumulative reward of every seed, sorted — the population objective',
        })),
        totalsChart.root,
      );
      totalsChart.refresh();
    };
    repaintTotals();
  }

  Promise.all([loadRunDetail(id), loadTape(id)]).then(
    ([loadedDetail, loadedTape]) => {
      if (destroyed) return;
      detail = loadedDetail;
      tape = loadedTape;
      const shares = envShares(loadedDetail.meta);
      if (selectedEnv === null || !shares.some((sh) => sh.envId === selectedEnv)) {
        selectedEnv = shares[0]?.envId ?? null;
      }
      body.append(bannerSlot, row);
      paint();
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

  return {
    setEnv(envId: string) {
      if (envId === selectedEnv) return;
      selectedEnv = envId;
      paint();
    },
    setTape(next: Tape) {
      if (tape !== null && next.rows === tape.rows) return;
      tape = next;
      paint();
    },
    setPlayhead(step: number | null) {
      playheadStep = step;
      for (const chart of playheadCharts) chart.setMarker(step);
    },
    setHighlight(seed: number | null) {
      if (seed === heroSeed) return;
      heroSeed = seed;
      if (reward !== null) reward.chart.setData(rewardData(reward));
      repaintTotals?.();
    },
    destroy() {
      destroyed = true;
      clearCharts();
      dropReward();
      totalsChart?.destroy();
      totalsChart = null;
      for (const item of disposables) item.destroy();
      body.remove();
    },
  };
}
