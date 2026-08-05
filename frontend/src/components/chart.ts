// uPlot wrappers, dark-observatory themed.
//
// Every chart here is resize-observing and destroy()able; construction is
// deferred until the container actually has width, so a chart can be built
// before its view is appended to the document.
//
// Colors come from the validated data palette in styles/main.css:
//   series  #2b98d9   critical #e0614f   ordinal ramp #22587a -> #79c9f2
// (checked with the dataviz validator against surface #121722: categorical
// PASS on all five checks; ramp PASS on all four ordinal checks.)

import uPlot from 'uplot';
import { el } from './table';
import { formatRate, formatStep } from '../state';

export const VIZ = {
  series: '#2b98d9',
  seriesSoft: 'rgba(43, 152, 217, 0.16)',
  seriesLight: '#79c9f2',
  crit: '#e0614f',
  critSoft: 'rgba(224, 97, 79, 0.18)',
  ramp: ['#22587a', '#2a7cae', '#2f9ade', '#79c9f2'] as const,
  grid: 'rgba(132, 148, 171, 0.13)',
  axis: '#5c6a80',
  lane: 'rgba(132, 148, 171, 0.26)',
  laneFocus: '#79c9f2',
  zero: 'rgba(215, 223, 235, 0.28)',
} as const;

const AXIS_FONT = '10px ui-monospace, "SF Mono", Menlo, monospace';

export interface SeriesSpec {
  label: string;
  stroke: string;
  width?: number;
  fill?: string;
  dash?: number[];
  stepped?: boolean;
  /** draw as columns rather than a line (per-seed totals) */
  bars?: boolean;
  /** band edges and spaghetti lanes stay out of the legend and tooltip */
  quiet?: boolean;
  scale?: string;
}

export interface ChartSpec {
  height?: number;
  /** extra uPlot plugins prepended to the built-ins */
  plugins?: uPlot.Plugin[];
  series: readonly SeriesSpec[];
  data: uPlot.AlignedData;
  /** 1-based series indices; hi must be the upper edge */
  bands?: readonly { hi: number; lo: number; fill: string }[];
  yFormat?: (value: number) => string;
  xFormat?: (value: number) => string;
  /** axis ticks when they should read differently from the tooltip (rank vs seed) */
  xAxisFormat?: (value: number) => string;
  /** shown under the y axis as a micro-caps unit label */
  yUnit?: string;
  xUnit?: string;
  zeroLine?: boolean;
  legend?: boolean;
  tooltip?: 'all' | 'focus' | 'none';
  /** override the generated legend (a band reads as one region, not two edges) */
  legendItems?: readonly { label: string; color: string; shape?: 'line' | 'area' }[];
  /** enable series focus-on-proximity (spaghetti) */
  focus?: boolean;
  onFocus?(seriesIndex: number | null): void;
  /**
   * Scrub-on-chart: the x axis is the run's own timeline (steps). Press and
   * drag seeks continuously (phase 'drag'), release commits ('end'), and a
   * press without movement is a 'tap'. Charts with a per-item click (spaghetti
   * lanes) compose the tap themselves.
   */
  onSeek?(x: number, phase: 'drag' | 'end' | 'tap'): void;
  /** size the plot to fill the container's height (resizable cells) */
  fillHeight?: boolean;
  yRange?: (min: number, max: number) => [number, number];
  /**
   * Pin the x domain. Only for axes whose meaningful range is a FIXED property of the
   * quantity (a ratio that lives in [0,1]), never to hide inconvenient data — a chart
   * that clips must say so in its own caption.
   */
  xRange?: (min: number, max: number) => [number, number];
}

export interface ChartHandle {
  root: HTMLElement;
  setData(data: uPlot.AlignedData): void;
  /** re-measure after a layout change (tab switch, panel expand) */
  refresh(): void;
  focusSeries(index: number | null): void;
  /** vertical playhead at an x value (null hides it) — the wall's clock cursor */
  setMarker(x: number | null): void;
  destroy(): void;
}

function tooltipPlugin(spec: ChartSpec, tip: HTMLElement): uPlot.Plugin {
  const yFormat = spec.yFormat ?? ((v: number) => formatRate(v));
  const xFormat = spec.xFormat ?? formatStep;
  const mode = spec.tooltip ?? 'all';
  return {
    hooks: {
      setCursor: (u: uPlot) => {
        const { idx, left, top } = u.cursor;
        if (idx === null || idx === undefined || left === undefined || left < 0) {
          tip.hidden = true;
          return;
        }
        const xs = u.data[0];
        const x = xs?.[idx];
        if (x === undefined || x === null) {
          tip.hidden = true;
          return;
        }
        const rows: { value: number; node: HTMLElement }[] = [];
        for (let i = 1; i < u.series.length; i++) {
          const series = u.series[i];
          if (!series || series.show === false) continue;
          const meta = spec.series[i - 1];
          if (meta?.quiet && mode !== 'focus') continue;
          if (mode === 'focus' && !(series as { _focus?: boolean })._focus) continue;
          const value = u.data[i]?.[idx];
          if (value === null || value === undefined) continue;
          rows.push({
            value: Number(value),
            node: el('div', { class: 'tip-row' }, [
              el('span', { class: 'tip-swatch', style: { background: String(series.stroke ?? VIZ.series) } }),
              el('span', { class: 'tip-name', text: typeof series.label === 'string' ? series.label : meta?.label ?? `s${i}` }),
              el('span', { class: 'tip-value', text: yFormat(Number(value)) }),
            ]),
          });
        }
        if (rows.length === 0) {
          tip.hidden = true;
          return;
        }
        // read top-to-bottom in the same order the marks stack on the canvas
        rows.sort((a, b) => b.value - a.value);
        tip.replaceChildren(
          el('div', { class: 'tip-head', text: `${xFormat(Number(x))}${spec.xUnit ? ` ${spec.xUnit}` : ''}` }),
          ...rows.map((row) => row.node),
        );
        tip.hidden = false;
        const width = tip.offsetWidth;
        const flip = left + width + 24 > u.width;
        tip.style.transform = `translate(${Math.round(flip ? left - width - 14 : left + 14)}px, ${Math.round(Math.max(4, (top ?? 0) - 12))}px)`;
      },
    },
  };
}

function zeroLinePlugin(): uPlot.Plugin {
  return {
    hooks: {
      draw: (u: uPlot) => {
        const scale = u.scales['y'];
        if (!scale || scale.min === undefined || scale.max === undefined) return;
        if (scale.min > 0 || scale.max < 0) return;
        const y = Math.round(u.valToPos(0, 'y', true)) + 0.5;
        const ctx = u.ctx;
        ctx.save();
        ctx.strokeStyle = VIZ.zero;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(u.bbox.left, y);
        ctx.lineTo(u.bbox.left + u.bbox.width, y);
        ctx.stroke();
        ctx.restore();
      },
    },
  };
}

export function createChart(spec: ChartSpec): ChartHandle {
  const height = spec.height ?? 200;
  const holder = el('div', { class: 'chart-canvas' });
  const tip = el('div', { class: 'chart-tip' });
  tip.hidden = true;
  const legend = spec.legend === false ? null : el('div', { class: 'chart-legend' });
  const unit = spec.yUnit ? el('div', { class: 'chart-unit', text: spec.yUnit }) : null;
  const root = el('div', { class: spec.fillHeight ? 'chart chart-fill' : 'chart' }, [unit, holder, tip, legend]);

  if (legend) {
    const items = spec.legendItems
      ?? spec.series.filter((s) => !s.quiet).map((s) => ({ label: s.label, color: s.stroke, shape: 'line' as const }));
    for (const item of items) {
      legend.append(
        el('span', { class: 'legend-item' }, [
          el('span', {
            class: `legend-swatch legend-swatch-${item.shape ?? 'line'}`,
            style: { background: item.color },
          }),
          el('span', { text: item.label }),
        ]),
      );
    }
    if (legend.childElementCount === 0) legend.remove();
  }

  let plot: uPlot | null = null;
  let data = spec.data;
  let lastWidth = 0;
  let lastHeight = height;
  let frame = 0;
  let destroyed = false;
  let marker: number | null = null;

  /** fill mode: the plot's height is whatever the cell leaves after the chrome */
  function targetHeight(): number {
    if (!spec.fillHeight) return height;
    let chrome = 0;
    for (const child of Array.from(root.children)) {
      if (child === holder || child === tip) continue;
      chrome += (child as HTMLElement).offsetHeight;
    }
    return Math.max(64, root.clientHeight - chrome - 2);
  }

  const seekPlugin = (onSeek: (x: number, phase: 'drag' | 'end' | 'tap') => void): uPlot.Plugin => ({
    hooks: {
      init: (u) => {
        const over = u.over;
        over.style.cursor = 'crosshair';
        let down = false;
        let moved = false;
        let startX = 0;
        const xAt = (ev: PointerEvent): number | null => {
          const rect = over.getBoundingClientRect();
          const x = u.posToVal(ev.clientX - rect.left, 'x');
          return Number.isFinite(x) ? x : null;
        };
        over.addEventListener('pointerdown', (ev) => {
          if (ev.button !== 0) return;
          down = true;
          moved = false;
          startX = ev.clientX;
          try { over.setPointerCapture(ev.pointerId); } catch { /* inactive pointer */ }
        });
        over.addEventListener('pointermove', (ev) => {
          if (!down) return;
          if (Math.abs(ev.clientX - startX) > 3) moved = true;
          if (moved) {
            const x = xAt(ev);
            if (x !== null) onSeek(x, 'drag');
          }
        });
        const finish = (ev: PointerEvent): void => {
          if (!down) return;
          down = false;
          const x = xAt(ev);
          if (x !== null) onSeek(x, moved ? 'end' : 'tap');
        };
        over.addEventListener('pointerup', finish);
        over.addEventListener('pointercancel', () => { down = false; });
      },
    },
  });

  const markerPlugin = (): uPlot.Plugin => ({
    hooks: {
      draw: (u) => {
        if (marker === null) return;
        const x = u.valToPos(marker, 'x', true);
        const { top, height: h, left, width: w } = u.bbox;
        // never draw outside the plot box (a step-0 marker on a chart whose
        // data starts later used to hang half-clipped over the y axis)
        if (!Number.isFinite(x) || x < left || x > left + w) return;
        u.ctx.save();
        u.ctx.strokeStyle = 'rgba(255, 209, 102, 0.9)';
        u.ctx.lineWidth = 1;
        u.ctx.beginPath();
        u.ctx.moveTo(x, top);
        u.ctx.lineTo(x, top + h);
        u.ctx.stroke();
        u.ctx.restore();
      },
    },
  });

  const yFormat = spec.yFormat ?? ((v: number) => formatRate(v));
  const xFormat = spec.xFormat ?? formatStep;

  function build(width: number): void {
    const plugins: uPlot.Plugin[] = [...(spec.plugins ?? [])];
    if (spec.tooltip !== 'none') plugins.push(tooltipPlugin(spec, tip));
    if (spec.zeroLine) plugins.push(zeroLinePlugin());
    if (spec.onSeek) plugins.push(seekPlugin(spec.onSeek));
    plugins.push(markerPlugin());

    const series: uPlot.Series[] = [
      { label: 'x' },
      ...spec.series.map((s): uPlot.Series => {
        const entry: uPlot.Series = {
          label: s.label,
          stroke: s.stroke,
          width: s.width ?? 1.5,
          spanGaps: false,
          points: { show: false },
          scale: s.scale ?? 'y',
        };
        if (s.fill) entry.fill = s.fill;
        if (s.dash) entry.dash = s.dash;
        if (s.stepped) entry.paths = uPlot.paths.stepped?.({ align: 1 });
        // size: [factor, maxWidth] — factor 1 so a 250-column profile reads as a
        // continuous distribution, capped so a 6-column one does not read as a blob.
        if (s.bars) entry.paths = uPlot.paths.bars?.({ size: [1, 14], align: 0 });
        return entry;
      }),
    ];

    const options: uPlot.Options = {
      width,
      height: lastHeight,
      padding: [10, 12, 0, 0],
      cursor: {
        y: false,
        drag: { x: false, y: false, setScale: false },
        ...(spec.focus ? { focus: { prox: 20 } } : {}),
      },
      ...(spec.focus ? { focus: { alpha: 0.16 } } : {}),
      legend: { show: false },
      series,
      bands: spec.bands?.map((b) => ({ series: [b.hi, b.lo] as [number, number], fill: b.fill })),
      scales: {
        x: {
          time: false,
          ...(spec.xRange ? { range: (_u, min, max) => spec.xRange!(min, max) } : {}),
        },
        y: {
          range: spec.yRange
            ? (_u, min, max) => spec.yRange!(min, max)
            : (_u, min, max) => uPlot.rangeNum(min, max, 0.12, true),
        },
      },
      axes: [
        {
          stroke: VIZ.axis,
          font: AXIS_FONT,
          grid: { stroke: VIZ.grid, width: 1 },
          ticks: { show: false },
          size: 26,
          values: (_u, splits) => splits.map((v) => (spec.xAxisFormat ?? xFormat)(v)),
        },
        {
          stroke: VIZ.axis,
          font: AXIS_FONT,
          grid: { stroke: VIZ.grid, width: 1 },
          ticks: { show: false },
          size: 52,
          values: (_u, splits) => splits.map((v) => yFormat(v)),
        },
      ],
      plugins,
      hooks: spec.onFocus
        ? {
            setSeries: [
              (u: uPlot, index: number | null) => {
                for (let i = 1; i < u.series.length; i++) {
                  (u.series[i] as { _focus?: boolean })._focus = i === index;
                }
                spec.onFocus?.(index === null ? null : index - 1);
              },
            ],
          }
        : {},
    };

    plot = new uPlot(options, data, holder);
  }

  function measure(): void {
    if (destroyed) return;
    const width = Math.max(0, Math.floor(root.clientWidth));
    if (width < 8) return;
    const h = targetHeight();
    if (!plot) {
      lastWidth = width;
      lastHeight = h;
      build(width);
      return;
    }
    if (width !== lastWidth || h !== lastHeight) {
      lastWidth = width;
      lastHeight = h;
      plot.setSize({ width, height: h });
    }
  }

  const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(measure);
  });
  observer?.observe(root);
  queueMicrotask(measure);

  root.addEventListener('mouseleave', () => {
    tip.hidden = true;
  });

  return {
    root,
    setMarker(x) {
      marker = x;
      plot?.redraw(false, false);
    },
    setData(next) {
      data = next;
      if (plot) plot.setData(next);
      else measure();
    },
    refresh: measure,
    focusSeries(index) {
      plot?.setSeries(index === null ? null : index + 1, { focus: true });
    },
    destroy() {
      destroyed = true;
      cancelAnimationFrame(frame);
      observer?.disconnect();
      plot?.destroy();
      plot = null;
      root.remove();
    },
  };
}

// ---------- named forms ----------

export interface TimeSeriesSpec extends Omit<ChartSpec, 'series' | 'data'> {
  steps: readonly number[];
  lines: readonly { label: string; values: readonly (number | null)[]; stroke?: string; fill?: string; width?: number }[];
}

export function timeSeries(spec: TimeSeriesSpec): ChartHandle {
  const strokes = [VIZ.series, VIZ.crit, VIZ.ramp[3], VIZ.ramp[1]];
  return createChart({
    ...spec,
    series: spec.lines.map((line, i) => ({
      label: line.label,
      stroke: line.stroke ?? strokes[i % strokes.length] ?? VIZ.series,
      ...(line.fill ? { fill: line.fill } : {}),
      width: line.width ?? 1.75,
    })),
    data: [
      spec.steps as number[],
      ...spec.lines.map((line) => line.values as (number | null)[]),
    ] as unknown as uPlot.AlignedData,
  });
}

export interface BandSpec extends Omit<ChartSpec, 'series' | 'data' | 'bands'> {
  steps: readonly number[];
  p10: readonly (number | null)[];
  p50: readonly (number | null)[];
  p90: readonly (number | null)[];
  labels?: { low: string; mid: string; high: string };
  stroke?: string;
  fill?: string;
  /** overlay series drawn on top of the band (e.g. the hero seed's own curve) */
  extra?: { label: string; values: readonly (number | null)[]; stroke: string; width?: number }[];
  /** optional inner 95% CI band around the mid line */
  ci?: { lo: readonly (number | null)[]; hi: readonly (number | null)[] };
}

/** p10/p90 fill with the p50 line on top — the population headline chart. */
export function band(spec: BandSpec): ChartHandle {
  const labels = spec.labels ?? { low: 'P10', mid: 'Median', high: 'P90' };
  const stroke = spec.stroke ?? VIZ.series;
  const fill = spec.fill ?? VIZ.seriesSoft;
  return createChart({
    ...spec,
    // all three are readable in the tooltip; the legend names the band as one
    // region, because two edge entries would imply two independent series
    series: [
      { label: labels.high, stroke: 'rgba(43,152,217,0.45)', width: 1 },
      { label: labels.low, stroke: 'rgba(43,152,217,0.45)', width: 1 },
      ...(spec.ci
        ? [
            { label: 'CI hi', stroke: 'rgba(43,152,217,0.35)', width: 0.5 },
            { label: 'CI lo', stroke: 'rgba(43,152,217,0.35)', width: 0.5 },
          ]
        : []),
      { label: labels.mid, stroke, width: 2 },
      ...(spec.extra ?? []).map((x) => ({ label: x.label, stroke: x.stroke, width: x.width ?? 1.5 })),
    ],
    legendItems: [
      { label: `${labels.low}–${labels.high}`, color: fill, shape: 'area' },
      ...(spec.ci ? [{ label: '95% CI', color: 'rgba(43,152,217,0.34)', shape: 'area' as const }] : []),
      { label: labels.mid, color: stroke, shape: 'line' },
      ...(spec.extra ?? []).map((x) => ({ label: x.label, color: x.stroke, shape: 'line' as const })),
    ],
    bands: spec.ci
      ? [
          { hi: 1, lo: 2, fill },
          { hi: 3, lo: 4, fill: 'rgba(43,152,217,0.24)' },
        ]
      : [{ hi: 1, lo: 2, fill }],
    data: [
      spec.steps as number[],
      spec.p90 as (number | null)[],
      spec.p10 as (number | null)[],
      ...(spec.ci ? [spec.ci.hi as (number | null)[], spec.ci.lo as (number | null)[]] : []),
      spec.p50 as (number | null)[],
      ...(spec.extra ?? []).map((x) => x.values as (number | null)[]),
    ] as unknown as uPlot.AlignedData,
    legend: spec.legend ?? true,
  });
}

export interface SteppedSpec extends Omit<ChartSpec, 'series' | 'data'> {
  steps: readonly number[];
  values: readonly (number | null)[];
  label?: string;
  stroke?: string;
}

/** Kaplan-Meier survival: a step function, never interpolated between deaths. */
export function steppedLine(spec: SteppedSpec): ChartHandle {
  return createChart({
    ...spec,
    series: [{ label: spec.label ?? 'Surviving', stroke: spec.stroke ?? VIZ.series, width: 2, stepped: true }],
    data: [spec.steps as number[], spec.values as (number | null)[]] as unknown as uPlot.AlignedData,
    legend: spec.legend ?? false,
  });
}

export interface PopulationBand {
  p10: readonly (number | null)[];
  p90: readonly (number | null)[];
  iqm: readonly (number | null)[];
  ciLo?: readonly (number | null)[];
  ciHi?: readonly (number | null)[];
}

export interface PopulationChartSpec extends Omit<ChartSpec, 'series' | 'data' | 'bands' | 'tooltip'> {
  steps: readonly number[];
  /** the population statistics drawn OVER the lanes; omit when they don't share the x grid */
  band?: PopulationBand;
  lanes: readonly SpaghettiLane[];
  /** the hero seed's lane, lifted in the env accent */
  highlightKey?: string | null;
  highlightStroke?: string;
  onLaneFocus?(key: string | null): void;
  /** tap on a focused lane — watch that seed, at the tapped step */
  onLaneClick?(key: string, step: number): void;
}

/**
 * THE population chart: every seed's own curve ghosted under the population
 * statistics (P10–P90 spread, bootstrap 95% CI, IQM), with the hero seed lifted
 * in its env accent. One tile carries what used to be two — the RL-paper figure
 * and the per-seed spaghetti are the same picture at two alphas.
 */
export function populationChart(spec: PopulationChartSpec): ChartHandle {
  const b = spec.band;
  const hasCi = b?.ciLo !== undefined && b?.ciHi !== undefined;
  const laneOffset = b ? (hasCi ? 5 : 3) : 0;
  const spreadFill = 'rgba(43, 152, 217, 0.13)';
  const ciFill = 'rgba(43, 152, 217, 0.26)';

  const bandSeries: SeriesSpec[] = b
    ? [
        { label: 'P90', stroke: 'rgba(43,152,217,0.4)', width: 1 },
        { label: 'P10', stroke: 'rgba(43,152,217,0.4)', width: 1 },
        ...(hasCi
          ? [
              { label: 'CI hi', stroke: 'rgba(43,152,217,0.3)', width: 0.5 },
              { label: 'CI lo', stroke: 'rgba(43,152,217,0.3)', width: 0.5 },
            ]
          : []),
        { label: 'IQM', stroke: VIZ.series, width: 2 },
      ]
    : [];
  const bandData: (number | null)[][] = b
    ? [
        b.p90 as (number | null)[],
        b.p10 as (number | null)[],
        ...(hasCi ? [b.ciHi as (number | null)[], b.ciLo as (number | null)[]] : []),
        b.iqm as (number | null)[],
      ]
    : [];

  let focused: string | null = null;
  const handle = createChart({
    ...spec,
    tooltip: 'focus',
    focus: true,
    series: [
      ...bandSeries,
      ...spec.lanes.map((lane): SeriesSpec => (
        lane.key === spec.highlightKey
          ? { label: lane.label, stroke: spec.highlightStroke ?? '#ffd166', width: 1.8 }
          : { label: lane.label, stroke: VIZ.lane, width: 1, quiet: true }
      )),
    ],
    bands: b
      ? [
          { hi: 1, lo: 2, fill: spreadFill },
          ...(hasCi ? [{ hi: 3, lo: 4, fill: ciFill }] : []),
        ]
      : [],
    legendItems: [
      ...(b
        ? [
            { label: 'IQM', color: VIZ.series, shape: 'line' as const },
            ...(hasCi ? [{ label: '95% CI', color: ciFill, shape: 'area' as const }] : []),
            { label: 'P10–P90', color: spreadFill, shape: 'area' as const },
          ]
        : []),
      ...(spec.highlightKey
        ? [{ label: `seed ${spec.highlightKey}`, color: spec.highlightStroke ?? '#ffd166', shape: 'line' as const }]
        : []),
    ],
    data: [
      spec.steps as number[],
      ...bandData,
      ...spec.lanes.map((lane) => lane.values as (number | null)[]),
    ] as unknown as uPlot.AlignedData,
    legend: true,
    onFocus: (index) => {
      // band series are not lanes; focus semantics belong to seeds only
      focused = index === null || index < laneOffset ? null : spec.lanes[index - laneOffset]?.key ?? null;
      spec.onLaneFocus?.(focused);
      spec.onFocus?.(index);
    },
    onSeek: (x, phase) => {
      if (phase === 'tap' && focused !== null && spec.onLaneClick) spec.onLaneClick(focused, x);
      else spec.onSeek?.(x, phase);
    },
  });
  return handle;
}

export interface ColumnsSpec extends Omit<ChartSpec, 'series' | 'data'> {
  /** already in the order they should be drawn (rank order, usually) */
  values: readonly number[];
  /** per-column name for the tooltip header, e.g. "seed 312" */
  labels?: readonly string[];
  posLabel?: string;
  negLabel?: string;
  /** click a column (hovered index) — the wall uses this for click-to-watch */
  onColumnClick?(index: number): void;
  /** the hero seed's column, restroked in the env accent */
  highlight?: { index: number; color: string };
}

/**
 * A sorted column profile — per-seed totals, one bar per seed.
 *
 * Split into two series by SIGN rather than tinted per bar: "which seeds paid and which
 * cost" is the whole question a per-seed total answers, and a diverging pair states it
 * without a legend lookup. The x axis is rank, so its tick labels are suppressed and
 * the tooltip names the seed instead.
 */
export function columns(spec: ColumnsSpec): ChartHandle {
  const labels = spec.labels;
  // restroke the hero's column in the env accent, over the series paint
  const highlightPlugin = (): uPlot.Plugin => ({
    hooks: {
      draw: (u) => {
        const hi = spec.highlight;
        if (!hi || hi.index < 0 || hi.index >= spec.values.length) return;
        const value = spec.values[hi.index] ?? 0;
        const x = u.valToPos(hi.index, 'x', true);
        const y0 = u.valToPos(0, 'y', true);
        const y1 = u.valToPos(value, 'y', true);
        if (!Number.isFinite(x) || !Number.isFinite(y1)) return;
        const step = spec.values.length > 1 ? u.bbox.width / spec.values.length : u.bbox.width;
        const half = Math.min(7, Math.max(1, step * 0.5));
        u.ctx.save();
        u.ctx.fillStyle = hi.color;
        u.ctx.fillRect(x - half, Math.min(y0, y1), half * 2, Math.abs(y0 - y1));
        u.ctx.restore();
      },
    },
  });
  const clickPlugin = (): uPlot.Plugin => ({
    hooks: {
      ready: (u) => {
        u.over.style.cursor = spec.onColumnClick ? 'pointer' : '';
        u.over.addEventListener('click', () => {
          const idx = u.cursor.idx;
          if (idx !== null && idx !== undefined && spec.onColumnClick) spec.onColumnClick(idx);
        });
      },
    },
  });
  const positive = spec.values.map((v) => (v >= 0 ? v : null));
  const negative = spec.values.map((v) => (v < 0 ? v : null));
  const hasPositive = positive.some((v) => v !== null);
  const hasNegative = negative.some((v) => v !== null);
  return createChart({
    ...spec,
    plugins: [clickPlugin(), highlightPlugin()],
    series: [
      { label: spec.posLabel ?? 'above zero', stroke: VIZ.series, fill: VIZ.series, bars: true, width: 1 },
      { label: spec.negLabel ?? 'below zero', stroke: VIZ.crit, fill: VIZ.crit, bars: true, width: 1 },
    ],
    legendItems: [
      ...(hasPositive ? [{ label: spec.posLabel ?? 'above zero', color: VIZ.series, shape: 'area' as const }] : []),
      ...(hasNegative ? [{ label: spec.negLabel ?? 'below zero', color: VIZ.crit, shape: 'area' as const }] : []),
    ],
    data: [
      spec.values.map((_, i) => i),
      positive,
      negative,
    ] as unknown as uPlot.AlignedData,
    // rank is an ordering, not a quantity: the axis counts seeds, the tooltip names one.
    // Only INTEGER splits get a label. uPlot picks tick positions on a continuous scale,
    // so a short profile (8 seeds) lands ticks at 0, ⅓, ⅔, 1 … and rounding each one to
    // a rank printed "#1 #1 #2 #2 #2 #3 …" — a real axis with a stutter. Blanking the
    // fractional splits leaves each rank labelled exactly once.
    xFormat: spec.xFormat ?? ((v: number) => labels?.[Math.round(v)] ?? `#${Math.round(v) + 1}`),
    xAxisFormat:
      spec.xAxisFormat
      ?? ((v: number) => (Number.isInteger(v) && v >= 0 && v < spec.values.length ? `#${v + 1}` : '')),
    yRange: spec.yRange ?? ((min, max) => {
      const lo = Math.min(0, min);
      const hi = Math.max(0, max);
      const pad = (hi - lo) * 0.06 || 1;
      return [lo - (lo < 0 ? pad : 0), hi + (hi > 0 ? pad : 0)];
    }),
    zeroLine: spec.zeroLine ?? true,
    legend: spec.legend ?? true,
  });
}

export interface SpaghettiLane {
  key: string;
  label: string;
  values: readonly (number | null)[];
}

export interface SpaghettiSpec extends Omit<ChartSpec, 'series' | 'data' | 'tooltip'> {
  steps: readonly number[];
  lanes: readonly SpaghettiLane[];
  onLaneFocus?(key: string | null): void;
  /** click the focused (hovered) lane — watch that seed, at the clicked step */
  onLaneClick?(key: string, step: number): void;
}

/**
 * Per-lane spaghetti. Lanes are neutral low-alpha strokes — never generated
 * hues — with the hovered lane lifted to the light ramp step and named.
 */
export function spaghetti(spec: SpaghettiSpec): ChartHandle {
  let focused: string | null = null;
  const handle = createChart({
    ...spec,
    tooltip: 'focus',
    focus: true,
    series: spec.lanes.map((lane) => ({ label: lane.label, stroke: VIZ.lane, width: 1, quiet: true })),
    data: [
      spec.steps as number[],
      ...spec.lanes.map((lane) => lane.values as (number | null)[]),
    ] as unknown as uPlot.AlignedData,
    legend: false,
    onFocus: (index) => {
      focused = index === null ? null : spec.lanes[index]?.key ?? null;
      spec.onLaneFocus?.(focused);
      spec.onFocus?.(index);
    },
    onSeek: (x, phase) => {
      if (phase === 'tap' && focused !== null && spec.onLaneClick) spec.onLaneClick(focused, x);
      else spec.onSeek?.(x, phase);
    },
  });
  return handle;
}
