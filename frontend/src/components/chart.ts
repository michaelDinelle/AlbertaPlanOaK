
import uPlot from 'uplot';
import { el } from './table';
import { formatRate, formatStep } from '../state';

export const VIZ = {
  series: '#2b98d9',
  crit: '#e0614f',
  ppo: '#8b7ce0',
  ramp: ['#22587a', '#2a7cae', '#2f9ade', '#79c9f2'] as const,
  grid: 'rgba(132, 148, 171, 0.13)',
  axis: '#5c6a80',
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
  bars?: boolean;
  scale?: string;
}

export interface ChartSpec {
  height?: number;
  plugins?: uPlot.Plugin[];
  series: readonly SeriesSpec[];
  data: uPlot.AlignedData;
  bands?: readonly { hi: number; lo: number; fill: string }[];
  yFormat?: (value: number) => string;
  xFormat?: (value: number) => string;
  xAxisFormat?: (value: number) => string;
  yUnit?: string;
  xUnit?: string;
  zeroLine?: boolean;
  legend?: boolean;
  tooltip?: 'all' | 'none';
  legendItems?: readonly { label: string; color: string; shape?: 'line' | 'area' }[];
  onSeek?(x: number, phase: 'drag' | 'end' | 'tap'): void;
  fillHeight?: boolean;
  padRight?: number;
  yRange?: (min: number, max: number) => [number, number];
  xRange?: (min: number, max: number) => [number, number];
}

export interface ChartHandle {
  root: HTMLElement;
  setData(data: uPlot.AlignedData): void;
  refresh(): void;
  redraw(): void;
  setMarker(x: number | null): void;
  destroy(): void;
}

function tooltipPlugin(spec: ChartSpec, tip: HTMLElement): uPlot.Plugin {
  const yFormat = spec.yFormat ?? ((v: number) => formatRate(v));
  const xFormat = spec.xFormat ?? formatStep;
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
      ?? spec.series.map((s) => ({ label: s.label, color: s.stroke, shape: 'line' as const }));
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
          try { over.setPointerCapture(ev.pointerId); } catch {  }
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
        if (s.bars) entry.paths = uPlot.paths.bars?.({ size: [1, 14], align: 0 });
        return entry;
      }),
    ];

    const options: uPlot.Options = {
      width,
      height: lastHeight,
      padding: [10, spec.padRight ?? 12, 0, 0],
      cursor: {
        y: false,
        drag: { x: false, y: false, setScale: false },
      },
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
    redraw() {
      plot?.redraw(false, false);
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

export interface SteppedSpec extends Omit<ChartSpec, 'series' | 'data'> {
  steps: readonly number[];
  values: readonly (number | null)[];
  label?: string;
  stroke?: string;
}

export function steppedLine(spec: SteppedSpec): ChartHandle {
  return createChart({
    ...spec,
    series: [{ label: spec.label ?? 'Surviving', stroke: spec.stroke ?? VIZ.series, width: 2, stepped: true }],
    data: [spec.steps as number[], spec.values as (number | null)[]] as unknown as uPlot.AlignedData,
    legend: spec.legend ?? false,
  });
}

export interface BaselineBars {
  ppo: number | null;
  random: number | null;
  active: 'oak' | 'ppo' | 'random';
}

function baselineGutterPlugin(getBars: () => BaselineBars): uPlot.Plugin {
  const bar = (
    u: uPlot,
    value: number | null,
    offset: number,
    fill: string,
    cap: string,
    capWidth: number,
    alpha: number,
  ): void => {
    if (value === null || !Number.isFinite(value)) return;
    const scale = u.scales['y'];
    if (!scale || scale.min === undefined || scale.max === undefined || scale.min === null || scale.max === null) return;
    const x = u.bbox.left + u.bbox.width + offset;
    const base = u.valToPos(Math.max(0, scale.min), 'y', true);
    const top = u.valToPos(value, 'y', true);
    if (!Number.isFinite(base) || !Number.isFinite(top)) return;
    const ctx = u.ctx;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = fill;
    ctx.fillRect(x, Math.min(base, top), 12, Math.max(1, Math.abs(base - top)));
    ctx.strokeStyle = cap;
    ctx.lineWidth = capWidth;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x + 12, top);
    ctx.stroke();
    ctx.restore();
  };
  return {
    hooks: {
      draw: (u) => {
        const bars = getBars();
        const alpha = (p: 'ppo' | 'random'): number => (bars.active === 'oak' || bars.active === p ? 1 : 0.45);
        bar(u, bars.ppo, 8, 'rgba(139, 124, 224, 0.55)', VIZ.ppo, 2, alpha('ppo'));
        bar(u, bars.random, 26, 'rgba(132, 148, 171, 0.5)', '#8494ab', 1, alpha('random'));
      },
    },
  };
}

export interface PopulationChartSpec extends Omit<ChartSpec, 'series' | 'data' | 'bands' | 'tooltip'> {
  steps: readonly number[];
  p90: readonly (number | null)[];
  p10: readonly (number | null)[];
  iqm: readonly (number | null)[];
  hero: readonly (number | null)[];
  getBars(): BaselineBars;
}

export function populationChart(spec: PopulationChartSpec): ChartHandle {
  const spreadFill = 'rgba(43, 152, 217, 0.13)';
  return createChart({
    ...spec,
    padRight: 56,
    plugins: [...(spec.plugins ?? []), baselineGutterPlugin(spec.getBars)],
    series: [
      { label: 'P90', stroke: 'rgba(43,152,217,0.4)', width: 1 },
      { label: 'P10', stroke: 'rgba(43,152,217,0.4)', width: 1 },
      { label: 'IQM', stroke: VIZ.series, width: 2 },
      { label: 'hero seed', stroke: '#ffd166', width: 1.8 },
    ],
    bands: [{ hi: 1, lo: 2, fill: spreadFill }],
    legendItems: [
      { label: 'IQM', color: VIZ.series, shape: 'line' },
      { label: 'P10–P90', color: spreadFill, shape: 'area' },
      { label: 'hero seed', color: '#ffd166', shape: 'line' },
    ],
    data: [
      spec.steps as number[],
      spec.p90 as (number | null)[],
      spec.p10 as (number | null)[],
      spec.iqm as (number | null)[],
      spec.hero as (number | null)[],
    ] as unknown as uPlot.AlignedData,
    legend: true,
    yRange: (min, max) => {
      const bars = spec.getBars();
      let lo = min;
      let hi = max;
      for (const value of [bars.ppo, bars.random]) {
        if (value === null || !Number.isFinite(value)) continue;
        if (value < lo) lo = value;
        if (value > hi) hi = value;
      }
      const range = uPlot.rangeNum(lo, hi, 0.12, true);
      return [range[0] ?? lo, range[1] ?? hi];
    },
  });
}

export interface ColumnsSpec extends Omit<ChartSpec, 'series' | 'data'> {
  values: readonly number[];
  labels?: readonly string[];
  posLabel?: string;
  negLabel?: string;
  onColumnClick?(index: number): void;
  highlight?: { index: number; color: string };
}

export function columns(spec: ColumnsSpec): ChartHandle {
  const labels = spec.labels;
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

