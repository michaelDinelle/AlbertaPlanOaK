import uPlot from 'uplot';
import { formatPercent, formatRate, formatStep } from '../lib/format';
import { baselineGutterPlugin, cssVar, monoFont, withAlpha, type BaselineBars } from './plugins';

export interface Size {
  width: number;
  height: number;
}

export type OptionsBuilder = (size: Size) => uPlot.Options;

export interface SeriesLine {
  label: string;
  stroke: string;
  width?: number;
  fill?: string;
  dash?: number[];
  stepped?: boolean;
  scale?: string;
}

function toSeries(line: SeriesLine): uPlot.Series {
  const series: uPlot.Series = {
    label: line.label,
    stroke: line.stroke,
    width: line.width ?? 1.5,
    spanGaps: false,
    points: { show: false },
    scale: line.scale ?? 'y',
  };
  if (line.fill !== undefined) series.fill = line.fill;
  if (line.dash !== undefined) series.dash = line.dash;
  if (line.stepped === true) series.paths = uPlot.paths.stepped?.({ align: 1 });
  return series;
}

interface AxesParams {
  yFormat?: (value: number) => string;
  xFormat?: (value: number) => string;
}

export function chartAxes(params: AxesParams = {}): uPlot.Axis[] {
  const stroke = cssVar('--color-ink-faint');
  const grid = withAlpha(cssVar('--color-ink-dim'), 0.13);
  const font = monoFont(10);
  const yFormat = params.yFormat ?? ((v: number) => formatRate(v));
  const xFormat = params.xFormat ?? formatStep;
  return [
    {
      stroke,
      font,
      grid: { stroke: grid, width: 1 },
      ticks: { show: false },
      size: 26,
      values: (_u, splits) => splits.map((v) => xFormat(v)),
    },
    {
      stroke,
      font,
      grid: { stroke: grid, width: 1 },
      ticks: { show: false },
      size: 52,
      values: (_u, splits) => splits.map((v) => yFormat(v)),
    },
  ];
}

function quietCursor(): uPlot.Cursor {
  return { y: false, drag: { x: false, y: false, setScale: false } };
}

const padDefault: uPlot.Padding = [10, 12, 0, 0];

export interface TimeSeriesParams extends AxesParams {
  lines: SeriesLine[];
  plugins?: uPlot.Plugin[];
  bands?: uPlot.Band[];
  yRange?: (min: number, max: number) => [number, number];
}

export function timeSeriesOptions(params: TimeSeriesParams): OptionsBuilder {
  return (size) => ({
    width: size.width,
    height: size.height,
    padding: padDefault,
    cursor: quietCursor(),
    legend: { show: false },
    series: [{ label: 'x' }, ...params.lines.map(toSeries)],
    ...(params.bands ? { bands: params.bands } : {}),
    scales: {
      x: { time: false },
      y: {
        range: params.yRange
          ? (_u, min, max) => params.yRange!(min, max)
          : (_u, min, max) => uPlot.rangeNum(min, max, 0.12, true),
      },
    },
    axes: chartAxes(params),
    plugins: params.plugins ?? [],
  });
}

export interface SteppedParams extends AxesParams {
  label: string;
  stroke: string;
  plugins?: uPlot.Plugin[];
}

export function steppedOptions(params: SteppedParams): OptionsBuilder {
  return timeSeriesOptions({
    ...params,
    lines: [{ label: params.label, stroke: params.stroke, width: 2, stepped: true }],
  });
}

export interface PopulationParams extends AxesParams {
  getBars(): BaselineBars;
  plugins?: uPlot.Plugin[];
}

export function populationOptions(params: PopulationParams): OptionsBuilder {
  const oak = cssVar('--color-oak');
  const seed = cssVar('--color-seed');
  const edge = withAlpha(oak, 0.4);
  const spread = withAlpha(oak, 0.13);
  return (size) => ({
    width: size.width,
    height: size.height,
    padding: [10, 56, 0, 0],
    cursor: quietCursor(),
    legend: { show: false },
    series: [
      { label: 'x' },
      toSeries({ label: 'P90', stroke: edge, width: 1 }),
      toSeries({ label: 'P10', stroke: edge, width: 1 }),
      toSeries({ label: 'IQM', stroke: oak, width: 2 }),
      toSeries({ label: 'current seed', stroke: seed, width: 1.8 }),
    ],
    bands: [{ series: [1, 2], fill: spread }],
    scales: {
      x: { time: false },
      y: {
        range: (_u, min, max) => {
          const bars = params.getBars();
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
      },
    },
    axes: chartAxes(params),
    plugins: [baselineGutterPlugin(params.getBars), ...(params.plugins ?? [])],
  });
}

const FOREST_COLUMNS = [
  { value: 1, lo: 2, hi: 3, token: '--color-oak', shift: -6 },
  { value: 4, lo: 5, hi: 6, token: '--color-ppo', shift: 0 },
  { value: 7, lo: 8, hi: 9, token: '--color-random', shift: 6 },
] as const;

function forestPlugin(): uPlot.Plugin {
  return {
    hooks: {
      draw: (u) => {
        const xs = u.data[0];
        if (!xs) return;
        const pr = devicePixelRatio;
        const ctx = u.ctx;
        ctx.save();
        for (const column of FOREST_COLUMNS) {
          const color = cssVar(column.token);
          const values = u.data[column.value];
          const lows = u.data[column.lo];
          const highs = u.data[column.hi];
          if (!values) continue;
          ctx.strokeStyle = color;
          ctx.fillStyle = color;
          ctx.lineWidth = pr;
          for (let i = 0; i < xs.length; i++) {
            const value = values[i];
            if (value === null || value === undefined) continue;
            const x = u.valToPos(xs[i] ?? i, 'x', true) + column.shift * pr;
            const lo = lows?.[i];
            const hi = highs?.[i];
            if (lo !== null && lo !== undefined && hi !== null && hi !== undefined) {
              const yLo = u.valToPos(lo, 'y', true);
              const yHi = u.valToPos(hi, 'y', true);
              ctx.beginPath();
              ctx.moveTo(x, yLo);
              ctx.lineTo(x, yHi);
              ctx.stroke();
            }
            const y = u.valToPos(value, 'y', true);
            ctx.beginPath();
            ctx.arc(x, y, 2.5 * pr, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        ctx.restore();
      },
    },
  };
}

export interface ForestParams {
  envLabels: readonly string[];
  yFormat?: (value: number) => string;
  plugins?: uPlot.Plugin[];
}

export function forestOptions(params: ForestParams): OptionsBuilder {
  const labels = params.envLabels;
  const carrier = (label: string, token: string): uPlot.Series => ({
    label,
    stroke: cssVar(token),
    paths: () => null,
    points: { show: false },
  });
  const hidden = (label: string): uPlot.Series => ({ label, show: false, points: { show: false } });
  return (size) => ({
    width: size.width,
    height: size.height,
    padding: padDefault,
    cursor: quietCursor(),
    legend: { show: false },
    series: [
      { label: 'env' },
      carrier('oak', '--color-oak'),
      hidden('oak lo'),
      hidden('oak hi'),
      carrier('ppo', '--color-ppo'),
      hidden('ppo lo'),
      hidden('ppo hi'),
      carrier('random', '--color-random'),
      hidden('random lo'),
      hidden('random hi'),
    ],
    scales: {
      x: { time: false, range: () => [-0.5, labels.length - 0.5] },
      y: { range: (_u, min, max) => uPlot.rangeNum(min, max, 0.12, true) },
    },
    axes: chartAxes({
      ...(params.yFormat ? { yFormat: params.yFormat } : {}),
      xFormat: (v) => (Number.isInteger(v) ? labels[v] ?? '' : ''),
    }),
    plugins: [forestPlugin(), ...(params.plugins ?? [])],
  });
}

export interface ProfileParams {
  lines: SeriesLine[];
  plugins?: uPlot.Plugin[];
  xFormat?: (value: number) => string;
}

export function profileOptions(params: ProfileParams): OptionsBuilder {
  return timeSeriesOptions({
    lines: params.lines,
    ...(params.plugins ? { plugins: params.plugins } : {}),
    yFormat: (v) => formatPercent(v, 0),
    xFormat: params.xFormat ?? ((v) => formatRate(v, 2)),
    yRange: () => [0, 1],
  });
}

export interface FrontierParams {
  lines: SeriesLine[];
  plugins?: uPlot.Plugin[];
  yFormat?: (value: number) => string;
}

export function frontierOptions(params: FrontierParams): OptionsBuilder {
  return (size) => ({
    width: size.width,
    height: size.height,
    padding: padDefault,
    cursor: quietCursor(),
    legend: { show: false },
    series: [{ label: 'flops' }, ...params.lines.map(toSeries)],
    scales: {
      x: { time: false, distr: 3, log: 10 },
      y: { range: (_u, min, max) => uPlot.rangeNum(min, max, 0.12, true) },
    },
    axes: chartAxes({
      ...(params.yFormat ? { yFormat: params.yFormat } : {}),
      xFormat: (v) => (v > 0 ? v.toExponential(0) : '0'),
    }),
    plugins: params.plugins ?? [],
  });
}
