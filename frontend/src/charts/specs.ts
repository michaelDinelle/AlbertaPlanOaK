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
  span?: boolean;
}

export function toSeries(line: SeriesLine): uPlot.Series {
  const series: uPlot.Series = {
    label: line.label,
    stroke: line.stroke,
    width: line.width ?? 1.5,
    spanGaps: line.span ?? false,
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

export interface ProfileParams {
  lines: SeriesLine[];
  plugins?: uPlot.Plugin[];
  bands?: uPlot.Band[];
  xFormat?: (value: number) => string;
}

export function profileOptions(params: ProfileParams): OptionsBuilder {
  return timeSeriesOptions({
    lines: params.lines,
    ...(params.plugins ? { plugins: params.plugins } : {}),
    ...(params.bands ? { bands: params.bands } : {}),
    yFormat: (v) => formatPercent(v, 0),
    xFormat: params.xFormat ?? ((v) => formatRate(v, 2)),
    yRange: () => [0, 1.02],
  });
}

export interface FrontierParams {
  lines: SeriesLine[];
  plugins?: uPlot.Plugin[];
  yFormat?: (value: number) => string;
  xRange?: () => [number, number];
  yRange?: (min: number, max: number) => [number, number];
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
      x: {
        time: false,
        distr: 3,
        log: 10,
        ...(params.xRange ? { range: () => params.xRange!() } : {}),
      },
      y: {
        range: params.yRange
          ? (_u, min, max) => params.yRange!(min, max)
          : (_u, min, max) => uPlot.rangeNum(min, max, 0.12, true),
      },
    },
    axes: chartAxes({
      ...(params.yFormat ? { yFormat: params.yFormat } : {}),
      xFormat: (v) => (v > 0 ? v.toExponential(0) : '0'),
    }),
    plugins: params.plugins ?? [],
  });
}
