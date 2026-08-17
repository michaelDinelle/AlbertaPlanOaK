import { useEffect, useRef, type ReactNode } from 'react';
import { CartesianGrid, ComposedChart, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import { formatRate, formatStep } from '../lib/format';
import { subscribePosition, throttled } from '../playback/store';

export const CHART_MARGIN = { top: 6, right: 10, bottom: 0, left: 0 } as const;
export const AXIS_WIDTH = 42;
export const X_AXIS_HEIGHT = 18;

export interface XY {
  x: number;
  y: number | null;
}

export interface XBand {
  x: number;
  band: [number | null, number | null];
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function withAlpha(color: string, alpha: number): string {
  if (color.endsWith(')') && !color.includes('/') && !color.includes(',')) {
    return `${color.slice(0, -1)} / ${alpha})`;
  }
  return color;
}

export function seriesRows(x: readonly number[], y: readonly (number | null | undefined)[]): XY[] {
  const out: XY[] = [];
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i++) out.push({ x: x[i] ?? 0, y: Number.isFinite(y[i] ?? NaN) ? (y[i] as number) : null });
  return out;
}

export function bandRows(
  x: readonly number[],
  lo: readonly (number | null)[],
  hi: readonly (number | null)[],
): XBand[] {
  const out: XBand[] = [];
  const n = Math.min(x.length, lo.length, hi.length);
  for (let i = 0; i < n; i++) out.push({ x: x[i] ?? 0, band: [lo[i] ?? null, hi[i] ?? null] });
  return out;
}

export function lineProps(stroke: string, width = 1.4, dash?: string, axis: 'y' | 'r' = 'y') {
  return {
    dataKey: 'y',
    type: 'linear',
    stroke,
    strokeWidth: width,
    strokeDasharray: dash,
    dot: false,
    activeDot: false,
    isAnimationActive: false,
    yAxisId: axis,
  } as const;
}

export function bandProps(fill: string, opacity = 0.12) {
  return {
    dataKey: 'band',
    stroke: 'none',
    fill,
    fillOpacity: opacity,
    isAnimationActive: false,
    yAxisId: 'y',
  } as const;
}

const TICK = { fill: 'var(--color-ink-faint)', fontSize: 10, fontFamily: 'var(--font-mono)' } as const;

export function TimeChart({
  height,
  xMin = 0,
  xMax,
  xScale,
  xFormat,
  yFormat,
  rightFormat,
  overlay,
  children,
}: {
  height: number;
  xMin?: number;
  xMax: number;
  xScale?: 'log';
  xFormat?: (value: number) => string;
  yFormat?: (value: number) => string;
  rightFormat?: (value: number) => string;
  overlay?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="relative" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart margin={CHART_MARGIN}>
          <CartesianGrid stroke="var(--color-line)" strokeOpacity={0.45} vertical={false} />
          <XAxis
            dataKey="x"
            type="number"
            scale={xScale ?? 'linear'}
            domain={[xMin, xMax]}
            allowDataOverflow
            height={X_AXIS_HEIGHT}
            tickFormatter={xFormat ?? formatStep}
            tick={TICK}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            yAxisId="y"
            width={AXIS_WIDTH}
            tickFormatter={yFormat ?? ((value: number) => formatRate(value))}
            tick={TICK}
            axisLine={false}
            tickLine={false}
          />
          {rightFormat !== undefined && (
            <YAxis
              yAxisId="r"
              orientation="right"
              width={AXIS_WIDTH}
              tickFormatter={rightFormat}
              tick={TICK}
              axisLine={false}
              tickLine={false}
            />
          )}
          {children}
        </ComposedChart>
      </ResponsiveContainer>
      {overlay}
    </div>
  );
}

export function Playhead({ total }: { total: number }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const totalRef = useRef(total);
  totalRef.current = total;
  useEffect(() => {
    const el = ref.current;
    if (el === null) return;
    const move = throttled((step: number) => {
      const width = el.parentElement?.clientWidth ?? 0;
      const inner = width - CHART_MARGIN.left - CHART_MARGIN.right - AXIS_WIDTH;
      if (inner <= 0 || totalRef.current <= 0) return;
      const frac = Math.min(1, Math.max(0, step / totalRef.current));
      el.style.transform = `translateX(${(CHART_MARGIN.left + AXIS_WIDTH + frac * inner).toFixed(1)}px)`;
    });
    return subscribePosition(move);
  }, []);
  return (
    <div
      ref={ref}
      className="pointer-events-none absolute left-0 w-px bg-seed/70"
      style={{ top: CHART_MARGIN.top, bottom: X_AXIS_HEIGHT, willChange: 'transform' }}
    />
  );
}

export function Chip({ swatch, label, dashed }: { swatch: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1 font-mono text-2xs text-ink-dim">
      <span
        className="h-0.5 w-3 rounded-full"
        style={
          dashed === true
            ? { backgroundImage: `repeating-linear-gradient(90deg, ${swatch} 0 3px, transparent 3px 5px)` }
            : { background: swatch }
        }
      />
      {label}
    </span>
  );
}
