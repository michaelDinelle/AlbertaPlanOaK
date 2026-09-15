import { useEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { clsx } from 'clsx';
import { CartesianGrid, ComposedChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { TooltipPayload } from 'recharts/types/state/tooltipSlice';
import type { Policy } from '../data/types';
import { formatRate, formatStep } from '../lib/format';
import { currentPosition, subscribePosition, throttled } from '../playback/store';

export const CHART_MARGIN = { top: 22, right: 30, bottom: 0, left: 0 } as const;
export const AXIS_WIDTH = 62;
export const X_AXIS_HEIGHT = 18;

export const POLICY_TOKENS: Record<Policy, string> = {
  oak: 'var(--color-oak)',
  ppo: 'var(--color-ppo)',
  random: 'var(--color-random)',
};

export const POLICY_ORDER = Object.keys(POLICY_TOKENS) as readonly Policy[];
export const BEST_TOKEN = 'var(--color-best)';

export interface XY {
  x: number;
  y: number | null;
}

export interface XBand {
  x: number;
  band: [number | null, number | null];
}

const cssVarCache = new Map<string, string>();

export function cssVar(name: string): string {
  const hit = cssVarCache.get(name);
  if (hit !== undefined) return hit;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  cssVarCache.set(name, value);
  return value;
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

export function lineProps(stroke: string, width = 1.4, dash?: string) {
  return {
    dataKey: 'y',
    type: 'linear',
    stroke,
    strokeWidth: width,
    strokeDasharray: dash,
    dot: false,
    activeDot: { r: 2.5, fill: stroke, strokeWidth: 0 },
    isAnimationActive: false,
    yAxisId: 'y',
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
    tooltipType: 'none',
  } as const;
}

const TICK = { fill: 'var(--color-ink-dim)', fontSize: 11, fontFamily: 'var(--font-mono)' } as const;

const TIP_CURSOR = { stroke: 'var(--color-ink-faint)', strokeOpacity: 0.6, strokeDasharray: '3 3' } as const;

function logTicks(min: number, max: number): number[] {
  const lo = Math.log10(Math.max(min, 1));
  const hi = Math.log10(Math.max(max, 1));
  const out = [Math.max(min, 1)];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
    if (e - lo > 0.25 && hi - e > 0.25) out.push(10 ** e);
  }
  out.push(Math.max(max, 1));
  return out;
}

function TipCard({
  active,
  payload,
  label,
  xFormat,
  yFormat,
}: {
  active: boolean;
  payload: TooltipPayload;
  label: string | number | undefined;
  xFormat: (value: number) => string;
  yFormat: (value: number) => string;
}) {
  if (!active || typeof label !== 'number') return null;
  const rows = payload.filter(
    (entry) =>
      entry.type !== 'none'
      && typeof entry.value === 'number'
      && Number.isFinite(entry.value)
      && (entry.payload as { x?: number } | undefined)?.x === label,
  );
  if (rows.length === 0) return null;
  return (
    <div className="glass rounded-md px-2.5 py-1.5 text-xs text-ink">
      <div className="font-mono tabular-nums text-ink-dim">{xFormat(label)}</div>
      {rows.map((entry, i) => (
        <div key={i} className="mt-0.5 flex items-center gap-1.5">
          <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: entry.color ?? 'var(--color-ink)' }} />
          <span className="font-mono tabular-nums">
            {yFormat(entry.value as number)}
          </span>
          {typeof entry.name === 'string' && entry.name !== 'y' && (
            <span className="text-ink-dim">{entry.name}</span>
          )}
        </div>
      ))}
    </div>
  );
}

export function TimeChart({
  label,
  height,
  xMin = 0,
  xMax,
  xScale,
  xFormat,
  yFormat,
  xLabel,
  yLabel,
  overlay,
  yDomain,
  children,
}: {
  label: string;
  height: number;
  xMin?: number;
  xMax: number;
  xScale?: 'log';
  xFormat?: (value: number) => string;
  yFormat?: (value: number) => string;
  xLabel?: string;
  yLabel?: string;
  overlay?: ReactNode;
  yDomain?: readonly [number, number];
  children: ReactNode;
}) {
  return (
    <div role="img" aria-label={label} className="relative shrink-0">
      {yLabel !== undefined && (
        <span className="pointer-events-none absolute top-0 font-mono text-2xs text-ink-dim" style={{ left: 2 }}>
          {yLabel}
        </span>
      )}
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
            minTickGap={28}
            ticks={xScale === 'log' ? logTicks(xMin, xMax) : undefined}
            tickFormatter={xFormat ?? formatStep}
            tick={TICK}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            yAxisId="y"
            width={AXIS_WIDTH}
            domain={yDomain === undefined ? ['auto', 'auto'] : [yDomain[0], yDomain[1]]}
            allowDataOverflow={yDomain !== undefined}
            tickFormatter={yFormat ?? ((value: number) => formatRate(value))}
            tick={TICK}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            isAnimationActive={false}
            cursor={TIP_CURSOR}
            content={({ active, payload, label }) => (
              <TipCard
                active={active}
                payload={payload}
                label={label}
                xFormat={xFormat ?? formatStep}
                yFormat={yFormat ?? ((value: number) => formatRate(value))}
              />
            )}
          />
          {children}
        </ComposedChart>
      </ResponsiveContainer>
      {overlay}
      </div>
      {xLabel !== undefined && (
        <div className="pr-1 text-right font-mono text-2xs text-ink-dim">{xLabel}</div>
      )}
    </div>
  );
}

export function Playhead({ total, from = 0 }: { total: number; from?: number }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const totalRef = useRef(total);
  totalRef.current = total;
  const fromRef = useRef(from);
  fromRef.current = from;
  useEffect(() => {
    const el = ref.current;
    const parent = el?.parentElement;
    if (el == null || parent == null) return;
    let width = parent.clientWidth;
    const observer = new ResizeObserver((entries) => {
      width = entries[0]?.contentRect.width ?? width;
    });
    observer.observe(parent);
    const move = throttled((step: number) => {
      const inner = width - CHART_MARGIN.left - CHART_MARGIN.right - AXIS_WIDTH;
      if (inner <= 0 || totalRef.current <= 0) return;
      const span = totalRef.current - fromRef.current;
      if (span <= 0) return;
      const frac = Math.min(1, Math.max(0, (step - fromRef.current) / span));
      el.style.transform = `translateX(${(CHART_MARGIN.left + AXIS_WIDTH + frac * inner).toFixed(1)}px)`;
    });
    move(currentPosition());
    const unsubscribe = subscribePosition(move);
    return () => {
      observer.disconnect();
      unsubscribe();
    };
  }, []);
  return (
    <div
      ref={ref}
      className="pointer-events-none absolute left-0 w-px bg-seed/70"
      style={{ top: CHART_MARGIN.top, bottom: X_AXIS_HEIGHT, willChange: 'transform' }}
    />
  );
}

export const TXT = {
  label: 'text-2xs uppercase tracking-label text-ink',
  body: 'text-xs text-ink',
  bodyDim: 'text-xs text-ink-dim',
  data: 'font-mono text-xs tabular-nums text-ink',
  dataDim: 'font-mono text-xs tabular-nums text-ink-dim',
} as const;

export function Label({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <span className={clsx(TXT.label, hint !== undefined && 'cursor-help')} title={hint}>
      {children}
    </span>
  );
}

export function Hotkey({
  label,
  keyName,
  side,
  align,
  children,
}: {
  label: string;
  keyName?: string;
  side?: 'above' | 'below';
  align?: 'start' | 'center' | 'end';
  children: ReactNode;
}) {
  const anchor = useRef<HTMLSpanElement | null>(null);
  const [box, setBox] = useState<DOMRect | null>(null);
  const show = (): void => {
    const el = anchor.current;
    if (el !== null) setBox(el.getBoundingClientRect());
  };
  const hide = (): void => setBox(null);
  const above = side === 'above';
  const at = align ?? 'start';
  return (
    <span
      ref={anchor}
      className="flex"
      onPointerEnter={show}
      onPointerLeave={hide}
      onPointerDown={hide}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      {box !== null &&
        createPortal(
          <div
            aria-hidden
            className="pointer-events-none fixed z-[100] flex w-0"
            style={{
              left: at === 'center' ? box.left + box.width / 2 : at === 'end' ? box.right : box.left,
              justifyContent: at === 'center' ? 'center' : at === 'end' ? 'flex-end' : 'flex-start',
              ...(above
                ? { bottom: window.innerHeight - box.top + 6 }
                : { top: box.bottom + 6 }),
            }}
          >
            <span className="flex items-center gap-1.5 whitespace-nowrap rounded-md border border-white/15 bg-ground px-2 py-1 text-2xs text-ink shadow-lg motion-safe:animate-fade-in">
              {label}
              {keyName !== undefined && (
                <kbd className="rounded-sm border border-white/15 bg-white/10 px-1 font-mono leading-4">{keyName}</kbd>
              )}
            </span>
          </div>,
          document.body,
        )}
    </span>
  );
}

export function Chip({ swatch, label, dashed, block }: { swatch: string; label: string; dashed?: boolean; block?: boolean }) {
  return (
    <span className={clsx('flex items-center gap-1', TXT.bodyDim)}>
      <span
        className={block === true ? 'h-2 w-3 rounded-sm opacity-40' : 'h-0.5 w-3 rounded-full'}
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
