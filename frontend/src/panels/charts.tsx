import { useEffect, useRef, type PointerEvent, type ReactNode } from 'react';
import { CartesianGrid, ComposedChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { TooltipPayload } from 'recharts/types/state/tooltipSlice';
import type { Policy } from '../data/oak';
import { formatRate, formatStep } from '../format';
import { frameListeners, type Frame } from '../theater/stage';

const CHART_MARGIN = { top: 22, right: 30, bottom: 0, left: 0 } as const;
const AXIS_WIDTH = 62;
const X_AXIS_HEIGHT = 18;

export const POLICY_ORDER = ['oak', 'ppo', 'random'] as const;
export const POLICY_NAMES: Record<Policy, string> = { oak: 'OaK', ppo: 'PPO', random: 'Random' };
export const POLICY_TOKENS: Record<Policy, string> = { oak: 'var(--color-oak)', ppo: 'var(--color-ppo)', random: 'var(--color-random)' };
export const BEST_TOKEN = 'var(--color-best)';

export interface XY { x: number; y: number | null }

export function seriesRows(x: readonly number[], y: readonly (number | null)[]): XY[] {
  const out: XY[] = [];
  for (let i = 0; i < Math.min(x.length, y.length); i++) out.push({ x: x[i] ?? 0, y: Number.isFinite(y[i] ?? NaN) ? y[i]! : null });
  return out;
}

export function bandRows(x: readonly number[], lo: readonly (number | null)[], hi: readonly (number | null)[]): { x: number; band: [number | null, number | null] }[] {
  const out: { x: number; band: [number | null, number | null] }[] = [];
  for (let i = 0; i < Math.min(x.length, lo.length, hi.length); i++) out.push({ x: x[i] ?? 0, band: [lo[i] ?? null, hi[i] ?? null] });
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

export function bandProps(fill: string) {
  return { dataKey: 'band', stroke: 'none', fill, fillOpacity: 0.12, isAnimationActive: false, yAxisId: 'y', tooltipType: 'none' } as const;
}

export function scrubHandlers(seek: (step: number) => void, xMax: number, inset = { left: 0, right: 0 }, onStart?: () => void) {
  const at = (event: PointerEvent<HTMLElement>): void => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const width = Math.max(1, rect.width - inset.left - inset.right);
    seek(Math.round(Math.max(0, Math.min(1, (event.clientX - rect.left - inset.left) / width)) * xMax));
    if (event.type === 'pointerup') event.currentTarget.releasePointerCapture(event.pointerId);
  };
  return {
    onPointerDown(event: PointerEvent<HTMLElement>) {
      if (event.button !== 0 || !event.isPrimary) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      onStart?.();
      at(event);
    },
    onPointerMove: at,
    onPointerUp: at,
  };
}

export function Key({ tone, label, dashed, block }: { tone: string; label: string; dashed?: boolean; block?: boolean }) {
  return (
    <span className="flex items-center gap-1 text-xs text-ink-dim">
      <span
        className={block === true ? 'h-2 w-3 rounded-sm opacity-40' : 'h-0.5 w-3 rounded-full'}
        style={dashed === true ? { backgroundImage: `repeating-linear-gradient(90deg, ${tone} 0 3px, transparent 3px 5px)` } : { background: tone }}
      />
      {label}
    </span>
  );
}

const TICK = { fill: 'var(--color-ink-dim)', fontSize: 11, fontFamily: 'var(--font-mono)' } as const;
const TIP_CURSOR = { stroke: 'var(--color-ink-faint)', strokeOpacity: 0.6, strokeDasharray: '3 3' } as const;

function TipCard({ active, payload, label, yFormat }: { active: boolean; payload: TooltipPayload; label: string | number | undefined; yFormat: (value: number) => string }) {
  if (!active || typeof label !== 'number') return null;
  const rows = payload.filter((entry) => entry.type !== 'none' && typeof entry.value === 'number' && Number.isFinite(entry.value) && (entry.payload as { x?: number } | undefined)?.x === label);
  if (rows.length === 0) return null;
  return (
    <div className="glass rounded-md px-2.5 py-1.5 text-xs text-ink">
      <div className="font-mono tabular-nums text-ink-dim">{formatStep(label)}</div>
      {rows.map((entry, i) => (
        <div key={i} className="mt-0.5 flex items-center gap-1.5">
          <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: entry.color ?? 'var(--color-ink)' }} />
          <span className="font-mono tabular-nums">{yFormat(entry.value as number)}</span>
          {typeof entry.name === 'string' && entry.name !== 'y' && <span className="text-ink-dim">{entry.name}</span>}
        </div>
      ))}
    </div>
  );
}

export function TimeChart({ label, height, xMax, yFormat = formatRate, yLabel, overlay, seek, children }: {
  label: string;
  height: number;
  xMax: number;
  yFormat?: (value: number) => string;
  yLabel: string;
  overlay?: ReactNode;
  seek?: (step: number) => void;
  children: ReactNode;
}) {
  return (
    <div
      role="img"
      aria-label={label}
      className={`relative shrink-0 ${seek === undefined ? '' : 'cursor-ew-resize touch-none select-none rounded-md hover:bg-white/5'}`}
      {...(seek === undefined ? {} : scrubHandlers(seek, xMax, { left: CHART_MARGIN.left + AXIS_WIDTH, right: CHART_MARGIN.right }))}
    >
      <span className="pointer-events-none absolute left-0.5 top-0 font-mono text-2xs text-ink-dim">{yLabel}</span>
      <div className="relative" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={CHART_MARGIN}>
            <CartesianGrid stroke="var(--color-line)" strokeOpacity={0.45} vertical={false} />
            <XAxis dataKey="x" type="number" domain={[0, xMax]} allowDataOverflow height={X_AXIS_HEIGHT} minTickGap={28} tickFormatter={formatStep} tick={TICK} axisLine={false} tickLine={false} />
            <YAxis yAxisId="y" width={AXIS_WIDTH} domain={['auto', 'auto']} tickFormatter={yFormat} tick={TICK} axisLine={false} tickLine={false} />
            <Tooltip isAnimationActive={false} cursor={TIP_CURSOR} content={({ active, payload, label }) => <TipCard active={active} payload={payload} label={label} yFormat={yFormat} />} />
            {children}
          </ComposedChart>
        </ResponsiveContainer>
        {overlay}
      </div>
      <div className="pr-1 text-right font-mono text-2xs text-ink-dim">steps</div>
    </div>
  );
}

export function Playhead({ total }: { total: number }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const totalRef = useRef(total);
  totalRef.current = total;
  useEffect(() => {
    const el = ref.current!;
    const parent = el.parentElement!;
    let raf = 0;
    let step = 0;
    const draw = (): void => {
      raf = 0;
      const inner = parent.clientWidth - CHART_MARGIN.left - CHART_MARGIN.right - AXIS_WIDTH;
      if (inner <= 0 || totalRef.current <= 0) return;
      const frac = Math.min(1, Math.max(0, step / totalRef.current));
      el.style.transform = `translateX(${(CHART_MARGIN.left + AXIS_WIDTH + frac * inner).toFixed(1)}px)`;
    };
    const listen = (frame: Frame): void => {
      step = Math.max(0, frame.step);
      if (raf === 0) raf = requestAnimationFrame(draw);
    };
    frameListeners.add(listen);
    return () => {
      frameListeners.delete(listen);
      cancelAnimationFrame(raf);
    };
  }, []);
  return <div ref={ref} className="pointer-events-none absolute left-0 w-px bg-seed/70" style={{ top: CHART_MARGIN.top, bottom: X_AXIS_HEIGHT, willChange: 'transform' }} />;
}
