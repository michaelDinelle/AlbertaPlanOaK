import { useEffect, useMemo, useRef, useSyncExternalStore, type RefObject } from 'react';
import { clsx } from 'clsx';
import { Area, Line } from 'recharts';
import type { Policy, RunDetail } from '@shared/types';
import { formatStep } from '../lib/format';
import { envLanes } from '../lib/run';
import { heroSeed } from '../lib/selection';
import { useTransport } from '../playback/store';
import { useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { retentionWindow, type CumulativeReward } from '../tape/derive';
import type { PenCurves } from '../tape/pen';
import { PEN_KEYS, penStore, type PenKey, type PenSource } from '../tape/penStore';
import { RewardStream } from '../tape/rewardStream';
import { rewardUrl } from '../data/api';
import { useTapeSnapshot, type DeriveSpec } from '../tape/store';
import { scrubToStep } from '../theater/controller';
import { navigate } from '../url/useUrlState';
import {
  AXIS_WIDTH,
  CHART_MARGIN,
  Chip,
  Label,
  BEST_TOKEN,
  POLICY_ORDER,
  POLICY_TOKENS,
  Playhead,
  TXT,
  TimeChart,
  X_AXIS_HEIGHT,
  bandProps,
  bandRows,
  cssVar,
  lineProps,
  seriesRows,
  type XY,
} from './charts';
import { SeedList } from './SeedList';

function summaryPoints(run: RunDetail | null, envId: string | null): XY[] | null {
  if (run === null || envId === null) return null;
  const summary = run.summary;
  const env = summary?.envs[envId];
  const iqm = env?.curves.iqm;
  if (summary === null || summary === undefined || env === undefined || iqm === undefined) return null;
  return seriesRows(summary.curveSteps, iqm);
}

const SPARSE_SAMPLES = 4;
const TRAIL_WINDOWS = 100;
const PEN_SAMPLES = 192;
const CHIP_INTERVAL_MS = 100;

const rewardStreams = new Map<string, RewardStream>();

function useRewardStream(runId: string | null, envId: string | null): RewardStream | null {
  return useMemo(() => {
    if (runId === null || envId === null) return null;
    const url = rewardUrl(runId, envId, 'pack');
    let have = rewardStreams.get(url);
    if (have === undefined) {
      have = new RewardStream(url, rewardUrl(runId, envId, 'idx'));
      rewardStreams.set(url, have);
    }
    return have;
  }, [runId, envId]);
}

function sparseOrigin(points: readonly { x: number }[], horizon: number): number {
  if (points.length === 0 || points.length >= SPARSE_SAMPLES) return 0;
  const first = points[0]?.x ?? 0;
  return first >= horizon ? 0 : first;
}

function resolveColor(token: string): string {
  const match = /^var\((--[^)]+)\)$/.exec(token);
  return match === null ? token : cssVar(match[1]!);
}

function lastFinite(values: readonly (number | null)[]): number | null {
  for (let i = values.length - 1; i >= 0; i--) {
    const value = values[i];
    if (value !== null && value !== undefined && Number.isFinite(value)) return value;
  }
  return null;
}

interface PenSeries {
  key: PenKey;
  label: string;
  token: string;
}

function LiveChip({ swatch, label, pen }: { swatch: string; label: string; pen: PenKey }) {
  const ref = useRef<HTMLSpanElement | null>(null);
  useEffect(() => {
    let at = 0;
    const write = (): void => {
      const now = performance.now();
      if (now - at < CHIP_INTERVAL_MS) return;
      at = now;
      const frame = penStore.frame();
      const value = frame === null ? null : frame.pens[pen] === null ? null : lastFinite(frame.pens[pen]!.iqm);
      if (ref.current !== null) ref.current.textContent = value === null ? STR.reward.noValue : value.toPrecision(3);
    };
    write();
    return penStore.subscribeFrame(write);
  }, [pen]);
  return (
    <span className={clsx('flex items-center gap-1', TXT.bodyDim)}>
      <span className="h-0.5 w-3 rounded-full" style={{ background: swatch }} />
      <span>{STR.reward.average(label)}</span>
      <span ref={ref} className="font-mono tabular-nums" />
    </span>
  );
}

function nearestIndex(xs: readonly number[], x: number): number {
  let lo = 0;
  let hi = xs.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((xs[mid] ?? 0) < x) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs((xs[lo - 1] ?? 0) - x) <= Math.abs((xs[lo] ?? 0) - x)) return lo - 1;
  return lo;
}

function valueAt(pen: PenCurves | null, step: number, values?: readonly (number | null)[]): number | null {
  if (pen === null) return null;
  const i = nearestIndex(pen.x, step);
  const value = (values ?? pen.iqm)[i];
  return value === null || value === undefined || !Number.isFinite(value) ? null : value;
}

const PLOT_BOX = {
  left: CHART_MARGIN.left + AXIS_WIDTH,
  right: CHART_MARGIN.right,
  top: CHART_MARGIN.top,
  bottom: CHART_MARGIN.bottom + X_AXIS_HEIGHT,
} as const;

interface ReadoutRow {
  id: string;
  swatch: string;
  label: string;
}

function PenReadout({
  host,
  primary,
  series,
  heroLane,
}: {
  host: RefObject<HTMLDivElement | null>;
  primary: PenSeries;
  series: readonly PenSeries[];
  heroLane: number;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const cursorRef = useRef<HTMLDivElement | null>(null);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const stepRef = useRef<HTMLDivElement | null>(null);
  const valueRefs = useRef(new Map<string, HTMLSpanElement>());
  const rows: ReadoutRow[] = [
    { id: primary.key, swatch: primary.token, label: STR.reward.average(primary.label) },
    ...(heroLane >= 0 ? [{ id: 'hero', swatch: 'var(--color-seed)', label: STR.reward.hero }] : []),
    { id: 'range', swatch: primary.token, label: STR.reward.range },
    ...series.map((item) => ({ id: item.key, swatch: item.token, label: STR.reward.average(item.label) })),
  ];
  useEffect(() => {
    const container = host.current;
    const wrap = wrapRef.current;
    const cursor = cursorRef.current;
    const card = cardRef.current;
    if (container === null || wrap === null || cursor === null || card === null) return;
    let frac: number | null = null;
    const set = (id: string, text: string): void => {
      const el = valueRefs.current.get(id);
      if (el !== undefined) el.textContent = text;
    };
    const fmt = (value: number | null): string => (value === null ? STR.reward.noValue : value.toPrecision(3));
    const render = (): void => {
      const frame = penStore.frame();
      if (frac === null || frame === null) {
        cursor.hidden = true;
        card.hidden = true;
        return;
      }
      const width = wrap.clientWidth;
      const step = frac * Math.max(1, frame.step);
      const main = frame.pens[primary.key];
      set(primary.key, fmt(valueAt(main, step)));
      if (heroLane >= 0) set('hero', fmt(valueAt(main, step, main?.perLane[heroLane])));
      const lo = valueAt(main, step, main?.p10);
      const hi = valueAt(main, step, main?.p90);
      set('range', lo === null || hi === null ? STR.reward.noValue : `${fmt(lo)} … ${fmt(hi)}`);
      for (const item of series) set(item.key, fmt(valueAt(frame.pens[item.key], step)));
      if (stepRef.current !== null) stepRef.current.textContent = formatStep(Math.round(step));
      const x = frac * width;
      cursor.style.transform = `translateX(${x.toFixed(1)}px)`;
      cursor.hidden = false;
      card.hidden = false;
      const cardWidth = card.offsetWidth;
      const preferred = x + 12 + cardWidth <= width ? x + 12 : x - 12 - cardWidth;
      card.style.left = `${Math.max(0, Math.min(width - cardWidth, preferred)).toFixed(1)}px`;
    };
    const move = (event: PointerEvent): void => {
      const rect = wrap.getBoundingClientRect();
      const f = (event.clientX - rect.left) / Math.max(1, rect.width);
      frac = f < 0 || f > 1 ? null : f;
      render();
    };
    const leave = (): void => {
      frac = null;
      render();
    };
    container.addEventListener('pointermove', move);
    container.addEventListener('pointerleave', leave);
    const unsubscribe = penStore.subscribeFrame(() => {
      if (frac !== null) render();
    });
    return () => {
      container.removeEventListener('pointermove', move);
      container.removeEventListener('pointerleave', leave);
      unsubscribe();
    };
  }, [host, primary, series, heroLane]);
  return (
    <div
      ref={wrapRef}
      className="pointer-events-none absolute"
      style={{ left: PLOT_BOX.left, right: PLOT_BOX.right, top: PLOT_BOX.top, bottom: PLOT_BOX.bottom }}
    >
      <div ref={cursorRef} hidden className="absolute inset-y-0 left-0 w-px bg-ink/40" />
      <div ref={cardRef} hidden className="glass absolute top-1 whitespace-nowrap rounded-md px-2.5 py-1.5 text-xs text-ink">
        <div ref={stepRef} className="font-mono tabular-nums text-ink-dim" />
        {rows.map((row) => (
          <div key={row.id} className="mt-0.5 flex items-center gap-1.5">
            <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: row.swatch }} />
            <span
              ref={(el) => {
                if (el === null) valueRefs.current.delete(row.id);
                else valueRefs.current.set(row.id, el);
              }}
              className="font-mono tabular-nums"
            />
            <span className="text-ink-dim">{row.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PenLayer({ primary, series, heroLane }: { primary: PenSeries; series: readonly PenSeries[]; heroLane: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const parent = canvas?.parentElement;
    if (canvas == null || parent == null) return;
    let width = 0;
    let height = 0;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect === undefined) return;
      width = rect.width;
      height = rect.height;
      draw();
    });
    observer.observe(parent);
    const draw = (): void => {
      const frame = penStore.frame();
      const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
      const bw = Math.round(width * dpr);
      const bh = Math.round(height * dpr);
      if (bw <= 0 || bh <= 0) return;
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }
      const ctx = canvas.getContext('2d');
      if (ctx === null) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      if (frame === null) return;
      const axis = penStore.axis();
      const xMax = Math.max(1, frame.step);
      const span = axis.yMax - axis.yMin;
      if (!(span > 0)) return;
      const px = (x: number): number => (x / xMax) * width;
      const py = (y: number): number => height - ((y - axis.yMin) / span) * height;
      const stroke = (xs: readonly number[], ys: readonly (number | null)[], color: string, lineWidth: number): void => {
        ctx.beginPath();
        let open = false;
        for (let i = 0; i < xs.length; i++) {
          const y = ys[i];
          if (y === null || y === undefined || !Number.isFinite(y)) {
            open = false;
            continue;
          }
          const X = px(xs[i] ?? 0);
          const Y = py(y);
          if (open) ctx.lineTo(X, Y);
          else ctx.moveTo(X, Y);
          open = true;
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.lineJoin = 'round';
        ctx.stroke();
      };
      const dot = (ys: readonly (number | null)[], color: string): void => {
        const y = lastFinite(ys);
        if (y === null) return;
        ctx.beginPath();
        ctx.arc(px(xMax), py(y), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      };
      const main = frame.pens[primary.key];
      const mainColor = resolveColor(primary.token);
      if (main !== null) {
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < main.x.length; i++) {
          const y = main.p90[i];
          if (y === null || y === undefined) continue;
          const X = px(main.x[i] ?? 0);
          if (started) ctx.lineTo(X, py(y));
          else ctx.moveTo(X, py(y));
          started = true;
        }
        for (let i = main.x.length - 1; i >= 0; i--) {
          const y = main.p10[i];
          if (y === null || y === undefined) continue;
          ctx.lineTo(px(main.x[i] ?? 0), py(y));
        }
        if (started) {
          ctx.closePath();
          ctx.globalAlpha = 0.18;
          ctx.fillStyle = mainColor;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }
      for (const item of series) {
        const pen = frame.pens[item.key];
        if (pen === null) continue;
        stroke(pen.x, pen.iqm, resolveColor(item.token), 1.2);
      }
      if (main !== null) {
        stroke(main.x, main.iqm, mainColor, 1.6);
        const heroValues = heroLane >= 0 ? main.perLane[heroLane] : undefined;
        if (heroValues !== undefined) stroke(main.x, heroValues, cssVar('--color-seed'), 1.2);
        for (const item of series) {
          const pen = frame.pens[item.key];
          if (pen !== null) dot(pen.iqm, resolveColor(item.token));
        }
        dot(main.iqm, mainColor);
        if (heroValues !== undefined) dot(heroValues, cssVar('--color-seed'));
      }
    };
    const unsubscribe = penStore.subscribeFrame(draw);
    return () => {
      unsubscribe();
      observer.disconnect();
    };
  }, [primary, series, heroLane]);
  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute"
      style={{
        left: PLOT_BOX.left,
        right: PLOT_BOX.right,
        top: PLOT_BOX.top,
        bottom: PLOT_BOX.bottom,
        width: `calc(100% - ${PLOT_BOX.left + PLOT_BOX.right}px)`,
        height: `calc(100% - ${PLOT_BOX.top + PLOT_BOX.bottom}px)`,
      }}
    />
  );
}

export function RewardView({
  runs,
  best,
  active,
}: {
  runs: Record<Policy, RunDetail | null>;
  best: RunDetail | null;
  active: RunDetail | null;
}) {
  const sel = useSelection();
  const transport = useTransport();
  const slicing = transport.slice;
  const envId = sel.envId;
  const lanes = useMemo(
    () => (active === null || envId === null ? null : envLanes(active.meta, envId)),
    [active, envId],
  );
  const horizon = active?.meta.steps ?? runs.oak?.meta.steps ?? 0;
  const sliceWin = Math.max(1, Math.round(horizon / TRAIL_WINDOWS));
  const window = slicing ? sliceWin : retentionWindow(horizon);
  const spec = useMemo<(DeriveSpec & { mode: 'windowedRates' }) | null>(
    () => (lanes === null || active === null ? null : { mode: 'windowedRates', lanes: lanes.lanes, win: retentionWindow(active.meta.steps) }),
    [lanes, active],
  );
  const rates = useTapeSnapshot(slicing ? null : active?.id ?? null, envId, 'windowedRates', spec);

  const cumulativeSpec = (run: RunDetail | null): (DeriveSpec & { mode: 'cumulativeReward' }) | null =>
    !slicing || run === null || envId === null ? null : { mode: 'cumulativeReward', lanes: envLanes(run.meta, envId).lanes };
  const oakSpec = useMemo(() => cumulativeSpec(runs.oak), [slicing, runs.oak, envId]);
  const ppoSpec = useMemo(() => cumulativeSpec(runs.ppo), [slicing, runs.ppo, envId]);
  const randomSpec = useMemo(() => cumulativeSpec(runs.random), [slicing, runs.random, envId]);
  const bestSpec = useMemo(() => cumulativeSpec(best), [slicing, best, envId]);
  const cumulative: Record<PenKey, CumulativeReward | null> = {
    oak: useTapeSnapshot(oakSpec === null ? null : runs.oak?.id ?? null, envId, 'cumulativeReward', oakSpec),
    ppo: useTapeSnapshot(ppoSpec === null ? null : runs.ppo?.id ?? null, envId, 'cumulativeReward', ppoSpec),
    random: useTapeSnapshot(randomSpec === null ? null : runs.random?.id ?? null, envId, 'cumulativeReward', randomSpec),
    best: useTapeSnapshot(bestSpec === null ? null : best?.id ?? null, envId, 'cumulativeReward', bestSpec),
  };
  const streams: Record<PenKey, RewardStream | null> = {
    oak: useRewardStream(slicing ? runs.oak?.id ?? null : null, envId),
    ppo: useRewardStream(slicing ? runs.ppo?.id ?? null : null, envId),
    random: useRewardStream(slicing ? runs.random?.id ?? null : null, envId),
    best: useRewardStream(slicing ? best?.id ?? null : null, envId),
  };

  const hero = heroSeed(sel.selection);
  const heroIdx = hero === null || lanes === null ? -1 : lanes.seeds.indexOf(hero);
  const tone = POLICY_TOKENS[sel.policy];

  useEffect(() => {
    const sources = { oak: null, ppo: null, random: null, best: null } as Record<PenKey, PenSource | null>;
    for (const key of PEN_KEYS) {
      const cum = cumulative[key];
      if (cum !== null) sources[key] = { cumulative: cum, stream: streams[key] };
    }
    penStore.setInputs({ active: slicing, win: sliceWin, samples: PEN_SAMPLES, primary: sel.policy, sources, heroLane: heroIdx });
  }, [slicing, sliceWin, sel.policy, heroIdx, cumulative.oak, cumulative.ppo, cumulative.random, cumulative.best, streams.oak, streams.ppo, streams.random, streams.best]);
  useEffect(() => () => penStore.setInputs({ active: false, win: 1, samples: PEN_SAMPLES, primary: 'oak', sources: { oak: null, ppo: null, random: null, best: null }, heroLane: -1 }), []);
  const axis = useSyncExternalStore(penStore.subscribeAxis.bind(penStore), penStore.axis.bind(penStore));

  const avgPts = useMemo(() => (rates === null ? [] : seriesRows(rates.x, rates.iqm)), [rates]);
  const band = useMemo(() => (rates === null ? [] : bandRows(rates.x, rates.p10, rates.p90)), [rates]);
  const heroPts = useMemo(
    () => (rates === null || heroIdx < 0 ? [] : seriesRows(rates.x, rates.perLane[heroIdx] ?? [])),
    [rates, heroIdx],
  );
  const primary: PenSeries = { key: sel.policy, label: STR.policy[sel.policy], token: tone };
  const comparisons = useMemo<PenSeries[]>(
    () => [
      ...POLICY_ORDER.filter((policy) => policy !== sel.policy && runs[policy] !== null).map((policy) => ({ key: policy, label: STR.policy[policy], token: POLICY_TOKENS[policy] })),
      ...(best === null ? [] : [{ key: 'best' as const, label: STR.best, token: BEST_TOKEN }]),
    ],
    [runs, best, sel.policy],
  );
  const summaryLines = useMemo(
    () => comparisons.map((item) => ({ ...item, pts: item.key === 'best' ? summaryPoints(best, envId) : summaryPoints(runs[item.key as Policy], envId) })),
    [comparisons, best, runs, envId],
  );
  const stale = slicing ? [] : summaryLines.filter(({ pts }) => pts === null).map(({ label }) => label);

  const xMax = slicing ? axis.xMax : Math.max(1, horizon);
  const dragging = useRef(false);
  const chartRef = useRef<HTMLDivElement | null>(null);
  const stepAt = (event: React.PointerEvent<HTMLDivElement>): number => {
    const rect = event.currentTarget.getBoundingClientRect();
    const inner = Math.max(1, rect.width - CHART_MARGIN.left - CHART_MARGIN.right - AXIS_WIDTH);
    const frac = Math.min(1, Math.max(0, (event.clientX - rect.left - CHART_MARGIN.left - AXIS_WIDTH) / inner));
    return Math.round(frac * xMax);
  };
  const waiting = slicing ? cumulative[sel.policy] === null : rates === null;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="px-2">
        <Label hint={STR.reward.titleHint(formatStep(window))}>{STR.reward.title}</Label>
      </div>
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 px-2">
        {slicing ? <LiveChip swatch={tone} label={primary.label} pen={primary.key} /> : <Chip swatch={tone} label={STR.reward.average(primary.label)} />}
        <Chip swatch="var(--color-seed)" label={STR.reward.hero} />
        <Chip swatch={tone} label={STR.reward.range} block />
        {slicing
          ? comparisons.map((item) => <LiveChip key={item.key} swatch={item.token} label={item.label} pen={item.key} />)
          : summaryLines.map(({ key, label, token, pts }) => pts !== null && <Chip key={key} swatch={token} label={STR.reward.average(label)} />)}
      </div>
      <div
        ref={chartRef}
        className="relative cursor-ew-resize touch-none select-none rounded-md transition-colors hover:bg-white/5"
        onPointerDown={(event) => {
          dragging.current = true;
          try {
            event.currentTarget.setPointerCapture(event.pointerId);
          } catch {}
          scrubToStep(stepAt(event));
        }}
        onPointerMove={(event) => {
          if (dragging.current) scrubToStep(stepAt(event));
        }}
        onPointerUp={(event) => {
          if (!dragging.current) return;
          dragging.current = false;
          const step = stepAt(event);
          scrubToStep(step);
          navigate({ at: step });
        }}
        onPointerCancel={() => {
          dragging.current = false;
        }}
      >
        {slicing ? (
          <>
            <TimeChart
              label={STR.reward.title}
              height={188}
              xMax={xMax}
              yDomain={[axis.yMin, axis.yMax]}
              xLabel={STR.axis.steps}
              yLabel={STR.axis.rewardRate}
              overlay={<PenLayer primary={primary} series={comparisons} heroLane={heroIdx} />}
            >
              {null}
            </TimeChart>
            <PenReadout host={chartRef} primary={primary} series={comparisons} heroLane={heroIdx} />
          </>
        ) : (
          <TimeChart
            label={STR.reward.title}
            height={188}
            xMin={sparseOrigin(avgPts, horizon)}
            xMax={xMax}
            xLabel={STR.axis.steps}
            yLabel={STR.axis.rewardRate}
            overlay={<Playhead total={xMax} from={sparseOrigin(avgPts, horizon)} />}
          >
            <Area {...bandProps(tone)} data={band} />
            <Line {...lineProps(tone, 1.6)} name={STR.reward.average(primary.label)} data={avgPts} />
            <Line {...lineProps('var(--color-seed)', 1.2)} name={STR.reward.hero} data={heroPts} />
            {summaryLines.map(
              ({ key, label, token, pts }) =>
                pts !== null && <Line key={key} {...lineProps(token, 1.2)} name={STR.reward.average(label)} data={pts} />,
            )}
          </TimeChart>
        )}
        {waiting && (
          <div className={clsx('absolute inset-0 flex items-center justify-center', TXT.bodyDim)}>
            {active === null ? STR.reward.waiting : STR.reward.reading}
          </div>
        )}
      </div>
      {stale.length > 0 && (
        <div className={clsx('px-2', TXT.bodyDim)}>{STR.reward.stale(stale.join(' and '))}</div>
      )}
      <SeedList active={active} live={slicing} penSeeds={lanes?.seeds ?? null} />
    </div>
  );
}
