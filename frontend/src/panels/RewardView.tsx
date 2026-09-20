import { memo, useCallback, useMemo, type PointerEvent, type RefObject } from 'react';
import type { EnvId } from '../data/envs';
import type { Recording } from '../data/layout';
import { cached, deathStep, deliveredRate, envLanes, rateWindow, windowedRates } from '../derive';
import { formatCount, formatRate, formatSigned, formatStep } from '../format';
import { selectSeed, store } from '../store';
import type { Stage } from '../theater/stage';
import { Key, POLICY_NAMES, POLICY_TOKENS } from './charts';
import { Plot } from './Plot';
import type { PlotSpec } from './plotter';

const SPARK_H = 16;

function scrubHandlers(seek: (step: number) => void, xMax: number, inset = { left: 0, right: 0 }, onStart?: () => void) {
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

const SPARK_PAD = 3;

function sparkPath(values: readonly (number | null)[], xs: readonly number[]): { line: string; fill: string; zeroY: number | null } | null {
  let min = 0;
  let max = 0;
  let any = false;
  for (const v of values) {
    if (v === null) continue;
    any = true;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!any) return null;
  if (min === max) {
    min -= Math.abs(min) * 0.1 + 1e-6;
    max += Math.abs(max) * 0.1 + 1e-6;
  }
  const xMax = Math.max(1, xs[xs.length - 1] ?? 0);
  const yAt = (v: number): number => SPARK_H - SPARK_PAD - ((v - min) / (max - min)) * (SPARK_H - SPARK_PAD * 2);
  const y0 = yAt(0);
  const points: string[] = [`0,${y0.toFixed(2)}`];
  let lastY = y0;
  values.forEach((v, i) => {
    if (v === null) return;
    lastY = yAt(v);
    points.push(`${(Math.min(1, (xs[i] ?? 0) / xMax) * 100).toFixed(3)},${lastY.toFixed(2)}`);
  });
  points.push(`100,${lastY.toFixed(2)}`);
  const zero = min < 0 && max > 0;
  const baseY = zero ? y0 : SPARK_H - SPARK_PAD;
  return { line: points.join(' '), fill: `${points.join(' ')} 100,${baseY.toFixed(2)} 0,${baseY.toFixed(2)}`, zeroY: zero ? y0 : null };
}

const Spark = memo(function Spark({ path, stroke, xMax, seed, seek }: { path: ReturnType<typeof sparkPath>; stroke: string; xMax: number; seed: number; seek: (step: number) => void }) {
  return (
    <span className="block min-w-0 flex-1 cursor-ew-resize touch-none select-none" title="Drag to seek" {...scrubHandlers(seek, xMax, undefined, () => selectSeed(seed))} onClick={(event) => event.stopPropagation()}>
      <svg viewBox={`0 0 100 ${SPARK_H}`} preserveAspectRatio="none" className="block w-full" style={{ height: SPARK_H }} aria-hidden="true">
        {path !== null && (
          <>
            {path.zeroY !== null && <line x1="0" x2="100" y1={path.zeroY} y2={path.zeroY} stroke="var(--color-ink)" strokeOpacity={0.22} strokeDasharray="2 3" vectorEffect="non-scaling-stroke" />}
            <polygon points={path.fill} fill={stroke} fillOpacity={0.13} />
            <polyline points={path.line} fill="none" stroke={stroke} strokeWidth={1.25} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
          </>
        )}
      </svg>
    </span>
  );
});

export interface Comparison {
  key: string;
  label: string;
  tone: string;
  runId: string;
}

const seriesFor = (rec: Recording, envId: string) =>
  cached(rec, `reward/${envId}`, () => {
    const lanes = envLanes(rec.meta, envId);
    const rates = windowedRates(rec, lanes.lanes, rateWindow(rec.meta.steps));
    const rows = lanes.seeds
      .map((s, i) => ({ seed: s, rate: deliveredRate(rec, lanes.lanes[i]!), death: deathStep(rec, lanes.lanes[i]!), spark: sparkPath(rates.perLane[i] ?? [], rates.x) }))
      .sort((a, b) => (b.rate ?? -Infinity) - (a.rate ?? -Infinity) || a.seed - b.seed);
    return { lanes, rates, rows };
  });

export function RewardView({ stage, runId, envId, seed, compare }: { stage: RefObject<Stage | null>; runId: string; envId: string; seed: number; compare: Comparison[] }) {
  const picked = store.use((s) => s.seed);
  const recordings = store.use((s) => s.recordings);
  const active = recordings[runId]!;
  const policy = active.meta.policy;
  const tone = POLICY_TOKENS[policy];
  const horizon = Math.max(1, active.meta.steps);
  const win = rateWindow(active.meta.steps);
  const { lanes, rates, rows } = seriesFor(active, envId);
  const primary = `${POLICY_NAMES[policy]} average`;
  const spec = useMemo<PlotSpec>(() => ({
    xMax: horizon,
    yFormat: formatRate,
    band: { x: rates.x, lo: rates.p10, hi: rates.p90, tone },
    lines: [
      ...compare.map((c) => {
        const other = seriesFor(recordings[c.runId]!, envId).rates;
        return { x: other.x, y: other.iqm, tone: c.tone, width: 1.2, label: `${c.label} average` };
      }),
      { x: rates.x, y: rates.iqm, tone, width: 1.6, label: primary },
      { x: rates.x, y: rates.perLane[lanes.seeds.indexOf(seed)] ?? [], tone: 'var(--color-seed)', width: 1.2, label: 'selected seed' },
    ],
  }), [horizon, rates, tone, compare, recordings, envId, primary, lanes, seed]);
  const seek = useCallback((step: number): void => stage.current?.seekStep(step), [stage]);
  const last = rates.x[rates.x.length - 1] ?? 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="px-2 text-2xs uppercase tracking-label" title={`trailing ${formatStep(win)}-step mean · IQM of seeds · band 10–90th percentile`}>Reward per step</div>
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 px-2">
        <Key tone={tone} label={primary} />
        <Key tone="var(--color-seed)" label="selected seed" />
        <Key tone={tone} label="10–90% of seeds" block />
        {compare.map((c) => <Key key={c.key} tone={c.tone} label={`${c.label} average`} />)}
      </div>
      <Plot label="Reward per step" height={188} spec={spec} yLabel="reward / step" playhead seek={seek} />
      <div className="flex items-center gap-2 px-2">
        <span className="text-2xs uppercase tracking-label" title="lifetime reward per step · click to stage">Seeds</span>
        <span className="font-mono text-xs text-ink-dim">{formatCount(rows.length)}</span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {rows.map((row, rank) => (
          <button
            key={rank}
            type="button"
            title={picked === row.seed ? `Release seed ${row.seed}` : `Put seed ${row.seed} on the stage`}
            onClick={() => selectSeed(picked === row.seed ? null : row.seed)}
            className={`flex h-7 w-full items-center gap-2 rounded-md px-2 text-left ${row.seed === seed ? 'bg-seed/10 ring-1 ring-inset ring-seed/40' : 'hover:bg-white/5'}`}
          >
            <span className="w-6 text-right font-mono text-xs text-ink-dim">{rank + 1}</span>
            <span className={`w-10 font-mono text-xs ${row.seed === seed ? 'text-seed' : ''}`}>{row.seed}</span>
            <span className={`w-16 text-right font-mono text-xs tabular-nums ${row.rate === null ? 'text-ink-faint' : row.rate > 0 ? 'text-good' : 'text-bad'}`}>{formatSigned(row.rate)}</span>
            <Spark path={row.spark} stroke={row.death !== null ? 'var(--color-crit)' : row.seed === seed ? 'var(--color-seed)' : tone} xMax={last} seed={row.seed} seek={seek} />
            {row.death !== null && (
              <span className="rounded-sm border border-crit/40 bg-crit/10 px-1 font-mono text-2xs leading-4 text-crit" title={`diverged at step ${formatCount(row.death)}`}>diverged</span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
