import { useMemo } from 'react';
import { Line } from 'recharts';
import type { RunDetail } from '@shared/types';
import { STR } from '../strings';
import { Chip, TimeChart, lineProps, seriesRows } from './charts';

const POLICIES = [
  { key: 'oak', token: 'var(--color-oak)' },
  { key: 'ppo', token: 'var(--color-ppo)' },
  { key: 'random', token: 'var(--color-random)' },
] as const;

function exp(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toExponential(1);
}

export function ComputeView({
  detail,
  ppo,
  random,
}: {
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
}) {
  const summaries = useMemo(
    () => [detail?.summary ?? null, ppo?.summary ?? null, random?.summary ?? null],
    [detail, ppo, random],
  );
  const keeper = summaries[0] ?? null;

  const frontier = useMemo(() => {
    let min = Infinity;
    let max = 1;
    const traces = POLICIES.flatMap((policy, i) => {
      const f = summaries[i]?.aggregate.frontier;
      if (f === undefined || f.cumFlopsPerLane.length === 0) return [];
      for (const x of f.cumFlopsPerLane) {
        if (x > 0 && x < min) min = x;
        if (x > max) max = x;
      }
      return [{ key: policy.key, token: policy.token, line: seriesRows(f.cumFlopsPerLane, f.zIQM) }];
    });
    return Number.isFinite(min) && traces.length > 0 ? { min, max, traces } : null;
  }, [summaries]);

  const ratios = keeper?.vsBaselines?.ppo?.computeRatioAt;
  const ratioEntries = ratios === undefined ? [] : Object.entries(ratios).sort(([a], [b]) => Number(a) - Number(b));

  if (keeper === null) {
    return <div className="px-2 py-6 text-center text-2xs text-ink-faint">{STR.standings.needsSummary}</div>;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto px-2">
      <div>
        <div className="text-2xs uppercase tracking-[0.16em] text-ink-faint">{STR.compute.efficiency}</div>
        {POLICIES.map((policy, i) => (
          <div key={policy.key} className="mt-1 flex items-baseline gap-2">
            <Chip swatch={policy.token} label={STR.policy[policy.key]} />
            <span className="ml-auto font-mono text-xs tabular-nums text-ink">
              {summaries[i] === null ? '—' : summaries[i]?.compute.omega.toExponential(2)}
            </span>
          </div>
        ))}
      </div>
      <div>
        <div className="text-2xs uppercase tracking-[0.16em] text-ink-faint">{STR.compute.perStep}</div>
        {POLICIES.map((policy, i) => (
          <div key={policy.key} className="mt-1 flex items-baseline gap-2">
            <Chip swatch={policy.token} label={STR.policy[policy.key]} />
            <span className="ml-auto font-mono text-xs tabular-nums text-ink">
              {exp(summaries[i]?.compute.flopsPerStep)}
            </span>
          </div>
        ))}
      </div>
      {ratioEntries.length > 0 && (
        <div>
          <div className="text-2xs uppercase tracking-[0.16em] text-ink-faint">{STR.compute.ratios}</div>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {ratioEntries.map(([threshold, ratio]) => {
              const n = ratio === null || ratio === 0 ? null : 1 / ratio;
              const share = `${Math.round(Number(threshold) * 100)}%`;
              return (
                <span
                  key={threshold}
                  className="rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-2xs text-ink-dim"
                >
                  {n === null
                    ? STR.compute.notReached(share)
                    : STR.compute.ratioChip(share, n >= 10 ? n.toFixed(0) : n.toFixed(1))}
                </span>
              );
            })}
          </div>
        </div>
      )}
      {frontier !== null && (
        <div>
          <div className="flex items-center gap-2.5">
            <span className="text-2xs uppercase tracking-[0.16em] text-ink-faint">{STR.compute.frontier}</span>
            {frontier.traces.map((trace) => (
              <Chip key={trace.key} swatch={trace.token} label={STR.policy[trace.key as 'oak' | 'ppo' | 'random']} />
            ))}
          </div>
          <TimeChart
            height={170}
            xScale="log"
            xMin={frontier.min}
            xMax={frontier.max}
            xFormat={exp}
            yFormat={(v) => v.toFixed(2)}
          >
            {frontier.traces.map((trace) => (
              <Line key={trace.key} {...lineProps(trace.token, 1.4)} data={trace.line} connectNulls />
            ))}
          </TimeChart>
        </div>
      )}
    </div>
  );
}
