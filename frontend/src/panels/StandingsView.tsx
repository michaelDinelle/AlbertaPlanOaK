import { useMemo } from 'react';
import { clsx } from 'clsx';
import { Area, Line, ReferenceLine } from 'recharts';
import { ENVS, isEnvId } from '@shared/envs';
import type { RunDetail, RunSummary } from '@shared/types';
import { formatPercent, formatRate } from '../lib/format';
import { selectEnv, useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { Chip, TimeChart, bandProps, bandRows, lineProps, seriesRows } from './charts';

const POLICIES = [
  { key: 'oak', token: 'var(--color-oak)', top: '30%' },
  { key: 'ppo', token: 'var(--color-ppo)', top: '50%' },
  { key: 'random', token: 'var(--color-random)', top: '70%' },
] as const;

interface Mark {
  token: string;
  name: string;
  z: number;
  lo: number;
  hi: number;
  top: string;
}

export function StandingsView({
  detail,
  ppo,
  random,
}: {
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
}) {
  const sel = useSelection();
  const summaries = useMemo(
    () => [detail?.summary ?? null, ppo?.summary ?? null, random?.summary ?? null],
    [detail, ppo, random],
  );
  const keeper = summaries[0] ?? null;

  const { rows, lo, hi, wins, total } = useMemo(() => {
    if (keeper === null) return { rows: [], lo: 0, hi: 1, wins: 0, total: 0 };
    let lo = 0;
    let hi = 1;
    let wins = 0;
    let total = 0;
    const rows = Object.keys(keeper.envs).map((envId) => {
      const marks: Mark[] = [];
      POLICIES.forEach((policy, i) => {
        const env = summaries[i]?.envs[envId];
        if (env === undefined || !Number.isFinite(env.finalRate.iqmZ)) return;
        const { iqmZ, ci95Z } = env.finalRate;
        lo = Math.min(lo, ci95Z[0], iqmZ);
        hi = Math.max(hi, ci95Z[1], iqmZ);
        marks.push({ token: policy.token, name: STR.policy[policy.key], z: iqmZ, lo: ci95Z[0], hi: ci95Z[1], top: policy.top });
      });
      const oakZ = keeper.envs[envId]?.finalRate.iqmZ ?? null;
      const ppoZ = summaries[1]?.envs[envId]?.finalRate.iqmZ;
      if (oakZ !== null && ppoZ !== undefined && Number.isFinite(oakZ) && Number.isFinite(ppoZ)) {
        total++;
        if (oakZ > ppoZ) wins++;
      }
      return { envId, name: isEnvId(envId) ? ENVS[envId].name : envId, oakZ, marks };
    });
    rows.sort((a, b) => (b.oakZ ?? -Infinity) - (a.oakZ ?? -Infinity));
    const pad = (hi - lo) * 0.04;
    return { rows, lo: lo - pad, hi: hi + pad, wins, total };
  }, [keeper, summaries]);

  const profile = useMemo(() => {
    const base = summaries.find((s): s is RunSummary => s !== null);
    if (base === undefined) return null;
    const tau = base.aggregate.profile.tau;
    const traces = POLICIES.flatMap((policy, i) => {
      const p = summaries[i]?.aggregate.profile;
      if (p === undefined || p.tau.length !== tau.length) return [];
      return [{
        key: policy.key,
        token: policy.token,
        band: bandRows(p.tau, p.ci95[0], p.ci95[1]),
        line: seriesRows(p.tau, p.fraction),
      }];
    });
    return { min: Math.min(...tau, 0), max: Math.max(...tau, 1), traces };
  }, [summaries]);

  const span = hi - lo || 1;
  const pos = (z: number): number => ((z - lo) / span) * 100;
  const probs = keeper?.vsBaselines?.ppo?.probImprovement.envs;

  if (keeper === null) {
    return <div className="px-2 py-6 text-center text-2xs text-ink-faint">{STR.standings.needsSummary}</div>;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
      <div className="px-2 text-sm text-ink">{STR.standings.headline(wins, total)}</div>
      <div className="flex items-center gap-2.5 px-2">
        <span className="text-2xs uppercase tracking-[0.16em] text-ink-faint">{STR.standings.profile}</span>
        {POLICIES.map((p) => (
          <Chip key={p.key} swatch={p.token} label={STR.policy[p.key]} />
        ))}
      </div>
      {profile !== null && (
        <TimeChart
          height={150}
          xMin={profile.min}
          xMax={profile.max}
          xFormat={(v) => v.toFixed(1)}
          yFormat={(v) => formatPercent(v, 0)}
        >
          <ReferenceLine
            x={0}
            yAxisId="y"
            stroke="var(--color-ink-faint)"
            strokeDasharray="3 3"
            label={{ value: STR.standings.floorGuide, position: 'insideTopLeft', fill: 'var(--color-ink-faint)', fontSize: 9 }}
          />
          <ReferenceLine
            x={1}
            yAxisId="y"
            stroke="var(--color-ink-faint)"
            strokeDasharray="3 3"
            label={{ value: STR.standings.ceilingGuide, position: 'insideTopRight', fill: 'var(--color-ink-faint)', fontSize: 9 }}
          />
          {profile.traces.map((trace) => (
            <Area key={`${trace.key}-band`} {...bandProps(trace.token)} data={trace.band} />
          ))}
          {profile.traces.map((trace) => (
            <Line key={trace.key} {...lineProps(trace.token, 1.5)} data={trace.line} />
          ))}
        </TimeChart>
      )}
      <div className="px-2 text-2xs uppercase tracking-[0.16em] text-ink-faint">{STR.standings.perEnv}</div>
      <div className="flex flex-col gap-px">
        {rows.map((row) => {
          const on = sel.envId === row.envId;
          const p = probs?.[row.envId]?.p;
          return (
            <button
              key={row.envId}
              type="button"
              onClick={() => {
                if (isEnvId(row.envId)) selectEnv(row.envId);
              }}
              title={STR.standings.rowHint(row.name)}
              className={clsx(
                'flex h-7 items-stretch rounded-md text-left transition-colors',
                on ? 'bg-accent/10 ring-1 ring-inset ring-accent/35' : 'hover:bg-white/5',
              )}
            >
              <span className={clsx('flex w-26 shrink-0 items-center truncate pl-2 font-mono text-2xs', on ? 'text-accent' : 'text-ink-dim')}>
                {row.name}
              </span>
              <span className="relative min-w-0 flex-1">
                <span className="absolute inset-y-0 w-px bg-ink-faint/30" style={{ left: `${pos(0)}%` }} />
                <span className="absolute inset-y-0 w-px bg-ink-faint/30" style={{ left: `${pos(1)}%` }} />
                {row.marks.map((mark) => (
                  <span key={mark.name}>
                    <span
                      className="absolute h-[1.5px] -translate-y-1/2 rounded-full opacity-55"
                      style={{
                        top: mark.top,
                        left: `${Math.min(pos(mark.lo), pos(mark.hi))}%`,
                        width: `${Math.abs(pos(mark.hi) - pos(mark.lo))}%`,
                        background: mark.token,
                      }}
                    />
                    <span
                      className="absolute size-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
                      style={{ top: mark.top, left: `${pos(mark.z)}%`, background: mark.token }}
                      title={`${mark.name} ${mark.z.toFixed(2)}`}
                    />
                  </span>
                ))}
              </span>
              <span
                className="flex w-12 shrink-0 items-center justify-end pr-2 font-mono text-2xs tabular-nums text-ink-faint"
                title={p === undefined ? undefined : STR.standings.chance(formatPercent(p, 0))}
              >
                {p === undefined ? '' : formatPercent(p, 0)}
              </span>
            </button>
          );
        })}
      </div>
      <div className="flex items-center gap-2.5 px-2 pb-1 text-2xs text-ink-faint">
        <span>{STR.standings.spread}</span>
        {POLICIES.map((policy, i) => (
          <Chip key={policy.key} swatch={policy.token} label={formatRate(summaries[i]?.aggregate.iprZ90 ?? null)} />
        ))}
      </div>
    </div>
  );
}
