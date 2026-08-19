import { useMemo } from 'react';
import { clsx } from 'clsx';
import { Area, Line, ReferenceLine } from 'recharts';
import { isEnvId } from '@shared/envs';
import { nameOf } from '../scenes/table';
import type { Ci95, Policy, RunDetail, RunSummary } from '@shared/types';
import { formatPercent, formatRate } from '../lib/format';
import { selectEnv, useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { Chip, Label, POLICY_ORDER, POLICY_TOKENS, TXT, TimeChart, bandProps, bandRows, lineProps, seriesRows } from './charts';

const POLICIES = POLICY_ORDER.map((key, i) => ({ key, token: POLICY_TOKENS[key], top: `${30 + i * 20}%` }));

const RAIL_LO = -0.1;
const RAIL_HI = 1.5;

function ProbLabel({ suffix }: { suffix?: string }) {
  return (
    <>
      P(<span style={{ color: POLICY_TOKENS.oak }}>{STR.policy.oak}</span>
      {'>'}
      <span style={{ color: POLICY_TOKENS.ppo }}>{STR.policy.ppo}</span>)
      {suffix !== undefined && ` ${suffix}`}
    </>
  );
}

function railPos(z: number): number {
  return ((Math.min(RAIL_HI, Math.max(RAIL_LO, z)) - RAIL_LO) / (RAIL_HI - RAIL_LO)) * 100;
}

interface Mark {
  policy: Policy;
  token: string;
  name: string;
  z: number;
  ci: Ci95;
  top: string;
}

function Caret({ mark, at, side }: { mark: Mark; at: number; side: 'left-0' | 'right-0' }) {
  return (
    <span
      className={clsx('absolute -translate-y-1/2 font-mono text-[9px] leading-none', side)}
      style={{ top: mark.top, color: mark.token }}
      title={STR.standings.beyond(mark.name, formatRate(at, 3))}
    >
      {side === 'right-0' ? '›' : '‹'}
    </span>
  );
}

function RailMark({ mark }: { mark: Mark }) {
  const lo = Math.min(mark.z, mark.ci[0]);
  const hi = Math.max(mark.z, mark.ci[1]);
  return (
    <>
      <span
        className="absolute h-[1.5px] -translate-y-1/2 rounded-full opacity-55"
        style={{
          top: mark.top,
          left: `${railPos(mark.ci[0])}%`,
          width: `${railPos(mark.ci[1]) - railPos(mark.ci[0])}%`,
          background: mark.token,
        }}
      />
      {mark.z >= RAIL_LO && mark.z <= RAIL_HI && (
        <span
          className="absolute size-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{ top: mark.top, left: `${railPos(mark.z)}%`, background: mark.token }}
          title={STR.standings.zCell(mark.name, formatRate(mark.z, 3), formatRate(mark.ci[0], 3), formatRate(mark.ci[1], 3))}
        />
      )}
      {lo < RAIL_LO && <Caret mark={mark} at={lo} side="left-0" />}
      {hi > RAIL_HI && <Caret mark={mark} at={hi} side="right-0" />}
    </>
  );
}

export function StandingsView({ runs }: { runs: Record<Policy, RunDetail | null> }) {
  const sel = useSelection();
  const summaries = useMemo(() => POLICY_ORDER.map((policy) => runs[policy]?.summary ?? null), [runs]);
  const keeper = summaries[0] ?? null;

  const { rows, oakAhead, ppoAhead, close } = useMemo(() => {
    if (keeper === null) return { rows: [], oakAhead: 0, ppoAhead: 0, close: 0 };
    let oakAhead = 0;
    let ppoAhead = 0;
    let close = 0;
    const rows = Object.keys(keeper.envs).map((envId) => {
      const marks: Mark[] = [];
      POLICIES.forEach((policy, i) => {
        const env = summaries[i]?.envs[envId];
        if (env === undefined || !Number.isFinite(env.finalRate.iqmZ)) return;
        marks.push({
          policy: policy.key,
          token: policy.token,
          name: STR.policy[policy.key],
          z: env.finalRate.iqmZ,
          ci: env.finalRate.ci95Z,
          top: policy.top,
        });
      });
      const oak = marks.find((m) => m.policy === 'oak') ?? null;
      const ppoMark = marks.find((m) => m.policy === 'ppo') ?? null;
      if (oak !== null && ppoMark !== null) {
        if (oak.ci[0] > ppoMark.ci[1]) oakAhead++;
        else if (ppoMark.ci[0] > oak.ci[1]) ppoAhead++;
        else close++;
      }
      return { envId, name: isEnvId(envId) ? nameOf(envId) : envId, oakZ: oak?.z ?? null, marks };
    });
    rows.sort((a, b) => (b.oakZ ?? -Infinity) - (a.oakZ ?? -Infinity));
    return { rows, oakAhead, ppoAhead, close };
  }, [keeper, summaries]);

  const profile = useMemo(() => {
    const base = summaries.find((s): s is RunSummary => s !== null);
    if (base === undefined) return null;
    const tau = base.aggregate.profile.tau;
    const mismatched: string[] = [];
    const traces = POLICIES.flatMap((policy, i) => {
      const p = summaries[i]?.aggregate.profile;
      if (p === undefined) return [];
      if (p.tau.length !== tau.length || p.tau.some((t, j) => t !== tau[j])) {
        mismatched.push(STR.policy[policy.key]);
        return [];
      }
      return [{
        key: policy.key,
        token: policy.token,
        band: bandRows(p.tau, p.ci95[0], p.ci95[1]),
        line: seriesRows(p.tau, p.fraction),
      }];
    });
    return { min: Math.min(...tau, 0), max: Math.max(...tau, 1), traces, mismatched };
  }, [summaries]);

  const prob = keeper?.vsBaselines?.ppo?.probImprovement ?? null;

  if (keeper === null) {
    return <div className={clsx('px-2 py-6 text-center', TXT.bodyDim)}>{STR.standings.needsSummary}</div>;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
      <div className="px-2">
        <span className={clsx(TXT.body, 'cursor-help')} title={STR.standings.headlineHint}>
          {STR.standings.headline(oakAhead, ppoAhead, close)}
        </span>
      </div>
      <div className="flex items-center gap-2.5 px-2">
        {POLICIES.map((policy) => (
          <Chip key={policy.key} swatch={policy.token} label={STR.policy[policy.key]} />
        ))}
      </div>
      <table className="w-full table-fixed border-separate border-spacing-0 px-1">
        <thead>
          <tr>
            <th scope="col" className="w-22 truncate pb-1 pl-2 pr-2 text-left font-normal">
              <Label>{STR.standings.colEnv}</Label>
            </th>
            <th scope="col" className="pb-1 pl-1 text-left font-normal">
              <Label hint={STR.standings.railHint(String(RAIL_LO), String(RAIL_HI))}>{STR.standings.colRail}</Label>
            </th>
            <th scope="col" className="w-22 whitespace-nowrap pb-1 pl-1 pr-2 text-right font-normal">
              <Label hint={STR.standings.probHint}><ProbLabel /></Label>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const on = sel.envId === row.envId;
            const p = prob?.envs[row.envId];
            return (
              <tr
                key={row.envId}
                tabIndex={0}
                onClick={() => {
                  if (isEnvId(row.envId)) selectEnv(row.envId);
                }}
                onKeyDown={(event) => {
                  if ((event.key === 'Enter' || event.key === ' ') && isEnvId(row.envId)) {
                    event.preventDefault();
                    selectEnv(row.envId);
                  }
                }}
                title={STR.standings.rowHint(row.name)}
                className={clsx(
                  'h-8 cursor-pointer transition-colors',
                  on ? 'bg-accent/10' : 'hover:bg-white/5',
                )}
              >
                <td className={clsx('truncate rounded-l-md pl-2 font-mono text-xs', on ? 'text-accent' : 'text-ink-dim')}>
                  {row.name}
                </td>
                <td className="relative">
                  <span className="absolute inset-y-1 w-px bg-ink-faint/30" style={{ left: `${railPos(0)}%` }} />
                  <span className="absolute inset-y-1 w-px bg-ink-faint/30" style={{ left: `${railPos(1)}%` }} />
                  {row.marks.map((mark) => (
                    <RailMark key={mark.policy} mark={mark} />
                  ))}
                </td>
                <td
                  className={clsx('whitespace-nowrap rounded-r-md pr-2 text-right', TXT.data)}
                  title={
                    p === undefined
                      ? undefined
                      : STR.standings.probCell(
                          formatPercent(p.p, 0),
                          formatPercent(p.ci95[0], 0),
                          formatPercent(p.ci95[1], 0),
                        )
                  }
                >
                  {p === undefined ? '—' : formatPercent(p.p, 0)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {prob !== null && (
        <div className="flex items-baseline gap-2 px-2">
          <Label hint={STR.standings.probHint}><ProbLabel suffix={STR.standings.mean} /></Label>
          <span className={TXT.data}>{formatPercent(prob.mean, 1)}</span>
          <span className={TXT.dataDim}>
            {STR.standings.ci(formatPercent(prob.ci95[0], 1), formatPercent(prob.ci95[1], 1))}
          </span>
        </div>
      )}
      {profile !== null && (
        <>
          <div className="flex items-center gap-2.5 px-2 pt-1">
            <Label hint={STR.standings.profileHint}>{STR.standings.profile}</Label>
            {profile.traces.map((trace) => (
              <Chip key={trace.key} swatch={trace.token} label={STR.policy[trace.key]} />
            ))}
          </div>
          <TimeChart
            label={STR.standings.profile}
            height={150}
            xMin={profile.min}
            xMax={profile.max}
            xFormat={(v) => v.toFixed(1)}
            yFormat={(v) => formatPercent(v, 0)}
            xLabel={STR.axis.tau}
            yLabel={STR.axis.runShare}
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
              <Line key={trace.key} {...lineProps(trace.token, 1.5)} name={STR.policy[trace.key]} data={trace.line} />
            ))}
          </TimeChart>
          {profile.mismatched.length > 0 && (
            <div className="px-2 text-xs text-crit">{STR.standings.profileMismatch(profile.mismatched.join(', '))}</div>
          )}
        </>
      )}
      <div className={clsx('flex items-center gap-2.5 px-2 pb-1', TXT.bodyDim)}>
        <span>{STR.standings.spread}</span>
        {POLICIES.map((policy, i) => (
          <span key={policy.key} className="flex items-center gap-1 font-mono tabular-nums">
            <span style={{ color: policy.token }}>{STR.policy[policy.key]}</span>
            <span className="text-ink">{formatRate(summaries[i]?.aggregate.iprZ90 ?? null)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
