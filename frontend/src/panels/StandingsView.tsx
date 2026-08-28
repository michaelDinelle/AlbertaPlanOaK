import { useMemo } from 'react';
import { clsx } from 'clsx';
import { isEnvId } from '@shared/envs';
import { nameOf } from '../scenes/table';
import type { Ci95, Policy, RunDetail } from '@shared/types';
import { formatPercent, formatRate } from '../lib/format';
import { selectEnv, useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { POLICY_ORDER, POLICY_TOKENS, TXT } from './charts';

interface Mark {
  policy: Policy;
  value: number;
  ci: Ci95;
}

export function StandingsView({ runs }: { runs: Record<Policy, RunDetail | null> }) {
  const selection = useSelection();
  const keeper = runs.oak?.summary ?? null;
  const { rows, oakAhead, ppoAhead, close } = useMemo(() => {
    if (keeper === null) return { rows: [], oakAhead: 0, ppoAhead: 0, close: 0 };
    let oakAhead = 0;
    let ppoAhead = 0;
    let close = 0;
    const rows = Object.keys(keeper.envs).map((envId) => {
      const marks = POLICY_ORDER.flatMap((policy): Mark[] => {
        const final = runs[policy]?.summary?.envs[envId]?.finalRate;
        return final === undefined || !Number.isFinite(final.iqm)
          ? []
          : [{ policy, value: final.iqm, ci: final.ci95 }];
      });
      const oak = marks.find((mark) => mark.policy === 'oak');
      const ppo = marks.find((mark) => mark.policy === 'ppo');
      if (oak !== undefined && ppo !== undefined) {
        if (oak.ci[0] > ppo.ci[1]) oakAhead++;
        else if (ppo.ci[0] > oak.ci[1]) ppoAhead++;
        else close++;
      }
      return {
        envId,
        name: isEnvId(envId) ? nameOf(envId) : envId,
        marks,
        oak: oak?.value ?? -Infinity,
        probability: keeper.vsBaselines?.ppo?.probImprovement.envs[envId],
      };
    });
    rows.sort((a, b) => b.oak - a.oak);
    return { rows, oakAhead, ppoAhead, close };
  }, [keeper, runs]);

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
      <table className="w-full table-fixed border-separate border-spacing-0 px-1">
        <thead>
          <tr>
            <th className="w-24 pb-1 pl-2 text-left font-normal">{STR.standings.colEnv}</th>
            {POLICY_ORDER.map((policy) => (
              <th key={policy} className="pb-1 text-right font-normal" style={{ color: POLICY_TOKENS[policy] }}>
                {STR.policy[policy]}
              </th>
            ))}
            <th className="w-16 pb-1 pr-2 text-right font-normal" title={STR.standings.probHint}>
              {STR.standings.probability}
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
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
              className={clsx(
                'h-8 cursor-pointer transition-colors',
                selection.envId === row.envId ? 'bg-accent/10' : 'hover:bg-white/5',
              )}
            >
              <td className="truncate rounded-l-md pl-2 font-mono text-xs text-ink-dim">{row.name}</td>
              {POLICY_ORDER.map((policy) => {
                const mark = row.marks.find((candidate) => candidate.policy === policy);
                return (
                  <td
                    key={policy}
                    className={clsx('text-right', TXT.data)}
                    title={mark === undefined ? undefined : STR.standings.rateCell(
                      STR.policy[policy], formatRate(mark.ci[0]), formatRate(mark.ci[1]))}
                  >
                    {formatRate(mark?.value ?? null)}
                  </td>
                );
              })}
              <td
                className={clsx('rounded-r-md pr-2 text-right', TXT.data)}
                title={row.probability === undefined ? undefined : STR.standings.probCell(
                  formatPercent(row.probability.p, 0),
                  formatPercent(row.probability.ci95[0], 0),
                  formatPercent(row.probability.ci95[1], 0),
                )}
              >
                {row.probability === undefined ? '—' : formatPercent(row.probability.p, 0)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
