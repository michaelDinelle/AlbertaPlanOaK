import { useMemo } from 'react';
import { clsx } from 'clsx';
import { isEnvId } from '../data/envs';
import { nameOf } from '../scenes/table';
import type { Policy, RunDetail } from '../data/types';
import { formatPercent, formatRate } from '../lib/format';
import { envLanes } from '../lib/run';
import { selectEnv, useSelection } from '../state/selectionStore';
import { STR } from '../strings';
import { useTapeSnapshot, type DeriveSpec } from '../tape/store';
import { POLICY_ORDER, POLICY_TOKENS, TXT } from './charts';

const allLanes = (run: RunDetail | null): (DeriveSpec & { mode: 'delivered' }) | null =>
  run === null ? null : { mode: 'delivered', lanes: run.meta.seeds.map((_, lane) => lane) };

export function StandingsView({ runs }: { runs: Record<Policy, RunDetail | null> }) {
  const selection = useSelection();
  const oakSpec = useMemo(() => allLanes(runs.oak), [runs.oak]);
  const ppoSpec = useMemo(() => allLanes(runs.ppo), [runs.ppo]);
  const randomSpec = useMemo(() => allLanes(runs.random), [runs.random]);
  const delivered: Record<Policy, (number | null)[] | null> = {
    oak: useTapeSnapshot(runs.oak?.id ?? null, null, 'delivered', oakSpec),
    ppo: useTapeSnapshot(runs.ppo?.id ?? null, null, 'delivered', ppoSpec),
    random: useTapeSnapshot(runs.random?.id ?? null, null, 'delivered', randomSpec),
  };
  const rows = useMemo(() => {
    const oak = runs.oak;
    if (oak === null) return [];
    const perEnv = (policy: Policy, envId: string): number[] | null => {
      const run = runs[policy];
      const values = delivered[policy];
      return run === null || values === null ? null : envLanes(run.meta, envId).lanes.map((lane) => values[lane] ?? NaN);
    };
    const rows = [...new Set(oak.meta.envSchedule.map((entry) => entry.envId))].map((envId) => {
      const means = Object.fromEntries(POLICY_ORDER.map((policy) => {
        const values = perEnv(policy, envId);
        return [policy, values === null ? null : values.reduce((sum, v) => sum + v, 0) / values.length];
      })) as Record<Policy, number | null>;
      const a = perEnv('oak', envId);
      const b = perEnv('ppo', envId);
      const wins = a === null || b === null ? null
        : a.reduce((sum, v, i) => sum + (v === b[i] ? 0.5 : v > (b[i] ?? NaN) ? 1 : 0), 0) / a.length;
      return { envId, name: isEnvId(envId) ? nameOf(envId) : envId, means, wins };
    });
    return rows.sort((x, y) => (y.means.oak ?? -Infinity) - (x.means.oak ?? -Infinity));
  }, [runs, delivered.oak, delivered.ppo, delivered.random]);
  const ahead = rows.filter((row) => row.wins !== null && row.wins > 0.5).length;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
      <div className="px-2">
        <span className={clsx(TXT.body, 'cursor-help')} title={STR.standings.winsHint}>
          {STR.standings.headline(ahead, rows.length)}
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
            <th className="w-16 pb-1 pr-2 text-right font-normal" title={STR.standings.winsHint}>
              {STR.standings.wins}
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
              {POLICY_ORDER.map((policy) => (
                <td key={policy} className={clsx('text-right', TXT.data)}>
                  {formatRate(row.means[policy])}
                </td>
              ))}
              <td className={clsx('rounded-r-md pr-2 text-right', TXT.data)}>
                {row.wins === null ? '—' : formatPercent(row.wins, 0)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
