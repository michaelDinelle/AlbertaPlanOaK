import { useMemo } from 'react';
import { baselineId } from '../app';
import type { Policy } from '../data/layout';
import { deliveredRate, envLanes, runEnvs } from '../derive';
import { formatPercent, formatRate } from '../format';
import { recordedFamilies, configurationName } from '../data/envs';
import { selectEnv, store } from '../store';
import { POLICY_NAMES, POLICY_ORDER, POLICY_TOKENS } from './charts';

const WINS_HINT = 'mean reward per step over all seeds · wins paired per seed vs PPO, ties count half';

export function StandingsView() {
  const { runId, recordings, envId, catalog } = store.use();
  const oak = runId === null ? null : recordings[runId] ?? null;
  const rows = useMemo(() => {
    if (oak === null || !catalog.length) return [];
    const families = recordedFamilies(catalog, runEnvs(oak.meta));
    const names = new Map(families.flatMap(family => family.configurations.map(env => [env.envId, configurationName(env, family)])));
    const perEnv = (policy: Policy, env: string): number[] | null => {
      const id = baselineId(oak, policy, env);
      const rec = id === null ? undefined : recordings[id];
      return rec === undefined ? null : envLanes(rec.meta, env).lanes.map((lane) => deliveredRate(rec, lane) ?? NaN);
    };
    return runEnvs(oak.meta)
      .map((env) => {
        const means = Object.fromEntries(POLICY_ORDER.map((policy) => {
          const values = perEnv(policy, env);
          return [policy, values === null ? null : values.reduce((sum, v) => sum + v, 0) / values.length];
        })) as Record<Policy, number | null>;
        const a = perEnv('oak', env);
        const b = perEnv('ppo', env);
        const wins = a === null || b === null ? null : a.reduce((sum, v, i) => sum + (v === b[i] ? 0.5 : v > (b[i] ?? NaN) ? 1 : 0), 0) / a.length;
        return { env, name: names.get(env)!, means, wins };
      })
      .sort((x, y) => (y.means.oak ?? -Infinity) - (x.means.oak ?? -Infinity));
  }, [oak, recordings, catalog]);
  const ahead = rows.filter((row) => row.wins !== null && row.wins > 0.5).length;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
      <div className="px-2 text-xs" title={WINS_HINT}>OaK ahead in {ahead} of {rows.length}</div>
      <table className="w-full table-fixed border-separate border-spacing-0 px-1">
        <thead>
          <tr>
            <th className="w-24 pb-1 pl-2 text-left font-normal">Configuration</th>
            {POLICY_ORDER.map((policy) => <th key={policy} className="pb-1 text-right font-normal" style={{ color: POLICY_TOKENS[policy] }}>{POLICY_NAMES[policy]}</th>)}
            <th className="w-16 pb-1 pr-2 text-right font-normal" title={WINS_HINT}>OaK wins</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.env}
              tabIndex={0}
              onClick={() => selectEnv(row.env)}
              onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectEnv(row.env); } }}
              className={`h-8 cursor-pointer ${envId === row.env ? 'bg-accent/10' : 'hover:bg-white/5'}`}
            >
              <td title={row.name} className="truncate rounded-l-md pl-2 font-mono text-xs text-ink-dim">{row.name}</td>
              {POLICY_ORDER.map((policy) => <td key={policy} className="text-right font-mono text-xs tabular-nums">{formatRate(row.means[policy])}</td>)}
              <td className="rounded-r-md pr-2 text-right font-mono text-xs tabular-nums">{row.wins === null ? '—' : formatPercent(row.wins, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
