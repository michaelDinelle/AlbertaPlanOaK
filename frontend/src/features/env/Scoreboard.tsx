import { clsx } from 'clsx';
import type { EnvSummary, Policy, RunDetail } from '@shared/types';
import { Tooltip, TooltipContent, TooltipTrigger } from '../../components/ui/tooltip';
import { formatRate } from '../../lib/format';
import { useSelection } from '../../state/selectionStore';

const CEILING_WORDS: Record<string, string> = {
  'closed-form': 'closed-form optimum',
  enumerated: 'enumerated optimum',
  'ppo-iqm': 'crowned-PPO IQM',
};

function formatZ(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return value.toFixed(2);
}

interface ScoreRow {
  policy: Policy;
  name: string;
  token: string;
  env: EnvSummary | null;
}

function envOf(detail: RunDetail | null, envId: string | null): EnvSummary | null {
  if (detail === null || envId === null) return null;
  return detail.summary?.envs[envId] ?? null;
}

export function Scoreboard({
  detail,
  ppo,
  random,
}: {
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
}) {
  const sel = useSelection();
  const keeperEnv = envOf(detail, sel.envId);
  const rows: ScoreRow[] = [
    { policy: 'oak', name: 'OaK', token: 'var(--color-oak)', env: keeperEnv },
    { policy: 'ppo', name: 'PPO', token: 'var(--color-ppo)', env: envOf(ppo, sel.envId) },
    { policy: 'random', name: 'random', token: 'var(--color-random)', env: envOf(random, sel.envId) },
  ];

  const zs = rows.map((row) => row.env?.finalRate.iqmZ ?? null);
  const lo = Math.min(0, ...zs.filter((z): z is number => z !== null && Number.isFinite(z)));
  const hi = Math.max(1, ...zs.filter((z): z is number => z !== null && Number.isFinite(z)));
  const span = hi - lo || 1;
  const pos = (z: number): number => ((z - lo) / span) * 100;

  const prob = sel.envId === null
    ? undefined
    : detail?.summary?.vsBaselines?.ppo?.probImprovement.envs[sel.envId];
  const ceilingWord = keeperEnv === null
    ? null
    : CEILING_WORDS[keeperEnv.ceilingSource] ?? keeperEnv.ceilingSource;

  return (
    <div className="veil rounded-md px-3 py-2.5">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-2xs uppercase tracking-[0.16em] text-ink-faint">Scoreboard</span>
        <span className="text-2xs text-ink-faint">final rate</span>
        <Tooltip>
          <TooltipTrigger className="ml-auto cursor-help rounded-xs border border-line px-1 font-mono text-2xs leading-4 text-ink-dim">
            z scale
          </TooltipTrigger>
          <TooltipContent side="left" className="max-w-56">
            Final-rate IQM on the per-env z scale: 0 = the random baseline's floor, 1 = the ceiling
            {ceilingWord === null ? '' : ` (here: ${ceilingWord}${keeperEnv === null ? '' : `, ${formatRate(keeperEnv.ceiling)} raw`})`}.
            Values are never clipped and may leave [0, 1].
          </TooltipContent>
        </Tooltip>
      </div>
      <div className="flex flex-col gap-1.5">
        {rows.map((row) => {
          const z = row.env?.finalRate.iqmZ ?? null;
          const raw = row.env?.finalRate.iqm ?? null;
          const active = sel.policy === row.policy;
          const finiteZ = z !== null && Number.isFinite(z);
          return (
            <div key={row.policy} className="flex items-center gap-2">
              <span
                className={clsx('size-2 rounded-full', active && 'ring-2 ring-offset-1 ring-offset-sunken')}
                style={{ background: row.token, ...(active ? { ['--tw-ring-color' as string]: row.token } : {}) }}
              />
              <span className={clsx('w-12 font-mono text-xs', active ? 'text-ink' : 'text-ink-dim')}>{row.name}</span>
              <span className="relative h-1 min-w-0 flex-1 rounded-full bg-line/50">
                <span className="absolute inset-y-0 w-px bg-ink-faint/60" style={{ left: `${pos(0)}%` }} />
                <span className="absolute inset-y-0 w-px bg-ink-faint/60" style={{ left: `${pos(1)}%` }} />
                {finiteZ && (
                  <span
                    className="absolute inset-y-0 rounded-full"
                    style={{
                      left: `${Math.min(pos(0), pos(z))}%`,
                      width: `${Math.abs(pos(z) - pos(0))}%`,
                      background: row.token,
                      opacity: active ? 0.9 : 0.5,
                    }}
                  />
                )}
              </span>
              <span
                className={clsx('w-10 text-right font-mono text-xs tabular-nums', active ? 'text-ink' : 'text-ink-dim')}
                title="final-rate IQM, z scale"
              >
                {formatZ(z)}
              </span>
              <span className="w-14 text-right font-mono text-2xs tabular-nums text-ink-faint" title="final-rate IQM, raw reward per step">
                {formatRate(raw)}
              </span>
            </div>
          );
        })}
      </div>
      {prob !== undefined && (
        <div
          className={clsx(
            'mt-2 flex items-baseline gap-2 rounded-sm border px-2 py-1',
            prob.p > 0.5 && 'border-good/35 bg-good/10',
            prob.p < 0.5 && 'border-bad/35 bg-bad/10',
            prob.p === 0.5 && 'border-line bg-sunken/60',
          )}
          title="Probability a random OaK seed beats a random PPO seed on final rate (Mann-Whitney), with 95% CI"
        >
          <span className="font-mono text-2xs text-ink-dim">P(OaK &gt; PPO)</span>
          <span className={clsx('font-mono text-xs tabular-nums', prob.p > 0.5 ? 'text-good' : prob.p < 0.5 ? 'text-bad' : 'text-ink')}>
            {prob.p.toFixed(2)}
          </span>
          <span className="ml-auto font-mono text-2xs tabular-nums text-ink-faint">
            [{prob.ci95[0].toFixed(2)}, {prob.ci95[1].toFixed(2)}]
          </span>
        </div>
      )}
    </div>
  );
}
