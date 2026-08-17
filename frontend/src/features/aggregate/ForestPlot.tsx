import { useMemo } from 'react';
import { clsx } from 'clsx';
import { ENVS, isEnvId } from '@shared/envs';
import type { RunDetail } from '@shared/types';
import { Card, CardHead, ChartNote, LegendChip } from '../../components/ui/card';
import { selectEnv, useSelection } from '../../state/selectionStore';

const POLICIES = [
  { key: 'oak' as const, name: 'OaK', token: 'var(--color-oak)', top: '28%' },
  { key: 'ppo' as const, name: 'PPO', token: 'var(--color-ppo)', top: '50%' },
  { key: 'random' as const, name: 'random', token: 'var(--color-random)', top: '72%' },
];

interface Mark {
  token: string;
  name: string;
  z: number;
  lo: number;
  hi: number;
  top: string;
}

interface ForestRow {
  envId: string;
  name: string;
  keeperZ: number | null;
  marks: Mark[];
}

export function ForestPlot({
  detail,
  ppo,
  random,
  className,
}: {
  detail: RunDetail | null;
  ppo: RunDetail | null;
  random: RunDetail | null;
  className?: string;
}) {
  const sel = useSelection();
  const summaries = useMemo(
    () => [detail?.summary ?? null, ppo?.summary ?? null, random?.summary ?? null],
    [detail, ppo, random],
  );

  const { rows, lo, hi } = useMemo(() => {
    const keeper = summaries[0] ?? null;
    if (keeper === null) return { rows: [] as ForestRow[], lo: 0, hi: 1 };
    let lo = 0;
    let hi = 1;
    const rows: ForestRow[] = Object.keys(keeper.envs).map((envId) => {
      const marks: Mark[] = [];
      POLICIES.forEach((policy, i) => {
        const env = summaries[i]?.envs[envId];
        if (env === undefined) return;
        const { iqmZ, ci95Z } = env.finalRate;
        if (!Number.isFinite(iqmZ)) return;
        lo = Math.min(lo, ci95Z[0], iqmZ);
        hi = Math.max(hi, ci95Z[1], iqmZ);
        marks.push({ token: policy.token, name: policy.name, z: iqmZ, lo: ci95Z[0], hi: ci95Z[1], top: policy.top });
      });
      return {
        envId,
        name: isEnvId(envId) ? ENVS[envId].name : envId,
        keeperZ: keeper.envs[envId]?.finalRate.iqmZ ?? null,
        marks,
      };
    });
    rows.sort((a, b) => (b.keeperZ ?? -Infinity) - (a.keeperZ ?? -Infinity));
    const pad = (hi - lo) * 0.04;
    return { rows, lo: lo - pad, hi: hi + pad };
  }, [summaries]);

  const span = hi - lo || 1;
  const pos = (z: number): number => ((z - lo) / span) * 100;

  const jump = (envId: string): void => {
    if (isEnvId(envId)) selectEnv(envId);
    document.getElementById('env-analytics')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <Card className={className}>
      <CardHead title={<span title="Per-environment final-rate IQM with 95% CI, on each env's z scale — sorted by OaK">Final rate by environment (z)</span>}>
        {POLICIES.map((p) => (
          <LegendChip key={p.key} swatch={p.token} label={p.name} />
        ))}
      </CardHead>
      {rows.length === 0 && <ChartNote>statistics need summary v2 on this run</ChartNote>}
      {rows.length > 0 && (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex items-stretch pb-1">
            <span className="w-32 shrink-0" />
            <span className="relative h-4 min-w-0 flex-1">
              <span
                className="absolute -translate-x-1/2 font-mono text-2xs text-ink-faint"
                style={{ left: `${pos(0)}%` }}
              >
                0 · random
              </span>
              <span
                className="absolute -translate-x-1/2 font-mono text-2xs text-ink-faint"
                style={{ left: `${pos(1)}%` }}
              >
                1 · ceiling
              </span>
            </span>
            <span className="w-12 shrink-0 text-right font-mono text-2xs text-ink-faint">OaK</span>
          </div>
          <div className="flex min-h-0 flex-1 flex-col justify-between gap-px">
            {rows.map((row) => {
              const on = sel.envId === row.envId;
              return (
                <button
                  key={row.envId}
                  type="button"
                  onClick={() => jump(row.envId)}
                  title={`Jump to ${row.name}`}
                  className={clsx(
                    'flex min-h-7 flex-1 items-stretch rounded-sm px-0 text-left transition-colors',
                    on ? 'bg-accent/10 ring-1 ring-inset ring-accent/35' : 'hover:bg-hover/70',
                  )}
                >
                  <span
                    className={clsx(
                      'flex w-32 shrink-0 items-center truncate pl-2 font-mono text-2xs',
                      on ? 'text-accent' : 'text-ink-dim',
                    )}
                  >
                    {row.name}
                  </span>
                  <span className="relative min-w-0 flex-1">
                    <span className="absolute inset-y-0 w-px bg-ink-faint/30" style={{ left: `${pos(0)}%` }} />
                    <span className="absolute inset-y-0 w-px bg-ink-faint/30" style={{ left: `${pos(1)}%` }} />
                    {row.marks.map((mark) => (
                      <span key={mark.name}>
                        <span
                          className="absolute h-[1.5px] -translate-y-1/2 rounded-full"
                          style={{
                            top: mark.top,
                            left: `${Math.min(pos(mark.lo), pos(mark.hi))}%`,
                            width: `${Math.abs(pos(mark.hi) - pos(mark.lo))}%`,
                            background: mark.token,
                            opacity: 0.55,
                          }}
                        />
                        <span
                          className="absolute size-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
                          style={{ top: mark.top, left: `${pos(mark.z)}%`, background: mark.token }}
                          title={`${mark.name} ${mark.z.toFixed(2)} [${mark.lo.toFixed(2)}, ${mark.hi.toFixed(2)}]`}
                        />
                      </span>
                    ))}
                  </span>
                  <span
                    className={clsx(
                      'flex w-12 shrink-0 items-center justify-end pr-2 font-mono text-2xs tabular-nums',
                      on ? 'text-ink' : 'text-ink-faint',
                    )}
                  >
                    {row.keeperZ === null ? '—' : row.keeperZ.toFixed(2)}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </Card>
  );
}
